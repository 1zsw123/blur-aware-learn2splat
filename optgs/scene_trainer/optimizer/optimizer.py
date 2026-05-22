from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Generic, Optional, TYPE_CHECKING, Any
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import Tensor
import numpy as np
import os
from optgs.dataset.camera_datasets.camera import get_scene_scale
from optgs.misc.io import FrequencyScheduler
from optgs.dataset.data_types import BatchedViews
from optgs.model.decoder import Decoder
from optgs.model.decoder.decoder import DecoderOutput
from optgs.model.types import Gaussians
from optgs.scene_trainer.adc.base import BaseStrategyCfg
from optgs.scene_trainer.initializer.initializer import InitializerOutput
from optgs.scene_trainer.optimizer.layer import AdamState
from optgs.scene_trainer.initializer import InitializerCfg
from optgs.misc.detaching_cpu_list import DetachingCPUList
from optgs.scene_trainer.optimizer.lr_scheduler import LrSchedulerCfgType, get_scheduler

if TYPE_CHECKING:
    from optgs.scene_trainer.adc.vanilla import VanillaStrategyState
    from optgs.scene_trainer.adc.mcmc import McmcStrategyState


@dataclass
class OptimizerState:
    state: torch.Tensor | None = None
    init_state: torch.Tensor | None = None  # state at the beginning of the optimization
    adam_state: AdamState | None = None
    adc_state: Any = None  # VanillaStrategyState | McmcStrategyState | None


@dataclass
class OptimizerPreviousOutput:
    gaussians: Gaussians
    state: OptimizerState | None = None


@dataclass
class OptimizerInput:
    context: BatchedViews
    renderer: Decoder
    prev_output: InitializerOutput | OptimizerPreviousOutput
    num_refine: int
    iter_batch_size: int | None
    target: BatchedViews | None = None
    context_remain: dict | None = None
    debug_dict: dict | None = None
    additional_info: tuple | None = None

    @property
    def device(self) -> torch.device:
        return self.context["image"].device


@dataclass
class OptimizerOutput:
    # TODO Naama: should we add here iterations?
    gaussian_list: DetachingCPUList[Gaussians]
    t: int | None = None
    T: int | None = None
    last_prev_output: OptimizerPreviousOutput | None = None
    target_render_list: DetachingCPUList[DecoderOutput] | None = None
    context_render_list: DetachingCPUList[DecoderOutput] | None = None
    info: dict | None = None
    context_index_list: list[int] = field(default_factory=list)
    target_index_list: list[int] = field(default_factory=list)

    def get_render_list(self, which: str) -> DetachingCPUList[DecoderOutput] | None:
        if which == "target":
            return self.target_render_list
        elif which == "context":
            return self.context_render_list
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def get_index_list(self, which: str):
        if which == "target":
            return self.target_index_list
        elif which == "context":
            return self.context_index_list
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    @classmethod
    def empty(cls, t=None) -> "OptimizerOutput":
        new = cls(gaussian_list=DetachingCPUList(), t=t)
        new.target_render_list = DetachingCPUList()
        new.context_render_list = DetachingCPUList()
        # info is a dict of lists of dicts, should all be stored in cpu
        new.info: dict[str, list[dict[str, Tensor]]] = {}
        return new


@dataclass
class OptimizerCfg:
    
    # subset optimization flags
    no_refine_mean: bool
    no_refine_scale: bool
    no_refine_rotation: bool
    no_refine_opacity: bool
    no_refine_sh0: bool
    no_refine_shN: bool

    # lr scheduler
    lr_scheduler: LrSchedulerCfgType
    
    refiner: BaseStrategyCfg

    # gradients
    input_gradients_chunk_size: int | None  # if None, use full image

    # L1 opacity regularization from 3DGS-MCMC (arXiv:2404.09591); 0.0 to disable
    opacity_reg_lambda: float

    def update(self, initializer_cfg: InitializerCfg):
        pass

    @property
    def any_adc(self) -> bool:
        return self.refiner.do_densify or self.refiner.do_prune or self.refiner.do_opacity_reset

    @property
    def need_2d_grads(self) -> bool:
        return self.refiner.do_densify

    @property
    def optimize_all(self):
        # All the no_refine_* are False
        return not any([
            self.no_refine_mean,
            self.no_refine_scale,
            self.no_refine_rotation,
            self.no_refine_opacity,
            self.no_refine_sh0,
            self.no_refine_shN,
        ])


T = TypeVar("T")


class Optimizer(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T, save_every: Optional[FrequencyScheduler] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_every = save_every

        # for timing
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        # decoder_event_start/end bracket only the rendering-for-gradients call inside
        # apply_one_update_step, letting us split iter_time into decoder vs optimizer.
        self.decoder_event_start = torch.cuda.Event(enable_timing=True)
        self.decoder_event_end = torch.cuda.Event(enable_timing=True)
        # scene_start_event_start/end bracket optimizer.on_scene_start() (KNN, Adam init).
        # Read after the post-loop cuda.synchronize() in scene_trainer.get_optimized_gaussians.
        self.scene_start_event_start = torch.cuda.Event(enable_timing=True)
        self.scene_start_event_end = torch.cuda.Event(enable_timing=True)

        # Init logs for densification/pruning
        self.radii_max_log = []
        self.grads_max_log = []
        self.nr_cloned_log = []
        self.nr_splitted_log = []
        self.nr_pruned_log = []
        self.nr_gaussians_log = []
        self.iter_time_log = []       # total ms per iteration
        self.decoder_time_log = []    # ms spent in rendering-for-gradients per iteration
        self.optimizer_time_log = []  # ms spent in update step (iter_time - decoder_time)
        self.scene_start_ms = 0.0    # ms for on_scene_start (KNN lookup, Adam state init)
        self.nr_nonzero_grad_log = []

        # LR scheduler
        self.scheduler = get_scheduler(self.cfg.lr_scheduler)

    def forward(self, i, optimizer_input: OptimizerInput, optimizer_output: OptimizerOutput, **kwargs) -> OptimizerOutput:
        return self._forward_impl(i, optimizer_input, optimizer_output, **kwargs)

    def _record_iter_timing(self) -> None:
        """Record per-iteration timing into iter/decoder/optimizer_time_log.
        Call right after the timed region; iter_start must already be recorded."""
        self.iter_end.record()
        torch.cuda.synchronize()
        elapsed_time = self.iter_start.elapsed_time(self.iter_end)
        self.iter_time_log.append(elapsed_time)
        decoder_ms = self.decoder_event_start.elapsed_time(self.decoder_event_end)
        self.decoder_time_log.append(decoder_ms)
        self.optimizer_time_log.append(elapsed_time - decoder_ms)

    def on_scene_start(self, optimizer_input: OptimizerInput) -> None:
        self._on_scene_start_impl(optimizer_input)

    def _on_scene_start_impl(self, optimizer_input: OptimizerInput) -> None:
        init_output = optimizer_input.prev_output
        assert isinstance(init_output, InitializerOutput), \
            (f"base Optimizer class on_scene_start just convert the InitializerOutput to OptimizerPreviousOutput, "
             f"without handling the state. "
             f"It also initialize a new state for density control."
             f"Got type {type(init_output)}")

        # Converting the initializer output to optimizer previous output
        optimizer_prev_output = OptimizerPreviousOutput(
            gaussians=init_output.gaussians.clone(),
            state=None,
        )
        optimizer_input.prev_output = optimizer_prev_output

        if self.cfg.any_adc:
            self.reset_logs()
            optimizer_prev_output.state = OptimizerState()  # init to empty state
            self.initialize_adc_state(self.cfg, optimizer_input)

    def on_scene_end(self) -> None:
        pass

    def reset_logs(self):
        self.radii_max_log = []
        self.grads_max_log = []
        self.nr_cloned_log = []
        self.nr_splitted_log = []
        self.nr_pruned_log = []
        self.nr_gaussians_log = []
        self.iter_time_log = []
        self.decoder_time_log = []
        self.optimizer_time_log = []
        self.scene_start_ms = 0.0
        self.nr_nonzero_grad_log = []

    @staticmethod
    def initialize_adc_state(cfg: OptimizerCfg, optimizer_input: OptimizerInput) -> None:
        # Lazy import to avoid circular dependency
        from optgs.scene_trainer.adc import init_strategy_state

        # get number of points
        init_gaussians = optimizer_input.prev_output.gaussians
        nr_points = init_gaussians.means.shape[1]
        # get scene extent
        context = optimizer_input.context
        target = optimizer_input.target
        assert (
                context["extrinsics"].shape[0] == context["intrinsics"].shape[0] == 1
        ), "scene batch size > 1 not supported yet..."

        scene_scale = context["scene_scale"][0].item()
        # Initialize ADC state
        optimizer_input.prev_output.state.adc_state = init_strategy_state(
            cfg=cfg.refiner,
            nr_points=nr_points,
            device=init_gaussians.means.device,
            scene_extent=scene_scale
        )
        print("Initialized ADC state with", nr_points, "points and scene extent", scene_scale)
        
    def _forward_impl(self, i, optimizer_input: OptimizerInput, optimizer_output: OptimizerOutput, **kwargs) -> OptimizerOutput:
        raise NotImplementedError()

    def validate_input(self, optimizer_input: OptimizerInput) -> None:
        pass

    def _save_post_update_renders(
            self,
            i: int,
            optimizer_input: OptimizerInput,
            optimizer_output: OptimizerOutput,
            updated_gaussians: Gaussians,
            full_context: BatchedViews,
            full_target: BatchedViews,
    ) -> None:
        """Render and append post-update context+target views.

        Renders every iteration during training (so per-step renders can feed the meta-loss);
        otherwise renders only when save_every fires for the given tag. The per-iter subset
        (optimizer_input.context/target) is used in training when sampling indices exist,
        otherwise the full views.
        """
        for tag, full, iter_views in (
            ("context", full_context, optimizer_input.context),
            ("target", full_target, optimizer_input.target),
        ):
            if not (self.training or self.save_every(i + 1, tag=tag)):
                continue
            index_list = optimizer_output.get_index_list(tag)
            subset = iter_views if (index_list and self.training) else full
            render_output = optimizer_input.renderer.forward_batch_subset(
                updated_gaussians,
                subset,
                iter_batch_size=optimizer_input.iter_batch_size,
            )
            optimizer_output.get_render_list(tag).append(
                render_output,
                detach_and_cpu=not self.training,
            )

    @torch.no_grad()
    def apply_adc(self, i, v, h, w, adc_state, gaussians, meta, object_dict_to_adjust=None):
        """
        Apply adaptive density control (ADC) based on 2D gradient norms.
        Implements densification and pruning of Gaussians during optimization, as in vanilla 3DGS.

        Args:
            gaussians: Gaussians to be densified/pruned in place.
            h: Height of the rendered images.
            i: Current optimization iteration.
            v: Number of views.
            meta: Metadata dict from the rendering, including visibility masks and radii.
            w: Width of the rendered images.
            object_dict_to_adjust: Dict of object to adjust after pruning and densification, if needed.
        """
        # Lazy import to avoid circular dependency
        from optgs.scene_trainer.adc import post_backward

        visibility_mask = meta["visibility_filter"]  # [B, V, N]
        radii_2d = meta["radii"].float()  # [B, V, N, 2]
        means2d_grads = meta["means_2d_grads"]  # [B, V, N, 2] or None
        
        # means lr for MCMC noise injection
        # check if optimizer has means_lr_scheduler
        if hasattr(self, "means_lr_scheduler"):
            assert self.means_lr_scheduler is not None, "means_lr_scheduler is None."
            lr = self.means_lr_scheduler(i)
        else:
            # Use fallback_means_lr from the refiner config so noise magnitude matches the
            # original paper (means_lr * noise_lr ≈ 1.6e-4 * 5e5 = 80 covariance-units).
            lr = self.cfg.refiner.fallback_means_lr

        # Post-backward (ADC)
        nr_cloned, nr_splitted, nr_pruned, max_radii, max_grad2d = post_backward(
            cfg=self.cfg.refiner,
            step=i,
            gaussians=gaussians,
            adc_state=adc_state,
            smoothers=object_dict_to_adjust,
            radii_2d=radii_2d,  # [V, N]
            means2d_grads=means2d_grads,  # [V, N, 2]
            visibility_mask=visibility_mask,  # [V, N]
            iter_batch_size=v,
            w=w,
            h=h,
            lr=lr
        )
        
        self.nr_cloned_log.append(nr_cloned)
        self.nr_splitted_log.append(nr_splitted)
        self.nr_pruned_log.append(nr_pruned)
        if max_radii is not None:
            self.radii_max_log.append(max_radii)
        else:
            self.radii_max_log.append(0.0)
        if max_grad2d is not None:
            self.grads_max_log.append(max_grad2d)
        else:
            self.grads_max_log.append(0.0)

    def plot_info(self, step, output_path: Path | None = None, scene_name: str | None = None) -> None:

        if output_path is None:
            return

        if scene_name is None:
            return 
        
        save_path = output_path / "plots" / scene_name
        os.makedirs(save_path, exist_ok=True)
        
        # Define datasets and labels in a compact structure
        data = []
        
        if len(self.radii_max_log) == len(self.iter_time_log):
            data.append((range(len(self.iter_time_log)), self.radii_max_log, "Max Radius"))
        if len(self.grads_max_log) == len(self.iter_time_log):
            data.append((range(len(self.iter_time_log)), self.grads_max_log, "Max Grad magnitude"))
        if len(self.nr_cloned_log) == len(self.iter_time_log):
            data.append((range(len(self.iter_time_log)), self.nr_cloned_log, "Cloned"))
        if len(self.nr_splitted_log) == len(self.iter_time_log):
            data.append((range(len(self.iter_time_log)), self.nr_splitted_log, "Splitted"))
        if len(self.nr_pruned_log) == len(self.iter_time_log):
            data.append((range(len(self.iter_time_log)), self.nr_pruned_log, "Pruned"))

        data.append((range(len(self.iter_time_log)), self.nr_gaussians_log, "Total"))
        data.append((range(len(self.iter_time_log)), self.iter_time_log, "Iteration Time (ms)"))

        # Create a larger figure with shared x-axis
        nr_rows = len(data)
        fig, axes = plt.subplots(nr_rows, 1, figsize=(10, 15), sharex=True)

        # Define some styles for visual variety
        styles = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        assert nr_rows <= len(styles), "Not enough styles defined for the number of subplots."
        
        # Loop through subplots
        for ax, (x, y, label), color in zip(axes, data, styles):
            ax.plot(x, y, label=label, color=color, linewidth=2)
            ax.set_ylabel("Value", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right", fontsize=10)
            ax.set_title(f"{label} Gaussians", fontsize=13, pad=5)
            # show x-axis ticks on all plots
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            # set y-axis vmin to 0
            # ax.set_ylim(bottom=0)
        
        # Shared x-axis label
        axes[-1].set_xlabel("Iteration", fontsize=12)
        # Improve layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        #
        # module_name = self.__class__.__name__.lower()
        
        # Save and close
        save_path = save_path / f"stats_{step}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved optimizer stats plot to:", save_path)


class LearnedOptimizer(Optimizer[T], ABC):
    @property
    def strategy(self) -> str:
        return "learned"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class NonlearnedOptimizer(Optimizer[T], ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # nn.Module.__init__ sets training=True (a plain attribute, not via
        # train()); a non-learned optimizer has no trainable parameters, so pin
        # it to eval at construction.
        self.eval()

    @property
    def strategy(self) -> str:
        return "nonlearned"

    def train(self, mode: bool = True):
        # train mode is meaningless here, and `self.training` gates
        # meta-training-only code paths (e.g. _save_post_update_renders
        # retaining full-scene renders on GPU). Pin to eval, even under a
        # generic `module.train()` recursion.
        return super().train(False)

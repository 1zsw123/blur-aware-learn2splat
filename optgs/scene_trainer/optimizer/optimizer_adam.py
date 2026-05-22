from dataclasses import dataclass
from functools import partial
from typing import Literal, List, Optional

import torch
from torch import Tensor

from optgs.dataset.data_types import BatchedViews
from optgs.misc.general_utils import get_expon_lr_func
from optgs.misc.io import FrequencyScheduler
from optgs.model.decoder.decoder import Decoder
from optgs.model.types import Gaussians
from optgs.scene_trainer.initializer import InitializerCfg
from optgs.scene_trainer.optimizer.layer import AdamInputSmoothing
from optgs.scene_trainer.optimizer.optimizer import (
    OptimizerInput,
    OptimizerOutput,
    OptimizerCfg, NonlearnedOptimizer,
)
from optgs.scene_trainer.optimizer.optimizer_utils import (
    calc_input_gradients,
    squeeze_grad_dict,
    smooth_grads,
)


@dataclass
class AdamOptimizerCfg(OptimizerCfg):
    name: Literal["adam"]

    # adam params
    betas: List[float | int]  # Typically a list of two floats, e.g., [0.9, 0.999]
    eps: float
    weight_decay: float

    # learning rates
    base_lr: int | float
    means_lr_init: float
    means_lr_final: float
    means_lr_delay_mult: float
    means_lr_max_steps: int  # should be equal to total optimization steps
    scales_lr: float
    rotations_lr: float
    opacities_lr: float
    sh0s_lr: float
    shNs_lr: float  # 20 times less as sh0s_lr in original paper

    def update(self, initializer_cfg: InitializerCfg):
        pass


class AdamOptimizer(NonlearnedOptimizer[AdamOptimizerCfg]):
    def __init__(
            self, cfg: AdamOptimizerCfg, save_every: Optional[FrequencyScheduler] = None
    ) -> None:
        super().__init__(cfg, save_every)

        self.smoothers = None
        self.means_lr_scheduler = None
        self._meta_bufs: dict = {}  # reused across steps: radii, visibility buffers

        # NOTE: AdamOptimizer is evaluation-only (3DGS baseline); not used during meta-training.

    def _on_scene_start_impl(self, optimizer_input: OptimizerInput) -> None:
        super()._on_scene_start_impl(optimizer_input)

        # assert scene batch size 1
        context = optimizer_input.context
        assert (
                context["extrinsics"].shape[0] == context["intrinsics"].shape[0] == 1
        ), "scene batch size > 1 not supported yet..."

        # instantiate Adam optimizers for each parameter type
        nr_gaussians = optimizer_input.prev_output.gaussians.means.shape[1]
        device = optimizer_input.prev_output.gaussians.means.device
        smoother_cls = partial(AdamInputSmoothing, beta1=self.cfg.betas[0], beta2=self.cfg.betas[1], eps=self.cfg.eps,
                               device=device)
        means_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.means.shape[1:])
        scales_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.scales.shape[1:])
        rotations_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.rotations.shape[1:])
        opacities_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.opacities.shape[1:])
        sh0s_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.harmonics[..., :, :1].shape[1:])

        init_gaussians = optimizer_input.prev_output.gaussians
        if init_gaussians.harmonics.shape[-1] > 1:
            shNs_smoother = smoother_cls(shape=(init_gaussians.harmonics[..., :, 1:]).shape[1:])
        else:
            shNs_smoother = None

        self.smoothers = {
            "means": means_smoother,
            "scales": scales_smoother,
            "rotations": rotations_smoother,
            "opacities": opacities_smoother,
            "sh0s": sh0s_smoother,
            "shNs": shNs_smoother,
        }

        # get scene extent
        scene_scale = optimizer_input.context["scene_scale"]
        if scene_scale is None:
            scene_scale = torch.ones(1, 1, device=device)
        scene_scale = scene_scale[0].item()

        # initialize learning rate scheduler for means
        self.means_lr_scheduler = get_expon_lr_func(
            lr_init=self.cfg.means_lr_init * scene_scale,
            lr_final=self.cfg.means_lr_final * scene_scale,
            lr_delay_mult=self.cfg.means_lr_delay_mult,
            max_steps=self.cfg.means_lr_max_steps
        )

    def on_scene_end(self) -> None:
        super().on_scene_end()
        self.smoothers = None
        self.means_lr_scheduler = None
        self._meta_bufs.clear()

    def _forward_impl(
            self,
            i: int,
            optimizer_input: OptimizerInput,
            optimizer_output: OptimizerOutput,
            full_context: BatchedViews,
            full_target: BatchedViews,
            **kwargs
    ) -> OptimizerOutput:

        # Timing
        self.iter_start.record()

        # Unpack
        iter_context: BatchedViews = optimizer_input.context
        target: BatchedViews = optimizer_input.target
        renderer: Decoder = optimizer_input.renderer
        b, v, _, h, w = iter_context["image"].shape
        assert b == 1, "Batch size > 1 not supported for post-processing"

        # Log number of gaussians
        self.nr_gaussians_log.append(
            optimizer_input.prev_output.gaussians.means.shape[1]
        )

        # One optimization step
        res = self.apply_one_update_step(i, optimizer_input, optimizer_output, sh_degree=kwargs.get("sh_degree", None))
        gaussians: Gaussians = res[0]
        meta_for_adc: dict = res[1]
        updates: dict[str, Tensor] = res[2]
        grads_raw: dict[str, Tensor] = res[3]
        normalized_grads: dict[str, Tensor] = res[4]
        learning_rates: dict[str, float] = res[5]

        # Densification and Pruning
        if self.cfg.any_adc:
            # Apply ADC
            self.apply_adc(
                i=i, v=v, h=h, w=w,
                adc_state=optimizer_input.prev_output.state.adc_state,
                gaussians=gaussians,
                meta=meta_for_adc,
                object_dict_to_adjust=self.smoothers
            )
            # ADC changes N → cached buffers are invalid; re-make tensors as fresh leaves.
            # torch.cat (used by add_new/relocate) produces a non-leaf even with requires_grad=True,
            # so .grad is never populated by backward(). detach() cuts the grad_fn first.
            buf_nr_gaussians = self._meta_bufs['N']
            actual_nr_gaussians = gaussians.means.shape[1]
            if buf_nr_gaussians != actual_nr_gaussians:
                self._meta_bufs.clear()
                # TODO Naama: need to think if the detach is necessary (was added during mcmc implementation)
                gaussians.means = gaussians.means.detach().requires_grad_(True)
                gaussians.scales = gaussians.scales.detach().requires_grad_(True)
                gaussians.rotations_unnorm = gaussians.rotations_unnorm.detach().requires_grad_(True)
                gaussians.opacities = gaussians.opacities.detach().requires_grad_(True)
                gaussians.harmonics = gaussians.harmonics.detach().requires_grad_(True)

        # Timing
        self._record_iter_timing()

        # TODO Naama: we can log stats with save_every, but need to change stuff later.
        # Log stats — guard with save_every
        if grads_raw is not None:  # and self.save_every(i + 1, tag="info"):
            G = grads_raw["means"].shape[0]
            nonzero_grads = [(g.reshape(G, -1) != 0).any(dim=-1) for g in grads_raw.values() if g is not None]
            nonzero_grads = torch.stack(nonzero_grads)  # [num_params, G]
            nonzero_grads = nonzero_grads.any(dim=0)  # [G]
            self.nr_nonzero_grad_log.append(nonzero_grads.sum().item())

        # Save updated gaussians (for next iteration)
        optimizer_input.prev_output.gaussians = gaussians

        # Info
        if self.save_every(i + 1, tag="info"):

            # save gaussians
            optimizer_output.gaussian_list.append(gaussians, detach_and_cpu=True, save_to_disk=False, no_cache=False)

            # Save delta stats
            assert optimizer_output.info is not None

            # log deltas
            if "deltas" not in optimizer_output.info:
                optimizer_output.info["deltas"] = []
            optimizer_output.info["deltas"].append({k: v.cpu() for k, v in updates.items() if v is not None})

            # log gradients
            if "grads" not in optimizer_output.info:
                optimizer_output.info["grads"] = []
            optimizer_output.info["grads"].append({k: v.cpu() for k, v in grads_raw.items() if v is not None})

            # log normalized gradients
            if "normalized_grads" not in optimizer_output.info:
                optimizer_output.info["normalized_grads"] = []
            optimizer_output.info["normalized_grads"].append(
                {k: v.cpu() for k, v in normalized_grads.items() if v is not None})

            # log learning rates
            if "learning_rates" not in optimizer_output.info:
                optimizer_output.info["learning_rates"] = []
            optimizer_output.info["learning_rates"].append(learning_rates)

            # Check if output_path in kwargs
            output_path = kwargs.get("output_path", None)
            scene_name = kwargs.get("scene_name", None)

            # Plot stats
            # if self.cfg.any_adc:
            #     self.plot_info(i, output_path=output_path, scene_name=scene_name)

        # Post-update context + target renders
        self._save_post_update_renders(
            i, optimizer_input, optimizer_output, gaussians,
            full_context, full_target,
        )

        # Optimizer output is being changed in place, but for clarity we return it
        return optimizer_output

    def apply_one_update_step(
            self, i, optimizer_input: OptimizerInput, optimizer_output: OptimizerOutput, sh_degree: int | None = None
    ) -> tuple[Gaussians, dict | None, dict, dict[str, Tensor], dict[str, Tensor], dict[str, float]]:

        iter_context = optimizer_input.context
        b, v, _, h, w = iter_context["image"].shape
        renderer = optimizer_input.renderer
        gaussians = optimizer_input.prev_output.gaussians

        # if first iteration
        if i == 0:
            # assert gaussians stores activated values
            assert gaussians.stores_activated, "Gaussians must store activated values."
            # deactivate values in-place (avoids allocating new tensors)
            gaussians.scales.log_()  # [B, N, 3]
            gaussians.opacities.logit_()
            gaussians.stores_activated = False
            # enable requires_grad once — .grad buffers persist across steps,
            # so backward() reuses them instead of allocating new tensors each call
            gaussians.means.requires_grad_(True)
            gaussians.scales.requires_grad_(True)
            gaussians.rotations_unnorm.requires_grad_(True)
            gaussians.opacities.requires_grad_(True)
            gaussians.harmonics.requires_grad_(True)
        else:
            # assert gaussians does not store activated values
            assert not gaussians.stores_activated, "Gaussians must not store activated values."

        # learning rates
        # TODO Naama: use current cfg field lr_scheduler, which also defines the lr per param
        assert self.means_lr_scheduler is not None, "means_lr_scheduler is not initialized"
        means_lr = self.means_lr_scheduler(i) * self.cfg.base_lr
        scales_lr = self.cfg.scales_lr * self.cfg.base_lr
        rotations_lr = self.cfg.rotations_lr * self.cfg.base_lr
        opacities_lr = self.cfg.opacities_lr * self.cfg.base_lr
        sh0s_lr = self.cfg.sh0s_lr * self.cfg.base_lr
        shNs_lr = self.cfg.shNs_lr * self.cfg.base_lr

        # scale learning rates by number of views in the batch
        # means_lr *= v
        # scales_lr *= v
        # rotations_lr *= v
        # opacities_lr *= v
        # sh0s_lr *= v
        # shNs_lr *= v

        assert (
                iter_context["extrinsics"].shape[0] == iter_context["extrinsics"].shape[0] == 1
        ), "scene batch size > 1 not supported for yet..."

        # unpack gaussians
        means = gaussians.means  # [B, N, 3]
        rotations_unnorm = gaussians.rotations_unnorm  # [B, N, 4]
        scales_raw = gaussians.scales  # [B, N, 3]
        opacities_raw = gaussians.opacities  # [B, N]
        shs = gaussians.harmonics  # [B, N, 3, sh_d]

        self.decoder_event_start.record()
        loss, grads_raw, meta_for_adc = calc_input_gradients(
            iter_context,
            means,
            scales_raw,
            rotations_unnorm,
            opacities_raw,
            shs,
            renderer,
            need_2d_grads=self.cfg.need_2d_grads,
            chunk_size=self.cfg.input_gradients_chunk_size,
            any_adc=self.cfg.any_adc,
            sh_degree=sh_degree,
            meta_bufs=self._meta_bufs,
            opacity_reg_lambda=self.cfg.opacity_reg_lambda,
        )
        self.decoder_event_end.record()

        # get updates from adam optimizer
        grads_raw = squeeze_grad_dict(grads_raw)
        assert self.smoothers is not None, "Smoothers not initialized"
        grads_adam = smooth_grads(grads_raw, self.smoothers)

        # update the gaussians parameters
        # Batch delta computation for contiguous params with _foreach_mul to reduce kernel launches.
        # no_refine flags are handled by excluding the param from the batch (delta stays None).
        _grad_lr_pairs = [
            (grads_adam["means"], -means_lr, self.cfg.no_refine_mean),
            (grads_adam["scales"], -scales_lr, self.cfg.no_refine_scale),
            (grads_adam["rotations"], -rotations_lr, self.cfg.no_refine_rotation),
            (grads_adam["opacities"], -opacities_lr, self.cfg.no_refine_opacity),
        ]
        _active_grads = [g for g, lr, skip in _grad_lr_pairs if not skip]
        _active_lrs = [lr for g, lr, skip in _grad_lr_pairs if not skip]
        _active_deltas = torch._foreach_mul(_active_grads, _active_lrs) if _active_grads else []

        _delta_iter = iter(_active_deltas)
        delta_means = next(_delta_iter) if not self.cfg.no_refine_mean else None
        delta_scales_raw = next(_delta_iter) if not self.cfg.no_refine_scale else None
        delta_rotations_unnorm = next(_delta_iter) if not self.cfg.no_refine_rotation else None
        delta_opacities_raw = next(_delta_iter) if not self.cfg.no_refine_opacity else None

        # SH deltas stay separate (non-contiguous slice views)
        delta_sh0s = None if self.cfg.no_refine_sh0 else -sh0s_lr * grads_adam["sh0s"]
        delta_shNs = None
        if grads_adam["shNs"] is not None and not self.cfg.no_refine_shN:
            delta_shNs = -shNs_lr * grads_adam["shNs"]

        # step — batch contiguous params with _foreach_add_ to reduce kernel launches;
        # SH slice views are non-contiguous so they stay separate
        _params = [means, scales_raw, rotations_unnorm, opacities_raw]
        _deltas = [delta_means, delta_scales_raw, delta_rotations_unnorm, delta_opacities_raw]
        _active = [(p, d) for p, d in zip(_params, _deltas) if d is not None]
        if _active:
            torch._foreach_add_([p for p, d in _active], [d for p, d in _active])
        self.safe_inplace_update(delta_sh0s, shs[..., 0:1])
        self.safe_inplace_update(delta_shNs, shs[..., 1:])

        # assign (means/scales/rotations/harmonics are the same objects; in-place ops above
        # already updated their storage. opacities_raw is a view — do NOT reassign
        # gaussians.opacities here, as that would replace the persistent leaf with a non-leaf
        # view and break retain_grad() on subsequent steps.)
        gaussians.means = means
        gaussians.scales = scales_raw
        gaussians.rotations_unnorm = rotations_unnorm
        gaussians.harmonics = shs

        # group updates
        updates = {
            "means": delta_means,
            "scales": delta_scales_raw,
            "rotations": delta_rotations_unnorm,
            "opacities": delta_opacities_raw,
            "sh0s": delta_sh0s,
            "shNs": delta_shNs,
        }

        learning_rates = {
            "means": means_lr,
            "scales": scales_lr,
            "rotations": rotations_lr,
            "opacities": opacities_lr,
            "sh0s": sh0s_lr,
            "shNs": shNs_lr,
        }

        return gaussians, meta_for_adc, updates, grads_raw, grads_adam, learning_rates

    @staticmethod
    def safe_inplace_update(delta_means: Tensor | None, means: Tensor):
        if delta_means is not None:
            means += delta_means

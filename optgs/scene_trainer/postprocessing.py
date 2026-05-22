from dataclasses import dataclass, field
from typing import List
import tqdm as tqdm
import numpy as np
import torch
from torch import Tensor
import math
from pytorch_optimizer import load_optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from einops import rearrange
from optgs.evaluation.metrics import compute_rgb_metrics
from optgs.misc.io import FrequencyScheduler
from optgs.scene_trainer.gaussian_module import GaussiansModule, gaussians2module, module2gaussians
from optgs.model.types import Gaussians
from optgs.scene_trainer.optimizer.optimizer import OptimizerOutput
from optgs.scene_trainer.optimizer.optimizer_utils import Number3DGSCfg
from optgs.misc.detaching_cpu_list import DetachingCPUList
from optgs.dataset.camera_datasets.camera import get_scene_scale
from optgs.misc.general_utils import get_expon_lr_func
from fused_ssim import fused_ssim
from optgs.model.decoder.decoder import Decoder, DecoderOutput


@dataclass
class PostProcessADCCfg:
    """ADC (Adaptive Density Control) config for postprocessing.
    Defaults match vanilla 3DGS (config/scene_trainer/scene_optimizer/refiner/default.yaml).
    """
    do_densify: bool = True
    do_prune: bool = True
    do_opacity_reset: bool = True

    # Scheduling
    pause_refine_after_reset: int = 0
    refine_every: int = 100
    reset_every: int = 3000
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    refine_scale2d_stop_iter: int = 0

    # Densification thresholds
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01  # aka percent_dense
    grow_scale2d: float = 0.05

    # Pruning thresholds
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    min_opacity: float = 0.005

    revised_opacity: bool = False


@dataclass
class PostProcessCfg:
    name: str
    steps: int
    compute_metrics_every: int
    lr_data: Number3DGSCfg
    scheduler: str | None
    scheduler_warm_up_ratio: float

    # SGD-specific
    momentum: float = 0.0
    nesterov: bool = False

    # Adam-specific
    betas: List[float] | None = None
    eps: float = 1e-8
    amsgrad: bool = False

    # Shared
    weight_decay: float = 0.0

    # LR scheduling: steps already done by scene trainer (offsets the schedule)
    prior_steps: int = 0

    # Means LR scheduling (defaults match vanilla optimizer behavior)
    means_lr_final_ratio: float = 0.0625  # ratio of final/initial means LR (vanilla: 1e-5 / 1.6e-4)
    means_lr_delay_mult: float = 0.01    # ramp-up delay multiplier (vanilla default: 0.01)
    means_lr_scale_by_scene_extent: bool = True  # scale means LR by scene extent (vanilla default)

    # View chunking for gradient accumulation
    chunk_size: int = -1  # -1 = all views at once

    # ADC (Adaptive Density Control)
    adc: PostProcessADCCfg | None = None

    @property
    def is_active(self) -> bool:
        return self.name != "none" and self.steps > 0

    def get_dir_name(self, with_name=True):
        dir_str = self._get_dir_name()
        return f"{self.name}_{dir_str}" if with_name else dir_str

    def _get_dir_name(self):
        if self.name == "sgd":
            return f"lr{self.lr_data.base}_mom{self.momentum}"
        elif self.name == "adam":
            return f"lr{self.lr_data.base}_betas{'-'.join(map(str, self.betas or []))}_eps{self.eps}"
        return ""


def _module_to_deactivated_gaussians(gm: GaussiansModule) -> Gaussians:
    """Convert GaussiansModule to Gaussians with deactivated (raw) values for ADC."""
    return Gaussians(
        means=gm.means.detach().unsqueeze(0),
        scales=gm.scales_raw.detach().unsqueeze(0),       # log space
        opacities=gm.opacities_raw.detach().unsqueeze(0),  # logit space
        rotations=gm.rotations.detach().unsqueeze(0),
        rotations_unnorm=gm.rotations_unnorm.detach().unsqueeze(0),
        harmonics=gm.harmonics.detach().unsqueeze(0),
        stores_activated=False,
    )


def _deactivated_gaussians_to_module(gaussians: Gaussians, device: torch.device) -> GaussiansModule:
    """Convert deactivated Gaussians back to GaussiansModule."""
    assert not gaussians.stores_activated
    return GaussiansModule(
        means=gaussians.means[0].to(device),
        harmonics=gaussians.harmonics[0].to(device),
        opacities=torch.sigmoid(gaussians.opacities[0]).to(device),
        scales=torch.exp(gaussians.scales[0]).to(device),
        rotations_unnorm=gaussians.rotations_unnorm[0].to(device),
    )


class PostProcessing3DGS:

    def __init__(self, cfg: PostProcessCfg, save_every: FrequencyScheduler):
        self.cfg = cfg
        self.save_every = save_every

        # Timing
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)

        self.reset_logs()

    def reset_logs(self):
        self.radii_max_log = []
        self.grads_max_log = []
        self.nr_cloned_log = []
        self.nr_splitted_log = []
        self.nr_pruned_log = []
        self.nr_gaussians_log = []
        self.nr_nonzero_grad_log = []
        self.iter_time_log = []

    def _calc_loss(
        self, context, output_renderer: DecoderOutput
    ) -> Tensor:
        # compute scalar loss
        # assume batch size 1
        assert context["image"].shape[0] == 1
        assert context["image"].shape == output_renderer.color.shape
        l1_render_error = (output_renderer.color - context["image"]).abs().mean()

        ssim_score = fused_ssim(
            rearrange(output_renderer.color, "b v c h w -> (b v) c h w"),
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            padding="valid"
        )
        loss = 0.8 * l1_render_error + 0.2 * (1 - ssim_score)

        return loss

    def _chunked_forward_backward(self, gaussian_module, iter_context, decoder, render_res, adc_state):
        """Render views in chunks, accumulate gradients, and collect ADC metadata.

        Matches the gradient accumulation approach of calc_input_gradients in the vanilla optimizer:
        each chunk computes a mean loss, gradients accumulate, then are averaged by nr_chunks.
        """
        v = iter_context["image"].shape[1]
        chunk_size = self.cfg.chunk_size if self.cfg.chunk_size > 0 else v
        nr_chunks = math.ceil(v / chunk_size)

        # Accumulate means2d grads and radii for ADC across chunks
        need_adc = adc_state is not None
        h, w = render_res
        if need_adc:
            N = gaussian_module.means.shape[0]
            means2d_grads_all = torch.zeros((1, v, N, 2), device=gaussian_module.means.device)
            radii_all = torch.zeros((1, v, N, 2), device=gaussian_module.means.device)
            visibility_all = torch.zeros((1, v, N), dtype=torch.bool, device=gaussian_module.means.device)

        for chunk_start in range(0, v, chunk_size):
            chunk_end = min(chunk_start + chunk_size, v)

            # Slice views for this chunk
            chunk_context = {
                "image": iter_context["image"][:, chunk_start:chunk_end],
                "extrinsics": iter_context["extrinsics"][:, chunk_start:chunk_end],
                "intrinsics": iter_context["intrinsics"][:, chunk_start:chunk_end],
                "near": iter_context["near"][:, chunk_start:chunk_end],
                "far": iter_context["far"][:, chunk_start:chunk_end],
            }

            # Render
            chunk_output = decoder.forward_batch_subset(gaussian_module, chunk_context, render_res)

            # Retain means2d grad for ADC
            if need_adc and chunk_output.means2d is not None:
                chunk_output.means2d.retain_grad()

            # Loss and backward (gradients accumulate across chunks)
            chunk_loss = self._calc_loss(chunk_context, chunk_output)
            chunk_loss.backward()

            # Collect ADC metadata from this chunk
            if need_adc:
                if chunk_output.radii is not None:
                    radii_all[:, chunk_start:chunk_end] = chunk_output.radii.detach()
                if chunk_output.visibility_filter is not None:
                    visibility_all[:, chunk_start:chunk_end] = chunk_output.visibility_filter.detach()
                if chunk_output.means2d is not None and chunk_output.means2d.grad is not None:
                    means2d_grads_all[:, chunk_start:chunk_end] = chunk_output.means2d.grad.detach()

        # Average gradients across chunks (matches vanilla behavior)
        if nr_chunks > 1:
            for param in gaussian_module.parameters():
                if param.grad is not None:
                    param.grad /= nr_chunks

        # Return ADC metadata
        if need_adc:
            return {
                "radii": radii_all,
                "visibility_filter": visibility_all,
                "means_2d_grads": means2d_grads_all,
            }
        return None

    def _apply_adc(self, step, gaussian_module, adc_state, device):
        """Apply ADC (clone/split/prune/opacity reset) using the same logic as vanilla 3DGS.

        Returns (gaussian_module, optimizer_needs_rebuild).
        """
        from optgs.scene_trainer.adc.vanilla import cloning, splitting, prune, reset_adc_state

        adc_cfg = self.cfg.adc
        changed = False
        nr_cloned, nr_splitted, nr_pruned = 0, 0, 0

        # Convert to deactivated Gaussians for ADC (ADC functions expect Gaussians, not GaussiansModule)
        gaussians = _module_to_deactivated_gaussians(gaussian_module)

        if step < adc_cfg.refine_stop_iter:
            grads = adc_state.grad2d_norm_accum / adc_state.denom.clamp_min(1.0)
            scene_extent = adc_state.scene_extent

            if (
                step >= adc_cfg.refine_start_iter
                and step % adc_cfg.refine_every == 0
                and step % adc_cfg.reset_every >= adc_cfg.pause_refine_after_reset
            ):
                if adc_cfg.do_densify:
                    scales = torch.exp(gaussians.scales.squeeze(0))  # activate
                    is_grad_high = grads > adc_cfg.grow_grad2d
                    is_small = scales.max(dim=-1).values <= adc_cfg.grow_scale3d * scene_extent

                    clone_mask = is_grad_high & is_small
                    split_mask = is_grad_high & ~is_small

                    if step < adc_cfg.refine_scale2d_stop_iter:
                        split_mask |= adc_state.radii2d > adc_cfg.grow_scale2d

                    # Clone
                    cloning(gaussians, adc_state, clone_mask)
                    nr_cloned = int(clone_mask.sum().item())

                    # Extend split_mask for newly cloned points (they should not be split)
                    split_mask = torch.cat([
                        split_mask,
                        torch.zeros(nr_cloned, dtype=torch.bool, device=split_mask.device),
                    ])

                    # Split
                    splitting(gaussians, adc_state, split_mask, N=2,
                              revised_opacity=adc_cfg.revised_opacity)
                    nr_splitted = int(split_mask.sum().item())

                    changed = True

                if adc_cfg.do_prune:
                    opacities = torch.sigmoid(gaussians.opacities.squeeze(0))  # activate
                    scales = torch.exp(gaussians.scales.squeeze(0))            # activate

                    prune_mask = opacities < adc_cfg.min_opacity
                    if step > adc_cfg.reset_every:
                        is_too_big = scales.max(dim=-1).values > adc_cfg.prune_scale3d * scene_extent
                        if step < adc_cfg.refine_scale2d_stop_iter:
                            is_too_big |= adc_state.radii2d > adc_cfg.prune_scale2d
                        prune_mask = prune_mask | is_too_big

                    prune(gaussians, adc_state, prune_mask)
                    nr_pruned = int(prune_mask.sum().item())
                    changed = True

                reset_adc_state(adc_state)
                print(
                    f"ADC @ iter {step}: cloned {nr_cloned}, split {nr_splitted}, "
                    f"pruned {nr_pruned}, total {gaussians.means.shape[1]}"
                )

        # Opacity reset
        if adc_cfg.do_opacity_reset:
            if step % adc_cfg.reset_every == 0 and step > 0:
                opacities = torch.sigmoid(gaussians.opacities)  # activate
                value = adc_cfg.min_opacity * 2.0
                new_opacities = torch.min(opacities, torch.ones_like(opacities) * value)
                gaussians.opacities = torch.logit(new_opacities)  # deactivate back
                changed = True
                print(f"Opacity reset @ iter {step}")

        self.nr_cloned_log.append(nr_cloned)
        self.nr_splitted_log.append(nr_splitted)
        self.nr_pruned_log.append(nr_pruned)

        if changed:
            # Rebuild GaussiansModule from modified Gaussians
            gaussian_module = _deactivated_gaussians_to_module(gaussians, device)

        return gaussian_module, changed

    @torch.no_grad()
    def apply(
        self,
        batch,
        gaussians: Gaussians,
        decoder,
        metrics=["psnr", "ssim"],
        iter_batch_size: int = -1,
        batchify_fn=None,
        visualization_dump=None
    ) -> OptimizerOutput | None:

        target_render_list = DetachingCPUList()
        context_render_list = DetachingCPUList()

        if self.cfg.steps == 0:
            return None

        # [Improvement 1] Calculate scene_scale from both context + target (matches vanilla optimizer)
        camtoworlds_context = batch['context']['extrinsics'][0].cpu().numpy()  # [Vc, 4, 4]
        camtoworlds_target = batch['target']['extrinsics'][0].cpu().numpy()    # [Vt, 4, 4]
        camtoworlds = np.concatenate([camtoworlds_context, camtoworlds_target], axis=0)
        scene_scale = get_scene_scale(camtoworlds)
        print("scene_scale:", scene_scale)

        device = batch['context']['image'].device

        # convert Gaussians to GaussiansModule
        gaussian_module = gaussians2module(gaussians, device=device)

        optimizer = self.get_optimizer(gaussian_module, scene_scale)
        scheduler = self.get_scheduler(optimizer, scene_scale=scene_scale, prior_steps=self.cfg.prior_steps)

        # print all optimizer param groups
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Param group {i}: lr={param_group['lr']}, weight_decay={param_group.get('weight_decay', 0.0)}, requires_grad={param_group['params'][0].requires_grad}")

        assert batch["context"]["extrinsics"].shape[0] == batch["context"]["extrinsics"].shape[0] == 1, \
            "Batch size > 1 not supported for post-processing"

        nr_context_views, _, h, w = batch["context"]["image"][0].shape

        # controlling number of context views seen at each iteration (for rendering chunk size)
        _iter_batch_size = iter_batch_size if iter_batch_size > 0 else nr_context_views
        print("using iter_batch_size =", _iter_batch_size)

        render_res = (h, w)

        # [Improvement 3] Initialize ADC state if configured
        adc_state = None
        if self.cfg.adc is not None:
            from optgs.scene_trainer.adc.vanilla import VanillaStrategyState
            nr_points = gaussian_module.means.shape[0]
            adc_state = VanillaStrategyState.initialize(
                nr_points=nr_points,
                device=device,
                scene_extent=scene_scale,
            )
            print(f"Initialized ADC state with {nr_points} points")

        # render before first step
        context_render_output = decoder.forward_batch_subset(gaussian_module, batch["context"], render_res, iter_batch_size=_iter_batch_size)
        context_render_list.append(context_render_output, detach_and_cpu=True)  # initial rendering

        target_render_output = decoder.forward_batch_subset(gaussian_module, batch["target"], render_res, iter_batch_size=_iter_batch_size)
        target_render_list.append(target_render_output, detach_and_cpu=True)  # initial rendering

        # Reset viewpoint stack for fresh sampling in postprocessing
        batch["context"].viewpoint_stack = None

        pbar = tqdm.tqdm(range(self.cfg.steps), desc=f"PP {self.cfg.name}", ncols=120)
        pbar_postfix = {}
        for i in pbar:

            self.iter_start.record()

            with torch.enable_grad():

                # Log number of gaussians
                self.nr_gaussians_log.append(gaussian_module.means.shape[0])

                # reset gradients
                optimizer.zero_grad()

                # Sample context views using the same strategy as the optimizer
                iter_context, _ = batchify_fn(batch, "context")

                # [Improvement 4] Render in chunks, accumulate gradients, collect ADC metadata
                meta_for_adc = self._chunked_forward_backward(
                    gaussian_module, iter_context, decoder, render_res, adc_state
                )

                # step
                optimizer.step()

                # update scheduler
                if scheduler is not None:
                    scheduler.step()

            # [Improvement 3] ADC: update state and apply densification/pruning
            if adc_state is not None and meta_for_adc is not None:
                from optgs.scene_trainer.adc.vanilla import update_vanilla_strategy_state

                v_chunk = iter_context["image"].shape[1]
                update_vanilla_strategy_state(
                    adc_state,
                    radii_2d=meta_for_adc["radii"],
                    means2d_grads=meta_for_adc["means_2d_grads"],
                    visibility_mask=meta_for_adc["visibility_filter"],
                    v=v_chunk,
                    w=w,
                    h=h,
                )

                gaussian_module, adc_changed = self._apply_adc(i, gaussian_module, adc_state, device)
                if adc_changed:
                    # Rebuild optimizer and scheduler after ADC changed Gaussian count
                    optimizer = self.get_optimizer(gaussian_module, scene_scale)
                    scheduler = self.get_scheduler(
                        optimizer, scene_scale=scene_scale, prior_steps=self.cfg.prior_steps
                    )
                    # Fast-forward scheduler to current step
                    for _ in range(i + 1):
                        scheduler.step() if scheduler is not None else None

            # Timing
            self.iter_end.record()
            torch.cuda.synchronize()

            elapsed_time = self.iter_start.elapsed_time(self.iter_end)
            self.iter_time_log.append(elapsed_time)

            if self.save_every(i + 1, tag="context"):
                with torch.no_grad():
                    context_render_output = decoder.forward_context(gaussian_module, batch, (h, w))
                    context_render_list.append(context_render_output, detach_and_cpu=True)
                    context_rgb = context_render_output.color[0]  # [Vc, 3, Hc, Wc]
                    ctx_scores: dict = compute_rgb_metrics(
                        rgb=context_rgb,
                        rgb_gt=batch["context"]["image"][0],
                        metrics=metrics,
                        iter_batch_size=iter_batch_size if "lpips" in metrics else -1
                    )
                    for k, v in ctx_scores.items():
                        pbar_postfix[f"ctx_{k}"] = f"{v.item():.2f}"

            if self.save_every(i + 1, tag="target"):
                with torch.no_grad():
                    target_render_output = decoder.forward_target(gaussian_module, batch, (h, w))
                    target_render_list.append(target_render_output, detach_and_cpu=True)
                    target_rgb = target_render_output.color[0]  # [Vt, 3, Ht, Wt]
                    tgt_scores: dict = compute_rgb_metrics(
                        rgb=target_rgb,
                        rgb_gt=batch["target"]["image"][0],
                        metrics=metrics,
                        iter_batch_size=iter_batch_size if "lpips" in metrics else -1
                    )
                    for k, v in tgt_scores.items():
                        pbar_postfix[f"tgt_{k}"] = f"{v.item():.2f}"

            pbar_postfix["gs"] = gaussian_module.means.shape[0]
            pbar.set_postfix(pbar_postfix)

            if visualization_dump is not None and "grads" in visualization_dump:
                self.debug_grads(gaussian_module, visualization_dump, i)

        # convert back to Gaussians

        postprocessed_gaussians = module2gaussians(gaussian_module)
        postprocessed_gaussians_list = DetachingCPUList()
        postprocessed_gaussians_list.append(postprocessed_gaussians, detach_and_cpu=True)
        output = OptimizerOutput(
            target_render_list=target_render_list,
            context_render_list=context_render_list,
            gaussian_list=postprocessed_gaussians_list,
            info = {}
        )

        return output

    def debug_grads(self, gaussians: GaussiansModule, debug_dict, step):
        if debug_dict["grads"] is None:
            # First iteration, first scene
            debug_dict["grads"] = [[]]
        elif step == 0:
            # New iteration, new scene
            debug_dict["grads"].append([])

        grads = [param.grad for name, param in gaussians.named_parameters() if param.grad is not None]
        gaussian_num = gaussians.means.shape[0]
        grads = [g.view(gaussian_num, -1) for g in grads]
        grads = [g.detach().cpu() for g in grads]
        grads = torch.cat(grads, dim=-1)  # [num_gaussians, total_param_dim]

        debug_dict["grads"][-1].append(grads)

    def get_optimizer(self, gaussians: GaussiansModule, scene_scale: float):

        # TODO Naama: support different batch sizes
        batch_size: int = 1

        # Build params list (name, parameter, lr)
        named_parameters = dict(gaussians.named_parameters())
        params = []
        for key in named_parameters.keys():
            lr_data_attr = key
            lr_data_attr = lr_data_attr.replace("_raw", "")
            lr_data_attr = lr_data_attr.replace("_unnorm", "")
            params.append((key, named_parameters[key], getattr(self.cfg.lr_data, lr_data_attr)))

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        print(f"World size: {world_size}")

        BS = batch_size * world_size
        # Build parameter groups for a single optimizer
        param_groups = [
            {
                "name": name,
                "params": param,
                "lr": lr * math.sqrt(BS),  # individual learning rate
            }
            for name, param, lr in params
        ]

        # Get other optimizer parameters
        opt_params = self.extract_opt_params()

        # Manipulate opt_params with BS if needed
        if "weight_decay" in opt_params:
            opt_params["weight_decay"] *= BS
        if "eps" in opt_params:
            opt_params["eps"] /= math.sqrt(BS)
        if "betas" in opt_params:
            beta1, beta2 = opt_params["betas"]
            opt_params["betas"] = (1 - BS * (1 - beta1), 1 - BS * (1 - beta2))

        # Instantiate a single optimizer with all parameter groups
        optimizer_class = load_optimizer(self.cfg.name)
        optimizer = optimizer_class(
            param_groups,
            **opt_params
        )

        # Print out info for debugging
        print("Optimizer with parameter groups:")
        for i, group in enumerate(optimizer.param_groups):
            print(
                f"Group {i} ({group.get('name', 'unnamed')}): "
                f"lr={group['lr']} params={len(group['params'])}"
            )

        return optimizer


    _OPT_PARAMS = {
        "sgd":  ("momentum", "weight_decay", "nesterov"),
        "adam": ("betas", "eps", "weight_decay", "amsgrad"),
    }

    def extract_opt_params(self):
        allowed = self._OPT_PARAMS.get(self.cfg.name, ())
        return {k: getattr(self.cfg, k) for k in allowed if getattr(self.cfg, k, None) is not None}

    def get_scheduler(self, optimizer, scene_scale: float = 1.0, prior_steps: int = 0):
        if self.cfg.scheduler is None:
            return None

        total_steps = prior_steps + self.cfg.steps

        if self.cfg.scheduler == "exponential":
            print(f"Using exponential LR scheduler (total_steps={total_steps}, prior_steps={prior_steps})")

            # [Improvement 2] Per-param-group scheduling:
            # - Means: exponential decay optionally scaled by scene_extent (matching vanilla optimizer)
            # - Other params: constant LR
            lambdas = []
            for group in optimizer.param_groups:
                if group["name"] == "means" and self.cfg.means_lr_scale_by_scene_extent:
                    # Vanilla-style means LR: exponential decay with scene_extent scaling
                    base_lr = group["lr"]  # initial means LR from param group
                    means_lr_func = get_expon_lr_func(
                        lr_init=base_lr * scene_scale,
                        lr_final=base_lr * scene_scale * self.cfg.means_lr_final_ratio,
                        lr_delay_mult=self.cfg.means_lr_delay_mult,
                        max_steps=total_steps,
                    )
                    # LambdaLR computes: effective_lr = base_lr * lambda(step)
                    # We want: effective_lr = means_lr_func(step)
                    # So: lambda(step) = means_lr_func(step) / base_lr
                    _base_lr = base_lr  # capture for closure
                    _func = means_lr_func
                    lambdas.append(lambda step, f=_func, b=_base_lr: f(step) / b)
                else:
                    # Constant LR for all other param groups
                    lambdas.append(lambda step: 1.0)

            scheduler = LambdaLR(optimizer, lr_lambda=lambdas)
            # Fast-forward to prior_steps so LR continues from where scene trainer left off
            for _ in range(prior_steps):
                scheduler.step()
            return scheduler

        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")

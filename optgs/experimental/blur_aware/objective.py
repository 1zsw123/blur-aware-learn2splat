"""Blur-aware image-formation objective for the Learn2Splat optimizer."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from fused_ssim import FusedSSIMMap
from torch import Tensor, nn


@dataclass(frozen=True)
class BlurAwareObjectiveConfig:
    kernel_size: int = 9
    kernel_dilation: int = 2
    camera_embedding_dim: int = 32
    bpn_learning_rate: float = 1e-3
    bpn_weight_decay: float = 1e-6
    bpn_grad_clip: float = 1.0
    sharp_supervision_weight: float = 10.0
    sharp_weight_in_sampler: bool = False
    raw_ramp_start: float = 0.05
    raw_ramp_end: float = 0.25
    center_regularization: float = 1e-3
    mask_regularization: float = 1e-2
    mask_tv_regularization: float = 1e-3
    # Spatial, confidence-gated edge supervision for non-sharp observations.
    # Set to 0 for exact rollback to the reconstruction-only objective.
    laplacian_loss_weight: float = 0.1
    laplacian_loss_mode: str = "spatial"
    surplus_ema_decay: float = 0.95
    coupled_dual_bpn: bool = False

    def __post_init__(self):
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        if self.kernel_dilation < 1:
            raise ValueError("kernel_dilation must be >= 1")
        if not 0.0 <= self.raw_ramp_start < self.raw_ramp_end <= 1.0:
            raise ValueError("raw ramp must satisfy 0 <= start < end <= 1")
        if self.sharp_supervision_weight < 1.0:
            raise ValueError("sharp_supervision_weight must be >= 1")
        if self.laplacian_loss_weight < 0.0:
            raise ValueError("laplacian_loss_weight must be non-negative")
        if self.laplacian_loss_mode not in {"spatial", "energy", "surplus"}:
            raise ValueError(
                "laplacian_loss_mode must be 'spatial', 'energy', or 'surplus'"
            )
        if not 0.0 <= self.surplus_ema_decay < 1.0:
            raise ValueError("surplus_ema_decay must satisfy 0 <= decay < 1")


class FactorizedBlurFormation(nn.Module):
    """A low-rank BPN that cannot encode high-frequency texture as a kernel.

    Each training view gets one positive, normalized blur kernel. A shared
    low-resolution mask network decides where that kernel is active. This is
    expressive enough for camera-motion and defocus blur while ruling out the
    old per-pixel-kernel shortcut that copied scene edges into the blur field.
    """

    def __init__(self, num_views: int, cfg: BlurAwareObjectiveConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.camera_embedding_dim
        k2 = cfg.kernel_size**2
        self.camera_embedding = nn.Embedding(num_views, d)
        self.kernel_head = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, k2),
        )
        self.mask_net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.normal_(self.camera_embedding.weight, std=0.02)
        nn.init.zeros_(self.kernel_head[-1].weight)
        nn.init.constant_(self.kernel_head[-1].bias, -4.0)
        self.kernel_head[-1].bias.data[k2 // 2] = 4.0
        nn.init.zeros_(self.mask_net[-1].weight)
        nn.init.constant_(self.mask_net[-1].bias, -2.0)
        self.strength_head: nn.Linear | None = None
        if cfg.coupled_dual_bpn:
            self.strength_head = nn.Linear(d, 2)
            nn.init.zeros_(self.strength_head.weight)
            # EVSSM starts near identity; RAW starts with substantially more
            # of the same blur mode while remaining learnable per view.
            self.strength_head.bias.data.copy_(torch.tensor((-4.0, 2.0)))

        radius = cfg.kernel_size // 2
        axis = torch.linspace(-1.0, 1.0, cfg.kernel_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("kernel_x", xx.reshape(1, -1), persistent=False)
        self.register_buffer("kernel_y", yy.reshape(1, -1), persistent=False)

    def kernel_family(self, view_indices: Tensor) -> dict[str, Tensor]:
        """Return coupled EVSSM/RAW kernels in Gaussian split-independent order."""
        indices = view_indices.reshape(-1).long()
        embedding = self.camera_embedding(indices)
        base = torch.softmax(self.kernel_head(embedding), dim=-1)
        identity = torch.zeros_like(base)
        identity[:, base.shape[-1] // 2] = 1.0
        if self.strength_head is None:
            teacher_strength = base.new_zeros(base.shape[0])
            raw_strength = base.new_ones(base.shape[0])
        else:
            strength_logits = self.strength_head(embedding)
            teacher_strength = torch.sigmoid(strength_logits[:, 0])
            raw_residual = torch.sigmoid(strength_logits[:, 1])
            raw_strength = teacher_strength + (
                1.0 - teacher_strength
            ) * raw_residual

        def mix(strength: Tensor) -> Tensor:
            return identity + strength[:, None] * (base - identity)

        return {
            "base_kernels": base,
            "teacher_kernels": mix(teacher_strength),
            "raw_kernels": mix(raw_strength),
            "teacher_strength": teacher_strength,
            "raw_strength": raw_strength,
        }

    def _apply_kernels(self, sharp: Tensor, kernels: Tensor) -> Tensor:
        count, channels, height, width = sharp.shape
        kernel_2d = kernels.view(
            count, 1, self.cfg.kernel_size, self.cfg.kernel_size
        )
        grouped_weight = kernel_2d.repeat_interleave(channels, dim=0)
        grouped_input = sharp.reshape(1, count * channels, height, width)
        pad = (self.cfg.kernel_size // 2) * self.cfg.kernel_dilation
        grouped_input = F.pad(
            grouped_input, (pad, pad, pad, pad), mode="reflect"
        )
        return F.conv2d(
            grouped_input,
            grouped_weight,
            dilation=self.cfg.kernel_dilation,
            groups=count * channels,
        ).reshape(count, channels, height, width)

    @staticmethod
    def _gray(images: Tensor) -> Tensor:
        weights = images.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        return (images * weights).sum(dim=1, keepdim=True)

    def forward(
        self,
        sharp: Tensor,
        depth: Tensor | None,
        raw: Tensor,
        target: Tensor,
        view_indices: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        b, v, c, h, w = sharp.shape
        flat_count = b * v
        sharp_flat = sharp.reshape(flat_count, c, h, w)
        raw_flat = raw.reshape(flat_count, c, h, w)
        target_flat = target.reshape(flat_count, c, h, w)
        family = self.kernel_family(view_indices)
        raw_blurred = self._apply_kernels(sharp_flat, family["raw_kernels"])
        teacher_blurred = self._apply_kernels(
            sharp_flat, family["teacher_kernels"]
        )

        low_h, low_w = max(8, (h + 15) // 16), max(8, (w + 15) // 16)
        raw_gray = self._gray(raw_flat.detach())
        target_gray = self._gray(target_flat.detach())
        raw_low = F.interpolate(raw_gray, (low_h, low_w), mode="area")
        target_low = F.interpolate(target_gray, (low_h, low_w), mode="area")
        residual_low = (target_low - raw_low).abs()
        if depth is None:
            depth_low = torch.zeros_like(raw_low)
        else:
            depth_flat = depth.detach().reshape(flat_count, 1, h, w)
            depth_low = F.interpolate(depth_flat, (low_h, low_w), mode="area")
            depth_min = depth_low.flatten(1).amin(dim=1).view(-1, 1, 1, 1)
            depth_max = depth_low.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
            depth_low = (depth_low - depth_min) / (depth_max - depth_min + 1e-6)
        mask_input = torch.cat((raw_low, target_low, residual_low, depth_low), dim=1)
        mask_low = torch.sigmoid(self.mask_net(mask_input))
        mask = F.interpolate(mask_low, (h, w), mode="bilinear", align_corners=False)
        formed = mask * raw_blurred + (1.0 - mask) * sharp_flat
        teacher_formed = mask * teacher_blurred + (1.0 - mask) * sharp_flat

        kernels = family["raw_kernels"]
        center_x = (family["base_kernels"] * self.kernel_x).sum(dim=-1)
        center_y = (family["base_kernels"] * self.kernel_y).sum(dim=-1)
        center_loss = (center_x.square() + center_y.square()).mean()
        residual_scale = residual_low.flatten(1).quantile(0.9, dim=1).view(-1, 1, 1, 1)
        mask_target = (residual_low / (residual_scale + 1e-6)).clamp(0.0, 1.0)
        mask_loss = F.l1_loss(mask_low, mask_target)
        mask_tv = (mask_low[..., 1:, :] - mask_low[..., :-1, :]).abs().mean()
        mask_tv = mask_tv + (mask_low[..., :, 1:] - mask_low[..., :, :-1]).abs().mean()
        entropy = -(kernels * kernels.clamp_min(1e-8).log()).sum(dim=-1)

        return formed.view(b, v, c, h, w), {
            "kernels": kernels,
            **family,
            "teacher_formed": teacher_formed.view(b, v, c, h, w),
            "mask": mask.view(b, v, 1, h, w),
            "mask_low": mask_low,
            "center_loss": center_loss,
            "mask_loss": mask_loss,
            "mask_tv": mask_tv,
            "kernel_entropy": entropy.mean(),
        }


class BlurAwareObjective(nn.Module):
    """Forms Learn2Splat input gradients from reliable sharp and RAW targets."""

    def __init__(
        self,
        num_views: int,
        cfg: BlurAwareObjectiveConfig | None = None,
        known_sharp_mask: Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg or BlurAwareObjectiveConfig()
        self.bpn = FactorizedBlurFormation(num_views, self.cfg)
        if known_sharp_mask is None:
            known_sharp_mask = torch.zeros(num_views, dtype=torch.bool)
        known_sharp_mask = torch.as_tensor(known_sharp_mask, dtype=torch.bool)
        if known_sharp_mask.shape != (num_views,):
            raise ValueError("known_sharp_mask must have shape [num_views]")
        self.register_buffer("known_sharp_mask", known_sharp_mask.clone())
        self.register_buffer("surplus_gain_ema", torch.zeros(num_views))
        self.register_buffer("surplus_gain_square_ema", torch.zeros(num_views))
        self.register_buffer("surplus_gain_updates", torch.zeros(num_views))
        self.num_steps = 1
        self.step = 0
        self._optimizer: torch.optim.Optimizer | None = None
        self._gradient_accumulator: list[Tensor | None] = []
        self._gradient_chunks = 0
        self.last_diagnostics: dict[str, float] = {}
        self._densification_feedback_revision = 0
        self._densification_feedback: dict[str, float | int | bool] | None = None
        self._freeze_statistics_depth = 0

    def configure_run(self, num_steps: int) -> None:
        self.num_steps = max(1, int(num_steps))
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.bpn_learning_rate,
                weight_decay=self.cfg.bpn_weight_decay,
            )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @contextmanager
    def frozen_observation(self, step: int | None = None):
        """Evaluate the joint objective without training BPN or reliability state.

        LeGS queries the same objective to construct its per-Gaussian policy
        state.  That query is observational: it must not perform a second BPN
        optimizer step or count the same render twice in the surplus EMA.
        """
        previous_step = self.step
        previous_training = self.training
        self.step = previous_step if step is None else int(step)
        self._freeze_statistics_depth += 1
        self.eval()
        try:
            yield self
        finally:
            self._freeze_statistics_depth -= 1
            self.step = previous_step
            self.train(previous_training)

    def set_densification_feedback(
        self,
        *,
        probe_psnr: float,
        probe_surplus: float,
        has_surplus: bool,
    ) -> None:
        """Publish one fixed-training-probe observation for delayed ADC reward.

        This channel is deliberately detached from the image-loss graph.  The
        ADC consumes changes between successive observations, so a structural
        action is judged only after its complete control interval.
        """
        values = (float(probe_psnr), float(probe_surplus))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("densification feedback must be finite")
        self._densification_feedback_revision += 1
        self._densification_feedback = {
            "revision": self._densification_feedback_revision,
            "probe_psnr": values[0],
            "probe_surplus": values[1],
            "has_surplus": bool(has_surplus),
        }

    def densification_feedback(self) -> dict[str, float | int | bool] | None:
        """Return the latest immutable ADC feedback snapshot, if available."""
        return (
            None
            if self._densification_feedback is None
            else dict(self._densification_feedback)
        )

    def begin_step(self, step: int | None, context) -> None:
        self.step = int(step or 0)
        self.train()
        self._gradient_accumulator = [None for _ in self.trainable_parameters()]
        self._gradient_chunks = 0

    def accumulate_parameter_grads(self, gradients) -> None:
        self._gradient_chunks += 1
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            detached = gradient.detach()
            current = self._gradient_accumulator[index]
            self._gradient_accumulator[index] = (
                detached.clone() if current is None else current + detached
            )

    def end_step(self) -> None:
        if self._optimizer is None or self._gradient_chunks == 0:
            return
        self._optimizer.zero_grad(set_to_none=True)
        parameters = self.trainable_parameters()
        scale = 1.0 / self._gradient_chunks
        for parameter, gradient in zip(parameters, self._gradient_accumulator):
            if gradient is not None:
                parameter.grad = gradient * scale
        torch.nn.utils.clip_grad_norm_(parameters, self.cfg.bpn_grad_clip)
        self._optimizer.step()

    @staticmethod
    def _per_view_loss(
        prediction: Tensor,
        target: Tensor,
        with_ssim: bool,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        b, v = prediction.shape[:2]
        absolute_error = (prediction - target).abs()
        if valid_mask is None:
            l1 = absolute_error.mean(dim=(2, 3, 4))
        else:
            valid = valid_mask.to(dtype=absolute_error.dtype)
            denominator = valid.sum(dim=(2, 3, 4)).clamp_min(1.0) * target.shape[2]
            l1 = (absolute_error * valid).sum(dim=(2, 3, 4)) / denominator
        if not with_ssim:
            return l1
        pred_flat = prediction.reshape(b * v, *prediction.shape[2:]).contiguous()
        target_flat = target.reshape(b * v, *target.shape[2:]).contiguous()
        ssim_map = FusedSSIMMap.apply(
            0.01**2, 0.03**2, pred_flat, target_flat, "valid", True
        )
        ssim_loss = (1.0 - ssim_map).mean(dim=(1, 2, 3)).view(b, v)
        return 0.8 * l1 + 0.2 * ssim_loss

    @staticmethod
    def _laplacian_log_energy(
        images: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Return resolution-bounded log Laplacian energy per view.

        RAW, target, and render are reduced by the same image-only transform,
        so their log ratios are invariant to a common intensity scale. The
        bounded working resolution keeps this loss inexpensive relative to
        rendering and avoids making its compute depend on the dataset name.
        """
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,V,3,H,W]")
        b, v, _, h, w = images.shape
        flat = images.reshape(b * v, 3, h, w).float()
        max_side = max(h, w)
        if max_side > 192:
            scale = 192.0 / max_side
            size = (max(8, round(h * scale)), max(8, round(w * scale)))
            flat = F.interpolate(flat, size=size, mode="area")
        gray_weights = flat.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        gray = (flat * gray_weights).sum(dim=1, keepdim=True)
        kernel = gray.new_tensor(
            ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
        ).view(1, 1, 3, 3)
        laplacian = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel)

        if valid_mask is None:
            energy = laplacian.square().mean(dim=(1, 2, 3))
        else:
            valid = valid_mask.reshape(b * v, 1, h, w).float()
            if valid.shape[-2:] != laplacian.shape[-2:]:
                valid = F.interpolate(valid, size=laplacian.shape[-2:], mode="nearest")
            # Exclude one pixel around invalid camera-domain boundaries so the
            # Laplacian cannot turn a crop/undistortion edge into sharpness.
            valid = 1.0 - F.max_pool2d(1.0 - valid, 3, stride=1, padding=1)
            denominator = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
            energy = (laplacian.square() * valid).sum(dim=(1, 2, 3)) / denominator
        return torch.log(energy + 1e-8).view(b, v)

    @classmethod
    def _laplacian_energy_objective(
        cls,
        prediction: Tensor,
        raw: Tensor,
        target: Tensor,
        known_sharp: Tensor,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Measure whether the render recovers the target's sharpness gain.

        For a non-sharp observation, only a deficit relative to the EVSSM
        gain is penalized; surpassing EVSSM is recorded but never rewarded with
        an unbounded negative loss. For an authoritative sharp observation,
        the render's log energy is matched directly to RAW.
        """
        render_log = cls._laplacian_log_energy(prediction, valid_mask)
        raw_log = cls._laplacian_log_energy(raw.detach(), valid_mask)
        target_log = cls._laplacian_log_energy(target.detach(), valid_mask)
        teacher_gain = target_log - raw_log
        render_gain = render_log - raw_log
        relative_gain = render_gain - teacher_gain
        catchup = F.relu(-relative_gain)
        sharp_match = (render_log - raw_log).abs()
        per_view = torch.where(known_sharp, sharp_match, catchup)
        return per_view, {
            "laplacian_teacher_gain": teacher_gain,
            "laplacian_render_gain": render_gain,
            "laplacian_relative_gain": relative_gain,
            "laplacian_catchup": catchup,
            "laplacian_sharp_match": sharp_match,
        }

    @staticmethod
    def _laplacian_pyramid(
        images: Tensor,
        valid_mask: Tensor | None,
    ) -> list[tuple[Tensor, Tensor | None]]:
        """Return an antialiased signed-Laplacian pyramid.

        Unlike a global sharpness scalar, signed spatial responses retain edge
        location and polarity. A sparse impulse therefore cannot compensate
        for a missing edge elsewhere in the image.
        """
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,V,3,H,W]")
        b, v, _, h, w = images.shape
        flat = images.reshape(b * v, 3, h, w).float()
        max_side = max(h, w)
        if max_side > 256:
            resize_scale = 256.0 / max_side
            base_size = (
                max(8, round(h * resize_scale)),
                max(8, round(w * resize_scale)),
            )
            flat = F.interpolate(flat, size=base_size, mode="area")
        gray_weights = flat.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        gray = (flat * gray_weights).sum(dim=1, keepdim=True)

        valid_base = None
        if valid_mask is not None:
            valid_base = valid_mask.reshape(b * v, 1, h, w).float()
            if valid_base.shape[-2:] != gray.shape[-2:]:
                valid_base = F.interpolate(
                    valid_base, size=gray.shape[-2:], mode="nearest"
                )

        gaussian = gray.new_tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))
        ).view(1, 1, 3, 3) / 16.0
        laplace = gray.new_tensor(
            ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
        ).view(1, 1, 3, 3)

        pyramid = []
        for divisor in (1, 2, 4):
            if divisor == 1:
                current = gray
                current_valid = valid_base
            else:
                size = (
                    max(4, gray.shape[-2] // divisor),
                    max(4, gray.shape[-1] // divisor),
                )
                current = F.interpolate(gray, size=size, mode="area")
                current_valid = (
                    None
                    if valid_base is None
                    else F.interpolate(valid_base, size=size, mode="nearest")
                )
            smoothed = F.conv2d(
                F.pad(current, (1, 1, 1, 1), mode="reflect"), gaussian
            )
            response = F.conv2d(
                F.pad(smoothed, (1, 1, 1, 1), mode="reflect"), laplace
            )
            if current_valid is not None:
                current_valid = 1.0 - F.max_pool2d(
                    1.0 - current_valid, 5, stride=1, padding=2
                )
                current_valid = current_valid.view(
                    b, v, 1, *current_valid.shape[-2:]
                )
            pyramid.append(
                (response.view(b, v, 1, *response.shape[-2:]), current_valid)
            )
        return pyramid

    @staticmethod
    def _masked_mean_per_view(values: Tensor, valid: Tensor | None) -> Tensor:
        if valid is None:
            return values.mean(dim=(2, 3, 4))
        denominator = valid.sum(dim=(2, 3, 4)).clamp_min(1.0)
        return (values * valid).sum(dim=(2, 3, 4)) / denominator

    @staticmethod
    def _masked_supported_energy(
        response: Tensor,
        support: Tensor,
        valid: Tensor | None,
    ) -> Tensor:
        weight = support if valid is None else support * valid
        denominator = weight.sum(dim=(2, 3, 4)).clamp_min(1e-8)
        return (response.square() * weight).sum(dim=(2, 3, 4)) / denominator

    @classmethod
    @torch.no_grad()
    def measure_probe_surplus(
        cls,
        prediction: Tensor,
        raw: Tensor,
        target: Tensor,
        known_sharp: Tensor,
        confidence: Tensor,
        valid_mask: Tensor | None,
    ) -> dict[str, float | bool]:
        """Measure supported render-over-EVSSM sharpness on fixed train views.

        The statistic uses the same three-scale, RAW/EVSSM-supported Laplacian
        energy as the surplus loss.  Known-sharp observations are excluded:
        their teacher is authoritative, so PSNR rather than teacher surplus is
        the meaningful densification signal.
        """
        prediction_pyramid = cls._laplacian_pyramid(prediction, valid_mask)
        target_pyramid = cls._laplacian_pyramid(target.detach(), valid_mask)
        raw_pyramid = cls._laplacian_pyramid(raw.detach(), valid_mask)
        scale_weights = prediction.new_tensor((1.0, 0.5, 0.25))
        scale_weights = scale_weights / scale_weights.sum()
        teacher_energies: list[Tensor] = []
        render_energies: list[Tensor] = []
        raw_energies: list[Tensor] = []
        for (predicted, valid), (expected, _), (raw_response, _) in zip(
            prediction_pyramid, target_pyramid, raw_pyramid
        ):
            evidence = torch.maximum(expected.abs(), raw_response.abs())
            evidence_scale = cls._masked_mean_per_view(evidence, valid)
            evidence_scale = evidence_scale[..., None, None, None].clamp_min(1e-6)
            support = (1.0 - torch.exp(-evidence / evidence_scale)).detach()
            teacher_energies.append(
                cls._masked_supported_energy(expected, support, valid)
            )
            render_energies.append(
                cls._masked_supported_energy(predicted, support, valid)
            )
            raw_energies.append(
                cls._masked_supported_energy(raw_response, support, valid)
            )

        def multiscale_log_energy(energies: list[Tensor]) -> Tensor:
            stacked = torch.stack(
                [torch.log(value + 1e-8) for value in energies], dim=-1
            )
            return (stacked * scale_weights.view(1, 1, -1)).sum(dim=-1)

        raw_log = multiscale_log_energy(raw_energies)
        teacher_gain = multiscale_log_energy(teacher_energies) - raw_log
        render_gain = multiscale_log_energy(render_energies) - raw_log
        relative_gain = render_gain - teacher_gain
        eligible = (~known_sharp.bool()).to(relative_gain.dtype)
        weights = eligible * confidence.detach().clamp(0.0, 1.0)
        denominator = weights.sum()
        has_surplus = bool(float(denominator) > 0.0)
        if has_surplus:
            surplus = (weights * torch.tanh(relative_gain)).sum() / denominator
            teacher = (weights * teacher_gain).sum() / denominator
            render = (weights * render_gain).sum() / denominator
        else:
            surplus = relative_gain.new_zeros(())
            teacher = relative_gain.new_zeros(())
            render = relative_gain.new_zeros(())
        return {
            "surplus": float(surplus),
            "teacher_gain": float(teacher),
            "render_gain": float(render),
            "has_surplus": has_surplus,
        }

    @torch.no_grad()
    def _update_surplus_reliability(
        self,
        relative_gain: Tensor,
        static_confidence: Tensor,
        known_sharp: Tensor,
        view_indices: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Convert stable cross-view sharpness surplus into teacher uncertainty.

        A single instantaneous render never controls its own supervision weight.
        Positive render-over-EVSSM gain must persist in a per-view EMA and then
        contribute to a scene consensus. EMA variance discounts unstable gains,
        while ``1 - decay**updates`` provides a threshold-free maturity ramp.
        Every returned weight is detached from the rendering graph.
        """
        if view_indices.shape != relative_gain.shape:
            raise ValueError("view_indices and relative_gain must have the same shape")
        flat_indices = view_indices.reshape(-1).long()
        if flat_indices.numel() and (
            int(flat_indices.min()) < 0
            or int(flat_indices.max()) >= self.surplus_gain_ema.numel()
        ):
            raise IndexError("view index is outside the objective's view state")

        decay = self.cfg.surplus_ema_decay
        if self._freeze_statistics_depth == 0:
            flat_gain = relative_gain.detach().float().reshape(-1)
            flat_sharp = known_sharp.reshape(-1)
            for index, value, is_sharp in zip(flat_indices, flat_gain, flat_sharp):
                if bool(is_sharp):
                    continue
                idx = int(index)
                count = self.surplus_gain_updates[idx]
                if float(count) == 0.0:
                    self.surplus_gain_ema[idx] = value
                    self.surplus_gain_square_ema[idx] = value.square()
                else:
                    self.surplus_gain_ema[idx].mul_(decay).add_(
                        value, alpha=1.0 - decay
                    )
                    self.surplus_gain_square_ema[idx].mul_(
                        decay
                    ).add_(value.square(), alpha=1.0 - decay)
                self.surplus_gain_updates[idx].add_(1.0)

        variance = (
            self.surplus_gain_square_ema - self.surplus_gain_ema.square()
        ).clamp_min(0.0)
        stable_positive_gain = F.relu(self.surplus_gain_ema - variance.sqrt())
        maturity = 1.0 - torch.pow(
            self.surplus_gain_updates.new_tensor(decay), self.surplus_gain_updates
        )
        eligible = (~self.known_sharp_mask).to(stable_positive_gain.dtype)
        eligible_count = eligible.sum().clamp_min(1.0)
        mature_gain = stable_positive_gain * maturity * eligible
        consensus_gain = mature_gain.sum() / eligible_count
        consensus_signal = torch.tanh(consensus_gain)

        selected_gain = stable_positive_gain[flat_indices].view_as(relative_gain)
        selected_maturity = maturity[flat_indices].view_as(relative_gain)
        local_signal = torch.tanh(selected_gain) * selected_maturity
        dynamic_uncertainty = local_signal * consensus_signal
        dynamic_uncertainty = torch.where(
            known_sharp, torch.zeros_like(dynamic_uncertainty), dynamic_uncertainty
        )
        effective_confidence = static_confidence.detach() * (1.0 - dynamic_uncertainty)
        effective_confidence = torch.where(
            known_sharp, torch.ones_like(effective_confidence), effective_confidence
        ).clamp(0.0, 1.0)
        shape = relative_gain.shape
        return effective_confidence, {
            "dynamic_uncertainty": dynamic_uncertainty,
            "effective_confidence": effective_confidence,
            "surplus_consensus": consensus_signal.expand(shape),
            "surplus_target": torch.tanh(consensus_gain).expand(shape),
        }

    def _laplacian_surplus_objective(
        self,
        prediction: Tensor,
        raw: Tensor,
        target: Tensor,
        known_sharp: Tensor,
        confidence: Tensor,
        valid_mask: Tensor | None,
        view_indices: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Use EVSSM as a one-sided floor and propagate stable view surplus.

        Extra response at RAW/EVSSM-supported edges is allowed. Extra response
        in unsupported flat regions is penalized, so isolated impulses cannot
        manufacture teacher uncertainty. Stable positive gain observed across
        optimization views raises the shared edge-amplitude floor smoothly.
        """
        reference = target.detach()
        prediction_pyramid = self._laplacian_pyramid(prediction, valid_mask)
        reference_pyramid = self._laplacian_pyramid(reference, valid_mask)
        raw_pyramid = self._laplacian_pyramid(raw.detach(), valid_mask)
        scale_weights = prediction.new_tensor((1.0, 0.5, 0.25))
        scale_weights = scale_weights / scale_weights.sum()
        robust_epsilon = 1e-3

        supports = []
        teacher_energies = []
        render_energies = []
        raw_energies = []
        for (predicted, valid), (expected, _), (raw_response, _) in zip(
            prediction_pyramid, reference_pyramid, raw_pyramid
        ):
            evidence = torch.maximum(expected.abs(), raw_response.abs())
            evidence_scale = self._masked_mean_per_view(evidence, valid)
            evidence_scale = evidence_scale[..., None, None, None].clamp_min(1e-6)
            support = (1.0 - torch.exp(-evidence / evidence_scale)).detach()
            supports.append(support)
            teacher_energies.append(
                self._masked_supported_energy(expected, support, valid)
            )
            render_energies.append(
                self._masked_supported_energy(predicted, support, valid)
            )
            raw_energies.append(
                self._masked_supported_energy(raw_response, support, valid)
            )

        def multiscale_log_gain(energies: list[Tensor]) -> Tensor:
            stacked = torch.stack([torch.log(value + 1e-8) for value in energies], -1)
            return (stacked * scale_weights.view(1, 1, -1)).sum(dim=-1)

        raw_log = multiscale_log_gain(raw_energies)
        teacher_gain = multiscale_log_gain(teacher_energies) - raw_log
        render_gain = multiscale_log_gain(render_energies) - raw_log
        relative_gain = render_gain - teacher_gain
        effective_confidence, dynamic_stats = self._update_surplus_reliability(
            relative_gain,
            confidence,
            known_sharp,
            view_indices,
        )

        # ``teacher_gain`` is a log *energy* ratio.  Convert the remaining
        # confidence-weighted teacher gain, plus demonstrated cross-view
        # surplus, into a log-amplitude target by multiplying by one half.
        # This gives uncertain teachers bounded headroom without a hand-set
        # sharpness margin, while known-sharp views still bypass this branch.
        static_headroom = (1.0 - confidence.detach()) * torch.tanh(
            F.relu(teacher_gain)
        )
        surplus_target = static_headroom + dynamic_stats["surplus_target"]
        amplitude_multiplier = torch.exp(0.5 * surplus_target)[
            ..., None, None, None
        ]
        floors = []
        overshoots = []
        artifacts = []
        for ((predicted, valid), (expected, _), (raw_response, _), support) in zip(
            prediction_pyramid,
            reference_pyramid,
            raw_pyramid,
            supports,
        ):
            desired_magnitude = expected.abs() * amplitude_multiplier
            aligned_prediction = expected.sign() * predicted
            deficit = F.relu(desired_magnitude - aligned_prediction)
            floor_error = torch.sqrt(deficit.square() + robust_epsilon**2)
            floor_error = floor_error - robust_epsilon
            floors.append(self._masked_mean_per_view(floor_error, valid))

            # Overshooting a reliable teacher is softly anchored, not banned.
            # Quadratic confidence makes this restraint disappear rapidly when
            # either static evidence or stable render surplus says EVSSM is
            # unreliable.
            overshoot = F.relu(aligned_prediction - desired_magnitude)
            overshoot_error = torch.sqrt(
                overshoot.square() + robust_epsilon**2
            ) - robust_epsilon
            overshoots.append(
                self._masked_mean_per_view(support * overshoot_error, valid)
            )

            supported_limit = (
                torch.maximum(expected.abs(), raw_response.abs())
                * amplitude_multiplier
            )
            unsupported_excess = F.relu(predicted.abs() - supported_limit)
            artifact_error = torch.sqrt(
                unsupported_excess.square() + robust_epsilon**2
            ) - robust_epsilon
            artifacts.append(
                self._masked_mean_per_view((1.0 - support) * artifact_error, valid)
            )

        floor = (torch.stack(floors, -1) * scale_weights.view(1, 1, -1)).sum(-1)
        overshoot = (
            torch.stack(overshoots, -1) * scale_weights.view(1, 1, -1)
        ).sum(-1)
        artifact = (
            torch.stack(artifacts, -1) * scale_weights.view(1, 1, -1)
        ).sum(-1)
        non_sharp = (~known_sharp).float()
        sharp_anchor_coverage = self.known_sharp_mask.float().mean()
        per_view = non_sharp * (
            effective_confidence * floor
            + effective_confidence.square() * overshoot
            + artifact
        )
        return per_view, {
            **dynamic_stats,
            "laplacian_teacher_gain": teacher_gain,
            "laplacian_render_gain": render_gain,
            "laplacian_relative_gain": relative_gain,
            "laplacian_catchup": floor,
            "laplacian_sharp_match": floor,
            "laplacian_floor": floor,
            "laplacian_overshoot": overshoot,
            "laplacian_artifact": artifact,
            "surplus_target": surplus_target,
            "sharp_anchor_coverage": sharp_anchor_coverage.expand_as(floor),
        }

    @classmethod
    def _laplacian_spatial_objective(
        cls,
        prediction: Tensor,
        raw: Tensor,
        target: Tensor,
        known_sharp: Tensor,
        confidence: Tensor,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Match robust, spatially aligned edge responses at three scales."""
        selector = known_sharp[..., None, None, None]
        reference = torch.where(selector, raw, target).detach()
        prediction_pyramid = cls._laplacian_pyramid(prediction, valid_mask)
        reference_pyramid = cls._laplacian_pyramid(reference, valid_mask)
        raw_pyramid = cls._laplacian_pyramid(raw.detach(), valid_mask)
        scale_weights = prediction.new_tensor((1.0, 0.5, 0.25))
        scale_weights = scale_weights / scale_weights.sum()
        robust_epsilon = 1e-3

        errors = []
        for (predicted, valid), (expected, _), _ in zip(
            prediction_pyramid, reference_pyramid, raw_pyramid
        ):
            residual = predicted - expected
            charbonnier = torch.sqrt(residual.square() + robust_epsilon**2)
            charbonnier = charbonnier - robust_epsilon
            errors.append(cls._masked_mean_per_view(charbonnier, valid))
        spatial_error = (
            torch.stack(errors, dim=-1) * scale_weights.view(1, 1, -1)
        ).sum(dim=-1)
        # Sharp observations already have exact RGB/SSIM supervision and no
        # deblurring gap (target == RAW). Applying this term to them duplicates
        # the reconstruction gradient and makes sharp-heavy scenes dominate the
        # edge objective. The Laplacian term is only for trusted deblurred views.
        edge_confidence = ((~known_sharp).float() * confidence).detach()
        per_view = edge_confidence * spatial_error

        predicted_finest, finest_valid = prediction_pyramid[0]
        reference_finest, _ = reference_pyramid[0]
        raw_finest, _ = raw_pyramid[0]
        predicted_energy = cls._masked_mean_per_view(
            predicted_finest.square(), finest_valid
        )
        reference_energy = cls._masked_mean_per_view(
            reference_finest.square(), finest_valid
        )
        raw_energy = cls._masked_mean_per_view(raw_finest.square(), finest_valid)
        teacher_gain = torch.log(reference_energy + 1e-8) - torch.log(
            raw_energy + 1e-8
        )
        render_gain = torch.log(predicted_energy + 1e-8) - torch.log(
            raw_energy + 1e-8
        )
        return per_view, {
            "laplacian_teacher_gain": teacher_gain,
            "laplacian_render_gain": render_gain,
            "laplacian_relative_gain": render_gain - teacher_gain,
            "laplacian_catchup": spatial_error,
            "laplacian_sharp_match": spatial_error,
        }

    def _laplacian_objective(
        self,
        prediction: Tensor,
        raw: Tensor,
        target: Tensor,
        known_sharp: Tensor,
        confidence: Tensor,
        valid_mask: Tensor | None,
        view_indices: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.cfg.laplacian_loss_weight == 0.0 or (
            self.cfg.laplacian_loss_mode in {"spatial", "surplus"}
            and bool(known_sharp.all())
        ):
            zero = prediction.new_zeros(prediction.shape[:2])
            return zero, {
                "laplacian_teacher_gain": zero,
                "laplacian_render_gain": zero,
                "laplacian_relative_gain": zero,
                "laplacian_catchup": zero,
                "laplacian_sharp_match": zero,
                "laplacian_floor": zero,
                "laplacian_overshoot": zero,
                "laplacian_artifact": zero,
                "dynamic_uncertainty": zero,
                "effective_confidence": confidence.detach(),
                "surplus_consensus": zero,
                "surplus_target": zero,
                "sharp_anchor_coverage": self.known_sharp_mask.float()
                .mean()
                .expand_as(zero),
            }
        if self.cfg.laplacian_loss_mode == "surplus":
            if view_indices is None:
                raise ValueError("surplus mode requires view_indices")
            return self._laplacian_surplus_objective(
                prediction,
                raw,
                target,
                known_sharp,
                confidence,
                valid_mask,
                view_indices,
            )
        if self.cfg.laplacian_loss_mode == "energy":
            per_view, stats = self._laplacian_energy_objective(
                prediction, raw, target, known_sharp, valid_mask
            )
        else:
            per_view, stats = self._laplacian_spatial_objective(
                prediction,
                raw,
                target,
                known_sharp,
                confidence,
                valid_mask,
            )
        zero = per_view.new_zeros(per_view.shape)
        stats.update(
            {
                "laplacian_floor": stats["laplacian_catchup"],
                "laplacian_overshoot": zero,
                "laplacian_artifact": zero,
                "dynamic_uncertainty": zero,
                "effective_confidence": confidence.detach(),
                "surplus_consensus": zero,
                "surplus_target": zero,
                "sharp_anchor_coverage": self.known_sharp_mask.float()
                .mean()
                .expand_as(zero),
            }
        )
        return per_view, stats

    def _raw_gate(self) -> float:
        progress = (self.step + 1) / self.num_steps
        value = (progress - self.cfg.raw_ramp_start) / (
            self.cfg.raw_ramp_end - self.cfg.raw_ramp_start
        )
        value = min(1.0, max(0.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _supervision_weights(self, known_sharp: Tensor) -> Tensor:
        """Apply relative w10 while preserving the optimizer's loss scale."""
        if self.cfg.sharp_weight_in_sampler:
            return torch.ones_like(known_sharp, dtype=torch.float32)
        base = known_sharp.float()
        weights = torch.where(
            known_sharp,
            base.new_full(base.shape, self.cfg.sharp_supervision_weight),
            base.new_ones(base.shape),
        )
        return weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-8)

    def compute_loss(
        self,
        *,
        context,
        output_renderer,
        start: int,
        stop: int,
        reduction: str,
        with_ssim: bool,
        fallback_loss: Callable,
    ) -> Tensor:
        raw_all = context.get("raw_image")
        confidence_all = context.get("target_confidence")
        sharp_all = context.get("known_sharp")
        if raw_all is None or confidence_all is None or sharp_all is None:
            return fallback_loss(
                context["image"][:, start:stop],
                output_renderer,
                reduction=reduction,
                with_ssim=with_ssim,
            )

        target = context["image"][:, start:stop]
        raw = raw_all[:, start:stop]
        valid_all = context.get("valid_mask")
        valid = None if valid_all is None else valid_all[:, start:stop]
        confidence = confidence_all[:, start:stop].clamp(0.0, 1.0)
        known_sharp = sharp_all[:, start:stop].bool()
        direct_all = context.get("direct_supervision")
        direct_supervision = (
            known_sharp
            if direct_all is None
            else direct_all[:, start:stop].bool()
        )
        supervision_confidence_all = context.get("supervision_confidence")
        supervision_confidence = (
            torch.ones_like(confidence)
            if supervision_confidence_all is None
            else supervision_confidence_all[:, start:stop].clamp(0.0, 1.0)
        )
        view_indices = context["index"][:, start:stop]

        # A blur formation model is undefined for an authoritative all-sharp
        # batch. Bypass it entirely instead of letting its mask/kernel
        # regularizers leak gradients into otherwise direct supervision.
        if bool(direct_supervision.all()):
            per_view = self._per_view_loss(
                output_renderer.color, target, with_ssim, valid
            )
            laplacian_per_view, laplacian_stats = self._laplacian_objective(
                output_renderer.color,
                raw,
                target,
                known_sharp,
                confidence,
                valid,
                view_indices,
            )
            per_view = supervision_confidence * per_view
            laplacian_per_view = supervision_confidence * laplacian_per_view
            if reduction == "mean":
                reconstruction = per_view.mean()
                laplacian_loss = laplacian_per_view.mean()
            elif reduction == "sum":
                pixels = target.shape[2] * target.shape[3] * target.shape[4]
                reconstruction = per_view.sum() * pixels
                laplacian_loss = laplacian_per_view.sum() * pixels
            elif reduction == "mean_pixels_sum_views":
                reconstruction = per_view.sum(dim=1).mean()
                laplacian_loss = laplacian_per_view.sum(dim=1).mean()
            else:
                raise ValueError(f"Unknown reduction: {reduction!r}")
            loss = reconstruction + self.cfg.laplacian_loss_weight * laplacian_loss
            self.last_diagnostics = {
                "step": float(self.step),
                "raw_gate": 0.0,
                "loss": float(loss.detach()),
                "direct_loss": float(reconstruction.detach()),
                "raw_loss": 0.0,
                "direct_weight": 1.0,
                "raw_weight": 0.0,
                "sharp_fraction": float(known_sharp.float().detach().mean()),
                "direct_supervision_fraction": float(
                    direct_supervision.float().detach().mean()
                ),
                "sharp_relative_weight": float(self.cfg.sharp_supervision_weight),
                "supervision_weight_mean": float(
                    supervision_confidence.detach().mean()
                ),
                "supervision_confidence_mean": float(
                    supervision_confidence.detach().mean()
                ),
                "kernel_entropy": 0.0,
                "mask_mean": 0.0,
                "mask_loss": 0.0,
                "center_loss": 0.0,
                "bpn_bypassed": 1.0,
                "laplacian_loss": float(laplacian_loss.detach()),
                "laplacian_teacher_gain": float(
                    laplacian_stats["laplacian_teacher_gain"].detach().mean()
                ),
                "laplacian_render_gain": float(
                    laplacian_stats["laplacian_render_gain"].detach().mean()
                ),
                "laplacian_relative_gain": float(
                    laplacian_stats["laplacian_relative_gain"].detach().mean()
                ),
                "laplacian_floor": float(
                    laplacian_stats["laplacian_floor"].detach().mean()
                ),
                "laplacian_overshoot": float(
                    laplacian_stats["laplacian_overshoot"].detach().mean()
                ),
                "laplacian_artifact": float(
                    laplacian_stats["laplacian_artifact"].detach().mean()
                ),
                "static_confidence": float(confidence.detach().mean()),
                "effective_confidence": float(
                    laplacian_stats["effective_confidence"].detach().mean()
                ),
                "dynamic_uncertainty": float(
                    laplacian_stats["dynamic_uncertainty"].detach().mean()
                ),
                "surplus_consensus": float(
                    laplacian_stats["surplus_consensus"].detach().mean()
                ),
                "surplus_target": float(
                    laplacian_stats["surplus_target"].detach().mean()
                ),
                "sharp_anchor_coverage": float(
                    laplacian_stats["sharp_anchor_coverage"].detach().mean()
                ),
            }
            return loss

        formed, bpn_stats = self.bpn(
            output_renderer.color,
            output_renderer.depth,
            raw,
            target,
            view_indices,
        )

        teacher_prediction = (
            bpn_stats["teacher_formed"]
            if self.cfg.coupled_dual_bpn
            else output_renderer.color
        )
        if self.cfg.coupled_dual_bpn:
            # Authoritative sharp views are identity observations. The weak
            # teacher kernel is only allowed to explain residual blur in the
            # deblurred, non-authoritative EVSSM targets.
            teacher_prediction = torch.where(
                direct_supervision[..., None, None, None],
                output_renderer.color,
                teacher_prediction,
            )
        direct_loss = self._per_view_loss(
            teacher_prediction, target, with_ssim, valid
        )
        raw_loss = self._per_view_loss(formed, raw, with_ssim, valid)
        laplacian_per_view, laplacian_stats = self._laplacian_objective(
            output_renderer.color,
            raw,
            target,
            known_sharp,
            confidence,
            valid,
            view_indices,
        )
        effective_confidence = laplacian_stats["effective_confidence"]
        gate = self._raw_gate()
        raw_weight = (
            (~direct_supervision).float()
            * (1.0 - effective_confidence)
            * gate
        )
        direct_weight = 1.0 - raw_weight
        supervision_weight = (
            self._supervision_weights(known_sharp) * supervision_confidence
        )
        per_view = supervision_weight * (
            direct_weight * direct_loss + raw_weight * raw_loss
        )
        laplacian_per_view = supervision_confidence * laplacian_per_view

        if reduction == "mean":
            reconstruction = per_view.mean()
            laplacian_loss = laplacian_per_view.mean()
        elif reduction == "sum":
            pixels = target.shape[2] * target.shape[3] * target.shape[4]
            reconstruction = per_view.sum() * pixels
            laplacian_loss = laplacian_per_view.sum() * pixels
        elif reduction == "mean_pixels_sum_views":
            reconstruction = per_view.sum(dim=1).mean()
            laplacian_loss = laplacian_per_view.sum(dim=1).mean()
        else:
            raise ValueError(f"Unknown reduction: {reduction!r}")

        regularization = (
            self.cfg.center_regularization * bpn_stats["center_loss"]
            + self.cfg.mask_regularization * bpn_stats["mask_loss"]
            + self.cfg.mask_tv_regularization * bpn_stats["mask_tv"]
        )
        loss = (
            reconstruction
            + self.cfg.laplacian_loss_weight * laplacian_loss
            + gate * regularization
        )
        self.last_diagnostics = {
            "step": float(self.step),
            "raw_gate": float(gate),
            "loss": float(loss.detach()),
            "direct_loss": float(direct_loss.detach().mean()),
            "raw_loss": float(raw_loss.detach().mean()),
            "direct_weight": float(direct_weight.detach().mean()),
            "raw_weight": float(raw_weight.detach().mean()),
            "sharp_fraction": float(known_sharp.float().detach().mean()),
            "direct_supervision_fraction": float(
                direct_supervision.float().detach().mean()
            ),
            "sharp_relative_weight": float(self.cfg.sharp_supervision_weight),
            "supervision_weight_mean": float(supervision_weight.detach().mean()),
            "supervision_confidence_mean": float(
                supervision_confidence.detach().mean()
            ),
            "kernel_entropy": float(bpn_stats["kernel_entropy"].detach()),
            "mask_mean": float(bpn_stats["mask"].detach().mean()),
            "mask_loss": float(bpn_stats["mask_loss"].detach()),
            "center_loss": float(bpn_stats["center_loss"].detach()),
            "bpn_bypassed": 0.0,
            "laplacian_loss": float(laplacian_loss.detach()),
            "laplacian_teacher_gain": float(
                laplacian_stats["laplacian_teacher_gain"].detach().mean()
            ),
            "laplacian_render_gain": float(
                laplacian_stats["laplacian_render_gain"].detach().mean()
            ),
            "laplacian_relative_gain": float(
                laplacian_stats["laplacian_relative_gain"].detach().mean()
            ),
            "laplacian_floor": float(
                laplacian_stats["laplacian_floor"].detach().mean()
            ),
            "laplacian_overshoot": float(
                laplacian_stats["laplacian_overshoot"].detach().mean()
            ),
            "laplacian_artifact": float(
                laplacian_stats["laplacian_artifact"].detach().mean()
            ),
            "static_confidence": float(confidence.detach().mean()),
            "effective_confidence": float(effective_confidence.detach().mean()),
            "dynamic_uncertainty": float(
                laplacian_stats["dynamic_uncertainty"].detach().mean()
            ),
            "surplus_consensus": float(
                laplacian_stats["surplus_consensus"].detach().mean()
            ),
            "surplus_target": float(
                laplacian_stats["surplus_target"].detach().mean()
            ),
            "sharp_anchor_coverage": float(
                laplacian_stats["sharp_anchor_coverage"].detach().mean()
            ),
        }
        return loss

    def export_config(self) -> dict:
        return asdict(self.cfg)

    def optimizer_state_dict(self) -> dict | None:
        return None if self._optimizer is None else self._optimizer.state_dict()

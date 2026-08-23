"""LeGS per-Gaussian PPO densification for the Learn2Splat runtime.

The policy/state/reward implementation follows the official LeGS release at
commit 8eb120b1f0c0fe0727e0440f4e372b412f275572. In particular, this module uses
the release's 11-D state, keep/clone/split actor, separate opacity-prune actor,
parent-child delayed credit assignment, normalized leave-one-out sensitivity
reward, and PPO update. The sensitivity itself comes from the official LeGS
FastGS CUDA extension; no screen-gradient or global-probe proxy is substituted.

The host runtime uses the release's 500/100/15000 structural schedule and no
global primitive cap. Both facts are recorded in experiment receipts; a capped
or rescaled run is a separate adapted ablation and is not exposed as exact
``--adc legs``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from optgs.model.types import Gaussians
from optgs.scene_trainer.adc.base import (
    _clone_objects,
    _prune_objects,
    _split_objects,
)
from optgs.scene_trainer.adc.fastgs import (
    FastGSStrategyState,
    reset_fastgs_state,
)
from optgs.scene_trainer.adc.legs_config import LeGSStrategyCfg
from optgs.scene_trainer.adc.vanilla import cloning, prune, splitting
from optgs.scene_trainer.gaussian_module import GaussiansModule
from optgs.scene_trainer.optimizer.optimizer_utils import (
    calc_input_gradients,
    squeeze_grad_dict,
)


class MLPStateEncoder(nn.Module):
    """The gated residual state encoder from the official LeGS release."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64) -> None:
        super().__init__()
        expand_dim = hidden_dim * 4
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "gate": nn.Linear(hidden_dim, expand_dim),
                        "up": nn.Linear(hidden_dim, expand_dim),
                        "down": nn.Linear(expand_dim, hidden_dim),
                    }
                )
                for _ in range(3)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: Tensor) -> Tensor:
        encoded = self.input_proj(state)
        for layer in self.layers:
            residual = encoded
            normalized = layer["norm"](encoded)
            gate = F.silu(layer["gate"](normalized))
            encoded = layer["down"](gate * layer["up"](normalized)) + residual
        return self.output_norm(encoded)


class PPOActor(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, encoded: Tensor) -> Tensor:
        return F.softmax(self.mlp(encoded), dim=-1)


class PPOPruneEstimator(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, encoded: Tensor) -> Tensor:
        return torch.sigmoid(self.mlp(encoded))


class ZeroInitializedBlurAdapter(nn.Module):
    """Inject global blur state without perturbing the exact LeGS prior."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_dim, input_dim))

    def forward(self, state: Tensor) -> Tensor:
        # A bias would learn even while the curriculum feeds an all-zero blur
        # state, silently changing exact LeGS during the intended warmup.
        return F.linear(state, self.weight)


def _scatter_mean(source: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """torch_scatter.scatter_mean equivalent without an extra dependency."""
    result = source.new_zeros((dim_size, *source.shape[1:]))
    counts = source.new_zeros(dim_size)
    result.index_add_(0, index, source)
    counts.index_add_(0, index, torch.ones_like(index, dtype=source.dtype))
    shape = (dim_size,) + (1,) * (source.ndim - 1)
    return result / counts.clamp_min(1).reshape(shape)


class LeGSPPOController:
    def __init__(self, cfg: LeGSStrategyCfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.base_state_dim = (
            cfg.state_dim - cfg.blur_feature_dim
            if cfg.blur_conditioned
            else cfg.state_dim
        )
        if self.base_state_dim != 11:
            raise ValueError(
                "LeGS requires the official 11-D local state before blur conditioning"
            )
        self.encoder = MLPStateEncoder(self.base_state_dim, cfg.hidden_dim).to(device)
        self.actor = PPOActor(cfg.hidden_dim).to(device)
        self.prune_estimator = PPOPruneEstimator(cfg.hidden_dim).to(device)
        self.blur_adapter = (
            ZeroInitializedBlurAdapter(cfg.blur_feature_dim, cfg.hidden_dim).to(device)
            if cfg.blur_conditioned
            else None
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=cfg.actor_lr_init, weight_decay=1e-4
        )
        encoder_parameters = list(self.encoder.parameters())
        if self.blur_adapter is not None:
            encoder_parameters.extend(self.blur_adapter.parameters())
        self.encoder_optimizer = torch.optim.AdamW(
            encoder_parameters,
            lr=cfg.state_encoder_lr_init,
            weight_decay=1e-4,
        )
        self.prune_optimizer = torch.optim.AdamW(
            self.prune_estimator.parameters(),
            lr=cfg.prune_lr_init,
            weight_decay=1e-4,
        )
        self.transitions: list[dict[str, Any]] = []
        self.update_count = 0
        self.last_policy_loss = 0.0
        self.last_entropy = 0.0

    def _encode(self, states: Tensor) -> Tensor:
        encoded = self.encoder(states[..., : self.base_state_dim])
        if self.blur_adapter is not None:
            blur_state = states[..., self.base_state_dim :]
            if blur_state.shape[-1] != self.cfg.blur_feature_dim:
                raise RuntimeError(
                    f"blur adapter expected {self.cfg.blur_feature_dim} features, "
                    f"got {blur_state.shape[-1]}"
                )
            encoded = encoded + self.blur_adapter(blur_state)
        return encoded

    def _encode_chunks(self, states: Tensor, requires_grad: bool) -> Tensor:
        chunks = []
        size = max(1, self.cfg.ppo_chunk_size)
        for start in range(0, states.shape[0], size):
            chunk = states[start : start + size].to(self.device)
            if requires_grad:
                chunk = chunk.detach().requires_grad_(True)
            chunks.append(self._encode(chunk))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def act(
        self,
        states: Tensor,
        valid_mask: Tensor,
        prune_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        actions = torch.zeros(states.shape[0], dtype=torch.long, device=self.device)
        confidence = torch.zeros(states.shape[0], dtype=states.dtype, device=self.device)
        indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            return actions, confidence

        encoded = self._encode_chunks(states[indices], requires_grad=False)
        local_prune = prune_mask[indices]
        local_actions = torch.zeros(indices.numel(), dtype=torch.long, device=self.device)
        local_confidence = torch.ones(indices.numel(), dtype=states.dtype, device=self.device)

        non_prune = ~local_prune
        if non_prune.any():
            probs = self.actor(encoded[non_prune])
            sampled = torch.distributions.Categorical(probs).sample()
            local_actions[non_prune] = sampled
            local_confidence[non_prune] = probs.gather(
                1, sampled[:, None]
            ).squeeze(1)
        if local_prune.any():
            delete_prob = self.prune_estimator(encoded[local_prune]).squeeze(1)
            delete = torch.bernoulli(delete_prob).bool()
            local_actions[local_prune] = torch.where(
                delete,
                torch.full_like(delete_prob, 3, dtype=torch.long),
                torch.zeros_like(delete_prob, dtype=torch.long),
            )
            local_confidence[local_prune] = torch.where(
                delete, delete_prob, 1.0 - delete_prob
            )

        actions[indices] = local_actions
        confidence[indices] = local_confidence
        return actions, confidence

    def store_transition(
        self,
        states: Tensor,
        actions: Tensor,
        parent_mapping: Tensor | None,
        valid_mask: Tensor,
        prune_mask: Tensor,
    ) -> int:
        self.transitions.append(
            {
                "states": states.detach().cpu(),
                "actions": actions.detach().cpu(),
                "reward": torch.zeros(actions.shape[0], 1),
                "value": None,
                "parent_mapping": (
                    None if parent_mapping is None else parent_mapping.detach().cpu()
                ),
                "valid_mask": valid_mask.detach().cpu(),
                "prune_mask": prune_mask.detach().cpu(),
            }
        )
        return len(self.transitions) - 1

    def set_reward(self, index: int, reward: Tensor, valid_mask: Tensor) -> None:
        transition = self.transitions[index]
        actions = transition["actions"].to(reward.device)
        prune_mask = transition["prune_mask"].to(reward.device)
        value = torch.zeros_like(reward)
        prune_keep = valid_mask & prune_mask & (actions == 0)
        normal_keep = valid_mask & (~prune_mask) & (actions == 0)
        if prune_keep.any():
            value[valid_mask & prune_mask] = reward[prune_keep].mean()
        if normal_keep.any():
            value[valid_mask & (~prune_mask)] = reward[normal_keep].mean()
        transition["reward"] = reward[:, None].detach().cpu()
        transition["value"] = value[:, None].detach().cpu()
        transition["valid_mask"] = valid_mask.detach().cpu()

    def ready(self) -> bool:
        return (
            len(self.transitions) >= self.cfg.rollout_batch_size
            and all(item["value"] is not None for item in self.transitions)
        )

    def _selected_log_prob(
        self,
        encoded: Tensor,
        actions: Tensor,
        prune_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        log_prob = encoded.new_zeros((encoded.shape[0], 1))
        entropy = encoded.new_zeros((encoded.shape[0], 1))
        non_prune = ~prune_mask
        if non_prune.any():
            probs = self.actor(encoded[non_prune])
            selected = probs.gather(1, actions[non_prune, None])
            log_prob[non_prune] = torch.log(selected.clamp_min(1e-8))
            entropy[non_prune] = -(
                probs * torch.log(probs.clamp_min(1e-8))
            ).sum(dim=1, keepdim=True)
        if prune_mask.any():
            probability = self.prune_estimator(encoded[prune_mask])
            selected = torch.where(
                actions[prune_mask, None] == 3,
                probability,
                1.0 - probability,
            )
            log_prob[prune_mask] = torch.log(selected.clamp_min(1e-8))
            binary = torch.cat([probability, 1.0 - probability], dim=1)
            entropy[prune_mask] = -(
                binary * torch.log(binary.clamp_min(1e-8))
            ).sum(dim=1, keepdim=True)
        return log_prob, entropy

    def _set_learning_rates(self, step: int) -> None:
        span = max(1, self.cfg.refine_stop_iter - self.cfg.refine_start_iter)
        progress = min(1.0, max(0.0, (step - self.cfg.refine_start_iter) / span))

        def interpolate(initial: float, final: float) -> float:
            if initial <= 0 or final <= 0:
                return final
            return initial * ((final / initial) ** progress)

        for group in self.actor_optimizer.param_groups:
            group["lr"] = interpolate(
                self.cfg.actor_lr_init, self.cfg.actor_lr_final
            )
        for group in self.encoder_optimizer.param_groups:
            group["lr"] = interpolate(
                self.cfg.state_encoder_lr_init,
                self.cfg.state_encoder_lr_final,
            )
        for group in self.prune_optimizer.param_groups:
            group["lr"] = interpolate(
                self.cfg.prune_lr_init, self.cfg.prune_lr_final
            )

    def learn(self, step: int) -> None:
        transitions = self.transitions[: self.cfg.rollout_batch_size]
        old_log_probs = []
        values = [item["value"] for item in transitions]

        with torch.no_grad():
            for item in transitions:
                valid = item["valid_mask"]
                encoded = self._encode_chunks(
                    item["states"][valid], requires_grad=False
                )
                log_prob, _ = self._selected_log_prob(
                    encoded,
                    item["actions"][valid].to(self.device),
                    item["prune_mask"][valid].to(self.device),
                )
                old_log_probs.append(log_prob.cpu())

            advantages_reversed = []
            last_gae: Tensor | int = 0
            for index in reversed(range(len(transitions))):
                reward = transitions[index]["reward"].to(self.device)
                value = values[index].to(self.device)
                if index < len(transitions) - 1:
                    next_value = values[index + 1].to(self.device)
                    mapping = transitions[index + 1]["parent_mapping"]
                    if mapping is None:
                        raise RuntimeError("LeGS rollout lost its parent-child mapping")
                    mapping = mapping.to(self.device)
                    next_value = _scatter_mean(next_value, mapping, value.shape[0])
                    next_gae = _scatter_mean(last_gae, mapping, value.shape[0])
                    delta = reward + self.cfg.gamma * next_value - value
                    last_gae = (
                        delta
                        + self.cfg.gamma * self.cfg.gae_lambda * next_gae
                    )
                else:
                    last_gae = reward - value
                advantages_reversed.append(last_gae.cpu())
            advantages = advantages_reversed[::-1]

        filtered = []
        for item, advantage, old_log_prob in zip(
            transitions, advantages, old_log_probs
        ):
            valid = item["valid_mask"]
            filtered.append(
                (
                    item["states"][valid],
                    item["actions"][valid],
                    item["prune_mask"][valid],
                    advantage[valid].detach(),
                    old_log_prob,
                )
            )
        if not any(states.numel() for states, *_ in filtered):
            self.transitions = self.transitions[self.cfg.rollout_batch_size :]
            return

        self._set_learning_rates(step)
        n_rollout = len(filtered)
        policy_loss_sum = 0.0
        entropy_sum = 0.0
        for _ in range(self.cfg.ppo_epochs):
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.prune_optimizer.zero_grad(set_to_none=True)
            for states, actions, prune_mask, advantage, old_log_prob in filtered:
                points_in_rollout = states.shape[0]
                for start in range(0, states.shape[0], self.cfg.ppo_chunk_size):
                    stop = min(start + self.cfg.ppo_chunk_size, states.shape[0])
                    with torch.amp.autocast(
                        "cuda", enabled=self.cfg.use_mixed_precision
                    ):
                        encoded = self._encode(
                            states[start:stop]
                            .to(self.device)
                            .detach()
                            .requires_grad_(True)
                        )
                        action_chunk = actions[start:stop].to(self.device)
                        prune_chunk = prune_mask[start:stop].to(self.device)
                        advantage_chunk = advantage[start:stop].to(self.device)
                        old_chunk = old_log_prob[start:stop].to(self.device)
                        log_prob, entropy = self._selected_log_prob(
                            encoded, action_chunk, prune_chunk
                        )
                        ratio = torch.exp(log_prob - old_chunk)
                        loss_unclipped = -advantage_chunk * ratio
                        loss_clipped = -advantage_chunk * torch.clamp(
                            ratio,
                            1.0 - self.cfg.policy_clip,
                            1.0 + self.cfg.policy_clip,
                        )
                        policy_loss = torch.maximum(
                            loss_unclipped, loss_clipped
                        ).mean()
                    # Official LeGS gives each rollout equal mass, then weights
                    # chunks by their fraction of that rollout.
                    weight = (stop - start) / points_in_rollout / n_rollout
                    (policy_loss * weight).backward()
                    policy_loss_sum += float(policy_loss.detach()) * weight
                    entropy_sum += float(entropy.mean().detach()) * weight
            self.actor_optimizer.step()
            self.encoder_optimizer.step()
            self.prune_optimizer.step()

        self.update_count += 1
        self.last_policy_loss = policy_loss_sum / self.cfg.ppo_epochs
        self.last_entropy = entropy_sum / self.cfg.ppo_epochs
        self.transitions = self.transitions[self.cfg.rollout_batch_size :]


@dataclass
class LeGSStrategyState(FastGSStrategyState):
    controller: LeGSPPOController | None = None
    parent_mapping: Tensor | None = None
    pending_transition_index: int = -1
    pending_reward_step: int = -1
    pending_pre_metric: Tensor | None = None
    pending_pre_visible: Tensor | None = None
    pending_views: dict[str, Tensor] | None = None
    pending_blur_views: dict[str, Tensor] | None = None
    last_event_step: int = -1
    last_reward_step: int = -1
    last_valid_count: int = 0
    last_clone_count: int = 0
    last_split_count: int = 0
    last_prune_count: int = 0
    last_reward_mean: float = 0.0
    last_reward_std: float = 0.0
    last_cap_truncation: int = 0
    last_event_kind: str = "none"
    last_sampled_view_indices: list[int] = field(default_factory=list)
    last_blur_probe_view_indices: list[int] = field(default_factory=list)
    opacity_reset_count: int = 0
    final_prune_count: int = 0
    event_log: list[dict[str, Any]] = field(default_factory=list)
    initial_gaussian_count: int = 0
    blur_feature_history: list[list[float]] = field(default_factory=list)
    blur_quality_delta_history: list[list[float]] = field(default_factory=list)
    fixed_blur_probe_positions: list[int] = field(default_factory=list)
    pending_pre_blur_observation: dict[str, float | bool] | None = None
    last_blur_quality_reward: float = 0.0
    last_blur_psnr_delta: float = 0.0
    last_blur_raw_psnr_delta: float = 0.0
    last_blur_surplus_delta: float = 0.0
    last_blur_structural_fraction: float = 0.0
    last_blur_capacity_cost: float = 0.0
    last_blur_feature_raw: list[float] = field(default_factory=list)
    last_blur_feature_normalized: list[float] = field(default_factory=list)
    last_blur_condition_scale: float = 0.0
    last_blur_action_support_mean: float = 0.0
    last_blur_birth_penalty_gate_mean: float = 0.0
    last_blur_net_action_direction: float = 0.0

    def external_pruning(self, valid_points_mask: Tensor) -> None:
        super().external_pruning(valid_points_mask)
        if (
            self.parent_mapping is not None
            and self.parent_mapping.shape[0] == valid_points_mask.shape[0]
        ):
            self.parent_mapping = self.parent_mapping[valid_points_mask]


def transfer_legs_runtime_state(
    source: FastGSStrategyState | None,
    target: FastGSStrategyState | None,
) -> None:
    """Carry the per-scene policy and delayed rollout across optimizer phases."""
    if not isinstance(source, LeGSStrategyState) or not isinstance(
        target, LeGSStrategyState
    ):
        return
    for name in (
        "controller",
        "parent_mapping",
        "pending_transition_index",
        "pending_reward_step",
        "pending_pre_metric",
        "pending_pre_visible",
        "pending_views",
        "pending_blur_views",
        "last_event_step",
        "last_reward_step",
        "last_valid_count",
        "last_clone_count",
        "last_split_count",
        "last_prune_count",
        "last_reward_mean",
        "last_reward_std",
        "last_cap_truncation",
        "last_event_kind",
        "last_sampled_view_indices",
        "last_blur_probe_view_indices",
        "opacity_reset_count",
        "final_prune_count",
        "event_log",
        "initial_gaussian_count",
        "blur_feature_history",
        "blur_quality_delta_history",
        "fixed_blur_probe_positions",
        "pending_pre_blur_observation",
        "last_blur_quality_reward",
        "last_blur_psnr_delta",
        "last_blur_raw_psnr_delta",
        "last_blur_surplus_delta",
        "last_blur_structural_fraction",
        "last_blur_capacity_cost",
        "last_blur_feature_raw",
        "last_blur_feature_normalized",
        "last_blur_condition_scale",
        "last_blur_action_support_mean",
        "last_blur_birth_penalty_gate_mean",
        "last_blur_net_action_direction",
    ):
        setattr(target, name, getattr(source, name))


def _normalize_features(features: Tensor) -> Tensor:
    correction = 1 if features.shape[0] > 1 else 0
    return (features - features.mean(0, keepdim=True)) / (
        features.std(0, keepdim=True, correction=correction) + 1e-6
    )


def _raw_parameter_clones(gaussians: Gaussians) -> tuple[Tensor, ...]:
    means = gaussians.means.detach().clone().requires_grad_(True)
    rotations = gaussians.rotations_unnorm.detach().clone().requires_grad_(True)
    harmonics = gaussians.harmonics.detach().clone().requires_grad_(True)
    if gaussians.stores_activated:
        scales = (
            gaussians.scales.detach().clamp_min(1e-12).log().clone()
            .requires_grad_(True)
        )
        opacities = (
            torch.logit(
                gaussians.opacities.detach().clamp(1e-6, 1.0 - 1e-6)
            )
            .clone()
            .requires_grad_(True)
        )
    else:
        scales = gaussians.scales.detach().clone().requires_grad_(True)
        opacities = gaussians.opacities.detach().clone().requires_grad_(True)
    return means, scales, rotations, opacities, harmonics


def _activated_gaussians(raw: tuple[Tensor, ...]) -> Gaussians:
    means, scales, rotations, opacities, harmonics = raw
    return Gaussians(
        means=means,
        covariances=None,
        harmonics=harmonics,
        opacities=torch.sigmoid(opacities),
        scales=torch.exp(scales),
        rotations=F.normalize(rotations, dim=-1),
        rotations_unnorm=rotations,
        stores_activated=True,
    )


def _sample_official_views(
    cfg: LeGSStrategyCfg,
    context: dict[str, Tensor],
) -> tuple[dict[str, Tensor], list[int]]:
    """Sample the ten policy/reward cameras used by the LeGS release."""
    total = int(context["image"].shape[1])
    if total < cfg.state_view_count:
        raise RuntimeError(
            "exact LeGS requires at least "
            f"{cfg.state_view_count} training cameras, got {total}"
        )
    selected = random.sample(range(total), cfg.state_view_count)
    index = torch.tensor(selected, device=context["image"].device)
    views = {
        key: context[key].index_select(1, index).detach()
        for key in ("image", "extrinsics", "intrinsics", "near", "far")
    }
    for key in (
        "raw_image",
        "target_confidence",
        "known_sharp",
        "direct_supervision",
        "valid_mask",
    ):
        if key in context and context[key] is not None:
            views[key] = context[key].index_select(1, index).detach()
    if "index" in context and context["index"] is not None:
        views["index"] = context["index"].index_select(1, index).detach()
        source = views["index"].reshape(-1)
        sampled = [int(value) for value in source.detach().cpu()]
    else:
        sampled = selected
    return views, sampled


def _fixed_blur_probe_views(
    cfg: LeGSStrategyCfg,
    context: dict[str, Tensor],
    adc_state: LeGSStrategyState,
) -> tuple[dict[str, Tensor], list[int]]:
    """Select one coverage-oriented training probe set for the whole scene."""
    total = int(context["image"].shape[1])
    if not adc_state.fixed_blur_probe_positions:
        probe_mask = context.get("policy_probe")
        if probe_mask is not None:
            positions = torch.nonzero(
                probe_mask.reshape(-1).bool(), as_tuple=False
            ).flatten()
        else:
            count = min(cfg.state_view_count, total)
            positions = torch.linspace(
                0, total - 1, steps=count, device=context["image"].device
            ).round().long()
        if positions.numel() == 0:
            raise RuntimeError("legs_blur requires at least one training probe view")
        adc_state.fixed_blur_probe_positions = [
            int(value) for value in positions.detach().cpu()
        ]

    selected = adc_state.fixed_blur_probe_positions
    if min(selected) < 0 or max(selected) >= total:
        raise RuntimeError("legs_blur probe positions changed across optimizer phases")
    index = torch.tensor(selected, device=context["image"].device)
    views = {
        key: context[key].index_select(1, index).detach()
        for key in ("image", "extrinsics", "intrinsics", "near", "far")
    }
    for key in (
        "raw_image",
        "target_confidence",
        "known_sharp",
        "direct_supervision",
        "valid_mask",
    ):
        if key in context and context[key] is not None:
            views[key] = context[key].index_select(1, index).detach()
    if "index" in context and context["index"] is not None:
        views["index"] = context["index"].index_select(1, index).detach()
        sampled = [int(value) for value in views["index"].reshape(-1).cpu()]
    else:
        sampled = list(selected)
    return views, sampled


def _metric_score(
    gaussians: Gaussians,
    renderer,
    views: dict[str, Tensor],
    *,
    clamp: bool,
) -> tuple[Tensor, Tensor]:
    sensitivity = getattr(renderer, "render_legs_sensitivity", None)
    if sensitivity is None:
        raise RuntimeError(
            "exact LeGS requires the FastGS decoder with the official LeGS "
            "per-Gaussian sensitivity extension"
        )
    image_shape = tuple(views["image"].shape[-2:])
    metric, weight = sensitivity(
        gaussians,
        views["extrinsics"],
        views["intrinsics"],
        views["near"],
        views["far"],
        image_shape,
        views["image"],
    )
    score = metric.sum(dim=0) / max(1, metric.shape[0])
    score = score.sign() * torch.log1p(score.abs())
    if clamp:
        score = score.clamp(-6.0, 6.0)
    visible = weight.sum(dim=0) > 0
    return score, visible


def build_legs_state(
    cfg: LeGSStrategyCfg,
    gaussians: Gaussians,
    renderer,
    context: dict[str, Tensor],
    objective=None,
    step: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], list[int]]:
    """Build the official 11-D gradient+sensitivity state."""
    views, sampled = _sample_official_views(cfg, context)
    raw = _raw_parameter_clones(gaussians)
    direct = views.get("direct_supervision")
    known_sharp = views.get("known_sharp")
    all_direct = bool(direct.all()) if direct is not None else (
        bool(known_sharp.all()) if known_sharp is not None else False
    )
    local_objective = (
        objective
        if cfg.blur_conditioned
        and cfg.local_objective_conditioned
        and not all_direct
        else None
    )
    if cfg.local_objective_conditioned and objective is None:
        raise RuntimeError(
            "objective-conditioned LeGS local state requires BlurAwareObjective"
        )
    _, gradients, _ = calc_input_gradients(
        views,
        *raw,
        renderer,
        need_2d_grads=False,
        chunk_size=-1,
        any_adc=False,
        input_objective=local_objective,
        optimize_input_objective=False,
        step=step,
        clamp_images=True,
    )
    gradients = squeeze_grad_dict(gradients)
    gradient_features = torch.cat(
        [
            gradients["means"],
            gradients["scales"],
            gradients["opacities"][:, None],
            gradients["sh0s"].squeeze(-1),
        ],
        dim=-1,
    )
    gradient_features = _normalize_features(gradient_features)
    metric_score, visible = _metric_score(
        _activated_gaussians(raw), renderer, views, clamp=False
    )
    metric_feature = _normalize_features(metric_score[:, None])
    state = torch.cat([gradient_features, metric_feature], dim=-1)
    return state.detach(), metric_score.detach(), visible.detach(), views, sampled


@torch.no_grad()
def _render_blur_policy_observation(
    gaussians: Gaussians,
    renderer,
    views: dict[str, Tensor],
    objective,
) -> dict[str, float | bool]:
    """Measure blur-aware policy signals on LeGS's fixed training views."""
    if objective is None:
        raise RuntimeError("legs_blur requires a configured BlurAwareObjective")
    required = {"raw_image", "target_confidence", "known_sharp", "index"}
    missing = sorted(key for key in required if key not in views)
    if missing:
        raise RuntimeError(f"legs_blur training views lack {missing}")

    image_shape = tuple(views["image"].shape[-2:])
    output = renderer.forward(
        gaussians=gaussians,
        extrinsics=views["extrinsics"],
        intrinsics=views["intrinsics"],
        near=views["near"],
        far=views["far"],
        image_shape=image_shape,
    )
    prediction = output.color.clamp(0.0, 1.0)
    target = views["image"]
    raw = views["raw_image"]
    confidence = views["target_confidence"].float().clamp(0.0, 1.0)
    known_sharp = views["known_sharp"].bool()
    confidence = torch.where(known_sharp, torch.ones_like(confidence), confidence)
    valid_mask = views.get("valid_mask")

    squared_error = (prediction - target).square()
    if valid_mask is None:
        mse = squared_error.mean(dim=(2, 3, 4))
    else:
        valid = valid_mask.to(dtype=squared_error.dtype)
        denominator = valid.sum(dim=(2, 3, 4)).clamp_min(1.0) * target.shape[2]
        mse = (squared_error * valid).sum(dim=(2, 3, 4)) / denominator
    per_view_psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    confidence_sum = confidence.sum().clamp_min(1e-8)
    weighted_psnr = (per_view_psnr * confidence).sum() / confidence_sum

    surplus = objective.measure_probe_surplus(
        prediction,
        raw,
        target,
        known_sharp,
        confidence,
        valid_mask,
    )
    direct = views.get("direct_supervision")
    direct = known_sharp if direct is None else direct.bool()
    active = ~direct.reshape(-1)
    raw_prediction = prediction
    kernel_entropy = prediction.new_zeros(())
    kernel_radius = prediction.new_zeros(())
    mask_mean = prediction.new_zeros(())
    if active.any():
        formed, bpn_stats = objective.bpn(
            prediction,
            output.depth,
            raw,
            target,
            views["index"],
        )
        direct_image_mask = direct[..., None, None, None]
        raw_prediction = torch.where(direct_image_mask, prediction, formed)
        kernels = bpn_stats["kernels"][active]
        entropy = -(kernels * kernels.clamp_min(1e-8).log()).sum(dim=-1)
        kernel_entropy = entropy.mean() / math.log(kernels.shape[-1])
        radius_squared = (
            kernels
            * (
                objective.bpn.kernel_x.square()
                + objective.bpn.kernel_y.square()
            )
        ).sum(dim=-1)
        kernel_radius = radius_squared.clamp_min(0.0).sqrt().mean() / math.sqrt(2.0)
        mask_mean = bpn_stats["mask"][0, active].mean()

    raw_squared_error = (raw_prediction - raw).square()
    if valid_mask is None:
        raw_mse = raw_squared_error.mean(dim=(2, 3, 4))
    else:
        valid = valid_mask.to(dtype=raw_squared_error.dtype)
        denominator = valid.sum(dim=(2, 3, 4)).clamp_min(1.0) * raw.shape[2]
        raw_mse = (raw_squared_error * valid).sum(dim=(2, 3, 4)) / denominator
    raw_per_view_psnr = -10.0 * torch.log10(raw_mse.clamp_min(1e-12))
    # RAW/BPN consistency is complementary evidence for blurry views. Direct
    # sharp views are already measured by weighted_psnr and must not be counted
    # a second time here, otherwise a few sharp anchors can drown out the RAW
    # consistency signal that the coupled kernel was introduced to measure.
    raw_weight = (~direct).float() * (1.0 - confidence)
    raw_weight_sum = raw_weight.sum()
    if float(raw_weight_sum) == 0.0:
        raw_weight = torch.ones_like(raw_weight)
        raw_weight_sum = raw_weight.sum()
    weighted_raw_psnr = (raw_per_view_psnr * raw_weight).sum() / raw_weight_sum

    return {
        "weighted_psnr": float(weighted_psnr),
        "weighted_raw_psnr": float(weighted_raw_psnr),
        "surplus": float(surplus["surplus"]),
        "has_surplus": bool(surplus["has_surplus"]),
        "reliability_mean": float(confidence.mean()),
        "reliability_std": float(
            confidence.std(correction=1 if confidence.numel() > 1 else 0)
        ),
        "kernel_entropy": float(kernel_entropy),
        "kernel_radius": float(kernel_radius),
        "mask_mean": float(mask_mean),
        "bpn_active": bool(active.any()),
    }


def _standardize_blur_features(
    adc_state: LeGSStrategyState,
    features: Tensor,
) -> Tensor:
    """Keep dimensionless blur signals on one scene-independent scale."""
    adc_state.blur_feature_history.append(
        [float(value) for value in features.detach().cpu()]
    )
    # Every input is analytically normalized before this point: confidence,
    # normalized entropy/radius/mask and tanh(log-ratio) pressure all have a
    # physical [-1, 1] contract. Temporal z-scoring would erase constant but
    # important scene identity such as EVSSM reliability.
    return features.clamp(-1.0, 1.0)


def _has_blur_evidence(observation: dict[str, float | bool]) -> bool:
    """True only when blur-specific state/reward has observable support."""
    return bool(observation["bpn_active"]) or bool(observation["has_surplus"])


def _all_views_are_direct(views: dict[str, Tensor]) -> bool:
    """Return whether this view set has authoritative direct supervision only."""
    direct = views.get("direct_supervision")
    if direct is not None:
        return bool(direct.all())
    known_sharp = views.get("known_sharp")
    return bool(known_sharp.all()) if known_sharp is not None else False


def _direct_only_blur_observation() -> dict[str, float | bool]:
    """Identity observation used when a scene has no blur-specific learning task."""
    return {
        "weighted_psnr": 0.0,
        "weighted_raw_psnr": 0.0,
        "surplus": 0.0,
        "has_surplus": False,
        "reliability_mean": 1.0,
        "reliability_std": 0.0,
        "kernel_entropy": 0.0,
        "kernel_radius": 0.0,
        "mask_mean": 0.0,
        "bpn_active": False,
    }


def _blur_condition_scale(cfg: LeGSStrategyCfg, step: int) -> float:
    """Introduce blur control only after the scene representation is established."""
    if step <= cfg.blur_condition_start_iter:
        return 0.0
    return min(
        1.0,
        (step - cfg.blur_condition_start_iter)
        / max(1, cfg.blur_condition_ramp_iters),
    )


def _normalize_blur_quality_delta(
    adc_state: LeGSStrategyState,
    psnr_delta: float,
    surplus_delta: float,
    has_surplus: bool,
    reliability: float,
    device: torch.device,
    raw_psnr_delta: float | None = None,
    raw_evidence_weight: float = 1.0,
) -> float:
    """Normalize reward changes by their causal per-scene RMS."""
    values = [psnr_delta, surplus_delta]
    if raw_psnr_delta is not None:
        values.append(raw_psnr_delta)
    adc_state.blur_quality_delta_history.append(values)
    history = torch.tensor(
        adc_state.blur_quality_delta_history,
        dtype=torch.float32,
        device=device,
    )
    current = history[-1]
    rms = history.square().mean(dim=0).sqrt().clamp_min(1e-8)
    normalized = torch.tanh(current / rms)
    if raw_psnr_delta is None and not has_surplus:
        return float(normalized[0])
    reliability = min(1.0, max(0.0, float(reliability)))
    if raw_psnr_delta is not None:
        raw_evidence_weight = min(1.0, max(0.0, float(raw_evidence_weight)))
        if has_surplus:
            single_auxiliary = normalized[1]
            dual_auxiliary = 0.5 * (normalized[1] + normalized[2])
        else:
            single_auxiliary = normalized[0]
            dual_auxiliary = normalized[2]
        auxiliary = torch.lerp(
            single_auxiliary,
            dual_auxiliary,
            raw_evidence_weight,
        )
        return float(
            reliability * normalized[0]
            + (1.0 - reliability) * auxiliary
        )
    return float(
        reliability * normalized[0]
        + (1.0 - reliability) * normalized[1]
    )


def _compose_blur_conditioned_reward(
    reward: Tensor,
    actions: Tensor,
    valid: Tensor,
    quality_reward: float,
    cfg: LeGSStrategyCfg,
    step: int,
) -> tuple[Tensor, float, float, float, float, float, float]:
    """Fuse global blur quality with local delayed sensitivity credit."""
    changed = (actions != 0) & valid
    structural_fraction = float(changed.sum() / valid.sum().clamp_min(1))
    birth = ((actions == 1) | (actions == 2)) & valid
    removed = (actions == 3) & valid
    birth_count = int(birth.sum())
    removed_count = int(removed.sum())
    relative_growth = max(
        0.0,
        float(birth_count - removed_count) / max(1, actions.numel()),
    )
    net_action_direction = float(birth_count - removed_count) / max(
        1, birth_count + removed_count
    )
    condition_scale = _blur_condition_scale(cfg, step)
    # The local reward is already standardized per event. Its sigmoid is a
    # threshold-free estimate of whether this particular action was supported
    # by delayed leave-one-out sensitivity. The factor of two keeps a neutral
    # action's global-credit magnitude unchanged while redistributing credit.
    local_support = torch.sigmoid(reward.detach())
    supported_gate = 2.0 * local_support
    unsupported_gate = 2.0 * (1.0 - local_support)
    # Quality is measured after the complete structural event. Birth and prune
    # can both be necessary in the same successful reallocation, so net count
    # direction is diagnostic only and must not erase or invert event credit.
    quality_gate = supported_gate if quality_reward >= 0.0 else unsupported_gate
    capacity_gate = 1.0 - max(0.0, min(1.0, quality_reward))
    reward = reward.clone()
    reward[changed] += (
        condition_scale
        * cfg.blur_quality_weight
        * quality_reward
        * quality_gate[changed]
    )
    reward[birth] -= (
        condition_scale
        * cfg.blur_capacity_weight
        * capacity_gate
        * relative_growth
        * unsupported_gate[birth]
    )
    action_support_mean = (
        float(local_support[changed].mean()) if changed.any() else 0.0
    )
    birth_penalty_gate_mean = (
        float(unsupported_gate[birth].mean()) if birth.any() else 0.0
    )
    return (
        reward,
        structural_fraction,
        relative_growth,
        condition_scale,
        action_support_mean,
        birth_penalty_gate_mean,
        net_action_direction,
    )


def build_blur_conditioned_legs_state(
    cfg: LeGSStrategyCfg,
    gaussians: Gaussians,
    adc_state: LeGSStrategyState,
    renderer,
    context: dict[str, Tensor],
    objective,
    step: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    dict[str, Tensor],
    list[int],
    dict[str, float | bool],
    dict[str, Tensor],
    list[int],
]:
    state, metric_score, visible, views, sampled = build_legs_state(
        cfg, gaussians, renderer, context, objective=objective, step=step
    )
    blur_views, blur_sampled = _fixed_blur_probe_views(cfg, context, adc_state)
    observation = (
        _direct_only_blur_observation()
        if _all_views_are_direct(context)
        else _render_blur_policy_observation(
            gaussians, renderer, blur_views, objective
        )
    )
    if adc_state.initial_gaussian_count <= 0:
        adc_state.initial_gaussian_count = int(gaussians.means.shape[1])
    pressure = math.log(
        max(1, int(gaussians.means.shape[1])) / adc_state.initial_gaussian_count
    )
    bpn_active = bool(observation["bpn_active"])
    raw_features = state.new_tensor(
        [
            2.0 * float(observation["reliability_mean"]) - 1.0,
            2.0 * float(observation["reliability_std"]),
            float(observation["surplus"]),
            2.0 * float(observation["kernel_entropy"]) - 1.0
            if bpn_active
            else 0.0,
            2.0 * float(observation["kernel_radius"]) - 1.0
            if bpn_active
            else 0.0,
            2.0 * float(observation["mask_mean"]) - 1.0
            if bpn_active
            else 0.0,
            math.tanh(pressure),
        ]
    )
    blur_evidence_active = _has_blur_evidence(observation)
    normalized = (
        _standardize_blur_features(adc_state, raw_features)
        if blur_evidence_active
        else torch.zeros_like(raw_features)
    )
    condition_scale = (
        _blur_condition_scale(cfg, step) if blur_evidence_active else 0.0
    )
    effective = normalized * condition_scale
    adc_state.last_blur_feature_raw = [float(value) for value in raw_features]
    adc_state.last_blur_feature_normalized = [
        float(value) for value in effective
    ]
    adc_state.last_blur_condition_scale = condition_scale
    if normalized.numel() != cfg.blur_feature_dim:
        raise RuntimeError(
            f"legs_blur expected {cfg.blur_feature_dim} blur features, "
            f"got {normalized.numel()}"
        )
    conditioned = effective[None].expand(state.shape[0], -1)
    state = torch.cat([state, conditioned], dim=-1)
    if state.shape[-1] != cfg.state_dim:
        raise RuntimeError(
            f"legs_blur state has {state.shape[-1]} dimensions, "
            f"config declares {cfg.state_dim}"
        )
    return (
        state,
        metric_score,
        visible,
        views,
        sampled,
        observation,
        blur_views,
        blur_sampled,
    )


def _cosine_opacity(cfg: LeGSStrategyCfg, step: int) -> float:
    span = max(1, cfg.refine_stop_iter - cfg.refine_start_iter)
    progress = min(1.0, max(0.0, (step - cfg.refine_start_iter) / span))
    weight = 0.5 * (1.0 - math.cos(math.pi * progress))
    return cfg.min_opacity_init + weight * (
        cfg.min_opacity_final - cfg.min_opacity_init
    )


def _finish_delayed_reward(
    cfg: LeGSStrategyCfg,
    step: int,
    gaussians: Gaussians,
    adc_state: LeGSStrategyState,
    renderer,
    objective=None,
) -> None:
    if step != adc_state.pending_reward_step:
        return
    if (
        adc_state.controller is None
        or adc_state.pending_views is None
        or adc_state.pending_pre_metric is None
        or adc_state.pending_pre_visible is None
        or adc_state.parent_mapping is None
    ):
        raise RuntimeError("incomplete exact-LeGS delayed reward state")
    if cfg.blur_conditioned and adc_state.pending_blur_views is None:
        raise RuntimeError("legs_blur delayed reward lost its fixed probe views")

    new_metric, new_visible = _metric_score(
        gaussians, renderer, adc_state.pending_views, clamp=True
    )
    parent_mapping = adc_state.parent_mapping
    old_count = adc_state.pending_pre_metric.shape[0]
    aggregated_metric = new_metric.new_zeros(old_count)
    aggregated_visible = new_metric.new_zeros(old_count)
    aggregated_metric.index_add_(0, parent_mapping, new_metric)
    aggregated_visible.index_add_(0, parent_mapping, new_visible.float())

    transition = adc_state.controller.transitions[
        adc_state.pending_transition_index
    ]
    valid = transition["valid_mask"].to(new_metric.device)
    actions = transition["actions"].to(new_metric.device)
    jointly_visible = adc_state.pending_pre_visible & aggregated_visible.bool()
    valid[(actions != 3) & (~jointly_visible)] = False
    reward = torch.zeros(old_count, device=new_metric.device)
    reward[valid] = (
        aggregated_metric[valid] - adc_state.pending_pre_metric[valid]
    )
    if cfg.reward_normalize and valid.any():
        values = reward[valid]
        mean = values.mean()
        std = values.std(correction=1 if values.numel() > 1 else 0)
        reward[valid] = (values - mean) / (std + 1e-6)
        adc_state.last_reward_mean = float(mean)
        adc_state.last_reward_std = float(std)
    if cfg.blur_conditioned:
        if adc_state.pending_pre_blur_observation is None:
            raise RuntimeError("legs_blur delayed reward lost its pre-action observation")
        pre_observation = adc_state.pending_pre_blur_observation
        pre_has_blur_evidence = _has_blur_evidence(pre_observation)
        post_observation = (
            _render_blur_policy_observation(
                gaussians, renderer, adc_state.pending_blur_views, objective
            )
            if pre_has_blur_evidence
            else _direct_only_blur_observation()
        )
        psnr_delta = (
            float(post_observation["weighted_psnr"])
            - float(pre_observation["weighted_psnr"])
            if pre_has_blur_evidence
            else 0.0
        )
        raw_psnr_delta = (
            float(post_observation["weighted_raw_psnr"])
            - float(pre_observation["weighted_raw_psnr"])
            if pre_has_blur_evidence
            else 0.0
        )
        has_surplus = bool(pre_observation["has_surplus"]) and bool(
            post_observation["has_surplus"]
        )
        surplus_delta = (
            float(post_observation["surplus"])
            - float(pre_observation["surplus"])
            if has_surplus
            else 0.0
        )
        blur_evidence_active = _has_blur_evidence(
            pre_observation
        ) or _has_blur_evidence(post_observation)
        if blur_evidence_active:
            quality_reward = _normalize_blur_quality_delta(
                adc_state,
                psnr_delta,
                surplus_delta,
                has_surplus,
                float(pre_observation["reliability_mean"]),
                new_metric.device,
                raw_psnr_delta=(
                    raw_psnr_delta if objective.cfg.coupled_dual_bpn else None
                ),
                raw_evidence_weight=math.sqrt(
                    max(0.0, float(pre_observation["mask_mean"]))
                    * max(0.0, float(post_observation["mask_mean"]))
                ),
            )
            (
                reward,
                structural_fraction,
                capacity_cost,
                condition_scale,
                action_support_mean,
                birth_penalty_gate_mean,
                net_action_direction,
            ) = _compose_blur_conditioned_reward(
                reward,
                actions,
                valid,
                quality_reward,
                cfg,
                step,
            )
        else:
            # An all-authoritative-sharp scene is exactly the original LeGS
            # problem.  Do not let confidence constants or capacity terms create
            # a synthetic blur policy where no blur observation exists.
            quality_reward = 0.0
            changed = (actions != 0) & valid
            birth = ((actions == 1) | (actions == 2)) & valid
            removed = (actions == 3) & valid
            structural_fraction = float(changed.sum() / valid.sum().clamp_min(1))
            capacity_cost = max(
                0.0,
                float(int(birth.sum()) - int(removed.sum()))
                / max(1, actions.numel()),
            )
            condition_scale = 0.0
            action_support_mean = 0.0
            birth_penalty_gate_mean = 0.0
            net_action_direction = float(
                int(birth.sum()) - int(removed.sum())
            ) / max(1, int(birth.sum()) + int(removed.sum()))
        adc_state.last_blur_quality_reward = quality_reward
        adc_state.last_blur_psnr_delta = psnr_delta
        adc_state.last_blur_raw_psnr_delta = raw_psnr_delta
        adc_state.last_blur_surplus_delta = surplus_delta
        adc_state.last_blur_structural_fraction = structural_fraction
        adc_state.last_blur_capacity_cost = capacity_cost
        adc_state.last_blur_condition_scale = condition_scale
        adc_state.last_blur_action_support_mean = action_support_mean
        adc_state.last_blur_birth_penalty_gate_mean = birth_penalty_gate_mean
        adc_state.last_blur_net_action_direction = net_action_direction
    adc_state.controller.set_reward(
        adc_state.pending_transition_index, reward, valid
    )
    adc_state.last_reward_step = step

    if adc_state.controller.ready():
        with torch.enable_grad():
            # The release schedules policy LR by the decision iteration, not
            # the later reward-observation iteration.
            adc_state.controller.learn(step - cfg.reward_delay)
        adc_state.parent_mapping = None

    adc_state.pending_transition_index = -1
    adc_state.pending_reward_step = -1
    adc_state.pending_pre_metric = None
    adc_state.pending_pre_visible = None
    adc_state.pending_views = None
    adc_state.pending_blur_views = None
    adc_state.pending_pre_blur_observation = None


def _enforce_safety_cap(
    cfg: LeGSStrategyCfg,
    actions: Tensor,
    confidence: Tensor,
) -> int:
    if cfg.cap_max <= 0:
        return 0
    delete_count = int((actions == 3).sum())
    birth_indices = torch.nonzero(
        (actions == 1) | (actions == 2), as_tuple=False
    ).flatten()
    current = actions.shape[0]
    available = max(0, cfg.cap_max - current + delete_count)
    if birth_indices.numel() <= available:
        return 0
    keep = (
        birth_indices[
            torch.topk(
                confidence[birth_indices], k=available, sorted=False
            ).indices
        ]
        if available > 0
        else birth_indices[:0]
    )
    accepted = torch.zeros_like(actions, dtype=torch.bool)
    accepted[keep] = True
    rejected = birth_indices[~accepted[birth_indices]]
    actions[rejected] = 0
    return int(rejected.numel())


def _activated_opacity(gaussians: Gaussians) -> Tensor:
    opacity = gaussians.opacities
    return opacity if gaussians.stores_activated else torch.sigmoid(opacity)


def _store_activated_opacity(gaussians: Gaussians, opacity: Tensor) -> None:
    gaussians.opacities = (
        opacity
        if gaussians.stores_activated
        else torch.logit(opacity.clamp(1e-7, 1.0 - 1e-7))
    )


def _reset_legs_opacity(
    cfg: LeGSStrategyCfg,
    step: int,
    gaussians: Gaussians,
    adc_state: LeGSStrategyState,
    smoothers: dict[str, Any],
    *,
    zero_t: bool,
) -> None:
    # dynamic_reset_opacity=False in the official release, so reset_opacity is
    # called with 0.005 + 0.005 = 0.01 every 3K iterations.
    ceiling = cfg.min_opacity_init + 0.005
    opacity = torch.minimum(
        _activated_opacity(gaussians),
        torch.full_like(gaussians.opacities, ceiling),
    )
    _store_activated_opacity(gaussians, opacity)
    if smoothers.get("opacities") is not None:
        smoothers["opacities"].zero_out(zero_t=zero_t)
    adc_state.opacity_reset_count += 1
    print(f"LeGS Opacity reset @ iter {step}")


def _apply_legs_final_prune(
    cfg: LeGSStrategyCfg,
    step: int,
    gaussians: Gaussians,
    adc_state: LeGSStrategyState,
    smoothers: dict[str, Any],
) -> int:
    prune_mask = _activated_opacity(gaussians).squeeze(0) < cfg.min_opacity_final
    prune(gaussians, adc_state, prune_mask)
    _prune_objects(prune_mask, smoothers)
    if (
        adc_state.parent_mapping is not None
        and adc_state.parent_mapping.shape[0] == prune_mask.shape[0]
    ):
        adc_state.parent_mapping = adc_state.parent_mapping[~prune_mask]
    # The shared Vanilla prune helper resizes the normal-gradient state but
    # not FastGS's absolute-gradient accumulator. Policy events reset it after
    # clone/split/prune; final-prune events need the same synchronization.
    reset_fastgs_state(adc_state)
    count = int(prune_mask.sum())
    adc_state.last_event_step = step
    adc_state.last_event_kind = "final_prune"
    adc_state.last_valid_count = 0
    adc_state.last_clone_count = 0
    adc_state.last_split_count = 0
    adc_state.last_prune_count = count
    adc_state.final_prune_count += count
    row = {
        "step": step,
        "kind": "final_prune",
        "valid": 0,
        "cloned": 0,
        "split": 0,
        "pruned": count,
        "cap_truncated": 0,
        "num_gaussians": int(gaussians.means.shape[1]),
        "ppo_updates": 0 if adc_state.controller is None else adc_state.controller.update_count,
        "sampled_view_indices": [],
    }
    adc_state.event_log.append(row)
    print(
        f"LeGS Final Pruning @ iter {step}: pruned={count}, "
        f"total={row['num_gaussians']}"
    )
    return count


@torch.no_grad()
def apply_legs_strategy(
    cfg: LeGSStrategyCfg,
    step: int,
    gaussians: Gaussians | GaussiansModule,
    adc_state: LeGSStrategyState,
    smoothers: dict[str, Any],
    renderer,
    context: dict[str, Tensor],
    objective=None,
    zero_t: bool = False,
) -> tuple[int, int, int, float | None, float | None]:
    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("LeGS ADC is implemented for Gaussians")
    if adc_state.controller is None:
        adc_state.controller = LeGSPPOController(cfg, gaussians.means.device)

    _finish_delayed_reward(
        cfg, step, gaussians, adc_state, renderer, objective=objective
    )

    denom = adc_state.denom.clamp_min(1.0)
    grads = adc_state.grad2d_norm_accum / denom
    abs_grads = (
        adc_state.grad2d_abs_norm_accum / denom
        if adc_state.uses_abs_grad
        else grads
    )
    max_grad = float(grads.max()) if grads.numel() else 0.0
    max_radii = float(adc_state.radii2d.max()) if adc_state.radii2d.numel() else 0.0
    event = (
        step < cfg.refine_stop_iter
        and step > cfg.refine_start_iter
        and step % cfg.refine_every == 0
        and step % cfg.reset_every >= cfg.pause_refine_after_reset
    )
    if not event:
        # Exact post-densification LeGS cleanup: after 15K, prune opacity below
        # 0.1 every 3K steps. There is no policy action in this phase.
        if (
            cfg.do_prune
            and step > cfg.refine_stop_iter
            and step % cfg.reset_every == 0
        ):
            nr_pruned = _apply_legs_final_prune(
                cfg, step, gaussians, adc_state, smoothers
            )
            return 0, 0, nr_pruned, max_radii, max_grad
        return 0, 0, 0, max_radii, max_grad
    if adc_state.pending_reward_step >= 0:
        raise RuntimeError(
            "LeGS reward delay overlaps the next structural event; increase "
            "refine_every or reduce reward_delay"
        )

    pre_blur_observation = None
    blur_views = None
    blur_sampled_views: list[int] = []
    if cfg.blur_conditioned:
        (
            states,
            pre_metric,
            pre_visible,
            views,
            sampled_views,
            pre_blur_observation,
            blur_views,
            blur_sampled_views,
        ) = build_blur_conditioned_legs_state(
            cfg, gaussians, adc_state, renderer, context, objective, step
        )
    else:
        states, pre_metric, pre_visible, views, sampled_views = build_legs_state(
            cfg, gaussians, renderer, context, objective=objective, step=step
        )
    valid = ((grads >= cfg.grad_thresh) | (abs_grads >= cfg.grad_abs_thresh))
    valid &= pre_visible

    opacities = gaussians.opacities.squeeze(0)
    if not gaussians.stores_activated:
        opacities = torch.sigmoid(opacities)
    prune_eligible = opacities < _cosine_opacity(cfg, step)
    if cfg.use_prune_estimator:
        prune_eligible &= pre_visible
        valid |= prune_eligible
    else:
        valid[prune_eligible] = False

    actions, confidence = adc_state.controller.act(
        states, valid, prune_eligible
    )
    if not cfg.use_prune_estimator:
        actions[prune_eligible] = 3
    adc_state.last_cap_truncation = _enforce_safety_cap(
        cfg, actions, confidence
    )

    transition_index = adc_state.controller.store_transition(
        states,
        actions,
        adc_state.parent_mapping,
        valid,
        prune_eligible,
    )
    adc_state.pending_transition_index = transition_index
    adc_state.pending_reward_step = step + cfg.reward_delay
    adc_state.pending_pre_metric = pre_metric
    adc_state.pending_pre_visible = pre_visible
    adc_state.pending_views = views
    adc_state.pending_blur_views = blur_views
    adc_state.pending_pre_blur_observation = pre_blur_observation

    n_before = actions.shape[0]
    clone_mask = actions == 1
    split_mask = actions == 2
    delete_mask = actions == 3
    parent_mapping = torch.arange(n_before, device=actions.device)

    cloning(gaussians, adc_state, clone_mask)
    _clone_objects(clone_mask, smoothers, zero_t=zero_t)
    nr_cloned = int(clone_mask.sum())
    clone_indices = torch.nonzero(clone_mask, as_tuple=False).flatten()
    parent_mapping = torch.cat([parent_mapping, clone_indices])
    delete_after_clone = torch.cat(
        [delete_mask, torch.zeros(nr_cloned, dtype=torch.bool, device=actions.device)]
    )

    split_after_clone = torch.cat(
        [split_mask, torch.zeros(nr_cloned, dtype=torch.bool, device=actions.device)]
    )
    split_indices = torch.nonzero(split_after_clone, as_tuple=False).flatten()
    split_rest = ~split_after_clone
    splitting(gaussians, adc_state, split_after_clone, N=2)
    _split_objects(split_after_clone, smoothers, N=2, zero_t=zero_t)
    nr_split = int(split_after_clone.sum())
    parent_mapping = torch.cat(
        [parent_mapping[split_rest], parent_mapping[split_indices].repeat(2)]
    )
    delete_after_split = torch.cat(
        [
            delete_after_clone[split_rest],
            torch.zeros(nr_split * 2, dtype=torch.bool, device=actions.device),
        ]
    )

    prune(gaussians, adc_state, delete_after_split)
    _prune_objects(delete_after_split, smoothers)
    nr_pruned = int(delete_after_split.sum())
    parent_mapping = parent_mapping[~delete_after_split]
    adc_state.parent_mapping = parent_mapping

    reset_fastgs_state(adc_state)
    adc_state.last_event_step = step
    adc_state.last_event_kind = "policy"
    adc_state.last_sampled_view_indices = sampled_views
    adc_state.last_blur_probe_view_indices = blur_sampled_views
    adc_state.last_valid_count = int(valid.sum())
    adc_state.last_clone_count = nr_cloned
    adc_state.last_split_count = nr_split
    adc_state.last_prune_count = nr_pruned
    if cfg.do_opacity_reset and step > 0 and step % cfg.reset_every == 0:
        _reset_legs_opacity(
            cfg,
            step,
            gaussians,
            adc_state,
            smoothers,
            zero_t=zero_t,
        )
    event_row = {
        "step": step,
        "kind": "policy",
        "valid": adc_state.last_valid_count,
        "cloned": nr_cloned,
        "split": nr_split,
        "pruned": nr_pruned,
        "cap_truncated": adc_state.last_cap_truncation,
        "num_gaussians": int(gaussians.means.shape[1]),
        "ppo_updates": adc_state.controller.update_count,
        "sampled_view_indices": sampled_views,
        "blur_probe_view_indices": blur_sampled_views,
        "opacity_reset": bool(
            cfg.do_opacity_reset and step > 0 and step % cfg.reset_every == 0
        ),
    }
    adc_state.event_log.append(event_row)
    print(
        "LeGS Densification/Pruning "
        f"@ iter {step}: valid={event_row['valid']}, cloned={nr_cloned}, "
        f"split={nr_split}, pruned={nr_pruned}, "
        f"cap_truncated={event_row['cap_truncated']}, "
        f"total={event_row['num_gaussians']}"
    )
    return nr_cloned, nr_split, nr_pruned, max_radii, max_grad

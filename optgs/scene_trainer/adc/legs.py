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
        self.encoder = MLPStateEncoder(cfg.state_dim, cfg.hidden_dim).to(device)
        self.actor = PPOActor(cfg.hidden_dim).to(device)
        self.prune_estimator = PPOPruneEstimator(cfg.hidden_dim).to(device)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=cfg.actor_lr_init, weight_decay=1e-4
        )
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
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

    def _encode_chunks(self, states: Tensor, requires_grad: bool) -> Tensor:
        chunks = []
        size = max(1, self.cfg.ppo_chunk_size)
        for start in range(0, states.shape[0], size):
            chunk = states[start : start + size].to(self.device)
            if requires_grad:
                chunk = chunk.detach().requires_grad_(True)
            chunks.append(self.encoder(chunk))
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
                        encoded = self.encoder(
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
    opacity_reset_count: int = 0
    final_prune_count: int = 0
    event_log: list[dict[str, Any]] = field(default_factory=list)

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
        "opacity_reset_count",
        "final_prune_count",
        "event_log",
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
    if "index" in context and context["index"] is not None:
        source = context["index"].index_select(1, index).reshape(-1)
        sampled = [int(value) for value in source.detach().cpu()]
    else:
        sampled = selected
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
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], list[int]]:
    """Build the official 11-D gradient+sensitivity state."""
    views, sampled = _sample_official_views(cfg, context)
    raw = _raw_parameter_clones(gaussians)
    _, gradients, _ = calc_input_gradients(
        views,
        *raw,
        renderer,
        need_2d_grads=False,
        chunk_size=-1,
        any_adc=False,
        input_objective=None,
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
    zero_t: bool = False,
) -> tuple[int, int, int, float | None, float | None]:
    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("LeGS ADC is implemented for Gaussians")
    if adc_state.controller is None:
        adc_state.controller = LeGSPPOController(cfg, gaussians.means.device)

    _finish_delayed_reward(cfg, step, gaussians, adc_state, renderer)

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

    states, pre_metric, pre_visible, views, sampled_views = build_legs_state(
        cfg, gaussians, renderer, context
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

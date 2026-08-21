"""Public API: use optgs's learned optimizer in external 3DGS codebases.

Typical inria (graphdeco-inria/gaussian-splatting) integration — replace the
hand-written training loop with three lines::

    from optgs.experimental.api import OptGS

    gaussians = GaussianModel(sh_degree)        # set up as usual (SfM init)
    scene = Scene(dataset, gaussians)
    optgs = OptGS(checkpoint="hf://org/repo/model.ckpt", device="cuda")
    optgs.initialize(scene)                      # ingest scene + build optimizer
    optgs.optimize(scene)                        # learned optimization, written back in place
    scene.save(iteration)                        # proceed as normal

Full-replacement semantics: ``optimize`` overwrites ``scene.gaussians`` in
place and nulls the inria Adam optimizer + densification accumulators. If you
later want to resume inria Adam, call ``gaussians.training_setup(...)`` again.

For non-inria codebases use :meth:`OptGS.initialize_from_ply` /
:meth:`OptGS.initialize_from_tensors` + :meth:`OptGS.export_ply`.

External SfM scenes carry no optgs encoder features, so checkpoints trained
with ``init_state_wo_features=False`` are coerced at construction (with a
warning): the feature-conditioned ``state_proj`` weights are dropped and the
optimizer state is initialized standard-normal.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import torch

from optgs.experimental.api.integration.scene_protocol import OptGSError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from optgs.model.types import Gaussians

__all__ = ["OptGS", "OptGSError"]


class OptGS:
    """Facade around the learned per-scene optimizer."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str | torch.device = "cuda",
        num_refine: int | None = None,
        iter_batch_size: int | None = None,
        opt_batch_size: int | None = None,
        opt_batch_strategy: str | None = None,
        rollout_horizon: int | None = None,
        background_color: Sequence[float] | None = None,
        decoder_backend: str | None = None,
        rasterize_mode: str | None = None,
        eps2d: float | None = None,
        strict_load: bool = True,
    ) -> None:
        if not checkpoint:
            raise OptGSError(
                "OptGS(checkpoint=...) is required (an 'hf://org/repo/file' "
                "reference or a local checkpoint path)."
            )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise OptGSError(
                "OptGS requires a CUDA device (the learned optimizer uses "
                "CUDA/KNN kernels). Pass device='cuda'."
            )
        # float32 only — the learned optimizer's CUDA/KNN kernels and the
        # gsplat rasterizer require it (and the checkpoint trained with it).
        self.dtype = torch.float32
        self.iter_batch_size = iter_batch_size
        self.opt_batch_size = opt_batch_size
        self.opt_batch_strategy = opt_batch_strategy
        self.rollout_horizon = rollout_horizon

        from optgs.config import _find_config_for_checkpoint
        from optgs.experimental.api.integration.config_bridge import (
            build_decoder,
            build_optimizer,
            build_optimizer_cfg,
            get_checkpoint_scalar,
            get_scene_trainer_scalar,
            load_optimizer_state,
        )
        from optgs.misc.hf_ckpt import hf_sibling_config, maybe_resolve_hf_ref

        local_ckpt = maybe_resolve_hf_ref(checkpoint)
        # For hf:// refs, hf_hub_download fetches only the ckpt; pull the
        # sibling config.yaml so the architecture can be rebuilt.
        cfg_path = hf_sibling_config(checkpoint) or _find_config_for_checkpoint(local_ckpt)
        if cfg_path is None:
            raise OptGSError(
                f"no config.yaml found next to checkpoint {local_ckpt!r} "
                f"(looked for <ckpt>/../../config.yaml and the wandb "
                f"latest-run fallback). OptGS needs the training config to "
                f"rebuild the optimizer architecture."
        )

        opt_cfg, num_update_steps = build_optimizer_cfg(cfg_path)
        self.checkpoint_num_refine = (
            None if num_update_steps is None else int(num_update_steps)
        )
        if self.rollout_horizon is None:
            self.rollout_horizon = self.checkpoint_num_refine or 0
        self.rollout_horizon = int(self.rollout_horizon)
        if self.rollout_horizon < 0:
            raise OptGSError("rollout_horizon must be zero or positive")
        self.reference_context_views = int(
            get_checkpoint_scalar(
                cfg_path,
                "dataset.view_sampler.num_context_views",
                1,
            )
        )
        reference_init = get_checkpoint_scalar(
            cfg_path,
            "scene_trainer.scene_initializer.eval_fixed_gaussians_num",
            None,
        )
        if reference_init is None:
            reference_init = get_checkpoint_scalar(
                cfg_path,
                "scene_trainer.scene_initializer.train_fixed_gaussians_num",
                getattr(opt_cfg, "max_active_gaussians", 0),
            )
        self.reference_initial_gaussians = int(reference_init)
        if self.reference_context_views <= 0:
            raise OptGSError("checkpoint reference context-view count must be positive")
        if self.reference_initial_gaussians <= 0:
            raise OptGSError("checkpoint reference Gaussian count must be positive")

        if not getattr(opt_cfg, "init_state_wo_features", False):
            warnings.warn(
                "this checkpoint was trained WITH encoder features "
                "(scene_trainer.scene_optimizer.init_state_wo_features=False). "
                "External SfM/inria scenes carry no optgs encoder features; "
                "proceeding with init_state_wo_features=True — the "
                "feature-conditioned state_proj weights are dropped and the "
                "initial optimizer state is set to a standard-normal random "
                "vector (init_state_type='random', init_state_scale=1.0)."
            )
            opt_cfg.init_state_wo_features = True
            opt_cfg.init_state_type = "random"
            opt_cfg.init_state_scale = 1.0

        optimizer = build_optimizer(opt_cfg)  # asserts cfg.name; nn.Module
        load_optimizer_state(
            optimizer, local_ckpt, init_state_wo_features=True, strict=strict_load
        )
        self.optimizer = optimizer.to(device=self.device, dtype=self.dtype).eval()

        from types import SimpleNamespace

        bg = list(background_color) if background_color is not None else [0.0, 0.0, 0.0]
        # Build the renderer the checkpoint trained with (gsplat by default;
        # NOT a hardcoded backend — see build_decoder). rasterize_mode / eps2d,
        # when given, override the checkpoint's decoder config.
        decoder_overrides = {
            k: v
            for k, v in (
                ("name", decoder_backend),
                ("rasterize_mode", rasterize_mode),
                ("eps2d", eps2d),
            )
            if v is not None
        }
        self.decoder = build_decoder(
            cfg_path, SimpleNamespace(background_color=bg), decoder_overrides
        ).to(self.device)

        resolved = num_refine if num_refine is not None else num_update_steps
        if resolved is None:
            raise OptGSError(
                "num_refine could not be determined: pass OptGS(num_refine=...) "
                "or use a checkpoint whose config has "
                "scene_trainer.num_update_steps."
            )
        self.num_refine = int(resolved)

        # Render-batching size: user override, else the checkpoint's
        # scene_trainer.iter_batch_size (-1 = render all views per step).
        if self.iter_batch_size is None:
            self.iter_batch_size = int(
                get_scene_trainer_scalar(cfg_path, "iter_batch_size", -1)
            )

        # Per-step view minibatch — opt_batch_size views are fed to the
        # optimizer each step (the checkpoint's scene_trainer.opt_batch_size /
        # opt_batch_strategy, i.e. the regime it was trained with). -1 = all.
        if self.opt_batch_size is None:
            self.opt_batch_size = int(
                get_scene_trainer_scalar(cfg_path, "opt_batch_size", -1)
            )
        if self.opt_batch_strategy is None:
            self.opt_batch_strategy = str(
                get_scene_trainer_scalar(cfg_path, "opt_batch_strategy", "random")
            )
        if self.opt_batch_strategy not in (
            "random",
            "sequential",
            "fps",
            "supervision_fps",
        ):
            raise OptGSError(
                f"opt_batch_strategy={self.opt_batch_strategy!r} is not supported "
                "by the API (supported: 'random', 'sequential', 'fps', "
                "'supervision_fps'). Pass "
                f"OptGS(opt_batch_strategy='random')."
            )

        self._opt_cfg = opt_cfg
        # SH degree the checkpoint's Gaussians use — derived from the optimizer
        # cfg's init_sh_d (= (sh_degree + 1) ** 2, set by opt_cfg.update from the
        # initializer cfg). API consumers build/render Gaussians with this; it is
        # dictated by the checkpoint, not a free choice.
        self.sh_degree = int(round(opt_cfg.init_sh_d ** 0.5)) - 1
        self._initialized = False
        self._scene_ref = None
        self._context = None
        self._init_output = None
        self._refined: "Gaussians | None" = None
        self._input_objective = None
        self.capacity_events: list[dict] = []
        self.optimizer_transitions: list[dict] = []
        self._reset_supervision_sampler()

    def _reset_supervision_sampler(self) -> None:
        self._supervision_draws = 0
        self._supervision_sharp_draws = 0
        self._supervision_pool_stacks: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def initialize(self, scene: object) -> "OptGS":
        """Ingest an already-initialized inria-style scene.

        This does NOT run optgs's learned Initializer — the scene already has
        Gaussians (e.g. from SfM / inria ``create_from_pcd``).
        """
        from optgs.experimental.api.integration.inria_bridge import (
            batched_views_from_cameras,
            optgs_gaussians_from_inria_model,
        )
        from optgs.experimental.api.integration.scene_protocol import (
            assert_scene_protocol,
        )
        from optgs.scene_trainer.initializer.initializer import InitializerOutput

        assert_scene_protocol(scene)
        g = optgs_gaussians_from_inria_model(
            scene.gaussians, device=self.device, dtype=self.dtype
        )
        self._init_output = InitializerOutput(gaussians=g, features=None, depths=None)
        self._context = batched_views_from_cameras(
            list(scene.getTrainCameras()),
            scene_scale=float(scene.cameras_extent),
            device=self.device,
            dtype=self.dtype,
        )
        self._reset_supervision_sampler()
        self._scene_ref = scene
        self._initialized = True
        return self

    def initialize_from_ply(
        self,
        ply_path: str,
        cameras: Sequence[object],
        *,
        sh_degree: int,
        scene_scale: float,
    ) -> "OptGS":
        """Low-level ingest for non-inria codebases (no inria ``Scene``).

        ``cameras`` is a sequence of inria-``Camera``-like objects (``R``,
        ``T``, ``FoVx``, ``FoVy``, ``image_width``, ``image_height``,
        ``original_image``).
        """
        from optgs.experimental.api.integration.inria_bridge import (
            batched_views_from_cameras,
            optgs_gaussians_from_ply,
        )
        from optgs.scene_trainer.initializer.initializer import InitializerOutput

        g = optgs_gaussians_from_ply(
            ply_path, sh_degree=sh_degree, device=self.device, dtype=self.dtype
        )
        self._init_output = InitializerOutput(gaussians=g, features=None, depths=None)
        self._context = batched_views_from_cameras(
            list(cameras), scene_scale=scene_scale, device=self.device, dtype=self.dtype
        )
        self._reset_supervision_sampler()
        self._scene_ref = None
        self._initialized = True
        return self

    def initialize_from_tensors(self, gaussians: object, batched_views: object) -> "OptGS":
        """Low-level ingest from optgs-native objects (power users).

        ``gaussians``: an optgs ``Gaussians`` (batch=1, post-activation).
        ``batched_views``: an optgs ``BatchedViews`` or a dict accepted by
        ``BatchedViews.from_dict``.
        """
        from optgs.dataset.data_types import BatchedViews
        from optgs.model.types import Gaussians
        from optgs.scene_trainer.initializer.initializer import InitializerOutput

        if not isinstance(gaussians, Gaussians):
            raise OptGSError(
                "initialize_from_tensors expects an optgs Gaussians instance "
                "(use initialize_from_ply for raw 3DGS PLY input)."
            )
        bv = (
            batched_views
            if isinstance(batched_views, BatchedViews)
            else BatchedViews.from_dict(batched_views)
        )
        self._init_output = InitializerOutput(
            gaussians=gaussians.to(device=self.device, dtype=self.dtype),
            features=None,
            depths=None,
        )
        self._context = bv
        self._reset_supervision_sampler()
        self._scene_ref = None
        self._initialized = True
        return self

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    def _view_minibatch(self, views):
        """Sample the next per-step view minibatch from ``views``.

        Mirrors SceneTrainer's viewpoint-stack cycling: views are drawn
        ``opt_batch_size`` at a time and the stack is refilled once exhausted,
        so every view is seen before any repeats. ``random``/``sequential`` take
        the front of the (shuffled/ordered) stack; ``fps`` picks a
        farthest-point spread over the remaining views' camera positions.
        Returns ``views`` unchanged when ``opt_batch_size`` is <= 0 or already
        covers the whole scene.
        """
        v = views.image.shape[1]
        bs = self.opt_batch_size
        if bs <= 0 or bs >= v:
            return views

        if self.opt_batch_strategy == "supervision_fps":
            return self._supervision_fps_minibatch(views, bs)

        views.reset_viewpoint_stack_if_needed(self.opt_batch_strategy, bs)
        stack = views.viewpoint_stack  # [B, V_stack]

        if self.opt_batch_strategy == "fps":
            from optgs.dataset.view_sampler.view_sampler_bounded_v2 import (
                farthest_point_sample,
            )

            b = stack.shape[0]
            arange = torch.arange(b, device=stack.device)[:, None]
            # FPS over the camera positions of the views still in the stack.
            positions = views.extrinsics[arange, stack][:, :, :3, 3]  # [B, V_stack, 3]
            local = farthest_point_sample(positions, bs, first_idx_strategy="random")
            idx = stack[arange, local]  # [B, bs]
            keep = ~(stack.unsqueeze(-1) == idx.unsqueeze(1)).any(-1)  # [B, V_stack]
            views.viewpoint_stack = stack[keep].view(b, -1)
        else:  # random / sequential — take the front of the stack
            idx = stack[:, :bs]
            views.viewpoint_stack = stack[:, bs:]
        return views.batchify_views(idx)

    def _draw_fps_pool(
        self,
        views,
        pool: torch.Tensor,
        count: int,
        key: str,
        selected: list[int],
    ) -> list[int]:
        """Draw unique views while cycling every member of one quality pool."""
        if count <= 0 or pool.numel() == 0:
            return []
        stack = self._supervision_pool_stacks.get(key)
        if stack is None:
            stack = pool.clone()
        result: list[int] = []
        selected_set = set(selected)
        positions = views.extrinsics[0, :, :3, 3]
        while len(result) < count:
            if stack.numel() == 0:
                stack = pool.clone()
            keep = torch.tensor(
                [int(index) not in selected_set for index in stack],
                device=stack.device,
                dtype=torch.bool,
            )
            candidates = stack[keep]
            if candidates.numel() == 0:
                break
            if selected:
                selected_positions = positions[
                    torch.tensor(selected, device=positions.device)
                ]
                distances = torch.cdist(
                    positions[candidates], selected_positions
                ).amin(dim=1)
                chosen = int(candidates[torch.argmax(distances)])
            else:
                chosen = int(
                    candidates[
                        torch.randint(
                            candidates.numel(), (1,), device=candidates.device
                        )
                    ]
                )
            selected.append(chosen)
            selected_set.add(chosen)
            result.append(chosen)
            stack = stack[stack != chosen]
        self._supervision_pool_stacks[key] = stack
        return result

    def _draw_supervision_pool(
        self,
        views,
        pool: torch.Tensor,
        count: int,
        key: str,
        selected: list[int],
    ) -> list[int]:
        """Draw a quality-pool quota, repeating only when quota exceeds size.

        FPS supplies every distinct camera first. A repeated observation then
        represents additional supervision mass, which is required when a rare
        sharp pool contains fewer views than its w10 batch quota.
        """
        if count <= 0 or pool.numel() == 0:
            return []
        distinct = self._draw_fps_pool(
            views,
            pool,
            min(count, int(pool.numel())),
            key,
            selected,
        )
        if len(distinct) != min(count, int(pool.numel())):
            raise OptGSError(f"failed to draw distinct views from {key} pool")
        result = list(distinct)
        while len(result) < count:
            chosen = distinct[(len(result) - len(distinct)) % len(distinct)]
            selected.append(chosen)
            result.append(chosen)
        return result

    def _supervision_fps_minibatch(self, views, batch_size: int):
        """Realize the scene-level supervision risk through view sampling.

        ``sampling_mass`` defines the desired expected exposure. Known-sharp
        and other views are scheduled by cumulative weighted fair queuing;
        FPS inside each pool preserves camera-space coverage. This makes a
        rare sharp frame's w10 contract real even in thousand-frame streams,
        without a dataset label or a frame-count threshold.
        """
        if views.extrinsics.shape[0] != 1:
            raise OptGSError("supervision_fps currently requires scene batch size 1")
        if views.known_sharp is None or views.sampling_mass is None:
            raise OptGSError(
                "supervision_fps requires known_sharp and sampling_mass"
            )
        sharp_mask = views.known_sharp[0].bool()
        mass = views.sampling_mass[0].float().clamp_min(0.0)
        if mass.shape != sharp_mask.shape or not bool(torch.isfinite(mass).all()):
            raise OptGSError("invalid supervision sampling mass")
        sharp_pool = torch.nonzero(sharp_mask, as_tuple=False).flatten()
        other_pool = torch.nonzero(~sharp_mask, as_tuple=False).flatten()
        total_mass = float(mass.sum())
        if total_mass <= 0.0:
            raise OptGSError("supervision sampling mass must have positive sum")
        sharp_fraction = float(mass[sharp_mask].sum()) / total_mass

        draws_after = self._supervision_draws + batch_size
        target_sharp_after = round(draws_after * sharp_fraction)
        sharp_slots = max(
            0,
            min(batch_size, target_sharp_after - self._supervision_sharp_draws),
        )
        if sharp_pool.numel() == 0:
            sharp_slots = 0
        if other_pool.numel() == 0:
            sharp_slots = batch_size

        selected: list[int] = []
        sharp_selected = self._draw_supervision_pool(
            views,
            sharp_pool,
            sharp_slots,
            "sharp",
            selected,
        )
        other_slots = batch_size - sharp_slots
        other_selected = self._draw_supervision_pool(
            views,
            other_pool,
            other_slots,
            "other",
            selected,
        )
        if len(selected) != batch_size:
            raise OptGSError(
                f"supervision_fps selected {len(selected)} of {batch_size} views"
            )

        self._supervision_draws += batch_size
        self._supervision_sharp_draws += len(sharp_selected)
        idx = torch.tensor(selected, device=sharp_mask.device).unsqueeze(0)
        return views.batchify_views(idx)

    def configure_adc(
        self, strategy: str, *, reward_conditioned: bool = False
    ) -> "OptGS":
        """Set the ADC (densification) strategy for subsequent ``optimize`` runs.

        ``strategy`` is a raw refiner name: ``"none"`` (the default the checkpoints
        ship with) leaves the Gaussian set fixed; ``"default"``/``"edgs"``/``"mcmc"``/…
        enable clone/split/prune with the schedule rescaled to ``self.num_refine``
        (see ``build_refiner_cfg``). Call before ``optimize``/``optimize_iter`` — the
        ADC state is created in ``on_scene_start`` and gated on the refiner's flags.
        """
        from dataclasses import replace

        from optgs.experimental.api.integration.config_bridge import build_refiner_cfg
        from optgs.scene_trainer.adc.vanilla import AdaptiveStrategyCfg

        refiner = build_refiner_cfg(strategy, self.num_refine)
        if isinstance(refiner, AdaptiveStrategyCfg):
            # This is an architectural limit of the released learned optimizer,
            # not a dataset-tuned point budget. The controller converts it to a
            # scene-wide capacity using online visibility statistics.
            active_budget = int(
                getattr(self.optimizer.cfg, "max_active_gaussians", refiner.active_budget)
            )
            refiner = replace(
                refiner,
                active_budget=active_budget,
                reward_conditioned=bool(reward_conditioned),
            )
        elif reward_conditioned:
            raise OptGSError(
                "reward-conditioned densification requires adc='adaptive'"
            )
        self.optimizer.cfg.refiner = refiner
        return self

    def configure_input_objective(self, objective) -> "OptGS":
        """Attach an optional per-scene objective used to form optimizer gradients.

        The learned optimizer remains unchanged: it still consumes per-Gaussian
        gradients and predicts the Gaussian updates. The objective only defines
        which differentiable image-formation loss produces those gradients. A
        value of ``None`` restores the checkpoint's original photometric loss.
        """
        self._input_objective = objective
        if objective is not None:
            objective.to(device=self.device, dtype=self.dtype)
            objective.configure_run(num_steps=self.num_refine)
        return self

    def _restart_learned_rollout(self, opt, inp):
        """Restart checkpoint-horizon state while preserving the current scene.

        The released optimizer's recurrent state was trained on finite
        rollouts. Long scenes use several architecture-matched rollouts rather
        than extrapolating that state indefinitely. Scene Gaussians, the BPN,
        supervision scheduler, and accumulated capacity receipts are retained;
        only the learned latent, gradient normalizer, and ADC observation state
        are refreshed. If a checkpoint enables time encoding, its local time is
        reset by the fresh ``OptimizerOutput`` as well.
        """
        from optgs.scene_trainer.initializer.initializer import InitializerOutput
        from optgs.scene_trainer.optimizer.optimizer import OptimizerOutput
        from optgs.scene_trainer.adc.vanilla import (
            transfer_adaptive_reward_state,
        )
        from optgs.scene_trainer.adc.legs import transfer_legs_runtime_state

        capacity_events = list(getattr(opt, "capacity_events", []))
        current = inp.prev_output.gaussians
        previous_adc_state = getattr(inp.prev_output.state, "adc_state", None)
        opt.on_scene_end()
        inp.prev_output = InitializerOutput(
            gaussians=current,
            features=None,
            depths=None,
        )
        opt.on_scene_start(inp)
        transfer_adaptive_reward_state(
            previous_adc_state,
            getattr(inp.prev_output.state, "adc_state", None),
        )
        transfer_legs_runtime_state(
            previous_adc_state,
            getattr(inp.prev_output.state, "adc_state", None),
        )
        if hasattr(opt, "capacity_events"):
            opt.capacity_events = capacity_events
        restarted = OptimizerOutput.empty(t=0)
        restarted.T = self.rollout_horizon
        return restarted

    @torch.no_grad()
    def optimize(self, scene: object | None = None, *, optimizer=None):
        """Run the learned optimization.

        inria path: refined Gaussians are written back into ``scene.gaussians``
        in place and ``scene.gaussians`` is returned. Low-level path: the
        refined optgs ``Gaussians`` is returned (use :meth:`export_ply` to
        persist).

        ``optimizer`` swaps in a different optgs ``Optimizer`` (e.g. an Adam
        baseline) — running the *same* per-scene pipeline (init, view minibatch,
        step budget, renderer) with another update rule, i.e. a fair
        comparison. Defaults to the checkpoint's learned optimizer.
        """
        if scene is not None and scene is not self._scene_ref:
            self.initialize(scene)
        if not self._initialized:
            raise OptGSError("call initialize(scene) before optimize().")

        opt = optimizer if optimizer is not None else self.optimizer
        opt.input_objective = self._input_objective

        from optgs.scene_trainer.optimizer.optimizer import (
            OptimizerInput,
            OptimizerOutput,
            OptimizerPreviousOutput,
        )

        inp = OptimizerInput(
            context=self._context,
            renderer=self.decoder,
            prev_output=self._init_output,
            iter_batch_size=self.iter_batch_size,
            target=self._context,
        )
        opt.validate_input(inp)
        opt.on_scene_start(inp)  # InitializerOutput -> OptimizerPreviousOutput (+ADC)
        if not isinstance(inp.prev_output, OptimizerPreviousOutput):
            raise OptGSError(
                "optimizer.on_scene_start did not produce an "
                f"OptimizerPreviousOutput (got {type(inp.prev_output)})."
            )

        out = OptimizerOutput.empty(t=0)
        out.T = self.num_refine
        steps = range(self.num_refine)
        try:
            from tqdm import tqdm

            steps = tqdm(steps, desc=f"optimize[{type(opt).__name__}]")
        except Exception:
            pass
        for step in steps:
            if (
                step > 0
                and self.rollout_horizon > 0
                and step % self.rollout_horizon == 0
            ):
                out = self._restart_learned_rollout(opt, inp)
            # Feed the optimizer a fresh view minibatch each step (the regime it
            # was trained with); full_context/full_target stay the whole scene.
            batch = self._view_minibatch(self._context)
            inp.context = batch
            inp.target = batch
            out = opt(
                step, inp, out, full_context=self._context, full_target=self._context
            )
            out.t = (out.t or 0) + 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        opt.on_scene_end()

        final = inp.prev_output.gaussians
        self._refined = final

        if self._scene_ref is not None:
            from optgs.experimental.api.integration.inria_bridge import (
                write_back_to_inria_model,
            )

            write_back_to_inria_model(self._scene_ref.gaussians, final)
            return self._scene_ref.gaussians
        return final

    def optimize_iter(
        self,
        *,
        optimizer=None,
        fallback_optimizer=None,
        switch_step: int | None = None,
    ):
        """Generator form of :meth:`optimize`: yields ``(step, gaussians)`` after
        each optimization step.

        Lets a caller drive the learned optimization one step at a time and
        render the Gaussians in between — used by ``demo.py``'s ``--with-gui``.
        ``on_scene_end()`` runs even if the caller closes the generator early
        (e.g. a GUI Reset), via the ``finally`` block.
        """
        if not self._initialized:
            raise OptGSError("call initialize(...) before optimize_iter().")

        opt = optimizer if optimizer is not None else self.optimizer
        opt.input_objective = self._input_objective
        if fallback_optimizer is not None:
            if switch_step is None or not 0 < switch_step < self.num_refine:
                raise OptGSError(
                    "fallback_optimizer requires 0 < switch_step < num_refine"
                )
            fallback_optimizer.input_objective = self._input_objective
        self.capacity_events = []
        self.optimizer_transitions = []

        from optgs.scene_trainer.optimizer.optimizer import (
            OptimizerInput,
            OptimizerOutput,
            OptimizerPreviousOutput,
        )

        # torch's grad mode is thread-local, and a generator consumer (e.g.
        # gradio) may resume this generator on a *different* worker thread each
        # step — so a single ``with torch.no_grad()`` spanning the ``yield`` does
        # not reliably hold across steps. Enter no_grad *per step* and yield
        # outside it: the Adam baseline's in-place parameter updates require
        # no_grad ambient (calc_input_gradients re-enables grad internally just
        # for the gradient computation), so a step that resumes on a grad-enabled
        # thread would otherwise raise "a leaf Variable that requires grad is
        # being used in an in-place operation" and leak the autograd graph.
        inp = OptimizerInput(
            context=self._context,
            renderer=self.decoder,
            prev_output=self._init_output,
            iter_batch_size=self.iter_batch_size,
            target=self._context,
        )
        with torch.no_grad():
            opt.validate_input(inp)
            opt.on_scene_start(inp)  # InitializerOutput -> OptimizerPreviousOutput
        if not isinstance(inp.prev_output, OptimizerPreviousOutput):
            raise OptGSError(
                "optimizer.on_scene_start did not produce an "
                f"OptimizerPreviousOutput (got {type(inp.prev_output)})."
            )

        out = OptimizerOutput.empty(t=0)
        out.T = self.num_refine
        try:
            for step in range(self.num_refine):
                with torch.no_grad():
                    if fallback_optimizer is not None and step == switch_step:
                        from optgs.scene_trainer.adc.vanilla import (
                            transfer_adaptive_reward_state,
                        )
                        from optgs.scene_trainer.adc.legs import (
                            transfer_legs_runtime_state,
                        )

                        self.capacity_events.extend(
                            list(getattr(opt, "capacity_events", []))
                        )
                        current = inp.prev_output.gaussians
                        previous_adc_state = getattr(
                            inp.prev_output.state, "adc_state", None
                        )
                        opt.on_scene_end()
                        from optgs.scene_trainer.initializer.initializer import (
                            InitializerOutput,
                        )

                        inp.prev_output = InitializerOutput(
                            gaussians=current,
                            features=None,
                            depths=None,
                        )
                        opt = fallback_optimizer
                        opt.validate_input(inp)
                        opt.on_scene_start(inp)
                        transfer_adaptive_reward_state(
                            previous_adc_state,
                            getattr(inp.prev_output.state, "adc_state", None),
                        )
                        transfer_legs_runtime_state(
                            previous_adc_state,
                            getattr(inp.prev_output.state, "adc_state", None),
                        )
                        out = OptimizerOutput.empty(t=0)
                        out.T = self.num_refine - step
                        self.optimizer_transitions.append(
                            {
                                "step": step,
                                "from": type(self.optimizer).__name__,
                                "to": type(opt).__name__,
                                "reason": "released_checkpoint_rollout_horizon",
                            }
                        )
                    if (
                        step > 0
                        and self.rollout_horizon > 0
                        and step % self.rollout_horizon == 0
                        and opt is self.optimizer
                    ):
                        out = self._restart_learned_rollout(opt, inp)
                    # Fresh view minibatch each step (the regime the optimizer
                    # was trained with); full_context/target stay the whole scene.
                    batch = self._view_minibatch(self._context)
                    inp.context = batch
                    inp.target = batch
                    out = opt(
                        step, inp, out,
                        full_context=self._context, full_target=self._context,
                    )
                    out.t = (out.t or 0) + 1
                    g = inp.prev_output.gaussians
                yield step, g
        finally:
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                opt.on_scene_end()
            self.capacity_events.extend(list(getattr(opt, "capacity_events", [])))
            self._refined = inp.prev_output.gaussians

    def export_ply(self, path: str) -> None:
        """Write the most recently refined Gaussians to a 3DGS PLY."""
        if self._refined is None:
            raise OptGSError("nothing to export — call optimize() first.")
        from pathlib import Path

        from optgs.model.ply_export import save_gaussian_ply

        save_gaussian_ply(self._refined, save_path=Path(path))

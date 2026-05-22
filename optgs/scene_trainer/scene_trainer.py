import math
import random
from pathlib import Path
from typing import Optional, Mapping, Any

import torch
from einops import rearrange
from lightning_fabric.utilities import move_data_to_device
from torch import Tensor, nn
from tqdm import tqdm

from optgs.dataset import DatasetCfg
from optgs.dataset.data_types import BatchedExample
from optgs.dataset.view_sampler.view_sampler_bounded_v2 import farthest_point_sample
from optgs.loss.loss_monodepth import get_monodepth_model
from optgs.misc.benchmarker import Benchmarker
from optgs.misc.io import FrequencyScheduler
from optgs.misc.step_tracker import StepTracker
from optgs.model.decoder.decoder import Decoder
from optgs.model.types import Gaussians
from optgs.paths import DEBUG
from optgs.scene_trainer.initializer import get_scene_initializer
from optgs.scene_trainer.initializer.initializer import InitializerOutput
from optgs.scene_trainer.optimizer import get_scene_optimizer
from optgs.scene_trainer.optimizer.optimizer import OptimizerInput, Optimizer, OptimizerOutput, OptimizerPreviousOutput
from optgs.scene_trainer.postprocessing import PostProcessing3DGS
from optgs.scene_trainer.scene_trainer_cfg import SceneTrainerCfg, TestCfg, TrainCfg
from optgs.scripts.dev.debugging_optimizer import debugging_convergence, debugging_invisible_gaussians


class SceneTrainer(nn.Module):
    test_cfg: TestCfg
    train_cfg: TrainCfg
    scene_trainer_cfg: SceneTrainerCfg
    decoder: Decoder
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]

    def __init__(
            self,
            test_cfg: TestCfg,
            train_cfg: TrainCfg,
            scene_trainer_cfg: SceneTrainerCfg,
            decoder: Decoder,
            step_tracker: StepTracker | None,
            eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__()
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg
        self.scene_trainer_cfg = scene_trainer_cfg

        # Set up the model
        self.initializer = get_scene_initializer(scene_trainer_cfg.scene_initializer)

        # Scene trainer performs updates
        if self.scene_trainer_cfg.num_update_steps > 0:
            optimizer_save_every = FrequencyScheduler(
                frequencies=self.test_cfg.save_every_freq,
                steps=self.test_cfg.save_every_steps,
                iters=self.test_cfg.save_at_iters,
                last_step=self.scene_trainer_cfg.num_update_steps,
                enable_context=self.test_cfg.eval_context_views,
            )
            self.optimizer: Optimizer | None = get_scene_optimizer(scene_trainer_cfg.scene_optimizer)
            self.optimizer.save_every = optimizer_save_every
        else:
            self.optimizer = None

        self.decoder = decoder

        self.benchmarker = Benchmarker()

        if self.train_cfg.monodepth_loss_weight > 0:
            self.pretrained_monodepth = get_monodepth_model()

        if self.test_cfg.postprocessing is not None and self.test_cfg.postprocessing.is_active:
            self.postprocess_save_every = FrequencyScheduler(
                frequencies=self.test_cfg.save_every_freq,
                steps=self.test_cfg.save_every_steps,
                iters=self.test_cfg.save_at_iters,
                last_step=self.test_cfg.postprocessing.steps,
                enable_context=self.test_cfg.eval_context_views,
            )
            self.postprocess = PostProcessing3DGS(
                cfg=self.test_cfg.postprocessing,
                save_every=self.postprocess_save_every
            )
        else:
            self.postprocess = None

    @property
    def device(self):
        # Use try/except to catch StopIteration explicitly rather than letting it
        # propagate, which silently terminates PL's generator-based test loop.
        try:
            return next(self.parameters()).device
        except StopIteration:
            pass
        try:
            return next(self.buffers()).device
        except StopIteration:
            pass
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Load weights into initializer and optimizer, skipping non-learned strategies."""
        # Remove scene_trainer prefix from state dict keys if it exists
        state_dict = {k.replace("scene_trainer.", ""): v for k, v in state_dict.items()}

        prefixes = {s.split(".")[0] for s in state_dict.keys()}
        assert all([p in ["initializer", "optimizer"] for p in
                    prefixes]), f"State dict keys must start with 'initializer.' or 'optimizer.', got {prefixes}"

        if self.initializer.strategy == "learned":
            initializer_state_dict = {k[len("initializer."):]: v for k, v in state_dict.items() if
                                      k.startswith("initializer.")}
            self.initializer.load_state_dict(initializer_state_dict, strict=strict)

        if self.optimizer is not None:
            if self.optimizer.strategy == "learned":
                optimizer_state_dict = {k[len("optimizer."):]: v for k, v in state_dict.items() if
                                        k.startswith("optimizer.")}
                self.optimizer.load_state_dict(optimizer_state_dict, strict=strict)

    def get_optimized_gaussians(
            self,
            batch: BatchedExample,
            prev_output: InitializerOutput | OptimizerPreviousOutput,
            curr_iter=0,
            debug_dict=None,
            num_update_steps=None,
            disable_tqdm=False,
            **kwargs,
    ) -> OptimizerOutput:
        """
        Optimize the Gaussians for a single scene in the batch.
        Can be used for both training and testing.
        Can handle both new scenes and continuing from the replay buffer.

        Args:
            batch: BatchedExample, the input batch containing context and target views.
            prev_output: InitializerOutput | OptimizerPreviousOutput.
                If we optimize a new scene it will be of type InitializerOutput, the output from the initializer
                containing initial Gaussians and optional features.
                In this case, on_scene_start of the optimizer should transform the InitializerOutput to
                OptimizerPreviousOutput.
                If we resample from the replay buffer it will be of type OptimizerPreviousOutput, the output of each
                intermidiate update step of the optimizer, which contain gaussians and optional state.
            curr_iter: int, the current iteration.
                Should be 0 when starting a new scene.
            debug_dict: Optional[dict], a dictionary to store debug information.
            num_update_steps: Optional[int], number of update steps to perform. If None, use default from config.
        Returns:
            OptimizerOutput: The output from the optimizer containing the optimized Gaussians and renderings for
                intermediate optimization steps.
        """

        assert self.optimizer is not None, "Optimizer is not initialized."

        if num_update_steps is None:
            num_update_steps = self.get_num_update_steps()

        optimizer_input = OptimizerInput(
            context=batch["context"],  # this is full context, not iter batch
            target=batch["target"],
            prev_output=prev_output,
            renderer=self.decoder,
            num_refine=num_update_steps,
            iter_batch_size=self.scene_trainer_cfg.iter_batch_size,  # For rendering in batches
            debug_dict=debug_dict,
        )

        # Handles both new scenes (InitializerOutput) and replay buffer continuations (OptimizerPreviousOutput).
        self.optimizer.validate_input(optimizer_input)
        self.optimizer.scene_start_event_start.record()
        self.optimizer.on_scene_start(optimizer_input)
        self.optimizer.scene_start_event_end.record()
        assert isinstance(optimizer_input.prev_output, OptimizerPreviousOutput), \
            f"Should be OptimizerPreviousOutput after on_scene_start, got {type(optimizer_input.prev_output)}"

        # Initialize empty output to store intermediate and final results
        optimizer_output: OptimizerOutput = OptimizerOutput.empty(t=curr_iter)
        optimizer_output.T = num_update_steps

        # Insert the initialization into position 0 of the output lists so downstream
        # consumers (evaluation, plotting, replay buffer) can treat init uniformly with
        # the optimizer steps. No-op when there's no init render to attach (train path
        # before init renders are wired through; replay buffer continuations).
        if isinstance(prev_output, InitializerOutput):
            self._insert_init_into_output(optimizer_output, prev_output)

        # SH degree scheduling (inspired by gsplat simple_trainer):
        # sh_degree_to_use = min(step // sh_degree_interval, max_sh_degree)
        sh_degree_interval = self.scene_trainer_cfg.sh_degree_interval
        if sh_degree_interval > 0:
            max_sh_degree = int(math.sqrt(optimizer_input.prev_output.gaussians.harmonics.shape[-1])) - 1

        # Loop over update steps
        for step in tqdm(range(num_update_steps),
                         disable=(self.training or num_update_steps < 20 or DEBUG) or disable_tqdm,
                         total=num_update_steps):

            # Sample minibatch of context/target views and move to device
            optimizer_input.context, batch_idx = self.batchify_views(batch, "context", self.device)
            if batch_idx is not None:
                optimizer_output.context_index_list.append(batch_idx)
            optimizer_input.target, batch_idx = self.batchify_views(batch, "target", self.device)
            if batch_idx is not None:
                optimizer_output.target_index_list.append(batch_idx)

            # Build per-step kwargs, adding SH degree if scheduler is active
            step_kwargs = dict(kwargs)
            if sh_degree_interval > 0:
                step_kwargs["sh_degree"] = min(step // sh_degree_interval, max_sh_degree)

            # Single optimization step
            # Optimizer output is updated in place, but we return it for clarity
            optimizer_output = self.optimizer(
                step,
                optimizer_input,
                optimizer_output,
                full_context=batch["context"],
                full_target=batch["target"],
                **step_kwargs
            )
            optimizer_output.t += 1

        # Sync GPU before reading scene_start elapsed time (events were recorded before the loop).
        torch.cuda.synchronize()
        self.optimizer.scene_start_ms = self.optimizer.scene_start_event_start.elapsed_time(
            self.optimizer.scene_start_event_end
        )

        self.optimizer.on_scene_end()

        # Extract the last output (for replay buffer)
        optimizer_output.last_prev_output = optimizer_input.prev_output

        return optimizer_output

    def batchify_views(self, scene_batch, input_str, device, batch_size=None):
        """
        Sample a subset of views from the batch for the current optimization step.

        Args:
            scene_batch: Full batch containing context/target views
            input_str: "context" or "target"
            device: Target device for the subset
            batch_size: Override batch size. If None, uses config-based batch size.

        Returns:
            Tuple of (subset_batch, indices) where indices is None if no subsampling
        """
        scene_batch_split = scene_batch[input_str]

        # Determine batch size (may be randomized during training)
        if batch_size is None:
            batch_size = self._get_batch_size()
        v_all = scene_batch_split["image"].shape[1]
        if batch_size <= 0 or batch_size >= v_all:
            return scene_batch_split, None

        strategy = self.scene_trainer_cfg.opt_batch_strategy
        views_idxs = self._sample_indices(scene_batch_split, batch_size, strategy)  # [scene_batch, views_batch]
        views_batch = scene_batch_split.batchify_views(views_idxs)
        views_batch = move_data_to_device(views_batch, device)
        return views_batch, views_idxs

    def _get_batch_size(self) -> int:
        """Determine the batch size, potentially randomized during training."""
        batch_size = self.scene_trainer_cfg.opt_batch_size

        # Randomize batch size if configured (training or promoting buffer)
        if self.scene_trainer_cfg.opt_batch_size_max > 0:
            if self.training or self.promoting_buffer_sample:
                batch_size = random.randint(
                    self.scene_trainer_cfg.opt_batch_size_min,
                    self.scene_trainer_cfg.opt_batch_size_max
                )
        return batch_size

    def _sample_indices(self, batch_split, views_batch_size: int, strategy: str) -> torch.Tensor:
        """Sample a minibatch of view indices using the configured strategy.
        Uses viewpoint_stack to cycle through all views before reshuffling."""

        # Initialize or reset viewpoint stack for new epoch
        batch_split.reset_viewpoint_stack_if_needed(strategy, views_batch_size)
        viewpoint_stack = batch_split.viewpoint_stack  # [B, V]
        scene_batch, v = viewpoint_stack.shape

        views_batch_size = min(views_batch_size, v)

        if strategy in ["random", "sequential"]:
            # Take views from the front of the stack (shuffled if random)
            batch_idxs = viewpoint_stack[:, :views_batch_size]
            idx_to_remove = batch_idxs

        elif strategy == "neighbors":
            # Use first view in stack as center, select its neighbors
            extrinsics = batch_split["extrinsics"]
            if extrinsics.ndim == 4:  # [B, V, 4, 4]
                assert extrinsics.shape[0] == 1, "Batch size must be 1 for neighbor sampling"
                extrinsics = extrinsics[0]

            center_idx = viewpoint_stack[0, 0]
            batch_idxs = self._get_neighbor_indices(extrinsics, center_idx, views_batch_size)
            idx_to_remove = torch.tensor([[center_idx]])  # Only remove center from stack
        elif strategy == "fps":
            # FPS on camera positions of the remaining views in the stack
            extrinsics = batch_split["extrinsics"]  # [B, V_total, 4, 4]
            B = extrinsics.shape[0]
            batch_arange = torch.arange(B, device=self.device)[:, None]
            stack_positions = extrinsics[batch_arange, viewpoint_stack][:, :, :3, 3]  # [B, V_stack, 3]
            fps_local_idxs = farthest_point_sample(stack_positions, views_batch_size,
                                                   first_idx_strategy="random")  # [B, K]
            batch_idxs = viewpoint_stack[batch_arange, fps_local_idxs]  # [B, K]
            idx_to_remove = batch_idxs
        else:
            raise ValueError(f"Unknown opt_batch_strategy: {strategy}")

        # Remove used indices from the stack, preserving order between the views separately for each batch.
        remove_mask = (viewpoint_stack.unsqueeze(-1) == idx_to_remove.unsqueeze(1)).any(-1)  # [B, V]
        batch_split.viewpoint_stack = viewpoint_stack[~remove_mask].view(scene_batch, -1)  # [B, V_used]

        return batch_idxs

    def _get_neighbor_indices(self, extrinsics, center_idx, batch_size: int) -> torch.Tensor:
        """Get indices of nearest neighbor views based on camera pose distance."""
        combined_metric = self.calc_extrinsics_dist(center_idx, extrinsics)
        return torch.argsort(combined_metric)[:batch_size].unsqueeze(0)  # [1, K]

    @staticmethod
    def calc_extrinsics_dist(center_idx, extrinsics):
        """Combined position + rotation distance from a center view to all views. Returns [V]."""
        rotations = extrinsics[:, :3, :3]  # [V, 3, 3]
        # Calculate camera center as -R^T * t
        translation = extrinsics[:, :3, [3]]  # [V, 3, 1]
        poses = -rotations.transpose(1, 2) @ translation  # [V, 3, 1]
        center_pose = poses[center_idx]  # [3, 1]
        # Calculate Euclidean distances to the center view
        dists = torch.norm(poses - center_pose.unsqueeze(0), dim=1)[0]  # [V]
        # Calculate angular differences to the center view
        center_rot = extrinsics[center_idx, :3, :3]  # [3, 3]
        # Compute rotation difference
        rot_diffs = torch.matmul(rotations, center_rot.transpose(0, 1))  # [V, 3, 3]
        # Compute angles from rotation matrices
        cos_angles = (rot_diffs[:, 0, 0] + rot_diffs[:, 1, 1] + rot_diffs[:, 2, 2] - 1) / 2  # [V]
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)  # Numerical stability
        angles = torch.acos(cos_angles)  # [V]
        # Combine distance and angle into a single metric
        combined_metric = dists + angles  # [V]
        return combined_metric

    def get_num_update_steps(self) -> int:
        """Return number of optimizer steps, randomly sampled during training if train_max_refine is set."""
        if self.training and self.scene_trainer_cfg.train_max_refine > 0:
            num_updates = random.randint(
                self.scene_trainer_cfg.train_min_refine,
                self.scene_trainer_cfg.train_max_refine
            )
        else:
            num_updates = self.scene_trainer_cfg.num_update_steps
        return num_updates

    def get_init_gaussians(self, batch, is_training: bool, **kwargs) -> InitializerOutput:
        """Run the initializer to produce Gaussians from context views, with optional sliding window.

        Gradients are disabled when not training so the init model is frozen during refine-only runs.
        """
        window_size = self.train_cfg.train_window_size if is_training else self.test_cfg.inference_window_size
        with torch.set_grad_enabled(is_training):
            if window_size is not None:
                initializer_output = self.init_gaussians_with_window(batch, window_size, **kwargs)
            else:
                # In some cases we might want to pass the target as well
                # (e.g., to manipulate the poses in colmap dataset)
                initializer_output = self.initializer(batch["context"], scene=batch["scene"],
                                                      target=batch["target"], device=self.device, **kwargs)
            return initializer_output

    def init_gaussians_with_window(self, batch, window, **kwargs) -> InitializerOutput:
        """Run the initializer in a sliding window over views, then combine the per-window Gaussians."""
        assert self.initializer.cfg.per_view, "Sliding window initialization only supports per-pixel initialization."
        b, v, _, h, w = batch["context"]["image"].shape
        assert window > 0

        window_indices = sliding_window_indices(v, window, 0)
        all_gaussians = []
        all_states = []
        all_pred_depths = []
        for indices in window_indices:

            start, end = indices
            view_indices = torch.arange(start, end, device=batch["context"]["image"].device).unsqueeze(0).expand(b, -1)
            curr_window_input = batch["context"].batchify_views(view_indices)

            initializer_output = self.initializer(curr_window_input, **kwargs)

            curr_gaussians = initializer_output.gaussians  # Gaussians object with tensors shape [B, G, D1, ...]
            curr_features = initializer_output.features  # [BV, C, H, W]

            all_gaussians.append(curr_gaussians)
            all_states.append(curr_features)
            if initializer_output.depths is not None:
                all_pred_depths.append(initializer_output.depths)

        # merge all gaussians
        def combine_gaussians_attribute(attr_name):
            all_attr = [getattr(g, attr_name) for g in all_gaussians[:-1]]
            last_g = all_gaussians[-1]
            last_g_attr = getattr(last_g, attr_name)
            # handle the overlapping in the last window
            if v % window != 0:
                x = v % window
                b, vhw, *d = last_g_attr.shape
                if self.initializer.cfg.per_pixel:
                    # per-pixel initialization
                    h_gaussians = h // self.initializer.cfg.latent_downsample
                    w_gaussians = w // self.initializer.cfg.latent_downsample
                else:
                    raise NotImplementedError
                last_g_attr = last_g_attr.view(b, window, h_gaussians, w_gaussians, *d)  # [B, V, H, W, ...]
                last_g_attr = last_g_attr[:, -x:, ...]  # [B, x, H, W, ...]
                last_g_attr = last_g_attr.view(b, -1, *d)  # [B, x*H*W, ...]
            all_attr.append(last_g_attr)
            return torch.cat(all_attr, dim=1)

        gaussians = Gaussians(
            means=combine_gaussians_attribute('means'),
            covariances=combine_gaussians_attribute('covariances'),
            harmonics=combine_gaussians_attribute('harmonics'),
            opacities=combine_gaussians_attribute('opacities'),
            scales=combine_gaussians_attribute('scales'),
            rotations=combine_gaussians_attribute('rotations'),
            rotations_unnorm=combine_gaussians_attribute('rotations_unnorm'),
        )

        # Collect condition features for the optimizer (only needed if optimizer is active)
        if self.scene_trainer_cfg.num_update_steps > 0:
            out = []
            is_ori_feature = True  # set by first window; True = [BV,C,H,W], False = [BVHW,C]
            for i in range(len(all_states)):
                # Assuming no overlap between windows
                curr = all_states[i]
                if curr.dim() == 4:
                    # [BV, C, H, W]
                    curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                    is_ori_feature = True
                elif curr.dim() == 2:
                    # [BVHW, C]
                    curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b,
                                     h=h // self.initializer.cfg.latent_downsample,
                                     w=w // self.initializer.cfg.latent_downsample,
                                     )
                    is_ori_feature = False
                else:
                    raise NotImplementedError

                # Only need to handle the overlaping in the last window
                if i == len(all_states) - 1 and v % window != 0:
                    # last window with overlap
                    x = v % window
                    curr = curr[:, -x:, ...]
                out.append(curr)

            # concat
            if is_ori_feature:
                concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                concat = rearrange(concat, "b v c h w -> (b v) c h w")
            else:
                concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                concat = rearrange(concat, "b v h w c -> (b v) c h w")

            condition_features = concat
        else:
            condition_features = None

        return InitializerOutput(gaussians=gaussians,
                                 features=condition_features,
                                 depths=all_pred_depths)

    def debugging(self, optimizer_output, output_path: Path, scene_name: str):  # TODO (release): remove in public code

        # Debugging reprojection errors
        # if 'reprojection_error' in visualization_dump:
        #     self.debugging_reprojection_error(visualization_dump)

        assert "deltas" in optimizer_output.info, "Deltas not found in optimizer output info."
        assert "grads" in optimizer_output.info, "Grads not found in optimizer output info."
        assert "normalized_grads" in optimizer_output.info, "Normalized grads not found in optimizer output info."
        # assert "learning_rates" in optimizer_output.info, "Learning rates not found in optimizer output info."

        # Unpack Optimizer output
        deltas_list: list[dict[str, Tensor]] = optimizer_output.info["deltas"]

        grads_raw_list: list[dict[str, Tensor]] = optimizer_output.info["grads"]
        normalized_grads_list: list[dict[str, Tensor]] = optimizer_output.info["normalized_grads"]

        # Get PSNR list
        module_name = self.optimizer.__class__.__name__.lower()
        psnr_list = self.test_step_outputs_target[f"{module_name}_psnr"][0]  # list of psnr for target views per scene

        # Get iterations list
        iterations_list = self.optimizer.save_every.get_iterations(len(psnr_list))

        means2d_list = [render.means2d for render in optimizer_output.target_render_list]
        radii_list = [render.radii for render in optimizer_output.target_render_list]

        debugging_invisible_gaussians(
            optimizer_output.gaussian_list,
            grads_raw_list,
            normalized_grads_list,
            means2d_list,
            radii_list,
            psnr_list,
            iterations_list,
            output_path / module_name,
            scene_name
        )

        # Remove init.
        psnr_list = psnr_list[1:]
        iterations_list = iterations_list[1:]

        if "states_norms" in optimizer_output.info:
            states_norms_list: list[Tensor] = optimizer_output.info["states_norms"]
            debugging_convergence(
                deltas_list,
                states_norms_list,
                grads_raw_list,
                normalized_grads_list,
                psnr_list,
                iterations_list,
                output_path / module_name,
                scene_name
            )


    def _insert_init_into_output(
            self,
            optimizer_output: OptimizerOutput,
            init_output: InitializerOutput,
    ) -> None:
        """Prepend the init render/gaussians at position 0 of the optimizer_output lists.

        No-op if no init render exists yet (train path before init renders are wired
        through). Otherwise inserts gaussians plus whichever of context/target renders
        are populated. detach_and_cpu mirrors the per-step append policy.
        """
        if init_output.context_render is None and init_output.target_render is None:
            return

        optimizer_output.gaussian_list.insert(0, init_output.gaussians)
        if init_output.context_render is not None:
            optimizer_output.context_render_list.insert(
                0, init_output.context_render, detach_and_cpu=not self.training
            )
        if init_output.target_render is not None:
            optimizer_output.target_render_list.insert(
                0, init_output.target_render, detach_and_cpu=not self.training
            )

    def init_gaussians_and_render(
            self, batch, visualization_dump,
            render_context: bool, render_target: bool, grad_enabled: bool,
            **kwargs,
    ) -> InitializerOutput:
        """Run the initializer and optionally render its output to context/target views.

        Used in both training (grad_enabled=True, outputs stay on GPU so the init-loss term
        can backward through them) and test (grad_enabled=False, outputs moved to CPU to save
        memory since they're only consumed for evaluation/saving).
        """

        # run initializer
        with self.benchmarker.time("initializer"):
            init_output: InitializerOutput = self.get_init_gaussians(batch, is_training=grad_enabled, **kwargs)

        # to_cpu freezes the render off the GPU for evaluation/saving; with grads enabled we
        # keep it on GPU so the init-loss term can backward through it.
        to_cpu = not grad_enabled

        with torch.set_grad_enabled(grad_enabled):
            for input_str, should_render in (
                ("context", render_context),
                ("target", render_target),
            ):
                attr = f"{input_str}_render"
                if not should_render or getattr(init_output, attr) is not None:
                    continue
                views = batch[input_str]
                h, w = views["image"].shape[-2:]
                rendered = self.decoder.forward_batch(
                    init_output.gaussians.to(batch["target"]["image"].device),
                    batch, (h, w),
                    input_str=input_str,
                    to_cpu=to_cpu,
                    iter_batch_size=self.scene_trainer_cfg.iter_batch_size,
                )
                # TODO Naama: should we make it a render list as in OptimizerOutput and then the flow will be more unified?
                setattr(init_output, attr, rendered)

        return init_output

    def test_postprocess_gaussians(self, batch, gaussians, visualization_dump) -> OptimizerOutput | None:
        """Run optional post-processing on the final Gaussians. Returns None if disabled."""
        postprocess_output = None
        if self.postprocess is not None:
            postprocess_output = self.postprocess.apply(
                batch,
                gaussians=gaussians,
                decoder=self.decoder,
                visualization_dump=visualization_dump,
                iter_batch_size=self.scene_trainer_cfg.iter_batch_size,
                batchify_fn=lambda b, input_str: self.batchify_views(
                    b, input_str, self.device,
                ),
            )

        return postprocess_output


def sliding_window_indices(N, x, y):
    """Return [start, end] pairs for a sliding window of size x with overlap y over N views.
    The last window is always [N-x, N] to cover any remainder."""
    indices = []
    start = 0
    while start + x < N:  # Ensure the last window is not processed here
        end = min(start + x, N)
        indices.append([start, end])
        start += (x - y)  # Move the start by the window size minus overlap

    # Append the last window [N-x, N]
    indices.append([N - x, N])

    return indices

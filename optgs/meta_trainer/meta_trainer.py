import json
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, runtime_checkable, Protocol, Literal

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import wandb
from einops import rearrange, repeat, pack
from jaxtyping import Float
from lightning_fabric.utilities import rank_zero_only
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import optim, Tensor, nn
from tqdm import tqdm

from optgs.config import RootCfg, MetaTrainerCfg
from optgs.dataset import DatasetCfg
from optgs.dataset.data_module import get_data_shim
from optgs.dataset.data_types import BatchedExample, BatchedViews
from optgs.evaluation.depth_metrics import compute_depth_errors
from optgs.evaluation.metrics import compute_psnr, compute_ssim, compute_rgb_metrics
from optgs.loss import Loss
from optgs.loss.loss_depth_smooth import get_smooth_loss
from optgs.loss.loss_stability import LossStability
from optgs.meta_trainer.replay_buffer import GaussianEpisodeEntry
from optgs.misc.LocalLogger import LocalLogger, LOG_PATH
from optgs.misc.batchify import batched_select
from optgs.misc.benchmarker import Benchmarker
from optgs.misc.console import rule, warn
from optgs.misc.general_utils import SkipBatchException
from optgs.misc.image_io import prep_image, save_video, save_image
from optgs.misc.io import CustomPath
from optgs.misc.stablize_camera import render_stabilization_path
from optgs.misc.step_tracker import StepTracker
from optgs.model.colmap_utils.convert_to_colmap import save_opencv_camera
from optgs.model.colmap_utils.extract_sparse_view_extrinsics import extract_sparse_images_bin
from optgs.model.decoder import get_decoder
from optgs.model.ply_export import save_gaussian_ply
from optgs.paths import DEBUG
from optgs.scene_trainer.initializer.initializer import InitializerOutput, Initializer
from optgs.scene_trainer.optimizer.optimizer import OptimizerPreviousOutput, OptimizerOutput, Optimizer
from optgs.scene_trainer.postprocessing import PostProcessing3DGS
from optgs.scene_trainer.scene_trainer import SceneTrainer  # Use existing SceneTrainer
from optgs.scene_trainer.scene_trainer_cfg import SceneTrainerCfg, MetaOptimizerCfg, TestCfg, TrainCfg
from optgs.visualization.annotation import add_label
from optgs.visualization.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from optgs.visualization.camera_trajectory.wobble import generate_wobble, generate_wobble_transformation
from optgs.visualization.color_map import apply_color_map_to_image
from optgs.visualization.layout import hcat, vcat, add_border
from optgs.visualization.validation_in_3d import render_projections
from optgs.visualization.vis_depth import viz_depth_tensor

try:
    from bitsandbytes.optim import AdamW8bit
except:
    pass

try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
            self,
            t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


slurm_id_logged = False
debug_count = 0


class _SkipStepException(Exception):
    """Raised inside meta_training_step to signal that this step should be
    skipped.  Caught in training_step, which then does a single all_reduce so
    every rank skips together — preventing NCCL hangs."""
    pass


class MetaTrainer(LightningModule):
    """
    Meta-level trainer that handles the outer loop of meta-learning.

    This class focuses on:
    - Meta-level training loop and replay buffer management
    - Delegating scene-level optimization to the existing SceneTrainer
    - Meta-optimization of the SceneTrainer's parameters
    """

    meta_optimizer_cfg: MetaOptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    logger: Optional[WandbLogger]
    scene_trainer_cfg: SceneTrainerCfg
    losses: nn.ModuleList
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]
    meta_trainer_cfg: MetaTrainerCfg

    def __init__(
            self,
            cfg: RootCfg,
            meta_optimizer_cfg: MetaOptimizerCfg,
            test_cfg: TestCfg,
            train_cfg: TrainCfg,
            scene_trainer_cfg: SceneTrainerCfg,
            losses: list[Loss],
            step_tracker: StepTracker | None,
            eval_data_cfg: Optional[DatasetCfg] = None,
    ) -> None:
        super().__init__()
        self.meta_optimizer_cfg = cfg.meta_optimizer
        self.test_cfg = cfg.meta_trainer.test
        self.train_cfg = cfg.meta_trainer.train
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg
        self.scene_trainer_cfg = cfg.scene_trainer
        self.meta_trainer_cfg = cfg.meta_trainer

        # Create the existing SceneTrainer that contains all the scene-level logic
        # This includes the initializer, optimizer, decoder, and get_optimized_gaussians method
        self.scene_trainer = SceneTrainer(
            test_cfg=test_cfg,
            train_cfg=train_cfg,
            scene_trainer_cfg=scene_trainer_cfg,
            decoder=get_decoder(cfg.scene_trainer.decoder, cfg.dataset),
            step_tracker=step_tracker,
            eval_data_cfg=eval_data_cfg,
        )

        self.initializer_data_shim = get_data_shim(self.scene_initializer)
        self.losses = nn.ModuleList(losses)

        # Testing utilities
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs_target = defaultdict(list)
            self.test_step_outputs_context = defaultdict(list)

        if cfg.mode == "train" and self.train_cfg.use_replay_buffer and self.scene_trainer_cfg.num_update_steps > 0:
            assert self.scene_optimizer is not None
            assert self.scene_optimizer.strategy == "learned"

            if getattr(self.scene_optimizer.cfg, 'concat_init_state', False):
                raise NotImplementedError("Replay buffer with concat_init_state is not supported")
            if getattr(self.scene_optimizer.cfg, 'replace_init_state', False):
                raise NotImplementedError("Replay buffer with replace_init_state is not supported")
            from optgs.meta_trainer.replay_buffer import EpisodeReplayBuffer
            self.buffer = EpisodeReplayBuffer(self.train_cfg.replay_buffer_cfg)
        else:
            self.buffer = None

        self._use_dataloader_batch = True  # default
        self._new_scenes_cnt = -1
        self.gaussian_timestep_list = []
        self.gaussian_timestep_table = wandb.Table(columns=["epoch", "gaussian_timestep", "count"])

        self.promoting_buffer_sample = False

        if self.training:
            self._inner_iteration_data = []  # Store data for logging inner iterations psnr across meta iterations

    # ==================== Lightning Hooks ====================

    def on_before_batch_transfer(self, batch: BatchedExample, dataloader_idx: int) -> BatchedExample:
        """Decide before device transfer whether this step should draw from the replay buffer or the dataloader."""
        # Decide whether we'll use the buffer
        if self.training and self.buffer is not None and self.buffer.should_sample():
            self._use_dataloader_batch = False
        else:
            self._use_dataloader_batch = True
        return batch

    def on_after_batch_transfer(self, batch: BatchedExample, dataloader_idx: int) -> BatchedExample:
        """Convert raw context/target dicts into typed BatchedViews after the batch lands on the device."""
        batch["context"] = BatchedViews.from_dict(batch["context"])
        batch["target"] = BatchedViews.from_dict(batch["target"])
        return batch

    def will_move_minibatch_to_device(self, batch):
        """True when only a sub-batch of views needs to move to device (non-learned init + opt_batch_size < V)."""
        # TODO Naama: check if used
        # When we sabsample a minibatch, we can move only it to device
        return (self.scene_initializer.strategy == "nonlearned" and
                self.scene_optimizer is not None and
                self.scene_trainer_cfg.opt_batch_size != batch["context"]["image"].shape[1])

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Only transfer if we're going to use this batch
        if self.training:
            if self._use_dataloader_batch:
                should_move = True  # move if using dataloader batch
                # Also, if the initializer is not learned and the optimizer uses inner batch size, then we also don't want to
                # move the batch
                # if self.will_move_minibatch_to_device(batch):
                #     should_move = False
            else:
                should_move = False  # don't move if using buffer sample (we'll move it in the buffer sampling code)
        else:
            should_move = True  # always move during validation and testing

        if should_move:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            return batch  # Don't move — we're going to ignore this batch anyway

    def on_save_checkpoint(self, checkpoint):
        # Remove the monodepth_model weights from the checkpoint
        if 'state_dict' in checkpoint:
            keys_to_remove = [k for k in checkpoint['state_dict'] if k.startswith('pretrained_monodepth')]
            for k in keys_to_remove:
                del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        # Override scheduler total_steps to match current max_steps so LR doesn't
        # hit 0 early when resuming for extended training.
        for scheduler in checkpoint.get("lr_schedulers", []):
            saved_steps = scheduler.get("total_steps")
            if saved_steps is not None and saved_steps != self.trainer.max_steps:
                print(
                    f"Resuming with extended training: scheduler total_steps "
                    f"{saved_steps} → {self.trainer.max_steps}. "
                    f"LR schedule will be stretched, not restarted from scratch."
                )
                scheduler["total_steps"] = self.trainer.max_steps

    def on_train_epoch_start(self):
        """Handle epoch start for scene-based training."""
        if hasattr(self.scene_trainer, 'on_train_epoch_start'):
            return self.scene_trainer.on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.global_rank == 0:
            if self.buffer is not None:
                print(f"Buffer size: {len(self.buffer)}")

            if self.logger is not None and isinstance(self.logger, WandbLogger):
                # counts = Counter(self.gaussian_timestep_list)
                # for eid, c in counts.items():
                #     self.gaussian_timestep_table.add_data(self.current_epoch, eid, c)

                # wandb.log({"replay_buffer/event_counts_table": self.gaussian_timestep_table})
                # log also histogram
                wandb.log({"replay_buffer/gaussian_timestep_histogram": wandb.Histogram(self.gaussian_timestep_list)})

        if self.buffer is not None:
            self.buffer.clear()
        self.gaussian_timestep_list = []

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    def on_test_epoch_start(self):
        """Handle test epoch start."""
        if hasattr(self.scene_trainer, 'on_test_epoch_start'):
            return self.scene_trainer.on_test_epoch_start()

    def on_test_epoch_end(self):
        """Handle test epoch end."""
        if hasattr(self.scene_trainer, 'on_test_epoch_end'):
            return self.scene_trainer.on_test_epoch_end()

    def on_test_end(self) -> None:
        out_dir = self.test_cfg.output_path

        # Merge sub-module benchmarkers so all tags land in one file.
        # scene_trainer.benchmarker holds "initializer" (wall-clock, from init_gaussians_and_render).
        # optimizer.benchmarker is unused here — decoder/optimizer split is recorded per-scene
        # in meta_test_step via benchmarker.record() directly on self.benchmarker.
        self.benchmarker.merge(self.scene_trainer.benchmarker)

        # saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for output_dict, input_str in zip([self.test_step_outputs_context, self.test_step_outputs_target],
                                              ["context", "target"]):
                for metric_name, metric_scores in output_dict.items():
                    metric_scores = torch.tensor(metric_scores)  # [scenes, update_steps]
                    if metric_scores.numel() == 0:
                        continue
                    metric_scores = metric_scores.float()  # [scenes, update_steps]
                    update_step_scores = metric_scores.mean(dim=0).tolist()  # [update_steps]
                    # saved_scores[f"{input_str}_{metric_name}"] = update_step_scores[-1]
                    print(input_str, metric_name, update_step_scores)
                    with (out_dir / "metrics" / f"{input_str}_{metric_name}.json").open("w") as f:
                        json.dump(metric_scores.tolist(), f)

            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "metrics" / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "metrics" / "peak_memory.json")
            self.benchmarker.summarize()

    # ==================== Training ====================

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move a batch dict to the current device."""

        def move_tensor(x):
            if isinstance(x, Tensor):
                return x.to(self.device)
            elif isinstance(x, dict):
                return {k: move_tensor(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [move_tensor(v) for v in x]
            return x

        return move_tensor(batch)

    def training_step(self, batch, batch_idx):
        """
        This is a meta trainer class. Each training step and test step corresponds to training on one scene.
        We delegate the actual training to meta_training_step and meta_test_step.
        The loop over inner training steps (training within a specific scene) is performed in
        self.get_optimized_gaussians.
        Each inner iteration is done by calling the forward call of the optimizer.
        """
        # DDP-safe skip: all ranks must call all_reduce together, so we do it
        # here in training_step (which is always called on every rank) rather
        # than inside meta_training_step (which may return early on only one rank).
        #
        # Each rank sets skip_flag=1 if it wants to skip, 0 otherwise.
        # After MAX all_reduce, every rank sees 1 if *any* rank wants to skip,
        # and all return zero loss together — keeping NCCL collectives in sync.
        is_dist = dist.is_available() and dist.is_initialized()
        wants_skip = torch.zeros(1, device=self.device)

        try:
            loss = self.meta_training_step(batch, batch_idx)
        except _SkipStepException:
            wants_skip.fill_(1)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        if is_dist:
            dist.all_reduce(wants_skip, op=dist.ReduceOp.MAX)
            if wants_skip.item() > 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

    def meta_training_step(self, scene_batch, batch_idx):
        """One meta-training step: initialize Gaussians, run optimizer refinement, compute loss, optionally push to replay buffer."""
        batch_size, init_target_render_output = None, None
        optimizer_output: OptimizerOutput | None = None

        # Prepare input (from dataloader or replay buffer)
        if self._use_dataloader_batch:
            # Use new batch from dataloader
            scene_batch: BatchedExample = self.initializer_data_shim(scene_batch)

            # Get initialization Gaussians
            try:
                init_output = self.get_init_gaussians(scene_batch, is_training=self.scene_trainer_cfg.train_scene_init)
            except SkipBatchException as e:
                self.log("skip_zero_gaussians_batch", 1, prog_bar=True)
                if self.global_rank == 0:
                    warn(f"Skipping batch {batch_idx} due to {e}. t meta {self.global_step}")
                raise _SkipStepException(f"SkipBatch(init): {e}")

            prev_output = init_output

            # Render the init gaussians for loss calculation (only when training the initializer)
            if self.scene_trainer_cfg.train_scene_init:
                batch_size, init_target_render_output = (
                    self.train_render_output_for_init_gaussians(scene_batch, init_output.gaussians))

            curr_inner_iter = 0
            self._new_scenes_cnt += 1
        else:
            # Resample from replay buffer intermediate optimized Gaussians (only when training the optimizer)
            assert self.scene_trainer_cfg.train_scene_opt
            assert not self.scene_trainer_cfg.train_scene_init

            # Sample from buffer
            gaussian_episode_entry: GaussianEpisodeEntry = self.buffer.sample(device=self.device,
                                                                              leave_batch_fn=self.will_move_minibatch_to_device)

            # Adjust sample
            scene_batch = gaussian_episode_entry.batch
            prev_output = OptimizerPreviousOutput(gaussians=gaussian_episode_entry.gaussians,
                                                  state=gaussian_episode_entry.state)
            curr_inner_iter = gaussian_episode_entry.t

            # Simulate init_output for logging (no training of the init_model in this case)
            init_output = InitializerOutput(gaussians=gaussian_episode_entry.gaussians)

        # Log the current timestep for analysis
        self.gaussian_timestep_list.append(curr_inner_iter)

        # Optimize the gaussians
        if self.scene_trainer.optimizer is not None and self.scene_trainer_cfg.train_scene_opt:
            # During optimization, we render the context and target images for:
            # 1. error/gradients calculation
            # 2. loss calculation
            # Although it is not necessary, we also render the init target image again for loss calculation
            # In the case or training both initializer and optimizer, this is redundant.

            try:
                optimizer_output: OptimizerOutput = self.get_optimized_gaussians(scene_batch, prev_output,
                                                                                 curr_iter=curr_inner_iter)
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                self.log("skip_oom_batch", 1, prog_bar=True)
                print(
                    f"[rank {self.global_rank}]  skipping batch {batch_idx} t meta {self.global_step} t inner {curr_inner_iter}: {e}")
                torch.cuda.empty_cache()
                raise _SkipStepException("OOM")
            except SkipBatchException as e:
                self.log("skip_nan_batch", 1, prog_bar=True)
                if self.global_rank == 0:
                    warn(f"Skipping batch {batch_idx} due to {e}. "
                         f"t meta {self.global_step} t inner {curr_inner_iter}")
                raise _SkipStepException(f"SkipBatch(opt): {e}")
            curr_inner_iter = optimizer_output.t

            if optimizer_output.last_prev_output.state.state is not None:
                state_norm = optimizer_output.last_prev_output.state.state.norm(dim=1).mean()
                self.log("info/state_norm", state_norm)

        # Compute and log loss.
        init_gaussians = init_output.gaussians

        try:
            total_loss = self.train_calc_total_loss(scene_batch, optimizer_output, init_gaussians,
                                                    init_target_render_output, init_output.depths)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            self.log("skip_oom_batch", 1, prog_bar=True)
            print(
                f"[rank {self.global_rank}] OOM: {e}, skipping batch {batch_idx} t meta {self.global_step} t inner {curr_inner_iter} num of inner {len(optimizer_output.gaussian_list)}")
            torch.cuda.empty_cache()
            raise _SkipStepException("OOM")

        # More logging
        if optimizer_output is not None:
            last_gaussians = optimizer_output.gaussian_list[-1]
        else:
            last_gaussians = init_gaussians
        self.train_logging(scene_batch, optimizer_output, last_gaussians, total_loss)

        # Check for NaN loss
        # Skipping pushing to the replat buffer
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.log("skip_nan_batch", 1, prog_bar=True)
            if self.global_rank == 0:
                warn(f"Skipping batch {batch_idx} due to NaN loss. "
                     f"t meta {self.global_step} t inner {optimizer_output.t}")
            raise _SkipStepException("NaN/Inf loss")

        # Push back to buffer
        if self.buffer is not None and self.buffer.should_push(new_sample=self._use_dataloader_batch,
                                                               t=curr_inner_iter):
            push = True
            if self.train_cfg.replay_buffer_cfg.simulate_ahead:
                min_steps = self.train_cfg.replay_buffer_cfg.simulate_ahead_min_steps
                cfg_max_steps = self.train_cfg.replay_buffer_cfg.simulate_ahead_max_steps

                if self.train_cfg.replay_buffer_cfg.simulate_ahead_grow > 0:
                    t_meta = self.global_step
                    T_grow = self.train_cfg.replay_buffer_cfg.simulate_ahead_grow
                    max_steps = min_steps + (cfg_max_steps - min_steps) * min(1.0, t_meta / T_grow)
                    max_steps = int(max_steps)
                else:
                    max_steps = cfg_max_steps

                if min_steps == max_steps:
                    steps = min_steps
                else:
                    steps = np.random.randint(low=min_steps, high=max_steps + 1)
                with torch.no_grad():
                    # Set eval mode
                    self.eval()
                    self.scene_optimizer.save_every.set_all_tags(False)
                    self.promoting_buffer_sample = True

                    try:
                        optimizer_output = self.get_optimized_gaussians(scene_batch, optimizer_output.last_prev_output,
                                                                        curr_iter=optimizer_output.t,
                                                                        num_update_steps=steps,
                                                                        disable_tqdm=True)
                        last_gaussians = optimizer_output.last_prev_output.gaussians
                    # catching multiple errors
                    except (ValueError, SkipBatchException) as e:
                        warn(f"Skipping pushing batch {batch_idx} to buffer due to {e}.")
                        push = False
                    self.train()
                    self.scene_optimizer.save_every.set_all_tags(True)
                    self.promoting_buffer_sample = False

                    # assert len(optimizer_output.target_render_list) == 0  # no rendering needed

            if optimizer_output.last_prev_output.state.state is not None:
                with torch.no_grad():
                    state_norm = optimizer_output.last_prev_output.state.state.norm(dim=1).mean()
                if state_norm > 500:
                    warnings.warn(f"Pushing sample norm state {state_norm} {optimizer_output.t} {self.global_step}")
            if push:
                self.buffer.push(GaussianEpisodeEntry(t=optimizer_output.t,
                                                      batch=scene_batch,
                                                      gaussians=last_gaussians,
                                                      state=optimizer_output.last_prev_output.state,
                                                      id=self._new_scenes_cnt), to_cpu=True)

                self.log("replay_buffer/size", len(self.buffer.buffer))
                if self.train_cfg.replay_buffer_cfg.simulate_ahead:
                    self.log("replay_buffer/simulate_ahead", steps)
                self.log("replay_buffer/stored_step", optimizer_output.t)

        return total_loss

    def train_logging(self, batch, optimizer_output, gaussians, total_loss):
        self.log("loss/total", total_loss)
        if (
                self.global_rank == 0
                and (self.global_step % self.train_cfg.print_log_every_n_steps == 0 or total_loss > 5)
        ):
            print(
                f"train step {self.global_step}; "
                f"scene_name = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}; "
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # log gaussians scales
        if self.scene_trainer_cfg.num_update_steps > 0 and "deltas" in optimizer_output.info:
            delta_means = [deltas["means"] for deltas in optimizer_output.info["deltas"]]
            delta_scales = [deltas["scales"] for deltas in optimizer_output.info["deltas"]]

            for i in range(len(delta_means)):
                self.log(f"update{i}/delta_means_min", delta_means[i].abs().min().item())
                self.log(f"update{i}/delta_means_mean", delta_means[i].abs().mean().item())
                self.log(f"update{i}/delta_means_max", delta_means[i].abs().max().item())

            for i in range(len(delta_scales)):
                self.log(f"update{i}/delta_scales_min", delta_scales[i].abs().min().item())
                self.log(f"update{i}/delta_scales_mean", delta_scales[i].abs().mean().item())
                self.log(f"update{i}/delta_scales_max", delta_scales[i].abs().max().item())

        self.log("info/gaussian_scale_min", gaussians.scales.min().item())
        self.log("info/gaussian_scale_max", gaussians.scales.max().item())
        self.log("info/gaussian_scale_mean", gaussians.scales.mean().item())

        # log gaussians opacities
        self.log("info/gaussian_opacity_min", gaussians.opacities.min().item())
        self.log("info/gaussian_opacity_max", gaussians.opacities.max().item())
        self.log("info/gaussian_opacity_mean", gaussians.opacities.mean().item())

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        if self.global_step == 5 and self.global_rank == 0:
            os.system("nvidia-smi")
        global slurm_id_logged
        if self.global_rank == 0 and not slurm_id_logged:
            print('slurm id:', os.environ.get('SLURM_JOB_ID'))
            slurm_id_logged = True

    def compute_losses(self, gaussians, i, num_output, render_output, curr_gt_rgb, valid_depth_mask,
                       error_idx=None, all_gt_rgb=None, tag="target"):
        """Compute weighted sum of all configured losses at one optimizer step.

        curr_gt_rgb [B, V_rendered] is pre-indexed to the views the optimizer rendered;
        all_gt_rgb [B, V_all] is the full GT passed only to losses that need every view (e.g. LossSh0).
        intermediate_loss_weight discounts earlier refinement steps.
        """
        # curr_gt_rgb: [B, V_rendered, C, H, W] — already narrowed to the views the optimizer rendered
        # all_gt_rgb:  [B, V_all, C, H, W]      — full GT; passed to losses that need all views (e.g. LossSh0)
        if all_gt_rgb is None:
            all_gt_rgb = curr_gt_rgb
        total_loss = 0
        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (num_output - 1 - i)

        gt_rgb, pred_rgb, valid_depth_mask = Loss.extract_pred_gt(
            curr_gt_rgb, render_output, error_idx, valid_depth_mask
        )

        for loss_fn in self.losses:
            if isinstance(loss_fn, LossStability):
                # Stability loss is applied on all intermediate outputs
                # Will be calculated outside of the inner steps loop
                continue
            # TODO Naama review
            loss = loss_fn(
                render_output,
                gaussians,
                self.global_step,
                gt_rgb=gt_rgb,
                pred_rgb=pred_rgb,
                gt_image=all_gt_rgb,
                valid_depth_mask=valid_depth_mask,
                l1_loss=self.train_cfg.l1_loss,
                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                half_res_lpips=self.train_cfg.half_res_lpips_loss,
            )

            loss_tag = f"{tag}_" + loss_fn.name
            loss_tag += f"_{i + 1}" if i > 0 else ""
            self.log(f"loss/{loss_tag}", loss)

            total_loss += curr_loss_weight * loss

        return total_loss

    def train_calc_total_loss(self, batch, optimizer_output: OptimizerOutput | None, init_gaussians,
                              init_target_render_output, pred_depths):
        """Accumulate total training loss: init + optimizer steps + depth + monodepth losses."""
        total_loss = 0
        valid_depth_mask = None

        target_gt_rgb = batch["target"]["image"]
        t = optimizer_output.t if optimizer_output is not None else 0

        # Log and calculate loss of init
        if self.scene_trainer_cfg.train_scene_init:
            total_loss += self._calc_init_loss(init_gaussians, init_target_render_output, target_gt_rgb,
                                               valid_depth_mask)
        else:
            # Still log init psnr, but init_target_render_output is None
            self._log_init_metrics_from_optimizer(batch, optimizer_output)

        # Log and calculate loss of intermediate outputs during refinement
        if self.scene_trainer_cfg.train_scene_opt:
            total_loss += self._calc_opt_loss(batch, optimizer_output, t, valid_depth_mask)

        # More loss on the last prediction
        assert self.scene_trainer_cfg.train_scene_init ^ self.scene_trainer_cfg.train_scene_opt
        last_target_decoder_output = optimizer_output.target_render_list[
            -1] if optimizer_output is not None else init_target_render_output

        # render depth loss
        if self.train_cfg.render_depth_loss_weight > 0:
            # [B, V, H, W]
            near = batch["target"]["near"][..., None, None]  # [B, V, 1, 1]
            far = batch["target"]["far"][..., None, None]

            target_gt_depth = batch["target"]["depth"]
            render_depth = last_target_decoder_output.depth

            valid = (target_gt_depth >= near) & (target_gt_depth <= far) & (render_depth >= near) & (
                    render_depth <= far)

            render_depth_loss = self.train_cfg.render_depth_loss_weight * (
                    torch.log(target_gt_depth[valid]) - torch.log(render_depth[valid])).abs().mean()

            self.log(f"loss/render_depth", render_depth_loss)
            total_loss = total_loss + render_depth_loss

        # depth loss
        if self.train_cfg.depth_loss_weight > 0:
            near = batch["context"]["near"][..., None, None]  # [B, V, 1, 1]
            far = batch["context"]["far"][..., None, None]

            depth_gt = batch['context']["depth"]  # [B, V, H, W]

            valid = (depth_gt >= near) & (depth_gt <= far)

            # in case there is no valid gt depth (loss will be nan)
            if valid.max() > 0.5:
                # log or inverse depth loss
                if self.train_cfg.log_depth_loss:
                    depth_loss = (
                            torch.log(pred_depths[valid]) - torch.log(depth_gt[valid])).abs().mean()
                else:
                    depth_loss = (
                            1. / pred_depths[valid] - 1. / depth_gt[valid]).abs().mean()

                depth_loss = self.train_cfg.depth_loss_weight * depth_loss

                self.log(f"loss/depth", depth_loss)
                total_loss = total_loss + depth_loss

        # depth smooth loss
        if self.train_cfg.depth_smooth_loss_weight > 0:
            imgs = batch["context"]["image"].flatten(0, 1)  # [BV, 3, H, W]

            depth = pred_depths.flatten(0, 1).unsqueeze(1)

            disp = 1. / depth
            if self.train_cfg.depth_smooth_loss_nonorm:
                norm_disp = disp
            else:
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)

            # resize to depth's resolution
            if imgs.shape[-2:] != norm_disp.shape[-2:]:
                imgs = F.interpolate(imgs, size=norm_disp.shape[-2:], mode='bilinear', align_corners=True)

            depth_smooth_loss = get_smooth_loss(norm_disp, imgs)

            depth_smooth_loss = self.train_cfg.depth_smooth_loss_weight * depth_smooth_loss

            self.log(f"loss/depth_smooth", depth_smooth_loss)
            total_loss = total_loss + depth_smooth_loss
        # depth smooth loss for novel views
        if self.train_cfg.depth_smooth_loss_weight_nvs > 0:
            imgs = batch["target"]["image"].flatten(0, 1)  # [BV, 3, H, W]

            depth = last_target_decoder_output.depth.flatten(0, 1).unsqueeze(1)

            disp = 1. / depth.clamp(min=1e-3, max=1000.)
            if self.train_cfg.depth_smooth_loss_nonorm:
                norm_disp = disp
            else:
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)

            depth_smooth_loss_nvs = get_smooth_loss(norm_disp, imgs)

            depth_smooth_loss_nvs = self.train_cfg.depth_smooth_loss_weight_nvs * depth_smooth_loss_nvs

            self.log(f"loss/depth_smooth_nvs", depth_smooth_loss_nvs)
            total_loss = total_loss + depth_smooth_loss_nvs
        # monodepth loss
        if self.train_cfg.monodepth_loss_weight > 0:
            imgs = batch["context"]["image"].flatten(0, 1)  # [BV, 3, H, W]

            pred_disp = 1. / pred_depths.flatten(0, 1).clamp(min=1e-2)  # [BV, H, W]

            # resize to max size 518
            max_width = 518

            ori_h, ori_w = imgs.shape[-2:]

            # resize the max size to 518
            assert ori_h <= ori_w
            if ori_w != max_width:
                new_h = int(ori_h * max_width / ori_w) // 14 * 14  # make sure divisible by 14
                new_w = max_width
                imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=True)

            # normalize images
            imgs = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )(imgs)

            # monodepth prediction: disparity
            with torch.no_grad():
                monodepth_pred = self.pretrained_monodepth(imgs)

            monodepth_pred = F.interpolate(monodepth_pred.unsqueeze(1), size=(ori_h, ori_w), mode='nearest').squeeze(
                1)  # [BV, H, W]

            def normalize_disp(disp):
                median = disp.median(dim=-1, keepdim=True)[0]  # [BV]
                var = (disp - median).abs().mean(dim=-1, keepdim=True)

                return (disp - median) / (var + 1e-6)

            norm_pred_disp = normalize_disp(pred_disp.flatten(1, 2))
            norm_mono_disp = normalize_disp(monodepth_pred.flatten(1, 2))

            monodepth_loss = (norm_pred_disp - norm_mono_disp).abs().mean()

            monodepth_loss = self.train_cfg.monodepth_loss_weight * monodepth_loss

            self.log(f"loss/monodepth", monodepth_loss)
            total_loss = total_loss + monodepth_loss
        return total_loss

    def _calc_opt_loss(self, batch, optimizer_output, t, valid_depth_mask):
        """Compute loss over all optimizer refinement steps for both target and context views."""
        opt_loss = 0
        assert optimizer_output is not None
        refine_step_num = len(optimizer_output.context_render_list) - 1  # first render is initialization

        # (tag, loss_enabled, loss_num)  — render/index lists accessed via optimizer_output methods
        view_loss_cfg = [
            ("target", self.train_cfg.loss_on_target_views, self.train_cfg.loss_on_target_views_num),
            ("context", self.train_cfg.loss_on_input_views, self.train_cfg.loss_on_input_views_num),
        ]

        for i in range(refine_step_num):
            for tag, loss_enabled, loss_num in view_loss_cfg:
                render_list = optimizer_output.get_render_list(tag)
                index_list = optimizer_output.get_index_list(tag)
                # all_gt_rgb: full GT for all views in the batch [B, V_all, C, H, W]
                all_gt_rgb = batch[tag]["image"]

                if index_list:
                    # opt_batch_size < V_all: optimizer rendered a subset of views this step
                    train_idx = index_list[i]  # [B, V_rendered] — from scene_trainer.opt_batch_size
                    curr_gt_rgb = batched_select(all_gt_rgb, train_idx)  # [B, V_rendered, C, H, W]
                else:
                    curr_gt_rgb = all_gt_rgb
                self._log_train_metrics(i + 1, render_list[i + 1].color, curr_gt_rgb, tag=tag, t=t)

                if loss_enabled:
                    b, actual_v = curr_gt_rgb.shape[:2]
                    num_loss = actual_v if loss_num < 0 else loss_num
                    # error_idx: subsample rendered views down to loss_num for the loss computation
                    error_idx = torch.randperm(actual_v, device=curr_gt_rgb.device)[:num_loss]
                    error_idx = error_idx.unsqueeze(0).expand(b, -1)
                    opt_loss += self.compute_losses(optimizer_output.gaussian_list[i], i, refine_step_num,
                                                    render_list[i + 1], curr_gt_rgb, valid_depth_mask,
                                                    error_idx=error_idx, all_gt_rgb=all_gt_rgb, tag=tag)
        if any(isinstance(loss, LossStability) for loss in self.losses):
            stability_loss_fn = next(loss for loss in self.losses if isinstance(loss, LossStability))
            stability_loss = stability_loss_fn(optimizer_output, batch)
            opt_loss += stability_loss
            self.log(f"loss/stability", stability_loss)

        return opt_loss

    def _log_init_metrics_from_optimizer(self, batch, optimizer_output):
        assert optimizer_output is not None
        for tag, is_target in [("context", False), ("target", True)]:
            render_list = optimizer_output.get_render_list(tag)
            index_list = optimizer_output.get_index_list(tag)
            all_gt_rgb = batch[tag]["image"]
            # Using the first optimization step indices (which was used for rendering during optimization)
            curr_gt_rgb = batched_select(all_gt_rgb, index_list[0]) if index_list else all_gt_rgb
            self._log_train_metrics(0, render_list[0].color, curr_gt_rgb, tag=tag)

    def _calc_init_loss(self, init_gaussians, init_target_render_output, target_gt_rgb, valid_depth_mask):
        assert not self.train_cfg.loss_on_input_views
        self._log_train_metrics(0, init_target_render_output.color, target_gt_rgb, tag="target")
        # TODO Naama: train init model on context+target?
        return self.compute_losses(init_gaussians, 0, 1, init_target_render_output, target_gt_rgb, valid_depth_mask)

    def _log_train_metrics(self, i, pred, gt, tag, t=-1):
        psnr = compute_psnr(
            rearrange(gt, "b v c h w -> (b v) c h w"),
            rearrange(pred, "b v c h w -> (b v) c h w"),
        )
        self.log(f"train/{tag}_psnr_{i}", psnr.mean().item())

        if self.global_step < (100000 if DEBUG else 10) and self.global_rank == 0:
            print(
                f"Training step {self.global_step}, inner step {t} i {i} train psnr {psnr.mean().item()}")

    def train_render_output_for_init_gaussians(self, batch, gaussians):
        b, v, _, h, w = batch["context"]["image"].shape
        assert gaussians.means.size(0) == batch["target"]["extrinsics"].size(0), \
            "num_scales must be 1; multi-scale depth supervision is not supported"
        batch_size = batch["target"]["extrinsics"].size(0)
        output = self.scene_decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode='depth' if self.train_cfg.render_depth_loss_weight > 0 else None,
        )
        return batch_size, output

    # ==================== Meta Optimizer Configuration ====================

    def _split_params(self, filter_key: str) -> tuple[list, list]:
        """Split parameters into (matched, rest) based on whether name contains filter_key."""
        matched, rest = [], []
        for name, param in self.named_parameters():
            (matched if filter_key in name else rest).append(param)
        return matched, rest

    def _build_adamw(self, params_or_groups, weight_decay: float, **kwargs):
        """Instantiate AdamW or AdamW8bit depending on config."""
        cls = AdamW8bit if self.meta_optimizer_cfg.adamw_8bit else optim.AdamW
        return cls(params_or_groups, weight_decay=weight_decay, **kwargs)

    def configure_optimizers(self):
        # This is the *meta* optimizer — it optimizes the parameters of the learned optimizer itself
        # (i.e. the KnnBasedOptimizer weights), not individual scene Gaussians.
        # Only called by Lightning during fit(); skipped entirely in test mode.
        cfg = self.meta_optimizer_cfg

        if cfg.lr_depth > 0:
            pretrained, rest = self._split_params("depth_predictor")
            param_groups = [{"params": pretrained, "lr": cfg.lr_depth}, {"params": rest, "lr": cfg.lr}]
            scheduler_lrs = [cfg.lr_monodepth, cfg.lr]
        elif cfg.lr_monodepth > 0:
            pretrained, rest = self._split_params("pretrained")
            param_groups = [{"params": pretrained, "lr": cfg.lr_monodepth}, {"params": rest, "lr": cfg.lr}]
            scheduler_lrs = [cfg.lr_monodepth, cfg.lr]
        else:
            param_groups = self.parameters()
            scheduler_lrs = cfg.lr

        meta_optimizer = self._build_adamw(param_groups, cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            meta_optimizer,
            scheduler_lrs,
            self.trainer.max_steps + 10,
            pct_start=cfg.warm_up_ratio,
            cycle_momentum=False,
            anneal_strategy="cos",
        )

        return {
            "optimizer": meta_optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ==================== Testing ====================

    @torch.no_grad()
    def test_step(self, scene_batch: BatchedExample, batch_idx: int):
        """
        This is a meta trainer class. Each training step and test step corresponds to training on one scene.
        We delegate the actual training/testing to meta_training_step and meta_test_step.
        The loop over inner training steps (training within a specific scene) is performed in
        self.get_optimized_gaussians.
        Each inner iteration is done by calling the forward call of the optimizer.
        """
        return self.meta_test_step(scene_batch, batch_idx)

    @torch.no_grad()
    def meta_test_step(self, scene_batch: BatchedExample, batch_idx: int):
        """Run the full test pipeline for one scene: initialize, optimize, then evaluate and save."""
        if self.test_cfg.scenes_filter is not None and scene_batch['scene'][0] not in self.test_cfg.scenes_filter:
            print(f"Scenes filter: {self.test_cfg.scenes_filter}")
            print(f"Skipping scene {scene_batch['scene'][0]} (not in scenes_filter)")
            return

        output_path = self.test_cfg.output_path

        if output_path is not None and self.test_cfg.skip_if_outputs_exist:
            optimizer_name = self.scene_trainer.optimizer.__class__.__name__.lower() if self.scene_trainer.optimizer is not None else "no_optimizer"
            target_metric_path = output_path / optimizer_name / "metrics" / f"{scene_batch['scene'][0]}" / f"target_{optimizer_name}.json"
            context_metric_path = output_path / optimizer_name / "metrics" / f"{scene_batch['scene'][0]}" / f"context_{optimizer_name}.json"
            should_eval_context = self.test_cfg.eval_context_views
            should_eval_target = True  # always evaluate target views

            skip_target = (should_eval_target and target_metric_path.exists()) or not should_eval_target
            skip_context = (should_eval_context and context_metric_path.exists()) or not should_eval_context

            if skip_target and skip_context:
                print(
                    f"Metrics for scene {scene_batch['scene'][0]} already exist at {target_metric_path} and {context_metric_path}. Skipping...")
                return

        rule(f"Testing scene {batch_idx}: {scene_batch['scene'][0]}")

        # input (context and target)
        batch: BatchedExample = self.initializer_data_shim(scene_batch)

        # Process batch for experiments, e.g., add noise (skip if not needed)
        if self.test_cfg.experimental_add_noise_to_images:
            batch = self.experimental_process_batch(batch)

        # Save cameras as JSON (before optimization, cameras are fixed)
        if self.test_cfg.save_cameras_json:
            scene_name = batch["scene"][0]
            relevant_keys = ["extrinsics", "intrinsics"]
            context_info = {key: batch["context"][key][0].cpu().tolist() for key in relevant_keys}
            target_info = {key: batch["target"][key][0].cpu().tolist() for key in relevant_keys}
            resolution = list(batch["context"]["image"].shape[-2:])
            cameras_data = {
                "scene": scene_name,
                "context": context_info,
                "target": target_info,
                "resolution": resolution,
            }
            cameras_dir = output_path / "cameras"
            cameras_dir.mkdir(parents=True, exist_ok=True)
            cameras_path = cameras_dir / f"{scene_name}_cameras.json"
            with open(cameras_path, "w") as f:
                json.dump(cameras_data, f, indent=4)
            print(f"Saved cameras JSON to {cameras_path}")

        # Save cameras as NPZ in the exact form fed to the rasterizer:
        #   viewmats = inverse(extrinsics)  (world-to-camera, [V,4,4])
        #   Ks       = intrinsics * diag(W, H, 1)  (pixel-space, [V,3,3])
        # Mirrors GSplatDecoderSplattingCUDA.forward (gsplat_decoder_splatting_cuda.py:137-140).
        if self.test_cfg.save_cameras_npz:
            scene_name = batch["scene"][0]
            cameras_dir = output_path / "cameras"
            cameras_dir.mkdir(parents=True, exist_ok=True)
            npz_data = {"scene": scene_name}
            for input_str in ("context", "target"):
                view = batch[input_str]
                extrinsics = view["extrinsics"][0]  # [V, 4, 4] cam-to-world
                intrinsics = view["intrinsics"][0]  # [V, 3, 3] normalized
                h, w = view["image"].shape[-2:]
                viewmats = extrinsics.inverse()  # [V, 4, 4] world-to-cam
                scale = intrinsics.new_tensor([[w], [h], [1]])
                Ks = intrinsics * scale  # [V, 3, 3] pixel-space
                npz_data[f"{input_str}_viewmats"] = viewmats.cpu().numpy()
                npz_data[f"{input_str}_Ks"] = Ks.cpu().numpy()
                npz_data[f"{input_str}_image_shape"] = np.array([h, w], dtype=np.int64)
            cameras_npz_path = cameras_dir / f"{scene_name}_cameras.npz"
            np.savez(cameras_npz_path, **npz_data)
            print(f"Saved renderer-ready cameras NPZ to {cameras_npz_path}")

        self.scene_initializer.preprocessing(batch, self.train_cfg)

        # Infer Gaussians.

        # init
        scene_name = batch["scene"][0]
        init_output: InitializerOutput = self.init_gaussians_and_render(
            batch,
            visualization_dump={},
            render_context=True,
            render_target=True,
            grad_enabled=False,
            cached_data_path=Path(os.path.join("cache", "edgs", scene_name)),  # for EDGS only for now  # TODO Naame: review
        )

        if self.test_cfg.eval_initialization:
            print("\nEvaluating initialization...")

            # Evaluate and save initialization
            self._eval_and_save(
                self.scene_initializer,
                batch,
                batch_idx,
                init_output,
                output_path
            )

        # Optimization
        if self.scene_trainer.optimizer is None:
            optimizer_output = None
        else:
            # run optimizer
            torch.cuda.reset_peak_memory_stats()
            try:
                optimizer_output = self.get_optimized_gaussians(
                    batch,
                    init_output,
                    output_path=output_path / self.scene_trainer.optimizer.__class__.__name__.lower(),
                    scene_name=scene_name,
                    debug_dict=defaultdict(list),
                )
            except (torch.OutOfMemoryError, RuntimeError) as e:
                warn('ran out of memory during optimization. Skipping scene.')
                torch.cuda.empty_cache()
                return None
            except SkipBatchException as e:
                warn(f'skipping scene due to SkipBatch during optimization: {e}')
                return None

            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.benchmarker.record("peak_vram_mb", peak_vram_mb)

            # Record per-scene timing from CUDA event logs (all in ms).
            # optimizer_net = on_scene_start + all iteration steps. Excludes save-every renders
            # (which happen after iter_end.record() and are therefore not in iter_time_log).
            opt = self.scene_trainer.optimizer
            decoder_ms = sum(opt.decoder_time_log)
            optimizer_ms = sum(opt.optimizer_time_log)
            optimizer_net_ms = opt.scene_start_ms + decoder_ms + optimizer_ms
            self.benchmarker.record("decoder", decoder_ms)
            self.benchmarker.record("optimizer", optimizer_ms)
            self.benchmarker.record("optimizer_net", optimizer_net_ms)
            print(
                f"[timing] scene={scene_name} "
                f"scene_start={opt.scene_start_ms:.0f}ms "
                f"decoder={decoder_ms:.0f}ms "
                f"optimizer={optimizer_ms:.0f}ms "
                f"optimizer_net={optimizer_net_ms:.0f}ms "
                f"peak_vram={peak_vram_mb:.0f}MB"
            )
            opt.decoder_time_log.clear()
            opt.optimizer_time_log.clear()

            # Collected here; written into target_*.json / context_*.json by _eval_and_save below.
            _scene_timing_metrics = {
                "peak_vram_mb": peak_vram_mb,
                "decoder_ms": decoder_ms,
                "optimizer_ms": optimizer_ms,
                "optimizer_net_ms": optimizer_net_ms,
                "scene_start_ms": opt.scene_start_ms,
            }

        #
        plot_phases = []  # (label, metrics_dict) for combined plotting

        if optimizer_output is not None:
            # Init is already spliced into position 0 of optimizer_output lists by
            # SceneTrainer.get_optimized_gaussians (see _insert_init_into_output).

            # Run evaluation and saving
            opt_metrics = self._eval_and_save(
                self.scene_trainer.optimizer,
                batch,
                batch_idx,
                optimizer_output,
                output_path,
                extra_scene_metrics=_scene_timing_metrics,
            )
            opt_label = self.scene_trainer.optimizer.__class__.__name__.lower()
            plot_phases.append((opt_label, opt_metrics))

            # updates, parameters and gradients visualizations
            # self.debugging(optimizer_output, output_path, batch["scene"][0])

        # Post-processing
        postprocessed_output = self.test_postprocess_gaussians(
            batch,
            gaussians=optimizer_output.gaussian_list[-1] if optimizer_output is not None else init_output.gaussians,
            visualization_dump={}
        )

        # Evaluate and save post-processing
        if postprocessed_output is not None:
            pp_metrics = self._eval_and_save(
                self.scene_trainer.postprocess,
                batch,
                batch_idx,
                postprocessed_output,
                output_path
            )
            pp_label = self.scene_trainer.postprocess.__class__.__name__.lower()
            plot_phases.append((pp_label, pp_metrics))

        # Combined metrics plot (optimizer + postprocessing)
        if plot_phases:
            pass
            # self._plot_combined_metrics(
            #     output_path=output_path,
            #     scene_name=scene_name,
            #     phases=plot_phases,
            # )

    def experimental_process_batch(self, batch: BatchedExample) -> BatchedExample:
        noise_std = self.test_cfg.experimental_add_noise_to_images_std
        for key in ["context", "target"]:
            images = batch[key]["image"]  # [B, V, 3, H, W]
            noise = torch.randn_like(images) * noise_std
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
            batch[key]["image"] = noisy_images
            batch[key]["clean_image"] = images  # keep clean images for evaluation
        return batch

    @torch.no_grad()
    @rank_zero_only
    def validation_step(self, scene_batch: BatchedExample, batch_idx: int):
        scene_batch: BatchedExample = self.initializer_data_shim(scene_batch)

        self.scene_initializer.preprocessing(scene_batch, self.train_cfg)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene_name = {[a[:20] for a in scene_batch['scene']]}; "
                f"context = {scene_batch['context']['index'].tolist()}; "
                f"target = {scene_batch['target']['index'].tolist()}"
            )

        # Render Gaussians.
        b, v, _, h, w = scene_batch["context"]["image"].shape
        assert b == 1

        try:
            initializer_output = self.get_init_gaussians(scene_batch, is_training=False)
        except SkipBatchException as e:
            warn(f"Skipping validation for scene {scene_batch['scene'][0]} due to error in initialization: {e}")
            return

        output_softmax = self.scene_decoder.forward_target(
            initializer_output.gaussians, scene_batch, (h, w),
            depth_mode='depth' if self.train_cfg.eval_render_depth or self.train_cfg.viz_render_depth else None,
        )

        # refine
        debug_dict = {}
        if self.scene_optimizer is not None:
            try:
                optimizer_output = self.get_optimized_gaussians(
                    scene_batch,
                    initializer_output,
                    debug_dict=debug_dict
                )
            except SkipBatchException as e:
                warn(f"Skipping validation for scene {scene_batch['scene'][0]} due to error: {e}")
                return
            render_output = optimizer_output.target_render_list
            output_softmax = render_output[-1]

        rgb_softmax = output_softmax.color[0]

        # Move prediction back to device
        rgb_softmax = rgb_softmax.to(scene_batch["target"]["image"].device)

        # Compute validation metrics.
        rgb_gt = scene_batch["target"]["image"][0]
        for tag, rgb in zip(("val",), (rgb_softmax,)):
            psnr = compute_psnr(rgb_gt, rgb)
            self.log(f"val/psnr_{tag}", psnr)
            ssim = compute_ssim(rgb_gt, rgb)
            self.log(f"val/ssim_{tag}", ssim)

        # viz depth
        if initializer_output.depths is not None and self.train_cfg.viz_depth_separate:
            # only visualize predicted depth
            pred_depths = initializer_output.depths[0]  # [V, H, W]

            # gaussian downsample
            # downsample image to depth resolution
            if pred_depths.shape[1:] != scene_batch["context"]["image"].shape[-2:]:
                input_images = F.interpolate(
                    scene_batch["context"]["image"][0],
                    size=pred_depths.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            else:
                input_images = scene_batch["context"]["image"][0]  # [N, 3, H, W]

            concat = self._make_depth_viz(1.0 / pred_depths, input_images)

            # reshape when the number of input images is too large
            # otherwise the image will be too wide
            num_inputs = input_images.shape[0]
            width = input_images.shape[-1]
            if num_inputs > 8:
                rows = 4
                assert num_inputs % rows == 0
                stride = num_inputs // rows
                out = []
                for i in range(rows):
                    out.append(concat[:, :, width * stride * i: width * stride * (i + 1)])

                concat = torch.cat(out, dim=1)  # [3, H*2*R, W*N/R]

                # resize to half resolution to save space
                concat = F.interpolate(concat.unsqueeze(0), scale_factor=0.5, mode='bilinear',
                                       align_corners=True).squeeze(0)

            self.logger.log_image(
                "depth",
                [concat],
                step=self.global_step,
                caption=scene_batch["scene"],
            )

        # viz rendered depth
        if self.train_cfg.eval_render_depth or self.train_cfg.viz_render_depth:
            render_depth = output_softmax.depth[0]  # [V, H, W]
            input_images = scene_batch["target"]["image"][0]  # [N, 3, H, W]
            concat = self._make_depth_viz(1.0 / render_depth.clamp(min=0.01, max=1000.), input_images)

            self.logger.log_image(
                "render_depth",
                [concat],
                step=self.global_step,
                caption=scene_batch["scene"],
            )

        # Subsample context images when there are too many to fit comfortably side-by-side
        n_ctx = scene_batch["context"]["image"][0].shape[0]
        stride = 4 if n_ctx > 16 else (2 if n_ctx > 8 else 1)
        viz_input = scene_batch["context"]["image"][0][::stride]
        tag = "Context" if stride == 1 else f"Context (1/{stride})"

        comparison = self._build_comparison_image(
            initializer_output, viz_input, tag, rgb_gt, rgb_softmax, stride
        )

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=scene_batch["scene"],
        )

        if not self.train_cfg.no_log_projections:
            # Render projections and construct projection image.
            projections = hcat(
                *render_projections(
                    initializer_output.gaussians,
                    256,
                    extra_label="(Prediction)",
                )[0]
            )
            self.logger.log_image(
                "projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )

            # Draw cameras.
            # cameras = hcat(*render_cameras(batch, 256))
            # self.logger.log_image(
            #     "cameras", [prep_image(add_border(cameras))], step=self.global_step
            # )

        # Run video validation step.
        if not self.train_cfg.no_viz_video:
            self.render_video_interpolation(scene_batch)
            # self.render_video_wobble(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(scene_batch)

    def _build_comparison_image(
            self,
            initializer_output: InitializerOutput,
            viz_input: Tensor,
            tag: str,
            rgb_gt: Tensor,
            rgb_softmax: Tensor,
            stride: int,
    ) -> Tensor:
        """Build the side-by-side comparison image for validation logging."""
        cols = [
            add_label(vcat(*viz_input), tag),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Prediction)"),
        ]

        if not self.train_cfg.viz_depth_separate and initializer_output.depths is not None:
            pred_depths = initializer_output.depths[0]  # [V, H, W]
            inverse_depth_pred = 1.0 / pred_depths
            concat = rearrange(inverse_depth_pred, "v h w -> (v h) w")
            depth_viz = viz_depth_tensor(concat.cpu().detach()).to(pred_depths.device).float() / 255.
            depth_viz = rearrange(depth_viz, "c (v h) w -> v c h w", v=pred_depths.shape[0])

            if depth_viz.shape[-2:] != viz_input.shape[-2:]:
                depth_viz = F.interpolate(depth_viz, size=viz_input.shape[-2:], mode='bilinear', align_corners=True)

            depth_viz = depth_viz[::stride]
            cols.insert(1, add_label(vcat(*depth_viz), "Depth (Prediction)"))

        return hcat(*cols)

    @staticmethod
    def _make_depth_viz(inverse_depth: Tensor, images: Tensor) -> Tensor:
        """Combine inverse-depth colormap with RGB images for logging. Returns [3, H*2, W*V]."""
        depth_viz = viz_depth_tensor(torch.cat(list(inverse_depth), dim=1).cpu().detach())  # [3, H, W*V]
        concat_img = torch.cat(list(images), dim=-1).cpu().detach() * 255  # [3, H, W*V]
        return torch.cat((concat_img, depth_viz), dim=1)  # [3, H*2, W*V]

    def on_fit_start(self):
        run = self.logger.experiment
        if run is not None:
            run.define_metric("inner_iteration")
            run.define_metric("test/psnr/*", step_metric="inner_iteration")

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        """Run evaluation on the full test set during training (rank-zero only). Logs PSNR/SSIM to wandb table."""
        print(
            f"Validation step at global step {self.global_step}. Running evaluation on {self.train_cfg.eval_data_length} test sets...")
        start_t = time.time()

        pred_depths = None
        depth_gt = None

        full_testsets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )
        scores_dict = defaultdict(lambda: defaultdict(list))

        self.benchmarker.clear_history()
        time_skip_first_n_steps = min(
            self.train_cfg.eval_time_skip_steps, len(full_testsets)
        )
        time_skip_steps_dict = {"encoder": 0, "decoder": 0}
        for batch_idx, batch in tqdm(
                enumerate(full_testsets),
                total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break

            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            batch = self.on_after_batch_transfer(batch, dataloader_idx=batch_idx)
            batch = self.initializer_data_shim(batch)

            # use gt depth range instead of a fixed one
            self.scene_initializer.preprocessing(batch, self.train_cfg)

            # Render Gaussians.
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1
            if batch_idx < time_skip_first_n_steps:
                time_skip_steps_dict["encoder"] += 1
                time_skip_steps_dict["decoder"] += v

            with self.benchmarker.time("encoder"):
                init_output = self.get_init_gaussians(batch, is_training=False)

            with self.benchmarker.time("decoder", num_calls=v):
                output_probabilistic = self.scene_decoder.forward_target(
                    init_output.gaussians, batch, (h, w),
                    depth_mode='depth' if self.train_cfg.eval_render_depth or self.train_cfg.viz_render_depth else None,
                )

            init_rgb = output_probabilistic.color[0]

            # refine
            if self.scene_optimizer is not None:
                try:
                    optimizer_output = self.get_optimized_gaussians(batch, init_output)
                except SkipBatchException as e:
                    warn(f'Skipping batch due to SkipBatch during optimization: {e}')
                    continue
                render_output = optimizer_output.target_render_list
                output_probabilistic = render_output[-1]

            rgbs = [init_rgb]
            if self.scene_trainer_cfg.num_update_steps > 0:
                rgbs += [render.color[0] for render in render_output]
            tags = ["probabilistic"] * len(rgbs)

            if self.train_cfg.eval_deterministic:
                gaussians_deterministic = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=True,
                )
                output_deterministic = self.scene_decoder.forward(
                    gaussians_deterministic,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )
                rgbs.append(output_deterministic.color[0])
                tags.append("deterministic")

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            if self.scene_optimizer is not None:
                steps = self.scene_optimizer.save_every.get_iterations(len(rgbs))
            else:
                steps = [0]
            for i, (tag, rgb) in enumerate(zip(tags, rgbs)):
                # Move prediction back to device
                rgb = rgb.to(batch["target"]["image"].device)
                metric_scores: dict = compute_rgb_metrics(
                    rgb, rgb_gt,
                    metrics=["psnr", "ssim", "lpips"],
                    iter_batch_size=-1,
                )
                for name, score in metric_scores.items():
                    if name == "lpips":
                        # tuple of (alex, vgg)
                        scores_dict[f"alex_lpips_{steps[i]}"][tag].append(score[0].item())
                        scores_dict[f"vgg_lpips_{steps[i]}"][tag].append(score[1].item())
                    else:
                        scores_dict[f"{name}_{steps[i]}"][tag].append(score.item())
                    # log the last step metrics to compare between runs
                    if i == len(rgbs) - 1:
                        if name == "lpips":
                            # tuple of (alex, vgg)
                            scores_dict[f"alex_lpips"][tag].append(score[0].item())
                            scores_dict[f"vgg_lpips"][tag].append(score[1].item())
                        else:
                            scores_dict[f"{name}"][tag].append(score.item())

            # compute depth metrics
            if pred_depths is not None and depth_gt is not None and depth_gt.max() > 0:
                assert pred_depths is not None and depth_gt is not None

                pred_depths = pred_depths[0]  # [V, H, W]

                # gaussian downsample
                if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                    pred_depths = F.interpolate(
                        pred_depths.unsqueeze(1),
                        size=batch["context"]["image"].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(1)

                depth_gt = depth_gt[0]  # [V, H, W]

                near = batch["context"]["near"][...,
                None, None][0]  # [V, 1, 1]
                far = batch["context"]["far"][..., None, None][0]  # [V, 1, 1]

                valid = (depth_gt >= near) & (depth_gt <= far)

                all_metrics = compute_depth_errors(depth_gt[valid].detach().cpu().numpy(),
                                                   pred_depths[valid].detach().cpu().numpy())
                scores_dict["abs_rel"]["probabilistic"].append(all_metrics[0])
                scores_dict["rmse"]["probabilistic"].append(all_metrics[2])
                scores_dict["a1"]["probabilistic"].append(all_metrics[4])

            # compute rendered depth metrics
            if self.train_cfg.eval_render_depth:
                render_depth = output_probabilistic.depth
                target_depth_gt = batch["target"]["depth"]

                pred_depths = render_depth[0]  # [V, H, W]
                depth_gt = target_depth_gt[0]  # [V, H, W]

                near = batch["target"]["near"][..., None, None][0]  # [V, 1, 1]
                far = batch["target"]["far"][..., None, None][0]  # [V, 1, 1]

                valid = (depth_gt >= near) & (depth_gt <= far)

                all_metrics = compute_depth_errors(depth_gt[valid].detach().cpu().numpy(),
                                                   pred_depths[valid].detach().cpu().numpy())

                scores_dict["render_abs_rel"]["probabilistic"].append(all_metrics[0])
                scores_dict["render_rmse"]["probabilistic"].append(all_metrics[2])
                scores_dict["render_a1"]["probabilistic"].append(all_metrics[4])

        # summarise scores and log to logger
        # Create wandb table for inner iteration visualization
        # For now, log only psnr
        if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
            # Extract metrics that have step numbers (e.g., "psnr_0", "psnr_1", etc.)
            inner_iteration_data = []
            for score_tag, methods in scores_dict.items():
                # Check if this is a step-specific metric (e.g., "psnr_0", "psnr_1", etc.)
                if '_' in score_tag and score_tag.split('_')[-1].isdigit():
                    metric_name, step_str = score_tag.rsplit('_', 1)
                    inner_step = int(step_str)

                    if metric_name not in ["psnr"]:
                        continue

                    for method_tag, cur_scores in methods.items():
                        if len(cur_scores) > 0:
                            cur_mean = sum(cur_scores) / len(cur_scores)
                            inner_iteration_data.append({
                                'meta_iteration': self.global_step,
                                'inner_iteration': inner_step,
                                'metric_name': metric_name,
                                'method': method_tag,
                                'value': cur_mean
                            })

            # Log the table if we have inner iteration data
            if inner_iteration_data:
                try:
                    # Rewrite the chart (wandb cannot append to the current figure (?))
                    df = pd.DataFrame(inner_iteration_data)
                    df["meta_iteration_str"] = df["meta_iteration"].astype(str)
                    metric_to_plot = "psnr"
                    df_metric = df[df["metric_name"] == metric_to_plot]

                    table = wandb.Table(dataframe=df_metric)

                    # self.logger.experiment.log({f"{metric_to_plot}_line": wandb.plot.line(
                    #     table,
                    #     x="inner_iteration",
                    #     y="value",
                    #     title=f"{metric_to_plot} per inner iteration",
                    #     stroke="meta_iteration_str",
                    # )})

                    # Plot psnr for current meta iteration in a separate chart
                    # run = self.logger.experiment
                    current_meta = self.global_step

                    df_current = df_metric[df_metric["meta_iteration"] == current_meta]

                    if len(df_current) > 1:
                        run = self.logger.experiment
                        run.define_metric("inner_iteration")
                        run.define_metric(f"test/psnr/meta_{current_meta}", step_metric="inner_iteration")
                        for _, row in df_current.iterrows():
                            run.log({
                                "inner_iteration": row["inner_iteration"],
                                f"test/psnr/meta_{current_meta}": row["value"],
                            })

                except Exception as e:
                    warn(f"Could not create automatic charts: {e}")
                    # Fallback: just log the table
                    pass

        # Keep the original logging
        for score_tag, methods in scores_dict.items():
            for method_tag, cur_scores in methods.items():
                if len(cur_scores) > 0:
                    cur_mean = sum(cur_scores) / len(cur_scores)
                    self.log(f"test/{score_tag}", cur_mean)
        # summarise run time
        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(time_skip_steps_dict[tag]):]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        psnr_list = [scores_dict[f"psnr_{i}"]["probabilistic"] for i in
                     range(self.scene_trainer_cfg.num_update_steps + 1)]
        psnr_list = [sum(pnsr) / len(pnsr) for pnsr in psnr_list if len(pnsr) > 0]
        psnr_str = ", ".join(f"psnr_{i}: {np.mean(pnsr):.3f}" for i, pnsr in enumerate(psnr_list))
        example_num = len(scores_dict['psnr_0']['probabilistic'])
        print(f"Eval total time cost: {overall_eval_time:.3f}s, {psnr_str}, example_num: {example_num} ")
        self.log("test/runtime_all", overall_eval_time)

    @staticmethod
    def _get_renders_list(
            output: OptimizerOutput | InitializerOutput,
            input_str: str,
            module: Initializer | Optimizer | PostProcessing3DGS,
    ) -> tuple[list, list[int]]:
        """Get render list and corresponding iteration indices for a given view tag ('context' or 'target')."""
        if isinstance(output, OptimizerOutput):
            renders_list = output.get_render_list(input_str)
        elif isinstance(output, InitializerOutput):
            if input_str == "context":
                assert output.context_render is not None, "InitializerOutput must contain context_render"
                renders_list = [output.context_render]
            elif input_str == "target":
                assert output.target_render is not None, "InitializerOutput must contain target_render"
                renders_list = [output.target_render]
            else:
                raise ValueError(f"Unknown input_str: {input_str}")
        else:
            raise ValueError(f"Unknown output type: {type(output)}")
        iterations = [0] if isinstance(module, Initializer) else module.save_every.get_iterations(len(renders_list))
        return renders_list, iterations

    @staticmethod
    def _compute_depth_range(
            output: OptimizerOutput | InitializerOutput,
            input_strs: list[str],
            module: Initializer | Optimizer | PostProcessing3DGS,
    ) -> tuple[float, float]:
        """Scan all renders to find the global depth min/max for consistent visualization."""
        depth_vmin, depth_vmax = np.inf, -np.inf
        have_depths = True
        for input_str in input_strs:
            renders_list, _ = MetaTrainer._get_renders_list(output, input_str, module)
            assert renders_list is not None, f"No renders found for {input_str}"
            for iter_renders in renders_list:
                iter_depths = iter_renders.depth  # (1, V, H, W)
                if iter_depths is None:
                    have_depths = False
                    continue
                iter_depths = iter_depths[0]  # (V, H, W)
                depth_vmin = min(depth_vmin, iter_depths.min().item())
                depth_vmax = max(depth_vmax, iter_depths.max().item())
        if not have_depths:
            depth_vmin, depth_vmax = 0.0, 1.0
        return depth_vmin, depth_vmax

    def _compute_error_vmax(
            self,
            output: OptimizerOutput | InitializerOutput,
            input_strs: list[str],
            module: Initializer | Optimizer | PostProcessing3DGS,
            batch: BatchedExample,
    ) -> float:
        # Per-scene error range for error-map visualization. The 99th percentile
        # (rather than the raw max) keeps a single outlier pixel from flattening
        # the magma color scale across the whole scene.
        if not self.test_cfg.save_error_image:
            return 1.0
        error_values = []
        for input_str in input_strs:
            renders_list, _ = self._get_renders_list(output, input_str, module)
            if renders_list is None:
                continue
            if "clean_image" in batch[input_str]:
                rgb_gt = batch[input_str]["clean_image"][0]  # (V, 3, H, W)
            else:
                rgb_gt = batch[input_str]["image"][0]  # (V, 3, H, W)
            for iter_renders in renders_list:
                iter_rgbs = iter_renders.color[0]  # (V, 3, H, W)
                err = (iter_rgbs - rgb_gt.to(iter_rgbs)).abs().mean(1)  # (V, H, W)
                error_values.append(err.flatten().cpu().numpy())
                del iter_rgbs, err
        if not error_values:
            return 1.0
        return float(np.percentile(np.concatenate(error_values), 99))

    def _compute_and_save_scores(
            self,
            module: Initializer | Optimizer | PostProcessing3DGS,
            output: OptimizerOutput | InitializerOutput,
            renders_list: list,
            rgb_gt: Tensor,
            iterations: list[int],
            input_str: str,
            module_name: str,
            out_dir: Path,
            extra_scene_metrics: dict | None,
    ) -> None:
        """Compute RGB metrics per iteration, accumulate into output_dict, and save per-scene JSON."""
        # Collect per-step stats logs from the module
        if isinstance(module, Initializer):
            nr_gaussians_log = [output.gaussians.means.shape[1]]
            iter_time_log = [0.0]
            nr_nonzero_grads_log = [0.0]
        elif isinstance(module, (Optimizer, PostProcessing3DGS)):
            nr_gaussians_log = module.nr_gaussians_log
            nr_nonzero_grads_log = module.nr_nonzero_grad_log
            iter_time_log = module.iter_time_log
            iter_time_log[0] = 0.0
        else:
            raise ValueError(f"Unknown module type: {type(module)}")

        self.init_output_dict_for_new_scene(input_str=input_str, tag=module_name)
        output_dict = self.test_step_outputs_context if input_str == "context" else self.test_step_outputs_target
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, step in tqdm(enumerate(iterations), desc=f"Evaluating {input_str}", total=len(iterations)):
            is_last = (i == len(iterations) - 1)
            nr_iter = iterations[i]
            # j: index into per-step logs; last step uses nr_iter-1 because logs are 0-indexed up to nr_iter
            j = nr_iter - 1 if is_last else nr_iter
            iter_rgb = renders_list[i].color[0]  # (V, 3, H, W)

            scores: dict = compute_rgb_metrics(
                iter_rgb,
                rgb_gt,
                metrics=self.test_cfg.compute_scores_metrics,
                iter_batch_size=self.test_cfg.metrics_batch_size,
            )

            if nr_gaussians_log is not None:
                assert j <= len(nr_gaussians_log), f"{j}, {len(nr_gaussians_log)}"
                scores["gaussians"] = torch.tensor(nr_gaussians_log[j])

            if nr_nonzero_grads_log is not None and nr_nonzero_grads_log:
                assert j <= len(nr_nonzero_grads_log), f"{j}, {len(nr_nonzero_grads_log)}"
                scores["nonzero_grads"] = torch.tensor(nr_nonzero_grads_log[j])

            if iter_time_log is not None:
                assert j <= len(iter_time_log), f"{j}, {len(iter_time_log)}"
                scores["time"] = torch.tensor(sum(iter_time_log[:j + 1]))

            for name, score in scores.items():
                if name == "lpips":
                    output_dict[f"{module_name}_alex_lpips"][-1].append(score[0].item())
                    output_dict[f"{module_name}_vgg_lpips"][-1].append(score[1].item())
                else:
                    output_dict[f"{module_name}_{name}"][-1].append(score.item())
            output_dict[f"{module_name}_iterations"][-1].append(nr_iter)

            del iter_rgb

        # Save per-scene metrics to JSON
        last_scene_metrics = {key: vals[-1] for key, vals in output_dict.items()}
        if extra_scene_metrics:
            last_scene_metrics.update(extra_scene_metrics)
        metrics_save_path = out_dir / f"{input_str}_{module_name}.json"
        with metrics_save_path.open("w") as f:
            print(f"Saving metrics to {metrics_save_path}")
            json.dump(last_scene_metrics, f, indent=4)

    @torch.no_grad()
    def _eval_and_save(
            self,
            module: Initializer | Optimizer | PostProcessing3DGS,
            batch: BatchedExample,
            batch_idx: int,
            output: OptimizerOutput | InitializerOutput,
            output_path: Path,
            extra_scene_metrics: dict | None = None,
    ) -> dict:
        """Evaluate and save results. Returns collected metrics dict (keyed by module_name_metric)."""
        module_name = module.__class__.__name__.lower()

        output_path = CustomPath(output_path / module_name)
        output_path.mkdir(parents=True, exist_ok=True)

        target_shape = batch["target"]["image"].shape  # [B, V, 3, H, W]
        context_shape = batch["context"]["image"].shape  # [B, V, 3, H, W]
        assert target_shape[-3:] == context_shape[-3:], f"{target_shape}, {context_shape}"
        b, v, _, h, w = target_shape
        assert b == 1, "Evaluation only supports scene batch size 1."
        scene_name = batch["scene"][0]

        # Save poses
        if self.test_cfg.save_poses:
            poses_data = {
                "context": {"shape": context_shape[-2:]},
                "target": {"shape": target_shape[-2:]},
            }
            for key in ["extrinsics", "intrinsics", "near", "far"]:
                poses_data["context"][key] = batch["context"][key][0].cpu().numpy().tolist()
                poses_data["target"][key] = batch["target"][key][0].cpu().numpy().tolist()
            save_path = output_path / 'poses' / f"{scene_name}_poses.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving poses to {save_path.parent}")
            with open(save_path, 'w') as f:
                json.dump(poses_data, f, indent=4)

        # Save gaussians
        if self.test_cfg.save_gaussian:
            if isinstance(output, InitializerOutput):
                save_path = output_path / 'gaussians' / scene_name / 'init.ply'
                save_gaussian_ply(output.gaussians, save_path)
            elif isinstance(output, OptimizerOutput):
                iterations = module.save_every.get_iterations(len(output.gaussian_list))
                for step, iter_gaussians in zip(iterations, output.gaussian_list):
                    save_path = output_path / 'gaussians' / scene_name / f'step{step}.ply'
                    save_gaussian_ply(iter_gaussians, save_path)
            else:
                raise ValueError(f"Unknown output type: {type(output)}")

        input_strs = ["target"]
        if self.test_cfg.eval_context_views:
            input_strs.insert(0, "context")

        depth_vmin, depth_vmax = self._compute_depth_range(output, input_strs, module)
        error_vmax = self._compute_error_vmax(output, input_strs, module, batch)

        for input_str in input_strs:
            indices = batch[input_str]["index"][0]  # (V,)  # TODO Naama: bug when using opt bs > 0
            renders_list, iterations = self._get_renders_list(output, input_str, module)
            if renders_list is None:
                continue

            if "clean_image" in batch[input_str]:
                rgb_gt = batch[input_str]["clean_image"][0].cpu()  # (V, 3, H, W)
            else:
                rgb_gt = batch[input_str]["image"][0].cpu()  # (V, 3, H, W)
            depth_gt = batch[input_str].get("depth", None)

            # save pred rgbs
            if self.test_cfg.save_render_image:
                if self.test_cfg.save_render_image_last_only:
                    self.test_save_last_rendered_images(renders_list, indices, output_path, scene_name, input_str)
                else:
                    self.test_save_rendered_images(renders_list, indices, output_path, scene_name, input_str)

            # save gt rgbs
            if self.test_cfg.save_gt_image and rgb_gt is not None:
                self.test_save_gt_images(rgb_gt, indices, output_path, scene_name, input_str)

            # save rgb error maps
            if self.test_cfg.save_error_image and rgb_gt is not None:
                self.test_save_rendered_errors(
                    renders_list, rgb_gt, indices, output_path, scene_name, input_str,
                    vmin=0.0,
                    vmax=error_vmax,
                )

            # save depths
            if self.test_cfg.save_render_depth:
                self.test_save_rendered_depth(renders_list, indices, output_path, scene_name, input_str,
                                              vmin=depth_vmin, vmax=depth_vmax)

            # save gt depths
            if self.test_cfg.save_gt_depth and depth_gt is not None:
                self.test_save_gt_depth(depth_gt, indices, output_path, scene_name, input_str,
                                        vmin=depth_vmin, vmax=depth_vmax)

            # save video
            # TODO Naama: reorganize video rendering
            # Note: when video mode is enabled this returns early, skipping score computation.
            if module is not None and self.test_cfg.save_video and isinstance(output, OptimizerOutput):
                # Generate only for the first view in the batch
                # Generate a video with optimization trajectory for the first view (using ffmpeg)
                if input_str == "target":
                    if self.test_cfg.save_video_fixed_view:
                        self.render_supp_videos(batch, h, input_str, iterations, output.gaussian_list, output_path,
                                                scene_name, v, w, fixed_view_video=True, video_type="fixed_view")
                    if self.test_cfg.save_video_fixed_iteration:
                        for t in self.test_cfg.save_video_fixed_iteration_indices:
                            self.render_supp_videos(batch, h, input_str, iterations, output.gaussian_list, output_path,
                                                    scene_name, v, w,
                                                    fixed_view_video=self.test_cfg.save_video_fixed_iteration_render_fixed_view,
                                                    # render a fixed view until the required iteration
                                                    fixed_iteration_video=True,
                                                    fixed_iteration_indices=[t],
                                                    video_type="fixed_iteration")
                    if self.test_cfg.save_video_combined:
                        self.render_supp_videos(batch, h, input_str, iterations, output.gaussian_list, output_path,
                                                scene_name, v, w,
                                                fixed_view_video=True,
                                                fixed_iteration_video=True,
                                                fixed_iteration_indices=self.test_cfg.save_video_combined_iterations,
                                                fixed_iteration_length=self.test_cfg.save_video_combined_fixed_iteration_length,
                                                video_type="combined")
                return

            # Compute scores
            if self.test_cfg.compute_scores:
                print("\nComputing scores...")
                self._compute_and_save_scores(
                    module, output, renders_list, rgb_gt, iterations, input_str, module_name,
                    out_dir=output_path / "metrics" / scene_name,
                    extra_scene_metrics=extra_scene_metrics,
                )

        # Merge metrics from target (and context if evaluated) for combined plotting
        all_metrics = {}
        if self.test_cfg.compute_scores:
            for key, vals in self.test_step_outputs_target.items():
                if vals:
                    all_metrics[key] = vals[-1]
            for key, vals in self.test_step_outputs_context.items():
                if vals:
                    all_metrics[key] = vals[-1]
        return all_metrics

    @staticmethod
    def _plot_combined_metrics(
            output_path: Path,
            scene_name: str,
            phases: list[tuple[str, dict]],
    ):
        """Create a combined metrics plot for optimizer + postprocessing per scene.

        Args:
            output_path: Root output directory.
            scene_name: Name of the current scene.
            phases: List of (label, metrics_dict) tuples in order. Each metrics_dict
                    has keys like "{label}_psnr", "{label}_iterations", etc.
        """
        try:
            MetaTrainer._plot_combined_metrics_impl(output_path, scene_name, phases)
        except Exception as e:
            warn(f"[plot] failed to create combined metrics plot: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _plot_combined_metrics_impl(
            output_path: Path,
            scene_name: str,
            phases: list[tuple[str, dict]],
    ):
        from matplotlib import pyplot as plt

        # Filter out phases with no data
        phases = [(label, data) for label, data in phases if data]
        if not phases:
            print("[plot] No metrics data available, skipping combined plot.")
            return

        plot_metrics = ["psnr", "ssim", "alex_lpips", "vgg_lpips", "gaussians"]

        # Build combined series for each metric
        plots = []  # (title, x_values_list, y_values_list, labels_list, divider_x)
        for metric in plot_metrics:
            combined_x = []
            combined_y = []
            combined_labels = []
            x_offset = 0
            divider_x = None

            for phase_idx, (label, data) in enumerate(phases):
                iter_key = f"{label}_iterations"
                metric_key = f"{label}_{metric}"

                iterations = data.get(iter_key, [])
                values = data.get(metric_key, [])

                if not iterations or not values:
                    continue

                n = min(len(iterations), len(values))
                xs = [x_offset + iterations[j] for j in range(n)]
                ys = values[:n]

                combined_x.append(xs)
                combined_y.append(ys)
                combined_labels.append(label)

                if phase_idx < len(phases) - 1 and xs:
                    divider_x = xs[-1]
                    x_offset = divider_x

            if combined_x:
                plots.append((metric, combined_x, combined_y, combined_labels, divider_x))

        if not plots:
            print("[plot] No plottable metrics found, skipping combined plot.")
            return

        fig, axes = plt.subplots(len(plots), 1, figsize=(10, 3.5 * len(plots)), squeeze=False)
        axes = axes[:, 0]

        # Metrics where lower is better
        lower_is_better = {"alex_lpips", "vgg_lpips"}

        for ax, (metric_name, x_lists, y_lists, labels, divider_x) in zip(axes, plots):
            for xs, ys, label in zip(x_lists, y_lists, labels):
                ax.plot(xs, ys, marker=".", markersize=3, label=label)
            if divider_x is not None:
                ax.axvline(x=divider_x, color="gray", linestyle="--", linewidth=1, alpha=0.7)

            # Find and annotate the best value across all phases
            all_ys = [v for ys in y_lists for v in ys]
            if all_ys:
                if metric_name in lower_is_better:
                    best_val = min(all_ys)
                else:
                    best_val = max(all_ys)
                ax.axhline(y=best_val, color="red", linestyle=":", linewidth=1, alpha=0.6)
                ax.text(
                    1.0, best_val, f" best={best_val:.4f}",
                    transform=ax.get_yaxis_transform(),
                    va="bottom", ha="right", fontsize=7, color="red",
                )

            ax.set_title(metric_name)
            ax.set_xlabel("iteration")
            ax.set_ylabel(metric_name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Scene: {scene_name}", fontsize=12, y=1.0)
        fig.tight_layout()

        plot_dir = output_path / "plots" / scene_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / "combined_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined metrics plot to {save_path}")

    # region ==================== Save Results Methods =======================
    @staticmethod
    def test_save_rendered_images(renders_list: list, indices, output_path, scene_name, input_str):
        out_dir = output_path / "images" / scene_name / f"color_{input_str}"
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} images"):
            color = []
            for iter_renders in renders_list:
                iter_rgbs = iter_renders.color[0]  # (V, 3, H, W)
                color.append(iter_rgbs[i])
            color = torch.cat(color, dim=-1)  # concat along width
            save_image(color, out_dir / f"{index:06d}.png")
            del iter_rgbs
            del color
        # save last image separately too
        MetaTrainer.test_save_last_rendered_images(renders_list, indices, output_path, scene_name, input_str)

    @staticmethod
    def test_save_last_rendered_images(renders_list: list, indices, output_path, scene_name, input_str):
        out_dir = output_path / "images" / scene_name / "last" / f"color_{input_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} last images"):
            iter_renders = renders_list[-1]
            iter_rgbs = iter_renders.color[0]  # (V, 3, H, W)
            color = iter_rgbs[i]
            save_image(color, out_dir / f"{index:06d}.png")
            del iter_rgbs
            del color

    @staticmethod
    def test_save_gt_images(rgb_gt, indices, output_path, scene_name, input_str):
        out_dir = output_path / "images" / scene_name / f"color_{input_str}"
        for index, gt in tqdm(zip(indices, rgb_gt), desc=f"Saving {input_str} GT images"):
            save_image(gt, out_dir / f"{index:06d}_gt.png")

    @staticmethod
    def test_save_rendered_depth(renders_list: list, indices, output_path, scene_name, input_str,
                                 vmin: float = 0.0,
                                 vmax: float = 1.0):
        out_dir = output_path / "images" / scene_name / f"depth_{input_str}"
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} depths"):
            depth = []
            for iter_renders in renders_list:
                iter_depths = iter_renders.depth  # (1, V, 3, H, W)
                assert iter_depths is not None, "Depths not found in renders."
                iter_depths = iter_depths[0]  # (V, 3, H, W)
                depth.append(iter_depths[i])
            depth = torch.cat(depth, dim=-1)  # concat along width
            color = viz_depth_tensor(depth, return_numpy=False, as_uint8=False, vmin=vmin, vmax=vmax)
            save_image(color, out_dir / f"{index:06d}.png")
            del iter_depths
            del color

    @staticmethod
    def test_save_gt_depth(depth_gt, indices, output_path, scene_name, input_str, vmin: float = 0.0,
                           vmax: float = 1.0):
        out_dir = output_path / "images" / scene_name / f"depth_{input_str}"
        for index, gt in tqdm(zip(indices, depth_gt), desc=f"Saving {input_str} GT depths"):
            color = viz_depth_tensor(gt, return_numpy=False, as_uint8=False, vmin=vmin, vmax=vmax)
            save_image(color, out_dir / f"{index:06d}_gt.png")

    @staticmethod
    def test_save_rendered_errors(renders_list: list, rgb_gt, indices, output_path, scene_name, input_str,
                                  vmin: float = 0.0,
                                  vmax: float = 1.0):
        # rgb_gt is (V, 3, H, W). Looping per view (outer) and per iteration (inner)
        # keeps only one view's iterations in memory at a time.
        out_dir = output_path / "images" / scene_name / f"error_{input_str}"
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} errors"):
            error_maps = []
            for iter_renders in renders_list:
                iter_rgbs = iter_renders.color[0]  # (V, 3, H, W)
                error_maps.append((iter_rgbs[i] - rgb_gt[i].to(iter_rgbs)).abs().mean(0))  # (H, W)
            error_map = torch.cat(error_maps, dim=-1)  # concat along width
            color = viz_depth_tensor(error_map, return_numpy=False, as_uint8=False, colormap='magma',
                                     vmin=vmin, vmax=vmax)
            save_image(color, out_dir / f"{index:06d}.png")
            del iter_rgbs, error_maps, error_map

    def save_colmap_test_train_views(self, batch, h, w):
        # load the distortion parameters from the original colmap data
        assert self.test_cfg.ori_colmap_data_path is not None
        (scene_name,) = batch["scene"]
        output_path = self.test_cfg.output_path
        # training views
        input_images = batch["context"]["image"][0]  # [V, 3, H, W]
        index = batch["context"]["index"][0]
        for idx, color in zip(index, input_images):
            # NOTE: the original image id starts from 1
            save_image(color, output_path / scene_name / "images_train" / f"frame_{idx + 1:05d}.png")
        # testing views
        target_images = batch["target"]["image"][0]  # [V, 3, H, W]
        index = batch["target"]["index"][0]
        for idx, color in zip(index, target_images):
            # NOTE: the original image id starts from 1
            save_image(color, output_path / scene_name / "images_test" / f"frame_{idx + 1:05d}.png")
        # save the camera intrinsics
        intrinsics = batch["context"]["intrinsics"][0][0].clone()  # [3, 3]
        # need to rescale to the image size
        intrinsics[0, :] *= w
        intrinsics[1, :] *= h
        # distortion parameters
        json_path = os.path.join(self.test_cfg.ori_colmap_data_path, scene_name, "nerfstudio", "transforms.json")
        assert os.path.exists(json_path), f"Cannot find {json_path}"
        sparse_save_dir = output_path / scene_name / "sparse" / "0"
        sparse_save_dir_train = output_path / scene_name / "sparse_train" / "0"
        # save to cameras.bin
        save_opencv_camera(intrinsics.cpu().numpy(), json_path, sparse_save_dir, image_size=(w, h))
        save_opencv_camera(intrinsics.cpu().numpy(), json_path, sparse_save_dir_train, image_size=(w, h))
        # extract extrinsics from the dense view images.bin
        dense_view_extrinsics = os.path.join(self.test_cfg.ori_colmap_data_path, scene_name,
                                             "nerfstudio/colmap/sparse/0")
        selected_train_ids = [idx + 1 for idx in batch["context"]["index"][0].tolist()]
        selected_test_ids = [idx + 1 for idx in batch["target"]["index"][0].tolist()]
        selected_ids = selected_train_ids + selected_test_ids
        # also save the sparse features and points3D
        extract_sparse_images_bin(dense_view_extrinsics, sparse_save_dir, selected_ids, keep_features=False)
        # only for training views: reconstruct the sparse point cloud only from training views
        extract_sparse_images_bin(dense_view_extrinsics, sparse_save_dir_train, selected_train_ids)
        return

    def compute_depth_scores(self, batch, depth_gt, init_pred_depths):
        pred_depths = init_pred_depths[0]  # [V, H, W]
        depth_gt = depth_gt[0]  # [V, H, W]
        near = batch["context"]["near"][...,
        None, None][0]  # [V, 1, 1]
        far = batch["context"]["far"][..., None, None][0]  # [V, 1, 1]
        valid = (depth_gt >= near) & (depth_gt <= far)
        all_metrics = compute_depth_errors(depth_gt[valid].detach().cpu().numpy(),
                                           pred_depths[valid].detach().cpu().numpy())
        print(all_metrics)
        self.test_step_outputs_target[f"abs_rel"].append(
            float(all_metrics[0]))
        self.test_step_outputs_target[f"rmse"].append(float(all_metrics[2]))
        self.test_step_outputs_target[f"a1"].append(float(all_metrics[4]))

    def init_output_dict_for_new_scene(self, input_str, tag=None):
        tag = "" if tag is None else f"{tag}_"

        if input_str == "target":
            output_dict = self.test_step_outputs_target
        elif input_str == "context":
            output_dict = self.test_step_outputs_context
        else:
            raise ValueError(f"Unknown input_str={input_str}")

        for metric in ["psnr", "ssim", "lpips", "iterations", "time", "gaussians", "nonzero_grads"]:
            if metric == "lpips":
                # alex
                key = f"{tag}alex_lpips"
                output_dict[key].append([])
                # vgg
                key = f"{tag}vgg_lpips"
                output_dict[key].append([])
            else:
                key = f"{tag}{metric}"
                output_dict[key].append([])

    # endregion

    # region ==================== Video Rendering Methods ====================
    def render_supp_videos(self, batch, h, input_str, all_iterations, gaussian_list, output_path, scene_name,
                           v, w, fixed_view_video=False, fixed_iteration_video=False,
                           fixed_iteration_indices=None,
                           fixed_iteration_length=-1,
                           video_type=None):
        out_dir = output_path / "supp_videos" / scene_name
        out_dir.mkdir(parents=True, exist_ok=True)
        combined_iterations = []

        view = self.test_cfg.save_video_fixed_view_index

        all_frames = []

        duplicate = self.test_cfg.save_video_fixed_view_duplicate  # to focus on the optimization steps

        if fixed_iteration_length > 0:
            start = view
            end = view + fixed_iteration_length
        else:
            start = None
            end = None

        for i, t in enumerate(all_iterations):
            # Render only the view
            if fixed_view_video:
                decoder_output = self.test_render_videos_views(batch, gaussian_list[i], h, v, w, input_str,
                                                               start=view, end=view + 1)
                frames_t = decoder_output.color[0].detach().cpu()  # (1, 3, H, W)
                assert frames_t.shape[0] == 1, f"{frames_t.shape}"
                all_frames += [frames_t[0]] * duplicate  # (3, H, W)
                combined_iterations.extend([t] * duplicate)

            if fixed_iteration_video:
                if t in fixed_iteration_indices:
                    # Render a trajectory around the scene
                    decoder_output = self.test_render_videos_views(batch, gaussian_list[i], h, v, w, input_str,
                                                                   start=start, end=end)
                    frames_t = decoder_output.color[0].detach().cpu()  # (num_frames, 3, H, W)
                    for i in range(3):  # forward and backward
                        for frame in frames_t:
                            all_frames += [frame] * 3
                        for frame in frames_t.flip(0):
                            all_frames += [frame] * 3
                    combined_iterations.extend(['orbit'] * frames_t.shape[0] * 2)
                    if fixed_iteration_video and not fixed_view_video:
                        break  # no need to continue

        if video_type == "combined":
            save_str = f"_combined_{view}"
        elif video_type == "fixed_view":
            save_str = f"_fixed_view_{view}"
        elif video_type == "fixed_iteration":
            assert len(fixed_iteration_indices) == 1, f"{fixed_iteration_indices}"
            save_str = f"_fixed_iteration_{fixed_iteration_indices[0]}"
        else:
            raise ValueError
        save_video(all_frames, out_dir / f"{input_str}_{save_str}.mp4")
        with (open(out_dir / f"{input_str}_{save_str}_iterations.json", 'w')) as f:
            json.dump(combined_iterations, f, indent=4)

    def test_render_videos_views(self, batch, gaussians, h, v, w, input_str="target", poses=None, start=None, end=None):
        gaussians = gaussians.to(batch["target"]["image"].device)
        with self.benchmarker.time("decoder", num_calls=v):
            if poses is None:
                camera_poses = batch[input_str]["extrinsics"]

                if self.test_cfg.stablize_camera:
                    stable_poses = render_stabilization_path(
                        camera_poses[0].detach().cpu().numpy(),
                        k_size=self.test_cfg.stab_camera_kernel,
                    )

                    stable_poses = list(
                        map(
                            lambda x: np.concatenate(
                                (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                            ),
                            stable_poses,
                        )
                    )
                    stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                        camera_poses
                    )
                    camera_poses = stable_poses.unsqueeze(0)
            else:
                camera_poses = poses.unsqueeze(0)

            if self.test_cfg.render_chunk_size is not None:
                assert start is None
                assert end is None
                chunk_size = self.test_cfg.render_chunk_size
                num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                output = None
                for i in range(num_chunks):
                    start = chunk_size * i
                    end = chunk_size * (i + 1)
                    curr_output = self.scene_decoder.forward_batch(gaussians, batch, (h, w),
                                                                   input_str=input_str,
                                                                   start=start, end=end, camera_poses=camera_poses)

                    if i == 0:
                        output = curr_output
                    else:
                        # ignore depth
                        output.color = torch.cat((output.color, curr_output.color), dim=1)

            else:
                output = self.scene_decoder.forward_batch(gaussians.to(batch["target"]["image"].device),
                                                          batch, (h, w),
                                                          input_str=input_str,
                                                          camera_poses=camera_poses,
                                                          start=start,
                                                          end=end)
        return output

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
            self,
            batch: BatchedExample,
            trajectory_fn: TrajectoryFn,
            name: str,
            num_frames: int = 30,
            smooth: bool = True,
            loop_reverse: bool = True,
    ) -> None:
        if self.train_cfg.no_log_video:
            return
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.reshape(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.scene_decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        rgb_pred = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(rgb_pred, rgb_pred)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    # endregion

    # region ==================== Delegation Methods =========================

    def get_optimized_gaussians(self, *args, **kwargs):
        """Delegate to SceneTrainer's get_optimized_gaussians."""
        return self.scene_trainer.get_optimized_gaussians(*args, **kwargs)

    def get_init_gaussians(self, *args, **kwargs):
        """Delegate to SceneTrainer's get_init_gaussians."""
        return self.scene_trainer.get_init_gaussians(*args, **kwargs)

    def init_gaussians_and_render(self, *args, **kwargs):
        """Delegate to SceneTrainer's init_gaussians_and_render."""
        return self.scene_trainer.init_gaussians_and_render(*args, **kwargs)

    def test_postprocess_gaussians(self, *args, **kwargs):
        """Delegate to SceneTrainer's test_postprocess_gaussians."""
        return self.scene_trainer.test_postprocess_gaussians(*args, **kwargs)

    @property
    def scene_initializer(self):
        """Delegate to SceneTrainer's initializer."""
        return self.scene_trainer.initializer

    @property
    def scene_optimizer(self):
        """Delegate to SceneTrainer's optimizer."""
        return self.scene_trainer.optimizer

    @property
    def scene_decoder(self):
        """Delegate to SceneTrainer's decoder."""
        return self.scene_trainer.decoder

    # endregion

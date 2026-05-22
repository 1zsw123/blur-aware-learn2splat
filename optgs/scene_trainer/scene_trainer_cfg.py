from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from optgs.meta_trainer.replay_buffer import ReplayBufferCfg
from .initializer import InitializerCfg
from .optimizer import SceneOptimizerCfg
from .postprocessing import PostProcessCfg
from ..model.decoder import DecoderCfg
from ..model.decoder.decoder import DepthRenderingMode


@dataclass
class SceneTrainerCfg:
    scene_initializer: InitializerCfg
    scene_optimizer: SceneOptimizerCfg | None
    decoder: DecoderCfg
    use_fsdp: bool
    train_scene_init: bool
    train_scene_opt: bool
    num_update_steps: int
    iter_batch_size: int  # if -1, use full batch

    train_min_refine: int
    train_max_refine: int

    opt_batch_size: int  # if -1, use full batch
    opt_batch_size_min: int  # if > 0, use random sub-batch
    opt_batch_size_max: int  # if > 0, use random sub-batch
    opt_batch_strategy: Literal["random", "sequential", "neighbors", "fps"]  # strategy for sub-batch
    sh_degree_interval: int  # 0 = disabled; N = steps between SH degree increments (like gsplat simple_trainer)

    def __post_init__(self):
        if self.scene_optimizer is not None:
            self.scene_optimizer.update(self.scene_initializer)


# TODO Naama, probably need to move into meta_trainer cfg file

@dataclass
class MetaOptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    lr_depth: float
    weight_decay: float
    warm_up_ratio: float
    adamw_8bit: bool


@dataclass
class TestCfg:
    output_path: Path | None
    compute_scores: bool
    compute_scores_metrics: list[str] | None
    metrics_batch_size: int
    eval_initialization: bool
    save_render_image: bool
    save_render_image_last_only: bool
    save_gt_image: bool
    save_render_depth: bool
    save_gt_depth: bool
    save_error_image: bool
    save_video: bool
    save_video_fixed_view: bool
    save_video_fixed_view_index: int
    save_video_fixed_view_duplicate: int
    save_video_fixed_iteration: bool
    save_video_fixed_iteration_indices: list | None
    save_video_fixed_iteration_render_fixed_view: bool
    save_video_combined: bool
    save_video_combined_iterations: list | None
    save_video_combined_fixed_iteration_length: int
    eval_time_skip_steps: int
    save_gaussian: bool
    save_poses: bool
    save_cameras_json: bool
    save_cameras_npz: bool
    save_point_cloud: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int
    eval_context_views: bool
    inference_window_size: int | None
    profile_model: bool
    save_colmap_train_test_views: bool
    ori_colmap_data_path: str | None
    postprocessing: PostProcessCfg | None
    save_at_iters: list[int] | None
    save_every_freq: list[int] | None
    save_every_steps: list[int] | None
    skip_if_outputs_exist: bool
    scenes_filter: list[str] | None

    experimental_add_noise_to_images: bool
    experimental_add_noise_to_images_std: float | int | None


# TODO Naama split into scene and meta trainer cfgs
@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_deterministic: bool
    eval_time_skip_steps: int
    eval_save_model: bool
    l1_loss: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    eval_depth: bool
    train_ignore_large_loss: float
    no_log_projections: bool

    depth_loss_weight: float
    log_depth_loss: bool
    depth_smooth_loss_weight: float
    depth_teacher_loss_weight: float
    viz_depth_teacher: bool
    depth_smooth_loss_nonorm: bool
    depth_smooth_loss_weight_nvs: float  # for novel views
    monodepth_loss_weight: float  # for monocular depth loss

    eval_render_depth: bool
    render_depth_loss_weight: float
    viz_render_depth: bool
    viz_depth_separate: bool

    use_gt_depth_range: bool
    depth_range_from_disparity: bool
    max_disparity: float
    min_disparity: float

    no_log_video: bool

    # when doing refinement, supervise input view or not since we also render input views
    loss_on_target_views: bool
    loss_on_target_views_num: int
    loss_on_input_views: bool
    loss_on_input_views_num: int

    # half res lpips loss to save memory
    half_res_lpips_loss: bool

    # local window training
    train_window_size: int | None

    # Replay buffer
    use_replay_buffer: bool
    replay_buffer_cfg: ReplayBufferCfg | None

    # L2 weight decay regularization on Gaussian properties (meta-loss)
    scale_l2_loss_weight: float
    sh_l2_loss_weight: float
    opacity_l2_loss_weight: float

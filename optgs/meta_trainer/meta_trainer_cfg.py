from dataclasses import dataclass
from pathlib import Path

from optgs.meta_trainer.ckpt_buffer import CkptBufferCfg
from optgs.scene_trainer.postprocessing import PostProcessCfg
from optgs.model.decoder.decoder import DepthRenderingMode


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
    save_init_pred_depth: bool  # save the initializer's predicted input-view depth (not the rendered depth)
    save_error_image: bool
    save_video: bool
    save_video_optim: bool
    save_video_view_index: int
    save_video_frame_repeat: int
    save_video_orbit: bool
    save_video_orbit_steps: list | None
    save_video_orbit_with_optim: bool
    save_video_optim_orbit: bool
    save_video_optim_orbit_steps: list | None
    save_video_orbit_span: int
    eval_time_skip_steps: int
    save_gaussian: bool
    save_poses: bool
    save_cameras_json: bool
    save_cameras_npz: bool
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


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_time_skip_steps: int
    eval_save_model: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    eval_depth: bool
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

    no_log_video: bool

    # when doing refinement, supervise input view or not since we also render input views
    loss_on_target_views: bool
    loss_on_target_views_num: int
    loss_on_input_views: bool
    loss_on_input_views_num: int

    # local window training
    train_window_size: int | None

    # Ckpt buffer
    use_ckpt_buffer: bool
    ckpt_buffer_cfg: CkptBufferCfg | None

    # L2 weight decay regularization on Gaussian properties (meta-loss)
    scale_l2_loss_weight: float
    sh_l2_loss_weight: float
    opacity_l2_loss_weight: float
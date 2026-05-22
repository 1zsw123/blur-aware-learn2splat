import math
import random
from dataclasses import dataclass
from typing import Literal, Optional, Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
from einops import rearrange
from torch import nn, Tensor

from optgs.dataset.data_types import BatchedExample, DataShim
from optgs.dataset.data_types import BatchedViews
from optgs.dataset.shims.patch_shim import apply_patch_shim
from optgs.geometry.projection import project, sample_image_grid
from optgs.misc.general_utils import SkipBatchException
from optgs.misc.io import FrequencyScheduler
from optgs.model.decoder.decoder import Decoder
from optgs.model.encoder.layer import ResNetFeatureWarpper
from optgs.model.types import Gaussians
from optgs.scene_trainer.common.gaussian_adapter import build_covariance
from optgs.scene_trainer.initializer import InitializerCfg, InitializerColmapCfg, InitializerEdgsCfg, \
    InitializerRandomCfg, InitializerPointcloudCfg
from optgs.scene_trainer.initializer import InitializerPlyCfg
from optgs.scene_trainer.initializer.initializer_resplat import ResplatInitializerCfg
from optgs.scene_trainer.optimizer.optimizer import OptimizerInput, LearnedOptimizer, OptimizerOutput, OptimizerState, \
    OptimizerPreviousOutput, OptimizerCfg
from optgs.scene_trainer.optimizer.optimizer_utils import Number3DGSCfg, Bool3DGSCfg
from optgs.scene_trainer.optimizer.optimizer_utils import unpack_gaussians, \
    get_visibility_contribution_from_gaussian_obj

try:
    from optgs.model.encoder.point_transformer.layer import (PlainPointTransformer, SubsampleBlock, PointLinearWrapper,
                                                             MultiScalePointTransformer,
                                                             MultViewLowresAttn)
except:
    pass

try:
    from simple_knn._C import distCUDA2
except:
    pass

from optgs.scene_trainer.optimizer.layer import CustomGroupNorm, AdamInputSmoothing, SlicedG3RNorm
from optgs.scene_trainer.initializer.initializer import InitializerOutput
from optgs.scene_trainer.optimizer.time_embed import get_embedder, TimeEncodingWrapper

from optgs.loss.loss_depth_smooth import get_smooth_loss
from optgs.scene_trainer.optimizer.optimizer_utils import (
    inner_loss_for_input_gradients,
    chunk_index_iter,
    split_grads,
    get_gaussian_param_slices,
    get_gaussian_param_sizes,
    pack_gaussians,
)

_IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@dataclass
class KnnBasedOptimizerCfg(OptimizerCfg):
    name: Literal["knn_based", "resplat_v1", "resplat_v2", "clogs", "l2s"]  # TODO (release) remove clogs
    # iterative refine
    no_render_error: bool
    input_error_shallow_resnet_feature: bool
    input_error_resnet_feature_layers: int
    refine_sh_only: bool
    num_basic_refine_blocks: int
    num_refine_blocks: int
    concat_init_state: bool  # always concat init state during updates
    replace_init_state: bool  # always use the init state during updates
    state_channels: int
    refine_block_rmsnorm: bool
    refine_block_layernorm: bool
    pt_qk_norm: bool
    norm_pt_block: bool
    refine_gaussian_multiple: int  # predict more gaussian residuals based on the previous gaussian center
    refine_residual_init_state: bool  # add residual connection in the prediction head to the inital state
    clamp_refine_max_scale: float
    clamp_min_scale: float | int
    clamp_min_raw_scales: float | int
    clamp_max_raw_scales: float | int
    clamp_min_raw_opacities: float | int
    clamp_max_raw_opacities: float | int
    clamp_min_sh0: float | int
    clamp_max_sh0: float | int
    clamp_min_shs: float | int
    clamp_max_shs: float | int
    clamp_shs_soft: bool

    gaussian_head_multiple: int  # use multiple non-weight sharing heads to predict multiple gaussians
    gradient_update_scale: float | int
    input_gradient_with_ssim_loss: bool
    update_attn_proj_channels: int | None
    update_no_knn_attn: bool
    update_no_tran_block_norm: bool
    update_tran_block_act: str | None
    multi_gaussian_scale_smaller: bool
    refine_condition_pt_feature: bool
    reinit_gaussian_when_refine_multiple: bool
    refine_same_num_points: bool  # when init_gaussian_multiple > 1, refine directly works on it instead of subsampling points

    refine_knn_samples: int
    refine_multi_scale_pt: bool

    # KNN
    use_fused_attn: bool
    prune_invisible_gaussians: bool
    knn_idx_update_every: int

    # point transformer
    pt_heads: int

    # inputs
    input_alpha: bool
    input_depth: bool
    input_depth_smooth_error: bool

    # input error
    input_error: bool  # render error as input to the refine head
    input_error_rgb_no_shuffle: bool  # sample single pixel instead of pixel unshuffling
    input_error_add_rgb_feature: bool

    # resnet
    input_error_resnet_feature: bool
    input_error_cache_resnet_feature: bool
    input_error_no_freeze_resnet_feature: bool

    # number of views for render error
    input_error_num_views: int
    input_error_additional_cross_attn: bool
    input_error_num_intermediate_views: int

    # render error with remaining context views
    input_error_remain_context: bool
    input_error_merge_remain_context: bool
    input_error_warp_remain_context: bool
    input_error_random_num_remain_context: bool
    input_error_num_remain_context_test: int

    # render error mv attn
    input_error_mv_attn: bool
    input_error_mv_attn_blocks: int

    # refine global attention
    refine_with_mv_attn: bool
    refine_with_mv_attn_lowres: bool
    refine_no_mv_attn: bool  # remove only the attn
    mv_attn_conv_with_norm: bool  # unet-attn conv with norm
    refine_mv_shuffle_attn: bool  # use pixel shuffle to save computation instead of unet
    refine_mv_attn_with_pos_enc: bool
    refine_shuffle_attn_no_norm: bool
    refine_mv_unimatch_attn: bool

    # input gradients
    input_gradient: bool
    input_gradient_log: bool
    input_gradient_log_clip_deltas: float | int
    input_gradient_scale: float | int
    input_gradient_same_loss: bool  # use the same loss as the gaussian update
    input_gradient_loss_reduction: str
    scale_residual_grads: bool

    # sliding window
    window_local_refine: bool  # refine each local window separately and then combine all windows
    window_global_refine: bool  # refine all windows together
    window_local_global_refine: bool  # first refine each window seprately, and then refine all windows together

    # sliding window update instead of update all gaussians together
    update_window_size: int
    local_gaussian_render: bool

    # time encoding
    use_time_encoding: bool
    time_encoding_max_steps: int

    train_global_update_only: bool

    # random size refine
    # update more for low resolution, less for high
    random_update_with_size: bool

    # amp
    use_amp: bool
    pt_head_amp: bool
    pt_update_amp: bool

    use_checkpointing: bool
    recurrent_use_checkpointing: bool

    # Debugging
    debug_refine_update_module: bool

    # Normalizing input
    input_gradient_normalize: bool
    input_gradient_normalize_type: str
    input_normalize_state: bool
    input_normalize_gaussians: bool

    # State scaling
    predict_state_scale: bool
    predict_state_scale_norm: bool  # whether to normalize the state before scaling

    # Use optimizer without condition features
    init_state_wo_features: bool
    init_state_type: Literal["random", "constant"]
    init_state_scale: float | int

    opt_scales_before_act: bool  # optimize scale before activation (raw -> exp -> scale -> log -> raw)

    # Preprocessing the init gaussians
    scale_initial_opacities: float | int

    # Experimental
    experimental_run: bool
    experimental_update: Bool3DGSCfg
    experimental_use_grads: bool
    experimental_use_norm_grads: Bool3DGSCfg
    experimental_lr: Number3DGSCfg
    # Deactivate gaussians
    local_prune_zero_radii: bool
    local_prune_low_weights: bool
    local_prune_low_weights_thresh: float | int
    update_only_nonzero_grad: bool

    # update learn residual state
    residual_state: bool

    # Update head
    update_head_layer_num: int
    update_head_concat_img: bool
    update_head_act: str | None  # update_head activation to predict the deltas
    update_head_final_act: str | None  # final activation in the update_head
    update_head_hidden_dim_matches: str  # rebuttal or submission version

    update_head_scale_mag: bool  # predict deltas as scale * 0.01 * jnp.exp(mag * 0.01)
    update_head_scalar_scale: bool  # predict deltas as scalar * delta / norm(delta)
    update_head_scalar_scale_act: str  # activation for the scalar scale output

    # Per-parameter-group update head (Feature A)
    update_head_per_param_heads: bool  # separate heads per param group, each with own normalize+scale
    update_head_per_param_hidden_dim: int  # hidden dim for per-param heads (SH head gets 2x)
    # Per-parameter scalar scales (Feature B) — requires update_head_scalar_scale=true
    update_head_per_param_scales: bool  # per-group scalar scales instead of one global scalar

    # Config from initializer
    sh_d: int | None
    init_gaussian_param_num: int | None = None
    init_sh_d: int | None = None
    # Fow initialization from feed forward, gaussians are aligned with pixels.
    init_gaussian_multiple: int | None = None
    latent_downsample: int | None = None

    delta_adam_combine_step: int = 0  # combine deltas and adam updates

    def update(self, initializer_cfg: InitializerCfg):
        """ Update the optimizer config based on the initializer config"""

        # General settings
        self.init_gaussian_param_num = initializer_cfg.get_gaussian_param_num()
        self.init_sh_d = initializer_cfg.get_sh_d()
        if self.sh_d is None:
            # get sh_d from initializer if not set
            self.sh_d = initializer_cfg.get_sh_d()

        # Settings specific to DepthSplat initializer
        if isinstance(initializer_cfg, ResplatInitializerCfg):
            self.latent_downsample = initializer_cfg.latent_downsample
            self.init_gaussian_multiple = initializer_cfg.init_gaussian_multiple

            # update proj channels
            if self.refine_condition_pt_feature:
                self.condition_channels = initializer_cfg.gaussian_regressor_channels
            else:
                self.condition_channels = initializer_cfg.get_pt_in_channels()
        # Settings specific to Colmap initializer
        elif isinstance(initializer_cfg,
                        (InitializerPlyCfg, InitializerColmapCfg, InitializerEdgsCfg, InitializerRandomCfg,
                         InitializerPointcloudCfg)):
            # Since pixels and gaussians are not alligned, we can not use pixel attributes
            assert not self.input_error, "The error calculation assumes per pixel gaussians"
            assert not self.update_head_concat_img
            assert not self.input_alpha
            assert not self.local_gaussian_render, "The local rendering assumes per view gaussians"

            assert self.init_state_wo_features, "Colmap initializer does not have point features, init_state_wo_features must be set to True"

            self.init_gaussian_multiple = 1
            self.latent_downsample = 1
        else:
            raise ValueError(f"Unsupported initializer config type: {type(initializer_cfg)}")


class KnnBasedOptimizerState:
    # TODO Naama: OptimizerState class already exists
    def __init__(self, state: torch.Tensor):
        self.state = state

    def clone(self, clone_mask: torch.Tensor, zero_t: bool) -> None:
        cloned_state = self.state[clone_mask]
        if zero_t:
            cloned_state = torch.zeros_like(cloned_state)
        self.state = torch.cat([self.state, cloned_state], dim=0)

    def split(self, split_mask, num_splits: int, zero_t: bool) -> None:
        states_to_split = self.state[split_mask]
        split_states = states_to_split.chunk(num_splits, dim=0)
        new_states = []
        for i in range(num_splits):
            if zero_t:
                new_states.append(torch.zeros_like(split_states[i]))
            else:
                new_states.append(split_states[i])
        self.state = torch.cat([self.state, *new_states], dim=0)

    def replace(self, from_indices: torch.Tensor, dest_indices: torch.Tensor, zero_t: bool) -> None:
        if zero_t:
            self.state[dest_indices] = 0.0
        else:
            self.state[dest_indices] = self.state[from_indices]

    def prune(self, prune_mask: torch.Tensor) -> None:
        self.state = self.state[~prune_mask]

    def add(self, num_new: int) -> None:
        if num_new <= 0:
            return
        device = self.state.device
        dtype = self.state.dtype
        input_dim = self.state.shape[1:]
        self.state = torch.cat([self.state, torch.zeros((num_new, *input_dim), device=device, dtype=dtype)], dim=0)

    def extend(self, num_new):
        self.add(num_new)


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


def get_activation_cls(activation: Optional[str] = None):
    if activation in ['none', None, 'identity']:
        return nn.Identity
    elif activation == 'tanh':
        return nn.Tanh
    elif activation == "gelu":
        return nn.GELU
    elif activation == 'sigmoid':
        return nn.Sigmoid
    elif activation == 'relu':
        return nn.ReLU
    elif activation == "softplus":
        return nn.Softplus
    elif activation == "abs":
        return Abs
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class KnnBasedOptimizer(LearnedOptimizer[KnnBasedOptimizerCfg]):
    OPTIMIZER_NAME = "knn_based"
    OPTIMIZER_NAME_ALIASES: tuple[str, ...] = ()

    def __init__(self, cfg: KnnBasedOptimizerCfg, save_every: Optional[FrequencyScheduler] = None) -> None:
        valid = {self.OPTIMIZER_NAME, *self.OPTIMIZER_NAME_ALIASES}
        assert cfg.name in valid, f"Expected optimizer name {valid}, got {cfg.name}"

        super().__init__(cfg, save_every)

        if self.cfg.residual_state:
            assert not self.cfg.refine_residual_init_state

        # State channel
        self.state_channels = self.cfg.state_channels

        # time embedder
        if self.cfg.use_time_encoding:
            self.time_encoder_fn, self.time_embedding_dim = get_embedder(multires=6)
        else:
            self.time_encoder_fn = None
            self.time_embedding_dim = 0

        # update_proj
        if not self.cfg.init_state_wo_features:
            self.update_proj = nn.Conv2d(self.cfg.condition_channels, self.state_channels, 1)

        channels, in_channels, update_gaussian_param_num, out_channels, error_features_channels = (
            self.define_update_channels(self.cfg.init_gaussian_param_num))
        self.error_features_channels = error_features_channels
        self.gaussian_param_num = out_channels

        if self.cfg.input_error:

            self.update_feature = self.get_input_error_feature_extractor()
            if self.cfg.input_error_add_rgb_feature:
                if self.cfg.init_gaussian_multiple == 4:  # re10k
                    self.update_rgb_error_proj = nn.Sequential(
                        nn.Linear(3, error_features_channels),
                        nn.LayerNorm(error_features_channels)
                    )
                else:
                    self.update_rgb_error_proj = nn.Sequential(
                        nn.Linear(3 * self.cfg.latent_downsample ** 2, error_features_channels),
                        nn.LayerNorm(error_features_channels)
                    )
        self.update_input_norm = self.get_update_input_norm(in_channels)
        self.update_module = self.get_update_module(channels, in_channels)

        # predict multiple gaussians
        out_channels = out_channels * self.cfg.refine_gaussian_multiple

        if not self.cfg.refine_same_num_points:
            out_channels = out_channels * self.cfg.init_gaussian_multiple

        # make sure the input size of the gaussian head is updated accordingly
        if self.cfg.use_time_encoding:
            channels += self.time_embedding_dim

        # Compute per-param group dims (needed by per_param_heads and per_param_scales)
        if self.cfg.update_head_per_param_heads or self.cfg.update_head_per_param_scales:
            self._per_param_group_dims = self._compute_per_param_group_dims(out_channels)

        # Scaling state for update head
        if self.cfg.predict_state_scale:
            self.state_scale_head = self.get_state_scale_head(in_channels)

        self.update_head = self.get_update_head(in_channels, channels, out_channels)

        # multiple gaussian heads to predict multiple gaussians
        if self.cfg.gaussian_head_multiple > 1:
            self.update_head_list = self.get_update_head_list(channels, out_channels)

        # Define error calculation
        # add global attention to the render error
        if self.cfg.input_error and self.cfg.input_error_mv_attn:
            assert self.cfg.input_error_resnet_feature
            self.update_error_attn = nn.ModuleList([
                MultViewLowresAttn(error_features_channels)
                for _ in range(self.cfg.input_error_mv_attn_blocks)
            ])

        self.param_slices = get_gaussian_param_slices(self.cfg.sh_d)

    def _reset_knn_caches(self) -> None:
        """Invalidate cached KNN indices on all point-transformer sub-modules.

        Must be called whenever the number of Gaussians changes (e.g. after add_new)
        so the next forward recomputes KNN from scratch instead of using stale indices
        that index out-of-bounds into the grown point cloud.
        """
        for module in self.modules():
            if hasattr(module, "cache_knn_idx"):
                module.cache_knn_idx = None

    @property
    def adc_object_dict_to_adjust(self):
        if self.cfg.any_adc:
            object_dict: dict[str, Any] = {"depthsplat_state": None}
            # For ADC
            if self.cfg.input_gradient_normalize and self.cfg.input_gradient_normalize_type == "adam":
                object_dict.update(self.update_input_norm.subgroups_view(self.param_slices))
        else:
            return None

        return object_dict

    def _compute_per_param_group_dims(self, out_channels):
        """Compute per-parameter-group output dimensions from total out_channels.

        Returns a dict {group_name: dim} in the same order as split_delta_gaussians.
        Accounts for no_refine_rotation, no_refine_mean, refine_sh_only, and multipliers.
        """

        # TODO Naama: allow combination of no_refine_*
        p = get_gaussian_param_sizes(self.cfg.sh_d)

        all_params = [
            ("means", "means"),
            ("scales", "scales"),
            ("rotations", "quats"),
            ("opacities", "opacities"),
            ("shs", "shs"),
        ]

        if self.cfg.refine_sh_only:
            excluded = {"means", "scales", "rotations", "opacities"}
        elif self.cfg.no_refine_rotation:
            excluded = {"rotations"}
        elif self.cfg.no_refine_mean:
            excluded = {"means"}
        else:
            excluded = set()

        multiplier = self.cfg.refine_gaussian_multiple
        if not self.cfg.refine_same_num_points:
            multiplier *= self.cfg.init_gaussian_multiple

        group_dims = {name: p[key] * multiplier for name, key in all_params if name not in excluded}

        assert sum(group_dims.values()) == out_channels, (
            f"Per-param group dims {dict(group_dims)} sum={sum(group_dims.values())} != out_channels={out_channels}"
        )
        return group_dims

    def _build_per_param_heads(self, channels, out_channels):
        """Build per-parameter-group heads (Feature A).

        Each head: Linear(channels, hidden) -> act -> Linear(hidden, dim+1)
        The +1 is a per-group scalar scale. Each head independently normalizes + scales.
        """
        act_cls = get_activation_cls(self.cfg.update_head_act)
        hidden_dim = self.cfg.update_head_per_param_hidden_dim

        # Set up scale activation (shared across all per-param heads)
        scale_act_name = self.cfg.update_head_scalar_scale_act
        init_bias_map = {'softplus': -1, 'relu': 1e-8, 'abs': 1e-8}
        if scale_act_name not in init_bias_map:
            raise ValueError(f"Unsupported scalar_scale_act: {scale_act_name}")
        act_class = get_activation_cls(scale_act_name)
        self.scale_act = act_class(beta=1) if scale_act_name == 'softplus' else act_class()

        heads = nn.ModuleDict()
        for name, dim in self._per_param_group_dims.items():
            # SH head gets 2x hidden dim (more outputs to predict)
            h = hidden_dim * 2 if name == "shs" else hidden_dim

            layers = [nn.Linear(channels, h), act_cls()]
            for _ in range(self.cfg.update_head_layer_num - 2):
                layers += [nn.Linear(h, h), act_cls()]
            layers.append(nn.Linear(h, dim + 1))  # +1 for scalar scale

            head = nn.Sequential(*layers)

            # Zero-init last layer (deltas start at 0)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            # Init scale bias
            nn.init.constant_(head[-1].bias[-1], init_bias_map[scale_act_name])

            heads[name] = head

        return heads

    def get_update_head(self, in_channels, channels, out_channels):
        update_head_activation_cls = get_activation_cls(self.cfg.update_head_act)
        final_head_activation_cls = get_activation_cls(self.cfg.update_head_final_act)

        # skip connection to the image color
        if self.cfg.update_head_concat_img:
            channels += 3 * (self.cfg.latent_downsample ** 2)

        # Feature A: per-parameter-group heads (early return — builds ModuleDict instead of Sequential)
        if self.cfg.update_head_per_param_heads:
            assert not self.cfg.update_head_scale_mag, "update_head_scale_mag not supported with per_param_heads"
            assert not self.cfg.update_head_per_param_scales, "per_param_heads already includes per-group scales"
            return self._build_per_param_heads(channels, out_channels)

        # predict delta = scale * 0.01 * jnp.exp(mag * 0.01)
        if self.cfg.update_head_scale_mag:
            out_channels = out_channels * 2

        if self.cfg.update_head_scalar_scale:
            if self.cfg.update_head_per_param_scales:
                # Feature B: one scalar scale per parameter group
                out_channels = out_channels + len(self._per_param_group_dims)
            else:
                out_channels = out_channels + 1

        # Determine hidden layer size
        # TODO: update_head_hidden_dim_source should be "output" (out_channels).
        #       Using "input" currently as default to reproduce rebuttal results.
        if self.cfg.update_head_hidden_dim_matches == "input":
            hidden_dim = channels  # rebuttal version
        else:
            hidden_dim = out_channels  # submitted version

        # Build update head
        layers_list = [
            nn.Linear(channels, hidden_dim),
            update_head_activation_cls()
        ]
        for i in range(self.cfg.update_head_layer_num - 2):
            layers_list += [
                nn.Linear(hidden_dim, hidden_dim),
                update_head_activation_cls(),
            ]

        layers_list += [
            nn.Linear(hidden_dim, out_channels),
            final_head_activation_cls()
        ]
        update_head = nn.Sequential(*layers_list)

        # init the delta as 0
        nn.init.zeros_(update_head[-2].weight)
        if final_head_activation_cls == torch.nn.Sigmoid:
            desired_init_delta = 0.005
            bias = math.log(desired_init_delta / (1 - desired_init_delta))  # ~= -4.6
            nn.init.constant_(update_head[-2].bias, bias)
        else:
            nn.init.zeros_(update_head[-2].bias)

        # Scalar scale output
        if self.cfg.update_head_scalar_scale:
            # Set the initial scale to very low number, to get the gradients flow
            init_bias_map = {
                'softplus': -1,
                'relu': 1e-8,
                'abs': 1e-8,
            }

            act_name = self.cfg.update_head_scalar_scale_act
            if act_name not in init_bias_map:
                raise ValueError(f"Unsupported scalar_scale_out_act: {act_name}")

            # Initialize bias for scale output(s)
            if self.cfg.update_head_per_param_scales:
                num_groups = len(self._per_param_group_dims)
                for i in range(num_groups):
                    nn.init.constant_(update_head[-2].bias[-(num_groups - i)], init_bias_map[act_name])
            else:
                nn.init.constant_(update_head[-2].bias[-1], init_bias_map[act_name])

            # Create activation
            act_class = get_activation_cls(act_name)
            self.scale_act = act_class(beta=1) if act_name == 'softplus' else act_class()

        return update_head

    def get_update_head_list(self, channels, out_channels):
        update_head_activation = get_activation_cls(self.cfg.update_head_act)
        final_head_activation = get_activation_cls(self.cfg.final_head_act)
        update_head_list = nn.ModuleList()
        for i in range(self.cfg.gaussian_head_multiple - 1):
            update_head_list.append(
                nn.Sequential(
                    nn.Linear(channels, channels),
                    update_head_activation(),
                    nn.Linear(channels, out_channels),
                    final_head_activation()
                )
            )

            # init the delta as 0
            nn.init.zeros_(update_head_list[i][-2].weight)
            nn.init.zeros_(update_head_list[i][-2].bias)

        return update_head_list

    def get_update_input_norm(self, in_channels):
        if self.cfg.input_gradient_normalize:
            assert self.cfg.input_gradient, "for now we only normalize when using gradient as input"
            if self.cfg.input_gradient_normalize_type == 'layer':
                return nn.LayerNorm(in_channels)
            elif self.cfg.input_gradient_normalize_type == 'group':
                return CustomGroupNorm([self.gaussian_param_num, self.state_channels, self.gaussian_param_num])
            elif self.cfg.input_gradient_normalize_type == 'batch':
                return nn.BatchNorm1d(in_channels, affine=False)
            elif self.cfg.input_gradient_normalize_type == 'g3r':
                return SlicedG3RNorm(in_channels, slice(-self.gaussian_param_num, None))
            elif self.cfg.input_gradient_normalize_type == 'adam':
                assert not self.cfg.input_gradient_log and self.cfg.input_gradient_scale == 1
                return AdamInputSmoothing(input_slice=slice(-self.gaussian_param_num, None))
            else:
                raise ValueError(f"normalization type not supported {self.cfg.input_gradient_normalize_type}")
        else:
            return nn.Identity()

    def get_update_module(self, channels, in_channels):
        if not self.cfg.debug_refine_update_module:
            return None

        if self.cfg.refine_multi_scale_pt:
            update_module = nn.Sequential(
                PointLinearWrapper(in_channels, channels),
                MultiScalePointTransformer(channels,
                                           self.cfg.refine_knn_samples,
                                           subsample_method=self.cfg.subsample_method,
                                           attn_proj_channels=self.cfg.update_attn_proj_channels,
                                           )
            )
        else:
            update_module = nn.Sequential(
                PointLinearWrapper(in_channels, channels),
                PlainPointTransformer(channels, self.cfg.refine_knn_samples,
                                      num_blocks=self.cfg.num_basic_refine_blocks,
                                      qk_norm=self.cfg.pt_qk_norm,
                                      norm_pt_block=self.cfg.norm_pt_block,
                                      num_heads=self.cfg.pt_heads,
                                      no_rpe=True,
                                      no_attn=self.cfg.update_no_knn_attn,
                                      no_norm=self.cfg.update_no_tran_block_norm,
                                      act=self.cfg.update_tran_block_act,
                                      attn_proj_channels=self.cfg.update_attn_proj_channels,
                                      with_mv_attn=self.cfg.refine_with_mv_attn,
                                      with_mv_attn_lowres=self.cfg.refine_with_mv_attn_lowres,
                                      no_mv_attn=self.cfg.refine_no_mv_attn,
                                      conv_with_norm=self.cfg.mv_attn_conv_with_norm,
                                      mv_shuffle_attn=self.cfg.refine_mv_shuffle_attn,
                                      with_pos_enc=self.cfg.refine_mv_attn_with_pos_enc,
                                      shuffle_attn_no_norm=self.cfg.refine_shuffle_attn_no_norm,
                                      mv_unimatch_attn=self.cfg.refine_mv_unimatch_attn,
                                      use_checkpointing=self.cfg.use_checkpointing,
                                      use_fused_attn=self.cfg.use_fused_attn,
                                      knn_idx_update_every=self.cfg.knn_idx_update_every
                                      )
            )

        # Init normalization layers
        if self.cfg.input_normalize_state:
            for block in update_module[1].blocks:
                nn.init.zeros_(block.norm1.bias)
                nn.init.zeros_(block.norm2.bias)
                nn.init.ones_(block.norm1.weight)
                nn.init.ones_(block.norm2.weight)

        return update_module

    def get_state_scale_head(self, in_channels):
        state_scale_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
            nn.ReLU()
        )

        # Init the scale to 1
        # nn.init.zeros_(state_scale_head[-2].weight)
        nn.init.ones_(state_scale_head[-2].bias)

        return state_scale_head

    def define_update_channels(self, init_gaussian_param_num):
        if self.cfg.init_gaussian_multiple > 1:
            gaussian_param_num = init_gaussian_param_num // self.cfg.init_gaussian_multiple
        else:
            gaussian_param_num = init_gaussian_param_num

        # no pixel offset
        gaussian_param_num -= 2

        # update position
        gaussian_param_num += 3

        # SHs
        if self.cfg.sh_d != self.cfg.init_sh_d:
            gaussian_param_num += 3 * (self.cfg.sh_d - self.cfg.init_sh_d)

        # Get error channels
        if self.cfg.input_error:
            error_channels, error_feature_channels = self.define_error_channels()
        else:
            error_channels, error_feature_channels = 0, 0

        # Get gradient channels
        if self.cfg.input_gradient:
            gradient_channels = gaussian_param_num * self.cfg.init_gaussian_multiple
        else:
            gradient_channels = 0

        # final input channels
        input_signal_channels = gradient_channels + error_channels

        if self.cfg.refine_same_num_points:
            in_channels = (gaussian_param_num
                           + self.state_channels
                           + input_signal_channels)
        else:
            in_channels = (gaussian_param_num * self.cfg.init_gaussian_multiple
                           + self.state_channels
                           + input_signal_channels)

        if self.cfg.concat_init_state:
            in_channels += self.state_channels

        out_channels = gaussian_param_num
        if self.cfg.no_refine_mean:
            out_channels -= 3
        channels = self.state_channels
        if self.cfg.input_alpha:
            # pixel shuffle the alpha channel to the latent resolution
            in_channels += self.cfg.latent_downsample ** 2  # alpha
        if self.cfg.input_depth or self.cfg.input_depth_smooth_error:
            # pixel shuffle the depth channel to the latent resolution
            in_channels += self.cfg.latent_downsample ** 2  # depth
        return channels, in_channels, gaussian_param_num, out_channels, error_feature_channels

    def define_error_channels(self):
        if self.cfg.no_render_error:
            error_channels = 0
        else:
            if self.cfg.input_error_rgb_no_shuffle:
                error_channels = 3
            else:
                error_channels = 3 * self.cfg.latent_downsample ** 2

        if self.cfg.input_error_resnet_feature:
            # 3 scales: 1/2, 1/4, 1/8, channels: 64, 64, 128
            if self.cfg.input_error_resnet_feature_layers in (18, 34):
                error_feature_channels = 64 + 64 if self.cfg.input_error_shallow_resnet_feature else 64 + 64 + 128
            elif self.cfg.input_error_resnet_feature_layers == 50:
                error_feature_channels = 64 + 256 + 512
            else:
                raise NotImplementedError
            error_channels = error_feature_channels
        else:
            error_feature_channels = 256

        return error_channels, error_feature_channels

    def optimizer_preprocessing(self, optimizer_input: OptimizerInput, from_init: bool) -> None:
        if self.cfg.input_error_remain_context or self.cfg.input_error_merge_remain_context:
            assert self.cfg.input_error_cache_resnet_feature

        # Image dimensions
        context = optimizer_input.context
        b, v, _, h, w = context["image"].shape

        # Prepare Gaussians
        if from_init:
            # Scale initial opacities (in normal scale)
            # TODO Naama: add option to reset opacities and randomly reset/scale opacities of intermidiate updates
            opacities = optimizer_input.prev_output.gaussians.opacities  # post activation, in [0, 1]
            scaled_opacities = opacities * self.cfg.scale_initial_opacities  # default to 1.0
            optimizer_input.prev_output.gaussians.opacities = scaled_opacities

            # Process shs
            shs = optimizer_input.prev_output.gaussians.harmonics  # [B, N, 3, init_sh_d]
            init_sh_d = shs.shape[-1]
            if init_sh_d != self.cfg.sh_d:
                if init_sh_d > self.cfg.sh_d:
                    shs = shs[:, :, :, :self.cfg.sh_d]  # truncate  [B, N, 3, sh_d]
                else:
                    pad = self.cfg.sh_d - init_sh_d
                    shs = F.pad(shs, (0, pad), "constant", 0)
                optimizer_input.prev_output.gaussians.harmonics = shs

        # Right now, this does not do anything, since we do not use windows
        local_window_update, test_window_size, window_end, window_start = self.get_window_size(v)
        optimizer_input.additional_info = local_window_update, test_window_size, window_end, window_start
        self.update_gaussians_for_window(v, h, w, optimizer_input)

        # Prepare state
        # Gaussians dimensions
        n = optimizer_input.prev_output.gaussians.means.shape[1]
        vector_state = self.get_vector_state(b, v, n, optimizer_input, from_init)

        if from_init:
            # Set everything so that the optimizer isn't aware whether it's a new scene
            # Convert InitializerOutput to OptimizerPreviousOutput
            optimizer_input.prev_output = OptimizerPreviousOutput(gaussians=optimizer_input.prev_output.gaussians,
                                                                  state=OptimizerState())
        optimizer_input.prev_output.state.state = vector_state
        # init_state captures the scene-start state used by some experiments;
        # only set it on a fresh scene so replay-buffer resumes preserve the original value.
        if from_init:
            optimizer_input.prev_output.state.init_state = vector_state

    def update_gaussians_for_window(self, v, h, w, optimizer_input):
        # Get window parameters and set gaussians accordingly
        local_window_update, test_window_size, window_end, window_start = optimizer_input.additional_info

        if local_window_update and self.cfg.local_gaussian_render:
            init_gaussians = optimizer_input.prev_output.gaussians
            # select subset of gaussians
            init_gaussians_subset = select_gaussian_subset(init_gaussians, window_start, window_end,
                                                           v=v,
                                                           h=h // self.cfg.latent_downsample,
                                                           w=w // self.cfg.latent_downsample,
                                                           )
            optimizer_input.prev_output.gaussians = init_gaussians_subset

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
        res = self.apply_one_update_step(
            i, optimizer_input, optimizer_output
        )
        updated_gaussians: Gaussians = res[0]
        state: Tensor = res[1]
        meta_for_adc: dict = res[2]
        updates: dict[str, Tensor] = res[3]
        grads_raw: Tensor | None = res[4]
        normalized_grads: Tensor | None = res[5]
        scaled_state: Tensor | None = res[6]
        gaussians_sel: Tensor | None = res[7]

        # Timing
        self._record_iter_timing()

        # Log stats
        if grads_raw is not None:
            grads = grads_raw  # [B, G, D]
            nonzero_grads = (grads != 0).any(-1)  # [B, G]
            # Filter out strictly zero gradients for logging
            grads = grads[nonzero_grads].unsqueeze(0)  # [1, N_nonzero, D]
            assert nonzero_grads.shape[0] == 1
            self.nr_nonzero_grad_log.append(nonzero_grads[0].sum().item())

        # Local ADC
        # if optimizer_output.t == 500:
        #     weight_vis_contribution, _ = get_visibility_contribution_from_gaussian_obj(iter_context, updated_gaussians)  # [N]
        #     prune_mask = weight_vis_contribution < 5
        #     print(f"Pruning {torch.sum(prune_mask)} gaussians out of {prune_mask.shape[0]} at iteration {i}")
        #     updated_gaussians = updated_gaussians[:, ~prune_mask]
        #     state = state[~prune_mask]
        #     if self.cfg.normalize_update_input and self.cfg.normalize_update_input_type == "adam":
        #         if not self.update_input_norm.is_reset():
        #             self.update_input_norm.prune(prune_mask)

        # Densification and Pruning
        if self.cfg.any_adc:

            n_before_adc = updated_gaussians.means.shape[1]

            # Prepare objects to adjust during ADC
            object_dict = self.adc_object_dict_to_adjust
            object_dict["depthsplat_state"] = KnnBasedOptimizerState(state)
            object_dict["depthsplat_init_state"] = KnnBasedOptimizerState(optimizer_input.prev_output.state.init_state)

            # Apply ADC
            self.apply_adc(
                i=i, v=v, h=h, w=w, adc_state=optimizer_input.prev_output.state.adc_state,
                gaussians=updated_gaussians, meta=meta_for_adc, object_dict_to_adjust=object_dict
            )

            # Update state after ADC
            state = object_dict["depthsplat_state"].state
            optimizer_input.prev_output.state.init_state = object_dict["depthsplat_init_state"].state

            del object_dict["depthsplat_state"]
            if self.cfg.input_gradient_normalize and self.cfg.input_gradient_normalize_type == "adam":
                self.update_input_norm.aggregate_from_subgroups(object_dict, self.param_slices)

            # If N changed (add_new grew the population), stale KNN caches in the
            # point transformer modules would index out-of-bounds on the next forward
            # pass → CUDA illegal memory access. Reset them so they are recomputed.
            if updated_gaussians.means.shape[1] != n_before_adc:
                self._reset_knn_caches()

        # Save updated gaussians and state
        optimizer_input.prev_output.gaussians = updated_gaussians
        optimizer_input.prev_output.state.state = state

        if self.cfg.input_gradient_normalize_type == "adam":
            optimizer_input.prev_output.state.adam_state = self.update_input_norm.get_state()

        if self.training:
            optimizer_output.gaussian_list.append(updated_gaussians)

        # Info
        if not self.training and self.save_every(i + 1, tag="info"):
            # TODO Naama: review and refactor

            # save guassians
            optimizer_output.gaussian_list.append(updated_gaussians, detach_and_cpu=True, save_to_disk=False)

            # Save delta stats
            assert optimizer_output.info is not None

            # log updates

            # unpack shs
            shs = updates.pop("shs")  # [1, N, 3*sh_d]
            assert shs.shape[0] == 1, "Batch size > 1 not supported"
            shs = shs.squeeze(0)  # [N, 3*sh_d]
            shs = rearrange(shs, "n (c x) -> n c x", c=3, x=self.cfg.sh_d)  # [N, 3, sh_d]
            updates["sh0s"] = shs[..., 0:1]
            if self.cfg.sh_d > 1:
                updates["shNs"] = shs[..., 1:]
            else:
                updates["shNs"] = None

            # log deltas
            if "deltas" not in optimizer_output.info:
                optimizer_output.info["deltas"] = []
            optimizer_output.info["deltas"].append(
                {k: v.squeeze(0).cpu() if v is not None else None for k, v in updates.items()})

            # Split each vector grad into gaussians components
            if grads_raw is not None:
                if gaussians_sel is not None:
                    # Restore the zero gradients for tracking
                    b, g_valid, d = grads_raw.shape
                    g = state.shape[0]
                    grads_raw_full = torch.zeros((b, g, d))
                    normalized_grads_full = torch.zeros((b, g, d))
                    grads_raw_full[:, gaussians_sel, :] = grads_raw.cpu()
                    normalized_grads_full[:, gaussians_sel, :] = normalized_grads.cpu()
                    grads_raw = grads_raw_full
                    normalized_grads = normalized_grads_full

                grads_raw: dict[str, Tensor] = split_grads(grads_raw.cpu(), self.cfg)

                # Split each vector normalized_grads into gaussians components
                if normalized_grads is not None:
                    normalized_grads: dict[str, Tensor] = split_grads(normalized_grads.cpu(), self.cfg)

                    assert grads_raw["means"].shape == normalized_grads["means"].shape, \
                        f"Shape mismatch between grads and normalized_grads: {grads_raw['means'].shape} vs {normalized_grads['means'].shape}"

                # log states
                if scaled_state is not None:
                    if "states_norms" not in optimizer_output.info:
                        optimizer_output.info["states_norms"] = []
                    state_norm = torch.norm(scaled_state, dim=-1)  # [B, N]
                    optimizer_output.info["states_norms"].append(state_norm.cpu())

                # log gradients
                if "grads" not in optimizer_output.info:
                    optimizer_output.info["grads"] = []
                optimizer_output.info["grads"].append(grads_raw)

                # log normalized gradients
                if "normalized_grads" not in optimizer_output.info:
                    optimizer_output.info["normalized_grads"] = []
                optimizer_output.info["normalized_grads"].append(normalized_grads)

            # Check if output_path in kwargs
            output_path = kwargs.get("output_path", None)
            scene_name = kwargs.get("scene_name", None)

            if self.cfg.any_adc:
                pass
                # Plot stats
                # self.plot_info(i, output_path=output_path, scene_name=scene_name)

        # Post-update context + target renders
        self._save_post_update_renders(
            i, optimizer_input, optimizer_output, updated_gaussians,
            full_context, full_target,
        )

        # Optimizer output is being changed in place, but for clarity we return it
        return optimizer_output

    def apply_one_update_step(
            self,
            i,
            optimizer_input: OptimizerInput,
            optimizer_output: OptimizerOutput
    ) -> tuple[Gaussians, Tensor, dict, dict[str, Tensor], Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        # Unpacking
        context = optimizer_input.context
        target = optimizer_input.target
        renderer = optimizer_input.renderer
        debug_dict = optimizer_input.debug_dict
        num_refine = optimizer_input.num_refine
        gaussians = optimizer_input.prev_output.gaussians  # Gaussian object of [B, N, C]
        state = optimizer_input.prev_output.state.state  # [N, C]
        init_state = optimizer_input.prev_output.state.init_state  # [N, C]
        local_window_update, test_window_size, window_end, window_start = optimizer_input.additional_info
        # Get input signal for the optimizer model (erros/gradients)
        self.decoder_event_start.record()
        input_signal, gaussian_grads_raw, gaussian_grads, grad_sign, context_render_output, means2d_grads = (
            self.prepare_input_signal(context, i, gaussians, local_window_update, renderer, window_end,
                                      window_start, num_refine)
        )
        self.decoder_event_end.record()

        # Preparing meta for ADC
        if means2d_grads is not None:
            means2d_grads = means2d_grads.detach()  # [B, V, N, 2]
        meta_for_adc = {
            "visibility_filter": context_render_output.visibility_filter.detach(),  # [B, V, N]
            "radii": context_render_output.radii.detach(),  # [B, V, N, 1]
            "means_2d_grads": means2d_grads,  # [B, V, N, 2]
        }

        # Handle zero gradient gaussians
        # We either prune them, or exclude them from the input/output update
        if self.cfg.update_only_nonzero_grad and gaussian_grads is not None:
            gaussian_grads, gaussian_grads_raw, gaussians, grad_sign, init_state, input_signal, state = (
                self.handle_zero_grad_gaussians(
                    context,
                    context_render_output,
                    gaussian_grads,
                    gaussian_grads_raw,
                    gaussians,
                    grad_sign,
                    init_state,
                    input_signal,
                    means2d_grads,
                    meta_for_adc,
                    optimizer_input,
                    state)
            )

        # For training, if the number of active gaussians is too high, skip this batch
        # TODO Naama: maybe sampling?
        active_gaussians_num = state.shape[0]
        if self.training:
            if active_gaussians_num > 100_000:
                print(f"Skipping batch at iteration {i} with {active_gaussians_num} active gaussians.")
                raise SkipBatchException()
        if active_gaussians_num < self.cfg.refine_knn_samples:
            print(
                f"Skipping batch at iteration {i} with only {active_gaussians_num} active gaussians (need >= {self.cfg.refine_knn_samples}).")
            raise SkipBatchException()

        # Training only: save the rendering of initialization for logging
        # Will not be used for loss calculation
        # TODO Naama: this cause to many confusion. Pull it out of this function
        if self.training and i == 0:
            # Append context images initialization
            assert context_render_output is not None
            optimizer_output.context_render_list.append(context_render_output, detach_and_cpu=False)

            # render target images initialization
            target_render_output = renderer.forward_batch_subset(gaussians, target)
            optimizer_output.target_render_list.append(target_render_output, detach_and_cpu=False)

        # Unpack Gaussians
        means, scales, rotations_unnorm, opacities_raw, shs = unpack_gaussians(
            gaussians,
            scales_log=self.cfg.opt_scales_before_act,
            opacities_logit=True,
            opacities_unsqueeze=True,
            detach=True,  # stop gradient of last predictions
            scales_lims=(self.cfg.clamp_min_scale, self.cfg.clamp_refine_max_scale),
            raw_opacities_lims=(self.cfg.clamp_min_raw_opacities, self.cfg.clamp_max_raw_opacities)
        )

        gaussians_concat = pack_gaussians(means, scales, rotations_unnorm, opacities_raw, shs)  # [B, N, C]

        b, v, c, h, w = context["image"].shape
        latent_h = h // self.cfg.latent_downsample
        latent_w = w // self.cfg.latent_downsample
        # Debugging reprojection error
        if debug_dict is not None and (not self.training and self.save_every(i, tag="debug")):
            if "reprojection_error" in debug_dict:
                self.debug_reprojection_error(means, debug_dict, context, i, latent_h, latent_w)

        # prepare pt input
        point_cloud, tmp_batch_size = self.get_point_cloud(latent_h, latent_w, local_window_update, means,
                                                           test_window_size, v)
        # Create offset directly on device to avoid CPU-GPU transfer
        offset = torch.arange(1, b + 1, device=state.device, dtype=torch.long) * tmp_batch_size

        # reshape
        tmp_gaussian = self.reshape_gaussians_to_nc(latent_h, latent_w, gaussians_concat, v)  # [B, N, C] --> [BN, C]
        # add global attention to exchange info across views
        if self.cfg.input_error_mv_attn:
            input_signal = self.apply_global_attn(b, h, input_signal, latent_h,
                                                  latent_w, local_window_update, test_window_size, v, w)

        tmp_input_signal = input_signal.reshape(-1,
                                                input_signal.shape[-1])  # [B, N, C] --> [BN, C] - faster than rearrange
        tmp_input_signal = self.append_to_input_signal(b, context, context_render_output, tmp_input_signal, v)

        # Normalize state before input it to the update module
        if self.cfg.input_normalize_state:
            state_norm = state.norm(dim=1, keepdim=True) / math.sqrt(state.shape[-1])  # [BG, 1]
            state = state / (state_norm + 1e-8)  # [BG, C]

        normalized_input_signal = self.update_input_norm(tmp_input_signal)

        if self.cfg.input_normalize_gaussians:
            tmp_gaussian_mean = tmp_gaussian.mean()
            tmp_gaussian_std = tmp_gaussian.std()
            tmp_gaussian = (tmp_gaussian - tmp_gaussian_mean) / (tmp_gaussian_std + 1e-8)

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
            point_cloud, tmp_gaussian, state, update_input = self.prepare_update_input(b, i, init_state,
                                                                                       normalized_input_signal,
                                                                                       latent_h,
                                                                                       latent_w,
                                                                                       local_window_update,
                                                                                       point_cloud,
                                                                                       tmp_gaussian,
                                                                                       # gradients/errors + additional pixel related quantities
                                                                                       state, v, window_end,
                                                                                       window_start)

            # if self.cfg.refine_with_mv_attn:
            #     state = concat
            #     for i in range(len(self.update_module)):
            #         print(i, len(self.update_module), self.update_module[i])
            #         state = self.update_module[i]([point_cloud, state, offset])  # [N, C]
            # else:
            updated_state = self.apply_update_module(b, latent_h, latent_w, offset,
                                                     point_cloud, update_input, v, state, i)

            # Hard coded extract normalized gradients
            if self.cfg.input_gradient and self.cfg.input_gradient_normalize:
                normalized_grads = normalized_input_signal
            else:
                normalized_grads = None

            # Recover the state norm
            if self.cfg.input_normalize_state:
                # state = state * state_std + state_mean
                updated_state = updated_state * state_norm

            # Predict a scale for the updtaed scale for the MLP deltas prediction
            # The updated state for the next stage remains the same
            if self.cfg.predict_state_scale:
                state_scale = self.state_scale_head(update_input.detach())
                if self.cfg.predict_state_scale_norm:
                    # Normalize the state vector
                    state_scale = state_scale / (state_scale.norm(p=2, dim=1, keepdim=True) + 1e-8)
            else:
                state_scale = torch.tensor([1], device=state.device, dtype=state.dtype)
            updated_state_scaled = state_scale * updated_state

            # optionally append time encodiing to normalize input
            with TimeEncodingWrapper(self.cfg.use_time_encoding,
                                     self.time_encoder_fn,
                                     optimizer_output.t,
                                     self.cfg.time_encoding_max_steps,
                                     updated_state_scaled) as embedded_state:
                if self.cfg.use_time_encoding:
                    assert not self.cfg.concat_init_state
                    assert not self.cfg.replace_init_state

                # delta gaussian head
                delta_gaussians = self.apply_delta_gaussian_head(b, context, init_state, embedded_state, v)

        visibility_scale = None  # disable for now

        delta_means, delta_opacities, delta_rotations, delta_scales, delta_shs, init_repeat, delta_gaussians = (
            self.postprocess_deltas(b, delta_gaussians, gaussian_grads, gaussians_concat, grad_sign, latent_h, latent_w,
                                    local_window_update, normalized_grads, state, test_window_size, v, window_end,
                                    window_start, optimizer_output.t, optimizer_output.T, visibility_scale)
        )

        means, opacities_raw, rotations_unnorm, scales, shs = self.repeat_gaussians(means, opacities_raw,
                                                                                    rotations_unnorm, scales, shs)

        covariances, means, scales, rotations, rotations_unnorm, opacities_raw, shs = self.update_gaussians_params(
            delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs,
            means, scales, rotations_unnorm, opacities_raw, shs, init_repeat)

        # Recover the state in non valid gaussians (and grad for logging)
        if gaussians.sel is not None:
            sel = gaussians.sel  # [B, G]
            full_state = optimizer_input.prev_output.state.state

            # Convert full state to the dtype of state
            full_state = full_state.to(state.dtype)
            # Use non-in-place index_put to avoid in-place modification of tensors
            # in the autograd computation graph (fixes version mismatch errors with stability loss)
            updated_state = full_state.index_put((sel,), updated_state)
        else:
            sel = None

        # update gaussians (only where mask is True)
        # Use view instead of rearrange for speed
        shs_reshaped = shs.view(shs.shape[0], shs.shape[1], 3, -1)
        gaussians = gaussians.update_object_by_curr_mask(
            means=means,
            covariances=covariances,
            harmonics=shs_reshaped,
            opacities=opacities_raw.squeeze(-1).sigmoid(),
            scales=scales,
            rotations=rotations,
            rotations_unnorm=rotations_unnorm,
            sel=None,
            deltas=delta_gaussians if self.training else None,
            gradients=gaussian_grads_raw if self.training else None,
            norm_gradients=normalized_grads.unsqueeze(0) if normalized_grads is not None and self.training else None
        )

        updates = {
            "means": delta_means.detach(),
            "scales": delta_scales.detach(),
            "rotations": delta_rotations.detach(),
            "opacities": delta_opacities.detach(),
            "shs": delta_shs.detach()
        }

        grads_raw = gaussian_grads.detach() if gaussian_grads is not None else None
        grads_adam = normalized_grads.detach() if normalized_grads is not None else None

        return gaussians, updated_state, meta_for_adc, updates, grads_raw, grads_adam, updated_state_scaled, sel

    def postprocess_deltas(self, b, delta_gaussians, gaussian_grads, gaussians_concat, grad_sign, latent_h, latent_w,
                           local_window_update, normalized_grads, state, test_window_size, v, window_end, window_start,
                           t, T, visibility_scale):
        # Updates for gradient input (scale, log scale, )
        delta_gaussians_raw = delta_gaussians
        delta_gaussians = self.update_delta_for_gradients_input(delta_gaussians_raw, grad_sign, normalized_grads,
                                                                visibility_scale)

        # Rearrange back to [B, N, C]
        delta_gaussians, delta_gaussians_raw = self.rearrange_delta_gaussians(b, delta_gaussians,
                                                                              delta_gaussians_raw, latent_h,
                                                                              latent_w, local_window_update,
                                                                              gaussians_concat,
                                                                              test_window_size, v, window_end,
                                                                              window_start)

        # TODO Naama: shouldn't it be before rearranging?
        # multiple gaussian heads to predict multiple gaussians
        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
            if self.cfg.gaussian_head_multiple > 1:
                num_additional_heads = self.cfg.gaussian_head_multiple - 1
                delta_gaussian_list = [delta_gaussians]  # list of [B, N, C]
                for i in range(num_additional_heads):
                    curr_delta = self.update_head_list[i](state)
                    curr_delta = rearrange(curr_delta, "(b n) c -> b n c", b=b)
                    delta_gaussian_list.append(curr_delta)
                delta_gaussians = torch.cat(delta_gaussian_list, dim=1)  # [B, K*N, C]

        # Experimental overide deltas
        if self.cfg.experimental_run:
            self.experimental_update_deltas(delta_gaussians, gaussian_grads, normalized_grads)

        # Split
        delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs, init_repeat = (
            self.split_delta_gaussians(delta_gaussians)
        )

        # Apply lr
        delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs = self.scale_deltas_with_lr(
            t, delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs
        )

        # Linear combination with adam normalized gradients
        if self.cfg.delta_adam_combine_step > 0 and normalized_grads is not None:
            assert t <= T
            if t > self.cfg.delta_adam_combine_step:
                alpha = 0.0
                beta = 1 - ((t - self.cfg.delta_adam_combine_step) / (T - self.cfg.delta_adam_combine_step)) ** alpha
                # Linear combination with adam normalized gradients
                # Use the inverse of the normalized gradients
                # TODO Naama: hard coded lr
                # means
                delta_means = beta * delta_means + (1 - beta) * -normalized_grads[
                    ..., self.param_slices["means"]] * 1.6e-4
                # scales
                delta_scales = beta * delta_scales + (1 - beta) * -normalized_grads[
                    ..., self.param_slices["scales"]] * 5e-3
                # rotations
                delta_rotations = beta * delta_rotations + (1 - beta) * -normalized_grads[
                    ..., self.param_slices["quats"]] * 1e-3
                # opacities
                delta_opacities = beta * delta_opacities + (1 - beta) * -normalized_grads[
                    ..., self.param_slices["opacities"]] * 5e-2
                # sh0 - use view instead of rearrange for speed
                delta_shs_bgdc = delta_shs.view(delta_shs.shape[0], delta_shs.shape[1], 3, -1)  # [b, g, 3, c]
                delta_sh0 = delta_shs_bgdc[..., 0]  # [b, g, 3]
                delta_shN = delta_shs_bgdc[..., 1:]  # [b, g, 3, d-1]
                delta_shN = delta_shN.flatten(-2)  # [b, g, 3*(d-1)] - faster than rearrange

                new_delta_sh0 = beta * delta_sh0 + (1 - beta) * -normalized_grads[
                    ..., self.param_slices["sh0"]] * 2.5e-3
                new_delta_shN = beta * delta_shN + (1 - beta) * -normalized_grads[
                    ..., self.param_slices["shN"]] * 1.25e-4
                new_delta_shN = new_delta_shN.view(new_delta_shN.shape[0], new_delta_shN.shape[1], 3,
                                                   -1)  # [b, g, 3, d-1]
                delta_shs[..., ::self.cfg.sh_d] = new_delta_sh0
                # shN
                for i in range(1, self.cfg.sh_d):
                    delta_shs[..., i::self.cfg.sh_d] = new_delta_shN[..., i - 1]

        return delta_means, delta_opacities, delta_rotations, delta_scales, delta_shs, init_repeat, delta_gaussians

    def handle_zero_grad_gaussians(self, context, context_render_output, gaussian_grads, gaussian_grads_raw, gaussians,
                                   grad_sign, init_state, input_signal, means2d_grads, meta_for_adc, optimizer_input,
                                   state):
        # Compute a mask for gaussian that did not contribute to any pixel of context views
        # Their gradients are strictly zero.
        # We don't want to prune them, as they might be relevant in other views (in dense views).
        if self.cfg.prune_invisible_gaussians:
            gaussian_grads, gaussians, grad_sign, input_signal, state = self.prune_invisible_gaussians(context,
                                                                                                       context_render_output,
                                                                                                       gaussian_grads,
                                                                                                       gaussian_grads_raw,
                                                                                                       gaussians,
                                                                                                       grad_sign,
                                                                                                       input_signal,
                                                                                                       means2d_grads,
                                                                                                       meta_for_adc,
                                                                                                       optimizer_input,
                                                                                                       state)
        else:
            assert not self.cfg.local_prune_zero_radii
            assert not self.cfg.local_prune_low_weights
            assert gaussian_grads.shape[0] == 1, "Batch size > 1 not supported with mask"

            # radii_mask = (context_render_output.radii != 0).all(1).all(-1)  # [B, G]
            # valid_mask = valid_mask & radii_mask  # only consider gaussians with non-zero radius as valid

            # radii = context_render_output.radii  # [B, V, G, 2]
            #
            # # XOR on radii last dimension to find gaussians that have zero radius in only one dimension
            # assert ((radii[..., 0] == 0) ^ (radii[..., 1] == 0)).sum() == 0  # [B, V, G]
            #
            # # Check that all zero radius gaussians are in the zero gradient mask (but not necessarily the opposite)
            # zero_radius_mask = (radii == 0).any(1).any(-1)  # [B, G]
            # zero_grad_mask = ~valid_mask  # [B, G]
            # zero_radius_cnt = zero_radius_mask.sum()
            # zero_grad_of_zero_radii_cnt = zero_grad_mask[zero_radius_mask].sum()
            # assert zero_grad_of_zero_radii_cnt == zero_radius_cnt, (f"All zero radius gaussians should have zero "
            #                                                         f"gradients. Found {zero_radius_cnt} zero radius gaussians, but only {zero_grad_of_zero_radii_cnt} of "
            #                                                         f"them have zero gradients.")
            # print(f"Found {zero_grad_of_zero_radii_cnt} / {zero_radius_cnt} zero radius gaussians with zero gradients.")

            # Contribution of zero gradient gaussians
            # gaussian_grads_zero_radii = gaussian_grads[zero_radius_mask]  # [G_zero_radius, D]
            # assert gaussian_grads_zero_radii.abs().sum() == 0, "Gaussians with zero radius should have zero gradients."

            # radii of zero gradient gaussians
            # radii_zero_grad = radii[:, :, zero_grad_mask[0]]  # [G_zero_grad, V, 2]
            # zero_grad_radii_cont = radii_zero_grad.sum()

            # Compute [G] mask without materializing [B,G,D] bool
            # any() on floats treats nonzero as True
            valid_g = gaussian_grads[0].any(dim=-1)  # [G] bool
            sel = None

            # if everything is valid, skip all slicing work
            if not valid_g.all():
                sel = valid_g.nonzero(as_tuple=True)[0]  # [G_valid]

                input_signal = input_signal[:, sel, :]  # [B, G_valid, C]

                gaussian_grads = gaussian_grads[:, sel, :]  # [B, G_valid, D]
                if gaussian_grads_raw is not None:
                    gaussian_grads_raw = gaussian_grads_raw[:, sel, :]
                if grad_sign is not None:
                    grad_sign = grad_sign[:, sel, :]

                state = state[sel, :]  # [G_valid, C]
                init_state = init_state[sel, :]  # [G_valid, C]

                valid_mask = valid_g.unsqueeze(0)  # [1, G]
            gaussians.sel = sel

            if self.cfg.input_gradient_normalize_type == "adam":
                self.update_input_norm.sel = sel
        return gaussian_grads, gaussian_grads_raw, gaussians, grad_sign, init_state, input_signal, state

    def prune_invisible_gaussians(self, context, context_render_output, gaussian_grads, gaussian_grads_raw, gaussians,
                                  grad_sign, input_signal, means2d_grads, meta_for_adc, optimizer_input, state):
        # Get visible gaussians mask, based on the last rendering
        with torch.no_grad():
            visible_mask = self.get_visible_gaussian_mask(gaussian_grads, gaussians,
                                                          context_render_output.visibility_filter, context)  # [B, N, 1]
            if visible_mask is None:
                return gaussian_grads, gaussians, grad_sign, input_signal, state
        assert visible_mask.shape[0] == 1
        visible_mask = visible_mask[0, :, 0]  # [N], squeeze batch and last dim
        # Apply mask
        gaussians = gaussians[:, visible_mask]
        state = state[visible_mask]
        input_signal = input_signal[:, visible_mask]  # [B, N, C]
        if gaussian_grads is not None:
            gaussian_grads = gaussian_grads[:, visible_mask]  # [B, N, C]
        if gaussian_grads_raw is not None:
            gaussian_grads_raw = gaussian_grads_raw[:, visible_mask]  # [B, N, C]
        if grad_sign is not None:
            grad_sign = grad_sign[:, visible_mask]  # [B, N, C]
        meta_for_adc["visibility_filter"] = context_render_output.visibility_filter[:, :, visible_mask]
        meta_for_adc["radii"] = context_render_output.radii[:, :, visible_mask]
        if means2d_grads is not None:
            meta_for_adc["means_2d_grads"] = means2d_grads[:, :, visible_mask]
        if self.cfg.input_gradient_normalize and self.cfg.input_gradient_normalize_type == "adam":
            if not self.update_input_norm.is_reset():
                prune_mask = ~visible_mask
                self.update_input_norm.prune(prune_mask)  # the prune fn will invert the mask again
        if self.cfg.any_adc:
            optimizer_input.prev_output.state.adc_state.external_pruning(visible_mask)
        return gaussian_grads, gaussians, grad_sign, input_signal, state

    def deactivate_updates(self, subset, gaussians, radii_vis_mask, deltas, gaussian_grads):
        """ Deactivate updates for gaussians that are not visible in any view """
        visible_mask = self.get_visible_gaussian_mask(gaussian_grads, gaussians, radii_vis_mask, subset)
        deltas = deltas * visible_mask  # [B, N, C]
        return deltas

    def get_visible_gaussian_mask(self, gaussian_grads, gaussians, radii_vis_mask, subset):
        """
        Get mask for gaussians that are visible in at least one view.

        We calculate two criteria:
        1. Whether the projected 2d radius is visible in at least one view.
        2. Whether the gaussian has a non-zero weight contribution to the rendering.

        If neither pruning criterion is enabled, returns None.

        Args:
            gaussian_grads: [B, N, C] or None
            gaussians: Gaussians object
            radii_vis_mask: [B, V, N], bool
            subset: dict, context or target
        """
        # If no pruning criteria are active, return None
        if not (self.cfg.local_prune_zero_radii or self.cfg.local_prune_low_weights):
            return None

        b, v, n = radii_vis_mask.shape

        # Criterion 1: Projected radius visibility
        if self.cfg.local_prune_zero_radii:
            radii_vis_mask = radii_vis_mask.any(dim=1).unsqueeze(-1)  # [B, N, 1]
        else:
            radii_vis_mask = torch.ones((b, n, 1), dtype=torch.bool, device=radii_vis_mask.device)

        # Criterion 2: Weight contribution visibility
        if self.cfg.local_prune_low_weights:
            threshold = self.cfg.local_prune_low_weights_thresh
            weight_vis_contribution, _ = get_visibility_contribution_from_gaussian_obj(subset, gaussians)  # [N]
            weight_cont_mask = (weight_vis_contribution > threshold).view(1, -1, 1)
        else:
            weight_cont_mask = torch.ones((b, n, 1), dtype=torch.bool, device=radii_vis_mask.device)

        visible_mask = radii_vis_mask & weight_cont_mask  # [B, N, 1]
        return visible_mask

    def experimental_inplace_update_delta(self, deltas, grads, normalized_grads, cfg_attr):
        # Slicing of the gradients vector per parameter
        param_num = grads.shape[-1]
        assert param_num == 11 + self.cfg.sh_d * 3
        param_slices = self.param_slices

        update = getattr(self.cfg.experimental_update, cfg_attr)
        if update:
            # Update this parameter
            use_norm_grad = getattr(self.cfg.experimental_use_norm_grads, cfg_attr)
            use_grad = self.cfg.experimental_use_grads and not use_norm_grad
            use_resplat = not use_grad and not use_norm_grad
            assert not (use_grad and use_norm_grad)
            if use_grad:
                # Use the inverse of the gradients
                # TODO Naama: hard coded learning rate for SGD
                deltas[..., param_slices[cfg_attr]] = -(grads[..., param_slices[cfg_attr]]).to(deltas.dtype) * 30
            elif use_norm_grad:
                # Use the inverse of the normalized gradients
                updated_delta = -normalized_grads[..., param_slices[cfg_attr]] * getattr(self.cfg.experimental_lr,
                                                                                         cfg_attr)
                deltas[..., param_slices[cfg_attr]] = updated_delta.to(deltas.dtype)
            else:
                # Use the network prediction (already negated before)
                pass
        else:
            # Do not update this parameter
            deltas[..., param_slices[cfg_attr]] = 0

    def experimental_update_deltas(self, deltas, grads, normalized_grads):
        # Verify that at least one parameter is actually using norm_grads or grads override
        any_norm_grad = any(
            getattr(self.cfg.experimental_use_norm_grads, p) for p in self.cfg.experimental_update.param_names)
        any_grad = self.cfg.experimental_use_grads
        any_override = any_norm_grad or any_grad
        assert any_override, (
            "experimental_run=true but no parameter has use_norm_grads or use_grads enabled. "
            "Check that experimental_use_norm_grads._base=true (it gates all other fields via property)."
        )
        if any_norm_grad:
            assert normalized_grads is not None, (
                "experimental_use_norm_grads is enabled but normalized_grads is None. "
                "Ensure input_gradient=true and input_gradient_normalize=true."
            )

        for p in self.cfg.experimental_update.param_names:
            self.experimental_inplace_update_delta(deltas, grads, normalized_grads, p)

    def scale_deltas_with_lr(self, t, delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs):
        # Scale deltas with learning rates
        delta_means = delta_means * self.scheduler.get_lr(t, "means")
        delta_scales = delta_scales * self.scheduler.get_lr(t, "scales")
        if delta_rotations is not None:
            delta_rotations = delta_rotations * self.scheduler.get_lr(t, "rotations")
        delta_opacities = delta_opacities * self.scheduler.get_lr(t, "opacities")

        # Use view instead of rearrange for speed
        delta_shs = delta_shs.view(delta_shs.shape[0], delta_shs.shape[1], 3, -1)  # [b, g, 3, c]
        delta_sh0 = delta_shs[..., 0]  # [B, N, C]
        delta_shN = delta_shs[..., 1:]
        delta_sh0 = delta_sh0 * self.scheduler.get_lr(t, "sh0")
        delta_shN = delta_shN * self.scheduler.get_lr(t, "shN")
        delta_shs = torch.cat((delta_sh0.unsqueeze(-1), delta_shN), dim=-1)
        delta_shs = delta_shs.flatten(-2)  # [b, g, d*c] - faster than rearrange
        return delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs

    def append_to_input_signal(self, b, context, context_render, tmp_input_signal, v):
        if self.cfg.input_alpha:
            render_alpha = rearrange(context_render.accumulated_alpha, "b v h w -> (b v) () h w")
            render_alpha = F.pixel_unshuffle(render_alpha, downscale_factor=self.cfg.latent_downsample)
            render_alpha = rearrange(render_alpha, "(b v) c h w -> (b v h w) c", b=b, v=v)
            tmp_input_signal = torch.cat((tmp_input_signal, render_alpha), dim=-1)
        if self.cfg.input_depth:
            render_depth = rearrange(context_render.depth, "b v h w -> (b v) () h w")
            render_depth = F.pixel_unshuffle(render_depth, downscale_factor=self.cfg.latent_downsample)
            render_depth = rearrange(render_depth, "(b v) c h w -> (b v h w) c", b=b, v=v)
            tmp_input_signal = torch.cat((tmp_input_signal, render_depth), dim=-1)
        if self.cfg.input_depth_smooth_error:
            disp = 1. / context_render.depth.clamp(min=1e-3, max=1000.)  # [B, V, H, W]
            disp = rearrange(disp, "b v h w -> (b v) () h w")

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            tmp_imgs = rearrange(context["image"], "b v c h w -> (b v) c h w")

            depth_smooth_error = get_smooth_loss(norm_disp, tmp_imgs, no_mean=True)

            depth_smooth_error = F.pixel_unshuffle(depth_smooth_error, downscale_factor=self.cfg.latent_downsample)
            depth_smooth_error = rearrange(depth_smooth_error, "(b v) c h w -> (b v h w) c", b=b, v=v)
            tmp_input_signal = torch.cat((tmp_input_signal, depth_smooth_error), dim=-1)
        return tmp_input_signal

    def repeat_gaussians(self, prev_means, prev_opacities_raw, prev_rotations_unnorm, prev_scales, prev_shs):
        if self.cfg.gaussian_head_multiple > 1:
            # predict multiple gaussians for each point
            prev_means = prev_means.repeat(1, self.cfg.gaussian_head_multiple, 1)
            prev_scales = prev_scales.repeat(1, self.cfg.gaussian_head_multiple, 1)
            prev_rotations_unnorm = prev_rotations_unnorm.repeat(1, self.cfg.gaussian_head_multiple, 1)
            prev_opacities_raw = prev_opacities_raw.repeat(1, self.cfg.gaussian_head_multiple,
                                                           1) / self.cfg.gaussian_head_multiple  # smaller opacities, important
            prev_shs = prev_shs.repeat(1, self.cfg.gaussian_head_multiple, 1)
        # NOTE: only repeat at the first iteration
        refine_repeat = self.cfg.refine_gaussian_multiple
        if refine_repeat > 1:
            # predict multiple gaussians for each point
            prev_means = prev_means.repeat(1, refine_repeat, 1)
            prev_scales = prev_scales.repeat(1, refine_repeat, 1)
            prev_rotations_unnorm = prev_rotations_unnorm.repeat(1, refine_repeat, 1)
            prev_opacities_raw = prev_opacities_raw.repeat(1, refine_repeat, 1)  # smaller opacities, important
            prev_shs = prev_shs.repeat(1, refine_repeat, 1)
        return prev_means, prev_opacities_raw, prev_rotations_unnorm, prev_scales, prev_shs

    def split_delta_gaussians(self, delta_gaussians):
        delta_rotations = None

        if self.cfg.init_gaussian_multiple > 1 and not self.cfg.refine_same_num_points:
            init_repeat = self.cfg.init_gaussian_multiple
        else:
            init_repeat = 1
        p = get_gaussian_param_sizes(self.cfg.sh_d)
        if self.cfg.refine_sh_only:
            delta_shs = delta_gaussians
            delta_means = delta_scales = delta_opacities = 0.
        elif self.cfg.no_refine_rotation:
            delta_means, delta_scales, delta_opacities, delta_shs = delta_gaussians.split(
                (p["means"] * init_repeat, p["scales"] * init_repeat, p["opacities"] * init_repeat,
                 p["shs"] * init_repeat), dim=-1
            )
        elif self.cfg.no_refine_mean:
            delta_scales, delta_rotations, delta_opacities, delta_shs = delta_gaussians.split(
                (p["scales"] * init_repeat, p["quats"] * init_repeat, p["opacities"] * init_repeat,
                 p["shs"] * init_repeat), dim=-1
            )
            delta_means = torch.zeros_like(delta_scales)
        else:
            delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs = delta_gaussians.split(
                (p["means"] * init_repeat, p["scales"] * init_repeat, p["quats"] * init_repeat,
                 p["opacities"] * init_repeat, p["shs"] * init_repeat), dim=-1
            )
        if (
                self.cfg.refine_gaussian_multiple > 1 or self.cfg.init_gaussian_multiple > 1) and not self.cfg.refine_same_num_points:
            delta_means = rearrange(delta_means, "b n (c k) -> b (n k) c", k=init_repeat)
            delta_scales = rearrange(delta_scales, "b n (c k) -> b (n k) c", k=init_repeat)
            delta_rotations = rearrange(delta_rotations, "b n (c k) -> b (n k) c", k=init_repeat)
            delta_opacities = rearrange(delta_opacities, "b n (c k) -> b (n k) c", k=init_repeat)
            delta_shs = rearrange(delta_shs, "b n (c k) -> b (n k) c", k=init_repeat)
        return delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs, init_repeat

    def rearrange_delta_gaussians(self, b, delta_gaussians, delta_gaussians_raw, latent_h, latent_w,
                                  local_window_update, prev_gaussians_concat, test_window_size, v, window_end,
                                  window_start):
        # [BV, C]
        # update gaussian parameters
        delta_gaussians = rearrange(delta_gaussians, "(b n) c -> b n c", b=b)
        delta_gaussians_raw = rearrange(delta_gaussians_raw, "(b n) c -> b n c", b=b)
        if local_window_update and not self.cfg.local_gaussian_render:
            # zero padding for non-updated gaussians
            # curr_v = self.cfg.update_window_size if self.training else test_window_size
            curr_v = test_window_size
            tmp_delta = rearrange(delta_gaussians, "b (v h w) c -> b v h w c", b=b, v=curr_v, h=latent_h,
                                  w=latent_w)

            all_delta = []
            # padding
            if window_start > 0:
                tmp_size = rearrange(prev_gaussians_concat, "b (v h w) c -> b v h w c", b=b, v=v, h=latent_h,
                                     w=latent_w)
                pad_left = torch.zeros_like(tmp_size[:, :window_start, :, :, :], requires_grad=False)
                all_delta.append(pad_left)

            all_delta.append(tmp_delta)

            if window_end < v:
                tmp_size = rearrange(prev_gaussians_concat, "b (v h w) c -> b v h w c", b=b, v=v, h=latent_h,
                                     w=latent_w)
                pad_right = torch.zeros_like(tmp_size[:, window_end:, :, :, :], requires_grad=False)
                all_delta.append(pad_right)

            tmp_delta = torch.cat(all_delta, dim=1)  # [B, V, H, W, C]
            delta_gaussians = rearrange(tmp_delta, "b v h w c -> b (v h w) c")  # [B, N, C]
        return delta_gaussians, delta_gaussians_raw

    def update_gaussians_params(self, delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs,
                                means, scales, rotations_unnorm, opacities_raw, shs,
                                repeat):
        means = self.update_means(delta_means, means)

        # clamp the scale
        scales = self.update_scales(delta_scales, scales, repeat)
        if self.cfg.opt_scales_before_act:
            scales = scales.exp()

        if not self.cfg.no_refine_rotation:
            rotations, rotations_unnorm = self.update_rotations(delta_rotations, rotations_unnorm)
        else:
            rotations = F.normalize(rotations_unnorm, dim=-1)

        # compute covariance
        covariances = build_covariance(scales, rotations)  # ([1, VHW, 3, 3])

        opacities_raw = self.update_opacities(delta_opacities, opacities_raw, repeat)
        shs = self.update_shs(delta_shs, shs)
        return covariances, means, scales, rotations, rotations_unnorm, opacities_raw, shs

    def update_shs(self, delta_shs, prev_shs):
        shs = prev_shs + delta_shs  # [B, N, 3*sh_d]

        if self.cfg.clamp_shs_soft:
            assert self.cfg.clamp_min_shs == -self.cfg.clamp_max_shs, "For soft clamp, min and max should be symmetric around 0"
            shs = torch.tanh(shs / self.cfg.clamp_max_shs) * self.cfg.clamp_max_shs
        else:
            shs = shs.clamp(min=self.cfg.clamp_min_shs, max=self.cfg.clamp_max_shs)

        return shs

    def update_opacities(self, delta_opacities, prev_opacities_raw, repeat):
        # update init opacities when predicting multiple gaussians
        if repeat > 1 and not self.cfg.multi_gaussian_scale_smaller and (self.cfg.init_gaussian_multiple == 1):
            # Given y = sigmoid(x), to get new x' such that sigmoid(x') = y / K:
            # The formula is: x' = x + log((1 - y) / (K - y))
            # This adjusts x so that the sigmoid output is scaled down by a factor of K
            tmp_sigmoid = prev_opacities_raw.sigmoid()
            prev_opacities_raw = prev_opacities_raw + torch.log(
                (1 - tmp_sigmoid) / (repeat - tmp_sigmoid)) + delta_opacities
        else:
            prev_opacities_raw = prev_opacities_raw + delta_opacities
            # prev_opacities_raw = prev_opacities_raw.clamp(min=-5, max=5)
        return prev_opacities_raw

    @staticmethod
    def update_rotations(delta_rotations, prev_rotations_unnorm):
        assert delta_rotations is not None
        prev_rotations_unnorm = prev_rotations_unnorm + delta_rotations
        # normazlie
        prev_rotations = prev_rotations_unnorm / (prev_rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8)
        return prev_rotations, prev_rotations_unnorm

    def update_scales(self, delta_scales, prev_scales, repeat):
        if repeat > 1 and self.cfg.multi_gaussian_scale_smaller:
            # smaller initial scales
            new_scales = (prev_scales / repeat + delta_scales).clamp(min=self.cfg.gaussian_adapter.clamp_min_scale)
        else:
            new_scales = (prev_scales + delta_scales)

        if self.cfg.opt_scales_before_act:
            min_scale = self.cfg.clamp_min_raw_scales
            max_scale = self.cfg.clamp_max_raw_scales
        else:
            min_scale = self.cfg.clamp_min_scale
            max_scale = self.cfg.clamp_refine_max_scale

        new_scales = new_scales.clamp(min=min_scale)
        new_scales = new_scales.clamp(max=max_scale)

        return new_scales

    @staticmethod
    def update_means(delta_means, prev_means):
        prev_means = (prev_means + delta_means)
        return prev_means

    def _on_scene_start_impl(self, optimizer_input: OptimizerInput) -> None:
        # Reset the state
        if isinstance(optimizer_input.prev_output, InitializerOutput):  # New scene
            from_init = True
            # Reset the optimizer state for a new scene
            # We cannot just use super().on_scene_start() because we need to process the InitializerOutput in case it
            # contain conditioning features
            self.reset_logs()

            if self.cfg.input_gradient_normalize_type == "adam":
                self.update_input_norm.reset()
                nr_gaussians = rearrange(optimizer_input.prev_output.gaussians.means, "b n c -> (b n) c").shape[0]
                param_num = self.gaussian_param_num
                self.update_input_norm.initialize(shape=(nr_gaussians, param_num),
                                                  device=optimizer_input.prev_output.gaussians.means.device)

            # make sure xyz are contiguous
            optimizer_input.prev_output.gaussians.means = optimizer_input.prev_output.gaussians.means.contiguous()
        elif isinstance(optimizer_input.prev_output, OptimizerPreviousOutput):
            from_init = False
            if self.cfg.input_gradient_normalize_type == "adam":
                # Continuing previous optimization from replay buffer
                self.update_input_norm.update_state(optimizer_input.prev_output.state.adam_state)

            # TODO Naama: logs are not handled right now for continuing from replay buffer
            self.reset_logs()
        else:
            raise ValueError(f"Unknown prev_output type {type(optimizer_input.prev_output)}")

        # Preparing the input for a new scene (will  handle both new scene and continuing from replay buffer)
        # Will convert init_output to prev_output internally
        self.optimizer_preprocessing(optimizer_input, from_init=from_init)

        # initialize adc state, after converting to prev_output
        if from_init and self.cfg.any_adc:
            self.initialize_adc_state(self.cfg, optimizer_input)

    def reshape_gaussians_to_nc(self, latent_h, latent_w, prev_gaussians_concat, v):
        if self.cfg.init_gaussian_multiple == 4 and not self.cfg.refine_same_num_points:
            # gaussians are with more points, reshape
            tmp_gaussian = rearrange(prev_gaussians_concat, "b (v h x w y) c -> (b v h w) (c x y)",
                                     v=v, h=latent_h, w=latent_w, x=2, y=2)
        elif self.cfg.init_gaussian_multiple == 16 and not self.cfg.refine_same_num_points:
            tmp_gaussian = rearrange(prev_gaussians_concat, "b (v h x w y) c -> (b v h w) (c x y)",
                                     v=v, h=latent_h, w=latent_w, x=4, y=4)
        else:
            tmp_gaussian = rearrange(prev_gaussians_concat, "b n c -> (b n) c")
        return tmp_gaussian

    def get_point_cloud(self, latent_h, latent_w, local_window_update, prev_means, test_window_size, v):
        # TODO: when the initial model predicts multiple gaussians, the number of points also increases
        if self.cfg.init_gaussian_multiple == 4 and not self.cfg.refine_same_num_points:
            point_cloud = rearrange(prev_means, "b (v h w) c -> b v h w c",
                                    v=v, h=latent_h * 2, w=latent_w * 2,
                                    )
            tmp_batch_size = v * latent_h * latent_w
            # simply use uniform grid subsample of point cloud to reduce points
            point_cloud = point_cloud[:, :, ::2, ::2]
            point_cloud = rearrange(point_cloud, "b v h w c -> (b v h w) c")
        elif self.cfg.init_gaussian_multiple == 16 and not self.cfg.refine_same_num_points:
            point_cloud = rearrange(prev_means, "b (v h w) c -> b v h w c",
                                    v=v, h=latent_h * 4, w=latent_w * 4,
                                    )
            tmp_batch_size = v * latent_h * latent_w
            # simply use uniform grid subsample of point cloud to reduce points
            point_cloud = point_cloud[:, :, ::4, ::4]
            point_cloud = rearrange(point_cloud, "b v h w c -> (b v h w) c")
        else:
            point_cloud = rearrange(prev_means, "b n c -> (b n) c")
            if local_window_update:
                tmp_batch_size = test_window_size * latent_h * latent_w
            else:
                tmp_batch_size = prev_means.shape[1]
        return point_cloud, tmp_batch_size

    def get_vector_state(self, b, v, n, optimizer_input, from_init):
        if from_init:
            # Starting a new scene directly from the initializer
            # State should not be provided
            # Create initial state
            # optimizer_input.prev_output is of type InitializerOutput
            if optimizer_input.prev_output.features is None or self.cfg.init_state_wo_features:
                # Creating state without initializer features
                assert self.cfg.init_state_wo_features
                with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
                    dtype = torch.get_autocast_dtype('cuda')
                    if self.cfg.init_state_type == "constant":
                        state = torch.ones((b, n, self.cfg.state_channels), device=self.device, dtype=dtype)
                    elif self.cfg.init_state_type == "random":
                        state = torch.randn((b, n, self.cfg.state_channels), device=self.device, dtype=dtype)
                    else:
                        raise ValueError(f"Unknown init_state_type {self.cfg.init_state_type}")
                    state = state * self.cfg.init_state_scale
            else:
                # Calculating state from initializer features
                state = self.get_state_from_condition_features(b, optimizer_input.prev_output.features,
                                                               v)  # [B, N, C]

        else:
            # Restarting optimizing a scene from a replay buffer
            state = optimizer_input.prev_output.state.state
            # TODO Naama: need to understand why rearrange here, perhaps something with pruning
            state = rearrange(state, "(b n) c -> b n c", b=b)

        # combine gaussians of all scnes in the batch [B*N, C]
        state = rearrange(state, "b n c -> (b n) c")  # [B*N, C]

        # Do something with window size
        _, _, _, h, w = optimizer_input.context["image"].shape  # [B, V, C, H, W]
        local_window_update, test_window_size, window_end, window_start = optimizer_input.additional_info
        # select initial state
        if local_window_update and self.cfg.local_gaussian_render:
            state = rearrange(state, "(b v h w) c -> b v h w c", b=b, v=v,
                              h=h // self.cfg.latent_downsample,
                              w=w // self.cfg.latent_downsample)
            state = state[:, window_start:window_end, :, :, :]
            state = rearrange(state, "b v h w c -> (b v h w) c")

        return state

    @staticmethod
    def _align_features(features, latent_h: int, latent_w: int) -> list:
        """Resize each feature map to (latent_h, latent_w) if needed and return as a list."""
        out = []
        vals = features.values() if isinstance(features, dict) else features
        for feat in vals:
            if feat.shape[-2:] != (latent_h, latent_w):
                feat = F.interpolate(feat, size=(latent_h, latent_w), mode='bilinear', align_corners=True)
            out.append(feat)
        return out

    def _get_latent_size(self, h: int, w: int) -> tuple[int, int]:
        """Compute latent (H, W) from image (H, W), accounting for init_gaussian_multiple upsampling."""
        latent_h = h // self.cfg.latent_downsample
        latent_w = w // self.cfg.latent_downsample
        if self.cfg.init_gaussian_multiple == 4 and self.cfg.refine_same_num_points:
            latent_h *= 2
            latent_w *= 2
        elif self.cfg.init_gaussian_multiple == 16 and self.cfg.refine_same_num_points:
            latent_h *= 4
            latent_w *= 4
        return latent_h, latent_w

    def render_input_views_for_error_calc(self, context,
                                          local_window_update,
                                          prev_gaussians,
                                          renderer,
                                          window_end,
                                          window_start,
                                          num_refine,
                                          i):
        _, _, _, h, w = context["image"].shape  # [B, V, C, H, W]

        render_res = (h, w)

        # Default rendering parameters
        input_info = context
        start = None
        end = None
        cfg = self.cfg

        # Use only first N views
        if cfg.input_error_num_views > 0:
            end = cfg.input_error_num_views

        # Local window update logic
        elif local_window_update:
            if i >= num_refine - 1:
                return None  # Skip rendering on the last iteration
            start = window_start
            end = window_end

        # Final unified rendering call
        return renderer.forward_batch_subset(
            prev_gaussians,
            input_info,
            render_res,
            start=start,
            end=end,
            return_radii=False
        )

    def get_state_from_condition_features(self, b, condition_features, v):
        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
            if not self.cfg.pt_update_amp and condition_features.dtype == torch.bfloat16:
                condition_features = condition_features.float()
            state = self.update_proj(condition_features.detach())  # [B, C, H, W]
        if self.cfg.init_gaussian_multiple == 4 and self.cfg.refine_same_num_points:
            state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        elif self.cfg.init_gaussian_multiple == 16 and self.cfg.refine_same_num_points:
            state = F.interpolate(state, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            pass
        # Convert to vector of Gaussians per batch [B, N, C]
        state = rearrange(state, "(b v) c h w -> b (v h w) c", b=b, v=v)  # N = v * h * w
        return state

    def get_window_size(self, v):
        test_window_size = None
        if self.cfg.update_window_size > 0:

            local_window_update = True
            # if self.training:
            #     window_start = random.randint(0, v - self.cfg.update_window_size)
            #     window_end = window_start + self.cfg.update_window_size
            # else:
            # fixed window at test time, uniformly move from left to right
            # TODO: loop closure, connect left and right
            if self.training:
                test_window_size = self.cfg.update_window_size
                window_start = random.randint(0, test_window_size)
                window_end = window_start + test_window_size

                if window_end == v:
                    # restart
                    window_start = random.randint(0, test_window_size)
                    window_end = window_start + test_window_size
            else:
                # at least do a full pass of all input views
                # test_window_size = int(np.ceil(v / self.cfg.num_refine))
                test_window_size = self.cfg.update_window_size
                window_start = 0
                window_end = window_start + test_window_size

        else:
            local_window_update = False
            window_start = 0
            window_end = v
        return local_window_update, test_window_size, window_end, window_start

    def prepare_update_input(self, b, i, init_state, input_signal, latent_h, latent_w, local_window_update, point_cloud,
                             tmp_gaussian, state, v, window_end, window_start):
        if self.cfg.replace_init_state:
            state = init_state

        if self.cfg.no_render_error:
            update_input = torch.cat((tmp_gaussian, state), dim=-1)
        else:
            if local_window_update and not self.cfg.local_gaussian_render:
                # select local window
                tmp_gaussian = rearrange(tmp_gaussian, "(b v h w) c -> b v h w c", b=b, v=v, h=latent_h,
                                         w=latent_w)
                tmp_gaussian = tmp_gaussian[:, window_start:window_end, :, :, :]
                tmp_gaussian = rearrange(tmp_gaussian, "b v h w c -> (b v h w) c")

                if i == 0:
                    state = rearrange(state, "(b v h w) c -> b v h w c", b=b, v=v, h=latent_h,
                                      w=latent_w)
                    state = state[:, window_start:window_end, :, :, :]
                    state = rearrange(state, "b v h w c -> (b v h w) c")

                # local point cloud
                point_cloud = rearrange(point_cloud, "(b v h w) c -> b v h w c", b=b, v=v, h=latent_h,
                                        w=latent_w)
                point_cloud = point_cloud[:, window_start:window_end, :, :, :]
                point_cloud = rearrange(point_cloud, "b v h w c -> (b v h w) c")

            update_input = torch.cat((tmp_gaussian, state, input_signal), dim=-1)
        if self.cfg.concat_init_state:
            update_input = torch.cat((update_input, init_state), dim=-1)
        return point_cloud, tmp_gaussian, state, update_input

    def apply_update_module(self, b, latent_h, latent_w, offset, point_cloud, update_input, v, state, iter):

        def recurrent_chunk(update_input, point_cloud, offset):
            pxo = self.update_module[0]([point_cloud, update_input, offset])
            state = self.update_module[1](pxo, iter=iter, b=b, v=v, h=latent_h, w=latent_w)
            return state

        if self.cfg.use_checkpointing or self.cfg.recurrent_use_checkpointing:
            new_state = torch.utils.checkpoint.checkpoint(
                recurrent_chunk,
                update_input, point_cloud, offset,
                use_reentrant=False,
            )
        else:
            new_state = recurrent_chunk(update_input, point_cloud, offset)

        if self.cfg.residual_state:
            new_state = new_state + state
        return new_state

    def apply_delta_gaussian_head(self, b, context, init_state, state, v):
        if self.cfg.update_head_concat_img:
            # pixel unshuffle image
            img_unshuffle = rearrange(context["image"], "b v c h w -> (b v) c h w")
            img_unshuffle = F.pixel_unshuffle(img_unshuffle, downscale_factor=self.cfg.latent_downsample)
            img_unshuffle = rearrange(img_unshuffle, "(b v) c h w -> (b v h w) c", b=b, v=v)
            head_input = torch.cat((state, img_unshuffle), dim=-1)

        else:
            if self.cfg.refine_residual_init_state:
                head_input = state + init_state
            else:
                head_input = state

        if self.cfg.update_head_per_param_heads:
            delta_gaussians = self._apply_per_param_heads(head_input)
        else:
            delta_gaussians = self.update_head(head_input)

        return delta_gaussians

    def _apply_per_param_heads(self, head_input):
        """Run per-parameter-group heads and concatenate results.

        Each head outputs [N, dim+1] where the last dim is the scalar scale.
        Per-group normalize + scale is applied independently.
        """
        deltas = []
        for name, dim in self._per_param_group_dims.items():
            raw = self.update_head[name](head_input)  # [N, dim+1]
            scale = self.scale_act(raw[:, -1:])  # [N, 1]
            delta = raw[:, :-1]  # [N, dim]
            if dim > 1:
                delta = delta / (delta.norm(p=2, dim=-1, keepdim=True) + 1e-8) * scale
            else:
                # 1-d (e.g. opacities): no direction to normalize, just scale magnitude
                delta = delta * scale
            deltas.append(delta)
        return torch.cat(deltas, dim=-1)

    def apply_global_attn(self, b, h, input_signal, latent_h, latent_w,
                          local_window_update, test_window_size, v, w):
        # TODO Naama: do we need local_window?
        assert self.cfg.input_error_resnet_feature
        assert self.cfg.input_error

        if self.cfg.input_gradient and self.cfg.input_error:
            input_render_error = input_signal[..., :self.error_features_channels]
        else:
            input_render_error = input_signal

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.use_amp, dtype=torch.bfloat16):
            for blk in self.update_error_attn:
                if self.cfg.refine_same_num_points:
                    # no downsample, for re10k 256
                    input_render_error = blk(input_render_error, v=v, h=h, w=w)
                else:
                    curr_v = test_window_size if local_window_update else v
                    input_render_error = blk(input_render_error, v=curr_v, h=latent_h, w=latent_w)

        if self.cfg.input_gradient and self.cfg.input_error:
            input_signal[..., :self.error_features_channels] = input_render_error
        else:
            input_signal = input_render_error

        return input_signal

    def prepare_input_signal(self, context, i, gaussians,
                             local_window_update, renderer,
                             window_end, window_start, num_refine):
        # TODO Naama: review
        # make sure at least one of the following is True
        assert self.cfg.input_gradient or self.cfg.input_error
        input_view_features = None
        input_signal = None
        input_render_error = None
        context_render_output = None
        gaussian_grads_raw = None
        gaussian_grads = None
        grad_sign = None
        means2d_grads = None

        # calculate input gradients
        if self.cfg.input_gradient:
            gaussian_grads_raw, gaussian_grads, grad_sign, context_render_output, means2d_grads = (
                self._calc_input_gradients(context, gaussians, renderer)
            )

            input_signal = gaussian_grads_raw

        # When using gradients, context_render_output cannot be used for the meta-training,
        # because there was already one backward pass.
        # So we render again if in training.
        if context_render_output is None or self.training:
            context_render_output = self.render_input_views_for_error_calc(context, local_window_update,
                                                                           gaussians, renderer, window_end,
                                                                           window_start, num_refine, i)

        # calculate input rendering errors
        if self.cfg.input_error:
            if means2d_grads is None and self.cfg.need_2d_grads:
                raise NotImplementedError("Calculating 2dgrad for ADC is not implemented for error input alone")
            input_render_error = self._calc_input_errors(context, i, context_render_output,
                                                         input_view_features,
                                                         local_window_update,
                                                         gaussians.means.detach(),
                                                         window_end,
                                                         window_start)
            input_signal = input_render_error

        if self.cfg.input_gradient and self.cfg.input_error:
            # Concatenate both gradients and errors
            input_signal = torch.cat((input_render_error, gaussian_grads), dim=-1)

        return input_signal, gaussian_grads_raw, gaussian_grads, grad_sign, context_render_output, means2d_grads

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                           * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None

    def debug_reprojection_error(self, means, debug_dict, context, i, latent_h, latent_w):
        # Prepare means (remove singleton dim)
        means = rearrange(means, "b (v h w) c -> b v (h w) c", h=latent_h, w=latent_w)  # [B, V, H*W, 3]

        # Expand extrinsics/intrinsics for broadcast
        extrinsics = context["extrinsics"].unsqueeze(2)  # [B, V, 1, 4, 4]
        intrinsics = context["intrinsics"].unsqueeze(2)  # [B, V, 1, 3, 3]

        # Project
        xy_ray_reconstructed, in_front = project(means, extrinsics, intrinsics)  # [B, V, H*W, 2], [B, V, H*W]

        xy_ray, _ = sample_image_grid((latent_h, latent_w), xy_ray_reconstructed.device)  # [B, V, H*W, 1, 2]
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

        xy_ray = xy_ray.squeeze(-2)  # [B, V, H*W, 2]

        xy_ray_unnorm = xy_ray.clone()
        xy_ray_unnorm[..., 0] *= latent_w
        xy_ray_unnorm[..., 1] *= latent_h

        xy_ray_reconstructed_unnorm = xy_ray_reconstructed.clone()
        xy_ray_reconstructed_unnorm[..., 0] *= latent_w
        xy_ray_reconstructed_unnorm[..., 1] *= latent_h

        reprojection_error = (xy_ray_unnorm - xy_ray_reconstructed_unnorm).abs()

        if debug_dict["reprojection_error"] is None:
            # First iteration, first scene
            debug_dict["reprojection_error"] = [[]]
        elif i == 0:
            # New iteration, new scene
            debug_dict["reprojection_error"].append([])

        debug_dict["reprojection_error"][-1].append(reprojection_error.detach().cpu())

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 6))
        # plt.hist(reprojection_error.flatten().detach().cpu(), bins=100, range=(0, 10))
        # plt.title(f"Reprojection Error - step {i}")
        # plt.xlabel("Error (pixels)")
        # plt.ylabel("Frequency")
        # plt.show()

    def _calc_input_errors(self, context, i, input_render, input_view_features,
                           local_window_update, prev_means,
                           window_end, window_start):
        b, v, _, h, w = context["image"].shape
        # Detach the last rendered object
        input_rgb = input_render.color.detach()
        # compute input view rendering error
        if self.cfg.input_error_resnet_feature:
            input0 = rearrange(input_rgb, "b v c h w -> (b v) c h w")
            if self.cfg.input_error_num_views > 0:
                gt_input = context["image"][:, :self.cfg.input_error_num_views, :, :, :]
            elif local_window_update:
                gt_input = context["image"][:, window_start:window_end, :, :, :]
            else:
                gt_input = context["image"]
            input1 = rearrange(gt_input, "b v c h w -> (b v) c h w")

            transform = _IMAGENET_NORM

            if input_view_features is None:
                assert i == 0
                # first time: extract all features
                concat = torch.cat((input0, input1), dim=0)

                input_tensor = transform(concat)
                with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp,
                                        dtype=torch.bfloat16):
                    # Extract features
                    with torch.no_grad():
                        features = self.update_feature(input_tensor)

                # align to the latent resolution
                latent_h, latent_w = self._get_latent_size(h, w)

                all_features = torch.cat(self._align_features(features, latent_h, latent_w), dim=1)

                render_view_features = all_features[:input0.shape[0]]
                input_view_features = all_features[input0.shape[0]:]

            else:
                # only extract render view features
                with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp,
                                        dtype=torch.bfloat16):
                    # Extract features
                    with torch.no_grad():
                        features = self.update_feature(transform(input0))

                # align to the latent resolution
                latent_h, latent_w = self._get_latent_size(h, w)

                render_view_features = torch.cat(self._align_features(features, latent_h, latent_w), dim=1)

            corr = render_view_features - input_view_features

            if self.cfg.input_error_num_views > 0:
                # pad to V views
                curr_v = self.cfg.input_error_num_views
                indices = torch.arange(v) * curr_v // v
                corr = rearrange(corr, "(b v) c h w -> b v c h w", b=b)
                corr = corr[torch.arange(b).unsqueeze(1), indices, :, :, :]
                input_render_error = rearrange(corr, "b v c h w -> b (v h w) c")
            else:
                input_render_error = rearrange(corr, "(b v) c h w -> b (v h w) c", b=b)

        else:
            input_render_error = (input_render.color - context["image"]).abs()  # [B, V, 3, H, W]
            input_render_error = rearrange(input_render_error, "b v c h w -> (b v) c h w")

            if self.cfg.input_error_rgb_no_shuffle:
                # bilinear
                input_render_error = F.interpolate(input_render_error,
                                                   scale_factor=1. / self.cfg.latent_downsample,
                                                   mode='bilinear', align_corners=True)
            else:
                # pixel unshuffle
                # TODO: when fps is used, how to reshape the render error to make sure its somehow pixel aligned to the gaussians
                input_render_error = F.pixel_unshuffle(input_render_error,
                                                       downscale_factor=self.cfg.latent_downsample)

            input_render_error = rearrange(input_render_error, "(b v) c h w -> b (v h w) c", b=b,
                                           v=v)  # [B, N, C]

        # include both feature error and image error
        if self.cfg.input_error_add_rgb_feature:
            rgb_render_error = input_render.color - context["image"]
            rgb_render_error = rearrange(rgb_render_error, "b v c h w -> (b v) c h w")

            if self.cfg.input_error_rgb_no_shuffle:
                # bilinear
                rgb_render_error = F.interpolate(rgb_render_error, scale_factor=1. / self.cfg.latent_downsample,
                                                 mode='bilinear', align_corners=True)
            else:
                # pixel unshuffle
                # TODO: when fps is used, how to reshape the render error to make sure its somehow pixel aligned to the gaussians
                rgb_render_error = F.pixel_unshuffle(rgb_render_error,
                                                     downscale_factor=self.cfg.latent_downsample)

            rgb_render_error = rearrange(rgb_render_error, "(b v) c h w -> b (v h w) c", b=b, v=v)  # [B, N, C]

            rgb_render_error = self.update_rgb_error_proj(rgb_render_error)
            input_render_error = input_render_error + rgb_render_error

        return input_render_error

    def get_input_error_feature_extractor(self):
        update_feature = None
        # resnet feature
        if self.cfg.input_error_resnet_feature:
            update_feature = ResNetFeatureWarpper(
                shallow_resnet_feature=self.cfg.input_error_shallow_resnet_feature)

            if self.cfg.input_error_no_freeze_resnet_feature:
                # remove unused layers
                # NOTE: layer 3 is also not used
                update_feature.layer3 = nn.Identity()
                update_feature.train()
                for params in update_feature.parameters():
                    params.requires_grad = True
            else:
                update_feature.eval()

                for params in update_feature.parameters():
                    params.requires_grad = False

        return update_feature

    def update_delta_for_gradients_input(self, delta_gaussians, grad_sign, normalized_grad,
                                         visibility_scale: Tensor | None = None):
        if self.cfg.input_gradient:
            delta_gaussians = delta_gaussians / self.cfg.input_gradient_scale
            if self.cfg.input_gradient_log:
                grad_sign = rearrange(grad_sign, "b n c -> (b n) c")
                # recover log scale for applying the deltas.
                # For loss calculation the delta should still be in log scale

                delta_gaussians = grad_sign * (delta_gaussians.exp() - 1e-8)

                if self.cfg.input_gradient_log_clip_deltas > 0:
                    # clip the delta to avoid too large updates
                    clip_value = self.cfg.input_gradient_log_clip_deltas
                    clip_mask = delta_gaussians.abs() > clip_value
                    delta_gaussians[clip_mask] = delta_gaussians[clip_mask].sign() * clip_value

            # TODO Naama: move these two, as they are not related to gradients
            if self.cfg.update_head_scale_mag:
                out_channels = delta_gaussians.shape[-1]
                param_num = out_channels / 2
                assert param_num.is_integer()
                param_num = int(param_num)
                scale = delta_gaussians[:, :param_num]
                mag = delta_gaussians[:, param_num:]
                delta_gaussians = scale * 0.01 * torch.exp(mag * 0.01)

            if self.cfg.update_head_scalar_scale:
                if self.cfg.update_head_per_param_heads:
                    # Already handled in _apply_per_param_heads — nothing to do here
                    pass
                elif self.cfg.update_head_per_param_scales:
                    # Feature B: per-group scalar scales
                    num_groups = len(self._per_param_group_dims)
                    scales = delta_gaussians[:, -num_groups:]  # [G, num_groups]
                    scales = self.scale_act(scales)
                    deltas = delta_gaussians[:, :-num_groups]  # [G, D]

                    normalized_deltas = []
                    offset = 0
                    for i, (name, dim) in enumerate(self._per_param_group_dims.items()):
                        group_delta = deltas[:, offset:offset + dim]  # [G, dim]
                        group_scale = scales[:, i:i + 1]  # [G, 1]
                        if dim > 1:
                            group_delta = group_delta / (group_delta.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                        group_delta = group_delta * group_scale
                        normalized_deltas.append(group_delta)
                        offset += dim

                    delta_gaussians = torch.cat(normalized_deltas, dim=-1)
                else:
                    # Original global scalar scale
                    scale = delta_gaussians[:, -1:]  # [G, 1]
                    scale = self.scale_act(scale)  # make sure scale is positive
                    deltas_unnorm = delta_gaussians[:, :-1]  # [G, D]
                    deltas_norm = deltas_unnorm / (deltas_unnorm.norm(p=2, dim=1, keepdim=True) + 1e-8)  # [G, D]
                    delta_gaussians = deltas_norm * scale

            if visibility_scale is not None:
                delta_gaussians = delta_gaussians * visibility_scale

            if self.cfg.scale_residual_grads:
                delta_gaussians = delta_gaussians * normalized_grad * self.cfg.gradient_update_scale  # 1.0

            # To match the default behavior of SGD, Adam, and other optimizers, deltas are negated.
            # SGD applies the gradients as `x = x - lr * grad`, while resaplt applies them as `x = x + lr * deltas`.
            delta_gaussians = -delta_gaussians

        return delta_gaussians

    def _calc_input_gradients(self, context, gaussians, renderer):
        assert not self.cfg.input_gradient_same_loss, "input_gradient_same_loss is not implemented"
        _, v, _, h, w = context["image"].shape

        with torch.enable_grad():

            # Unpack gaussians
            means, scales, rotations_unnorm, opacities_raw, shs = unpack_gaussians(
                gaussians,
                scales_log=self.cfg.opt_scales_before_act,
                opacities_logit=True,
                opacities_unsqueeze=True,
                detach=True,
                clone=False,
                requires_grad=True,
                scales_lims=(self.cfg.clamp_min_scale, self.cfg.clamp_refine_max_scale),
                raw_opacities_lims=(self.cfg.clamp_min_raw_opacities, self.cfg.clamp_max_raw_opacities)
            )

            # Create temporary Gaussians with same values but requires_grad=True
            grad_batch_size = self.cfg.input_gradients_chunk_size
            if grad_batch_size == -1:
                grad_batch_size = v
            gaussian_grads = 0
            means2d_grads_chunks = []
            nr_chunks = math.ceil(v / grad_batch_size)

            # Pre-compute shapes and config flags outside the loop
            shs_shape = (shs.shape[0], shs.shape[1], 3, -1)
            opt_scales_before_act = self.cfg.opt_scales_before_act
            # Pre-compute normalized rotations once (not in gradient inputs, so no grad needed)
            with torch.no_grad():
                rotations = rotations_unnorm / (rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8)

            for chunk_idx, start, stop in chunk_index_iter(v, grad_batch_size):
                # zero grads

                means = means.detach().requires_grad_(True)
                scales = scales.detach().requires_grad_(True)
                rotations_unnorm = rotations_unnorm.detach().requires_grad_(True)
                opacities_raw = opacities_raw.detach().requires_grad_(True)
                shs = shs.detach().requires_grad_(True)

                # Apply activation to scales if needed (before calculating covariance)
                scales_act = scales.exp() if opt_scales_before_act else scales

                tmp_gaussians = Gaussians(
                    means=means,
                    covariances=None,
                    harmonics=shs.view(shs_shape),
                    opacities=torch.sigmoid(opacities_raw.squeeze(-1)),
                    scales=scales_act,
                    rotations=rotations,
                    rotations_unnorm=rotations_unnorm,
                )

                # render input views, calculate inner loss and backprop to get gradients
                context_render_output = renderer.forward_batch_subset(
                    tmp_gaussians,
                    context,
                    start=start,
                    end=stop,
                    image_shape=(h, w),
                )

                inputs = [means, scales, rotations_unnorm, opacities_raw, shs]

                if self.cfg.need_2d_grads:
                    assert context_render_output.means2d is not None, "output_renderer.means2d is None"
                    means2d = context_render_output.means2d  # [B, V, N, 2]
                    # means2d.retain_grad()  # retain grad for means2d grads computation
                    inputs.append(means2d)

                inner_loss = inner_loss_for_input_gradients(
                    context["image"][:, start:stop],
                    context_render_output,
                    reduction=self.cfg.input_gradient_loss_reduction,
                    with_ssim=self.cfg.input_gradient_with_ssim_loss,
                )
                if self.cfg.opacity_reg_lambda > 0.0:
                    inner_loss = inner_loss + self.cfg.opacity_reg_lambda * torch.sigmoid(opacities_raw).mean()
                grads = torch.autograd.grad(outputs=inner_loss,
                                            inputs=inputs,
                                            create_graph=False,
                                            retain_graph=False,
                                            )

                gaussian_grads = gaussian_grads + torch.cat(grads[:5], dim=-1)  # [B, G, D]
                assert not torch.isnan(gaussian_grads).any(), "NaN detected in gaussian_grads"
                if self.cfg.need_2d_grads:
                    means2d_grads_chunks.append(grads[5])  # [B, V_chunk, N, 2]

            gaussian_grads = gaussian_grads / nr_chunks

            if self.cfg.need_2d_grads:
                means2d_grads = torch.cat(means2d_grads_chunks, dim=1)  # [B, V, N, 2]
                if self.cfg.input_gradient_loss_reduction == "mean_pixels_sum_views":
                    means2d_grads = means2d_grads / v
            else:
                means2d_grads = None

            gaussian_grads_raw = gaussian_grads * self.cfg.input_gradient_scale
            if self.cfg.input_gradient_log:
                # log gradients
                grads_sign = gaussian_grads.sign()
                gaussian_grads_raw = (gaussian_grads_raw.abs() + 1e-8).log()
            else:
                grads_sign = None

            # Detach gradients to avoid gradient flow through the input
            gaussian_grads = gaussian_grads.detach()
            gaussian_grads_raw = gaussian_grads_raw.detach()
            if grads_sign is not None:
                grads_sign = grads_sign.detach()

        # Returning also the render output, but it can only be used for visualization,
        # as we already backpropogate gradients through it
        return gaussian_grads_raw, gaussian_grads, grads_sign, context_render_output, means2d_grads


def select_gaussian_subset(gaussians, window_start, window_end, v, h, w):
    """Select a subset of gaussians based on view window. Optimized to avoid rearrange overhead."""
    b = gaussians.means.shape[0]
    hw = h * w
    window_v = window_end - window_start
    new_n = window_v * hw

    # Helper to slice view dimension efficiently using view+slice+reshape instead of rearrange
    def slice_tensor(t, extra_dims):
        # t shape: [b, v*h*w, *extra_dims] -> [b, window_v*h*w, *extra_dims]
        shape = (b, v, hw) + extra_dims
        new_shape = (b, new_n) + extra_dims
        return t.view(shape)[:, window_start:window_end, :].reshape(new_shape)

    means = slice_tensor(gaussians.means, (3,))
    covariances = slice_tensor(gaussians.covariances, (3, 3)) if gaussians.covariances is not None else None
    shs = slice_tensor(gaussians.harmonics, gaussians.harmonics.shape[2:])
    opacities = slice_tensor(gaussians.opacities.unsqueeze(-1), ()).squeeze(-1)
    scales = slice_tensor(gaussians.scales, (3,))
    rotations = slice_tensor(gaussians.rotations, (4,)) if gaussians.rotations is not None else None
    rotations_unnorm = slice_tensor(gaussians.rotations_unnorm, (4,))

    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=shs,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        rotations_unnorm=rotations_unnorm,
    )


def replace_window(original, window, window_start, window_end, dim=1):
    slices = []
    if window_start > 0:
        # TODO: detach or not
        # slices.append(original[:, :window_start].detach())
        slices.append(original[:, :window_start])
    slices.append(window)
    if window_end < original.shape[dim]:
        # TODO: detach or not
        # slices.append(original[:, window_end:].detach())
        slices.append(original[:, window_end:])
    return torch.cat(slices, dim=dim)


def freeze_batchnorm_layers(model):
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module,
                                                                                                  nn.BatchNorm3d):
            module.eval()  # Set to evaluation mode
            for param in module.parameters():
                param.requires_grad = False  # Freeze parameters

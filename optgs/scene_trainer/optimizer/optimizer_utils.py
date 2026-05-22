import warnings
from dataclasses import dataclass
import torch.nn.functional as F
import math
from typing import List, Tuple, Iterator
from typing import Literal, Generic, TypeVar
from fused_ssim import allowed_padding, FusedSSIMMap
import torch
from gsplat import fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_indices_in_range
from nerfacc import accumulate_along_rays, render_weight_from_alpha
from torch import Tensor
from optgs.model.decoder.decoder import Decoder, DecoderOutput
from optgs.model.types import Gaussians
from einops import rearrange
from tqdm import tqdm
import gc
import torch.autograd.profiler as profiler
from optgs.misc.memory_profiler import profile_gpu_memory, report_gpu_tensors

T = TypeVar("T")
GPU_MEM_PROFILING = False  # set to True to enable GPU memory profiling


def split_grads(grads_tensor, cfg):
    
    assert isinstance(grads_tensor, Tensor), "grads_tensor is not a Tensor"
    
    # handle case where grads_tensor has batch dimension
    if grads_tensor.ndim == 3:
        assert grads_tensor.shape[0] == 1, "Batch size > 1 not supported for grads_tensor with ndim 3"
        grads_tensor = grads_tensor.squeeze(0)  # [N, D]
    
    # Split the last dimension
    means, scales, rotations, opacities, shs = torch.split(
        grads_tensor, (3, 3, 4, 1, 3 * cfg.sh_d), dim=-1
    )
    
    shs = rearrange(shs, "n (c x) -> n c x", c=3, x=cfg.sh_d)  # [N, 3, sh_d]
    sh0s = shs[..., 0:1]
    if cfg.sh_d > 1:
        shNs = shs[..., 1:]
    else:
        shNs = None
    grads: dict = {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "sh0s": sh0s,
        "shNs": shNs,
    }
    return grads


def inner_loss_for_input_gradients(
    gt_images,
    output_renderer: DecoderOutput,
    reduction: str = "mean",
    with_ssim: bool = True,
) -> Tensor:
    # compute scalar loss
    # assume batch size 1
    assert gt_images.shape[0] == 1
    assert gt_images.shape == output_renderer.color.shape

    l1_loss = (output_renderer.color - gt_images).abs()
    if reduction == "mean":
        l1_loss = l1_loss.mean()
    elif reduction == "sum":
        l1_loss = l1_loss.sum()
    elif reduction == "mean_pixels_sum_views":
        l1_loss = l1_loss.mean(dim=(-1, -2, -3)).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")

    if not with_ssim:
        return l1_loss

    gt_images_for_ssim = gt_images.clone() if gt_images.is_inference() else gt_images
    ssim_loss = fused_ssim_with_reduction(
        rearrange(output_renderer.color, "b v c h w -> (b v) c h w"),
        rearrange(gt_images_for_ssim, "b v c h w -> (b v) c h w"),
        padding="valid",
        reduction=reduction,
        loss=True,  # returns mean(1 - ssim), i.e. the SSIM loss
    )
    return 0.8 * l1_loss + 0.2 * ssim_loss


def squeeze_grad_dict(grad_dict):
    for k, v in grad_dict.items():
        if v is not None:
            grad_dict[k] = v.squeeze(0)
    return grad_dict


def smooth_grads(grads: dict, smoothers: dict) -> dict:
    smoothed_grads = {}
    for k, v in grads.items():
        if k not in smoothers:
            continue
        else:
            if v is not None:
                smoothed_grads[k] = smoothers[k](v)
            else:
                smoothed_grads[k] = None
    return smoothed_grads

def chunk_ranges(v: int, chunk_size: int) -> List[Tuple[int, int]]:
    """
    Return a list of (start, stop) index ranges that partition [0, v).
    Last chunk may be smaller if v % chunk_size != 0.
    Example: chunk_ranges(10, 4) -> [(0,4),(4,8),(8,10)]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    ranges = []
    start = 0
    while start < v:
        stop = min(start + chunk_size, v)
        ranges.append((start, stop))
        start = stop
    return ranges

def chunk_slices(v: int, chunk_size: int, dim: int = 1) -> List[slice]:
    """
    Return a list of slice objects that slice along axis `dim`.
    Use like: tensor[(slice(None), slice_start_stop, ...)] — easier: use helper below.
    NOTE: slice objects don't encode the axis; they only give start/stop; see usage.
    """
    return [slice(s, e) for s, e in chunk_ranges(v, chunk_size)]

def chunk_index_iter(v: int, chunk_size: int) -> Iterator[Tuple[int,int,int]]:
    """
    Iterate chunk info as (chunk_idx, start, stop) for convenience.
    """
    for idx, (s, e) in enumerate(chunk_ranges(v, chunk_size)):
        yield idx, s, e



def fused_ssim_with_reduction(img1, img2, padding="same", train=True, reduction="mean", loss=False):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)  # [v c h w]

    if loss:
        ssim_map = 1 - ssim_map

    if reduction == "mean":
        return ssim_map.mean()
    elif reduction == "sum":
        return ssim_map.sum()
    elif reduction == "mean_pixels_sum_views":
        # Mean over spatial (h, w) and channel (c) dims, then sum over views (v)
        return ssim_map.mean(dim=(-1, -2, -3)).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def calc_input_gradients(
    iter_context,
    prev_means,
    prev_scales_raw,
    prev_rotations_unnorm,
    prev_opacities_raw,   # [B, N] — may be a non-leaf view of gaussians.opacities
    prev_shs,             # [B, N, 3, sh_d]
    renderer: Decoder,
    need_2d_grads: bool,
    chunk_size: int | None,
    any_adc: bool = True,
    sh_degree: int | None = None,
    meta_bufs: dict | None = None,  # mutable dict populated/reused across calls for radii & visibility
    loss_reduction: str = "mean",
    loss_with_ssim: bool = True,
    opacity_reg_lambda: float = 0.0,  # L1 opacity regularization weight (3DGS-MCMC)
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor | None] | None]:

    b, v, _, h, w = iter_context["image"].shape
    assert b == 1, "Batch size > 1 not supported for post-processing"

    if chunk_size == -1:
        chunk_size = v
    nr_chunks = math.ceil(v / chunk_size)
    N = prev_means.shape[1]
    device = prev_means.device

    # --- Grad setup ---
    # Gradients are obtained functionally via torch.autograd.grad below, so .grad buffers
    # are never read or written. Enable requires_grad on the leaf params as a fallback if
    # the caller did not already set it up. Order matters: it defines the autograd.grad
    # input order and therefore the order of the returned per-param gradients.
    _leaf_params = [prev_means, prev_scales_raw, prev_rotations_unnorm, prev_opacities_raw, prev_shs]
    for t in _leaf_params:
        if not t.requires_grad:
            t.requires_grad_(True)

    # --- Allocate or reuse radii / visibility buffers (only needed when any_adc) ---
    bufs_valid = (
        any_adc
        and meta_bufs is not None
        and meta_bufs.get("N") == N
        and meta_bufs.get("v") == v
    )
    if bufs_valid:
        radii_all = meta_bufs["radii"]
        visibility_all = meta_bufs["visibility"]
        means2d_grads_all = meta_bufs.get("means2d_grads")
        if need_2d_grads and means2d_grads_all is None:
            means2d_grads_all = torch.empty((b, v, N, 2), dtype=torch.float32, device=device)
            meta_bufs["means2d_grads"] = means2d_grads_all
    elif any_adc:
        radii_all       = torch.empty((b, v, N, 2), dtype=torch.float32, device=device)
        visibility_all  = torch.empty((b, v, N),    dtype=torch.bool,    device=device)
        means2d_grads_all = (
            torch.empty((b, v, N, 2), dtype=torch.float32, device=device)
            if need_2d_grads else None
        )
        if meta_bufs is not None:
            meta_bufs.update({"N": N, "v": v, "radii": radii_all,
                              "visibility": visibility_all, "means2d_grads": means2d_grads_all})
    else:
        radii_all = visibility_all = means2d_grads_all = None

    # --- Forward + autograd.grad loop ---
    # Per-chunk gradients for the leaf params are summed here, then averaged below.
    accumulated_grads: list[Tensor] | None = None
    with torch.enable_grad():
        assert not torch.is_inference_mode_enabled()

        for chunk_idx, start, stop in tqdm(chunk_index_iter(v, chunk_size), disable=nr_chunks <= 1,
                                           desc="Computing input gradients in chunks"):
            image_chunk       = iter_context["image"][:, start:stop]
            extrinsics_chunk  = iter_context["extrinsics"][:, start:stop]
            intrinsics_chunk  = iter_context["intrinsics"][:, start:stop]
            near_chunk        = iter_context["near"][:, start:stop]
            far_chunk         = iter_context["far"][:, start:stop]

            prev_opacities = torch.sigmoid(prev_opacities_raw)
            prev_scales    = torch.exp(prev_scales_raw)
            prev_rotations = F.normalize(prev_rotations_unnorm, dim=-1)

            if sh_degree is not None:
                prev_shs_for_render = prev_shs[..., :(sh_degree + 1) ** 2]
            else:
                prev_shs_for_render = prev_shs

            tmp_gaussians = Gaussians(
                means=prev_means,
                covariances=None,
                harmonics=prev_shs_for_render,
                opacities=prev_opacities,
                scales=prev_scales,
                rotations=prev_rotations,
                rotations_unnorm=prev_rotations_unnorm,
                stores_activated=True,
            )

            if GPU_MEM_PROFILING:
                output_renderer: DecoderOutput = profile_gpu_memory(
                    fn=renderer.forward, gaussians=tmp_gaussians,
                    extrinsics=extrinsics_chunk, intrinsics=intrinsics_chunk,
                    near=near_chunk, far=far_chunk, image_shape=(h, w))
            else:
                output_renderer: DecoderOutput = renderer.forward(
                    gaussians=tmp_gaussians,
                    extrinsics=extrinsics_chunk, intrinsics=intrinsics_chunk,
                    near=near_chunk, far=far_chunk, image_shape=(h, w))

            loss = inner_loss_for_input_gradients(image_chunk, output_renderer,
                                                  reduction=loss_reduction, with_ssim=loss_with_ssim)

            # L1 opacity regularization (3DGS-MCMC) folded into the differentiated loss.
            grad_loss = loss
            if opacity_reg_lambda > 0.0:
                grad_loss = loss + opacity_reg_lambda * torch.sigmoid(prev_opacities_raw).mean()

            grad_inputs = list(_leaf_params)
            if need_2d_grads:
                assert output_renderer.means2d is not None
                grad_inputs.append(output_renderer.means2d)

            chunk_grads = torch.autograd.grad(grad_loss, grad_inputs,
                                              create_graph=False, retain_graph=False)

            param_grads = [g.detach() for g in chunk_grads[:5]]
            if accumulated_grads is None:
                accumulated_grads = param_grads
            else:
                accumulated_grads = [a + g for a, g in zip(accumulated_grads, param_grads)]

            # store per-chunk meta
            if any_adc:
                radii_all[:, start:stop]      = output_renderer.radii
                visibility_all[:, start:stop] = output_renderer.visibility_filter
                if need_2d_grads:
                    means2d_grads_all[:, start:stop] = chunk_grads[5].detach()

    # --- Average grads for multi-chunk ---
    if nr_chunks > 1:
        inv = 1.0 / nr_chunks
        accumulated_grads = [g * inv for g in accumulated_grads]

    means_grads, scales_raw_grads, rotations_unnorm_grads, opacities_raw_grads, harmonics_grads = accumulated_grads

    sh0s_grads = harmonics_grads[..., 0:1]
    shNs_grads = harmonics_grads[..., 1:] if harmonics_grads.shape[-1] > 1 else None

    grads = {
        "means":     means_grads,
        "scales":    scales_raw_grads,
        "rotations": rotations_unnorm_grads,
        "opacities": opacities_raw_grads,
        "sh0s":      sh0s_grads,
        "shNs":      shNs_grads,
    }

    meta_for_adc = {
        "visibility_filter": visibility_all,
        "radii":             radii_all,
        "means_2d_grads":    means2d_grads_all if need_2d_grads else None,
    } if any_adc else None

    return loss, grads, meta_for_adc
    

def unpack_gaussians(
    gaussians: Gaussians,
    scales_log: bool, 
    opacities_logit: bool,
    opacities_unsqueeze: bool,
    detach: bool = True, 
    clone: bool = False, 
    requires_grad: bool = False,
    scales_lims: tuple | None = None,  # post activation (1e-6, 3)
    raw_opacities_lims: tuple | None = None,  # pre activation (-7, 7)
):
    """ Unpack Gaussian parameters and invert opacities and scales.

    # TODO Naama: fix this
    Clamp values for scales are in post-activation space, i.e., after exponentiation.
    Clamp values for opacities are in pre-activation space, i.e., before sigmoid

    """

    # Means
    means = gaussians.means  # [B, N, 3]

    # Scales
    scales = gaussians.scales  # [B, N, 3]
    if scales_lims is not None:
        scales = torch.clamp(scales, scales_lims[0], scales_lims[1])
    # if self.cfg.opt_scales_before_act:
    if scales_log:
        # Invert also scales
        scales = torch.log(scales + 1e-8)

    # Quaternions
    # use unnormalized rotations since we are going to refine the unnormed rotations
    rotations_unnorm = gaussians.rotations_unnorm  # [B, N, 4]

    # Opacities
    # before sigmoid, eps is necessary, otherwise might be nan
    if opacities_logit:
        opacities_raw = torch.logit(gaussians.opacities, eps=1e-7)  # [B, N]
        if raw_opacities_lims is not None:
            opacities_raw = torch.clamp(opacities_raw, raw_opacities_lims[0], raw_opacities_lims[1])
    else:
        opacities_raw = gaussians.opacities  # [B, N]

    if opacities_unsqueeze:
        opacities_raw = opacities_raw.unsqueeze(-1)  # [B, N, 1]

    # SHs - use flatten instead of rearrange for speed
    shs = gaussians.harmonics  # [B, N, 3, 9]
    shs = shs.flatten(-2)  # [B, N, C] - faster than rearrange

    if gaussians.sel is not None:
        # TODO Naama: move method to Gaussians class
        sel = gaussians.sel  #  [B, N]
        means = means[:, sel]
        opacities_raw = opacities_raw[:, sel]
        rotations_unnorm = rotations_unnorm[:, sel]
        scales = scales[:, sel]
        shs = shs[:, sel]

    if detach:
        means = means.detach()
        opacities_raw = opacities_raw.detach()
        rotations_unnorm = rotations_unnorm.detach()
        scales = scales.detach()
        shs = shs.detach()

    if clone:
        means = means.clone()
        opacities_raw = opacities_raw.clone()
        rotations_unnorm = rotations_unnorm.clone()
        scales = scales.clone()
        shs = shs.clone()

    if requires_grad:
        means.requires_grad_(True)
        opacities_raw.requires_grad_(True)
        rotations_unnorm.requires_grad_(True)
        scales.requires_grad_(True)
        shs.requires_grad_(True)

    # # predicting multiple gaussians per point, init new gaussians by copying with scaled opacities
    # if self.cfg.reinit_gaussian_when_refine_multiple and self.cfg.refine_gaussian_multiple > 1:
    #     raise NotImplementedError
    #     # This should only be called at the first iteration
    #     # TODO Naama: might be bug if we use replay buffer
    #     repeat = self.cfg.refine_gaussian_multiple
    #     prev_means = prev_means.repeat(1, repeat, 1)
    #     prev_scales = prev_scales.repeat(1, repeat, 1)
    #     prev_rotations_unnorm = prev_rotations_unnorm.repeat(1, repeat, 1)
    #
    #     # scale down opacities
    #     prev_opacities_raw = prev_opacities_raw.repeat(1, repeat, 1)  # smaller opacities, important
    #     # Given y = sigmoid(x), to get new x' such that sigmoid(x') = y / K:
    #     # The formula is: x' = x + log((1 - y) / (K - y))
    #     # This adjusts x so that the sigmoid output is scaled down by a factor of K
    #     tmp_sigmoid = prev_opacities_raw.sigmoid()
    #     # print(tmp_sigmoid.mean().item())
    #     prev_opacities_raw = prev_opacities_raw + torch.log((1 - tmp_sigmoid) / (repeat - tmp_sigmoid))
    #
    #     prev_shs = prev_shs.repeat(1, repeat, 1)
    #
    #     # TODO: this part not ready

    return means, scales, rotations_unnorm, opacities_raw, shs


def get_gaussian_param_slices(sh_d: int) -> dict:
    """Return index slices for each Gaussian parameter group in the packed vector.

    Layout (must match pack_gaussians):
        [means(3) | scales(3) | quats(4) | opacities(1) | shs(3*sh_d)]
    """
    sh_end = 11 + 3 * sh_d
    return {
        "means":     slice(0, 3),
        "scales":    slice(3, 6),
        "quats":     slice(6, 10),
        "opacities": slice(10, 11),
        "sh0":       slice(11, sh_end, sh_d),
        "shN":       [i for i in range(11, sh_end) if (i - 11) % sh_d != 0],
    }


def get_gaussian_param_sizes(sh_d: int) -> dict:
    """Return the element count for each Gaussian parameter group.

    Layout matches pack_gaussians / get_gaussian_param_slices:
        [means(3) | scales(3) | quats(4) | opacities(1) | shs(3*sh_d)]
    """
    return {
        "means":     3,
        "scales":    3,
        "quats":     4,
        "opacities": 1,
        "shs":       3 * sh_d,
    }


def pack_gaussians(
    means: Tensor,
    scales: Tensor,
    rotations_unnorm: Tensor,
    opacities_raw: Tensor,
    shs: Tensor,
) -> Tensor:
    """Concatenate unpacked Gaussian parameters into a single [B, N, C] vector.

    Layout (must match get_gaussian_param_slices):
        [means(3) | scales(3) | quats(4) | opacities(1) | shs(3*sh_d)]
    """
    return torch.cat((means, scales, rotations_unnorm, opacities_raw, shs), dim=-1)


def get_visibility_contribution_from_gaussian_obj(views_info, gaussians, image_shape=None, render_image=False) -> tuple[Tensor, dict]:
    """
    Args:
        views_info: dict containing:
            "extrinsics": Tensor of shape [B, V, 4, 4]
            "intrinsics": Tensor of shape [B, V, 3, 3]
            "image": Tensor of shape [B, V, C, H, W] (Optional, only for shape reference)
            "near": Tensor of shape [B, 1]
            "far": Tensor of shape [B, 1]
        gaussians: Gaussian object containing:
            .means: Tensor of shape [B, N, 3]
            .rotations_unnorm: Tensor of shape [B, N, 4]
            .scales: Tensor of shape [B, N, 3]
            .opacities: Tensor of shape [B, N]
        image_shape: Optional tuple (width, height). If None, use the shape from views_info["image"].
    Returns a (N,) shaped tensor whose entry k is the visibility contribution of the k-th Gaussian.
    out[k] = sum_{c,i,j}^{C, H, W} w_{k,c,i,j}
    ️
    """
    # Context can be either context or target
    # TODO Naama: check visibility for both context and target views
    b = gaussians.means.shape[0]
    assert b == 1
    # Data preparation
    means = gaussians.means[0]  # [N, 3]

    # Not sure why, the rendering uses it and says the rastereization will normalize
    quats = gaussians.rotations_unnorm[0]
    quats = quats[:, [3, 0, 1, 2]]  # [N, 4]  # xyzw to wxyz

    scales = gaussians.scales[0]  # [N, 3]

    opacities = gaussians.opacities[0]  # [N]

    viewmats = views_info["extrinsics"][0]  # [V, 4, 4]
    viewmats = viewmats.inverse()

    Ks = views_info["intrinsics"][0].clone()  # [V, 3, 3]
    if image_shape is not None:
        width, height = image_shape
    else:
        width = views_info["image"].shape[-1]
        height = views_info["image"].shape[-2]
    Ks[:, 0] *= width
    Ks[:, 1] *= height

    near = views_info["near"][0, 0].item()
    far = views_info["far"][0, 0].item()

    with torch.no_grad():
        weight_vis_contribution, info = get_gaussians_visibility_contribution(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=near,
            far_plane=far,
            eps2d=0.1,
            rasterize_mode="antialiased",
        )

    return weight_vis_contribution, info


def get_gaussians_visibility_contribution(
        means: Tensor,  # [N, 3]
        quats: Tensor,  # [N, 4]
        scales: Tensor,  # [N, 3]
        opacities: Tensor,  # [N]
        viewmats: Tensor,  # [V, 4, 4]
        Ks: Tensor,  # [V, 3, 3]
        width: int,
        height: int,
        # set these as in your render function
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        eps2d: float = 0.3,
        tile_size: int = 16,
        rasterize_mode: Literal["classic", "antialiased"] = "antialiased",
        batch_per_iter: int = 100,
) -> tuple[Tensor, dict]:
    """
    Returns a (N,) shaped tensor whose entry k is the visibility contribution of the k-th Gaussian.
    out[k] = sum_{c,i,j}^{C, H, W} w_{k,c,i,j}
    """
    N = means.shape[0]
    V = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (V, 4, 4), viewmats.shape
    assert Ks.shape == (V, 3, 3), Ks.shape

    # Project Gaussians to 2D.
    # The results are with shape [V, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=(rasterize_mode == "antialiased"),
    )

    # import matplotlib.pyplot as plt
    # view_id = 0  # choose a view to inspect
    # image = torch.ones((3, height, width))  # [3, H, W]
    # image = image.permute(1, 2, 0)
    # image = (image * 255).clamp(0, 255).byte().cpu().detach().numpy()
    #
    # # Get 2D projected points and depth
    # x = means2d[view_id, :, 0].cpu().detach().numpy()
    # y = means2d[view_id, :, 1].cpu().detach().numpy()
    #
    # # Optional: mask out invalid points (e.g., outside image or radius == 0)
    # H, W = image.shape[:2]
    # valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    #
    # # Plot
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)  # Background image
    # plt.scatter(x[valid_mask], y[valid_mask], c=means[:, -1][valid_mask].cpu().detach().numpy(), cmap='viridis', s=2)
    # # plt.gca().invert_yaxis()  # Optional: for image coordinate convention
    # plt.title("Overlay: Projected Gaussians (colored by depth)")
    # plt.colorbar(label="Depth")
    # plt.show()


    opacities = opacities.repeat(V, 1)  # [V, N]

    if compensations is not None:
        opacities = opacities * compensations

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_images=V,
        image_ids=None,
        gaussian_ids=None,
    )
    isect_offsets = isect_offset_encode(isect_ids, V, tile_width, tile_height)

    vis_contributions_sum, render_alphas, gaussian_weights_per_view = _gaussians_vis_contribution(
        means2d,
        conics,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        batch_per_iter=batch_per_iter,
    )  # (N,)

    return vis_contributions_sum, {"alphas": render_alphas,
                                   "radii": radii,
                                   "means2d": means2d,
                                   "conics": conics,
                                   "depths": depths,
                                   "weights_per_view": gaussian_weights_per_view}  # (N,)


def _gaussians_vis_contribution(
        means2d: Tensor,  # [V, N, 2]
        conics: Tensor,  # [V, N, 3]
        opacities: Tensor,  # [V, N]
        image_width: int,
        image_height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [V, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        batch_per_iter: int = 100,
):
    V, N = means2d.shape[:2]
    n_isects = len(flatten_ids)
    device = means2d.device

    render_alphas = torch.zeros((V, image_height, image_width, 1), device=device)
    gaussian_weights = torch.zeros(N, dtype=opacities.dtype, device=device)
    gaussian_weights_per_view = torch.zeros((V, N), dtype=opacities.dtype, device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    isect_offsets_fl = torch.cat(
        [isect_offsets.flatten(), torch.tensor([n_isects], device=device)]
    )
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    total_pixels = V * image_height * image_width

    # Pre-allocate accumulator reused across loop iterations to avoid per-step allocation
    out = torch.zeros(N, dtype=opacities.dtype, device=device)

    # Loop over batches of Gaussians
    for step in range(0, num_batches, batch_per_iter):
        # Current transmittance
        transmittances = 1.0 - render_alphas[..., 0]

        gs_ids, image_ids, indices, pixel_ids, weights = get_m_intersection_weights(batch_per_iter, conics, flatten_ids,
                                                                                    image_height, image_width,
                                                                                    isect_offsets, means2d, opacities,
                                                                                    step, tile_size, total_pixels,
                                                                                    transmittances)

        # Sum weights over gaussian indices (reuse pre-allocated buffer)
        out.zero_()
        out.index_add_(0, gs_ids, weights)  # (N,)
        gaussian_weights_per_view[image_ids, gs_ids] += weights

        # Add to the global sum
        gaussian_weights += out

        # Accumulate alpha along rays
        alphas = accumulate_along_rays(
            weights, None, ray_indices=indices, n_rays=total_pixels
        )
        alphas = alphas.reshape(V, image_height, image_width, 1)

        render_alphas.add_(alphas * transmittances[..., None])

    return gaussian_weights, render_alphas, gaussian_weights_per_view


def get_m_intersection_weights(range_size, conics, flatten_ids, image_height, image_width, isect_offsets, means2d,
                               opacities, step, tile_size, total_pixels, transmittances):
    # Find the M intersections between pixels and gaussians.
    # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
    gs_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        step,
        step + range_size,
        transmittances,
        means2d,
        conics,
        opacities,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )  # [M], [M]
    # if len(gs_ids) == 0:
    #     break
    # Compute gaussian-pixel alpha values (reduced opacity due to gaussian intensity in 2D) -> (M,)
    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = pixel_coords - means2d[image_ids, gs_ids]  # [M, 2]
    c = conics[image_ids, gs_ids]  # [M, 3]
    sigmas = (
            0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
            + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    alphas = opacities[image_ids, gs_ids] * torch.exp(-sigmas)
    # alphas = torch.clamp_max(
    #     opacities[image_ids, gs_ids] * torch.exp(-sigmas), 0.999
    # )
    if (alphas > 1).any():
        warnings.warn(f"Not all alphas <= 1, max alpha: {alphas.max().item()}")
    # indices of the samples with shape (all_samples,)
    indices = image_ids * image_height * image_width + pixel_ids  # (M,)
    # `weights` is a flattened tensor with shape (all_samples,)
    weights, _ = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels
    )  # (M,)
    return gs_ids, image_ids, indices, pixel_ids, weights


@dataclass
class Base3DGSAttributeCfg(Generic[T]):
    _base: T
    _means: T
    _scales: T
    _opacities: T
    _quats: T
    _sh0: T
    _shN: T

    @property
    def base(self) -> T:
        return self._base

    @property
    def means(self) -> T:
        return self.base * self._means

    @property
    def scales(self) -> T:
        return self.base * self._scales

    @property
    def opacities(self) -> T:
        return self.base * self._opacities

    @property
    def quats(self) -> T:
        return self.base * self._quats

    @property
    def rotations(self) -> T:
        return self.quats

    @property
    def sh0(self) -> T:
        return self.base * self._sh0

    @property
    def shN(self) -> T:
        return self.base * self._shN

    @property
    def param_names(self) -> list[str]:
        return ['means', 'scales', 'quats', 'opacities', 'sh0', 'shN']

    def dict(self):
        return {name: getattr(self, name) for name in self.param_names}


@dataclass
class Bool3DGSCfg(Base3DGSAttributeCfg[bool]):
    # Config loading via dacite doesn't seem to support generic type, so need to write types explicitly
    _base: bool
    _means: bool
    _scales: bool
    _opacities: bool
    _quats: bool
    _sh0: bool
    _shN: bool

    def all_true(self):
        # return all attributes that are True
        return all([getattr(self, attr) for attr in self.param_names])

    def __str__(self):
        if self.all_true:
            return "all"
        else:
            return "_".join([f"{attr}" for attr in self.param_names if getattr(self, attr)])

class Number3DGSCfg(Base3DGSAttributeCfg[float | int]):
    # Config loading via dacite doesn't seem to support generic type, so need to write types explicitly
    _base: float | int
    _means: float | int
    _scales: float | int
    _opacities: float | int
    _quats: float | int
    _sh0: float | int
    _shN: float | int

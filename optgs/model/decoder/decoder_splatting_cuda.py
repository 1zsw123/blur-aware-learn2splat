from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from ...dataset import DatasetCfg
from ...scene_trainer.gaussian_module import GaussiansModule
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["inria"]
    scale_invariant: bool
    # False: pass scales+rotations and let the CUDA kernel compute the covariance
    # (matches 3DGS-LM byte-for-byte). True: precompute Python-side and pass
    # cov3D_precomp (~42 dB pixel drift from LM, slightly faster on repeat calls).
    use_covariances: bool = False


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians | GaussiansModule,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        return_radii: bool = False,
        iter_batch_size: int = -1,
        to_cpu: bool = False,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        bv = b * v

        # Flatten camera params to (B*V)
        flat_ext = rearrange(extrinsics, "b v i j -> (b v) i j")
        flat_int = rearrange(intrinsics, "b v i j -> (b v) i j")
        flat_near = rearrange(near, "b v -> (b v)")
        flat_far = rearrange(far, "b v -> (b v)")
        flat_bg = repeat(self.background_color, "c -> (b v) c", b=b, v=v)

        # Prepare Gaussian tensors in flat (B*V) format
        scales = rotations_wxyz = covars = None
        if isinstance(gaussians, GaussiansModule):
            means = repeat(gaussians.means, "g xyz -> bv g xyz", bv=bv)
            shs = repeat(gaussians.harmonics, "g c d -> bv g c d", bv=bv)
            opacities = repeat(gaussians.opacities, "g -> bv g", bv=bv)
            if self.cfg.use_covariances:
                covars = repeat(gaussians.covariances, "g i j -> bv g i j", bv=bv)
            else:
                scales = repeat(gaussians.scales, "g d -> bv g d", bv=bv)
                # gaussians.rotations is xyzw post-normalization; the rasterizer wants wxyz.
                rotations_wxyz = repeat(gaussians.rotations[:, [3, 0, 1, 2]], "g d -> bv g d", bv=bv)

        elif isinstance(gaussians, Gaussians):
            means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v)
            shs = repeat(gaussians.harmonics, "b g c d -> (b v) g c d", v=v)
            opacities = repeat(gaussians.opacities, "b g -> (b v) g", v=v)
            if self.cfg.use_covariances:
                if gaussians.covariances is None:
                    raise ValueError("use_covariances=true but gaussians.covariances is None.")
                covars = repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v)
            else:
                _scales = gaussians.scales if gaussians.stores_activated else torch.exp(gaussians.scales)
                scales = repeat(_scales, "b g d -> (b v) g d", v=v)
                rotations_wxyz = repeat(gaussians.rotations[..., [3, 0, 1, 2]], "b g d -> (b v) g d", v=v)
            if not gaussians.stores_activated:
                opacities = torch.sigmoid(opacities)
        else:
            raise ValueError(f"Unknown gaussians type: {type(gaussians)}")

        def _render_flat(s: slice):
            imgs, radii, means2d = render_cuda(
                flat_ext[s],
                flat_int[s],
                flat_near[s],
                flat_far[s],
                image_shape,
                flat_bg[s],
                means[s],
                covars[s] if covars is not None else None,
                shs[s],
                opacities[s],
                scale_invariant=self.cfg.scale_invariant,
                gaussian_scales=scales[s] if scales is not None else None,
                gaussian_rotations=rotations_wxyz[s] if rotations_wxyz is not None else None,
            )
            return imgs, radii, means2d

        if iter_batch_size < 0:
            imgs, radii_flat, means2d_flat = _render_flat(slice(None))
            if to_cpu:
                imgs = imgs.detach().cpu()
                radii_flat = radii_flat.detach().cpu()
                means2d_flat = means2d_flat.detach().cpu()
        else:
            all_imgs, all_radii, all_means2d = [], [], []
            for i in tqdm(range(0, bv, iter_batch_size), desc="Rendering in batches"):
                s = slice(i, min(i + iter_batch_size, bv))
                imgs_c, rad_c, m2d_c = _render_flat(s)
                if to_cpu:
                    imgs_c = imgs_c.detach().cpu()
                    rad_c = rad_c.detach().cpu()
                    m2d_c = m2d_c.detach().cpu()
                all_imgs.append(imgs_c)
                all_radii.append(rad_c)
                all_means2d.append(m2d_c)
            imgs = torch.cat(all_imgs, dim=0)
            radii_flat = torch.cat(all_radii, dim=0)
            means2d_flat = torch.cat(all_means2d, dim=0)

        # Reshape (B*V) → (B, V)
        color = rearrange(imgs, "(b v) c h w -> b v c h w", b=b, v=v)
        radii_bv = rearrange(radii_flat, "(b v) n -> b v n", b=b, v=v)
        means2d_bv = rearrange(means2d_flat, "(b v) n d -> b v n d", b=b, v=v)

        # Expand scalar radii [B, V, N] → [B, V, N, 2] to match gsplat interface
        radii_out = radii_bv.unsqueeze(-1).expand(-1, -1, -1, 2).contiguous()
        visibility_filter = radii_bv > 0  # [B, V, N]

        depth = (
            self._render_depth(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode)
            if depth_mode is not None
            else None
        )

        return DecoderOutput(
            color=color,
            depth=depth,
            accumulated_alpha=None,
            means2d=means2d_bv,
            radii=radii_out,
            visibility_filter=visibility_filter,
        )

    def _render_depth(
        self,
        gaussians: Gaussians | GaussiansModule,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape

        if isinstance(gaussians, GaussiansModule):
            means = repeat(gaussians.means, "g xyz -> (b v) g xyz", b=b, v=v)
            covars = repeat(gaussians.covariances, "g i j -> (b v) g i j", b=b, v=v)
            opacities = repeat(gaussians.opacities, "g -> (b v) g", b=b, v=v)
        else:
            means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v)
            covars = repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v)
            opacities = repeat(gaussians.opacities, "b g -> (b v) g", v=v)
            if not gaussians.stores_activated:
                opacities = torch.sigmoid(opacities)

        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            means,
            covars,
            opacities,
            mode=mode,
            scale_invariant=self.cfg.scale_invariant,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)

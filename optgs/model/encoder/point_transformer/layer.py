import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import pointops
from pointops import grouping, grouping2
from einops import rearrange
import time

from ..unimatch.dinov2.layers.block import Block as MultiViewBlock
from ..unimatch.utils import mv_feature_add_position
from ..unimatch.mv_transformer import MultiViewFeatureTransformer


USE_PYTORCH_ATTN = False
USE_FLASH_ATTN3 = False

# try:
#     from flash_attn_interface import flash_attn_func
#     FA3_AVAILABLE = True
#     warnings.warn('flash attention 3 is available (point attn)')

# except ImportError:
#     FA3_AVAILABLE = False
#     warnings.warn('flash attention 3 is not available (point attn)')


class KNNAttention(nn.Module):
    # TODO: multi-head
    def __init__(self, channels, knn_samples=16, no_rpe=True,
        qk_norm=False,
        num_heads=1,
        proj_channels=None,
        use_fused=False,
        ):
        super().__init__()
        self.proj_channels = proj_channels

        self.knn_samples = knn_samples
        self.no_rpe = no_rpe
        self.num_heads = num_heads
        assert self.num_heads == 1

        self.use_fused = use_fused
        if use_fused:
            try:
                import sys
                from optgs.paths import PROJECT_DIR
                sys.path.append(str(PROJECT_DIR / "submodules"))
                from fused_knn_attn import fused_knn_attention, FUSED_KNN_ATTN_CUDA_AVAILABLE
                self._fused_knn_attention = fused_knn_attention
                if not FUSED_KNN_ATTN_CUDA_AVAILABLE:
                    import warnings
                    warnings.warn(
                        "Fused KNN attention CUDA extension not available, "
                        "using PyTorch fallback (still avoids [N,K,C] intermediates)"
                    )
            except ImportError:
                import warnings
                warnings.warn(
                    "fused_knn_attn package not found, falling back to unfused attention"
                )
                self.use_fused = False

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(channels)
            self.k_norm = nn.RMSNorm(channels)

        if self.proj_channels is not None:
            self.qkv = nn.Linear(channels, self.proj_channels * 3, bias=False)
            self.proj = nn.Linear(self.proj_channels, channels)
        else:
            self.qkv = nn.Linear(channels, channels * 3, bias=False)
            self.proj = nn.Linear(channels, channels)

        if not self.no_rpe:
            self.rpe = nn.Sequential(
                nn.Linear(3, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )


    def forward(self, pxo, knn_idx=None):
        # [N, 3], [N, C], [B]
        p, x, o = pxo
        c = x.size(1)

        if self.proj_channels is not None:
            c = self.proj_channels

        assert c % self.num_heads == 0
        head_dim = c // self.num_heads
        scale_factor = head_dim ** -0.5

        qkv = self.qkv(x)  # [N, 3*C]
        x_q, x_k, x_v = torch.chunk(qkv, chunks=3, dim=-1)  # each [N, C]

        # ---- Fused path: gather + attention in one kernel ----
        if self.use_fused and self.no_rpe:
            # Ensure we have KNN indices
            if knn_idx is None:
                knn_idx, _ = pointops.knn_query(
                    self.knn_samples, p, o, p, o
                )

            # qk_norm: RMSNorm normalizes each C-dim vector independently,
            # so applying before gather is equivalent to applying after gather.
            if self.qk_norm:
                x_q = self.q_norm(x_q)
                x_k = self.k_norm(x_k)

            out = self._fused_knn_attention(
                x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
                knn_idx.contiguous(), scale_factor
            )
            out = self.proj(out)
            return out

        # ---- Original unfused path ----
        # # [N, K, C], [N, K]
        # x_k, idx = pointops.knn_query_and_group(
        #     x_k.contiguous(), p, o, new_xyz=p, new_offset=o,
        #     idx=knn_idx,
        #     nsample=self.knn_samples, with_xyz=False
        # )  # [N, K, C]
        #
        # # [N, K, C]
        # x_v, _ = pointops.knn_query_and_group(
        #     x_v.contiguous(),
        #     p,
        #     o,
        #     new_xyz=p,
        #     new_offset=o,
        #     idx=idx,
        #     nsample=self.knn_samples,
        #     with_xyz=False,
        # )

        # ---- Initial improved version ----
        x_kv = torch.cat([x_k, x_v], dim=-1) # [N, 2C/3]
        x_kv_query, _ = pointops.knn_query_and_group(
            x_kv.contiguous(), p, o, new_xyz=p, new_offset=o,
            idx=knn_idx, nsample=self.knn_samples, with_xyz=False
        ) # [N, K, 2C/3]
        x_k, x_v = torch.chunk(x_kv_query, chunks=2, dim=-1)

        # [N, K, 3], [N, K, C]
        # NOTE: without xyz in knn
        # p_r, x_k = x_k[:, :, :3], x_k[:, :, 3:]

        # [N, 1, K]
        assert self.no_rpe
        if not self.no_rpe:
            rpe = self.rpe(p_r).permute(0, 2, 1)
        else:
            rpe = 0

        if self.qk_norm:
            x_q = self.q_norm(x_q)
            x_k = self.k_norm(x_k)

        n, k, c = x_k.shape

        # attention
        if USE_PYTORCH_ATTN:
            out = F.scaled_dot_product_attention(
                x_q.view(n, 1, c),
                x_k.view(n, k, c),
                x_v.view(n, k, c),
                ).reshape(n, c)  # [N, C]

        elif (USE_FLASH_ATTN3 and FA3_AVAILABLE and self.no_rpe):
            # no relative pos enc
            out = flash_attn_func(
                x_q.view(n, 1, self.num_heads, head_dim).to(torch.bfloat16),
                x_k.view(n, k, self.num_heads, head_dim).to(torch.bfloat16),
                x_v.view(n, k, self.num_heads, head_dim).to(torch.bfloat16),
            )[0].reshape(n, c).float()  # [N, C]
        else:
            # [N, 1, K]
            scores = torch.matmul(x_q.unsqueeze(1), x_k.permute(0, 2, 1)) * scale_factor + rpe
            # [N, C]
            out = torch.matmul(torch.softmax(scores, dim=2), x_v).squeeze(1)

        out = self.proj(out)

        return out
    

class MLP(nn.Module):
    def __init__(
        self,
        channels,
        act="gelu",
    ):
        super().__init__()

        expansion = 4
        
        self.fc1 = nn.Linear(channels, channels * expansion)
        if act is None or act in ['none', 'identity']:
            self.act = nn.Identity()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"unsupported activation {act}")
        self.fc2 = nn.Linear(channels * expansion, channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, channels, knn_samples=16, post_norm=False,
        no_rpe=False,
        no_attn=False,
        no_norm=False,
        act="gelu",
        qk_norm=False,
        norm_pt_block=False,
        num_heads=1,
        attn_proj_channels=None,
        use_fused_attn=False,
        ):
        super().__init__()
        self.post_norm = post_norm
        self.no_attn = no_attn
        self.norm_pt_block = norm_pt_block

        if no_norm:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)

        if self.no_attn:
            self.linear = nn.Linear(channels, channels)
        else:
            self.attn = KNNAttention(channels, knn_samples=knn_samples, no_rpe=no_rpe,
            qk_norm=qk_norm,
            num_heads=num_heads,
            proj_channels=attn_proj_channels,
            use_fused=use_fused_attn,
            )
        self.mlp = MLP(channels, act=act)

        if self.norm_pt_block:
            self.norm3 = nn.LayerNorm(channels)

    def forward(self, pxo, knn_idx=None):
        p, x, o = pxo

        if self.post_norm:
            if self.no_attn:
                x = x + self.norm1(self.linear(x))
            else:
                x = x + self.norm1(self.attn((p, x, o), knn_idx=knn_idx))
            x = x + self.norm2(self.mlp(x))
        else:
            if self.no_attn:
                x = x + self.linear(self.norm1(x))
            else:
                x = x + self.attn((p, self.norm1(x), o), knn_idx=knn_idx)
            x = x + self.mlp(self.norm2(x))

        if self.norm_pt_block:
            x = self.norm3(x)

        return x


class FPSSubsample(nn.Module):
    def __init__(self, in_planes, out_planes, stride=4, nsample=16,
        agg_func='attn',
        subsample_method='fps',
        return_idx=False,
        fps_num_samples=None,
        attn_channels=64,
        ):
        super().__init__()

        assert stride > 0

        self.agg_func = agg_func
        self.subsample_method = subsample_method
        self.knn_samples = nsample
        self.return_idx = return_idx

        self.stride, self.nsample = stride, nsample

        if fps_num_samples is not None:
            self.nsample = fps_num_samples

        # if stride != 1:
        #     # xyz + feature
        #     # self.linear = nn.Linear(3 + in_planes, out_planes, bias=not post_norm)
        #     # only feature
        #     # TODO: attention aggregation
        #     if agg_func == 'maxpool':
        #         self.agg = nn.MaxPool1d(nsample)
        #     elif agg_func == 'avgpool':
        #         self.agg = nn.AvgPool1d(nsample)
        #     else:
        #         raise ValueError(f"unsupported agg_func {agg_func}")

        # fewer channels to save memory
        assert agg_func in ['attn', 'avgpool']
        if self.agg_func == 'attn':
            self.q = nn.Linear(in_planes, attn_channels, bias=False)
            self.k = nn.Linear(in_planes, attn_channels, bias=False)
            self.v = nn.Linear(in_planes, attn_channels, bias=False)
            
            self.proj = nn.Linear(attn_channels, out_planes, bias=True)
            self.residual = nn.Linear(in_planes, out_planes, bias=True)
        else:
            self.proj = nn.Linear(in_planes, out_planes, bias=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            if self.subsample_method == 'density':
                assert False  # not well tested
                n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
                for i in range(1, o.shape[0]):
                    count += (o[i].item() - o[i - 1].item()) // self.stride
                    n_o.append(count)
                n_o = torch.tensor(n_o, dtype=torch.int32, device=x.device)

                # [N, K, C+3]
                x_k, _ = pointops.knn_query_and_group(
                    x.contiguous(), p, o, new_xyz=p, new_offset=o, nsample=self.knn_samples, with_xyz=True
                )

                p_r = x_k[:, :, 0:3]
                density = torch.mean(torch.norm(p_r, dim=-1), dim=-1)  # [N]

                # TODO: normalize the distance
                weights = (density - density.min()) / (density.max() - density.min() + 1e-6)
                # weights = density

                # weights = 1.0 / (density + 1e-6)  # Inverse density weighting

                # to batch
                lists = [weights[:o[0]]]
                for i in range(o.shape[0] - 1):
                    lists.append(weights[o[i]:o[i+1]])

                weights = torch.stack(lists, dim=0)  # [B, N]

                weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights

                # Sample points based on weights
                batch = n_o.shape[0]
                num_samples = o[0].item() // self.stride
                sampled_indices = torch.stack([
                    torch.multinomial(weights[b], num_samples, replacement=False)
                    for b in range(batch)
                ], dim=0)  # (B, num_samples)

                idx = rearrange(sampled_indices, "b n -> (b n)")

                point_list = [p[:o[0]]]
                for i in range(o.shape[0] - 1):
                    point_list.append(p[o[i]:o[i+1], :])

                points = torch.stack(point_list, dim=0)  # [B, N, 3]

                # Gather sampled points
                sampled_points = torch.gather(points, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, 3))

                # print(sampled_points.shape)  # [B, M, 3]

                sampled_points = rearrange(sampled_points, "b m c -> (b m) c")

                # average pooling
                # TODO: try others
                x = x_k.mean(dim=1)  # [N, C]
                x_list = [x[:o[0]]]
                for i in range(o.shape[0] - 1):
                    x_list.append(x[o[i]:o[i+1], :])
                x = torch.stack(x_list, dim=0)  # [B, N, C]

                # Gather sampled points
                x = torch.gather(x, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
                x = rearrange(x, "b n c -> (b n) c")

                # TODO: do we need to add residual to x here?
                # use the index to subsample the initial features
                x = self.proj(x)

                p, o = sampled_points, n_o
            elif self.subsample_method in ['fps', 'grid']:
                n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
                for i in range(1, o.shape[0]):
                    count += (o[i].item() - o[i - 1].item()) // self.stride
                    n_o.append(count)
                n_o = torch.tensor(n_o, dtype=torch.int32, device=x.device)

                if self.subsample_method == 'fps':
                    idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
                else:
                    # uniform sampling: sanity check
                    # first reshape to V, H, W, then do grid sampling
                    # Generate grid indices
                    # TODO: grid sample in the image space
                    idx = torch.arange(0, p.size(0), self.stride).to(x.device)

                n_p = p[idx.long(), :]  # (m, 3)
                x_subsample = x[idx.long(), :]  # [M, C]
                if self.agg_func == 'attn':
                    x_q = self.q(x_subsample)  # [M, C]
                    # [M, K, C]
                    x_k = self.k(x)  # [N, C]
                else:
                    x_k = x

                x_k, knn_idx = pointops.knn_query_and_group(
                    x_k,
                    p,
                    offset=o,
                    new_xyz=n_p,
                    new_offset=n_o,
                    nsample=self.nsample,
                    with_xyz=False,  # remove xyz
                )

                if self.agg_func == 'attn':
                    x_v = self.v(x)
                    x_v, _ = pointops.knn_query_and_group(
                        x_v,
                        p,
                        offset=o,
                        new_xyz=n_p,
                        new_offset=n_o,
                        idx=knn_idx,
                        nsample=self.nsample,
                        with_xyz=False,  # remove xyz
                    )

                    # attention
                    # x_q: [M, C], x_k: [M, K, C], x_v: [M, K, C]
                    scale_factor = x_q.shape[-1] ** -0.5

                    # [M, 1, K]
                    # no relative posenc
                    scores = torch.matmul(x_q.unsqueeze(1), x_k.permute(0, 2, 1)) * scale_factor
                    # [M, C]
                    x = torch.matmul(torch.softmax(scores, dim=2), x_v).squeeze(1)

                    # if self.agg_func in ['avgpool', 'maxpool']:
                    #     x = self.agg(x.transpose(1, 2).contiguous()).squeeze(-1)  # (m, c)
                    # else:
                    #     raise NotImplementedError

                    # add residual to x here?
                    # use the index to subsample the initial features
                    x = self.residual(x_subsample) + self.proj(x)
                else:
                    x = x_k.mean(dim=1)
                    x = self.proj(x)

                p, o = n_p, n_o

            else:
                raise ValueError(f"unsupported subsampling method {self.subsample_method}")
        else:
            # add residual to x here?
            x = x + self.proj(x)
                
            idx = torch.arange(0, p.size(0)).to(x.device)

        if self.return_idx:
            return [p, x, o], idx
        return [p, x, o]


class SubsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=4, knn_samples=16, post_norm=False,
        agg_func='attn',
        subsample_method='fps',
        return_idx=False,
        fps_num_samples=None,
        attn_proj_channels=None,
        ):
        super().__init__()

        assert not post_norm

        self.return_idx = return_idx

        self.post_norm = post_norm
        self.norm1 = nn.LayerNorm(in_channels)
        self.fps = FPSSubsample(in_channels, out_channels, stride=stride, nsample=knn_samples,
            agg_func=agg_func,
            subsample_method=subsample_method,
            return_idx=return_idx,
            fps_num_samples=fps_num_samples,
            attn_channels=attn_proj_channels,
            )

        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = MLP(out_channels)

    def forward(self, pxo):

        # pre norm
        p, x, o = pxo
        x = self.norm1(x)

        if self.return_idx:
            pxo, idx = self.fps([p, x, o])
        else:
            pxo = self.fps([p, x, o])

        p, x, o = pxo

        x = x + self.mlp(self.norm2(x))

        if self.return_idx:
            return [p, x, o], idx

        return [p, x, o]


class SkipConnect(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj1 = nn.Linear(out_channels, out_channels)
        self.proj2 = nn.Linear(in_channels, out_channels)
        self.proj3 = nn.Linear(out_channels, out_channels)

    def forward(self, pxo1, pxo2):
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2

        # TODO: support half precision
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
            x = self.proj1(x1) + pointops.interpolation2(
                    p2, p1, self.proj2(x2), o2, o1
                )
        
        x = self.proj3(x)

        return x



class PlainPointTransformer(nn.Module):
    def __init__(self, channels, knn_samples=16, num_blocks=4, post_norm=False,
        no_rpe=False,
        no_attn=False,
        no_norm=False,
        act="gelu",
        qk_norm=False,
        norm_pt_block=False,
        num_heads=1,
        attn_proj_channels=None,
        cache_knn_idx=None,
        knn_idx_update_every=1,
        with_mv_attn=False,
        with_mv_attn_lowres=False,
        mv_attn_first=False,
        no_mv_attn=False,
        conv_with_norm=False,
        mv_shuffle_attn=False,
        with_pos_enc=False,
        shuffle_attn_no_norm=False,
        mv_unimatch_attn=False,
        use_checkpointing=False,
        init_use_checkpointing=False,
        use_fused_attn=False,
        ):
        super().__init__()

        self.cache_knn_idx = cache_knn_idx
        self.knn_idx_update_every = knn_idx_update_every
        self.knn_samples = knn_samples
        self.use_checkpointing = use_checkpointing
        self.init_use_checkpointing = init_use_checkpointing

        self.with_mv_attn = with_mv_attn
        self.with_mv_attn_lowres = with_mv_attn_lowres
        if with_pos_enc:
            assert mv_shuffle_attn

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(TransformerBlock(channels, knn_samples=knn_samples,
                post_norm=post_norm,
                no_rpe=no_rpe,
                no_attn=no_attn,
                no_norm=no_norm,
                act=act,
                qk_norm=qk_norm,
                norm_pt_block=norm_pt_block,
                num_heads=num_heads,
                attn_proj_channels=attn_proj_channels,
                use_fused_attn=use_fused_attn,
                ))

        # multi-view attention
        if self.with_mv_attn:
            self.mv_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                # if mv_shuffle_attn:
                if self.with_mv_attn_lowres:
                    self.mv_blocks.append(
                        MultViewLowresAttn(
                            channels,
                        )
                    )
                else:
                    self.mv_blocks.append(
                        MultiViewBlock(
                            channels,
                            num_heads=4,
                        )
                    )
                # elif mv_unimatch_attn:
                #     self.mv_blocks.append(
                #         MultViewUniMatchAttn(
                #             channels,
                #         )
                #     )
                # else:
                #     self.mv_blocks.append(
                #         MultViewUnetAttn(channels,
                #         no_mv_attn=no_mv_attn,
                #         conv_with_norm=conv_with_norm,
                #         )
                #     )

    def forward(self, pxo, iter=0, b=None, v=None, h=None, w=None):
        p, x, o = pxo
        # compute knn idx here only once and pass it to the model
        # the positions are not changed inside the blocks
        if self.cache_knn_idx is None or (iter % self.knn_idx_update_every) == 0:
            knn_idx, _ = pointops.knn_query(self.knn_samples, p, o, p, o)
            self.cache_knn_idx = knn_idx
            # print(knn_idx.float().mean().item())
        else:
            knn_idx = self.cache_knn_idx

        if self.with_mv_attn:
            assert b is not None and v is not None and h is not None and w is not None
            if self.use_checkpointing:
                raise NotImplementedError

            for i in range(len(self.blocks)):
                # knn attention
                x = self.blocks[i]([p, x, o], knn_idx=knn_idx)
                # global multi-view attention
                x = rearrange(x, "(b v h w) c -> b (v h w) c", b=b, v=v, h=h, w=w)
                if self.with_mv_attn_lowres:
                    x = self.mv_blocks[i](x, v=v, h=h, w=w)
                    # # TODO: hard-coded for now
                    # if x.size(1) == 8 * 512 // 4 * 960 // 4:
                    #     x = self.mv_blocks[i](x, v=8, h=512 // 4, w=960 // 4)
                    # elif x.size(1) == 8 * 256 // 4 * 448 // 4:
                    #     x = self.mv_blocks[i](x, v=8, h=256 // 4, w=448 // 4)
                    # else:
                    #     raise ValueError(f"unsupported input size {x.size(1)} for multi-view attention")
                    # # print(x.shape)
                else:
                    x = self.mv_blocks[i](x)
                # x = x.squeeze(0)
                x = rearrange(x, "b (v h w) c -> (b v h w) c",
                    b=b, v=v, h=h, w=w)
        else:
            for blk in self.blocks:
                if self.init_use_checkpointing:
                    # checkpointing the inital reconstruction model
                    # NOTE: cannot cache knn_idx here, otherwise index out error
                    def custom_forward(p, x, o):
                        return blk((p, x, o), knn_idx=None)  # knn_idx is closed over
                    x = torch.utils.checkpoint.checkpoint(custom_forward, p, x, o)
                else:
                    x = blk((p, x, o), knn_idx=knn_idx)

        return x


class MultViewUnetAttn(nn.Module):
    def __init__(self, channels, no_mv_attn=False, conv_with_norm=False):
        super().__init__()

        self.conv_with_norm = conv_with_norm

        self.down1 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down2 = nn.Conv2d(channels, channels, 3, 2, 1)

        self.up2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.up1 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.attn = MultiViewBlock(channels, 4, no_attn=no_mv_attn)

        if self.conv_with_norm:
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
            self.norm3 = nn.LayerNorm(channels)
            self.norm4 = nn.LayerNorm(channels)
    
    def forward(self, x):
        v = 8
        h = 256 // 4
        w = 448 // 4
        b = 1
        assert x.size(0) == b * v * h * w
        residual = x
        x = rearrange(x, "(b v h w) c -> (b v) c h w", b=b, v=v, h=h, w=w)
        x1 = self.down1(x)  # 1/2
        if self.conv_with_norm:
            x1 = self.norm1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = self.down2(x1)  # 1/4
        if self.conv_with_norm:
            x2 = self.norm2(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = rearrange(x2, "(b v) c h w -> b (v h w) c", b=b, v=v)
        x2 = self.attn(x2)  # 1/4
        x2 = rearrange(x2, "b (v h w) c -> (b v) c h w", b=b, v=v, h=h//4, w=w//4)
        x2 = self.up2(x1 + F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True))  # 1/2
        if self.conv_with_norm:
            x2 = self.norm3(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.up1(x + F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True))  # 1
        if self.conv_with_norm:
            x = self.norm4(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = rearrange(x, "(b v) c h w -> (b v h w) c", b=b, v=v)

        x = residual + x

        return x


class MultViewShuffleAttn(nn.Module):
    def __init__(self, channels, no_mv_attn=False, with_pos_enc=False, shuffle_attn_no_norm=False):
        super().__init__()

        self.down_factor = 4
        self.with_pos_enc = with_pos_enc

        self.proj1 = nn.Linear(channels * self.down_factor ** 2, channels)
        if shuffle_attn_no_norm:
            self.norm1 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(channels)

        self.proj2 = nn.Linear(channels, channels * self.down_factor ** 2)

        if shuffle_attn_no_norm:
            self.norm2 = nn.Identity()
        else:
            self.norm2 = nn.LayerNorm(channels * self.down_factor ** 2)

        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

        if no_mv_attn:
            self.attn = nn.Identity()
        else:
            self.attn = MultiViewBlock(channels, 4, no_attn=no_mv_attn)

    def forward(self, x):
        v = 8
        h = 256 // 4
        w = 448 // 4
        b = 1
        assert x.size(0) == b * v * h * w
        residual = x
        x = rearrange(x, "(b v h w) c -> (b v) c h w", b=b, v=v, h=h, w=w)

        # TODO: add positional encoding to x
        if self.with_pos_enc:
            x = mv_feature_add_position(x, attn_splits=1, feature_channels=x.size(1))
            # print(x.shape)

        x = F.pixel_unshuffle(x, self.down_factor)

        x = rearrange(x, "(b v) c h w -> b (v h w) c", b=b)
        x = self.proj1(x)
        x = self.norm1(x)

        x = self.attn(x)

        x = self.proj2(x)
        x = self.norm2(x)

        x = rearrange(x, "b (v h w) c -> (b v) c h w", b=b, v=v, h=h // self.down_factor, w=w // self.down_factor)
        x = F.pixel_shuffle(x, self.down_factor)
        x = self.conv(x)
        x = rearrange(x, "(b v) c h w -> (b v h w) c", b=b, v=v)
        x = x + residual

        return x


class MultViewLowresAttn(nn.Module):
    def __init__(self, channels, no_mv_attn=False, with_pos_enc=False, shuffle_attn_no_norm=False,
        down_factor=4,
        attn_proj_channels=None,
        ):
        super().__init__()

        self.down_factor = down_factor
        self.with_pos_enc = with_pos_enc

        self.attn_proj_channels = attn_proj_channels

        if attn_proj_channels:
            ori_channels = channels
            self.proj0 = nn.Linear(channels, attn_proj_channels)
            channels = attn_proj_channels

        if self.down_factor == 8:
            down_factor = 4
        else:
            down_factor = self.down_factor

        self.proj1 = nn.Linear(channels * down_factor ** 2, channels)
        if shuffle_attn_no_norm:
            self.norm1 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(channels)

        self.proj2 = nn.Linear(channels, channels * down_factor ** 2)

        if shuffle_attn_no_norm:
            self.norm2 = nn.Identity()
        else:
            self.norm2 = nn.LayerNorm(channels * down_factor ** 2)

        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

        if attn_proj_channels:
            self.proj3 = nn.Linear(channels, ori_channels)

        if no_mv_attn:
            self.attn = nn.Identity()
        else:
            num_heads = 1 if self.attn_proj_channels else 4
            self.attn = MultiViewBlock(channels, num_heads, no_attn=no_mv_attn)

    def forward(self, x, v=None, h=None, w=None, y=None):
        if y is not None:
            return self.forward_cross_attn(x, y, v, h, w)
        residual = x
        if self.attn_proj_channels:
            x = self.proj0(x)

        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)

        # TODO: add positional encoding to x
        if self.with_pos_enc:
            x = mv_feature_add_position(x, attn_splits=1, feature_channels=x.size(1))
            # print(x.shape)

        if self.down_factor == 8:
            # bilinear to half first to save channels
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
            down_factor = 4
        else:
            down_factor = self.down_factor

        x = F.pixel_unshuffle(x, down_factor)

        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        x = self.proj1(x)
        x = self.norm1(x)

        x = self.attn(x)

        x = self.proj2(x)
        x = self.norm2(x)

        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h // self.down_factor, w=w // self.down_factor)
        x = F.pixel_shuffle(x, down_factor)
        x = self.conv(x)
        if self.down_factor == 8:
            # bilinear to full
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        if self.attn_proj_channels:
            x = self.proj3(x)
        x = x + residual

        return x

    def forward_cross_attn(self, x, y, v=None, h=None, w=None):
        residual = x
        if self.attn_proj_channels:
            x = self.proj0(x)

        assert y is not None
        y = rearrange(y, "b (v h w) c -> (b v) c h w", h=h, w=w)  # different v with x
        num_cross_view = y.shape[0] // x.shape[0]

        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)

        # TODO: add positional encoding to x
        if self.with_pos_enc:
            x = mv_feature_add_position(x, attn_splits=1, feature_channels=x.size(1))
            # print(x.shape)

        if self.down_factor == 8:
            # bilinear to half first to save channels
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
            y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
            down_factor = 4
        else:
            down_factor = self.down_factor

        x = F.pixel_unshuffle(x, down_factor)
        y = F.pixel_unshuffle(y, down_factor)

        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        y = rearrange(y, "(b v) c h w -> b (v h w) c", v=num_cross_view)
        x = self.proj1(x)
        x = self.norm1(x)

        y = self.proj1(y)
        y = self.norm1(y)

        # x_tmp = self.attn(x)

        # print((x - y).abs().max().item())

        x = self.attn(x, y)

        # there will be slight diff for self and cross attn caused by flash3
        # print((x_tmp - x).abs().max().item())

        x = self.proj2(x)
        x = self.norm2(x)

        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h // self.down_factor, w=w // self.down_factor)
        x = F.pixel_shuffle(x, down_factor)
        x = self.conv(x)
        if self.down_factor == 8:
            # bilinear to full
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        if self.attn_proj_channels:
            x = self.proj3(x)
        x = x + residual

        return x



class GaussianErrorCrossAttn(nn.Module):
    def __init__(self, gaussian_channels,
        error_channels, 
        model_channels=256,
        no_mv_attn=False, with_pos_enc=False, shuffle_attn_no_norm=False,
        down_factor=4,
        attn_proj_channels=None,
        num_heads=4,
        with_mlp=False,
        ):
        super().__init__()

        self.num_heads = num_heads
        self.model_channels = model_channels
        self.down_factor = down_factor
        self.with_mlp = with_mlp

        # self.q_norm = nn.LayerNorm(gaussian_channels)
        self.q_proj = nn.Linear(gaussian_channels, model_channels)

        kv_channels = error_channels * (down_factor ** 2)
        # self.kv_norm = nn.LayerNorm(kv_channels)
        self.kv_proj = nn.Linear(kv_channels, 2 * model_channels)

        # self.out_proj = nn.Linear(model_channels, gaussian_channels)
        # concat
        self.out_proj = nn.Linear(model_channels + gaussian_channels, gaussian_channels)

        if with_mlp:
            self.mlp_norm = nn.LayerNorm(gaussian_channels)
            self.mlp = MLP(gaussian_channels)


    def forward(self, gaussian, error, v=None, h=None, w=None, mask=None):
        # [B, VHW, C]
        residual = gaussian
        b = gaussian.size(0)

        # x = self.q_norm(gaussian)
        x = gaussian
        q = self.q_proj(x)  # [B, VHW, C]

        # spatial reshape to save computation
        error = rearrange(error, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)
        error = F.pixel_unshuffle(error, self.down_factor)
        error = rearrange(error, "(b v) c h w -> b (v h w) c", v=v)
        # error = self.kv_norm(error)

        kv = self.kv_proj(error)
        k, v = kv.chunk(2, dim=-1)  # [B, VHW, C]

        # attention
        c = self.model_channels
        head_dim = c // self.num_heads

        # [B, N, C] → [B, num_heads, N, head_dim]
        def reshape(x):
            return x.view(b, -1, self.num_heads, head_dim).transpose(1, 2)  # [B, H, N, D]

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        # Fast fused attention
        out = F.scaled_dot_product_attention(q, k, v)

        # [B, H, N, D] → [B, N, C]
        out = out.transpose(1, 2).contiguous().view(b, -1, c)

        # return self.out_proj(out)

        # out = residual + self.out_proj(out)
        # concat
        out = self.out_proj(torch.cat([out, gaussian], dim=-1))

        # if self.with_mlp:
        #     out = out + self.mlp(self.mlp_norm(out))

        return out




class MultViewUniMatchAttn(nn.Module):
    def __init__(self, channels, no_mv_attn=False, with_pos_enc=False, shuffle_attn_no_norm=False):
        super().__init__()

        self.attn = MultiViewFeatureTransformer(num_layers=1,
            d_model=channels,
            )

    def forward(self, x, v=None, h=None, w=None):
        residual = x
        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)

        attn_splits = 4

        # add pos enc
        x = mv_feature_add_position(x, attn_splits, feature_channels=x.size(1))
        x = rearrange(x, "(b v) c h w -> b v c h w", v=v)

        x_list = list(torch.unbind(x, dim=1))

        x_list = self.attn(x_list, attn_splits)

        x = torch.stack(x_list, dim=1)

        x = rearrange(x, "b v c h w -> b (v h w) c")

        return x



class MultiScalePointTransformer(nn.Module):
    def __init__(self, channels, knn_samples=16, post_norm=False,
        no_rpe=True,
        no_attn=False,
        qk_norm=False,
        norm_pt_block=False,
        num_heads=1,
        num_scales=3,
        stride=4,
        downsample_agg_func='attn',
        subsample_method='fps',
        fps_num_samples=None,
        attn_proj_channels=None,
        ):
        super().__init__()

        self.blocks = nn.ModuleList()
        # knn 4 at 1
        self.blocks.append(TransformerBlock(channels, knn_samples=4, 
            post_norm=post_norm,
            no_rpe=no_rpe,
            no_attn=no_attn,
            qk_norm=qk_norm,
            norm_pt_block=norm_pt_block,
            num_heads=num_heads,
            attn_proj_channels=attn_proj_channels,
            ))

        for i in range(num_scales - 2, -1, -1):
            # knn 8 at 1/4
            # knn 16 at 1/16
            self.blocks.append(TransformerBlock(channels * (2 ** i), knn_samples= knn_samples // (2 ** i), 
                post_norm=post_norm,
                no_rpe=no_rpe,
                no_attn=no_attn,
                qk_norm=qk_norm,
                norm_pt_block=norm_pt_block,
                num_heads=num_heads,
                attn_proj_channels=attn_proj_channels,
                ))

        self.down_blocks = nn.ModuleList()
        for i in range(num_scales - 1):
            self.down_blocks.append(
                    SubsampleBlock(
                        channels * (2 ** i), channels * (2 ** (i + 1)), 
                        stride=stride,
                        knn_samples=knn_samples // (2 ** (num_scales - 1 - i)),
                        subsample_method=subsample_method,
                        agg_func=downsample_agg_func,
                        fps_num_samples=fps_num_samples,
                        attn_proj_channels=attn_proj_channels,
                    )
            )

        self.down_agg = nn.ModuleList()
        for i in range(num_scales - 1):
            self.down_agg.append(
                TransformerBlock(channels * (2 ** (i + 1)), knn_samples=knn_samples // (2 ** (num_scales - 1 - i)), 
                                post_norm=post_norm,
                                no_rpe=no_rpe,
                                no_attn=no_attn,
                                qk_norm=qk_norm,
                                norm_pt_block=norm_pt_block,
                                num_heads=num_heads,
                                attn_proj_channels=attn_proj_channels,
                                )
            )

        self.skip_blocks = nn.ModuleList()
        for i in range(num_scales - 1, 0, -1):
            self.skip_blocks.append(
                SkipConnect(
                    channels * (2 ** i),
                    channels * (2 ** (i - 1))
                )
            )
        
    def forward(self, pxo):
        x1 = self.blocks[0](pxo)  # 1
        p1, o1 = pxo[0], pxo[2]
        p2, x2, o2 = self.down_blocks[0]([p1, x1, o1])  # 1/4
        x2 = self.down_agg[0]([p2, x2, o2])  # 1/4
        p3, x3, o3 = self.down_blocks[1]([p2, x2, o2])  # 1/16
        x3 = self.down_agg[1]([p3, x3, o3])  # 1/16

        x4 = self.skip_blocks[0]([p2, x2, o2], [p3, x3, o3])  # 1/4
        p4, o4 = p2, o2
        x4 = self.blocks[1]([p4, x4, o4])
        x5 = self.skip_blocks[1]([p1, x1, o1], [p4, x4, o4])  # 1
        p5, o5 = p1, o1
        x5 = self.blocks[2]([p5, x5, o5])

        return x5


class PointLinearWrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, pxo, b=None, v=None, h=None, w=None):
        p, x, o = pxo
        x = self.linear(x)

        return [p, x, o]


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


def test_fps():
    model = FPSSubsample(256, 256,
    fps_num_samples=16,
    subsample_method='fps',
    ).cuda()
    print(model)

    # FPS is significantly slower than grid with many points

    c = 256
    b, n = 2, 40480

    x = torch.randn(b, n, c).cuda()
    offset = torch.tensor([n * (i + 1) for i in range(b)]).to(x.device)
    p = torch.randn(b, n, 3).cuda()
    pxo = [p.view(-1, 3), x.view(-1, c), offset]
    y = model(pxo)
    print(y[1].shape)

    count = 100

    for _ in range(5):
        model(pxo)

    torch.cuda.synchronize()
    start = time.time()

    for i in range(count):
        model(pxo)

    torch.cuda.synchronize()
    print(time.time() - start)

def test_knn_query_and_group():
    c = 256
    # b, n = 2, 80480
    b, n = 8, 57344
    knn_samples = 16

    x = torch.randn(b, n, c).cuda()
    offset = torch.tensor([n * (i + 1) for i in range(b)]).to(x.device)
    o = offset
    p = torch.randn(b, n, 3).cuda()
    p = p.view(-1, 3)

    knn_idx, _ = pointops.knn_query(knn_samples, p, o, p, o)

    print(knn_idx.shape)

    c_qkv = 192
    qkv = torch.randn(b*n, c_qkv).cuda()
    T = 1000

    # chunk first, then query twice
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(T):
        x_q, x_k, x_v = torch.chunk(qkv, chunks=3, dim=-1)
        x_k_query, idx = pointops.knn_query_and_group(
            x_k.contiguous(), p, o, new_xyz=p, new_offset=o,
            idx=knn_idx,
            nsample=knn_samples, with_xyz=False
        )  # [N, K, C/3]
        x_v_query, _ = pointops.knn_query_and_group(
            x_v.contiguous(),
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=knn_samples,
            with_xyz=False,
        )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"KNN query and group time: {(end_time - start_time) / T * 1000:.2f} ms")

    # query first, then chunk
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(T):
        x_qkv_query = pointops.knn_query_and_group(
            qkv.contiguous(), p, o, new_xyz=p, new_offset=o,
            idx=knn_idx,
            nsample=knn_samples, with_xyz=False
        )[0]  # [N, K, C*3]
        x_q, x_k, x_v = torch.chunk(x_qkv_query, chunks=3, dim=-1)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"KNN query and group time: {(end_time - start_time) / T * 1000:.2f} ms")

    # chunk first, then query once
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(T):
        x_q, x_k, x_v = torch.chunk(qkv, chunks=3, dim=-1)
        x_kv = torch.cat([x_k, x_v], dim=-1)          # [N, 2C/3]
        x_kv_query = pointops.knn_query_and_group(
            x_kv.contiguous(), p, o, new_xyz=p, new_offset=o,
            idx=knn_idx, nsample=knn_samples, with_xyz=False
        )[0]                                        # [N, K, 2C/3]
        x_k_query, x_v_query = torch.chunk(x_kv_query, 2, dim=-1)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"KNN query and group time: {(end_time - start_time) / T * 1000:.2f} ms")

def test_knn():
    c = 256
    b, n = 2, 80480
    model = KNNAttention(channels=c,
    # proj_feature=64,
    ).cuda()
    print(model)

    x = torch.randn(b, n, c).cuda()
    offset = torch.tensor([n * (i + 1) for i in range(b)]).to(x.device)
    p = torch.randn(b, n, 3).cuda()
    pxo = [p.view(-1, 3), x.view(-1, c), offset]
    y = model(pxo)
    print(y.shape)

    count = 100

    for _ in range(5):
        model(pxo)

    torch.cuda.synchronize()
    start = time.time()

    for i in range(count):
        model(pxo)

    torch.cuda.synchronize()
    print(time.time() - start)


def test_faiss_knn():
    # cannot install faiss unfortunately
    # TODO: maybe implement a sliding window knn search later
    c = 256
    b, n = 2, 80480
    knn_samples = 16

    x = torch.randn(b, n, c).cuda()
    offset = torch.tensor([n * (i + 1) for i in range(b)]).to(x.device)
    o = offset
    p = torch.randn(b, n, 3).cuda()
    p = p.view(-1, 3)
    # pxo = [p.view(-1, 3), x.view(-1, c), offset]

    # print(p.shape, o.shape)
    # print(o)

    knn_idx, _ = pointops.knn_query(knn_samples, p, o, p, o)

    print(knn_idx.shape)

    count = 100

    for _ in range(5):
        pointops.knn_query(knn_samples, p, o, p, o)

    torch.cuda.synchronize()
    start = time.time()

    for i in range(count):
        pointops.knn_query(knn_samples, p, o, p, o)

    torch.cuda.synchronize()
    print(time.time() - start)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_mlp():
    b, n, c = 2, 40240, 256
    model = MLP(c).cuda()
    x = torch.randn(b, n, c).cuda()

    # model = SwiGLUFFN(c, c * 3).cuda()

    print('parameters:', count_parameters(model))

    x = x.to(torch.bfloat16)
    model.to(dtype=torch.bfloat16)

    with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        y = model(x)
    print(y.shape)

    count = 100

    for _ in range(5):
        model(x)

    torch.cuda.synchronize()
    start = time.time()

    for i in range(count):
        model(x)

    torch.cuda.synchronize()
    print(time.time() - start)


def test_mv_block():
    c = 256
    num_heads = 4
    model = MultiViewBlock(c, num_heads).cuda()
    x = torch.rand(2, 256, c).cuda()

    print(model)

    y = model(x)

    print(y.shape)


def test_cross_attn():
    c = 256
    v, h, w = 8, 64, 128
    num_heads = 4
    model = GaussianErrorCrossAttn(512, c, c).cuda()
    x = torch.rand(2, v * h * w, 512).cuda()
    y = torch.rand(2, v * h * w, c).cuda()

    print(model)

    y = model(x, y, v=v, h=h, w=w)

    print(x.shape, y.shape)


def test_grouping():
    c = 256
    # b, n = 2, 80480
    b, n = 1, 57344
    knn_samples = 16

    x = torch.randn(b, n, c).cuda()
    offset = torch.tensor([n * (i + 1) for i in range(b)]).to(x.device)
    o = offset
    p = torch.randn(b, n, 3).cuda()
    p = p.view(-1, 3)

    knn_idx, _ = pointops.knn_query(knn_samples, p, o, p, o)

    print(knn_idx.shape)

    c_qkv = 192
    qkv = torch.randn(b*n, c_qkv).cuda()
    x_q, x_k, x_v = torch.chunk(qkv, chunks=3, dim=-1)
    x_kv = torch.cat([x_k, x_v], dim=-1)          # [N, 2C/3]

    m, nsample, c = knn_idx.shape[0], knn_idx.shape[1], x_kv.shape[1]
    feat = torch.cat([x_kv, torch.zeros([1, c]).to(x_kv.device)], dim=0)
    T = 1000

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(T):
        grouping(idx=knn_idx, feat=x_kv, xyz=p, new_xyz=p, with_xyz=False)
        # grouping_idx = feat[knn_idx.view(-1).long(), :].view(
        #     m, nsample, c
        # )  # (m, num_sample, c)
    torch.cuda.synchronize()
    end_time = time.time()
    # print(f"Grouping via indexing: {(end_time - start_time) / T * 1000:.2f} ms")
    print(f"grouping pytorch: {(end_time - start_time) / T * 1000:.2f} ms")

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(T):
        grouping2(x_kv, knn_idx)
        # grouping_embed = torch.nn.functional.embedding(knn_idx, feat)  # [m,num_sample,c]
    torch.cuda.synchronize()
    end_time = time.time()
    # print(f"Grouping via embedding: {(end_time - start_time) / T * 1000:.2f} ms")
    print(f"grouping cuda: {(end_time - start_time) / T * 1000:.2f} ms")

if __name__ == '__main__':
    # test_fps()
    # test_knn()
    # test_mlp()
    # test_mv_block()
    # test_cross_attn()
    # test_faiss_knn()
    # test_knn_query_and_group()
    test_grouping()


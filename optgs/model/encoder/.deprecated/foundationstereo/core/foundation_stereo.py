# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,logging
import torch.nn as nn
import torch.nn.functional as F
import sys,os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.update import *
from core.extractor import *
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.utils.utils import *
from Utils import *
import time
import cv2

from .unimatch_matching import group_correlation_softmax_depth, CorrBlock, coords_grid, warp_with_pose_depth_candidates
from .unimatch_geometry import compute_flow_with_depth_pose


count = 0


try:
    # autocast = torch.cuda.amp.autocast
    autocast = torch.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()


class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
        self.atts = nn.ModuleDict({
          "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp']//16),
        })
        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )

        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)
        x = self.conv_patch(x)
        x = self.atts["4"](x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
        conv = conv + x
        conv = self.conv_out(conv)

        return conv



class FoundationStereo(nn.Module):
    def __init__(self, args,
        bilinear_init_depth=False,
        flow_corr=False,
        flow_corr_levels=4,
        amp_bf16=True,
        vit_type='vitl',
        no_geo_volume=False,
        concat_geo_volume=False,
        depth_sample_geo_volume=False,
        no_freeze_mono=False,
        bilinear_up_depth=False,
        local_match_radius=0,
        supervise_init_depth=False,
        sample_log_depth=False,
        ):
        super().__init__()
        self.args = args

        self.bilinear_init_depth = bilinear_init_depth
        self.flow_corr = flow_corr
        self.flow_corr_levels = flow_corr_levels
        self.amp_bf16 = amp_bf16
        self.no_geo_volume = no_geo_volume
        self.concat_geo_volume = concat_geo_volume
        self.depth_sample_geo_volume = depth_sample_geo_volume
        self.bilinear_up_depth = bilinear_up_depth
        self.local_match_radius = local_match_radius
        self.supervise_init_depth = supervise_init_depth
        self.sample_log_depth = sample_log_depth

        if local_match_radius > 0:
            self.flow_corr = flow_corr = False

        if concat_geo_volume:
            assert flow_corr or local_match_radius > 0

        if depth_sample_geo_volume:
            assert concat_geo_volume

        context_dims = args.hidden_dims
        self.cv_group = 8
        volume_dim = 28

        self.cnet = ContextNetDino(output_dim=[args.hidden_dims, context_dims], downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim,
            depth_head_dim=2 * local_match_radius + 1,
            )
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])

        # unused parameters
        # self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, kernel_size=3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feature = Feature(vit_type=vit_type, no_freeze_mono=no_freeze_mono)
        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        # self.stem_4 = nn.Sequential(
        #     BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(48, 48, 3, 1, 1, bias=False),
        #     nn.InstanceNorm2d(48), nn.ReLU()
        #     )

        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),
          )


        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            )
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
        self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out)
        self.classifier = nn.Sequential(
          BasicConv(volume_dim, volume_dim//2, kernel_size=3, padding=1, is_3d=True),
          ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
          nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3),
        )

        r = self.args.corr_radius
        dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1)
        self.dx = dx

        if self.flow_corr:
            flow_corr_channels = self.flow_corr_levels * (2 * 4 + 1) ** 2
            if self.concat_geo_volume:
                self.corr_proj = nn.Conv2d(flow_corr_channels + 504, 522, 1)
            else:
                self.corr_proj = nn.Conv2d(flow_corr_channels, 522, 1)

        if self.local_match_radius > 0:
            # TODO: multi-scale matching, maybe also combine geometry volume sampled from geo_volume
            corr_channels = 2 * self.local_match_radius + 1
            self.correlation_proj = nn.Conv2d(corr_channels, 522, 1)

        if self.no_geo_volume:
            self.corr_proj = nn.Conv2d(18, 522, 1)

        # unused parameters
        # del self.cnet.down[0].weight
        # del self.cnet.down[0].bias
        # del self.stem_4[0].conv.weight
        # del self.stem_4[1].weight


    def upsample_disp(self, disp, mask_feat_4, stem_2x, task='stereo'):
        assert task in ['stereo', 'depth']

        dtype = torch.bfloat16 if self.amp_bf16 else torch.float16

        with autocast('cuda', enabled=self.args.mixed_precision, dtype=dtype):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            if task == 'depth':
                # no 4x disp since we predict inverse depth
                up_disp = context_upsample(disp, spx_pred, task='depth').unsqueeze(1)
            else:
                up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp.float()


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None, 
        no_norm_img=False,
        task='stereo',
        intrinsics=None,
        pose=None,  # relative pose transform
        min_depth=1. / 0.5,  # inverse depth range
        max_depth=1. / 10,
        num_depth_candidates=64,
        pred_bidir_depth=False,
        return_features=False,
        rectified_stereo=False,
        ):
        """ Estimate disparity between pair of frames """
        assert task in ['stereo', 'depth']

        if self.sample_log_depth:
            min_depth, max_depth = np.log(1. / max_depth), np.log(1. / min_depth)
            # print(min_depth, max_depth)

        if rectified_stereo:
            from PIL import Image
            ori_img1 = image1[0].permute(1, 2, 0).cpu().numpy()
            ori_img2 = image2[0].permute(1, 2, 0).cpu().numpy()
            ori_concat = np.concatenate((ori_img1, ori_img2), axis=1)

            save_dir = 'tmp_rectified_stereo'
            os.makedirs(save_dir, exist_ok=True)

            # Image.fromarray(ori_concat.astype(np.uint8)).save(save_dir + '/ori.png')

            img_left = ori_img1
            img_right = ori_img2

            ori_h, ori_w = image1.shape[-2:]
            tmp_intrinsics = intrinsics.clone()
            tmp_intrinsics[:, 0] *= ori_w
            tmp_intrinsics[:, 1] *= ori_h

            K1 = K2 = tmp_intrinsics[0].cpu().numpy()
            D1 = np.zeros(5)  # Assuming no distortion
            D2 = np.zeros(5)

            R = pose[0].cpu().numpy()[:3, :3]
            T = pose[0].cpu().numpy()[:3, 3:]

            # TODO: choose left or right
            # if T[0] < 0:
            #     img_left, img_right = ori_img1, ori_img2
            # else:
            #     # swap
            #     img_left, img_right = ori_img2, ori_img1
            #     # inverse
            #     R, T = R.T, -R.T @ T

            # [w, h]
            img_size = [ori_img1.shape[1], ori_img1.shape[0]]

            # === 3. Stereo rectification ===
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T)

            # === 4. Create rectification maps ===
            map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

            # === 5. Remap (rectify) images ===
            rect_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
            rect_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

            # print(rect_left.shape, rect_right.shape)

            rect_concat = np.concatenate((rect_left, rect_right), axis=1)

            concat = np.concatenate((ori_concat, rect_concat), axis=0)
            global count
            Image.fromarray(concat.astype(np.uint8)).save(save_dir + f'/rect_{count}.png')
            count += 1

            if count > 20:
                assert False

        B = len(image1)
        low_memory = low_memory or (self.args.get('low_memory', False))
        if not no_norm_img:
            image1 = normalize_image(image1)
            image2 = normalize_image(image2)

        dtype = torch.bfloat16 if self.amp_bf16 else torch.float16
        with autocast('cuda', enabled=self.args.mixed_precision, dtype=dtype):
            out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
            if not pred_bidir_depth:
                vit_feat = vit_feat[:B]
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            if pred_bidir_depth:
                stem_2x = self.stem_2(torch.cat([image1, image2], dim=0))
                for i in range(len(features_left)):
                    features_left[i], features_right[i] = torch.cat([features_left[i], features_right[i]], dim=0), torch.cat([features_right[i], features_left[i]], dim=0)
            else:
                stem_2x = self.stem_2(image1)

            if task == 'depth':
                assert intrinsics is not None and pose is not None
                # NOTE: in this codebase, intrinsics are normalized by image width and height
                # in unimatch's codebase, no normalization
                ori_h, ori_w = image1.shape[-2:]
                intrinsics = intrinsics.clone()
                intrinsics[:, 0] *= ori_w
                intrinsics[:, 1] *= ori_h

                # scale intrinsics
                intrinsics_curr = intrinsics.clone()
                intrinsics_curr[:, :2] = intrinsics_curr[:, :2] / 4

                if pred_bidir_depth:
                    intrinsics_curr = intrinsics_curr.repeat(2, 1, 1)
                    pose = torch.cat((pose, torch.inverse(pose)), dim=0)

                b, _, h, w = features_left[0].shape

                depth_candidates = torch.linspace(min_depth, max_depth, num_depth_candidates).type_as(image1)
                depth_candidates = depth_candidates.view(1, num_depth_candidates, 1, 1).repeat(b, 1, h,
                                                                                               w)  # [B, D, H, W]

                # gwc_volume: [B, G, D, H, W]
                gwc_volume, warped_feature1 = group_correlation_softmax_depth(features_left[0], features_right[0],
                    intrinsics_curr,
                    pose,
                    depth_candidates=depth_candidates,
                    num_groups=self.cv_group,
                    sample_log_depth=self.sample_log_depth,
                    )

                left_tmp = self.proj_cmb(features_left[0]).unsqueeze(2).repeat(1, 1, num_depth_candidates, 1, 1)  # [B, C, D, H, W]
                right_tmp = self.proj_cmb(warped_feature1.reshape(b, features_left[0].size(1), -1, w)).reshape(b, -1, num_depth_candidates, h, w)
                concat_volume = torch.cat((left_tmp, right_tmp), dim=1)  # [B, 2C, D, H, W]
                del left_tmp, right_tmp
                
            else:
                gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group)  # Group-wise correlation volume (B, N_group, max_disp, H, W)
                left_tmp = self.proj_cmb(features_left[0])
                right_tmp = self.proj_cmb(features_right[0])
                concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp//4)
                del left_tmp, right_tmp

            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            comb_volume = self.corr_stem(comb_volume)
            comb_volume = self.corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(comb_volume, features_left)

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)  #(B, max_disp, H, W)
            if init_disp is None:
                if task == 'depth':
                    init_disp = (prob * depth_candidates).sum(dim=1, keepdim=True)  # [B, 1, H, W]
                else:
                    init_disp = disparity_regression(prob, self.args.max_disp//4)  # Weighted  sum of disparity

            if pred_bidir_depth:
                cnet_list = self.cnet(torch.cat((image1, image2), dim=0), vit_feat=vit_feat, num_layers=self.args.n_gru_layers)   #(1/4, 1/8, 1/16)
            else:
                cnet_list = self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)   #(1/4, 1/8, 1/16)
            cnet_list = list(cnet_list)
            net_list = [torch.tanh(x[0]) for x in cnet_list]   # Hidden information
            inp_list = [torch.relu(x[1]) for x in cnet_list]   # Context information list of pyramid levels
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]

        if self.flow_corr:
            geo_fn = CorrBlock(features_left[0].float(), features_right[0].float(), num_levels=self.flow_corr_levels)
            if self.concat_geo_volume:
                geo_volume_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), 
                    num_levels=self.args.corr_levels, dx=self.dx, no_corr=True)

        else:
            geo_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), num_levels=self.args.corr_levels, dx=self.dx)

        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)  # (B,H,W,1) Horizontal only
        disp = init_disp.float()
        disp_preds = []

        # GRUs iterations to update disparity (1/4 resolution)
        for itr in range(iters):
            disp = disp.detach()
            if self.flow_corr:
                proj_coords_from_depth = compute_flow_with_depth_pose(
                    torch.exp(disp.squeeze(1)) if self.sample_log_depth else 1. / disp.squeeze(1),
                    intrinsics_curr,
                    extrinsics_rel=pose,
                    return_coords=True,
                    )
                geo_feat = geo_fn(proj_coords_from_depth)

                if self.concat_geo_volume:
                    if self.depth_sample_geo_volume:
                        indices = torch.sum(disp >= depth_candidates, dim=1, keepdim=True) - 1
                        # Clamp to ensure indices are within [0, D-1]
                        indices = indices.clamp(min=0, max=num_depth_candidates-1).float()
                        tmp = geo_volume_fn(indices, coords)
                    else:
                        tmp = geo_volume_fn(disp, coords)
                    geo_feat = torch.cat((geo_feat, tmp), dim=1)

                # use the pre-trained weights
                geo_feat = self.corr_proj(geo_feat)

            elif self.local_match_radius > 0:
                # 2x smaller interval for each iteration
                disp_interval = (max_depth - min_depth) / num_depth_candidates / (2 ** itr)
                disp_range_min = (disp - disp_interval * self.local_match_radius).clamp(min=min_depth)  # [B, 1, H, W]
                disp_range_max = (disp + disp_interval * self.local_match_radius).clamp(max=max_depth)
                linear_space = torch.linspace(0, 1, 2 * self.local_match_radius + 1
                    ).type_as(disp).view(1, -1, 1, 1)  # [1, K, 1, 1]
                disp_candidates = disp_range_min + linear_space * (disp_range_max - disp_range_min)  # [B, K, H, W]

                warped_feature1 = warp_with_pose_depth_candidates(features_right[0].float(), 
                                                                intrinsics_curr,
                                                                 pose,
                                                                 torch.exp(disp_candidates) if self.sample_log_depth else (1. / disp_candidates),
                                                                 )  # [B, C, K, H, W]
                corr = (F.normalize(features_left[0].float().unsqueeze(2), dim=1) * F.normalize(warped_feature1, dim=1)).sum(1)  # [B, K, H, W]
                geo_feat = self.correlation_proj(corr)
                
            else:
                geo_feat = geo_fn(disp, coords, low_memory=low_memory, no_geo_volume=self.no_geo_volume)

                if self.no_geo_volume:
                    geo_feat = self.corr_proj(geo_feat)

            dtype = torch.bfloat16 if self.amp_bf16 else torch.float16
            with autocast('cuda', enabled=self.args.mixed_precision, dtype=dtype):
              net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)

            if self.local_match_radius > 0:
                match_prob = F.softmax(delta_disp.float(), dim=1)
                disp = (match_prob * disp_candidates).sum(1, keepdim=True)
            else:
                disp = disp + delta_disp.float()

            if task == 'depth':
                disp = disp.clamp(min=min_depth, max=max_depth)
                
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if self.bilinear_up_depth:
                disp_up = F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)
            else:
                disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float(), task=task)
                
            disp_preds.append(disp_up)

        if iters == 0 and task == 'depth':
            disp = F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True).squeeze(1)
            disp_up = torch.exp(disp) if self.sample_log_depth else (1. / disp)
            # else:
            #     # no refine, check the base model
            #     disp = disp.detach()
            #     geo_feat = geo_fn(disp, coords, low_memory=low_memory)

            #     dtype = torch.bfloat16 if self.amp_bf16 else torch.float16
            #     with autocast('cuda', enabled=self.args.mixed_precision, dtype=dtype):
            #         net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)

            #     # upsample predictions
            #     disp_up = 1. / self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float(), task=task).squeeze(1)

            if test_mode:
                if pred_bidir_depth:
                    half = disp_up.size(0) // 2

                    if return_features:
                        return disp_up[:half], disp_up[half:], torch.cat((vit_feat, features_left[0]), dim=1), prob

                return disp_up
            else:
                depth_preds = [disp_up]

                if pred_bidir_depth:
                    half = depth_preds[0].size(0) // 2
                    fwd_depth_preds = [pred[:half] for pred in depth_preds]
                    bwd_depth_preds = [pred[half:] for pred in depth_preds]

                    if return_features:
                        return fwd_depth_preds, bwd_depth_preds, torch.cat((vit_feat, features_left[0]), dim=1), prob

                    return fwd_depth_preds, bwd_depth_preds

                return depth_preds

        if task == 'depth':
            # convert inverse depth to depth
            disp_up = torch.exp(disp_up.squeeze(1)) if self.sample_log_depth else (1. / disp_up.squeeze(1))
            init_disp = torch.exp(init_disp) if self.sample_log_depth else (1. / init_disp)
            for i in range(len(disp_preds)):
                disp_preds[i] = torch.exp(disp_preds[i].squeeze(1)) if self.sample_log_depth else (1. / disp_preds[i].squeeze(1))  # [B, H, W]

        if test_mode or not self.training:
            if task == 'depth':
                # disp_up = disp_up.clamp(min=min_depth, max=max_depth)

                if pred_bidir_depth:
                    half = disp_up.size(0) // 2

                    if return_features:
                        return disp_up[:half], disp_up[half:], torch.cat((vit_feat, features_left[0]), dim=1), prob

                    return disp_up[:half], disp_up[half:]

            return disp_up

        if task == 'depth':
            # upsample to the full resolution to add supervison
            init_disp = F.interpolate(init_disp, scale_factor=4, mode='bilinear', align_corners=True).squeeze(1)

            if self.supervise_init_depth:
                depth_preds = [init_disp] + disp_preds
            else:
                depth_preds = disp_preds

            if pred_bidir_depth:
                half = depth_preds[0].size(0) // 2
                fwd_depth_preds = [pred[:half] for pred in depth_preds]
                bwd_depth_preds = [pred[half:] for pred in depth_preds]

                if return_features:
                    return fwd_depth_preds, bwd_depth_preds, torch.cat((vit_feat, features_left[0]), dim=1), prob

                return fwd_depth_preds, bwd_depth_preds

            return depth_preds

        return init_disp, disp_preds


    def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
      B,_,H,W = image1.shape
      img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
      img1_small, img2_small = padder.pad(img1_small, img2_small)
      disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
      disp_small = padder.unpad(disp_small.float())
      disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio
      disp_small_up = disp_small_up.clip(0, None)

      padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
      image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up)
      disp_small_up += padder._pad[0]
      init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25   # Init disp will be 1/4
      disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
      disp = padder.unpad(disp.float())
      return disp


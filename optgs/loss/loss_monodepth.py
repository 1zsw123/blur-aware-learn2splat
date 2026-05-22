import torch


import cv2
import torch

from optgs.model.encoder.depth_anything_v2.dpt import DepthAnythingV2



def get_monodepth_model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'pretrained/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model



def get_monodepth_pred(img, model):

    with torch.no_grad():
        pass


def get_monodepth_loss(pred_depth, img):
    pass




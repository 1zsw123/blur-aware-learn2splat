import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from optgs.model.types import Gaussians
from optgs.scene_trainer.common.gaussians import build_covariance


class GaussiansModule(nn.Module):
    def __init__(
        self, 
        means: Float[Tensor, "gaussian 3"],
        harmonics: Float[Tensor, "gaussian 3 d_sh"],
        opacities: Float[Tensor, "gaussian"],
        scales: Float[Tensor, "gaussian 3"],
        rotations_unnorm: Float[Tensor, "gaussian 4"]
    ):
        # all gaussians parameters are post-activation
        
        super().__init__()
        
        def _register_param(name, value):
            if value is None:
                setattr(self, name, None)
            else:
                param = nn.Parameter(value)
                setattr(self, name, param)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.rotation_activation = F.normalize

        # Register parameters
        means = means.detach().clone()
        means.requires_grad_(True)
        
        harmonics = harmonics.detach().clone()  # [G, sh_d, C]
        d_sh = harmonics.shape[-1]
        sh0 = harmonics[..., 0:1]  # [G, 3, 1]
        if d_sh == 1:
            # sh_degree = 0
            shN = None
        else:
            # sh_degree > 0
            shN = harmonics[..., 1:]  # [G, 3, d_sh-1]

        sh0.requires_grad_(True)
        if shN is not None:
            shN.requires_grad_(True)

        # Invert the opacity to optimize in the unconstrained space
        opacities_raw = self.inverse_opacity_activation(opacities.detach().clone(), eps=1e-6)
        opacities_raw.requires_grad_(True)
        
        # Invert the scales
        scales_raw = self.scaling_inverse_activation(scales.detach().clone())
        scales_raw.requires_grad_(True)
        
        # Rotations in xyzw (scalar last)
        # remember to convert to wxyz (scalar first) before rendering and saving to ply
        rotations_unnorm = rotations_unnorm.detach().clone()
        rotations_unnorm.requires_grad_(True)
        
        _register_param("opacities_raw", opacities_raw)
        _register_param("scales_raw", scales_raw)
        _register_param("means", means)
        _register_param("rotations_unnorm", rotations_unnorm)
        _register_param("sh0", sh0)
        if shN is not None:
            _register_param("shN", shN)
        
        for name, param in self.named_parameters():
            print(f"Registered parameter: {name}, shape: {param.shape}, dtype: {param.dtype}, min: {param.min()}, max: {param.max()}, requires_grad: {param.requires_grad}")

    @property
    def scales(self):
        scales = self.scaling_activation(self.scales_raw)
        return scales
    
    @property
    def opacities(self):
        opacities = self.opacity_activation(self.opacities_raw)
        return opacities
    
    @property
    def rotations(self):
        rotations = self.rotation_activation(self.rotations_unnorm, dim=-1)
        return rotations

    @property
    def harmonics(self):
        # returns [G, 3, d_sh]
        shN = getattr(self, "shN", None)
        if shN is not None:
            harmonics_ = torch.cat([self.sh0, shN], dim=-1)
        else:
            harmonics_ = self.sh0
        return harmonics_

    @property
    def covariances(self):
        rotation_xyzw = self.rotations
        covariances = self.covariance_activation(self.scales, rotation_xyzw)  # [G, 3, 3]
        return covariances
    
    def reset_opacity(self, optimizer):
        opacities_old = self.opacity_activation(self.opacities_raw)
        opacities_raw_new = self.inverse_opacity_activation(torch.min(opacities_old, torch.ones_like(opacities_old)*0.01), eps=1e-6)
        # optimizable_tensors = self.replace_tensor_to_optimizer(optimizer, opacities_raw_new, "opacity")
        # self.opacities_raw = optimizable_tensors["opacity"]
    


def gaussians2module(gaussians: Gaussians, device: torch.device) -> GaussiansModule:
    bs = gaussians.means.shape[0]
    assert bs == 1, "Batch size > 1 not supported for post-processing"
    # bs = 1
    # convert Gaussians to GaussiansModule
    gaussian_module = GaussiansModule(
        means=gaussians.means[0].to(device),
        harmonics=gaussians.harmonics[0].to(device),
        opacities=gaussians.opacities[0].to(device),
        scales=gaussians.scales[0].to(device),
        rotations_unnorm=gaussians.rotations_unnorm[0].to(device),
    )
    return gaussian_module


def module2gaussians(gaussian_module: GaussiansModule) -> Gaussians:
    gaussians = Gaussians(
        means=gaussian_module.means.unsqueeze(0),  # [1, G, 3]
        covariances=gaussian_module.covariances.unsqueeze(0),  # [1, G, 3, 3]
        harmonics=gaussian_module.harmonics.unsqueeze(0),  # [1, G, sh_d, C]
        opacities=gaussian_module.opacities.unsqueeze(0),  # [1, G]
        scales=gaussian_module.scales.unsqueeze(0),  # [1, G, 3]
        rotations=gaussian_module.rotations.unsqueeze(0),  # [1, G, 4]
        rotations_unnorm=gaussian_module.rotations.unsqueeze(0),
    )
    return gaussians

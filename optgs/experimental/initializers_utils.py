from sklearn.neighbors import NearestNeighbors
import torch
from torch import Tensor

from optgs.scene_trainer.common.gaussian_adapter import RGB2SH


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def points_to_gaussians(
    points_dict: dict[str, Tensor],
    sh_degree: int = 3,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Tensor]:
    
    xyz = points_dict["xyz"].clone().to(device)
    N = xyz.shape[0]
    
    # color is SH coefficients
    rgbs = points_dict["rgb"].clone().to(device)  # [N, 3], in [0, 1]
    
    # if sh_degree > 0:
    shs = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=device)  # [N, K, 3]
    shs[:, 0, :] = RGB2SH(rgbs)
    sh0 = shs[:, :1, :]  # [N, 1, 3]
    if sh_degree > 0:
        shN = shs[:, 1:, :]  # [N, K-1, 3]
    else:
        shN = None

    quats_unnorm = torch.rand((N, 4), device=device)  # [N, 4]

    scales = points_dict["scales"].clone().to(device)  # [N, 3]
    scales_raw = torch.log(scales)

    opacities = points_dict["opacities"].clone().to(device)  # [N,]
    opacities_raw = torch.logit(opacities)
    
    return {
        "xyz": xyz,
        "sh0": sh0,  # [N, 1, 3]
        "shN": shN,  # [N, sh_d-1, 3] or None
        "scales_raw": scales_raw,
        "rotations_unnorm": quats_unnorm,
        "opacities_raw": opacities_raw,
    }

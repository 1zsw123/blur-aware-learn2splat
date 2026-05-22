import gsplat
import torch
from gsplat import spherical_harmonics

if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"gsplat version: {gsplat.__version__}")

    b = 1
    v = 2
    g = 10
    d = 1
    sh_degree_to_use = 0

    # Directions [b, v, g, 3]
    dirs = torch.tensor([[[[0.0, 0.0, 1.0]] * g] * v] * b)  # [b, v, g, 3]
    dirs = dirs.to(dtype=torch.float32, device="cuda")

    # SHs [b, v, g, d, 3]
    shs = torch.ones(b, v, g, d, 3) * 0.1  # [b, v, g, d, 3]
    shs = shs.to(dtype=torch.float32, device="cuda")

    # Masks (optional) [b, v, g]
    masks = torch.rand(b, v, g) > 0.5  # Random boolean mask
    masks = masks.to(device="cuda")

    print("======================== With Mask ========================")
    for i in range(5):
        dirs_copy = dirs.clone()
        shs_copy = shs.clone()
        masks_copy = masks.clone()

        colors = spherical_harmonics(
            sh_degree_to_use, dirs_copy, shs_copy, masks=masks_copy
        )  # [..., C, N, 3]

        print(
            f"Iteration {i}: colors max {colors.max().item():.4f}, min {colors.min().item():.4f}, mean {colors.mean().item():.4f}")

    print("======================== Without Mask ========================")
    for i in range(5):
        dirs_copy = dirs.clone()
        shs_copy = shs.clone()

        colors = spherical_harmonics(
            sh_degree_to_use, dirs_copy, shs_copy
        )  # [..., C, N, 3]

        print(
            f"Iteration {i}: colors max {colors.max().item():.4f}, min {colors.min().item():.4f}, mean {colors.mean().item():.4f}")
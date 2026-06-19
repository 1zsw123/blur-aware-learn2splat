import pathlib

import torch

from optgs.dataset.dataset_colmap import Parser
import json

from einops import rearrange, repeat

if __name__ == '__main__':
    scene = "14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49"
    colmap_cache_dir = pathlib.Path("datasets/dl3dv-colmap-cache/1K")
    colmap_benchmark_dir = pathlib.Path("datasets/dl3dv-benchmark")
    chunk_path = pathlib.Path("datasets/dl3dv-480p-chunks/test/000004.torch")

    # Extract points and cameras from colmap cache
    parser_colmap_cache = Parser(
        data_dir=str(colmap_cache_dir / scene),
        factor=1,  # not used for point cloud
        normalize=False,  # not used for point cloud
        load_images=False,  # not used for point cloud
        dl3dv_settings=None
    )
    c2w_colmap_cache = torch.from_numpy(parser_colmap_cache.camtoworlds)
    points_xyz_colmap_cache = torch.from_numpy(parser_colmap_cache.points)

    # Load colmap cache transform
    with open(colmap_cache_dir / scene / "transforms.json", 'r') as f:
        transform_colmap_cache_data = json.load(f)
    transforms_colmap_c2ws = []
    for frame in transform_colmap_cache_data['frames']:
        c2w = torch.tensor(frame['transform_matrix'], dtype=c2w_colmap_cache.dtype)
        transforms_colmap_c2ws.append(c2w)
    transforms_colmap_c2ws = torch.stack(transforms_colmap_c2ws, dim=0)

    # Extract points and cameras from colmap benchmark
    # images.bin is missing, so we do not have the poses of colmap from the benchmark
    parser_benchmark = Parser(
        data_dir=str(colmap_benchmark_dir / scene / "nerfstudio" / "colmap"),
        # The sparse dir is not in the same hyrarchy of the images, for debugging we need to copy the spase dir one step up
        factor=1,  # not used for point cloud
        normalize=False,  # not used for point cloud
        load_images=False,  # not used for point cloud
        dl3dv_settings=None
    )
    c2w_benchmark = torch.from_numpy(parser_benchmark.camtoworlds)
    points_xyz_benchmark = torch.from_numpy(parser_benchmark.points)

    # Load transforms.json from nerfstudio format
    with open(colmap_benchmark_dir / scene / "nerfstudio" / "transforms.json", 'r') as f:
        transforms_benchmark_data = json.load(f)
    transforms_benchmark_c2ws = []
    for frame in transforms_benchmark_data['frames']:
        c2w = torch.tensor(frame['transform_matrix'], dtype=c2w_colmap_cache.dtype)
        transforms_benchmark_c2ws.append(c2w)
    transforms_benchmark_c2ws = torch.stack(transforms_benchmark_c2ws, dim=0)
    applied_transform = torch.tensor(transforms_benchmark_data["applied_transform"],
                                     dtype=c2w_colmap_cache.dtype)  # [3, 4]

    # Loading chunk example cameras
    chunk = torch.load(chunk_path)
    chunk = chunk[0]
    assert chunk["url"] == scene
    cameras = chunk["cameras"]
    w2c = repeat(torch.eye(4, dtype=c2w_colmap_cache.dtype),
                 "h w -> b h w", b=len(cameras)).clone()
    w2c[:, :3] = rearrange(cameras[:, 6:], "b (h w) -> b h w", h=3, w=4)
    c2w_chunk = w2c.inverse()

    blender2opencv = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=c2w_colmap_cache.dtype,
        device=c2w_colmap_cache.device
    )
    c2w_chunk_transformed = c2w_chunk @ blender2opencv

    # Compare c2w_chunk_transformed with transforms_colmap_cache_c2ws
    diff = c2w_chunk_transformed - transforms_benchmark_c2ws
    max_diff = diff.abs().max()
    print(f"Max difference between chunk poses and benchmark colmap poses: {max_diff.item()}")
    assert max_diff < 1e-4, "Chunk camera poses do not match benchmark colmap poses after transformation."



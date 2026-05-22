import json
import numpy as np
from pathlib import Path
from .read_write_model import Camera, write_cameras_binary


def save_opencv_camera(K: np.ndarray, json_path: str, output_path, image_size: tuple, camera_id=1):
    assert K.shape == (3, 3), "Intrinsic matrix K must be 3x3"

    # Extract intrinsics from K
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Load distortion coefficients from JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    k1 = data.get("k1", 0.0)
    k2 = data.get("k2", 0.0)
    p1 = data.get("p1", 0.0)
    p2 = data.get("p2", 0.0)

    # Image resolution
    w, h = image_size

    # COLMAP OPENCV model requires 8 parameters
    params = (fx, fy, cx, cy, k1, k2, p1, p2)

    # Create and write Camera object
    camera = Camera(
        id=camera_id,
        model="OPENCV",
        width=w,
        height=h,
        params=params
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    write_cameras_binary({camera_id: camera}, output_path / "cameras.bin")

# Example usage
if __name__ == "__main__":
    K = np.array([
        [629.23, 0,     480.0],
        [0,     625.73, 270.0],
        [0,     0,      1.0]
    ])
    save_opencv_camera(
        K=K,
        json_path="transforms.json",
        output_path="output_model",
        image_size=(960, 540),  # e.g., 4x downsampled from 3840x2160
        camera_id=1
    )

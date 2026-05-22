import numpy as np
from pathlib import Path
from .read_write_model import (
    read_images_binary,
    write_images_binary,
    write_points3D_binary,
    Image as ColmapImage
)

def extract_sparse_images_bin(input_model_dir, output_model_dir, selected_image_ids, keep_features=False):
    input_model_dir = Path(input_model_dir)
    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    # Load full images.bin
    images = read_images_binary(input_model_dir / "images.bin")

    # Select and blank the images
    sparse_images = {}
    for image_id in selected_image_ids:
        image = images[image_id]

        if keep_features:
            xys = image.xys
            point3D_ids = image.point3D_ids
        else:
            xys = np.empty((0, 2))
            point3D_ids = np.empty((0,), dtype=int)

        blank_image = ColmapImage(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=xys,
            point3D_ids=point3D_ids
        )
        sparse_images[image_id] = blank_image

    # Save sparse images.bin
    write_images_binary(sparse_images, output_model_dir / "images.bin")

    # Save empty points3D.bin
    write_points3D_binary({}, output_model_dir / "points3D.bin")

# Example usage
if __name__ == "__main__":
    selected_ids = [1, 4, 10, 20]  # Replace with your sparse frame IDs
    extract_sparse_images_bin(
        input_model_dir="dense_model",
        output_model_dir="sparse_model",
        selected_image_ids=selected_ids
    )

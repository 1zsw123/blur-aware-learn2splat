"""
This script verifies that the scenes in the chunk files match the scenes in the colmap directory.
"""
import json

import torch
from tqdm import tqdm

from optgs.misc.io import CustomPath
from optgs.scripts.dl3dv_hf_download import validate_sfm_structure

if __name__ == '__main__':
    chunk_dir = CustomPath("datasets/dl3dv-480p-chunks/train")
    colmap_dir = CustomPath("datasets/dl3dv-colmap-sfm")

    assert chunk_dir.is_dir(), f"Chunk directory {chunk_dir:link}"
    assert colmap_dir.is_dir(), f"Colmap directory {colmap_dir:link}"

    # First check if we have already saved the chunk scene names to a text file
    chunk_scene_names_file = chunk_dir / "dl3dv_chunk_scenes.txt"
    if chunk_scene_names_file.is_file():
        with chunk_scene_names_file.open("r") as f:
            chunk_scene_names = set(line.strip() for line in f)
        print(f"Loaded {len(chunk_scene_names)} scene names from {chunk_scene_names_file}")
    else:
        # Collect scene names from chunk files
        chunk_scene_names = set()
        for i, chunk_path in tqdm(enumerate(chunk_dir.glob("*.torch"))):
            chunk = torch.load(chunk_path)
            for scene in chunk:
                scene_name = scene["key"]
                scene_name = scene_name.replace("dl3dv_", "")
                chunk_scene_names.add(scene_name)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} chunk files, collected {len(chunk_scene_names)} unique scene names so far...")

        print(f"Scenes in chunk files: {len(chunk_scene_names)}")
        # Save chunk scene names to a text file for reuse
        with open(chunk_scene_names_file, "w") as f:
            for scene_name in sorted(chunk_scene_names):
                f.write(f"{scene_name}\n")

    # Collect scene names from colmap directory
    colmap_scene_names = set()
    unsucc_count = 0
    for scene in colmap_dir.iterdir():
        # Verify dir structure: should be
        # scene_name/
        #   - transforms.json  (for now, we don't have this)
        #   - sparse/
        #      - 0/
        #         - cameras.bin
        #         - images.bin
        #         - points3D.bin
        if not validate_sfm_structure(scene, unsucc_count=unsucc_count):
            unsucc_count += 1
            continue
        # if not scene.is_dir():
        #     print(f"Warning: {scene:link} is not a directory, skipping...")
        #     continue
        #
        # if not (scene / "sparse").is_dir():
        #     print(f"Warning: {scene:link} does not contain a 'sparse' directory, skipping...")
        #     continue
        #
        # if not (scene / "sparse" / "0").is_dir():
        #     print(f"Warning: {scene:link} does not contain a 'sparse/0' directory, skipping...")
        #     continue
        # for file in ["cameras.bin", "images.bin", "points3D.bin"]:
        #     if not (scene / "sparse" / "0" / file).is_file():
        #         print(f"Warning: {scene:link} does not contain a 'sparse/0/{file}' file, skipping...")
        #         continue

        colmap_scene_names.add(scene.name)

    # Compare the two sets
    in_chunk_not_colmap = chunk_scene_names - colmap_scene_names
    in_colmap_not_chunk = colmap_scene_names - chunk_scene_names

    print(f"Scenes in chunk but not in colmap: {len(in_chunk_not_colmap)}")
    for scene_name in sorted(in_chunk_not_colmap):
        print(f"- {scene_name}")

    print(f"\nScenes in colmap but not in chunk: {len(in_colmap_not_chunk)}")
    # for scene_name in sorted(in_colmap_not_chunk):
    #     print(f"- {scene_name}")

    # Generate index_colmap.json
    target_train_path = CustomPath("datasets/dl3dv-480p-chunks/train/index_colmap.json")
    target_test_path = CustomPath("datasets/dl3dv-480p-chunks/test/index_colmap.json")

    full_train_index_path = CustomPath("datasets/dl3dv-480p-chunks/train/index.json")
    full_test_index_path = CustomPath("datasets/dl3dv-480p-chunks/test/index.json")

    # Load the full index files
    with open(full_train_index_path, "r") as f:
        full_train_index = json.load(f)  # with "dl3dv_" prefix in scene names
    with open(full_test_index_path, "r") as f:
        full_test_index = json.load(f)  # without "dl3dv_" prefix in scene names

    # Filter the full index to only include scenes that has colmap data
    filtered_train_index = {scene_name: data for scene_name, data in full_train_index.items() if
                            scene_name.replace("dl3dv_", "") in colmap_scene_names}
    filtered_test_index = {scene_name: data for scene_name, data in full_test_index.items() if
                           scene_name in colmap_scene_names}

    # Save the filtered index files
    target_train_path.parent.mkdir(parents=True, exist_ok=True)
    target_test_path.parent.mkdir(parents=True, exist_ok=True)
    with target_train_path.open("w") as f:
        json.dump(filtered_train_index, f, indent=4)
    with target_test_path.open("w") as f:
        json.dump(filtered_test_index, f, indent=4)

    print(f"Saved filtered train index with {len(filtered_train_index)} scenes to {target_train_path.resolve()}")
    print(f"Saved filtered test index with {len(filtered_test_index)} scenes to {target_test_path.resolve()}")

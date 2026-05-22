import pathlib

if __name__ == '__main__':
    colmap_dir = pathlib.Path("datasets/dl3dv-colmap-cache/1K")
    testset_scene_example_dir = pathlib.Path("results/dl3dv/8_views_140_scenes/best_adam/2000/optimizervanilla/metrics")

    available_scenes = list(colmap_dir.iterdir())
    testset_scenes = list(testset_scene_example_dir.iterdir())

    available_scene_names = set([scene.name for scene in available_scenes])
    testset_scene_names = set([scene.name for scene in testset_scenes])

    intersection = available_scene_names.intersection(testset_scene_names)

    print(f"Number of available scenes: {len(available_scene_names)}")
    print(f"Number of testset scenes: {len(testset_scene_names)}")
    print(f"Number of scenes in intersection: {len(intersection)}")
    print("Scenes in intersection:")
    for scene_name in sorted(intersection):
        print(f"- {scene_name}")
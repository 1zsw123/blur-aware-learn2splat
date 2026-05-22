import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal, Tuple, List, Union
from pathlib import Path
from itertools import product, combinations
from optgs.visualization.plots3d.utils import PointCloud, Camera
from optgs.dataset.camera_datasets.camera import get_scene_scale


TRANSPARENT = False
BBOX_INCHES = "tight"  # "tight" or "auto"
PAD_INCHES = 0.1
DPI = 100
COLORBAR_FRACTION = 0.04625
LARGE_SCALE_MULTIPLIER = 0.05
SCALE_MULTIPLIER = 0.05
RAY_LENGTH_MULTIPLIER = 1.5


def get_scale(scene_radius: float) -> float:
    scale = SCALE_MULTIPLIER
    if scene_radius <= 1.0:
        return scale
    else:
        return scale + (scene_radius * LARGE_SCALE_MULTIPLIER)


def _draw_3d_init(
    ax: plt.Axes,
    scene_radius: float = 1.0,
    elevation_deg: float = 60.0,
    azimuth_deg: float = 30.0,
    up: Literal["z", "y"] = "z",
):
    if scene_radius < 1.0:
        lim = 1.0
    else:
        lim = scene_radius
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([max(-1, -lim), lim])

    ax.set_xlabel("X")
    ax.set_ylabel("Y") if up == "z" else ax.set_ylabel("Z")
    ax.set_zlabel("Z") if up == "z" else ax.set_zlabel("Y")

    # axis equal
    ax.set_aspect("equal")
    ax.view_init(elevation_deg, azimuth_deg)


def _draw_rays(
    ax: plt.Axes,
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    t_near: Optional[np.ndarray] = None,
    t_far: Optional[np.ndarray] = None,
    rgbs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    max_nr_rays: Optional[int] = None,
    up: Literal["z", "y"] = "z",
    scene_radius: float = 1.0,
):
    if rays_o is None or rays_d is None:
        return

    assert (
        rays_o.shape[0] == rays_d.shape[0]
    ), "ray_o and ray_d must have the same length"

    # subsample
    if max_nr_rays is not None:
        if max_nr_rays < rays_o.shape[0]:
            idx = np.random.permutation(rays_o.shape[0])[:max_nr_rays]
            rays_o = rays_o[idx]
            rays_d = rays_d[idx]
            if rgbs is not None:
                rgbs = rgbs[idx]
            if masks is not None:
                masks = masks[idx]
            if t_near is not None:
                t_near = t_near[idx]
            if t_far is not None:
                t_far = t_far[idx]

    ray_lenght = RAY_LENGTH_MULTIPLIER * scene_radius

    # draw rays
    for i, (ray_o, ray_d) in enumerate(zip(rays_o, rays_d)):
        start_point = ray_o
        end_point = ray_o + ray_d * ray_lenght
        if rgbs is not None:
            color = rgbs[i]
            # check if color is in [0, 255]
            if np.max(color) > 1.0:
                color = color / 255.0
        else:
            color = "blue"
        alpha = 0.75
        if masks is not None:
            mask = masks[i]
            if mask < 0.5:
                alpha = 0.5
        # plot line segment
        ax.plot(
            [start_point[0], end_point[0]],
            (
                [start_point[1], end_point[1]]
                if up == "z"
                else [start_point[2], end_point[2]]
            ),
            (
                [start_point[2], end_point[2]]
                if up == "z"
                else [start_point[1], end_point[1]]
            ),
            color=color,
            alpha=0.3 * float(alpha),
        )

    # draw t_near, t_far points
    _draw_near_far_points(
        ax=ax,
        rays_o=rays_o,
        rays_d=rays_d,
        t_near=t_near,
        t_far=t_far,
        up=up,
        scene_radius=scene_radius,
    )


def _draw_point_cloud(
    ax: plt.Axes,
    point_cloud: PointCloud,
    alpha: Optional[float] = None,
    max_nr_points: Optional[int] = None,
    up: Literal["z", "y"] = "z",
    scene_radius: float = 1.0,
):
    if point_cloud is None:
        return

    scale = get_scale(scene_radius)

    points_3d = point_cloud.points_3d
    points_rgb = point_cloud.points_rgb  # could be None

    # subsample
    if max_nr_points is not None and max_nr_points < point_cloud.points_3d.shape[0]:
        # random subsample
        idx = np.random.permutation(points_3d.shape[0])[:max_nr_points]
    else:
        # keep all points
        idx = np.arange(points_3d.shape[0])

    points_3d = points_3d[idx]
    if points_rgb is not None:
        points_rgb = points_rgb[idx]

    colors = point_cloud.color
    if colors is None:
        colors = "black"

    # prioritize points_rgb over color
    if points_rgb is not None:
        colors = points_rgb / 255.0

    size = point_cloud.size
    if size is None:
        size = 10.0
    size = max(5.0, size * scale)

    marker = point_cloud.marker
    if marker is None:
        marker = "o"

    label = point_cloud.label
    # if None, keep it None

    if alpha is None:
        alpha = 0.5

    # draw points
    if up == "z":
        ax.scatter(
            points_3d[:, 0],
            points_3d[:, 1],
            points_3d[:, 2],
            s=size,
            color=colors,
            alpha=alpha,
            marker=marker,
            label=label,
        )
    else:  # up = "y"
        ax.scatter(
            points_3d[:, 0],
            points_3d[:, 2],
            points_3d[:, 1],
            s=size,
            color=colors,
            alpha=alpha,
            marker=marker,
            label=label,
        )

    if label is not None:
        ax.legend()


def _draw_frame(
    ax: plt.Axes,
    pose: np.ndarray,
    idx: int = 0,
    up: Literal["z", "y"] = "z",
    scene_radius: float = 1.0,
):
    if pose is None:
        return

    scale = get_scale(scene_radius)

    # get axis directions (normalized)
    x_dir = pose[:3, 0]
    x_dir /= np.linalg.norm(x_dir)
    y_dir = pose[:3, 1]
    y_dir /= np.linalg.norm(y_dir)
    z_dir = pose[:3, 2]
    z_dir /= np.linalg.norm(z_dir)

    # frame center
    pos = pose[:3, 3]

    # draw bb frame
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        x_dir[0],
        x_dir[1] if up == "z" else x_dir[2],
        x_dir[2] if up == "z" else x_dir[1],
        length=scale,
        color="r",
    )
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        y_dir[0],
        y_dir[1] if up == "z" else y_dir[2],
        y_dir[2] if up == "z" else y_dir[1],
        length=scale,
        color="g",
    )
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        z_dir[0],
        z_dir[1] if up == "z" else z_dir[2],
        z_dir[2] if up == "z" else z_dir[1],
        length=scale,
        color="b",
    )
    eps = 0.2 * scale
    ax.text(
        pos[0] + eps,  # x
        pos[1] + eps if up == "z" else pos[2] + eps,  # y
        pos[2] + eps if up == "z" else pos[1] + eps,  # z
        str(idx),
    )


def _draw_cartesian_axis(
    ax: plt.Axes, up: Literal["z", "y"] = "z", scene_radius: float = 1.0
):
    _draw_frame(ax=ax, pose=np.eye(4), idx="w", up=up, scene_radius=scene_radius)


def _draw_image_plane(
    ax: plt.Axes, camera: Camera, up: Literal["z", "y"] = "z", scene_radius: float = 1.0
):
    if camera is None:
        return

    scale = get_scale(scene_radius)

    # get image plane corner points in 3D
    # from screen coordinates
    corner_points_2d_screen = np.array(
        [[0, 0], [camera.width, 0], [0, camera.height], [camera.width, camera.height]]
    )

    _, corner_points_d, _ = camera.get_rays(
        points_2d_screen=torch.from_numpy(corner_points_2d_screen).float()
    )  # torch.Tensor
    corner_points_d = corner_points_d.cpu().numpy()

    camera_center = camera.get_center()
    corner_points_3d_world = camera_center + corner_points_d * scale

    for i, j in combinations(range(4), 2):
        if up == "z":
            ax.plot3D(
                *zip(corner_points_3d_world[i], corner_points_3d_world[j]),
                color="black",
                linewidth=1.0,
                alpha=0.5,
            )
        else:
            ax.plot3D(
                *zip(
                    corner_points_3d_world[:, [0, 2, 1]][i],
                    corner_points_3d_world[:, [0, 2, 1]][j],
                ),
                color="black",
                linewidth=1.0,
                alpha=0.5,
            )


def _draw_frustum(
    ax: plt.Axes, camera: Camera, up: Literal["z", "y"] = "z", scene_radius: float = 1.0
):
    if camera is None:
        return

    # get image plane corner points in 3D
    # from screen coordinates
    image_plane_vertices_2d = np.array(
        [[0, 0], [camera.width, 0], [0, camera.height], [camera.width, camera.height]]
    )

    rays_o, rays_d, _ = camera.get_rays(
        points_2d_screen=torch.from_numpy(image_plane_vertices_2d).float()
    )  # torch.Tensor
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()

    _draw_rays(
        ax=ax,
        rays_o=rays_o,
        rays_d=rays_d,
        rgbs=np.zeros((rays_o.shape[0], 3)),
        masks=np.ones((rays_o.shape[0], 1)),
        up=up,
        scene_radius=scene_radius,
    )


def _draw_camera_frame(
    ax: plt.Axes,
    pose: np.ndarray,
    label: str = "c",
    up: Literal["z", "y"] = "z",
    scene_radius: float = 1.0,
):
    if pose is None:
        return

    scale = get_scale(scene_radius)

    # get axis directions (normalized)
    x_dir = pose[:3, 0]
    x_dir /= np.linalg.norm(x_dir)
    y_dir = pose[:3, 1]
    y_dir /= np.linalg.norm(y_dir)
    z_dir = pose[:3, 2]
    z_dir /= np.linalg.norm(z_dir)
    # frame center
    pos = pose[:3, 3]

    # draw camera frame
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        x_dir[0],
        x_dir[1] if up == "z" else x_dir[2],
        x_dir[2] if up == "z" else x_dir[1],
        length=scale,
        color="r",
    )
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        y_dir[0],
        y_dir[1] if up == "z" else y_dir[2],
        y_dir[2] if up == "z" else y_dir[1],
        length=scale,
        color="g",
    )
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        z_dir[0],
        z_dir[1] if up == "z" else z_dir[2],
        z_dir[2] if up == "z" else z_dir[1],
        length=scale,
        color="b",
    )
    ax.text(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        label,
    )


def _draw_point_clouds(
    ax: plt.Axes,
    point_clouds: List[PointCloud] = None,
    max_nr_points: Optional[int] = None,
    up: Literal["z", "y"] = "z",
    scene_radius: float = 1.0,
):
    if point_clouds is None:
        return

    if not isinstance(point_clouds, list):
        raise ValueError("point_clouds must be a list of PointClouds")

    # if pc are given
    if len(point_clouds) > 0:

        # split max_nr_points among point clouds
        if max_nr_points is not None:
            max_nr_points_per_pc = max_nr_points // len(point_clouds)
            if max_nr_points_per_pc == 0:
                max_nr_points_per_pc = 1
        else:
            max_nr_points_per_pc = None

        # plot point clouds
        for i, pc in enumerate(point_clouds):
            _draw_point_cloud(
                ax=ax,
                point_cloud=pc,
                max_nr_points=max_nr_points_per_pc,
                up=up,
                scene_radius=scene_radius,
            )


def _draw_cameras(
    ax: plt.Axes,
    cameras: List[Camera] = None,
    nr_rays: int = 0,
    draw_every_n_cameras: int = 1,
    up: Literal["z", "y"] = "z",
    scene_radius: float = 1.0,
    draw_image_planes=True,
    draw_cameras_frustums=True,
):
    if cameras is None:
        return

    if not isinstance(cameras, list):
        raise ValueError("cameras must be a list of Cameras")

    if len(cameras) > 0:
        nr_cameras = len(cameras) // draw_every_n_cameras
        nr_rays_per_camera = nr_rays // nr_cameras

        # draw camera frames
        for i, camera in enumerate(cameras):

            if i % draw_every_n_cameras == 0:

                pose = camera.get_pose()
                label = camera.label
                _draw_camera_frame(
                    ax=ax,
                    pose=pose,
                    label=label,
                    up=up,
                    scene_radius=scene_radius,
                )
                if draw_image_planes:
                    _draw_image_plane(
                        ax=ax, camera=camera, up=up, scene_radius=scene_radius
                    )
                if draw_cameras_frustums:
                    _draw_frustum(
                        ax=ax, camera=camera, up=up, scene_radius=scene_radius
                    )
                if nr_rays_per_camera > 0:
                    _draw_camera_rays(
                        ax=ax,
                        camera=camera,
                        nr_rays=nr_rays_per_camera,
                        up=up,
                        scene_radius=scene_radius,
                    )

            else:
                # skip camera
                pass


def plot_3d(
    cameras: List[Camera] = None,
    point_clouds: List[PointCloud] = None,
    nr_rays: int = 0,
    draw_every_n_cameras: int = 1,
    max_nr_points: int = 1000,
    azimuth_deg: float = 60.0,
    elevation_deg: float = 30.0,
    scene_radius: Optional[float] = None,
    up: Literal["z", "y"] = "z",
    draw_origin: bool = True,
    draw_image_planes: bool = True,
    draw_cameras_frustums: bool = True,
    figsize: Tuple[int, int] = (15, 15),
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Path] = None,  # if set, saves the figure to the given path
) -> None:
    """
    Returns:
        None
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")
    
    # 
    if scene_radius is None:
        if cameras is not None and len(cameras) > 0:
            camtoworlds = [camera.get_pose() for camera in cameras]  # list of (4, 4)
            # stack to numpy array
            camtoworlds = np.stack(camtoworlds, axis=0)  # (N, 4, 4)
            scene_radius = get_scene_scale(camtoworlds)
        else:
            scene_radius = 1.0

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _draw_3d_init(
        ax=ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
    )

    if draw_origin:
        _draw_cartesian_axis(ax=ax, up=up, scene_radius=scene_radius)

    # draw points
    _draw_point_clouds(
        ax=ax,
        point_clouds=point_clouds,
        # points_3d=points_3d,
        # points_3d_colors=points_3d_colors,
        # points_3d_labels=points_3d_labels,
        # points_3d_sizes=points_3d_sizes,
        # points_3d_markers=points_3d_markers,
        max_nr_points=max_nr_points,
        up=up,
        scene_radius=scene_radius,
    )

    # draw camera frames
    _draw_cameras(
        ax=ax,
        cameras=cameras,
        nr_rays=nr_rays,
        draw_every_n_cameras=draw_every_n_cameras,
        up=up,
        scene_radius=scene_radius,
        draw_image_planes=draw_image_planes,
        draw_cameras_frustums=draw_cameras_frustums,
    )

    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )
        print(f"saved figure to {save_path}")

    if show:
        plt.show()

    plt.close()


def _draw_camera_rays(
    ax: plt.Axes,
    camera,
    nr_rays,
    frame_idx=0,
    up: Literal["z", "y"] = "z",
    scene_radius: float = 1.0,
):
    rays_o, rays_d, points_2d_screen = camera.get_rays()  # torch.Tensor
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()

    # color rays with their uv coordinates
    xy = points_2d_screen  # [:, [1, 0]]
    z = np.zeros((xy.shape[0], 1))
    rgbs = np.concatenate([xy, z], axis=1)
    rgbs[:, 0] /= np.max(rgbs[:, 0])
    rgbs[:, 1] /= np.max(rgbs[:, 1])

    # set to ones
    masks = np.ones((camera.height, camera.width, 1)).reshape(-1, 1) * 0.5

    # draw rays
    _draw_rays(
        ax=ax,
        rays_o=rays_o,
        rays_d=rays_d,
        rgbs=rgbs,
        masks=masks,
        max_nr_rays=nr_rays,
        up=up,
        scene_radius=scene_radius,
    )


def _draw_near_far_points(
    ax: plt.Axes,
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    t_near: float,
    t_far: float,
    up: Literal["z", "y"] = "z",
    scene_radius: float = 1.0,
):
    if rays_o is None or rays_d is None:
        return
    if t_near is None or t_far is None:
        return

    assert (
        rays_o.shape[0] == rays_d.shape[0]
    ), "ray_o and ray_d must have the same length"
    assert (
        t_near.shape[0] == t_far.shape[0]
    ), "t_near and t_far must have the same length"
    assert (
        rays_o.shape[0] == t_near.shape[0]
    ), "ray_o and t_near must have the same length"

    # unsqueeze t_near, t_far if needed
    if t_near.ndim == 1:
        t_near = t_near[:, np.newaxis]
    if t_far.ndim == 1:
        t_far = t_far[:, np.newaxis]

    # draw t_near, t_far points
    p_near = rays_o + rays_d * t_near
    p_far = rays_o + rays_d * t_far

    # unsqueeze p_near, p_far if needed
    if p_near.ndim == 1:
        p_near = p_near[np.newaxis, :]
    if p_far.ndim == 1:
        p_far = p_far[np.newaxis, :]

    p_boundaries = np.concatenate(
        [p_near[:, np.newaxis, :], p_far[:, np.newaxis, :]], axis=1
    )

    pc = PointCloud(
        points_3d=p_boundaries.reshape(-1, 3), size=200, color="black", marker="x"
    )

    for i in range(p_boundaries.shape[0]):
        # draw t_near, t_far points
        _draw_point_cloud(
            ax=ax,
            point_cloud=pc,
            up=up,
            scene_radius=scene_radius,
        )


def plot_current_batch(
    cameras: List[Camera],
    cameras_idx: np.ndarray,
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    rgbs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    azimuth_deg: float = 60.0,
    elevation_deg: float = 30.0,
    scene_radius: float = 1.0,
    up: Literal["z", "y"] = "z",
    draw_origin: bool = True,
    draw_image_planes: bool = True,
    figsize: Tuple[int, int] = (15, 15),
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Path] = None,  # if set, saves the figure to the given path
) -> None:
    """
    Returns:
        None
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    if rgbs is None:
        # if rgb is not given, color rays blue
        rgbs = np.zeros((rays_o.shape[0], 3))
        rgbs[:, 2] = 1.0

    if masks is None:
        # if mask is not given, set to 0.5
        masks = np.ones((rays_o.shape[0], 1)) * 0.5

    # get unique camera idxs
    unique_cameras_idx = np.unique(cameras_idx, axis=0)

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _draw_3d_init(
        ax=ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
    )

    if draw_origin:
        _draw_cartesian_axis(ax=ax, up=up, scene_radius=scene_radius)

    # iterate over all unique cameras in batch
    for idx in unique_cameras_idx:
        camera = cameras[idx]
        pose = camera.get_pose()
        label = camera.label
        _draw_camera_frame(
            ax=ax, pose=pose, label=label, up=up, scene_radius=scene_radius
        )
        if draw_image_planes:
            _draw_image_plane(ax=ax, camera=camera, up=up, scene_radius=scene_radius)

    # draw rays
    _draw_rays(
        ax=ax,
        rays_o=rays_o,
        rays_d=rays_d,
        rgbs=rgbs,
        masks=masks,
        max_nr_rays=None,
        up=up,
        scene_radius=scene_radius,
    )

    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )
        print(f"saved figure to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_rays_samples(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    t_near: Optional[np.ndarray] = None,
    t_far: Optional[np.ndarray] = None,
    nr_rays: int = 32,
    point_clouds: List[PointCloud] = None,
    camera: Camera = None,
    azimuth_deg: float = 60.0,
    elevation_deg: float = 30.0,
    scene_radius: float = 1.0,
    up: Literal["z", "y"] = "z",
    draw_origin: bool = True,
    figsize: Tuple[int, int] = (15, 15),
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Path] = None,  # if set, saves the figure to the given path
) -> None:
    """
    Returns:
        None
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _draw_3d_init(
        ax=ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
    )

    if draw_origin:
        _draw_cartesian_axis(ax=ax, up=up, scene_radius=scene_radius)

    # draw points
    _draw_point_clouds(
        ax=ax,
        point_clouds=point_clouds,
        # points_3d=points_samples,
        # points_3d_colors=points_samples_colors,
        # points_3d_labels=points_samples_labels,
        # points_3d_sizes=points_samples_sizes,
        up=up,
        scene_radius=scene_radius,
    )

    # draw rays
    _draw_rays(
        ax=ax,
        rays_o=rays_o,
        rays_d=rays_d,
        t_near=t_near,
        t_far=t_far,
        max_nr_rays=nr_rays,
        up=up,
        scene_radius=scene_radius,
    )

    # draw camera
    if camera is not None:
        _draw_cameras(
            ax=ax,
            cameras=[camera],
            up=up,
            scene_radius=scene_radius,
            draw_image_planes=True,
            draw_cameras_frustums=True,
        )

    # Get current axes and check if there are any labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Only display legend if there are labels
    if labels:
        plt.legend()

    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )
        print(f"saved figure to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_image(
    image: np.ndarray,  # (W, H)
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    draw_colorbar: bool = False,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (15, 15),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plots an image.

    Args:
        image (np.ndarray): (W, H) or (W, H, 1) or (W, H, 3) or (W, H, 4):.
        title (str, optional): Defaults to None.
    """

    # init figure
    plt.figure(figsize=figsize)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    # transpose to (H, W, C)
    image = np.transpose(image, (1, 0, 2))

    plt.imshow(image, cmap=cmap)

    # Calculate (height_of_image / width_of_image)
    im_ratio = image.shape[0] / image.shape[1]

    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("W")

    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("H")

    if title is not None:
        plt.title(title)

    if draw_colorbar:
        plt.colorbar(fraction=COLORBAR_FRACTION * im_ratio)

    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )
        print(f"saved figure to {save_path}")

    if show:
        plt.show()

    plt.close()

import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple, Union


class Camera:

    def __init__(
        self,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,  # c2w
        width: int,
        height: int,
        color: Optional[str] = None,
        label: Optional[str] = None,
        alpha: Optional[float] = None,
        line_width: Optional[float] = None,
    ):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.width = width
        self.height = height

        # plotting attributes
        self.color = color
        self.label = label
        self.alpha = alpha
        self.line_width = line_width

    def get_intrinsics_inv(self) -> np.ndarray:
        """Get inverse of intrinsic matrix."""
        # check if matrix is invertible
        # if np.linalg.matrix_rank(self.intrinsic) < 3:
        #     print(self.intrinsic)
        #     raise ValueError("Intrinsic matrix is not invertible.")
        return np.linalg.inv(self.intrinsic)

    def get_rays(
        self,
        points_2d_screen: Optional[Tensor] = None,
        nr_rays_per_pixel: int = 1,
        jitter_pixels: bool = False,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get rays from 2D screen points.

        Args:
            points_2d_screen (Tensor): (N, 2) tensor of 2D screen points.
        """

        """returns image rays origins and directions
        for 2d points on the image plane.
        If points are not provided, they are sampled
        from the image plane for every pixel.

        Args:
            points_2d_screen (torch.Tensor, float or int, optional): (N, 2)
                                                Values in [0, W-1], [0, H-1].
                                                Default is None.
            device (str, optional): device to store tensors. Defaults to "cpu".
            jitter_pixels (bool, optional): Whether to jitter pixels.
                                            Only used if points_2d_screen is None.
                                            Defaults to False.
        Returns:
            rays_o (torch.Tensor): rays origins (N, 3)
            rays_d (torch.Tensor): rays directions (N, 3)
            points_2d_screen (torch.Tensor, float): (N, 2) screen space sampling coordinates
        """

        # sample points if not provided
        if points_2d_screen is None:

            assert nr_rays_per_pixel > 0, "nr_rays_per_pixel must be > 0"
            assert nr_rays_per_pixel == 1 or (
                nr_rays_per_pixel > 1 and jitter_pixels is True
            ), "jitter_pixels must be True if nr_rays_per_pixel > 1"

            pixels = get_pixels(self.height, self.width, device=device)  # (W, H, 2)
            # reshape pixels to (N, 2) repeat pixels nr_rays_per_pixel times
            pixels = pixels.reshape(-1, 2)  # (N, 2)
            pixels = pixels.repeat_interleave(nr_rays_per_pixel, dim=0)
            # get points in screen space
            points_2d_screen = pixels_to_points_2d_screen(
                pixels, jitter_pixels
            )  # (N, 2)

        c2w = torch.from_numpy(self.get_pose()).float().to(device)
        intrinsics_inv = torch.from_numpy(self.get_intrinsics_inv()).float().to(device)

        rays_o, rays_d = get_rays_per_points_2d_screen(
            c2w, intrinsics_inv, points_2d_screen
        )

        return rays_o, rays_d, points_2d_screen

    def get_center(self) -> np.ndarray:
        """Get camera center in world coordinates."""
        return self.extrinsic[:3, 3]

    def get_pose(self) -> np.ndarray:
        """Get camera pose (extrinsic matrix)."""
        return self.extrinsic


class PointCloud:

    def __init__(
        self,
        points_3d: np.ndarray,
        points_rgb: Optional[np.ndarray] = None,  # (N, 3) or (3,)
        color: Optional[str] = None,
        label: Optional[str] = None,
        size: Optional[float] = None,
        marker: Optional[str] = None,
    ):
        self.points_3d = points_3d
        self.points_rgb = points_rgb

        if self.points_rgb is not None:
            # check if dimensions are correct
            if self.points_rgb.ndim == 2:
                # first dimension must be the same as points_3d
                if self.points_rgb.shape[0] != self.points_3d.shape[0]:
                    raise ValueError(
                        f"Points RGB must have the same number of points as points 3D, got {self.points_rgb.shape[0]} and {self.points_3d.shape[0]}"
                    )
                # second dimension must be 3
                if self.points_rgb.shape[1] != 3:
                    raise ValueError(
                        f"Points RGB must have shape (N, 3), got {self.points_rgb.shape}"
                    )
            elif self.points_rgb.ndim == 1:
                # first dimension must be 3
                if self.points_rgb.shape[0] != 3:
                    raise ValueError(
                        f"Points RGB must have shape (3,), got {self.points_rgb.shape}"
                    )
            else:
                raise ValueError(
                    f"Points RGB must have shape (N, 3) or (3,), got {self.points_rgb.shape}"
                )

        # plotting attributes
        self.color = color
        self.label = label
        self.size = size
        self.marker = marker

    def downsample(self, nr_points: int):
        if nr_points >= self.points_3d.shape[0]:
            # do nothing
            return

        idxs = np.random.choice(self.points_3d.shape[0], nr_points, replace=False)
        self.points_3d = self.points_3d[idxs]

        if self.points_rgb is not None:
            self.points_rgb = self.points_rgb[idxs]

    def mask(self, mask: np.ndarray):
        self.points_3d = self.points_3d[mask]

        if self.points_rgb is not None:
            self.points_rgb = self.points_rgb[mask]

    def shape(self):
        return self.points_3d.shape

    def __str__(self) -> str:
        return f"PointCloud with {self.points_3d.shape[0]} points"

    def transform(self, transformation: np.ndarray):
        self.points_3d = apply_transformation_3d(self.points_3d, transformation)


def get_mask_points_in_image_range(
    points_2d_screen: Union[np.ndarray, torch.Tensor], width: int, height: int
) -> Union[np.ndarray, torch.Tensor]:
    """Filter out points that are outside the image."""
    mask = (points_2d_screen[:, 0] >= 0) & (points_2d_screen[:, 0] < width)
    mask &= (points_2d_screen[:, 1] >= 0) & (points_2d_screen[:, 1] < height)
    return mask


def apply_transformation_3d(
    points_3d: Union[np.ndarray, torch.Tensor],
    transform: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies a 3D affine transformation to a set of points.

    Args:
        points_3d (numpy.ndarray or torch.Tensor): A (N, 3) array of 3D points.
        transform (numpy.ndarray or torch.Tensor): A (4, 4) affine transformation matrix
                                                    or (N, 4, 4) for per-point transformations.

    Returns:
        numpy.ndarray or torch.Tensor: A (N, 3) array of transformed 3D points.

    Raises:
        ValueError: If the shapes of `points_3d` or `transform` are invalid.
        TypeError: If the input types are inconsistent (mixing NumPy and PyTorch).
    """
    # Check dimensionality of points_3d
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("`points_3d` must be a 2D array of shape (N, 3).")

    # Check dimensionality of transform
    if transform.ndim == 2 and transform.shape == (4, 4):
        batched_transform = False
    elif transform.ndim == 3 and transform.shape[1:] == (4, 4):
        batched_transform = True
    else:
        raise ValueError("`transform` must be of shape (4, 4) or (N, 4, 4).")

    # Ensure consistent types between inputs
    if isinstance(points_3d, np.ndarray) and not isinstance(transform, np.ndarray):
        raise TypeError("Both inputs must be of the same type (NumPy or PyTorch).")
    if isinstance(points_3d, torch.Tensor) and not isinstance(transform, torch.Tensor):
        raise TypeError("Both inputs must be of the same type (NumPy or PyTorch).")

    # Convert points_3d to homogeneous coordinates
    points_homogeneous = euclidean_to_homogeneous(points_3d)

    # Apply transformation
    if isinstance(points_3d, np.ndarray):
        if batched_transform:
            transformed_points = np.einsum("nij,nj->ni", transform, points_homogeneous)
        else:
            transformed_points = points_homogeneous @ transform.T
        return transformed_points[:, :3]
    elif isinstance(points_3d, torch.Tensor):
        if batched_transform:
            transformed_points = torch.einsum(
                "nij,nj->ni", transform, points_homogeneous
            )
        else:
            transformed_points = points_homogeneous @ transform.T
        return transformed_points[:, :3]


def euclidean_to_homogeneous(
    points: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts Euclidean coordinates to homogeneous coordinates by appending a column of ones.

    Args:
        points (np.ndarray or torch.Tensor): A 2D array of shape (N, C) representing Euclidean points.

    Returns:
        np.ndarray or torch.Tensor: A 2D array of shape (N, C+1) in homogeneous coordinates.

    Raises:
        TypeError: If `points` is not a NumPy array or PyTorch tensor.
        ValueError: If `points` is not a 2D array.
    """
    # Check if input is a 2D array
    if points.ndim != 2:
        raise ValueError("`points` must be a 2D array of shape (N, C).")

    if isinstance(points, np.ndarray):
        ones = np.ones((points.shape[0], 1))
        return np.hstack((points, ones))
    elif isinstance(points, torch.Tensor):
        ones = torch.ones(
            (points.shape[0], 1), dtype=points.dtype, device=points.device
        )
        return torch.cat((points, ones), dim=1)
    else:
        raise TypeError("`points` must be either a numpy.ndarray or torch.Tensor.")


def get_pixels(height: int, width: int, device: str = "cpu") -> torch.Tensor:
    """returns all image pixels coords
    Args:
        height (int): frame height
        width (int): frame width
        device (str, optional): Defaults to "cpu".
    Returns:
        pixels (torch.Tensor): dtype int32, shape (W, H, 2), values in [0, W-1], [0, H-1]
    """

    pixels_x, pixels_y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="ij",
    )
    pixels = torch.stack([pixels_x, pixels_y], dim=-1).type(torch.int32)

    return pixels


def get_random_pixels(
    height: int, width: int, nr_pixels: int, device: str = "cpu"
) -> torch.Tensor:
    """given a number or pixels, return random pixels
    Args:
        height (int): frame height
        width (int): frame width
        nr_pixels (int): number of pixels to sample
        device (str, optional): Defaults to "cpu".
    Returns:
        pixels (torch.Tensor, int): (N, 2) with values in [0, W-1], [0, H-1]
    """
    # sample nr_pixels random pixels
    pixels = torch.rand(nr_pixels, 2, device=device)
    pixels[:, 0] *= width
    pixels[:, 1] *= height
    pixels = pixels.type(torch.int32)
    return pixels


def get_pixels_centers(pixels: torch.Tensor) -> torch.Tensor:
    """return the center of each pixel
    Args:
        pixels (torch.Tensor): (N, 2) list of pixels
    Returns:
        pixels_centers (torch.Tensor): (N, 2) list of pixels centers
    """

    points_2d_screen = pixels.float()  # cast to float32
    points_2d_screen = points_2d_screen + 0.5  # pixels centers

    return points_2d_screen


def pixels_to_points_2d_screen(pixels: torch.Tensor, jitter_pixels: bool = False):
    """convert pixels to 2d points on the image plane

    Args:
        pixels (torch.Tensor): (W, H, 2) or (N, 2) list of pixels
        jitter_pixels (bool): whether to jitter pixels
    Returns:
        points_2d_screen (torch.Tensor): (N, 2) list of pixels centers (in screen space)
    """
    assert pixels.dtype == torch.int32, "pixels must be int32"

    # get pixels as 3d points on a plane at z=-1 (in camera space)
    points_2d_screen = get_pixels_centers(pixels)
    points_2d_screen = points_2d_screen.reshape(-1, 2)
    if jitter_pixels:
        points_2d_screen = jitter_points(points_2d_screen)

    return points_2d_screen  # (N, 2)


def jitter_points(points: torch.Tensor) -> torch.Tensor:
    """apply noise to points

    Args:
        points (torch.Tensor): (..., 2) list of pixels centers (in screen space)
    Returns:
        jittered_pixels (torch.Tensor): (..., 2) list of pixels
    """

    assert points.dtype == torch.float32, "points must be float32"

    # # sample offsets from gaussian distribution
    # std = 0.16
    # offsets = torch.normal(
    #     mean=0.0, std=std, size=jittered_points.shape, device=points.device
    # )
    # clamp offsets to [-0.5 + eps, 0.5 - eps]

    # uniformlu sampled offsets
    offsets = torch.rand_like(points, device=points.device)
    offsets -= 0.5  # [-0.5, 0.5]
    eps = 1e-6
    offsets = torch.clamp(offsets, -0.5 + eps, 0.5 - eps)
    return points + offsets


def get_rays_per_points_2d_screen(
    c2w: torch.Tensor, intrinsics_inv: torch.Tensor, points_2d_screen: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """given a list of pixels, return rays origins and directions

    Args:
        c2w (torch.Tensor): (N, 4, 4) or (4, 4)
        intrinsics_inv (torch.Tensor): (N, 3, 3) or (3, 3)
        points_2d_screen (torch.Tensor, float): (N, 2) with values in [0, W-1], [0, H-1]

    Returns:
        rays_o (torch.Tensor): (N, 3)
        rays_d (torch.Tensor): (N, 3)
    """

    # check input shapes
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    elif c2w.ndim == 3:
        pass
    else:
        raise ValueError(f"c2w: {c2w.shape} must be (4, 4) or (N, 4, 4)")

    if c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w: {c2w.shape} must be (4, 4) or (N, 4, 4)")

    if intrinsics_inv.ndim == 2:
        intrinsics_inv = intrinsics_inv.unsqueeze(0)
    elif intrinsics_inv.ndim == 3:
        pass
    else:
        raise ValueError(
            f"intrinsics_inv: {intrinsics_inv} must be (N, 3, 3) or (3, 3)"
        )

    if intrinsics_inv.shape[1:] != (3, 3):
        raise ValueError(
            f"intrinsics_inv: {intrinsics_inv} must be (N, 3, 3) or (3, 3)"
        )

    if points_2d_screen.ndim != 2 or points_2d_screen.shape[1] != 2:
        raise ValueError(f"points_2d_screen: {points_2d_screen.shape} must be (N, 2)")
    if c2w.shape[0] != points_2d_screen.shape[0] and c2w.shape[0] != 1:
        raise ValueError(
            f"input shapes do not match: c2w: {c2w.shape} and points_2d_screen: {points_2d_screen.shape}"
        )
    if (
        intrinsics_inv.shape[0] != points_2d_screen.shape[0]
        and intrinsics_inv.shape[0] != 1
    ):
        raise ValueError(
            f"input shapes do not match: intrinsics_inv: {intrinsics_inv.shape} and points_2d_screen: {points_2d_screen.shape}"
        )

    # ray origin are the cameras centers
    if c2w.shape[0] == points_2d_screen.shape[0]:
        rays_o = c2w[:, :3, -1]
    else:
        rays_o = c2w[0, :3, -1].repeat(points_2d_screen.shape[0], 1)

    # unproject points to 3d camera space
    points_3d_camera = local_inv_perspective_projection(
        intrinsics_inv,
        points_2d_screen,
    )  # (N, 3)
    # points_3d_unprojected have all z=1

    # rotate points with c2w rotation
    rot = c2w[:, :3, :3]
    points_3d_rotated = apply_rotation_3d(points_3d_camera, rot)  # (N, 3)

    # normalize rays
    rays_d = torch.nn.functional.normalize(points_3d_rotated, dim=-1)  # (N, 3)

    return rays_o, rays_d


def local_inv_perspective_projection(
    intrinsics_inv: Union[np.ndarray, torch.Tensor],
    points_2d_screen: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply inverse perspective projection to 2D screen points.

    Args:
        intrinsics_inv (np.ndarray or torch.Tensor): Inverse of camera intrinsic matrix of shape (N, 3, 3) or (3, 3).
        points_2d_screen (np.ndarray or torch.Tensor): 2D points in screen coordinates of shape (N, 2).

    Returns:
        np.ndarray or torch.Tensor: Unprojected 3D points of shape (N, 3).

    Raises:
        ValueError: If inputs have invalid shapes or types.
    """

    # check input shapes
    if intrinsics_inv.ndim == 2:
        intrinsics_inv = intrinsics_inv[None, ...]  # Add batch dimension
    elif intrinsics_inv.ndim == 3:
        pass
    else:
        raise ValueError(
            f"intrinsics_inv: {intrinsics_inv.shape} must have shape (N, 3, 3) or (3, 3)."
        )

    if intrinsics_inv.shape[1:] != (3, 3):
        raise ValueError(
            f"intrinsics_inv: {intrinsics_inv.shape} must have shape (N, 3, 3) or (3, 3)."
        )

    if (
        intrinsics_inv.shape[0] != points_2d_screen.shape[0]
        and intrinsics_inv.shape[0] != 1
    ):
        raise ValueError(
            f"input shapes do not match: intrinsics_inv: {intrinsics_inv.shape} and points_2d_screen: {points_2d_screen.shape}."
        )

    if points_2d_screen.ndim == 2 and points_2d_screen.shape[-1] != 2:
        raise ValueError("`points_2d_screen` must have shape (N, 2).")

    augmented_points_2d_screen = euclidean_to_homogeneous(points_2d_screen)  # (N, 3)
    augmented_points_2d_screen = augmented_points_2d_screen[..., None]  # (N, 3, 1)
    augmented_points_3d_camera = (
        intrinsics_inv @ augmented_points_2d_screen
    )  # (N, 3, 3) @ (N, 3, 1)
    # reshape to (N, 3)
    augmented_points_3d_camera = augmented_points_3d_camera.squeeze(-1)  # (N, 3)

    return augmented_points_3d_camera


def apply_rotation_3d(
    points_3d: Union[np.ndarray, torch.Tensor], rot: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies a 3D rotation to a set of points.

    Args:
        points_3d (numpy.ndarray or torch.Tensor): A (N, 3) array of 3D points.
        rot (numpy.ndarray or torch.Tensor): A (3, 3) rotation matrix or a batch (N, 3, 3) of rotation matrices.

    Returns:
        numpy.ndarray or torch.Tensor: A (N, 3) array of rotated 3D points.

    Raises:
        ValueError: If the shapes of `points_3d` or `rot` are invalid.
        TypeError: If the input types are inconsistent (mixing NumPy and PyTorch).
    """
    # Validate points_3d shape
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("`points_3d` must be a 2D array of shape (N, 3).")

    # Validate rotation matrix shape
    if rot.ndim == 2 and rot.shape == (3, 3):
        batched_rotation = False
    elif rot.ndim == 3 and rot.shape[1:] == (3, 3):
        batched_rotation = True
    else:
        raise ValueError("`rot` must be of shape (3, 3) or (N, 3, 3).")

    # Ensure consistent types between inputs
    if isinstance(points_3d, np.ndarray) and not isinstance(rot, np.ndarray):
        raise TypeError("Both inputs must be of the same type (NumPy or PyTorch).")
    if isinstance(points_3d, torch.Tensor) and not isinstance(rot, torch.Tensor):
        raise TypeError("Both inputs must be of the same type (NumPy or PyTorch).")

    # Apply rotation
    if isinstance(points_3d, np.ndarray):
        if batched_rotation:
            rotated_points = np.einsum("nij,nj->ni", rot, points_3d)
        else:
            rotated_points = points_3d @ rot.T
        return rotated_points
    elif isinstance(points_3d, torch.Tensor):
        if batched_rotation:
            rotated_points = torch.einsum("nij,nj->ni", rot, points_3d)
        else:
            rotated_points = points_3d @ rot.T
        return rotated_points

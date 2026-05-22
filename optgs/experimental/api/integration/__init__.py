"""Integration helpers for using optgs's learned optimizer in external
(inria-style) 3D Gaussian Splatting codebases.

Public surface is :class:`optgs.OptGS`; the symbols here are the building
blocks (kept importable for advanced/low-level use and testing).
"""

from optgs.experimental.api.integration.scene_protocol import (
    OptGSError,
    CameraLike,
    GaussiansLike,
    SceneLike,
    assert_scene_protocol,
)

__all__ = [
    "OptGSError",
    "CameraLike",
    "GaussiansLike",
    "SceneLike",
    "assert_scene_protocol",
]

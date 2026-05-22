"""optgs experimental API — use the learned optimizer in external 3DGS codebases.

The public entry point is :class:`OptGS`. It is exposed lazily via PEP 562
``__getattr__`` so that ``import optgs.experimental.api`` stays cheap (no
torch/hydra import) until ``OptGS`` is actually accessed.
"""

__all__ = ["OptGS", "OptGSError"]


def __getattr__(name: str):
    if name == "OptGS":
        from optgs.experimental.api.api import OptGS

        return OptGS
    if name == "OptGSError":
        from optgs.experimental.api.integration.scene_protocol import OptGSError

        return OptGSError
    raise AttributeError(f"module 'optgs.experimental.api' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

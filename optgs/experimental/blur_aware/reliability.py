"""Representation-independent confidence for deblurred supervision.

The score answers two separate questions: did the deblurred target preserve
the observation, and did it add useful sharpness? It intentionally contains
no dataset label, frame-number rule, or reconstruction PSNR, so it can be
computed before scene optimization without leaking hold/test information.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _gray(images: Tensor) -> Tensor:
    weights = images.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def _sobel(gray: Tensor) -> tuple[Tensor, Tensor]:
    kx = gray.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).view(1, 1, 3, 3) / 8.0
    return F.conv2d(gray, kx, padding=1), F.conv2d(
        gray, kx.transpose(-1, -2), padding=1
    )


def _laplacian_energy(gray: Tensor) -> Tensor:
    kernel = gray.new_tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
    ).view(1, 1, 3, 3)
    energies = []
    current = gray
    for _ in range(3):
        lap = F.conv2d(current, kernel, padding=1)
        energies.append(lap.square().mean(dim=(1, 2, 3)))
        if min(current.shape[-2:]) >= 32:
            current = F.avg_pool2d(current, 2, 2)
    return torch.stack(energies, dim=1).mean(dim=1)


@torch.no_grad()
def estimate_evssm_reliability(
    raw_images: Tensor,
    deblurred_images: Tensor,
    known_sharp: Tensor | None = None,
    confidence_floor: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a soft confidence for each RAW/deblurred image pair.

    Args:
        raw_images: ``[V,3,H,W]`` observations in ``[0,1]``.
        deblurred_images: aligned EVSSM outputs with the same shape.
        known_sharp: optional boolean ``[V]`` mask. These observations are
            trusted directly and receive confidence one.

    The score is a geometric conjunction of low-frequency/color fidelity,
    edge-direction agreement, clipping safety, and a saturating multiscale
    Laplacian gain. All terms are dimensionless and resolution-normalized.
    """
    if raw_images.shape != deblurred_images.shape or raw_images.ndim != 4:
        raise ValueError(
            "raw_images and deblurred_images must have identical [V,3,H,W] shapes"
        )
    raw = raw_images.float().clamp(0.0, 1.0)
    deblurred = deblurred_images.float().clamp(0.0, 1.0)
    h, w = raw.shape[-2:]
    low_size = (min(64, h), min(96, w))
    raw_low = F.interpolate(raw, size=low_size, mode="area")
    deblurred_low = F.interpolate(deblurred, size=low_size, mode="area")

    contrast = raw_low.flatten(1).std(dim=1).clamp_min(0.05)
    low_frequency_error = (raw_low - deblurred_low).abs().mean(dim=(1, 2, 3))
    color_fidelity = torch.exp(-low_frequency_error / contrast)

    raw_gray = _gray(raw_low)
    deblurred_gray = _gray(deblurred_low)
    raw_gx, raw_gy = _sobel(raw_gray)
    deb_gx, deb_gy = _sobel(deblurred_gray)
    dot = (raw_gx * deb_gx + raw_gy * deb_gy).flatten(1).sum(dim=1)
    raw_norm = (raw_gx.square() + raw_gy.square()).flatten(1).sum(dim=1).sqrt()
    deb_norm = (deb_gx.square() + deb_gy.square()).flatten(1).sum(dim=1).sqrt()
    edge_agreement = (dot / (raw_norm * deb_norm + 1e-8)).clamp(0.0, 1.0)

    raw_clip = ((raw <= 1.0 / 255.0) | (raw >= 254.0 / 255.0)).float().mean(
        dim=(1, 2, 3)
    )
    deb_clip = (
        (deblurred <= 1.0 / 255.0) | (deblurred >= 254.0 / 255.0)
    ).float().mean(dim=(1, 2, 3))
    clipping_safety = torch.exp(-8.0 * (deb_clip - raw_clip).clamp_min(0.0))

    raw_laplacian = _laplacian_energy(_gray(raw))
    deblurred_laplacian = _laplacian_energy(_gray(deblurred))
    log_laplacian_gain = torch.log(
        (deblurred_laplacian + 1e-8) / (raw_laplacian + 1e-8)
    )
    useful_gain = torch.tanh(log_laplacian_gain.clamp_min(0.0))

    fidelity = (
        color_fidelity.clamp_min(1e-6)
        * edge_agreement.clamp_min(1e-6)
        * clipping_safety.clamp_min(1e-6)
    ).pow(1.0 / 3.0)
    improvement = 0.25 + 0.75 * useful_gain
    confidence = confidence_floor + (1.0 - confidence_floor) * fidelity * improvement
    confidence = confidence.clamp(confidence_floor, 1.0)

    if known_sharp is not None:
        known_sharp = known_sharp.to(device=confidence.device, dtype=torch.bool)
        if known_sharp.shape != confidence.shape:
            raise ValueError("known_sharp must have shape [V]")
        confidence = torch.where(known_sharp, torch.ones_like(confidence), confidence)

    diagnostics = {
        "confidence": confidence,
        "fidelity": fidelity,
        "improvement": improvement,
        "color_fidelity": color_fidelity,
        "edge_agreement": edge_agreement,
        "clipping_safety": clipping_safety,
        "raw_laplacian": raw_laplacian,
        "deblurred_laplacian": deblurred_laplacian,
        "log_laplacian_gain": log_laplacian_gain,
    }
    return confidence, diagnostics

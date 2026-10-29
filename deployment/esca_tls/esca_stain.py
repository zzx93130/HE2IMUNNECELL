"""Macenko stain normalisation (Macenko et al., ISBI 2009) with per-slide stain estimation.

The H and E stain vectors and 99th-percentile concentrations are estimated once per slide from
optical-density pixels pooled over random full-resolution (level-0) patches inside artifact-free
tissue, then every patch of that slide is mapped onto the reference slide's stain matrix and
concentration scale. Unlike Reinhard colour-moment transfer, this does not depend on how much
of each tissue type a slide contains. Glass (all OD < beta) is left untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IO, ALPHA, BETA = 240.0, 1.0, 0.15  # standard Macenko settings (as in torchstain/staintools)


def optical_density(rgb):
    return -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1) / IO)


def fit(od):
    """Stain matrix (3x2, columns H then E) and 99th-percentile concentrations from OD pixels."""
    od = od[(od >= BETA).all(axis=1)]
    if len(od) < 1000:
        raise ValueError("Too few stained pixels for Macenko estimation")
    _, vectors = np.linalg.eigh(np.cov(od.T))
    plane = vectors[:, 1:3]
    projected = od @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    lo, hi = np.percentile(angles, [ALPHA, 100 - ALPHA])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    # Haematoxylin has the larger red-channel OD component.
    stains = np.stack([v1, v2], 1) if v1[0] > v2[0] else np.stack([v2, v1], 1)
    stains *= np.sign(stains.sum(axis=0, keepdims=True))  # non-negative orientation
    stains /= np.linalg.norm(stains, axis=0, keepdims=True)
    concentrations = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    return stains, np.percentile(concentrations, 99, axis=1)


def sample_od(slide_path, valid8, n_patches=60, size=512, seed=0, max_pixels=3_000_000):
    """OD pixels from random level-0 patches whose centres lie on valid tissue (8 µm QC grid)."""
    import openslide
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(valid8)
    pixels = []
    with openslide.OpenSlide(str(slide_path)) as slide:
        mpp = float(slide.properties["openslide.mpp-x"])
        for i in rng.choice(len(xs), size=min(n_patches, len(xs)), replace=False):
            cx, cy = (xs[i] + .5) * 8 / mpp, (ys[i] + .5) * 8 / mpp
            x = int(np.clip(cx - size / 2, 0, slide.dimensions[0] - size))
            y = int(np.clip(cy - size / 2, 0, slide.dimensions[1] - size))
            pixels.append(optical_density(np.asarray(slide.read_region((x, y), 0, (size, size)).convert("RGB"))))
    od = np.concatenate(pixels)
    if len(od) > max_pixels:
        od = od[rng.choice(len(od), max_pixels, replace=False)]
    return od


def slide_stains(slide_path, qc_folder, seed=0):
    """Fit Macenko parameters for one slide using its artifact-aware QC mask."""
    from esca_qc import artifact_masks
    qc_folder = Path(qc_folder)
    thumb = np.asarray(Image.open(qc_folder / "thumbnail_8um.jpg").convert("RGB"))
    tissue = np.load(qc_folder / "maps_4um.npz")["tissue_mask_2um"][::4, ::4]
    valid = artifact_masks(thumb, tissue)["valid"]
    return fit(sample_od(slide_path, valid, seed=seed))


def normalise(rgb, source, target):
    """Map an RGB patch from the source slide's stains onto the target's."""
    (src_stains, src_max), (tgt_stains, tgt_max) = source, target
    od = optical_density(rgb)
    concentrations = np.linalg.lstsq(src_stains, od.T, rcond=None)[0]
    concentrations *= (tgt_max / src_max)[:, None]
    out = IO * np.exp(-(tgt_stains @ concentrations))
    out = np.clip(out.T, 0, 255)
    glass = (od < BETA).all(axis=1)
    out[glass] = rgb.reshape(-1, 3)[glass]
    return out.reshape(rgb.shape).astype(np.uint8)

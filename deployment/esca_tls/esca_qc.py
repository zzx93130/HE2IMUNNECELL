"""Artifact-aware tissue QC on the 8 µm thumbnail: pen ink, dark folds/ink, grey non-tissue.

H&E pixels are pink-to-purple: hue in [250°, 360°] ∪ [0°, 25°]. Saturated pixels outside that
range are ink (green/blue/cyan/yellow). Translucent red marker overlapping tissue is not separable
from strongly eosinophilic tissue by colour or texture at 8 µm and is NOT removed. Very dark pixels are black ink or thick folds; unsaturated grey pixels are mounting medium,
bubbles or glass. Artifacts are dilated by 48 µm to exclude their margins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def artifact_masks(thumb_rgb, tissue8):
    hsv = np.asarray(Image.fromarray(thumb_rgb).convert("HSV"), dtype=np.float32) / 255
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    he_hue = (hue >= 250) | (hue <= 25)
    ink_colour = (sat > .15) & ~he_hue & (val < .97)
    dark = val < .30
    grey = (sat < .07) & (val < .90)
    artifact = ndimage.binary_opening(ink_colour | dark, iterations=1)
    artifact = ndimage.binary_dilation(artifact, iterations=6)  # 48 µm margin
    grey = ndimage.binary_opening(grey, iterations=2)
    tissue = tissue8[:thumb_rgb.shape[0], :thumb_rgb.shape[1]].astype(bool)
    valid = tissue & ~artifact & ~grey & (sat >= .07)
    valid = ndimage.binary_opening(valid, iterations=1)
    return {"valid": valid, "artifact": artifact & tissue, "grey": grey & tissue}


def overlay(thumb_rgb, masks):
    image = thumb_rgb.copy()
    image[masks["artifact"]] = (image[masks["artifact"]] * .4 + np.array([0, 0, 255]) * .6).astype(np.uint8)
    image[masks["grey"]] = (image[masks["grey"]] * .4 + np.array([255, 200, 0]) * .6).astype(np.uint8)
    edge = masks["valid"] ^ ndimage.binary_erosion(masks["valid"], iterations=2)
    image[edge] = (0, 160, 0)
    return image


def slide_qc(slide_folder):
    """Cached per-slide QC masks on the 8 µm thumbnail grid of the HookNet pass."""
    slide_folder = Path(slide_folder)
    cache = slide_folder / "qc" / "qc_8um.npz"
    if cache.exists():
        data = np.load(cache)
        return {name: data[name] for name in data.files}
    thumb = np.asarray(Image.open(slide_folder / "hooknet" / "thumbnail_8um.jpg").convert("RGB"))
    tissue8 = np.load(slide_folder / "hooknet" / "maps_4um.npz")["tissue_mask_2um"][::4, ::4]
    masks = artifact_masks(thumb, tissue8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **masks)
    Image.fromarray(overlay(thumb, masks)).save(slide_folder / "qc" / "qc_overlay.jpg", quality=85)
    return masks

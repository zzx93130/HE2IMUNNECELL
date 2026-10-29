"""Run published HookNet-TLS on the USZ Visium TIFFs as a reference method.

The plain ImageJ TIFFs are rewritten as tiled pyramidal TIFFs with their physical pixel size (so
OpenSlide reports MPP), then deployment/esca_tls/esca_hooknet.py processes them unchanged.
Run with .runtime/hooknet_tls/venv/bin/python.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "TLS_VISIUM_USZ"
OUT = ROOT / "outputs" / "visium_tls_validation"
SAMPLES = ["LC1", "LC2", "LC3", "LC4", "LC5", "KC1", "KC2", "KC3"]


def pyramid(sample):
    target = OUT / "pyramids" / f"{sample}.tiff"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(DATA / "tif_slides" / f"{sample}.tif") as tif:
        numerator, denominator = tif.pages[0].tags["XResolution"].value
        image = tif.pages[0].asarray()[..., :3]
    mpp = denominator / numerator
    levels = [image]
    while min(levels[-1].shape[:2]) > 1024:
        levels.append(levels[-1][::2, ::2])
    partial = target.with_suffix(".partial.tiff")
    with tifffile.TiffWriter(partial, bigtiff=True) as writer:
        for index, level in enumerate(levels):
            scale = mpp * 2 ** index
            writer.write(np.ascontiguousarray(level), photometric="rgb", tile=(256, 256), compression="deflate",
                         subfiletype=0 if index == 0 else 1, resolution=(1e4 / scale, 1e4 / scale),
                         resolutionunit="CENTIMETER", metadata=None)
    partial.replace(target)
    return target


def main():
    for sample in SAMPLES:
        slide = pyramid(sample)
        out = OUT / "hooknet" / sample
        if (out / "summary.json").exists():
            continue
        subprocess.run([sys.executable, str(ROOT / "deployment" / "esca_tls" / "esca_hooknet.py"),
                        "--slide", str(slide), "--out", str(out)], check=True)


if __name__ == "__main__":
    main()

"""Published HookNet-TLS on the TNBC_ST H&E arrays (imagesHD), as a reference/baseline arm.

Each imagesHD jpg is rewritten as a temporary tiled pyramidal TIFF with its physical pixel size (scale.json),
then deployment/esca_tls/esca_hooknet.process() runs unchanged (0.5 µm input, 2.0 µm context, Otsu tissue
mask, published postprocessing). Its module-level output paths are redirected to this analysis so that no
file is written into outputs/esca_tls_20261019 (model-audit copy and temporary folder live here).
Outputs: outputs/tnbc_st_20261101/hooknet/<array>/ (summary.json, maps_4um.npz, polygons, thumbnail).
Run with .runtime/hooknet_tls/venv/bin/python under deployment/ablations/gpu_locked.sh (TensorFlow uses the GPU).
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import tnbc_common as C  # noqa: E402

OUT = C.OUT / "hooknet"


def pyramid(entry, mpp, target):
    import cv2
    import tifffile
    image = C.read_image(entry["hd"])
    levels = [image]
    while min(levels[-1].shape[:2]) > 1024:
        levels.append(cv2.resize(levels[-1], (levels[-1].shape[1] // 2, levels[-1].shape[0] // 2), interpolation=cv2.INTER_AREA))
    partial = target.with_suffix(".partial.tiff")
    with tifffile.TiffWriter(partial, bigtiff=True) as writer:
        for index, level in enumerate(levels):
            scale = mpp * image.shape[1] / level.shape[1]
            writer.write(np.ascontiguousarray(level), photometric="rgb", tile=(256, 256), compression="deflate",
                         subfiletype=0 if index == 0 else 1, resolution=(1e4 / scale, 1e4 / scale),
                         resolutionunit="CENTIMETER", metadata=None)
    partial.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", nargs="+", required=True)
    args = parser.parse_args()
    import esca_hooknet as H
    from hooknet_tls_native import load_model
    import tensorflow as tf
    H.OUT = OUT / "_meta"          # infer_and_verify writes model_verification.json under OUT / hooknet_meta
    H.TMP_ROOT = OUT / "_tmp"
    audit = OUT / "_meta" / "hooknet_meta" / "source_audit"
    audit.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    with tf.device("/CPU:0"):
        model, config, _ = load_model(audit)
    state = {}
    inventory = json.loads((C.OUT / "inventory.json").read_text())["arrays"]
    mpp = json.loads((C.OUT / "scale.json").read_text())["hd_mpp"]
    for array in args.arrays:
        out = OUT / array
        if (out / "summary.json").exists():
            continue
        if C.free_gb() < C.MIN_FREE_GB:
            print(f"STOP: {C.free_gb():.1f} GB free", flush=True)
            break
        tmp = OUT / "_tmp_pyramid"
        tmp.mkdir(parents=True, exist_ok=True)
        target = tmp / f"{array}.tiff"
        try:
            pyramid(inventory[array], mpp, target)
            H.process(target, model, config, state, out)
        finally:
            target.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

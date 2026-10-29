"""Test Macenko stain normalisation (to the LUAD reference) on slides where Reinhard left TLS B-empty."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import esca_windows as W
import esca_diagnose as D
from esca_common import OUT, save_json
from slide_io import open_slide


def od(rgb):
    return -np.log((rgb.astype(np.float64) + 1) / 256)


def macenko_fit(rgb, beta=.15, alpha=1):
    o = od(rgb).reshape(-1, 3)
    o = o[(o > beta).all(1)]
    _, vecs = np.linalg.eigh(np.cov(o.T))
    v = vecs[:, [1, 2]]
    proj = o @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    a = v @ np.array([np.cos(lo), np.sin(lo)]); b = v @ np.array([np.cos(hi), np.sin(hi)])
    stains = np.array([a, b]) if a[0] > b[0] else np.array([b, a])  # haematoxylin first
    stains = stains * np.sign(stains.sum(1, keepdims=True))
    stains /= np.linalg.norm(stains, axis=1, keepdims=True)
    conc = np.linalg.lstsq(stains.T, o.T, rcond=None)[0]
    return stains, np.percentile(conc, 99, axis=1)


def overview(slide_path):
    with open_slide(slide_path) as s:
        mpp = float(s.properties["openslide.mpp-x"])
        level = min(range(s.level_count), key=lambda l: abs(s.level_downsamples[l] * mpp - 2))
        return np.asarray(s.read_region((0, 0), level, s.level_dimensions[level]).convert("RGB"))


def main():
    target = macenko_fit(overview(W.REFERENCE_SLIDE))
    report = {"target_stains": target[0].tolist(), "target_maxC": target[1].tolist()}
    for barcode in ["TCGA-L5-A4OJ-01Z-00-DX1", "TCGA-JY-A938-01Z-00-DX1", "TCGA-L5-A4OE-01Z-00-DX1"]:
        folder = OUT / "slides" / barcode
        slide = Path(json.loads((folder / "hooknet" / "summary.json").read_text())["slide"])
        source = macenko_fit(overview(slide))

        def transform(patch, source=source):
            o = od(patch).reshape(-1, 3)
            conc = np.linalg.lstsq(source[0].T, o.T, rcond=None)[0]
            conc *= (target[1] / source[1])[:, None]
            out = 256 * np.exp(-(target[0].T @ conc)) - 1
            return np.clip(out.T.reshape(patch.shape), 0, 255).astype(np.uint8)
        windows = pd.read_csv(folder / "he2cell" / "windows.csv")
        windows = windows[windows.category == "tls"].head(2)
        cells = D.run(slide, windows, f"{barcode}_macenko", transform)
        reinhard = pd.read_csv(folder / "he2cell" / "cells.csv.gz")
        for w in windows.itertuples():
            report[f"{barcode} {w.window_id}"] = {
                "reinhard": reinhard[reinhard.window_id == w.window_id].celltype.value_counts().to_dict(),
                "macenko": cells[cells.window_id == w.window_id].celltype.value_counts().to_dict(),
                "source_stains": source[0].round(3).tolist()}
    save_json(report, OUT / "diagnostics" / "macenko_test.json")
    for k, v in report.items():
        if "reinhard" in v: print(k, "Reinhard", v["reinhard"], "| Macenko", v["macenko"])


if __name__ == "__main__":
    main()

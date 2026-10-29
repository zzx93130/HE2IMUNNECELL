"""Gate-A chromogen QC: which chromogen is CD3 and which is CD20, decided from morphology.

For each processed IHC slide: aggregates of the RED chromogen (blob density >= 2,000/mm² after sigma 25 µm,
>= 20,000 µm²) and of the BROWN chromogen (same rule) are found; inside each aggregate's core (inner half by
distance) and in a 50 µm rim outside it, red and brown blob densities are measured. CD20 (B follicles)
predicts: red aggregates compact, red density core > rim, brown enriched in the rim relative to the core.
Writes qc/stage0/chromogen_assignment.{png,json}.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402

GRID = 8.0


def grids(sample):
    s = json.loads((C.OUT / "ihc" / f"TNBC{sample}" / "summary.json").read_text())
    p = np.load(C.OUT / "ihc" / f"TNBC{sample}" / "points.npz")
    m = np.load(C.OUT / "ihc" / f"TNBC{sample}" / "maps_2um.npz")
    l0 = s["mpp_level0"]
    W, H = s["dimensions_level0"]
    nx, ny = int(np.ceil(W * l0 / GRID)), int(np.ceil(H * l0 / GRID))
    out = {}
    for key, name in [("CD3", "brown"), ("CD20", "red")]:
        pts = p[key]
        gx = (pts[:, 0] * l0 / GRID).astype(int).clip(0, nx - 1)
        gy = (pts[:, 1] * l0 / GRID).astype(int).clip(0, ny - 1)
        out[name] = np.bincount(gy * nx + gx, minlength=nx * ny).reshape(ny, nx).astype(float)
    tissue = cv2.resize(m["tissue"], (nx, ny), interpolation=cv2.INTER_NEAREST) > 0
    return out, tissue, s


def aggregates(counts, tissue, threshold=2000, sigma=25, min_area=20000):
    d = ndimage.gaussian_filter(counts, sigma / GRID, mode="constant") / (GRID ** 2 / 1e6)
    labels, n = ndimage.label((d >= threshold) & tissue)
    return [labels == i for i in range(1, n + 1) if (labels == i).sum() * GRID ** 2 >= min_area]


def profile(objs, counts, tissue):
    rows = []
    ring = int(round(50 / GRID))
    for obj in objs:
        dist = ndimage.distance_transform_edt(obj)
        core = obj & (dist >= .5 * dist.max())
        rim = ndimage.binary_dilation(obj, iterations=ring) & ~obj & tissue
        area = lambda m: m.sum() * GRID ** 2 / 1e6  # noqa: E731
        r = {"area_um2": float(obj.sum() * GRID ** 2)}
        for name, c in counts.items():
            r[f"{name}_core"] = c[core].sum() / max(area(core), 1e-9)
            r[f"{name}_rim"] = c[rim].sum() / max(area(rim), 1e-9)
        lab = obj.astype(np.uint8)
        contours, _ = cv2.findContours(lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        r["compactness"] = float(4 * np.pi * obj.sum() / max(perimeter, 1) ** 2)
        rows.append(r)
    return rows


def main():
    samples = sorted(int(p.name[4:]) for p in (C.OUT / "ihc").glob("TNBC*") if (p / "summary.json").exists())
    result = {}
    for sample in samples:
        counts, tissue, s = grids(sample)
        entry = {"label_CD2": s["label_CD2_instead_of_CD3"], "total_brown": int(counts["brown"].sum()), "total_red": int(counts["red"].sum())}
        for name in ["red", "brown"]:
            rows = profile(aggregates(counts[name], tissue), counts, tissue)
            if rows:
                entry[f"{name}_aggregates"] = {
                    "n": len(rows), "median_compactness": float(np.median([r["compactness"] for r in rows])),
                    "median_red_core_over_rim": float(np.median([r["red_core"] / max(r["red_rim"], 1) for r in rows])),
                    "median_brown_rim_over_core": float(np.median([r["brown_rim"] / max(r["brown_core"], 1) for r in rows])),
                    "median_brown_share_core": float(np.median([r["brown_core"] / max(r["brown_core"] + r["red_core"], 1) for r in rows])),
                    "median_brown_share_rim": float(np.median([r["brown_rim"] / max(r["brown_rim"] + r["red_rim"], 1) for r in rows])),
                }
            else:
                entry[f"{name}_aggregates"] = {"n": 0}
        result[f"TNBC{sample}"] = entry
    C.save_json(result, C.OUT / "qc" / "stage0" / "chromogen_assignment.json")
    print(json.dumps(result, indent=1))
    # Figure: slide-45 thumbnail with dots, follicle zoom, deconvolution panel, per-slide core/rim bars.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 11))
    qc = cv2.cvtColor(cv2.imread(str(C.OUT / "ihc" / "TNBC45" / "qc_4um.jpg")), cv2.COLOR_BGR2RGB)
    for i, (img, title) in enumerate([
            (qc, "TNBC45 IHC (left RGB, right: brown = DAB blobs, red = red-chromogen blobs), 4 µm/px"),
            (cv2.cvtColor(cv2.imread(str(C.OUT / "qc" / "stage0" / "ihc45_zoom_l1.jpg")), cv2.COLOR_BGR2RGB),
             "TNBC45 follicle, 0.46 µm/px: red fills the follicle core, brown surrounds it"),
            (cv2.cvtColor(cv2.imread(str(C.OUT / "qc" / "stage0" / "ihc_deconv_45_follicle.jpg")), cv2.COLOR_BGR2RGB),
             "Deconvolution: RGB | DAB conc + blobs | red conc + blobs | haematoxylin")]):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    names = [k for k, v in result.items() if v["red_aggregates"]["n"]]
    x = np.arange(len(names))
    a = [result[k]["red_aggregates"]["median_brown_share_core"] for k in names]
    b = [result[k]["red_aggregates"]["median_brown_share_rim"] for k in names]
    ax.bar(x - .2, a, .4, label="brown share, red-aggregate core")
    ax.bar(x + .2, b, .4, label="brown share, 50 µm rim")
    ax.set_xticks(x, [f"{k}\n({result[k]['red_aggregates']['n']} red agg.)" + (" CD2" if result[k]["label_CD2"] else "") for k in names])
    ax.set_ylabel("brown / (brown + red) blobs")
    ax.legend()
    ax.set_title("Red aggregates: brown (DAB) is depleted in the core and enriched in the rim -> red = CD20, brown = CD3")
    fig.tight_layout()
    fig.savefig(C.OUT / "qc" / "stage0" / "chromogen_assignment.png", dpi=90)


if __name__ == "__main__":
    main()

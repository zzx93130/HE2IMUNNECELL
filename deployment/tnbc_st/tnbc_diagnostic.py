"""Descriptive domain-shift diagnostic (not an analysis change): HE2CELL label composition per pathologist class on
frozen-section ST H&E (TNBC_ST) vs FFPE references.

TNBC: every HE2CELL cell takes the annotation class at its position (annotation PNG, nearest pixel); per patient and
class: label shares (Other/T/Macrophage/B) and nuclei per mm² of that class's area inside the tissue mask.
Classes: Lymphoid nodule, Lymphocyte, High TIL stroma, Tumor, Low TIL stroma. Two QC crops per class (seeded).
FFPE references (same checkpoint, tau 0.5 prior, D4):
  - TCGA-ESCA test windows: inside HookNet-TLS, outside TLS in 'random' windows, 'dense' non-TLS windows
    (outputs/esca_tls_20261019/he2cell_ft_tau05 cells; densities on the esca_analysis valid grid).
  - USZ Visium (FFPE, 0.79 µm/px upsampled): spot labels TLS / INFL / TUM / NOR (finetuned_tau05 spot table).
Writes analysis/diagnostic/{tnbc_per_patient_class.csv, tnbc_pooled.csv, esca_reference.csv, visium_reference.csv,
summary.json} and qc/diagnostic/<class>_<k>.jpg.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import tnbc_common as C  # noqa: E402
import tnbc_cells as TC  # noqa: E402
import tnbc_grids as G  # noqa: E402

OUT = C.OUT / "analysis" / "diagnostic"
QC = C.OUT / "qc" / "diagnostic"
CLASSES = ["Lymphoid nodule", "Lymphocyte", "High TIL stroma", "Tumor", "Low TIL stroma"]
TYPES = ["Other", "T", "Macrophage", "B"]


def tnbc(arrays):
    names = list(TC.ANNOTATION_COLOURS)
    rows, crops = [], {c: [] for c in CLASSES}
    inv = G.inventory()
    mpp = G.scale()["hd_mpp"]
    for array in arrays:
        entry = inv[array]
        labels = TC.annotation_labels(entry)
        f = labels.shape[0] / G.HD
        cells = pd.read_csv(C.OUT / "cells" / array / "cells.csv.gz", usecols=["x_hd", "y_hd", "celltype"])
        lab = labels[np.clip((cells.y_hd * f).astype(int), 0, labels.shape[0] - 1), np.clip((cells.x_hd * f).astype(int), 0, labels.shape[1] - 1)]
        tissue = cv2.imread(str(C.OUT / "cells" / array / "tissue_mask.png"), 0) > 0
        tl = cv2.resize(labels.astype(np.int16), (tissue.shape[1], tissue.shape[0]), interpolation=cv2.INTER_NEAREST)
        px_mm2 = (G.HD / tissue.shape[0] * mpp) ** 2 / 1e6
        for cls in CLASSES:
            k = names.index(cls)
            sel = lab == k
            area = float(((tl == k) & tissue).sum() * px_mm2)
            if sel.sum() < 20 or area <= 0:
                continue
            vc = cells.celltype[sel].value_counts()
            rows.append({"array": array, "case_id": f"TNBC{entry['sample']}", "class": cls, "cells": int(sel.sum()), "area_mm2": area,
                         "nuclei_per_mm2": sel.sum() / area, **{f"share_{t}": vc.get(t, 0) / sel.sum() for t in TYPES}})
            crops[cls].append((array, entry, cells[sel]))
    per = pd.DataFrame(rows)
    rng = np.random.default_rng(20261101)
    QC.mkdir(parents=True, exist_ok=True)
    colours = {"T": (0, 170, 0), "B": (230, 0, 0), "Macrophage": (240, 190, 0)}
    for cls, items in crops.items():
        if not items:
            continue
        for k, i in enumerate(rng.choice(len(items), min(2, len(items)), replace=False)):
            array, entry, c = items[i]
            centre = c.iloc[rng.integers(len(c))]
            image = C.read_image(entry["hd"])
            h = int(100 / mpp)
            y0, x0 = max(int(centre.y_hd) - h, 0), max(int(centre.x_hd) - h, 0)
            crop = image[y0:y0 + 2 * h, x0:x0 + 2 * h].copy()
            del image
            allc = pd.read_csv(C.OUT / "cells" / array / "cells.csv.gz", usecols=["x_hd", "y_hd", "celltype"])
            allc = allc[(allc.x_hd >= x0) & (allc.x_hd < x0 + 2 * h) & (allc.y_hd >= y0) & (allc.y_hd < y0 + 2 * h)]
            over = crop.copy()
            for t, col in colours.items():
                for xx, yy in zip(allc[allc.celltype == t].x_hd - x0, allc[allc.celltype == t].y_hd - y0):
                    cv2.circle(over, (int(xx), int(yy)), 7, col, 2)
            panel = np.concatenate([crop, over], 1)
            cv2.putText(panel, f"{cls} {array} (T green, B red, Mac yellow; other unmarked)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2)
            cv2.imwrite(str(QC / f"{cls.replace(' ', '_')}_{k + 1}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return per


def esca():
    import esca_analysis as A
    A.RESULT = "he2cell_ft_tau05"
    windows, inputs = A.load_cohort()
    rows = []
    for w in windows[windows.split == "test"].itertuples():
        x = inputs[(w.barcode, w.window_id)]
        ref = x["reference"] & x["valid"]
        regions = {"HookNet TLS": ref} if w.category == "tls" else {}
        if w.category == "random":
            regions["random window, outside TLS"] = x["valid"] & ~x["raw_tls"]
        if w.category == "dense" and w.hooknet_label == "none":
            regions["dense non-TLS window"] = x["valid"]
        for name, m in regions.items():
            if m.sum() == 0:
                continue
            counts = {t: x["counts"][t][m].sum() for t in TYPES}
            total = sum(counts.values())
            if total < 20:
                continue
            rows.append({"case_id": w.case_id, "region": name, "cells": total, "area_mm2": m.sum() * A.GRID_UM ** 2 / 1e6,
                         "nuclei_per_mm2": total / (m.sum() * A.GRID_UM ** 2 / 1e6), **{f"share_{t}": counts[t] / total for t in TYPES}})
    return pd.DataFrame(rows)


def visium():
    spots = pd.read_csv(C.ROOT / "outputs" / "visium_tls_validation" / "analysis" / "spot_table.csv.gz")
    spots = spots[(spots.classifier == "finetuned_tau05") & (spots.in_tissue == 1)]
    group = spots.label.map(lambda l: "TLS" if l in ("ETLS", "MTLS", "GC") else l)
    spots = spots.assign(group=group)[group.isin(["TLS", "INFL", "TUM", "NOR"])]
    rows = []
    for (sample, g), f in spots.groupby(["sample", "group"]):
        total = f.cells.sum()
        rows.append({"case_id": sample, "region": g, "cells": total, "share_B": f.B.sum() / total, "share_T": f["T"].sum() / total,
                     "share_Other_or_Macrophage": 1 - (f.B.sum() + f["T"].sum()) / total,
                     "nuclei_per_mm2": total / (len(f) * np.pi * 27.5 ** 2 / 1e6)})
    return pd.DataFrame(rows)


def pooled(frame, key):
    share_cols = [c for c in frame if c.startswith("share_")]
    out = frame.groupby(key)[list(frame.columns)].apply(lambda f: pd.Series({"patients": f.case_id.nunique(), "cells": f.cells.sum(),
                                                         **{c: np.average(f[c], weights=f.cells) for c in share_cols},
                                                         "median_patient_nuclei_per_mm2": f.groupby("case_id").nuclei_per_mm2.median().median()}))
    return out.reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = G.inventory()
    arrays = [a for a, e in inv.items() if e["annotated"] and (C.OUT / "cells" / a / "cells.csv.gz").exists()]
    per = tnbc(arrays)
    per.to_csv(OUT / "tnbc_per_patient_class.csv", index=False)
    tp = pooled(per, "class")
    tp.to_csv(OUT / "tnbc_pooled.csv", index=False)
    summary = {"tnbc_arrays": len(arrays), "tnbc_frozen_section_by_class": tp.to_dict("records")}
    ref_e = C.OUT / "analysis" / "diagnostic" / "esca_reference.csv"
    if not ref_e.exists():
        esca().to_csv(ref_e, index=False)
    e = pd.read_csv(ref_e)
    summary["esca_ffpe_test_reference"] = pooled(e, "region").to_dict("records")
    v = visium()
    v.to_csv(OUT / "visium_reference.csv", index=False)
    summary["visium_ffpe_reference"] = pooled(v, "region").to_dict("records")
    C.save_json(summary, OUT / "summary.json")
    for k in ["tnbc_frozen_section_by_class", "esca_ffpe_test_reference", "visium_ffpe_reference"]:
        print(k)
        print(pd.DataFrame(summary[k]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

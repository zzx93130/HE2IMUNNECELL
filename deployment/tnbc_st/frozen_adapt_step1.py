"""Step 1 (CPU): segmentation vs appearance diagnosis inside pathologist Lymphoid-nodule and Lymphocyte regions.

TNBC: completed priority arrays (cells.csv.gz); mask-based metrics only on arrays whose stage-1 Cellpose masks were captured.
ESCA reference: TCGA-ESCA dev cropbank crops inside HookNet-TLS (outputs/esca_tls_20261019/cropbank_dev) for crop/mask
metrics; nuclei density from outputs/tnbc_st_20261101/analysis/diagnostic/esca_reference.csv (HookNet TLS, test windows).
Writes outputs/tnbc_frozen_adapt_20261101/step1/{per_cell_tnbc.csv.gz, per_cell_esca.csv.gz, density.csv, metrics.json,
overlay_*.jpg, crops_tnbc_vs_esca.jpg}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_adapt_common as F  # noqa: E402
import tnbc_cells as TC  # noqa: E402

OUT = F.OUT / "step1"
CLASSES = ["Lymphoid nodule", "Lymphocyte"]
NAMES = list(TC.ANNOTATION_COLOURS)
N_CROPS = 3000
rng = np.random.default_rng(F.SEED)


def crop_metrics(rgb, mask):
    from skimage.color import rgb2hed
    from skimage.measure import regionprops
    m = mask > 0
    rows = []
    for r, k in zip(rgb, m):
        hed = rgb2hed(r)
        props = regionprops(k.astype(np.uint8))
        rows.append({"crop_mean_intensity": float(r.mean()),
                     "nucleus_mean_intensity": float(r[k].mean()) if k.any() else np.nan,
                     "H_OD_nucleus": float(hed[..., 0][k].mean()) if k.any() else np.nan,
                     "E_OD_background": float(hed[..., 1][~k].mean()),
                     "crop_mask_area_px": int(k.sum()),
                     "crop_solidity": float(props[0].solidity) if props else np.nan,
                     "touches_border": bool(k[0].any() or k[-1].any() or k[:, 0].any() or k[:, -1].any())})
    return pd.DataFrame(rows)


def class_of(array, cells):
    entry = F.inventory()[array]
    labels = TC.annotation_labels(entry)
    f = labels.shape[0] / 31744
    lab = labels[np.clip((cells.y_hd * f).astype(int), 0, labels.shape[0] - 1), np.clip((cells.x_hd * f).astype(int), 0, labels.shape[1] - 1)]
    return labels, lab


def tnbc():
    dens, percell = [], []
    masked = set(F.done_arrays())
    mpp = F.hd_mpp()
    overlay_pool = []
    for array in F.priority():
        if not (F.MAIN / "cells" / array / "cells.csv.gz").exists():
            continue
        entry = F.inventory()[array]
        if not entry.get("annotation"):
            continue
        case = f"TNBC{entry['sample']}"
        cells = F.load_cells(array)
        labels, lab = class_of(array, cells)
        tissue = cv2.imread(str(F.MAIN / "cells" / array / "tissue_mask.png"), 0) > 0
        tl = cv2.resize(labels.astype(np.int16), (tissue.shape[1], tissue.shape[0]), interpolation=cv2.INTER_NEAREST)
        px_um2 = (31744 / tissue.shape[0] * mpp) ** 2
        for cls in CLASSES:
            k = NAMES.index(cls)
            sel = lab == k
            area = float(((tl == k) & tissue).sum() * px_um2)
            if sel.sum() < 20 or area <= 0:
                continue
            a = cells.area_um2[sel]
            dens.append({"array": array, "case_id": case, "class": cls, "cells": int(sel.sum()), "area_um2": area,
                         "nuclei_per_1000um2": 1000 * sel.sum() / area, "median_area_um2": float(a.median()),
                         "share_area_gt_2x_median": float((a > 2 * a.median()).mean()),
                         "share_B": float((cells.celltype[sel] == "B").mean()), "share_Other": float((cells.celltype[sel] == "Other").mean())})
        if array in masked:
            sel = np.flatnonzero(np.isin(lab, [NAMES.index(c) for c in CLASSES]))
            if len(sel) == 0:
                continue
            pick = np.sort(rng.choice(sel, min(N_CROPS, len(sel)), replace=False))
            sub = cells.iloc[pick].reset_index(drop=True)
            crops, found, props = F.build_crops(array, sub, with_props=True)
            rgb, mask = crops[1.0]
            m = crop_metrics(rgb[found], mask[found])
            m.insert(0, "class", [NAMES[i] for i in lab[pick][found]])
            m.insert(0, "case_id", case)
            m.insert(0, "array", array)
            m["area_um2"] = sub.area_um2.to_numpy()[found]
            m["solidity"] = props[found, 1]
            m["celltype"] = sub.celltype.to_numpy()[found]
            percell.append(m)
            np.savez_compressed(OUT / f"_crops_{array}.npz", rgb=rgb[found][:64], mask=mask[found][:64])
            overlay_pool.append((array, sub[found]))
    return pd.DataFrame(dens), (pd.concat(percell) if percell else pd.DataFrame()), overlay_pool


def esca():
    from PIL import Image
    bank = F.ROOT / "outputs" / "esca_tls_20261019" / "cropbank_dev"
    rows, keep = [], []
    for d in sorted(bank.glob("TCGA-*")):
        c = pd.read_csv(d / "cells.csv")
        c = c[c.in_hooknet_tls.astype(bool)]
        if len(c) == 0:
            continue
        c = c.iloc[np.sort(rng.choice(len(c), min(N_CROPS // 12 + 1, len(c)), replace=False))]
        rgb = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in c.cell_image_crop_path])
        mask = np.stack([np.asarray(Image.open(p).convert("L")) for p in c.cell_mask_crop_path])
        m = crop_metrics(rgb, mask)
        m.insert(0, "case_id", d.name[:12])
        m["area_um2"] = c.area_pixels.to_numpy() * 0.0625
        rows.append(m)
        keep.append((rgb[:8], mask[:8]))
    ref = pd.read_csv(F.MAIN / "analysis" / "diagnostic" / "esca_reference.csv")
    ref = ref[ref.region == "HookNet TLS"].assign(nuclei_per_1000um2=lambda f: f.nuclei_per_mm2 / 1000)
    return pd.concat(rows), ref, keep


def boot(values_by_patient, n=2000, seed=F.SEED, stat=np.mean):
    v = np.asarray(values_by_patient, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"value": np.nan, "CI95": [np.nan, np.nan], "patients": 0}
    r = np.random.default_rng(seed)
    b = np.array([stat(v[r.integers(0, len(v), len(v))]) for _ in range(n)])
    return {"value": float(stat(v)), "CI95": np.percentile(b, [2.5, 97.5]).tolist(), "patients": int(len(v))}


def merged_rate(f):
    med = f.area_um2.median()
    return float(((f.crop_solidity < 0.85) & (f.area_um2 > 2 * med)).mean())


def outline(img, mask, colour=(0, 255, 0)):
    out = img.copy()
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, colour, 1)
    return out


def overlays(pool):
    """6 dense 384-px (96 um) windows at 0.25 um/px: raw | Cellpose instance outlines; densest Lymphoid-nodule cells."""
    import tifffile
    made = 0
    per_array = max(1, int(np.ceil(6 / max(len(pool), 1))))
    for array, cells in pool:
        image = F.C.read_image(F.inventory()[array]["hd"])
        s = F.hd_mpp() / F.TARGET_MPP
        pidx = pd.read_csv(F.CAPTURE / array / f"{array}_patch_index.csv").set_index("num_patch")
        top = cells.patch_id.value_counts().index[:per_array]
        for pid in top:
            rec = pidx.loc[pid]
            rgb = F.patch_rgb(image, rec.x, rec.y, s)
            mask = tifffile.imread(F.CAPTURE / array / "patch_masks" / array / f"{Path(rec.path).stem}_cp_mask.tif").astype(np.int32)
            c = cells[cells.patch_id == pid]
            cx, cy = int(np.median(c.centroid_x_global - rec.x)), int(np.median(c.centroid_y_global - rec.y))
            x0, y0 = int(np.clip(cx - 192, 0, 1024 - 384)), int(np.clip(cy - 192, 0, 1024 - 384))
            win, mw = rgb[y0:y0 + 384, x0:x0 + 384], mask[y0:y0 + 384, x0:x0 + 384]
            edge = np.zeros(mw.shape, bool)
            edge[:-1] |= mw[:-1] != mw[1:]
            edge[:, :-1] |= mw[:, :-1] != mw[:, 1:]
            ov = win.copy()
            ov[edge & (mw > 0)] = (0, 255, 0)
            nuc = len(np.unique(mw)) - 1
            panel = np.concatenate([win, np.full((384, 6, 3), 255, np.uint8), ov], 1)
            panel = cv2.resize(panel, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
            cv2.putText(panel, f"{array} patch {pid}: {nuc} instances in 96x96 um ({1000 * nuc / 96 ** 2:.1f}/1000um2)", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 2)
            made += 1
            cv2.imwrite(str(OUT / f"overlay_{made}_{array}_p{pid}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            if made >= 6:
                return made
        del image
    return made


def montage(esca_keep):
    tn = [np.load(p) for p in sorted(OUT.glob("_crops_*.npz"))]
    row = lambda rgb, mask: np.concatenate([outline(r, m) for r, m in zip(rgb, mask)], 1)  # noqa: E731
    t_rgb = np.concatenate([t["rgb"][:16] for t in tn])[:16]
    t_mask = np.concatenate([t["mask"][:16] for t in tn])[:16]
    e_rgb = np.concatenate([k[0] for k in esca_keep])[:16]
    e_mask = np.concatenate([k[1] for k in esca_keep])[:16]
    rows = [row(t_rgb[:8], t_mask[:8]), row(t_rgb[8:16], t_mask[8:16]), np.full((8, 512, 3), 255, np.uint8),
            row(e_rgb[:8], e_mask[:8]), row(e_rgb[8:16], e_mask[8:16])]
    img = cv2.resize(np.concatenate(rows, 0), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "crops_tnbc_vs_esca.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dens, tc, pool = tnbc()
    ec, eref, keep = esca()
    dens.to_csv(OUT / "density.csv", index=False)
    tc.to_csv(OUT / "per_cell_tnbc.csv.gz", index=False)
    ec.to_csv(OUT / "per_cell_esca.csv.gz", index=False)
    res = {"tnbc_arrays_density": sorted(dens.array.unique().tolist()), "tnbc_arrays_masks": sorted(tc.array.unique().tolist()) if len(tc) else [],
           "esca_patients_crops": int(ec.case_id.nunique()), "metrics": {}}
    met = res["metrics"]
    # density (per patient: cells/area pooled over LN+Lymphocyte)
    g = dens.groupby("case_id").apply(lambda f: 1000 * f.cells.sum() / f.area_um2.sum())
    met["nuclei_per_1000um2"] = {"TNBC_LN_plus_Lymphocyte": boot(g), "ESCA_HookNet_TLS": boot(eref.groupby("case_id").nuclei_per_1000um2.median())}
    for cls in CLASSES:
        met["nuclei_per_1000um2"][f"TNBC_{cls}"] = boot(dens[dens["class"] == cls].groupby("case_id").nuclei_per_1000um2.mean())
    met["median_area_um2"] = {"TNBC": boot(dens.groupby("case_id").median_area_um2.mean()), "ESCA_TLS": boot(ec.groupby("case_id").area_um2.median())}
    for name, f in [("TNBC", tc), ("ESCA_TLS", ec)]:
        if len(f) == 0:
            continue
        by = f.groupby("case_id")
        met.setdefault("merged_nucleus_rate", {})[name] = boot(by.apply(merged_rate))
        met.setdefault("border_touch_fraction", {})[name] = boot(by.touches_border.mean())
        met.setdefault("crop_mean_intensity", {})[name] = boot(by.crop_mean_intensity.mean())
        met.setdefault("nucleus_mean_intensity", {})[name] = boot(by.nucleus_mean_intensity.mean())
        met.setdefault("H_OD_nucleus", {})[name] = boot(by.H_OD_nucleus.mean())
        met.setdefault("E_OD_background", {})[name] = boot(by.E_OD_background.mean())
        met.setdefault("crop_solidity_median", {})[name] = boot(by.crop_solidity.median())
    if len(tc):
        met["TNBC_full_instance_solidity_lt_0.85"] = boot(tc.groupby("case_id").solidity.apply(lambda s: (s < .85).mean()))
    # ratios + verdict (preregistered rule)
    r = lambda k: met[k]["TNBC" if "TNBC" in met[k] else "TNBC_LN_plus_Lymphocyte"]["value"] / max(met[k]["ESCA_TLS" if "ESCA_TLS" in met[k] else "ESCA_HookNet_TLS"]["value"], 1e-3)  # noqa: E731 (ESCA floor 0.001)
    ratios = {k: r(k) for k in ["nuclei_per_1000um2", "median_area_um2", "merged_nucleus_rate", "border_touch_fraction",
                                "crop_mean_intensity", "nucleus_mean_intensity", "H_OD_nucleus"] if k in met}
    res["ratios_TNBC_over_ESCA"] = ratios
    seg = (ratios.get("nuclei_per_1000um2", 1) < 0.6) or (ratios.get("merged_nucleus_rate", 1) > 2) or (ratios.get("border_touch_fraction", 1) > 2)
    app = any(abs(ratios.get(k, 1) - 1) > 0.15 for k in ["crop_mean_intensity", "nucleus_mean_intensity", "H_OD_nucleus"])
    res["verdict_rule_flags"] = {"segmentation": bool(seg), "appearance": bool(app)}
    res["verdict"] = "both" if seg and app else "segmentation" if seg else "appearance" if app else "neither flagged"
    if pool:
        res["overlays"] = overlays(pool)
        montage(keep)
    F.C.save_json(res, OUT / "metrics.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

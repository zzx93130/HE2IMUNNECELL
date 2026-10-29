"""External validation of HE2CELL TLS identification against pathologist Visium spot labels.

Truth: manual_anno_tls per 55 µm spot (USZ Visium, Zenodo 14620362). TLS = ETLS + MTLS + GC;
negatives = NOR, TUM, INFL; LN reported separately; UNASSIGNED excluded.
All detector parameters are the ones frozen on TCGA-ESCA dev patients; nothing is tuned here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage, sparse, stats
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import visium_cells as V  # noqa: E402
import esca_analysis as A  # noqa: E402

OUT = V.OUT
GRID = A.GRID_UM
SPOT_RADIUS_UM = 27.5
POSITIVE, NEGATIVE = {"ETLS", "MTLS", "GC"}, {"NOR", "TUM", "INFL"}
import os
CELLS_DIR = os.environ.get("VISIUM_CELLS", "cells_0.25um")
ANALYSIS_DIR = "analysis" if CELLS_DIR == "cells_0.25um" else "analysis_" + CELLS_DIR.split("_")[-1]
CLASSIFIERS = ["original", "finetuned", "finetuned_tau05"] + (["finetuned40", "finetuned40_tau05"] if CELLS_DIR != "cells_0.25um" else [])
ESCA_ANALYSES = V.ROOT / "outputs" / "esca_tls_20261019"
# Per classifier: ESCA analysis whose dev-tuned detector parameters are frozen here.
FROZEN_FROM = {"original": "analysis_he2cell_ft", "finetuned": "analysis_he2cell_ft", "finetuned_tau05": "analysis_he2cell_ft_tau05",
               "finetuned40": "analysis_he2cell_ft40", "finetuned40_tau05": "analysis_he2cell_ft40_tau05"}
PRIORS = {"finetuned_tau05": ("finetuned", V.ROOT / "outputs" / "classifier_finetune_20261019" / "prior_correction_tau05.json"),
          "finetuned40_tau05": ("finetuned40", V.ROOT / "outputs" / "classifier_finetune_40pt_20261022" / "prior_correction_tau05.json")}
# Prior-corrected fine-tune: probabilities x training-class prior^0.5, renormalised (no re-scoring needed).
FROZEN = json.loads((V.ROOT / "outputs" / "esca_tls_20261019" / "analysis_he2cell_ft" / "report_summary.json").read_text())["chosen_parameters"]
MARKERS = {"B_genes": ["MS4A1", "CD79A", "CD19"], "T_genes": ["CD3E", "CD3D"], "TLS_genes": ["CXCL13", "CR2", "LTB"]}


def expression(sample, barcodes):
    f = h5py.File(V.DATA / "h5ad_preprocessed" / f"{sample}.h5ad", "r")
    X = f["X"]
    matrix = X[:] if isinstance(X, h5py.Dataset) else sparse.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:])).toarray()
    genes = [g.decode() for g in f["var"]["gene_name"][:]]
    index = [b.decode() for b in f["obs"]["_index"][:]]
    library = matrix.sum(1, keepdims=True)
    cpm = np.log1p(matrix / np.maximum(library, 1) * 1e4)
    frame = pd.DataFrame(index=index)
    for name, panel in MARKERS.items():
        frame[name] = cpm[:, [genes.index(g) for g in panel if g in genes]].mean(1)
    frame["umi"] = library[:, 0]
    return frame.loc[barcodes]


def sample_inputs(sample, classifier):
    """Cell-count grids (8 µm) for one classifier, the in-tissue mask and spot table."""
    cells = pd.read_csv(OUT / CELLS_DIR / sample / "cells.csv.gz")
    mpp = V.native_mpp(sample)
    spots = V.spots(sample)
    with __import__("tifffile").TiffFile(V.DATA / "tif_slides" / f"{sample}.tif") as tif:
        height, width = tif.pages[0].shape[:2]
    ny, nx = int(np.ceil(height * mpp / GRID)), int(np.ceil(width * mpp / GRID))
    gx = (cells.x_native * mpp / GRID).astype(int).clip(0, nx - 1).to_numpy()
    gy = (cells.y_native * mpp / GRID).astype(int).clip(0, ny - 1).to_numpy()
    if classifier in PRIORS and f"celltype_{classifier}" not in cells:
        base, path = PRIORS[classifier]
        correction = json.loads(path.read_text())
        names = ["Other", "T", "Macrophage", "B"]
        p = cells[[f"prob_{k}_{base}" for k in names]].to_numpy() * np.asarray(correction["prior_Other_T_Macrophage_B"]) ** correction["tau"]
        cells[f"celltype_{classifier}"] = np.array(names)[p.argmax(1)]
    counts = {}
    for name in ["B", "T", "Other", "Macrophage"]:
        m = (cells[f"celltype_{classifier}"] == name).to_numpy()
        counts[name] = np.bincount(gy[m] * nx + gx[m], minlength=nx * ny).reshape(ny, nx).astype(float)
    valid = np.zeros((ny, nx), np.uint8)
    for s in spots[spots.in_tissue == 1].itertuples():
        cv2.circle(valid, (int(s.x * mpp / GRID), int(s.y * mpp / GRID)), int(round(62 / GRID)), 1, -1)
    return {"counts": counts, "valid": valid.astype(bool), "n": max(nx, ny), "shape": (ny, nx), "mpp": mpp,
            "cells": cells, "spots": spots}


def spot_disks(spots, mpp, shape):
    """Boolean disk mask (8 µm grid) per spot, as index arrays."""
    radius = SPOT_RADIUS_UM / GRID
    r = int(np.ceil(radius))
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= radius * radius]
    out = []
    for s in spots.itertuples():
        cy, cx = s.y * mpp / GRID, s.x * mpp / GRID
        ys = np.clip([int(cy + dy) for dy, _ in offsets], 0, shape[0] - 1)
        xs = np.clip([int(cx + dx) for _, dx in offsets], 0, shape[1] - 1)
        out.append((ys, xs))
    return out


def detectors(x, frozen=None):
    """Frozen-parameter detectors -> boolean masks on the 8 µm grid."""
    FROZEN = frozen or globals()["FROZEN"]
    masks = {}
    p = FROZEN["B (HE2CELL)"]
    masks["B density"] = A.detect(x, ["B"], **p)[0]
    p = FROZEN["B+T (HE2CELL)"]
    masks["B+T density"] = A.detect(x, ["B", "T"], **p)[0]
    p = FROZEN["All nuclei (no classifier)"]
    masks["All nuclei (no classifier)"] = A.detect(x, None, **p)[0]
    p = FROZEN[A.TWO_STAGE]
    proposals, _ = A.detect(x, None, **{k: p[k] for k in ["sigma_um", "threshold", "min_area_um2", "min_cells"]})
    total = sum(x["counts"].values())
    keep = np.zeros_like(proposals)
    for obj in A.objects(proposals):
        share = x["counts"]["B"][obj].sum() / max(total[obj].sum(), 1)
        if share >= p["b_share"]:
            keep |= obj
    masks["Two-stage"] = keep
    return masks


def tls_objects(spots, spacing_px):
    """Connected groups of TLS-labelled spots (centre distance <= 1.2 spot pitch)."""
    tls = spots[spots.label.isin(POSITIVE)]
    if tls.empty:
        return pd.Series(dtype=int)
    tree = cKDTree(tls[["x", "y"]].to_numpy())
    pairs = tree.query_pairs(1.2 * spacing_px, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(tls), len(tls)))
    _, labels = sparse.csgraph.connected_components(graph, directed=False)
    return pd.Series(labels, index=tls.index)


def main():
    rows, regions, objects_rows = [], [], []
    hooknet_available = {}
    for sample in V.SAMPLES:
        if not (OUT / CELLS_DIR / sample / "cells.csv.gz").exists():
            print(f"{sample}: cells missing, skipped")
            continue
        base = None
        for classifier in CLASSIFIERS:
            x = sample_inputs(sample, classifier)
            spots, mpp, shape = x["spots"], x["mpp"], x["shape"]
            disks = spot_disks(spots, mpp, shape)
            masks = detectors(x, json.loads((ESCA_ANALYSES / FROZEN_FROM[classifier] / "report_summary.json").read_text())["chosen_parameters"])
            total = sum(x["counts"].values())
            b_density = A.density(x["counts"], ["B"], 25)
            nuclei_density = A.density(x["counts"], None, 25)
            bt_density = A.density(x["counts"], ["B", "T"], 25)
            for i, s in enumerate(spots.itertuples()):
                ys, xs = disks[i]
                n_all = total[ys, xs].sum()
                row = {"sample": sample, "classifier": classifier, "barcode": s.barcode, "label": s.label,
                       "in_tissue": s.in_tissue, "x": s.x, "y": s.y, "cells": n_all,
                       "B": x["counts"]["B"][ys, xs].sum(), "T": x["counts"]["T"][ys, xs].sum(),
                       "B_density_at_spot": float(b_density[ys, xs].mean()),
                       "BT_density_at_spot": float(bt_density[ys, xs].mean()),
                       "nuclei_density_at_spot": float(nuclei_density[ys, xs].mean())}
                row["B_fraction"] = row["B"] / n_all if n_all else np.nan
                for name, mask in masks.items():
                    row[f"cover_{name}"] = float(mask[ys, xs].mean())
                rows.append(row)
            # Detector components and what the pathologist says under them.
            for name, mask in masks.items():
                labelled, count = ndimage.label(mask)
                for index in range(1, count + 1):
                    covered = [spots.label.iat[i] for i, (ys, xs) in enumerate(disks)
                               if (labelled[ys, xs] == index).mean() >= .5]
                    regions.append({"sample": sample, "classifier": classifier, "detector": name, "component": index,
                                    "area_um2": float((labelled == index).sum() * GRID ** 2),
                                    "covers_TLS": any(l in POSITIVE for l in covered),
                                    "covers_negative_only": bool(covered) and all(l in NEGATIVE for l in covered),
                                    "covered_labels": ",".join(sorted(set(covered)))})
        # HookNet-TLS reference on the same image, if available.
        summary = OUT / "hooknet" / sample / "summary.json"
        if summary.exists():
            hk = json.loads(summary.read_text())
            ratio = hk["source_dimensions"][0] / hk["dimensions_at_0p5um"][0]
            features = [f for f in json.loads((OUT / "hooknet" / sample / "native_polygons.geojson").read_text())["features"]
                        if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]]
            canvas = np.zeros(shape, np.uint8)
            for feature in features:
                ring = np.asarray(feature["geometry"]["coordinates"][0], float) * ratio * mpp / GRID
                cv2.fillPoly(canvas, [np.round(ring).astype(np.int32)], 1)
            cover = [float(canvas.astype(bool)[ys, xs].mean()) for ys, xs in disks]
            hooknet_available[sample] = dict(zip(spots.barcode, cover))
    frame = pd.DataFrame(rows)
    frame["cover_HookNet-TLS"] = [hooknet_available.get(s, {}).get(b, np.nan) for s, b in zip(frame["sample"], frame.barcode)]
    genes = pd.concat([expression(s, frame[(frame["sample"] == s) & (frame.classifier == "original")].barcode)
                       .assign(sample=s) for s in frame["sample"].unique()])
    frame = frame.merge(genes.reset_index().rename(columns={"index": "barcode"}), on=["sample", "barcode"], how="left")
    analysis = OUT / ANALYSIS_DIR
    analysis.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis / "spot_table.csv.gz", index=False)
    pd.DataFrame(regions).to_csv(analysis / "detector_components.csv", index=False)
    results = evaluate(frame, pd.DataFrame(regions))
    (analysis / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(json.dumps(results["headline"], indent=1, default=float))


def cluster_bootstrap(frame, function, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    samples = frame["sample"].unique()
    groups = {s: frame[frame["sample"] == s] for s in samples}
    values = [function(pd.concat([groups[s] for s in rng.choice(samples, len(samples))])) for _ in range(n)]
    return np.nanpercentile(np.asarray(values, float), [2.5, 97.5]).tolist()


def binary(frame, column, threshold=.5):
    y = frame.label.isin(POSITIVE).to_numpy()
    p = (frame[column] >= threshold).to_numpy()
    tp, fp, fn, tn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    return {"sensitivity": sens, "specificity": spec, "PPV": ppv, "F1": 2 * ppv * sens / max(ppv + sens, 1e-9),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def evaluate(frame, regions):
    scored = frame[frame.label.isin(POSITIVE | NEGATIVE) & (frame.in_tissue == 1)]
    results = {"spots": {k: int(v) for k, v in scored[scored.classifier == "original"].label.value_counts().items()},
               "samples": sorted(scored["sample"].unique()),
               "frozen_parameters": {c: json.loads((ESCA_ANALYSES / FROZEN_FROM[c] / "report_summary.json").read_text())["chosen_parameters"] for c in CLASSIFIERS}}
    scores = {"B_density_at_spot": "Smoothed B density", "B": "B cells in spot", "B_fraction": "B fraction in spot",
              "BT_density_at_spot": "Smoothed B+T density", "nuclei_density_at_spot": "Nuclear density (no classifier)"}
    auroc = []
    for classifier in CLASSIFIERS:
        part = scored[scored.classifier == classifier]
        for column, name in scores.items():
            if column == "nuclei_density_at_spot" and classifier != "original":
                continue  # identical segmentation; reported once
            for contrast, sub in [("TLS vs all negatives", part), ("TLS vs INFL", part[part.label.isin(POSITIVE | {"INFL"})]),
                                  ("TLS vs TUM", part[part.label.isin(POSITIVE | {"TUM"})])]:
                y = sub.label.isin(POSITIVE)
                value = A.auc(y, sub[column].fillna(0))
                ci = cluster_bootstrap(sub, lambda s: A.auc(s.label.isin(POSITIVE), s[column].fillna(0)))
                auroc.append({"classifier": "none" if column == "nuclei_density_at_spot" else classifier, "score": name,
                              "contrast": contrast, "AUROC": value, "CI95": ci})
    results["spot_AUROC"] = auroc
    hk = scored[(scored.classifier == "original") & scored["cover_HookNet-TLS"].notna()]
    if len(hk):
        for contrast, sub in [("TLS vs all negatives", hk), ("TLS vs INFL", hk[hk.label.isin(POSITIVE | {"INFL"})])]:
            results["spot_AUROC"].append({"classifier": "HookNet-TLS (reference)", "score": "HookNet TLS coverage",
                                          "contrast": contrast, "AUROC": A.auc(sub.label.isin(POSITIVE), sub["cover_HookNet-TLS"]),
                                          "CI95": cluster_bootstrap(sub, lambda s: A.auc(s.label.isin(POSITIVE), s["cover_HookNet-TLS"]))})
    detections = []
    for classifier in CLASSIFIERS:
        part = scored[scored.classifier == classifier]
        for name in ["B density", "Two-stage", "B+T density", "All nuclei (no classifier)"]:
            if name == "All nuclei (no classifier)" and classifier != "original":
                continue
            entry = {"classifier": "none" if name.startswith("All nuclei") else classifier, "detector": name,
                     **binary(part, f"cover_{name}"),
                     "INFL_called_TLS": float((part[part.label == "INFL"][f"cover_{name}"] >= .5).mean()),
                     "CI95_F1": cluster_bootstrap(part, lambda s: binary(s, f"cover_{name}")["F1"])}
            detections.append(entry)
    if len(hk):
        detections.append({"classifier": "HookNet-TLS (reference)", "detector": "HookNet-TLS published filter",
                           **binary(hk, "cover_HookNet-TLS"),
                           "INFL_called_TLS": float((hk[hk.label == "INFL"]["cover_HookNet-TLS"] >= .5).mean()),
                           "CI95_F1": cluster_bootstrap(hk, lambda s: binary(s, "cover_HookNet-TLS")["F1"])})
    results["spot_detection_frozen"] = detections
    # Object level: pathologist TLS groups detected; detector components over negative-only spots.
    objects = []
    for (sample, classifier), part in frame[frame.in_tissue == 1].groupby(["sample", "classifier"]):
        spacing = 100 / V.native_mpp(sample)
        groups = tls_objects(part.reset_index(drop=True), spacing)
        sub = part.reset_index(drop=True)
        for g in groups.unique():
            members = sub.loc[groups.index[groups == g]]
            labels = members.label.value_counts()
            objects.append({"sample": sample, "classifier": classifier, "object": int(g), "spots": len(members),
                            "maturity": "mature (MTLS/GC)" if labels.reindex(["MTLS", "GC"]).fillna(0).sum() >= labels.get("ETLS", 0) else "early (ETLS)",
                            "B_fraction": members.B.sum() / max(members.cells.sum(), 1),
                            "B_density": members.B_density_at_spot.mean(),
                            **{f"detected_{d}": bool((members[f"cover_{d}"] >= .5).any())
                               for d in ["B density", "Two-stage", "B+T density", "All nuclei (no classifier)"]},
                            "detected_HookNet-TLS": bool((members["cover_HookNet-TLS"] >= .5).any()) if members["cover_HookNet-TLS"].notna().all() else np.nan})
    objects = pd.DataFrame(objects)
    objects.to_csv(OUT / ANALYSIS_DIR / "tls_objects.csv", index=False)
    object_summary = []
    for classifier in CLASSIFIERS:
        part = objects[objects.classifier == classifier]
        for d in ["B density", "Two-stage", "B+T density", "All nuclei (no classifier)", "HookNet-TLS"]:
            column = f"detected_{d}"
            regs = regions[(regions.classifier == classifier) & (regions.detector == d)] if d != "HookNet-TLS" else pd.DataFrame()
            object_summary.append({"classifier": classifier, "detector": d, "TLS_objects": len(part),
                                   "recall_all": float(part[column].mean()),
                                   "recall_early": float(part[part.maturity.str.startswith("early")][column].mean()),
                                   "recall_mature": float(part[part.maturity.str.startswith("mature")][column].mean()),
                                   "components": len(regs), "components_on_TLS": int(regs.covers_TLS.sum()) if len(regs) else None,
                                   "components_on_negative_only": int(regs.covers_negative_only.sum()) if len(regs) else None,
                                   "component_precision_labelled": (float(regs.covers_TLS.sum() / max(regs.covers_TLS.sum() + regs.covers_negative_only.sum(), 1))
                                                                    if len(regs) else None)})
    results["object_level_frozen"] = object_summary
    maturity = {}
    for classifier in CLASSIFIERS:
        part = objects[objects.classifier == classifier]
        mature = part.maturity.str.startswith("mature")
        if mature.sum() >= 3 and (~mature).sum() >= 3:
            maturity[classifier] = {f: {"median_mature": float(part.loc[mature, f].median()), "median_early": float(part.loc[~mature, f].median()),
                                        "AUROC_mature": A.auc(mature, part[f])} for f in ["B_fraction", "B_density", "spots"]}
    results["maturity"] = maturity
    # Spot-level cell-type check against the spot's own RNA (all in-tissue spots, labelled or not).
    genes = []
    tissue = frame[frame.in_tissue == 1]
    for classifier in CLASSIFIERS:
        for score, target in [("B", "B_genes"), ("B_fraction", "B_genes"), ("T", "T_genes"), ("cells", "B_genes"), ("cells", "T_genes")]:
            per_sample = [stats.spearmanr(g[score], g[target], nan_policy="omit").statistic
                          for _, g in tissue[tissue.classifier == classifier].groupby("sample")]
            genes.append({"classifier": classifier, "score": score, "genes": "+".join(MARKERS[target]),
                          "median_spearman_across_samples": float(np.nanmedian(per_sample)),
                          "per_sample": [round(float(v), 3) for v in per_sample]})
    results["spot_RNA_check"] = genes
    best = [r for r in auroc if r["classifier"] == "finetuned" and r["score"] == "Smoothed B density"]
    results["headline"] = {"spots": results["spots"],
                           "AUROC_finetuned_B_density": {r["contrast"]: round(r["AUROC"], 3) for r in best},
                           "AUROC_original_B_density": {r["contrast"]: round(r["AUROC"], 3) for r in auroc
                                                        if r["classifier"] == "original" and r["score"] == "Smoothed B density"},
                           "AUROC_nuclei": {r["contrast"]: round(r["AUROC"], 3) for r in auroc if r["classifier"] == "none"}}
    return results


if __name__ == "__main__":
    main()

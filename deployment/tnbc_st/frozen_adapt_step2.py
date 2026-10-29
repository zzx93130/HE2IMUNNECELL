"""Step 2: label-free adaptation arms A0-A5 (preregistration.json + amendment_1).

Stages
  prep  <arrays>   CPU: AdaBN crop sample (2,500/array) + per-array Macenko source fit (3,000 crops) -> work/<array>/
  bn               GPU: per fold k, AdaBN BN buffers on training-fold samples, raw (A1_k) and Macenko (A3_k) -> work/bn/
  infer <arrays>   GPU: eval subsample (1/3) crops; A0, A2, A5 with the stored model; A1/A3 with the held-out fold's BN;
                   training scoring (A1_j/A3_j for the other folds j) on a further 1/3 -> probs/<array>.npz (uncorrected D4)
  ffpe             GPU: Immunocto CV fold models on held-out crops: clean / A1 / A2 / A3 -> ffpe/
GPU stages must be run under deployment/ablations/gpu_locked.sh with GPU_LOCK_OWNER=tnbc-frozen-adapt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_adapt_common as F  # noqa: E402

import os
TEST = os.environ.get("FA_TEST") == "1"  # smoke test on CPU: tiny subsets, separate output dirs
WORK = F.OUT / "work"
PROBS = F.OUT / ("work/test_probs" if TEST else "probs")
BN = WORK / ("test_bn" if TEST else "bn")
FRACTION = 1 / 3
N_ADABN = 2500
N_MACENKO = 3000


def eval_subsample(array, cells):
    sample = F.inventory()[array]["sample"]
    r = np.random.default_rng([F.SEED, sample]).random(len(cells))
    ev = r < FRACTION
    tr = r < FRACTION / 3  # subset of ev
    return ev, tr


def prep(arrays):
    target = F.esca_target()
    for array in arrays:
        out = WORK / array
        if (out / "prep.npz").exists():
            print(array, "prep done")
            continue
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        cells = F.load_cells(array)
        image = F.C.read_image(F.inventory()[array]["hd"])
        rng = np.random.default_rng([F.SEED, 1, F.inventory()[array]["sample"]])
        pick = np.sort(rng.choice(len(cells), min(N_ADABN + N_MACENKO, len(cells)), replace=False))
        crops, found = F.build_crops(array, cells.iloc[pick].reset_index(drop=True), image=image)
        rgb, mask = crops[1.0]
        rgb, mask = rgb[found], mask[found]
        order = rng.permutation(len(rgb))
        ad, mk = order[:N_ADABN], order[N_ADABN:]
        src = F.macenko_fit(rgb[mk].reshape(-1, 3))
        src["crops"] = int(len(mk))
        F.C.save_json(src, out / "macenko_source.json")
        np.savez(out / "prep.npz", rgb=rgb[ad], mask=mask[ad], rgb_macenko=F.macenko_apply(rgb[ad], src, target))
        print(array, f"prep {len(ad)} adabn crops, found {found.mean():.4f}, {time.time() - t:.0f}s", flush=True)


def bn(device):
    import torch
    base = F.load_model(device)
    out = BN
    out.mkdir(parents=True, exist_ok=True)
    arrays = [a for a in F.priority() if (WORK / a / "prep.npz").exists()]
    cases = {a: f"TNBC{F.inventory()[a]['sample']}" for a in arrays}
    log = {}
    for k, test in F.folds().items():
        train = [a for a in arrays if cases[a] not in test]
        data = [np.load(WORK / a / "prep.npz") for a in train]
        if not data:
            print(k, "no training arrays yet", flush=True)
            continue
        lim = 300 if TEST else None
        mask = np.concatenate([d["mask"][:lim] for d in data])
        for arm, key in [("A1", "rgb"), ("A3", "rgb_macenko")]:
            rgb = np.concatenate([d[key][:lim] for d in data])
            m = F.adabn(base, rgb, mask, device)
            torch.save({n: b.cpu() for n, b in m.state_dict().items() if "running_" in n or "num_batches" in n}, out / f"{arm}_{k}.pt")
        log[k] = {"train_arrays": train, "crops": int(len(mask)), "meets_50k": bool(len(mask) >= 50000)}
        print(k, log[k]["crops"], flush=True)
    F.C.save_json(log, out / "bn_log.json")


def fold_model(base, arm, k, device):
    import copy
    import torch
    m = copy.deepcopy(base)
    m.load_state_dict(torch.load(BN / f"{arm}_{k}.pt"), strict=False)
    return m.to(device).eval()


def infer(arrays, device):
    target = F.esca_target()
    base = F.load_model(device)
    models = {(arm, k): fold_model(base, arm, k, device) for arm in ["A1", "A3"] for k in F.folds()}
    for array in arrays:
        path = PROBS / f"{array}.npz"
        if path.exists():
            print(array, "done")
            continue
        PROBS.mkdir(parents=True, exist_ok=True)
        t = time.time()
        cells = F.load_cells(array)
        ev, tr = eval_subsample(array, cells)
        if TEST:
            ev[np.flatnonzero(ev)[300:]] = False
            tr &= ev
        sub = cells[ev].reset_index(drop=True)
        crops, found = F.build_crops(array, sub, scales=(1.0, 0.88))
        rgb, mask = crops[1.0]
        rgb5, mask5 = crops[0.88]
        src = json.loads((WORK / array / "macenko_source.json").read_text())
        rgbm = np.concatenate([F.macenko_apply(rgb[i:i + 4096], src, target) for i in range(0, len(rgb), 4096)])
        t_build = time.time() - t
        case = f"TNBC{F.inventory()[array]['sample']}"
        k_out = F.fold_of(case)
        trs = tr[ev]
        res = {"A0": F.predict_d4(base, rgb, mask, device), "A2": F.predict_d4(base, rgbm, mask, device),
               "A5": F.predict_d4(base, rgb5, mask5, device),
               "A1": F.predict_d4(models[("A1", k_out)], rgb, mask, device),
               "A3": F.predict_d4(models[("A3", k_out)], rgbm, mask, device)}
        for k in F.folds():
            if k == k_out:
                continue
            res[f"train_A1_{k}"] = F.predict_d4(models[("A1", k)], rgb[trs], mask[trs], device)
            res[f"train_A3_{k}"] = F.predict_d4(models[("A3", k)], rgbm[trs], mask[trs], device)
        np.savez_compressed(path, global_instance_id=sub.global_instance_id.to_numpy(), found=found, train_subset=trs,
                            **{k: v.astype(np.float32) for k, v in res.items()})
        print(array, f"{len(sub)} cells (found {found.mean():.4f}), build {t_build:.0f}s, total {time.time() - t:.0f}s", flush=True)


def ffpe(device, per_patient=1000):
    """Immunocto CV held-out folds: does A1/A2/A3 hurt FFPE?"""
    import torch
    import classifier_cv as CV
    import immunocto_holdout_eval as E
    target = F.esca_target()
    out = F.OUT / ("work/test_ffpe" if TEST else "ffpe")
    out.mkdir(parents=True, exist_ok=True)
    spec = CV.load_folds()["folds"]
    crops = pd.read_csv(CV.OUT / "crops.csv.gz")
    # TNBC AdaBN sample: training-fold crops of TNBC fold0 (any fold works; it's a transform of the FFPE model)
    arrays = [a for a in F.priority() if (WORK / a / "prep.npz").exists()]
    tn = [np.load(WORK / a / "prep.npz") for a in arrays if f"TNBC{F.inventory()[a]['sample']}" not in F.folds()["fold0"]]
    if TEST and not tn:
        tn = [np.load(WORK / a / "prep.npz") for a in arrays]
    tn_rgb = np.concatenate([d["rgb"] for d in tn])
    tn_rgbm = np.concatenate([d["rgb_macenko"] for d in tn])
    tn_mask = np.concatenate([d["mask"] for d in tn])
    rows = []
    if TEST:
        spec = {"fold0": {**spec["fold0"], "test_patients": spec["fold0"]["test_patients"][:2]}}
        per_patient, tn_rgb, tn_rgbm, tn_mask = 20, tn_rgb[:300], tn_rgbm[:300], tn_mask[:300]
    for name, f in spec.items():
        model = CV.load_fold_model(name, device)
        corr = CV.fold_prior(name)
        a1 = F.adabn(model, tn_rgb, tn_mask, device)
        a3 = F.adabn(model, tn_rgbm, tn_mask, device)
        for patient in f["test_patients"]:
            c = crops[crops.patient == patient]
            c = c.iloc[np.sort(np.random.default_rng([F.SEED, int(patient[3:])]).choice(len(c), min(per_patient, len(c)), replace=False))]
            rgb, mask = E.load_crops(c)
            src = F.macenko_fit(rgb.reshape(-1, 3))
            rgbm = F.macenko_apply(rgb, src, target)
            for arm, m, x in [("clean", model, rgb), ("A1", a1, rgb), ("A2", model, rgbm), ("A3", a3, rgbm)]:
                p = F.predict_d4(m, x, mask, device)
                p = E.apply_prior(p, corr) if corr is not None else p
                for lab, pred in zip(c.label.to_numpy(), p.argmax(1)):
                    rows.append((name, patient, arm, int(lab), int(pred)))
        print(name, "done", flush=True)
    pd.DataFrame(rows, columns=["fold", "patient", "arm", "label", "pred"]).to_csv(out / "predictions.csv.gz", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "bn", "infer", "ffpe"])
    ap.add_argument("--arrays", nargs="*")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    arrays = a.arrays or [x for x in F.done_arrays()]
    if F.C.free_gb() < 8:
        raise SystemExit("STOP: < 8 GB free")
    if a.stage == "prep":
        prep(arrays)
    elif a.stage == "bn":
        bn(a.device)
    elif a.stage == "infer":
        infer(arrays, a.device)
    else:
        ffpe(a.device)


if __name__ == "__main__":
    main()

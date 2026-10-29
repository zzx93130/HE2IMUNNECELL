"""Cell-level validation of HE2CELL T/B identification on Immunocto patients CRC11-CRC40 (NOT held out, see CAVEATS).

Ground truth: the immunofluorescence-derived Immunocto class in each crop's file name
(data/immunocto/CRC##/{HE,mask}/<class>_<x>_<y>.png; other / cd4 / cd8 / macrophage / bcell, T = cd4 + cd8).
The adopted fine-tune (models/he2cell_esca_finetuned.pth) was fine-tuned on CRC01-CRC10 only, but it is initialised
from models/best_model2.pth, which was trained on all 40 Immunocto patients (user statement 2026-10-26). CRC11-CRC40
are therefore training data for A3 and only unseen by the fine-tuning step for A1/A2. outputs/classifier_finetune_40pt_20261022/ trained on these patients and is NOT evaluated.

Arms
  A1  fine-tune + class-prior correction (default; sidecar models/he2cell_esca_finetuned.prior.json, tau 0.5)
  A2  fine-tune, --prior_correction none
  A3  original models/best_model2.pth (no sidecar)
A1 and A2 share one forward pass: the correction is applied after D4 averaging exactly as model_eval.py does.

Preprocessing reproduces script/src/model_eval.py (PIL RGB / 255, mask / max(mask), ImageNet normalisation of RGB,
D4 mean softmax via prediction_tta.average_views). `verify` checks this on 1000 crops against model_eval.py itself.

Stain robustness: every crop is re-scored under the LAB colour-style shift and HED stain perturbation of
deployment/finetune/finetune_classifier.py:augment at three fixed severities. Each crop draws one seeded random
direction (target LAB mean/std scale from the training STYLE range, HED scale +-15% and shift +-0.012); severity
lambda scales the move away from the crop's own colour: 0.5 (mild), 1.0 (= a full-strength training draw), 1.5
(beyond the training range). The same draws are used for both checkpoints, so comparisons are paired.

Stages (all resumable; predictions are cached per patient and condition):
  index    list crops, sample <= 50,000 per patient (seed 20261026), crops.csv.gz
  verify   1000-crop equivalence check against model_eval.py -> verification.json
  infer    D4 probabilities per checkpoint x condition -> predictions/<model>_<condition>/CRC##.csv.gz
  metrics  confusion matrices, P/R/F1, macro-F1, balanced accuracy, B-vs-T AUROC, ECE, prevalence, patient-cluster
           bootstrap (2000) CIs, paired differences, robustness drops -> summary.json, per_patient_metrics.csv
The HTML report is written by immunocto_holdout_report.py.

Device mps, batch 512. This script never takes outputs/_gpu.lock (classification-only inference shares the GPU).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "script" / "src"
DATA = ROOT / "data" / "immunocto"
OUT = ROOT / "outputs" / "cell_validation_20261026"
PRED = OUT / "predictions"
PY = ROOT / ".venv" / "bin" / "python"
SEED = 20261026
N_PER_PATIENT = 50_000
N_BOOT = 2000
PATIENTS = [f"CRC{i:02d}" for i in range(11, 41)]
CLASS = {"other": 0, "cd4": 1, "cd8": 1, "macrophage": 2, "bcell": 3}
NAMES = ["Other", "T", "Macrophage", "B"]
MODELS = {"ft": ROOT / "models" / "he2cell_esca_finetuned.pth", "orig": ROOT / "models" / "best_model2.pth"}
CONDITIONS = {"clean": None, "sev1": 0.5, "sev2": 1.0, "sev3": 1.5}
ARMS = {"A1": {"label": "Fine-tune + prior correction (default)", "model": "ft", "prior": True},
        "A2": {"label": "Fine-tune, no prior correction", "model": "ft", "prior": False},
        "A3": {"label": "Original best_model2.pth", "model": "orig", "prior": False}}
QUESTION = 'Cell-level T/B identification on Immunocto CRC11-CRC40 (IF-derived labels); NOT held out: best_model2.pth trained on all 40 patients and the fine-tune is initialised from it (see caveats)'
CAVEATS = [
 "Training-set fact (source: user statement 2026-10-26): models/best_model2.pth was trained on ALL 40 Immunocto patients (CRC01-CRC40).",
 "1. CRC11-CRC40 are NOT held out for A3 (original checkpoint): its clean-crop numbers are training-set performance.",
 "2. They are also not strictly held out for A1/A2: the fine-tune was initialised from best_model2.pth, so CRC11-CRC40 are only unseen by the fine-tuning step. The A1 vs A3 clean comparison (e.g. macro-F1 -0.036) compares two models on crops both have seen in some form and cannot speak to generalisation.",
 "3. Still supported: (a) the stain-perturbation robustness ranking (macro-F1 drop at lambda 1.0: A1 +0.022 vs A3 +0.260; differences of drops exclude 0), with the caveat that the perturbation family equals the fine-tune's training augmentation family; (b) the prior-correction trade-off A1 vs A2 (B recall -0.196 for higher B precision and better calibration), since both share the same training data; (c) the calibration and prevalence match.",
 "4. Not supported: any absolute held-out T/B accuracy claim for either model. A genuine held-out estimate needs external CD20/CD3 data (plan item 2) or a from-scratch patient-level cross-validation with the same recipe (neither was run)."
]
PAIRS = [("A1", "A3"), ("A2", "A3"), ("A1", "A2")]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# Colour constants copied from deployment/finetune/finetune_classifier.py (asserted equal in `verify`).
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11],
                         np.cross([0.65, 0.70, 0.29], [0.07, 0.99, 0.11])], np.float32)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
HED_ALPHA, HED_BETA = .15, .012
STYLE = {"L": (110, 190), "a": (135, 175), "b": (100, 132), "std_scale": (0.75, 1.35)}


def save_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


def write_csv(df, path):
    tmp = Path(str(path) + ".tmp.csv.gz")
    df.to_csv(tmp, index=False, float_format="%.7g")
    tmp.replace(path)


# ----------------------------------------------------------------------------------------------- index
def index(args):
    (OUT / "crop_index").mkdir(parents=True, exist_ok=True)
    for patient in PATIENTS:
        path = OUT / "crop_index" / f"{patient}.csv.gz"
        if path.exists():
            continue
        folder = DATA / patient
        if not (folder / ".complete").exists():
            raise SystemExit(f"{patient} has no .complete marker")
        t = time.time()
        names = sorted(e.name for e in os.scandir(folder / "HE") if e.name.endswith(".png"))
        cls = [n.split("_")[0] for n in names]
        keep = [i for i, c in enumerate(cls) if c in CLASS]
        names = [names[i] for i in keep]
        full = pd.Series([cls[i] for i in keep]).value_counts().to_dict()
        rng = np.random.default_rng([SEED, int(patient[3:])])
        n = min(len(names), N_PER_PATIENT)
        pick = np.sort(rng.choice(len(names), n, replace=False))
        rows = []
        for i in pick:
            c, x, y = names[i][:-4].split("_")
            rows.append((patient, names[i], c, CLASS[c], int(x), int(y)))
        df = pd.DataFrame(rows, columns=["patient", "name", "immunocto_class", "label", "x", "y"])
        df.insert(1, "row", np.arange(len(df)))
        df.attrs["population"] = full
        write_csv(df, path)
        save_json(OUT / "crop_index" / f"{patient}.population.json",
                  {"population_crops": len(names), "sampled": n, "class_counts": full})
        print(f"{patient}: {len(names)} crops, sampled {n} ({time.time() - t:.0f}s)", flush=True)
    crops = pd.concat([pd.read_csv(OUT / "crop_index" / f"{p}.csv.gz") for p in PATIENTS], ignore_index=True)
    write_csv(crops, OUT / "crops.csv.gz")
    print("crops:", len(crops), crops.groupby("label").size().to_dict())


def crop_paths(df):
    he = [str(DATA / p / "HE" / n) for p, n in zip(df.patient, df.name)]
    return he, [h.replace("/HE/", "/mask/") for h in he]


# ----------------------------------------------------------------------------------------------- loading
def _read(pair):
    from PIL import Image
    he, mask = pair
    rgb = np.asarray(Image.open(he).convert("RGB"), dtype=np.uint8)
    m = np.asarray(Image.open(mask).convert("L"), dtype=np.uint8)
    return rgb, m


def load_crops(df, threads=16):
    he, mask = crop_paths(df)
    with ThreadPoolExecutor(threads) as pool:
        out = list(pool.map(_read, zip(he, mask), chunksize=256))
    rgb = np.stack([o[0] for o in out])
    m = np.stack([o[1] for o in out])
    if rgb.shape[1:] != (64, 64, 3) or m.shape[1:] != (64, 64):
        raise ValueError(f"unexpected crop shape {rgb.shape} {m.shape}")
    return rgb, m


def to_input(rgb, mask):
    """model_eval.py convention: RGB/255 then ImageNet Normalize; mask/max(mask) (0 if empty)."""
    import torch
    from torchvision.transforms.v2 import functional as F
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
    x = F.normalize(x, mean=MEAN.tolist(), std=STD.tolist())
    m = mask.astype(np.float32)
    mx = m.reshape(len(m), -1).max(1)
    m = np.where(mx[:, None, None] > 0, m / np.maximum(mx, 1e-12)[:, None, None], m)
    return torch.cat([x, torch.from_numpy(m)[:, None]], 1).contiguous()


# ----------------------------------------------------------------------------------------------- perturbation
def lab_stats(rgb, n=400, seed=0):
    """Patient colour statistics as finetune_classifier.patient_lab_stats (400 crops, pixels with L < 215)."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rgb), min(n, len(rgb)), replace=False)
    pixels = []
    for i in pick:
        lab = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2LAB).reshape(-1, 3)
        pixels.append(lab[lab[:, 0] < 215])
    pixels = np.concatenate(pixels).astype(np.float32)
    return pixels.mean(0), pixels.std(0)


def draws(n, patient):
    """One seeded perturbation direction per crop (identical for every severity and checkpoint)."""
    rng = np.random.default_rng([SEED, 7, int(patient[3:])])
    return np.column_stack([rng.uniform(*STYLE["L"], n), rng.uniform(*STYLE["a"], n), rng.uniform(*STYLE["b"], n),
                            rng.uniform(*STYLE["std_scale"], (n, 3)),
                            rng.uniform(1 - HED_ALPHA, 1 + HED_ALPHA, (n, 3)),
                            rng.uniform(-HED_BETA, HED_BETA, (n, 3))]).astype(np.float32)


def perturb_one(rgb, stats, d, lam):
    """finetune_classifier.augment's LAB style shift + HED perturbation, with the random move scaled by lam."""
    mean, std = stats
    x = rgb.astype(np.float32) / 255
    lab = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    target_mean = mean + lam * (d[0:3] - mean)
    target_std = std * (1 + lam * (d[3:6] - 1))
    out = (lab - mean) / std * target_std + target_mean
    weight = np.clip((240 - lab[..., :1]) / 40, 0, 1)
    out = weight * out + (1 - weight) * lab
    x = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255
    od = -np.log(np.clip(x, 1e-6, 1)).reshape(-1, 3) / -np.log(1e-6)
    hed = np.maximum(od @ HED_FROM_RGB, 0)
    hed = hed * (1 + lam * (d[6:9] - 1)) + lam * d[9:12]
    x = np.clip(np.exp(-(hed @ RGB_FROM_HED) * -np.log(1e-6)), 0, 1).reshape(64, 64, 3).astype(np.float32)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def perturb_chunk(job):
    rgb, stats, d, lam = job
    return np.stack([perturb_one(rgb[i], stats, d[i], lam) for i in range(len(rgb))])


def perturb(rgb, stats, d, lam, pool, chunk=2000):
    jobs = [(rgb[i:i + chunk], stats, d[i:i + chunk], lam) for i in range(0, len(rgb), chunk)]
    return np.concatenate(list(pool.map(perturb_chunk, jobs)))


# ----------------------------------------------------------------------------------------------- model
def load_model(key, device):
    import torch
    sys.path.insert(0, str(SRC))
    from model_eval import SAM_ConvNet
    from runtime import load_classifier_state
    model = SAM_ConvNet(num_classes=4, loss_function="CrossEntropyLoss", pretrained=False)
    model.load_state_dict(load_classifier_state(MODELS[key]))
    return model.to(device).eval()


def predict(model, rgb, mask, device, batch=512):
    """D4-mean softmax (uncorrected), exactly prediction_tta.average_views on model_eval-preprocessed input."""
    import torch
    sys.path.insert(0, str(SRC))
    from prediction_tta import average_views
    out = []
    with torch.inference_mode():
        for i in range(0, len(rgb), batch):
            x = to_input(rgb[i:i + batch], mask[i:i + batch]).to(device)
            p, _ = average_views(model, x)
            out.append(p.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def prior_weights():
    sys.path.insert(0, str(SRC))
    from model_eval import load_prior_correction
    return load_prior_correction(MODELS["ft"], "auto")


def apply_prior(p, correction):
    sys.path.insert(0, str(SRC))
    from model_eval import apply_prior as ap
    return ap(p, correction)


# ----------------------------------------------------------------------------------------------- verify
def verify(args):
    import torch
    sys.path.insert(0, str(ROOT / "deployment" / "finetune"))
    import finetune_classifier as FC
    assert np.allclose(FC.RGB_FROM_HED, RGB_FROM_HED) and FC.STYLE == STYLE and (FC.HED_ALPHA, FC.HED_BETA) == (HED_ALPHA, HED_BETA)
    crops = pd.read_csv(OUT / "crops.csv.gz")
    sample = crops.sample(1000, random_state=SEED).reset_index(drop=True)
    vdir = OUT / "verification"
    vdir.mkdir(exist_ok=True)
    he, mask = crop_paths(sample)
    meta = pd.DataFrame({"cell_image_crop_path": he, "cell_mask_crop_path": mask,
                         "global_instance_id": np.arange(len(sample)), "label": sample.label})
    meta.to_csv(vdir / "meta_1000.csv", index=False)
    runs = {"A1": ("ft", "auto"), "A2": ("ft", "none"), "A3": ("orig", "auto")}
    ref = {}
    for arm, (key, prior) in runs.items():
        out = vdir / f"model_eval_{arm}.csv"
        if not out.exists():
            subprocess.run([str(PY), str(SRC / "model_eval.py"), "--meta_path", str(vdir / "meta_1000.csv"),
                            "--save_path", str(MODELS[key]), "--output_path", str(out), "--device", "mps",
                            "--batch_size", "512", "--num_workers", "0", "--tta", "d4", "--prior_correction", prior],
                           check=True, cwd=str(SRC))
        ref[arm] = pd.read_csv(out)
    device = torch.device(args.device)
    rgb, m = load_crops(sample)
    mine = {}
    for key in MODELS:
        mine[key] = predict(load_model(key, device), rgb, m, device, args.batch)
    corr = prior_weights()
    ours = {"A1": apply_prior(mine["ft"], corr), "A2": mine["ft"], "A3": mine["orig"]}
    result = {"n_crops": len(sample), "batch": args.batch, "device": args.device, "arms": {}}
    cols = [f"prob_{n}" for n in NAMES]
    for arm in runs:
        r = ref[arm][cols].to_numpy()
        diff = np.abs(r - ours[arm])
        result["arms"][arm] = {"identical_predictions": int((r.argmax(1) == ours[arm].argmax(1)).sum()),
                               "model_eval_prediction_equals_argmax": bool((ref[arm].prediction.to_numpy() == r.argmax(1)).all()),
                               "max_abs_prob_diff": float(diff.max()), "mean_abs_prob_diff": float(diff.mean()),
                               "prior_correction_tau": float(ref[arm].prior_correction_tau.iloc[0])}
    # PIL (model_eval) vs this loader's decoded arrays are the same code path; also check against cv2 decoding.
    cv = np.stack([cv2.cvtColor(cv2.imread(h), cv2.COLOR_BGR2RGB) for h in he])
    result["pil_equals_cv2_rgb"] = bool((cv == rgb).all())
    result["mask_values"] = sorted(np.unique(m).tolist())
    result["passed"] = all(v["identical_predictions"] == len(sample) and v["max_abs_prob_diff"] < 1e-4
                           for v in result["arms"].values())
    save_json(OUT / "verification.json", result)
    print(json.dumps(result, indent=1))
    if not result["passed"]:
        raise SystemExit("verification FAILED")


# ----------------------------------------------------------------------------------------------- infer
def infer(args):
    import torch
    device = torch.device(args.device)
    crops = pd.read_csv(OUT / "crops.csv.gz")
    models = {}
    conditions = [c for c in CONDITIONS if c in args.conditions]
    with ProcessPoolExecutor(args.workers) as pool:
        for patient in args.patients:
            todo = [(k, c) for c in conditions for k in MODELS
                    if not (PRED / f"{k}_{c}" / f"{patient}.csv.gz").exists()]
            if not todo:
                continue
            t0 = time.time()
            df = crops[crops.patient == patient].reset_index(drop=True)
            rgb, m = load_crops(df)
            t_load = time.time() - t0
            stats = lab_stats(rgb)
            d = draws(len(df), patient)
            for cond in conditions:
                keys = [k for k, c in todo if c == cond]
                if not keys:
                    continue
                t1 = time.time()
                x = rgb if CONDITIONS[cond] is None else perturb(rgb, stats, d, CONDITIONS[cond], pool)
                t_pert = time.time() - t1
                for key in keys:
                    if key not in models:
                        models[key] = load_model(key, device)
                    t2 = time.time()
                    p = predict(models[key], x, m, device, args.batch)
                    folder = PRED / f"{key}_{cond}"
                    folder.mkdir(parents=True, exist_ok=True)
                    out = pd.DataFrame({"row": df.row, "label": df.label, **{f"p_{n}": p[:, i] for i, n in enumerate(NAMES)}})
                    write_csv(out, folder / f"{patient}.csv.gz")
                    print(f"{patient} {key}_{cond}: {len(df)} crops, load {t_load:.0f}s perturb {t_pert:.0f}s "
                          f"infer {time.time() - t2:.0f}s", flush=True)
            if args.save_examples and not (OUT / "examples" / f"{patient}.npz").exists():
                (OUT / "examples").mkdir(exist_ok=True)
                rng = np.random.default_rng([SEED, 3, int(patient[3:])])
                pick = np.concatenate([rng.choice(np.flatnonzero(df.label == c), 2, replace=False)
                                       for c in range(4) if (df.label == c).sum() >= 2])
                ex = {"clean": rgb[pick], "label": df.label.to_numpy()[pick]}
                for cond, lam in CONDITIONS.items():
                    if lam is not None:
                        ex[cond] = np.stack([perturb_one(rgb[i], stats, d[i], lam) for i in pick])
                np.savez_compressed(OUT / "examples" / f"{patient}.npz", **ex)


# ----------------------------------------------------------------------------------------------- metrics
def load_predictions(key, cond):
    parts = []
    for patient in PATIENTS:
        f = PRED / f"{key}_{cond}" / f"{patient}.csv.gz"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        df.insert(0, "patient", patient)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def arm_probs(arm, cond, cache, corr):
    key = ARMS[arm]["model"]
    if (key, cond) not in cache:
        cache[(key, cond)] = load_predictions(key, cond)
    df = cache[(key, cond)]
    if df is None:
        return None, None, None
    p = df[[f"p_{n}" for n in NAMES]].to_numpy()
    if ARMS[arm]["prior"]:
        p = apply_prior(p, corr)
    pid = df.patient.map({q: i for i, q in enumerate(PATIENTS)}).to_numpy()
    return p, df.label.to_numpy(), pid


def cm_metrics(cm):
    """cm[..., true, pred] -> dict of arrays (broadcast over leading axes).
    Precision is undefined (nan) for a class never predicted; F1 = 2TP / (support + predicted), so a class with
    support but no predictions scores 0. Macro-F1 and balanced accuracy average the classes with support."""
    cm = np.asarray(cm, np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, called = cm.sum(-1), cm.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(called > 0, tp / called, np.nan)
        rec = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(support + called > 0, 2 * tp / (support + called), np.nan)
        total = cm.sum((-2, -1))
        out = {"macro_F1": np.nanmean(np.where(support > 0, f1, np.nan), -1),
               "balanced_accuracy": np.nanmean(rec, -1),
               "accuracy": tp.sum(-1) / total}
        for i, n in enumerate(NAMES):
            out[f"precision_{n}"], out[f"recall_{n}"], out[f"F1_{n}"] = prec[..., i], rec[..., i], f1[..., i]
            out[f"true_prevalence_{n}"] = support[..., i] / total
            out[f"predicted_prevalence_{n}"] = called[..., i] / total
    return out


def auc_from_hist(pos, neg):
    """AUROC from binned score counts (ties within a bin count 1/2); pos/neg [..., bins]."""
    below = np.cumsum(neg, -1) - neg
    num = (pos * below).sum(-1) + 0.5 * (pos * neg).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / (pos.sum(-1) * neg.sum(-1))


def exact_auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))) if len(pos) and len(neg) else float("nan")


CAL_BINS = np.linspace(0, 1, 11)


def ece_from(cnt, sp, sy):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(sp - sy).sum(-1) / cnt.sum(-1)


def stats_for(p, y, pid, n_pat):
    """Per-patient additive sufficient statistics."""
    pred = p.argmax(1)
    cm = np.zeros((n_pat, 4, 4))
    np.add.at(cm, (pid, y, pred), 1)
    lymph = (y == 1) | (y == 3)
    score = p[lymph, 3]
    edges = np.unique(np.quantile(score, np.linspace(0, 1, 4097)))
    b = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 1)
    pos = np.zeros((n_pat, len(edges)))
    neg = np.zeros((n_pat, len(edges)))
    np.add.at(pos, (pid[lymph][y[lymph] == 3], b[y[lymph] == 3]), 1)
    np.add.at(neg, (pid[lymph][y[lymph] == 1], b[y[lymph] == 1]), 1)
    cal = {}
    for cls, n in [(3, "B"), (1, "T")]:
        k = np.clip(np.digitize(p[:, cls], CAL_BINS[1:-1]), 0, 9)
        cnt, sp, sy = (np.zeros((n_pat, 10)) for _ in range(3))
        np.add.at(cnt, (pid, k), 1)
        np.add.at(sp, (pid, k), p[:, cls])
        np.add.at(sy, (pid, k), (y == cls).astype(float))
        cal[n] = (cnt, sp, sy)
    yl, pl = y[lymph], pid[lymph]
    exact = exact_auc(score[yl == 3], score[yl == 1])
    per_pat = np.array([exact_auc(score[(pl == i) & (yl == 3)], score[(pl == i) & (yl == 1)]) for i in range(n_pat)])
    return {"cm": cm, "pos": pos, "neg": neg, "cal": cal, "auc_exact": exact, "auc_exact_per_patient": per_pat}


def metrics_from(st, w):
    """Metrics under patient weights w [..., n_pat] (w = ones -> point estimate; resample counts -> bootstrap)."""
    cm = np.tensordot(w, st["cm"], axes=(-1, 0))
    out = cm_metrics(cm)
    out["AUROC_B_vs_T"] = auc_from_hist(np.tensordot(w, st["pos"], (-1, 0)), np.tensordot(w, st["neg"], (-1, 0)))
    for n, (cnt, sp, sy) in st["cal"].items():
        out[f"ECE_{n}"] = ece_from(*(np.tensordot(w, a, (-1, 0)) for a in (cnt, sp, sy)))
    return out


def per_patient(st):
    n_pat = st["cm"].shape[0]
    out = cm_metrics(st["cm"])
    out["AUROC_B_vs_T"] = st["auc_exact_per_patient"]  # exact Mann-Whitney within each patient
    for n, (cnt, sp, sy) in st["cal"].items():
        out[f"ECE_{n}"] = ece_from(cnt, sp, sy)
    out["n_cells"] = st["cm"].sum((1, 2))
    return {k: np.asarray(v).reshape(n_pat) for k, v in out.items()}


def ci(boot):
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) else [None, None]


def boot_p(diff):
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    return float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))) if len(diff) else None


def metrics(args):
    corr = prior_weights()
    n_pat = len(PATIENTS)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_pat, (N_BOOT, n_pat))
    W = np.zeros((N_BOOT, n_pat))
    np.add.at(W, (np.repeat(np.arange(N_BOOT), n_pat), idx.ravel()), 1)
    ones = np.ones(n_pat)
    cache, results, pp_rows, boots, pmean_boots = {}, {}, [], {}, {}
    reliability = {}
    for arm in ARMS:
        for cond in CONDITIONS:
            p, y, pid = arm_probs(arm, cond, cache, corr)
            if p is None:
                print(f"skip {arm} {cond}: predictions incomplete")
                continue
            st = stats_for(p, y, pid, n_pat)
            point = {k: float(v) for k, v in metrics_from(st, ones).items()}
            boot = metrics_from(st, W)
            pp = per_patient(st)
            # patient-mean metrics: mean over patients of the per-patient value, resampled by the same draws
            pm_point = {k: float(np.nanmean(v)) for k, v in pp.items() if k != "n_cells"}
            pm_boot = {k: np.nanmean(v[idx], 1) for k, v in pp.items() if k != "n_cells"}
            boots[(arm, cond)], pmean_boots[(arm, cond)] = boot, pm_boot
            results.setdefault(arm, {})[cond] = {
                "n_cells": int(len(y)), "n_patients": n_pat,
                "AUROC_B_vs_T_exact_pooled": st["auc_exact"],
                "pooled": {k: {"value": v, "CI95": ci(boot[k])} for k, v in point.items()},
                "patient_mean": {k: {"value": v, "CI95": ci(pm_boot[k]),
                                     "patient_median": float(np.nanmedian(pp[k])),
                                     "patient_IQR": [float(np.nanpercentile(pp[k], 25)), float(np.nanpercentile(pp[k], 75))]}
                                 for k, v in pm_point.items()},
                "confusion_matrix_true_rows_pred_cols": st["cm"].sum(0).astype(int).tolist()}
            if cond == "clean":
                cal = {}
                for n, (cnt, sp, sy) in st["cal"].items():
                    c, s_p, s_y = cnt.sum(0), sp.sum(0), sy.sum(0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        cal[n] = {"bin_edges": CAL_BINS.tolist(), "count": c.astype(int).tolist(),
                                  "mean_predicted": (s_p / c).tolist(), "observed_fraction": (s_y / c).tolist(),
                                  "observed_CI95": [ci(v) for v in (np.tensordot(W, sy, (-1, 0)) / np.tensordot(W, cnt, (-1, 0))).T]}
                reliability[arm] = cal
                results[arm][cond]["per_patient_confusion"] = {q: st["cm"][i].astype(int).tolist() for i, q in enumerate(PATIENTS)}
            for i, q in enumerate(PATIENTS):
                pp_rows.append({"arm": arm, "arm_label": ARMS[arm]["label"], "condition": cond, "patient": q,
                                **{k: (float(v[i]) if k != "n_cells" else int(v[i])) for k, v in pp.items()}})
            print(f"{arm} {cond}: macro-F1 {point['macro_F1']:.3f} B-F1 {point['F1_B']:.3f} T-F1 {point['F1_T']:.3f} "
                  f"AUROC {point['AUROC_B_vs_T']:.4f} (exact {st['auc_exact']:.4f})", flush=True)

    key_metrics = ["macro_F1", "balanced_accuracy", "accuracy"] + [f"{m}_{n}" for n in NAMES for m in ("precision", "recall", "F1")] \
        + ["AUROC_B_vs_T", "ECE_B", "ECE_T"] + [f"predicted_prevalence_{n}" for n in NAMES]

    def diff_block(a, b, metrics_list, pooled=True):
        src = boots if pooled else pmean_boots
        res_a, res_b = results[a[0]][a[1]], results[b[0]][b[1]]
        out = {}
        for k in metrics_list:
            sa, sb = src[a].get(k), src[b].get(k)
            if sa is None:
                continue
            va = res_a["pooled" if pooled else "patient_mean"][k]["value"]
            vb = res_b["pooled" if pooled else "patient_mean"][k]["value"]
            d = sa - sb
            out[k] = {"difference": va - vb, "CI95": ci(d), "p_bootstrap": boot_p(d)}
        return out

    paired = {}
    for a, b in PAIRS:
        if (a, "clean") in boots and (b, "clean") in boots:
            paired[f"{a}-{b}"] = {"pooled": diff_block((a, "clean"), (b, "clean"), key_metrics),
                                  "patient_mean": diff_block((a, "clean"), (b, "clean"), key_metrics, pooled=False)}
    rob_metrics = ["macro_F1", "recall_B", "F1_B", "F1_T", "recall_T", "precision_B", "AUROC_B_vs_T", "balanced_accuracy",
                   "predicted_prevalence_B"]
    robustness = {"severities": {c: v for c, v in CONDITIONS.items() if v is not None}, "drop_clean_minus_perturbed": {},
                  "drop_difference_between_arms": {}}
    for arm in ARMS:
        for cond, lam in CONDITIONS.items():
            if lam is None or (arm, cond) not in boots:
                continue
            robustness["drop_clean_minus_perturbed"].setdefault(arm, {})[cond] = {
                "pooled": diff_block((arm, "clean"), (arm, cond), rob_metrics),
                "patient_mean": diff_block((arm, "clean"), (arm, cond), rob_metrics, pooled=False)}
    for a, b in PAIRS:
        for cond, lam in CONDITIONS.items():
            if lam is None or any((x, c) not in boots for x in (a, b) for c in ("clean", cond)):
                continue
            block = {}
            for k in rob_metrics:
                da = boots[(a, "clean")][k] - boots[(a, cond)][k]
                db = boots[(b, "clean")][k] - boots[(b, cond)][k]
                pa = results[a]["clean"]["pooled"][k]["value"] - results[a][cond]["pooled"][k]["value"]
                pb = results[b]["clean"]["pooled"][k]["value"] - results[b][cond]["pooled"][k]["value"]
                block[k] = {"difference_of_drops": pa - pb, "CI95": ci(da - db), "p_bootstrap": boot_p(da - db)}
            robustness["drop_difference_between_arms"].setdefault(f"{a}-{b}", {})[cond] = block

    pops = {q: json.loads((OUT / "crop_index" / f"{q}.population.json").read_text()) for q in PATIENTS}
    pop_counts = pd.DataFrame({q: v["class_counts"] for q, v in pops.items()}).T.fillna(0)
    pop_label = pd.DataFrame({n: pop_counts[[c for c, l in CLASS.items() if l == i and c in pop_counts]].sum(1)
                              for i, n in enumerate(NAMES)})
    summary = {
        "question": QUESTION, "caveats": CAVEATS,
        "patients": PATIENTS, "seed": SEED, "n_bootstrap": N_BOOT,
        "sampling": {"per_patient_cap": N_PER_PATIENT, "population_crops": int(sum(v["population_crops"] for v in pops.values())),
                     "evaluated_crops": int(sum(v["sampled"] for v in pops.values())),
                     "population_class_counts": {n: int(pop_label[n].sum()) for n in NAMES},
                     "population_prevalence": {n: float(pop_label[n].sum() / pop_label.values.sum()) for n in NAMES},
                     "per_patient": pops},
        "arms": {a: {**v, "checkpoint": str(MODELS[v["model"]].relative_to(ROOT)),
                     "prior_correction": ({"tau": corr["tau"], "weights": corr["weights"].tolist(),
                                           "sidecar": str(Path(corr["path"]).relative_to(ROOT))} if v["prior"] else None)}
                 for a, v in ARMS.items()},
        "excluded": "outputs/classifier_finetune_40pt_20261022/ (trained on CRC11-CRC40)",
        "verification": json.loads((OUT / "verification.json").read_text()) if (OUT / "verification.json").exists() else None,
        "results": results, "paired_differences_clean": paired, "robustness": robustness, "reliability_pooled": reliability,
        "notes": ["Pooled = all evaluated cells pooled (each patient <= 50,000 random crops); patient_mean = mean of per-patient values.",
                  "CIs: patient-cluster bootstrap, 2000 resamples of the 30 patients, 95% percentile; the same resamples for every arm and condition (paired).",
                  "AUROC_B_vs_T: P(B) separating true B from true T cells (cells with IF label T or B); bootstrap uses 4096 quantile bins of the pooled score (ties within a bin count 1/2); AUROC_B_vs_T_exact_pooled is the exact Mann-Whitney value.",
                  "ECE: 10 equal-width bins, one-vs-rest, all cells.",
                  "p_bootstrap: two-sided, 2 x min(share of resampled differences <= 0, >= 0); minimum resolvable 0.001."]}
    save_json(OUT / "summary.json", summary)
    pd.DataFrame(pp_rows).to_csv(OUT / "per_patient_metrics.csv", index=False, float_format="%.5g")
    print("wrote", OUT / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["index", "verify", "infer", "metrics", "all"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8, help="CPU processes for colour perturbation")
    parser.add_argument("--patients", nargs="+", default=PATIENTS)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--save-examples", action="store_true", default=True)
    args = parser.parse_args()
    PRED.mkdir(parents=True, exist_ok=True)
    stages = ["index", "verify", "infer", "metrics"] if args.stage == "all" else [args.stage]
    for s in stages:
        globals()[s](args)


if __name__ == "__main__":
    main()

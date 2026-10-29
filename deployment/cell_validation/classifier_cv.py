"""4-fold patient-level cross-validation of the deployed HE2CELL classifier recipe (held-out T/B cell accuracy).

Why: models/best_model2.pth was trained on all 40 Immunocto patients and the deployed fine-tune
(models/he2cell_esca_finetuned.pth) is initialised from it, so outputs/cell_validation_20261026/ reports
training-set numbers. Here the same fine-tuning recipe (deployment/finetune/finetune_classifier.py with the
settings of outputs/classifier_finetune_20261019/settings.json: strong stain/colour/scale/mask augmentation,
10 epochs x 200,000 class-balanced samples, batch 256, AdamW lr 1e-4 one-cycle, label smoothing 0.05,
checkpoint selection by holdout macro-F1 + ESCA-dev AUROC) is re-run four times from ImageNet weights
(`--init imagenet`, NOT from best_model2.pth) on 30 patients and scored on the 10 held-out patients.

Folds (contiguous blocks, folds.json): CRC01-10, CRC11-20, CRC21-30, CRC31-40. Within each fold's 30 training
patients, two are drawn (seeded) as the recipe's early-stopping/selection patients ("holdout" of the fine-tune
script); the remaining 28 are fitted. The tau 0.5 class-prior sidecar is fitted exactly as the deployed one:
normalised class counts of the fitted (28) patients, written next to the checkpoint as best_finetuned.prior.json
so script/src/model_eval.py picks it up automatically.

Evaluation reuses deployment/cell_validation/immunocto_holdout_eval.py by import: the same crop index for
CRC11-CRC40 (outputs/cell_validation_20261026/crop_index, read-only), the same <= 50,000 crops/patient sampling
with seed 20261026 for CRC01-CRC10, model_eval.py preprocessing, D4 mean softmax, the same seeded stain
perturbation draws (lambda 0.5 / 1.0 / 1.5), and the same patient-cluster bootstrap (2000 resamples).

Arms
  CV1  held-out fold model + its training-patient prior (tau 0.5)   <- the deployed recipe
  CV2  held-out fold model, no prior correction
Pooled over the 40 patients (each scored by the model that never saw it). Side by side with the existing
A1/A2/A3 numbers; for CRC11-CRC40 the comparison with A1-A3 is paired (same crops, same perturbation draws).

Stages (resumable):
  folds    write folds.json (fold assignment, selection patients, recipe)
  train    fine-tune the 4 fold models; run under deployment/ablations/gpu_locked.sh (GPU_LOCK_OWNER=classifier-cv)
  prior    fit each fold's tau 0.5 prior sidecar from its 28 fitted patients (checks the deployed prior is reproduced)
  index    crop index for CRC01-CRC10 (same rule as cell_validation), crops.csv.gz for all 40 patients
  verify   250 held-out crops per fold scored by model_eval.py (sidecar auto / none) vs this script
  infer    D4 probabilities per held-out patient and condition -> predictions/cv_<condition>/CRC##.csv.gz (mps, no lock)
  metrics  summary.json, per_patient_metrics.csv
Report: classifier_cv_report.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import immunocto_holdout_eval as E  # noqa: E402

OUT = ROOT / "outputs" / "classifier_cv_20261026"
PRED = OUT / "predictions"
A1_OUT = ROOT / "outputs" / "cell_validation_20261026"          # read-only: existing (training-set) evaluation
DEPLOYED = ROOT / "outputs" / "classifier_finetune_20261019"     # read-only: deployed fine-tune settings / prior
FINETUNE = ROOT / "deployment" / "finetune" / "finetune_classifier.py"
PY = ROOT / ".venv" / "bin" / "python"
SEED = E.SEED
N_BOOT = E.N_BOOT
NAMES = E.NAMES
CLASS = E.CLASS
CONDITIONS = E.CONDITIONS
ALL = [f"CRC{i:02d}" for i in range(1, 41)]
FOLDS = {f"fold{k}": [f"CRC{i:02d}" for i in range(10 * k + 1, 10 * k + 11)] for k in range(4)}
ARMS = {"CV1": {"label": "CV recipe + prior correction (tau 0.5)", "prior": True},
        "CV2": {"label": "CV recipe, no prior correction", "prior": False}}
EXISTING_ARMS = {"A1": {"model": "ft", "prior": True, "label": "Deployed fine-tune + prior (training set)"},
                 "A2": {"model": "ft", "prior": False, "label": "Deployed fine-tune, no prior (training set)"},
                 "A3": {"model": "orig", "prior": False, "label": "Original best_model2.pth (training set)"}}
TAU = 0.5
save_json, write_csv = E.save_json, E.write_csv


def fold_of(patient):
    return next(f for f, pts in FOLDS.items() if patient in pts)


def recipe():
    s = json.loads((DEPLOYED / "settings.json").read_text())
    return {k: s[k] for k in ["epochs", "samples_per_epoch", "batch", "lr", "workers", "style_range", "hed_alpha", "hed_beta", "scale_range"]}


# ----------------------------------------------------------------------------------------------- folds
def folds(args):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "folds.json"
    if path.exists():
        return json.loads(path.read_text())
    spec = {"design": "4-fold patient-level cross-validation, contiguous blocks of 10 Immunocto patients. Each fold's model is "
                      "trained on the other 30 patients (28 fitted + 2 selection patients drawn with seed [20261026, fold]) "
                      "and scored on its 10 held-out patients.",
            "seed": SEED, "init": "imagenet (torchvision ResNet50_Weights.DEFAULT); NOT models/best_model2.pth, which saw all 40 patients",
            "recipe": {**recipe(), "augmentation": "strong", "selection": "holdout macro-F1 + ESCA-dev AUROC of P(B) (TLS vs dense), as deployed",
                       "deployed_settings": str(DEPLOYED.relative_to(ROOT) / "settings.json")},
            "prior": {"tau": TAU, "rule": "normalised class counts (Other, T, Macrophage, B) of the fold's 28 fitted patients, as the deployed sidecar"},
            "folds": {}}
    for k, (name, test) in enumerate(FOLDS.items()):
        train = [p for p in ALL if p not in test]
        rng = np.random.default_rng([SEED, k])
        sel = sorted(rng.choice(train, 2, replace=False).tolist())
        spec["folds"][name] = {"test_patients": test, "fit_patients": [p for p in train if p not in sel], "selection_patients": sel,
                               "out": str((OUT / name).relative_to(ROOT))}
    save_json(path, spec)
    print(json.dumps({k: v["selection_patients"] for k, v in spec["folds"].items()}))
    return spec


def load_folds():
    return json.loads((OUT / "folds.json").read_text())


# ----------------------------------------------------------------------------------------------- train
def train(args):
    spec = load_folds()
    r = spec["recipe"]
    for name, f in spec["folds"].items():
        fdir = OUT / name
        if (fdir / "settings.json").exists() and (fdir / "best_finetuned.pth").exists():
            print(f"{name}: trained", flush=True)
            continue
        fdir.mkdir(parents=True, exist_ok=True)
        cmd = [str(PY), str(FINETUNE), "--init", "imagenet", "--out", str(fdir),
               "--patients", *f["fit_patients"], *f["selection_patients"], "--holdout", *f["selection_patients"],
               "--epochs", str(r["epochs"]), "--samples-per-epoch", str(r["samples_per_epoch"]), "--batch", str(r["batch"]),
               "--lr", str(r["lr"]), "--workers", str(r["workers"]), "--augmentation", "strong"]
        (fdir / "command.txt").write_text(" ".join(cmd) + "\n")
        t = time.time()
        print(f"{time.strftime('%F %T')} {name}: training ({len(f['fit_patients'])} fitted + {f['selection_patients']} selection)", flush=True)
        with open(fdir / "train.log", "a") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
        print(f"{time.strftime('%F %T')} {name}: finished rc={rc} in {(time.time() - t) / 60:.0f} min", flush=True)
        if rc != 0:
            raise SystemExit(f"{name} training failed (see {fdir / 'train.log'})")


# ----------------------------------------------------------------------------------------------- prior
def prior_from_counts(counts_csv, patients):
    counts = pd.read_csv(counts_csv, index_col=0)
    counts.columns = [int(c) for c in counts.columns]
    c = counts.loc[patients, [0, 1, 2, 3]].fillna(0).sum(0).to_numpy(float)
    return c / c.sum(), c


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior(args):
    spec = load_folds()
    # The rule must reproduce the deployed sidecar from the deployed run's class counts.
    dep = json.loads((DEPLOYED / "prior_correction_tau05.json").read_text())
    dep_fit = [p for p in [f"CRC{i:02d}" for i in range(1, 11)] if p not in json.loads((DEPLOYED / "settings.json").read_text())["holdout"]]
    p_dep, _ = prior_from_counts(DEPLOYED / "class_counts.csv", dep_fit)
    assert np.allclose(p_dep, dep["prior_Other_T_Macrophage_B"], atol=1e-12), (p_dep, dep)
    print("deployed prior reproduced from class_counts.csv:", np.round(p_dep, 6).tolist())
    for name, f in spec["folds"].items():
        fdir = OUT / name
        ckpt = fdir / "best_finetuned.pth"
        if not ckpt.exists():
            print(f"{name}: no checkpoint yet"); continue
        hist = json.loads((fdir / "history.json").read_text())
        scored = [h for h in hist if h["epoch"] > 0]
        best = max(scored, key=lambda h: h.get("selection_score", h["holdout_macro_F1"]))
        p, c = prior_from_counts(fdir / "class_counts.csv", f["fit_patients"])
        side = {"prior_Other_T_Macrophage_B": p.tolist(), "tau": TAU,
                "rule": "probabilities x prior**tau, renormalised; applied after D4 averaging",
                "source": f"class counts of the {len(f['fit_patients'])} fitted patients of {name} (selection patients {f['selection_patients']} excluded), "
                          "the same rule as outputs/classifier_finetune_20261019/prior_correction_tau05.json",
                "class_counts_Other_T_Macrophage_B": c.astype(int).tolist(),
                "checkpoint_sha256": sha256(ckpt), "checkpoint_source": f"{fdir.relative_to(ROOT)}/best_finetuned.pth (epoch {best['epoch']}; ImageNet init)",
                "selected_epoch": best["epoch"], "selection_score": best.get("selection_score"),
                "fit_patients": f["fit_patients"], "selection_patients": f["selection_patients"], "test_patients": f["test_patients"]}
        save_json(fdir / "best_finetuned.prior.json", side)
        print(f"{name}: prior {np.round(p, 4).tolist()} epoch {best['epoch']} score {best.get('selection_score', float('nan')):.4f}")


# ----------------------------------------------------------------------------------------------- index
def index_path(patient):
    return (A1_OUT if patient in E.PATIENTS else OUT) / "crop_index" / f"{patient}.csv.gz"


def index(args):
    """CRC01-CRC10 are indexed with the cell_validation rule (sorted names, rng [SEED, patient], <= 50,000); CRC11-CRC40
    reuse outputs/cell_validation_20261026/crop_index unchanged (same seeds, so the same crops)."""
    (OUT / "crop_index").mkdir(parents=True, exist_ok=True)
    for patient in ALL:
        if patient in E.PATIENTS:
            if not index_path(patient).exists():
                raise SystemExit(f"missing {index_path(patient)}")
            continue
        path = index_path(patient)
        if path.exists():
            continue
        folder = E.DATA / patient
        t = time.time()
        names = sorted(e.name for e in os.scandir(folder / "HE") if e.name.endswith(".png"))
        cls = [n.split("_")[0] for n in names]
        keep = [i for i, c in enumerate(cls) if c in CLASS]
        names = [names[i] for i in keep]
        full = pd.Series([cls[i] for i in keep]).value_counts().to_dict()
        rng = np.random.default_rng([SEED, int(patient[3:])])
        n = min(len(names), E.N_PER_PATIENT)
        pick = np.sort(rng.choice(len(names), n, replace=False))
        rows = []
        for i in pick:
            c, x, y = names[i][:-4].split("_")
            rows.append((patient, names[i], c, CLASS[c], int(x), int(y)))
        df = pd.DataFrame(rows, columns=["patient", "name", "immunocto_class", "label", "x", "y"])
        df.insert(1, "row", np.arange(len(df)))
        write_csv(df, path)
        save_json(OUT / "crop_index" / f"{patient}.population.json", {"population_crops": len(names), "sampled": n, "class_counts": full})
        print(f"{patient}: {len(names)} crops, sampled {n} ({time.time() - t:.0f}s)", flush=True)
    crops = pd.concat([pd.read_csv(index_path(p)) for p in ALL], ignore_index=True)
    write_csv(crops, OUT / "crops.csv.gz")
    print("crops:", len(crops), crops.groupby("label").size().to_dict())


def population(patient):
    return json.loads(((A1_OUT if patient in E.PATIENTS else OUT) / "crop_index" / f"{patient}.population.json").read_text())


# ----------------------------------------------------------------------------------------------- model
def load_fold_model(name, device):
    import torch
    sys.path.insert(0, str(E.SRC))
    from model_eval import SAM_ConvNet
    from runtime import load_classifier_state
    model = SAM_ConvNet(num_classes=4, loss_function="CrossEntropyLoss", pretrained=False)
    model.load_state_dict(load_classifier_state(OUT / name / "best_finetuned.pth"))
    return model.to(device).eval()


def fold_prior(name):
    sys.path.insert(0, str(E.SRC))
    from model_eval import load_prior_correction
    corr = load_prior_correction(OUT / name / "best_finetuned.pth", "auto")
    if corr is None:
        raise SystemExit(f"{name}: sidecar missing; run `prior` first")
    return corr


# ----------------------------------------------------------------------------------------------- verify
def verify(args):
    import torch
    crops = pd.read_csv(OUT / "crops.csv.gz")
    vdir = OUT / "verification"
    vdir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    result = {"n_per_fold": 250, "batch": args.batch, "device": args.device, "folds": {}}
    ok = True
    for name, test in FOLDS.items():
        sample = crops[crops.patient.isin(test)].sample(250, random_state=SEED).reset_index(drop=True)
        he, mask = E.crop_paths(sample)
        meta = vdir / f"meta_{name}.csv"
        pd.DataFrame({"cell_image_crop_path": he, "cell_mask_crop_path": mask, "global_instance_id": np.arange(len(sample)),
                      "label": sample.label}).to_csv(meta, index=False)
        ref = {}
        for arm, mode in [("CV1", "auto"), ("CV2", "none")]:
            out = vdir / f"model_eval_{name}_{arm}.csv"
            if not out.exists():
                subprocess.run([str(PY), str(E.SRC / "model_eval.py"), "--meta_path", str(meta), "--save_path", str(OUT / name / "best_finetuned.pth"),
                                "--output_path", str(out), "--device", args.device, "--batch_size", "512", "--num_workers", "0",
                                "--tta", "d4", "--prior_correction", mode], check=True, cwd=str(E.SRC))
            ref[arm] = pd.read_csv(out)
        rgb, m = E.load_crops(sample)
        p = E.predict(load_fold_model(name, device), rgb, m, device, args.batch)
        ours = {"CV1": E.apply_prior(p, fold_prior(name)), "CV2": p}
        cols = [f"prob_{n}" for n in NAMES]
        result["folds"][name] = {}
        for arm in ours:
            r = ref[arm][cols].to_numpy()
            diff = np.abs(r - ours[arm])
            result["folds"][name][arm] = {"identical_predictions": int((r.argmax(1) == ours[arm].argmax(1)).sum()),
                                          "max_abs_prob_diff": float(diff.max()), "mean_abs_prob_diff": float(diff.mean()),
                                          "prior_correction_tau": float(ref[arm].prior_correction_tau.iloc[0])}
            ok &= result["folds"][name][arm]["identical_predictions"] == len(sample) and diff.max() < 1e-4
        ok &= result["folds"][name]["CV1"]["prior_correction_tau"] == TAU and result["folds"][name]["CV2"]["prior_correction_tau"] == 0
    result["passed"] = bool(ok)
    save_json(OUT / "verification.json", result)
    print(json.dumps(result, indent=1))
    if not ok:
        raise SystemExit("verification FAILED")


# ----------------------------------------------------------------------------------------------- infer
def infer(args):
    import torch
    device = torch.device(args.device)
    crops = pd.read_csv(OUT / "crops.csv.gz")
    conditions = [c for c in CONDITIONS if c in args.conditions]
    with ProcessPoolExecutor(args.workers) as pool, open(OUT / "infer.log", "a") as log:
        for name, test in FOLDS.items():
            model = None
            for patient in test:
                if patient not in args.patients:
                    continue
                todo = [c for c in conditions if not (PRED / f"cv_{c}" / f"{patient}.csv.gz").exists()]
                if not todo:
                    continue
                if model is None:
                    model = load_fold_model(name, device)
                t0 = time.time()
                df = crops[crops.patient == patient].reset_index(drop=True)
                rgb, m = E.load_crops(df)
                t_load = time.time() - t0
                stats = E.lab_stats(rgb)
                d = E.draws(len(df), patient)
                for cond in todo:
                    t1 = time.time()
                    x = rgb if CONDITIONS[cond] is None else E.perturb(rgb, stats, d, CONDITIONS[cond], pool)
                    t_pert = time.time() - t1
                    t2 = time.time()
                    p = E.predict(model, x, m, device, args.batch)
                    folder = PRED / f"cv_{cond}"
                    folder.mkdir(parents=True, exist_ok=True)
                    out = pd.DataFrame({"row": df.row, "label": df.label, **{f"p_{n}": p[:, i] for i, n in enumerate(NAMES)}})
                    write_csv(out, folder / f"{patient}.csv.gz")
                    msg = f"{time.strftime('%F %T')} {patient} ({name}) cv_{cond}: {len(df)} crops, load {t_load:.0f}s perturb {t_pert:.0f}s infer {time.time() - t2:.0f}s"
                    print(msg, flush=True); log.write(msg + "\n"); log.flush()


# ----------------------------------------------------------------------------------------------- metrics
def load_pred(folder, patients):
    parts = []
    for q in patients:
        f = folder / f"{q}.csv.gz"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        df.insert(0, "patient", q)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def probs_cv(arm, cond, patients, priors):
    df = load_pred(PRED / f"cv_{cond}", patients)
    if df is None:
        return None
    p = df[[f"p_{n}" for n in NAMES]].to_numpy()
    if ARMS[arm]["prior"]:
        for name in FOLDS:
            sel = df.patient.isin(FOLDS[name]).to_numpy()
            if sel.any():
                p[sel] = E.apply_prior(p[sel], priors[name])
    return p, df.label.to_numpy(), df.patient.map({q: i for i, q in enumerate(patients)}).to_numpy()


def probs_existing(arm, cond, patients, corr_a1):
    key = EXISTING_ARMS[arm]["model"]
    df = load_pred(A1_OUT / "predictions" / f"{key}_{cond}", patients)
    if df is None:
        return None
    p = df[[f"p_{n}" for n in NAMES]].to_numpy()
    if EXISTING_ARMS[arm]["prior"]:
        p = E.apply_prior(p, corr_a1)
    return p, df.label.to_numpy(), df.patient.map({q: i for i, q in enumerate(patients)}).to_numpy()


def boot_weights(n_pat, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pat, (N_BOOT, n_pat))
    W = np.zeros((N_BOOT, n_pat))
    np.add.at(W, (np.repeat(np.arange(N_BOOT), n_pat), idx.ravel()), 1)
    return idx, W


def summarise(st, idx, W, patients):
    ones = np.ones(len(patients))
    point = {k: float(v) for k, v in E.metrics_from(st, ones).items()}
    boot = E.metrics_from(st, W)
    pp = E.per_patient(st)
    pm_boot = {k: np.nanmean(v[idx], 1) for k, v in pp.items() if k != "n_cells"}
    res = {"n_cells": int(st["cm"].sum()), "n_patients": len(patients), "AUROC_B_vs_T_exact_pooled": st["auc_exact"],
           "pooled": {k: {"value": v, "CI95": E.ci(boot[k])} for k, v in point.items()},
           "patient_mean": {k: {"value": float(np.nanmean(pp[k])), "CI95": E.ci(pm_boot[k]),
                                "patient_median": float(np.nanmedian(pp[k])),
                                "patient_IQR": [float(np.nanpercentile(pp[k], 25)), float(np.nanpercentile(pp[k], 75))]}
                            for k in pm_boot},
           "confusion_matrix_true_rows_pred_cols": st["cm"].sum(0).astype(int).tolist()}
    return res, boot, pm_boot, pp


KEY_METRICS = ["macro_F1", "balanced_accuracy", "accuracy"] + [f"{m}_{n}" for n in NAMES for m in ("precision", "recall", "F1")] \
    + ["AUROC_B_vs_T", "ECE_B", "ECE_T"] + [f"predicted_prevalence_{n}" for n in NAMES]
ROB_METRICS = ["macro_F1", "recall_B", "F1_B", "F1_T", "recall_T", "precision_B", "AUROC_B_vs_T", "balanced_accuracy", "predicted_prevalence_B"]


def diff_block(res, boots, a, b, metrics_list, scope="pooled"):
    out = {}
    for k in metrics_list:
        if k not in boots[a] or k not in boots[b]:
            continue
        d = boots[a][k] - boots[b][k]
        out[k] = {"difference": res[a][scope][k]["value"] - res[b][scope][k]["value"], "CI95": E.ci(d), "p_bootstrap": E.boot_p(d)}
    return out


def metrics(args):
    spec = load_folds()
    priors = {name: fold_prior(name) for name in FOLDS}
    sys.path.insert(0, str(E.SRC))
    from model_eval import load_prior_correction
    corr_a1 = load_prior_correction(E.MODELS["ft"], "auto")
    # (1) main: 40 held-out patients, CV1 / CV2, 4 conditions
    idx40, W40 = boot_weights(len(ALL), SEED)
    results, boots, pmboots, pp_rows, reliability = {}, {}, {}, [], {}
    for arm in ARMS:
        for cond in CONDITIONS:
            got = probs_cv(arm, cond, ALL, priors)
            if got is None:
                print(f"skip {arm} {cond}: predictions incomplete"); continue
            p, y, pid = got
            st = E.stats_for(p, y, pid, len(ALL))
            res, boot, pm_boot, pp = summarise(st, idx40, W40, ALL)
            results.setdefault(arm, {})[cond] = res
            boots[(arm, cond)], pmboots[(arm, cond)] = boot, pm_boot
            if cond == "clean":
                cal = {}
                for n, (cnt, sp, sy) in st["cal"].items():
                    c, s_p, s_y = cnt.sum(0), sp.sum(0), sy.sum(0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        cal[n] = {"bin_edges": E.CAL_BINS.tolist(), "count": c.astype(int).tolist(), "mean_predicted": (s_p / c).tolist(),
                                  "observed_fraction": (s_y / c).tolist(),
                                  "observed_CI95": [E.ci(v) for v in (np.tensordot(W40, sy, (-1, 0)) / np.tensordot(W40, cnt, (-1, 0))).T]}
                reliability[arm] = cal
                res["per_patient_confusion"] = {q: st["cm"][i].astype(int).tolist() for i, q in enumerate(ALL)}
                # per fold: point estimates and a 10-patient bootstrap within the fold
                res["per_fold"] = {}
                for k, (name, test) in enumerate(FOLDS.items()):
                    sel = np.isin(ALL, test)
                    sub = {"cm": st["cm"][sel], "pos": st["pos"][sel], "neg": st["neg"][sel], "cal": {n: tuple(a[sel] for a in v) for n, v in st["cal"].items()}}
                    idx10, W10 = boot_weights(10, [SEED, k])
                    pt = E.metrics_from(sub, np.ones(10)); bt = E.metrics_from(sub, W10)
                    res["per_fold"][name] = {m: {"value": float(pt[m]), "CI95": E.ci(bt[m])} for m in KEY_METRICS + [f"true_prevalence_{n}" for n in NAMES]}
            for i, q in enumerate(ALL):
                pp_rows.append({"arm": arm, "arm_label": ARMS[arm]["label"], "condition": cond, "patient": q, "fold": fold_of(q),
                                **{k: (float(v[i]) if k != "n_cells" else int(v[i])) for k, v in pp.items()}})
            print(f"{arm} {cond}: macro-F1 {res['pooled']['macro_F1']['value']:.3f} B-F1 {res['pooled']['F1_B']['value']:.3f} "
                  f"T-F1 {res['pooled']['F1_T']['value']:.3f} AUROC {res['pooled']['AUROC_B_vs_T']['value']:.4f} (exact {st['auc_exact']:.4f})", flush=True)
    key_res = {(a, c): results[a][c] for a in results for c in results[a]}
    robustness = {"severities": {c: v for c, v in CONDITIONS.items() if v is not None}, "drop_clean_minus_perturbed": {}}
    for arm in ARMS:
        for cond, lam in CONDITIONS.items():
            if lam is None or (arm, cond) not in boots:
                continue
            robustness["drop_clean_minus_perturbed"].setdefault(arm, {})[cond] = {
                "pooled": diff_block(key_res, boots, (arm, "clean"), (arm, cond), ROB_METRICS),
                "patient_mean": diff_block(key_res, pmboots, (arm, "clean"), (arm, cond), ROB_METRICS, "patient_mean")}
    paired_cv = {}
    if ("CV1", "clean") in boots and ("CV2", "clean") in boots:
        paired_cv["CV1-CV2"] = {"pooled": diff_block(key_res, boots, ("CV1", "clean"), ("CV2", "clean"), KEY_METRICS),
                                "patient_mean": diff_block(key_res, pmboots, ("CV1", "clean"), ("CV2", "clean"), KEY_METRICS, "patient_mean")}

    # (2) paired comparison with the existing arms on CRC11-CRC40 (same crops, same perturbation draws, same 30-patient resamples)
    P30 = E.PATIENTS
    idx30, W30 = boot_weights(len(P30), SEED)   # identical to immunocto_holdout_eval.metrics -> reproduces A1-A3 numbers
    res30, boots30 = {}, {}
    for arm in list(ARMS) + list(EXISTING_ARMS):
        for cond in CONDITIONS:
            got = probs_cv(arm, cond, P30, priors) if arm in ARMS else probs_existing(arm, cond, P30, corr_a1)
            if got is None:
                continue
            p, y, pid = got
            st = E.stats_for(p, y, pid, len(P30))
            res, boot, _, _ = summarise(st, idx30, W30, P30)
            res30.setdefault(arm, {})[cond] = {k: res[k] for k in ["n_cells", "n_patients", "AUROC_B_vs_T_exact_pooled", "pooled", "patient_mean", "confusion_matrix_true_rows_pred_cols"]}
            boots30[(arm, cond)] = boot
    key30 = {(a, c): res30[a][c] for a in res30 for c in res30[a]}
    paired30 = {}
    for a in ARMS:
        for b in EXISTING_ARMS:
            if (a, "clean") in boots30 and (b, "clean") in boots30:
                paired30[f"{a}-{b}"] = diff_block(key30, boots30, (a, "clean"), (b, "clean"), KEY_METRICS)
    rob30 = {"drop_clean_minus_perturbed": {}, "drop_difference_between_arms": {}}
    for arm in res30:
        for cond, lam in CONDITIONS.items():
            if lam is None or (arm, cond) not in boots30:
                continue
            rob30["drop_clean_minus_perturbed"].setdefault(arm, {})[cond] = diff_block(key30, boots30, (arm, "clean"), (arm, cond), ROB_METRICS)
    for a in ARMS:
        for b in EXISTING_ARMS:
            for cond, lam in CONDITIONS.items():
                if lam is None or any((x, c) not in boots30 for x in (a, b) for c in ("clean", cond)):
                    continue
                block = {}
                for k in ROB_METRICS:
                    da = boots30[(a, "clean")][k] - boots30[(a, cond)][k]
                    db = boots30[(b, "clean")][k] - boots30[(b, cond)][k]
                    pa = res30[a]["clean"]["pooled"][k]["value"] - res30[a][cond]["pooled"][k]["value"]
                    pb = res30[b]["clean"]["pooled"][k]["value"] - res30[b][cond]["pooled"][k]["value"]
                    block[k] = {"difference_of_drops": pa - pb, "CI95": E.ci(da - db), "p_bootstrap": E.boot_p(da - db)}
                rob30["drop_difference_between_arms"].setdefault(f"{a}-{b}", {})[cond] = block
    # consistency check against the published A1 summary
    a1_pub = json.loads((A1_OUT / "summary.json").read_text())["results"]
    check = {}
    for arm in EXISTING_ARMS:
        if arm in res30 and "clean" in res30[arm]:
            check[arm] = {m: {"published": a1_pub[arm]["clean"]["pooled"][m]["value"], "recomputed": res30[arm]["clean"]["pooled"][m]["value"]}
                          for m in ["macro_F1", "F1_B", "F1_T", "AUROC_B_vs_T"]}
            check[arm]["max_abs_diff"] = max(abs(v["published"] - v["recomputed"]) for v in check[arm].values() if isinstance(v, dict))

    pops = {q: population(q) for q in ALL}
    pop_counts = pd.DataFrame({q: v["class_counts"] for q, v in pops.items()}).T.fillna(0)
    pop_label = pd.DataFrame({n: pop_counts[[c for c, l in CLASS.items() if l == i and c in pop_counts]].sum(1) for i, n in enumerate(NAMES)})
    training = {}
    for name in FOLDS:
        fdir = OUT / name
        if (fdir / "history.json").exists():
            hist = json.loads((fdir / "history.json").read_text())
            side = json.loads((fdir / "best_finetuned.prior.json").read_text())
            training[name] = {"selected_epoch": side["selected_epoch"], "selection_score": side["selection_score"],
                              "prior_Other_T_Macrophage_B": side["prior_Other_T_Macrophage_B"], "checkpoint_sha256": side["checkpoint_sha256"],
                              "history": [{k: v for k, v in h.items() if not isinstance(v, dict)} for h in hist],
                              "settings": json.loads((fdir / "settings.json").read_text()) if (fdir / "settings.json").exists() else None}
    summary = {
        "question": "Held-out cell-level T/B accuracy of the deployed classifier recipe: 4-fold patient-level cross-validation on Immunocto CRC01-CRC40 "
                    "(ImageNet initialisation, same fine-tuning recipe and tau 0.5 prior rule as models/he2cell_esca_finetuned.pth)",
        "interpretation": [
            "CV1/CV2 numbers are out-of-patient estimates of the RECIPE: every patient is scored by a model that never saw it (neither in fitting nor in checkpoint selection).",
            "The deployed checkpoint itself cannot be evaluated out of patient (its initialisation saw all 40 patients). It was trained with the same recipe on fewer patients (8 fitted + 2 selection) "
            "from a stronger initialisation; on new patients of this domain (Immunocto-style colorectal H&E, IF-derived labels) it is expected to perform about like the CV estimate.",
            "A1/A2/A3 (outputs/cell_validation_20261026) are training-set numbers on CRC11-CRC40; CV minus A on the same 30 patients measures the optimism of those numbers (same crops, paired bootstrap).",
            "ImageNet-init decision: best_model2.pth was trained on all 40 patients, so any fold initialised from it would leak the held-out patients; torchvision ImageNet ResNet-50 weights are the only leakage-free initialisation available. "
            "This is the one way the CV recipe differs from the deployed fine-tune (plus 28 vs 8 fitted patients)."],
        "folds": spec, "patients": ALL, "seed": SEED, "n_bootstrap": N_BOOT,
        "sampling": {"per_patient_cap": E.N_PER_PATIENT, "population_crops": int(sum(v["population_crops"] for v in pops.values())),
                     "evaluated_crops": int(sum(v["sampled"] for v in pops.values())),
                     "population_class_counts": {n: int(pop_label[n].sum()) for n in NAMES},
                     "population_prevalence": {n: float(pop_label[n].sum() / pop_label.values.sum()) for n in NAMES},
                     "per_patient": pops},
        "arms": {a: {**v, "checkpoints": {name: f"{(OUT / name).relative_to(ROOT)}/best_finetuned.pth" for name in FOLDS},
                     "prior_correction": ({name: {"tau": priors[name]["tau"], "weights": priors[name]["weights"].tolist()} for name in FOLDS} if v["prior"] else None)}
                 for a, v in ARMS.items()},
        "training": training,
        "verification": json.loads((OUT / "verification.json").read_text()) if (OUT / "verification.json").exists() else None,
        "results": results, "paired_differences_clean": paired_cv, "robustness": robustness, "reliability_pooled": reliability,
        "comparison_CRC11_40": {"note": "CV arms and the existing A1/A2/A3 arms on the same 30 patients (CRC11-CRC40), same crops and perturbation draws, "
                                        "same 2000 patient resamples (seed 20261026) as outputs/cell_validation_20261026/summary.json; A-arm numbers are training-set numbers.",
                                "arms": {**{a: v["label"] for a, v in ARMS.items()}, **{a: v["label"] for a, v in EXISTING_ARMS.items()}},
                                "results": res30, "paired_differences_clean": paired30, "robustness": rob30, "recomputation_check_vs_published": check},
        "notes": ["Pooled = all evaluated cells pooled (each patient <= 50,000 random crops); patient_mean = mean of per-patient values.",
                  "CIs: patient-cluster bootstrap, 2000 resamples of the 40 patients (or of the 30 patients CRC11-CRC40 in comparison_CRC11_40), 95% percentile; "
                  "the same resamples for every arm and condition (paired). per_fold CIs resample the fold's 10 patients.",
                  "AUROC_B_vs_T: P(B) separating true B from true T cells; bootstrap uses 4096 quantile bins (ties within a bin count 1/2); AUROC_B_vs_T_exact_pooled is exact.",
                  "ECE: 10 equal-width bins, one-vs-rest, all cells.",
                  "p_bootstrap: two-sided, 2 x min(share of resampled differences <= 0, >= 0); minimum resolvable 0.001."]}
    save_json(OUT / "summary.json", summary)
    pd.DataFrame(pp_rows).to_csv(OUT / "per_patient_metrics.csv", index=False, float_format="%.5g")
    print("wrote", OUT / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=["folds", "train", "prior", "index", "verify", "infer", "metrics"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8, help="CPU processes for colour perturbation")
    parser.add_argument("--patients", nargs="+", default=ALL)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    for s in args.stages:
        globals()[s](args)


if __name__ == "__main__":
    main()

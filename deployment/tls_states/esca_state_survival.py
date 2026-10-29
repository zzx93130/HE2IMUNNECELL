"""TLS states and overall survival in TCGA-ESCA (Part C, prognosis hook). Re-runnable; INTERIM until the
whole-slide HE2CELL run (esca_wsi_burden.py --run) has finished every analysed slide.

Input per completed slide: slides/<barcode>/wsi_burden/{cells.csv.gz, tls.csv, summary.json, screen.npz}.
tls.csv has the frozen two-stage features of every stage-1 aggregate but not the state features (radial
profile, shape, macrophage share), so the frozen detection of esca_wsi_burden.detect_regions is re-run on the
saved cells and processed cores (CPU, no segmentation) to recover each accepted TLS mask; the reproduction is
checked against tls.csv (same aggregates, same B share). The frozen ESCA state model
(esca_tls_states.assign_states, fitted on window aggregates only) assigns a state to every accepted TLS.
Per-slide features are cached in outputs/tls_states_20261026/wsi_features/ and refreshed when a slide's
summary.json changes.

Per patient (only patients whose analysed slides are all complete, as in esca_survival.py):
state-specific TLS per cm² of artefact-free tissue (mean over slides) and the fraction of TLS in each state
(pooled over slides). Cox models are esca_survival.analyse unchanged: OS, stratified by histology, adjusted for
stage group, age and sex; HR per SD of log1p (densities) or per SD (fractions); tertiles; Schoenfeld; KM.

PRE-REGISTERED (fixed 2026-10-26, before any state-level survival estimate was computed; see PREREGISTRATION):
primary hypotheses H1-H3 below, Holm-adjusted across the three; everything else is secondary/exploratory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import esca_tls_states as S  # noqa: E402
import esca_analysis as A  # noqa: E402
import esca_wsi_burden as W  # noqa: E402
import esca_survival as SV  # noqa: E402
from esca_common import OUT as ESCA_OUT  # noqa: E402
from esca_qc import slide_qc  # noqa: E402

SURV = S.OUT / "survival"
CACHE = S.OUT / "wsi_features"
MIN_TLS_FOR_FRACTION = 3
PREREGISTRATION = {
    "date_fixed": "2026-10-26",
    "fixed_before": "any state-level survival estimate (the first interim run of this script)",
    "state_model": "tls_state_model.json 'primary' (BIC-chosen; k and names in the model file); B-follicular = the state with the highest mean B share",
    "primary_hypotheses": {
        "H1": {"exposure": "total_TLS_per_cm2", "definition": "accepted whole-slide TLS per cm² artefact-free tissue (log1p, per SD)",
               "expected_direction": "HR < 1 (more TLS, longer OS)"},
        "H2": {"exposure": "Bfol_TLS_per_cm2", "definition": "B-follicular-state TLS per cm² (log1p, per SD)",
               "expected_direction": "HR < 1"},
        "H3": {"exposure": "Bfol_fraction", "definition": f"fraction of accepted TLS in the B-follicular state (per SD), patients with >= {MIN_TLS_FOR_FRACTION} accepted TLS",
               "expected_direction": "HR < 1"}},
    "model": "Cox PH for OS stratified by histology (ESCC/EAC), adjusted for stage group (I/II, III/IV, missing), age per 10 y, sex; esca_survival.analyse",
    "multiplicity": "Holm across H1-H3 on the adjusted per-SD p-values; alpha 0.05 two-sided",
    "secondary": ["other primary-model states per cm²", "k = 3 model states per cm² and k = 3 B-follicular fraction",
                  "mean B-follicular posterior", "tertile models, complete-case stage, within-histology estimates",
                  "HookNet TLS per cm² on the same patients (comparator)"],
    "interim_rule": "interim results (incomplete whole-slide run) are descriptive only; the confirmatory reading is the final run",
}


def wsi_slide_features(barcode):
    """State features of every accepted whole-slide TLS of one slide (mirrors esca_wsi_burden.detect_regions)."""
    folder = ESCA_OUT / "slides" / barcode
    out = folder / W.RESULT
    summary_path = out / "summary.json"
    cache = CACHE / f"{barcode}.csv"
    if cache.exists() and cache.stat().st_mtime >= summary_path.stat().st_mtime:
        return pd.read_csv(cache)
    summary = json.loads(summary_path.read_text())
    hk_summary = json.loads((folder / "hooknet" / "summary.json").read_text())
    mpp = hk_summary["source_mpp"]
    cores = np.load(out / "screen.npz")["cores"]
    cells = pd.read_csv(out / "cells.csv.gz")
    saved = pd.read_csv(out / "tls.csv")
    maps_h = {"prediction_mask": np.load(folder / "hooknet" / "maps_4um.npz")["prediction_mask"]}
    features = W.tls_features(barcode)
    qc = slide_qc(folder)
    labels, _ = ndimage.label(cores, structure=np.ones((3, 3)))
    cells = cells.copy()
    core_label = labels[np.clip((cells.centroid_y_global // W.STRIDE).astype(int), 0, labels.shape[0] - 1),
                        np.clip((cells.centroid_x_global // W.STRIDE).astype(int), 0, labels.shape[1] - 1)]
    cells["window_id"] = [f"region_{r}" for r in core_label]
    rows = []
    for region, slc in enumerate(ndimage.find_objects(labels), 1):
        x0, y0 = slc[1].start * W.STRIDE, slc[0].start * W.STRIDE
        side = max(slc[1].stop - slc[1].start, slc[0].stop - slc[0].start) * W.STRIDE
        window = W.Window(x0, y0, x0 + side, y0 + side, f"region_{region}")
        x = A.window_inputs(folder, window, hk_summary, maps_h, features, qc, cells)
        inside = np.zeros((x["n"], x["n"]), bool)
        step = W.STRIDE * mpp / A.GRID_UM
        for cy, cx in zip(*np.nonzero(labels[slc] == region)):
            inside[int(cy * step):int(np.ceil((cy + 1) * step)), int(cx * step):int(np.ceil((cx + 1) * step))] = True
        x["valid"] = x["valid"] & inside
        proposals, _ = A.proposal_features(x, **W.STAGE1)
        A._PROPOSALS.clear()
        for index, proposal in enumerate(proposals, 1):
            accepted, _ = A.accept(proposal, W.PARAMS["b_share"], W.PARAMS["require_b_core"])
            if not accepted:
                continue
            f = S.object_features(proposal["mask"], x)
            rows.append({"barcode": barcode, "case_id": barcode[:12], "region": region, "proposal": index,
                         "matches_HookNet": proposal["matches_HookNet"], **f})
    frame = pd.DataFrame(rows)
    check = saved[saved.accepted]
    if len(frame) != len(check) or (len(frame) and not np.allclose(
            frame.sort_values(["region", "proposal"]).B_share.to_numpy(),
            check.sort_values(["region", "proposal"]).B_share.to_numpy(), equal_nan=True)):
        raise RuntimeError(f"{barcode}: re-run detection does not reproduce tls.csv ({len(frame)} vs {len(check)} accepted)")
    frame["valid_tissue_mm2"] = summary["valid_tissue_mm2"]
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def completed_slides():
    windows = pd.read_csv(S.ANALYSIS / "windows.csv")
    analysed = sorted(set(windows.barcode))
    done = [b for b in analysed if (ESCA_OUT / "slides" / b / W.RESULT / "summary.json").exists()
            and json.loads((ESCA_OUT / "slides" / b / W.RESULT / "summary.json").read_text()).get("status") == "complete"]
    expected = windows.groupby("case_id").barcode.nunique()
    return analysed, done, expected


def patient_state_table(model):
    analysed, done, expected = completed_slides()
    slides, tls = [], []
    for barcode in done:
        f = wsi_slide_features(barcode)
        tissue = json.loads((ESCA_OUT / "slides" / barcode / W.RESULT / "summary.json").read_text())["valid_tissue_mm2"]
        slides.append({"barcode": barcode, "case_id": barcode[:12], "valid_tissue_mm2": tissue, "accepted_TLS": len(f)})
        if len(f):
            tls.append(f)
    slides = pd.DataFrame(slides)
    tls = pd.concat(tls, ignore_index=True) if tls else pd.DataFrame(columns=["barcode", "case_id", *S.STATE_FEATURES])
    names = {w: model[w]["state_names"] for w in ["primary", "k3"]}
    if len(tls):
        p = S.assign_states(tls, "primary", model)
        k3 = S.assign_states(tls, "k3", model)
        tls["state_name"], tls["p_B_follicular"] = p.state_name, p.p_B_follicular
        tls["state_k3"] = k3.state_name
    complete_cases = [c for c, n in expected.items() if (slides.case_id == c).sum() == n] if len(slides) else []
    slides = slides[slides.case_id.isin(complete_cases)]
    tls = tls[tls.case_id.isin(complete_cases)]
    rows = []
    for case, s in slides.groupby("case_id"):
        t = tls[tls.case_id == case]
        row = {"case_id": case, "wsi_slides": len(s), "wsi_tissue_mm2": s.valid_tissue_mm2.sum(), "accepted_TLS": len(t)}
        per_slide = []
        for b in s.itertuples():
            tb = t[t.barcode == b.barcode]
            cm2 = b.valid_tissue_mm2 / 100
            d = {"total_TLS_per_cm2": len(tb) / cm2}
            for w, col, prefix in [("primary", "state_name", ""), ("k3", "state_k3", "k3_")]:
                for n in names[w]:
                    d[f"{prefix}{key_of(n)}_TLS_per_cm2"] = (tb[col] == n).sum() / cm2 if len(tb) else 0.0
            per_slide.append(d)
        row.update(pd.DataFrame(per_slide).mean().to_dict())
        for w, col, prefix in [("primary", "state_name", ""), ("k3", "state_k3", "k3_")]:
            for n in names[w]:
                row[f"{prefix}{key_of(n)}_fraction"] = (t[col] == n).mean() if len(t) >= MIN_TLS_FOR_FRACTION else np.nan
                row[f"{prefix}{key_of(n)}_fraction_min1"] = (t[col] == n).mean() if len(t) >= 1 else np.nan
        row["mean_p_B_follicular"] = t.p_B_follicular.mean() if len(t) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("case_id") if rows else pd.DataFrame(), tls, analysed, done


def key_of(name):
    return {"B-follicular": "Bfol", "T-dominant": "Tdom", "Macrophage-rich/mixed": "Macrich",
            "Lymphoid-poor (Other-rich)": "Lpoor"}.get(name, "".join(ch for ch in name if ch.isalnum())[:12])


def holm(pvalues):
    keys = [k for k, p in pvalues.items() if p is not None and np.isfinite(p)]
    order = sorted(keys, key=lambda k: pvalues[k])
    adjusted, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        adjusted[k] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    figures = SURV / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    prereg = S.OUT / "preregistration.json"
    if not prereg.exists():  # written once; never overwritten
        S.save_json({**PREREGISTRATION, "written": dt.datetime.now().isoformat(timespec="seconds")}, prereg)
    model = S.load_model()
    states, tls, analysed, done = patient_state_table(model)
    tls.to_csv(SURV / "wsi_tls_states.csv", index=False)
    burden = pd.read_csv(SV.ROOT / "outputs" / "esca_survival_20261025" / "patient_burden.csv").set_index("case_id")
    clinical = pd.read_csv(ESCA_OUT / "rna" / "clinical.csv").set_index("case_id")
    base_cols = ["slides", "hooknet_TLS_per_cm2", "hooknet_TLS_area_pct", "he2cell_random_B_per_mm2", "chemokine12_PC1"]
    table = burden[base_cols].join(states, how="left").join(clinical, how="left")
    # esca_survival.model_frame casts every BURDENS column; the whole-slide columns are rebuilt from states.
    table["he2cell_wsi_TLS_per_cm2"] = table.get("total_TLS_per_cm2")
    table["he2cell_wsi_B_TLS_area_pct"] = np.nan
    names = model["primary"]["state_names"]
    new = {"total_TLS_per_cm2": ("H1 Total HE2CELL TLS per cm²", True, "TLS/cm²"),
           "Bfol_TLS_per_cm2": ("H2 B-follicular-state TLS per cm²", True, "TLS/cm²"),
           "Bfol_fraction": (f"H3 Fraction of TLS in B-follicular state (≥ {MIN_TLS_FOR_FRACTION} TLS)", False, "fraction")}
    for n in names[1:]:
        new[f"{key_of(n)}_TLS_per_cm2"] = (f"{n}-state TLS per cm²", True, "TLS/cm²")
    for n in model["k3"]["state_names"]:
        new[f"k3_{key_of(n)}_TLS_per_cm2"] = (f"k=3 {n}-state TLS per cm²", True, "TLS/cm²")
    new["k3_Bfol_fraction"] = (f"k=3 fraction of TLS in B-follicular state (≥ {MIN_TLS_FOR_FRACTION} TLS)", False, "fraction")
    new["Bfol_fraction_min1"] = ("Fraction of TLS in B-follicular state (≥ 1 TLS)", False, "fraction")
    new["mean_p_B_follicular"] = ("Mean B-follicular posterior of TLS", False, "probability")
    for k in new:
        if k not in table:
            table[k] = np.nan
    SV.BURDENS.update(new)
    frame = SV.model_frame(table)
    eligible = frame[frame.time_months.notna() & (frame.time_months > 0) & frame.age_at_index.notna()]
    with_wsi = eligible[eligible.total_TLS_per_cm2.notna()]
    final = len(done) == len(analysed)
    label = "FINAL" if final else f"INTERIM, n = {len(with_wsi)} patients ({int(with_wsi.event.sum())} deaths; {len(done)}/{len(analysed)} slides complete)"
    rows, results, skipped = [], {}, {}
    for key in ["hooknet_TLS_per_cm2", *new]:
        data = frame if key != "hooknet_TLS_per_cm2" else frame[frame.index.isin(with_wsi.index)]
        available = eligible[eligible.index.isin(data.index) & eligible[key].notna()]
        if len(available) < SV.MIN_PATIENTS or available.event.sum() < SV.MIN_DEATHS:
            skipped[key] = {"patients": len(available), "deaths": int(available.event.sum()),
                            "reason": f"fewer than {SV.MIN_PATIENTS} patients or {SV.MIN_DEATHS} deaths"}
            continue
        try:
            r, result, _ = SV.analyse(data, key, figures)
        except Exception as error:
            skipped[key] = {"patients": len(available), "deaths": int(available.event.sum()), "reason": repr(error)}
            continue
        if key == "hooknet_TLS_per_cm2":
            result["label"] = "HookNet TLS per cm², same patients (comparator)"
        rows += r
        results[key] = result
    cox = pd.DataFrame(rows)
    cox.to_csv(SURV / "cox_tables.csv", index=False)
    adjusted = {}
    for h, spec in PREREGISTRATION["primary_hypotheses"].items():
        r = cox[(cox.burden == spec["exposure"]) & (cox.model == "adjusted (stratified by histology)") & (cox.term == "burden_sd")] if len(cox) else []
        adjusted[h] = None if len(r) == 0 else {"HR": float(r.HR.iloc[0]), "CI95": [float(r.CI95_low.iloc[0]), float(r.CI95_high.iloc[0])],
                                               "p": float(r.p.iloc[0]), "n": int(r.n.iloc[0]), "events": int(r.events.iloc[0])}
    holm_p = holm({h: v["p"] for h, v in adjusted.items() if v})
    for h in adjusted:
        if adjusted[h]:
            adjusted[h]["p_holm"] = holm_p.get(h)
    per_patient = frame[[c for c in dict.fromkeys(["total_TLS_per_cm2", *new, "accepted_TLS", "wsi_slides", "wsi_tissue_mm2",
                                     "hooknet_TLS_per_cm2", "histology", "stage_group", "age_at_index", "gender", "os_days", "os_event"]) if c in frame]]
    per_patient = per_patient[per_patient.total_TLS_per_cm2.notna()]
    per_patient.to_csv(SURV / "patient_state_burden.csv")
    summary = {"label": label, "final": final, "date": dt.date.today().isoformat(),
               "slides_complete": len(done), "slides_analysed": len(analysed),
               "patients_with_wsi": len(with_wsi), "deaths_with_wsi": int(with_wsi.event.sum()),
               "accepted_wsi_TLS": len(tls), "wsi_TLS_by_state": tls.state_name.value_counts().to_dict() if len(tls) else {},
               "wsi_TLS_by_state_k3": tls.state_k3.value_counts().to_dict() if len(tls) else {},
               "preregistration": json.loads(prereg.read_text()), "primary": adjusted, "results": results, "skipped": skipped,
               "power": SV.power(int(with_wsi.event.sum())) if len(with_wsi) else None,
               "patients_with_fraction": int(per_patient.Bfol_fraction.notna().sum()) if len(per_patient) else 0}
    S.save_json(summary, SURV / "state_survival_summary.json")
    S.save_json(summary, S.PARTS / "survival_states.json")
    print(label)
    print(json.dumps(adjusted, indent=1))


if __name__ == "__main__":
    main()

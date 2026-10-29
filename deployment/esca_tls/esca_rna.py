"""TCGA-ESCA bulk RNA TLS signatures and histology from public GDC/Xena endpoints.

Only public TCGA identifiers are sent; no images or predictions leave this Mac.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from esca_common import OUT, save_json

CHEMOKINES = ["CCL2", "CCL3", "CCL4", "CCL5", "CCL8", "CCL18", "CCL19", "CCL21",
              "CXCL9", "CXCL10", "CXCL11", "CXCL13"]
B_CELL = ["MS4A1", "CD19", "CD79A", "CR2"]
OTHER = ["LAMP3", "CD3E", "CD8A", "PTPRC"]
GENES = CHEMOKINES + B_CELL + OTHER
HOST = "https://gdc.xenahubs.net"
DATASET = "TCGA-ESCA.star_tpm.tsv"
DEST = OUT / "rna"


def xena(name, params):
    cache = DEST / "xena_cache"
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / f"{name}.xq"
    if not source.exists():
        url = f"https://raw.githubusercontent.com/ucscXena/xenaPython/master/xenaPython/queries/{name}.xq"
        source.write_bytes(urllib.request.urlopen(url, timeout=60).read())

    def marshall(value):
        if value is None:
            return "nil"
        if isinstance(value, list):
            return "[" + " ".join(marshall(item) for item in value) + "]"
        return json.dumps(value)

    query = "(" + source.read_text() + " " + " ".join(marshall(p) for p in params) + ")"
    request = urllib.request.Request(HOST + "/data/", query.encode(), headers={"Content-Type": "text/plain"})
    return json.loads(urllib.request.urlopen(request, timeout=120).read())


def histology_of(primary):
    return ("ESCC" if re.search(r"squamous", primary, re.I) else
            "EAC" if re.search(r"adenocarcinoma", primary, re.I) else "Other")


def stage_group(stage):
    """AJCC pathologic stage -> I/II, III/IV or missing (None, Unknown, Not Reported)."""
    match = re.fullmatch(r"Stage (I{1,3}|IV)[A-C]?", str(stage))
    if not match:
        return "missing"
    return "I/II" if match.group(1) in ("I", "II") else "III/IV"


def clinical():
    """One row per case. The first five columns keep their original meaning (first GDC diagnosis record).

    The *_primary columns come from the record flagged diagnosis_is_primary_disease: for 1 in 3 cases
    the first record is a metastasis, recurrence or prior primary without stage. Survival fields:
    OS time = days_to_death if Dead, else days_to_last_follow_up (primary record; else the largest
    over records). GDC renamed demographic.gender to sex_at_birth; gender takes whichever is present.
    """
    filters = {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-ESCA"]}}
    query = urllib.parse.urlencode({"filters": json.dumps(filters), "size": 1000, "format": "JSON",
        "fields": "submitter_id,diagnoses.primary_diagnosis,diagnoses.tissue_or_organ_of_origin,"
                  "diagnoses.ajcc_pathologic_stage,diagnoses.diagnosis_is_primary_disease,"
                  "diagnoses.days_to_last_follow_up,demographic.vital_status,demographic.days_to_death,"
                  "demographic.age_at_index,demographic.gender,demographic.sex_at_birth,demographic.cause_of_death"})
    hits = json.loads(urllib.request.urlopen("https://api.gdc.cancer.gov/cases?" + query, timeout=120).read())["data"]["hits"]
    rows = []
    for hit in hits:
        diagnoses = hit.get("diagnoses") or [{}]
        diagnosis = diagnoses[0]
        primary_record = next((d for d in diagnoses if d.get("diagnosis_is_primary_disease")), {})
        demographic = hit.get("demographic") or {}
        primary = diagnosis.get("primary_diagnosis", "Not Reported")
        follow_up = primary_record.get("days_to_last_follow_up")
        if follow_up is None:
            follow_up = max((d["days_to_last_follow_up"] for d in diagnoses if d.get("days_to_last_follow_up") is not None),
                            default=None)
        vital, death = demographic.get("vital_status"), demographic.get("days_to_death")
        os_days = death if vital == "Dead" else follow_up
        rows.append({"case_id": hit["submitter_id"], "primary_diagnosis": primary, "histology": histology_of(primary),
                     "site": diagnosis.get("tissue_or_organ_of_origin"),
                     "stage": diagnosis.get("ajcc_pathologic_stage"),
                     "primary_diagnosis_primary": primary_record.get("primary_diagnosis", "Not Reported"),
                     "histology_primary": histology_of(primary_record.get("primary_diagnosis", "Not Reported")),
                     "site_primary": primary_record.get("tissue_or_organ_of_origin"),
                     "stage_primary": primary_record.get("ajcc_pathologic_stage"),
                     "stage_group": stage_group(primary_record.get("ajcc_pathologic_stage")),
                     "vital_status": vital, "days_to_death": death, "days_to_last_follow_up": follow_up,
                     "os_days": os_days, "os_event": int(vital == "Dead") if vital in ("Dead", "Alive") else None,
                     "age_at_index": demographic.get("age_at_index"),
                     "gender": demographic.get("gender") or demographic.get("sex_at_birth"),
                     "cause_of_death": demographic.get("cause_of_death")})
    return pd.DataFrame(rows).sort_values("case_id")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    clinical().to_csv(DEST / "clinical.csv", index=False)
    if "--clinical-only" in sys.argv:
        print("clinical rows written")
        return
    metadata = xena("datasetMetadata", [DATASET])
    text = metadata[0].get("text", "{}")
    unit = str((json.loads(text) if isinstance(text, str) else text).get("unit", "")).lower().replace(" ", "")
    log_unit = re.fullmatch(r"log2\(tpm\+([0-9.]+)\)", unit)
    if not log_unit:
        raise ValueError(f"Unverified RNA unit {unit}")
    samples = [s for s in xena("datasetSamples", [DATASET, None])
               if s.startswith("TCGA-") and s[13:15] == "01"]
    values = xena("datasetGeneProbeAvg", [DATASET, samples, GENES])
    matrix = {}
    for row in values:
        scores = np.asarray(row["scores"], dtype=float).reshape(-1)
        if len(scores) != len(samples) or not np.isfinite(scores).all():
            raise ValueError(f"Missing values for {row.get('gene')}")
        matrix[row["gene"]] = scores  # log2(TPM+1), kept on the log scale
    missing = set(GENES) - set(matrix)
    if missing:
        raise ValueError(f"Genes missing from {DATASET}: {sorted(missing)}")
    expression = pd.DataFrame(matrix)[GENES]
    expression.insert(0, "sample_id", samples)
    expression.insert(0, "case_id", [s[:12] for s in samples])
    expression.to_csv(DEST / "log2_tpm_signature_genes.csv", index=False)

    patients = expression.groupby("case_id")[GENES].median()
    z = (patients - patients.mean()) / patients.std(ddof=1)
    # 12-chemokine score (Coppola 2011 / Messina 2012) as the first principal component, sign-aligned.
    u, s, vt = np.linalg.svd(z[CHEMOKINES].to_numpy(), full_matrices=False)
    loading = vt[0] * np.sign(vt[0].sum())
    scores = pd.DataFrame({
        "case_id": patients.index,
        "chemokine12_PC1": z[CHEMOKINES].to_numpy() @ loading,
        "chemokine12_mean_z": z[CHEMOKINES].mean(axis=1).to_numpy(),
        "B_cell_mean_z": z[B_CELL].mean(axis=1).to_numpy(),
        **{f"log2TPM_{gene}": patients[gene].to_numpy() for gene in GENES},
    })
    scores.to_csv(DEST / "patient_signatures.csv", index=False)
    save_json({"source": HOST, "dataset": DATASET, "unit": unit, "primary_tumor_samples": len(samples),
               "patients": len(patients), "genes": GENES,
               "chemokine12_PC1_loadings": dict(zip(CHEMOKINES, loading.round(4).tolist())),
               "chemokine12_PC1_variance_fraction": float(s[0] ** 2 / (s ** 2).sum()),
               "aggregation": "patient median of primary-tumor (01) aliquots on log2(TPM+1); z-scored within ESCA"},
              DEST / "rna_provenance.json")
    print(f"RNA: {len(samples)} samples / {len(patients)} patients; clinical rows written")


if __name__ == "__main__":
    main()

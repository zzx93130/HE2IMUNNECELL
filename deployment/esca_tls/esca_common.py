"""Shared paths and helpers for the TCGA-ESCA TLS validation workflow."""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "TCGA-ESCA_raw" / "TCGA-ESCA"
OUT = ROOT / "outputs" / "esca_tls_20261019"
TAR = Path("/Volumes/KESU 1/TCGA-ESCA.tar")
CLASSIFIER = ROOT / "models" / "best_model2.pth"
LABELS = ["Other", "T", "Macrophage", "B"]

# Histologies whose lymphoid aggregates are not TLS (haematological primaries).
EXCLUDED_DIAGNOSES = {"Malignant lymphoma, non-Hodgkin, NOS", "Multiple myeloma"}


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def slide_barcode(name):
    """TCGA-XX-XXXX-01Z-00-DX1 from a GDC diagnostic slide file name."""
    match = re.match(r"(TCGA-\w\w-\w{4}-\d\d\w-\d\d-\w{3}\d*)", Path(name).name)
    if not match:
        raise ValueError(f"Not a TCGA slide name: {name}")
    return match.group(1)


def slide_paths():
    return sorted(RAW.glob("*/*.svs"))


def slide_dir(stem):
    return OUT / "slides" / slide_barcode(stem)

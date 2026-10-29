"""Derive prior-corrected predictions (probabilities x prior^tau, renormalised) from a scored ESCA run."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True); parser.add_argument("--result", required=True)
parser.add_argument("--prior-json", required=True)
args = parser.parse_args()
correction = json.loads(Path(args.prior_json).read_text())
names = ["Other", "T", "Macrophage", "B"]
weights = np.asarray(correction["prior_Other_T_Macrophage_B"]) ** correction["tau"]
for folder in sorted(Path("outputs/esca_tls_20261019/slides").glob(f"*/{args.source}")):
    out = folder.parent / args.result
    out.mkdir(exist_ok=True)
    if (folder / "excluded.json").exists():
        (out / "excluded.json").write_text((folder / "excluded.json").read_text()); continue
    if not (folder / "cells.csv.gz").exists():
        continue
    cells = pd.read_csv(folder / "cells.csv.gz")
    p = cells[[f"prob_{k}" for k in names]].to_numpy() * weights
    p /= p.sum(1, keepdims=True)
    for i, k in enumerate(names):
        cells[f"prob_{k}"] = p[:, i]
    cells["celltype"], cells["confidence"] = np.array(names)[p.argmax(1)], p.max(1)
    cells.drop(columns=["cell_image_crop_path", "cell_mask_crop_path"], errors="ignore").to_csv(out / "cells.csv.gz", index=False)
    summary = json.loads((folder / "summary.json").read_text())
    summary.update({"prior_correction": correction, "counts": cells.celltype.value_counts().to_dict()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
print("done", args.result)

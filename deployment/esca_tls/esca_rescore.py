"""Re-classify the kept ESCA window crops with another checkpoint (no re-segmentation).

Reads each slide's he2cell_ft/cells.csv.gz (which lists the retained 64-px crops) and writes the
same table, with new predictions, to <result>/cells.csv.gz so esca_report.py can run with
ESCA_RESULT=<result>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esca_common import OUT, ROOT  # noqa: E402

SOURCE = "he2cell_ft"
PREDICTIONS = ["celltype", "prob_Other", "prob_T", "prob_Macrophage", "prob_B", "confidence", "raw_celltype", "raw_confidence"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--result", required=True, help="New per-slide folder name, e.g. he2cell_ft40")
    parser.add_argument("--tta", choices=["d4", "none"], default="d4", help="Test-time augmentation passed to model_eval.py")
    args = parser.parse_args()
    for folder in sorted((OUT / "slides").glob(f"*/{SOURCE}")):
        out = folder.parent / args.result
        if (out / "cells.csv.gz").exists():
            continue
        if (folder / "excluded.json").exists():
            out.mkdir(exist_ok=True)
            shutil.copy2(folder / "excluded.json", out / "excluded.json")
            continue
        if not (folder / "cells.csv.gz").exists():
            continue
        started = time.perf_counter()
        cells = pd.read_csv(folder / "cells.csv.gz")
        out.mkdir(exist_ok=True)
        meta = out / "_meta.csv"
        cells[["global_instance_id", "cell_image_crop_path", "cell_mask_crop_path"]].to_csv(meta, index=False)
        result = out / "_pred.csv"
        command = [sys.executable, ROOT / "script" / "src" / "model_eval.py", "--meta_path", meta,
                   "--save_path", args.classifier, "--output_path", result, "--device", "mps",
                   "--batch_size", 256, "--num_workers", 0, "--tta", args.tta]
        with open(out / "pipeline.log", "w") as log:
            subprocess.run(list(map(str, command)), check=True, stdout=log, stderr=subprocess.STDOUT)
        predicted = pd.read_csv(result)
        assert (predicted.global_instance_id.to_numpy() == cells.global_instance_id.to_numpy()).all()
        for column in PREDICTIONS:
            cells[column] = predicted[column].to_numpy()
        cells.to_csv(out / "cells.csv.gz", index=False)
        summary = json.loads((folder / "summary.json").read_text())
        summary.update({"classifier": args.classifier, "tta": args.tta, "rescored_from": SOURCE, "cells": len(cells),
                        "counts": cells.celltype.value_counts().to_dict(),
                        "rescore_seconds": time.perf_counter() - started})
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        meta.unlink(); result.unlink()
        print(f"{folder.parent.name}: {len(cells)} cells, {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

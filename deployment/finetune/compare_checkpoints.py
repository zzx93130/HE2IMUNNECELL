"""Score several classifier checkpoints on the same held-out CRC patients and the dev-ESCA crop bank.

Held-out metrics are reported on a class-balanced sample (≤ 6,000 per class, as during fine-tuning)
and on a natural-prevalence random sample, where B precision reflects how rare B cells really are.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import finetune_classifier as F
from model_eval import SAM_ConvNet
from runtime import load_classifier_state

HOLDOUT = ["CRC03", "CRC09", "CRC15", "CRC19", "CRC31", "CRC33", "CRC37", "CRC39"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    table = F.index_crops()
    hold = table[table.patient.isin(HOLDOUT)]
    balanced = pd.concat([g.sample(min(len(g), 6000), random_state=0) for _, g in hold.groupby("label")], ignore_index=True)
    natural = hold.sample(min(len(hold), 60000), random_state=0)
    bank = pd.read_csv(F.BANK)
    loaders = {"balanced": (DataLoader(F.Crops(balanced, False), batch_size=512, num_workers=args.workers), balanced.label.to_numpy()),
               "natural": (DataLoader(F.Crops(natural, False), batch_size=512, num_workers=args.workers), natural.label.to_numpy())}
    bank_loader = DataLoader(F.BankCrops(bank), batch_size=512, num_workers=args.workers)
    results = {"holdout_patients": HOLDOUT, "balanced_n": len(balanced), "natural_n": len(natural),
               "natural_class_counts": natural.label.value_counts().sort_index().to_dict(), "models": {}}
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        model = SAM_ConvNet(num_classes=4, pretrained=False)
        model.load_state_dict(load_classifier_state(Path(path)))
        model.to(device)
        entry = {}
        for split, (loader, y) in loaders.items():
            p = F.predict(model, loader, device)
            f1, recall, precision = F.macro_f1(y, p)
            entry[split] = {"macro_F1": f1, "recall": recall, "precision": precision}
        esca = F.evaluate(model, loaders["balanced"][0], loaders["balanced"][1], bank, bank_loader, device)
        entry["ESCA_dev"] = {k: v for k, v in esca.items() if k.startswith("ESCA_dev")}
        results["models"][name] = entry
        print(name, json.dumps({s: round(entry[s]["macro_F1"], 3) for s in loaders}),
              round(entry["ESCA_dev"]["ESCA_dev_AUROC_PB_TLS_vs_dense"], 3), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()

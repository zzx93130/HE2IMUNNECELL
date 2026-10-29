"""Test logit adjustment (add log training-class prior) for class-balanced fine-tunes."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import finetune_classifier as F
from compare_checkpoints import HOLDOUT
from model_eval import SAM_ConvNet
from runtime import load_classifier_state



def main():
    device = torch.device("mps")
    table = F.index_crops()
    train = table[~table.patient.isin(HOLDOUT)]
    prior = train.label.value_counts(normalize=True).sort_index().to_numpy()
    hold = table[table.patient.isin(HOLDOUT)]
    natural = hold.sample(60000, random_state=0)
    balanced = pd.concat([g.sample(min(len(g), 6000), random_state=0) for _, g in hold.groupby("label")], ignore_index=True)
    bank = pd.read_csv(F.BANK)
    loaders = {"natural": DataLoader(F.Crops(natural, False), batch_size=512, num_workers=10),
               "balanced": DataLoader(F.Crops(balanced, False), batch_size=512, num_workers=10),
               "bank": DataLoader(F.BankCrops(bank), batch_size=512, num_workers=10)}
    out = {"training_prior": prior.tolist()}
    for name, path in [("finetune10", "../../outputs/classifier_finetune_20261019/best_finetuned.pth"),
                       ("finetune40", "../../outputs/classifier_finetune_40pt_20261022/best_finetuned.pth")]:
        model = SAM_ConvNet(num_classes=4, pretrained=False); model.load_state_dict(load_classifier_state(Path(path))); model.to(device)
        probs = {k: F.predict(model, l, device) for k, l in loaders.items()}
        for tau in [0.0, 0.5, 1.0]:
            adj = {k: p * prior ** tau / (p * prior ** tau).sum(1, keepdims=True) for k, p in probs.items()}
            nat = F.macro_f1(natural.label.to_numpy(), adj["natural"]); bal = F.macro_f1(balanced.label.to_numpy(), adj["balanced"])
            q = adj["bank"]; call = q.argmax(1); tls = bank.in_hooknet_tls.to_numpy(); dense = (bank.category == "dense").to_numpy() & ~tls
            frac = pd.DataFrame({"b": bank.barcode, "t": tls, "B": call == 3})[tls].groupby("b").B.mean()
            r = {"natural_macro_F1": nat[0], "natural_precision": nat[2], "natural_recall": nat[1], "balanced_macro_F1": bal[0],
                 "ESCA_AUROC": F.auc(q[tls, 3], q[dense, 3]), "ESCA_B_in_TLS": float((call[tls] == 3).mean()),
                 "ESCA_B_in_dense": float((call[dense] == 3).mean()), "ESCA_T_in_TLS": float((call[tls] == 1).mean()),
                 "ESCA_slides_B_lt_2pct": int((frac < .02).sum())}
            out[f"{name}_tau{tau}"] = r
            print(name, tau, round(r["natural_macro_F1"], 3), round(r["balanced_macro_F1"], 3), {k: round(v, 2) for k, v in r["natural_precision"].items()},
                  round(r["ESCA_AUROC"], 3), round(r["ESCA_B_in_TLS"], 3), round(r["ESCA_B_in_dense"], 3), r["ESCA_slides_B_lt_2pct"], flush=True)
    Path("../../outputs/classifier_finetune_40pt_20261022/prior_correction.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

"""Fine-tune the HE2CELL four-class classifier with strong stain, colour, scale and mask augmentation.

Training data: Immunocto-style CRC crops (data/immunocto/CRC##/{HE,mask}/<class>_<x>_<y>.png).
Label order matches models/best_model2.pth: 0 Other, 1 T (cd4+cd8), 2 Macrophage, 3 B (bcell).
Starts from best_model2.pth. Two patients are held out for in-domain validation. Checkpoint
selection (pre-specified): holdout macro-F1 + dev-ESCA AUROC of P(B) separating cells inside
HookNet-TLS from cells in dense non-TLS windows. ESCA test patients are never used.
--init imagenet starts instead from torchvision's ImageNet ResNet-50 weights (mask encoder and head random);
used by the patient-level cross-validation (deployment/cell_validation/classifier_cv.py). Default unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "script" / "src"))
from model_eval import SAM_ConvNet  # noqa: E402
from runtime import load_classifier_state  # noqa: E402

DATA = ROOT / "data" / "immunocto"
OUT = ROOT / "outputs" / "classifier_finetune_20261019"
BANK = ROOT / "outputs" / "esca_tls_20261019" / "cropbank_dev" / "bank.csv"
CLASS = {"other": 0, "cd4": 1, "cd8": 1, "macrophage": 2, "bcell": 3}
NAMES = ["Other", "T", "Macrophage", "B"]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
# Ruifrok H&E(+residual) vectors, as in script/src/Utils/ColourAugment.py
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], np.cross([0.65, 0.70, 0.29], [0.07, 0.99, 0.11])], np.float32)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
# Broad H&E colour-style range for LAB style shifts (OpenCV 8-bit LAB units), not fitted to ESCA.
HED_ALPHA, HED_BETA = .15, .012
STYLE = {"L": (110, 190), "a": (135, 175), "b": (100, 132), "std_scale": (0.75, 1.35)}


def index_crops():
    rows = []
    for patient in sorted(DATA.glob("CRC*")):
        # CRC11+ are extracted with a .complete marker; skip any patient still being extracted.
        if int(patient.name[3:]) > 10 and not (patient / ".complete").exists():
            continue
        for path in (patient / "HE").glob("*.png"):
            cls = path.name.split("_")[0]
            if cls in CLASS:
                rows.append((patient.name, str(path), CLASS[cls]))
    return pd.DataFrame(rows, columns=["patient", "path", "label"])


def patient_lab_stats(table, n=400, seed=0):
    stats = {}
    rng = np.random.default_rng(seed)
    for patient, group in table.groupby("patient"):
        pixels = []
        for path in group.path.to_numpy()[rng.choice(len(group), min(n, len(group)), replace=False)]:
            lab = cv2.cvtColor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2LAB).reshape(-1, 3)
            pixels.append(lab[lab[:, 0] < 215])
        pixels = np.concatenate(pixels).astype(np.float32)
        stats[patient] = (pixels.mean(0), pixels.std(0))
    return stats


def to_tensor(rgb, mask):
    x = (rgb.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(np.concatenate([x, (mask > 0).astype(np.float32)[..., None]], 2)).permute(2, 0, 1).contiguous()


class Crops(Dataset):
    def __init__(self, table, augment, lab_stats=None, seed=0):
        self.paths, self.labels, self.patients = table.path.to_numpy(), table.label.to_numpy(), table.patient.to_numpy()
        self.augment, self.lab_stats, self.seed = augment, lab_stats, seed

    def __len__(self):
        return len(self.paths)

    def load(self, i):
        rgb = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.paths[i].replace("/HE/", "/mask/"), cv2.IMREAD_GRAYSCALE)
        return rgb, mask

    def __getitem__(self, i):
        rgb, mask = self.load(i)
        if self.augment == "weak":
            rgb, mask = augment_weak(rgb, mask, np.random.default_rng())
        elif self.augment:
            rng = np.random.default_rng()
            rgb, mask = augment(rgb, mask, rng, self.lab_stats[self.patients[i]])
        return to_tensor(rgb, mask), int(self.labels[i])


def augment(rgb, mask, rng, patient_stats):
    # Scale: training crops (~0.325 µm/px) vs TCGA 40x (~0.25 µm/px) -> zoom range covers both.
    s = rng.uniform(0.9, 1.45)
    size = max(8, int(round(64 * s)))
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    if size >= 64:
        o = (size - 64) // 2
        rgb, mask = rgb[o:o + 64, o:o + 64], mask[o:o + 64, o:o + 64]
    else:
        p = 64 - size
        a, b = p // 2, p - p // 2
        rgb = cv2.copyMakeBorder(rgb, a, b, a, b, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, a, b, a, b, cv2.BORDER_CONSTANT, value=0)
    # Mask jitter: Cellpose masks at inference are larger than the training masks.
    k = rng.choice([-1, 0, 1, 1, 2])
    if k > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=int(k))
    elif k < 0:
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = eroded if eroded.any() else mask
    # D4 symmetry, joint
    r = rng.integers(4)
    rgb, mask = np.rot90(rgb, r), np.rot90(mask, r)
    if rng.random() < .5:
        rgb, mask = rgb[:, ::-1], mask[:, ::-1]
    rgb = np.ascontiguousarray(rgb)
    mask = np.ascontiguousarray(mask)
    x = rgb.astype(np.float32) / 255
    # Global LAB style shift (p=0.7): move the patient's colour statistics to a random H&E style.
    if rng.random() < .7:
        lab = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        mean, std = patient_stats
        target_mean = np.array([rng.uniform(*STYLE["L"]), rng.uniform(*STYLE["a"]), rng.uniform(*STYLE["b"])], np.float32)
        target_std = std * rng.uniform(*STYLE["std_scale"], size=3).astype(np.float32)
        out = (lab - mean) / std * target_std + target_mean
        weight = np.clip((240 - lab[..., :1]) / 40, 0, 1)  # fade the shift out smoothly towards glass
        out = weight * out + (1 - weight) * lab
        x = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255
    # HED stain perturbation: strong multiplicative (±15%), additive kept at the original 0.012
    od = -np.log(np.clip(x, 1e-6, 1)).reshape(-1, 3) / -np.log(1e-6)
    hed = np.maximum(od @ HED_FROM_RGB, 0)
    hed = hed * rng.uniform(1 - HED_ALPHA, 1 + HED_ALPHA, 3) + rng.uniform(-HED_BETA, HED_BETA, 3)
    x = np.clip(np.exp(-(hed @ RGB_FROM_HED) * -np.log(1e-6)), 0, 1).reshape(64, 64, 3).astype(np.float32)
    # Brightness / contrast / saturation / hue / gamma
    x = x * rng.uniform(.9, 1.1) + rng.uniform(-.04, .04)
    m = x.mean()
    x = (x - m) * rng.uniform(.8, 1.2) + m
    hsv = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(.7, 1.35), 0, 1)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    x = np.clip(x, 0, 1) ** rng.uniform(.85, 1.2)
    if rng.random() < .25:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(.3, 1.1))
    if rng.random() < .2:
        x = x + rng.normal(0, .02, x.shape).astype(np.float32)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8), mask


def augment_weak(rgb, mask, rng):
    """Ablation control: the original HE2CELL training augmentation only (script/src/model_eval.py
    get_transforms): horizontal and vertical flips (p=0.4 each) and uniform HED perturbation ±0.012."""
    if rng.random() < .4:
        rgb, mask = rgb[:, ::-1], mask[:, ::-1]
    if rng.random() < .4:
        rgb, mask = rgb[::-1], mask[::-1]
    x = np.ascontiguousarray(rgb).astype(np.float32) / 255
    od = -np.log(np.clip(x, 1e-6, 1)).reshape(-1, 3) / -np.log(1e-6)
    hed = np.maximum(od @ HED_FROM_RGB, 0)
    sigma = 0.012  # the original get_transforms colour_sigma, used for both scale and shift
    hed = hed * rng.uniform(1 - sigma, 1 + sigma, 3) + rng.uniform(-sigma, sigma, 3)
    x = np.clip(np.exp(-(hed @ RGB_FROM_HED) * -np.log(1e-6)), 0, 1).reshape(64, 64, 3)
    return (x * 255).astype(np.uint8), np.ascontiguousarray(mask)


class BankCrops(Dataset):
    def __init__(self, table):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        rgb = cv2.cvtColor(cv2.imread(row.cell_image_crop_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row.cell_mask_crop_path, cv2.IMREAD_GRAYSCALE)
        return to_tensor(rgb, mask), i


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    return np.concatenate(out)


def macro_f1(y, p):
    pred = p.argmax(1)
    f1s, recall, precision = [], {}, {}
    for c in range(4):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        recall[NAMES[c]], precision[NAMES[c]] = float(rc), float(pr)
        f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
    return float(np.mean(f1s)), recall, precision


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))) if len(pos) and len(neg) else float("nan")


def evaluate(model, holdout_loader, holdout_y, bank, bank_loader, device):
    p = predict(model, holdout_loader, device)
    f1, recall, precision = macro_f1(holdout_y, p)
    result = {"holdout_macro_F1": f1, "holdout_recall": recall, "holdout_precision": precision}
    if bank is not None:
        q = predict(model, bank_loader, device)
        b = q[:, 3]
        call = q.argmax(1)
        tls = bank.in_hooknet_tls.to_numpy()
        dense = (bank.category == "dense").to_numpy() & ~tls
        per_slide = pd.DataFrame({"barcode": bank.barcode, "tls": tls, "isB": call == 3, "isT": call == 1})
        frac = per_slide[per_slide.tls].groupby("barcode").isB.mean()
        result.update({"ESCA_dev_AUROC_PB_TLS_vs_dense": auc(b[tls], b[dense]),
                       "ESCA_dev_B_fraction_in_TLS": float((call[tls] == 3).mean()),
                       "ESCA_dev_B_fraction_in_dense": float((call[dense] == 3).mean()),
                       "ESCA_dev_T_fraction_in_TLS": float((call[tls] == 1).mean()),
                       "ESCA_dev_T_fraction_in_dense": float((call[dense] == 1).mean()),
                       "ESCA_dev_slides_with_B_fraction_in_TLS_lt_0.02": int((frac < .02).sum()),
                       "ESCA_dev_slides": int(len(frac)),
                       "ESCA_dev_per_slide_B_fraction_in_TLS": frac.round(4).to_dict()})
        result["selection_score"] = f1 + result["ESCA_dev_AUROC_PB_TLS_vs_dense"]
    return result


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", nargs="+", default=["CRC03", "CRC09"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--init", default=str(ROOT / "models" / "best_model2.pth"),
                        help="Checkpoint to start from (default best_model2.pth), or 'imagenet' for torchvision ImageNet ResNet-50 weights")
    parser.add_argument("--out", default=str(OUT), help="Output folder (default: the 10-patient run)")
    parser.add_argument("--augmentation", choices=["strong", "weak"], default="strong",
                        help="strong: stain/colour/scale/mask augmentation (default); weak: original flips + HED ±0.012 only")
    parser.add_argument("--patients", nargs="+", default=None,
                        help="Restrict training+holdout to these patients (default: every extracted CRC folder)")
    args = parser.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(569); np.random.seed(569)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    table = index_crops()
    if args.patients:
        table = table[table.patient.isin(args.patients)]
    table.groupby(["patient", "label"]).size().unstack().to_csv(OUT / "class_counts.csv")
    train, hold = table[~table.patient.isin(args.holdout)], table[table.patient.isin(args.holdout)]
    hold = pd.concat([g.sample(min(len(g), 6000), random_state=0) for _, g in hold.groupby("label")], ignore_index=True)
    stats = patient_lab_stats(train)
    counts = train.label.value_counts().sort_index().to_numpy()
    weights = (1 / counts)[train.label.to_numpy()]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), args.samples_per_epoch, replacement=True)
    loader = DataLoader(Crops(train, True if args.augmentation == "strong" else "weak", stats), batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                        persistent_workers=True, prefetch_factor=4)
    holdout_loader = DataLoader(Crops(hold, False), batch_size=512, num_workers=args.workers)
    bank = pd.read_csv(BANK) if BANK.exists() else None
    bank_loader = DataLoader(BankCrops(bank), batch_size=512, num_workers=args.workers) if bank is not None else None

    if args.init == "imagenet":
        model = SAM_ConvNet(num_classes=4, pretrained=True)  # ImageNet backbone; mask_encoder and fc are random
        note = "ImageNet ResNet-50 initialisation (torchvision ResNet50_Weights.DEFAULT), untrained head"
    else:
        model = SAM_ConvNet(num_classes=4, pretrained=False)
        model.load_state_dict(load_classifier_state(Path(args.init)))
        note = "original best_model2.pth"
    model.to(device)
    history = [{"epoch": 0, "note": note,
                **evaluate(model, holdout_loader, hold.label.to_numpy(), bank, bank_loader, device)}]
    print(json.dumps({k: v for k, v in history[-1].items() if not isinstance(v, dict)}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(args.samples_per_epoch / args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=.1)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=.05)
    best = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total, seen = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(); scheduler.step()
            total += loss.item() * len(y); seen += len(y)
        result = {"epoch": epoch, "train_loss": total / seen, "epoch_seconds": time.time() - started,
                  **evaluate(model, holdout_loader, hold.label.to_numpy(), bank, bank_loader, device)}
        history.append(result)
        torch.save(model.state_dict(), OUT / f"epoch{epoch:02d}.pth")
        score = result.get("selection_score", result["holdout_macro_F1"])
        if score > best:
            best = score
            torch.save(model.state_dict(), OUT / "best_finetuned.pth")
        (OUT / "history.json").write_text(json.dumps(history, indent=1))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items() if not isinstance(v, dict)}), flush=True)
    (OUT / "settings.json").write_text(json.dumps({**vars(args), "doc": __doc__, "style_range": STYLE,
                                                   "hed_alpha": HED_ALPHA, "hed_beta": HED_BETA, "scale_range": [0.9, 1.45]}, indent=1))


if __name__ == "__main__":
    main()

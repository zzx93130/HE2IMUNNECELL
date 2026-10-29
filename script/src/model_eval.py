import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2

from torchmetrics.functional import accuracy, precision, recall, f1_score
from tqdm import tqdm
# 你自己的模块
import dataset
from Utils import ColourAugment
import argparse
import time
import json
from pathlib import Path
from runtime import select_device, load_classifier_state


def parse_args():
    parser = argparse.ArgumentParser(description="H&E cell-type inference on Apple GPU, CUDA, or CPU")

    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="Path to input meta CSV"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to model checkpoint .pth"
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use. Use 0 for CPU."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output CSV path. If not set, overwrite meta_path"
    )

    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto",
                        help="auto selects Apple GPU on this Mac; --num_gpus 0 forces CPU")
    parser.add_argument('--tta',choices=['d4','none'],default='d4',
                        help='Average eight joint RGB/mask softmax predictions by default')
    parser.add_argument('--prior_correction', choices=['auto', 'none'], default='auto',
                        help='auto applies <checkpoint>.prior.json when present (probabilities x prior**tau, renormalised); none disables it')
    return parser.parse_args()


def load_prior_correction(save_path, mode):
    """Class-prior correction stored next to a checkpoint, or None."""
    sidecar = Path(save_path).with_suffix(".prior.json")
    if mode == "none" or not sidecar.is_file():
        return None
    correction = json.loads(sidecar.read_text())
    weights = np.asarray(correction["prior_Other_T_Macrophage_B"], dtype=np.float64) ** float(correction["tau"])
    if weights.shape != (4,) or not np.all(weights > 0):
        raise ValueError(f"Invalid prior correction in {sidecar}")
    return {"path": str(sidecar), "tau": float(correction["tau"]), "weights": weights}


def apply_prior(probabilities, correction):
    adjusted = probabilities * correction["weights"]
    return adjusted / adjusted.sum(axis=1, keepdims=True)



class SAM_ConvNet(nn.Module):
    def __init__(self, num_classes=2, loss_function='CrossEntropyLoss', pretrained=True):
        super(SAM_ConvNet, self).__init__()
        self.num_classes = num_classes
        self.loss_fcn = getattr(nn, loss_function)()
        if pretrained:
            backbone = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            )
        else:
            backbone = torchvision.models.resnet50(weights=None)
        self.backbone = backbone
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        # 用于第 4 通道 mask
        self.mask_encoder = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=True
        )
    def forward(self, data):
        # data: [B, C, H, W]
        # 前 3 通道作为 RGB
        x = self.backbone.conv1(data[:, 0:3, :, :])
        # 如果有第 4 通道，则作为 mask 加入第一层特征
        if data.shape[1] > 3:
            mask = data[:, 3:4, :, :]
            x = x + self.mask_encoder(mask)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x
    def training_step(self, train_batch):
        image, labels = train_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        return loss, logits
    def validation_step(self, val_batch):
        image, labels = val_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        return loss, logits
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
        return optimizer, scheduler
    def get_metrics(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        metrics = {
            "accuracy": accuracy(
                preds,
                labels,
                task="multiclass",
                num_classes=self.num_classes
            ),
            "precision": precision(
                preds,
                labels,
                task="multiclass",
                average="macro",
                num_classes=self.num_classes
            ),
            "recall": recall(
                preds,
                labels,
                task="multiclass",
                average="macro",
                num_classes=self.num_classes
            ),
            "f1": f1_score(
                preds,
                labels,
                task="multiclass",
                average="macro",
                num_classes=self.num_classes
            )
        }
        return metrics


class ApplyToRGB(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
    def forward(self, x):
        x = x.clone()
        x[:3] = self.transform(x[:3])
        return x


def get_transforms(colour_sigma=0.012, colour_mode="uniform"):
    train_transform = v2.Compose([
        v2.ToImage(),
        # 如果你的图像输入是 0-255，建议 scale=True
        # 如果 dataset 里已经除以 255 变成 0-1，则用 scale=False
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.4),
        v2.RandomVerticalFlip(p=0.4),
        ApplyToRGB(
            ColourAugment.ColourAugment(
                sigma=colour_sigma,
                mode=colour_mode
            )
        ),
        ApplyToRGB(
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),

        ApplyToRGB(
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    ])
    return train_transform, val_transform


def set_seed(seed=569):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    meta_path = args.meta_path
    save_path = args.save_path
    output_path = args.output_path if args.output_path is not None else meta_path

    meta = pd.read_csv(meta_path)
    meta = meta.drop(['prediction', 'celltype'], axis=1, errors='ignore')
    required = {"cell_image_crop_path", "cell_mask_crop_path", "global_instance_id"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    device = select_device("cpu" if args.num_gpus == 0 else args.device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if meta.empty:
        meta["prediction"] = pd.Series(dtype="int64")
        meta["celltype"] = pd.Series(dtype="str")
        for column in ["prob_Other", "prob_T", "prob_Macrophage", "prob_B", "confidence", "probability_margin", "normalized_entropy"]:
            meta[column] = pd.Series(dtype="float64")
        meta['raw_prediction']=pd.Series(dtype='int64')
        meta['raw_celltype']=pd.Series(dtype='str')
        meta['prediction_method']=pd.Series(dtype='str')
        for name in ['Other','T','Macrophage','B']:
            meta['raw_prob_'+name]=pd.Series(dtype='float64')
        meta['raw_confidence']=pd.Series(dtype='float64')
        meta.to_csv(output, index=False)
        print(f"No cells to classify. Saved empty result to: {output}")
        return
    meta2 = meta[
        [
            "cell_image_crop_path",
            "cell_mask_crop_path",
            "global_instance_id"
        ]
    ]

    _, val_transform = get_transforms()
    val_dataset = dataset.CellSamdataset_our(meta2, val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None
    )

    num_classes = 4

    use_gpus = min(args.num_gpus, torch.cuda.device_count()) if device.type == "cuda" else 0
    print(f"Device: {device}; batch size: {args.batch_size}")

    model = SAM_ConvNet(
        num_classes=num_classes,
        loss_function="CrossEntropyLoss",
        pretrained=False
    )

    model.load_state_dict(load_classifier_state(save_path))

    model = model.to(device)

    if use_gpus > 1:
        device_ids = list(range(use_gpus))
        print(f"Using DataParallel on GPUs: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()

    all_samples = []
    all_predictions = []
    all_probabilities = []
    all_identity_probabilities = []
    inference_started = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, (images, sample) in enumerate(tqdm(val_loader)):
            images = images.to(device, non_blocking=True)

            if args.tta == 'd4':
                from prediction_tta import average_views
                probabilities_batch,views = average_views(model,images)
                identity = views[0]
            else:
                identity = model(images).softmax(dim=1)
                probabilities_batch = identity
            predictions = probabilities_batch.argmax(dim=1)
            all_probabilities.append(probabilities_batch.cpu().numpy())
            all_identity_probabilities.append(identity.cpu().numpy())

            all_samples.append(sample.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_samples = np.concatenate(all_samples)
    all_predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    identity_probabilities = np.concatenate(all_identity_probabilities)
    correction = load_prior_correction(save_path, args.prior_correction)
    uncorrected = probabilities
    if correction is not None:
        probabilities = apply_prior(probabilities, correction)
        identity_probabilities = apply_prior(identity_probabilities, correction)
        all_predictions = probabilities.argmax(axis=1)
        print(f"Applied class-prior correction (tau {correction['tau']}) from {correction['path']}")
    np.testing.assert_array_equal(all_samples,meta.global_instance_id.to_numpy())

    res = pd.DataFrame({
        "sample": all_samples,
        "prediction": all_predictions
    })

    # rename = {
    #     0: "Other",
    #     1: "CD4+T",
    #     2: "CD8+T",
    #     3: "Macrophage",
    #     4: "B"
    # }

    rename = {
            0: "Other",
            1: "T",
            2: "Macrophage",
            3: "B"
        }

    res["celltype"] = res["prediction"].map(rename)

    # Predictions follow the DataLoader's input order; preserve the slide's sample column.
    meta["prediction"] = all_predictions
    meta["celltype"] = res["celltype"].to_numpy()
    meta['raw_prediction'] = identity_probabilities.argmax(axis=1)
    meta['raw_celltype'] = np.array(['Other','T','Macrophage','B'])[meta.raw_prediction.to_numpy()]
    meta['raw_confidence'] = identity_probabilities.max(axis=1)
    meta['prediction_method'] = 'D4_mean_softmax' if args.tta=='d4' else 'identity_softmax'
    meta['prior_correction_tau'] = correction["tau"] if correction is not None else 0.0
    for index, name in enumerate(["Other", "T", "Macrophage", "B"]):
        meta[f"prob_{name}"] = probabilities[:, index]
        meta[f"raw_prob_{name}"] = identity_probabilities[:, index]
        if correction is not None:
            meta[f"uncorrected_prob_{name}"] = uncorrected[:, index]
    sorted_probs = np.sort(probabilities, axis=1)
    meta["confidence"] = sorted_probs[:, -1]
    meta["probability_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    meta["normalized_entropy"] = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1) / np.log(4)

    meta.to_csv(output_path, index=False)
    metrics = {"inference_including_data_loading_seconds": time.perf_counter() - inference_started,
               "cells": len(meta), "device": str(device), "test_time_augmentation":args.tta,
               "averaging":'softmax probabilities' if args.tta=='d4' else 'identity only',
               "prior_correction": ({"sidecar": correction["path"], "tau": correction["tau"],
                                     "weights": correction["weights"].tolist()} if correction is not None else None)}
    if device.type == "mps":
        metrics["mps_driver_allocated_bytes_at_end"] = torch.mps.driver_allocated_memory()
    output.with_suffix(".inference.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()

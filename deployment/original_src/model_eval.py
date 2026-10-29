import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

from torchmetrics.functional import accuracy, precision, recall, f1_score
from tqdm import tqdm
# 你自己的模块
import dataset
from Utils import ColourAugment
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SAM ConvNet multi-GPU inference")

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
        default=2048 * 8,
        help="Batch size for inference"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output CSV path. If not set, overwrite meta_path"
    )

    return parser.parse_args()



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
    meta = meta.drop(['prediction', 'celltype', "sample"], axis=1, errors='ignore')
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
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None
    )

    num_classes = 4

    available_gpus = torch.cuda.device_count()

    if args.num_gpus > 0 and torch.cuda.is_available():
        use_gpus = min(args.num_gpus, available_gpus)
        device = torch.device("cuda:0")
    else:
        use_gpus = 0
        device = torch.device("cpu")

    print(f"Available GPUs: {available_gpus}")
    print(f"Requested GPUs: {args.num_gpus}")
    print(f"Using GPUs: {use_gpus}")
    print(f"Device: {device}")

    model = SAM_ConvNet(
        num_classes=num_classes,
        loss_function="CrossEntropyLoss",
        pretrained=True
    )

    checkpoint = torch.load(save_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    if use_gpus > 1:
        device_ids = list(range(use_gpus))
        print(f"Using DataParallel on GPUs: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()

    all_samples = []
    all_predictions = []

    with torch.inference_mode():
        for batch_idx, (images, sample) in enumerate(tqdm(val_loader)):
            images = images.to(device, non_blocking=True)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            all_samples.append(sample.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_samples = np.concatenate(all_samples)
    all_predictions = np.concatenate(all_predictions)

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

    meta = meta.merge(
        res[["sample", "prediction", "celltype"]],
        left_on="global_instance_id",
        right_on="sample",
        how="left"
    )

    meta.to_csv(output_path, index=False)

    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
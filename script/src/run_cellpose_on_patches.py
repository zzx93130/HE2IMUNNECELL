import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm
from skimage.measure import regionprops

from cellpose import models


def read_patch_image(path):
    """
    读取 patch 图像，返回 H, W, C 格式。
    支持 png / tif。
    """
    path = Path(path)

    if path.suffix.lower() in [".tif", ".tiff"]:
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path).convert("RGB"))

    if img.ndim == 2:
        img = img[:, :, None]

    # 如果是 RGBA，只保留 RGB
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    # 如果单通道，复制成 3 通道
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    return img


def save_mask_tif(mask, mask_path):
    """
    保存 Cellpose 输出的 label mask。
    0 = background
    1,2,3... = cell instance
    """
    mask = mask.astype(np.uint32)
    tifffile.imwrite(
        mask_path,
        mask,
        dtype=np.uint32,
        compression="zlib",
    )


def get_core_region(
    patch_x,
    patch_y,
    patch_size,
    stride,
):
    """
    如果 stride < patch_size，说明 patch 之间有 overlap。
    为减少重复细胞，只保留 patch 中央 core 区域内的实例。

    这里返回 patch 内局部坐标范围：
    local_x_min, local_y_min, local_x_max, local_y_max

    如果 stride == patch_size，则整个 patch 都是 core。
    """
    overlap = patch_size - stride

    if overlap <= 0:
        return 0, 0, patch_size, patch_size

    margin = overlap // 2

    local_x_min = margin
    local_y_min = margin
    local_x_max = patch_size - margin
    local_y_max = patch_size - margin

    return local_x_min, local_y_min, local_x_max, local_y_max


def run_cellpose_on_patch_table(
    patch_csv,
    out_dir,
    model_name="cpsam_v2",
    gpu=True,
    batch_size=8,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    use_core_region=True,
):
    """
    对 patch_index.csv 中的所有 patch 运行 Cellpose-SAM，并输出：
    1. 每个 patch 的 mask tif
    2. 每个细胞实例在原始 TIFF 中的全局坐标信息 CSV
    """

    patch_csv = Path(patch_csv)
    out_dir = Path(out_dir)

    mask_dir = out_dir / "patch_masks"
    table_dir = out_dir / "mask_tables"

    mask_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    patch_df = pd.read_csv(patch_csv)

    required_cols = ["path", "sample", "x", "y", "patch_size", "stride"]
    for col in required_cols:
        if col not in patch_df.columns:
            raise ValueError(f"patch csv 缺少必要字段: {col}")

    print(f"[INFO] patch 数量: {len(patch_df)}")
    print(f"[INFO] Cellpose model: {model_name}")
    print(f"[INFO] gpu: {gpu}")

    model = models.CellposeModel(
        gpu=gpu,
        pretrained_model=model_name,
    )

    all_instance_rows = []
    global_instance_id = 1

    patch_records = patch_df.to_dict("records")

    for start in tqdm(
        range(0, len(patch_records), batch_size),
        desc="Running Cellpose-SAM",
    ):
        batch_records = patch_records[start:start + batch_size]

        imgs = []
        valid_records = []

        for record in batch_records:
            patch_path = Path(record["path"])

            if not patch_path.exists():
                print(f"[WARNING] patch 不存在，跳过: {patch_path}")
                continue

            img = read_patch_image(patch_path)
            imgs.append(img)
            valid_records.append(record)

        if len(imgs) == 0:
            continue

        masks, flows, styles = model.eval(
            imgs,
            channel_axis=-1,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            batch_size=batch_size,
        )

        for record, mask in zip(valid_records, masks):
            patch_path = Path(record["path"])

            sample = str(record["sample"])
            patch_x = int(record["x"])
            patch_y = int(record["y"])
            patch_size = int(record["patch_size"])
            stride = int(record["stride"])

            patch_id = int(record["num_patch"]) if "num_patch" in record else start

            sample_mask_dir = mask_dir / sample
            sample_mask_dir.mkdir(parents=True, exist_ok=True)

            mask_path = sample_mask_dir / f"{patch_path.stem}_cp_mask.tif"

            save_mask_tif(mask, mask_path)

            if use_core_region:
                core_x1, core_y1, core_x2, core_y2 = get_core_region(
                    patch_x=patch_x,
                    patch_y=patch_y,
                    patch_size=patch_size,
                    stride=stride,
                )
            else:
                core_x1, core_y1, core_x2, core_y2 = 0, 0, patch_size, patch_size

            for region in regionprops(mask.astype(np.int32)):
                local_label = int(region.label)

                # regionprops 的 centroid 顺序是 y, x
                centroid_y_local, centroid_x_local = region.centroid

                # 如果有 overlap，只保留 centroid 落在 core 区域的实例，减少重复
                if use_core_region:
                    if not (
                        core_x1 <= centroid_x_local < core_x2
                        and core_y1 <= centroid_y_local < core_y2
                    ):
                        continue

                min_y, min_x, max_y, max_x = region.bbox

                centroid_x_global = patch_x + centroid_x_local
                centroid_y_global = patch_y + centroid_y_local

                bbox_xmin_global = patch_x + min_x
                bbox_ymin_global = patch_y + min_y
                bbox_xmax_global = patch_x + max_x
                bbox_ymax_global = patch_y + max_y

                all_instance_rows.append({
                    "global_instance_id": global_instance_id,

                    "sample": sample,
                    "source_tif": record.get("source_tif", None),

                    "patch_path": str(patch_path),
                    "patch_mask_path": str(mask_path),
                    "patch_id": patch_id,

                    "patch_x": patch_x,
                    "patch_y": patch_y,
                    "patch_size": patch_size,
                    "stride": stride,

                    "local_label": local_label,

                    "area_pixels": int(region.area),

                    "centroid_x_local": float(centroid_x_local),
                    "centroid_y_local": float(centroid_y_local),

                    "centroid_x_global": float(centroid_x_global),
                    "centroid_y_global": float(centroid_y_global),

                    "bbox_xmin_local": int(min_x),
                    "bbox_ymin_local": int(min_y),
                    "bbox_xmax_local_exclusive": int(max_x),
                    "bbox_ymax_local_exclusive": int(max_y),

                    "bbox_xmin_global": int(bbox_xmin_global),
                    "bbox_ymin_global": int(bbox_ymin_global),
                    "bbox_xmax_global_exclusive": int(bbox_xmax_global),
                    "bbox_ymax_global_exclusive": int(bbox_ymax_global),

                    "core_xmin_local": int(core_x1),
                    "core_ymin_local": int(core_y1),
                    "core_xmax_local": int(core_x2),
                    "core_ymax_local": int(core_y2),
                })

                global_instance_id += 1

    instance_df = pd.DataFrame(all_instance_rows)

    out_csv = table_dir / f"{patch_csv.stem}_cellpose_mask_instances.csv"
    instance_df.to_csv(out_csv, index=False)

    print(f"[OK] Cellpose mask 已保存到: {mask_dir}")
    print(f"[OK] mask 实例坐标表已保存到: {out_csv}")
    print(f"[OK] 细胞实例数量: {len(instance_df)}")

    return instance_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patch_csv",
        type=str,
        required=True,
        help="前面 patch 提取生成的 patch_index.csv 或 all_patch_index.csv",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Cellpose 输出目录",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="cpsam_v2",
        help="Cellpose-SAM 模型名，默认 cpsam_v2",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="使用 GPU",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Cellpose batch size，默认 8",
    )

    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="细胞直径，默认 None。若细胞大小差异大，可先保持 None",
    )

    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=0.4,
        help="Cellpose flow threshold，默认 0.4",
    )

    parser.add_argument(
        "--cellprob_threshold",
        type=float,
        default=0.0,
        help="Cellpose cellprob threshold，默认 0.0",
    )

    parser.add_argument(
        "--min_size",
        type=int,
        default=15,
        help="最小实例面积，默认 15",
    )

    parser.add_argument(
        "--no_core_region",
        action="store_true",
        help="如果设置，则不使用 core region 去除 overlap 重复实例",
    )

    args = parser.parse_args()

    run_cellpose_on_patch_table(
        patch_csv=args.patch_csv,
        out_dir=args.out_dir,
        model_name=args.model_name,
        gpu=args.gpu,
        batch_size=args.batch_size,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        use_core_region=not args.no_core_region,
    )


if __name__ == "__main__":
    main()

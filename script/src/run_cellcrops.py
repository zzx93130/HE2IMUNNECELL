import argparse
import os
import multiprocessing as mp
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm
from skimage.measure import regionprops


INSTANCE_COLUMNS = [
    "worker_id",
    "worker_instance_id",
    "sample",
    "source_tif",
    "patch_path",
    "patch_mask_path",
    "patch_id",
    "patch_x",
    "patch_y",
    "patch_size",
    "stride",
    "local_label",
    "area_pixels",
    "centroid_x_local",
    "centroid_y_local",
    "centroid_x_global",
    "centroid_y_global",
    "bbox_xmin_local",
    "bbox_ymin_local",
    "bbox_xmax_local_exclusive",
    "bbox_ymax_local_exclusive",
    "bbox_xmin_global",
    "bbox_ymin_global",
    "bbox_xmax_global_exclusive",
    "bbox_ymax_global_exclusive",
    "core_xmin_local",
    "core_ymin_local",
    "core_xmax_local",
    "core_ymax_local",
    "cell_image_crop_path",
    "cell_mask_crop_path",
    "cell_mask_area_crop",
    "cell_crop_size",
    "cell_patch_result_dir",
    "cell_crop_status",
    "cell_file_status",
]


def safe_name(name):
    """
    将字符串转换为安全的文件夹/文件名。
    """
    name = str(name)
    name = name.replace("\\", "_")
    name = name.replace("/", "_")
    name = name.replace(":", "_")
    name = name.replace(" ", "_")
    name = name.replace("\t", "_")
    return name


def to_uint8_for_png(arr):
    """
    保存 PNG 前转成 uint8。

    如果 patch 本身就是 uint8，则不改变。
    如果是更高位深，默认进行 clip 到 0-255，避免直接报错。
    如需严格保存 16-bit 原始强度，建议把下面的 PNG 保存逻辑改成 tifffile.imwrite。
    """
    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr)
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    return np.clip(arr, 0, 255).astype(np.uint8)


def crop_with_padding(arr, center_x, center_y, crop_size=64, pad_value=0):
    """
    以 center_x, center_y 为中心裁剪 crop_size x crop_size。
    如果靠近 patch 边缘，则自动 padding。

    支持：
    - H x W mask
    - H x W x C image
    """

    half = crop_size // 2

    center_x = int(round(center_x))
    center_y = int(round(center_y))

    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    h, w = arr.shape[:2]

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if arr.ndim == 2:
        crop = np.full(
            (crop_size, crop_size),
            pad_value,
            dtype=arr.dtype,
        )

        crop[dst_y1:dst_y2, dst_x1:dst_x2] = arr[
            src_y1:src_y2,
            src_x1:src_x2,
        ]

    else:
        crop = np.full(
            (crop_size, crop_size, arr.shape[2]),
            pad_value,
            dtype=arr.dtype,
        )

        crop[dst_y1:dst_y2, dst_x1:dst_x2, :] = arr[
            src_y1:src_y2,
            src_x1:src_x2,
            :,
        ]

    return crop


def make_cell_name(sample, patch_path, patch_id, local_label):
    """
    生成稳定的单细胞文件名。

    注意：多 GPU 合并后才会重新生成 global_instance_id，
    所以这里不用 global_instance_id 命名，而使用 patch_id + local_label。
    """
    sample = safe_name(sample)
    patch_stem = safe_name(Path(patch_path).stem)

    try:
        patch_id = int(patch_id)
        patch_id_text = f"{patch_id:08d}"
    except Exception:
        patch_id_text = safe_name(patch_id)

    local_label = int(local_label)

    return f"{sample}_patch_{patch_id_text}_{patch_stem}_label_{local_label:04d}"


def save_cell_crop_outputs(
    patch_img,
    label_mask,
    local_label,
    centroid_x_local,
    centroid_y_local,
    sample,
    patch_path,
    patch_id,
    cell_out_dir,
    crop_size=64,
    background="white",
    overwrite=False,
    cell_image_mode="raw",
):
    """
    根据当前 patch 图像和 label mask，直接输出单细胞图像 crop 和单细胞二值 mask crop。

    cell_image_mode:
    - raw:       保存以细胞中心裁剪的原始图像 crop。
    - segmented:非当前细胞区域设置为 background 后保存。
    """

    patch_path = Path(patch_path)
    cell_out_dir = Path(cell_out_dir)

    if background == "white":
        bg_value = 255
        image_pad_value = 255
    elif background == "black":
        bg_value = 0
        image_pad_value = 0
    else:
        raise ValueError("background 只能是 'white' 或 'black'")

    # Crop labels before binarizing: compare 64x64 pixels instead of the full
    # 1024x1024 patch for every cell. Keep the historical empty/label-0 behavior.
    if local_label > 0:
        label_crop = crop_with_padding(label_mask, centroid_x_local, centroid_y_local,
                                       crop_size=crop_size, pad_value=0)
        mask_crop = (label_crop == local_label).astype(np.uint8) * 255
    else:
        mask_crop = crop_with_padding((label_mask == local_label).astype(np.uint8) * 255,
                                      centroid_x_local, centroid_y_local, crop_size=crop_size, pad_value=0)

    if not np.any(mask_crop) and not np.any(label_mask == local_label):
        return {
            "cell_image_crop_path": "",
            "cell_mask_crop_path": "",
            "cell_mask_area_crop": 0,
            "cell_crop_size": crop_size,
            "cell_patch_result_dir": "",
            "cell_crop_status": "label_not_found",
            "cell_file_status": "",
        }

    image_crop = crop_with_padding(
        patch_img,
        center_x=centroid_x_local,
        center_y=centroid_y_local,
        crop_size=crop_size,
        pad_value=image_pad_value,
    )

    if cell_image_mode == "segmented":
        image_to_save = image_crop.copy()
        if image_to_save.ndim == 2:
            image_to_save[mask_crop == 0] = bg_value
        else:
            image_to_save[mask_crop == 0, :] = bg_value
    elif cell_image_mode == "raw":
        image_to_save = image_crop
    else:
        raise ValueError("cell_image_mode 只能是 'raw' 或 'segmented'")

    folder_name = safe_name(patch_path.parent.name)
    patch_stem = safe_name(patch_path.stem)
    sample_name = safe_name(sample)

    patch_out_dir = cell_out_dir / sample_name / folder_name / patch_stem
    image_dir = patch_out_dir / f"cell_images_{crop_size}"
    mask_dir = patch_out_dir / f"cell_masks_{crop_size}"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cell_name = make_cell_name(sample, patch_path, patch_id, local_label)

    image_path = image_dir / f"{cell_name}_image_{crop_size}.png"
    mask_path = mask_dir / f"{cell_name}_mask_{crop_size}.png"

    if overwrite or not image_path.exists():
        Image.fromarray(to_uint8_for_png(image_to_save)).save(image_path)

    if overwrite or not mask_path.exists():
        Image.fromarray(mask_crop.astype(np.uint8)).save(mask_path)

    image_ok = image_path.exists() and os.path.getsize(image_path) > 0
    mask_ok = mask_path.exists() and os.path.getsize(mask_path) > 0
    file_status = "ok" if image_ok and mask_ok else "none"

    return {
        "cell_image_crop_path": str(image_path),
        "cell_mask_crop_path": str(mask_path),
        "cell_mask_area_crop": int(np.sum(mask_crop == 255)),
        "cell_crop_size": crop_size,
        "cell_patch_result_dir": str(patch_out_dir),
        "cell_crop_status": "ok" if file_status == "ok" else "file_error",
        "cell_file_status": file_status,
    }


def empty_cell_crop_info(crop_size=64, status="not_enabled"):
    return {
        "cell_image_crop_path": "",
        "cell_mask_crop_path": "",
        "cell_mask_area_crop": 0,
        "cell_crop_size": crop_size,
        "cell_patch_result_dir": "",
        "cell_crop_status": status,
        "cell_file_status": "",
    }


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


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def parse_gpu_ids(gpu_ids_text, num_gpus):
    if gpu_ids_text is None or str(gpu_ids_text).strip() == "":
        return list(range(num_gpus))

    gpu_ids = [int(x.strip()) for x in str(gpu_ids_text).split(",") if x.strip() != ""]
    if len(gpu_ids) < num_gpus:
        raise ValueError(
            f"--gpu_ids 只给了 {len(gpu_ids)} 张卡，但 --num_gpus={num_gpus}"
        )
    return gpu_ids[:num_gpus]


def infer_patch_id(record, fallback):
    if "num_patch" in record:
        return safe_int(record.get("num_patch"), fallback)
    return fallback


def run_cellpose_on_patch_records(
    records,
    out_dir,
    model_name="cpsam_v2",
    gpu=True,
    batch_size=8,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    use_core_region=True,
    worker_id=0,
    desc="Running Cellpose-SAM",
    crop_cells=True,
    cell_out_dir=None,
    crop_size=64,
    cell_background="white",
    overwrite_cell_crops=False,
    cell_image_mode="raw",
    device=None,
    attention="auto",
):
    """
    单个进程处理一部分 patch records。

    多 GPU 模式下，每个 worker 会在自己的进程中加载一个 CellposeModel。
    每个 patch 得到 Cellpose mask 后，会立即输出该 patch 内每个细胞的 image crop 和 mask crop。
    """
    from cellpose import models
    from runtime import select_device

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected_device = select_device(device or ("auto" if gpu else "cpu"))
    if model_name not in {"cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"} and not Path(model_name).is_file():
        raise FileNotFoundError(f"Cellpose model not found: {model_name}")

    out_dir = Path(out_dir)
    mask_dir = out_dir / "patch_masks"
    table_dir = out_dir / "mask_tables"
    mask_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if cell_out_dir is None:
        cell_out_dir = out_dir / "single_cells"
    else:
        cell_out_dir = Path(cell_out_dir)

    if crop_cells:
        cell_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO][worker {worker_id}] patch 数量: {len(records)}")
    print(f"[INFO][worker {worker_id}] Cellpose model: {model_name}")
    print(f"[INFO][worker {worker_id}] device: {selected_device}")
    print(f"[INFO][worker {worker_id}] crop_cells: {crop_cells}")
    if crop_cells:
        print(f"[INFO][worker {worker_id}] cell_out_dir: {cell_out_dir}")
        print(f"[INFO][worker {worker_id}] crop_size: {crop_size}")
        print(f"[INFO][worker {worker_id}] cell_image_mode: {cell_image_mode}")

    if not records:
        return pd.DataFrame(columns=INSTANCE_COLUMNS)
    print(f"[INFO] Segmentation device: {selected_device}")
    model = models.CellposeModel(
        gpu=selected_device.type != "cpu",
        device=selected_device,
        pretrained_model=model_name,
        use_bfloat16=False,
    )
    use_sdpa = attention == "sdpa" or (
        attention == "auto" and selected_device.type == "mps" and model.net.backbone == "sam_vitl")
    if use_sdpa:
        from segmentation_acceleration import enable_sdpa_attention
        print(f"[INFO] Optimized attention: {enable_sdpa_attention(model)} layers")

    all_instance_rows = []
    worker_instance_id = 1
    import torch
    processing_started = time.perf_counter()
    eval_seconds = 0.0
    maximum_observed_mps_driver_bytes = 0
    progress_path = out_dir / "progress.json"

    for start in tqdm(
        range(0, len(records), batch_size),
        desc=desc,
        position=worker_id,
        leave=True,
    ):
        batch_records = records[start:start + batch_size]

        imgs = []
        valid_records = []

        for record in batch_records:
            patch_path = Path(record["path"])

            if not patch_path.exists():
                print(f"[WARNING][worker {worker_id}] patch 不存在，跳过: {patch_path}")
                continue

            img = read_patch_image(patch_path)
            imgs.append(img)
            valid_records.append(record)

        if len(imgs) == 0:
            continue

        eval_started = time.perf_counter()
        masks, flows, styles = model.eval(
            imgs,
            channel_axis=-1,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            batch_size=batch_size,
        )

        eval_seconds += time.perf_counter() - eval_started
        if selected_device.type == "mps":
            maximum_observed_mps_driver_bytes = max(maximum_observed_mps_driver_bytes, torch.mps.driver_allocated_memory())

        for record, patch_img, mask in zip(valid_records, imgs, masks):
            patch_path = Path(record["path"])

            sample = str(record["sample"])
            patch_x = safe_int(record["x"])
            patch_y = safe_int(record["y"])
            patch_size = safe_int(record["patch_size"])
            stride = safe_int(record["stride"])

            patch_id = infer_patch_id(record, fallback=start)

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

                if crop_cells:
                    try:
                        crop_info = save_cell_crop_outputs(
                            patch_img=patch_img,
                            label_mask=mask,
                            local_label=local_label,
                            centroid_x_local=centroid_x_local,
                            centroid_y_local=centroid_y_local,
                            sample=sample,
                            patch_path=patch_path,
                            patch_id=patch_id,
                            cell_out_dir=cell_out_dir,
                            crop_size=crop_size,
                            background=cell_background,
                            overwrite=overwrite_cell_crops,
                            cell_image_mode=cell_image_mode,
                        )
                    except Exception as e:
                        print(
                            f"[WARNING][worker {worker_id}] 单细胞 crop 失败: "
                            f"{patch_path}, label={local_label}, error={repr(e)}"
                        )
                        crop_info = empty_cell_crop_info(crop_size=crop_size, status="failed")
                else:
                    crop_info = empty_cell_crop_info(crop_size=crop_size, status="not_enabled")

                row = {
                    "worker_id": worker_id,
                    "worker_instance_id": worker_instance_id,

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
                }

                row.update(crop_info)
                all_instance_rows.append(row)

                worker_instance_id += 1

        progress = {
            "patches_completed": min(start + batch_size, len(records)),
            "patches_total": len(records), "cells_retained": len(all_instance_rows),
            "elapsed_seconds": time.perf_counter() - processing_started,
            "cellpose_eval_seconds": eval_seconds,
            "maximum_observed_mps_driver_bytes": maximum_observed_mps_driver_bytes,
        }
        temporary_progress = progress_path.with_suffix(".tmp")
        temporary_progress.write_text(json.dumps(progress, indent=2) + "\n")
        temporary_progress.replace(progress_path)

    return pd.DataFrame(all_instance_rows, columns=INSTANCE_COLUMNS)


def _worker_entry(
    worker_id,
    gpu_id,
    worker_csv,
    partial_csv,
    out_dir,
    model_name,
    batch_size,
    diameter,
    flow_threshold,
    cellprob_threshold,
    min_size,
    use_core_region,
    crop_cells,
    cell_out_dir,
    crop_size,
    cell_background,
    overwrite_cell_crops,
    cell_image_mode,
):
    """
    多 GPU worker 入口。
    必须在导入 cellpose/torch 前设置 CUDA_VISIBLE_DEVICES。
    这样每个子进程只看到一张指定的物理 GPU。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 可选：减少 CPU 线程互相抢占。如果你的机器 CPU 核很多，也可以删掉这几行。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    worker_df = pd.read_csv(worker_csv)
    records = worker_df.to_dict("records")

    df = run_cellpose_on_patch_records(
        records=records,
        out_dir=out_dir,
        model_name=model_name,
        gpu=True,
        batch_size=batch_size,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        use_core_region=use_core_region,
        worker_id=worker_id,
        desc=f"GPU {gpu_id}",
        crop_cells=crop_cells,
        cell_out_dir=cell_out_dir,
        crop_size=crop_size,
        cell_background=cell_background,
        overwrite_cell_crops=overwrite_cell_crops,
        cell_image_mode=cell_image_mode,
    )

    partial_csv = Path(partial_csv)
    partial_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(partial_csv, index=False)

    print(f"[OK][worker {worker_id}] GPU {gpu_id} 完成，实例数: {len(df)}")
    print(f"[OK][worker {worker_id}] partial csv: {partial_csv}")


def run_cellpose_on_patch_table_single_gpu(
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
    crop_cells=True,
    cell_out_dir=None,
    crop_size=64,
    cell_background="white",
    overwrite_cell_crops=False,
    cell_image_mode="raw",
    device=None,
    attention="auto",
):
    """
    原单进程版本：适合 1 张 GPU 或 CPU。
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

    records = patch_df.to_dict("records")
    instance_df = run_cellpose_on_patch_records(
        records=records,
        out_dir=out_dir,
        model_name=model_name,
        gpu=gpu,
        device=device,
        attention=attention,
        batch_size=batch_size,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        use_core_region=use_core_region,
        worker_id=0,
        crop_cells=crop_cells,
        cell_out_dir=cell_out_dir,
        crop_size=crop_size,
        cell_background=cell_background,
        overwrite_cell_crops=overwrite_cell_crops,
        cell_image_mode=cell_image_mode,
    )

    instance_df.insert(
        0,
        "global_instance_id",
        np.arange(1, len(instance_df) + 1, dtype=np.int64),
    )

    out_csv = table_dir / f"{patch_csv.stem}_cellpose_mask_instances.csv"
    instance_df.to_csv(out_csv, index=False)

    print(f"[OK] Cellpose mask 已保存到: {mask_dir}")
    print(f"[OK] mask 实例坐标表已保存到: {out_csv}")
    if crop_cells:
        print(f"[OK] 单细胞 image/mask crop 已保存到: {cell_out_dir or (out_dir / 'single_cells')}")
    print(f"[OK] 细胞实例数量: {len(instance_df)}")

    return instance_df


def run_cellpose_on_patch_table_multi_gpu(
    patch_csv,
    out_dir,
    model_name="cpsam_v2",
    num_gpus=8,
    gpu_ids=None,
    batch_size=8,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    use_core_region=True,
    tmp_dir=None,
    crop_cells=True,
    cell_out_dir=None,
    crop_size=64,
    cell_background="white",
    overwrite_cell_crops=False,
    cell_image_mode="raw",
):
    """
    多 GPU 版本：
    1. 主进程读取 patch_csv；
    2. 按行把 patch 平均切分给 num_gpus 个 worker；
    3. 每个 worker 绑定一张 GPU，并各自加载一个 CellposeModel；
    4. 每个 patch 得到 mask 后，立即输出该 patch 内每个细胞的 image crop 和 mask crop；
    5. 每个 worker 输出 partial csv；
    6. 主进程合并 partial csv，并重新生成唯一 global_instance_id。
    """
    patch_csv = Path(patch_csv)
    out_dir = Path(out_dir)

    mask_dir = out_dir / "patch_masks"
    table_dir = out_dir / "mask_tables"
    mask_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if cell_out_dir is None:
        cell_out_dir = out_dir / "single_cells"
    else:
        cell_out_dir = Path(cell_out_dir)

    if crop_cells:
        cell_out_dir.mkdir(parents=True, exist_ok=True)

    if tmp_dir is None:
        tmp_dir = out_dir / "multi_gpu_tmp"
    else:
        tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    patch_df = pd.read_csv(patch_csv)

    required_cols = ["path", "sample", "x", "y", "patch_size", "stride"]
    for col in required_cols:
        if col not in patch_df.columns:
            raise ValueError(f"patch csv 缺少必要字段: {col}")

    gpu_ids = parse_gpu_ids(gpu_ids, num_gpus)

    print(f"[INFO] patch 总数量: {len(patch_df)}")
    print(f"[INFO] 使用 GPU 数量: {num_gpus}")
    print(f"[INFO] GPU IDs: {gpu_ids}")
    print(f"[INFO] 每张 GPU 的 batch_size: {batch_size}")
    print(f"[INFO] 理论总并行 batch_size: {batch_size * num_gpus}")
    print(f"[INFO] crop_cells: {crop_cells}")
    if crop_cells:
        print(f"[INFO] 单细胞输出目录: {cell_out_dir}")
        print(f"[INFO] 单细胞 crop_size: {crop_size}")
        print(f"[INFO] 单细胞图像模式: {cell_image_mode}")

    ctx = mp.get_context("spawn")
    processes = []
    partial_csvs = []

    for worker_id, gpu_id in enumerate(gpu_ids):
        # 轮转切分，比连续切分更容易平衡不同 sample/路径带来的 I/O 差异
        worker_df = patch_df.iloc[worker_id::num_gpus].reset_index(drop=True)

        worker_csv = tmp_dir / f"worker_{worker_id}_patch_index.csv"
        partial_csv = tmp_dir / f"worker_{worker_id}_instances.csv"
        worker_df.to_csv(worker_csv, index=False)
        partial_csvs.append(partial_csv)

        p = ctx.Process(
            target=_worker_entry,
            args=(
                worker_id,
                gpu_id,
                str(worker_csv),
                str(partial_csv),
                str(out_dir),
                model_name,
                batch_size,
                diameter,
                flow_threshold,
                cellprob_threshold,
                min_size,
                use_core_region,
                crop_cells,
                str(cell_out_dir),
                crop_size,
                cell_background,
                overwrite_cell_crops,
                cell_image_mode,
            ),
        )
        p.start()
        processes.append(p)

    failed = []
    for worker_id, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            failed.append((worker_id, p.exitcode))

    if failed:
        raise RuntimeError(f"有 worker 失败: {failed}")

    dfs = []
    for partial_csv in partial_csvs:
        if not partial_csv.exists() or partial_csv.stat().st_size == 0:
            continue

        df = pd.read_csv(partial_csv)
        if len(df) > 0:
            dfs.append(df)

    if len(dfs) == 0:
        instance_df = pd.DataFrame(columns=["global_instance_id"] + INSTANCE_COLUMNS)
    else:
        instance_df = pd.concat(dfs, ignore_index=True)

        # 为了输出稳定，按 sample、patch_id、local_label 排序后重新编号
        sort_cols = [c for c in ["sample", "patch_id", "local_label"] if c in instance_df.columns]
        if sort_cols:
            instance_df = instance_df.sort_values(sort_cols).reset_index(drop=True)

        instance_df.insert(
            0,
            "global_instance_id",
            np.arange(1, len(instance_df) + 1, dtype=np.int64),
        )

    out_csv = table_dir / f"{patch_csv.stem}_cellpose_mask_instances.csv"
    instance_df.to_csv(out_csv, index=False)

    print(f"[OK] Cellpose mask 已保存到: {mask_dir}")
    print(f"[OK] mask 实例坐标表已保存到: {out_csv}")
    if crop_cells:
        print(f"[OK] 单细胞 image/mask crop 已保存到: {cell_out_dir}")
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
        "--num_gpus",
        type=int,
        default=1,
        help="同时使用的 GPU 数量。8 张卡时设置为 8。",
    )

    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="指定物理 GPU 编号，例如 0,1,2,3,4,5,6,7。默认使用 0 到 num_gpus-1。",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="每张 GPU 上的 Cellpose batch size。总并行 batch 约等于 batch_size * num_gpus。",
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

    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="多 GPU 临时文件目录，默认 out_dir/multi_gpu_tmp",
    )

    parser.add_argument(
        "--no_crop_cells",
        action="store_true",
        help="如果设置，则只输出 patch mask 和实例表，不立即输出单细胞 image/mask crop",
    )

    parser.add_argument(
        "--cell_out_dir",
        type=str,
        default=None,
        help="单细胞 image/mask crop 输出目录，默认 out_dir/single_cells",
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=64,
        help="单细胞 crop 大小，默认 64",
    )

    parser.add_argument(
        "--cell_background",
        type=str,
        default="white",
        choices=["white", "black"],
        help="cell_image_mode=segmented 时非当前细胞区域的背景颜色，默认 white",
    )

    parser.add_argument(
        "--cell_image_mode",
        type=str,
        default="raw",
        choices=["raw", "segmented"],
        help=(
            "raw 保存以细胞中心裁剪的原始图像；"
            "segmented 会把非当前细胞区域设置为背景。默认 raw"
        ),
    )

    parser.add_argument(
        "--overwrite_cell_crops",
        action="store_true",
        help="如果单细胞 image/mask crop 已存在，是否覆盖",
    )

    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None,
                        help="Explicit device; auto selects Apple GPU on Mac")
    parser.add_argument("--attention", choices=["auto", "original", "sdpa"], default="auto",
                        help="auto uses optimized SAM attention on Apple GPU")
    args = parser.parse_args()
    if args.num_gpus > 1:
        import torch
        if not torch.cuda.is_available() or args.device in {"mps", "cpu"}:
            parser.error("Multiple GPUs require CUDA. On this Mac use --num_gpus 1 --device mps.")

    crop_cells = not args.no_crop_cells

    if args.num_gpus <= 1:
        run_cellpose_on_patch_table_single_gpu(
            patch_csv=args.patch_csv,
            out_dir=args.out_dir,
            model_name=args.model_name,
            gpu=args.gpu,
            device=args.device,
            attention=args.attention,
            batch_size=args.batch_size,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            use_core_region=not args.no_core_region,
            crop_cells=crop_cells,
            cell_out_dir=args.cell_out_dir,
            crop_size=args.crop_size,
            cell_background=args.cell_background,
            overwrite_cell_crops=args.overwrite_cell_crops,
            cell_image_mode=args.cell_image_mode,
        )
    else:
        if not args.gpu:
            print("[WARNING] --num_gpus > 1 时会自动启用 GPU。建议命令中显式加 --gpu。")

        run_cellpose_on_patch_table_multi_gpu(
            patch_csv=args.patch_csv,
            out_dir=args.out_dir,
            model_name=args.model_name,
            num_gpus=args.num_gpus,
            gpu_ids=args.gpu_ids,
            batch_size=args.batch_size,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            use_core_region=not args.no_core_region,
            tmp_dir=args.tmp_dir,
            crop_cells=crop_cells,
            cell_out_dir=args.cell_out_dir,
            crop_size=args.crop_size,
            cell_background=args.cell_background,
            overwrite_cell_crops=args.overwrite_cell_crops,
            cell_image_mode=args.cell_image_mode,
        )


if __name__ == "__main__":
    main()

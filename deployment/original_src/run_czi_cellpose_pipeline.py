import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage.filters import threshold_otsu
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import openslide


def make_positions(length, patch_size, stride, include_edges=True):
    """
    生成 patch 左上角坐标。
    include_edges=True 时，会额外补最后一个 patch，使边缘区域也被覆盖。
    """
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if include_edges:
        last = length - patch_size
        if positions[-1] != last:
            positions.append(last)
    return positions


def detect_tissue_regions(
    slide,
    thumbnail_size=(1024,1024),
    kernel_size=5,
    tissue_mode = "light",
    save_qc_dir=None,
    sample_name=None,
):
    thumbnail = slide.get_thumbnail(thumbnail_size).convert("RGB")
    thumbnail_rgb = np.array(thumbnail)
    thumbnail_gray = np.array(thumbnail.convert("L"))
    otsu_thresh = threshold_otsu(thumbnail_gray)
    print(f"[INFO] Otsu 阈值（灰度）: {otsu_thresh}")
    if tissue_mode == "dark":
        # HE 病理图像常用：组织比白色背景暗
        binary_mask = (thumbnail_gray < otsu_thresh).astype(np.uint8) * 255
    elif tissue_mode == "light":
        # 如果你的组织区域比背景亮，可以用这个
        binary_mask = (thumbnail_gray > otsu_thresh).astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 去除小噪声
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # 填补小空洞
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    if save_qc_dir is not None and sample_name is not None:
        save_qc_dir = Path(save_qc_dir)
        save_qc_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(thumbnail_rgb).save(save_qc_dir / f"{sample_name}_thumbnail.png")
        Image.fromarray(binary_mask).save(save_qc_dir / f"{sample_name}_tissue_mask.png")
    return binary_mask, thumbnail_rgb


def extract_patches_from_roi(
    slide,
    binary_mask,
    patch_size=512,
    stride=512,
    min_coverage=0.25,
    out_dir=None,
    sample_name=None,
    source_tif=None,
    save_format="png",
    files_per_folder=1000,
    include_edges=True,
    num_workers=8,
):
    """
    根据低分辨率组织 mask，在原始 TIFF 最高分辨率 level 0 上多线程切 patch。

    注意：
    - patch_size 是原始 TIFF level 0 上的像素大小，例如 512。
    - stride 是原始 TIFF level 0 上的滑窗步长。
    - 输出 CSV 中的 x, y 是 patch 左上角在原始 TIFF 中的位置。
    - 多线程时，每个线程内部单独打开 OpenSlide，避免多个线程共用同一个 slide 句柄。
    """

    if out_dir is None:
        raise ValueError("out_dir 不能为空")

    if sample_name is None:
        sample_name = "sample"

    if source_tif is None:
        raise ValueError("多线程提取 patch 时 source_tif 不能为空，需要用它在线程中重新打开 OpenSlide")

    out_dir = Path(out_dir)

    wsi_width, wsi_height = slide.dimensions

    thumbnail_height, thumbnail_width = binary_mask.shape

    scale_x = wsi_width / thumbnail_width
    scale_y = wsi_height / thumbnail_height

    print(f"[INFO] WSI size: width={wsi_width}, height={wsi_height}")
    print(f"[INFO] Thumbnail mask size: width={thumbnail_width}, height={thumbnail_height}")
    print(f"[INFO] scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    print(f"[INFO] num_workers={num_workers}")

    xs = make_positions(
        length=wsi_width,
        patch_size=patch_size,
        stride=stride,
        include_edges=include_edges,
    )

    ys = make_positions(
        length=wsi_height,
        patch_size=patch_size,
        stride=stride,
        include_edges=include_edges,
    )

    # 第一步：先根据 tissue mask 筛选需要保存的 patch 坐标
    tasks = []

    for y in ys:
        for x in xs:
            thumb_x1 = int(np.floor(x / scale_x))
            thumb_y1 = int(np.floor(y / scale_y))
            thumb_x2 = int(np.ceil((x + patch_size) / scale_x))
            thumb_y2 = int(np.ceil((y + patch_size) / scale_y))

            thumb_x1 = max(0, min(thumb_x1, thumbnail_width - 1))
            thumb_y1 = max(0, min(thumb_y1, thumbnail_height - 1))
            thumb_x2 = max(thumb_x1 + 1, min(thumb_x2, thumbnail_width))
            thumb_y2 = max(thumb_y1 + 1, min(thumb_y2, thumbnail_height))

            patch_mask = binary_mask[thumb_y1:thumb_y2, thumb_x1:thumb_x2]
            tissue_ratio = np.mean(patch_mask == 0)

            if tissue_ratio < min_coverage:
                continue

            patch_id = len(tasks)

            tasks.append({
                "patch_id": patch_id,
                "x": int(x),
                "y": int(y),
                "tissue_ratio": float(tissue_ratio),
                "thumb_x1": int(thumb_x1),
                "thumb_y1": int(thumb_y1),
                "thumb_x2": int(thumb_x2),
                "thumb_y2": int(thumb_y2),
            })

    print(f"[INFO] 待提取 patch 数量: {len(tasks)}")

    if len(tasks) == 0:
        return pd.DataFrame()

    # 每个线程维护自己的 OpenSlide 对象
    thread_local = threading.local()

    def get_thread_slide():
        if not hasattr(thread_local, "slide"):
            thread_local.slide = openslide.OpenSlide(str(source_tif))
        return thread_local.slide

    def save_one_patch(task):
        """
        单个 patch 的读取和保存函数。
        在线程池中并行执行。
        """
        patch_id = task["patch_id"]
        x = task["x"]
        y = task["y"]

        local_slide = get_thread_slide()
        level_tmp = 1 if slide.level_count > 1 else 0
        patch = local_slide.read_region(
            location=(x, y),
            level=0,
            size=(patch_size, patch_size),
        ).convert("RGB")

        folder_number = patch_id // files_per_folder + 1
        folder_name = out_dir / sample_name / f"folder_{folder_number}"
        folder_name.mkdir(parents=True, exist_ok=True)

        if save_format.lower() in ["tif", "tiff"]:
            patch_path = folder_name / f"patch{patch_id}_x{x}_y{y}.tif"
            patch.save(patch_path, compression="tiff_lzw")
        elif save_format.lower() == "png":
            patch_path = folder_name / f"patch{patch_id}_x{x}_y{y}.png"
            patch.save(patch_path)
        else:
            raise ValueError("save_format 只能是 'png'、'tif' 或 'tiff'")

        row = {
            "path": str(patch_path),
            "num_patch": int(patch_id),
            "sample": sample_name,
            "source_tif": str(source_tif),
            "x": int(x),
            "y": int(y),
            "patch_size": int(patch_size),
            "stride": int(stride),
            "level": 0,
            "tissue_ratio": float(task["tissue_ratio"]),
            "thumb_x1": int(task["thumb_x1"]),
            "thumb_y1": int(task["thumb_y1"]),
            "thumb_x2": int(task["thumb_x2"]),
            "thumb_y2": int(task["thumb_y2"]),
        }

        return row
    rows = []
    # 第二步：多线程提取 patch
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_one_patch, task) for task in tasks]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Extracting patches from {sample_name}",
        ):
            row = future.result()
            rows.append(row)

    image_dir = pd.DataFrame(rows)

    # 按 patch 编号排序，保证 CSV 顺序稳定
    image_dir = image_dir.sort_values("num_patch").reset_index(drop=True)

    return image_dir


def process_one_tif(
    tif_path,
    out_dir,
    patch_size=512,
    stride=512,
    thumbnail_size=(1024,1024),
    min_coverage=0.25,
    save_format="png",
    files_per_folder=1000,
    include_edges=True,
    tissue_mode="dark",
    save_qc=True,
    num_workers=8
):
    """
    处理单个 TIFF 文件。
    """

    tif_path = Path(tif_path)
    sample_name = tif_path.stem

    print("\n" + "=" * 80)
    print(f"[INFO] Processing: {tif_path}")
    print("=" * 80)

    slide = openslide.OpenSlide(str(tif_path))
    w, h = slide.level_dimensions[0]
    qc_dir = Path(out_dir) / "qc" if save_qc else None
    thumbnail_size = (int(w/100), int(h/100))
    binary_mask, thumbnail_rgb = detect_tissue_regions(
        slide=slide,
        thumbnail_size=thumbnail_size,
        kernel_size=5,
        tissue_mode=tissue_mode,
        save_qc_dir=qc_dir,
        sample_name=sample_name,
    )

    image_dir = extract_patches_from_roi(
        slide=slide,
        binary_mask=binary_mask,
        patch_size=patch_size,
        stride=stride,
        min_coverage=min_coverage,
        out_dir=out_dir,
        sample_name=sample_name,
        source_tif=str(tif_path),
        save_format=save_format,
        files_per_folder=files_per_folder,
        num_workers=num_workers,
        include_edges=include_edges,
    )

    table_dir = Path(out_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    csv_path = table_dir / f"{sample_name}_patch_index.csv"
    image_dir.to_csv(csv_path, index=False)

    print(f"[OK] {sample_name}")
    print(f"[OK] patch 数量: {len(image_dir)}")
    print(f"[OK] 坐标表: {csv_path}")

    slide.close()

    return image_dir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tif_dir",
        type=str,
        required=True,
        help="输入 TIFF 文件夹",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="patch 输出文件夹",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="patch 大小，原始 TIFF level 0 坐标，默认 512",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="滑窗步长，默认 512；如果需要 overlap，可设为 448、384 等",
    )

    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.25,
        help="patch 中组织区域最小占比，默认 0.25",
    )

    parser.add_argument(
        "--save_format",
        type=str,
        default="png",
        choices=["png", "tif", "tiff"],
        help="保存 patch 的格式，默认 png",
    )

    parser.add_argument(
        "--files_per_folder",
        type=int,
        default=1000,
        help="每个子文件夹保存多少个 patch，默认 1000",
    )

    parser.add_argument(
        "--no_edges",
        action="store_true",
        help="如果设置，则不额外补边缘 patch",
    )

    parser.add_argument(
        "--tissue_mode",
        type=str,
        default="dark",
        choices=["dark", "light"],
        help="组织相对背景的灰度模式。HE 一般用 dark",
    )

    parser.add_argument(
        "--no_qc",
        action="store_true",
        help="如果设置，则不保存 thumbnail 和 tissue mask 质控图",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="多线程提取 patch 的线程数，默认 4",
    )

    args = parser.parse_args()

    tif_dir = Path(args.tif_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(
        list(tif_dir.glob("*.tif")) +
        list(tif_dir.glob("*.tiff")) +
        list(tif_dir.glob("*.svs"))
    )

    if len(tif_files) == 0:
        raise FileNotFoundError(f"在 {tif_dir} 中没有找到 .tif 或 .tiff 文件")

    all_tables = []

    for tif_path in tif_files:
        try:
            image_dir = process_one_tif(
                tif_path=tif_path,
                out_dir=out_dir,
                patch_size=args.patch_size,
                stride=args.stride,
                min_coverage=args.min_coverage,
                save_format=args.save_format,
                files_per_folder=args.files_per_folder,
                tissue_mode=args.tissue_mode,
                include_edges=not args.no_edges,
                num_workers=args.num_workers,
                save_qc=not args.no_qc,
            )

            if len(image_dir) > 0:
                all_tables.append(image_dir)

        except Exception as e:
            print(f"[ERROR] 处理失败: {tif_path}")
            print(f"[ERROR] {repr(e)}")

    # if len(all_tables) > 0:
    #     all_df = pd.concat(all_tables, ignore_index=True)
    #     all_csv = out_dir / "tables" / "all_patch_index.csv"
    #     all_df.to_csv(all_csv, index=False)
    #     print(f"\n[OK] 总 patch 数量: {len(all_df)}")
    #     print(f"[OK] 总坐标表: {all_csv}")
    # else:
    #     print("[WARNING] 没有成功提取任何 patch")


if __name__ == "__main__":
    main()

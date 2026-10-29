
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize selected cells on the original TIF/WSI with colored bounding boxes.

Features
--------
1. Read instances_with_cell64_paths.csv (or similar instance CSV).
2. Filter by sample and by selected cell types.
3. Draw global bounding boxes on a thumbnail of the original TIF/WSI.
4. Different cell types are rendered in different colors.
5. Save one overlay image per sample.

Recommended usage
-----------------
python visualize_cells_on_tif.py \
    --csv_path /path/to/instances_with_cell64_paths.csv \
    --tif_dir /path/to/tif \
    --celltype_col cell_name \
    --sample_col source_tif \
    --sample_name SR386_40X_HE_T404_01.tif \
    --types tumor,lymphocyte,stroma \
    --output_dir /path/to/vis_outputs

If your CSV stores sample names without ".tif", the script will try to match them automatically.
"""

import argparse
from pathlib import Path
import math
import warnings

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import tifffile

# Pillow safety limit for large images
Image.MAX_IMAGE_PIXELS = None

try:
    import openslide
    HAS_OPENSLIDE = True
except Exception:
    HAS_OPENSLIDE = False

try:
    import tifffile
    HAS_TIFFFILE = True
except Exception:
    HAS_TIFFFILE = False


def normalize_sample_key(x: str) -> str:
    x = str(x)
    x = Path(x).name
    for suf in [".ome.tif", ".ome.tiff", ".tif", ".tiff"]:
        if x.lower().endswith(suf):
            return x[:-len(suf)]
    return Path(x).stem


def find_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_bbox_columns(df):
    xmin = find_existing_column(df, ["bbox_xmin_global", "xmin", "x_min", "bbox_xmin"])
    ymin = find_existing_column(df, ["bbox_ymin_global", "ymin", "y_min", "bbox_ymin"])
    xmax = find_existing_column(df, [
        "bbox_xmax_global_exclusive", "bbox_xmax_global", "xmax", "x_max", "bbox_xmax"
    ])
    ymax = find_existing_column(df, [
        "bbox_ymax_global_exclusive", "bbox_ymax_global", "ymax", "y_max", "bbox_ymax"
    ])
    if not all([xmin, ymin, xmax, ymax]):
        raise ValueError(
            "Cannot infer bbox columns. Please ensure CSV has global bbox columns, e.g. "
            "bbox_xmin_global, bbox_ymin_global, bbox_xmax_global_exclusive, bbox_ymax_global_exclusive."
        )
    return xmin, ymin, xmax, ymax


def parse_types(types_str):
    if not types_str:
        return None
    return [t.strip() for t in types_str.split(",") if t.strip()]


def build_color_map(type_names):
    # fixed palette, extend if needed
    palette = [
        (230, 25, 75),   # red
        (60, 180, 75),   # green
        (0, 130, 200),   # blue
        (245, 130, 48),  # orange
        (145, 30, 180),  # purple
        (70, 240, 240),  # cyan
        (240, 50, 230),  # magenta
        (210, 245, 60),  # lime
        (250, 190, 190), # pink
        (0, 128, 128),   # teal
        (230, 190, 255), # lavender
        (170, 110, 40),  # brown
        (255, 250, 200), # beige
        (128, 0, 0),     # maroon
        (170, 255, 195), # mint
        (128, 128, 0),   # olive
        (255, 215, 180), # apricot
        (0, 0, 128),     # navy
        (128, 128, 128), # gray
        (0, 0, 0),       # black
    ]
    color_map = {}
    for i, t in enumerate(type_names):
        color_map[t] = palette[i % len(palette)]
    return color_map


def open_slide_any(tif_path):
    tif_path = Path(tif_path)
    if HAS_OPENSLIDE:
        try:
            slide = openslide.OpenSlide(str(tif_path))
            level_tmp = 1 if slide.level_count > 1 else 0
            w, h = slide.level_dimensions[0]
            return ("openslide", slide, int(w), int(h))
        except Exception:
            pass

    if HAS_TIFFFILE:
        arr = tifffile.imread(str(tif_path))
        if arr.ndim == 2:
            h, w = arr.shape
        elif arr.ndim == 3:
            # HWC or CHW guess
            if arr.shape[0] <= 4 and arr.shape[-1] > 4:
                arr = np.transpose(arr, (1, 2, 0))
            h, w = arr.shape[:2]
        else:
            raise ValueError(f"Unsupported tif shape: {arr.shape}")
        return ("tifffile", arr, int(w), int(h))

    raise RuntimeError("Neither openslide nor tifffile is available.")


def get_thumbnail_from_source(source_tuple, max_dim=3000):
    source_type, obj, w, h = source_tuple
    scale = min(max_dim / max(w, h), 1.0)
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))

    if source_type == "openslide":
        thumb = obj.get_thumbnail((tw, th)).convert("RGB")
    else:
        arr = obj
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]

        # normalize if not uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            amin, amax = float(arr.min()), float(arr.max())
            if amax > amin:
                arr = (arr - amin) / (amax - amin) * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        if (img.width, img.height) != (tw, th):
            img = img.resize((tw, th))
        thumb = img.convert("RGB")

    return thumb, (w, h), (tw, th)


def resolve_tif_path(sample_value, tif_dir):
    tif_dir = Path(tif_dir)
    sample_value = str(sample_value)
    sample_name = Path(sample_value).name

    direct = tif_dir / sample_name
    if direct.exists():
        return direct

    key = normalize_sample_key(sample_value)
    candidates = []
    for p in tif_dir.glob("*"):
        if p.is_file() and p.suffix.lower() in [".tif", ".tiff"]:
            if normalize_sample_key(p.name) == key:
                candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        raise RuntimeError(f"Multiple tif files matched sample {sample_value}: {candidates}")
    else:
        raise FileNotFoundError(f"Cannot find tif for sample {sample_value} in {tif_dir}")


def add_legend(img, counts_dict, color_map, title="Cell types"):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    padding = 12
    box = 18
    line_h = 24
    title_h = 28

    labels = [f"{k}: {v}" for k, v in counts_dict.items()]
    max_text_w = 0
    for txt in [title] + labels:
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            text_w = bbox[2] - bbox[0]
        except Exception:
            text_w = len(txt) * 8
        max_text_w = max(max_text_w, text_w)

    legend_w = padding * 3 + box + max_text_w
    legend_h = padding * 2 + title_h + line_h * len(labels)

    x0 = max(0, img.width - legend_w - 20)
    y0 = 20
    x1 = x0 + legend_w
    y1 = y0 + legend_h

    # background
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((x0 + padding, y0 + 6), title, fill=(0, 0, 0), font=font)

    yy = y0 + title_h
    for k, v in counts_dict.items():
        color = color_map[k]
        draw.rectangle([x0 + padding, yy + 2, x0 + padding + box, yy + 2 + box], fill=color, outline=(0, 0, 0))
        draw.text((x0 + padding * 2 + box, yy), f"{k}: {v}", fill=(0, 0, 0), font=font)
        yy += line_h

def overlay_mask_on_thumbnail(
    thumb,
    mask_tif_path,
    mask_alpha=255,
    mask_color=(255, 255, 0),   # 黄色
    background_value=0,
    ring_width=3,               # 外环宽度
):
    """
    将 mask 以黄色外环形式叠加到 thumbnail 上。

    thumb: PIL.Image, RGB
    mask_tif_path: mask tif 路径
    thumb_size: (tw, th)
    mask_alpha: 0-255，越大越不透明
    mask_color: 外环颜色，默认黄色
    background_value: 背景值，通常是 0
    ring_width: 外环宽度，单位是 thumbnail 像素
    """

    tw, th = thumb.size

    mask = tifffile.imread(mask_tif_path)

    # 如果是多通道，只取第一个通道
    if mask.ndim == 3:
        mask = mask[..., 0]

    # resize 到 thumbnail 大小，保持 label 不插值
    mask_img = Image.fromarray(mask)
    mask_thumb = mask_img.resize((tw, th), resample=Image.Resampling.NEAREST)
    mask_thumb = np.array(mask_thumb)

    # 所有非背景区域都认为是 mask
    mask_area = mask_thumb != background_value

    # 用 PIL 的 MaxFilter 做膨胀
    mask_uint8 = (mask_area.astype(np.uint8) * 255)
    mask_pil = Image.fromarray(mask_uint8, mode="L")

    # MaxFilter 的 size 必须是奇数
    kernel_size = ring_width * 2 + 1
    dilated = mask_pil.filter(ImageFilter.MaxFilter(kernel_size))
    dilated_area = np.array(dilated) > 0

    # 外环 = 膨胀区域 - 原始 mask 区域
    outer_ring = dilated_area & (~mask_area)

    thumb_rgba = thumb.convert("RGBA")
    overlay_np = np.zeros((th, tw, 4), dtype=np.uint8)

    overlay_np[outer_ring, 0] = mask_color[0]
    overlay_np[outer_ring, 1] = mask_color[1]
    overlay_np[outer_ring, 2] = mask_color[2]
    overlay_np[outer_ring, 3] = mask_alpha

    overlay = Image.fromarray(overlay_np, mode="RGBA")
    thumb_rgba = Image.alpha_composite(thumb_rgba, overlay)

    return thumb_rgba.convert("RGB")


def draw_boxes_on_thumbnail(
    tif_path,
    df_sample,
    output_path,
    celltype_col,
    bbox_cols,
    max_dim=3000,
    line_width=2,
    annotate_count=True,
    amplify_box = None,
    mask_tif_path = None
):
    xmin_col, ymin_col, xmax_col, ymax_col = bbox_cols

    source = open_slide_any(tif_path)
    thumb, (w, h), (tw, th) = get_thumbnail_from_source(source, max_dim=max_dim)

    sx = tw / w
    sy = th / h

    draw = ImageDraw.Draw(thumb)
    font = ImageFont.load_default()

    type_names = sorted(df_sample[celltype_col].astype(str).unique())
    color_map = build_color_map(type_names)

    # draw boxes
    for _, row in df_sample.iterrows():
        cell_type = str(row[celltype_col])
        color = color_map[cell_type]

        x0 = float(row[xmin_col]) * sx
        y0 = float(row[ymin_col]) * sy
        x1 = float(row[xmax_col]) * sx
        y1 = float(row[ymax_col]) * sy

        # Keep visible at thumbnail scale
        if x1 <= x0:
            x1 = x0 + 10
        if y1 <= y0:
            y1 = y0 + 10
        if amplify_box is not None:
            amplify = amplify_box*(x1 - x0)
            while x1 - x0 < amplify:
                x1 = x1 + 10
                y1 = y1 + 10 

        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

    counts = df_sample[celltype_col].astype(str).value_counts().to_dict()

    if annotate_count:
        add_legend(thumb, counts, color_map, title="Cell types")

    if mask_tif_path is not None:
        thumb = overlay_mask_on_thumbnail(
            thumb=thumb,
            mask_tif_path=mask_tif_path,
            mask_color=(255, 255, 0),
            mask_alpha=255,
            background_value=0,
            ring_width=3,
        )

    thumb.save(output_path)

    # Close openslide handle if used
    if source[0] == "openslide":
        source[1].close()

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to instances_with_cell64_paths.csv")
    parser.add_argument("--tif_dir", required=True, help="Directory containing original tif/tiff files")
    parser.add_argument("--output_dir", required=True, help="Directory to save visualization PNGs")

    parser.add_argument("--celltype_col", default="cell_name", help="Column storing cell type/name")
    parser.add_argument("--sample_col", default="source_tif", help="Column storing source tif/sample name")
    parser.add_argument("--sample_name", default=None, help="Only visualize one sample (optional)")
    parser.add_argument("--types", default=None, help="Comma-separated selected cell types, e.g. tumor,lymphocyte")
    parser.add_argument("--max_dim", type=int, default=3000, help="Max width/height of output thumbnail image")
    parser.add_argument("--line_width", type=int, default=2, help="Bounding box line width")
    parser.add_argument("--max_cells", type=int, default=None, help="Optional cap on number of cells to draw")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed if max_cells is used")
    parser.add_argument("--amplify_box", default=None, help="Zoom in on the cell box")
    parser.add_argument("--mask_tif_path", default=None, help="Zoom in on the cell box")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    tif_dir = Path(args.tif_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.celltype_col not in df.columns:
        raise ValueError(
            f"Column '{args.celltype_col}' not found in CSV. Available columns: {list(df.columns)}"
        )
    if args.sample_col not in df.columns:
        raise ValueError(
            f"Column '{args.sample_col}' not found in CSV. Available columns: {list(df.columns)}"
        )

    bbox_cols = infer_bbox_columns(df)

    # clean NA
    use_cols = [args.celltype_col, args.sample_col, *bbox_cols]
    df = df.dropna(subset=use_cols).copy()

    # filter sample
    if args.sample_name:
        target_key = normalize_sample_key(args.sample_name)
        df = df[df[args.sample_col].astype(str).map(normalize_sample_key) == target_key].copy()

    # filter cell types
    selected_types = parse_types(args.types)
    if selected_types is not None:
        selected_types_set = set(map(str, selected_types))
        df = df[df[args.celltype_col].astype(str).isin(selected_types_set)].copy()

    if len(df) == 0:
        raise ValueError("No rows remain after filtering. Please check sample_name / types / column names.")

    # optional downsampling of rows
    if args.max_cells is not None and len(df) > args.max_cells:
        df = df.sample(n=args.max_cells, random_state=args.random_seed).copy()

    # process each sample
    sample_values = list(df[args.sample_col].astype(str).unique())

    print(f"Total selected rows: {len(df)}")
    print(f"Samples to render: {len(sample_values)}")

    for sample_value in sample_values:
        df_sample = df[df[args.sample_col].astype(str) == sample_value].copy()
        tif_path = resolve_tif_path(sample_value, tif_dir)

        sample_key = normalize_sample_key(sample_value)
        output_path = output_dir / f"{sample_key}_boxes_overlay.png"

        print(f"[Rendering] sample={sample_value}  tif={tif_path}")
        draw_boxes_on_thumbnail(
            tif_path=tif_path,
            df_sample=df_sample,
            output_path=output_path,
            celltype_col=args.celltype_col,
            bbox_cols=bbox_cols,
            max_dim=args.max_dim,
            line_width=args.line_width,
            annotate_count=True,
            amplify_box = args.amplify_box,
            mask_tif_path = args.mask_tif_path
        )
        print(f"[Saved] {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()




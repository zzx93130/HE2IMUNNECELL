"""Shared paths and tar-member access for the TNBC_ST (Zenodo 14204217) validation.

Images are read straight out of the uncompressed tars by seeking to each member's data offset
(index cached in outputs/tnbc_st_20261101/tar_index.json); nothing is extracted wholesale.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "TNBC_ST"
EXTRAS = DATA / "extras"
HE_TAR = DATA / "HE" / "Images.tar"
IHC_TAR = DATA / "IHC" / "Images IHC.tar"
OUT = ROOT / "outputs" / "tnbc_st_20261101"
INDEX = OUT / "tar_index.json"
MIN_FREE_GB = 8.0
ARRAY_RE = re.compile(r"TNBC(\d+)_CN(\d+)_([A-Z]\d)")


def tar_index():
    if not INDEX.exists():
        out = {}
        for key, path in [("HE", HE_TAR), ("IHC", IHC_TAR)]:
            with tarfile.open(path, "r:") as tar:
                out[key] = [(m.name, m.offset_data, m.size) for m in tar.getmembers() if m.isfile()]
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        INDEX.write_text(json.dumps(out))
    raw = json.loads(INDEX.read_text())
    return {key: {name: (offset, size) for name, offset, size in members} for key, members in raw.items()}


_INDEX = None


def member_bytes(key, name):
    global _INDEX
    _INDEX = _INDEX or tar_index()
    offset, size = _INDEX[key][name]
    with open(HE_TAR if key == "HE" else IHC_TAR, "rb") as handle:
        handle.seek(offset)
        return handle.read(size)


def copy_member(key, name, target):
    """Stream one tar member to a file (for NDPIs, which OpenSlide must open from disk)."""
    global _INDEX
    _INDEX = _INDEX or tar_index()
    offset, size = _INDEX[key][name]
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    with open(HE_TAR if key == "HE" else IHC_TAR, "rb") as src, open(partial, "wb") as dst:
        src.seek(offset)
        left = size
        while left:
            chunk = src.read(min(left, 1 << 24))
            dst.write(chunk)
            left -= len(chunk)
    partial.replace(target)
    return target


def read_image(name, reduce=1):
    """Decode an H&E jpg/png member to RGB uint8. reduce in {1,2,4,8} uses libjpeg DCT scaling."""
    import cv2
    data = np.frombuffer(member_bytes("HE", name), np.uint8)
    flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
             8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"could not decode {name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def arrays():
    """All H&E arrays: name -> dict(sample, cn, subarray, hd, large, annotation or None)."""
    global _INDEX
    _INDEX = _INDEX or tar_index()
    he = _INDEX["HE"]
    out = {}
    for name in he:
        if name.startswith("Images/imagesHD/") and not os.path.basename(name).startswith("."):
            stem = Path(name).stem
            m = ARRAY_RE.fullmatch(stem)
            if not m:
                continue
            ann = f"Images/imageAnnotations/{stem}.png"
            out[stem] = {"array": stem, "sample": int(m.group(1)), "cn": int(m.group(2)), "subarray": m.group(3),
                         "hd": name, "large": f"Images/imagesLarge/{stem}.jpg", "annotation": ann if ann in he else None}
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["sample"], kv[0])))


def ihc_slides():
    """sample -> NDPI member name and declared stains."""
    global _INDEX
    _INDEX = _INDEX or tar_index()
    out = {}
    for name in _INDEX["IHC"]:
        base = os.path.basename(name)
        if base.startswith(".") or not base.endswith(".ndpi"):
            continue
        m = re.match(r"ST_TNBC-(\d+)", base)
        out[int(m.group(1))] = {"member": name, "label_CD2": "CD2" in base.replace("CD20", ""), "size": _INDEX["IHC"][name][1]}
    return dict(sorted(out.items()))


def free_gb(path=OUT):
    import shutil
    return shutil.disk_usage(path).free / 1e9


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=default))
    tmp.replace(path)

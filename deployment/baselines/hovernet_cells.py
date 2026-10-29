"""HoVer-Net (PanNuke) nuclear instance segmentation + classification on the TLS validation images.

Published baseline, independent of HE2CELL: TIAToolbox 1.6.0
`NucleusInstanceSegmentor(pretrained_model="hovernet_fast-pannuke")` (HoVer-Net "fast" mode, Graham et al.
2019, trained on PanNuke, Gamper et al. 2020; weights from the TIAToolbox model zoo, provenance written to
cells/provenance.json). PanNuke classes: 1 neoplastic, 2 inflammatory, 3 connective, 4 dead,
5 non-neoplastic epithelial (0 = no label). "Inflammatory" is the lymphocyte proxy.

Inputs, pixel-identical to HE2CELL's:
  esca    every window of the adopted analysis (outputs/esca_tls_20261019/analysis_he2cell_ft_tau05/windows.csv),
          regenerated from the SVS at native level 0 (~0.25 µm/px), no stain normalisation (as the adopted run).
          The image is the 3328-px square covered by esca_windows' 4x4 patches of 1024 px at stride 768
          (starting 128 px before the window origin); HoVer-Net tiles it itself (256 -> 164 px, merged 1024-px
          tiles with a 128-px margin) instead of being run 16 times on overlapping patches (1.8x less compute,
          no patch-border cuts), and a nucleus is kept only when its centroid lies in the 3072-px window core,
          i.e. the union of the 16 patch cores HE2CELL keeps.
  visium  the 8 USZ sections, patches from visium_cells.patch_grid (bicubic upsampling to 0.25 µm/px,
          1024 px / stride 768, tissue under in-tissue spots), one HoVer-Net call per patch, nuclei kept in the
          768-px patch core (as HE2CELL).
Outputs (cells/): esca/<barcode>/<window_id>.csv.gz with centroid_x_global / centroid_y_global in slide
level-0 pixels (the HE2CELL ESCA frame); visium/<sample>.csv.gz with centroid_*_global in the 0.25 µm
frame and x_native / y_native in TIFF pixels (the HE2CELL Visium frame). Columns type (0-5), type_name,
type_prob (HoVer-Net's per-instance type probability), area_um2. Resumable and shareable: work units (ESCA
slides, Visium chunks of 48 patches) are claimed with mkdir locks in cells/_locks/, finished units are skipped,
so several CPU workers and one MPS worker can run at once. Timing per unit goes to cells/timing.csv.

Run with .runtime/tiatoolbox/venv/bin/python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deployment" / "visium_tls"))
sys.path.insert(0, str(ROOT / "deployment" / "esca_tls"))
sys.path.insert(0, str(ROOT / "script" / "src"))
import visium_cells as V  # noqa: E402
from esca_windows import CORE_MARGIN, GRID, PATCH, STRIDE  # noqa: E402
from esca_common import OUT as ESCA_OUT  # noqa: E402
from slide_io import open_slide  # noqa: E402

assert (PATCH, STRIDE, CORE_MARGIN) == (V.PATCH, V.STRIDE, V.MARGIN)
OUT = ROOT / "outputs" / "baselines_hovernet_20261026"
CELLS = OUT / "cells"
WINDOWS = ESCA_OUT / "analysis_he2cell_ft_tau05" / "windows.csv"
MODEL = "hovernet_fast-pannuke"
TYPES = {0: "no_label", 1: "neoplastic", 2: "inflammatory", 3: "connective", 4: "dead", 5: "epithelial"}
TARGET_MPP = 0.25
VISIUM_CHUNK = 48
SINGLE_TILE = 21 * 164  # 3444 px >= the 3328-px ESCA window; a multiple of HoVer-Net's 164-px output
COLUMNS = ["patch_id", "centroid_x_global", "centroid_y_global", "type", "type_name", "type_prob", "area_um2"]


def save_atomic(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.gz")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def log_timing(**row):
    path = CELLS / "timing.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"time": time.strftime("%F %T"), **row}]).to_csv(path, mode="a", header=not path.exists(), index=False)


class Segmentor:
    """TIAToolbox NucleusInstanceSegmentor over in-memory square RGB images (tile mode, one call per batch)."""

    def __init__(self, device, batch_size, postproc_workers, tile_shape=SINGLE_TILE):
        import copy
        from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
        self.device = device
        self.engine = NucleusInstanceSegmentor(pretrained_model=MODEL, batch_size=batch_size,
                                               num_loader_workers=0, num_postproc_workers=postproc_workers,
                                               auto_generate_mask=False, verbose=False)
        # The model, its 256 -> 164 px sliding window and its post-processing are the pretrained ioconfig's.
        # Only the post-processing tile is enlarged so that a whole image is one tile: the default 1024-px tile
        # (984 after rounding to the 164-px output grid) splits every 1024-px patch and 3328-px window into
        # tiles plus three sets of seam strips, each re-running inference (~2x compute). tile_shape=None keeps it.
        self.ioconfig = None
        if tile_shape:
            self.ioconfig = copy.deepcopy(self.engine.ioconfig)
            self.ioconfig.tile_shape = [tile_shape, tile_shape]
            self.ioconfig._kwargs["tile_shape"] = [tile_shape, tile_shape]  # to_baseline() rebuilds from _kwargs
        provenance()

    def __call__(self, images, core):
        """images: RGB uint8 arrays at 0.25 µm/px; core: (low, high) image-pixel bounds on both axes.
        Returns one frame per image with the nuclei whose centroid lies in the core (centroids in image px)."""
        import joblib
        low, high = core
        work = Path(tempfile.mkdtemp(prefix="hovernet_", dir=OUT / "_tmp"))
        try:
            paths = []
            for i, image in enumerate(images):
                path = work / f"{i:05d}.png"
                cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                paths.append(path)
            outputs = self.engine.predict([str(p) for p in paths], mode="tile", device=self.device, ioconfig=self.ioconfig,
                                          save_dir=work / "out", crash_on_exception=True)
            by_input = {Path(i).name: Path(o) for i, o in outputs}
            frames = []
            for path in paths:
                instances = joblib.load(str(by_input[path.name]) + ".dat")
                rows = []
                for info in instances.values():
                    cx, cy = (float(v) for v in info["centroid"])
                    if not (low <= cx < high and low <= cy < high):
                        continue
                    contour = np.asarray(info["contour"], np.float32)
                    kind = int(info["type"]) if info["type"] is not None else 0
                    rows.append((cx, cy, kind, float(info["prob"]) if info["prob"] is not None else np.nan,
                                 float(cv2.contourArea(contour)) * TARGET_MPP ** 2 if len(contour) >= 3 else np.nan))
                frames.append(pd.DataFrame(rows, columns=["cx", "cy", "type", "type_prob", "area_um2"]))
            return frames
        finally:
            shutil.rmtree(work, ignore_errors=True)


def claim(name):
    """Atomic work claim (mkdir) shared by all workers (CPU and MPS); a lock whose pid is dead is taken over."""
    lock = CELLS / "_locks" / name
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        try:
            pid = int((lock / "pid").read_text())
            os.kill(pid, 0)
            return False
        except (FileNotFoundError, ValueError):
            if time.time() - lock.stat().st_mtime < 60:
                return False  # just created by another worker, pid not yet written
        except ProcessLookupError:
            pass
        shutil.rmtree(lock, ignore_errors=True)
        try:
            lock.mkdir()
        except FileExistsError:
            return False
    (lock / "pid").write_text(str(os.getpid()))
    return True


def release(name):
    shutil.rmtree(CELLS / "_locks" / name, ignore_errors=True)


def provenance():
    path = CELLS / "provenance.json"
    if path.exists():
        return
    import tiatoolbox
    import torch
    from tiatoolbox import rcParam
    from tiatoolbox.models.architecture import get_pretrained_model  # noqa: F401  (registry lookup below)
    import yaml
    registry = yaml.safe_load((Path(tiatoolbox.__file__).parent / "data" / "pretrained_model.yaml").read_text())[MODEL]
    weights = Path(rcParam["TIATOOLBOX_HOME"]) / "models" / f"{MODEL}.pth"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest() if weights.exists() else None
    record = {"engine": "tiatoolbox.models.engine.nucleus_instance_segmentor.NucleusInstanceSegmentor",
              "call": f'NucleusInstanceSegmentor(pretrained_model="{MODEL}") .predict(mode="tile")',
              "tiatoolbox_version": tiatoolbox.__version__, "torch_version": torch.__version__,
              "pretrained_model": MODEL, "weights_url": registry["url"], "weights_file": str(weights),
              "weights_sha256": digest, "weights_bytes": weights.stat().st_size if weights.exists() else None,
              "architecture": registry["architecture"], "ioconfig": registry["ioconfig"],
              "citation": "Graham S. et al., HoVer-Net, Med Image Anal 2019; weights trained on PanNuke "
                          "(Gamper J. et al., 2019/2020) and distributed by the TIA Centre (TIAToolbox model zoo).",
              "postprocessing_tile_shape": [SINGLE_TILE, SINGLE_TILE],
              "postprocessing_tile_note": "ioconfig tile_shape enlarged from 1024 so each image is one tile (no seam strips "
                                          "re-inferred); model, 256->164 px window and stride unchanged",
              "install": ".runtime/tiatoolbox/venv (python 3.12, pip tiatoolbox==1.6.0, numcodecs<0.16 for zarr 2); "
                         "pip freeze in .runtime/tiatoolbox/requirements.lock.txt",
              "classes": TYPES, "recorded": time.strftime("%F %T")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str))


# ---------------------------------------------------------------- ESCA windows
def esca_units(limit_windows=None, select=None):
    """All adopted windows; slides in a seeded random order (seed 20261026) so any finished prefix is an
    unbiased subset of patients (dev and test alike) for interim reports."""
    windows = pd.read_csv(WINDOWS)
    if select is not None:
        windows = windows.merge(select, on=["barcode", "window_id"])
    barcodes = sorted(windows.barcode.unique())
    order = {b: i for i, b in enumerate(np.random.default_rng(20261026).permutation(barcodes))}
    windows = windows.assign(_order=windows.barcode.map(order)).sort_values(["_order", "window_id"]).drop(columns="_order")
    if limit_windows:
        windows = windows.head(limit_windows)
    return windows


def slide_path(barcode):
    return Path(json.loads((ESCA_OUT / "slides" / barcode / "hooknet" / "summary.json").read_text())["slide"])


def run_esca(segment, windows):
    """One image per window: the 3328-px square covered by its 4x4 patches (x0 - 128 ... x0 + 3200), read at
    level 0; HoVer-Net's own tiling (256/164 px, margin-merged 1024-px tiles) runs inside it, and only nuclei
    in the 3072-px window core (the union of the 16 patch cores) are kept. patch_id is the core cell the
    nucleus falls in (0-15, row-major, as esca_windows numbers the patches)."""
    size = CORE_MARGIN + GRID * STRIDE + (PATCH - STRIDE - CORE_MARGIN)
    for barcode, part in windows.groupby("barcode", sort=False):
        todo = [w for w in part.window_id if not (CELLS / "esca" / barcode / f"{w}.csv.gz").exists()]
        if not todo or not claim(f"esca_{barcode}"):
            continue
        try:
            geometry = pd.read_csv(ESCA_OUT / "slides" / barcode / "he2cell" / "windows.csv").set_index("window_id")
            with open_slide(slide_path(barcode)) as handle:
                for window_id in todo:
                    if (CELLS / "esca" / barcode / f"{window_id}.csv.gz").exists():
                        continue
                    started = time.perf_counter()
                    w = geometry.loc[window_id]
                    x, y = int(w.x0) - CORE_MARGIN, int(w.y0) - CORE_MARGIN
                    try:
                        image = np.asarray(handle.read_region((x, y), 0, (size, size)).convert("RGB"))
                        (frame,) = segment([image], (CORE_MARGIN, CORE_MARGIN + GRID * STRIDE))
                    except Exception as error:  # keep the worker alive; the unit stays unfinished and is retried
                        print(f"{time.strftime('%F %T')} {barcode} {window_id}: FAILED {error!r}", flush=True)
                        continue
                    cells = frame.assign(
                        patch_id=((frame.cy - CORE_MARGIN) // STRIDE * GRID + (frame.cx - CORE_MARGIN) // STRIDE).astype(int),
                        centroid_x_global=frame.cx + x, centroid_y_global=frame.cy + y)
                    cells["type_name"] = cells.type.map(TYPES)
                    save_atomic(cells[COLUMNS], CELLS / "esca" / barcode / f"{window_id}.csv.gz")
                    seconds = time.perf_counter() - started
                    log_timing(unit=f"esca:{barcode}:{window_id}", images=1, pixels=size * size, device=segment.device,
                               seconds=round(seconds, 1), cells=len(cells), pid=os.getpid())
                    print(f"{time.strftime('%F %T')} {barcode} {window_id}: {len(cells)} cells, {seconds:.0f}s", flush=True)
        finally:
            release(f"esca_{barcode}")


# ---------------------------------------------------------------- Visium sections
def run_visium(segment, samples, max_patches=None):
    """Chunks of VISIUM_CHUNK consecutive patch_grid patches are the work unit (claimed per chunk, so CPU
    and MPS workers can share a section); the worker that sees every chunk finished writes the section."""
    for sample in samples:
        final = CELLS / "visium" / f"{sample}.csv.gz"
        if final.exists():
            continue
        parts_dir = CELLS / "visium" / "_parts" / sample
        parts_dir.mkdir(parents=True, exist_ok=True)
        scale = V.native_mpp(sample) / TARGET_MPP
        batch, total, claimed = [], 0, None

        def flush(batch, chunk):
            t0 = time.perf_counter()
            frames = segment([b[3] for b in batch], (CORE_MARGIN, CORE_MARGIN + STRIDE))
            out = [f.assign(patch_id=pid, centroid_x_global=f.cx + x, centroid_y_global=f.cy + y)
                   for (pid, x, y, _), f in zip(batch, frames)]
            cells = pd.concat(out, ignore_index=True)
            cells["type_name"] = cells.type.map(TYPES)
            save_atomic(cells[COLUMNS], parts_dir / f"chunk_{chunk:04d}.csv.gz")
            seconds = time.perf_counter() - t0
            log_timing(unit=f"visium:{sample}:chunk{chunk}", images=len(batch), pixels=len(batch) * PATCH * PATCH,
                       device=segment.device, seconds=round(seconds, 1), cells=len(cells), pid=os.getpid())
            print(f"{time.strftime('%F %T')} {sample} chunk {chunk}: {len(batch)} patches, {len(cells)} cells, {seconds:.0f}s", flush=True)

        for patch_id, (x, y, patch, _) in enumerate(V.patch_grid(sample, TARGET_MPP)):
            total += 1
            chunk = patch_id // VISIUM_CHUNK
            if max_patches and patch_id >= max_patches:
                continue
            if patch_id % VISIUM_CHUNK == 0:  # entering a new chunk: decide whether this worker does it
                claimed = None
                if not (parts_dir / f"chunk_{chunk:04d}.csv.gz").exists() and claim(f"visium_{sample}_{chunk:04d}"):
                    claimed = chunk
            if claimed != chunk:
                continue
            batch.append((patch_id, x, y, patch))
            if len(batch) == VISIUM_CHUNK:
                try:
                    flush(batch, chunk)
                except Exception as error:  # keep the worker alive; the chunk stays unfinished and is retried
                    print(f"{time.strftime('%F %T')} {sample} chunk {chunk}: FAILED {error!r}", flush=True)
                finally:
                    release(f"visium_{sample}_{chunk:04d}")
                batch, claimed = [], None
        if batch:
            try:
                flush(batch, claimed)
            except Exception as error:
                print(f"{time.strftime('%F %T')} {sample} chunk {claimed}: FAILED {error!r}", flush=True)
            finally:
                release(f"visium_{sample}_{claimed:04d}")
        if max_patches:
            continue
        chunks = (total + VISIUM_CHUNK - 1) // VISIUM_CHUNK
        if not all((parts_dir / f"chunk_{c:04d}.csv.gz").exists() for c in range(chunks)) or not claim(f"visium_{sample}_final"):
            continue  # other workers still busy on this section; the last one to finish writes it
        try:
            if final.exists():
                continue
            cells = pd.concat([pd.read_csv(parts_dir / f"chunk_{c:04d}.csv.gz") for c in range(chunks)], ignore_index=True)
            cells["x_native"], cells["y_native"] = cells.centroid_x_global / scale, cells.centroid_y_global / scale
            save_atomic(cells[COLUMNS + ["x_native", "y_native"]], final)
            (CELLS / "visium" / f"{sample}.summary.json").write_text(json.dumps(
                {"sample": sample, "patches": total, "chunks": chunks, "cells": len(cells), "upsampling": scale,
                 "target_mpp": TARGET_MPP, "types": cells.type_name.value_counts().to_dict()}, indent=2))
            shutil.rmtree(parts_dir, ignore_errors=True)
            print(f"{time.strftime('%F %T')} {sample}: {total} patches, {len(cells)} cells", flush=True)
        finally:
            release(f"visium_{sample}_final")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cohorts", nargs="+", choices=["esca", "visium"], help="Processed in this order")
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    parser.add_argument("--batch-size", type=int, default=8, help="CPU throughput collapses at 16 (memory)")
    parser.add_argument("--postproc-workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("HOVERNET_THREADS", "4")),
                        help="torch CPU threads; 4 per worker is the most efficient on this Mac, run several workers")
    parser.add_argument("--samples", nargs="+", default=V.SAMPLES)
    parser.add_argument("--windows-csv", help="Only these (barcode, window_id) rows (e.g. the timing sample)")
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--max-patches", type=int, help="Smoke test: stop each Visium section after this many patches")
    parser.add_argument("--tile-shape", type=int, default=SINGLE_TILE, help="Post-processing tile (0 = TIAToolbox default 1024)")
    parser.add_argument("--cells-dir", help="Write cell tables here instead of outputs/baselines_hovernet_20261026/cells")
    args = parser.parse_args()
    global CELLS
    if args.cells_dir:
        CELLS = Path(args.cells_dir)
    if args.max_patches and not args.cells_dir:
        parser.error("--max-patches writes partial sections; give a separate --cells-dir")
    import torch
    torch.set_num_threads(args.threads)
    (OUT / "_tmp").mkdir(parents=True, exist_ok=True)
    segment = Segmentor(args.device, args.batch_size, args.postproc_workers, args.tile_shape or None)
    for cohort in args.cohorts:
        if cohort == "esca":
            select = pd.read_csv(args.windows_csv)[["barcode", "window_id"]] if args.windows_csv else None
            run_esca(segment, esca_units(args.limit_windows, select))
        else:
            run_visium(segment, args.samples, args.max_patches)

if __name__ == "__main__":
    main()

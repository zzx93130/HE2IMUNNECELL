#!/usr/bin/env python3
"""Native Mac launcher for slide extraction, segmentation, and classification."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "script" / "src"
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(ROOT / "models" / "cellpose_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
sys.path.insert(0, str(SRC))


def default_classifier_path():
    # best_model.pth: a user-supplied override. he2cell_esca_finetuned.pth: the stain-robust default
    # (applies its .prior.json class-prior correction). best_model2.pth: the original checkpoint.
    for name in ("best_model.pth", "he2cell_esca_finetuned.pth", "best_model2.pth"):
        candidate = ROOT / "models" / name
        if candidate.is_file():
            return candidate
    return ROOT / "models" / "best_model.pth"


def doctor():
    import platform
    import torch
    import openslide
    from runtime import select_device

    device = select_device()
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    for package in ("torch", "torchvision", "cellpose", "openslide-python", "openslide-bin", "h5py"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"OpenSlide library: {openslide.__library_version__}")
    print(f"Apple GPU available: {torch.backends.mps.is_available()}")
    value = (torch.ones(4, device=device) * 2).sum().cpu().item()
    if value != 8:
        raise RuntimeError("Device computation failed")
    print(f"Device computation: PASS ({device})")
    checkpoint = default_classifier_path()
    print(f"Trained classifier: {checkpoint if checkpoint.is_file() else 'MISSING — copy a checkpoint into models/'}")
    sidecar = checkpoint.with_suffix(".prior.json")
    print(f"Class-prior correction: {'tau ' + str(json.loads(sidecar.read_text())['tau']) + ' (' + sidecar.name + ')' if sidecar.is_file() else 'none'}")
    custom = ROOT / "models" / "cpsam_v2"
    print(f"Local segmentation weights: {'present' if custom.is_file() else 'not provided; built-in cpsam_v2 is available by name'}")
    print(f"Official cpsam_v2 cache: {'present' if (ROOT / 'models' / 'cellpose_cache' / 'cpsam_v2').is_file() else 'downloads on first use'}")


def stage(script, *arguments, metrics_dir=None):
    command = [sys.executable, str(SRC / script), *map(str, arguments)]
    timing_path = None
    if metrics_dir is not None and sys.platform == "darwin":
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)
        timing_path = Path(metrics_dir) / f"{Path(script).stem}.resources.txt"
        command = ["/usr/bin/time", "-l", "-o", str(timing_path), *command]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    result = {"wall_seconds": time.perf_counter() - started}
    if timing_path is not None:
        for line in timing_path.read_text().splitlines():
            if "maximum resident set size" in line:
                result["peak_process_resident_bytes"] = int(line.split()[0])
    return result


def run(args):
    pipeline_started = time.perf_counter()
    from runtime import select_device, load_classifier_state
    from model_eval import SAM_ConvNet
    if args.storage == "on-demand":
        from archive_results import archive_slide

    source = Path(args.input).expanduser().resolve()
    checkpoint = Path(args.classifier).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")
    extensions = {".svs", ".tif", ".tiff"}
    slides = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    if not slides or any(p.suffix.lower() not in extensions for p in slides):
        raise ValueError("Select an SVS/TIF/TIFF slide or a folder containing those slides.")
    if len({p.stem for p in slides}) != len(slides):
        raise ValueError("Slide names must be unique within a run.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output folder is not empty; select a new output folder: {output}")
    if not 0 < args.stride <= args.patch_size:
        raise ValueError("stride must be positive and no larger than patch_size")
    if min(args.segmentation_batch, args.classification_batch, args.workers) < 1:
        raise ValueError("Batch sizes and workers must be positive")
    if not 0 <= args.min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")
    if args.max_patches is not None and args.max_patches < 1:
        raise ValueError("max_patches must be positive")
    device = select_device(args.device)
    # Validate architecture before reading any large slides or downloading segmentation weights.
    validation_model = SAM_ConvNet(num_classes=4, pretrained=False)
    validation_model.load_state_dict(load_classifier_state(checkpoint))
    del validation_model
    builtins = {"cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"}
    segmentation_model = args.cellpose_model
    if segmentation_model not in builtins:
        segmentation_model = str(Path(segmentation_model).expanduser().resolve())
        if not Path(segmentation_model).is_file():
            raise FileNotFoundError(f"Segmentation model not found: {segmentation_model}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now().astimezone().isoformat(),
        "input": str(source), "classifier": str(checkpoint),
        "segmentation_model": segmentation_model, "device": str(device),
        "settings": vars(args), "status": "running", "slides": [],
        "scope": "sampled tissue regions" if args.max_patches else "full slide",
        "stages": {},
        "software": {"python":sys.version.split()[0],
                     **{package:importlib.metadata.version(package) for package in
                        ["torch", "torchvision", "cellpose", "openslide-python"]}},
    }
    manifest_path = output / "run.json"

    def save_manifest():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    def finish_slide(entry, sample_output):
        if args.storage == "on-demand":
            entry["storage"] = {"mode":"on-demand", "status":"archiving",
                                "archive":str(sample_output / "results.h5")}
        manifest["slides"].append(entry)
        save_manifest()
        if args.storage == "on-demand":
            entry["storage"] = archive_slide(sample_output, entry["slide"], manifest_path,
                                             cleanup=True, patch_size=args.patch_size)
            entry["results"] = entry["storage"]["archive"]
            entry["results_format"] = "he2cell.slide-results.v1"
            manifest["stages"]["archiving"] = {"wall_seconds":entry["storage"]["wall_seconds"]}
        else:
            entry["storage"] = {"mode":"files"}
            entry["results_format"] = "csv"
        save_manifest()

    save_manifest()
    try:
        import pandas as pd
        from run_czi_cellpose_pipeline import process_one_tif

        for slide in slides:
            sample_output = output / slide.stem
            patches_output = sample_output / "patches"
            print(f"\nProcessing {slide.name} on {device}", flush=True)
            extraction_started = time.perf_counter()
            patches = process_one_tif(
                slide, patches_output, patch_size=args.patch_size, stride=args.stride,
                min_coverage=args.min_coverage, tissue_mode=args.tissue_mode,
                num_workers=args.workers,
                max_patches=args.max_patches,
            )
            manifest["stages"]["extraction"] = {"wall_seconds": time.perf_counter() - extraction_started,
                                                   "patches": len(patches)}
            save_manifest()
            if patches.empty:
                result_csv = sample_output / "cell_types.csv"
                pd.DataFrame(columns=["global_instance_id", "celltype", "prediction"]).to_csv(result_csv,index=False)
                pd.DataFrame(columns=["celltype", "count"]).to_csv(sample_output / "cell_counts.csv",index=False)
                finish_slide({"slide":str(slide),"status":"no tissue patches","cells":0,
                              "counts":{},"results":str(result_csv)},sample_output)
                continue
            patch_csv = patches_output / "tables" / f"{slide.stem}_patch_index.csv"
            cells_output = sample_output / "cells"
            manifest["stages"]["segmentation"] = stage("run_cellcrops.py", "--patch_csv", patch_csv, "--out_dir", cells_output,
                  "--model_name", segmentation_model, "--device", device,
                  "--num_gpus", 1, "--batch_size", args.segmentation_batch, "--crop_size", 64,
                  "--attention", args.segmentation_attention,
                  metrics_dir=sample_output / "performance")
            save_manifest()
            cells_csv = cells_output / "mask_tables" / f"{patch_csv.stem}_cellpose_mask_instances.csv"
            result_csv = sample_output / "cell_types.csv"
            manifest["stages"]["classification"] = stage("model_eval.py", "--meta_path", cells_csv, "--save_path", checkpoint,
                  "--output_path", result_csv, "--device", device,
                  "--batch_size", args.classification_batch, "--num_workers", 0,
                  "--tta", args.classification_tta, "--prior_correction", args.prior_correction,
                  metrics_dir=sample_output / "performance")
            save_manifest()
            result = pd.read_csv(result_csv)
            if not result.empty:
                manifest["stages"]["overlay"] = stage("draw_cell.py", "--csv_path", result_csv, "--tif_dir", slide.parent,
                      "--output_dir", sample_output / "overlay", "--celltype_col", "celltype",
                      "--sample_col", "source_tif", "--types", "Other,T,Macrophage,B", "--line_width", 1,
                      metrics_dir=sample_output / "performance")
            counts = {str(k): int(v) for k, v in result["celltype"].value_counts().items()}
            pd.DataFrame(list(counts.items()), columns=["celltype", "count"]).to_csv(
                sample_output / "cell_counts.csv", index=False)
            finish_slide({"slide": str(slide), "status": "complete",
                          "cells": len(result), "counts": counts, "results": str(result_csv)}, sample_output)
        manifest["status"] = "complete"
        print(f"\nFinished. Results: {output}")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - pipeline_started
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        save_manifest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check installation and Apple GPU access")
    runner = commands.add_parser("run", help="Process one slide or a folder of slides")
    runner.add_argument("--input", required=True)
    runner.add_argument("--output", required=True)
    runner.add_argument("--classifier", default=str(default_classifier_path()))
    runner.add_argument("--cellpose-model", default=str(ROOT / "models" / "cpsam_v2")
                        if (ROOT / "models" / "cpsam_v2").is_file() else "cpsam_v2")
    runner.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    runner.add_argument("--patch-size", type=int, default=1024)
    runner.add_argument("--stride", type=int, default=512)
    runner.add_argument("--min-coverage", type=float, default=0.25)
    runner.add_argument("--tissue-mode", choices=["dark", "light"], default="dark")
    runner.add_argument("--segmentation-batch", type=int, default=32)
    runner.add_argument("--segmentation-attention", choices=["auto", "original", "sdpa"], default="auto")
    runner.add_argument("--classification-batch", type=int, default=128)
    runner.add_argument("--classification-tta", choices=["d4","none"], default="d4",
                        help="Average eight joint RGB/mask views by default; none reproduces identity-only classification")
    runner.add_argument("--prior-correction", choices=["auto", "none"], default="auto",
                        help="auto applies the checkpoint's .prior.json class-prior correction when present")
    runner.add_argument("--workers", type=int, default=4)
    runner.add_argument("--storage", choices=["on-demand", "files"], default="on-demand",
                        help="Default: verify a compact archive, then remove archived loose files; files keeps PNG/TIFF/CSV outputs")
    runner.add_argument("--max-patches", type=int, default=None,
                        help="Preview only: process at most this many evenly spaced tissue patches")
    exporter = commands.add_parser("export", help="Restore archived results for review or legacy analysis scripts")
    exporter.add_argument("--archive", required=True)
    exporter.add_argument("--output", required=True, help="New or empty output directory")
    exporter.add_argument("--source-slide", help="Relocated original slide (checksum verified)")
    exporter.add_argument("--tables-only", action="store_true",
                          help="Export tables and review files without reconstructing images; no source slide required")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            doctor()
        elif args.command == "export":
            from archive_results import export_archive
            result = export_archive(args.archive,args.output,args.source_slide,args.tables_only)
            print(f"Exported results: {result}")
        else:
            run(args)
    except (Exception, KeyboardInterrupt) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

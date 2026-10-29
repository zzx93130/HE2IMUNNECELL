"""Compare segmentation settings on fixed real patches against saved float32 masks."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(ROOT / "models/cellpose_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
sys.path.insert(0, str(ROOT / "script/src"))


def compare_masks(reference, candidate):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    a, b = reference.ravel().astype(int), candidate.ravel().astype(int)
    overlap = coo_matrix((np.ones(len(a), dtype=np.int64), (a, b)),
                         shape=(int(a.max())+1, int(b.max())+1)).toarray()
    areas_a = np.bincount(a, minlength=overlap.shape[0])
    areas_b = np.bincount(b, minlength=overlap.shape[1])
    intersection = overlap[1:,1:]
    union = areas_a[1:,None] + areas_b[None,1:] - intersection
    ious = intersection / np.maximum(union,1)
    if ious.size:
        rows, cols = linear_sum_assignment(-ious)
        scores = ious[rows,cols]
        matches = int((scores >= .5).sum())
        mean_iou = float(scores.mean())
    else:
        matches, mean_iou = 0, None
    return {"reference_instances":len(areas_a)-1, "candidate_instances":len(areas_b)-1,
            "instance_matches_iou_at_least_0_5":matches, "mean_matched_iou":mean_iou,
            "foreground_pixel_agreement":float(((a>0)==(b>0)).mean()),
            "exact_label_image_equal":bool(np.array_equal(reference,candidate))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="8,16,32")
    parser.add_argument("--precision", choices=["float32","bfloat16","float16"], default="float32")
    parser.add_argument("--attention", choices=["original","sdpa"], default="original")
    parser.add_argument("--patches", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    import torch
    from cellpose import models
    from run_cellcrops import read_patch_image
    torch.set_num_threads(8)
    run = ROOT / "outputs/whole-slide-20261012"
    manifest = json.loads((run / "run.json").read_text())
    sample = Path(manifest["slides"][0]["results"]).parent
    patches = pd.read_csv(next((sample / "patches/tables").glob("*_patch_index.csv")))
    chosen = patches.iloc[np.linspace(int(len(patches)*.1), int(len(patches)*.9), args.patches, dtype=int)]
    images = [read_patch_image(p) for p in chosen.path]
    references = [tifffile.imread(sample / "cells/patch_masks" / row.sample / (Path(row.path).stem + "_cp_mask.tif"))
                  for row in chosen.itertuples()]
    model = models.CellposeModel(gpu=True, device=torch.device("mps"), pretrained_model="cpsam_v2",
                                use_bfloat16=args.precision == "bfloat16")
    if args.precision == "float16":
        model.net.dtype = torch.float16
    if args.attention == "sdpa":
        from segmentation_acceleration import enable_sdpa_attention
        enable_sdpa_attention(model)
    report = {"precision":args.precision, "attention":args.attention,
              "torch_version":torch.__version__,
              "patch_ids":chosen.num_patch.tolist(), "patches":len(images), "results":[]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for batch in map(int,args.batches.split(",")):
        # Warm the network and the MPS graph caches for this batch shape.
        model.eval(images[:1], batch_size=batch, channel_axis=-1, compute_masks=False)
        torch.mps.synchronize()
        neural_started = time.perf_counter()
        model.eval(images, batch_size=batch, channel_axis=-1, compute_masks=False)
        torch.mps.synchronize()
        neural_seconds = time.perf_counter() - neural_started
        timings = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            masks,_,_ = model.eval(images, batch_size=batch, channel_axis=-1, min_size=15)
            torch.mps.synchronize()
            timings.append(time.perf_counter()-started)
        item = {"batch_size":batch, "seconds":timings, "median_seconds":float(np.median(timings)),
                "network_without_mask_dynamics_seconds":neural_seconds,
                "seconds_per_patch":float(np.median(timings)/len(images)),
                "observed_mps_driver_bytes":torch.mps.driver_allocated_memory(),
                "mask_comparisons":[compare_masks(a,b) for a,b in zip(references,masks)]}
        report["results"].append(item)
        output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(item), flush=True)
    print(f"Saved benchmark: {output}",flush=True)


if __name__ == "__main__":
    main()

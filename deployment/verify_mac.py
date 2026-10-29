"""Deployment smoke test, using artificial images and untrained classifier weights.

This tests execution and file contracts, never biological accuracy.
Run: .venv/bin/python deployment/verify_mac.py --device mps
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(ROOT / "models" / "cellpose_cache"))
sys.path.insert(0, str(ROOT / "script" / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--skip-segmentation", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw
    import torch
    from model_eval import SAM_ConvNet, get_transforms
    from runtime import select_device
    from dataset import CellSamdataset_our
    from run_cellcrops import crop_with_padding
    from run_czi_cellpose_pipeline import detect_tissue_regions, process_one_tif
    from slide_io import open_slide

    torch.set_num_threads(8)
    started = time.monotonic()
    device = select_device(args.device)
    checks = {}
    (ROOT / ".runtime").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verification-", dir=ROOT / ".runtime") as temporary:
        temp = Path(temporary)
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((16, 16, 239, 239), fill=(200, 120, 175))
        for y in range(40, 230, 32):
            for x in range(40, 230, 32):
                draw.ellipse((x - 8, y - 6, x + 8, y + 6), fill=(90, 35, 118))
        slide_path = temp / "artificial_slide.tif"
        image.save(slide_path)
        with open_slide(slide_path) as slide:
            mask, _ = detect_tissue_regions(slide, kernel_size=1, tissue_mode="dark")
            assert mask[0, 0] == 0 and mask[40, 40] == 255
        patches = process_one_tif(slide_path, temp / "patches", patch_size=256,
                                 stride=256, min_coverage=0.25, tissue_mode="dark", num_workers=2)
        assert len(patches) == 1 and patches.iloc[0]["tissue_ratio"] > 0.5
        checks["ordinary_tiff_extraction_and_dark_tissue_selection"] = "pass"
        empty_path = temp / "blank.tif"
        Image.new("RGB", (256, 256), "white").save(empty_path)
        blank = process_one_tif(empty_path, temp / "blank", patch_size=256,
                               stride=256, tissue_mode="dark", num_workers=1)
        assert blank.empty and "path" in blank.columns
        assert pd.read_csv(temp / "blank" / "tables" / "blank_patch_index.csv").empty
        checks["blank_slide_retains_csv_headers"] = "pass"
        a = np.arange(100).reshape(10, 10)
        padded = crop_with_padding(a, 0, 0, crop_size=4, pad_value=-1)
        assert (padded[:2, :] == -1).all() and (padded[2:, 2:] == a[:2, :2]).all()
        checks["cell_crop_edge_padding"] = "pass"

        crop_path = temp / "cell.png"
        mask_path = temp / "mask.png"
        image.resize((64, 64)).save(crop_path)
        Image.new("L", (64, 64), 255).save(mask_path)
        metadata = pd.DataFrame({"global_instance_id": [5, 8], "sample": ["slide", "slide"],
                                 "cell_image_crop_path": [str(crop_path)] * 2,
                                 "cell_mask_crop_path": [str(mask_path)] * 2})
        _, transform = get_transforms()
        transformed, _ = CellSamdataset_our(metadata, transform)[0]
        rgb = np.asarray(Image.open(crop_path)).astype(np.float32) / 255
        expected = (torch.from_numpy(rgb).permute(2, 0, 1) - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        torch.testing.assert_close(transformed[:3], expected)
        assert torch.all(transformed[3] == 1)
        checks["rgb_normalization_and_fourth_mask_channel"] = "pass"
        torch.manual_seed(569)
        model = SAM_ConvNet(num_classes=4, pretrained=False).eval()
        with torch.inference_mode():
            cpu_logits = model(transformed[None])
            model = model.to(device)
            gpu_logits = model(transformed[None].to(device)).cpu()
        torch.testing.assert_close(gpu_logits, cpu_logits, rtol=1e-3, atol=1e-3)
        checks["classifier_cpu_device_logit_agreement"] = {
            "status": "pass", "maximum_absolute_error": float((gpu_logits - cpu_logits).abs().max())}
        model = model.cpu()
        with torch.no_grad():
            model.backbone.fc.weight.zero_()
            model.backbone.fc.bias.copy_(torch.tensor([0., 10., 0., 0.]))
        checkpoint = temp / "UNTRAINED_TEST_ONLY.pth"
        torch.save({"model_state_dict": {"module." + key: value for key, value in model.state_dict().items()}}, checkpoint)
        del model
        input_csv = temp / "input.csv"
        output_csv = temp / "classification.csv"
        metadata.to_csv(input_csv, index=False)
        subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "script" / "src" / "model_eval.py"),
                        "--meta_path", str(input_csv), "--output_path", str(output_csv),
                        "--save_path", str(checkpoint), "--device", args.device,
                        "--batch_size", "2", "--num_workers", "0"], check=True)
        result = pd.read_csv(output_csv)
        assert len(result) == 2 and result["celltype"].tolist() == ["T", "T"]
        assert result["global_instance_id"].tolist() == [5, 8] and result["sample"].tolist() == ["slide", "slide"]
        checks["linux_dataparallel_checkpoint_loading_and_csv_classification"] = "pass"
        metadata.iloc[:0].to_csv(temp / "empty.csv", index=False)
        subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "script" / "src" / "model_eval.py"),
                        "--meta_path", str(temp / "empty.csv"), "--output_path", str(temp / "empty_result.csv"),
                        "--save_path", str(checkpoint), "--device", args.device], check=True)
        assert pd.read_csv(temp / "empty_result.csv").empty
        checks["empty_classification_csv"] = "pass"
        if not args.skip_segmentation:
            output = temp / "pipeline"
            subprocess.run([str(ROOT / "he2cell"), "run", "--input", str(slide_path),
                            "--output", str(output), "--classifier", str(checkpoint),
                            "--cellpose-model", "cpsam_v2", "--device", args.device,
                            "--patch-size", "256", "--stride", "256", "--segmentation-batch", "2"], check=True)
            manifest = json.loads((output / "run.json").read_text())
            assert manifest["status"] == "complete"
            from result_store import SlideResults
            archive = output / "artificial_slide" / "results.h5"
            assert manifest["slides"][0]["storage"]["loose_files_removed"]
            assert not (output / "artificial_slide" / "cell_types.csv").exists()
            with SlideResults(archive) as stored:
                full_result = stored.table()
                assert set(full_result["celltype"]).issubset({"T"})
                if not full_result.empty:
                    rgb, mask = stored.read_cell(0)
                    assert rgb.shape == (64,64,3) and mask.shape == (64,64)
                    assert len(stored.h5["files/overlay"]) == 1
            checks["official_cellpose_pipeline"] = {"status": "pass", "synthetic_cells": len(full_result)}

    report = {"device": str(device), "seconds": round(time.monotonic() - started, 2),
              "classifier_weights": "synthetic untrained test only; deleted after test",
              "biological_accuracy_validated": False, "checks": checks}
    report_path = ROOT / "deployment" / f"verification-{args.device}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

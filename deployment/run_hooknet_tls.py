"""Run the published HookNet-TLS network and filtering on a complete slide on Mac."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from hooknet_tls_native import ROOT, RUNTIME, load_model, load_postprocessing, NativeAnnotation, NativeLabel

import cv2
import numpy as np
import openslide
import tifffile

if np.__version__ != "1.26.4":
    raise RuntimeError("HookNet-TLS requires its pinned NumPy 1.26.4 environment; run setup_hooknet_tls.py again")


def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def padded_patch(array, center_x, center_y, size):
    """Integer centered reads and black out-of-image padding, as in upstream readers."""
    x, y = int(center_x) - size//2, int(center_y) - size//2
    result = np.zeros((size, size, *array.shape[2:]), dtype=array.dtype)
    sx, sy = max(0, x), max(0, y)
    ex, ey = min(array.shape[1], x+size), min(array.shape[0], y+size)
    if ex > sx and ey > sy:
        result[sy-y:ey-y, sx-x:ex-x] = array[sy:ey, sx:ex]
    return result


def write_pyramid(path, arrays, spacing=.5, rgb=False):
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for level, array in enumerate(arrays):
            scale = spacing * 2**level
            writer.write(array, photometric="rgb" if rgb else "minisblack", tile=(256, 256),
                compression="deflate", compressionargs={"level": 3}, metadata=None,
                subfiletype=0 if level == 0 else 1, resolution=(10000/scale, 10000/scale), resolutionunit="CENTIMETER")


def physical_remap_grid(w,h,col,row,ratio_x,ratio_y,sx,sy):
    # NumPy 1.x can promote float32 arrays when an integer origin exceeds
    # 65535. OpenCV requires both maps to remain float32, even on large WSIs.
    map_x,map_y=np.meshgrid((np.arange(w,dtype=np.float32)+col+.5)*ratio_x-.5-sx,
                           (np.arange(h,dtype=np.float32)+row+.5)*ratio_y-.5-sy)
    return map_x.astype(np.float32,copy=False),map_y.astype(np.float32,copy=False)


def prepare_slide(slide_path, out):
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_holes, remove_small_objects
    folder = out / "preprocessed"
    folder.mkdir(exist_ok=True)
    stage_started = time.perf_counter()
    with openslide.OpenSlide(slide_path) as slide:
        mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        if not np.isclose(mpp, float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])):
            raise ValueError("Anisotropic slide pixels require an explicit resampling adapter")
        width, height = (int(round(d*mpp/.5)) for d in slide.dimensions)
        shape = (height, width, 3)
        base_path = folder / "RGB_0p5um.npy"
        completion=folder/'RGB_0p5um.complete.json'
        # Older completed runs have preprocessing.json. A bare NPY may be a
        # partially written cache after interruption and must never be reused.
        cache_reused = base_path.exists() and (completion.exists() or (folder/'preprocessing.json').exists())
        if not cache_reused:
            partial=folder/'RGB_0p5um.partial.npy'
            base = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=shape)
            ratio_x, ratio_y = slide.dimensions[0]/width, slide.dimensions[1]/height
            for row in range(0, height, 1024):
                for col in range(0, width, 1024):
                    h, w = min(1024, height-row), min(1024, width-col)
                    sx, sy = max(0, int(col*ratio_x)-1), max(0, int(row*ratio_y)-1)
                    sw = int(np.ceil((col+w)*ratio_x)) - sx + 1
                    sh = int(np.ceil((row+h)*ratio_y)) - sy + 1
                    source = np.asarray(slide.read_region((sx, sy), 0, (sw, sh)))[..., :3]
                    map_x,map_y=physical_remap_grid(w,h,col,row,ratio_x,ratio_y,sx,sy)
                    base[row:row+h, col:col+w] = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR)
                base.flush()
                if row%8192==0: print(f'Physical RGB preparation: {row+h}/{height} rows',flush=True)
            del base
            partial.replace(base_path)
            save_json({'source_slide':str(slide_path),'shape':list(shape),'complete':True},completion)
            print("Lossless 0.5 µm image prepared", shape, flush=True)
        base = np.load(base_path, mmap_mode="r")
        if base.shape != shape:
            raise ValueError("Cached physical image dimensions do not match this slide")
        pyramid = [base]
        for level in range(1, 6):
            path = folder / f"RGB_level_{level}.npy"
            if not path.exists():
                previous = pyramid[-1]
                resized = cv2.resize(previous, ((previous.shape[1]+1)//2, (previous.shape[0]+1)//2), interpolation=cv2.INTER_LINEAR)
                np.save(path, resized)
            pyramid.append(np.load(path, mmap_mode="r"))
        dense = folder / "slide_0p5um_pyramid.tif"
        if not dense.exists():
            write_pyramid(dense, pyramid, rgb=True)
        # Same Otsu + 64-pixel component/hole cleanup as the published preprocessing.
        gray = rgb2gray(pyramid[4])  # 8 µm
        threshold = float(threshold_otsu(gray))
        mask8 = remove_small_holes(remove_small_objects(gray < threshold, min_size=64), area_threshold=64)
        mask2 = np.repeat(np.repeat(mask8, 4, axis=0), 4, axis=1)[:pyramid[2].shape[0], :pyramid[2].shape[1]].astype(np.uint8)
        mask = np.repeat(np.repeat(mask2, 4, axis=0), 4, axis=1)[:height, :width]
        np.save(folder / "tissue_mask_0p5um.npy", mask)
        write_pyramid(folder / "tissue_mask_2um.tif", [mask2, mask2[::2, ::2], mask2[::4, ::4]], spacing=2.)
        info = {"source_slide": str(slide_path), "source_dimensions": list(slide.dimensions), "source_mpp": mpp,
            "physical_input_spacing_um": .5, "context_spacing_um": 2., "dimensions_at_0p5um": [width, height],
            "maximum_frame_rounding_error_um": float(max(abs(slide.dimensions[0]*mpp-width*.5), abs(slide.dimensions[1]*mpp-height*.5))),
            "Otsu_gray_threshold": threshold, "mask_processing_spacing_um": 8, "mask_spacing_um": 2,
            "mask_minimum_object_and_hole_pixels": 64, "tissue_mask_pixels": int(mask.sum()),
            "RGB_pyramid_cache_reused": cache_reused,
            "preprocessing_stage_seconds": time.perf_counter()-stage_started,
            "I_O_adaptation": "OpenSlide 4 + lossless, bilinear-resampled pyramidal TIFF; replaces Linux ASAP/JPEG I/O"}
        save_json(info, folder / "preprocessing.json")
    return pyramid, mask, info


def infer_and_verify(model, fine, context, center, out, requested):
    import tensorflow as tf
    high = padded_patch(fine, *center, 1244)[None].astype(np.float32)
    low = padded_patch(context, center[0]//4, center[1]//4, 1244)[None].astype(np.float32)
    functions = {}
    for name in ["CPU", "GPU"]:
        with tf.device(f"/{name}:0"):
            functions[name] = tf.function(lambda a, b: model([a, b], training=False), autograph=False)
    start = time.perf_counter()
    with tf.device("/CPU:0"):
        cpu = functions["CPU"](high, low).numpy().reshape(model.output_shape)
    cpu_seconds = time.perf_counter() - start
    assert np.isfinite(cpu).all()
    np.testing.assert_allclose(cpu.sum(axis=-1), 1, atol=1e-5)
    with tf.device("/CPU:0"):
        native = model.predict_on_batch([high, low], argmax=False)[0]
    np.testing.assert_allclose(cpu, native, atol=1e-5, rtol=1e-5)
    result = {"official_predict_on_batch_agrees_with_direct_forward": True,
              "all_checkpoint_weights_loaded": True, "model_output_shape": list(model.output_shape),
              "writer_tile_shape": [1024, 1024], "writer_crop": "center-crop tissue mask to 1030; upper-left crop prediction/mask to 1024, exactly as published",
              "CPU_first_pass_seconds": cpu_seconds, "CPU_softmax_verified": True,
              "GPU_available": bool(tf.config.list_physical_devices("GPU"))}
    selected = "CPU"
    if requested != "cpu" and result["GPU_available"]:
        try:
            start = time.perf_counter()
            with tf.device("/GPU:0"):
                gpu = functions["GPU"](high, low).numpy().reshape(model.output_shape)
            agreement = float(np.mean(cpu.argmax(-1) == gpu.argmax(-1)))
            error = float(np.max(np.abs(cpu-gpu)))
            result.update({"GPU_first_pass_seconds": time.perf_counter()-start,
                           "CPU_GPU_pixel_class_agreement": agreement, "CPU_GPU_max_probability_difference": error})
            if agreement < .995 or error > .01:
                raise ValueError("GPU predictions differ beyond the declared verification tolerance")
            selected = "GPU"
        except Exception as error:
            result["GPU_fallback_reason"] = str(error)
            if requested == "gpu":
                raise
    result["selected_device"] = selected
    save_json(result, out / "model_verification.json")
    print("Model verification", result, flush=True)
    return functions[selected], selected


def run_inference(model, function, device, pyramid, mask, out):
    import tensorflow as tf
    shape = mask.shape
    maps = {name: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.uint8, shape=shape)
            for name in ["prediction_mask", "TLS_confidence", "GC_confidence"]}
    jobs = []
    # Authored PatchConfiguration: 1244 input, overlap 220, offset (-512,-512), centered.
    for y in range(-512, shape[0], 1024):
        for x in range(-512, shape[1], 1024):
            if x-512 < 0 or y-512 < 0:
                continue  # upstream writer drops out-of-image tile coordinates
            if np.any(padded_patch(mask, x, y, 1244)):
                jobs.append((x, y))
    if not jobs:
        raise ValueError("No tissue tiles found")
    records = []
    started = time.perf_counter()
    for number, (x, y) in enumerate(jobs, 1):
        high = padded_patch(pyramid[0], x, y, 1244)[None].astype(np.float32)
        low = padded_patch(pyramid[2], x//4, y//4, 1244)[None].astype(np.float32)
        start = time.perf_counter()
        with tf.device(f"/{device}:0"):
            probabilities = function(high, low).numpy().reshape(model.output_shape)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite model output at tile {number}")
        local_mask = padded_patch(mask, x, y, 1244)[107:-107, 107:-107][:1024, :1024]
        values = {"prediction_mask": ((probabilities.argmax(-1)+1)[:1024, :1024]*local_mask).astype(np.uint8),
                  "TLS_confidence": (probabilities[:1024, :1024, 1]*255*local_mask).astype(np.uint8),
                  "GC_confidence": (probabilities[:1024, :1024, 2]*255*local_mask).astype(np.uint8)}
        col, row = x-512, y-512
        h, w = min(1024, shape[0]-row), min(1024, shape[1]-col)
        for name, value in values.items():
            maps[name][row:row+h, col:col+w] = value[:h, :w]
            maps[name].flush()
        records.append({"tile": number, "center_x_at_0p5um": x, "center_y_at_0p5um": y,
                        "inference_seconds": time.perf_counter()-start})
        if number == 1 or number % 5 == 0 or number == len(jobs):
            elapsed = time.perf_counter()-started
            save_json({"stage": "whole_slide_inference", "completed": number, "total": len(jobs), "elapsed_seconds": elapsed}, out / "progress.json")
            print(f"HookNet-TLS {number}/{len(jobs)} tissue tiles; {elapsed:.1f}s elapsed", flush=True)
    with (out / "tile_timings.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    for name, array in maps.items():
        write_pyramid(out / f"{name}.tif", [array[::2**level, ::2**level] for level in range(6)])
    return maps, {"whole_slide_tissue_tiles": len(jobs), "all_tissue_tiles_completed": True,
                  "inference_and_map_write_seconds": time.perf_counter()-started}


class MaskReader:
    def __init__(self, array):
        self.array = array

    def get_downsampling_from_spacing(self, spacing):
        return spacing/.5

    def get_slide(self, spacing):
        stride = int(spacing/.5)
        return self.array[::stride, ::stride]


def postprocess(maps, config, out):
    from shapely.geometry import Polygon, mapping
    functions = load_postprocessing(out / "source_audit")
    polygons = functions["cv2_polygonize"](MaskReader(maps["prediction_mask"]), spacing=4., dilation_iterations=1,
        erose_iterations=1, fill_holes=True, values=[2, 3])
    annotation_groups = {"tls": [], "gc": []}
    heatmaps = {"tls": maps["TLS_confidence"], "gc": maps["GC_confidence"]}
    for value, name in [(2, "tls"), (3, "gc")]:
        for vertices in polygons[value]:
            geometry = Polygon(vertices).buffer(0)
            parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
            for part in parts:
                if part.is_empty or part.area <= 0:
                    continue
                xmin, ymin, xmax, ymax = map(int, part.bounds)
                local = heatmaps[name][max(0,ymin):min(heatmaps[name].shape[0],ymax), max(0,xmin):min(heatmaps[name].shape[1],xmax)]
                values, counts = np.unique(local, return_counts=True)
                histogram = dict(zip(map(int, values), map(int, counts)))
                annotation_groups[name].append(NativeAnnotation(len(annotation_groups[name])+1, part, NativeLabel(name, histogram)))
    tls_settings, gc_settings = config["tls_filter_settings"], config["gc_filter_settings"]
    tls = functions["filter_tls_annotations"](annotation_groups["tls"], tls_settings["confidence_value"], tls_settings["confidence_count"])
    gc = functions["filter_gc_annotations"](tls, annotation_groups["gc"], gc_settings["confidence_value"], gc_settings["confidence_count"])
    results = []
    for name in ["tls", "gc"]:
        retained = tls if name == "tls" else gc
        for annotation in annotation_groups[name]:
            geometry = annotation.geometry
            results.append({"type": "Feature", "geometry": mapping(geometry), "properties": {
                "region_id": f"HookNet_{name.upper()}_{annotation.index}", "predicted_label": name,
                "passes_published_filter": any(annotation is keep for keep in retained),
                "coordinate_units": "pixels_at_0.5_micrometers", "area_um2": float(geometry.area*.25),
                "confidence_map_8bit_in_bounding_box": annotation.label.confidence_map}})
    save_json({"type": "FeatureCollection", "features": results}, out / "native_polygons.geojson")
    info = {"raw_TLS_polygons": len(annotation_groups["tls"]), "filtered_TLS_polygons": len(tls),
            "raw_GC_polygons": len(annotation_groups["gc"]), "filtered_GC_polygons": len(gc),
            "TLS_filter": tls_settings, "GC_filter": gc_settings, "polygonization_spacing_um": 4,
            "polygonization": "Authored dilation(1), fill holes, erosion(1), OpenCV RETR_LIST/CHAIN_APPROX_NONE",
            "confidence_support": "Authored 8-bit heatmap histogram over annotation bounding rectangle, not masked polygon",
            "GC_association": "Authored >=50% GC polygon area inside a retained TLS bounding rectangle"}
    save_json(info, out / "native_summary.json")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "whole-slide-20261012")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "tls_tool_comparison" / "hooknet_tls"
    out.mkdir(parents=True, exist_ok=True)
    audit = out / "source_audit"; audit.mkdir(exist_ok=True)
    summary = json.loads((run / "tls_analysis" / "spatial_summary.json").read_text())
    started = time.perf_counter()
    if args.postprocess_only:
        import yaml
        manifest = json.loads((RUNTIME / "manifest.json").read_text())
        config = yaml.safe_load((Path(manifest["sources"]["hooknet_tls"]["directory"])/"hooknettls/configs/config.yml").read_text())["default"]
        maps = {name: np.load(out / f"{name}.npy", mmap_mode="r") for name in ["prediction_mask", "TLS_confidence", "GC_confidence"]}
        postprocess(maps, config, out)
        return
    pyramid, mask, preprocessing = prepare_slide(summary["slide"], out)
    if args.prepare_only:
        return
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with tf.device("/CPU:0"):
        model, config, manifest = load_model(audit)
    candidates = json.loads((run / "tls_analysis" / "candidate_coordinates.json").read_text())
    center = [int(round(value/.5)) for value in candidates[0]["center_um"]]
    function, device = infer_and_verify(model, pyramid[0], pyramid[2], center, out, args.device)
    maps, timings = run_inference(model, function, device, pyramid, mask, out)
    post_summary = postprocess(maps, config, out)
    provenance = {"status": "inference_complete", "scope": "whole slide, all tissue-mask-positive grid tiles",
        "source_slide": summary["slide"], "device": device, "setup": manifest, "preprocessing": preprocessing,
        "tensorflow": tf.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
        "openslide_library": openslide.__library_version__, "model_parameters": model.count_params(), "loaded_weight_arrays": len(model.weights),
        "model_output_shape": list(model.output_shape), "timings": timings, "native_summary": post_summary,
        "wall_seconds_including_preprocessing_verification_and_postprocessing": time.perf_counter()-started,
        "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "independent_of_our_cell_labels": True, "reference_TLS_annotations_available": False,
        "limits": ["Mac native port; no same-input Linux/ASAP reference predictions available for numerical equivalence",
                  "Different TensorFlow version, reader and lossless physical resampling from the Linux Docker recipe",
                  "Predicted TLS and GC are model outputs, not confirmed structures or maturity labels"]}
    save_json(provenance, out / "provenance.json")
    for name in ["run_hooknet_tls.py", "hooknet_tls_native.py", "setup_hooknet_tls.py"]:
        shutil.copy2(ROOT / "deployment" / name, audit / name)
    checks = RUNTIME / "fixture_checks.log"
    if checks.exists():
        shutil.copy2(checks, audit / "fixture_checks.log")
    for name, source in manifest["sources"].items():
        for path in Path(source["directory"]).glob("LICENSE*"):
            shutil.copy2(path, audit / f"{name}_{path.name}")
    print("HookNet-TLS whole-slide inference complete", post_summary, flush=True)


if __name__ == "__main__":
    main()

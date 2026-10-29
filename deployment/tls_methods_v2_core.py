"""Reproducible TTA and phenotype-independent aggregate analysis.

Saved Cellpose objects are nuclear-size proxies, not verified nuclei. No
reference cell labels, calibrated cell accuracy, or TLS diagnosis are inferred.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/whole-slide-20261012"
OUT = RUN / "tls_methods_v2"
CLASSES = ["Other", "T", "Macrophage", "B"]
MPP = .252
GRID = 10.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(x) for x in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def load_source():
    manifest = json.loads((RUN / "run.json").read_text())
    raw = pd.read_csv(manifest["slides"][0]["results"])
    canonical = pd.read_csv(RUN / "tls_analysis/spatial_cells.csv")
    assert len(canonical) == 129470 and not canonical.global_instance_id.duplicated().any()
    raw = raw.set_index("global_instance_id").loc[canonical.global_instance_id].reset_index()
    np.testing.assert_allclose(raw[["centroid_x_global", "centroid_y_global"]].to_numpy()*MPP,
                               canonical[["x_um", "y_um"]], rtol=0, atol=1e-8)
    raw["x_um"] = canonical.x_um.to_numpy()
    raw["y_um"] = canonical.y_um.to_numpy()
    return raw, manifest


def d4(images):
    """The eight unique square symmetries, applied to all four channels."""
    sys.path.insert(0,str(ROOT/'script/src'))
    from prediction_tta import d4 as shared_d4
    return shared_d4(images)


def average_views(model, images):
    sys.path.insert(0,str(ROOT/'script/src'))
    from prediction_tta import average_views as shared_average
    return shared_average(model,images)


def run_tta(batch_size=128, force=False):
    import torch
    sys.path.insert(0, str(ROOT / "script/src"))
    from dataset import CellSamdataset_our
    from model_eval import SAM_ConvNet, get_transforms
    from runtime import load_classifier_state
    OUT.mkdir(parents=True, exist_ok=True)
    raw, manifest = load_source()
    fp = {"classifier_sha256": digest(manifest["classifier"]),
          "canonical_sha256": digest(RUN / "tls_analysis/spatial_cells.csv"),
          "views": "D4, mean softmax", "classes": CLASSES, "cells": len(raw)}
    done = OUT / "tta_summary.json"
    if done.exists() and not force:
        saved = json.loads(done.read_text())
        if saved["fingerprint"] != fp:
            raise ValueError("TTA cache provenance differs; use --force-tta")
        if (OUT / "tta_cells.csv").exists():
            print("Reusing verified whole-slide TTA", flush=True)
            return pd.read_csv(OUT / "tta_cells.csv")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU unavailable; run with local GPU access")
    torch.set_num_threads(8)
    model = SAM_ConvNet(num_classes=4, pretrained=False).eval()
    model.load_state_dict(load_classifier_state(manifest["classifier"]))
    model.to("mps")
    _, transform = get_transforms()
    dataset = CellSamdataset_our(raw, transform)
    progress_path = OUT / "tta_progress.json"
    array_path = OUT / "tta_view_probabilities.npy"
    start_index, previous_seconds = 0, 0.
    if progress_path.exists() and array_path.exists() and not force:
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] == fp:
            start_index, previous_seconds = progress["completed_cells"], progress["seconds"]
    mode = "r+" if start_index else "w+"
    views = np.lib.format.open_memmap(array_path, mode=mode, dtype="float32", shape=(len(raw), 8, 4))
    subset = torch.utils.data.Subset(dataset, range(start_index, len(raw)))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    started = time.perf_counter()
    index = start_index
    with torch.inference_mode():
        for step, (images, ids) in enumerate(loader):
            _, probabilities = average_views(model, images.to("mps"))
            n = len(images)
            np.testing.assert_array_equal(ids.numpy(), raw.global_instance_id.iloc[index:index+n])
            views[index:index+n] = probabilities.permute(1, 0, 2).cpu().numpy()
            index += n
            if step % 40 == 0 or index == len(raw):
                views.flush()
                save_json({"fingerprint": fp, "completed_cells": index,
                           "seconds": previous_seconds+time.perf_counter()-started}, progress_path)
                print(f"TTA: {index:,}/{len(raw):,} cells", flush=True)
    elapsed = previous_seconds+time.perf_counter()-started
    p = np.asarray(views).mean(axis=1)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    identity = np.asarray(views[:, 0])
    identity_delta = float(np.max(np.abs(identity-raw[["prob_"+c for c in CLASSES]].to_numpy())))
    identity_agreement = float((identity.argmax(1) == raw.prediction.to_numpy()).mean())
    aug_labels = np.asarray(views).argmax(-1)
    tta_labels = p.argmax(1)
    modal = np.stack([(aug_labels == c).sum(1) for c in range(4)], axis=1).max(1)
    frame = raw[["global_instance_id", "x_um", "y_um"]].copy()
    for name in ["celltype", "prediction", "confidence"]+["prob_"+c for c in CLASSES]:
        frame["raw_"+name] = raw[name].to_numpy()
    for i, c in enumerate(CLASSES):
        frame["prob_"+c] = p[:, i]
        frame["mean_view_prob_"+c] = p[:, i]
    frame["prediction"] = tta_labels
    frame["celltype"] = np.array(CLASSES)[tta_labels]
    frame["confidence"] = p.max(1)
    frame["normalized_entropy"] = -(p*np.log(np.maximum(p, 1e-12))).sum(1)/np.log(4)
    frame["orientation_disagreement"] = 1-modal/8
    frame["views_agree_with_tta_fraction"] = (aug_labels == tta_labels[:, None]).mean(1)
    frame["view_JS_divergence"] = (np.asarray(views)*np.log(np.maximum(views, 1e-12)/
                                              np.maximum(p[:, None], 1e-12))).sum(2).mean(1)
    frame["label_changed"] = frame.raw_prediction != frame.prediction
    frame.to_csv(OUT / "tta_cells.csv", index=False)
    transition = pd.crosstab(frame.raw_celltype, frame.celltype).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    transition.to_csv(OUT / "class_transitions.csv")
    # Direct model check: averaging a complete D4 orbit must be invariant to its starting view.
    check_ids = np.linspace(0, len(raw)-1, 8, dtype=int)
    sample = torch.stack([dataset[int(i)][0] for i in check_ids])
    with torch.inference_mode():
        base, _ = average_views(model, sample.to("mps"))
        deltas = []
        for transformed in d4(sample):
            v, _ = average_views(model, transformed.to("mps"))
            deltas.append(float((v-base).abs().max().cpu()))
        model.to("cpu")
        cpu, _ = average_views(model, sample)
    cpu_delta = float((cpu-base.cpu()).abs().max())
    assert max(deltas) < 2e-5 and cpu_delta < 2e-4
    save_json({"fingerprint": fp, "cells": len(frame), "raw_counts": frame.raw_celltype.value_counts().to_dict(),
               "tta_counts": frame.celltype.value_counts().to_dict(), "changed_cells": int(frame.label_changed.sum()),
               "seconds": elapsed, "identity_repeat_label_agreement": identity_agreement,
               "identity_repeat_max_probability_delta": identity_delta,
               "D4_invariance_max_probability_delta": max(deltas), "CPU_MPS_TTA_max_probability_delta": cpu_delta,
               "verification_sample_IDs": raw.global_instance_id.iloc[check_ids].tolist(),
               "class_order_source": "User confirmed exact checkpoint mapping on 2026-10-13",
               "checkpoint_class_metadata": "absent; four output weights; legacy dataset code has five classes",
               "interpretation": "Orientation-averaged predictions; no cell-label accuracy established"}, done)
    print("TTA complete:", frame.celltype.value_counts().to_dict(), flush=True)
    return frame


def extract_object_features(force=False):
    """Measure complete object masks and H signal, never the truncated classifier mask."""
    from PIL import Image
    import tifffile
    from skimage.measure import regionprops
    from skimage.color import rgb2hed
    path = OUT / "object_features.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    raw, manifest = load_source()
    rows = []
    started = time.perf_counter()
    for number, (mask_path, group) in enumerate(raw.groupby("patch_mask_path", sort=False)):
        mask = tifffile.imread(mask_path)
        rgb = np.asarray(Image.open(group.patch_path.iloc[0]).convert("RGB"))
        hematoxylin = rgb2hed(rgb)[..., 0]
        for c in group.itertuples():
            y0, y1 = c.bbox_ymin_local, c.bbox_ymax_local_exclusive
            x0, x1 = c.bbox_xmin_local, c.bbox_xmax_local_exclusive
            binary = mask[y0:y1, x0:x1] == c.local_label
            assert int(binary.sum()) == c.area_pixels
            prop = regionprops(np.pad(binary.astype("uint8"), 1))[0]
            rows.append({"global_instance_id": c.global_instance_id, "area_um2": c.area_pixels*MPP**2,
                         "equivalent_diameter_um": prop.equivalent_diameter_area*MPP,
                         "eccentricity": prop.eccentricity, "solidity": prop.solidity,
                         "perimeter_um": prop.perimeter*MPP,
                         "circularity": min(1., 4*np.pi*prop.area/max(prop.perimeter**2, 1)),
                         "hematoxylin_mean": float(hematoxylin[y0:y1, x0:x1][binary].mean()),
                         "crop_truncated": c.cell_mask_area_crop < c.area_pixels,
                         "crop_retained_fraction": c.cell_mask_area_crop/c.area_pixels,
                         "patch_id": c.patch_id, "patch_x": c.patch_x, "patch_y": c.patch_y,
                         "stride_px": c.stride, "core_edge_distance_um": min(
                             c.centroid_x_local-c.core_xmin_local,c.core_xmax_local-c.centroid_x_local,
                             c.centroid_y_local-c.core_ymin_local,c.core_ymax_local-c.centroid_y_local)*MPP})
        if number % 100 == 0:
            print(f"Full-mask features: {number+1} patches; {len(rows):,} cells", flush=True)
    frame = pd.DataFrame(rows).set_index("global_instance_id").loc[raw.global_instance_id].reset_index()
    assert len(frame) == len(raw)
    frame.to_csv(path, index=False)
    save_json({"cells": len(frame), "crop_truncated": int(frame.crop_truncated.sum()),
               "seconds": time.perf_counter()-started,
               "physical_classifier_crop_width_um": float(raw.cell_crop_size.median()*MPP),
               "mask_semantics": "Full official Cellpose-SAM object; nuclear-size proxy",
               "stain_method": "skimage rgb2hed fixed H&E deconvolution; mean H inside full object"}, OUT / "object_feature_summary.json")
    return frame


def tissue_grid(slide_path, spacing=GRID):
    from PIL import Image
    import openslide
    from skimage.filters import threshold_otsu
    with openslide.OpenSlide(str(slide_path)) as slide:
        mppx = float(slide.properties['openslide.mpp-x'])
        mppy = float(slide.properties['openslide.mpp-y'])
        if abs(mppx-mppy) > 1e-6:
            raise ValueError("Anisotropic pixels require explicit geometry handling")
        w, h = slide.dimensions
        nx, ny = int(np.ceil(w*mppx/spacing)), int(np.ceil(h*mppx/spacing))
        small = slide.get_thumbnail((nx, ny)).convert("RGB").resize((nx, ny), Image.Resampling.BILINEAR)
    gray = np.asarray(small).mean(2)
    tissue = gray < threshold_otsu(gray)
    tissue = ndi.binary_fill_holes(tissue)
    return tissue, (w*mppx, h*mppx), np.asarray(small)


def field_components(xy, areas, tissue, sigma_um=25, spacing=GRID):
    """All-object density segmentation. Labels are intentionally not accepted."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk, remove_small_holes
    from skimage.measure import regionprops
    ny, nx = tissue.shape
    idx = np.floor(np.asarray(xy)/spacing).astype(int)
    valid = (idx[:,0]>=0)&(idx[:,0]<nx)&(idx[:,1]>=0)&(idx[:,1]<ny)&np.isfinite(areas)&(areas>0)
    bins = np.zeros((ny, nx)); weighted = bins.copy()
    np.add.at(bins, (idx[valid, 1], idx[valid, 0]), 1)
    np.add.at(weighted, (idx[valid, 1], idx[valid, 0]), np.asarray(areas)[valid])
    counts = ndi.gaussian_filter(bins, sigma_um/spacing, mode="constant")
    coverage = ndi.gaussian_filter(tissue.astype(float), sigma_um/spacing, mode="constant")
    density = counts/np.maximum(coverage, .5)/spacing**2*1e6
    mean_area = ndi.gaussian_filter(weighted, sigma_um/spacing, mode="constant")/np.maximum(counts, 1e-12)
    domain = tissue & (coverage >= .5) & (counts > 0)
    if not np.any(domain) or not valid.any():
        labels = np.zeros(tissue.shape, dtype=int)
        return [], labels, density, mean_area, {"density_threshold_per_mm2": None, "area_threshold_um2": None}
    log_threshold = float(threshold_otsu(np.log1p(density[domain])))
    area_threshold = float(np.quantile(np.asarray(areas)[valid], .4))
    binary = domain & (np.log1p(density) > log_threshold) & (mean_area <= area_threshold)
    binary = binary_closing(binary, footprint=disk(max(1, round(10/spacing)))) & domain
    binary = remove_small_holes(binary, area_threshold=round(2500/spacing**2)+1, connectivity=2) & domain
    labels, _ = ndi.label(binary, structure=np.ones((3,3)))
    label_at_cell = np.zeros(len(xy), dtype=int)
    label_at_cell[valid] = labels[idx[valid,1],idx[valid,0]]
    records = []
    for prop in regionprops(labels):
        selected = np.flatnonzero(label_at_cell == prop.label)
        a = np.asarray(areas)[selected]
        area = prop.area*spacing**2
        median = float(np.median(a)) if len(a) else np.nan
        mad = float(np.median(np.abs(a-median))/median) if len(a) else np.nan
        record = {"native_label": int(prop.label), "sigma_um": sigma_um,
                  "area_um2": area, "equivalent_diameter_um": np.sqrt(4*area/np.pi),
                  "all_objects": len(selected), "density_objects_per_mm2": len(selected)/area*1e6,
                  "mean_smoothed_density_per_mm2": float(density[labels == prop.label].mean()),
                  "median_object_area_um2": median, "relative_area_MAD": mad,
                  "solidity": prop.solidity, "x_um": (prop.centroid[1]+.5)*spacing,
                  "y_um": (prop.centroid[0]+.5)*spacing,
                  "retained": bool(len(selected)>=50 and area>=5000), "members": selected}
        records.append(record)
    return records, labels, density, mean_area, {"density_threshold_per_mm2": np.expm1(log_threshold),
                "area_threshold_um2": area_threshold, "tissue_correction": "Gaussian tissue fraction >=0.5; density denominator clipped to 0.5",
                "spacing_um": spacing, "sigma_um": sigma_um, "closing_radius_um": 10,
                "max_filled_hole_um2": 2500, "minimum_objects": 50, "minimum_area_um2": 5000}


def label_polygon(labels, label, spacing=GRID):
    """Union raster row runs: exact area and holes on the physical grid."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    mask = labels == label
    rectangles = []
    for y in np.flatnonzero(mask.any(1)):
        padded = np.pad(mask[y].astype(int), (1,1))
        starts = np.flatnonzero(np.diff(padded)==1)
        stops = np.flatnonzero(np.diff(padded)==-1)
        rectangles.extend(box(x0*spacing,y*spacing,x1*spacing,(y+1)*spacing) for x0,x1 in zip(starts,stops))
    return unary_union(rectangles)


def run_components(cells, features):
    from shapely.geometry import mapping
    _, manifest = load_source()
    tissue, dimensions, thumbnail = tissue_grid(manifest["slides"][0]["slide"])
    xy = cells[["x_um","y_um"]].to_numpy()
    areas = features.set_index("global_instance_id").loc[cells.global_instance_id].area_um2.to_numpy()
    all_rows, native_geoms, baseline_geoms, membership, settings = [], {}, {}, [], []
    for sigma in [25,20,35]:
        records, labels, density, mean_area, parameters = field_components(xy,areas,tissue,sigma)
        frame = pd.DataFrame([{k:v for k,v in r.items() if k != 'members'} for r in records])
        if frame.empty:
            frame = pd.DataFrame(columns=['native_label','sigma_um','area_um2','equivalent_diameter_um','all_objects',
                 'density_objects_per_mm2','mean_smoothed_density_per_mm2','median_object_area_um2','relative_area_MAD',
                 'solidity','x_um','y_um','retained'])
        frame['morphology_score'] = np.nan
        retained = frame.retained.astype(bool)
        if retained.any():
            score = (frame.loc[retained,'density_objects_per_mm2'].rank(pct=True)+
                     frame.loc[retained,'median_object_area_um2'].rank(pct=True,ascending=False)+
                     frame.loc[retained,'relative_area_MAD'].rank(pct=True,ascending=False)+
                     frame.loc[retained,'solidity'].rank(pct=True))/4
            frame.loc[retained,'morphology_score'] = score
        frame = frame.sort_values(['retained','morphology_score','native_label'],ascending=[False,False,True])
        frame['region_id'] = [f"R{i+1}" if sigma==25 else f"s{sigma}_R{i+1}" for i in range(len(frame))]
        by_label = {int(r.native_label):r for r in frame.itertuples()}
        for record in records:
            r = by_label[record['native_label']]
            selected = record['members']
            local = cells.iloc[selected]
            pos = frame.index[frame.native_label.eq(r.native_label)][0]
            for c in CLASSES:
                frame.loc[pos,'tta_'+c] = int(local.celltype.eq(c).sum())
            frame.loc[pos,'tta_BT_fraction_all_objects'] = float(local.celltype.isin(['B','T']).mean()) if len(local) else np.nan
            if r.retained:
                geom = label_polygon(labels,r.native_label)
                np.testing.assert_allclose(geom.area,r.area_um2)
                native_geoms[r.region_id] = geom
                if sigma==25:
                    baseline_geoms[r.region_id] = geom
                    membership.extend((r.region_id,int(c.global_instance_id),c.celltype) for c in local.itertuples())
        frame.to_csv(OUT / f"components_sigma{sigma}.csv",index=False)
        all_rows.append(frame)
        settings.append({**parameters,'retained_components':int(retained.sum()),'rejected_components':int((~retained).sum())})
        np.savez_compressed(OUT / f"fields_sigma{sigma}.npz",labels=labels,density=density,mean_area=mean_area,tissue=tissue)
        print(f"Morphology σ={sigma}: {int(retained.sum())} retained components",flush=True)
    baseline = all_rows[0]
    baseline.to_csv(OUT/'components.csv',index=False)
    sensitivity=[]
    base_labels=np.load(OUT/'fields_sigma25.npz')['labels']
    for sigma,other in zip([20,35],all_rows[1:]):
        other_labels=np.load(OUT/f'fields_sigma{sigma}.npz')['labels']
        for b in baseline[baseline.retained].itertuples():
            selected=base_labels==b.native_label
            native_ids,counts=np.unique(other_labels[selected],return_counts=True)
            for label,count in zip(native_ids,counts):
                target=other[(other.native_label==label)&other.retained]
                if target.empty: continue
                r=target.iloc[0];area=int(count)*GRID**2
                sensitivity.append({'baseline_region_id':b.region_id,'sigma_um':sigma,'sensitivity_region_id':r.region_id,
                    'intersection_area_um2':area,'baseline_area_coverage':area/b.area_um2,
                    'grid_polygon_IoU':area/(b.area_um2+r.area_um2-area)})
    pd.DataFrame(sensitivity,columns=['baseline_region_id','sigma_um','sensitivity_region_id','intersection_area_um2',
              'baseline_area_coverage','grid_polygon_IoU']).to_csv(OUT/'component_sensitivity_pairs.csv',index=False)
    pd.DataFrame(membership,columns=['region_id','global_instance_id','tta_celltype']).to_csv(OUT/'component_membership.csv.gz',index=False)
    save_json({'type':'FeatureCollection','coordinate_units':'micrometers','features':[
        {'type':'Feature','geometry':mapping(g),'properties':{'region_id':k,'sigma_um':25,'geometry_basis':'10 µm grid; exact raster union'}}
        for k,g in baseline_geoms.items()]},OUT/'components.geojson')
    save_json({'settings':settings,'ranking':'Equal mean percentile ranks: density (+), median object area (-), relative area MAD (-), solidity (+)',
               'labels_used_for_selection_or_ranking':False,'dimensions_um':dimensions,
               'limitation':'Segmentation objects are unvalidated nuclear proxies; morphology candidates are not confirmed TLS'},OUT/'component_summary.json')
    return baseline, baseline_geoms, tissue, thumbnail


def run_tools(cells):
    from tls_tools import load_finder, finder_input, r_input
    from shapely.geometry import Point
    tool_dir = OUT/'tools'; tool_dir.mkdir(exist_ok=True)
    adapted, _ = load_finder(tool_dir/'adapter')
    frame = finder_input(cells)
    start = time.perf_counter(); native = adapted(frame,160); seconds = time.perf_counter()-start
    native.to_csv(tool_dir/'finder_default.csv',index=False)
    geometries, records, memberships = {}, [], []
    xy = cells[['x_um','y_um']].to_numpy()
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    for i,row in enumerate(native.itertuples(index=False),1):
        center = row.Aggregate_Center_Coordinates
        key = f'finder_{i}'
        geometries[key] = Point(*center).buffer(160,resolution=128)
        selected = tree.query_ball_point(center,160)
        local = cells.iloc[selected]
        bt = local[local.celltype.isin(['B','T'])]
        memberships.extend((key,int(c.global_instance_id),c.celltype) for c in bt.itertuples())
        records.append({'region_id':key,'method':'TLS-Finder','run_id':'finder_default','x_um':center[0],
                        'y_um':center[1],'B':int(local.celltype.eq('B').sum()),'T':int(local.celltype.eq('T').sum()),
                        'membership_basis':'Geometrically derived 160 µm disk'})
    settings = [{'method':'TLS-Finder','run_id':'finder_default','detections':len(native),'seconds':seconds,'radius_um':160,'min_BT':50}]
    r_input(cells).to_csv(tool_dir/'tlsr_input.csv',index=False)
    jobs = pd.DataFrame([{'run_id':f'tlsr_{t}','threshold':t,'variant':'all','input_csv':str(tool_dir/'tlsr_input.csv')} for t in [10,40,50]])
    jobs.to_csv(tool_dir/'tlsr_jobs.csv',index=False)
    manifest = json.loads((ROOT/'.runtime/tls_tools/manifest.json').read_text())
    env = os.environ.copy(); env['R_LIBS_USER']=manifest['R_library']
    env['PATH']=str(Path(manifest['Rscript']).parent)+os.pathsep+env.get('PATH','')
    with (tool_dir/'R_execution.log').open('w') as log:
        subprocess.run([manifest['Rscript'],str(ROOT/'deployment/run_tlsr.R'),str(tool_dir/'tlsr_jobs.csv'),str(tool_dir)],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics = pd.read_csv(tool_dir/'tlsr_metrics.csv')
    for metric in metrics.itertuples():
        regions = pd.read_csv(tool_dir/f'{metric.run_id}_regions.csv')
        members = pd.read_csv(tool_dir/f'{metric.run_id}_cells.csv.gz')
        np.testing.assert_array_equal(np.sort(members.global_instance_id),np.sort(cells.global_instance_id))
        for r in regions.itertuples():
            key = f'{metric.run_id}_{r.native_id}'
            local = members[(members.tls_id_knn==r.native_id)&members.phenotype.isin(['B cell','T cell'])]
            memberships.extend((key,int(c.global_instance_id),c.source_celltype) for c in local.itertuples())
            records.append({'region_id':key,'method':'tlsR','run_id':metric.run_id,'x_um':r.x_um,'y_um':r.y_um,
               'B':int(local.phenotype.eq('B cell').sum()),'T':int(local.phenotype.eq('T cell').sum()),'icat':r.icat,
               'membership_basis':'Native returned tls_id_knn; no invented polygon'})
        settings.append({'method':'tlsR','run_id':metric.run_id,'detections':metric.detections,
            'dense_B':metric.dense_B,'seconds':metric.detection_seconds,'density_ceiling_um':metric.threshold,
            'k':30,'min_B':50,'min_T':10,'T_distance_um':50,'expansion_um':80})
    pd.DataFrame(records).to_csv(tool_dir/'regions.csv',index=False)
    pd.DataFrame(memberships,columns=['region_id','global_instance_id','tta_celltype']).to_csv(tool_dir/'BT_membership.csv.gz',index=False)
    save_json({'runs':settings,'sources':manifest['sources'],'R_environment':manifest['R_environment']},tool_dir/'summary.json')
    return records,geometries,memberships


def hook_regions(cells):
    from shapely.geometry import shape
    from shapely.affinity import scale
    from shapely import contains_xy
    path = RUN/'tls_tool_comparison/hooknet_tls/native_polygons.geojson'
    features = json.loads(path.read_text())['features']
    records, geometries, memberships, gc = [], {}, [], {}
    for f in features:
        p=f['properties']
        if not p['passes_published_filter']: continue
        g=scale(shape(f['geometry']),xfact=.5,yfact=.5,origin=(0,0))
        if p['predicted_label']!='tls': gc[p['region_id']]=g; continue
        key=p['region_id']; geometries[key]=g
        local=cells[contains_xy(g,cells.x_um.to_numpy(),cells.y_um.to_numpy())]
        bt=local[local.celltype.isin(['B','T'])]
        memberships.extend((key,int(c.global_instance_id),c.celltype) for c in bt.itertuples())
        records.append({'region_id':key,'method':'HookNet-TLS','run_id':'hooknet_cached','x_um':g.centroid.x,
            'y_um':g.centroid.y,'area_um2':g.area,'B':int(local.celltype.eq('B').sum()),'T':int(local.celltype.eq('T').sum()),
            'membership_basis':'TTA cell centers in reused native H&E polygon'})
    pd.DataFrame(records).to_csv(OUT/'hooknet_regions_TTA.csv',index=False)
    save_json({'native_polygons_sha256':digest(path),'reused_H_and_E_predictions':True,'TLS_regions':len(records),
               'GC_regions':len(gc),'GC_maturity_claims':False},OUT/'hooknet_reuse.json')
    return records,geometries,memberships,gc


def compare_regions(cells,components,component_geoms,tool_records,tool_geoms,tool_members,hook_records,hook_geoms,hook_members):
    from shapely.geometry import Point
    from shapely import contains_xy
    candidates=pd.read_csv(RUN/'tls_analysis/ranked_candidate_regions.csv')
    region_info={r['region_id']:r for r in tool_records+hook_records}
    geometries={**tool_geoms,**hook_geoms,**component_geoms}
    for r in components[components.retained].to_dict('records'):
        region_info[r['region_id']]={**r,'method':'Morphology','run_id':'sigma25'}
    member_sets={key:set() for key in region_info}
    for key,cid,_ in tool_members+hook_members: member_sets[key].add(cid)
    all_component_members=pd.read_csv(OUT/'component_membership.csv.gz')
    for key,g in all_component_members.groupby('region_id'):
        member_sets[key]=set(map(int,g.loc[g.tta_celltype.isin(['B','T']),'global_instance_id']))
    pairs=[]; summaries=[]; tool_summaries=[]
    for a in candidates.itertuples():
        point=Point(a.x_um,a.y_um); disk=point.buffer(150,resolution=128)
        inside=((cells.x_um-a.x_um)**2+(cells.y_um-a.y_um)**2)<=150**2
        ids=set(map(int,cells.loc[inside & cells.celltype.isin(['B','T']),'global_instance_id']))
        all_ids=set(map(int,cells.loc[inside,'global_instance_id']))
        local=[]
        for key,r in region_info.items():
            g=geometries.get(key); overlap=disk.intersection(g).area if g is not None else np.nan
            common=len(ids & member_sets[key]); union=len(ids | member_sets[key])
            distance=point.distance(Point(r['x_um'],r['y_um']))
            row={'candidate_id':a.candidate_id,'region_id':key,'method':r['method'],'run_id':r['run_id'],
                'center_distance_um':distance,'legacy_center_match':distance<=150,
                'intersection_BT':common,'candidate_BT_coverage':common/len(ids) if ids else 0.,
                'BT_jaccard':common/union if union else 0.,'intersection_area_um2':overlap,
                'candidate_disk_area_coverage':overlap/disk.area if g is not None else np.nan,
                'disk_region_IoU':overlap/disk.union(g).area if g is not None else np.nan,
                'membership_basis':r.get('membership_basis','Grid component cell-center membership')}
            if r['method']=='Morphology':
                members=set(map(int,all_component_members.loc[all_component_members.region_id.eq(key),'global_instance_id']))
                row['candidate_all_objects_coverage']=len(members & all_ids)/len(all_ids) if all_ids else 0
            pairs.append(row);local.append(row)
        hits=[r for r in local if r['method']=='Morphology' and r['intersection_area_um2']>0]
        summaries.append({'candidate_id':a.candidate_id,'raw_B':a.B_150um,'raw_T':a.T_150um,
            'TTA_B':int((inside & cells.celltype.eq('B')).sum()),'TTA_T':int((inside & cells.celltype.eq('T')).sum()),
            'TTA_Other':int((inside & cells.celltype.eq('Other')).sum()),'all_objects':int(inside.sum()),
            'morphology_regions':','.join(r['region_id'] for r in hits),'split_into_multiple_components':len(hits)>1,
            'morphology_disk_area_coverage':sum(r['candidate_disk_area_coverage'] for r in hits),
            'morphology_BT_coverage':len(ids & set().union(*(member_sets[r['region_id']] for r in hits)))/len(ids) if hits and ids else 0})
        expected=[('Morphology','sigma25'),('TLS-Finder','finder_default'),('tlsR','tlsr_10'),
                  ('tlsR','tlsr_40'),('tlsR','tlsr_50'),('HookNet-TLS','hooknet_cached')]
        for method,run_id in expected:
            group=[r for r in local if r['method']==method and r['run_id']==run_id]
            matched=[r for r in group if r['legacy_center_match']]
            supported=[r for r in group if r['intersection_BT'] or (np.isfinite(r['intersection_area_um2']) and r['intersection_area_um2']>0)]
            covered=set().union(*(member_sets[r['region_id']] for r in group)) if group else set()
            best=max(group,key=lambda r:r['BT_jaccard']) if group else None
            tool_summaries.append({'candidate_id':a.candidate_id,'method':method,'run_id':run_id,
                'total_detections':len(group),'centers_in_original_disk':len(matched),'overlapping_detections':len(supported),
                'all_detection_BT_coverage':len(ids & covered)/len(ids) if ids else 0,
                'best_BT_Jaccard':best['BT_jaccard'] if best else 0,
                'best_disk_area_coverage':max((r['candidate_disk_area_coverage'] for r in group if np.isfinite(r['candidate_disk_area_coverage'])),default=np.nan),
                'nearest_center_distance_um':min((r['center_distance_um'] for r in group),default=np.nan),
                'split_location_match':len(matched)>1})
    comparisons=pd.DataFrame(pairs);comparisons.to_csv(OUT/'A1_A6_all_region_pairs.csv',index=False)
    summary=pd.DataFrame(summaries)
    summary['shares_component_with_other_candidate']=summary.morphology_regions.map(lambda s:any(
        k and sum(k in x.split(',') for x in summary.morphology_regions)>1 for k in s.split(',')))
    summary.to_csv(OUT/'A1_A6_summary.csv',index=False)
    pd.DataFrame(tool_summaries).to_csv(OUT/'A1_A6_tool_summary.csv',index=False)
    rows=[]
    for key,g in component_geoms.items():
        for target,h in {**tool_geoms,**hook_geoms}.items():
            area=g.intersection(h).area
            if area<=0: continue
            rows.append({'component_id':key,'tool_region_id':target,'method':region_info[target]['method'],
                'area_intersection_um2':area,'component_area_coverage':area/g.area,'polygon_IoU':area/g.union(h).area})
    pd.DataFrame(rows,columns=['component_id','tool_region_id','method','area_intersection_um2','component_area_coverage','polygon_IoU']).to_csv(OUT/'component_tool_polygon_pairs.csv',index=False)
    rows=[]
    for key in component_geoms:
        for target,r in region_info.items():
            if r['method']=='Morphology': continue
            a,b=member_sets[key],member_sets[target]; common=len(a&b)
            if common:
                rows.append({'component_id':key,'tool_region_id':target,'method':r['method'],'run_id':r['run_id'],
                    'intersection_BT':common,'component_BT_coverage':common/len(a) if a else 0,'BT_jaccard':common/len(a|b)})
    pd.DataFrame(rows,columns=['component_id','tool_region_id','method','run_id','intersection_BT','component_BT_coverage','BT_jaccard']).to_csv(OUT/'component_tool_membership_pairs.csv',index=False)
    return summary,comparisons

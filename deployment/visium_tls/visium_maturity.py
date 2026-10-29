"""GC morphology features (plan item 5a) on the USZ Visium pathologist TLS objects.

Object = connected group of TLS-labelled spots (visium_eval.tls_objects); mature if any MTLS or GC
spot, early if ETLS only. The aggregate measured for an object is the frozen stage-1 nuclear-density
proposal(s) overlapping the object's spot tiles (55 µm discs), clipped to the tiles dilated by 50 µm,
so one large proposal spanning several TLS does not give them all the same features. Objects with no
overlapping proposal fall back to the spot tiles themselves (`aggregate_source`). A sensitivity variant
measures every object on its spot tiles (`region == "spot tiles"`).

Images are used at native resolution (~0.79 µm/px, no upsampling) for the OD map; nuclear areas come
from the 0.25 µm/px upsampled segmentation. Cell types: fine-tuned classifier + tau 0.5 prior, as for
ESCA. Feature code is shared with esca_maturity.py; nothing here is fitted.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import esca_maturity as M  # noqa: E402  (sets the ESCA classifier before esca_analysis is imported)
import esca_analysis as A  # noqa: E402
import visium_cells as V  # noqa: E402
import visium_eval as E  # noqa: E402

CLASSIFIER = "finetuned_tau05"
TILE_UM, NEIGHBOURHOOD_UM = 55.0, 50.0


def disc_mask(spots, mpp, shape, radius_um):
    mask = np.zeros(shape, bool)
    gy, gx = np.mgrid[:shape[0], :shape[1]]
    for s in spots.itertuples():
        cy, cx = s.y * mpp / A.GRID_UM, s.x * mpp / A.GRID_UM
        r = radius_um / A.GRID_UM
        y0, y1 = int(max(cy - r - 1, 0)), int(min(cy + r + 2, shape[0]))
        x0, x1 = int(max(cx - r - 1, 0)), int(min(cx + r + 2, shape[1]))
        mask[y0:y1, x0:x1] |= (gy[y0:y1, x0:x1] + .5 - cy) ** 2 + (gx[y0:y1, x0:x1] + .5 - cx) ** 2 <= r * r
    return mask


def measure(image, mpp, mask8, x, cells, keep_geometry=False):
    filled8 = ndimage.binary_fill_holes(mask8)
    box = M.crop_box(filled8)
    px = [int(max(np.floor(v / mpp), 0)) for v in box]
    rgb = image[px[1]:min(int(np.ceil(box[3] / mpp)), image.shape[0]), px[0]:min(int(np.ceil(box[2] / mpp)), image.shape[1])]
    x_um0, y_um0 = px[0] * mpp, px[1] * mpp
    filled = M.grid_to_crop(filled8, x_um0, y_um0, rgb.shape[:2], mpp)
    xy = np.stack([cells.x_native.to_numpy() - px[0], cells.y_native.to_numpy() - px[1]], 1)
    near = (xy[:, 0] >= -1) & (xy[:, 1] >= -1) & (xy[:, 0] <= rgb.shape[1]) & (xy[:, 1] <= rgb.shape[0])
    values, geo = M.gc_features_all(rgb, mpp, filled, xy[near], cells.area_um2.to_numpy()[near],
                                cells[f"celltype_{CLASSIFIER}"].to_numpy()[near])
    values = {**M.composition(mask8, x["counts"], x["valid"]), "filled_hole_um2": float((filled8 & ~mask8).sum() * A.GRID_UM ** 2),
              "image_mpp": mpp, **values}
    return (values, {"rgb": rgb.copy(), "filled": filled, **(geo or {})}) if keep_geometry else (values, None)


def sample_objects(sample, keep=None):
    """Feature rows for every pathologist TLS object of one section (and gallery geometry for `keep`)."""
    x = E.sample_inputs(sample, CLASSIFIER)
    shape, mpp, cells = x["shape"], x["mpp"], x["cells"]
    x.update({"gc": np.zeros(shape, bool), "reference": np.zeros(shape, bool)})
    proposals, _ = A.proposal_features(x, **M.STAGE1)
    A._PROPOSALS.clear()
    image = tifffile.imread(V.DATA / "tif_slides" / f"{sample}.tif")[..., :3]
    spots = pd.read_csv(V.OUT / "analysis" / "spot_table.csv.gz") if not hasattr(sample_objects, "table") else sample_objects.table
    sample_objects.table = spots
    part = spots[(spots["sample"] == sample) & (spots.classifier == CLASSIFIER) & (spots.in_tissue == 1)].reset_index(drop=True)
    groups = E.tls_objects(part, 100 / mpp)
    rows, geometry = [], {}
    for g in sorted(groups.unique()):
        members = part.loc[groups.index[groups == g]]
        labels = members.label.value_counts()
        tiles = disc_mask(members, mpp, shape, TILE_UM)
        near = ndimage.distance_transform_edt(~tiles) * A.GRID_UM <= NEIGHBOURHOOD_UM
        overlapping = [p for p in proposals
                       if (p["mask"] & tiles).sum() >= A.MATCH_FRACTION * min(p["mask"].sum(), tiles.sum())]
        aggregate = np.zeros(shape, bool)
        for p in overlapping:
            aggregate |= p["mask"]
        aggregate &= near
        source = "stage-1 proposal" if aggregate.sum() * A.GRID_UM ** 2 >= 2000 else "spot tiles"
        base = {"sample": sample, "object": int(g), "spots": len(members),
                "labels": ",".join(f"{k}:{v}" for k, v in labels.items()),
                "mature": bool(labels.reindex(["MTLS", "GC"]).fillna(0).sum() > 0),
                "has_GC_spot": bool(labels.get("GC", 0) > 0),
                "mature_majority_rule": bool(labels.reindex(["MTLS", "GC"]).fillna(0).sum() >= labels.get("ETLS", 0)),
                "proposed": bool(overlapping),
                "accepted_two_stage": any(p["B_share"] >= M.FROZEN["b_share"] for p in overlapping),
                "B_fraction_spots": members.B.sum() / max(members.cells.sum(), 1),
                "B_density_spots": members.B_density_at_spot.mean()}
        for region, mask in [("aggregate", aggregate if source == "stage-1 proposal" else tiles), ("spot tiles", tiles)]:
            values, geo = measure(image, mpp, mask, x, cells, keep_geometry=keep is not None and (sample, int(g)) in keep and region == "aggregate")
            rows.append({**base, "region": region, "aggregate_source": source if region == "aggregate" else "spot tiles", **values})
            if geo is not None:
                geometry[(sample, int(g))] = geo
    return (rows, geometry) if keep is not None else rows


def main():
    rows = []
    for sample in V.SAMPLES:
        part = sample_objects(sample)
        print(sample, len(part) // 2, "objects", flush=True)
        rows += part
    frame = pd.DataFrame(rows)
    M.MATURITY.mkdir(parents=True, exist_ok=True)
    frame.to_csv(M.MATURITY / "visium_object_features.csv", index=False)
    main_rows = frame[frame.region == "aggregate"]
    print(main_rows.groupby("sample").mature.agg(["size", "sum"]), "\n", main_rows.aggregate_source.value_counts())


if __name__ == "__main__":
    main()

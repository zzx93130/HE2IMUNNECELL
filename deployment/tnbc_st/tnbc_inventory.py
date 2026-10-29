"""Stage 0 inventory: arrays <-> samples <-> patients, annotations, IHC slides, spot tables, ST count matrices.

Keys are the ids.RDS array ids (CN<slide>_<subarray>), mapped to `TNBC<sample>_CN<slide>_<subarray>` image names.
Two image-name inconsistencies in the Zenodo tar are resolved here (see `name_fixes`): imagesHD
`TNBC24_CN12_C1` is ids.RDS CN12_C1 = sample 23 (imagesLarge `TNBC23_CN12_C1`; thumbnails correlate r = 0.9998),
and imagesHD `TNBC70_CN35_D2` (truncated JPEG, no ids.RDS row, different tissue from imagesLarge
`TNBC64_CN32_D2`) is excluded; CN32_D2 (sample 64) has no imagesHD and is excluded from HE2CELL.
Writes outputs/tnbc_st_20261101/inventory.json.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402

BYARRAY = C.EXTRAS / "byArray" / "byArray"
RAW = C.EXTRAS / "rawCountsMatrices"


def main():
    index = C.tar_index()
    he = index["HE"]
    ids = pd.read_csv(C.OUT / "ids.csv", index_col=0)
    clinical = pd.read_csv(C.OUT / "clinical.csv")  # Clinical.xlsx sheet 1, exported with the .runtime/rdata venv
    hd = {Path(n).stem: n for n in he if n.startswith("Images/imagesHD/") and n.endswith(".jpg")}
    large = {Path(n).stem: n for n in he if n.startswith("Images/imagesLarge/") and n.endswith(".jpg")}
    annotations = {Path(n).stem: n for n in he if n.startswith("Images/imageAnnotations/") and n.endswith(".png")}
    artefact_files = sorted(Path(n).name for n in he if n.startswith("Images/artefacts/") and not Path(n).name.startswith("."))
    fixes = {"Images/imagesHD/TNBC24_CN12_C1.jpg": "CN12_C1 (sample 23); misnamed TNBC24 in imagesHD",
             "Images/imagesHD/TNBC70_CN35_D2.jpg": "excluded: truncated JPEG, no ids.RDS row, not the tissue of imagesLarge TNBC64_CN32_D2",
             "Images/imagesLarge/TNBC64_CN32_D2.jpg": "CN32_D2 (sample 64) has no imagesHD: excluded from HE2CELL"}
    hd_by_array = {}
    for stem, name in hd.items():
        m = C.ARRAY_RE.fullmatch(stem)
        hd_by_array.setdefault(f"CN{m.group(2)}_{m.group(3)}", []).append(name)
    large_by_array = {f"CN{C.ARRAY_RE.fullmatch(s).group(2)}_{C.ARRAY_RE.fullmatch(s).group(3)}": n for s, n in large.items()}
    ann_by_array = {f"CN{C.ARRAY_RE.fullmatch(s).group(2)}_{C.ARRAY_RE.fullmatch(s).group(3)}": n for s, n in annotations.items()}
    raw_files = {Path(p).name: p for p in glob.glob(str(RAW / "**" / "*.tsv*"), recursive=True)}
    ihc = C.ihc_slides()
    arrays = {}
    for array, row in ids.iterrows():
        sample = int(row["id"])
        hd_names = [n for n in hd_by_array.get(array, []) if "TNBC70_CN35_D2" not in n]
        entry = {"array": array, "sample": sample, "patient": f"TNBC{sample}", "st_slide": row["slide"], "subarray": row["subslide"],
                 "hd": hd_names[0] if hd_names else None, "large": large_by_array.get(array),
                 "annotation": ann_by_array.get(array), "ids_hasAnnot": bool(row["hasAnnot"]),
                 "count_matrix": row["count"], "count_matrix_present": row["count"] in raw_files,
                 "ihc_member": ihc.get(sample, {}).get("member"), "ihc_label_CD2": ihc.get(sample, {}).get("label_CD2")}
        if entry["hd"]:
            entry["image_name"] = Path(entry["hd"]).stem
        folder = BYARRAY / str(row["slide"]) / str(row["subslide"])
        spot_all = sorted(folder.glob("spot_data-all*.tsv"))
        spot_sel = sorted(folder.glob("spot_data-selection*.tsv"))
        entry["byArray_folder"] = str(folder.relative_to(C.ROOT)) if folder.exists() else None
        entry["byArray_files"] = sorted(p.name for p in folder.glob("*")) if folder.exists() else []
        if spot_all:
            spots = pd.read_csv(spot_all[0], sep="\t")
            entry["spot_table_all"] = str(spot_all[0].relative_to(C.ROOT))
            entry["spot_table_selection"] = str(spot_sel[0].relative_to(C.ROOT)) if spot_sel else None
            entry["spots_all"] = int(len(spots))
            entry["spots_selected"] = int(spots.selected.sum()) if "selected" in spots else None
            margin = 200.0 / 0.904  # 200 µm in imagesLarge px (nominal scale; frame only)
            entry["spots_frame_large"] = [float(spots.pixel_x.min() - margin), float(spots.pixel_y.min() - margin),
                                          float(spots.pixel_x.max() + margin), float(spots.pixel_y.max() + margin)]
            entry["spot_frame"] = "imagesLarge (pixel_x, pixel_y), checked by overlay on CN48_E1"
        entry["annotBySpot"] = (folder / "annotBySpot.RDS").exists()
        arrays[array] = entry
    # Annotated arrays: the one per sample with an annotation PNG.
    for e in arrays.values():
        e["annotated"] = e["annotation"] is not None and e["hd"] is not None
        e["register"] = e["annotated"] and e["ihc_member"] is not None
    samples = sorted({e["sample"] for e in arrays.values()})
    summary = {
        "arrays_in_ids": len(ids), "arrays_with_hd": sum(e["hd"] is not None for e in arrays.values()),
        "samples": len(samples), "patients": len(samples),
        "patient_key": "sample id (TNBC<n>, ids.RDS `id` = Clinical ST_TNBC_ID, 94 rows, one per patient); CN<k> is the ST glass slide, which carries two patients (3 subarrays each), so CN is NOT the patient",
        "annotated_arrays": sum(e["annotated"] for e in arrays.values()),
        "annotation_pngs": len(annotations),
        "ihc_slides": len(ihc), "ihc_samples_with_CD2_label": [s for s, v in ihc.items() if v["label_CD2"]],
        "samples_with_he_and_ihc": len(set(ihc) & set(samples)),
        "annotated_arrays_with_ihc": sum(e["register"] for e in arrays.values()),
        "arrays_with_spot_table": sum("spot_table_all" in e for e in arrays.values()),
        "arrays_with_count_matrix": sum(e["count_matrix_present"] for e in arrays.values()),
        "arrays_with_spot_table_and_counts_and_hd": sum("spot_table_all" in e and e["count_matrix_present"] and e["hd"] is not None
                                                        for e in arrays.values()),
        "annotated_arrays_with_spot_table_and_counts": sum("spot_table_all" in e and e["count_matrix_present"] and e["annotated"]
                                                           for e in arrays.values()),
        "qupath_artefact_files": len(artefact_files),
        "qupath_artefacts_used": False,
        "name_fixes": fixes,
        "tar_members_used": {"imagesHD": len(hd), "imagesLarge": len(large), "imageAnnotations": len(annotations),
                             "artefacts_qpdata_not_used": len(artefact_files), "ihc_ndpi": len(ihc)},
    }
    C.save_json({"summary": summary, "arrays": arrays, "ihc": {str(k): v for k, v in ihc.items()},
                 "clinical_columns": list(clinical.columns)}, C.OUT / "inventory.json")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

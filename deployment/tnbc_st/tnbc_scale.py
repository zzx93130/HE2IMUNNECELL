"""H&E pixel size of the TNBC_ST images -> outputs/tnbc_st_20261101/scale.json.

Estimates
1. Registration to the IHC NDPI (Hamamatsu, 40x, 0.2283 µm/px from the scanner metadata): implied imagesLarge µm/px
   (sqrt |det| of the accepted affine, x NDPI µm/px) for every accepted H&E-IHC pair; median and IQR.
2. ST spot lattice: least-squares pixel_x/pixel_y vs array x/y over every byArray spot table (imagesLarge frame;
   the byArray HE jpgs are byte-identical to imagesLarge). The arrays are a 66 x 63 checkerboard (~2,070 spots),
   not the 33 x 35 legacy layout, so the lattice pitch is not documented here; the estimate is reported for the
   assumption "array unit = 100 µm" (in-line spot pitch 200 µm) on each axis.
3. imagesHD / imagesLarge = 31744 / 9523 (fixed ratio).
4. Isotropy: spectral second-moment ratio of H&E tissue patches (x vs y) on a few arrays.
5. Nucleus-area sanity check: median HE2CELL T-cell nucleus area vs TCGA-ESCA (0.25 µm/px native).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402

HD, LARGE = 31744, 9523


def main(final=False):
    reg = []
    for f in sorted((C.OUT / "registration").glob("*.json")):
        d = json.loads(f.read_text())
        reg.append({"array": d["array"], "accept": d["accept"], "large_mpp": d["implied_large_mpp"],
                    "x_axis": d["implied_large_mpp_x_axis"], "y_axis": d["implied_large_mpp_y_axis"],
                    "median_residual_um": d["median_residual_um"]})
    reg = pd.DataFrame(reg)
    acc = reg[reg.accept]
    lattice = []
    for f in glob.glob(str(C.EXTRAS / "byArray" / "byArray" / "*" / "*" / "spot_data-all*.tsv")):
        d = pd.read_csv(f, sep="\t")
        A = np.c_[d.x, d.y, np.ones(len(d))]
        cx = np.linalg.lstsq(A, d.pixel_x, rcond=None)[0]
        cy = np.linalg.lstsq(A, d.pixel_y, rcond=None)[0]
        lattice.append({"x_unit_px": cx[0], "y_unit_px": cy[1]})
    lattice = pd.DataFrame(lattice)
    x_unit, y_unit = lattice.x_unit_px.median(), lattice.y_unit_px.median()
    large = float(acc.large_mpp.median())
    out = {
        "status": "FINAL (Gate A)" if final else "PRELIMINARY",
        "large_mpp": large, "hd_over_large": HD / LARGE, "hd_mpp": large * LARGE / HD,
        "registration": {"pairs_accepted": int(len(acc)), "pairs_total": int(len(reg)),
                         "large_mpp_median": large, "large_mpp_IQR": acc.large_mpp.quantile([.25, .75]).tolist(),
                         "x_axis_median": float(acc.x_axis.median()), "y_axis_median": float(acc.y_axis.median()),
                         "ndpi_mpp": 0.22834699609526637},
        "spot_lattice": {"arrays": int(len(lattice)), "x_unit_large_px_median": float(x_unit), "y_unit_large_px_median": float(y_unit),
                         "x_unit_sd": float(lattice.x_unit_px.std()), "y_unit_sd": float(lattice.y_unit_px.std()),
                         "large_mpp_if_x_unit_100um": 100 / x_unit, "large_mpp_if_y_unit_100um": 100 / y_unit,
                         "large_mpp_if_isotropic_100um_geometric_mean": 100 / np.sqrt(x_unit * y_unit),
                         "note": "66 x 63 checkerboard lattice (~2,070 spots); y unit / x unit = %.3f although the H&E is isotropic, so the "
                                 "lattice is not square in physical units and its pitch is not documented; agreement is checked against the "
                                 "geometric-mean assumption" % (y_unit / x_unit)},
    }
    out["agreement_registration_vs_lattice_geometric_mean"] = large / out["spot_lattice"]["large_mpp_if_isotropic_100um_geometric_mean"] - 1
    extra = C.OUT / "scale_checks.json"
    if extra.exists():
        out.update(json.loads(extra.read_text()))
    C.save_json(out, C.OUT / "scale.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(final="--final" in sys.argv)

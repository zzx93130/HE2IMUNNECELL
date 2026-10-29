"""Scientific contract checks for input adapters and the actual authors' methods."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from tls_tools import ROOT, RUNTIME, finder_input, load_finder, r_input


def fixture(b=70, t=30, dispersed=False):
    labels = ["B"] * b + ["T"] * t + ["Other"] * 12 + ["Macrophage"] * 2
    xy = np.random.default_rng(569).normal(0, 3, (len(labels), 2))
    if dispersed:
        xy[:, 0] = np.arange(len(labels)) * 1000.
    return pd.DataFrame({"global_instance_id": 3 * np.arange(len(labels)) + 101,
                         "celltype": labels, "confidence": .95, "x_um": xy[:, 0], "y_um": xy[:, 1],
                         "stable_both_flips": True}, index=np.arange(len(labels)) + 900)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adapted, reference = load_finder()
        cls.adapted, cls.reference = staticmethod(adapted), staticmethod(reference)

    def test_exact_upstream_equivalence_with_markers_used_only_as_test_aliases(self):
        for name, cells in {"mixed": fixture(), "T_only": fixture(b=0, t=70),
                            "dispersed": fixture(dispersed=True), "B_only": fixture(t=0)}.items():
            for seed in [None, 569, 570]:
                with self.subTest(name=name, seed=seed):
                    frame = finder_input(cells, seed=seed)
                    reference = frame.copy()
                    reference["Phenotype"] = reference.Phenotype.replace({"B": "CD20", "T": "CD8"})
                    pd.testing.assert_frame_equal(self.adapted(frame, 160), self.reference(reference, 160))

    def test_geometry_translation_and_index_reset(self):
        cells = fixture()
        original = finder_input(cells)
        shifted = cells.copy()
        shifted["x_um"] += 1500
        shifted["y_um"] -= 2300
        transformed = finder_input(shifted)
        self.assertEqual(list(original.index), list(range(len(cells))))
        before = self.adapted(original, 160)
        after = self.adapted(transformed, 160)
        self.assertEqual(before.Aggregate_Center_Index.tolist(), after.Aggregate_Center_Index.tolist())
        np.testing.assert_allclose(np.array(after.Aggregate_Center_Coordinates.tolist()),
                                   np.array(before.Aggregate_Center_Coordinates.tolist()) + [1500, -2300])

    def test_safe_labels_and_quality_checks_keep_cell_universe(self):
        cells = fixture()
        cells.iloc[0, cells.columns.get_loc("confidence")] = .6
        cells.iloc[1, cells.columns.get_loc("stable_both_flips")] = False
        for variant in ["all", "confidence_0.7", "stable_flips"]:
            result = r_input(cells, variant)
            self.assertEqual(len(result), len(cells))
            self.assertFalse(result.loc[result.source_celltype.eq("Other"), "phenotype"].str.contains("T", case=False).any())
            expected_b = 70 if variant == "all" else 69
            self.assertEqual(result.phenotype.eq("B cell").sum(), expected_b)
            self.assertEqual(result.phenotype.eq("T cell").sum(), 30)

    def test_actual_tlsr_and_finder_distinguish_mixed_t_only_and_dispersed(self):
        runtime = __import__("json").loads((RUNTIME / "manifest.json").read_text())
        env = os.environ.copy()
        env["R_LIBS_USER"] = runtime["R_library"]
        with tempfile.TemporaryDirectory(dir=RUNTIME) as folder:
            path = Path(folder)
            cases = {"mixed": (fixture(), 1, 1), "T_only": (fixture(b=0, t=70), 1, 0),
                     "B_only": (fixture(t=0), 1, 0), "dispersed": (fixture(dispersed=True), 0, 0),
                     "too_few_B": (fixture(b=25, t=35), 1, 0)}
            jobs = []
            for name, (cells, finder_count, r_count) in cases.items():
                self.assertEqual(len(self.adapted(finder_input(cells), 160)), finder_count)
                input_path = path / f"{name}.csv"
                r_input(cells).to_csv(input_path, index=False)
                jobs.append({"run_id": name, "input_csv": str(input_path), "threshold": 10})
            pd.DataFrame(jobs).to_csv(path / "jobs.csv", index=False)
            completed = subprocess.run([runtime["Rscript"], str(ROOT / "deployment" / "run_tlsr.R"),
                                       str(path / "jobs.csv"), str(path / "out")],
                                      env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            metrics = pd.read_csv(path / "out" / "tlsr_metrics.csv").set_index("run_id")
            for name, (_, _, r_count) in cases.items():
                self.assertEqual(metrics.at[name, "detections"], r_count)
                native = pd.read_csv(path / "out" / f"{name}_cells.csv.gz")
                self.assertEqual(len(native), len(cases[name][0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

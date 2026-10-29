"""Verify authored filtering, pyramid alignment and the model's published writer crop."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import openslide
from shapely.geometry import Polygon, box, Point

from hooknet_tls_native import RUNTIME, ROOT, NativeAnnotation, NativeLabel, load_postprocessing
from run_hooknet_tls import padded_patch, MaskReader


class HookNetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_postprocessing()

    def annotation(self, histogram, bounds=(0, 0, 300, 300), name="tls"):
        return NativeAnnotation(1, box(*bounds), NativeLabel(name, histogram))

    def test_exact_TLS_default_support_threshold(self):
        below = self.annotation({210: 21608, 209: 100000})
        equal = self.annotation({210: 21609})
        above = self.annotation({255: 21610})
        result = self.functions["filter_tls_annotations"]([below, equal, above], 210, 147)
        self.assertEqual([id(a) for a in result], [id(equal), id(above)])

    def test_GC_confidence_and_half_area_bounding_box_rule(self):
        tls = self.annotation({255: 30000})
        inside = self.annotation({245: 3600}, bounds=(10, 10, 70, 70), name="gc")
        below = self.annotation({245: 3599, 244: 50000}, bounds=(80, 80, 140, 140), name="gc")
        outside = self.annotation({255: 10000}, bounds=(500, 500, 600, 600), name="gc")
        half = self.annotation({255: 4000}, bounds=(250, 100, 350, 200), name="gc")
        result = self.functions["filter_gc_annotations"]([tls], [inside, below, outside, half], 245, 60)
        self.assertEqual([id(a) for a in result], [id(inside), id(half)])

    def test_native_polygonization_fills_GC_hole_in_TLS(self):
        array = np.zeros((160, 160), dtype=np.uint8)
        array[16:144, 16:144] = 2
        array[56:96, 56:96] = 3
        result = self.functions["cv2_polygonize"](MaskReader(array), spacing=4, dilation_iterations=1,
            erose_iterations=1, fill_holes=True, values=[2, 3])
        self.assertEqual(len(result[2]), 1)
        self.assertEqual(len(result[3]), 1)
        self.assertTrue(Polygon(result[2][0]).contains(Point(76, 76)))

    def test_centered_mask_crop_matches_upstream_function(self):
        manifest = json.loads((RUNTIME / "manifest.json").read_text())
        path = Path(manifest["sources"]["wholeslidedata"]["directory"])/"wholeslidedata/samplers/utils.py"
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "crop_data"]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        grid = np.arange(1244*1244, dtype=np.int32).reshape(1244, 1244)
        np.testing.assert_array_equal(namespace["crop_data"](grid, (1030, 1030))[:1024, :1024], grid[107:-107, 107:-107][:1024, :1024])
        self.assertEqual(padded_patch(np.ones((10, 10), dtype=np.uint8), -5, -5, 4).sum(), 0)

    def test_lossless_pyramid_coordinates_match_saved_pixel_array(self):
        folder = ROOT/"outputs/whole-slide-20261012/tls_tool_comparison/hooknet_tls/preprocessed"
        if not (folder/"slide_0p5um_pyramid.tif").exists():
            self.skipTest("Slide preprocessing has not run yet")
        array = np.load(folder/"RGB_0p5um.npy", mmap_mode="r")
        with openslide.OpenSlide(str(folder/"slide_0p5um_pyramid.tif")) as slide:
            self.assertEqual(slide.dimensions, (array.shape[1], array.shape[0]))
            self.assertAlmostEqual(float(slide.properties[openslide.PROPERTY_NAME_MPP_X]), .5)
            patch = np.asarray(slide.read_region((7975, 7025), 0, (128, 128)))[..., :3]
            np.testing.assert_array_equal(patch, array[7025:7153, 7975:8103])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Mac compatibility layer; published model topology and filter functions retained."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime" / "hooknet_tls"


def load_model(audit_dir=None):
    import tensorflow as tf
    import yaml
    manifest = json.loads((RUNTIME / "manifest.json").read_text())
    source_root = Path(manifest["sources"]["hooknet"]["directory"])
    tls_root = Path(manifest["sources"]["hooknet_tls"]["directory"])
    sys.path.insert(0, str(source_root))
    source_file = source_root / "hooknet" / "models" / "tensorflowmodel.py"
    original = source_file.read_text()
    replacements = {
        "from tensorflow.keras.layers.experimental.preprocessing import Rescaling": "from tensorflow.keras.layers import Rescaling",
        "from tensorflow.python.framework.ops import Tensor": "from tensorflow import Tensor",
        "            sample_weight_mode=sample_weights_mode,\n": "",
    }
    adapted = original
    for before, after in replacements.items():
        if adapted.count(before) != 1:
            raise ValueError("Reviewed model compatibility substitution no longer matches source")
        adapted = adapted.replace(before, after)
    adapter_dir = RUNTIME / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    adapter_file = adapter_dir / "tensorflowmodel_mac.py"
    adapter_file.write_text(adapted)
    spec = importlib.util.spec_from_file_location("hooknet_tensorflow_mac", adapter_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = tls_root / "hooknettls" / "configs" / "config.yml"
    config = yaml.safe_load(config_path.read_text())["default"]
    params = config["model"]
    model = module.HookNet(input_shape=params["input_shape"], hook_indices=params["hook_indices@replace(true)"],
        n_classes=params["n_classes"], batch_norm=params["batch_norm"], n_filters=params["n_filters"],
        activation=tf.keras.layers.LeakyReLU(alpha=params["activation"]["alpha"]),
        model_weights=str(RUNTIME / "weights.h5"))
    # Published model emits 1030²; its writer retains the upper-left 1024².
    if tuple(model.output_shape) != (1030, 1030, 3):
        raise ValueError(f"Unexpected published-model output shape: {model.output_shape}")
    if audit_dir:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "tensorflowmodel_upstream.py").write_text(original)
        (audit_dir / "tensorflowmodel_mac.py").write_text(adapted)
        (audit_dir / "tensorflowmodel_mac.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True), fromfile="upstream", tofile="Mac_compatibility")))
        (audit_dir / "published_config.yml").write_text(config_path.read_text())
        (audit_dir / "compatibility.json").write_text(json.dumps({"source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "adapted_sha256": hashlib.sha256(adapted.encode()).hexdigest(), "changes": list(replacements),
            "topology_or_weight_changes": False, "training_only_compile_option_removed": "sample_weight_mode",
            "input_rescaling": "Authored in-model Rescaling(1/255), no external stain normalization"}, indent=2)+"\n")
    return model, config, manifest


@dataclass
class NativeLabel:
    name: str
    confidence_map: dict


@dataclass
class NativeAnnotation:
    index: int
    geometry: object
    label: NativeLabel


def load_postprocessing(audit_dir=None):
    """Execute the authors' polygonization/filtering functions with reader/API adapters."""
    import cv2
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
    from shapely.geometry import box
    from shapely.strtree import STRtree
    manifest = json.loads((RUNTIME / "manifest.json").read_text())
    source_file = Path(manifest["sources"]["hooknet_tls"]["directory"]) / "hooknettls" / "postprocessing.py"
    names = {"cv2_polygonize", "filter_tls_annotations", "in_tls", "filter_gc_annotations"}
    source = source_file.read_text()
    parsed = ast.parse(source)
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != 4:
        raise ValueError("Published postprocessing functions differ from inspected source")

    class LegacySTRtree:
        def __init__(self, geometries):
            self.geometries = list(geometries)
            self.tree = STRtree(self.geometries)

        def query(self, geometry):
            # Shapely 2 returns indices; the author's Shapely 1.8 returns geometry.
            return [self.geometries[int(i)] for i in self.tree.query(geometry)]

    namespace = {"cv2": cv2, "np": np, "binary_dilation": binary_dilation, "binary_erosion": binary_erosion,
                 "binary_fill_holes": binary_fill_holes, "box": box, "STRtree": LegacySTRtree,
                 "ShapelyPolygon": __import__("shapely.geometry", fromlist=["Polygon"]).Polygon}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(source_file), "exec"), namespace)
    if audit_dir:
        audit_dir = Path(audit_dir)
        (audit_dir / "postprocessing_upstream.py").write_text(source)
        (audit_dir / "postprocessing_functions.py").write_text(ast.unparse(module)+"\n")
    return {name: namespace[name] for name in names}

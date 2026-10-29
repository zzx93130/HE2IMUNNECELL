"""Auditable input adapters for the pinned authors' spatial TLS algorithms."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime" / "tls_tools"
NOTEBOOK = RUNTIME / "sources" / "TLS-Finder" / "TLS_Finder_Code1_LA_Query_and_Plot_GitH.ipynb"


def eligible_labels(cells: pd.DataFrame, variant: str = "all") -> np.ndarray:
    labels = cells.celltype.map({"B": "B", "T": "T", "Other": "Unclassified", "Macrophage": "Macrophage"})
    if labels.isna().any():
        raise ValueError("Unexpected source phenotype")
    labels = labels.to_numpy(copy=True)
    if variant == "confidence_0.7":
        eligible = cells.confidence.to_numpy() >= .7
    elif variant == "stable_flips":
        eligible = cells.stable_both_flips.to_numpy(dtype=bool)
    elif variant == "all":
        eligible = np.ones(len(cells), dtype=bool)
    else:
        raise ValueError(f"Unknown input variant: {variant}")
    labels[np.isin(labels, ["B", "T"]) & ~eligible] = "Unclassified"
    return labels


def finder_input(cells: pd.DataFrame, variant: str = "all", seed: int | None = None) -> pd.DataFrame:
    ordered = cells.sort_values("global_instance_id", kind="stable").reset_index(drop=True)
    df = pd.DataFrame({"X": ordered.x_um, "Y": ordered.y_um, "Phenotype": eligible_labels(ordered, variant),
                       "global_instance_id": ordered.global_instance_id})
    if seed is not None:
        df = df.iloc[np.random.default_rng(seed).permutation(len(df))]
    return df.reset_index(drop=True)


def r_input(cells: pd.DataFrame, variant: str = "all") -> pd.DataFrame:
    ordered = cells.sort_values("global_instance_id", kind="stable").reset_index(drop=True)
    labels = eligible_labels(ordered, variant)
    df = pd.DataFrame({"global_instance_id": ordered.global_instance_id, "x": ordered.x_um, "y": ordered.y_um,
                       "phenotype": pd.Series(labels).replace({"B": "B cell", "T": "T cell"}),
                       "source_celltype": ordered.celltype, "confidence": ordered.confidence})
    # detect_TLS uses substring matching. Verify it recognizes exactly the intended groups.
    assert np.array_equal(df.phenotype.str.contains("B", case=False), labels == "B")
    assert np.array_equal(df.phenotype.str.contains("T", case=False), labels == "T")
    return df


def load_finder(audit_dir: Path | None = None):
    """Extract only measure_TLS; preserve upstream logic apart from two explicit edits."""
    notebook = json.loads(NOTEBOOK.read_text())
    matches = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def measure_TLS(" in source:
            matches.extend(node for node in ast.parse(source).body
                           if isinstance(node, ast.FunctionDef) and node.name == "measure_TLS")
    if len(matches) != 1:
        raise RuntimeError("Pinned notebook must contain exactly one measure_TLS function")
    original = ast.Module(body=[matches[0]], type_ignores=[])
    adapted = copy.deepcopy(original)
    function = adapted.body[0]
    removed = []
    kept = []
    for statement in function.body:
        graph_assignment = isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "G" for t in statement.targets)
        graph_loop = isinstance(statement, ast.For) and any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == "G" for n in ast.walk(statement))
        if graph_assignment or graph_loop:
            removed.append(ast.unparse(statement))
        else:
            kept.append(statement)
    if len(removed) != 3:
        raise RuntimeError("Upstream graph code differs from the reviewed version")
    function.body = kept

    class GenericLabels(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value in ("CD20", "CD8"):
                return ast.copy_location(ast.Constant("B" if node.value == "CD20" else "T"), node)
            return node

    adapted = ast.fix_missing_locations(GenericLabels().visit(adapted))
    original_text = "import pandas as pd\nfrom scipy.spatial import KDTree\nimport networkx as nx\n\n" + ast.unparse(original) + "\n"
    adapted_text = "import pandas as pd\nfrom scipy.spatial import KDTree\n\n" + ast.unparse(adapted) + "\n"
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "TLS_Finder_reference.py").write_text(original_text)
        (audit_dir / "TLS_Finder_adapter.py").write_text(adapted_text)
        import difflib
        (audit_dir / "TLS_Finder_adapter.diff").write_text("".join(difflib.unified_diff(original_text.splitlines(True), adapted_text.splitlines(True), fromfile="upstream_reference", tofile="generic_label_adapter")))
        (audit_dir / "adapter_manifest.json").write_text(json.dumps({
            "notebook": str(NOTEBOOK), "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
            "adapter_sha256": hashlib.sha256(adapted_text.encode()).hexdigest(),
            "changes": ["CD20/CD8 internal selection literals replaced by generic B/T", "Unused graph construction removed"],
            "unchanged": "Greedy selection, >=50 unused B/T rule, radius query, and used_cells updates",
        }, indent=2) + "\n")
    reference_namespace = {"pd": pd, "KDTree": KDTree}
    import networkx as nx
    reference_namespace["nx"] = nx
    namespace = {"pd": pd, "KDTree": KDTree}
    exec(compile(original, str(NOTEBOOK), "exec"), reference_namespace)
    exec(compile(adapted, str(NOTEBOOK) + " [generic-label adapter]", "exec"), namespace)
    return namespace["measure_TLS"], reference_namespace["measure_TLS"]

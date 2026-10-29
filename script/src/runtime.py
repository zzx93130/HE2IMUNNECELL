"""Shared device and checkpoint handling for native macOS inference."""
from pathlib import Path

import torch


def select_device(requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU is unavailable to this Python process. Try --device cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. On Apple Silicon use --device mps or auto.")
    return torch.device(requested)


def load_classifier_state(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained classifier checkpoint not found: {path}")
    # Linux training checkpoints can include NumPy values and optimizer metadata.
    # Only load trusted checkpoints produced by this project's training run.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if "backbone.fc.weight" not in state or state["backbone.fc.weight"].shape[0] != 4:
        raise ValueError("Expected the four-class SAM_ConvNet checkpoint (Other, T, Macrophage, B).")
    return state

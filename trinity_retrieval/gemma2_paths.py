#!/usr/bin/env python3
"""Cross-platform resolution for google/embeddinggemma-2 weights (torch-free).

Priority order:
  1. $GEMMA2_WEIGHTS_DIR (explicit operator directory)
  2. <repo>/models/embeddinggemma-2/ (if populated locally)
  3. $GEMMA2_MODEL_PATH (legacy env override)
  4. Hugging Face Hub identifier ("google/embeddinggemma-2")
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODELS_DIR = _HERE.parent / "models" / "embeddinggemma-2"
_HUB_ID = "google/embeddinggemma-2"


def _has_config(dir_path: Path) -> bool:
    return dir_path.is_dir() and (dir_path / "config.json").exists()


def model_dir() -> str:
    """Return local weights directory containing config.json, or HF hub ID."""
    for env_key in ("GEMMA2_WEIGHTS_DIR", "GEMMA2_MODEL_PATH"):
        env_val = os.environ.get(env_key, "").strip()
        if env_val and _has_config(Path(env_val)):
            return env_val
    if _has_config(_MODELS_DIR):
        return str(_MODELS_DIR)
    return _HUB_ID


def weights_file() -> str:
    """Return path to model.safetensors inside resolved local dir (or expected repo path)."""
    resolved = Path(model_dir())
    cand = resolved / "model.safetensors"
    if cand.is_file():
        return str(cand)
    return str(_MODELS_DIR / "model.safetensors")


def missing_weights_hint() -> str:
    return (
        "EmbeddingGemma-2 weights not found locally. Download google/embeddinggemma-2 "
        "from Hugging Face or set GEMMA2_WEIGHTS_DIR to a directory containing "
        "config.json and model.safetensors."
    )

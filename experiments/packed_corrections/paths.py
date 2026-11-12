"""Repository-relative sources and configurable, untracked experiment data."""
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("LANGUAGE_U_EXPERIMENT_DIR", str(Path(__file__).resolve().parent / "data"))).resolve()

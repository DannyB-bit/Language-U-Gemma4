"""
codebooks.py — Deterministic S^7 Codebook Registry & Sentinel Gatekeeper (V2)
=============================================================================
Resolves the Three-Codebook Law across DannyB-bit/Language-U-Gemma4 and
DannyB-bit/Neon-Runner with fail-closed sentinel verification on row #1419:

  1. `compass_s7` (`s7_codebook_2048x8.bin`, 65,536 B):
     - Neon-Runner Semantic Compass / True North codebook.
     - Row #1419 sentinel: [-1/sqrt(6), -1/sqrt(6), +1/sqrt(6), 0, +1/sqrt(6), +1/sqrt(6), -1/sqrt(6), 0]
       (~ [-0.408248, -0.408248, 0.408248, 0.0, 0.408248, 0.408248, -0.408248, 0.0]).
     - Passes Neon-Runner's hardcoded `neon_stream_accelerator.c` and
       `benchmark_114m_neon_runner.py` sentinel checks.

  2. `trit_rvq` (`trit_codebook_2048x8.bin`, 65,536 B):
     - Ternary Lattice {-1, 0, +1}^8 k=1..5 sparsity codebook (`build_trit_codebook`).
     - Row #1419 sentinel: [0.0, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, 0.5].
     - Required by `UniversalLayerOrderedCortex` (`cortex.py`).

  3. `golden_s7` (65,536 B):
     - Analytical golden-ratio phase codebook (`build_canonical_s7_codebook`).
     - Row #1419 sentinel: [-0.216944, 0.504493, -0.217914, 0.330880,
                            -0.454508, 0.162780, -0.444247, 0.333399].
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


CODEBOOK_SHAPE: Tuple[int, int] = (2048, 8)
CODEBOOK_BYTES: int = 2048 * 8 * 4  # 65,536 bytes
SENTINEL_INDEX: int = 1419

SENTINEL_COMPASS_S7 = np.array(
    [-0.40824829, -0.40824829, 0.40824829, 0.0, 0.40824829, 0.40824829, -0.40824829, 0.0],
    dtype=np.float32,
)
SENTINEL_TRIT_RVQ = np.array(
    [0.0, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, 0.5],
    dtype=np.float32,
)
SENTINEL_GOLDEN_S7 = np.array(
    [-0.2169442, 0.5044933, -0.2179136, 0.3308804, -0.4545085, 0.1627798, -0.4442466, 0.3333986],
    dtype=np.float32,
)

SENTINEL_TABLE: Dict[str, np.ndarray] = {
    "compass_s7": SENTINEL_COMPASS_S7,
    "trit_rvq": SENTINEL_TRIT_RVQ,
    "golden_s7": SENTINEL_GOLDEN_S7,
}


@dataclass(frozen=True)
class LoadedCodebookV2:
    family: str
    array: np.ndarray
    source_path: Optional[str]
    sentinel_max_diff: float


def build_trit_rvq_codebook() -> np.ndarray:
    """Deterministic 2048x8 unit-norm ternary lattice codebook (k=1..5 sparsity)."""
    states = np.zeros((6561, 8), dtype=np.float32)
    k_sparsity = np.zeros(6561, dtype=np.int32)
    for i in range(6561):
        val = i
        count = 0
        for d in range(8):
            trit = float(val % 3) - 1.0
            states[i, d] = trit
            if trit != 0.0:
                count += 1
            val //= 3
        k_sparsity[i] = count

    selected = np.concatenate([
        np.where(k_sparsity == 1)[0],
        np.where(k_sparsity == 2)[0],
        np.where(k_sparsity == 3)[0],
        np.where(k_sparsity == 4)[0],
        np.where(k_sparsity == 5)[0][:352],
    ])
    assert selected.shape == (2048,), f"Expected 2048 trit states, got {selected.shape}"
    raw = states[selected]
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cb = (raw / np.maximum(norms, 1e-12)).astype(np.float32)
    assert cb.shape == CODEBOOK_SHAPE
    return np.ascontiguousarray(cb, dtype=np.float32)


def build_golden_s7_codebook() -> np.ndarray:
    """Deterministic 2048x8 analytical golden-ratio phase codebook on S^7."""
    idx = np.arange(2048, dtype=np.float32)[:, None]
    d = np.arange(8, dtype=np.float32)[None, :]
    phases = idx * (d + 1.0) * 0.61803398875 * math.pi
    cb = np.cos(phases) + 0.5 * np.sin(phases * 1.41421356)
    norms = np.linalg.norm(cb, axis=1, keepdims=True)
    out = (cb / np.maximum(norms, 1e-12)).astype(np.float32)
    assert out.shape == CODEBOOK_SHAPE
    return np.ascontiguousarray(out, dtype=np.float32)


class CodebookRegistryV2:
    """
    Fail-closed multi-tier codebook resolver and sentinel verifier:
      Tier 1: Explicit constructor/argument path
      Tier 2: Environment variable (`LANGU_CODEBOOK_PATH`, `NEON_RUNNER_DIR`)
      Tier 3: Repo-relative `neon_runner/codebooks/*.bin`
      Tier 4: Deterministic mathematical synthesis (for `trit_rvq` / `golden_s7`)
    """

    @staticmethod
    def identify_family(cb: np.ndarray, atol: float = 1e-3) -> Tuple[str, float]:
        if cb.shape != CODEBOOK_SHAPE or cb.dtype != np.float32:
            raise ValueError(f"Invalid codebook tensor shape/dtype: {cb.shape} {cb.dtype}")
        norms = np.linalg.norm(cb, axis=1)
        max_norm_err = float(np.max(np.abs(norms - 1.0)))
        if max_norm_err > 1e-4:
            raise ValueError(f"Codebook failed S^7 unit-norm invariant (max_err={max_norm_err:.6e})")

        row1419 = cb[SENTINEL_INDEX]
        best_family = "unknown"
        best_diff = float("inf")
        for fam, sentinel in SENTINEL_TABLE.items():
            diff = float(np.max(np.abs(row1419 - sentinel)))
            if diff < best_diff:
                best_diff = diff
                best_family = fam

        if best_diff <= atol:
            return best_family, best_diff
        raise ValueError(
            f"Codebook row #{SENTINEL_INDEX} failed all known S^7 sentinels "
            f"(closest={best_family}, max_diff={best_diff:.6f}, row={row1419.tolist()})"
        )

    @classmethod
    def load_from_file(cls, path: str, expected_family: Optional[str] = None) -> LoadedCodebookV2:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Codebook artifact missing at '{p}' (expected {CODEBOOK_BYTES:,} bytes)"
            )
        size = p.stat().st_size
        if size != CODEBOOK_BYTES:
            raise ValueError(
                f"Codebook byte-size law violated at '{p}': got {size:,} bytes, expected {CODEBOOK_BYTES:,} bytes"
            )
        arr = np.fromfile(str(p), dtype=np.float32).reshape(CODEBOOK_SHAPE)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        fam, diff = cls.identify_family(arr)
        if expected_family is not None and fam != expected_family:
            raise ValueError(
                f"Codebook family mismatch at '{p}': expected '{expected_family}', detected '{fam}'"
            )
        return LoadedCodebookV2(family=fam, array=arr, source_path=str(p.resolve()), sentinel_max_diff=diff)

    @classmethod
    def resolve(
        cls,
        family: str = "compass_s7",
        explicit_path: Optional[str] = None,
    ) -> LoadedCodebookV2:
        if family not in SENTINEL_TABLE and explicit_path is None:
            raise ValueError(f"Unsupported codebook family '{family}'. Expected one of {list(SENTINEL_TABLE)}")

        if explicit_path:
            expected = family if family in SENTINEL_TABLE else None
            return cls.load_from_file(explicit_path, expected_family=expected)

        env_cb = os.environ.get("LANGU_CODEBOOK_PATH")
        if env_cb and os.path.isfile(env_cb):
            loaded = cls.load_from_file(env_cb)
            if family in ("auto", loaded.family):
                return loaded

        filename = "s7_codebook_2048x8.bin" if family == "compass_s7" else (
            "trit_codebook_2048x8.bin" if family == "trit_rvq" else ""
        )

        if filename:
            here = Path(__file__).resolve().parent
            candidates = [
                here / "codebooks" / filename,
                here.parent / filename,
            ]
            neon_dir = os.environ.get("NEON_RUNNER_DIR")
            if neon_dir:
                candidates.insert(0, Path(neon_dir) / filename)

            for cand in candidates:
                if cand.is_file():
                    return cls.load_from_file(str(cand), expected_family=family)

        if family == "trit_rvq":
            arr = build_trit_rvq_codebook()
            _, diff = cls.identify_family(arr)
            return LoadedCodebookV2(family="trit_rvq", array=arr, source_path=None, sentinel_max_diff=diff)

        if family == "golden_s7":
            arr = build_golden_s7_codebook()
            _, diff = cls.identify_family(arr)
            return LoadedCodebookV2(family="golden_s7", array=arr, source_path=None, sentinel_max_diff=diff)

        raise FileNotFoundError(
            f"Physical '{family}' codebook binary ('s7_codebook_2048x8.bin', {CODEBOOK_BYTES:,} B) not found "
            f"in explicit path, $NEON_RUNNER_DIR, or neon_runner/codebooks/."
        )


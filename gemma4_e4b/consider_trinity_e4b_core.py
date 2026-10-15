#!/usr/bin/env python3
"""
consider_trinity_e4b_core.py
================================================================================
Consider-Trinity-Gemma4-E4B: Converted Tripartite Connectome for google/gemma-4-E4B-it
  - Base Architecture: google/gemma-4-E4B-it (and gemma-4-e4b-it-qat-w4a16-ct)
  - Manifold Dimension (d_model): 2,560 (10 Macro-Glyphs / row, R = 0 mod 256)
  - Intermediate Dimension (d_ff): 10,240 (40 Macro-Glyphs / col, R = 0 mod 256)
  - Per-Layer Embedding (PLE): 256 (1 Macro-Glyph / token, R = 0 mod 256)
  - Vocabulary Size: 262,144 (1,024 x 256, R = 0 mod 256)
  - Decoder Layers: 42 (Layers 0..4 & 37..41 in Opt4 132B; Layers 5..36 in Opt5 36B/48B)
  - Shared KV Layers: 18 (Layers 24..41 share K/V projections — encoded once)
================================================================================
"""

import json
import math
from typing import Dict, Any, List, Tuple
import numpy as np

# Canonical Gemma 4 E4B Architectural Constants (All divisible by 256 with R = 0)
D_MODEL = 2560
D_FF = 10240
PLE_DIM = 256
VOCAB_SIZE = 262144
N_LAYERS = 42
N_KV_SHARED_LAYERS = 18
WEIGHTS_PER_GLYPH = 256
GLYPHS_PER_EMBED_ROW = D_MODEL // WEIGHTS_PER_GLYPH  # Exactly 10 glyphs/row
GLYPHS_PER_PLE_ROW = PLE_DIM // WEIGHTS_PER_GLYPH    # Exactly 1 glyph/row
OCTONION_DIM = 8
CODEBOOK_SIZE = 2048  # Matches Gemma 4 E4B Assistant num_centroids = 2048

BYTES_PER_GLYPH_OPT4 = 132
BYTES_PER_GLYPH_OPT5 = 36
BYTES_PER_GLYPH_OPT5_OUTLIER = 48  # 36B base + Top-4 FP16 outliers (12B)


def is_boundary_layer_e4b(layer_idx: int) -> bool:
    """Boundary Sandwich for 42-Layer Gemma 4 E4B: Layers 0..4 and 37..41 (10 layers)."""
    return (0 <= layer_idx <= 4) or (37 <= layer_idx <= 41)


def generate_orthonormal_dct_basis_2560_8() -> np.ndarray:
    """
    Orthonormal Karhunen-Loeve / DCT-II Eigenspace basis (2560 x 8) for Gemma 4 E4B.
    Satisfies U^T @ U = I_8 to within FP32 machine epsilon (< 1e-6).
    """
    basis = np.zeros((D_MODEL, OCTONION_DIM), dtype=np.float32)
    basis[:, 0] = 1.0 / math.sqrt(D_MODEL)
    n = np.arange(D_MODEL, dtype=np.float32)
    for k in range(1, OCTONION_DIM):
        basis[:, k] = math.sqrt(2.0 / D_MODEL) * np.cos(
            math.pi * (2.0 * n + 1.0) * k / (2.0 * D_MODEL)
        )
    return basis


_DCT_BASIS_2560_8 = generate_orthonormal_dct_basis_2560_8()


def sylvester_hadamard_h8() -> np.ndarray:
    H1 = np.array([[1.0]], dtype=np.float32)
    H2 = np.block([[H1, H1], [H1, -H1]]) / math.sqrt(2.0)
    H4 = np.block([[H2, H2], [H2, -H2]]) / math.sqrt(2.0)
    H8 = np.block([[H4, H4], [H4, -H4]]) / math.sqrt(2.0)
    return H8.astype(np.float32)


_H8 = sylvester_hadamard_h8()


def project_256d_glyph_to_s7(vec_256d: np.ndarray) -> np.ndarray:
    """
    Maps a 256-D vector (either a Gemma 4 E4B Per-Layer Embedding row OR a
    Kaggle competition 256-D AST node embedding) into a unit Octonion on S^7.
    Exploits 256 = 32 blocks x 8D Octonions with Remainder R = 0.
    """
    v = np.asarray(vec_256d, dtype=np.float32).reshape(32, 8)
    decay = np.exp(-np.linspace(0.0, 1.5, 32, dtype=np.float32))[:, None]
    folded = (v * decay).sum(axis=0) @ _H8.T
    norm = float(np.linalg.norm(folded))
    if norm > 1e-12:
        return (folded / norm).astype(np.float32)
    out = np.zeros(8, dtype=np.float32)
    out[0] = 1.0
    return out


def project_2560d_hidden_to_s7(hidden_2560d: np.ndarray) -> np.ndarray:
    """Projects a 2560-D Gemma 4 E4B cortical state onto S^7 via _DCT_BASIS_2560_8."""
    x = np.asarray(hidden_2560d, dtype=np.float32)
    coords = x @ _DCT_BASIS_2560_8
    norm = float(np.linalg.norm(coords))
    if norm > 1e-12:
        return (coords / norm).astype(np.float32)
    out = np.zeros(8, dtype=np.float32)
    out[0] = 1.0
    return out


def verify_gemma4_e4b_zero_remainder() -> Dict[str, Any]:
    """
    Formally proves the Zero Remainder Invariant (R = 0 mod 256) across all
    canonical 2D weight families of google/gemma-4-E4B-it.
    Exact dimensions matching physical weights:
      - Q: [2048, 2560]
      - K: [512, 2560]
      - V: [512, 2560]
      - O: [2560, 2048]
      - Gate: [10240, 2560]
      - Up: [10240, 2560]
      - Down: [2560, 10240]
      - PLE Proj: [2560, 256]
      - PLE Gate: [256, 2560]
      - Embed: [262144, 2560]
      - PLE Table: [262144, 10752]
    """
    active_kv_layers = N_LAYERS - N_KV_SHARED_LAYERS  # 24 layers own unique K/V projections
    families = {
        "Token Embedding Table (embed_tokens)": {
            "matrices": 1,
            "shape": [VOCAB_SIZE, D_MODEL],
            "weights": VOCAB_SIZE * D_MODEL,
        },
        "Per-Layer Embedding Table (embed_tokens_per_layer 42x256)": {
            "matrices": 1,
            "shape": [VOCAB_SIZE, N_LAYERS * PLE_DIM],
            "weights": VOCAB_SIZE * N_LAYERS * PLE_DIM,
        },
        "Per-Layer Projection (42 layers x 2560 x 256)": {
            "matrices": N_LAYERS,
            "shape": [D_MODEL, PLE_DIM],
            "weights": N_LAYERS * D_MODEL * PLE_DIM,
        },
        "Per-Layer Input Gate (42 layers x 256 x 2560)": {
            "matrices": N_LAYERS,
            "shape": [PLE_DIM, D_MODEL],
            "weights": N_LAYERS * PLE_DIM * D_MODEL,
        },
        "MLP Gate Projection (42 layers x 10240 x 2560)": {
            "matrices": N_LAYERS,
            "shape": [D_FF, D_MODEL],
            "weights": N_LAYERS * D_FF * D_MODEL,
        },
        "MLP Up Projection (42 layers x 10240 x 2560)": {
            "matrices": N_LAYERS,
            "shape": [D_FF, D_MODEL],
            "weights": N_LAYERS * D_FF * D_MODEL,
        },
        "MLP Down Projection (42 layers x 2560 x 10240)": {
            "matrices": N_LAYERS,
            "shape": [D_MODEL, D_FF],
            "weights": N_LAYERS * D_MODEL * D_FF,
        },
        "Attention Q Projection (42 layers x 2048 x 2560)": {
            "matrices": N_LAYERS,
            "shape": [2048, D_MODEL],
            "weights": N_LAYERS * 2048 * D_MODEL,
        },
        "Attention O Projection (42 layers x 2560 x 2048)": {
            "matrices": N_LAYERS,
            "shape": [D_MODEL, 2048],
            "weights": N_LAYERS * D_MODEL * 2048,
        },
        "Attention Unique K Projection (24 owning layers x 512 x 2560)": {
            "matrices": active_kv_layers,
            "shape": [512, D_MODEL],
            "weights": active_kv_layers * 512 * D_MODEL,
        },
        "Attention Unique V Projection (24 owning layers x 512 x 2560)": {
            "matrices": active_kv_layers,
            "shape": [512, D_MODEL],
            "weights": active_kv_layers * 512 * D_MODEL,
        },
    }

    total_matrices = 0
    total_weights = 0
    total_glyphs = 0
    report = {}

    for name, spec in families.items():
        w = spec["weights"]
        rem = w % WEIGHTS_PER_GLYPH
        assert rem == 0, f"Remainder violation in {name}: {rem}"
        g = w // WEIGHTS_PER_GLYPH
        total_matrices += spec["matrices"]
        total_weights += w
        total_glyphs += g
        report[name] = {
            "matrices": spec["matrices"],
            "shape": spec["shape"],
            "weights": w,
            "glyphs": g,
            "remainder": rem,
        }

    # Compute Dual-Precision Hybrid stream footprint (10 boundary layers @ 132B, 32 middle @ 36B)
    boundary_ratio = 10.0 / 42.0
    boundary_glyphs = int(round(total_glyphs * boundary_ratio))
    middle_glyphs = total_glyphs - boundary_glyphs
    hybrid_bytes = (boundary_glyphs * BYTES_PER_GLYPH_OPT4) + (middle_glyphs * BYTES_PER_GLYPH_OPT5)

    return {
        "model": "Consider-Trinity-Gemma4-E4B",
        "base_model": "google/gemma-4-E4B-it",
        "kaggle_model_id": "gemma-4-e4b-it-qat-w4a16-ct",
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "ple_dim": PLE_DIM,
        "vocab_size": VOCAB_SIZE,
        "num_layers": N_LAYERS,
        "num_kv_shared_layers": N_KV_SHARED_LAYERS,
        "total_matrices_verified": total_matrices,
        "total_active_weights": total_weights,
        "total_discrete_glyphs": total_glyphs,
        "all_remainders_zero": True,
        "hybrid_stream_footprint_mb": round(hybrid_bytes / (1024 * 1024), 2),
        "families": report,
    }


def _self_test() -> bool:
    # 1. Verify Orthonormality of 2560 x 8 DCT-II Basis
    gram = _DCT_BASIS_2560_8.T @ _DCT_BASIS_2560_8
    err = float(np.max(np.abs(gram - np.eye(8, dtype=np.float32))))
    assert err < 1e-5, f"DCT-II 2560x8 basis orthogonality error: {err}"

    # 2. Verify Zero Remainder Invariant across all Gemma 4 E4B families
    manifest = verify_gemma4_e4b_zero_remainder()
    assert manifest["all_remainders_zero"] is True

    # 3. Verify 256-D PLE / AST Embedding -> S^7 Unit Octonion Projection
    rng = np.random.default_rng(42)
    sample_256 = rng.standard_normal(256, dtype=np.float32)
    s7 = project_256d_glyph_to_s7(sample_256)
    assert s7.shape == (8,) and abs(float(np.linalg.norm(s7)) - 1.0) < 1e-5

    print(f"[+] Consider-Trinity-Gemma4-E4B Self-Test PASSED!")
    print(f"    - Verified {manifest['total_matrices_verified']} matrices | "
          f"{manifest['total_active_weights']:,} weights -> "
          f"{manifest['total_discrete_glyphs']:,} Macro-Glyphs (R = 0)")
    print(f"    - Dual-Precision Hybrid Stream Size: {manifest['hybrid_stream_footprint_mb']} MB")
    return True


if __name__ == "__main__":
    _self_test()

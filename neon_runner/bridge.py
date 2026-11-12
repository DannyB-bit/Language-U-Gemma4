#!/usr/bin/env python3
"""
bridge.py — Language-U-Gemma4 <-> Neon-Runner Unified Bridge (V2, v2.1.0 — GlyphAttentionV2)
================================================================================
Connects `DannyB-bit/Language-U-Gemma4` (Gemma 4 E4B & 31B Dual-Road Option 4 +
Option 5 streams) directly to `DannyB-bit/Neon-Runner` (`libneon_stream_accelerator`).

Resolves all 4 cross-repository compatibility seams:
  1. Option 4 + Option 5 Unified Execution:
     - Uses native C/SIMD (`gemv_opt4_fwht_mmap`, `gemv_opt4_mmap`, `gemv_opt5_mmap`,
       `decode_glyphs_c`, `decode_glyphs_opt5_c`) when V2 library is loaded.
     - Automatically falls back to blockwise FWHT-8 input pre-rotation + vectorized
       Option 5 algebraic factoring when paired with an unmodified Neon-Runner v1 `.so`.
  2. Three-Codebook Sentinel Gatekeeper (`CodebookRegistryV2`):
     - Validates row #1419 sentinels fail-closed across `compass_s7` (True North),
       `trit_rvq` (Ternary Lattice), and `golden_s7`.
  3. Hadamard Pre-Rotation Parity (`rotation_mode = "fwht8" | "direct"`):
     - Ensures `gemv(W, x)` matches `decode(W) @ x` to within FP32 tolerance (< 1e-4)
       under both `encode_glyph_132` (`fwht8`) and `RVQCompressor` (`direct`) laws.
  4. Multi-Architecture & Multi-Tier Path Resolution:
     - Parameterized for Gemma 4 E4B (`d_model=2560`, 42L) and Gemma 4 31B
       (`d_model=5376`, 60L) without hardcoded `/mnt/SHE-VAULT` or fixed-size traps.
================================================================================
"""

import argparse
import ctypes
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from .codebooks import CodebookRegistryV2, LoadedCodebookV2
    from .build_native import build_shared_library
    from .pi4_stream import validate_projection, checked_layout
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neon_runner.codebooks import CodebookRegistryV2, LoadedCodebookV2
    from neon_runner.build_native import build_shared_library
    from neon_runner.pi4_stream import validate_projection, checked_layout


BRIDGE_VERSION = "2.1.0"
BRIDGE_COMPONENT_ID = "Language-U-Gemma4-NeonRunner-Bridge-V2"
LANU_MAGIC = b"LANU"
LANU_PACKED_MAGIC = b"LANP"
LANU_HEADER_BYTES = 16
WEIGHTS_PER_GLYPH = 256
BPG_OPT4 = 132
BPG_OPT4_PACKED = 92
BPG_OPT5_STD = 36
BPG_OPT5_OUTLIER = 48

H8_MATRIX = (
    np.array(
        [
            [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
            [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
            [ 1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0],
            [ 1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0],
            [ 1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
        ],
        dtype=np.float32,
    )
    / math.sqrt(8.0)
)


@dataclass(frozen=True)
class StreamHeaderV2:
    magic: bytes
    version: int
    flags: int
    total_glyphs: int
    boundary_glyphs: int


def parse_lanu_header(raw_16: bytes) -> StreamHeaderV2:
    if len(raw_16) != LANU_HEADER_BYTES:
        raise ValueError(f"Header slice must be {LANU_HEADER_BYTES} bytes, got {len(raw_16)}")
    magic, ver, flags, total_g, bound_g = struct.unpack("<4sHHII", raw_16)
    if (magic, ver) not in ((LANU_MAGIC, 1), (LANU_MAGIC, 2), (LANU_PACKED_MAGIC, 3)):
        raise ValueError(f"Invalid stream magic/version: {magic!r}/{ver}")
    if bound_g > total_g:
        raise ValueError(f"Lying header detected: boundary_glyphs ({bound_g}) > total_glyphs ({total_g})")
    return StreamHeaderV2(
        magic=magic,
        version=ver,
        flags=flags,
        total_glyphs=total_g,
        boundary_glyphs=bound_g,
    )


def pack_lanu_header(total_glyphs: int, boundary_glyphs: int, version: int = 2, flags: int = 45) -> bytes:
    if total_glyphs < 0 or boundary_glyphs < 0 or boundary_glyphs > total_glyphs:
        raise ValueError(f"Invalid glyph counts: total={total_glyphs}, boundary={boundary_glyphs}")
    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported container version: {version}")
    magic = LANU_PACKED_MAGIC if version == 3 else LANU_MAGIC
    return struct.pack("<4sHHII", magic, version, flags, total_glyphs, boundary_glyphs)


def fwht8_vec_numpy(x: np.ndarray) -> np.ndarray:
    """Vectorized orthonormal Sylvester-Hadamard H_8 transform across an array divisible by 8."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1 or (arr.shape[0] % 8) != 0:
        raise ValueError(f"fwht8_vec_numpy requires 1D array divisible by 8, got shape {arr.shape}")
    return np.ascontiguousarray((arr.reshape(-1, 8) @ H8_MATRIX.T).reshape(-1), dtype=np.float32)


def encode_matrix_opt4_v2(
    weights_2d: np.ndarray,
    codebook_2048x8: np.ndarray,
    apply_fwht8: bool = True,
) -> Tuple[bytes, Dict[str, float]]:
    """
    Vectorized Option 4 (132-byte Dual-Stage RVQ) encoder compatible with Neon-Runner.
    Matches `[fp16 s1][fp16 s2][32 x uint16 idx1][32 x uint16 idx2]` per 256 weights.
    """
    w_mat = np.asarray(weights_2d, dtype=np.float32)
    if w_mat.ndim == 1:
        if (w_mat.size % WEIGHTS_PER_GLYPH) != 0:
            raise ValueError(f"1D weight count {w_mat.size} violates R = 0 mod 256")
        w_mat = w_mat.reshape(1, -1)
    d_out, d_in = w_mat.shape
    if (d_in % WEIGHTS_PER_GLYPH) != 0:
        raise ValueError(f"d_in={d_in} violates Zero-Remainder Census Law (R = {d_in % 256})")
    if codebook_2048x8.shape != (2048, 8):
        raise ValueError(f"Codebook must have shape (2048, 8), got {codebook_2048x8.shape}")

    cb = np.ascontiguousarray(codebook_2048x8, dtype=np.float32)
    n_glyphs = (d_out * d_in) // WEIGHTS_PER_GLYPH
    blocks = w_mat.reshape(n_glyphs, 32, 8)
    target = (blocks @ H8_MATRIX.T) if apply_fwht8 else blocks

    # Stage 1: vectorized centroid selection across all (n_glyphs * 32) octets
    flat_octs = target.reshape(-1, 8)
    idx1 = np.argmax(flat_octs @ cb.T, axis=1).astype(np.uint16).reshape(n_glyphs, 32)
    c1 = cb[idx1]  # (n_glyphs, 32, 8)

    denom1 = np.sum(c1 * c1, axis=(1, 2)) + 1e-12
    s1_f16 = (np.sum(target * c1, axis=(1, 2)) / denom1).astype(np.float16)
    s1_f32 = s1_f16.astype(np.float32)[:, None, None]
    recon1 = s1_f32 * c1
    residual = target - recon1

    # Stage 2: vectorized residual centroid selection
    idx2 = np.argmax(residual.reshape(-1, 8) @ cb.T, axis=1).astype(np.uint16).reshape(n_glyphs, 32)
    c2 = cb[idx2]
    denom2 = np.sum(c2 * c2, axis=(1, 2)) + 1e-12
    s2_f16 = (np.sum(residual * c2, axis=(1, 2)) / denom2).astype(np.float16)
    s2_f32 = s2_f16.astype(np.float32)[:, None, None]

    recon_rot = recon1 + s2_f32 * c2
    recon_real = (recon_rot @ H8_MATRIX.T) if apply_fwht8 else recon_rot

    orig_flat = w_mat.reshape(-1)
    rec_flat = recon_real.reshape(-1)
    cos = float(np.dot(orig_flat, rec_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(rec_flat) + 1e-12))
    sig_p = float(np.sum(orig_flat * orig_flat)) + 1e-12
    err_p = float(np.sum((orig_flat - rec_flat) ** 2)) + 1e-12
    snr_db = 10.0 * math.log10(sig_p / err_p)

    # Pack structured 132-byte glyph records
    glyph_dtype = np.dtype([
        ("s1", "<f2"),
        ("s2", "<f2"),
        ("idx1", "<u2", (32,)),
        ("idx2", "<u2", (32,)),
    ])
    assert glyph_dtype.itemsize == BPG_OPT4
    records = np.empty(n_glyphs, dtype=glyph_dtype)
    records["s1"] = s1_f16
    records["s2"] = s2_f16
    records["idx1"] = idx1
    records["idx2"] = idx2
    payload = records.tobytes()
    assert len(payload) == n_glyphs * BPG_OPT4

    return payload, {
        "glyphs": float(n_glyphs),
        "bytes": float(len(payload)),
        "cosine_similarity": cos,
        "snr_db": snr_db,
    }


def encode_matrix_opt5_v2(
    weights_2d: np.ndarray,
    outlier_preserved: bool = True,
) -> Tuple[bytes, Dict[str, float]]:
    """
    Vectorized Option 5 (36B standard or 48B Top-4 outlier-preserved) encoder.
    """
    w_mat = np.asarray(weights_2d, dtype=np.float32)
    if w_mat.ndim == 1:
        if (w_mat.size % WEIGHTS_PER_GLYPH) != 0:
            raise ValueError(f"1D weight count {w_mat.size} violates R = 0 mod 256")
        w_mat = w_mat.reshape(1, -1)
    d_out, d_in = w_mat.shape
    if (d_in % WEIGHTS_PER_GLYPH) != 0:
        raise ValueError(f"d_in={d_in} violates Zero-Remainder Census Law (R = {d_in % 256})")

    bpg = BPG_OPT5_OUTLIER if outlier_preserved else BPG_OPT5_STD
    n_glyphs = (d_out * d_in) // WEIGHTS_PER_GLYPH
    blocks = w_mat.reshape(n_glyphs, 256)

    bias_f16 = np.mean(blocks, axis=1).astype(np.float16)
    bias_f32 = bias_f16.astype(np.float32)[:, None]
    centered = blocks - bias_f32
    scale_f16 = np.mean(np.abs(centered), axis=1).astype(np.float16)
    scale_f32 = scale_f16.astype(np.float32)[:, None]

    bits = (centered >= 0.0).astype(np.uint8)
    packed_bits = np.packbits(bits, axis=1)  # (n_glyphs, 32)
    recon = np.where(bits == 1, scale_f32, -scale_f32) + bias_f32

    if outlier_preserved:
        residual = blocks - recon
        top4_idx = np.argpartition(np.abs(residual), -4, axis=1)[:, -4:].astype(np.uint8)
        rows_idx = np.arange(n_glyphs)[:, None]
        top4_res_f16 = residual[rows_idx, top4_idx].astype(np.float16)
        recon[rows_idx, top4_idx] += top4_res_f16.astype(np.float32)

        glyph_dtype = np.dtype([
            ("scale", "<f2"),
            ("bias", "<f2"),
            ("bitmask", "u1", (32,)),
            ("outlier_idx", "u1", (4,)),
            ("outlier_res", "<f2", (4,)),
        ])
        assert glyph_dtype.itemsize == BPG_OPT5_OUTLIER
        rec = np.empty(n_glyphs, dtype=glyph_dtype)
        rec["scale"] = scale_f16
        rec["bias"] = bias_f16
        rec["bitmask"] = packed_bits
        rec["outlier_idx"] = top4_idx
        rec["outlier_res"] = top4_res_f16
        payload = rec.tobytes()
    else:
        glyph_dtype = np.dtype([
            ("scale", "<f2"),
            ("bias", "<f2"),
            ("bitmask", "u1", (32,)),
        ])
        assert glyph_dtype.itemsize == BPG_OPT5_STD
        rec = np.empty(n_glyphs, dtype=glyph_dtype)
        rec["scale"] = scale_f16
        rec["bias"] = bias_f16
        rec["bitmask"] = packed_bits
        payload = rec.tobytes()

    orig_flat = w_mat.reshape(-1)
    rec_flat = recon.reshape(-1)
    cos = float(np.dot(orig_flat, rec_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(rec_flat) + 1e-12))
    sig_p = float(np.sum(orig_flat * orig_flat)) + 1e-12
    err_p = float(np.sum((orig_flat - rec_flat) ** 2)) + 1e-12
    snr_db = 10.0 * math.log10(sig_p / err_p)

    assert len(payload) == n_glyphs * bpg
    return payload, {
        "glyphs": float(n_glyphs),
        "bytes": float(len(payload)),
        "cosine_similarity": cos,
        "snr_db": snr_db,
    }


class NeonStreamAccelerator:
    """
    Drop-in superset of `DannyB-bit/Neon-Runner`'s `NeonStreamAccelerator`.
    Works with both unmodified Neon-Runner v1 `.so` binaries and upgraded V2 `.so`/`.dll` binaries.
    """

    def __init__(self, so_path: Optional[str] = None, auto_build: bool = True):
        resolved = self._resolve_library(so_path, auto_build=auto_build)
        self.so_path = str(resolved)
        self.lib = ctypes.CDLL(self.so_path)
        self._bind_signatures()

    @staticmethod
    def _resolve_library(explicit_path: Optional[str], auto_build: bool) -> Path:
        if explicit_path:
            p = Path(explicit_path)
            if not p.is_file():
                raise FileNotFoundError(f"Requested Neon-Runner shared library not found: {p}")
            return p.resolve()

        here = Path(__file__).resolve().parent
        neon_env = os.environ.get("NEON_RUNNER_DIR")
        so_env = os.environ.get("LANGU_GEMV_SO")

        candidates: List[Path] = []
        if so_env:
            candidates.append(Path(so_env))
        if neon_env:
            candidates.extend([
                Path(neon_env) / "libneon_stream_accelerator.so",
                Path(neon_env) / "neon_stream_accelerator.dll",
            ])
        candidates.extend([
            here / "neon_stream_accelerator.dll",
            here / "libneon_stream_accelerator.so",
            here / "libneon_stream_accelerator.dylib",
            Path("/home/astronautshe/libneon_stream_accelerator.so"),
        ])

        for cand in candidates:
            if cand.is_file():
                return cand.resolve()

        if auto_build:
            return build_shared_library(here).resolve()

        raise FileNotFoundError(
            "Neon-Runner shared library not found. Run `python neon_runner/build_native.py` or `make -C neon_runner`."
        )

    def _bind_signatures(self) -> None:
        # Neon-Runner v1 core symbols (mandatory)
        self.lib.decode_glyphs_c.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        self.lib.decode_glyphs_c.restype = None

        self.lib.benchmark_projection_neon.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_int
        ]
        self.lib.benchmark_projection_neon.restype = ctypes.c_double

        self.lib.gemv_opt4_mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        self.lib.gemv_opt4_mmap.restype = None

        self.has_opt4_multi = hasattr(self.lib, "gemv_opt4_mmap_multi")
        if self.has_opt4_multi:
            self.lib.gemv_opt4_mmap_multi.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self.lib.gemv_opt4_mmap_multi.restype = None

        self.lib.neon_accelerator_self_test.argtypes = []
        self.lib.neon_accelerator_self_test.restype = ctypes.c_int

        # V2 Dual-Road & Hadamard-Fused extensions (probed dynamically for v1 .so compatibility)
        self.has_v2_opt5 = hasattr(self.lib, "gemv_opt5_mmap") and hasattr(self.lib, "decode_glyphs_opt5_c")
        self.has_v2_opt5_multi = hasattr(self.lib, "gemv_opt5_mmap_multi")
        self.has_v2_fwht_gemv = hasattr(self.lib, "gemv_opt4_fwht_mmap")
        self.has_v2_opt4_mode = hasattr(self.lib, "decode_glyphs_opt4_mode_c")
        self.has_packed_opt4 = hasattr(self.lib, "decode_glyphs_opt4_packed_c") and hasattr(self.lib, "gemv_opt4_packed_mmap")
        if self.has_packed_opt4:
            self.lib.decode_glyphs_opt4_packed_c.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self.lib.decode_glyphs_opt4_packed_c.restype = None
            self.lib.gemv_opt4_packed_mmap.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ]
            self.lib.gemv_opt4_packed_mmap.restype = None

        if self.has_v2_opt5:
            self.lib.decode_glyphs_opt5_c.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            self.lib.decode_glyphs_opt5_c.restype = None
            self.lib.gemv_opt5_mmap.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            self.lib.gemv_opt5_mmap.restype = None

        if self.has_v2_opt5_multi:
            self.lib.gemv_opt5_mmap_multi.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self.lib.gemv_opt5_mmap_multi.restype = None

        if self.has_v2_fwht_gemv:
            self.lib.gemv_opt4_fwht_mmap.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
            ]
            self.lib.gemv_opt4_fwht_mmap.restype = None

        if self.has_v2_opt4_mode:
            self.lib.decode_glyphs_opt4_mode_c.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            self.lib.decode_glyphs_opt4_mode_c.restype = None

        # V2.1 GlyphAttentionV2 extensions (O(1) online KV encode + Flash-Glyph GQA)
        self.has_v2_glyph_attn = (
            hasattr(self.lib, "encode_kv_glyph_fast_c")
            and hasattr(self.lib, "flash_glyph_attention_gqa_c")
            and hasattr(self.lib, "build_ternary_lut_c")
        )
        self.has_v2_opt5_mode = hasattr(self.lib, "decode_glyphs_opt5_mode_c")

        if self.has_v2_opt5_mode:
            self.lib.decode_glyphs_opt5_mode_c.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            self.lib.decode_glyphs_opt5_mode_c.restype = None

        if self.has_v2_glyph_attn:
            self.lib.build_ternary_lut_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.lib.build_ternary_lut_c.restype = ctypes.c_int

            self.lib.encode_kv_glyph_fast_c.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self.lib.encode_kv_glyph_fast_c.restype = None

            self.lib.flash_glyph_attention_gqa_c.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self.lib.flash_glyph_attention_gqa_c.restype = ctypes.c_int

        self.has_v2_paged_bf16 = hasattr(self.lib, "paged_bf16_attention_gqa_c")
        if self.has_v2_paged_bf16:
            self.lib.paged_bf16_attention_gqa_c.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ]
            self.lib.paged_bf16_attention_gqa_c.restype = ctypes.c_int

    def self_test(self) -> bool:
        return bool(self.lib.neon_accelerator_self_test() == 1)

    def benchmark_projection(
        self,
        mmap_ptr: int,
        byte_offset: int,
        n_glyphs: int,
        codebook_fp32: np.ndarray,
        num_threads: int = 4,
    ) -> float:
        cb = np.ascontiguousarray(codebook_fp32, dtype=np.float32)
        assert cb.shape == (2048, 8)
        return float(
            self.lib.benchmark_projection_neon(
                ctypes.c_void_p(mmap_ptr),
                byte_offset,
                n_glyphs,
                cb.ctypes.data_as(ctypes.c_void_p),
                num_threads,
            )
        )

    def decode(
        self,
        glyph_stream_bytes: Any,
        codebook_fp32: np.ndarray,
        num_threads: int = 4,
        apply_fwht8: bool = True,
    ) -> np.ndarray:
        cb = np.ascontiguousarray(codebook_fp32, dtype=np.float32)
        assert cb.shape == (2048, 8)
        raw_arr = np.frombuffer(glyph_stream_bytes, dtype=np.uint8)
        if (raw_arr.size % BPG_OPT4) != 0:
            raise ValueError(f"Option 4 buffer length {raw_arr.size} is not a multiple of {BPG_OPT4}")
        n_glyphs = raw_arr.size // BPG_OPT4
        out = np.empty(n_glyphs * WEIGHTS_PER_GLYPH, dtype=np.float32)

        in_ptr = raw_arr.ctypes.data_as(ctypes.c_void_p)
        cb_ptr = cb.ctypes.data_as(ctypes.c_void_p)
        out_ptr = out.ctypes.data_as(ctypes.c_void_p)

        if apply_fwht8:
            self.lib.decode_glyphs_c(in_ptr, cb_ptr, out_ptr, n_glyphs, num_threads)
        elif self.has_v2_opt4_mode:
            self.lib.decode_glyphs_opt4_mode_c(in_ptr, cb_ptr, out_ptr, n_glyphs, 0, num_threads)
        else:
            # v1 library always applies FWHT-8; invert self-inverse FWHT-8 to recover direct RVQ weights
            self.lib.decode_glyphs_c(in_ptr, cb_ptr, out_ptr, n_glyphs, num_threads)
            out = fwht8_vec_numpy(out)
        return out

    def decode_opt5(
        self,
        glyph_stream_bytes: Any,
        bytes_per_glyph: int = 48,
        num_threads: int = 4,
        apply_fwht8: bool = False,
    ) -> np.ndarray:
        bpg = 48 if bytes_per_glyph == 48 else 36
        raw_arr = np.frombuffer(glyph_stream_bytes, dtype=np.uint8)
        if (raw_arr.size % bpg) != 0:
            raise ValueError(f"Option 5 buffer length {raw_arr.size} is not a multiple of {bpg}")
        n_glyphs = raw_arr.size // bpg

        if apply_fwht8 and self.has_v2_opt5_mode:
            out = np.empty(n_glyphs * WEIGHTS_PER_GLYPH, dtype=np.float32)
            self.lib.decode_glyphs_opt5_mode_c(
                raw_arr.ctypes.data_as(ctypes.c_void_p),
                out.ctypes.data_as(ctypes.c_void_p),
                n_glyphs,
                bpg,
                1,
                num_threads,
            )
            return out

        if self.has_v2_opt5:
            out = np.empty(n_glyphs * WEIGHTS_PER_GLYPH, dtype=np.float32)
            self.lib.decode_glyphs_opt5_c(
                raw_arr.ctypes.data_as(ctypes.c_void_p),
                out.ctypes.data_as(ctypes.c_void_p),
                n_glyphs,
                bpg,
                num_threads,
            )
            return fwht8_vec_numpy(out) if apply_fwht8 else out

        # Vectorized fallback when paired with an unmodified Neon-Runner v1 .so
        rows = raw_arr.reshape(n_glyphs, bpg)
        scales = rows[:, 0:2].copy().view("<f2").astype(np.float32)
        biases = rows[:, 2:4].copy().view("<f2").astype(np.float32)
        bits = np.unpackbits(rows[:, 4:36], axis=1)
        out_2d = np.where(bits == 1, scales, -scales) + biases
        if bpg == 48:
            oi = rows[:, 36:40]
            ores = rows[:, 40:48].copy().view("<f2").astype(np.float32)
            for col in range(4):
                np.add.at(out_2d, (np.arange(n_glyphs), oi[:, col]), ores[:, col])
        flat = out_2d.reshape(-1).astype(np.float32)
        return fwht8_vec_numpy(flat) if apply_fwht8 else flat

    def decode_packed_opt4(self, glyph_bytes, codebook_fp32, *, apply_fwht8=True, num_threads=4):
        """Decode 92-byte records using the upgraded native ABI; never misread as 132B."""
        if not self.has_packed_opt4:
            raise RuntimeError("Rebuild the native library for 92-byte packed Opt4 support")
        raw = np.frombuffer(glyph_bytes, dtype=np.uint8)
        if raw.size % BPG_OPT4_PACKED or raw.size // BPG_OPT4_PACKED > 2**31 - 1:
            raise ValueError("Packed Opt4 buffer length is invalid")
        cb = np.ascontiguousarray(codebook_fp32, dtype=np.float32)
        if cb.shape != (2048, 8):
            raise ValueError("Codebook must have shape (2048, 8)")
        count = raw.size // BPG_OPT4_PACKED
        out = np.empty(count * 256, dtype=np.float32)
        self.lib.decode_glyphs_opt4_packed_c(
            raw.ctypes.data_as(ctypes.c_void_p), cb.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p), count, int(apply_fwht8), num_threads,
        )
        return out

    def build_ternary_lut(self, codebook_fp32: np.ndarray) -> Optional[np.ndarray]:
        """Returns a 6,561-entry uint16 LUT for O(1) ternary lattice RVQ encoding, or None if non-ternary."""
        if not self.has_v2_glyph_attn:
            return None
        cb = np.ascontiguousarray(codebook_fp32, dtype=np.float32)
        assert cb.shape == (2048, 8)
        lut = np.empty(6561, dtype=np.uint16)
        ok = self.lib.build_ternary_lut_c(
            cb.ctypes.data_as(ctypes.c_void_p),
            lut.ctypes.data_as(ctypes.c_void_p),
        )
        return lut if ok == 1 else None


class NeonRunnerBridgeV2:
    """
    End-to-End Stream Execution Bridge between Language-U-Gemma4 (E4B & 31B)
    and Neon-Runner (`NeonStreamAccelerator`).
    """

    def __init__(
        self,
        stream_path: Optional[str] = None,
        layout_path: Optional[str] = None,
        codebook_family: str = "compass_s7",
        codebook_path: Optional[str] = None,
        so_path: Optional[str] = None,
        verify_size_law: bool = True,
    ):
        self.accelerator = NeonStreamAccelerator(so_path=so_path, auto_build=True)
        if not self.accelerator.self_test():
            raise RuntimeError(f"Native Neon-Runner library failed self-test: {self.accelerator.so_path}")

        self.stream_path: Optional[str] = stream_path or os.environ.get("LANGU_STREAM_PATH")
        self.layout_path: Optional[str] = layout_path or os.environ.get("LANGU_LAYOUT_PATH")
        self.layout: Dict[str, Any] = {}
        self.projections_by_key: Dict[str, Dict[str, Any]] = {}
        self._memmap: Optional[np.memmap] = None
        self._mmap_ptr: int = 0
        self.header: Optional[StreamHeaderV2] = None

        if self.layout_path:
            lp = Path(self.layout_path)
            if not lp.is_file():
                raise FileNotFoundError(f"Layout JSON missing: {lp}")
            with open(lp, "r", encoding="utf-8") as f:
                self.layout = json.load(f)
            if codebook_family == "auto":
                codebook_family = self.layout.get("codebook_family", "auto" if codebook_path else "compass_s7")

        self.codebook: LoadedCodebookV2 = CodebookRegistryV2.resolve(
            family=codebook_family if (codebook_family != "auto" or codebook_path) else "compass_s7",
            explicit_path=codebook_path,
        )

        if self.stream_path:
            self._open_stream(self.stream_path, verify_size_law=verify_size_law)

    def _open_stream(self, path: str, verify_size_law: bool = True) -> None:
        sp = Path(path)
        if not sp.is_file():
            raise FileNotFoundError(f"Language-U stream missing: {sp}")
        actual_size = sp.stat().st_size
        if actual_size < LANU_HEADER_BYTES:
            raise ValueError(f"Stream '{sp}' truncated ({actual_size} B < {LANU_HEADER_BYTES} B)")

        self._memmap = np.memmap(str(sp), dtype=np.uint8, mode="r")
        self._mmap_ptr = int(self._memmap.ctypes.data_as(ctypes.c_void_p).value or 0)
        self.header = parse_lanu_header(bytes(self._memmap[:LANU_HEADER_BYTES]))

        if self.layout:
            if verify_size_law:
                try:
                    checked_layout(sp, self.layout_path)
                except Exception:
                    self.close()
                    raise
            projs = self.layout.get("projections", [])
            expected_bytes = LANU_HEADER_BYTES + sum(int(p["tensor_bytes"]) for p in projs)
            if verify_size_law and actual_size != expected_bytes:
                self.close()
                raise ValueError(
                    f"Stream Size Law violated on '{sp}': physical={actual_size:,} B, "
                    f"expected={expected_bytes:,} B"
                )
            for p in projs:
                key = p.get("full_key") or p.get("tensor_name") or p.get("name")
                if key:
                    self.projections_by_key[key] = p

    def gemv(
        self,
        proj_meta: Dict[str, Any],
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Executes fused zero-heap GEMV `y = W @ x` for either Option 4 (132B) or
        Option 5 (36B/48B) projections directly from the memory-mapped stream.
        """
        if self._memmap is None or self._mmap_ptr == 0:
            raise RuntimeError("No stream memory-mapped in NeonRunnerBridgeV2")

        validate_projection(proj_meta, self._memmap.size)

        d_in = int(proj_meta["d_in"])
        d_out = int(proj_meta["d_out"])
        offset = int(proj_meta.get("stream_byte_offset", proj_meta.get("byte_offset", LANU_HEADER_BYTES)))
        codec_id = int(proj_meta.get("codec_id", 4 if proj_meta.get("codec", "OPT4") == "OPT4" else 5))
        bpg = int(proj_meta.get("bytes_per_glyph", BPG_OPT4 if codec_id == 4 else BPG_OPT5_OUTLIER))
        rot_mode = proj_meta.get("rotation_mode", self.layout.get("rotation_mode", "direct"))
        if rot_mode not in ("fwht8", "none", "direct"):
            raise ValueError(f"Unsupported rotation_mode: {rot_mode}")

        x_vec = np.ascontiguousarray(x, dtype=np.float32)
        if x_vec.shape != (d_in,):
            raise ValueError(f"GEMV input shape mismatch: got {x_vec.shape}, expected ({d_in},)")
        y = np.empty(d_out, dtype=np.float32)
        assert y.shape == (d_out,)

        if codec_id == 4:
            cb_ptr = self.codebook.array.ctypes.data_as(ctypes.c_void_p)
            y_ptr = y.ctypes.data_as(ctypes.c_void_p)
            if bpg == BPG_OPT4_PACKED:
                if not self.accelerator.has_packed_opt4:
                    raise RuntimeError("Rebuild the native library for 92-byte packed Opt4 support")
                x_native = fwht8_vec_numpy(x_vec) if rot_mode == "fwht8" else x_vec
                self.accelerator.lib.gemv_opt4_packed_mmap(
                    ctypes.c_void_p(self._mmap_ptr), offset, cb_ptr,
                    x_native.ctypes.data_as(ctypes.c_void_p), y_ptr, d_out, d_in,
                )
                return y
            if rot_mode == "fwht8":
                if self.accelerator.has_v2_fwht_gemv:
                    self.accelerator.lib.gemv_opt4_fwht_mmap(
                        ctypes.c_void_p(self._mmap_ptr),
                        offset,
                        cb_ptr,
                        x_vec.ctypes.data_as(ctypes.c_void_p),
                        y_ptr,
                        d_out,
                        d_in,
                    )
                else:
                    # Unmodified Neon-Runner v1 .so path: pre-rotate x once in O(d_in)
                    x_rot = fwht8_vec_numpy(x_vec)
                    self.accelerator.lib.gemv_opt4_mmap(
                        ctypes.c_void_p(self._mmap_ptr),
                        offset,
                        cb_ptr,
                        x_rot.ctypes.data_as(ctypes.c_void_p),
                        y_ptr,
                        d_out,
                        d_in,
                    )
            else:
                self.accelerator.lib.gemv_opt4_mmap(
                    ctypes.c_void_p(self._mmap_ptr),
                    offset,
                    cb_ptr,
                    x_vec.ctypes.data_as(ctypes.c_void_p),
                    y_ptr,
                    d_out,
                    d_in,
                )
            return y

        # Codec 5 (Option 5: 36B or 48B Top-4 outlier)
        # H8 is self-adjoint: decoded H8 weights require H8 input here as well.
        # The previous bridge silently ignored rotation_mode for Option 5 GEMV.
        x_native = fwht8_vec_numpy(x_vec) if rot_mode == "fwht8" else x_vec
        if self.accelerator.has_v2_opt5:
            self.accelerator.lib.gemv_opt5_mmap(
                ctypes.c_void_p(self._mmap_ptr),
                offset,
                x_native.ctypes.data_as(ctypes.c_void_p),
                y.ctypes.data_as(ctypes.c_void_p),
                d_out,
                d_in,
                bpg,
            )
            return y

        # Unmodified Neon-Runner v1 .so fallback for Option 5
        tensor_bytes = int(proj_meta.get("tensor_bytes", (d_out * d_in // 256) * bpg))
        raw_slice = self._memmap[offset : offset + tensor_bytes]
        w_mat = self.accelerator.decode_opt5(raw_slice, bytes_per_glyph=bpg).reshape(d_out, d_in)
        return (w_mat @ x_native).astype(np.float32)

    def close(self) -> None:
        if self._memmap is not None:
            mm_obj = getattr(self._memmap, "_mmap", None)
            self._memmap = None
            self._mmap_ptr = 0
            if mm_obj is not None:
                mm_obj.close()

    def __enter__(self) -> "NeonRunnerBridgeV2":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _self_test() -> int:
    """
    Mandatory 6-Gate Self-Verification Battery (Fail-Closed).
    Raises SystemExit(failures) if any gate fails.
    """
    import tempfile

    print("=" * 80)
    print(f"{BRIDGE_COMPONENT_ID} (v{BRIDGE_VERSION}) -- 6-GATE PHYSICAL SILICON BATTERY")
    print("=" * 80)
    failures = 0

    def record_gate(gate_num: int, title: str, ok: bool, detail: str) -> None:
        nonlocal failures
        tag = "PASS" if ok else "FAIL"
        print(f"[Gate {gate_num}] [{tag}] {title} -- {detail}")
        if not ok:
            failures += 1

    here = Path(__file__).resolve().parent
    repo_root = here.parent
    sample_opt5_path = repo_root / "model" / "gemma4_e4b_sample_layer0_q_proj.glyph.bin"

    # -------------------------------------------------------------------------
    # Gate 1: Stream Integrity, Header ABI & Lying-Header Fail-Closed Rejection
    # -------------------------------------------------------------------------
    try:
        acc = NeonStreamAccelerator(auto_build=True)
        valid_hdr = parse_lanu_header(pack_lanu_header(total_glyphs=100, boundary_glyphs=40))
        lying_caught = False
        try:
            parse_lanu_header(struct.pack("<4sHHII", b"LANU", 2, 45, 10, 50))
        except ValueError:
            lying_caught = True
        bad_magic_caught = False
        try:
            parse_lanu_header(struct.pack("<4sHHII", b"FAKE", 2, 45, 10, 5))
        except ValueError:
            bad_magic_caught = True
        g1_ok = acc.self_test() and (valid_hdr.total_glyphs == 100) and lying_caught and bad_magic_caught
        record_gate(1, "Stream Header ABI & Lying-Header Gate", g1_ok, f"lib={Path(acc.so_path).name}")
    except Exception as exc:
        record_gate(1, "Stream Header ABI & Lying-Header Gate", False, repr(exc))
        raise SystemExit(failures)

    # -------------------------------------------------------------------------
    # Gate 2: Codebook Decode Sanity & Physical Gemma 4 Weight Variance
    # -------------------------------------------------------------------------
    try:
        cb_compass = CodebookRegistryV2.resolve("compass_s7")
        cb_trit = CodebookRegistryV2.resolve("trit_rvq")
        if not sample_opt5_path.is_file():
            raise FileNotFoundError(f"Physical Gemma 4 E4B sample missing: {sample_opt5_path}")
        raw_opt5 = sample_opt5_path.read_bytes()
        decoded_e4b_w = acc.decode_opt5(raw_opt5, bytes_per_glyph=48, num_threads=2)
        w_std = float(np.std(decoded_e4b_w))
        g2_ok = (
            cb_compass.sentinel_max_diff < 1e-4
            and cb_trit.sentinel_max_diff < 1e-4
            and decoded_e4b_w.shape == (4096,)
            and 0.001 < w_std < 10.0
            and bool(np.all(np.isfinite(decoded_e4b_w)))
        )
        record_gate(
            2,
            "Codebook Sentinels & Physical Weight Decode Sanity",
            g2_ok,
            f"compass_diff={cb_compass.sentinel_max_diff:.2e}, trit_diff={cb_trit.sentinel_max_diff:.2e}, "
            f"e4b_qproj_std={w_std:.5f}",
        )
    except Exception as exc:
        record_gate(2, "Codebook Sentinels & Physical Weight Decode Sanity", False, repr(exc))

    # -------------------------------------------------------------------------
    # Gate 3: Row / Tensor Distinctness (Non-Rank-1 Geometry)
    # -------------------------------------------------------------------------
    try:
        # Reshape physical 4096 Gemma 4 E4B Q-proj weights as (8, 512) matrix (2 glyphs/row)
        w_rows = decoded_e4b_w.reshape(8, 512)
        norms = np.linalg.norm(w_rows, axis=1, keepdims=True) + 1e-12
        normed = w_rows / norms
        gram = np.abs(normed @ normed.T)
        off_diag_max = float(np.max(gram - np.eye(8, dtype=np.float32)))
        g3_ok = off_diag_max < 0.99
        record_gate(3, "Row/Tensor Distinctness (Non-Rank-1)", g3_ok, f"max_off_diag_cos={off_diag_max:.4f}")
    except Exception as exc:
        record_gate(3, "Row/Tensor Distinctness (Non-Rank-1)", False, repr(exc))

    # -------------------------------------------------------------------------
    # Gate 4: Real-Space Semantic & Fused GEMV Parity (Opt4 FWHT + Opt4 Direct + Opt5)
    # -------------------------------------------------------------------------
    try:
        opt4_fwht_bytes, m_fwht = encode_matrix_opt4_v2(w_rows, cb_compass.array, apply_fwht8=True)
        opt4_dir_bytes, m_dir = encode_matrix_opt4_v2(w_rows, cb_compass.array, apply_fwht8=False)
        opt5_48_bytes, m_opt5 = encode_matrix_opt5_v2(w_rows, outlier_preserved=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            s_path = tmp_dir / "hybrid_e4b_test.lanu"
            l_path = tmp_dir / "hybrid_e4b_test.layout.json"

            off0 = LANU_HEADER_BYTES
            off1 = off0 + len(opt4_fwht_bytes)
            off2 = off1 + len(opt4_dir_bytes)
            total_g = 16 + 16 + 16
            hdr = pack_lanu_header(total_glyphs=total_g, boundary_glyphs=32)
            s_path.write_bytes(hdr + opt4_fwht_bytes + opt4_dir_bytes + opt5_48_bytes)

            layout_doc = {
                "model": "google/gemma-4-E4B-it",
                "codebook_family": "compass_s7",
                "projections": [
                    {
                        "layer": 0,
                        "tensor_name": "q_proj_opt4_fwht",
                        "full_key": "layer0.q_proj_opt4_fwht",
                        "codec": "OPT4",
                        "codec_id": 4,
                        "rotation_mode": "fwht8",
                        "bytes_per_glyph": 132,
                        "d_out": 8,
                        "d_in": 512,
                        "glyphs": 16,
                        "tensor_bytes": len(opt4_fwht_bytes),
                        "stream_byte_offset": off0,
                    },
                    {
                        "layer": 0,
                        "tensor_name": "q_proj_opt4_direct",
                        "full_key": "layer0.q_proj_opt4_direct",
                        "codec": "OPT4",
                        "codec_id": 4,
                        "rotation_mode": "direct",
                        "bytes_per_glyph": 132,
                        "d_out": 8,
                        "d_in": 512,
                        "glyphs": 16,
                        "tensor_bytes": len(opt4_dir_bytes),
                        "stream_byte_offset": off1,
                    },
                    {
                        "layer": 5,
                        "tensor_name": "q_proj_opt5_outlier",
                        "full_key": "layer5.q_proj_opt5_outlier",
                        "codec": "OPT5",
                        "codec_id": 5,
                        "bytes_per_glyph": 48,
                        "d_out": 8,
                        "d_in": 512,
                        "glyphs": 16,
                        "tensor_bytes": len(opt5_48_bytes),
                        "stream_byte_offset": off2,
                    },
                ],
            }
            l_path.write_text(json.dumps(layout_doc, indent=2), encoding="utf-8")

            x_query = (w_rows[0] / (np.linalg.norm(w_rows[0]) + 1e-12)).astype(np.float32)
            with NeonRunnerBridgeV2(str(s_path), str(l_path), codebook_family="compass_s7") as bridge:
                y_fwht = bridge.gemv(bridge.projections_by_key["layer0.q_proj_opt4_fwht"], x_query)
                y_dir = bridge.gemv(bridge.projections_by_key["layer0.q_proj_opt4_direct"], x_query)
                y_opt5 = bridge.gemv(bridge.projections_by_key["layer5.q_proj_opt5_outlier"], x_query)

                # Also verify unmodified Neon-Runner v1 .so dispatch path (only v1 symbols active)
                orig_fwht_flag = bridge.accelerator.has_v2_fwht_gemv
                orig_opt5_flag = bridge.accelerator.has_v2_opt5
                try:
                    bridge.accelerator.has_v2_fwht_gemv = False
                    bridge.accelerator.has_v2_opt5 = False
                    y_fwht_v1 = bridge.gemv(bridge.projections_by_key["layer0.q_proj_opt4_fwht"], x_query)
                    y_opt5_v1 = bridge.gemv(bridge.projections_by_key["layer5.q_proj_opt5_outlier"], x_query)
                finally:
                    bridge.accelerator.has_v2_fwht_gemv = orig_fwht_flag
                    bridge.accelerator.has_v2_opt5 = orig_opt5_flag

            w_dec_fwht = acc.decode(opt4_fwht_bytes, cb_compass.array, apply_fwht8=True).reshape(8, 512)
            w_dec_dir = acc.decode(opt4_dir_bytes, cb_compass.array, apply_fwht8=False).reshape(8, 512)
            w_dec_opt5 = acc.decode_opt5(opt5_48_bytes, bytes_per_glyph=48).reshape(8, 512)

            xq64 = x_query.astype(np.float64)
            err_fwht = float(np.max(np.abs(y_fwht.astype(np.float64) - (w_dec_fwht.astype(np.float64) @ xq64))))
            err_dir = float(np.max(np.abs(y_dir.astype(np.float64) - (w_dec_dir.astype(np.float64) @ xq64))))
            err_opt5 = float(np.max(np.abs(y_opt5.astype(np.float64) - (w_dec_opt5.astype(np.float64) @ xq64))))
            err_v1_compat = max(
                float(np.max(np.abs(y_fwht_v1 - y_fwht))),
                float(np.max(np.abs(y_opt5_v1 - y_opt5))),
            )
            semantic_peak_ok = int(np.argmax(y_fwht)) == 0 and int(np.argmax(y_opt5)) == 0

            g4_ok = (
                err_fwht < 1e-4
                and err_dir < 1e-4
                and err_opt5 < 1e-4
                and err_v1_compat < 1e-4
                and m_fwht["cosine_similarity"] > 0.90
                and semantic_peak_ok
            )
            record_gate(
                4,
                "Fused GEMV vs Decode Parity (V2 + Unmodified Neon-Runner v1 ABI)",
                g4_ok,
                f"err_fwht={err_fwht:.2e}, err_v1={err_v1_compat:.2e}, err_opt5={err_opt5:.2e}, "
                f"opt4_cos={m_fwht['cosine_similarity']:.4f} ({m_fwht['snr_db']:.2f} dB)",
            )
    except Exception as exc:
        record_gate(4, "Fused GEMV vs Decode Parity & Semantic Alignment", False, repr(exc))

    # -------------------------------------------------------------------------
    # Gate 5: Exact Census & Boundary Accounting (Gemma 4 E4B + Gemma 4 31B)
    # -------------------------------------------------------------------------
    try:
        from gemma4_e4b.consider_trinity_e4b_core import verify_gemma4_e4b_zero_remainder

        e4b_manifest = verify_gemma4_e4b_zero_remainder()
        e4b_ok = (
            e4b_manifest["all_remainders_zero"] is True
            and e4b_manifest["total_discrete_glyphs"] == 28_715_008
            and e4b_manifest["total_discrete_glyphs"] * 256 == e4b_manifest["total_active_weights"]
        )
        # Verify physical 31B layout + stream if present in parent workspace
        ws_31b_layout = repo_root.parent / "gemma4_31b_tensor_layout.json"
        ws_31b_stream = repo_root.parent / "gemma4_31b_language_u_sandwich_stream.bin"
        ws_31b_cb = repo_root.parent / "s7_codebook_2048x8.bin"
        detail_31b = "E4B census verified"
        ok_31b = True
        if ws_31b_layout.is_file() and ws_31b_stream.is_file() and ws_31b_cb.is_file():
            with NeonRunnerBridgeV2(
                stream_path=str(ws_31b_stream),
                layout_path=str(ws_31b_layout),
                codebook_family="golden_s7",
                codebook_path=str(ws_31b_cb),
            ) as b31:
                x31 = np.linspace(-0.5, 0.5, 5376, dtype=np.float32)
                p_l0 = b31.layout["projections"][0]   # Layer 0 OPT4 (8192 x 5376)
                p_l5 = b31.layout["projections"][33]  # Layer 5 OPT5 (16384 x 5376)
                y_l0 = b31.gemv(p_l0, x31)
                y_l5 = b31.gemv(p_l5, x31)
                ok_31b = (
                    len(b31.layout["projections"]) == 410
                    and b31.header is not None
                    and b31.header.total_glyphs == 114_401_280
                    and y_l0.shape == (p_l0["d_out"],)
                    and y_l5.shape == (p_l5["d_out"],)
                    and bool(np.all(np.isfinite(y_l0)))
                    and bool(np.all(np.isfinite(y_l5)))
                )
                detail_31b = (
                    f"E4B={e4b_manifest['total_discrete_glyphs']:,} glyphs + "
                    f"Physical 31B=114,401,280 glyphs (L0 OPT4 + L5 OPT5 live GEMV OK)"
                )
        record_gate(5, "Exact Census & Boundary Accounting (E4B + 31B)", e4b_ok and ok_31b, detail_31b)
    except Exception as exc:
        record_gate(5, "Exact Census & Boundary Accounting (E4B + 31B)", False, repr(exc))

    # -------------------------------------------------------------------------
    # Gate 6: End-to-End Throughput & Clean Lifecycle Execution Without Exceptions
    # -------------------------------------------------------------------------
    try:
        # Benchmark 4,096 Option 4 glyphs (1,048,576 FP32 parameters) via benchmark_projection_neon
        bench_payload = opt4_fwht_bytes * 256  # 16 * 256 = 4,096 glyphs = 540,672 bytes
        raw_buf = np.frombuffer(bench_payload, dtype=np.uint8)
        ptr = int(raw_buf.ctypes.data_as(ctypes.c_void_p).value or 0)
        _ = acc.benchmark_projection(ptr, 0, 4096, cb_compass.array, num_threads=4)
        elapsed = acc.benchmark_projection(ptr, 0, 4096, cb_compass.array, num_threads=4)
        g_per_s = 4096.0 / max(elapsed, 1e-9)
        g6_ok = g_per_s > 50_000.0
        record_gate(
            6,
            "End-to-End Native Throughput & Clean Resource Lifecycle",
            g6_ok,
            f"{g_per_s:,.0f} glyphs/s ({g_per_s * 256.0 / 1e6:.1f} M params/s)",
        )
    except Exception as exc:
        record_gate(6, "End-to-End Native Throughput & Clean Resource Lifecycle", False, repr(exc))

    print("=" * 80)
    if failures > 0:
        print(f"VERDICT: FAILED ({failures} gate(s) failed)")
        raise SystemExit(failures)
    print("VERDICT: ALL 6 FAIL-CLOSED GATES PASSED ON PHYSICAL SILICON")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{BRIDGE_COMPONENT_ID} CLI")
    parser.add_argument("--self-test", action="store_true", help="Run the 6-gate self-verification battery")
    parser.add_argument("--stream", type=str, default=None, help="Path to .lanu / .bin glyph stream")
    parser.add_argument("--layout", type=str, default=None, help="Path to tensor_layout.json")
    parser.add_argument("--codebook", type=str, default="compass_s7", help="compass_s7 | trit_rvq | golden_s7 | auto")
    parser.add_argument("--codebook-path", type=str, default=None, help="Explicit path to 64 KB codebook .bin")
    parser.add_argument("--lib", type=str, default=None, help="Explicit path to libneon_stream_accelerator")
    parser.add_argument("--bench-projections", type=int, default=4, help="Number of OPT4 projections to benchmark")
    args = parser.parse_args()

    if args.self_test or not args.stream:
        return _self_test()

    with NeonRunnerBridgeV2(
        stream_path=args.stream,
        layout_path=args.layout,
        codebook_family=args.codebook,
        codebook_path=args.codebook_path,
        so_path=args.lib,
    ) as bridge:
        print(f"[+] Opened stream: {args.stream} (header={bridge.header})")
        print(f"[+] Codebook: {bridge.codebook.family} (sentinel_diff={bridge.codebook.sentinel_max_diff:.2e})")
        projs = [p for p in bridge.layout.get("projections", []) if p.get("codec", "OPT4") == "OPT4"]
        for p in projs[: args.bench_projections]:
            dt = bridge.accelerator.benchmark_projection(
                bridge._mmap_ptr,
                int(p["stream_byte_offset"]),
                int(p["glyphs"]),
                bridge.codebook.array,
                num_threads=4,
            )
            gps = int(p["glyphs"]) / max(dt, 1e-9)
            print(
                f"  L{p.get('layer', 0):02d} {p.get('tensor_name', p.get('name', 'proj')):<30} | "
                f"{int(p['glyphs']):8,d} glyphs | {dt * 1000.0:6.2f} ms | {gps:,.0f} glyphs/s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

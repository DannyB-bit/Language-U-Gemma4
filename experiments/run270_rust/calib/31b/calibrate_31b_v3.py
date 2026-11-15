#!/usr/bin/env python3
"""
V3 Sensitivity-Weighted Mixed-Precision Calibration for Gemma 4 31B (`.t16c`).

Architecture & Recipe:
  - Balanced SmoothQuant (alpha=0.5, bos_weight=0.15) + blockwise FWHT (bsize=256)
  - Fast RTN bit-serial / 5-trit ternary packing (`fast_quant.c`)
  - Gram-eigh + ActALS Int8LowRank (`rank=16..32`) fitted on balanced tokens
  - Sensitivity-weighted knapsack allocation within `--budget-bytes` (`<= 6.38 GB`)
  - Omitted `v_proj` on full-attention layers (`k_eq_v = true`)

Run with no arguments or `--selftest` to execute the 6-gate offline battery.
"""

import os

# Enforce single-threaded OpenBLAS BEFORE importing numpy so Gram eigh/solve
# never contend with outer 4-thread OpenMP GEMM/packing loops on Cortex-A72.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import ctypes
import gc
import json
import math
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pack_t16c_v3 import (
    MODE_TO_T16C,
    SafetensorsReader,
    assemble_t16c_v3,
    resolve_safetensors_path,
    tile16_blk_bytes,
)

# Disjoint calibration prompts (strictly separate from held-out eval prompts)
DEFAULT_CAL_PROMPTS = [
    "Explain why the sky appears blue during a clear afternoon in one concise paragraph.",
    "Write a clean Python function that returns the n-th Fibonacci number iteratively.",
    "Summarize the water cycle including evaporation, condensation, and precipitation.",
    "List three primary differences between TCP and UDP in computer networking.",
    "Translate the phrase 'Good morning, how are you today?' into French and Spanish.",
    "Solve step by step: If a train travels at 60 km/h for 2.5 hours, how far does it go?",
]


def _rss_mb() -> float:
    """Return current Linux VmRSS in MiB, or 0.0 on non-procfs hosts."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


class _NativeCalOps:
    """Lazy wrapper around compiled C/NEON helpers with NumPy fallback for selftests."""

    def __init__(self, work_dir: Optional[Path] = None, require_native: bool = True) -> None:
        self.lib_g = None
        self.lfast = None
        self.lfq = None
        self.lib_n = None
        if work_dir is None:
            return
        work_dir.mkdir(parents=True, exist_ok=True)
        arch = ["-march=armv8-a+simd", "-mcpu=cortex-a72"] if platform.machine().lower() in ("aarch64", "arm64") else ["-march=native"]
        neon_c = SCRIPT_DIR.parents[2] / "neon_runner" / "neon_stream_accelerator_v2.c"
        specs = [
            ("libgptq_fast6.so", SCRIPT_DIR / "gptq_fast6.c", []),
            ("libfast_sgemm.so", SCRIPT_DIR / "fast_sgemm.c", []),
            ("libfast_quant.so", SCRIPT_DIR / "fast_quant.c", []),
            ("libneon_stream_accelerator.so", neon_c, ["-DNEON_SHARED_LIB_ONLY"]),
        ]
        for so_name, c_path, extra_defs in specs:
            so_path = work_dir / so_name
            if not so_path.exists():
                if not c_path.exists():
                    if require_native:
                        raise FileNotFoundError(f"Missing C source {c_path}")
                    return
                subprocess.check_call(
                    [
                        "gcc", "-O3", *arch, "-fopenmp", "-Wall", "-Wextra", "-Werror",
                        *extra_defs, "-shared", "-fPIC", str(c_path), "-o", str(so_path), "-lm",
                    ]
                )
        self.lib_g = ctypes.CDLL(str(work_dir / "libgptq_fast6.so"))
        self.lib_g.fwht_rows_f32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib_g.unpack_qat_rows_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

        self.lfast = ctypes.CDLL(str(work_dir / "libfast_sgemm.so"))
        self.lfast.fast_sgemm_nt_f32.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float] * 2

        self.lfq = ctypes.CDLL(str(work_dir / "libfast_quant.so"))
        self.lfq.t16c_tblk_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lfq.t16c_tblk_bytes.restype = ctypes.c_size_t
        self.lfq.quant_and_pack_t16c_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
        ]

        self.lib_n = ctypes.CDLL(str(work_dir / "libneon_stream_accelerator.so"))
        self.lib_n.gemm_bitserial_tile16_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_int, ctypes.c_void_p,
        ]
        self.lib_n.gemm_bitserial_tile16_batch.restype = ctypes.c_int

    def mm_xt(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Compute `X @ W.T` in FP32 via NEON SGEMM when loaded, else NumPy."""
        x_c = np.ascontiguousarray(x, np.float32)
        w_c = np.ascontiguousarray(w, np.float32)
        if self.lfast is None:
            return np.ascontiguousarray(x_c @ w_c.T, np.float32)
        y = np.empty((x_c.shape[0], w_c.shape[0]), np.float32)
        self.lfast.fast_sgemm_nt_f32(x_c.ctypes.data, w_c.ctypes.data, y.ctypes.data, x_c.shape[0], w_c.shape[0], x_c.shape[1], 1.0, 0.0)
        return y

    def scale_t16c_pbytes_rows(
        self, pbytes: np.ndarray, d_out: int, d_in: int, bsize: int, mode_i: int, ar: np.ndarray
    ) -> None:
        """Apply least-squares per-row scaling `ar` directly to packed tile16 FP16 scales/biases."""
        if self.lfq is not None:
            tblk_b = int(self.lfq.t16c_tblk_bytes(bsize, mode_i))
        else:
            nb, asym = MODE_TO_T16C[mode_i]
            tblk_b = tile16_blk_bytes(bsize, nb, asym)
        n_tiles = (d_out + 15) // 16
        n_blks = d_in // bsize
        view = pbytes.reshape(n_tiles, n_blks, tblk_b)
        ar_pad = np.ones((n_tiles * 16,), dtype=np.float32)
        ar_pad[:d_out] = ar
        ar_t = ar_pad.reshape(n_tiles, 1, 16)
        sc16 = view[:, :, :32].view(np.float16).astype(np.float32)
        sc16 *= ar_t
        view[:, :, :32] = sc16.astype(np.float16).view(np.uint8)
        if mode_i in (0, 3):
            bi16 = view[:, :, 32:64].view(np.float16).astype(np.float32)
            bi16 *= ar_t
            view[:, :, 32:64] = bi16.astype(np.float16).view(np.uint8)

    def mm_neon_t16c(
        self, pbytes: np.ndarray, inv_s: np.ndarray, d_out: int, d_in: int, bsize: int, mode_i: int, x: np.ndarray
    ) -> np.ndarray:
        """Execute exact runtime-parity `gemm_bitserial_tile16_batch` on `(x * inv_s) -> FWHT`."""
        if self.lib_n is None or self.lib_g is None:
            raise RuntimeError("Native NEON stream accelerator is required for mm_neon_t16c")
        nb, asym = MODE_TO_T16C[mode_i]
        xr = np.ascontiguousarray(x * inv_s[None, :], np.float32)
        self.lib_g.fwht_rows_f32(xr.ctypes.data, len(xr), d_in, bsize)
        y = np.empty((len(xr), d_out), np.float32)
        rc = self.lib_n.gemm_bitserial_tile16_batch(
            pbytes.ctypes.data, None, xr.ctypes.data, y.ctypes.data,
            d_out, d_in, bsize, nb, 1 if asym else 0, len(xr), 0.0, 4, None,
        )
        if rc != 0:
            raise RuntimeError(f"gemm_bitserial_tile16_batch failed rc={rc} (mode={mode_i}, shape={d_out}x{d_in})")
        return y


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute flattened FP64 cosine similarity between two tensors."""
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-18))


def rmsn(x: np.ndarray, w: Optional[np.ndarray], eps: float) -> np.ndarray:
    """Apply Gemma-style RMSNorm with raw gain `w`."""
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y * w if w is not None else y


def rope(x: np.ndarray, theta: float, rot_dims: int, hd: int) -> np.ndarray:
    """Apply partial/full rotary positional embedding on `[T, H, hd]`."""
    hr = rot_dims // 2
    hh = hd // 2
    pos = np.arange(len(x), dtype=np.float32)[:, None]
    ang = pos * (1.0 / (theta ** (np.arange(0, 2 * hr, 2, dtype=np.float32) / float(hd))))[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    out = x.copy()
    out[:, :, :hr] = x[:, :, :hr] * c - x[:, :, hh : hh + hr] * s
    out[:, :, hh : hh + hr] = x[:, :, :hr] * s + x[:, :, hh : hh + hr] * c
    return out


def gelu(g: np.ndarray) -> np.ndarray:
    """Gemma-4 tanh-approximated GELU activation."""
    c0 = math.sqrt(2.0 / math.pi)
    return 0.5 * g * (1.0 + np.tanh(c0 * (g + 0.044715 * g * g * g)))


def compute_sq(x_bal: np.ndarray, w: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Compute clipped SmoothQuant scaling vector `s` for weights `W * s[None, :]`."""
    x_rms = np.sqrt(np.mean(x_bal * x_bal, axis=0) + 1e-12)
    w_rms = np.sqrt(np.mean(w * w, axis=0) + 1e-12)
    return np.clip((x_rms ** alpha) / (w_rms ** (1.0 - alpha)), 0.05, 20.0).astype(np.float32)


def balance_tokens_with_scale(
    x: np.ndarray, prompt_starts: List[int], bos_weight: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(xb, sf)` where `xb = x * sf` normalizes per-token RMS and down-weights BOS sinks."""
    x_c = np.ascontiguousarray(x, np.float32)
    trms = np.sqrt(np.mean(x_c * x_c, axis=1, keepdims=True) + 1e-12)
    sf = (1.0 / trms).astype(np.float32)
    for p0 in prompt_starts:
        if 0 <= p0 < len(sf):
            sf[p0] *= bos_weight
    return np.ascontiguousarray(x_c * sf, np.float32), sf


def balance_tokens(x: np.ndarray, prompt_starts: List[int], bos_weight: float = 0.15) -> np.ndarray:
    """Normalize per-token RMS and down-weight BOS sink rows."""
    return balance_tokens_with_scale(x, prompt_starts, bos_weight=bos_weight)[0]


def precompute_x_pinv_t(x_bal: np.ndarray, ops: _NativeCalOps, ridge: float = 1e-2) -> np.ndarray:
    """Precompute transposed regularized ridge pseudo-inverse of `[N_tr, d_in]`."""
    n_tr = x_bal.shape[0]
    xxt = ops.mm_xt(x_bal, x_bal).astype(np.float64)
    lam = ridge * np.trace(xxt) / max(n_tr, 1)
    x_pinv = np.linalg.solve(xxt + lam * np.eye(n_tr), x_bal).astype(np.float32)
    return np.ascontiguousarray(x_pinv.T, np.float32)


def gram_svd_top(y_err: np.ndarray, max_rk: int, ops: _NativeCalOps) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-`max_rk` singular triplets of tall `[d_out, N_tr]` matrix via `[N_tr, N_tr]` Gram eigh."""
    y_t = np.ascontiguousarray(y_err.T, np.float32)
    g = ops.mm_xt(y_t, y_t).astype(np.float64)
    evals, evecs = np.linalg.eigh(g)
    idx = np.argsort(-evals)[:max_rk]
    s = np.sqrt(np.maximum(evals[idx], 0.0)).astype(np.float32)
    vt = np.ascontiguousarray(evecs[:, idx].T, np.float32)
    u = ops.mm_xt(y_err, vt)
    s_safe = np.where(s > 1e-12, s, 1.0).astype(np.float32)
    u /= s_safe[None, :]
    return u, s, vt


def fit_act_als_i8(
    y_err: np.ndarray,
    x_bal: np.ndarray,
    x_pinv_t: np.ndarray,
    rk: int,
    ops: _NativeCalOps,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit activation-aware Int8LowRank (`U_q @ V_q`) to minimize `||Y_err - U V X_bal.T||_F`."""
    u0, s0, vt0 = gram_svd_top(y_err, rk, ops)
    u = np.ascontiguousarray(u0[:, :rk], np.float32).copy()
    b_target = np.ascontiguousarray(s0[:rk, None] * vt0[:rk, :], np.float32)
    v = ops.mm_xt(b_target, x_pinv_t)
    for _ in range(2):
        z = ops.mm_xt(x_bal, v).T
        zzt = ops.mm_xt(z, z).astype(np.float64) + 1e-4 * np.eye(rk)
        yzt = ops.mm_xt(y_err, z)
        u = ops.mm_xt(yzt, np.linalg.inv(zzt).T.astype(np.float32))
        utu = ops.mm_xt(u.T, u.T).astype(np.float64) + 1e-4 * np.eye(rk)
        uty = ops.mm_xt(u.T, y_err.T)
        b_target = np.linalg.solve(utu, uty).astype(np.float32)
        v = ops.mm_xt(b_target, x_pinv_t)
    su = (np.max(np.abs(u), 1) / 127.0 + 1e-12).astype(np.float32)
    sv = (np.max(np.abs(v), 1) / 127.0 + 1e-12).astype(np.float32)
    u_i8 = np.clip(np.round(u / su[:, None]), -127, 127).astype(np.int8)
    v_i8 = np.clip(np.round(v / sv[:, None]), -127, 127).astype(np.int8)
    u_q = (u_i8.astype(np.float32) * su[:, None]).astype(np.float32)
    v_q = (v_i8.astype(np.float32) * sv[:, None]).astype(np.float32)
    return u_q, v_q, su, sv, u_i8, v_i8


def _mat_packed_size(name: str, d_out: int, d_in: int, bsize: int, mode_i: int, rk: int, has_inv_s: bool = True) -> int:
    """Compute exact `.t16c` byte footprint of one matrix entry."""
    nb, asym = MODE_TO_T16C[mode_i]
    tblk = tile16_blk_bytes(bsize, nb, asym)
    pbytes = ((d_out + 15) // 16) * (d_in // bsize) * tblk
    inv_b = (d_in * 4) if has_inv_s else 0
    lr_b = (d_out * 4 + rk * 4 + d_out * rk + rk * d_in) if rk > 0 else 0
    return 34 + len(name) + 32 + pbytes + inv_b + lr_b


def build_optimal_allocation(
    cfg: dict,
    xt_list: List[np.ndarray],
    xf_list: List[np.ndarray],
    budget_bytes: int = 6_360_000_000,
) -> Tuple[Dict[int, dict], int]:
    """Allocate per-projection modes and low-rank ranks within exact `.t16c` byte budget.

    Uses all-5-trit (mode 1) attention on active layers to maximize full `(1, 1, 1)` 5-trit MLP
    coverage across cliff/mid layers (`L38..L57`), zeroes rank on quiet layers (`dx <= 0.145`),
    and locks the final peak tail (`L58..L59`) at 3-bit (`mode 4`).
    """
    n_layers = int(cfg["num_hidden_layers"])
    dx_last: List[float] = []
    for l in range(n_layers):
        dx2 = np.linalg.norm(xt_list[l + 1][-1] - xt_list[l][-1]) / (np.linalg.norm(xt_list[l][-1]) + 1e-12)
        dxf = np.linalg.norm(xf_list[l + 1][-1] - xf_list[l][-1]) / (np.linalg.norm(xf_list[l][-1]) + 1e-12)
        dx_last.append(max(float(dx2), float(dxf)))

    imp: List[float] = []
    for l in range(n_layers):
        w = dx_last[l]
        if l in (0, 1):
            w *= 1.35
        elif l >= max(n_layers - 9, 2):
            w *= 1.30
        elif 25 <= l <= max(n_layers - 10, 25):
            w *= 1.15
        imp.append(w)

    emb_b = _mat_packed_size("model.language_model.embed_tokens.weight", cfg["vocab_size"], cfg["hidden_size"], 256, 4, 0, False)
    norms_b = 7_500_000
    alloc: Dict[int, dict] = {}
    for l in range(n_layers):
        is_full = cfg["layer_types"][l] == "full_attention"
        dx = dx_last[l]
        if l >= max(n_layers - 2, 0):
            alloc[l] = {
                "q": (4, 256, 24),
                "k": (2, 256, 0),
                "v": None if is_full else (2, 256, 0),
                "o": (4, 256, 24),
                "g": (4, 256, 32),
                "u": (4, 256, 32),
                "d": (4, 256, 32),
            }
        elif dx <= 0.145:
            alloc[l] = {
                "q": (0, 256, 0),
                "k": (1, 256, 0),
                "v": None if is_full else (1, 256, 0),
                "o": (0, 256, 0),
                "g": (0, 256, 0),
                "u": (0, 256, 0),
                "d": (0, 256, 0),
            }
        else:
            rka_qo = 24 if (l <= 1 or dx >= 0.50) else 16
            rkm = 32 if (l <= 1 or dx >= 0.65) else (24 if dx >= 0.35 else 16)
            alloc[l] = {
                "q": (1, 256, rka_qo),
                "k": (1, 256, 0),
                "v": None if is_full else (1, 256, 0),
                "o": (1, 256, rka_qo),
                "g": (0, 256, rkm),
                "u": (0, 256, rkm),
                "d": (0, 256, rkm),
            }

    def total_bytes() -> int:
        tot = 24 + norms_b + emb_b
        d = int(cfg["hidden_size"])
        ff = int(cfg["intermediate_size"])
        nh = int(cfg.get("num_attention_heads", 32))
        nkv_s = int(cfg.get("num_key_value_heads", 16))
        nkv_f = int(cfg.get("num_global_key_value_heads", 4))
        hd_s = int(cfg.get("head_dim", 256))
        hd_f = int(cfg.get("global_head_dim", 512))
        for l in range(n_layers):
            is_full = cfg["layer_types"][l] == "full_attention"
            hd = hd_f if is_full else hd_s
            nkv = nkv_f if is_full else nkv_s
            shapes = {
                "q": (nh * hd, d),
                "k": (nkv * hd, d),
                "v": (nkv * hd, d),
                "o": (d, nh * hd),
                "g": (ff, d),
                "u": (ff, d),
                "d": (d, ff),
            }
            for s, spec in alloc[l].items():
                if spec is None:
                    continue
                md, bs, rk = spec
                dout, din = shapes[s]
                tot += _mat_packed_size(f"model.language_model.layers.{l}.{s}.weight", dout, din, bs, md, rk, True)
        return tot

    order = sorted([l for l in range(n_layers - 2) if dx_last[l] > 0.145], key=lambda l: -imp[l])
    # First upgrade full (g, u, d) triples to 5-trit (mode 1) in descending priority order.
    for l in order:
        old = {k: alloc[l][k] for k in ("g", "u", "d")}
        for k in ("g", "u", "d"):
            alloc[l][k] = (1, 256, old[k][2])
        if total_bytes() > budget_bytes:
            for k in ("g", "u", "d"):
                alloc[l][k] = old[k]

    # Spend leftover budget upgrading remaining down_proj -> mode 1, then peak layers -> mode 2.
    for l in order:
        if alloc[l]["d"][0] == 0:
            old = alloc[l]["d"]
            alloc[l]["d"] = (1, 256, old[2])
            if total_bytes() > budget_bytes:
                alloc[l]["d"] = old
    for l in order:
        if imp[l] >= 0.65:
            for k in ("d", "g", "u"):
                old = alloc[l][k]
                if old[0] < 2:
                    alloc[l][k] = (2, 256, old[2])
                    if total_bytes() > budget_bytes:
                        alloc[l][k] = old

    tb = total_bytes()
    if tb > budget_bytes:
        raise ValueError(f"Base allocation {tb:,} exceeds budget {budget_bytes:,}")
    return alloc, tb


def _rqat(sr: SafetensorsReader, pfx: str, ops: _NativeCalOps) -> np.ndarray:
    """Unpack a packed INT4 QAT weight matrix from `model.safetensors`."""
    ip, isc = sr.hdr[pfx + ".weight_packed"], sr.hdr[pfx + ".weight_scale"]
    o0, o1 = ip["data_offsets"]
    sr._fp.seek(sr.data_base + o0)
    pk = np.frombuffer(sr._fp.read(o1 - o0), np.int32)
    so0, so1 = isc["data_offsets"]
    sr._fp.seek(sr.data_base + so0)
    sc = np.frombuffer(sr._fp.read(so1 - so0), np.uint16)
    rows, din = ip["shape"][0], ip["shape"][1] * 8
    out = np.empty((rows, din), np.float32)
    ops.lib_g.unpack_qat_rows_f32(pk.ctypes.data, sc.ctypes.data, out.ctypes.data, rows, din)
    return out


def _read_bf16_rows(sr: SafetensorsReader, key: str, r0: int, r1: int) -> np.ndarray:
    info = sr.hdr[key]
    o0, _ = info["data_offsets"]
    d = int(info["shape"][1])
    rb = d * 2
    sr._fp.seek(sr.data_base + o0 + r0 * rb)
    raw = sr._fp.read((r1 - r0) * rb)
    return (np.frombuffer(raw, np.uint16).reshape(r1 - r0, d).astype(np.uint32) << 16).view(np.float32).copy()


def _quant_pack_eval(
    w_orig: np.ndarray,
    x_fit: np.ndarray,
    x_bal: np.ndarray,
    sf_fit: np.ndarray,
    x_pinv_t: np.ndarray,
    x_eval_list: List[np.ndarray],
    spec: Tuple[int, int, int],
    out_dict: dict,
    mat_key: str,
    ops: _NativeCalOps,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Quantize + FWHT + pack, apply least-squares scale + ActALS Int8LowRank via exact NEON GEMM."""
    mode_i, bsize, rk = spec
    d_out, d_in = w_orig.shape
    y_tgt = ops.mm_xt(x_fit, w_orig)
    sq_s = compute_sq(x_bal, w_orig, 0.5)
    inv_s = (1.0 / sq_s).astype(np.float32)

    # Explicit copy so SmoothQuant + in-place FWHT/pack never corrupts w_orig (needed for shared full-attn k/v).
    q = np.array(w_orig, dtype=np.float32, copy=True, order="C")
    q *= sq_s[None, :]
    ops.lib_g.fwht_rows_f32(q.ctypes.data, d_out, d_in, bsize)

    tblk_b = int(ops.lfq.t16c_tblk_bytes(bsize, mode_i))
    pbytes = np.empty(((d_out + 15) // 16) * (d_in // bsize) * tblk_b, np.uint8)
    ops.lfq.quant_and_pack_t16c_f32(q.ctypes.data, d_out, d_in, bsize, mode_i, pbytes.ctypes.data)
    del q

    n_fit = len(x_fit)
    x_cat = np.vstack([x_fit, *x_eval_list]) if x_eval_list else x_fit
    y_all = ops.mm_neon_t16c(pbytes, inv_s, d_out, d_in, bsize, mode_i, x_cat)
    yq_raw = y_all[:n_fit]

    yt_b = y_tgt * sf_fit
    yq_b = yq_raw * sf_fit
    ar = np.clip(
        np.sum(yt_b * yq_b, axis=0, dtype=np.float64) / (np.sum(yq_b * yq_b, axis=0, dtype=np.float64) + 1e-12),
        0.70,
        1.40,
    ).astype(np.float32)
    ops.scale_t16c_pbytes_rows(pbytes, d_out, d_in, bsize, mode_i, ar)
    y_all *= ar[None, :]

    yq_fit = np.ascontiguousarray(y_all[:n_fit], np.float32)
    outs: List[np.ndarray] = []
    off = n_fit
    for xe in x_eval_list:
        outs.append(np.ascontiguousarray(y_all[off : off + len(xe)], np.float32))
        off += len(xe)

    out_dict[f"{mat_key}__meta"] = np.array([d_out, d_in, bsize, mode_i, rk], dtype=np.int32)
    out_dict[f"{mat_key}__pbytes"] = pbytes
    out_dict[f"{mat_key}__inv_s"] = inv_s

    if rk > 0:
        y_err = np.ascontiguousarray(((y_tgt - yq_fit) * sf_fit).T, np.float32)
        u_q, v_q, su, sv, u_i8, v_i8 = fit_act_als_i8(y_err, x_bal, x_pinv_t, rk, ops)
        yq_fit = yq_fit + ops.mm_xt(ops.mm_xt(x_fit, v_q), u_q)
        outs = [o + ops.mm_xt(ops.mm_xt(xe, v_q), u_q) for o, xe in zip(outs, x_eval_list)]
        out_dict[f"{mat_key}__su"] = su
        out_dict[f"{mat_key}__sv"] = sv
        out_dict[f"{mat_key}__ui8"] = u_i8
        out_dict[f"{mat_key}__vi8"] = v_i8
        del y_err, u_q, v_q

    del w_orig, y_tgt, y_all, yt_b, yq_b
    gc.collect()
    return yq_fit, outs


def run_calibration(
    sf_path: Path,
    work_dir: Path,
    layers_dir: Path,
    out_t16c: Optional[Path],
    budget_bytes: int = 6_360_000_000,
) -> None:
    """Execute full 60-layer cascaded V3 calibration and optional `.t16c` assembly."""
    model_dir = sf_path.parent
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json at {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["text_config"]
    eps = float(cfg["rms_norm_eps"])
    d = int(cfg["hidden_size"])
    vocab = int(cfg["vocab_size"])
    n_layers = int(cfg["num_hidden_layers"])
    escale = math.sqrt(float(d))

    meta_path = work_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing snapshot metadata {meta_path}; generate layer snapshots first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ids_all = meta["all_ids"]
    cal_ids = ids_all[: meta["cal_end"]]
    t2_ids = ids_all[meta["test2"][0] : meta["test2"][1]]
    tf_ids = ids_all[meta["testf"][0] : meta["testf"][1]]
    pstarts = [i for i, t in enumerate(cal_ids) if t == 2] + [len(cal_ids)]
    cal_segs = list(zip(pstarts[:-1], pstarts[1:]))

    xt_qat = [np.load(work_dir / f"snaps_L{l:02d}.npz")["x_t2"].astype(np.float32) for l in range(n_layers)]
    xf_qat = [np.load(work_dir / f"snaps_L{l:02d}.npz")["x_tf"].astype(np.float32) for l in range(n_layers)]
    x_after = np.load(work_dir / f"x_after_L{n_layers - 1}.npy").astype(np.float32)
    xt_qat.append(x_after[meta["test2"][0] : meta["test2"][1]])
    xf_qat.append(x_after[meta["testf"][0] : meta["testf"][1]])
    del x_after

    ops = _NativeCalOps(work_dir, require_native=True)
    alloc, tot_b = build_optimal_allocation(cfg, xt_qat, xf_qat, budget_bytes=budget_bytes)
    layers_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "alloc_v3.json").write_text(json.dumps({str(l): alloc[l] for l in range(n_layers)}, indent=2), encoding="utf-8")
    print(f"[ALLOC] target_bytes={tot_b:,} B ({tot_b / 1e9:.3f} GB / {tot_b / (1024**3):.3f} GiB)", flush=True)

    with SafetensorsReader(sf_path) as sr:
        emb_npz = layers_dir / "embed_3b256.npz"
        if not emb_npz.exists():
            t0 = time.perf_counter()
            tblk_b = int(ops.lfq.t16c_tblk_bytes(256, 4))
            emb_pbytes = np.empty((vocab // 16) * (d // 256) * tblk_b, np.uint8)
            xt_e = np.empty((len(t2_ids), d), np.float32)
            xf_e = np.empty((len(tf_ids), d), np.float32)
            for r0 in range(0, vocab, 16384):
                r1 = min(r0 + 16384, vocab)
                emb = _read_bf16_rows(sr, "model.language_model.embed_tokens.weight", r0, r1)
                ops.lib_g.fwht_rows_f32(emb.ctypes.data, r1 - r0, d, 256)
                sub_b = np.empty(((r1 - r0) // 16) * (d // 256) * tblk_b, np.uint8)
                ops.lfq.quant_and_pack_t16c_f32(emb.ctypes.data, r1 - r0, d, 256, 4, sub_b.ctypes.data)
                off = (r0 // 16) * (d // 256) * tblk_b
                emb_pbytes[off : off + len(sub_b)] = sub_b
                ops.lib_g.fwht_rows_f32(emb.ctypes.data, r1 - r0, d, 256)
                for i, tid in enumerate(t2_ids):
                    if r0 <= tid < r1:
                        xt_e[i] = emb[tid - r0] * escale
                for i, tid in enumerate(tf_ids):
                    if r0 <= tid < r1:
                        xf_e[i] = emb[tid - r0] * escale
                del emb, sub_b
            np.savez(emb_npz, pbytes=emb_pbytes, xt0=xt_e, xf0=xf_e)
            del emb_pbytes
            gc.collect()
            print(f"[EMBED 3b_256] {time.perf_counter()-t0:.1f}s cos_t2={cos_sim(xt_e, xt_qat[0]):.5f}", flush=True)
        else:
            with np.load(emb_npz) as ed:
                xt_e, xf_e = ed["xt0"].astype(np.float32), ed["xf0"].astype(np.float32)

        xt_c, xf_c = xt_e.copy(), xf_e.copy()
        with np.load(work_dir / "snaps_L00.npz") as sn0:
            xc_c = sn0["x_cal"].astype(np.float32)
        state_ckpt = layers_dir / "cascade_state.npz"
        start_l = 0
        if state_ckpt.exists():
            with np.load(state_ckpt) as st:
                next_l = int(st["next_l"])
                if all((layers_dir / f"L{k:02d}.npz").is_file() for k in range(next_l)):
                    if "xc_c" in st.files:
                        xc_c = st["xc_c"].astype(np.float32)
                    elif next_l < n_layers:
                        with np.load(work_dir / f"snaps_L{next_l:02d}.npz") as snn:
                            xc_c = snn["x_cal"].astype(np.float32)
                    xt_c = st["xt_c"].astype(np.float32)
                    xf_c = st["xf_c"].astype(np.float32)
                    start_l = next_l
                    print(f"[RESUME] starting at layer {start_l}", flush=True)

        for l in range(start_l, n_layers):
            tl = time.perf_counter()
            p = f"model.language_model.layers.{l}."
            is_full = cfg["layer_types"][l] == "full_attention"

            g_in = sr.read_f32_tensor(p + "input_layernorm.weight")
            qn = sr.read_f32_tensor(p + "self_attn.q_norm.weight")
            kn = sr.read_f32_tensor(p + "self_attn.k_norm.weight")
            g_pa = sr.read_f32_tensor(p + "post_attention_layernorm.weight")
            g_pf = sr.read_f32_tensor(p + "pre_feedforward_layernorm.weight")
            g_pff = sr.read_f32_tensor(p + "post_feedforward_layernorm.weight")
            ls = float(np.asarray(sr.read_f32_tensor(p + "layer_scalar")).reshape(-1)[0])

            nh, nkv, hd = 32, (4 if is_full else 16), (512 if is_full else 256)
            rot_d, theta, gqa = (128 if is_full else 256), (1e6 if is_full else 1e4), 32 // (4 if is_full else 16)

            inp_fit = rmsn(xc_c, g_in, eps)
            inpb, sf_a = balance_tokens_with_scale(inp_fit, pstarts[:-1], 0.15)
            pinv_a = precompute_x_pinv_t(inpb, ops)
            inp_t2, inp_tf = rmsn(xt_c, g_in, eps), rmsn(xf_c, g_in, eps)

            layer_dict: dict = {}
            wq = _rqat(sr, p + "self_attn.q_proj", ops)
            wk = _rqat(sr, p + "self_attn.k_proj", ops)
            wv = wk if is_full else _rqat(sr, p + "self_attn.v_proj", ops)
            yq_cal_q, (yq_2, yq_f) = _quant_pack_eval(wq, inp_fit, inpb, sf_a, pinv_a, [inp_t2, inp_tf], alloc[l]["q"], layer_dict, "q_proj", ops)
            yk_cal_q, (yk_2, yk_f) = _quant_pack_eval(wk, inp_fit, inpb, sf_a, pinv_a, [inp_t2, inp_tf], alloc[l]["k"], layer_dict, "k_proj", ops)
            if is_full:
                yv_cal_q, yv_2, yv_f = yk_cal_q, yk_2, yk_f
            else:
                yv_cal_q, (yv_2, yv_f) = _quant_pack_eval(wv, inp_fit, inpb, sf_a, pinv_a, [inp_t2, inp_tf], alloc[l]["v"], layer_dict, "v_proj", ops)
            del wq, wk, wv, inp_fit, inpb, sf_a, pinv_a, inp_t2, inp_tf

            def _attn_ctx(yq: np.ndarray, yk: np.ndarray, yv: np.ndarray) -> np.ndarray:
                t_len = len(yq)
                qt = rope(rmsn(yq.reshape(t_len, nh, hd), qn, eps), theta, rot_d, hd)
                kt = rope(rmsn(yk.reshape(t_len, nkv, hd), kn, eps), theta, rot_d, hd)
                vt = rmsn(yv.reshape(t_len, nkv, hd), None, eps)
                c_out = np.empty((t_len, nh, hd), np.float32)
                m_c = np.triu(np.full((t_len, t_len), -1e9, np.float32), 1)
                for h_i in range(nh):
                    s_m = qt[:, h_i, :] @ kt[:, h_i // gqa, :].T + m_c
                    s_m -= np.max(s_m, -1, keepdims=True)
                    p_m = np.exp(s_m)
                    p_m /= np.sum(p_m, -1, keepdims=True)
                    c_out[:, h_i, :] = p_m @ vt[:, h_i // gqa, :]
                return c_out.reshape(t_len, nh * hd)

            ctx_cal = np.concatenate([_attn_ctx(yq_cal_q[a:b], yk_cal_q[a:b], yv_cal_q[a:b]) for a, b in cal_segs], 0)
            ctxb, sf_c = balance_tokens_with_scale(ctx_cal, pstarts[:-1], 0.15)
            pinv_c = precompute_x_pinv_t(ctxb, ops)
            wo = _rqat(sr, p + "self_attn.o_proj", ops)
            yo_cal_q, (yo_2, yo_f) = _quant_pack_eval(
                wo, ctx_cal, ctxb, sf_c, pinv_c, [_attn_ctx(yq_2, yk_2, yv_2), _attn_ctx(yq_f, yk_f, yv_f)],
                alloc[l]["o"], layer_dict, "o_proj", ops,
            )
            del wo, ctx_cal, ctxb, sf_c, pinv_c, yq_cal_q, yk_cal_q, yv_cal_q, yq_2, yk_2, yv_2, yq_f, yk_f, yv_f

            xm_cal = xc_c + rmsn(yo_cal_q, g_pa, eps)
            xm_t2, xm_tf = xt_c + rmsn(yo_2, g_pa, eps), xf_c + rmsn(yo_f, g_pa, eps)
            hm_cal = rmsn(xm_cal, g_pf, eps)
            hm_t2, hm_tf = rmsn(xm_t2, g_pf, eps), rmsn(xm_tf, g_pf, eps)
            del yo_cal_q, yo_2, yo_f

            hmb, sf_h = balance_tokens_with_scale(hm_cal, pstarts[:-1], 0.15)
            pinv_h = precompute_x_pinv_t(hmb, ops)
            wg = _rqat(sr, p + "mlp.gate_proj", ops)
            g_cal_q, (gq_2, gq_f) = _quant_pack_eval(wg, hm_cal, hmb, sf_h, pinv_h, [hm_t2, hm_tf], alloc[l]["g"], layer_dict, "gate_proj", ops)
            wu = _rqat(sr, p + "mlp.up_proj", ops)
            u_cal_q, (uq_2, uq_f) = _quant_pack_eval(wu, hm_cal, hmb, sf_h, pinv_h, [hm_t2, hm_tf], alloc[l]["u"], layer_dict, "up_proj", ops)
            del wg, wu, hmb, sf_h, pinv_h, hm_cal, hm_t2, hm_tf

            act_q_cal = gelu(g_cal_q) * u_cal_q
            actb, sf_d = balance_tokens_with_scale(act_q_cal, pstarts[:-1], 0.15)
            del g_cal_q, u_cal_q
            pinv_d = precompute_x_pinv_t(actb, ops)

            wd = _rqat(sr, p + "mlp.down_proj", ops)
            d_cal_q, (dq_2, dq_f) = _quant_pack_eval(
                wd, act_q_cal, actb, sf_d, pinv_d, [gelu(gq_2) * uq_2, gelu(gq_f) * uq_f],
                alloc[l]["d"], layer_dict, "down_proj", ops,
            )
            del wd, act_q_cal, actb, sf_d, pinv_d, gq_2, uq_2, gq_f, uq_f

            xc_c = (xm_cal + rmsn(d_cal_q, g_pff, eps)) * ls
            xt_c = (xm_t2 + rmsn(dq_2, g_pff, eps)) * ls
            xf_c = (xm_tf + rmsn(dq_f, g_pff, eps)) * ls

            tmp_l = layers_dir / f"L{l:02d}.npz.tmp.npz"
            np.savez(tmp_l, **layer_dict)
            os.replace(tmp_l, layers_dir / f"L{l:02d}.npz")
            tmp_s = layers_dir / "cascade_state.tmp.npz"
            np.savez(tmp_s, xc_c=xc_c, xt_c=xt_c, xf_c=xf_c, next_l=np.array(l + 1, dtype=np.int32))
            os.replace(tmp_s, state_ckpt)
            del layer_dict, xm_cal, xm_t2, xm_tf, d_cal_q, dq_2, dq_f
            gc.collect()

            print(
                f"L{l:02d} ({alloc[l]['g'][0]}/{alloc[l]['u'][0]}/{alloc[l]['d'][0]} r={alloc[l]['d'][2]}) "
                f"casc_t2={cos_sim(xt_c, xt_qat[l+1]):.5f}(last={cos_sim(xt_c[-1], xt_qat[l+1][-1]):.5f}) "
                f"casc_tf={cos_sim(xf_c, xf_qat[l+1]):.5f}(last={cos_sim(xf_c[-1], xf_qat[l+1][-1]):.5f}) "
                f"nr={np.linalg.norm(xt_c)/np.linalg.norm(xt_qat[l+1]):.4f} [{time.perf_counter()-tl:.1f}s] rss={_rss_mb():.0f}MB",
                flush=True,
            )

        # Final norm + 262k LM-head logit parity diagnostic against QAT reference
        final_norm = sr.read_f32_tensor("model.language_model.norm.weight")
        softcap = float(cfg.get("final_logit_softcapping", 30.0) or 0.0)
        h_t2 = rmsn(xt_c[-1:], final_norm, eps)
        h_tf = rmsn(xf_c[-1:], final_norm, eps)
        q_t2 = rmsn(xt_qat[n_layers][-1:], final_norm, eps)
        q_tf = rmsn(xf_qat[n_layers][-1:], final_norm, eps)
        print(
            f"[FINAL_HIDDEN] cos_t2={cos_sim(h_t2, q_t2):.5f} nr_t2={np.linalg.norm(h_t2)/(np.linalg.norm(q_t2)+1e-12):.4f} | "
            f"cos_tf={cos_sim(h_tf, q_tf):.5f} nr_tf={np.linalg.norm(h_tf)/(np.linalg.norm(q_tf)+1e-12):.4f}",
            flush=True,
        )
        with np.load(emb_npz) as ed:
            emb_pb = np.ascontiguousarray(ed["pbytes"], np.uint8)
        ones_d = np.ones((d,), np.float32)
        logits_v3 = ops.mm_neon_t16c(emb_pb, ones_d, vocab, d, 256, 4, np.vstack([h_t2, h_tf]))
        if softcap > 0.0:
            logits_v3 = softcap * np.tanh(logits_v3 / softcap)
        del emb_pb
        for lbl, lv in [("test_2p2 V3_TOP5", logits_v3[0]), ("test_fr V3_TOP5", logits_v3[1])]:
            top5 = np.argsort(-lv)[:5]
            print(f"[{lbl}] {[(int(i), round(float(lv[i]), 3)) for i in top5]}", flush=True)

    if out_t16c is not None:
        sz = assemble_t16c_v3(sf_path, layers_dir, out_t16c, n_layers=n_layers)
        print(f"[PACKED] {out_t16c} ({sz:,} B / {sz / (1024**3):.3f} GiB)", flush=True)


def _self_test() -> int:
    """Run 6 fail-closed unit gates on pure calibration primitives."""
    failures = 0
    rng = np.random.default_rng(7)
    ops = _NativeCalOps(None, require_native=False)

    # Gate 1: balance_tokens normalizes non-BOS RMS to ~1.0 and scales BOS
    try:
        x = rng.standard_normal((16, 64)).astype(np.float32) * 5.0
        xb = balance_tokens(x, [0, 8], bos_weight=0.15)
        rms_rows = np.sqrt(np.mean(xb * xb, axis=1))
        assert np.allclose(rms_rows[[1, 2, 9]], 1.0, atol=1e-4)
        assert np.allclose(rms_rows[[0, 8]], 0.15, atol=1e-4)
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    # Gate 2: SmoothQuant range & finiteness
    try:
        x = rng.standard_normal((16, 64)).astype(np.float32) * 50.0
        w = rng.standard_normal((32, 64)).astype(np.float32) * 0.01
        sq = compute_sq(x, w, alpha=0.5)
        assert sq.shape == (64,) and sq.dtype == np.float32
        assert np.all((sq >= 0.05) & (sq <= 20.0))
    except Exception as exc:
        print(f"[Gate 2 FAIL] {exc}")
        failures += 1

    # Gate 3: Ridge pseudo-inverse solves X_bal
    try:
        xb = rng.standard_normal((24, 64)).astype(np.float32)
        pinv_t = precompute_x_pinv_t(xb, ops)
        assert pinv_t.shape == (64, 24) and np.all(np.isfinite(pinv_t))
        eye_approx = ops.mm_xt(pinv_t.T, xb)
        assert eye_approx.shape == (24, 24) and np.mean(np.diag(eye_approx)) > 0.85
    except Exception as exc:
        print(f"[Gate 3 FAIL] {exc}")
        failures += 1

    # Gate 4: Gram SVD + ActALS decreases Frobenius error strictly on low-rank residual
    try:
        xb = rng.standard_normal((32, 64)).astype(np.float32)
        pinv_t = precompute_x_pinv_t(xb, ops)
        u_true = rng.standard_normal((48, 4)).astype(np.float32)
        v_true = rng.standard_normal((4, 64)).astype(np.float32)
        y_err = (u_true @ v_true) @ xb.T
        u_q, v_q, *_ = fit_act_als_i8(y_err, xb, pinv_t, rk=4, ops=ops)
        res_err = np.linalg.norm(y_err - (u_q @ v_q) @ xb.T)
        assert res_err < 0.05 * np.linalg.norm(y_err), f"res_err={res_err}"
    except Exception as exc:
        print(f"[Gate 4 FAIL] {exc}")
        failures += 1

    # Gate 5: Budgeted sensitivity allocation stays <= budget and skips full-attn v_proj
    try:
        n_l = 60
        cfg = {
            "num_hidden_layers": n_l,
            "vocab_size": 262144,
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "layer_types": ["full_attention" if (i % 6 == 5) else "sliding_attention" for i in range(n_l)],
        }
        xt = [np.ones((4, 8), np.float32)]
        xf = [np.ones((4, 8), np.float32)]
        for l in range(n_l):
            step = 0.25 if (l < 2 or l > 50) else 0.08
            xt.append(xt[-1] + step)
            xf.append(xf[-1] + step)
        alloc, tot_b = build_optimal_allocation(cfg, xt, xf, budget_bytes=6_380_000_000)
        assert tot_b <= 6_380_000_000
        assert alloc[5]["v"] is None and alloc[0]["v"] is not None
    except Exception as exc:
        print(f"[Gate 5 FAIL] {exc}")
        failures += 1

    # Gate 6: RMSNorm, partial RoPE norm preservation, and GELU zero/sign checks
    try:
        v = rng.standard_normal((5, 8, 16)).astype(np.float32)
        vr = rope(v, 10000.0, 8, 16)
        assert np.allclose(np.linalg.norm(v, axis=-1), np.linalg.norm(vr, axis=-1), rtol=1e-4)
        assert np.isclose(float(gelu(np.array([0.0]))[0]), 0.0)
        rn = rmsn(v, np.ones((16,), np.float32), 1e-6)
        assert np.allclose(np.mean(rn * rn, axis=-1), 1.0, atol=1e-4)
    except Exception as exc:
        print(f"[Gate 6 FAIL] {exc}")
        failures += 1

    if failures == 0:
        print("[PASS] calibrate_31b_v3.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser(description="Calibrate Gemma 4 31B V3 mixed-precision .t16c artifact.")
    ap.add_argument("--safetensors", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b"))
    ap.add_argument("--layers-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b/layers_v3"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v3.t16c"))
    ap.add_argument("--budget-bytes", type=int, default=6_360_000_000)
    args = ap.parse_args()

    sf_path = resolve_safetensors_path(args.safetensors)
    run_calibration(sf_path, args.work_dir.resolve(), args.layers_dir.resolve(), args.out.resolve(), args.budget_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
V9.2-CliffFix-VLM Unified Pi-4 Native Calibration & Assembly Pipeline (`0` Colab CU).

Targeted upgrades on top of verified `V9.1-Fused-VLM` inside physical Pi 4 RAM
(`<= ~7.20M kB` COCO Vision `vm_hwm`, `ssd_read_bytes = 0`):
  1. Pinpointed Cliff Upgrade (`1b_rm -> 2b_gauss`, mode `0x82`/`0xa2` COLBLOCK):
     SmoothQuant + FWHT-256 + chunked `libgauss_gptq_v2.so` (`use_gauss=1`) +
     per-row AR calibration + held-out validated ActALS INT8 low-rank residual.
  2. Lossless GeGLU Channel Clustering (`pi_l`): permutes intermediate rows of
     `gate_proj`/`up_proj` and columns of `down_proj` into shared heavy +
     co-firing `256`-col cortical blocks so sparse CRTR routing preserves recall.
  3. 60-Layer Causal Student Propagation (`abits=6` NEON) on decontaminated
     calibration sequences (`snaps_L00..L59`), solving closed-form Wiener
     shrinkage with held-out MSE rollback on `post_attention_layernorm`,
     `post_feedforward_layernorm`, `layer_scalar`, and `model.language_model.norm.weight`.
  4. Unified Training of Fresh `CRTR` (`rank=64 + MoD`), `HeadRouter` (`rank=32 + MoD`),
     and Accept-Only `C8DB` (preserving `embed_vision` from `V4.3-VLM` + retraining
     bottleneck adapters & `post_cell60`).

Fail-closed 6-gate `--selftest` verified locally and on Pi 4.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import heapq
import json
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_v9_1_fused_vlm as b91  # noqa: E402
import calibrate_31b_v3 as c3  # noqa: E402
import calibrate_31b_v4 as c4  # noqa: E402
import pack_t16c_v3 as p3  # noqa: E402
import train_c8d_pattern_bank as c8d  # noqa: E402
import train_cortical_router as crtr  # noqa: E402


T16C_MAGIC = b"T16C"
MODE_2B_GAUSS = 5
NB_2B_GAUSS_TM = 2 | 0x80
NB_2B_GAUSS_CB = 2 | 0x80 | 0x20
BSIZE_DEFAULT = 256
# Default pinpointed cliff layers sorted by V9.1 empirical 60-layer scan (abits=6)
DEFAULT_UPGRADE_FULL = (49, 51, 50, 34, 43, 46, 44, 45, 40, 48, 38, 9)
DEFAULT_UPGRADE_GU: Tuple[int, ...] = ()
DEFAULT_UPGRADE_DOWN = (53,)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file using streaming 4 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_layer_csv(spec: str) -> Tuple[int, ...]:
    """Parse comma-separated layer indices, rejecting out-of-range values."""
    s = spec.strip()
    if not s:
        return ()
    out: List[int] = []
    seen = set()
    for part in s.split(","):
        tok = part.strip()
        if not tok:
            continue
        val = int(tok)
        if not (0 <= val < 60):
            raise ValueError(f"Layer index out of [0, 59]: {val}")
        if val not in seen:
            seen.add(val)
            out.append(val)
    return tuple(out)


def read_t16c_norms_and_header(path: Path) -> Tuple[Tuple[bytes, int, int, int, int, int], List[Tuple[str, np.ndarray]], int]:
    """Read exact T16C v3 24-byte header and ordered norm records."""
    with open(path, "rb") as fp:
        hdr = struct.unpack("<4s5I", fp.read(24))
        magic, ver, flags, n_norms, n_mats, reserved = hdr
        if magic != T16C_MAGIC or ver != 1:
            raise ValueError(f"Unsupported T16C header in {path}: magic={magic!r}, ver={ver}")
        norms: List[Tuple[str, np.ndarray]] = []
        for _ in range(n_norms):
            (nlen,) = struct.unpack("<H", fp.read(2))
            name = fp.read(nlen).decode("utf-8")
            (dlen,) = struct.unpack("<I", fp.read(4))
            arr = np.frombuffer(fp.read(dlen * 4), dtype="<f4").astype(np.float32, copy=True)
            norms.append((name, arr))
        mat_start_pos = fp.tell()
    return hdr, norms, mat_start_pos


def write_t16c_header_and_norms(
    fp,
    hdr: Tuple[bytes, int, int, int, int, int],
    norms: Sequence[Tuple[str, np.ndarray]],
) -> int:
    """Write T16C v3 header and norm records cleanly, returning first matrix offset."""
    magic, ver, flags, _, n_mats, reserved = hdr
    fp.seek(0)
    fp.write(struct.pack("<4s5I", magic, ver, flags, len(norms), n_mats, reserved))
    for name, arr in norms:
        nb = name.encode("utf-8")
        v = np.ascontiguousarray(arr, dtype="<f4").reshape(-1)
        fp.write(struct.pack("<H", len(nb)))
        fp.write(nb)
        fp.write(struct.pack("<I", int(v.size)))
        fp.write(v.tobytes())
    return fp.tell()


class NativeMemEval:
    """Thread-configured NEON v2 forward evaluator on in-memory packed T16C buffers."""

    def __init__(self, so_path: Optional[Path] = None, threads: int = 4, a_bits: int = 6) -> None:
        self.threads = int(threads)
        self.a_bits = int(a_bits)
        self._lib = None
        if so_path is not None and Path(so_path).is_file() and os.name == "posix":
            lib = ctypes.CDLL(str(so_path))
            lib.neon_set_andvcnt_abits.argtypes = [ctypes.c_int]
            lib.neon_set_andvcnt_abits.restype = ctypes.c_int
            lib.neon_set_andvcnt_abits(self.a_bits)
            lib.inv_s_and_fwht_f32_c.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.inv_s_and_fwht_f32_c.restype = ctypes.c_int
            lib.gemm_bitserial_tile16_sparse_batch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            lib.gemm_bitserial_tile16_sparse_batch.restype = ctypes.c_int
            lib.int8_lowrank_add_batch_c.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.int8_lowrank_add_batch_c.restype = ctypes.c_int
            self._lib = lib

    def linear(
        self,
        x_in: np.ndarray,
        d_out: int,
        d_in: int,
        bsize: int,
        n_bits: int,
        is_asym: bool,
        has_fwht: bool,
        pb_arr: np.ndarray | bytes,
        inv_s: Optional[np.ndarray],
        low_rank: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Evaluate packed Tile-16 / COLBLOCK weight buffer on `x_in [K, d_in]`."""
        x_c = np.ascontiguousarray(x_in, dtype=np.float32)
        k_tok = int(x_c.shape[0])
        if self._lib is None:
            w = decode_pbytes_to_f32_np(pb_arr, d_out, d_in, bsize, n_bits, is_asym, has_fwht, inv_s)
            y = x_c @ w.T
            if low_rank is not None:
                su, sv, ui8, vi8 = low_rank
                v_w = vi8.astype(np.float32) * sv[:, None]
                u_w = ui8.astype(np.float32) * su[:, None]
                y = y + (x_c @ v_w.T) @ u_w.T
            return np.ascontiguousarray(y, dtype=np.float32)

        x_rot = np.array(x_c, dtype=np.float32, copy=True, order="C")
        inv_ptr = inv_s.ctypes.data if inv_s is not None else 0
        fwht_b = int(bsize) if has_fwht else 1
        rc_r = self._lib.inv_s_and_fwht_f32_c(x_rot.ctypes.data, inv_ptr, k_tok, int(d_in), fwht_b)
        if rc_r != 0:
            raise RuntimeError(f"inv_s_and_fwht_f32_c failed rc={rc_r}")

        pb_u8 = pb_arr if isinstance(pb_arr, np.ndarray) else np.frombuffer(pb_arr, dtype=np.uint8)
        pb_u8 = np.ascontiguousarray(pb_u8, dtype=np.uint8)
        y = np.zeros((k_tok, int(d_out)), dtype=np.float32)
        rc_g = self._lib.gemm_bitserial_tile16_sparse_batch(
            pb_u8.ctypes.data,
            0,
            x_rot.ctypes.data,
            y.ctypes.data,
            int(d_out),
            int(d_in),
            int(bsize),
            int(n_bits),
            1 if is_asym else 0,
            k_tok,
            np.float32(0.0),
            0,
            0,
            self.threads,
            0,
        )
        if rc_g != 0:
            raise RuntimeError(f"gemm_bitserial_tile16_sparse_batch failed rc={rc_g}")

        if low_rank is not None:
            su, sv, ui8, vi8 = low_rank
            rk = int(sv.shape[0])
            su_c = np.ascontiguousarray(su, dtype=np.float32)
            sv_c = np.ascontiguousarray(sv, dtype=np.float32)
            ui_c = np.ascontiguousarray(ui8, dtype=np.int8)
            vi_c = np.ascontiguousarray(vi8, dtype=np.int8)
            rc_lr = self._lib.int8_lowrank_add_batch_c(
                ui_c.ctypes.data,
                su_c.ctypes.data,
                vi_c.ctypes.data,
                sv_c.ctypes.data,
                x_c.ctypes.data,
                y.ctypes.data,
                int(d_out),
                int(d_in),
                rk,
                k_tok,
                self.threads,
            )
            if rc_lr != 0:
                raise RuntimeError(f"int8_lowrank_add_batch_c failed rc={rc_lr}")
        return y


def _fwht_blocks_np(x: np.ndarray, bsize: int) -> np.ndarray:
    """Vectorized orthonormal Sylvester-Hadamard FWHT along last dim in block chunks."""
    orig_shape = x.shape
    d = int(orig_shape[-1])
    v = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, d // bsize, bsize).copy()
    step = 1
    while step < bsize:
        v2 = v.reshape(-1, bsize // (2 * step), 2, step)
        a = v2[:, :, 0, :].copy()
        b = v2[:, :, 1, :].copy()
        v2[:, :, 0, :] = a + b
        v2[:, :, 1, :] = a - b
        step *= 2
    v *= 1.0 / math.sqrt(float(bsize))
    return v.reshape(orig_shape)


def decode_pbytes_to_f32_np(
    pb_arr: np.ndarray | bytes,
    d_out: int,
    d_in: int,
    bsize: int,
    n_bits: int,
    is_asym: bool,
    has_fwht: bool,
    inv_s: Optional[np.ndarray],
) -> np.ndarray:
    """Reference NumPy decoder for Tile-16 / COLBLOCK Gaussian/uniform/ternary payloads."""
    is_cb = bool(n_bits & 0x20)
    is_rm = bool(n_bits & 0x40)
    is_gauss = bool(n_bits & 0x80)
    eff = n_bits & 0x1F
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    hdr = 64 if is_asym else 32
    if eff == 5:
        raise ValueError("Reference decoder expects bit-serial mode 1..4")
    bpr = bsize // 8
    plane_b = bpr * 16
    tblk = hdr + eff * plane_b
    raw = np.frombuffer(pb_arr, dtype=np.uint8) if isinstance(pb_arr, bytes) else np.asarray(pb_arr, dtype=np.uint8)
    arr = raw.reshape(n_blks, n_tiles, tblk).transpose(1, 0, 2) if is_cb else raw.reshape(n_tiles, n_blks, tblk)
    arr = np.ascontiguousarray(arr)

    sc = arr[:, :, :32].view(np.float16).astype(np.float32)  # [n_tiles, n_blks, 16]
    bi = arr[:, :, 32:64].view(np.float16).astype(np.float32) if is_asym else np.zeros_like(sc)

    codes = np.zeros((n_tiles, n_blks, 16, bsize), dtype=np.int32)
    for p in range(eff):
        pl = arr[:, :, hdr + p * plane_b : hdr + (p + 1) * plane_b]
        if is_rm:
            pl_bytes = pl.reshape(n_tiles, n_blks, 16, bpr)
        else:
            pl_bytes = np.ascontiguousarray(pl.reshape(n_tiles, n_blks, bpr, 16).transpose(0, 1, 3, 2))
        bits = np.unpackbits(pl_bytes, axis=-1, bitorder="big").reshape(n_tiles, n_blks, 16, bsize)
        codes |= (bits.astype(np.int32) << p)

    if is_gauss and eff == 2:
        lv = np.array([-3.174, -1.0, 1.0, 3.174], dtype=np.float32)
        w_b = sc[:, :, :, None] * lv[codes]
    elif is_gauss and eff == 3:
        lv = np.array([-7.430, -5.430, -3.270, -1.270, 1.270, 3.270, 5.430, 7.430], dtype=np.float32)
        w_b = sc[:, :, :, None] * lv[codes]
    else:
        max_lvl = float((1 << eff) - 1)
        w_b = bi[:, :, :, None] + sc[:, :, :, None] * (2.0 * codes.astype(np.float32) - max_lvl)

    w = w_b.transpose(0, 2, 1, 3).reshape(n_tiles * 16, d_in)[:d_out, :]
    w = np.ascontiguousarray(w, dtype=np.float32)
    if has_fwht:
        w = _fwht_blocks_np(w, bsize)
    if inv_s is not None:
        w = w * inv_s[None, :]
    return np.ascontiguousarray(w, dtype=np.float32)


def pack_2b_gauss_rtn_np(w_rot: np.ndarray, bsize: int, is_colblock: bool) -> np.ndarray:
    """Reference NumPy Lloyd-Max 2b_gauss packer used when native lib is unavailable."""
    d_out, d_in = w_rot.shape
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    lv = np.array([-3.174, -1.0, 1.0, 3.174], dtype=np.float32)
    mid = np.array([-2.087, 0.0, 2.087], dtype=np.float32)
    pad_rows = n_tiles * 16 - d_out
    wp = np.pad(w_rot, ((0, pad_rows), (0, 0))) if pad_rows > 0 else w_rot
    wv = wp.reshape(n_tiles, 16, n_blks, bsize).transpose(0, 2, 1, 3)  # [T, B, 16, bsz]

    sc = 0.470 * np.sqrt(np.mean(wv * wv, axis=-1))
    for _ in range(4):
        inv = np.where(sc > 1e-12, 1.0 / np.maximum(sc, 1e-12), 0.0)[:, :, :, None]
        qi = np.sum((wv * inv)[:, :, :, :, None] >= mid[None, None, None, None, :], axis=-1)
        ql = lv[qi]
        den = np.sum(ql * ql, axis=-1)
        sc = np.where(den > 0.0, np.sum(ql * wv, axis=-1) / np.maximum(den, 1e-12), 0.0)
    sc16 = sc.astype(np.float16)
    scf = sc16.astype(np.float32)
    inv = np.where(scf > 1e-12, 1.0 / np.maximum(scf, 1e-12), 0.0)[:, :, :, None]
    qi = np.sum((wv * inv)[:, :, :, :, None] >= mid[None, None, None, None, :], axis=-1).astype(np.uint8)

    bpr = bsize // 8
    plane_b = bpr * 16
    tblk = 32 + 2 * plane_b
    out = np.zeros((n_tiles, n_blks, tblk), dtype=np.uint8)
    out[:, :, :32] = np.ascontiguousarray(sc16).view(np.uint8).reshape(n_tiles, n_blks, 32)
    for p in range(2):
        bits = ((qi >> p) & 1).astype(np.uint8)
        pk = np.packbits(bits, axis=-1, bitorder="big")  # [T, B, 16, bpr]
        out[:, :, 32 + p * plane_b : 32 + (p + 1) * plane_b] = np.ascontiguousarray(
            pk.transpose(0, 1, 3, 2)
        ).reshape(n_tiles, n_blks, plane_b)
    if is_colblock:
        return np.ascontiguousarray(out.transpose(1, 0, 2)).reshape(-1)
    return np.ascontiguousarray(out).reshape(-1)


def reorder_t16c_colblock(pb: np.ndarray, d_out: int, d_in: int, bsize: int, eff_bits: int, is_asym: bool, to_cb: bool) -> np.ndarray:
    """Convert packed Tile-16 buffer between TileMajor and COLBLOCK layout cleanly."""
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    hdr = 64 if is_asym else 32
    tblk = hdr + eff_bits * (bsize // 8) * 16
    arr = np.ascontiguousarray(pb, dtype=np.uint8)
    if to_cb:
        return np.ascontiguousarray(arr.reshape(n_tiles, n_blks, tblk).transpose(1, 0, 2)).reshape(-1)
    return np.ascontiguousarray(arr.reshape(n_blks, n_tiles, tblk).transpose(1, 0, 2)).reshape(-1)


def compute_geglu_colblock_perm(
    act_samples: np.ndarray,
    shared_blocks: int = 8,
    block_size: int = 256,
    iters: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Losslessly permute GeGLU channels into `shared_blocks` heavy + co-firing 256-col blocks.

    Returns `(perm, inv_perm)` where `w_g_new = w_g[perm]`, `w_u_new = w_u[perm]`,
    and `w_d_new = w_d[:, perm]`, guaranteeing exact functional equivalence before quantization.
    """
    act = np.asarray(act_samples, dtype=np.float32)
    assert act.ndim == 2
    n_tok, ff = act.shape
    if ff % block_size != 0:
        raise ValueError(f"ff={ff} must be divisible by block_size={block_size}")
    n_blocks = ff // block_size
    if not (0 <= shared_blocks < n_blocks):
        raise ValueError(f"shared_blocks={shared_blocks} invalid for n_blocks={n_blocks}")

    energy = np.mean(act * act, axis=0)
    order_e = np.argsort(-energy, kind="stable")
    n_shared_c = shared_blocks * block_size
    shared_ch = order_e[:n_shared_c]
    rem_ch = order_e[n_shared_c:]

    n_rem_blocks = n_blocks - shared_blocks
    if n_rem_blocks <= 1:
        perm = np.concatenate([shared_ch, rem_ch]).astype(np.int64)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(ff, dtype=np.int64)
        return perm, inv

    a_rem = act[:, rem_ch]
    norms = np.linalg.norm(a_rem, axis=0, keepdims=True) + 1e-8
    a_norm = (a_rem / norms).astype(np.float32)  # [N, C_rem]

    # Deterministic balanced spherical k-means assignment across n_rem_blocks buckets of size block_size
    c_rem = len(rem_ch)
    assign = np.repeat(np.arange(n_rem_blocks, dtype=np.int32), block_size)
    rng = np.random.default_rng(seed)
    centroids = np.zeros((n_rem_blocks, n_tok), dtype=np.float32)
    for b in range(n_rem_blocks):
        mu = a_norm[:, assign == b].mean(axis=1)
        n_mu = float(np.linalg.norm(mu))
        centroids[b] = (mu / n_mu) if n_mu > 1e-8 else rng.normal(0.0, 1.0, size=(n_tok,)).astype(np.float32)

    for _ in range(iters):
        sims = centroids @ a_norm  # [B, C_rem]
        # Rank every channel by peak affinity confidence (best - second_best margin) and greedily fill capacity
        sorted_vals = np.partition(sims, -2, axis=0)
        margins = sorted_vals[-1] - sorted_vals[-2]
        ch_visit = np.argsort(-margins, kind="stable")
        pref = np.argsort(-sims, axis=0)  # [B, C_rem]
        counts = np.zeros(n_rem_blocks, dtype=np.int32)
        new_assign = np.empty(c_rem, dtype=np.int32)
        # Maintain smallest-index tie break cleanly via pass over preference ranks
        avail = set(range(n_rem_blocks))
        for ci in ch_visit:
            placed = False
            for r in range(n_rem_blocks):
                b = int(pref[r, ci])
                if counts[b] < block_size:
                    new_assign[ci] = b
                    counts[b] += 1
                    if counts[b] == block_size:
                        avail.discard(b)
                    placed = True
                    break
            if not placed:
                b = min(avail)
                new_assign[ci] = b
                counts[b] += 1
                if counts[b] == block_size:
                    avail.discard(b)
        assign = new_assign
        for b in range(n_rem_blocks):
            mu = a_norm[:, assign == b].mean(axis=1)
            n_mu = float(np.linalg.norm(mu))
            if n_mu > 1e-8:
                centroids[b] = mu / n_mu

    rem_grouped = np.concatenate([rem_ch[assign == b] for b in range(n_rem_blocks)])
    perm = np.concatenate([shared_ch, rem_grouped]).astype(np.int64)
    assert len(perm) == ff
    inv = np.empty(ff, dtype=np.int64)
    inv[perm] = np.arange(ff, dtype=np.int64)
    return perm, inv


def solve_wiener_post_norm_np(
    z_tr: np.ndarray,
    dy_tr: np.ndarray,
    z_va: np.ndarray,
    dy_va: np.ndarray,
    g0: np.ndarray,
    max_rel_delta: float = 0.35,
    lambdas: Tuple[float, ...] = (2.0, 0.5, 0.1, 0.02),
) -> Tuple[np.ndarray, float, float]:
    """Closed-form elementwise Wiener post-norm solver with strict held-out MSE rollback."""
    g_base = np.asarray(g0, dtype=np.float32).copy()
    mse0 = float(np.mean((z_va.astype(np.float32) * g_base[None, :] - dy_va.astype(np.float32)) ** 2))
    best_g, best_mse = g_base, mse0
    num = np.mean(z_tr.astype(np.float32) * dy_tr.astype(np.float32), axis=0)
    den = np.mean(z_tr.astype(np.float32) * z_tr.astype(np.float32), axis=0)
    lim = np.maximum(np.abs(g_base), 1e-4) * float(max_rel_delta)
    for lam in lambdas:
        raw = (num + lam * g_base) / np.maximum(den + lam, 1e-8)
        cand = (g_base + np.clip(raw - g_base, -lim, lim)).astype(np.float32)
        m = float(np.mean((z_va.astype(np.float32) * cand[None, :] - dy_va.astype(np.float32)) ** 2))
        if m < best_mse * 0.999:
            best_mse, best_g = m, cand
    return np.ascontiguousarray(best_g, dtype=np.float32), mse0, best_mse


def solve_wiener_layer_scalar_np(
    pred_tr: np.ndarray,
    targ_tr: np.ndarray,
    pred_va: np.ndarray,
    targ_va: np.ndarray,
    lsc0: float,
    lambdas: Tuple[float, ...] = (2.0, 0.5, 0.1, 0.02),
) -> Tuple[float, float, float]:
    """Closed-form scalar Wiener gain for Gemma 4 `layer_scalar` with held-out rollback."""
    mse0 = float(np.mean((pred_va.astype(np.float32) * float(lsc0) - targ_va.astype(np.float32)) ** 2))
    best_s, best_mse = float(lsc0), mse0
    num = float(np.mean(pred_tr.astype(np.float32) * targ_tr.astype(np.float32)))
    den = float(np.mean(pred_tr.astype(np.float32) * pred_tr.astype(np.float32)))
    lo = 0.70 * abs(float(lsc0))
    hi = 1.35 * max(abs(float(lsc0)), 1e-4)
    for lam in lambdas:
        cand = (num + lam * float(lsc0)) / max(den + lam, 1e-8)
        cand = max(lo, min(hi, cand))
        m = float(np.mean((pred_va.astype(np.float32) * cand - targ_va.astype(np.float32)) ** 2))
        if m < best_mse * 0.999:
            best_mse, best_s = m, cand
    return float(best_s), mse0, best_mse


def quantize_2b_gauss_gptq(
    w_orig: np.ndarray,
    x_cal: np.ndarray,
    x_val: np.ndarray,
    use_colblock: bool,
    rk: int,
    hblk: int,
    mem_eval: NativeMemEval,
    libv2: Optional[ctypes.CDLL],
    ops: c3._NativeCalOps,
    prompt_starts: Sequence[int] = (0,),
    bsize: int = BSIZE_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict]:
    """SmoothQuant + FWHT-256 + chunked GPTQ + AR row scales + held-out ActALS `2b_gauss`."""
    t0 = time.perf_counter()
    d_out, d_in = int(w_orig.shape[0]), int(w_orig.shape[1])
    x_fit = np.ascontiguousarray(x_cal, dtype=np.float32)
    x_v = np.ascontiguousarray(x_val, dtype=np.float32)
    y_tgt_fit = ops.mm_xt(x_fit, w_orig)
    y_tgt_val = ops.mm_xt(x_v, w_orig)

    sq_s = c3.compute_sq(x_fit[:min(len(x_fit), 256)], w_orig, alpha=0.5)
    inv_s = (1.0 / sq_s).astype(np.float32)

    w_rot = np.array(w_orig, dtype=np.float32, copy=True, order="C") * sq_s[None, :]
    x_rot = np.array(x_fit[:min(len(x_fit), 256)], dtype=np.float32, copy=True, order="C") * inv_s[None, :]
    if ops.lib_g is not None:
        ops.lib_g.fwht_rows_f32(w_rot.ctypes.data, d_out, d_in, bsize)
        ops.lib_g.fwht_rows_f32(x_rot.ctypes.data, x_rot.shape[0], d_in, bsize)
    else:
        w_rot = _fwht_blocks_np(w_rot, bsize)
        x_rot = _fwht_blocks_np(x_rot, bsize)

    if libv2 is not None and ops.lib_g is not None:
        n_tiles = (d_out + 15) // 16
        n_blks = d_in // bsize
        tblk = 32 + 2 * (bsize // 8) * 16
        pb_tm = np.empty(n_tiles * n_blks * tblk, dtype=np.uint8)
        for c0 in range(0, d_in, hblk):
            c1 = min(d_in, c0 + hblk)
            cw = c1 - c0
            xc = np.ascontiguousarray(x_rot[:, c0:c1], dtype=np.float32)
            wc = np.ascontiguousarray(w_rot[:, c0:c1], dtype=np.float32)
            u_buf = np.zeros((cw, cw), dtype=np.float32)
            rc_u = libv2.build_gptq_u_fast_f32(xc.ctypes.data, int(xc.shape[0]), int(cw), np.float32(0.01), u_buf.ctypes.data)
            if rc_u != 0:
                raise RuntimeError(f"build_gptq_u_fast_f32 failed rc={rc_u}")
            pb_c = np.empty(n_tiles * (cw // bsize) * tblk, dtype=np.uint8)
            libv2.gptq_quant_and_pack_v2_f32(
                wc.ctypes.data, u_buf.ctypes.data, int(d_out), int(cw), int(bsize), 2, 0, 1, pb_c.ctypes.data
            )
            b0 = c0 // bsize
            nb_c = cw // bsize
            pb_tm.reshape(n_tiles, n_blks, tblk)[:, b0 : b0 + nb_c, :] = pb_c.reshape(n_tiles, nb_c, tblk)
        pb_out = reorder_t16c_colblock(pb_tm, d_out, d_in, bsize, 2, False, True) if use_colblock else pb_tm
        del pb_tm
    else:
        pb_out = pack_2b_gauss_rtn_np(w_rot, bsize, is_colblock=use_colblock)

    del w_rot, x_rot
    nb_flag = NB_2B_GAUSS_CB if use_colblock else NB_2B_GAUSS_TM

    # Exact native-kernel AR row scale optimization on fit activations
    yq_raw = mem_eval.linear(x_fit, d_out, d_in, bsize, nb_flag, False, True, pb_out, inv_s, None)
    x_bal, sf_fit = c3.balance_tokens_with_scale(x_fit, list(prompt_starts), bos_weight=0.15)
    yt_b = y_tgt_fit * sf_fit
    yq_b = yq_raw * sf_fit
    ar = np.clip(
        np.sum(yt_b * yq_b, axis=0, dtype=np.float64) / (np.sum(yq_b * yq_b, axis=0, dtype=np.float64) + 1e-12),
        0.50,
        2.00,
    ).astype(np.float32)

    if use_colblock:
        pb_tm_tmp = reorder_t16c_colblock(pb_out, d_out, d_in, bsize, 2, False, False)
        if ops.lib_g is not None:
            c4.scale_t16c_pbytes_rows(pb_tm_tmp, d_out, d_in, bsize, MODE_2B_GAUSS, ar)
        else:
            _scale_tilemajor_sym_np(pb_tm_tmp, d_out, d_in, bsize, 2, ar)
        pb_out = reorder_t16c_colblock(pb_tm_tmp, d_out, d_in, bsize, 2, False, True)
        del pb_tm_tmp
    else:
        if ops.lib_g is not None:
            c4.scale_t16c_pbytes_rows(pb_out, d_out, d_in, bsize, MODE_2B_GAUSS, ar)
        else:
            _scale_tilemajor_sym_np(pb_out, d_out, d_in, bsize, 2, ar)

    yq_ar_fit = mem_eval.linear(x_fit, d_out, d_in, bsize, nb_flag, False, True, pb_out, inv_s, None)
    yq_ar_val = mem_eval.linear(x_v, d_out, d_in, bsize, nb_flag, False, True, pb_out, inv_s, None)
    mse_val0 = float(np.mean((yq_ar_val - y_tgt_val) ** 2))
    cos_val0 = c3.cos_sim(yq_ar_val, y_tgt_val)

    lr_tuple: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    cos_val1 = cos_val0
    mse_val1 = mse_val0
    if rk > 0:
        pinv_t = c3.precompute_x_pinv_t(x_bal, ops)
        y_err = np.ascontiguousarray(((y_tgt_fit - yq_ar_fit) * sf_fit).T, dtype=np.float32)
        u_q, v_q, su, sv, u_i8, v_i8 = c3.fit_act_als_i8(y_err, x_bal, pinv_t, rk, ops)
        cand_lr = (su, sv, u_i8, v_i8)
        yq_lr_val = mem_eval.linear(x_v, d_out, d_in, bsize, nb_flag, False, True, pb_out, inv_s, cand_lr)
        mse_cand = float(np.mean((yq_lr_val - y_tgt_val) ** 2))
        if mse_cand < mse_val0 * 0.999:
            lr_tuple = cand_lr
            mse_val1 = mse_cand
            cos_val1 = c3.cos_sim(yq_lr_val, y_tgt_val)
        del y_err, u_q, v_q, pinv_t

    stats = {
        "cos_before_lr": round(cos_val0, 5),
        "cos_after_lr": round(cos_val1, 5),
        "mse_before_lr": round(mse_val0, 6),
        "mse_after_lr": round(mse_val1, 6),
        "accepted_rk": int(lr_tuple[1].shape[0]) if lr_tuple is not None else 0,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return pb_out, inv_s, lr_tuple, stats


def _scale_tilemajor_sym_np(pb: np.ndarray, d_out: int, d_in: int, bsize: int, eff: int, ar: np.ndarray) -> None:
    """Reference in-place FP16 row scale multiplier for symmetric Tile-16 buffer."""
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    tblk = 32 + eff * (bsize // 8) * 16
    view = pb.reshape(n_tiles, n_blks, tblk)
    sc = view[:, :, :32].view(np.float16).astype(np.float32)
    ar_pad = np.pad(ar, (0, n_tiles * 16 - d_out), constant_values=1.0).reshape(n_tiles, 1, 16)
    sc_new = (sc * ar_pad).astype(np.float16)
    view[:, :, :32] = sc_new.view(np.uint8).reshape(n_tiles, n_blks, 32)


def _select_calib_and_val_batches(
    work_dir: Path,
    cfg: dict,
    n_layers: int,
) -> Tuple[List[int], List[Tuple[int, int]], List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Load exact W4A16 Teacher calibration & held-out trajectories from `work_dir`."""
    meta = json.loads((work_dir / "meta.json").read_text(encoding="utf-8"))
    ids_all = [int(x) for x in meta["all_ids"]]
    cal_end = int(meta["cal_end"])
    cal_ids = ids_all[:cal_end]
    t2_a, t2_b = int(meta["test2"][0]), int(meta["test2"][1])
    tf_a, tf_b = int(meta["testf"][0]), int(meta["testf"][1])
    pstarts = [i for i, t in enumerate(cal_ids) if t == 2]
    if not pstarts or pstarts[0] != 0:
        pstarts = [0] + pstarts
    seg_bounds = pstarts + [len(cal_ids)]
    cal_segs = list(zip(seg_bounds[:-1], seg_bounds[1:]))

    xcal_t: List[np.ndarray] = []
    xt2_t: List[np.ndarray] = []
    xtf_t: List[np.ndarray] = []
    for l in range(n_layers):
        with np.load(work_dir / f"snaps_L{l:02d}.npz") as z:
            xcal_t.append(z["x_cal"].astype(np.float32))
            xt2_t.append(z["x_t2"].astype(np.float32))
            xtf_t.append(z["x_tf"].astype(np.float32))
    x_after = np.load(work_dir / f"x_after_L{n_layers - 1}.npy").astype(np.float32)
    xcal_last = x_after[:cal_end]
    xt2_t.append(x_after[t2_a:t2_b])
    xtf_t.append(x_after[tf_a:tf_b])
    del x_after
    return cal_ids, cal_segs, pstarts, xcal_t, xt2_t, xtf_t, xcal_last


def _layer_attn_ctx(
    yq: np.ndarray,
    yk: np.ndarray,
    yv: np.ndarray,
    qn: np.ndarray,
    kn: np.ndarray,
    nh: int,
    nkv: int,
    hd: int,
    rot_d: int,
    theta: float,
    eps: float,
    head_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact causal multi-head attention context matching Gemma 4 31B sliding/global geometry."""
    t_len = len(yq)
    gqa = nh // nkv
    qt = c3.rope(c3.rmsn(yq.reshape(t_len, nh, hd), qn, eps), theta, rot_d, hd)
    kt = c3.rope(c3.rmsn(yk.reshape(t_len, nkv, hd), kn, eps), theta, rot_d, hd)
    vt = c3.rmsn(yv.reshape(t_len, nkv, hd), None, eps)
    c_out = np.empty((t_len, nh, hd), dtype=np.float32)
    m_c = np.triu(np.full((t_len, t_len), -1e9, dtype=np.float32), 1)
    for h_i in range(nh):
        kv_h = h_i // gqa
        s_m = qt[:, h_i, :] @ kt[:, kv_h, :].T + m_c
        s_m -= np.max(s_m, axis=-1, keepdims=True)
        p_m = np.exp(s_m)
        p_m /= np.sum(p_m, axis=-1, keepdims=True)
        c_out[:, h_i, :] = p_m @ vt[:, kv_h, :]
    if head_mask is not None:
        c_out *= head_mask[:, :, None]
    return c_out.reshape(t_len, nh * hd)


def _attach_c8db_radicals_from_file(path: Path, entry: dict) -> dict:
    """Attach raw 4-byte `expert_radicals` read from `path` so preserved C8DB entries serialize bit-identically."""
    out = dict(entry)
    target_name = str(entry["name"])
    with open(path, "rb") as f:
        magic, _ver, n_norms, n_entries, _flags, _res = struct.unpack("<4s5I", f.read(24))
        if magic != c8d.C8DB_MAGIC:
            raise ValueError(f"Invalid C8DB magic {magic!r} in {path}")
        pos = 24
        for _ in range(n_norms):
            nlen = struct.unpack("<H", f.read(2))[0]
            f.seek(nlen, 1)
            cnt = struct.unpack("<I", f.read(4))[0]
            f.seek(cnt * 4, 1)
            pos += 2 + nlen + 4 + cnt * 4
        for _ in range(n_entries):
            hdr = f.read(40)
            nlen, _d_o, _d_i, _rk, n_exp, _tk, _r0, _r1, _rc, _rf, _ra, _bt, _tp, _gs, _r2, pbytes = struct.unpack(
                "<HIIHBBBBBBBBffIQ", hdr
            )
            ename = f.read(nlen).decode("utf-8")
            pos += 40 + nlen
            pad = (64 - (pos % 64)) % 64
            if pad:
                f.seek(pad, 1)
                pos += pad
            if ename == target_name:
                f.seek(n_exp * 8 * 4, 1)
                rad_raw = f.read(n_exp * 4)
                out["expert_radicals"] = [tuple(rad_raw[i * 4 : (i + 1) * 4]) for i in range(n_exp)]
                return out
            f.seek(pbytes, 1)
            pos += pbytes
    out["expert_radicals"] = [tuple(entry["c6d"])] * len(entry["experts"])
    return out


def build_v9_2_cliff_fix_vlm(args: argparse.Namespace) -> int:
    """Execute full surgical V9.2-CliffFix-VLM build, Wiener norms, C8DB, and CRTR training."""
    t_start = time.perf_counter()
    src_t16c = args.src_t16c.resolve()
    sf_path = args.model.resolve()
    cfg_path = args.config.resolve()
    work_dir = args.work_dir.resolve()
    out_dir = args.out_t16c.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    work_layers_dir = out_dir / "v92_tmp_layers"
    work_layers_dir.mkdir(parents=True, exist_ok=True)

    upgrade_full = set(parse_layer_csv(args.upgrade_layers))
    upgrade_gu = set(parse_layer_csv(args.upgrade_gu_layers)) - upgrade_full
    upgrade_down = set(parse_layer_csv(args.upgrade_down_layers)) - upgrade_full - upgrade_gu
    print(
        f"[V9.2 INIT] src={src_t16c} upgrade_full={sorted(upgrade_full)} "
        f"upgrade_gu={sorted(upgrade_gu)} upgrade_down={sorted(upgrade_down)} hblk={args.hblk} rk={args.rk}",
        flush=True,
    )

    cfg_full = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = cfg_full["text_config"]
    eps = float(cfg["rms_norm_eps"])
    d_mod = int(cfg["hidden_size"])
    ff_dim = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    nh = int(cfg.get("num_attention_heads", 32))
    nkv_s = int(cfg.get("num_key_value_heads", 16))
    nkv_f = int(cfg.get("num_global_key_value_heads", 4))
    hd_s = int(cfg.get("head_dim", 256))
    hd_f = int(cfg.get("global_head_dim", 512))

    cal_ids, cal_segs, pstarts, xcal_t, xt2_t, xtf_t, xcal_last = _select_calib_and_val_batches(work_dir, cfg, n_layers)

    # Load helpers cleanly
    so_path = args.so.resolve() if args.so else (THIS_DIR.parent.parent / "libneon_stream_accelerator.so")
    mem_eval = NativeMemEval(so_path, threads=args.threads, a_bits=6)
    nlin_file = b91.NativeNeonLinear(so_path, threads=args.threads)
    if nlin_file.lib is not None:
        nlin_file.lib.neon_set_andvcnt_abits(6)
    ops = c3._NativeCalOps(work_dir, require_native=True)
    libv2 = c4._load_v2_gptq_lib(work_dir, require_native=True)
    hdr, norms_list, _ = read_t16c_norms_and_header(src_t16c)
    norm_dict: Dict[str, np.ndarray] = {k: v.copy() for k, v in norms_list}
    _, _, mat_map, mat_names = b91.parse_t16c_metadata(src_t16c)

    # Student cascaded trajectories initialized directly after quantized embeddings (matches V4/V9.1 embed)
    xc_c = xcal_t[0].copy()
    xt_c = xt2_t[0].copy()
    xf_c = xtf_t[0].copy()

    def eval_src_mat(fp_src, key: str, x_in: np.ndarray) -> np.ndarray:
        meta_m = mat_map[key]
        return nlin_file.linear(fp_src, meta_m, np.ascontiguousarray(x_in, dtype=np.float32))

    # Reserve exact C8DB entries: always preserve V4.3 vision projection, retrain text bottlenecks
    c8_norms: Dict[str, np.ndarray] = {}
    c8_entries: Dict[str, dict] = {}
    if args.v43_c8d is not None and args.v43_c8d.is_file():
        _, v43_entries = c8d.read_c8db_file(str(args.v43_c8d))
        vis_key = "model.embed_vision.embedding_projection.weight"
        if vis_key in v43_entries:
            c8_entries[vis_key] = _attach_c8db_radicals_from_file(args.v43_c8d, v43_entries[vis_key])
            print(f"[V9.2 C8DB] Preserved Vision adapter {vis_key} (rank={v43_entries[vis_key]['rank']})", flush=True)

    c8_target_layers = set(parse_layer_csv(args.c8d_layers))
    crtr_layers: List[dict] = []
    head_crtr_layers: List[dict] = []
    layer_profiles: List[dict] = []

    with open(src_t16c, "rb") as fp_src, c8d.MmapSafetensorsReader(sf_path) as sf:
        # Phase 1: Stream all 60 layers sequentially, upgrading cliff MLPs & collecting routers + Wiener norms
        for l_idx in range(n_layers):
            tl0 = time.perf_counter()
            pfx = f"model.language_model.layers.{l_idx}"
            is_full = cfg["layer_types"][l_idx] == "full_attention"
            hd = hd_f if is_full else hd_s
            nkv = nkv_f if is_full else nkv_s
            rot_d = 128 if is_full else 256
            theta = 1e6 if is_full else 1e4

            g_in = norm_dict[f"{pfx}.input_layernorm.weight"]
            qn = norm_dict[f"{pfx}.self_attn.q_norm.weight"]
            kn = norm_dict[f"{pfx}.self_attn.k_norm.weight"]
            g_pa = norm_dict[f"{pfx}.post_attention_layernorm.weight"]
            g_pf = norm_dict[f"{pfx}.pre_feedforward_layernorm.weight"]
            g_pff = norm_dict[f"{pfx}.post_feedforward_layernorm.weight"]
            lsc = float(np.asarray(norm_dict[f"{pfx}.layer_scalar"]).reshape(-1)[0])

            # 1a. Student Attention Forward (using resident V9.1 attention projections)
            inp_cal = c3.rmsn(xc_c, g_in, eps)
            inp_t2 = c3.rmsn(xt_c, g_in, eps)
            inp_tf = c3.rmsn(xf_c, g_in, eps)

            q_k = f"{pfx}.self_attn.q_proj.weight"
            k_k = f"{pfx}.self_attn.k_proj.weight"
            v_k = f"{pfx}.self_attn.v_proj.weight"
            o_k = f"{pfx}.self_attn.o_proj.weight"

            yq_cal = eval_src_mat(fp_src, q_k, inp_cal)
            yk_cal = eval_src_mat(fp_src, k_k, inp_cal)
            yv_cal = yk_cal if is_full else eval_src_mat(fp_src, v_k, inp_cal)

            yq_t2 = eval_src_mat(fp_src, q_k, inp_t2)
            yk_t2 = eval_src_mat(fp_src, k_k, inp_t2)
            yv_t2 = yk_t2 if is_full else eval_src_mat(fp_src, v_k, inp_t2)

            yq_tf = eval_src_mat(fp_src, q_k, inp_tf)
            yk_tf = eval_src_mat(fp_src, k_k, inp_tf)
            yv_tf = yk_tf if is_full else eval_src_mat(fp_src, v_k, inp_tf)

            ctx_cal = np.concatenate(
                [_layer_attn_ctx(yq_cal[a:b], yk_cal[a:b], yv_cal[a:b], qn, kn, nh, nkv, hd, rot_d, theta, eps) for a, b in cal_segs],
                axis=0,
            )
            ctx_t2 = _layer_attn_ctx(yq_t2, yk_t2, yv_t2, qn, kn, nh, nkv, hd, rot_d, theta, eps)
            ctx_tf = _layer_attn_ctx(yq_tf, yk_tf, yv_tf, qn, kn, nh, nkv, hd, rot_d, theta, eps)

            # Train 32-Head Attention Router (HeadRouter-V9.2) on Student RMSNorm inputs & per-head context energy
            head_e_cal = np.sqrt(np.mean(ctx_cal.reshape(len(ctx_cal), nh, hd) ** 2, axis=-1) + 1e-12)
            head_e_va = np.sqrt(np.mean(ctx_t2.reshape(len(ctx_t2), nh, hd) ** 2, axis=-1) + 1e-12)
            mod_sal_cal = np.linalg.norm(ctx_cal, axis=-1)
            mod_sal_va = np.linalg.norm(ctx_t2, axis=-1)
            thr_h = float(np.quantile(mod_sal_cal, 0.25))
            hr_net, hr_met = crtr.train_layer_router(
                torch.from_numpy(inp_cal),
                torch.from_numpy(head_e_cal),
                torch.from_numpy(inp_t2),
                torch.from_numpy(head_e_va),
                top_p=0.50,
                rank=args.head_crtr_rank,
                steps=args.crtr_steps,
                lr=3e-3,
                mod_targets_tr=torch.from_numpy((mod_sal_cal > thr_h).astype(np.float32)),
                mod_targets_va=torch.from_numpy((mod_sal_va > thr_h).astype(np.float32)),
                device="cpu",
                seed=200 + l_idx,
            )
            head_crtr_layers.append(crtr.quantize_layer_for_crtr(hr_net, block_size=hd_s))

            yo_cal = eval_src_mat(fp_src, o_k, ctx_cal)
            yo_t2 = eval_src_mat(fp_src, o_k, ctx_t2)
            yo_tf = eval_src_mat(fp_src, o_k, ctx_tf)
            xm_cal = xc_c + c3.rmsn(yo_cal, g_pa, eps)
            xm_t2 = xt_c + c3.rmsn(yo_t2, g_pa, eps)
            xm_tf = xf_c + c3.rmsn(yo_tf, g_pa, eps)
            del inp_cal, inp_t2, inp_tf, yq_cal, yk_cal, yv_cal, yq_t2, yk_t2, yv_t2, yq_tf, yk_tf, yv_tf, ctx_cal, ctx_t2, ctx_tf, yo_cal, yo_t2, yo_tf

            # 1b. Pre-feedforward norm + GeGLU MLP
            hm_cal = c3.rmsn(xm_cal, g_pf, eps)
            hm_t2 = c3.rmsn(xm_t2, g_pf, eps)
            hm_tf = c3.rmsn(xm_tf, g_pf, eps)
            xt_teach_next_cal = xcal_t[l_idx + 1] if (l_idx + 1 < n_layers) else xcal_last
            xt_teach_next_t2 = xt2_t[l_idx + 1]
            xt_teach_next_tf = xtf_t[l_idx + 1]

            g_k = f"{pfx}.mlp.gate_proj.weight"
            u_k = f"{pfx}.mlp.up_proj.weight"
            d_k = f"{pfx}.mlp.down_proj.weight"

            mode_tag = f"keep(0x{mat_map[g_k]['n_bits']:02x})"
            perm_pi: Optional[np.ndarray] = None
            layer_upg_stats: dict = {}

            if l_idx in upgrade_full or l_idx in upgrade_gu:
                wg_t = sf.read_qat_rows_f32(f"{pfx}.mlp.gate_proj", 0, ff_dim)
                wu_t = sf.read_qat_rows_f32(f"{pfx}.mlp.up_proj", 0, ff_dim)
                # Compute lossless channel clustering permutation pi_l on Teacher GeGLU activations
                gt_cal = ops.mm_xt(hm_cal, wg_t)
                ut_cal = ops.mm_xt(hm_cal, wu_t)
                act_t_cal = c3.gelu(gt_cal) * ut_cal
                if args.apply_geglu_perm:
                    perm_pi, _ = compute_geglu_colblock_perm(act_t_cal, shared_blocks=8, block_size=BSIZE_DEFAULT, iters=6, seed=42 + l_idx)
                    wg_work = np.ascontiguousarray(wg_t[perm_pi, :], dtype=np.float32)
                    wu_work = np.ascontiguousarray(wu_t[perm_pi, :], dtype=np.float32)
                    act_t_perm = np.ascontiguousarray(act_t_cal[:, perm_pi], dtype=np.float32)
                else:
                    wg_work, wu_work, act_t_perm = wg_t, wu_t, act_t_cal

                pb_g, is_g, lr_g, st_g = quantize_2b_gauss_gptq(
                    wg_work, hm_cal, hm_t2, False, args.rk, args.hblk, mem_eval, libv2, ops, prompt_starts=pstarts
                )
                pb_u, is_u, lr_u, st_u = quantize_2b_gauss_gptq(
                    wu_work, hm_cal, hm_t2, False, args.rk, args.hblk, mem_eval, libv2, ops, prompt_starts=pstarts
                )
                del wg_t, wu_t, wg_work, wu_work, gt_cal, ut_cal, act_t_cal

                g_cal = mem_eval.linear(hm_cal, ff_dim, d_mod, BSIZE_DEFAULT, NB_2B_GAUSS_TM, False, True, pb_g, is_g, lr_g)
                u_cal = mem_eval.linear(hm_cal, ff_dim, d_mod, BSIZE_DEFAULT, NB_2B_GAUSS_TM, False, True, pb_u, is_u, lr_u)
                g_t2 = mem_eval.linear(hm_t2, ff_dim, d_mod, BSIZE_DEFAULT, NB_2B_GAUSS_TM, False, True, pb_g, is_g, lr_g)
                u_t2 = mem_eval.linear(hm_t2, ff_dim, d_mod, BSIZE_DEFAULT, NB_2B_GAUSS_TM, False, True, pb_u, is_u, lr_u)
                g_tf = mem_eval.linear(hm_tf, ff_dim, d_mod, BSIZE_DEFAULT, NB_2B_GAUSS_TM, False, True, pb_g, is_g, lr_g)
                u_tf = mem_eval.linear(hm_tf, ff_dim, d_mod, BSIZE_DEFAULT, NB_2B_GAUSS_TM, False, True, pb_u, is_u, lr_u)

                act_s_cal = c3.gelu(g_cal) * u_cal
                act_s_t2 = c3.gelu(g_t2) * u_t2
                act_s_tf = c3.gelu(g_tf) * u_tf

                if l_idx in upgrade_full:
                    wd_t = sf.read_qat_rows_f32(f"{pfx}.mlp.down_proj", 0, d_mod)
                    wd_work = np.ascontiguousarray(wd_t[:, perm_pi], dtype=np.float32) if perm_pi is not None else wd_t
                    pb_d, is_d, lr_d, st_d = quantize_2b_gauss_gptq(
                        wd_work, act_s_cal, act_s_t2, True, args.rk, args.hblk, mem_eval, libv2, ops, prompt_starts=pstarts
                    )
                    wd_t_orig = wd_work
                    d_cal = mem_eval.linear(act_s_cal, d_mod, ff_dim, BSIZE_DEFAULT, NB_2B_GAUSS_CB, False, True, pb_d, is_d, lr_d)
                    d_t2 = mem_eval.linear(act_s_t2, d_mod, ff_dim, BSIZE_DEFAULT, NB_2B_GAUSS_CB, False, True, pb_d, is_d, lr_d)
                    d_tf = mem_eval.linear(act_s_tf, d_mod, ff_dim, BSIZE_DEFAULT, NB_2B_GAUSS_CB, False, True, pb_d, is_d, lr_d)
                    mode_tag = "2b_gauss_full"
                    layer_upg_stats = {"g": st_g, "u": st_u, "d": st_d}
                    # Save upgraded layer payload incrementally so Stage 2 writes sequentially without buffering in RAM
                    np.savez(
                        work_layers_dir / f"L{l_idx:02d}.npz",
                        pb_g=pb_g, is_g=is_g, lr_g_su=lr_g[0] if lr_g else np.empty(0, np.float32),
                        lr_g_sv=lr_g[1] if lr_g else np.empty(0, np.float32),
                        lr_g_ui=lr_g[2] if lr_g else np.empty(0, np.int8),
                        lr_g_vi=lr_g[3] if lr_g else np.empty(0, np.int8),
                        pb_u=pb_u, is_u=is_u, lr_u_su=lr_u[0] if lr_u else np.empty(0, np.float32),
                        lr_u_sv=lr_u[1] if lr_u else np.empty(0, np.float32),
                        lr_u_ui=lr_u[2] if lr_u else np.empty(0, np.int8),
                        lr_u_vi=lr_u[3] if lr_u else np.empty(0, np.int8),
                        has_d=np.array([1], dtype=np.int32),
                        pb_d=pb_d, is_d=is_d, lr_d_su=lr_d[0] if lr_d else np.empty(0, np.float32),
                        lr_d_sv=lr_d[1] if lr_d else np.empty(0, np.float32),
                        lr_d_ui=lr_d[2] if lr_d else np.empty(0, np.int8),
                        lr_d_vi=lr_d[3] if lr_d else np.empty(0, np.int8),
                    )
                    del pb_d, is_d, lr_d
                else:
                    # Only g+u upgraded (rare); preserve existing down_proj from src_t16c unpermuted
                    assert perm_pi is None
                    d_cal = eval_src_mat(fp_src, d_k, act_s_cal)
                    d_t2 = eval_src_mat(fp_src, d_k, act_s_t2)
                    d_tf = eval_src_mat(fp_src, d_k, act_s_tf)
                    wd_t_orig = sf.read_qat_rows_f32(f"{pfx}.mlp.down_proj", 0, d_mod)
                    mode_tag = "2b_gauss_gu"
                    layer_upg_stats = {"g": st_g, "u": st_u}
                    np.savez(
                        work_layers_dir / f"L{l_idx:02d}.npz",
                        pb_g=pb_g, is_g=is_g, lr_g_su=lr_g[0] if lr_g else np.empty(0, np.float32),
                        lr_g_sv=lr_g[1] if lr_g else np.empty(0, np.float32),
                        lr_g_ui=lr_g[2] if lr_g else np.empty(0, np.int8),
                        lr_g_vi=lr_g[3] if lr_g else np.empty(0, np.int8),
                        pb_u=pb_u, is_u=is_u, lr_u_su=lr_u[0] if lr_u else np.empty(0, np.float32),
                        lr_u_sv=lr_u[1] if lr_u else np.empty(0, np.float32),
                        lr_u_ui=lr_u[2] if lr_u else np.empty(0, np.int8),
                        lr_u_vi=lr_u[3] if lr_u else np.empty(0, np.int8),
                        has_d=np.array([0], dtype=np.int32),
                    )
                del pb_g, is_g, lr_g, pb_u, is_u, lr_u, act_t_perm
            else:
                g_cal = eval_src_mat(fp_src, g_k, hm_cal)
                u_cal = eval_src_mat(fp_src, u_k, hm_cal)
                g_t2 = eval_src_mat(fp_src, g_k, hm_t2)
                u_t2 = eval_src_mat(fp_src, u_k, hm_t2)
                g_tf = eval_src_mat(fp_src, g_k, hm_tf)
                u_tf = eval_src_mat(fp_src, u_k, hm_tf)
                act_s_cal = c3.gelu(g_cal) * u_cal
                act_s_t2 = c3.gelu(g_t2) * u_t2
                act_s_tf = c3.gelu(g_tf) * u_tf
                if l_idx in upgrade_down:
                    wd_t = sf.read_qat_rows_f32(f"{pfx}.mlp.down_proj", 0, d_mod)
                    pb_d, is_d, lr_d, st_d = quantize_2b_gauss_gptq(
                        wd_t, act_s_cal, act_s_t2, True, args.rk, args.hblk, mem_eval, libv2, ops, prompt_starts=pstarts
                    )
                    wd_t_orig = wd_t
                    d_cal = mem_eval.linear(act_s_cal, d_mod, ff_dim, BSIZE_DEFAULT, NB_2B_GAUSS_CB, False, True, pb_d, is_d, lr_d)
                    d_t2 = mem_eval.linear(act_s_t2, d_mod, ff_dim, BSIZE_DEFAULT, NB_2B_GAUSS_CB, False, True, pb_d, is_d, lr_d)
                    d_tf = mem_eval.linear(act_s_tf, d_mod, ff_dim, BSIZE_DEFAULT, NB_2B_GAUSS_CB, False, True, pb_d, is_d, lr_d)
                    mode_tag = "2b_gauss_dn"
                    layer_upg_stats = {"d": st_d}
                    np.savez(
                        work_layers_dir / f"L{l_idx:02d}.npz",
                        has_g=np.array([0], dtype=np.int32),
                        has_d=np.array([1], dtype=np.int32),
                        pb_d=pb_d, is_d=is_d, lr_d_su=lr_d[0] if lr_d else np.empty(0, np.float32),
                        lr_d_sv=lr_d[1] if lr_d else np.empty(0, np.float32),
                        lr_d_ui=lr_d[2] if lr_d else np.empty(0, np.int8),
                        lr_d_vi=lr_d[3] if lr_d else np.empty(0, np.int8),
                    )
                    del pb_d, is_d, lr_d
                else:
                    d_cal = eval_src_mat(fp_src, d_k, act_s_cal)
                    d_t2 = eval_src_mat(fp_src, d_k, act_s_t2)
                    d_tf = eval_src_mat(fp_src, d_k, act_s_tf)
                    wd_t_orig = None

            # 1c. Train 84-Block MLP Cortical Router (CRTR-V9.2 r=64 + MoD) on Student activations
            n_blks_mlp = ff_dim // BSIZE_DEFAULT
            blk_e_cal = np.sum(act_s_cal.reshape(len(act_s_cal), n_blks_mlp, BSIZE_DEFAULT) ** 2, axis=-1)
            blk_e_va = np.sum(act_s_t2.reshape(len(act_s_t2), n_blks_mlp, BSIZE_DEFAULT) ** 2, axis=-1)
            mod_mlp_tr = np.linalg.norm(d_cal, axis=-1)
            mod_mlp_va = np.linalg.norm(d_t2, axis=-1)
            thr_m = float(np.quantile(mod_mlp_cal := mod_mlp_tr, 0.25))
            mr_net, mr_met = crtr.train_layer_router(
                torch.from_numpy(hm_cal),
                torch.from_numpy(blk_e_cal),
                torch.from_numpy(hm_t2),
                torch.from_numpy(blk_e_va),
                top_p=0.50,
                rank=args.crtr_rank,
                steps=args.crtr_steps,
                lr=3e-3,
                mod_targets_tr=torch.from_numpy((mod_mlp_cal > thr_m).astype(np.float32)),
                mod_targets_va=torch.from_numpy((mod_mlp_va > thr_m).astype(np.float32)),
                device="cpu",
                seed=100 + l_idx,
            )
            crtr_layers.append(crtr.quantize_layer_for_crtr(mr_net, block_size=BSIZE_DEFAULT))

            # 1d. Wiener Post-Feedforward Norm + LayerScalar Calibration (strict held-out MSE rollback)
            z_pff_tr = c3.rmsn(d_cal, None, eps)
            z_pff_va = c3.rmsn(d_t2, None, eps)
            inv_lsc = 1.0 / (lsc if abs(lsc) > 1e-4 else 1.0)
            dy_pff_tr = (xt_teach_next_cal * inv_lsc) - xm_cal
            dy_pff_va = (xt_teach_next_t2 * inv_lsc) - xm_t2
            g_pff_new, pff_mse0, pff_mse1 = solve_wiener_post_norm_np(z_pff_tr, dy_pff_tr, z_pff_va, dy_pff_va, g_pff)
            norm_dict[f"{pfx}.post_feedforward_layernorm.weight"] = g_pff_new

            pred_tr = xm_cal + z_pff_tr * g_pff_new[None, :]
            pred_va = xm_t2 + z_pff_va * g_pff_new[None, :]
            lsc_new, lsc_mse0, lsc_mse1 = solve_wiener_layer_scalar_np(pred_tr, xt_teach_next_cal, pred_va, xt_teach_next_t2, lsc)
            norm_dict[f"{pfx}.layer_scalar"] = np.array([lsc_new], dtype=np.float32)

            # 1e. Optional Accept-Only C8DB pattern bank on target down_proj / attention bottleneck
            c8_acc = False
            if (l_idx in c8_target_layers) and (len(c8_entries) < args.max_c8d_entries):
                if wd_t_orig is None:
                    wd_t_orig = sf.read_qat_rows_f32(f"{pfx}.mlp.down_proj", 0, d_mod)
                yt_d_cal = ops.mm_xt(act_s_cal, wd_t_orig)
                yt_d_va = ops.mm_xt(act_s_t2, wd_t_orig)
                entry_d, st_c8 = c8d.train_projection_pattern_bank(
                    d_k,
                    act_s_cal,
                    d_cal,
                    yt_d_cal,
                    act_s_t2,
                    d_t2,
                    yt_d_va,
                    ["general"] * len(act_s_cal),
                    rank=args.c8d_rank,
                    n_experts=8,
                    top_k=2,
                    steps=args.c8d_steps,
                    lr=3e-3,
                )
                if entry_d is not None:
                    c8_entries[d_k] = entry_d
                    c8_acc = True
                    # Evaluate INT8 quantized C8DB improvement and add delta onto student residual!
                    d_cal = d_cal + c8d.eval_quantized_adapter_np(
                        act_s_cal, entry_d["centroids"], entry_d["experts"], entry_d["top_k"], entry_d["temperature"], entry_d["gate_scale"]
                    )
                    d_t2 = d_t2 + c8d.eval_quantized_adapter_np(
                        act_s_t2, entry_d["centroids"], entry_d["experts"], entry_d["top_k"], entry_d["temperature"], entry_d["gate_scale"]
                    )
                    d_tf = d_tf + c8d.eval_quantized_adapter_np(
                        act_s_tf, entry_d["centroids"], entry_d["experts"], entry_d["top_k"], entry_d["temperature"], entry_d["gate_scale"]
                    )

            del wd_t_orig, hm_cal, hm_t2, hm_tf, g_cal, u_cal, g_t2, u_t2, g_tf, u_tf, act_s_cal, act_s_t2, act_s_tf

            # 1f. Advance cascaded student states cleanly
            xc_c = (xm_cal + c3.rmsn(d_cal, g_pff_new, eps)) * lsc_new
            xt_c = (xm_t2 + c3.rmsn(d_t2, g_pff_new, eps)) * lsc_new
            xf_c = (xm_tf + c3.rmsn(d_tf, g_pff_new, eps)) * lsc_new

            cos_t2 = c3.cos_sim(xt_c, xt_teach_next_t2)
            cos_tf = c3.cos_sim(xf_c, xt_teach_next_tf)
            nr_t2 = float(np.linalg.norm(xt_c) / max(np.linalg.norm(xt_teach_next_t2), 1e-12))
            nr_tf = float(np.linalg.norm(xf_c) / max(np.linalg.norm(xt_teach_next_tf), 1e-12))
            dt_l = time.perf_counter() - tl0
            print(
                f"L{l_idx:02d} [{mode_tag:14s}] cos_t2={cos_t2:.5f} nr_t2={nr_t2:.4f} | "
                f"cos_tf={cos_tf:.5f} nr_tf={nr_tf:.4f} | lsc={lsc:.3f}->{lsc_new:.3f} c8d={int(c8_acc)} "
                f"crtr_rec={mr_met['val_recall_after']:.3f} hr_rec={hr_met['val_recall_after']:.3f} [{dt_l:.1f}s]",
                flush=True,
            )
            layer_profiles.append(
                {
                    "layer": l_idx,
                    "mode": mode_tag,
                    "cos_t2": round(cos_t2, 5),
                    "nr_t2": round(nr_t2, 4),
                    "cos_tf": round(cos_tf, 5),
                    "nr_tf": round(nr_tf, 4),
                    "wiener_pff": [round(pff_mse0, 5), round(pff_mse1, 5)],
                    "lsc": [round(lsc, 4), round(lsc_new, 4)],
                    "crtr_recall": round(float(mr_met["val_recall_after"]), 4),
                    "head_crtr_recall": round(float(hr_met["val_recall_after"]), 4),
                    "c8d_accepted": c8_acc,
                    "upg_stats": layer_upg_stats,
                }
            )
            del xm_cal, xm_t2, xm_tf, d_cal, d_t2, d_tf
            gc.collect()

        # Phase 2: Wiener Final Norm + Post-Cell60 C8DB Bridge on Final Hidden States
        fn_key = "model.language_model.norm.weight"
        g_fn = norm_dict[fn_key].copy()
        h_unit_cal = c3.rmsn(xc_c, None, eps)
        h_unit_t2 = c3.rmsn(xt_c, None, eps)
        h_unit_tf = c3.rmsn(xf_c, None, eps)
        t_fn_cal = c3.rmsn(xcal_last, g_fn, eps)
        t_fn_t2 = c3.rmsn(xt2_t[n_layers], g_fn, eps)
        t_fn_tf = c3.rmsn(xtf_t[n_layers], g_fn, eps)

        g_fn_new, fn_m0, fn_m1 = solve_wiener_post_norm_np(h_unit_cal, t_fn_cal, h_unit_t2, t_fn_t2, g_fn, max_rel_delta=0.45)
        norm_dict[fn_key] = g_fn_new
        c8_norms[fn_key] = g_fn_new

        s_fn_cal = h_unit_cal * g_fn_new[None, :]
        s_fn_t2 = h_unit_t2 * g_fn_new[None, :]
        s_fn_tf = h_unit_tf * g_fn_new[None, :]
        print(
            f"[FINAL_NORM] cos_t2={c3.cos_sim(s_fn_t2, t_fn_t2):.5f} nr_t2={np.linalg.norm(s_fn_t2)/max(np.linalg.norm(t_fn_t2),1e-12):.4f} "
            f"cos_tf={c3.cos_sim(s_fn_tf, t_fn_tf):.5f} mse={fn_m0:.5f}->{fn_m1:.5f}",
            flush=True,
        )

        # Train `post_cell60` C8DB pattern bank (rank=32) bridging student final normalized hidden -> teacher final hidden
        pc60_key = "model.language_model.norm.post_cell60"
        entry_pc60, st_pc60 = c8d.train_projection_pattern_bank(
            pc60_key,
            s_fn_cal,
            s_fn_cal,
            t_fn_cal,
            s_fn_t2,
            s_fn_t2,
            t_fn_t2,
            ["general"] * len(s_fn_cal),
            rank=32,
            n_experts=8,
            top_k=2,
            steps=max(60, args.c8d_steps),
            lr=3e-3,
        )
        if entry_pc60 is not None:
            c8_entries[pc60_key] = entry_pc60
            p_t2 = s_fn_t2 + c8d.eval_quantized_adapter_np(
                s_fn_t2, entry_pc60["centroids"], entry_pc60["experts"], entry_pc60["top_k"], entry_pc60["temperature"], entry_pc60["gate_scale"]
            )
            p_tf = s_fn_tf + c8d.eval_quantized_adapter_np(
                s_fn_tf, entry_pc60["centroids"], entry_pc60["experts"], entry_pc60["top_k"], entry_pc60["temperature"], entry_pc60["gate_scale"]
            )
            print(
                f"[POST_CELL60 C8DB] ACCEPTED cos_t2={c3.cos_sim(p_t2, t_fn_t2):.5f} "
                f"cos_tf={c3.cos_sim(p_tf, t_fn_tf):.5f} stats={st_pc60}",
                flush=True,
            )

        # Phase 3: Assemble V9.2 .t16c streaming cleanly from src_t16c + upgraded layers
        updated_norms = [(k, norm_dict[k]) for k, _ in norms_list]
        tmp_t16c = args.out_t16c.with_suffix(".t16c.tmp")
        with open(tmp_t16c, "wb") as fp_out:
            pos = write_t16c_header_and_norms(fp_out, hdr, updated_norms)
            loaded_cache: Dict[int, dict] = {}
            for mname in mat_names:
                m_meta = mat_map[mname]
                # Determine if matrix belongs to an upgraded MLP
                replaced = False
                for l_chk in upgrade_full | upgrade_gu | upgrade_down:
                    mpfx = f"model.language_model.layers.{l_chk}.mlp."
                    if mname.startswith(mpfx):
                        short = mname[len(mpfx) :]  # gate_proj.weight / up_proj.weight / down_proj.weight
                        proj_k = short.split(".")[0]
                        if (proj_k in ("gate_proj", "up_proj") and (l_chk in upgrade_full or l_chk in upgrade_gu)) or (
                            proj_k == "down_proj" and (l_chk in upgrade_full or l_chk in upgrade_down)
                        ):
                            if l_chk not in loaded_cache:
                                loaded_cache.clear()
                                loaded_cache[l_chk] = dict(np.load(work_layers_dir / f"L{l_chk:02d}.npz"))
                            zd = loaded_cache[l_chk]
                            tag_s = proj_k[0]
                            pb = zd[f"pb_{tag_s}"]
                            is_a = zd[f"is_{tag_s}"]
                            su_a = zd[f"lr_{tag_s}_su"]
                            lr_t = None
                            if su_a.size > 0:
                                lr_t = (
                                    su_a,
                                    zd[f"lr_{tag_s}_sv"],
                                    zd[f"lr_{tag_s}_ui"],
                                    zd[f"lr_{tag_s}_vi"],
                                )
                            nb_w = NB_2B_GAUSS_CB if proj_k == "down_proj" else NB_2B_GAUSS_TM
                            pos = p3._write_mat_record(
                                fp_out,
                                pos,
                                mname,
                                m_meta["d_out"],
                                m_meta["d_in"],
                                m_meta["bsize"],
                                nb_w,
                                False,
                                True,
                                pb,
                                is_a,
                                lr_t,
                            )
                            replaced = True
                        break
                if not replaced:
                    pb_r, is_r, lr_r = b91.read_mat_parts(fp_src, m_meta)
                    pos = p3._write_mat_record(
                        fp_out,
                        pos,
                        mname,
                        m_meta["d_out"],
                        m_meta["d_in"],
                        m_meta["bsize"],
                        m_meta["n_bits"],
                        m_meta["is_asym"],
                        m_meta["has_fwht"],
                        np.frombuffer(pb_r, dtype=np.uint8),
                        is_r,
                        lr_r,
                    )
        os.replace(tmp_t16c, args.out_t16c)

    # Phase 4: Export C8DB, CRTR, HeadRouter & Reports
    if c8_entries or c8_norms:
        tmp_c8 = args.out_c8d.with_suffix(".c8d.tmp")
        c8d.write_c8db_file(tmp_c8, list(c8_norms.items()), list(c8_entries.values()))
        os.replace(tmp_c8, args.out_c8d)

    tmp_cr = args.out_crtr.with_suffix(".crtr.tmp")
    crtr.export_crtr(crtr_layers, tmp_cr)
    os.replace(tmp_cr, args.out_crtr)

    tmp_hr = args.out_head_crtr.with_suffix(".crtr.tmp")
    crtr.export_crtr(head_crtr_layers, tmp_hr)
    os.replace(tmp_hr, args.out_head_crtr)

    # Compute hashes & reports
    out_bytes = args.out_t16c.stat().st_size
    c8_bytes = args.out_c8d.stat().st_size if args.out_c8d.is_file() else 0
    cr_bytes = args.out_crtr.stat().st_size
    hr_bytes = args.out_head_crtr.stat().st_size
    report = {
        "version": "V9.2-CliffFix-VLM",
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "upgrade_full_layers": sorted(upgrade_full),
        "upgrade_gu_layers": sorted(upgrade_gu),
        "upgrade_down_layers": sorted(upgrade_down),
        "hblk": int(args.hblk),
        "actals_rk": int(args.rk),
        "apply_geglu_perm": bool(args.apply_geglu_perm),
        "t16c_bytes": out_bytes,
        "t16c_gib": round(out_bytes / (1024**3), 4),
        "c8db_bytes": c8_bytes,
        "c8db_entries": list(c8_entries.keys()),
        "crtr_bytes": cr_bytes,
        "crtr_mean_recall": round(float(np.mean([p["crtr_recall"] for p in layer_profiles])), 5),
        "head_crtr_bytes": hr_bytes,
        "head_crtr_mean_recall": round(float(np.mean([p["head_crtr_recall"] for p in layer_profiles])), 5),
        "final_layer_cos_t2": layer_profiles[-1]["cos_t2"],
        "final_layer_nr_t2": layer_profiles[-1]["nr_t2"],
        "final_layer_cos_tf": layer_profiles[-1]["cos_tf"],
        "final_layer_nr_tf": layer_profiles[-1]["nr_tf"],
        "sha256": {
            args.out_t16c.name: sha256_file(args.out_t16c),
            args.out_c8d.name: sha256_file(args.out_c8d) if args.out_c8d.is_file() else None,
            args.out_crtr.name: sha256_file(args.out_crtr),
            args.out_head_crtr.name: sha256_file(args.out_head_crtr),
        },
        "layers": layer_profiles,
    }
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sums_lines = [f"{v}  {k}" for k, v in report["sha256"].items() if v is not None]
    (out_dir / "SHA256SUMS_V9_2.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    print(f"[V9.2 DONE] t16c={out_bytes:,}B ({report['t16c_gib']} GiB) crtr_rec={report['crtr_mean_recall']} final={report['final_layer_cos_t2']} elapsed={report['elapsed_s']}s", flush=True)
    return 0


def finalize_head_router_only(args: argparse.Namespace) -> int:
    """Replay already-built V9.2 .t16c/.c8d/.crtr to export uniform-block HeadRouter & sealed report."""
    t_start = time.perf_counter()
    for req in (args.out_t16c, args.out_crtr, args.model, args.config):
        if not req.is_file():
            raise FileNotFoundError(f"Missing required artifact for --finalize-head-router-only: {req}")

    out_dir = args.out_t16c.resolve().parent
    cfg = json.loads(args.config.resolve().read_text(encoding="utf-8"))["text_config"]
    eps = float(cfg["rms_norm_eps"])
    n_layers = int(cfg["num_hidden_layers"])
    nh = int(cfg.get("num_attention_heads", 32))
    nkv_s = int(cfg.get("num_key_value_heads", 16))
    nkv_f = int(cfg.get("num_global_key_value_heads", 4))
    hd_s = int(cfg.get("head_dim", 256))
    hd_f = int(cfg.get("global_head_dim", 512))

    _, cal_segs, _, xcal_t, xt2_t, xtf_t, xcal_last = _select_calib_and_val_batches(args.work_dir.resolve(), cfg, n_layers)
    so_path = args.so.resolve() if args.so else (THIS_DIR.parent.parent / "libneon_stream_accelerator.so")
    nlin_file = b91.NativeNeonLinear(so_path, threads=args.threads)
    if nlin_file.lib is not None:
        nlin_file.lib.neon_set_andvcnt_abits(6)

    _, norms_list, _ = read_t16c_norms_and_header(args.out_t16c)
    norm_dict = {k: v for k, v in norms_list}
    _, _, mat_map, _ = b91.parse_t16c_metadata(args.out_t16c)
    c8_entries = c8d.read_c8db_file(str(args.out_c8d))[1] if args.out_c8d.is_file() else {}
    prior_crtr_rec: Dict[int, float] = {}
    blog = out_dir / "build_v9_2.log"
    if blog.is_file():
        import re as _re
        for line in blog.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _re.match(r"L(\d+)\s+.*crtr_rec=([0-9.]+)", line)
            if m:
                prior_crtr_rec[int(m.group(1))] = float(m.group(2))

    xc_c, xt_c, xf_c = xcal_t[0].copy(), xt2_t[0].copy(), xtf_t[0].copy()
    head_crtr_layers: List[dict] = []
    layer_profiles: List[dict] = []

    with open(args.out_t16c, "rb") as fp_v92:
        def ev(key: str, x_in: np.ndarray) -> np.ndarray:
            return nlin_file.linear(fp_v92, mat_map[key], np.ascontiguousarray(x_in, dtype=np.float32))

        for l_idx in range(n_layers):
            tl0 = time.perf_counter()
            pfx = f"model.language_model.layers.{l_idx}"
            is_full = cfg["layer_types"][l_idx] == "full_attention"
            hd, nkv = (hd_f, nkv_f) if is_full else (hd_s, nkv_s)
            rot_d, theta = (128, 1e6) if is_full else (256, 1e4)
            g_in = norm_dict[f"{pfx}.input_layernorm.weight"]
            qn = norm_dict[f"{pfx}.self_attn.q_norm.weight"]
            kn = norm_dict[f"{pfx}.self_attn.k_norm.weight"]
            g_pa = norm_dict[f"{pfx}.post_attention_layernorm.weight"]
            g_pf = norm_dict[f"{pfx}.pre_feedforward_layernorm.weight"]
            g_pff = norm_dict[f"{pfx}.post_feedforward_layernorm.weight"]
            lsc = float(np.asarray(norm_dict[f"{pfx}.layer_scalar"]).reshape(-1)[0])

            inp_cal, inp_t2, inp_tf = c3.rmsn(xc_c, g_in, eps), c3.rmsn(xt_c, g_in, eps), c3.rmsn(xf_c, g_in, eps)
            yq_cal, yk_cal = ev(f"{pfx}.self_attn.q_proj.weight", inp_cal), ev(f"{pfx}.self_attn.k_proj.weight", inp_cal)
            yv_cal = yk_cal if is_full else ev(f"{pfx}.self_attn.v_proj.weight", inp_cal)
            yq_t2, yk_t2 = ev(f"{pfx}.self_attn.q_proj.weight", inp_t2), ev(f"{pfx}.self_attn.k_proj.weight", inp_t2)
            yv_t2 = yk_t2 if is_full else ev(f"{pfx}.self_attn.v_proj.weight", inp_t2)
            yq_tf, yk_tf = ev(f"{pfx}.self_attn.q_proj.weight", inp_tf), ev(f"{pfx}.self_attn.k_proj.weight", inp_tf)
            yv_tf = yk_tf if is_full else ev(f"{pfx}.self_attn.v_proj.weight", inp_tf)

            ctx_cal = np.concatenate(
                [_layer_attn_ctx(yq_cal[a:b], yk_cal[a:b], yv_cal[a:b], qn, kn, nh, nkv, hd, rot_d, theta, eps) for a, b in cal_segs],
                axis=0,
            )
            ctx_t2 = _layer_attn_ctx(yq_t2, yk_t2, yv_t2, qn, kn, nh, nkv, hd, rot_d, theta, eps)
            ctx_tf = _layer_attn_ctx(yq_tf, yk_tf, yv_tf, qn, kn, nh, nkv, hd, rot_d, theta, eps)

            head_e_cal = np.sqrt(np.mean(ctx_cal.reshape(len(ctx_cal), nh, hd) ** 2, axis=-1) + 1e-12)
            head_e_va = np.sqrt(np.mean(ctx_t2.reshape(len(ctx_t2), nh, hd) ** 2, axis=-1) + 1e-12)
            mod_sal_cal = np.linalg.norm(ctx_cal, axis=-1)
            mod_sal_va = np.linalg.norm(ctx_t2, axis=-1)
            thr_h = float(np.quantile(mod_sal_cal, 0.25))
            hr_net, hr_met = crtr.train_layer_router(
                torch.from_numpy(inp_cal),
                torch.from_numpy(head_e_cal),
                torch.from_numpy(inp_t2),
                torch.from_numpy(head_e_va),
                top_p=0.50,
                rank=args.head_crtr_rank,
                steps=args.crtr_steps,
                lr=3e-3,
                mod_targets_tr=torch.from_numpy((mod_sal_cal > thr_h).astype(np.float32)),
                mod_targets_va=torch.from_numpy((mod_sal_va > thr_h).astype(np.float32)),
                device="cpu",
                seed=200 + l_idx,
            )
            head_crtr_layers.append(crtr.quantize_layer_for_crtr(hr_net, block_size=hd_s))

            xm_cal = xc_c + c3.rmsn(ev(f"{pfx}.self_attn.o_proj.weight", ctx_cal), g_pa, eps)
            xm_t2 = xt_c + c3.rmsn(ev(f"{pfx}.self_attn.o_proj.weight", ctx_t2), g_pa, eps)
            xm_tf = xf_c + c3.rmsn(ev(f"{pfx}.self_attn.o_proj.weight", ctx_tf), g_pa, eps)
            hm_cal, hm_t2, hm_tf = c3.rmsn(xm_cal, g_pf, eps), c3.rmsn(xm_t2, g_pf, eps), c3.rmsn(xm_tf, g_pf, eps)

            g_k, u_k, d_k = f"{pfx}.mlp.gate_proj.weight", f"{pfx}.mlp.up_proj.weight", f"{pfx}.mlp.down_proj.weight"
            act_cal = c3.gelu(ev(g_k, hm_cal)) * ev(u_k, hm_cal)
            act_t2 = c3.gelu(ev(g_k, hm_t2)) * ev(u_k, hm_t2)
            act_tf = c3.gelu(ev(g_k, hm_tf)) * ev(u_k, hm_tf)
            d_cal, d_t2, d_tf = ev(d_k, act_cal), ev(d_k, act_t2), ev(d_k, act_tf)
            if d_k in c8_entries:
                ed = c8_entries[d_k]
                d_cal = d_cal + c8d.eval_quantized_adapter_np(act_cal, ed["centroids"], ed["experts"], ed["top_k"], ed["temperature"], ed["gate_scale"])
                d_t2 = d_t2 + c8d.eval_quantized_adapter_np(act_t2, ed["centroids"], ed["experts"], ed["top_k"], ed["temperature"], ed["gate_scale"])
                d_tf = d_tf + c8d.eval_quantized_adapter_np(act_tf, ed["centroids"], ed["experts"], ed["top_k"], ed["temperature"], ed["gate_scale"])

            xc_c = (xm_cal + c3.rmsn(d_cal, g_pff, eps)) * lsc
            xt_c = (xm_t2 + c3.rmsn(d_t2, g_pff, eps)) * lsc
            xf_c = (xm_tf + c3.rmsn(d_tf, g_pff, eps)) * lsc

            xt_teach_t2 = xt2_t[l_idx + 1]
            xt_teach_tf = xtf_t[l_idx + 1]
            cos_t2 = c3.cos_sim(xt_c, xt_teach_t2)
            nr_t2 = float(np.linalg.norm(xt_c) / max(np.linalg.norm(xt_teach_t2), 1e-12))
            cos_tf = c3.cos_sim(xf_c, xt_teach_tf)
            nr_tf = float(np.linalg.norm(xf_c) / max(np.linalg.norm(xt_teach_tf), 1e-12))
            print(
                f"L{l_idx:02d} [HR_REPLAY(0x{mat_map[g_k]['n_bits']:02x})] cos_t2={cos_t2:.5f} nr_t2={nr_t2:.4f} | "
                f"cos_tf={cos_tf:.5f} hr_rec={hr_met['val_recall_after']:.3f} [{time.perf_counter() - tl0:.1f}s]",
                flush=True,
            )
            layer_profiles.append({
                "layer": l_idx,
                "mode": f"0x{mat_map[g_k]['n_bits']:02x}",
                "cos_t2": round(cos_t2, 5),
                "nr_t2": round(nr_t2, 4),
                "cos_tf": round(cos_tf, 5),
                "nr_tf": round(nr_tf, 4),
                "crtr_recall": round(prior_crtr_rec.get(l_idx, 0.0), 4),
                "head_crtr_recall": round(float(hr_met["val_recall_after"]), 4),
                "c8d_accepted": d_k in c8_entries,
            })
            del inp_cal, inp_t2, inp_tf, yq_cal, yk_cal, yv_cal, yq_t2, yk_t2, yv_t2, yq_tf, yk_tf, yv_tf, ctx_cal, ctx_t2, ctx_tf
            del xm_cal, xm_t2, xm_tf, hm_cal, hm_t2, hm_tf, act_cal, act_t2, act_tf, d_cal, d_t2, d_tf
            gc.collect()

    tmp_hr = args.out_head_crtr.with_suffix(".crtr.tmp")
    crtr.export_crtr(head_crtr_layers, tmp_hr)
    os.replace(tmp_hr, args.out_head_crtr)

    out_bytes = args.out_t16c.stat().st_size
    c8_bytes = args.out_c8d.stat().st_size if args.out_c8d.is_file() else 0
    cr_bytes = args.out_crtr.stat().st_size
    hr_bytes = args.out_head_crtr.stat().st_size
    report = {
        "version": "V9.2-CliffFix-VLM",
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "upgrade_full_layers": sorted(set(parse_layer_csv(args.upgrade_layers))),
        "upgrade_gu_layers": sorted(set(parse_layer_csv(args.upgrade_gu_layers))),
        "upgrade_down_layers": sorted(set(parse_layer_csv(args.upgrade_down_layers))),
        "hblk": int(args.hblk),
        "actals_rk": int(args.rk),
        "apply_geglu_perm": bool(args.apply_geglu_perm),
        "t16c_bytes": out_bytes,
        "t16c_gib": round(out_bytes / (1024**3), 4),
        "c8db_bytes": c8_bytes,
        "c8db_entries": list(c8_entries.keys()),
        "crtr_bytes": cr_bytes,
        "crtr_mean_recall": round(float(np.mean([p["crtr_recall"] for p in layer_profiles])), 5),
        "head_crtr_bytes": hr_bytes,
        "head_crtr_mean_recall": round(float(np.mean([p["head_crtr_recall"] for p in layer_profiles])), 5),
        "final_layer_cos_t2": layer_profiles[-1]["cos_t2"],
        "final_layer_nr_t2": layer_profiles[-1]["nr_t2"],
        "final_layer_cos_tf": layer_profiles[-1]["cos_tf"],
        "final_layer_nr_tf": layer_profiles[-1]["nr_tf"],
        "sha256": {
            args.out_t16c.name: sha256_file(args.out_t16c),
            args.out_c8d.name: sha256_file(args.out_c8d) if args.out_c8d.is_file() else None,
            args.out_crtr.name: sha256_file(args.out_crtr),
            args.out_head_crtr.name: sha256_file(args.out_head_crtr),
        },
        "layers": layer_profiles,
    }
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sums_lines = [f"{v}  {k}" for k, v in report["sha256"].items() if v is not None]
    (out_dir / "SHA256SUMS_V9_2.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    print(f"[V9.2 FINALIZE DONE] t16c={out_bytes:,}B ({report['t16c_gib']} GiB) hr_rec={report['head_crtr_mean_recall']} final={report['final_layer_cos_t2']} elapsed={report['elapsed_s']}s", flush=True)
    return 0


def _self_test() -> int:
    """Mandatory 6-gate fail-closed verification battery."""
    failures = 0
    rng = np.random.default_rng(2026)

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Bijective GeGLU channel clustering increases top-block Pareto energy
    try:
        ff = 1024
        bsz = 256
        act = rng.normal(0.0, 0.1, size=(48, ff)).astype(np.float32)
        # Plant coherent high-energy clusters scattered every 4th channel
        for b in range(4):
            ch_idx = np.arange(b, ff, 4)
            sig = rng.normal(0.0, 2.0, size=(48, 1)).astype(np.float32)
            act[:, ch_idx] += sig
        # Make half of channel 0..ff//4 dominate energy
        act[:, ::8] *= 3.0
        perm, inv = compute_geglu_colblock_perm(act, shared_blocks=1, block_size=bsz, iters=6, seed=7)
        assert len(perm) == ff and len(set(perm.tolist())) == ff
        assert np.array_equal(perm[inv], np.arange(ff))
        raw_blk_e = np.sum(act.reshape(48, ff // bsz, bsz) ** 2, axis=-1).mean(axis=0)
        act_p = act[:, perm]
        perm_blk_e = np.sum(act_p.reshape(48, ff // bsz, bsz) ** 2, axis=-1).mean(axis=0)
        top1_raw = float(np.max(raw_blk_e) / np.sum(raw_blk_e))
        top1_perm = float(perm_blk_e[0] / np.sum(perm_blk_e))
        gate("Gate 1: Bijective pi_l GeGLU clustering", top1_perm > top1_raw * 1.15, f"top1={top1_raw:.3f}->{top1_perm:.3f}")
    except Exception as exc:
        gate("Gate 1: Bijective pi_l GeGLU clustering", False, str(exc))

    # Gate 2: 2b_gauss pack + COLBLOCK reorder + decode + AR scale calibration parity
    try:
        d_o, d_i, bsz = 64, 512, 256
        w_base = rng.normal(0.0, 0.03, size=(d_o, d_i)).astype(np.float32)
        u_w = rng.normal(0.0, 0.45, size=(d_o, 4)).astype(np.float32)
        v_w = rng.normal(0.0, 0.45, size=(4, d_i)).astype(np.float32)
        w = (w_base + u_w @ v_w).astype(np.float32)
        # Correlated activation manifold (authentic low effective rank + isotropic noise)
        basis = rng.normal(0.0, 1.0, size=(16, d_i)).astype(np.float32)
        x = (rng.normal(0.0, 1.0, size=(96, 16)) @ basis + rng.normal(0.0, 0.12, size=(96, d_i))).astype(np.float32)
        xv = (rng.normal(0.0, 1.0, size=(32, 16)) @ basis + rng.normal(0.0, 0.12, size=(32, d_i))).astype(np.float32)
        mem = NativeMemEval(None, threads=1, a_bits=6)
        ops_g = c3._NativeCalOps(None, require_native=False)
        pb_cb, inv_s, lr, stats = quantize_2b_gauss_gptq(w, x, xv, True, 4, 256, mem, None, ops_g, bsize=bsz)
        pb_tm = reorder_t16c_colblock(pb_cb, d_o, d_i, bsz, 2, False, False)
        w_cb = decode_pbytes_to_f32_np(pb_cb, d_o, d_i, bsz, NB_2B_GAUSS_CB, False, True, inv_s)
        w_tm = decode_pbytes_to_f32_np(pb_tm, d_o, d_i, bsz, NB_2B_GAUSS_TM, False, True, inv_s)
        y_q = mem.linear(xv, d_o, d_i, bsz, NB_2B_GAUSS_CB, False, True, pb_cb, inv_s, lr)
        y_t = xv @ w.T
        cos_q = c3.cos_sim(y_q, y_t)
        ok_cb = np.allclose(w_cb, w_tm, atol=1e-6)
        ok_lr = stats["cos_after_lr"] >= stats["cos_before_lr"] - 1e-4
        gate("Gate 2: 2b_gauss COLBLOCK + AR + ActALS parity", ok_cb and ok_lr and cos_q > 0.955 and np.all(np.isfinite(y_q)), f"cos={cos_q:.5f}")
    except Exception as exc:
        gate("Gate 2: 2b_gauss COLBLOCK + AR + ActALS parity", False, str(exc))

    # Gate 3: Wiener post-norm & layer_scalar gain improvement + held-out rollback
    try:
        z_tr = rng.normal(0.0, 1.0, size=(64, 32)).astype(np.float32)
        z_va = rng.normal(0.0, 1.0, size=(32, 32)).astype(np.float32)
        g0 = np.ones((32,), dtype=np.float32)
        dy_tr = z_tr * 1.25
        dy_va = z_va * 1.25
        g1, m0, m1 = solve_wiener_post_norm_np(z_tr, dy_tr, z_va, dy_va, g0)
        assert m1 < m0 * 0.5
        # Verify strict held-out rollback on adversarial shift
        g_adv, ma0, ma1 = solve_wiener_post_norm_np(z_tr, z_tr * 1.3, z_va, z_va * 0.7, g0)
        assert np.allclose(g_adv, g0) and ma1 == ma0
        lsc1, lm0, lm1 = solve_wiener_layer_scalar_np(z_tr, z_tr * 1.2, z_va, z_va * 1.2, 1.0)
        gate("Gate 3: Wiener norm + layer_scalar + rollback", lsc1 > 1.1 and lm1 < lm0, f"m={m0:.3f}->{m1:.4f} lsc={lsc1:.3f}")
    except Exception as exc:
        gate("Gate 3: Wiener norm + layer_scalar + rollback", False, str(exc))

    # Gate 4: T16C v3 header/norm rewrite + low-rank record roundtrip
    try:
        with tempfile.TemporaryDirectory() as td_s:
            tp = Path(td_s) / "mini.t16c"
            d_o, d_i, bsz = 32, 256, 256
            pb = np.zeros(((d_o + 15) // 16) * (d_i // bsz) * (32 + 2 * 32 * 16), dtype=np.uint8)
            inv_s = np.ones((d_i,), dtype=np.float32)
            su = np.full((d_o,), 0.02, dtype=np.float32)
            sv = np.full((4,), 0.5, dtype=np.float32)
            ui8 = np.ones((d_o, 4), dtype=np.int8)
            vi8 = np.ones((4, d_i), dtype=np.int8)
            norm_a = np.linspace(0.5, 1.5, d_i, dtype=np.float32)
            hdr_t = (T16C_MAGIC, 1, 3, 1, 1, 0)
            with open(tp, "wb") as f:
                pos = write_t16c_header_and_norms(f, hdr_t, [("model.norm.weight", norm_a)])
                p3._write_mat_record(f, pos, "layers.0.mlp.down_proj.weight", d_o, d_i, bsz, NB_2B_GAUSS_CB, False, True, pb, inv_s, (su, sv, ui8, vi8))
            h2, nl2, _ = read_t16c_norms_and_header(tp)
            nn2, _, mm2, mn2 = b91.parse_t16c_metadata(tp)
            with open(tp, "rb") as f:
                pb2, is2, lr2 = b91.read_mat_parts(f, mm2[mn2[0]])
            assert len(pb2) == len(pb) and np.allclose(is2, inv_s)
            assert lr2 is not None and lr2[0].shape == (d_o,) and lr2[2].shape == (d_o, 4)
            gate("Gate 4: T16C v3 norms & record roundtrip", np.allclose(nl2[0][1], norm_a), f"mats={len(mn2)}")
    except Exception as exc:
        gate("Gate 4: T16C v3 norms & record roundtrip", False, str(exc))

    # Gate 5: C8DB embed_vision preserve + accept-only post_cell60 roundtrip
    try:
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            d_o, d_i = 32, 64
            xtr = rng.normal(0.0, 0.5, size=(128, d_i)).astype(np.float32)
            xva = rng.normal(0.0, 0.5, size=(32, d_i)).astype(np.float32)
            wt = rng.normal(0.0, 0.4, size=(d_o, d_i)).astype(np.float32)
            u_t = rng.normal(0.0, 0.5, size=(d_o, 4)).astype(np.float32)
            v_t = rng.normal(0.0, 0.5, size=(4, d_i)).astype(np.float32)
            yb_tr, yt_tr = xtr @ wt.T, xtr @ wt.T + (xtr @ v_t.T) @ u_t.T
            yb_va, yt_va = xva @ wt.T, xva @ wt.T + (xva @ v_t.T) @ u_t.T
            ent, st = c8d.train_projection_pattern_bank("post_cell60", xtr, yb_tr, yt_tr, xva, yb_va, yt_va, ["g"] * 128, rank=8, n_experts=4, top_k=2, steps=40)
            c8p = td / "test.c8d"
            c8d.write_c8db_file(c8p, [("n", np.ones(d_o, np.float32))], [ent] if ent else [])
            if ent is not None:
                delta_c8 = c8d.eval_quantized_adapter_np(
                    xva, ent["centroids"], ent["experts"], ent["top_k"], ent["temperature"], ent["gate_scale"]
                )
                assert delta_c8.shape == yb_va.shape and np.all(np.isfinite(delta_c8))
            nr_r, en_r = c8d.read_c8db_file(str(c8p))
            preserved = _attach_c8db_radicals_from_file(c8p, en_r["post_cell60"])
            c8p2 = td / "test2.c8d"
            c8d.write_c8db_file(c8p2, list(nr_r.items()), [preserved])
            _, en_r2 = c8d.read_c8db_file(str(c8p2))
            gate("Gate 5: C8DB training & preserve roundtrip", ent is not None and "post_cell60" in en_r2 and preserved["expert_radicals"] == ent["expert_radicals"], f"mse={st.get('val_mse_after', 0):.5f}")
    except Exception as exc:
        gate("Gate 5: C8DB training & roundtrip", False, str(exc))

    # Gate 6: CRTR & HeadRouter export + load roundtrip cleanly
    try:
        with tempfile.TemporaryDirectory() as td_s:
            crp = Path(td_s) / "r.crtr"
            xr = torch.randn(96, 64)
            xv = torch.randn(32, 64)
            er = torch.rand(96, 8)
            ev = torch.rand(32, 8)
            mt = (xr[:, 0] > 0).float()
            mv = (xv[:, 0] > 0).float()
            rn, rmet = crtr.train_layer_router(xr, er, xv, ev, top_p=0.5, rank=8, steps=40, mod_targets_tr=mt, mod_targets_va=mv, seed=11)
            assert "val_recall_after" in rmet and np.isfinite(rmet["val_recall_after"])
            rq0 = crtr.quantize_layer_for_crtr(rn, block_size=16)
            rq1 = crtr.quantize_layer_for_crtr(rn, block_size=16)
            sz = crtr.export_crtr([rq0, rq1], crp)
            m_r, l_r = crtr.load_crtr(crp)
            gate("Gate 6: CRTR + HeadRouter export roundtrip", sz > 0 and m_r["n_layers"] == 2 and len(l_r) == 2, f"bytes={sz}")
    except Exception as exc:
        gate("Gate 6: CRTR + HeadRouter export roundtrip", False, str(exc))

    print(f"V9_2_BUILDER_SELFTEST: {6 - failures}/6", flush=True)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build V9.2-CliffFix-VLM (.t16c + .c8d + .crtr + head_router)")
    ap.add_argument("--selftest", action="store_true", help="Run mandatory 6-gate fail-closed verification")
    ap.add_argument("--src-t16c", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_1/gemma4_31b_v9_1_fused_vlm.t16c"))
    ap.add_argument("--model", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/model.safetensors"))
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--so", type=Path, default=None)
    ap.add_argument("--v43-c8d", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_1/gemma4_31b_v4_3_vlm_c8d.c8d"))
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b"))
    ap.add_argument("--out-t16c", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_cliff_fix_vlm.t16c"))
    ap.add_argument("--out-c8d", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_c8d.c8d"))
    ap.add_argument("--out-crtr", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_router.crtr"))
    ap.add_argument("--out-head-crtr", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_head_router.crtr"))
    ap.add_argument("--out-report", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/v9_2_build_report.json"))
    ap.add_argument("--upgrade-layers", type=str, default=",".join(str(x) for x in DEFAULT_UPGRADE_FULL))
    ap.add_argument("--upgrade-gu-layers", type=str, default="")
    ap.add_argument("--upgrade-down-layers", type=str, default=",".join(str(x) for x in DEFAULT_UPGRADE_DOWN))
    ap.add_argument("--c8d-layers", type=str, default="39,52,36,42,37,53,55,57")
    ap.add_argument("--hblk", type=int, default=256)
    ap.add_argument("--rk", type=int, default=16)
    ap.add_argument("--c8d-rank", type=int, default=16)
    ap.add_argument("--c8d-steps", type=int, default=45)
    ap.add_argument("--max-c8d-entries", type=int, default=10)
    ap.add_argument("--crtr-rank", type=int, default=64)
    ap.add_argument("--head-crtr-rank", type=int, default=32)
    ap.add_argument("--crtr-steps", type=int, default=160)
    ap.add_argument("--apply-geglu-perm", action="store_true", default=True)
    ap.add_argument("--no-geglu-perm", dest="apply_geglu_perm", action="store_false")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--finalize-head-router-only", action="store_true")
    args = ap.parse_args()

    if len(sys.argv) == 1 or args.selftest:
        return _self_test()
    if args.finalize_head_router_only:
        return finalize_head_router_only(args)
    return build_v9_2_cliff_fix_vlm(args)


if __name__ == "__main__":
    raise SystemExit(main())

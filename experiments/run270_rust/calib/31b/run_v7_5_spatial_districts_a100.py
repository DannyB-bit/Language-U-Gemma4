#!/usr/bin/env python3
"""
Gemma 4 31B V7.5 Full 60-Layer Spatial Semantic Districts + GTA LOD Residual Split
+ Live CRTR/MoD (q20) + 3-Bit Vision + Multimodal C8DB + Target-Aligned Assistant-V7 (BHDR-384).

Implements Method 1 (Width-Axis Conversion: Co-Activation City Districts + GTA LOD Residual Split)
across ALL 60 layers of Gemma 4 31B, architected to max out NVIDIA A100-SXM4-80GB compute/HBM2e:
  1. Zero 1-Bit Attention Anywhere (`L00..L59` Attention is >= 5-trit ternary / 3-bit, `0` 1-bit attn).
  2. `L00..L01` & `L56..L59` MLPs: High-precision pinned (`5-trit` / `3-bit`), aligned by `perm_district`.
  3. `L02..L55` Cold Middle MLPs (`54` layers = `162` matrices):
     - Co-activation Spatial District clustering (`Zone 0 Shared Core = 16 blocks = 4,096 neurons`
       + `M = 8 semantic city districts = 68 blocks = 17,408 neurons`, exact `block_size=256` multiples).
     - Optional District-Sparsity Distillation (`loss.backward()` + `AdamW`) when native Top-2
       energy capture falls below threshold, sharpening block-level locality without output drift.
     - GTA LOD Residual Split in shared SmoothQuant + FWHT rotated basis:
         W_3b ≈ W_base (1-bit Row-Major pinned in RAM)
              + ΔW_hi-res (2-bit Tile-16 Cholesky-GPTQ residual partitioned into
                1 pinned `lod_shared.t16c` + 8 SSD semantic shards `lod_d0..lod_d7.t16c`).
  4. Prompt-Level District Router (`fit_district_router`), Live `CRTR v1 + MoD (q20)`, 3-bit Vision,
     Multimodal `C8DB`, and Target-Aligned `Assistant-V7 (Q2/Q4/Q8 + BHDR-384)`.
  5. Fail-closed zero-hole `.partial`-first Drive sync across ALL base + LOD shards + 6-gate `--selftest`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_v6_1_skeptic_proof as spmod  # noqa: E402
import pack_assistant_t16c as pa  # noqa: E402
import pack_t16c_v3 as p3  # noqa: E402
import run_v7_0_fullrank_crtr_gpu as v70  # noqa: E402
import spatial_district_converter as sdc  # noqa: E402
import train_assistant_v7_byte_head as bhmod  # noqa: E402
import train_c8d_pattern_bank as c8d  # noqa: E402
import train_cortical_router as crtr  # noqa: E402
import v4_corpus as vc  # noqa: E402

T16C_GAUSS_BIT = 0x80
T16C_ROWMAJOR_BIT = 0x40
T16C_COLBLOCK_BIT = 0x20

FULL_LAYERS = v70.FULL_LAYERS
ATTN_3B_LAYERS = v70.ATTN_3B_LAYERS
ALL_BLACKLIST = v70.ALL_BLACKLIST

N_DISTRICTS = 12
BLOCK_SIZE = 256
# 12 / 84 blocks = 3,072 universal Shared Core neurons; 72 blocks = 12 districts x 6 blocks (1,536 neurons each).
SHARED_FRAC = 12.0 / 84.0
LOD_LAYERS: Set[int] = set(range(2, 58))  # L02..L57 (56 middle layers = 168 MLP matrices)
TOP_K_DISTRICTS = int(os.environ.get("V7_5_DISTRICT_TOP_K", "2"))

MAX_BASE_T16C_BYTES = 5_905_580_032        # 5.50 GiB hard base guard (incl. evictable Vision; text LM base ~5.255 GiB)
MAX_ACTIVE_TEXT_RAM_BYTES = 7_623_566_950  # 7.10 GiB <= 7.6 GiB Pi 4 RAM during text decode

OUT_DIR = Path(os.environ.get("V7_5_OUT_DIR", "/content/v7_5_outputs"))
_DRV_ENV = os.environ.get("V7_5_DRIVE_DIR", "").strip()
if not _DRV_ENV and Path("/content/drive/MyDrive").is_dir():
    _DRV_ENV = "/content/drive/MyDrive/gemma4_v7_5_spatial_districts"
DRIVE_DIR: Optional[Path] = Path(_DRV_ENV) if _DRV_ENV else None

CKPT_DIR = OUT_DIR / "ckpt_v7_5"
OUT_BASE_T16C = OUT_DIR / "gemma4_31b_v7_5_base.t16c"
OUT_LOD_SHARED = OUT_DIR / "gemma4_31b_v7_5_lod_shared.t16c"
OUT_LOD_DISTRICTS = [OUT_DIR / f"gemma4_31b_v7_5_lod_d{d}.t16c" for d in range(N_DISTRICTS)]
OUT_CRTR = OUT_DIR / "gemma4_31b_v7_5_router.crtr"
OUT_C8D = OUT_DIR / "gemma4_31b_v7_5_c8d.c8d"
OUT_DISTRICT_ROUTER = OUT_DIR / "gemma4_31b_v7_5_district_router.pt"
OUT_DISTRICT_MANIFEST = OUT_DIR / "gemma4_31b_v7_5_district_manifest.json"
OUT_BHDR = OUT_DIR / "gemma4_31b_v7_5_assistant.bhdr"
OUT_AST_Q2 = OUT_DIR / "gemma4_31b_v7_5_assistant_body_q2.t16c"
OUT_AST_Q4 = OUT_DIR / "gemma4_31b_v7_5_assistant_body_q4.t16c"
OUT_AST_Q8 = OUT_DIR / "gemma4_31b_v7_5_assistant_body_q8.t16c"
OUT_REPORT = OUT_DIR / "v7_5_report.json"
OUT_SHA256 = OUT_DIR / "SHA256SUMS.txt"
PROGRESS_JSON = CKPT_DIR / "progress.json"


# ---------------------------------------------------------------------------
# Hardware & A100 Max-Utilization Configuration
# ---------------------------------------------------------------------------
def configure_a100_max_utilization(
    gpu_vram_gb: float, has_cuda: bool, *, respect_env: bool = True
) -> Dict[str, object]:
    """Configure TF32, cudnn, HBM residency, micro-batch size, and async I/O pools across G4/A100/L4."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
    force_max = (os.environ.get("V7_5_A100_MAX_UTIL", "1").strip() == "1") if respect_env else True
    is_big_gpu = gpu_vram_gb >= 64.0
    resident_env = os.environ.get("V7_5_RESIDENT_GPU", "").strip() if respect_env else ""
    resident_gpu = (resident_env == "1") if resident_env else (is_big_gpu and force_max and has_cuda)

    if gpu_vram_gb >= 88.0:
        default_bs, default_ev, distill_bs, ast_bs, bh_bs = 256, 4096, 64, 16, 512
        profile = "G4-96GB-MAX"
    elif is_big_gpu:
        default_bs, default_ev, distill_bs, ast_bs, bh_bs = 192, 4096, 64, 16, 512
        profile = "A100-80GB-MAX"
    elif gpu_vram_gb >= 36.0:
        default_bs, default_ev, distill_bs, ast_bs, bh_bs = 128, 4096, 48, 12, 512
        profile = "HIGH-VRAM-GPU"
    elif has_cuda:
        default_bs, default_ev, distill_bs, ast_bs, bh_bs = 64, 4096, 24, 8, 512
        profile = "L4-22GB-MAX"
    else:
        default_bs, default_ev, distill_bs, ast_bs, bh_bs = 16, 512, 8, 4, 128
        profile = "CPU-SELFTEST"

    bs_override = os.environ.get("V7_5_BATCH_SEQ", "").strip() if respect_env else ""
    batch_seq = (
        (default_bs if (not is_big_gpu and int(bs_override) > default_bs) else max(int(bs_override), default_bs))
        if force_max
        else int(bs_override)
    ) if bs_override else default_bs
    ev_override = os.environ.get("V7_5_ALS_EV", "").strip() if respect_env else ""
    als_ev_toks = (max(int(ev_override), default_ev) if force_max else int(ev_override)) if ev_override else default_ev

    if has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    return {
        "profile": profile,
        "gpu_vram_gb": round(float(gpu_vram_gb), 2),
        "resident_gpu": bool(resident_gpu),
        "batch_seq": max(1, batch_seq),
        "als_ev_toks": max(64, als_ev_toks),
        "distill_bs": distill_bs,
        "ast_bs": ast_bs,
        "bh_bs": bh_bs,
        "tf32": bool(has_cuda),
        "distill_enabled": os.environ.get("V7_5_DISTRICT_DISTILL", "1").strip() == "1",
        "distill_thresh": float(os.environ.get("V7_5_DISTILL_THRESH", "0.72")),
        "distill_steps": int(os.environ.get("V7_5_DISTILL_STEPS", "40")),
    }


def _gpu_vram_stats_gb(device: torch.device) -> Tuple[float, float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    alloc = torch.cuda.memory_allocated(device) / (1024**3)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    tot = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    return round(alloc, 2), round(peak, 2), round(tot, 2)


def symmetrize_matrix_inplace(h: torch.Tensor, tile: int = 1024) -> torch.Tensor:
    """In-place tiled matrix symmetrization without allocating a second `[N, N]` tensor."""
    n = h.shape[0]
    for i in range(0, n, tile):
        i2 = min(i + tile, n)
        blk = h[i:i2, i:i2]
        blk.add_(blk.T.clone()).mul_(0.5)
        for j in range(i + tile, n, tile):
            j2 = min(j + tile, n)
            b_ij, b_ji = h[i:i2, j:j2], h[j:j2, i:i2]
            avg = (b_ij + b_ji.T) * 0.5
            b_ij.copy_(avg)
            b_ji.copy_(avg.T)
    return h


# ---------------------------------------------------------------------------
# V7.5 Spatial Districts + GTA LOD Architecture & Exact Byte Budgets
# ---------------------------------------------------------------------------
def get_lm_spec_v7_5(l_idx: int, pname: str) -> Tuple[int, int, int, bool]:
    """
    Return `(mode_i, bsize=256, lr_rank, to_rowmajor)` for Base (`gemma4_31b_v7_5_base.t16c`):
      - `L00..L59` Attention: Identical high-precision schedule as V7.0 (ZERO 1-bit Attention!).
      - `L00..L01` & `L56..L57` MLPs: 5-trit ternary (`mode=1`, `down r=16`, `g/u r=8`).
      - `L58..L59` MLPs: 3-bit (`mode=4`, `r=16`).
      - `L02..L55` MLPs (`LOD_LAYERS`): `W_base(1b_asym)` Row-Major (`mode=0, lr_rank=0, rm=True`),
        backed by 2-bit symmetric rotated residual shards `ΔW_hi-res` (`Shared + Districts d0..d7`).
    """
    is_attn = pname in ("q_proj", "k_proj", "v_proj", "o_proj")
    if l_idx in (58, 59):
        rk = 24 if pname in ("k_proj", "v_proj") else 16
        return (4, 256, rk, False)
    if l_idx in ATTN_3B_LAYERS and pname in ("q_proj", "o_proj"):
        return (4, 256, 16, False)
    if is_attn:
        rk = 16 if (l_idx in (0, 1, 56, 57) or l_idx in FULL_LAYERS) else 0
        return (1, 256, rk, False)
    if l_idx in (0, 1):
        return (1, 256, 16 if pname == "down_proj" else 8, False)
    # L02..L57 middle MLPs: pinned 1-bit Row-Major base + 2-bit GTA LOD residual patches
    return (0, 256, 0, True)


def compute_v7_5_budget_bytes(
    vocab_size: int = 262144,
    d_mod: int = 5376,
    n_layers: int = 60,
    nh: int = 32,
    nkv_s: int = 16,
    nkv_f: int = 4,
    hd_s: int = 256,
    hd_f: int = 512,
    ff: int = 21504,
    n_districts: int = N_DISTRICTS,
    top_k_districts: int = TOP_K_DISTRICTS,
) -> Dict[str, int]:
    """Compute exact predicted bytes across Base, Zone 0 Shared Core, District shards & Pi 4 RAM."""
    emb_b = p3.expected_tile16_bytes(vocab_size, d_mod, 256, 3, False) + 80
    lm_body_b = 0
    lm_mats = 1
    attn_1b_cnt = 0
    for l_idx in range(n_layers):
        is_f = l_idx in FULL_LAYERS
        hd = hd_f if is_f else hd_s
        nkv = nkv_f if is_f else nkv_s
        sh = {
            "q_proj": (nh * hd, d_mod),
            "k_proj": (nkv * hd, d_mod),
            "o_proj": (d_mod, nh * hd),
            "gate_proj": (ff, d_mod),
            "up_proj": (ff, d_mod),
            "down_proj": (d_mod, ff),
        }
        if not is_f:
            sh["v_proj"] = (nkv * hd, d_mod)
        for pn, (do, di) in sh.items():
            m_i, bsz, rk, to_rm = get_lm_spec_v7_5(l_idx, pn)
            nb, asy = v70.spec_to_t16c_bits(m_i, to_rm)
            if pn in ("q_proj", "k_proj", "v_proj", "o_proj") and (nb & 0x1F) == 1:
                attn_1b_cnt += 1
            pb = p3.expected_tile16_bytes(do, di, bsz, nb, asy)
            lr_b = (do * 4 + rk * 4 + do * rk + rk * di) if rk > 0 else 0
            lm_body_b += pb + di * 4 + lr_b + 80
            lm_mats += 1

    vis_b = (
        2 * (p3.expected_tile16_bytes(10240, 1152, 64, 3, False) + 80)
        + p3.expected_tile16_bytes(768, 1152, 64, 3, False) + 80
        + p3.expected_tile16_bytes(d_mod, 1152, 64, 3, False) + 80
        + 27 * (
            4 * (p3.expected_tile16_bytes(1152, 1152, 64, 3, False) + 80)
            + 2 * (p3.expected_tile16_bytes(4304, 1152, 64, 3, False) + 80)
            + (p3.expected_tile16_bytes(1152, 4352, 64, 3, False) + 80)
        )
    )

    n_blks = ff // BLOCK_SIZE
    n_shared_blks = max(0, min(int(round(SHARED_FRAC * n_blks)), n_blks - n_districts))
    rem_blks = n_blks - n_shared_blks
    max_d_blks = (rem_blks + n_districts - 1) // n_districts

    def _lod_shard_bytes(k_blks: int) -> int:
        k_neurons = k_blks * BLOCK_SIZE
        # gate + up: [k_neurons, d_mod] + inv_s; down: [d_mod, k_neurons] + inv_s[k_neurons]
        per_layer = (
            2 * (p3.expected_tile16_bytes(k_neurons, d_mod, 256, 2, False) + d_mod * 4 + 80)
            + (p3.expected_tile16_bytes(d_mod, k_neurons, 256, 2, False) + k_neurons * 4 + 80)
        )
        return 24 + len(LOD_LAYERS) * per_layer

    lod_shared_b = _lod_shard_bytes(n_shared_blks)
    lod_all_districts_b = _lod_shard_bytes(rem_blks)
    lod_one_district_max_b = _lod_shard_bytes(max_d_blks)
    lod_topk_arena_b = _lod_shard_bytes(min(rem_blks, top_k_districts * max_d_blks))
    base_lm_b = emb_b + lm_body_b + 6_500_000
    base_tot_b = base_lm_b + vis_b

    # Active text-decode Pi 4 RAM (Vision evicted via MADV_DONTNEED):
    # Base LM + Shared Core + Top-K active district arena + GlyphKV-LOD + C8DB/CRTR/Assistant-Q2 (~180 MiB)
    aux_runtime_b = 180 * 1024 * 1024
    active_text_ram_b = base_lm_b + lod_shared_b + lod_topk_arena_b + aux_runtime_b
    total_ssd_backed_b = base_tot_b + lod_shared_b + lod_all_districts_b

    return {
        "pred_lm_mats": lm_mats,
        "pred_lod_mats_per_shard": len(LOD_LAYERS) * 3,
        "attn_1b_cnt": attn_1b_cnt,
        "base_lm_bytes": base_lm_b,
        "base_vis_bytes": vis_b,
        "base_total_bytes": base_tot_b,
        "lod_shared_bytes": lod_shared_b,
        "lod_district_max_bytes": lod_one_district_max_b,
        "lod_all_districts_bytes": lod_all_districts_b,
        "lod_topk_arena_bytes": lod_topk_arena_b,
        "active_text_ram_bytes": active_text_ram_b,
        "total_ssd_backed_bytes": total_ssd_backed_b,
        "n_shared_blks": n_shared_blks,
        "rem_blks": rem_blks,
    }


# ---------------------------------------------------------------------------
# Pure-NumPy Exact Tile-16 Packers & Verifier (for CPU Self-Tests & Cross-Check)
# ---------------------------------------------------------------------------
def pack_tile16_asym1_np(w_rot_f32: np.ndarray, bsize: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Pack `[d_out, d_in]` Column-Major 1-bit asymmetric Tile-16 matching `fast_quant.c`."""
    w = np.ascontiguousarray(w_rot_f32, dtype=np.float32)
    d_out, d_in = w.shape
    n_tiles, n_blks, bpr = (d_out + 15) // 16, d_in // bsize, bsize // 8
    tblk_b = 64 + 16 * bpr
    w_pad = np.zeros((n_tiles * 16, n_blks, bsize), dtype=np.float32)
    w_pad[:d_out] = w.reshape(d_out, n_blks, bsize)
    wb = w_pad.reshape(n_tiles, 16, n_blks, bsize).transpose(0, 2, 1, 3)

    bias = wb.mean(axis=3)
    for _ in range(2):
        m1 = wb >= bias[..., None]
        c1 = m1.sum(axis=3)
        c0 = bsize - c1
        mp = np.where(c1 > 0, np.where(m1, wb, 0.0).sum(axis=3) / np.maximum(c1, 1), bias)
        mn = np.where(c0 > 0, np.where(m1, 0.0, wb).sum(axis=3) / np.maximum(c0, 1), bias)
        bias = 0.5 * (mp + mn)
        sc = 0.5 * (mp - mn)

    sc16 = sc.astype(np.float16)
    bi16 = bias.astype(np.float16)
    scf = sc16.astype(np.float32)
    bif = bi16.astype(np.float32)
    m_fin = wb >= bif[..., None]
    wq = np.where(m_fin, bif[..., None] + scf[..., None], bif[..., None] - scf[..., None])

    pb = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
    pb[:, :, 0:32] = np.ascontiguousarray(sc16, dtype="<f2").view(np.uint8).reshape(n_tiles, n_blks, 32)
    pb[:, :, 32:64] = np.ascontiguousarray(bi16, dtype="<f2").view(np.uint8).reshape(n_tiles, n_blks, 32)
    bits_u8 = np.packbits(m_fin.astype(np.uint8), axis=3, bitorder="big")  # [n_tiles, n_blks, 16, bpr]
    pb[:, :, 64:] = np.ascontiguousarray(bits_u8.transpose(0, 1, 3, 2)).reshape(n_tiles, n_blks, 16 * bpr)
    w_q_out = wq.transpose(0, 2, 1, 3).reshape(n_tiles * 16, d_in)[:d_out, :].copy()
    return pb.reshape(-1), w_q_out


def pack_tile16_sym2_np(w_rot_f32: np.ndarray, bsize: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Pack `[d_out, d_in]` Column-Major 2-bit symmetric Tile-16 matching `fast_quant.c` (`mode=2`)."""
    w = np.ascontiguousarray(w_rot_f32, dtype=np.float32)
    d_out, d_in = w.shape
    n_tiles, n_blks, bpr = (d_out + 15) // 16, d_in // bsize, bsize // 8
    plane_b = bpr * 16
    tblk_b = 32 + 2 * plane_b
    w_pad = np.zeros((n_tiles * 16, n_blks, bsize), dtype=np.float32)
    w_pad[:d_out] = w.reshape(d_out, n_blks, bsize)
    wb = w_pad.reshape(n_tiles, 16, n_blks, bsize).transpose(0, 2, 1, 3)

    sc = 0.498 * np.sqrt(np.mean(wb * wb, axis=3))
    max_lvl = 3.0
    for _ in range(4):
        inv_s = np.where(sc > 1e-12, 1.0 / np.maximum(sc, 1e-12), 0.0)
        q = np.clip(np.round((wb * inv_s[..., None] + max_lvl) * 0.5), 0.0, max_lvl)
        lv = 2.0 * q - max_lvl
        den = (lv * lv).sum(axis=3)
        sc = np.where(den > 0.0, (lv * wb).sum(axis=3) / np.maximum(den, 1e-12), 0.0)

    sc16 = sc.astype(np.float16)
    scf = sc16.astype(np.float32)
    inv_s = np.where(scf > 1e-12, 1.0 / np.maximum(scf, 1e-12), 0.0)
    qi = np.clip(np.round((wb * inv_s[..., None] + max_lvl) * 0.5), 0.0, max_lvl).astype(np.uint8)
    wq = scf[..., None] * (2.0 * qi.astype(np.float32) - max_lvl)

    pb = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
    pb[:, :, 0:32] = np.ascontiguousarray(sc16, dtype="<f2").view(np.uint8).reshape(n_tiles, n_blks, 32)
    for p in range(2):
        plane_bits = np.packbits((qi >> p) & 1, axis=3, bitorder="big")
        pb[:, :, 32 + p * plane_b : 32 + (p + 1) * plane_b] = np.ascontiguousarray(plane_bits.transpose(0, 1, 3, 2)).reshape(n_tiles, n_blks, plane_b)
    w_q_out = wq.transpose(0, 2, 1, 3).reshape(n_tiles * 16, d_in)[:d_out, :].copy()
    return pb.reshape(-1), w_q_out


def decode_tile16_sym2_np(pb: np.ndarray, d_out: int, d_in: int, bsize: int = 256, has_fwht: bool = True) -> np.ndarray:
    """Reference CPU decoder for 2-bit symmetric Column-Major Tile-16 (`mode=2`)."""
    n_tiles, n_blks, bpr = (d_out + 15) // 16, d_in // bsize, bsize // 8
    plane_b = bpr * 16
    tblk_b = 32 + 2 * plane_b
    arr = pb.reshape(n_tiles, n_blks, tblk_b)
    sc = arr[:, :, 0:32].copy().view("<f2").reshape(n_tiles, n_blks, 16).astype(np.float32)
    p0 = np.unpackbits(arr[:, :, 32 : 32 + plane_b].reshape(n_tiles, n_blks, bpr, 16).transpose(0, 1, 3, 2), axis=-1, bitorder="big")
    p1 = np.unpackbits(arr[:, :, 32 + plane_b :].reshape(n_tiles, n_blks, bpr, 16).transpose(0, 1, 3, 2), axis=-1, bitorder="big")
    qi = p0.astype(np.float32) + 2.0 * p1.astype(np.float32)
    wq = sc[..., None] * (2.0 * qi - 3.0)
    w_rot = wq.transpose(0, 2, 1, 3).reshape(n_tiles * 16, d_in)[:d_out, :].copy()
    if has_fwht:
        h = pa.make_sylvester_hadamard(bsize)
        w_rot = (w_rot.reshape(d_out, n_blks, bsize) @ h).reshape(d_out, d_in)
    return w_rot


# ---------------------------------------------------------------------------
# Spatial District Permutation, Prompt District Router & Sparsity Distillation
# ---------------------------------------------------------------------------
def permute_geglu_weights_idx(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    perm: np.ndarray | torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute GeGLU `[ff, d]` gate/up rows and `[d, ff]` down columns identically."""
    p_g = torch.as_tensor(perm, dtype=torch.long, device=w_gate.device)
    p_d = p_g.to(w_down.device)
    return w_gate[p_g].contiguous(), w_up[p_g].contiguous(), w_down[:, p_d].contiguous()


class DistrictRouterNet(nn.Module):
    """Lightweight prompt-level Spatial District Router predicting active city districts."""

    def __init__(self, d_in: int, n_districts: int = N_DISTRICTS, rank: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, n_districts, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x), approximate="tanh"))


def fit_district_router(
    inp_tr: torch.Tensor,
    dist_e_tr: torch.Tensor,
    inp_va: torch.Tensor,
    dist_e_va: torch.Tensor,
    *,
    rank: int = 64,
    top_k: int = TOP_K_DISTRICTS,
    steps: int = 240,
    lr: float = 3e-3,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Train per-layer District Router with real `loss.backward()` + `AdamW`."""
    torch.manual_seed(seed)
    n_d = int(dist_e_tr.shape[1])
    k_eff = min(top_k, n_d)
    net = DistrictRouterNet(int(inp_tr.shape[1]), n_districts=n_d, rank=rank).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=2e-2)

    xt = inp_tr.to(device, dtype=torch.float32)
    xv = inp_va.to(device, dtype=torch.float32)
    yt_p = (dist_e_tr.to(device, dtype=torch.float32) + 1e-12)
    yt_p = yt_p / yt_p.sum(dim=-1, keepdim=True)
    yv_e = dist_e_va.to(device, dtype=torch.float32)
    yv_total = yv_e.sum(dim=-1).clamp(min=1e-12)

    def _eval_recall() -> Tuple[float, float, float]:
        with torch.no_grad():
            lg = net(xv)
            pred_top = torch.topk(lg, k=k_eff, dim=-1).indices
            true_top = torch.topk(yv_e, k=k_eff, dim=-1).indices
            cap = yv_e.gather(1, pred_top).sum(dim=-1) / yv_total
            oracle = yv_e.gather(1, true_top).sum(dim=-1) / yv_total
            recall = float((cap / oracle.clamp(min=1e-12)).mean().item())
            top1_acc = float((lg.argmax(dim=-1) == yv_e.argmax(dim=-1)).float().mean().item())
            return recall, top1_acc, float(cap.mean().item())

    rec0, acc0, _ = _eval_recall()
    best_rec = -1.0
    best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    loss_start = None
    loss_end = None
    for st in range(steps):
        opt.zero_grad()
        lg = net(xt)
        loss_kl = F.kl_div(F.log_softmax(lg, dim=-1), yt_p, reduction="batchmean")
        loss_bce = F.binary_cross_entropy_with_logits(lg, (yt_p > (1.0 / float(n_d))).float())
        loss = loss_kl + 0.25 * loss_bce
        if loss_start is None:
            loss_start = float(loss.item())
        loss_end = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (st + 1) % 40 == 0 or st + 1 == steps:
            rec_v, _, _ = _eval_recall()
            if rec_v > best_rec:
                best_rec = rec_v
                best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    net.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    rec1, acc1, cap1 = _eval_recall()
    state_out = {k: v.to(torch.float16).cpu() for k, v in best_sd.items()}
    return state_out, {
        "loss_before": round(float(loss_start or 0.0), 5),
        "loss_after": round(float(loss_end or 0.0), 5),
        "val_top2_recall_before": round(rec0, 4),
        "val_top2_recall_after": round(rec1, 4),
        "val_top1_acc_after": round(acc1, 4),
        "val_district_cap": round(cap1, 4),
    }


def _compute_district_capture_stats(
    seq_energy: np.ndarray,
    district_slices: Sequence[Tuple[str, int, int]],
    top_k: int = TOP_K_DISTRICTS,
) -> Tuple[float, float, np.ndarray]:
    """Compute mean Shared+Top-K energy capture on ALREADY permuted `[N_seq, ff]` energy."""
    non_shared = [(s, e) for nm, s, e in district_slices if nm != "shared_core"]
    shared = [(s, e) for nm, s, e in district_slices if nm == "shared_core"]
    n_seq = seq_energy.shape[0]
    sh_e = seq_energy[:, shared[0][0] : shared[0][1]].sum(axis=1) if shared else np.zeros(n_seq, dtype=np.float64)
    d_e = np.stack([seq_energy[:, s:e].sum(axis=1) for s, e in non_shared], axis=1)
    tot_e = np.maximum(seq_energy.sum(axis=1), 1e-12)
    k_eff = min(top_k, len(non_shared))
    top_idx = np.argsort(-d_e, axis=1)[:, :k_eff]
    top_sum = np.take_along_axis(d_e, top_idx, axis=1).sum(axis=1)
    cap_mean = float(np.mean((sh_e + top_sum) / tot_e))
    oracle_mask = np.zeros_like(seq_energy, dtype=bool)
    if shared:
        oracle_mask[:, shared[0][0] : shared[0][1]] = True
    for r in range(n_seq):
        for di in top_idx[r]:
            s, e = non_shared[int(di)]
            oracle_mask[r, s:e] = True
    return cap_mean, float(np.mean(sh_e / tot_e)), oracle_mask


def distill_district_locality(
    hm_tr: torch.Tensor,
    hm_va: torch.Tensor,
    w_gate_cpu: torch.Tensor,
    w_up_cpu: torch.Tensor,
    w_down_cpu: torch.Tensor,
    district_slices: Sequence[Tuple[str, int, int]],
    *,
    top_k: int = TOP_K_DISTRICTS,
    steps: int = 40,
    lr: float = 2e-4,
    l1_coeff: float = 0.05,
    max_rel_mse_drift: float = 0.05,
    batch_size: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    Lightweight real backprop District-Sparsity Distillation (`loss.backward()` + `AdamW`)
    pushing off-district GeGLU activations toward zero while anchoring MLP outputs to Teacher.
    Rolls back deterministically if validation MSE drifts > `max_rel_mse_drift`.
    """
    compute_dt = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    ff = int(w_gate_cpu.shape[0])
    wg = nn.Parameter(w_gate_cpu.to(device, dtype=torch.float32).clone())
    wu = nn.Parameter(w_up_cpu.to(device, dtype=torch.float32).clone())
    wd = nn.Parameter(w_down_cpu.to(device, dtype=torch.float32).clone())

    hm_t = hm_tr.to(device, dtype=torch.float32)  # [N_tr, T, D] or [N_tr, D]
    hm_v = hm_va.to(device, dtype=torch.float32)
    single_tok = hm_t.ndim == 2
    if single_tok:
        hm_t = hm_t.unsqueeze(1)
        hm_v = hm_v.unsqueeze(1)

    with torch.no_grad():
        def _mlp_teach_and_energy(x_in: torch.Tensor):
            outs, ens = [], []
            for b0 in range(0, x_in.shape[0], 32):
                xb = x_in[b0 : min(b0 + 32, x_in.shape[0])]
                act = F.gelu(F.linear(xb.to(compute_dt), wg.to(compute_dt)).float(), approximate="tanh") * F.linear(xb.to(compute_dt), wu.to(compute_dt)).float()
                yb = F.linear(act.to(compute_dt), wd.to(compute_dt)).float()
                outs.append(yb)
                ens.append((act * act).sum(dim=1).cpu().numpy())
            return torch.cat(outs, 0), np.concatenate(ens, axis=0)

        yt_tr, en_tr = _mlp_teach_and_energy(hm_t)
        yt_va, en_va = _mlp_teach_and_energy(hm_v)

    cap_tr0, _, mask_tr_np = _compute_district_capture_stats(en_tr, district_slices, top_k=top_k)
    cap_va0, _, mask_va_np = _compute_district_capture_stats(en_va, district_slices, top_k=top_k)
    off_mask_tr = torch.from_numpy(~mask_tr_np).to(device=device, dtype=torch.float32).unsqueeze(1)  # [N_tr, 1, ff]
    va_var = float(yt_va.pow(2).mean().clamp(min=1e-8).item())

    opt = torch.optim.AdamW([wg, wu, wd], lr=lr, weight_decay=1e-4)
    n_seq = hm_t.shape[0]
    bs = min(int(batch_size or 32), n_seq)
    loss0 = loss1 = None
    for st in range(steps):
        perm_b = torch.randperm(n_seq, device=device)[:bs]
        opt.zero_grad()
        xb = hm_t[perm_b]
        act = F.gelu(F.linear(xb.to(compute_dt), wg.to(compute_dt)).float(), approximate="tanh") * F.linear(xb.to(compute_dt), wu.to(compute_dt)).float()
        yp = F.linear(act.to(compute_dt), wd.to(compute_dt)).float()
        l_mse = F.mse_loss(yp, yt_tr[perm_b]) / max(va_var, 1e-6)
        act_sq = act * act
        l_off = (act_sq * off_mask_tr[perm_b]).sum(dim=-1).mean() / act_sq.sum(dim=-1).mean().clamp(min=1e-8)
        loss = l_mse + l1_coeff * l_off
        if loss0 is None:
            loss0 = float(loss.item())
        loss1 = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([wg, wu, wd], 1.0)
        opt.step()

    with torch.no_grad():
        yp_va, en_va1 = _mlp_teach_and_energy(hm_v)
        rel_mse = float(F.mse_loss(yp_va, yt_va).item() / max(va_var, 1e-6))
    cap_va1, _, _ = _compute_district_capture_stats(en_va1, district_slices, top_k=top_k)

    accepted = (cap_va1 > cap_va0 + 0.005) and (rel_mse <= max_rel_mse_drift)
    if accepted:
        w_out = (wg.detach().to(w_gate_cpu.dtype).cpu(), wu.detach().to(w_up_cpu.dtype).cpu(), wd.detach().to(w_down_cpu.dtype).cpu())
    else:
        w_out = (w_gate_cpu, w_up_cpu, w_down_cpu)
    del wg, wu, wd, opt, off_mask_tr, yt_tr, yt_va
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return w_out[0], w_out[1], w_out[2], {
        "accepted": bool(accepted),
        "cap_tr_before": round(cap_tr0, 4),
        "cap_va_before": round(cap_va0, 4),
        "cap_va_after": round(cap_va1, 4),
        "val_rel_mse": round(rel_mse, 5),
        "loss_before": round(float(loss0 or 0.0), 5),
        "loss_after": round(float(loss1 or 0.0), 5),
    }


# ---------------------------------------------------------------------------
# Rotated-Basis GTA LOD Residual Split + Tile-16 Packing
# ---------------------------------------------------------------------------
def split_and_pack_lod_matrix(
    w_cpu: torch.Tensor,
    h_gpu: Optional[torch.Tensor],
    x_rms_gpu: Optional[torch.Tensor],
    district_slices: Sequence[Tuple[str, int, int]],
    is_down_proj: bool,
    *,
    bsize: int,
    device: torch.device,
    fwht_gpu_fn: Callable[[torch.Tensor, int], torch.Tensor],
    fwht_2sided_fn: Callable[[torch.Tensor, Optional[torch.Tensor], int, bool], torch.Tensor],
    gptq_gpu_fn: Callable[[torch.Tensor, torch.Tensor, int, int], None],
    pack_asym1_fn: Callable[[np.ndarray, int, int, int], Tuple[np.ndarray, np.ndarray]],
    pack_sym2_fn: Callable[[np.ndarray, int, int, int], Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, object]:
    """
    Compute SmoothQuant `inv_s` and FWHT `w_rot_orig`, run ONE Cholesky `U_chol` pass to pack:
      1) `W_base(1b)` Row-Major (`1 | 0x40`) for pinned RAM execution,
      2) `ΔW_hi-res(2b)` (`E_rot = W_rot_orig - W_rot_1b`, GPTQ `mode=2`) sliced into
         `Zone 0 Shared Core` + `M` Semantic District `.t16c` records.
    Returns packed buffers plus `w_eff_1b`, `w_eff_full`, `base_cos`, `lod_full_cos`, `mse_ratio`.
    """
    d_out, d_in = int(w_cpu.shape[0]), int(w_cpu.shape[1])
    w_gpu = w_cpu.to(device, dtype=torch.float32)

    if x_rms_gpu is not None:
        xr_w = x_rms_gpu.to(device, dtype=torch.float32)
        sq = (xr_w.clamp(min=1e-5).pow(0.5) / w_gpu.abs().amax(dim=0).clamp(min=1e-5).pow(0.5)).clamp(0.25, 4.0)
        sq = sq / sq.mean().clamp(min=1e-5)
        inv_s_gpu = 1.0 / sq
        w_scaled = w_gpu * sq.unsqueeze(0)
        inv_s_np = inv_s_gpu.detach().cpu().numpy().astype("<f4")
        del xr_w, sq
    else:
        inv_s_gpu = None
        w_scaled = w_gpu
        inv_s_np = np.ones((d_in,), dtype="<f4")

    w_rot_orig = fwht_gpu_fn(w_scaled, bsize).contiguous()
    if w_scaled is not w_gpu:
        del w_scaled

    U_chol = None
    if h_gpu is not None and h_gpu.shape[0] == d_in:
        with torch.no_grad():
            h_rot = fwht_2sided_fn(h_gpu.to(device), inv_s_gpu, bsize, consume=True)
            if torch.isfinite(h_rot).all():
                diag_m = h_rot.diagonal().mean().clamp(min=1e-6)
                L = None
                for delta_f in (0.02, 0.05, 0.15, 0.50, 1.50):
                    h_rot.diagonal().add_(delta_f * diag_m)
                    try:
                        L = torch.linalg.cholesky(h_rot)
                        break
                    except RuntimeError:
                        L = None
                del h_rot
                if L is not None:
                    try:
                        h_inv = torch.cholesky_inverse(L)
                        del L
                        symmetrize_matrix_inplace(h_inv)
                        U_chol = torch.linalg.cholesky(h_inv, upper=True).contiguous()
                        del h_inv
                    except Exception:
                        U_chol = None
            else:
                del h_rot

    w_rot_1b = w_rot_orig.clone()
    if U_chol is not None:
        gptq_gpu_fn(w_rot_1b, U_chol, bsize, 0)
    w_rot_1b_np = w_rot_1b.detach().cpu().numpy().astype(np.float32, copy=True)
    pb_1b_col, w_q_1b_np = pack_asym1_fn(w_rot_1b_np, d_out, d_in, bsize)
    pb_1b_rm = v70.transpose_t16c_1b_col_to_rm(pb_1b_col, d_out, d_in, bsize)
    w_rot_1b = torch.from_numpy(np.ascontiguousarray(w_q_1b_np, dtype=np.float32)).to(device)
    del pb_1b_col, w_rot_1b_np, w_q_1b_np

    # Reconstruct exact discrete 1b rotated weights
    w_eff_1b = fwht_gpu_fn(w_rot_1b, bsize)
    if inv_s_gpu is not None:
        w_eff_1b = w_eff_1b * inv_s_gpu.unsqueeze(0)

    # Residual ΔW_hi-res in rotated basis: E_rot = W_rot_orig - Q_1b(W_rot)
    w_res_rot = (w_rot_orig - w_rot_1b).contiguous()
    del w_rot_orig, w_rot_1b
    if U_chol is not None:
        gptq_gpu_fn(w_res_rot, U_chol, bsize, 2)
        del U_chol

    w_res_np = w_res_rot.detach().cpu().numpy().astype(np.float32, copy=False)
    del w_res_rot

    shard_records: Dict[str, Tuple[int, int, np.ndarray, np.ndarray]] = {}
    w_res_q_np = np.empty_like(w_res_np, dtype=np.float32)
    for sl_name, s_idx, e_idx in district_slices:
        if is_down_proj:
            sub = np.ascontiguousarray(w_res_np[:, s_idx:e_idx], dtype=np.float32)
            do_s, di_s = d_out, e_idx - s_idx
            inv_s_sl = np.ascontiguousarray(inv_s_np[s_idx:e_idx], dtype="<f4")
            pb_sl, w_q_sl = pack_sym2_fn(sub, do_s, di_s, bsize)
            w_res_q_np[:, s_idx:e_idx] = w_q_sl
        else:
            sub = np.ascontiguousarray(w_res_np[s_idx:e_idx, :], dtype=np.float32)
            do_s, di_s = e_idx - s_idx, d_in
            inv_s_sl = inv_s_np
            pb_sl, w_q_sl = pack_sym2_fn(sub, do_s, di_s, bsize)
            w_res_q_np[s_idx:e_idx, :] = w_q_sl
        shard_records[sl_name] = (do_s, di_s, pb_sl, inv_s_sl)
    del w_res_np

    w_eff_res = fwht_gpu_fn(torch.from_numpy(w_res_q_np).to(device), bsize)
    del w_res_q_np
    if inv_s_gpu is not None:
        w_eff_res = w_eff_res * inv_s_gpu.unsqueeze(0)
        del inv_s_gpu
    w_eff_full = w_eff_1b + w_eff_res

    mse_1b = float(F.mse_loss(w_eff_1b, w_gpu).item())
    mse_full = float(F.mse_loss(w_eff_full, w_gpu).item())
    mse_ratio = mse_1b / max(mse_full, 1e-12)
    del w_gpu

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "pb_base_rm": pb_1b_rm,
        "inv_s_base": inv_s_np,
        "shard_records": shard_records,
        "w_eff_1b": w_eff_1b.to(torch.float16),
        "w_eff_full": w_eff_full.to(torch.float16),
        "mse_1b": mse_1b,
        "mse_full": mse_full,
        "mse_ratio": mse_ratio,
    }


def forward_geglu_lod_batch(
    hm: torch.Tensor,
    wg_1b: torch.Tensor,
    wg_full: torch.Tensor,
    wu_1b: torch.Tensor,
    wu_full: torch.Tensor,
    wd_1b: torch.Tensor,
    wd_full: torch.Tensor,
    active_col_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate exact per-sequence GTA LOD MLP output:
      - Inactive columns (`~active_col_mask`): execute at 1-bit `W_base`
      - Active columns (`Shared Core + Top-K Districts`): execute at full `W_base + ΔW_hi-res`
    `active_col_mask` has shape `[B, ff]` bool on `hm.device`.
    """
    compute_dt = hm.dtype
    m = active_col_mask.unsqueeze(1)  # [B, 1, ff]
    g_1b = F.linear(hm, wg_1b.to(compute_dt))
    g_hi = F.linear(hm, (wg_full - wg_1b).to(compute_dt))
    g_tot = g_1b + torch.where(m, g_hi, torch.zeros_like(g_hi))
    del g_1b, g_hi
    u_1b = F.linear(hm, wu_1b.to(compute_dt))
    u_hi = F.linear(hm, (wu_full - wu_1b).to(compute_dt))
    u_tot = u_1b + torch.where(m, u_hi, torch.zeros_like(u_hi))
    del u_1b, u_hi
    act = F.gelu(g_tot.float(), approximate="tanh") * u_tot.float()
    del g_tot, u_tot
    act_1b = act.to(compute_dt)
    y_out = F.linear(act_1b, wd_1b.to(compute_dt)).float()
    del act_1b
    act_hi = torch.where(m, act, torch.zeros_like(act)).to(compute_dt)
    del act
    y_out.add_(F.linear(act_hi, (wd_full - wd_1b).to(compute_dt)).float())
    del act_hi
    return y_out


# ---------------------------------------------------------------------------
# Fail-Closed Multi-Shard Drive Sync
# ---------------------------------------------------------------------------
def sync_v7_5_checkpoints_to_drive(
    expected_positions: Dict[Path, int],
    drive_dir: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
) -> None:
    """Sync ALL base + LOD `.partial` shards & verify sizes BEFORE writing `stage3_layer_ckpt.pt` LAST."""
    d_dir = DRIVE_DIR if drive_dir is None else drive_dir
    c_dir = CKPT_DIR if ckpt_dir is None else ckpt_dir
    if d_dir is None:
        return
    drive_ckpt = d_dir / "ckpt_v7_5"
    drive_ckpt.mkdir(parents=True, exist_ok=True)

    for src_p, exp_pos in expected_positions.items():
        if not src_p.is_file():
            print(f"  [Drive Sync Abort] Missing local {src_p}; skipping Drive sync.", flush=True)
            return
        cur_sz = src_p.stat().st_size
        if cur_sz != exp_pos:
            print(f"  [Drive Sync Abort] Size mismatch for {src_p.name}: {cur_sz:,} != {exp_pos:,}", flush=True)
            return
        dst_p = d_dir / (src_p.name + ".partial")
        tmp_p = dst_p.with_suffix(dst_p.suffix + ".tmp")
        shutil.copy2(src_p, tmp_p)
        os.replace(tmp_p, dst_p)
        if not dst_p.is_file() or dst_p.stat().st_size != cur_sz:
            print(f"  [Drive Sync Abort] Short Drive copy for {dst_p.name}; keeping prior ckpt.", flush=True)
            return

    names = [p.name for p in sorted(c_dir.glob("*")) if p.is_file() and not p.name.endswith(".tmp")]
    names = [n for n in names if n != "stage3_layer_ckpt.pt"] + (["stage3_layer_ckpt.pt"] if "stage3_layer_ckpt.pt" in names else [])
    for fn in names:
        sp = c_dir / fn
        dp = drive_ckpt / fn
        tp = dp.with_suffix(dp.suffix + ".tmp")
        shutil.copy2(sp, tp)
        os.replace(tp, dp)


def persist_v7_5_file(src: Path, drive_dir: Optional[Path] = None) -> Tuple[int, str]:
    sz = src.stat().st_size
    dig = v70.sha256_file(src)
    d_dir = DRIVE_DIR if drive_dir is None else drive_dir
    if d_dir is not None:
        d_dir.mkdir(parents=True, exist_ok=True)
        dst = d_dir / src.name
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        if v70.sha256_file(dst) != dig or dst.stat().st_size != sz:
            raise IOError(f"Drive SHA-256 mismatch for {src.name}")
        print(f"  [DRIVE PERSISTED] {dst.name} ({sz:,} B, sha256={dig[:16]}...)", flush=True)
    return sz, dig


# ---------------------------------------------------------------------------
# Fail-Closed 6-Gate CPU Self-Test
# ---------------------------------------------------------------------------
def run_selftest() -> int:
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        st = "PASS" if ok else "FAIL"
        print(f"[{st}] {name}" + (f" | {detail}" if detail else ""), flush=True)
        if not ok:
            failures += 1

    # Gate 1: V7.5 budget & Pi 4 RAM headroom invariants
    bgt = compute_v7_5_budget_bytes()
    g1_ok = (
        bgt["attn_1b_cnt"] == 0
        and bgt["base_total_bytes"] <= MAX_BASE_T16C_BYTES
        and bgt["active_text_ram_bytes"] <= MAX_ACTIVE_TEXT_RAM_BYTES
        and bgt["n_shared_blks"] == 12
        and bgt["rem_blks"] == 72
    )
    gate(
        "Gate 1: V7.5 Spatial Districts byte & Pi 4 RAM budget",
        g1_ok,
        f"base_tot={bgt['base_total_bytes']/1024**3:.3f}GiB (<=5.50) shared={bgt['lod_shared_bytes']/1024**3:.3f}GiB "
        f"d_max={bgt['lod_district_max_bytes']/1024**3:.3f}GiB active_ram={bgt['active_text_ram_bytes']/1024**3:.3f}GiB (<=7.10)",
    )

    rng = np.random.default_rng(2026)
    torch.manual_seed(2026)

    # Gate 2: Spatial District clustering bijection + 256-align + lossless GeGLU permutation invariance
    d_mod_t, ff_t, bsz_t = 256, 2560, 256
    # Plant Shared Core + 4 distinct blocks on 4 domains (multiples of bsz_t)
    e_dom = np.full((64, ff_t), 0.02, dtype=np.float64)
    e_dom[:, :512] = 6.0  # universal Shared Core (2 blocks)
    for dom in range(4):
        idx_p = np.arange(dom, 64, 4)
        e_dom[idx_p, 512 + dom * 512 : 512 + (dom + 1) * 512] = 5.0
    perm_np, slices_np, _ = sdc.cluster_spatial_districts(e_dom, n_districts=4, shared_frac=512.0 / ff_t, block_size=bsz_t)
    wg = torch.randn(ff_t, d_mod_t)
    wu = torch.randn(ff_t, d_mod_t)
    wd = torch.randn(d_mod_t, ff_t)
    x_probe = torch.randn(8, d_mod_t)
    y_orig = F.linear(F.gelu(F.linear(x_probe, wg), approximate="tanh") * F.linear(x_probe, wu), wd)
    wg_p, wu_p, wd_p = permute_geglu_weights_idx(wg, wu, wd, perm_np)
    y_perm = F.linear(F.gelu(F.linear(x_probe, wg_p), approximate="tanh") * F.linear(x_probe, wu_p), wd_p)
    perm_err = float((y_orig - y_perm).abs().max().item())
    aligned = all((s % bsz_t == 0) and (e % bsz_t == 0) and (e > s) for _, s, e in slices_np)
    gate(
        "Gate 2: Spatial District bijection, 256-align & lossless GeGLU invariance",
        np.unique(perm_np).size == ff_t and aligned and perm_err < 1e-4,
        f"slices={len(slices_np)} max_diff={perm_err:.2e}",
    )

    # Gate 3: Rotated-basis GTA LOD Residual Split + Tile-16 binary decode parity + >=4x MSE reduction
    def _fwht_cpu(t: torch.Tensor, bsz: int) -> torch.Tensor:
        h = torch.from_numpy(pa.make_sylvester_hadamard(bsz)).to(t.dtype)
        return (t.reshape(-1, bsz) @ h).reshape(t.shape)

    def _pack_a1_cpu(w_np: np.ndarray, do: int, di: int, bsz: int) -> Tuple[np.ndarray, np.ndarray]:
        return pack_tile16_asym1_np(w_np, bsz)

    def _pack_s2_cpu(w_np: np.ndarray, do: int, di: int, bsz: int) -> Tuple[np.ndarray, np.ndarray]:
        return pack_tile16_sym2_np(w_np, bsz)

    w_down_t = wd_p[:64, :].clone()  # [64, 2560] smooth gaussian + outliers
    w_down_t[:, ::17] *= 3.5
    xr_down = torch.linspace(0.5, 2.5, ff_t)
    lod_res = split_and_pack_lod_matrix(
        w_down_t, None, xr_down, slices_np, is_down_proj=True,
        bsize=bsz_t, device=torch.device("cpu"),
        fwht_gpu_fn=_fwht_cpu,
        fwht_2sided_fn=lambda h, *_args: h,
        gptq_gpu_fn=lambda *_args: None,
        pack_asym1_fn=_pack_a1_cpu,
        pack_sym2_fn=_pack_s2_cpu,
    )
    # Reassemble residual shards from raw Tile-16 bytes and compare against w_eff_full - w_eff_1b!
    res_parts = []
    for nm, s_idx, e_idx in slices_np:
        do_s, di_s, pb_s, inv_sl = lod_res["shard_records"][nm]
        dec_s = decode_tile16_sym2_np(pb_s, do_s, di_s, bsz_t, has_fwht=True) * inv_sl[None, :]
        res_parts.append(dec_s)
    dec_full_delta = np.concatenate(res_parts, axis=1)
    rt_err = float(np.max(np.abs(dec_full_delta - (lod_res["w_eff_full"] - lod_res["w_eff_1b"]).float().numpy())))
    cos_1b = float(F.cosine_similarity(lod_res["w_eff_1b"].float().flatten(), w_down_t.flatten(), dim=0))
    cos_lod = float(F.cosine_similarity(lod_res["w_eff_full"].float().flatten(), w_down_t.flatten(), dim=0))
    gate(
        "Gate 3: GTA LOD Residual Split (1b Base + 2b Shared/District shards)",
        lod_res["mse_ratio"] >= 4.0 and cos_lod > cos_1b and rt_err < 1e-2,
        f"mse_red={lod_res['mse_ratio']:.2f}x cos: {cos_1b:.4f}->{cos_lod:.4f} tile16_rt_err={rt_err:.2e}",
    )

    # Gate 4: Real PyTorch backprop District Router + District-Sparsity Distillation + forward_geglu_lod_batch
    n_tok_g4 = 48
    hm_tr_g4 = torch.randn(36, 4, d_mod_t)
    hm_va_g4 = torch.randn(12, 4, d_mod_t)
    # Construct planted domain signals aligned to slices_np
    wg_s = 0.05 * torch.randn(ff_t, d_mod_t)
    wu_s = 0.05 * torch.randn(ff_t, d_mod_t)
    wd_s = 0.05 * torch.randn(d_mod_t, ff_t)
    dom_vecs = F.normalize(torch.randn(4, d_mod_t), dim=-1)
    for idx_b in range(36):
        dom = idx_b % 4
        hm_tr_g4[idx_b] = dom_vecs[dom] + 0.1 * torch.randn(4, d_mod_t)
    for idx_b in range(12):
        dom = idx_b % 4
        hm_va_g4[idx_b] = dom_vecs[dom] + 0.1 * torch.randn(4, d_mod_t)
    # Plant moderate off-district noise then run real distill_district_locality
    for dom in range(4):
        s_d, e_d = slices_np[1 + dom][1], slices_np[1 + dom][2]
        wg_s[s_d:e_d] = 0.8 * dom_vecs[dom].unsqueeze(0) + 0.05 * torch.randn(e_d - s_d, d_mod_t)
        wu_s[s_d:e_d] = 0.8 * dom_vecs[dom].unsqueeze(0) + 0.05 * torch.randn(e_d - s_d, d_mod_t)
    # Train District Router cleanly
    with torch.no_grad():
        e_tr_all = (F.gelu(F.linear(hm_tr_g4, wg_s), approximate="tanh") * F.linear(hm_tr_g4, wu_s)).pow(2).sum(dim=1)
        e_va_all = (F.gelu(F.linear(hm_va_g4, wg_s), approximate="tanh") * F.linear(hm_va_g4, wu_s)).pow(2).sum(dim=1)
        de_tr = torch.stack([e_tr_all[:, s:e].sum(dim=1) for nm, s, e in slices_np if nm != "shared_core"], dim=1)
        de_va = torch.stack([e_va_all[:, s:e].sum(dim=1) for nm, s, e in slices_np if nm != "shared_core"], dim=1)
    r_sd, r_met = fit_district_router(
        hm_tr_g4.mean(dim=1), de_tr, hm_va_g4.mean(dim=1), de_va,
        rank=32, top_k=2, steps=160, lr=5e-3, device=torch.device("cpu"),
    )
    # Test District-Sparsity Distillation on a diffuse unclustered probe matrix (native cap ~0.55..0.68)
    hm_d_tr = torch.zeros(48, 4, d_mod_t)
    hm_d_va = torch.zeros(16, 4, d_mod_t)
    for idx_b in range(48):
        hm_d_tr[idx_b] = dom_vecs[idx_b % 4] + 0.08 * torch.randn(4, d_mod_t)
    for idx_b in range(16):
        hm_d_va[idx_b] = dom_vecs[idx_b % 4] + 0.08 * torch.randn(4, d_mod_t)
    wg_diff = 0.25 * torch.randn(ff_t, d_mod_t)
    wu_diff = 0.25 * torch.randn(ff_t, d_mod_t)
    for dom in range(4):
        s_d, e_d = slices_np[1 + dom][1], slices_np[1 + dom][2]
        wg_diff[s_d:e_d] += 0.22 * dom_vecs[dom].unsqueeze(0)
        wu_diff[s_d:e_d] += 0.22 * dom_vecs[dom].unsqueeze(0)
    _, _, _, d_met = distill_district_locality(
        hm_d_tr, hm_d_va, wg_diff, wu_diff, wd_s, slices_np,
        top_k=2, steps=45, lr=1.5e-3, l1_coeff=2.0, max_rel_mse_drift=0.35,
    )
    gate(
        "Gate 4: Real backprop District Router & Sparsity Distillation",
        r_met["val_top2_recall_after"] >= 0.90 and r_met["loss_after"] < r_met["loss_before"] and d_met["accepted"] and d_met["cap_va_after"] > d_met["cap_va_before"],
        f"router_recall={r_met['val_top2_recall_before']}->{r_met['val_top2_recall_after']} "
        f"distill_cap={d_met['cap_va_before']}->{d_met['cap_va_after']} (acc={d_met['accepted']}, rel_mse={d_met['val_rel_mse']})",
    )

    # Gate 5 & 6: Multi-shard T16C base + shared + 4 district shards write/verify + fail-closed checks
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        drv_p = td_p / "drv"
        ckp_p = td_p / "ckp"
        drv_p.mkdir(); ckp_p.mkdir()
        shard_paths: Dict[str, Path] = {"base": td_p / "base.t16c"}
        for nm, _, _ in slices_np:
            shard_paths[nm] = td_p / f"lod_{nm}.t16c"
        n_shard_mats = 1
        with open(shard_paths["base"], "wb") as fb:
            fb.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, 1, 1, 0))
            fb.write(struct.pack("<H", 4) + b"nrm0" + struct.pack("<I", 4) + np.ones(4, dtype="<f4").tobytes())
            p3._write_mat_record(
                fb, fb.tell(), "model.language_model.layers.2.mlp.down_proj.weight",
                64, ff_t, bsz_t, 1 | T16C_ROWMAJOR_BIT, True, True,
                lod_res["pb_base_rm"], lod_res["inv_s_base"], None,
            )
        exp_pos: Dict[Path, int] = {shard_paths["base"]: shard_paths["base"].stat().st_size}
        for nm, _, _ in slices_np:
            do_s, di_s, pb_s, inv_sl = lod_res["shard_records"][nm]
            sp_i = shard_paths[nm]
            with open(sp_i, "wb") as fs:
                fs.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, 0, n_shard_mats, 0))
                p3._write_mat_record(
                    fs, fs.tell(), "model.language_model.layers.2.mlp.down_proj.weight",
                    do_s, di_s, bsz_t, 2, False, True, pb_s, inv_sl, None,
                )
            exp_pos[sp_i] = sp_i.stat().st_size

        all_ver = [v70.verify_t16c_integrity(p)[0] == 1 for p in shard_paths.values()]
        (ckp_p / "stage3_layer_ckpt.pt").write_bytes(b"ckpt")
        bad_exp = dict(exp_pos)
        bad_exp[shard_paths["base"]] += 64
        sync_v7_5_checkpoints_to_drive(bad_exp, drive_dir=drv_p, ckpt_dir=ckp_p)
        refused = not (drv_p / "ckpt_v7_5" / "stage3_layer_ckpt.pt").exists()
        sync_v7_5_checkpoints_to_drive(exp_pos, drive_dir=drv_p, ckpt_dir=ckp_p)
        synced = all((drv_p / (p.name + ".partial")).is_file() for p in exp_pos) and (drv_p / "ckpt_v7_5" / "stage3_layer_ckpt.pt").is_file()
        gate(
            "Gate 5: Multi-shard Base + LOD Shared/District T16C integrity & .partial-first sync",
            all(all_ver) and refused and synced,
            f"shards_verified={len(all_ver)} refused_mismatch={refused} synced_all={synced}",
        )

    dj = spmod.assert_holdouts_disjoint()
    caught_misalign = False
    try:
        sdc.cluster_spatial_districts(np.ones((4, 300)), n_districts=2, shared_frac=0.25, block_size=256)
    except ValueError:
        caught_misalign = True
    a100_cfg = configure_a100_max_utilization(80.0, has_cuda=False, respect_env=False)
    g4_cfg = configure_a100_max_utilization(96.0, has_cuda=False, respect_env=False)
    l4_cfg = configure_a100_max_utilization(22.5, has_cuda=True, respect_env=False)
    gate(
        "Gate 6: Holdout disjointness, fail-closed checks & G4/A100/L4 config",
        dj["overlap_4grams"] == 0
        and caught_misalign
        and g4_cfg["profile"] == "G4-96GB-MAX"
        and g4_cfg["batch_seq"] == 256
        and a100_cfg["profile"] == "A100-80GB-MAX"
        and a100_cfg["batch_seq"] == 192
        and l4_cfg["profile"] == "L4-22GB-MAX"
        and l4_cfg["batch_seq"] == 64
        and l4_cfg["als_ev_toks"] == 4096,
        f"overlap={dj['overlap_4grams']} misaligned_caught={caught_misalign} "
        f"g4={g4_cfg['profile']}/bs={g4_cfg['batch_seq']} a100={a100_cfg['profile']}/bs={a100_cfg['batch_seq']} l4={l4_cfg['profile']}/bs={l4_cfg['batch_seq']}",
    )

    print(f"V7_5_SELFTEST: {6 - failures}/6 gates passed", flush=True)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Full 60-Layer A100-SXM4-80GB Pipeline (All 60 Layers, Max GPU Utilization)
# ---------------------------------------------------------------------------
def run_pipeline() -> Dict[str, object]:
    """Run complete Gemma 4 31B V7.5 60-Layer Spatial Districts + GTA LOD + CRTR + Assistant-V7 build."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_pipeline() requires CUDA GPU (pass --selftest on CPU)")
    if run_selftest() != 0:
        raise RuntimeError("V7.5 pre-flight CPU selftest failed")

    t0_all = time.perf_counter()
    device = torch.device("cuda:0")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    hw_cfg = configure_a100_max_utilization(gpu_vram_gb, has_cuda=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    shard_map: Dict[str, Path] = {"shared_core": OUT_LOD_SHARED}
    for d_i in range(N_DISTRICTS):
        shard_map[f"district_{d_i}"] = OUT_LOD_DISTRICTS[d_i]
    all_t16c_paths = [OUT_BASE_T16C] + list(shard_map.values())

    if DRIVE_DIR is not None:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        d_ckpt = DRIVE_DIR / "ckpt_v7_5"
        if d_ckpt.is_dir():
            print(f"[Drive Resume] Restoring checkpoints from {d_ckpt}...", flush=True)
            for p in d_ckpt.glob("*"):
                if p.is_file() and not p.name.endswith(".tmp") and not (CKPT_DIR / p.name).is_file():
                    shutil.copy2(p, CKPT_DIR / p.name)
        for sp in all_t16c_paths:
            pp = DRIVE_DIR / (sp.name + ".partial")
            if pp.is_file() and not sp.is_file():
                print(f"[Drive Resume] Restoring {pp.name} ({pp.stat().st_size:,} B)...", flush=True)
                shutil.copy2(pp, sp)

    print(
        f"[V7.5 INIT] GPU={gpu_name} ({gpu_vram_gb:.1f} GiB) | profile={hw_cfg['profile']} "
        f"| resident_gpu={hw_cfg['resident_gpu']} | micro_batch={hw_cfg['batch_seq']} "
        f"| tf32={hw_cfg['tf32']} | distill={hw_cfg['distill_enabled']} | DRIVE={DRIVE_DIR}",
        flush=True,
    )

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageDraw
    from safetensors import safe_open
    from safetensors.torch import save_file as st_save_file
    from tokenizers import Tokenizer
    from transformers import AutoConfig, Gemma4ImageProcessor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

    fq_c = THIS_DIR / "fast_quant.c"
    fq_so = Path("/tmp/libfast_quant_v7_5.so")
    subprocess.run(["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", str(fq_c), "-o", str(fq_so), "-lm"], check=True)
    LIB_FQ = ctypes.CDLL(str(fq_so))
    LIB_FQ.t16c_tblk_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    LIB_FQ.t16c_tblk_bytes.restype = ctypes.c_size_t
    LIB_FQ.quant_and_pack_t16c_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]

    def c_pack_tile16_with_q(arr_f32: np.ndarray, do: int, di: int, bsize: int, mode_i: int) -> Tuple[np.ndarray, np.ndarray]:
        n_bits_m, is_asym_m = p3.MODE_TO_T16C[mode_i]
        exp_b = p3.expected_tile16_bytes(do, di, bsize, n_bits_m, is_asym_m)
        buf = np.empty(exp_b, dtype=np.uint8)
        arr_c = np.ascontiguousarray(arr_f32, dtype=np.float32)
        LIB_FQ.quant_and_pack_t16c_f32(arr_c.ctypes.data, do, di, bsize, mode_i, buf.ctypes.data)
        return buf, arr_c

    def c_pack_tile16(arr_f32: np.ndarray, do: int, di: int, bsize: int, mode_i: int) -> np.ndarray:
        return c_pack_tile16_with_q(arr_f32, do, di, bsize, mode_i)[0]

    _h_cache: Dict[int, torch.Tensor] = {}

    def get_h_gpu(bsize: int, dev: torch.device = device) -> torch.Tensor:
        if bsize not in _h_cache:
            h = torch.ones((1, 1), dtype=torch.float32, device=dev)
            k = 1
            while k < bsize:
                h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
                k *= 2
            _h_cache[bsize] = h / math.sqrt(float(bsize))
        return _h_cache[bsize]

    def fwht_gpu(t: torch.Tensor, bsize: int) -> torch.Tensor:
        return (t.reshape(-1, bsize) @ get_h_gpu(bsize, t.device)).reshape(t.shape)

    symmetrize_inplace_ = symmetrize_matrix_inplace

    def fwht_2sided_gpu(h_in: torch.Tensor, inv_s_gpu: Optional[torch.Tensor], bsize: int, consume: bool) -> torch.Tensor:
        h = h_in if consume else h_in.clone()
        if inv_s_gpu is not None:
            inv_s_loc = inv_s_gpu.to(h.device)
            h.mul_(inv_s_loc.unsqueeze(1))
            h.mul_(inv_s_loc.unsqueeze(0))
        d = h.shape[0]
        hb = get_h_gpu(bsize, h.device)
        step = min(d, max(bsize, 2048 - (2048 % bsize)))
        for r0 in range(0, d, step):
            r1 = min(r0 + step, d)
            h[r0:r1] = torch.matmul(hb, h[r0:r1].reshape((r1 - r0) // bsize, bsize, d)).reshape(r1 - r0, d)
        for c0 in range(0, d, step):
            c1 = min(c0 + step, d)
            h[:, c0:c1] = torch.matmul(h[:, c0:c1].reshape(d, (c1 - c0) // bsize, bsize), hb).reshape(d, c1 - c0)
        return symmetrize_inplace_(h)

    def accum_hess(h_acc: torch.Tensor, xsq_acc: torch.Tensor, x_f32: torch.Tensor) -> None:
        xsq_acc.add_((x_f32 * x_f32).sum(dim=0))
        sc = (x_f32.abs().amax() / 2.0).clamp(min=1e-6)
        xs = (x_f32 / sc).to(torch.float16)
        h_acc.add_((xs.T @ xs).float() * (sc * sc))

    def gptq_quant_gpu(W: torch.Tensor, U: torch.Tensor, bsize: int, mode: int) -> None:
        rows, cw = W.shape
        nblks = cw // bsize
        diag_u = torch.diagonal(U).clamp(min=1e-12)
        E_blk = torch.empty((rows, bsize), device=W.device, dtype=torch.float32)
        for b in range(nblks):
            b0, b1 = b * bsize, (b + 1) * bsize
            blk = W[:, b0:b1]
            if mode == 0:
                bias = blk.mean(dim=1)
                for _ in range(2):
                    m1 = blk >= bias.unsqueeze(1)
                    c1 = m1.sum(dim=1)
                    c0 = bsize - c1
                    mp = torch.where(c1 > 0, torch.where(m1, blk, 0.0).sum(dim=1) / c1.clamp(min=1), bias)
                    mn = torch.where(c0 > 0, torch.where(m1, 0.0, blk).sum(dim=1) / c0.clamp(min=1), bias)
                    bias = 0.5 * (mp + mn)
                    sc = 0.5 * (mp - mn)
                bias = bias.to(torch.float16).float()
                sc = sc.to(torch.float16).float()
                sc_p, sc_n = bias + sc, bias - sc
            elif mode == 1:
                sc = 1.25 * blk.abs().mean(dim=1)
                for _ in range(4):
                    tau = 0.5 * sc.unsqueeze(1)
                    pos, neg = blk > tau, blk < -tau
                    num = torch.where(pos, blk, 0.0).sum(dim=1) - torch.where(neg, blk, 0.0).sum(dim=1)
                    cnt = (pos | neg).sum(dim=1)
                    sc = torch.where(cnt > 0, num / cnt.clamp(min=1), 0.0)
                sc = sc.to(torch.float16).float()
                tau_vec = 0.5 * sc
            else:
                max_lvl = 3.0 if mode == 2 else 7.0
                init_k = 0.498 if mode == 2 else 0.293
                sc = init_k * torch.sqrt((blk * blk).mean(dim=1))
                for _ in range(4):
                    inv_s = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0).unsqueeze(1)
                    q = torch.clamp(torch.round((blk * inv_s + max_lvl) * 0.5), 0.0, max_lvl)
                    lv = 2.0 * q - max_lvl
                    den = (lv * lv).sum(dim=1)
                    sc = torch.where(den > 0.0, (lv * blk).sum(dim=1) / den.clamp(min=1e-12), 0.0)
                sc = sc.to(torch.float16).float()
                inv_s_vec = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0)
            for k in range(bsize):
                j = b0 + k
                val = W[:, j].clone()
                if mode == 0:
                    qval = torch.where(val >= bias, sc_p, sc_n)
                elif mode == 1:
                    qval = torch.where(val > tau_vec, sc, torch.where(val < -tau_vec, -sc, 0.0))
                else:
                    q = torch.clamp(torch.round((val * inv_s_vec + max_lvl) * 0.5), 0.0, max_lvl)
                    qval = sc * (2.0 * q - max_lvl)
                W[:, j] = qval
                err = (val - qval) / diag_u[j]
                E_blk[:, k] = err
                if k + 1 < bsize:
                    W[:, j + 1 : b1].sub_(err.unsqueeze(1) * U[j, j + 1 : b1].unsqueeze(0))
            if b1 < cw:
                W[:, b1:].sub_(E_blk @ U[b0:b1, b1:])

    def pack_matrix_gpu_gptq(
        w_cpu: torch.Tensor,
        h_gpu: Optional[torch.Tensor],
        x_rms_gpu: Optional[torch.Tensor],
        mode: int,
        bsize: int = 256,
        lr_rank: int = 0,
        x_eval_gpu: Optional[torch.Tensor] = None,
        to_rowmajor: bool = False,
        out_device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple], torch.Tensor]:
        d_out, d_in = int(w_cpu.shape[0]), int(w_cpu.shape[1])
        n_bits, is_asym = p3.MODE_TO_T16C[mode]
        exp_bytes = p3.expected_tile16_bytes(d_out, d_in, bsize, n_bits, is_asym)
        ret_dev = device if out_device is None else out_device

        if h_gpu is None and lr_rank == 0 and d_out > 32768:
            pbytes = np.empty(exp_bytes, dtype=np.uint8)
            w_eff_cpu = torch.empty((d_out, d_in), dtype=torch.float16)
            tblk_b = p3.tile16_blk_bytes(bsize, n_bits, is_asym) * (d_in // bsize)
            for r0 in range(0, d_out, 16384):
                r1 = min(r0 + 16384, d_out)
                wc_rot = fwht_gpu(w_cpu[r0:r1].to(device, dtype=torch.float32), bsize).contiguous().cpu().numpy().astype(np.float32)
                chunk_pb = c_pack_tile16(wc_rot, r1 - r0, d_in, bsize, mode)
                pbytes[(r0 // 16) * tblk_b : (r0 // 16) * tblk_b + chunk_pb.size] = chunk_pb
                w_eff_cpu[r0:r1] = fwht_gpu(torch.from_numpy(wc_rot).to(device), bsize).to(torch.float16).cpu()
                del wc_rot, chunk_pb
            if to_rowmajor:
                pbytes = v70.transpose_t16c_1b_col_to_rm(pbytes, d_out, d_in, bsize)
            return pbytes, None, None, w_eff_cpu

        w_gpu = w_cpu.to(device, dtype=torch.float32)
        if x_rms_gpu is not None:
            xr_w = x_rms_gpu.to(device, dtype=torch.float32)
            sq = (xr_w.clamp(min=1e-5).pow(0.5) / w_gpu.abs().amax(dim=0).clamp(min=1e-5).pow(0.5)).clamp(0.25, 4.0)
            sq = sq / sq.mean().clamp(min=1e-5)
            inv_s_gpu = 1.0 / sq
            w_scaled = w_gpu * sq.unsqueeze(0)
            inv_s_np = inv_s_gpu.detach().cpu().numpy().astype("<f4")
            del xr_w, sq
        else:
            inv_s_gpu, w_scaled, inv_s_np = None, w_gpu, None

        w_rot = fwht_gpu(w_scaled, bsize).contiguous()
        del w_scaled

        if h_gpu is not None:
            h_t = h_gpu.pop(0) if isinstance(h_gpu, list) else h_gpu
            h_gpu = None
            if h_t.shape[0] == d_in:
                with torch.no_grad():
                    h_rot = fwht_2sided_gpu(h_t.to(device), inv_s_gpu, bsize, consume=True)
                    del h_t
                    if torch.isfinite(h_rot).all():
                        diag_m = h_rot.diagonal().mean().clamp(min=1e-6)
                        L = None
                        for delta_f in (0.02, 0.05, 0.15, 0.50, 1.50):
                            h_rot.diagonal().add_(delta_f * diag_m)
                            try:
                                L = torch.linalg.cholesky(h_rot)
                                break
                            except RuntimeError:
                                L = None
                        del h_rot
                        if L is not None:
                            h_inv = U_chol = None
                            try:
                                h_inv = torch.cholesky_inverse(L)
                                del L; L = None
                                symmetrize_inplace_(h_inv)
                                U_chol = torch.linalg.cholesky(h_inv, upper=True).contiguous()
                                del h_inv; h_inv = None
                                gptq_quant_gpu(w_rot, U_chol, bsize, mode)
                            except Exception as exc:
                                print(f"    [GPTQ fallback RTN {d_out}x{d_in}: {exc}]", flush=True)
                            finally:
                                del L, h_inv, U_chol
                    else:
                        del h_rot
            else:
                del h_t

        w_work_np = w_rot.detach().cpu().numpy().astype(np.float32, copy=True)
        del w_rot
        pbytes = c_pack_tile16(w_work_np, d_out, d_in, bsize, mode)
        w_eff_gpu = fwht_gpu(torch.from_numpy(w_work_np).to(device, dtype=torch.float32), bsize)
        del w_work_np
        if inv_s_gpu is not None:
            w_eff_gpu.mul_(inv_s_gpu.unsqueeze(0))
            del inv_s_gpu

        low_rank = None
        if lr_rank > 0 and x_eval_gpu is not None and x_eval_gpu.shape[0] >= lr_rank * 2:
            with torch.no_grad():
                n_ev = min(x_eval_gpu.shape[0], int(hw_cfg["als_ev_toks"]))
                xv = x_eval_gpu[:n_ev].to(device, dtype=torch.float32)
                y_err = (xv @ w_gpu.T) - (xv @ w_eff_gpu.T)
                del w_gpu
                if n_ev <= d_out:
                    _, evecs_s = torch.linalg.eigh(y_err @ y_err.T)
                    U_f = y_err.T @ evecs_s[:, -lr_rank:].flip(1).contiguous()
                    U_f = U_f / U_f.norm(dim=0, keepdim=True).clamp(min=1e-6)
                else:
                    _, evecs = torch.linalg.eigh(y_err.T @ y_err)
                    U_f = evecs[:, -lr_rank:].flip(1).contiguous()
                reg = max(0.02 * float((xv * xv).sum().item()) / float(max(d_in, 1)), 1e-6)
                use_dual = n_ev < d_in
                gram_r = (xv @ xv.T) if use_dual else (xv.T @ xv)
                gram_r.diagonal().add_(reg)

                def _solve_ridge(rhs_p: torch.Tensor) -> torch.Tensor:
                    if use_dual:
                        return xv.T @ torch.linalg.solve(gram_r, rhs_p)
                    return torch.linalg.solve(gram_r, xv.T @ rhs_p)

                V_f = _solve_ridge(y_err @ U_f).T.contiguous()
                eye_lr = 1e-4 * torch.eye(lr_rank, device=device)
                for _ in range(2):
                    xv_v = xv @ V_f.T
                    U_f = torch.linalg.solve(xv_v.T @ xv_v + eye_lr, xv_v.T @ y_err).T
                    U_f = U_f / U_f.norm(dim=0, keepdim=True).clamp(min=1e-6)
                    V_f = _solve_ridge(y_err @ U_f).T.contiguous()
                del y_err, gram_r, xv, eye_lr
                su = (U_f.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
                sv = (V_f.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
                u_i8 = torch.clamp(torch.round(U_f / su.unsqueeze(1)), -127, 127).to(torch.int8)
                v_i8 = torch.clamp(torch.round(V_f / sv.unsqueeze(1)), -127, 127).to(torch.int8)
                u_sc = u_i8.float() * su.unsqueeze(1)
                v_sc = v_i8.float() * sv.unsqueeze(1)
                if torch.isfinite(u_sc).all() and torch.isfinite(v_sc).all():
                    w_eff_gpu.addmm_(u_sc, v_sc)
                    low_rank = (
                        su.cpu().numpy().astype("<f4"),
                        sv.cpu().numpy().astype("<f4"),
                        u_i8.cpu().numpy().astype(np.int8),
                        v_i8.cpu().numpy().astype(np.int8),
                    )
                del u_sc, v_sc, u_i8, v_i8, su, sv, U_f, V_f
        else:
            del w_gpu

        if to_rowmajor:
            pbytes = v70.transpose_t16c_1b_col_to_rm(pbytes, d_out, d_in, bsize)
        out_t = w_eff_gpu.to(device=ret_dev, dtype=torch.float16)
        del w_eff_gpu
        torch.cuda.empty_cache()
        return pbytes, inv_s_np, low_rank, out_t

    H8_GPU = torch.from_numpy(c8d.H8_NP).to(device, dtype=torch.float32)

    def eval_c8d_gpu(
        x_gpu: torch.Tensor, cents: np.ndarray, experts: List[tuple], top_k: int = 2, temp: float = 0.5, gscale: float = 1.0
    ) -> torch.Tensor:
        xf = x_gpu.float()
        n_t, d_i = xf.shape
        d_o = experts[0][0].shape[0]
        kb = d_i // 8
        u_bar = xf[:, : kb * 8].reshape(n_t, kb, 8).mean(dim=1) @ H8_GPU
        g_hat = u_bar / u_bar.norm(dim=1, keepdim=True).clamp(min=1e-9)
        logits = (g_hat @ torch.from_numpy(cents).to(device, dtype=torch.float32).T) / temp
        topv, topi = torch.topk(logits, top_k, dim=-1)
        ws = F.softmax(topv, dim=-1) * gscale
        out = torch.zeros((n_t, d_o), device=device, dtype=torch.float32)
        for e_idx, (su, sv, ui, vi) in enumerate(experts):
            mask_e = topi == e_idx
            if not mask_e.any():
                continue
            w_e = (ws * mask_e.float()).sum(dim=-1, keepdim=True)
            nz = w_e.squeeze(1) > 0
            vf = torch.from_numpy(vi).to(device, dtype=torch.float32) * torch.from_numpy(sv).to(device, dtype=torch.float32).unsqueeze(1)
            uf = torch.from_numpy(ui).to(device, dtype=torch.float32) * torch.from_numpy(su).to(device, dtype=torch.float32).unsqueeze(1)
            out[nz] += w_e[nz] * ((xf[nz] @ vf.T) @ uf.T)
        return out

    def save_progress(state: dict) -> None:
        used_gb, tot_gb = v70.host_ram_gb()
        va, vr, vt = _gpu_vram_stats_gb(device)
        payload = {
            **state,
            "host_ram_used_gb": used_gb,
            "host_ram_total_gb": tot_gb,
            "vram_alloc_gb": va,
            "vram_reserved_gb": vr,
            "vram_total_gb": vt,
            "elapsed_s": round(time.perf_counter() - t0_all, 2),
        }
        tmp_p = PROGRESS_JSON.with_suffix(".tmp")
        tmp_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_p, PROGRESS_JSON)

    # Stage 1: Model & Corpus
    print("\n[Stage 1] Downloading/loading google/gemma-4-31B-it-qat-w4a16-ct...", flush=True)
    model_dir = Path(snapshot_download("google/gemma-4-31B-it-qat-w4a16-ct", local_dir="/tmp/gemma4_31b", ignore_patterns=["*.bin", "*.pt"]))
    sf_path = model_dir / "model.safetensors"
    tok_path = model_dir / "tokenizer.json"
    cfg_full = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tk = Tokenizer.from_file(str(tok_path))

    tc = cfg_full["text_config"]
    N_LAYERS = int(tc["num_hidden_layers"])
    D_MOD = int(tc["hidden_size"])
    EPS = float(tc["rms_norm_eps"])
    ESCALE = math.sqrt(float(D_MOD))
    NH = int(tc["num_attention_heads"])
    NKV_S = int(tc["num_key_value_heads"])
    NKV_F = int(tc["num_global_key_value_heads"])
    HD_S = int(tc["head_dim"])
    HD_F = int(tc["global_head_dim"])
    K_EQ_V = bool(tc.get("attention_k_eq_v", False))
    LAYER_TYPES = tc["layer_types"]
    RP = tc["rope_parameters"]
    THETA_G, PART_G = float(RP["full_attention"]["rope_theta"]), float(RP["full_attention"].get("partial_rotary_factor", 1.0))
    THETA_L, PART_L = float(RP["sliding_attention"]["rope_theta"]), float(RP["sliding_attention"].get("partial_rotary_factor", 1.0))
    SOFTCAP = float(tc.get("final_logit_softcapping") or 30.0)

    bgt = compute_v7_5_budget_bytes(
        vocab_size=int(tc["vocab_size"]), d_mod=D_MOD, n_layers=N_LAYERS,
        nh=NH, nkv_s=NKV_S, nkv_f=NKV_F, hd_s=HD_S, hd_f=HD_F, ff=21504,
    )
    print(
        f"[Budget Guard] V7.5 Base={bgt['base_total_bytes']/1024**3:.3f} GiB (<=5.45) "
        f"| Shared={bgt['lod_shared_bytes']/1024**3:.3f} GiB | Top2 Arena={bgt['lod_topk_arena_bytes']/1024**3:.3f} GiB "
        f"| Active Pi4 RAM={bgt['active_text_ram_bytes']/1024**3:.3f} GiB (<=7.10) "
        f"| SSD Backed={bgt['total_ssd_backed_bytes']/1024**3:.3f} GiB",
        flush=True,
    )
    assert bgt["attn_1b_cnt"] == 0 and bgt["base_total_bytes"] <= MAX_BASE_T16C_BYTES
    assert bgt["active_text_ram_bytes"] <= MAX_ACTIVE_TEXT_RAM_BYTES

    def is_clean_row(txt: str) -> bool:
        low = txt.lower()
        return not any(b in low for b in ALL_BLACKLIST)

    clean_turns: List[Tuple[str, str, str]] = []
    for row in load_dataset("openai/gsm8k", "main", split="train"):
        q, a = row["question"].strip(), row["answer"].strip()
        if is_clean_row(q + " " + a):
            clean_turns.append(("math", q, a))
    for row in load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train"):
        p = (row.get("instruction", "") + ("\n" + row.get("input", "") if row.get("input", "") else "")).strip()
        a = row.get("output", "").strip()
        if p and a and is_clean_row(p + " " + a):
            clean_turns.append(("code", p, a))
    for row in load_dataset("yahma/alpaca-cleaned", split="train"):
        p = (row.get("instruction", "") + ("\n" + row.get("input", "") if row.get("input", "") else "")).strip()
        a = row.get("output", "").strip()
        if p and a and is_clean_row(p + " " + a):
            clean_turns.append(("alpaca", p, a))
    wiki_buf = [row["text"].strip() for row in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train") if len(row["text"].strip()) > 60 and not row["text"].strip().startswith("=") and is_clean_row(row["text"])]
    for idx in range(0, len(wiki_buf) - 1, 2):
        clean_turns.append(("wiki", "Summarize and continue accurately:\n" + wiki_buf[idx][:320], wiki_buf[idx + 1][:500]))

    N_TEXT_SEQ, N_VLM_SEQ, SEQ_LEN = 608, 160, 352
    N_SEQ = N_TEXT_SEQ + N_VLM_SEQ
    GEMMA4_BOI_ID, GEMMA4_IMAGE_ID, GEMMA4_EOI_ID = 255999, 258880, 258882
    by_dom = defaultdict(list)
    for idx in np.random.default_rng(42).permutation(len(clean_turns)):
        dn, qt, at = clean_turns[idx]
        by_dom[dn].append((qt, at))
    dom_names = ["math", "code", "alpaca", "wiki"]
    ptrs = {d: 0 for d in dom_names}
    seq_ids_list, seq_nxt_list, seq_ans_list, seq_vlm_ans_list, seq_dom_list = [], [], [], [], []
    t_cursor = 0
    for _ in range(N_TEXT_SEQ):
        dn = dom_names[t_cursor % len(dom_names)]
        seq_dom_list.append(dn)
        cur_toks, cur_ans = [2], [False]
        while len(cur_toks) < SEQ_LEN + 1:
            qt, at = by_dom[dn][ptrs[dn] % len(by_dom[dn])]
            ptrs[dn] += 1; t_cursor += 1
            ptok = list(tk.encode(vc.format_gemma4_turn(qt, None), add_special_tokens=False).ids)
            rtok = list(tk.encode(f"{at}<turn|>\n", add_special_tokens=False).ids)
            if not ptok or not rtok:
                continue
            cur_toks.extend(ptok); cur_ans.extend([False] * len(ptok))
            cur_toks.extend(rtok); cur_ans.extend([True] * len(rtok))
            dn = dom_names[t_cursor % len(dom_names)]
        cur_toks, cur_ans = cur_toks[: SEQ_LEN + 1], cur_ans[: SEQ_LEN + 1]
        seq_ids_list.append(cur_toks[:-1]); seq_nxt_list.append(cur_toks[1:])
        seq_ans_list.append(cur_ans[1:]); seq_vlm_ans_list.append([False] * SEQ_LEN)

    # Stage 2: 3-Bit Vision Calibration
    VIS_CKPT_PATH = CKPT_DIR / "stage2_vision.pt"
    VIS_META_PATH = CKPT_DIR / "stage2_meta.pt"
    VLM_SOFT_CACHE = CKPT_DIR / "stage2_vlm_soft.pt"

    def _build_stage2_vision() -> dict:
        print("\n[Stage 2] 3-Bit Vision Calibration (V00..V26 + embed_vision)...", flush=True)
        cfg_auto = AutoConfig.from_pretrained(str(model_dir))
        proc = Gemma4ImageProcessor.from_pretrained(str(model_dir))

        def make_optical_va_pil(idx: int) -> Tuple[Image.Image, str]:
            im = Image.new("RGB", (672, 672), (248, 248, 248))
            dr = ImageDraw.Draw(im)
            dr.text((18, 18), f"GEMMA4 31B V7.5 LM=60 VIS=27 FRAME={idx:03d}", fill=(12, 18, 28))
            dr.text((18, 42), "OPTICAL VA SPATIAL DISTRICTS C8DB-H8", fill=(12, 18, 28))
            rng_v = np.random.default_rng(1000 + idx)
            for b in range(60):
                bh = int(rng_v.integers(25, 118))
                col = (35, 115, 210) if b >= 58 else ((30, 165, 95) if b % 5 != 0 else (210, 65, 45))
                dr.rectangle([24 + b * 10, 550 - bh, 31 + b * 10, 550], fill=col)
            cap = f"An optical diagnostic chart labeled GEMMA4 31B V7.5 LM=60 VIS=27 FRAME={idx:03d} displaying 60 vertical green, red, and blue bar segments on a light background."
            return im, cap

        vis_va_pairs = [make_optical_va_pil(i) for i in range(32)]
        vis_coco_pairs, heldout_pairs = [], []
        for row_idx, row in enumerate(load_dataset("lmms-lab/COCO-Caption2017", split="val", streaming=True)):
            im, ans_raw = row.get("image"), row.get("answer")
            if im is None or not ans_raw:
                continue
            cap = (ans_raw[0] if isinstance(ans_raw, list) else str(ans_raw)).strip()
            if not cap:
                continue
            if row_idx < 4:
                heldout_pairs.append((im.convert("RGB"), cap)); continue
            if row_idx < 64 or not is_clean_row(cap):
                continue
            vis_coco_pairs.append((im.convert("RGB"), cap))
            if len(vis_coco_pairs) >= 128:
                break
        assert len(vis_coco_pairs) == 128 and len(heldout_pairs) == 4
        vis_pairs = []
        for i in range(32):
            vis_pairs.append(vis_va_pairs[i])
            vis_pairs.extend(vis_coco_pairs[i * 4 : (i + 1) * 4])
        vis_images, vis_captions = [p[0] for p in vis_pairs], [p[1] for p in vis_pairs]

        with safe_open(sf_path, framework="pt", device="cpu") as f:
            vis_sd = {k[len("model.vision_tower."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.vision_tower.")}
            embv_sd = {k[len("model.embed_vision."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.embed_vision.")}
            std_bias_f32 = vis_sd["std_bias"].float().numpy().astype("<f4")
            std_scale_f32 = vis_sd["std_scale"].float().numpy().astype("<f4")
            pos_tbl_f32 = vis_sd["patch_embedder.position_embedding_table"].float()

        vis_teacher = Gemma4VisionModel(cfg_auto.vision_config).to(compute_dtype).to(device).eval()
        vis_teacher.load_state_dict({k: v.to(device, dtype=compute_dtype) for k, v in vis_sd.items()}, strict=True)
        mm_teacher = Gemma4MultimodalEmbedder(cfg_auto.vision_config, cfg_auto.text_config).to(compute_dtype).to(device).eval()
        mm_teacher.load_state_dict({k: v.to(device, dtype=compute_dtype) for k, v in embv_sd.items()}, strict=True)

        vis_hess: Dict[str, torch.Tensor] = {}
        vis_xsq: Dict[str, torch.Tensor] = {}
        vis_cnt: Dict[str, int] = defaultdict(int)
        vis_x_eval: Dict[str, torch.Tensor] = {}

        def reg_vis_hook(mod, key: str):
            def _hk(_m, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                if key not in vis_hess:
                    d = x.shape[1]
                    vis_hess[key] = torch.zeros((d, d), device=device, dtype=torch.float32)
                    vis_xsq[key] = torch.zeros((d,), device=device, dtype=torch.float32)
                    vis_x_eval[key] = x[:1024].cpu().clone()
                accum_hess(vis_hess[key], vis_xsq[key], x)
                vis_cnt[key] += x.shape[0]
            return mod.register_forward_hook(_hk)

        hooks = [reg_vis_hook(vis_teacher.patch_embedder.input_proj, "model.vision_tower.patch_embedder.input_proj.weight")]
        for lv, lvm in enumerate(vis_teacher.encoder.layers):
            pv = f"model.vision_tower.encoder.layers.{lv}"
            for attr, kn in [("self_attn.q_proj", "self_attn.q_proj"), ("self_attn.k_proj", "self_attn.k_proj"), ("self_attn.v_proj", "self_attn.v_proj"), ("self_attn.o_proj", "self_attn.o_proj"), ("mlp.gate_proj", "mlp.gate_proj"), ("mlp.up_proj", "mlp.up_proj"), ("mlp.down_proj", "mlp.down_proj")]:
                obj = lvm
                for part in attr.split("."):
                    obj = getattr(obj, part)
                hooks.append(reg_vis_hook(obj, f"{pv}.{kn}.linear.weight"))
        hooks.append(reg_vis_hook(mm_teacher.embedding_projection, "model.embed_vision.embedding_projection.weight"))

        vis_teach_soft, vis_eval_pvs, vis_eval_poss = [], [], []
        with torch.no_grad():
            for im in vis_images:
                pin = proc(images=[im], return_tensors="pt")
                pv, pos = pin["pixel_values"].to(device, dtype=compute_dtype), pin["image_position_ids"].to(device)
                st = mm_teacher(vis_teacher(pixel_values=pv, pixel_position_ids=pos).last_hidden_state).reshape(-1, D_MOD)
                vis_eval_pvs.append(pv.cpu()); vis_eval_poss.append(pos.cpu()); vis_teach_soft.append(st.to(torch.float16).cpu())
        for h in hooks:
            h.remove()
        heldout_pvs, heldout_poss, heldout_teach_soft = [], [], []
        with torch.no_grad():
            for im_h, _ in heldout_pairs:
                pin = proc(images=[im_h], return_tensors="pt")
                pv, pos = pin["pixel_values"].to(device, dtype=compute_dtype), pin["image_position_ids"].to(device)
                st = mm_teacher(vis_teacher(pixel_values=pv, pixel_position_ids=pos).last_hidden_state).reshape(-1, D_MOD)
                heldout_pvs.append(pv.cpu()); heldout_poss.append(pos.cpu()); heldout_teach_soft.append(st.to(torch.float16).cpu())
        del vis_teacher, mm_teacher
        torch.cuda.empty_cache()

        vision_records, vis_deq = [], {}
        vision_norms = {"model.vision_tower.std_bias": std_bias_f32, "model.vision_tower.std_scale": std_scale_f32}

        def pack_vis(name: str, wt: torch.Tensor, lr_r: int, pad_in: int = 0):
            do, di = wt.shape
            hu = vis_hess.pop(name, None)
            xr = torch.sqrt((vis_xsq.pop(name) / max(vis_cnt[name], 1)).clamp(min=1e-8)) if name in vis_xsq else None
            xev = vis_x_eval.pop(name).to(device) if name in vis_x_eval else None
            if pad_in > di:
                wp = torch.zeros((do, pad_in), dtype=torch.float32); wp[:, :di] = wt.float()
                if hu is not None:
                    hp = torch.zeros((pad_in, pad_in), device=device); hp[:di, :di] = hu; hp.diagonal()[di:].fill_(1.0); hu = hp
                if xr is not None:
                    xp = torch.ones((pad_in,), device=device); xp[:di] = xr; xr = xp
                if xev is not None:
                    xe = torch.zeros((xev.shape[0], pad_in), device=device); xe[:, :di] = xev; xev = xe
                die = pad_in
            else:
                wp, die = wt.float(), di
            pb, invs, lr_t, weff = pack_matrix_gpu_gptq(wp, [hu] if hu is not None else None, xr, mode=4, bsize=64, lr_rank=lr_r, x_eval_gpu=xev)
            del hu, xr, xev
            vision_records.append((name, do, die, 64, 3, False, True, pb, invs, lr_t))
            vis_deq[name] = weff[:, :di].to(compute_dtype)

        for ax in (0, 1):
            nm_p = f"model.vision_tower.patch_embedder.position_embedding_table.{ax}"
            pb, invs, lrt, we = pack_matrix_gpu_gptq(pos_tbl_f32[ax], None, None, mode=4, bsize=64, lr_rank=0)
            vision_records.append((nm_p, pos_tbl_f32.shape[1], pos_tbl_f32.shape[2], 64, 3, False, True, pb, invs, lrt))
            vis_deq[nm_p] = we.to(compute_dtype)
        pack_vis("model.vision_tower.patch_embedder.input_proj.weight", vis_sd["patch_embedder.input_proj.weight"], 16)
        for lv in range(27):
            fp_v = f"model.vision_tower.encoder.layers.{lv}"
            for nk in ["input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]:
                vision_norms[f"{fp_v}.{nk}"] = vis_sd[f"encoder.layers.{lv}.{nk}"].float().numpy().astype("<f4")
            vrk = 8 if lv in (0, 1, 24, 25, 26) else 0
            for pj in ["self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight", "self_attn.o_proj.linear.weight", "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight"]:
                pack_vis(f"{fp_v}.{pj}", vis_sd[f"encoder.layers.{lv}.{pj}"], vrk)
            pack_vis(f"{fp_v}.mlp.down_proj.linear.weight", vis_sd[f"encoder.layers.{lv}.mlp.down_proj.linear.weight"], vrk, pad_in=4352)
        pack_vis("model.embed_vision.embedding_projection.weight", embv_sd["embedding_projection.weight"], 32)
        v_bytes = sum(len(r[7]) + (len(r[8]) * 4 if r[8] is not None else 0) + ((r[9][0].nbytes + r[9][1].nbytes + r[9][2].nbytes + r[9][3].nbytes) if r[9] is not None else 0) + 80 for r in vision_records)

        vis_stud = Gemma4VisionModel(cfg_auto.vision_config).to(compute_dtype)
        vis_stud.load_state_dict({k: v.to(dtype=compute_dtype) for k, v in vis_sd.items()}, strict=True)
        with torch.no_grad():
            vis_stud.patch_embedder.input_proj.weight.copy_(vis_deq["model.vision_tower.patch_embedder.input_proj.weight"])
            vis_stud.patch_embedder.position_embedding_table[0].copy_(vis_deq["model.vision_tower.patch_embedder.position_embedding_table.0"])
            vis_stud.patch_embedder.position_embedding_table[1].copy_(vis_deq["model.vision_tower.patch_embedder.position_embedding_table.1"])
            for lv, lvm in enumerate(vis_stud.encoder.layers):
                fp_v = f"model.vision_tower.encoder.layers.{lv}"
                for sub, ms in [("self_attn.q_proj", lvm.self_attn.q_proj), ("self_attn.k_proj", lvm.self_attn.k_proj), ("self_attn.v_proj", lvm.self_attn.v_proj), ("self_attn.o_proj", lvm.self_attn.o_proj), ("mlp.gate_proj", lvm.mlp.gate_proj), ("mlp.up_proj", lvm.mlp.up_proj), ("mlp.down_proj", lvm.mlp.down_proj)]:
                    ms.linear.weight.copy_(vis_deq[f"{fp_v}.{sub}.linear.weight"])
        vis_stud = vis_stud.to(device).eval()
        mm_stud = Gemma4MultimodalEmbedder(cfg_auto.vision_config, cfg_auto.text_config).to(compute_dtype)
        mm_stud.load_state_dict({k: v.to(dtype=compute_dtype) for k, v in embv_sd.items()}, strict=True)
        with torch.no_grad():
            mm_stud.embedding_projection.weight.copy_(vis_deq["model.embed_vision.embedding_projection.weight"])
        mm_stud = mm_stud.to(device).eval()

        vis_pre, vis_base, h_pre, h_base = [], [], [], []
        with torch.no_grad():
            for pv, pos in zip(vis_eval_pvs, vis_eval_poss):
                hn = mm_stud.embedding_pre_projection_norm(vis_stud(pixel_values=pv.to(device), pixel_position_ids=pos.to(device)).last_hidden_state)
                vis_pre.append(hn.reshape(-1, 1152).float().cpu()); vis_base.append(mm_stud.embedding_projection(hn).reshape(-1, D_MOD).float().cpu())
            for pv, pos in zip(heldout_pvs, heldout_poss):
                hn = mm_stud.embedding_pre_projection_norm(vis_stud(pixel_values=pv.to(device), pixel_position_ids=pos.to(device)).last_hidden_state)
                h_pre.append(hn.reshape(-1, 1152).float().cpu()); h_base.append(mm_stud.embedding_projection(hn).reshape(-1, D_MOD).float().cpu())
        del vis_stud, mm_stud, vis_sd, embv_sd, vis_deq, vis_eval_pvs, vis_eval_poss, heldout_pvs, heldout_poss
        torch.cuda.empty_cache()

        vx = torch.cat(vis_pre, 0); vyt = torch.cat([t.float() for t in vis_teach_soft], 0); vys = torch.cat(vis_base, 0)
        nv_tr = (vx.shape[0] * 4) // 5
        vx_tr, vyt_tr, vys_tr = vx[:nv_tr].to(device), vyt[:nv_tr].to(device), vys[:nv_tr].to(device)
        vx_va, vyt_va, vys_va = vx[nv_tr:].to(device), vyt[nv_tr:].to(device), vys[nv_tr:].to(device)
        v_cos0 = float(F.cosine_similarity(vys_va, vyt_va, dim=-1).mean().item())
        c8_vis = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=1152, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(device)
        opt_v = torch.optim.AdamW(c8_vis.parameters(), lr=1.5e-3, weight_decay=5e-2)
        best_vc, best_vsd = v_cos0, None
        for st in range(300):
            opt_v.zero_grad()
            ib = torch.randint(0, vx_tr.shape[0], (min(1024, vx_tr.shape[0]),), device=device)
            dv, _, lb = c8_vis(vx_tr[ib])
            pred = vys_tr[ib] + dv
            loss_v = F.mse_loss(pred, vyt_tr[ib]) + 0.5 * (1.0 - F.cosine_similarity(pred, vyt_tr[ib], dim=-1).mean()) + 0.005 * lb
            loss_v.backward(); torch.nn.utils.clip_grad_norm_(c8_vis.parameters(), 1.0); opt_v.step()
            if (st + 1) % 50 == 0:
                with torch.no_grad():
                    cv = float(F.cosine_similarity(vys_va + c8_vis(vx_va)[0], vyt_va, dim=-1).mean().item())
                    if cv > best_vc:
                        best_vc, best_vsd = cv, {k: v.detach().cpu().clone() for k, v in c8_vis.state_dict().items()}
        c8_cpu = c8_vis.cpu()
        v_keep, c_v, e_v, v_cos1 = False, None, None, v_cos0
        if best_vsd is not None:
            c8_cpu.load_state_dict(best_vsd)
            c_v, e_v = c8_cpu.export_quantized_arrays()
            v_cos1 = float(F.cosine_similarity(vys_va + eval_c8d_gpu(vx_va, c_v, e_v), vyt_va, dim=-1).mean().item())
            v_keep = v_cos1 > v_cos0 + 0.001
        with torch.no_grad():
            v_stud_s = [(b.to(device) + (eval_c8d_gpu(p.to(device), c_v, e_v) if v_keep else 0.0)).to(torch.float16).cpu() for p, b in zip(vis_pre, vis_base)]
            h_stud_s = [(b.to(device) + (eval_c8d_gpu(p.to(device), c_v, e_v) if v_keep else 0.0)).to(torch.float16).cpu() for p, b in zip(h_pre, h_base)]
        del vx, vyt, vys, vx_tr, vyt_tr, vys_tr, vx_va, vyt_va, vys_va, c8_vis, c8_cpu, vis_pre, vis_base, h_pre, h_base
        torch.cuda.empty_cache()

        v70._atomic_torch_save({"vision_records": vision_records}, VIS_CKPT_PATH)
        v70._atomic_torch_save({"vis_teach_soft": vis_teach_soft, "vis_stud_soft": v_stud_s, "vis_captions": vis_captions, "heldout_teach_soft": heldout_teach_soft, "heldout_stud_soft": h_stud_s, "heldout_captions": [p[1] for p in heldout_pairs]}, VLM_SOFT_CACHE)
        meta = {"vis_total_bytes": int(v_bytes), "n_vis_mats": len(vision_records), "vision_norms": vision_norms, "vis_keep_c8d": v_keep, "cents_vis": c_v, "experts_vis": e_v, "v_cos0": v_cos0, "v_cos1": v_cos1}
        v70._atomic_torch_save(meta, VIS_META_PATH)
        return meta

    if VIS_META_PATH.is_file() and VIS_CKPT_PATH.is_file() and VLM_SOFT_CACHE.is_file():
        vm = torch.load(VIS_META_PATH, map_location="cpu", weights_only=False)
    else:
        vm = _build_stage2_vision()
    vision_norms, vis_total_bytes, n_vis_mats = vm["vision_norms"], int(vm["vis_total_bytes"]), int(vm["n_vis_mats"])
    v_cos0, v_cos1, vis_keep_c8d, cents_vis, experts_vis = float(vm["v_cos0"]), float(vm["v_cos1"]), bool(vm["vis_keep_c8d"]), vm["cents_vis"], vm["experts_vis"]
    del vm

    vsc = torch.load(VLM_SOFT_CACHE, map_location="cpu", weights_only=False)
    vlm_teach_soft, vlm_stud_soft, vlm_captions = vsc["vis_teach_soft"], vsc["vis_stud_soft"], vsc["vis_captions"]
    heldout_teach_soft, heldout_stud_soft, heldout_captions = vsc["heldout_teach_soft"], vsc["heldout_stud_soft"], vsc["heldout_captions"]
    del vsc

    vlm_spans: Dict[int, Tuple[int, int]] = {}
    tok_user_open = list(tk.encode("<|turn>user\n", add_special_tokens=False).ids)
    tok_model_open = list(tk.encode("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", add_special_tokens=False).ids)
    vqa_q = ["Describe this image concisely in one sentence.", "What is shown in this image?", "Provide a brief caption for this image.", "Explain what you see in this picture."]
    for vi in range(N_VLM_SEQ):
        ns = int(vlm_teach_soft[vi].shape[0])
        qv = list(tk.encode(vqa_q[vi % 4], add_special_tokens=False).ids)
        av = list(tk.encode(f"{vlm_captions[vi]}<turn|>\n", add_special_tokens=False).ids)
        ct = [2] + tok_user_open + [GEMMA4_BOI_ID] + [GEMMA4_IMAGE_ID] * ns + [GEMMA4_EOI_ID] + qv + tok_model_open
        s0v = 1 + len(tok_user_open) + 1
        vlm_spans[N_TEXT_SEQ + vi] = (s0v, s0v + ns)
        ca = [False] * len(ct) + [True] * len(av)
        ct = ct + av
        while len(ct) < SEQ_LEN + 1:
            ct.append(0); ca.append(False)
        seq_ids_list.append(ct[:SEQ_LEN]); seq_nxt_list.append(ct[1 : SEQ_LEN + 1])
        seq_ans_list.append(ca[1 : SEQ_LEN + 1]); seq_vlm_ans_list.append(ca[1 : SEQ_LEN + 1])
        seq_dom_list.append("vlm")

    all_seq_ids = np.array(seq_ids_list, dtype=np.int32)
    all_seq_nxt = np.array(seq_nxt_list, dtype=np.int32)
    all_ans_mask = np.array(seq_ans_list, dtype=bool)
    all_vlm_ans_mask = np.array(seq_vlm_ans_list, dtype=bool)

    heldout_spans, heldout_ids_list, heldout_ans_starts, heldout_ans_lens = {}, [], [], []
    for hi, ch in enumerate(heldout_captions):
        ns = int(heldout_teach_soft[hi].shape[0])
        qh = list(tk.encode("Describe this image concisely in one sentence.", add_special_tokens=False).ids)
        ah = list(tk.encode(f"{ch}<turn|>\n", add_special_tokens=False).ids)
        ct = [2] + tok_user_open + [GEMMA4_BOI_ID] + [GEMMA4_IMAGE_ID] * ns + [GEMMA4_EOI_ID] + qh + tok_model_open
        s0h = 1 + len(tok_user_open) + 1
        heldout_spans[hi] = (s0h, s0h + ns)
        ast = len(ct) - 1; ct = ct + ah; alen = min(len(ah), SEQ_LEN - ast)
        while len(ct) < SEQ_LEN + 1:
            ct.append(0)
        heldout_ids_list.append(ct[:SEQ_LEN]); heldout_ans_starts.append(ast); heldout_ans_lens.append(alen)
    heldout_seq_ids = np.array(heldout_ids_list, dtype=np.int32)

    # Pre-build sliding causal+bidirectional attention masks once on GPU!
    def _build_mask_tensor(n_items: int, spans: Dict[int, Tuple[int, int]]) -> Optional[torch.Tensor]:
        if not spans:
            return None
        base_tril = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=device, dtype=torch.bool))
        out = base_tril.unsqueeze(0).expand(n_items, -1, -1).clone()
        for gi, (s0, s1) in spans.items():
            if 0 <= gi < n_items:
                out[gi, s0:s1, s0:s1] = True
        return out.unsqueeze(1)

    vlm_full_mask_gpu = _build_mask_tensor(N_SEQ, vlm_spans)
    heldout_full_mask_gpu = _build_mask_tensor(len(heldout_captions), heldout_spans)

    def make_rope(theta: float, rot_d: int, hd: int, max_len: int = SEQ_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
        p1 = torch.arange(max_len, device=device, dtype=torch.float32)
        hr = rot_d // 2
        inv_f = 1.0 / (theta ** (torch.arange(0, 2 * hr, 2, device=device, dtype=torch.float32) / float(hd)))
        ang = p1[:, None] * inv_f[None, :]
        return torch.cos(ang).to(compute_dtype), torch.sin(ang).to(compute_dtype)

    def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_d: int, hd: int) -> torch.Tensor:
        hr, hh = rot_d // 2, hd // 2
        c, s = cos[None, :, None, :].to(t.dtype), sin[None, :, None, :].to(t.dtype)
        x0, x1 = t[..., :hr], t[..., hh : hh + hr]
        r0, r1 = x0 * c - x1 * s, x0 * s + x1 * c
        if hr == hh:
            return torch.cat([r0, r1], dim=-1)
        return torch.cat([r0, t[..., hr:hh], r1, t[..., hh + hr :]], dim=-1)

    def rmsn_gpu(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return ((y * w.float()) if w is not None else y).to(compute_dtype)

    # Stage 3: Full 60-Layer Spatial Districts + GTA LOD Residual Split + CRTR + MoD (q20)
    STAGE3_CKPT = CKPT_DIR / "stage3_layer_ckpt.pt"
    CRTR_CKPT = CKPT_DIR / "crtr_layers.pt"
    D_ROUTER_CKPT = CKPT_DIR / "district_routers.pt"
    CKPT_X_TEACH = CKPT_DIR / "v7_5_x_teach.pt"
    CKPT_X_STUD = CKPT_DIR / "v7_5_x_stud.pt"

    _tr_lin = np.linspace(0, (N_SEQ - 48) * SEQ_LEN - 1, 16384, dtype=int)
    val_set = set(np.concatenate([np.linspace(0, N_TEXT_SEQ - 1, 32, dtype=int), np.linspace(N_TEXT_SEQ, N_SEQ - 1, 16, dtype=int)]).tolist())
    tr_seq_idx = np.array([i for i in range(N_SEQ) if i not in val_set], dtype=int)
    va_seq_idx = np.array(sorted(val_set), dtype=int)
    tr_crtr_s, tr_crtr_t = tr_seq_idx[_tr_lin // SEQ_LEN], _tr_lin % SEQ_LEN
    _va_lin = np.linspace(0, len(va_seq_idx) * SEQ_LEN - 1, 4096, dtype=int)
    va_crtr_s, va_crtr_t = va_seq_idx[_va_lin // SEQ_LEN], _va_lin % SEQ_LEN

    MB_FWD = int(hw_cfg["batch_seq"])
    RES_GPU = bool(hw_cfg["resident_gpu"])
    _PIN_CPU = (not RES_GPU) and torch.cuda.is_available()

    def _alloc_act_store(shape: Tuple[int, ...], dev: torch.device) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.float16, device=dev, pin_memory=(dev.type == "cpu" and _PIN_CPU))

    def _to_store(t_gpu: torch.Tensor) -> torch.Tensor:
        return t_gpu if RES_GPU else t_gpu.cpu()

    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        lm_norms: Dict[str, np.ndarray] = {}
        for _l in range(N_LAYERS):
            _p = f"model.language_model.layers.{_l}."
            for _nk in ("input_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar"):
                _v = sf.get_tensor(_p + _nk).float().numpy().astype("<f4")
                lm_norms[_p + _nk] = np.array([float(_v.reshape(-1)[0])], dtype="<f4") if _nk == "layer_scalar" else _v
        fn_orig_cpu = sf.get_tensor("model.language_model.norm.weight").float()
        all_norms = {**lm_norms, "model.language_model.norm.weight": fn_orig_cpu.numpy().astype("<f4"), **vision_norms}
        n_all_norms = len(all_norms)
        expected_lm_mats = bgt["pred_lm_mats"]
        total_base_mats = expected_lm_mats + n_vis_mats
        expected_lod_mats = bgt["pred_lod_mats_per_shard"]

        def load_layer_from_safetensors(li: int):
            with safe_open(sf_path, framework="pt", device="cpu") as sf_l:
                p_l = f"model.language_model.layers.{li}."
                is_fl = LAYER_TYPES[li] == "full_attention"
                ng_l = {
                    "in": sf_l.get_tensor(p_l + "input_layernorm.weight"),
                    "qn": sf_l.get_tensor(p_l + "self_attn.q_norm.weight"),
                    "kn": sf_l.get_tensor(p_l + "self_attn.k_norm.weight"),
                    "pa": sf_l.get_tensor(p_l + "post_attention_layernorm.weight"),
                    "pf": sf_l.get_tensor(p_l + "pre_feedforward_layernorm.weight"),
                    "pff": sf_l.get_tensor(p_l + "post_feedforward_layernorm.weight"),
                    "lsc": sf_l.get_tensor(p_l + "layer_scalar").float().item(),
                }
                w_l = {}
                for pn in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                    if is_fl and K_EQ_V and pn == "v_proj":
                        continue
                    pfx = f"{p_l}{'self_attn' if pn in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'}.{pn}"
                    wp = sf_l.get_tensor(f"{pfx}.weight_packed").to(device)
                    ws = sf_l.get_tensor(f"{pfx}.weight_scale").to(device, dtype=torch.float32)
                    sh = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
                    w_l[pn] = ((((wp.unsqueeze(-1) >> sh) & 0xF) - 8).to(torch.float32).reshape(wp.shape[0], -1, 32) * ws.unsqueeze(-1)).reshape(wp.shape[0], -1).cpu()
                    del wp, ws
            return ng_l, w_l

        def teacher_layer_step(
            X_store: torch.Tensor,
            w_f16: Dict[str, torch.Tensor],
            ng: Dict[str, torch.Tensor],
            li: int,
            mask_full: Optional[torch.Tensor],
            ret_seq_energy: bool = False,
        ):
            is_f = LAYER_TYPES[li] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cr, sr = make_rope(THETA_G if is_f else THETA_L, rot_d, hd)
            rep = NH // nkv
            lsc = float(ng["lsc"])
            n_items = X_store.shape[0]
            out_store = _alloc_act_store((n_items, SEQ_LEN, D_MOD), X_store.device)
            seq_e = np.empty((n_items, 21504), dtype=np.float32) if ret_seq_energy else None
            with torch.no_grad():
                for b0 in range(0, n_items, MB_FWD):
                    b1 = min(b0 + MB_FWD, n_items)
                    xb = X_store[b0:b1].to(device, dtype=compute_dtype, non_blocking=True)
                    inp = rmsn_gpu(xb, ng["in"])
                    q = F.linear(inp, w_f16["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                    k = F.linear(inp, w_f16["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    v = k.clone() if (is_f and K_EQ_V) else F.linear(inp, w_f16["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    qn = apply_rope(rmsn_gpu(q, ng["qn"]), cr, sr, rot_d, hd).transpose(1, 2)
                    kn = apply_rope(rmsn_gpu(k, ng["kn"]), cr, sr, rot_d, hd).transpose(1, 2)
                    vn = rmsn_gpu(v, None).transpose(1, 2)
                    am = None if (is_f or mask_full is None) else mask_full[b0:b1]
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(rep, 1), vn.repeat_interleave(rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0)
                    del inp, q, k, v, qn, kn, vn
                    xm = xb + rmsn_gpu(F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_f16["o_proj"]), ng["pa"])
                    del xb, ctx
                    hm = rmsn_gpu(xm, ng["pf"])
                    act = F.gelu(F.linear(hm, w_f16["gate_proj"]).float(), approximate="tanh")
                    act.mul_(F.linear(hm, w_f16["up_proj"]).float())
                    del hm
                    if seq_e is not None:
                        seq_e[b0:b1] = act.pow(2).sum(dim=1).cpu().numpy()
                    down_out = rmsn_gpu(F.linear(act.to(compute_dtype), w_f16["down_proj"]), ng["pff"]).float()
                    del act
                    out_store[b0:b1] = ((xm.float() + down_out) * lsc).to(torch.float16).to(out_store.device)
                    del xm, down_out
            return out_store, seq_e

        crtr_layers: Dict[int, dict] = torch.load(CRTR_CKPT, map_location="cpu", weights_only=False) if CRTR_CKPT.is_file() else {}
        district_routers: Dict[int, dict] = torch.load(D_ROUTER_CKPT, map_location="cpu", weights_only=False) if D_ROUTER_CKPT.is_file() else {}

        shard_fps: Dict[str, object] = {}
        shard_pos: Dict[str, int] = {}

        if STAGE3_CKPT.is_file() and OUT_BASE_T16C.is_file() and CKPT_X_TEACH.is_file() and CKPT_X_STUD.is_file():
            lc = torch.load(STAGE3_CKPT, map_location="cpu", weights_only=False)
            start_layer = int(lc["next_layer"])
            base_mats_count = int(lc["base_mats_count"])
            bridge_cache: Dict[str, torch.Tensor] = lc["bridge_cache"]
            H_teach_eval = lc["H_teach_eval"].to(device if RES_GPU else "cpu", dtype=torch.float16)
            H_stud_eval = lc["H_stud_eval"].to(device if RES_GPU else "cpu", dtype=torch.float16)
            layer_cos_log: List[dict] = lc["layer_cos_log"]
            district_manifest: List[dict] = lc["district_manifest"]
            shard_pos_ckpt: Dict[str, int] = lc["shard_pos"]
            lod_mats_written = int(lc["lod_mats_written"])
            X_teach_store = torch.load(CKPT_X_TEACH, map_location="cpu", weights_only=True).to(device if RES_GPU else "cpu")
            X_stud_store = torch.load(CKPT_X_STUD, map_location="cpu", weights_only=True).to(device if RES_GPU else "cpu")
            del lc

            if OUT_BASE_T16C.stat().st_size < shard_pos_ckpt["base"]:
                raise RuntimeError(f"FAIL-CLOSED Resume: {OUT_BASE_T16C} size < ckpt pos {shard_pos_ckpt['base']}")
            v70.verify_t16c_integrity(OUT_BASE_T16C, expected_mats=base_mats_count, expected_end_pos=shard_pos_ckpt["base"])
            for sn, sp_f in shard_map.items():
                if not sp_f.is_file() or sp_f.stat().st_size < shard_pos_ckpt[sn]:
                    raise RuntimeError(f"FAIL-CLOSED Resume: {sp_f} truncated below {shard_pos_ckpt[sn]}")
                if lod_mats_written > 0:
                    v70.verify_t16c_integrity(sp_f, expected_mats=lod_mats_written, expected_end_pos=shard_pos_ckpt[sn])
            f_base = open(OUT_BASE_T16C, "r+b"); f_base.seek(shard_pos_ckpt["base"]); f_base.truncate(shard_pos_ckpt["base"])
            shard_fps["base"] = f_base; shard_pos["base"] = shard_pos_ckpt["base"]
            for sn, sp_f in shard_map.items():
                fs_l = open(sp_f, "r+b"); fs_l.seek(shard_pos_ckpt[sn]); fs_l.truncate(shard_pos_ckpt[sn])
                shard_fps[sn] = fs_l; shard_pos[sn] = shard_pos_ckpt[sn]
            print(f"[Resume Stage 3] L{start_layer:02d} (base_mats={base_mats_count}, lod_mats={lod_mats_written})", flush=True)
        else:
            start_layer, base_mats_count, bridge_cache, layer_cos_log, district_manifest, lod_mats_written = 0, 1, {}, [], [], 0
            emb_bf16 = sf.get_tensor("model.language_model.embed_tokens.weight")
            ids_t, eids_t = torch.from_numpy(all_seq_ids).long(), torch.from_numpy(heldout_seq_ids).long()
            X_teach_store = (emb_bf16[ids_t].float() * ESCALE).to(torch.float16).to(device if RES_GPU else "cpu")
            H_teach_eval = (emb_bf16[eids_t].float() * ESCALE).to(torch.float16).to(device if RES_GPU else "cpu")
            for vi in range(N_VLM_SEQ):
                s0, s1 = vlm_spans[N_TEXT_SEQ + vi]; X_teach_store[N_TEXT_SEQ + vi, s0:s1] = vlm_teach_soft[vi].to(X_teach_store.device)
            for hi in range(len(heldout_captions)):
                s0, s1 = heldout_spans[hi]; H_teach_eval[hi, s0:s1] = heldout_teach_soft[hi].to(H_teach_eval.device)
            pb_e, inv_e, lr_e, emb_deq = pack_matrix_gpu_gptq(emb_bf16, None, None, mode=4, bsize=256, lr_rank=0, out_device=torch.device("cpu"))
            X_stud_store = (emb_deq[ids_t] * ESCALE).to(torch.float16).to(device if RES_GPU else "cpu")
            H_stud_eval = (emb_deq[eids_t] * ESCALE).to(torch.float16).to(device if RES_GPU else "cpu")
            for vi in range(N_VLM_SEQ):
                s0, s1 = vlm_spans[N_TEXT_SEQ + vi]; X_stud_store[N_TEXT_SEQ + vi, s0:s1] = vlm_stud_soft[vi].to(X_stud_store.device)
            for hi in range(len(heldout_captions)):
                s0, s1 = heldout_spans[hi]; H_stud_eval[hi, s0:s1] = heldout_stud_soft[hi].to(H_stud_eval.device)
            del emb_bf16, emb_deq, ids_t, eids_t

            f_base = open(OUT_BASE_T16C, "wb")
            f_base.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, n_all_norms, total_base_mats, 0))
            b_pos = 24
            for _nm, _vc_a in all_norms.items():
                _nb = _nm.encode("utf-8")
                _vf = np.ascontiguousarray(_vc_a, dtype="<f4").ravel()
                f_base.write(struct.pack("<H", len(_nb))); f_base.write(_nb); f_base.write(struct.pack("<I", _vf.size)); f_base.write(_vf.tobytes())
                b_pos += 2 + len(_nb) + 4 + _vf.size * 4
            b_pos = p3._write_mat_record(f_base, b_pos, "model.language_model.embed_tokens.weight", int(tc["vocab_size"]), D_MOD, 256, 3, False, True, pb_e, inv_e, lr_e)
            f_base.flush()
            shard_fps["base"] = f_base; shard_pos["base"] = b_pos
            del pb_e, inv_e, lr_e

            for sn, sp_f in shard_map.items():
                fs_l = open(sp_f, "wb")
                fs_l.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, 0, expected_lod_mats, 0))
                fs_l.flush()
                shard_fps[sn] = fs_l; shard_pos[sn] = 24

        del vlm_teach_soft, vlm_stud_soft, heldout_teach_soft, heldout_stud_soft
        gc.collect()
        torch.cuda.empty_cache()

        io_pool = ThreadPoolExecutor(max_workers=2)
        prefetch_fut = io_pool.submit(load_layer_from_safetensors, start_layer) if start_layer < N_LAYERS else None

        try:
            for l_idx in range(start_layer, N_LAYERS):
                tl = time.perf_counter()
                p = f"model.language_model.layers.{l_idx}."
                is_f = LAYER_TYPES[l_idx] == "full_attention"
                nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
                rot_d = int(round(hd * (PART_G if is_f else PART_L)))
                cr_l, sr_l = make_rope(THETA_G if is_f else THETA_L, rot_d, hd)
                rep = NH // nkv

                assert prefetch_fut is not None
                ng_cpu, w_teach = prefetch_fut.result()
                prefetch_fut = io_pool.submit(load_layer_from_safetensors, l_idx + 1) if (l_idx + 1 < N_LAYERS) else None
                ng = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ng_cpu.items()}
                lsc = float(ng["lsc"])

                w_f16 = {k: v.to(device, dtype=compute_dtype) for k, v in w_teach.items()}
                X_teach_next, seq_energy_np = teacher_layer_step(X_teach_store, w_f16, ng, l_idx, vlm_full_mask_gpu, ret_seq_energy=True)
                H_teach_eval, _ = teacher_layer_step(H_teach_eval, w_f16, ng, l_idx, heldout_full_mask_gpu, ret_seq_energy=False)
                del w_f16
                if l_idx < 58:
                    del X_teach_store

                # Spatial Semantic District Clustering on prompt/sequence GeGLU energy [N_SEQ, 21504]
                perm_np, slices_np, _ = sdc.cluster_spatial_districts(
                    seq_energy_np, n_districts=N_DISTRICTS, shared_frac=SHARED_FRAC, block_size=BLOCK_SIZE, n_iters=25
                )
                seq_energy_perm = seq_energy_np[:, perm_np]
                cap_native, sh_frac_native, oracle_mask_np = _compute_district_capture_stats(seq_energy_perm, slices_np, top_k=TOP_K_DISTRICTS)

                w_teach["gate_proj"], w_teach["up_proj"], w_teach["down_proj"] = permute_geglu_weights_idx(
                    w_teach["gate_proj"], w_teach["up_proj"], w_teach["down_proj"], perm_np
                )

                if l_idx == 58:
                    v70._atomic_torch_save(X_stud_store.cpu(), Path("/tmp/v7_5_x_stud_pre58.pt"))
                    v70._atomic_torch_save(H_stud_eval.cpu(), Path("/tmp/v7_5_h_eval_pre58.pt"))

                n_tok_l = N_SEQ * SEQ_LEN
                w_stud: Dict[str, torch.Tensor] = {}
                h_in = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32)
                xsq_in = torch.zeros((D_MOD,), device=device, dtype=torch.float32)
                x_ev_in = []
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        inp = rmsn_gpu(X_stud_store[b0:b1].to(device, dtype=compute_dtype), ng["in"]).reshape(-1, D_MOD).float()
                        accum_hess(h_in, xsq_in, inp)
                        if (len(x_ev_in) < 3 or b0 >= N_TEXT_SEQ) and len(x_ev_in) < 4:
                            x_ev_in.append(inp[:1024].clone())
                xrms_in, xev_in = torch.sqrt((xsq_in / n_tok_l).clamp(min=1e-8)), torch.cat(x_ev_in, 0)[: hw_cfg["als_ev_toks"]]
                for pn in (["q_proj", "k_proj"] if (is_f and K_EQ_V) else ["q_proj", "k_proj", "v_proj"]):
                    mi, bsi, rki, trm = get_lm_spec_v7_5(l_idx, pn)
                    nb, asy = v70.spec_to_t16c_bits(mi, trm)
                    pb, invs, lrt, we = pack_matrix_gpu_gptq(w_teach[pn], h_in, xrms_in, mi, bsi, rki, xev_in, to_rowmajor=trm)
                    shard_pos["base"] = p3._write_mat_record(shard_fps["base"], shard_pos["base"], f"{p}self_attn.{pn}.weight", w_teach[pn].shape[0], D_MOD, bsi, nb, asy, True, pb, invs, lrt)
                    base_mats_count += 1; w_stud[pn] = we.to(compute_dtype)
                    del pb, invs, lrt
                del h_in, xsq_in, xrms_in, xev_in, x_ev_in

                if l_idx in (58, 59):
                    sub_idx = np.concatenate([np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int), np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int)])
                    sub_t = torch.from_numpy(sub_idx).long().to(X_teach_store.device)
                    with torch.no_grad():
                        inpt = rmsn_gpu(X_teach_store[sub_t].to(device, dtype=compute_dtype), ng["in"]).reshape(-1, D_MOD).float()
                        if l_idx == 58:
                            inps = rmsn_gpu(X_stud_store[sub_t].to(device, dtype=compute_dtype), ng["in"]).reshape(-1, D_MOD).float()
                            bridge_cache["in_58_stud"] = inps.to(torch.float16).cpu()
                            bridge_cache["kt58"] = (inpt @ w_teach["k_proj"].to(device).T).to(torch.float16).cpu()
                            bridge_cache["ks58_base"] = (inps @ w_stud["k_proj"].float().T).to(torch.float16).cpu()
                            bridge_cache["vt58"] = (inpt @ w_teach["v_proj"].to(device).T).to(torch.float16).cpu()
                            bridge_cache["vs58_base"] = (inps @ w_stud["v_proj"].float().T).to(torch.float16).cpu()
                            del inps
                        else:
                            bridge_cache["kt59"] = (inpt @ w_teach["k_proj"].to(device).T).to(torch.float16).cpu()
                        del inpt, X_teach_store

                ctx_all = _alloc_act_store((N_SEQ, SEQ_LEN, NH * hd), device if RES_GPU else torch.device("cpu"))
                do_o, di_o = w_teach["o_proj"].shape
                h_o, xsq_o, x_ev_o = torch.zeros((di_o, di_o), device=device, dtype=torch.float32), torch.zeros((di_o,), device=device, dtype=torch.float32), []
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        inp = rmsn_gpu(X_stud_store[b0:b1].to(device, dtype=compute_dtype), ng["in"])
                        q = F.linear(inp, w_stud["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                        k = F.linear(inp, w_stud["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                        v = k.clone() if (is_f and K_EQ_V) else F.linear(inp, w_stud["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                        qn, kn, vn = apply_rope(rmsn_gpu(q, ng["qn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2), apply_rope(rmsn_gpu(k, ng["kn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2), rmsn_gpu(v, None).transpose(1, 2)
                        del inp, q, k, v
                        am = None if (is_f or vlm_full_mask_gpu is None) else vlm_full_mask_gpu[b0:b1]
                        ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(rep, 1), vn.repeat_interleave(rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0).transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd)
                        del qn, kn, vn
                        accum_hess(h_o, xsq_o, ctx.reshape(-1, di_o).float())
                        if (len(x_ev_o) < 3 or b0 >= N_TEXT_SEQ) and len(x_ev_o) < 4:
                            x_ev_o.append(ctx.reshape(-1, di_o).float()[:1024].clone())
                        ctx_all[b0:b1] = ctx.to(torch.float16).to(ctx_all.device)
                        del ctx
                mi, bsi, rki, trm = get_lm_spec_v7_5(l_idx, "o_proj")
                nb, asy = v70.spec_to_t16c_bits(mi, trm)
                pb, invs, lrt, w_o_eff = pack_matrix_gpu_gptq(w_teach["o_proj"], [h_o], torch.sqrt((xsq_o / n_tok_l).clamp(min=1e-8)), mi, bsi, rki, torch.cat(x_ev_o, 0)[: hw_cfg["als_ev_toks"]], to_rowmajor=trm)
                shard_pos["base"] = p3._write_mat_record(shard_fps["base"], shard_pos["base"], f"{p}self_attn.o_proj.weight", do_o, di_o, bsi, nb, asy, True, pb, invs, lrt)
                base_mats_count += 1; w_o_eff_c = w_o_eff.to(compute_dtype)
                if l_idx in (58, 59):
                    w_stud["o_proj"] = w_o_eff.cpu()
                del h_o, xsq_o, x_ev_o, pb, invs, lrt

                with torch.no_grad():
                    _hg = H_stud_eval.to(device, dtype=compute_dtype)
                    _hi = rmsn_gpu(_hg, ng["in"])
                    he_inp_mean = _hi.float().mean(dim=1).cpu()
                    _hq = F.linear(_hi, w_stud["q_proj"]).view(_hg.shape[0], SEQ_LEN, NH, hd)
                    _hk = F.linear(_hi, w_stud["k_proj"]).view(_hg.shape[0], SEQ_LEN, nkv, hd)
                    _hv = _hk.clone() if (is_f and K_EQ_V) else F.linear(_hi, w_stud["v_proj"]).view(_hg.shape[0], SEQ_LEN, nkv, hd)
                    _amh = None if is_f else heldout_full_mask_gpu
                    _hc = F.scaled_dot_product_attention(apply_rope(rmsn_gpu(_hq, ng["qn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2), apply_rope(rmsn_gpu(_hk, ng["kn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2).repeat_interleave(rep, 1), rmsn_gpu(_hv, None).transpose(1, 2).repeat_interleave(rep, 1), attn_mask=_amh, is_causal=(_amh is None), scale=1.0).transpose(1, 2).contiguous().view(_hg.shape[0], SEQ_LEN, NH * hd)
                    H_stud_eval_mid = (_hg.float() + rmsn_gpu(F.linear(_hc, w_o_eff_c), ng["pa"]).float()).to(compute_dtype)
                    del _hg, _hi, _hq, _hk, _hv, _amh, _hc

                # Extract CRTR + District Router slices before freeing X_stud_store
                xp_crtr_tr = X_stud_store[torch.from_numpy(tr_crtr_s).long().to(X_stud_store.device), torch.from_numpy(tr_crtr_t).long().to(X_stud_store.device)].cpu()
                xp_crtr_va = X_stud_store[torch.from_numpy(va_crtr_s).long().to(X_stud_store.device), torch.from_numpy(va_crtr_t).long().to(X_stud_store.device)].cpu()
                with torch.no_grad():
                    seq_inp_mean = torch.empty((N_SEQ, D_MOD), dtype=torch.float32, device=device)
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        seq_inp_mean[b0:b1] = rmsn_gpu(X_stud_store[b0:b1].to(device, dtype=compute_dtype), ng["in"]).float().mean(dim=1)

                h_mlp, xsq_mlp, x_ev_m = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32), torch.zeros((D_MOD,), device=device, dtype=torch.float32), []
                xm_all = _alloc_act_store((N_SEQ, SEQ_LEN, D_MOD), device if RES_GPU else torch.device("cpu"))
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        xm = X_stud_store[b0:b1].to(device, dtype=compute_dtype) + rmsn_gpu(F.linear(ctx_all[b0:b1].to(device, dtype=compute_dtype), w_o_eff_c), ng["pa"])
                        xm_all[b0:b1] = xm.to(torch.float16).to(xm_all.device)
                        hmf = rmsn_gpu(xm, ng["pf"]).reshape(-1, D_MOD).float()
                        accum_hess(h_mlp, xsq_mlp, hmf)
                        if (len(x_ev_m) < 3 or b0 >= N_TEXT_SEQ) and len(x_ev_m) < 4:
                            x_ev_m.append(hmf[:1024].clone())
                del ctx_all, X_stud_store, w_o_eff_c, w_o_eff
                for apn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    w_teach.pop(apn, None)
                    if l_idx in (58, 59) and apn in w_stud:
                        w_stud[apn] = w_stud[apn].cpu()
                    else:
                        w_stud.pop(apn, None)
                torch.cuda.empty_cache()

                # Optional District-Sparsity Distillation on middle layers if native Top-2 capture < threshold
                distill_met = None
                if (l_idx in LOD_LAYERS) and hw_cfg["distill_enabled"] and (cap_native < float(hw_cfg["distill_thresh"])):
                    sub_tr = tr_seq_idx[np.linspace(0, len(tr_seq_idx) - 1, min(128, len(tr_seq_idx)), dtype=int)]
                    sub_va = va_seq_idx[: min(32, len(va_seq_idx))]
                    with torch.no_grad():
                        hm_d_tr = rmsn_gpu(xm_all[torch.from_numpy(sub_tr).long().to(xm_all.device)].to(device, dtype=compute_dtype), ng["pf"]).cpu()
                        hm_d_va = rmsn_gpu(xm_all[torch.from_numpy(sub_va).long().to(xm_all.device)].to(device, dtype=compute_dtype), ng["pf"]).cpu()
                    wg_n, wu_n, wd_n, distill_met = distill_district_locality(
                        hm_d_tr, hm_d_va, w_teach["gate_proj"], w_teach["up_proj"], w_teach["down_proj"],
                        slices_np, top_k=TOP_K_DISTRICTS, steps=int(hw_cfg["distill_steps"]), lr=1.5e-4, batch_size=int(hw_cfg["distill_bs"]), device=device,
                    )
                    w_teach["gate_proj"], w_teach["up_proj"], w_teach["down_proj"] = wg_n, wu_n, wd_n
                    del hm_d_tr, hm_d_va

                # Fit Prompt-Level District Router for this layer (trained on real seq_energy_perm)
                non_sh = [(s, e) for nm, s, e in slices_np if nm != "shared_core"]
                dist_e_all = torch.from_numpy(np.stack([seq_energy_perm[:, s:e].sum(axis=1) for s, e in non_sh], axis=1)).float()
                dr_sd, dr_met = fit_district_router(
                    seq_inp_mean[torch.from_numpy(tr_seq_idx).long().to(device)],
                    dist_e_all[torch.from_numpy(tr_seq_idx).long()],
                    seq_inp_mean[torch.from_numpy(va_seq_idx).long().to(device)],
                    dist_e_all[torch.from_numpy(va_seq_idx).long()],
                    rank=64, top_k=TOP_K_DISTRICTS, steps=200, device=device, seed=200 + l_idx,
                )
                district_routers[l_idx] = {"state_dict": dr_sd, "district_slices": slices_np, "perm": perm_np.astype(np.int32)}
                del seq_inp_mean, dist_e_all

                # Quantize gate_proj, up_proj, down_proj (V7.5 LOD split on L02..L55, full-prec on boundary layers)
                xrms_mlp = torch.sqrt((xsq_mlp / n_tok_l).clamp(min=1e-8))
                xev_mlp = torch.cat(x_ev_m, 0)[: hw_cfg["als_ev_toks"]]
                lod_stats_l = {}
                if l_idx in LOD_LAYERS:
                    for pn in ("gate_proj", "up_proj"):
                        lr_res = split_and_pack_lod_matrix(
                            w_teach.pop(pn), h_mlp.clone() if pn == "gate_proj" else h_mlp, xrms_mlp,
                            slices_np, is_down_proj=False, bsize=256, device=device,
                            fwht_gpu_fn=fwht_gpu, fwht_2sided_fn=fwht_2sided_gpu, gptq_gpu_fn=gptq_quant_gpu,
                            pack_asym1_fn=lambda a, do, di, bsz: c_pack_tile16_with_q(a, do, di, bsz, 0),
                            pack_sym2_fn=lambda a, do, di, bsz: c_pack_tile16_with_q(a, do, di, bsz, 2),
                        )
                        shard_pos["base"] = p3._write_mat_record(
                            shard_fps["base"], shard_pos["base"], f"{p}mlp.{pn}.weight", 21504, D_MOD, 256, 1 | T16C_ROWMAJOR_BIT, True, True,
                            lr_res["pb_base_rm"], lr_res["inv_s_base"], None,
                        )
                        base_mats_count += 1
                        for sn, (do_s, di_s, pb_s, inv_sl) in lr_res["shard_records"].items():
                            shard_pos[sn] = p3._write_mat_record(shard_fps[sn], shard_pos[sn], f"{p}mlp.{pn}.weight", do_s, di_s, 256, 2, False, True, pb_s, inv_sl, None)
                        w_stud[f"{pn}_1b"] = lr_res["w_eff_1b"].cpu()
                        w_stud[f"{pn}_full"] = lr_res["w_eff_full"].cpu()
                        lod_stats_l[pn] = round(float(lr_res["mse_ratio"]), 2)
                        del lr_res
                    del h_mlp, xsq_mlp, xrms_mlp, xev_mlp, x_ev_m

                    oracle_t = torch.from_numpy(oracle_mask_np).to(device)
                    do_d, di_d = w_teach["down_proj"].shape
                    h_d, xsq_d = torch.zeros((di_d, di_d), device=device, dtype=torch.float32), torch.zeros((di_d,), device=device, dtype=torch.float32)
                    wg1_g, wgf_g = w_stud["gate_proj_1b"].to(device, dtype=compute_dtype), w_stud["gate_proj_full"].to(device, dtype=compute_dtype)
                    wu1_g, wuf_g = w_stud["up_proj_1b"].to(device, dtype=compute_dtype), w_stud["up_proj_full"].to(device, dtype=compute_dtype)
                    with torch.no_grad():
                        dw_g = wgf_g - wg1_g
                        dw_u = wuf_g - wu1_g
                        for b0 in range(0, N_SEQ, MB_FWD):
                            b1 = min(b0 + MB_FWD, N_SEQ)
                            hm = rmsn_gpu(xm_all[b0:b1].to(device, dtype=compute_dtype), ng["pf"])
                            m_b = oracle_t[b0:b1].unsqueeze(1)
                            g_t = F.linear(hm, wg1_g)
                            g_t.add_(F.linear(hm, dw_g).mul_(m_b))
                            af = F.gelu(g_t.float(), approximate="tanh")
                            del g_t
                            u_t = F.linear(hm, wu1_g)
                            u_t.add_(F.linear(hm, dw_u).mul_(m_b))
                            del hm, m_b
                            af.mul_(u_t.float())
                            del u_t
                            accum_hess(h_d, xsq_d, af.reshape(-1, di_d))
                            del af
                        del dw_g, dw_u
                    del wg1_g, wgf_g, wu1_g, wuf_g

                    lr_d = split_and_pack_lod_matrix(
                        w_teach.pop("down_proj"), h_d, torch.sqrt((xsq_d / n_tok_l).clamp(min=1e-8)),
                        slices_np, is_down_proj=True, bsize=256, device=device,
                        fwht_gpu_fn=fwht_gpu, fwht_2sided_fn=fwht_2sided_gpu, gptq_gpu_fn=gptq_quant_gpu,
                        pack_asym1_fn=lambda a, do, di, bsz: c_pack_tile16_with_q(a, do, di, bsz, 0),
                        pack_sym2_fn=lambda a, do, di, bsz: c_pack_tile16_with_q(a, do, di, bsz, 2),
                    )
                    shard_pos["base"] = p3._write_mat_record(
                        shard_fps["base"], shard_pos["base"], f"{p}mlp.down_proj.weight", do_d, di_d, 256, 1 | T16C_ROWMAJOR_BIT, True, True,
                        lr_d["pb_base_rm"], lr_d["inv_s_base"], None,
                    )
                    base_mats_count += 1
                    for sn, (do_s, di_s, pb_s, inv_sl) in lr_d["shard_records"].items():
                        shard_pos[sn] = p3._write_mat_record(shard_fps[sn], shard_pos[sn], f"{p}mlp.down_proj.weight", do_s, di_s, 256, 2, False, True, pb_s, inv_sl, None)
                    lod_mats_written += 3
                    lod_stats_l["down_proj"] = round(float(lr_d["mse_ratio"]), 2)
                    wd1_g, wdf_g = lr_d["w_eff_1b"].to(device, dtype=compute_dtype), lr_d["w_eff_full"].to(device, dtype=compute_dtype)
                    del lr_d, h_d, xsq_d, w_teach

                    # Forward Held-Out & Student States using GTA LOD approximation (Shared + Top-2 hi-res, else 1b!)
                    wg1_g, wgf_g = w_stud["gate_proj_1b"].to(device, dtype=compute_dtype), w_stud["gate_proj_full"].to(device, dtype=compute_dtype)
                    wu1_g, wuf_g = w_stud["up_proj_1b"].to(device, dtype=compute_dtype), w_stud["up_proj_full"].to(device, dtype=compute_dtype)
                    with torch.no_grad():
                        _hm_he = rmsn_gpu(H_stud_eval_mid, ng["pf"])
                        # For heldout images use predicted router top-2 + Shared Core on pre-attn normalized input!
                        _he_inp = he_inp_mean.to(device)
                        _rnet = DistrictRouterNet(D_MOD, N_DISTRICTS, 64).to(device)
                        _rnet.load_state_dict({k: v.to(device, dtype=torch.float32) for k, v in dr_sd.items()})
                        _top_he = torch.topk(_rnet(_he_inp), k=min(TOP_K_DISTRICTS, N_DISTRICTS), dim=-1).indices.cpu().numpy()
                        _m_he = torch.zeros((H_stud_eval_mid.shape[0], 21504), dtype=torch.bool, device=device)
                        _sh_sl = [x for x in slices_np if x[0] == "shared_core"]
                        _ns_sl = [x for x in slices_np if x[0] != "shared_core"]
                        if _sh_sl:
                            _m_he[:, _sh_sl[0][1] : _sh_sl[0][2]] = True
                        for r_i in range(H_stud_eval_mid.shape[0]):
                            for d_i in _top_he[r_i]:
                                _, s_i, e_i = _ns_sl[int(d_i)]
                                _m_he[r_i, s_i:e_i] = True
                        _he_out = forward_geglu_lod_batch(_hm_he, wg1_g, wgf_g, wu1_g, wuf_g, wd1_g, wdf_g, _m_he)
                        H_stud_eval = ((H_stud_eval_mid.float() + rmsn_gpu(_he_out.to(compute_dtype), ng["pff"]).float()) * lsc).to(torch.float16).to(device if RES_GPU else "cpu")
                        del _hm_he, _he_inp, _rnet, _top_he, _m_he, _he_out, H_stud_eval_mid

                    X_stud_next = _alloc_act_store((N_SEQ, SEQ_LEN, D_MOD), device if RES_GPU else torch.device("cpu"))
                    with torch.no_grad():
                        for b0 in range(0, N_SEQ, MB_FWD):
                            b1 = min(b0 + MB_FWD, N_SEQ)
                            xm_g = xm_all[b0:b1].to(device, dtype=compute_dtype)
                            hm = rmsn_gpu(xm_g, ng["pf"])
                            y_mlp = forward_geglu_lod_batch(hm, wg1_g, wgf_g, wu1_g, wuf_g, wd1_g, wdf_g, oracle_t[b0:b1])
                            X_stud_next[b0:b1] = ((xm_g.float() + rmsn_gpu(y_mlp.to(compute_dtype), ng["pff"]).float()) * lsc).to(torch.float16).to(X_stud_next.device)

                    # Train CRTR using full high-res gate/up representation on hm (matches V7.0 gate routing!)
                    wg_cr, wu_cr = wgf_g, wuf_g
                    del wg1_g, wu1_g, wd1_g, wdf_g, oracle_t
                else:
                    for pn in ("gate_proj", "up_proj"):
                        mi, bsi, rki, trm = get_lm_spec_v7_5(l_idx, pn)
                        nb, asy = v70.spec_to_t16c_bits(mi, trm)
                        pb, invs, lrt, we = pack_matrix_gpu_gptq(w_teach.pop(pn), h_mlp, xrms_mlp, mi, bsi, rki, xev_mlp, to_rowmajor=trm)
                        shard_pos["base"] = p3._write_mat_record(shard_fps["base"], shard_pos["base"], f"{p}mlp.{pn}.weight", 21504, D_MOD, bsi, nb, asy, True, pb, invs, lrt)
                        base_mats_count += 1; w_stud[pn] = we.to(compute_dtype)
                        del pb, invs, lrt
                    del h_mlp, xsq_mlp, xrms_mlp, xev_mlp, x_ev_m

                    do_d, di_d = w_teach["down_proj"].shape
                    h_d, xsq_d, x_ev_d = torch.zeros((di_d, di_d), device=device, dtype=torch.float32), torch.zeros((di_d,), device=device, dtype=torch.float32), []
                    with torch.no_grad():
                        for b0 in range(0, N_SEQ, MB_FWD):
                            hm = rmsn_gpu(xm_all[b0 : min(b0 + MB_FWD, N_SEQ)].to(device, dtype=compute_dtype), ng["pf"])
                            act = F.gelu(F.linear(hm, w_stud["gate_proj"]).float(), approximate="tanh")
                            act.mul_(F.linear(hm, w_stud["up_proj"]).float())
                            del hm
                            af = act.reshape(-1, di_d)
                            accum_hess(h_d, xsq_d, af)
                            if (len(x_ev_d) < 3 or b0 >= N_TEXT_SEQ) and len(x_ev_d) < 4:
                                x_ev_d.append(af[:1024].clone())
                            del act, af
                    w_stud["gate_proj"] = w_stud["gate_proj"].cpu()
                    w_stud["up_proj"] = w_stud["up_proj"].cpu()
                    torch.cuda.empty_cache()

                    mi, bsi, rki, trm = get_lm_spec_v7_5(l_idx, "down_proj")
                    nb, asy = v70.spec_to_t16c_bits(mi, trm)
                    pb, invs, lrt, w_d_eff = pack_matrix_gpu_gptq(w_teach.pop("down_proj"), [h_d], torch.sqrt((xsq_d / n_tok_l).clamp(min=1e-8)), mi, bsi, rki, torch.cat(x_ev_d, 0)[: hw_cfg["als_ev_toks"]], to_rowmajor=trm)
                    shard_pos["base"] = p3._write_mat_record(shard_fps["base"], shard_pos["base"], f"{p}mlp.down_proj.weight", do_d, di_d, bsi, nb, asy, True, pb, invs, lrt)
                    base_mats_count += 1
                    del h_d, xsq_d, x_ev_d, w_teach, pb, invs, lrt

                    if l_idx in (58, 59):
                        v70._atomic_torch_save({"w": {k: v.cpu() for k, v in w_stud.items()} | {"down_proj": w_d_eff.cpu()}, "norms": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in ng.items()}}, Path(f"/tmp/v7_5_l{l_idx}_stud.pt"))

                    wg_cr, wu_cr = w_stud["gate_proj"].to(device, dtype=compute_dtype), w_stud["up_proj"].to(device, dtype=compute_dtype)
                    w_d_c = w_d_eff.to(compute_dtype)
                    with torch.no_grad():
                        _hm_h = rmsn_gpu(H_stud_eval_mid, ng["pf"])
                        _hact_f = F.gelu(F.linear(_hm_h, wg_cr).float(), approximate="tanh")
                        _hact_f.mul_(F.linear(_hm_h, wu_cr).float())
                        del _hm_h
                        _hact = _hact_f.to(compute_dtype)
                        del _hact_f
                        H_stud_eval = ((H_stud_eval_mid.float() + rmsn_gpu(F.linear(_hact, w_d_c), ng["pff"]).float()) * lsc).to(torch.float16).to(device if RES_GPU else "cpu")
                        del H_stud_eval_mid, _hact
                    X_stud_next = _alloc_act_store((N_SEQ, SEQ_LEN, D_MOD), device if RES_GPU else torch.device("cpu"))
                    with torch.no_grad():
                        for b0 in range(0, N_SEQ, MB_FWD):
                            b1 = min(b0 + MB_FWD, N_SEQ)
                            xm_g = xm_all[b0:b1].to(device, dtype=compute_dtype); hm = rmsn_gpu(xm_g, ng["pf"])
                            act_f = F.gelu(F.linear(hm, wg_cr).float(), approximate="tanh")
                            act_f.mul_(F.linear(hm, wu_cr).float())
                            del hm
                            act = act_f.to(compute_dtype)
                            del act_f
                            X_stud_next[b0:b1] = ((xm_g.float() + rmsn_gpu(F.linear(act, w_d_c), ng["pff"]).float()) * lsc).to(torch.float16).to(X_stud_next.device)
                            del xm_g, act
                    del w_d_eff, w_d_c

                xm_crtr_tr = xm_all[torch.from_numpy(tr_crtr_s).long().to(xm_all.device), torch.from_numpy(tr_crtr_t).long().to(xm_all.device)].cpu()
                xm_crtr_va = xm_all[torch.from_numpy(va_crtr_s).long().to(xm_all.device), torch.from_numpy(va_crtr_t).long().to(xm_all.device)].cpu()
                del xm_all, w_stud
                torch.cuda.empty_cache()

                crec, cm = v70.train_layer_crtr_v7_0(
                    l_idx,
                    xp_crtr_tr, xm_crtr_tr, X_stud_next[torch.from_numpy(tr_crtr_s).long().to(X_stud_next.device), torch.from_numpy(tr_crtr_t).long().to(X_stud_next.device)].cpu(),
                    xp_crtr_va, xm_crtr_va, X_stud_next[torch.from_numpy(va_crtr_s).long().to(X_stud_next.device), torch.from_numpy(va_crtr_t).long().to(X_stud_next.device)].cpu(),
                    ng["in"], ng["pf"], wg_cr, wu_cr, rmsn_gpu,
                    block_size=256, rank=32, top_p=0.30, mod_quantile=0.20, steps=200, eps=EPS, device=device,
                )
                crtr_layers[l_idx] = crec
                del xp_crtr_tr, xp_crtr_va, xm_crtr_tr, xm_crtr_va, wg_cr, wu_cr

                va_idx_t = torch.from_numpy(va_seq_idx[:16]).long().to(X_stud_next.device)
                cos_l_txt = float(F.cosine_similarity(X_stud_next[va_idx_t].reshape(-1, D_MOD).float().to(device), X_teach_next[va_idx_t].reshape(-1, D_MOD).float().to(device), dim=-1).mean().item())
                cos_l_vlm = float(F.cosine_similarity(H_stud_eval.reshape(-1, D_MOD).float().to(device), H_teach_eval.reshape(-1, D_MOD).float().to(device), dim=-1).mean().item())
                del va_idx_t
                d_entry = {
                    "layer": l_idx,
                    "is_lod_layer": l_idx in LOD_LAYERS,
                    "cap_native": round(cap_native, 4),
                    "shared_frac_native": round(sh_frac_native, 4),
                    "router": dr_met,
                    "distill": distill_met,
                    "lod_mse_ratios": lod_stats_l,
                    "slices": slices_np,
                }
                district_manifest.append(d_entry)
                layer_cos_log.append({"l": l_idx, "txt_cos": round(cos_l_txt, 5), "vlm_cos": round(cos_l_vlm, 5), "cap_native": round(cap_native, 4), "router_rec": dr_met["val_top2_recall_after"], "crtr": cm})
                X_teach_store, X_stud_store = X_teach_next, X_stud_next
                del X_teach_next, X_stud_next, ng, cr_l, sr_l
                gc.collect()
                torch.cuda.empty_cache()

                if (l_idx + 1) % 2 == 0 or l_idx == 0:
                    for fp_o in shard_fps.values():
                        fp_o.flush(); os.fsync(fp_o.fileno())
                    v70._atomic_torch_save(X_teach_store.cpu(), CKPT_X_TEACH)
                    v70._atomic_torch_save(X_stud_store.cpu(), CKPT_X_STUD)
                    v70._atomic_torch_save(crtr_layers, CRTR_CKPT)
                    v70._atomic_torch_save(district_routers, D_ROUTER_CKPT)
                    v70._atomic_torch_save({
                        "next_layer": l_idx + 1, "base_mats_count": base_mats_count, "bridge_cache": bridge_cache,
                        "H_teach_eval": H_teach_eval.cpu(), "H_stud_eval": H_stud_eval.cpu(), "layer_cos_log": layer_cos_log,
                        "district_manifest": district_manifest, "shard_pos": dict(shard_pos), "lod_mats_written": lod_mats_written,
                    }, STAGE3_CKPT)
                    exp_sync = {OUT_BASE_T16C: shard_pos["base"]}
                    for sn_k, sp_k in shard_map.items():
                        exp_sync[sp_k] = shard_pos[sn_k]
                    sync_v7_5_checkpoints_to_drive(exp_sync)
                    gc.collect()

                save_progress({"stage": f"L{l_idx:02d}_done", "next_layer": l_idx + 1, "txt_cos": round(cos_l_txt, 5), "vlm_cos": round(cos_l_vlm, 5)})
                va_gb, vp_gb, vt_gb = _gpu_vram_stats_gb(device)
                print(
                    f"  [V7.5 L{l_idx:02d}/59 {'F' if is_f else 'S'}{'*' if l_idx in LOD_LAYERS else ''}] "
                    f"txt_cos={cos_l_txt:.5f} vlm_cos={cos_l_vlm:.5f} | cap={cap_native:.3f} r_rec={dr_met['val_top2_recall_after']:.3f} "
                    f"crtr={cm['val_recall_after']:.3f} | vram={va_gb}/peak={vp_gb}/{vt_gb}GiB | dt={time.perf_counter()-tl:.1f}s",
                    flush=True,
                )

            assert base_mats_count == expected_lm_mats and lod_mats_written == expected_lod_mats
            v_tail = torch.load(VIS_CKPT_PATH, map_location="cpu", weights_only=False)
            for rec_v in v_tail["vision_records"]:
                shard_pos["base"] = p3._write_mat_record(shard_fps["base"], shard_pos["base"], *rec_v)
            del v_tail
            for fp_o in shard_fps.values():
                fp_o.flush(); os.fsync(fp_o.fileno())
        finally:
            io_pool.shutdown(wait=True)
            for fp_o in shard_fps.values():
                try:
                    fp_o.close()
                except Exception:
                    pass

    # Verify ALL 10 T16C files (Base + Shared + District 0..7)
    base_mats_v, base_end_v = v70.verify_t16c_integrity(OUT_BASE_T16C)
    base_sz = OUT_BASE_T16C.stat().st_size
    assert base_mats_v == total_base_mats and base_end_v == base_sz and base_sz <= MAX_BASE_T16C_BYTES
    for sn_k, sp_k in shard_map.items():
        m_k, e_k = v70.verify_t16c_integrity(sp_k)
        assert m_k == expected_lod_mats and e_k == sp_k.stat().st_size, f"Bad LOD shard {sp_k}: mats={m_k}"

    crtr_sz = crtr.export_crtr([crtr_layers[l] for l in range(N_LAYERS)], OUT_CRTR)
    v70._atomic_torch_save({"n_layers": N_LAYERS, "n_districts": N_DISTRICTS, "block_size": BLOCK_SIZE, "top_k": TOP_K_DISTRICTS, "layers": district_routers}, OUT_DISTRICT_ROUTER)
    OUT_DISTRICT_MANIFEST.write_text(json.dumps({"layers": district_manifest}, indent=2), encoding="utf-8")
    print(f"[Stage 3 DONE] Base={base_sz/1024**3:.3f} GiB + 13 LOD shards | CRTR={crtr_sz:,} B", flush=True)

    # Persist ALL completed Stage 3 artifacts to Drive IMMEDIATELY before Stage 4A/4B (never delete checkpoints first!)
    for fp_s3 in all_t16c_paths + [OUT_CRTR, OUT_DISTRICT_ROUTER, OUT_DISTRICT_MANIFEST]:
        persist_v7_5_file(fp_s3)
    print(f"[Stage 3 PERSISTED TO DRIVE] {len(all_t16c_paths) + 3} verified artifacts safely saved.", flush=True)

    # Stage 4A: Multimodal C8DB + Capture L58/L59 Student KV
    print("\n[Stage 4A] Multimodal C8DB + Capturing V7.5 L58/L59 KV (128 seqs)...", flush=True)
    c8db_entries: List[dict] = []
    if vis_keep_c8d and cents_vis is not None:
        exp_rv = [c8d._c6d_radicals("vision", "patch", "soft", "native", "layer", "positive", e) for e in range(8)]
        c8db_entries.append({"name": "model.embed_vision.embedding_projection.weight", "d_out": D_MOD, "d_in": 1152, "rank": 32, "top_k": 2, "c6d": exp_rv[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents_vis, "expert_radicals": exp_rv, "experts": experts_vis})

    def train_c8d_proj(name: str, x_tr, yt_tr, ys_tr, x_va, yt_va, ys_va, rank: int = 32, steps: int = 300):
        cos0 = float(F.cosine_similarity(ys_va, yt_va, dim=-1).mean().item())
        m = c8d.Cuneiform8DPatternAdapter(d_out=yt_tr.shape[1], d_in=x_tr.shape[1], rank=rank, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=2e-2)
        best_c, best_sd = cos0, None
        for st in range(steps):
            opt.zero_grad()
            ib = torch.randint(0, x_tr.shape[0], (min(2048, x_tr.shape[0]),), device=device)
            dv, _, lb = m(x_tr[ib]); p_y = ys_tr[ib] + dv
            loss = F.mse_loss(p_y, yt_tr[ib]) + 0.3 * (1.0 - F.cosine_similarity(p_y, yt_tr[ib], dim=-1).mean()) + 0.005 * lb
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if (st + 1) % 60 == 0:
                with torch.no_grad():
                    cv = float(F.cosine_similarity(ys_va + m(x_va)[0], yt_va, dim=-1).mean().item())
                    if cv > best_c:
                        best_c, best_sd = cv, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        if best_sd is None:
            return None, None, None
        mc = m.cpu(); mc.load_state_dict(best_sd)
        ct, ex = mc.export_quantized_arrays()
        cos1 = float(F.cosine_similarity(ys_va + eval_c8d_gpu(x_va, ct, ex), yt_va, dim=-1).mean().item())
        print(f"  [C8DB-KV] {name}: val_cos {cos0:.5f}->{cos1:.5f} (keep={cos1 > cos0 + 0.0005})", flush=True)
        if cos1 <= cos0 + 0.0005:
            return None, None, None
        exp_r = [c8d._c6d_radicals("kv", "head", "shared", "native", "layer", "positive", e) for e in range(8)]
        return {"name": name, "d_out": yt_tr.shape[1], "d_in": x_tr.shape[1], "rank": rank, "top_k": 2, "c6d": exp_r[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": ct, "expert_radicals": exp_r, "experts": ex}, ct, ex

    def forward_adapted_layer(X_in_c: torch.Tensor, ckpt_p: Path, li: int, adp_k=None, adp_v=None, cap_sub=None, cap_ast=None, mask_full=None, capture_only: bool = False):
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=True)
        wg = {k: v.to(device, dtype=compute_dtype) for k, v in ck["w"].items()}
        ng = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ck["norms"].items()}
        is_f = LAYER_TYPES[li] == "full_attention"
        nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
        rot_d = int(round(hd * (PART_G if is_f else PART_L)))
        cr_l, sr_l = make_rope(THETA_G if is_f else THETA_L, rot_d, hd)
        rep, lsc = NH // nkv, float(ng["lsc"])
        if capture_only:
            ia = torch.as_tensor(cap_sub, dtype=torch.long)
            ci, ck_l = [], []
            with torch.no_grad():
                for s0 in range(0, ia.numel(), MB_FWD):
                    sel = ia[s0 : min(s0 + MB_FWD, ia.numel())]
                    inp = rmsn_gpu(X_in_c[sel].to(device, dtype=compute_dtype), ng["in"])
                    k2 = F.linear(inp.reshape(-1, D_MOD), wg["k_proj"]).float()
                    if adp_k is not None:
                        k2 = k2 + eval_c8d_gpu(inp.reshape(-1, D_MOD), adp_k[0], adp_k[1])
                    ci.append(inp.cpu()); ck_l.append(k2.reshape(sel.numel(), SEQ_LEN, -1).to(torch.float16).cpu())
            del ck, wg, ng
            return None, {"in": torch.cat(ci, 0), "k": torch.cat(ck_l, 0)}, None
        ast_set = set(cap_ast) if cap_ast is not None else set()
        ast_k, ast_v = {}, {}
        with torch.no_grad():
            for b0 in range(0, X_in_c.shape[0], MB_FWD):
                b1 = min(b0 + MB_FWD, X_in_c.shape[0])
                xb = X_in_c[b0:b1].to(device, dtype=compute_dtype); inp = rmsn_gpu(xb, ng["in"]); inp2 = inp.reshape(-1, D_MOD)
                q = F.linear(inp, wg["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                k2 = F.linear(inp2, wg["k_proj"]).float()
                if adp_k is not None:
                    k2 = k2 + eval_c8d_gpu(inp2, adp_k[0], adp_k[1])
                k = k2.to(compute_dtype).view(b1 - b0, SEQ_LEN, nkv, hd)
                if is_f and K_EQ_V:
                    v = k.clone()
                else:
                    v2 = F.linear(inp2, wg["v_proj"]).float()
                    if adp_v is not None:
                        v2 = v2 + eval_c8d_gpu(inp2, adp_v[0], adp_v[1])
                    v = v2.to(compute_dtype).view(b1 - b0, SEQ_LEN, nkv, hd)
                qn = apply_rope(rmsn_gpu(q, ng["qn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2)
                kn = apply_rope(rmsn_gpu(k, ng["kn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2)
                vn = rmsn_gpu(v, None).transpose(1, 2)
                for gi in range(b0, b1):
                    if gi in ast_set:
                        ast_k[gi] = kn[gi - b0].to(torch.float16).cpu()
                        if not (is_f and K_EQ_V):
                            ast_v[gi] = vn[gi - b0].to(torch.float16).cpu()
                am = None if (is_f or mask_full is None) else mask_full[b0:b1]
                ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(rep, 1), vn.repeat_interleave(rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0).transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd)
                xm = xb + rmsn_gpu(F.linear(ctx, wg["o_proj"]), ng["pa"])
                hm = rmsn_gpu(xm, ng["pf"])
                act = (F.gelu(F.linear(hm, wg["gate_proj"]).float(), approximate="tanh") * F.linear(hm, wg["up_proj"]).float()).to(compute_dtype)
                X_in_c[b0:b1] = ((xm.float() + rmsn_gpu(F.linear(act, wg["down_proj"]), ng["pff"]).float()) * lsc).to(torch.float16).cpu()
        del ck, wg, ng; torch.cuda.empty_cache()
        return X_in_c, None, {"k_rope": ast_k, "v_norm": ast_v}

    adp_k58 = adp_v58 = adp_k59 = None
    for attr_t, nm_k, ytk, ysk, popx in [
        ("k58", "model.language_model.layers.58.self_attn.k_proj.weight", "kt58", "ks58_base", False),
        ("v58", "model.language_model.layers.58.self_attn.v_proj.weight", "vt58", "vs58_base", True),
    ]:
        xk = (bridge_cache.pop("in_58_stud") if popx else bridge_cache["in_58_stud"]).float()
        ytk_t, ysk_t = bridge_cache.pop(ytk).float(), bridge_cache.pop(ysk).float()
        ntr = (xk.shape[0] * 4) // 5
        ent, ct, ex = train_c8d_proj(nm_k, xk[:ntr].to(device), ytk_t[:ntr].to(device), ysk_t[:ntr].to(device), xk[ntr:].to(device), ytk_t[ntr:].to(device), ysk_t[ntr:].to(device))
        if ent is not None:
            c8db_entries.append(ent)
            if attr_t == "k58":
                adp_k58 = (ct, ex)
            else:
                adp_v58 = (ct, ex)
        del xk, ytk_t, ysk_t

    del X_stud_store; gc.collect()
    AST_SEQS = np.linspace(0, N_TEXT_SEQ - 1, 128, dtype=int).tolist()
    x58 = torch.load("/tmp/v7_5_x_stud_pre58.pt", map_location="cpu", weights_only=True); Path("/tmp/v7_5_x_stud_pre58.pt").unlink(missing_ok=True)
    x59_in, _, cap_kv58 = forward_adapted_layer(x58, Path("/tmp/v7_5_l58_stud.pt"), 58, adp_k58, adp_v58, cap_ast=AST_SEQS, mask_full=vlm_full_mask_gpu)
    hev58 = torch.load("/tmp/v7_5_h_eval_pre58.pt", map_location="cpu", weights_only=True); Path("/tmp/v7_5_h_eval_pre58.pt").unlink(missing_ok=True)
    hev59, _, _ = forward_adapted_layer(hev58, Path("/tmp/v7_5_l58_stud.pt"), 58, adp_k58, adp_v58, mask_full=heldout_full_mask_gpu)
    Path("/tmp/v7_5_l58_stud.pt").unlink(missing_ok=True)

    sub_59 = np.concatenate([np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int), np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int)])
    _, c59, _ = forward_adapted_layer(x59_in, Path("/tmp/v7_5_l59_stud.pt"), 59, cap_sub=sub_59, capture_only=True)
    xk59, ytk59, ysk59 = c59["in"].reshape(-1, D_MOD).float(), bridge_cache.pop("kt59").float(), c59["k"].reshape(-1, 2048).float()
    del c59, bridge_cache
    ntr59 = (xk59.shape[0] * 4) // 5
    ent59, c59n, e59n = train_c8d_proj("model.language_model.layers.59.self_attn.k_proj.weight", xk59[:ntr59].to(device), ytk59[:ntr59].to(device), ysk59[:ntr59].to(device), xk59[ntr59:].to(device), ytk59[ntr59:].to(device), ysk59[ntr59:].to(device))
    if ent59 is not None:
        c8db_entries.append(ent59); adp_k59 = (c59n, e59n)
    del xk59, ytk59, ysk59
    X_stud_cpu, _, cap_kv59 = forward_adapted_layer(x59_in, Path("/tmp/v7_5_l59_stud.pt"), 59, adp_k59, cap_ast=AST_SEQS, mask_full=vlm_full_mask_gpu)
    H_stud_eval, _, _ = forward_adapted_layer(hev59, Path("/tmp/v7_5_l59_stud.pt"), 59, adp_k59, mask_full=heldout_full_mask_gpu)
    Path("/tmp/v7_5_l59_stud.pt").unlink(missing_ok=True)

    cos_l59 = float(F.cosine_similarity(X_stud_cpu[-16:].reshape(-1, D_MOD).float().to(device), X_teach_store[-16:].reshape(-1, D_MOD).float().to(device), dim=-1).mean().item())
    print(f"  [L59 V7.5 VLM cos] {cos_l59:.5f}", flush=True)

    fn_g = fn_orig_cpu.to(device)
    num_w, den_w = torch.zeros((D_MOD,), device=device), torch.zeros((D_MOD,), device=device)
    val_sq_o = val_sq_w = 0.0
    with torch.no_grad():
        for b0 in range(0, N_SEQ, MB_FWD):
            b1 = min(b0 + MB_FWD, N_SEQ)
            xtb = rmsn_gpu(X_teach_store[b0:b1].to(device), fn_g).reshape(-1, D_MOD).float()
            xub = rmsn_gpu(X_stud_cpu[b0:b1].to(device), None).reshape(-1, D_MOD).float()
            iv = torch.arange(xtb.shape[0], device=device) % 10 == 0
            num_w.add_((xub[~iv] * xtb[~iv]).sum(dim=0)); den_w.add_((xub[~iv] * xub[~iv]).sum(dim=0))
        fn_wiener = ((num_w / (N_SEQ * SEQ_LEN * 0.9)) + 0.05 * fn_g) / ((den_w / (N_SEQ * SEQ_LEN * 0.9)) + 0.05).clamp(min=1e-8)
        for b0 in range(0, N_SEQ, MB_FWD):
            b1 = min(b0 + MB_FWD, N_SEQ)
            xtb = rmsn_gpu(X_teach_store[b0:b1].to(device), fn_g).reshape(-1, D_MOD).float()
            xub = rmsn_gpu(X_stud_cpu[b0:b1].to(device), None).reshape(-1, D_MOD).float()
            iv = torch.arange(xtb.shape[0], device=device) % 10 == 0
            val_sq_o += float(((xub[iv] * fn_g - xtb[iv]) ** 2).sum().item())
            val_sq_w += float(((xub[iv] * fn_wiener - xtb[iv]) ** 2).sum().item())
    fn_eff_g = fn_wiener if val_sq_w < val_sq_o else fn_g

    vlm_pos = np.flatnonzero(all_vlm_ans_mask.reshape(-1))
    txt_pos = np.flatnonzero((all_ans_mask & (~all_vlm_ans_mask)).reshape(-1))
    sel_v = vlm_pos[np.linspace(0, vlm_pos.size - 1, min(8192, vlm_pos.size), dtype=int)] if vlm_pos.size > 0 else np.array([], dtype=int)
    sel_t = txt_pos[np.linspace(0, txt_pos.size - 1, min(32768 - sel_v.size, txt_pos.size), dtype=int)]
    sel_pos = np.sort(np.concatenate([sel_t, sel_v]))
    s_t, t_t = torch.from_numpy(sel_pos // SEQ_LEN).long(), torch.from_numpy(sel_pos % SEQ_LEN).long()
    ht_ans = rmsn_gpu(X_teach_store[s_t.to(X_teach_store.device), t_t.to(X_teach_store.device)].to(device).unsqueeze(1), fn_g).squeeze(1).float()
    hs_ans = rmsn_gpu(X_stud_cpu[s_t, t_t].to(device).unsqueeze(1), fn_eff_g).squeeze(1).float()
    nxt_ans = torch.from_numpy(all_seq_nxt.reshape(-1)[sel_pos]).to(device, dtype=torch.long)
    ast_x59_stud = X_stud_cpu[AST_SEQS].clone()
    del X_teach_store, X_stud_cpu; gc.collect(); torch.cuda.empty_cache()

    with safe_open(sf_path, framework="pt", device="cpu") as sf_e:
        emb_gpu = sf_e.get_tensor("model.language_model.embed_tokens.weight").to(device, dtype=compute_dtype)
    N_ANS, TAU = hs_ans.shape[0], 1.5
    t_lse = torch.full((N_ANS,), -1e9, device=device)
    q_lse = torch.full((N_ANS,), -1e9, device=device)
    top_it, top_iq = [], []
    with torch.no_grad():
        for r0 in range(0, emb_gpu.shape[0], 8192):
            wc = emb_gpu[r0 : r0 + 8192].float()
            zt = SOFTCAP * torch.tanh((ht_ans @ wc.T) / SOFTCAP) / TAU
            zq = SOFTCAP * torch.tanh((hs_ans @ wc.T) / SOFTCAP) / TAU
            t_lse = torch.logaddexp(t_lse, torch.logsumexp(zt, dim=-1))
            q_lse = torch.logaddexp(q_lse, torch.logsumexp(zq, dim=-1))
            top_it.append((torch.topk(zt, 32, dim=-1).indices + r0).reshape(-1))
            top_iq.append((torch.topk(zq, 32, dim=-1).indices + r0).reshape(-1))
        uniq, cnts = torch.unique(torch.cat(top_it + top_iq), return_counts=True)
        cand_ids = torch.unique(torch.cat([uniq[torch.topk(cnts, min(4096, uniq.numel())).indices], nxt_ans]))
        W_cand_g = emb_gpu[cand_ids].float()
    id_map = {int(t): ci for ci, t in enumerate(cand_ids.tolist())}
    y_star = torch.tensor([id_map[int(t)] for t in nxt_ans.tolist()], device=device, dtype=torch.long)
    t_top1 = torch.empty((N_ANS,), device=device, dtype=torch.long)
    log_tail_q0 = torch.empty((N_ANS,), device=device)
    p_teach_store = torch.empty((N_ANS, int(cand_ids.numel()) + 1), device="cpu", dtype=torch.float16)
    with torch.no_grad():
        for c0 in range(0, N_ANS, 4096):
            c1 = min(c0 + 4096, N_ANS)
            zt_c = SOFTCAP * torch.tanh((ht_ans[c0:c1] @ W_cand_g.T) / SOFTCAP) / TAU
            zq_c = SOFTCAP * torch.tanh((hs_ans[c0:c1] @ W_cand_g.T) / SOFTCAP) / TAU
            t_top1[c0:c1] = zt_c.argmax(dim=-1)
            ltt = t_lse[c0:c1] + torch.log((1.0 - torch.exp((torch.logsumexp(zt_c, dim=-1) - t_lse[c0:c1]).clamp(max=0.0))).clamp(min=1e-12))
            log_tail_q0[c0:c1] = q_lse[c0:c1] + torch.log((1.0 - torch.exp((torch.logsumexp(zq_c, dim=-1) - q_lse[c0:c1]).clamp(max=0.0))).clamp(min=1e-12))
            p_teach_store[c0:c1] = F.softmax(torch.cat([zt_c, ltt.unsqueeze(1)], dim=-1), dim=-1).to(device="cpu", dtype=torch.float16)
    del t_lse, q_lse, top_it, top_iq, uniq, cnts

    va_a = (torch.arange(N_ANS, device=device) % 5 == 0)
    tr_a = ~va_a
    m60 = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=D_MOD, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(device)
    o60 = torch.optim.AdamW(m60.parameters(), lr=3e-3, weight_decay=3e-2)
    s60 = torch.optim.lr_scheduler.CosineAnnealingLR(o60, T_max=300, eta_min=3e-4)

    def eval_c60(xs, isub):
        with torch.no_grad():
            it = torch.nonzero(isub, as_tuple=False).squeeze(1) if isub.dtype == torch.bool else isub
            kl_s = top_s = agr_s = mar_s = 0.0
            for b0 in range(0, it.numel(), 2048):
                b1 = min(b0 + 2048, it.numel())
                ib = it[b0:b1]
                zc = SOFTCAP * torch.tanh((xs[b0:b1] @ W_cand_g.T) / SOFTCAP)
                pb, ys = zc.argmax(dim=-1), y_star[ib]
                top_s += float((pb == ys).float().sum().item()); agr_s += float((pb == t_top1[ib]).float().sum().item())
                zy = zc.gather(1, ys.unsqueeze(1)).squeeze(1); zm = zc.clone(); zm.scatter_(1, ys.unsqueeze(1), -1e9)
                mar_s += float((zy - zm.max(dim=-1).values).sum().item())
                zw = torch.cat([zc / TAU, log_tail_q0[ib].unsqueeze(1)], dim=1)
                ptb = p_teach_store[ib.cpu()].to(device, dtype=torch.float32)
                kl_s += float(((ptb * (torch.log(ptb.clamp(min=1e-12)) - F.log_softmax(zw, dim=-1))).sum(dim=-1) * (TAU * TAU)).sum().item())
            ns = float(max(it.numel(), 1))
            return kl_s / ns, top_s / ns, agr_s / ns, mar_s / ns

    kl0, acc0, agr0, marg0 = eval_c60(hs_ans[va_a], va_a)
    tr_idx_a = torch.nonzero(tr_a, as_tuple=False).squeeze(1)
    best_sc, best_sd60 = -1e9, None
    for st in range(300):
        o60.zero_grad()
        pb = tr_idx_a[torch.randperm(tr_idx_a.numel(), device=device)[:2048]]
        da, _, lb = m60(hs_ans[pb]); hp = hs_ans[pb] + da
        zc = SOFTCAP * torch.tanh((hp @ W_cand_g.T) / SOFTCAP)
        zw = torch.cat([zc / TAU, log_tail_q0[pb].unsqueeze(1)], dim=1)
        ptb = p_teach_store[pb.cpu()].to(device, dtype=torch.float32)
        zy = zc.gather(1, y_star[pb].unsqueeze(1)).squeeze(1); zm = zc.clone(); zm.scatter_(1, y_star[pb].unsqueeze(1), -1e9)
        loss = (
            ((ptb * (torch.log(ptb.clamp(min=1e-12)) - F.log_softmax(zw, dim=-1))).sum(dim=-1) * (TAU * TAU)).mean()
            + 0.5 * F.relu(2.0 - zy + zm.max(dim=-1).values).mean()
            + 2.0 * F.mse_loss(hp, ht_ans[pb]) + 0.5 * (1.0 - F.cosine_similarity(hp, ht_ans[pb], dim=-1).mean())
            + 0.005 * lb + 2e-2 * (m60.A_bank.pow(2).sum() + m60.B_bank.pow(2).sum()) / float(pb.numel() * D_MOD)
        )
        loss.backward(); torch.nn.utils.clip_grad_norm_(m60.parameters(), 1.0); o60.step(); s60.step()
        if (st + 1) % 60 == 0:
            with torch.no_grad():
                klv, accv, agrv, margv = eval_c60(hs_ans[va_a] + m60(hs_ans[va_a])[0], va_a)
                scv = agrv * 2.0 + margv * 0.5 - 0.5 * klv
                if scv > best_sc:
                    best_sc, best_sd60 = scv, {k: v.detach().cpu().clone() for k, v in m60.state_dict().items()}
    mc60 = m60.cpu()
    if best_sd60 is not None:
        mc60.load_state_dict(best_sd60)
    cents60, experts60 = mc60.export_quantized_arrays()
    kl_q, acc_q, agr_q, marg_q = eval_c60(hs_ans[va_a] + eval_c8d_gpu(hs_ans[va_a], cents60, experts60), va_a)
    keep_c60 = kl_q <= kl0
    print(f"  [post_cell60] Val KL {kl0:.4f}->{kl_q:.4f} | Top1 {acc0*100:.1f}%->{acc_q*100:.1f}% | Agree {agr0*100:.1f}%->{agr_q*100:.1f}% (keep={keep_c60})", flush=True)
    if keep_c60:
        exp_r60 = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
        c8db_entries.append({"name": "model.language_model.norm.post_cell60", "d_out": D_MOD, "d_in": D_MOD, "rank": 32, "top_k": 2, "c6d": exp_r60[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents60, "expert_radicals": exp_r60, "experts": experts60})
    c8d.write_c8db_file(OUT_C8D, [("model.language_model.norm.weight", fn_eff_g.detach().cpu().numpy().astype("<f4"))], c8db_entries)
    del p_teach_store, ht_ans, hs_ans, nxt_ans, y_star, t_top1, log_tail_q0, m60, mc60
    torch.cuda.empty_cache()

    # Stage 4B: Target-Aligned Assistant-V7 + BHDR-384
    print("\n[Stage 4B] Target-Aligning Assistant-V7 + BHDR (d_byte=384)...", flush=True)
    ast_dir = Path(snapshot_download("google/gemma-4-31B-it-assistant", local_dir="/tmp/gemma4_31b_assistant", ignore_patterns=["*.bin", "*.pt"]))
    with safe_open(ast_dir / "model.safetensors", framework="pt", device="cpu") as asf:
        ast_sd = {k: asf.get_tensor(k) for k in asf.keys()}

    class Gemma4AssistantTrain(nn.Module):
        def __init__(self, sd: Dict[str, torch.Tensor]) -> None:
            super().__init__()
            self.pre = nn.Parameter(sd["pre_projection.weight"].float().clone())
            self.post = nn.Parameter(sd["post_projection.weight"].float().clone())
            self.fn = nn.Parameter(sd["model.norm.weight"].float().clone())
            self.register_buffer("emb", sd["model.embed_tokens.weight"].to(compute_dtype), persistent=False)
            self.layers = nn.ModuleList()
            for la in range(4):
                pa_s = f"model.layers.{la}."
                m = nn.Module()
                m.q_proj = nn.Parameter(sd[pa_s + "self_attn.q_proj.weight"].float().clone())
                m.o_proj = nn.Parameter(sd[pa_s + "self_attn.o_proj.weight"].float().clone())
                m.gate = nn.Parameter(sd[pa_s + "mlp.gate_proj.weight"].float().clone())
                m.up = nn.Parameter(sd[pa_s + "mlp.up_proj.weight"].float().clone())
                m.down = nn.Parameter(sd[pa_s + "mlp.down_proj.weight"].float().clone())
                m.n_in = nn.Parameter(sd[pa_s + "input_layernorm.weight"].float().clone())
                m.n_qn = nn.Parameter(sd[pa_s + "self_attn.q_norm.weight"].float().clone())
                m.n_pa = nn.Parameter(sd[pa_s + "post_attention_layernorm.weight"].float().clone())
                m.n_pf = nn.Parameter(sd[pa_s + "pre_feedforward_layernorm.weight"].float().clone())
                m.n_pff = nn.Parameter(sd[pa_s + "post_feedforward_layernorm.weight"].float().clone())
                m.lsc = nn.Parameter(sd[pa_s + "layer_scalar"].float().reshape(1).clone())
                self.layers.append(m)
            cr_loc, sr_loc = make_rope(THETA_L, int(round(HD_S * PART_L)), HD_S)
            cr_glb, sr_glb = make_rope(THETA_G, int(round(HD_F * PART_G)), HD_F)
            self.register_buffer("cr_l", cr_loc, persistent=False); self.register_buffer("sr_l", sr_loc, persistent=False)
            self.register_buffer("cr_g", cr_glb, persistent=False); self.register_buffer("sr_g", sr_glb, persistent=False)

        def forward_seq(self, tok_emb_seq, back_h_seq, k58, v58, k59):
            B, Sm1, _ = tok_emb_seq.shape
            state = F.linear(torch.cat([tok_emb_seq, back_h_seq], dim=-1).to(compute_dtype), self.pre.to(compute_dtype)).float()
            q_mask = torch.tril(torch.ones((Sm1, SEQ_LEN), device=state.device, dtype=torch.bool), diagonal=0).unsqueeze(0).unsqueeze(0)
            for la, m in enumerate(self.layers):
                is_f = (la == 3)
                hda, nkva = (HD_F, NKV_F) if is_f else (HD_S, NKV_S)
                rota = int(round(HD_F * PART_G)) if is_f else int(round(HD_S * PART_L))
                ga = NH // nkva
                inp = rmsn_gpu(state, m.n_in).to(compute_dtype)
                q = F.linear(inp, m.q_proj.to(compute_dtype)).view(B, Sm1, NH, hda)
                cr = (self.cr_g if is_f else self.cr_l)[1 : Sm1 + 1]
                sr = (self.sr_g if is_f else self.sr_l)[1 : Sm1 + 1]
                qn = apply_rope(rmsn_gpu(q, m.n_qn), cr, sr, rota, hda).transpose(1, 2).to(compute_dtype)
                kk = (k59 if is_f else k58).to(compute_dtype).repeat_interleave(ga, dim=1)
                vv = (k59 if is_f else v58).to(compute_dtype).repeat_interleave(ga, dim=1)
                ctx = F.scaled_dot_product_attention(qn, kk, vv, attn_mask=q_mask, scale=1.0)
                state = state + rmsn_gpu(F.linear(ctx.transpose(1, 2).contiguous().view(B, Sm1, NH * hda), m.o_proj.to(compute_dtype)), m.n_pa).float()
                hm = rmsn_gpu(state, m.n_pf).to(compute_dtype)
                act = (F.gelu(F.linear(hm, m.gate.to(compute_dtype)).float(), approximate="tanh") * F.linear(hm, m.up.to(compute_dtype)).float()).to(compute_dtype)
                state = (state + rmsn_gpu(F.linear(act, m.down.to(compute_dtype)), m.n_pff).float()) * m.lsc
            return rmsn_gpu(state, self.fn).float()

    ast_mod = Gemma4AssistantTrain(ast_sd).to(device)
    with torch.no_grad():
        ast_h_final = torch.empty((len(AST_SEQS), SEQ_LEN, D_MOD), dtype=torch.float16)
        for ai in range(len(AST_SEQS)):
            hf = rmsn_gpu(ast_x59_stud[ai].to(device), fn_eff_g).float()
            if keep_c60:
                hf = hf + eval_c8d_gpu(hf, cents60, experts60)
            ast_h_final[ai] = hf.to(torch.float16).cpu()
        W_ast_cand = ast_mod.emb[cand_ids].float()
    del ast_x59_stud

    def eval_ast():
        with torch.no_grad():
            agrs = kls = 0.0; cnt = 0
            for ai in range(112, 128):
                gi = AST_SEQS[ai]
                te = (emb_gpu[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE).unsqueeze(0)
                ha = ast_mod.forward_seq(te, ast_h_final[ai : ai + 1, :-1].to(device).float(), cap_kv58["k_rope"][gi].unsqueeze(0).to(device), cap_kv58["v_norm"][gi].unsqueeze(0).to(device), cap_kv59["k_rope"][gi].unsqueeze(0).to(device)).squeeze(0)
                zs = SOFTCAP * torch.tanh((ast_h_final[ai, 1:].to(device).float() @ W_cand_g.T) / SOFTCAP)
                za = ha @ W_ast_cand.T
                ps = F.softmax(zs / TAU, dim=-1)
                kls += float(((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU, dim=-1))).sum(dim=-1) * (TAU * TAU)).sum().item())
                agrs += float((za.argmax(dim=-1) == zs.argmax(dim=-1)).float().sum().item())
                cnt += ha.shape[0]
            return kls / max(cnt, 1), agrs / max(cnt, 1)

    akl0, aagr0 = eval_ast()
    opt_a = torch.optim.AdamW(ast_mod.parameters(), lr=2.5e-4, weight_decay=1e-4)
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=300, eta_min=2e-5)
    best_aagr, best_asd = aagr0, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_bs = int(hw_cfg["ast_bs"])
    for st in range(300):
        opt_a.zero_grad()
        bais = np.random.choice(112, size=ast_bs, replace=False).tolist()
        te_l, bh_l, k58_l, v58_l, k59_l, ht_l = [], [], [], [], [], []
        for ai in bais:
            gi = AST_SEQS[ai]
            te_l.append(emb_gpu[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE)
            bh_l.append(ast_h_final[ai, :-1].to(device).float())
            k58_l.append(cap_kv58["k_rope"][gi].to(device)); v58_l.append(cap_kv58["v_norm"][gi].to(device)); k59_l.append(cap_kv59["k_rope"][gi].to(device))
            ht_l.append(ast_h_final[ai, 1:].to(device).float())
        ha = ast_mod.forward_seq(torch.stack(te_l), torch.stack(bh_l), torch.stack(k58_l), torch.stack(v58_l), torch.stack(k59_l))
        hst = torch.stack(ht_l)
        with torch.no_grad():
            zs = SOFTCAP * torch.tanh((hst.reshape(-1, D_MOD) @ W_cand_g.T) / SOFTCAP)
            ps, ys_t1 = F.softmax(zs / TAU, dim=-1), zs.argmax(dim=-1)
        za = ha.reshape(-1, 1024) @ W_ast_cand.T
        hpost = F.linear(ha.to(compute_dtype), ast_mod.post.to(compute_dtype)).float()
        loss_a = (
            ((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU, dim=-1))).sum(dim=-1) * (TAU * TAU)).mean()
            + 0.5 * F.cross_entropy(za, ys_t1)
            + 0.5 * (F.mse_loss(hpost, hst) / (hst.var() + 1e-5))
        )
        loss_a.backward(); torch.nn.utils.clip_grad_norm_(ast_mod.parameters(), 1.0); opt_a.step(); sch_a.step()
        if (st + 1) % 60 == 0:
            klv, agrv = eval_ast()
            print(f"    [Ast step {st+1:03d}/300] Val KL={klv:.4f} Top1Agree={agrv*100:.1f}%", flush=True)
            if agrv > best_aagr:
                best_aagr, best_asd = agrv, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_mod.load_state_dict({k: v.to(device) for k, v in best_asd.items()})
    akl1, aagr1 = eval_ast()
    print(f"  [Assistant V7.5 Aligned] Val KL={akl0:.4f}->{akl1:.4f} | Top1Agree={aagr0*100:.1f}%->{aagr1*100:.1f}%", flush=True)

    ast_sd["pre_projection.weight"] = ast_mod.pre.detach().to(torch.bfloat16).cpu()
    ast_sd["post_projection.weight"] = ast_mod.post.detach().to(torch.bfloat16).cpu()
    ast_sd["model.norm.weight"] = ast_mod.fn.detach().to(torch.bfloat16).cpu()
    for la, m in enumerate(ast_mod.layers):
        pa_s = f"model.layers.{la}."
        for k_s, par in [("self_attn.q_proj.weight", m.q_proj), ("self_attn.o_proj.weight", m.o_proj), ("mlp.gate_proj.weight", m.gate), ("mlp.up_proj.weight", m.up), ("mlp.down_proj.weight", m.down), ("input_layernorm.weight", m.n_in), ("self_attn.q_norm.weight", m.n_qn), ("post_attention_layernorm.weight", m.n_pa), ("pre_feedforward_layernorm.weight", m.n_pf), ("post_feedforward_layernorm.weight", m.n_pff)]:
            ast_sd[pa_s + k_s] = par.detach().to(torch.bfloat16).cpu()
        ast_sd[pa_s + "layer_scalar"] = m.lsc.detach().to(torch.bfloat16).cpu().reshape(ast_sd[pa_s + "layer_scalar"].shape)
    ft_sf = Path("/tmp/assistant_v7_5_ft.safetensors")
    st_save_file(ast_sd, str(ft_sf))
    for bits, bp in [(2, OUT_AST_Q2), (4, OUT_AST_Q4), (8, OUT_AST_Q8)]:
        bhmod.pack_assistant_body_from_safetensors(ft_sf, bp, body_bits=bits, block_size=64, has_fwht=(bits == 4))
    ft_sf.unlink(missing_ok=True)
    del opt_a, sch_a, ast_sd, best_asd, W_cand_g, W_ast_cand

    h_ast_chunks = []
    with torch.no_grad():
        for b0 in range(0, len(AST_SEQS), 32):
            bais = list(range(b0, min(b0 + 32, len(AST_SEQS))))
            gis = [AST_SEQS[ai] for ai in bais]
            te_s = torch.stack([emb_gpu[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE for gi in gis])
            bh_s = ast_h_final[b0 : b0 + len(bais), :-1].to(device).float()
            k58_s = torch.stack([cap_kv58["k_rope"][gi].to(device) for gi in gis])
            v58_s = torch.stack([cap_kv58["v_norm"][gi].to(device) for gi in gis])
            k59_s = torch.stack([cap_kv59["k_rope"][gi].to(device) for gi in gis])
            h_ast_chunks.append(ast_mod.forward_seq(te_s, bh_s, k58_s, v58_s, k59_s)[:, :-1, :].cpu())
    del ast_mod, cap_kv58, cap_kv59; torch.cuda.empty_cache()

    H_ast_all = torch.cat(h_ast_chunks, dim=0).reshape(-1, 1024)
    H_stud_tgt = ast_h_final[:, 1:-1, :].reshape(-1, D_MOD)
    top_ids_t = torch.empty((H_stud_tgt.shape[0], 64), dtype=torch.int32)
    top_probs_t = torch.empty((H_stud_tgt.shape[0], 64), dtype=torch.float32)
    with torch.no_grad():
        for i0 in range(0, H_stud_tgt.shape[0], 512):
            i1 = min(i0 + 512, H_stud_tgt.shape[0])
            lg_b = SOFTCAP * torch.tanh((H_stud_tgt[i0:i1].to(device, dtype=compute_dtype) @ emb_gpu.T).float() / SOFTCAP)
            tv, ti = torch.topk(lg_b, k=64, dim=-1)
            top_probs_t[i0:i1] = F.softmax(tv, dim=-1).cpu()
            top_ids_t[i0:i1] = ti.to(torch.int32).cpu()
    trie = bhmod.BpePrefixTrie.from_tokenizer_json(tok_path)
    student_argmax_ids = torch.tensor(v70.select_first_valid_trie_ids(top_ids_t, trie), dtype=torch.long)
    del H_stud_tgt, ast_h_final

    act_bundle = Path("/tmp/v7_5_bhdr_activations.pt")
    torch.save({"h_final": H_ast_all.half(), "gold_ids": student_argmax_ids, "top_ids": top_ids_t, "top_probs": top_probs_t}, act_bundle)
    bh_rpt = bhmod.train_from_activations(
        act_bundle, tok_path, OUT_BHDR,
        d_byte=384, max_bytes=48, steps=2400, batch_size=int(hw_cfg["bh_bs"]), lr=3e-3, device=str(device),
    )
    act_bundle.unlink(missing_ok=True)
    bh_loaded = bhmod.load_bhdr_v1(OUT_BHDR)
    val_h = H_ast_all[-256:].numpy().astype(np.float32)
    val_g = student_argmax_ids[-256:].tolist()
    beam_hits = sum(1 for hv, gv in zip(val_h, val_g) if bhmod.decode_bhdr_np_beam(bh_loaded, trie, hv, beam_width=2)[0] == gv)
    bh_metrics = {**bh_rpt["metrics"], "heldout_beam_top1_acc_vs_v75_argmax": round(beam_hits / 256.0, 4), "heldout_beam_samples": 256}
    del H_ast_all, student_argmax_ids, top_ids_t, top_probs_t

    # Stage 5: Strictly Held-Out COCO val[0..3]
    heldout_vlm_report = []
    with torch.no_grad():
        for hi, cap_gt in enumerate(heldout_captions):
            p0, plen = heldout_ans_starts[hi], heldout_ans_lens[hi]
            ht_hi = rmsn_gpu(H_teach_eval[hi, p0 : p0 + plen].to(device).unsqueeze(0), fn_g).squeeze(0).float()
            hs_hi = rmsn_gpu(H_stud_eval[hi, p0 : p0 + plen].to(device).unsqueeze(0), fn_eff_g).squeeze(0).float()
            if keep_c60:
                hs_hi = hs_hi + eval_c8d_gpu(hs_hi, cents60, experts60)
            h_cos_hi = float(F.cosine_similarity(hs_hi, ht_hi, dim=-1).mean().item())
            t_bv, t_bi = torch.full((plen, 5), -1e9, device=device), torch.zeros((plen, 5), device=device, dtype=torch.long)
            s_bv, s_bi = torch.full((plen, 5), -1e9, device=device), torch.zeros((plen, 5), device=device, dtype=torch.long)
            for r0 in range(0, emb_gpu.shape[0], 16384):
                wc = emb_gpu[r0 : r0 + 16384].float()
                tv, ti = torch.topk(SOFTCAP * torch.tanh((ht_hi @ wc.T) / SOFTCAP), 5, dim=-1)
                sv, si = torch.topk(SOFTCAP * torch.tanh((hs_hi @ wc.T) / SOFTCAP), 5, dim=-1)
                t_bv, p1 = torch.topk(torch.cat([t_bv, tv], -1), 5, dim=-1); t_bi = torch.cat([t_bi, ti + r0], -1).gather(1, p1)
                s_bv, p2 = torch.topk(torch.cat([s_bv, sv], -1), 5, dim=-1); s_bi = torch.cat([s_bi, si + r0], -1).gather(1, p2)
            t1, s1 = t_bi[:, 0], s_bi[:, 0]
            heldout_vlm_report.append({
                "val_idx": hi, "gt_caption": cap_gt, "ans_cos": round(h_cos_hi, 5),
                "ans_top1_agree": round(float((s1 == t1).float().mean().item()), 4),
                "ans_top5_agree": round(float((s_bi == t1.unsqueeze(1)).any(dim=-1).float().mean().item()), 4),
                "student_forced_pred": tk.decode([int(x) for x in s1.tolist()]),
            })
            print(f"  [Held-Out val[{hi}]] cos={h_cos_hi:.5f} top1={heldout_vlm_report[-1]['ans_top1_agree']*100:.1f}% | {heldout_vlm_report[-1]['student_forced_pred']!r}", flush=True)
    del emb_gpu; torch.cuda.empty_cache()

    all_artifacts = all_t16c_paths + [
        OUT_CRTR, OUT_C8D, OUT_DISTRICT_ROUTER, OUT_DISTRICT_MANIFEST,
        OUT_BHDR, OUT_AST_Q2, OUT_AST_Q4, OUT_AST_Q8,
    ]
    artifacts: Dict[str, dict] = {}
    sha_lines: List[str] = []
    for fp in all_artifacts:
        sz_f, dig_f = persist_v7_5_file(fp)
        artifacts[fp.name] = {"bytes": sz_f, "sha256": dig_f}
        sha_lines.append(f"{dig_f}  {fp.name}")

    crtr_vals = [r["crtr"] for r in layer_cos_log]
    lod_layers_log = [d for d in district_manifest if d["is_lod_layer"]]
    report = {
        "version": "V7.5-SPATIAL-DISTRICTS-GTA-LOD-60L-A100",
        "gpu": gpu_name,
        "hw_profile": hw_cfg,
        "elapsed_s": round(time.perf_counter() - t0_all, 2),
        "base_t16c_bytes": base_sz,
        "base_t16c_gib": round(base_sz / (1024**3), 4),
        "lod_shared_bytes": OUT_LOD_SHARED.stat().st_size,
        "lod_district_bytes": {p.name: p.stat().st_size for p in OUT_LOD_DISTRICTS},
        "mean_native_capture": round(float(np.mean([d["cap_native"] for d in lod_layers_log])), 4),
        "mean_district_router_top2_recall": round(float(np.mean([d["router"]["val_top2_recall_after"] for d in lod_layers_log])), 4),
        "mean_crtr_val_recall": round(float(np.mean([c["val_recall_after"] for c in crtr_vals])), 5),
        "mean_mod_val_acc": round(float(np.mean([c["mod_val_acc"] for c in crtr_vals])), 5),
        "vision_val_cos": [round(v_cos0, 5), round(v_cos1, 5)],
        "l59_vlm_state_cos": round(cos_l59, 5),
        "post_cell60_val_kl": [round(kl0, 4), round(kl_q, 4)],
        "post_cell60_val_top1": [round(acc0, 4), round(acc_q, 4)],
        "post_cell60_val_agree": [round(agr0, 4), round(agr_q, 4)],
        "assistant_v7_align_kl": [round(akl0, 4), round(akl1, 4)],
        "assistant_v7_align_top1_agree": [round(aagr0, 4), round(aagr1, 4)],
        "assistant_v7_bhdr": bh_metrics,
        "heldout_coco_val0_3": heldout_vlm_report,
        "layer_cos_summary": [r for r in layer_cos_log if r["l"] in (0, 2, 5, 10, 20, 30, 40, 50, 55, 57, 58, 59)],
        "artifacts": artifacts,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _, r_dig = persist_v7_5_file(OUT_REPORT)
    sha_lines.append(f"{r_dig}  {OUT_REPORT.name}")
    OUT_SHA256.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    persist_v7_5_file(OUT_SHA256)
    for rmp in (VIS_CKPT_PATH, VIS_META_PATH, VLM_SOFT_CACHE, STAGE3_CKPT, CRTR_CKPT, D_ROUTER_CKPT, CKPT_X_TEACH, CKPT_X_STUD):
        rmp.unlink(missing_ok=True)
    if DRIVE_DIR is not None:
        shutil.rmtree(DRIVE_DIR / "ckpt_v7_5", ignore_errors=True)
        for sp_k in all_t16c_paths:
            (DRIVE_DIR / (sp_k.name + ".partial")).unlink(missing_ok=True)
    print(f"[V7.5 COMPLETE] Elapsed={(time.perf_counter()-t0_all)/60:.1f} min | Report={OUT_REPORT}", flush=True)

    if os.environ.get("COLAB_AUTO_UNASSIGN", "1") == "1":
        try:
            from google.colab import runtime
            print("[Auto-Unassign] Releasing Colab runtime...", flush=True)
            runtime.unassign()
        except Exception as exc:
            print(f"[Auto-Unassign Note] {exc}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemma 4 31B V7.5 60-Layer Spatial Districts + GTA LOD Calibrator")
    ap.add_argument("--selftest", action="store_true", help="Run 6-gate CPU self-verification battery and exit.")
    ap.add_argument("--run", action="store_true", help="Run full 60-layer A100-SXM4-80GB calibration & packaging.")
    args = ap.parse_args()
    if args.selftest or (not args.run and not torch.cuda.is_available()):
        return run_selftest()
    run_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

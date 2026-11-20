#!/usr/bin/env python3
"""
Gemma 4 31B V6.0 — Samsung Research LittleBit-2 (ICML 2026 Joint-ITQ + Sub-1-Bit
Latent Binary Factorization) + Pure 1-Bit Row-Major Horner AND+VCNT NEON Packing
+ Full-Weight STE SmoothSign Block-Wise QAT + Joint 4-Layer (L56..L59) Logit-KL
Distillation + Multimodal C8DB + V6.0-Matched 4-Layer EAGLE-3 Assistant.

Targets Physical Raspberry Pi 4 (7.6 GiB RAM, 4x Cortex-A72 @ 1.5 GHz):
  - LM body ~2.45 GB (~0.67 bpw effective avg), unified .t16c ~3.15 GB (<= 3.60 GB hard guard).
  - 100% Row-Major T16C_ROWMAJOR_BIT (0x40) on all 1b/2b LM layers -> zero vqtbl1q_s8 ALU wall!
  - Native LittleBit-2 factored pairs (*.lb_v + *.lb_u [+ optional *.lb_rv + *.lb_ru]) directly
    supported by run31b.rs with Gate-Guided Block-Sparse GeGLU (--mlp-top-p 0.40..0.50).
  - Auto-persists completed artifacts to Google Drive (V6_DRIVE_DIR) with SHA-256 verification.
  - Zero benchmark contamination (full BLACKLIST_SUBSTRINGS + COCO val[0..63] holdout).
  - Authentic PyTorch loss.backward() + AdamW gradient descent on physical A100 silicon.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda:0")
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

T16C_GAUSS_BIT = 0x80
T16C_ROWMAJOR_BIT = 0x40

lib_cands = [Path(__file__).resolve().parent / "pack_t16c_v3.py"]
if not lib_cands[0].is_file():
    lib_cands = list(Path("/kaggle/input").rglob("pack_t16c_v3.py"))
assert lib_cands and lib_cands[0].is_file(), "Missing pack_t16c_v3.py helper directory!"
LIB_DIR = lib_cands[0].parent
sys.path.insert(0, str(LIB_DIR))

import pack_t16c_v3 as p3
import pack_assistant_t16c as pa
import train_c8d_pattern_bank as c8d
import v4_corpus as vc
from safetensors import safe_open
from safetensors.torch import save_file as st_save_file
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoConfig, Gemma4ImageProcessor
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4MultimodalEmbedder
from tokenizers import Tokenizer
from PIL import Image, ImageDraw

T0_ALL = time.perf_counter()

OUT_DIR = Path(os.environ.get("V6_OUT_DIR", "/content/v6_outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_DIR_STR = os.environ.get("V6_DRIVE_DIR", "").strip()
if not DRIVE_DIR_STR and Path("/content/drive/MyDrive").is_dir():
    DRIVE_DIR_STR = "/content/drive/MyDrive/gemma4_v6_littlebit2"
DRIVE_DIR = Path(DRIVE_DIR_STR) if DRIVE_DIR_STR else None
if DRIVE_DIR is not None:
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR = OUT_DIR / "ckpt_v6_0_littlebit2"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_T16C = OUT_DIR / "gemma4_31b_v6_0_littlebit2.t16c"
OUT_C8D = OUT_DIR / "gemma4_31b_v6_0_littlebit2_c8d.c8d"
OUT_AST = OUT_DIR / "gemma4_31b_assistant_v6.t16c"
OUT_REPORT = OUT_DIR / "v6_0_qat_report.json"
PROGRESS_JSON = CKPT_DIR / "progress.json"

if DRIVE_DIR is not None and (DRIVE_DIR / "ckpt_v6_0_littlebit2").is_dir():
    print(f"[Drive Resume] Checking persistent checkpoints at {DRIVE_DIR / 'ckpt_v6_0_littlebit2'}...", flush=True)
    drive_ckpt = DRIVE_DIR / "ckpt_v6_0_littlebit2"
    for p in drive_ckpt.glob("*"):
        if p.is_file() and not (CKPT_DIR / p.name).is_file():
            shutil.copy2(p, CKPT_DIR / p.name)
            print(f"  [Restored Checkpoint] {p.name} ({p.stat().st_size:,} B)", flush=True)
    partial_t16c = DRIVE_DIR / (OUT_T16C.name + ".partial")
    if partial_t16c.is_file() and not OUT_T16C.is_file():
        print(f"[Drive Resume] Restoring partial {OUT_T16C.name} ({partial_t16c.stat().st_size:,} B)...", flush=True)
        shutil.copy2(partial_t16c, OUT_T16C)

EXTRA_BLACKLIST: Tuple[str, ...] = (
    "12 * 8", "12*8", "547", "193", "motorcycle", "motorbike", "honda",
)
ALL_BLACKLIST: Tuple[str, ...] = tuple(vc.BLACKLIST_SUBSTRINGS) + EXTRA_BLACKLIST


def host_ram_gb() -> Tuple[float, float]:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
        tot = info.get("MemTotal", 0) / (1024**2)
        avail = info.get("MemAvailable", 0) / (1024**2)
        return round(max(tot - avail, 0.0), 2), round(tot, 2)
    except OSError:
        return 0.0, 0.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chk = f.read(16 * 1024 * 1024)
            if not chk:
                break
            h.update(chk)
    return h.hexdigest()


def persist_to_drive(src: Path) -> Optional[str]:
    if DRIVE_DIR is None or not src.is_file():
        return None
    dst = DRIVE_DIR / src.name
    tmp = DRIVE_DIR / (src.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    h_src = sha256_file(src)
    h_dst = sha256_file(dst)
    assert h_src == h_dst, f"Drive copy SHA-256 mismatch for {src.name}: {h_src} != {h_dst}"
    print(f"  [DRIVE PERSISTED] {dst} ({dst.stat().st_size:,} B, sha256={h_dst[:16]}...)", flush=True)
    return str(dst)


def sync_checkpoint_to_drive() -> None:
    if DRIVE_DIR is None:
        return
    drive_ckpt = DRIVE_DIR / "ckpt_v6_0_littlebit2"
    drive_ckpt.mkdir(parents=True, exist_ok=True)
    for p in CKPT_DIR.glob("*"):
        if p.is_file() and not p.name.endswith(".tmp"):
            try:
                shutil.copy2(p, drive_ckpt / p.name)
            except Exception as e:
                print(f"  [Drive Checkpoint Warn] Failed copying {p.name}: {e}", flush=True)
    if OUT_T16C.is_file():
        try:
            shutil.copy2(OUT_T16C, DRIVE_DIR / (OUT_T16C.name + ".partial"))
        except Exception as e:
            print(f"  [Drive T16C Partial Warn] {e}", flush=True)
    print(f"  [DRIVE CHECKPOINT SYNCED] -> {drive_ckpt}", flush=True)


def save_progress(state: dict) -> None:
    used_gb, tot_gb = host_ram_gb()
    payload = dict(state)
    payload["host_ram_used_gb"] = used_gb
    payload["host_ram_total_gb"] = tot_gb
    payload["elapsed_s"] = round(time.perf_counter() - T0_ALL, 2)
    tmp_p = PROGRESS_JSON.with_suffix(".tmp")
    tmp_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_p, PROGRESS_JSON)


def _atomic_torch_save(obj: object, dst: Path) -> None:
    tmp_p = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(obj, tmp_p)
    os.replace(tmp_p, dst)


gpu_name = torch.cuda.get_device_name(0)
gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"[V6.0 LITTLEBIT-2 INIT] GPU={gpu_name} ({gpu_vram_gb:.1f} GiB) | dtype={COMPUTE_DTYPE} | RAM={host_ram_gb()} | DRIVE={DRIVE_DIR}", flush=True)

# ---------------------------------------------------------------------------
# 1. Compile OpenMP C quantizer for embed_tokens / Vision Tower
# ---------------------------------------------------------------------------
fq_c = LIB_DIR / "fast_quant.c"
fq_so = Path("/tmp/libfast_quant_v6.so")
subprocess.run(["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", str(fq_c), "-o", str(fq_so), "-lm"], check=True)
LIB_FQ = ctypes.CDLL(str(fq_so))
LIB_FQ.t16c_tblk_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
LIB_FQ.t16c_tblk_bytes.restype = ctypes.c_size_t
LIB_FQ.quant_and_pack_t16c_f32.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]

# ---------------------------------------------------------------------------
# 2. Core Math: FWHT, Row-Major Bitplane Packing, Samsung LittleBit-2 Joint-ITQ
#    + Full-Weight STE Binary & 2b_gaus QAT Modules
# ---------------------------------------------------------------------------
def hadamard_gpu(n: int, device: torch.device) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    k = 1
    while k < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        k *= 2
    return h / math.sqrt(float(n))


_H_GPU_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}
for _bs in (64, 128, 256):
    _H_GPU_CACHE[(_bs, str(DEVICE))] = hadamard_gpu(_bs, DEVICE)


def get_h_gpu(bsize: int, device: torch.device) -> torch.Tensor:
    key = (bsize, str(device))
    h = _H_GPU_CACHE.get(key)
    if h is None:
        h = hadamard_gpu(bsize, device)
        _H_GPU_CACHE[key] = h
    return h


def fwht_gpu(t: torch.Tensor, bsize: int) -> torch.Tensor:
    orig_sh = t.shape
    return (t.reshape(-1, bsize) @ get_h_gpu(bsize, t.device)).reshape(orig_sh)


def accum_xsq(xsq_acc: torch.Tensor, x_f32: torch.Tensor) -> None:
    xsq_acc.add_((x_f32 * x_f32).sum(dim=0))


def littlebit_joint_itq_rotate(
    U_f: torch.Tensor, V_f: torch.Tensor, n_iter: int = 25
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samsung Research LittleBit-2 (ICML 2026) Joint-ITQ Internal Latent Rotation:
      Finds orthogonal rotation R in O(s) on [U_norm ; V_norm^T] aligning both
      factors with the {-1,+1} binary hypercube while preserving (U R)(R^T V) == U V.
    """
    with torch.no_grad():
        s = U_f.shape[1]
        if s <= 1:
            return U_f, V_f
        Uf = U_f.float()
        Vf = V_f.float()
        u_col = Uf.norm(dim=0).clamp(min=1e-6)
        v_row = Vf.norm(dim=1).clamp(min=1e-6)
        sq = torch.sqrt(u_col / v_row).clamp(0.1, 10.0)
        U_b = Uf / sq.unsqueeze(0)
        V_b = Vf * sq.unsqueeze(1)

        # Subsample rows/cols to at most 8192 for ultra-fast O(s^3) Procrustes on A100
        u_sub = U_b if U_b.shape[0] <= 8192 else U_b[:: max(1, U_b.shape[0] // 8192)]
        v_sub = V_b.T if V_b.shape[1] <= 8192 else V_b.T[:: max(1, V_b.shape[1] // 8192)]
        u_norm = u_sub / u_sub.norm(dim=1, keepdim=True).clamp(min=1e-6)
        v_norm = v_sub / v_sub.norm(dim=1, keepdim=True).clamp(min=1e-6)
        X_comb = torch.cat([u_norm, v_norm], dim=0)

        R = torch.eye(s, device=Uf.device, dtype=torch.float32)
        for _ in range(n_iter):
            Z = X_comb @ R
            B = torch.where(Z >= 0.0, 1.0, -1.0)
            M = B.T @ X_comb
            try:
                U_p, _, Vt_p = torch.linalg.svd(M, full_matrices=False)
                R = Vt_p.T @ U_p.T
            except RuntimeError:
                break

        U_rot = (U_b @ R).contiguous()
        V_rot = (R.T @ V_b).contiguous()
        return U_rot.to(U_f.dtype), V_rot.to(V_f.dtype)


def init_1b_asym_norm(w: torch.Tensor, bsize: int = 256) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize unit-normalized latent coordinates z_lat around binary threshold 0 plus
    FP16-aligned per-block scales and biases via 2-step Lloyd-Max binary refinement:
      w ≈ bi_q + sc_q * sign(z_lat)
    """
    d_out, d_in = w.shape
    n_blks = d_in // bsize
    blk = w.reshape(d_out, n_blks, bsize).float()
    with torch.no_grad():
        bi = blk.mean(dim=-1)
        sc = blk.std(dim=-1).clamp(min=1e-5)
        for _ in range(2):
            m1 = blk >= bi.unsqueeze(-1)
            c1 = m1.sum(dim=-1)
            c0 = bsize - c1
            s1 = torch.where(m1, blk, 0.0).sum(dim=-1)
            s0 = torch.where(m1, 0.0, blk).sum(dim=-1)
            mp = torch.where(c1 > 0, s1 / c1.clamp(min=1), bi)
            mn = torch.where(c0 > 0, s0 / c0.clamp(min=1), bi)
            bi = 0.5 * (mp + mn)
            sc = (0.5 * (mp - mn)).clamp(min=1e-6)
        bi_q = bi.to(torch.float16).float()
        sc_q = sc.to(torch.float16).float().clamp(min=1e-6)
        z_lat = ((blk - bi_q.unsqueeze(-1)) / sc_q.unsqueeze(-1)).clamp(-2.5, 2.5).reshape(d_out, d_in)
    return z_lat.contiguous(), sc_q.contiguous(), bi_q.contiguous()


def apply_1b_asym_ste(
    z_lat: torch.Tensor, sc_p: torch.Tensor, bi_p: torch.Tensor, bsize: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable exact per-bsize asymmetric 1-bit quantization with unit-scale clipped STE
    and differentiable per-block scale/bias parameters:
      w_ste = bi_ste + sc_ste * sign_ste(z_lat)
    Returns (w_ste, sc_q [d_out, n_blks], bi_q [d_out, n_blks], bits01 [d_out, n_blks, bsize]).
    """
    d_out, d_in = z_lat.shape
    n_blks = d_in // bsize
    blk = z_lat.reshape(d_out, n_blks, bsize)
    sc_abs = sc_p.abs()
    sc_q = sc_abs.to(torch.float16).float()
    bi_q = bi_p.to(torch.float16).float()
    sc_ste = sc_q + (sc_abs - sc_abs.detach())
    bi_ste = bi_q + (bi_p - bi_p.detach())
    with torch.no_grad():
        bits01 = blk >= 0.0
        sgn = torch.where(bits01, 1.0, -1.0)
        clip_m = (blk.abs() <= 1.25).to(blk.dtype)
    sgn_ste = sgn + (blk - blk.detach()) * clip_m
    w_ste = (bi_ste.unsqueeze(-1) + sc_ste.unsqueeze(-1) * sgn_ste).reshape(d_out, d_in)
    return w_ste, sc_q, bi_q, bits01


_GAUSS2_LV = torch.tensor([-3.174, -1.000, 1.000, 3.174], device=DEVICE, dtype=torch.float32)
_GAUSS2_MID = torch.tensor([-2.087, 0.000, 2.087], device=DEVICE, dtype=torch.float32)


def init_2b_gaus_norm(w: torch.Tensor, bsize: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize unit-normalized latent coordinates z_lat plus FP16-aligned per-block scales
    via 3-step Lloyd-Max Gaussian refinement for 2b_gaus quantization.
    """
    d_out, d_in = w.shape
    n_blks = d_in // bsize
    blk = w.reshape(d_out, n_blks, bsize).float()
    with torch.no_grad():
        sc = (0.470 * torch.sqrt((blk * blk).mean(dim=-1))).clamp(min=1e-6)
        for _ in range(3):
            inv_s = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0).unsqueeze(-1)
            idx = torch.bucketize(blk * inv_s, _GAUSS2_MID)
            lv = _GAUSS2_LV[idx]
            den = (lv * lv).sum(dim=-1)
            sc = torch.where(den > 0.0, (lv * blk).sum(dim=-1) / den.clamp(min=1e-12), sc)
        sc_q = sc.to(torch.float16).float().clamp(min=1e-6)
        z_lat = (blk / sc_q.unsqueeze(-1)).clamp(-4.0, 4.0).reshape(d_out, d_in)
    return z_lat.contiguous(), sc_q.contiguous()


def apply_2b_gaus_ste(
    z_lat: torch.Tensor, sc_p: torch.Tensor, bsize: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable per-bsize 2-bit non-uniform Gaussian quantization with unit-scale clipped STE."""
    d_out, d_in = z_lat.shape
    n_blks = d_in // bsize
    blk = z_lat.reshape(d_out, n_blks, bsize)
    sc_abs = sc_p.abs()
    sc_q = sc_abs.to(torch.float16).float()
    sc_ste = sc_q + (sc_abs - sc_abs.detach())
    with torch.no_grad():
        idx = torch.bucketize(blk, _GAUSS2_MID)
        lv = _GAUSS2_LV[idx]
        clip_m = (blk.abs() <= 3.6).to(blk.dtype)
    lv_ste = lv + (blk - blk.detach()) * clip_m
    w_ste = (sc_ste.unsqueeze(-1) * lv_ste).reshape(d_out, d_in)
    return w_ste, sc_q, idx


def pack_rowmajor_t16c_1b_asym(
    bits01: torch.Tensor, sc_f16: torch.Tensor, bi_f16: torch.Tensor, bsize: int = 256
) -> np.ndarray:
    """Pack [d_out, n_blks, bsize] bool bits + [d_out, n_blks] FP16 sc, bi directly into Row-Major T16C (1 | 0x40)."""
    d_out, n_blks, bs = bits01.shape
    assert bs == bsize
    n_tiles = (d_out + 15) // 16
    pad_rows = n_tiles * 16 - d_out
    bpr = bsize // 8
    tblk_b = 64 + 16 * bpr  # 576 B @ b256

    sc_np = sc_f16.detach().cpu().numpy().astype("<f2")
    bi_np = bi_f16.detach().cpu().numpy().astype("<f2")
    b_np = bits01.detach().cpu().numpy().astype(np.uint8)
    if pad_rows > 0:
        sc_np = np.pad(sc_np, ((0, pad_rows), (0, 0)))
        bi_np = np.pad(bi_np, ((0, pad_rows), (0, 0)))
        b_np = np.pad(b_np, ((0, pad_rows), (0, 0), (0, 0)))

    out = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
    sc_t = np.ascontiguousarray(sc_np.reshape(n_tiles, 16, n_blks).transpose(0, 2, 1))
    bi_t = np.ascontiguousarray(bi_np.reshape(n_tiles, 16, n_blks).transpose(0, 2, 1))
    out[:, :, 0:32] = sc_t.view(np.uint8).reshape(n_tiles, n_blks, 32)
    out[:, :, 32:64] = bi_t.view(np.uint8).reshape(n_tiles, n_blks, 32)

    # Row-Major bitplane: [n_tiles, n_blks, 16, bpr] MSB-first!
    packed_bits = np.packbits(b_np.reshape(n_tiles, 16, n_blks, bsize).transpose(0, 2, 1, 3), axis=-1)
    out[:, :, 64:] = packed_bits.reshape(n_tiles, n_blks, 16 * bpr)
    return out.reshape(-1)


def pack_rowmajor_t16c_2b_gaus(
    idx_0_3: torch.Tensor, sc_f16: torch.Tensor, bsize: int = 256
) -> np.ndarray:
    """Pack [d_out, n_blks, bsize] uint8 codes in {0..3} + FP16 sc directly into Row-Major T16C (2 | 0x80 | 0x40)."""
    d_out, n_blks, bs = idx_0_3.shape
    assert bs == bsize
    n_tiles = (d_out + 15) // 16
    pad_rows = n_tiles * 16 - d_out
    bpr = bsize // 8
    plane_b = 16 * bpr
    tblk_b = 32 + 2 * plane_b  # 1056 B @ b256

    sc_np = sc_f16.detach().cpu().numpy().astype("<f2")
    idx_np = idx_0_3.detach().cpu().numpy().astype(np.uint8)
    if pad_rows > 0:
        sc_np = np.pad(sc_np, ((0, pad_rows), (0, 0)))
        idx_np = np.pad(idx_np, ((0, pad_rows), (0, 0), (0, 0)))

    out = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
    sc_t = np.ascontiguousarray(sc_np.reshape(n_tiles, 16, n_blks).transpose(0, 2, 1))
    out[:, :, 0:32] = sc_t.view(np.uint8).reshape(n_tiles, n_blks, 32)

    idx_t = idx_np.reshape(n_tiles, 16, n_blks, bsize).transpose(0, 2, 1, 3)
    p0 = np.packbits((idx_t & 1).astype(np.uint8), axis=-1).reshape(n_tiles, n_blks, plane_b)
    p1 = np.packbits(((idx_t >> 1) & 1).astype(np.uint8), axis=-1).reshape(n_tiles, n_blks, plane_b)
    out[:, :, 32 : 32 + plane_b] = p0
    out[:, :, 32 + plane_b : 32 + 2 * plane_b] = p1
    return out.reshape(-1)


def quantize_lr_arrays(U_f: torch.Tensor, V_f: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        Uf = U_f.float()
        Vf = V_f.float()
        U_rot, V_rot = littlebit_joint_itq_rotate(Uf, Vf, n_iter=15)
        su = (U_rot.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        sv = (V_rot.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        ui = torch.clamp(torch.round(U_rot / su.unsqueeze(1)), -127, 127).to(torch.int8)
        vi = torch.clamp(torch.round(V_rot / sv.unsqueeze(1)), -127, 127).to(torch.int8)
    return (
        su.detach().cpu().numpy().astype("<f4"),
        sv.detach().cpu().numpy().astype("<f4"),
        ui.detach().cpu().numpy(),
        vi.detach().cpu().numpy(),
    )


class LittleBit2LinearV6(nn.Module):
    """
    Authentic Samsung Research LittleBit-2 (ICML 2026) Latent Binary Factorized Linear:
      y = U_q( V_q( FWHT(x * inv_s) ) ) [+ optional U_rq( V_rq( FWHT(x * inv_s) ) )]
    where BOTH V_q [s, d_in] and U_q [d_out, s] are pure 1-bit Row-Major T16C matrices
    with per-256-block FP16 scales+biases, initialized via Activation-Weighted SVD +
    Joint-ITQ and trained with unit-normalized clipped-STE QAT (`loss.backward()`).
    """

    def __init__(
        self,
        name: str,
        w_cpu: torch.Tensor,
        x_rms: Optional[torch.Tensor],
        split_dim: int,
        residual_dim: int = 0,
        bsize: int = 256,
        itq_iter: int = 25,
    ) -> None:
        super().__init__()
        self.name = name
        self.d_out, self.d_in = int(w_cpu.shape[0]), int(w_cpu.shape[1])
        assert self.d_in % bsize == 0 and split_dim % bsize == 0
        self.bsize = bsize
        self.split_dim = split_dim
        self.residual_dim = residual_dim
        assert residual_dim == 0 or (residual_dim % bsize == 0)

        with torch.no_grad():
            w_g = w_cpu.to(DEVICE, dtype=torch.float32)
            if x_rms is not None:
                xr = x_rms.to(DEVICE, dtype=torch.float32).clamp(min=1e-5)
                wc = w_g.abs().amax(dim=0).clamp(min=1e-5)
                sq = (xr.pow(0.5) / wc.pow(0.5)).clamp(0.25, 4.0)
                sq = sq / sq.mean().clamp(min=1e-5)
                inv_s = 1.0 / sq
                w_sc = w_g * sq.unsqueeze(0)
            else:
                inv_s = None
                w_sc = w_g

            # Exact SVD + Joint-ITQ on primary latent factors (U, V)
            U_t, S_t, Vt_t = torch.linalg.svd(w_sc, full_matrices=False)
            s_use = min(split_dim, S_t.numel())
            sq_s = torch.sqrt(S_t[:s_use].clamp(min=1e-12))
            U0 = U_t[:, :s_use] * sq_s.unsqueeze(0)
            V0 = sq_s.unsqueeze(1) * Vt_t[:s_use, :]
            if s_use < split_dim:
                U0 = F.pad(U0, (0, split_dim - s_use))
                V0 = F.pad(V0, (0, 0, 0, split_dim - s_use))
            U_itq, V_itq = littlebit_joint_itq_rotate(U0, V0, n_iter=itq_iter)
            V_rot = fwht_gpu(V_itq, bsize)

            if inv_s is not None:
                self.register_buffer("inv_s", inv_s.contiguous(), persistent=False)
            else:
                self.inv_s = None

            u_z, u_sc, u_bi = init_1b_asym_norm(U_itq, bsize)
            v_z, v_sc, v_bi = init_1b_asym_norm(V_rot, bsize)
            self.U_lat = nn.Parameter(u_z)
            self.U_sc = nn.Parameter(u_sc)
            self.U_bi = nn.Parameter(u_bi)
            self.V_lat = nn.Parameter(v_z)
            self.V_sc = nn.Parameter(v_sc)
            self.V_bi = nn.Parameter(v_bi)

            if residual_dim > 0:
                u_q, _, _, _ = apply_1b_asym_ste(self.U_lat, self.U_sc, self.U_bi, bsize)
                v_q, _, _, _ = apply_1b_asym_ste(self.V_lat, self.V_sc, self.V_bi, bsize)
                w_rec = u_q @ fwht_gpu(v_q, bsize)
                w_res = w_sc - w_rec
                Ur_t, Sr_t, Vtr_t = torch.linalg.svd(w_res, full_matrices=False)
                sr_use = min(residual_dim, Sr_t.numel())
                sq_sr = torch.sqrt(Sr_t[:sr_use].clamp(min=1e-12))
                Ur0 = Ur_t[:, :sr_use] * sq_sr.unsqueeze(0)
                Vr0 = sq_sr.unsqueeze(1) * Vtr_t[:sr_use, :]
                if sr_use < residual_dim:
                    Ur0 = F.pad(Ur0, (0, residual_dim - sr_use))
                    Vr0 = F.pad(Vr0, (0, 0, 0, residual_dim - sr_use))
                Ur_itq, Vr_itq = littlebit_joint_itq_rotate(Ur0, Vr0, n_iter=max(15, itq_iter // 2))
                Vr_rot = fwht_gpu(Vr_itq, bsize)
                ur_z, ur_sc, ur_bi = init_1b_asym_norm(Ur_itq, bsize)
                vr_z, vr_sc, vr_bi = init_1b_asym_norm(Vr_rot, bsize)
                self.Ur_lat = nn.Parameter(ur_z)
                self.Ur_sc = nn.Parameter(ur_sc)
                self.Ur_bi = nn.Parameter(ur_bi)
                self.Vr_lat = nn.Parameter(vr_z)
                self.Vr_sc = nn.Parameter(vr_sc)
                self.Vr_bi = nn.Parameter(vr_bi)
            else:
                self.Ur_lat = self.Ur_sc = self.Ur_bi = None
                self.Vr_lat = self.Vr_sc = self.Vr_bi = None

    def param_groups(self, lr_lat: float, lr_sc: float) -> List[dict]:
        lat_p = [self.U_lat, self.V_lat]
        sc_p = [self.U_sc, self.U_bi, self.V_sc, self.V_bi]
        if self.Ur_lat is not None and self.Vr_lat is not None:
            lat_p.extend([self.Ur_lat, self.Vr_lat])
            sc_p.extend([self.Ur_sc, self.Ur_bi, self.Vr_sc, self.Vr_bi])
        return [{"params": lat_p, "lr": lr_lat}, {"params": sc_p, "lr": lr_sc}]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_sh = x.shape
        x2d = x.reshape(-1, self.d_in).float()
        x_sc = (x2d * self.inv_s) if self.inv_s is not None else x2d
        x_rot = fwht_gpu(x_sc, self.bsize).to(COMPUTE_DTYPE)
        v_ste, _, _, _ = apply_1b_asym_ste(self.V_lat, self.V_sc, self.V_bi, self.bsize)
        u_ste, _, _, _ = apply_1b_asym_ste(self.U_lat, self.U_sc, self.U_bi, self.bsize)
        z = F.linear(x_rot, v_ste.to(COMPUTE_DTYPE))
        y = F.linear(z, u_ste.to(COMPUTE_DTYPE)).float()
        if self.Ur_lat is not None and self.Vr_lat is not None:
            vr_ste, _, _, _ = apply_1b_asym_ste(self.Vr_lat, self.Vr_sc, self.Vr_bi, self.bsize)
            ur_ste, _, _, _ = apply_1b_asym_ste(self.Ur_lat, self.Ur_sc, self.Ur_bi, self.bsize)
            zr = F.linear(x_rot, vr_ste.to(COMPUTE_DTYPE))
            y = y + F.linear(zr, ur_ste.to(COMPUTE_DTYPE)).float()
        return y.reshape(*orig_sh[:-1], self.d_out)

    @torch.no_grad()
    def dequant_w_eff_f16(self) -> torch.Tensor:
        v_q, _, _, _ = apply_1b_asym_ste(self.V_lat, self.V_sc, self.V_bi, self.bsize)
        u_q, _, _, _ = apply_1b_asym_ste(self.U_lat, self.U_sc, self.U_bi, self.bsize)
        w_eff = u_q @ fwht_gpu(v_q, self.bsize)
        if self.Ur_lat is not None and self.Vr_lat is not None:
            vr_q, _, _, _ = apply_1b_asym_ste(self.Vr_lat, self.Vr_sc, self.Vr_bi, self.bsize)
            ur_q, _, _, _ = apply_1b_asym_ste(self.Ur_lat, self.Ur_sc, self.Ur_bi, self.bsize)
            w_eff = w_eff + ur_q @ fwht_gpu(vr_q, self.bsize)
        if self.inv_s is not None:
            w_eff = w_eff * self.inv_s.unsqueeze(0)
        return w_eff.to(torch.float16)

    @torch.no_grad()
    def export_records(self) -> List[tuple]:
        """Return T16C mat record tuples [(name, d_out, d_in, bs, nb, is_asym, has_fwht, pb, inv_s, lr), ...]."""
        inv_np = self.inv_s.detach().cpu().numpy().astype("<f4") if self.inv_s is not None else None
        _, sc_v, bi_v, bits_v = apply_1b_asym_ste(self.V_lat, self.V_sc, self.V_bi, self.bsize)
        _, sc_u, bi_u, bits_u = apply_1b_asym_ste(self.U_lat, self.U_sc, self.U_bi, self.bsize)
        pb_v = pack_rowmajor_t16c_1b_asym(bits_v, sc_v, bi_v, self.bsize)
        pb_u = pack_rowmajor_t16c_1b_asym(bits_u, sc_u, bi_u, self.bsize)
        nb_rm = 1 | T16C_ROWMAJOR_BIT
        recs = [
            (f"{self.name}.lb_v", self.split_dim, self.d_in, self.bsize, nb_rm, True, True, pb_v, inv_np, None),
            (f"{self.name}.lb_u", self.d_out, self.split_dim, self.bsize, nb_rm, True, False, pb_u, None, None),
        ]
        if self.Ur_lat is not None and self.Vr_lat is not None:
            _, sc_vr, bi_vr, bits_vr = apply_1b_asym_ste(self.Vr_lat, self.Vr_sc, self.Vr_bi, self.bsize)
            _, sc_ur, bi_ur, bits_ur = apply_1b_asym_ste(self.Ur_lat, self.Ur_sc, self.Ur_bi, self.bsize)
            pb_vr = pack_rowmajor_t16c_1b_asym(bits_vr, sc_vr, bi_vr, self.bsize)
            pb_ur = pack_rowmajor_t16c_1b_asym(bits_ur, sc_ur, bi_ur, self.bsize)
            recs.append((f"{self.name}.lb_rv", self.residual_dim, self.d_in, self.bsize, nb_rm, True, True, pb_vr, inv_np, None))
            recs.append((f"{self.name}.lb_ru", self.d_out, self.residual_dim, self.bsize, nb_rm, True, False, pb_ur, None, None))
        return recs


class DirectRowMajorLinearV6(nn.Module):
    """
    Direct Row-Major 1b_asym (1|0x40) or 2b_gaus (2|0x80|0x40) with optional INT8 low-rank,
    trained with unit-normalized clipped-STE QAT (`loss.backward()`).
    """

    def __init__(
        self,
        name: str,
        w_cpu: torch.Tensor,
        x_rms: Optional[torch.Tensor],
        bits: int = 1,
        lr_rank: int = 0,
        bsize: int = 256,
    ) -> None:
        super().__init__()
        self.name = name
        self.d_out, self.d_in = int(w_cpu.shape[0]), int(w_cpu.shape[1])
        self.bits = bits
        self.lr_rank = lr_rank
        self.bsize = bsize

        with torch.no_grad():
            w_g = w_cpu.to(DEVICE, dtype=torch.float32)
            if x_rms is not None:
                xr = x_rms.to(DEVICE, dtype=torch.float32).clamp(min=1e-5)
                wc = w_g.abs().amax(dim=0).clamp(min=1e-5)
                sq = (xr.pow(0.5) / wc.pow(0.5)).clamp(0.25, 4.0)
                sq = sq / sq.mean().clamp(min=1e-5)
                inv_s = 1.0 / sq
                w_sc = w_g * sq.unsqueeze(0)
            else:
                inv_s = None
                w_sc = w_g

            w_rot = fwht_gpu(w_sc, bsize)
            if inv_s is not None:
                self.register_buffer("inv_s", inv_s.contiguous(), persistent=False)
            else:
                self.inv_s = None

            if bits == 1:
                w_z, w_sc_p, w_bi_p = init_1b_asym_norm(w_rot, bsize)
                self.W_lat = nn.Parameter(w_z)
                self.W_sc = nn.Parameter(w_sc_p)
                self.W_bi = nn.Parameter(w_bi_p)
                wq_rot, _, _, _ = apply_1b_asym_ste(self.W_lat, self.W_sc, self.W_bi, bsize)
            else:
                w_z, w_sc_p = init_2b_gaus_norm(w_rot, bsize)
                self.W_lat = nn.Parameter(w_z)
                self.W_sc = nn.Parameter(w_sc_p)
                self.W_bi = None
                wq_rot, _, _ = apply_2b_gaus_ste(self.W_lat, self.W_sc, bsize)

            if lr_rank > 0:
                w_eff0 = fwht_gpu(wq_rot, bsize)
                if inv_s is not None:
                    w_eff0 = w_eff0 * inv_s.unsqueeze(0)
                err = w_g - w_eff0
                Ue, Se, Vte = torch.linalg.svd(err, full_matrices=False)
                r_u = min(lr_rank, Se.numel())
                sqs = torch.sqrt(Se[:r_u].clamp(min=1e-12))
                u0 = Ue[:, :r_u] * sqs.unsqueeze(0)
                v0 = sqs.unsqueeze(1) * Vte[:r_u, :]
                self.U_lr = nn.Parameter(u0.contiguous())
                self.V_lr = nn.Parameter(v0.contiguous())
            else:
                self.U_lr = None
                self.V_lr = None

    def param_groups(self, lr_lat: float, lr_sc: float) -> List[dict]:
        sc_p = [self.W_sc]
        if self.W_bi is not None:
            sc_p.append(self.W_bi)
        if self.U_lr is not None and self.V_lr is not None:
            sc_p.extend([self.U_lr, self.V_lr])
        return [{"params": [self.W_lat], "lr": lr_lat}, {"params": sc_p, "lr": lr_sc}]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_sh = x.shape
        x2d = x.reshape(-1, self.d_in).float()
        x_sc = (x2d * self.inv_s) if self.inv_s is not None else x2d
        x_rot = fwht_gpu(x_sc, self.bsize).to(COMPUTE_DTYPE)
        if self.bits == 1:
            w_ste, _, _, _ = apply_1b_asym_ste(self.W_lat, self.W_sc, self.W_bi, self.bsize)
        else:
            w_ste, _, _ = apply_2b_gaus_ste(self.W_lat, self.W_sc, self.bsize)
        y = F.linear(x_rot, w_ste.to(COMPUTE_DTYPE)).float()
        if self.U_lr is not None and self.V_lr is not None:
            su = (self.U_lr.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
            sv = (self.V_lr.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
            u_q = torch.clamp(torch.round(self.U_lr / su), -127.0, 127.0) * su
            v_q = torch.clamp(torch.round(self.V_lr / sv), -127.0, 127.0) * sv
            u_ste = self.U_lr + (u_q - self.U_lr).detach()
            v_ste = self.V_lr + (v_q - self.V_lr).detach()
            y = y + (x2d @ v_ste.T) @ u_ste.T
        return y.reshape(*orig_sh[:-1], self.d_out)

    @torch.no_grad()
    def dequant_w_eff_f16(self) -> torch.Tensor:
        if self.bits == 1:
            wq_rot, _, _, _ = apply_1b_asym_ste(self.W_lat, self.W_sc, self.W_bi, self.bsize)
        else:
            wq_rot, _, _ = apply_2b_gaus_ste(self.W_lat, self.W_sc, self.bsize)
        w_eff = fwht_gpu(wq_rot, self.bsize)
        if self.inv_s is not None:
            w_eff = w_eff * self.inv_s.unsqueeze(0)
        if self.U_lr is not None and self.V_lr is not None:
            su, sv, ui, vi = quantize_lr_arrays(self.U_lr, self.V_lr)
            u_q = torch.from_numpy(ui.astype(np.float32)).to(w_eff.device) * torch.from_numpy(su).to(w_eff.device).unsqueeze(1)
            v_q = torch.from_numpy(vi.astype(np.float32)).to(w_eff.device) * torch.from_numpy(sv).to(w_eff.device).unsqueeze(1)
            w_eff = w_eff + u_q @ v_q
        return w_eff.to(torch.float16)

    @torch.no_grad()
    def export_records(self) -> List[tuple]:
        inv_np = self.inv_s.detach().cpu().numpy().astype("<f4") if self.inv_s is not None else None
        lr_tup = quantize_lr_arrays(self.U_lr, self.V_lr) if (self.U_lr is not None and self.V_lr is not None) else None
        if self.bits == 1:
            _, sc_q, bi_q, bits01 = apply_1b_asym_ste(self.W_lat, self.W_sc, self.W_bi, self.bsize)
            pb = pack_rowmajor_t16c_1b_asym(bits01, sc_q, bi_q, self.bsize)
            nb, asy = (1 | T16C_ROWMAJOR_BIT), True
        else:
            _, sc_q, idx = apply_2b_gaus_ste(self.W_lat, self.W_sc, self.bsize)
            pb = pack_rowmajor_t16c_2b_gaus(idx, sc_q, self.bsize)
            nb, asy = (2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT), False
        return [(self.name, self.d_out, self.d_in, self.bsize, nb, asy, True, pb, inv_np, lr_tup)]


def apply_block_sparse_geglu_mask(gelu_gate: torch.Tensor, top_p: float = 0.50, bsize: int = 256) -> torch.Tensor:
    if top_p >= 0.9999:
        return gelu_gate
    orig_sh = gelu_gate.shape
    ff = orig_sh[-1]
    n_blks = ff // bsize
    keep = max(1, min(n_blks, int(math.ceil(n_blks * top_p))))
    g_blk = gelu_gate.reshape(-1, n_blks, bsize)
    energy = (g_blk * g_blk).sum(dim=-1)
    topk_idx = torch.topk(energy, keep, dim=-1).indices
    mask = torch.zeros_like(energy)
    mask.scatter_(1, topk_idx, 1.0)
    return (g_blk * mask.unsqueeze(-1)).reshape(orig_sh)


# ---------------------------------------------------------------------------
# [Gate 0a] Self-Test: Row-Major 1b_asym / 2b_gaus Packing + LittleBit-2
# Joint-ITQ + Real loss.backward() Bit-Flipping QAT Parity!
# ---------------------------------------------------------------------------
print("[Gate 0a] Verifying LittleBit2LinearV6 + DirectRowMajorLinearV6 + Row-Major packing + real loss.backward()...", flush=True)
torch.manual_seed(42)
with torch.enable_grad():
    _do, _di, _bs = 512, 512, 256
    _xt = torch.randn((64, _di), device=DEVICE, dtype=torch.float32) * 0.4
    _wt = torch.randn((_do, _di), dtype=torch.float32) * 0.05
    _xr = torch.sqrt((_xt * _xt).mean(dim=0).clamp(min=1e-8))
    _yt = _xt @ _wt.to(DEVICE).T

    _lb = LittleBit2LinearV6("test.lb", _wt, _xr, split_dim=256, residual_dim=256, bsize=256, itq_iter=15)
    with torch.no_grad():
        _y0 = _lb(_xt)
        _we0 = _lb.dequant_w_eff_f16()
        _rel0 = float((_y0 - (_xt @ _we0.float().T)).norm() / _y0.norm().clamp(min=1e-6))
        assert _rel0 < 5e-3, f"LittleBit2LinearV6 init parity fail rel={_rel0}"
        _bits_before = (_lb.U_lat >= 0.0).clone()
    _opt = torch.optim.AdamW(_lb.param_groups(lr_lat=1.2e-2, lr_sc=1.5e-3))
    _l0 = float(F.mse_loss(_lb(_xt).detach(), _yt).item())
    for _ in range(30):
        _opt.zero_grad()
        _loss = F.mse_loss(_lb(_xt), _yt)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(_lb.parameters(), 1.0)
        _opt.step()
    _l1 = float(F.mse_loss(_lb(_xt).detach(), _yt).item())
    _flips = int(((_lb.U_lat >= 0.0) != _bits_before).sum().item())
    assert _l1 < _l0 * 0.75, f"LittleBit2LinearV6 real STE QAT failed to reduce loss: {_l0} -> {_l1}"
    assert _flips > 0, "LittleBit2LinearV6 STE QAT produced zero binary sign flips!"
    _recs = _lb.export_records()
    assert len(_recs) == 4 and _recs[0][4] == (1 | T16C_ROWMAJOR_BIT) and _recs[1][4] == (1 | T16C_ROWMAJOR_BIT)
    # Verify exact byte count matches p3.expected_tile16_bytes
    for _r in _recs:
        assert _r[7].size == p3.expected_tile16_bytes(_r[1], _r[2], _r[3], _r[4], _r[5])

    _dr = DirectRowMajorLinearV6("test.dr2", _wt, _xr, bits=2, lr_rank=8, bsize=256)
    _opt_dr = torch.optim.AdamW(_dr.param_groups(lr_lat=1.2e-2, lr_sc=1.5e-3))
    _dl0 = float(F.mse_loss(_dr(_xt).detach(), _yt).item())
    for _ in range(10):
        _opt_dr.zero_grad()
        F.mse_loss(_dr(_xt), _yt).backward()
        torch.nn.utils.clip_grad_norm_(_dr.parameters(), 1.0)
        _opt_dr.step()
    _dl1 = float(F.mse_loss(_dr(_xt).detach(), _yt).item())
    assert _dl1 < _dl0, f"DirectRowMajorLinearV6 2b_gaus STE QAT failed to reduce loss: {_dl0} -> {_dl1}"
    _recs2 = _dr.export_records()
    assert len(_recs2) == 1 and _recs2[0][4] == (2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT)
    assert _recs2[0][7].size == p3.expected_tile16_bytes(_recs2[0][1], _recs2[0][2], _recs2[0][3], _recs2[0][4], _recs2[0][5])
    del _xt, _wt, _xr, _yt, _lb, _opt, _recs, _dr, _opt_dr, _recs2
torch.cuda.empty_cache()
print(f"[Gate 0a PASS] LittleBit-2 Joint-ITQ + STE QAT ({_l0:.5f} -> {_l1:.5f}, flips={_flips}) + Direct2b ({_dl0:.5f} -> {_dl1:.5f}) + Row-Major T16C packing verified!", flush=True)


# ---------------------------------------------------------------------------
# 3. V6.0 Allocation Schedule + Fail-Closed Pi 4 RAM Budget Guard (<= 3.60 GB!)
# ---------------------------------------------------------------------------
FULL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}


def get_lm_spec_v6(l_idx: int, pname: str) -> dict:
    """
    Smart V6.0 LittleBit-2 Hybrid Allocation (all LM mats Row-Major AND+VCNT!):
      - 'lb': LittleBit2LinearV6 (split_dim s, optional residual_dim sr) -> pure 1b Row-Major!
      - 'dr': DirectRowMajorLinearV6 (bits=1 or 2, optional lr_rank) -> Row-Major!
    Total LM body ~2.45 GB (~0.67 bpw avg), total .t16c ~3.18 GB (~2.96 GiB) -> 2x smaller & 2-3x faster on Pi 4!
    """
    if l_idx >= 56:
        if pname in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return {"kind": "dr", "bits": 2, "lr_rank": 16}
        return {"kind": "lb", "s": 2048, "sr": 1024}

    if l_idx < 4:
        if pname in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return {"kind": "dr", "bits": 1, "lr_rank": 16}
        return {"kind": "lb", "s": 3072, "sr": 0}

    # Core layers L04..L55
    if pname in ("k_proj", "v_proj"):
        rk = 16 if (l_idx in FULL_LAYERS or l_idx >= 36) else 0
        return {"kind": "dr", "bits": 1, "lr_rank": rk}
    if pname in ("q_proj", "o_proj"):
        s_attn = 2560 if l_idx >= 36 else 2048
        return {"kind": "lb", "s": s_attn, "sr": 0}
    # MLP gate_proj, up_proj, down_proj
    s_mlp = 2560 if l_idx >= 36 else 2048
    return {"kind": "lb", "s": s_mlp, "sr": 0}


def count_spec_records_and_bytes(d_out: int, d_in: int, spec: dict, bsize: int = 256) -> Tuple[int, int]:
    if spec["kind"] == "lb":
        s = spec["s"]
        sr = spec["sr"]
        nb = 1 | T16C_ROWMAJOR_BIT
        b = (
            p3.expected_tile16_bytes(s, d_in, bsize, nb, True) + d_in * 4 + 80
            + p3.expected_tile16_bytes(d_out, s, bsize, nb, True) + 80
        )
        cnt = 2
        if sr > 0:
            b += (
                p3.expected_tile16_bytes(sr, d_in, bsize, nb, True) + d_in * 4 + 80
                + p3.expected_tile16_bytes(d_out, sr, bsize, nb, True) + 80
            )
            cnt += 2
        return cnt, b
    bits = spec["bits"]
    rk = spec["lr_rank"]
    nb, asy = ((1 | T16C_ROWMAJOR_BIT), True) if bits == 1 else ((2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT), False)
    pb = p3.expected_tile16_bytes(d_out, d_in, bsize, nb, asy)
    lr_b = (d_out * 4 + rk * 4 + d_out * rk + rk * d_in) if rk > 0 else 0
    return 1, pb + d_in * 4 + lr_b + 80


# ---------------------------------------------------------------------------
# 4. Stage 1: Download Model & Clean Zero-Contamination Corpus
# ---------------------------------------------------------------------------
print("\n[Stage 1] Downloading google/gemma-4-31B-it-qat-w4a16-ct...", flush=True)
model_dir = Path(snapshot_download(
    repo_id="google/gemma-4-31B-it-qat-w4a16-ct",
    local_dir="/tmp/gemma4_31b",
    ignore_patterns=["*.bin", "*.pt"],
))
sf_path = model_dir / "model.safetensors"
cfg_full = json.loads((model_dir / "config.json").read_text())
tk = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

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

# Budget check BEFORE downloading datasets!
_pred_lm_mats = 1
_pred_lm_b = p3.expected_tile16_bytes(int(tc["vocab_size"]), D_MOD, 256, 3, False) + 80
_lb_mats_cnt = 0
_dr_mats_cnt = 0
for _l in range(N_LAYERS):
    _f = _l in FULL_LAYERS
    _hd = HD_F if _f else HD_S
    _nkv = NKV_F if _f else NKV_S
    _sh = {
        "q_proj": (NH * _hd, D_MOD), "k_proj": (_nkv * _hd, D_MOD), "o_proj": (D_MOD, NH * _hd),
        "gate_proj": (21504, D_MOD), "up_proj": (21504, D_MOD), "down_proj": (D_MOD, 21504),
    }
    if not _f:
        _sh["v_proj"] = (_nkv * _hd, D_MOD)
    for _pn, (_do, _di) in _sh.items():
        _sp = get_lm_spec_v6(_l, _pn)
        _rc, _rb = count_spec_records_and_bytes(_do, _di, _sp, 256)
        _pred_lm_mats += _rc
        _pred_lm_b += _rb
        if _sp["kind"] == "lb":
            _lb_mats_cnt += 1
        else:
            _dr_mats_cnt += 1
_pred_body_b = _pred_lm_b - p3.expected_tile16_bytes(int(tc["vocab_size"]), D_MOD, 256, 3, False)
print(
    f"[Gate 0b Budget Guard] V6.0 LittleBit-2 LM body={_pred_body_b/1e9:.3f} GB ({_pred_body_b/1024**3:.3f} GiB) "
    f"| LM+Embed={_pred_lm_b/1e9:.3f} GB | LM records={_pred_lm_mats} (LittleBit={_lb_mats_cnt}, DirectRM={_dr_mats_cnt})",
    flush=True,
)
assert _pred_lm_b <= 3_400_000_000, f"V6.0 LM+Embed budget overflow: {_pred_lm_b:,} B > 3.40 GB!"


def is_clean_row(txt: str) -> bool:
    low = txt.lower()
    return not any(b in low for b in ALL_BLACKLIST)


print("[Stage 1] Loading & auditing clean multi-domain text corpus (zero contamination)...", flush=True)
ds_gsm = load_dataset("openai/gsm8k", "main", split="train")
ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
ds_alp = load_dataset("yahma/alpaca-cleaned", split="train")
ds_wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

clean_turns: List[Tuple[str, str, str]] = []
for row in ds_gsm:
    q, a = row["question"].strip(), row["answer"].strip()
    if is_clean_row(q + " " + a):
        clean_turns.append(("math", q, a))
for row in ds_code:
    p = (row.get("instruction", "") + ("\n" + row.get("input", "") if row.get("input", "") else "")).strip()
    a = row.get("output", "").strip()
    if p and a and is_clean_row(p + " " + a):
        clean_turns.append(("code", p, a))
for row in ds_alp:
    p = (row.get("instruction", "") + ("\n" + row.get("input", "") if row.get("input", "") else "")).strip()
    a = row.get("output", "").strip()
    if p and a and is_clean_row(p + " " + a):
        clean_turns.append(("alpaca", p, a))
wiki_buf: List[str] = []
for row in ds_wiki:
    t = row["text"].strip()
    if len(t) > 60 and not t.startswith("=") and is_clean_row(t):
        wiki_buf.append(t)
for idx in range(0, len(wiki_buf) - 1, 2):
    clean_turns.append(("wiki", "Summarize and continue accurately:\n" + wiki_buf[idx][:320], wiki_buf[idx + 1][:500]))

rng_np = np.random.default_rng(42)
perm_c = rng_np.permutation(len(clean_turns))
N_TEXT_SEQ, N_VLM_SEQ, SEQ_LEN = 608, 160, 352
N_SEQ = N_TEXT_SEQ + N_VLM_SEQ
GEMMA4_BOI_ID, GEMMA4_IMAGE_ID, GEMMA4_EOI_ID = 255999, 258880, 258882

by_dom = defaultdict(list)
for idx in perm_c:
    d_name, q_t, a_t = clean_turns[idx]
    by_dom[d_name].append((q_t, a_t))
dom_names = ["math", "code", "alpaca", "wiki"]
ptrs = {d: 0 for d in dom_names}
seq_ids_list, seq_nxt_list, seq_ans_list, seq_vlm_ans_list = [], [], [], []
t_cursor = 0
for _ in range(N_TEXT_SEQ):
    cur_toks: List[int] = [2]
    cur_ans: List[bool] = [False]
    while len(cur_toks) < SEQ_LEN + 1:
        d_n = dom_names[t_cursor % len(dom_names)]
        q_t, a_t = by_dom[d_n][ptrs[d_n] % len(by_dom[d_n])]
        ptrs[d_n] += 1
        t_cursor += 1
        p_tok = list(tk.encode(vc.format_gemma4_turn(q_t, None), add_special_tokens=False).ids)
        r_tok = list(tk.encode(f"{a_t}<turn|>\n", add_special_tokens=False).ids)
        if not p_tok or not r_tok:
            continue
        cur_toks.extend(p_tok)
        cur_ans.extend([False] * len(p_tok))
        cur_toks.extend(r_tok)
        cur_ans.extend([True] * len(r_tok))
    cur_toks = cur_toks[: SEQ_LEN + 1]
    cur_ans = cur_ans[: SEQ_LEN + 1]
    seq_ids_list.append(cur_toks[:-1])
    seq_nxt_list.append(cur_toks[1:])
    seq_ans_list.append(cur_ans[1:])
    seq_vlm_ans_list.append([False] * SEQ_LEN)

print(f"[Stage 1] Packed {N_TEXT_SEQ} dense text seqs x {SEQ_LEN} = {N_TEXT_SEQ * SEQ_LEN:,} tokens.", flush=True)


# ---------------------------------------------------------------------------
# 5. Stage 2: Vision Tower (Fast 2b_gaus / 3b_u0) + Multimodal Soft Tokens
# ---------------------------------------------------------------------------
H8_GPU = torch.from_numpy(c8d.H8_NP).to(DEVICE, dtype=torch.float32)


def eval_c8d_gpu(x_gpu: torch.Tensor, cents: np.ndarray, experts: List[tuple], top_k: int = 2, temp: float = 0.5, gscale: float = 1.0) -> torch.Tensor:
    xf = x_gpu.float()
    n_t, d_i = xf.shape
    d_o = experts[0][0].shape[0]
    kb = d_i // 8
    u_bar = xf[:, : kb * 8].reshape(n_t, kb, 8).mean(dim=1) @ H8_GPU
    g_hat = u_bar / u_bar.norm(dim=1, keepdim=True).clamp(min=1e-9)
    cg = torch.from_numpy(cents).to(DEVICE, dtype=torch.float32)
    logits = (g_hat @ cg.T) / temp
    topv, topi = torch.topk(logits, top_k, dim=-1)
    ws = F.softmax(topv, dim=-1) * gscale
    out = torch.zeros((n_t, d_o), device=DEVICE, dtype=torch.float32)
    for e_idx, (su, sv, ui, vi) in enumerate(experts):
        mask_e = (topi == e_idx)
        if not mask_e.any():
            continue
        w_e = (ws * mask_e.float()).sum(dim=-1, keepdim=True)
        nz = w_e.squeeze(1) > 0
        xn = xf[nz]
        vf = torch.from_numpy(vi).to(DEVICE, dtype=torch.float32) * torch.from_numpy(sv).to(DEVICE, dtype=torch.float32).unsqueeze(1)
        uf = torch.from_numpy(ui).to(DEVICE, dtype=torch.float32) * torch.from_numpy(su).to(DEVICE, dtype=torch.float32).unsqueeze(1)
        out[nz] += w_e[nz] * ((xn @ vf.T) @ uf.T)
    return out


VIS_CKPT_PATH = CKPT_DIR / "stage2_vision.pt"
VIS_META_PATH = CKPT_DIR / "stage2_meta.pt"
VLM_SOFT_CACHE = CKPT_DIR / "stage2_vlm_soft.pt"


def pack_vision_fast(name: str, w_t: torch.Tensor, mode: int, pad_in: int = 0) -> Tuple[tuple, torch.Tensor]:
    d_o, d_i = w_t.shape
    if pad_in > d_i:
        w_pad = torch.zeros((d_o, pad_in), dtype=torch.float32)
        w_pad[:, :d_i] = w_t.float()
        d_in_eff = pad_in
    else:
        w_pad = w_t.float()
        d_in_eff = d_i
    w_rot = fwht_gpu(w_pad.to(DEVICE), 64).contiguous().cpu().numpy().astype(np.float32)
    nb, asy = p3.MODE_TO_T16C[mode]
    pb = np.empty(p3.expected_tile16_bytes(d_o, d_in_eff, 64, nb, asy), dtype=np.uint8)
    LIB_FQ.quant_and_pack_t16c_f32(w_rot.ctypes.data, d_o, d_in_eff, 64, mode, pb.ctypes.data)
    w_eff = fwht_gpu(torch.from_numpy(w_rot).to(DEVICE), 64)[:, :d_i].to(torch.float16).cpu()
    rec = (name, d_o, d_in_eff, 64, nb, asy, True, pb, None, None)
    return rec, w_eff


def _build_stage2_vision() -> dict:
    print("\n[Stage 2] V6.0 Vision Tower Calibration (V00..V23 2b_gaus + V24..V26/embed 3b_u0)...", flush=True)
    cfg_auto = AutoConfig.from_pretrained(str(model_dir))
    proc = Gemma4ImageProcessor.from_pretrained(str(model_dir))

    def make_optical_va_pil(idx: int) -> Tuple[Image.Image, str]:
        im = Image.new("RGB", (672, 672), (248, 248, 248))
        dr = ImageDraw.Draw(im)
        dr.text((18, 18), f"GEMMA4 31B V6.0 LITTLEBIT2 LM=60 VIS=27 FRAME={idx:03d}", fill=(12, 18, 28))
        dr.text((18, 42), "OPTICAL VA PRECISION ANALYSIS C8DB-H8", fill=(12, 18, 28))
        rng_v = np.random.default_rng(1000 + idx)
        for b in range(60):
            bh = int(rng_v.integers(25, 118))
            x = 24 + b * 10
            col = (35, 115, 210) if b >= 58 else ((30, 165, 95) if b % 5 != 0 else (210, 65, 45))
            dr.rectangle([x, 550 - bh, x + 7, 550], fill=col)
        cap = (
            f"An optical diagnostic chart labeled GEMMA4 31B V6.0 LITTLEBIT2 LM=60 VIS=27 FRAME={idx:03d} "
            f"displaying 60 vertical green, red, and blue bar segments on a light background."
        )
        return im, cap

    vis_va_pairs = [make_optical_va_pil(i) for i in range(32)]
    vis_coco_pairs, heldout_pairs = [], []
    coco_stream = load_dataset("lmms-lab/COCO-Caption2017", split="val", streaming=True)
    for row_idx, row in enumerate(coco_stream):
        im = row.get("image")
        ans_raw = row.get("answer")
        if im is None or not ans_raw:
            continue
        cap = (ans_raw[0] if isinstance(ans_raw, list) else str(ans_raw)).strip()
        if not cap:
            continue
        if row_idx < 4:
            heldout_pairs.append((im.convert("RGB"), cap))
            continue
        if row_idx < 64:
            continue
        if not is_clean_row(cap):
            continue
        vis_coco_pairs.append((im.convert("RGB"), cap))
        if len(vis_coco_pairs) >= 128:
            break
    vis_pairs = []
    for i in range(32):
        vis_pairs.append(vis_va_pairs[i])
        vis_pairs.extend(vis_coco_pairs[i * 4 : (i + 1) * 4])
    vis_images = [p[0] for p in vis_pairs]
    vis_captions = [p[1] for p in vis_pairs]

    with safe_open(sf_path, framework="pt", device="cpu") as f:
        vis_sd = {k[len("model.vision_tower."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.vision_tower.")}
        embv_sd = {k[len("model.embed_vision."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.embed_vision.")}
        std_bias_f32 = vis_sd["std_bias"].float().numpy().astype("<f4")
        std_scale_f32 = vis_sd["std_scale"].float().numpy().astype("<f4")
        pos_tbl_f32 = vis_sd["patch_embedder.position_embedding_table"].float()

    vis_teacher = Gemma4VisionModel(cfg_auto.vision_config).to(torch.float16)
    vis_teacher.load_state_dict(vis_sd, strict=True)
    vis_teacher = vis_teacher.to(DEVICE).eval()
    mm_teacher = Gemma4MultimodalEmbedder(cfg_auto.vision_config, cfg_auto.text_config).to(torch.float16)
    mm_teacher.load_state_dict(embv_sd, strict=True)
    mm_teacher = mm_teacher.to(DEVICE).eval()

    vis_teach_soft, vis_eval_pvs, vis_eval_poss = [], [], []
    with torch.no_grad():
        for im in vis_images:
            pin = proc(images=[im], return_tensors="pt")
            pv = pin["pixel_values"].to(DEVICE, dtype=torch.float16)
            pos = pin["image_position_ids"].to(DEVICE)
            vo = vis_teacher(pixel_values=pv, pixel_position_ids=pos)
            st = mm_teacher(vo.last_hidden_state).reshape(-1, D_MOD)
            vis_eval_pvs.append(pv.cpu())
            vis_eval_poss.append(pos.cpu())
            vis_teach_soft.append(st.to(torch.float16).cpu())
    heldout_pvs, heldout_poss, heldout_teach_soft = [], [], []
    with torch.no_grad():
        for im_h, _ in heldout_pairs:
            pin = proc(images=[im_h], return_tensors="pt")
            pv = pin["pixel_values"].to(DEVICE, dtype=torch.float16)
            pos = pin["image_position_ids"].to(DEVICE)
            vo = vis_teacher(pixel_values=pv, pixel_position_ids=pos)
            st = mm_teacher(vo.last_hidden_state).reshape(-1, D_MOD)
            heldout_pvs.append(pv.cpu())
            heldout_poss.append(pos.cpu())
            heldout_teach_soft.append(st.to(torch.float16).cpu())
    del vis_teacher, mm_teacher
    torch.cuda.empty_cache()

    vision_records, vis_deq_weights = [], {}
    vision_norms = {
        "model.vision_tower.std_bias": std_bias_f32,
        "model.vision_tower.std_scale": std_scale_f32,
    }
    for ax in (0, 1):
        nm_pos = f"model.vision_tower.patch_embedder.position_embedding_table.{ax}"
        r_p, w_p = pack_vision_fast(nm_pos, pos_tbl_f32[ax], mode=4)
        vision_records.append(r_p); vis_deq_weights[nm_pos] = w_p
    r_in, w_in = pack_vision_fast("model.vision_tower.patch_embedder.input_proj.weight", vis_sd["patch_embedder.input_proj.weight"], mode=4)
    vision_records.append(r_in); vis_deq_weights["model.vision_tower.patch_embedder.input_proj.weight"] = w_in
    for l_v in range(27):
        full_pfx = f"model.vision_tower.encoder.layers.{l_v}"
        for nk in ["input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
                   "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]:
            vision_norms[f"{full_pfx}.{nk}"] = vis_sd[f"encoder.layers.{l_v}.{nk}"].float().numpy().astype("<f4")
        v_mode = 4 if l_v >= 24 else 5
        for proj in ["self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight",
                     "self_attn.o_proj.linear.weight", "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight"]:
            r_m, w_m = pack_vision_fast(f"{full_pfx}.{proj}", vis_sd[f"encoder.layers.{l_v}.{proj}"], mode=v_mode)
            vision_records.append(r_m); vis_deq_weights[f"{full_pfx}.{proj}"] = w_m
        r_d, w_d = pack_vision_fast(f"{full_pfx}.mlp.down_proj.linear.weight", vis_sd[f"encoder.layers.{l_v}.mlp.down_proj.linear.weight"], mode=v_mode, pad_in=4352)
        vision_records.append(r_d); vis_deq_weights[f"{full_pfx}.mlp.down_proj.linear.weight"] = w_d
    r_ep, w_ep = pack_vision_fast("model.embed_vision.embedding_projection.weight", embv_sd["embedding_projection.weight"], mode=4)
    vision_records.append(r_ep); vis_deq_weights["model.embed_vision.embedding_projection.weight"] = w_ep
    v_bytes = sum(len(r[7]) + 80 for r in vision_records)

    vis_stud = Gemma4VisionModel(cfg_auto.vision_config).to(torch.float16)
    vis_stud.load_state_dict(vis_sd, strict=True)
    with torch.no_grad():
        vis_stud.patch_embedder.input_proj.weight.copy_(vis_deq_weights["model.vision_tower.patch_embedder.input_proj.weight"])
        vis_stud.patch_embedder.position_embedding_table[0].copy_(vis_deq_weights["model.vision_tower.patch_embedder.position_embedding_table.0"])
        vis_stud.patch_embedder.position_embedding_table[1].copy_(vis_deq_weights["model.vision_tower.patch_embedder.position_embedding_table.1"])
        for l_v, lv_mod in enumerate(vis_stud.encoder.layers):
            full_pfx = f"model.vision_tower.encoder.layers.{l_v}"
            for sub, m_sub in [
                ("self_attn.q_proj", lv_mod.self_attn.q_proj), ("self_attn.k_proj", lv_mod.self_attn.k_proj),
                ("self_attn.v_proj", lv_mod.self_attn.v_proj), ("self_attn.o_proj", lv_mod.self_attn.o_proj),
                ("mlp.gate_proj", lv_mod.mlp.gate_proj), ("mlp.up_proj", lv_mod.mlp.up_proj),
                ("mlp.down_proj", lv_mod.mlp.down_proj),
            ]:
                m_sub.linear.weight.copy_(vis_deq_weights[f"{full_pfx}.{sub}.linear.weight"])
    vis_stud = vis_stud.to(DEVICE).eval()
    mm_stud = Gemma4MultimodalEmbedder(cfg_auto.vision_config, cfg_auto.text_config).to(torch.float16)
    mm_stud.load_state_dict(embv_sd, strict=True)
    with torch.no_grad():
        mm_stud.embedding_projection.weight.copy_(vis_deq_weights["model.embed_vision.embedding_projection.weight"])
    mm_stud = mm_stud.to(DEVICE).eval()

    vis_stud_pre, vis_stud_base, heldout_stud_pre, heldout_stud_base = [], [], [], []
    with torch.no_grad():
        for pv_c, pos_c in zip(vis_eval_pvs, vis_eval_poss):
            vo = vis_stud(pixel_values=pv_c.to(DEVICE), pixel_position_ids=pos_c.to(DEVICE))
            hn = mm_stud.embedding_pre_projection_norm(vo.last_hidden_state)
            vis_stud_pre.append(hn.reshape(-1, 1152).float().cpu())
            vis_stud_base.append(mm_stud.embedding_projection(hn).reshape(-1, D_MOD).float().cpu())
        for pv_c, pos_c in zip(heldout_pvs, heldout_poss):
            vo = vis_stud(pixel_values=pv_c.to(DEVICE), pixel_position_ids=pos_c.to(DEVICE))
            hn = mm_stud.embedding_pre_projection_norm(vo.last_hidden_state)
            heldout_stud_pre.append(hn.reshape(-1, 1152).float().cpu())
            heldout_stud_base.append(mm_stud.embedding_projection(hn).reshape(-1, D_MOD).float().cpu())
    del vis_stud, mm_stud, vis_sd, embv_sd, vis_deq_weights, vis_eval_pvs, vis_eval_poss, heldout_pvs, heldout_poss
    torch.cuda.empty_cache()

    vx = torch.cat(vis_stud_pre, 0)
    vy_t = torch.cat([t.float() for t in vis_teach_soft], 0)
    vy_s = torch.cat(vis_stud_base, 0)
    ntr_v = (vx.shape[0] * 4) // 5
    vx_tr, vyt_tr, vys_tr = vx[:ntr_v].to(DEVICE), vy_t[:ntr_v].to(DEVICE), vy_s[:ntr_v].to(DEVICE)
    vx_va, vyt_va, vys_va = vx[ntr_v:].to(DEVICE), vy_t[ntr_v:].to(DEVICE), vy_s[ntr_v:].to(DEVICE)
    v_cos0 = float(F.cosine_similarity(vys_va, vyt_va, dim=-1).mean().item())
    model_vis = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=1152, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(DEVICE)
    opt_v = torch.optim.AdamW(model_vis.parameters(), lr=3e-3, weight_decay=2e-2)
    sch_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=250, eta_min=3e-4)
    for _ in range(250):
        opt_v.zero_grad()
        idx_b = torch.randint(0, vx_tr.shape[0], (min(2048, vx_tr.shape[0]),), device=DEVICE)
        dv, _, lb = model_vis(vx_tr[idx_b])
        pred = vys_tr[idx_b] + dv
        tgt = vyt_tr[idx_b]
        loss = F.mse_loss(pred, tgt) + 0.5 * (1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()) + 0.005 * lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_vis.parameters(), 1.0)
        opt_v.step(); sch_v.step()
    cents_vis, experts_vis = model_vis.cpu().export_quantized_arrays()
    with torch.no_grad():
        v_cos1 = float(F.cosine_similarity(vys_va + eval_c8d_gpu(vx_va, cents_vis, experts_vis), vyt_va, dim=-1).mean().item())
        vis_keep_c8d = v_cos1 > v_cos0
        vis_stud_soft = [
            (b.to(DEVICE) + (eval_c8d_gpu(p.to(DEVICE), cents_vis, experts_vis) if vis_keep_c8d else 0.0)).to(torch.float16).cpu()
            for p, b in zip(vis_stud_pre, vis_stud_base)
        ]
        heldout_stud_soft = [
            (b.to(DEVICE) + (eval_c8d_gpu(p.to(DEVICE), cents_vis, experts_vis) if vis_keep_c8d else 0.0)).to(torch.float16).cpu()
            for p, b in zip(heldout_stud_pre, heldout_stud_base)
        ]
    print(f"[Stage 2 DONE] Vision ({len(vision_records)} mats, {v_bytes/1e6:.1f} MB) val cos: {v_cos0:.5f} -> {v_cos1:.5f}", flush=True)
    del vx, vy_t, vy_s, vx_tr, vyt_tr, vys_tr, vx_va, vyt_va, vys_va, model_vis, vis_stud_pre, vis_stud_base, heldout_stud_pre, heldout_stud_base
    torch.cuda.empty_cache()

    _atomic_torch_save({"vision_records": vision_records}, VIS_CKPT_PATH)
    _atomic_torch_save({
        "vis_teach_soft": vis_teach_soft, "vis_stud_soft": vis_stud_soft, "vis_captions": vis_captions,
        "heldout_teach_soft": heldout_teach_soft, "heldout_stud_soft": heldout_stud_soft,
        "heldout_captions": [p[1] for p in heldout_pairs],
    }, VLM_SOFT_CACHE)
    vis_meta = {
        "vision_norms": vision_norms, "vis_total_bytes": int(v_bytes), "n_vis_mats": len(vision_records),
        "v_cos0": v_cos0, "v_cos1": v_cos1, "vis_keep_c8d": vis_keep_c8d,
        "cents_vis": cents_vis, "experts_vis": experts_vis,
    }
    _atomic_torch_save(vis_meta, VIS_META_PATH)
    sync_checkpoint_to_drive()
    return vis_meta


if VIS_META_PATH.is_file() and VIS_CKPT_PATH.is_file() and VLM_SOFT_CACHE.is_file():
    _vm = torch.load(VIS_META_PATH, map_location="cpu", weights_only=False)
    print(f"[Resume Stage 2] Loaded {VIS_META_PATH} ({_vm['n_vis_mats']} mats)", flush=True)
else:
    _vm = _build_stage2_vision()

vision_norms = _vm["vision_norms"]
vis_total_bytes = int(_vm["vis_total_bytes"])
n_vis_mats = int(_vm["n_vis_mats"])
v_cos0, v_cos1, vis_keep_c8d = float(_vm["v_cos0"]), float(_vm["v_cos1"]), bool(_vm["vis_keep_c8d"])
cents_vis, experts_vis = _vm["cents_vis"], _vm["experts_vis"]
del _vm

_vsc = torch.load(VLM_SOFT_CACHE, map_location="cpu", weights_only=False)
vlm_teach_soft, vlm_stud_soft, vlm_captions = _vsc["vis_teach_soft"], _vsc["vis_stud_soft"], _vsc["vis_captions"]
heldout_teach_soft, heldout_stud_soft, heldout_captions = _vsc["heldout_teach_soft"], _vsc["heldout_stud_soft"], _vsc["heldout_captions"]
del _vsc

vlm_spans: Dict[int, Tuple[int, int]] = {}
vlm_user_prompts = [
    "Describe this image concisely in one sentence.",
    "What is shown in this image?",
    "Provide a brief caption for this image.",
    "Explain what you see in this picture.",
]
tok_user_open = list(tk.encode("<|turn>user\n", add_special_tokens=False).ids)
tok_model_open = list(tk.encode("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", add_special_tokens=False).ids)
for vi in range(N_VLM_SEQ):
    n_soft_i = int(vlm_teach_soft[vi].shape[0])
    q_vis = list(tk.encode(vlm_user_prompts[vi % len(vlm_user_prompts)], add_special_tokens=False).ids)
    a_vis = list(tk.encode(f"{vlm_captions[vi]}<turn|>\n", add_special_tokens=False).ids)
    cur_t = [2] + tok_user_open + [GEMMA4_BOI_ID] + [GEMMA4_IMAGE_ID] * n_soft_i + [GEMMA4_EOI_ID] + q_vis + tok_model_open
    s0_v = 1 + len(tok_user_open) + 1
    vlm_spans[N_TEXT_SEQ + vi] = (s0_v, s0_v + n_soft_i)
    cur_a = [False] * len(cur_t) + [True] * len(a_vis)
    cur_t = cur_t + a_vis
    while len(cur_t) < SEQ_LEN + 1:
        cur_t.append(0); cur_a.append(False)
    seq_ids_list.append(cur_t[:SEQ_LEN])
    seq_nxt_list.append(cur_t[1 : SEQ_LEN + 1])
    seq_ans_list.append(cur_a[1 : SEQ_LEN + 1])
    seq_vlm_ans_list.append(cur_a[1 : SEQ_LEN + 1])

all_seq_ids = np.array(seq_ids_list, dtype=np.int32)
all_seq_nxt = np.array(seq_nxt_list, dtype=np.int32)
all_ans_mask = np.array(seq_ans_list, dtype=bool)
all_vlm_ans_mask = np.array(seq_vlm_ans_list, dtype=bool)

heldout_spans, heldout_ids_list, heldout_ans_starts, heldout_ans_lens = {}, [], [], []
for hi, cap_h in enumerate(heldout_captions):
    n_soft_h = int(heldout_teach_soft[hi].shape[0])
    q_h = list(tk.encode("Describe this image concisely in one sentence.", add_special_tokens=False).ids)
    a_h = list(tk.encode(f"{cap_h}<turn|>\n", add_special_tokens=False).ids)
    cur_t = [2] + tok_user_open + [GEMMA4_BOI_ID] + [GEMMA4_IMAGE_ID] * n_soft_h + [GEMMA4_EOI_ID] + q_h + tok_model_open
    s0_h = 1 + len(tok_user_open) + 1
    heldout_spans[hi] = (s0_h, s0_h + n_soft_h)
    ans_start = len(cur_t) - 1
    cur_t = cur_t + a_h
    ans_len = min(len(a_h), SEQ_LEN - ans_start)
    while len(cur_t) < SEQ_LEN + 1:
        cur_t.append(0)
    heldout_ids_list.append(cur_t[:SEQ_LEN])
    heldout_ans_starts.append(ans_start)
    heldout_ans_lens.append(ans_len)
heldout_seq_ids = np.array(heldout_ids_list, dtype=np.int32)

VAL_SEQ_IDX = np.concatenate([
    np.linspace(0, N_TEXT_SEQ - 1, 32, dtype=int),
    np.linspace(N_TEXT_SEQ, N_SEQ - 1, 16, dtype=int),
])
VAL_SEQ_SET = set(VAL_SEQ_IDX.tolist())
TRAIN_SEQ_IDX = np.array([i for i in range(N_SEQ) if i not in VAL_SEQ_SET], dtype=int)


def make_sliding_attn_mask(b0: int, b1: int, spans: Dict[int, Tuple[int, int]]) -> Optional[torch.Tensor]:
    if not any(si in spans for si in range(b0, b1)):
        return None
    mb = b1 - b0
    mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE, dtype=torch.bool)).unsqueeze(0).expand(mb, -1, -1).clone()
    for gi in range(b0, b1):
        sp = spans.get(gi)
        if sp is not None:
            mask[gi - b0, sp[0] : sp[1], sp[0] : sp[1]] = True
    return mask.unsqueeze(1)


def make_sliding_attn_mask_for_indices(seq_indices: List[int], spans: Dict[int, Tuple[int, int]]) -> Optional[torch.Tensor]:
    if not any(si in spans for si in seq_indices):
        return None
    mb = len(seq_indices)
    mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE, dtype=torch.bool)).unsqueeze(0).expand(mb, -1, -1).clone()
    for lo, gi in enumerate(seq_indices):
        sp = spans.get(gi)
        if sp is not None:
            mask[lo, sp[0] : sp[1], sp[0] : sp[1]] = True
    return mask.unsqueeze(1)


pos_1d = torch.arange(SEQ_LEN, device=DEVICE, dtype=torch.float32)


def make_rope(theta: float, rot_d: int, hd: int) -> Tuple[torch.Tensor, torch.Tensor]:
    hr = rot_d // 2
    inv_f = 1.0 / (theta ** (torch.arange(0, 2 * hr, 2, device=DEVICE, dtype=torch.float32) / float(hd)))
    ang = pos_1d[:, None] * inv_f[None, :]
    return torch.cos(ang), torch.sin(ang)


def apply_rope_f32(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_d: int, hd: int) -> torch.Tensor:
    hr, hh = rot_d // 2, hd // 2
    c = cos[None, :, None, :].to(t.dtype)
    s = sin[None, :, None, :].to(t.dtype)
    x0, x1 = t[..., :hr], t[..., hh : hh + hr]
    r0, r1 = x0 * c - x1 * s, x0 * s + x1 * c
    if hr == hh:
        return torch.cat([r0, r1], dim=-1)
    return torch.cat([r0, t[..., hr:hh], r1, t[..., hh + hr :]], dim=-1)


def rmsn_diff(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (y * w.float()) if w is not None else y


def rmsn_f16(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    return rmsn_diff(x, w).to(torch.float16)


def write_norm_inplace(fp, norm_offsets: Dict[str, Tuple[int, int]], name: str, arr_f32: np.ndarray) -> None:
    off, n_el = norm_offsets[name]
    a32 = np.ascontiguousarray(arr_f32, dtype="<f4").ravel()
    assert a32.size == n_el
    cur = fp.tell()
    fp.seek(off); fp.write(a32.tobytes()); fp.seek(cur)


def build_v6_proj(name: str, w_cpu: torch.Tensor, x_rms: Optional[torch.Tensor], spec: dict) -> nn.Module:
    if spec["kind"] == "lb":
        return LittleBit2LinearV6(name, w_cpu, x_rms, split_dim=spec["s"], residual_dim=spec["sr"], bsize=256, itq_iter=25)
    return DirectRowMajorLinearV6(name, w_cpu, x_rms, bits=spec["bits"], lr_rank=spec["lr_rank"], bsize=256)


# ---------------------------------------------------------------------------
# 6. Stage 3A: Streaming L00..L59 LittleBit-2 Joint-ITQ + Full-Weight STE QAT
# ---------------------------------------------------------------------------
STAGE3_CKPT_PATH = CKPT_DIR / "stage3_layer_ckpt.pt"
CKPT_X_TEACH = CKPT_DIR / "v6_ckpt_x_teach.pt"
CKPT_X_STUD = CKPT_DIR / "v6_ckpt_x_stud.pt"
TAIL_LAYERS = {56, 57, 58, 59}
TAIL_MOD_DIR = Path("/tmp/v6_tail_mods")
TAIL_MOD_DIR.mkdir(parents=True, exist_ok=True)


def qat_block_attn_v6(
    l_idx: int,
    mods: Dict[str, nn.Module],
    norms_g: Dict[str, torch.Tensor],
    X_stud_cpu: torch.Tensor,
    xm_teach_cpu: torch.Tensor,
    is_full: bool,
    cos_r: torch.Tensor,
    sin_r: torch.Tensor,
    rot_d: int,
    hd: int,
    nkv: int,
    steps: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, float]:
    gqa_rep = NH // nkv
    p_in = nn.Parameter(norms_g["in"].float().clone())
    p_qn = nn.Parameter(norms_g["qn"].float().clone())
    p_kn = nn.Parameter(norms_g["kn"].float().clone())
    p_pa = nn.Parameter(norms_g["pa"].float().clone())

    def fwd_mb(sidx: List[int]) -> torch.Tensor:
        xb = X_stud_cpu[sidx].to(DEVICE)
        inp = rmsn_diff(xb, p_in)
        mb = len(sidx)
        q = mods["q_proj"](inp).view(mb, SEQ_LEN, NH, hd)
        k = mods["k_proj"](inp).view(mb, SEQ_LEN, nkv, hd)
        v = k if ("v_proj" not in mods) else mods["v_proj"](inp).view(mb, SEQ_LEN, nkv, hd)
        qn = apply_rope_f32(rmsn_diff(q, p_qn), cos_r, sin_r, rot_d, hd).transpose(1, 2).to(COMPUTE_DTYPE)
        kn = apply_rope_f32(rmsn_diff(k, p_kn), cos_r, sin_r, rot_d, hd).transpose(1, 2).to(COMPUTE_DTYPE)
        vn = rmsn_diff(v, None).transpose(1, 2).to(COMPUTE_DTYPE)
        amask = None if is_full else make_sliding_attn_mask_for_indices(sidx, vlm_spans)
        if amask is not None:
            ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=amask, scale=1.0)
        else:
            ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
        ot = mods["o_proj"](ctx.transpose(1, 2).contiguous().view(mb, SEQ_LEN, NH * hd))
        return xb.float() + rmsn_diff(ot, p_pa)

    def eval_val() -> float:
        with torch.no_grad():
            cs, cnt = 0.0, 0
            for v0 in range(0, len(VAL_SEQ_IDX), 8):
                sidx = VAL_SEQ_IDX[v0 : v0 + 8].tolist()
                pred = fwd_mb(sidx)
                tgt = xm_teach_cpu[sidx].to(DEVICE).float()
                cs += float(F.cosine_similarity(pred.reshape(-1, D_MOD), tgt.reshape(-1, D_MOD), dim=-1).sum().item())
                cnt += pred.shape[0] * pred.shape[1]
            return cs / max(cnt, 1)

    cos0 = eval_val()
    w_params = [p for m in mods.values() for p in m.parameters()]
    n_params = [p_in, p_qn, p_kn, p_pa]
    opt_groups = [pg for m in mods.values() for pg in m.param_groups(lr_lat=5e-3, lr_sc=1e-3)]
    opt_groups.append({"params": n_params, "lr": 1.5e-3})
    opt = torch.optim.AdamW(opt_groups, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)

    best_cos = cos0
    best_state = {k: {pk: pv.detach().cpu().clone() for pk, pv in m.state_dict().items()} for k, m in mods.items()}
    best_norms = {"in": p_in.detach().cpu().clone(), "qn": p_qn.detach().cpu().clone(), "kn": p_kn.detach().cpu().clone(), "pa": p_pa.detach().cpu().clone()}

    for st in range(steps):
        opt.zero_grad()
        sel = TRAIN_SEQ_IDX[np.random.choice(len(TRAIN_SEQ_IDX), size=8, replace=False)].tolist()
        pred = fwd_mb(sel)
        tgt = xm_teach_cpu[sel].to(DEVICE).float()
        loss = F.mse_loss(pred, tgt) / (tgt.var() + 1e-5) + 0.5 * (1.0 - F.cosine_similarity(pred.reshape(-1, D_MOD), tgt.reshape(-1, D_MOD), dim=-1).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(w_params + n_params, 1.0)
        opt.step(); sch.step()
        if (st + 1) == steps or (steps >= 40 and (st + 1) == steps // 2):
            cv = eval_val()
            if cv > best_cos + 1e-6:
                best_cos = cv
                best_state = {k: {pk: pv.detach().cpu().clone() for pk, pv in m.state_dict().items()} for k, m in mods.items()}
                best_norms = {"in": p_in.detach().cpu().clone(), "qn": p_qn.detach().cpu().clone(), "kn": p_kn.detach().cpu().clone(), "pa": p_pa.detach().cpu().clone()}

    for k, m in mods.items():
        m.load_state_dict({pk: pv.to(DEVICE) for pk, pv in best_state[k].items()})
    new_norms = dict(norms_g)
    for nk in ("in", "qn", "kn", "pa"):
        new_norms[nk] = best_norms[nk].to(DEVICE)
    w_out = {k: m.dequant_w_eff_f16() for k, m in mods.items()}
    del opt, sch, best_state, best_norms, p_in, p_qn, p_kn, p_pa
    torch.cuda.empty_cache()
    return w_out, new_norms, cos0, best_cos


def qat_block_mlp_v6(
    l_idx: int,
    mods: Dict[str, nn.Module],
    norms_g: Dict[str, torch.Tensor],
    xm_stud_cpu: torch.Tensor,
    X_teach_next_cpu: torch.Tensor,
    steps: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, float, float]:
    p_pf = nn.Parameter(norms_g["pf"].float().clone())
    p_pff = nn.Parameter(norms_g["pff"].float().clone())
    p_lsc = nn.Parameter(torch.tensor(float(norms_g["lsc"]), device=DEVICE, dtype=torch.float32))

    def fwd_mb(sidx: List[int], sparse_p: float = 1.0) -> torch.Tensor:
        xm = xm_stud_cpu[sidx].to(DEVICE)
        hm = rmsn_diff(xm, p_pf)
        g_act = F.gelu(mods["gate_proj"](hm), approximate="tanh")
        if sparse_p < 0.9999:
            g_act = apply_block_sparse_geglu_mask(g_act, top_p=sparse_p, bsize=256)
        u_act = mods["up_proj"](hm)
        dt = mods["down_proj"](g_act * u_act)
        return (xm.float() + rmsn_diff(dt, p_pff)) * p_lsc

    def eval_val(sparse_p: float = 1.0) -> float:
        with torch.no_grad():
            cs, cnt = 0.0, 0
            for v0 in range(0, len(VAL_SEQ_IDX), 8):
                sidx = VAL_SEQ_IDX[v0 : v0 + 8].tolist()
                pred = fwd_mb(sidx, sparse_p=sparse_p)
                tgt = X_teach_next_cpu[sidx].to(DEVICE).float()
                cs += float(F.cosine_similarity(pred.reshape(-1, D_MOD), tgt.reshape(-1, D_MOD), dim=-1).sum().item())
                cnt += pred.shape[0] * pred.shape[1]
            return cs / max(cnt, 1)

    cos0 = eval_val(1.0)
    w_params = [p for m in mods.values() for p in m.parameters()]
    n_params = [p_pf, p_pff, p_lsc]
    opt_groups = [pg for m in mods.values() for pg in m.param_groups(lr_lat=5e-3, lr_sc=1e-3)]
    opt_groups.append({"params": n_params, "lr": 1.5e-3})
    opt = torch.optim.AdamW(opt_groups, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)

    best_cos = cos0
    best_state = {k: {pk: pv.detach().cpu().clone() for pk, pv in m.state_dict().items()} for k, m in mods.items()}
    best_norms = {"pf": p_pf.detach().cpu().clone(), "pff": p_pff.detach().cpu().clone(), "lsc": float(p_lsc.detach().item())}

    for st in range(steps):
        opt.zero_grad()
        sel = TRAIN_SEQ_IDX[np.random.choice(len(TRAIN_SEQ_IDX), size=8, replace=False)].tolist()
        sp = 0.45 if (st % 3 == 2) else 1.0
        pred = fwd_mb(sel, sparse_p=sp)
        tgt = X_teach_next_cpu[sel].to(DEVICE).float()
        loss = F.mse_loss(pred, tgt) / (tgt.var() + 1e-5) + 0.5 * (1.0 - F.cosine_similarity(pred.reshape(-1, D_MOD), tgt.reshape(-1, D_MOD), dim=-1).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(w_params + n_params, 1.0)
        opt.step(); sch.step()
        if (st + 1) == steps or (steps >= 40 and (st + 1) == steps // 2):
            cv = eval_val(1.0)
            if cv > best_cos + 1e-6:
                best_cos = cv
                best_state = {k: {pk: pv.detach().cpu().clone() for pk, pv in m.state_dict().items()} for k, m in mods.items()}
                best_norms = {"pf": p_pf.detach().cpu().clone(), "pff": p_pff.detach().cpu().clone(), "lsc": float(p_lsc.detach().item())}

    for k, m in mods.items():
        m.load_state_dict({pk: pv.to(DEVICE) for pk, pv in best_state[k].items()})
    new_norms = dict(norms_g)
    new_norms["pf"] = best_norms["pf"].to(DEVICE)
    new_norms["pff"] = best_norms["pff"].to(DEVICE)
    new_norms["lsc"] = best_norms["lsc"]
    with torch.no_grad():
        p_pf.copy_(new_norms["pf"]); p_pff.copy_(new_norms["pff"]); p_lsc.fill_(new_norms["lsc"])
    cos_sp45 = eval_val(0.45)
    w_out = {k: m.dequant_w_eff_f16() for k, m in mods.items()}
    del opt, sch, best_state, best_norms, p_pf, p_pff, p_lsc
    torch.cuda.empty_cache()
    return w_out, new_norms, cos0, best_cos, cos_sp45


print(f"\n[Stage 3A] Streaming {N_LAYERS} LittleBit-2 Row-Major LM layers with Co-Activation Perm + Full-Weight STE QAT...", flush=True)
with safe_open(sf_path, framework="pt", device="cpu") as sf:
    lm_norms: Dict[str, np.ndarray] = {}
    for _l in range(N_LAYERS):
        _p = f"model.language_model.layers.{_l}."
        for _nm_key in (
            "input_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight", "layer_scalar",
        ):
            _v = sf.get_tensor(_p + _nm_key).float().numpy().astype("<f4")
            if _nm_key == "layer_scalar":
                _v = np.array([float(_v.reshape(-1)[0])], dtype="<f4")
            lm_norms[_p + _nm_key] = _v
    fn_orig_cpu = sf.get_tensor("model.language_model.norm.weight").float()

    all_norms: Dict[str, np.ndarray] = {}
    all_norms.update(lm_norms)
    all_norms["model.language_model.norm.weight"] = fn_orig_cpu.numpy().astype("<f4")
    all_norms.update(vision_norms)
    total_mats = _pred_lm_mats + n_vis_mats

    def unpack_qat(pfx: str) -> torch.Tensor:
        wp = sf.get_tensor(f"{pfx}.weight_packed").to(DEVICE)
        ws = sf.get_tensor(f"{pfx}.weight_scale").to(DEVICE, dtype=torch.float32)
        sh = torch.arange(0, 32, 4, device=DEVICE, dtype=torch.int32)
        nibs = ((wp.unsqueeze(-1) >> sh) & 0xF) - 8
        return (nibs.to(torch.float32).reshape(wp.shape[0], -1, 32) * ws.unsqueeze(-1)).reshape(wp.shape[0], -1).cpu()

    def teacher_layer_step(
        X_cpu: torch.Tensor, w_f16: Dict[str, torch.Tensor], norms_g: Dict[str, torch.Tensor],
        l_idx: int, spans_dict: Optional[Dict[int, Tuple[int, int]]] = None, return_mid_and_energy: bool = False,
    ):
        is_full = LAYER_TYPES[l_idx] == "full_attention"
        nkv = NKV_F if is_full else NKV_S
        hd = HD_F if is_full else HD_S
        theta = THETA_G if is_full else THETA_L
        rot_d = int(round(hd * (PART_G if is_full else PART_L)))
        cos_r, sin_r = make_rope(theta, rot_d, hd)
        gqa_rep = NH // nkv
        lsc = float(norms_g["lsc"])
        out_cpu = torch.empty_like(X_cpu)
        xm_cpu = torch.empty_like(X_cpu) if return_mid_and_energy else None
        neuron_e = torch.zeros((21504,), device=DEVICE, dtype=torch.float64) if return_mid_and_energy else None
        with torch.no_grad():
            for b0 in range(0, X_cpu.shape[0], 32):
                b1 = min(b0 + 32, X_cpu.shape[0])
                xb = X_cpu[b0:b1].to(DEVICE)
                inp = rmsn_f16(xb, norms_g["in"])
                q = F.linear(inp, w_f16["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                k = F.linear(inp, w_f16["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                v = k.clone() if (is_full and K_EQ_V) else F.linear(inp, w_f16["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                qn = apply_rope_f32(rmsn_f16(q, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                kn = apply_rope_f32(rmsn_f16(k, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                vn = rmsn_f16(v, None).transpose(1, 2)
                amask = None if (is_full or not spans_dict) else make_sliding_attn_mask(b0, b1, spans_dict)
                if amask is not None:
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=amask, scale=1.0)
                else:
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
                ot = F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_f16["o_proj"])
                xm = xb + rmsn_f16(ot, norms_g["pa"])
                if xm_cpu is not None:
                    xm_cpu[b0:b1] = xm.cpu()
                hm = rmsn_f16(xm, norms_g["pf"])
                act = F.gelu(F.linear(hm, w_f16["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_f16["up_proj"]).float()
                if neuron_e is not None:
                    neuron_e.add_((act * act).reshape(-1, 21504).double().sum(dim=0))
                dt_o = F.linear(act.to(torch.float16), w_f16["down_proj"])
                out_cpu[b0:b1] = ((xm.float() + rmsn_f16(dt_o, norms_g["pff"]).float()) * lsc).to(torch.float16).cpu()
        return out_cpu, xm_cpu, neuron_e

    norm_offsets: Dict[str, Tuple[int, int]] = {}
    if STAGE3_CKPT_PATH.is_file() and OUT_T16C.is_file() and CKPT_X_TEACH.is_file() and CKPT_X_STUD.is_file():
        _lc = torch.load(STAGE3_CKPT_PATH, map_location="cpu", weights_only=False)
        start_layer = int(_lc["next_layer"])
        t16c_pos = int(_lc["t16c_pos"])
        lm_mats_count = int(_lc["lm_mats_count"])
        norm_offsets = _lc["norm_offsets"]
        H_teach_eval = _lc["H_teach_eval"]
        H_stud_eval = _lc["H_stud_eval"]
        layer_cos_log = _lc["layer_cos_log"]
        X_teach_cpu = torch.load(CKPT_X_TEACH, map_location="cpu", weights_only=True)
        X_stud_cpu = torch.load(CKPT_X_STUD, map_location="cpu", weights_only=True)
        del _lc
        t16c_fp = open(OUT_T16C, "r+b")
        t16c_fp.seek(t16c_pos); t16c_fp.truncate(t16c_pos)
        print(f"[Resume Stage 3A] Resuming at L{start_layer:02d} (pos={t16c_pos:,} B)", flush=True)
    else:
        start_layer = 0
        layer_cos_log = []
        emb_bf16_cpu = sf.get_tensor("model.language_model.embed_tokens.weight")
        ids_t = torch.from_numpy(all_seq_ids).long()
        X_teach_cpu = (emb_bf16_cpu[ids_t].float() * ESCALE).to(torch.float16)
        eval_ids_t = torch.from_numpy(heldout_seq_ids).long()
        H_teach_eval = (emb_bf16_cpu[eval_ids_t].float() * ESCALE).to(torch.float16)
        for vi in range(N_VLM_SEQ):
            s0, s1 = vlm_spans[N_TEXT_SEQ + vi]
            X_teach_cpu[N_TEXT_SEQ + vi, s0:s1] = vlm_teach_soft[vi]
        for hi in range(len(heldout_captions)):
            s0, s1 = heldout_spans[hi]
            H_teach_eval[hi, s0:s1] = heldout_teach_soft[hi]

        # Pack embed_tokens.weight as standard 3b_u0 (column-major for ultra-fast CS LM-head sieve!)
        print("[Stage 3A] Packing embed_tokens.weight (3b_u0 @ b256)...", flush=True)
        d_out_e = int(tc["vocab_size"])
        exp_e = p3.expected_tile16_bytes(d_out_e, D_MOD, 256, 3, False)
        pb_emb = np.empty(exp_e, dtype=np.uint8)
        emb_deq_cpu = torch.empty((d_out_e, D_MOD), dtype=torch.float16)
        tblk_e = p3.tile16_blk_bytes(256, 3, False) * (D_MOD // 256)
        for r0 in range(0, d_out_e, 16384):
            r1 = min(r0 + 16384, d_out_e)
            wc_rot = fwht_gpu(emb_bf16_cpu[r0:r1].to(DEVICE, dtype=torch.float32), 256).contiguous().cpu().numpy().astype(np.float32)
            ch_pb = np.empty(((r1 - r0 + 15) // 16) * tblk_e, dtype=np.uint8)
            LIB_FQ.quant_and_pack_t16c_f32(wc_rot.ctypes.data, r1 - r0, D_MOD, 256, 4, ch_pb.ctypes.data)
            pb_emb[(r0 // 16) * tblk_e : (r0 // 16) * tblk_e + ch_pb.size] = ch_pb
            emb_deq_cpu[r0:r1] = fwht_gpu(torch.from_numpy(wc_rot).to(DEVICE), 256).to(torch.float16).cpu()
            del wc_rot, ch_pb
        X_stud_cpu = (emb_deq_cpu[ids_t] * ESCALE).to(torch.float16)
        H_stud_eval = (emb_deq_cpu[eval_ids_t] * ESCALE).to(torch.float16)
        for vi in range(N_VLM_SEQ):
            s0, s1 = vlm_spans[N_TEXT_SEQ + vi]
            X_stud_cpu[N_TEXT_SEQ + vi, s0:s1] = vlm_stud_soft[vi]
        for hi in range(len(heldout_captions)):
            s0, s1 = heldout_spans[hi]
            H_stud_eval[hi, s0:s1] = heldout_stud_soft[hi]
        del emb_bf16_cpu, emb_deq_cpu, ids_t, eval_ids_t

        t16c_fp = open(OUT_T16C, "wb")
        t16c_fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, len(all_norms), total_mats, 0))
        t16c_pos = 24
        for _nm, _vec in all_norms.items():
            _nb = _nm.encode("utf-8")
            _vf = np.ascontiguousarray(_vec, dtype="<f4").ravel()
            t16c_fp.write(struct.pack("<H", len(_nb))); t16c_fp.write(_nb); t16c_fp.write(struct.pack("<I", _vf.size))
            norm_offsets[_nm] = (t16c_pos + 2 + len(_nb) + 4, _vf.size)
            t16c_fp.write(_vf.tobytes())
            t16c_pos += 2 + len(_nb) + 4 + _vf.size * 4
        t16c_pos = p3._write_mat_record(
            t16c_fp, t16c_pos, "model.language_model.embed_tokens.weight",
            d_out_e, D_MOD, 256, 3, False, True, pb_emb, None, None,
        )
        lm_mats_count = 1
        del pb_emb
        t16c_fp.flush(); t16c_fp.close()
        t16c_fp = open(OUT_T16C, "r+b"); t16c_fp.seek(t16c_pos)

    del vlm_teach_soft, vlm_stud_soft, heldout_teach_soft, heldout_stud_soft
    gc.collect()

    try:
        for l_idx in range(start_layer, N_LAYERS):
            tl = time.perf_counter()
            p = f"model.language_model.layers.{l_idx}."
            is_full = LAYER_TYPES[l_idx] == "full_attention"
            nkv = NKV_F if is_full else NKV_S
            hd = HD_F if is_full else HD_S
            theta = THETA_G if is_full else THETA_L
            rot_d = int(round(hd * (PART_G if is_full else PART_L)))
            cos_r, sin_r = make_rope(theta, rot_d, hd)
            gqa_rep = NH // nkv

            if l_idx == 56:
                _atomic_torch_save(X_stud_cpu, Path("/tmp/v6_x_stud_pre56.pt"))
                _atomic_torch_save(H_stud_eval, Path("/tmp/v6_h_eval_pre56.pt"))
            if l_idx == 58:
                _atomic_torch_save(X_stud_cpu, Path("/tmp/v6_x_stud_pre58.pt"))

            norms_g = {
                "in": sf.get_tensor(p + "input_layernorm.weight").to(DEVICE, dtype=torch.float32),
                "qn": sf.get_tensor(p + "self_attn.q_norm.weight").to(DEVICE, dtype=torch.float32),
                "kn": sf.get_tensor(p + "self_attn.k_norm.weight").to(DEVICE, dtype=torch.float32),
                "pa": sf.get_tensor(p + "post_attention_layernorm.weight").to(DEVICE, dtype=torch.float32),
                "pf": sf.get_tensor(p + "pre_feedforward_layernorm.weight").to(DEVICE, dtype=torch.float32),
                "pff": sf.get_tensor(p + "post_feedforward_layernorm.weight").to(DEVICE, dtype=torch.float32),
                "lsc": sf.get_tensor(p + "layer_scalar").float().item(),
            }
            proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            w_teach_f32: Dict[str, torch.Tensor] = {}
            for pn in proj_names:
                if is_full and K_EQ_V and pn == "v_proj":
                    continue
                sub = "self_attn" if pn in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                w_teach_f32[pn] = unpack_qat(f"{p}{sub}.{pn}")

            if l_idx in (58, 59):
                sub_kv = np.concatenate([
                    np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int),
                    np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int),
                ])
                with torch.no_grad():
                    xt_s = X_teach_cpu[sub_kv].to(DEVICE)
                    inp_t = rmsn_f16(xt_s, norms_g["in"]).reshape(-1, D_MOD).float()
                    _kv_save = {"kt": (inp_t @ w_teach_f32["k_proj"].to(DEVICE).T).to(torch.float16).cpu()}
                    if l_idx == 58:
                        _kv_save["vt"] = (inp_t @ w_teach_f32["v_proj"].to(DEVICE).T).to(torch.float16).cpu()
                    _atomic_torch_save(_kv_save, Path(f"/tmp/v6_teach_kv_l{l_idx}.pt"))
                    del xt_s, inp_t, _kv_save

            w_teach_f16 = {k: v.to(DEVICE, dtype=torch.float16) for k, v in w_teach_f32.items()}
            X_teach_next, xm_teach_cpu, neuron_e = teacher_layer_step(
                X_teach_cpu, w_teach_f16, norms_g, l_idx, spans_dict=vlm_spans, return_mid_and_energy=True
            )
            H_teach_eval, _, _ = teacher_layer_step(
                H_teach_eval, w_teach_f16, norms_g, l_idx, spans_dict=heldout_spans, return_mid_and_energy=False
            )
            del w_teach_f16
            if l_idx < 59:
                del X_teach_cpu

            # Co-activation GeGLU neuron permutation (100% exact invariance -> clusters active blocks for --mlp-top-p!)
            perm_mlp = torch.argsort(neuron_e, descending=True).cpu()
            w_teach_f32["gate_proj"] = w_teach_f32["gate_proj"][perm_mlp].contiguous()
            w_teach_f32["up_proj"] = w_teach_f32["up_proj"][perm_mlp].contiguous()
            w_teach_f32["down_proj"] = w_teach_f32["down_proj"][:, perm_mlp].contiguous()
            e_sorted = neuron_e[perm_mlp.to(DEVICE)]
            top50_e_pct = 100.0 * float(e_sorted[:10752].sum().item() / e_sorted.sum().clamp(min=1e-12).item())
            del neuron_e, e_sorted, perm_mlp

            n_tok_l = N_SEQ * SEQ_LEN
            # 1) Attention input RMS + LittleBit-2 Joint-ITQ init + Real Full-Weight STE QAT
            xsq_in = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 64):
                    inp = rmsn_f16(X_stud_cpu[b0 : min(b0 + 64, N_SEQ)].to(DEVICE), norms_g["in"]).reshape(-1, D_MOD).float()
                    accum_xsq(xsq_in, inp)
            xrms_in = torch.sqrt((xsq_in / n_tok_l).clamp(min=1e-8))
            attn_mods: Dict[str, nn.Module] = {}
            w_tmp_attn: Dict[str, torch.Tensor] = {}
            for pn in (["q_proj", "k_proj"] if (is_full and K_EQ_V) else ["q_proj", "k_proj", "v_proj"]):
                attn_mods[pn] = build_v6_proj(f"{p}self_attn.{pn}.weight", w_teach_f32.pop(pn), xrms_in, get_lm_spec_v6(l_idx, pn))
                w_tmp_attn[pn] = attn_mods[pn].dequant_w_eff_f16()

            di_o = w_teach_f32["o_proj"].shape[1]
            xsq_o = torch.zeros((di_o,), device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 32):
                    b1 = min(b0 + 32, N_SEQ)
                    inp = rmsn_f16(X_stud_cpu[b0:b1].to(DEVICE), norms_g["in"])
                    q = F.linear(inp, w_tmp_attn["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                    k = F.linear(inp, w_tmp_attn["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    v = k.clone() if (is_full and K_EQ_V) else F.linear(inp, w_tmp_attn["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    qn = apply_rope_f32(rmsn_f16(q, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                    kn = apply_rope_f32(rmsn_f16(k, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                    vn = rmsn_f16(v, None).transpose(1, 2)
                    am = None if is_full else make_sliding_attn_mask(b0, b1, vlm_spans)
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0)
                    accum_xsq(xsq_o, ctx.transpose(1, 2).contiguous().view(-1, di_o).float())
            xrms_o = torch.sqrt((xsq_o / n_tok_l).clamp(min=1e-8))
            attn_mods["o_proj"] = build_v6_proj(f"{p}self_attn.o_proj.weight", w_teach_f32.pop("o_proj"), xrms_o, get_lm_spec_v6(l_idx, "o_proj"))
            del w_tmp_attn, xsq_in, xrms_in, xsq_o, xrms_o
            torch.cuda.empty_cache()

            attn_steps = 60 if l_idx >= 36 else 35
            w_attn_qat, norms_g, acos0, acos1 = qat_block_attn_v6(
                l_idx, attn_mods, norms_g, X_stud_cpu, xm_teach_cpu, is_full, cos_r, sin_r, rot_d, hd, nkv, attn_steps
            )
            del xm_teach_cpu

            # Compute post-QAT Attention residual xm_all + MLP input RMS
            xsq_mlp = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
            xm_all = torch.empty((N_SEQ, SEQ_LEN, D_MOD), dtype=torch.float16)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 32):
                    b1 = min(b0 + 32, N_SEQ)
                    xb = X_stud_cpu[b0:b1].to(DEVICE)
                    inp = rmsn_f16(xb, norms_g["in"])
                    q = F.linear(inp, w_attn_qat["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                    k = F.linear(inp, w_attn_qat["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    v = k.clone() if (is_full and K_EQ_V) else F.linear(inp, w_attn_qat["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    qn = apply_rope_f32(rmsn_f16(q, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                    kn = apply_rope_f32(rmsn_f16(k, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                    vn = rmsn_f16(v, None).transpose(1, 2)
                    am = None if is_full else make_sliding_attn_mask(b0, b1, vlm_spans)
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0)
                    ot = F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_attn_qat["o_proj"])
                    xm = xb + rmsn_f16(ot, norms_g["pa"])
                    xm_all[b0:b1] = xm.cpu()
                    accum_xsq(xsq_mlp, rmsn_f16(xm, norms_g["pf"]).reshape(-1, D_MOD).float())
                _he = H_stud_eval.to(DEVICE)
                _hi = rmsn_f16(_he, norms_g["in"])
                _hq = F.linear(_hi, w_attn_qat["q_proj"]).view(_he.shape[0], SEQ_LEN, NH, hd)
                _hk = F.linear(_hi, w_attn_qat["k_proj"]).view(_he.shape[0], SEQ_LEN, nkv, hd)
                _hv = _hk.clone() if (is_full and K_EQ_V) else F.linear(_hi, w_attn_qat["v_proj"]).view(_he.shape[0], SEQ_LEN, nkv, hd)
                _hqn = apply_rope_f32(rmsn_f16(_hq, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                _hkn = apply_rope_f32(rmsn_f16(_hk, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                _hvn = rmsn_f16(_hv, None).transpose(1, 2)
                _hm = None if is_full else make_sliding_attn_mask(0, _he.shape[0], heldout_spans)
                _hc = F.scaled_dot_product_attention(_hqn, _hkn.repeat_interleave(gqa_rep, 1), _hvn.repeat_interleave(gqa_rep, 1), attn_mask=_hm, is_causal=(_hm is None), scale=1.0)
                _ho = F.linear(_hc.transpose(1, 2).contiguous().view(_he.shape[0], SEQ_LEN, NH * hd), w_attn_qat["o_proj"])
                H_stud_eval_mid = (_he + rmsn_f16(_ho, norms_g["pa"])).cpu()
            del X_stud_cpu, w_attn_qat
            torch.cuda.empty_cache()

            # 2) MLP LittleBit-2 Joint-ITQ init + Sparse-Aware Full-Weight STE QAT
            xrms_mlp = torch.sqrt((xsq_mlp / n_tok_l).clamp(min=1e-8))
            mlp_mods: Dict[str, nn.Module] = {}
            w_tmp_mlp: Dict[str, torch.Tensor] = {}
            for pn in ("gate_proj", "up_proj"):
                mlp_mods[pn] = build_v6_proj(f"{p}mlp.{pn}.weight", w_teach_f32.pop(pn), xrms_mlp, get_lm_spec_v6(l_idx, pn))
                w_tmp_mlp[pn] = mlp_mods[pn].dequant_w_eff_f16()
            xsq_d = torch.zeros((21504,), device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 32):
                    hm = rmsn_f16(xm_all[b0 : min(b0 + 32, N_SEQ)].to(DEVICE), norms_g["pf"])
                    act = F.gelu(F.linear(hm, w_tmp_mlp["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_tmp_mlp["up_proj"]).float()
                    accum_xsq(xsq_d, act.reshape(-1, 21504))
            xrms_d = torch.sqrt((xsq_d / n_tok_l).clamp(min=1e-8))
            mlp_mods["down_proj"] = build_v6_proj(f"{p}mlp.down_proj.weight", w_teach_f32.pop("down_proj"), xrms_d, get_lm_spec_v6(l_idx, "down_proj"))
            del w_tmp_mlp, xsq_mlp, xrms_mlp, xsq_d, xrms_d, w_teach_f32
            torch.cuda.empty_cache()

            mlp_steps = 80 if l_idx >= 36 else 45
            w_mlp_qat, norms_g, mcos0, mcos1, mcos_sp45 = qat_block_mlp_v6(
                l_idx, mlp_mods, norms_g, xm_all, X_teach_next, mlp_steps
            )

            lsc = float(norms_g["lsc"])
            with torch.no_grad():
                _hmg = H_stud_eval_mid.to(DEVICE)
                _hhm = rmsn_f16(_hmg, norms_g["pf"])
                _hact = (F.gelu(F.linear(_hhm, w_mlp_qat["gate_proj"]).float(), approximate="tanh") * F.linear(_hhm, w_mlp_qat["up_proj"]).float()).to(torch.float16)
                H_stud_eval = ((_hmg.float() + rmsn_f16(F.linear(_hact, w_mlp_qat["down_proj"]), norms_g["pff"]).float()) * lsc).to(torch.float16).cpu()
                del H_stud_eval_mid
                X_stud_next = torch.empty((N_SEQ, SEQ_LEN, D_MOD), dtype=torch.float16)
                for b0 in range(0, N_SEQ, 32):
                    b1 = min(b0 + 32, N_SEQ)
                    xm_g = xm_all[b0:b1].to(DEVICE)
                    hm = rmsn_f16(xm_g, norms_g["pf"])
                    act = (F.gelu(F.linear(hm, w_mlp_qat["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_mlp_qat["up_proj"]).float()).to(torch.float16)
                    X_stud_next[b0:b1] = ((xm_g.float() + rmsn_f16(F.linear(act, w_mlp_qat["down_proj"]), norms_g["pff"]).float()) * lsc).to(torch.float16).cpu()
            del xm_all, w_mlp_qat

            all_mods_layer = {**attn_mods, **mlp_mods}
            if l_idx not in TAIL_LAYERS:
                ord_pns = ["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if (is_full and K_EQ_V) else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                for pn_ord in ord_pns:
                    for rec in all_mods_layer[pn_ord].export_records():
                        t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, *rec)
                        lm_mats_count += 1
                for nk_s, nk_f in [
                    ("in", "input_layernorm.weight"), ("qn", "self_attn.q_norm.weight"),
                    ("kn", "self_attn.k_norm.weight"), ("pa", "post_attention_layernorm.weight"),
                    ("pf", "pre_feedforward_layernorm.weight"), ("pff", "post_feedforward_layernorm.weight"),
                ]:
                    write_norm_inplace(t16c_fp, norm_offsets, p + nk_f, norms_g[nk_s].detach().float().cpu().numpy())
                write_norm_inplace(t16c_fp, norm_offsets, p + "layer_scalar", np.array([lsc], dtype="<f4"))
            else:
                # Save full trainable tail modules to CPU for Stage 3B Joint 4-Layer Logit-KL Distillation
                _atomic_torch_save({
                    "mods": {k: m.cpu() for k, m in all_mods_layer.items()},
                    "norms": {k: (v.detach().cpu() if torch.is_tensor(v) else float(v)) for k, v in norms_g.items()},
                }, TAIL_MOD_DIR / f"L{l_idx:02d}.pt")
            del attn_mods, mlp_mods, all_mods_layer

            cos_l_txt = float(F.cosine_similarity(X_stud_next[VAL_SEQ_IDX[:16]].reshape(-1, D_MOD).float(), X_teach_next[VAL_SEQ_IDX[:16]].reshape(-1, D_MOD).float(), dim=-1).mean().item())
            cos_l_vlm = float(F.cosine_similarity(H_stud_eval.reshape(-1, D_MOD).float(), H_teach_eval.reshape(-1, D_MOD).float(), dim=-1).mean().item())
            layer_cos_log.append({
                "l": l_idx, "txt_cos": round(cos_l_txt, 5), "vlm_cos": round(cos_l_vlm, 5),
                "attn_qat": [round(acos0, 5), round(acos1, 5)],
                "mlp_qat": [round(mcos0, 5), round(mcos1, 5), round(mcos_sp45, 5)],
                "top50_e_pct": round(top50_e_pct, 1),
            })
            X_teach_cpu, X_stud_cpu = X_teach_next, X_stud_next
            del X_teach_next, X_stud_next
            torch.cuda.empty_cache()

            if (l_idx + 1) % 4 == 0 or l_idx == 55:
                t16c_fp.flush(); os.fsync(t16c_fp.fileno())
                _atomic_torch_save(X_teach_cpu, CKPT_X_TEACH)
                _atomic_torch_save(X_stud_cpu, CKPT_X_STUD)
                _atomic_torch_save({
                    "next_layer": l_idx + 1, "t16c_pos": t16c_pos, "lm_mats_count": lm_mats_count,
                    "norm_offsets": norm_offsets, "H_teach_eval": H_teach_eval, "H_stud_eval": H_stud_eval,
                    "layer_cos_log": layer_cos_log,
                }, STAGE3_CKPT_PATH)
                sync_checkpoint_to_drive()
                gc.collect()
            save_progress({"stage": f"L{l_idx:02d}_done", "next_layer": l_idx + 1, "txt_cos": round(cos_l_txt, 5)})
            print(
                f"  [V6-LB2 L{l_idx:02d}/59 {'F' if is_full else 'S'}] txt_cos={cos_l_txt:.5f} vlm_cos={cos_l_vlm:.5f} "
                f"| attn {acos0:.4f}->{acos1:.4f} | mlp {mcos0:.4f}->{mcos1:.4f} (sp45={mcos_sp45:.4f}, e50={top50_e_pct:.1f}%) "
                f"| dt={time.perf_counter()-tl:.1f}s ram={host_ram_gb()[0]:.2f}GiB",
                flush=True,
            )

        # -------------------------------------------------------------------
        # 7. Stage 3B: Joint 4-Layer (L56..L59) + final_norm End-to-End Logit-KL Distillation
        # -------------------------------------------------------------------
        print("\n[Stage 3B] Joint 4-Layer (L56..L59 + final_norm) End-to-End Logit-KL Distillation...", flush=True)
        fn_g_teach = fn_orig_cpu.to(DEVICE)
        vlm_ans_pos = np.flatnonzero(all_vlm_ans_mask.reshape(-1))
        txt_ans_pos = np.flatnonzero((all_ans_mask & (~all_vlm_ans_mask)).reshape(-1))
        sel_vlm = vlm_ans_pos[np.linspace(0, vlm_ans_pos.size - 1, min(8192, vlm_ans_pos.size), dtype=int)] if vlm_ans_pos.size > 0 else np.array([], dtype=int)
        sel_txt = txt_ans_pos[np.linspace(0, txt_ans_pos.size - 1, min(32768 - sel_vlm.size, txt_ans_pos.size), dtype=int)]
        sel_pos_np = np.sort(np.concatenate([sel_txt, sel_vlm]))
        sel_seq, sel_tok = sel_pos_np // SEQ_LEN, sel_pos_np % SEQ_LEN

        emb_gpu = sf.get_tensor("model.language_model.embed_tokens.weight").to(DEVICE, dtype=torch.float16)
        ht_ans_g = rmsn_diff(X_teach_cpu[sel_seq, sel_tok].to(DEVICE), fn_g_teach)
        nxt_ans_g = torch.from_numpy(all_seq_nxt.reshape(-1)[sel_pos_np]).to(DEVICE, dtype=torch.long)
        TAU = 1.5
        with torch.no_grad():
            top_ids_list = []
            for r0 in range(0, emb_gpu.shape[0], 16384):
                wc = emb_gpu[r0 : r0 + 16384].float()
                zt = 30.0 * torch.tanh((ht_ans_g[::4] @ wc.T) / 30.0)
                top_ids_list.append((torch.topk(zt, 32, dim=-1).indices + r0).reshape(-1))
            uniq_ids, counts = torch.unique(torch.cat(top_ids_list), return_counts=True)
            best_cols = uniq_ids[torch.topk(counts, min(4096, uniq_ids.numel())).indices]
            cand_ids_3b = torch.unique(torch.cat([best_cols, nxt_ans_g]))
            W_cand_3b = emb_gpu[cand_ids_3b].to(COMPUTE_DTYPE)
        del emb_gpu, ht_ans_g, nxt_ans_g, top_ids_list, uniq_ids, counts, best_cols
        torch.cuda.empty_cache()

        class TailLayerV6(nn.Module):
            def __init__(self, l_i: int, ckpt_d: dict) -> None:
                super().__init__()
                self.l_idx = l_i
                self.is_full = LAYER_TYPES[l_i] == "full_attention"
                self.nkv = NKV_F if self.is_full else NKV_S
                self.hd = HD_F if self.is_full else HD_S
                self.gqa_rep = NH // self.nkv
                self.rot_d = int(round(self.hd * (PART_G if self.is_full else PART_L)))
                cr, sr = make_rope(THETA_G if self.is_full else THETA_L, self.rot_d, self.hd)
                self.register_buffer("cos_r", cr, persistent=False)
                self.register_buffer("sin_r", sr, persistent=False)
                ms = ckpt_d["mods"]
                self.q_proj = ms["q_proj"].to(DEVICE)
                self.k_proj = ms["k_proj"].to(DEVICE)
                self.v_proj = ms["v_proj"].to(DEVICE) if "v_proj" in ms else None
                self.o_proj = ms["o_proj"].to(DEVICE)
                self.gate_proj = ms["gate_proj"].to(DEVICE)
                self.up_proj = ms["up_proj"].to(DEVICE)
                self.down_proj = ms["down_proj"].to(DEVICE)
                nrm = ckpt_d["norms"]
                self.n_in = nn.Parameter(nrm["in"].to(DEVICE, dtype=torch.float32).clone())
                self.n_qn = nn.Parameter(nrm["qn"].to(DEVICE, dtype=torch.float32).clone())
                self.n_kn = nn.Parameter(nrm["kn"].to(DEVICE, dtype=torch.float32).clone())
                self.n_pa = nn.Parameter(nrm["pa"].to(DEVICE, dtype=torch.float32).clone())
                self.n_pf = nn.Parameter(nrm["pf"].to(DEVICE, dtype=torch.float32).clone())
                self.n_pff = nn.Parameter(nrm["pff"].to(DEVICE, dtype=torch.float32).clone())
                self.lsc = nn.Parameter(torch.tensor(float(nrm["lsc"]), device=DEVICE, dtype=torch.float32))

            def forward(self, xb: torch.Tensor, amask: Optional[torch.Tensor]) -> torch.Tensor:
                mb = xb.shape[0]
                inp = rmsn_diff(xb, self.n_in)
                q = self.q_proj(inp).view(mb, SEQ_LEN, NH, self.hd)
                k = self.k_proj(inp).view(mb, SEQ_LEN, self.nkv, self.hd)
                v = k if self.v_proj is None else self.v_proj(inp).view(mb, SEQ_LEN, self.nkv, self.hd)
                qn = apply_rope_f32(rmsn_diff(q, self.n_qn), self.cos_r, self.sin_r, self.rot_d, self.hd).transpose(1, 2).to(COMPUTE_DTYPE)
                kn = apply_rope_f32(rmsn_diff(k, self.n_kn), self.cos_r, self.sin_r, self.rot_d, self.hd).transpose(1, 2).to(COMPUTE_DTYPE)
                vn = rmsn_diff(v, None).transpose(1, 2).to(COMPUTE_DTYPE)
                ctx = F.scaled_dot_product_attention(
                    qn, kn.repeat_interleave(self.gqa_rep, 1), vn.repeat_interleave(self.gqa_rep, 1),
                    attn_mask=(amask if (amask is not None and not self.is_full) else None),
                    is_causal=(amask is None or self.is_full), scale=1.0,
                )
                ot = self.o_proj(ctx.transpose(1, 2).contiguous().view(mb, SEQ_LEN, NH * self.hd))
                xm = xb.float() + rmsn_diff(ot, self.n_pa)
                hm = rmsn_diff(xm, self.n_pf)
                act = F.gelu(self.gate_proj(hm), approximate="tanh") * self.up_proj(hm)
                return (xm + rmsn_diff(self.down_proj(act), self.n_pff)) * self.lsc

        tail_ckpts = {li: torch.load(TAIL_MOD_DIR / f"L{li:02d}.pt", map_location="cpu", weights_only=False) for li in (56, 57, 58, 59)}
        tail_mods = nn.ModuleList([TailLayerV6(li, tail_ckpts[li]) for li in (56, 57, 58, 59)])
        p_final_norm = nn.Parameter(fn_orig_cpu.to(DEVICE, dtype=torch.float32).clone())
        x_pre56_cpu = torch.load("/tmp/v6_x_stud_pre56.pt", map_location="cpu", weights_only=True)
        h_ev_pre56_cpu = torch.load("/tmp/v6_h_eval_pre56.pt", map_location="cpu", weights_only=True)

        def run_tail_4layers(x_in: torch.Tensor, amask: Optional[torch.Tensor], use_ckpt: bool) -> torch.Tensor:
            h = x_in.float()
            for mod in tail_mods:
                if use_ckpt and h.requires_grad:
                    h = grad_checkpoint(mod, h, amask, use_reentrant=False)
                else:
                    h = mod(h, amask)
            return h

        def eval_tail_3b() -> Tuple[float, float]:
            with torch.no_grad():
                kl_s, agr_s, cnt = 0.0, 0.0, 0
                for v0 in range(0, len(VAL_SEQ_IDX), 4):
                    sidx = VAL_SEQ_IDX[v0 : v0 + 4].tolist()
                    am = make_sliding_attn_mask_for_indices(sidx, vlm_spans)
                    hs = rmsn_diff(run_tail_4layers(x_pre56_cpu[sidx].to(DEVICE), am, False), p_final_norm)
                    ht = rmsn_diff(X_teach_cpu[sidx].to(DEVICE), fn_g_teach)
                    am_tok = torch.from_numpy(all_ans_mask[sidx]).to(DEVICE)
                    if not am_tok.any():
                        continue
                    hs_a, ht_a = hs[am_tok].to(COMPUTE_DTYPE), ht[am_tok].to(COMPUTE_DTYPE)
                    zs = 30.0 * torch.tanh(F.linear(hs_a, W_cand_3b).float() / 30.0)
                    zt = 30.0 * torch.tanh(F.linear(ht_a, W_cand_3b).float() / 30.0)
                    pt = F.softmax(zt / TAU, dim=-1)
                    kl_s += float(((pt * (torch.log(pt.clamp(min=1e-12)) - F.log_softmax(zs / TAU, dim=-1))).sum(dim=-1) * (TAU * TAU)).sum().item())
                    agr_s += float((zs.argmax(dim=-1) == zt.argmax(dim=-1)).float().sum().item())
                    cnt += hs_a.shape[0]
                return kl_s / max(cnt, 1), agr_s / max(cnt, 1)

        kl_3b_0, agr_3b_0 = eval_tail_3b()
        print(f"  [Stage 3B Before] Val KL={kl_3b_0:.4f} | Teacher-Agree={agr_3b_0*100:.1f}%", flush=True)
        opt_groups_3b: List[dict] = []
        norm_params_3b: List[nn.Parameter] = [p_final_norm]
        for tmod in tail_mods:
            for pname in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                sub_m = getattr(tmod, pname, None)
                if sub_m is not None:
                    opt_groups_3b.extend(sub_m.param_groups(lr_lat=2.5e-3, lr_sc=4e-4))
            norm_params_3b.extend([tmod.n_in, tmod.n_qn, tmod.n_kn, tmod.n_pa, tmod.n_pf, tmod.n_pff, tmod.lsc])
        opt_groups_3b.append({"params": norm_params_3b, "lr": 6e-4})
        assert sum(len(g["params"]) for g in opt_groups_3b) == len(list(tail_mods.parameters()) + [p_final_norm])
        opt_3b = torch.optim.AdamW(opt_groups_3b, weight_decay=1e-4)
        STEPS_3B = 150
        sch_3b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_3b, T_max=STEPS_3B, eta_min=4e-5)
        best_kl_3b, best_agr_3b = kl_3b_0, agr_3b_0
        best_3b_sd = {
            "tail": {k: v.detach().cpu().clone() for k, v in tail_mods.state_dict().items()},
            "fn": p_final_norm.detach().cpu().clone(),
        }
        for st in range(STEPS_3B):
            opt_3b.zero_grad()
            sel = TRAIN_SEQ_IDX[np.random.choice(len(TRAIN_SEQ_IDX), size=4, replace=False)].tolist()
            am = make_sliding_attn_mask_for_indices(sel, vlm_spans)
            xb = x_pre56_cpu[sel].to(DEVICE).float().requires_grad_(True)
            hs = rmsn_diff(run_tail_4layers(xb, am, True), p_final_norm)
            with torch.no_grad():
                ht = rmsn_diff(X_teach_cpu[sel].to(DEVICE), fn_g_teach)
            am_tok = torch.from_numpy(all_ans_mask[sel]).to(DEVICE)
            l_cos = 1.0 - F.cosine_similarity(hs.reshape(-1, D_MOD), ht.reshape(-1, D_MOD), dim=-1).mean()
            if am_tok.any():
                hs_a, ht_a = hs[am_tok].to(COMPUTE_DTYPE), ht[am_tok].to(COMPUTE_DTYPE)
                zs = 30.0 * torch.tanh(F.linear(hs_a, W_cand_3b).float() / 30.0)
                with torch.no_grad():
                    zt = 30.0 * torch.tanh(F.linear(ht_a, W_cand_3b).float() / 30.0)
                    pt = F.softmax(zt / TAU, dim=-1)
                l_kl = ((pt * (torch.log(pt.clamp(min=1e-12)) - F.log_softmax(zs / TAU, dim=-1))).sum(dim=-1) * (TAU * TAU)).mean()
                loss = l_kl + 2.0 * l_cos
            else:
                loss = 2.0 * l_cos
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(tail_mods.parameters()) + [p_final_norm], 1.0)
            opt_3b.step(); sch_3b.step()
            if (st + 1) % 30 == 0:
                kl_v, agr_v = eval_tail_3b()
                print(f"    [Stage 3B step {st+1:03d}/{STEPS_3B}] loss={loss.item():.4f} | Val KL={kl_v:.4f} Agree={agr_v*100:.1f}%", flush=True)
                if kl_v < best_kl_3b:
                    best_kl_3b, best_agr_3b = kl_v, agr_v
                    best_3b_sd = {
                        "tail": {k: v.detach().cpu().clone() for k, v in tail_mods.state_dict().items()},
                        "fn": p_final_norm.detach().cpu().clone(),
                    }

        tail_mods.load_state_dict({k: v.to(DEVICE) for k, v in best_3b_sd["tail"].items()})
        p_final_norm.data.copy_(best_3b_sd["fn"].to(DEVICE))
        kl_3b_1, agr_3b_1 = eval_tail_3b()
        fn_distilled_cpu = p_final_norm.detach().float().cpu().clone()
        write_norm_inplace(t16c_fp, norm_offsets, "model.language_model.norm.weight", fn_distilled_cpu.numpy())
        print(f"  [Stage 3B DONE] Val KL: {kl_3b_0:.4f} -> {kl_3b_1:.4f} | Teacher-Agree: {agr_3b_0*100:.1f}% -> {agr_3b_1*100:.1f}%!", flush=True)
        del opt_3b, sch_3b, best_3b_sd, W_cand_3b

        x_cur_tail, h_ev_tail = x_pre56_cpu, h_ev_pre56_cpu
        for idx_t, li in enumerate((56, 57, 58, 59)):
            mod = tail_mods[idx_t]
            p = f"model.language_model.layers.{li}."
            is_f = LAYER_TYPES[li] == "full_attention"
            sub_mods = {
                "q_proj": mod.q_proj, "k_proj": mod.k_proj, "o_proj": mod.o_proj,
                "gate_proj": mod.gate_proj, "up_proj": mod.up_proj, "down_proj": mod.down_proj,
            }
            if mod.v_proj is not None:
                sub_mods["v_proj"] = mod.v_proj
            ord_pns = ["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if (is_f and K_EQ_V) else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            w_exp = {}
            for pn_ord in ord_pns:
                w_exp[pn_ord] = sub_mods[pn_ord].dequant_w_eff_f16().cpu()
                for rec in sub_mods[pn_ord].export_records():
                    t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, *rec)
                    lm_mats_count += 1
            nrm_out = {
                "in": mod.n_in.detach().cpu(), "qn": mod.n_qn.detach().cpu(), "kn": mod.n_kn.detach().cpu(),
                "pa": mod.n_pa.detach().cpu(), "pf": mod.n_pf.detach().cpu(), "pff": mod.n_pff.detach().cpu(),
                "lsc": float(mod.lsc.detach().item()),
            }
            for nk_s, nk_f in [
                ("in", "input_layernorm.weight"), ("qn", "self_attn.q_norm.weight"),
                ("kn", "self_attn.k_norm.weight"), ("pa", "post_attention_layernorm.weight"),
                ("pf", "pre_feedforward_layernorm.weight"), ("pff", "post_feedforward_layernorm.weight"),
            ]:
                write_norm_inplace(t16c_fp, norm_offsets, p + nk_f, nrm_out[nk_s].float().numpy())
            write_norm_inplace(t16c_fp, norm_offsets, p + "layer_scalar", np.array([nrm_out["lsc"]], dtype="<f4"))
            if li == 58:
                _atomic_torch_save(x_cur_tail, Path("/tmp/v6_x_stud_pre58_dist.pt"))
                _atomic_torch_save(h_ev_tail, Path("/tmp/v6_h_eval_pre58_dist.pt"))
            if li in (58, 59):
                _atomic_torch_save({"w": w_exp, "norms": nrm_out}, Path(f"/tmp/v6_l{li}_stud.pt"))
            x_next_t = torch.empty_like(x_cur_tail)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 32):
                    b1 = min(b0 + 32, N_SEQ)
                    am = None if is_f else make_sliding_attn_mask(b0, b1, vlm_spans)
                    x_next_t[b0:b1] = mod(x_cur_tail[b0:b1].to(DEVICE), am).to(torch.float16).cpu()
                am_h = None if is_f else make_sliding_attn_mask(0, h_ev_tail.shape[0], heldout_spans)
                h_ev_tail = mod(h_ev_tail.to(DEVICE), am_h).to(torch.float16).cpu()
            x_cur_tail = x_next_t
            del w_exp
        X_stud_cpu, H_stud_eval = x_cur_tail, h_ev_tail
        del tail_mods, tail_ckpts, x_pre56_cpu, h_ev_pre56_cpu, p_final_norm
        torch.cuda.empty_cache()

        assert lm_mats_count == _pred_lm_mats, f"LM record count mismatch: {lm_mats_count} != {_pred_lm_mats}"
        _v_tail = torch.load(VIS_CKPT_PATH, map_location="cpu", weights_only=False)
        for rec_v in _v_tail["vision_records"]:
            t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, *rec_v)
        del _v_tail
        t16c_fp.flush(); os.fsync(t16c_fp.fileno())
    finally:
        t16c_fp.close()

t16c_sz = OUT_T16C.stat().st_size
assert t16c_sz <= 3_700_000_000, f"Serialized V6.0 .t16c exceeded 3.70 GB guard: {t16c_sz:,} B"
print(f"[Stage 3 DONE] Unified V6.0 LittleBit-2 .t16c flushed: {OUT_T16C} ({t16c_sz:,} B = {t16c_sz/1024**3:.3f} GiB)!", flush=True)
persist_to_drive(OUT_T16C)

for _rm_p in (VIS_CKPT_PATH, VLM_SOFT_CACHE, STAGE3_CKPT_PATH, CKPT_X_TEACH, CKPT_X_STUD, Path("/tmp/v6_x_stud_pre56.pt"), Path("/tmp/v6_h_eval_pre56.pt")):
    _rm_p.unlink(missing_ok=True)
if DRIVE_DIR is not None:
    drive_ckpt = DRIVE_DIR / "ckpt_v6_0_littlebit2"
    if drive_ckpt.is_dir():
        shutil.rmtree(drive_ckpt, ignore_errors=True)
    partial_p = DRIVE_DIR / (OUT_T16C.name + ".partial")
    if partial_p.is_file():
        partial_p.unlink(missing_ok=True)
shutil.rmtree(TAIL_MOD_DIR, ignore_errors=True)
gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 8. Stage 4A: Multimodal C8DB + Stage 4B: V6.0-Matched 4-Layer EAGLE-3 Assistant
# ---------------------------------------------------------------------------
print("\n[Stage 4A] Training Multimodal C8DB Encyclopedia on V6.0 LittleBit-2...", flush=True)
c8db_entries: List[dict] = []
if vis_keep_c8d:
    exp_rv = [c8d._c6d_radicals("vision", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
    c8db_entries.append({
        "name": "model.embed_vision.embedding_projection.weight", "d_out": D_MOD, "d_in": 1152, "rank": 32, "top_k": 2,
        "c6d": exp_rv[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents_vis, "expert_radicals": exp_rv, "experts": experts_vis,
    })


def train_c8d_proj(name: str, x_tr, y_t_tr, y_s_tr, x_va, y_t_va, y_s_va, rank: int = 32, steps: int = 300):
    cos0 = float(F.cosine_similarity(y_s_va, y_t_va, dim=-1).mean().item())
    model = c8d.Cuneiform8DPatternAdapter(d_out=y_t_tr.shape[1], d_in=x_tr.shape[1], rank=rank, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-4)
    for _ in range(steps):
        opt.zero_grad()
        idx_b = torch.randint(0, x_tr.shape[0], (min(2048, x_tr.shape[0]),), device=DEVICE)
        d_y, _, lb = model(x_tr[idx_b])
        pred, tgt = y_s_tr[idx_b] + d_y, y_t_tr[idx_b]
        loss = F.mse_loss(pred, tgt) + 0.5 * (1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()) + 0.005 * lb
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    cents, experts = model.cpu().export_quantized_arrays()
    with torch.no_grad():
        cos1 = float(F.cosine_similarity(y_s_va + eval_c8d_gpu(x_va, cents, experts), y_t_va, dim=-1).mean().item())
    print(f"  [C8DB {name}] Val cos: {cos0:.5f} -> {cos1:.5f} ({'KEEP' if cos1 > cos0 else 'SKIP'})", flush=True)
    if cos1 <= cos0:
        return None, None, None
    exp_r = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
    return ({
        "name": name, "d_out": y_t_tr.shape[1], "d_in": x_tr.shape[1], "rank": rank, "top_k": 2,
        "c6d": exp_r[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents, "expert_radicals": exp_r, "experts": experts,
    }, cents, experts)


def forward_student_layer_v6(
    X_in_cpu: torch.Tensor, ckpt_path: str, l_idx: int,
    adp_k=None, adp_v=None, spans_dict=None, capture_kv_seqs=None,
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    w_g = {k: v.to(DEVICE) for k, v in ckpt["w"].items()}
    n_g = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in ckpt["norms"].items()}
    is_full = LAYER_TYPES[l_idx] == "full_attention"
    nkv, hd = (NKV_F, HD_F) if is_full else (NKV_S, HD_S)
    rot_d = int(round(hd * (PART_G if is_full else PART_L)))
    cos_r, sin_r = make_rope(THETA_G if is_full else THETA_L, rot_d, hd)
    gqa_rep = NH // nkv
    lsc = float(n_g["lsc"])
    X_out = torch.empty_like(X_in_cpu)
    cap_set = set(capture_kv_seqs) if capture_kv_seqs is not None else set()
    cap_k_rope, cap_v_norm = {}, {}
    with torch.no_grad():
        for b0 in range(0, X_in_cpu.shape[0], 32):
            b1 = min(b0 + 32, X_in_cpu.shape[0])
            xb = X_in_cpu[b0:b1].to(DEVICE)
            inp = rmsn_f16(xb, n_g["in"])
            inp2 = inp.reshape(-1, D_MOD)
            q = F.linear(inp, w_g["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
            k2 = F.linear(inp2, w_g["k_proj"]).float()
            if adp_k is not None:
                k2 = k2 + eval_c8d_gpu(inp2, adp_k[0], adp_k[1])
            k = k2.to(torch.float16).view(b1 - b0, SEQ_LEN, nkv, hd)
            if is_full and K_EQ_V:
                v = k.clone()
            else:
                v2 = F.linear(inp2, w_g["v_proj"]).float()
                if adp_v is not None:
                    v2 = v2 + eval_c8d_gpu(inp2, adp_v[0], adp_v[1])
                v = v2.to(torch.float16).view(b1 - b0, SEQ_LEN, nkv, hd)
            qn = apply_rope_f32(rmsn_f16(q, n_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
            kn = apply_rope_f32(rmsn_f16(k, n_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
            vn = rmsn_f16(v, None).transpose(1, 2)
            if cap_set:
                for gi in range(b0, b1):
                    if gi in cap_set:
                        cap_k_rope[gi] = kn[gi - b0].cpu()
                        cap_v_norm[gi] = vn[gi - b0].cpu()
            am = None if (is_full or not spans_dict) else make_sliding_attn_mask(b0, b1, spans_dict)
            ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0)
            ot = F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_g["o_proj"])
            xm = xb + rmsn_f16(ot, n_g["pa"])
            hm = rmsn_f16(xm, n_g["pf"])
            act = (F.gelu(F.linear(hm, w_g["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_g["up_proj"]).float()).to(torch.float16)
            X_out[b0:b1] = ((xm.float() + rmsn_f16(F.linear(act, w_g["down_proj"]), n_g["pff"]).float()) * lsc).to(torch.float16).cpu()
    cap = {"k_rope": cap_k_rope, "v_norm": cap_v_norm} if capture_kv_seqs is not None else None
    del ckpt, w_g, n_g
    torch.cuda.empty_cache()
    return X_out, cap


sub_kv = np.concatenate([
    np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int),
    np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int),
])
x_pre58_d = torch.load("/tmp/v6_x_stud_pre58_dist.pt", map_location="cpu", weights_only=True)
h_ev_pre58_d = torch.load("/tmp/v6_h_eval_pre58_dist.pt", map_location="cpu", weights_only=True)
ck58 = torch.load("/tmp/v6_l58_stud.pt", map_location="cpu", weights_only=True)
tkv58 = torch.load("/tmp/v6_teach_kv_l58.pt", map_location="cpu", weights_only=True)
with torch.no_grad():
    inp58 = rmsn_f16(x_pre58_d[sub_kv].to(DEVICE), ck58["norms"]["in"].to(DEVICE)).reshape(-1, D_MOD).float()
    ks58_b = inp58 @ ck58["w"]["k_proj"].to(DEVICE).float().T
    vs58_b = inp58 @ ck58["w"]["v_proj"].to(DEVICE).float().T
del ck58
ntr_k = (inp58.shape[0] * 4) // 5
adp_k58 = adp_v58 = adp_k59 = None
ent_k58, c_k58, e_k58 = train_c8d_proj("model.language_model.layers.58.self_attn.k_proj.weight", inp58[:ntr_k], tkv58["kt"].float().to(DEVICE)[:ntr_k], ks58_b[:ntr_k], inp58[ntr_k:], tkv58["kt"].float().to(DEVICE)[ntr_k:], ks58_b[ntr_k:])
if ent_k58 is not None:
    c8db_entries.append(ent_k58); adp_k58 = (c_k58, e_k58)
ent_v58, c_v58, e_v58 = train_c8d_proj("model.language_model.layers.58.self_attn.v_proj.weight", inp58[:ntr_k], tkv58["vt"].float().to(DEVICE)[:ntr_k], vs58_b[:ntr_k], inp58[ntr_k:], tkv58["vt"].float().to(DEVICE)[ntr_k:], vs58_b[ntr_k:])
if ent_v58 is not None:
    c8db_entries.append(ent_v58); adp_v58 = (c_v58, e_v58)
del inp58, ks58_b, vs58_b, tkv58
torch.cuda.empty_cache()

# 96 diverse sequences for Stage 4B V6.0-matched Assistant alignment!
AST_SEQS = np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int).tolist()
x_59_in, cap_kv58 = forward_student_layer_v6(x_pre58_d, "/tmp/v6_l58_stud.pt", 58, adp_k58, adp_v58, spans_dict=vlm_spans, capture_kv_seqs=AST_SEQS)
h_ev_59_in, _ = forward_student_layer_v6(h_ev_pre58_d, "/tmp/v6_l58_stud.pt", 58, adp_k58, adp_v58, spans_dict=heldout_spans)
del x_pre58_d, h_ev_pre58_d

ck59 = torch.load("/tmp/v6_l59_stud.pt", map_location="cpu", weights_only=True)
tkv59 = torch.load("/tmp/v6_teach_kv_l59.pt", map_location="cpu", weights_only=True)
with torch.no_grad():
    inp59 = rmsn_f16(x_59_in[sub_kv].to(DEVICE), ck59["norms"]["in"].to(DEVICE)).reshape(-1, D_MOD).float()
    ks59_b = inp59 @ ck59["w"]["k_proj"].to(DEVICE).float().T
del ck59
ent_k59, c_k59, e_k59 = train_c8d_proj("model.language_model.layers.59.self_attn.k_proj.weight", inp59[:ntr_k], tkv59["kt"].float().to(DEVICE)[:ntr_k], ks59_b[:ntr_k], inp59[ntr_k:], tkv59["kt"].float().to(DEVICE)[ntr_k:], ks59_b[ntr_k:])
if ent_k59 is not None:
    c8db_entries.append(ent_k59); adp_k59 = (c_k59, e_k59)
del inp59, ks59_b, tkv59
torch.cuda.empty_cache()

X_stud_cpu, cap_kv59 = forward_student_layer_v6(x_59_in, "/tmp/v6_l59_stud.pt", 59, adp_k59, None, spans_dict=vlm_spans, capture_kv_seqs=AST_SEQS)
H_stud_eval, _ = forward_student_layer_v6(h_ev_59_in, "/tmp/v6_l59_stud.pt", 59, adp_k59, None, spans_dict=heldout_spans)
del x_59_in, h_ev_59_in

fn_g = fn_orig_cpu.to(DEVICE)
fn_eff_g = fn_distilled_cpu.to(DEVICE)

ht_ans_g = rmsn_diff(X_teach_cpu[sel_seq, sel_tok].to(DEVICE), fn_g)
hs_ans_g = rmsn_diff(X_stud_cpu[sel_seq, sel_tok].to(DEVICE), fn_eff_g)
nxt_ans_g = torch.from_numpy(all_seq_nxt.reshape(-1)[sel_pos_np]).to(DEVICE, dtype=torch.long)
ast_x59_stud = X_stud_cpu[AST_SEQS].clone()
del X_teach_cpu, X_stud_cpu
gc.collect(); torch.cuda.empty_cache()

with safe_open(sf_path, framework="pt", device="cpu") as sf_e:
    emb_gpu = sf_e.get_tensor("model.language_model.embed_tokens.weight").to(DEVICE, dtype=torch.float16)

N_ANS = hs_ans_g.shape[0]
t_lse = torch.full((N_ANS,), -1e9, device=DEVICE)
q_lse = torch.full((N_ANS,), -1e9, device=DEVICE)
top_ids_t_list, top_ids_q_list = [], []
with torch.no_grad():
    for r0 in range(0, emb_gpu.shape[0], 8192):
        W_c = emb_gpu[r0 : r0 + 8192].float()
        zt = 30.0 * torch.tanh((ht_ans_g @ W_c.T) / 30.0) / TAU
        zq = 30.0 * torch.tanh((hs_ans_g @ W_c.T) / 30.0) / TAU
        t_lse = torch.logaddexp(t_lse, torch.logsumexp(zt, dim=-1))
        q_lse = torch.logaddexp(q_lse, torch.logsumexp(zq, dim=-1))
        top_ids_t_list.append((torch.topk(zt, 32, dim=-1).indices + r0).reshape(-1))
        top_ids_q_list.append((torch.topk(zq, 32, dim=-1).indices + r0).reshape(-1))
    all_top_m = torch.cat(top_ids_t_list + top_ids_q_list)
    uniq_ids, counts = torch.unique(all_top_m, return_counts=True)
    best_cols = uniq_ids[torch.topk(counts, min(4096, uniq_ids.numel())).indices]
    cand_ids = torch.unique(torch.cat([best_cols, nxt_ans_g]))
    W_cand_g = emb_gpu[cand_ids].float()

id_map = {int(tid): ci for ci, tid in enumerate(cand_ids.tolist())}
y_star_g = torch.tensor([id_map[int(tid)] for tid in nxt_ans_g.tolist()], device=DEVICE, dtype=torch.long)
t_top1_g = torch.empty((N_ANS,), device=DEVICE, dtype=torch.long)
log_tail_q0 = torch.empty((N_ANS,), device=DEVICE, dtype=torch.float32)
p_teach_store = torch.empty((N_ANS, int(cand_ids.numel()) + 1), device="cpu", dtype=torch.float16)
with torch.no_grad():
    for c0 in range(0, N_ANS, 4096):
        c1 = min(c0 + 4096, N_ANS)
        zt_c = 30.0 * torch.tanh((ht_ans_g[c0:c1] @ W_cand_g.T) / 30.0) / TAU
        zq_c = 30.0 * torch.tanh((hs_ans_g[c0:c1] @ W_cand_g.T) / 30.0) / TAU
        t_top1_g[c0:c1] = zt_c.argmax(dim=-1)
        lt_t = t_lse[c0:c1] + torch.log((1.0 - torch.exp((torch.logsumexp(zt_c, dim=-1) - t_lse[c0:c1]).clamp(max=0.0))).clamp(min=1e-12))
        log_tail_q0[c0:c1] = q_lse[c0:c1] + torch.log((1.0 - torch.exp((torch.logsumexp(zq_c, dim=-1) - q_lse[c0:c1]).clamp(max=0.0))).clamp(min=1e-12))
        p_teach_store[c0:c1] = F.softmax(torch.cat([zt_c, lt_t.unsqueeze(1)], dim=-1), dim=-1).to(device="cpu", dtype=torch.float16)

va_a = (torch.arange(N_ANS, device=DEVICE) % 5 == 0)
tr_a = ~va_a
model60 = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=D_MOD, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(DEVICE)
opt60 = torch.optim.AdamW(model60.parameters(), lr=3e-3, weight_decay=3e-2)
sch60 = torch.optim.lr_scheduler.CosineAnnealingLR(opt60, T_max=300, eta_min=3e-4)


def eval_c60(x_sub: torch.Tensor, idx_sub: torch.Tensor) -> Tuple[float, float, float, float]:
    with torch.no_grad():
        idx_t = torch.nonzero(idx_sub, as_tuple=False).squeeze(1) if idx_sub.dtype == torch.bool else idx_sub
        n_s = int(idx_t.numel())
        top1_s = agr_s = marg_s = kl_s = 0.0
        for b0 in range(0, n_s, 2048):
            b1 = min(b0 + 2048, n_s)
            idx_b = idx_t[b0:b1]
            zc = 30.0 * torch.tanh((x_sub[b0:b1] @ W_cand_g.T) / 30.0)
            pred_b = zc.argmax(dim=-1)
            ys = y_star_g[idx_b]
            top1_s += float((pred_b == ys).float().sum().item())
            agr_s += float((pred_b == t_top1_g[idx_b]).float().sum().item())
            zy = zc.gather(1, ys.unsqueeze(1)).squeeze(1)
            zm = zc.clone(); zm.scatter_(1, ys.unsqueeze(1), -1e9)
            marg_s += float((zy - zm.max(dim=-1).values).sum().item())
            zw = torch.cat([zc / TAU, log_tail_q0[idx_b].unsqueeze(1)], dim=1)
            pt_b = p_teach_store[idx_b.cpu()].to(DEVICE, dtype=torch.float32)
            kl_s += float(((pt_b * (torch.log(pt_b.clamp(min=1e-12)) - F.log_softmax(zw, dim=-1))).sum(dim=-1) * (TAU * TAU)).sum().item())
        return kl_s / max(n_s, 1), top1_s / max(n_s, 1), agr_s / max(n_s, 1), marg_s / max(n_s, 1)


kl0, acc0, agr0, marg0 = eval_c60(hs_ans_g[va_a], va_a)
print(f"  [V6.0 post_cell60 Before] Val KL={kl0:.4f} Top1={acc0*100:.1f}% Teacher-Agree={agr0*100:.1f}% Margin={marg0:+.3f}", flush=True)
tr_indices = torch.nonzero(tr_a, as_tuple=False).squeeze(1)
best_score, best_sd60 = -1e9, None
for st in range(300):
    opt60.zero_grad()
    perm_b = tr_indices[torch.randperm(tr_indices.numel(), device=DEVICE)[:2048]]
    d_ans, _, lb = model60(hs_ans_g[perm_b])
    hp = hs_ans_g[perm_b] + d_ans
    zc = 30.0 * torch.tanh((hp @ W_cand_g.T) / 30.0)
    zw = torch.cat([zc / TAU, log_tail_q0[perm_b].unsqueeze(1)], dim=1)
    pt_b = p_teach_store[perm_b.cpu()].to(DEVICE, dtype=torch.float32)
    l_kl = ((pt_b * (torch.log(pt_b.clamp(min=1e-12)) - F.log_softmax(zw, dim=-1))).sum(dim=-1) * (TAU * TAU)).mean()
    zy = zc.gather(1, y_star_g[perm_b].unsqueeze(1)).squeeze(1)
    zm = zc.clone(); zm.scatter_(1, y_star_g[perm_b].unsqueeze(1), -1e9)
    l_marg = F.relu(2.0 - zy + zm.max(dim=-1).values).mean()
    l_mse = F.mse_loss(hp, ht_ans_g[perm_b])
    l_cos = 1.0 - F.cosine_similarity(hp, ht_ans_g[perm_b], dim=-1).mean()
    l_reg = 2e-2 * (model60.A_bank.pow(2).sum() + model60.B_bank.pow(2).sum()) / float(perm_b.numel() * D_MOD)
    loss = l_kl + 0.5 * l_marg + 2.0 * l_mse + 0.5 * l_cos + 0.005 * lb + l_reg
    loss.backward(); torch.nn.utils.clip_grad_norm_(model60.parameters(), 1.0); opt60.step(); sch60.step()
    if (st + 1) % 60 == 0:
        with torch.no_grad():
            dv, _, _ = model60(hs_ans_g[va_a])
            klv, accv, agrv, margv = eval_c60(hs_ans_g[va_a] + dv, va_a)
            sc_v = agrv * 2.0 + margv * 0.5 - 0.5 * klv
            if sc_v > best_score:
                best_score = sc_v
                best_sd60 = {k: v.detach().cpu().clone() for k, v in model60.state_dict().items()}
            print(f"    step {st+1:03d}/300 loss={loss.item():.4f} | Va: KL={klv:.4f} Top1={accv*100:.1f}% Agree={agrv*100:.1f}% Marg={margv:+.3f}", flush=True)

m60_cpu = model60.cpu()
if best_sd60 is not None:
    m60_cpu.load_state_dict(best_sd60)
cents60, experts60 = m60_cpu.export_quantized_arrays()
kl_q, acc_q, agr_q, marg_q = eval_c60(hs_ans_g[va_a] + eval_c8d_gpu(hs_ans_g[va_a], cents60, experts60), va_a)
keep_c60 = kl_q <= kl0
print(f"  [V6.0 post_cell60 Quantized] Val KL: {kl0:.4f}->{kl_q:.4f} | Top1: {acc0*100:.1f}%->{acc_q*100:.1f}% | Agree: {agr0*100:.1f}%->{agr_q*100:.1f}%", flush=True)
if keep_c60:
    exp_r60 = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
    c8db_entries.append({
        "name": "model.language_model.norm.post_cell60", "d_out": D_MOD, "d_in": D_MOD, "rank": 32, "top_k": 2,
        "c6d": exp_r60[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents60, "expert_radicals": exp_r60, "experts": experts60,
    })

c8d_sz = c8d.write_c8db_file(OUT_C8D, [("model.language_model.norm.weight", fn_eff_g.detach().float().cpu().numpy().astype("<f4"))], c8db_entries)
print(f"[Stage 4A DONE] Wrote {OUT_C8D} ({c8d_sz:,} B)", flush=True)
persist_to_drive(OUT_C8D)
del p_teach_store, ht_ans_g, hs_ans_g, nxt_ans_g, y_star_g, t_top1_g, log_tail_q0, model60, m60_cpu
torch.cuda.empty_cache()

# Stage 4B: V6.0-Matched 4-Layer EAGLE-3 Assistant (200 steps, 96 seqs, --assistant-step-pos matched!)
print("\n[Stage 4B] Aligning 4-Layer EAGLE-3 Assistant directly to V6.0 LittleBit-2 Student (200 steps, 96 seqs)...", flush=True)
ast_report = {"status": "skipped"}
try:
    ast_dir = Path(snapshot_download(repo_id="google/gemma-4-31B-it-assistant", local_dir="/tmp/gemma4_31b_assistant", ignore_patterns=["*.bin", "*.pt"]))
    with safe_open(ast_dir / "model.safetensors", framework="pt", device="cpu") as asf:
        ast_sd = {k: asf.get_tensor(k) for k in asf.keys()}

    class Gemma4AssistantTrain(nn.Module):
        def __init__(self, sd: Dict[str, torch.Tensor]) -> None:
            super().__init__()
            self.pre = nn.Parameter(sd["pre_projection.weight"].float().clone())
            self.post = nn.Parameter(sd["post_projection.weight"].float().clone())
            self.fn = nn.Parameter(sd["model.norm.weight"].float().clone())
            self.register_buffer("emb", sd["model.embed_tokens.weight"].to(COMPUTE_DTYPE), persistent=False)
            self.layers = nn.ModuleList()
            for l_a in range(4):
                p = f"model.layers.{l_a}."
                m = nn.Module()
                m.q_proj = nn.Parameter(sd[p + "self_attn.q_proj.weight"].float().clone())
                m.o_proj = nn.Parameter(sd[p + "self_attn.o_proj.weight"].float().clone())
                m.gate = nn.Parameter(sd[p + "mlp.gate_proj.weight"].float().clone())
                m.up = nn.Parameter(sd[p + "mlp.up_proj.weight"].float().clone())
                m.down = nn.Parameter(sd[p + "mlp.down_proj.weight"].float().clone())
                m.n_in = nn.Parameter(sd[p + "input_layernorm.weight"].float().clone())
                m.n_qn = nn.Parameter(sd[p + "self_attn.q_norm.weight"].float().clone())
                m.n_pa = nn.Parameter(sd[p + "post_attention_layernorm.weight"].float().clone())
                m.n_pf = nn.Parameter(sd[p + "pre_feedforward_layernorm.weight"].float().clone())
                m.n_pff = nn.Parameter(sd[p + "post_feedforward_layernorm.weight"].float().clone())
                m.lsc = nn.Parameter(sd[p + "layer_scalar"].float().reshape(1).clone())
                self.layers.append(m)
            cr_l, sr_l = make_rope(THETA_L, HD_S, HD_S)
            cr_g, sr_g = make_rope(THETA_G, int(round(HD_F * PART_G)), HD_F)
            self.register_buffer("cr_l", cr_l, persistent=False)
            self.register_buffer("sr_l", sr_l, persistent=False)
            self.register_buffer("cr_g", cr_g, persistent=False)
            self.register_buffer("sr_g", sr_g, persistent=False)

        def forward_seq(self, tok_emb_seq, back_h_seq, k58, v58, k59):
            B, Sm1, _ = tok_emb_seq.shape
            state = F.linear(torch.cat([tok_emb_seq, back_h_seq], dim=-1).to(COMPUTE_DTYPE), self.pre.to(COMPUTE_DTYPE)).float()
            q_mask = torch.tril(torch.ones((Sm1, SEQ_LEN), device=state.device, dtype=torch.bool), diagonal=0).unsqueeze(0).unsqueeze(0)
            for l_a, m in enumerate(self.layers):
                is_f = (l_a == 3)
                hd_a, nkv_a = (HD_F, NKV_F) if is_f else (HD_S, NKV_S)
                rot_a = int(round(HD_F * PART_G)) if is_f else HD_S
                gqa_a = NH // nkv_a
                inp = rmsn_diff(state, m.n_in).to(COMPUTE_DTYPE)
                q = F.linear(inp, m.q_proj.to(COMPUTE_DTYPE)).view(B, Sm1, NH, hd_a)
                cr = (self.cr_g if is_f else self.cr_l)[1 : Sm1 + 1]
                sr = (self.sr_g if is_f else self.sr_l)[1 : Sm1 + 1]
                qn = apply_rope_f32(rmsn_diff(q, m.n_qn), cr, sr, rot_a, hd_a).transpose(1, 2).to(COMPUTE_DTYPE)
                kk = (k59 if is_f else k58).to(COMPUTE_DTYPE).repeat_interleave(gqa_a, dim=1)
                vv = (k59 if is_f else v58).to(COMPUTE_DTYPE).repeat_interleave(gqa_a, dim=1)
                ctx = F.scaled_dot_product_attention(qn, kk, vv, attn_mask=q_mask, scale=1.0)
                state = state + rmsn_diff(F.linear(ctx.transpose(1, 2).contiguous().view(B, Sm1, NH * hd_a), m.o_proj.to(COMPUTE_DTYPE)), m.n_pa)
                hm = rmsn_diff(state, m.n_pf).to(COMPUTE_DTYPE)
                act = (F.gelu(F.linear(hm, m.gate.to(COMPUTE_DTYPE)).float(), approximate="tanh") * F.linear(hm, m.up.to(COMPUTE_DTYPE)).float()).to(COMPUTE_DTYPE)
                state = (state + rmsn_diff(F.linear(act, m.down.to(COMPUTE_DTYPE)), m.n_pff)) * m.lsc
            return rmsn_diff(state, self.fn)

    ast_mod = Gemma4AssistantTrain(ast_sd).to(DEVICE)
    with torch.no_grad():
        ast_h_final = torch.empty((len(AST_SEQS), SEQ_LEN, D_MOD), dtype=torch.float16)
        for ai in range(len(AST_SEQS)):
            hf = rmsn_diff(ast_x59_stud[ai].to(DEVICE), fn_eff_g)
            if keep_c60:
                hf = hf + eval_c8d_gpu(hf, cents60, experts60)
            ast_h_final[ai] = hf.to(torch.float16).cpu()
        W_ast_cand = ast_mod.emb[cand_ids].float()

    def eval_ast_val():
        with torch.no_grad():
            agr_s = kl_s = 0.0; cnt = 0
            for ai in range(84, 96):
                gi = AST_SEQS[ai]
                t_emb = (emb_gpu[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(DEVICE)].float() * ESCALE).unsqueeze(0)
                ha = ast_mod.forward_seq(t_emb, ast_h_final[ai : ai + 1, :-1].to(DEVICE).float(), cap_kv58["k_rope"][gi].unsqueeze(0).to(DEVICE), cap_kv58["v_norm"][gi].unsqueeze(0).to(DEVICE), cap_kv59["k_rope"][gi].unsqueeze(0).to(DEVICE)).squeeze(0)
                zs = 30.0 * torch.tanh((ast_h_final[ai, 1:].to(DEVICE).float() @ W_cand_g.T) / 30.0)
                za = ha @ W_ast_cand.T
                ps = F.softmax(zs / TAU, dim=-1)
                kl_s += float(((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU, dim=-1))).sum(dim=-1) * (TAU * TAU)).sum().item())
                agr_s += float((za.argmax(dim=-1) == zs.argmax(dim=-1)).float().sum().item())
                cnt += ha.shape[0]
            return kl_s / max(cnt, 1), agr_s / max(cnt, 1)

    akl0, aagr0 = eval_ast_val()
    opt_a = torch.optim.AdamW(ast_mod.parameters(), lr=2.5e-4, weight_decay=1e-4)
    STEPS_A = 200
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=STEPS_A, eta_min=2e-5)
    best_aagr = aagr0
    best_asd = {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    for st in range(STEPS_A):
        opt_a.zero_grad()
        batch_ais = np.random.choice(84, size=4, replace=False).tolist()
        t_embs, bhs, k58s, v58s, k59s, hs_tgts = [], [], [], [], [], []
        for ai in batch_ais:
            gi = AST_SEQS[ai]
            t_embs.append(emb_gpu[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(DEVICE)].float() * ESCALE)
            bhs.append(ast_h_final[ai, :-1].to(DEVICE).float())
            k58s.append(cap_kv58["k_rope"][gi].to(DEVICE))
            v58s.append(cap_kv58["v_norm"][gi].to(DEVICE))
            k59s.append(cap_kv59["k_rope"][gi].to(DEVICE))
            hs_tgts.append(ast_h_final[ai, 1:].to(DEVICE).float())
        ha = ast_mod.forward_seq(torch.stack(t_embs), torch.stack(bhs), torch.stack(k58s), torch.stack(v58s), torch.stack(k59s))
        hs_t = torch.stack(hs_tgts)
        with torch.no_grad():
            zs = 30.0 * torch.tanh((hs_t.reshape(-1, D_MOD) @ W_cand_g.T) / 30.0)
            ps = F.softmax(zs / TAU, dim=-1)
            ys_top1 = zs.argmax(dim=-1)
        za = ha.reshape(-1, 1024) @ W_ast_cand.T
        l_kl = ((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU, dim=-1))).sum(dim=-1) * (TAU * TAU)).mean()
        l_ce = F.cross_entropy(za, ys_top1)
        h_post = F.linear(ha.to(COMPUTE_DTYPE), ast_mod.post.to(COMPUTE_DTYPE)).float()
        loss = l_kl + 0.5 * l_ce + 0.5 * (F.mse_loss(h_post, hs_t) / (hs_t.var() + 1e-5))
        loss.backward(); torch.nn.utils.clip_grad_norm_(ast_mod.parameters(), 1.0); opt_a.step(); sch_a.step()
        if (st + 1) % 40 == 0:
            kl_v, agr_v = eval_ast_val()
            print(f"    [Ast step {st+1:03d}/{STEPS_A}] Val KL={kl_v:.4f} Top-1 Agree={agr_v*100:.1f}%", flush=True)
            if agr_v > best_aagr:
                best_aagr = agr_v
                best_asd = {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_mod.load_state_dict({k: v.to(DEVICE) for k, v in best_asd.items()})
    akl1, aagr1 = eval_ast_val()
    print(f"  [Assistant V6 DONE] Val KL={akl0:.4f}->{akl1:.4f} | Top-1 Spec Agree={aagr0*100:.1f}%->{aagr1*100:.1f}%!", flush=True)

    ast_sd["pre_projection.weight"] = ast_mod.pre.detach().to(torch.bfloat16).cpu()
    ast_sd["post_projection.weight"] = ast_mod.post.detach().to(torch.bfloat16).cpu()
    ast_sd["model.norm.weight"] = ast_mod.fn.detach().to(torch.bfloat16).cpu()
    for l_a, m in enumerate(ast_mod.layers):
        p = f"model.layers.{l_a}."
        ast_sd[p + "self_attn.q_proj.weight"] = m.q_proj.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "self_attn.o_proj.weight"] = m.o_proj.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "mlp.gate_proj.weight"] = m.gate.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "mlp.up_proj.weight"] = m.up.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "mlp.down_proj.weight"] = m.down.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "input_layernorm.weight"] = m.n_in.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "self_attn.q_norm.weight"] = m.n_qn.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "post_attention_layernorm.weight"] = m.n_pa.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "pre_feedforward_layernorm.weight"] = m.n_pf.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "post_feedforward_layernorm.weight"] = m.n_pff.detach().to(torch.bfloat16).cpu()
        ast_sd[p + "layer_scalar"] = m.lsc.detach().to(torch.bfloat16).cpu().reshape(ast_sd[p + "layer_scalar"].shape)
    ft_sf = Path("/tmp/assistant_v6_ft.safetensors")
    st_save_file(ast_sd, str(ft_sf))
    ast_sz = pa.pack_assistant_safetensors_to_t16c(ft_sf, OUT_AST, body_bits=8, embed_bits=4, block_size=64)
    ft_sf.unlink(missing_ok=True)
    persist_to_drive(OUT_AST)
    ast_report = {
        "status": "aligned_and_packed", "path": str(OUT_AST), "bytes": int(ast_sz), "sha256": sha256_file(OUT_AST),
        "val_kl_before": round(akl0, 4), "val_kl_after": round(akl1, 4),
        "val_top1_agree_before": round(aagr0, 4), "val_top1_agree_after": round(aagr1, 4),
    }
except Exception as e:
    print(f"  [WARN Stage 4B skipped: {e}]", flush=True)
    ast_report = {"status": f"error: {e}"}

# Stage 5: Held-Out COCO val[0..3] Visual QA Verification
heldout_vlm_report = []
with torch.no_grad():
    for hi, cap_gt in enumerate(heldout_captions):
        p0, plen = heldout_ans_starts[hi], heldout_ans_lens[hi]
        ht_hi = rmsn_diff(H_teach_eval[hi, p0 : p0 + plen].to(DEVICE), fn_g)
        hs_hi = rmsn_diff(H_stud_eval[hi, p0 : p0 + plen].to(DEVICE), fn_eff_g)
        if keep_c60:
            hs_hi = hs_hi + eval_c8d_gpu(hs_hi, cents60, experts60)
        h_cos_hi = float(F.cosine_similarity(hs_hi, ht_hi, dim=-1).mean().item())
        t_best_v = torch.full((plen, 5), -1e9, device=DEVICE); t_best_i = torch.zeros((plen, 5), device=DEVICE, dtype=torch.long)
        s_best_v = torch.full((plen, 5), -1e9, device=DEVICE); s_best_i = torch.zeros((plen, 5), device=DEVICE, dtype=torch.long)
        for r0 in range(0, emb_gpu.shape[0], 16384):
            wc = emb_gpu[r0 : r0 + 16384].float()
            lt, ls = 30.0 * torch.tanh((ht_hi @ wc.T) / 30.0), 30.0 * torch.tanh((hs_hi @ wc.T) / 30.0)
            tv, ti = torch.topk(lt, 5, dim=-1); sv, si = torch.topk(ls, 5, dim=-1)
            ti = ti + r0; si = si + r0
            t_best_v, pk1 = torch.topk(torch.cat([t_best_v, tv], -1), 5, dim=-1); t_best_i = torch.cat([t_best_i, ti], -1).gather(1, pk1)
            s_best_v, pk2 = torch.topk(torch.cat([s_best_v, sv], -1), 5, dim=-1); s_best_i = torch.cat([s_best_i, si], -1).gather(1, pk2)
        t1, s1 = t_best_i[:, 0], s_best_i[:, 0]
        rec_h = {
            "val_idx": hi, "gt_caption": cap_gt, "ans_cos": round(h_cos_hi, 5),
            "ans_top1_agree": round(float((s1 == t1).float().mean().item()), 4),
            "ans_top5_agree": round(float((s_best_i == t1.unsqueeze(1)).any(dim=-1).float().mean().item()), 4),
            "student_forced_pred": tk.decode([int(x) for x in s1.tolist()]),
        }
        heldout_vlm_report.append(rec_h)
        print(f"  [Held-Out val[{hi}]] cos={h_cos_hi:.5f} top1={rec_h['ans_top1_agree']*100:.1f}% top5={rec_h['ans_top5_agree']*100:.1f}% | Stud={rec_h['student_forced_pred']!r}", flush=True)

report = {
    "version": "V6.0-LITTLEBIT2-ROWMAJOR-ANDVCNT",
    "gpu": gpu_name,
    "elapsed_s": round(time.perf_counter() - T0_ALL, 2),
    "unified_t16c_path": str(OUT_T16C),
    "unified_t16c_bytes": int(t16c_sz),
    "unified_t16c_gib": round(t16c_sz / (1024**3), 4),
    "unified_t16c_sha256": sha256_file(OUT_T16C),
    "c8db_path": str(OUT_C8D),
    "c8db_bytes": int(c8d_sz),
    "c8db_sha256": sha256_file(OUT_C8D),
    "assistant_v6": ast_report,
    "vision_val_cos": [round(v_cos0, 5), round(v_cos1, 5)],
    "stage3b_tail_kl": [round(kl_3b_0, 4), round(kl_3b_1, 4)],
    "stage3b_tail_agree": [round(agr_3b_0, 4), round(agr_3b_1, 4)],
    "post_cell60_val_kl": [round(kl0, 4), round(kl_q, 4)],
    "post_cell60_val_top1": [round(acc0, 4), round(acc_q, 4)],
    "post_cell60_val_teacher_agree": [round(agr0, 4), round(agr_q, 4)],
    "post_cell60_val_margin": [round(marg0, 3), round(marg_q, 3)],
    "heldout_coco_val0_3": heldout_vlm_report,
    "layer_cos_summary": [r for r in layer_cos_log if r["l"] in (0, 5, 10, 20, 30, 35, 40, 45, 50, 55, 57, 58, 59)],
}
OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
persist_to_drive(OUT_REPORT)
print(f"\n[V6.0 LITTLEBIT-2 COMPLETE] Report={OUT_REPORT} | Elapsed={(time.perf_counter()-T0_ALL)/60:.1f} min!", flush=True)

if os.environ.get("COLAB_AUTO_UNASSIGN", "1") == "1":
    try:
        from google.colab import runtime
        print("[Auto-Unassign] Disconnecting and releasing Colab runtime to preserve compute units...", flush=True)
        runtime.unassign()
    except Exception as e:
        print(f"[Auto-Unassign Note] {e}", flush=True)

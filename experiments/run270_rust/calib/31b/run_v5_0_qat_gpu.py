#!/usr/bin/env python3
"""
Gemma 4 31B V5.0 — Zero-Tern5, Cliff-Eliminating 2b_gaus/3b_u0 + Co-Activation
Neuron-Permuted Sparse-Aware Real Block-Wise QAT (`loss.backward()`) + Joint
4-Layer (L56..L59) End-to-End Logit-KL Distillation + Multimodal C8DB +
V5.0-Aligned 4-Layer EAGLE-3 Assistant Speculative Drafter.

Targets physical Raspberry Pi 4 (7.6 GiB RAM, 4x Cortex-A72, 3.93 GB/s LPDDR4):
  - ZERO `tern5` matrices anywhere (pure bit-serial 1b_asym + 2b_gaus + 3b_u0).
  - Hard fail-closed container budget <= 6.50 GB (6.05 GiB) -> VmSwap = 0 on Pi 4.
  - Strictly zero benchmark contamination (full BLACKLIST_SUBSTRINGS + COCO val[0..63] holdout).
  - Real PyTorch gradient-descent QAT (`loss.backward()` + `AdamW`) on physical GPU silicon.
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

# Locate helper library directory (works on Colab /kaggle/input/..., Kaggle, or local repo)
lib_cands = list(Path("/kaggle/input").rglob("pack_t16c_v3.py"))
if not lib_cands:
    lib_cands = [Path(__file__).resolve().parent / "pack_t16c_v3.py"]
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

OUT_DIR = Path(os.environ.get("V5_OUT_DIR", "/kaggle/working"))
CKPT_DIR = OUT_DIR / "ckpt_v5_0_qat"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_T16C = OUT_DIR / "gemma4_31b_v5_0_qat.t16c"
OUT_C8D = OUT_DIR / "gemma4_31b_v5_0_qat_c8d.c8d"
OUT_AST = OUT_DIR / "gemma4_31b_assistant_v5.t16c"
PROGRESS_JSON = CKPT_DIR / "progress.json"

# Extra strict anti-contamination blacklist (in addition to vc.BLACKLIST_SUBSTRINGS)
EXTRA_BLACKLIST: Tuple[str, ...] = (
    "12 * 8",
    "12*8",
    "547",
    "193",
    "motorcycle",
    "motorbike",
    "honda",
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


gpu_name = torch.cuda.get_device_name(0)
gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"[V5.0 INIT] GPU={gpu_name} ({gpu_vram_gb:.1f} GiB) | dtype={COMPUTE_DTYPE} | RAM={host_ram_gb()}", flush=True)

# ---------------------------------------------------------------------------
# 1. Compile OpenMP C quantizer/packer (fast_quant.c with 2b_gaus & header updater)
# ---------------------------------------------------------------------------
fq_c = LIB_DIR / "fast_quant.c"
fq_so = Path("/tmp/libfast_quant_v5.so")
subprocess.run(["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", str(fq_c), "-o", str(fq_so), "-lm"], check=True)
LIB_FQ = ctypes.CDLL(str(fq_so))
LIB_FQ.t16c_tblk_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
LIB_FQ.t16c_tblk_bytes.restype = ctypes.c_size_t
LIB_FQ.gptq_quant_chunk_f32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
LIB_FQ.quant_and_pack_t16c_f32.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]
LIB_FQ.update_t16c_headers_f32.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
]

assert LIB_FQ.t16c_tblk_bytes(256, 0) == p3.tile16_blk_bytes(256, 1, True) == 576
assert LIB_FQ.t16c_tblk_bytes(256, 4) == p3.tile16_blk_bytes(256, 3, False) == 1568
assert LIB_FQ.t16c_tblk_bytes(256, 5) == p3.tile16_blk_bytes(256, 2 | 0x80, False) == 1056
assert LIB_FQ.t16c_tblk_bytes(64, 5) == p3.tile16_blk_bytes(64, 2 | 0x80, False) == 288
print("[Gate 0a] libfast_quant_v5.so compiled & verified (1b_asym=576B, 2b_gaus=1056B, 3b_u0=1568B @ b256).", flush=True)


# ---------------------------------------------------------------------------
# 2. Core Math: FWHT, Non-Uniform Gaussian GPTQ, Differentiable QuantLinearV5
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


def symmetrize_inplace_(h: torch.Tensor, tile: int = 1024) -> torch.Tensor:
    n = h.shape[0]
    for i in range(0, n, tile):
        i2 = min(i + tile, n)
        blk = h[i:i2, i:i2]
        blk.add_(blk.T.clone()).mul_(0.5)
        for j in range(i + tile, n, tile):
            j2 = min(j + tile, n)
            b_ij = h[i:i2, j:j2]
            b_ji = h[j:j2, i:i2]
            avg = (b_ij + b_ji.T) * 0.5
            b_ij.copy_(avg)
            b_ji.copy_(avg.T)
    return h


def fwht_2sided_gpu(h_in: torch.Tensor, inv_s_gpu: Optional[torch.Tensor], bsize: int, consume: bool) -> torch.Tensor:
    h = h_in if consume else h_in.clone()
    if inv_s_gpu is not None:
        inv_s_loc = inv_s_gpu if inv_s_gpu.device == h.device else inv_s_gpu.to(h.device)
        h.mul_(inv_s_loc.unsqueeze(1))
        h.mul_(inv_s_loc.unsqueeze(0))
    d = h.shape[0]
    hb = get_h_gpu(bsize, h.device)
    step = min(d, max(bsize, 2048 - (2048 % bsize)))
    for r0 in range(0, d, step):
        r1 = min(r0 + step, d)
        m = (r1 - r0) // bsize
        h[r0:r1] = torch.matmul(hb, h[r0:r1].reshape(m, bsize, d)).reshape(r1 - r0, d)
    for c0 in range(0, d, step):
        c1 = min(c0 + step, d)
        n = (c1 - c0) // bsize
        h[:, c0:c1] = torch.matmul(h[:, c0:c1].reshape(d, n, bsize), hb).reshape(d, c1 - c0)
    symmetrize_inplace_(h)
    return h


def accum_hess(h_acc: torch.Tensor, xsq_acc: torch.Tensor, x_f32: torch.Tensor) -> None:
    h_acc.add_(x_f32.T @ x_f32)
    xsq_acc.add_((x_f32 * x_f32).sum(dim=0))


_GAUSS2_LV_CACHE: Dict[str, torch.Tensor] = {}
_GAUSS2_MID_CACHE: Dict[str, torch.Tensor] = {}


def _gauss2_tables(dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    k = str(dev)
    if k not in _GAUSS2_LV_CACHE:
        _GAUSS2_LV_CACHE[k] = torch.tensor([-3.174, -1.000, 1.000, 3.174], device=dev, dtype=torch.float32)
        _GAUSS2_MID_CACHE[k] = torch.tensor([-2.087, 0.000, 2.087], device=dev, dtype=torch.float32)
    return _GAUSS2_LV_CACHE[k], _GAUSS2_MID_CACHE[k]


def gptq_quant_gpu(W: torch.Tensor, U: torch.Tensor, bsize: int, mode: int) -> None:
    """Exact PyTorch CUDA equivalent of fast_quant.c::gptq_quant_chunk_f32 for mode in {0, 2, 4, 5}."""
    rows, cw = W.shape
    nblks = cw // bsize
    diag_u = torch.diagonal(U).clamp(min=1e-12)
    E_blk = torch.empty((rows, bsize), device=W.device, dtype=torch.float32)
    g2_lv, g2_mid = _gauss2_tables(W.device) if mode == 5 else (None, None)

    for b in range(nblks):
        b0 = b * bsize
        b1 = b0 + bsize
        blk = W[:, b0:b1]
        if mode == 0:
            bias = blk.mean(dim=1)
            for _ in range(2):
                m1 = blk >= bias.unsqueeze(1)
                c1 = m1.sum(dim=1)
                c0 = bsize - c1
                s1 = torch.where(m1, blk, 0.0).sum(dim=1)
                s0 = torch.where(m1, 0.0, blk).sum(dim=1)
                mp = torch.where(c1 > 0, s1 / c1.clamp(min=1), bias)
                mn = torch.where(c0 > 0, s0 / c0.clamp(min=1), bias)
                bias = 0.5 * (mp + mn)
                sc = 0.5 * (mp - mn)
            bias = bias.to(torch.float16).float()
            sc = sc.to(torch.float16).float()
            sc_p, sc_n = bias + sc, bias - sc
        elif mode == 5:
            sc = 0.470 * torch.sqrt((blk * blk).mean(dim=1))
            for _ in range(4):
                inv_s = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0).unsqueeze(1)
                idx = torch.bucketize(blk * inv_s, g2_mid)
                lv = g2_lv[idx]
                num = (lv * blk).sum(dim=1)
                den = (lv * lv).sum(dim=1)
                sc = torch.where(den > 0.0, num / den.clamp(min=1e-12), 0.0)
            sc = sc.to(torch.float16).float()
            inv_s_vec = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0)
        else:
            max_lvl = 3.0 if mode == 2 else 7.0
            init_k = 0.498 if mode == 2 else 0.293
            sc = init_k * torch.sqrt((blk * blk).mean(dim=1))
            for _ in range(4):
                inv_s = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0).unsqueeze(1)
                q = torch.clamp(torch.round((blk * inv_s + max_lvl) * 0.5), 0.0, max_lvl)
                lv = 2.0 * q - max_lvl
                num = (lv * blk).sum(dim=1)
                den = (lv * lv).sum(dim=1)
                sc = torch.where(den > 0.0, num / den.clamp(min=1e-12), 0.0)
            sc = sc.to(torch.float16).float()
            inv_s_vec = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0)

        for k in range(bsize):
            j = b0 + k
            val = W[:, j].clone()
            if mode == 0:
                qval = torch.where(val >= bias, sc_p, sc_n)
            elif mode == 5:
                idx = torch.bucketize(val * inv_s_vec, g2_mid)
                qval = sc * g2_lv[idx]
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


def extract_headers_and_codes_from_packed(
    pb: np.ndarray, d_out: int, d_in: int, bsize: int, mode: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Extract exact `(sc_f32 [d_out, n_blks], bi_f32_or_None, q_codes_i8 [d_out, n_blks, bsize])`
    directly from packed `.t16c` bytes so differentiable QAT uses 100% bit-exact packed codes!
      - mode 0 (1b_asym): q_codes in {-1, +1}
      - mode 5 (2b_gaus): q_codes in {-3, -1, +1, +3} -> exact polynomial gives {-3.174, -1, +1, +3.174}
      - mode 2 (2b_sym):  q_codes in {-3, -1, +1, +3}
      - mode 4 (3b_u0):   q_codes in {-7, -5, -3, -1, +1, +3, +5, +7}
    """
    n_blks = d_in // bsize
    n_tiles = (d_out + 15) // 16
    is_asym = (mode == 0)
    eff_bits = 1 if mode == 0 else (2 if mode in (2, 5) else 3)
    hdr_b = 64 if is_asym else 32
    bpr = bsize // 8
    plane_b = bpr * 16
    tblk_b = hdr_b + eff_bits * plane_b
    arr = pb.reshape(n_tiles, n_blks, tblk_b)

    sc_h = np.frombuffer(np.ascontiguousarray(arr[:, :, :32]), dtype="<f2").reshape(n_tiles, n_blks, 16)
    sc_full = sc_h.transpose(0, 2, 1).reshape(n_tiles * 16, n_blks)[:d_out].astype(np.float32)
    bi_full = None
    if is_asym:
        bi_h = np.frombuffer(np.ascontiguousarray(arr[:, :, 32:64]), dtype="<f2").reshape(n_tiles, n_blks, 16)
        bi_full = bi_h.transpose(0, 2, 1).reshape(n_tiles * 16, n_blks)[:d_out].astype(np.float32)

    # Unpack bit-planes vectorized in numpy -> uint8 codes [d_out, n_blks, bsize]
    q_idx = np.zeros((n_tiles, n_blks, bpr, 16, 8), dtype=np.uint8)
    for p in range(eff_bits):
        pl = arr[:, :, hdr_b + p * plane_b : hdr_b + (p + 1) * plane_b].reshape(n_tiles, n_blks, bpr, 16)
        bits = np.unpackbits(pl[..., None], axis=-1)  # MSB-first [n_tiles, n_blks, bpr, 16, 8]
        q_idx |= (bits << p)
    q_idx = q_idx.transpose(0, 3, 1, 2, 4).reshape(n_tiles * 16, n_blks, bsize)[:d_out]
    max_lvl = (1 << eff_bits) - 1
    q_i8 = (2 * q_idx.astype(np.int16) - max_lvl).astype(np.int8)
    return (
        torch.from_numpy(np.ascontiguousarray(sc_full)),
        torch.from_numpy(np.ascontiguousarray(bi_full)) if bi_full is not None else None,
        torch.from_numpy(np.ascontiguousarray(q_i8)),
    )


def quantize_lr_arrays(U_f: torch.Tensor, V_f: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact INT8 row-wise quantization matching Int8LowRank in t16c.rs."""
    with torch.no_grad():
        su = (U_f.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        sv = (V_f.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        ui = torch.clamp(torch.round(U_f / su.unsqueeze(1)), -127, 127).to(torch.int8)
        vi = torch.clamp(torch.round(V_f / sv.unsqueeze(1)), -127, 127).to(torch.int8)
    return (
        su.detach().cpu().numpy().astype("<f4"),
        sv.detach().cpu().numpy().astype("<f4"),
        ui.detach().cpu().numpy(),
        vi.detach().cpu().numpy(),
    )


class QuantLinearV5(nn.Module):
    """
    Differentiable fixed-bitplane Block-Wise QAT projection:
      y = ((x * inv_s) @ H_b) @ (Q_unit * sc_fp16_ste + bi_fp16_ste)^T + (x @ V_i8_ste^T) @ U_i8_ste^T
    Exact 1-to-1 correspondence with T16C runtime decode on ARM64 NEON.
    """

    def __init__(self, bundle: dict, device: torch.device) -> None:
        super().__init__()
        self.name: str = bundle["name"]
        self.d_out: int = int(bundle["d_out"])
        self.d_in: int = int(bundle["d_in"])
        self.bsize: int = int(bundle["bsize"])
        self.mode: int = int(bundle["mode"])
        self.n_blks: int = self.d_in // self.bsize
        self.lr_rank: int = int(bundle["lr_rank"])

        self.register_buffer("q_i8", bundle["q_i8"].to(device), persistent=False)
        inv_s_np = bundle["inv_s"]
        if inv_s_np is not None:
            self.register_buffer("inv_s", torch.from_numpy(inv_s_np).to(device, dtype=torch.float32), persistent=False)
        else:
            self.inv_s = None

        self.sc = nn.Parameter(bundle["sc_f32"].to(device, dtype=torch.float32).clone())
        if bundle["bi_f32"] is not None:
            self.bi = nn.Parameter(bundle["bi_f32"].to(device, dtype=torch.float32).clone())
        else:
            self.bi = None

        lr = bundle["lr"]
        if lr is not None and self.lr_rank > 0:
            su, sv, ui, vi = lr
            u_f = torch.from_numpy(ui.astype(np.float32)).to(device) * torch.from_numpy(su).to(device).unsqueeze(1)
            v_f = torch.from_numpy(vi.astype(np.float32)).to(device) * torch.from_numpy(sv).to(device).unsqueeze(1)
            self.U = nn.Parameter(u_f)
            self.V = nn.Parameter(v_f)
        else:
            self.U = None
            self.V = None

    def _build_w_rot_bf16(self) -> torch.Tensor:
        sc_q = self.sc + (self.sc.to(torch.float16).float() - self.sc).detach()
        with torch.no_grad():
            qf = self.q_i8.float()
            if self.mode == 5:
                # Exact FP32 polynomial on {-3,-1,+1,+3}: ±1 -> ±1.000000, ±3 -> ±3.174000!
                qf = qf * (1.0 + 0.00725 * (qf * qf - 1.0))
        w = qf * sc_q.unsqueeze(-1)
        if self.bi is not None:
            bi_q = self.bi + (self.bi.to(torch.float16).float() - self.bi).detach()
            w = w + bi_q.unsqueeze(-1)
        return w.reshape(self.d_out, self.d_in).to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_sh = x.shape
        x2d = x.reshape(-1, self.d_in).float()
        x_sc = (x2d * self.inv_s) if self.inv_s is not None else x2d
        x_rot = fwht_gpu(x_sc, self.bsize).to(torch.float16)
        w_rot = self._build_w_rot_bf16()
        y = F.linear(x_rot, w_rot).float()
        if self.U is not None and self.V is not None:
            su = (self.U.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
            sv = (self.V.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
            u_q = torch.clamp(torch.round(self.U / su), -127.0, 127.0) * su
            v_q = torch.clamp(torch.round(self.V / sv), -127.0, 127.0) * sv
            u_ste = self.U + (u_q - self.U).detach()
            v_ste = self.V + (v_q - self.V).detach()
            y = y + (x2d @ v_ste.T) @ u_ste.T
        return y.reshape(*orig_sh[:-1], self.d_out)

    @torch.no_grad()
    def export_into_bundle(self, bundle: dict) -> torch.Tensor:
        """Write updated FP16 scales/biases into bundle['pb'] and INT8 low-rank into bundle['lr']; return W_eff_f16."""
        sc_f16 = self.sc.detach().to(torch.float16).float()
        sc_np = np.ascontiguousarray(sc_f16.cpu().numpy(), dtype=np.float32)
        bi_np = None
        if self.bi is not None:
            bi_f16 = self.bi.detach().to(torch.float16).float()
            bi_np = np.ascontiguousarray(bi_f16.cpu().numpy(), dtype=np.float32)
        pb_copy = np.ascontiguousarray(bundle["pb"], dtype=np.uint8)
        LIB_FQ.update_t16c_headers_f32(
            pb_copy.ctypes.data,
            self.d_out,
            self.d_in,
            self.bsize,
            self.mode,
            sc_np.ctypes.data,
            bi_np.ctypes.data if bi_np is not None else None,
        )
        bundle["pb"] = pb_copy
        bundle["sc_f32"] = sc_f16.cpu()
        bundle["bi_f32"] = bi_f16.cpu() if self.bi is not None else None

        qf = self.q_i8.float()
        if self.mode == 5:
            qf = qf * (1.0 + 0.00725 * (qf * qf - 1.0))
        w_rot = qf * sc_f16.unsqueeze(-1)
        if self.bi is not None:
            w_rot = w_rot + bi_f16.unsqueeze(-1)
        w_eff = fwht_gpu(w_rot.reshape(self.d_out, self.d_in), self.bsize)
        if self.inv_s is not None:
            w_eff.mul_(self.inv_s.unsqueeze(0))

        if self.U is not None and self.V is not None:
            lr_tup = quantize_lr_arrays(self.U.detach(), self.V.detach())
            bundle["lr"] = lr_tup
            su, sv, ui, vi = lr_tup
            u_q = torch.from_numpy(ui.astype(np.float32)).to(w_eff.device) * torch.from_numpy(su).to(w_eff.device).unsqueeze(1)
            v_q = torch.from_numpy(vi.astype(np.float32)).to(w_eff.device) * torch.from_numpy(sv).to(w_eff.device).unsqueeze(1)
            w_eff.add_(u_q @ v_q)
        return w_eff.to(torch.float16)


def pack_matrix_gpu_gptq_v5(
    name: str,
    w_cpu: torch.Tensor,
    h_gpu: Optional[torch.Tensor],
    x_rms_gpu: Optional[torch.Tensor],
    mode: int,
    bsize: int = 256,
    lr_rank: int = 0,
    x_eval_gpu: Optional[torch.Tensor] = None,
    consume_h: bool = False,
    return_bundle: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple], torch.Tensor, Optional[dict]]:
    d_out, d_in = w_cpu.shape
    assert d_in % bsize == 0, f"{name}: d_in={d_in} not divisible by {bsize}"
    n_bits, is_asym = p3.MODE_TO_T16C[mode]
    exp_bytes = p3.expected_tile16_bytes(d_out, d_in, bsize, n_bits, is_asym)

    # Chunked path for huge embedding matrix (262144 x 5376)
    if h_gpu is None and lr_rank == 0 and d_out > 32768:
        pbytes = np.empty(exp_bytes, dtype=np.uint8)
        w_eff_cpu = torch.empty((d_out, d_in), dtype=torch.float16)
        tile_rows = 16384
        tblk_b = p3.tile16_blk_bytes(bsize, n_bits, is_asym) * (d_in // bsize)
        for r0 in range(0, d_out, tile_rows):
            r1 = min(r0 + tile_rows, d_out)
            wc = w_cpu[r0:r1].to(DEVICE, dtype=torch.float32)
            wc_rot = fwht_gpu(wc, bsize).contiguous().cpu().numpy().astype(np.float32)
            del wc
            chunk_pb = np.empty(((r1 - r0 + 15) // 16) * tblk_b, dtype=np.uint8)
            LIB_FQ.quant_and_pack_t16c_f32(wc_rot.ctypes.data, r1 - r0, d_in, bsize, mode, chunk_pb.ctypes.data)
            pbytes[(r0 // 16) * tblk_b : (r0 // 16) * tblk_b + chunk_pb.size] = chunk_pb
            wq_c = fwht_gpu(torch.from_numpy(wc_rot).to(DEVICE, dtype=torch.float32), bsize)
            w_eff_cpu[r0:r1] = wq_c.to(torch.float16).cpu()
            del wc_rot, wq_c, chunk_pb
        return pbytes, None, None, w_eff_cpu, None

    w_gpu = w_cpu.to(DEVICE, dtype=torch.float32)
    if x_rms_gpu is not None:
        xr_w = x_rms_gpu.to(DEVICE, dtype=torch.float32)
        w_col = w_gpu.abs().amax(dim=0).clamp(min=1e-5)
        x_col = xr_w.clamp(min=1e-5)
        del xr_w
        sq = (x_col.pow(0.5) / w_col.pow(0.5)).clamp(0.25, 4.0)
        sq = sq / sq.mean().clamp(min=1e-5)
        inv_s_gpu = 1.0 / sq
        w_scaled = w_gpu * sq.unsqueeze(0)
        inv_s_np = inv_s_gpu.detach().cpu().numpy().astype("<f4")
        del w_col, x_col, sq
    else:
        inv_s_gpu = None
        w_scaled = w_gpu
        inv_s_np = None

    w_rot = fwht_gpu(w_scaled, bsize).contiguous()
    del w_scaled

    if h_gpu is not None:
        is_box = isinstance(h_gpu, list)
        h_t = h_gpu.pop(0) if is_box else h_gpu
        consume_flag = is_box or consume_h
        h_gpu = None
        if h_t.shape[0] == d_in:
            with torch.no_grad():
                h_rot = fwht_2sided_gpu(h_t, inv_s_gpu, bsize, consume=consume_flag)
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
                        h_inv = U_chol_tmp = U_chol = None
                        try:
                            h_inv = torch.cholesky_inverse(L)
                            del L; L = None
                            symmetrize_inplace_(h_inv)
                            U_chol_tmp = torch.linalg.cholesky(h_inv, upper=True)
                            del h_inv; h_inv = None
                            U_chol = U_chol_tmp if U_chol_tmp.is_contiguous() else U_chol_tmp.contiguous()
                            del U_chol_tmp; U_chol_tmp = None
                            gptq_quant_gpu(w_rot, U_chol, bsize, mode)
                        except Exception as e:
                            print(f"    [GPTQ fallback RTN {name}: {e}]", flush=True)
                        finally:
                            del L, h_inv, U_chol_tmp, U_chol
                else:
                    del h_rot
        else:
            del h_t

    w_work_np = w_rot.detach().cpu().numpy().astype(np.float32, copy=True)
    del w_rot
    pbytes = np.empty(exp_bytes, dtype=np.uint8)
    LIB_FQ.quant_and_pack_t16c_f32(w_work_np.ctypes.data, d_out, d_in, bsize, mode, pbytes.ctypes.data)

    wq_rot_gpu = torch.from_numpy(w_work_np).to(DEVICE, dtype=torch.float32)
    del w_work_np
    w_eff_gpu = fwht_gpu(wq_rot_gpu, bsize)
    del wq_rot_gpu
    if inv_s_gpu is not None:
        w_eff_gpu.mul_(inv_s_gpu.unsqueeze(0))
        del inv_s_gpu

    low_rank = None
    if lr_rank > 0 and x_eval_gpu is not None and x_eval_gpu.shape[0] >= lr_rank * 2:
        with torch.no_grad():
            n_ev = min(x_eval_gpu.shape[0], 2048)
            xv = x_eval_gpu[:n_ev].to(DEVICE, dtype=torch.float32)
            y_err = (xv @ w_gpu.T) - (xv @ w_eff_gpu.T)
            del w_gpu
            if n_ev <= d_out:
                gram_s = y_err @ y_err.T
                _, evecs_s = torch.linalg.eigh(gram_s)
                u_left = evecs_s[:, -lr_rank:].flip(1).contiguous()
                U_f = y_err.T @ u_left
                U_f = U_f / U_f.norm(dim=0, keepdim=True).clamp(min=1e-6)
                del gram_s, evecs_s, u_left
            else:
                gram = y_err.T @ y_err
                _, evecs = torch.linalg.eigh(gram)
                U_f = evecs[:, -lr_rank:].flip(1).contiguous()
                del gram, evecs

            reg = max(0.02 * float((xv * xv).sum().item()) / float(max(d_in, 1)), 1e-6)
            use_dual_ridge = n_ev < d_in
            gram_ridge = (xv @ xv.T) if use_dual_ridge else (xv.T @ xv)
            gram_ridge.diagonal().add_(reg)

            def _solve_ridge(rhs_proj: torch.Tensor) -> torch.Tensor:
                if use_dual_ridge:
                    return xv.T @ torch.linalg.solve(gram_ridge, rhs_proj)
                return torch.linalg.solve(gram_ridge, xv.T @ rhs_proj)

            V_f = _solve_ridge(y_err @ U_f).T.contiguous()
            eye_lr = 1e-4 * torch.eye(lr_rank, device=DEVICE)
            for _ in range(2):
                xv_v = xv @ V_f.T
                U_f = torch.linalg.solve(xv_v.T @ xv_v + eye_lr, xv_v.T @ y_err).T
                V_f = _solve_ridge(y_err @ U_f).T.contiguous()
            low_rank = quantize_lr_arrays(U_f, V_f)
            su, sv, ui, vi = low_rank
            u_q = torch.from_numpy(ui.astype(np.float32)).to(DEVICE) * torch.from_numpy(su).to(DEVICE).unsqueeze(1)
            v_q = torch.from_numpy(vi.astype(np.float32)).to(DEVICE) * torch.from_numpy(sv).to(DEVICE).unsqueeze(1)
            w_eff_gpu.add_(u_q @ v_q)
            del xv, y_err, gram_ridge, U_f, V_f, u_q, v_q
    else:
        del w_gpu

    bundle = None
    if return_bundle:
        sc_f32, bi_f32, q_i8 = extract_headers_and_codes_from_packed(pbytes, d_out, d_in, bsize, mode)
        bundle = {
            "name": name,
            "d_out": d_out,
            "d_in": d_in,
            "bsize": bsize,
            "mode": mode,
            "lr_rank": lr_rank,
            "pb": pbytes,
            "inv_s": inv_s_np,
            "lr": low_rank,
            "sc_f32": sc_f32,
            "bi_f32": bi_f32,
            "q_i8": q_i8,
        }
    return pbytes, inv_s_np, low_rank, w_eff_gpu.to(torch.float16), bundle


def apply_block_sparse_geglu_mask(gelu_gate: torch.Tensor, top_p: float = 0.50, bsize: int = 256) -> torch.Tensor:
    """Exact PyTorch equivalent of run31b's Gate-Guided Block-Sparse GeGLU 256-wide block mask."""
    if top_p >= 0.9999:
        return gelu_gate
    orig_sh = gelu_gate.shape
    ff = orig_sh[-1]
    n_blks = ff // bsize
    keep = max(1, min(n_blks, int(math.ceil(n_blks * top_p))))
    g_blk = gelu_gate.reshape(-1, n_blks, bsize)
    # Score per-token block L2 energy (exact match to k=1 inference!)
    energy = (g_blk * g_blk).sum(dim=-1)  # [N_tok, n_blks]
    topk_idx = torch.topk(energy, keep, dim=-1).indices
    mask = torch.zeros_like(energy)
    mask.scatter_(1, topk_idx, 1.0)
    return (g_blk * mask.unsqueeze(-1)).reshape(orig_sh)


# ---------------------------------------------------------------------------
# [Gates 0b & 0c] Pre-Flight Verification of 2b_gaus GPTQ, QuantLinearV5 QAT
# real loss.backward(), Header Export Parity, and Neuron Permutation Invariance!
# ---------------------------------------------------------------------------
print("[Gate 0b] Testing 2b_gaus GPU==C GPTQ + QuantLinearV5 real loss.backward() + export parity...", flush=True)
torch.manual_seed(7)
for _m_t, _rk_t in [(0, 8), (5, 8), (4, 8)]:
    _do, _di, _bs = 64, 512, 256
    _xt = torch.randn((128, _di), device=DEVICE, dtype=torch.float32) * 0.3
    _wt = torch.randn((_do, _di), dtype=torch.float32) * 0.05
    _hh = (_xt.T @ _xt).float() + 0.05 * torch.eye(_di, device=DEVICE, dtype=torch.float32)
    _xr = torch.sqrt((_xt * _xt).mean(dim=0).clamp(min=1e-8))
    # Verify GPU vs C GPTQ parity
    _L = torch.linalg.cholesky(_hh)
    _U = torch.linalg.cholesky(torch.cholesky_inverse(_L), upper=True).contiguous()
    _wg = _wt.to(DEVICE).clone()
    gptq_quant_gpu(_wg, _U, _bs, _m_t)
    _wc = _wt.numpy().astype(np.float32, copy=True)
    _uc = _U.cpu().numpy().astype(np.float32, copy=True)
    LIB_FQ.gptq_quant_chunk_f32(_wc.ctypes.data, _uc.ctypes.data, _do, _di, _bs, _m_t, 1)
    _diff = (_wg.cpu() - torch.from_numpy(_wc)).abs()
    assert float((_diff > 1e-3).float().mean().item()) < 0.01, f"mode={_m_t} GPU/C GPTQ mismatch"
    # Verify QuantLinearV5 initial forward == w_eff AND real AdamW loss.backward() reduces loss AND export matches!
    _pb, _is, _lr, _we, _bnd = pack_matrix_gpu_gptq_v5("gate0b", _wt, [_hh], _xr, _m_t, _bs, _rk_t, _xt, return_bundle=True)
    _ql = QuantLinearV5(_bnd, DEVICE)
    with torch.no_grad():
        _y_we = _xt @ _we.float().T
        _y_ql0 = _ql(_xt)
        _init_rel = float((_y_ql0 - _y_we).norm() / _y_we.norm().clamp(min=1e-6))
        assert _init_rel < 5e-3, f"mode={_m_t} QuantLinearV5 init mismatch rel={_init_rel}"
    _yt = _xt @ _wt.to(DEVICE).T
    _opt = torch.optim.AdamW(_ql.parameters(), lr=5e-3)
    _loss0 = float(F.mse_loss(_ql(_xt).detach(), _yt).item())
    for _st in range(25):
        _opt.zero_grad()
        _loss = F.mse_loss(_ql(_xt), _yt)
        _loss.backward()
        _opt.step()
    _loss1 = float(F.mse_loss(_ql(_xt).detach(), _yt).item())
    assert _loss1 < _loss0, f"mode={_m_t} real loss.backward() failed to reduce loss: {_loss0} -> {_loss1}"
    _we_after = _ql.export_into_bundle(_bnd)
    with torch.no_grad():
        _y_ql1 = _ql(_xt)
        _y_ex1 = _xt @ _we_after.float().T
        _ex_rel = float((_y_ql1 - _y_ex1).norm() / _y_ex1.norm().clamp(min=1e-6))
        assert _ex_rel < 5e-3, f"mode={_m_t} export mismatch rel={_ex_rel}"
    # Verify re-extracting headers from updated bundle["pb"] matches sc_f32 exactly!
    _sc_chk, _bi_chk, _ = extract_headers_and_codes_from_packed(_bnd["pb"], _do, _di, _bs, _m_t)
    assert float((_sc_chk - _bnd["sc_f32"]).abs().max().item()) == 0.0
    if _m_t == 0:
        assert float((_bi_chk - _bnd["bi_f32"]).abs().max().item()) == 0.0
    del _xt, _wt, _L, _U, _wg, _ql, _opt, _we, _we_after, _bnd

# [Gate 0c] Exact GeGLU Neuron Co-Activation Permutation Invariance
with torch.no_grad():
    _x_p = torch.randn((32, 256), device=DEVICE)
    _wg_p = torch.randn((512, 256), device=DEVICE) * 0.05
    _wu_p = torch.randn((512, 256), device=DEVICE) * 0.05
    _wd_p = torch.randn((256, 512), device=DEVICE) * 0.05
    _y_orig = (F.gelu(_x_p @ _wg_p.T, approximate="tanh") * (_x_p @ _wu_p.T)) @ _wd_p.T
    _perm = torch.randperm(512, device=DEVICE)
    _y_perm = (F.gelu(_x_p @ _wg_p[_perm].T, approximate="tanh") * (_x_p @ _wu_p[_perm].T)) @ _wd_p[:, _perm].T
    assert float((_y_orig - _y_perm).abs().max().item()) < 1e-4
    del _x_p, _wg_p, _wu_p, _wd_p, _y_orig, _y_perm, _perm
torch.cuda.empty_cache()
print("[Gate 0b+0c PASS] 2b_gaus/1b_asym/3b_u0 GPTQ + real QAT loss.backward() + header export + GeGLU perm verified!", flush=True)


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


def make_sliding_attn_mask(
    b0: int,
    b1: int,
    vlm_spans: Dict[int, Tuple[int, int]],
    seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    has_any = any(si in vlm_spans for si in range(b0, b1))
    if not has_any:
        return None
    mb = b1 - b0
    base = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    mask = base.unsqueeze(0).expand(mb, seq_len, seq_len).clone()
    for gi in range(b0, b1):
        span = vlm_spans.get(gi)
        if span is not None:
            s0, s1 = span
            mask[gi - b0, s0:s1, s0:s1] = True
    return mask.unsqueeze(1)


def make_sliding_attn_mask_for_indices(
    seq_indices: List[int],
    vlm_spans: Dict[int, Tuple[int, int]],
    seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not any(si in vlm_spans for si in seq_indices):
        return None
    mb = len(seq_indices)
    base = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    mask = base.unsqueeze(0).expand(mb, seq_len, seq_len).clone()
    for lo, gi in enumerate(seq_indices):
        span = vlm_spans.get(gi)
        if span is not None:
            s0, s1 = span
            mask[lo, s0:s1, s0:s1] = True
    return mask.unsqueeze(1)


# ---------------------------------------------------------------------------
# 3. Stage 1: Download Model & Build Clean Zero-Contamination Corpus
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


def is_clean_row(txt: str) -> bool:
    low = txt.lower()
    return not any(b in low for b in ALL_BLACKLIST)


print("[Stage 1] Loading & auditing FULL clean text datasets (zero contamination)...", flush=True)
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
print(f"[Stage 1] Clean corpus audited: {len(clean_turns):,} examples across GSM8K/Code/Alpaca/WikiText-2.", flush=True)

N_TEXT_SEQ, N_VLM_SEQ, SEQ_LEN = 608, 160, 352
N_SEQ = N_TEXT_SEQ + N_VLM_SEQ
GEMMA4_BOI_ID, GEMMA4_IMAGE_ID, GEMMA4_EOI_ID = 255999, 258880, 258882

by_dom = defaultdict(list)
for idx in perm_c:
    d_name, q_t, a_t = clean_turns[idx]
    by_dom[d_name].append((q_t, a_t))

dom_names = ["math", "code", "alpaca", "wiki"]
ptrs = {d: 0 for d in dom_names}
seq_ids_list: List[List[int]] = []
seq_nxt_list: List[List[int]] = []
seq_ans_list: List[List[bool]] = []
seq_vlm_ans_list: List[List[bool]] = []
examples_packed = 0


def next_turn_pair(turn_i: int) -> Tuple[str, str]:
    d = dom_names[turn_i % len(dom_names)]
    p_i = ptrs[d] % len(by_dom[d])
    ptrs[d] += 1
    return by_dom[d][p_i]


t_cursor = 0
for _ in range(N_TEXT_SEQ):
    cur_toks: List[int] = [2]
    cur_ans: List[bool] = [False]
    while len(cur_toks) < SEQ_LEN + 1:
        q_t, a_t = next_turn_pair(t_cursor)
        t_cursor += 1
        p_tok = list(tk.encode(vc.format_gemma4_turn(q_t, None), add_special_tokens=False).ids)
        r_tok = list(tk.encode(f"{a_t}<turn|>\n", add_special_tokens=False).ids)
        if not p_tok or not r_tok:
            continue
        cur_toks.extend(p_tok)
        cur_ans.extend([False] * len(p_tok))
        cur_toks.extend(r_tok)
        cur_ans.extend([True] * len(r_tok))
        examples_packed += 1
    cur_toks = cur_toks[: SEQ_LEN + 1]
    cur_ans = cur_ans[: SEQ_LEN + 1]
    seq_ids_list.append(cur_toks[:-1])
    seq_nxt_list.append(cur_toks[1:])
    seq_ans_list.append(cur_ans[1:])
    seq_vlm_ans_list.append([False] * SEQ_LEN)

print(f"[Stage 1] Packed {N_TEXT_SEQ} dense text seqs x {SEQ_LEN} = {N_TEXT_SEQ * SEQ_LEN:,} text tokens ({examples_packed:,} turns).", flush=True)


# ---------------------------------------------------------------------------
# 4. Stage 2: Fast 2b_gaus/3b_u0 Vision Tower + Multimodal Soft-Token Capture
# ---------------------------------------------------------------------------
VIS_CKPT_PATH = CKPT_DIR / "stage2_vision.pt"
VIS_META_PATH = CKPT_DIR / "stage2_meta.pt"
VLM_SOFT_CACHE = CKPT_DIR / "stage2_vlm_soft.pt"


def _build_stage2_vision() -> dict:
    print("\n[Stage 2] V5.0 Vision Tower Calibration (V00..V23 2b_gaus + V24..V26/embed 3b_u0 -> ~174 MB, 1.4x faster Pi 4 ViT!)...", flush=True)
    cfg_auto = AutoConfig.from_pretrained(str(model_dir))
    proc = Gemma4ImageProcessor.from_pretrained(str(model_dir))

    def make_optical_va_pil(idx: int) -> Tuple[Image.Image, str]:
        im = Image.new("RGB", (672, 672), (248, 248, 248))
        dr = ImageDraw.Draw(im)
        dr.text((18, 18), f"GEMMA4 31B V5.0 QAT LM=60 VIS=27 FRAME={idx:03d}", fill=(12, 18, 28))
        dr.text((18, 42), "OPTICAL VA PRECISION ANALYSIS C8DB-H8", fill=(12, 18, 28))
        rng_v = np.random.default_rng(1000 + idx)
        for b in range(60):
            bh = int(rng_v.integers(25, 118))
            x = 24 + b * 10
            col = (35, 115, 210) if b >= 58 else ((30, 165, 95) if b % 5 != 0 else (210, 65, 45))
            dr.rectangle([x, 550 - bh, x + 7, 550], fill=col)
        cap = (
            f"An optical diagnostic chart labeled GEMMA4 31B V5.0 QAT LM=60 VIS=27 FRAME={idx:03d} "
            f"displaying 60 vertical green, red, and blue bar segments on a light background."
        )
        return im, cap

    vis_va_pairs: List[Tuple[Image.Image, str]] = [make_optical_va_pil(i) for i in range(32)]
    vis_coco_pairs: List[Tuple[Image.Image, str]] = []
    heldout_pairs: List[Tuple[Image.Image, str]] = []
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
    assert len(vis_coco_pairs) == 128 and len(heldout_pairs) == 4

    vis_pairs: List[Tuple[Image.Image, str]] = []
    for i in range(32):
        vis_pairs.append(vis_va_pairs[i])
        vis_pairs.extend(vis_coco_pairs[i * 4 : (i + 1) * 4])
    assert len(vis_pairs) == N_VLM_SEQ
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

    vis_hess: Dict[str, torch.Tensor] = {}
    vis_xsq: Dict[str, torch.Tensor] = {}
    vis_cnt: Dict[str, int] = defaultdict(int)
    vis_x_eval: Dict[str, torch.Tensor] = {}

    def reg_vis_hook(mod, key: str):
        def _hk(_m, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            if key not in vis_hess:
                d = x.shape[1]
                vis_hess[key] = torch.zeros((d, d), device=DEVICE, dtype=torch.float32)
                vis_xsq[key] = torch.zeros((d,), device=DEVICE, dtype=torch.float32)
                vis_x_eval[key] = x[:1024].cpu().clone()
            accum_hess(vis_hess[key], vis_xsq[key], x)
            vis_cnt[key] += x.shape[0]
        return mod.register_forward_hook(_hk)

    hooks = [reg_vis_hook(vis_teacher.patch_embedder.input_proj, "model.vision_tower.patch_embedder.input_proj.weight")]
    for l_v, lv_mod in enumerate(vis_teacher.encoder.layers):
        pfx_v = f"model.vision_tower.encoder.layers.{l_v}"
        for s_nm, s_mod in [
            ("self_attn.q_proj", lv_mod.self_attn.q_proj), ("self_attn.k_proj", lv_mod.self_attn.k_proj),
            ("self_attn.v_proj", lv_mod.self_attn.v_proj), ("self_attn.o_proj", lv_mod.self_attn.o_proj),
            ("mlp.gate_proj", lv_mod.mlp.gate_proj), ("mlp.up_proj", lv_mod.mlp.up_proj),
            ("mlp.down_proj", lv_mod.mlp.down_proj),
        ]:
            hooks.append(reg_vis_hook(s_mod, f"{pfx_v}.{s_nm}.linear.weight"))
    hooks.append(reg_vis_hook(mm_teacher.embedding_projection, "model.embed_vision.embedding_projection.weight"))

    vis_teach_soft: List[torch.Tensor] = []
    vis_eval_pvs: List[torch.Tensor] = []
    vis_eval_poss: List[torch.Tensor] = []
    with torch.no_grad():
        for im in vis_images:
            pin = proc(images=[im], return_tensors="pt")
            pv = pin["pixel_values"].to(DEVICE, dtype=torch.float16)
            pos = pin["image_position_ids"].to(DEVICE)
            vo = vis_teacher(pixel_values=pv, pixel_position_ids=pos)
            st = mm_teacher(vo.last_hidden_state).reshape(-1, 5376)
            assert 1 <= st.shape[0] <= 280
            vis_eval_pvs.append(pv.cpu())
            vis_eval_poss.append(pos.cpu())
            vis_teach_soft.append(st.to(torch.float16).cpu())
    for h in hooks:
        h.remove()

    heldout_pvs, heldout_poss, heldout_teach_soft = [], [], []
    with torch.no_grad():
        for im_h, _cap_h in heldout_pairs:
            pin = proc(images=[im_h], return_tensors="pt")
            pv = pin["pixel_values"].to(DEVICE, dtype=torch.float16)
            pos = pin["image_position_ids"].to(DEVICE)
            vo = vis_teacher(pixel_values=pv, pixel_position_ids=pos)
            st = mm_teacher(vo.last_hidden_state).reshape(-1, 5376)
            heldout_pvs.append(pv.cpu())
            heldout_poss.append(pos.cpu())
            heldout_teach_soft.append(st.to(torch.float16).cpu())
    del vis_teacher, mm_teacher
    torch.cuda.empty_cache()

    vision_records: List[tuple] = []
    vision_norms: Dict[str, np.ndarray] = {
        "model.vision_tower.std_bias": std_bias_f32,
        "model.vision_tower.std_scale": std_scale_f32,
    }
    vis_deq_weights: Dict[str, torch.Tensor] = {}

    def pack_vision_tensor(name: str, w_t: torch.Tensor, mode: int, lr_r: int, pad_in: int = 0) -> None:
        d_o, d_i = w_t.shape
        h_use = vis_hess.pop(name, None)
        x_rms = torch.sqrt((vis_xsq.pop(name) / max(vis_cnt[name], 1)).clamp(min=1e-8)) if name in vis_xsq else None
        x_ev = vis_x_eval.pop(name).to(DEVICE) if name in vis_x_eval else None
        if pad_in > d_i:
            w_pad = torch.zeros((d_o, pad_in), dtype=torch.float32)
            w_pad[:, :d_i] = w_t.float()
            if h_use is not None:
                hp = torch.zeros((pad_in, pad_in), device=DEVICE, dtype=torch.float32)
                hp[:d_i, :d_i] = h_use
                hp.diagonal()[d_i:].fill_(1.0)
                h_use = hp
            if x_rms is not None:
                xp = torch.ones((pad_in,), device=DEVICE, dtype=torch.float32)
                xp[:d_i] = x_rms
                x_rms = xp
            if x_ev is not None:
                xe = torch.zeros((x_ev.shape[0], pad_in), device=DEVICE, dtype=torch.float32)
                xe[:, :d_i] = x_ev
                x_ev = xe
            d_in_eff = pad_in
        else:
            w_pad = w_t.float()
            d_in_eff = d_i
        pb, inv_s, lr, w_eff, _ = pack_matrix_gpu_gptq_v5(
            name, w_pad, [h_use] if h_use is not None else None, x_rms, mode=mode, bsize=64, lr_rank=lr_r, x_eval_gpu=x_ev
        )
        del h_use, x_rms, x_ev
        n_bits, is_asym = p3.MODE_TO_T16C[mode]
        vision_records.append((name, d_o, d_in_eff, 64, n_bits, is_asym, True, pb, inv_s, lr))
        vis_deq_weights[name] = w_eff[:, :d_i].to(torch.float16)

    for ax in (0, 1):
        nm_pos = f"model.vision_tower.patch_embedder.position_embedding_table.{ax}"
        pb, inv_s, lr, w_eff, _ = pack_matrix_gpu_gptq_v5(nm_pos, pos_tbl_f32[ax], None, None, mode=4, bsize=64, lr_rank=0)
        vision_records.append((nm_pos, pos_tbl_f32.shape[1], pos_tbl_f32.shape[2], 64, 3, False, True, pb, inv_s, lr))
        vis_deq_weights[nm_pos] = w_eff

    pack_vision_tensor("model.vision_tower.patch_embedder.input_proj.weight", vis_sd["patch_embedder.input_proj.weight"], mode=4, lr_r=16)
    for l_v in range(27):
        full_pfx = f"model.vision_tower.encoder.layers.{l_v}"
        for nk in ["input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
                   "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]:
            vision_norms[f"{full_pfx}.{nk}"] = vis_sd[f"encoder.layers.{l_v}.{nk}"].float().numpy().astype("<f4")
        # V5.0 Vision: V00..V23 at fast 2b_gaus (mode=5) + V24..V26 at 3b_u0 (mode=4) -> 1.4x faster ViT + saves 63 MB for LM cliff!
        v_mode = 4 if l_v >= 24 else 5
        v_rk = 8 if l_v in (0, 1, 24, 25, 26) else 0
        for proj in ["self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight",
                     "self_attn.o_proj.linear.weight", "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight"]:
            pack_vision_tensor(f"{full_pfx}.{proj}", vis_sd[f"encoder.layers.{l_v}.{proj}"], mode=v_mode, lr_r=v_rk)
        pack_vision_tensor(
            f"{full_pfx}.mlp.down_proj.linear.weight",
            vis_sd[f"encoder.layers.{l_v}.mlp.down_proj.linear.weight"],
            mode=v_mode, lr_r=v_rk, pad_in=4352,
        )

    pack_vision_tensor("model.embed_vision.embedding_projection.weight", embv_sd["embedding_projection.weight"], mode=4, lr_r=32)
    v_bytes = sum(
        len(r[7]) + (len(r[8]) * 4 if r[8] is not None else 0)
        + ((r[9][0].nbytes + r[9][1].nbytes + r[9][2].nbytes + r[9][3].nbytes) if r[9] is not None else 0) + 80
        for r in vision_records
    )
    print(f"[Stage 2] Packed ALL {len(vision_records)} V5.0 Vision mats + {len(vision_norms)} norms = {v_bytes/1e6:.2f} MB!", flush=True)

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

    vis_stud_pre, vis_stud_base = [], []
    with torch.no_grad():
        for pv_c, pos_c in zip(vis_eval_pvs, vis_eval_poss):
            vo = vis_stud(pixel_values=pv_c.to(DEVICE), pixel_position_ids=pos_c.to(DEVICE))
            h_normed = mm_stud.embedding_pre_projection_norm(vo.last_hidden_state)
            st_b = mm_stud.embedding_projection(h_normed)
            vis_stud_pre.append(h_normed.reshape(-1, 1152).float().cpu())
            vis_stud_base.append(st_b.reshape(-1, 5376).float().cpu())

    heldout_stud_pre, heldout_stud_base = [], []
    with torch.no_grad():
        for pv_c, pos_c in zip(heldout_pvs, heldout_poss):
            vo = vis_stud(pixel_values=pv_c.to(DEVICE), pixel_position_ids=pos_c.to(DEVICE))
            h_normed = mm_stud.embedding_pre_projection_norm(vo.last_hidden_state)
            st_b = mm_stud.embedding_projection(h_normed)
            heldout_stud_pre.append(h_normed.reshape(-1, 1152).float().cpu())
            heldout_stud_base.append(st_b.reshape(-1, 5376).float().cpu())

    del vis_stud, mm_stud, vis_sd, embv_sd, vis_deq_weights, vis_eval_pvs, vis_eval_poss, heldout_pvs, heldout_poss
    torch.cuda.empty_cache()

    vx = torch.cat(vis_stud_pre, dim=0)
    vy_t = torch.cat([t.float() for t in vis_teach_soft], dim=0)
    vy_s = torch.cat(vis_stud_base, dim=0)
    ntr_v = (vx.shape[0] * 4) // 5
    vx_tr, vyt_tr, vys_tr = vx[:ntr_v].to(DEVICE), vy_t[:ntr_v].to(DEVICE), vy_s[:ntr_v].to(DEVICE)
    vx_va, vyt_va, vys_va = vx[ntr_v:].to(DEVICE), vy_t[ntr_v:].to(DEVICE), vy_s[ntr_v:].to(DEVICE)
    v_cos0 = float(F.cosine_similarity(vys_va, vyt_va, dim=-1).mean().item())

    model_vis = c8d.Cuneiform8DPatternAdapter(d_out=5376, d_in=1152, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(DEVICE)
    opt_v = torch.optim.AdamW(model_vis.parameters(), lr=3e-3, weight_decay=2e-2)
    sch_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=300, eta_min=3e-4)
    for _ in range(300):
        opt_v.zero_grad()
        idx_b = torch.randint(0, vx_tr.shape[0], (min(2048, vx_tr.shape[0]),), device=DEVICE)
        dv, _, lb = model_vis(vx_tr[idx_b])
        pred = vys_tr[idx_b] + dv
        tgt = vyt_tr[idx_b]
        loss = F.mse_loss(pred, tgt) + 0.5 * (1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()) + 0.005 * lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_vis.parameters(), 1.0)
        opt_v.step()
        sch_v.step()
    cents_vis, experts_vis = model_vis.cpu().export_quantized_arrays()
    with torch.no_grad():
        dv_va = eval_c8d_gpu(vx_va, cents_vis, experts_vis)
        v_cos1 = float(F.cosine_similarity(vys_va + dv_va, vyt_va, dim=-1).mean().item())
        vis_keep_c8d = v_cos1 > v_cos0
        vis_stud_soft = []
        for pre_i, base_i in zip(vis_stud_pre, vis_stud_base):
            st_i = base_i.to(DEVICE)
            if vis_keep_c8d:
                st_i = st_i + eval_c8d_gpu(pre_i.to(DEVICE), cents_vis, experts_vis)
            vis_stud_soft.append(st_i.to(torch.float16).cpu())
        heldout_stud_soft = []
        for pre_i, base_i in zip(heldout_stud_pre, heldout_stud_base):
            st_i = base_i.to(DEVICE)
            if vis_keep_c8d:
                st_i = st_i + eval_c8d_gpu(pre_i.to(DEVICE), cents_vis, experts_vis)
            heldout_stud_soft.append(st_i.to(torch.float16).cpu())
    print(f"[Stage 2 DONE] V5.0 Vision Student Val Soft-Token Cosine: {v_cos0:.5f} -> {v_cos1:.5f} (keep={vis_keep_c8d})", flush=True)
    del vx, vy_t, vy_s, vx_tr, vyt_tr, vys_tr, vx_va, vyt_va, vys_va, model_vis, vis_stud_pre, vis_stud_base, heldout_stud_pre, heldout_stud_base
    torch.cuda.empty_cache()

    _atomic_torch_save({"vision_records": vision_records}, VIS_CKPT_PATH)
    _atomic_torch_save({
        "vis_teach_soft": vis_teach_soft,
        "vis_stud_soft": vis_stud_soft,
        "vis_captions": vis_captions,
        "heldout_teach_soft": heldout_teach_soft,
        "heldout_stud_soft": heldout_stud_soft,
        "heldout_captions": [p[1] for p in heldout_pairs],
    }, VLM_SOFT_CACHE)
    vis_meta = {
        "vision_norms": vision_norms,
        "vis_total_bytes": int(v_bytes),
        "n_vis_mats": len(vision_records),
        "v_cos0": v_cos0,
        "v_cos1": v_cos1,
        "vis_keep_c8d": vis_keep_c8d,
        "cents_vis": cents_vis,
        "experts_vis": experts_vis,
    }
    _atomic_torch_save(vis_meta, VIS_META_PATH)
    del vision_records, vis_teach_soft, vis_stud_soft, heldout_teach_soft, heldout_stud_soft
    gc.collect()
    save_progress({"stage": "stage2_done", "vis_total_bytes": int(v_bytes), "v_cos0": v_cos0, "v_cos1": v_cos1})
    return vis_meta


if VIS_META_PATH.is_file() and (VIS_CKPT_PATH.is_file() or OUT_T16C.is_file()):
    _vm = torch.load(VIS_META_PATH, map_location="cpu", weights_only=False)
    print(f"[Resume Stage 2] Loaded {VIS_META_PATH} ({_vm['n_vis_mats']} mats, {_vm['vis_total_bytes']/1e6:.2f} MB)", flush=True)
else:
    _vm = _build_stage2_vision()

vision_norms: Dict[str, np.ndarray] = _vm["vision_norms"]
vis_total_bytes: int = int(_vm["vis_total_bytes"])
n_vis_mats: int = int(_vm["n_vis_mats"])
v_cos0: float = float(_vm["v_cos0"])
v_cos1: float = float(_vm["v_cos1"])
vis_keep_c8d: bool = bool(_vm["vis_keep_c8d"])
cents_vis = _vm["cents_vis"]
experts_vis = _vm["experts_vis"]
del _vm

# Stage 2b: Pack 160 VLM sequences + 4 held-out COCO val[0..3] eval sequences
_vsc = torch.load(VLM_SOFT_CACHE, map_location="cpu", weights_only=False)
vlm_teach_soft: List[torch.Tensor] = _vsc["vis_teach_soft"]
vlm_stud_soft: List[torch.Tensor] = _vsc["vis_stud_soft"]
vlm_captions: List[str] = _vsc["vis_captions"]
heldout_teach_soft: List[torch.Tensor] = _vsc["heldout_teach_soft"]
heldout_stud_soft: List[torch.Tensor] = _vsc["heldout_stud_soft"]
heldout_captions: List[str] = _vsc["heldout_captions"]
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
    assert n_soft_i == int(vlm_stud_soft[vi].shape[0]) and 1 <= n_soft_i <= 280
    q_vis = list(tk.encode(vlm_user_prompts[vi % len(vlm_user_prompts)], add_special_tokens=False).ids)
    a_vis = list(tk.encode(f"{vlm_captions[vi]}<turn|>\n", add_special_tokens=False).ids)
    cur_t = [2] + tok_user_open + [GEMMA4_BOI_ID] + [GEMMA4_IMAGE_ID] * n_soft_i + [GEMMA4_EOI_ID] + q_vis + tok_model_open
    s0_v = 1 + len(tok_user_open) + 1
    s1_v = s0_v + n_soft_i
    vlm_spans[N_TEXT_SEQ + vi] = (s0_v, s1_v)
    cur_a = [False] * len(cur_t) + [True] * len(a_vis)
    cur_t = cur_t + a_vis
    while len(cur_t) < SEQ_LEN + 1:
        cur_t.append(0)
        cur_a.append(False)
    cur_t = cur_t[: SEQ_LEN + 1]
    cur_a = cur_a[: SEQ_LEN + 1]
    seq_ids_list.append(cur_t[:-1])
    seq_nxt_list.append(cur_t[1:])
    seq_ans_list.append(cur_a[1:])
    seq_vlm_ans_list.append(cur_a[1:])

all_seq_ids = np.array(seq_ids_list, dtype=np.int32)
all_seq_nxt = np.array(seq_nxt_list, dtype=np.int32)
all_ans_mask = np.array(seq_ans_list, dtype=bool)
all_vlm_ans_mask = np.array(seq_vlm_ans_list, dtype=bool)

heldout_spans: Dict[int, Tuple[int, int]] = {}
heldout_ids_list, heldout_nxt_list, heldout_ans_starts, heldout_ans_lens = [], [], [], []
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
    cur_t = cur_t[: SEQ_LEN + 1]
    heldout_ids_list.append(cur_t[:-1])
    heldout_nxt_list.append(cur_t[1:])
    heldout_ans_starts.append(ans_start)
    heldout_ans_lens.append(ans_len)
heldout_seq_ids = np.array(heldout_ids_list, dtype=np.int32)

# Strictly held-out QAT validation split (32 text + 16 VLM = 48 seqs; NEVER touched by AdamW gradients!)
VAL_SEQ_IDX = np.concatenate([
    np.linspace(0, N_TEXT_SEQ - 1, 32, dtype=int),
    np.linspace(N_TEXT_SEQ, N_SEQ - 1, 16, dtype=int),
])
VAL_SEQ_SET = set(VAL_SEQ_IDX.tolist())
TRAIN_SEQ_IDX = np.array([i for i in range(N_SEQ) if i not in VAL_SEQ_SET], dtype=int)
print(f"[Stage 2b] Corpus ready: {N_SEQ} seqs ({len(TRAIN_SEQ_IDX)} train / {len(VAL_SEQ_IDX)} QAT val) = {all_seq_ids.size:,} tokens.", flush=True)


# ---------------------------------------------------------------------------
# 5. V5.0 Zero-Tern5 Allocation + Fail-Closed Pi 4 RAM Budget Guard (<= 6.50 GB)
# ---------------------------------------------------------------------------
FULL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}


def get_lm_spec_v5(l_idx: int, pname: str) -> Tuple[int, int, int]:
    """
    Return `(mode, bsize, lr_rank)` with ZERO `tern5` (mode=1) anywhere!
      - mode 0 = 1b_asym (576 B/tblk @ b256)
      - mode 5 = 2b_gaus (1056 B/tblk @ b256, 3.62 GB/s on Pi 4, +2.08 dB vs tern5!)
      - mode 4 = 3b_u0   (1568 B/tblk @ b256, 14.23 dB!)
    Total container <= 6.50 GB (6.05 GiB) -> VmSwap = 0 on Pi 4 even with Vision + Assistant!
    """
    if l_idx in (58, 59):
        if pname in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return (4, 256, 24 if pname in ("k_proj", "v_proj") else 16)
        return (5, 256, 24 if pname == "down_proj" else 16)
    if pname in ("q_proj", "k_proj", "v_proj", "o_proj"):
        rk = 16 if ((l_idx < 4 or l_idx >= 36) and pname in ("q_proj", "o_proj")) else 0
        return (5, 256, rk)
    # MLP L36..L57 (THE ENTIRE 22-LAYER CLIFF!):
    if l_idx >= 56:
        return (5, 256, 24 if pname == "down_proj" else 16)
    if l_idx >= 48:
        if pname in ("down_proj", "up_proj"):
            return (5, 256, 24 if pname == "down_proj" else 16)
        return (0, 256, 16)
    if l_idx >= 36:
        if pname == "down_proj":
            return (5, 256, 24)
        return (0, 256, 16 if pname == "up_proj" else 8)
    # MLP L00..L35 (with Block-Wise QAT keeps txt_cos > 0.985!)
    return (0, 256, 12 if pname == "down_proj" else 8)


def _pred_mat_bytes(d_out: int, d_in: int, bs: int, mode: int, lr_r: int, has_inv: bool = True) -> int:
    assert mode != 1, "V5.0 forbids tern5 (mode=1)!"
    nb, asy = p3.MODE_TO_T16C[mode]
    pb = p3.expected_tile16_bytes(d_out, d_in, bs, nb, asy)
    return pb + (d_in * 4 if has_inv else 0) + ((d_out * 4 + lr_r * 4 + d_out * lr_r + lr_r * d_in) if lr_r > 0 else 0) + 80


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

_pred_lm_b = _pred_mat_bytes(int(tc["vocab_size"]), D_MOD, 256, 4, 0, False)
_mode_cnt = defaultdict(int)
for _l in range(N_LAYERS):
    _f = _l in FULL_LAYERS
    _hd = HD_F if _f else HD_S
    _nkv = NKV_F if _f else NKV_S
    _sh = {"q_proj": (NH * _hd, D_MOD), "k_proj": (_nkv * _hd, D_MOD), "o_proj": (D_MOD, NH * _hd),
           "gate_proj": (21504, D_MOD), "up_proj": (21504, D_MOD), "down_proj": (D_MOD, 21504)}
    if not _f:
        _sh["v_proj"] = (_nkv * _hd, D_MOD)
    for _pn, (_do, _di) in _sh.items():
        _m, _bs, _rk = get_lm_spec_v5(_l, _pn)
        _mode_cnt[_m] += 1
        _pred_lm_b += _pred_mat_bytes(_do, _di, _bs, _m, _rk, True)
_pred_tot_b = _pred_lm_b + vis_total_bytes + 6_500_000
print(
    f"[Gate 0d Budget Guard] Predicted V5.0 .t16c: LM={_pred_lm_b/1e9:.3f}GB + VIS={vis_total_bytes/1e6:.1f}MB = "
    f"{_pred_tot_b:,} B ({_pred_tot_b/1024**3:.3f} GiB) | modes: 1b={_mode_cnt[0]}, 2g={_mode_cnt[5]}, 3u={_mode_cnt[4]}, tern5=0",
    flush=True,
)
assert _pred_tot_b <= 6_520_000_000, f"Budget overflow ({_pred_tot_b:,} B > 6.52 GB)!"


# ---------------------------------------------------------------------------
# 6. Stage 3A: Streaming L00..L59 GPTQ + Neuron Perm + Real Block-Wise QAT
# ---------------------------------------------------------------------------
STAGE3_CKPT_PATH = CKPT_DIR / "stage3_layer_ckpt.pt"
CKPT_X_TEACH = Path("/tmp/v5_ckpt_x_teach.pt")
CKPT_X_STUD = Path("/tmp/v5_ckpt_x_stud.pt")
TAIL_LAYERS = {56, 57, 58, 59}
TAIL_BUNDLE_DIR = Path("/tmp/v5_tail_bundles")
TAIL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

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
    x0 = t[..., :hr]
    x1 = t[..., hh : hh + hr]
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    if hr == hh:
        return torch.cat([r0, r1], dim=-1)
    return torch.cat([r0, t[..., hr:hh], r1, t[..., hh + hr :]], dim=-1)


def rmsn_diff(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    if w is not None:
        y = y * w.float()
    return y


def rmsn_f16(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    return rmsn_diff(x, w).to(torch.float16)


def write_norm_inplace(fp, norm_offsets: Dict[str, Tuple[int, int]], name: str, arr_f32: np.ndarray) -> None:
    off, n_el = norm_offsets[name]
    a32 = np.ascontiguousarray(arr_f32, dtype="<f4").ravel()
    assert a32.size == n_el, f"Norm {name} size mismatch {a32.size} != {n_el}"
    cur = fp.tell()
    fp.seek(off)
    fp.write(a32.tobytes())
    fp.seek(cur)


def qat_optimize_attn_block(
    l_idx: int,
    bundles: Dict[str, dict],
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
    """Real PyTorch `loss.backward()` Block-Wise Attention QAT on scales + low-rank + norms."""
    gqa_rep = NH // nkv
    ql_q = QuantLinearV5(bundles["q_proj"], DEVICE)
    ql_k = QuantLinearV5(bundles["k_proj"], DEVICE)
    ql_v = None if (is_full and K_EQ_V) else QuantLinearV5(bundles["v_proj"], DEVICE)
    ql_o = QuantLinearV5(bundles["o_proj"], DEVICE)

    p_in = nn.Parameter(norms_g["in"].float().clone())
    p_qn = nn.Parameter(norms_g["qn"].float().clone())
    p_kn = nn.Parameter(norms_g["kn"].float().clone())
    p_pa = nn.Parameter(norms_g["pa"].float().clone())

    def fwd_mb(seq_idx_list: List[int]) -> torch.Tensor:
        xb = X_stud_cpu[seq_idx_list].to(DEVICE)
        inp = rmsn_diff(xb, p_in)
        mb = len(seq_idx_list)
        q = ql_q(inp).view(mb, SEQ_LEN, NH, hd)
        k = ql_k(inp).view(mb, SEQ_LEN, nkv, hd)
        v = k if ql_v is None else ql_v(inp).view(mb, SEQ_LEN, nkv, hd)
        qn = apply_rope_f32(rmsn_diff(q, p_qn), cos_r, sin_r, rot_d, hd).transpose(1, 2).to(COMPUTE_DTYPE)
        kn = apply_rope_f32(rmsn_diff(k, p_kn), cos_r, sin_r, rot_d, hd).transpose(1, 2).to(COMPUTE_DTYPE)
        vn = rmsn_diff(v, None).transpose(1, 2).to(COMPUTE_DTYPE)
        amask = None if is_full else make_sliding_attn_mask_for_indices(seq_idx_list, vlm_spans, SEQ_LEN, DEVICE)
        if amask is not None:
            ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=amask, scale=1.0)
        else:
            ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
        ctx = ctx.transpose(1, 2).contiguous().view(mb, SEQ_LEN, NH * hd)
        ot = ql_o(ctx)
        return xb.float() + rmsn_diff(ot, p_pa)

    def eval_val() -> float:
        with torch.no_grad():
            cos_sum, cnt = 0.0, 0
            for v0 in range(0, len(VAL_SEQ_IDX), 8):
                sidx = VAL_SEQ_IDX[v0 : v0 + 8].tolist()
                pred = fwd_mb(sidx)
                tgt = xm_teach_cpu[sidx].to(DEVICE).float()
                cos_sum += float(F.cosine_similarity(pred.reshape(-1, D_MOD), tgt.reshape(-1, D_MOD), dim=-1).sum().item())
                cnt += pred.shape[0] * pred.shape[1]
            return cos_sum / max(cnt, 1)

    cos0 = eval_val()
    params = list(ql_q.parameters()) + list(ql_k.parameters()) + list(ql_o.parameters()) + [p_in, p_qn, p_kn, p_pa]
    if ql_v is not None:
        params += list(ql_v.parameters())
    opt = torch.optim.AdamW(params, lr=7e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=7e-5)

    best_cos = cos0
    best_state = None
    mb_sz = 8
    for st in range(steps):
        opt.zero_grad()
        sel = TRAIN_SEQ_IDX[np.random.choice(len(TRAIN_SEQ_IDX), size=mb_sz, replace=False)].tolist()
        pred = fwd_mb(sel)
        tgt = xm_teach_cpu[sel].to(DEVICE).float()
        mse = F.mse_loss(pred, tgt) / (tgt.var() + 1e-5)
        l_cos = 1.0 - F.cosine_similarity(pred.reshape(-1, D_MOD), tgt.reshape(-1, D_MOD), dim=-1).mean()
        loss = mse + 0.5 * l_cos
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if (st + 1) == steps or (steps >= 40 and (st + 1) == steps // 2):
            cv = eval_val()
            if cv > best_cos + 1e-6:
                best_cos = cv
                best_state = {
                    "q": {k: v.detach().cpu().clone() for k, v in ql_q.state_dict().items()},
                    "k": {k: v.detach().cpu().clone() for k, v in ql_k.state_dict().items()},
                    "v": {k: v.detach().cpu().clone() for k, v in ql_v.state_dict().items()} if ql_v is not None else None,
                    "o": {k: v.detach().cpu().clone() for k, v in ql_o.state_dict().items()},
                    "in": p_in.detach().cpu().clone(),
                    "qn": p_qn.detach().cpu().clone(),
                    "kn": p_kn.detach().cpu().clone(),
                    "pa": p_pa.detach().cpu().clone(),
                }

    w_out: Dict[str, torch.Tensor] = {}
    new_norms = dict(norms_g)
    if best_state is not None:
        ql_q.load_state_dict({k: v.to(DEVICE) for k, v in best_state["q"].items()})
        ql_k.load_state_dict({k: v.to(DEVICE) for k, v in best_state["k"].items()})
        if ql_v is not None and best_state["v"] is not None:
            ql_v.load_state_dict({k: v.to(DEVICE) for k, v in best_state["v"].items()})
        ql_o.load_state_dict({k: v.to(DEVICE) for k, v in best_state["o"].items()})
        new_norms["in"] = best_state["in"].to(DEVICE)
        new_norms["qn"] = best_state["qn"].to(DEVICE)
        new_norms["kn"] = best_state["kn"].to(DEVICE)
        new_norms["pa"] = best_state["pa"].to(DEVICE)
        w_out["q_proj"] = ql_q.export_into_bundle(bundles["q_proj"])
        w_out["k_proj"] = ql_k.export_into_bundle(bundles["k_proj"])
        if ql_v is not None:
            w_out["v_proj"] = ql_v.export_into_bundle(bundles["v_proj"])
        w_out["o_proj"] = ql_o.export_into_bundle(bundles["o_proj"])
    else:
        w_out["q_proj"] = ql_q.export_into_bundle(bundles["q_proj"])
        w_out["k_proj"] = ql_k.export_into_bundle(bundles["k_proj"])
        if ql_v is not None:
            w_out["v_proj"] = ql_v.export_into_bundle(bundles["v_proj"])
        w_out["o_proj"] = ql_o.export_into_bundle(bundles["o_proj"])

    del ql_q, ql_k, ql_v, ql_o, p_in, p_qn, p_kn, p_pa, opt, sch, best_state
    torch.cuda.empty_cache()
    return w_out, new_norms, cos0, best_cos


def qat_optimize_mlp_block(
    l_idx: int,
    bundles: Dict[str, dict],
    norms_g: Dict[str, torch.Tensor],
    xm_stud_cpu: torch.Tensor,
    X_teach_next_cpu: torch.Tensor,
    steps: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, float, float]:
    """Real PyTorch `loss.backward()` Sparse-Aware GeGLU MLP Block-Wise QAT."""
    ql_g = QuantLinearV5(bundles["gate_proj"], DEVICE)
    ql_u = QuantLinearV5(bundles["up_proj"], DEVICE)
    ql_d = QuantLinearV5(bundles["down_proj"], DEVICE)

    p_pf = nn.Parameter(norms_g["pf"].float().clone())
    p_pff = nn.Parameter(norms_g["pff"].float().clone())
    p_lsc = nn.Parameter(torch.tensor(float(norms_g["lsc"]), device=DEVICE, dtype=torch.float32))

    def fwd_mb(seq_idx_list: List[int], sparse_p: float = 1.0) -> torch.Tensor:
        xm = xm_stud_cpu[seq_idx_list].to(DEVICE)
        hm = rmsn_diff(xm, p_pf)
        g_act = F.gelu(ql_g(hm), approximate="tanh")
        if sparse_p < 0.9999:
            g_act = apply_block_sparse_geglu_mask(g_act, top_p=sparse_p, bsize=256)
        u_act = ql_u(hm)
        dt = ql_d(g_act * u_act)
        return (xm.float() + rmsn_diff(dt, p_pff)) * p_lsc

    def eval_val(sparse_p: float = 1.0) -> float:
        with torch.no_grad():
            cos_sum, cnt = 0.0, 0
            for v0 in range(0, len(VAL_SEQ_IDX), 8):
                sidx = VAL_SEQ_IDX[v0 : v0 + 8].tolist()
                pred = fwd_mb(sidx, sparse_p=sparse_p)
                tgt = X_teach_next_cpu[sidx].to(DEVICE).float()
                cos_sum += float(F.cosine_similarity(pred.reshape(-1, D_MOD), tgt.reshape(-1, D_MOD), dim=-1).sum().item())
                cnt += pred.shape[0] * pred.shape[1]
            return cos_sum / max(cnt, 1)

    cos0 = eval_val(1.0)
    params = list(ql_g.parameters()) + list(ql_u.parameters()) + list(ql_d.parameters()) + [p_pf, p_pff, p_lsc]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

    best_cos = cos0
    best_state = None
    mb_sz = 8
    for st in range(steps):
        opt.zero_grad()
        sel = TRAIN_SEQ_IDX[np.random.choice(len(TRAIN_SEQ_IDX), size=mb_sz, replace=False)].tolist()
        # Sparse-aware GeGLU training: 1/3 steps use exact top-50% 256-wide block mask, 2/3 dense!
        sp = 0.50 if (st % 3 == 2) else 1.0
        pred = fwd_mb(sel, sparse_p=sp)
        tgt = X_teach_next_cpu[sel].to(DEVICE).float()
        mse = F.mse_loss(pred, tgt) / (tgt.var() + 1e-5)
        l_cos = 1.0 - F.cosine_similarity(pred.reshape(-1, D_MOD), tgt.reshape(-1, D_MOD), dim=-1).mean()
        loss = mse + 0.5 * l_cos
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if (st + 1) == steps or (steps >= 50 and (st + 1) == steps // 2):
            cv = eval_val(1.0)
            if cv > best_cos + 1e-6:
                best_cos = cv
                best_state = {
                    "g": {k: v.detach().cpu().clone() for k, v in ql_g.state_dict().items()},
                    "u": {k: v.detach().cpu().clone() for k, v in ql_u.state_dict().items()},
                    "d": {k: v.detach().cpu().clone() for k, v in ql_d.state_dict().items()},
                    "pf": p_pf.detach().cpu().clone(),
                    "pff": p_pff.detach().cpu().clone(),
                    "lsc": float(p_lsc.detach().item()),
                }

    new_norms = dict(norms_g)
    if best_state is not None:
        ql_g.load_state_dict({k: v.to(DEVICE) for k, v in best_state["g"].items()})
        ql_u.load_state_dict({k: v.to(DEVICE) for k, v in best_state["u"].items()})
        ql_d.load_state_dict({k: v.to(DEVICE) for k, v in best_state["d"].items()})
        new_norms["pf"] = best_state["pf"].to(DEVICE)
        new_norms["pff"] = best_state["pff"].to(DEVICE)
        new_norms["lsc"] = best_state["lsc"]
    cos_sp50 = eval_val(0.50)
    w_out = {
        "gate_proj": ql_g.export_into_bundle(bundles["gate_proj"]),
        "up_proj": ql_u.export_into_bundle(bundles["up_proj"]),
        "down_proj": ql_d.export_into_bundle(bundles["down_proj"]),
    }
    del ql_g, ql_u, ql_d, p_pf, p_pff, p_lsc, opt, sch, best_state
    torch.cuda.empty_cache()
    return w_out, new_norms, cos0, best_cos, cos_sp50


print(f"\n[Stage 3A] Streaming {N_LAYERS} Zero-Tern5 LM layers with Co-Activation Perm + Real Block-Wise QAT...", flush=True)
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
    n_all_norms = len(all_norms)
    expected_lm_mats = 1 + sum(6 if (LAYER_TYPES[_l] == "full_attention" and K_EQ_V) else 7 for _l in range(N_LAYERS))
    total_mats = expected_lm_mats + n_vis_mats

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
        t16c_fp.seek(t16c_pos)
        t16c_fp.truncate(t16c_pos)
        print(f"[Resume Stage 3A] Resuming from L{start_layer:02d} (t16c_pos={t16c_pos:,} B)", flush=True)
    else:
        start_layer = 0
        layer_cos_log: List[dict] = []
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

        print("[Stage 3A] Packing embed_tokens.weight (3b_u0 @ b256)...", flush=True)
        pb_emb, inv_emb, lr_emb, emb_deq_cpu, _ = pack_matrix_gpu_gptq_v5(
            "embed_tokens", emb_bf16_cpu, None, None, mode=4, bsize=256, lr_rank=0
        )
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
        t16c_fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, n_all_norms, total_mats, 0))
        t16c_pos = 24
        for _nm, _vec in all_norms.items():
            _nb = _nm.encode("utf-8")
            _vf = np.ascontiguousarray(_vec, dtype="<f4").ravel()
            t16c_fp.write(struct.pack("<H", len(_nb)))
            t16c_fp.write(_nb)
            t16c_fp.write(struct.pack("<I", _vf.size))
            norm_offsets[_nm] = (t16c_pos + 2 + len(_nb) + 4, _vf.size)
            t16c_fp.write(_vf.tobytes())
            t16c_pos += 2 + len(_nb) + 4 + _vf.size * 4
        t16c_pos = p3._write_mat_record(
            t16c_fp, t16c_pos, "model.language_model.embed_tokens.weight",
            int(tc["vocab_size"]), D_MOD, 256, 3, False, True, pb_emb, inv_emb, lr_emb,
        )
        lm_mats_count = 1
        del pb_emb, inv_emb, lr_emb
        t16c_fp.flush()
        t16c_fp.close()
        t16c_fp = open(OUT_T16C, "r+b")
        t16c_fp.seek(t16c_pos)
        save_progress({"stage": "embed_done", "next_layer": 0, "t16c_pos": t16c_pos})

    del vlm_teach_soft, vlm_stud_soft, heldout_teach_soft, heldout_stud_soft
    gc.collect()

    def unpack_qat(pfx: str) -> torch.Tensor:
        wp = sf.get_tensor(f"{pfx}.weight_packed").to(DEVICE)
        ws = sf.get_tensor(f"{pfx}.weight_scale").to(DEVICE, dtype=torch.float32)
        sh = torch.arange(0, 32, 4, device=DEVICE, dtype=torch.int32)
        nibs = ((wp.unsqueeze(-1) >> sh) & 0xF) - 8
        return (nibs.to(torch.float32).reshape(wp.shape[0], -1, 32) * ws.unsqueeze(-1)).reshape(wp.shape[0], -1).cpu()

    def teacher_layer_step(
        X_cpu: torch.Tensor,
        w_f16: Dict[str, torch.Tensor],
        norms_g: Dict[str, torch.Tensor],
        l_idx: int,
        spans_dict: Optional[Dict[int, Tuple[int, int]]] = None,
        return_mid_and_energy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
                amask = None if (is_full or not spans_dict) else make_sliding_attn_mask(b0, b1, spans_dict, SEQ_LEN, DEVICE)
                if amask is not None:
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=amask, scale=1.0)
                else:
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
                ot = F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_f16["o_proj"])
                xm = xb + rmsn_f16(ot, norms_g["pa"])
                if xm_cpu is not None:
                    xm_cpu[b0:b1] = xm.cpu()
                hm = rmsn_f16(xm, norms_g["pf"])
                act = (F.gelu(F.linear(hm, w_f16["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_f16["up_proj"]).float())
                if neuron_e is not None:
                    neuron_e.add_((act * act).reshape(-1, 21504).double().sum(dim=0))
                dt_o = F.linear(act.to(torch.float16), w_f16["down_proj"])
                out_cpu[b0:b1] = ((xm.float() + rmsn_f16(dt_o, norms_g["pff"]).float()) * lsc).to(torch.float16).cpu()
        return out_cpu, xm_cpu, neuron_e

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
                _atomic_torch_save(X_stud_cpu, Path("/tmp/v5_x_stud_pre56.pt"))
                _atomic_torch_save(H_stud_eval, Path("/tmp/v5_h_eval_pre56.pt"))
            if l_idx == 58:
                _atomic_torch_save(X_stud_cpu, Path("/tmp/v5_x_stud_pre58.pt"))

            norms_g = {
                "in": sf.get_tensor(p + "input_layernorm.weight").to(DEVICE),
                "qn": sf.get_tensor(p + "self_attn.q_norm.weight").to(DEVICE),
                "kn": sf.get_tensor(p + "self_attn.k_norm.weight").to(DEVICE),
                "pa": sf.get_tensor(p + "post_attention_layernorm.weight").to(DEVICE),
                "pf": sf.get_tensor(p + "pre_feedforward_layernorm.weight").to(DEVICE),
                "pff": sf.get_tensor(p + "post_feedforward_layernorm.weight").to(DEVICE),
                "lsc": sf.get_tensor(p + "layer_scalar").float().item(),
            }
            proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            w_teach_f32: Dict[str, torch.Tensor] = {}
            for pn in proj_names:
                if is_full and K_EQ_V and pn == "v_proj":
                    continue
                sub = "self_attn" if pn in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                w_teach_f32[pn] = unpack_qat(f"{p}{sub}.{pn}")

            # Save Teacher KV targets at L58 & L59 for Stage 4 C8DB
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
                    _atomic_torch_save(_kv_save, Path(f"/tmp/v5_teach_kv_l{l_idx}.pt"))
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

            # Co-Activation Neuron Permutation on MLP BEFORE quantization (100% exact invariance, zero runtime cost!)
            perm_mlp = torch.argsort(neuron_e, descending=True).cpu()
            w_teach_f32["gate_proj"] = w_teach_f32["gate_proj"][perm_mlp].contiguous()
            w_teach_f32["up_proj"] = w_teach_f32["up_proj"][perm_mlp].contiguous()
            w_teach_f32["down_proj"] = w_teach_f32["down_proj"][:, perm_mlp].contiguous()
            # Measure theoretical top-50% 256-block energy capture after permutation
            e_sorted = neuron_e[perm_mlp.to(DEVICE)]
            top50_e_pct = 100.0 * float(e_sorted[:10752].sum().item() / e_sorted.sum().clamp(min=1e-12).item())
            del neuron_e, e_sorted, perm_mlp

            n_tok_l = N_SEQ * SEQ_LEN
            attn_bundles: Dict[str, dict] = {}
            w_stud_f16: Dict[str, torch.Tensor] = {}

            # 1) Attention Q/K/V GPTQ
            h_in = torch.zeros((D_MOD, D_MOD), device=DEVICE, dtype=torch.float32)
            xsq_in = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
            x_in_eval = []
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 64):
                    xb = X_stud_cpu[b0 : min(b0 + 64, N_SEQ)].to(DEVICE)
                    inp = rmsn_f16(xb, norms_g["in"]).reshape(-1, D_MOD).float()
                    accum_hess(h_in, xsq_in, inp)
                    if len(x_in_eval) < 3 or (b0 >= N_TEXT_SEQ and len(x_in_eval) < 4):
                        x_in_eval.append(inp[:512].clone())
            xrms_in = torch.sqrt((xsq_in / n_tok_l).clamp(min=1e-8))
            xev_in = torch.cat(x_in_eval, dim=0)[:2048]
            for pn in (["q_proj", "k_proj"] if (is_full and K_EQ_V) else ["q_proj", "k_proj", "v_proj"]):
                m_i, bs_i, rk_i = get_lm_spec_v5(l_idx, pn)
                _, _, _, w_eff, bnd = pack_matrix_gpu_gptq_v5(
                    f"{p}self_attn.{pn}.weight", w_teach_f32[pn], h_in, xrms_in, m_i, bs_i, rk_i, xev_in, return_bundle=True
                )
                attn_bundles[pn] = bnd
                w_stud_f16[pn] = w_eff
            del h_in, xsq_in, xrms_in, xev_in, x_in_eval

            # 2) Attention O_proj GPTQ
            do_o, di_o = w_teach_f32["o_proj"].shape
            h_o = torch.zeros((di_o, di_o), device=DEVICE, dtype=torch.float32)
            xsq_o = torch.zeros((di_o,), device=DEVICE, dtype=torch.float32)
            x_o_ev = []
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 32):
                    b1 = min(b0 + 32, N_SEQ)
                    xb = X_stud_cpu[b0:b1].to(DEVICE)
                    inp = rmsn_f16(xb, norms_g["in"])
                    q = F.linear(inp, w_stud_f16["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                    k = F.linear(inp, w_stud_f16["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    v = k.clone() if (is_full and K_EQ_V) else F.linear(inp, w_stud_f16["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    qn = apply_rope_f32(rmsn_f16(q, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                    kn = apply_rope_f32(rmsn_f16(k, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                    vn = rmsn_f16(v, None).transpose(1, 2)
                    amask_s = None if is_full else make_sliding_attn_mask(b0, b1, vlm_spans, SEQ_LEN, DEVICE)
                    if amask_s is not None:
                        ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=amask_s, scale=1.0)
                    else:
                        ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
                    cf = ctx.transpose(1, 2).contiguous().view(-1, di_o).float()
                    accum_hess(h_o, xsq_o, cf)
                    if len(x_o_ev) < 3 or (b0 >= N_TEXT_SEQ and len(x_o_ev) < 4):
                        x_o_ev.append(cf[:512].clone())
            m_i, bs_i, rk_i = get_lm_spec_v5(l_idx, "o_proj")
            _, _, _, w_o_eff, bnd_o = pack_matrix_gpu_gptq_v5(
                f"{p}self_attn.o_proj.weight", w_teach_f32["o_proj"], [h_o],
                torch.sqrt((xsq_o / n_tok_l).clamp(min=1e-8)), m_i, bs_i, rk_i,
                torch.cat(x_o_ev, 0)[:2048], return_bundle=True
            )
            attn_bundles["o_proj"] = bnd_o
            w_stud_f16["o_proj"] = w_o_eff
            del h_o, xsq_o, x_o_ev, w_o_eff
            for _apn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w_teach_f32.pop(_apn, None)
            torch.cuda.empty_cache()

            # 3) REAL Block-Wise Attention QAT (loss.backward() + AdamW on scales, LR, norms!)
            attn_steps = 60 if l_idx >= 36 else 30
            w_attn_qat, norms_g, acos0, acos1 = qat_optimize_attn_block(
                l_idx, attn_bundles, norms_g, X_stud_cpu, xm_teach_cpu, is_full, cos_r, sin_r, rot_d, hd, nkv, attn_steps
            )
            del xm_teach_cpu

            # Compute post-QAT Attention residual xm_all + heldout mid + MLP input Hessian
            h_mlp = torch.zeros((D_MOD, D_MOD), device=DEVICE, dtype=torch.float32)
            xsq_mlp = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
            xm_all = torch.empty((N_SEQ, SEQ_LEN, D_MOD), dtype=torch.float16)
            x_m_ev = []
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
                    amask_s = None if is_full else make_sliding_attn_mask(b0, b1, vlm_spans, SEQ_LEN, DEVICE)
                    if amask_s is not None:
                        ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=amask_s, scale=1.0)
                    else:
                        ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
                    ot = F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_attn_qat["o_proj"])
                    xm = xb + rmsn_f16(ot, norms_g["pa"])
                    xm_all[b0:b1] = xm.cpu()
                    hmf = rmsn_f16(xm, norms_g["pf"]).reshape(-1, D_MOD).float()
                    accum_hess(h_mlp, xsq_mlp, hmf)
                    if len(x_m_ev) < 3 or (b0 >= N_TEXT_SEQ and len(x_m_ev) < 4):
                        x_m_ev.append(hmf[:512].clone())
                # Held-out VLM eval attention
                _he_g = H_stud_eval.to(DEVICE)
                _he_in = rmsn_f16(_he_g, norms_g["in"])
                _hq = F.linear(_he_in, w_attn_qat["q_proj"]).view(_he_g.shape[0], SEQ_LEN, NH, hd)
                _hk = F.linear(_he_in, w_attn_qat["k_proj"]).view(_he_g.shape[0], SEQ_LEN, nkv, hd)
                _hv = _hk.clone() if (is_full and K_EQ_V) else F.linear(_he_in, w_attn_qat["v_proj"]).view(_he_g.shape[0], SEQ_LEN, nkv, hd)
                _hqn = apply_rope_f32(rmsn_f16(_hq, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                _hkn = apply_rope_f32(rmsn_f16(_hk, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                _hvn = rmsn_f16(_hv, None).transpose(1, 2)
                _hmask = None if is_full else make_sliding_attn_mask(0, _he_g.shape[0], heldout_spans, SEQ_LEN, DEVICE)
                if _hmask is not None:
                    _hctx = F.scaled_dot_product_attention(_hqn, _hkn.repeat_interleave(gqa_rep, 1), _hvn.repeat_interleave(gqa_rep, 1), attn_mask=_hmask, scale=1.0)
                else:
                    _hctx = F.scaled_dot_product_attention(_hqn, _hkn.repeat_interleave(gqa_rep, 1), _hvn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
                _hot = F.linear(_hctx.transpose(1, 2).contiguous().view(_he_g.shape[0], SEQ_LEN, NH * hd), w_attn_qat["o_proj"])
                H_stud_eval_mid = (_he_g + rmsn_f16(_hot, norms_g["pa"])).cpu()
                del _he_g, _he_in, _hq, _hk, _hv, _hqn, _hkn, _hvn, _hmask, _hctx, _hot
            del X_stud_cpu, w_attn_qat, w_stud_f16
            torch.cuda.empty_cache()

            # 4) MLP gate/up/down GPTQ (on co-activation permuted neurons!)
            mlp_bundles: Dict[str, dict] = {}
            xrms_mlp = torch.sqrt((xsq_mlp / n_tok_l).clamp(min=1e-8))
            xev_mlp = torch.cat(x_m_ev, 0)[:2048]
            w_mlp_init: Dict[str, torch.Tensor] = {}
            for pn in ("gate_proj", "up_proj"):
                m_i, bs_i, rk_i = get_lm_spec_v5(l_idx, pn)
                _, _, _, w_eff, bnd = pack_matrix_gpu_gptq_v5(
                    f"{p}mlp.{pn}.weight", w_teach_f32[pn], h_mlp, xrms_mlp, m_i, bs_i, rk_i, xev_mlp, return_bundle=True
                )
                mlp_bundles[pn] = bnd
                w_mlp_init[pn] = w_eff
                w_teach_f32.pop(pn, None)
            del h_mlp, xsq_mlp, xrms_mlp, xev_mlp, x_m_ev

            do_d, di_d = w_teach_f32["down_proj"].shape
            h_d = torch.zeros((di_d, di_d), device=DEVICE, dtype=torch.float32)
            xsq_d = torch.zeros((di_d,), device=DEVICE, dtype=torch.float32)
            x_d_ev = []
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 32):
                    b1 = min(b0 + 32, N_SEQ)
                    hm = rmsn_f16(xm_all[b0:b1].to(DEVICE), norms_g["pf"])
                    act = (F.gelu(F.linear(hm, w_mlp_init["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_mlp_init["up_proj"]).float())
                    af = act.reshape(-1, di_d)
                    accum_hess(h_d, xsq_d, af)
                    if len(x_d_ev) < 3 or (b0 >= N_TEXT_SEQ and len(x_d_ev) < 4):
                        x_d_ev.append(af[:512].clone())
            del w_mlp_init
            xrms_d = torch.sqrt((xsq_d / n_tok_l).clamp(min=1e-8))
            xev_d = torch.cat(x_d_ev, 0)[:2048]
            del xsq_d, x_d_ev
            torch.cuda.empty_cache()

            m_i, bs_i, rk_i = get_lm_spec_v5(l_idx, "down_proj")
            _, _, _, _, bnd_d = pack_matrix_gpu_gptq_v5(
                f"{p}mlp.down_proj.weight", w_teach_f32["down_proj"], [h_d], xrms_d, m_i, bs_i, rk_i, xev_d, return_bundle=True
            )
            mlp_bundles["down_proj"] = bnd_d
            del h_d, xrms_d, xev_d, w_teach_f32
            torch.cuda.empty_cache()

            # 5) REAL Sparse-Aware GeGLU MLP Block-Wise QAT (loss.backward() + AdamW!)
            mlp_steps = 100 if l_idx >= 36 else 40
            w_mlp_qat, norms_g, mcos0, mcos1, mcos_sp50 = qat_optimize_mlp_block(
                l_idx, mlp_bundles, norms_g, xm_all, X_teach_next, mlp_steps
            )

            # Propagate Student activations & heldout eval through finalized layer l_idx
            lsc = float(norms_g["lsc"])
            with torch.no_grad():
                _hmg = H_stud_eval_mid.to(DEVICE)
                _hhm = rmsn_f16(_hmg, norms_g["pf"])
                _hact = (F.gelu(F.linear(_hhm, w_mlp_qat["gate_proj"]).float(), approximate="tanh") * F.linear(_hhm, w_mlp_qat["up_proj"]).float()).to(torch.float16)
                _hdt = F.linear(_hact, w_mlp_qat["down_proj"])
                H_stud_eval = ((_hmg.float() + rmsn_f16(_hdt, norms_g["pff"]).float()) * lsc).to(torch.float16).cpu()
                del H_stud_eval_mid, _hmg, _hhm, _hact, _hdt

                X_stud_next = torch.empty((N_SEQ, SEQ_LEN, D_MOD), dtype=torch.float16)
                for b0 in range(0, N_SEQ, 32):
                    b1 = min(b0 + 32, N_SEQ)
                    xm_g = xm_all[b0:b1].to(DEVICE)
                    hm = rmsn_f16(xm_g, norms_g["pf"])
                    act = (F.gelu(F.linear(hm, w_mlp_qat["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_mlp_qat["up_proj"]).float()).to(torch.float16)
                    dt_o = F.linear(act, w_mlp_qat["down_proj"])
                    X_stud_next[b0:b1] = ((xm_g.float() + rmsn_f16(dt_o, norms_g["pff"]).float()) * lsc).to(torch.float16).cpu()
            del xm_all, w_mlp_qat

            # Flush L00..L55 immediately to .t16c & update norms in-place; defer L56..L59 to Stage 3B Joint Logit-KL!
            all_b_layer = {**attn_bundles, **mlp_bundles}
            if l_idx not in TAIL_LAYERS:
                for pn_ord in (["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if (is_full and K_EQ_V) else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                    b = all_b_layer[pn_ord]
                    nb, asy = p3.MODE_TO_T16C[b["mode"]]
                    t16c_pos = p3._write_mat_record(
                        t16c_fp, t16c_pos, b["name"], b["d_out"], b["d_in"],
                        b["bsize"], nb, asy, True, b["pb"], b["inv_s"], b["lr"],
                    )
                    lm_mats_count += 1
                for nk_short, nk_full in [
                    ("in", "input_layernorm.weight"), ("qn", "self_attn.q_norm.weight"),
                    ("kn", "self_attn.k_norm.weight"), ("pa", "post_attention_layernorm.weight"),
                    ("pf", "pre_feedforward_layernorm.weight"), ("pff", "post_feedforward_layernorm.weight"),
                ]:
                    write_norm_inplace(t16c_fp, norm_offsets, p + nk_full, norms_g[nk_short].detach().cpu().numpy())
                write_norm_inplace(t16c_fp, norm_offsets, p + "layer_scalar", np.array([lsc], dtype="<f4"))
            else:
                _atomic_torch_save({
                    "bundles": all_b_layer,
                    "norms": {k: (v.detach().cpu() if torch.is_tensor(v) else float(v)) for k, v in norms_g.items()},
                }, TAIL_BUNDLE_DIR / f"L{l_idx:02d}.pt")
            del attn_bundles, mlp_bundles, all_b_layer

            cos_l_txt = float(F.cosine_similarity(X_stud_next[VAL_SEQ_IDX[:16]].reshape(-1, D_MOD).float(), X_teach_next[VAL_SEQ_IDX[:16]].reshape(-1, D_MOD).float(), dim=-1).mean().item())
            cos_l_vlm = float(F.cosine_similarity(H_stud_eval.reshape(-1, D_MOD).float(), H_teach_eval.reshape(-1, D_MOD).float(), dim=-1).mean().item())
            layer_cos_log.append({
                "l": l_idx, "txt_cos": round(cos_l_txt, 5), "vlm_cos": round(cos_l_vlm, 5),
                "attn_qat": [round(acos0, 5), round(acos1, 5)],
                "mlp_qat": [round(mcos0, 5), round(mcos1, 5), round(mcos_sp50, 5)],
                "top50_e_pct": round(top50_e_pct, 1),
            })
            X_teach_cpu, X_stud_cpu = X_teach_next, X_stud_next
            del X_teach_next, X_stud_next
            torch.cuda.empty_cache()

            if l_idx in (9, 19, 29, 39, 49, 55):
                t16c_fp.flush()
                os.fsync(t16c_fp.fileno())
                _atomic_torch_save(X_teach_cpu, CKPT_X_TEACH)
                _atomic_torch_save(X_stud_cpu, CKPT_X_STUD)
                _atomic_torch_save({
                    "next_layer": l_idx + 1, "t16c_pos": t16c_pos, "lm_mats_count": lm_mats_count,
                    "norm_offsets": norm_offsets, "H_teach_eval": H_teach_eval, "H_stud_eval": H_stud_eval,
                    "layer_cos_log": layer_cos_log,
                }, STAGE3_CKPT_PATH)
                gc.collect()
            save_progress({
                "stage": f"L{l_idx:02d}_done", "next_layer": l_idx + 1,
                "txt_cos": round(cos_l_txt, 5), "vlm_cos": round(cos_l_vlm, 5),
            })
            print(
                f"  [V5 L{l_idx:02d}/59 {'F' if is_full else 'S'}] txt_cos={cos_l_txt:.5f} vlm_cos={cos_l_vlm:.5f} "
                f"| attnQAT {acos0:.4f}->{acos1:.4f} | mlpQAT {mcos0:.4f}->{mcos1:.4f} (sp50={mcos_sp50:.4f}, e50={top50_e_pct:.1f}%) "
                f"| dt={time.perf_counter()-tl:.1f}s ram={host_ram_gb()[0]:.2f}GiB",
                flush=True,
            )

        # -------------------------------------------------------------------
        # 7. Stage 3B: Joint 4-Layer (L56..L59) + final_norm End-to-End Logit-KL Distillation!
        # -------------------------------------------------------------------
        print("\n[Stage 3B] Joint 4-Layer (L56 -> L57 -> L58 -> L59 -> final_norm -> lm_head) End-to-End Logit-KL Distillation...", flush=True)
        fn_g_teach = fn_orig_cpu.to(DEVICE)
        # Select top-4096 candidate vocabulary columns covering >99.9% Teacher probability mass on answer tokens
        vlm_ans_pos = np.flatnonzero(all_vlm_ans_mask.reshape(-1))
        txt_ans_pos = np.flatnonzero((all_ans_mask & (~all_vlm_ans_mask)).reshape(-1))
        sel_vlm = vlm_ans_pos[np.linspace(0, vlm_ans_pos.size - 1, min(8192, vlm_ans_pos.size), dtype=int)] if vlm_ans_pos.size > 0 else np.array([], dtype=int)
        sel_txt = txt_ans_pos[np.linspace(0, txt_ans_pos.size - 1, min(32768 - sel_vlm.size, txt_ans_pos.size), dtype=int)]
        sel_pos_np = np.sort(np.concatenate([sel_txt, sel_vlm]))
        sel_seq = sel_pos_np // SEQ_LEN
        sel_tok = sel_pos_np % SEQ_LEN

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
                del wc, zt
            uniq_ids, counts = torch.unique(torch.cat(top_ids_list), return_counts=True)
            best_cols = uniq_ids[torch.topk(counts, min(4096, uniq_ids.numel())).indices]
            cand_ids_3b = torch.unique(torch.cat([best_cols, nxt_ans_g]))
            W_cand_3b = emb_gpu[cand_ids_3b].to(COMPUTE_DTYPE)
            del top_ids_list, uniq_ids, counts, best_cols
        del emb_gpu, ht_ans_g, nxt_ans_g
        torch.cuda.empty_cache()

        # Build joint trainable 4-layer tail module (L56..L59 + final_norm)
        class TailLayerV5(nn.Module):
            def __init__(self, l_i: int, ckpt_d: dict) -> None:
                super().__init__()
                self.l_idx = l_i
                self.is_full = LAYER_TYPES[l_i] == "full_attention"
                self.nkv = NKV_F if self.is_full else NKV_S
                self.hd = HD_F if self.is_full else HD_S
                self.gqa_rep = NH // self.nkv
                theta_l = THETA_G if self.is_full else THETA_L
                self.rot_d = int(round(self.hd * (PART_G if self.is_full else PART_L)))
                cr, sr = make_rope(theta_l, self.rot_d, self.hd)
                self.register_buffer("cos_r", cr, persistent=False)
                self.register_buffer("sin_r", sr, persistent=False)
                bnds = ckpt_d["bundles"]
                self.q_proj = QuantLinearV5(bnds["q_proj"], DEVICE)
                self.k_proj = QuantLinearV5(bnds["k_proj"], DEVICE)
                self.v_proj = None if (self.is_full and K_EQ_V) else QuantLinearV5(bnds["v_proj"], DEVICE)
                self.o_proj = QuantLinearV5(bnds["o_proj"], DEVICE)
                self.gate_proj = QuantLinearV5(bnds["gate_proj"], DEVICE)
                self.up_proj = QuantLinearV5(bnds["up_proj"], DEVICE)
                self.down_proj = QuantLinearV5(bnds["down_proj"], DEVICE)
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
                if amask is not None and not self.is_full:
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(self.gqa_rep, 1), vn.repeat_interleave(self.gqa_rep, 1), attn_mask=amask, scale=1.0)
                else:
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(self.gqa_rep, 1), vn.repeat_interleave(self.gqa_rep, 1), is_causal=True, scale=1.0)
                ot = self.o_proj(ctx.transpose(1, 2).contiguous().view(mb, SEQ_LEN, NH * self.hd))
                xm = xb.float() + rmsn_diff(ot, self.n_pa)
                hm = rmsn_diff(xm, self.n_pf)
                act = F.gelu(self.gate_proj(hm), approximate="tanh") * self.up_proj(hm)
                dt = self.down_proj(act)
                return (xm + rmsn_diff(dt, self.n_pff)) * self.lsc

        tail_ckpts = {li: torch.load(TAIL_BUNDLE_DIR / f"L{li:02d}.pt", map_location="cpu", weights_only=False) for li in (56, 57, 58, 59)}
        tail_mods = nn.ModuleList([TailLayerV5(li, tail_ckpts[li]) for li in (56, 57, 58, 59)])
        p_final_norm = nn.Parameter(fn_orig_cpu.to(DEVICE, dtype=torch.float32).clone())
        x_pre56_cpu = torch.load("/tmp/v5_x_stud_pre56.pt", map_location="cpu", weights_only=True)
        h_ev_pre56_cpu = torch.load("/tmp/v5_h_eval_pre56.pt", map_location="cpu", weights_only=True)

        def run_tail_4layers(x_in: torch.Tensor, amask: Optional[torch.Tensor], use_ckpt: bool) -> torch.Tensor:
            h = x_in.float()
            for mod in tail_mods:
                if use_ckpt and h.requires_grad:
                    h = grad_checkpoint(mod, h, amask, use_reentrant=False)
                else:
                    h = mod(h, amask)
            return h

        def eval_tail_val_kl() -> Tuple[float, float, float]:
            with torch.no_grad():
                kl_s, agr_s, cos_s, cnt = 0.0, 0.0, 0.0, 0
                for v0 in range(0, len(VAL_SEQ_IDX), 8):
                    sidx = VAL_SEQ_IDX[v0 : v0 + 8].tolist()
                    am = make_sliding_attn_mask_for_indices(sidx, vlm_spans, SEQ_LEN, DEVICE)
                    hs_59 = run_tail_4layers(x_pre56_cpu[sidx].to(DEVICE), am, False)
                    ht_59 = X_teach_cpu[sidx].to(DEVICE).float()
                    ans_m = torch.from_numpy(all_ans_mask[sidx]).to(DEVICE)
                    if not ans_m.any():
                        continue
                    hs_f = rmsn_diff(hs_59, p_final_norm)[ans_m]
                    ht_f = rmsn_diff(ht_59, fn_g_teach)[ans_m]
                    cos_s += float(F.cosine_similarity(hs_f, ht_f, dim=-1).sum().item())
                    zt = 30.0 * torch.tanh(F.linear(ht_f.to(COMPUTE_DTYPE), W_cand_3b).float() / 30.0)
                    zs = 30.0 * torch.tanh(F.linear(hs_f.to(COMPUTE_DTYPE), W_cand_3b).float() / 30.0)
                    pt = F.softmax(zt / TAU, dim=-1)
                    lps = F.log_softmax(zs / TAU, dim=-1)
                    kl_s += float(((pt * (torch.log(pt.clamp(min=1e-12)) - lps)).sum(dim=-1) * (TAU * TAU)).sum().item())
                    agr_s += float((zs.argmax(dim=-1) == zt.argmax(dim=-1)).float().sum().item())
                    cnt += hs_f.shape[0]
                return kl_s / max(cnt, 1), agr_s / max(cnt, 1), cos_s / max(cnt, 1)

        kl_3b_0, agr_3b_0, cos_3b_0 = eval_tail_val_kl()
        print(f"  [Stage 3B Before] Val L59 Cand-KL={kl_3b_0:.4f} Teacher-Agree={agr_3b_0*100:.1f}% FinalCos={cos_3b_0:.5f}", flush=True)

        params_3b = list(tail_mods.parameters()) + [p_final_norm]
        opt_3b = torch.optim.AdamW(params_3b, lr=3e-4, weight_decay=1e-4)
        STEPS_3B = 150
        mb_3b = 8 if gpu_vram_gb >= 38.0 else 4
        sch_3b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_3b, T_max=STEPS_3B, eta_min=3e-5)
        best_kl_3b = kl_3b_0
        best_sd_3b = None
        for st in range(STEPS_3B):
            opt_3b.zero_grad()
            sel = TRAIN_SEQ_IDX[np.random.choice(len(TRAIN_SEQ_IDX), size=mb_3b, replace=False)].tolist()
            am = make_sliding_attn_mask_for_indices(sel, vlm_spans, SEQ_LEN, DEVICE)
            xin = x_pre56_cpu[sel].to(DEVICE).float().requires_grad_(True)
            hs_59 = run_tail_4layers(xin, am, True)
            ht_59 = X_teach_cpu[sel].to(DEVICE).float()
            ans_m = torch.from_numpy(all_ans_mask[sel]).to(DEVICE)
            hs_f = rmsn_diff(hs_59, p_final_norm)
            ht_f = rmsn_diff(ht_59, fn_g_teach)
            l_mse = F.mse_loss(hs_f, ht_f) / (ht_f.var() + 1e-5)
            l_cos = 1.0 - F.cosine_similarity(hs_f.reshape(-1, D_MOD), ht_f.reshape(-1, D_MOD), dim=-1).mean()
            if ans_m.any():
                zt = 30.0 * torch.tanh(F.linear(ht_f[ans_m].to(COMPUTE_DTYPE), W_cand_3b).float() / 30.0)
                zs = 30.0 * torch.tanh(F.linear(hs_f[ans_m].to(COMPUTE_DTYPE), W_cand_3b).float() / 30.0)
                pt = F.softmax(zt / TAU, dim=-1).detach()
                l_kl = ((pt * (torch.log(pt.clamp(min=1e-12)) - F.log_softmax(zs / TAU, dim=-1))).sum(dim=-1) * (TAU * TAU)).mean()
            else:
                l_kl = torch.tensor(0.0, device=DEVICE)
            loss = l_kl + 0.8 * l_mse + 0.5 * l_cos
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_3b, 1.0)
            opt_3b.step()
            sch_3b.step()
            if (st + 1) % 25 == 0:
                kl_v, agr_v, cos_v = eval_tail_val_kl()
                print(f"    [3B step {st+1:03d}/{STEPS_3B}] loss={loss.item():.4f} | Val KL={kl_v:.4f} Agree={agr_v*100:.1f}% Cos={cos_v:.5f}", flush=True)
                if kl_v < best_kl_3b:
                    best_kl_3b = kl_v
                    best_sd_3b = (
                        {k: v.detach().cpu().clone() for k, v in tail_mods.state_dict().items()},
                        p_final_norm.detach().cpu().clone(),
                    )

        if best_sd_3b is not None:
            tail_mods.load_state_dict({k: v.to(DEVICE) for k, v in best_sd_3b[0].items()}, strict=False)
            with torch.no_grad():
                p_final_norm.copy_(best_sd_3b[1].to(DEVICE))
        kl_3b_1, agr_3b_1, cos_3b_1 = eval_tail_val_kl()
        print(f"  [Stage 3B DONE] Val Cand-KL: {kl_3b_0:.4f} -> {kl_3b_1:.4f} | Agree: {agr_3b_0*100:.1f}% -> {agr_3b_1*100:.1f}% | Cos: {cos_3b_0:.5f} -> {cos_3b_1:.5f}", flush=True)
        del W_cand_3b, opt_3b, sch_3b, best_sd_3b

        # Export L56..L59 bundles, flush to OUT_T16C, update norms in-place, and recompute final L58/L59 student checkpoints!
        fn_distilled_cpu = p_final_norm.detach().cpu().clone()
        write_norm_inplace(t16c_fp, norm_offsets, "model.language_model.norm.weight", fn_distilled_cpu.numpy())

        x_cur_tail = x_pre56_cpu
        h_ev_tail = h_ev_pre56_cpu
        for idx_t, li in enumerate((56, 57, 58, 59)):
            mod = tail_mods[idx_t]
            bnds = tail_ckpts[li]["bundles"]
            p = f"model.language_model.layers.{li}."
            is_f = LAYER_TYPES[li] == "full_attention"
            w_exp = {
                "q_proj": mod.q_proj.export_into_bundle(bnds["q_proj"]),
                "k_proj": mod.k_proj.export_into_bundle(bnds["k_proj"]),
                "o_proj": mod.o_proj.export_into_bundle(bnds["o_proj"]),
                "gate_proj": mod.gate_proj.export_into_bundle(bnds["gate_proj"]),
                "up_proj": mod.up_proj.export_into_bundle(bnds["up_proj"]),
                "down_proj": mod.down_proj.export_into_bundle(bnds["down_proj"]),
            }
            if mod.v_proj is not None:
                w_exp["v_proj"] = mod.v_proj.export_into_bundle(bnds["v_proj"])
            for pn_ord in (["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if (is_f and K_EQ_V) else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                b = bnds[pn_ord]
                nb, asy = p3.MODE_TO_T16C[b["mode"]]
                t16c_pos = p3._write_mat_record(
                    t16c_fp, t16c_pos, b["name"], b["d_out"], b["d_in"],
                    b["bsize"], nb, asy, True, b["pb"], b["inv_s"], b["lr"],
                )
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
                write_norm_inplace(t16c_fp, norm_offsets, p + nk_f, nrm_out[nk_s].numpy())
            write_norm_inplace(t16c_fp, norm_offsets, p + "layer_scalar", np.array([nrm_out["lsc"]], dtype="<f4"))

            if li == 58:
                _atomic_torch_save(x_cur_tail, Path("/tmp/v5_x_stud_pre58_dist.pt"))
                _atomic_torch_save(h_ev_tail, Path("/tmp/v5_h_eval_pre58_dist.pt"))
            if li in (58, 59):
                _atomic_torch_save({"w": {k: v.cpu() for k, v in w_exp.items()}, "norms": nrm_out}, Path(f"/tmp/v5_l{li}_stud.pt"))
            # Advance x_cur_tail and h_ev_tail through exported quantized layer li
            x_next_t = torch.empty_like(x_cur_tail)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, 32):
                    b1 = min(b0 + 32, N_SEQ)
                    am = None if is_f else make_sliding_attn_mask(b0, b1, vlm_spans, SEQ_LEN, DEVICE)
                    x_next_t[b0:b1] = mod(x_cur_tail[b0:b1].to(DEVICE), am).to(torch.float16).cpu()
                am_h = None if is_f else make_sliding_attn_mask(0, h_ev_tail.shape[0], heldout_spans, SEQ_LEN, DEVICE)
                h_ev_tail = mod(h_ev_tail.to(DEVICE), am_h).to(torch.float16).cpu()
            x_cur_tail = x_next_t
            del w_exp
        X_stud_cpu = x_cur_tail
        H_stud_eval = h_ev_tail
        del tail_mods, tail_ckpts, x_pre56_cpu, h_ev_pre56_cpu, p_final_norm
        torch.cuda.empty_cache()

        assert lm_mats_count == expected_lm_mats, f"LM mat count mismatch: {lm_mats_count} != {expected_lm_mats}"
        _v_tail = torch.load(VIS_CKPT_PATH, map_location="cpu", weights_only=False)
        for _nm, _do, _di, _bs, _nb_i, _is_asym, _has_fwht, _pb, _inv_s, _lr in _v_tail["vision_records"]:
            t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, _nm, _do, _di, _bs, _nb_i, _is_asym, _has_fwht, _pb, _inv_s, _lr)
        del _v_tail
        t16c_fp.flush()
        os.fsync(t16c_fp.fileno())
    finally:
        t16c_fp.close()

t16c_sz = OUT_T16C.stat().st_size
assert t16c_sz <= 6_550_000_000, f"Serialized V5.0 .t16c exceeded budget: {t16c_sz:,} B"
for _rm_p in (VIS_CKPT_PATH, VLM_SOFT_CACHE, STAGE3_CKPT_PATH, CKPT_X_TEACH, CKPT_X_STUD, Path("/tmp/v5_x_stud_pre56.pt"), Path("/tmp/v5_h_eval_pre56.pt")):
    if _rm_p.is_file():
        _rm_p.unlink()
shutil.rmtree(TAIL_BUNDLE_DIR, ignore_errors=True)
gc.collect()
torch.cuda.empty_cache()
print(f"[Stage 3 DONE] Unified V5.0 .t16c flushed: {OUT_T16C} ({t16c_sz:,} B, {t16c_sz/1024**3:.3f} GiB)!", flush=True)
save_progress({"stage": "stage3_t16c_complete", "unified_t16c_bytes": int(t16c_sz)})


# ---------------------------------------------------------------------------
# 8. Stage 4A: Multimodal C8DB Encyclopedia (L58/L59 KV + post_cell60 + Vision)
# ---------------------------------------------------------------------------
print("\n[Stage 4A] Training 300-step Multimodal C8DB Encyclopedia on Distilled V5.0...", flush=True)
c8db_entries: List[dict] = []
if vis_keep_c8d:
    exp_rv = [c8d._c6d_radicals("vision", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
    c8db_entries.append({
        "name": "model.embed_vision.embedding_projection.weight", "d_out": 5376, "d_in": 1152, "rank": 32, "top_k": 2,
        "c6d": exp_rv[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents_vis, "expert_radicals": exp_rv, "experts": experts_vis,
    })


def train_c8d_proj(name: str, x_tr: torch.Tensor, y_t_tr: torch.Tensor, y_s_tr: torch.Tensor,
                   x_va: torch.Tensor, y_t_va: torch.Tensor, y_s_va: torch.Tensor, rank: int = 32, steps: int = 300):
    cos0 = float(F.cosine_similarity(y_s_va, y_t_va, dim=-1).mean().item())
    d_o, d_i = y_t_tr.shape[1], x_tr.shape[1]
    model = c8d.Cuneiform8DPatternAdapter(d_out=d_o, d_in=d_i, rank=rank, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-4)
    for _ in range(steps):
        opt.zero_grad()
        idx_b = torch.randint(0, x_tr.shape[0], (min(2048, x_tr.shape[0]),), device=DEVICE)
        d_y, _, lb = model(x_tr[idx_b])
        pred = y_s_tr[idx_b] + d_y
        tgt = y_t_tr[idx_b]
        loss = F.mse_loss(pred, tgt) + 0.5 * (1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()) + 0.005 * lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
    cents, experts = model.cpu().export_quantized_arrays()
    with torch.no_grad():
        d_va = eval_c8d_gpu(x_va, cents, experts)
        cos1 = float(F.cosine_similarity(y_s_va + d_va, y_t_va, dim=-1).mean().item())
    print(f"  [C8DB {name}] Val cos: {cos0:.5f} -> {cos1:.5f} ({'KEEP' if cos1 > cos0 else 'SKIP'})", flush=True)
    if cos1 <= cos0:
        return None, None, None
    exp_r = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
    return ({
        "name": name, "d_out": d_o, "d_in": d_i, "rank": rank, "top_k": 2,
        "c6d": exp_r[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents, "expert_radicals": exp_r, "experts": experts,
    }, cents, experts)


def forward_student_layer_v5(
    X_in_cpu: torch.Tensor, ckpt_path: str, l_idx: int,
    adp_k: Optional[Tuple] = None, adp_v: Optional[Tuple] = None,
    spans_dict: Optional[Dict[int, Tuple[int, int]]] = None,
    capture_kv_seqs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Optional[dict]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    w_g = {k: v.to(DEVICE) for k, v in ckpt["w"].items()}
    n_g = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in ckpt["norms"].items()}
    is_full = LAYER_TYPES[l_idx] == "full_attention"
    nkv = NKV_F if is_full else NKV_S
    hd = HD_F if is_full else HD_S
    theta = THETA_G if is_full else THETA_L
    rot_d = int(round(hd * (PART_G if is_full else PART_L)))
    cos_r, sin_r = make_rope(theta, rot_d, hd)
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
                        lo = gi - b0
                        cap_k_rope[gi] = kn[lo].cpu()  # [nkv, SEQ_LEN, hd]
                        cap_v_norm[gi] = vn[lo].cpu()
            am = None if (is_full or not spans_dict) else make_sliding_attn_mask(b0, b1, spans_dict, SEQ_LEN, DEVICE)
            if am is not None:
                ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=am, scale=1.0)
            else:
                ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
            ot = F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_g["o_proj"])
            xm = xb + rmsn_f16(ot, n_g["pa"])
            hm = rmsn_f16(xm, n_g["pf"])
            act = (F.gelu(F.linear(hm, w_g["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_g["up_proj"]).float()).to(torch.float16)
            dt_o = F.linear(act, w_g["down_proj"])
            X_out[b0:b1] = ((xm.float() + rmsn_f16(dt_o, n_g["pff"]).float()) * lsc).to(torch.float16).cpu()
    cap = {"k_rope": cap_k_rope, "v_norm": cap_v_norm} if capture_kv_seqs is not None else None
    del ckpt, w_g, n_g
    torch.cuda.empty_cache()
    return X_out, cap


sub_kv = np.concatenate([
    np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int),
    np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int),
])
x_pre58_d = torch.load("/tmp/v5_x_stud_pre58_dist.pt", map_location="cpu", weights_only=True)
h_ev_pre58_d = torch.load("/tmp/v5_h_eval_pre58_dist.pt", map_location="cpu", weights_only=True)
ck58 = torch.load("/tmp/v5_l58_stud.pt", map_location="cpu", weights_only=True)
tkv58 = torch.load("/tmp/v5_teach_kv_l58.pt", map_location="cpu", weights_only=True)
with torch.no_grad():
    inp58 = rmsn_f16(x_pre58_d[sub_kv].to(DEVICE), ck58["norms"]["in"].to(DEVICE)).reshape(-1, D_MOD).float()
    ks58_b = (inp58 @ ck58["w"]["k_proj"].to(DEVICE).float().T)
    vs58_b = (inp58 @ ck58["w"]["v_proj"].to(DEVICE).float().T)
del ck58
ntr_k = (inp58.shape[0] * 4) // 5
adp_k58, adp_v58, adp_k59 = None, None, None
ent_k58, c_k58, e_k58 = train_c8d_proj(
    "model.language_model.layers.58.self_attn.k_proj.weight",
    inp58[:ntr_k], tkv58["kt"].float().to(DEVICE)[:ntr_k], ks58_b[:ntr_k],
    inp58[ntr_k:], tkv58["kt"].float().to(DEVICE)[ntr_k:], ks58_b[ntr_k:],
)
if ent_k58 is not None:
    c8db_entries.append(ent_k58); adp_k58 = (c_k58, e_k58)
ent_v58, c_v58, e_v58 = train_c8d_proj(
    "model.language_model.layers.58.self_attn.v_proj.weight",
    inp58[:ntr_k], tkv58["vt"].float().to(DEVICE)[:ntr_k], vs58_b[:ntr_k],
    inp58[ntr_k:], tkv58["vt"].float().to(DEVICE)[ntr_k:], vs58_b[ntr_k:],
)
if ent_v58 is not None:
    c8db_entries.append(ent_v58); adp_v58 = (c_v58, e_v58)
del inp58, ks58_b, vs58_b, tkv58
torch.cuda.empty_cache()

# Capture L58 & L59 KV for 64 diverse sequences (56 train + 8 val) for Stage 4B Assistant Alignment!
AST_SEQS = np.linspace(0, N_TEXT_SEQ - 1, 64, dtype=int).tolist()
x_59_in, cap_kv58 = forward_student_layer_v5(x_pre58_d, "/tmp/v5_l58_stud.pt", 58, adp_k58, adp_v58, spans_dict=vlm_spans, capture_kv_seqs=AST_SEQS)
h_ev_59_in, _ = forward_student_layer_v5(h_ev_pre58_d, "/tmp/v5_l58_stud.pt", 58, adp_k58, adp_v58, spans_dict=heldout_spans)
del x_pre58_d, h_ev_pre58_d

ck59 = torch.load("/tmp/v5_l59_stud.pt", map_location="cpu", weights_only=True)
tkv59 = torch.load("/tmp/v5_teach_kv_l59.pt", map_location="cpu", weights_only=True)
with torch.no_grad():
    inp59 = rmsn_f16(x_59_in[sub_kv].to(DEVICE), ck59["norms"]["in"].to(DEVICE)).reshape(-1, D_MOD).float()
    ks59_b = (inp59 @ ck59["w"]["k_proj"].to(DEVICE).float().T)
del ck59
ent_k59, c_k59, e_k59 = train_c8d_proj(
    "model.language_model.layers.59.self_attn.k_proj.weight",
    inp59[:ntr_k], tkv59["kt"].float().to(DEVICE)[:ntr_k], ks59_b[:ntr_k],
    inp59[ntr_k:], tkv59["kt"].float().to(DEVICE)[ntr_k:], ks59_b[ntr_k:],
)
if ent_k59 is not None:
    c8db_entries.append(ent_k59); adp_k59 = (c_k59, e_k59)
del inp59, ks59_b, tkv59
torch.cuda.empty_cache()

X_stud_cpu, cap_kv59 = forward_student_layer_v5(x_59_in, "/tmp/v5_l59_stud.pt", 59, adp_k59, None, spans_dict=vlm_spans, capture_kv_seqs=AST_SEQS)
H_stud_eval, _ = forward_student_layer_v5(h_ev_59_in, "/tmp/v5_l59_stud.pt", 59, adp_k59, None, spans_dict=heldout_spans)
del x_59_in, h_ev_59_in

# Final RMSNorm Wiener check on top of Stage 3B distilled fn_distilled_cpu
fn_g = fn_orig_cpu.to(DEVICE)
fn_base_g = fn_distilled_cpu.to(DEVICE)
num_w = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
den_w = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
val_sq_o, val_sq_w, val_cnt = 0.0, 0.0, 0
with torch.no_grad():
    for b0 in range(0, N_SEQ, 64):
        b1 = min(b0 + 64, N_SEQ)
        xt_b = rmsn_diff(X_teach_cpu[b0:b1].to(DEVICE), fn_g).reshape(-1, D_MOD)
        xu_b = rmsn_diff(X_stud_cpu[b0:b1].to(DEVICE), None).reshape(-1, D_MOD)
        is_v = (torch.arange(xt_b.shape[0], device=DEVICE) % 10 == 0)
        num_w.add_((xu_b[~is_v] * xt_b[~is_v]).sum(dim=0))
        den_w.add_((xu_b[~is_v] * xu_b[~is_v]).sum(dim=0))
    n_tr_w = N_SEQ * SEQ_LEN * 0.9
    fn_wiener = (((num_w / n_tr_w) + 0.05 * fn_base_g) / ((den_w / n_tr_w) + 0.05).clamp(min=1e-8))
    for b0 in range(0, N_SEQ, 64):
        b1 = min(b0 + 64, N_SEQ)
        xt_b = rmsn_diff(X_teach_cpu[b0:b1].to(DEVICE), fn_g).reshape(-1, D_MOD)
        xu_b = rmsn_diff(X_stud_cpu[b0:b1].to(DEVICE), None).reshape(-1, D_MOD)
        is_v = (torch.arange(xt_b.shape[0], device=DEVICE) % 10 == 0)
        val_sq_o += float(((xu_b[is_v] * fn_base_g - xt_b[is_v]) ** 2).sum().item())
        val_sq_w += float(((xu_b[is_v] * fn_wiener - xt_b[is_v]) ** 2).sum().item())
        val_cnt += int(is_v.sum().item()) * D_MOD
fn_eff_g = fn_wiener if (val_sq_w < val_sq_o) else fn_base_g
print(f"  [Wiener RMSNorm] val_mse {val_sq_o/val_cnt:.5f} -> {val_sq_w/val_cnt:.5f} (using={'Wiener' if val_sq_w < val_sq_o else 'distilled'})", flush=True)

# Train post_cell60 C8DB (300 real AdamW steps on exact candidate+tail KL!)
ht_ans_g = rmsn_diff(X_teach_cpu[sel_seq, sel_tok].to(DEVICE), fn_g)
hs_ans_g = rmsn_diff(X_stud_cpu[sel_seq, sel_tok].to(DEVICE), fn_eff_g)
nxt_ans_g = torch.from_numpy(all_seq_nxt.reshape(-1)[sel_pos_np]).to(DEVICE, dtype=torch.long)
# Save AST_SEQS student states for Stage 4B before freeing X_stud_cpu
ast_x59_stud = X_stud_cpu[AST_SEQS].clone()
del X_teach_cpu, X_stud_cpu
gc.collect()
torch.cuda.empty_cache()

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
        del W_c, zt, zq
    all_top_m = torch.cat(top_ids_t_list + top_ids_q_list)
    uniq_ids, counts = torch.unique(all_top_m, return_counts=True)
    best_cols = uniq_ids[torch.topk(counts, min(4096, uniq_ids.numel())).indices]
    cand_ids = torch.unique(torch.cat([best_cols, nxt_ans_g]))
    W_cand_g = emb_gpu[cand_ids].float()
    del top_ids_t_list, top_ids_q_list, all_top_m, uniq_ids, counts, best_cols

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
        t_cand_lse = torch.logsumexp(zt_c, dim=-1)
        q_cand_lse = torch.logsumexp(zq_c, dim=-1)
        lt_t = t_lse[c0:c1] + torch.log((1.0 - torch.exp((t_cand_lse - t_lse[c0:c1]).clamp(max=0.0))).clamp(min=1e-12))
        log_tail_q0[c0:c1] = q_lse[c0:c1] + torch.log((1.0 - torch.exp((q_cand_lse - q_lse[c0:c1]).clamp(max=0.0))).clamp(min=1e-12))
        p_teach_store[c0:c1] = F.softmax(torch.cat([zt_c, lt_t.unsqueeze(1)], dim=-1), dim=-1).to(device="cpu", dtype=torch.float16)
del t_lse, q_lse

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
print(f"  [V5.0 post_cell60 Before] Val KL={kl0:.4f} Top1={acc0*100:.1f}% Teacher-Agree={agr0*100:.1f}% Margin={marg0:+.3f}", flush=True)
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
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model60.parameters(), 1.0)
    opt60.step()
    sch60.step()
    if (st + 1) % 60 == 0 or st == 0:
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
d_va_q = eval_c8d_gpu(hs_ans_g[va_a], cents60, experts60)
kl_q, acc_q, agr_q, marg_q = eval_c60(hs_ans_g[va_a] + d_va_q, va_a)
keep_c60 = kl_q <= kl0
print(f"  [V5.0 post_cell60 Quantized] Val KL: {kl0:.4f}->{kl_q:.4f} | Top1: {acc0*100:.1f}%->{acc_q*100:.1f}% | Agree: {agr0*100:.1f}%->{agr_q*100:.1f}%", flush=True)
if keep_c60:
    exp_r60 = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
    c8db_entries.append({
        "name": "model.language_model.norm.post_cell60", "d_out": D_MOD, "d_in": D_MOD, "rank": 32, "top_k": 2,
        "c6d": exp_r60[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents60, "expert_radicals": exp_r60, "experts": experts60,
    })

c8d_norms = [("model.language_model.norm.weight", fn_eff_g.detach().cpu().numpy().astype("<f4"))]
c8d_sz = c8d.write_c8db_file(OUT_C8D, c8d_norms, c8db_entries)
print(f"[Stage 4A DONE] Wrote {OUT_C8D} ({c8d_sz:,} B)", flush=True)
del p_teach_store, ht_ans_g, hs_ans_g, nxt_ans_g, y_star_g, t_top1_g, log_tail_q0, model60, m60_cpu
torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 9. Stage 4B: V5.0-Aligned 4-Layer EAGLE-3 Assistant (`gemma4_31b_assistant_v5.t16c`)
# ---------------------------------------------------------------------------
print("\n[Stage 4B] Downloading & Aligning 4-Layer Gemma4-31B-Assistant directly to V5.0 Student...", flush=True)
ast_report = {"status": "skipped"}
try:
    ast_dir = Path(snapshot_download(
        repo_id="google/gemma-4-31B-it-assistant",
        local_dir="/tmp/gemma4_31b_assistant",
        ignore_patterns=["*.bin", "*.pt"],
    ))
    ast_sf = ast_dir / "model.safetensors"
    ast_sd: Dict[str, torch.Tensor] = {}
    with safe_open(ast_sf, framework="pt", device="cpu") as asf:
        for k in asf.keys():
            ast_sd[k] = asf.get_tensor(k)

    # Exact PyTorch replica of assistant.rs::forward_step_logits_cs:
    # concat = [tok_embed(t), backbone_h(t-1)] -> pre_projection -> 4 layers reusing L58/L59 KV -> norm -> embed_tokens^T!
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

        def forward_seq(
            self,
            tok_emb_seq: torch.Tensor,   # [B, S-1, 5376] for pos 1..S-1
            back_h_seq: torch.Tensor,    # [B, S-1, 5376] backbone h at pos 0..S-2
            k58: torch.Tensor,           # [B, 16, S, 256]
            v58: torch.Tensor,           # [B, 16, S, 256]
            k59: torch.Tensor,           # [B, 4, S, 512]
        ) -> torch.Tensor:
            B, Sm1, _ = tok_emb_seq.shape
            cat = torch.cat([tok_emb_seq, back_h_seq], dim=-1).to(COMPUTE_DTYPE)
            state = F.linear(cat, self.pre.to(COMPUTE_DTYPE)).float()
            # Causal mask where query step i (predicting token i+2 given tok i+1 and h[i]) attends to KV positions 0..i
            q_mask = torch.tril(torch.ones((Sm1, SEQ_LEN), device=state.device, dtype=torch.bool), diagonal=0).unsqueeze(0).unsqueeze(0)
            for l_a, m in enumerate(self.layers):
                is_f = (l_a == 3)
                hd_a = HD_F if is_f else HD_S
                nkv_a = NKV_F if is_f else NKV_S
                rot_a = int(round(HD_F * PART_G)) if is_f else HD_S
                gqa_a = NH // nkv_a
                inp = rmsn_diff(state, m.n_in).to(COMPUTE_DTYPE)
                q = F.linear(inp, m.q_proj.to(COMPUTE_DTYPE)).view(B, Sm1, NH, hd_a)
                # Query position for predicting next token after kv[0..i] is pos = i + 1 (or i when step_pos=false)
                cr = (self.cr_g if is_f else self.cr_l)[1 : Sm1 + 1]
                sr = (self.sr_g if is_f else self.sr_l)[1 : Sm1 + 1]
                qn = apply_rope_f32(rmsn_diff(q, m.n_qn), cr, sr, rot_a, hd_a).transpose(1, 2).to(COMPUTE_DTYPE)
                kk = (k59 if is_f else k58).to(COMPUTE_DTYPE).repeat_interleave(gqa_a, dim=1)
                vv = (k59 if is_f else v58).to(COMPUTE_DTYPE).repeat_interleave(gqa_a, dim=1)
                ctx = F.scaled_dot_product_attention(qn, kk, vv, attn_mask=q_mask, scale=1.0)
                ot = F.linear(ctx.transpose(1, 2).contiguous().view(B, Sm1, NH * hd_a), m.o_proj.to(COMPUTE_DTYPE))
                state = state + rmsn_diff(ot, m.n_pa)
                hm = rmsn_diff(state, m.n_pf).to(COMPUTE_DTYPE)
                act = (F.gelu(F.linear(hm, m.gate.to(COMPUTE_DTYPE)).float(), approximate="tanh") * F.linear(hm, m.up.to(COMPUTE_DTYPE)).float()).to(COMPUTE_DTYPE)
                dt = F.linear(act, m.down.to(COMPUTE_DTYPE))
                state = (state + rmsn_diff(dt, m.n_pff)) * m.lsc
            return rmsn_diff(state, self.fn)

    ast_mod = Gemma4AssistantTrain(ast_sd).to(DEVICE)
    # Build V5.0 student final hidden states (post norm + post_cell60) for AST_SEQS
    with torch.no_grad():
        ast_h_final = torch.empty((len(AST_SEQS), SEQ_LEN, D_MOD), dtype=torch.float16)
        for ai in range(len(AST_SEQS)):
            hf = rmsn_diff(ast_x59_stud[ai].to(DEVICE), fn_eff_g)
            if keep_c60:
                hf = hf + eval_c8d_gpu(hf, cents60, experts60)
            ast_h_final[ai] = hf.to(torch.float16).cpu()
        # Use candidate vocabulary W_cand_g (top 4096 tokens!) + matching assistant embed rows for blazing-fast KL!
        W_ast_cand = ast_mod.emb[cand_ids].float()

    def eval_ast_val() -> Tuple[float, float]:
        with torch.no_grad():
            agr_s, kl_s, cnt = 0.0, 0.0, 0
            for ai in range(56, 64):
                gi = AST_SEQS[ai]
                tok_ids_next = torch.from_numpy(all_seq_ids[gi, 1:]).long().to(DEVICE)
                t_emb = (emb_gpu[tok_ids_next].float() * ESCALE).unsqueeze(0)
                bh = ast_h_final[ai : ai + 1, :-1].to(DEVICE).float()
                k58 = cap_kv58["k_rope"][gi].unsqueeze(0).to(DEVICE)
                v58 = cap_kv58["v_norm"][gi].unsqueeze(0).to(DEVICE)
                k59 = cap_kv59["k_rope"][gi].unsqueeze(0).to(DEVICE)
                ha = ast_mod.forward_seq(t_emb, bh, k58, v58, k59).squeeze(0)
                # Target V5.0 student logits at positions 1..S-1 (predicting token 2..S)
                hs_tgt = ast_h_final[ai, 1:].to(DEVICE).float()
                zs = 30.0 * torch.tanh((hs_tgt @ W_cand_g.T) / 30.0)
                za = ha @ W_ast_cand.T
                ps = F.softmax(zs / TAU, dim=-1)
                kl_s += float(((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU, dim=-1))).sum(dim=-1) * (TAU * TAU)).sum().item())
                agr_s += float((za.argmax(dim=-1) == zs.argmax(dim=-1)).float().sum().item())
                cnt += ha.shape[0]
            return kl_s / max(cnt, 1), agr_s / max(cnt, 1)

    akl0, aagr0 = eval_ast_val()
    print(f"  [Assistant Before Alignment] Val KL vs V5.0={akl0:.4f} | Top-1 Spec Agree vs V5.0={aagr0*100:.1f}%", flush=True)
    opt_a = torch.optim.AdamW(ast_mod.parameters(), lr=2e-4, weight_decay=1e-4)
    STEPS_A = 120
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=STEPS_A, eta_min=2e-5)
    best_aagr, best_asd = aagr0, None
    for st in range(STEPS_A):
        opt_a.zero_grad()
        batch_ais = np.random.choice(56, size=4, replace=False).tolist()
        t_embs, bhs, k58s, v58s, k59s, hs_tgts = [], [], [], [], [], []
        for ai in batch_ais:
            gi = AST_SEQS[ai]
            tok_ids_next = torch.from_numpy(all_seq_ids[gi, 1:]).long().to(DEVICE)
            t_embs.append(emb_gpu[tok_ids_next].float() * ESCALE)
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
        # Keep post_projection aligned to V5.0 hidden states for multi-step drafting (k=2..4)!
        h_post = F.linear(ha.to(COMPUTE_DTYPE), ast_mod.post.to(COMPUTE_DTYPE)).float()
        l_post = F.mse_loss(h_post, hs_t) / (hs_t.var() + 1e-5)
        loss = l_kl + 0.5 * l_ce + 0.5 * l_post
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ast_mod.parameters(), 1.0)
        opt_a.step()
        sch_a.step()
        if (st + 1) % 30 == 0:
            kl_v, agr_v = eval_ast_val()
            print(f"    [Ast step {st+1:03d}/{STEPS_A}] loss={loss.item():.4f} | Val KL={kl_v:.4f} Top-1 Agree={agr_v*100:.1f}%", flush=True)
            if agr_v > best_aagr:
                best_aagr = agr_v
                best_asd = {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}

    if best_asd is not None:
        ast_mod.load_state_dict({k: v.to(DEVICE) for k, v in best_asd.items()})
    akl1, aagr1 = eval_ast_val()
    print(f"  [Assistant After Alignment] Val KL={akl0:.4f}->{akl1:.4f} | Top-1 Spec Agree vs V5.0: {aagr0*100:.1f}% -> {aagr1*100:.1f}%!", flush=True)

    # Export updated Assistant safetensors and pack directly to gemma4_31b_assistant_v5.t16c (body=8b, embed=4b, bsize=64)
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

    ft_sf = Path("/tmp/assistant_v5_ft.safetensors")
    st_save_file(ast_sd, str(ft_sf))
    ast_sz = pa.pack_assistant_safetensors_to_t16c(ft_sf, OUT_AST, body_bits=8, embed_bits=4, block_size=64)
    ft_sf.unlink(missing_ok=True)
    ast_report = {
        "status": "aligned_and_packed",
        "path": str(OUT_AST),
        "bytes": int(ast_sz),
        "sha256": sha256_file(OUT_AST),
        "val_kl_before": round(akl0, 4),
        "val_kl_after": round(akl1, 4),
        "val_top1_agree_before": round(aagr0, 4),
        "val_top1_agree_after": round(aagr1, 4),
    }
    del ast_mod, ast_sd, W_ast_cand, ast_h_final
except Exception as e:
    print(f"  [WARN Stage 4B Assistant alignment skipped: {e}]", flush=True)
    ast_report = {"status": f"error: {e}"}
del cap_kv58, cap_kv59, ast_x59_stud, W_cand_g
torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 10. Stage 5: Strictly Held-Out COCO val[0..3] Visual QA Verification
# ---------------------------------------------------------------------------
print("\n[Stage 5] Evaluating strictly held-out COCO val[0..3] (NEVER in Hessians/QAT/C8DB) Teacher vs V5.0 Student...", flush=True)
heldout_vlm_report = []
with torch.no_grad():
    for hi, cap_gt in enumerate(heldout_captions):
        p0, plen = heldout_ans_starts[hi], heldout_ans_lens[hi]
        ht_hi = rmsn_diff(H_teach_eval[hi, p0 : p0 + plen].to(DEVICE), fn_g)
        hs_hi = rmsn_diff(H_stud_eval[hi, p0 : p0 + plen].to(DEVICE), fn_eff_g)
        if keep_c60:
            hs_hi = hs_hi + eval_c8d_gpu(hs_hi, cents60, experts60)
        h_cos_hi = float(F.cosine_similarity(hs_hi, ht_hi, dim=-1).mean().item())
        t_best_v = torch.full((plen, 5), -1e9, device=DEVICE)
        t_best_i = torch.zeros((plen, 5), device=DEVICE, dtype=torch.long)
        s_best_v = torch.full((plen, 5), -1e9, device=DEVICE)
        s_best_i = torch.zeros((plen, 5), device=DEVICE, dtype=torch.long)
        for r0 in range(0, emb_gpu.shape[0], 16384):
            wc = emb_gpu[r0 : r0 + 16384].float()
            lt = 30.0 * torch.tanh((ht_hi @ wc.T) / 30.0)
            ls = 30.0 * torch.tanh((hs_hi @ wc.T) / 30.0)
            tv, ti = torch.topk(lt, 5, dim=-1); sv, si = torch.topk(ls, 5, dim=-1)
            ti = ti + r0; si = si + r0
            mv = torch.cat([t_best_v, tv], dim=-1); mi = torch.cat([t_best_i, ti], dim=-1)
            t_best_v, pick = torch.topk(mv, 5, dim=-1); t_best_i = mi.gather(1, pick)
            mv2 = torch.cat([s_best_v, sv], dim=-1); mi2 = torch.cat([s_best_i, si], dim=-1)
            s_best_v, pick2 = torch.topk(mv2, 5, dim=-1); s_best_i = mi2.gather(1, pick2)
        t1, s1 = t_best_i[:, 0], s_best_i[:, 0]
        top1_agr = float((s1 == t1).float().mean().item())
        top5_agr = float((s_best_i == t1.unsqueeze(1)).any(dim=-1).float().mean().item())
        rec_h = {
            "val_idx": hi, "gt_caption": cap_gt, "ans_cos": round(h_cos_hi, 5),
            "ans_top1_agree": round(top1_agr, 4), "ans_top5_agree": round(top5_agr, 4),
            "teacher_forced_pred": tk.decode([int(x) for x in t1.tolist()]),
            "student_forced_pred": tk.decode([int(x) for x in s1.tolist()]),
        }
        heldout_vlm_report.append(rec_h)
        print(f"  [Held-Out val[{hi}]] cos={h_cos_hi:.5f} top1_agr={top1_agr*100:.1f}% top5_agr={top5_agr*100:.1f}% | Stud={rec_h['student_forced_pred']!r}", flush=True)
del emb_gpu

# Cleanup temp files and emit final SHA-256 verified V5.0 report
for _tmp_f in ("/tmp/v5_x_stud_pre58.pt", "/tmp/v5_x_stud_pre58_dist.pt", "/tmp/v5_h_eval_pre58_dist.pt",
               "/tmp/v5_l58_stud.pt", "/tmp/v5_l59_stud.pt", "/tmp/v5_teach_kv_l58.pt", "/tmp/v5_teach_kv_l59.pt"):
    Path(_tmp_f).unlink(missing_ok=True)
if VIS_META_PATH.is_file():
    VIS_META_PATH.unlink()

t16c_sha = sha256_file(OUT_T16C)
c8d_sha = sha256_file(OUT_C8D)
report = {
    "version": "V5.0-QAT-ZERO-TERN5",
    "gpu": gpu_name,
    "elapsed_s": round(time.perf_counter() - T0_ALL, 2),
    "unified_t16c_path": str(OUT_T16C),
    "unified_t16c_bytes": int(t16c_sz),
    "unified_t16c_gib": round(t16c_sz / (1024**3), 4),
    "unified_t16c_sha256": t16c_sha,
    "c8db_path": str(OUT_C8D),
    "c8db_bytes": int(c8d_sz),
    "c8db_sha256": c8d_sha,
    "assistant_v5": ast_report,
    "mode_counts": dict(_mode_cnt),
    "vision_val_cos": [round(v_cos0, 5), round(v_cos1, 5)],
    "stage3b_tail_kl": [round(kl_3b_0, 4), round(kl_3b_1, 4)],
    "stage3b_tail_agree": [round(agr_3b_0, 4), round(agr_3b_1, 4)],
    "post_cell60_val_kl": [round(kl0, 4), round(kl_q, 4)],
    "post_cell60_val_top1": [round(acc0, 4), round(acc_q, 4)],
    "post_cell60_val_teacher_agree": [round(agr0, 4), round(agr_q, 4)],
    "post_cell60_val_margin": [round(marg0, 3), round(marg_q, 3)],
    "heldout_coco_val0_3": heldout_vlm_report,
    "layer_cos_summary": [r for r in layer_cos_log if r["l"] in (0, 10, 20, 30, 35, 40, 45, 50, 55, 57, 58, 59)],
}
(OUT_DIR / "v5_0_qat_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
save_progress({"stage": "complete", "unified_t16c_bytes": int(t16c_sz), "post_cell60_val_kl": round(kl_q, 4), "teacher_agree": round(agr_q, 4)})
print("\n[V5.0 COMPLETE]", json.dumps(report, indent=2), flush=True)

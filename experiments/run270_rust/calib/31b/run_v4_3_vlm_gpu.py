#!/usr/bin/env python3
"""
Full-Model (31.273B: 60 LM Layers + 27 Vision Layers + Embedders) Full-Dataset
300-Step GPU Calibration, C8DB Pattern Bank Training, and Unified .t16c Assembler.
"""

import ctypes
import gc
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
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda:0")

# Locate private helper library dataset mounted at /kaggle/input/
lib_cands = list(Path("/kaggle/input").rglob("pack_t16c_v3.py"))
assert lib_cands, "Missing devs01/gemma4-31b-v4-calib-libs dataset!"
LIB_DIR = lib_cands[0].parent
sys.path.insert(0, str(LIB_DIR))

import pack_t16c_v3 as p3
import train_c8d_pattern_bank as c8d
import v4_corpus as vc
from safetensors import safe_open
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoConfig, Gemma4ImageProcessor
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4MultimodalEmbedder
from tokenizers import Tokenizer
from PIL import Image, ImageDraw

T0_ALL = time.perf_counter()

OUT_DIR = Path("/kaggle/working")
CKPT_DIR = OUT_DIR / "ckpt_v4_3_vlm"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_T16C = OUT_DIR / "gemma4_31b_v4_3_vlm.t16c"
OUT_C8D = OUT_DIR / "gemma4_31b_v4_3_vlm_c8d.c8d"
PROGRESS_JSON = CKPT_DIR / "progress.json"


def host_ram_gb() -> Tuple[float, float]:
    """Return `(used_gb, total_gb)` from Linux `/proc/meminfo` for container cgroup safety telemetry."""
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
    """Persist atomic JSON telemetry to `/kaggle/working/ckpt_v4_2/progress.json`."""
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


# Auto-restore checkpoint state if a prior run's `/kaggle/working` output is mounted under `/kaggle/input/*/ckpt_v4_3_vlm/`
if not PROGRESS_JSON.is_file():
    for _cand_p in sorted(Path("/kaggle/input").glob("*/ckpt_v4_3_vlm/progress.json")):
        _src_root = _cand_p.parent.parent
        print(f"[Resume] Found prior checkpoint under {_src_root}; staging into {OUT_DIR}...", flush=True)
        for _rel in ("ckpt_v4_3_vlm/progress.json", "ckpt_v4_3_vlm/stage2_vision.pt", "ckpt_v4_3_vlm/stage2_vlm_soft.pt",
                     "ckpt_v4_3_vlm/stage3_layer_ckpt.pt", "ckpt_v4_3_vlm/ckpt_x_teach.pt", "ckpt_v4_3_vlm/ckpt_x_stud.pt",
                     "ckpt_v4_3_vlm/stage4_l59_ans.pt", "gemma4_31b_v4_3_vlm_c8d.c8d", "gemma4_31b_v4_3_vlm.t16c"):
            _sf = _src_root / _rel
            _df = OUT_DIR / _rel
            if _sf.is_file() and not _df.is_file():
                _df.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_sf, _df)
        break

# ---------------------------------------------------------------------------
# 1. Compile portable OpenMP C quantizer/packer (libfast_quant.so) & verify
# ---------------------------------------------------------------------------
fq_c = LIB_DIR / "fast_quant.c"
fq_so = Path("/tmp/libfast_quant.so")
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

assert LIB_FQ.t16c_tblk_bytes(64, 1) == p3.tile16_blk_bytes(64, 5, False) == 240
assert LIB_FQ.t16c_tblk_bytes(64, 0) == p3.tile16_blk_bytes(64, 1, True) == 192
assert LIB_FQ.t16c_tblk_bytes(64, 4) == p3.tile16_blk_bytes(64, 3, False) == 416
print("[Gate 0] libfast_quant.so compiled and verified exact t16c_tblk_bytes parity.", flush=True)


def hadamard_gpu(n: int, device: torch.device) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    k = 1
    while k < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        k *= 2
    return h / math.sqrt(float(n))


H_GPU = {bs: hadamard_gpu(bs, DEVICE) for bs in (64, 128, 256)}
_H_GPU_CACHE: Dict[Tuple[int, str], torch.Tensor] = {(bs, str(DEVICE)): H_GPU[bs] for bs in (64, 128, 256)}


def get_h_gpu(bsize: int, device: torch.device) -> torch.Tensor:
    key = (bsize, str(device))
    h = _H_GPU_CACHE.get(key)
    if h is None:
        h = hadamard_gpu(bsize, device)
        _H_GPU_CACHE[key] = h
    return h


def symmetrize_inplace_(h: torch.Tensor, tile: int = 1024) -> torch.Tensor:
    """In-place 0.5*(H + H^T) using tiled scratch (~8 MB) instead of full H.T.clone() (1.85 GiB at d=21504)."""
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


def fwht_gpu(t: torch.Tensor, bsize: int) -> torch.Tensor:
    orig_sh = t.shape
    return (t.reshape(-1, bsize) @ get_h_gpu(bsize, t.device)).reshape(orig_sh)


def fwht_2sided_gpu(h_in: torch.Tensor, inv_s_gpu: Optional[torch.Tensor], bsize: int, consume: bool) -> torch.Tensor:
    """In-place/chunked (hb @ diag(inv_s) H diag(inv_s) @ hb) using ~176 MB scratch instead of 3.7 GiB einsum."""
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


def gptq_quant_gpu(W: torch.Tensor, U: torch.Tensor, bsize: int, mode: int) -> None:
    """Exact PyTorch CUDA equivalent of fast_quant.c::gptq_quant_chunk_f32 (~100x faster on T4!)."""
    rows, cw = W.shape
    nblks = cw // bsize
    diag_u = torch.diagonal(U).clamp(min=1e-12)
    E_blk = torch.empty((rows, bsize), device=W.device, dtype=torch.float32)
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
        elif mode == 1:
            sc = 1.25 * blk.abs().mean(dim=1)
            for _ in range(4):
                tau = 0.5 * sc.unsqueeze(1)
                pos = blk > tau
                neg = blk < -tau
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
    bsize: int = 64,
    lr_rank: int = 0,
    x_eval_gpu: Optional[torch.Tensor] = None,
    consume_h: bool = False,
    out_device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple], torch.Tensor]:
    """
    SmoothQuant -> Block-{64,128,256} FWHT -> GPU Hessian GPTQ + exact C tile16 pack -> optional Int8LowRank.
    Uses Woodbury / Dual Ridge when n_ev < d_in and tiled in-place symmetrization to eliminate 21504x21504 OOMs.
    """
    d_out, d_in = w_cpu.shape
    assert d_in % bsize == 0, f"d_in={d_in} not divisible by {bsize}"
    n_bits, is_asym = p3.MODE_TO_T16C[mode]
    exp_bytes = p3.expected_tile16_bytes(d_out, d_in, bsize, n_bits, is_asym)
    ret_dev = DEVICE if out_device is None else out_device

    # Chunked path for huge embedding matrices (e.g. 262144 x 5376) in BF16/FP32 without 5.6 GiB host bloat
    if h_gpu is None and lr_rank == 0 and d_out > 32768:
        pbytes = np.empty(exp_bytes, dtype=np.uint8)
        w_eff_cpu = torch.empty((d_out, d_in), dtype=torch.float16)
        tile_rows = 16384  # divisible by 16 for tile16 alignment
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
        return pbytes, None, None, w_eff_cpu

    # Offload wide d_in >= 16384 (down_proj 5376x21504) to clean GPU 1 when 2x T4s are available
    work_dev = torch.device("cuda:1") if (torch.cuda.device_count() > 1 and d_in >= 16384) else DEVICE
    w_gpu = w_cpu.to(work_dev, dtype=torch.float32)

    if x_rms_gpu is not None:
        xr_w = x_rms_gpu.to(work_dev, dtype=torch.float32)
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
                if h_t.device != work_dev:
                    h_moved = h_t.to(work_dev)
                    del h_t
                    h_t = h_moved
                    consume_flag = True
                h_rot = fwht_2sided_gpu(h_t, inv_s_gpu, bsize, consume=consume_flag)
                del h_t
                if not torch.isfinite(h_rot).all():
                    print(f"    [GPTQ fallback RTN {d_out}x{d_in}: non-finite h_rot]", flush=True)
                    del h_rot
                else:
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
                            del L
                            L = None
                            symmetrize_inplace_(h_inv)
                            U_chol_tmp = torch.linalg.cholesky(h_inv, upper=True)
                            del h_inv
                            h_inv = None
                            U_chol = U_chol_tmp if U_chol_tmp.is_contiguous() else U_chol_tmp.contiguous()
                            del U_chol_tmp
                            U_chol_tmp = None
                            gptq_quant_gpu(w_rot, U_chol, bsize, mode)
                        except Exception as e:
                            print(f"    [GPTQ fallback RTN {d_out}x{d_in}: {e}]", flush=True)
                        finally:
                            del L, h_inv, U_chol_tmp, U_chol
                    else:
                        print(f"    [GPTQ fallback RTN {d_out}x{d_in}: non-PD after 2.22x ridge]", flush=True)
        else:
            del h_t

    w_work_np = w_rot.detach().cpu().numpy().astype(np.float32, copy=True)
    del w_rot
    pbytes = np.empty(exp_bytes, dtype=np.uint8)
    LIB_FQ.quant_and_pack_t16c_f32(w_work_np.ctypes.data, d_out, d_in, bsize, mode, pbytes.ctypes.data)

    wq_rot_gpu = torch.from_numpy(w_work_np).to(work_dev, dtype=torch.float32)
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
            xv = x_eval_gpu[:n_ev].to(work_dev, dtype=torch.float32)
            y_err = (xv @ w_gpu.T) - (xv @ w_eff_gpu.T)  # [n_ev, d_out] without allocating dw [d_out, d_in]
            del w_gpu
            # Dual Gram [n_ev, n_ev] (16 MB!) when n_ev <= d_out avoids [d_out, d_out] cuSOLVER eigh OOM!
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

            # Exact Woodbury / Dual Ridge: (X^T X + reg I)^{-1} X^T P == X^T (X X^T + reg I)^{-1} P
            # Avoids [21504, 21504] (1.85 GiB + 3.7 GiB cuSOLVER workspace) on down_proj!
            reg = max(0.02 * float((xv * xv).sum().item()) / float(max(d_in, 1)), 1e-6)
            use_dual_ridge = n_ev < d_in
            if use_dual_ridge:
                gram_ridge = xv @ xv.T  # [n_ev, n_ev]
            else:
                gram_ridge = xv.T @ xv  # [d_in, d_in]
            gram_ridge.diagonal().add_(reg)

            def _solve_ridge(rhs_proj: torch.Tensor) -> torch.Tensor:
                if use_dual_ridge:
                    return xv.T @ torch.linalg.solve(gram_ridge, rhs_proj)
                return torch.linalg.solve(gram_ridge, xv.T @ rhs_proj)

            proj = y_err @ U_f  # [n_ev, rk]
            V_f = _solve_ridge(proj).T.contiguous()  # [rk, d_in]
            eye_lr = 1e-4 * torch.eye(lr_rank, device=work_dev)
            for _ in range(2):
                xv_v = xv @ V_f.T  # [n_ev, rk]
                g_v = xv_v.T @ xv_v + eye_lr
                U_f = torch.linalg.solve(g_v, xv_v.T @ y_err).T
                U_f = U_f / U_f.norm(dim=0, keepdim=True).clamp(min=1e-6)
                proj2 = y_err @ U_f
                V_f = _solve_ridge(proj2).T.contiguous()
            del y_err, gram_ridge, xv, eye_lr, proj, proj2, xv_v, g_v
            su = (U_f.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
            sv = (V_f.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
            u_i8 = torch.clamp(torch.round(U_f / su.unsqueeze(1)), -127, 127).to(torch.int8)
            v_i8 = torch.clamp(torch.round(V_f / sv.unsqueeze(1)), -127, 127).to(torch.int8)
            u_scaled = u_i8.float() * su.unsqueeze(1)
            v_scaled = v_i8.float() * sv.unsqueeze(1)
            if torch.isfinite(u_scaled).all() and torch.isfinite(v_scaled).all():
                w_eff_gpu.addmm_(u_scaled, v_scaled)
                low_rank = (
                    su.cpu().numpy().astype("<f4"),
                    sv.cpu().numpy().astype("<f4"),
                    u_i8.cpu().numpy().astype(np.int8),
                    v_i8.cpu().numpy().astype(np.int8),
                )
            del u_scaled, v_scaled, u_i8, v_i8, su, sv, U_f, V_f
    else:
        del w_gpu
    out_t = w_eff_gpu.to(device=ret_dev, dtype=torch.float16)
    del w_eff_gpu
    if work_dev != DEVICE:
        with torch.cuda.device(work_dev):
            torch.cuda.empty_cache()
    return pbytes, inv_s_np, low_rank, out_t


# Overflow-proof T4 Tensor Core Hessian accumulation: |xs| <= 2.0 across <= 8192 rows -> sum(xs^2) <= 32,768 < 65,504!
def accum_hess_t4(h_acc: torch.Tensor, xsq_acc: torch.Tensor, x_f32: torch.Tensor) -> None:
    xsq_acc.add_((x_f32 * x_f32).sum(dim=0))
    sc = (x_f32.abs().amax() / 2.0).clamp(min=1e-6)
    xs = (x_f32 / sc).to(torch.float16)
    h_acc.add_((xs.T @ xs).float() * (sc * sc))


# [Gate 0b] End-to-end synthetic verify BEFORE downloading 23 GB so any shape/API bug fails in <2s!
with torch.no_grad():
    for _m, _bs, _rk, _do, _di in [(4, 256, 24, 128, 256), (0, 256, 16, 128, 512), (4, 64, 16, 64, 128), (0, 256, 8, 4096, 512)]:
        _xt = torch.randn((320, _di), device=DEVICE, dtype=torch.float32)
        _wt = torch.randn((_do, _di), dtype=torch.float32)
        _hh = torch.zeros((_di, _di), device=DEVICE, dtype=torch.float32)
        _xq = torch.zeros((_di,), device=DEVICE, dtype=torch.float32)
        accum_hess_t4(_hh, _xq, _xt)
        _xr = torch.sqrt((_xq / 320.0).clamp(min=1e-8))
        # Assert fwht_2sided_gpu (including symmetrize_inplace_) matches full einsum reference identically
        _nb = _di // _bs; _hb = H_GPU[_bs]
        _ref_h = torch.einsum("ac,ijcd,db->ijab", _hb, _hh.reshape(_nb, _bs, _nb, _bs).permute(0, 2, 1, 3), _hb).permute(0, 2, 1, 3).reshape(_di, _di)
        _ref_h = 0.5 * (_ref_h + _ref_h.T)
        _got_h = fwht_2sided_gpu(_hh, None, _bs, consume=False)
        assert float((_got_h - _ref_h).abs().max().item()) < 1e-2 * float(_ref_h.abs().max().item() + 1.0)
        # Assert Woodbury / Dual Ridge matches Primal Ridge to FP32 tolerance when n_ev < d_in
        if _xt.shape[0] < _di:
            _reg_t = max(0.02 * float((_xt * _xt).sum().item()) / float(_di), 1e-6)
            _p_t = torch.randn((_xt.shape[0], _rk), device=DEVICE, dtype=torch.float32)
            _primal = torch.linalg.solve(_xt.T @ _xt + _reg_t * torch.eye(_di, device=DEVICE), _xt.T @ _p_t)
            _dual = _xt.T @ torch.linalg.solve(_xt @ _xt.T + _reg_t * torch.eye(_xt.shape[0], device=DEVICE), _p_t)
            assert float((_primal - _dual).abs().max().item()) < 1e-4 * float(_primal.abs().max().item() + 1.0)
            del _p_t, _primal, _dual
        # Assert gptq_quant_gpu matches C LIB_FQ.gptq_quant_chunk_f32 within rounding
        _ug = torch.eye(_di, device=DEVICE, dtype=torch.float32) * 0.5 + 0.01 * torch.triu(torch.ones((_di, _di), device=DEVICE), 1)
        _wg = _wt.to(DEVICE, dtype=torch.float32).clone()
        _wc = _wg.cpu().numpy().copy()
        _uc = _ug.cpu().numpy().copy()
        gptq_quant_gpu(_wg, _ug, _bs, _m)
        LIB_FQ.gptq_quant_chunk_f32(_wc.ctypes.data, _uc.ctypes.data, _do, _di, _bs, _m, 1)
        _diff = (_wg.cpu() - torch.from_numpy(_wc)).abs()
        assert float(_diff.median().item()) < 1e-5 and float((_diff > 1e-3).float().mean().item()) < 0.005, (
            f"GPTQ GPU/C parity mismatch: med={_diff.median().item()}, frac>1e-3={(_diff>1e-3).float().mean().item()}"
        )
        _hbox = [_hh]; del _hh
        _pb, _is, _lr, _we = pack_matrix_gpu_gptq(_wt, _hbox, _xr, mode=_m, bsize=_bs, lr_rank=_rk, x_eval_gpu=_xt)
        assert len(_hbox) == 0
        assert _lr is not None and _lr[0].shape == (_do,) and _lr[1].shape == (_rk,) and _lr[2].shape == (_do, _rk) and _lr[3].shape == (_rk, _di)
        assert _we.shape == (_do, _di) and torch.isfinite(_we).all()
        del _xt, _wt, _xq, _xr, _ref_h, _got_h, _ug, _wg, _wc, _uc, _diff, _hbox, _pb, _is, _lr, _we
print("[Gate 0b] pack_matrix_gpu_gptq + Woodbury Dual + fwht_2sided_gpu + gptq_quant_gpu == C parity verified.", flush=True)

# [Gate 0c] Full-scale down_proj (5376 x 21504, b256, r=16, n_ev=2048) peak-VRAM stress test BEFORE 23 GB download!
with torch.no_grad():
    for _d_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(_d_idx)
    _do_c, _di_c, _bs_c, _rk_c, _nev_c = 5376, 21504, 256, 16, 2048
    _t0_c = time.perf_counter()
    _xt_c = torch.randn((_nev_c, _di_c), device=DEVICE, dtype=torch.float32) * 0.25
    _wt_c = torch.randn((_do_c, _di_c), dtype=torch.float32) * 0.05
    _hh_c = (_xt_c.T @ _xt_c).float() + 0.05 * torch.eye(_di_c, device=DEVICE, dtype=torch.float32)
    _xr_c = torch.sqrt(((_xt_c * _xt_c).mean(dim=0)).clamp(min=1e-8))
    _hbox_c = [_hh_c]; del _hh_c
    _pb_c, _is_c, _lr_c, _we_c = pack_matrix_gpu_gptq(
        _wt_c, _hbox_c, _xr_c, mode=0, bsize=_bs_c, lr_rank=_rk_c, x_eval_gpu=_xt_c, out_device=DEVICE
    )
    assert len(_hbox_c) == 0
    assert _lr_c is not None and _lr_c[2].shape == (_do_c, _rk_c) and _lr_c[3].shape == (_rk_c, _di_c)
    assert _we_c.shape == (_do_c, _di_c) and _we_c.device == DEVICE and torch.isfinite(_we_c).all()
    _peaks = [f"gpu{di}={torch.cuda.max_memory_allocated(di)/1024**3:.2f}GiB" for di in range(torch.cuda.device_count())]
    for _d_idx in range(torch.cuda.device_count()):
        assert torch.cuda.max_memory_allocated(_d_idx) / 1024**3 < 13.5, f"Peak VRAM exceeded safety bound on GPU {_d_idx}!"
    del _xt_c, _wt_c, _xr_c, _hbox_c, _pb_c, _is_c, _lr_c, _we_c
gc.collect()
for _d_idx in range(torch.cuda.device_count()):
    with torch.cuda.device(_d_idx):
        torch.cuda.empty_cache()
print(f"[Gate 0c] Full-scale down_proj (5376x21504, r=16, n_ev=2048) peak VRAM PASS ({', '.join(_peaks)}, dt={time.perf_counter()-_t0_c:.1f}s).", flush=True)

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
    """Return exact HF Gemma 4 sliding bidirectional vision mask `[mb, 1, S, S]` (True=attend) if any seq in `[b0, b1)` has an image span, else None."""
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


# [Gate 0d] Exact HF bidirectional sliding vision mask parity check for SDPA (True=attend, scale=1.0)
with torch.no_grad():
    _q_m = torch.randn((2, 4, 16, 32), device=DEVICE, dtype=torch.float32) * 0.2
    _k_m = torch.randn((2, 4, 16, 32), device=DEVICE, dtype=torch.float32) * 0.2
    _v_m = torch.randn((2, 4, 16, 32), device=DEVICE, dtype=torch.float32) * 0.2
    _mb = make_sliding_attn_mask(0, 2, {1: (4, 10)}, 16, DEVICE)
    assert _mb is not None and _mb.shape == (2, 1, 16, 16)
    assert bool(_mb[1, 0, 4, 9].item()) and not bool(_mb[0, 0, 4, 9].item())
    _sdpa_o = F.scaled_dot_product_attention(_q_m, _k_m, _v_m, attn_mask=_mb, scale=1.0)
    _ref_s = torch.matmul(_q_m, _k_m.transpose(-2, -1))
    _ref_s = _ref_s.masked_fill(~_mb, float("-inf"))
    _ref_o = torch.matmul(F.softmax(_ref_s, dim=-1), _v_m)
    assert float((_sdpa_o - _ref_o).abs().max().item()) < 1e-4, "Gate 0d bidirectional SDPA mask mismatch!"
    del _q_m, _k_m, _v_m, _mb, _sdpa_o, _ref_s, _ref_o
print("[Gate 0d] Exact HF bidirectional vision sliding-window SDPA mask parity verified.", flush=True)


# ---------------------------------------------------------------------------
# 2. Download Model & Build Clean Complete Full-Dataset Corpora
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
    return not any(b in low for b in vc.BLACKLIST_SUBSTRINGS)

print("[Stage 1] Loading & auditing FULL clean text datasets (zero sample truncation, zero contamination)...", flush=True)
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
perm = rng_np.permutation(len(clean_turns))
print(f"[Stage 1] Total clean corpus items audited: {len(clean_turns):,} examples across GSM8K/Code/Alpaca/WikiText-2.", flush=True)

# Dense multi-turn packing into (N_TEXT_SEQ=608, N_VLM_SEQ=160 -> N_SEQ=768, SEQ_LEN=352) -> 270,336 live multimodal+text tokens!
N_TEXT_SEQ, N_VLM_SEQ, SEQ_LEN = 608, 160, 352
N_SEQ = N_TEXT_SEQ + N_VLM_SEQ
GEMMA4_BOI_ID, GEMMA4_IMAGE_ID, GEMMA4_EOI_ID = 255999, 258880, 258882

by_dom = defaultdict(list)
for idx in perm:
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

print(f"[Stage 1] Packed {N_TEXT_SEQ} dense text sequences x {SEQ_LEN} = {N_TEXT_SEQ * SEQ_LEN:,} text tokens ({examples_packed:,} clean turns); awaiting Stage 2 for {N_VLM_SEQ} VLM sequences.", flush=True)


# ---------------------------------------------------------------------------
# 3. Vision Tower 100% 3-Bit Full Calibration, Packing & Multimodal Soft-Token Capture
# ---------------------------------------------------------------------------
VIS_CKPT_PATH = CKPT_DIR / "stage2_vision.pt"
VIS_META_PATH = CKPT_DIR / "stage2_meta.pt"
VLM_SOFT_CACHE = CKPT_DIR / "stage2_vlm_soft.pt"


def _build_stage2_vision() -> dict:
    """Run Stage 2 100% 3-bit Vision calibration & extract exact teacher+student visual soft tokens for Stage 3 VLM."""
    print("\n[Stage 2] 100% 3-Bit Vision Tower Calibration (128 disjoint COCO val[64:] + 32 Optical VA -> 193 T16C mats; val[0..3] held out)...", flush=True)
    cfg_auto = AutoConfig.from_pretrained(str(model_dir))
    proc = Gemma4ImageProcessor.from_pretrained(str(model_dir))

    def make_optical_va_pil(idx: int) -> Tuple[Image.Image, str]:
        im = Image.new("RGB", (672, 672), (248, 248, 248))
        dr = ImageDraw.Draw(im)
        dr.text((18, 18), f"GEMMA4 31B FULL LM=60 VIS=27 FRAME={idx:03d}", fill=(12, 18, 28))
        dr.text((18, 42), "OPTICAL VA PRECISION ANALYSIS C8DB-H8", fill=(12, 18, 28))
        rng_v = np.random.default_rng(1000 + idx)
        for b in range(60):
            bh = int(rng_v.integers(25, 118))
            x = 24 + b * 10
            col = (35, 115, 210) if b >= 58 else ((30, 165, 95) if b % 5 != 0 else (210, 65, 45))
            dr.rectangle([x, 550 - bh, x + 7, 550], fill=col)
        cap = (
            f"An optical diagnostic chart labeled GEMMA4 31B FULL LM=60 VIS=27 FRAME={idx:03d} "
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
            # Strictly held-out evaluation split (includes val[0..3] / frame_01_coco.png — NEVER enters Hessians or C8DB!)
            heldout_pairs.append((im.convert("RGB"), cap))
            continue
        if row_idx < 64:
            # Full 64-image exclusion buffer covering all vis_frames_32 evaluation images (zero contamination)
            continue
        if not is_clean_row(cap):
            continue
        vis_coco_pairs.append((im.convert("RGB"), cap))
        if len(vis_coco_pairs) >= 128:
            break
    assert len(vis_coco_pairs) == 128, f"Expected 128 disjoint COCO calibration pairs, got {len(vis_coco_pairs)}"
    assert len(heldout_pairs) == 4, f"Expected 4 held-out COCO eval pairs, got {len(heldout_pairs)}"

    vis_pairs: List[Tuple[Image.Image, str]] = []
    for i in range(32):
        vis_pairs.append(vis_va_pairs[i])
        vis_pairs.extend(vis_coco_pairs[i * 4 : (i + 1) * 4])
    assert len(vis_pairs) == N_VLM_SEQ
    vis_images = [p[0] for p in vis_pairs]
    vis_captions = [p[1] for p in vis_pairs]
    n_frames = len(vis_images)
    print(f"[Stage 2] Prepared {n_frames} multimodal VLM pairs (128 disjoint COCO val[64:] + 32 Optical VA) + {len(heldout_pairs)} strictly held-out COCO val[0..3] eval pairs.", flush=True)

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
            accum_hess_t4(vis_hess[key], vis_xsq[key], x)
            vis_cnt[key] += x.shape[0]
        return mod.register_forward_hook(_hk)

    hooks = [reg_vis_hook(vis_teacher.patch_embedder.input_proj, "model.vision_tower.patch_embedder.input_proj.weight")]
    for l_v, lv_mod in enumerate(vis_teacher.encoder.layers):
        pfx_v = f"model.vision_tower.encoder.layers.{l_v}"
        hooks.append(reg_vis_hook(lv_mod.self_attn.q_proj, f"{pfx_v}.self_attn.q_proj.linear.weight"))
        hooks.append(reg_vis_hook(lv_mod.self_attn.k_proj, f"{pfx_v}.self_attn.k_proj.linear.weight"))
        hooks.append(reg_vis_hook(lv_mod.self_attn.v_proj, f"{pfx_v}.self_attn.v_proj.linear.weight"))
        hooks.append(reg_vis_hook(lv_mod.self_attn.o_proj, f"{pfx_v}.self_attn.o_proj.linear.weight"))
        hooks.append(reg_vis_hook(lv_mod.mlp.gate_proj, f"{pfx_v}.mlp.gate_proj.linear.weight"))
        hooks.append(reg_vis_hook(lv_mod.mlp.up_proj, f"{pfx_v}.mlp.up_proj.linear.weight"))
        hooks.append(reg_vis_hook(lv_mod.mlp.down_proj, f"{pfx_v}.mlp.down_proj.linear.weight"))
    hooks.append(reg_vis_hook(mm_teacher.embedding_projection, "model.embed_vision.embedding_projection.weight"))

    # Per-image forward (batch=1) guarantees exact non-padded `[n_soft_i, 5376]` soft-token count per image!
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
            assert 1 <= st.shape[0] <= 280, f"Unexpected n_soft={st.shape[0]} (> 280)"
            vis_eval_pvs.append(pv.cpu())
            vis_eval_poss.append(pos.cpu())
            vis_teach_soft.append(st.to(torch.float16).cpu())

    for h in hooks:
        h.remove()

    # Capture strictly held-out teacher soft tokens AFTER removing all calibration hooks!
    heldout_pvs: List[torch.Tensor] = []
    heldout_poss: List[torch.Tensor] = []
    heldout_teach_soft: List[torch.Tensor] = []
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

    print(f"[Stage 2] Vision Hessians collected over {vis_cnt['model.embed_vision.embedding_projection.weight']:,} soft tokens (160 calibration frames, 0 held-out frames).", flush=True)

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
        pb, inv_s, lr, w_eff = pack_matrix_gpu_gptq(w_pad, [h_use] if h_use is not None else None, x_rms, mode=mode, bsize=64, lr_rank=lr_r, x_eval_gpu=x_ev)
        del h_use, x_rms, x_ev
        n_bits, is_asym = p3.MODE_TO_T16C[mode]
        vision_records.append((name, d_o, d_in_eff, 64, n_bits, is_asym, True, pb, inv_s, lr))
        vis_deq_weights[name] = w_eff[:, :d_i].to(torch.float16)

    for ax in (0, 1):
        nm_pos = f"model.vision_tower.patch_embedder.position_embedding_table.{ax}"
        pb, inv_s, lr, w_eff = pack_matrix_gpu_gptq(pos_tbl_f32[ax], None, None, mode=4, bsize=64, lr_rank=0)
        vision_records.append((nm_pos, pos_tbl_f32.shape[1], pos_tbl_f32.shape[2], 64, 3, False, True, pb, inv_s, lr))
        vis_deq_weights[nm_pos] = w_eff

    pack_vision_tensor("model.vision_tower.patch_embedder.input_proj.weight", vis_sd["patch_embedder.input_proj.weight"], mode=4, lr_r=16)
    for l_v in range(27):
        pfx = f"vision_tower.encoder.layers.{l_v}"
        full_pfx = f"model.{pfx}"
        for nk in ["input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
                   "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]:
            vision_norms[f"{full_pfx}.{nk}"] = vis_sd[f"encoder.layers.{l_v}.{nk}"].float().numpy().astype("<f4")
        # V4.3 VLM: ALL 27 ViT layers upgraded to 3-bit symmetric (mode=4) for high soft-token parity (>0.96 cosine)!
        v_mode = 4
        v_rk = 8 if l_v in (0, 1, 24, 25, 26) else 0
        for proj in ["self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight",
                     "self_attn.o_proj.linear.weight", "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight"]:
            w_t = vis_sd[f"encoder.layers.{l_v}.{proj}"]
            pack_vision_tensor(f"{full_pfx}.{proj}", w_t, mode=v_mode, lr_r=v_rk)
        w_d = vis_sd[f"encoder.layers.{l_v}.mlp.down_proj.linear.weight"]
        pack_vision_tensor(f"{full_pfx}.mlp.down_proj.linear.weight", w_d, mode=v_mode, lr_r=v_rk, pad_in=4352)

    pack_vision_tensor("model.embed_vision.embedding_projection.weight", embv_sd["embedding_projection.weight"], mode=4, lr_r=32)
    v_bytes = sum(len(r[7]) + (len(r[8]) * 4 if r[8] is not None else 0) + ((r[9][0].nbytes + r[9][1].nbytes + r[9][2].nbytes + r[9][3].nbytes) if r[9] is not None else 0) + 80 for r in vision_records)
    print(f"[Stage 2] Packed ALL {len(vision_records)} 3-bit Vision mats + {len(vision_norms)} norms = {v_bytes/1e6:.2f} MB!", flush=True)

    vis_stud = Gemma4VisionModel(cfg_auto.vision_config).to(torch.float16)
    vis_stud.load_state_dict(vis_sd, strict=True)
    with torch.no_grad():
        vis_stud.patch_embedder.input_proj.weight.copy_(vis_deq_weights["model.vision_tower.patch_embedder.input_proj.weight"])
        vis_stud.patch_embedder.position_embedding_table[0].copy_(vis_deq_weights["model.vision_tower.patch_embedder.position_embedding_table.0"])
        vis_stud.patch_embedder.position_embedding_table[1].copy_(vis_deq_weights["model.vision_tower.patch_embedder.position_embedding_table.1"])
        for l_v, lv_mod in enumerate(vis_stud.encoder.layers):
            full_pfx = f"model.vision_tower.encoder.layers.{l_v}"
            for sub, m_sub in [("self_attn.q_proj", lv_mod.self_attn.q_proj), ("self_attn.k_proj", lv_mod.self_attn.k_proj),
                               ("self_attn.v_proj", lv_mod.self_attn.v_proj), ("self_attn.o_proj", lv_mod.self_attn.o_proj),
                               ("mlp.gate_proj", lv_mod.mlp.gate_proj), ("mlp.up_proj", lv_mod.mlp.up_proj),
                               ("mlp.down_proj", lv_mod.mlp.down_proj)]:
                m_sub.linear.weight.copy_(vis_deq_weights[f"{full_pfx}.{sub}.linear.weight"])
    vis_stud = vis_stud.to(DEVICE).eval()
    mm_stud = Gemma4MultimodalEmbedder(cfg_auto.vision_config, cfg_auto.text_config).to(torch.float16)
    mm_stud.load_state_dict(embv_sd, strict=True)
    with torch.no_grad():
        mm_stud.embedding_projection.weight.copy_(vis_deq_weights["model.embed_vision.embedding_projection.weight"])
    mm_stud = mm_stud.to(DEVICE).eval()

    vis_stud_pre_in: List[torch.Tensor] = []
    vis_stud_soft_before: List[torch.Tensor] = []
    with torch.no_grad():
        for pv, pos in zip(vis_eval_pvs, vis_eval_poss):
            vo_s = vis_stud(pixel_values=pv.to(DEVICE), pixel_position_ids=pos.to(DEVICE))
            pre_s = mm_stud.embedding_pre_projection_norm(vo_s.last_hidden_state).reshape(-1, 1152)
            soft_s = mm_stud.embedding_projection(pre_s).reshape(-1, 5376)
            vis_stud_pre_in.append(pre_s.float().cpu())
            vis_stud_soft_before.append(soft_s.float().cpu())

    X_v_all = torch.cat(vis_stud_pre_in, dim=0)
    Y_v_all = torch.cat([t.float() for t in vis_teach_soft], dim=0)
    S_v_all = torch.cat(vis_stud_soft_before, dim=0)
    perm_v = torch.randperm(X_v_all.shape[0], generator=torch.Generator().manual_seed(42))
    nv_tr = (X_v_all.shape[0] * 4) // 5
    tr_v, va_v = perm_v[:nv_tr], perm_v[nv_tr:]
    X_v_tr, Y_v_tr, S_v_tr = X_v_all[tr_v].to(DEVICE), Y_v_all[tr_v].to(DEVICE), S_v_all[tr_v].to(DEVICE)
    X_v_va, Y_v_va, S_v_va = X_v_all[va_v].to(DEVICE), Y_v_all[va_v].to(DEVICE), S_v_all[va_v].to(DEVICE)
    assert X_v_tr.ndim == 2 and Y_v_tr.shape == S_v_tr.shape and Y_v_va.shape == S_v_va.shape
    v_cos0 = float(F.cosine_similarity(S_v_va, Y_v_va, dim=-1).mean().item())

    c8_vis = c8d.Cuneiform8DPatternAdapter(d_out=5376, d_in=1152, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(DEVICE)
    opt_v = torch.optim.AdamW(c8_vis.parameters(), lr=1.5e-3, weight_decay=5e-2)
    best_vc, best_vsd = v_cos0, None
    for step in range(300):
        opt_v.zero_grad()
        idx_b = torch.randint(0, X_v_tr.shape[0], (min(1024, X_v_tr.shape[0]),), device=DEVICE)
        dv, _, lbv = c8_vis(X_v_tr[idx_b])
        pred = S_v_tr[idx_b] + dv
        loss_v = F.mse_loss(pred, Y_v_tr[idx_b]) + 0.5 * (1.0 - F.cosine_similarity(pred, Y_v_tr[idx_b], dim=-1).mean()) + 0.005 * lbv
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(c8_vis.parameters(), 1.0)
        opt_v.step()
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                dv_v, _, _ = c8_vis(X_v_va)
                cv_i = float(F.cosine_similarity(S_v_va + dv_v, Y_v_va, dim=-1).mean().item())
                if cv_i > best_vc:
                    best_vc = cv_i
                    best_vsd = {k: v.detach().cpu().clone() for k, v in c8_vis.state_dict().items()}

    c8_vis_cpu = c8_vis.cpu()
    v_keep = False
    if best_vsd is not None:
        c8_vis_cpu.load_state_dict(best_vsd)
        c_v, e_v = c8_vis_cpu.export_quantized_arrays()
        dq_v = torch.from_numpy(c8d.eval_quantized_adapter_np(X_v_va.cpu().numpy()[:512], c_v, e_v, 2, 0.5, 1.0)).to(DEVICE)
        v_cos1 = float(F.cosine_similarity(S_v_va[:512] + dq_v, Y_v_va[:512], dim=-1).mean().item())
        v_keep = v_cos1 > v_cos0 + 0.001
    else:
        c_v, e_v = None, None
        v_cos1 = v_cos0

    # Produce exact post-C8DB student soft tokens for Stage 3 VLM calibration AND strictly held-out COCO val[0..3] eval!
    vlm_stud_soft: List[torch.Tensor] = []
    with torch.no_grad():
        for pre_cpu, soft_cpu in zip(vis_stud_pre_in, vis_stud_soft_before):
            s_cur = soft_cpu.to(DEVICE)
            if v_keep and c_v is not None:
                s_cur = s_cur + eval_c8d_gpu(pre_cpu.to(DEVICE), c_v, e_v)
            vlm_stud_soft.append(s_cur.to(torch.float16).cpu())

    heldout_stud_soft: List[torch.Tensor] = []
    heldout_vis_coss: List[float] = []
    with torch.no_grad():
        for pv_h, pos_h, t_soft_h in zip(heldout_pvs, heldout_poss, heldout_teach_soft):
            vo_h = vis_stud(pixel_values=pv_h.to(DEVICE), pixel_position_ids=pos_h.to(DEVICE))
            pre_h = mm_stud.embedding_pre_projection_norm(vo_h.last_hidden_state).reshape(-1, 1152)
            soft_h = mm_stud.embedding_projection(pre_h).reshape(-1, 5376).float()
            if v_keep and c_v is not None:
                soft_h = soft_h + eval_c8d_gpu(pre_h, c_v, e_v)
            heldout_vis_coss.append(float(F.cosine_similarity(soft_h, t_soft_h.to(DEVICE).float(), dim=-1).mean().item()))
            heldout_stud_soft.append(soft_h.to(torch.float16).cpu())
    heldout_vis_cos = float(np.mean(heldout_vis_coss))
    print(f"[Stage 2 DONE] 3-Bit Vision Student Val Soft-Token Cosine: {v_cos0:.5f} -> {v_cos1:.5f} (keep={v_keep}) | Held-Out COCO val[0..3] Soft-Token Cosine={heldout_vis_cos:.5f}", flush=True)

    _atomic_torch_save({
        "vlm_captions": vis_captions,
        "vlm_teach_soft": vis_teach_soft,
        "vlm_stud_soft": vlm_stud_soft,
        "heldout_captions": [p[1] for p in heldout_pairs],
        "heldout_teach_soft": heldout_teach_soft,
        "heldout_stud_soft": heldout_stud_soft,
        "heldout_vis_cos": heldout_vis_cos,
    }, VLM_SOFT_CACHE)

    meta = {
        "vis_total_bytes": int(v_bytes),
        "n_vis_mats": len(vision_records),
        "vision_norms": vision_norms,
        "vis_keep_c8d": bool(v_keep),
        "cents_vis": c_v,
        "experts_vis": e_v,
        "vis_cos_before": float(v_cos0),
        "vis_cos_after": float(v_cos1),
        "heldout_vis_cos": float(heldout_vis_cos),
        "vis_frames_calibrated": int(n_frames),
    }
    _atomic_torch_save(meta, VIS_META_PATH)
    _atomic_torch_save({**meta, "vision_records": vision_records}, VIS_CKPT_PATH)
    if v_keep and c_v is not None:
        exp_rv0 = [c8d._c6d_radicals("vision", "patch", "soft", "native", "layer", "positive", e) for e in range(8)]
        c8d.write_c8db_file(OUT_C8D, [], [{
            "name": "model.embed_vision.embedding_projection.weight", "d_out": 5376, "d_in": 1152, "rank": 32, "top_k": 2,
            "c6d": exp_rv0[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": c_v, "expert_radicals": exp_rv0, "experts": e_v,
        }])
    save_progress({"stage": "stage2_done", "vis_total_bytes": int(v_bytes), "n_vis_mats": len(vision_records), "vis_cos_after": float(v_cos1), "heldout_vis_cos": float(heldout_vis_cos)})
    return meta


if VIS_CKPT_PATH.is_file() and VLM_SOFT_CACHE.is_file():
    _vm = torch.load(VIS_CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"[Resume Stage 2] Loaded {VIS_CKPT_PATH} ({_vm['n_vis_mats']} mats, {_vm['vis_total_bytes']/1e6:.2f} MB, vis_cos={_vm['vis_cos_after']:.5f})", flush=True)
elif VIS_META_PATH.is_file() and VLM_SOFT_CACHE.is_file() and OUT_T16C.is_file():
    _vm = torch.load(VIS_META_PATH, map_location="cpu", weights_only=False)
    print(f"[Resume Stage 2] Loaded {VIS_META_PATH} (OUT_T16C already finalized)", flush=True)
else:
    _vm = _build_stage2_vision()

vis_total_bytes = int(_vm["vis_total_bytes"])
n_vis_mats = int(_vm["n_vis_mats"])
vision_norms = _vm["vision_norms"]
vis_keep_c8d = bool(_vm["vis_keep_c8d"])
cents_vis = _vm["cents_vis"]
experts_vis = _vm["experts_vis"]
vis_cos_before = float(_vm["vis_cos_before"])
vis_cos_after = float(_vm["vis_cos_after"])
heldout_vis_cos = float(_vm.get("heldout_vis_cos", 0.0))
vis_frames_calibrated = int(_vm["vis_frames_calibrated"])
del _vm  # Do NOT hold 228 MB vision_records in Host RAM during Stage 3!
gc.collect()
torch.cuda.empty_cache()

# Stage 2b: Pack 160 real VLM image-caption sequences (`608..767`) + 4 strictly held-out COCO val[0..3] eval sequences!
_vlm_blob = torch.load(VLM_SOFT_CACHE, map_location="cpu", weights_only=False)
vlm_captions: List[str] = _vlm_blob["vlm_captions"]
vlm_teach_soft: List[torch.Tensor] = _vlm_blob["vlm_teach_soft"]
vlm_stud_soft: List[torch.Tensor] = _vlm_blob["vlm_stud_soft"]
heldout_captions: List[str] = _vlm_blob["heldout_captions"]
heldout_teach_soft: List[torch.Tensor] = _vlm_blob["heldout_teach_soft"]
heldout_stud_soft: List[torch.Tensor] = _vlm_blob["heldout_stud_soft"]
del _vlm_blob

VQA_PROMPTS = [
    "Describe this image in one short sentence.",
    "What is shown in this image?",
    "Describe what you see in detail.",
    "Caption this photograph accurately.",
]
u_pfx_toks = list(tk.encode("<|turn>user\n", add_special_tokens=False).ids)
vlm_spans: Dict[int, Tuple[int, int]] = {}
vlm_ans_count = 0

for vi in range(N_VLM_SEQ):
    n_soft = int(vlm_teach_soft[vi].shape[0])
    vqa_q = VQA_PROMPTS[vi % len(VQA_PROMPTS)]
    cap_txt = vlm_captions[vi]
    tail_prompt = list(tk.encode(f"{vqa_q}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", add_special_tokens=False).ids)
    cap_toks = list(tk.encode(f"{cap_txt}<turn|>\n", add_special_tokens=False).ids)

    cur_toks = [2] + u_pfx_toks + [GEMMA4_BOI_ID]
    img_start = len(cur_toks)
    cur_toks.extend([GEMMA4_IMAGE_ID] * n_soft)
    img_end = len(cur_toks)
    cur_toks.append(GEMMA4_EOI_ID)
    cur_toks.extend(tail_prompt)
    cur_ans = [False] * len(cur_toks)
    cur_vlm_ans = [False] * len(cur_toks)
    cur_toks.extend(cap_toks)
    cur_ans.extend([True] * len(cap_toks))
    cur_vlm_ans.extend([True] * len(cap_toks))
    examples_packed += 1

    # Pack clean continuation text turns if sequence < SEQ_LEN + 1 so zero pad tokens exist!
    while len(cur_toks) < SEQ_LEN + 1:
        q_t, a_t = next_turn_pair(t_cursor)
        t_cursor += 1
        p_tok = list(tk.encode(vc.format_gemma4_turn(q_t, None), add_special_tokens=False).ids)
        r_tok = list(tk.encode(f"{a_t}<turn|>\n", add_special_tokens=False).ids)
        if not p_tok or not r_tok:
            continue
        cur_toks.extend(p_tok)
        cur_ans.extend([False] * len(p_tok))
        cur_vlm_ans.extend([False] * len(p_tok))
        cur_toks.extend(r_tok)
        cur_ans.extend([True] * len(r_tok))
        cur_vlm_ans.extend([False] * len(r_tok))

    cur_toks = cur_toks[: SEQ_LEN + 1]
    cur_ans = cur_ans[: SEQ_LEN + 1]
    cur_vlm_ans = cur_vlm_ans[: SEQ_LEN + 1]
    assert img_end < SEQ_LEN, f"VLM image span [{img_start},{img_end}) exceeded SEQ_LEN={SEQ_LEN}"
    seq_idx = N_TEXT_SEQ + vi
    vlm_spans[seq_idx] = (img_start, img_end)
    seq_ids_list.append(cur_toks[:-1])
    seq_nxt_list.append(cur_toks[1:])
    seq_ans_list.append(cur_ans[1:])
    seq_vlm_ans_list.append(cur_vlm_ans[1:])
    vlm_ans_count += sum(cur_vlm_ans[1:])

all_seq_ids = np.array(seq_ids_list, dtype=np.int32)          # [768, 352]
all_seq_nxt = np.array(seq_nxt_list, dtype=np.int32)          # [768, 352]
all_ans_mask = np.array(seq_ans_list, dtype=bool)             # [768, 352]
all_vlm_ans_mask = np.array(seq_vlm_ans_list, dtype=bool)     # [768, 352]
n_ans_total = int(all_ans_mask.sum())
print(f"[Stage 2b] Final Multimodal Calibration Tensor: {all_seq_ids.shape} = {all_seq_ids.size:,} live tokens ({N_TEXT_SEQ} text + {N_VLM_SEQ} VLM), ans_tokens={n_ans_total:,} (vlm_caption_ans={vlm_ans_count:,})", flush=True)
save_progress({"stage": "stage2b_packed", "tokens": int(all_seq_ids.size), "ans_tokens": n_ans_total, "vlm_ans_tokens": int(vlm_ans_count)})

# Pack 4 strictly held-out COCO val[0..3] evaluation sequences (NEVER enters Hessians, SmoothQuant, or C8DB!)
heldout_ids_list: List[List[int]] = []
heldout_nxt_list: List[List[int]] = []
heldout_spans: Dict[int, Tuple[int, int]] = {}
heldout_ans_starts: List[int] = []
heldout_ans_lens: List[int] = []
eval_tail_toks = list(tk.encode("Describe this image in one short sentence.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", add_special_tokens=False).ids)
for hi in range(len(heldout_captions)):
    n_soft = int(heldout_teach_soft[hi].shape[0])
    cap_toks = list(tk.encode(f"{heldout_captions[hi]}<turn|>\n", add_special_tokens=False).ids)
    cur_t = [2] + u_pfx_toks + [GEMMA4_BOI_ID]
    s0 = len(cur_t)
    cur_t.extend([GEMMA4_IMAGE_ID] * n_soft)
    s1 = len(cur_t)
    cur_t.append(GEMMA4_EOI_ID)
    cur_t.extend(eval_tail_toks)
    ans_pos0 = len(cur_t) - 1  # position whose next-token logit predicts cap_toks[0]!
    cur_t.extend(cap_toks)
    ans_len = min(len(cap_toks), max(1, SEQ_LEN - ans_pos0))
    while len(cur_t) < SEQ_LEN + 1:
        cur_t.append(106)
    cur_t = cur_t[: SEQ_LEN + 1]
    heldout_spans[hi] = (s0, s1)
    heldout_ans_starts.append(ans_pos0)
    heldout_ans_lens.append(ans_len)
    heldout_ids_list.append(cur_t[:-1])
    heldout_nxt_list.append(cur_t[1:])
heldout_seq_ids = np.array(heldout_ids_list, dtype=np.int32)  # [4, 352]
heldout_seq_nxt = np.array(heldout_nxt_list, dtype=np.int32)  # [4, 352]

# ---------------------------------------------------------------------------
# 4. Full-Model V4.3 VLM Allocation (6.115 GB LM + 228 MB 100% 3-Bit Vision = 6.345 GB <= 6.38 GB TOTAL)
# ---------------------------------------------------------------------------
# ZERO 1-bit in ANY Attention layer (q,k,v,o across all 60 layers >= tern5 @ b256)
# ALL 27 Vision layers (V00..V26) + embed_vision at 3-bit symmetric (mode=4)
# 1b_asym (mode 0 @ b256, down_proj r=16, gate/up r=8) on 38 sliding-layer MLPs (reclaiming ~72 MB for 3-bit Vision!)
FULL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}
MLP_1B_LAYERS = {7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52}
ATTN_3B_LAYERS = {2, 4, 10, 14, 19, 58, 59}
assert len(MLP_1B_LAYERS) == 38 and MLP_1B_LAYERS.isdisjoint(FULL_LAYERS)

def get_lm_spec(l_idx: int, pname: str) -> Tuple[int, int, int]:
    """Return (mode_i, bsize, lr_rank). Zero 1-bit in Attention anywhere; bsize=256 matches Pi 4 VmHWM <= 6.5 GiB."""
    if l_idx in (58, 59):
        rk = 24 if pname in ("k_proj", "v_proj") else 16
        return (4, 256, rk)
    if l_idx in ATTN_3B_LAYERS and pname in ("q_proj", "o_proj"):
        return (4, 256, 16)
    if l_idx in (0, 1, 56, 57) and pname in ("q_proj", "o_proj", "down_proj"):
        return (1, 256, 16)
    if l_idx in MLP_1B_LAYERS and pname in ("gate_proj", "up_proj", "down_proj"):
        return (0, 256, 16 if pname == "down_proj" else 8)
    rk_early = 16 if (l_idx < 10 and pname in ("q_proj", "o_proj", "down_proj")) else 0
    return (1, 256, rk_early)

# Pre-flight fail-closed .t16c byte budget verification (must stay under 6.38 GB so Pi 4 VmHWM <= 6,815,744 kB)
def _pred_mat_bytes(d_out: int, d_in: int, bs: int, mode: int, lr_r: int, has_inv: bool = True) -> int:
    nb, asy = p3.MODE_TO_T16C[mode]
    pb = p3.expected_tile16_bytes(d_out, d_in, bs, nb, asy)
    return pb + (d_in * 4 if has_inv else 0) + ((d_out * 4 + lr_r * 4 + d_out * lr_r + lr_r * d_in) if lr_r > 0 else 0) + 80

_pred_lm_b = _pred_mat_bytes(int(cfg_full["text_config"]["vocab_size"]), int(cfg_full["text_config"]["hidden_size"]), 256, 4, 0, False)
for _l in range(int(cfg_full["text_config"]["num_hidden_layers"])):
    _f = _l in FULL_LAYERS
    _hd = 512 if _f else 256
    _nkv = 4 if _f else 16
    _sh = {"q_proj": (32 * _hd, 5376), "k_proj": (_nkv * _hd, 5376), "o_proj": (5376, 32 * _hd),
           "gate_proj": (21504, 5376), "up_proj": (21504, 5376), "down_proj": (5376, 21504)}
    if not _f:
        _sh["v_proj"] = (_nkv * _hd, 5376)
    for _pn, (_do, _di) in _sh.items():
        _m, _bs, _rk = get_lm_spec(_l, _pn)
        _pred_lm_b += _pred_mat_bytes(_do, _di, _bs, _m, _rk, True)

_pred_tot_b = _pred_lm_b + vis_total_bytes + 6_500_000
print(f"[Budget Guard] Predicted unified .t16c: LM={_pred_lm_b/1e9:.3f}GB + VIS={vis_total_bytes/1e6:.1f}MB = {_pred_tot_b:,} B ({_pred_tot_b/1024**3:.3f} GiB)", flush=True)
assert _pred_tot_b <= 6_380_000_000, f"Budget overflow ({_pred_tot_b:,} B > 6.38 GB)! Would risk Pi 4 VmHWM > 6.50 GiB."

# ---------------------------------------------------------------------------
# 5. Stream L00..L59 Layer-by-Layer on GPU: Teacher vs Quantized Student
# ---------------------------------------------------------------------------
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

STAGE3_CKPT_PATH = CKPT_DIR / "stage3_layer_ckpt.pt"
CKPT_X_TEACH_W = CKPT_DIR / "ckpt_x_teach.pt"
CKPT_X_STUD_W = CKPT_DIR / "ckpt_x_stud.pt"
CKPT_X_TEACH_TMP = Path("/tmp/ckpt_x_teach.pt")
CKPT_X_STUD_TMP = Path("/tmp/ckpt_x_stud.pt")

print(f"\n[Stage 3] Streaming {N_LAYERS} LM layers across {all_seq_ids.size:,} tokens on T4 GPU...", flush=True)
with safe_open(sf_path, framework="pt", device="cpu") as sf:
    lm_norms: Dict[str, np.ndarray] = {}
    for _l in range(N_LAYERS):
        _p = f"model.language_model.layers.{_l}."
        for _nm_key in (
            "input_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "layer_scalar",
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

    _ckpt_t_p = CKPT_X_TEACH_W if CKPT_X_TEACH_W.is_file() else CKPT_X_TEACH_TMP
    _ckpt_s_p = CKPT_X_STUD_W if CKPT_X_STUD_W.is_file() else CKPT_X_STUD_TMP
    if STAGE3_CKPT_PATH.is_file() and OUT_T16C.is_file() and _ckpt_t_p.is_file() and _ckpt_s_p.is_file():
        _lc = torch.load(STAGE3_CKPT_PATH, map_location="cpu", weights_only=False)
        start_layer = int(_lc["next_layer"])
        t16c_pos = int(_lc["t16c_pos"])
        lm_mats_count = int(_lc["lm_mats_count"])
        bridge_cache: Dict[str, torch.Tensor] = _lc.get("bridge_cache", {})
        H_teach_eval = _lc["H_teach_eval"]
        H_stud_eval = _lc["H_stud_eval"]
        X_teach_cpu = torch.load(_ckpt_t_p, map_location="cpu", weights_only=True)
        X_stud_cpu = torch.load(_ckpt_s_p, map_location="cpu", weights_only=True)
        del _lc
        t16c_fp = open(OUT_T16C, "r+b")
        t16c_fp.seek(t16c_pos)
        t16c_fp.truncate(t16c_pos)
        print(f"[Resume Stage 3] Resuming from L{start_layer:02d} (t16c_pos={t16c_pos:,} B, lm_mats={lm_mats_count}/{expected_lm_mats}) | ram={host_ram_gb()[0]:.2f}GiB", flush=True)
    else:
        start_layer = 0
        bridge_cache = {}
        emb_bf16_cpu = sf.get_tensor("model.language_model.embed_tokens.weight")
        ids_t = torch.from_numpy(all_seq_ids).long()
        X_teach_cpu = (emb_bf16_cpu[ids_t].float() * ESCALE).to(torch.float16)  # [768, 352, 5376] (~2.91 GiB CPU)
        eval_ids_t = torch.from_numpy(heldout_seq_ids).long()
        H_teach_eval = (emb_bf16_cpu[eval_ids_t].float() * ESCALE).to(torch.float16)  # [4, 352, 5376]

        # Scatter Teacher visual soft tokens into VLM calibration sequences & strictly held-out COCO val[0..3] sequences
        for vi in range(N_VLM_SEQ):
            s0, s1 = vlm_spans[N_TEXT_SEQ + vi]
            X_teach_cpu[N_TEXT_SEQ + vi, s0:s1] = vlm_teach_soft[vi]
        for hi in range(len(heldout_captions)):
            s0, s1 = heldout_spans[hi]
            H_teach_eval[hi, s0:s1] = heldout_teach_soft[hi]

        # Pack embed_tokens.weight (3b_u0 @ b256) cleanly in chunked GPU mode -> ~539.5 MB
        print("[Stage 3] Packing embed_tokens.weight (3b_u0 @ b256)...", flush=True)
        pb_emb, inv_emb, lr_emb, emb_deq_cpu = pack_matrix_gpu_gptq(emb_bf16_cpu, None, None, mode=4, bsize=256, lr_rank=0)
        X_stud_cpu = (emb_deq_cpu[ids_t] * ESCALE).to(torch.float16)
        H_stud_eval = (emb_deq_cpu[eval_ids_t] * ESCALE).to(torch.float16)

        # Scatter Student (3-bit ViT + embed_vision + Vision C8DB) visual soft tokens into VLM & held-out sequences
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
            t16c_fp.write(_vf.tobytes())
            t16c_pos += 2 + len(_nb) + 4 + _vf.size * 4
        t16c_pos = p3._write_mat_record(
            t16c_fp, t16c_pos, "model.language_model.embed_tokens.weight",
            int(tc["vocab_size"]), D_MOD, 256, 3, False, True, pb_emb, inv_emb, lr_emb,
        )
        lm_mats_count = 1
        del pb_emb, inv_emb, lr_emb
        gc.collect()
        t16c_fp.flush()
        save_progress({"stage": "embed_done", "next_layer": 0, "t16c_pos": t16c_pos})

    del vlm_teach_soft, vlm_stud_soft, heldout_teach_soft, heldout_stud_soft
    gc.collect()

    # Precompute causal mask & RoPE tables on GPU for seq_len=256
    pos_1d = torch.arange(SEQ_LEN, device=DEVICE, dtype=torch.float32)

    def make_rope(theta: float, rot_d: int, hd: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = rot_d // 2
        inv_f = 1.0 / (theta ** (torch.arange(0, 2 * hr, 2, device=DEVICE, dtype=torch.float32) / float(hd)))
        ang = pos_1d[:, None] * inv_f[None, :]
        return torch.cos(ang).to(torch.float16), torch.sin(ang).to(torch.float16)

    def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_d: int, hd: int) -> torch.Tensor:
        # t: [B, S, H, hd]
        hr, hh = rot_d // 2, hd // 2
        c = cos[None, :, None, :]
        s = sin[None, :, None, :]
        x0 = t[..., :hr]
        x1 = t[..., hh : hh + hr]
        out = t.clone()
        out[..., :hr] = x0 * c - x1 * s
        out[..., hh : hh + hr] = x0 * s + x1 * c
        return out

    def rmsn_gpu(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
        if w is not None:
            y = y * w.float()
        return y.to(torch.float16)

    def unpack_qat(pfx: str) -> torch.Tensor:
        wp = sf.get_tensor(f"{pfx}.weight_packed").to(DEVICE)
        ws = sf.get_tensor(f"{pfx}.weight_scale").to(DEVICE, dtype=torch.float32)
        sh = torch.arange(0, 32, 4, device=DEVICE, dtype=torch.int32)
        nibs = ((wp.unsqueeze(-1) >> sh) & 0xF) - 8
        return (nibs.to(torch.float32).reshape(wp.shape[0], -1, 32) * ws.unsqueeze(-1)).reshape(wp.shape[0], -1).cpu()

    def layer_step_block(X_cpu: torch.Tensor, weights: Dict[str, torch.Tensor], norms_g: Dict[str, torch.Tensor],
                         l_idx: int, record_io: bool = False,
                         spans_dict: Optional[Dict[int, Tuple[int, int]]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        is_full = LAYER_TYPES[l_idx] == "full_attention"
        nkv = NKV_F if is_full else NKV_S
        hd = HD_F if is_full else HD_S
        theta = THETA_G if is_full else THETA_L
        rot_d = int(round(hd * (PART_G if is_full else PART_L)))
        cos_r, sin_r = make_rope(theta, rot_d, hd)
        gqa_rep = NH // nkv
        lsc = float(norms_g["lsc"])
        out_cpu = torch.empty_like(X_cpu)
        stats = {}
        # Accumulate input stats or save bridge slices if l_idx in (56, 57, 58, 59)
        mb_sz = 32  # 32 * 352 = 11,264 tokens/chunk -> fits comfortably in L4 24 GiB VRAM!
        if record_io:
            h_attn_in = torch.zeros((D_MOD, D_MOD), device=DEVICE, dtype=torch.float32)
            xsq_in = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
            do_o = weights["o_proj"].shape[0]
            di_o = weights["o_proj"].shape[1]
            h_o = torch.zeros((di_o, di_o), device=DEVICE, dtype=torch.float32)
            xsq_o = torch.zeros((di_o,), device=DEVICE, dtype=torch.float32)
            h_mlp = torch.zeros((D_MOD, D_MOD), device=DEVICE, dtype=torch.float32)
            xsq_mlp = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
            di_d = weights["down_proj"].shape[1]
            h_d = torch.zeros((di_d, di_d), device=DEVICE, dtype=torch.float32)
            xsq_d = torch.zeros((di_d,), device=DEVICE, dtype=torch.float32)
            stats["n_tok"] = X_cpu.shape[0] * X_cpu.shape[1]
        for b0 in range(0, X_cpu.shape[0], mb_sz):
            b1 = min(b0 + mb_sz, X_cpu.shape[0])
            xb = X_cpu[b0:b1].to(DEVICE)  # [mb, SEQ_LEN, 5376]
            inp = rmsn_gpu(xb, norms_g["in"])
            inpf = inp.reshape(-1, D_MOD).float()
            if record_io:
                h_attn_in.add_(inpf.T @ inpf)
                xsq_in.add_((inpf * inpf).sum(dim=0))
            q = F.linear(inp, weights["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
            k = F.linear(inp, weights["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
            v = k.clone() if (is_full and K_EQ_V) else F.linear(inp, weights["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
            qn = apply_rope(rmsn_gpu(q, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
            kn = apply_rope(rmsn_gpu(k, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
            vn = rmsn_gpu(v, None).transpose(1, 2)
            kn_exp = kn.repeat_interleave(gqa_rep, dim=1)
            vn_exp = vn.repeat_interleave(gqa_rep, dim=1)
            amask = None if (is_full or not spans_dict) else make_sliding_attn_mask(b0, b1, spans_dict, SEQ_LEN, DEVICE)
            if amask is not None:
                ctx = F.scaled_dot_product_attention(qn, kn_exp, vn_exp, attn_mask=amask, scale=1.0)
            else:
                ctx = F.scaled_dot_product_attention(qn, kn_exp, vn_exp, is_causal=True, scale=1.0)
            ctx = ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd)
            if record_io:
                cf = ctx.reshape(-1, di_o).float()
                h_o.add_(cf.T @ cf)
                xsq_o.add_((cf * cf).sum(dim=0))
            ot = F.linear(ctx, weights["o_proj"])
            xm = xb + rmsn_gpu(ot, norms_g["pa"])
            hm = rmsn_gpu(xm, norms_g["pf"])
            if record_io:
                hmf = hm.reshape(-1, D_MOD).float()
                h_mlp.add_(hmf.T @ hmf)
                xsq_mlp.add_((hmf * hmf).sum(dim=0))
            gt = F.linear(hm, weights["gate_proj"]).float()
            ut = F.linear(hm, weights["up_proj"]).float()
            act = (F.gelu(gt, approximate="tanh") * ut).to(torch.float16)
            if record_io:
                af = act.reshape(-1, di_d).float()
                h_d.add_(af.T @ af)
                xsq_d.add_((af * af).sum(dim=0))
            dt_o = F.linear(act, weights["down_proj"])
            out_b = ((xm.float() + rmsn_gpu(dt_o, norms_g["pff"]).float()) * lsc).to(torch.float16)
            out_cpu[b0:b1] = out_b.cpu()
        if record_io:
            stats.update({"h_in": h_attn_in, "xsq_in": xsq_in, "h_o": h_o, "xsq_o": xsq_o,
                          "h_mlp": h_mlp, "xsq_mlp": xsq_mlp, "h_d": h_d, "xsq_d": xsq_d})
        return out_cpu, stats

    try:
        for l_idx in range(start_layer, N_LAYERS):
            tl = time.perf_counter()
            p = f"model.language_model.layers.{l_idx}."
            is_full = LAYER_TYPES[l_idx] == "full_attention"
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

            w_teach_f16 = {k: v.to(DEVICE, dtype=torch.float16) for k, v in w_teach_f32.items()}
            X_teach_next, _ = layer_step_block(X_teach_cpu, w_teach_f16, norms_g, l_idx, record_io=False, spans_dict=vlm_spans)
            H_teach_eval, _ = layer_step_block(H_teach_eval, w_teach_f16, norms_g, l_idx, record_io=False, spans_dict=heldout_spans)
            del w_teach_f16
            if l_idx < 58:
                del X_teach_cpu

            if l_idx == 58:
                _atomic_torch_save(X_stud_cpu, Path("/tmp/x_stud_pre58.pt"))
                _atomic_torch_save(H_stud_eval, Path("/tmp/h_stud_eval_pre58.pt"))

            # Compute Student input stats for GPTQ + ActALS directly on Student X_stud_cpu (true multimodal causal recovery)
            w_stud_f16: Dict[str, torch.Tensor] = {}
            n_tok_l = X_stud_cpu.shape[0] * X_stud_cpu.shape[1]

            # 1) Attention Q/K/V inputs
            h_in = torch.zeros((D_MOD, D_MOD), device=DEVICE, dtype=torch.float32)
            xsq_in = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
            x_in_eval = []
            for b0 in range(0, X_stud_cpu.shape[0], 64):
                xb = X_stud_cpu[b0 : min(b0 + 64, X_stud_cpu.shape[0])].to(DEVICE)
                inp = rmsn_gpu(xb, norms_g["in"]).reshape(-1, D_MOD).float()
                accum_hess_t4(h_in, xsq_in, inp)
                if len(x_in_eval) < 3 or b0 >= N_TEXT_SEQ:
                    if len(x_in_eval) < 4:
                        x_in_eval.append(inp[:512].clone())
            xrms_in = torch.sqrt((xsq_in / n_tok_l).clamp(min=1e-8))
            xev_in = torch.cat(x_in_eval, dim=0)[:2048]
            for pn in (["q_proj", "k_proj", "o_proj"] if (is_full and K_EQ_V) else ["q_proj", "k_proj", "v_proj"]):
                if pn == "o_proj": continue
                m_i, bs_i, rk_i = get_lm_spec(l_idx, pn)
                pb, inv_s, lr, w_eff = pack_matrix_gpu_gptq(w_teach_f32[pn].cpu(), h_in, xrms_in, m_i, bs_i, rk_i, xev_in)
                nb, asy = p3.MODE_TO_T16C[m_i]
                t16c_pos = p3._write_mat_record(
                    t16c_fp, t16c_pos, f"{p}self_attn.{pn}.weight", w_teach_f32[pn].shape[0], D_MOD,
                    bs_i, nb, asy, True, pb, inv_s, lr,
                )
                lm_mats_count += 1
                w_stud_f16[pn] = w_eff
                del pb, inv_s, lr
            del h_in, xsq_in, xrms_in, xev_in, x_in_eval

            # Save Teacher & Student KV slices at L58..L59 (balanced 96 text + 32 VLM = 128 seqs = 45,056 tokens!)
            if l_idx in (58, 59):
                sub_idx = np.concatenate([
                    np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int),
                    np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int),
                ])
                xt_s = X_teach_cpu[sub_idx].to(DEVICE)
                inp_t = rmsn_gpu(xt_s, norms_g["in"]).reshape(-1, D_MOD).float()
                if l_idx == 58:
                    xs_s = X_stud_cpu[sub_idx].to(DEVICE)
                    inp_s = rmsn_gpu(xs_s, norms_g["in"]).reshape(-1, D_MOD).float()
                    bridge_cache["in_58_stud"] = inp_s.to(torch.float16).cpu()
                    bridge_cache["kt58"] = (inp_t @ w_teach_f32["k_proj"].to(DEVICE).T).to(torch.float16).cpu()
                    bridge_cache["ks58_base"] = (inp_s @ w_stud_f16["k_proj"].float().T).to(torch.float16).cpu()
                    bridge_cache["vt58"] = (inp_t @ w_teach_f32["v_proj"].to(DEVICE).T).to(torch.float16).cpu()
                    bridge_cache["vs58_base"] = (inp_s @ w_stud_f16["v_proj"].float().T).to(torch.float16).cpu()
                    del xs_s, inp_s
                elif l_idx == 59:
                    bridge_cache["kt59"] = (inp_t @ w_teach_f32["k_proj"].to(DEVICE).T).to(torch.float16).cpu()
                del xt_s, inp_t, X_teach_cpu

            # 2) Compute Attention ctx on Student weights (with exact HF bidirectional sliding mask for VLM spans!)
            nkv = NKV_F if is_full else NKV_S
            hd = HD_F if is_full else HD_S
            theta = THETA_G if is_full else THETA_L
            rot_d = int(round(hd * (PART_G if is_full else PART_L)))
            cos_r, sin_r = make_rope(theta, rot_d, hd)
            gqa_rep = NH // nkv
            ctx_all = torch.empty((N_SEQ, SEQ_LEN, NH * hd), dtype=torch.float16)
            do_o, di_o = w_teach_f32["o_proj"].shape
            h_o = torch.zeros((di_o, di_o), device=DEVICE, dtype=torch.float32)
            xsq_o = torch.zeros((di_o,), device=DEVICE, dtype=torch.float32)
            x_o_ev = []
            for b0 in range(0, N_SEQ, 32):
                b1 = min(b0 + 32, N_SEQ)
                xb = X_stud_cpu[b0:b1].to(DEVICE)
                inp = rmsn_gpu(xb, norms_g["in"])
                q = F.linear(inp, w_stud_f16["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                k = F.linear(inp, w_stud_f16["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                v = k.clone() if (is_full and K_EQ_V) else F.linear(inp, w_stud_f16["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                qn = apply_rope(rmsn_gpu(q, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                kn = apply_rope(rmsn_gpu(k, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                vn = rmsn_gpu(v, None).transpose(1, 2)
                amask_s = None if is_full else make_sliding_attn_mask(b0, b1, vlm_spans, SEQ_LEN, DEVICE)
                if amask_s is not None:
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, dim=1), vn.repeat_interleave(gqa_rep, dim=1), attn_mask=amask_s, scale=1.0)
                else:
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, dim=1), vn.repeat_interleave(gqa_rep, dim=1), is_causal=True, scale=1.0)
                ctx = ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd)
                cf = ctx.reshape(-1, di_o).float()
                accum_hess_t4(h_o, xsq_o, cf)
                if len(x_o_ev) < 3 or b0 >= N_TEXT_SEQ:
                    if len(x_o_ev) < 4: x_o_ev.append(cf[:512].clone())
                ctx_all[b0:b1] = ctx.cpu()
            m_i, bs_i, rk_i = get_lm_spec(l_idx, "o_proj")
            ho_box = [h_o]; del h_o
            pb, inv_s, lr, w_o_eff = pack_matrix_gpu_gptq(
                w_teach_f32["o_proj"], ho_box, torch.sqrt((xsq_o / n_tok_l).clamp(min=1e-8)), m_i, bs_i, rk_i, torch.cat(x_o_ev, 0)[:2048]
            )
            nb, asy = p3.MODE_TO_T16C[m_i]
            t16c_pos = p3._write_mat_record(
                t16c_fp, t16c_pos, f"{p}self_attn.o_proj.weight", do_o, di_o,
                bs_i, nb, asy, True, pb, inv_s, lr,
            )
            lm_mats_count += 1
            w_stud_f16["o_proj"] = w_o_eff
            del ho_box, xsq_o, x_o_ev, pb, inv_s, lr

            # Propagate strictly held-out eval attention on Student weights (zero Hessian / zero SmoothQuant touch!)
            with torch.no_grad():
                _he_g = H_stud_eval.to(DEVICE)
                _he_in = rmsn_gpu(_he_g, norms_g["in"])
                _hq = F.linear(_he_in, w_stud_f16["q_proj"]).view(_he_g.shape[0], SEQ_LEN, NH, hd)
                _hk = F.linear(_he_in, w_stud_f16["k_proj"]).view(_he_g.shape[0], SEQ_LEN, nkv, hd)
                _hv = _hk.clone() if (is_full and K_EQ_V) else F.linear(_he_in, w_stud_f16["v_proj"]).view(_he_g.shape[0], SEQ_LEN, nkv, hd)
                _hqn = apply_rope(rmsn_gpu(_hq, norms_g["qn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                _hkn = apply_rope(rmsn_gpu(_hk, norms_g["kn"]), cos_r, sin_r, rot_d, hd).transpose(1, 2)
                _hvn = rmsn_gpu(_hv, None).transpose(1, 2)
                _hmask = None if is_full else make_sliding_attn_mask(0, _he_g.shape[0], heldout_spans, SEQ_LEN, DEVICE)
                if _hmask is not None:
                    _hctx = F.scaled_dot_product_attention(_hqn, _hkn.repeat_interleave(gqa_rep, dim=1), _hvn.repeat_interleave(gqa_rep, dim=1), attn_mask=_hmask, scale=1.0)
                else:
                    _hctx = F.scaled_dot_product_attention(_hqn, _hkn.repeat_interleave(gqa_rep, dim=1), _hvn.repeat_interleave(gqa_rep, dim=1), is_causal=True, scale=1.0)
                _hot = F.linear(_hctx.transpose(1, 2).contiguous().view(_he_g.shape[0], SEQ_LEN, NH * hd), w_o_eff)
                H_stud_eval_mid = (_he_g + rmsn_gpu(_hot, norms_g["pa"])).cpu()
                del _he_g, _he_in, _hq, _hk, _hv, _hqn, _hkn, _hvn, _hmask, _hctx, _hot

            # Compute xm & hm -> gate/up Hessians
            h_mlp = torch.zeros((D_MOD, D_MOD), device=DEVICE, dtype=torch.float32)
            xsq_mlp = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
            xm_all = torch.empty((N_SEQ, SEQ_LEN, D_MOD), dtype=torch.float16)
            x_m_ev = []
            for b0 in range(0, N_SEQ, 32):
                b1 = min(b0 + 32, N_SEQ)
                xb = X_stud_cpu[b0:b1].to(DEVICE)
                cb = ctx_all[b0:b1].to(DEVICE)
                ot = F.linear(cb, w_o_eff)
                xm = xb + rmsn_gpu(ot, norms_g["pa"])
                xm_all[b0:b1] = xm.cpu()
                hmf = rmsn_gpu(xm, norms_g["pf"]).reshape(-1, D_MOD).float()
                accum_hess_t4(h_mlp, xsq_mlp, hmf)
                if len(x_m_ev) < 3 or b0 >= N_TEXT_SEQ:
                    if len(x_m_ev) < 4: x_m_ev.append(hmf[:512].clone())
            del ctx_all, X_stud_cpu, w_o_eff
            # Release/offload attention weights before MLP GPTQ to minimize peak GPU VRAM
            for _apn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w_teach_f32.pop(_apn, None)
                if l_idx in (58, 59) and _apn in w_stud_f16:
                    w_stud_f16[_apn] = w_stud_f16[_apn].cpu()
                else:
                    w_stud_f16.pop(_apn, None)
            torch.cuda.empty_cache()

            xrms_mlp = torch.sqrt((xsq_mlp / n_tok_l).clamp(min=1e-8))
            xev_mlp = torch.cat(x_m_ev, 0)[:2048]
            for pn in ("gate_proj", "up_proj"):
                m_i, bs_i, rk_i = get_lm_spec(l_idx, pn)
                pb, inv_s, lr, w_eff = pack_matrix_gpu_gptq(w_teach_f32[pn].cpu(), h_mlp, xrms_mlp, m_i, bs_i, rk_i, xev_mlp)
                nb, asy = p3.MODE_TO_T16C[m_i]
                t16c_pos = p3._write_mat_record(
                    t16c_fp, t16c_pos, f"{p}mlp.{pn}.weight", w_teach_f32[pn].shape[0], D_MOD,
                    bs_i, nb, asy, True, pb, inv_s, lr,
                )
                lm_mats_count += 1
                w_stud_f16[pn] = w_eff
                w_teach_f32.pop(pn, None)
                del pb, inv_s, lr
            del h_mlp, xsq_mlp, xrms_mlp, xev_mlp, x_m_ev

            # Compute act & down_proj Hessian inline without allocating 5.6 GiB act_all in RAM!
            do_d, di_d = w_teach_f32["down_proj"].shape
            h_d = torch.zeros((di_d, di_d), device=DEVICE, dtype=torch.float32)
            xsq_d = torch.zeros((di_d,), device=DEVICE, dtype=torch.float32)
            x_d_ev = []
            for b0 in range(0, N_SEQ, 32):
                b1 = min(b0 + 32, N_SEQ)
                hm = rmsn_gpu(xm_all[b0:b1].to(DEVICE), norms_g["pf"])
                gt = F.linear(hm, w_stud_f16["gate_proj"]).float()
                ut = F.linear(hm, w_stud_f16["up_proj"]).float()
                act = (F.gelu(gt, approximate="tanh") * ut)
                af = act.reshape(-1, di_d)
                accum_hess_t4(h_d, xsq_d, af)
                if len(x_d_ev) < 3 or b0 >= N_TEXT_SEQ:
                    if len(x_d_ev) < 4: x_d_ev.append(af[:512].clone())
            # Offload gate_proj & up_proj (462 MB) to CPU during down_proj GPTQ, leaving max VRAM on GPU 0/1
            w_stud_f16["gate_proj"] = w_stud_f16["gate_proj"].cpu()
            w_stud_f16["up_proj"] = w_stud_f16["up_proj"].cpu()
            xrms_d = torch.sqrt((xsq_d / n_tok_l).clamp(min=1e-8))
            xev_d = torch.cat(x_d_ev, 0)[:2048]
            del xsq_d, x_d_ev
            torch.cuda.empty_cache()

            m_i, bs_i, rk_i = get_lm_spec(l_idx, "down_proj")
            hd_box = [h_d]; del h_d
            pb, inv_s, lr, w_d_eff = pack_matrix_gpu_gptq(
                w_teach_f32["down_proj"], hd_box, xrms_d, m_i, bs_i, rk_i, xev_d, out_device=DEVICE
            )
            nb, asy = p3.MODE_TO_T16C[m_i]
            t16c_pos = p3._write_mat_record(
                t16c_fp, t16c_pos, f"{p}mlp.down_proj.weight", do_d, di_d,
                bs_i, nb, asy, True, pb, inv_s, lr,
            )
            lm_mats_count += 1
            del hd_box, xrms_d, xev_d, w_teach_f32, pb, inv_s, lr

            lsc = float(norms_g["lsc"])
            if l_idx in (58, 59):
                _atomic_torch_save({
                    "w": {k: v.cpu() for k, v in w_stud_f16.items()} | {"down_proj": w_d_eff.cpu()},
                    "norms": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in norms_g.items()},
                }, Path(f"/tmp/l{l_idx}_stud.pt"))
            wg_gpu = w_stud_f16["gate_proj"].to(DEVICE)
            wu_gpu = w_stud_f16["up_proj"].to(DEVICE)
            with torch.no_grad():
                _hmg = H_stud_eval_mid.to(DEVICE)
                _hhm = rmsn_gpu(_hmg, norms_g["pf"])
                _hact = (F.gelu(F.linear(_hhm, wg_gpu).float(), approximate="tanh") * F.linear(_hhm, wu_gpu).float()).to(torch.float16)
                _hdt = F.linear(_hact, w_d_eff)
                H_stud_eval = ((_hmg.float() + rmsn_gpu(_hdt, norms_g["pff"]).float()) * lsc).to(torch.float16).cpu()
                del H_stud_eval_mid, _hmg, _hhm, _hact, _hdt
            X_stud_next = torch.empty((N_SEQ, SEQ_LEN, D_MOD), dtype=torch.float16)
            for b0 in range(0, N_SEQ, 32):
                b1 = min(b0 + 32, N_SEQ)
                xm_g = xm_all[b0:b1].to(DEVICE)
                hm = rmsn_gpu(xm_g, norms_g["pf"])
                act = (F.gelu(F.linear(hm, wg_gpu).float(), approximate="tanh") * F.linear(hm, wu_gpu).float()).to(torch.float16)
                dt_o = F.linear(act, w_d_eff)
                xb = ((xm_g.float() + rmsn_gpu(dt_o, norms_g["pff"]).float()) * lsc).to(torch.float16)
                X_stud_next[b0:b1] = xb.cpu()
            del xm_all, w_stud_f16, wg_gpu, wu_gpu, w_d_eff

            cos_l_txt = float(F.cosine_similarity(X_stud_next[:16].reshape(-1, D_MOD).float(), X_teach_next[:16].reshape(-1, D_MOD).float(), dim=-1).mean().item())
            cos_l_vlm = float(F.cosine_similarity(H_stud_eval.reshape(-1, D_MOD).float(), H_teach_eval.reshape(-1, D_MOD).float(), dim=-1).mean().item())
            X_teach_cpu, X_stud_cpu = X_teach_next, X_stud_next
            del X_teach_next, X_stud_next
            torch.cuda.empty_cache()

            if l_idx in (9, 19, 29, 39, 49, 57):
                t16c_fp.flush()
                os.fsync(t16c_fp.fileno())
                _save_to_work = shutil.disk_usage(OUT_DIR).free > 8_500_000_000
                _xt_dst = CKPT_X_TEACH_W if _save_to_work else CKPT_X_TEACH_TMP
                _xs_dst = CKPT_X_STUD_W if _save_to_work else CKPT_X_STUD_TMP
                _atomic_torch_save(X_teach_cpu, _xt_dst)
                _atomic_torch_save(X_stud_cpu, _xs_dst)
                _atomic_torch_save({
                    "next_layer": l_idx + 1,
                    "t16c_pos": t16c_pos,
                    "lm_mats_count": lm_mats_count,
                    "bridge_cache": bridge_cache,
                    "H_teach_eval": H_teach_eval,
                    "H_stud_eval": H_stud_eval,
                }, STAGE3_CKPT_PATH)
                gc.collect()
            save_progress({"stage": f"L{l_idx:02d}_done", "next_layer": l_idx + 1, "t16c_pos": t16c_pos, "stud_vs_teach_cos_txt": cos_l_txt, "stud_vs_teach_cos_vlm_heldout": cos_l_vlm})
            if l_idx % 5 == 0 or l_idx >= 55 or l_idx in (2, 4, 10, 14, 19):
                print(f"  [LM L{l_idx:02d}/59 {'F' if is_full else 'S'}] txt_cos={cos_l_txt:.5f} heldout_vlm_cos={cos_l_vlm:.5f} dt={time.perf_counter()-tl:.1f}s elap={time.perf_counter()-T0_ALL:.1f}s ram={host_ram_gb()[0]:.2f}GiB", flush=True)

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
    assert t16c_sz <= 6_420_000_000, f"Serialized .t16c exceeded budget: {t16c_sz:,} B"
    for _rm_p in (VIS_CKPT_PATH, VLM_SOFT_CACHE, STAGE3_CKPT_PATH, CKPT_X_TEACH_W, CKPT_X_STUD_W, CKPT_X_TEACH_TMP, CKPT_X_STUD_TMP):
        if _rm_p.is_file():
            _rm_p.unlink()
    del all_norms, lm_norms, vision_norms
    gc.collect()
    torch.cuda.empty_cache()
    save_progress({"stage": "stage3_t16c_complete", "unified_t16c_bytes": int(t16c_sz), "total_mats": total_mats, "total_norms": n_all_norms})
    print(f"[Stage 3 DONE] Unified .t16c flushed to {OUT_T16C} ({t16c_sz:,} B, {t16c_sz/1024**3:.3f} GiB) | ram={host_ram_gb()[0]:.2f}GiB", flush=True)

# ---------------------------------------------------------------------------
# 6. Train 300-Step Multimodal C8DB Encyclopedia (L58/L59 KV + Wiener + post_cell60 + Vision!)
# ---------------------------------------------------------------------------
print("\n[Stage 4] Training 300-step Multi-Layer Multimodal C8DB Encyclopedia (L58->L59 Causal + Wiener + post_cell60 + Vision)...", flush=True)
c8db_entries: List[dict] = []

def train_c8d_proj(name: str, x_tr: torch.Tensor, y_t_tr: torch.Tensor, y_s_tr: torch.Tensor,
                   x_va: torch.Tensor, y_t_va: torch.Tensor, y_s_va: torch.Tensor, rank: int = 32, steps: int = 300):
    cos0 = float(F.cosine_similarity(y_s_va, y_t_va, dim=-1).mean().item())
    m = c8d.Cuneiform8DPatternAdapter(d_out=y_t_tr.shape[1], d_in=x_tr.shape[1], rank=rank, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=2e-2)
    best_c, best_sd = cos0, None
    for st in range(steps):
        opt.zero_grad()
        idx_b = torch.randint(0, x_tr.shape[0], (min(2048, x_tr.shape[0]),), device=DEVICE)
        delta, _, lb = m(x_tr[idx_b])
        p = y_s_tr[idx_b] + delta
        loss = F.mse_loss(p, y_t_tr[idx_b]) + 0.3 * (1.0 - F.cosine_similarity(p, y_t_tr[idx_b], dim=-1).mean()) + 0.005 * lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (st + 1) % 60 == 0:
            with torch.no_grad():
                dv, _, _ = m(x_va)
                cv = float(F.cosine_similarity(y_s_va + dv, y_t_va, dim=-1).mean().item())
                if cv > best_c:
                    best_c = cv
                    best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    if best_sd is None:
        print(f"  [C8DB-KV] {name}: val_cos base={cos0:.5f} (no gain, skipped)", flush=True)
        return None, None, None
    m_cpu = m.cpu()
    m_cpu.load_state_dict(best_sd)
    cents, experts = m_cpu.export_quantized_arrays()
    dq = eval_c8d_gpu(x_va, cents, experts)
    cos_q = float(F.cosine_similarity(y_s_va + dq, y_t_va, dim=-1).mean().item())
    keep = cos_q > cos0 + 0.0005
    print(f"  [C8DB-KV] {name}: val_cos {cos0:.5f} -> {cos_q:.5f} (300 steps, keep={keep})", flush=True)
    if not keep:
        return None, None, None
    exp_r = [c8d._c6d_radicals("kv", "head", "shared", "native", "layer", "positive", e) for e in range(8)]
    ent = {"name": name, "d_out": y_t_tr.shape[1], "d_in": x_tr.shape[1], "rank": rank, "top_k": 2,
           "c6d": exp_r[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents, "expert_radicals": exp_r, "experts": experts}
    return ent, cents, experts

# Add Vision embed_vision C8DB entry only if validated improved
if vis_keep_c8d and cents_vis is not None:
    exp_rv = [c8d._c6d_radicals("vision", "patch", "soft", "native", "layer", "positive", e) for e in range(8)]
    c8db_entries.append({
        "name": "model.embed_vision.embedding_projection.weight", "d_out": 5376, "d_in": 1152, "rank": 32, "top_k": 2,
        "c6d": exp_rv[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents_vis, "expert_radicals": exp_rv, "experts": experts_vis,
    })

def forward_adapted_student_layer(
    X_in_cpu: torch.Tensor, ckpt_path: str, l_idx: int,
    adp_k: Optional[Tuple] = None, adp_v: Optional[Tuple] = None,
    capture_sub_idx: Optional[np.ndarray] = None,
    inplace: bool = False, capture_only: bool = False,
    spans_dict: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    w_g = {k: v.to(DEVICE) for k, v in ckpt["w"].items()}
    n_g = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in ckpt["norms"].items()}
    is_full = LAYER_TYPES[l_idx] == "full_attention"
    nkv = NKV_F if is_full else NKV_S
    hd = HD_F if is_full else HD_S
    theta = THETA_G if is_full else THETA_L
    rot_d = int(round(hd * (PART_G if is_full else PART_L)))
    hr = rot_d // 2
    inv_f = 1.0 / (theta ** (torch.arange(0, 2 * hr, 2, device=DEVICE, dtype=torch.float32) / float(hd)))
    ang = torch.arange(SEQ_LEN, device=DEVICE, dtype=torch.float32)[:, None] * inv_f[None, :]
    cos_r, sin_r = torch.cos(ang).to(torch.float16), torch.sin(ang).to(torch.float16)
    gqa_rep = NH // nkv
    lsc = float(n_g["lsc"])
    def rms_l(x, w=None):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return (y * w.float() if w is not None else y).to(torch.float16)
    def rope_l(t):
        hh = hd // 2
        c, s = cos_r[None, :, None, :], sin_r[None, :, None, :]
        x0, x1 = t[..., :hr], t[..., hh : hh + hr]
        o = t.clone()
        o[..., :hr] = x0 * c - x1 * s
        o[..., hh : hh + hr] = x0 * s + x1 * c
        return o

    if capture_only:
        assert capture_sub_idx is not None
        idx_all = torch.as_tensor(capture_sub_idx, dtype=torch.long)
        cap_in, cap_k = [], []
        with torch.no_grad():
            for s0 in range(0, idx_all.numel(), 32):
                sel = idx_all[s0 : min(s0 + 32, idx_all.numel())]
                xb = X_in_cpu[sel].to(DEVICE)
                inp = rms_l(xb, n_g["in"])
                k2 = F.linear(inp.reshape(-1, D_MOD), w_g["k_proj"]).float()
                if adp_k is not None:
                    k2 = k2 + eval_c8d_gpu(inp.reshape(-1, D_MOD), adp_k[0], adp_k[1])
                cap_in.append(inp.cpu())
                cap_k.append(k2.reshape(sel.numel(), SEQ_LEN, -1).to(torch.float16).cpu())
        cap = {"in": torch.cat(cap_in, 0), "k": torch.cat(cap_k, 0)}
        del ckpt, w_g, n_g
        torch.cuda.empty_cache()
        return None, cap

    X_out = X_in_cpu if inplace else torch.empty_like(X_in_cpu)
    sub_set = set(capture_sub_idx.tolist()) if capture_sub_idx is not None else set()
    cap_in, cap_k = [], []
    with torch.no_grad():
        for b0 in range(0, X_in_cpu.shape[0], 32):
            b1 = min(b0 + 32, X_in_cpu.shape[0])
            xb = X_in_cpu[b0:b1].to(DEVICE)
            inp = rms_l(xb, n_g["in"])
            inp2 = inp.reshape(-1, D_MOD)
            q = F.linear(inp, w_g["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
            k2 = F.linear(inp2, w_g["k_proj"]).float()
            if adp_k is not None:
                k2 = k2 + eval_c8d_gpu(inp2, adp_k[0], adp_k[1])
            if capture_sub_idx is not None:
                for gi in range(b0, b1):
                    if gi in sub_set:
                        lo = gi - b0
                        cap_in.append(inp[lo].cpu())
                        cap_k.append(k2.reshape(b1 - b0, SEQ_LEN, -1)[lo].to(torch.float16).cpu())
            k = k2.to(torch.float16).view(b1 - b0, SEQ_LEN, nkv, hd)
            if is_full and K_EQ_V:
                v = k.clone()
            else:
                v2 = F.linear(inp2, w_g["v_proj"]).float()
                if adp_v is not None:
                    v2 = v2 + eval_c8d_gpu(inp2, adp_v[0], adp_v[1])
                v = v2.to(torch.float16).view(b1 - b0, SEQ_LEN, nkv, hd)
            qn = rope_l(rms_l(q, n_g["qn"])).transpose(1, 2)
            kn = rope_l(rms_l(k, n_g["kn"])).transpose(1, 2)
            vn = rms_l(v, None).transpose(1, 2)
            amask_a = None if (is_full or not spans_dict) else make_sliding_attn_mask(b0, b1, spans_dict, SEQ_LEN, DEVICE)
            if amask_a is not None:
                ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), attn_mask=amask_a, scale=1.0)
            else:
                ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
            ot = F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_g["o_proj"])
            xm = xb + rms_l(ot, n_g["pa"])
            hm = rms_l(xm, n_g["pf"])
            act = (F.gelu(F.linear(hm, w_g["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_g["up_proj"]).float()).to(torch.float16)
            dt_o = F.linear(act, w_g["down_proj"])
            X_out[b0:b1] = ((xm.float() + rms_l(dt_o, n_g["pff"]).float()) * lsc).to(torch.float16).cpu()
    cap = {"in": torch.cat(cap_in, 0), "k": torch.cat(cap_k, 0)} if capture_sub_idx is not None else None
    del ckpt, w_g, n_g
    torch.cuda.empty_cache()
    return X_out, cap

# 1) Fit L58.k & L58.v adapters, then causally re-run L58 -> L59 in-place (< 8.5 GiB Host RAM!)
adp_k58, adp_v58, adp_k59 = None, None, None
for attr_target, nm_kv, xin_k, yt_k, ys_k, pop_x in [
    ("k58", "model.language_model.layers.58.self_attn.k_proj.weight", "in_58_stud", "kt58", "ks58_base", False),
    ("v58", "model.language_model.layers.58.self_attn.v_proj.weight", "in_58_stud", "vt58", "vs58_base", True),
]:
    xk_t = bridge_cache.pop(xin_k) if pop_x else bridge_cache[xin_k]
    xk, ytk, ysk = xk_t.float(), bridge_cache.pop(yt_k).float(), bridge_cache.pop(ys_k).float()
    del xk_t
    ntr_k = (xk.shape[0] * 4) // 5
    ent, c_np, e_np = train_c8d_proj(
        nm_kv, xk[:ntr_k].to(DEVICE), ytk[:ntr_k].to(DEVICE), ysk[:ntr_k].to(DEVICE),
        xk[ntr_k:].to(DEVICE), ytk[ntr_k:].to(DEVICE), ysk[ntr_k:].to(DEVICE), rank=32, steps=300
    )
    if ent is not None:
        c8db_entries.append(ent)
        if attr_target == "k58": adp_k58 = (c_np, e_np)
        else: adp_v58 = (c_np, e_np)
    del xk, ytk, ysk
    gc.collect()
    torch.cuda.empty_cache()

cos_l59_base = float(F.cosine_similarity(X_stud_cpu[-16:].reshape(-1, D_MOD).float(), X_teach_cpu[-16:].reshape(-1, D_MOD).float(), dim=-1).mean().item())
del X_stud_cpu
gc.collect()

sub_idx_59 = np.concatenate([
    np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int),
    np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int),
])
x_pre58 = torch.load("/tmp/x_stud_pre58.pt", map_location="cpu", weights_only=True)
Path("/tmp/x_stud_pre58.pt").unlink(missing_ok=True)
x_59_in, _ = forward_adapted_student_layer(x_pre58, "/tmp/l58_stud.pt", 58, adp_k58, adp_v58, capture_sub_idx=None, inplace=True, spans_dict=vlm_spans)
assert x_59_in is not None
del x_pre58
h_ev_pre58 = torch.load("/tmp/h_stud_eval_pre58.pt", map_location="cpu", weights_only=True)
Path("/tmp/h_stud_eval_pre58.pt").unlink(missing_ok=True)
h_ev_59_in, _ = forward_adapted_student_layer(h_ev_pre58, "/tmp/l58_stud.pt", 58, adp_k58, adp_v58, capture_sub_idx=None, inplace=True, spans_dict=heldout_spans)
assert h_ev_59_in is not None
del h_ev_pre58
Path("/tmp/l58_stud.pt").unlink(missing_ok=True)
gc.collect()

# Capture adapted L59 inputs & K base for L59.k training without allocating X_out (2.91 GiB saved!)
_, cap59 = forward_adapted_student_layer(x_59_in, "/tmp/l59_stud.pt", 59, None, None, capture_sub_idx=sub_idx_59, capture_only=True)
assert cap59 is not None
xk59 = cap59["in"].reshape(-1, D_MOD).float()
ytk59 = bridge_cache.pop("kt59").float()
ysk59 = cap59["k"].reshape(-1, ytk59.shape[1]).float()
del cap59, bridge_cache
ntr_59 = (xk59.shape[0] * 4) // 5
ent59, c59_np, e59_np = train_c8d_proj(
    "model.language_model.layers.59.self_attn.k_proj.weight",
    xk59[:ntr_59].to(DEVICE), ytk59[:ntr_59].to(DEVICE), ysk59[:ntr_59].to(DEVICE),
    xk59[ntr_59:].to(DEVICE), ytk59[ntr_59:].to(DEVICE), ysk59[ntr_59:].to(DEVICE), rank=32, steps=300
)
if ent59 is not None:
    c8db_entries.append(ent59)
    adp_k59 = (c59_np, e59_np)
del xk59, ytk59, ysk59
gc.collect()
torch.cuda.empty_cache()

# Incremental .c8d checkpoint after Stage 4 KV adapters
c8d.write_c8db_file(OUT_C8D, [("model.language_model.norm.weight", fn_orig_cpu.numpy().astype("<f4"))], c8db_entries)
save_progress({"stage": "stage4_kv_done", "c8d_entries": [e["name"] for e in c8db_entries]})

X_stud_cpu, _ = forward_adapted_student_layer(x_59_in, "/tmp/l59_stud.pt", 59, adp_k59, None, capture_sub_idx=None, inplace=True, spans_dict=vlm_spans)
assert X_stud_cpu is not None
del x_59_in
H_stud_eval, _ = forward_adapted_student_layer(h_ev_59_in, "/tmp/l59_stud.pt", 59, adp_k59, None, capture_sub_idx=None, inplace=True, spans_dict=heldout_spans)
assert H_stud_eval is not None
del h_ev_59_in
Path("/tmp/l59_stud.pt").unlink(missing_ok=True)
gc.collect()
cos_l59_adapted = float(F.cosine_similarity(X_stud_cpu[-16:].reshape(-1, D_MOD).float(), X_teach_cpu[-16:].reshape(-1, D_MOD).float(), dim=-1).mean().item())
print(f"  [Causal L58->L59 C8DB Parity] L59 VLM state_cos: {cos_l59_base:.5f} -> {cos_l59_adapted:.5f} | ram={host_ram_gb()[0]:.2f}GiB", flush=True)

# Final RMSNorm Wiener fit (streaming across all 270,336 tokens cleanly!) + balanced multimodal answer sampling for post_cell60
fn_g = fn_orig_cpu.to(DEVICE)
num_w = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
den_w = torch.zeros((D_MOD,), device=DEVICE, dtype=torch.float32)
val_sq_o, val_sq_w, val_cnt = 0.0, 0.0, 0
with torch.no_grad():
    for b0 in range(0, N_SEQ, 64):
        b1 = min(b0 + 64, N_SEQ)
        xt_b = rmsn_gpu(X_teach_cpu[b0:b1].to(DEVICE), fn_g).reshape(-1, D_MOD).float()
        xu_b = rmsn_gpu(X_stud_cpu[b0:b1].to(DEVICE), None).reshape(-1, D_MOD).float()
        is_val = (torch.arange(xt_b.shape[0], device=DEVICE) % 10 == 0)
        num_w.add_((xu_b[~is_val] * xt_b[~is_val]).sum(dim=0))
        den_w.add_((xu_b[~is_val] * xu_b[~is_val]).sum(dim=0))
    n_tr_w = N_SEQ * SEQ_LEN * 0.9
    fn_wiener = (((num_w / n_tr_w) + 0.05 * fn_g) / ((den_w / n_tr_w) + 0.05).clamp(min=1e-8))
    for b0 in range(0, N_SEQ, 64):
        b1 = min(b0 + 64, N_SEQ)
        xt_b = rmsn_gpu(X_teach_cpu[b0:b1].to(DEVICE), fn_g).reshape(-1, D_MOD).float()
        xu_b = rmsn_gpu(X_stud_cpu[b0:b1].to(DEVICE), None).reshape(-1, D_MOD).float()
        is_val = (torch.arange(xt_b.shape[0], device=DEVICE) % 10 == 0)
        val_sq_o += float(((xu_b[is_val] * fn_g - xt_b[is_val]) ** 2).sum().item())
        val_sq_w += float(((xu_b[is_val] * fn_wiener - xt_b[is_val]) ** 2).sum().item())
        val_cnt += int(is_val.sum().item()) * D_MOD
mse_o, mse_w = val_sq_o / max(val_cnt, 1), val_sq_w / max(val_cnt, 1)
fn_eff_g = fn_wiener if mse_w < mse_o else fn_g
print(f"  [Wiener RMSNorm] val_mse {mse_o:.5f} -> {mse_w:.5f} using={'Wiener' if mse_w < mse_o else 'orig'}", flush=True)

# Extract balanced multimodal answer tokens (all VLM caption answers up to 8,192 + 24,576 text answers -> ~32,768 total)
vlm_ans_pos = np.flatnonzero(all_vlm_ans_mask.reshape(-1))
txt_ans_pos = np.flatnonzero((all_ans_mask & (~all_vlm_ans_mask)).reshape(-1))
sel_vlm = vlm_ans_pos[np.linspace(0, vlm_ans_pos.size - 1, min(8192, vlm_ans_pos.size), dtype=int)] if vlm_ans_pos.size > 0 else np.array([], dtype=int)
n_txt_need = min(32768 - sel_vlm.size, txt_ans_pos.size)
sel_txt = txt_ans_pos[np.linspace(0, txt_ans_pos.size - 1, n_txt_need, dtype=int)]
sel_pos_np = np.sort(np.concatenate([sel_txt, sel_vlm]))
print(f"  [post_cell60 Samples] Total={sel_pos_np.size:,} (text_ans={sel_txt.size:,} + vlm_caption_ans={sel_vlm.size:,})", flush=True)
sel_seq = sel_pos_np // SEQ_LEN
sel_tok = sel_pos_np % SEQ_LEN
ht_ans_g = rmsn_gpu(X_teach_cpu[sel_seq, sel_tok].to(DEVICE).unsqueeze(1), fn_g).squeeze(1).float()
hs_ans_g = rmsn_gpu(X_stud_cpu[sel_seq, sel_tok].to(DEVICE).unsqueeze(1), fn_eff_g).squeeze(1).float()
nxt_ans_g = torch.from_numpy(all_seq_nxt.reshape(-1)[sel_pos_np]).to(DEVICE, dtype=torch.long)
del X_teach_cpu, X_stud_cpu
gc.collect()
torch.cuda.empty_cache()

_atomic_torch_save({
    "ht_ans_fp16": ht_ans_g.half().cpu(),
    "hs_ans_fp16": hs_ans_g.half().cpu(),
    "nxt_ans_cpu": nxt_ans_g.cpu(),
    "fn_eff_cpu": fn_eff_g.cpu(),
}, CKPT_DIR / "stage4_l59_ans.pt")
save_progress({"stage": "stage4_ans_cached", "n_ans": int(hs_ans_g.shape[0])})

# GPU streaming Top-32 Teacher/Student candidate sets across all 262,144 vocab rows in 8192-row chunks!
with safe_open(sf_path, framework="pt", device="cpu") as sf:
    emb_gpu = sf.get_tensor("model.language_model.embed_tokens.weight").to(DEVICE, dtype=torch.float16)
N_ANS = hs_ans_g.shape[0]
TAU = 1.5
t_lse = torch.full((N_ANS,), -1e9, device=DEVICE)
q_lse = torch.full((N_ANS,), -1e9, device=DEVICE)
top_ids_t_list, top_ids_q_list = [], []
for r0 in range(0, emb_gpu.shape[0], 8192):
    W_c = emb_gpu[r0 : r0 + 8192].float()
    zt = 30.0 * torch.tanh((ht_ans_g @ W_c.T) / 30.0) / TAU
    zq = 30.0 * torch.tanh((hs_ans_g @ W_c.T) / 30.0) / TAU
    t_lse = torch.logaddexp(t_lse, torch.logsumexp(zt, dim=-1))
    q_lse = torch.logaddexp(q_lse, torch.logsumexp(zq, dim=-1))
    top_ids_t_list.append((torch.topk(zt, 32, dim=-1).indices + r0).reshape(-1))
    top_ids_q_list.append((torch.topk(zq, 32, dim=-1).indices + r0).reshape(-1))
    del W_c, zt, zq

# Keep compact top-4096 most frequent candidate columns (~84 MB VRAM, covers 99.9%+ probability mass!)
all_top_merged = torch.cat(top_ids_t_list + top_ids_q_list)
uniq_ids, counts = torch.unique(all_top_merged, return_counts=True)
keep_k = min(4096, uniq_ids.numel())
best_cols = uniq_ids[torch.topk(counts, keep_k).indices]
cand_ids = torch.unique(torch.cat([best_cols, nxt_ans_g]))
W_cand_g = emb_gpu[cand_ids].float()
del emb_gpu, top_ids_t_list, top_ids_q_list, all_top_merged, uniq_ids, counts, best_cols
torch.cuda.empty_cache()
id_map = {int(tid): ci for ci, tid in enumerate(cand_ids.tolist())}
y_star_g = torch.tensor([id_map[int(tid)] for tid in nxt_ans_g.tolist()], device=DEVICE, dtype=torch.long)

OFFLOAD_C60 = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu")
t_top1_g = torch.empty((N_ANS,), device=DEVICE, dtype=torch.long)
log_tail_q0 = torch.empty((N_ANS,), device=DEVICE, dtype=torch.float32)
p_teach_store = torch.empty((N_ANS, int(cand_ids.numel()) + 1), device=OFFLOAD_C60, dtype=torch.float16)

for c0 in range(0, N_ANS, 4096):
    c1 = min(c0 + 4096, N_ANS)
    zt_c = 30.0 * torch.tanh((ht_ans_g[c0:c1] @ W_cand_g.T) / 30.0) / TAU
    zq_c = 30.0 * torch.tanh((hs_ans_g[c0:c1] @ W_cand_g.T) / 30.0) / TAU
    t_top1_g[c0:c1] = zt_c.argmax(dim=-1)
    t_cand_lse_c = torch.logsumexp(zt_c, dim=-1)
    q_cand_lse_c = torch.logsumexp(zq_c, dim=-1)
    log_tail_t_c = t_lse[c0:c1] + torch.log((1.0 - torch.exp((t_cand_lse_c - t_lse[c0:c1]).clamp(max=0.0))).clamp(min=1e-12))
    log_tail_q0[c0:c1] = q_lse[c0:c1] + torch.log((1.0 - torch.exp((q_cand_lse_c - q_lse[c0:c1]).clamp(max=0.0))).clamp(min=1e-12))
    p_teach_store[c0:c1] = F.softmax(torch.cat([zt_c, log_tail_t_c.unsqueeze(1)], dim=-1), dim=-1).to(device=OFFLOAD_C60, dtype=torch.float16)
    del zt_c, zq_c, t_cand_lse_c, q_cand_lse_c, log_tail_t_c
del t_lse, q_lse
torch.cuda.empty_cache()

# 80/20 Train/Val split on 32,768 real multi-corpus answer tokens
va_a = (torch.arange(N_ANS, device=DEVICE) % 5 == 0)
tr_a = ~va_a
model60 = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=D_MOD, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(DEVICE)
opt60 = torch.optim.AdamW(model60.parameters(), lr=3e-3, weight_decay=3e-2)
sched60 = torch.optim.lr_scheduler.CosineAnnealingLR(opt60, T_max=300, eta_min=3e-4)

def eval_c60(x_sub: torch.Tensor, idx_sub: torch.Tensor) -> Tuple[float, float, float, float]:
    with torch.no_grad():
        idx_t = torch.nonzero(idx_sub, as_tuple=False).squeeze(1) if idx_sub.dtype == torch.bool else idx_sub
        n_s = int(idx_t.numel())
        assert x_sub.shape[0] == n_s
        top1_s, agr_s, marg_s, kl_s = 0.0, 0.0, 0.0, 0.0
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
            pt_b = p_teach_store[idx_b.to(OFFLOAD_C60)].to(DEVICE, dtype=torch.float32)
            kl_s += float(((pt_b * (torch.log(pt_b.clamp(min=1e-12)) - F.log_softmax(zw, dim=-1))).sum(dim=-1) * (TAU * TAU)).sum().item())
            del zc, pred_b, zm, zw, pt_b
        denom_n = float(max(n_s, 1))
        return kl_s / denom_n, top1_s / denom_n, agr_s / denom_n, marg_s / denom_n

kl0, acc0, agr0, marg0 = eval_c60(hs_ans_g[va_a], va_a)
print(f"  [post_cell60 Before] Val (n={int(va_a.sum())}): KL={kl0:.4f} Top1={acc0*100:.1f}% Agree={agr0*100:.1f}% Margin={marg0:+.3f}", flush=True)

tr_indices = torch.nonzero(tr_a, as_tuple=False).squeeze(1)
best_score, best_sd60 = -1e9, None
for st in range(300):
    opt60.zero_grad()
    perm_b = tr_indices[torch.randperm(tr_indices.numel(), device=DEVICE)[:2048]]
    d_ans, _, lb = model60(hs_ans_g[perm_b])
    hp = hs_ans_g[perm_b] + d_ans
    zc = 30.0 * torch.tanh((hp @ W_cand_g.T) / 30.0)
    zw = torch.cat([zc / TAU, log_tail_q0[perm_b].unsqueeze(1)], dim=1)
    pt_b = p_teach_store[perm_b.to(OFFLOAD_C60)].to(DEVICE, dtype=torch.float32)
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
    sched60.step()
    if (st + 1) % 50 == 0 or st == 0:
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
d_va_np_chk = torch.from_numpy(c8d.eval_quantized_adapter_np(hs_ans_g[va_a][:64].cpu().numpy(), cents60, experts60, 2, 0.5, 1.0)).to(DEVICE)
assert float((d_va_q[:64] - d_va_np_chk).abs().max().item()) < 1e-3, "GPU/CPU C8DB parity mismatch!"
kl_q, acc_q, agr_q, marg_q = eval_c60(hs_ans_g[va_a] + d_va_q, va_a)
print(f"  [post_cell60 Quantized INT8] Val KL: {kl0:.4f}->{kl_q:.4f}, Top1: {acc0*100:.1f}%->{acc_q*100:.1f}%, Agree: {agr0*100:.1f}%->{agr_q*100:.1f}%, Margin: {marg0:+.3f}->{marg_q:+.3f}", flush=True)
if kl_q <= kl0:
    exp_r60 = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
    c8db_entries.append({
        "name": "model.language_model.norm.post_cell60", "d_out": D_MOD, "d_in": D_MOD, "rank": 32, "top_k": 2,
        "c6d": exp_r60[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents60, "expert_radicals": exp_r60, "experts": experts60,
    })

# ---------------------------------------------------------------------------
# 6b. Stage 5: Strictly Held-Out COCO val[0..3] Visual QA Verification (Teacher vs V4.3 Student + C8DB)
# ---------------------------------------------------------------------------
print("\n[Stage 5] Evaluating strictly held-out COCO val[0..3] (NEVER in Hessians/C8DB) Teacher vs V4.3 Student...", flush=True)
heldout_vlm_report = []
with safe_open(sf_path, framework="pt", device="cpu") as sf_ev:
    emb_ev = sf_ev.get_tensor("model.language_model.embed_tokens.weight").to(DEVICE, dtype=torch.float16)
with torch.no_grad():
    for hi, cap_gt in enumerate(heldout_captions):
        p0 = heldout_ans_starts[hi]
        plen = heldout_ans_lens[hi]
        ht_hi = rmsn_gpu(H_teach_eval[hi, p0 : p0 + plen].to(DEVICE).unsqueeze(0), fn_g).squeeze(0).float()
        hs_hi = rmsn_gpu(H_stud_eval[hi, p0 : p0 + plen].to(DEVICE).unsqueeze(0), fn_eff_g).squeeze(0).float()
        if kl_q <= kl0:
            hs_hi = hs_hi + eval_c8d_gpu(hs_hi, cents60, experts60)
        h_cos_hi = float(F.cosine_similarity(hs_hi, ht_hi, dim=-1).mean().item())
        # Exact full-262K-vocab logits on answer slice [plen, 262144] in chunks
        t_top5_all, s_top5_all = [], []
        t_best_v = torch.full((plen, 5), -1e9, device=DEVICE)
        t_best_i = torch.zeros((plen, 5), device=DEVICE, dtype=torch.long)
        s_best_v = torch.full((plen, 5), -1e9, device=DEVICE)
        s_best_i = torch.zeros((plen, 5), device=DEVICE, dtype=torch.long)
        for r0 in range(0, emb_ev.shape[0], 16384):
            wc = emb_ev[r0 : r0 + 16384].float()
            lt = 30.0 * torch.tanh((ht_hi @ wc.T) / 30.0)
            ls = 30.0 * torch.tanh((hs_hi @ wc.T) / 30.0)
            tv, ti = torch.topk(lt, 5, dim=-1)
            sv, si = torch.topk(ls, 5, dim=-1)
            ti = ti + r0; si = si + r0
            mv = torch.cat([t_best_v, tv], dim=-1); mi = torch.cat([t_best_i, ti], dim=-1)
            t_best_v, pick = torch.topk(mv, 5, dim=-1); t_best_i = mi.gather(1, pick)
            mv2 = torch.cat([s_best_v, sv], dim=-1); mi2 = torch.cat([s_best_i, si], dim=-1)
            s_best_v, pick2 = torch.topk(mv2, 5, dim=-1); s_best_i = mi2.gather(1, pick2)
            del wc, lt, ls
        t1 = t_best_i[:, 0]
        s1 = s_best_i[:, 0]
        top1_agr = float((s1 == t1).float().mean().item())
        top5_agr = float((s_best_i == t1.unsqueeze(1)).any(dim=-1).float().mean().item())
        t_first5 = [f"{int(tid)}:{tk.decode([int(tid)])!r}" for tid in t_best_i[0].tolist()]
        s_first5 = [f"{int(tid)}:{tk.decode([int(tid)])!r}" for tid in s_best_i[0].tolist()]
        s_greedy_caption = tk.decode([int(x) for x in s1.tolist()])
        t_greedy_caption = tk.decode([int(x) for x in t1.tolist()])
        rec_h = {
            "val_idx": hi,
            "gt_caption": cap_gt,
            "ans_cos": round(h_cos_hi, 5),
            "ans_top1_agree": round(top1_agr, 4),
            "ans_top5_agree": round(top5_agr, 4),
            "teacher_first_top5": t_first5,
            "student_first_top5": s_first5,
            "teacher_forced_pred": t_greedy_caption,
            "student_forced_pred": s_greedy_caption,
        }
        heldout_vlm_report.append(rec_h)
        print(f"  [Held-Out val[{hi}]] cos={h_cos_hi:.5f} top1_agr={top1_agr*100:.1f}% top5_agr={top5_agr*100:.1f}% | GT={cap_gt!r}", flush=True)
        print(f"    Teacher top5[0]={t_first5} | Student top5[0]={s_first5}", flush=True)
        print(f"    Student TF-pred={s_greedy_caption!r}", flush=True)
del emb_ev
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 7. Finalize C8DB .c8d & Write Completion Report to /kaggle/working/
# ---------------------------------------------------------------------------
c8d_norms = [("model.language_model.norm.weight", fn_eff_g.detach().cpu().numpy().astype("<f4"))]
c8d_sz = c8d.write_c8db_file(OUT_C8D, c8d_norms, c8db_entries)

t16c_sz = OUT_T16C.stat().st_size
assert t16c_sz <= 6_420_000_000, f"Serialized .t16c exceeded budget: {t16c_sz:,} B"
if VIS_META_PATH.is_file():
    VIS_META_PATH.unlink()
report = {
    "version": "V4.3-VLM",
    "total_time_s": round(time.perf_counter() - T0_ALL, 2),
    "unified_t16c": str(OUT_T16C),
    "unified_t16c_bytes": t16c_sz,
    "unified_t16c_gib": round(t16c_sz / (1024**3), 4),
    "lm_mats": lm_mats_count,
    "vision_mats": n_vis_mats,
    "total_mats": total_mats,
    "total_norms": n_all_norms,
    "c8db_path": str(OUT_C8D),
    "c8db_bytes": c8d_sz,
    "c8db_entries": [e["name"] for e in c8db_entries],
    "vision_val_cos_before": vis_cos_before,
    "vision_val_cos_after": vis_cos_after,
    "heldout_vis_soft_cos": heldout_vis_cos,
    "l59_state_cos_before_c8d": cos_l59_base,
    "l59_state_cos_after_c8d": cos_l59_adapted,
    "post_cell60_val_kl_before": kl0,
    "post_cell60_val_kl_after": kl_q,
    "post_cell60_val_top1_before": acc0,
    "post_cell60_val_top1_after": acc_q,
    "post_cell60_val_agree_before": agr0,
    "post_cell60_val_agree_after": agr_q,
    "tokens_calibrated": int(all_seq_ids.size),
    "vlm_sequences_calibrated": N_VLM_SEQ,
    "vlm_caption_ans_tokens": int(vlm_ans_count),
    "dense_turns_packed": examples_packed,
    "vision_frames_calibrated": vis_frames_calibrated,
    "clean_dataset_examples": len(clean_turns),
    "heldout_coco_val0_3": heldout_vlm_report,
}
(OUT_DIR / "stage2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
save_progress({"stage": "complete", "unified_t16c_bytes": t16c_sz, "c8db_bytes": c8d_sz})
print("\n[STAGE 2 COMPLETE]", json.dumps(report, indent=2), flush=True)

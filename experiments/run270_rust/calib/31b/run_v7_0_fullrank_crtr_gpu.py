#!/usr/bin/env python3
"""
Gemma 4 31B V7.0 Full-Rank Hybrid (`~5.64 GiB`) + Live CRTR v1 / Softer MoD (`q20`)
+ 3-Bit Vision + Target-Aligned Assistant-V7 (`BHDR d_byte=384`, `student_argmax`).

Architecture & Engineering Upgrades Over V6.1 (`~1.13 bpw`, `2/15`) and V4.3 VLM (`5.92 GiB`):
  1. Zero Rank-Truncating Bottleneck: All 411 LM matrices use full-rank Cholesky Hessian
     GPTQ + Dual-Ridge ActALS (no 2048-dim factorized truncation) in `~5.642 GiB`
     (`<= 5.75 GiB` hard Pi 4 guard, leaving `~1.96 GiB` free RAM on Pi 4 `7.6 GiB`).
  2. Zero 1-Bit Attention Anywhere: All `q/k/v/o` projections across L00..L59 are
     >= 5-trit ternary (3-bit on L58..L59 and critical `ATTN_3B_LAYERS`).
  3. Permutation-Packed Row-Major 1-Bit MLPs: Teacher co-activation `perm_mlp` packs
     high-energy GeGLU channels before SmoothQuant/GPTQ, and all 153 1-bit sliding MLPs
     are transposed to ARMv8 NEON Row-Major (`1 | 0x40`) for `CRTR` block skipping.
  4. Real `CRTR` v1 + Softer `MoD (q20)`: `train_cortical_router.py` fits 84-block MLP
     predictors on `hm` and `MoD` gates on `inp` (`mod_q20` threshold vs V6.1's aggressive
     `q40`) directly from quantized V7.0 student activations (`~10.86 MB`).
  5. Target-Aligned `Assistant-V7` (`BHDR d_byte=384`): Aligns 4-layer Assistant on `V7.0`'s
     post-C8DB `L58/L59` KV + logits (`300` steps, `128` seqs), packs embed-free Q2/Q4/Q8
     bodies (`54 / 102 / 198 MiB`), and distills `BHDR` (`d_byte=384`, `2400` steps) directly
     against `V7.0` student full-262K argmax + top-64 probabilities (`arXiv:2609.12303` F.1).
  6. Fail-Closed Checkpointing & Holdout Hygiene: `.partial`-first Drive sync, non-extending
     resume verification, COCO `val[0..63]` exclusion, and full blacklist audit.
"""

from __future__ import annotations

import argparse
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
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
import repack_t16c_rowmajor as rrm  # noqa: E402
import train_assistant_v7_byte_head as bhmod  # noqa: E402
import train_c8d_pattern_bank as c8d  # noqa: E402
import train_cortical_router as crtr  # noqa: E402
import v4_corpus as vc  # noqa: E402

T16C_GAUSS_BIT = 0x80
T16C_ROWMAJOR_BIT = 0x40

FULL_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}
ATTN_3B_LAYERS = {2, 4, 10, 14, 19, 58, 59}
EXTRA_BLACKLIST: Tuple[str, ...] = (
    "12 * 8", "12*8", "547", "193", "motorcycle", "motorbike", "honda",
    "5x + 3 = 38", "18 kilometers per hour", "cold-blooded", "whales",
)
ALL_BLACKLIST: Tuple[str, ...] = tuple(vc.BLACKLIST_SUBSTRINGS) + EXTRA_BLACKLIST

OUT_DIR = Path(os.environ.get("V7_OUT_DIR", "/content/v7_0_outputs"))
_DRV_ENV = os.environ.get("V7_DRIVE_DIR", "").strip()
if not _DRV_ENV and Path("/content/drive/MyDrive").is_dir():
    _DRV_ENV = "/content/drive/MyDrive/gemma4_v7_0"
DRIVE_DIR: Optional[Path] = Path(_DRV_ENV) if _DRV_ENV else None

CKPT_DIR = OUT_DIR / "ckpt_v7_0"
OUT_T16C = OUT_DIR / "gemma4_31b_v7_0_fullrank.t16c"
OUT_CRTR = OUT_DIR / "gemma4_31b_v7_router.crtr"
OUT_C8D = OUT_DIR / "gemma4_31b_v7_c8d.c8d"
OUT_BHDR = OUT_DIR / "gemma4_31b_v7_assistant.bhdr"
OUT_AST_Q2 = OUT_DIR / "gemma4_31b_v7_assistant_body_q2.t16c"
OUT_AST_Q4 = OUT_DIR / "gemma4_31b_v7_assistant_body_q4.t16c"
OUT_AST_Q8 = OUT_DIR / "gemma4_31b_v7_assistant_body_q8.t16c"
OUT_REPORT = OUT_DIR / "v7_0_report.json"
PROGRESS_JSON = CKPT_DIR / "progress.json"

MAX_T16C_BYTES = 6_174_015_488  # 5.75 GiB hard Pi 4 headroom guard


def configure_v7_0_max_vram(
    gpu_vram_gb: float, has_cuda: bool, *, respect_env: bool = True
) -> Dict[str, object]:
    """Configure TF32, cudnn, HBM residency, micro-batch size & ALS token count for G4/A100/L4."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
    force_max = (os.environ.get("V7_MAX_VRAM", "1").strip() == "1") if respect_env else True
    is_big_gpu = gpu_vram_gb >= 64.0
    resident_env = os.environ.get("V7_RESIDENT_GPU", "").strip() if respect_env else ""
    resident_gpu = (resident_env == "1") if resident_env else (is_big_gpu and force_max and has_cuda)

    if gpu_vram_gb >= 88.0:
        default_bs, default_ev, ast_bs, bh_bs = 256, 4096, 16, 512
        profile = "G4-96GB-MAX"
    elif is_big_gpu:
        default_bs, default_ev, ast_bs, bh_bs = 192, 4096, 16, 512
        profile = "A100-80GB-MAX"
    elif gpu_vram_gb >= 36.0:
        default_bs, default_ev, ast_bs, bh_bs = 128, 4096, 12, 512
        profile = "HIGH-VRAM-GPU"
    elif has_cuda:
        default_bs, default_ev, ast_bs, bh_bs = 96, 4096, 8, 512
        profile = "L4-22GB-MAX"
    else:
        default_bs, default_ev, ast_bs, bh_bs = 16, 512, 4, 128
        profile = "CPU-SELFTEST"

    bs_ov = os.environ.get("V7_BATCH_SEQ", "").strip() if respect_env else ""
    batch_seq = (max(int(bs_ov), default_bs) if force_max else int(bs_ov)) if bs_ov else default_bs
    ev_ov = os.environ.get("V7_ALS_EV", "").strip() if respect_env else ""
    als_ev = (max(int(ev_ov), default_ev) if force_max else int(ev_ov)) if ev_ov else default_ev

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
        "als_ev_toks": max(64, als_ev),
        "ast_bs": ast_bs,
        "bh_bs": bh_bs,
        "tf32": bool(has_cuda),
    }


def _v70_gpu_vram_stats_gb(device: torch.device) -> Tuple[float, float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    return (
        round(torch.cuda.memory_allocated(device) / (1024**3), 2),
        round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
        round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2),
    )


def host_ram_gb() -> Tuple[float, float]:
    try:
        info: Dict[str, int] = {}
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
    with open(path, "rb") as fp:
        while True:
            chk = fp.read(16 * 1024 * 1024)
            if not chk:
                break
            h.update(chk)
    return h.hexdigest()


def _atomic_torch_save(obj: object, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(obj, tmp_p)
    os.replace(tmp_p, dst)


def verify_t16c_integrity(
    path: Path,
    expected_mats: Optional[int] = None,
    expected_end_pos: Optional[int] = None,
) -> Tuple[int, int]:
    """Fail-closed binary structure verifier for T16C v1 headers, norms, and tensor records."""
    if not path.is_file():
        raise FileNotFoundError(f"verify_t16c_integrity: missing {path}")
    file_size = path.stat().st_size
    if file_size < 24:
        raise RuntimeError(f"verify_t16c_integrity: {path} truncated header ({file_size} < 24)")
    with open(path, "rb") as fp:
        hdr = fp.read(24)
        magic, ver, _flags, n_norms, n_mats, reserved = struct.unpack("<4s5I", hdr)
        if magic != p3.T16C_MAGIC or ver != p3.T16C_VERSION or reserved != 0 or n_mats == 0:
            raise RuntimeError(
                f"verify_t16c_integrity: invalid header magic={magic!r} ver={ver} reserved={reserved} n_mats={n_mats}"
            )
        for ni in range(n_norms):
            kb = fp.read(2)
            if len(kb) != 2:
                raise RuntimeError(f"verify_t16c_integrity: EOF reading norm[{ni}] key len at {fp.tell()}")
            klen = struct.unpack("<H", kb)[0]
            if klen == 0 or fp.seek(klen, 1) > file_size:
                raise RuntimeError(f"verify_t16c_integrity: invalid norm[{ni}] key len={klen}")
            ab = fp.read(4)
            if len(ab) != 4:
                raise RuntimeError(f"verify_t16c_integrity: EOF reading norm[{ni}] len at {fp.tell()}")
            alen = struct.unpack("<I", ab)[0]
            if alen == 0 or fp.seek(alen * 4, 1) > file_size:
                raise RuntimeError(f"verify_t16c_integrity: invalid norm[{ni}] len={alen}")
        target_mats = n_mats if expected_mats is None else int(expected_mats)
        if target_mats <= 0 or target_mats > n_mats:
            raise RuntimeError(f"verify_t16c_integrity: invalid target_mats={target_mats} (header n_mats={n_mats})")
        for mi in range(target_mats):
            rec_pos = fp.tell()
            raw_hdr = fp.read(34)
            if len(raw_hdr) != 34:
                raise RuntimeError(f"verify_t16c_integrity: EOF reading mat[{mi}] header at {rec_pos}")
            (
                nlen, d_out, d_in, block_size, n_bits, is_asym, has_fwht,
                inv_s_kind, low_rank, packed_bytes, lr_bytes,
            ) = struct.unpack("<HIIHBBBBHQQ", raw_hdr)
            if nlen == 0 or d_out == 0 or d_in == 0 or is_asym > 1 or has_fwht > 1 or inv_s_kind > 2:
                raise RuntimeError(
                    f"verify_t16c_integrity: invalid mat[{mi}] @ {rec_pos:,}: "
                    f"nlen={nlen} d=({d_out},{d_in}) bsz={block_size} nb=0x{n_bits:02x}"
                )
            nbuf = fp.read(nlen)
            if len(nbuf) != nlen:
                raise RuntimeError(f"verify_t16c_integrity: EOF reading mat[{mi}] name at {rec_pos}")
            name = nbuf.decode("utf-8", errors="strict")
            pos = fp.tell()
            pad = (64 - (pos % 64)) % 64
            if pad:
                fp.seek(pad, 1)
            exp_p = p3.expected_tile16_bytes(d_out, d_in, block_size, n_bits, is_asym == 1)
            if packed_bytes != exp_p:
                raise RuntimeError(f"verify_t16c_integrity: {name} packed_bytes={packed_bytes} != exp={exp_p}")
            inv_s_bytes = (d_in * 4) if inv_s_kind == 1 else ((d_in * 2) if inv_s_kind == 2 else 0)
            exp_lr = (d_out * 4 + low_rank * 4 + d_out * low_rank + low_rank * d_in) if low_rank > 0 else 0
            if lr_bytes != exp_lr:
                raise RuntimeError(f"verify_t16c_integrity: {name} lr_bytes={lr_bytes} != exp={exp_lr}")
            end_rec = fp.seek(packed_bytes + inv_s_bytes + lr_bytes, 1)
            if end_rec > file_size:
                raise RuntimeError(f"verify_t16c_integrity: {name} overruns EOF ({end_rec} > {file_size})")
        end_pos = fp.tell()
        if expected_end_pos is not None and end_pos != expected_end_pos:
            raise RuntimeError(
                f"verify_t16c_integrity: verified end_pos={end_pos:,} != expected_end_pos={expected_end_pos:,}"
            )
        if expected_end_pos is None and expected_mats is None and end_pos != file_size:
            raise RuntimeError(
                f"verify_t16c_integrity: trailing unparsed bytes after {target_mats} mats ({end_pos:,} != {file_size:,})"
            )
    return target_mats, end_pos


def sync_checkpoint_to_drive(
    expected_t16c_pos: Optional[int] = None,
    drive_dir: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
    out_t16c: Optional[Path] = None,
) -> None:
    """Commit `.t16c.partial` FIRST and verify exact size BEFORE syncing `stage3_layer_ckpt.pt` LAST."""
    d_dir = DRIVE_DIR if drive_dir is None else drive_dir
    c_dir = CKPT_DIR if ckpt_dir is None else ckpt_dir
    t_path = OUT_T16C if out_t16c is None else out_t16c
    if d_dir is None:
        return
    drive_ckpt = d_dir / "ckpt_v7_0"
    drive_ckpt.mkdir(parents=True, exist_ok=True)
    if expected_t16c_pos is not None or t_path.is_file():
        if not t_path.is_file():
            print(f"  [Drive Checkpoint Abort] Missing local {t_path}; skipping Drive sync.", flush=True)
            return
        local_sz = t_path.stat().st_size
        if expected_t16c_pos is not None and local_sz != expected_t16c_pos:
            print(
                f"  [Drive Checkpoint Abort] {t_path.name} size={local_sz:,} != expected={expected_t16c_pos:,}",
                flush=True,
            )
            return
        partial_dst = d_dir / (t_path.name + ".partial")
        try:
            shutil.copy2(t_path, partial_dst)
            remote_sz = partial_dst.stat().st_size
            if remote_sz != local_sz:
                raise IOError(f"partial size mismatch local={local_sz:,} drive={remote_sz:,}")
        except Exception as e:
            print(
                f"  [Drive T16C Partial Abort] {e} — NOT updating stage3_layer_ckpt.pt to avoid zero-hole mismatch.",
                flush=True,
            )
            return
    ck_files = sorted(
        (p for p in c_dir.glob("*") if p.is_file() and not p.name.endswith(".tmp")),
        key=lambda p: (1 if p.name == "stage3_layer_ckpt.pt" else 0, p.name),
    )
    for p in ck_files:
        try:
            dst = drive_ckpt / p.name
            shutil.copy2(p, dst)
            if dst.stat().st_size != p.stat().st_size:
                raise IOError(f"size mismatch local={p.stat().st_size} drive={dst.stat().st_size}")
        except Exception as e:
            print(f"  [Drive Checkpoint Abort] Failed copying {p.name}: {e}", flush=True)
            return
    print(f"  [DRIVE CHECKPOINT SYNCED] -> {drive_ckpt}", flush=True)


def persist_verified(src: Path, drive_dir: Optional[Path] = None) -> Tuple[int, str]:
    if not src.is_file():
        raise FileNotFoundError(f"Missing output artifact: {src}")
    sz = int(src.stat().st_size)
    digest = sha256_file(src)
    d_dir = DRIVE_DIR if drive_dir is None else drive_dir
    if d_dir is not None:
        d_dir.mkdir(parents=True, exist_ok=True)
        dst = d_dir / src.name
        tmp = d_dir / (src.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        dst_dig = sha256_file(dst)
        if dst_dig != digest or dst.stat().st_size != sz:
            raise IOError(f"Drive SHA-256 mismatch for {src.name}: {dst_dig} != {digest}")
        print(f"  [DRIVE PERSISTED] {dst.name} ({sz:,} B, sha256={digest[:16]}...)", flush=True)
    return sz, digest


# ---------------------------------------------------------------------------
# V7-5.45 Full-Rank Allocation Schedule (`<= 5.75 GiB`, Zero 1-Bit Attn)
# ---------------------------------------------------------------------------
def get_lm_spec_v7_0(l_idx: int, pname: str) -> Tuple[int, int, int, bool]:
    """
    Return `(mode_i, bsize=256, lr_rank, to_rowmajor)` for V7.0 Full-Rank Hybrid:
      - `L58..L59`: 3-bit (`mode=4`, `k/v r=24`, others `r=16`)
      - `ATTN_3B_LAYERS`: `q/o` 3-bit (`mode=4`, `r=16`)
      - All remaining Attention (`q/k/v/o` across L00..L59): 5-trit ternary (`mode=1`, ZERO 1-bit attn!)
      - `L00..L01` & `L56..L57` MLPs + `FULL_LAYERS` `down_proj`: 5-trit ternary (`mode=1`)
      - Remaining `153` sliding/gate/up MLPs (`L02..L55`): Full-Rank `1b_asym` (`mode=0`, `down r=16`,
        `g/u r=8`) transposed to ARMv8 NEON Row-Major (`1 | 0x40`).
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
    if l_idx in (0, 1, 56, 57):
        return (1, 256, 16 if pname == "down_proj" else 8, False)
    if l_idx in FULL_LAYERS and pname == "down_proj":
        return (1, 256, 16, False)
    return (0, 256, 16 if pname == "down_proj" else 8, True)


def spec_to_t16c_bits(mode_i: int, to_rowmajor: bool) -> Tuple[int, bool]:
    nb, asy = p3.MODE_TO_T16C[mode_i]
    if to_rowmajor:
        if (nb & 0x3F) != 1:
            raise ValueError(f"V7.0 Row-Major flag expects mode_i=0 (1-bit), got mode_i={mode_i}")
        nb = nb | T16C_ROWMAJOR_BIT
    return nb, asy


def compute_v7_0_budget_bytes(
    vocab_size: int = 262144,
    d_mod: int = 5376,
    n_layers: int = 60,
    nh: int = 32,
    nkv_s: int = 16,
    nkv_f: int = 4,
    hd_s: int = 256,
    hd_f: int = 512,
    ff: int = 21504,
) -> Dict[str, int]:
    """Compute exact predicted LM+Embed and Vision 3-bit `.t16c` byte budgets."""
    emb_b = p3.expected_tile16_bytes(vocab_size, d_mod, 256, 3, False) + 80
    lm_body_b = 0
    lm_mats = 1
    attn_1b_cnt = 0
    rm1_cnt = 0
    tern_cnt = 0
    b3_cnt = 0
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
            m_i, bsz, rk, to_rm = get_lm_spec_v7_0(l_idx, pn)
            nb, asy = spec_to_t16c_bits(m_i, to_rm)
            if pn in ("q_proj", "k_proj", "v_proj", "o_proj") and (nb & 0x3F) == 1:
                attn_1b_cnt += 1
            if to_rm:
                rm1_cnt += 1
            elif m_i == 1:
                tern_cnt += 1
            elif m_i == 4:
                b3_cnt += 1
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
    est_tot = emb_b + lm_body_b + vis_b + 6_500_000
    return {
        "pred_lm_mats": lm_mats,
        "pred_body_b": lm_body_b,
        "pred_lm_b": emb_b + lm_body_b,
        "pred_vis_3b": vis_b,
        "pred_total_b": est_tot,
        "attn_1b_cnt": attn_1b_cnt,
        "rm1_cnt": rm1_cnt,
        "tern_cnt": tern_cnt,
        "b3_cnt": b3_cnt,
    }


def fit_mod_gate_on_inp(
    inp_tr: torch.Tensor,
    mod_tr: torch.Tensor,
    inp_va: torch.Tensor,
    mod_va: torch.Tensor,
    steps: int = 200,
    lr: float = 3e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[np.ndarray, float, float, float]:
    """Fit KV-safe MoD linear gate on post-`input_layernorm` `inp` (`run31b.rs::predict_mod_exec_prob`)."""
    torch.manual_seed(seed)
    d_mod = int(inp_tr.shape[1])
    gate_lin = nn.Linear(d_mod, 1).to(device)
    x_tr_d = inp_tr.to(device=device, dtype=torch.float32)
    y_tr_d = mod_tr.to(device=device, dtype=torch.float32).view(-1)
    x_va_d = inp_va.to(device=device, dtype=torch.float32)
    y_va_d = mod_va.to(device=device, dtype=torch.float32).view(-1)

    opt = torch.optim.AdamW(gate_lin.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    bce = nn.BCEWithLogitsLoss()
    n_tr = x_tr_d.shape[0]
    bs = min(1024, n_tr)
    best_loss = float("inf")
    best_sd = {k: v.detach().cpu().clone() for k, v in gate_lin.state_dict().items()}

    with torch.enable_grad():
        for st in range(steps):
            perm = torch.randint(0, n_tr, (bs,), device=device)
            opt.zero_grad()
            logits = gate_lin(x_tr_d[perm]).squeeze(-1)
            loss = bce(logits, y_tr_d[perm])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_lin.parameters(), 1.0)
            opt.step()
            sch.step()
            if (st + 1) % 50 == 0 or (st + 1) == steps:
                with torch.no_grad():
                    vl = float(bce(gate_lin(x_va_d).squeeze(-1), y_va_d).item())
                    if vl < best_loss:
                        best_loss = vl
                        best_sd = {k: v.detach().cpu().clone() for k, v in gate_lin.state_dict().items()}

    gate_lin.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    with torch.no_grad():
        va_logits = gate_lin(x_va_d).squeeze(-1)
        val_acc = float(((torch.sigmoid(va_logits) >= 0.5).float() == y_va_d).float().mean().item())
        w_np = best_sd["weight"].numpy().astype(np.float32).reshape(1, d_mod)
        b_f = float(best_sd["bias"].numpy().astype(np.float32)[0])
    mw_i8, m_sc = crtr._quantize_rows_i8(w_np)
    return mw_i8[0], float(m_sc[0]), b_f, val_acc


def train_layer_crtr_v7_0(
    l_idx: int,
    xp_tr_cpu: torch.Tensor,
    xm_tr_cpu: torch.Tensor,
    xo_tr_cpu: torch.Tensor,
    xp_va_cpu: torch.Tensor,
    xm_va_cpu: torch.Tensor,
    xo_va_cpu: torch.Tensor,
    norm_in_g: torch.Tensor,
    norm_pf_g: torch.Tensor,
    w_gate_f16: torch.Tensor,
    w_up_f16: torch.Tensor,
    rmsn_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    *,
    block_size: int = 256,
    rank: int = 32,
    top_p: float = 0.30,
    mod_quantile: float = 0.20,
    steps: int = 200,
    eps: float = 1e-6,
    device: torch.device = torch.device("cpu"),
) -> Tuple[dict, Dict[str, float]]:
    """
    Train representation-matched V7.0 CRTR v1 layer:
      - `V/U` 84-block predictor on student `hm` (`pre_feedforward_layernorm`) & Student GeGLU energy.
      - Softer `MoD` linear gate (`mod_q20` threshold at `mod_quantile=0.20` vs V6.1 `0.40`) on `inp`.
    """
    ff = int(w_gate_f16.shape[0])
    n_blks = ff // block_size
    wg = w_gate_f16.to(device)
    wu = w_up_f16.to(device)
    nin = norm_in_g.to(device)
    npf = norm_pf_g.to(device)

    def featurize(xp_c: torch.Tensor, xm_c: torch.Tensor, xo_c: torch.Tensor):
        n = xp_c.shape[0]
        inp_l, hm_l, blk_l, drel_l = [], [], [], []
        with torch.no_grad():
            for i0 in range(0, n, 2048):
                i1 = min(i0 + 2048, n)
                xp = xp_c[i0:i1].to(device)
                xm = xm_c[i0:i1].to(device)
                xo = xo_c[i0:i1].to(device)
                inp_b = rmsn_fn(xp, nin).float()
                hm_16 = rmsn_fn(xm, npf)
                act = F.gelu(F.linear(hm_16, wg).float(), approximate="tanh") * F.linear(hm_16, wu).float()
                blk_e = act.reshape(-1, n_blks, block_size).pow(2).mean(dim=-1).sqrt()
                drel = (xo.float() - xp.float()).norm(dim=-1) / xp.float().norm(dim=-1).clamp(min=eps)
                inp_l.append(inp_b.cpu())
                hm_l.append(hm_16.float().cpu())
                blk_l.append(blk_e.cpu())
                drel_l.append(drel.cpu())
        return torch.cat(inp_l, 0), torch.cat(hm_l, 0), torch.cat(blk_l, 0), torch.cat(drel_l, 0)

    inp_tr, hm_tr, e_tr, d_tr = featurize(xp_tr_cpu, xm_tr_cpu, xo_tr_cpu)
    inp_va, hm_va, e_va, d_va = featurize(xp_va_cpu, xm_va_cpu, xo_va_cpu)
    mod_q = float(torch.quantile(d_tr, mod_quantile).item())
    m_tr = (d_tr >= mod_q).float()
    m_va = (d_va >= mod_q).float()

    router_net, blk_m = crtr.train_layer_router(
        hm_tr, e_tr, hm_va, e_va,
        top_p=top_p, rank=rank, steps=steps, lr=3e-3,
        mod_targets_tr=None, mod_targets_va=None,
        device=str(device), seed=42 + l_idx,
    )
    rec = crtr.quantize_layer_for_crtr(router_net, block_size=block_size)
    mw_i8, m_scale, m_bias, mod_acc = fit_mod_gate_on_inp(
        inp_tr, m_tr, inp_va, m_va, steps=steps, lr=3e-3, device=device, seed=1042 + l_idx
    )
    rec["mod_w_i8"] = mw_i8
    rec["mod_scale"] = m_scale
    rec["mod_bias"] = m_bias
    metrics = {
        "val_recall_before": blk_m["val_recall_before"],
        "val_recall_after": blk_m["val_recall_after"],
        "mod_val_acc": round(mod_acc, 5),
        "mod_thresh_q": round(mod_q, 5),
        "mod_pos_frac_va": round(float(m_va.mean().item()), 4),
    }
    return rec, metrics


def permute_geglu_weights(
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, neuron_e: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Permute GeGLU intermediate neurons by descending energy while preserving `gate/up @ down` identically."""
    perm = torch.argsort(neuron_e.float().cpu(), descending=True)
    wg_p = w_gate[perm].contiguous()
    wu_p = w_up[perm].contiguous()
    wd_p = w_down[:, perm].contiguous()
    e_sorted = neuron_e.float().cpu()[perm]
    top50 = 100.0 * float(e_sorted[: w_gate.shape[0] // 2].sum().item() / e_sorted.sum().clamp(min=1e-12).item())
    return wg_p, wu_p, wd_p, top50


def transpose_t16c_1b_col_to_rm(pb: np.ndarray, d_out: int, d_in: int, bsize: int = 256) -> np.ndarray:
    """Transpose Column-Major 1-bit T16C payload (`n_bits=1`) into ARMv8 NEON Row-Major (`1 | 0x40`)."""
    out = rrm.transpose_planes_col_to_rowmajor(np.ascontiguousarray(pb, dtype=np.uint8), d_out, d_in, bsize, 1, True)
    exp_len = p3.expected_tile16_bytes(d_out, d_in, bsize, 1 | T16C_ROWMAJOR_BIT, True)
    if out.size != exp_len:
        raise AssertionError(f"Row-Major 1b transpose len {out.size} != expected {exp_len}")
    return out


def reference_decode_rm1_tile16(
    pb: np.ndarray, d_out: int, d_in: int, bsize: int = 256, has_fwht: bool = False
) -> np.ndarray:
    """Reference CPU decoder for Row-Major 1-bit asymmetric T16C (`n_bits=1|0x40`) matching `t16c.rs`."""
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    bpr = bsize // 8
    tblk_b = 64 + 16 * bpr
    arr = pb.reshape(n_tiles, n_blks, tblk_b)
    sc = arr[:, :, 0:32].copy().view("<f2").reshape(n_tiles, n_blks, 16).transpose(0, 2, 1).astype(np.float32)
    bi = arr[:, :, 32:64].copy().view("<f2").reshape(n_tiles, n_blks, 16).transpose(0, 2, 1).astype(np.float32)
    bits = np.unpackbits(arr[:, :, 64:].reshape(n_tiles, n_blks, 16, bpr), axis=-1, bitorder="big")
    bits = bits.transpose(0, 2, 1, 3).astype(np.float32)
    w_rot = (bi[..., None] + sc[..., None] * (2.0 * bits - 1.0)).reshape(n_tiles * 16, d_in)[:d_out, :]
    if has_fwht:
        h = pa.make_sylvester_hadamard(bsize)
        w_rot = (w_rot.reshape(d_out, n_blks, bsize) @ h).reshape(d_out, d_in)
    return w_rot


def select_first_valid_trie_ids(top_ids_2d: torch.Tensor, trie: bhmod.BpePrefixTrie) -> List[int]:
    """Select exact `V7.0` student greedy argmax (`top_ids[:, 0]`), falling back only if piece is trie-empty."""
    out: List[int] = []
    rows = top_ids_2d.tolist()
    for row in rows:
        chosen = int(row[0])
        if not trie.tok_syms[chosen]:
            for cand in row[1:]:
                if trie.tok_syms[int(cand)]:
                    chosen = int(cand)
                    break
        out.append(chosen)
    return out


# ---------------------------------------------------------------------------
# CPU-Runnable 6-Gate Self-Test Battery (`--selftest`)
# ---------------------------------------------------------------------------
def run_selftest() -> int:
    """Mandatory 6-gate CPU self-verification battery (fail-closed)."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: V7-5.45 Allocation Schedule, <= 5.75 GiB Budget, Zero 1-Bit Attention
    bgt = compute_v7_0_budget_bytes()
    specs_ok = bgt["attn_1b_cnt"] == 0 and bgt["pred_lm_mats"] == 411 and bgt["pred_total_b"] <= MAX_T16C_BYTES
    gate(
        "Gate 1: V7-5.45 schedule & <= 5.75 GiB budget (0 1-bit attn)",
        specs_ok,
        f"LM+Emb={bgt['pred_lm_b']/1024**3:.3f}GiB Vis={bgt['pred_vis_3b']/1024**3:.3f}GiB "
        f"Total={bgt['pred_total_b']/1024**3:.3f}GiB ({bgt['pred_total_b']/1e9:.3f}GB) "
        f"3b={bgt['b3_cnt']} tern={bgt['tern_cnt']} rm1={bgt['rm1_cnt']} attn1b={bgt['attn_1b_cnt']}",
    )

    # Gate 2: Co-activation GeGLU permutation exact mathematical invariance + Row-Major 1b transpose parity
    torch.manual_seed(17)
    d_o_t, d_i_t, ff_t = 32, 256, 512
    x_t = torch.randn((8, d_i_t))
    wg = torch.randn((ff_t, d_i_t)) * 0.2
    wu = torch.randn((ff_t, d_i_t)) * 0.2
    wd = torch.randn((d_o_t, ff_t)) * 0.2
    ne = torch.rand((ff_t,))
    y_ref = F.linear(F.gelu(F.linear(x_t, wg), approximate="tanh") * F.linear(x_t, wu), wd)
    wg_p, wu_p, wd_p, top50_pct = permute_geglu_weights(wg, wu, wd, ne)
    y_perm = F.linear(F.gelu(F.linear(x_t, wg_p), approximate="tanh") * F.linear(x_t, wu_p), wd_p)
    perm_err = float((y_ref - y_perm).abs().max().item() / y_ref.abs().max().clamp(min=1e-6).item())

    # Verify Column-Major -> Row-Major 1b decode equivalence against bitplane packing
    w_rm_src = np.sign(np.random.default_rng(7).normal(0.0, 1.0, size=(32, 256))).astype(np.float32) * 0.25
    # Build synthetic Col-Major 1-bit Tile-16 block (sc=0.25, bi=0.0, col-major bitplane)
    n_t = 2
    bpr = 256 // 8
    pb_col = np.zeros((n_t, 1, 64 + 16 * bpr), dtype=np.uint8)
    pb_col[:, :, 0:32] = np.full((n_t, 1, 16), 0.25, dtype="<f2").view(np.uint8).reshape(n_t, 1, 32)
    b01 = (w_rm_src.reshape(2, 16, bpr, 8) > 0).astype(np.uint8)
    packed_cm = np.packbits(b01, axis=-1, bitorder="big")[..., 0]  # [2, 16, 32]
    pb_col[:, :, 64:] = np.ascontiguousarray(packed_cm.transpose(0, 2, 1)).reshape(n_t, 1, 16 * bpr)
    pb_rm = transpose_t16c_1b_col_to_rm(pb_col.reshape(-1), 32, 256, 256)
    w_rm_dec = reference_decode_rm1_tile16(pb_rm, 32, 256, 256, has_fwht=False)
    rm_err = float(np.max(np.abs(w_rm_dec - w_rm_src)))
    gate(
        "Gate 2: GeGLU permutation invariance & RM-1b transpose parity",
        perm_err < 1e-5 and rm_err == 0.0,
        f"perm_err={perm_err:.2e} top50_e={top50_pct:.1f}% rm_err={rm_err:.2e}",
    )

    # Gate 3: Representation-matched CRTR + Softer MoD (q20) training & container roundtrip
    rng = np.random.default_rng(91)
    dm_c, ff_c, bsz_c = 128, 768, 64  # n_blks=12
    n_blks_c = ff_c // bsz_c
    q_p, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(dm_c, n_blks_c)).astype(np.float32))
    wg_s = np.zeros((ff_c, dm_c), dtype=np.float32)
    wu_s = np.zeros((ff_c, dm_c), dtype=np.float32)
    for b in range(n_blks_c):
        wg_s[b * bsz_c : (b + 1) * bsz_c] = 1.5 * q_p[:, b : b + 1].T
        wu_s[b * bsz_c : (b + 1) * bsz_c] = 1.5 * q_p[:, b : b + 1].T

    sal_axis = rng.normal(0.0, 1.0, size=(dm_c,)).astype(np.float32)
    sal_axis /= np.linalg.norm(sal_axis) + 1e-8

    def mk_crtr_split(n: int):
        xp = rng.normal(0.0, 0.12, size=(n, dm_c)).astype(np.float32)
        xm = rng.normal(0.0, 0.10, size=(n, dm_c)).astype(np.float32)
        xo = xp.copy()
        for i in range(n):
            acts = rng.choice(n_blks_c, size=3, replace=False)
            for a in acts:
                xm[i] += 2.2 * q_p[:, a]
            is_sal = (i % 5 != 0)
            xp[i] += (2.2 if is_sal else -2.2) * sal_axis
            xo[i] = xp[i] * (1.8 if is_sal else 1.05)
        return torch.from_numpy(xp), torch.from_numpy(xm), torch.from_numpy(xo)

    xpt, xmt, xot = mk_crtr_split(384)
    xpv, xmv, xov = mk_crtr_split(96)
    rec, crtr_m = train_layer_crtr_v7_0(
        0, xpt, xmt, xot, xpv, xmv, xov,
        torch.ones(dm_c), torch.ones(dm_c),
        torch.from_numpy(wg_s), torch.from_numpy(wu_s),
        lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6),
        block_size=bsz_c, rank=16, top_p=0.25, mod_quantile=0.20, steps=180, device=torch.device("cpu"),
    )
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        cp = td_p / "v7_test.crtr"
        csz = crtr.export_crtr([rec], cp)
        cmeta, clayers = crtr.load_crtr(cp)
        ok_crtr = (
            crtr_m["val_recall_after"] >= 0.85
            and crtr_m["mod_val_acc"] >= 0.80
            and abs(crtr_m["mod_pos_frac_va"] - 0.80) <= 0.15
            and (cmeta["flags"] & crtr.CRTR_FLAG_MOD_GATE) != 0
            and len(clayers) == 1
        )
        gate(
            "Gate 3: CRTR v1 + softer MoD (q20) train & roundtrip",
            ok_crtr,
            f"blk_rec={crtr_m['val_recall_after']:.3f} mod_acc={crtr_m['mod_val_acc']:.3f} "
            f"pos_frac={crtr_m['mod_pos_frac_va']:.2f} bytes={csz}",
        )

        # Gate 4: BHDR d_byte=96 student_argmax Appendix F.1 distillation >= 0.90 beam acc + size formula check
        bh_exp_sz = 32 + 384 * 1024 + 384 * 8 + 3 * 384 * 1024 + 3 * 384 * 4 + 261 * 384 + 261 * 4 + 3 * 384 * 384 + 3 * 384 * 8 + 3 * 384 * 384 + 3 * 384 * 8 + 261 * 384 + 261 * 8
        id_to_tok = [""] * 120
        id_to_tok[0], id_to_tok[1], id_to_tok[2], id_to_tok[106] = "<pad>", "<eos>", "<bos>", "<turn|>"
        id_to_tok[10], id_to_tok[11], id_to_tok[12], id_to_tok[13] = "\u2581a", "\u2581an", "\u2581and", "4"
        trie = bhmod.BpePrefixTrie(id_to_tok)
        aids = [tid for tid, s in enumerate(id_to_tok) if s]
        q_bh, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(64, len(aids))).astype(np.float32))
        protos = q_bh.T

        def mk_bh(n: int):
            hs = np.zeros((n, 64), dtype=np.float32)
            sids, sps = [], []
            for i in range(n):
                c = i % len(aids)
                hs[i] = 2.6 * protos[c] + rng.normal(0.0, 0.08, size=(64,)).astype(np.float32)
                pr = np.full((len(aids),), 0.01 / max(len(aids) - 1, 1), dtype=np.float32)
                pr[c] = 0.99
                sids.append(list(aids))
                sps.append([float(v) for v in pr])
            order = np.argsort(-np.array(sps), axis=1)
            top_ids_t = torch.tensor([[sids[r][j] for j in order[r]] for r in range(n)], dtype=torch.long)
            gids = select_first_valid_trie_ids(top_ids_t, trie)
            return torch.from_numpy(hs), sids, sps, gids

        ht_b, sit_b, spt_b, gt_b = mk_bh(192)
        hv_b, siv_b, spv_b, gv_b = mk_bh(48)
        bm, bmet = bhmod.train_byte_head_eot_distill(
            trie, ht_b, sit_b, spt_b, gt_b, hv_b, siv_b, spv_b, gv_b,
            d_byte=96, max_bytes=16, steps=180, batch_size=32, lr=6e-3, seed=7,
        )
        bp = td_p / "v7_bhdr.bhdr"
        bsz_w = bhmod.export_bhdr_v1(bm, bp)
        bl = bhmod.load_bhdr_v1(bp)
        hits = sum(1 for vi in range(len(gv_b)) if bhmod.decode_bhdr_np_beam(bl, trie, hv_b[vi].numpy(), 2)[0] == gv_b[vi])
        bacc = hits / len(gv_b)
        gate(
            "Gate 4: BHDR student_argmax distillation >= 0.90 + 384d byte count",
            bacc >= 0.90 and bh_exp_sz < 4_000_000 and bsz_w == bp.stat().st_size,
            f"beam_acc={bacc:.3f} kl={bmet['val_kl_before']:.3f}->{bmet['val_kl_after']:.3f} bhdr384_bytes={bh_exp_sz:,}",
        )

        # Gate 5: Fail-closed T16C zero-hole check + .partial-first Drive sync
        t16_ok = td_p / "ok.t16c"
        with open(t16_ok, "wb") as fp:
            fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, 1, 1, 0))
            fp.write(struct.pack("<H", 4) + b"nrm0" + struct.pack("<I", 4) + np.ones(4, dtype="<f4").tobytes())
            p3._write_mat_record(fp, fp.tell(), "m0", 16, 256, 256, 1 | T16C_ROWMAJOR_BIT, True, False, pb_rm[:576], np.ones(256, dtype="<f4"), None)
        vm, ve = verify_t16c_integrity(t16_ok)
        raw_bad = bytearray(t16_ok.read_bytes())
        off_m = 24 + 2 + 4 + 4 + 16
        raw_bad[off_m : off_m + 34] = b"\x00" * 34
        t16_bad = td_p / "hole.t16c"
        t16_bad.write_bytes(bytes(raw_bad))
        hole_ok = False
        try:
            verify_t16c_integrity(t16_bad)
        except RuntimeError:
            hole_ok = True
        drv = td_p / "drv"
        ckp = td_p / "ckp"
        drv.mkdir(); ckp.mkdir()
        (ckp / "stage3_layer_ckpt.pt").write_bytes(b"ckpt")
        sync_checkpoint_to_drive(expected_t16c_pos=ve + 128, drive_dir=drv, ckpt_dir=ckp, out_t16c=t16_ok)
        refused = not (drv / "ckpt_v7_0" / "stage3_layer_ckpt.pt").exists()
        sync_checkpoint_to_drive(expected_t16c_pos=ve, drive_dir=drv, ckpt_dir=ckp, out_t16c=t16_ok)
        synced = (drv / "ok.t16c.partial").is_file() and (drv / "ckpt_v7_0" / "stage3_layer_ckpt.pt").is_file()
        gate(
            "Gate 5: Fail-closed T16C integrity & .partial-first Drive sync",
            vm == 1 and ve == t16_ok.stat().st_size and hole_ok and refused and synced,
            f"mats={vm} hole_caught={hole_ok} refused={refused} synced={synced}",
        )

    # Gate 6: Holdout prompt disjointness + blacklist audit + G4/A100/L4 max-VRAM config
    dj = spmod.assert_holdouts_disjoint()
    all_t_low = " ".join(spmod.HELDOUT_CHAT_QUESTIONS + [p[1] for p in spmod.HELDOUT_NLL_PASSAGES]).lower()
    bl_hits = [b for b in ALL_BLACKLIST if b in all_t_low]
    g4_cfg = configure_v7_0_max_vram(96.0, has_cuda=False, respect_env=False)
    a100_cfg = configure_v7_0_max_vram(80.0, has_cuda=False, respect_env=False)
    l4_cfg = configure_v7_0_max_vram(22.5, has_cuda=True, respect_env=False)
    gate(
        "Gate 6: Held-out disjointness, blacklist & G4/A100/L4 max-VRAM config",
        dj["overlap_4grams"] == 0
        and not bl_hits
        and g4_cfg["profile"] == "G4-96GB-MAX"
        and g4_cfg["batch_seq"] == 256
        and a100_cfg["profile"] == "A100-80GB-MAX"
        and a100_cfg["batch_seq"] == 192
        and l4_cfg["profile"] == "L4-22GB-MAX"
        and l4_cfg["batch_seq"] == 96
        and l4_cfg["als_ev_toks"] == 4096,
        f"passages={dj['passages']} overlap_4grams={dj['overlap_4grams']} g4={g4_cfg['profile']}/bs={g4_cfg['batch_seq']} a100={a100_cfg['profile']}/bs={a100_cfg['batch_seq']} l4={l4_cfg['profile']}/bs={l4_cfg['batch_seq']}",
    )

    print(f"V7_0_SELFTEST: {6 - failures}/6 gates passed", flush=True)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Full GPU Calibration + CRTR + C8DB + Target-Aligned Assistant-V7 Pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> Dict[str, object]:
    """Run complete Gemma 4 31B V7.0 Full-Rank Hybrid + CRTR/MoD + Assistant-V7 GPU build."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_pipeline() requires CUDA GPU (pass --selftest on CPU)")
    if run_selftest() != 0:
        raise RuntimeError("V7.0 pre-flight CPU selftest failed")

    t0_all = time.perf_counter()
    device = torch.device("cuda:0")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    hw_cfg = configure_v7_0_max_vram(gpu_vram_gb, has_cuda=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if DRIVE_DIR is not None:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        drive_ckpt = DRIVE_DIR / "ckpt_v7_0"
        if drive_ckpt.is_dir():
            print(f"[Drive Resume] Restoring checkpoints from {drive_ckpt}...", flush=True)
            for p in drive_ckpt.glob("*"):
                if p.is_file() and not p.name.endswith(".tmp") and not (CKPT_DIR / p.name).is_file():
                    shutil.copy2(p, CKPT_DIR / p.name)
        partial_t16c = DRIVE_DIR / (OUT_T16C.name + ".partial")
        if partial_t16c.is_file() and not OUT_T16C.is_file():
            print(f"[Drive Resume] Restoring {partial_t16c.name} ({partial_t16c.stat().st_size:,} B)...", flush=True)
            shutil.copy2(partial_t16c, OUT_T16C)

    print(
        f"[V7.0 INIT] GPU={gpu_name} ({gpu_vram_gb:.1f} GiB) | profile={hw_cfg['profile']} "
        f"| resident_gpu={hw_cfg['resident_gpu']} | micro_batch={hw_cfg['batch_seq']} "
        f"| als_ev={hw_cfg['als_ev_toks']} | dtype={compute_dtype} | RAM={host_ram_gb()} | DRIVE={DRIVE_DIR}",
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
    fq_so = Path("/tmp/libfast_quant_v7.so")
    subprocess.run(["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", str(fq_c), "-o", str(fq_so), "-lm"], check=True)
    LIB_FQ = ctypes.CDLL(str(fq_so))
    LIB_FQ.t16c_tblk_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    LIB_FQ.t16c_tblk_bytes.restype = ctypes.c_size_t
    LIB_FQ.gptq_quant_chunk_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    LIB_FQ.quant_and_pack_t16c_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]

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

    def symmetrize_inplace_(h: torch.Tensor, tile: int = 1024) -> torch.Tensor:
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
                chunk_pb = np.empty(((r1 - r0 + 15) // 16) * tblk_b, dtype=np.uint8)
                LIB_FQ.quant_and_pack_t16c_f32(wc_rot.ctypes.data, r1 - r0, d_in, bsize, mode, chunk_pb.ctypes.data)
                pbytes[(r0 // 16) * tblk_b : (r0 // 16) * tblk_b + chunk_pb.size] = chunk_pb
                w_eff_cpu[r0:r1] = fwht_gpu(torch.from_numpy(wc_rot).to(device), bsize).to(torch.float16).cpu()
                del wc_rot, chunk_pb
            if to_rowmajor:
                pbytes = transpose_t16c_1b_col_to_rm(pbytes, d_out, d_in, bsize)
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
                                U_tmp = torch.linalg.cholesky(h_inv, upper=True)
                                del h_inv; h_inv = None
                                U_chol = U_tmp.contiguous()
                                del U_tmp
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
        pbytes = np.empty(exp_bytes, dtype=np.uint8)
        LIB_FQ.quant_and_pack_t16c_f32(w_work_np.ctypes.data, d_out, d_in, bsize, mode, pbytes.ctypes.data)
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
            pbytes = transpose_t16c_1b_col_to_rm(pbytes, d_out, d_in, bsize)
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
        used_gb, tot_gb = host_ram_gb()
        payload = {**state, "host_ram_used_gb": used_gb, "host_ram_total_gb": tot_gb, "elapsed_s": round(time.perf_counter() - t0_all, 2)}
        tmp_p = PROGRESS_JSON.with_suffix(".tmp")
        tmp_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_p, PROGRESS_JSON)

    # 1. Stage 1: Model & Corpus
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

    bgt = compute_v7_0_budget_bytes(
        vocab_size=int(tc["vocab_size"]), d_mod=D_MOD, n_layers=N_LAYERS,
        nh=NH, nkv_s=NKV_S, nkv_f=NKV_F, hd_s=HD_S, hd_f=HD_F, ff=21504,
    )
    print(
        f"[Budget Guard] V7.0 Total={bgt['pred_total_b']:,} B ({bgt['pred_total_b']/1024**3:.3f} GiB <= 5.75 GiB) "
        f"| LM+Emb={bgt['pred_lm_b']/1024**3:.3f} GiB | Vis={bgt['pred_vis_3b']/1024**3:.3f} GiB",
        flush=True,
    )
    assert bgt["pred_total_b"] <= MAX_T16C_BYTES and bgt["attn_1b_cnt"] == 0

    def is_clean_row(txt: str) -> bool:
        low = txt.lower()
        return not any(b in low for b in ALL_BLACKLIST)

    print("[Stage 1] Packing full clean corpus (GSM8K/Code/Alpaca/WikiText-2)...", flush=True)
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
    seq_ids_list, seq_nxt_list, seq_ans_list, seq_vlm_ans_list = [], [], [], []
    t_cursor = 0
    for _ in range(N_TEXT_SEQ):
        cur_toks, cur_ans = [2], [False]
        while len(cur_toks) < SEQ_LEN + 1:
            dn = dom_names[t_cursor % len(dom_names)]
            qt, at = by_dom[dn][ptrs[dn] % len(by_dom[dn])]
            ptrs[dn] += 1; t_cursor += 1
            ptok = list(tk.encode(vc.format_gemma4_turn(qt, None), add_special_tokens=False).ids)
            rtok = list(tk.encode(f"{at}<turn|>\n", add_special_tokens=False).ids)
            if not ptok or not rtok:
                continue
            cur_toks.extend(ptok); cur_ans.extend([False] * len(ptok))
            cur_toks.extend(rtok); cur_ans.extend([True] * len(rtok))
        cur_toks, cur_ans = cur_toks[: SEQ_LEN + 1], cur_ans[: SEQ_LEN + 1]
        seq_ids_list.append(cur_toks[:-1]); seq_nxt_list.append(cur_toks[1:])
        seq_ans_list.append(cur_ans[1:]); seq_vlm_ans_list.append([False] * SEQ_LEN)
    print(f"[Stage 1] Packed {N_TEXT_SEQ} x {SEQ_LEN} = {N_TEXT_SEQ * SEQ_LEN:,} text tokens.", flush=True)

    # 2. Stage 2: 3-Bit Vision Calibration & Soft-Token Extraction
    VIS_CKPT_PATH = CKPT_DIR / "stage2_vision.pt"
    VIS_META_PATH = CKPT_DIR / "stage2_meta.pt"
    VLM_SOFT_CACHE = CKPT_DIR / "stage2_vlm_soft.pt"

    def _build_stage2_vision() -> dict:
        print("\n[Stage 2] 3-Bit Vision Calibration (V00..V26 + embed_vision, COCO val[64:] + Optical VA)...", flush=True)
        cfg_auto = AutoConfig.from_pretrained(str(model_dir))
        proc = Gemma4ImageProcessor.from_pretrained(str(model_dir))

        def make_optical_va_pil(idx: int) -> Tuple[Image.Image, str]:
            im = Image.new("RGB", (672, 672), (248, 248, 248))
            dr = ImageDraw.Draw(im)
            dr.text((18, 18), f"GEMMA4 31B V7.0 LM=60 VIS=27 FRAME={idx:03d}", fill=(12, 18, 28))
            dr.text((18, 42), "OPTICAL VA PRECISION ANALYSIS C8DB-H8", fill=(12, 18, 28))
            rng_v = np.random.default_rng(1000 + idx)
            for b in range(60):
                bh = int(rng_v.integers(25, 118))
                col = (35, 115, 210) if b >= 58 else ((30, 165, 95) if b % 5 != 0 else (210, 65, 45))
                dr.rectangle([24 + b * 10, 550 - bh, 31 + b * 10, 550], fill=col)
            cap = f"An optical diagnostic chart labeled GEMMA4 31B V7.0 LM=60 VIS=27 FRAME={idx:03d} displaying 60 vertical green, red, and blue bar segments on a light background."
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

        vis_teacher = Gemma4VisionModel(cfg_auto.vision_config).to(torch.float16).to(device).eval()
        vis_teacher.load_state_dict({k: v.to(device, dtype=torch.float16) for k, v in vis_sd.items()}, strict=True)
        mm_teacher = Gemma4MultimodalEmbedder(cfg_auto.vision_config, cfg_auto.text_config).to(torch.float16).to(device).eval()
        mm_teacher.load_state_dict({k: v.to(device, dtype=torch.float16) for k, v in embv_sd.items()}, strict=True)

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
                pv, pos = pin["pixel_values"].to(device, dtype=torch.float16), pin["image_position_ids"].to(device)
                st = mm_teacher(vis_teacher(pixel_values=pv, pixel_position_ids=pos).last_hidden_state).reshape(-1, D_MOD)
                vis_eval_pvs.append(pv.cpu()); vis_eval_poss.append(pos.cpu()); vis_teach_soft.append(st.to(torch.float16).cpu())
        for h in hooks:
            h.remove()
        heldout_pvs, heldout_poss, heldout_teach_soft = [], [], []
        with torch.no_grad():
            for im_h, _ in heldout_pairs:
                pin = proc(images=[im_h], return_tensors="pt")
                pv, pos = pin["pixel_values"].to(device, dtype=torch.float16), pin["image_position_ids"].to(device)
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
            vis_deq[name] = weff[:, :di].to(torch.float16)

        for ax in (0, 1):
            nm_p = f"model.vision_tower.patch_embedder.position_embedding_table.{ax}"
            pb, invs, lrt, we = pack_matrix_gpu_gptq(pos_tbl_f32[ax], None, None, mode=4, bsize=64, lr_rank=0)
            vision_records.append((nm_p, pos_tbl_f32.shape[1], pos_tbl_f32.shape[2], 64, 3, False, True, pb, invs, lrt))
            vis_deq[nm_p] = we
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

        vis_stud = Gemma4VisionModel(cfg_auto.vision_config).to(torch.float16)
        vis_stud.load_state_dict(vis_sd, strict=True)
        with torch.no_grad():
            vis_stud.patch_embedder.input_proj.weight.copy_(vis_deq["model.vision_tower.patch_embedder.input_proj.weight"])
            vis_stud.patch_embedder.position_embedding_table[0].copy_(vis_deq["model.vision_tower.patch_embedder.position_embedding_table.0"])
            vis_stud.patch_embedder.position_embedding_table[1].copy_(vis_deq["model.vision_tower.patch_embedder.position_embedding_table.1"])
            for lv, lvm in enumerate(vis_stud.encoder.layers):
                fp_v = f"model.vision_tower.encoder.layers.{lv}"
                for sub, ms in [("self_attn.q_proj", lvm.self_attn.q_proj), ("self_attn.k_proj", lvm.self_attn.k_proj), ("self_attn.v_proj", lvm.self_attn.v_proj), ("self_attn.o_proj", lvm.self_attn.o_proj), ("mlp.gate_proj", lvm.mlp.gate_proj), ("mlp.up_proj", lvm.mlp.up_proj), ("mlp.down_proj", lvm.mlp.down_proj)]:
                    ms.linear.weight.copy_(vis_deq[f"{fp_v}.{sub}.linear.weight"])
        vis_stud = vis_stud.to(device).eval()
        mm_stud = Gemma4MultimodalEmbedder(cfg_auto.vision_config, cfg_auto.text_config).to(torch.float16)
        mm_stud.load_state_dict(embv_sd, strict=True)
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

        print(f"[Stage 2 DONE] 3-Bit Vision ({len(vision_records)} mats, {v_bytes/1e6:.1f} MB) val cos: {v_cos0:.5f}->{v_cos1:.5f} (keep={v_keep})", flush=True)
        _atomic_torch_save({"vision_records": vision_records}, VIS_CKPT_PATH)
        _atomic_torch_save({"vis_teach_soft": vis_teach_soft, "vis_stud_soft": v_stud_s, "vis_captions": vis_captions, "heldout_teach_soft": heldout_teach_soft, "heldout_stud_soft": h_stud_s, "heldout_captions": [p[1] for p in heldout_pairs]}, VLM_SOFT_CACHE)
        meta = {"vis_total_bytes": int(v_bytes), "n_vis_mats": len(vision_records), "vision_norms": vision_norms, "vis_keep_c8d": v_keep, "cents_vis": c_v, "experts_vis": e_v, "v_cos0": v_cos0, "v_cos1": v_cos1}
        _atomic_torch_save(meta, VIS_META_PATH)
        sync_checkpoint_to_drive()
        return meta

    if VIS_META_PATH.is_file() and VIS_CKPT_PATH.is_file() and VLM_SOFT_CACHE.is_file():
        vm = torch.load(VIS_META_PATH, map_location="cpu", weights_only=False)
        print(f"[Resume Stage 2] Loaded {VIS_META_PATH} ({vm['n_vis_mats']} mats)", flush=True)
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

    def _build_mask_tensor(n_items: int, spans: Dict[int, Tuple[int, int]]) -> Optional[torch.Tensor]:
        if not spans:
            return None
        mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=device, dtype=torch.bool)).unsqueeze(0).expand(n_items, -1, -1).clone()
        for gi, (s0, s1) in spans.items():
            if 0 <= gi < n_items:
                mask[gi, s0:s1, s0:s1] = True
        return mask.unsqueeze(1)

    vlm_full_mask_gpu = _build_mask_tensor(N_SEQ, vlm_spans)
    heldout_full_mask_gpu = _build_mask_tensor(len(heldout_captions), heldout_spans)

    pos_1d = torch.arange(SEQ_LEN, device=device, dtype=torch.float32)

    def make_rope(theta: float, rot_d: int, hd: int, max_len: int = SEQ_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
        p1 = torch.arange(max_len, device=device, dtype=torch.float32)
        hr = rot_d // 2
        inv_f = 1.0 / (theta ** (torch.arange(0, 2 * hr, 2, device=device, dtype=torch.float32) / float(hd)))
        ang = p1[:, None] * inv_f[None, :]
        return torch.cos(ang).to(torch.float16), torch.sin(ang).to(torch.float16)

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
        return ((y * w.float()) if w is not None else y).to(torch.float16)

    # 3. Stage 3: Full-Rank GPTQ + perm_mlp + Live CRTR/MoD(q20) across L00..L59
    STAGE3_CKPT = CKPT_DIR / "stage3_layer_ckpt.pt"
    CRTR_CKPT = CKPT_DIR / "crtr_layers.pt"
    CKPT_X_TEACH = CKPT_DIR / "v7_ckpt_x_teach.pt"
    CKPT_X_STUD = CKPT_DIR / "v7_ckpt_x_stud.pt"
    _tr_lin = np.linspace(0, (N_SEQ - 48) * SEQ_LEN - 1, 16384, dtype=int)
    val_set = set(np.concatenate([np.linspace(0, N_TEXT_SEQ - 1, 32, dtype=int), np.linspace(N_TEXT_SEQ, N_SEQ - 1, 16, dtype=int)]).tolist())
    tr_seq_idx = np.array([i for i in range(N_SEQ) if i not in val_set], dtype=int)
    va_seq_idx = np.array(sorted(val_set), dtype=int)
    tr_crtr_s, tr_crtr_t = tr_seq_idx[_tr_lin // SEQ_LEN], _tr_lin % SEQ_LEN
    _va_lin = np.linspace(0, len(va_seq_idx) * SEQ_LEN - 1, 4096, dtype=int)
    va_crtr_s, va_crtr_t = va_seq_idx[_va_lin // SEQ_LEN], _va_lin % SEQ_LEN

    MB_FWD = int(hw_cfg["batch_seq"])
    RES_GPU = bool(hw_cfg["resident_gpu"])
    ALS_EV = int(hw_cfg["als_ev_toks"])
    _PIN_CPU = (not RES_GPU) and torch.cuda.is_available()

    def _alloc_act_store(shape: Tuple[int, ...], dev: torch.device) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.float16, device=dev, pin_memory=(dev.type == "cpu" and _PIN_CPU))

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
        total_mats = expected_lm_mats + n_vis_mats

        def unpack_qat(pfx: str) -> torch.Tensor:
            wp = sf.get_tensor(f"{pfx}.weight_packed").to(device)
            ws = sf.get_tensor(f"{pfx}.weight_scale").to(device, dtype=torch.float32)
            sh = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
            return ((((wp.unsqueeze(-1) >> sh) & 0xF) - 8).to(torch.float32).reshape(wp.shape[0], -1, 32) * ws.unsqueeze(-1)).reshape(wp.shape[0], -1).cpu()

        def teacher_layer_step(X_c: torch.Tensor, w_f16: Dict[str, torch.Tensor], ng: Dict[str, torch.Tensor], li: int, mask_full: Optional[torch.Tensor], ret_energy: bool = False):
            is_f = LAYER_TYPES[li] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cr, sr = make_rope(THETA_G if is_f else THETA_L, rot_d, hd)
            rep = NH // nkv
            lsc = float(ng["lsc"])
            out_c = _alloc_act_store(tuple(X_c.shape), X_c.device)
            neuron_e = torch.zeros((21504,), device=device, dtype=torch.float64) if ret_energy else None
            with torch.no_grad():
                for b0 in range(0, X_c.shape[0], MB_FWD):
                    b1 = min(b0 + MB_FWD, X_c.shape[0])
                    xb = X_c[b0:b1].to(device, non_blocking=True)
                    inp = rmsn_gpu(xb, ng["in"])
                    q = F.linear(inp, w_f16["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                    k = F.linear(inp, w_f16["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    v = k.clone() if (is_f and K_EQ_V) else F.linear(inp, w_f16["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                    qn = apply_rope(rmsn_gpu(q, ng["qn"]), cr, sr, rot_d, hd).transpose(1, 2)
                    kn = apply_rope(rmsn_gpu(k, ng["kn"]), cr, sr, rot_d, hd).transpose(1, 2)
                    vn = rmsn_gpu(v, None).transpose(1, 2)
                    am = None if (is_f or mask_full is None) else mask_full[b0:b1]
                    ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(rep, 1), vn.repeat_interleave(rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0)
                    xm = xb + rmsn_gpu(F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd), w_f16["o_proj"]), ng["pa"])
                    hm = rmsn_gpu(xm, ng["pf"])
                    act = F.gelu(F.linear(hm, w_f16["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_f16["up_proj"]).float()
                    if neuron_e is not None:
                        neuron_e.add_((act * act).reshape(-1, 21504).double().sum(dim=0))
                    out_c[b0:b1] = ((xm.float() + rmsn_gpu(F.linear(act.to(torch.float16), w_f16["down_proj"]), ng["pff"]).float()) * lsc).to(torch.float16).to(out_c.device)
            return out_c, neuron_e

        crtr_layers: Dict[int, dict] = {}
        if CRTR_CKPT.is_file():
            crtr_layers = torch.load(CRTR_CKPT, map_location="cpu", weights_only=False)

        if STAGE3_CKPT.is_file() and OUT_T16C.is_file() and CKPT_X_TEACH.is_file() and CKPT_X_STUD.is_file():
            lc = torch.load(STAGE3_CKPT, map_location="cpu", weights_only=False)
            start_layer = int(lc["next_layer"])
            t16c_pos = int(lc["t16c_pos"])
            lm_mats_count = int(lc["lm_mats_count"])
            bridge_cache: Dict[str, torch.Tensor] = lc["bridge_cache"]
            H_teach_eval = lc["H_teach_eval"].to(device if RES_GPU else "cpu", dtype=torch.float16)
            H_stud_eval = lc["H_stud_eval"].to(device if RES_GPU else "cpu", dtype=torch.float16)
            layer_cos_log: List[dict] = lc["layer_cos_log"]
            X_teach_cpu = torch.load(CKPT_X_TEACH, map_location="cpu", weights_only=True).to(device if RES_GPU else "cpu")
            X_stud_cpu = torch.load(CKPT_X_STUD, map_location="cpu", weights_only=True).to(device if RES_GPU else "cpu")
            del lc
            if OUT_T16C.stat().st_size < t16c_pos:
                raise RuntimeError(f"FAIL-CLOSED Resume: {OUT_T16C} size={OUT_T16C.stat().st_size:,} < ckpt pos={t16c_pos:,}")
            verify_t16c_integrity(OUT_T16C, expected_mats=lm_mats_count, expected_end_pos=t16c_pos)
            assert all(k in crtr_layers for k in range(start_layer)), "Missing CRTR layers up to resume point"
            t16c_fp = open(OUT_T16C, "r+b"); t16c_fp.seek(t16c_pos); t16c_fp.truncate(t16c_pos)
            print(f"[Resume Stage 3] L{start_layer:02d} (mats={lm_mats_count}, pos={t16c_pos:,} B)", flush=True)
        else:
            start_layer, lm_mats_count, bridge_cache, layer_cos_log = 0, 1, {}, []
            emb_bf16 = sf.get_tensor("model.language_model.embed_tokens.weight")
            ids_t, eids_t = torch.from_numpy(all_seq_ids).long(), torch.from_numpy(heldout_seq_ids).long()
            X_teach_cpu = (emb_bf16[ids_t].float() * ESCALE).to(torch.float16).to(device if RES_GPU else "cpu")
            H_teach_eval = (emb_bf16[eids_t].float() * ESCALE).to(torch.float16).to(device if RES_GPU else "cpu")
            for vi in range(N_VLM_SEQ):
                s0, s1 = vlm_spans[N_TEXT_SEQ + vi]; X_teach_cpu[N_TEXT_SEQ + vi, s0:s1] = vlm_teach_soft[vi].to(X_teach_cpu.device)
            for hi in range(len(heldout_captions)):
                s0, s1 = heldout_spans[hi]; H_teach_eval[hi, s0:s1] = heldout_teach_soft[hi].to(H_teach_eval.device)
            pb_e, inv_e, lr_e, emb_deq = pack_matrix_gpu_gptq(emb_bf16, None, None, mode=4, bsize=256, lr_rank=0, out_device=torch.device("cpu"))
            X_stud_cpu = (emb_deq[ids_t] * ESCALE).to(torch.float16).to(device if RES_GPU else "cpu")
            H_stud_eval = (emb_deq[eids_t] * ESCALE).to(torch.float16).to(device if RES_GPU else "cpu")
            for vi in range(N_VLM_SEQ):
                s0, s1 = vlm_spans[N_TEXT_SEQ + vi]; X_stud_cpu[N_TEXT_SEQ + vi, s0:s1] = vlm_stud_soft[vi].to(X_stud_cpu.device)
            for hi in range(len(heldout_captions)):
                s0, s1 = heldout_spans[hi]; H_stud_eval[hi, s0:s1] = heldout_stud_soft[hi].to(H_stud_eval.device)
            del emb_bf16, emb_deq, ids_t, eids_t
            t16c_fp = open(OUT_T16C, "wb")
            t16c_fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, n_all_norms, total_mats, 0))
            t16c_pos = 24
            for _nm, _vc_a in all_norms.items():
                _nb = _nm.encode("utf-8")
                _vf = np.ascontiguousarray(_vc_a, dtype="<f4").ravel()
                t16c_fp.write(struct.pack("<H", len(_nb))); t16c_fp.write(_nb); t16c_fp.write(struct.pack("<I", _vf.size)); t16c_fp.write(_vf.tobytes())
                t16c_pos += 2 + len(_nb) + 4 + _vf.size * 4
            t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, "model.language_model.embed_tokens.weight", int(tc["vocab_size"]), D_MOD, 256, 3, False, True, pb_e, inv_e, lr_e)
            del pb_e, inv_e, lr_e
            t16c_fp.flush()

        del vlm_teach_soft, vlm_stud_soft, heldout_teach_soft, heldout_stud_soft
        gc.collect()

        try:
            for l_idx in range(start_layer, N_LAYERS):
                tl = time.perf_counter()
                p = f"model.language_model.layers.{l_idx}."
                is_f = LAYER_TYPES[l_idx] == "full_attention"
                nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
                rot_d = int(round(hd * (PART_G if is_f else PART_L)))
                cr_l, sr_l = make_rope(THETA_G if is_f else THETA_L, rot_d, hd)
                rep = NH // nkv
                ng = {
                    "in": sf.get_tensor(p + "input_layernorm.weight").to(device), "qn": sf.get_tensor(p + "self_attn.q_norm.weight").to(device),
                    "kn": sf.get_tensor(p + "self_attn.k_norm.weight").to(device), "pa": sf.get_tensor(p + "post_attention_layernorm.weight").to(device),
                    "pf": sf.get_tensor(p + "pre_feedforward_layernorm.weight").to(device), "pff": sf.get_tensor(p + "post_feedforward_layernorm.weight").to(device),
                    "lsc": sf.get_tensor(p + "layer_scalar").float().item(),
                }
                w_teach = {pn: unpack_qat(f"{p}{'self_attn' if pn in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'}.{pn}") for pn in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if not (is_f and K_EQ_V and pn == "v_proj")}
                w_f16 = {k: v.to(device, dtype=torch.float16) for k, v in w_teach.items()}
                X_teach_next, neuron_e = teacher_layer_step(X_teach_cpu, w_f16, ng, l_idx, vlm_full_mask_gpu, ret_energy=True)
                H_teach_eval, _ = teacher_layer_step(H_teach_eval, w_f16, ng, l_idx, heldout_full_mask_gpu, ret_energy=False)
                del w_f16
                if l_idx < 58:
                    del X_teach_cpu

                # Co-activation GeGLU permutation BEFORE SmoothQuant/GPTQ
                w_teach["gate_proj"], w_teach["up_proj"], w_teach["down_proj"], top50_e_pct = permute_geglu_weights(
                    w_teach["gate_proj"], w_teach["up_proj"], w_teach["down_proj"], neuron_e
                )
                del neuron_e

                if l_idx == 58:
                    _atomic_torch_save(X_stud_cpu.cpu(), Path("/tmp/v7_x_stud_pre58.pt"))
                    _atomic_torch_save(H_stud_eval.cpu(), Path("/tmp/v7_h_eval_pre58.pt"))

                n_tok_l = N_SEQ * SEQ_LEN
                w_stud: Dict[str, torch.Tensor] = {}
                h_in = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32)
                xsq_in = torch.zeros((D_MOD,), device=device, dtype=torch.float32)
                x_ev_in = []
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        inp = rmsn_gpu(X_stud_cpu[b0 : min(b0 + MB_FWD, N_SEQ)].to(device, non_blocking=True), ng["in"]).reshape(-1, D_MOD).float()
                        accum_hess(h_in, xsq_in, inp)
                        if (len(x_ev_in) < 3 or b0 >= N_TEXT_SEQ) and len(x_ev_in) < 4:
                            x_ev_in.append(inp[:1024].clone())
                xrms_in, xev_in = torch.sqrt((xsq_in / n_tok_l).clamp(min=1e-8)), torch.cat(x_ev_in, 0)[:ALS_EV]
                for pn in (["q_proj", "k_proj"] if (is_f and K_EQ_V) else ["q_proj", "k_proj", "v_proj"]):
                    mi, bsi, rki, trm = get_lm_spec_v7_0(l_idx, pn)
                    nb, asy = spec_to_t16c_bits(mi, trm)
                    pb, invs, lrt, we = pack_matrix_gpu_gptq(w_teach[pn], h_in, xrms_in, mi, bsi, rki, xev_in, to_rowmajor=trm)
                    t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, f"{p}self_attn.{pn}.weight", w_teach[pn].shape[0], D_MOD, bsi, nb, asy, True, pb, invs, lrt)
                    lm_mats_count += 1; w_stud[pn] = we
                    del pb, invs, lrt
                del h_in, xsq_in, xrms_in, xev_in, x_ev_in

                if l_idx in (58, 59):
                    sub_idx = np.concatenate([np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int), np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int)])
                    sub_t_t = torch.from_numpy(sub_idx).long().to(X_teach_cpu.device)
                    sub_t_s = torch.from_numpy(sub_idx).long().to(X_stud_cpu.device)
                    with torch.no_grad():
                        inpt = rmsn_gpu(X_teach_cpu[sub_t_t].to(device, non_blocking=True), ng["in"]).reshape(-1, D_MOD).float()
                        if l_idx == 58:
                            inps = rmsn_gpu(X_stud_cpu[sub_t_s].to(device, non_blocking=True), ng["in"]).reshape(-1, D_MOD).float()
                            bridge_cache["in_58_stud"] = inps.to(torch.float16).cpu()
                            bridge_cache["kt58"] = (inpt @ w_teach["k_proj"].to(device).T).to(torch.float16).cpu()
                            bridge_cache["ks58_base"] = (inps @ w_stud["k_proj"].float().T).to(torch.float16).cpu()
                            bridge_cache["vt58"] = (inpt @ w_teach["v_proj"].to(device).T).to(torch.float16).cpu()
                            bridge_cache["vs58_base"] = (inps @ w_stud["v_proj"].float().T).to(torch.float16).cpu()
                            del inps
                        else:
                            bridge_cache["kt59"] = (inpt @ w_teach["k_proj"].to(device).T).to(torch.float16).cpu()
                        del inpt, X_teach_cpu

                ctx_all = _alloc_act_store((N_SEQ, SEQ_LEN, NH * hd), device if RES_GPU else torch.device("cpu"))
                do_o, di_o = w_teach["o_proj"].shape
                h_o, xsq_o, x_ev_o = torch.zeros((di_o, di_o), device=device, dtype=torch.float32), torch.zeros((di_o,), device=device, dtype=torch.float32), []
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        inp = rmsn_gpu(X_stud_cpu[b0:b1].to(device, non_blocking=True), ng["in"])
                        q = F.linear(inp, w_stud["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                        k = F.linear(inp, w_stud["k_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                        v = k.clone() if (is_f and K_EQ_V) else F.linear(inp, w_stud["v_proj"]).view(b1 - b0, SEQ_LEN, nkv, hd)
                        qn, kn, vn = apply_rope(rmsn_gpu(q, ng["qn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2), apply_rope(rmsn_gpu(k, ng["kn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2), rmsn_gpu(v, None).transpose(1, 2)
                        am = None if (is_f or vlm_full_mask_gpu is None) else vlm_full_mask_gpu[b0:b1]
                        ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(rep, 1), vn.repeat_interleave(rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0).transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd)
                        accum_hess(h_o, xsq_o, ctx.reshape(-1, di_o).float())
                        if (len(x_ev_o) < 3 or b0 >= N_TEXT_SEQ) and len(x_ev_o) < 4:
                            x_ev_o.append(ctx.reshape(-1, di_o).float()[:1024].clone())
                        ctx_all[b0:b1] = ctx.to(ctx_all.device)
                mi, bsi, rki, trm = get_lm_spec_v7_0(l_idx, "o_proj")
                nb, asy = spec_to_t16c_bits(mi, trm)
                ho_box = [h_o]; del h_o
                pb, invs, lrt, w_o_eff = pack_matrix_gpu_gptq(w_teach["o_proj"], ho_box, torch.sqrt((xsq_o / n_tok_l).clamp(min=1e-8)), mi, bsi, rki, torch.cat(x_ev_o, 0)[:ALS_EV], to_rowmajor=trm)
                t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, f"{p}self_attn.o_proj.weight", do_o, di_o, bsi, nb, asy, True, pb, invs, lrt)
                lm_mats_count += 1; w_stud["o_proj"] = w_o_eff
                del ho_box, xsq_o, x_ev_o, pb, invs, lrt

                with torch.no_grad():
                    _hg = H_stud_eval.to(device)
                    _hi = rmsn_gpu(_hg, ng["in"])
                    _hq = F.linear(_hi, w_stud["q_proj"]).view(_hg.shape[0], SEQ_LEN, NH, hd)
                    _hk = F.linear(_hi, w_stud["k_proj"]).view(_hg.shape[0], SEQ_LEN, nkv, hd)
                    _hv = _hk.clone() if (is_f and K_EQ_V) else F.linear(_hi, w_stud["v_proj"]).view(_hg.shape[0], SEQ_LEN, nkv, hd)
                    _amh = None if is_f else heldout_full_mask_gpu
                    _hc = F.scaled_dot_product_attention(apply_rope(rmsn_gpu(_hq, ng["qn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2), apply_rope(rmsn_gpu(_hk, ng["kn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2).repeat_interleave(rep, 1), rmsn_gpu(_hv, None).transpose(1, 2).repeat_interleave(rep, 1), attn_mask=_amh, is_causal=(_amh is None), scale=1.0).transpose(1, 2).contiguous().view(_hg.shape[0], SEQ_LEN, NH * hd)
                    H_stud_eval_mid = (_hg + rmsn_gpu(F.linear(_hc, w_o_eff), ng["pa"])).to(device if RES_GPU else "cpu")
                    del _hg, _hi, _hq, _hk, _hv, _amh, _hc

                # Extract CRTR MoD input slices before freeing X_stud_cpu
                xp_crtr_tr = X_stud_cpu[torch.from_numpy(tr_crtr_s).long().to(X_stud_cpu.device), torch.from_numpy(tr_crtr_t).long().to(X_stud_cpu.device)].cpu()
                xp_crtr_va = X_stud_cpu[torch.from_numpy(va_crtr_s).long().to(X_stud_cpu.device), torch.from_numpy(va_crtr_t).long().to(X_stud_cpu.device)].cpu()
                h_mlp, xsq_mlp, x_ev_m = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32), torch.zeros((D_MOD,), device=device, dtype=torch.float32), []
                xm_all = _alloc_act_store((N_SEQ, SEQ_LEN, D_MOD), device if RES_GPU else torch.device("cpu"))
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        xm = X_stud_cpu[b0:b1].to(device, non_blocking=True) + rmsn_gpu(F.linear(ctx_all[b0:b1].to(device, non_blocking=True), w_o_eff), ng["pa"])
                        xm_all[b0:b1] = xm.to(xm_all.device)
                        hmf = rmsn_gpu(xm, ng["pf"]).reshape(-1, D_MOD).float()
                        accum_hess(h_mlp, xsq_mlp, hmf)
                        if (len(x_ev_m) < 3 or b0 >= N_TEXT_SEQ) and len(x_ev_m) < 4:
                            x_ev_m.append(hmf[:1024].clone())
                del ctx_all, X_stud_cpu, w_o_eff
                for apn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    w_teach.pop(apn, None)
                    if l_idx in (58, 59) and apn in w_stud:
                        w_stud[apn] = w_stud[apn].cpu()
                    else:
                        w_stud.pop(apn, None)
                torch.cuda.empty_cache()

                xrms_mlp, xev_mlp = torch.sqrt((xsq_mlp / n_tok_l).clamp(min=1e-8)), torch.cat(x_ev_m, 0)[:ALS_EV]
                for pn in ("gate_proj", "up_proj"):
                    mi, bsi, rki, trm = get_lm_spec_v7_0(l_idx, pn)
                    nb, asy = spec_to_t16c_bits(mi, trm)
                    pb, invs, lrt, we = pack_matrix_gpu_gptq(w_teach.pop(pn), h_mlp, xrms_mlp, mi, bsi, rki, xev_mlp, to_rowmajor=trm)
                    t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, f"{p}mlp.{pn}.weight", 21504, D_MOD, bsi, nb, asy, True, pb, invs, lrt)
                    lm_mats_count += 1; w_stud[pn] = we
                    del pb, invs, lrt
                del h_mlp, xsq_mlp, xrms_mlp, xev_mlp, x_ev_m

                do_d, di_d = w_teach["down_proj"].shape
                h_d, xsq_d, x_ev_d = torch.zeros((di_d, di_d), device=device, dtype=torch.float32), torch.zeros((di_d,), device=device, dtype=torch.float32), []
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        hm = rmsn_gpu(xm_all[b0 : min(b0 + MB_FWD, N_SEQ)].to(device, non_blocking=True), ng["pf"])
                        act = F.gelu(F.linear(hm, w_stud["gate_proj"]).float(), approximate="tanh") * F.linear(hm, w_stud["up_proj"]).float()
                        af = act.reshape(-1, di_d)
                        accum_hess(h_d, xsq_d, af)
                        if (len(x_ev_d) < 3 or b0 >= N_TEXT_SEQ) and len(x_ev_d) < 4:
                            x_ev_d.append(af[:1024].clone())
                w_stud["gate_proj"] = w_stud["gate_proj"].cpu()
                w_stud["up_proj"] = w_stud["up_proj"].cpu()
                torch.cuda.empty_cache()

                mi, bsi, rki, trm = get_lm_spec_v7_0(l_idx, "down_proj")
                nb, asy = spec_to_t16c_bits(mi, trm)
                hd_box = [h_d]; del h_d
                pb, invs, lrt, w_d_eff = pack_matrix_gpu_gptq(w_teach.pop("down_proj"), hd_box, torch.sqrt((xsq_d / n_tok_l).clamp(min=1e-8)), mi, bsi, rki, torch.cat(x_ev_d, 0)[:ALS_EV], to_rowmajor=trm)
                t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, f"{p}mlp.down_proj.weight", do_d, di_d, bsi, nb, asy, True, pb, invs, lrt)
                lm_mats_count += 1
                del hd_box, xsq_d, x_ev_d, w_teach, pb, invs, lrt

                lsc = float(ng["lsc"])
                if l_idx in (58, 59):
                    _atomic_torch_save({"w": {k: v.cpu() for k, v in w_stud.items()} | {"down_proj": w_d_eff.cpu()}, "norms": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in ng.items()}}, Path(f"/tmp/v7_l{l_idx}_stud.pt"))
                wg_g, wu_g = w_stud["gate_proj"].to(device), w_stud["up_proj"].to(device)
                with torch.no_grad():
                    _hmg = H_stud_eval_mid.to(device)
                    _hact = (F.gelu(F.linear(rmsn_gpu(_hmg, ng["pf"]), wg_g).float(), approximate="tanh") * F.linear(rmsn_gpu(_hmg, ng["pf"]), wu_g).float()).to(torch.float16)
                    H_stud_eval = ((_hmg.float() + rmsn_gpu(F.linear(_hact, w_d_eff), ng["pff"]).float()) * lsc).to(torch.float16).to(device if RES_GPU else "cpu")
                    del H_stud_eval_mid
                X_stud_next = _alloc_act_store((N_SEQ, SEQ_LEN, D_MOD), device if RES_GPU else torch.device("cpu"))
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        xm_g = xm_all[b0:b1].to(device, non_blocking=True); hm = rmsn_gpu(xm_g, ng["pf"])
                        act = (F.gelu(F.linear(hm, wg_g).float(), approximate="tanh") * F.linear(hm, wu_g).float()).to(torch.float16)
                        X_stud_next[b0:b1] = ((xm_g.float() + rmsn_gpu(F.linear(act, w_d_eff), ng["pff"]).float()) * lsc).to(torch.float16).to(X_stud_next.device)
                xm_crtr_tr = xm_all[torch.from_numpy(tr_crtr_s).long().to(xm_all.device), torch.from_numpy(tr_crtr_t).long().to(xm_all.device)].cpu()
                xm_crtr_va = xm_all[torch.from_numpy(va_crtr_s).long().to(xm_all.device), torch.from_numpy(va_crtr_t).long().to(xm_all.device)].cpu()
                del xm_all, w_d_eff, w_stud
                torch.cuda.empty_cache()

                crec, cm = train_layer_crtr_v7_0(
                    l_idx,
                    xp_crtr_tr, xm_crtr_tr, X_stud_next[torch.from_numpy(tr_crtr_s).long().to(X_stud_next.device), torch.from_numpy(tr_crtr_t).long().to(X_stud_next.device)].cpu(),
                    xp_crtr_va, xm_crtr_va, X_stud_next[torch.from_numpy(va_crtr_s).long().to(X_stud_next.device), torch.from_numpy(va_crtr_t).long().to(X_stud_next.device)].cpu(),
                    ng["in"], ng["pf"], wg_g, wu_g, rmsn_gpu,
                    block_size=256, rank=32, top_p=0.30, mod_quantile=0.20, steps=200, eps=EPS, device=device,
                )
                crtr_layers[l_idx] = crec
                del xp_crtr_tr, xp_crtr_va, xm_crtr_tr, xm_crtr_va, wg_g, wu_g

                va16_t = torch.from_numpy(va_seq_idx[:16]).long().to(X_stud_next.device)
                cos_l_txt = float(F.cosine_similarity(X_stud_next[va16_t].reshape(-1, D_MOD).float().to(device), X_teach_next[va16_t].reshape(-1, D_MOD).float().to(device), dim=-1).mean().item())
                cos_l_vlm = float(F.cosine_similarity(H_stud_eval.reshape(-1, D_MOD).float().to(device), H_teach_eval.reshape(-1, D_MOD).float().to(device), dim=-1).mean().item())
                layer_cos_log.append({"l": l_idx, "txt_cos": round(cos_l_txt, 5), "vlm_cos": round(cos_l_vlm, 5), "top50_e_pct": round(top50_e_pct, 1), "crtr": cm})
                X_teach_cpu, X_stud_cpu = X_teach_next, X_stud_next
                del X_teach_next, X_stud_next
                torch.cuda.empty_cache()

                if (l_idx + 1) % 4 == 0:
                    t16c_fp.flush(); os.fsync(t16c_fp.fileno())
                    _atomic_torch_save(X_teach_cpu.cpu(), CKPT_X_TEACH)
                    _atomic_torch_save(X_stud_cpu.cpu(), CKPT_X_STUD)
                    _atomic_torch_save(crtr_layers, CRTR_CKPT)
                    _atomic_torch_save({"next_layer": l_idx + 1, "t16c_pos": t16c_pos, "lm_mats_count": lm_mats_count, "bridge_cache": bridge_cache, "H_teach_eval": H_teach_eval.cpu(), "H_stud_eval": H_stud_eval.cpu(), "layer_cos_log": layer_cos_log}, STAGE3_CKPT)
                    sync_checkpoint_to_drive(expected_t16c_pos=t16c_pos)
                    gc.collect()
                save_progress({"stage": f"L{l_idx:02d}_done", "next_layer": l_idx + 1, "txt_cos": round(cos_l_txt, 5), "vlm_cos": round(cos_l_vlm, 5)})
                va_gb, vp_gb, vt_gb = _v70_gpu_vram_stats_gb(device)
                print(
                    f"  [V7.0 L{l_idx:02d}/59 {'F' if is_f else 'S'}] txt_cos={cos_l_txt:.5f} vlm_cos={cos_l_vlm:.5f} "
                    f"| e50={top50_e_pct:.1f}% crtr_rec={cm['val_recall_after']:.3f} mod_acc={cm['mod_val_acc']:.3f} "
                    f"| vram={va_gb}/peak={vp_gb}/{vt_gb}GiB | dt={time.perf_counter()-tl:.1f}s",
                    flush=True,
                )

            assert lm_mats_count == expected_lm_mats, f"LM mat count mismatch: {lm_mats_count} != {expected_lm_mats}"
            v_tail = torch.load(VIS_CKPT_PATH, map_location="cpu", weights_only=False)
            for rec_v in v_tail["vision_records"]:
                t16c_pos = p3._write_mat_record(t16c_fp, t16c_pos, *rec_v)
            del v_tail
            t16c_fp.flush(); os.fsync(t16c_fp.fileno())
        finally:
            t16c_fp.close()

    # Verify & persist .t16c + .crtr atomically
    vm_t, ve_t = verify_t16c_integrity(OUT_T16C)
    t16c_sz = OUT_T16C.stat().st_size
    assert vm_t == total_mats and ve_t == t16c_sz and t16c_sz <= MAX_T16C_BYTES, f"Bad V7.0 T16C: mats={vm_t}/{total_mats} sz={t16c_sz}"
    assert len(crtr_layers) == N_LAYERS and all(k in crtr_layers for k in range(N_LAYERS))
    crtr_sz = crtr.export_crtr([crtr_layers[l] for l in range(N_LAYERS)], OUT_CRTR)
    cmeta, clayers = crtr.load_crtr(OUT_CRTR)
    assert len(clayers) == N_LAYERS and (cmeta["flags"] & crtr.CRTR_FLAG_MOD_GATE) != 0
    print(f"[Stage 3 DONE] {OUT_T16C.name}: {t16c_sz:,} B ({t16c_sz/1024**3:.3f} GiB) | {OUT_CRTR.name}: {crtr_sz:,} B", flush=True)
    persist_verified(OUT_CRTR)
    persist_verified(OUT_T16C)
    for rmp in (VIS_CKPT_PATH, VIS_META_PATH, VLM_SOFT_CACHE, STAGE3_CKPT, CRTR_CKPT, CKPT_X_TEACH, CKPT_X_STUD):
        rmp.unlink(missing_ok=True)
    if DRIVE_DIR is not None:
        shutil.rmtree(DRIVE_DIR / "ckpt_v7_0", ignore_errors=True)
        (DRIVE_DIR / (OUT_T16C.name + ".partial")).unlink(missing_ok=True)

    # 4. Stage 4A: Multimodal C8DB + Capture 128-Seq L58/L59 Student KV
    print("\n[Stage 4A] Training Multimodal C8DB + Capturing V7.0 L58/L59 KV (128 seqs)...", flush=True)
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
            delta, _, lb = m(x_tr[ib])
            p_y = ys_tr[ib] + delta
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
        wg = {k: v.to(device) for k, v in ck["w"].items()}
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
                    inp = rmsn_gpu(X_in_c[sel].to(device), ng["in"])
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
                xb = X_in_c[b0:b1].to(device); inp = rmsn_gpu(xb, ng["in"]); inp2 = inp.reshape(-1, D_MOD)
                q = F.linear(inp, wg["q_proj"]).view(b1 - b0, SEQ_LEN, NH, hd)
                k2 = F.linear(inp2, wg["k_proj"]).float()
                if adp_k is not None:
                    k2 = k2 + eval_c8d_gpu(inp2, adp_k[0], adp_k[1])
                k = k2.to(torch.float16).view(b1 - b0, SEQ_LEN, nkv, hd)
                if is_f and K_EQ_V:
                    v = k.clone()
                else:
                    v2 = F.linear(inp2, wg["v_proj"]).float()
                    if adp_v is not None:
                        v2 = v2 + eval_c8d_gpu(inp2, adp_v[0], adp_v[1])
                    v = v2.to(torch.float16).view(b1 - b0, SEQ_LEN, nkv, hd)
                qn = apply_rope(rmsn_gpu(q, ng["qn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2)
                kn = apply_rope(rmsn_gpu(k, ng["kn"]), cr_l, sr_l, rot_d, hd).transpose(1, 2)
                vn = rmsn_gpu(v, None).transpose(1, 2)
                for gi in range(b0, b1):
                    if gi in ast_set:
                        ast_k[gi] = kn[gi - b0].cpu()
                        if not (is_f and K_EQ_V):
                            ast_v[gi] = vn[gi - b0].cpu()
                am = None if (is_f or mask_full is None) else mask_full[b0:b1]
                ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(rep, 1), vn.repeat_interleave(rep, 1), attn_mask=am, is_causal=(am is None), scale=1.0).transpose(1, 2).contiguous().view(b1 - b0, SEQ_LEN, NH * hd)
                xm = xb + rmsn_gpu(F.linear(ctx, wg["o_proj"]), ng["pa"])
                hm = rmsn_gpu(xm, ng["pf"])
                act = (F.gelu(F.linear(hm, wg["gate_proj"]).float(), approximate="tanh") * F.linear(hm, wg["up_proj"]).float()).to(torch.float16)
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

    del X_stud_cpu; gc.collect()
    AST_SEQS = np.linspace(0, N_TEXT_SEQ - 1, 128, dtype=int).tolist()
    x58 = torch.load("/tmp/v7_x_stud_pre58.pt", map_location="cpu", weights_only=True); Path("/tmp/v7_x_stud_pre58.pt").unlink(missing_ok=True)
    x59_in, _, cap_kv58 = forward_adapted_layer(x58, Path("/tmp/v7_l58_stud.pt"), 58, adp_k58, adp_v58, cap_ast=AST_SEQS, mask_full=vlm_full_mask_gpu)
    hev58 = torch.load("/tmp/v7_h_eval_pre58.pt", map_location="cpu", weights_only=True); Path("/tmp/v7_h_eval_pre58.pt").unlink(missing_ok=True)
    hev59, _, _ = forward_adapted_layer(hev58, Path("/tmp/v7_l58_stud.pt"), 58, adp_k58, adp_v58, mask_full=heldout_full_mask_gpu)
    Path("/tmp/v7_l58_stud.pt").unlink(missing_ok=True)

    sub_59 = np.concatenate([np.linspace(0, N_TEXT_SEQ - 1, 96, dtype=int), np.linspace(N_TEXT_SEQ, N_SEQ - 1, 32, dtype=int)])
    _, c59, _ = forward_adapted_layer(x59_in, Path("/tmp/v7_l59_stud.pt"), 59, cap_sub=sub_59, capture_only=True)
    xk59, ytk59, ysk59 = c59["in"].reshape(-1, D_MOD).float(), bridge_cache.pop("kt59").float(), c59["k"].reshape(-1, 2048).float()
    del c59, bridge_cache
    ntr59 = (xk59.shape[0] * 4) // 5
    ent59, c59n, e59n = train_c8d_proj("model.language_model.layers.59.self_attn.k_proj.weight", xk59[:ntr59].to(device), ytk59[:ntr59].to(device), ysk59[:ntr59].to(device), xk59[ntr59:].to(device), ytk59[ntr59:].to(device), ysk59[ntr59:].to(device))
    if ent59 is not None:
        c8db_entries.append(ent59); adp_k59 = (c59n, e59n)
    del xk59, ytk59, ysk59
    X_stud_cpu, _, cap_kv59 = forward_adapted_layer(x59_in, Path("/tmp/v7_l59_stud.pt"), 59, adp_k59, cap_ast=AST_SEQS, mask_full=vlm_full_mask_gpu)
    H_stud_eval, _, _ = forward_adapted_layer(hev59, Path("/tmp/v7_l59_stud.pt"), 59, adp_k59, mask_full=heldout_full_mask_gpu)
    Path("/tmp/v7_l59_stud.pt").unlink(missing_ok=True)

    cos_l59 = float(F.cosine_similarity(X_stud_cpu[-16:].reshape(-1, D_MOD).float().to(device), X_teach_cpu[-16:].reshape(-1, D_MOD).float().to(device), dim=-1).mean().item())
    print(f"  [L59 Student vs Teacher VLM cos] {cos_l59:.5f}", flush=True)

    fn_g = fn_orig_cpu.to(device)
    num_w, den_w = torch.zeros((D_MOD,), device=device), torch.zeros((D_MOD,), device=device)
    val_sq_o = val_sq_w = 0.0; val_cnt = 0
    with torch.no_grad():
        for b0 in range(0, N_SEQ, MB_FWD):
            b1 = min(b0 + MB_FWD, N_SEQ)
            xtb = rmsn_gpu(X_teach_cpu[b0:b1].to(device), fn_g).reshape(-1, D_MOD).float()
            xub = rmsn_gpu(X_stud_cpu[b0:b1].to(device), None).reshape(-1, D_MOD).float()
            iv = torch.arange(xtb.shape[0], device=device) % 10 == 0
            num_w.add_((xub[~iv] * xtb[~iv]).sum(dim=0)); den_w.add_((xub[~iv] * xub[~iv]).sum(dim=0))
        fn_wiener = ((num_w / (N_SEQ * SEQ_LEN * 0.9)) + 0.05 * fn_g) / ((den_w / (N_SEQ * SEQ_LEN * 0.9)) + 0.05).clamp(min=1e-8)
        for b0 in range(0, N_SEQ, MB_FWD):
            b1 = min(b0 + MB_FWD, N_SEQ)
            xtb = rmsn_gpu(X_teach_cpu[b0:b1].to(device), fn_g).reshape(-1, D_MOD).float()
            xub = rmsn_gpu(X_stud_cpu[b0:b1].to(device), None).reshape(-1, D_MOD).float()
            iv = torch.arange(xtb.shape[0], device=device) % 10 == 0
            val_sq_o += float(((xub[iv] * fn_g - xtb[iv]) ** 2).sum().item())
            val_sq_w += float(((xub[iv] * fn_wiener - xtb[iv]) ** 2).sum().item())
            val_cnt += int(iv.sum().item()) * D_MOD
    fn_eff_g = fn_wiener if val_sq_w < val_sq_o else fn_g

    vlm_pos = np.flatnonzero(all_vlm_ans_mask.reshape(-1))
    txt_pos = np.flatnonzero((all_ans_mask & (~all_vlm_ans_mask)).reshape(-1))
    sel_v = vlm_pos[np.linspace(0, vlm_pos.size - 1, min(8192, vlm_pos.size), dtype=int)] if vlm_pos.size > 0 else np.array([], dtype=int)
    sel_t = txt_pos[np.linspace(0, txt_pos.size - 1, min(32768 - sel_v.size, txt_pos.size), dtype=int)]
    sel_pos = np.sort(np.concatenate([sel_t, sel_v]))
    s_t_cpu, t_t_cpu = torch.from_numpy(sel_pos // SEQ_LEN).long(), torch.from_numpy(sel_pos % SEQ_LEN).long()
    ht_ans = rmsn_gpu(X_teach_cpu[s_t_cpu.to(X_teach_cpu.device), t_t_cpu.to(X_teach_cpu.device)].to(device).unsqueeze(1), fn_g).squeeze(1).float()
    hs_ans = rmsn_gpu(X_stud_cpu[s_t_cpu, t_t_cpu].to(device).unsqueeze(1), fn_eff_g).squeeze(1).float()
    nxt_ans = torch.from_numpy(all_seq_nxt.reshape(-1)[sel_pos]).to(device, dtype=torch.long)
    ast_x59_stud = X_stud_cpu[AST_SEQS].clone()
    del X_teach_cpu, X_stud_cpu; gc.collect(); torch.cuda.empty_cache()

    with safe_open(sf_path, framework="pt", device="cpu") as sf_e:
        emb_gpu = sf_e.get_tensor("model.language_model.embed_tokens.weight").to(device, dtype=torch.float16)
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
    c8d_sz = c8d.write_c8db_file(OUT_C8D, [("model.language_model.norm.weight", fn_eff_g.detach().cpu().numpy().astype("<f4"))], c8db_entries)
    persist_verified(OUT_C8D)
    del p_teach_store, ht_ans, hs_ans, nxt_ans, y_star, t_top1, log_tail_q0, m60, mc60
    torch.cuda.empty_cache()

    # 5. Stage 4B: Target-Aligned 4-Layer Assistant (300 steps, 128 seqs) + BHDR d_byte=384 (`student_argmax`)
    print("\n[Stage 4B] Target-Aligning 4-Layer Assistant + Packing Q2/Q4/Q8 + Distilling BHDR (d_byte=384, student_argmax)...", flush=True)
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
                pa = f"model.layers.{la}."
                m = nn.Module()
                m.q_proj = nn.Parameter(sd[pa + "self_attn.q_proj.weight"].float().clone())
                m.o_proj = nn.Parameter(sd[pa + "self_attn.o_proj.weight"].float().clone())
                m.gate = nn.Parameter(sd[pa + "mlp.gate_proj.weight"].float().clone())
                m.up = nn.Parameter(sd[pa + "mlp.up_proj.weight"].float().clone())
                m.down = nn.Parameter(sd[pa + "mlp.down_proj.weight"].float().clone())
                m.n_in = nn.Parameter(sd[pa + "input_layernorm.weight"].float().clone())
                m.n_qn = nn.Parameter(sd[pa + "self_attn.q_norm.weight"].float().clone())
                m.n_pa = nn.Parameter(sd[pa + "post_attention_layernorm.weight"].float().clone())
                m.n_pf = nn.Parameter(sd[pa + "pre_feedforward_layernorm.weight"].float().clone())
                m.n_pff = nn.Parameter(sd[pa + "post_feedforward_layernorm.weight"].float().clone())
                m.lsc = nn.Parameter(sd[pa + "layer_scalar"].float().reshape(1).clone())
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
    print(f"  [Assistant V7 Aligned] Val KL={akl0:.4f}->{akl1:.4f} | Top1Agree={aagr0*100:.1f}%->{aagr1*100:.1f}%", flush=True)

    ast_sd["pre_projection.weight"] = ast_mod.pre.detach().to(torch.bfloat16).cpu()
    ast_sd["post_projection.weight"] = ast_mod.post.detach().to(torch.bfloat16).cpu()
    ast_sd["model.norm.weight"] = ast_mod.fn.detach().to(torch.bfloat16).cpu()
    for la, m in enumerate(ast_mod.layers):
        pa = f"model.layers.{la}."
        for k_s, par in [("self_attn.q_proj.weight", m.q_proj), ("self_attn.o_proj.weight", m.o_proj), ("mlp.gate_proj.weight", m.gate), ("mlp.up_proj.weight", m.up), ("mlp.down_proj.weight", m.down), ("input_layernorm.weight", m.n_in), ("self_attn.q_norm.weight", m.n_qn), ("post_attention_layernorm.weight", m.n_pa), ("pre_feedforward_layernorm.weight", m.n_pf), ("post_feedforward_layernorm.weight", m.n_pff)]:
            ast_sd[pa + k_s] = par.detach().to(torch.bfloat16).cpu()
        ast_sd[pa + "layer_scalar"] = m.lsc.detach().to(torch.bfloat16).cpu().reshape(ast_sd[pa + "layer_scalar"].shape)
    ft_sf = Path("/tmp/assistant_v7_ft.safetensors")
    st_save_file(ast_sd, str(ft_sf))
    for bits, bp in [(2, OUT_AST_Q2), (4, OUT_AST_Q4), (8, OUT_AST_Q8)]:
        bhmod.pack_assistant_body_from_safetensors(ft_sf, bp, body_bits=bits, block_size=64, has_fwht=(bits == 4))
    ft_sf.unlink(missing_ok=True)
    del opt_a, sch_a, ast_sd, best_asd, W_cand_g, W_ast_cand

    # Extract 128 x 350 = 44,800 aligned Assistant hidden states + V7.0 Student full-262K argmax & top-64 probs!
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
    student_argmax_ids = torch.tensor(select_first_valid_trie_ids(top_ids_t, trie), dtype=torch.long)
    del H_stud_tgt, ast_h_final

    act_bundle = Path("/tmp/v7_bhdr_activations.pt")
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
    bh_metrics = {**bh_rpt["metrics"], "heldout_beam_top1_acc_vs_v7_argmax": round(beam_hits / 256.0, 4), "heldout_beam_samples": 256}
    print(f"  [BHDR-384 DONE] bytes={bh_rpt['bytes']:,} kl={bh_metrics['val_kl_before']}->{bh_metrics['val_kl_after']} beam_top1={bh_metrics['heldout_beam_top1_acc_vs_v7_argmax']}", flush=True)
    del H_ast_all, student_argmax_ids, top_ids_t, top_probs_t

    # 6. Stage 5: Strictly Held-Out COCO val[0..3] Visual QA Parity
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

    artifacts: Dict[str, dict] = {}
    for fp in [OUT_T16C, OUT_CRTR, OUT_C8D, OUT_BHDR, OUT_AST_Q2, OUT_AST_Q4, OUT_AST_Q8]:
        sz_f, dig_f = persist_verified(fp)
        artifacts[fp.name] = {"bytes": sz_f, "sha256": dig_f}

    crtr_vals = [r["crtr"] for r in layer_cos_log]
    report = {
        "version": "V7.0-FULLRANK-HYBRID-CRTR-MODQ20-BHDR384",
        "gpu": gpu_name,
        "elapsed_s": round(time.perf_counter() - t0_all, 2),
        "unified_t16c_bytes": t16c_sz,
        "unified_t16c_gib": round(t16c_sz / (1024**3), 4),
        "crtr_mean_val_recall": round(float(np.mean([c["val_recall_after"] for c in crtr_vals])), 5),
        "crtr_mean_mod_val_acc": round(float(np.mean([c["mod_val_acc"] for c in crtr_vals])), 5),
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
    persist_verified(OUT_REPORT)
    print(f"[V7.0 COMPLETE] Elapsed={(time.perf_counter()-t0_all)/60:.1f} min | Report={OUT_REPORT}", flush=True)

    if os.environ.get("COLAB_AUTO_UNASSIGN", "1") == "1":
        try:
            from google.colab import runtime
            print("[Auto-Unassign] Releasing Colab runtime...", flush=True)
            runtime.unassign()
        except Exception as exc:
            print(f"[Auto-Unassign Note] {exc}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemma 4 31B V7.0 Full-Rank Hybrid + CRTR/MoD + Assistant-V7 Calibrator")
    ap.add_argument("--selftest", action="store_true", help="Run 6-gate CPU self-verification battery and exit.")
    ap.add_argument("--run", action="store_true", help="Run full GPU calibration, CRTR/MoD, and BHDR-384 build.")
    args = ap.parse_args()
    if args.selftest or (not args.run and not torch.cuda.is_available()):
        return run_selftest()
    run_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

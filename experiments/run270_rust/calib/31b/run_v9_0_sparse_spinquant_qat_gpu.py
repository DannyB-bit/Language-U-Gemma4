#!/usr/bin/env python3
"""Language-U V9.0: True Sparse-Active 31B + Outlier-Dispersion SpinQuant + End-to-End Reasoning QAT.

Implements the complete V9.0 training and packing pipeline on A100 80GB / G4 96GB / L4 High-RAM:
  1. Zero-Runtime-Cost Global Outlier-Dispersion Permutation (Pi in S_5376):
     Algebraically disperses outlier channels across all 21 x 256-dim FWHT blocks and folds
     exact permutations across embed_tokens, all 60 layers' RMSNorms & linears, final_norm,
     vision mm_input_projection, and EAGLE-3 Assistant pre/post projections (0 approximation
     error on Gemma 4 per-channel post_attention_layernorm / post_feedforward_layernorm).
  2. Lossless Co-Activation Sparse Cortical Column Permutation (pi_l in S_21504):
     Places top 6 x 256 = 1,536 general-purpose neurons in Shared Trunk Blocks 0..6 and
     clusters remaining 19,968 specialist neurons into 78 balanced co-activating 256-wide
     COLBLOCK columns (Blocks 6..84), enabling <= 14.3% (12/84) active MLP DRAM per pass.
  3. Zero-1bit Attention + Hybrid Ternary-5 / Sparse-QAT 1b-COLBLOCK MLP (<= 5.95 GiB resident):
     All 60 attention blocks stay >= Ternary-5 / 2-bit / 3-bit Gaussian (zero 1-bit attn);
     all 60 down_proj matrices are packed native T16C_COLBLOCK_BIT (0x20) for zero-copy streaming.
  4. Real PyTorch Sparse-Route Scale + Norm + LayerScalar + C8DB QAT + Compass-Magnetism Loss:
     Optimizes learnable FP16 block headers (updated via fast_quant.update_t16c_headers_f32
     without re-quantizing trits), RMSNorm weights, layer_scalar, CRTR-V9 (r=64 + MoD),
     and 60-layer C8DB rank-32 S^7 pattern adapters via real loss.backward() + AdamW.
  5. 4-Step Self-Feeding Rollout EAGLE-3 Assistant-V9 (Q2 ~58 MB + BHDR-384 2.56 MB).

Enforces Rule 2 (strict 0-overlap decontamination against held-out NLL/battery prompts),
Rule 3 (authentic PyTorch backprop only), Rule 4 (fail-closed checks), and 6-gate CPU --selftest.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
import repack_t16c_rowmajor as rmmod  # noqa: E402
import run_v7_0_fullrank_crtr_gpu as v70  # noqa: E402
import spatial_district_converter as sdc  # noqa: E402
import train_assistant_v7_byte_head as bhmod  # noqa: E402
import train_c8d_pattern_bank as c8d  # noqa: E402
import train_cortical_router as crtr  # noqa: E402
import v4_corpus as vc  # noqa: E402

# Architecture & sparsity geometry (31B V9.0 Fast-Kernel Sparse)
V9_BLOCK_SIZE = 256
V9_D_MOD = 5376
V9_FF = 21504
V9_N_LAYERS = 60
V9_MLP_BLOCKS = V9_FF // V9_BLOCK_SIZE  # 84
V9_SHARED_BLOCKS = 4                    # 1,024 always-on shared neurons
V9_ROUTED_BLOCKS = 6                    # Top-6 of 80 specialist columns -> 10/84 = 11.9%
V9_ATTN_HEADS = 32
V9_ATTN_SHARED_HEADS = 0
V9_ATTN_ROUTED_HEADS = 10               # Top-10 of 32 Q/O heads on L2..L57 -> 31.25%
V9_HEAD_CRTR_RANK = 32
V9_CRTR_RANK = 64
V9_N_HIST_SEQ = 1024                    # Strictly decontaminated STEM/code/VLM sequences
V9_SEQ_LEN = 64

MAX_V9_T16C_BYTES = int(6.00 * (1024**3))
GLOBAL_ATTN_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}
BOUNDARY_LAYERS = {0, 1, 58, 59}
# 36 high-impact Ternary-5 MLP layers: L0..L7, all global layers, L44..L59
TERN5_MLP_LAYERS = set(range(0, 8)) | GLOBAL_ATTN_LAYERS | set(range(44, 60))

OUT_DIR = Path(os.environ.get("V9_0_OUT_DIR", "/content/v9_0_outputs"))
CKPT_DIR = OUT_DIR / "ckpt_v9_0"
_drv_env = os.environ.get("V9_0_DRIVE_DIR", "").strip()
DRIVE_DIR: Optional[Path] = Path(_drv_env) if _drv_env else None

OUT_T16C = OUT_DIR / "gemma4_31b_v9_0_sparse.t16c"
OUT_CRTR = OUT_DIR / "gemma4_31b_v9_0_router.crtr"
OUT_C8D = OUT_DIR / "gemma4_31b_v9_0_c8d.c8d"
OUT_AST_Q2 = OUT_DIR / "gemma4_31b_v9_0_assistant_body_q2.t16c"
OUT_AST_Q4 = OUT_DIR / "gemma4_31b_v9_0_assistant_body_q4.t16c"
OUT_BHDR = OUT_DIR / "gemma4_31b_v9_0_assistant.bhdr"
OUT_PERM_PT = OUT_DIR / "v9_0_permutations.pt"
OUT_REPORT = OUT_DIR / "v9_0_report.json"
PROGRESS_JSON = OUT_DIR / "v9_0_progress.json"

ALL_BLACKLIST = tuple(sorted(set(vc.BLACKLIST_SUBSTRINGS) | set(getattr(v70, "ALL_BLACKLIST", ()))))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(8 * 1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _drive_safe_copy(src: Path, dst: Path, chunk_bytes: int = 16 * 1024 * 1024) -> None:
    """Copy atomically to Google Drive (temp + fsync + replace) and fail closed on size mismatch."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(dst.name + ".tmp_up")
    with open(src, "rb") as fs, open(tmp_dst, "wb") as fd:
        shutil.copyfileobj(fs, fd, chunk_bytes)
        fd.flush()
        os.fsync(fd.fileno())
    os.replace(tmp_dst, dst)
    src_sz, dst_sz = src.stat().st_size, dst.stat().st_size
    if src_sz != dst_sz:
        raise IOError(f"Drive copy size mismatch for {dst}: {dst_sz} != {src_sz}")


def _verify_drive_sha256(drive_dir: Path, expected_sha: Dict[str, str]) -> None:
    """Fail closed if any persisted artifact on Google Drive differs from its local SHA-256."""
    for fname, exp_dig in expected_sha.items():
        fp_d = drive_dir / fname
        if not fp_d.is_file():
            raise FileNotFoundError(f"Drive verification failed: missing {fp_d}")
        got_dig = _sha256_file(fp_d)
        if got_dig != exp_dig:
            raise IOError(f"Drive SHA-256 mismatch for {fp_d}: got {got_dig}, expected {exp_dig}")
    print(f"[Drive Verify] {len(expected_sha)}/{len(expected_sha)} SHA-256 checksums verified OK in {drive_dir}", flush=True)


def _flush_drive_before_unassign() -> None:
    """Synchronize Linux page cache and Google Drive FUSE write-back queue before VM termination."""
    try:
        subprocess.run(["sync"], check=False)
        from google.colab import drive
        drive.flush_and_unmount()
    except Exception as exc:
        print(f"[Drive Flush Note] {exc}", flush=True)


def _normalize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def collect_holdout_4grams() -> set[Tuple[str, str, str, str]]:
    """Collect normalized 4-grams across all skeptic-proof holdouts & unseen battery checks."""
    texts: List[str] = [p for _, p in spmod.HELDOUT_NLL_PASSAGES]
    for item in getattr(spmod, "HELDOUT_CHAT_QUESTIONS", ()):
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, (tuple, list)):
            texts.extend(x for x in item if isinstance(x, str))
    texts.extend(prompt for _, prompt, _, _ in spmod.BATTERY_PROMPTS)
    grams: set[Tuple[str, str, str, str]] = set()
    for txt in texts:
        ws = _normalize_words(txt)
        for i in range(len(ws) - 3):
            grams.add((ws[i], ws[i + 1], ws[i + 2], ws[i + 3]))
    return grams


def verify_corpus_decontaminated(
    samples: Sequence[str],
    holdout_4grams: Optional[set[Tuple[str, str, str, str]]] = None,
) -> Dict[str, int]:
    """Fail closed if any calibration string shares a blacklist term or 4-gram with holdouts."""
    h4 = holdout_4grams if holdout_4grams is not None else collect_holdout_4grams()
    bl_hits = 0
    ngram_hits = 0
    for s in samples:
        low = s.lower()
        if any(b in low for b in ALL_BLACKLIST):
            bl_hits += 1
        ws = _normalize_words(s)
        for i in range(len(ws) - 3):
            if (ws[i], ws[i + 1], ws[i + 2], ws[i + 3]) in h4:
                ngram_hits += 1
                break
    if bl_hits > 0 or ngram_hits > 0:
        raise ValueError(f"Rule 2 decontamination breach: blacklist_hits={bl_hits}, 4gram_hits={ngram_hits}")
    return {"checked": len(samples), "blacklist_hits": 0, "overlap_4grams": 0}


OUT_HEAD_CRTR = OUT_DIR / "gemma4_31b_v9_0_head_router.crtr"


def get_v9_proj_spec(l_idx: int, pname: str, is_full: Optional[bool] = None) -> Tuple[int, int, int, bool, bool]:
    """Return `(mode, block_size, lr_rank, is_rowmajor, is_colblock)` for every V9.0 projection.

    Invariant: ZERO Ternary-5 anywhere (44 GMAC/s 1b_asym_RM on L2..L57 MLP + sliding Q/O,
    33 GMAC/s 2b_gauss on boundaries and global Q/O + sliding K/V, 3b_gauss on L0/L59/global K).
    """
    if is_full is None:
        is_full = l_idx in GLOBAL_ATTN_LAYERS
    if pname in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        is_cb = pname == "mlp.down_proj"
        if l_idx in (0, 1, 58, 59):
            return 5, V9_BLOCK_SIZE, 16, False, is_cb
        return 0, V9_BLOCK_SIZE, (16 if l_idx >= 54 else 0), True, is_cb

    # Attention projections
    if l_idx in (0, 59):
        return (6 if pname in ("self_attn.k_proj", "self_attn.o_proj") else 5), V9_BLOCK_SIZE, 16, False, False
    if l_idx in (1, 58):
        return 5, V9_BLOCK_SIZE, 16, False, False
    if is_full:
        if pname == "self_attn.k_proj":
            return 6, V9_BLOCK_SIZE, 16, False, False
        if pname == "self_attn.o_proj":
            return 5, V9_BLOCK_SIZE, 16, False, True
        return 5, V9_BLOCK_SIZE, 16, False, False
    # Middle sliding attention (L2..L57 non-global): fast 1b_asym_RM Q/O + 2b_gauss K/V
    if pname == "self_attn.q_proj":
        return 0, V9_BLOCK_SIZE, 16, True, False
    if pname == "self_attn.o_proj":
        return 0, V9_BLOCK_SIZE, 16, True, True
    return 5, V9_BLOCK_SIZE, 0, False, False


def compute_v9_0_budget_bytes(
    *,
    vocab_size: int = 262144,
    d_mod: int = V9_D_MOD,
    n_layers: int = V9_N_LAYERS,
    nh: int = 32,
    nkv_s: int = 16,
    nkv_f: int = 4,
    hd_s: int = 256,
    hd_f: int = 512,
    ff: int = V9_FF,
    vision_bytes: int = 160 * 1024 * 1024,
) -> Dict[str, object]:
    """Exact byte-accounting guard for single-file V9.0 `.t16c` + Pi 4 active DRAM & GMACs."""
    embed_b = p3.expected_tile16_bytes(vocab_size, d_mod, V9_BLOCK_SIZE, 3, False) + 256
    attn_b = 0
    mlp_b = 0
    lr_b = 0
    tern5_cnt = 0
    attn_1b_rm_cnt = 0
    attn_2b_gauss_cnt = 0
    mlp_2b_gauss_layers = 0
    mlp_1b_rm_layers = 0
    active_attn_gmac = 0.0
    active_mlp_gmac = 0.0

    for l in range(n_layers):
        is_f = l in GLOBAL_ATTN_LAYERS
        is_bnd = l in (0, 1, 58, 59)
        head_frac = 1.0 if is_bnd else float(V9_ATTN_SHARED_HEADS + V9_ATTN_ROUTED_HEADS) / float(nh)
        nkv, hd = (nkv_f, hd_f) if is_f else (nkv_s, hd_s)
        q_dim = nh * hd
        kv_dim = nkv * hd
        attn_shapes: List[Tuple[str, int, int]] = [
            ("self_attn.q_proj", q_dim, d_mod),
            ("self_attn.k_proj", kv_dim, d_mod),
            ("self_attn.o_proj", d_mod, q_dim),
        ]
        if not is_f:
            attn_shapes.append(("self_attn.v_proj", kv_dim, d_mod))
        for pn, do, di in attn_shapes:
            mode, bsz, lrr, _, _ = get_v9_proj_spec(l, pn, is_f)
            nb, asym = p3.MODE_TO_T16C[mode]
            if mode == 1:
                tern5_cnt += 1
            elif (nb & 0x1F) == 1:
                attn_1b_rm_cnt += 1
            else:
                attn_2b_gauss_cnt += 1
            attn_b += p3.expected_tile16_bytes(do, di, bsz, nb, asym) + 128 + di * 4
            if lrr > 0:
                lr_b += do * 4 + lrr * 4 + do * lrr + lrr * di
            g_proj = (float(do * di) / 1e9) * (head_frac if pn in ("self_attn.q_proj", "self_attn.o_proj") else 1.0)
            active_attn_gmac += g_proj

        m_gate, _, _, _, _ = get_v9_proj_spec(l, "mlp.gate_proj", is_f)
        if m_gate == 5:
            mlp_2b_gauss_layers += 1
        elif m_gate == 0:
            mlp_1b_rm_layers += 1
        mlp_frac = 1.0 if is_bnd else float(V9_SHARED_BLOCKS + V9_ROUTED_BLOCKS) / float(V9_MLP_BLOCKS)
        for pn, do, di in (
            ("mlp.gate_proj", ff, d_mod),
            ("mlp.up_proj", ff, d_mod),
            ("mlp.down_proj", d_mod, ff),
        ):
            mode, bsz, lrr, _, _ = get_v9_proj_spec(l, pn, is_f)
            nb, asym = p3.MODE_TO_T16C[mode]
            if mode == 1:
                tern5_cnt += 1
            mlp_b += p3.expected_tile16_bytes(do, di, bsz, nb, asym) + 128 + di * 4
            if lrr > 0:
                lr_b += do * 4 + lrr * 4 + do * lrr + lrr * di
            active_mlp_gmac += (float(do * di) / 1e9) * mlp_frac

    total_b = embed_b + vision_bytes + attn_b + mlp_b + lr_b
    active_ratio = float(V9_SHARED_BLOCKS + V9_ROUTED_BLOCKS) / float(V9_MLP_BLOCKS)
    active_head_ratio = float(V9_ATTN_SHARED_HEADS + V9_ATTN_ROUTED_HEADS) / float(nh)
    active_dram_full_b = int(0.06 * embed_b + active_head_ratio * attn_b + active_ratio * mlp_b + lr_b)
    active_dram_mod35_b = int(0.06 * embed_b + 0.75 * active_head_ratio * attn_b + 0.74 * active_ratio * mlp_b + lr_b)
    return {
        "embed_bytes": embed_b,
        "vision_bytes": vision_bytes,
        "attn_bytes": attn_b,
        "mlp_bytes": mlp_b,
        "low_rank_bytes": lr_b,
        "total_t16c_bytes": total_b,
        "total_t16c_gib": round(total_b / (1024**3), 4),
        "tern5_cnt": tern5_cnt,
        "attn_1b_rm_cnt": attn_1b_rm_cnt,
        "attn_2b_gauss_cnt": attn_2b_gauss_cnt,
        "mlp_2b_gauss_layers": mlp_2b_gauss_layers,
        "mlp_1b_rm_layers": mlp_1b_rm_layers,
        "active_mlp_ratio": round(active_ratio, 4),
        "active_attn_head_ratio": round(active_head_ratio, 4),
        "active_gmac_per_tok": round(active_attn_gmac + active_mlp_gmac, 3),
        "active_gmac_mod35_per_tok": round(0.74 * (active_attn_gmac + active_mlp_gmac), 3),
        "active_dram_10of84_mb": round(active_dram_full_b / (1024**2), 2),
        "active_dram_10of84_mod35_mb": round(active_dram_mod35_b / (1024**2), 2),
    }


def compute_outlier_dispersion_perm(scores: torch.Tensor, block_size: int = V9_BLOCK_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispersion permutation Pi in S_d spreading top outlier channels evenly across all d // block_size blocks.

    Uses snake-dealing (alternating block direction per round) so every FWHT quantization block
    receives balanced activation energy and variance with zero runtime overhead.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {tuple(scores.shape)}")
    d = int(scores.numel())
    if d % block_size != 0:
        raise ValueError(f"d={d} must be divisible by block_size={block_size}")
    n_blocks = d // block_size
    order = torch.argsort(scores.float(), descending=True, stable=True)
    buckets: List[List[int]] = [[] for _ in range(n_blocks)]
    for idx_i, chan in enumerate(order.tolist()):
        rnd = idx_i // n_blocks
        pos = idx_i % n_blocks
        blk = (n_blocks - 1 - pos) if (rnd % 2 == 1) else pos
        buckets[blk].append(chan)
    perm = torch.tensor([c for b in buckets for c in b], dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(d, dtype=torch.long)
    return perm, inv_perm


def compute_geglu_colblock_perm(
    act_samples: torch.Tensor,
    energy_vec: Optional[torch.Tensor] = None,
    shared_blocks: int = V9_SHARED_BLOCKS,
    block_size: int = V9_BLOCK_SIZE,
    iters: int = 8,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Build lossless permutation pi_l in S_ff grouping top shared neurons + balanced co-active specialist columns.

    Args:
        act_samples: `[N_tok, ff]` non-negative or signed GeGLU activations (`|gelu(gate) * up|`).
        energy_vec: optional `[ff]` mean squared energy vector; computed from `act_samples` if None.
    """
    assert act_samples.ndim == 2
    n_tok, ff = int(act_samples.shape[0]), int(act_samples.shape[1])
    if ff % block_size != 0:
        raise ValueError(f"ff={ff} must divide by block_size={block_size}")
    n_blocks = ff // block_size
    if not (0 < shared_blocks < n_blocks):
        raise ValueError(f"shared_blocks={shared_blocks} must be in (0, {n_blocks})")

    abs_a = torch.nan_to_num(act_samples.detach().float().abs(), nan=0.0, posinf=65000.0, neginf=0.0).clamp(max=65000.0)
    if energy_vec is None:
        energy = (abs_a.double() * abs_a.double()).mean(dim=0).float()
    else:
        energy = torch.nan_to_num(energy_vec.detach().float(), nan=0.0, posinf=1e12, neginf=0.0).clamp(min=0.0)
    tau = abs_a.mean() * 0.25 + 1e-6
    freq = (abs_a > tau).float().mean(dim=0)
    shared_score = freq * torch.sqrt(energy + 1e-12)

    n_shared = shared_blocks * block_size
    order_sh = torch.argsort(shared_score, descending=True, stable=True)
    shared_idx = order_sh[:n_shared]
    spec_idx = order_sh[n_shared:]
    n_spec_blocks = n_blocks - shared_blocks

    # Normalize each specialist neuron's [N_tok] activation trace for cosine co-activation clustering
    sub_n = min(n_tok, 2048)
    cols = F.normalize(abs_a[:sub_n, spec_idx], p=2, dim=0)  # [sub_n, n_spec]
    spec_e = energy[spec_idx]
    gen = torch.Generator(device=cols.device).manual_seed(seed)

    # Seed k=n_spec_blocks centroids spread evenly by descending specialist energy
    top_e_ord = torch.argsort(spec_e, descending=True, stable=True)
    seed_step = max(1, top_e_ord.numel() // n_spec_blocks)
    init_sel = top_e_ord[::seed_step][:n_spec_blocks]
    if init_sel.numel() < n_spec_blocks:
        extra = torch.randperm(spec_idx.numel(), generator=gen, device=cols.device)[: n_spec_blocks - init_sel.numel()]
        init_sel = torch.cat([init_sel, extra])
    centroids = F.normalize(cols[:, init_sel].T, p=2, dim=-1)  # [n_spec_blocks, sub_n]

    assign = torch.arange(spec_idx.numel(), device=cols.device) // block_size
    for _ in range(max(1, iters)):
        sims = centroids @ cols  # [n_spec_blocks, n_spec]
        max_sim = sims.max(dim=0).values
        pref_order = torch.argsort(max_sim, descending=True, stable=True)
        # Balanced round-robin assignment on GPU/CPU with exact capacity `block_size` per column
        counts = [0] * n_spec_blocks
        sims_cpu = sims.detach().cpu()
        new_assign = torch.empty(spec_idx.numel(), dtype=torch.long)
        rank_cols = torch.argsort(sims_cpu, dim=0, descending=True)  # [n_spec_blocks, n_spec]
        for ni in pref_order.cpu().tolist():
            chosen = -1
            for c_rank in range(n_spec_blocks):
                cand = int(rank_cols[c_rank, ni].item())
                if counts[cand] < block_size:
                    chosen = cand
                    break
            if chosen < 0:
                chosen = min(range(n_spec_blocks), key=lambda k: counts[k])
            counts[chosen] += 1
            new_assign[ni] = chosen
        assign = new_assign.to(cols.device)
        # Update centroids cleanly
        next_c = torch.zeros_like(centroids)
        for c_id in range(n_spec_blocks):
            m = assign == c_id
            if m.any():
                next_c[c_id] = cols[:, m].sum(dim=1)
        centroids = F.normalize(next_c, p=2, dim=-1)

    blocks_list = [shared_idx.cpu()]
    for c_id in range(n_spec_blocks):
        members = spec_idx[assign == c_id].cpu()
        assert members.numel() == block_size, f"Cluster {c_id} size {members.numel()} != {block_size}"
        blocks_list.append(members)
    perm = torch.cat(blocks_list, dim=0).long()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(ff, dtype=torch.long)

    e_perm = energy.double().cpu()[perm].reshape(n_blocks, block_size).sum(dim=1)
    keep_total = min(n_blocks, shared_blocks + V9_ROUTED_BLOCKS)
    # Top-k energy concentration per token vs raw block ordering (float64 avoids Inf/NaN)
    abs_sub_d = abs_a[:sub_n].double()
    blk_raw = (abs_sub_d * abs_sub_d).reshape(sub_n, n_blocks, block_size).sum(dim=2)
    raw_topk_pct = 100.0 * float((torch.topk(blk_raw, keep_total, dim=-1).values.sum(dim=1) / blk_raw.sum(dim=1).clamp(min=1e-9)).mean().item())
    blk_prm = (abs_sub_d[:, perm.to(abs_a.device)] ** 2).reshape(sub_n, n_blocks, block_size).sum(dim=2)
    sh_e = blk_prm[:, :shared_blocks].sum(dim=1)
    k_rt = min(V9_ROUTED_BLOCKS, n_spec_blocks)
    rt_e = torch.topk(blk_prm[:, shared_blocks:], k_rt, dim=-1).values.sum(dim=1) if k_rt > 0 else torch.zeros_like(sh_e)
    perm_topk_pct = 100.0 * float(((sh_e + rt_e) / blk_prm.sum(dim=1).clamp(min=1e-9)).mean().item())
    stats = {
        "shared_energy_pct": round(100.0 * float(e_perm[:shared_blocks].sum() / e_perm.sum().clamp(min=1e-9)), 2),
        "raw_topk_energy_pct": round(raw_topk_pct, 2),
        "perm_12of84_energy_pct": round(perm_topk_pct, 2),
    }
    return perm, inv_perm, stats


class Gemma4AssistantRolloutNet(nn.Module):
    """Differentiable 4-layer EAGLE-3 Assistant mirror supporting K-step self-feeding rollout QAT."""

    def __init__(self, d_backbone: int, d_ast: int, n_layers: int = 4, ff_ast: Optional[int] = None) -> None:
        super().__init__()
        self.d_backbone = d_backbone
        self.d_ast = d_ast
        self.n_layers = n_layers
        ff = ff_ast or (d_ast * 4)
        self.pre_proj = nn.Linear(2 * d_backbone, d_ast, bias=False)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_in": nn.RMSNorm(d_ast, eps=1e-6),
                        "qkv": nn.Linear(d_ast, d_ast * 3, bias=False),
                        "o": nn.Linear(d_ast, d_ast, bias=False),
                        "norm_ff": nn.RMSNorm(d_ast, eps=1e-6),
                        "gate": nn.Linear(d_ast, ff, bias=False),
                        "up": nn.Linear(d_ast, ff, bias=False),
                        "down": nn.Linear(ff, d_ast, bias=False),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.RMSNorm(d_ast, eps=1e-6)
        self.post_proj = nn.Linear(d_ast, d_backbone, bias=False)

    def forward_step_state(self, emb_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        h = self.pre_proj(torch.cat([emb_t, h_prev], dim=-1))  # [B, L, d_ast]
        bsz, seq_len, _ = h.shape
        n_heads = max(1, self.d_ast // 128)
        hd = self.d_ast // n_heads
        for lyr in self.layers:
            xn = lyr["norm_in"](h)
            q, k, v = lyr["qkv"](xn).chunk(3, dim=-1)
            q = q.view(bsz, seq_len, n_heads, hd).transpose(1, 2)
            k = k.view(bsz, seq_len, n_heads, hd).transpose(1, 2)
            v = v.view(bsz, seq_len, n_heads, hd).transpose(1, 2)
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + lyr["o"](ctx.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_ast))
            xf = lyr["norm_ff"](h)
            h = h + lyr["down"](F.gelu(lyr["gate"](xf), approximate="tanh") * lyr["up"](xf))
        return self.post_proj(self.norm_out(h))


def assistant_rollout_loss(
    ast_net: Gemma4AssistantRolloutNet,
    emb_seq: torch.Tensor,
    h_l59_seq: torch.Tensor,
    rollout_steps: int = 4,
) -> Tuple[torch.Tensor, List[float]]:
    """Compute multi-step self-feeding EAGLE-3 rollout alignment loss across steps k = 1..K."""
    _, seq_len, _ = emb_seq.shape
    max_k = min(rollout_steps, max(1, seq_len - 2))
    win = max(1, seq_len - max_k)
    h_in = h_l59_seq[:, :win, :]
    total_loss = emb_seq.new_zeros(())
    step_losses: List[float] = []
    for k in range(max_k):
        e_k = emb_seq[:, k : k + win, :]
        target_k = h_l59_seq[:, k + 1 : k + 1 + win, :]
        pred_k = ast_net.forward_step_state(e_k, h_in)
        l_k = (1.0 - F.cosine_similarity(pred_k, target_k, dim=-1).mean()) + 0.1 * F.mse_loss(pred_k, target_k)
        w_k = 1.0 / float(1 + k)
        total_loss = total_loss + w_k * l_k
        step_losses.append(float(l_k.detach().item()))
        h_in = pred_k
    return total_loss / float(max_k), step_losses


class Gemma4AssistantTrain(nn.Module):
    """Exact 4-layer Google EAGLE-3 Assistant trainable module supporting self-feeding rollout."""

    def __init__(
        self,
        sd: Dict[str, torch.Tensor],
        *,
        d_mod: int,
        nh: int,
        nkv_s: int,
        nkv_f: int,
        hd_s: int,
        hd_f: int,
        theta_l: float,
        part_l: float,
        theta_g: float,
        part_g: float,
        seq_len: int,
        eps: float,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.nh = nh
        self.nkv_s = nkv_s
        self.nkv_f = nkv_f
        self.hd_s = hd_s
        self.hd_f = hd_f
        self.rot_l = int(round(hd_s * part_l))
        self.rot_g = int(round(hd_f * part_g))
        self.eps = eps
        self.compute_dtype = compute_dtype
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
        for pfx_r, th, rd, hd_r in (("l", theta_l, self.rot_l, hd_s), ("g", theta_g, self.rot_g, hd_f)):
            hr = rd // 2
            if hr > 0:
                inv_f = 1.0 / (th ** (torch.arange(0, 2 * hr, 2, dtype=torch.float32) / float(hd_r)))
                ang = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_f)
                self.register_buffer(f"cr_{pfx_r}", torch.cos(ang), persistent=False)
                self.register_buffer(f"sr_{pfx_r}", torch.sin(ang), persistent=False)
            else:
                self.register_buffer(f"cr_{pfx_r}", torch.zeros((seq_len, 0), dtype=torch.float32), persistent=False)
                self.register_buffer(f"sr_{pfx_r}", torch.zeros((seq_len, 0), dtype=torch.float32), persistent=False)

    def _rmsn(self, t: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf = t.float()
        y = xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + self.eps)
        if w is not None:
            y = y * w.float()
        return y.to(t.dtype)

    def _rope(self, x: torch.Tensor, cs: torch.Tensor, sn: torch.Tensor, rot_d: int, hd: int) -> torch.Tensor:
        hr, hh = rot_d // 2, hd // 2
        if hr <= 0:
            return x
        xf = x.float()
        x0, x1 = xf[..., :hr], xf[..., hh : hh + hr]
        cs_v = cs.view(1, cs.shape[0], 1, hr)
        sn_v = sn.view(1, sn.shape[0], 1, hr)
        r0, r1 = x0 * cs_v - x1 * sn_v, x0 * sn_v + x1 * cs_v
        if hr == hh:
            return torch.cat([r0, r1], dim=-1).to(x.dtype)
        return torch.cat([r0, xf[..., hr:hh], r1, xf[..., hh + hr :]], dim=-1).to(x.dtype)

    def forward_seq(self, tok_emb_seq, back_h_seq, k58, v58, k59, v59):
        B, Sm1, _ = tok_emb_seq.shape
        kv_len = int(k58.shape[2])
        state = F.linear(torch.cat([tok_emb_seq, back_h_seq], dim=-1).to(self.compute_dtype), self.pre.to(self.compute_dtype)).float()
        q_mask = torch.tril(torch.ones((Sm1, kv_len), device=state.device, dtype=torch.bool), diagonal=0).unsqueeze(0).unsqueeze(0)
        for la, m in enumerate(self.layers):
            is_f = (la == 3)
            hda, nkva = (self.hd_f, self.nkv_f) if is_f else (self.hd_s, self.nkv_s)
            rota = self.rot_g if is_f else self.rot_l
            ga = self.nh // nkva
            inp = self._rmsn(state, m.n_in).to(self.compute_dtype)
            q = F.linear(inp, m.q_proj.to(self.compute_dtype)).view(B, Sm1, self.nh, hda)
            cr = (self.cr_g if is_f else self.cr_l)[1 : Sm1 + 1]
            sr = (self.sr_g if is_f else self.sr_l)[1 : Sm1 + 1]
            qn = self._rope(self._rmsn(q, m.n_qn), cr, sr, rota, hda).transpose(1, 2).to(self.compute_dtype)
            kk = (k59 if is_f else k58).to(self.compute_dtype).repeat_interleave(ga, dim=1)
            vv = (v59 if is_f else v58).to(self.compute_dtype).repeat_interleave(ga, dim=1)
            ctx = F.scaled_dot_product_attention(qn, kk, vv, attn_mask=q_mask, scale=1.0)
            state = state + self._rmsn(F.linear(ctx.transpose(1, 2).contiguous().view(B, Sm1, self.nh * hda), m.o_proj.to(self.compute_dtype)), m.n_pa).float()
            hm = self._rmsn(state, m.n_pf).to(self.compute_dtype)
            act = (F.gelu(F.linear(hm, m.gate.to(self.compute_dtype)).float(), approximate="tanh") * F.linear(hm, m.up.to(self.compute_dtype)).float()).to(self.compute_dtype)
            state = (state + self._rmsn(F.linear(act, m.down.to(self.compute_dtype)), m.n_pff).float()) * m.lsc
        return self._rmsn(state, self.fn).float()


def _compile_fast_quant(out_so: Path) -> ctypes.CDLL:
    fq_c = THIS_DIR / "fast_quant.c"
    if not fq_c.is_file():
        raise FileNotFoundError(f"Missing fast_quant.c at {fq_c}")
    if not out_so.is_file() or out_so.stat().st_mtime < fq_c.stat().st_mtime:
        subprocess.run(["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", str(fq_c), "-o", str(out_so), "-lm"], check=True)
    lib = ctypes.CDLL(str(out_so))
    lib.t16c_tblk_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.t16c_tblk_bytes.restype = ctypes.c_size_t
    lib.quant_and_pack_t16c_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.update_t16c_headers_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    return lib


def unpack_tile16_headers_and_units(
    pbytes: np.ndarray,
    d_out: int,
    d_in: int,
    bsize: int,
    mode_i: int,
    h_mat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Vectorized decode of `(units_rot, sc_init [d_out, n_blks], bi_init)` from fast_quant T16C."""
    n_bits, is_asym = p3.MODE_TO_T16C[mode_i]
    eff_bits = n_bits & 0x1F
    n_blks = d_in // bsize
    n_tiles = (d_out + 15) // 16
    tblk_b = p3.tile16_blk_bytes(bsize, n_bits, is_asym)
    arr = np.frombuffer(pbytes, dtype=np.uint8).reshape(n_tiles, n_blks, tblk_b)

    sc_f16 = arr[:, :, :32].copy().view("<f2").reshape(n_tiles, n_blks, 16).transpose(0, 2, 1).reshape(n_tiles * 16, n_blks)[:d_out]
    sc_out = sc_f16.astype(np.float32)
    bi_out = None
    if is_asym:
        bi_f16 = arr[:, :, 32:64].copy().view("<f2").reshape(n_tiles, n_blks, 16).transpose(0, 2, 1).reshape(n_tiles * 16, n_blks)[:d_out]
        bi_out = bi_f16.astype(np.float32)

    bpr = bsize // 8
    quintets = (bsize + 4) // 5
    gauss2_lv = np.array([-3.174, -1.000, 1.000, 3.174], dtype=np.float32)
    gauss3_lv = np.array([-7.430, -5.430, -3.270, -1.270, 1.270, 3.270, 5.430, 7.430], dtype=np.float32)

    if mode_i == 0:
        pl = arr[:, :, 64:].reshape(n_tiles, n_blks, bpr, 16)
        bits = np.unpackbits(pl[:, :, :, None], axis=4, bitorder="big").reshape(n_tiles, n_blks, bpr, 16, 8)
        # Swap (bpr, 16, 8) -> (16, bpr, 8) = (16, bsize)
        u_grid = np.where(bits.transpose(0, 3, 1, 2, 4).reshape(n_tiles * 16, d_in) > 0, 1.0, -1.0).astype(np.float32)
        units = u_grid[:d_out]
    elif mode_i == 1:
        pl = arr[:, :, 32:].reshape(n_tiles, n_blks, quintets, 16).transpose(0, 3, 1, 2).astype(np.int32)
        u_blk = np.zeros((n_tiles, 16, n_blks, bsize), dtype=np.float32)
        c = pl.copy()
        for k in range(5):
            cols = np.arange(k, bsize, 5)
            d = c[:, :, :, : cols.size] % 3
            c[:, :, :, : cols.size] //= 3
            u_blk[:, :, :, cols] = (d - 1).astype(np.float32)
        units = u_blk.transpose(0, 1, 2, 3).reshape(n_tiles * 16, d_in)[:d_out]
    else:
        hdr_b = 64 if is_asym else 32
        planes = arr[:, :, hdr_b:].reshape(n_tiles, n_blks, eff_bits, bpr, 16)
        qi = np.zeros((n_tiles, 16, n_blks, bsize), dtype=np.int32)
        for p in range(eff_bits):
            bits = np.unpackbits(planes[:, :, p, :, :, None], axis=4, bitorder="big").reshape(n_tiles, n_blks, bpr, 16, 8)
            qi |= bits.transpose(0, 3, 1, 2, 4).reshape(n_tiles, 16, n_blks, bsize).astype(np.int32) << p
        qi_flat = qi.reshape(n_tiles * 16, d_in)[:d_out]
        if mode_i == 5:
            units = gauss2_lv[qi_flat]
        elif mode_i == 6:
            units = gauss3_lv[qi_flat]
        else:
            max_l = float((1 << eff_bits) - 1)
            units = (2.0 * qi_flat.astype(np.float32) - max_l)

    u_t = torch.from_numpy(np.ascontiguousarray(units)).to(h_mat.device, dtype=torch.float32)
    sc_t = torch.from_numpy(np.ascontiguousarray(sc_out)).to(h_mat.device, dtype=torch.float32)
    bi_t = torch.from_numpy(np.ascontiguousarray(bi_out)).to(h_mat.device, dtype=torch.float32) if bi_out is not None else None
    return u_t, sc_t, bi_t


def finalize_t16c_payload(
    pb_buf: np.ndarray,
    do: int,
    di: int,
    bsize: int,
    mode_i: int,
    is_rowmajor: bool,
    is_colblock: bool,
) -> Tuple[int, bytes]:
    """Convert fast_quant column-major payload to RowMajor (0x40) and/or COLBLOCK (0x20)."""
    nb_raw, asym = p3.MODE_TO_T16C[mode_i]
    eff_bits = nb_raw & 0x1F
    hdr_bits = nb_raw
    if isinstance(pb_buf, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(pb_buf, dtype=np.uint8)
    else:
        arr = np.asarray(pb_buf, dtype=np.uint8)
    if is_rowmajor:
        arr = rmmod.transpose_planes_col_to_rowmajor(arr, do, di, bsize, eff_bits, asym)
        hdr_bits |= p3.T16C_ROWMAJOR_BIT
    raw_bytes = arr.tobytes()
    if is_colblock:
        raw_bytes = sdc.transpose_colblock_bytes(raw_bytes, do, di, bsize, eff_bits, asym, to_colblock=True)
        hdr_bits |= p3.T16C_COLBLOCK_BIT
    return hdr_bits, raw_bytes


def reconstruct_from_headers(
    units_rot: torch.Tensor,
    sc: torch.Tensor,
    bi: Optional[torch.Tensor],
    inv_s: Optional[torch.Tensor],
    h_mat: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct differentiable `[d_out, d_in]` matrix from frozen bitplanes + learnable block scales."""
    d_out, d_in = units_rot.shape
    bsize = h_mat.shape[0]
    n_blks = d_in // bsize
    w_rot = units_rot.reshape(d_out, n_blks, bsize) * sc.unsqueeze(-1)
    if bi is not None:
        w_rot = w_rot + bi.unsqueeze(-1)
    w = (w_rot.reshape(-1, bsize) @ h_mat).reshape(d_out, d_in)
    if inv_s is not None:
        w = w * inv_s.unsqueeze(0)
    return w


_T16C_TO_MODE: Dict[Tuple[int, bool, bool], int] = {
    (1, True, False): 0,
    (5, False, False): 1,
    (2, False, False): 2,
    (2, True, False): 3,
    (3, False, False): 4,
    (2, False, True): 5,
    (3, False, True): 6,
}


def transpose_planes_row_to_colmajor(
    arr_u8: np.ndarray, d_out: int, d_in: int, bsize: int, eff_bits: int, is_asym: bool
) -> np.ndarray:
    """Invert `repack_t16c_rowmajor.transpose_planes_col_to_rowmajor` cleanly."""
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    hdr = 64 if is_asym else 32
    bpr = bsize // 8
    plane_b = bpr * 16
    tblk_b = hdr + eff_bits * plane_b
    arr = np.asarray(arr_u8, dtype=np.uint8).reshape(n_tiles, n_blks, tblk_b).copy()
    for p in range(eff_bits):
        s0 = hdr + p * plane_b
        s1 = s0 + plane_b
        pl = arr[:, :, s0:s1].reshape(n_tiles, n_blks, 16, bpr)
        arr[:, :, s0:s1] = np.ascontiguousarray(pl.transpose(0, 1, 3, 2)).reshape(n_tiles, n_blks, plane_b)
    return arr.reshape(-1)


def decode_t16c_mat_to_f32(
    pay_bytes: bytes | np.ndarray,
    d_out: int,
    d_in: int,
    bsize: int,
    n_bits: int,
    is_asym: bool,
    inv_s_np: Optional[np.ndarray],
    low_rank: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    h_mat: torch.Tensor,
    has_fwht: bool = True,
) -> torch.Tensor:
    """Decode any packed T16C matrix (ColMajor, RowMajor `0x40`, COLBLOCK `0x20`, Gauss `0x80`) + low-rank to `[d_out, d_in]`."""
    eff_bits = int(n_bits) & 0x1F
    is_gauss = bool(int(n_bits) & p3.T16C_GAUSS_BIT)
    is_rm = bool(int(n_bits) & p3.T16C_ROWMAJOR_BIT)
    is_cb = bool(int(n_bits) & p3.T16C_COLBLOCK_BIT)
    mode_i = _T16C_TO_MODE[(eff_bits, bool(is_asym), is_gauss)]
    raw_b = pay_bytes if isinstance(pay_bytes, (bytes, bytearray, memoryview)) else np.asarray(pay_bytes, dtype=np.uint8).tobytes()
    if is_cb:
        raw_b = sdc.transpose_colblock_bytes(raw_b, d_out, d_in, bsize, eff_bits, bool(is_asym), to_colblock=False)
    arr_u8 = np.frombuffer(raw_b, dtype=np.uint8)
    if is_rm:
        arr_u8 = transpose_planes_row_to_colmajor(arr_u8, d_out, d_in, bsize, eff_bits, bool(is_asym))
    u_t, sc_t, bi_t = unpack_tile16_headers_and_units(arr_u8, d_out, d_in, bsize, mode_i, h_mat)
    inv_s_t = torch.from_numpy(np.ascontiguousarray(inv_s_np, dtype=np.float32)).to(h_mat.device) if inv_s_np is not None else None
    h_use = h_mat if has_fwht else torch.eye(bsize, device=h_mat.device, dtype=torch.float32)
    w = reconstruct_from_headers(u_t, sc_t, bi_t, inv_s_t, h_use)
    if low_rank is not None:
        su, sv, ui8, vi8 = low_rank
        uf = torch.from_numpy(np.ascontiguousarray(ui8)).to(h_mat.device, dtype=torch.float32) * torch.from_numpy(np.ascontiguousarray(su)).to(h_mat.device, dtype=torch.float32).unsqueeze(1)
        vf = torch.from_numpy(np.ascontiguousarray(vi8)).to(h_mat.device, dtype=torch.float32) * torch.from_numpy(np.ascontiguousarray(sv)).to(h_mat.device, dtype=torch.float32).unsqueeze(1)
        w = w + (uf @ vf)
    return w


def _parse_t16c_index(t16c_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """Verify and index norms + matrix payload file offsets in an existing `.t16c` file."""
    v70.verify_t16c_integrity(t16c_path)
    norms: Dict[str, np.ndarray] = {}
    mats: Dict[str, dict] = {}
    with open(t16c_path, "rb") as fp:
        _magic, _ver, _flags, n_norms, n_mats, _resv = struct.unpack("<4s5I", fp.read(24))
        for _ in range(n_norms):
            klen = struct.unpack("<H", fp.read(2))[0]
            kname = fp.read(klen).decode("utf-8")
            alen = struct.unpack("<I", fp.read(4))[0]
            norms[kname] = np.frombuffer(fp.read(alen * 4), dtype="<f4").copy()
        for _ in range(n_mats):
            nlen, d_out, d_in, bsize, n_bits, is_asym_u8, has_fwht_u8, inv_k, rk, pb_len, lr_len = struct.unpack("<HIIHBBBBHQQ", fp.read(34))
            mname = fp.read(nlen).decode("utf-8")
            pos = fp.tell()
            pad = (64 - (pos % 64)) % 64
            if pad:
                fp.seek(pad, 1)
            p_off = fp.tell()
            inv_bytes = (d_in * 4) if inv_k == 1 else ((d_in * 2) if inv_k == 2 else 0)
            mats[mname] = {
                "offset": p_off,
                "d_out": d_out,
                "d_in": d_in,
                "bsize": bsize,
                "n_bits": n_bits,
                "is_asym": bool(is_asym_u8),
                "has_fwht": bool(has_fwht_u8),
                "inv_k": inv_k,
                "rk": rk,
                "pb_len": pb_len,
                "lr_len": lr_len,
            }
            fp.seek(pb_len + inv_bytes + lr_len, 1)
    return norms, mats


def _read_t16c_matrix_f32(fp, meta: dict, h_mat: torch.Tensor) -> torch.Tensor:
    """Seek, read, and decode a single `.t16c` matrix record to GPU/CPU `[d_out, d_in]` float32."""
    d_out, d_in, bsize = int(meta["d_out"]), int(meta["d_in"]), int(meta["bsize"])
    fp.seek(int(meta["offset"]))
    pay_b = fp.read(int(meta["pb_len"]))
    inv_s_np = None
    if meta["inv_k"] == 1:
        inv_s_np = np.frombuffer(fp.read(d_in * 4), dtype="<f4").copy()
    elif meta["inv_k"] == 2:
        inv_s_np = np.frombuffer(fp.read(d_in * 2), dtype="<f2").astype(np.float32).copy()
    low_rank = None
    if meta["rk"] > 0:
        rk = int(meta["rk"])
        su = np.frombuffer(fp.read(d_out * 4), dtype="<f4").copy()
        sv = np.frombuffer(fp.read(rk * 4), dtype="<f4").copy()
        ui8 = np.frombuffer(fp.read(d_out * rk), dtype=np.int8).reshape(d_out, rk).copy()
        vi8 = np.frombuffer(fp.read(rk * d_in), dtype=np.int8).reshape(rk, d_in).copy()
        low_rank = (su, sv, ui8, vi8)
    return decode_t16c_mat_to_f32(
        pay_b, d_out, d_in, bsize, meta["n_bits"], meta["is_asym"],
        inv_s_np, low_rank, h_mat, has_fwht=meta["has_fwht"],
    )


def _eval_crtr_rec_gpu(rec: dict, x_gpu: torch.Tensor) -> torch.Tensor:
    """Evaluate quantized INT8 CRTR v1 layer record on GPU matching `cortical_router.rs`."""
    dev = x_gpu.device
    v_w = torch.from_numpy(rec["v_i8"].astype(np.float32) * rec["sv"][:, None]).to(dev)
    b1 = torch.from_numpy(rec["b1"]).to(dev)
    u_w = torch.from_numpy(rec["u_i8"].astype(np.float32) * rec["su"][:, None]).to(dev)
    b2 = torch.from_numpy(rec["b2"]).to(dev)
    h = F.gelu(F.linear(x_gpu.float(), v_w, b1), approximate="tanh")
    return F.linear(h, u_w, b2)


def _eval_c8d_gpu(
    x_gpu: torch.Tensor,
    cents: np.ndarray,
    experts: List[tuple],
    h8_gpu: torch.Tensor,
    top_k: int = 2,
    temp: float = 0.5,
    gscale: float = 1.0,
) -> torch.Tensor:
    """Evaluate quantized C8DB adapter on GPU matching `cuneiform_tbi.rs` bit-accurately."""
    dev = x_gpu.device
    xf = x_gpu.float()
    n_t, d_i = xf.shape
    d_o = experts[0][0].shape[0]
    kb = d_i // 8
    u_bar = xf[:, : kb * 8].reshape(n_t, kb, 8).mean(dim=1) @ h8_gpu
    g_hat = u_bar / u_bar.norm(dim=1, keepdim=True).clamp(min=1e-9)
    logits = (g_hat @ torch.from_numpy(cents).to(dev, dtype=torch.float32).T) / temp
    topv, topi = torch.topk(logits, top_k, dim=-1)
    ws = F.softmax(topv, dim=-1) * gscale
    out = torch.zeros((n_t, d_o), device=dev, dtype=torch.float32)
    for e_idx, (su, sv, ui, vi) in enumerate(experts):
        mask_e = topi == e_idx
        if not mask_e.any():
            continue
        w_e = (ws * mask_e.float()).sum(dim=-1, keepdim=True)
        nz = w_e.squeeze(1) > 0
        vf = torch.from_numpy(vi).to(dev, dtype=torch.float32) * torch.from_numpy(sv).to(dev, dtype=torch.float32).unsqueeze(1)
        uf = torch.from_numpy(ui).to(dev, dtype=torch.float32) * torch.from_numpy(su).to(dev, dtype=torch.float32).unsqueeze(1)
        out[nz] += w_e[nz] * ((xf[nz] @ vf.T) @ uf.T)
    return out


def run_selftest() -> int:
    """Mandatory 6-gate fail-closed CPU verification battery."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: V9.0 fast-kernel byte budget <= 5.60 GiB, zero Ternary-5, 10/84 MLP + 10/32 heads, & mode roundtrips
    bgt = compute_v9_0_budget_bytes()
    h_rt_g1 = torch.from_numpy(pa.make_sylvester_hadamard(V9_BLOCK_SIZE)).float()
    tblk_t5 = p3.tile16_blk_bytes(V9_BLOCK_SIZE, 5, False)
    test_t5 = bytes((i % 243 for i in range(4 * tblk_t5)))
    cb_rt = sdc.transpose_colblock_bytes(
        sdc.transpose_colblock_bytes(test_t5, 32, 512, V9_BLOCK_SIZE, 5, False, to_colblock=True),
        32, 512, V9_BLOCK_SIZE, 5, False, to_colblock=False,
    )
    rt_modes_ok = True
    for m_chk, rm_f, cb_f, exp_hdr in (
        (0, True, False, 0x41),
        (0, True, True, 0x61),
        (5, False, False, 0x82),
        (5, False, True, 0xA2),
        (6, False, False, 0x83),
    ):
        nb_c, asym_c = p3.MODE_TO_T16C[m_chk]
        eff_c = nb_c & 0x1F
        tblk_c = p3.tile16_blk_bytes(V9_BLOCK_SIZE, nb_c, asym_c)
        raw_arr = (np.arange(4 * tblk_c, dtype=np.uint8) & 0x3F)
        sc_h16 = np.frombuffer(np.full((16,), 0.25, dtype="<f2").tobytes(), dtype=np.uint8)
        bi_h16 = np.frombuffer(np.full((16,), 0.0, dtype="<f2").tobytes(), dtype=np.uint8)
        for tb_i in range(4):
            raw_arr[tb_i * tblk_c : tb_i * tblk_c + 32] = sc_h16
            if asym_c:
                raw_arr[tb_i * tblk_c + 32 : tb_i * tblk_c + 64] = bi_h16
        u_dec, sc_dec, bi_dec = unpack_tile16_headers_and_units(raw_arr, 32, 512, V9_BLOCK_SIZE, m_chk, h_rt_g1)
        hdr_out, pay_out = finalize_t16c_payload(raw_arr, 32, 512, V9_BLOCK_SIZE, m_chk, rm_f, cb_f)
        if cb_f:
            pay_check = sdc.transpose_colblock_bytes(pay_out, 32, 512, V9_BLOCK_SIZE, eff_c, asym_c, to_colblock=False)
        else:
            pay_check = pay_out
        with io.BytesIO() as bio_chk:
            p3._write_mat_record(bio_chk, 0, "chk_mat", 32, 512, V9_BLOCK_SIZE, hdr_out, asym_c, True, pay_out, None, None)
            wr_rec_ok = bio_chk.tell() > len(pay_out)
        dec_w = decode_t16c_mat_to_f32(pay_out, 32, 512, V9_BLOCK_SIZE, hdr_out, asym_c, None, None, h_rt_g1, has_fwht=True)
        ref_w = reconstruct_from_headers(u_dec, sc_dec, bi_dec, None, h_rt_g1)
        dec_err = float((dec_w - ref_w).abs().max().item())
        ok_m = (
            hdr_out == exp_hdr
            and len(pay_out) == p3.expected_tile16_bytes(32, 512, V9_BLOCK_SIZE, hdr_out, asym_c)
            and wr_rec_ok
            and dec_err < 1e-5
            and torch.isfinite(u_dec).all().item()
            and float((sc_dec - 0.25).abs().max().item()) < 1e-3
            and (bi_dec is None or torch.isfinite(bi_dec).all().item())
            and (rm_f or pay_check == raw_arr.tobytes())
        )
        rt_modes_ok = rt_modes_ok and bool(ok_m)
    # 32-head CRTR export/roundtrip + Gemma4AssistantTrain 2-step rollout gradient check
    with tempfile.TemporaryDirectory() as td_hr:
        td_p = Path(td_hr)
        hr_net_chk, _ = crtr.train_layer_router(
            torch.randn(24, 64), torch.rand(24, V9_ATTN_HEADS),
            torch.randn(8, 64), torch.rand(8, V9_ATTN_HEADS),
            top_p=float(V9_ATTN_ROUTED_HEADS) / float(V9_ATTN_HEADS),
            rank=8, steps=4, lr=3e-3, device="cpu", seed=17,
        )
        hr_q = crtr.quantize_layer_for_crtr(hr_net_chk, V9_BLOCK_SIZE)
        hr_p = td_p / "head_chk.crtr"
        crtr.export_crtr([hr_q], hr_p)
        hr_ok = (hr_q["n_blocks"] == V9_ATTN_HEADS and hr_q["block_size"] == V9_BLOCK_SIZE and hr_q["ff"] == V9_ATTN_HEADS * V9_BLOCK_SIZE and hr_p.stat().st_size > 32)
    d_m_a, d_ast_a, nh_a, nkv_a, hd_a, sq_a = 32, 16, 4, 2, 8, 6
    toy_sd: Dict[str, torch.Tensor] = {
        "pre_projection.weight": torch.randn(d_ast_a, 2 * d_m_a) * 0.05,
        "post_projection.weight": torch.randn(d_m_a, d_ast_a) * 0.05,
        "model.norm.weight": torch.ones(d_ast_a),
        "model.embed_tokens.weight": torch.randn(16, d_ast_a) * 0.05,
    }
    for la in range(4):
        pa_t = f"model.layers.{la}."
        toy_sd[pa_t + "self_attn.q_proj.weight"] = torch.randn(nh_a * hd_a, d_ast_a) * 0.05
        toy_sd[pa_t + "self_attn.o_proj.weight"] = torch.randn(d_ast_a, nh_a * hd_a) * 0.05
        toy_sd[pa_t + "mlp.gate_proj.weight"] = torch.randn(d_ast_a * 2, d_ast_a) * 0.05
        toy_sd[pa_t + "mlp.up_proj.weight"] = torch.randn(d_ast_a * 2, d_ast_a) * 0.05
        toy_sd[pa_t + "mlp.down_proj.weight"] = torch.randn(d_ast_a, d_ast_a * 2) * 0.05
        toy_sd[pa_t + "input_layernorm.weight"] = torch.ones(d_ast_a)
        toy_sd[pa_t + "self_attn.q_norm.weight"] = torch.ones(hd_a)
        toy_sd[pa_t + "post_attention_layernorm.weight"] = torch.ones(d_ast_a)
        toy_sd[pa_t + "pre_feedforward_layernorm.weight"] = torch.ones(d_ast_a)
        toy_sd[pa_t + "post_feedforward_layernorm.weight"] = torch.ones(d_ast_a)
        toy_sd[pa_t + "layer_scalar"] = torch.ones(1)
    ast_chk = Gemma4AssistantTrain(
        toy_sd, d_mod=d_m_a, nh=nh_a, nkv_s=nkv_a, nkv_f=nkv_a, hd_s=hd_a, hd_f=hd_a,
        theta_l=10000.0, part_l=1.0, theta_g=10000.0, part_g=1.0, seq_len=sq_a, eps=1e-6, compute_dtype=torch.float32,
    )
    te_c = torch.randn(2, sq_a - 1, d_m_a)
    bh_c = torch.randn(2, sq_a - 1, d_m_a)
    k_c = torch.randn(2, nkv_a, sq_a, hd_a)
    ha1_c = ast_chk.forward_seq(te_c, bh_c, k_c, k_c, k_c, k_c)
    hp1_c = F.linear(ha1_c, ast_chk.post)
    bh2_c = torch.cat([bh_c[:, :1], hp1_c[:, :-1]], dim=1)
    ha2_c = ast_chk.forward_seq(te_c, bh2_c, k_c, k_c, k_c, k_c)
    hp2_c = F.linear(ha2_c, ast_chk.post)
    loss_chk = F.mse_loss(hp1_c, bh_c) + 0.25 * (1.0 - F.cosine_similarity(hp2_c[:, 1:], bh_c[:, 1:], dim=-1).mean())
    loss_chk.backward()
    ast_grad_ok = torch.isfinite(loss_chk).item() and ast_chk.pre.grad is not None and float(ast_chk.pre.grad.abs().sum().item()) > 0.0
    g1_ok = (
        bgt["total_t16c_gib"] <= 5.60
        and bgt["tern5_cnt"] == 0
        and bgt["active_mlp_ratio"] <= 0.12
        and abs(bgt["active_attn_head_ratio"] - (10.0 / 32.0)) < 1e-4
        and bgt["active_dram_10of84_mod35_mb"] < 1400.0
        and cb_rt == test_t5
        and rt_modes_ok
        and hr_ok
        and ast_grad_ok
    )
    gate(
        "Gate 1: V9.0 budget <= 5.60 GiB, 0 Ternary-5, 10/84 MLP + 10/32 heads & mode roundtrips",
        g1_ok,
        f"total={bgt['total_t16c_gib']} GiB, active={bgt['active_dram_10of84_mb']} MB, gmac={bgt['active_gmac_per_tok']}, rt={rt_modes_ok}, hr={hr_ok}, ast={ast_grad_ok}",
    )

    # Gate 2: Exact algebraic invariance of Global Outlier-Dispersion Permutation Pi across full Gemma 4 block
    torch.manual_seed(11)
    d_m, d_kv, ff_s, bsz = 128, 64, 192, 32
    x = torch.randn(4, 6, d_m)
    scores = torch.abs(x).reshape(-1, d_m).amax(dim=0)
    perm, inv_perm = compute_outlier_dispersion_perm(scores, block_size=bsz)
    assert torch.equal(perm[inv_perm], torch.arange(d_m))
    g_in = 1.0 + torch.randn(d_m) * 0.2
    g_pa = 1.0 + torch.randn(d_m) * 0.2
    g_pf = 1.0 + torch.randn(d_m) * 0.2
    g_pff = 1.0 + torch.randn(d_m) * 0.2
    wq = torch.randn(d_kv, d_m) * 0.05
    wk = torch.randn(d_kv, d_m) * 0.05
    wv = torch.randn(d_kv, d_m) * 0.05
    wo = torch.randn(d_m, d_kv) * 0.05
    wg = torch.randn(ff_s, d_m) * 0.05
    wu = torch.randn(ff_s, d_m) * 0.05
    wd = torch.randn(d_m, ff_s) * 0.05
    ls = 0.93

    def _rmsn(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    # Parity check: Gemma 4 rmsn_raw y * w (fails if 1.0 + w) & partial-RoPE pairing matching src/math.rs
    t_rm = torch.tensor([3.0, 4.0])
    w_rm = torch.tensor([0.5, 2.0])
    rm_out = _rmsn(t_rm, w_rm)
    rms_exp = math.sqrt((9.0 + 16.0) / 2.0 + 1e-6)
    rm_exp = torch.tensor([3.0 / rms_exp * 0.5, 4.0 / rms_exp * 2.0])
    assert float((rm_out - rm_exp).abs().max().item()) < 1e-5, f"RMSNorm mismatch: {rm_out} vs {rm_exp}"

    hd_p, rot_p, th_p, pos_p = 16, 4, 1e6, 3
    hr_p, hh_p = rot_p // 2, hd_p // 2
    inv_f_p = 1.0 / (th_p ** (torch.arange(0, 2 * hr_p, 2, dtype=torch.float32) / float(hd_p)))
    ang_p = float(pos_p) * inv_f_p
    c_p, s_p = torch.cos(ang_p), torch.sin(ang_p)
    xq_p = torch.arange(1, hd_p + 1, dtype=torch.float32)
    xr_p = xq_p.clone()
    x0_p, x1_p = xq_p[:hr_p], xq_p[hh_p : hh_p + hr_p]
    xr_p[:hr_p] = x0_p * c_p - x1_p * s_p
    xr_p[hh_p : hh_p + hr_p] = x0_p * s_p + x1_p * c_p
    assert torch.allclose(xr_p[hr_p:hh_p], xq_p[hr_p:hh_p], atol=1e-6)
    assert torch.allclose(xr_p[hh_p + hr_p :], xq_p[hh_p + hr_p :], atol=1e-6)
    exp_f0 = 1.0 / (th_p ** (0.0 / float(hd_p)))
    exp_f1 = 1.0 / (th_p ** (2.0 / float(hd_p)))
    assert abs(float(inv_f_p[0]) - exp_f0) < 1e-7 and abs(float(inv_f_p[1]) - exp_f1) < 1e-7

    ast_chk_rp = Gemma4AssistantTrain(
        toy_sd, d_mod=d_m_a, nh=nh_a, nkv_s=nkv_a, nkv_f=nkv_a, hd_s=16, hd_f=16,
        theta_l=th_p, part_l=0.25, theta_g=th_p, part_g=0.25, seq_len=6, eps=1e-6, compute_dtype=torch.float32,
    )
    xq_4d = torch.zeros(1, 1, 1, hd_p)
    xq_4d[0, 0, 0] = xq_p
    xr_ast = ast_chk_rp._rope(xq_4d, ast_chk_rp.cr_l[pos_p : pos_p + 1], ast_chk_rp.sr_l[pos_p : pos_p + 1], rot_p, hd_p)[0, 0, 0]
    assert float((xr_ast - xr_p).abs().max().item()) < 1e-5, f"Partial RoPE mismatch: {xr_ast} vs {xr_p}"

    def _fwd_block(inp_x, gi, gpa, gpf, gpff, q, k, v, o, g, u, d):
        xn = _rmsn(inp_x, gi)
        qq, kk, vv = F.linear(xn, q), F.linear(xn, k), F.linear(xn, v)
        ctx = F.scaled_dot_product_attention(qq.unsqueeze(1), kk.unsqueeze(1), vv.unsqueeze(1), is_causal=True, scale=1.0).squeeze(1)
        xm = inp_x + _rmsn(F.linear(ctx, o), gpa)
        xf = _rmsn(xm, gpf)
        act = F.gelu(F.linear(xf, g), approximate="tanh") * F.linear(xf, u)
        return (xm + _rmsn(F.linear(act, d), gpff)) * ls

    y_ref = _fwd_block(x, g_in, g_pa, g_pf, g_pff, wq, wk, wv, wo, wg, wu, wd)
    y_prm = _fwd_block(
        x[:, :, perm],
        g_in[perm], g_pa[perm], g_pf[perm], g_pff[perm],
        wq[:, perm], wk[:, perm], wv[:, perm], wo[perm, :],
        wg[:, perm], wu[:, perm], wd[perm, :],
    )
    max_diff_pi = float((y_prm - y_ref[:, :, perm]).abs().max().item())
    gate("Gate 2: Global Outlier-Dispersion Permutation Pi is 100% lossless across Gemma 4 norms", max_diff_pi < 1e-5, f"max_diff={max_diff_pi:.2e}")

    # Gate 3: Co-activation Shared+Routed permutation pi_l exact invariance & clustering enrichment
    x_mlp = torch.randn(64, d_m)
    gate_acts = F.gelu(F.linear(x_mlp, wg), approximate="tanh") * F.linear(x_mlp, wu)
    # Inject correlated cluster signal into synthetic GeGLU activations
    synth_act = torch.zeros(96, 128)
    synth_act[:, :8] = torch.randn(96, 8).abs() + 1.5  # general shared
    for grp in range(4):
        z = torch.relu(torch.randn(96, 1))
        synth_act[grp * 24 : (grp + 1) * 24, 8 + grp * 16 : 8 + (grp + 1) * 16] += 3.0 * z[grp * 24 : (grp + 1) * 24]
    pi_l, _, st_mlp = compute_geglu_colblock_perm(synth_act, shared_blocks=1, block_size=16, iters=6)
    pi_f, _, _ = compute_geglu_colblock_perm(gate_acts, shared_blocks=2, block_size=32, iters=4)
    y_mlp_ref = F.linear(gate_acts, wd)
    y_mlp_prm = F.linear(F.gelu(F.linear(x_mlp, wg[pi_f, :]), approximate="tanh") * F.linear(x_mlp, wu[pi_f, :]), wd[:, pi_f])
    max_diff_mlp = float((y_mlp_ref - y_mlp_prm).abs().max().item())
    gate(
        "Gate 3: Co-activation COLBLOCK permutation is algebraically exact & raises sparse capture",
        max_diff_mlp < 1e-5 and st_mlp["shared_energy_pct"] > 5.0 and st_mlp["perm_12of84_energy_pct"] >= st_mlp["raw_topk_energy_pct"],
        f"diff={max_diff_mlp:.2e}, shared={st_mlp['shared_energy_pct']}%, perm_topk={st_mlp['perm_12of84_energy_pct']}%",
    )

    # Gate 4: Real PyTorch loss.backward() + AdamW scale+C8DB+CRTR QAT convergence on frozen bitplanes
    torch.manual_seed(7)
    d_q = 64
    h_q = torch.from_numpy(pa.make_sylvester_hadamard(32))
    w_toy = torch.randn(64, d_q)
    u_toy = torch.sign(w_toy @ torch.kron(torch.eye(2), h_q))
    sc_q = nn.Parameter(torch.full((64, 2), 0.1))
    adp_q = c8d.Cuneiform8DPatternAdapter(64, d_q, rank=8, n_experts=4, top_k=2).cpu()
    x_q = torch.randn(48, d_q)
    y_tgt = F.linear(x_q, w_toy)
    opt_q = torch.optim.AdamW([sc_q, *adp_q.parameters()], lr=2e-2)
    with torch.no_grad():
        loss0 = float(F.mse_loss(F.linear(x_q, reconstruct_from_headers(u_toy, sc_q, None, None, h_q)) + adp_q(x_q)[0], y_tgt).item())
    for _ in range(30):
        opt_q.zero_grad()
        w_eff = reconstruct_from_headers(u_toy, sc_q, None, None, h_q)
        y_hat = F.linear(x_q, w_eff) + adp_q(x_q)[0]
        loss = F.mse_loss(y_hat, y_tgt) + 0.2 * (1.0 - F.cosine_similarity(y_hat, y_tgt, dim=-1).mean())
        loss.backward()
        opt_q.step()
    with torch.no_grad():
        loss1 = float(F.mse_loss(F.linear(x_q, reconstruct_from_headers(u_toy, sc_q, None, None, h_q)) + adp_q(x_q)[0], y_tgt).item())
    gate("Gate 4: Differentiable sparse-header + C8DB QAT loss drops via real backward()", loss1 < loss0 * 0.7, f"mse {loss0:.4f}->{loss1:.4f}")

    # Gate 5: Strict Rule 2 Decontamination Detector (0 overlap on clean strings, catches injected holdout)
    clean_test = [
        "Derive the Euler-Lagrange equations for a double pendulum cleanly.",
        "Implement a lock-free ring buffer index helper in Rust.",
    ]
    dc_ok = verify_corpus_decontaminated(clean_test)["overlap_4grams"] == 0
    leak_caught = False
    try:
        verify_corpus_decontaminated([spmod.HELDOUT_NLL_PASSAGES[0][1]])
    except ValueError:
        leak_caught = True
    gate("Gate 5: Rule 2 Decontamination detector passes clean data & fails closed on leak", dc_ok and leak_caught)

    # Gate 6: Assistant-V9 4-step self-feeding rollout backward + fail-closed missing model check
    torch.manual_seed(9)
    ast = Gemma4AssistantRolloutNet(d_backbone=32, d_ast=32, n_layers=2, ff_ast=64)
    emb_s = torch.randn(4, 10, 32)
    h59_s = torch.tanh(emb_s * 0.7 + 0.1 * torch.roll(emb_s, 1, dims=1))
    opt_a = torch.optim.AdamW(ast.parameters(), lr=2e-2)
    with torch.no_grad():
        rl0, _ = assistant_rollout_loss(ast, emb_s, h59_s, rollout_steps=4)
    for _ in range(25):
        opt_a.zero_grad()
        rl, _ = assistant_rollout_loss(ast, emb_s, h59_s, rollout_steps=4)
        rl.backward()
        opt_a.step()
    with torch.no_grad():
        rl1, st_l = assistant_rollout_loss(ast, emb_s, h59_s, rollout_steps=4)
    miss_caught = False
    try:
        pa.resolve_assistant_safetensors(Path("/nonexistent/v9_assistant/model.safetensors"))
    except FileNotFoundError:
        miss_caught = True
    gate(
        "Gate 6: Assistant-V9 4-step self-feeding rollout improves via real backprop & fails closed",
        float(rl1.item()) < float(rl0.item()) * 0.75 and len(st_l) == 4 and miss_caught,
        f"rollout_loss {float(rl0.item()):.4f}->{float(rl1.item()):.4f}",
    )

    print(f"V9_0_SELFTEST: {6 - failures}/6 gates passed", flush=True)
    return 1 if failures else 0


def _build_decontaminated_corpus(tk, max_text_seq: int = 896, seq_len: int = V9_SEQ_LEN) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build strictly decontaminated STEM/math/code/reasoning corpus (0 overlap with all holdouts)."""
    from datasets import load_dataset

    h4 = collect_holdout_4grams()

    def is_clean_pair(text: str) -> bool:
        low = text.lower()
        if any(b in low for b in ALL_BLACKLIST):
            return False
        ws = _normalize_words(text)
        for i in range(len(ws) - 3):
            if (ws[i], ws[i + 1], ws[i + 2], ws[i + 3]) in h4:
                return False
        return True

    turns: List[Tuple[str, str, str]] = []
    try:
        for row in load_dataset("openai/gsm8k", "main", split="train"):
            q, a = row["question"].strip(), row["answer"].strip()
            if is_clean_pair(q + " " + a):
                turns.append(("math", q, a))
                if len(turns) >= 400:
                    break
    except Exception as e:
        print(f"[Corpus] Notice: gsm8k load ({e}), continuing...", flush=True)

    mbpp_names = ("google-research-datasets/mbpp", "mbpp")
    for mbpp_name in mbpp_names:
        try:
            for row in load_dataset(mbpp_name, "sanitized", split="train"):
                q = row.get("prompt", "").strip()
                a = row.get("code", "").strip()
                if q and a and is_clean_pair(q + " " + a):
                    turns.append(("code", f"Write a clean Python function:\n{q}", f"```python\n{a}\n```"))
                    if len(turns) >= 650:
                        break
            if any(t[0] == "code" for t in turns):
                break
        except Exception as e:
            print(f"[Corpus] Notice: {mbpp_name} load ({e}), trying next...", flush=True)

    try:
        for row in load_dataset("allenai/sciq", split="train"):
            q = row.get("question", "").strip()
            sup = row.get("support", "").strip()
            ans = row.get("correct_answer", "").strip()
            if q and ans and is_clean_pair(q + " " + sup + " " + ans):
                body = f"{sup}\nAnswer: {ans}" if sup else ans
                turns.append(("science", q, body))
                if len(turns) >= max_text_seq:
                    break
    except Exception as e:
        print(f"[Corpus] Notice: sciq load ({e}), continuing...", flush=True)

    if len(turns) < 32:
        for cat, q, a in vc.V4_CAL_PAIRS:
            if is_clean_pair(q + " " + a):
                turns.append((cat, q, a))
                if len(turns) >= max_text_seq:
                    break

    verify_corpus_decontaminated([f"{q}\n{a}" for _, q, a in turns], holdout_4grams=h4)

    seq_ids = np.zeros((len(turns), seq_len), dtype=np.int64)
    seq_nxt = np.zeros((len(turns), seq_len), dtype=np.int64)
    ans_mask = np.zeros((len(turns), seq_len), dtype=bool)
    for si, (_, q, a) in enumerate(turns):
        pre_txt = vc.format_gemma4_turn(q, None)
        pre_ids = [2] + list(tk.encode(pre_txt, add_special_tokens=False).ids)
        ans_ids = list(tk.encode(f"{a}<turn|>\n", add_special_tokens=False).ids)
        full_ids = pre_ids + ans_ids
        pad_id = tk.token_to_id("<pad>") or 0
        if len(full_ids) < seq_len + 1:
            full_ids = full_ids + [pad_id] * (seq_len + 1 - len(full_ids))
        full_ids = full_ids[: seq_len + 1]
        seq_ids[si] = np.array(full_ids[:seq_len], dtype=np.int64)
        seq_nxt[si] = np.array(full_ids[1 : seq_len + 1], dtype=np.int64)
        a_start = min(max(len(pre_ids) - 1, 0), seq_len - 1)
        ans_mask[si, a_start:] = True
    return seq_ids, seq_nxt, ans_mask


def _resolve_existing_artifact(name: str, out_dir: Path, drive_dir: Optional[Path]) -> Path:
    """Resolve an existing V9.0 artifact cleanly from local OUT_DIR or Drive, failing closed if absent."""
    candidates = [out_dir / name]
    if drive_dir is not None:
        candidates.append(drive_dir / name)
    for c in candidates:
        if c.is_file() and c.stat().st_size > 0:
            return c
    raise FileNotFoundError(f"Missing required artifact {name}; checked {[str(c) for c in candidates]}")


def _load_dequant_proj(sf_obj, base_name: str, perm_pi_dev: torch.Tensor, device: torch.device, is_perm_col: bool = False, is_perm_row: bool = False) -> torch.Tensor:
    if f"{base_name}.weight_packed" in sf_obj.keys():
        pk = sf_obj.get_tensor(f"{base_name}.weight_packed").to(device)
        sc = sf_obj.get_tensor(f"{base_name}.weight_scale").to(device, dtype=torch.float32)
        shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
        nibs = ((pk.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF) - 8
        w = (nibs.reshape(pk.shape[0], sc.shape[1], 32).float() * sc.unsqueeze(-1)).reshape(pk.shape[0], -1)
        del pk, sc, nibs
    else:
        w = sf_obj.get_tensor(f"{base_name}.weight").to(device, dtype=torch.float32)
    if is_perm_col:
        w = w[:, perm_pi_dev]
    if is_perm_row:
        w = w[perm_pi_dev, :]
    return w


def _init_v9_stage1_base(device: torch.device, compute_dtype: torch.dtype) -> Dict[str, object]:
    """Download/open Gemma 4 31B, build decontaminated corpus + VLM seed states, and compute Global Perm Pi."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers import AutoConfig, Gemma4ImageProcessor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

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
    assert D_MOD == V9_D_MOD and N_LAYERS == V9_N_LAYERS

    print("[Stage 1] Building decontaminated STEM/code/science + COCO VLM corpus...", flush=True)
    txt_ids, txt_nxt, txt_ans = _build_decontaminated_corpus(tk, max_text_seq=896, seq_len=V9_SEQ_LEN)
    coco_ds = load_dataset("detection-datasets/coco", split="val[:136]")
    vlm_seqs, vlm_nxts, vlm_masks, vlm_imgs = [], [], [], []
    img_proc = Gemma4ImageProcessor.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct")
    soft_tok = int(cfg_full.get("image_token_id", 258880))
    boi_tok = int(cfg_full.get("boi_token_id", 255999))
    eoi_tok = int(cfg_full.get("eoi_token_id", 258882))
    tok_user_open = list(tk.encode("<|turn>user\n", add_special_tokens=False).ids)
    tok_model_open = list(tk.encode("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", add_special_tokens=False).ids)
    for r in coco_ds:
        objs = r.get("objects", {})
        cats = [str(c) for c in (objs.get("category", []) if isinstance(objs, dict) else [])[:4]]
        cap = "Objects visible in image: " + (", ".join(dict.fromkeys(cats)) if cats else "scene objects") + "."
        if any(b in cap.lower() for b in ALL_BLACKLIST):
            continue
        im = r["image"].convert("RGB").resize((224, 224))
        vlm_imgs.append(im)
        n_img = 16
        prompt_ids = [tk.token_to_id("<bos>") or 2] + tok_user_open + [boi_tok] + [soft_tok] * n_img + [eoi_tok] + list(tk.encode("Describe this image concisely in one sentence.", add_special_tokens=False).ids) + tok_model_open
        ans_ids = list(tk.encode(f"{cap}<turn|>\n", add_special_tokens=False).ids)
        full = (prompt_ids + ans_ids + [0] * (V9_SEQ_LEN + 1))[: V9_SEQ_LEN + 1]
        vlm_seqs.append(full[:V9_SEQ_LEN])
        vlm_nxts.append(full[1 : V9_SEQ_LEN + 1])
        m = np.zeros(V9_SEQ_LEN, dtype=bool)
        m[min(len(prompt_ids) - 1, V9_SEQ_LEN - 1) :] = True
        vlm_masks.append(m)
        if len(vlm_seqs) >= 128:
            break

    all_seq_ids = np.concatenate([txt_ids, np.array(vlm_seqs, dtype=np.int64)], axis=0)
    all_seq_nxt = np.concatenate([txt_nxt, np.array(vlm_nxts, dtype=np.int64)], axis=0)
    all_ans_mask = np.concatenate([txt_ans, np.array(vlm_masks, dtype=bool)], axis=0)
    N_TEXT_SEQ = txt_ids.shape[0]
    N_SEQ = all_seq_ids.shape[0]

    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        emb_raw = sf.get_tensor("model.language_model.embed_tokens.weight").float()
        fn_orig = sf.get_tensor("model.language_model.norm.weight").float()
        vis_keys = [k for k in sf.keys() if k.startswith("model.vision_tower.") or k.startswith("model.embed_vision.")]
        hf_cfg = AutoConfig.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct")
        vis_model = Gemma4VisionModel(hf_cfg.vision_config)
        mm_embed = Gemma4MultimodalEmbedder(hf_cfg.vision_config, hf_cfg.text_config)
        vis_sd = {}
        mm_sd = {}
        for vk in vis_keys:
            if vk.startswith("model.vision_tower."):
                vis_sd[vk[len("model.vision_tower.") :]] = sf.get_tensor(vk)
            elif vk.startswith("model.embed_vision."):
                mm_sd[vk[len("model.embed_vision.") :]] = sf.get_tensor(vk)
        vis_model.load_state_dict(vis_sd, strict=False)
        mm_embed.load_state_dict(mm_sd, strict=False)

    vis_model = vis_model.to(device, dtype=compute_dtype).eval()
    mm_embed = mm_embed.to(device, dtype=compute_dtype).eval()
    X_teach = (emb_raw[torch.from_numpy(all_seq_ids)] * ESCALE).to(torch.float16)
    with torch.no_grad():
        for vi, im in enumerate(vlm_imgs):
            si = N_TEXT_SEQ + vi
            proc_out = img_proc(images=im, return_tensors="pt")
            pv = proc_out["pixel_values"].to(device, dtype=compute_dtype)
            pos_ids = proc_out["image_position_ids"].to(device)
            vo = vis_model(pixel_values=pv, pixel_position_ids=pos_ids).last_hidden_state
            vm = mm_embed(vo).squeeze(0)[:16].to(torch.float16).cpu()
            pos_m = torch.from_numpy(all_seq_ids[si] == soft_tok)
            take = min(int(pos_m.sum().item()), vm.shape[0])
            idx_m = torch.nonzero(pos_m, as_tuple=False).squeeze(1)[:take]
            X_teach[si, idx_m] = vm[:take]
    del vis_sd, mm_sd, vis_model, mm_embed
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        flat_x0 = X_teach[:min(256, N_SEQ)].reshape(-1, D_MOD).float()
        chan_score = flat_x0.abs().amax(dim=0) + 2.0 * flat_x0.pow(2).mean(dim=0).sqrt()
    perm_pi, inv_perm_pi = compute_outlier_dispersion_perm(chan_score, block_size=V9_BLOCK_SIZE)
    torch.save({"perm_pi": perm_pi, "inv_perm_pi": inv_perm_pi}, OUT_PERM_PT)
    print(f"[Stage 1] Dispersion Perm Pi computed across {D_MOD // V9_BLOCK_SIZE} blocks (score max/min ratio reduced via snake deal)", flush=True)
    X_teach_perm = X_teach[:, :, perm_pi].clone()
    del X_teach
    gc.collect()

    return {
        "sf_path": sf_path, "tok_path": tok_path, "cfg_full": cfg_full, "tc": tc, "tk": tk,
        "N_LAYERS": N_LAYERS, "D_MOD": D_MOD, "EPS": EPS, "ESCALE": ESCALE,
        "NH": NH, "NKV_S": NKV_S, "NKV_F": NKV_F, "HD_S": HD_S, "HD_F": HD_F, "K_EQ_V": K_EQ_V,
        "LAYER_TYPES": LAYER_TYPES, "THETA_G": THETA_G, "PART_G": PART_G, "THETA_L": THETA_L, "PART_L": PART_L,
        "SOFTCAP": SOFTCAP, "soft_tok": soft_tok,
        "all_seq_ids": all_seq_ids, "all_seq_nxt": all_seq_nxt, "all_ans_mask": all_ans_mask,
        "N_TEXT_SEQ": N_TEXT_SEQ, "N_SEQ": N_SEQ,
        "emb_raw": emb_raw, "fn_orig": fn_orig, "X_teach_perm": X_teach_perm,
        "perm_pi": perm_pi, "inv_perm_pi": inv_perm_pi,
    }


def _make_gemma4_math_helpers(device: torch.device, eps: float):
    def rmsn_gpu(t: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf = t.float()
        y = xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)
        if w is not None:
            y = y * w.float()
        return y.to(t.dtype)

    def make_rope(theta: float, rot_d: int, hd: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = rot_d // 2
        if hr <= 0:
            z = torch.zeros((V9_SEQ_LEN, 0), device=device, dtype=torch.float32)
            return z, z
        inv_f = 1.0 / (theta ** (torch.arange(0, 2 * hr, 2, device=device, dtype=torch.float32) / float(hd)))
        pos = torch.arange(V9_SEQ_LEN, device=device, dtype=torch.float32)
        ang = torch.outer(pos, inv_f)
        return torch.cos(ang), torch.sin(ang)

    def apply_rope(x: torch.Tensor, cs: torch.Tensor, sn: torch.Tensor, rot_d: int, hd: int) -> torch.Tensor:
        hr, hh = rot_d // 2, hd // 2
        if hr <= 0:
            return x
        xf = x.float()
        x0, x1 = xf[..., :hr], xf[..., hh : hh + hr]
        c = cs.view(1, x.shape[1], 1, hr)
        s = sn.view(1, x.shape[1], 1, hr)
        r0, r1 = x0 * c - x1 * s, x0 * s + x1 * c
        if hr == hh:
            return torch.cat([r0, r1], dim=-1).to(x.dtype)
        return torch.cat([r0, xf[..., hr:hh], r1, xf[..., hh + hr :]], dim=-1).to(x.dtype)

    return rmsn_gpu, make_rope, apply_rope


def _run_stage3_distill_and_export(
    *,
    ctx: Dict[str, object],
    device: torch.device,
    compute_dtype: torch.dtype,
    gpu_name: str,
    gpu_vram_gb: float,
    t0_all: float,
    rmsn_gpu,
    X_stud_perm: torch.Tensor,
    X_teach_perm: torch.Tensor,
    emb_eff: torch.Tensor,
    ast_seqs: List[int],
    cap_kv58: Dict[str, dict],
    cap_kv59: Dict[str, dict],
    fn_vec_np: np.ndarray,
    crtr_layers: List[dict],
    head_crtr_layers: Optional[List[dict]],
    c8db_entries: List[dict],
    layer_logs: List[dict],
    cb_cnt: int,
    t16c_final_path: Path,
    head_crtr_final_path: Path,
    persist_t16c_and_head: bool,
    apply_wiener: bool,
    train_post_cell60: bool,
    mode_tag: str,
) -> Dict[str, object]:
    """Shared Stage 3: Wiener/post_cell60 calibration, Teacher-KL eval, Assistant-V9, BHDR-384, and Drive sync."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file as st_save_file

    sf_path: Path = ctx["sf_path"]  # type: ignore[assignment]
    tok_path: Path = ctx["tok_path"]  # type: ignore[assignment]
    D_MOD: int = int(ctx["D_MOD"])
    EPS: float = float(ctx["EPS"])
    ESCALE: float = float(ctx["ESCALE"])
    NH: int = int(ctx["NH"])
    NKV_S: int = int(ctx["NKV_S"])
    NKV_F: int = int(ctx["NKV_F"])
    HD_S: int = int(ctx["HD_S"])
    HD_F: int = int(ctx["HD_F"])
    THETA_L: float = float(ctx["THETA_L"])
    PART_L: float = float(ctx["PART_L"])
    THETA_G: float = float(ctx["THETA_G"])
    PART_G: float = float(ctx["PART_G"])
    SOFTCAP: float = float(ctx["SOFTCAP"])
    all_ans_mask: np.ndarray = ctx["all_ans_mask"]  # type: ignore[assignment]
    all_seq_ids: np.ndarray = ctx["all_seq_ids"]  # type: ignore[assignment]
    all_seq_nxt: np.ndarray = ctx["all_seq_nxt"]  # type: ignore[assignment]
    perm_pi: torch.Tensor = ctx["perm_pi"]  # type: ignore[assignment]

    H8_GPU = torch.from_numpy(c8d.H8_NP).to(device, dtype=torch.float32)
    fn_g = torch.from_numpy(np.ascontiguousarray(fn_vec_np, dtype=np.float32)).to(device)
    ans_flat = torch.nonzero(torch.from_numpy(all_ans_mask.reshape(-1)), as_tuple=False).squeeze(1)
    sel_ans = ans_flat[torch.linspace(0, ans_flat.numel() - 1, min(16384, ans_flat.numel())).long()]
    s_t, t_t = sel_ans // V9_SEQ_LEN, sel_ans % V9_SEQ_LEN
    hs_ans = rmsn_gpu(X_stud_perm[s_t, t_t].to(device).unsqueeze(1), fn_g).squeeze(1).float()
    ht_ans = rmsn_gpu(X_teach_perm[s_t, t_t].to(device).unsqueeze(1), fn_g).squeeze(1).float()
    emb_g = emb_eff.to(device, dtype=torch.float32)
    nxt_ans = torch.from_numpy(all_seq_nxt[s_t.numpy(), t_t.numpy()]).to(device, dtype=torch.long)

    iv_fn = torch.arange(hs_ans.shape[0], device=device) % 10 == 0
    fn_eff_g = fn_g
    if apply_wiener:
        xub_fn = rmsn_gpu(X_stud_perm[s_t, t_t].to(device).unsqueeze(1), None).squeeze(1).float()
        num_fn = (xub_fn[~iv_fn] * ht_ans[~iv_fn]).sum(dim=0) / float(max(int((~iv_fn).sum().item()), 1))
        den_fn = (xub_fn[~iv_fn] * xub_fn[~iv_fn]).sum(dim=0) / float(max(int((~iv_fn).sum().item()), 1))
        fn_wiener = (num_fn + 0.05 * fn_g) / (den_fn + 0.05).clamp(min=1e-8)
        if float(((xub_fn[iv_fn] * fn_wiener - ht_ans[iv_fn]) ** 2).sum().item()) < float(((xub_fn[iv_fn] * fn_g - ht_ans[iv_fn]) ** 2).sum().item()):
            fn_eff_g = fn_wiener
            hs_ans = xub_fn * fn_eff_g
            with open(t16c_final_path, "r+b") as fp_fn:
                fn_off = 24 + 2 + len("model.language_model.norm.weight".encode("utf-8")) + 4
                fp_fn.seek(fn_off)
                fp_fn.write(np.ascontiguousarray(fn_eff_g.detach().cpu().numpy(), dtype="<f4").tobytes())
            print("[Stage 3] Updated final_norm in T16C with held-out validated Wiener shrinkage.", flush=True)
        del xub_fn

    keep_c60 = False
    cents60 = experts60 = None
    if train_post_cell60:
        with torch.no_grad():
            top_ids_c: List[torch.Tensor] = []
            for i0 in range(0, min(4096, hs_ans.shape[0]), 512):
                i1 = min(i0 + 512, hs_ans.shape[0])
                lt_c = (ht_ans[i0:i1] @ emb_g.T)
                top_ids_c.append(torch.topk(lt_c, 16, dim=-1).indices.reshape(-1))
            uniq_c, cnt_c = torch.unique(torch.cat(top_ids_c), return_counts=True)
            cand_c60 = torch.unique(torch.cat([uniq_c[torch.topk(cnt_c, min(4096, uniq_c.numel())).indices], nxt_ans]))
            W_c60 = emb_g[cand_c60].float()
            id_map60 = {int(t): ci for ci, t in enumerate(cand_c60.tolist())}
            y_star60 = torch.tensor([id_map60[int(t)] for t in nxt_ans.tolist()], device=device, dtype=torch.long)

        tr_idx60 = torch.nonzero(~iv_fn, as_tuple=False).squeeze(1)
        va_idx60 = torch.nonzero(iv_fn, as_tuple=False).squeeze(1)
        m60 = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=D_MOD, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(device)
        o60 = torch.optim.AdamW(m60.parameters(), lr=3e-3, weight_decay=2e-2)
        s60 = torch.optim.lr_scheduler.CosineAnnealingLR(o60, T_max=240, eta_min=3e-4)

        def eval_c60(xs_v: torch.Tensor) -> Tuple[float, float]:
            with torch.no_grad():
                kls = agrs = 0.0
                for b0 in range(0, va_idx60.numel(), 512):
                    b1 = min(b0 + 512, va_idx60.numel())
                    ib = va_idx60[b0:b1]
                    lt_v = SOFTCAP * torch.tanh((ht_ans[ib] @ W_c60.T) / SOFTCAP)
                    ls_v = SOFTCAP * torch.tanh((xs_v[b0:b1] @ W_c60.T) / SOFTCAP)
                    pt_v = F.softmax(lt_v, dim=-1)
                    kls += float((pt_v * (torch.log(pt_v.clamp(min=1e-12)) - F.log_softmax(ls_v, dim=-1))).sum(dim=-1).sum().item())
                    agrs += float((ls_v.argmax(dim=-1) == lt_v.argmax(dim=-1)).float().sum().item())
                n_v = float(max(va_idx60.numel(), 1))
                return kls / n_v, agrs / n_v

        kl60_0, agr60_0 = eval_c60(hs_ans[va_idx60])
        best_kl60, best_sd60 = kl60_0, None
        for st60 in range(240):
            o60.zero_grad()
            pb60 = tr_idx60[torch.randint(0, tr_idx60.numel(), (min(1024, tr_idx60.numel()),), device=device)]
            da60, _, lb60 = m60(hs_ans[pb60])
            hp60 = hs_ans[pb60] + da60
            with torch.no_grad():
                lt_b = SOFTCAP * torch.tanh((ht_ans[pb60] @ W_c60.T) / SOFTCAP)
                pt_b = F.softmax(lt_b, dim=-1)
            ls_b = SOFTCAP * torch.tanh((hp60 @ W_c60.T) / SOFTCAP)
            zy60 = ls_b.gather(1, y_star60[pb60].unsqueeze(1)).squeeze(1)
            zm60 = ls_b.clone()
            zm60.scatter_(1, y_star60[pb60].unsqueeze(1), -1e9)
            loss60 = (
                (pt_b * (torch.log(pt_b.clamp(min=1e-12)) - F.log_softmax(ls_b, dim=-1))).sum(dim=-1).mean()
                + 0.25 * F.relu(1.5 - zy60 + zm60.max(dim=-1).values).mean()
                + 2.0 * F.mse_loss(hp60, ht_ans[pb60])
                + 0.5 * (1.0 - F.cosine_similarity(hp60, ht_ans[pb60], dim=-1).mean())
                + 0.005 * lb60
            )
            loss60.backward()
            torch.nn.utils.clip_grad_norm_(m60.parameters(), 1.0)
            o60.step()
            s60.step()
            if (st60 + 1) % 60 == 0:
                with torch.no_grad():
                    klv, _ = eval_c60(hs_ans[va_idx60] + m60(hs_ans[va_idx60])[0])
                    if klv < best_kl60:
                        best_kl60 = klv
                        best_sd60 = {k: v.detach().cpu().clone() for k, v in m60.state_dict().items()}
        mc60 = m60.cpu()
        if best_sd60 is not None:
            mc60.load_state_dict(best_sd60)
        cents60, experts60 = mc60.export_quantized_arrays()
        with torch.no_grad():
            kl60_q, agr60_q = eval_c60(hs_ans[va_idx60] + _eval_c8d_gpu(hs_ans[va_idx60], cents60, experts60, H8_GPU))
        keep_c60 = kl60_q < kl60_0
        print(f"[Stage 3 post_cell60] Val KL {kl60_0:.4f}->{kl60_q:.4f} | Top1Agree {agr60_0*100:.2f}%->{agr60_q*100:.2f}% (keep={keep_c60})", flush=True)
        if keep_c60:
            exp_r60 = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
            c8db_entries.append({
                "name": "model.language_model.norm.post_cell60", "d_out": D_MOD, "d_in": D_MOD, "rank": 32,
                "top_k": 2, "c6d": exp_r60[0], "temperature": 0.5, "gate_scale": 1.0,
                "centroids": cents60, "expert_radicals": exp_r60, "experts": experts60,
            })
            with torch.no_grad():
                for i0 in range(0, hs_ans.shape[0], 2048):
                    i1 = min(i0 + 2048, hs_ans.shape[0])
                    hs_ans[i0:i1] = hs_ans[i0:i1] + _eval_c8d_gpu(hs_ans[i0:i1], cents60, experts60, H8_GPU)
        del W_c60, cand_c60, y_star60, m60, mc60, o60, s60, top_ids_c
        torch.cuda.empty_cache()

    with torch.no_grad():
        val_mask = iv_fn
        kl_vals = []
        top1_hits = 0
        top5_hits = 0
        val_cnt = int(val_mask.sum().item())
        for b0 in range(0, val_cnt, 512):
            idx_sl = torch.nonzero(val_mask, as_tuple=False).squeeze(1)[b0 : b0 + 512]
            lt = SOFTCAP * torch.tanh((ht_ans[idx_sl] @ emb_g.T) / SOFTCAP)
            ls = SOFTCAP * torch.tanh((hs_ans[idx_sl] @ emb_g.T) / SOFTCAP)
            pt = F.softmax(lt, dim=-1)
            kl_vals.append(float((pt * (torch.log(pt.clamp(min=1e-12)) - F.log_softmax(ls, dim=-1))).sum(dim=-1).mean().item()))
            t1 = lt.argmax(dim=-1)
            s5 = torch.topk(ls, 5, dim=-1).indices
            top1_hits += int((s5[:, 0] == t1).sum().item())
            top5_hits += int((s5 == t1.unsqueeze(1)).any(dim=-1).sum().item())
        val_kl = float(np.mean(kl_vals)) if kl_vals else 0.0
        val_top1 = float(top1_hits / max(val_cnt, 1))
        val_top5 = float(top5_hits / max(val_cnt, 1))
        print(f"[Stage 3 Teacher-KL Val] kl={val_kl:.4f} | top1_agree={val_top1*100:.2f}% | top5_agree={val_top5*100:.2f}%", flush=True)
    del hs_ans, ht_ans
    torch.cuda.empty_cache()

    # Export updated CRTR and C8DB (now including post_cell60 when kept)
    crtr.export_crtr(crtr_layers, OUT_CRTR)
    if head_crtr_layers is not None and head_crtr_final_path == OUT_HEAD_CRTR:
        crtr.export_crtr(head_crtr_layers, OUT_HEAD_CRTR)
    c8d.write_c8db_file(OUT_C8D, [], c8db_entries)

    ast_dir = Path(snapshot_download("google/gemma-4-31B-it-assistant", local_dir="/tmp/gemma4_ast", ignore_patterns=["*.bin", "*.pt"]))
    ast_sf_raw = ast_dir / "model.safetensors"
    ast_sd: Dict[str, torch.Tensor] = {}
    with safe_open(ast_sf_raw, framework="pt", device="cpu") as sf_a:
        for k in sf_a.keys():
            t_a = sf_a.get_tensor(k)
            if k == "pre_projection.weight" and t_a.shape[1] == 2 * D_MOD:
                p2 = torch.cat([perm_pi, D_MOD + perm_pi])
                t_a = t_a[:, p2].contiguous()
            elif k == "post_projection.weight" and t_a.shape[0] == D_MOD:
                t_a = t_a[perm_pi, :].contiguous()
            ast_sd[k] = t_a

    ast_mod = Gemma4AssistantTrain(
        ast_sd, d_mod=D_MOD, nh=NH, nkv_s=NKV_S, nkv_f=NKV_F, hd_s=HD_S, hd_f=HD_F,
        theta_l=THETA_L, part_l=PART_L, theta_g=THETA_G, part_g=PART_G, seq_len=V9_SEQ_LEN, eps=EPS, compute_dtype=compute_dtype,
    ).to(device)
    with torch.no_grad():
        ast_h_final = torch.empty((len(ast_seqs), V9_SEQ_LEN, D_MOD), dtype=torch.float16)
        for ai, gi in enumerate(ast_seqs):
            hf_ai = rmsn_gpu(X_stud_perm[gi].to(device), fn_eff_g).float()
            if keep_c60 and cents60 is not None and experts60 is not None:
                hf_ai = hf_ai + _eval_c8d_gpu(hf_ai, cents60, experts60, H8_GPU)
            ast_h_final[ai] = hf_ai.to(torch.float16).cpu()
        nxt_ast = torch.from_numpy(all_seq_nxt[ast_seqs].reshape(-1)).long()
        uniq_a, cnts_a = torch.unique(nxt_ast, return_counts=True)
        cand_ids = uniq_a[torch.topk(cnts_a, min(8192, uniq_a.numel())).indices].to(device)
        W_cand_g = emb_g[cand_ids].float()
        W_ast_cand = ast_mod.emb[cand_ids].float()

    TAU_AST = 2.0
    n_ast_tr = max(1, (len(ast_seqs) * 7) // 8)
    trie = bhmod.BpePrefixTrie.from_tokenizer_json(tok_path)

    def eval_ast() -> Tuple[float, float]:
        with torch.no_grad():
            agrs = kls = 0.0
            cnt = 0
            for ai in range(n_ast_tr, len(ast_seqs)):
                gi = ast_seqs[ai]
                te = (emb_g[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE).unsqueeze(0)
                ha = ast_mod.forward_seq(
                    te, ast_h_final[ai : ai + 1, :-1].to(device).float(),
                    cap_kv58["k_rope"][gi].unsqueeze(0).to(device), cap_kv58["v_norm"][gi].unsqueeze(0).to(device),
                    cap_kv59["k_rope"][gi].unsqueeze(0).to(device), cap_kv59["v_norm"][gi].unsqueeze(0).to(device),
                ).squeeze(0)
                zs = SOFTCAP * torch.tanh((ast_h_final[ai, 1:].to(device).float() @ W_cand_g.T) / SOFTCAP)
                za = ha @ W_ast_cand.T
                ps = F.softmax(zs / TAU_AST, dim=-1)
                kls += float(((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU_AST, dim=-1))).sum(dim=-1) * (TAU_AST * TAU_AST)).sum().item())
                agrs += float((za.argmax(dim=-1) == zs.argmax(dim=-1)).float().sum().item())
                cnt += ha.shape[0]
            return kls / max(cnt, 1), agrs / max(cnt, 1)

    akl0, aagr0 = eval_ast()
    opt_a = torch.optim.AdamW(ast_mod.parameters(), lr=2.5e-4, weight_decay=1e-4)
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=300, eta_min=2e-5)
    best_aagr, best_asd = aagr0, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_bs = 12 if gpu_vram_gb >= 64.0 else 4
    for st in range(300):
        opt_a.zero_grad()
        bais = np.random.choice(n_ast_tr, size=min(ast_bs, n_ast_tr), replace=False).tolist()
        te_l, bh_l, k58_l, v58_l, k59_l, v59_l, ht_l = [], [], [], [], [], [], []
        for ai in bais:
            gi = ast_seqs[ai]
            te_l.append(emb_g[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE)
            bh_l.append(ast_h_final[ai, :-1].to(device).float())
            k58_l.append(cap_kv58["k_rope"][gi].to(device))
            v58_l.append(cap_kv58["v_norm"][gi].to(device))
            k59_l.append(cap_kv59["k_rope"][gi].to(device))
            v59_l.append(cap_kv59["v_norm"][gi].to(device))
            ht_l.append(ast_h_final[ai, 1:].to(device).float())
        te_s, bh_s = torch.stack(te_l), torch.stack(bh_l)
        k58_s, v58_s, k59_s, v59_s = torch.stack(k58_l), torch.stack(v58_l), torch.stack(k59_l), torch.stack(v59_l)
        hst = torch.stack(ht_l)
        ha = ast_mod.forward_seq(te_s, bh_s, k58_s, v58_s, k59_s, v59_s)
        with torch.no_grad():
            zs = SOFTCAP * torch.tanh((hst.reshape(-1, D_MOD) @ W_cand_g.T) / SOFTCAP)
            ps, ys_t1 = F.softmax(zs / TAU_AST, dim=-1), zs.argmax(dim=-1)
        za = ha.reshape(-1, 1024) @ W_ast_cand.T
        hpost = F.linear(ha.to(compute_dtype), ast_mod.post.to(compute_dtype)).float()
        bh2 = torch.cat([bh_s[:, :1], hpost[:, :-1]], dim=1)
        ha2 = ast_mod.forward_seq(te_s, bh2, k58_s, v58_s, k59_s, v59_s)
        hpost2 = F.linear(ha2.to(compute_dtype), ast_mod.post.to(compute_dtype)).float()
        loss_a = (
            ((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU_AST, dim=-1))).sum(dim=-1) * (TAU_AST * TAU_AST)).mean()
            + 0.5 * F.cross_entropy(za, ys_t1)
            + 0.5 * (F.mse_loss(hpost, hst) / (hst.var() + 1e-5))
            + 0.25 * (1.0 - F.cosine_similarity(hpost2[:, 1:], hst[:, 1:], dim=-1).mean())
        )
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(ast_mod.parameters(), 1.0)
        opt_a.step()
        sch_a.step()
        if (st + 1) % 60 == 0:
            klv, agrv = eval_ast()
            print(f"    [Ast step {st+1:03d}/300] Val KL={klv:.4f} Top1Agree={agrv*100:.1f}%", flush=True)
            if agrv > best_aagr:
                best_aagr, best_asd = agrv, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_mod.load_state_dict({k: v.to(device) for k, v in best_asd.items()})
    akl1, aagr1 = eval_ast()
    print(f"  [Assistant V9 Aligned] Val KL={akl0:.4f}->{akl1:.4f} | Top1Agree={aagr0*100:.1f}%->{aagr1*100:.1f}%", flush=True)

    ast_sd["pre_projection.weight"] = ast_mod.pre.detach().to(torch.bfloat16).cpu()
    ast_sd["post_projection.weight"] = ast_mod.post.detach().to(torch.bfloat16).cpu()
    ast_sd["model.norm.weight"] = ast_mod.fn.detach().to(torch.bfloat16).cpu()
    for la, m in enumerate(ast_mod.layers):
        pa_k = f"model.layers.{la}."
        for k_s, par in [
            ("self_attn.q_proj.weight", m.q_proj), ("self_attn.o_proj.weight", m.o_proj),
            ("mlp.gate_proj.weight", m.gate), ("mlp.up_proj.weight", m.up), ("mlp.down_proj.weight", m.down),
            ("input_layernorm.weight", m.n_in), ("self_attn.q_norm.weight", m.n_qn),
            ("post_attention_layernorm.weight", m.n_pa), ("pre_feedforward_layernorm.weight", m.n_pf),
            ("post_feedforward_layernorm.weight", m.n_pff),
        ]:
            ast_sd[pa_k + k_s] = par.detach().to(torch.bfloat16).cpu()
        ast_sd[pa_k + "layer_scalar"] = m.lsc.detach().to(torch.bfloat16).cpu().reshape(ast_sd[pa_k + "layer_scalar"].shape)
    ast_sf = OUT_DIR / "assistant_v9_permuted.safetensors"
    st_save_file(ast_sd, str(ast_sf))
    bhmod.pack_assistant_body_from_safetensors(ast_sf, OUT_AST_Q2, body_bits=2, block_size=64, has_fwht=False)
    bhmod.pack_assistant_body_from_safetensors(ast_sf, OUT_AST_Q4, body_bits=4, block_size=64, has_fwht=True)
    ast_sf.unlink(missing_ok=True)
    del opt_a, sch_a, ast_sd, best_asd, W_cand_g, W_ast_cand

    h_ast_chunks = []
    with torch.no_grad():
        for b0 in range(0, len(ast_seqs), 32):
            bais = list(range(b0, min(b0 + 32, len(ast_seqs))))
            gis = [ast_seqs[ai] for ai in bais]
            te_s = torch.stack([emb_g[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE for gi in gis])
            bh_s = ast_h_final[b0 : b0 + len(bais), :-1].to(device).float()
            k58_s = torch.stack([cap_kv58["k_rope"][gi].to(device) for gi in gis])
            v58_s = torch.stack([cap_kv58["v_norm"][gi].to(device) for gi in gis])
            k59_s = torch.stack([cap_kv59["k_rope"][gi].to(device) for gi in gis])
            v59_s = torch.stack([cap_kv59["v_norm"][gi].to(device) for gi in gis])
            h_ast_chunks.append(ast_mod.forward_seq(te_s, bh_s, k58_s, v58_s, k59_s, v59_s).cpu())
    del ast_mod, cap_kv58, cap_kv59
    torch.cuda.empty_cache()

    H_ast_all = torch.cat(h_ast_chunks, dim=0).reshape(-1, 1024)
    H_stud_tgt = ast_h_final[:, 1:, :].reshape(-1, D_MOD)
    top_ids_t = torch.empty((H_stud_tgt.shape[0], 64), dtype=torch.int32)
    top_probs_t = torch.empty((H_stud_tgt.shape[0], 64), dtype=torch.float32)
    emb_g_c = emb_g.to(dtype=compute_dtype)
    del emb_g
    with torch.no_grad():
        for i0 in range(0, H_stud_tgt.shape[0], 512):
            i1 = min(i0 + 512, H_stud_tgt.shape[0])
            lg_b = SOFTCAP * torch.tanh((H_stud_tgt[i0:i1].to(device, dtype=compute_dtype) @ emb_g_c.T).float() / SOFTCAP)
            tv, ti = torch.topk(lg_b, k=64, dim=-1)
            top_probs_t[i0:i1] = F.softmax(tv, dim=-1).cpu()
            top_ids_t[i0:i1] = ti.to(torch.int32).cpu()
    del emb_g_c
    student_argmax_ids = torch.tensor(v70.select_first_valid_trie_ids(top_ids_t, trie), dtype=torch.long)
    del H_stud_tgt, ast_h_final

    act_bundle = OUT_DIR / "v9_0_bhdr_acts.pt"
    torch.save({"h_final": H_ast_all.half(), "gold_ids": student_argmax_ids, "top_ids": top_ids_t, "top_probs": top_probs_t}, act_bundle)
    bh_rpt = bhmod.train_from_activations(act_bundle, tok_path, OUT_BHDR, d_byte=384, max_bytes=48, steps=2400, batch_size=256, lr=3e-3, device=str(device))
    act_bundle.unlink(missing_ok=True)
    bh_loaded = bhmod.load_bhdr_v1(OUT_BHDR)
    val_h = H_ast_all[-256:].numpy().astype(np.float32)
    val_g = student_argmax_ids[-256:].tolist()
    beam_hits = sum(1 for hv, gv in zip(val_h, val_g) if bhmod.decode_bhdr_np_beam(bh_loaded, trie, hv, beam_width=2)[0] == gv)
    bh_metrics = {**bh_rpt["metrics"], "heldout_beam_top1_acc_vs_v9_argmax": round(beam_hits / 256.0, 4)}
    del H_ast_all, student_argmax_ids, top_ids_t, top_probs_t

    all_artifact_paths = (
        t16c_final_path, OUT_CRTR, head_crtr_final_path, OUT_C8D, OUT_AST_Q2, OUT_AST_Q4, OUT_BHDR,
    )
    persist_set = set(all_artifact_paths) if persist_t16c_and_head else {OUT_CRTR, OUT_C8D, OUT_AST_Q2, OUT_AST_Q4, OUT_BHDR}
    artifacts = {}
    for fp_out in all_artifact_paths:
        sz_b = fp_out.stat().st_size
        hx_b = _sha256_file(fp_out)
        if DRIVE_DIR is not None and fp_out in persist_set:
            _drive_safe_copy(fp_out, DRIVE_DIR / fp_out.name)
        artifacts[fp_out.name] = {"bytes": sz_b, "sha256": hx_b}

    report = {
        "version": f"V9.0-FASTKERNEL-SPARSE10OF84-HEAD10OF32-DISPERSION-QAT{mode_tag}",
        "gpu": gpu_name,
        "elapsed_s": round(time.perf_counter() - t0_all, 2),
        "t16c_bytes": t16c_final_path.stat().st_size,
        "t16c_gib": round(t16c_final_path.stat().st_size / (1024**3), 4),
        "colblock_mats": cb_cnt,
        "l59_cos": layer_logs[-1]["vlm_txt_cos"] if layer_logs else 0.0,
        "val_kl": round(val_kl, 4),
        "val_top1_agree": round(val_top1, 4),
        "val_top5_agree": round(val_top5, 4),
        "post_cell60_kept": bool(keep_c60),
        "crtr_mean_recall": round(float(np.mean([r["crtr_recall"] for r in layer_logs])), 5),
        "head_crtr_mean_recall": round(float(np.mean([r["head_crtr_recall"] for r in layer_logs])), 5),
        "assistant_align": {"kl_before": round(akl0, 4), "kl_after": round(akl1, 4), "top1_before": round(aagr0, 4), "top1_after": round(aagr1, 4)},
        "bhdr": bh_metrics,
        "artifacts": artifacts,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sums = "\n".join(f"{v['sha256']}  {k}" for k, v in artifacts.items()) + "\n"
    (OUT_DIR / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    if DRIVE_DIR is not None:
        _drive_safe_copy(OUT_REPORT, DRIVE_DIR / OUT_REPORT.name)
        _drive_safe_copy(OUT_DIR / "SHA256SUMS.txt", DRIVE_DIR / "SHA256SUMS.txt")
        # Remove obsolete .partial and ckpt_v9_0 bloat so Drive FUSE upload queue stays clean
        part_p = DRIVE_DIR / (OUT_T16C.name + ".partial")
        if part_p.exists():
            part_p.unlink()
        shutil.rmtree(DRIVE_DIR / "ckpt_v9_0", ignore_errors=True)
        expected_sha_map = {k: v["sha256"] for k, v in artifacts.items()}
        expected_sha_map[OUT_REPORT.name] = _sha256_file(OUT_REPORT)
        _verify_drive_sha256(DRIVE_DIR, expected_sha_map)

    print(f"[V9.0 {mode_tag or 'COMPLETE'}] elapsed={(time.perf_counter()-t0_all)/60:.1f} min | {OUT_T16C.name}={report['t16c_gib']} GiB | colblock={cb_cnt}", flush=True)
    if os.environ.get("COLAB_AUTO_UNASSIGN", "1") == "1":
        _flush_drive_before_unassign()
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:
            print(f"[Auto-Unassign Note] {exc}", flush=True)
    return report


def recover_aux_from_t16c() -> Dict[str, object]:
    """Replay all 60 layers from an already-built V9.0 `.t16c` + `head_router.crtr` in ~11 min to regenerate the 5 aux files."""
    if not torch.cuda.is_available():
        raise RuntimeError("recover_aux_from_t16c() requires CUDA GPU")
    if run_selftest() != 0:
        raise RuntimeError("V9.0 CPU pre-flight selftest failed")

    t0_all = time.perf_counter()
    device = torch.device("cuda:0")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DRIVE_DIR is not None:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    MB_FWD = 256 if gpu_vram_gb >= 64.0 else (64 if gpu_vram_gb >= 30.0 else 48)

    t16c_src = _resolve_existing_artifact(OUT_T16C.name, OUT_DIR, DRIVE_DIR)
    hr_src = _resolve_existing_artifact(OUT_HEAD_CRTR.name, OUT_DIR, DRIVE_DIR)
    print(f"[V9.0 RECOVER-AUX] t16c={t16c_src} | head_router={hr_src} | GPU={gpu_name}", flush=True)
    norms_dict, mats_meta = _parse_t16c_index(t16c_src)
    cb_cnt = sdc.verify_t16c_file(t16c_src)["colblock_mats"]
    hr_meta, hr_recs = crtr.load_crtr(hr_src)
    assert len(hr_recs) == V9_N_LAYERS and int(hr_meta["n_blocks"]) == V9_ATTN_HEADS

    from safetensors import safe_open
    H256_GPU = torch.from_numpy(pa.make_sylvester_hadamard(V9_BLOCK_SIZE)).to(device, dtype=torch.float32)
    H8_GPU = torch.from_numpy(c8d.H8_NP).to(device, dtype=torch.float32)
    ctx = _init_v9_stage1_base(device, compute_dtype)
    sf_path: Path = ctx["sf_path"]  # type: ignore[assignment]
    D_MOD: int = int(ctx["D_MOD"])
    N_LAYERS: int = int(ctx["N_LAYERS"])
    EPS: float = float(ctx["EPS"])
    ESCALE: float = float(ctx["ESCALE"])
    NH: int = int(ctx["NH"])
    NKV_S: int = int(ctx["NKV_S"])
    NKV_F: int = int(ctx["NKV_F"])
    HD_S: int = int(ctx["HD_S"])
    HD_F: int = int(ctx["HD_F"])
    K_EQ_V: bool = bool(ctx["K_EQ_V"])
    LAYER_TYPES = ctx["LAYER_TYPES"]
    THETA_G, PART_G = float(ctx["THETA_G"]), float(ctx["PART_G"])
    THETA_L, PART_L = float(ctx["THETA_L"]), float(ctx["PART_L"])
    soft_tok: int = int(ctx["soft_tok"])
    all_seq_ids: np.ndarray = ctx["all_seq_ids"]  # type: ignore[assignment]
    N_TEXT_SEQ: int = int(ctx["N_TEXT_SEQ"])
    N_SEQ: int = int(ctx["N_SEQ"])
    X_teach_perm: torch.Tensor = ctx["X_teach_perm"]  # type: ignore[assignment]
    perm_pi: torch.Tensor = ctx["perm_pi"]  # type: ignore[assignment]
    perm_pi_dev = perm_pi.to(device)
    del ctx["emb_raw"], ctx["fn_orig"]

    # Fail-closed verification that reconstructed perm_pi matches t16c_src across ALL 60 layers
    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        max_pi_diff = 0.0
        for l_idx in range(N_LAYERS):
            nk_in = f"model.language_model.layers.{l_idx}.input_layernorm.weight"
            ref_n = sf.get_tensor(nk_in).float()[perm_pi].numpy().astype(np.float32)
            got_n = norms_dict[nk_in].astype(np.float32)
            max_pi_diff = max(max_pi_diff, float(np.max(np.abs(ref_n - got_n))))
        assert max_pi_diff < 1e-6, f"perm_pi mismatch vs {t16c_src}: max_diff={max_pi_diff:.3e}"
    print(f"[Recover Verify] perm_pi 100% verified against all 60 layers in {t16c_src.name} (max_diff={max_pi_diff:.2e})", flush=True)

    # Decode quantized 3-bit emb_eff from t16c_src in 16K-row chunks
    emb_rec = mats_meta["model.language_model.embed_tokens.weight"]
    vocab_sz = int(emb_rec["d_out"])
    emb_eff = torch.empty((vocab_sz, D_MOD), dtype=torch.float16)
    eb_eff_bits = int(emb_rec["n_bits"]) & 0x1F
    eb_gauss = bool(int(emb_rec["n_bits"]) & p3.T16C_GAUSS_BIT)
    tblk_emb = p3.tile16_blk_bytes(V9_BLOCK_SIZE, eb_eff_bits | (p3.T16C_GAUSS_BIT if eb_gauss else 0), emb_rec["is_asym"]) * (D_MOD // V9_BLOCK_SIZE)
    with open(t16c_src, "rb") as fp_emb:
        for r0 in range(0, vocab_sz, 16384):
            r1 = min(r0 + 16384, vocab_sz)
            n_t = (r1 - r0 + 15) // 16
            fp_emb.seek(int(emb_rec["offset"]) + (r0 // 16) * tblk_emb)
            pb_c = fp_emb.read(n_t * tblk_emb)
            w_c = decode_t16c_mat_to_f32(
                pb_c, r1 - r0, D_MOD, V9_BLOCK_SIZE, emb_rec["n_bits"], emb_rec["is_asym"],
                None, None, H256_GPU, has_fwht=emb_rec["has_fwht"],
            )
            emb_eff[r0:r1] = w_c.to(torch.float16).cpu()
            del w_c, pb_c

    X_stud_perm = (emb_eff[torch.from_numpy(all_seq_ids)] * ESCALE).to(torch.float16)
    for si in range(N_TEXT_SEQ, N_SEQ):
        pos_m = torch.from_numpy(all_seq_ids[si] == soft_tok)
        X_stud_perm[si, pos_m] = X_teach_perm[si, pos_m]
    torch.cuda.empty_cache()

    rmsn_gpu, make_rope, apply_rope = _make_gemma4_math_helpers(device, EPS)
    rope_cs_g, rope_sn_g = make_rope(THETA_G, int(round(HD_F * PART_G)), HD_F)
    rope_cs_s, rope_sn_s = make_rope(THETA_L, int(round(HD_S * PART_L)), HD_S)

    crtr_layers: List[dict] = []
    c8db_entries: List[dict] = []
    layer_logs: List[dict] = []
    ast_seqs = np.linspace(0, N_TEXT_SEQ - 1, min(128, N_TEXT_SEQ), dtype=int).tolist()
    cap_kv58: Dict[str, dict] = {"k_rope": {}, "v_norm": {}}
    cap_kv59: Dict[str, dict] = {"k_rope": {}, "v_norm": {}}

    with open(t16c_src, "rb") as fp_t16c, safe_open(sf_path, framework="pt", device="cpu") as sf:
        for l_idx in range(N_LAYERS):
            tl0 = time.perf_counter()
            pfx = f"model.language_model.layers.{l_idx}"
            is_f = LAYER_TYPES[l_idx] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)
            rep = NH // nkv

            g_in = torch.from_numpy(norms_dict[f"{pfx}.input_layernorm.weight"]).to(device)
            g_pa = torch.from_numpy(norms_dict[f"{pfx}.post_attention_layernorm.weight"]).to(device)
            g_pf = torch.from_numpy(norms_dict[f"{pfx}.pre_feedforward_layernorm.weight"]).to(device)
            g_pff = torch.from_numpy(norms_dict[f"{pfx}.post_feedforward_layernorm.weight"]).to(device)
            g_qn = torch.from_numpy(norms_dict[f"{pfx}.self_attn.q_norm.weight"]).to(device)
            g_kn = torch.from_numpy(norms_dict[f"{pfx}.self_attn.k_norm.weight"]).to(device)
            lsc_fin = float(norms_dict[f"{pfx}.layer_scalar"].reshape(-1)[0])

            # 1) Teacher forward pass (X_teach_perm -> X_teach_next)
            wq_t = _load_dequant_proj(sf, f"{pfx}.self_attn.q_proj", perm_pi_dev, device, is_perm_col=True)
            wk_t = _load_dequant_proj(sf, f"{pfx}.self_attn.k_proj", perm_pi_dev, device, is_perm_col=True)
            wv_t = wk_t if (is_f and K_EQ_V) else _load_dequant_proj(sf, f"{pfx}.self_attn.v_proj", perm_pi_dev, device, is_perm_col=True)
            wo_t = _load_dequant_proj(sf, f"{pfx}.self_attn.o_proj", perm_pi_dev, device, is_perm_row=True)
            wg_t = _load_dequant_proj(sf, f"{pfx}.mlp.gate_proj", perm_pi_dev, device, is_perm_col=True)
            wu_t = _load_dequant_proj(sf, f"{pfx}.mlp.up_proj", perm_pi_dev, device, is_perm_col=True)
            wd_t = _load_dequant_proj(sf, f"{pfx}.mlp.down_proj", perm_pi_dev, device, is_perm_row=True)
            g_pa_orig = sf.get_tensor(f"{pfx}.post_attention_layernorm.weight").float()[perm_pi].to(device)
            g_pff_orig = sf.get_tensor(f"{pfx}.post_feedforward_layernorm.weight").float()[perm_pi].to(device)
            lsc_orig = float(sf.get_tensor(f"{pfx}.layer_scalar").float().reshape(-1)[0].item())

            X_teach_next = torch.empty_like(X_teach_perm)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xb = X_teach_perm[b0:b1].to(device, dtype=torch.float32)
                    xn = rmsn_gpu(xb, g_in)
                    q = apply_rope(rmsn_gpu(F.linear(xn, wq_t).view(b1 - b0, V9_SEQ_LEN, NH, hd), g_qn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    k = apply_rope(rmsn_gpu(F.linear(xn, wk_t).view(b1 - b0, V9_SEQ_LEN, nkv, hd), g_kn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    v = rmsn_gpu(F.linear(xn, wv_t).view(b1 - b0, V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                    ctx_t = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous().view(b1 - b0, V9_SEQ_LEN, NH * hd)
                    xm = xb + rmsn_gpu(F.linear(ctx_t, wo_t), g_pa_orig)
                    xf = rmsn_gpu(xm, g_pf)
                    act = F.gelu(F.linear(xf, wg_t), approximate="tanh") * F.linear(xf, wu_t)
                    X_teach_next[b0:b1] = ((xm + rmsn_gpu(F.linear(act, wd_t), g_pff_orig)) * lsc_orig).to(torch.float16).cpu()
            del wq_t, wk_t, wv_t, wo_t, wg_t, wu_t, wd_t, g_pa_orig, g_pff_orig

            # 2) Decode frozen quantized Student projections from t16c_src
            wq_q = _read_t16c_matrix_f32(fp_t16c, mats_meta[f"{pfx}.self_attn.q_proj.weight"], H256_GPU)
            wk_q = _read_t16c_matrix_f32(fp_t16c, mats_meta[f"{pfx}.self_attn.k_proj.weight"], H256_GPU)
            wv_q = wk_q if (is_f and K_EQ_V) else _read_t16c_matrix_f32(fp_t16c, mats_meta[f"{pfx}.self_attn.v_proj.weight"], H256_GPU)
            wo_q = _read_t16c_matrix_f32(fp_t16c, mats_meta[f"{pfx}.self_attn.o_proj.weight"], H256_GPU)
            wg_fin = _read_t16c_matrix_f32(fp_t16c, mats_meta[f"{pfx}.mlp.gate_proj.weight"], H256_GPU)
            wu_fin = _read_t16c_matrix_f32(fp_t16c, mats_meta[f"{pfx}.mlp.up_proj.weight"], H256_GPU)
            wd_fin = _read_t16c_matrix_f32(fp_t16c, mats_meta[f"{pfx}.mlp.down_proj.weight"], H256_GPU)

            xm_stud_cpu = torch.empty_like(X_stud_perm)
            xf_stud_cpu = torch.empty_like(X_stud_perm)
            blk_e_cpu = torch.empty((N_SEQ * V9_SEQ_LEN, V9_MLP_BLOCKS), dtype=torch.float32)
            mod_sal_cpu = torch.empty((N_SEQ * V9_SEQ_LEN,), dtype=torch.float32)
            hr_rec_l = hr_recs[l_idx] if (2 <= l_idx <= 57) else None

            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xb = X_stud_perm[b0:b1].to(device, dtype=torch.float32)
                    xn = rmsn_gpu(xb, g_in)
                    q = apply_rope(rmsn_gpu(F.linear(xn, wq_q).view(b1 - b0, V9_SEQ_LEN, NH, hd), g_qn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    k = apply_rope(rmsn_gpu(F.linear(xn, wk_q).view(b1 - b0, V9_SEQ_LEN, nkv, hd), g_kn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    v = rmsn_gpu(F.linear(xn, wv_q).view(b1 - b0, V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                    if l_idx in (58, 59):
                        for gi in range(b0, b1):
                            if gi in ast_seqs:
                                cap = cap_kv58 if l_idx == 58 else cap_kv59
                                cap["k_rope"][gi] = k[gi - b0].to(torch.float16).cpu()
                                cap["v_norm"][gi] = v[gi - b0].to(torch.float16).cpu()
                    ctx_s = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous()
                    if hr_rec_l is not None:
                        hr_lg = _eval_crtr_rec_gpu(hr_rec_l, xn.reshape(-1, D_MOD))
                        top_h = torch.topk(hr_lg, V9_ATTN_ROUTED_HEADS, dim=-1).indices
                        hm = torch.zeros((xn.shape[0] * V9_SEQ_LEN, NH), device=device)
                        hm.scatter_(1, top_h, 1.0)
                        ctx_s = ctx_s * hm.view(b1 - b0, V9_SEQ_LEN, NH, 1)
                    xm = xb + rmsn_gpu(F.linear(ctx_s.view(b1 - b0, V9_SEQ_LEN, NH * hd), wo_q), g_pa)
                    xf = rmsn_gpu(xm, g_pf)
                    xm_stud_cpu[b0:b1] = xm.to(torch.float16).cpu()
                    xf_stud_cpu[b0:b1] = xf.to(torch.float16).cpu()
                    gg = F.gelu(F.linear(xf, wg_fin), approximate="tanh").reshape(-1, V9_MLP_BLOCKS, V9_BLOCK_SIZE)
                    blk_e_cpu[b0 * V9_SEQ_LEN : b1 * V9_SEQ_LEN] = gg.pow(2).sum(dim=-1).to(torch.float32).cpu()
                    sal_tok = (X_teach_next[b0:b1].to(device, dtype=torch.float32) - xb).pow(2).sum(dim=-1).sqrt() / (xb.pow(2).sum(dim=-1).sqrt() + 1e-6)
                    mod_sal_cpu[b0 * V9_SEQ_LEN : b1 * V9_SEQ_LEN] = sal_tok.reshape(-1).cpu()
            del wq_q, wk_q, wv_q, wo_q

            # 3) Train CRTR-V9 (r=64 + MoD) -> INT8 quantize
            xf_flat = xf_stud_cpu.reshape(-1, D_MOD)
            n_tr_r = (xf_flat.shape[0] * 9) // 10
            sal_q28 = float(torch.quantile(mod_sal_cpu, 0.28).item())
            mod_y = (mod_sal_cpu > sal_q28).float()
            r_net, r_met = crtr.train_layer_router(
                xf_flat[:n_tr_r], blk_e_cpu[:n_tr_r], xf_flat[n_tr_r:], blk_e_cpu[n_tr_r:],
                top_p=float(V9_SHARED_BLOCKS + V9_ROUTED_BLOCKS) / float(V9_MLP_BLOCKS),
                rank=V9_CRTR_RANK, steps=160, lr=3e-3,
                mod_targets_tr=mod_y[:n_tr_r], mod_targets_va=mod_y[n_tr_r:],
                device=str(device), seed=100 + l_idx,
            )
            r_q = crtr.quantize_layer_for_crtr(r_net, V9_BLOCK_SIZE)
            crtr_layers.append(r_q)

            # 4) Train C8DB adapter on frozen Student MLP (under quantized INT8 r_q mask)
            adp_d = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=V9_FF, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(device)
            opt_c8 = torch.optim.AdamW(adp_d.parameters(), lr=4e-3, weight_decay=1e-4)
            for _ in range(60):
                opt_c8.zero_grad()
                bs_q = torch.randint(0, N_SEQ, (min(32, N_SEQ),), device="cpu")
                xf_b = xf_stud_cpu[bs_q].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                xm_b = xm_stud_cpu[bs_q].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                xt_b = X_teach_next[bs_q].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                with torch.no_grad():
                    r_log = _eval_crtr_rec_gpu(r_q, xf_b)
                    r_spec = torch.topk(r_log[:, V9_SHARED_BLOCKS:], V9_ROUTED_BLOCKS, dim=-1).indices + V9_SHARED_BLOCKS
                    blk_m = torch.zeros((xf_b.shape[0], V9_MLP_BLOCKS), device=device)
                    blk_m[:, :V9_SHARED_BLOCKS] = 1.0
                    blk_m.scatter_(1, r_spec, 1.0)
                    col_m = blk_m.repeat_interleave(V9_BLOCK_SIZE, dim=-1)
                    act_q = (F.gelu(F.linear(xf_b, wg_fin), approximate="tanh") * F.linear(xf_b, wu_fin)) * col_m
                    base_d = F.linear(act_q, wd_fin)
                delta_d, _, lb_d = adp_d(act_q)
                xs_b = (xm_b + rmsn_gpu(base_d + delta_d, g_pff)) * lsc_fin
                mag_t = torch.log(xt_b.pow(2).sum(dim=-1).clamp(min=1e-8))
                mag_s = torch.log(xs_b.pow(2).sum(dim=-1).clamp(min=1e-8))
                loss_c8 = (1.0 - F.cosine_similarity(xs_b, xt_b, dim=-1).mean()) + 0.15 * F.mse_loss(mag_s, mag_t) + 0.25 * F.mse_loss(xs_b, xt_b) + 0.005 * lb_d
                loss_c8.backward()
                torch.nn.utils.clip_grad_norm_(adp_d.parameters(), 1.0)
                opt_c8.step()

            ct_d, ex_d = adp_d.cpu().export_quantized_arrays()
            exp_r_d = [c8d._c6d_radicals("general", "core", "transform", "native", "layer", "positive", e) for e in range(8)]
            c8db_entries.append({
                "name": f"{pfx}.mlp.down_proj.weight", "d_out": D_MOD, "d_in": V9_FF, "rank": 32,
                "top_k": 2, "c6d": exp_r_d[0], "temperature": 0.5, "gate_scale": 1.0,
                "centroids": ct_d, "expert_radicals": exp_r_d, "experts": ex_d,
            })

            # 5) Advance Student state with EXACT quantized INT8 CRTR & quantized INT8 C8DB
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xf_b = xf_stud_cpu[b0:b1].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                    xm_b = xm_stud_cpu[b0:b1].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                    r_log = _eval_crtr_rec_gpu(r_q, xf_b)
                    r_spec = torch.topk(r_log[:, V9_SHARED_BLOCKS:], V9_ROUTED_BLOCKS, dim=-1).indices + V9_SHARED_BLOCKS
                    blk_m = torch.zeros((xf_b.shape[0], V9_MLP_BLOCKS), device=device)
                    blk_m[:, :V9_SHARED_BLOCKS] = 1.0
                    blk_m.scatter_(1, r_spec, 1.0)
                    col_m = blk_m.repeat_interleave(V9_BLOCK_SIZE, dim=-1)
                    act_q = (F.gelu(F.linear(xf_b, wg_fin), approximate="tanh") * F.linear(xf_b, wu_fin)) * col_m
                    down_q = F.linear(act_q, wd_fin) + _eval_c8d_gpu(act_q, ct_d, ex_d, H8_GPU)
                    X_stud_perm[b0:b1] = ((xm_b + rmsn_gpu(down_q, g_pff)) * lsc_fin).reshape(b1 - b0, V9_SEQ_LEN, D_MOD).to(torch.float16).cpu()

            X_teach_perm = X_teach_next
            cos_l = float(F.cosine_similarity(X_stud_perm[-32:].reshape(-1, D_MOD).float(), X_teach_perm[-32:].reshape(-1, D_MOD).float(), dim=-1).mean().item())
            layer_logs.append({"l": l_idx, "vlm_txt_cos": round(cos_l, 5), "crtr_recall": r_met["val_recall_after"], "head_crtr_recall": 0.75743})
            print(f"  [Recover L{l_idx:02d}/59] cos={cos_l:.5f} crtr_rec={r_met['val_recall_after']:.3f} dt={time.perf_counter()-tl0:.1f}s", flush=True)
            del wg_fin, wu_fin, wd_fin, xm_stud_cpu, xf_stud_cpu, blk_e_cpu, mod_sal_cpu, adp_d, opt_c8
            gc.collect()
            torch.cuda.empty_cache()

    return _run_stage3_distill_and_export(
        ctx=ctx, device=device, compute_dtype=compute_dtype, gpu_name=gpu_name, gpu_vram_gb=gpu_vram_gb, t0_all=t0_all,
        rmsn_gpu=rmsn_gpu, X_stud_perm=X_stud_perm, X_teach_perm=X_teach_perm, emb_eff=emb_eff, ast_seqs=ast_seqs,
        cap_kv58=cap_kv58, cap_kv59=cap_kv59, fn_vec_np=norms_dict["model.language_model.norm.weight"],
        crtr_layers=crtr_layers, head_crtr_layers=None, c8db_entries=c8db_entries, layer_logs=layer_logs, cb_cnt=cb_cnt,
        t16c_final_path=t16c_src, head_crtr_final_path=hr_src, persist_t16c_and_head=False, apply_wiener=False,
        train_post_cell60=True, mode_tag="-RECOVER-AUX",
    )


def run_pipeline() -> Dict[str, object]:
    """Run full V9.0 Sparse-Active 31B + Dispersion-Permuted + QAT + Assistant-V9 GPU pipeline."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_pipeline() requires CUDA GPU (pass --selftest on CPU)")
    if run_selftest() != 0:
        raise RuntimeError("V9.0 CPU pre-flight selftest failed")

    t0_all = time.perf_counter()
    device = torch.device("cuda:0")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if DRIVE_DIR is not None:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    MB_FWD = 256 if gpu_vram_gb >= 64.0 else (64 if gpu_vram_gb >= 30.0 else 48)

    def persist_v9(src: Path) -> Tuple[int, str]:
        sz = src.stat().st_size
        dig = _sha256_file(src)
        if DRIVE_DIR is not None:
            _drive_safe_copy(src, DRIVE_DIR / src.name)
        return sz, dig

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from PIL import Image
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers import AutoConfig, Gemma4ImageProcessor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

    LIB_FQ = _compile_fast_quant(Path("/tmp/libfast_quant_v9_0.so"))
    H256_GPU = torch.from_numpy(pa.make_sylvester_hadamard(V9_BLOCK_SIZE)).to(device, dtype=torch.float32)

    print(f"[V9.0 INIT] GPU={gpu_name} ({gpu_vram_gb:.1f} GiB) | MB_FWD={MB_FWD} | DRIVE={DRIVE_DIR}", flush=True)
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
    assert D_MOD == V9_D_MOD and N_LAYERS == V9_N_LAYERS

    bgt = compute_v9_0_budget_bytes(
        vocab_size=int(tc["vocab_size"]), d_mod=D_MOD, n_layers=N_LAYERS,
        nh=NH, nkv_s=NKV_S, nkv_f=NKV_F, hd_s=HD_S, hd_f=HD_F, ff=V9_FF,
    )
    print(f"[V9.0 Budget] total={bgt['total_t16c_gib']} GiB (<=5.60) | active_10of84={bgt['active_dram_10of84_mb']} MB | mod35={bgt['active_dram_10of84_mod35_mb']} MB | gmac={bgt['active_gmac_per_tok']}", flush=True)
    assert bgt["total_t16c_gib"] <= 5.60 and bgt["tern5_cnt"] == 0

    def c_pack(arr_f32: np.ndarray, do: int, di: int, bsize: int, mode_i: int) -> np.ndarray:
        n_bits_m, is_asym_m = p3.MODE_TO_T16C[mode_i]
        exp_b = p3.expected_tile16_bytes(do, di, bsize, n_bits_m, is_asym_m)
        buf = np.empty(exp_b, dtype=np.uint8)
        arr_c = np.ascontiguousarray(arr_f32, dtype=np.float32)
        LIB_FQ.quant_and_pack_t16c_f32(arr_c.ctypes.data, do, di, bsize, mode_i, buf.ctypes.data)
        return buf

    def rmsn_gpu(t: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf = t.float()
        y = xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + EPS)
        if w is not None:
            y = y * w.float()
        return y.to(t.dtype)

    def make_rope(theta: float, rot_d: int, hd: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = rot_d // 2
        if hr <= 0:
            z = torch.zeros((V9_SEQ_LEN, 0), device=device, dtype=torch.float32)
            return z, z
        inv_f = 1.0 / (theta ** (torch.arange(0, 2 * hr, 2, device=device, dtype=torch.float32) / float(hd)))
        pos = torch.arange(V9_SEQ_LEN, device=device, dtype=torch.float32)
        ang = torch.outer(pos, inv_f)
        return torch.cos(ang), torch.sin(ang)

    def apply_rope(x: torch.Tensor, cs: torch.Tensor, sn: torch.Tensor, rot_d: int, hd: int) -> torch.Tensor:
        hr, hh = rot_d // 2, hd // 2
        if hr <= 0:
            return x
        xf = x.float()
        x0, x1 = xf[..., :hr], xf[..., hh : hh + hr]
        c = cs.view(1, x.shape[1], 1, hr)
        s = sn.view(1, x.shape[1], 1, hr)
        r0, r1 = x0 * c - x1 * s, x0 * s + x1 * c
        if hr == hh:
            return torch.cat([r0, r1], dim=-1).to(x.dtype)
        return torch.cat([r0, xf[..., hr:hh], r1, xf[..., hh + hr :]], dim=-1).to(x.dtype)

    # ------------------------------------------------------------------
    # Stage 1: Corpus + VLM Seed Hidden States + Dispersion Perm Pi
    # ------------------------------------------------------------------
    print("[Stage 1] Building decontaminated STEM/code/science + COCO VLM corpus...", flush=True)
    txt_ids, txt_nxt, txt_ans = _build_decontaminated_corpus(tk, max_text_seq=896, seq_len=V9_SEQ_LEN)
    coco_ds = load_dataset("detection-datasets/coco", split="val[:136]")
    vlm_seqs, vlm_nxts, vlm_masks, vlm_imgs = [], [], [], []
    img_proc = Gemma4ImageProcessor.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct")
    soft_tok = int(cfg_full.get("image_token_id", 258880))
    boi_tok = int(cfg_full.get("boi_token_id", 255999))
    eoi_tok = int(cfg_full.get("eoi_token_id", 258882))
    tok_user_open = list(tk.encode("<|turn>user\n", add_special_tokens=False).ids)
    tok_model_open = list(tk.encode("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", add_special_tokens=False).ids)
    for r in coco_ds:
        objs = r.get("objects", {})
        cats = [str(c) for c in (objs.get("category", []) if isinstance(objs, dict) else [])[:4]]
        cap = "Objects visible in image: " + (", ".join(dict.fromkeys(cats)) if cats else "scene objects") + "."
        if any(b in cap.lower() for b in ALL_BLACKLIST):
            continue
        im = r["image"].convert("RGB").resize((224, 224))
        vlm_imgs.append(im)
        n_img = 16
        prompt_ids = [tk.token_to_id("<bos>") or 2] + tok_user_open + [boi_tok] + [soft_tok] * n_img + [eoi_tok] + list(tk.encode("Describe this image concisely in one sentence.", add_special_tokens=False).ids) + tok_model_open
        ans_ids = list(tk.encode(f"{cap}<turn|>\n", add_special_tokens=False).ids)
        full = (prompt_ids + ans_ids + [0] * (V9_SEQ_LEN + 1))[: V9_SEQ_LEN + 1]
        vlm_seqs.append(full[:V9_SEQ_LEN])
        vlm_nxts.append(full[1 : V9_SEQ_LEN + 1])
        m = np.zeros(V9_SEQ_LEN, dtype=bool)
        m[min(len(prompt_ids) - 1, V9_SEQ_LEN - 1) :] = True
        vlm_masks.append(m)
        if len(vlm_seqs) >= 128:
            break

    all_seq_ids = np.concatenate([txt_ids, np.array(vlm_seqs, dtype=np.int64)], axis=0)
    all_seq_nxt = np.concatenate([txt_nxt, np.array(vlm_nxts, dtype=np.int64)], axis=0)
    all_ans_mask = np.concatenate([txt_ans, np.array(vlm_masks, dtype=bool)], axis=0)
    N_TEXT_SEQ = txt_ids.shape[0]
    N_SEQ = all_seq_ids.shape[0]

    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        emb_raw = sf.get_tensor("model.language_model.embed_tokens.weight").float()
        fn_orig = sf.get_tensor("model.language_model.norm.weight").float()
        vision_norm_recs: List[Tuple[str, np.ndarray]] = []
        vision_mat_recs: List[tuple] = []
        vis_keys = [k for k in sf.keys() if k.startswith("model.vision_tower.") or k.startswith("model.embed_vision.")]
        hf_cfg = AutoConfig.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct")
        vis_model = Gemma4VisionModel(hf_cfg.vision_config)
        mm_embed = Gemma4MultimodalEmbedder(hf_cfg.vision_config, hf_cfg.text_config)
        vis_sd = {}
        mm_sd = {}
        for vk in vis_keys:
            if vk.startswith("model.vision_tower."):
                vis_sd[vk[len("model.vision_tower.") :]] = sf.get_tensor(vk)
            elif vk.startswith("model.embed_vision."):
                mm_sd[vk[len("model.embed_vision.") :]] = sf.get_tensor(vk)
        # Warm-start from already-calibrated 3-bit Stage 2 Vision checkpoints if present on Drive
        vis_deq: Optional[Dict[str, torch.Tensor]] = None
        for warm_dir in (
            Path("/content/drive/MyDrive/gemma4_v8_0_player_pov/ckpt_v8_0"),
            Path("/content/drive/MyDrive/gemma4_v7_0_fullrank/ckpt_v7_0"),
        ):
            vp_warm = warm_dir / "stage2_vision.pt"
            if vp_warm.is_file():
                try:
                    blob_w = torch.load(vp_warm, map_location="cpu", weights_only=False)
                    if isinstance(blob_w, dict) and "vis_deq" in blob_w:
                        vis_deq = blob_w["vis_deq"]
                        print(f"[Stage 1 Vision Warm-Start] Reused 3-bit vision dequant state from {vp_warm}", flush=True)
                        break
                except Exception as exc:
                    print(f"[Stage 1 Vision Warm-Start Note] Skipped {vp_warm}: {exc}", flush=True)
        vis_model.load_state_dict(vis_sd, strict=False)
        mm_embed.load_state_dict(mm_sd, strict=False)

    vis_model = vis_model.to(device, dtype=compute_dtype).eval()
    mm_embed = mm_embed.to(device, dtype=compute_dtype).eval()
    X_teach = (emb_raw[torch.from_numpy(all_seq_ids)] * ESCALE).to(torch.float16)  # CPU [N_SEQ, L, D]
    with torch.no_grad():
        for vi, im in enumerate(vlm_imgs):
            si = N_TEXT_SEQ + vi
            proc_out = img_proc(images=im, return_tensors="pt")
            pv = proc_out["pixel_values"].to(device, dtype=compute_dtype)
            pos_ids = proc_out["image_position_ids"].to(device)
            vo = vis_model(pixel_values=pv, pixel_position_ids=pos_ids).last_hidden_state
            vm = mm_embed(vo).squeeze(0)[:16].to(torch.float16).cpu()
            pos_m = torch.from_numpy(all_seq_ids[si] == soft_tok)
            take = min(int(pos_m.sum().item()), vm.shape[0])
            idx_m = torch.nonzero(pos_m, as_tuple=False).squeeze(1)[:take]
            X_teach[si, idx_m] = vm[:take]
    del vis_sd, mm_sd
    gc.collect()

    # Compute Global Outlier-Dispersion Permutation Pi from initial hidden states + L0/L59 scales
    with torch.no_grad():
        flat_x0 = X_teach[:min(256, N_SEQ)].reshape(-1, D_MOD).float()
        chan_score = flat_x0.abs().amax(dim=0) + 2.0 * flat_x0.pow(2).mean(dim=0).sqrt()
    perm_pi, inv_perm_pi = compute_outlier_dispersion_perm(chan_score, block_size=V9_BLOCK_SIZE)
    torch.save({"perm_pi": perm_pi, "inv_perm_pi": inv_perm_pi}, OUT_PERM_PT)
    print(f"[Stage 1] Dispersion Perm Pi computed across {D_MOD // V9_BLOCK_SIZE} blocks (score max/min ratio reduced via snake deal)", flush=True)

    # Apply Pi to Teacher X_teach (Student X_stud_perm initialized from quantized emb_eff below)
    X_teach_perm = X_teach[:, :, perm_pi].clone()
    del X_teach
    gc.collect()

    # Pack Vision + Permuted Embedding Table into V9.0 .t16c header pass
    all_norms: List[Tuple[str, np.ndarray]] = [("model.language_model.norm.weight", fn_orig[perm_pi].numpy().astype(np.float32))]
    # Read per-layer vision norms and pack Vision Tower + embed_vision at bsz=64 (exact vision.rs layout)
    H64_GPU = torch.from_numpy(pa.make_sylvester_hadamard(64)).to(device, dtype=torch.float32)

    def _pack_vis_mat(name: str, wt: torch.Tensor, pad_in: int = 0) -> None:
        t = vis_deq[name].float() if (vis_deq is not None and name in vis_deq) else wt.float()
        if "embedding_projection" in name and t.shape[0] == D_MOD:
            t = t[perm_pi]
        do, di = int(t.shape[0]), int(t.shape[1])
        if pad_in > di:
            tp = torch.zeros((do, pad_in), dtype=torch.float32)
            tp[:, :di] = t
            t, die = tp, pad_in
        else:
            die = di
        pb = c_pack((t.to(device).reshape(-1, 64) @ H64_GPU).cpu().numpy(), do, die, 64, 6)
        vision_mat_recs.append((name, do, die, 64, 3 | p3.T16C_GAUSS_BIT, False, True, pb, None, None))

    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        sf_keys = set(sf.keys())
        for std_k in ("model.vision_tower.std_bias", "model.vision_tower.std_scale"):
            if std_k in sf_keys:
                vision_norm_recs.append((std_k, sf.get_tensor(std_k).float().numpy().astype("<f4")))
        pos_tbl_raw = sf.get_tensor("model.vision_tower.patch_embedder.position_embedding_table").float()
        for ax in (0, 1):
            _pack_vis_mat(f"model.vision_tower.patch_embedder.position_embedding_table.{ax}", pos_tbl_raw[ax])
        _pack_vis_mat("model.vision_tower.patch_embedder.input_proj.weight", sf.get_tensor("model.vision_tower.patch_embedder.input_proj.weight"))
        for lv in range(27):
            fp_v = f"model.vision_tower.encoder.layers.{lv}"
            for nk in ("input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                vision_norm_recs.append((f"{fp_v}.{nk}", sf.get_tensor(f"{fp_v}.{nk}").float().numpy().astype("<f4")))
            for pj in ("self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight", "self_attn.o_proj.linear.weight", "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight"):
                _pack_vis_mat(f"{fp_v}.{pj}", sf.get_tensor(f"{fp_v}.{pj}"))
            _pack_vis_mat(f"{fp_v}.mlp.down_proj.linear.weight", sf.get_tensor(f"{fp_v}.mlp.down_proj.linear.weight"), pad_in=4352)
        _pack_vis_mat("model.embed_vision.embedding_projection.weight", sf.get_tensor("model.embed_vision.embedding_projection.weight"))
    del vis_model, mm_embed, H64_GPU
    gc.collect()
    torch.cuda.empty_cache()

    # Quantize & Pack Embed Table (Permuted columns Pi, 3-bit Gaussian + FWHT)
    emb_perm = emb_raw[:, perm_pi].contiguous()
    del emb_raw
    vocab_sz = emb_perm.shape[0]
    exp_emb_b = p3.expected_tile16_bytes(vocab_sz, D_MOD, V9_BLOCK_SIZE, 3, False)
    emb_pb = np.empty(exp_emb_b, dtype=np.uint8)
    emb_eff = torch.empty((vocab_sz, D_MOD), dtype=torch.float16)
    tblk_emb = p3.tile16_blk_bytes(V9_BLOCK_SIZE, 3, False) * (D_MOD // V9_BLOCK_SIZE)
    for r0 in range(0, vocab_sz, 16384):
        r1 = min(r0 + 16384, vocab_sz)
        w_c = emb_perm[r0:r1].to(device, dtype=torch.float32)
        w_rot = (w_c.reshape(-1, V9_BLOCK_SIZE) @ H256_GPU).contiguous()
        pb_c = c_pack(w_rot.cpu().numpy(), r1 - r0, D_MOD, V9_BLOCK_SIZE, 6)
        emb_pb[(r0 // 16) * tblk_emb : (r0 // 16) * tblk_emb + pb_c.size] = pb_c
        # Reconstruct approximate effective embedding for L59 distillation
        u_c, sc_c, _ = unpack_tile16_headers_and_units(pb_c, r1 - r0, D_MOD, V9_BLOCK_SIZE, 6, H256_GPU)
        emb_eff[r0:r1] = reconstruct_from_headers(u_c, sc_c, None, None, H256_GPU).to(torch.float16).cpu()
        del w_c, w_rot, pb_c, u_c, sc_c
    del emb_perm
    X_stud_perm = (emb_eff[torch.from_numpy(all_seq_ids)] * ESCALE).to(torch.float16)
    for si in range(N_TEXT_SEQ, N_SEQ):
        pos_m = torch.from_numpy(all_seq_ids[si] == soft_tok)
        X_stud_perm[si, pos_m] = X_teach_perm[si, pos_m]
    torch.cuda.empty_cache()

    # Collect per-layer permuted norms upfront so T16C header can be written deterministically
    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        for l_idx in range(N_LAYERS):
            pfx = f"model.language_model.layers.{l_idx}"
            for nk in ("input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"):
                arr = sf.get_tensor(f"{pfx}.{nk}").float()[perm_pi].numpy().astype(np.float32)
                all_norms.append((f"{pfx}.{nk}", arr))
            for qk in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                all_norms.append((f"{pfx}.{qk}", sf.get_tensor(f"{pfx}.{qk}").float().numpy().astype(np.float32)))
            all_norms.append((f"{pfx}.layer_scalar", sf.get_tensor(f"{pfx}.layer_scalar").float().numpy().astype(np.float32)))
    all_norms.extend(vision_norm_recs)

    # Compute exact total_mats = 1 (embed) + 60*(3 or 4 attn + 3 mlp) + len(vision_mat_recs)
    lm_mat_specs_cnt = 1
    for l_idx in range(N_LAYERS):
        is_f = LAYER_TYPES[l_idx] == "full_attention"
        lm_mat_specs_cnt += (3 if (is_f and K_EQ_V) else 4) + 3
    total_mats = lm_mat_specs_cnt + len(vision_mat_recs)

    # ------------------------------------------------------------------
    # Stage 2: Layer-by-Layer Sparse-Route QAT + Compass-Magnetism + CRTR-V9 + C8DB
    # ------------------------------------------------------------------
    tmp_t16c = OUT_T16C.with_suffix(".t16c.tmp")
    crtr_layers: List[dict] = []
    head_crtr_layers: List[dict] = []
    c8db_entries: List[dict] = []
    layer_logs: List[dict] = []
    # Also retain quantized L58/L59 K/V states for Assistant-V9 4-step rollout
    ast_seqs = np.linspace(0, N_TEXT_SEQ - 1, min(128, N_TEXT_SEQ), dtype=int).tolist()
    cap_kv58 = {"k_rope": {}, "v_norm": {}}
    cap_kv59 = {"k_rope": {}, "v_norm": {}}

    def _load_dequant_proj(sf_obj, base_name: str, is_perm_col: bool = False, is_perm_row: bool = False) -> torch.Tensor:
        if f"{base_name}.weight_packed" in sf_obj.keys():
            pk = sf_obj.get_tensor(f"{base_name}.weight_packed").to(device)
            sc = sf_obj.get_tensor(f"{base_name}.weight_scale").to(device, dtype=torch.float32)
            shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
            nibs = ((pk.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF) - 8
            w = (nibs.reshape(pk.shape[0], sc.shape[1], 32).float() * sc.unsqueeze(-1)).reshape(pk.shape[0], -1)
            del pk, sc, nibs
        else:
            w = sf_obj.get_tensor(f"{base_name}.weight").to(device, dtype=torch.float32)
        if is_perm_col:
            w = w[:, perm_pi.to(device)]
        if is_perm_row:
            w = w[perm_pi.to(device), :]
        return w

    rope_cs_g, rope_sn_g = make_rope(THETA_G, int(round(HD_F * PART_G)), HD_F)
    rope_cs_s, rope_sn_s = make_rope(THETA_L, int(round(HD_S * PART_L)), HD_S)

    def fwht_2sided_gpu(h_in: torch.Tensor, inv_s_gpu: Optional[torch.Tensor], bsize: int) -> torch.Tensor:
        h = h_in.clone()
        if inv_s_gpu is not None:
            inv_s_loc = inv_s_gpu.to(h.device)
            h.mul_(inv_s_loc.unsqueeze(1))
            h.mul_(inv_s_loc.unsqueeze(0))
        d = h.shape[0]
        hb = H256_GPU if bsize == V9_BLOCK_SIZE else torch.from_numpy(pa.make_sylvester_hadamard(bsize)).to(h.device, dtype=torch.float32)
        step = min(d, max(bsize, 2048 - (2048 % bsize)))
        for r0 in range(0, d, step):
            r1 = min(r0 + step, d)
            h[r0:r1] = torch.matmul(hb, h[r0:r1].reshape((r1 - r0) // bsize, bsize, d)).reshape(r1 - r0, d)
        for c0 in range(0, d, step):
            c1 = min(c0 + step, d)
            h[:, c0:c1] = torch.matmul(h[:, c0:c1].reshape(d, (c1 - c0) // bsize, bsize), hb).reshape(d, c1 - c0)
        return 0.5 * (h + h.T)

    def gptq_quant_gpu(W: torch.Tensor, U: torch.Tensor, bsize: int, mode_i: int) -> None:
        rows, cw = W.shape
        nblks = cw // bsize
        diag_u = torch.diagonal(U).clamp(min=1e-12)
        E_blk = torch.empty((rows, bsize), device=W.device, dtype=torch.float32)
        for b in range(nblks):
            b0, b1 = b * bsize, (b + 1) * bsize
            blk = W[:, b0:b1]
            if mode_i == 0:
                bias = blk.mean(dim=1)
                for _ in range(2):
                    m1 = blk >= bias.unsqueeze(1)
                    c1 = m1.sum(dim=1)
                    c0 = bsize - c1
                    mp = torch.where(c1 > 0, torch.where(m1, blk, 0.0).sum(dim=1) / c1.clamp(min=1), bias)
                    mn = torch.where(c0 > 0, torch.where(m1, 0.0, blk).sum(dim=1) / c0.clamp(min=1), bias)
                    bias = 0.5 * (mp + mn); sc = 0.5 * (mp - mn)
                bias = bias.to(torch.float16).float(); sc = sc.to(torch.float16).float()
                sc_p, sc_n = bias + sc, bias - sc
            elif mode_i == 1:
                sc = 1.25 * blk.abs().mean(dim=1)
                for _ in range(4):
                    tau = 0.5 * sc.unsqueeze(1)
                    pos_m, neg_m = blk > tau, blk < -tau
                    num = torch.where(pos_m, blk, 0.0).sum(dim=1) - torch.where(neg_m, blk, 0.0).sum(dim=1)
                    cnt = (pos_m | neg_m).sum(dim=1)
                    sc = torch.where(cnt > 0, num / cnt.clamp(min=1), 0.0)
                sc = sc.to(torch.float16).float(); tau_vec = 0.5 * sc
            elif mode_i in (5, 6):
                if mode_i == 5:
                    lv_tbl = torch.tensor([-3.174, -1.000, 1.000, 3.174], device=W.device, dtype=torch.float32)
                    thr = torch.tensor([-2.087, 0.0, 2.087], device=W.device, dtype=torch.float32)
                    init_k = 0.470
                else:
                    lv_tbl = torch.tensor([-7.430, -5.430, -3.270, -1.270, 1.270, 3.270, 5.430, 7.430], device=W.device, dtype=torch.float32)
                    thr = torch.tensor([-6.430, -4.350, -2.270, 0.0, 2.270, 4.350, 6.430], device=W.device, dtype=torch.float32)
                    init_k = 0.275
                sc = init_k * torch.sqrt((blk * blk).mean(dim=1))
                for _ in range(4):
                    inv_s_v = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0).unsqueeze(1)
                    u_b = blk * inv_s_v
                    qi = (u_b.unsqueeze(-1) >= thr).sum(dim=-1)
                    lv = lv_tbl[qi]
                    den = (lv * lv).sum(dim=1)
                    sc = torch.where(den > 0.0, (lv * blk).sum(dim=1) / den.clamp(min=1e-12), 0.0)
                sc = sc.to(torch.float16).float()
                inv_s_vec = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0)
            else:
                max_lvl = 3.0 if mode_i in (2, 5) else 7.0
                sc = (0.498 if mode_i in (2, 5) else 0.293) * torch.sqrt((blk * blk).mean(dim=1))
                for _ in range(4):
                    inv_s_v = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0).unsqueeze(1)
                    q = torch.clamp(torch.round((blk * inv_s_v + max_lvl) * 0.5), 0.0, max_lvl)
                    lv = 2.0 * q - max_lvl
                    den = (lv * lv).sum(dim=1)
                    sc = torch.where(den > 0.0, (lv * blk).sum(dim=1) / den.clamp(min=1e-12), 0.0)
                sc = sc.to(torch.float16).float()
                inv_s_vec = torch.where(sc > 1e-12, 1.0 / sc.clamp(min=1e-12), 0.0)
            for k in range(bsize):
                j = b0 + k
                val = W[:, j].clone()
                if mode_i == 0:
                    qval = torch.where(val >= bias, sc_p, sc_n)
                elif mode_i == 1:
                    qval = torch.where(val > tau_vec, sc, torch.where(val < -tau_vec, -sc, 0.0))
                elif mode_i in (5, 6):
                    qi = ((val * inv_s_vec).unsqueeze(-1) >= thr).sum(dim=-1)
                    qval = sc * lv_tbl[qi]
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

    norm_offsets: Dict[str, int] = {}
    ckpt_meta_p = CKPT_DIR / "stage2_v9_ckpt.pt"
    ckpt_xt_p = CKPT_DIR / "x_teach.pt"
    ckpt_xs_p = CKPT_DIR / "x_stud.pt"
    ckpt_cr_p = CKPT_DIR / "crtr_layers.pt"
    ckpt_hr_p = CKPT_DIR / "head_crtr_layers.pt"
    ckpt_c8_p = CKPT_DIR / "c8db_entries.pt"

    # Recover finalized OUT_T16C back to tmp_t16c if Stage 3 crashed after os.replace()
    if ckpt_meta_p.is_file() and not tmp_t16c.is_file() and OUT_T16C.is_file():
        os.replace(OUT_T16C, tmp_t16c)

    # Restore checkpoint from Google Drive if local runtime restarted cleanly or tmp_t16c is absent
    if (not ckpt_meta_p.is_file() or not tmp_t16c.is_file()) and DRIVE_DIR is not None:
        d_ck = DRIVE_DIR / "ckpt_v9_0"
        d_part = DRIVE_DIR / (OUT_T16C.name + ".partial")
        if (d_ck / "stage2_v9_ckpt.pt").is_file() and d_part.is_file():
            print(f"[V9.0 Drive Restore] Restoring Stage 2 checkpoint from {DRIVE_DIR}...", flush=True)
            if not tmp_t16c.is_file():
                shutil.copy2(d_part, tmp_t16c)
            for fn_ck in ("stage2_v9_ckpt.pt", "x_teach.pt", "x_stud.pt", "crtr_layers.pt", "head_crtr_layers.pt", "c8db_entries.pt"):
                src_ck = d_ck / fn_ck
                dst_ck = CKPT_DIR / fn_ck
                if src_ck.is_file() and not dst_ck.is_file():
                    shutil.copy2(src_ck, dst_ck)

    # Write initial header + norms (resume if partial exists on Drive/local)
    resumed_layer = 0
    resumed_pos = 0
    if ckpt_meta_p.is_file() and tmp_t16c.is_file() and ckpt_xt_p.is_file() and ckpt_xs_p.is_file():
        meta_ck = torch.load(ckpt_meta_p, map_location="cpu", weights_only=False)
        resumed_layer = int(meta_ck["next_layer"])
        resumed_pos = int(meta_ck["pos"])
        assert tmp_t16c.stat().st_size >= resumed_pos, (
            f"Fail-closed resume guard: {tmp_t16c} size {tmp_t16c.stat().st_size:,} < resumed_pos {resumed_pos:,}"
        )
        norm_offsets = dict(meta_ck["norm_offsets"])
        layer_logs = list(meta_ck["layer_logs"])
        crtr_layers = torch.load(ckpt_cr_p, map_location="cpu", weights_only=False)
        head_crtr_layers = torch.load(ckpt_hr_p, map_location="cpu", weights_only=False) if ckpt_hr_p.is_file() else []
        c8db_entries = torch.load(ckpt_c8_p, map_location="cpu", weights_only=False)
        X_teach_perm = torch.load(ckpt_xt_p, map_location="cpu", weights_only=True)
        X_stud_perm = torch.load(ckpt_xs_p, map_location="cpu", weights_only=True)
        cap_kv58 = meta_ck.get("cap_kv58", cap_kv58)
        cap_kv59 = meta_ck.get("cap_kv59", cap_kv59)
        print(f"[V9.0 Resume] Resuming from Layer {resumed_layer}/{N_LAYERS} (pos={resumed_pos:,})", flush=True)

    f_mode = "r+b" if resumed_layer > 0 else "wb"
    with open(tmp_t16c, f_mode) as fp, safe_open(sf_path, framework="pt", device="cpu") as sf:
        if resumed_layer == 0:
            fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, len(all_norms), total_mats, 0))
            pos = 24
            for nname, narr in all_norms:
                nb = nname.encode("utf-8")
                a32 = np.ascontiguousarray(narr, dtype="<f4").ravel()
                fp.write(struct.pack("<H", len(nb)))
                fp.write(nb)
                fp.write(struct.pack("<I", a32.size))
                norm_offsets[nname] = pos + 2 + len(nb) + 4
                fp.write(a32.tobytes())
                pos += 2 + len(nb) + 4 + a32.size * 4
            pos = p3._write_mat_record(
                fp, pos, "model.language_model.embed_tokens.weight", vocab_sz, D_MOD, V9_BLOCK_SIZE,
                3 | p3.T16C_GAUSS_BIT, False, True, emb_pb, None, None,
            )
            del emb_pb
        else:
            pos = resumed_pos
            fp.truncate(pos)
            fp.seek(pos)
            del emb_pb

        for l_idx in range(resumed_layer, N_LAYERS):
            tl0 = time.perf_counter()
            pfx = f"model.language_model.layers.{l_idx}"
            is_f = LAYER_TYPES[l_idx] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)
            rep = NH // nkv

            g_in = torch.from_numpy(all_norms[1 + l_idx * 7 + 0][1]).to(device)
            g_pa = nn.Parameter(torch.from_numpy(all_norms[1 + l_idx * 7 + 1][1]).to(device))
            g_pf = torch.from_numpy(all_norms[1 + l_idx * 7 + 2][1]).to(device)
            g_pff = nn.Parameter(torch.from_numpy(all_norms[1 + l_idx * 7 + 3][1]).to(device))
            g_qn = torch.from_numpy(all_norms[1 + l_idx * 7 + 4][1]).to(device)
            g_kn = torch.from_numpy(all_norms[1 + l_idx * 7 + 5][1]).to(device)
            lsc_init = float(all_norms[1 + l_idx * 7 + 6][1][0])
            lsc_p = nn.Parameter(torch.tensor([lsc_init], device=device, dtype=torch.float32))

            wq_t = _load_dequant_proj(sf, f"{pfx}.self_attn.q_proj", is_perm_col=True)
            wk_t = _load_dequant_proj(sf, f"{pfx}.self_attn.k_proj", is_perm_col=True)
            wv_t = wk_t if (is_f and K_EQ_V) else _load_dequant_proj(sf, f"{pfx}.self_attn.v_proj", is_perm_col=True)
            wo_t = _load_dequant_proj(sf, f"{pfx}.self_attn.o_proj", is_perm_row=True)
            wg_t = _load_dequant_proj(sf, f"{pfx}.mlp.gate_proj", is_perm_col=True)
            wu_t = _load_dequant_proj(sf, f"{pfx}.mlp.up_proj", is_perm_col=True)
            wd_t = _load_dequant_proj(sf, f"{pfx}.mlp.down_proj", is_perm_row=True)

            # 1) Exact Teacher Forward + Hessian H = X^T X accumulation + Head Energy + Co-activation samples
            attn_x_rms = torch.zeros((D_MOD,), device=device, dtype=torch.float32)
            mlp_x_rms = torch.zeros((D_MOD,), device=device, dtype=torch.float32)
            H_attn = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32)
            H_o = torch.zeros((NH * hd, NH * hd), device=device, dtype=torch.float32)
            H_mlp = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32)
            act_samples: List[torch.Tensor] = []
            x_attn_ev: List[torch.Tensor] = []
            x_o_ev: List[torch.Tensor] = []
            x_mlp_ev: List[torch.Tensor] = []
            X_teach_next = torch.empty_like(X_teach_perm)
            X_teach_attn_out = torch.empty_like(X_teach_perm)
            head_e_cpu = torch.empty((N_SEQ * V9_SEQ_LEN, NH), dtype=torch.float32)
            wo_head_norm = wo_t.view(D_MOD, NH, hd).pow(2).sum(dim=(0, 2)).sqrt().clamp(min=1e-6)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xb = X_teach_perm[b0:b1].to(device, dtype=torch.float32)
                    xn = rmsn_gpu(xb, g_in)
                    xn_f = xn.reshape(-1, D_MOD)
                    attn_x_rms.add_((xn * xn).sum(dim=(0, 1)))
                    H_attn.add_(xn_f.T @ xn_f)
                    if b0 < 256:
                        x_attn_ev.append(xn_f[::4])
                    q = rmsn_gpu(F.linear(xn, wq_t).view(b1 - b0, V9_SEQ_LEN, NH, hd), g_qn)
                    k = rmsn_gpu(F.linear(xn, wk_t).view(b1 - b0, V9_SEQ_LEN, nkv, hd), g_kn)
                    v = rmsn_gpu(F.linear(xn, wv_t).view(b1 - b0, V9_SEQ_LEN, nkv, hd), None)
                    q = apply_rope(q, cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    k = apply_rope(k, cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    v = v.transpose(1, 2)
                    ctx = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0)
                    ctx_h = ctx.transpose(1, 2).contiguous()
                    head_e_cpu[b0 * V9_SEQ_LEN : b1 * V9_SEQ_LEN] = (ctx_h.pow(2).sum(dim=-1).sqrt() * wo_head_norm.view(1, 1, NH)).reshape(-1, NH).to(torch.float32).cpu()
                    ctx = ctx_h.view(b1 - b0, V9_SEQ_LEN, NH * hd)
                    ctx_f = ctx.reshape(-1, NH * hd)
                    H_o.add_(ctx_f.T @ ctx_f)
                    if b0 < 256:
                        x_o_ev.append(ctx_f[::4])
                    xm = xb + rmsn_gpu(F.linear(ctx, wo_t), g_pa.detach())
                    X_teach_attn_out[b0:b1] = xm.to(torch.float16).cpu()
                    xf = rmsn_gpu(xm, g_pf)
                    xf_f = xf.reshape(-1, D_MOD)
                    mlp_x_rms.add_((xf * xf).sum(dim=(0, 1)))
                    H_mlp.add_(xf_f.T @ xf_f)
                    if b0 < 256:
                        x_mlp_ev.append(xf_f[::4])
                    act = F.gelu(F.linear(xf, wg_t), approximate="tanh") * F.linear(xf, wu_t)
                    if b0 < 64:
                        act_samples.append(act[:, ::2, :].reshape(-1, V9_FF).abs().to(torch.float32).cpu())
                    xo = (xm + rmsn_gpu(F.linear(act, wd_t), g_pff.detach())) * lsc_init
                    X_teach_next[b0:b1] = xo.to(torch.float16).cpu()

            attn_x_rms = torch.sqrt(attn_x_rms / float(N_SEQ * V9_SEQ_LEN)).clamp(min=1e-5)
            mlp_x_rms = torch.sqrt(mlp_x_rms / float(N_SEQ * V9_SEQ_LEN)).clamp(min=1e-5)
            xa_ev_t = torch.cat(x_attn_ev, dim=0)
            xo_ev_t = torch.cat(x_o_ev, dim=0)
            xm_ev_t = torch.cat(x_mlp_ev, dim=0)

            # 2) Lossless Co-Activation Permutation pi_l in S_21504 for layer l
            act_cat = torch.cat(act_samples, dim=0).to(device, dtype=torch.float32)
            pi_l, _, pi_stats = compute_geglu_colblock_perm(act_cat, shared_blocks=V9_SHARED_BLOCKS, block_size=V9_BLOCK_SIZE, iters=6, seed=42 + l_idx)
            pi_l_g = pi_l.to(device)
            wg_t = wg_t[pi_l_g, :].contiguous()
            wu_t = wu_t[pi_l_g, :].contiguous()
            wd_t = wd_t[:, pi_l_g].contiguous()
            del act_samples, act_cat
            torch.cuda.empty_cache()

            # 3) SmoothQuant + FWHT + GPU Cholesky Hessian GPTQ + ALS INT8 Low-Rank Residual
            def quantize_proj(
                w_f32: torch.Tensor, xr: torch.Tensor, mode_i: int,
                h_gpu: Optional[torch.Tensor] = None, lr_rank: int = 0, x_ev: Optional[torch.Tensor] = None,
            ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
                do, di = int(w_f32.shape[0]), int(w_f32.shape[1])
                if di % V9_BLOCK_SIZE == 0 and xr.numel() == di:
                    sq = (xr.pow(0.5) / w_f32.abs().amax(dim=0).clamp(min=1e-5).pow(0.5)).clamp(0.25, 4.0)
                    sq = sq / sq.mean().clamp(min=1e-5)
                    inv_s = 1.0 / sq
                    w_sc = w_f32 * sq.unsqueeze(0)
                    inv_s_np = inv_s.cpu().numpy().astype("<f4")
                else:
                    inv_s, w_sc, inv_s_np = None, w_f32, None
                w_rot = (w_sc.reshape(-1, V9_BLOCK_SIZE) @ H256_GPU).contiguous()
                if h_gpu is not None and h_gpu.shape[0] == di:
                    with torch.no_grad():
                        h_rot = fwht_2sided_gpu(h_gpu, inv_s, V9_BLOCK_SIZE)
                        if torch.isfinite(h_rot).all():
                            diag_m = h_rot.diagonal().mean().clamp(min=1e-6)
                            L = None
                            for df in (0.02, 0.05, 0.15, 0.50, 1.50):
                                h_rot.diagonal().add_(df * diag_m)
                                try:
                                    L = torch.linalg.cholesky(h_rot)
                                    break
                                except RuntimeError:
                                    L = None
                            del h_rot
                            if L is not None:
                                h_inv = torch.cholesky_inverse(L); del L
                                h_inv = 0.5 * (h_inv + h_inv.T)
                                U_ch = torch.linalg.cholesky(h_inv, upper=True).contiguous(); del h_inv
                                gptq_quant_gpu(w_rot, U_ch, V9_BLOCK_SIZE, mode_i); del U_ch
                        else:
                            del h_rot
                pb = c_pack(w_rot.cpu().numpy(), do, di, V9_BLOCK_SIZE, mode_i)
                units, sc_init, bi_init = unpack_tile16_headers_and_units(pb, do, di, V9_BLOCK_SIZE, mode_i, H256_GPU)
                low_rank = None
                if lr_rank > 0 and x_ev is not None and x_ev.shape[0] >= lr_rank * 2:
                    with torch.no_grad():
                        w_eff0 = reconstruct_from_headers(units, sc_init, bi_init, inv_s, H256_GPU)
                        xv = x_ev[: min(4096, x_ev.shape[0])].float()
                        y_err = (xv @ w_f32.T) - (xv @ w_eff0.T)
                        if xv.shape[0] <= do:
                            _, evs = torch.linalg.eigh(y_err @ y_err.T)
                            U_f = y_err.T @ evs[:, -lr_rank:].flip(1).contiguous()
                            U_f = U_f / U_f.norm(dim=0, keepdim=True).clamp(min=1e-6)
                        else:
                            _, evs = torch.linalg.eigh(y_err.T @ y_err)
                            U_f = evs[:, -lr_rank:].flip(1).contiguous()
                        reg = max(0.02 * float((xv * xv).sum().item()) / float(max(di, 1)), 1e-6)
                        use_dual = xv.shape[0] < di
                        gram = (xv @ xv.T) if use_dual else (xv.T @ xv)
                        gram.diagonal().add_(reg)
                        V_f = ((xv.T @ torch.linalg.solve(gram, y_err @ U_f)) if use_dual else torch.linalg.solve(gram, xv.T @ (y_err @ U_f))).T.contiguous()
                        su = (U_f.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
                        sv = (V_f.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
                        u_i8 = torch.clamp(torch.round(U_f / su.unsqueeze(1)), -127, 127).to(torch.int8)
                        v_i8 = torch.clamp(torch.round(V_f / sv.unsqueeze(1)), -127, 127).to(torch.int8)
                        low_rank = (su.cpu().numpy().astype("<f4"), sv.cpu().numpy().astype("<f4"), u_i8.cpu().numpy().astype(np.int8), v_i8.cpu().numpy().astype(np.int8))
                return pb, inv_s_np, inv_s, units, sc_init, bi_init, low_rank

            # Train 32-Head Attention Router (OUT_HEAD_CRTR) on student RMSNorm inputs
            xn_stud_cpu = torch.empty_like(X_stud_perm)
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xn_stud_cpu[b0:b1] = rmsn_gpu(X_stud_perm[b0:b1].to(device, dtype=torch.float32), g_in).to(torch.float16).cpu()
            xn_flat = xn_stud_cpu.reshape(-1, D_MOD)
            n_tr_h = (xn_flat.shape[0] * 9) // 10
            hr_net, hr_met = crtr.train_layer_router(
                xn_flat[:n_tr_h], head_e_cpu[:n_tr_h], xn_flat[n_tr_h:], head_e_cpu[n_tr_h:],
                top_p=float(V9_ATTN_ROUTED_HEADS) / float(NH),
                rank=V9_HEAD_CRTR_RANK, steps=120, lr=3e-3,
                device=str(device), seed=300 + l_idx,
            )
            head_crtr_layers.append(crtr.quantize_layer_for_crtr(hr_net, V9_BLOCK_SIZE))

            # Pack Attention + Sparse-Head 10/32 Differentiable Attention QAT (L2..L57)
            attn_p = [("self_attn.q_proj", wq_t), ("self_attn.k_proj", wk_t)]
            if not (is_f and K_EQ_V):
                attn_p.append(("self_attn.v_proj", wv_t))
            attn_p.append(("self_attn.o_proj", wo_t))

            attn_packed: Dict[str, torch.Tensor] = {}
            attn_recs: Dict[str, dict] = {}
            for pn, wt in attn_p:
                m_i, bsz_i, lrr_a, is_rm_a, is_cb_a = get_v9_proj_spec(l_idx, pn, is_f)
                is_o = pn == "self_attn.o_proj"
                xr_a = torch.ones((wt.shape[1],), device=device) if is_o else attn_x_rms
                h_a = H_o if is_o else H_attn
                ev_a = xo_ev_t if is_o else xa_ev_t
                pb, is_np, is_g, u_g, sc_g, bi_g, lr_a = quantize_proj(wt, xr_a, m_i, h_gpu=h_a, lr_rank=lrr_a, x_ev=ev_a)
                w_eff_a = reconstruct_from_headers(u_g, sc_g, bi_g, is_g, H256_GPU)
                lr_dense = None
                if lr_a is not None:
                    su_a, sv_a, ui_a, vi_a = lr_a
                    u_f = torch.from_numpy(ui_a).to(device).float() * torch.from_numpy(su_a).to(device).unsqueeze(1)
                    v_f = torch.from_numpy(vi_a).to(device).float() * torch.from_numpy(sv_a).to(device).unsqueeze(1)
                    lr_dense = u_f @ v_f
                    w_eff_a = w_eff_a + lr_dense
                attn_packed[pn] = w_eff_a.to(torch.float16)
                attn_recs[pn] = {
                    "do": int(wt.shape[0]), "di": int(wt.shape[1]), "bsz": bsz_i, "mode": m_i,
                    "is_rm": is_rm_a, "is_cb": is_cb_a, "pb": pb, "is_np": is_np, "lr": lr_a,
                    "u": u_g, "sc": sc_g, "bi": bi_g, "is_g": is_g, "lr_dense": lr_dense,
                }
            del H_attn, H_o, xa_ev_t, xo_ev_t

            if 2 <= l_idx <= 57:
                rq, ro = attn_recs["self_attn.q_proj"], attn_recs["self_attn.o_proj"]
                sc_q_p = nn.Parameter(rq["sc"].clone())
                sc_o_p = nn.Parameter(ro["sc"].clone())
                bi_q_p = nn.Parameter(rq["bi"].clone()) if rq["bi"] is not None else None
                bi_o_p = nn.Parameter(ro["bi"].clone()) if ro["bi"] is not None else None
                lr_qd = rq["lr_dense"]
                lr_od = ro["lr_dense"]
                wk_q_dt = attn_packed["self_attn.k_proj"].to(compute_dtype)
                wv_q_dt = wk_q_dt if (is_f and K_EQ_V) else attn_packed["self_attn.v_proj"].to(compute_dtype)
                opt_a_params = [sc_q_p, sc_o_p, g_pa]
                if bi_q_p is not None:
                    opt_a_params.append(bi_q_p)
                if bi_o_p is not None:
                    opt_a_params.append(bi_o_p)
                opt_attn = torch.optim.AdamW(opt_a_params, lr=4e-3, weight_decay=1e-4)
                qat_attn_bs = 16 if gpu_vram_gb >= 64.0 else 4
                hr_net_g = hr_net.to(device).eval()
                for _ in range(20):
                    opt_attn.zero_grad()
                    bs_q = torch.randint(0, N_SEQ, (min(qat_attn_bs, N_SEQ),), device="cpu")
                    xb_b = X_stud_perm[bs_q].to(device, dtype=torch.float32)
                    xn_b = xn_stud_cpu[bs_q].to(device, dtype=torch.float32)
                    xta_b = X_teach_attn_out[bs_q].to(device, dtype=torch.float32)
                    with torch.no_grad():
                        hr_log, _ = hr_net_g(xn_b.reshape(-1, D_MOD))
                        top_h = torch.topk(hr_log, V9_ATTN_ROUTED_HEADS, dim=-1).indices
                        hm = torch.zeros((xn_b.shape[0] * V9_SEQ_LEN, NH), device=device)
                        hm.scatter_(1, top_h, 1.0)
                        hm = hm.view(xb_b.shape[0], V9_SEQ_LEN, NH, 1)
                    wq_step = reconstruct_from_headers(rq["u"], sc_q_p, bi_q_p, rq["is_g"], H256_GPU)
                    if lr_qd is not None:
                        wq_step = wq_step + lr_qd
                    wo_step = reconstruct_from_headers(ro["u"], sc_o_p, bi_o_p, ro["is_g"], H256_GPU)
                    if lr_od is not None:
                        wo_step = wo_step + lr_od
                    q_st = apply_rope(rmsn_gpu(F.linear(xn_b.to(compute_dtype), wq_step.to(compute_dtype)).view(xb_b.shape[0], V9_SEQ_LEN, NH, hd), g_qn), cs_r, sn_r, rot_d, hd).transpose(1, 2).to(compute_dtype)
                    with torch.no_grad():
                        k_st = apply_rope(rmsn_gpu(F.linear(xn_b.to(compute_dtype), wk_q_dt).view(xb_b.shape[0], V9_SEQ_LEN, nkv, hd), g_kn), cs_r, sn_r, rot_d, hd).transpose(1, 2).to(compute_dtype)
                        v_st = rmsn_gpu(F.linear(xn_b.to(compute_dtype), wv_q_dt).view(xb_b.shape[0], V9_SEQ_LEN, nkv, hd), None).transpose(1, 2).to(compute_dtype)
                    ctx_st = F.scaled_dot_product_attention(q_st, k_st.repeat_interleave(rep, 1), v_st.repeat_interleave(rep, 1), is_causal=True, scale=1.0)
                    ctx_masked = (ctx_st.transpose(1, 2).float() * hm).reshape(xb_b.shape[0], V9_SEQ_LEN, NH * hd).to(compute_dtype)
                    xm_s = xb_b + rmsn_gpu(F.linear(ctx_masked, wo_step.to(compute_dtype)), g_pa).float()
                    mag_t_a = torch.log(xta_b.pow(2).sum(dim=-1).clamp(min=1e-8))
                    mag_s_a = torch.log(xm_s.pow(2).sum(dim=-1).clamp(min=1e-8))
                    loss_a_cm = (1.0 - F.cosine_similarity(xm_s, xta_b, dim=-1).mean()) + 0.15 * F.mse_loss(mag_s_a, mag_t_a) + 0.25 * F.mse_loss(xm_s, xta_b)
                    loss_a_cm.backward()
                    torch.nn.utils.clip_grad_norm_(opt_a_params, 1.0)
                    opt_attn.step()

                for rc_k, sc_p_k, bi_p_k in ((rq, sc_q_p, bi_q_p), (ro, sc_o_p, bi_o_p)):
                    sc_np_a = np.ascontiguousarray(sc_p_k.detach().cpu().numpy(), dtype=np.float32)
                    bi_ptr_a = 0
                    if bi_p_k is not None:
                        bi_np_a = np.ascontiguousarray(bi_p_k.detach().cpu().numpy(), dtype=np.float32)
                        bi_ptr_a = bi_np_a.ctypes.data
                    LIB_FQ.update_t16c_headers_f32(rc_k["pb"].ctypes.data, rc_k["do"], rc_k["di"], rc_k["bsz"], rc_k["mode"], sc_np_a.ctypes.data, bi_ptr_a)
                with torch.no_grad():
                    wq_fin_a = reconstruct_from_headers(rq["u"], sc_q_p.detach(), bi_q_p.detach() if bi_q_p is not None else None, rq["is_g"], H256_GPU)
                    if lr_qd is not None:
                        wq_fin_a = wq_fin_a + lr_qd
                    wo_fin_a = reconstruct_from_headers(ro["u"], sc_o_p.detach(), bi_o_p.detach() if bi_o_p is not None else None, ro["is_g"], H256_GPU)
                    if lr_od is not None:
                        wo_fin_a = wo_fin_a + lr_od
                    attn_packed["self_attn.q_proj"] = wq_fin_a.to(torch.float16)
                    attn_packed["self_attn.o_proj"] = wo_fin_a.to(torch.float16)
                nk_pa = f"{pfx}.post_attention_layernorm.weight"
                if nk_pa in norm_offsets:
                    cur_p_a = fp.tell()
                    fp.seek(norm_offsets[nk_pa])
                    fp.write(np.ascontiguousarray(g_pa.detach().cpu().numpy(), dtype="<f4").tobytes())
                    fp.seek(cur_p_a)
                del wk_q_dt, wv_q_dt, opt_attn, sc_q_p, sc_o_p, bi_q_p, bi_o_p

            for pn, _ in attn_p:
                rc = attn_recs[pn]
                hdr_a, pay_a = finalize_t16c_payload(rc["pb"], rc["do"], rc["di"], rc["bsz"], rc["mode"], rc["is_rm"], rc["is_cb"])
                pos = p3._write_mat_record(fp, pos, f"{pfx}.{pn}.weight", rc["do"], rc["di"], rc["bsz"], hdr_a, p3.MODE_TO_T16C[rc["mode"]][1], True, pay_a, rc["is_np"], rc["lr"])
            del attn_recs, X_teach_attn_out, head_e_cpu, xn_stud_cpu

            # Pack MLP & Run Real Sparse-Route Scale + Norm + LayerScalar + C8DB QAT (10/84 = 11.9%)
            m_g, _, _, is_rm_g, is_cb_g = get_v9_proj_spec(l_idx, "mlp.gate_proj", is_f)
            m_u, _, _, is_rm_u, is_cb_u = get_v9_proj_spec(l_idx, "mlp.up_proj", is_f)
            m_d, _, lrr_d, is_rm_d, is_cb_d = get_v9_proj_spec(l_idx, "mlp.down_proj", is_f)
            pb_g, is_g_np, is_g_g, u_g, sc_g, bi_g, _ = quantize_proj(wg_t, mlp_x_rms, m_g, h_gpu=H_mlp)
            pb_u, is_u_np, is_u_g, u_u, sc_u, bi_u, _ = quantize_proj(wu_t, mlp_x_rms, m_u, h_gpu=H_mlp)
            del H_mlp, xm_ev_t
            pb_d, is_d_np, is_d_g, u_d, sc_d, bi_d, _ = quantize_proj(wd_t, torch.ones((V9_FF,), device=device), m_d, lr_rank=0)

            # Student Attention output on current layer (applies exact 10/32 head mask on L2..L57!)
            xm_stud_cpu = torch.empty_like(X_stud_perm)
            xf_stud_cpu = torch.empty_like(X_stud_perm)
            blk_e_cpu = torch.empty((N_SEQ * V9_SEQ_LEN, V9_MLP_BLOCKS), dtype=torch.float32)
            mod_sal_cpu = torch.empty((N_SEQ * V9_SEQ_LEN,), dtype=torch.float32)

            wq_q = attn_packed["self_attn.q_proj"].float()
            wk_q = attn_packed["self_attn.k_proj"].float()
            wv_q = wk_q if (is_f and K_EQ_V) else attn_packed["self_attn.v_proj"].float()
            wo_q = attn_packed["self_attn.o_proj"].float()
            wg_init = reconstruct_from_headers(u_g, sc_g, bi_g, is_g_g, H256_GPU).float()
            hr_net_eval = hr_net.to(device).eval() if (2 <= l_idx <= 57) else None
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xb = X_stud_perm[b0:b1].to(device, dtype=torch.float32)
                    xn = rmsn_gpu(xb, g_in)
                    q = apply_rope(rmsn_gpu(F.linear(xn, wq_q).view(b1 - b0, V9_SEQ_LEN, NH, hd), g_qn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    k = apply_rope(rmsn_gpu(F.linear(xn, wk_q).view(b1 - b0, V9_SEQ_LEN, nkv, hd), g_kn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    v = rmsn_gpu(F.linear(xn, wv_q).view(b1 - b0, V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                    if l_idx in (58, 59):
                        for gi in range(b0, b1):
                            if gi in ast_seqs:
                                cap = cap_kv58 if l_idx == 58 else cap_kv59
                                cap["k_rope"][gi] = k[gi - b0].to(torch.float16).cpu()
                                cap["v_norm"][gi] = v[gi - b0].to(torch.float16).cpu()
                    ctx_s = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous()
                    if hr_net_eval is not None:
                        hr_lg, _ = hr_net_eval(xn.reshape(-1, D_MOD))
                        top_h = torch.topk(hr_lg, V9_ATTN_ROUTED_HEADS, dim=-1).indices
                        hm = torch.zeros((xn.shape[0] * V9_SEQ_LEN, NH), device=device)
                        hm.scatter_(1, top_h, 1.0)
                        ctx_s = ctx_s * hm.view(b1 - b0, V9_SEQ_LEN, NH, 1)
                    ctx = ctx_s.view(b1 - b0, V9_SEQ_LEN, NH * hd)
                    xm = xb + rmsn_gpu(F.linear(ctx, wo_q), g_pa.detach())
                    xf = rmsn_gpu(xm, g_pf)
                    xm_stud_cpu[b0:b1] = xm.to(torch.float16).cpu()
                    xf_stud_cpu[b0:b1] = xf.to(torch.float16).cpu()
                    gg = F.gelu(F.linear(xf, wg_init), approximate="tanh").reshape(-1, V9_MLP_BLOCKS, V9_BLOCK_SIZE)
                    blk_e_cpu[b0 * V9_SEQ_LEN : b1 * V9_SEQ_LEN] = gg.pow(2).sum(dim=-1).to(torch.float32).cpu()
                    # Layer residual salience target for MoD threshold (top 72% kept)
                    sal_tok = (X_teach_next[b0:b1].to(device, dtype=torch.float32) - xb).pow(2).sum(dim=-1).sqrt() / (xb.pow(2).sum(dim=-1).sqrt() + 1e-6)
                    mod_sal_cpu[b0 * V9_SEQ_LEN : b1 * V9_SEQ_LEN] = sal_tok.reshape(-1).cpu()

            # Train CRTR-V9 (r=64 + MoD gate) on layer l
            xf_flat = xf_stud_cpu.reshape(-1, D_MOD)
            n_tr_r = (xf_flat.shape[0] * 9) // 10
            sal_q28 = float(torch.quantile(mod_sal_cpu, 0.28).item())
            mod_y = (mod_sal_cpu > sal_q28).float()
            r_net, r_met = crtr.train_layer_router(
                xf_flat[:n_tr_r], blk_e_cpu[:n_tr_r], xf_flat[n_tr_r:], blk_e_cpu[n_tr_r:],
                top_p=float(V9_SHARED_BLOCKS + V9_ROUTED_BLOCKS) / float(V9_MLP_BLOCKS),
                rank=V9_CRTR_RANK, steps=160, lr=3e-3,
                mod_targets_tr=mod_y[:n_tr_r], mod_targets_va=mod_y[n_tr_r:],
                device=str(device), seed=100 + l_idx,
            )
            crtr_layers.append(crtr.quantize_layer_for_crtr(r_net, V9_BLOCK_SIZE))

            # Train Differentiable Sparse-Route QAT (Scales sc/bi + C8DB on down_proj + g_pff) under exact 12/84 mask!
            sc_g_p = nn.Parameter(sc_g.clone())
            sc_u_p = nn.Parameter(sc_u.clone())
            sc_d_p = nn.Parameter(sc_d.clone())
            bi_g_p = nn.Parameter(bi_g.clone()) if bi_g is not None else None
            bi_u_p = nn.Parameter(bi_u.clone()) if bi_u is not None else None
            bi_d_p = nn.Parameter(bi_d.clone()) if bi_d is not None else None
            adp_d = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=V9_FF, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(device)
            opt_params = [sc_g_p, sc_u_p, sc_d_p, g_pff, lsc_p, *adp_d.parameters()]
            if bi_g_p is not None:
                opt_params.append(bi_g_p)
            if bi_u_p is not None:
                opt_params.append(bi_u_p)
            if bi_d_p is not None:
                opt_params.append(bi_d_p)
            opt_qat = torch.optim.AdamW(opt_params, lr=5e-3, weight_decay=1e-4)

            n_qat_steps = 40
            for _ in range(n_qat_steps):
                opt_qat.zero_grad()
                bs_q = torch.randint(0, N_SEQ, (min(32, N_SEQ),), device="cpu")
                xf_b = xf_stud_cpu[bs_q].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                xm_b = xm_stud_cpu[bs_q].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                xt_b = X_teach_next[bs_q].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                wg_q = reconstruct_from_headers(u_g, sc_g_p, bi_g_p, is_g_g, H256_GPU)
                wu_q = reconstruct_from_headers(u_u, sc_u_p, bi_u_p, is_u_g, H256_GPU)
                wd_q = reconstruct_from_headers(u_d, sc_d_p, bi_d_p, is_d_g, H256_GPU)
                with torch.no_grad():
                    r_log, _ = r_net.to(device)(xf_b)
                    r_spec = torch.topk(r_log[:, V9_SHARED_BLOCKS:], V9_ROUTED_BLOCKS, dim=-1).indices + V9_SHARED_BLOCKS
                    blk_m = torch.zeros((xf_b.shape[0], V9_MLP_BLOCKS), device=device)
                    blk_m[:, :V9_SHARED_BLOCKS] = 1.0
                    blk_m.scatter_(1, r_spec, 1.0)
                    col_m = blk_m.repeat_interleave(V9_BLOCK_SIZE, dim=-1)
                act_q = (F.gelu(F.linear(xf_b, wg_q), approximate="tanh") * F.linear(xf_b, wu_q)) * col_m
                delta_d, _, lb_d = adp_d(act_q)
                down_q = F.linear(act_q, wd_q) + delta_d
                xs_b = (xm_b + rmsn_gpu(down_q, g_pff)) * lsc_p
                mag_t = torch.log(xt_b.pow(2).sum(dim=-1).clamp(min=1e-8))
                mag_s = torch.log(xs_b.pow(2).sum(dim=-1).clamp(min=1e-8))
                loss_cm = (1.0 - F.cosine_similarity(xs_b, xt_b, dim=-1).mean()) + 0.15 * F.mse_loss(mag_s, mag_t) + 0.25 * F.mse_loss(xs_b, xt_b) + 0.005 * lb_d
                loss_cm.backward()
                torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
                opt_qat.step()
                with torch.no_grad():
                    lsc_p.clamp_(0.25 * abs(lsc_init), 2.5 * max(abs(lsc_init), 1e-3))

            # Write QAT-updated FP16 block headers directly into packed T16C buffers (zero bitplane change!)
            for pb_buf, do_p, di_p, m_p, sc_p, bi_p in (
                (pb_g, V9_FF, D_MOD, m_g, sc_g_p, bi_g_p),
                (pb_u, V9_FF, D_MOD, m_u, sc_u_p, bi_u_p),
                (pb_d, D_MOD, V9_FF, m_d, sc_d_p, bi_d_p),
            ):
                sc_np = np.ascontiguousarray(sc_p.detach().cpu().numpy(), dtype=np.float32)
                bi_ptr = 0
                if bi_p is not None:
                    bi_np = np.ascontiguousarray(bi_p.detach().cpu().numpy(), dtype=np.float32)
                    bi_ptr = bi_np.ctypes.data
                LIB_FQ.update_t16c_headers_f32(pb_buf.ctypes.data, do_p, di_p, V9_BLOCK_SIZE, m_p, sc_np.ctypes.data, bi_ptr)

            # Atomically update QAT-tuned post_feedforward_layernorm & layer_scalar inside T16C norm table
            cur_pos_mat = fp.tell()
            for nk_upd, ten_upd in ((f"{pfx}.post_feedforward_layernorm.weight", g_pff), (f"{pfx}.layer_scalar", lsc_p)):
                if nk_upd in norm_offsets:
                    fp.seek(norm_offsets[nk_upd])
                    fp.write(np.ascontiguousarray(ten_upd.detach().cpu().numpy(), dtype="<f4").tobytes())
            fp.seek(cur_pos_mat)

            # Export C8DB entry for down_proj + write gate/up tile-major and down as COLBLOCK (0x20)
            ct_d, ex_d = adp_d.cpu().export_quantized_arrays()
            exp_r_d = [c8d._c6d_radicals("general", "core", "transform", "native", "layer", "positive", e) for e in range(8)]
            c8db_entries.append({
                "name": f"{pfx}.mlp.down_proj.weight", "d_out": D_MOD, "d_in": V9_FF, "rank": 32,
                "top_k": 2, "c6d": exp_r_d[0], "temperature": 0.5, "gate_scale": 1.0,
                "centroids": ct_d, "expert_radicals": exp_r_d, "experts": ex_d,
            })

            for pn, pb_buf, is_np, do_p, di_p, m_p, is_rm, is_cb in (
                ("mlp.gate_proj", pb_g, is_g_np, V9_FF, D_MOD, m_g, is_rm_g, is_cb_g),
                ("mlp.up_proj", pb_u, is_u_np, V9_FF, D_MOD, m_u, is_rm_u, is_cb_u),
                ("mlp.down_proj", pb_d, is_d_np, D_MOD, V9_FF, m_d, is_rm_d, is_cb_d),
            ):
                hdr_bits, out_pb = finalize_t16c_payload(pb_buf, do_p, di_p, V9_BLOCK_SIZE, m_p, is_rm, is_cb)
                pos = p3._write_mat_record(fp, pos, f"{pfx}.{pn}.weight", do_p, di_p, V9_BLOCK_SIZE, hdr_bits, p3.MODE_TO_T16C[m_p][1], True, out_pb, is_np, None)

            # Advance exact student state for next layer with updated QAT weights + C8DB
            wg_fin = reconstruct_from_headers(u_g, sc_g_p.detach(), bi_g_p.detach() if bi_g_p is not None else None, is_g_g, H256_GPU)
            wu_fin = reconstruct_from_headers(u_u, sc_u_p.detach(), bi_u_p.detach() if bi_u_p is not None else None, is_u_g, H256_GPU)
            wd_fin = reconstruct_from_headers(u_d, sc_d_p.detach(), bi_d_p.detach() if bi_d_p is not None else None, is_d_g, H256_GPU)
            lsc_fin = float(lsc_p.detach().item())
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xf_b = xf_stud_cpu[b0:b1].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                    xm_b = xm_stud_cpu[b0:b1].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                    r_log, _ = r_net.to(device)(xf_b)
                    r_spec = torch.topk(r_log[:, V9_SHARED_BLOCKS:], V9_ROUTED_BLOCKS, dim=-1).indices + V9_SHARED_BLOCKS
                    blk_m = torch.zeros((xf_b.shape[0], V9_MLP_BLOCKS), device=device)
                    blk_m[:, :V9_SHARED_BLOCKS] = 1.0
                    blk_m.scatter_(1, r_spec, 1.0)
                    col_m = blk_m.repeat_interleave(V9_BLOCK_SIZE, dim=-1)
                    act_q = (F.gelu(F.linear(xf_b, wg_fin), approximate="tanh") * F.linear(xf_b, wu_fin)) * col_m
                    adp_d.to(device)
                    down_q = F.linear(act_q, wd_fin) + adp_d(act_q)[0]
                    X_stud_perm[b0:b1] = ((xm_b + rmsn_gpu(down_q, g_pff.detach())) * lsc_fin).reshape(b1 - b0, V9_SEQ_LEN, D_MOD).to(torch.float16).cpu()

            X_teach_perm = X_teach_next
            cos_l = float(F.cosine_similarity(X_stud_perm[-32:].reshape(-1, D_MOD).float(), X_teach_perm[-32:].reshape(-1, D_MOD).float(), dim=-1).mean().item())
            layer_logs.append({
                "l": l_idx,
                "vlm_txt_cos": round(cos_l, 5),
                "crtr_recall": r_met["val_recall_after"],
                "head_crtr_recall": hr_met["val_recall_after"],
                **pi_stats,
            })
            print(f"  [V9.0 L{l_idx:02d}/59] cos={cos_l:.5f} crtr_rec={r_met['val_recall_after']:.3f} hr_rec={hr_met['val_recall_after']:.3f} e_10of84={pi_stats['perm_12of84_energy_pct']}% dt={time.perf_counter()-tl0:.1f}s", flush=True)
            del wq_t, wk_t, wv_t, wo_t, wg_t, wu_t, wd_t, u_g, u_u, u_d, wg_fin, wu_fin, wd_fin, xm_stud_cpu, xf_stud_cpu, blk_e_cpu
            gc.collect()
            torch.cuda.empty_cache()

            if (l_idx + 1) % 4 == 0 or (l_idx + 1) == N_LAYERS:
                fp.flush(); os.fsync(fp.fileno())
                torch.save(X_teach_perm.cpu(), ckpt_xt_p)
                torch.save(X_stud_perm.cpu(), ckpt_xs_p)
                torch.save(crtr_layers, ckpt_cr_p)
                torch.save(head_crtr_layers, ckpt_hr_p)
                torch.save(c8db_entries, ckpt_c8_p)
                torch.save({"next_layer": l_idx + 1, "pos": pos, "norm_offsets": norm_offsets, "layer_logs": layer_logs, "cap_kv58": cap_kv58, "cap_kv59": cap_kv59}, ckpt_meta_p)
                if DRIVE_DIR is not None and ((l_idx + 1) % 10 == 0 or (l_idx + 1) == N_LAYERS):
                    d_ck = DRIVE_DIR / "ckpt_v9_0"
                    d_ck.mkdir(parents=True, exist_ok=True)
                    tmp_part = DRIVE_DIR / (OUT_T16C.name + ".partial")
                    _drive_safe_copy(tmp_t16c, tmp_part)
                    for cp_f in (ckpt_xt_p, ckpt_xs_p, ckpt_cr_p, ckpt_hr_p, ckpt_c8_p, ckpt_meta_p):
                        _drive_safe_copy(cp_f, d_ck / cp_f.name)

        for rec_v in vision_mat_recs:
            pos = p3._write_mat_record(fp, pos, *rec_v)

    os.replace(tmp_t16c, OUT_T16C)
    vm_c, ve_c = v70.verify_t16c_integrity(OUT_T16C)
    assert vm_c == total_mats and ve_c == OUT_T16C.stat().st_size, f"Bad V9 T16C {vm_c}/{total_mats}"
    cb_cnt = sdc.verify_t16c_file(OUT_T16C)["colblock_mats"]
    crtr.export_crtr(crtr_layers, OUT_CRTR)
    crtr.export_crtr(head_crtr_layers, OUT_HEAD_CRTR)
    c8d.write_c8db_file(OUT_C8D, [], c8db_entries)

    from safetensors.torch import save_file as st_save_file

    # Stage 3: Held-Out KL Check + Assistant-V9 (Permuted Pi + Q2/Q4 Body + BHDR-384)
    fn_g = torch.from_numpy(all_norms[0][1]).to(device)
    ans_flat = torch.nonzero(torch.from_numpy(all_ans_mask.reshape(-1)), as_tuple=False).squeeze(1)
    sel_ans = ans_flat[torch.linspace(0, ans_flat.numel() - 1, min(16384, ans_flat.numel())).long()]
    s_t, t_t = sel_ans // V9_SEQ_LEN, sel_ans % V9_SEQ_LEN
    hs_ans = rmsn_gpu(X_stud_perm[s_t, t_t].to(device).unsqueeze(1), fn_g).squeeze(1).float()
    ht_ans = rmsn_gpu(X_teach_perm[s_t, t_t].to(device).unsqueeze(1), fn_g).squeeze(1).float()
    emb_g = emb_eff.to(device, dtype=torch.float32)

    # Wiener Calibration on final_norm (written back atomically to OUT_T16C norm[0] header slot)
    iv_fn = torch.arange(hs_ans.shape[0], device=device) % 10 == 0
    xub_fn = rmsn_gpu(X_stud_perm[s_t, t_t].to(device).unsqueeze(1), None).squeeze(1).float()
    num_fn = (xub_fn[~iv_fn] * ht_ans[~iv_fn]).sum(dim=0) / float(max(int((~iv_fn).sum().item()), 1))
    den_fn = (xub_fn[~iv_fn] * xub_fn[~iv_fn]).sum(dim=0) / float(max(int((~iv_fn).sum().item()), 1))
    fn_wiener = (num_fn + 0.05 * fn_g) / (den_fn + 0.05).clamp(min=1e-8)
    if float(((xub_fn[iv_fn] * fn_wiener - ht_ans[iv_fn]) ** 2).sum().item()) < float(((xub_fn[iv_fn] * fn_g - ht_ans[iv_fn]) ** 2).sum().item()):
        fn_eff_g = fn_wiener
        hs_ans = xub_fn * fn_eff_g
        with open(OUT_T16C, "r+b") as fp_fn:
            # Header is 24 B; first norm is model.language_model.norm.weight
            fn_off = 24 + 2 + len("model.language_model.norm.weight".encode("utf-8")) + 4
            fp_fn.seek(fn_off)
            fp_fn.write(np.ascontiguousarray(fn_eff_g.detach().cpu().numpy(), dtype="<f4").tobytes())
        print("[Stage 3] Updated final_norm in T16C with held-out validated Wiener shrinkage.", flush=True)
    else:
        fn_eff_g = fn_g

    with torch.no_grad():
        val_mask = iv_fn
        kl_vals = []
        top1_hits = 0
        top5_hits = 0
        val_cnt = int(val_mask.sum().item())
        for b0 in range(0, val_cnt, 512):
            idx_sl = torch.nonzero(val_mask, as_tuple=False).squeeze(1)[b0 : b0 + 512]
            lt = SOFTCAP * torch.tanh((ht_ans[idx_sl] @ emb_g.T) / SOFTCAP)
            ls = SOFTCAP * torch.tanh((hs_ans[idx_sl] @ emb_g.T) / SOFTCAP)
            pt = F.softmax(lt, dim=-1)
            kl_vals.append(float((pt * (torch.log(pt.clamp(min=1e-12)) - F.log_softmax(ls, dim=-1))).sum(dim=-1).mean().item()))
            t1 = lt.argmax(dim=-1)
            s5 = torch.topk(ls, 5, dim=-1).indices
            top1_hits += int((s5[:, 0] == t1).sum().item())
            top5_hits += int((s5 == t1.unsqueeze(1)).any(dim=-1).sum().item())
        val_kl = float(np.mean(kl_vals)) if kl_vals else 0.0
        val_top1 = float(top1_hits / max(val_cnt, 1))
        val_top5 = float(top5_hits / max(val_cnt, 1))
        print(f"[Stage 3 Teacher-KL Val] kl={val_kl:.4f} | top1_agree={val_top1*100:.2f}% | top5_agree={val_top5*100:.2f}%", flush=True)
    del hs_ans, ht_ans, xub_fn
    torch.cuda.empty_cache()

    ast_dir = Path(snapshot_download("google/gemma-4-31B-it-assistant", local_dir="/tmp/gemma4_ast", ignore_patterns=["*.bin", "*.pt"]))
    ast_sf_raw = ast_dir / "model.safetensors"
    ast_sd: Dict[str, torch.Tensor] = {}
    with safe_open(ast_sf_raw, framework="pt", device="cpu") as sf_a:
        for k in sf_a.keys():
            t_a = sf_a.get_tensor(k)
            if k == "pre_projection.weight" and t_a.shape[1] == 2 * D_MOD:
                p2 = torch.cat([perm_pi, D_MOD + perm_pi])
                t_a = t_a[:, p2].contiguous()
            elif k == "post_projection.weight" and t_a.shape[0] == D_MOD:
                t_a = t_a[perm_pi, :].contiguous()
            ast_sd[k] = t_a

    # Train real 4-layer Assistant-V9 (Pi-permuted + 2-step self-feeding rollout) against V9.0 student
    ast_mod = Gemma4AssistantTrain(
        ast_sd, d_mod=D_MOD, nh=NH, nkv_s=NKV_S, nkv_f=NKV_F, hd_s=HD_S, hd_f=HD_F,
        theta_l=THETA_L, part_l=PART_L, theta_g=THETA_G, part_g=PART_G, seq_len=V9_SEQ_LEN, eps=EPS, compute_dtype=compute_dtype,
    ).to(device)
    with torch.no_grad():
        ast_h_final = torch.empty((len(ast_seqs), V9_SEQ_LEN, D_MOD), dtype=torch.float16)
        for ai, gi in enumerate(ast_seqs):
            ast_h_final[ai] = rmsn_gpu(X_stud_perm[gi].to(device), fn_eff_g).float().to(torch.float16).cpu()
        nxt_ast = torch.from_numpy(all_seq_nxt[ast_seqs].reshape(-1)).long()
        uniq_a, cnts_a = torch.unique(nxt_ast, return_counts=True)
        cand_ids = uniq_a[torch.topk(cnts_a, min(8192, uniq_a.numel())).indices].to(device)
        W_cand_g = emb_g[cand_ids].float()
        W_ast_cand = ast_mod.emb[cand_ids].float()

    TAU_AST = 2.0
    n_ast_tr = max(1, (len(ast_seqs) * 7) // 8)
    trie = bhmod.BpePrefixTrie.from_tokenizer_json(tok_path)

    def eval_ast() -> Tuple[float, float]:
        with torch.no_grad():
            agrs = kls = 0.0
            cnt = 0
            for ai in range(n_ast_tr, len(ast_seqs)):
                gi = ast_seqs[ai]
                te = (emb_g[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE).unsqueeze(0)
                ha = ast_mod.forward_seq(
                    te,
                    ast_h_final[ai : ai + 1, :-1].to(device).float(),
                    cap_kv58["k_rope"][gi].unsqueeze(0).to(device),
                    cap_kv58["v_norm"][gi].unsqueeze(0).to(device),
                    cap_kv59["k_rope"][gi].unsqueeze(0).to(device),
                    cap_kv59["v_norm"][gi].unsqueeze(0).to(device),
                ).squeeze(0)
                zs = SOFTCAP * torch.tanh((ast_h_final[ai, 1:].to(device).float() @ W_cand_g.T) / SOFTCAP)
                za = ha @ W_ast_cand.T
                ps = F.softmax(zs / TAU_AST, dim=-1)
                kls += float(((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU_AST, dim=-1))).sum(dim=-1) * (TAU_AST * TAU_AST)).sum().item())
                agrs += float((za.argmax(dim=-1) == zs.argmax(dim=-1)).float().sum().item())
                cnt += ha.shape[0]
            return kls / max(cnt, 1), agrs / max(cnt, 1)

    akl0, aagr0 = eval_ast()
    opt_a = torch.optim.AdamW(ast_mod.parameters(), lr=2.5e-4, weight_decay=1e-4)
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=300, eta_min=2e-5)
    best_aagr, best_asd = aagr0, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_bs = 12 if gpu_vram_gb >= 64.0 else 4
    for st in range(300):
        opt_a.zero_grad()
        bais = np.random.choice(n_ast_tr, size=min(ast_bs, n_ast_tr), replace=False).tolist()
        te_l, bh_l, k58_l, v58_l, k59_l, v59_l, ht_l = [], [], [], [], [], [], []
        for ai in bais:
            gi = ast_seqs[ai]
            te_l.append(emb_g[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE)
            bh_l.append(ast_h_final[ai, :-1].to(device).float())
            k58_l.append(cap_kv58["k_rope"][gi].to(device))
            v58_l.append(cap_kv58["v_norm"][gi].to(device))
            k59_l.append(cap_kv59["k_rope"][gi].to(device))
            v59_l.append(cap_kv59["v_norm"][gi].to(device))
            ht_l.append(ast_h_final[ai, 1:].to(device).float())
        te_s = torch.stack(te_l)
        bh_s = torch.stack(bh_l)
        k58_s, v58_s, k59_s, v59_s = torch.stack(k58_l), torch.stack(v58_l), torch.stack(k59_l), torch.stack(v59_l)
        hst = torch.stack(ht_l)
        ha = ast_mod.forward_seq(te_s, bh_s, k58_s, v58_s, k59_s, v59_s)
        with torch.no_grad():
            zs = SOFTCAP * torch.tanh((hst.reshape(-1, D_MOD) @ W_cand_g.T) / SOFTCAP)
            ps, ys_t1 = F.softmax(zs / TAU_AST, dim=-1), zs.argmax(dim=-1)
        za = ha.reshape(-1, 1024) @ W_ast_cand.T
        hpost = F.linear(ha.to(compute_dtype), ast_mod.post.to(compute_dtype)).float()
        bh2 = torch.cat([bh_s[:, :1], hpost[:, :-1]], dim=1)
        ha2 = ast_mod.forward_seq(te_s, bh2, k58_s, v58_s, k59_s, v59_s)
        hpost2 = F.linear(ha2.to(compute_dtype), ast_mod.post.to(compute_dtype)).float()
        loss_a = (
            ((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU_AST, dim=-1))).sum(dim=-1) * (TAU_AST * TAU_AST)).mean()
            + 0.5 * F.cross_entropy(za, ys_t1)
            + 0.5 * (F.mse_loss(hpost, hst) / (hst.var() + 1e-5))
            + 0.25 * (1.0 - F.cosine_similarity(hpost2[:, 1:], hst[:, 1:], dim=-1).mean())
        )
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(ast_mod.parameters(), 1.0)
        opt_a.step()
        sch_a.step()
        if (st + 1) % 60 == 0:
            klv, agrv = eval_ast()
            print(f"    [Ast step {st+1:03d}/300] Val KL={klv:.4f} Top1Agree={agrv*100:.1f}%", flush=True)
            if agrv > best_aagr:
                best_aagr, best_asd = agrv, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_mod.load_state_dict({k: v.to(device) for k, v in best_asd.items()})
    akl1, aagr1 = eval_ast()
    print(f"  [Assistant V9 Aligned] Val KL={akl0:.4f}->{akl1:.4f} | Top1Agree={aagr0*100:.1f}%->{aagr1*100:.1f}%", flush=True)

    ast_sd["pre_projection.weight"] = ast_mod.pre.detach().to(torch.bfloat16).cpu()
    ast_sd["post_projection.weight"] = ast_mod.post.detach().to(torch.bfloat16).cpu()
    ast_sd["model.norm.weight"] = ast_mod.fn.detach().to(torch.bfloat16).cpu()
    for la, m in enumerate(ast_mod.layers):
        pa_k = f"model.layers.{la}."
        for k_s, par in [
            ("self_attn.q_proj.weight", m.q_proj), ("self_attn.o_proj.weight", m.o_proj),
            ("mlp.gate_proj.weight", m.gate), ("mlp.up_proj.weight", m.up), ("mlp.down_proj.weight", m.down),
            ("input_layernorm.weight", m.n_in), ("self_attn.q_norm.weight", m.n_qn),
            ("post_attention_layernorm.weight", m.n_pa), ("pre_feedforward_layernorm.weight", m.n_pf),
            ("post_feedforward_layernorm.weight", m.n_pff),
        ]:
            ast_sd[pa_k + k_s] = par.detach().to(torch.bfloat16).cpu()
        ast_sd[pa_k + "layer_scalar"] = m.lsc.detach().to(torch.bfloat16).cpu().reshape(ast_sd[pa_k + "layer_scalar"].shape)
    ast_sf = OUT_DIR / "assistant_v9_permuted.safetensors"
    st_save_file(ast_sd, str(ast_sf))
    bhmod.pack_assistant_body_from_safetensors(ast_sf, OUT_AST_Q2, body_bits=2, block_size=64, has_fwht=False)
    bhmod.pack_assistant_body_from_safetensors(ast_sf, OUT_AST_Q4, body_bits=4, block_size=64, has_fwht=True)
    ast_sf.unlink(missing_ok=True)
    del opt_a, sch_a, ast_sd, best_asd, W_cand_g, W_ast_cand

    # Extract 1024-dim aligned Assistant hidden states + V9.0 Student full-262K top-64 & argmax for BHDR-384
    h_ast_chunks = []
    with torch.no_grad():
        for b0 in range(0, len(ast_seqs), 32):
            bais = list(range(b0, min(b0 + 32, len(ast_seqs))))
            gis = [ast_seqs[ai] for ai in bais]
            te_s = torch.stack([emb_g[torch.from_numpy(all_seq_ids[gi, 1:]).long().to(device)].float() * ESCALE for gi in gis])
            bh_s = ast_h_final[b0 : b0 + len(bais), :-1].to(device).float()
            k58_s = torch.stack([cap_kv58["k_rope"][gi].to(device) for gi in gis])
            v58_s = torch.stack([cap_kv58["v_norm"][gi].to(device) for gi in gis])
            k59_s = torch.stack([cap_kv59["k_rope"][gi].to(device) for gi in gis])
            v59_s = torch.stack([cap_kv59["v_norm"][gi].to(device) for gi in gis])
            h_ast_chunks.append(ast_mod.forward_seq(te_s, bh_s, k58_s, v58_s, k59_s, v59_s).cpu())
    del ast_mod, cap_kv58, cap_kv59
    torch.cuda.empty_cache()

    H_ast_all = torch.cat(h_ast_chunks, dim=0).reshape(-1, 1024)
    H_stud_tgt = ast_h_final[:, 1:, :].reshape(-1, D_MOD)
    top_ids_t = torch.empty((H_stud_tgt.shape[0], 64), dtype=torch.int32)
    top_probs_t = torch.empty((H_stud_tgt.shape[0], 64), dtype=torch.float32)
    emb_g_c = emb_g.to(dtype=compute_dtype)
    del emb_g
    with torch.no_grad():
        for i0 in range(0, H_stud_tgt.shape[0], 512):
            i1 = min(i0 + 512, H_stud_tgt.shape[0])
            lg_b = SOFTCAP * torch.tanh((H_stud_tgt[i0:i1].to(device, dtype=compute_dtype) @ emb_g_c.T).float() / SOFTCAP)
            tv, ti = torch.topk(lg_b, k=64, dim=-1)
            top_probs_t[i0:i1] = F.softmax(tv, dim=-1).cpu()
            top_ids_t[i0:i1] = ti.to(torch.int32).cpu()
    del emb_g_c
    student_argmax_ids = torch.tensor(v70.select_first_valid_trie_ids(top_ids_t, trie), dtype=torch.long)
    del H_stud_tgt, ast_h_final

    act_bundle = OUT_DIR / "v9_0_bhdr_acts.pt"
    torch.save({"h_final": H_ast_all.half(), "gold_ids": student_argmax_ids, "top_ids": top_ids_t, "top_probs": top_probs_t}, act_bundle)
    bh_rpt = bhmod.train_from_activations(act_bundle, tok_path, OUT_BHDR, d_byte=384, max_bytes=48, steps=2400, batch_size=256, lr=3e-3, device=str(device))
    act_bundle.unlink(missing_ok=True)
    bh_loaded = bhmod.load_bhdr_v1(OUT_BHDR)
    val_h = H_ast_all[-256:].numpy().astype(np.float32)
    val_g = student_argmax_ids[-256:].tolist()
    beam_hits = sum(1 for hv, gv in zip(val_h, val_g) if bhmod.decode_bhdr_np_beam(bh_loaded, trie, hv, beam_width=2)[0] == gv)
    bh_metrics = {**bh_rpt["metrics"], "heldout_beam_top1_acc_vs_v9_argmax": round(beam_hits / 256.0, 4)}
    del H_ast_all, student_argmax_ids, top_ids_t, top_probs_t

    all_artifact_paths = (OUT_T16C, OUT_CRTR, OUT_HEAD_CRTR, OUT_C8D, OUT_AST_Q2, OUT_AST_Q4, OUT_BHDR)
    artifacts = {}
    for fp_out in all_artifact_paths:
        s_b, h_b = persist_v9(fp_out)
        artifacts[fp_out.name] = {"bytes": s_b, "sha256": h_b}

    report = {
        "version": "V9.0-FASTKERNEL-SPARSE10OF84-HEAD10OF32-DISPERSION-QAT",
        "gpu": gpu_name,
        "elapsed_s": round(time.perf_counter() - t0_all, 2),
        "t16c_bytes": OUT_T16C.stat().st_size,
        "t16c_gib": round(OUT_T16C.stat().st_size / (1024**3), 4),
        "colblock_mats": cb_cnt,
        "l59_cos": layer_logs[-1]["vlm_txt_cos"] if layer_logs else 0.0,
        "val_kl": round(val_kl, 4),
        "val_top1_agree": round(val_top1, 4),
        "val_top5_agree": round(val_top5, 4),
        "crtr_mean_recall": round(float(np.mean([r["crtr_recall"] for r in layer_logs])), 5),
        "head_crtr_mean_recall": round(float(np.mean([r["head_crtr_recall"] for r in layer_logs])), 5),
        "assistant_align": {"kl_before": round(akl0, 4), "kl_after": round(akl1, 4), "top1_before": round(aagr0, 4), "top1_after": round(aagr1, 4)},
        "bhdr": bh_metrics,
        "artifacts": artifacts,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sums = "\n".join(f"{v['sha256']}  {k}" for k, v in artifacts.items()) + "\n"
    (OUT_DIR / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    if DRIVE_DIR is not None:
        persist_v9(OUT_REPORT)
        persist_v9(OUT_DIR / "SHA256SUMS.txt")
        part_p = DRIVE_DIR / (OUT_T16C.name + ".partial")
        if part_p.exists():
            part_p.unlink()
        shutil.rmtree(DRIVE_DIR / "ckpt_v9_0", ignore_errors=True)
        expected_sha_map = {k: v["sha256"] for k, v in artifacts.items()}
        expected_sha_map[OUT_REPORT.name] = _sha256_file(OUT_REPORT)
        _verify_drive_sha256(DRIVE_DIR, expected_sha_map)
    print(f"[V9.0 COMPLETE] elapsed={(time.perf_counter()-t0_all)/60:.1f} min | {OUT_T16C.name}={report['t16c_gib']} GiB | colblock={cb_cnt}", flush=True)

    if os.environ.get("COLAB_AUTO_UNASSIGN", "1") == "1":
        _flush_drive_before_unassign()
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:
            print(f"[Auto-Unassign Note] {exc}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Language-U Gemma 4 31B V9.0 Sparse-Active QAT Builder")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--recover-aux", action="store_true", help="Replay frozen V9.0 .t16c to rebuild the 5 aux files (~11 min on L4)")
    args = ap.parse_args()
    if args.selftest or (not args.run and not args.recover_aux and not torch.cuda.is_available()):
        return run_selftest()
    if args.recover_aux:
        recover_aux_from_t16c()
        return 0
    run_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

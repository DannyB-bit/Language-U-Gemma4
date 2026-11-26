#!/usr/bin/env python3
"""Language-U V9.3: Zero-Tern5 / Zero-Cliff + True 60-Layer E2E Logit-KL QAT & Matched EAGLE-3 Assistant.

Eliminates ALL slow Ternary-5 kernels across 60 layers (replacing all 58 Attention layers
L0..L57 and upper stack MLPs L36..L57 with fast COLBLOCK 2b_gauss 0xA2, while keeping
resilient L2..L35 MLPs in COLBLOCK 1b_rm 0x61 r=16 and boundary L58..L59 in 3b), then
runs genuine 60-layer End-to-End Differentiable Teacher Top-128 Logit-KL + Cosine QAT
via custom memory-constant `_BitSerialLinearFn` + gradient checkpointing optimizing all
FP32 block scales `sc`, asymmetric biases `bi`, STE INT8 low-rank residuals `(U, V)`,
RMSNorms, `layer_scalar`, and `final_norm`.

Finally replays the Stage-2.5 E2E-tuned student to distill target-matched `HeadRouter`,
`CRTR-V9`, `C8DB`, and `Assistant-V9` (`Q2/Q4 + BHDR-384`), enforces Rule 2 strict 0-overlap
decontamination, Rule 3 authentic PyTorch backprop only, and 6-gate CPU `--selftest`.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import shutil
import struct
import sys
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
import run_v7_0_fullrank_crtr_gpu as v70  # noqa: E402
import run_v9_0_sparse_spinquant_qat_gpu as v90  # noqa: E402
import spatial_district_converter as sdc  # noqa: E402
import train_c8d_pattern_bank as c8d  # noqa: E402
import train_cortical_router as crtr  # noqa: E402

V93_BLOCK_SIZE = v90.V9_BLOCK_SIZE
V93_D_MOD = v90.V9_D_MOD
V93_FF = v90.V9_FF
V93_N_LAYERS = v90.V9_N_LAYERS
V93_MLP_BLOCKS = v90.V9_MLP_BLOCKS
V93_SHARED_BLOCKS = v90.V9_SHARED_BLOCKS
V93_CRTR_TOP_BLOCKS = 42   # 50% MLP sparse routing target for high-accuracy Pi 4 speedup
V93_HEAD_ROUTED = 20       # 62.5% (20/32) HeadRouter target
GLOBAL_ATTN_LAYERS = set(v90.GLOBAL_ATTN_LAYERS)

# Zero Ternary-5 anywhere; Zero 1b_rm cliff above L35; Entry L0..1 + L36..57 in 2b_gauss COLBLOCK
V93_2B_MLP_LAYERS = {0, 1} | set(range(36, 58))
MAX_V93_T16C_BYTES = int(6.85 * (1024**3))

OUT_DIR = Path(os.environ.get("V9_3_OUT_DIR", "/content/v9_3_outputs"))
CKPT_DIR = OUT_DIR / "ckpt_v9_3"
_drv_env = os.environ.get("V9_3_DRIVE_DIR", "").strip()
DRIVE_DIR: Optional[Path] = Path(_drv_env) if _drv_env else None

OUT_T16C = OUT_DIR / "gemma4_31b_v9_3_e2e.t16c"
OUT_CRTR = OUT_DIR / "gemma4_31b_v9_3_router.crtr"
OUT_HEAD_CRTR = OUT_DIR / "gemma4_31b_v9_3_head_router.crtr"
OUT_C8D = OUT_DIR / "gemma4_31b_v9_3_c8d.c8d"
OUT_AST_Q2 = OUT_DIR / "gemma4_31b_v9_3_assistant_body_q2.t16c"
OUT_AST_Q4 = OUT_DIR / "gemma4_31b_v9_3_assistant_body_q4.t16c"
OUT_BHDR = OUT_DIR / "gemma4_31b_v9_3_assistant.bhdr"
OUT_REPORT = OUT_DIR / "v9_3_report.json"


def get_v93_proj_spec(l_idx: int, pname: str, is_full: Optional[bool] = None) -> Tuple[int, int, int, bool, bool]:
    """Return `(mode_i, block_size, lr_rank, is_rowmajor, is_colblock)` for V9.3 Zero-Tern5 Zero-Cliff layout."""
    if l_idx >= 58:
        lr_rk = 24 if ("k_proj" in pname or "v_proj" in pname) else 16
        return (4, V93_BLOCK_SIZE, lr_rk, False, ("o_proj" in pname or "down_proj" in pname))
    if pname.startswith("self_attn."):
        lr_rk = 16 if ("q_proj" in pname or "o_proj" in pname) else 8
        return (5, V93_BLOCK_SIZE, lr_rk, False, True)
    if l_idx in V93_2B_MLP_LAYERS:
        return (5, V93_BLOCK_SIZE, 16, False, True)
    return (0, V93_BLOCK_SIZE, 16, True, True)


def compute_v93_budget_bytes(
    *,
    vocab_size: int = 262144,
    d_mod: int = V93_D_MOD,
    n_layers: int = V93_N_LAYERS,
    nh: int = 32,
    nkv_s: int = 16,
    nkv_f: int = 4,
    hd_s: int = 256,
    hd_f: int = 512,
    ff: int = V93_FF,
    vision_bytes: int = 237_253_152,
) -> Dict[str, object]:
    """Exact fail-closed byte census for V9.3 `.t16c` (Zero Tern5, Zero cliff above L35)."""
    embed_b = p3.expected_tile16_bytes(vocab_size, d_mod, V93_BLOCK_SIZE, 3, False) + 256
    attn_b = 0
    mlp_b = 0
    lr_b = 0
    tern5_cnt = 0
    attn_1b_cnt = 0
    mlp_1b_upper_cnt = 0
    mlp_2b_layers = 0
    mlp_1b_layers = 0

    for l in range(n_layers):
        is_f = l in GLOBAL_ATTN_LAYERS
        nkv, hd = (nkv_f, hd_f) if is_f else (nkv_s, hd_s)
        q_dim, kv_dim = nh * hd, nkv * hd
        attn_shapes: List[Tuple[str, int, int]] = [
            ("self_attn.q_proj", q_dim, d_mod),
            ("self_attn.k_proj", kv_dim, d_mod),
            ("self_attn.o_proj", d_mod, q_dim),
        ]
        if not is_f:
            attn_shapes.append(("self_attn.v_proj", kv_dim, d_mod))
        for pn, do, di in attn_shapes:
            mode, bsz, lrr, _, _ = get_v93_proj_spec(l, pn, is_f)
            nb, asym = p3.MODE_TO_T16C[mode]
            if mode == 1:
                tern5_cnt += 1
            if (nb & 0x1F) == 1:
                attn_1b_cnt += 1
            attn_b += p3.expected_tile16_bytes(do, di, bsz, nb, asym) + 128 + di * 4
            if lrr > 0:
                lr_b += do * 4 + lrr * 4 + do * lrr + lrr * di

        m_gate, _, _, _, _ = get_v93_proj_spec(l, "mlp.gate_proj", is_f)
        if m_gate == 5:
            mlp_2b_layers += 1
        elif m_gate == 0:
            mlp_1b_layers += 1
            if l > 35:
                mlp_1b_upper_cnt += 1
        for pn, do, di in (
            ("mlp.gate_proj", ff, d_mod),
            ("mlp.up_proj", ff, d_mod),
            ("mlp.down_proj", d_mod, ff),
        ):
            mode, bsz, lrr, _, _ = get_v93_proj_spec(l, pn, is_f)
            nb, asym = p3.MODE_TO_T16C[mode]
            if mode == 1:
                tern5_cnt += 1
            mlp_b += p3.expected_tile16_bytes(do, di, bsz, nb, asym) + 128 + di * 4
            if lrr > 0:
                lr_b += do * 4 + lrr * 4 + do * lrr + lrr * di

    total_b = embed_b + vision_bytes + attn_b + mlp_b + lr_b
    text_res_b = total_b - vision_bytes
    return {
        "embed_bytes": embed_b,
        "vision_bytes": vision_bytes,
        "attn_bytes": attn_b,
        "mlp_bytes": mlp_b,
        "low_rank_bytes": lr_b,
        "total_t16c_bytes": total_b,
        "total_t16c_gib": round(total_b / (1024**3), 4),
        "text_resident_gib": round(text_res_b / (1024**3), 4),
        "tern5_cnt": tern5_cnt,
        "attn_1b_cnt": attn_1b_cnt,
        "mlp_1b_upper_cnt": mlp_1b_upper_cnt,
        "mlp_2b_gauss_layers": mlp_2b_layers,
        "mlp_1b_rm_layers": mlp_1b_layers,
    }


def extract_planes_u8(pb_buf: np.ndarray, d_out: int, d_in: int, bsize: int, mode_i: int) -> torch.Tensor:
    """Slice packed `[n_tiles, n_blks, eff_bits, bpr, 16]` uint8 bitplanes from raw tile16 column-major payload."""
    nb_m, asym_m = p3.MODE_TO_T16C[mode_i]
    eff_bits = nb_m & 0x1F
    if eff_bits == 5 or eff_bits < 1 or eff_bits > 4:
        raise ValueError(f"Unsupported bitplane mode_i={mode_i} (eff_bits={eff_bits})")
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    bpr = bsize // 8
    hdr_b = 64 if asym_m else 32
    tblk_b = p3.tile16_blk_bytes(bsize, nb_m, asym_m)
    arr = np.frombuffer(pb_buf, dtype=np.uint8).reshape(n_tiles, n_blks, tblk_b)
    planes = np.ascontiguousarray(arr[:, :, hdr_b:].reshape(n_tiles, n_blks, eff_bits, bpr, 16))
    return torch.from_numpy(planes)


def make_unpack_tables(device: torch.device, dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Build GPU lookup tables `[256, 8]` bit-expansion + mode-specific level LUTs."""
    tbl_np = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="big")
    bits_tbl = torch.from_numpy(tbl_np).to(device=device, dtype=torch.uint8)
    lv_tbls = {
        0: torch.tensor([-1.0, 1.0], device=device, dtype=dtype),
        5: torch.tensor([-3.174, -1.000, 1.000, 3.174], device=device, dtype=dtype),
        6: torch.tensor([-7.430, -5.430, -3.270, -1.270, 1.270, 3.270, 5.430, 7.430], device=device, dtype=dtype),
        4: torch.tensor([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], device=device, dtype=dtype),
    }
    return bits_tbl, lv_tbls


def unpack_planes_to_units(
    planes_u8: torch.Tensor,
    mode_i: int,
    d_out: int,
    d_in: int,
    bsize: int,
    bits_tbl: torch.Tensor,
    lv_tbl: torch.Tensor,
) -> torch.Tensor:
    """Vectorized bitplane decode -> `[d_out, n_blks, bsize]` matching `v90.unpack_tile16_headers_and_units` bit-exact."""
    n_tiles, n_blks, eff_bits, bpr, _ = planes_u8.shape
    qi = bits_tbl[planes_u8[:, :, 0].long()].clone()
    for p in range(1, eff_bits):
        qi.bitwise_or_(bits_tbl[planes_u8[:, :, p].long()] << p)
    qi_flat = qi.permute(0, 3, 1, 2, 4).reshape(n_tiles * 16, n_blks, bsize)[:d_out]
    return lv_tbl[qi_flat.long()]


def ste_int8_lr(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Straight-through per-row symmetric INT8 quantization for low-rank residual factors `(U, V)`."""
    u_s = (u.detach().abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    v_s = (v.detach().abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    u_q = torch.clamp(torch.round(u.detach() / u_s), -127.0, 127.0) * u_s
    v_q = torch.clamp(torch.round(v.detach() / v_s), -127.0, 127.0) * v_s
    return u + (u_q - u.detach()), v + (v_q - v.detach())


def quant_lr_int8_tuple(u: torch.Tensor, v: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Export `(su, sv, u_i8, v_i8)` tuple matching `.t16c` binary format bit-accurately."""
    ud, vd = u.detach().float(), v.detach().float()
    su = (ud.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    sv = (vd.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    ui8 = torch.clamp(torch.round(ud / su.unsqueeze(1)), -127, 127).to(torch.int8)
    vi8 = torch.clamp(torch.round(vd / sv.unsqueeze(1)), -127, 127).to(torch.int8)
    return (
        su.cpu().numpy().astype("<f4"),
        sv.cpu().numpy().astype("<f4"),
        ui8.cpu().numpy().astype(np.int8),
        vi8.cpu().numpy().astype(np.int8),
    )


def _assemble_w_eff(
    units: torch.Tensor,
    sc: torch.Tensor,
    bi: Optional[torch.Tensor],
    inv_s: Optional[torch.Tensor],
    h_mat: torch.Tensor,
) -> torch.Tensor:
    do, n_blks, bsize = units.shape
    w_rot = units * sc.to(units.dtype).unsqueeze(-1)
    if bi is not None:
        w_rot = w_rot + bi.to(units.dtype).unsqueeze(-1)
    w = (w_rot.reshape(-1, bsize) @ h_mat).reshape(do, n_blks * bsize)
    if inv_s is not None:
        w = w * inv_s.unsqueeze(0)
    return w


class _BitSerialLinearFn(torch.autograd.Function):
    """Exact differentiable bit-serial linear layer that discards dense weights between layers."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        sc: torch.Tensor,
        bi: Optional[torch.Tensor],
        planes_u8: torch.Tensor,
        inv_s: Optional[torch.Tensor],
        h_mat: torch.Tensor,
        bits_tbl: torch.Tensor,
        lv_tbl: torch.Tensor,
        mode_i: int,
    ) -> torch.Tensor:
        dev = x.device
        dt = x.dtype
        do, n_blks = sc.shape
        bsize = int(h_mat.shape[0])
        di = n_blks * bsize
        p_dev = planes_u8 if planes_u8.device == dev else planes_u8.to(dev, non_blocking=True)
        inv_dev = None if inv_s is None else (inv_s if inv_s.device == dev else inv_s.to(dev, non_blocking=True))
        with torch.no_grad():
            units = unpack_planes_to_units(p_dev, mode_i, do, di, bsize, bits_tbl, lv_tbl)
            w_eff = _assemble_w_eff(units, sc.detach(), bi.detach() if bi is not None else None, inv_dev, h_mat)
            y = F.linear(x, w_eff)
            del units, w_eff
        ctx.save_for_backward(x, sc, bi if bi is not None else torch.empty(0, device=dev))
        ctx.has_bi = bi is not None
        ctx.planes_u8 = planes_u8
        ctx.inv_s = inv_s
        ctx.h_mat = h_mat
        ctx.bits_tbl = bits_tbl
        ctx.lv_tbl = lv_tbl
        ctx.mode_i = mode_i
        return y.to(dt)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, sc, bi_saved = ctx.saved_tensors
        bi = bi_saved if ctx.has_bi else None
        dev = x.device
        dt = x.dtype
        do, n_blks = sc.shape
        bsize = int(ctx.h_mat.shape[0])
        di = n_blks * bsize
        p_dev = ctx.planes_u8 if ctx.planes_u8.device == dev else ctx.planes_u8.to(dev, non_blocking=True)
        inv_dev = None if ctx.inv_s is None else (ctx.inv_s if ctx.inv_s.device == dev else ctx.inv_s.to(dev, non_blocking=True))
        gy_f = grad_y.reshape(-1, do).to(dt)
        x_f = x.reshape(-1, di).to(dt)

        grad_x = grad_sc = grad_bi = None
        with torch.no_grad():
            units = unpack_planes_to_units(p_dev, ctx.mode_i, do, di, bsize, ctx.bits_tbl, ctx.lv_tbl)
            if ctx.needs_input_grad[0]:
                w_eff = _assemble_w_eff(units, sc.detach(), bi.detach() if bi is not None else None, inv_dev, ctx.h_mat)
                grad_x = (gy_f @ w_eff).reshape_as(x)
                del w_eff
            if ctx.needs_input_grad[1] or (ctx.has_bi and ctx.needs_input_grad[2]):
                x_s = x_f * inv_dev.unsqueeze(0) if inv_dev is not None else x_f
                # H256 is symmetric orthonormal (H^T == H)
                x_tilde = (x_s.reshape(-1, bsize) @ ctx.h_mat).reshape(-1, di)
                gw_rot = (gy_f.T @ x_tilde).reshape(do, n_blks, bsize)
                if ctx.needs_input_grad[1]:
                    grad_sc = (gw_rot * units).sum(dim=-1).to(sc.dtype)
                if ctx.has_bi and ctx.needs_input_grad[2]:
                    grad_bi = gw_rot.sum(dim=-1).to(bi.dtype)
                del x_s, x_tilde, gw_rot
            del units
        return grad_x, grad_sc, grad_bi, None, None, None, None, None, None


def bitserial_linear(
    x: torch.Tensor,
    proj: dict,
    bits_tbl: torch.Tensor,
    lv_tbls: Dict[int, torch.Tensor],
    h_mat: torch.Tensor,
    use_ste_lr: bool = True,
) -> torch.Tensor:
    """Execute differentiable bit-serial linear projection + STE INT8 low-rank residual."""
    dt = h_mat.dtype
    x_c = x.to(dt)
    y = _BitSerialLinearFn.apply(
        x_c,
        proj["sc"],
        proj["bi"],
        proj["planes_u8"],
        proj["inv_s"],
        h_mat,
        bits_tbl,
        lv_tbls[proj["mode_i"]],
        proj["mode_i"],
    )
    if proj["u_lr"] is not None:
        u_l, v_l = ste_int8_lr(proj["u_lr"], proj["v_lr"]) if use_ste_lr else (proj["u_lr"], proj["v_lr"])
        y = y + F.linear(F.linear(x_c, v_l.to(dt)), u_l.to(dt))
    return y.float()


def compute_topk_kl_loss(
    logits_s: torch.Tensor,
    top_ids: torch.Tensor,
    top_probs: torch.Tensor,
    tok_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute strictly non-negative Teacher Top-K + Tail-Bucket KL divergence."""
    lg = logits_s.float()
    log_z = torch.logsumexp(lg, dim=-1, keepdim=True)
    stud_lp = lg.gather(-1, top_ids.long()) - log_z
    tp = top_probs.float()
    tp = tp / tp.sum(dim=-1, keepdim=True).clamp(min=1.0)
    kl_top = (tp * (torch.log(tp.clamp(min=1e-9)) - stud_lp)).sum(dim=-1)
    p_tail = (1.0 - tp.sum(dim=-1)).clamp(min=0.0)
    q_top_sum = torch.exp(stud_lp).sum(dim=-1).clamp(max=1.0 - 1e-7)
    log_q_tail = torch.log((1.0 - q_top_sum).clamp(min=1e-7))
    kl_tok = (kl_top + p_tail * (torch.log(p_tail.clamp(min=1e-9)) - log_q_tail)).clamp(min=0.0)
    if tok_weights is not None:
        return (kl_tok * tok_weights.float()).sum() / tok_weights.float().sum().clamp(min=1.0)
    return kl_tok.mean()


def forward_layer_e2e(
    x: torch.Tensor,
    lyr: dict,
    *,
    is_f: bool,
    k_eq_v: bool,
    nh: int,
    nkv: int,
    hd: int,
    rot_d: int,
    cs_r: torch.Tensor,
    sn_r: torch.Tensor,
    rmsn_fn,
    apply_rope_fn,
    bits_tbl: torch.Tensor,
    lv_tbls: Dict[int, torch.Tensor],
    h_mat: torch.Tensor,
    head_keep_frac: float = 1.0,
    mlp_keep_frac: float = 1.0,
    use_ste_lr: bool = True,
) -> torch.Tensor:
    """Exact Gemma 4 31B layer forward matching `run31b` with optional structured sparsity simulation."""
    bsz, seq_len, d_mod = x.shape
    rep = nh // nkv
    xn = rmsn_fn(x, lyr["g_in"])
    q_lin = bitserial_linear(xn, lyr["projs"]["self_attn.q_proj"], bits_tbl, lv_tbls, h_mat, use_ste_lr)
    k_lin = bitserial_linear(xn, lyr["projs"]["self_attn.k_proj"], bits_tbl, lv_tbls, h_mat, use_ste_lr)
    q = apply_rope_fn(rmsn_fn(q_lin.view(bsz, seq_len, nh, hd), lyr["g_qn"]), cs_r, sn_r, rot_d, hd).transpose(1, 2)
    k = apply_rope_fn(rmsn_fn(k_lin.view(bsz, seq_len, nkv, hd), lyr["g_kn"]), cs_r, sn_r, rot_d, hd).transpose(1, 2)
    if is_f and k_eq_v:
        v = rmsn_fn(k_lin.view(bsz, seq_len, nkv, hd), None).transpose(1, 2)
    else:
        v_lin = bitserial_linear(xn, lyr["projs"]["self_attn.v_proj"], bits_tbl, lv_tbls, h_mat, use_ste_lr)
        v = rmsn_fn(v_lin.view(bsz, seq_len, nkv, hd), None).transpose(1, 2)

    ctx_s = F.scaled_dot_product_attention(
        q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0
    ).transpose(1, 2).contiguous()
    if head_keep_frac < 0.999:
        k_h = max(1, min(nh, int(round(nh * head_keep_frac))))
        with torch.no_grad():
            h_norm = ctx_s.pow(2).sum(dim=-1).sqrt()
            top_h = torch.topk(h_norm, k_h, dim=-1).indices
            hm = torch.zeros_like(h_norm).scatter_(-1, top_h, 1.0)
        ctx_s = ctx_s * hm.unsqueeze(-1)

    o_lin = bitserial_linear(ctx_s.view(bsz, seq_len, nh * hd), lyr["projs"]["self_attn.o_proj"], bits_tbl, lv_tbls, h_mat, use_ste_lr)
    xm = x + rmsn_fn(o_lin, lyr["g_pa"])
    xf = rmsn_fn(xm, lyr["g_pf"])
    g_out = bitserial_linear(xf, lyr["projs"]["mlp.gate_proj"], bits_tbl, lv_tbls, h_mat, use_ste_lr)
    u_out = bitserial_linear(xf, lyr["projs"]["mlp.up_proj"], bits_tbl, lv_tbls, h_mat, use_ste_lr)
    act = F.gelu(g_out, approximate="tanh") * u_out
    if mlp_keep_frac < 0.999:
        n_blk = act.shape[-1] // V93_BLOCK_SIZE
        k_rt = max(V93_SHARED_BLOCKS + 1, min(n_blk, int(round(n_blk * mlp_keep_frac))))
        with torch.no_grad():
            ab = act.reshape(bsz, seq_len, n_blk, V93_BLOCK_SIZE).pow(2).sum(dim=-1)
            m_blk = torch.zeros_like(ab)
            m_blk[:, :, :V93_SHARED_BLOCKS] = 1.0
            if k_rt > V93_SHARED_BLOCKS:
                r_idx = torch.topk(ab[:, :, V93_SHARED_BLOCKS:], k_rt - V93_SHARED_BLOCKS, dim=-1).indices + V93_SHARED_BLOCKS
                m_blk.scatter_(-1, r_idx, 1.0)
        act = act * m_blk.repeat_interleave(V93_BLOCK_SIZE, dim=-1)
    d_lin = bitserial_linear(act, lyr["projs"]["mlp.down_proj"], bits_tbl, lv_tbls, h_mat, use_ste_lr)
    return (xm + rmsn_fn(d_lin, lyr["g_pff"])) * lyr["layer_scalar"]


def run_stage2_5_e2e_logit_kl_qat(
    *,
    e2e_layers: List[dict],
    X_stud_0: torch.Tensor,
    X_teach_l59: torch.Tensor,
    top_ids_cpu: torch.Tensor,
    top_probs_cpu: torch.Tensor,
    ans_mask_np: np.ndarray,
    emb_eff: torch.Tensor,
    fn_param: nn.Parameter,
    tr_seq_ids: torch.Tensor,
    va_seq_ids: torch.Tensor,
    ctx: Dict[str, object],
    rmsn_gpu,
    apply_rope,
    rope_cs_s: torch.Tensor,
    rope_sn_s: torch.Tensor,
    rope_cs_g: torch.Tensor,
    rope_sn_g: torch.Tensor,
    H256_GPU: torch.Tensor,
    device: torch.device,
    gpu_vram_gb: float,
    n_steps: int = 240,
    micro_bs: int = 4,
    grad_accum: int = 4,
) -> Dict[str, object]:
    """Run authentic 60-layer E2E Logit-KL + Cosine QAT with gradient checkpointing and best-val rollback."""
    NH = int(ctx["NH"])
    NKV_S = int(ctx["NKV_S"])
    NKV_F = int(ctx["NKV_F"])
    HD_S = int(ctx["HD_S"])
    HD_F = int(ctx["HD_F"])
    K_EQ_V = bool(ctx["K_EQ_V"])
    LAYER_TYPES = list(ctx["LAYER_TYPES"])
    PART_L = float(ctx["PART_L"])
    PART_G = float(ctx["PART_G"])
    SOFTCAP = float(ctx["SOFTCAP"])
    N_LAYERS = len(e2e_layers)

    dt = torch.bfloat16 if device.type == "cuda" else torch.float32
    bits_tbl, lv_tbls = make_unpack_tables(device, dtype=dt)
    h_mat_c = H256_GPU.to(device=device, dtype=dt)
    emb_gpu = emb_eff.to(device=device, dtype=dt)

    # Keep all bitplanes GPU-resident if VRAM >= 21.5 GiB (L4/A100); else pin CPU and stream on demand
    pin_on_gpu = (device.type == "cuda") and (gpu_vram_gb >= 21.5)
    sc_params: List[torch.nn.Parameter] = []
    lr_params: List[torch.nn.Parameter] = []
    norm_params: List[torch.nn.Parameter] = [fn_param]

    for lyr in e2e_layers:
        for nk in ("g_in", "g_pa", "g_pf", "g_pff", "g_qn", "g_kn", "layer_scalar"):
            lyr[nk] = nn.Parameter(lyr[nk].detach().to(device=device, dtype=torch.float32))
            norm_params.append(lyr[nk])
        for p in lyr["projs"].values():
            p["sc"] = nn.Parameter(p["sc"].detach().to(device=device, dtype=torch.float32))
            sc_params.append(p["sc"])
            if p["bi"] is not None:
                p["bi"] = nn.Parameter(p["bi"].detach().to(device=device, dtype=torch.float32))
                sc_params.append(p["bi"])
            if p["inv_s"] is not None:
                p["inv_s"] = p["inv_s"].to(device=device, dtype=dt)
            if p["u_lr"] is not None:
                p["u_lr"] = nn.Parameter(p["u_lr"].detach().to(device=device, dtype=torch.float32))
                p["v_lr"] = nn.Parameter(p["v_lr"].detach().to(device=device, dtype=torch.float32))
                lr_params.extend([p["u_lr"], p["v_lr"]])
            if pin_on_gpu:
                p["planes_u8"] = p["planes_u8"].to(device=device)
            elif device.type == "cuda":
                p["planes_u8"] = p["planes_u8"].pin_memory()

    ans_mask_t = torch.from_numpy(ans_mask_np)
    tok_weights_all = torch.where(ans_mask_t, torch.full_like(ans_mask_t, 1.5, dtype=torch.float32), torch.full_like(ans_mask_t, 0.5, dtype=torch.float32))

    opt = torch.optim.AdamW(
        [
            {"params": sc_params, "lr": 1.5e-4, "weight_decay": 0.0},
            {"params": lr_params, "lr": 1.5e-4, "weight_decay": 1e-4},
            {"params": norm_params, "lr": 3.0e-4, "weight_decay": 0.0},
        ]
    )
    warmup = max(1, min(20, n_steps // 6))

    def _lr_mult(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = float(step - warmup) / float(max(1, n_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    def _forward_model(x_in: torch.Tensor, head_kf: float = 1.0, mlp_kf: float = 1.0, use_ckpt: bool = True) -> torch.Tensor:
        h = x_in.float()
        for li in range(N_LAYERS):
            lyr = e2e_layers[li]
            is_f = LAYER_TYPES[li] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)

            def _fn(hx, _li=li, _lyr=lyr, _isf=is_f, _nkv=nkv, _hd=hd, _rd=rot_d, _cs=cs_r, _sn=sn_r):
                return forward_layer_e2e(
                    hx, _lyr, is_f=_isf, k_eq_v=K_EQ_V, nh=NH, nkv=_nkv, hd=_hd, rot_d=_rd,
                    cs_r=_cs, sn_r=_sn, rmsn_fn=rmsn_gpu, apply_rope_fn=apply_rope,
                    bits_tbl=bits_tbl, lv_tbls=lv_tbls, h_mat=h_mat_c,
                    head_keep_frac=head_kf, mlp_keep_frac=mlp_kf, use_ste_lr=True,
                )

            if use_ckpt:
                h = torch.utils.checkpoint.checkpoint(_fn, h, use_reentrant=False)
            else:
                h = _fn(h)
        return h

    def _eval_val() -> Tuple[float, float, float]:
        kl_list: List[float] = []
        cos_list: List[float] = []
        top1_hits = 0
        top1_tot = 0
        with torch.no_grad():
            for i0 in range(0, int(va_seq_ids.numel()), micro_bs):
                sl = va_seq_ids[i0 : i0 + micro_bs]
                xb = X_stud_0[sl].to(device=device, dtype=torch.float32)
                xt = X_teach_l59[sl].to(device=device, dtype=torch.float32)
                t_id = top_ids_cpu[sl].to(device=device)
                t_pr = top_probs_cpu[sl].to(device=device)
                tw = tok_weights_all[sl].to(device=device)
                hs = _forward_model(xb, 1.0, 1.0, use_ckpt=False)
                hn = rmsn_gpu(hs, fn_param).to(dt)
                lgs = SOFTCAP * torch.tanh((hn @ emb_gpu.T).float() / SOFTCAP)
                kl = compute_topk_kl_loss(lgs, t_id, t_pr, tw)
                kl_list.append(float(kl.item()))
                cos_list.append(float(F.cosine_similarity(hs.reshape(-1, hs.shape[-1]), xt.reshape(-1, xt.shape[-1]), dim=-1).mean().item()))
                pred_top1 = lgs.argmax(dim=-1)
                teach_top1 = t_id[:, :, 0].long()
                m = ans_mask_t[sl].to(device=device)
                if m.any():
                    top1_hits += int((pred_top1[m] == teach_top1[m]).sum().item())
                    top1_tot += int(m.sum().item())
        return float(np.mean(kl_list)), float(np.mean(cos_list)), float(top1_hits) / float(max(top1_tot, 1))

    def _snapshot_best() -> Dict[int, torch.Tensor]:
        return {id(p): p.detach().cpu().clone() for p in (sc_params + lr_params + norm_params)}

    def _restore_best(snap: Dict[int, torch.Tensor]) -> None:
        with torch.no_grad():
            for p in (sc_params + lr_params + norm_params):
                p.copy_(snap[id(p)].to(device=device, dtype=p.dtype))

    kl_before, cos_before, top1_before = _eval_val()
    best_kl = kl_before
    best_snap = _snapshot_best()
    print(f"[Stage 2.5 E2E KL Init] val_kl={kl_before:.4f} | l59_cos={cos_before:.5f} | top1_agree={top1_before*100:.2f}% | pin_gpu={pin_on_gpu}", flush=True)

    rng = np.random.default_rng(1337)
    tr_np = tr_seq_ids.numpy()
    t0_e2e = time.perf_counter()

    for step in range(n_steps):
        mult = _lr_mult(step)
        base_lrs = (1.5e-4, 1.5e-4, 3.0e-4)
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * mult

        opt.zero_grad(set_to_none=True)
        step_kl = 0.0
        step_cos = 0.0
        for acc_i in range(grad_accum):
            sel = rng.choice(tr_np, size=min(micro_bs, len(tr_np)), replace=False)
            sl = torch.from_numpy(sel).long()
            xb = X_stud_0[sl].to(device=device, dtype=torch.float32)
            xt = X_teach_l59[sl].to(device=device, dtype=torch.float32)
            t_id = top_ids_cpu[sl].to(device=device)
            t_pr = top_probs_cpu[sl].to(device=device)
            tw = tok_weights_all[sl].to(device=device)

            # Mix 25% structured sparse routing in last quarter to sharpen dense AND sparse resilience
            do_sp = (step >= (n_steps * 3) // 4) and (acc_i == grad_accum - 1)
            hk = 0.65 if do_sp else 1.0
            mk = 0.50 if do_sp else 1.0

            hs = _forward_model(xb, hk, mk, use_ckpt=True)
            hn = rmsn_gpu(hs, fn_param).to(dt)
            lgs = SOFTCAP * torch.tanh((hn @ emb_gpu.T).float() / SOFTCAP)
            kl = compute_topk_kl_loss(lgs, t_id, t_pr, tw)
            cos_l = 1.0 - F.cosine_similarity(hs.reshape(-1, hs.shape[-1]), xt.reshape(-1, xt.shape[-1]), dim=-1).mean()
            mag_t = torch.log(xt.pow(2).sum(dim=-1).clamp(min=1e-8))
            mag_s = torch.log(hs.pow(2).sum(dim=-1).clamp(min=1e-8))
            loss = (kl + 0.15 * cos_l + 0.02 * F.mse_loss(mag_s, mag_t)) / float(grad_accum)
            loss.backward()
            step_kl += float(kl.detach().item()) / float(grad_accum)
            step_cos += float(1.0 - cos_l.detach().item()) / float(grad_accum)

        torch.nn.utils.clip_grad_norm_(sc_params + lr_params + norm_params, 1.0)
        opt.step()
        with torch.no_grad():
            for lyr in e2e_layers:
                for p in lyr["projs"].values():
                    p["sc"].clamp_(min=0.0, max=65000.0)
                    if p["bi"] is not None:
                        p["bi"].clamp_(min=-65000.0, max=65000.0)
                lyr["layer_scalar"].clamp_(min=-65000.0, max=65000.0)

        if (step + 1) % 30 == 0 or (step + 1) == n_steps:
            v_kl, v_cos, v_t1 = _eval_val()
            improved = ""
            if v_kl < best_kl:
                best_kl = v_kl
                best_snap = _snapshot_best()
                improved = " [BEST]"
            print(
                f"  [E2E KL step {step+1:03d}/{n_steps}] tr_kl={step_kl:.4f} val_kl={v_kl:.4f} "
                f"l59_cos={v_cos:.5f} top1={v_t1*100:.2f}% dt={time.perf_counter()-t0_e2e:.1f}s{improved}",
                flush=True,
            )

    _restore_best(best_snap)
    kl_after, cos_after, top1_after = _eval_val()
    del emb_gpu, opt, best_snap, bits_tbl, lv_tbls
    if pin_on_gpu:
        for lyr in e2e_layers:
            for p in lyr["projs"].values():
                p["planes_u8"] = p["planes_u8"].cpu()
                p["sc"] = p["sc"].detach().cpu()
                if p["bi"] is not None:
                    p["bi"] = p["bi"].detach().cpu()
                if p["u_lr"] is not None:
                    p["u_lr"] = p["u_lr"].detach().cpu()
                    p["v_lr"] = p["v_lr"].detach().cpu()
                if p["inv_s"] is not None:
                    p["inv_s"] = p["inv_s"].cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[Stage 2.5 E2E KL Done] val_kl: {kl_before:.4f} -> {kl_after:.4f} | top1: {top1_before*100:.2f}% -> {top1_after*100:.2f}%", flush=True)
    return {
        "val_kl_before": round(kl_before, 4),
        "val_kl_after": round(kl_after, 4),
        "l59_cos_before": round(cos_before, 5),
        "l59_cos_after": round(cos_after, 5),
        "top1_agree_before": round(top1_before, 4),
        "top1_agree_after": round(top1_after, 4),
        "e2e_elapsed_s": round(time.perf_counter() - t0_e2e, 2),
    }


def run_selftest() -> int:
    """Mandatory 6-gate fail-closed CPU verification battery for V9.3 E2E Logit-KL QAT."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: V9.3 byte budget <= 6.85 GiB, Zero Ternary-5 anywhere, Zero 1b_rm above L35
    bgt = compute_v93_budget_bytes()
    g1_ok = (
        int(bgt["total_t16c_bytes"]) <= MAX_V93_T16C_BYTES
        and int(bgt["tern5_cnt"]) == 0
        and int(bgt["attn_1b_cnt"]) == 0
        and int(bgt["mlp_1b_upper_cnt"]) == 0
    )
    gate("Gate 1: V9.3 Budget & Zero-Tern5 / Zero-Cliff Architecture", g1_ok, f"total={bgt['total_t16c_gib']} GiB, text={bgt['text_resident_gib']} GiB, tern5={bgt['tern5_cnt']}")

    # Gate 2: Vectorized bitplane unpack bit-exact parity vs v90.unpack_tile16_headers_and_units across 1b/2b_gauss/3b
    rng = np.random.default_rng(93)
    h256 = torch.from_numpy(pa.make_sylvester_hadamard(V93_BLOCK_SIZE)).float()
    bits_tbl, lv_tbls = make_unpack_tables(torch.device("cpu"), dtype=torch.float32)
    max_unpack_diff = 0.0
    g2_ok = True
    try:
        lib_fq = v90._compile_fast_quant(Path(os.environ.get("TEMP", "/tmp")) / "libfast_quant_v93_selftest.so")
        has_c = True
    except Exception:
        lib_fq = None
        has_c = False
    for mode_i in (0, 5, 4, 6):
        do, di = 32, 512
        nb_m, asym_m = p3.MODE_TO_T16C[mode_i]
        n_tiles, n_blks = do // 16, di // V93_BLOCK_SIZE
        tblk_b = p3.tile16_blk_bytes(V93_BLOCK_SIZE, nb_m, asym_m)
        if lib_fq is not None:
            w_f = torch.from_numpy((rng.standard_normal((do, di)) * 0.05).astype(np.float32))
            pb = v90._c_quant_and_pack_t16c(lib_fq, w_f, V93_BLOCK_SIZE, mode_i)
        else:
            pb = rng.integers(0, 256, size=n_tiles * n_blks * tblk_b, dtype=np.uint8)
            hdr_f16 = np.full((n_tiles, n_blks, 16), 0.05, dtype="<f2").view(np.uint8).reshape(n_tiles, n_blks, 32)
            pb_r = pb.reshape(n_tiles, n_blks, tblk_b)
            pb_r[:, :, :32] = hdr_f16
            if asym_m:
                pb_r[:, :, 32:64] = 0
        u_ref, _, _ = v90.unpack_tile16_headers_and_units(pb, do, di, V93_BLOCK_SIZE, mode_i, h256)
        pl = extract_planes_u8(pb, do, di, V93_BLOCK_SIZE, mode_i)
        u_got = unpack_planes_to_units(pl, mode_i, do, di, V93_BLOCK_SIZE, bits_tbl, lv_tbls[mode_i])
        max_unpack_diff = max(max_unpack_diff, float((u_ref.reshape(do, n_blks, V93_BLOCK_SIZE) - u_got).abs().max().item()))
    g2_ok = max_unpack_diff < 1e-6
    gate("Gate 2: Bitplane Unpack Parity vs Reference (1b/2b_gauss/3b)", g2_ok, f"max_diff={max_unpack_diff:.2e} (has_c={has_c})")

    # Gate 3: STE INT8 LowRank exact export parity (0 ULP mismatch)
    u_lr = torch.from_numpy((rng.standard_normal((32, 8)) * 0.04).astype(np.float32))
    v_lr = torch.from_numpy((rng.standard_normal((8, 64)) * 0.04).astype(np.float32))
    u_ste, v_ste = ste_int8_lr(u_lr, v_lr)
    su, sv, ui8, vi8 = quant_lr_int8_tuple(u_lr, v_lr)
    w_ste = u_ste @ v_ste
    w_exp = (torch.from_numpy(ui8.astype(np.float32)) * torch.from_numpy(su).unsqueeze(1)) @ (torch.from_numpy(vi8.astype(np.float32)) * torch.from_numpy(sv).unsqueeze(1))
    lr_diff = float((w_ste - w_exp).abs().max().item())
    gate("Gate 3: STE INT8 Low-Rank Export Parity", lr_diff < 1e-6, f"max_diff={lr_diff:.2e}")

    # Gate 4: Custom `_BitSerialLinearFn` gradient finite-diff / autograd exactness + real backprop KL drop
    do, di, bsz = 32, 64, 64
    h64 = torch.from_numpy(pa.make_sylvester_hadamard(bsz)).float()
    for mode_test, asym_t in ((0, True), (5, False)):
        n_tiles, n_blks = do // 16, di // bsz
        eff_b = 1 if mode_test == 0 else 2
        pl_t = torch.from_numpy(rng.integers(0, 256, size=(n_tiles, n_blks, eff_b, bsz // 8, 16), dtype=np.uint8))
        sc_t = torch.tensor(rng.uniform(0.02, 0.08, size=(do, n_blks)), dtype=torch.float32, requires_grad=True)
        bi_t = torch.tensor(rng.uniform(-0.02, 0.02, size=(do, n_blks)), dtype=torch.float32, requires_grad=True) if asym_t else None
        inv_t = torch.tensor(rng.uniform(0.8, 1.2, size=(di,)), dtype=torch.float32)
        x_in = torch.tensor(rng.normal(0.0, 1.0, size=(3, di)), dtype=torch.float32, requires_grad=True)

        y_fast = _BitSerialLinearFn.apply(x_in, sc_t, bi_t, pl_t, inv_t, h64, bits_tbl, lv_tbls[mode_test], mode_test)
        y_fast.sum().backward()
        gx_f, gsc_f = x_in.grad.clone(), sc_t.grad.clone()
        gbi_f = bi_t.grad.clone() if bi_t is not None else None

        x_ref = x_in.detach().clone().requires_grad_(True)
        sc_ref = sc_t.detach().clone().requires_grad_(True)
        bi_ref = bi_t.detach().clone().requires_grad_(True) if bi_t is not None else None
        u_ref = unpack_planes_to_units(pl_t, mode_test, do, di, bsz, bits_tbl, lv_tbls[mode_test])
        w_ref = _assemble_w_eff(u_ref, sc_ref, bi_ref, inv_t, h64)
        F.linear(x_ref, w_ref).sum().backward()
        cos_gx = float(F.cosine_similarity(gx_f.flatten(), x_ref.grad.flatten(), dim=0).item())
        cos_gsc = float(F.cosine_similarity(gsc_f.flatten(), sc_ref.grad.flatten(), dim=0).item())
        cos_gbi = float(F.cosine_similarity(gbi_f.flatten(), bi_ref.grad.flatten(), dim=0).item()) if bi_ref is not None else 1.0
        assert cos_gx > 0.9999 and cos_gsc > 0.9999 and cos_gbi > 0.9999, f"Grad mismatch mode={mode_test}: {cos_gx} {cos_gsc} {cos_gbi}"

    # Verify real Logit-KL drop on synthetic Teacher distribution using AdamW + checkpoint
    x0 = torch.randn(8, di)
    w_teach = torch.randn(64, di) * 0.15
    emb_toy = torch.randn(128, 64) * 0.1
    lgt = 30.0 * torch.tanh(((x0 @ w_teach.T) @ emb_toy.T) / 30.0)
    tp_all, tid_all = torch.topk(F.softmax(lgt, dim=-1), 32, dim=-1)
    pl_m = torch.from_numpy(rng.integers(0, 256, size=(64 // 16, di // bsz, 2, bsz // 8, 16), dtype=np.uint8))
    proj_m = {"planes_u8": pl_m, "sc": nn.Parameter(torch.full((64, di // bsz), 0.05)), "bi": None, "inv_s": None, "u_lr": nn.Parameter(torch.randn(64, 4) * 0.01), "v_lr": nn.Parameter(torch.randn(4, di) * 0.01), "mode_i": 5}
    opt_m = torch.optim.AdamW([proj_m["sc"], proj_m["u_lr"], proj_m["v_lr"]], lr=2e-2)
    kl_0 = kl_end = 0.0
    for s_i in range(25):
        opt_m.zero_grad()
        hs_m = bitserial_linear(x0, proj_m, bits_tbl, lv_tbls, h64, use_ste_lr=True)
        lgs_m = 30.0 * torch.tanh((hs_m @ emb_toy.T) / 30.0)
        loss_m = compute_topk_kl_loss(lgs_m, tid_all, tp_all)
        if s_i == 0:
            kl_0 = float(loss_m.item())
        loss_m.backward()
        opt_m.step()
        kl_end = float(loss_m.item())
    gate("Gate 4: BitSerial Autograd Exactness & Real KL Convergence", kl_end < 0.6 * kl_0, f"kl_0={kl_0:.4f} -> kl_end={kl_end:.4f}")

    # Gate 5: Decontamination check
    hold = v90.collect_holdout_4grams()
    ok_clean = True
    try:
        v90.verify_corpus_decontaminated(["Clean STEM calibration sample on matrices and gradients."], hold)
    except Exception:
        ok_clean = False
    ok_dirty = False
    try:
        v90.verify_corpus_decontaminated(["Romeo and Juliet is written by William Shakespeare."], hold)
    except (ValueError, AssertionError):
        ok_dirty = True
    gate("Gate 5: Decontamination Fail-Closed Check", ok_clean and ok_dirty, f"hold_4grams={len(hold)}")

    # Gate 6: End-to-End CRTR + Router non-rank-1 export check
    x_r = torch.randn(64, 32)
    y_r = torch.rand(64, 16)
    rn, _ = crtr.train_layer_router(x_r[:48], y_r[:48], x_r[48:], y_r[48:], top_p=0.5, rank=8, steps=10, device="cpu", seed=1)
    rq = crtr.quantize_layer_for_crtr(rn, 256)
    g6_ok = np.all(np.isfinite(rq["sv"])) and rq["v_i8"].std() > 0.0
    gate("Gate 6: Auxiliary Router Export Non-Degeneracy", bool(g6_ok), f"v_i8_std={rq['v_i8'].std():.2f}")

    print(f"V9.3 Self-Test Result: {6 - failures}/6 gates passed", flush=True)
    return failures


def run_v93_pipeline(*, e2e_steps: int = 240, micro_bs: int = 4, grad_accum: int = 4) -> Dict[str, object]:
    """Execute complete V9.3 Zero-Tern5 build + Stage 2.5 60-layer E2E Logit-KL QAT + Stage 3 distill."""
    from safetensors import safe_open

    t0_all = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA GPU required for V9.3 training!"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    compute_dtype = torch.bfloat16

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    v90.OUT_DIR = OUT_DIR
    v90.DRIVE_DIR = DRIVE_DIR
    if DRIVE_DIR is not None:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    bgt = compute_v93_budget_bytes()
    assert bgt["tern5_cnt"] == 0 and bgt["total_t16c_bytes"] <= MAX_V93_T16C_BYTES, f"Bad budget: {bgt}"
    print(f"[V9.3 Start] GPU={gpu_name} ({gpu_vram_gb:.1f} GiB) | target={bgt['total_t16c_gib']} GiB | zero tern5 | e2e_steps={e2e_steps}", flush=True)

    lib_fq = v90._compile_fast_quant(OUT_DIR / "libfast_quant.so")
    st = v90._init_v9_stage1_base(device, compute_dtype)
    sf_path: Path = st["sf_path"]  # type: ignore[assignment]
    tok_path: Path = st["tok_path"]  # type: ignore[assignment]
    N_LAYERS = int(st["N_LAYERS"])
    D_MOD = int(st["D_MOD"])
    EPS = float(st["EPS"])
    ESCALE = float(st["ESCALE"])
    NH = int(st["NH"])
    NKV_S = int(st["NKV_S"])
    NKV_F = int(st["NKV_F"])
    HD_S = int(st["HD_S"])
    HD_F = int(st["HD_F"])
    K_EQ_V = bool(st["K_EQ_V"])
    LAYER_TYPES = list(st["LAYER_TYPES"])
    THETA_L, PART_L = float(st["THETA_L"]), float(st["PART_L"])
    THETA_G, PART_G = float(st["THETA_G"]), float(st["PART_G"])
    SOFTCAP = float(st["SOFTCAP"])
    soft_tok = int(st["soft_tok"])
    all_seq_ids = st["all_seq_ids"]
    all_seq_nxt = st["all_seq_nxt"]
    all_ans_mask = st["all_ans_mask"]
    N_TEXT_SEQ = int(st["N_TEXT_SEQ"])
    N_SEQ = int(st["N_SEQ"])
    emb_raw = st.pop("emb_raw")
    fn_orig = st["fn_orig"]
    X_teach_perm = st["X_teach_perm"]
    perm_pi = st["perm_pi"]
    H256_GPU = st["H256_GPU"]
    H8_GPU = st["H8_GPU"]
    rmsn_gpu = st["rmsn_gpu"]
    rope_cs_s, rope_sn_s = st["rope_cs_s"], st["rope_sn_s"]
    rope_cs_g, rope_sn_g = st["rope_cs_g"], st["rope_sn_g"]
    apply_rope = st["apply_rope"]

    # Pack Vision & Quantize Permuted Embedding Table (preserving full VLM capability)
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    H64_GPU = torch.from_numpy(pa.make_sylvester_hadamard(64)).to(device, dtype=torch.float32)
    vision_norm_recs: List[Tuple[str, np.ndarray]] = []
    vision_mat_recs: List[tuple] = []
    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        sf_keys = set(sf.keys())
        def _pack_vis(name: str, wt: torch.Tensor, pad_in: int = 0) -> None:
            t = wt.float()
            if "embedding_projection" in name and t.shape[0] == D_MOD:
                t = t[perm_pi]
            do_v, di_v = int(t.shape[0]), int(t.shape[1])
            die_v = pad_in if pad_in > di_v else di_v
            if die_v > di_v:
                tp = torch.zeros((do_v, die_v), dtype=torch.float32)
                tp[:, :di_v] = t
                t = tp
            pb_v = v90._c_quant_and_pack_t16c(lib_fq, (t.to(device).reshape(-1, 64) @ H64_GPU).cpu(), 64, 6)
            vision_mat_recs.append((name, do_v, die_v, 64, 3 | p3.T16C_GAUSS_BIT, False, True, pb_v, None, None))

        for sk_v in ("model.vision_tower.std_bias", "model.vision_tower.std_scale"):
            if sk_v in sf_keys:
                vision_norm_recs.append((sk_v, sf.get_tensor(sk_v).float().numpy().astype("<f4")))
        pos_tbl = sf.get_tensor("model.vision_tower.patch_embedder.position_embedding_table").float()
        for ax in (0, 1):
            _pack_vis(f"model.vision_tower.patch_embedder.position_embedding_table.{ax}", pos_tbl[ax])
        _pack_vis("model.vision_tower.patch_embedder.input_proj.weight", sf.get_tensor("model.vision_tower.patch_embedder.input_proj.weight"))
        for lv in range(27):
            fpv = f"model.vision_tower.encoder.layers.{lv}"
            for nk in ("input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                vision_norm_recs.append((f"{fpv}.{nk}", sf.get_tensor(f"{fpv}.{nk}").float().numpy().astype("<f4")))
            for pj in ("self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight", "self_attn.o_proj.linear.weight", "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight"):
                _pack_vis(f"{fpv}.{pj}", sf.get_tensor(f"{fpv}.{pj}"))
            _pack_vis(f"{fpv}.mlp.down_proj.linear.weight", sf.get_tensor(f"{fpv}.mlp.down_proj.linear.weight"), pad_in=4352)
        _pack_vis("model.embed_vision.embedding_projection.weight", sf.get_tensor("model.embed_vision.embedding_projection.weight"))
    del H64_GPU
    torch.cuda.empty_cache()

    emb_perm_teach = emb_raw[:, perm_pi].contiguous()
    del emb_raw
    vocab_sz = int(emb_perm_teach.shape[0])
    emb_eff = torch.empty((vocab_sz, D_MOD), dtype=torch.float16)
    exp_emb_b = p3.expected_tile16_bytes(vocab_sz, D_MOD, V93_BLOCK_SIZE, 3, False)
    emb_pb = np.empty(exp_emb_b, dtype=np.uint8)
    tblk_emb = p3.tile16_blk_bytes(V93_BLOCK_SIZE, 3, False) * (D_MOD // V93_BLOCK_SIZE)
    for r0 in range(0, vocab_sz, 16384):
        r1 = min(r0 + 16384, vocab_sz)
        w_c = emb_perm_teach[r0:r1].to(device, dtype=torch.float32)
        w_rot = (w_c.reshape(-1, V93_BLOCK_SIZE) @ H256_GPU).contiguous()
        pb_c = v90._c_quant_and_pack_t16c(lib_fq, w_rot, V93_BLOCK_SIZE, 6)
        emb_pb[(r0 // 16) * tblk_emb : (r0 // 16) * tblk_emb + pb_c.size] = pb_c
        u_c, sc_c, _ = v90.unpack_tile16_headers_and_units(pb_c, r1 - r0, D_MOD, V93_BLOCK_SIZE, 6, H256_GPU)
        emb_eff[r0:r1] = v90.reconstruct_from_headers(u_c, sc_c, None, None, H256_GPU).to(torch.float16).cpu()
        del w_c, w_rot, pb_c, u_c, sc_c
    torch.cuda.empty_cache()

    X_stud_0 = (emb_eff[torch.from_numpy(all_seq_ids)] * ESCALE).to(torch.float16)
    for si in range(N_TEXT_SEQ, N_SEQ):
        pm = torch.from_numpy(all_seq_ids[si] == soft_tok)
        X_stud_0[si, pm] = X_teach_perm[si, pm]
    X_stud_perm = X_stud_0.clone()

    # Pre-read Teacher norms
    layer_norms_raw: List[Dict[str, np.ndarray]] = []
    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        for li in range(N_LAYERS):
            pfx = f"model.language_model.layers.{li}"
            d_n = {}
            for nk in ("input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"):
                d_n[nk] = sf.get_tensor(f"{pfx}.{nk}").float()[perm_pi].numpy().astype(np.float32)
            for qk in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                d_n[qk] = sf.get_tensor(f"{pfx}.{qk}").float().numpy().astype(np.float32)
            d_n["layer_scalar"] = sf.get_tensor(f"{pfx}.layer_scalar").float().numpy().astype(np.float32)
            layer_norms_raw.append(d_n)

    va_seq_ids = torch.arange(15, N_SEQ, 16, device="cpu", dtype=torch.long)
    tr_seq_ids = torch.tensor([i for i in range(N_SEQ) if i % 16 != 15], device="cpu", dtype=torch.long)
    MB_FWD = 16 if gpu_vram_gb >= 35.0 else 8
    perm_pi_g = perm_pi.to(device)

    # Stage 2: Fast Dense-Consistent Layer Calibration + Bitplane & Scale/LR Extraction
    e2e_layers: List[dict] = []
    pi_l_list: List[torch.Tensor] = []
    start_l = 0
    s2_done_ckpt = CKPT_DIR / "stage2_meta.pt"
    if s2_done_ckpt.is_file():
        meta2 = torch.load(s2_done_ckpt, map_location="cpu", weights_only=False)
        start_l = int(meta2.get("next_layer", 0))
        if start_l > 0:
            X_teach_perm = torch.load(CKPT_DIR / "stage2_xt.pt", map_location="cpu", weights_only=True)
            X_stud_perm = torch.load(CKPT_DIR / "stage2_xs.pt", map_location="cpu", weights_only=True)
            pi_l_list = meta2["pi_l_list"]
            for li_r in range(start_l):
                e2e_layers.append(torch.load(CKPT_DIR / f"e2e_L{li_r:02d}.pt", map_location="cpu", weights_only=False))
            print(f"[Stage 2 Resume] Resumed from layer {start_l}/{N_LAYERS}", flush=True)

    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        for l_idx in range(start_l, N_LAYERS):
            tl0 = time.perf_counter()
            pfx = f"model.language_model.layers.{l_idx}"
            is_f = LAYER_TYPES[l_idx] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)
            rep = NH // nkv
            dn = layer_norms_raw[l_idx]
            g_in = torch.from_numpy(dn["input_layernorm.weight"]).to(device)
            g_pa = torch.from_numpy(dn["post_attention_layernorm.weight"]).to(device)
            g_pf = torch.from_numpy(dn["pre_feedforward_layernorm.weight"]).to(device)
            g_pff = torch.from_numpy(dn["post_feedforward_layernorm.weight"]).to(device)
            g_qn = torch.from_numpy(dn["self_attn.q_norm.weight"]).to(device)
            g_kn = torch.from_numpy(dn["self_attn.k_norm.weight"]).to(device)
            lsc_init = float(dn["layer_scalar"][0])

            wq_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.q_proj", perm_pi_g, device, is_perm_col=True)
            wk_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.k_proj", perm_pi_g, device, is_perm_col=True)
            wv_t = wk_t if (is_f and K_EQ_V) else v90._load_dequant_proj(sf, f"{pfx}.self_attn.v_proj", perm_pi_g, device, is_perm_col=True)
            wo_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.o_proj", perm_pi_g, device, is_perm_row=True)
            wg_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.gate_proj", perm_pi_g, device, is_perm_col=True)
            wu_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.up_proj", perm_pi_g, device, is_perm_col=True)
            wd_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.down_proj", perm_pi_g, device, is_perm_row=True)

            # 1) Teacher pass + Hessians + GeGLU co-activation samples
            attn_x_rms = torch.zeros((D_MOD,), device=device, dtype=torch.float32)
            mlp_x_rms = torch.zeros((D_MOD,), device=device, dtype=torch.float32)
            down_x_sq_orig = torch.zeros((V93_FF,), device=device, dtype=torch.float32)
            H_attn = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32)
            H_o = torch.zeros((NH * hd, NH * hd), device=device, dtype=torch.float32)
            H_mlp = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32)
            act_samples, x_attn_ev, x_o_ev, x_mlp_ev = [], [], [], []
            X_teach_next = torch.empty_like(X_teach_perm)
            X_teach_attn_out = torch.empty_like(X_teach_perm)
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
                    q = apply_rope(rmsn_gpu(F.linear(xn, wq_t).view(b1 - b0, v90.V9_SEQ_LEN, NH, hd), g_qn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    k = apply_rope(rmsn_gpu(F.linear(xn, wk_t).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), g_kn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    v = rmsn_gpu(F.linear(xn, wv_t).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                    ctx_h = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous()
                    ctx_f = ctx_h.reshape(-1, NH * hd)
                    H_o.add_(ctx_f.T @ ctx_f)
                    if b0 < 256:
                        x_o_ev.append(ctx_f[::4])
                    xm = xb + rmsn_gpu(F.linear(ctx_h.view(b1 - b0, v90.V9_SEQ_LEN, NH * hd), wo_t), g_pa)
                    X_teach_attn_out[b0:b1] = xm.to(torch.float16).cpu()
                    xf = rmsn_gpu(xm, g_pf)
                    xf_f = xf.reshape(-1, D_MOD)
                    mlp_x_rms.add_((xf * xf).sum(dim=(0, 1)))
                    H_mlp.add_(xf_f.T @ xf_f)
                    if b0 < 256:
                        x_mlp_ev.append(xf_f[::4])
                    act = F.gelu(F.linear(xf, wg_t), approximate="tanh") * F.linear(xf, wu_t)
                    down_x_sq_orig.add_((act * act).sum(dim=(0, 1)))
                    if b0 < 64:
                        act_samples.append(act[:, ::2, :].reshape(-1, V93_FF).abs().to(torch.float32).cpu())
                    X_teach_next[b0:b1] = ((xm + rmsn_gpu(F.linear(act, wd_t), g_pff)) * lsc_init).to(torch.float16).cpu()

            attn_x_rms = torch.sqrt(attn_x_rms / float(N_SEQ * v90.V9_SEQ_LEN)).clamp(min=1e-5)
            mlp_x_rms = torch.sqrt(mlp_x_rms / float(N_SEQ * v90.V9_SEQ_LEN)).clamp(min=1e-5)
            xa_ev_t = torch.cat(x_attn_ev, dim=0)
            xo_ev_t = torch.cat(x_o_ev, dim=0)
            xm_ev_t = torch.cat(x_mlp_ev, dim=0)

            # 2) Lossless GeGLU co-activation clustering pi_l
            act_cat = torch.cat(act_samples, dim=0).to(device, dtype=torch.float32)
            pi_l, _, pi_stats = v90.compute_geglu_colblock_perm(act_cat, shared_blocks=V93_SHARED_BLOCKS, block_size=V93_BLOCK_SIZE, iters=6, seed=42 + l_idx)
            pi_l_list.append(pi_l.cpu())
            pi_lg = pi_l.to(device)
            wg_t = wg_t[pi_lg, :].contiguous()
            wu_t = wu_t[pi_lg, :].contiguous()
            wd_t = wd_t[:, pi_lg].contiguous()
            down_x_rms = torch.sqrt((down_x_sq_orig[pi_lg] / float(N_SEQ * v90.V9_SEQ_LEN)).clamp(min=1e-10)).clamp(min=1e-5)
            del act_samples, act_cat, down_x_sq_orig

            H_down_blks = torch.zeros((V93_MLP_BLOCKS, V93_BLOCK_SIZE, V93_BLOCK_SIZE), device=device, dtype=torch.float32)
            xd_ev_list: List[torch.Tensor] = []
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xf_d = rmsn_gpu(X_teach_attn_out[b0:b1].to(device, dtype=torch.float32), g_pf)
                    act_p = F.gelu(F.linear(xf_d, wg_t), approximate="tanh") * F.linear(xf_d, wu_t)
                    ab = act_p.reshape(-1, V93_MLP_BLOCKS, V93_BLOCK_SIZE)
                    H_down_blks.add_(torch.einsum("nbi,nbj->bij", ab, ab))
                    if b0 < 256:
                        xd_ev_list.append(act_p.reshape(-1, V93_FF)[::8].to(torch.float16).cpu())
            xd_ev_t = torch.cat(xd_ev_list, dim=0).to(device) if xd_ev_list else None
            del xd_ev_list

            def _quant_pack_proj(
                w_f32: torch.Tensor, xr: torch.Tensor, pname: str,
                h_full: Optional[torch.Tensor] = None, h_blks_in: Optional[torch.Tensor] = None,
                x_ev: Optional[torch.Tensor] = None, val_x: Optional[torch.Tensor] = None,
            ) -> Tuple[dict, torch.Tensor]:
                mode_i, bsz_i, lrr, is_rm, is_cb = get_v93_proj_spec(l_idx, pname, is_f)
                do, di = int(w_f32.shape[0]), int(w_f32.shape[1])
                sq = (xr.pow(0.5) / w_f32.abs().amax(dim=0).clamp(min=1e-5).pow(0.5)).clamp(0.25, 4.0)
                sq = sq / sq.mean().clamp(min=1e-5)
                inv_s = 1.0 / sq
                w_rot = ((w_f32 * sq.unsqueeze(0)).reshape(-1, bsz_i) @ H256_GPU).reshape(do, di).contiguous()
                h_blks_rot: Optional[torch.Tensor] = None
                if h_full is not None:
                    h_blks_rot = v90._h_full_to_blks_rot(h_full, inv_s, H256_GPU, bsz_i)
                    v90.gptq_quant_blks_gpu(w_rot, torch.stack([h_full[b * bsz_i : (b + 1) * bsz_i, b * bsz_i : (b + 1) * bsz_i] for b in range(di // bsz_i)], dim=0), inv_s, H256_GPU, bsz_i, mode_i)
                elif h_blks_in is not None:
                    h_blks_rot = torch.einsum("ij,bjk,kl->bil", H256_GPU, (h_blks_in * inv_s.reshape(-1, bsz_i).unsqueeze(2) * inv_s.reshape(-1, bsz_i).unsqueeze(1)).float(), H256_GPU)
                    h_blks_rot = 0.5 * (h_blks_rot + h_blks_rot.mT)
                    v90.gptq_quant_blks_gpu(w_rot, h_blks_in, inv_s, H256_GPU, bsz_i, mode_i)
                pb = v90._c_quant_and_pack_t16c(lib_fq, w_rot, bsz_i, mode_i)
                u_g, sc_g, bi_g = v90.unpack_tile16_headers_and_units(pb, do, di, bsz_i, mode_i, H256_GPU)
                u_lr = v_lr = w_lr_d = None
                if lrr > 0 and x_ev is not None:
                    w_b0 = v90.reconstruct_from_headers(u_g, sc_g, bi_g, inv_s, H256_GPU)
                    lr_t, lr_d = v90._fit_als_lr_int8(w_f32, w_b0, x_ev, lrr)
                    if lr_t is not None and lr_d is not None:
                        su, sv, ui8, vi8 = lr_t
                        u_lr = (torch.from_numpy(ui8.astype(np.float32)).to(device) * torch.from_numpy(su).to(device).unsqueeze(1)).cpu()
                        v_lr = (torch.from_numpy(vi8.astype(np.float32)).to(device) * torch.from_numpy(sv).to(device).unsqueeze(1)).cpu()
                        w_lr_d = lr_d
                vx = val_x if val_x is not None else x_ev[: min(1024, x_ev.shape[0])].float()
                sc_g, bi_g, _, _, _ = v90.solve_block_ridge_headers(w_f32, u_g, sc_g, bi_g, h_blks_rot, inv_s, H256_GPU, w_lr=w_lr_d, val_x=vx, val_y_target=F.linear(vx, w_f32))
                w_eff = v90.reconstruct_from_headers(u_g, sc_g, bi_g, inv_s, H256_GPU) + (w_lr_d if w_lr_d is not None else 0.0)
                rec = {
                    "do": do, "di": di, "bsize": bsz_i, "mode_i": mode_i, "is_rm": is_rm, "is_cb": is_cb,
                    "planes_u8": extract_planes_u8(pb, do, di, bsz_i, mode_i).cpu(),
                    "sc": sc_g.detach().cpu(), "bi": bi_g.detach().cpu() if bi_g is not None else None,
                    "inv_s": inv_s.detach().cpu(), "u_lr": u_lr, "v_lr": v_lr,
                }
                del pb, u_g, h_blks_rot
                return rec, w_eff

            projs_rec: Dict[str, dict] = {}
            wq_e_rec, wq_s = _quant_pack_proj(wq_t, attn_x_rms, "self_attn.q_proj", h_full=H_attn, x_ev=xa_ev_t)
            wk_e_rec, wk_s = _quant_pack_proj(wk_t, attn_x_rms, "self_attn.k_proj", h_full=H_attn, x_ev=xa_ev_t)
            projs_rec["self_attn.q_proj"] = wq_e_rec
            projs_rec["self_attn.k_proj"] = wk_e_rec
            if is_f and K_EQ_V:
                wv_s = wk_s
            else:
                wv_e_rec, wv_s = _quant_pack_proj(wv_t, attn_x_rms, "self_attn.v_proj", h_full=H_attn, x_ev=xa_ev_t)
                projs_rec["self_attn.v_proj"] = wv_e_rec
            wo_e_rec, wo_s = _quant_pack_proj(wo_t, torch.ones((NH * hd,), device=device), "self_attn.o_proj", h_full=H_o, x_ev=xo_ev_t)
            projs_rec["self_attn.o_proj"] = wo_e_rec
            del H_attn, H_o, xa_ev_t, xo_ev_t

            wg_e_rec, wg_s = _quant_pack_proj(wg_t, mlp_x_rms, "mlp.gate_proj", h_full=H_mlp, x_ev=xm_ev_t)
            wu_e_rec, wu_s = _quant_pack_proj(wu_t, mlp_x_rms, "mlp.up_proj", h_full=H_mlp, x_ev=xm_ev_t)
            projs_rec["mlp.gate_proj"] = wg_e_rec
            projs_rec["mlp.up_proj"] = wu_e_rec
            del H_mlp, xm_ev_t
            wd_e_rec, wd_s = _quant_pack_proj(wd_t, down_x_rms, "mlp.down_proj", h_blks_in=H_down_blks, x_ev=xd_ev_t)
            projs_rec["mlp.down_proj"] = wd_e_rec
            del H_down_blks, xd_ev_t

            # 3) Dense forward + Dense Wiener norms (no sparse distortion in Stage 2!)
            z_pa_tr, dy_pa_tr, z_pa_va, dy_pa_va = [], [], [], []
            xm_stud_cpu = torch.empty_like(X_stud_perm)
            va_set = set(va_seq_ids.tolist())
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xb = X_stud_perm[b0:b1].to(device, dtype=torch.float32)
                    xn = rmsn_gpu(xb, g_in)
                    q = apply_rope(rmsn_gpu(F.linear(xn, wq_s).view(b1 - b0, v90.V9_SEQ_LEN, NH, hd), g_qn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    k = apply_rope(rmsn_gpu(F.linear(xn, wk_s).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), g_kn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    v = rmsn_gpu(F.linear(xn, wv_s).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                    ctx_s = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous()
                    z_b = rmsn_gpu(F.linear(ctx_s.view(b1 - b0, v90.V9_SEQ_LEN, NH * hd), wo_s), None)
                    dy_b = X_teach_attn_out[b0:b1].to(device, dtype=torch.float32) - xb
                    for off_s, si in enumerate(range(b0, b1)):
                        if si in va_set:
                            z_pa_va.append(z_b[off_s, ::4]); dy_pa_va.append(dy_b[off_s, ::4])
                        elif si < 256:
                            z_pa_tr.append(z_b[off_s, ::4]); dy_pa_tr.append(dy_b[off_s, ::4])
                g_pa, _, _ = v90.solve_wiener_post_norm(torch.cat(z_pa_tr, 0), torch.cat(dy_pa_tr, 0), torch.cat(z_pa_va, 0), torch.cat(dy_pa_va, 0), g_pa)
                del z_pa_tr, dy_pa_tr, z_pa_va, dy_pa_va

                act_tr, dy_pff_tr, pred_tr = [], [], []
                act_va, dy_pff_va, pred_va = [], [], []
                z_pff_tr, z_pff_va = [], []
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xb = X_stud_perm[b0:b1].to(device, dtype=torch.float32)
                    xn = rmsn_gpu(xb, g_in)
                    q = apply_rope(rmsn_gpu(F.linear(xn, wq_s).view(b1 - b0, v90.V9_SEQ_LEN, NH, hd), g_qn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    k = apply_rope(rmsn_gpu(F.linear(xn, wk_s).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), g_kn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    v = rmsn_gpu(F.linear(xn, wv_s).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                    ctx_s = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous()
                    xm = xb + rmsn_gpu(F.linear(ctx_s.view(b1 - b0, v90.V9_SEQ_LEN, NH * hd), wo_s), g_pa)
                    xm_stud_cpu[b0:b1] = xm.to(torch.float16).cpu()
                    xf = rmsn_gpu(xm, g_pf)
                    a_b = F.gelu(F.linear(xf, wg_s), approximate="tanh") * F.linear(xf, wu_s)
                    zp = rmsn_gpu(F.linear(a_b, wd_s), None)
                    xt = X_teach_next[b0:b1].to(device, dtype=torch.float32)
                    dyp = (xt / max(abs(lsc_init), 1e-4)) - xm
                    for off_s, si in enumerate(range(b0, b1)):
                        if si in va_set:
                            z_pff_va.append(zp[off_s, ::4]); dy_pff_va.append(dyp[off_s, ::4]); act_va.append(xm[off_s, ::4]); pred_va.append(xt[off_s, ::4])
                        elif si < 256:
                            z_pff_tr.append(zp[off_s, ::4]); dy_pff_tr.append(dyp[off_s, ::4]); act_tr.append(xm[off_s, ::4]); pred_tr.append(xt[off_s, ::4])
                ztr, zva = torch.cat(z_pff_tr, 0), torch.cat(z_pff_va, 0)
                g_pff, _, _ = v90.solve_wiener_post_norm(ztr, torch.cat(dy_pff_tr, 0), zva, torch.cat(dy_pff_va, 0), g_pff)
                lsc_fin, _, _ = v90.solve_wiener_layer_scalar(
                    torch.cat(act_tr, 0) + ztr * g_pff, torch.cat(pred_tr, 0),
                    torch.cat(act_va, 0) + zva * g_pff, torch.cat(pred_va, 0), lsc_init,
                )
                del z_pff_tr, z_pff_va, dy_pff_tr, dy_pff_va, act_tr, act_va, pred_tr, pred_va, ztr, zva

                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xm = xm_stud_cpu[b0:b1].to(device, dtype=torch.float32)
                    xf = rmsn_gpu(xm, g_pf)
                    a_b = F.gelu(F.linear(xf, wg_s), approximate="tanh") * F.linear(xf, wu_s)
                    X_stud_perm[b0:b1] = ((xm + rmsn_gpu(F.linear(a_b, wd_s), g_pff)) * lsc_fin).to(torch.float16).cpu()

            lyr_e2e = {
                "g_in": g_in.cpu(), "g_pa": g_pa.cpu(), "g_pf": g_pf.cpu(), "g_pff": g_pff.cpu(),
                "g_qn": g_qn.cpu(), "g_kn": g_kn.cpu(), "layer_scalar": torch.tensor([lsc_fin], dtype=torch.float32),
                "projs": projs_rec,
            }
            e2e_layers.append(lyr_e2e)
            torch.save(lyr_e2e, CKPT_DIR / f"e2e_L{l_idx:02d}.pt")
            X_teach_perm = X_teach_next
            cos_l = float(F.cosine_similarity(X_stud_perm[-32:].reshape(-1, D_MOD).float(), X_teach_perm[-32:].reshape(-1, D_MOD).float(), dim=-1).mean().item())
            print(f"  [V9.3 Stage2 L{l_idx:02d}/59] cos={cos_l:.5f} e_10of84={pi_stats['perm_12of84_energy_pct']}% dt={time.perf_counter()-tl0:.1f}s", flush=True)

            del wq_t, wk_t, wv_t, wo_t, wg_t, wu_t, wd_t, wq_s, wk_s, wv_s, wo_s, wg_s, wu_s, wd_s, X_teach_attn_out, xm_stud_cpu
            gc.collect()
            torch.cuda.empty_cache()

            if (l_idx + 1) % 10 == 0 or (l_idx + 1) == N_LAYERS:
                torch.save(X_teach_perm.cpu(), CKPT_DIR / "stage2_xt.pt")
                torch.save(X_stud_perm.cpu(), CKPT_DIR / "stage2_xs.pt")
                torch.save({"next_layer": l_idx + 1, "pi_l_list": pi_l_list}, s2_done_ckpt)

    # Precompute exact Teacher Top-128 Softmax Targets using unquantized Teacher final_norm + emb_perm_teach
    fn_teach_g = fn_orig[perm_pi].to(device=device, dtype=torch.float32)
    emb_teach_g = emb_perm_teach.to(device=device, dtype=torch.bfloat16)
    top_ids_cpu = torch.empty((N_SEQ, v90.V9_SEQ_LEN, 128), dtype=torch.int32)
    top_probs_cpu = torch.empty((N_SEQ, v90.V9_SEQ_LEN, 128), dtype=torch.float16)
    with torch.no_grad():
        for b0 in range(0, N_SEQ, 16):
            b1 = min(b0 + 16, N_SEQ)
            ht = rmsn_gpu(X_teach_perm[b0:b1].to(device=device, dtype=torch.float32), fn_teach_g).to(torch.bfloat16)
            lgt = SOFTCAP * torch.tanh((ht @ emb_teach_g.T).float() / SOFTCAP)
            pt = F.softmax(lgt, dim=-1)
            tp, ti = torch.topk(pt, 128, dim=-1)
            top_ids_cpu[b0:b1] = ti.to(torch.int32).cpu()
            top_probs_cpu[b0:b1] = tp.to(torch.float16).cpu()
    del emb_teach_g, emb_perm_teach, fn_teach_g
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2.5: End-to-End 60-Layer Differentiable Logit-KL + Cosine QAT
    fn_param = nn.Parameter(fn_orig[perm_pi].to(device=device, dtype=torch.float32))
    e2e_metrics = run_stage2_5_e2e_logit_kl_qat(
        e2e_layers=e2e_layers,
        X_stud_0=X_stud_0,
        X_teach_l59=X_teach_perm,
        top_ids_cpu=top_ids_cpu,
        top_probs_cpu=top_probs_cpu,
        ans_mask_np=all_ans_mask,
        emb_eff=emb_eff,
        fn_param=fn_param,
        tr_seq_ids=tr_seq_ids,
        va_seq_ids=va_seq_ids,
        ctx=st,
        rmsn_gpu=rmsn_gpu,
        apply_rope=apply_rope,
        rope_cs_s=rope_cs_s,
        rope_sn_s=rope_sn_s,
        rope_cs_g=rope_cs_g,
        rope_sn_g=rope_sn_g,
        H256_GPU=H256_GPU,
        device=device,
        gpu_vram_gb=gpu_vram_gb,
        n_steps=e2e_steps,
        micro_bs=micro_bs,
        grad_accum=grad_accum,
    )
    del top_ids_cpu, top_probs_cpu

    # Assemble final Stage-2.5 tuned T16C + replay once to train matched HeadRouter, CRTR-V9, C8DB, and KV58/59
    all_norms: List[Tuple[str, np.ndarray]] = [("model.language_model.norm.weight", fn_param.detach().cpu().numpy().astype(np.float32))]
    for li, lyr in enumerate(e2e_layers):
        pfx = f"model.language_model.layers.{li}"
        for nk, lk in (
            ("input_layernorm.weight", "g_in"),
            ("post_attention_layernorm.weight", "g_pa"),
            ("pre_feedforward_layernorm.weight", "g_pf"),
            ("post_feedforward_layernorm.weight", "g_pff"),
            ("self_attn.q_norm.weight", "g_qn"),
            ("self_attn.k_norm.weight", "g_kn"),
            ("layer_scalar", "layer_scalar"),
        ):
            all_norms.append((f"{pfx}.{nk}", lyr[lk].detach().cpu().numpy().astype(np.float32)))
    all_norms.extend(vision_norm_recs)

    total_mats = 1 + sum(len(l["projs"]) for l in e2e_layers) + len(vision_mat_recs)
    tmp_t16c = OUT_DIR / (OUT_T16C.name + ".tmp")
    ast_seqs = list(range(0, N_SEQ, 2))[:512]
    ast_set = set(ast_seqs)
    cap_kv58 = {"k_rope": torch.zeros((N_SEQ, NKV_S, v90.V9_SEQ_LEN, HD_S), dtype=torch.float16), "v_norm": torch.zeros((N_SEQ, NKV_S, v90.V9_SEQ_LEN, HD_S), dtype=torch.float16)}
    cap_kv59 = {"k_rope": torch.zeros((N_SEQ, NKV_F, v90.V9_SEQ_LEN, HD_F), dtype=torch.float16), "v_norm": torch.zeros((N_SEQ, NKV_F, v90.V9_SEQ_LEN, HD_F), dtype=torch.float16)}
    crtr_layers, head_crtr_layers, c8db_entries, layer_logs = [], [], [], []
    va_set = set(va_seq_ids.tolist())

    # Replay clean Stage-2.5 E2E-tuned weights, update headers/INT8 LR, write T16C, and distill routers/C8DB
    bits_tbl, lv_tbls = make_unpack_tables(device, dtype=torch.float32)
    X_replay = X_stud_0.clone()
    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        emb_raw_r = sf.get_tensor("model.language_model.embed_tokens.weight").float()
        X_teach_r = (emb_raw_r[torch.from_numpy(all_seq_ids)] * ESCALE).to(torch.float16)[:, :, perm_pi]
        del emb_raw_r
        for si in range(N_TEXT_SEQ, N_SEQ):
            pm = torch.from_numpy(all_seq_ids[si] == soft_tok)
            X_teach_r[si, pm] = X_stud_0[si, pm]

        with open(tmp_t16c, "wb") as fp:
            fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, len(all_norms), total_mats, 0))
            pos = 24
            for nname, narr in all_norms:
                nb = nname.encode("utf-8")
                a32 = np.ascontiguousarray(narr, dtype="<f4").ravel()
                fp.write(struct.pack("<H", len(nb)))
                fp.write(nb)
                fp.write(struct.pack("<I", a32.size))
                fp.write(a32.tobytes())
                pos += 2 + len(nb) + 4 + a32.size * 4
            pos = p3._write_mat_record(fp, pos, "model.language_model.embed_tokens.weight", vocab_sz, D_MOD, V93_BLOCK_SIZE, 3 | p3.T16C_GAUSS_BIT, False, True, emb_pb, None, None)
            del emb_pb

            for l_idx, lyr in enumerate(e2e_layers):
                pfx = f"model.language_model.layers.{l_idx}"
                is_f = LAYER_TYPES[l_idx] == "full_attention"
                nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
                rot_d = int(round(hd * (PART_G if is_f else PART_L)))
                cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)
                rep = NH // nkv
                dn_t = layer_norms_raw[l_idx]

                # Teacher layer forward for exact C8DB targets & HeadRouter salience
                wq_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.q_proj", perm_pi_g, device, is_perm_col=True)
                wk_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.k_proj", perm_pi_g, device, is_perm_col=True)
                wv_t = wk_t if (is_f and K_EQ_V) else v90._load_dequant_proj(sf, f"{pfx}.self_attn.v_proj", perm_pi_g, device, is_perm_col=True)
                wo_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.o_proj", perm_pi_g, device, is_perm_row=True)
                pi_lg = pi_l_list[l_idx].to(device)
                wg_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.gate_proj", perm_pi_g, device, is_perm_col=True)[pi_lg]
                wu_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.up_proj", perm_pi_g, device, is_perm_col=True)[pi_lg]
                wd_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.down_proj", perm_pi_g, device, is_perm_row=True)[:, pi_lg]
                wo_head_norm = wo_t.view(D_MOD, NH, hd).pow(2).sum(dim=(0, 2)).sqrt().clamp(min=1e-6)

                X_t_next = torch.empty_like(X_teach_r)
                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        xb = X_teach_r[b0:b1].to(device, dtype=torch.float32)
                        xn = rmsn_gpu(xb, torch.from_numpy(dn_t["input_layernorm.weight"]).to(device))
                        q = apply_rope(rmsn_gpu(F.linear(xn, wq_t).view(b1 - b0, v90.V9_SEQ_LEN, NH, hd), torch.from_numpy(dn_t["self_attn.q_norm.weight"]).to(device)), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                        k = apply_rope(rmsn_gpu(F.linear(xn, wk_t).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), torch.from_numpy(dn_t["self_attn.k_norm.weight"]).to(device)), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                        v = rmsn_gpu(F.linear(xn, wv_t).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                        ctx_t = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous().view(b1 - b0, v90.V9_SEQ_LEN, NH * hd)
                        xm = xb + rmsn_gpu(F.linear(ctx_t, wo_t), torch.from_numpy(dn_t["post_attention_layernorm.weight"]).to(device))
                        xf = rmsn_gpu(xm, torch.from_numpy(dn_t["pre_feedforward_layernorm.weight"]).to(device))
                        act = F.gelu(F.linear(xf, wg_t), approximate="tanh") * F.linear(xf, wu_t)
                        X_t_next[b0:b1] = ((xm + rmsn_gpu(F.linear(act, wd_t), torch.from_numpy(dn_t["post_feedforward_layernorm.weight"]).to(device))) * float(dn_t["layer_scalar"][0])).to(torch.float16).cpu()
                del wq_t, wk_t, wv_t, wo_t, wg_t, wu_t, wd_t

                # Reconstruct exact exported student projections + write to T16C
                w_stud: Dict[str, torch.Tensor] = {}
                for pn, prec in lyr["projs"].items():
                    do, di, bsz_p, mode_p = int(prec["do"]), int(prec["di"]), int(prec["bsize"]), int(prec["mode_i"])
                    nb_m, asym_m = p3.MODE_TO_T16C[mode_p]
                    eff_m = nb_m & 0x1F
                    n_tiles, n_blks, tblk_b = (do + 15) // 16, di // bsz_p, p3.tile16_blk_bytes(bsz_p, nb_m, asym_m)
                    hdr_b = 64 if asym_m else 32
                    arr_pb = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
                    arr_pb[:, :, hdr_b:] = prec["planes_u8"].cpu().numpy().reshape(n_tiles, n_blks, tblk_b - hdr_b)
                    pb_flat = arr_pb.reshape(-1)
                    v90._c_update_t16c_headers(lib_fq, pb_flat, do, di, bsz_p, mode_p, prec["sc"], prec["bi"])
                    lr_t = quant_lr_int8_tuple(prec["u_lr"], prec["v_lr"]) if prec["u_lr"] is not None else None
                    hdr_bits, out_pb = v90.finalize_t16c_payload(pb_flat, do, di, bsz_p, mode_p, prec["is_rm"], prec["is_cb"])
                    inv_np = prec["inv_s"].detach().float().cpu().numpy().astype("<f4") if prec["inv_s"] is not None else None
                    pos = p3._write_mat_record(fp, pos, f"{pfx}.{pn}.weight", do, di, bsz_p, hdr_bits, asym_m, True, out_pb, inv_np, lr_t)

                    u_g, sc_g, bi_g = v90.unpack_tile16_headers_and_units(pb_flat, do, di, bsz_p, mode_p, H256_GPU)
                    inv_g = prec["inv_s"].to(device, dtype=torch.float32) if prec["inv_s"] is not None else None
                    w_eff = v90.reconstruct_from_headers(u_g, sc_g, bi_g, inv_g, H256_GPU)
                    lrd = v90._lr_tuple_to_dense(lr_t, device)
                    if lrd is not None:
                        w_eff = w_eff + lrd
                    w_stud[pn] = w_eff

                # Distill matched HeadRouter, CRTR-V9, and C8DB on Stage-2.5 Student replay
                g_in_s = lyr["g_in"].to(device, dtype=torch.float32)
                g_pa_s = lyr["g_pa"].to(device, dtype=torch.float32)
                g_pf_s = lyr["g_pf"].to(device, dtype=torch.float32)
                g_pff_s = lyr["g_pff"].to(device, dtype=torch.float32)
                g_qn_s = lyr["g_qn"].to(device, dtype=torch.float32)
                g_kn_s = lyr["g_kn"].to(device, dtype=torch.float32)
                lsc_s = float(lyr["layer_scalar"].item())

                xn_cpu = torch.empty_like(X_replay)
                xm_cpu = torch.empty_like(X_replay)
                xf_cpu = torch.empty_like(X_replay)
                head_e_cpu = torch.empty((N_SEQ * v90.V9_SEQ_LEN, NH), dtype=torch.float32)
                blk_e_cpu = torch.empty((N_SEQ * v90.V9_SEQ_LEN, V93_MLP_BLOCKS), dtype=torch.float32)
                mod_sal_cpu = torch.empty((N_SEQ * v90.V9_SEQ_LEN,), dtype=torch.float32)

                wq_s, wk_s = w_stud["self_attn.q_proj"], w_stud["self_attn.k_proj"]
                wv_s = wk_s if (is_f and K_EQ_V) else w_stud["self_attn.v_proj"]
                wo_s = w_stud["self_attn.o_proj"]
                wg_s, wu_s, wd_s = w_stud["mlp.gate_proj"], w_stud["mlp.up_proj"], w_stud["mlp.down_proj"]

                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        xb = X_replay[b0:b1].to(device, dtype=torch.float32)
                        xn = rmsn_gpu(xb, g_in_s)
                        xn_cpu[b0:b1] = xn.to(torch.float16).cpu()
                        q = apply_rope(rmsn_gpu(F.linear(xn, wq_s).view(b1 - b0, v90.V9_SEQ_LEN, NH, hd), g_qn_s), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                        k = apply_rope(rmsn_gpu(F.linear(xn, wk_s).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), g_kn_s), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                        v = rmsn_gpu(F.linear(xn, wv_s).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                        if l_idx in (58, 59):
                            for gi in range(b0, b1):
                                if gi in ast_set:
                                    cap = cap_kv58 if l_idx == 58 else cap_kv59
                                    cap["k_rope"][gi] = k[gi - b0].to(torch.float16).cpu()
                                    cap["v_norm"][gi] = v[gi - b0].to(torch.float16).cpu()
                        ctx_s = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous()
                        head_e_cpu[b0 * v90.V9_SEQ_LEN : b1 * v90.V9_SEQ_LEN] = (ctx_s.pow(2).sum(dim=-1).sqrt() * wo_head_norm.view(1, 1, NH)).reshape(-1, NH).cpu()
                        xm = xb + rmsn_gpu(F.linear(ctx_s.view(b1 - b0, v90.V9_SEQ_LEN, NH * hd), wo_s), g_pa_s)
                        xf = rmsn_gpu(xm, g_pf_s)
                        xm_cpu[b0:b1] = xm.to(torch.float16).cpu()
                        xf_cpu[b0:b1] = xf.to(torch.float16).cpu()
                        gg = F.gelu(F.linear(xf, wg_s), approximate="tanh").reshape(-1, V93_MLP_BLOCKS, V93_BLOCK_SIZE)
                        blk_e_cpu[b0 * v90.V9_SEQ_LEN : b1 * v90.V9_SEQ_LEN] = gg.pow(2).sum(dim=-1).cpu()
                        sal = (X_t_next[b0:b1].to(device, dtype=torch.float32) - xb).pow(2).sum(dim=-1).sqrt() / (xb.pow(2).sum(dim=-1).sqrt() + 1e-6)
                        mod_sal_cpu[b0 * v90.V9_SEQ_LEN : b1 * v90.V9_SEQ_LEN] = sal.reshape(-1).cpu()

                xn_f = xn_cpu.reshape(-1, D_MOD)[::2]
                he_f = head_e_cpu[::2]
                ntr_h = (xn_f.shape[0] * 9) // 10
                hr_net, hr_met = crtr.train_layer_router(xn_f[:ntr_h], he_f[:ntr_h], xn_f[ntr_h:], he_f[ntr_h:], top_p=float(V93_HEAD_ROUTED) / float(NH), rank=v90.V9_HEAD_CRTR_RANK, steps=120, lr=3e-3, device=str(device), seed=300 + l_idx)
                head_crtr_layers.append(crtr.quantize_layer_for_crtr(hr_net, V93_BLOCK_SIZE))

                xf_f = xf_cpu.reshape(-1, D_MOD)[::2]
                be_f = blk_e_cpu[::2]
                my_f = (mod_sal_cpu[::2] > float(torch.quantile(mod_sal_cpu, 0.25).item())).float()
                ntr_r = (xf_f.shape[0] * 9) // 10
                r_net, r_met = crtr.train_layer_router(xf_f[:ntr_r], be_f[:ntr_r], xf_f[ntr_r:], be_f[ntr_r:], top_p=float(V93_CRTR_TOP_BLOCKS) / float(V93_MLP_BLOCKS), rank=v90.V9_CRTR_RANK, steps=160, lr=3e-3, mod_targets_tr=my_f[:ntr_r], mod_targets_va=my_f[ntr_r:], device=str(device), seed=100 + l_idx)
                crtr_layers.append(crtr.quantize_layer_for_crtr(r_net, V93_BLOCK_SIZE))

                # Accept-only C8DB residual on Dense Stage-2.5 replay
                tr_c8 = tr_seq_ids[:: max(1, len(tr_seq_ids) // 128)][:128]
                atr, btr, xmtr, xttr, ava, bva, xmva, xtva = [], [], [], [], [], [], [], []
                with torch.no_grad():
                    for grp, da, db, dm, dt_l in ((tr_c8, atr, btr, xmtr, xttr), (va_seq_ids, ava, bva, xmva, xtva)):
                        for i0 in range(0, len(grp), MB_FWD):
                            sl = grp[i0 : i0 + MB_FWD]
                            xf_b = xf_cpu[sl].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                            act_b = F.gelu(F.linear(xf_b, wg_s), approximate="tanh") * F.linear(xf_b, wu_s)
                            da.append(act_b[::4].to(torch.float16))
                            db.append(F.linear(act_b, wd_s)[::4].to(torch.float16))
                            dm.append(xm_cpu[sl].to(device, dtype=torch.float32).reshape(-1, D_MOD)[::4].to(torch.float16))
                            dt_l.append(X_t_next[sl].to(device, dtype=torch.float32).reshape(-1, D_MOD)[::4].to(torch.float16))
                c8_res = v90.train_c8d_accept_only(torch.cat(atr, 0), torch.cat(btr, 0), torch.cat(xmtr, 0), torch.cat(xttr, 0), torch.cat(ava, 0), torch.cat(bva, 0), torch.cat(xmva, 0), torch.cat(xtva, 0), g_pff_s, lsc_s, rmsn_gpu, H8_GPU, D_MOD, V93_FF, rank=16, steps=45)
                ct_d = ex_d = None
                if c8_res is not None:
                    ct_d, ex_d, _, _ = c8_res
                    exp_r = [c8d._c6d_radicals("general", "core", "transform", "native", "layer", "positive", e) for e in range(8)]
                    c8db_entries.append({"name": f"{pfx}.mlp.down_proj.weight", "d_out": D_MOD, "d_in": V93_FF, "rank": 16, "top_k": 2, "c6d": exp_r[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": ct_d, "expert_radicals": exp_r, "experts": ex_d})

                with torch.no_grad():
                    for b0 in range(0, N_SEQ, MB_FWD):
                        b1 = min(b0 + MB_FWD, N_SEQ)
                        xm_b = xm_cpu[b0:b1].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                        xf_b = xf_cpu[b0:b1].to(device, dtype=torch.float32).reshape(-1, D_MOD)
                        act_q = F.gelu(F.linear(xf_b, wg_s), approximate="tanh") * F.linear(xf_b, wu_s)
                        down_q = F.linear(act_q, wd_s)
                        if ct_d is not None and ex_d is not None:
                            down_q = down_q + v90._eval_c8d_gpu(act_q, ct_d, ex_d, H8_GPU)
                        X_replay[b0:b1] = ((xm_b + rmsn_gpu(down_q, g_pff_s)) * lsc_s).reshape(b1 - b0, v90.V9_SEQ_LEN, D_MOD).to(torch.float16).cpu()

                X_teach_r = X_t_next
                cos_rep = float(F.cosine_similarity(X_replay[-32:].reshape(-1, D_MOD).float(), X_teach_r[-32:].reshape(-1, D_MOD).float(), dim=-1).mean().item())
                layer_logs.append({"l": l_idx, "vlm_txt_cos": round(cos_rep, 5), "crtr_recall": r_met["val_recall_after"], "head_crtr_recall": hr_met["val_recall_after"]})
                print(f"  [V9.3 Replay L{l_idx:02d}/59] cos={cos_rep:.5f} crtr_rec={r_met['val_recall_after']:.3f} hr_rec={hr_met['val_recall_after']:.3f}", flush=True)
                del w_stud, xn_cpu, xm_cpu, xf_cpu, head_e_cpu, blk_e_cpu, mod_sal_cpu
                gc.collect()
                torch.cuda.empty_cache()

            for rec_v in vision_mat_recs:
                pos = p3._write_mat_record(fp, pos, *rec_v)

    os.replace(tmp_t16c, OUT_T16C)
    vm_c, ve_c = v70.verify_t16c_integrity(OUT_T16C)
    assert vm_c == total_mats and ve_c == OUT_T16C.stat().st_size, f"Bad V9.3 T16C {vm_c}/{total_mats}"
    cb_cnt = sdc.verify_t16c_file(OUT_T16C)["colblock_mats"]

    ctx_s3 = {
        "sf_path": sf_path, "tok_path": tok_path, "D_MOD": D_MOD, "EPS": EPS, "ESCALE": ESCALE,
        "NH": NH, "NKV_S": NKV_S, "NKV_F": NKV_F, "HD_S": HD_S, "HD_F": HD_F,
        "THETA_L": THETA_L, "PART_L": PART_L, "THETA_G": THETA_G, "PART_G": PART_G,
        "SOFTCAP": SOFTCAP, "all_ans_mask": all_ans_mask, "all_seq_ids": all_seq_ids,
        "all_seq_nxt": all_seq_nxt, "perm_pi": perm_pi,
    }
    rpt = v90._run_stage3_distill_and_export(
        ctx=ctx_s3, device=device, compute_dtype=compute_dtype, gpu_name=gpu_name, gpu_vram_gb=gpu_vram_gb, t0_all=t0_all,
        rmsn_gpu=rmsn_gpu, X_stud_perm=X_replay, X_teach_perm=X_teach_r, emb_eff=emb_eff, ast_seqs=ast_seqs,
        cap_kv58=cap_kv58, cap_kv59=cap_kv59, fn_vec_np=fn_param.detach().cpu().numpy().astype(np.float32),
        crtr_layers=crtr_layers, head_crtr_layers=head_crtr_layers, c8db_entries=c8db_entries,
        layer_logs=layer_logs, cb_cnt=cb_cnt, t16c_final_path=OUT_T16C, head_crtr_final_path=OUT_HEAD_CRTR,
        persist_t16c_and_head=True, apply_wiener=False, train_post_cell60=True, mode_tag="-V9.3-E2E-KL",
        crtr_final_path=OUT_CRTR, c8d_final_path=OUT_C8D, ast_q2_path=OUT_AST_Q2, ast_q4_path=OUT_AST_Q4,
        bhdr_path=OUT_BHDR, report_path=OUT_REPORT, sums_path=OUT_DIR / "SHA256SUMS_V9_3.txt",
        train_assistant=True, auto_unassign=False,
    )
    rpt["e2e_stage2_5"] = e2e_metrics
    OUT_REPORT.write_text(json.dumps(rpt, indent=2), encoding="utf-8")
    if DRIVE_DIR is not None:
        v90._drive_safe_copy(OUT_REPORT, DRIVE_DIR / OUT_REPORT.name)
        expected_sha = {k: v["sha256"] for k, v in rpt["artifacts"].items()}
        expected_sha[OUT_REPORT.name] = v90._sha256_file(OUT_REPORT)
        v90._verify_drive_sha256(DRIVE_DIR, expected_sha)
    if os.environ.get("COLAB_AUTO_UNASSIGN", "1") == "1":
        v90._flush_drive_before_unassign()
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:
            print(f"[Auto-Unassign Note] {exc}", flush=True)
    return rpt


def main() -> int:
    ap = argparse.ArgumentParser(description="Language-U Gemma 4 31B V9.3 Zero-Tern5 E2E Logit-KL QAT")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--e2e-steps", type=int, default=240)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()
    if args.selftest or (not args.run and not torch.cuda.is_available()):
        return run_selftest()
    run_v93_pipeline(e2e_steps=args.e2e_steps, micro_bs=args.micro_bs, grad_accum=args.grad_accum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

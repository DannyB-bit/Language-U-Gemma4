#!/usr/bin/env python3
"""Language-U V10: 3-Tier Cortical Cascade (<= 6.30 GiB T16C + Dual-Regime E2E KL QAT + Target-Shared EAGLE-3).

Optimizes all 60 layers of Gemma 4 31B VLM on NVIDIA L4 High-RAM (22.5 GiB VRAM / 53 GiB RAM)
within the strict Raspberry Pi 4 <= 6.50 GiB total resident footprint:
  1. Sub-6.30 GiB Zero-Tern5 Byte Budget:
     All 58 non-boundary Attention layers in COLBLOCK 2b_gauss (0xA2, zero Ternary-5, zero 1-bit attn);
     12 cliff-sensitive MLP layers (L0, L1, L48..L57) in 2b_gauss COLBLOCK;
     L2..L47 in fast 1b_rm COLBLOCK (0x61 r=16); boundary L58..L59 in 3b + 3b-gauss Vision (0.22 GiB).
     Leaves >= 240 MiB under 6.50 GiB for Assistant-V10 Q2 (~56 MiB, 0 MB embed via target sharing),
     CRTR, HeadRouter, C8DB, and GlyphKV.
  2. Dual-Regime 60-Layer E2E Logit-KL + Early-Exit (L29) QAT:
     Jointly trains bitplane headers (sc, bi), STE INT8 low-rank (U, V), RMSNorms, layer_scalar,
     final_norm, and `early_exit_l29_norm` across alternating Dense (60/60) and Tier-2 Sparse
     (12/84 MLP + 8/32 Head) micro-batches using L4-resident bitplanes + High-RAM pinned caches.
  3. Pretrained `google/gemma-4-31B-it-assistant` Target-Embed EAGLE-3 Alignment:
     Permutes official 4-layer assistant pre/post projections into V10's `perm_pi` basis and trains
     3-step self-feeding rollout directly against Target-Embed logits (`rmsn(post(h_final), fn) @ emb.T`).

Enforces Rule 2 (strict 0-overlap 4-gram decontamination), Rule 3 (authentic PyTorch backprop only),
Rule 4 (fail-closed artifact/budget verification), and 6-gate CPU `--selftest`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pack_assistant_t16c as pa  # noqa: E402
import pack_t16c_v3 as p3  # noqa: E402
import run_v7_0_fullrank_crtr_gpu as v70  # noqa: E402
import run_v9_0_sparse_spinquant_qat_gpu as v90  # noqa: E402
import run_v9_3_e2e_kl_qat_gpu as v93  # noqa: E402
import spatial_district_converter as sdc  # noqa: E402
import train_assistant_v7_byte_head as bhmod  # noqa: E402
import train_c8d_pattern_bank as c8d  # noqa: E402
import train_cortical_router as crtr  # noqa: E402

V10_BLOCK_SIZE = v90.V9_BLOCK_SIZE
V10_D_MOD = v90.V9_D_MOD
V10_FF = v90.V9_FF
V10_N_LAYERS = v90.V9_N_LAYERS
V10_MLP_BLOCKS = v90.V9_MLP_BLOCKS
V10_SHARED_BLOCKS = v90.V9_SHARED_BLOCKS
V10_CRTR_TOP_BLOCKS = 12   # 12/84 = 14.3% Tier-2 active MLP blocks (target 10..16/84)
V10_HEAD_ROUTED = 8        # 8/32 = 25.0% Tier-2 active attention heads
V10_EARLY_LAYER = 29       # Global full-attention midpoint layer for early-exit confidence gating
GLOBAL_ATTN_LAYERS = set(v90.GLOBAL_ATTN_LAYERS)

# Sub-6.30 GiB layout: entry L0..1 + cliff L48..57 in 2b_gauss COLBLOCK; L58..59 in 3b; L2..47 in 1b_rm COLBLOCK
V10_2B_MLP_LAYERS = {0, 1} | set(range(48, 58))
MAX_V10_T16C_BYTES = int(6.30 * (1024**3))

OUT_DIR = Path(os.environ.get("V10_OUT_DIR", "/content/v10_outputs"))
CKPT_DIR = OUT_DIR / "ckpt_v10"
_drv_env = os.environ.get("V10_DRIVE_DIR", "").strip()
DRIVE_DIR: Optional[Path] = Path(_drv_env) if _drv_env else None

OUT_T16C = OUT_DIR / "gemma4_31b_v10_cascade.t16c"
OUT_CRTR = OUT_DIR / "gemma4_31b_v10_router.crtr"
OUT_HEAD_CRTR = OUT_DIR / "gemma4_31b_v10_head_router.crtr"
OUT_C8D = OUT_DIR / "gemma4_31b_v10_c8d.c8d"
OUT_AST_Q2 = OUT_DIR / "gemma4_31b_v10_assistant_body_q2.t16c"
OUT_AST_Q4 = OUT_DIR / "gemma4_31b_v10_assistant_body_q4.t16c"
OUT_BHDR = OUT_DIR / "gemma4_31b_v10_assistant.bhdr"
OUT_REPORT = OUT_DIR / "v10_report.json"


def get_v10_proj_spec(l_idx: int, pname: str, is_full: Optional[bool] = None) -> Tuple[int, int, int, bool, bool]:
    """Return `(mode_i, block_size, lr_rank, is_rowmajor, is_colblock)` for V10 sub-6.30 GiB Cascade."""
    if l_idx >= 58:
        lr_rk = 24 if ("k_proj" in pname or "v_proj" in pname) else 16
        return (4, V10_BLOCK_SIZE, lr_rk, False, ("o_proj" in pname or "down_proj" in pname))
    if pname.startswith("self_attn."):
        lr_rk = 16 if ("q_proj" in pname or "o_proj" in pname) else 8
        return (5, V10_BLOCK_SIZE, lr_rk, False, True)
    if l_idx in V10_2B_MLP_LAYERS:
        return (5, V10_BLOCK_SIZE, 16, False, True)
    return (0, V10_BLOCK_SIZE, 16, True, True)


def compute_v10_budget_bytes(
    *,
    vocab_size: int = 262144,
    d_mod: int = V10_D_MOD,
    n_layers: int = V10_N_LAYERS,
    nh: int = 32,
    nkv_s: int = 16,
    nkv_f: int = 4,
    hd_s: int = 256,
    hd_f: int = 512,
    ff: int = V10_FF,
    vision_bytes: int = 237_253_152,
) -> Dict[str, object]:
    """Exact fail-closed byte census for V10 `.t16c` (Zero Tern5, <= 6.30 GiB total)."""
    embed_b = p3.expected_tile16_bytes(vocab_size, d_mod, V10_BLOCK_SIZE, 3, False) + 256
    attn_b = mlp_b = lr_b = 0
    tern5_cnt = attn_1b_cnt = mlp_1b_upper_cnt = mlp_2b_layers = mlp_1b_layers = 0

    for l_idx in range(n_layers):
        is_f = l_idx in GLOBAL_ATTN_LAYERS
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
            mode, bsz, lrr, _, _ = get_v10_proj_spec(l_idx, pn, is_f)
            nb, asym = p3.MODE_TO_T16C[mode]
            if mode == 1:
                tern5_cnt += 1
            if (nb & 0x1F) == 1:
                attn_1b_cnt += 1
            attn_b += p3.expected_tile16_bytes(do, di, bsz, nb, asym) + 128 + di * 4
            if lrr > 0:
                lr_b += do * 4 + lrr * 4 + do * lrr + lrr * di

        m_gate, _, _, _, _ = get_v10_proj_spec(l_idx, "mlp.gate_proj", is_f)
        if m_gate == 5:
            mlp_2b_layers += 1
        elif m_gate == 0:
            mlp_1b_layers += 1
            if l_idx > 47:
                mlp_1b_upper_cnt += 1
        for pn, do, di in (
            ("mlp.gate_proj", ff, d_mod),
            ("mlp.up_proj", ff, d_mod),
            ("mlp.down_proj", d_mod, ff),
        ):
            mode, bsz, lrr, _, _ = get_v10_proj_spec(l_idx, pn, is_f)
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


def permute_assistant_state_dict(ast_sd: Dict[str, torch.Tensor], perm_pi: torch.Tensor, d_mod: int = V10_D_MOD) -> Dict[str, torch.Tensor]:
    """Fold V10's global coordinate permutation `perm_pi` into `pre_projection` and `post_projection`."""
    p2 = torch.cat([perm_pi.long(), d_mod + perm_pi.long()])
    out: Dict[str, torch.Tensor] = {}
    for k, v in ast_sd.items():
        if k == "pre_projection.weight" and v.shape[1] == 2 * d_mod:
            out[k] = v[:, p2].contiguous()
        elif k == "post_projection.weight" and v.shape[0] == d_mod:
            out[k] = v[perm_pi.long(), :].contiguous()
        else:
            out[k] = v
    return out


def run_stage2_5_v10_dual_e2e_kl(
    *,
    e2e_layers: List[dict],
    X_stud_0: torch.Tensor,
    X_teach_l59: torch.Tensor,
    top_ids_cpu: torch.Tensor,
    top_probs_cpu: torch.Tensor,
    ans_mask_np: np.ndarray,
    emb_eff: torch.Tensor,
    fn_param: nn.Parameter,
    early_fn_param: nn.Parameter,
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
    n_steps: int = 200,
    micro_bs: int = 8,
    grad_accum: int = 2,
) -> Dict[str, object]:
    """Dual-regime 60-layer E2E Logit-KL + L29 early-exit QAT with High-RAM pinning & OOM backoff."""
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
    bits_tbl, lv_tbls = v93.make_unpack_tables(device, dtype=dt)
    h_mat_c = H256_GPU.to(device=device, dtype=dt)
    emb_gpu = emb_eff.to(device=device, dtype=dt)

    # Only pin all 60 bitplane layers directly in VRAM when >= 32 GiB (e.g. A100/H100).
    # On 22.5 GiB L4 with 53 GB High-RAM, pin_memory() host streaming avoids 98%-VRAM UVM thrashing.
    pin_on_gpu = (device.type == "cuda") and (gpu_vram_gb >= 32.0)
    if device.type == "cuda":
        X_stud_0 = X_stud_0.pin_memory()
        X_teach_l59 = X_teach_l59.pin_memory()
        top_ids_cpu = top_ids_cpu.pin_memory()
        top_probs_cpu = top_probs_cpu.pin_memory()

    sc_params: List[torch.nn.Parameter] = []
    lr_params: List[torch.nn.Parameter] = []
    norm_params: List[torch.nn.Parameter] = [fn_param, early_fn_param]

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
    tok_weights_all = torch.where(
        ans_mask_t,
        torch.full_like(ans_mask_t, 1.5, dtype=torch.float32),
        torch.full_like(ans_mask_t, 0.5, dtype=torch.float32),
    )

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

    def _forward_model(
        x_in: torch.Tensor,
        head_kf: float = 1.0,
        mlp_kf: float = 1.0,
        use_ckpt: bool = True,
        capture_layer: Optional[int] = V10_EARLY_LAYER,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = x_in.float()
        h_early: Optional[torch.Tensor] = None
        for li in range(N_LAYERS):
            lyr = e2e_layers[li]
            is_f = LAYER_TYPES[li] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)

            def _fn(hx, _li=li, _lyr=lyr, _isf=is_f, _nkv=nkv, _hd=hd, _rd=rot_d, _cs=cs_r, _sn=sn_r):
                return v93.forward_layer_e2e(
                    hx, _lyr, is_f=_isf, k_eq_v=K_EQ_V, nh=NH, nkv=_nkv, hd=_hd, rot_d=_rd,
                    cs_r=_cs, sn_r=_sn, rmsn_fn=rmsn_gpu, apply_rope_fn=apply_rope,
                    bits_tbl=bits_tbl, lv_tbls=lv_tbls, h_mat=h_mat_c,
                    head_keep_frac=head_kf, mlp_keep_frac=mlp_kf, use_ste_lr=True,
                )

            if use_ckpt:
                h = torch.utils.checkpoint.checkpoint(_fn, h, use_reentrant=False)
            else:
                h = _fn(h)
            if capture_layer is not None and li == capture_layer:
                h_early = h
        return h, h_early

    def _eval_val(bs_eval: int) -> Tuple[float, float, float, float, float]:
        kl_list: List[float] = []
        kl_sp_list: List[float] = []
        kl_early_list: List[float] = []
        cos_list: List[float] = []
        top1_hits = top1_tot = 0
        with torch.no_grad():
            for i0 in range(0, int(va_seq_ids.numel()), bs_eval):
                sl = va_seq_ids[i0 : i0 + bs_eval]
                xb = X_stud_0[sl].to(device=device, dtype=torch.float32, non_blocking=True)
                xt = X_teach_l59[sl].to(device=device, dtype=torch.float32, non_blocking=True)
                t_id = top_ids_cpu[sl].to(device=device, non_blocking=True)
                t_pr = top_probs_cpu[sl].to(device=device, dtype=torch.float32, non_blocking=True)
                tw = tok_weights_all[sl].to(device=device)

                hs, he = _forward_model(xb, 1.0, 1.0, use_ckpt=False)
                hn = rmsn_gpu(hs, fn_param).to(dt)
                lgs = SOFTCAP * torch.tanh((hn @ emb_gpu.T).float() / SOFTCAP)
                kl_list.append(float(v93.compute_topk_kl_loss(lgs, t_id, t_pr, tw).item()))
                cos_list.append(float(F.cosine_similarity(hs.reshape(-1, hs.shape[-1]), xt.reshape(-1, xt.shape[-1]), dim=-1).mean().item()))
                if he is not None:
                    hne = rmsn_gpu(he, early_fn_param).to(dt)
                    lg_e = SOFTCAP * torch.tanh((hne @ emb_gpu.T).float() / SOFTCAP)
                    kl_early_list.append(float(v93.compute_topk_kl_loss(lg_e, t_id, t_pr, tw).item()))

                pred_top1 = lgs.argmax(dim=-1)
                teach_top1 = t_id[:, :, 0].long()
                m = ans_mask_t[sl].to(device=device)
                if m.any():
                    top1_hits += int((pred_top1[m] == teach_top1[m]).sum().item())
                    top1_tot += int(m.sum().item())

                hs_sp, _ = _forward_model(xb, 0.25, float(V10_CRTR_TOP_BLOCKS) / float(V10_MLP_BLOCKS), use_ckpt=False, capture_layer=None)
                hn_sp = rmsn_gpu(hs_sp, fn_param).to(dt)
                lgs_sp = SOFTCAP * torch.tanh((hn_sp @ emb_gpu.T).float() / SOFTCAP)
                kl_sp_list.append(float(v93.compute_topk_kl_loss(lgs_sp, t_id, t_pr, tw).item()))

        return (
            float(np.mean(kl_list)),
            float(np.mean(kl_sp_list)),
            float(np.mean(kl_early_list)) if kl_early_list else 0.0,
            float(np.mean(cos_list)),
            float(top1_hits) / float(max(top1_tot, 1)),
        )

    def _snapshot_best() -> Dict[int, torch.Tensor]:
        return {id(p): p.detach().cpu().clone() for p in (sc_params + lr_params + norm_params)}

    def _restore_best(snap: Dict[int, torch.Tensor]) -> None:
        with torch.no_grad():
            for p in (sc_params + lr_params + norm_params):
                p.copy_(snap[id(p)].to(device=device, dtype=p.dtype))

    cur_mbs = micro_bs
    cur_acc = grad_accum
    kl_before, kl_sp_before, kl_e_before, cos_before, top1_before = _eval_val(min(4, cur_mbs))
    best_obj = 0.75 * kl_before + 0.25 * kl_sp_before
    best_snap = _snapshot_best()
    print(
        f"[V10 Stage 2.5 Dual E2E Init] dense_kl={kl_before:.4f} | sp12_kl={kl_sp_before:.4f} | "
        f"l29_kl={kl_e_before:.4f} | l59_cos={cos_before:.5f} | top1={top1_before*100:.2f}% | "
        f"pin_gpu={pin_on_gpu} mbs={cur_mbs}x{cur_acc}",
        flush=True,
    )

    rng = np.random.default_rng(1337)
    tr_np = tr_seq_ids.numpy()
    t0_e2e = time.perf_counter()

    for step in range(n_steps):
        mult = _lr_mult(step)
        for pg, blr in zip(opt.param_groups, (1.5e-4, 1.5e-4, 3.0e-4)):
            pg["lr"] = blr * mult

        while True:
            opt.zero_grad(set_to_none=True)
            step_kl = 0.0
            try:
                for acc_i in range(cur_acc):
                    sel = rng.choice(tr_np, size=min(cur_mbs, len(tr_np)), replace=False)
                    sl = torch.from_numpy(sel).long()
                    xb = X_stud_0[sl].to(device=device, dtype=torch.float32, non_blocking=True)
                    xt = X_teach_l59[sl].to(device=device, dtype=torch.float32, non_blocking=True)
                    t_id = top_ids_cpu[sl].to(device=device, non_blocking=True)
                    t_pr = top_probs_cpu[sl].to(device=device, dtype=torch.float32, non_blocking=True)
                    tw = tok_weights_all[sl].to(device=device)

                    # Alternate Dense (60/60) and Tier-2 Sparse (12/84 MLP + 8/32 Head) after warmup
                    do_sp = (step >= max(20, n_steps // 5)) and (acc_i % 2 == 1)
                    hk = 0.25 if do_sp else 1.0
                    mk = (float(V10_CRTR_TOP_BLOCKS) / float(V10_MLP_BLOCKS)) if do_sp else 1.0
                    regime_w = 0.35 if do_sp else 1.0

                    hs, he = _forward_model(xb, hk, mk, use_ckpt=True, capture_layer=(None if do_sp else V10_EARLY_LAYER))
                    hn = rmsn_gpu(hs, fn_param).to(dt)
                    lgs = SOFTCAP * torch.tanh((hn @ emb_gpu.T).float() / SOFTCAP)
                    kl = v93.compute_topk_kl_loss(lgs, t_id, t_pr, tw)
                    cos_l = 1.0 - F.cosine_similarity(hs.reshape(-1, hs.shape[-1]), xt.reshape(-1, xt.shape[-1]), dim=-1).mean()
                    mag_t = torch.log(xt.pow(2).sum(dim=-1).clamp(min=1e-8))
                    mag_s = torch.log(hs.pow(2).sum(dim=-1).clamp(min=1e-8))
                    loss_sub = kl + 0.15 * cos_l + 0.02 * F.mse_loss(mag_s, mag_t)
                    if he is not None:
                        hne = rmsn_gpu(he, early_fn_param).to(dt)
                        lg_e = SOFTCAP * torch.tanh((hne @ emb_gpu.T).float() / SOFTCAP)
                        loss_sub = loss_sub + 0.15 * v93.compute_topk_kl_loss(lg_e, t_id, t_pr, tw)
                    loss = (loss_sub * regime_w) / float(cur_acc)
                    loss.backward()
                    step_kl += float(kl.detach().item()) / float(cur_acc)
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if cur_mbs <= 2:
                    raise
                cur_mbs = max(2, cur_mbs // 2)
                cur_acc = min(8, cur_acc * 2)
                print(f"[Stage 2.5 OOM Backoff] Switched to micro_bs={cur_mbs}, grad_accum={cur_acc}", flush=True)

        torch.nn.utils.clip_grad_norm_(sc_params + lr_params + norm_params, 1.0)
        opt.step()
        with torch.no_grad():
            for lyr in e2e_layers:
                for p in lyr["projs"].values():
                    p["sc"].clamp_(min=0.0, max=65000.0)
                    if p["bi"] is not None:
                        p["bi"].clamp_(min=-65000.0, max=65000.0)
                lyr["layer_scalar"].clamp_(min=-65000.0, max=65000.0)

        if step == 0 or (step + 1) % 10 == 0 or (step + 1) == n_steps:
            if (step + 1) % 25 == 0 or (step + 1) == n_steps:
                v_kl, v_sp, v_e, v_cos, v_t1 = _eval_val(min(4, cur_mbs))
                obj = 0.75 * v_kl + 0.25 * v_sp
                improved = ""
                if obj < best_obj:
                    best_obj = obj
                    best_snap = _snapshot_best()
                    improved = " [BEST]"
                print(
                    f"  [V10 E2E step {step+1:03d}/{n_steps}] tr_kl={step_kl:.4f} dense_kl={v_kl:.4f} "
                    f"sp12_kl={v_sp:.4f} l29_kl={v_e:.4f} top1={v_t1*100:.2f}% dt={time.perf_counter()-t0_e2e:.1f}s{improved}",
                    flush=True,
                )
            else:
                print(
                    f"  [V10 E2E step {step+1:03d}/{n_steps}] tr_kl={step_kl:.4f} mbs={cur_mbs}x{cur_acc} dt={time.perf_counter()-t0_e2e:.1f}s",
                    flush=True,
                )

    _restore_best(best_snap)
    kl_after, kl_sp_after, kl_e_after, cos_after, top1_after = _eval_val(min(4, cur_mbs))
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

    print(
        f"[V10 Stage 2.5 Done] dense_kl: {kl_before:.4f}->{kl_after:.4f} | "
        f"sp12_kl: {kl_sp_before:.4f}->{kl_sp_after:.4f} | top1: {top1_before*100:.2f}%->{top1_after*100:.2f}%",
        flush=True,
    )
    return {
        "val_kl_before": round(kl_before, 4),
        "val_kl_after": round(kl_after, 4),
        "val_sp12_kl_before": round(kl_sp_before, 4),
        "val_sp12_kl_after": round(kl_sp_after, 4),
        "val_l29_kl_after": round(kl_e_after, 4),
        "l59_cos_before": round(cos_before, 5),
        "l59_cos_after": round(cos_after, 5),
        "top1_agree_before": round(top1_before, 4),
        "top1_agree_after": round(top1_after, 4),
        "effective_micro_bs": cur_mbs,
        "effective_grad_accum": cur_acc,
        "e2e_elapsed_s": round(time.perf_counter() - t0_e2e, 2),
    }


def run_selftest() -> int:
    """Mandatory 6-gate fail-closed CPU verification battery for V10 Cortical Cascade."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Strict V10 budget <= 6.30 GiB, Zero Tern5, Zero 1b Attn, Zero cliff above L47
    bgt = compute_v10_budget_bytes()
    g1_ok = (
        int(bgt["total_t16c_bytes"]) <= MAX_V10_T16C_BYTES
        and int(bgt["tern5_cnt"]) == 0
        and int(bgt["attn_1b_cnt"]) == 0
        and int(bgt["mlp_1b_upper_cnt"]) == 0
    )
    gate(
        "Gate 1: V10 Budget <= 6.30 GiB & Zero-Tern5 / Zero-Cliff",
        g1_ok,
        f"total={bgt['total_t16c_gib']} GiB, text={bgt['text_resident_gib']} GiB, 2b_mlps={bgt['mlp_2b_gauss_layers']}",
    )

    # Gate 2: Vectorized bitplane unpack bit-exact parity across 1b/2b_gauss/3b
    rng = np.random.default_rng(10)
    h256 = torch.from_numpy(pa.make_sylvester_hadamard(V10_BLOCK_SIZE)).float()
    bits_tbl, lv_tbls = v93.make_unpack_tables(torch.device("cpu"), dtype=torch.float32)
    max_diff = 0.0
    for mode_i in (0, 5, 4, 6):
        do, di = 32, 512
        nb_m, asym_m = p3.MODE_TO_T16C[mode_i]
        n_tiles, n_blks = do // 16, di // V10_BLOCK_SIZE
        tblk_b = p3.tile16_blk_bytes(V10_BLOCK_SIZE, nb_m, asym_m)
        pb = rng.integers(0, 256, size=n_tiles * n_blks * tblk_b, dtype=np.uint8)
        pb_r = pb.reshape(n_tiles, n_blks, tblk_b)
        pb_r[:, :, :32] = np.full((n_tiles, n_blks, 16), 0.05, dtype="<f2").view(np.uint8).reshape(n_tiles, n_blks, 32)
        if asym_m:
            pb_r[:, :, 32:64] = 0
        u_ref, _, _ = v90.unpack_tile16_headers_and_units(pb, do, di, V10_BLOCK_SIZE, mode_i, h256)
        pl = v93.extract_planes_u8(pb, do, di, V10_BLOCK_SIZE, mode_i)
        u_got = v93.unpack_planes_to_units(pl, mode_i, do, di, V10_BLOCK_SIZE, bits_tbl, lv_tbls[mode_i])
        max_diff = max(max_diff, float((u_ref.reshape(do, n_blks, V10_BLOCK_SIZE) - u_got).abs().max().item()))
    gate("Gate 2: Bitplane Unpack Parity (1b/2b_gauss/3b)", max_diff < 1e-6, f"max_diff={max_diff:.2e}")

    # Gate 3: STE INT8 Low-Rank exact export parity
    u_lr = torch.from_numpy((rng.standard_normal((32, 8)) * 0.04).astype(np.float32))
    v_lr = torch.from_numpy((rng.standard_normal((8, 64)) * 0.04).astype(np.float32))
    u_ste, v_ste = v93.ste_int8_lr(u_lr, v_lr)
    su, sv, ui8, vi8 = v93.quant_lr_int8_tuple(u_lr, v_lr)
    w_ste = u_ste @ v_ste
    w_exp = (torch.from_numpy(ui8.astype(np.float32)) * torch.from_numpy(su).unsqueeze(1)) @ (
        torch.from_numpy(vi8.astype(np.float32)) * torch.from_numpy(sv).unsqueeze(1)
    )
    lr_diff = float((w_ste - w_exp).abs().max().item())
    gate("Gate 3: STE INT8 Low-Rank Export Parity", lr_diff < 1e-6, f"max_diff={lr_diff:.2e}")

    # Gate 4: Authentic Dual-Regime E2E BitSerial KL convergence (dense + sparse mask)
    do, di, bsz = 64, 64, 64
    h64 = torch.from_numpy(pa.make_sylvester_hadamard(bsz)).float()
    x0 = torch.randn(12, di)
    w_t = torch.randn(do, di) * 0.15
    emb_t = torch.randn(128, do) * 0.1
    lgt = 30.0 * torch.tanh(((x0 @ w_t.T) @ emb_t.T) / 30.0)
    tp_all, tid_all = torch.topk(F.softmax(lgt, dim=-1), 32, dim=-1)
    pl_m = torch.from_numpy(rng.integers(0, 256, size=(do // 16, di // bsz, 2, bsz // 8, 16), dtype=np.uint8))
    proj_m = {
        "planes_u8": pl_m,
        "sc": nn.Parameter(torch.full((do, di // bsz), 0.05)),
        "bi": None,
        "inv_s": None,
        "u_lr": nn.Parameter(torch.randn(do, 4) * 0.01),
        "v_lr": nn.Parameter(torch.randn(4, di) * 0.01),
        "mode_i": 5,
    }
    opt_m = torch.optim.AdamW([proj_m["sc"], proj_m["u_lr"], proj_m["v_lr"]], lr=2e-2)
    kl0 = kl1 = 0.0
    for s_i in range(30):
        opt_m.zero_grad()
        hs_m = v93.bitserial_linear(x0, proj_m, bits_tbl, lv_tbls, h64, use_ste_lr=True)
        if s_i % 2 == 1:
            hs_m = hs_m * torch.tensor(([1.0] * 32 + [0.0] * 32), dtype=torch.float32)
        lgs_m = 30.0 * torch.tanh((hs_m @ emb_t.T) / 30.0)
        loss_m = v93.compute_topk_kl_loss(lgs_m, tid_all, tp_all)
        if s_i == 0:
            kl0 = float(loss_m.item())
        loss_m.backward()
        opt_m.step()
        if s_i % 2 == 0:
            kl1 = float(loss_m.item())
    gate("Gate 4: Dual-Regime BitSerial Real KL Convergence", kl1 < 0.65 * kl0, f"kl0={kl0:.4f} -> kl1={kl1:.4f}")

    # Gate 5: Strict Rule 2 decontamination check
    hold = v90.collect_holdout_4grams()
    ok_clean = True
    try:
        v90.verify_corpus_decontaminated(["Clean engineering derivation on orthogonal transforms."], hold)
    except Exception:
        ok_clean = False
    ok_dirty = False
    try:
        v90.verify_corpus_decontaminated(["If 5x + 3 = 38, what is the value of 2x - 1?"], hold)
    except (ValueError, AssertionError):
        ok_dirty = True
    gate("Gate 5: Rule 2 Decontamination Gate", ok_clean and ok_dirty, f"hold_4grams={len(hold)}")

    # Gate 6: Target-Embed Permutation Invariance + CRTR 12/84 Export Check
    d_toy = 64
    perm = torch.randperm(d_toy)
    x_a = torch.randn(4, d_toy)
    x_b = torch.randn(4, d_toy)
    pre_w = torch.randn(32, 2 * d_toy)
    post_w = torch.randn(d_toy, 32)
    sd_perm = permute_assistant_state_dict({"pre_projection.weight": pre_w, "post_projection.weight": post_w}, perm, d_toy)
    y_orig = F.linear(F.linear(torch.cat([x_a, x_b], dim=-1), pre_w), post_w)
    y_perm = F.linear(F.linear(torch.cat([x_a[:, perm], x_b[:, perm]], dim=-1), sd_perm["pre_projection.weight"]), sd_perm["post_projection.weight"])
    perm_err = float((y_orig[:, perm] - y_perm).abs().max().item())
    rn, _ = crtr.train_layer_router(x_a.repeat(16, 1), torch.rand(64, 16), x_a.repeat(4, 1), torch.rand(16, 16), top_p=0.143, rank=8, steps=10, device="cpu", seed=7)
    rq = crtr.quantize_layer_for_crtr(rn, 256)
    gate("Gate 6: Assistant Perm-Pi Equivariance & Router Non-Degeneracy", perm_err < 1e-4 and bool(np.all(np.isfinite(rq["sv"])) and rq["v_i8"].std() > 0), f"perm_err={perm_err:.2e}")

    print(f"V10 Self-Test Result: {6 - failures}/6 gates passed", flush=True)
    return failures


def run_v10_pipeline(*, e2e_steps: int = 200, micro_bs: int = 8, grad_accum: int = 2) -> Dict[str, object]:
    """Execute complete V10 Sub-6.30 GiB build + High-RAM/L4 E2E Dual KL + Target-Shared EAGLE-3 Assistant."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file as st_save_file

    t0_all = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA GPU required for V10 training!"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    v90.OUT_DIR = OUT_DIR
    v90.DRIVE_DIR = DRIVE_DIR
    v90.OUT_PERM_PT = OUT_DIR / "v10_permutations.pt"
    if DRIVE_DIR is not None:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    bgt = compute_v10_budget_bytes()
    assert bgt["tern5_cnt"] == 0 and int(bgt["total_t16c_bytes"]) <= MAX_V10_T16C_BYTES, f"Bad budget: {bgt}"
    print(f"[V10 Start] GPU={gpu_name} ({gpu_vram_gb:.1f} GiB) | target={bgt['total_t16c_gib']} GiB | zero tern5 | e2e_steps={e2e_steps} | mbs={micro_bs}x{grad_accum}", flush=True)

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

    H256_GPU = torch.from_numpy(pa.make_sylvester_hadamard(V10_BLOCK_SIZE)).to(device, dtype=torch.float32)
    H8_GPU = torch.from_numpy(c8d.H8_NP).to(device, dtype=torch.float32)
    rmsn_gpu, make_rope, apply_rope = v90._make_gemma4_math_helpers(device, EPS)
    rope_cs_g, rope_sn_g = make_rope(THETA_G, int(round(HD_F * PART_G)), HD_F)
    rope_cs_s, rope_sn_s = make_rope(THETA_L, int(round(HD_S * PART_L)), HD_S)

    # Pack Vision + Permuted Embedding
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
    exp_emb_b = p3.expected_tile16_bytes(vocab_sz, D_MOD, V10_BLOCK_SIZE, 3, False)
    emb_pb = np.empty(exp_emb_b, dtype=np.uint8)
    tblk_emb = p3.tile16_blk_bytes(V10_BLOCK_SIZE, 3, False) * (D_MOD // V10_BLOCK_SIZE)
    for r0 in range(0, vocab_sz, 16384):
        r1 = min(r0 + 16384, vocab_sz)
        w_c = emb_perm_teach[r0:r1].to(device, dtype=torch.float32)
        w_rot = (w_c.reshape(-1, V10_BLOCK_SIZE) @ H256_GPU).contiguous()
        pb_c = v90._c_quant_and_pack_t16c(lib_fq, w_rot, V10_BLOCK_SIZE, 6)
        emb_pb[(r0 // 16) * tblk_emb : (r0 // 16) * tblk_emb + pb_c.size] = pb_c
        u_c, sc_c, _ = v90.unpack_tile16_headers_and_units(pb_c, r1 - r0, D_MOD, V10_BLOCK_SIZE, 6, H256_GPU)
        emb_eff[r0:r1] = v90.reconstruct_from_headers(u_c, sc_c, None, None, H256_GPU).to(torch.float16).cpu()
        del w_c, w_rot, pb_c, u_c, sc_c
    torch.cuda.empty_cache()

    X_stud_0 = (emb_eff[torch.from_numpy(all_seq_ids)] * ESCALE).to(torch.float16)
    for si in range(N_TEXT_SEQ, N_SEQ):
        pm = torch.from_numpy(all_seq_ids[si] == soft_tok)
        X_stud_0[si, pm] = X_teach_perm[si, pm]
    X_stud_perm = X_stud_0.clone()

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
    MB_FWD = 32 if gpu_vram_gb >= 20.0 else 16
    perm_pi_g = perm_pi.to(device)

    # Stage 2: Layer-by-layer Hessian + GeGLU permutation + Wiener headers + High-RAM teacher telemetry cache
    e2e_layers: List[dict] = []
    pi_l_list: List[torch.Tensor] = []
    wo_head_norms: List[torch.Tensor] = []
    teach_mod_sal_list: List[torch.Tensor] = []
    start_l = 0
    s2_done_ckpt = CKPT_DIR / "stage2_meta.pt"
    if s2_done_ckpt.is_file():
        meta2 = torch.load(s2_done_ckpt, map_location="cpu", weights_only=False)
        start_l = int(meta2.get("next_layer", 0))
        if start_l > 0:
            X_teach_perm = torch.load(CKPT_DIR / "stage2_xt.pt", map_location="cpu", weights_only=True)
            X_stud_perm = torch.load(CKPT_DIR / "stage2_xs.pt", map_location="cpu", weights_only=True)
            pi_l_list = meta2["pi_l_list"]
            wo_head_norms = meta2["wo_head_norms"]
            teach_mod_sal_list = meta2["teach_mod_sal_list"]
            for li_r in range(start_l):
                e2e_layers.append(torch.load(CKPT_DIR / f"e2e_L{li_r:02d}.pt", map_location="cpu", weights_only=False))
            print(f"[V10 Stage 2 Resume] Resumed from layer {start_l}/{N_LAYERS}", flush=True)

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

            wo_head_norms.append(wo_t.view(D_MOD, NH, hd).pow(2).sum(dim=(0, 2)).sqrt().clamp(min=1e-6).cpu())
            attn_x_rms = torch.zeros((D_MOD,), device=device, dtype=torch.float32)
            mlp_x_rms = torch.zeros((D_MOD,), device=device, dtype=torch.float32)
            down_x_sq_orig = torch.zeros((V10_FF,), device=device, dtype=torch.float32)
            H_attn = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32)
            H_o = torch.zeros((NH * hd, NH * hd), device=device, dtype=torch.float32)
            H_mlp = torch.zeros((D_MOD, D_MOD), device=device, dtype=torch.float32)
            act_samples, x_attn_ev, x_o_ev, x_mlp_ev = [], [], [], []
            X_teach_next = torch.empty_like(X_teach_perm)
            X_teach_attn_out = torch.empty_like(X_teach_perm)
            layer_sal_cpu = torch.empty((N_SEQ * v90.V9_SEQ_LEN,), dtype=torch.float16)

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
                        act_samples.append(act[:, ::2, :].reshape(-1, V10_FF).abs().to(torch.float32).cpu())
                    x_nx = (xm + rmsn_gpu(F.linear(act, wd_t), g_pff)) * lsc_init
                    X_teach_next[b0:b1] = x_nx.to(torch.float16).cpu()
                    sal = (x_nx - xb).pow(2).sum(dim=-1).sqrt() / (xb.pow(2).sum(dim=-1).sqrt() + 1e-6)
                    layer_sal_cpu[b0 * v90.V9_SEQ_LEN : b1 * v90.V9_SEQ_LEN] = sal.reshape(-1).to(torch.float16).cpu()

            teach_mod_sal_list.append(layer_sal_cpu)
            attn_x_rms = torch.sqrt(attn_x_rms / float(N_SEQ * v90.V9_SEQ_LEN)).clamp(min=1e-5)
            mlp_x_rms = torch.sqrt(mlp_x_rms / float(N_SEQ * v90.V9_SEQ_LEN)).clamp(min=1e-5)
            xa_ev_t = torch.cat(x_attn_ev, dim=0)
            xo_ev_t = torch.cat(x_o_ev, dim=0)
            xm_ev_t = torch.cat(x_mlp_ev, dim=0)

            act_cat = torch.cat(act_samples, dim=0).to(device, dtype=torch.float32)
            pi_l, _, pi_stats = v90.compute_geglu_colblock_perm(act_cat, shared_blocks=V10_SHARED_BLOCKS, block_size=V10_BLOCK_SIZE, iters=6, seed=42 + l_idx)
            pi_l_list.append(pi_l.cpu())
            pi_lg = pi_l.to(device)
            wg_t = wg_t[pi_lg, :].contiguous()
            wu_t = wu_t[pi_lg, :].contiguous()
            wd_t = wd_t[:, pi_lg].contiguous()
            down_x_rms = torch.sqrt((down_x_sq_orig[pi_lg] / float(N_SEQ * v90.V9_SEQ_LEN)).clamp(min=1e-10)).clamp(min=1e-5)
            del act_samples, act_cat, down_x_sq_orig

            H_down_blks = torch.zeros((V10_MLP_BLOCKS, V10_BLOCK_SIZE, V10_BLOCK_SIZE), device=device, dtype=torch.float32)
            xd_ev_list: List[torch.Tensor] = []
            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xf_d = rmsn_gpu(X_teach_attn_out[b0:b1].to(device, dtype=torch.float32), g_pf)
                    act_p = F.gelu(F.linear(xf_d, wg_t), approximate="tanh") * F.linear(xf_d, wu_t)
                    ab = act_p.reshape(-1, V10_MLP_BLOCKS, V10_BLOCK_SIZE)
                    H_down_blks.add_(torch.einsum("nbi,nbj->bij", ab, ab))
                    if b0 < 256:
                        xd_ev_list.append(act_p.reshape(-1, V10_FF)[::8].to(torch.float16).cpu())
            xd_ev_t = torch.cat(xd_ev_list, dim=0).to(device) if xd_ev_list else None
            del xd_ev_list

            def _quant_pack_proj(
                w_f32: torch.Tensor, xr: torch.Tensor, pname: str,
                h_full: Optional[torch.Tensor] = None, h_blks_in: Optional[torch.Tensor] = None,
                x_ev: Optional[torch.Tensor] = None, val_x: Optional[torch.Tensor] = None,
            ) -> Tuple[dict, torch.Tensor]:
                mode_i, bsz_i, lrr, is_rm, is_cb = get_v10_proj_spec(l_idx, pname, is_f)
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
                    "planes_u8": v93.extract_planes_u8(pb, do, di, bsz_i, mode_i).cpu(),
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

                act_tr, dy_pff_tr, pred_tr, act_va, dy_pff_va, pred_va, z_pff_tr, z_pff_va = [], [], [], [], [], [], [], []
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
            print(f"  [V10 Stage2 L{l_idx:02d}/59] cos={cos_l:.5f} e_10of84={pi_stats['perm_12of84_energy_pct']}% dt={time.perf_counter()-tl0:.1f}s", flush=True)

            del wq_t, wk_t, wv_t, wo_t, wg_t, wu_t, wd_t, wq_s, wk_s, wv_s, wo_s, wg_s, wu_s, wd_s, X_teach_attn_out, xm_stud_cpu
            gc.collect()
            torch.cuda.empty_cache()

            if (l_idx + 1) % 10 == 0 or (l_idx + 1) == N_LAYERS:
                torch.save(X_teach_perm.cpu(), CKPT_DIR / "stage2_xt.pt")
                torch.save(X_stud_perm.cpu(), CKPT_DIR / "stage2_xs.pt")
                torch.save(
                    {"next_layer": l_idx + 1, "pi_l_list": pi_l_list, "wo_head_norms": wo_head_norms, "teach_mod_sal_list": teach_mod_sal_list},
                    s2_done_ckpt,
                )

    # Precompute Teacher Top-128 Softmax Targets
    fn_teach_g = fn_orig[perm_pi].to(device=device, dtype=torch.float32)
    emb_teach_g = emb_perm_teach.to(device=device, dtype=torch.bfloat16)
    top_ids_cpu = torch.empty((N_SEQ, v90.V9_SEQ_LEN, 128), dtype=torch.int32)
    top_probs_cpu = torch.empty((N_SEQ, v90.V9_SEQ_LEN, 128), dtype=torch.float16)
    with torch.no_grad():
        for b0 in range(0, N_SEQ, 32):
            b1 = min(b0 + 32, N_SEQ)
            ht = rmsn_gpu(X_teach_perm[b0:b1].to(device=device, dtype=torch.float32), fn_teach_g).to(torch.bfloat16)
            lgt = SOFTCAP * torch.tanh((ht @ emb_teach_g.T).float() / SOFTCAP)
            pt = F.softmax(lgt, dim=-1)
            tp, ti = torch.topk(pt, 128, dim=-1)
            top_ids_cpu[b0:b1] = ti.to(torch.int32).cpu()
            top_probs_cpu[b0:b1] = tp.to(torch.float16).cpu()
    del emb_teach_g, emb_perm_teach, fn_teach_g
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2.5: Dual-Regime E2E 60-Layer Logit-KL QAT + Early-Exit L29 Probe
    fn_param = nn.Parameter(fn_orig[perm_pi].to(device=device, dtype=torch.float32))
    early_fn_param = nn.Parameter(fn_orig[perm_pi].to(device=device, dtype=torch.float32))
    ctx_e2e = dict(st)
    e2e_metrics = run_stage2_5_v10_dual_e2e_kl(
        e2e_layers=e2e_layers,
        X_stud_0=X_stud_0,
        X_teach_l59=X_teach_perm,
        top_ids_cpu=top_ids_cpu,
        top_probs_cpu=top_probs_cpu,
        ans_mask_np=all_ans_mask,
        emb_eff=emb_eff,
        fn_param=fn_param,
        early_fn_param=early_fn_param,
        tr_seq_ids=tr_seq_ids,
        va_seq_ids=va_seq_ids,
        ctx=ctx_e2e,
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

    # Assemble final V10 T16C + replay Stage-2.5 tuned weights once using cached Stage-2 teacher signals (fast!)
    all_norms: List[Tuple[str, np.ndarray]] = [
        ("model.language_model.norm.weight", fn_param.detach().cpu().numpy().astype(np.float32)),
        ("model.language_model.early_exit_l29_norm.weight", early_fn_param.detach().cpu().numpy().astype(np.float32)),
    ]
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
    crtr_layers, head_crtr_layers, layer_logs = [], [], []
    X_replay = X_stud_0.clone()

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
        pos = p3._write_mat_record(fp, pos, "model.language_model.embed_tokens.weight", vocab_sz, D_MOD, V10_BLOCK_SIZE, 3 | p3.T16C_GAUSS_BIT, False, True, emb_pb, None, None)
        del emb_pb

        for l_idx, lyr in enumerate(e2e_layers):
            pfx = f"model.language_model.layers.{l_idx}"
            is_f = LAYER_TYPES[l_idx] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)
            rep = NH // nkv
            wo_head_norm = wo_head_norms[l_idx].to(device)
            mod_sal_cpu = teach_mod_sal_list[l_idx].float()

            w_stud: Dict[str, torch.Tensor] = {}
            for pn, prec in lyr["projs"].items():
                do, di, bsz_p, mode_p = int(prec["do"]), int(prec["di"]), int(prec["bsize"]), int(prec["mode_i"])
                nb_m, asym_m = p3.MODE_TO_T16C[mode_p]
                n_tiles, n_blks, tblk_b = (do + 15) // 16, di // bsz_p, p3.tile16_blk_bytes(bsz_p, nb_m, asym_m)
                hdr_b = 64 if asym_m else 32
                arr_pb = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
                arr_pb[:, :, hdr_b:] = prec["planes_u8"].cpu().numpy().reshape(n_tiles, n_blks, tblk_b - hdr_b)
                pb_flat = arr_pb.reshape(-1)
                v90._c_update_t16c_headers(lib_fq, pb_flat, do, di, bsz_p, mode_p, prec["sc"], prec["bi"])
                lr_t = v93.quant_lr_int8_tuple(prec["u_lr"], prec["v_lr"]) if prec["u_lr"] is not None else None
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

            g_in_s = lyr["g_in"].to(device, dtype=torch.float32)
            g_pa_s = lyr["g_pa"].to(device, dtype=torch.float32)
            g_pf_s = lyr["g_pf"].to(device, dtype=torch.float32)
            g_pff_s = lyr["g_pff"].to(device, dtype=torch.float32)
            g_qn_s = lyr["g_qn"].to(device, dtype=torch.float32)
            g_kn_s = lyr["g_kn"].to(device, dtype=torch.float32)
            lsc_s = float(lyr["layer_scalar"].item())

            xn_cpu = torch.empty_like(X_replay)
            xf_cpu = torch.empty_like(X_replay)
            head_e_cpu = torch.empty((N_SEQ * v90.V9_SEQ_LEN, NH), dtype=torch.float32)
            blk_e_cpu = torch.empty((N_SEQ * v90.V9_SEQ_LEN, V10_MLP_BLOCKS), dtype=torch.float32)

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
                    xf_cpu[b0:b1] = xf.to(torch.float16).cpu()
                    act_q = F.gelu(F.linear(xf, wg_s), approximate="tanh") * F.linear(xf, wu_s)
                    blk_e_cpu[b0 * v90.V9_SEQ_LEN : b1 * v90.V9_SEQ_LEN] = act_q.reshape(-1, V10_MLP_BLOCKS, V10_BLOCK_SIZE).pow(2).sum(dim=-1).cpu()
                    X_replay[b0:b1] = ((xm + rmsn_gpu(F.linear(act_q, wd_s), g_pff_s)) * lsc_s).to(torch.float16).cpu()

            xn_f = xn_cpu.reshape(-1, D_MOD)[::2]
            he_f = head_e_cpu[::2]
            ntr_h = (xn_f.shape[0] * 9) // 10
            hr_net, hr_met = crtr.train_layer_router(xn_f[:ntr_h], he_f[:ntr_h], xn_f[ntr_h:], he_f[ntr_h:], top_p=float(V10_HEAD_ROUTED) / float(NH), rank=v90.V9_HEAD_CRTR_RANK, steps=120, lr=3e-3, device=str(device), seed=300 + l_idx)
            head_crtr_layers.append(crtr.quantize_layer_for_crtr(hr_net, V10_BLOCK_SIZE))

            xf_f = xf_cpu.reshape(-1, D_MOD)[::2]
            be_f = blk_e_cpu[::2]
            my_f = (mod_sal_cpu[::2] > float(torch.quantile(mod_sal_cpu, 0.35).item())).float()
            ntr_r = (xf_f.shape[0] * 9) // 10
            r_net, r_met = crtr.train_layer_router(xf_f[:ntr_r], be_f[:ntr_r], xf_f[ntr_r:], be_f[ntr_r:], top_p=float(V10_CRTR_TOP_BLOCKS) / float(V10_MLP_BLOCKS), rank=v90.V9_CRTR_RANK, steps=160, lr=3e-3, mod_targets_tr=my_f[:ntr_r], mod_targets_va=my_f[ntr_r:], device=str(device), seed=100 + l_idx)
            crtr_layers.append(crtr.quantize_layer_for_crtr(r_net, V10_BLOCK_SIZE))

            layer_logs.append({"l": l_idx, "crtr_recall": r_met["val_recall_after"], "head_crtr_recall": hr_met["val_recall_after"]})
            if (l_idx + 1) % 10 == 0 or l_idx in (0, V10_EARLY_LAYER, 59):
                print(f"  [V10 Replay L{l_idx:02d}/59] crtr12_rec={r_met['val_recall_after']:.3f} hr8_rec={hr_met['val_recall_after']:.3f}", flush=True)
            del w_stud, xn_cpu, xf_cpu, head_e_cpu, blk_e_cpu, mod_sal_cpu

        for rec_v in vision_mat_recs:
            pos = p3._write_mat_record(fp, pos, *rec_v)

    os.replace(tmp_t16c, OUT_T16C)
    vm_c, ve_c = v70.verify_t16c_integrity(OUT_T16C)
    assert vm_c == total_mats and ve_c == OUT_T16C.stat().st_size, f"Bad V10 T16C {vm_c}/{total_mats}"
    cb_cnt = sdc.verify_t16c_file(OUT_T16C)["colblock_mats"]
    l59_cos_final = float(F.cosine_similarity(X_replay[-32:].reshape(-1, D_MOD).float(), X_teach_perm[-32:].reshape(-1, D_MOD).float(), dim=-1).mean().item())

    # Train Target-Embed-Shared 3-step Self-Feeding EAGLE-3 Assistant (and post_cell60 C8DB + BHDR-512)
    crtr.export_crtr(crtr_layers, OUT_CRTR)
    crtr.export_crtr(head_crtr_layers, OUT_HEAD_CRTR)
    fn_eff_g = fn_param.detach().to(device=device, dtype=torch.float32)

    # Train post_cell60 C8DB first on X_replay -> X_teach_perm
    c8db_entries: List[dict] = []
    emb_g = emb_eff.to(device=device, dtype=compute_dtype)
    ans_flat = torch.nonzero(torch.from_numpy(all_ans_mask.reshape(-1)), as_tuple=False).squeeze(1)
    sel_ans = ans_flat[torch.linspace(0, ans_flat.numel() - 1, min(16384, ans_flat.numel())).long()]
    s_t, t_t = sel_ans // v90.V9_SEQ_LEN, sel_ans % v90.V9_SEQ_LEN
    hs_ans = rmsn_gpu(X_replay[s_t, t_t].to(device).unsqueeze(1), fn_eff_g).squeeze(1).float()
    ht_ans = rmsn_gpu(X_teach_perm[s_t, t_t].to(device).unsqueeze(1), fn_eff_g).squeeze(1).float()
    nxt_ans = torch.from_numpy(all_seq_nxt[s_t.numpy(), t_t.numpy()]).to(device, dtype=torch.long)
    iv_fn = torch.arange(hs_ans.shape[0], device=device) % 10 == 0
    tr_idx60 = torch.nonzero(~iv_fn, as_tuple=False).squeeze(1)
    va_idx60 = torch.nonzero(iv_fn, as_tuple=False).squeeze(1)
    with torch.no_grad():
        top_ids_c = [torch.topk((ht_ans[i0:min(i0 + 512, ht_ans.shape[0])] @ emb_g.float().T), 16, dim=-1).indices.reshape(-1) for i0 in range(0, min(4096, ht_ans.shape[0]), 512)]
        uniq_c, cnt_c = torch.unique(torch.cat(top_ids_c), return_counts=True)
        cand_c60 = torch.unique(torch.cat([uniq_c[torch.topk(cnt_c, min(4096, uniq_c.numel())).indices], nxt_ans]))
        W_c60 = emb_g[cand_c60].float()
        id_map60 = {int(t): ci for ci, t in enumerate(cand_c60.tolist())}
        y_star60 = torch.tensor([id_map60[int(t)] for t in nxt_ans.tolist()], device=device, dtype=torch.long)
    m60 = c8d.Cuneiform8DPatternAdapter(d_out=D_MOD, d_in=D_MOD, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0).to(device)
    o60 = torch.optim.AdamW(m60.parameters(), lr=3e-3, weight_decay=2e-2)
    s60 = torch.optim.lr_scheduler.CosineAnnealingLR(o60, T_max=200, eta_min=3e-4)
    best_kl60, best_sd60 = 1e9, None
    for st60 in range(200):
        o60.zero_grad()
        pb60 = tr_idx60[torch.randint(0, tr_idx60.numel(), (min(1024, tr_idx60.numel()),), device=device)]
        da60, _, lb60 = m60(hs_ans[pb60])
        hp60 = hs_ans[pb60] + da60
        with torch.no_grad():
            pt_b = F.softmax(SOFTCAP * torch.tanh((ht_ans[pb60] @ W_c60.T) / SOFTCAP), dim=-1)
        ls_b = SOFTCAP * torch.tanh((hp60 @ W_c60.T) / SOFTCAP)
        loss60 = (pt_b * (torch.log(pt_b.clamp(min=1e-12)) - F.log_softmax(ls_b, dim=-1))).sum(dim=-1).mean() + 2.0 * F.mse_loss(hp60, ht_ans[pb60]) + 0.5 * (1.0 - F.cosine_similarity(hp60, ht_ans[pb60], dim=-1).mean()) + 0.005 * lb60
        loss60.backward()
        torch.nn.utils.clip_grad_norm_(m60.parameters(), 1.0)
        o60.step(); s60.step()
        if (st60 + 1) % 50 == 0:
            with torch.no_grad():
                lp_v = F.log_softmax(SOFTCAP * torch.tanh(((hs_ans[va_idx60] + m60(hs_ans[va_idx60])[0]) @ W_c60.T) / SOFTCAP), dim=-1)
                pt_v = F.softmax(SOFTCAP * torch.tanh((ht_ans[va_idx60] @ W_c60.T) / SOFTCAP), dim=-1)
                klv = float((pt_v * (torch.log(pt_v.clamp(min=1e-12)) - lp_v)).sum(dim=-1).mean().item())
                if klv < best_kl60:
                    best_kl60, best_sd60 = klv, {k: v.detach().cpu().clone() for k, v in m60.state_dict().items()}
    mc60 = m60.cpu()
    if best_sd60 is not None:
        mc60.load_state_dict(best_sd60)
    cents60, experts60 = mc60.export_quantized_arrays()
    with torch.no_grad():
        lp_0 = F.log_softmax(SOFTCAP * torch.tanh((hs_ans[va_idx60] @ W_c60.T) / SOFTCAP), dim=-1)
        lp_q = F.log_softmax(SOFTCAP * torch.tanh(((hs_ans[va_idx60] + v90._eval_c8d_gpu(hs_ans[va_idx60], cents60, experts60, H8_GPU)) @ W_c60.T) / SOFTCAP), dim=-1)
        pt_v = F.softmax(SOFTCAP * torch.tanh((ht_ans[va_idx60] @ W_c60.T) / SOFTCAP), dim=-1)
        kl60_0 = float((pt_v * (torch.log(pt_v.clamp(min=1e-12)) - lp_0)).sum(dim=-1).mean().item())
        kl60_q = float((pt_v * (torch.log(pt_v.clamp(min=1e-12)) - lp_q)).sum(dim=-1).mean().item())
    keep_c60 = kl60_q < kl60_0
    if keep_c60:
        exp_r60 = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
        c8db_entries.append({"name": "model.language_model.norm.post_cell60", "d_out": D_MOD, "d_in": D_MOD, "rank": 32, "top_k": 2, "c6d": exp_r60[0], "temperature": 0.5, "gate_scale": 1.0, "centroids": cents60, "expert_radicals": exp_r60, "experts": experts60})
    c8d.write_c8db_file(OUT_C8D, [], c8db_entries)
    del m60, mc60, o60, s60, hs_ans, ht_ans, W_c60, cand_c60, y_star60
    torch.cuda.empty_cache()

    # Stage 3B: Pretrained Assistant alignment optimizing Target-Embed Sharing (rmsn(post(ha), fn) @ emb_eff.T)
    hf_tok = os.environ.get("HF_TOKEN", "").strip() or None
    ast_dir = Path(snapshot_download("google/gemma-4-31B-it-assistant", local_dir="/tmp/gemma4_ast", token=hf_tok, ignore_patterns=["*.bin", "*.pt"]))
    ast_sd_raw: Dict[str, torch.Tensor] = {}
    with safe_open(ast_dir / "model.safetensors", framework="pt", device="cpu") as sf_a:
        for k in sf_a.keys():
            ast_sd_raw[k] = sf_a.get_tensor(k)
    ast_sd = permute_assistant_state_dict(ast_sd_raw, perm_pi, D_MOD)
    del ast_sd_raw

    ast_mod = v90.Gemma4AssistantTrain(
        ast_sd, d_mod=D_MOD, nh=NH, nkv_s=NKV_S, nkv_f=NKV_F, hd_s=HD_S, hd_f=HD_F,
        theta_l=THETA_L, part_l=PART_L, theta_g=THETA_G, part_g=PART_G, seq_len=v90.V9_SEQ_LEN, eps=EPS, compute_dtype=compute_dtype,
    ).to(device)
    with torch.no_grad():
        ast_h_final = torch.empty((len(ast_seqs), v90.V9_SEQ_LEN, D_MOD), dtype=torch.float16)
        for ai, gi in enumerate(ast_seqs):
            hf_ai = rmsn_gpu(X_replay[gi].to(device), fn_eff_g).float()
            if keep_c60:
                hf_ai = hf_ai + v90._eval_c8d_gpu(hf_ai, cents60, experts60, H8_GPU)
            ast_h_final[ai] = hf_ai.to(torch.float16).cpu()
        nxt_ast = torch.from_numpy(all_seq_nxt[ast_seqs].reshape(-1)).long()
        uniq_a, cnts_a = torch.unique(nxt_ast, return_counts=True)
        cand_ids = uniq_a[torch.topk(cnts_a, min(16384, uniq_a.numel())).indices].to(device)
        W_cand_g = emb_g[cand_ids].float()

    n_ast_tr = max(1, (len(ast_seqs) * 7) // 8)
    TAU_AST = 1.5

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
                hpost_n = rmsn_gpu(F.linear(ha.to(compute_dtype), ast_mod.post.to(compute_dtype)).float(), fn_eff_g)
                zs = SOFTCAP * torch.tanh((ast_h_final[ai, 1:].to(device).float() @ W_cand_g.T) / SOFTCAP)
                za = SOFTCAP * torch.tanh((hpost_n @ W_cand_g.T) / SOFTCAP)
                ps = F.softmax(zs / TAU_AST, dim=-1)
                kls += float(((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU_AST, dim=-1))).sum(dim=-1) * (TAU_AST * TAU_AST)).sum().item())
                agrs += float((za.argmax(dim=-1) == zs.argmax(dim=-1)).float().sum().item())
                cnt += ha.shape[0]
            return kls / max(cnt, 1), agrs / max(cnt, 1)

    akl0, aagr0 = eval_ast()
    opt_a = torch.optim.AdamW(ast_mod.parameters(), lr=2.5e-4, weight_decay=1e-4)
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=360, eta_min=2e-5)
    best_aagr, best_asd = aagr0, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_bs = 8 if gpu_vram_gb >= 20.0 else 4
    for st_a in range(360):
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

        with torch.no_grad():
            zs = SOFTCAP * torch.tanh((hst.reshape(-1, D_MOD) @ W_cand_g.T) / SOFTCAP)
            ps, ys_t1 = F.softmax(zs / TAU_AST, dim=-1), zs.argmax(dim=-1)

        # 3-step self-feeding EAGLE-3 rollout directly evaluated via Target-Embed Sharing!
        ha1 = ast_mod.forward_seq(te_s, bh_s, k58_s, v58_s, k59_s, v59_s)
        hp1 = rmsn_gpu(F.linear(ha1.to(compute_dtype), ast_mod.post.to(compute_dtype)).float(), fn_eff_g)
        za1 = SOFTCAP * torch.tanh((hp1.reshape(-1, D_MOD) @ W_cand_g.T) / SOFTCAP)

        bh2 = torch.cat([bh_s[:, :1], hp1[:, :-1]], dim=1)
        ha2 = ast_mod.forward_seq(te_s, bh2, k58_s, v58_s, k59_s, v59_s)
        hp2 = rmsn_gpu(F.linear(ha2.to(compute_dtype), ast_mod.post.to(compute_dtype)).float(), fn_eff_g)

        bh3 = torch.cat([bh_s[:, :1], hp2[:, :-1]], dim=1)
        ha3 = ast_mod.forward_seq(te_s, bh3, k58_s, v58_s, k59_s, v59_s)
        hp3 = rmsn_gpu(F.linear(ha3.to(compute_dtype), ast_mod.post.to(compute_dtype)).float(), fn_eff_g)

        loss_a = (
            ((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za1 / TAU_AST, dim=-1))).sum(dim=-1) * (TAU_AST * TAU_AST)).mean()
            + 0.40 * F.cross_entropy(za1, ys_t1)
            + 0.40 * (F.mse_loss(hp1, hst) / (hst.var() + 1e-5))
            + 0.30 * (1.0 - F.cosine_similarity(hp2[:, 1:], hst[:, 1:], dim=-1).mean())
            + 0.25 * (1.0 - F.cosine_similarity(hp3[:, 2:], hst[:, 2:], dim=-1).mean())
        )
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(ast_mod.parameters(), 1.0)
        opt_a.step()
        sch_a.step()
        if (st_a + 1) % 60 == 0:
            klv, agrv = eval_ast()
            print(f"    [V10 Ast step {st_a+1:03d}/360] Val KL={klv:.4f} TargetSharedTop1={agrv*100:.1f}%", flush=True)
            if agrv > best_aagr:
                best_aagr, best_asd = agrv, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}

    ast_mod.load_state_dict({k: v.to(device) for k, v in best_asd.items()})
    akl1, aagr1 = eval_ast()
    print(f"  [V10 Target-Shared Assistant] Val KL={akl0:.4f}->{akl1:.4f} | Top1Agree={aagr0*100:.1f}%->{aagr1*100:.1f}%", flush=True)

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
    ast_sf = OUT_DIR / "assistant_v10_permuted.safetensors"
    st_save_file(ast_sd, str(ast_sf))
    bhmod.pack_assistant_body_from_safetensors(ast_sf, OUT_AST_Q2, body_bits=2, block_size=64, has_fwht=False)
    bhmod.pack_assistant_body_from_safetensors(ast_sf, OUT_AST_Q4, body_bits=4, block_size=64, has_fwht=True)
    ast_sf.unlink(missing_ok=True)
    del opt_a, sch_a, ast_sd, best_asd, W_cand_g

    # Train upgraded BHDR-512 sidecar for optional fast breadth scoring
    trie = bhmod.BpePrefixTrie.from_tokenizer_json(tok_path)
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
    with torch.no_grad():
        for i0 in range(0, H_stud_tgt.shape[0], 512):
            i1 = min(i0 + 512, H_stud_tgt.shape[0])
            lg_b = SOFTCAP * torch.tanh((H_stud_tgt[i0:i1].to(device, dtype=compute_dtype) @ emb_g.T).float() / SOFTCAP)
            tv, ti = torch.topk(lg_b, k=64, dim=-1)
            top_probs_t[i0:i1] = F.softmax(tv, dim=-1).cpu()
            top_ids_t[i0:i1] = ti.to(torch.int32).cpu()
    del emb_g, H_stud_tgt, ast_h_final
    student_argmax_ids = torch.tensor(v70.select_first_valid_trie_ids(top_ids_t, trie), dtype=torch.long)
    act_bundle = OUT_DIR / "v10_bhdr_acts.pt"
    torch.save({"h_final": H_ast_all.half(), "gold_ids": student_argmax_ids, "top_ids": top_ids_t, "top_probs": top_probs_t}, act_bundle)
    bh_rpt = bhmod.train_from_activations(act_bundle, tok_path, OUT_BHDR, d_byte=512, max_bytes=48, steps=2600, batch_size=256, lr=3e-3, device=str(device))
    act_bundle.unlink(missing_ok=True)
    bh_loaded = bhmod.load_bhdr_v1(OUT_BHDR)
    beam_hits = sum(1 for hv, gv in zip(H_ast_all[-256:].numpy().astype(np.float32), student_argmax_ids[-256:].tolist()) if bhmod.decode_bhdr_np_beam(bh_loaded, trie, hv, beam_width=2)[0] == gv)
    bh_metrics = {**bh_rpt["metrics"], "heldout_beam_top1_acc": round(beam_hits / 256.0, 4)}
    del H_ast_all, student_argmax_ids, top_ids_t, top_probs_t

    all_artifact_paths = (OUT_T16C, OUT_CRTR, OUT_HEAD_CRTR, OUT_C8D, OUT_AST_Q2, OUT_AST_Q4, OUT_BHDR)
    artifacts = {}
    for fp_out in all_artifact_paths:
        sz_b = fp_out.stat().st_size
        hx_b = v90._sha256_file(fp_out)
        if DRIVE_DIR is not None:
            v90._drive_safe_copy(fp_out, DRIVE_DIR / fp_out.name)
        artifacts[fp_out.name] = {"bytes": sz_b, "sha256": hx_b}

    rpt = {
        "version": "V10-CORTICAL-CASCADE-DUAL-E2E-KL",
        "gpu": gpu_name,
        "elapsed_s": round(time.perf_counter() - t0_all, 2),
        "t16c_bytes": OUT_T16C.stat().st_size,
        "t16c_gib": round(OUT_T16C.stat().st_size / (1024**3), 4),
        "budget": bgt,
        "colblock_mats": cb_cnt,
        "l59_cos": round(l59_cos_final, 5),
        "e2e_stage2_5": e2e_metrics,
        "crtr_mean_recall": round(float(np.mean([r["crtr_recall"] for r in layer_logs])), 5),
        "head_crtr_mean_recall": round(float(np.mean([r["head_crtr_recall"] for r in layer_logs])), 5),
        "assistant_target_shared": {"kl_before": round(akl0, 4), "kl_after": round(akl1, 4), "top1_before": round(aagr0, 4), "top1_after": round(aagr1, 4)},
        "bhdr512": bh_metrics,
        "artifacts": artifacts,
    }
    OUT_REPORT.write_text(json.dumps(rpt, indent=2), encoding="utf-8")
    sums_path = OUT_DIR / "SHA256SUMS_V10.txt"
    sums_path.write_text("".join(f"{v['sha256']}  {k}\n" for k, v in artifacts.items()), encoding="utf-8")
    if DRIVE_DIR is not None:
        v90._drive_safe_copy(OUT_REPORT, DRIVE_DIR / OUT_REPORT.name)
        v90._drive_safe_copy(sums_path, DRIVE_DIR / sums_path.name)
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
        expected_sha = {k: v["sha256"] for k, v in artifacts.items()}
        expected_sha[OUT_REPORT.name] = v90._sha256_file(OUT_REPORT)
        expected_sha[sums_path.name] = v90._sha256_file(sums_path)
        v90._verify_drive_sha256(DRIVE_DIR, expected_sha)

    print(f"[V10 COMPLETE] elapsed={(time.perf_counter()-t0_all)/60:.1f} min | {OUT_T16C.name}={rpt['t16c_gib']} GiB | ast_top1={aagr1*100:.1f}%", flush=True)
    if os.environ.get("COLAB_AUTO_UNASSIGN", "1") == "1":
        v90._flush_drive_before_unassign()
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:
            print(f"[Auto-Unassign Note] {exc}", flush=True)
    return rpt


def main() -> int:
    ap = argparse.ArgumentParser(description="Language-U Gemma 4 31B V10 Cortical Cascade GPU Pipeline")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--e2e-steps", type=int, default=200)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    args = ap.parse_args()
    if args.selftest or (not args.run and not torch.cuda.is_available()):
        return run_selftest()
    run_v10_pipeline(e2e_steps=args.e2e_steps, micro_bs=args.micro_bs, grad_accum=args.grad_accum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

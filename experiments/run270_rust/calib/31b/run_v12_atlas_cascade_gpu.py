#!/usr/bin/env python3
"""Language-U V12: Atlas-Guided Cortical Cascade (<= 6.27 GiB T16C + Core-Protected E2E KL QAT + EAGLE-3 Q4/Q2).

Re-tiers Gemma 4 31B's 1-bit / 2-bit byte budget directly from the empirical 60-Layer Pi 4 `.t16c` +
1,000-Pair (`2,000`-prompt, 10-domain) A100 `bfloat16` Cognitive-Core-Atlas measurements:
  1. Atlas-Guided 1-Bit / 2-Bit Precision Swap (`~6.262 GiB <= 6.27 GiB`, strictly smaller than V10):
     - Keeps 100% of non-boundary Attention (`L0..57`) in fast `2b_gauss COLBLOCK` (`0xA2`) and `L58..59` in `3b`.
     - Upgrades the primary mid-stack & world-model Cognitive Core MLPs (`L26, 27, 28, 29, 32, 41, 46`) from
       `1b_rm` -> `2b_gauss COLBLOCK` (`0xA2`), joining `L0, 1, 48, 49, 54, 55` (`13` full `2b_gauss` MLPs).
     - Assigns `2b_gauss COLBLOCK` `down_proj` to the remaining 5 measured Cognitive Core layers (`L20, 24, 33, 36, 43`).
     - Demotes the 6 empirically verified non-core upper layers (`L50, 51, 52, 53, 56, 57`, all `<= 0` causal ΔNLL
       on both A100 BF16 and Pi 4 `.t16c`) + 4 null `down_proj` layers (`L40, 44, 45, 47`) to fast `1b_rm COLBLOCK`.
  2. Atlas-Aware Dual-Regime 60-Layer E2E Logit-KL + Null-Bypass QAT (`Stage 2.5`):
     - Alternates Dense (`60/60`) and `--cca-plan`-matched Tier-2 Sparse micro-batches where:
       * All 16 Cognitive Core layers run 100% dense (`mlp_keep=1.0, head_keep=1.0`).
       * The 4 dual-verified null layers (`L16, 18, 40, 45`) execute exact Tier-2 bypass (`h * layer_scalar`),
         training downstream layers via real PyTorch gradients to absorb the 4-layer skip seamlessly.
       * Remaining 40 body layers route `12/84` MLP blocks + `8/32` Attention heads.
     - Trains `early_exit_l29_norm` directly atop the newly upgraded `2b_gauss` `L26..29` Cognitive Core hub
       so `VA Mirror` neighborhood page lookups receive high-SNR semantic coordinates.
  3. Target-Embed-Shared 3-Step Self-Feeding EAGLE-3 Assistant (`Q4` & `Q2`) + `BHDR-256`.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
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
import run_v10_cortical_cascade_gpu as v10  # noqa: E402
import spatial_district_converter as sdc  # noqa: E402
import train_assistant_v7_byte_head as bhmod  # noqa: E402
import train_c8d_pattern_bank as c8d  # noqa: E402
import train_cortical_router as crtr  # noqa: E402

V12_BLOCK_SIZE = v10.V10_BLOCK_SIZE
V12_D_MOD = v10.V10_D_MOD
V12_FF = v10.V10_FF
V12_N_LAYERS = v10.V10_N_LAYERS
V12_MLP_BLOCKS = v10.V10_MLP_BLOCKS
V12_SHARED_BLOCKS = v10.V10_SHARED_BLOCKS
V12_CRTR_TOP_BLOCKS = 12
V12_HEAD_ROUTED = 8
V12_EARLY_LAYER = 29
GLOBAL_ATTN_LAYERS = set(v10.GLOBAL_ATTN_LAYERS)

V12_PIPELINE_REVISION = "v12.1_atlas_core_swap_cca_bypass_bhdr256"

# Snapshot immutable V10 baseline budget BEFORE `_bind_v12_hooks()` overrides `v10.get_v10_proj_spec`
_ORIG_V10_BUDGET: Dict[str, object] = v10.compute_v10_budget_bytes()

# Empirical 16-layer Cognitive Core union from 1,000-pair A100 BF16 + 60L Pi 4 .t16c Atlas (`84bab1b`)
V12_COGNITIVE_CORE_LAYERS = {20, 24, 26, 27, 28, 29, 32, 33, 36, 41, 43, 46, 48, 49, 55, 59}
# Dual-verified null-damage Tier-2 bypass layers (`cca_pi4_cluster_plan.json` -> `tier2_bypass_layers`)
V12_TIER2_BYPASS_LAYERS = {16, 18, 40, 45}

# Primary Core MLPs in full `2b_gauss COLBLOCK` (`0xA2`): entry L0..1 + core hubs L26..29, 32, 41, 46, 48, 49, 54, 55
V12_2B_MLP_LAYERS = {0, 1, 26, 27, 28, 29, 32, 41, 46, 48, 49, 54, 55}
# Remaining 5 Cognitive Core layers receive `2b_gauss COLBLOCK` `down_proj` output projection
V12_2B_DOWN_LAYERS = {20, 24, 33, 36, 43}
MAX_V12_T16C_BYTES = int(6.28 * (1024**3))

OUT_DIR = Path(os.environ.get("V12_OUT_DIR", "/content/v12_outputs"))
CKPT_DIR = OUT_DIR / "ckpt_v12"
_drv_env = os.environ.get("V12_DRIVE_DIR", "").strip()
DRIVE_DIR: Optional[Path] = Path(_drv_env) if _drv_env else None

OUT_T16C = OUT_DIR / "gemma4_31b_v12_atlas_cascade.t16c"
OUT_CRTR = OUT_DIR / "gemma4_31b_v12_router.crtr"
OUT_HEAD_CRTR = OUT_DIR / "gemma4_31b_v12_head_router.crtr"
OUT_C8D = OUT_DIR / "gemma4_31b_v12_c8d.c8d"
OUT_AST_Q2 = OUT_DIR / "gemma4_31b_v12_assistant_body_q2.t16c"
OUT_AST_Q4 = OUT_DIR / "gemma4_31b_v12_assistant_body_q4.t16c"
OUT_BHDR = OUT_DIR / "gemma4_31b_v12_assistant.bhdr"
OUT_REPORT = OUT_DIR / "v12_report.json"


def get_v12_proj_spec(l_idx: int, pname: str, is_full: Optional[bool] = None) -> Tuple[int, int, int, bool, bool]:
    """Return `(mode_i, block_size, lr_rank, is_rowmajor, is_colblock)` for V12 Atlas-Guided Cascade."""
    if l_idx >= 58:
        lr_rk = 24 if ("k_proj" in pname or "v_proj" in pname) else 16
        return (4, V12_BLOCK_SIZE, lr_rk, False, ("o_proj" in pname or "down_proj" in pname))
    if pname.startswith("self_attn."):
        return (5, V12_BLOCK_SIZE, 0, False, True)
    if l_idx in V12_2B_MLP_LAYERS:
        return (5, V12_BLOCK_SIZE, 0, False, True)
    if pname == "mlp.down_proj" and l_idx in V12_2B_DOWN_LAYERS:
        return (5, V12_BLOCK_SIZE, 0, False, True)
    lr_rk = 8 if pname == "mlp.down_proj" else 0
    return (0, V12_BLOCK_SIZE, lr_rk, True, True)


def compute_v12_budget_bytes(
    *,
    vocab_size: int = 262144,
    d_mod: int = V12_D_MOD,
    n_layers: int = V12_N_LAYERS,
    nh: int = 32,
    nkv_s: int = 16,
    nkv_f: int = 4,
    hd_s: int = 256,
    hd_f: int = 512,
    ff: int = V12_FF,
    vision_bytes: int = 237_253_152,
) -> Dict[str, object]:
    """Exact fail-closed byte census for V12 `.t16c` (Zero Tern5, <= 6.28 GiB, all 16 cores >= 2b down_proj)."""
    embed_b = p3.expected_tile16_bytes(vocab_size, d_mod, V12_BLOCK_SIZE, 3, False) + 256
    attn_b = mlp_b = lr_b = 0
    tern5_cnt = attn_1b_cnt = mlp_2b_layers = mlp_1b_layers = 0
    unprotected_cores: List[int] = []

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
            mode, bsz, lrr, _, _ = get_v12_proj_spec(l_idx, pn, is_f)
            nb, asym = p3.MODE_TO_T16C[mode]
            if mode == 1:
                tern5_cnt += 1
            if (nb & 0x1F) == 1:
                attn_1b_cnt += 1
            attn_b += p3.expected_tile16_bytes(do, di, bsz, nb, asym) + 128 + di * 4
            if lrr > 0:
                lr_b += do * 4 + lrr * 4 + do * lrr + lrr * di

        m_gate, _, _, _, _ = get_v12_proj_spec(l_idx, "mlp.gate_proj", is_f)
        m_down, _, _, _, _ = get_v12_proj_spec(l_idx, "mlp.down_proj", is_f)
        if m_gate == 5:
            mlp_2b_layers += 1
        elif m_gate == 0:
            mlp_1b_layers += 1
        if l_idx in V12_COGNITIVE_CORE_LAYERS and (p3.MODE_TO_T16C[m_down][0] & 0x1F) < 2:
            unprotected_cores.append(l_idx)

        for pn, do, di in (
            ("mlp.gate_proj", ff, d_mod),
            ("mlp.up_proj", ff, d_mod),
            ("mlp.down_proj", d_mod, ff),
        ):
            mode, bsz, lrr, _, _ = get_v12_proj_spec(l_idx, pn, is_f)
            nb, asym = p3.MODE_TO_T16C[mode]
            if mode == 1:
                tern5_cnt += 1
            mlp_b += p3.expected_tile16_bytes(do, di, bsz, nb, asym) + 128 + di * 4
            if lrr > 0:
                lr_b += do * 4 + lrr * 4 + do * lrr + lrr * di

    total_b = embed_b + vision_bytes + attn_b + mlp_b + lr_b
    text_res_b = total_b - vision_bytes
    v10_b = int(_ORIG_V10_BUDGET["total_t16c_bytes"])
    return {
        "embed_bytes": embed_b,
        "vision_bytes": vision_bytes,
        "attn_bytes": attn_b,
        "mlp_bytes": mlp_b,
        "low_rank_bytes": lr_b,
        "total_t16c_bytes": total_b,
        "total_t16c_gib": round(total_b / (1024**3), 4),
        "text_resident_gib": round(text_res_b / (1024**3), 4),
        "delta_vs_v10_mib": round((total_b - v10_b) / (1024**2), 2),
        "tern5_cnt": tern5_cnt,
        "attn_1b_cnt": attn_1b_cnt,
        "mlp_2b_gauss_layers": mlp_2b_layers,
        "mlp_1b_rm_layers": mlp_1b_layers,
        "unprotected_cores": unprotected_cores,
        "tier2_bypass_layers": sorted(V12_TIER2_BYPASS_LAYERS),
        "protected_core_layers": sorted(V12_COGNITIVE_CORE_LAYERS),
    }


def run_stage2_5_v12_atlas_dual_e2e_kl(
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
    """Atlas-aware Dual-Regime 60-Layer E2E Logit-KL + L29 Early-Exit + Null-Bypass QAT."""
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
        atlas_sparse_tier2: bool = False,
        use_ckpt: bool = True,
        capture_layer: Optional[int] = V12_EARLY_LAYER,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = x_in.float()
        h_early: Optional[torch.Tensor] = None
        body_mlp_kf = float(V12_CRTR_TOP_BLOCKS) / float(V12_MLP_BLOCKS)
        for li in range(N_LAYERS):
            lyr = e2e_layers[li]
            if atlas_sparse_tier2 and li in V12_TIER2_BYPASS_LAYERS:
                h = h * lyr["layer_scalar"]
                if capture_layer is not None and li == capture_layer:
                    h_early = h
                continue

            is_core = li in V12_COGNITIVE_CORE_LAYERS
            hk = 1.0 if (not atlas_sparse_tier2 or is_core) else 0.25
            mk = 1.0 if (not atlas_sparse_tier2 or is_core) else body_mlp_kf

            is_f = LAYER_TYPES[li] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)

            def _fn(hx, _lyr=lyr, _isf=is_f, _nkv=nkv, _hd=hd, _rd=rot_d, _cs=cs_r, _sn=sn_r, _hk=hk, _mk=mk):
                return v93.forward_layer_e2e(
                    hx, _lyr, is_f=_isf, k_eq_v=K_EQ_V, nh=NH, nkv=_nkv, hd=_hd, rot_d=_rd,
                    cs_r=_cs, sn_r=_sn, rmsn_fn=rmsn_gpu, apply_rope_fn=apply_rope,
                    bits_tbl=bits_tbl, lv_tbls=lv_tbls, h_mat=h_mat_c,
                    head_keep_frac=_hk, mlp_keep_frac=_mk, use_ste_lr=True,
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

                hs, he = _forward_model(xb, atlas_sparse_tier2=False, use_ckpt=False)
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

                hs_sp, _ = _forward_model(xb, atlas_sparse_tier2=True, use_ckpt=False, capture_layer=None)
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
    kl_before, kl_sp_before, kl_e_before, cos_before, top1_before = _eval_val(min(8, cur_mbs))
    best_obj = 0.70 * kl_before + 0.30 * kl_sp_before
    best_snap = _snapshot_best()
    print(
        f"[V12 Stage 2.5 Atlas E2E Init] dense_kl={kl_before:.4f} | cca_tier2_kl={kl_sp_before:.4f} | "
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

                    do_sp = (step >= max(20, n_steps // 5)) and (acc_i % 2 == 1)
                    regime_w = 0.40 if do_sp else 1.0

                    hs, he = _forward_model(xb, atlas_sparse_tier2=do_sp, use_ckpt=True, capture_layer=(None if do_sp else V12_EARLY_LAYER))
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
                print(f"[V12 Stage 2.5 OOM Backoff] Switched to micro_bs={cur_mbs}, grad_accum={cur_acc}", flush=True)

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
                v_kl, v_sp, v_e, v_cos, v_t1 = _eval_val(min(8, cur_mbs))
                obj = 0.70 * v_kl + 0.30 * v_sp
                improved = ""
                if obj < best_obj:
                    best_obj = obj
                    best_snap = _snapshot_best()
                    improved = " [BEST]"
                print(
                    f"  [V12 E2E step {step+1:03d}/{n_steps}] tr_kl={step_kl:.4f} dense_kl={v_kl:.4f} "
                    f"cca_tier2_kl={v_sp:.4f} l29_kl={v_e:.4f} top1={v_t1*100:.2f}% dt={time.perf_counter()-t0_e2e:.1f}s{improved}",
                    flush=True,
                )
            else:
                print(
                    f"  [V12 E2E step {step+1:03d}/{n_steps}] tr_kl={step_kl:.4f} mbs={cur_mbs}x{cur_acc} dt={time.perf_counter()-t0_e2e:.1f}s",
                    flush=True,
                )

    _restore_best(best_snap)
    kl_after, kl_sp_after, kl_e_after, cos_after, top1_after = _eval_val(min(8, cur_mbs))
    del emb_gpu, opt, best_snap, bits_tbl, lv_tbls
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
        f"[V12 Stage 2.5 Done] dense_kl: {kl_before:.4f}->{kl_after:.4f} | "
        f"cca_tier2_kl: {kl_sp_before:.4f}->{kl_sp_after:.4f} | top1: {top1_before*100:.2f}%->{top1_after*100:.2f}%",
        flush=True,
    )
    return {
        "val_kl_before": round(kl_before, 4),
        "val_kl_after": round(kl_after, 4),
        "val_cca_tier2_kl_before": round(kl_sp_before, 4),
        "val_cca_tier2_kl_after": round(kl_sp_after, 4),
        "val_l29_kl_after": round(kl_e_after, 4),
        "l59_cos_before": round(cos_before, 5),
        "l59_cos_after": round(cos_after, 5),
        "top1_agree_before": round(top1_before, 4),
        "top1_agree_after": round(top1_after, 4),
        "effective_micro_bs": cur_mbs,
        "effective_grad_accum": cur_acc,
        "e2e_elapsed_s": round(time.perf_counter() - t0_e2e, 2),
    }


def _bind_v12_hooks() -> None:
    """Bind V12 paths, revision, spec, and Stage 2.5 Atlas-Aware trainer into V10's proven scaffold."""
    v10.OUT_DIR = OUT_DIR
    v10.CKPT_DIR = CKPT_DIR
    v10.DRIVE_DIR = DRIVE_DIR
    v10.OUT_T16C = OUT_T16C
    v10.OUT_CRTR = OUT_CRTR
    v10.OUT_HEAD_CRTR = OUT_HEAD_CRTR
    v10.OUT_C8D = OUT_C8D
    v10.OUT_AST_Q2 = OUT_AST_Q2
    v10.OUT_AST_Q4 = OUT_AST_Q4
    v10.OUT_BHDR = OUT_BHDR
    v10.OUT_REPORT = OUT_REPORT
    v10.V10_PIPELINE_REVISION = V12_PIPELINE_REVISION
    v10.MAX_V10_T16C_BYTES = MAX_V12_T16C_BYTES
    v10.get_v10_proj_spec = get_v12_proj_spec
    v10.compute_v10_budget_bytes = compute_v12_budget_bytes
    v10.run_stage2_5_v10_dual_e2e_kl = run_stage2_5_v12_atlas_dual_e2e_kl


def run_v12_pipeline(*, e2e_steps: int = 200, micro_bs: int = 8, grad_accum: int = 2, fresh: bool = False) -> Dict[str, object]:
    """Execute complete V12 Atlas-Guided Cortical Cascade build by hooking V10's verified Stage 1..3 engine."""
    _bind_v12_hooks()

    if torch.cuda.is_available():
        vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
        if vram_gb >= 72.0 and micro_bs == 8:
            # On A100-80GB / G4-96GB, larger microbatch halves wall-clock step count overhead.
            micro_bs = 16
            grad_accum = 1

    # Prevent premature auto-unassign inside v10 so we write `SHA256SUMS_V12.txt` + `v12_report.json` cleanly first.
    orig_unassign = os.environ.get("COLAB_AUTO_UNASSIGN", "1")
    os.environ["COLAB_AUTO_UNASSIGN"] = "0"
    try:
        rpt = v10.run_v10_pipeline(e2e_steps=e2e_steps, micro_bs=micro_bs, grad_accum=grad_accum, fresh=fresh)
    finally:
        os.environ["COLAB_AUTO_UNASSIGN"] = orig_unassign

    rpt["version"] = "V12-ATLAS-CORTICAL-CASCADE-DUAL-E2E-KL"
    rpt["revision"] = V12_PIPELINE_REVISION
    rpt["protected_cognitive_core_layers"] = sorted(V12_COGNITIVE_CORE_LAYERS)
    rpt["tier2_bypass_layers"] = sorted(V12_TIER2_BYPASS_LAYERS)
    OUT_REPORT.write_text(json.dumps(rpt, indent=2) + "\n", encoding="utf-8")

    sums_v12 = OUT_DIR / "SHA256SUMS_V12.txt"
    sums_v12.write_text("".join(f"{v['sha256']}  {k}\n" for k, v in rpt["artifacts"].items()), encoding="utf-8")
    (OUT_DIR / "SHA256SUMS_V10.txt").unlink(missing_ok=True)
    if DRIVE_DIR is not None:
        (DRIVE_DIR / "SHA256SUMS_V10.txt").unlink(missing_ok=True)
        v90._drive_safe_copy(OUT_REPORT, DRIVE_DIR / OUT_REPORT.name)
        v90._drive_safe_copy(sums_v12, DRIVE_DIR / sums_v12.name)
        expected_sha = {k: v["sha256"] for k, v in rpt["artifacts"].items()}
        expected_sha[OUT_REPORT.name] = v90._sha256_file(OUT_REPORT)
        expected_sha[sums_v12.name] = v90._sha256_file(sums_v12)
        v90._verify_drive_sha256(DRIVE_DIR, expected_sha)

    print(
        f"[V12 COMPLETE] {OUT_T16C.name}={rpt['t16c_gib']} GiB "
        f"(delta={rpt['budget']['delta_vs_v10_mib']} MiB vs V10) | "
        f"cores={len(V12_COGNITIVE_CORE_LAYERS)} | bypass={sorted(V12_TIER2_BYPASS_LAYERS)}",
        flush=True,
    )
    if orig_unassign == "1":
        v90._flush_drive_before_unassign()
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:
            print(f"[Auto-Unassign Note] {exc}", flush=True)
    return rpt


def run_selftest() -> int:
    """Mandatory 6-gate fail-closed CPU verification battery for V12 Atlas-Guided Cortical Cascade."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: V12 budget <= V10 (`<= 6.28 GiB`), live `_bind_v12_hooks()` non-recursive parity, & 16/16 Cores Protected
    _bind_v12_hooks()
    bgt = v10.compute_v10_budget_bytes()
    v10_bgt = _ORIG_V10_BUDGET
    l32_spec = v10.get_v10_proj_spec(32, "mlp.gate_proj")
    l50_spec = v10.get_v10_proj_spec(50, "mlp.gate_proj")
    g1_ok = (
        int(bgt["total_t16c_bytes"]) <= MAX_V12_T16C_BYTES
        and int(bgt["total_t16c_bytes"]) < int(v10_bgt["total_t16c_bytes"])
        and int(bgt["tern5_cnt"]) == 0
        and int(bgt["attn_1b_cnt"]) == 0
        and len(bgt["unprotected_cores"]) == 0
        and V12_COGNITIVE_CORE_LAYERS.isdisjoint(V12_TIER2_BYPASS_LAYERS)
        and l32_spec[0] == 5
        and l50_spec[0] == 0
        and v10.OUT_T16C == OUT_T16C
    )
    gate(
        "Gate 1: V12 Atlas Budget < V10 & 16/16 Cores Protected",
        g1_ok,
        f"total={bgt['total_t16c_gib']} GiB ({bgt['delta_vs_v10_mib']} MiB vs V10), 2b_mlps={bgt['mlp_2b_gauss_layers']}, 1b_mlps={bgt['mlp_1b_rm_layers']}",
    )

    # Gate 2: Vectorized bitplane unpack bit-exact parity across 1b/2b_gauss/3b
    rng = np.random.default_rng(12)
    h256 = torch.from_numpy(pa.make_sylvester_hadamard(V12_BLOCK_SIZE)).float()
    bits_tbl, lv_tbls = v93.make_unpack_tables(torch.device("cpu"), dtype=torch.float32)
    max_diff = 0.0
    for mode_i in (0, 5, 4, 6):
        do, di = 32, 512
        nb_m, asym_m = p3.MODE_TO_T16C[mode_i]
        n_tiles, n_blks = do // 16, di // V12_BLOCK_SIZE
        tblk_b = p3.tile16_blk_bytes(V12_BLOCK_SIZE, nb_m, asym_m)
        pb = rng.integers(0, 256, size=n_tiles * n_blks * tblk_b, dtype=np.uint8)
        pb_r = pb.reshape(n_tiles, n_blks, tblk_b)
        pb_r[:, :, :32] = np.full((n_tiles, n_blks, 16), 0.05, dtype="<f2").view(np.uint8).reshape(n_tiles, n_blks, 32)
        if asym_m:
            pb_r[:, :, 32:64] = 0
        u_ref, _, _ = v90.unpack_tile16_headers_and_units(pb, do, di, V12_BLOCK_SIZE, mode_i, h256)
        pl = v93.extract_planes_u8(pb, do, di, V12_BLOCK_SIZE, mode_i)
        u_got = v93.unpack_planes_to_units(pl, mode_i, do, di, V12_BLOCK_SIZE, bits_tbl, lv_tbls[mode_i])
        max_diff = max(max_diff, float((u_ref.reshape(do, n_blks, V12_BLOCK_SIZE) - u_got).abs().max().item()))
    gate("Gate 2: Bitplane Unpack Parity (1b/2b_gauss/3b)", max_diff < 1e-6, f"max_diff={max_diff:.2e}")

    # Gate 3: STE INT8 Low-Rank parity & ALS fit
    u_lr = torch.from_numpy((rng.standard_normal((32, 8)) * 0.04).astype(np.float32))
    v_lr = torch.from_numpy((rng.standard_normal((8, 64)) * 0.04).astype(np.float32))
    u_ste, v_ste = v93.ste_int8_lr(u_lr, v_lr)
    su, sv, ui8, vi8 = v93.quant_lr_int8_tuple(u_lr, v_lr)
    w_ste = u_ste @ v_ste
    w_exp = (torch.from_numpy(ui8.astype(np.float32)) * torch.from_numpy(su).unsqueeze(1)) @ (
        torch.from_numpy(vi8.astype(np.float32)) * torch.from_numpy(sv).unsqueeze(1)
    )
    lr_diff = float((w_ste - w_exp).abs().max().item())
    x_als = torch.randn(48, 64)
    w_als_tgt = (u_lr * 2.5) @ (v_lr * 2.5) + torch.randn(32, 64) * 0.002
    als_t, als_d = v10.fit_v10_als_lr_int8(w_als_tgt, torch.zeros_like(w_als_tgt), x_als, 8)
    als_ok = (als_t is not None and als_d is not None)
    gate("Gate 3: STE INT8 Low-Rank Export Parity & ALS Fit", lr_diff < 1e-6 and als_ok, f"max_diff={lr_diff:.2e}")

    # Gate 4: Authentic Dual-Regime + Layer-Scalar Bypass BitSerial KL Convergence
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
    bypass_scalar = nn.Parameter(torch.tensor([1.0]))
    opt_m = torch.optim.AdamW([proj_m["sc"], proj_m["u_lr"], proj_m["v_lr"], bypass_scalar], lr=2e-2)
    kl0 = kl1 = 0.0
    for s_i in range(30):
        opt_m.zero_grad()
        hs_m = v93.bitserial_linear(x0 * bypass_scalar, proj_m, bits_tbl, lv_tbls, h64, use_ste_lr=True)
        lgs_m = 30.0 * torch.tanh((hs_m @ emb_t.T) / 30.0)
        loss_m = v93.compute_topk_kl_loss(lgs_m, tid_all, tp_all)
        if s_i == 0:
            kl0 = float(loss_m.item())
        loss_m.backward()
        opt_m.step()
        kl1 = float(loss_m.item())
    gate("Gate 4: Atlas Bypass + BitSerial Real KL Convergence", kl1 < 0.65 * kl0, f"kl0={kl0:.4f} -> kl1={kl1:.4f}")

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

    # Gate 6: Target-Embed Perm-Pi Equivariance
    d_toy = 64
    perm = torch.randperm(d_toy)
    x_a, x_b = torch.randn(4, d_toy), torch.randn(4, d_toy)
    pre_w, post_w = torch.randn(32, 2 * d_toy), torch.randn(d_toy, 32)
    sd_perm = v10.permute_assistant_state_dict({"pre_projection.weight": pre_w, "post_projection.weight": post_w}, perm, d_toy)
    y_orig = F.linear(F.linear(torch.cat([x_a, x_b], dim=-1), pre_w), post_w)
    y_perm = F.linear(F.linear(torch.cat([x_a[:, perm], x_b[:, perm]], dim=-1), sd_perm["pre_projection.weight"]), sd_perm["post_projection.weight"])
    perm_err = float((y_orig[:, perm] - y_perm).abs().max().item())
    gate("Gate 6: Assistant Perm-Pi Equivariance", perm_err < 1e-4, f"perm_err={perm_err:.2e}")

    print(f"V12 Self-Test Result: {6 - failures}/6 gates passed", flush=True)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="Language-U Gemma 4 31B V12 Atlas-Guided Cortical Cascade GPU Pipeline")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--e2e-steps", type=int, default=200)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.selftest or (not args.run and not torch.cuda.is_available()):
        return run_selftest()
    run_v12_pipeline(e2e_steps=args.e2e_steps, micro_bs=args.micro_bs, grad_accum=args.grad_accum, fresh=args.fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

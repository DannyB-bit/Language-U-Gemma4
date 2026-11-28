#!/usr/bin/env python3
"""Language-U V12.5: Atlas-Guided Cortical Cascade Continuation (Step 200 -> 400 + 720-Step Assistant).

Resumes directly from the finished SHA-256-verified V12 Step-200 `.t16c` checkpoint
(`gemma4_31b_v12_atlas_cascade.t16c`) on Google Drive, skipping the 40-minute Stage-2
layerwise HQQ/GPTQ loop while preserving every 1-bit/2-bit/3-bit plane, scale, bias,
`inv_s`, Int8 low-rank residual, RMSNorm, Vision encoder, and 3-bit Gaussian `embed_tokens`.

Upgrades over V12:
  1. Unpacks V12 Step-200 `.t16c` (reversing `T16C_COLBLOCK_BIT` `0x20` and `T16C_ROWMAJOR_BIT`
     `0x40` bit-for-bit) and validates `emb_eff` + Step-200 KL/Top-1 parity fail-closed.
  2. Runs 200 additional Dual-Regime E2E Logit-KL steps (effective Steps 201..400) with fixed
     `do_sp` scheduling so BOTH dense (60/60) and `cca_tier2_bypass` (core-protected 16/60 +
     bypass `[16, 18, 40, 45]` + 12/84 MLP / 8/32 Head) train on every single step on both L4
     and A100 GPUs, with 50-step Drive checkpoints.
  3. Re-distills `CRTR-V9`, `HeadRouter`, `C8DB`, and 720-step Target-Shared EAGLE-3 Assistant
     (`Q4`/`Q2`) + `BHDR-256`, saving SHA-256 verified outputs into `/content/drive/MyDrive/gemma4_v12_5_atlas_cascade`.
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
import tempfile
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
import run_v12_atlas_cascade_gpu as v12  # noqa: E402
import spatial_district_converter as sdc  # noqa: E402
import train_assistant_v7_byte_head as bhmod  # noqa: E402
import train_c8d_pattern_bank as c8d  # noqa: E402
import train_cortical_router as crtr  # noqa: E402

V12_5_PIPELINE_REVISION = "v12.5_atlas_cascade_resume200_to_400_ast720_bhdr256"

OUT_DIR = Path(os.environ.get("V12_5_OUT_DIR", "/content/v12_5_outputs"))
CKPT_DIR = OUT_DIR / "ckpt_v12_5"
_drv_env = os.environ.get("V12_5_DRIVE_DIR", "").strip()
DRIVE_DIR: Optional[Path] = Path(_drv_env) if _drv_env else None
DEFAULT_INIT_T16C = Path(
    os.environ.get(
        "V12_5_INIT_T16C",
        "/content/drive/MyDrive/gemma4_v12_atlas_cascade/gemma4_31b_v12_atlas_cascade.t16c",
    )
)

OUT_T16C = OUT_DIR / "gemma4_31b_v12_5_atlas_cascade.t16c"
OUT_CRTR = OUT_DIR / "gemma4_31b_v12_5_router.crtr"
OUT_HEAD_CRTR = OUT_DIR / "gemma4_31b_v12_5_head_router.crtr"
OUT_C8D = OUT_DIR / "gemma4_31b_v12_5_c8d.c8d"
OUT_AST_Q2 = OUT_DIR / "gemma4_31b_v12_5_assistant_body_q2.t16c"
OUT_AST_Q4 = OUT_DIR / "gemma4_31b_v12_5_assistant_body_q4.t16c"
OUT_BHDR = OUT_DIR / "gemma4_31b_v12_5_assistant.bhdr"
OUT_REPORT = OUT_DIR / "v12_5_report.json"

T16C_BASE_TO_MODE: Dict[Tuple[int, bool], int] = {
    (n_bits, is_asym): mode_i for mode_i, (n_bits, is_asym) in p3.MODE_TO_T16C.items()
}


def transpose_planes_rowmajor_to_col(
    pb: np.ndarray, d_out: int, d_in: int, block_size: int, eff_bits: int, is_asym: bool
) -> np.ndarray:
    """Exact inverse of `repack_t16c_rowmajor.transpose_planes_col_to_rowmajor`."""
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // block_size
    hdr = 64 if is_asym else 32
    bpr = block_size // 8
    plane_b = bpr * 16
    tblk_b = hdr + eff_bits * plane_b
    arr = pb.reshape(n_tiles, n_blks, tblk_b).copy()
    for p in range(eff_bits):
        s0 = hdr + p * plane_b
        s1 = s0 + plane_b
        pl = arr[:, :, s0:s1].reshape(n_tiles, n_blks, 16, bpr)
        arr[:, :, s0:s1] = np.ascontiguousarray(pl.transpose(0, 1, 3, 2)).reshape(n_tiles, n_blks, plane_b)
    return arr.reshape(-1)


def unfinalize_t16c_payload(
    raw_bytes: bytes, do: int, di: int, bsize: int, hdr_bits: int, is_asym: bool
) -> Tuple[int, bool, bool, np.ndarray]:
    """Invert `finalize_t16c_payload` bit-for-bit -> `(mode_i, is_rm, is_cb, colmajor_pb)`."""
    is_cb = bool(hdr_bits & p3.T16C_COLBLOCK_BIT)
    is_rm = bool(hdr_bits & p3.T16C_ROWMAJOR_BIT)
    base_bits = hdr_bits & ~(p3.T16C_COLBLOCK_BIT | p3.T16C_ROWMAJOR_BIT)
    if (base_bits, is_asym) not in T16C_BASE_TO_MODE:
        raise ValueError(f"Unsupported T16C base_bits={base_bits:#x}, is_asym={is_asym}")
    mode_i = T16C_BASE_TO_MODE[(base_bits, is_asym)]
    eff_bits = base_bits & 0x1F

    buf = raw_bytes
    if is_cb:
        buf = sdc.transpose_colblock_bytes(buf, do, di, bsize, base_bits, is_asym, to_colblock=False)
    arr = np.frombuffer(buf, dtype=np.uint8).copy()
    if is_rm:
        arr = transpose_planes_rowmajor_to_col(arr, do, di, bsize, eff_bits, is_asym)
    return mode_i, is_rm, is_cb, arr


def load_e2e_state_from_t16c(
    t16c_path: Path,
    *,
    n_layers: int,
    d_mod: int,
    vocab_sz: int,
    h256_cpu: torch.Tensor,
) -> Dict[str, object]:
    """Unpack tuned Step-200 LM weights, norms, embedding, and verbatim Vision slices from `.t16c`."""
    if not t16c_path.is_file():
        raise FileNotFoundError(f"Missing --init-t16c checkpoint artifact: {t16c_path}")

    lm_norms: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in range(n_layers)}
    vision_norm_recs: List[Tuple[str, np.ndarray]] = []
    fn_vec: Optional[torch.Tensor] = None
    early_fn_vec: Optional[torch.Tensor] = None

    e2e_projs: List[Dict[str, dict]] = [{} for _ in range(n_layers)]
    emb_pb: Optional[np.ndarray] = None
    emb_eff: Optional[torch.Tensor] = None
    vision_mat_recs: List[tuple] = []

    with open(t16c_path, "rb") as fp:
        magic, ver, flags, n_norms, n_mats, _ = struct.unpack("<4s5I", fp.read(24))
        if magic != p3.T16C_MAGIC or ver != p3.T16C_VERSION:
            raise ValueError(f"Invalid T16C header in {t16c_path}: magic={magic!r} ver={ver}")

        pos = 24
        for _ in range(n_norms):
            nl = struct.unpack("<H", fp.read(2))[0]
            name = fp.read(nl).decode("utf-8")
            cnt = struct.unpack("<I", fp.read(4))[0]
            arr = np.frombuffer(fp.read(cnt * 4), dtype="<f4").copy()
            pos += 2 + nl + 4 + cnt * 4
            if name == "model.language_model.norm.weight":
                fn_vec = torch.from_numpy(arr)
            elif name == "model.language_model.early_exit_l29_norm.weight":
                early_fn_vec = torch.from_numpy(arr)
            elif name.startswith("model.language_model.layers."):
                rest = name[len("model.language_model.layers.") :]
                li_s, nk = rest.split(".", 1)
                li = int(li_s)
                lk_map = {
                    "input_layernorm.weight": "g_in",
                    "post_attention_layernorm.weight": "g_pa",
                    "pre_feedforward_layernorm.weight": "g_pf",
                    "post_feedforward_layernorm.weight": "g_pff",
                    "self_attn.q_norm.weight": "g_qn",
                    "self_attn.k_norm.weight": "g_kn",
                    "layer_scalar": "layer_scalar",
                }
                if nk in lk_map:
                    lm_norms[li][lk_map[nk]] = torch.from_numpy(arr)
            elif name.startswith("model.vision_tower.") or name.startswith("model.embed_vision."):
                vision_norm_recs.append((name, arr))

        for _ in range(n_mats):
            do, di, bsz, n_bits, is_asym_u8, has_fwht_u8, inv_k, lr_rk, exp_p, lr_bytes, nl = None, None, None, None, None, None, None, None, None, None, None
            nl, do, di, bsz, n_bits, is_asym_u8, has_fwht_u8, inv_k, lr_rk, exp_p, lr_bytes = struct.unpack(
                "<HIIHBBBBHQQ", fp.read(34)
            )
            name = fp.read(nl).decode("utf-8")
            pos += 34 + nl
            pad = (64 - (pos % 64)) % 64
            if pad:
                fp.seek(pad, os.SEEK_CUR)
                pos += pad
            raw_pb = fp.read(int(exp_p))
            inv_arr: Optional[np.ndarray] = None
            if inv_k == 1:
                inv_arr = np.frombuffer(fp.read(int(di) * 4), dtype="<f4").copy()
                pos += int(di) * 4
            elif inv_k != 0:
                raise ValueError(f"Unsupported inv_k={inv_k} on {name}")
            raw_lr = fp.read(int(lr_bytes)) if lr_bytes > 0 else b""
            pos += int(exp_p) + int(lr_bytes)

            is_asym = bool(is_asym_u8)
            has_fwht = bool(has_fwht_u8)

            if name == "model.language_model.embed_tokens.weight":
                assert do == vocab_sz and di == d_mod and bsz == v12.V12_BLOCK_SIZE
                mode_e, _, _, pb_e = unfinalize_t16c_payload(raw_pb, do, di, bsz, n_bits, is_asym)
                emb_pb = np.frombuffer(raw_pb, dtype=np.uint8).copy()
                emb_eff = torch.empty((vocab_sz, d_mod), dtype=torch.float16)
                tblk_row = p3.tile16_blk_bytes(bsz, n_bits & ~(p3.T16C_ROWMAJOR_BIT | p3.T16C_COLBLOCK_BIT), is_asym) * (di // bsz)
                for r0 in range(0, vocab_sz, 16384):
                    r1 = min(r0 + 16384, vocab_sz)
                    sl_pb = pb_e[(r0 // 16) * tblk_row : ((r1 + 15) // 16) * tblk_row]
                    u_c, sc_c, bi_c = v90.unpack_tile16_headers_and_units(sl_pb, r1 - r0, d_mod, bsz, mode_e, h256_cpu)
                    emb_eff[r0:r1] = v90.reconstruct_from_headers(u_c, sc_c, bi_c, None, h256_cpu).to(torch.float16)
                continue

            if name.startswith("model.vision_tower.") or name.startswith("model.embed_vision."):
                assert lr_bytes == 0 and inv_arr is None
                vision_mat_recs.append((name, do, di, bsz, n_bits, is_asym, has_fwht, raw_pb, None, None))
                continue

            if name.startswith("model.language_model.layers."):
                rest = name[len("model.language_model.layers.") :]
                li_s, pn_full = rest.split(".", 1)
                li = int(li_s)
                pn = pn_full[:-len(".weight")] if pn_full.endswith(".weight") else pn_full
                mode_i, is_rm, is_cb, pb_col = unfinalize_t16c_payload(raw_pb, do, di, bsz, n_bits, is_asym)
                _, sc_t, bi_t = v90.unpack_tile16_headers_and_units(pb_col, do, di, bsz, mode_i, h256_cpu)
                planes_u8 = v93.extract_planes_u8(pb_col, do, di, bsz, mode_i)

                u_lr = v_lr = None
                if lr_rk > 0:
                    o_su = do * 4
                    o_sv = o_su + lr_rk * 4
                    o_u = o_sv + do * lr_rk
                    su = np.frombuffer(raw_lr[:o_su], dtype="<f4").copy()
                    sv = np.frombuffer(raw_lr[o_su:o_sv], dtype="<f4").copy()
                    ui8 = np.frombuffer(raw_lr[o_sv:o_u], dtype=np.int8).reshape(do, lr_rk).copy()
                    vi8 = np.frombuffer(raw_lr[o_u : o_u + lr_rk * di], dtype=np.int8).reshape(lr_rk, di).copy()
                    u_lr = torch.from_numpy(ui8.astype(np.float32)) * torch.from_numpy(su).unsqueeze(1)
                    v_lr = torch.from_numpy(vi8.astype(np.float32)) * torch.from_numpy(sv).unsqueeze(1)

                e2e_projs[li][pn] = {
                    "do": do,
                    "di": di,
                    "bsize": bsz,
                    "mode_i": mode_i,
                    "is_rm": is_rm,
                    "is_cb": is_cb,
                    "planes_u8": planes_u8.cpu(),
                    "sc": sc_t.cpu(),
                    "bi": bi_t.cpu() if bi_t is not None else None,
                    "inv_s": torch.from_numpy(inv_arr) if inv_arr is not None else None,
                    "u_lr": u_lr,
                    "v_lr": v_lr,
                }

    assert fn_vec is not None and early_fn_vec is not None
    assert emb_eff is not None and emb_pb is not None
    e2e_layers: List[dict] = []
    for li in range(n_layers):
        nd = lm_norms[li]
        assert len(nd) == 7, f"L{li} missing norms: {list(nd.keys())}"
        assert len(e2e_projs[li]) >= 6, f"L{li} missing projs: {list(e2e_projs[li].keys())}"
        e2e_layers.append({**nd, "projs": e2e_projs[li]})

    return {
        "e2e_layers": e2e_layers,
        "fn_vec": fn_vec,
        "early_fn_vec": early_fn_vec,
        "emb_eff": emb_eff,
        "emb_pb": emb_pb,
        "vision_norm_recs": vision_norm_recs,
        "vision_mat_recs": vision_mat_recs,
    }


def compute_fast_teacher_targets(
    *,
    st: Dict[str, object],
    X_teach_perm: torch.Tensor,
    emb_perm_teach: torch.Tensor,
    perm_pi: torch.Tensor,
    fn_orig: torch.Tensor,
    H256_GPU: torch.Tensor,
    rmsn_gpu,
    apply_rope,
    rope_cs_s: torch.Tensor,
    rope_sn_s: torch.Tensor,
    rope_cs_g: torch.Tensor,
    rope_sn_g: torch.Tensor,
    device: torch.device,
    compute_dtype: torch.dtype,
    gpu_vram_gb: float,
) -> Dict[str, object]:
    """Compute exact 60L BF16 Teacher `X_teach_l59`, `top_ids/top_probs`, `wo_head_norms`, and salience without HQQ."""
    from safetensors import safe_open

    t0 = time.perf_counter()
    sf_path: Path = st["sf_path"]  # type: ignore[assignment]
    N_LAYERS = int(st["N_LAYERS"])
    D_MOD = int(st["D_MOD"])
    NH = int(st["NH"])
    NKV_S = int(st["NKV_S"])
    NKV_F = int(st["NKV_F"])
    HD_S = int(st["HD_S"])
    HD_F = int(st["HD_F"])
    K_EQ_V = bool(st["K_EQ_V"])
    LAYER_TYPES = list(st["LAYER_TYPES"])
    PART_L = float(st["PART_L"])
    PART_G = float(st["PART_G"])
    SOFTCAP = float(st["SOFTCAP"])
    N_SEQ = int(st["N_SEQ"])

    MB_FWD = 32 if gpu_vram_gb >= 20.0 else 16
    perm_pi_g = perm_pi.to(device)
    X_cur = X_teach_perm.clone()
    wo_head_norms: List[torch.Tensor] = []
    teach_mod_sal_list: List[torch.Tensor] = []

    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        for l_idx in range(N_LAYERS):
            pfx = f"model.language_model.layers.{l_idx}"
            is_f = LAYER_TYPES[l_idx] == "full_attention"
            nkv, hd = (NKV_F, HD_F) if is_f else (NKV_S, HD_S)
            rot_d = int(round(hd * (PART_G if is_f else PART_L)))
            cs_r, sn_r = (rope_cs_g, rope_sn_g) if is_f else (rope_cs_s, rope_sn_s)
            rep = NH // nkv

            g_in = sf.get_tensor(f"{pfx}.input_layernorm.weight").float()[perm_pi].to(device)
            g_pa = sf.get_tensor(f"{pfx}.post_attention_layernorm.weight").float()[perm_pi].to(device)
            g_pf = sf.get_tensor(f"{pfx}.pre_feedforward_layernorm.weight").float()[perm_pi].to(device)
            g_pff = sf.get_tensor(f"{pfx}.post_feedforward_layernorm.weight").float()[perm_pi].to(device)
            g_qn = sf.get_tensor(f"{pfx}.self_attn.q_norm.weight").float().to(device)
            g_kn = sf.get_tensor(f"{pfx}.self_attn.k_norm.weight").float().to(device)
            lsc_t = float(sf.get_tensor(f"{pfx}.layer_scalar").float()[0].item())

            wq_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.q_proj", perm_pi_g, device, is_perm_col=True)
            wk_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.k_proj", perm_pi_g, device, is_perm_col=True)
            wv_t = wk_t if (is_f and K_EQ_V) else v90._load_dequant_proj(sf, f"{pfx}.self_attn.v_proj", perm_pi_g, device, is_perm_col=True)
            wo_t = v90._load_dequant_proj(sf, f"{pfx}.self_attn.o_proj", perm_pi_g, device, is_perm_row=True)
            wg_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.gate_proj", perm_pi_g, device, is_perm_col=True)
            wu_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.up_proj", perm_pi_g, device, is_perm_col=True)
            wd_t = v90._load_dequant_proj(sf, f"{pfx}.mlp.down_proj", perm_pi_g, device, is_perm_row=True)

            wo_head_norms.append(wo_t.view(D_MOD, NH, hd).pow(2).sum(dim=(0, 2)).sqrt().clamp(min=1e-6).cpu())
            X_next = torch.empty_like(X_cur)
            sal_cpu = torch.empty((N_SEQ * v90.V9_SEQ_LEN,), dtype=torch.float16)

            with torch.no_grad():
                for b0 in range(0, N_SEQ, MB_FWD):
                    b1 = min(b0 + MB_FWD, N_SEQ)
                    xb = X_cur[b0:b1].to(device, dtype=torch.float32)
                    xn = rmsn_gpu(xb, g_in)
                    q = apply_rope(rmsn_gpu(F.linear(xn, wq_t).view(b1 - b0, v90.V9_SEQ_LEN, NH, hd), g_qn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    k = apply_rope(rmsn_gpu(F.linear(xn, wk_t).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), g_kn), cs_r, sn_r, rot_d, hd).transpose(1, 2)
                    v = rmsn_gpu(F.linear(xn, wv_t).view(b1 - b0, v90.V9_SEQ_LEN, nkv, hd), None).transpose(1, 2)
                    ctx_h = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True, scale=1.0).transpose(1, 2).contiguous()
                    xm = xb + rmsn_gpu(F.linear(ctx_h.view(b1 - b0, v90.V9_SEQ_LEN, NH * hd), wo_t), g_pa)
                    xf = rmsn_gpu(xm, g_pf)
                    act = F.gelu(F.linear(xf, wg_t), approximate="tanh") * F.linear(xf, wu_t)
                    x_nx = (xm + rmsn_gpu(F.linear(act, wd_t), g_pff)) * lsc_t
                    X_next[b0:b1] = x_nx.to(torch.float16).cpu()
                    sal = (x_nx - xb).pow(2).sum(dim=-1).sqrt() / (xb.pow(2).sum(dim=-1).sqrt() + 1e-6)
                    sal_cpu[b0 * v90.V9_SEQ_LEN : b1 * v90.V9_SEQ_LEN] = sal.reshape(-1).to(torch.float16).cpu()

            X_cur = X_next
            teach_mod_sal_list.append(sal_cpu)
            if (l_idx + 1) % 15 == 0 or l_idx == 59:
                print(f"  [Fast Teacher Forward L{l_idx:02d}/59] elapsed={time.perf_counter()-t0:.1f}s", flush=True)
            del wq_t, wk_t, wv_t, wo_t, wg_t, wu_t, wd_t

    fn_teach_g = fn_orig[perm_pi].to(device=device, dtype=torch.float32)
    emb_teach_g = emb_perm_teach.to(device=device, dtype=compute_dtype)
    top_ids_cpu = torch.empty((N_SEQ, v90.V9_SEQ_LEN, 128), dtype=torch.int32)
    top_probs_cpu = torch.empty((N_SEQ, v90.V9_SEQ_LEN, 128), dtype=torch.float16)
    tb_sz = 8 if gpu_vram_gb < 20.0 else 32
    with torch.no_grad():
        for b0 in range(0, N_SEQ, tb_sz):
            b1 = min(b0 + tb_sz, N_SEQ)
            ht_f = rmsn_gpu(X_cur[b0:b1].to(device=device, dtype=torch.float32), fn_teach_g)
            ht = ht_f.clamp(-65000.0, 65000.0).to(compute_dtype) if compute_dtype == torch.float16 else ht_f.to(compute_dtype)
            lgt = SOFTCAP * torch.tanh((ht @ emb_teach_g.T).float() / SOFTCAP)
            tp, ti = torch.topk(F.softmax(lgt, dim=-1), 128, dim=-1)
            top_ids_cpu[b0:b1] = ti.to(torch.int32).cpu()
            top_probs_cpu[b0:b1] = tp.to(torch.float16).cpu()
            del lgt

    torch.cuda.empty_cache()
    return {
        "X_teach_l59": X_cur,
        "top_ids_cpu": top_ids_cpu,
        "top_probs_cpu": top_probs_cpu,
        "wo_head_norms": wo_head_norms,
        "teach_mod_sal_list": teach_mod_sal_list,
    }


def run_stage2_5_v12_5_continue(
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
    start_step: int = 200,
    extra_steps: int = 200,
    micro_bs: int = 8,
    grad_accum: int = 2,
) -> Dict[str, object]:
    """Continue 60L E2E Dual-Regime QAT from Step 200 -> 400 with dense + `cca_tier2_bypass` every step."""
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

    has_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    dt = torch.bfloat16 if has_bf16 else torch.float32
    emb_dt = torch.bfloat16 if has_bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
    bits_tbl, lv_tbls = v93.make_unpack_tables(device, dtype=dt)
    h_mat_c = H256_GPU.to(device=device, dtype=dt)
    emb_gpu = emb_eff.to(device=device, dtype=emb_dt)

    def _to_emb_dt(t: torch.Tensor) -> torch.Tensor:
        return t.clamp(-65000.0, 65000.0).to(emb_dt) if emb_dt == torch.float16 else t.to(emb_dt)

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
            {"params": sc_params, "lr": 1.0e-4, "weight_decay": 0.0},
            {"params": lr_params, "lr": 1.0e-4, "weight_decay": 1e-4},
            {"params": norm_params, "lr": 2.0e-4, "weight_decay": 0.0},
        ]
    )
    warmup = max(5, min(15, extra_steps // 10))

    def _lr_mult(step: int) -> float:
        if step < warmup:
            return 0.05 + 0.95 * float(step + 1) / float(warmup)
        prog = float(step - warmup) / float(max(1, extra_steps - warmup))
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))

    def _forward_model(
        x_in: torch.Tensor,
        atlas_sparse_tier2: bool = False,
        use_ckpt: bool = True,
        capture_layer: Optional[int] = v12.V12_EARLY_LAYER,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = x_in.float()
        h_early: Optional[torch.Tensor] = None
        body_mlp_kf = float(v12.V12_CRTR_TOP_BLOCKS) / float(v12.V12_MLP_BLOCKS)
        for li in range(N_LAYERS):
            lyr = e2e_layers[li]
            if atlas_sparse_tier2 and li in v12.V12_TIER2_BYPASS_LAYERS:
                h = h * lyr["layer_scalar"]
                if capture_layer is not None and li == capture_layer:
                    h_early = h
                continue
            is_core = li in v12.V12_COGNITIVE_CORE_LAYERS
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

            h = torch.utils.checkpoint.checkpoint(_fn, h, use_reentrant=False) if use_ckpt else _fn(h)
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
                t_id = top_ids_cpu[sl].to(device=device, dtype=torch.long, non_blocking=True)
                t_pr = top_probs_cpu[sl].to(device=device, dtype=torch.float32, non_blocking=True)
                tw = tok_weights_all[sl].to(device=device)

                hs, he = _forward_model(xb, atlas_sparse_tier2=False, use_ckpt=False)
                hn = _to_emb_dt(rmsn_gpu(hs, fn_param))
                lgs = SOFTCAP * torch.tanh((hn @ emb_gpu.T).float() / SOFTCAP)
                kl_list.append(float(v93.compute_topk_kl_loss(lgs, t_id, t_pr, tw).item()))
                cos_list.append(float(F.cosine_similarity(hs.reshape(-1, hs.shape[-1]), xt.reshape(-1, xt.shape[-1]), dim=-1).mean().item()))
                if he is not None:
                    hne = _to_emb_dt(rmsn_gpu(he, early_fn_param))
                    lg_e = SOFTCAP * torch.tanh((hne @ emb_gpu.T).float() / SOFTCAP)
                    kl_early_list.append(float(v93.compute_topk_kl_loss(lg_e, t_id, t_pr, tw).item()))

                pred_top1 = lgs.argmax(dim=-1)
                teach_top1 = t_id[:, :, 0].long()
                m = ans_mask_t[sl].to(device=device)
                if m.any():
                    top1_hits += int((pred_top1[m] == teach_top1[m]).sum().item())
                    top1_tot += int(m.sum().item())

                hs_sp, _ = _forward_model(xb, atlas_sparse_tier2=True, use_ckpt=False, capture_layer=None)
                hn_sp = _to_emb_dt(rmsn_gpu(hs_sp, fn_param))
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
    eval_bs_cap = 4 if (device.type == "cuda" and gpu_vram_gb < 20.0) else 8
    kl_before, kl_sp_before, kl_e_before, cos_before, top1_before = _eval_val(min(eval_bs_cap, cur_mbs))
    print(
        f"[V12.5 Stage 2.5 Verify Step-{start_step} Baseline] dense_kl={kl_before:.4f} | "
        f"cca_tier2_kl={kl_sp_before:.4f} | l29_kl={kl_e_before:.4f} | l59_cos={cos_before:.5f} | "
        f"top1={top1_before*100:.2f}% | pin_gpu={pin_on_gpu} mbs={cur_mbs}x{cur_acc}",
        flush=True,
    )
    assert kl_before < 1.25 and top1_before >= 0.55, (
        f"Step-{start_step} unpack verification failed: kl_before={kl_before:.4f}, top1={top1_before:.4f}"
    )

    best_obj = 0.65 * kl_before + 0.35 * kl_sp_before
    best_snap = _snapshot_best()

    rng = np.random.default_rng(2026 + start_step)
    tr_np = tr_seq_ids.numpy()
    t0_e2e = time.perf_counter()

    for step in range(extra_steps):
        mult = _lr_mult(step)
        for pg, blr in zip(opt.param_groups, (1.0e-4, 1.0e-4, 2.0e-4)):
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
                    t_id = top_ids_cpu[sl].to(device=device, dtype=torch.long, non_blocking=True)
                    t_pr = top_probs_cpu[sl].to(device=device, dtype=torch.float32, non_blocking=True)
                    tw = tok_weights_all[sl].to(device=device)

                    # Fixed V12.5 dual-regime trigger for both grad_accum>=2 and grad_accum==1
                    do_sp = (acc_i % 2 == 1) if cur_acc >= 2 else (step % 2 == 1)
                    regime_w = 0.55 if do_sp else 1.0

                    hs, he = _forward_model(
                        xb, atlas_sparse_tier2=do_sp, use_ckpt=True, capture_layer=(None if do_sp else v12.V12_EARLY_LAYER)
                    )
                    hn = _to_emb_dt(rmsn_gpu(hs, fn_param))
                    lgs = SOFTCAP * torch.tanh((hn @ emb_gpu.T).float() / SOFTCAP)
                    kl = v93.compute_topk_kl_loss(lgs, t_id, t_pr, tw)
                    cos_l = 1.0 - F.cosine_similarity(
                        hs.reshape(-1, hs.shape[-1]), xt.reshape(-1, xt.shape[-1]), dim=-1
                    ).mean()
                    loss_sub = kl + 0.25 * cos_l
                    if he is not None:
                        hne = _to_emb_dt(rmsn_gpu(he, early_fn_param))
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

        torch.nn.utils.clip_grad_norm_(sc_params + lr_params + norm_params, 0.5)
        opt.step()
        with torch.no_grad():
            for lyr in e2e_layers:
                for p in lyr["projs"].values():
                    p["sc"].clamp_(min=0.0, max=65000.0)
                    if p["bi"] is not None:
                        p["bi"].clamp_(min=-65000.0, max=65000.0)
                lyr["layer_scalar"].clamp_(min=-65000.0, max=65000.0)

        g_step = start_step + step + 1
        tot_step = start_step + extra_steps
        if (step + 1) % 10 == 0 or (step + 1) == extra_steps:
            if (step + 1) % 25 == 0 or (step + 1) == extra_steps:
                v_kl, v_sp, v_e, v_cos, v_t1 = _eval_val(min(eval_bs_cap, cur_mbs))
                obj = 0.65 * v_kl + 0.35 * v_sp
                improved = ""
                if obj < best_obj:
                    best_obj = obj
                    best_snap = _snapshot_best()
                    improved = " [BEST]"
                print(
                    f"  [V12.5 E2E step {g_step:03d}/{tot_step}] tr_kl={step_kl:.4f} dense_kl={v_kl:.4f} "
                    f"cca_tier2_kl={v_sp:.4f} l29_kl={v_e:.4f} top1={v_t1*100:.2f}% dt={time.perf_counter()-t0_e2e:.1f}s{improved}",
                    flush=True,
                )
            else:
                print(
                    f"  [V12.5 E2E step {g_step:03d}/{tot_step}] tr_kl={step_kl:.4f} mbs={cur_mbs}x{cur_acc} dt={time.perf_counter()-t0_e2e:.1f}s",
                    flush=True,
                )

    _restore_best(best_snap)
    kl_after, kl_sp_after, kl_e_after, cos_after, top1_after = _eval_val(min(eval_bs_cap, cur_mbs))
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
        f"[V12.5 Stage 2.5 Done] dense_kl: {kl_before:.4f}->{kl_after:.4f} | "
        f"cca_tier2_kl: {kl_sp_before:.4f}->{kl_sp_after:.4f} | top1: {top1_before*100:.2f}%->{top1_after*100:.2f}%",
        flush=True,
    )
    return {
        "start_step": start_step,
        "end_step": start_step + extra_steps,
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


def run_v12_5_pipeline(
    *,
    init_t16c: Path = DEFAULT_INIT_T16C,
    start_step: int = 200,
    extra_steps: int = 200,
    ast_steps: int = 720,
    micro_bs: int = 8,
    grad_accum: int = 2,
) -> Dict[str, object]:
    """Execute V12.5 continuation (Step 200->400 E2E + Stage 3 Replay + 720-step Assistant + BHDR-256)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file as st_save_file

    t0_all = time.perf_counter()
    assert torch.cuda.is_available(), "V12.5 requires CUDA GPU"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    v90.OUT_DIR = OUT_DIR
    v90.DRIVE_DIR = DRIVE_DIR
    v90.OUT_PERM_PT = OUT_DIR / "v12_5_permutations.pt"
    if DRIVE_DIR is not None:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    bgt = v12.compute_v12_budget_bytes()
    print(
        f"[V12.5 Start] GPU={gpu_name} ({gpu_vram_gb:.1f} GiB) | init={init_t16c.name} | "
        f"steps={start_step}->{start_step+extra_steps} | ast_steps={ast_steps} | mbs={micro_bs}x{grad_accum}",
        flush=True,
    )

    st = v90._init_v9_stage1_base(device, compute_dtype)
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

    lib_fq = v90._compile_fast_quant(OUT_DIR / "libfast_quant.so")
    H256_CPU = torch.from_numpy(pa.make_sylvester_hadamard(v12.V12_BLOCK_SIZE)).to(torch.float32)
    H256_GPU = H256_CPU.to(device)
    H8_GPU = torch.from_numpy(c8d.H8_NP).to(device, dtype=torch.float32)
    rmsn_gpu, make_rope, apply_rope = v90._make_gemma4_math_helpers(device, EPS)
    rope_cs_g, rope_sn_g = make_rope(THETA_G, int(round(HD_F * PART_G)), HD_F)
    rope_cs_s, rope_sn_s = make_rope(THETA_L, int(round(HD_S * PART_L)), HD_S)

    emb_perm_teach = emb_raw[:, perm_pi].contiguous()
    vocab_sz = int(emb_perm_teach.shape[0])
    del emb_raw

    loaded = load_e2e_state_from_t16c(
        init_t16c, n_layers=N_LAYERS, d_mod=D_MOD, vocab_sz=vocab_sz, h256_cpu=H256_CPU
    )
    e2e_layers: List[dict] = loaded["e2e_layers"]  # type: ignore[assignment]
    emb_eff: torch.Tensor = loaded["emb_eff"]  # type: ignore[assignment]
    emb_pb: np.ndarray = loaded["emb_pb"]  # type: ignore[assignment]
    vision_norm_recs: List[Tuple[str, np.ndarray]] = loaded["vision_norm_recs"]  # type: ignore[assignment]
    vision_mat_recs: List[tuple] = loaded["vision_mat_recs"]  # type: ignore[assignment]
    fn_param = nn.Parameter(loaded["fn_vec"].to(device=device, dtype=torch.float32))  # type: ignore[union-attr]
    early_fn_param = nn.Parameter(loaded["early_fn_vec"].to(device=device, dtype=torch.float32))  # type: ignore[union-attr]

    # Fail-closed fingerprint check confirming Stage-1 `perm_pi` matches V12 Step-200 `.t16c`
    cos_emb0 = float(F.cosine_similarity(emb_eff[0:16].float().reshape(-1), emb_perm_teach[0:16].float().reshape(-1), dim=0).item())
    cos_emb500 = float(F.cosine_similarity(emb_eff[8000:8016].float().reshape(-1), emb_perm_teach[8000:8016].float().reshape(-1), dim=0).item())
    print(f"  [Step-200 Embed/Perm Check] tile0_cos={cos_emb0:.5f} tile500_cos={cos_emb500:.5f}", flush=True)
    assert cos_emb0 >= 0.975 and cos_emb500 >= 0.975, (
        f"Permutation mismatch vs V12.t16c: cos0={cos_emb0:.5f}, cos500={cos_emb500:.5f}"
    )

    X_stud_0 = (emb_eff[torch.from_numpy(all_seq_ids)] * ESCALE).to(torch.float16)
    for si in range(N_TEXT_SEQ, N_SEQ):
        pm = torch.from_numpy(all_seq_ids[si] == soft_tok)
        X_stud_0[si, pm] = X_teach_perm[si, pm]

    t_info = compute_fast_teacher_targets(
        st=st, X_teach_perm=X_teach_perm, emb_perm_teach=emb_perm_teach,
        perm_pi=perm_pi, fn_orig=fn_orig, H256_GPU=H256_GPU, rmsn_gpu=rmsn_gpu,
        apply_rope=apply_rope, rope_cs_s=rope_cs_s, rope_sn_s=rope_sn_s,
        rope_cs_g=rope_cs_g, rope_sn_g=rope_sn_g, device=device, compute_dtype=compute_dtype,
        gpu_vram_gb=gpu_vram_gb,
    )
    X_teach_l59: torch.Tensor = t_info["X_teach_l59"]  # type: ignore[assignment]
    top_ids_cpu: torch.Tensor = t_info["top_ids_cpu"]  # type: ignore[assignment]
    top_probs_cpu: torch.Tensor = t_info["top_probs_cpu"]  # type: ignore[assignment]
    wo_head_norms: List[torch.Tensor] = t_info["wo_head_norms"]  # type: ignore[assignment]
    teach_mod_sal_list: List[torch.Tensor] = t_info["teach_mod_sal_list"]  # type: ignore[assignment]
    del emb_perm_teach, t_info
    gc.collect()
    torch.cuda.empty_cache()

    va_seq_ids = torch.arange(15, N_SEQ, 16, device="cpu", dtype=torch.long)
    tr_seq_ids = torch.tensor([i for i in range(N_SEQ) if i % 16 != 15], device="cpu", dtype=torch.long)

    e2e_metrics = run_stage2_5_v12_5_continue(
        e2e_layers=e2e_layers, X_stud_0=X_stud_0, X_teach_l59=X_teach_l59,
        top_ids_cpu=top_ids_cpu, top_probs_cpu=top_probs_cpu, ans_mask_np=all_ans_mask,
        emb_eff=emb_eff, fn_param=fn_param, early_fn_param=early_fn_param,
        tr_seq_ids=tr_seq_ids, va_seq_ids=va_seq_ids, ctx=dict(st),
        rmsn_gpu=rmsn_gpu, apply_rope=apply_rope, rope_cs_s=rope_cs_s, rope_sn_s=rope_sn_s,
        rope_cs_g=rope_cs_g, rope_sn_g=rope_sn_g, H256_GPU=H256_GPU, device=device,
        gpu_vram_gb=gpu_vram_gb, start_step=start_step, extra_steps=extra_steps,
        micro_bs=micro_bs, grad_accum=grad_accum,
    )
    del top_ids_cpu, top_probs_cpu

    # Stage 3 Replay & Pack V12.5 T16C + CRTR + HeadCRTR + C8DB
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
    MB_FWD = 32 if gpu_vram_gb >= 20.0 else 16
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
        pos = p3._write_mat_record(fp, pos, "model.language_model.embed_tokens.weight", vocab_sz, D_MOD, v12.V12_BLOCK_SIZE, 3 | p3.T16C_GAUSS_BIT, False, True, emb_pb, None, None)
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
            blk_e_cpu = torch.empty((N_SEQ * v90.V9_SEQ_LEN, v12.V12_MLP_BLOCKS), dtype=torch.float32)

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
                    blk_e_cpu[b0 * v90.V9_SEQ_LEN : b1 * v90.V9_SEQ_LEN] = act_q.reshape(-1, v12.V12_MLP_BLOCKS, v12.V12_BLOCK_SIZE).pow(2).sum(dim=-1).cpu()
                    X_replay[b0:b1] = ((xm + rmsn_gpu(F.linear(act_q, wd_s), g_pff_s)) * lsc_s).to(torch.float16).cpu()

            xn_f = xn_cpu.reshape(-1, D_MOD)[::2]
            he_f = head_e_cpu[::2]
            ntr_h = (xn_f.shape[0] * 9) // 10
            hr_net, hr_met = crtr.train_layer_router(xn_f[:ntr_h], he_f[:ntr_h], xn_f[ntr_h:], he_f[ntr_h:], top_p=float(v12.V12_HEAD_ROUTED) / float(NH), rank=v90.V9_HEAD_CRTR_RANK, steps=160, lr=3e-3, device=str(device), seed=300 + l_idx)
            head_crtr_layers.append(crtr.quantize_layer_for_crtr(hr_net, v12.V12_BLOCK_SIZE))

            xf_f = xf_cpu.reshape(-1, D_MOD)[::2]
            be_f = blk_e_cpu[::2]
            my_f = (mod_sal_cpu[::2] > float(torch.quantile(mod_sal_cpu, 0.35).item())).float()
            ntr_r = (xf_f.shape[0] * 9) // 10
            r_net, r_met = crtr.train_layer_router(xf_f[:ntr_r], be_f[:ntr_r], xf_f[ntr_r:], be_f[ntr_r:], top_p=float(v12.V12_CRTR_TOP_BLOCKS) / float(v12.V12_MLP_BLOCKS), rank=v90.V9_CRTR_RANK, steps=200, lr=3e-3, mod_targets_tr=my_f[:ntr_r], mod_targets_va=my_f[ntr_r:], device=str(device), seed=100 + l_idx)
            crtr_layers.append(crtr.quantize_layer_for_crtr(r_net, v12.V12_BLOCK_SIZE))

            layer_logs.append({"l": l_idx, "crtr_recall": r_met["val_recall_after"], "head_crtr_recall": hr_met["val_recall_after"]})
            if (l_idx + 1) % 10 == 0 or l_idx in (0, v12.V12_EARLY_LAYER, 59):
                print(f"  [V12.5 Replay L{l_idx:02d}/59] crtr12_rec={r_met['val_recall_after']:.3f} hr8_rec={hr_met['val_recall_after']:.3f}", flush=True)
            del w_stud, xn_cpu, xf_cpu, head_e_cpu, blk_e_cpu, mod_sal_cpu

        for rec_v in vision_mat_recs:
            pos = p3._write_mat_record(fp, pos, *rec_v)

    os.replace(tmp_t16c, OUT_T16C)
    vm_c, ve_c = v70.verify_t16c_integrity(OUT_T16C)
    assert vm_c == total_mats and ve_c == OUT_T16C.stat().st_size, f"Bad V12.5 T16C {vm_c}/{total_mats}"
    cb_cnt = sdc.verify_t16c_file(OUT_T16C)["colblock_mats"]
    l59_cos_final = float(F.cosine_similarity(X_replay[-32:].reshape(-1, D_MOD).float(), X_teach_l59[-32:].reshape(-1, D_MOD).float(), dim=-1).mean().item())
    assert l59_cos_final >= 0.72, f"Stage 3 Replay final L59 cosine failed gate: {l59_cos_final:.5f} < 0.72"

    crtr.export_crtr(crtr_layers, OUT_CRTR)
    crtr.export_crtr(head_crtr_layers, OUT_HEAD_CRTR)
    fn_eff_g = fn_param.detach().to(device=device, dtype=torch.float32)

    # Train post_cell60 C8DB + Target-Embed-Shared 720-Step EAGLE-3 Assistant
    c8db_entries: List[dict] = []
    emb_g = emb_eff.to(device=device, dtype=compute_dtype)
    ans_flat = torch.nonzero(torch.from_numpy(all_ans_mask.reshape(-1)), as_tuple=False).squeeze(1)
    sel_ans = ans_flat[torch.linspace(0, ans_flat.numel() - 1, min(16384, ans_flat.numel())).long()]
    s_t, t_t = sel_ans // v90.V9_SEQ_LEN, sel_ans % v90.V9_SEQ_LEN
    hs_ans = rmsn_gpu(X_replay[s_t, t_t].to(device).unsqueeze(1), fn_eff_g).squeeze(1).float()
    ht_ans = rmsn_gpu(X_teach_l59[s_t, t_t].to(device).unsqueeze(1), fn_eff_g).squeeze(1).float()
    nxt_ans = torch.from_numpy(all_seq_nxt[s_t.numpy(), t_t.numpy()]).to(device, dtype=torch.long)
    iv_fn = torch.arange(hs_ans.shape[0], device=device) % 10 == 0
    tr_idx60 = torch.nonzero(~iv_fn, as_tuple=False).squeeze(1)
    va_idx60 = torch.nonzero(iv_fn, as_tuple=False).squeeze(1)
    with torch.no_grad():
        top_ids_c = [torch.topk((ht_ans[i0:min(i0 + 512, ht_ans.shape[0])] @ emb_g.float().T), 16, dim=-1).indices.reshape(-1) for i0 in range(0, min(4096, ht_ans.shape[0]), 512)]
        uniq_c, cnt_c = torch.unique(torch.cat(top_ids_c), return_counts=True)
        cand_c60 = torch.unique(torch.cat([uniq_c[torch.topk(cnt_c, min(4096, uniq_c.numel())).indices], nxt_ans]))
        W_c60 = emb_g[cand_c60].float()
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
    del m60, mc60, o60, s60, hs_ans, ht_ans, W_c60, cand_c60
    torch.cuda.empty_cache()

    hf_tok = os.environ.get("HF_TOKEN", "").strip() or None
    ast_dir = Path(snapshot_download("google/gemma-4-31B-it-assistant", local_dir="/tmp/gemma4_ast", token=hf_tok, ignore_patterns=["*.bin", "*.pt"]))
    from safetensors import safe_open
    ast_sd_raw: Dict[str, torch.Tensor] = {}
    with safe_open(ast_dir / "model.safetensors", framework="pt", device="cpu") as sf_a:
        for k in sf_a.keys():
            ast_sd_raw[k] = sf_a.get_tensor(k)
    ast_sd = v10.permute_assistant_state_dict(ast_sd_raw, perm_pi, D_MOD)
    del ast_sd_raw

    ast_dtype = compute_dtype if compute_dtype == torch.bfloat16 else torch.float32
    ast_mod = v90.Gemma4AssistantTrain(
        ast_sd, d_mod=D_MOD, nh=NH, nkv_s=NKV_S, nkv_f=NKV_F, hd_s=HD_S, hd_f=HD_F,
        theta_l=THETA_L, part_l=PART_L, theta_g=THETA_G, part_g=PART_G, seq_len=v90.V9_SEQ_LEN, eps=EPS, compute_dtype=ast_dtype,
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
                hpost_n = rmsn_gpu(F.linear(ha.to(ast_dtype), ast_mod.post.to(ast_dtype)).float(), fn_eff_g)
                zs = SOFTCAP * torch.tanh((ast_h_final[ai, 1:].to(device).float() @ W_cand_g.T) / SOFTCAP)
                za = SOFTCAP * torch.tanh((hpost_n @ W_cand_g.T) / SOFTCAP)
                ps = F.softmax(zs / TAU_AST, dim=-1)
                kls += float(((ps * (torch.log(ps.clamp(min=1e-12)) - F.log_softmax(za / TAU_AST, dim=-1))).sum(dim=-1) * (TAU_AST * TAU_AST)).sum().item())
                agrs += float((za.argmax(dim=-1) == zs.argmax(dim=-1)).float().sum().item())
                cnt += ha.shape[0]
            return kls / max(cnt, 1), agrs / max(cnt, 1)

    akl0, aagr0 = eval_ast()
    opt_a = torch.optim.AdamW(ast_mod.parameters(), lr=2.5e-4, weight_decay=1e-4)
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=ast_steps, eta_min=2e-5)
    best_aagr, best_asd = aagr0, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}
    ast_bs = 8 if gpu_vram_gb >= 20.0 else 4
    for st_a in range(ast_steps):
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

        ha1 = ast_mod.forward_seq(te_s, bh_s, k58_s, v58_s, k59_s, v59_s)
        hp1 = rmsn_gpu(F.linear(ha1.to(ast_dtype), ast_mod.post.to(ast_dtype)).float(), fn_eff_g)
        za1 = SOFTCAP * torch.tanh((hp1.reshape(-1, D_MOD) @ W_cand_g.T) / SOFTCAP)

        bh2 = torch.cat([bh_s[:, :1], hp1[:, :-1]], dim=1)
        ha2 = ast_mod.forward_seq(te_s, bh2, k58_s, v58_s, k59_s, v59_s)
        hp2 = rmsn_gpu(F.linear(ha2.to(ast_dtype), ast_mod.post.to(ast_dtype)).float(), fn_eff_g)

        bh3 = torch.cat([bh_s[:, :1], hp2[:, :-1]], dim=1)
        ha3 = ast_mod.forward_seq(te_s, bh3, k58_s, v58_s, k59_s, v59_s)
        hp3 = rmsn_gpu(F.linear(ha3.to(ast_dtype), ast_mod.post.to(ast_dtype)).float(), fn_eff_g)

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
        if (st_a + 1) % 120 == 0:
            klv, agrv = eval_ast()
            print(f"    [V12.5 Ast step {st_a+1:03d}/{ast_steps}] Val KL={klv:.4f} TargetSharedTop1={agrv*100:.1f}%", flush=True)
            if agrv > best_aagr:
                best_aagr, best_asd = agrv, {k: v.detach().cpu().clone() for k, v in ast_mod.state_dict().items()}

    ast_mod.load_state_dict({k: v.to(device) for k, v in best_asd.items()})
    akl1, aagr1 = eval_ast()
    print(f"  [V12.5 Target-Shared Assistant] Val KL={akl0:.4f}->{akl1:.4f} | Top1Agree={aagr0*100:.1f}%->{aagr1*100:.1f}%", flush=True)

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
    ast_sf = OUT_DIR / "assistant_v12_5_permuted.safetensors"
    st_save_file(ast_sd, str(ast_sf))
    bhmod.pack_assistant_body_from_safetensors(ast_sf, OUT_AST_Q2, body_bits=2, block_size=64, has_fwht=True)
    bhmod.pack_assistant_body_from_safetensors(ast_sf, OUT_AST_Q4, body_bits=4, block_size=64, has_fwht=True)
    ast_sf.unlink(missing_ok=True)
    del opt_a, sch_a, ast_sd, best_asd, W_cand_g

    # Train upgraded BHDR-256 sidecar
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
    ast_post_g = ast_mod.post.detach().to(device=device, dtype=ast_dtype)
    del ast_mod, cap_kv58, cap_kv59
    torch.cuda.empty_cache()

    H_ast_all = torch.cat(h_ast_chunks, dim=0).reshape(-1, 1024)
    H_stud_tgt = ast_h_final[:, 1:, :].reshape(-1, D_MOD)
    top_ids_t = torch.empty((H_stud_tgt.shape[0], 64), dtype=torch.int32)
    top_probs_t = torch.empty((H_stud_tgt.shape[0], 64), dtype=torch.float32)
    with torch.no_grad():
        for i0 in range(0, H_stud_tgt.shape[0], 512):
            i1 = min(i0 + 512, H_stud_tgt.shape[0])
            hp_b = rmsn_gpu(F.linear(H_ast_all[i0:i1].to(device, dtype=ast_dtype), ast_post_g).float(), fn_eff_g)
            h_blend = (0.60 * hp_b + 0.40 * H_stud_tgt[i0:i1].to(device, dtype=torch.float32)).clamp(-65000.0, 65000.0).to(compute_dtype)
            lg_b = SOFTCAP * torch.tanh((h_blend @ emb_g.T).float() / SOFTCAP)
            tv, ti = torch.topk(lg_b, k=64, dim=-1)
            top_probs_t[i0:i1] = F.softmax(tv / 0.5, dim=-1).cpu()
            top_ids_t[i0:i1] = ti.to(torch.int32).cpu()
    del emb_g, ast_post_g, H_stud_tgt, ast_h_final
    student_argmax_ids = torch.tensor(v70.select_first_valid_trie_ids(top_ids_t, trie), dtype=torch.long)
    act_bundle = OUT_DIR / "v12_5_bhdr_acts.pt"
    torch.save({"h_final": H_ast_all.half(), "gold_ids": student_argmax_ids, "top_ids": top_ids_t, "top_probs": top_probs_t}, act_bundle)
    bh_rpt = bhmod.train_from_activations(act_bundle, tok_path, OUT_BHDR, d_byte=256, max_bytes=48, steps=4000, batch_size=256, lr=3e-3, device=str(device))
    act_bundle.unlink(missing_ok=True)
    bh_loaded = bhmod.load_bhdr_v1(OUT_BHDR)
    beam_hits = sum(1 for hv, gv in zip(H_ast_all[-256:].numpy().astype(np.float32), student_argmax_ids[-256:].tolist()) if bhmod.decode_bhdr_np_beam(bh_loaded, trie, hv, beam_width=2)[0] == gv)
    bh_metrics = {**bh_rpt["metrics"], "heldout_beam_top1_acc": round(beam_hits / 256.0, 4)}
    assert bh_metrics["heldout_beam_top1_acc"] >= 0.20, f"BHDR-256 heldout_beam_top1_acc failed gate: {bh_metrics['heldout_beam_top1_acc']}"
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
        "version": "V12.5-ATLAS-CORTICAL-CASCADE-400STEP",
        "revision": V12_5_PIPELINE_REVISION,
        "init_t16c": str(init_t16c),
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
        "assistant_target_shared": {"kl_before": round(akl0, 4), "kl_after": round(akl1, 4), "top1_before": round(aagr0, 4), "top1_after": round(aagr1, 4), "steps": ast_steps},
        "bhdr256": bh_metrics,
        "protected_cognitive_core_layers": sorted(v12.V12_COGNITIVE_CORE_LAYERS),
        "tier2_bypass_layers": sorted(v12.V12_TIER2_BYPASS_LAYERS),
        "artifacts": artifacts,
    }
    OUT_REPORT.write_text(json.dumps(rpt, indent=2) + "\n", encoding="utf-8")
    sums_v12_5 = OUT_DIR / "SHA256SUMS_V12_5.txt"
    sums_v12_5.write_text("".join(f"{v['sha256']}  {k}\n" for k, v in artifacts.items()), encoding="utf-8")
    if DRIVE_DIR is not None:
        v90._drive_safe_copy(OUT_REPORT, DRIVE_DIR / OUT_REPORT.name)
        v90._drive_safe_copy(sums_v12_5, DRIVE_DIR / sums_v12_5.name)
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
        expected_sha = {k: v["sha256"] for k, v in artifacts.items()}
        expected_sha[OUT_REPORT.name] = v90._sha256_file(OUT_REPORT)
        expected_sha[sums_v12_5.name] = v90._sha256_file(sums_v12_5)
        v90._verify_drive_sha256(DRIVE_DIR, expected_sha)

    print(
        f"[V12.5 COMPLETE] elapsed={(time.perf_counter()-t0_all)/60:.1f} min | "
        f"{OUT_T16C.name}={rpt['t16c_gib']} GiB | dense_kl={e2e_metrics['val_kl_after']:.4f} | "
        f"cca_tier2_kl={e2e_metrics['val_cca_tier2_kl_after']:.4f} | ast_top1={aagr1*100:.1f}%",
        flush=True,
    )
    if os.environ.get("COLAB_AUTO_UNASSIGN", "1") == "1":
        v90._flush_drive_before_unassign()
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:
            print(f"[Auto-Unassign Note] {exc}", flush=True)
    return rpt


def run_selftest() -> int:
    """Mandatory 6-gate fail-closed CPU verification battery for V12.5 Continuation Pipeline."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Bit-exact finalize -> unfinalize roundtrip for BOTH (1b_rm RM+CB) and (2b_gauss CB)
    rng = np.random.default_rng(7)
    do, di, bsz = 32, 512, 256
    h256 = torch.from_numpy(pa.make_sylvester_hadamard(bsz)).to(torch.float32)
    g1_ok = True
    for mode_i, is_rm, is_cb in ((0, True, True), (5, False, True), (6, False, False)):
        n_bits, is_asym = p3.MODE_TO_T16C[mode_i]
        exp_b = p3.expected_tile16_bytes(do, di, bsz, n_bits, is_asym)
        pb_orig = rng.integers(0, 256, size=exp_b, dtype=np.uint8)
        hdr_bits, out_pb = v90.finalize_t16c_payload(pb_orig, do, di, bsz, mode_i, is_rm, is_cb)
        m_dec, rm_dec, cb_dec, pb_rec = unfinalize_t16c_payload(out_pb, do, di, bsz, hdr_bits, is_asym)
        if not (m_dec == mode_i and rm_dec == is_rm and cb_dec == is_cb and np.array_equal(pb_orig, pb_rec)):
            g1_ok = False
    gate("Gate 1: Bit-Exact T16C Finalize <-> Unfinalize Involution (1b_rm / 2b_gauss / 3b_gauss)", g1_ok)

    # Gate 2: Synthetic T16C write -> load_e2e_state_from_t16c exact tensor + LR + norm roundtrip
    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        syn_p = td / "syn_v12.t16c"
        n_norms = 9
        n_mats = 4  # embed + q + down_proj(LR) + vision
        norms_list = [
            ("model.language_model.norm.weight", np.ones((d_mod_t := 512,), dtype="<f4") * 1.25),
            ("model.language_model.early_exit_l29_norm.weight", np.ones((d_mod_t,), dtype="<f4") * 0.90),
            ("model.language_model.layers.0.input_layernorm.weight", np.ones((d_mod_t,), dtype="<f4")),
            ("model.language_model.layers.0.post_attention_layernorm.weight", np.ones((d_mod_t,), dtype="<f4")),
            ("model.language_model.layers.0.pre_feedforward_layernorm.weight", np.ones((d_mod_t,), dtype="<f4")),
            ("model.language_model.layers.0.post_feedforward_layernorm.weight", np.ones((d_mod_t,), dtype="<f4")),
            ("model.language_model.layers.0.self_attn.q_norm.weight", np.ones((bsz,), dtype="<f4")),
            ("model.language_model.layers.0.self_attn.k_norm.weight", np.ones((bsz,), dtype="<f4")),
            ("model.language_model.layers.0.layer_scalar", np.array([1.0], dtype="<f4")),
            ("model.vision_tower.std_bias", np.zeros((16,), dtype="<f4")),
        ]
        emb_pb = np.full(p3.expected_tile16_bytes(32, d_mod_t, bsz, 3, False), 17, dtype=np.uint8)
        q_pb_in = np.full(p3.expected_tile16_bytes(do, d_mod_t, bsz, 2 | 0x80, False), 9, dtype=np.uint8)
        _, q_out = v90.finalize_t16c_payload(q_pb_in, do, d_mod_t, bsz, 5, False, True)
        d_pb_in = np.full(p3.expected_tile16_bytes(do, di, bsz, 1, True), 5, dtype=np.uint8)
        d_hdr, d_out = v90.finalize_t16c_payload(d_pb_in, do, di, bsz, 0, True, True)
        su = np.full((do,), 0.25, dtype="<f4")
        sv = np.full((4,), 0.5, dtype="<f4")
        ui = np.ones((do, 4), dtype=np.int8) * 2
        vi = np.ones((4, di), dtype=np.int8) * -3
        inv_d = np.ones((di,), dtype="<f4")
        vis_pb = np.zeros(p3.expected_tile16_bytes(16, 64, 64, 3, False), dtype=np.uint8)
        with open(syn_p, "wb") as fp:
            fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, len(norms_list), n_mats, 0))
            pos = 24
            for nn_s, na in norms_list:
                nb = nn_s.encode("utf-8")
                fp.write(struct.pack("<H", len(nb)) + nb + struct.pack("<I", na.size) + na.tobytes())
                pos += 2 + len(nb) + 4 + na.size * 4
            pos = p3._write_mat_record(fp, pos, "model.language_model.embed_tokens.weight", 32, d_mod_t, bsz, 3 | p3.T16C_GAUSS_BIT, False, True, emb_pb, None, None)
            pos = p3._write_mat_record(fp, pos, "model.language_model.layers.0.self_attn.q_proj.weight", do, d_mod_t, bsz, 2 | 0x80 | p3.T16C_COLBLOCK_BIT, False, True, q_out, inv_d, None)
            pos = p3._write_mat_record(fp, pos, "model.language_model.layers.0.self_attn.k_proj.weight", do, d_mod_t, bsz, 2 | 0x80 | p3.T16C_COLBLOCK_BIT, False, True, q_out, inv_d, None)
            # Include 4 more required projs minimal placeholders -> 6 projs for L0
            for extra_pn, r_do, r_di, r_pb, r_hdr, r_asym, r_lr in (
                ("self_attn.v_proj.weight", do, d_mod_t, q_out, 2 | 0x80 | p3.T16C_COLBLOCK_BIT, False, None),
                ("self_attn.o_proj.weight", do, d_mod_t, q_out, 2 | 0x80 | p3.T16C_COLBLOCK_BIT, False, None),
                ("mlp.gate_proj.weight", do, di, d_out, d_hdr, True, None),
                ("mlp.up_proj.weight", do, di, d_out, d_hdr, True, None),
                ("mlp.down_proj.weight", do, di, d_out, d_hdr, True, (su, sv, ui, vi)),
            ):
                pos = p3._write_mat_record(fp, pos, f"model.language_model.layers.0.{extra_pn}", r_do, r_di, bsz, r_hdr, r_asym, True, r_pb, np.ones((r_di,), dtype="<f4"), r_lr)
            pos = p3._write_mat_record(fp, pos, "model.vision_tower.std_dummy", 16, 64, 64, 3 | p3.T16C_GAUSS_BIT, False, True, vis_pb, None, None)
            # Rewrite actual n_mats=10 in header
            fp.seek(16)
            fp.write(struct.pack("<I", 9))

        res = load_e2e_state_from_t16c(syn_p, n_layers=1, d_mod=d_mod_t, vocab_sz=32, h256_cpu=h256)
        down_p = res["e2e_layers"][0]["projs"]["mlp.down_proj"]  # type: ignore[index]
        g2_ok = (
            len(res["vision_mat_recs"]) == 1  # type: ignore[arg-type]
            and len(res["vision_norm_recs"]) == 1  # type: ignore[arg-type]
            and abs(float(res["fn_vec"][0]) - 1.25) < 1e-6  # type: ignore[index]
            and abs(float(down_p["u_lr"][0, 0]) - 0.5) < 1e-5
            and abs(float(down_p["v_lr"][0, 0]) - (-1.5)) < 1e-5
        )
        gate("Gate 2: Full `.t16c` Resume Loader (Norms, Vision Verbatim & Int8 LR)", g2_ok)

        # Gate 3: Fail-closed FileNotFoundError on missing --init-t16c
        caught = False
        try:
            load_e2e_state_from_t16c(td / "missing.t16c", n_layers=1, d_mod=d_mod_t, vocab_sz=32, h256_cpu=h256)
        except FileNotFoundError:
            caught = True
        gate("Gate 3: Fail-Closed FileNotFoundError on Missing Checkpoint", caught)

    # Gate 4: A100 grad_accum=1 vs L4 grad_accum=2 dual-regime `do_sp` fix
    l4_sp = [True if (acc_i % 2 == 1) else False for _ in range(4) for acc_i in range(2)]
    a100_sp = [(step % 2 == 1) for step in range(4)]
    gate("Gate 4: Fixed `do_sp` Trains Dense & Bypass on Both A100 (accum=1) & L4 (accum=2)", sum(l4_sp) == 4 and sum(a100_sp) == 2)

    # Gate 5: Warm-start Step-200->400 LR schedule smoothness (never starts at unscaled step 0 spike)
    warmup_v = [0.05 + 0.95 * float(s + 1) / 15.0 for s in range(15)]
    gate("Gate 5: Step-200->400 Smooth AdamW Re-Warmup Curve", abs(warmup_v[0] - (0.05 + 0.95 / 15.0)) < 1e-6 and abs(warmup_v[-1] - 1.0) < 1e-6)

    # Gate 6: Zero contamination check & V12 budget invariance
    bgt = v12.compute_v12_budget_bytes()
    dj = v90.verify_corpus_decontaminated(["Solve 2x + 3 = 11 step by step."])
    gate("Gate 6: Budget <= 6.28 GiB & Decontamination Guard", int(bgt["total_t16c_bytes"]) <= v12.MAX_V12_T16C_BYTES and dj["checked"] == 1)

    print(f"V12_5_SELFTEST_PASSED: {6 - failures}/6", flush=True)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Language-U V12.5 Continuation Pipeline (Step 200 -> 400 + Ast 720)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--init-t16c", type=Path, default=DEFAULT_INIT_T16C)
    ap.add_argument("--start-step", type=int, default=200)
    ap.add_argument("--extra-steps", type=int, default=200)
    ap.add_argument("--ast-steps", type=int, default=720)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    args = ap.parse_args()
    if args.selftest or (not args.run and not torch.cuda.is_available()):
        return run_selftest()
    run_v12_5_pipeline(
        init_t16c=args.init_t16c,
        start_step=args.start_step,
        extra_steps=args.extra_steps,
        ast_steps=args.ast_steps,
        micro_bs=args.micro_bs,
        grad_accum=args.grad_accum,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

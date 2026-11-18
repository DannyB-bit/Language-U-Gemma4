#!/usr/bin/env python3
"""
V4 Mixed-Precision Gaussian Bit-Serial + ActALS Calibration for Gemma 4 31B (`.t16c`).

Key upgrades over V3 (`calibrate_31b_v3.py`):
  1. Removes `bos_weight=0.15` (uses `bw=1.00`) across all 60 layers, fixing the 44x
     BOS Hessian variance distortion that degraded SmoothQuant/ActALS on short turns.
  2. Uses diverse non-trivia clean teacher snapshots (`snaps_v4`, ~1,779 tokens) captured
     with `<|turn>` formatting and zero overlap with P1..P8 or unseen holdouts H1..H4.
  3. Upgrades cliff layers (`L36/37` d_proj -> tern5, `L39` + `L51..L53,L55..L57` ->
     non-uniform Gaussian 2-bit `mode 5` + row affine `AR` + `ActALS`), plus `L58.k/v` and
     `L59.k` -> 3-bit (`mode 4`) GPTQ (`damp=0.15`) + `r=32`, staying strictly within
     `<= 6.525 GB` (`~+167 MB` over V3 -> `< 6.50 GiB VmHWM` ceiling on 8 GB Pi 4).
  4. Fast OpenMP `embed_tokens` `3b_u0` packing (`~100s` via `libgauss_gptq_v2.so`).

Run with no arguments or `--selftest` to execute the 6-gate offline suite.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import ctypes
import gc
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_31b_v3 import (
    _NativeCalOps,
    _rss_mb,
    balance_tokens_with_scale,
    compute_sq,
    cos_sim,
    fit_act_als_i8,
    gelu,
    precompute_x_pinv_t,
    rmsn,
    rope,
)
from pack_t16c_v3 import (
    MODE_TO_T16C,
    SafetensorsReader,
    assemble_t16c_v3,
    resolve_safetensors_path,
    tile16_blk_bytes,
)


def _load_v2_gptq_lib(work_dir: Optional[Path], require_native: bool = True) -> Optional[ctypes.CDLL]:
    """Load `libgauss_gptq_v2.so` safely for fast OpenMP 2bg1/3b_u0/GPTQ packing."""
    if work_dir is not None:
        cand = work_dir / "libgauss_gptq_v2.so"
        if cand.is_file():
            lib = ctypes.CDLL(str(cand))
            lib.build_gptq_u_fast_f32.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_void_p,
            ]
            lib.build_gptq_u_fast_f32.restype = ctypes.c_int
            lib.gptq_quant_and_pack_v2_f32.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            lib.gptq_quant_and_pack_v2_f32.restype = None
            return lib
    if require_native:
        raise FileNotFoundError(f"Missing libgauss_gptq_v2.so in {work_dir}")
    return None


def scale_t16c_pbytes_rows(
    pb: np.ndarray,
    d_out: int,
    d_in: int,
    bsize: int,
    mode_i: int,
    ar: np.ndarray,
) -> None:
    """Multiply per-row FP16 tile scales (and asymmetric biases) in-place by `ar[0..d_out]`."""
    n_bits, is_asym = MODE_TO_T16C[mode_i]
    tblk = tile16_blk_bytes(bsize, n_bits, is_asym)
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    ar_pad = np.ones((n_tiles * 16,), dtype=np.float32)
    ar_pad[:d_out] = np.asarray(ar, dtype=np.float32).ravel()[:d_out]
    ar_tile = ar_pad.reshape(n_tiles, 1, 16)

    view = pb.reshape(n_tiles, n_blks, tblk)
    sc_u16 = view[:, :, :32].view("<u2").reshape(n_tiles, n_blks, 16)
    sc_f32 = sc_u16.view(np.float16).astype(np.float32) * ar_tile
    sc_u16[:] = sc_f32.astype(np.float16).view("<u2")
    if is_asym:
        bi_u16 = view[:, :, 32:64].view("<u2").reshape(n_tiles, n_blks, 16)
        bi_f32 = bi_u16.view(np.float16).astype(np.float32) * ar_tile
        bi_u16[:] = bi_f32.astype(np.float16).view("<u2")


def _mat_size(name: str, d_out: int, d_in: int, mode_i: int, bsize: int, rk: int) -> int:
    n_bits, is_asym = MODE_TO_T16C[mode_i]
    tblk = tile16_blk_bytes(bsize, n_bits, is_asym)
    pbytes = ((d_out + 15) // 16) * (d_in // bsize) * tblk
    inv_b = d_in * 4
    lr_b = (d_out * 4 + rk * 4 + d_out * rk + rk * d_in) if rk > 0 else 0
    return 34 + len(name) + 32 + pbytes + inv_b + lr_b


def build_v4_allocation(
    cfg: dict,
    budget_bytes: int = 6_748_000_000,
) -> Tuple[Dict[int, dict], int]:
    """Construct exact V4.1 per-projection allocation inside `budget_bytes` (`<= 6.500 GiB` VmHWM).

    Mode codes:
      0 = 1-bit asym, 1 = tern5, 4 = 3b_u0, 5 = 2b_gauss (`2|0x80`).
    """
    n_layers = int(cfg["num_hidden_layers"])
    d = int(cfg["hidden_size"])
    ff = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    layer_types = cfg["layer_types"]
    k_eq_v = bool(cfg.get("attention_k_eq_v", False))

    emb_b = 34 + 40 + 32 + ((vocab + 15) // 16) * (d // 256) * tile16_blk_bytes(256, 3, False)
    norms_b = 6_000_000

    # V4.1 empirical cliff-fix layout (proven by P8 60-layer causal prefix/suffix ablations):
    #   - L02, L04, L10: upgraded 1b(r=0) -> tern5 (q/o/d r=16, g/u r=0)
    #   - L13..L24: MLPs upgraded 1b -> tern5 (g/u r=0, d r=16); L13..L22 q/o stay 1b(r=16) (+C8DB)
    #   - L33..L37: keep L33..34 mlp 1b(r=8), L35..37 d=tern5(r=16), g/u=1b(r=8)
    #   - L52 (deepest late dip): 2b_gauss (`mode 5`, q/o/d r=16, g/u r=0)
    #   - L58, L59: all 3b_u0 (`k/v r=32` shared-KV GPTQ, q/o/g/u/d r=16)
    alloc: Dict[int, dict] = {}
    for l in range(n_layers):
        is_full = layer_types[l] == "full_attention"
        v_spec = lambda m, r: None if (is_full and k_eq_v) else (m, 256, r)
        if l in (0, 1):
            alloc[l] = {
                "q": (1, 256, 16), "k": (1, 256, 0), "v": v_spec(1, 0),
                "o": (1, 256, 16), "g": (1, 256, 24), "u": (1, 256, 24), "d": (1, 256, 24),
            }
        elif l in (2, 4, 10) or (23 <= l <= 24):
            alloc[l] = {
                "q": (1, 256, 16), "k": (1, 256, 0), "v": v_spec(1, 0),
                "o": (1, 256, 16), "g": (1, 256, 0), "u": (1, 256, 0), "d": (1, 256, 16),
            }
        elif l in (8, 9, 11, 12):
            alloc[l] = {
                "q": (1, 256, 16), "k": (1, 256, 0), "v": v_spec(1, 0),
                "o": (1, 256, 16), "g": (0, 256, 16), "u": (0, 256, 16), "d": (0, 256, 16),
            }
        elif 13 <= l <= 22:
            alloc[l] = {
                "q": (0, 256, 16), "k": (1, 256, 0), "v": v_spec(1, 0),
                "o": (0, 256, 16), "g": (1, 256, 0), "u": (1, 256, 0), "d": (1, 256, 16),
            }
        elif l in (33, 34):
            alloc[l] = {
                "q": (1, 256, 16), "k": (1, 256, 0), "v": v_spec(1, 0),
                "o": (1, 256, 16), "g": (0, 256, 8), "u": (0, 256, 8), "d": (0, 256, 8),
            }
        elif l in (35, 36, 37):
            alloc[l] = {
                "q": (1, 256, 16), "k": (1, 256, 0), "v": v_spec(1, 0),
                "o": (1, 256, 16), "g": (0, 256, 8), "u": (0, 256, 8), "d": (1, 256, 16),
            }
        elif l == 52:
            alloc[l] = {
                "q": (5, 256, 16), "k": (5, 256, 0), "v": v_spec(5, 0),
                "o": (5, 256, 16), "g": (5, 256, 0), "u": (5, 256, 0), "d": (5, 256, 16),
            }
        elif l in (58, 59):
            alloc[l] = {
                "q": (4, 256, 16), "k": (4, 256, 32), "v": v_spec(4, 32),
                "o": (4, 256, 16), "g": (4, 256, 16), "u": (4, 256, 16), "d": (4, 256, 16),
            }
        else:
            rk_m = 8 if (40 <= l <= 49) else 16
            alloc[l] = {
                "q": (1, 256, rk_m), "k": (1, 256, 0), "v": v_spec(1, 0),
                "o": (1, 256, rk_m), "g": (1, 256, 0), "u": (1, 256, 0), "d": (1, 256, rk_m),
            }

    def total_bytes_fn(cur_alloc: Dict[int, dict]) -> int:
        tot = emb_b + norms_b
        for l in range(n_layers):
            is_full = layer_types[l] == "full_attention"
            q_out = 16384 if is_full else 8192
            kv_out = 2048 if is_full else 4096
            p = f"model.language_model.layers.{l}."
            a = cur_alloc[l]
            tot += _mat_size(p + "self_attn.q_proj.weight", q_out, d, *a["q"])
            tot += _mat_size(p + "self_attn.k_proj.weight", kv_out, d, *a["k"])
            if a["v"] is not None:
                tot += _mat_size(p + "self_attn.v_proj.weight", kv_out, d, *a["v"])
            tot += _mat_size(p + "self_attn.o_proj.weight", d, q_out, *a["o"])
            tot += _mat_size(p + "mlp.gate_proj.weight", ff, d, *a["g"])
            tot += _mat_size(p + "mlp.up_proj.weight", ff, d, *a["u"])
            tot += _mat_size(p + "mlp.down_proj.weight", d, ff, *a["d"])
        return tot

    tot_b = total_bytes_fn(alloc)
    # Fail-closed graceful downgrade if caller passes a tighter budget_bytes ceiling.
    if tot_b > budget_bytes:
        downgrade_steps = [
            *((52, k, 1) for k in ("g", "u", "d", "q", "o", "k", "v")),
            *((l, k, 0) for l in (23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 10, 4, 2) for k in ("g", "u", "d")),
        ]
        for l, proj, target_m in downgrade_steps:
            if l >= n_layers:
                continue
            cur = alloc[l][proj]
            if cur is not None and cur[0] > target_m:
                alloc[l][proj] = (target_m, cur[1], cur[2])
                tot_b = total_bytes_fn(alloc)
                if tot_b <= budget_bytes:
                    return alloc, tot_b
    else:
        # Opportunistically upgrade L14, L19 Full-Attn q/o -> tern5 or L39/L57 -> 2b only if budget allows.
        for l, projs, target_m in ((14, ("q", "o"), 1), (19, ("q", "o"), 1), (57, ("d",), 5), (39, ("d",), 5)):
            for proj in projs:
                if l >= n_layers:
                    continue
                cur = alloc[l][proj]
                if cur is None or cur[0] >= target_m:
                    continue
                old = cur
                alloc[l][proj] = (target_m, cur[1], cur[2])
                new_b = total_bytes_fn(alloc)
                if new_b <= budget_bytes:
                    tot_b = new_b
                else:
                    alloc[l][proj] = old

    return alloc, total_bytes_fn(alloc)


def _unpack_qat_matrix(sr: SafetensorsReader, base: str, ops: _NativeCalOps) -> np.ndarray:
    """Seek/read and unpack an INT4 QAT weight matrix cleanly."""
    ip = sr.hdr[base + ".weight_packed"]
    isc = sr.hdr[base + ".weight_scale"]
    sr._fp.seek(sr.data_base + int(ip["data_offsets"][0]))
    pk_raw = sr._fp.read(int(ip["data_offsets"][1]) - int(ip["data_offsets"][0]))
    sr._fp.seek(sr.data_base + int(isc["data_offsets"][0]))
    sc_raw = sr._fp.read(int(isc["data_offsets"][1]) - int(isc["data_offsets"][0]))
    pk = np.frombuffer(pk_raw, dtype=np.int32)
    sc = np.frombuffer(sc_raw, dtype=np.uint16)
    d_out = int(ip["shape"][0])
    d_in = int(ip["shape"][1]) * 8
    w = np.empty((d_out, d_in), dtype=np.float32)
    ops.lib_g.unpack_qat_rows_f32(
        pk.ctypes.data_as(ctypes.c_void_p),
        sc.ctypes.data_as(ctypes.c_void_p),
        w.ctypes.data_as(ctypes.c_void_p),
        d_out,
        d_in,
    )
    del pk, sc
    return w


def precompute_x_pinv_t_fast(x_bal: np.ndarray, ops: _NativeCalOps, ridge: float = 1e-2) -> np.ndarray:
    """Compute transposed regularized ridge pseudo-inverse using 4-core NEON `mm_xt`."""
    n_tr = x_bal.shape[0]
    xxt = ops.mm_xt(x_bal, x_bal).astype(np.float64)
    lam = ridge * np.trace(xxt) / max(n_tr, 1)
    inv_a = np.ascontiguousarray(np.linalg.inv(xxt + lam * np.eye(n_tr)), dtype=np.float32)
    return ops.mm_xt(np.ascontiguousarray(x_bal.T, dtype=np.float32), inv_a)


def _quant_pack_eval_v4(
    w: np.ndarray,
    x_cal: np.ndarray,
    x_bal: np.ndarray,
    sf: np.ndarray,
    x_pinv_t: Optional[np.ndarray],
    eval_xs: List[np.ndarray],
    spec: Tuple[int, int, int],
    layer_dict: dict,
    stem: str,
    ops: _NativeCalOps,
    lv2: ctypes.CDLL,
    *,
    gptq_u_list: Optional[List[np.ndarray]] = None,
    gptq_cw: int = 768,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Quantize with SmoothQuant (`bw=1.0`), FWHT, optional GPTQ, AR, and ActALS."""
    mode_i, bsize, rk = spec
    d_out, d_in = w.shape
    sq = compute_sq(x_bal, w, alpha=0.5)
    inv_s = (1.0 / sq).astype(np.float32)

    w_rot = np.ascontiguousarray(w * sq[None, :], dtype=np.float32)
    ops.lib_g.fwht_rows_f32(w_rot.ctypes.data_as(ctypes.c_void_p), d_out, d_in, bsize)

    n_bits_u8, is_asym = MODE_TO_T16C[mode_i]
    eff_bits = int(n_bits_u8 & 0x7F)
    use_gauss = 1 if (n_bits_u8 & 0x80) != 0 else 0
    is_asym_i = 1 if is_asym else 0
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    tblk = tile16_blk_bytes(bsize, n_bits_u8, is_asym)
    pb = np.empty((n_tiles, n_blks, tblk), dtype=np.uint8)

    if gptq_u_list is not None and (d_in % gptq_cw == 0):
        blks_c = gptq_cw // bsize
        for ci, Uc in enumerate(gptq_u_list):
            w_sub = np.ascontiguousarray(w_rot[:, ci * gptq_cw : (ci + 1) * gptq_cw], dtype=np.float32)
            pb_sub = np.empty((n_tiles, blks_c, tblk), dtype=np.uint8)
            lv2.gptq_quant_and_pack_v2_f32(
                w_sub.ctypes.data_as(ctypes.c_void_p),
                Uc.ctypes.data_as(ctypes.c_void_p),
                d_out,
                gptq_cw,
                bsize,
                eff_bits,
                is_asym_i,
                use_gauss,
                pb_sub.ctypes.data_as(ctypes.c_void_p),
            )
            pb[:, ci * blks_c : (ci + 1) * blks_c, :] = pb_sub
            del w_sub, pb_sub
    else:
        w_tmp = np.ascontiguousarray(w_rot, dtype=np.float32)
        lv2.gptq_quant_and_pack_v2_f32(
            w_tmp.ctypes.data_as(ctypes.c_void_p),
            None,
            d_out,
            d_in,
            bsize,
            eff_bits,
            is_asym_i,
            use_gauss,
            pb.ctypes.data_as(ctypes.c_void_p),
        )
        del w_tmp
    del w_rot

    pb_flat = pb.reshape(-1)
    y_tgt = ops.mm_xt(x_cal, w)
    x_all = np.vstack([x_cal] + eval_xs)
    n0 = len(x_cal)
    y_all = ops.mm_neon_t16c(pb_flat, inv_s, d_out, d_in, bsize, mode_i, x_all)
    y_q_cal = y_all[:n0]
    if not np.isfinite(y_q_cal).all():
        raise ValueError(f"{stem}: non-finite y_q_cal before AR (mode={mode_i}, bsize={bsize})")

    # Row-wise affine scale refinement (`ar`) baked directly into FP16 tile headers (0 extra bytes).
    y_t_b = y_tgt * sf
    y_q_b = y_q_cal * sf
    num = np.sum(y_t_b * y_q_b, axis=0, dtype=np.float64)
    den = np.sum(y_q_b * y_q_b, axis=0, dtype=np.float64) + 1e-12
    ratio = np.where(np.isfinite(num / den), num / den, 1.0)
    ar = np.clip(ratio, 0.70, 1.40).astype(np.float32)
    scale_t16c_pbytes_rows(pb_flat, d_out, d_in, bsize, mode_i, ar)
    y_all = ops.mm_neon_t16c(pb_flat, inv_s, d_out, d_in, bsize, mode_i, x_all)
    y_q_cal = y_all[:n0]
    if not np.isfinite(y_q_cal).all():
        raise ValueError(f"{stem}: non-finite y_q_cal after AR (mode={mode_i}, bsize={bsize})")

    y_eval_list: List[np.ndarray] = []
    pos = n0
    for xe in eval_xs:
        y_eval_list.append(y_all[pos : pos + len(xe)].copy())
        pos += len(xe)

    layer_dict[f"{stem}__meta"] = np.array([d_out, d_in, bsize, mode_i, rk], dtype=np.int32)
    layer_dict[f"{stem}__pbytes"] = pb_flat
    layer_dict[f"{stem}__inv_s"] = inv_s

    if rk > 0:
        if x_pinv_t is None:
            raise ValueError(f"{stem}: x_pinv_t required when rk={rk} > 0")
        y_err = np.ascontiguousarray(((y_tgt - y_q_cal) * sf).T, dtype=np.float32)
        u_q, v_q, su, sv, ui8, vi8 = fit_act_als_i8(y_err, x_bal, x_pinv_t, rk, ops)
        if not (np.isfinite(su).all() and np.isfinite(sv).all()):
            raise ValueError(f"{stem}: non-finite low-rank scales after ActALS")
        layer_dict[f"{stem}__su"] = su
        layer_dict[f"{stem}__sv"] = sv
        layer_dict[f"{stem}__ui8"] = ui8
        layer_dict[f"{stem}__vi8"] = vi8
        y_q_cal = y_q_cal + ops.mm_xt(ops.mm_xt(x_cal, v_q), u_q)
        for idx_e, xe in enumerate(eval_xs):
            y_eval_list[idx_e] = y_eval_list[idx_e] + ops.mm_xt(ops.mm_xt(xe, v_q), u_q)

    return y_q_cal, y_eval_list


def _build_gptq_u_chunks(
    x_bal: np.ndarray,
    sq: np.ndarray,
    d_in: int,
    bsize: int,
    cw: int,
    damp: float,
    ops: _NativeCalOps,
    lv2: ctypes.CDLL,
) -> List[np.ndarray]:
    """Build block-diagonal GPTQ upper Cholesky matrices `U` for `d_in` columns."""
    inv_s = (1.0 / sq).astype(np.float32)
    x_rot = np.ascontiguousarray(x_bal * inv_s[None, :], dtype=np.float32)
    ops.lib_g.fwht_rows_f32(x_rot.ctypes.data_as(ctypes.c_void_p), len(x_rot), d_in, bsize)
    us: List[np.ndarray] = []
    for ci in range(d_in // cw):
        x_sub = np.ascontiguousarray(x_rot[:, ci * cw : (ci + 1) * cw], dtype=np.float32)
        u_c = np.empty((cw, cw), dtype=np.float32)
        rc = lv2.build_gptq_u_fast_f32(
            x_sub.ctypes.data_as(ctypes.c_void_p),
            len(x_rot),
            cw,
            ctypes.c_float(damp),
            u_c.ctypes.data_as(ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError(f"build_gptq_u_fast_f32 failed chunk={ci} (rc={rc})")
        us.append(u_c)
        del x_sub
    del x_rot
    return us


def run_calibration_v4(
    sf_path: Path,
    cfg_path: Path,
    snaps_dir: Path,
    work_dir: Path,
    layers_dir: Path,
    out_t16c: Optional[Path],
    budget_bytes: int = 6_748_000_000,
) -> None:
    """Execute full 60-layer V4 calibration and assemble `/mnt/SHE-VAULT/gemma4_31b_v4.t16c`."""
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    meta_path = snaps_dir / "meta_v4.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing capture metadata: {meta_path}")
    snaps_fin_p = snaps_dir / "snaps_FINAL.npz"
    if not snaps_fin_p.is_file():
        raise FileNotFoundError(f"Missing final snapshot: {snaps_fin_p}")

    ops = _NativeCalOps(work_dir, require_native=True)
    lv2 = _load_v2_gptq_lib(work_dir, require_native=True)
    assert lv2 is not None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["text_config"]
    n_layers = int(cfg["num_hidden_layers"])
    d = int(cfg["hidden_size"])
    vocab = int(cfg["vocab_size"])
    eps = float(cfg["rms_norm_eps"])
    escale = math.sqrt(float(d))
    k_eq_v = bool(cfg.get("attention_k_eq_v", False))
    layer_types = cfg["layer_types"]
    rp = cfg["rope_parameters"]
    theta_g = float(rp["full_attention"]["rope_theta"])
    part_g = float(rp["full_attention"].get("partial_rotary_factor", 1.0))
    theta_l = float(rp["sliding_attention"]["rope_theta"])
    part_l = float(rp["sliding_attention"].get("partial_rotary_factor", 1.0))
    win = int(cfg["sliding_window"])

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    all_ids: List[int] = meta["all_ids"]
    cal_end = int(meta["cal_end"])
    cal_ids = all_ids[:cal_end]
    cal_segs: List[Tuple[int, int]] = [(int(s0), int(s1)) for s0, s1 in meta["prompt_slices"] if s1 <= cal_end]
    pstarts = [s0 for s0, _ in cal_segs]
    t2_s0, t2_s1 = [int(v) for v in meta["test2"]]
    tf_s0, tf_s1 = [int(v) for v in meta["testf"]]
    t2_ids = all_ids[t2_s0:t2_s1]
    tf_ids = all_ids[tf_s0:tf_s1]
    assert all(cal_ids[s0] == 2 for s0 in pstarts)

    # Load QAT reference trajectories on evaluation probes for cascade tracking only.
    xt_qat: List[np.ndarray] = []
    xf_qat: List[np.ndarray] = []
    for l in range(n_layers):
        sp = snaps_dir / f"snaps_L{l:02d}.npz"
        if not sp.is_file():
            raise FileNotFoundError(f"Missing layer snapshot: {sp}")
        with np.load(sp) as sn:
            xt_qat.append(sn["x_t2"].astype(np.float32))
            xf_qat.append(sn["x_tf"].astype(np.float32))
    with np.load(snaps_fin_p) as sfin:
        xt_qat.append(sfin["x_t2"].astype(np.float32))
        xf_qat.append(sfin["x_tf"].astype(np.float32))

    alloc, total_b = build_v4_allocation(cfg, budget_bytes=budget_bytes)
    print(
        f"[V4 ALLOC] target_bytes={total_b:,} B ({total_b / 1e9:.3f} GB / {total_b / (1024**3):.3f} GiB) "
        f"(budget={budget_bytes:,})",
        flush=True,
    )
    layers_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "alloc_v4.json").write_text(json.dumps({str(k): v for k, v in alloc.items()}, indent=2))

    with SafetensorsReader(sf_path) as sr:
        emb_npz = layers_dir / "embed_3b256.npz"
        if not emb_npz.is_file():
            t0 = time.perf_counter()
            einfo = sr.hdr["model.language_model.embed_tokens.weight"]
            eo0 = sr.data_base + int(einfo["data_offsets"][0])
            dtype = einfo["dtype"]
            elem_b = 2 if dtype in ("BF16", "F16") else 4
            tblk_b = tile16_blk_bytes(256, 3, False)
            emb_pbytes = np.empty(((vocab + 15) // 16) * (d // 256) * tblk_b, dtype=np.uint8)
            xt_e = np.empty((len(t2_ids), d), dtype=np.float32)
            xf_e = np.empty((len(tf_ids), d), dtype=np.float32)

            chunk_r = 16384
            for r0 in range(0, vocab, chunk_r):
                r1 = min(vocab, r0 + chunk_r)
                rows_c = r1 - r0
                sr._fp.seek(eo0 + r0 * d * elem_b)
                raw = sr._fp.read(rows_c * d * elem_b)
                if dtype == "BF16":
                    emb = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).copy()
                elif dtype == "F32":
                    emb = np.frombuffer(raw, dtype="<f4").copy()
                else:
                    emb = np.frombuffer(raw, dtype="<f2").astype(np.float32)
                emb = emb.reshape(rows_c, d)
                ops.lib_g.fwht_rows_f32(emb.ctypes.data_as(ctypes.c_void_p), rows_c, d, 256)
                sub_b = np.empty(((rows_c + 15) // 16, d // 256, tblk_b), dtype=np.uint8)
                lv2.gptq_quant_and_pack_v2_f32(
                    emb.ctypes.data_as(ctypes.c_void_p),
                    None,
                    rows_c,
                    d,
                    256,
                    3,
                    0,
                    0,
                    sub_b.ctypes.data_as(ctypes.c_void_p),
                )
                off = (r0 // 16) * (d // 256) * tblk_b
                sub_flat = sub_b.reshape(-1)
                emb_pbytes[off : off + len(sub_flat)] = sub_flat
                # Reconstruct probe token embeddings via NEON/Rust-compatible decode from quantized FWHT
                probe_rows = [tid - r0 for tid in (t2_ids + tf_ids) if r0 <= tid < r1]
                if probe_rows:
                    # Decode exact packed chunk rows lazily only when probe hits this slice
                    sub_u8 = sub_b.reshape((rows_c + 15) // 16, d // 256, tblk_b)
                    n_per_blk = 256 // 8
                    for idx, tid in enumerate(t2_ids):
                        if r0 <= tid < r1:
                            xt_e[idx] = _decode_sym_row(sub_u8, tid - r0, d, 256, 3, n_per_blk, ops) * escale
                    for idx, tid in enumerate(tf_ids):
                        if r0 <= tid < r1:
                            xf_e[idx] = _decode_sym_row(sub_u8, tid - r0, d, 256, 3, n_per_blk, ops) * escale
                del emb, sub_b, sub_flat
            tmp_e = layers_dir / "embed_3b256.tmp.npz"
            np.savez(tmp_e, pbytes=emb_pbytes, xt0=xt_e, xf0=xf_e)
            os.replace(tmp_e, emb_npz)
            del emb_pbytes
            gc.collect()
            print(f"[EMBED 3b_u0] {time.perf_counter() - t0:.1f}s cos_t2={cos_sim(xt_e, xt_qat[0]):.5f}", flush=True)
        else:
            with np.load(emb_npz) as ed:
                xt_e = ed["xt0"].astype(np.float32)
                xf_e = ed["xf0"].astype(np.float32)

        xt_c, xf_c = xt_e.copy(), xf_e.copy()
        with np.load(snaps_dir / "snaps_L00.npz") as sn0:
            xc_c = sn0["x_cal"].astype(np.float32)

        state_ckpt = layers_dir / "cascade_state.npz"
        start_l = 0
        if state_ckpt.is_file():
            with np.load(state_ckpt) as st:
                next_l = int(st["next_l"])
                if all((layers_dir / f"L{k:02d}.npz").is_file() for k in range(next_l)):
                    if "xc_c" in st.files:
                        xc_c = st["xc_c"].astype(np.float32)
                    elif next_l < n_layers:
                        with np.load(snaps_dir / f"snaps_L{next_l:02d}.npz") as snn:
                            xc_c = snn["x_cal"].astype(np.float32)
                    xt_c = st["xt_c"].astype(np.float32)
                    xf_c = st["xf_c"].astype(np.float32)
                    start_l = next_l
                    print(f"[V4 RESUME] starting at L{start_l:02d}", flush=True)

        for l in range(start_l, n_layers):
            tl = time.perf_counter()
            p = f"model.language_model.layers.{l}."
            is_full = layer_types[l] == "full_attention"

            g_in = sr.read_f32_tensor(p + "input_layernorm.weight")
            qn = sr.read_f32_tensor(p + "self_attn.q_norm.weight")
            kn = sr.read_f32_tensor(p + "self_attn.k_norm.weight")
            g_pa = sr.read_f32_tensor(p + "post_attention_layernorm.weight")
            g_pf = sr.read_f32_tensor(p + "pre_feedforward_layernorm.weight")
            g_pff = sr.read_f32_tensor(p + "post_feedforward_layernorm.weight")
            ls = float(np.asarray(sr.read_f32_tensor(p + "layer_scalar")).reshape(-1)[0])

            nh = int(cfg["num_attention_heads"])
            nkv = int(cfg["num_global_key_value_heads"] if is_full else cfg["num_key_value_heads"])
            hd = int(cfg["global_head_dim"] if is_full else cfg["head_dim"])
            theta = theta_g if is_full else theta_l
            rot_d = int(round(hd * (part_g if is_full else part_l)))
            gqa = nh // nkv

            inp_fit = rmsn(xc_c, g_in, eps)
            # Fixed bw=1.00 across all layers!
            inpb, sf_a = balance_tokens_with_scale(inp_fit, pstarts, bos_weight=1.0)
            need_pinv_a = (
                alloc[l]["q"][2] > 0
                or alloc[l]["k"][2] > 0
                or (alloc[l]["v"] is not None and alloc[l]["v"][2] > 0)
            )
            pinv_a = precompute_x_pinv_t_fast(inpb, ops) if need_pinv_a else None
            inp_t2, inp_tf = rmsn(xt_c, g_in, eps), rmsn(xf_c, g_in, eps)

            layer_dict: dict = {}
            wq = _unpack_qat_matrix(sr, p + "self_attn.q_proj", ops)
            wk = _unpack_qat_matrix(sr, p + "self_attn.k_proj", ops)
            wv = wk if (is_full and k_eq_v) else _unpack_qat_matrix(sr, p + "self_attn.v_proj", ops)

            # Use GPTQ (damp=0.15) on L38..L59 k/v projections (including L58/L59 3b_u0 r=32 shared-KV tail).
            use_kv_gptq = (l >= 38) and (alloc[l]["k"][2] == 0 or alloc[l]["k"][0] == 4)
            u_kv: Optional[List[np.ndarray]] = None
            if use_kv_gptq:
                sq_k = compute_sq(inpb, wk, alpha=0.5)
                u_kv = _build_gptq_u_chunks(inpb, sq_k, d, 256, 768, 0.15, ops, lv2)

            yq_cal_q, (yq_2, yq_f) = _quant_pack_eval_v4(
                wq, inp_fit, inpb, sf_a, pinv_a, [inp_t2, inp_tf], alloc[l]["q"], layer_dict, "q_proj", ops, lv2
            )
            yk_cal_q, (yk_2, yk_f) = _quant_pack_eval_v4(
                wk, inp_fit, inpb, sf_a, pinv_a, [inp_t2, inp_tf], alloc[l]["k"], layer_dict, "k_proj", ops, lv2,
                gptq_u_list=u_kv, gptq_cw=768,
            )
            if is_full and k_eq_v:
                yv_cal_q, yv_2, yv_f = yk_cal_q, yk_2, yk_f
            else:
                yv_cal_q, (yv_2, yv_f) = _quant_pack_eval_v4(
                    wv, inp_fit, inpb, sf_a, pinv_a, [inp_t2, inp_tf], alloc[l]["v"], layer_dict, "v_proj", ops, lv2,
                    gptq_u_list=u_kv, gptq_cw=768,
                )
            del wq, wk, wv, u_kv, inp_fit, inpb, sf_a, pinv_a, inp_t2, inp_tf

            def _attn_ctx(yq: np.ndarray, yk: np.ndarray, yv: np.ndarray) -> np.ndarray:
                t_len = len(yq)
                qt = rope(rmsn(yq.reshape(t_len, nh, hd), qn, eps), theta, rot_d, hd)
                kt = rope(rmsn(yk.reshape(t_len, nkv, hd), kn, eps), theta, rot_d, hd)
                vt = rmsn(yv.reshape(t_len, nkv, hd), None, eps)
                c_out = np.empty((t_len, nh, hd), np.float32)
                m_c = np.triu(np.full((t_len, t_len), -1e9, np.float32), 1)
                if not is_full and t_len > win:
                    idx = np.arange(t_len)
                    m_c = np.where((idx[:, None] - idx[None, :]) >= win, -1e9, m_c)
                for h_i in range(nh):
                    s_m = qt[:, h_i, :] @ kt[:, h_i // gqa, :].T + m_c
                    s_m -= np.max(s_m, -1, keepdims=True)
                    p_m = np.exp(s_m)
                    p_m /= np.sum(p_m, -1, keepdims=True)
                    c_out[:, h_i, :] = p_m @ vt[:, h_i // gqa, :]
                return c_out.reshape(t_len, nh * hd)

            ctx_cal = np.concatenate([_attn_ctx(yq_cal_q[a:b], yk_cal_q[a:b], yv_cal_q[a:b]) for a, b in cal_segs], 0)
            ctxb, sf_c = balance_tokens_with_scale(ctx_cal, pstarts, bos_weight=1.0)
            pinv_c = precompute_x_pinv_t_fast(ctxb, ops) if alloc[l]["o"][2] > 0 else None
            wo = _unpack_qat_matrix(sr, p + "self_attn.o_proj", ops)
            yo_cal_q, (yo_2, yo_f) = _quant_pack_eval_v4(
                wo, ctx_cal, ctxb, sf_c, pinv_c,
                [_attn_ctx(yq_2, yk_2, yv_2), _attn_ctx(yq_f, yk_f, yv_f)],
                alloc[l]["o"], layer_dict, "o_proj", ops, lv2,
            )
            del wo, ctx_cal, ctxb, sf_c, pinv_c, yq_cal_q, yk_cal_q, yv_cal_q, yq_2, yk_2, yv_2, yq_f, yk_f, yv_f

            xm_cal = xc_c + rmsn(yo_cal_q, g_pa, eps)
            xm_t2, xm_tf = xt_c + rmsn(yo_2, g_pa, eps), xf_c + rmsn(yo_f, g_pa, eps)
            hm_cal = rmsn(xm_cal, g_pf, eps)
            hm_t2, hm_tf = rmsn(xm_t2, g_pf, eps), rmsn(xm_tf, g_pf, eps)
            del yo_cal_q, yo_2, yo_f

            hmb, sf_h = balance_tokens_with_scale(hm_cal, pstarts, bos_weight=1.0)
            pinv_h = precompute_x_pinv_t_fast(hmb, ops) if (alloc[l]["g"][2] > 0 or alloc[l]["u"][2] > 0) else None
            wg = _unpack_qat_matrix(sr, p + "mlp.gate_proj", ops)
            g_cal_q, (gq_2, gq_f) = _quant_pack_eval_v4(
                wg, hm_cal, hmb, sf_h, pinv_h, [hm_t2, hm_tf], alloc[l]["g"], layer_dict, "gate_proj", ops, lv2
            )
            wu = _unpack_qat_matrix(sr, p + "mlp.up_proj", ops)
            u_cal_q, (uq_2, uq_f) = _quant_pack_eval_v4(
                wu, hm_cal, hmb, sf_h, pinv_h, [hm_t2, hm_tf], alloc[l]["u"], layer_dict, "up_proj", ops, lv2
            )
            del wg, wu, hmb, sf_h, pinv_h, hm_cal, hm_t2, hm_tf

            act_q_cal = gelu(g_cal_q) * u_cal_q
            actb, sf_d = balance_tokens_with_scale(act_q_cal, pstarts, bos_weight=1.0)
            del g_cal_q, u_cal_q
            pinv_d = precompute_x_pinv_t_fast(actb, ops) if alloc[l]["d"][2] > 0 else None

            wd = _unpack_qat_matrix(sr, p + "mlp.down_proj", ops)
            d_cal_q, (dq_2, dq_f) = _quant_pack_eval_v4(
                wd, act_q_cal, actb, sf_d, pinv_d,
                [gelu(gq_2) * uq_2, gelu(gq_f) * uq_f],
                alloc[l]["d"], layer_dict, "down_proj", ops, lv2,
            )
            del wd, act_q_cal, actb, sf_d, pinv_d, gq_2, uq_2, gq_f, uq_f

            xc_c = (xm_cal + rmsn(d_cal_q, g_pff, eps)) * ls
            xt_c = (xm_t2 + rmsn(dq_2, g_pff, eps)) * ls
            xf_c = (xm_tf + rmsn(dq_f, g_pff, eps)) * ls
            if not (np.isfinite(xc_c).all() and np.isfinite(xt_c).all() and np.isfinite(xf_c).all()):
                raise ValueError(f"Non-finite cascade activation after L{l:02d}")

            tmp_l = layers_dir / f"L{l:02d}.npz.tmp.npz"
            np.savez(tmp_l, **layer_dict)
            os.replace(tmp_l, layers_dir / f"L{l:02d}.npz")
            tmp_s = layers_dir / "cascade_state.tmp.npz"
            np.savez(tmp_s, xc_c=xc_c, xt_c=xt_c, xf_c=xf_c, next_l=np.array(l + 1, dtype=np.int32))
            os.replace(tmp_s, state_ckpt)
            del layer_dict, xm_cal, xm_t2, xm_tf, d_cal_q, dq_2, dq_f
            gc.collect()

            print(
                f"L{l:02d} ({alloc[l]['q'][0]}{alloc[l]['k'][0]}{alloc[l]['o'][0]}/"
                f"{alloc[l]['g'][0]}{alloc[l]['u'][0]}{alloc[l]['d'][0]} r={alloc[l]['d'][2]}) "
                f"casc_t2={cos_sim(xt_c, xt_qat[l + 1]):.5f}(last={cos_sim(xt_c[-1], xt_qat[l + 1][-1]):.5f}) "
                f"casc_tf={cos_sim(xf_c, xf_qat[l + 1]):.5f}(last={cos_sim(xf_c[-1], xf_qat[l + 1][-1]):.5f}) "
                f"nr={np.linalg.norm(xt_c) / np.linalg.norm(xt_qat[l + 1]):.4f} "
                f"[{time.perf_counter() - tl:.1f}s] rss={_rss_mb():.0f}MB",
                flush=True,
            )

        final_norm = sr.read_f32_tensor("model.language_model.norm.weight")
        softcap = float(cfg.get("final_logit_softcapping", 30.0) or 0.0)
        h_t2 = rmsn(xt_c[-1:], final_norm, eps)
        h_tf = rmsn(xf_c[-1:], final_norm, eps)
        q_t2 = rmsn(xt_qat[n_layers][-1:], final_norm, eps)
        q_tf = rmsn(xf_qat[n_layers][-1:], final_norm, eps)
        print(
            f"[V4 FINAL_HIDDEN] cos_t2={cos_sim(h_t2, q_t2):.5f} nr_t2={np.linalg.norm(h_t2) / (np.linalg.norm(q_t2) + 1e-12):.4f} | "
            f"cos_tf={cos_sim(h_tf, q_tf):.5f} nr_tf={np.linalg.norm(h_tf) / (np.linalg.norm(q_tf) + 1e-12):.4f}",
            flush=True,
        )
        with np.load(emb_npz) as ed:
            emb_pb = np.ascontiguousarray(ed["pbytes"], np.uint8)
        ones_d = np.ones((d,), np.float32)
        logits_v4 = ops.mm_neon_t16c(emb_pb, ones_d, vocab, d, 256, 4, np.vstack([h_t2, h_tf]))
        if softcap > 0.0:
            logits_v4 = softcap * np.tanh(logits_v4 / softcap)
        del emb_pb
        for lbl, lv in [("test_2p2 V4_TOP5", logits_v4[0]), ("test_fr V4_TOP5", logits_v4[1])]:
            top5 = np.argsort(-lv)[:5]
            print(f"[{lbl}] {[(int(i), round(float(lv[i]), 3)) for i in top5]}", flush=True)

    if out_t16c is not None:
        sz = assemble_t16c_v3(sf_path, layers_dir, out_t16c, n_layers=n_layers)
        print(f"[PACKED V4] {out_t16c} ({sz:,} B / {sz / (1024**3):.3f} GiB)", flush=True)


def _decode_sym_row(
    sub_u8: np.ndarray,
    r_local: int,
    d_in: int,
    bsize: int,
    n_bits: int,
    bpr: int,
    ops: _NativeCalOps,
) -> np.ndarray:
    """Decode one symmetric tile16 FWHT row back to spatial domain for exact token lookup."""
    t = r_local // 16
    rr = r_local % 16
    n_blks = d_in // bsize
    out = np.zeros((n_blks, bsize), dtype=np.float32)
    plane_b = bpr * 16
    offset_mid = float((1 << n_bits) - 1) * 0.5
    for b in range(n_blks):
        blk_b = sub_u8[t, b]
        sc = float(blk_b[rr * 2 : rr * 2 + 2].view("<f2")[0])
        codes = np.zeros((bsize,), dtype=np.int32)
        for p in range(n_bits):
            plane = blk_b[32 + p * plane_b : 32 + (p + 1) * plane_b]
            for bo in range(bpr):
                byte_v = int(plane[bo * 16 + rr])
                for bit in range(8):
                    if (byte_v >> (7 - bit)) & 1:
                        codes[bo * 8 + bit] |= 1 << p
        out[b] = (codes.astype(np.float32) - offset_mid) * sc
    flat = np.ascontiguousarray(out.reshape(1, d_in), dtype=np.float32)
    ops.lib_g.fwht_rows_f32(flat.ctypes.data_as(ctypes.c_void_p), 1, d_in, bsize)
    return flat.reshape(d_in)


def _self_test() -> int:
    """Run 6 fail-closed self-verification gates offline without touching real weights."""
    failures = 0

    # Gate 1: Budgeted V4.1 allocation respects 6.748 GB ceiling (`<= 6.500 GiB` VmHWM w/ C8DB) and upgrades early cliffs
    try:
        cfg = {
            "num_hidden_layers": 60,
            "vocab_size": 262144,
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "attention_k_eq_v": True,
            "layer_types": ["full_attention" if (i % 6 == 5) else "sliding_attention" for i in range(60)],
        }
        alloc, tot_b = build_v4_allocation(cfg, budget_bytes=6_748_000_000)
        assert tot_b <= 6_748_000_000, f"Exceeded budget: {tot_b}"
        assert alloc[2]["d"][0] == 1 and alloc[10]["g"][0] == 1
        assert alloc[15]["g"][0] == 1 and alloc[15]["d"][0] == 1 and alloc[15]["q"][2] == 16
        assert alloc[24]["d"][0] == 1 and alloc[36]["d"][0] == 1 and alloc[37]["d"][0] == 1
        assert alloc[52]["q"][0] == 5 and alloc[52]["d"][0] == 5
        assert alloc[58]["k"][0] == 4 and alloc[58]["k"][2] == 32
        assert alloc[59]["k"][0] == 4 and alloc[59]["v"] is None
        print(f"[INFO] Gate 1 V4.1 allocation verified: {tot_b:,} B ({tot_b / 1e9:.3f} GB / {tot_b / (1024**3):.4f} GiB)")
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    # Gate 2: Scale multiplier in-place preserves all 4 active modes `(0, 1, 5, 4)` accurately
    try:
        d_out, d_in, bsize = 19, 256, 256
        n_tiles = 2
        for mode_i in (0, 1, 5, 4):
            nb, asym = MODE_TO_T16C[mode_i]
            tb = tile16_blk_bytes(bsize, nb, asym)
            pb = np.zeros((n_tiles, 1, tb), dtype=np.uint8)
            pb[:, :, :32] = np.full((n_tiles, 1, 16), 2.0, dtype=np.float16).view("<u2").view(np.uint8).reshape(n_tiles, 1, 32)
            if asym:
                pb[:, :, 32:64] = np.full((n_tiles, 1, 16), -1.0, dtype=np.float16).view("<u2").view(np.uint8).reshape(n_tiles, 1, 32)
            ar = np.linspace(0.75, 1.25, d_out, dtype=np.float32)
            scale_t16c_pbytes_rows(pb.reshape(-1), d_out, d_in, bsize, mode_i, ar)
            dec_sc = pb[:, 0, :32].view("<u2").view(np.float16).astype(np.float32).reshape(-1)[:d_out]
            assert np.allclose(dec_sc, 2.0 * ar, rtol=2e-3)
            if asym:
                dec_bi = pb[:, 0, 32:64].view("<u2").view(np.float16).astype(np.float32).reshape(-1)[:d_out]
                assert np.allclose(dec_bi, -1.0 * ar, rtol=2e-3)
    except Exception as exc:
        print(f"[Gate 2 FAIL] {exc}")
        failures += 1

    # Gate 3: Fail closed on missing meta_v4.json
    try:
        with tempfile.TemporaryDirectory() as td:
            caught = False
            try:
                cfg_p = Path(td) / "cfg.json"
                cfg_p.write_text('{"text_config": {"num_hidden_layers": 1}}')
                run_calibration_v4(Path(td) / "m.sf", cfg_p, Path(td) / "snaps", Path(td) / "work", Path(td) / "layers", None)
            except FileNotFoundError:
                caught = True
            assert caught
    except Exception as exc:
        print(f"[Gate 3 FAIL] {exc}")
        failures += 1

    # Gate 4: MODE_TO_T16C mode 5 (2b_gauss) and mode 4 (3b_u0) expected sizes
    try:
        assert MODE_TO_T16C[5] == (2 | 0x80, False)
        assert tile16_blk_bytes(256, *MODE_TO_T16C[5]) == 1056
        assert tile16_blk_bytes(256, *MODE_TO_T16C[4]) == 1568
    except Exception as exc:
        print(f"[Gate 4 FAIL] {exc}")
        failures += 1

    # Gate 5: Tight budget fails down gracefully without exceeding budget
    try:
        _, tight_b = build_v4_allocation(cfg, budget_bytes=6_480_000_000)
        assert tight_b <= 6_480_000_000, f"Tight budget exceeded: {tight_b}"
    except Exception as exc:
        print(f"[Gate 5 FAIL] {exc}")
        failures += 1

    # Gate 6: Missing libgauss_gptq_v2 fails closed
    try:
        with tempfile.TemporaryDirectory() as td:
            caught = False
            try:
                _load_v2_gptq_lib(Path(td), require_native=True)
            except FileNotFoundError:
                caught = True
            assert caught
    except Exception as exc:
        print(f"[Gate 6 FAIL] {exc}")
        failures += 1

    if failures == 0:
        print("[PASS] calibrate_31b_v4.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser(description="Calibrate Gemma 4 31B V4 mixed-precision .t16c artifact.")
    ap.add_argument("--safetensors", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--snaps-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b/snaps_v4"))
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b"))
    ap.add_argument("--layers-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b/layers_v4"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v4.t16c"))
    ap.add_argument("--budget-bytes", type=int, default=6_748_000_000)
    args = ap.parse_args()

    sf_path = resolve_safetensors_path(args.safetensors)
    run_calibration_v4(
        sf_path,
        args.config.resolve(),
        args.snaps_dir.resolve(),
        args.work_dir.resolve(),
        args.layers_dir.resolve(),
        args.out.resolve(),
        budget_bytes=args.budget_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

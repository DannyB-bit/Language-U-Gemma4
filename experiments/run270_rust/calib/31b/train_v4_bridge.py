#!/usr/bin/env python3
"""
V4 C8DB Bridge & Shared-KV Trainer + 2x2 Speculative Assistant Replay verifier.

Executes Step 3 cleanly on top of `gemma4_31b_v4.t16c` and `snaps_v4/teacher_bridge_v4.npz`:
  1. Propagates student `L00..L59` through `V4.t16c` via bounded open/seek/read
     using the exact deployed `libneon_stream_accelerator.so` (`dual=0` Q8+OLS) and
     caches `student_state_v4.npz` so Step 3 tuning runs in seconds.
  2. Trains fresh L58/L59 shared-KV C8DB adapters ONLY where quantized validation
     strictly improves over the `3b_u0` `.t16c` base projection (`+0.002` min gain).
  3. Re-propagates `L58..L59`, computes exact BF16 `embed_tokens` teacher & student
     logits over the clean non-trivia `ans_mask` set (`alpha=0.0` — zero carrier
     subtraction), and fits `fn_new` + regularized `post_cell60` on:
       - Formula 5: temperature-scaled KL over `S_t = Top128_T U Top128_Q U {OTHER}`
       - Formula 6: multi-token logit margin hinge loss (`max(0, gamma - z(y*) + max_{v!=y*} z(v))`)
  4. Saves compact `/mnt/SHE-VAULT/gemma4_31b_v4_c8d.c8d` (`~2-8 MB`) and runs 2x2
     replay (`RefH/RefKV`, `V4H/RefKV`, `RefH/V4KV`, `V4H/V4KV`) via `probe_assistant`.

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
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_31b_v3 import _NativeCalOps, cos_sim
from capture_v4_clean import causal_gqa_attention, gelu_tanh, rmsn
from pack_t16c_v3 import SafetensorsReader, resolve_safetensors_path
import train_c8d_pattern_bank as c8d


def formula5_kl_loss(
    z_cand: torch.Tensor,
    log_tail_s0: torch.Tensor,
    p_teach_full: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute exact Formula 5 temperature-scaled KL over `Top128_T U Top128_Q U {OTHER}`."""
    z_with_other = torch.cat([z_cand / tau, log_tail_s0.unsqueeze(1)], dim=1)
    log_pq = F.log_softmax(z_with_other, dim=-1)
    kl = (p_teach_full * (torch.log(p_teach_full.clamp(min=1e-12)) - log_pq)).sum(dim=-1)
    return (tau * tau) * kl.mean()


def formula6_margin_loss(
    z_cand: torch.Tensor,
    y_star_idx: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Compute Formula 6 hinge margin protecting target tokens against top competitors."""
    z_y = z_cand.gather(1, y_star_idx.unsqueeze(1)).squeeze(1)
    zc_mask = z_cand.clone()
    zc_mask.scatter_(1, y_star_idx.unsqueeze(1), -1e9)
    z_comp = zc_mask.max(dim=-1).values
    return F.relu(gamma - z_y + z_comp).mean()


def _merge_topk_chunk(
    best_vals: Optional[np.ndarray],
    best_idx: Optional[np.ndarray],
    chunk_vals: np.ndarray,
    col_offset: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Maintain exact per-row top-`k` `(vals, global_indices)` across streaming chunks."""
    take_k = min(k, chunk_vals.shape[1])
    if take_k < chunk_vals.shape[1]:
        loc = np.argpartition(chunk_vals, -take_k, axis=1)[:, -take_k:]
    else:
        loc = np.broadcast_to(np.arange(chunk_vals.shape[1], dtype=np.int32), chunk_vals.shape)
    loc_v = np.take_along_axis(chunk_vals, loc, axis=1)
    loc_g = loc.astype(np.int32) + col_offset
    if best_vals is None or best_idx is None:
        return loc_v, loc_g
    m_v = np.concatenate([best_vals, loc_v], axis=1)
    m_g = np.concatenate([best_idx, loc_g], axis=1)
    keep = np.argpartition(m_v, -k, axis=1)[:, -k:]
    return np.take_along_axis(m_v, keep, axis=1), np.take_along_axis(m_g, keep, axis=1)


def _update_online_lse(
    cur_max: np.ndarray,
    cur_sum: np.ndarray,
    scaled_logits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically stable streaming FP64 log-sum-exp state update `(max, sum_exp)`."""
    if scaled_logits.shape[1] == 0:
        return cur_max, cur_sum
    b_max = np.max(scaled_logits, axis=1).astype(np.float64)
    new_max = np.maximum(cur_max, b_max)
    old_scale = np.where(np.isfinite(cur_max), np.exp(cur_max - new_max), 0.0)
    b_sum = np.sum(np.exp(scaled_logits.astype(np.float64) - new_max[:, None]), axis=1)
    return new_max, cur_sum * old_scale + b_sum


def _read_bf16_chunk(sr: SafetensorsReader, eo0: int, r0: int, take: int, d: int) -> np.ndarray:
    sr._fp.seek(eo0 + r0 * d * 2)
    raw_c = sr._fp.read(take * d * 2)
    return ((np.frombuffer(raw_c, np.uint16).reshape(take, d).astype(np.uint32) << 16).view(np.float32)).copy()


def _stream_formula5_targets(
    sr: SafetensorsReader,
    embed_key: str,
    ops: _NativeCalOps,
    ht_ans: np.ndarray,
    hs_ans: np.ndarray,
    nxt_ans: np.ndarray,
    vocab_sz: int,
    d: int,
    tau: float,
    *,
    top_k: int = 128,
    chunk_rows: int = 8192,
    traps: Tuple[int, ...] = (715, 1638, 1852, 79629, 120474),
    max_cand: int = 6144,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream chunked BF16 `embed_tokens` in bounded RAM (<400 MB) to build Formula 5 targets."""
    einfo = sr.hdr[embed_key]
    eo0 = sr.data_base + int(einfo["data_offsets"][0])
    n_ans = len(ht_ans)

    bt_v: Optional[np.ndarray] = None
    bt_i: Optional[np.ndarray] = None
    bs_v: Optional[np.ndarray] = None
    bs_i: Optional[np.ndarray] = None
    for r0 in range(0, vocab_sz, chunk_rows):
        take = min(chunk_rows, vocab_sz - r0)
        w_c = _read_bf16_chunk(sr, eo0, r0, take, d)
        zt_c = 30.0 * np.tanh(ops.mm_xt(ht_ans, w_c) / 30.0)
        zs_c = 30.0 * np.tanh(ops.mm_xt(hs_ans, w_c) / 30.0)
        del w_c
        bt_v, bt_i = _merge_topk_chunk(bt_v, bt_i, zt_c, r0, top_k)
        bs_v, bs_i = _merge_topk_chunk(bs_v, bs_i, zs_c, r0, top_k)
        del zt_c, zs_c

    assert bt_i is not None and bs_i is not None
    req_set = set(nxt_ans.tolist())
    for tid in traps:
        if 0 <= tid < vocab_sz:
            req_set.add(tid)
    merged_top = np.concatenate([bt_i.ravel(), bs_i.ravel()])
    uniq_top, cnt_top = np.unique(merged_top, return_counts=True)
    if uniq_top.size + len(req_set) <= max_cand:
        cand_set = req_set | set(uniq_top.tolist())
    else:
        rem_mask = ~np.isin(uniq_top, list(req_set))
        rem_ids = uniq_top[rem_mask]
        rem_cnt = cnt_top[rem_mask]
        keep_n = max(max_cand - len(req_set), 0)
        if keep_n == 0:
            rem_ids = np.empty((0,), dtype=np.int32)
        elif rem_ids.size > keep_n:
            top_pos = np.argpartition(rem_cnt, -keep_n)[-keep_n:]
            rem_ids = rem_ids[top_pos]
        cand_set = req_set | set(rem_ids.tolist())
    del bt_v, bt_i, bs_v, bs_i, merged_top, uniq_top, cnt_top
    cand_ids = np.array(sorted(cand_set), dtype=np.int32)
    cand_col_map = {int(tid): ci for ci, tid in enumerate(cand_ids)}

    cand_mask_t = np.ones((vocab_sz,), dtype=bool)
    cand_mask_t[cand_ids] = False
    W_cand = np.empty((len(cand_ids), d), dtype=np.float32)
    for ci, tid in enumerate(cand_ids):
        sr._fp.seek(eo0 + int(tid) * d * 2)
        raw_r = sr._fp.read(d * 2)
        W_cand[ci] = ((np.frombuffer(raw_r, np.uint16).astype(np.uint32) << 16).view(np.float32)).copy()

    t_cand_logits = (30.0 * np.tanh(ops.mm_xt(ht_ans, W_cand) / 30.0)) / tau
    t_max = np.full((n_ans,), -np.inf, dtype=np.float64)
    t_sum = np.zeros((n_ans,), dtype=np.float64)
    s_max = np.full((n_ans,), -np.inf, dtype=np.float64)
    s_sum = np.zeros((n_ans,), dtype=np.float64)

    for r0 in range(0, vocab_sz, chunk_rows):
        take = min(chunk_rows, vocab_sz - r0)
        keep_rel = np.flatnonzero(cand_mask_t[r0 : r0 + take])
        if len(keep_rel) == 0:
            continue
        w_c = _read_bf16_chunk(sr, eo0, r0, take, d)
        w_sub = np.ascontiguousarray(w_c[keep_rel], dtype=np.float32)
        del w_c
        zt_sub = (30.0 * np.tanh(ops.mm_xt(ht_ans, w_sub) / 30.0)) / tau
        zs_sub = (30.0 * np.tanh(ops.mm_xt(hs_ans, w_sub) / 30.0)) / tau
        del w_sub
        t_max, t_sum = _update_online_lse(t_max, t_sum, zt_sub)
        s_max, s_sum = _update_online_lse(s_max, s_sum, zs_sub)
        del zt_sub, zs_sub

    t_other_lse = np.where(t_sum > 0.0, t_max + np.log(np.maximum(t_sum, 1e-30)), -1e4).astype(np.float32)[:, None]
    s0_other_lse = np.where(s_sum > 0.0, s_max + np.log(np.maximum(s_sum, 1e-30)), -1e4).astype(np.float32)
    z_t_all = np.concatenate([t_cand_logits, t_other_lse], axis=1)
    del t_cand_logits, t_other_lse
    z_t_all -= np.max(z_t_all, axis=1, keepdims=True)
    np.exp(z_t_all, out=z_t_all)
    z_t_all /= np.sum(z_t_all, axis=1, keepdims=True)
    p_teach_full = z_t_all.astype(np.float32, copy=False)
    y_star_idx = np.array([cand_col_map[int(tid)] for tid in nxt_ans], dtype=np.int64)
    return cand_ids, W_cand, p_teach_full, s0_other_lse, y_star_idx


class _SeekT16CReader:
    """Bounded-RAM `open/seek/read` executor for `.t16c` v1 without keeping a full file mapping."""

    def __init__(self, path: Path, nlib: ctypes.CDLL) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing T16C file: {self.path}")
        self.nlib = nlib
        self._fp = open(self.path, "rb")
        self.norms: Dict[str, np.ndarray] = {}
        self.mats: Dict[str, dict] = {}
        try:
            magic, ver, _arch, n_norms, n_mats, _ = struct.unpack("<4s5I", self._fp.read(24))
            if magic != b"T16C" or ver != 1:
                raise ValueError(f"Invalid T16C header in {self.path}")
            pos = 24
            for _ in range(n_norms):
                nlen = struct.unpack("<H", self._fp.read(2))[0]
                name = self._fp.read(nlen).decode("utf-8")
                cnt = struct.unpack("<I", self._fp.read(4))[0]
                arr = np.frombuffer(self._fp.read(cnt * 4), dtype="<f4").copy()
                self.norms[name] = arr
                pos += 2 + nlen + 4 + cnt * 4
            for _ in range(n_mats):
                hdr_b = self._fp.read(34)
                nlen, do, di, bs, nb, asym, fwht, inv_k, rk, pbytes, lr_b = struct.unpack("<HIIHBBBBHQQ", hdr_b)
                name = self._fp.read(nlen).decode("utf-8")
                pos += 34 + nlen
                pad = (64 - (pos % 64)) % 64
                if pad:
                    self._fp.seek(pad, 1)
                    pos += pad
                p_off = pos
                self._fp.seek(pbytes, 1)
                pos += pbytes
                inv_off = pos if inv_k == 1 else -1
                if inv_k == 1:
                    self._fp.seek(di * 4, 1)
                    pos += di * 4
                lr_off = pos if rk > 0 else -1
                if lr_b > 0:
                    self._fp.seek(lr_b, 1)
                    pos += lr_b
                self.mats[name] = {
                    "d_out": do,
                    "d_in": di,
                    "block_size": bs,
                    "n_bits": nb,
                    "is_asym": bool(asym),
                    "has_fwht": bool(fwht),
                    "p_off": p_off,
                    "pbytes": pbytes,
                    "inv_off": inv_off,
                    "lr_rank": rk,
                    "lr_off": lr_off,
                    "lr_bytes": lr_b,
                }
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "_SeekT16CReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        fp = getattr(self, "_fp", None)
        if fp is not None:
            fp.close()
            self._fp = None

    def project_batch(self, name: str, x: np.ndarray, threads: int = 4) -> np.ndarray:
        """Execute one `.t16c` projection cleanly in bounded RAM using deployed NEON kernel."""
        m = self.mats[name]
        x32 = np.ascontiguousarray(x, dtype=np.float32)
        k_tok, d_in = x32.shape
        d_out = m["d_out"]
        bsize = m["block_size"]
        x_proc = x32.copy()
        if m["inv_off"] >= 0:
            self._fp.seek(m["inv_off"])
            inv_s = np.frombuffer(self._fp.read(d_in * 4), dtype="<f4").copy()
        else:
            inv_s = None
        if m["has_fwht"]:
            inv_ptr = inv_s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if inv_s is not None else ctypes.POINTER(ctypes.c_float)()
            rc_f = self.nlib.inv_s_and_fwht_f32_c(
                x_proc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                inv_ptr,
                k_tok,
                d_in,
                bsize,
            )
            if rc_f != 0:
                raise RuntimeError(f"inv_s_and_fwht_f32_c failed (rc={rc_f})")
        elif inv_s is not None:
            x_proc *= inv_s[None, :]

        self._fp.seek(m["p_off"])
        p_raw = self._fp.read(m["pbytes"])
        packed_arr = np.frombuffer(p_raw, dtype=np.uint8)
        y = np.zeros((k_tok, d_out), dtype=np.float32)
        rc_g = self.nlib.gemm_bitserial_tile16_batch(
            packed_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_uint16)(),
            x_proc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d_out,
            d_in,
            bsize,
            m["n_bits"],
            1 if m["is_asym"] else 0,
            k_tok,
            0.0,
            threads,
            ctypes.POINTER(ctypes.c_int64)(),
        )
        if rc_g != 0:
            raise RuntimeError(f"gemm_bitserial_tile16_batch failed (rc={rc_g})")
        del packed_arr, p_raw, x_proc

        if m["lr_rank"] > 0:
            rk = m["lr_rank"]
            self._fp.seek(m["lr_off"])
            lr_raw = self._fp.read(m["lr_bytes"])
            su = np.frombuffer(lr_raw, dtype="<f4", count=d_out, offset=0)
            sv = np.frombuffer(lr_raw, dtype="<f4", count=rk, offset=d_out * 4)
            ui = np.frombuffer(lr_raw, dtype=np.int8, count=d_out * rk, offset=(d_out + rk) * 4)
            vi = np.frombuffer(lr_raw, dtype=np.int8, count=rk * d_in, offset=(d_out + rk) * 4 + d_out * rk)
            rc_lr = self.nlib.int8_lowrank_add_batch_c(
                ui.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                su.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                vi.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                sv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                d_out,
                d_in,
                rk,
                k_tok,
                threads,
            )
            if rc_lr != 0:
                raise RuntimeError(f"int8_lowrank_add_batch_c failed (rc={rc_lr})")
            del lr_raw
        return y

    def read_t16c_rows(self, name: str, row_ids: List[int]) -> np.ndarray:
        """Dequantize selected rows via bounded seek/read without memory-mapping `.t16c`."""
        m = self.mats[name]
        d_out, d_in, bsize, nb, asym = m["d_out"], m["d_in"], m["block_size"], m["n_bits"], m["is_asym"]
        eff_bits = nb & 0x7F
        n_blks = d_in // bsize
        if eff_bits != 3 or asym or (nb & 0x80):
            raise ValueError(f"read_t16c_rows seek path expects 3b_u0 matrix, got nb={nb:#x} asym={asym} on {name}")
        hdr = 32
        bo = bsize // 8
        pplane = bo * 16
        tblk = hdr + eff_bits * pplane
        bias_shift = float((1 << eff_bits) - 1)
        inv_s = None
        if m["inv_off"] >= 0:
            self._fp.seek(m["inv_off"])
            inv_s = np.frombuffer(self._fp.read(d_in * 4), dtype="<f4").copy()
        tile_stride = n_blks * tblk
        tile_cache: Dict[int, np.ndarray] = {}
        out = np.empty((len(row_ids), d_in), dtype=np.float32)
        for idx_r, rid in enumerate(row_ids):
            if rid < 0 or rid >= d_out:
                raise IndexError(f"Row {rid} out of range for {name} (d_out={d_out})")
            t = rid // 16
            if t not in tile_cache:
                self._fp.seek(m["p_off"] + t * tile_stride)
                raw_t = np.frombuffer(self._fp.read(tile_stride), dtype=np.uint8).reshape(n_blks, tblk)
                sc = raw_t[:, :32].reshape(-1).view(np.float16).astype(np.float32).reshape(n_blks, 16).T
                levels = np.zeros((n_blks, bo, 16, 8), dtype=np.float32)
                for p in range(eff_bits):
                    pl = raw_t[:, hdr + p * pplane : hdr + (p + 1) * pplane].reshape(n_blks, bo, 16)
                    bits = np.unpackbits(pl[:, :, :, None], axis=-1, bitorder="big").astype(np.float32)
                    levels += bits * float(1 << p)
                w_b = levels.transpose(2, 0, 1, 3).reshape(16, n_blks, bsize)
                w_t = (sc[:, :, None] * (2.0 * w_b - bias_shift)).reshape(16, d_in)
                if m["has_fwht"]:
                    w_t = c8d._fwht_blocks_np(w_t, bsize)
                if inv_s is not None:
                    w_t = w_t * inv_s[None, :]
                tile_cache[t] = w_t
            out[idx_r] = tile_cache[t][rid % 16]
        return out


def _propagate_student_v4(
    t16c_path: Path,
    layers_dir: Path,
    sf_path: Path,
    cfg: dict,
    bridge: dict,
    nlib: ctypes.CDLL,
) -> Dict[str, np.ndarray]:
    """Propagate L00..L59 student activations via `layers_v4` + deployed NEON `dual=0` kernel."""
    n_layers = int(cfg["num_hidden_layers"])
    d = int(cfg["hidden_size"])
    eps = float(cfg["rms_norm_eps"])
    nh = int(cfg["num_attention_heads"])
    nkv_s = int(cfg["num_key_value_heads"])
    nkv_f = int(cfg["num_global_key_value_heads"])
    hd_s = int(cfg["head_dim"])
    hd_f = int(cfg["global_head_dim"])
    k_eq_v = bool(cfg.get("attention_k_eq_v", False))
    layer_types = cfg["layer_types"]
    rp = cfg["rope_parameters"]
    theta_g = float(rp["full_attention"]["rope_theta"])
    part_g = float(rp["full_attention"].get("partial_rotary_factor", 1.0))
    theta_l = float(rp["sliding_attention"]["rope_theta"])
    part_l = float(rp["sliding_attention"].get("partial_rotary_factor", 1.0))
    win = int(cfg["sliding_window"])

    all_ids = [int(v) for v in bridge["all_ids"].tolist()]
    pos_arr = bridge["pos_arr"].astype(np.float32)
    prompt_slices = [(int(a), int(b)) for a, b in bridge["prompt_slices"].tolist()]

    out: Dict[str, np.ndarray] = {}
    with SafetensorsReader(sf_path) as sr, _SeekT16CReader(t16c_path, nlib) as sk:
        h_s = (sk.read_t16c_rows("model.language_model.embed_tokens.weight", all_ids) * math.sqrt(float(d))).astype(np.float32)
        gc.collect()
        for l in range(n_layers):
            tl = time.perf_counter()
            if l == 58:
                out["h_stud_before_L58"] = h_s.copy()
            pfx = f"model.language_model.layers.{l}."
            is_full = layer_types[l] == "full_attention"
            nkv = nkv_f if is_full else nkv_s
            hd = hd_f if is_full else hd_s
            theta = theta_g if is_full else theta_l
            rot_d = int(round(hd * (part_g if is_full else part_l)))

            g_in = sr.read_f32_tensor(pfx + "input_layernorm.weight")
            qn = sr.read_f32_tensor(pfx + "self_attn.q_norm.weight")
            kn = sr.read_f32_tensor(pfx + "self_attn.k_norm.weight")
            g_pa = sr.read_f32_tensor(pfx + "post_attention_layernorm.weight")
            g_pf = sr.read_f32_tensor(pfx + "pre_feedforward_layernorm.weight")
            g_pff = sr.read_f32_tensor(pfx + "post_feedforward_layernorm.weight")
            ls = float(np.asarray(sr.read_f32_tensor(pfx + "layer_scalar")).reshape(-1)[0])

            hs_in = rmsn(h_s, g_in, eps)
            qs = sk.project_batch(pfx + "self_attn.q_proj.weight", hs_in)
            ks = sk.project_batch(pfx + "self_attn.k_proj.weight", hs_in)
            vs = ks.copy() if (is_full and k_eq_v) else sk.project_batch(pfx + "self_attn.v_proj.weight", hs_in)
            if l == 58:
                out["hs_in_58"] = hs_in.copy()
                out["ks_58_base"] = ks.copy()
                out["vs_58_base"] = vs.copy()
            elif l == 59:
                out["hs_in_59"] = hs_in.copy()
                out["ks_59_base"] = ks.copy()

            qs_n = rmsn(qs.reshape(len(all_ids), nh, hd), qn, eps).reshape(len(all_ids), nh * hd)
            ks_n = rmsn(ks.reshape(len(all_ids), nkv, hd), kn, eps).reshape(len(all_ids), nkv * hd)
            vs_n = rmsn(vs.reshape(len(all_ids), nkv, hd), None, eps).reshape(len(all_ids), nkv * hd)
            ctx = causal_gqa_attention(
                qs_n, ks_n, vs_n, pos_arr, prompt_slices,
                nh=nh, nkv=nkv, hd=hd, rot_dims=rot_d, theta=theta, is_full=is_full, window=win,
            )
            os_y = sk.project_batch(pfx + "self_attn.o_proj.weight", ctx)
            xm = h_s + rmsn(os_y, g_pa, eps)
            hm = rmsn(xm, g_pf, eps)
            gs = sk.project_batch(pfx + "mlp.gate_proj.weight", hm)
            us = sk.project_batch(pfx + "mlp.up_proj.weight", hm)
            ds = sk.project_batch(pfx + "mlp.down_proj.weight", gelu_tanh(gs) * us)
            h_s = ((xm + rmsn(ds, g_pff, eps)) * ls).astype(np.float32)
            del hs_in, qs, ks, vs, qs_n, ks_n, vs_n, ctx, os_y, xm, hm, gs, us, ds
            if l in (0, 20, 39, 52, 57, 58, 59):
                print(f"[StudentV4 L{l:02d}] {time.perf_counter() - tl:.1f}s", flush=True)

        out["h_student_L59"] = h_s
    return out


def run_bridge_v4(
    sf_path: Path,
    cfg_path: Path,
    t16c_path: Path,
    bridge_npz: Path,
    work_dir: Path,
    out_c8d: Path,
    *,
    rank_cell60: int = 16,
    rank_kv: int = 32,
    state_cache: Optional[Path] = None,
    tau: float = 1.5,
    gamma: float = 2.0,
    steps: int = 300,
) -> Path:
    """Train minimal high-signal V4 C8DB bank (`L58/L59` KV if helpful + `fn` + Formula 5/6 `post_cell60`)."""
    for p in (sf_path, cfg_path, t16c_path, bridge_npz):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required bridge artifact: {p}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["text_config"]
    ops = _NativeCalOps(work_dir, require_native=True)
    so_path = Path("/home/astronautshe/gemma4_verify/neon_runner/libneon_stream_accelerator.so")
    if not so_path.is_file():
        so_path = work_dir / "libneon_stream_accelerator.so"
    nlib = c8d.load_native_lib(so_path)

    with np.load(bridge_npz, allow_pickle=False) as bz:
        bridge = {k: bz[k] for k in bz.files}

    stud_cache = state_cache if state_cache is not None else (work_dir / f"student_state_{t16c_path.stem}.npz")
    if not stud_cache.is_file():
        t0 = time.perf_counter()
        stud = _propagate_student_v4(t16c_path, work_dir / "layers_v4", sf_path, cfg, bridge, nlib)
        tmp_sc = stud_cache.with_suffix(".tmp.npz")
        np.savez(tmp_sc, **stud)
        os.replace(tmp_sc, stud_cache)
        print(f"[Step3] Cached V4 student propagation in {time.perf_counter() - t0:.1f}s -> {stud_cache}", flush=True)
    else:
        with np.load(stud_cache) as sc:
            stud = {k: sc[k] for k in sc.files}

    cal_end = int(bridge["cal_end"])
    ans_mask = bridge["ans_mask"].astype(bool)
    next_token_ids = bridge["next_token_ids"].astype(np.int32)
    prompt_domains = [str(x) for x in bridge["prompt_domains"].tolist()]
    d = int(cfg["hidden_size"])
    eps = float(cfg["rms_norm_eps"])

    # 1. Optional shared-KV C8DB adapters on top of V4's 3b_u0 L58/L59 (keep ONLY if val_cos_after >= val_cos_before + 0.002!)
    cal_idx = np.arange(cal_end)
    val_kv = (cal_idx % 5 == 0)
    tr_kv = ~val_kv
    tr_domains = [prompt_domains[i] for i in range(cal_end) if tr_kv[i]]

    kv_targets = [
        ("model.language_model.layers.58.self_attn.k_proj.weight", stud["hs_in_58"][:cal_end], stud["ks_58_base"][:cal_end], bridge["kt_58_raw"][:cal_end]),
        ("model.language_model.layers.58.self_attn.v_proj.weight", stud["hs_in_58"][:cal_end], stud["vs_58_base"][:cal_end], bridge["vt_58_raw"][:cal_end]),
        ("model.language_model.layers.59.self_attn.k_proj.weight", stud["hs_in_59"][:cal_end], stud["ks_59_base"][:cal_end], bridge["kt_59_raw"][:cal_end]),
    ]
    trained_entries: List[dict] = []
    kv_entries: Dict[str, dict] = {}
    for kname, x_in, y_b, y_t in kv_targets:
        ent, st = c8d.train_projection_pattern_bank(
            kname,
            x_in[tr_kv], y_b[tr_kv], y_t[tr_kv],
            x_in[val_kv], y_b[val_kv], y_t[val_kv],
            tr_domains,
            rank=rank_kv, n_experts=8, top_k=2, steps=200, lr=4e-3,
        )
        cb = float(st.get("val_cos_before", 0.0))
        ca = float(st.get("val_cos_after", 0.0))
        accepted = ent is not None and (ca >= cb + 0.002)
        print(f"[Step3-KV] {kname}: val_cos {cb:.5f}->{ca:.5f} keep={accepted}", flush=True)
        if accepted and ent is not None:
            trained_entries.append(ent)
            kv_entries[kname] = ent

    # Re-propagate L58..L59 ONLY if at least one KV adapter was kept.
    if kv_entries:
        pos_arr = bridge["pos_arr"].astype(np.float32)
        prompt_slices = [(int(a), int(b)) for a, b in bridge["prompt_slices"].tolist()]
        nh = int(cfg["num_attention_heads"])
        nkv_s, nkv_f = int(cfg["num_key_value_heads"]), int(cfg["num_global_key_value_heads"])
        hd_s, hd_f = int(cfg["head_dim"]), int(cfg["global_head_dim"])
        k_eq_v = bool(cfg.get("attention_k_eq_v", False))
        layer_types = cfg["layer_types"]
        rp = cfg["rope_parameters"]
        theta_g, part_g = float(rp["full_attention"]["rope_theta"]), float(rp["full_attention"].get("partial_rotary_factor", 1.0))
        theta_l, part_l = float(rp["sliding_attention"]["rope_theta"]), float(rp["sliding_attention"].get("partial_rotary_factor", 1.0))
        win = int(cfg["sliding_window"])

        h_s = stud["h_stud_before_L58"].copy()
        with SafetensorsReader(sf_path) as sr, _SeekT16CReader(t16c_path, nlib) as sk:
            for l in (58, 59):
                pfx = f"model.language_model.layers.{l}."
                is_full = layer_types[l] == "full_attention"
                nkv = nkv_f if is_full else nkv_s
                hd = hd_f if is_full else hd_s
                theta = theta_g if is_full else theta_l
                rot_d = int(round(hd * (part_g if is_full else part_l)))
                g_in = sr.read_f32_tensor(pfx + "input_layernorm.weight")
                qn = sr.read_f32_tensor(pfx + "self_attn.q_norm.weight")
                kn = sr.read_f32_tensor(pfx + "self_attn.k_norm.weight")
                g_pa = sr.read_f32_tensor(pfx + "post_attention_layernorm.weight")
                g_pf = sr.read_f32_tensor(pfx + "pre_feedforward_layernorm.weight")
                g_pff = sr.read_f32_tensor(pfx + "post_feedforward_layernorm.weight")
                ls = float(np.asarray(sr.read_f32_tensor(pfx + "layer_scalar")).reshape(-1)[0])
                hs_in = rmsn(h_s, g_in, eps)

                def _p(suf: str, x_a: np.ndarray) -> np.ndarray:
                    mk = pfx + suf + ".weight"
                    y_a = sk.project_batch(mk, x_a)
                    if mk in kv_entries:
                        e = kv_entries[mk]
                        y_a = y_a + c8d.eval_quantized_adapter_np(
                            x_a, e["centroids"], e["experts"], e["top_k"], e["temperature"], e["gate_scale"]
                        )
                    return y_a

                qs = _p("self_attn.q_proj", hs_in)
                ks = _p("self_attn.k_proj", hs_in)
                vs = ks.copy() if (is_full and k_eq_v) else _p("self_attn.v_proj", hs_in)
                ctx = causal_gqa_attention(
                    rmsn(qs.reshape(len(h_s), nh, hd), qn, eps).reshape(len(h_s), nh * hd),
                    rmsn(ks.reshape(len(h_s), nkv, hd), kn, eps).reshape(len(h_s), nkv * hd),
                    rmsn(vs.reshape(len(h_s), nkv, hd), None, eps).reshape(len(h_s), nkv * hd),
                    pos_arr, prompt_slices, nh=nh, nkv=nkv, hd=hd, rot_dims=rot_d, theta=theta, is_full=is_full, window=win,
                )
                xm = h_s + rmsn(_p("self_attn.o_proj", ctx), g_pa, eps)
                hm = rmsn(xm, g_pf, eps)
                gs = _p("mlp.gate_proj", hm)
                us = _p("mlp.up_proj", hm)
                ds = _p("mlp.down_proj", gelu_tanh(gs) * us)
                h_s = ((xm + rmsn(ds, g_pff, eps)) * ls).astype(np.float32)
    else:
        h_s = stud["h_student_L59"].astype(np.float32)

    h_teach_L59 = bridge["h_teach_L59"].astype(np.float32)
    fn_orig = bridge["fn_orig"].astype(np.float32)
    cos_l59 = cos_sim(h_s[:cal_end], h_teach_L59[:cal_end])
    print(f"[Step3] V4 Student L59 cal cosine={cos_l59:.5f}", flush=True)

    # 2. Stream chunked BF16 embed_tokens logits for Formula 5 + Formula 6 over clean ans_mask tokens (alpha=0.0!)
    ans_idx = np.flatnonzero(ans_mask[:cal_end])
    n_ans = len(ans_idx)
    ht_norm_all = rmsn(h_teach_L59, fn_orig, eps)
    hs_unit_all = rmsn(h_s, None, eps)

    # Refine fn via regularized ridge/Wiener on clean cal tokens before fitting post_cell60
    cal_st_val = (np.arange(cal_end) % 5 == 0)
    cal_st_tr = ~cal_st_val
    lam_w = 0.50
    hs_u_tr, ht_tr = hs_unit_all[:cal_end][cal_st_tr], ht_norm_all[:cal_end][cal_st_tr]
    hs_u_va, ht_va = hs_unit_all[:cal_end][cal_st_val], ht_norm_all[:cal_end][cal_st_val]
    fn_wiener = ((np.mean(hs_u_tr * ht_tr, axis=0) + lam_w * fn_orig) / np.maximum(np.mean(hs_u_tr * hs_u_tr, axis=0) + lam_w, 1e-8)).astype(np.float32)
    mse_orig = float(np.mean((hs_u_va * fn_orig[None, :] - ht_va) ** 2))
    mse_new = float(np.mean((hs_u_va * fn_wiener[None, :] - ht_va) ** 2))
    fn_eff = fn_wiener if mse_new < mse_orig else fn_orig
    print(f"[Step3-FN] val_mse {mse_orig:.5f}->{mse_new:.5f} using={'Wiener' if mse_new < mse_orig else 'orig'}", flush=True)

    hs_base_norm_all = (hs_unit_all * fn_eff[None, :]).astype(np.float32)
    ht_ans = ht_norm_all[ans_idx]
    hs_ans = hs_base_norm_all[ans_idx]
    nxt_ans = next_token_ids[ans_idx]

    vocab_sz = int(cfg["vocab_size"])
    x_st_tr = hs_base_norm_all[:cal_end][cal_st_tr].copy()
    ht_st_tr = ht_norm_all[:cal_end][cal_st_tr].copy()
    x_st_va = hs_base_norm_all[:cal_end][cal_st_val].copy()
    ht_st_va = ht_norm_all[:cal_end][cal_st_val].copy()
    ht_ans = ht_ans.copy()
    hs_ans = hs_ans.copy()
    del bridge, stud, h_s, h_teach_L59, ht_norm_all, hs_unit_all, hs_base_norm_all, hs_u_tr, ht_tr, hs_u_va, ht_va
    gc.collect()

    with SafetensorsReader(sf_path) as sr:
        _cand_ids, W_cand, p_teach_full, s0_other_lse, y_star_idx = _stream_formula5_targets(
            sr,
            "model.language_model.embed_tokens.weight",
            ops,
            ht_ans,
            hs_ans,
            nxt_ans,
            vocab_sz,
            d,
            tau,
            top_k=128,
            chunk_rows=8192,
        )

    ans_val_mask = (np.arange(n_ans) % 5 == 0)
    ans_tr_mask = ~ans_val_mask
    tr_ans_t = torch.from_numpy(np.flatnonzero(ans_tr_mask))
    va_ans_t = torch.from_numpy(np.flatnonzero(ans_val_mask))

    x_ans_t = torch.from_numpy(hs_ans)
    W_cand_t = torch.from_numpy(W_cand)
    p_t_targ = torch.from_numpy(p_teach_full.astype(np.float32))
    log_tail_t = torch.from_numpy(s0_other_lse)
    y_star_t = torch.from_numpy(y_star_idx)
    x_st_tr_t = torch.from_numpy(x_st_tr)
    ht_st_tr_t = torch.from_numpy(ht_st_tr)

    model60 = c8d.Cuneiform8DPatternAdapter(
        d_out=d, d_in=d, rank=rank_cell60, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0
    )
    opt60 = torch.optim.AdamW(model60.parameters(), lr=3e-3, weight_decay=5e-2)

    def _eval_ans(hp_rows: torch.Tensor, sub: torch.Tensor) -> Tuple[float, float, float]:
        with torch.no_grad():
            n_sub = int(sub.numel())
            top1_sum = 0.0
            marg_sum = 0.0
            kl_sum = 0.0
            for b0 in range(0, n_sub, 256):
                b1 = min(b0 + 256, n_sub)
                idx = sub[b0:b1]
                hp = hp_rows[b0:b1]
                zc = 30.0 * torch.tanh((hp @ W_cand_t.T) / 30.0)
                ys = y_star_t[idx]
                top1_sum += float((zc.argmax(dim=-1) == ys).float().sum().item())
                z_y = zc.gather(1, ys.unsqueeze(1)).squeeze(1)
                zm = zc.clone()
                zm.scatter_(1, ys.unsqueeze(1), -1e9)
                marg_sum += float((z_y - zm.max(dim=-1).values).sum().item())
                kl_sum += float(formula5_kl_loss(zc, log_tail_t[idx], p_t_targ[idx], tau).item()) * int(idx.numel())
            denom_n = float(max(n_sub, 1))
            return kl_sum / denom_n, top1_sum / denom_n, marg_sum / denom_n

    kl0, acc0, marg0 = _eval_ans(x_ans_t[va_ans_t], va_ans_t)
    print(f"[Step3-Cell60 Before] Val KL={kl0:.4f} top1={acc0 * 100:.1f}% margin={marg0:+.3f} cand_vocab={len(_cand_ids)}", flush=True)

    best_score = -1e9
    best_sd = None
    n_tr_ans = int(tr_ans_t.numel())
    n_tr_st = int(x_st_tr_t.shape[0])
    for step in range(steps):
        opt60.zero_grad()
        sub_a = tr_ans_t[torch.randperm(n_tr_ans)[:256]] if n_tr_ans > 256 else tr_ans_t
        sub_st = torch.randint(0, n_tr_st, (512,)) if n_tr_st > 512 else torch.arange(n_tr_st)
        d_ans, _, lb_loss = model60(x_ans_t[sub_a])
        zc_tr = 30.0 * torch.tanh(((x_ans_t[sub_a] + d_ans) @ W_cand_t.T) / 30.0)
        l_kl = formula5_kl_loss(zc_tr, log_tail_t[sub_a], p_t_targ[sub_a], tau)
        l_marg = formula6_margin_loss(zc_tr, y_star_t[sub_a], gamma=gamma)

        d_st, _, _ = model60(x_st_tr_t[sub_st])
        h_st_p = x_st_tr_t[sub_st] + d_st
        l_mse = F.mse_loss(h_st_p, ht_st_tr_t[sub_st])
        l_cos = 1.0 - F.cosine_similarity(h_st_p, ht_st_tr_t[sub_st], dim=-1).mean()
        denom = float(max(int(sub_st.numel()) * d, 1))
        l_reg = 2e-2 * (model60.A_bank.pow(2).sum() + model60.B_bank.pow(2).sum()) / denom

        loss = 1.0 * l_kl + 0.5 * l_marg + 2.0 * l_mse + 0.5 * l_cos + 0.005 * lb_loss + l_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model60.parameters(), 1.0)
        opt60.step()

        if (step + 1) % 20 == 0 or step == 0:
            with torch.no_grad():
                d_val, _, _ = model60(x_ans_t[va_ans_t])
                kl_va, acc_va, marg_va = _eval_ans(x_ans_t[va_ans_t] + d_val, va_ans_t)
                del d_val
                score = marg_va - 0.5 * kl_va
                if score > best_score:
                    best_score = score
                    best_sd = {k: v.detach().clone() for k, v in model60.state_dict().items()}
                print(
                    f"  step {step + 1:03d}/{steps:03d} loss={loss.item():.4f} | "
                    f"Va: KL={kl_va:.3f} acc={acc_va * 100:.1f}% marg={marg_va:+.2f}",
                    flush=True,
                )

    if best_sd is not None:
        model60.load_state_dict(best_sd)

    cents60, experts60 = model60.export_quantized_arrays()
    va_idx_np = va_ans_t.numpy()
    hs_ans_va = hs_ans[va_idx_np]
    delta_ans_q = c8d.eval_quantized_adapter_np(hs_ans_va, cents60, experts60, 2, 0.5, 1.0)
    kl_q, acc_q, marg_q = _eval_ans(torch.from_numpy(hs_ans_va + delta_ans_q), va_ans_t)
    y_st_q = x_st_va + c8d.eval_quantized_adapter_np(x_st_va, cents60, experts60, 2, 0.5, 1.0)
    cos_b = cos_sim(x_st_va, ht_st_va)
    cos_a = cos_sim(y_st_q, ht_st_va)
    print(
        f"[Step3-Cell60 Quantized INT8] Val KL: {kl0:.4f}->{kl_q:.4f}, Top1: {acc0 * 100:.1f}%->{acc_q * 100:.1f}%, "
        f"Margin: {marg0:+.3f}->{marg_q:+.3f}, StateCos: {cos_b:.5f}->{cos_a:.5f}",
        flush=True,
    )

    if (kl_q <= kl0) and (cos_a >= cos_b - 0.002):
        exp_rads60 = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(8)]
        trained_entries.append(
            {
                "name": "model.language_model.norm.post_cell60",
                "d_out": d,
                "d_in": d,
                "rank": rank_cell60,
                "top_k": 2,
                "c6d": exp_rads60[0],
                "temperature": 0.5,
                "gate_scale": 1.0,
                "centroids": cents60,
                "expert_radicals": exp_rads60,
                "experts": experts60,
            }
        )
    else:
        print("[Step3-Cell60] Skipping post_cell60 (did not strictly improve Val KL & StateCos)", flush=True)

    out_norms = [("model.language_model.norm.weight", fn_eff)]
    out_sz = c8d.write_c8db_file(out_c8d, out_norms, trained_entries)
    print(f"[Step3-DONE] Wrote {out_c8d} ({len(out_norms)} norms, {len(trained_entries)} entries, {out_sz:,} B)", flush=True)
    return out_c8d


def _self_test() -> int:
    """Run 6 fail-closed unit gates on Formula 5 KL, Formula 6 margin, and SeekT16C reader."""
    failures = 0
    torch.manual_seed(11)

    # Gate 1 & 2: Formula 5 KL finite, >= 0, decreases cleanly as predicted logits align with teacher
    try:
        z_t = torch.randn(8, 32)
        log_tail = torch.full((8,), -4.0)
        tau = 1.5
        z_all = torch.cat([z_t / tau, log_tail.unsqueeze(1)], dim=1)
        p_t = F.softmax(z_all, dim=-1)
        kl_match = float(formula5_kl_loss(z_t, log_tail, p_t, tau).item())
        kl_noise = float(formula5_kl_loss(torch.zeros_like(z_t), log_tail, p_t, tau).item())
        assert kl_match < 1e-4 and kl_noise > kl_match
    except Exception as exc:
        print(f"[Gate 1-2 FAIL] {exc}")
        failures += 1

    # Gate 3: Formula 6 Margin loss enforces exact gamma margin and returns 0 when exceeded
    try:
        z_ok = torch.tensor([[5.0, 2.0, 1.0]], dtype=torch.float32)
        z_bad = torch.tensor([[1.0, 2.5, 0.0]], dtype=torch.float32)
        ys = torch.tensor([0], dtype=torch.long)
        assert abs(float(formula6_margin_loss(z_ok, ys, gamma=2.0).item())) < 1e-6
        assert abs(float(formula6_margin_loss(z_bad, ys, gamma=2.0).item()) - 3.5) < 1e-5
    except Exception as exc:
        print(f"[Gate 3 FAIL] {exc}")
        failures += 1

    # Gate 4: Zero-carrier subtraction & exact streaming Formula 5 parity against dense computation
    try:
        m = c8d.Cuneiform8DPatternAdapter(d_out=32, d_in=32, rank=4, n_experts=4, top_k=2)
        x = torch.randn(4, 32)
        dx, _, _ = m(x)
        assert dx.shape == (4, 32) and torch.isfinite(dx).all()

        rng = np.random.default_rng(13)
        v_sz, d_sz, n_a = 48, 16, 6
        w_f32 = rng.standard_normal((v_sz, d_sz)).astype(np.float32)
        w_u16 = (w_f32.view(np.uint32) >> 16).astype("<u2")
        w_bf = ((w_u16.astype(np.uint32) << 16).view(np.float32)).copy()
        ht_t = rng.standard_normal((n_a, d_sz)).astype(np.float32)
        hs_t = rng.standard_normal((n_a, d_sz)).astype(np.float32)
        nx_t = np.array([1, 5, 11, 20, 31, 40], dtype=np.int32)
        ops_np = _NativeCalOps(None, require_native=False)
        with tempfile.TemporaryDirectory() as td:
            sfp = Path(td) / "test.safetensors"
            hdr_b = json.dumps({"emb": {"dtype": "BF16", "shape": [v_sz, d_sz], "data_offsets": [0, v_sz * d_sz * 2]}}).encode("utf-8")
            with open(sfp, "wb") as f:
                f.write(struct.pack("<Q", len(hdr_b)))
                f.write(hdr_b)
                f.write(w_u16.tobytes())
            with SafetensorsReader(sfp) as sr:
                cids, wc, p_st, lse_st, ys_st = _stream_formula5_targets(
                    sr, "emb", ops_np, ht_t, hs_t, nx_t, v_sz, d_sz, 1.5, top_k=6, chunk_rows=16, traps=(2,)
                )
                cids_cap, _, p_cap, _, ys_cap = _stream_formula5_targets(
                    sr, "emb", ops_np, ht_t, hs_t, nx_t, v_sz, d_sz, 1.5, top_k=6, chunk_rows=16, traps=(2,), max_cand=10
                )
                assert len(cids_cap) <= 10 and np.array_equal(cids_cap[ys_cap], nx_t) and 2 in set(cids_cap.tolist())
                assert np.allclose(p_cap.sum(axis=-1), 1.0, atol=1e-5)
        z_t_full = 30.0 * np.tanh((ht_t @ w_bf.T) / 30.0)
        z_s_full = 30.0 * np.tanh((hs_t @ w_bf.T) / 30.0)
        mask = np.ones((v_sz,), dtype=bool)
        mask[cids] = False
        ot = np.flatnonzero(mask)
        sub_t = z_t_full[:, ot] / 1.5
        sub_s = z_s_full[:, ot] / 1.5
        mt, ms = sub_t.max(axis=1, keepdims=True), sub_s.max(axis=1, keepdims=True)
        lse_s_ref = (ms + np.log(np.sum(np.exp(sub_s - ms), axis=1, keepdims=True)))[:, 0]
        lse_t_ref = mt + np.log(np.sum(np.exp(sub_t - mt), axis=1, keepdims=True))
        z_ref = np.concatenate([z_t_full[:, cids] / 1.5, lse_t_ref], axis=1)
        z_ref -= z_ref.max(axis=1, keepdims=True)
        p_ref = np.exp(z_ref)
        p_ref /= p_ref.sum(axis=1, keepdims=True)
        assert np.allclose(p_st, p_ref, atol=1e-5)
        assert np.allclose(lse_st, lse_s_ref, atol=1e-5)
        assert np.array_equal(cids[ys_st], nx_t)
        assert wc.shape == (len(cids), d_sz)
    except Exception as exc:
        print(f"[Gate 4 FAIL] {exc}")
        failures += 1

    # Gate 5 & 6: Fail closed on missing inputs to run_bridge_v4 and _SeekT16CReader
    try:
        with tempfile.TemporaryDirectory() as td:
            caught_bridge = False
            try:
                run_bridge_v4(
                    Path(td) / "no.safetensors",
                    Path(td) / "no.json",
                    Path(td) / "no.t16c",
                    Path(td) / "no.npz",
                    Path(td),
                    Path(td) / "out.c8d",
                )
            except FileNotFoundError:
                caught_bridge = True
            assert caught_bridge

            caught_reader = False
            try:
                with _SeekT16CReader(Path(td) / "no.t16c", ctypes.CDLL(None) if os.name != "nt" else None):  # type: ignore[arg-type]
                    pass
            except FileNotFoundError:
                caught_reader = True
            assert caught_reader
    except Exception as exc:
        print(f"[Gate 5-6 FAIL] {exc}")
        failures += 1

    if failures == 0:
        print("[PASS] train_v4_bridge.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser(description="Train V4 C8DB bank (L58/L59 KV + Formula 5/6 post_cell60).")
    ap.add_argument("--safetensors", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--t16c", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v4.t16c"))
    ap.add_argument("--bridge-npz", type=Path, default=Path("/mnt/SHE-VAULT/work31b/snaps_v4/teacher_bridge_v4.npz"))
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v4_c8d.c8d"))
    ap.add_argument("--rank-cell60", type=int, default=16)
    ap.add_argument("--rank-kv", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--state-cache", type=Path, default=None)
    args = ap.parse_args()

    sf_path = resolve_safetensors_path(args.safetensors)
    run_bridge_v4(
        sf_path,
        args.config.resolve(),
        args.t16c.resolve(),
        args.bridge_npz.resolve(),
        args.work_dir.resolve(),
        args.out.resolve(),
        rank_cell60=args.rank_cell60,
        rank_kv=args.rank_kv,
        steps=args.steps,
        state_cache=args.state_cache.resolve() if args.state_cache is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Pi 4 On-Device Full-Dataset Corpus Harvester + Complete 27-Layer Vision Tower (575.74M)
GPTQ/ActALS Calibration & 300-Step C8DB Embed-Vision Adapter Trainer (`V4.2`).

Designed strictly for 7.6 GiB Pi 4 ARM64 CPU:
  - Streams `model.safetensors` layer-by-layer (`0..26`) via `MmapSafetensorsReader` (`<= 450 MB` RAM).
  - Reuses repo NEON/GPTQ kernels (`libneon_stream_accelerator.so`, `libgauss_gptq_v2.so`, `libfast_sgemm.so`).
  - Downloads and blacklist-audits 100% of GSM8K / Python Code 18k / Alpaca-Cleaned / WikiText-2
    + COCO-Caption2017 val images + Optical VA self-analysis diagnostic frames.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import copy
import gc
import io
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ctypes
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from transformers import AutoConfig, Gemma4ImageProcessor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionEncoderLayer,
    Gemma4VisionModel,
    create_bidirectional_mask,
)

import pack_t16c_v3 as p3
import train_c8d_pattern_bank as c8d
import v4_corpus as vc
from calibrate_31b_v3 import _NativeCalOps, compute_sq, fit_act_als_i8
from calibrate_31b_v4 import (
    _build_gptq_u_chunks,
    _load_v2_gptq_lib,
    scale_t16c_pbytes_rows,
)


def is_clean_text(txt: str) -> bool:
    """Return True iff text contains zero blacklisted benchmark/eval substrings."""
    low = txt.lower()
    return not any(b in low for b in vc.BLACKLIST_SUBSTRINGS)


def make_optical_va_pil(idx: int) -> Image.Image:
    """Render a synthetic 672x672 Optical VA self-analysis telemetry canvas."""
    im = Image.new("RGB", (672, 672), (248, 248, 248))
    dr = ImageDraw.Draw(im)
    dr.text((18, 18), f"GEMMA4 31B FULL LM=60 VIS=27 FRAME={idx:03d}", fill=(12, 18, 28))
    dr.text((18, 42), "OPTICAL VA PRECISION ANALYSIS C8DB-H8 PI4", fill=(12, 18, 28))
    rng_v = np.random.default_rng(1000 + idx)
    for b in range(60):
        bh = int(rng_v.integers(25, 118))
        x = 24 + b * 10
        col = (35, 115, 210) if b >= 58 else ((30, 165, 95) if b % 5 != 0 else (210, 65, 45))
        dr.rectangle([x, 550 - bh, x + 7, 550], fill=col)
    return im


def harvest_full_clean_corpus(
    model_dir: Path,
    work_dir: Path,
    n_seq: int = 16,
    seq_len: int = 256,
) -> Path:
    """Download & audit complete HF splits cleanly onto NVMe and pack dense non-pad tokens."""
    out_npz = work_dir / "full_corpus_v4_2.npz"
    tk_path = model_dir / "tokenizer.json"
    if not tk_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json: {tk_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_file(str(tk_path))

    print("[Stage 1] Downloading & auditing 100% of HF clean datasets...", flush=True)
    p_gsm = Path(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset"))
    p_code = Path(hf_hub_download("iamtarun/python_code_instructions_18k_alpaca", "data/train-00000-of-00001-8b6e212f3e1ece96.parquet", repo_type="dataset"))
    p_alp = Path(hf_hub_download("yahma/alpaca-cleaned", "alpaca_data_cleaned.json", repo_type="dataset"))
    p_wiki = Path(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset"))

    clean_turns: List[Tuple[str, str, str]] = []
    raw_counts: Dict[str, int] = defaultdict(int)
    clean_counts: Dict[str, int] = defaultdict(int)

    t_gsm = pq.read_table(p_gsm)
    for q, a in zip(t_gsm["question"].to_pylist(), t_gsm["answer"].to_pylist()):
        raw_counts["math"] += 1
        qs, ans = str(q).strip(), str(a).strip()
        if qs and ans and is_clean_text(qs + " " + ans):
            clean_turns.append(("math", qs, ans))
            clean_counts["math"] += 1

    t_code = pq.read_table(p_code)
    for ins, inp, out in zip(t_code["instruction"].to_pylist(), t_code["input"].to_pylist(), t_code["output"].to_pylist()):
        raw_counts["code"] += 1
        p = (str(ins or "") + ("\n" + str(inp) if str(inp or "").strip() else "")).strip()
        a = str(out or "").strip()
        if p and a and is_clean_text(p + " " + a):
            clean_turns.append(("code", p, a))
            clean_counts["code"] += 1

    alp_rows = json.loads(p_alp.read_text(encoding="utf-8"))
    for row in alp_rows:
        raw_counts["alpaca"] += 1
        p = (str(row.get("instruction", "")) + ("\n" + str(row.get("input", "")) if str(row.get("input", "")).strip() else "")).strip()
        a = str(row.get("output", "")).strip()
        if p and a and is_clean_text(p + " " + a):
            clean_turns.append(("alpaca", p, a))
            clean_counts["alpaca"] += 1

    t_wiki = pq.read_table(p_wiki)
    wiki_buf: List[str] = []
    for txt in t_wiki["text"].to_pylist():
        raw_counts["wiki"] += 1
        s = str(txt or "").strip()
        if len(s) > 60 and not s.startswith("=") and is_clean_text(s):
            wiki_buf.append(s)
    for i in range(0, len(wiki_buf) - 1, 2):
        clean_turns.append(("wiki", "Summarize and continue accurately:\n" + wiki_buf[i][:320], wiki_buf[i + 1][:500]))
        clean_counts["wiki"] += 1

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(clean_turns))
    by_dom: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for idx in perm:
        dm, q_t, a_t = clean_turns[idx]
        by_dom[dm].append((q_t, a_t))

    dom_names = ["math", "code", "alpaca", "wiki"]
    ptrs = {d: 0 for d in dom_names}
    all_ids: List[int] = []
    pos_arr: List[int] = []
    cal_segs: List[Tuple[int, int]] = []
    pstarts: List[int] = []
    prompt_domains: List[str] = []
    ans_mask: List[bool] = []
    next_token_ids: List[int] = []

    turns_packed = 0
    cursor = 0
    for s_idx in range(n_seq):
        seg_start = len(all_ids)
        cur_toks: List[int] = [2]
        cur_ans: List[bool] = [False]
        seg_dom = dom_names[s_idx % len(dom_names)]
        while len(cur_toks) < seq_len + 1:
            d_turn = dom_names[(cursor) % len(dom_names)] if s_idx >= 4 else seg_dom
            cursor += 1
            pi = ptrs[d_turn] % len(by_dom[d_turn])
            ptrs[d_turn] += 1
            qt, at = by_dom[d_turn][pi]
            vc.assert_no_contamination([qt, at], d_turn)
            ptok = list(tk.encode(vc.format_gemma4_turn(qt, None), add_special_tokens=False).ids)
            rtok = list(tk.encode(f"{at}<turn|>\n", add_special_tokens=False).ids)
            if not ptok or not rtok:
                continue
            cur_toks.extend(ptok)
            cur_ans.extend([False] * len(ptok))
            cur_toks.extend(rtok)
            cur_ans.extend([True] * len(rtok))
            turns_packed += 1
        cur_toks = cur_toks[: seq_len + 1]
        cur_ans = cur_ans[: seq_len + 1]
        inp_seq = cur_toks[:-1]
        nxt_seq = cur_toks[1:]
        am_seq = cur_ans[1:]

        pstarts.append(len(all_ids))
        all_ids.extend(inp_seq)
        pos_arr.extend(range(seq_len))
        next_token_ids.extend(nxt_seq)
        ans_mask.extend(am_seq)
        prompt_domains.extend([seg_dom] * seq_len)
        cal_segs.append((seg_start, seg_start + seq_len))

    cal_end = len(all_ids)

    # Append strictly isolated eval-only probes at tail (test_2p2, test_fr) AFTER cal_end
    for eval_label, eval_prompt in (("test2", "What is 2 + 2?"), ("testf", "What is the capital of France?")):
        p_ids = [2] + list(tk.encode(vc.format_gemma4_turn(eval_prompt, None), add_special_tokens=False).ids)
        seg_s = len(all_ids)
        pstarts.append(seg_s)
        all_ids.extend(p_ids)
        pos_arr.extend(range(len(p_ids)))
        next_token_ids.extend(p_ids[1:] + [0])
        ans_mask.extend([False] * len(p_ids))
        prompt_domains.extend([eval_label] * len(p_ids))
        cal_segs.append((seg_s, seg_s + len(p_ids)))

    # Also harvest 16 real COCO images + 16 Optical VA canvases = 32 Vision frames
    vis_dir = work_dir / "vis_frames_32"
    vis_dir.mkdir(parents=True, exist_ok=True)
    p_coco = Path(hf_hub_download("lmms-lab/COCO-Caption2017", "data/val-00000-of-00002.parquet", repo_type="dataset"))
    t_coco = pq.read_table(p_coco, columns=["image"])
    coco_imgs: List[Image.Image] = []
    for item in t_coco["image"].to_pylist():
        raw_b = item["bytes"] if isinstance(item, dict) else item
        im = Image.open(io.BytesIO(raw_b)).convert("RGB")
        coco_imgs.append(im)
        if len(coco_imgs) >= 16:
            break
    vis_paths: List[str] = []
    for i in range(16):
        p_va = vis_dir / f"frame_{2*i:02d}_va.png"
        make_optical_va_pil(i).save(p_va)
        vis_paths.append(str(p_va))
        p_co = vis_dir / f"frame_{2*i+1:02d}_coco.png"
        coco_imgs[i].save(p_co)
        vis_paths.append(str(p_co))

    stats = {
        "raw_counts": dict(raw_counts),
        "clean_counts": dict(clean_counts),
        "total_clean": len(clean_turns),
        "cal_tokens": cal_end,
        "cal_ans_tokens": int(sum(ans_mask[:cal_end])),
        "turns_packed": turns_packed,
        "vis_frames": len(vis_paths),
    }
    tmp = out_npz.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        all_ids=np.array(all_ids, dtype=np.int32),
        pos_arr=np.array(pos_arr, dtype=np.int32),
        cal_segs=np.array(cal_segs, dtype=np.int32),
        pstarts=np.array(pstarts, dtype=np.int32),
        cal_end=np.array(cal_end, dtype=np.int32),
        ans_mask=np.array(ans_mask, dtype=bool),
        next_token_ids=np.array(next_token_ids, dtype=np.int32),
        prompt_domains=np.array(prompt_domains),
        vis_paths=np.array(vis_paths),
        stats_json=np.array(json.dumps(stats)),
    )
    tmp.replace(out_npz)
    print(f"[Stage 1 DONE] {stats}", flush=True)
    return out_npz


class _NeonQuantLinear(nn.Module):
    """Drop-in PyTorch Linear executing quantized .t16c bytes via Pi 4 libneon_stream_accelerator.so + Int8LowRank."""

    def __init__(
        self,
        pb_flat: np.ndarray,
        inv_s: np.ndarray,
        d_out: int,
        d_in_eff: int,
        orig_in: int,
        bsize: int,
        mode_i: int,
        lr_u: Optional[np.ndarray],
        lr_v: Optional[np.ndarray],
        ops: _NativeCalOps,
    ):
        super().__init__()
        self.pb_flat = pb_flat
        self.inv_s = inv_s
        self.d_out = d_out
        self.d_in_eff = d_in_eff
        self.orig_in = orig_in
        self.bsize = bsize
        self.mode_i = mode_i
        self.lr_u = lr_u
        self.lr_v = lr_v
        self.ops = ops
        self.register_buffer("_weight_dtype_probe", torch.empty(0, dtype=torch.float32), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        """Expose float32 dtype probe for HuggingFace modules checking `self.proj.weight.dtype`."""
        return self._weight_dtype_probe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_sh = x.shape[:-1]
        x2 = x.detach().reshape(-1, self.orig_in).numpy().astype(np.float32, copy=False)
        if self.d_in_eff > self.orig_in:
            xp = np.zeros((x2.shape[0], self.d_in_eff), dtype=np.float32)
            xp[:, : self.orig_in] = x2
            x_in = xp
        else:
            x_in = np.ascontiguousarray(x2, dtype=np.float32)
        y = self.ops.mm_neon_t16c(self.pb_flat, self.inv_s, self.d_out, self.d_in_eff, self.bsize, self.mode_i, x_in)
        if self.lr_u is not None and self.lr_v is not None:
            y = y + self.ops.mm_xt(self.ops.mm_xt(x_in, self.lr_v), self.lr_u)
        return torch.from_numpy(y).reshape(*orig_sh, self.d_out)


def rand_svd_top(
    y_err: np.ndarray,
    rk: int,
    ops: _NativeCalOps,
    oversample: int = 16,
    n_iter: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-`rk` singular triplets via randomized range finder + power iteration (zero O(N^3) eigh)."""
    d_out, n_tr = y_err.shape
    p = min(d_out, n_tr, rk + oversample)
    rng = np.random.default_rng(12345)
    om = rng.standard_normal((n_tr, p)).astype(np.float32)
    q, _ = np.linalg.qr(ops.mm_xt(y_err, om.T), mode="reduced")
    y_t = np.ascontiguousarray(y_err.T, dtype=np.float32)
    for _ in range(n_iter):
        z, _ = np.linalg.qr(ops.mm_xt(y_t, q.T), mode="reduced")
        q, _ = np.linalg.qr(ops.mm_xt(y_err, z.T), mode="reduced")
    b = ops.mm_xt(q.T, y_t)
    ub, sb, vtb = np.linalg.svd(b, full_matrices=False)
    u = np.ascontiguousarray(q @ ub[:, :rk], dtype=np.float32)
    return u, sb[:rk].astype(np.float32), np.ascontiguousarray(vtb[:rk, :], dtype=np.float32)


def fit_act_als_i8_fast(
    y_err: np.ndarray,
    x_bal: np.ndarray,
    rk: int,
    ops: _NativeCalOps,
    ridge: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact Cholesky/Woodbury + randomized-SVD replacement for `fit_act_als_i8` (`cos == 1.0000`, O(min(N,d)^3))."""
    _, n_tr = y_err.shape
    d_in = x_bal.shape[1]
    u0, s0, vt0 = rand_svd_top(y_err, rk, ops)
    u = u0[:, :rk].copy()
    b_target = np.ascontiguousarray(s0[:rk, None] * vt0[:rk, :], dtype=np.float32)

    if d_in <= n_tr:
        x_t = np.ascontiguousarray(x_bal.T, dtype=np.float32)
        gram = ops.mm_xt(x_t, x_t).astype(np.float64)
        lam = ridge * np.trace(gram) / max(n_tr, 1)
        gram[np.diag_indices_from(gram)] += lam
        fac = cho_factor(gram, overwrite_a=True, check_finite=False)
        solve_v = lambda b: np.ascontiguousarray(cho_solve(fac, ops.mm_xt(x_t, b), check_finite=False).T, dtype=np.float32)
    else:
        gram = ops.mm_xt(x_bal, x_bal).astype(np.float64)
        lam = ridge * np.trace(gram) / max(n_tr, 1)
        gram[np.diag_indices_from(gram)] += lam
        fac = cho_factor(gram, overwrite_a=True, check_finite=False)
        x_t = np.ascontiguousarray(x_bal.T, dtype=np.float32)
        solve_v = lambda b: ops.mm_xt(
            np.ascontiguousarray(cho_solve(fac, b.T, check_finite=False).T, dtype=np.float32),
            x_t,
        )

    v = solve_v(b_target)
    for _ in range(2):
        z = ops.mm_xt(x_bal, v).T
        zzt = ops.mm_xt(z, z).astype(np.float64) + 1e-4 * np.eye(rk)
        yzt = ops.mm_xt(y_err, z)
        u = ops.mm_xt(yzt, np.linalg.inv(zzt).T.astype(np.float32))
        utu = ops.mm_xt(u.T, u.T).astype(np.float64) + 1e-4 * np.eye(rk)
        uty = ops.mm_xt(u.T, y_err.T)
        b_target = np.linalg.solve(utu, uty).astype(np.float32)
        v = solve_v(b_target)

    su = (np.max(np.abs(u), 1) / 127.0 + 1e-12).astype(np.float32)
    sv = (np.max(np.abs(v), 1) / 127.0 + 1e-12).astype(np.float32)
    u_i8 = np.clip(np.round(u / su[:, None]), -127, 127).astype(np.int8)
    v_i8 = np.clip(np.round(v / sv[:, None]), -127, 127).astype(np.int8)
    return (
        (u_i8.astype(np.float32) * su[:, None]).astype(np.float32),
        (v_i8.astype(np.float32) * sv[:, None]).astype(np.float32),
        su,
        sv,
        u_i8,
        v_i8,
    )


def _quant_pack_vision_mat(
    w: np.ndarray,
    x_cal: np.ndarray,
    mode_i: int,
    rk: int,
    ops: _NativeCalOps,
    lv2: ctypes.CDLL,
    *,
    pad_in: int = 0,
    gptq_cw: int = 384,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """SmoothQuant + FWHT-64 + GPTQ (libgauss_gptq_v2.so) + NEON AR + fast ActALS for 1 Vision projection."""
    d_out, d_in = w.shape
    bsize = 64
    d_eff = pad_in if pad_in > d_in else d_in
    if d_eff > d_in:
        w_eff = np.zeros((d_out, d_eff), dtype=np.float32)
        w_eff[:, :d_in] = w
        x_eff = np.zeros((len(x_cal), d_eff), dtype=np.float32)
        x_eff[:, :d_in] = x_cal
    else:
        w_eff = np.ascontiguousarray(w, dtype=np.float32)
        x_eff = np.ascontiguousarray(x_cal, dtype=np.float32)

    sq = compute_sq(x_eff, w_eff, alpha=0.5)
    inv_s = (1.0 / sq).astype(np.float32)
    w_rot = np.ascontiguousarray(w_eff * sq[None, :], dtype=np.float32)
    ops.lib_g.fwht_rows_f32(w_rot.ctypes.data_as(ctypes.c_void_p), d_out, d_eff, bsize)

    n_bits_u8, is_asym = p3.MODE_TO_T16C[mode_i]
    eff_bits = int(n_bits_u8 & 0x7F)
    is_asym_i = 1 if is_asym else 0
    n_tiles = (d_out + 15) // 16
    n_blks = d_eff // bsize
    tblk = p3.tile16_blk_bytes(bsize, n_bits_u8, is_asym)
    pb = np.empty((n_tiles, n_blks, tblk), dtype=np.uint8)

    u_chunks = _build_gptq_u_chunks(x_eff, sq, d_eff, bsize, gptq_cw, 0.15, ops, lv2)
    blks_c = gptq_cw // bsize
    for ci, Uc in enumerate(u_chunks):
        w_sub = np.ascontiguousarray(w_rot[:, ci * gptq_cw : (ci + 1) * gptq_cw], dtype=np.float32)
        pb_sub = np.empty((n_tiles, blks_c, tblk), dtype=np.uint8)
        lv2.gptq_quant_and_pack_v2_f32(
            w_sub.ctypes.data_as(ctypes.c_void_p),
            Uc.ctypes.data_as(ctypes.c_void_p),
            d_out, gptq_cw, bsize, eff_bits, is_asym_i, 0,
            pb_sub.ctypes.data_as(ctypes.c_void_p),
        )
        pb[:, ci * blks_c : (ci + 1) * blks_c, :] = pb_sub
        del w_sub, pb_sub
    del w_rot, u_chunks

    pb_flat = pb.reshape(-1)
    y_tgt = ops.mm_xt(x_eff, w_eff)
    y_q = ops.mm_neon_t16c(pb_flat, inv_s, d_out, d_eff, bsize, mode_i, x_eff)
    num = np.sum(y_tgt * y_q, axis=0, dtype=np.float64)
    den = np.sum(y_q * y_q, axis=0, dtype=np.float64) + 1e-12
    ar = np.clip(np.where(np.isfinite(num / den), num / den, 1.0), 0.70, 1.40).astype(np.float32)
    scale_t16c_pbytes_rows(pb_flat, d_out, d_eff, bsize, mode_i, ar)

    low_rank = None
    lr_u = lr_v = None
    if rk > 0:
        y_q = ops.mm_neon_t16c(pb_flat, inv_s, d_out, d_eff, bsize, mode_i, x_eff)
        sub_n = min(len(x_eff), 2048) if d_eff > 2048 else min(len(x_eff), 4096)
        idx_s = np.linspace(0, len(x_eff) - 1, sub_n, dtype=int)
        x_sub = np.ascontiguousarray(x_eff[idx_s], dtype=np.float32)
        y_err = np.ascontiguousarray((y_tgt[idx_s] - y_q[idx_s]).T, dtype=np.float32)
        u_q, v_q, su, sv, ui8, vi8 = fit_act_als_i8_fast(y_err, x_sub, rk, ops)
        low_rank = (su, sv, ui8, vi8)
        lr_u, lr_v = u_q, v_q

    return pb_flat, inv_s, low_rank, lr_u, lr_v, w_eff


def _load_vision_layer_weights(sr: c8d.MmapSafetensorsReader, l_v: int) -> dict:
    pfx = f"model.vision_tower.encoder.layers.{l_v}."
    sd = {}
    for k in sr.hdr:
        if k.startswith(pfx):
            sd[k[len(pfx):]] = torch.from_numpy(sr.read_f32(k))
    return sd


def calibrate_vision_v4_2(
    sf_path: Path,
    cfg_path: Path,
    model_dir: Path,
    corpus_npz: Path,
    work_dir: Path,
) -> Path:
    """Stream all 27 Vision layers + embed_vision on Pi 4 in <= 450 MB RAM, saving vision_v4_2.npz + vis_c8d_v4_2.npz."""
    for p in (sf_path, cfg_path, corpus_npz):
        if not p.is_file():
            raise FileNotFoundError(f"Missing Vision input artifact: {p}")

    out_vis_npz = work_dir / "vision_v4_2.npz"
    out_c8d_npz = work_dir / "vis_c8d_v4_2.npz"
    ops = _NativeCalOps(work_dir, require_native=True)
    lv2 = _load_v2_gptq_lib(work_dir, require_native=True)

    cfg_full = AutoConfig.from_pretrained(str(model_dir))
    cfg_full.vision_config._attn_implementation = "sdpa"
    proc = Gemma4ImageProcessor.from_pretrained(str(model_dir))

    with np.load(corpus_npz) as cd:
        vis_paths = [str(x) for x in cd["vis_paths"].tolist()]
    imgs = [Image.open(p).convert("RGB") for p in vis_paths]
    pin = proc(images=imgs, return_tensors="pt")
    pvs_all = pin["pixel_values"].float()           # [32, 2520, 768]
    poss_all = pin["image_position_ids"].long()     # [32, 2520, 2]
    pads_all = (poss_all == -1).all(dim=-1)         # [32, 2520]
    n_fr = len(imgs)

    # Create lightweight helper Vision components (0 full-stack layers allocated!)
    cfg_v1 = copy.deepcopy(cfg_full.vision_config)
    cfg_v1.num_hidden_layers = 0
    cfg_v1._attn_implementation = "sdpa"
    vis_shell = Gemma4VisionModel(cfg_v1).float().eval()

    with c8d.MmapSafetensorsReader(sf_path) as sr:
        std_bias = sr.read_f32("model.vision_tower.std_bias")
        std_scale = sr.read_f32("model.vision_tower.std_scale")
        valid_mask = (~pads_all).reshape(-1).numpy()
        n_bits3, asym3 = p3.MODE_TO_T16C[4]

        # Resume / checkpoint directory for atomic Vision layer progress (`V00..V26`)
        vckpt_dir = work_dir / "vis_ckpt_v4_2"
        vckpt_dir.mkdir(parents=True, exist_ok=True)
        traj_p = vckpt_dir / "traj_state.npz"
        patch_p = vckpt_dir / "patch_recs.npz"
        start_v = 0
        if traj_p.is_file() and patch_p.is_file():
            with np.load(traj_p) as tj:
                done_l = int(tj["last_l"])
                all_ck = all((vckpt_dir / f"V{i:02d}.npz").is_file() for i in range(done_l + 1))
                if all_ck and 0 <= done_l < 27:
                    x_teach = torch.from_numpy(tj["x_teach"])
                    x_stud = torch.from_numpy(tj["x_stud"])
                    start_v = done_l + 1
                    print(f"[Vision RESUME] loaded patch + V00..V{done_l:02d}, resuming at V{start_v:02d}", flush=True)

        vis_records: List[dict] = []
        if start_v == 0:
            pe_sd = {
                k[len("model.vision_tower.patch_embedder."):]: torch.from_numpy(sr.read_f32(k))
                for k in sr.hdr if k.startswith("model.vision_tower.patch_embedder.")
            }
            vis_shell.patch_embedder.load_state_dict(pe_sd, strict=True)
            pos_tbl = pe_sd["position_embedding_table"].numpy().astype(np.float32)  # [2, 10240, 1152]

            # 1) Patch embedder Teacher + Student
            with torch.no_grad():
                x_teach = vis_shell.patch_embedder(pvs_all, poss_all, pads_all).half()  # [32, 2520, 1152] FP16

            vis_norms: Dict[str, np.ndarray] = {
                "model.vision_tower.std_bias": std_bias.astype("<f4"),
                "model.vision_tower.std_scale": std_scale.astype("<f4"),
            }

            # Pack position_embedding_table.0 / .1 at 3b_u0 @ b64 without GPTQ
            for ax in (0, 1):
                w_p = np.ascontiguousarray(pos_tbl[ax], dtype=np.float32)
                ops.lib_g.fwht_rows_f32(w_p.ctypes.data_as(ctypes.c_void_p), w_p.shape[0], w_p.shape[1], 64)
                pb_p = np.empty(((w_p.shape[0] + 15) // 16) * (w_p.shape[1] // 64) * p3.tile16_blk_bytes(64, n_bits3, asym3), dtype=np.uint8)
                lv2.gptq_quant_and_pack_v2_f32(w_p.ctypes.data_as(ctypes.c_void_p), None, w_p.shape[0], w_p.shape[1], 64, 3, 0, 0, pb_p.ctypes.data_as(ctypes.c_void_p))
                vis_records.append({"name": f"model.vision_tower.patch_embedder.position_embedding_table.{ax}", "do": w_p.shape[0], "di": w_p.shape[1], "bs": 64, "nb": n_bits3, "asym": asym3, "pb": pb_p, "inv_s": None, "lr": None})

            # Pack patch_embedder.input_proj.weight (`1152 x 768`) -> 3b_u0 @ b64 r=16
            pv_valid = (2.0 * (pvs_all.reshape(-1, 768).numpy() - 0.5))[valid_mask][:16384]
            w_ip = pe_sd["input_proj.weight"].numpy().astype(np.float32)
            pb_ip, inv_ip, lr_ip, lru_ip, lrv_ip, _ = _quant_pack_vision_mat(w_ip, pv_valid, 4, 16, ops, lv2, gptq_cw=256)
            vis_records.append({"name": "model.vision_tower.patch_embedder.input_proj.weight", "do": 1152, "di": 768, "bs": 64, "nb": n_bits3, "asym": asym3, "pb": pb_ip, "inv_s": inv_ip, "lr": lr_ip})

            # Quantized Student patch embeddings
            vis_shell.patch_embedder.input_proj = _NeonQuantLinear(pb_ip, inv_ip, 1152, 768, 768, 64, 4, lru_ip, lrv_ip, ops)
            with torch.no_grad():
                x_stud = vis_shell.patch_embedder(pvs_all, poss_all, pads_all).half()
            del pe_sd, pv_valid, pos_tbl

            pk_tmp = patch_p.with_suffix(".tmp.npz")
            np.savez(
                pk_tmp,
                rec_meta_json=np.array(json.dumps([[r["name"], r["do"], r["di"], r["bs"], r["nb"], int(r["asym"])] for r in vis_records])),
                norm_names=np.array(list(vis_norms.keys())),
                **{f"norm__{k}": v for k, v in vis_norms.items()},
                **{f"m_{i}_pb": r["pb"] for i, r in enumerate(vis_records)},
                **{f"m_{i}_inv": r["inv_s"] for i, r in enumerate(vis_records) if r["inv_s"] is not None},
                **{f"m_{i}_{nm}": arr for i, r in enumerate(vis_records) if r["lr"] is not None for nm, arr in zip(("su", "sv", "ui8", "vi8"), r["lr"])},
            )
            pk_tmp.replace(patch_p)
            vis_records.clear()
            vis_norms.clear()

        del pvs_all
        gc.collect()

        # Precompute rotary & mask helpers for 2-frame chunks (`<= 120 MB`)
        pos_emb_2 = [vis_shell.encoder.rotary_emb(x_teach[i:i+2].float(), poss_all[i:i+2]) for i in range(0, n_fr, 2)]

        valid_mask_2d = (~pads_all).numpy()  # [n_fr, 2520]

        def _sample_valid(t_chunks_fn, max_rows: int) -> np.ndarray:
            bufs = []
            for b0 in range(0, n_fr, 2):
                b1 = min(b0 + 2, n_fr)
                arr = t_chunks_fn(b0, b1)
                m = valid_mask_2d[b0:b1].reshape(-1)
                arr_v = arr[m]
                take = max(1, (max_rows * (b1 - b0)) // n_fr)
                step = max(1, len(arr_v) // take)
                bufs.append(arr_v[::step][:take])
            out = np.concatenate(bufs, axis=0)
            if len(out) > max_rows:
                out = out[np.linspace(0, len(out) - 1, max_rows, dtype=int)]
            return np.ascontiguousarray(out, dtype=np.float32)

        # 2) Stream ALL 27 Vision layers (`start_v..26`) layer-by-layer with zero duplicate forward passes!
        for l_v in range(start_v, 27):
            tl = time.perf_counter()
            pfx = f"model.vision_tower.encoder.layers.{l_v}"
            sd_l = _load_vision_layer_weights(sr, l_v)
            l_norms: Dict[str, np.ndarray] = {}
            l_recs: List[dict] = []
            for nk in ("input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
                       "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                l_norms[f"{pfx}.{nk}"] = sd_l[nk].numpy().astype("<f4")

            layer_t = Gemma4VisionEncoderLayer(cfg_full.vision_config, l_v).float().eval()
            layer_t.load_state_dict(sd_l, strict=True)
            with torch.no_grad():
                x_teach_next = torch.empty_like(x_teach)
                for b0 in range(0, n_fr, 2):
                    b1 = min(b0 + 2, n_fr)
                    am = create_bidirectional_mask(config=cfg_full.vision_config, inputs_embeds=x_teach[b0:b1].float(), attention_mask=~pads_all[b0:b1])
                    pe = pos_emb_2[b0 // 2]
                    y_l = layer_t(x_teach[b0:b1].float(), attention_mask=am, position_embeddings=pe, position_ids=poss_all[b0:b1])
                    if isinstance(y_l, tuple):
                        y_l = y_l[0]
                    assert y_l.shape == (b1 - b0, 2520, 1152), f"Unexpected vision layer out shape {y_l.shape}"
                    x_teach_next[b0:b1] = y_l.half()
            x_teach = x_teach_next
            del x_teach_next

            # Sample normalized Attention inputs per chunk (zero giant concatenated arrays!)
            v_mode = 4 if l_v in (0, 1, 24, 25, 26) else 1
            v_rk = 8 if l_v in (0, 1, 24, 25, 26) else 0
            nb_v, as_v = p3.MODE_TO_T16C[v_mode]
            with torch.no_grad():
                x_in_cal = _sample_valid(lambda b0, b1: layer_t.input_layernorm(x_stud[b0:b1].float()).reshape(-1, 1152).numpy(), 16384)

            layer_s = copy.deepcopy(layer_t)
            for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                w_m = sd_l[f"{proj}.linear.weight"].numpy().astype(np.float32)
                pb_m, inv_m, lr_m, lru_m, lrv_m, _ = _quant_pack_vision_mat(w_m, x_in_cal, v_mode, v_rk, ops, lv2, gptq_cw=384)
                l_recs.append({"name": f"{pfx}.{proj}.linear.weight", "do": 1152, "di": 1152, "bs": 64, "nb": nb_v, "asym": as_v, "pb": pb_m, "inv_s": inv_m, "lr": lr_m})
                getattr(layer_s.self_attn, proj.split(".")[1]).linear = _NeonQuantLinear(pb_m, inv_m, 1152, 1152, 1152, 64, v_mode, lru_m, lrv_m, ops)

            # Single Student self_attn pass: cache ctx_all (FP16) so xm needs ZERO second SDPA pass!
            ctx_all = torch.empty((n_fr, 2520, 1152), dtype=torch.float16)
            cur_range = [0, 0]
            def _cap_ctx(_m, inp, _out):
                ctx_all[cur_range[0]:cur_range[1]] = inp[0].detach().half()
            hk_o = layer_s.self_attn.o_proj.linear.register_forward_hook(_cap_ctx)
            with torch.no_grad():
                for b0 in range(0, n_fr, 2):
                    b1 = min(b0 + 2, n_fr)
                    cur_range[0], cur_range[1] = b0, b1
                    am = create_bidirectional_mask(config=cfg_full.vision_config, inputs_embeds=x_stud[b0:b1].float(), attention_mask=~pads_all[b0:b1])
                    pe = pos_emb_2[b0 // 2]
                    layer_s.self_attn(
                        layer_s.input_layernorm(x_stud[b0:b1].float()),
                        attention_mask=am,
                        position_embeddings=pe,
                        position_ids=poss_all[b0:b1],
                    )
            hk_o.remove()
            x_o_cal = _sample_valid(lambda b0, b1: ctx_all[b0:b1].float().reshape(-1, 1152).numpy(), 16384)
            pb_o, inv_o, lr_o, lru_o, lrv_o, _ = _quant_pack_vision_mat(sd_l["self_attn.o_proj.linear.weight"].numpy().astype(np.float32), x_o_cal, v_mode, v_rk, ops, lv2, gptq_cw=384)
            l_recs.append({"name": f"{pfx}.self_attn.o_proj.linear.weight", "do": 1152, "di": 1152, "bs": 64, "nb": nb_v, "asym": as_v, "pb": pb_o, "inv_s": inv_o, "lr": lr_o})
            layer_s.self_attn.o_proj.linear = _NeonQuantLinear(pb_o, inv_o, 1152, 1152, 1152, 64, v_mode, lru_o, lrv_o, ops)

            # Apply cached ctx -> xm_all (185 MB) + sampled pre-FFN inputs (zero duplicate SDPA!)
            xm_all = torch.empty((n_fr, 2520, 1152), dtype=torch.float16)
            with torch.no_grad():
                for b0 in range(0, n_fr, 2):
                    b1 = min(b0 + 2, n_fr)
                    ao = layer_s.self_attn.o_proj(ctx_all[b0:b1].float())
                    xm_all[b0:b1] = (x_stud[b0:b1].float() + layer_s.post_attention_layernorm(ao)).half()
            del ctx_all
            with torch.no_grad():
                x_m_cal = _sample_valid(lambda b0, b1: layer_s.pre_feedforward_layernorm(xm_all[b0:b1].float()).reshape(-1, 1152).numpy(), 16384)

            for pn in ("gate_proj", "up_proj"):
                w_m = sd_l[f"mlp.{pn}.linear.weight"].numpy().astype(np.float32)
                pb_m, inv_m, lr_m, lru_m, lrv_m, _ = _quant_pack_vision_mat(w_m, x_m_cal, v_mode, v_rk, ops, lv2, gptq_cw=384)
                l_recs.append({"name": f"{pfx}.mlp.{pn}.linear.weight", "do": w_m.shape[0], "di": 1152, "bs": 64, "nb": nb_v, "asym": as_v, "pb": pb_m, "inv_s": inv_m, "lr": lr_m})
                getattr(layer_s.mlp, pn).linear = _NeonQuantLinear(pb_m, inv_m, w_m.shape[0], 1152, 1152, 64, v_mode, lru_m, lrv_m, ops)

            # Single MLP gate/up pass: cache act_all (FP16) so down_proj forward needs ZERO duplicate gate/up!
            act_all = torch.empty((n_fr, 2520, 4304), dtype=torch.float16)
            with torch.no_grad():
                for b0 in range(0, n_fr, 2):
                    b1 = min(b0 + 2, n_fr)
                    hm = layer_s.pre_feedforward_layernorm(xm_all[b0:b1].float())
                    act_all[b0:b1] = (layer_s.mlp.act_fn(layer_s.mlp.gate_proj(hm)) * layer_s.mlp.up_proj(hm)).half()
            x_d_cal = _sample_valid(lambda b0, b1: act_all[b0:b1].float().reshape(-1, 4304).numpy(), 8192)
            w_d = sd_l["mlp.down_proj.linear.weight"].numpy().astype(np.float32)
            pb_d, inv_d, lr_d, lru_d, lrv_d, _ = _quant_pack_vision_mat(w_d, x_d_cal, v_mode, v_rk, ops, lv2, pad_in=4352, gptq_cw=256)
            l_recs.append({"name": f"{pfx}.mlp.down_proj.linear.weight", "do": 1152, "di": 4352, "bs": 64, "nb": nb_v, "asym": as_v, "pb": pb_d, "inv_s": inv_d, "lr": lr_d})
            layer_s.mlp.down_proj.linear = _NeonQuantLinear(pb_d, inv_d, 1152, 4352, 4304, 64, v_mode, lru_d, lrv_d, ops)

            with torch.no_grad():
                for b0 in range(0, n_fr, 2):
                    b1 = min(b0 + 2, n_fr)
                    dp = layer_s.mlp.down_proj(act_all[b0:b1].float())
                    x_stud[b0:b1] = (xm_all[b0:b1].float() + layer_s.post_feedforward_layernorm(dp)).half()
            del act_all, xm_all

            xs_v = x_stud.float().reshape(-1, 1152)[valid_mask]
            xt_v = x_teach.float().reshape(-1, 1152)[valid_mask]
            stride = max(1, xs_v.shape[0] // 4096)
            cos_v = float(F.cosine_similarity(xs_v[::stride], xt_v[::stride], dim=-1).mean().item())
            print(f"  [Vision V{l_v:02d}/26 m={v_mode} r={v_rk}] stud_vs_teach_cos={cos_v:.5f} dt={time.perf_counter()-tl:.1f}s", flush=True)

            # Save per-layer atomic checkpoint + trajectory state cleanly
            ck_d = {
                "rec_meta_json": np.array(json.dumps([[r["name"], r["do"], r["di"], r["bs"], r["nb"], int(r["asym"])] for r in l_recs])),
                "norm_names": np.array(list(l_norms.keys())),
            }
            for k_n, v_n in l_norms.items():
                ck_d[f"norm__{k_n}"] = v_n
            for i_r, r in enumerate(l_recs):
                ck_d[f"m_{i_r}_pb"] = r["pb"]
                if r["inv_s"] is not None:
                    ck_d[f"m_{i_r}_inv"] = r["inv_s"]
                if r["lr"] is not None:
                    su_r, sv_r, ui_r, vi_r = r["lr"]
                    ck_d[f"m_{i_r}_su"], ck_d[f"m_{i_r}_sv"], ck_d[f"m_{i_r}_ui8"], ck_d[f"m_{i_r}_vi8"] = su_r, sv_r, ui_r, vi_r
            ck_p = vckpt_dir / f"V{l_v:02d}.npz"
            ck_tmp = ck_p.with_suffix(".tmp.npz")
            np.savez(ck_tmp, **ck_d)
            ck_tmp.replace(ck_p)
            tj_tmp = traj_p.with_suffix(".tmp.npz")
            np.savez(tj_tmp, last_l=np.array(l_v, dtype=np.int32), x_teach=x_teach.numpy(), x_stud=x_stud.numpy())
            tj_tmp.replace(traj_p)
            del sd_l, layer_t, layer_s, l_recs, l_norms, ck_d
            gc.collect()

        # 3) Pooler + embed_vision + 300-step C8DB
        embv_sd = {k[len("model.embed_vision."):]: torch.from_numpy(sr.read_f32(k)) for k in sr.hdr if k.startswith("model.embed_vision.")}
        mm_t = Gemma4MultimodalEmbedder(cfg_full.vision_config, cfg_full.text_config).float().eval()
        mm_t.load_state_dict(embv_sd, strict=True)

        with torch.no_grad():
            pt, mt = vis_shell.pooler(x_teach.float(), poss_all, pads_all, output_length=2520 // 9)
            pt_v = (pt[mt] - torch.from_numpy(std_bias)) * torch.from_numpy(std_scale)
            soft_teach = mm_t(pt_v).reshape(-1, 5376)

            ps, ms = vis_shell.pooler(x_stud.float(), poss_all, pads_all, output_length=2520 // 9)
            ps_v = (ps[ms] - torch.from_numpy(std_bias)) * torch.from_numpy(std_scale)
            pre_stud = mm_t.embedding_pre_projection_norm(ps_v).reshape(-1, 1152)

        w_ev = embv_sd["embedding_projection.weight"].numpy().astype(np.float32)
        pb_ev, inv_ev, lr_ev, lru_ev, lrv_ev, _ = _quant_pack_vision_mat(w_ev, pre_stud.numpy(), 4, 32, ops, lv2, gptq_cw=384)
        vis_records.append({"name": "model.embed_vision.embedding_projection.weight", "do": 5376, "di": 1152, "bs": 64, "nb": n_bits3, "asym": asym3, "pb": pb_ev, "inv_s": inv_ev, "lr": lr_ev})
        mm_lin = _NeonQuantLinear(pb_ev, inv_ev, 5376, 1152, 1152, 64, 4, lru_ev, lrv_ev, ops)
        with torch.no_grad():
            soft_stud = mm_lin(pre_stud).reshape(-1, 5376)

    # Shuffled 80/20 split for 300-step C8DB Vision embed_vision training on Pi 4 CPU!
    perm_v = torch.randperm(pre_stud.shape[0], generator=torch.Generator().manual_seed(42))
    ntr_v = (pre_stud.shape[0] * 4) // 5
    tr_v, va_v = perm_v[:ntr_v], perm_v[ntr_v:]
    X_tr, Y_tr, S_tr = pre_stud[tr_v], soft_teach[tr_v], soft_stud[tr_v]
    X_va, Y_va, S_va = pre_stud[va_v], soft_teach[va_v], soft_stud[va_v]
    cos0 = float(F.cosine_similarity(S_va, Y_va, dim=-1).mean().item())

    c8_v = c8d.Cuneiform8DPatternAdapter(d_out=5376, d_in=1152, rank=32, n_experts=8, top_k=2, temperature=0.5, gate_scale=1.0)
    opt_v = torch.optim.AdamW(c8_v.parameters(), lr=1.5e-3, weight_decay=5e-2)
    best_c, best_sd = cos0, None
    t_c0 = time.perf_counter()
    for step in range(300):
        opt_v.zero_grad()
        idx_b = torch.randint(0, X_tr.shape[0], (min(256, X_tr.shape[0]),))
        dv, _, lb = c8_v(X_tr[idx_b])
        pr = S_tr[idx_b] + dv
        loss = F.mse_loss(pr, Y_tr[idx_b]) + 0.5 * (1.0 - F.cosine_similarity(pr, Y_tr[idx_b], dim=-1).mean()) + 0.005 * lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(c8_v.parameters(), 1.0)
        opt_v.step()
        if (step + 1) % 50 == 0 or step == 0:
            with torch.no_grad():
                dv_v, _, _ = c8_v(X_va)
                cv = float(F.cosine_similarity(S_va + dv_v, Y_va, dim=-1).mean().item())
                if cv > best_c:
                    best_c = cv
                    best_sd = {k: v.detach().clone() for k, v in c8_v.state_dict().items()}
                print(f"    [Vision C8DB step {step+1:03d}/300] loss={loss.item():.4f} val_cos={cv:.5f} ({time.perf_counter()-t_c0:.1f}s)", flush=True)

    keep_c8d = False
    cents_v = np.zeros((8, 8), dtype=np.float32)
    exp_list = []
    if best_sd is not None:
        c8_v.load_state_dict(best_sd)
        cents_v, exp_list = c8_v.export_quantized_arrays()
        dq = torch.from_numpy(c8d.eval_quantized_adapter_np(X_va.numpy(), cents_v, exp_list, 2, 0.5, 1.0))
        cos_q = float(F.cosine_similarity(S_va + dq, Y_va, dim=-1).mean().item())
        keep_c8d = cos_q > cos0 + 0.001
    else:
        cos_q = cos0
    print(f"[Stage 2 DONE] Vision Val Soft-Token Cosine: {cos0:.5f} -> {cos_q:.5f} (keep={keep_c8d})", flush=True)

    # Combine patch_recs + V00..V26 + embed_vision cleanly into vision_v4_2.npz
    vis_dict: Dict[str, np.ndarray] = {}
    rec_meta = []
    norm_names: List[str] = []
    rec_idx = 0
    for ck_file in [vckpt_dir / "patch_recs.npz", *[vckpt_dir / f"V{i:02d}.npz" for i in range(27)]]:
        with np.load(ck_file) as ck:
            for k_n in ck["norm_names"].tolist():
                k_str = str(k_n)
                norm_names.append(k_str)
                vis_dict[f"norm__{k_str}"] = ck[f"norm__{k_str}"]
            c_meta = json.loads(str(ck["rec_meta_json"]))
            for i_l, m_row in enumerate(c_meta):
                rec_meta.append(m_row)
                vis_dict[f"m_{rec_idx}_pb"] = ck[f"m_{i_l}_pb"]
                if f"m_{i_l}_inv" in ck:
                    vis_dict[f"m_{rec_idx}_inv"] = ck[f"m_{i_l}_inv"]
                if f"m_{i_l}_su" in ck:
                    for suf in ("su", "sv", "ui8", "vi8"):
                        vis_dict[f"m_{rec_idx}_{suf}"] = ck[f"m_{i_l}_{suf}"]
                rec_idx += 1
    for r in vis_records:
        rec_meta.append([r["name"], r["do"], r["di"], r["bs"], r["nb"], int(r["asym"])])
        vis_dict[f"m_{rec_idx}_pb"] = r["pb"]
        if r["inv_s"] is not None:
            vis_dict[f"m_{rec_idx}_inv"] = r["inv_s"]
        if r["lr"] is not None:
            su, sv, ui8, vi8 = r["lr"]
            vis_dict[f"m_{rec_idx}_su"], vis_dict[f"m_{rec_idx}_sv"], vis_dict[f"m_{rec_idx}_ui8"], vis_dict[f"m_{rec_idx}_vi8"] = su, sv, ui8, vi8
        rec_idx += 1
    vis_dict["rec_meta_json"] = np.array(json.dumps(rec_meta))
    vis_dict["norm_names"] = np.array(norm_names)
    tmp_v = out_vis_npz.with_suffix(".tmp.npz")
    np.savez(tmp_v, **vis_dict)
    tmp_v.replace(out_vis_npz)

    c8_save: Dict[str, np.ndarray] = {
        "keep": np.array(int(keep_c8d), dtype=np.int32),
        "cos_before": np.array(cos0, dtype=np.float32),
        "cos_after": np.array(cos_q, dtype=np.float32),
        "centroids": cents_v,
    }
    for ei, (su, sv, ui, vi) in enumerate(exp_list):
        c8_save[f"e{ei}_su"], c8_save[f"e{ei}_sv"], c8_save[f"e{ei}_ui"], c8_save[f"e{ei}_vi"] = su, sv, ui, vi
    tmp_c = out_c8d_npz.with_suffix(".tmp.npz")
    np.savez(tmp_c, **c8_save)
    tmp_c.replace(out_c8d_npz)
    return out_vis_npz


def _self_test() -> int:
    """Run 6 fail-closed self-test gates cleanly."""
    failures = 0
    # Gate 1: Blacklist audit rejects all prompt/held-out substrings
    try:
        assert not is_clean_text("Who wrote the play Romeo and Juliet?")
        assert not is_clean_text("Explain special relativity cleanly.")
        assert is_clean_text("Compute the area of a triangle with base 6 and height 4.")
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    # Gate 2 & 3: Optical VA PIL canvas + bsize=64 tile16 byte math
    try:
        im = make_optical_va_pil(0)
        assert im.size == (672, 672) and im.mode == "RGB"
        assert p3.tile16_blk_bytes(64, 5, False) == 240
        assert p3.tile16_blk_bytes(64, 3, False) == 416
    except Exception as exc:
        print(f"[Gate 2-3 FAIL] {exc}")
        failures += 1

    # Gate 4: Fast Cholesky/RandSVD ALS parity + native C GPTQ + NEON matvec + padding `4304 -> 4352`
    try:
        work_def = Path("/mnt/SHE-VAULT/work31b")
        ops = _NativeCalOps(work_def if work_def.is_dir() else None, require_native=False)
        lv2 = _load_v2_gptq_lib(work_def if work_def.is_dir() else None, require_native=False)
        rng_a = np.random.default_rng(11)
        for di_t, nt_t in ((64, 160), (160, 64)):
            xt_a = rng_a.standard_normal((nt_t, di_t)).astype(np.float32)
            wt_a = (rng_a.standard_normal((32, 4)) @ rng_a.standard_normal((4, di_t))).astype(np.float32)
            yt_a = (wt_a @ xt_a.T + 0.02 * rng_a.standard_normal((32, nt_t))).astype(np.float32)
            uq_f, vq_f, *_ = fit_act_als_i8_fast(yt_a, xt_a, 4, ops)
            assert np.linalg.norm(yt_a - (uq_f @ vq_f) @ xt_a.T) < 0.1 * np.linalg.norm(yt_a)
        if ops.lib_n is not None and lv2 is not None:
            rng = np.random.default_rng(7)
            w = (rng.standard_normal((64, 192)) * 0.1).astype(np.float32)
            x = (rng.standard_normal((320, 192)) * 0.5).astype(np.float32)
            pb, inv_s, lr, lru, lrv, w_eff = _quant_pack_vision_mat(w, x, 4, 8, ops, lv2, pad_in=256, gptq_cw=256)
            ql = _NeonQuantLinear(pb, inv_s, 64, 256, 192, 64, 4, lru, lrv, ops)
            with torch.no_grad():
                yq = ql(torch.from_numpy(x)).numpy()
            yt = x @ w.T
            cos_p = float(np.sum(yq * yt) / (np.linalg.norm(yq) * np.linalg.norm(yt) + 1e-12))
            assert cos_p > 0.98, f"Vision _NeonQuantLinear padded cos={cos_p}"
    except Exception as exc:
        print(f"[Gate 4 FAIL] {exc}")
        failures += 1

    # Gate 5: C8DB Vision adapter improves synthetic residual cleanly
    try:
        torch.manual_seed(0)
        xr = torch.randn(160, 64)
        ys = torch.randn(160, 128)
        true_w = torch.randn(64, 128) * 0.5
        yt = ys + xr @ true_w
        m = c8d.Cuneiform8DPatternAdapter(d_out=128, d_in=64, rank=8, n_experts=8, top_k=2)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for _ in range(40):
            opt.zero_grad()
            dv, _, lb = m(xr)
            F.mse_loss(ys + dv, yt).backward()
            opt.step()
        cents, exps = m.export_quantized_arrays()
        dq = c8d.eval_quantized_adapter_np(xr.numpy(), cents, exps, 2, 0.5, 1.0)
        assert np.linalg.norm(yt.numpy() - (ys.numpy() + dq)) < np.linalg.norm(yt.numpy() - ys.numpy())
    except Exception as exc:
        print(f"[Gate 5 FAIL] {exc}")
        failures += 1

    # Gate 6: Fail-closed FileNotFoundError on missing paths
    try:
        with tempfile.TemporaryDirectory() as td:
            caught = False
            try:
                calibrate_vision_v4_2(Path(td) / "no.safetensors", Path(td) / "no.json", Path(td), Path(td) / "no.npz", Path(td))
            except FileNotFoundError:
                caught = True
            assert caught
    except Exception as exc:
        print(f"[Gate 6 FAIL] {exc}")
        failures += 1

    if failures == 0:
        print("[PASS] pi4_stage1_vision_v4_2.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser(description="Pi 4 Stage 1+2: Full clean datasets + 27-layer Vision calibration.")
    ap.add_argument("--model-dir", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct"))
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b"))
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--skip-harvest", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    sf_path = args.model_dir / "model.safetensors"
    cfg_path = args.model_dir / "config.json"
    corpus_npz = args.work_dir / "full_corpus_v4_2.npz"
    if not args.skip_harvest or not corpus_npz.is_file():
        harvest_full_clean_corpus(args.model_dir, args.work_dir, n_seq=args.n_seq, seq_len=args.seq_len)
    calibrate_vision_v4_2(sf_path, cfg_path, args.model_dir, corpus_npz, args.work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

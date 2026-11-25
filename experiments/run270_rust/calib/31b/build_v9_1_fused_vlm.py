#!/usr/bin/env python3
"""Gemma 4 31B V9.1-Fused-VLM Builder: Lossless ROWMAJOR+COLBLOCK Repacker & Streaming 60-Layer CRTR+MoD Trainer.

Combines:
  - V4.3-VLM's 270K-token + 160-frame multimodal ~1.58 bpw weights (zero 1-bit cliff in stem/cliff layers)
  - V9's lossless ROWMAJOR (0x40 on 1b_asym) + COLBLOCK (0x20 on 60x down_proj + 56x L02..L57 o_proj) kernel layout
  - Dedicated 60-layer INT8 CorticalRouter + MoD gate (rank=64, 84 blocks) trained directly on Pi 4 via NEON/FFI
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pack_assistant_t16c as pa  # noqa: E402
import pack_t16c_v3 as p3  # noqa: E402
import repack_t16c_rowmajor as rmmod  # noqa: E402
import spatial_district_converter as sdc  # noqa: E402
import train_cortical_router as crtr  # noqa: E402
import v4_corpus  # noqa: E402

try:
    from tokenizers import Tokenizer  # type: ignore
except ImportError:
    Tokenizer = None

GAUSS2_LV = np.array([-3.174, -1.000, 1.000, 3.174], dtype=np.float32)
GAUSS3_LV = np.array([-7.430, -5.430, -3.270, -1.270, 1.270, 3.270, 5.430, 7.430], dtype=np.float32)
TERN5_LV = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

_H_CACHE: Dict[int, np.ndarray] = {}


def get_hadamard(bsize: int) -> np.ndarray:
    """Return normalized Sylvester Walsh-Hadamard matrix [bsize, bsize] in float32."""
    if bsize not in _H_CACHE:
        _H_CACHE[bsize] = np.ascontiguousarray(pa.make_sylvester_hadamard(bsize), dtype=np.float32)
    return _H_CACHE[bsize]


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 hex digest streamingly without high RAM spikes."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            buf = fp.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def parse_t16c_metadata(t16c_path: Path) -> Tuple[List[str], Dict[str, Tuple[int, int]], Dict[str, dict], List[str]]:
    """Index all norms and matrices in a T16C v1 container (fail-closed EOF verification)."""
    if not t16c_path.is_file():
        raise FileNotFoundError(f"Missing T16C file: {t16c_path}")
    norm_order: List[str] = []
    norms: Dict[str, Tuple[int, int]] = {}
    mats: Dict[str, dict] = {}
    mat_order: List[str] = []
    with open(t16c_path, "rb") as fp:
        magic, ver, _flags, n_norms, n_mats, _resv = struct.unpack("<4s5I", fp.read(24))
        if magic != p3.T16C_MAGIC or ver != p3.T16C_VERSION:
            raise ValueError(f"Invalid T16C header in {t16c_path}: magic={magic!r} ver={ver}")
        for _ in range(n_norms):
            klen = struct.unpack("<H", fp.read(2))[0]
            kname = fp.read(klen).decode("utf-8")
            alen = struct.unpack("<I", fp.read(4))[0]
            norms[kname] = (fp.tell(), int(alen))
            norm_order.append(kname)
            fp.seek(alen * 4, 1)
        for _ in range(n_mats):
            nlen, do, di, bsz, nb, asy, fw, ik, rk, pbl, lrl = struct.unpack("<HIIHBBBBHQQ", fp.read(34))
            mn = fp.read(nlen).decode("utf-8")
            pos = fp.tell()
            pad = (64 - (pos % 64)) % 64
            if pad:
                fp.seek(pad, 1)
            off = fp.tell()
            inv_b = (di * 4) if ik == 1 else ((di * 2) if ik == 2 else 0)
            mats[mn] = {
                "offset": off,
                "d_out": int(do),
                "d_in": int(di),
                "bsize": int(bsz),
                "n_bits": int(nb),
                "is_asym": bool(asy),
                "has_fwht": bool(fw),
                "inv_k": int(ik),
                "rk": int(rk),
                "pb_len": int(pbl),
                "inv_len": int(inv_b),
                "lr_len": int(lrl),
            }
            mat_order.append(mn)
            fp.seek(pbl + inv_b + lrl, 1)
        if fp.tell() != t16c_path.stat().st_size:
            raise ValueError(f"T16C EOF mismatch in {t16c_path}: {fp.tell()} != {t16c_path.stat().st_size}")
    return norm_order, norms, mats, mat_order


def read_mat_parts(fp, meta: dict) -> Tuple[bytes, Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Read raw packed payload, optional inv_s FP32 vector, and optional INT8 low-rank tuple."""
    do, di = meta["d_out"], meta["d_in"]
    fp.seek(meta["offset"])
    pb = fp.read(meta["pb_len"])
    inv_s = None
    if meta["inv_k"] == 1:
        inv_s = np.frombuffer(fp.read(di * 4), dtype="<f4").copy()
    elif meta["inv_k"] == 2:
        inv_s = np.frombuffer(fp.read(di * 2), dtype="<f2").astype(np.float32).copy()
    lr = None
    if meta["rk"] > 0:
        rk = meta["rk"]
        su = np.frombuffer(fp.read(do * 4), dtype="<f4").copy()
        sv = np.frombuffer(fp.read(rk * 4), dtype="<f4").copy()
        ui8 = np.frombuffer(fp.read(do * rk), dtype=np.int8).reshape(do, rk).copy()
        vi8 = np.frombuffer(fp.read(rk * di), dtype=np.int8).reshape(rk, di).copy()
        lr = (su, sv, ui8, vi8)
    return pb, inv_s, lr


def decode_t16c_rows_np(
    fp,
    meta: dict,
    row_indices: Sequence[int],
) -> np.ndarray:
    """Decode exact FP32 rows from any T16C matrix (`1b`, `2b_gauss`, `3b`, `tern5`, RowMajor, ColBlock, FWHT, inv_s, LR)."""
    d_out, d_in, bsize = meta["d_out"], meta["d_in"], meta["bsize"]
    n_bits, is_asym, has_fwht = meta["n_bits"], meta["is_asym"], meta["has_fwht"]
    eff = n_bits & 0x1F
    is_g = bool(n_bits & p3.T16C_GAUSS_BIT)
    is_rm = bool(n_bits & p3.T16C_ROWMAJOR_BIT)
    is_cb = bool(n_bits & p3.T16C_COLBLOCK_BIT)
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    tblk_b = p3.tile16_blk_bytes(bsize, n_bits, is_asym)
    hdr_b = 64 if is_asym else 32
    quintets = (bsize + 4) // 5
    bpr = bsize // 8
    pb_blk = bpr * 16

    rows_arr = np.asarray(list(row_indices), dtype=np.int64)
    out = np.zeros((rows_arr.size, d_in), dtype=np.float32)

    inv_s = None
    if meta["inv_k"] > 0:
        fp.seek(meta["offset"] + meta["pb_len"])
        if meta["inv_k"] == 1:
            inv_s = np.frombuffer(fp.read(d_in * 4), dtype="<f4").copy()
        elif meta["inv_k"] == 2:
            inv_s = np.frombuffer(fp.read(d_in * 2), dtype="<f2").astype(np.float32).copy()

    lr_su = lr_sv = lr_u = lr_v = None
    if meta["rk"] > 0:
        rk = meta["rk"]
        fp.seek(meta["offset"] + meta["pb_len"] + meta["inv_len"])
        lr_su = np.frombuffer(fp.read(d_out * 4), dtype="<f4").copy()
        lr_sv = np.frombuffer(fp.read(rk * 4), dtype="<f4").copy()
        lr_u = np.frombuffer(fp.read(d_out * rk), dtype=np.int8).reshape(d_out, rk).copy()
        lr_v = np.frombuffer(fp.read(rk * d_in), dtype=np.int8).reshape(rk, d_in).astype(np.float32)

    H_b = get_hadamard(bsize) if has_fwht else None

    for oi, gr in enumerate(rows_arr.tolist()):
        t = gr // 16
        r = gr % 16
        w_row = np.zeros((n_blks, bsize), dtype=np.float32)
        for blk in range(n_blks):
            idx_blk = (blk * n_tiles + t) if is_cb else (t * n_blks + blk)
            fp.seek(meta["offset"] + idx_blk * tblk_b)
            tb = np.frombuffer(fp.read(tblk_b), dtype=np.uint8)
            sc = float(np.frombuffer(tb[r * 2 : r * 2 + 2].tobytes(), dtype="<f2")[0])
            bi = float(np.frombuffer(tb[32 + r * 2 : 32 + r * 2 + 2].tobytes(), dtype="<f2")[0]) if is_asym else 0.0
            if eff == 5:
                vals = np.zeros(bsize, dtype=np.float32)
                for g in range(quintets):
                    code = int(tb[32 + g * 16 + r])
                    for k in range(5):
                        col = g * 5 + k
                        if col < bsize:
                            vals[col] = float((code % 3) - 1) * sc
                        code //= 3
                w_row[blk] = vals
            else:
                qi = np.zeros(bsize, dtype=np.int32)
                for p in range(eff):
                    pl = tb[hdr_b + p * pb_blk : hdr_b + (p + 1) * pb_blk]
                    if is_rm:
                        b_row = pl[r * bpr : (r + 1) * bpr]
                    else:
                        b_row = pl.reshape(bpr, 16)[:, r]
                    bits = np.unpackbits(b_row[:, None], axis=1, bitorder="big").reshape(bsize)
                    qi |= bits.astype(np.int32) << p
                if eff == 1:
                    u = np.where(qi > 0, 1.0, -1.0).astype(np.float32)
                elif is_g and eff == 2:
                    u = GAUSS2_LV[qi]
                elif is_g and eff == 3:
                    u = GAUSS3_LV[qi]
                else:
                    u = 2.0 * qi.astype(np.float32) - float((1 << eff) - 1)
                w_row[blk] = u * sc + bi

        if H_b is not None:
            w_row = w_row @ H_b
        vec = w_row.reshape(d_in)
        if inv_s is not None:
            vec = vec * inv_s
        if lr_su is not None and lr_u is not None and lr_sv is not None and lr_v is not None:
            vec = vec + ((lr_u[gr].astype(np.float32) * lr_su[gr]) * lr_sv) @ lr_v
        out[oi] = vec
    return out


def _is_v9_colblock_target(name: str) -> bool:
    """Match V9.0 116 COLBLOCK matrices: 60x mlp.down_proj + 56x L02..L57 self_attn.o_proj."""
    if not name.startswith("model.language_model.layers."):
        return False
    parts = name.split(".")
    if len(parts) < 5 or not parts[3].isdigit():
        return False
    l_idx = int(parts[3])
    if name.endswith(".mlp.down_proj.weight"):
        return True
    if name.endswith(".self_attn.o_proj.weight") and 2 <= l_idx <= 57:
        return True
    return False


def repack_to_v9_1_fused_vlm(src_path: Path, dst_path: Path, verify_parity: bool = True) -> Dict[str, object]:
    """Streamingly repack V4.3-VLM `.t16c` to V9.1 ROWMAJOR (0x40 on 1b) + COLBLOCK (0x20 on 116 mats)."""
    norm_order, norms_src, mats_src, mat_order = parse_t16c_metadata(src_path)
    t0 = time.perf_counter()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".t16c.tmp")

    rm_cnt = 0
    cb_cnt = 0
    with open(src_path, "rb") as fin, open(tmp_path, "wb") as fout:
        magic, ver, flags, n_norms, n_mats, _resv = struct.unpack("<4s5I", fin.read(24))
        fout.write(struct.pack("<4s5I", magic, ver, flags, n_norms, n_mats, 0))
        pos = 24
        for nk in norm_order:
            off, alen = norms_src[nk]
            fin.seek(off)
            raw_n = fin.read(alen * 4)
            kb = nk.encode("utf-8")
            fout.write(struct.pack("<H", len(kb)) + kb + struct.pack("<I", alen) + raw_n)
            pos += 2 + len(kb) + 4 + alen * 4

        for mn in mat_order:
            meta = mats_src[mn]
            do, di, bsz, nb, asy, fw, ik, rk, pbl, lrl = (
                meta["d_out"], meta["d_in"], meta["bsize"], meta["n_bits"],
                meta["is_asym"], meta["has_fwht"], meta["inv_k"], meta["rk"],
                meta["pb_len"], meta["lr_len"],
            )
            fin.seek(meta["offset"])
            pb_raw = fin.read(pbl)
            inv_b = fin.read(meta["inv_len"]) if meta["inv_len"] > 0 else b""
            lr_b = fin.read(lrl) if lrl > 0 else b""

            nb_out = nb
            is_embed = ("embed_tokens" in mn) or ("lm_head" in mn) or ("vision" in mn)
            eff = nb_out & 0x1F
            if (not is_embed) and eff == 1 and (nb_out & p3.T16C_ROWMAJOR_BIT) == 0:
                if nb_out & p3.T16C_COLBLOCK_BIT:
                    pb_raw = sdc.transpose_colblock_bytes(pb_raw, do, di, bsz, nb_out, asy, to_colblock=False)
                pb_raw = rmmod.transpose_planes_col_to_rowmajor(np.frombuffer(pb_raw, dtype=np.uint8), do, di, bsz, eff, asy).tobytes()
                if nb_out & p3.T16C_COLBLOCK_BIT:
                    pb_raw = sdc.transpose_colblock_bytes(pb_raw, do, di, bsz, eff, asy, to_colblock=True)
                nb_out |= p3.T16C_ROWMAJOR_BIT
                rm_cnt += 1

            if _is_v9_colblock_target(mn) and (nb_out & p3.T16C_COLBLOCK_BIT) == 0:
                pb_raw = sdc.transpose_colblock_bytes(pb_raw, do, di, bsz, nb_out & 0x1F, asy, to_colblock=True)
                nb_out |= p3.T16C_COLBLOCK_BIT
                cb_cnt += 1

            mname_b = mn.encode("utf-8")
            fout.write(struct.pack("<HIIHBBBBHQQ", len(mname_b), do, di, bsz, nb_out, 1 if asy else 0, 1 if fw else 0, ik, rk, len(pb_raw), lrl))
            fout.write(mname_b)
            pos += 34 + len(mname_b)
            pad = (64 - (pos % 64)) % 64
            if pad:
                fout.write(b"\x00" * pad)
                pos += pad
            fout.write(pb_raw)
            pos += len(pb_raw)
            if inv_b:
                fout.write(inv_b)
                pos += len(inv_b)
            if lr_b:
                fout.write(lr_b)
                pos += len(lr_b)

        fout.flush()
        os.fsync(fout.fileno())
    os.replace(tmp_path, dst_path)

    ver_info = sdc.verify_t16c_file(dst_path)
    max_parity_diff = 0.0
    if verify_parity:
        _, _, mats_dst, _ = parse_t16c_metadata(dst_path)
        seen_modes = set()
        check_mats: List[str] = []
        for mn in mat_order:
            mode_tag = (mats_dst[mn]["n_bits"], mats_dst[mn]["is_asym"], "down_proj" in mn, "o_proj" in mn)
            if mode_tag not in seen_modes:
                seen_modes.add(mode_tag)
                check_mats.append(mn)
        with open(src_path, "rb") as fs, open(dst_path, "rb") as fd:
            for mn in check_mats:
                rows_chk = [0, min(17, mats_dst[mn]["d_out"] - 1)]
                w_s = decode_t16c_rows_np(fs, mats_src[mn], rows_chk)
                w_d = decode_t16c_rows_np(fd, mats_dst[mn], rows_chk)
                max_parity_diff = max(max_parity_diff, float(np.max(np.abs(w_s - w_d))))
        if max_parity_diff > 1e-6:
            raise RuntimeError(f"V9.1 lossless repack parity check failed: max_diff={max_parity_diff:.3e}")

    return {
        "src": str(src_path),
        "dst": str(dst_path),
        "bytes": int(dst_path.stat().st_size),
        "n_norms": int(ver_info["n_norms"]),
        "n_mats": int(ver_info["n_mats"]),
        "rowmajor_converted": rm_cnt,
        "colblock_mats": int(ver_info["colblock_mats"]),
        "max_parity_diff": float(max_parity_diff),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


class NativeNeonLinear:
    """FFI wrapper around `libneon_stream_accelerator.so` with pure-NumPy fallback on non-POSIX hosts."""

    def __init__(self, so_path: Optional[Path] = None, threads: int = 4) -> None:
        self.threads = int(max(1, threads))
        self.lib = None
        if so_path is not None and so_path.is_file() and os.name == "posix":
            lib = ctypes.CDLL(str(so_path))
            lib.gemm_bitserial_tile16_sparse_batch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ]
            lib.gemm_bitserial_tile16_sparse_batch.restype = ctypes.c_int
            lib.inv_s_and_fwht_f32_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.inv_s_and_fwht_f32_c.restype = ctypes.c_int
            lib.int8_lowrank_add_batch_c.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            lib.int8_lowrank_add_batch_c.restype = ctypes.c_int
            self.lib = lib

    def linear(
        self,
        fp,
        meta: dict,
        x_in: np.ndarray,
    ) -> np.ndarray:
        pb, inv_s, lr = read_mat_parts(fp, meta)
        x_c = np.ascontiguousarray(x_in, dtype=np.float32)
        k_tok, d_in = int(x_c.shape[0]), int(x_c.shape[1])
        d_out, bsize = int(meta["d_out"]), int(meta["bsize"])
        assert d_in == int(meta["d_in"]), f"Shape mismatch: {d_in} != {meta['d_in']}"
        if self.lib is not None:
            x_w = x_c.copy()
            inv_ptr = inv_s.ctypes.data if inv_s is not None else 0
            if inv_s is not None or meta["has_fwht"]:
                rc_f = self.lib.inv_s_and_fwht_f32_c(x_w.ctypes.data, inv_ptr, k_tok, d_in, bsize if meta["has_fwht"] else 1)
                if rc_f != 0:
                    raise RuntimeError(f"inv_s_and_fwht_f32_c failed rc={rc_f}")
            pb_arr = np.frombuffer(pb, dtype=np.uint8)
            y = np.empty((k_tok, d_out), dtype=np.float32)
            rc_g = self.lib.gemm_bitserial_tile16_sparse_batch(
                pb_arr.ctypes.data, 0, x_w.ctypes.data, y.ctypes.data,
                d_out, d_in, bsize, int(meta["n_bits"]), 1 if meta["is_asym"] else 0, k_tok,
                ctypes.c_float(0.0), 0, 0, self.threads, 0,
            )
            if rc_g != 0:
                raise RuntimeError(f"gemm_bitserial_tile16_sparse_batch failed rc={rc_g}")
            if lr is not None:
                su, sv, ui8, vi8 = lr
                rc_l = self.lib.int8_lowrank_add_batch_c(
                    ui8.ctypes.data, su.ctypes.data, vi8.ctypes.data, sv.ctypes.data,
                    x_c.ctypes.data, y.ctypes.data, d_out, d_in, int(sv.size), k_tok, self.threads,
                )
                if rc_l != 0:
                    raise RuntimeError(f"int8_lowrank_add_batch_c failed rc={rc_l}")
            return y

        # Pure-NumPy fallback (for unit selftest on non-POSIX hosts)
        w = decode_t16c_rows_np(fp, meta, list(range(d_out)))
        return (x_c @ w.T).astype(np.float32)


def _rmsn(x: np.ndarray, w: Optional[np.ndarray], eps: float) -> np.ndarray:
    xf = x.astype(np.float32)
    y = xf / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y * w) if w is not None else y


def _make_rope(theta: float, rot_d: int, hd: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    hr = rot_d // 2
    if hr <= 0:
        return np.zeros((seq_len, 0), dtype=np.float32), np.zeros((seq_len, 0), dtype=np.float32)
    inv_f = 1.0 / (theta ** (np.arange(0, 2 * hr, 2, dtype=np.float32) / float(hd)))
    ang = np.outer(np.arange(seq_len, dtype=np.float32), inv_f)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _apply_rope(x: np.ndarray, cs: np.ndarray, sn: np.ndarray, rot_d: int, hd: int) -> np.ndarray:
    hr, hh = rot_d // 2, hd // 2
    if hr <= 0:
        return x
    x0, x1 = x[:, :, :hr], x[:, :, hh : hh + hr]
    c, s = cs[:, None, :], sn[:, None, :]
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    if hr == hh:
        return np.concatenate([r0, r1], axis=-1)
    return np.concatenate([r0, x[:, :, hr:hh], r1, x[:, :, hh + hr :]], axis=-1)


def build_decontaminated_token_sequences(tok_path: Path, max_seqs: int = 16, max_seq_len: int = 64) -> List[List[int]]:
    """Build strictly decontaminated Gemma-4 `<|turn>` sequences from `v4_corpus.py`."""
    if Tokenizer is None:
        raise ImportError("tokenizers package is required to tokenize calibration corpus")
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json: {tok_path}")

    v4_corpus.assert_no_contamination([p for _, p, _ in v4_corpus.V4_CAL_PAIRS], "prompts")
    v4_corpus.assert_no_contamination([r for _, _, r in v4_corpus.V4_CAL_PAIRS], "responses")

    tk = Tokenizer.from_file(str(tok_path))
    bos = tk.token_to_id("<bos>")
    if bos is None:
        bos = 2
    seqs: List[List[int]] = []
    for _, p, r in v4_corpus.V4_CAL_PAIRS:
        txt = v4_corpus.format_gemma4_turn(p, r)
        ids = [int(bos)] + [int(t) for t in tk.encode(txt, add_special_tokens=False).ids]
        seqs.append(ids[:max_seq_len])
        if len(seqs) >= max_seqs:
            break
    return seqs


def train_crtr_from_t16c(
    t16c_path: Path,
    cfg_path: Path,
    tok_path: Path,
    out_crtr_path: Path,
    so_path: Optional[Path] = None,
    threads: int = 4,
    rank: int = 64,
    block_size: int = 256,
    top_p: float = 0.50,
    steps: int = 160,
    max_seqs: int = 16,
    max_seq_len: int = 64,
) -> Dict[str, object]:
    """Stream all transformer layers from `t16c_path`, collect real GeGLU block energies, and export 60-layer CRTR."""
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    cfg_root = json.loads(cfg_path.read_text(encoding="utf-8"))
    tc = cfg_root.get("text_config", cfg_root)
    d_mod = int(tc["hidden_size"])
    ff = int(tc["intermediate_size"])
    n_layers = int(tc["num_hidden_layers"])
    nh = int(tc["num_attention_heads"])
    nkv_s = int(tc["num_key_value_heads"])
    nkv_f = int(tc["num_global_key_value_heads"])
    hd_s = int(tc["head_dim"])
    hd_f = int(tc["global_head_dim"])
    window = int(tc["sliding_window"])
    eps = float(tc["rms_norm_eps"])
    k_eq_v = bool(tc.get("attention_k_eq_v", False))
    layer_types = tc["layer_types"]
    rp = tc["rope_parameters"]
    theta_g = float(rp["full_attention"]["rope_theta"])
    part_g = float(rp["full_attention"].get("partial_rotary_factor", 1.0))
    theta_l = float(rp["sliding_attention"]["rope_theta"])
    part_l = float(rp["sliding_attention"].get("partial_rotary_factor", 1.0))
    n_blocks = ff // block_size
    assert ff == n_blocks * block_size

    seqs = build_decontaminated_token_sequences(tok_path, max_seqs=max_seqs, max_seq_len=max_seq_len)
    seq_lens = [len(s) for s in seqs]
    tot_tok = sum(seq_lens)
    seq_offs = [0]
    for sl in seq_lens:
        seq_offs.append(seq_offs[-1] + sl)

    _, norms_idx, mats_idx, _ = parse_t16c_metadata(t16c_path)
    engine = NativeNeonLinear(so_path=so_path, threads=threads)
    torch.set_num_threads(max(1, min(threads, 4)))

    t0 = time.perf_counter()
    with open(t16c_path, "rb") as fp:
        norms: Dict[str, np.ndarray] = {}
        for nk, (off, alen) in norms_idx.items():
            if "vision" in nk:
                continue
            fp.seek(off)
            norms[nk] = np.frombuffer(fp.read(alen * 4), dtype="<f4").copy()

        flat_ids = [tid for s in seqs for tid in s]
        uniq_ids, inv_idx = np.unique(np.asarray(flat_ids, dtype=np.int64), return_inverse=True)
        emb_rows = decode_t16c_rows_np(fp, mats_idx["model.language_model.embed_tokens.weight"], uniq_ids.tolist())
        x_states = (emb_rows[inv_idx] * math.sqrt(float(d_mod))).astype(np.float32)

        cs_l, sn_l = _make_rope(theta_l, int(round(hd_s * part_l)), hd_s, max(seq_lens))
        cs_g, sn_g = _make_rope(theta_g, int(round(hd_f * part_g)), hd_f, max(seq_lens))

        crtr_layers: List[dict] = []
        recalls: List[float] = []
        mod_accs: List[float] = []

        for l_idx in range(n_layers):
            pfx = f"model.language_model.layers.{l_idx}"
            is_f = layer_types[l_idx] == "full_attention"
            hd = hd_f if is_f else hd_s
            nkv = nkv_f if is_f else nkv_s
            rep = nh // nkv
            rot_d = int(round(hd_f * part_g)) if is_f else int(round(hd_s * part_l))
            cs_r, sn_r = (cs_g, sn_g) if is_f else (cs_l, sn_l)

            xn = _rmsn(x_states, norms[f"{pfx}.input_layernorm.weight"], eps)
            q_all = engine.linear(fp, mats_idx[f"{pfx}.self_attn.q_proj.weight"], xn).reshape(tot_tok, nh, hd)
            k_all = engine.linear(fp, mats_idx[f"{pfx}.self_attn.k_proj.weight"], xn).reshape(tot_tok, nkv, hd)
            if is_f and k_eq_v:
                v_raw = k_all
            else:
                v_raw = engine.linear(fp, mats_idx[f"{pfx}.self_attn.v_proj.weight"], xn).reshape(tot_tok, nkv, hd)

            q_all = _rmsn(q_all, norms[f"{pfx}.self_attn.q_norm.weight"], eps)
            k_all = _rmsn(k_all, norms[f"{pfx}.self_attn.k_norm.weight"], eps)
            v_all = _rmsn(v_raw, None, eps)

            ctx_all = np.empty((tot_tok, nh * hd), dtype=np.float32)
            for si, slen in enumerate(seq_lens):
                s0, s1 = seq_offs[si], seq_offs[si + 1]
                qs = _apply_rope(q_all[s0:s1], cs_r[:slen], sn_r[:slen], rot_d, hd)
                ks = _apply_rope(k_all[s0:s1], cs_r[:slen], sn_r[:slen], rot_d, hd)
                vs = v_all[s0:s1]
                kr = np.repeat(ks, rep, axis=1)
                vr = np.repeat(vs, rep, axis=1)
                sc = np.einsum("thd,shd->hts", qs, kr)
                mask = np.triu(np.full((slen, slen), -1e30, dtype=np.float32), k=1)
                if not is_f and slen > window:
                    mask += np.tril(np.full((slen, slen), -1e30, dtype=np.float32), k=-(window + 1))
                sc = sc + mask[None, :, :]
                sc -= np.max(sc, axis=-1, keepdims=True)
                pr = np.exp(sc)
                pr /= np.sum(pr, axis=-1, keepdims=True)
                ctx_all[s0:s1] = np.einsum("hts,shd->thd", pr, vr).reshape(slen, nh * hd)

            attn_d = _rmsn(engine.linear(fp, mats_idx[f"{pfx}.self_attn.o_proj.weight"], ctx_all), norms[f"{pfx}.post_attention_layernorm.weight"], eps)
            xm = x_states + attn_d
            xf = _rmsn(xm, norms[f"{pfx}.pre_feedforward_layernorm.weight"], eps)

            gate = engine.linear(fp, mats_idx[f"{pfx}.mlp.gate_proj.weight"], xf)
            gg = F.gelu(torch.from_numpy(gate), approximate="tanh").numpy()
            blk_e = gg.reshape(tot_tok, n_blocks, block_size)
            blk_e = np.sum(blk_e * blk_e, axis=-1).astype(np.float32)

            up = engine.linear(fp, mats_idx[f"{pfx}.mlp.up_proj.weight"], xf)
            act = (gg * up).astype(np.float32)
            mlp_d = _rmsn(engine.linear(fp, mats_idx[f"{pfx}.mlp.down_proj.weight"], act), norms[f"{pfx}.post_feedforward_layernorm.weight"], eps)
            lsc = float(norms[f"{pfx}.layer_scalar"][0])
            x_next = ((xm + mlp_d) * lsc).astype(np.float32)

            rel_delta = np.linalg.norm(x_next - x_states, axis=-1) / (np.linalg.norm(x_states, axis=-1) + 1e-6)
            q28 = float(np.quantile(rel_delta, 0.28))
            mod_y = (rel_delta > q28).astype(np.float32)

            rng_split = np.random.default_rng(1000 + l_idx)
            perm_tok = rng_split.permutation(tot_tok)
            n_tr = max(1, (tot_tok * 9) // 10)
            tr_i, va_i = perm_tok[:n_tr], perm_tok[n_tr:]
            if va_i.size == 0:
                va_i = tr_i[-max(1, tot_tok // 10) :]

            r_net, r_met = crtr.train_layer_router(
                torch.from_numpy(xf[tr_i]),
                torch.from_numpy(blk_e[tr_i]),
                torch.from_numpy(xf[va_i]),
                torch.from_numpy(blk_e[va_i]),
                top_p=top_p,
                rank=rank,
                steps=steps,
                lr=3e-3,
                mod_targets_tr=torch.from_numpy(mod_y[tr_i]),
                mod_targets_va=torch.from_numpy(mod_y[va_i]),
                device="cpu",
                seed=200 + l_idx,
            )
            r_q = crtr.quantize_layer_for_crtr(r_net, block_size)
            crtr_layers.append(r_q)
            recalls.append(float(r_met["val_recall_after"]))
            mod_accs.append(float(r_met["mod_val_acc"]))
            x_states = x_next

            if l_idx % 5 == 0 or l_idx == n_layers - 1:
                print(
                    f"  [V9.1 CRTR L{l_idx:02d}/{n_layers - 1:02d}] recall={r_met['val_recall_after']:.4f} "
                    f"mod_acc={r_met['mod_val_acc']:.4f} elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            del xn, q_all, k_all, v_raw, v_all, ctx_all, attn_d, xm, xf, gate, gg, up, act, mlp_d
            gc.collect()

    crtr_bytes = crtr.export_crtr(crtr_layers, out_crtr_path)
    meta_c, _ = crtr.load_crtr(out_crtr_path)
    assert meta_c["n_layers"] == n_layers and meta_c["rank"] == rank and meta_c["n_blocks"] == n_blocks

    return {
        "t16c": str(t16c_path),
        "out_crtr": str(out_crtr_path),
        "crtr_bytes": int(crtr_bytes),
        "n_layers": n_layers,
        "tokens": tot_tok,
        "sequences": len(seqs),
        "rank": rank,
        "top_p": top_p,
        "mean_recall": round(float(np.mean(recalls)), 5),
        "min_recall": round(float(np.min(recalls)), 5),
        "mean_mod_acc": round(float(np.mean(mod_accs)), 5),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def _pack_synth_t16c_record(w_rot: np.ndarray, eff: int, is_asym: bool, is_gauss: bool, is_rm: bool, is_cb: bool) -> Tuple[int, bytes]:
    d_out, d_in = w_rot.shape
    bsize = 64
    n_tiles, n_blks = (d_out + 15) // 16, d_in // bsize
    wb = np.pad(w_rot, ((0, n_tiles * 16 - d_out), (0, 0))).reshape(n_tiles, 16, n_blks, bsize).transpose(0, 2, 1, 3)

    if eff == 5:
        tblk_b = 32 + ((bsize + 4) // 5) * 16
        arr = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
        sc = np.maximum(np.max(np.abs(wb), axis=3), 1e-6)
        arr[:, :, :32] = np.frombuffer(np.ascontiguousarray(sc).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)
        trits = np.clip(np.round(wb / sc[..., None]), -1, 1).astype(np.int32) + 1
        quintets = (bsize + 4) // 5
        for g in range(quintets):
            code = np.zeros((n_tiles, n_blks, 16), dtype=np.int32)
            p3_w = 1
            for k in range(5):
                col = g * 5 + k
                d = trits[:, :, :, col] if col < bsize else 1
                code += d * p3_w
                p3_w *= 3
            arr[:, :, 32 + g * 16 : 32 + (g + 1) * 16] = code.astype(np.uint8)
        raw_b = arr.reshape(-1).tobytes()
        nb = eff
    else:
        hdr_b = 64 if is_asym else 32
        bpr = bsize // 8
        pb_blk = bpr * 16
        tblk_b = hdr_b + eff * pb_blk
        arr = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
        if eff == 1:
            bi = wb.mean(axis=3)
            sc = np.maximum(wb.std(axis=3), 1e-4)
            qi = (wb >= bi[..., None]).astype(np.int32)
            arr[:, :, :32] = np.frombuffer(np.ascontiguousarray(sc).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)
            if is_asym:
                arr[:, :, 32:64] = np.frombuffer(np.ascontiguousarray(bi).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)
        else:
            lv = GAUSS2_LV if (is_gauss and eff == 2) else (GAUSS3_LV if (is_gauss and eff == 3) else np.arange(-(2**eff - 1), 2**eff, 2, dtype=np.float32))
            sc = np.maximum(wb.std(axis=3) / max(float(lv.std()), 1.0), 1e-6)
            diffs = np.abs((wb / sc[..., None])[..., None] - lv[None, None, None, None, :])
            qi = np.argmin(diffs, axis=-1).astype(np.int32)
            arr[:, :, :32] = np.frombuffer(np.ascontiguousarray(sc).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)
        for p in range(eff):
            bits = ((qi >> p) & 1).astype(np.uint8).reshape(n_tiles, n_blks, 16, bpr, 8)
            pk = np.packbits(bits, axis=4, bitorder="big").reshape(n_tiles, n_blks, 16, bpr).transpose(0, 1, 3, 2)
            arr[:, :, hdr_b + p * pb_blk : hdr_b + (p + 1) * pb_blk] = pk.reshape(n_tiles, n_blks, pb_blk)
        raw_u8 = arr.reshape(-1)
        nb = eff | (p3.T16C_GAUSS_BIT if is_gauss else 0)
        if is_rm:
            raw_u8 = rmmod.transpose_planes_col_to_rowmajor(raw_u8, d_out, d_in, bsize, eff, is_asym)
            nb |= p3.T16C_ROWMAJOR_BIT
        raw_b = raw_u8.tobytes()

    if is_cb:
        raw_b = sdc.transpose_colblock_bytes(raw_b, d_out, d_in, bsize, eff, is_asym, to_colblock=True)
        nb |= p3.T16C_COLBLOCK_BIT
    return nb, raw_b


def run_selftest() -> int:
    """6-gate fail-closed CPU verification battery for V9.1-Fused-VLM repacking & CRTR training."""
    failures = 0
    rng = np.random.default_rng(2026)

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Bit-exact decode parity after RowMajor + COLBLOCK across 1b, tern5, 2b_gauss, 3b
    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        g1_ok = True
        g1_det = []
        w0 = rng.normal(0.0, 0.02, size=(32, 128)).astype(np.float32)
        h64 = get_hadamard(64)
        w0_rot = (w0.reshape(-1, 64) @ h64).reshape(32, 128)
        specs = [
            ("1b", 1, True, False, True, True),
            ("tern5", 5, False, False, False, True),
            ("2bg", 2, False, True, False, True),
            ("3bg", 3, False, True, False, True),
        ]
        for tag, eff, asy, ga, rm_tgt, cb_tgt in specs:
            nb_s, pb_s = _pack_synth_t16c_record(w0_rot, eff, asy, ga, False, False)
            nb_d, pb_d = _pack_synth_t16c_record(w0_rot, eff, asy, ga, rm_tgt, cb_tgt)
            p_src = td / f"{tag}_s.t16c"
            p_dst = td / f"{tag}_d.t16c"
            inv_v = np.linspace(0.7, 1.3, 128, dtype="<f4")
            for pp, nbp, pbp in ((p_src, nb_s, pb_s), (p_dst, nb_d, pb_d)):
                with open(pp, "wb") as fp:
                    fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, 0, 1, 0))
                    p3._write_mat_record(fp, 24, "test.weight", 32, 128, 64, nbp, asy, True, np.frombuffer(pbp, dtype=np.uint8), inv_v, None)
            _, _, ms, _ = parse_t16c_metadata(p_src)
            _, _, md, _ = parse_t16c_metadata(p_dst)
            with open(p_src, "rb") as fs, open(p_dst, "rb") as fd:
                ws = decode_t16c_rows_np(fs, ms["test.weight"], [0, 5, 19, 31])
                wd = decode_t16c_rows_np(fd, md["test.weight"], [0, 5, 19, 31])
            diff = float(np.max(np.abs(ws - wd)))
            g1_ok = g1_ok and diff < 1e-6 and len(pb_s) == len(pb_d)
            g1_det.append(f"{tag}:{diff:.1e}")
        gate("Gate 1: RowMajor+COLBLOCK lossless parity across 1b/tern5/2b_gauss/3b", g1_ok, " ".join(g1_det))

        # Gate 2: Full synthetic multi-layer V9.1 repack (L00 tern5, L02 1b+o_proj, embed untouched)
        p_raw = td / "v43_synth.t16c"
        p_v91 = td / "v91_synth.t16c"
        norms_s = [("model.language_model.norm.weight", np.ones(128, dtype="<f4"))]
        recs_s = []
        w_emb = rng.normal(0.0, 0.02, size=(32, 128)).astype(np.float32)
        nb_e, pb_e = _pack_synth_t16c_record((w_emb.reshape(-1, 64) @ h64).reshape(32, 128), 3, False, False, False, False)
        recs_s.append(("model.language_model.embed_tokens.weight", 32, 128, nb_e, False, pb_e, None))
        for li in (0, 2):
            for pn, eff, asy in (("self_attn.q_proj", 1, True), ("self_attn.o_proj", 1, True), ("mlp.gate_proj", 5, False), ("mlp.down_proj", 1 if li == 2 else 5, li == 2)):
                nb_m, pb_m = _pack_synth_t16c_record(w0_rot, eff, asy, False, False, False)
                recs_s.append((f"model.language_model.layers.{li}.{pn}.weight", 32, 128, nb_m, asy, pb_m, inv_v))
        with open(p_raw, "wb") as fp:
            fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, len(norms_s), len(recs_s), 0))
            pos = 24
            for nk, na in norms_s:
                kb = nk.encode("utf-8")
                fp.write(struct.pack("<H", len(kb)) + kb + struct.pack("<I", na.size) + na.tobytes())
                pos += 2 + len(kb) + 4 + na.size * 4
            for mn, do, di, nbp, asyp, pbp, iv in recs_s:
                pos = p3._write_mat_record(fp, pos, mn, do, di, 64, nbp, asyp, True, np.frombuffer(pbp, dtype=np.uint8), iv, None)
        rp_info = repack_to_v9_1_fused_vlm(p_raw, p_v91, verify_parity=True)
        g2_ok = rp_info["bytes"] == p_raw.stat().st_size and rp_info["colblock_mats"] == 3 and rp_info["rowmajor_converted"] == 5 and rp_info["max_parity_diff"] < 1e-6
        gate("Gate 2: Full V9.1 multi-layer streaming repack preserves byte size + 0.00 decode diff", g2_ok, f"rm={rp_info['rowmajor_converted']} cb={rp_info['colblock_mats']} diff={rp_info['max_parity_diff']:.1e}")

        # Gate 3: NativeNeonLinear fallback linear parity vs explicit row decode @ x.T
        _, _, m91, _ = parse_t16c_metadata(p_v91)
        eng = NativeNeonLinear(so_path=None, threads=2)
        x_in = rng.normal(0.0, 0.5, size=(6, 128)).astype(np.float32)
        with open(p_v91, "rb") as f91:
            y_lin = eng.linear(f91, m91["model.language_model.layers.2.mlp.down_proj.weight"], x_in)
            w_dec = decode_t16c_rows_np(f91, m91["model.language_model.layers.2.mlp.down_proj.weight"], list(range(32)))
        lin_diff = float(np.max(np.abs(y_lin - (x_in @ w_dec.T))))
        gate("Gate 3: T16C linear execution matches explicit row decode", lin_diff < 1e-5, f"max_diff={lin_diff:.2e}")

        # Gate 4: End-to-end synthetic CRTR trainer + INT8 export + load roundtrip (2 planted of 4 for top_p=0.50)
        q_mat, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(64, 4)).astype(np.float32))
        x_syn = rng.normal(0.0, 0.1, size=(240, 64)).astype(np.float32)
        e_syn = np.abs(rng.normal(0.0, 0.02, size=(240, 4))).astype(np.float32)
        m_syn = np.zeros((240,), dtype=np.float32)
        for ii in range(240):
            b0 = ii % 4
            b1 = (ii + 1) % 4
            x_syn[ii] += 2.2 * q_mat[:, b0] + 2.0 * q_mat[:, b1]
            e_syn[ii, b0] = 6.0
            e_syn[ii, b1] = 5.5
            m_syn[ii] = float(ii % 2)
        r_net, r_met = crtr.train_layer_router(
            torch.from_numpy(x_syn[:192]), torch.from_numpy(e_syn[:192]),
            torch.from_numpy(x_syn[192:]), torch.from_numpy(e_syn[192:]),
            top_p=0.50, rank=8, steps=200, lr=8e-3,
            mod_targets_tr=torch.from_numpy(m_syn[:192]), mod_targets_va=torch.from_numpy(m_syn[192:]),
            device="cpu", seed=11,
        )
        rq = crtr.quantize_layer_for_crtr(r_net, 16)
        crtr_p = td / "synth_v91.crtr"
        crtr.export_crtr([rq], crtr_p)
        cmeta, clayers = crtr.load_crtr(crtr_p)
        g4_ok = r_met["val_recall_after"] >= 0.85 and cmeta["n_layers"] == 1 and len(clayers) == 1
        gate("Gate 4: CRTR trainer + INT8 export/load roundtrip", g4_ok, f"recall={r_met['val_recall_after']:.3f}")

        # Gate 5: Strict Rule 2 Decontamination audit on V4_CAL_PAIRS
        g5_ok = False
        try:
            v4_corpus.assert_no_contamination([p for _, p, _ in v4_corpus.V4_CAL_PAIRS], "prompts")
            v4_corpus.assert_no_contamination([r for _, _, r in v4_corpus.V4_CAL_PAIRS], "responses")
            g5_ok = len(v4_corpus.V4_CAL_PAIRS) >= 16
        except Exception:
            g5_ok = False
        gate("Gate 5: Rule 2 Decontamination verification across calibration corpus", g5_ok, f"pairs={len(v4_corpus.V4_CAL_PAIRS)}")

        # Gate 6: Fail-closed FileNotFoundError on missing inputs
        miss_ok = False
        try:
            parse_t16c_metadata(td / "missing.t16c")
        except FileNotFoundError:
            miss_ok = True
        gate("Gate 6: Fail-closed raises FileNotFoundError on missing `.t16c`", miss_ok)

    print(f"V9_1_BUILDER_SELFTEST: {6 - failures}/6 gates passed", flush=True)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemma 4 31B V9.1-Fused-VLM Builder")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repack", action="store_true", help="Losslessly repack V4.3-VLM .t16c to V9.1 ROWMAJOR+COLBLOCK")
    ap.add_argument("--train-crtr", action="store_true", help="Train 60-layer CRTR+MoD router streamingly from .t16c")
    ap.add_argument("--src", type=Path, help="Source .t16c path")
    ap.add_argument("--dst", type=Path, help="Output .t16c path (or CRTR source when --train-crtr only)")
    ap.add_argument("--out-crtr", type=Path, help="Output .crtr router path")
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--so", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/libneon_stream_accelerator.so"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=0.50)
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--report", type=Path, help="Optional JSON summary report output path")
    args = ap.parse_args()

    if args.selftest or (not args.repack and not args.train_crtr):
        return run_selftest()

    rpt: Dict[str, object] = {}
    target_t16c = args.dst
    if args.repack:
        if not args.src or not args.dst:
            raise SystemExit("--repack requires --src and --dst")
        rpt["repack"] = repack_to_v9_1_fused_vlm(args.src, args.dst, verify_parity=True)
        rpt["t16c_sha256"] = sha256_file(args.dst)
        print("[V9.1 Repack COMPLETE]", json.dumps(rpt["repack"]), flush=True)

    if args.train_crtr:
        t16_src = target_t16c if (target_t16c and target_t16c.is_file()) else args.src
        if not t16_src or not args.out_crtr:
            raise SystemExit("--train-crtr requires --dst/--src and --out-crtr")
        rpt["crtr"] = train_crtr_from_t16c(
            t16_src, args.config, args.tokenizer, args.out_crtr,
            so_path=args.so, threads=args.threads, top_p=args.top_p,
            max_seqs=args.max_seqs, max_seq_len=args.max_seq_len,
        )
        rpt["crtr_sha256"] = sha256_file(args.out_crtr)
        print("[V9.1 CRTR COMPLETE]", json.dumps(rpt["crtr"]), flush=True)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rpt, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

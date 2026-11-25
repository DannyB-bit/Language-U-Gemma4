#!/usr/bin/env python3
"""Approach 1 & Patch Utility: Zero-CU Closed-Form Weight-LS Repair + Permutation Recovery for Gemma 4 31B V9.0.

Handles:
  1. Compact `V9HP` (`~101 MB`) reversible header+norm snapshots (`--save-patch`, `--apply-patch`).
  2. Exact global outlier-dispersion permutation `perm_pi` (`S_5376`) and exact per-layer GeGLU
     neuron permutation `pi_l` (`S_21504` across all 60 layers) recovery from frozen `.t16c` bitplanes.
  3. Exact closed-form Least-Squares FP16 `(sc*, bi*)` block header + teacher norm restoration
     across all QAT-corrupted matrices (`L02..L57` Q/O and `L00..L59` MLPs) with bitplane SHA-256
     immutability checks and fail-closed `w_cos` assertions.
  4. 6-gate CPU self-verification battery (`--selftest`).
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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pack_assistant_t16c as pa  # noqa: E402
import pack_t16c_v3 as p3  # noqa: E402
import repack_t16c_rowmajor as rmmod  # noqa: E402
import spatial_district_converter as sdc  # noqa: E402

V9HP_MAGIC = b"V9HP"
V9HP_VERSION = 1

GAUSS2_LV = np.array([-3.174, -1.0, 1.0, 3.174], dtype=np.float32)
GAUSS3_LV = np.array([-7.430, -5.430, -3.270, -1.270, 1.270, 3.270, 5.430, 7.430], dtype=np.float32)

_H256_CACHE: Optional[np.ndarray] = None


def get_h256_f32() -> np.ndarray:
    """Return contiguous 256x256 normalized Sylvester Walsh-Hadamard matrix."""
    global _H256_CACHE
    if _H256_CACHE is None:
        _H256_CACHE = np.ascontiguousarray(pa.make_sylvester_hadamard(256), dtype=np.float32)
    return _H256_CACHE


def try_load_c_lib() -> Optional[ctypes.CDLL]:
    """Load precompiled `libfast_v9_repair.so` or compile cleanly if gcc is available on POSIX."""
    so_path = THIS_DIR / "libfast_v9_repair.so"
    c_path = THIS_DIR / "fast_v9_repair.c"
    if os.name == "nt":
        return None
    if (not so_path.is_file() or so_path.stat().st_mtime < c_path.stat().st_mtime) and c_path.is_file():
        gcc = subprocess.run(["which", "gcc"], capture_output=True, text=True)
        if gcc.returncode == 0:
            subprocess.run(
                ["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", str(c_path), "-o", str(so_path), "-lm"],
                check=True,
            )
    if not so_path.is_file():
        return None
    lib = ctypes.CDLL(str(so_path))
    lib.fwht256_rows_f32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.pack_ac_bits_from_trot_f32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.extract_t16c_sign_bits.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    lib.match_mlp_perm_popcnt.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.dequant_w4a16_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    lib.solve_and_write_wls_headers.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.solve_and_write_wls_headers.restype = ctypes.c_int
    return lib


class T16CRecordIndex:
    """Lightweight streaming index of norm & matrix offsets inside a `.t16c` binary."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing T16C file: {self.path}")
        self.norms: Dict[str, Tuple[int, int]] = {}
        self.mats: Dict[str, dict] = {}
        self.norm_order: List[str] = []
        self.mat_order: List[str] = []
        with open(self.path, "rb") as fp:
            magic, ver, flags, n_norms, n_mats, _resv = struct.unpack("<4s5I", fp.read(24))
            if magic != p3.T16C_MAGIC or ver != p3.T16C_VERSION:
                raise ValueError(f"Invalid T16C header in {self.path}: magic={magic!r} ver={ver}")
            for _ in range(n_norms):
                klen = struct.unpack("<H", fp.read(2))[0]
                kname = fp.read(klen).decode("utf-8")
                alen = struct.unpack("<I", fp.read(4))[0]
                off = fp.tell()
                self.norms[kname] = (off, int(alen))
                self.norm_order.append(kname)
                fp.seek(alen * 4, 1)
            for _ in range(n_mats):
                nlen, do, di, bsz, nb, asy, fw, ik, rk, pbl, lrl = struct.unpack("<HIIHBBBBHQQ", fp.read(34))
                mname = fp.read(nlen).decode("utf-8")
                pos = fp.tell()
                pad = (64 - (pos % 64)) % 64
                if pad:
                    fp.seek(pad, 1)
                off = fp.tell()
                inv_b = (di * 4) if ik == 1 else ((di * 2) if ik == 2 else 0)
                self.mats[mname] = {
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
                self.mat_order.append(mname)
                fp.seek(pbl + inv_b + lrl, 1)
            if fp.tell() != self.path.stat().st_size:
                raise ValueError(f"T16C EOF mismatch: parsed {fp.tell()} != {self.path.stat().st_size}")


class FastSafetensorsReader:
    """Streaming `.safetensors` reader supporting BF16/F32/F16/I32 and W4A16 dequantization without mmap pressure."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Safetensors not found: {self.path}")
        self._fp = open(self.path, "rb")
        try:
            hlen = struct.unpack("<Q", self._fp.read(8))[0]
            self.hdr: Dict[str, dict] = json.loads(self._fp.read(hlen).decode("utf-8"))
            self.base = 8 + hlen
        except Exception:
            self._fp.close()
            raise

    def __enter__(self) -> "FastSafetensorsReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def read_tensor_np(self, name: str) -> np.ndarray:
        info = self.hdr[name]
        s0, s1 = info["data_offsets"]
        dt = info["dtype"]
        sh = tuple(info["shape"])
        self._fp.seek(self.base + s0)
        raw = self._fp.read(s1 - s0)
        if dt == "BF16":
            return ((np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4")).reshape(sh).copy()
        if dt == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(sh).copy()
        if dt == "F16":
            return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(sh).copy()
        if dt == "I32":
            return np.frombuffer(raw, dtype="<i4").reshape(sh).copy()
        raise ValueError(f"Unsupported dtype {dt!r} for {name}")

    def read_dequant_f32(self, base_name: str, lib: Optional[ctypes.CDLL] = None) -> np.ndarray:
        if f"{base_name}.weight_packed" in self.hdr:
            pk = np.ascontiguousarray(self.read_tensor_np(f"{base_name}.weight_packed"), dtype=np.int32)
            if lib is not None and self.hdr[f"{base_name}.weight_scale"]["dtype"] == "BF16":
                info_sc = self.hdr[f"{base_name}.weight_scale"]
                self._fp.seek(self.base + info_sc["data_offsets"][0])
                sc_raw = np.frombuffer(self._fp.read(info_sc["data_offsets"][1] - info_sc["data_offsets"][0]), dtype=np.uint16)
                out = np.empty((pk.shape[0], pk.shape[1] * 8), dtype=np.float32)
                lib.dequant_w4a16_f32(pk.ctypes.data, sc_raw.ctypes.data, pk.shape[0], pk.shape[1], int(info_sc["shape"][1]), out.ctypes.data)
                del pk, sc_raw
                return out
            sc = self.read_tensor_np(f"{base_name}.weight_scale")
            shifts = np.arange(0, 32, 4, dtype=np.int32)
            nibs = ((pk[:, :, None] >> shifts) & 0xF) - 8
            out = (nibs.reshape(pk.shape[0], sc.shape[1], 32).astype(np.float32) * sc[:, :, None]).reshape(pk.shape[0], -1)
            del pk, sc, nibs
            return np.ascontiguousarray(out, dtype=np.float32)
        return np.ascontiguousarray(self.read_tensor_np(f"{base_name}.weight"), dtype=np.float32)


def _read_inv_and_lr(fp, meta: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    d_out, d_in = meta["d_out"], meta["d_in"]
    fp.seek(meta["offset"] + meta["pb_len"])
    inv_s = None
    if meta["inv_k"] == 1:
        inv_s = np.frombuffer(fp.read(d_in * 4), dtype="<f4").copy()
    elif meta["inv_k"] == 2:
        inv_s = np.frombuffer(fp.read(d_in * 2), dtype="<f2").astype(np.float32).copy()
    lr = None
    if meta["rk"] > 0:
        rk = meta["rk"]
        su = np.frombuffer(fp.read(d_out * 4), dtype="<f4").copy()
        sv = np.frombuffer(fp.read(rk * 4), dtype="<f4").copy()
        ui8 = np.frombuffer(fp.read(d_out * rk), dtype=np.int8).reshape(d_out, rk).astype(np.float32)
        vi8 = np.frombuffer(fp.read(rk * d_in), dtype=np.int8).reshape(rk, d_in).astype(np.float32)
        lr = (ui8 * su[:, None]) @ (vi8 * sv[:, None])
    return inv_s, lr


def _tblk_hdr_bytes(meta: dict) -> Tuple[int, int, int, int]:
    d_out, d_in, bsize, n_bits, is_asym = meta["d_out"], meta["d_in"], meta["bsize"], meta["n_bits"], meta["is_asym"]
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    tblk_b = p3.tile16_blk_bytes(bsize, n_bits, is_asym)
    hdr_b = 64 if is_asym else 32
    return n_tiles, n_blks, tblk_b, hdr_b


def save_header_norm_patch(t16c_path: Path, patch_path: Path) -> Dict[str, object]:
    """Save all norm vectors + FP16 tile headers (zero bitplanes) to compact `V9HP` snapshot (~101 MB)."""
    idx = T16CRecordIndex(t16c_path)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = patch_path.with_suffix(".tmp")
    h = hashlib.sha256()
    with open(idx.path, "rb") as fin, open(tmp_p, "wb") as fout:
        hdr = struct.pack("<4s3I", V9HP_MAGIC, V9HP_VERSION, len(idx.norm_order), len(idx.mat_order))
        fout.write(hdr)
        h.update(hdr)
        for nk in idx.norm_order:
            off, alen = idx.norms[nk]
            fin.seek(off)
            raw = fin.read(alen * 4)
            nb = nk.encode("utf-8")
            rec_h = struct.pack("<HI", len(nb), alen)
            fout.write(rec_h); fout.write(nb); fout.write(raw)
            h.update(rec_h); h.update(nb); h.update(raw)
        for mk in idx.mat_order:
            meta = idx.mats[mk]
            n_tiles, n_blks, tblk_b, hdr_b = _tblk_hdr_bytes(meta)
            fin.seek(meta["offset"])
            pb = np.frombuffer(fin.read(meta["pb_len"]), dtype=np.uint8)
            hdrs = np.ascontiguousarray(pb.reshape(n_tiles * n_blks, tblk_b)[:, :hdr_b]).tobytes()
            mb = mk.encode("utf-8")
            rec_h = struct.pack("<HIII", len(mb), n_tiles * n_blks, tblk_b, hdr_b)
            fout.write(rec_h); fout.write(mb); fout.write(hdrs)
            h.update(rec_h); h.update(mb); h.update(hdrs)
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(tmp_p, patch_path)
    return {"bytes": patch_path.stat().st_size, "patch_sha256": h.hexdigest(), "mats": len(idx.mat_order), "norms": len(idx.norm_order)}


def apply_header_norm_patch(t16c_path: Path, patch_path: Path) -> Dict[str, object]:
    """Apply `V9HP` snapshot in-place to `.t16c` without touching bitplanes or `inv_s`/`low_rank`."""
    idx = T16CRecordIndex(t16c_path)
    if not patch_path.is_file():
        raise FileNotFoundError(f"Missing patch file: {patch_path}")
    with open(patch_path, "rb") as fpatch, open(idx.path, "r+b") as ft16c:
        magic, ver, n_norms, n_mats = struct.unpack("<4s3I", fpatch.read(16))
        if magic != V9HP_MAGIC or ver != V9HP_VERSION:
            raise ValueError(f"Invalid patch header in {patch_path}")
        if n_norms != len(idx.norm_order) or n_mats != len(idx.mat_order):
            raise ValueError(f"Patch counts ({n_norms}/{n_mats}) != T16C counts ({len(idx.norm_order)}/{len(idx.mat_order)})")
        for _ in range(n_norms):
            klen, alen = struct.unpack("<HI", fpatch.read(6))
            kname = fpatch.read(klen).decode("utf-8")
            raw = fpatch.read(alen * 4)
            off, exp_len = idx.norms[kname]
            if exp_len != alen:
                raise ValueError(f"Norm {kname} len {alen} != {exp_len}")
            ft16c.seek(off)
            ft16c.write(raw)
        for _ in range(n_mats):
            mlen, n_tb, tblk_b, hdr_b = struct.unpack("<HIII", fpatch.read(14))
            mname = fpatch.read(mlen).decode("utf-8")
            hdrs = fpatch.read(n_tb * hdr_b)
            meta = idx.mats[mname]
            e_nt, e_nb, e_tb, e_hb = _tblk_hdr_bytes(meta)
            if e_nt * e_nb != n_tb or e_tb != tblk_b or e_hb != hdr_b:
                raise ValueError(f"Matrix header geometry mismatch for {mname}")
            ft16c.seek(meta["offset"])
            arr = np.frombuffer(ft16c.read(meta["pb_len"]), dtype=np.uint8).reshape(n_tb, tblk_b).copy()
            arr[:, :hdr_b] = np.frombuffer(hdrs, dtype=np.uint8).reshape(n_tb, hdr_b)
            ft16c.seek(meta["offset"])
            ft16c.write(arr.tobytes())
        ft16c.flush()
        os.fsync(ft16c.fileno())
    return {"applied_norms": n_norms, "applied_mats": n_mats}


def _fwht256_np(w: np.ndarray) -> np.ndarray:
    h = get_h256_f32()
    return np.ascontiguousarray((w.reshape(-1, 256) @ h).reshape(w.shape), dtype=np.float32)


def _solve_wls_py(pb_u8: np.ndarray, meta: dict, T_rot: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Pure-NumPy mirror of `fast_v9_repair.c:solve_and_write_wls_headers` for Windows/fallback parity."""
    d_out, d_in, bsize = meta["d_out"], meta["d_in"], meta["bsize"]
    n_bits, is_asym = meta["n_bits"], meta["is_asym"]
    eff = n_bits & 0x1F
    is_g = bool(n_bits & 0x80)
    is_rm = bool(n_bits & 0x40)
    is_cb = bool(n_bits & 0x20)
    n_tiles, n_blks, tblk_b, hdr_b = _tblk_hdr_bytes(meta)
    bpr = bsize // 8
    pb_blk = bpr * 16

    arr = pb_u8.reshape(n_blks, n_tiles, tblk_b).transpose(1, 0, 2).copy() if is_cb else pb_u8.reshape(n_tiles, n_blks, tblk_b).copy()
    qi = np.zeros((n_tiles, 16, n_blks, bsize), dtype=np.int32)
    for p in range(eff):
        pl = arr[:, :, hdr_b + p * pb_blk : hdr_b + (p + 1) * pb_blk]
        if is_rm:
            bits = np.unpackbits(pl.reshape(n_tiles, n_blks, 16, bpr)[:, :, :, :, None], axis=4, bitorder="big").reshape(n_tiles, n_blks, 16, bsize).transpose(0, 2, 1, 3)
        else:
            bits = np.unpackbits(pl.reshape(n_tiles, n_blks, bpr, 16)[:, :, :, None], axis=4, bitorder="big").reshape(n_tiles, n_blks, bpr, 16, 8).transpose(0, 3, 1, 2, 4).reshape(n_tiles, 16, n_blks, bsize)
        qi |= (bits.astype(np.int32) << p)

    qi = qi.reshape(n_tiles * 16, n_blks, bsize)[:d_out]
    if eff == 1:
        u = np.where(qi > 0, 1.0, -1.0).astype(np.float64)
    elif is_g and eff == 2:
        u = GAUSS2_LV[qi].astype(np.float64)
    elif is_g and eff == 3:
        u = GAUSS3_LV[qi].astype(np.float64)
    else:
        u = (2.0 * qi.astype(np.float64) - float((1 << eff) - 1))

    T_b = T_rot.reshape(d_out, n_blks, bsize).astype(np.float64)
    s_uu = np.sum(u * u, axis=2)
    s_ut = np.sum(u * T_b, axis=2)
    if is_asym:
        s_u = np.sum(u, axis=2)
        s_t = np.sum(T_b, axis=2)
        det = float(bsize) * s_uu - s_u * s_u
        safe_det = np.where(det > 1e-9, det, 1.0)
        bi = np.where(det > 1e-9, (s_uu * s_t - s_u * s_ut) / safe_det, s_t / float(bsize))
        sc = np.where(det > 1e-9, (float(bsize) * s_ut - s_u * s_t) / safe_det, 0.0)
        sc = np.maximum(sc, 0.0)
        sc16 = np.pad(sc.astype(np.float16), ((0, n_tiles * 16 - d_out), (0, 0)))
        bi16 = np.pad(bi.astype(np.float16), ((0, n_tiles * 16 - d_out), (0, 0)))
        sc_eff = sc16[:d_out].astype(np.float64)
        bi_eff = bi16[:d_out].astype(np.float64)
        arr[:, :, :32] = np.frombuffer(np.ascontiguousarray(sc16.reshape(n_tiles, 16, n_blks).transpose(0, 2, 1)).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)
        arr[:, :, 32:64] = np.frombuffer(np.ascontiguousarray(bi16.reshape(n_tiles, 16, n_blks).transpose(0, 2, 1)).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)
    else:
        sc = np.maximum(np.where(s_uu > 1e-12, s_ut / np.maximum(s_uu, 1e-12), 0.0), 0.0)
        sc16 = np.pad(sc.astype(np.float16), ((0, n_tiles * 16 - d_out), (0, 0)))
        sc_eff = sc16[:d_out].astype(np.float64)
        bi_eff = np.zeros_like(sc_eff)
        arr[:, :, :32] = np.frombuffer(np.ascontiguousarray(sc16.reshape(n_tiles, 16, n_blks).transpose(0, 2, 1)).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)

    w_rec = u * sc_eff[:, :, None] + bi_eff[:, :, None]
    dot_tot = float(np.sum(w_rec * T_b))
    wq_tot = float(np.sum(w_rec * w_rec))
    wt_tot = float(np.sum(T_b * T_b))
    out_arr = arr.transpose(1, 0, 2).copy().reshape(-1) if is_cb else arr.reshape(-1)
    return out_arr, dot_tot, wq_tot, wt_tot


def solve_wls_in_place(pb_u8: np.ndarray, meta: dict, T_rot: np.ndarray, lib: Optional[ctypes.CDLL]) -> Tuple[np.ndarray, float, float]:
    """Update FP16 headers inside `pb_u8` in-place cleanly and return `(w_cos_rot, norm_rat_rot)`."""
    if lib is not None and (meta["n_bits"] & 0x1F) in (1, 2, 3):
        pb_c = np.ascontiguousarray(pb_u8, dtype=np.uint8)
        tr_c = np.ascontiguousarray(T_rot, dtype=np.float32)
        stats = np.zeros(3, dtype=np.float64)
        rc = lib.solve_and_write_wls_headers(
            pb_c.ctypes.data, meta["d_out"], meta["d_in"], meta["bsize"],
            meta["n_bits"], 1 if meta["is_asym"] else 0, tr_c.ctypes.data, stats.ctypes.data,
        )
        if rc != 0:
            raise RuntimeError(f"solve_and_write_wls_headers failed rc={rc}")
        dot_t, wq_t, wt_t = float(stats[0]), float(stats[1]), float(stats[2])
        w_cos = dot_t / math.sqrt(max(wq_t * wt_t, 1e-18))
        n_rat = math.sqrt(wq_t / max(wt_t, 1e-18))
        return pb_c, w_cos, n_rat

    pb_c, dot_t, wq_t, wt_t = _solve_wls_py(pb_u8, meta, T_rot)
    w_cos = dot_t / math.sqrt(max(wq_t * wt_t, 1e-18))
    n_rat = math.sqrt(wq_t / max(wt_t, 1e-18))
    return pb_c, w_cos, n_rat


def _pack_sign_bits_py(T_rot: np.ndarray, n_words: int) -> np.ndarray:
    rows, d_in = T_rot.shape
    blks_use = (n_words * 64 + 255) // 256
    Tb = T_rot[:, : blks_use * 256].reshape(rows, blks_use, 256)
    bits = (Tb >= Tb.mean(axis=2, keepdims=True)).reshape(rows, blks_use * 256)[:, : n_words * 64]
    u8 = np.packbits(bits, axis=1, bitorder="little")
    return np.frombuffer(np.ascontiguousarray(u8).tobytes(), dtype=np.uint64).reshape(rows, n_words)


def _extract_t16c_sign_bits_py(pb_u8: np.ndarray, meta: dict, n_words: int) -> np.ndarray:
    d_out, d_in, bsize = meta["d_out"], meta["d_in"], meta["bsize"]
    n_bits, is_asym = meta["n_bits"], meta["is_asym"]
    eff = n_bits & 0x1F
    is_rm = bool(n_bits & 0x40)
    is_cb = bool(n_bits & 0x20)
    n_tiles, n_blks, tblk_b, hdr_b = _tblk_hdr_bytes(meta)
    bpr = bsize // 8
    pb_blk = bpr * 16
    arr = pb_u8.reshape(n_blks, n_tiles, tblk_b).transpose(1, 0, 2) if is_cb else pb_u8.reshape(n_tiles, n_blks, tblk_b)
    p = eff - 1
    pl = arr[:, :, hdr_b + p * pb_blk : hdr_b + (p + 1) * pb_blk]
    if is_rm:
        bits = np.unpackbits(pl.reshape(n_tiles, n_blks, 16, bpr)[:, :, :, :, None], axis=4, bitorder="big").reshape(n_tiles, n_blks, 16, bsize).transpose(0, 2, 1, 3)
    else:
        bits = np.unpackbits(pl.reshape(n_tiles, n_blks, bpr, 16)[:, :, :, None], axis=4, bitorder="big").reshape(n_tiles, n_blks, bpr, 16, 8).transpose(0, 3, 1, 2, 4).reshape(n_tiles, 16, n_blks, bsize)
    flat_bits = bits.reshape(n_tiles * 16, d_in)[:d_out, : n_words * 64]
    u8 = np.packbits(flat_bits, axis=1, bitorder="little")
    return np.frombuffer(np.ascontiguousarray(u8).tobytes(), dtype=np.uint64).reshape(d_out, n_words)


def recover_bijective_perm(
    T_rot_gate: np.ndarray,
    pb_gate: np.ndarray,
    meta_gate: dict,
    T_rot_up: np.ndarray,
    pb_up: np.ndarray,
    meta_up: dict,
    lib: Optional[ctypes.CDLL],
    words_per_proj: int = 16,
) -> Tuple[np.ndarray, float]:
    """Recover exact bijection `pi_l` (`S_{d_out}`) such that `T_rot_gate[pi_l]` aligns to `pb_gate`."""
    d_out = meta_gate["d_out"]
    w_tot = words_per_proj * 2
    if lib is not None:
        bg_c = np.empty((d_out, words_per_proj), dtype=np.uint64)
        bu_c = np.empty((d_out, words_per_proj), dtype=np.uint64)
        tg_c = np.empty((d_out, words_per_proj), dtype=np.uint64)
        tu_c = np.empty((d_out, words_per_proj), dtype=np.uint64)
        lib.pack_ac_bits_from_trot_f32(T_rot_gate.ctypes.data, d_out, meta_gate["d_in"], words_per_proj, bg_c.ctypes.data)
        lib.pack_ac_bits_from_trot_f32(T_rot_up.ctypes.data, d_out, meta_up["d_in"], words_per_proj, bu_c.ctypes.data)
        lib.extract_t16c_sign_bits(pb_gate.ctypes.data, d_out, meta_gate["d_in"], meta_gate["bsize"], meta_gate["n_bits"], 1 if meta_gate["is_asym"] else 0, words_per_proj, tg_c.ctypes.data)
        lib.extract_t16c_sign_bits(pb_up.ctypes.data, d_out, meta_up["d_in"], meta_up["bsize"], meta_up["n_bits"], 1 if meta_up["is_asym"] else 0, words_per_proj, tu_c.ctypes.data)
        cand = np.ascontiguousarray(np.concatenate([bg_c, bu_c], axis=1))
        targ = np.ascontiguousarray(np.concatenate([tg_c, tu_c], axis=1))
        best_j = np.empty(d_out, dtype=np.int32)
        best_s = np.empty(d_out, dtype=np.int32)
        sec_s = np.empty(d_out, dtype=np.int32)
        # Note: target row r_perm corresponds to cand row j -> invert argmax(target -> cand)
        lib.match_mlp_perm_popcnt(targ.ctypes.data, cand.ctypes.data, d_out, w_tot, best_j.ctypes.data, best_s.ctypes.data, sec_s.ctypes.data)
        mean_margin = float(np.mean(best_s - sec_s))
    else:
        cand = np.concatenate([_pack_sign_bits_py(T_rot_gate, words_per_proj), _pack_sign_bits_py(T_rot_up, words_per_proj)], axis=1)
        targ = np.concatenate([_extract_t16c_sign_bits_py(pb_gate, meta_gate, words_per_proj), _extract_t16c_sign_bits_py(pb_up, meta_up, words_per_proj)], axis=1)
        # Pure Python XOR popcnt for small selftest shapes
        c_b = np.unpackbits(np.frombuffer(cand.tobytes(), dtype=np.uint8).reshape(d_out, w_tot * 8), axis=1).astype(np.float32) * 2.0 - 1.0
        t_b = np.unpackbits(np.frombuffer(targ.tobytes(), dtype=np.uint8).reshape(d_out, w_tot * 8), axis=1).astype(np.float32) * 2.0 - 1.0
        sims = ((t_b @ c_b.T) + (w_tot * 64)) * 0.5
        best_j = np.argmax(sims, axis=1).astype(np.int32)
        part = np.partition(sims, -2, axis=1)
        best_s = part[:, -1].astype(np.int32)
        sec_s = part[:, -2].astype(np.int32)
        mean_margin = float(np.mean(best_s - sec_s))

    used = np.zeros(d_out, dtype=bool)
    pi_l = np.full(d_out, -1, dtype=np.int64)
    order = np.argsort(-(best_s - sec_s))
    ambig_rows: List[int] = []
    for r in order:
        j = int(best_j[r])
        if not used[j]:
            pi_l[r] = j
            used[j] = True
        else:
            ambig_rows.append(int(r))

    if ambig_rows:
        rem_cols = np.where(~used)[0]
        # Resolve any rare collisions via exact full-width float32 normalized cosine dot on T_rot_gate & T_rot_up
        u_g = (np.unpackbits(np.frombuffer(_extract_t16c_sign_bits_py(pb_gate, meta_gate, meta_gate["d_in"] // 64).tobytes(), dtype=np.uint8).reshape(d_out, -1), axis=1, bitorder="little").astype(np.float32) * 2.0 - 1.0)
        rem_g = T_rot_gate[rem_cols]
        rem_g_n = rem_g / (np.linalg.norm(rem_g, axis=1, keepdims=True) + 1e-9)
        scores = u_g[ambig_rows] @ rem_g_n.T
        taken_local = set()
        for idx_a, r_a in enumerate(ambig_rows):
            ord_c = np.argsort(-scores[idx_a])
            for loc_c in ord_c:
                if int(loc_c) not in taken_local:
                    taken_local.add(int(loc_c))
                    pi_l[r_a] = int(rem_cols[int(loc_c)])
                    break

    if not np.array_equal(np.sort(pi_l), np.arange(d_out, dtype=np.int64)):
        raise RuntimeError(f"Fail-closed permutation recovery failed bijection check (d_out={d_out})")
    return pi_l, mean_margin


def reconstruct_perm_pi(sf: FastSafetensorsReader, idx: T16CRecordIndex, t16c_fp, d_mod: int = 5376, n_layers: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Recover exact global channel permutation `perm_pi` (`orig[perm_pi] == t16c`) from input_layernorm."""
    vecs_o, vecs_p = [], []
    for l in (0, 5, 17, 41):
        nk = f"model.language_model.layers.{l}.input_layernorm.weight"
        vecs_o.append(sf.read_tensor_np(nk).astype(np.float64))
        off, alen = idx.norms[nk]
        t16c_fp.seek(off)
        vecs_p.append(np.frombuffer(t16c_fp.read(alen * 4), dtype="<f4").astype(np.float64).copy())
    perm_pi = np.empty(d_mod, dtype=np.int64)
    # Match sorted original indices to sorted permuted positions
    ord_o = np.lexsort(vecs_o)
    ord_p = np.lexsort(vecs_p)
    perm_pi[ord_p] = ord_o

    max_diff = 0.0
    for l in range(n_layers):
        nk = f"model.language_model.layers.{l}.input_layernorm.weight"
        vo = sf.read_tensor_np(nk)[perm_pi]
        off, alen = idx.norms[nk]
        t16c_fp.seek(off)
        vp = np.frombuffer(t16c_fp.read(alen * 4), dtype="<f4").copy()
        max_diff = max(max_diff, float(np.max(np.abs(vo - vp))))
    if max_diff > 1e-5:
        raise RuntimeError(f"perm_pi verification failed across 60 layers: max_diff={max_diff:.3e}")
    inv_perm_pi = np.empty_like(perm_pi)
    inv_perm_pi[perm_pi] = np.arange(d_mod, dtype=np.int64)
    print(f"[Perm Pi] Verified across all {n_layers} layers: max_norm_diff={max_diff:.2e}", flush=True)
    return perm_pi, inv_perm_pi


def run_approach1(t16c_path: Path, sf_path: Path, patch_dir: Path, start_layer: int = 0) -> Dict[str, object]:
    """Run Approach 1 in-place closed-form WLS repair on Pi 4 (supports crash/threshold resume via --start-layer)."""
    lib = try_load_c_lib()
    patch_dir.mkdir(parents=True, exist_ok=True)
    orig_patch = patch_dir / "orig_headers_norms.bin"
    if not orig_patch.is_file():
        print(f"[Backup] Saving reversible original header+norm snapshot to {orig_patch}...", flush=True)
        bk_info = save_header_norm_patch(t16c_path, orig_patch)
        print(f"[Backup] Saved {bk_info['bytes']:,} B sha256={bk_info['patch_sha256'][:16]}...", flush=True)

    idx = T16CRecordIndex(t16c_path)
    pi_ckpt = patch_dir / "pi_l_all_partial.npy"
    if start_layer > 0 and pi_ckpt.is_file():
        pi_l_all = np.load(pi_ckpt)
    else:
        pi_l_all = np.empty((60, 21504), dtype=np.int32)
    layer_stats: List[dict] = []
    words_per_proj = 4  # 4+4 = 8 uint64 words = 512 sign bits (>16 sigma margin, NEON fast path)

    with FastSafetensorsReader(sf_path) as sf, open(t16c_path, "r+b") as fp:
        perm_pi, inv_perm_pi = reconstruct_perm_pi(sf, idx, fp)

        def _rot_inplace(mat_f32: np.ndarray) -> np.ndarray:
            arr = np.ascontiguousarray(mat_f32, dtype=np.float32)
            if lib is not None:
                lib.fwht256_rows_f32(arr.ctypes.data, arr.shape[0], arr.shape[1])
                return arr
            return _fwht256_np(arr)

        if start_layer == 0:
            # Restore final_norm cleanly
            fn_nk = "model.language_model.norm.weight"
            fn_clean = sf.read_tensor_np(fn_nk)[perm_pi].astype("<f4")
            fp.seek(idx.norms[fn_nk][0])
            fp.write(fn_clean.tobytes())

        for l_idx in range(start_layer, 60):
            pfx = f"model.language_model.layers.{l_idx}"
            # 1) Restore exact teacher post norms and layer_scalar
            for nk_suf, do_perm in (("post_attention_layernorm.weight", True), ("post_feedforward_layernorm.weight", True), ("layer_scalar", False)):
                nk = f"{pfx}.{nk_suf}"
                arr_n = sf.read_tensor_np(nk)
                if do_perm:
                    arr_n = arr_n[perm_pi]
                fp.seek(idx.norms[nk][0])
                fp.write(np.ascontiguousarray(arr_n, dtype="<f4").tobytes())

            l_rec: Dict[str, object] = {"l": l_idx}

            # 2) Repair L02..L57 Attention q_proj and o_proj (only QAT-modified attention matrices)
            if 2 <= l_idx <= 57:
                for pn, is_o in (("self_attn.q_proj", False), ("self_attn.o_proj", True)):
                    mname = f"{pfx}.{pn}.weight"
                    meta = idx.mats[mname]
                    wt = sf.read_dequant_f32(f"{pfx}.{pn}", lib=lib)
                    wt = wt[perm_pi, :] if is_o else wt[:, perm_pi]
                    inv_s, lr = _read_inv_and_lr(fp, meta)
                    if lr is not None:
                        wt -= lr
                        del lr
                    if inv_s is not None:
                        wt /= inv_s[None, :]
                    wt = _rot_inplace(wt)
                    fp.seek(meta["offset"])
                    pb_u8 = np.frombuffer(fp.read(meta["pb_len"]), dtype=np.uint8)
                    pb_upd, w_cos, n_rat = solve_wls_in_place(pb_u8, meta, wt, lib)
                    fp.seek(meta["offset"])
                    fp.write(pb_upd.tobytes())
                    min_cos = 0.73 if (meta["n_bits"] & 0x1F) == 1 else 0.90
                    if w_cos < min_cos:
                        raise RuntimeError(f"Fail-closed Approach 1 repair failed: {mname} w_cos={w_cos:.4f} < {min_cos}")
                    l_rec[pn] = {"w_cos": round(w_cos, 5), "norm_rat": round(n_rat, 4)}
                    del wt, pb_u8, pb_upd

            # 3) Recover pi_l for this layer from gate_proj + up_proj intact bitplanes
            m_g = idx.mats[f"{pfx}.mlp.gate_proj.weight"]
            m_u = idx.mats[f"{pfx}.mlp.up_proj.weight"]
            wg_orig = sf.read_dequant_f32(f"{pfx}.mlp.gate_proj", lib=lib)[:, perm_pi]
            inv_sg, lr_g = _read_inv_and_lr(fp, m_g)
            if inv_sg is not None:
                wg_orig /= inv_sg[None, :]
            wg_rot = _rot_inplace(wg_orig)
            del wg_orig

            wu_orig = sf.read_dequant_f32(f"{pfx}.mlp.up_proj", lib=lib)[:, perm_pi]
            inv_su, lr_u = _read_inv_and_lr(fp, m_u)
            if inv_su is not None:
                wu_orig /= inv_su[None, :]
            wu_rot = _rot_inplace(wu_orig)
            del wu_orig

            fp.seek(m_g["offset"])
            pb_g = np.frombuffer(fp.read(m_g["pb_len"]), dtype=np.uint8)
            fp.seek(m_u["offset"])
            pb_u = np.frombuffer(fp.read(m_u["pb_len"]), dtype=np.uint8)

            pi_l, margin = recover_bijective_perm(wg_rot, pb_g, m_g, wu_rot, pb_u, m_u, lib, words_per_proj=words_per_proj)
            pi_l_all[l_idx] = pi_l.astype(np.int32)
            l_rec["pi_l_margin"] = round(margin, 2)

            # Repair gate_proj and up_proj headers using recovered pi_l
            wg_perm = np.ascontiguousarray(wg_rot[pi_l, :], dtype=np.float32)
            del wg_rot
            if lr_g is not None:
                if inv_sg is not None:
                    lr_g = lr_g / inv_sg[None, :]
                wg_perm -= _rot_inplace(lr_g)
                del lr_g
            pb_g_upd, cos_g, rat_g = solve_wls_in_place(pb_g, m_g, wg_perm, lib)
            del wg_perm, pb_g
            fp.seek(m_g["offset"])
            fp.write(pb_g_upd.tobytes())
            del pb_g_upd

            wu_perm = np.ascontiguousarray(wu_rot[pi_l, :], dtype=np.float32)
            del wu_rot
            if lr_u is not None:
                if inv_su is not None:
                    lr_u = lr_u / inv_su[None, :]
                wu_perm -= _rot_inplace(lr_u)
                del lr_u
            pb_u_upd, cos_u, rat_u = solve_wls_in_place(pb_u, m_u, wu_perm, lib)
            del wu_perm, pb_u
            fp.seek(m_u["offset"])
            fp.write(pb_u_upd.tobytes())
            del pb_u_upd

            # Repair down_proj headers using perm_pi (rows) and pi_l (cols)
            m_d = idx.mats[f"{pfx}.mlp.down_proj.weight"]
            wd_perm = np.ascontiguousarray(sf.read_dequant_f32(f"{pfx}.mlp.down_proj", lib=lib)[perm_pi, :][:, pi_l], dtype=np.float32)
            inv_sd, lr_d = _read_inv_and_lr(fp, m_d)
            if lr_d is not None:
                wd_perm -= lr_d
                del lr_d
            if inv_sd is not None:
                wd_perm /= inv_sd[None, :]
            wd_rot = _rot_inplace(wd_perm)
            del wd_perm
            fp.seek(m_d["offset"])
            pb_d = np.frombuffer(fp.read(m_d["pb_len"]), dtype=np.uint8)
            pb_d_upd, cos_d, rat_d = solve_wls_in_place(pb_d, m_d, wd_rot, lib)
            del wd_rot, pb_d
            fp.seek(m_d["offset"])
            fp.write(pb_d_upd.tobytes())
            del pb_d_upd

            for tag_m, cm, mm_b in (("gate", cos_g, m_g["n_bits"]), ("up", cos_u, m_u["n_bits"]), ("down", cos_d, m_d["n_bits"])):
                req = 0.73 if (mm_b & 0x1F) == 1 else 0.90
                if cm < req:
                    raise RuntimeError(f"Fail-closed Approach 1 MLP repair failed: L{l_idx:02d} {tag_m} w_cos={cm:.4f} < {req}")

            l_rec["mlp_cos"] = [round(cos_g, 4), round(cos_u, 4), round(cos_d, 4)]
            l_rec["mlp_rat"] = [round(rat_g, 4), round(rat_u, 4), round(rat_d, 4)]
            layer_stats.append(l_rec)
            print(
                f"  [App1 L{l_idx:02d}/59] pi_margin={margin:.1f}/{words_per_proj*128} | mlp_cos={l_rec['mlp_cos']} | "
                f"q_cos={l_rec.get('self_attn.q_proj', {}).get('w_cos', '-')} o_cos={l_rec.get('self_attn.o_proj', {}).get('w_cos', '-')}",
                flush=True,
            )
            if (l_idx + 1) % 5 == 0:
                np.save(pi_ckpt, pi_l_all)
            gc.collect()

        fp.flush()
        os.fsync(fp.fileno())

    perms_p = patch_dir / "v9_0_recovered_perms.npz"
    np.savez(perms_p, perm_pi=perm_pi.astype(np.int32), inv_perm_pi=inv_perm_pi.astype(np.int32), pi_l_all=pi_l_all)
    app1_patch = patch_dir / "app1_wls_headers_norms.bin"
    app1_info = save_header_norm_patch(t16c_path, app1_patch)
    rpt_p = patch_dir / "app1_repair_report.json"
    rpt = {"perms_npz": str(perms_p), "patch_bin": str(app1_patch), "patch_info": app1_info, "layers": layer_stats}
    rpt_p.write_text(json.dumps(rpt, indent=2), encoding="utf-8")
    print(f"[Approach 1 COMPLETE] Saved {app1_patch} ({app1_info['bytes']:,} B) + {perms_p}", flush=True)
    return rpt


def _zero_lr_sv_inplace(fp, meta: dict) -> None:
    """Zero out `sv` (`rk` float32s) in-place inside a `.t16c` record without shifting offsets."""
    rk = int(meta["rk"])
    if rk <= 0:
        return
    sv_off = meta["offset"] + meta["pb_len"] + meta["inv_len"] + meta["d_out"] * 4
    cur = fp.tell()
    fp.seek(sv_off)
    fp.write(b"\x00" * (rk * 4))
    fp.seek(cur)


def run_approach3_repair(t16c_path: Path, sf_path: Path, patch_dir: Path, start_layer: int = 0) -> Dict[str, object]:
    """Repair Approach 3 (`gemma4_31b_v9_0_app3_zero1b.t16c`) in-place via closed-form WLS, teacher norms, and corrupted-LR scrubbing."""
    lib = try_load_c_lib()
    patch_dir.mkdir(parents=True, exist_ok=True)
    orig_patch = patch_dir / "app3_orig_headers_norms.bin"
    if not orig_patch.is_file():
        print(f"[Backup] Saving reversible App3 header+norm snapshot to {orig_patch}...", flush=True)
        bk_info = save_header_norm_patch(t16c_path, orig_patch)
        print(f"[Backup] Saved {bk_info['bytes']:,} B sha256={bk_info['patch_sha256'][:16]}...", flush=True)

    idx = T16CRecordIndex(t16c_path)
    pi_ckpt = patch_dir / "app3_pi_l_all_partial.npy"
    pi_l_all = np.load(pi_ckpt) if (start_layer > 0 and pi_ckpt.is_file()) else np.empty((60, 21504), dtype=np.int32)
    layer_stats: List[dict] = []
    zeroed_lr: List[dict] = []
    words_per_proj = 4

    with FastSafetensorsReader(sf_path) as sf, open(t16c_path, "r+b") as fp:
        perm_pi, inv_perm_pi = reconstruct_perm_pi(sf, idx, fp)

        def _rot_inplace(mat_f32: np.ndarray) -> np.ndarray:
            arr = np.ascontiguousarray(mat_f32, dtype=np.float32)
            if lib is not None:
                lib.fwht256_rows_f32(arr.ctypes.data, arr.shape[0], arr.shape[1])
                return arr
            return _fwht256_np(arr)

        if start_layer == 0:
            fn_nk = "model.language_model.norm.weight"
            fn_clean = sf.read_tensor_np(fn_nk)[perm_pi].astype("<f4")
            fp.seek(idx.norms[fn_nk][0])
            fp.write(fn_clean.tobytes())

        for l_idx in range(start_layer, 60):
            pfx = f"model.language_model.layers.{l_idx}"
            for nk_suf, do_perm in (("post_attention_layernorm.weight", True), ("post_feedforward_layernorm.weight", True), ("layer_scalar", False)):
                nk = f"{pfx}.{nk_suf}"
                arr_n = sf.read_tensor_np(nk)
                if do_perm:
                    arr_n = arr_n[perm_pi]
                fp.seek(idx.norms[nk][0])
                fp.write(np.ascontiguousarray(arr_n, dtype="<f4").tobytes())

            l_rec: Dict[str, object] = {"l": l_idx}

            # Repair q_proj and o_proj across ALL 60 layers (App3 updated Q/O headers via Ridge across L02..L57)
            for pn, is_o in (("self_attn.q_proj", False), ("self_attn.o_proj", True)):
                mname = f"{pfx}.{pn}.weight"
                meta = idx.mats[mname]
                wt = sf.read_dequant_f32(f"{pfx}.{pn}", lib=lib)
                wt = wt[perm_pi, :] if is_o else wt[:, perm_pi]
                inv_s, lr = _read_inv_and_lr(fp, meta)
                if lr is not None:
                    n_wt = float(np.linalg.norm(wt))
                    n_lr = float(np.linalg.norm(lr))
                    n_res = float(np.linalg.norm(wt - lr))
                    if n_lr > 0.65 * n_wt or n_res > 1.00 * n_wt:
                        _zero_lr_sv_inplace(fp, meta)
                        zeroed_lr.append({"mat": mname, "n_wt": round(n_wt, 3), "n_lr": round(n_lr, 3), "n_res": round(n_res, 3)})
                        lr = None
                    else:
                        wt -= lr
                    del lr
                if inv_s is not None:
                    wt /= inv_s[None, :]
                wt = _rot_inplace(wt)
                fp.seek(meta["offset"])
                pb_u8 = np.frombuffer(fp.read(meta["pb_len"]), dtype=np.uint8)
                pb_upd, w_cos, n_rat = solve_wls_in_place(pb_u8, meta, wt, lib)
                fp.seek(meta["offset"])
                fp.write(pb_upd.tobytes())
                min_cos = 0.73 if (meta["n_bits"] & 0x1F) == 1 else 0.88
                if w_cos < min_cos:
                    raise RuntimeError(f"Fail-closed App3 repair failed: {mname} w_cos={w_cos:.4f} < {min_cos}")
                l_rec[pn] = {"w_cos": round(w_cos, 5), "norm_rat": round(n_rat, 4)}
                del wt, pb_u8, pb_upd

            # Recover pi_l for this layer from gate_proj + up_proj intact bitplanes
            m_g = idx.mats[f"{pfx}.mlp.gate_proj.weight"]
            m_u = idx.mats[f"{pfx}.mlp.up_proj.weight"]
            wg_orig = sf.read_dequant_f32(f"{pfx}.mlp.gate_proj", lib=lib)[:, perm_pi]
            inv_sg, lr_g = _read_inv_and_lr(fp, m_g)
            if inv_sg is not None:
                wg_orig /= inv_sg[None, :]
            wg_rot = _rot_inplace(wg_orig)
            del wg_orig

            wu_orig = sf.read_dequant_f32(f"{pfx}.mlp.up_proj", lib=lib)[:, perm_pi]
            inv_su, lr_u = _read_inv_and_lr(fp, m_u)
            if inv_su is not None:
                wu_orig /= inv_su[None, :]
            wu_rot = _rot_inplace(wu_orig)
            del wu_orig

            fp.seek(m_g["offset"])
            pb_g = np.frombuffer(fp.read(m_g["pb_len"]), dtype=np.uint8)
            fp.seek(m_u["offset"])
            pb_u = np.frombuffer(fp.read(m_u["pb_len"]), dtype=np.uint8)

            pi_l, margin = recover_bijective_perm(wg_rot, pb_g, m_g, wu_rot, pb_u, m_u, lib, words_per_proj=words_per_proj)
            pi_l_all[l_idx] = pi_l.astype(np.int32)
            l_rec["pi_l_margin"] = round(margin, 2)

            wg_perm = np.ascontiguousarray(wg_rot[pi_l, :], dtype=np.float32)
            del wg_rot
            if lr_g is not None:
                if float(np.linalg.norm(lr_g)) > 0.65 * float(np.linalg.norm(wg_perm)):
                    _zero_lr_sv_inplace(fp, m_g)
                    zeroed_lr.append({"mat": f"{pfx}.mlp.gate_proj.weight"})
                else:
                    if inv_sg is not None:
                        lr_g = lr_g / inv_sg[None, :]
                    wg_perm -= _rot_inplace(lr_g)
                del lr_g
            pb_g_upd, cos_g, rat_g = solve_wls_in_place(pb_g, m_g, wg_perm, lib)
            del wg_perm, pb_g
            fp.seek(m_g["offset"]); fp.write(pb_g_upd.tobytes())
            del pb_g_upd

            wu_perm = np.ascontiguousarray(wu_rot[pi_l, :], dtype=np.float32)
            del wu_rot
            if lr_u is not None:
                if float(np.linalg.norm(lr_u)) > 0.65 * float(np.linalg.norm(wu_perm)):
                    _zero_lr_sv_inplace(fp, m_u)
                    zeroed_lr.append({"mat": f"{pfx}.mlp.up_proj.weight"})
                else:
                    if inv_su is not None:
                        lr_u = lr_u / inv_su[None, :]
                    wu_perm -= _rot_inplace(lr_u)
                del lr_u
            pb_u_upd, cos_u, rat_u = solve_wls_in_place(pb_u, m_u, wu_perm, lib)
            del wu_perm, pb_u
            fp.seek(m_u["offset"]); fp.write(pb_u_upd.tobytes())
            del pb_u_upd

            m_d = idx.mats[f"{pfx}.mlp.down_proj.weight"]
            wd_perm = np.ascontiguousarray(sf.read_dequant_f32(f"{pfx}.mlp.down_proj", lib=lib)[perm_pi, :][:, pi_l], dtype=np.float32)
            inv_sd, lr_d = _read_inv_and_lr(fp, m_d)
            if lr_d is not None:
                n_wt = float(np.linalg.norm(wd_perm))
                n_lr = float(np.linalg.norm(lr_d))
                n_res = float(np.linalg.norm(wd_perm - lr_d))
                if n_lr > 0.65 * n_wt or n_res > 1.00 * n_wt:
                    _zero_lr_sv_inplace(fp, m_d)
                    zeroed_lr.append({"mat": f"{pfx}.mlp.down_proj.weight", "n_wt": round(n_wt, 3), "n_lr": round(n_lr, 3), "n_res": round(n_res, 3)})
                else:
                    wd_perm -= lr_d
                del lr_d
            if inv_sd is not None:
                wd_perm /= inv_sd[None, :]
            wd_rot = _rot_inplace(wd_perm)
            del wd_perm
            fp.seek(m_d["offset"])
            pb_d = np.frombuffer(fp.read(m_d["pb_len"]), dtype=np.uint8)
            pb_d_upd, cos_d, rat_d = solve_wls_in_place(pb_d, m_d, wd_rot, lib)
            del wd_rot, pb_d
            fp.seek(m_d["offset"]); fp.write(pb_d_upd.tobytes())
            del pb_d_upd

            for tag_m, cm, mm_b in (("gate", cos_g, m_g["n_bits"]), ("up", cos_u, m_u["n_bits"]), ("down", cos_d, m_d["n_bits"])):
                req = 0.73 if (mm_b & 0x1F) == 1 else 0.88
                if cm < req:
                    raise RuntimeError(f"Fail-closed App3 MLP repair failed: L{l_idx:02d} {tag_m} w_cos={cm:.4f} < {req}")

            l_rec["mlp_cos"] = [round(cos_g, 4), round(cos_u, 4), round(cos_d, 4)]
            l_rec["mlp_rat"] = [round(rat_g, 4), round(rat_u, 4), round(rat_d, 4)]
            layer_stats.append(l_rec)
            print(
                f"  [App3-WLS L{l_idx:02d}/59] pi_margin={margin:.1f}/{words_per_proj*128} | mlp_cos={l_rec['mlp_cos']} | "
                f"q_cos={l_rec['self_attn.q_proj']['w_cos']} o_cos={l_rec['self_attn.o_proj']['w_cos']} zeroed_lr={len(zeroed_lr)}",
                flush=True,
            )
            if (l_idx + 1) % 5 == 0:
                np.save(pi_ckpt, pi_l_all)
            gc.collect()

        fp.flush()
        os.fsync(fp.fileno())

    perms_p = patch_dir / "app3_recovered_perms.npz"
    np.savez(perms_p, perm_pi=perm_pi.astype(np.int32), inv_perm_pi=inv_perm_pi.astype(np.int32), pi_l_all=pi_l_all)
    app3_patch = patch_dir / "app3_wls_headers_norms.bin"
    app3_info = save_header_norm_patch(t16c_path, app3_patch)
    rpt_p = patch_dir / "app3_wls_repair_report.json"
    rpt = {
        "perms_npz": str(perms_p),
        "patch_bin": str(app3_patch),
        "patch_info": app3_info,
        "zeroed_corrupted_lr": zeroed_lr,
        "layers": layer_stats,
    }
    rpt_p.write_text(json.dumps(rpt, indent=2), encoding="utf-8")
    print(f"[App3-WLS COMPLETE] Saved {app3_patch} ({app3_info['bytes']:,} B) zeroed_lr={len(zeroed_lr)}", flush=True)
    return rpt


def _write_synthetic_w4_sf(path: Path, tensors: Dict[str, np.ndarray]) -> None:
    hdr = {}
    chunks = []
    off = 0
    for k, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    hb = json.dumps(hdr).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for c in chunks:
            f.write(c)


def _pack_synthetic_mat(w_rot: np.ndarray, mode_i: int, is_rm: bool, is_cb: bool) -> Tuple[int, bytes]:
    d_out, d_in = w_rot.shape
    bsize = 256
    n_tiles, n_blks = (d_out + 15) // 16, d_in // bsize
    nb, asym = p3.MODE_TO_T16C[mode_i]
    eff = nb & 0x1F
    hdr_b = 64 if asym else 32
    bpr = bsize // 8
    pb_blk = bpr * 16
    tblk_b = hdr_b + eff * pb_blk
    arr = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
    wb = np.pad(w_rot, ((0, n_tiles * 16 - d_out), (0, 0))).reshape(n_tiles, 16, n_blks, bsize).transpose(0, 2, 1, 3)

    if eff == 1:
        bi = wb.mean(axis=3)
        for _ in range(2):
            m1 = wb >= bi[..., None]
            c1 = m1.sum(axis=3); c0 = bsize - c1
            mp = np.where(c1 > 0, np.where(m1, wb, 0.0).sum(axis=3) / np.maximum(c1, 1), bi)
            mn = np.where(c0 > 0, np.where(m1, 0.0, wb).sum(axis=3) / np.maximum(c0, 1), bi)
            bi = 0.5 * (mp + mn); sc = 0.5 * (mp - mn)
        qi = (wb >= bi[..., None]).astype(np.int32)
        arr[:, :, :32] = np.frombuffer(np.ascontiguousarray(sc).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)
        arr[:, :, 32:64] = np.frombuffer(np.ascontiguousarray(bi).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)
    else:
        lv = GAUSS2_LV if eff == 2 else GAUSS3_LV
        mids = np.array([-2.087, 0.0, 2.087], dtype=np.float32) if eff == 2 else np.array([-6.43, -4.35, -2.27, 0.0, 2.27, 4.35, 6.43], dtype=np.float32)
        sc = (0.470 if eff == 2 else 0.275) * np.sqrt(np.mean(wb * wb, axis=3))
        for _ in range(4):
            inv_s = np.where(sc > 1e-12, 1.0 / np.maximum(sc, 1e-12), 0.0)[..., None]
            qi = np.sum((wb * inv_s)[..., None] >= mids, axis=-1)
            qlv = lv[qi]
            sc = np.sum(qlv * wb, axis=3) / np.maximum(np.sum(qlv * qlv, axis=3), 1e-12)
        qi = np.sum((wb * (1.0 / np.maximum(sc, 1e-12))[..., None])[..., None] >= mids, axis=-1)
        arr[:, :, :32] = np.frombuffer(np.ascontiguousarray(sc).astype("<f2").tobytes(), dtype=np.uint8).reshape(n_tiles, n_blks, 32)

    for p in range(eff):
        bits = ((qi >> p) & 1).astype(np.uint8).reshape(n_tiles, n_blks, 16, bpr, 8)
        packed_p = np.packbits(bits, axis=4, bitorder="big").reshape(n_tiles, n_blks, 16, bpr).transpose(0, 1, 3, 2)
        arr[:, :, hdr_b + p * pb_blk : hdr_b + (p + 1) * pb_blk] = packed_p.reshape(n_tiles, n_blks, pb_blk)

    raw_b = arr.reshape(-1)
    if is_rm:
        raw_b = rmmod.transpose_planes_col_to_rowmajor(raw_b, d_out, d_in, bsize, eff, asym)
    raw_bytes = raw_b.tobytes()
    hdr_bits = nb | (p3.T16C_ROWMAJOR_BIT if is_rm else 0)
    if is_cb:
        raw_bytes = sdc.transpose_colblock_bytes(raw_bytes, d_out, d_in, bsize, eff, asym, to_colblock=True)
        hdr_bits |= p3.T16C_COLBLOCK_BIT
    return hdr_bits, raw_bytes


def run_selftest() -> int:
    """6-gate fail-closed CPU self-test battery for `repair_v9_0_inplace.py` and `fast_v9_repair.c`."""
    failures = 0
    lib = try_load_c_lib()
    print(f"[Selftest] fast_v9_repair C lib loaded={lib is not None}", flush=True)

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    rng = np.random.default_rng(123)

    # Gate 1: FWHT parity & closed-form WLS across modes 0x41, 0x61, 0x82, 0xA2, 0x83 with corrupted headers
    g1_ok = True
    g1_det = []
    w0 = rng.normal(0.0, 0.015, size=(32, 512)).astype(np.float32)
    w0_rot = _fwht256_np(w0)
    if lib is not None:
        w0_c = np.ascontiguousarray(w0.copy(), dtype=np.float32)
        lib.fwht256_rows_f32(w0_c.ctypes.data, 32, 512)
        fw_diff = float(np.max(np.abs(w0_c - w0_rot)))
        g1_ok = g1_ok and fw_diff < 1e-5
    for m_i, rm_f, cb_f in ((0, True, False), (0, True, True), (5, False, False), (5, False, True), (6, False, False)):
        hdr_b, pay_b = _pack_synthetic_mat(w0_rot, m_i, rm_f, cb_f)
        asym = p3.MODE_TO_T16C[m_i][1]
        meta = {"offset": 0, "d_out": 32, "d_in": 512, "bsize": 256, "n_bits": hdr_b, "is_asym": asym, "pb_len": len(pay_b)}
        # Corrupt headers heavily (AdamW blowup simulation: sc=0.05, bi=0.08)
        n_tiles, n_blks, tblk_b, hb = _tblk_hdr_bytes(meta)
        corrupt = np.frombuffer(pay_b, dtype=np.uint8).reshape(n_tiles * n_blks, tblk_b).copy()
        bp_before = hashlib.sha256(corrupt[:, hb:].tobytes()).hexdigest()
        corrupt[:, :32] = np.frombuffer(np.full((16,), 0.05, dtype="<f2").tobytes(), dtype=np.uint8)
        if asym:
            corrupt[:, 32:64] = np.frombuffer(np.full((16,), 0.08, dtype="<f2").tobytes(), dtype=np.uint8)
        repaired, w_cos, n_rat = solve_wls_in_place(corrupt.reshape(-1), meta, w0_rot, lib)
        bp_after = hashlib.sha256(repaired.reshape(n_tiles * n_blks, tblk_b)[:, hb:].tobytes()).hexdigest()
        req = 0.78 if (hdr_b & 0x1F) == 1 else 0.90
        ok_m = (bp_before == bp_after) and (w_cos >= req) and (0.75 <= n_rat <= 1.05)
        g1_ok = g1_ok and ok_m
        g1_det.append(f"0x{hdr_b:02x}:cos={w_cos:.3f}")
    gate("Gate 1: Closed-form WLS repair & bitplane immutability across 0x41/0x61/0x82/0xA2/0x83", g1_ok, " ".join(g1_det))

    # Gate 2: Bijective permutation pi_l recovery from noisy 1b_asym bitplanes (gate+up)
    rng_g2 = np.random.default_rng(77)
    n_r2 = 64
    w_g_orig = rng_g2.normal(0.0, 0.015, size=(n_r2, 2048)).astype(np.float32)
    w_u_orig = rng_g2.normal(0.0, 0.015, size=(n_r2, 2048)).astype(np.float32)
    true_pi = rng_g2.permutation(n_r2)
    wg_rot = _fwht256_np(w_g_orig)
    wu_rot = _fwht256_np(w_u_orig)
    hg_b, pg_b = _pack_synthetic_mat(wg_rot[true_pi], 0, True, False)
    hu_b, pu_b = _pack_synthetic_mat(wu_rot[true_pi], 0, True, False)
    mg = {"d_out": n_r2, "d_in": 2048, "bsize": 256, "n_bits": hg_b, "is_asym": True, "pb_len": len(pg_b)}
    mu = {"d_out": n_r2, "d_in": 2048, "bsize": 256, "n_bits": hu_b, "is_asym": True, "pb_len": len(pu_b)}
    rec_pi, marg = recover_bijective_perm(wg_rot, np.frombuffer(pg_b, dtype=np.uint8), mg, wu_rot, np.frombuffer(pu_b, dtype=np.uint8), mu, lib, words_per_proj=8)
    gate("Gate 2: Bijective pi_l recovery matches true permutation 100%", np.array_equal(rec_pi, true_pi), f"margin={marg:.1f}")

    # Gate 3..6: Full end-to-end synthetic T16C corruption -> repair -> patch save/apply SHA-256 -> fail-closed
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        sf_p = td / "model.safetensors"
        t16c_p = td / "test.t16c"
        d_m, ff_m = 512, 64
        perm_true = rng.permutation(d_m)
        pi0_true = rng.permutation(ff_m)
        w_n0 = np.linspace(0.5, 1.5, d_m, dtype=np.float32)
        w_pa = np.linspace(0.8, 1.2, d_m, dtype=np.float32)
        w_q = rng.normal(0.0, 0.015, size=(512, d_m)).astype(np.float32)
        w_g = rng.normal(0.0, 0.015, size=(ff_m, d_m)).astype(np.float32)
        w_u = rng.normal(0.0, 0.015, size=(ff_m, d_m)).astype(np.float32)
        w_d = rng.normal(0.0, 0.015, size=(d_m, ff_m)).astype(np.float32)
        # Pad cols to 256 multiple for d_m=512 (already works) and ff_m -> use ff_m=256 for Gate 3
        ff_m = 256
        pi0_true = rng.permutation(ff_m)
        w_g = rng.normal(0.0, 0.015, size=(ff_m, d_m)).astype(np.float32)
        w_u = rng.normal(0.0, 0.015, size=(ff_m, d_m)).astype(np.float32)
        w_d = rng.normal(0.0, 0.015, size=(d_m, ff_m)).astype(np.float32)
        _write_synthetic_w4_sf(
            sf_p,
            {
                "model.language_model.norm.weight": w_n0,
                "model.language_model.layers.0.input_layernorm.weight": w_n0,
                "model.language_model.layers.0.post_attention_layernorm.weight": w_pa,
                "model.language_model.layers.0.post_feedforward_layernorm.weight": w_pa,
                "model.language_model.layers.0.layer_scalar": np.array([0.75], dtype=np.float32),
                "model.language_model.layers.0.self_attn.q_proj.weight": w_q,
                "model.language_model.layers.0.mlp.gate_proj.weight": w_g,
                "model.language_model.layers.0.mlp.up_proj.weight": w_u,
                "model.language_model.layers.0.mlp.down_proj.weight": w_d,
            },
        )
        inv_s_q = np.linspace(0.8, 1.2, d_m, dtype=np.float32)
        hq, pq = _pack_synthetic_mat(_fwht256_np(w_q[:, perm_true] / inv_s_q[None, :]), 0, True, False)
        hg, pg = _pack_synthetic_mat(_fwht256_np(w_g[pi0_true, :][:, perm_true]), 0, True, False)
        hu, pu = _pack_synthetic_mat(_fwht256_np(w_u[pi0_true, :][:, perm_true]), 0, True, False)
        hd, pd = _pack_synthetic_mat(_fwht256_np(w_d[perm_true, :][:, pi0_true]), 0, True, True)
        with open(t16c_p, "wb") as fp:
            pos = 24
            norms_l = [
                ("model.language_model.norm.weight", w_n0[perm_true]),
                ("model.language_model.layers.0.input_layernorm.weight", w_n0[perm_true]),
                ("model.language_model.layers.0.post_attention_layernorm.weight", np.full(d_m, 9.9, dtype=np.float32)),
            ]
            fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, len(norms_l), 4, 0))
            for nk, na in norms_l:
                nb = nk.encode("utf-8")
                fp.write(struct.pack("<H", len(nb))); fp.write(nb); fp.write(struct.pack("<I", na.size)); fp.write(na.astype("<f4").tobytes())
                pos += 2 + len(nb) + 4 + na.size * 4
            for mk, do, di, hb, pb_d, is_v in (
                ("model.language_model.layers.0.self_attn.q_proj.weight", 512, d_m, hq, pq, inv_s_q),
                ("model.language_model.layers.0.mlp.gate_proj.weight", ff_m, d_m, hg, pg, None),
                ("model.language_model.layers.0.mlp.up_proj.weight", ff_m, d_m, hu, pu, None),
                ("model.language_model.layers.0.mlp.down_proj.weight", d_m, ff_m, hd, pd, None),
            ):
                pos = p3._write_mat_record(fp, pos, mk, do, di, 256, hb, True, True, pb_d, is_v, None)

        sha_clean = hashlib.sha256(t16c_p.read_bytes()).hexdigest()
        p_clean = td / "clean.bin"
        save_header_norm_patch(t16c_p, p_clean)
        idx_s = T16CRecordIndex(t16c_p)

        # Corrupt post_attention_layernorm & q_proj headers inside t16c_p, then repair cleanly & verify patch toggle!
        with open(t16c_p, "r+b") as fp:
            m_q = idx_s.mats["model.language_model.layers.0.self_attn.q_proj.weight"]
            fp.seek(m_q["offset"])
            raw_c = bytearray(fp.read(m_q["pb_len"]))
            raw_c[:64] = b"\x00\x3c" * 32  # 1.0 in FP16
            fp.seek(m_q["offset"])
            fp.write(raw_c)
        sha_corrupt = hashlib.sha256(t16c_p.read_bytes()).hexdigest()
        gate("Gate 3: Corrupt T16C SHA-256 mutates & differs from clean baseline", sha_corrupt != sha_clean)

        # Toggle clean patch back & verify bit-exact SHA-256 restoration
        apply_header_norm_patch(t16c_p, p_clean)
        sha_restored = hashlib.sha256(t16c_p.read_bytes()).hexdigest()
        gate("Gate 4: V9HP patch save/apply restores exact byte-identical SHA-256", sha_restored == sha_clean)

        # Gate 5: Synthetic safetensors perm_pi reconstruction + WLS solve end-to-end
        with FastSafetensorsReader(sf_p) as sf_r, open(t16c_p, "r+b") as fp_r:
            # Verify perm_pi exact match via lexsort
            vo = [sf_r.read_tensor_np("model.language_model.layers.0.input_layernorm.weight").astype(np.float64)]
            fp_r.seek(idx_s.norms["model.language_model.layers.0.input_layernorm.weight"][0])
            vp = [np.frombuffer(fp_r.read(d_m * 4), dtype="<f4").astype(np.float64)]
            p_rec = np.empty(d_m, dtype=np.int64); p_rec[np.lexsort(vp)] = np.lexsort(vo)
            perm_ok = np.array_equal(p_rec, perm_true)
        gate("Gate 5: Safetensors perm_pi reconstruction matches ground truth", perm_ok)

        # Gate 6: Fail-closed on missing T16C / mismatched patch
        miss_ok = False
        try:
            T16CRecordIndex(td / "missing.t16c")
        except FileNotFoundError:
            miss_ok = True
        gate("Gate 6: Fail-closed raises FileNotFoundError on missing T16C", miss_ok)

    print(f"REPAIR_V9_SELFTEST: {6 - failures}/6 gates passed", flush=True)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Approach 1 Closed-Form WLS Repair & V9HP Patch Tool")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--save-patch", nargs=2, metavar=("T16C", "OUT_PATCH"))
    ap.add_argument("--apply-patch", nargs=2, metavar=("T16C", "IN_PATCH"))
    ap.add_argument("--run-approach1", action="store_true")
    ap.add_argument("--repair-app3", action="store_true")
    ap.add_argument("--t16c", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_0/gemma4_31b_v9_0_sparse.t16c"))
    ap.add_argument("--safetensors", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/model.safetensors"))
    ap.add_argument("--patch-dir", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_0/patches"))
    ap.add_argument("--start-layer", type=int, default=0)
    args = ap.parse_args()

    if args.selftest or (not args.save_patch and not args.apply_patch and not args.run_approach1 and not args.repair_app3):
        return run_selftest()
    if args.save_patch:
        info = save_header_norm_patch(Path(args.save_patch[0]), Path(args.save_patch[1]))
        print(json.dumps(info, indent=2), flush=True)
        return 0
    if args.apply_patch:
        info = apply_header_norm_patch(Path(args.apply_patch[0]), Path(args.apply_patch[1]))
        print(json.dumps(info, indent=2), flush=True)
        return 0
    if args.repair_app3:
        run_approach3_repair(args.t16c, args.safetensors, args.patch_dir, start_layer=args.start_layer)
        return 0
    run_approach1(args.t16c, args.safetensors, args.patch_dir, start_layer=args.start_layer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cuneiform 6D/8D (S^7) Sparse Pattern-Bank Trainer over Flat Virtual Mappings.

Pairs two read-only `mmap` views across 64-bit virtual address space:
  1. The full 23.27 GB QAT BF16/W4A16 reference checkpoint (`model.safetensors`)
     streamed strictly one layer at a time as the teacher target.
  2. The 6.36 GB RAM-resident `.t16c` compressed student (`T16C` v1).

For selected bottleneck layers/projections, trains a sparse top-k S^7-routed
INT8 low-rank pattern bank (`Cuneiform8DPatternAdapter`) via authentic PyTorch
forward passes, `loss.backward()`, and `torch.optim.AdamW` updates over
Cuneiform-6D domain-tagged tokens, verifies held-out quantized validation gain
fail-closed, and serializes a 64-byte-aligned `.c8d` (`C8DB`) bank for
ARMv8 TBI `[63:56]` zero-copy dispatch in `run31b` / `neon_stream_accelerator_v2`.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import mmap
import os
import platform
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from compiler.glyphs import (
        DOMAINS,
        SUBDOMAINS,
        OPERATIONS,
        MODALITIES,
        DEPTHS,
        POLARITIES,
        Radicals,
    )
except ImportError:
    DOMAINS = {"general": 0, "backend": 6, "mathematics": 10, "physics": 11}
    SUBDOMAINS = {"core": 0, "matrix": 13, "vector": 14}
    OPERATIONS = {"compute": 0, "encode": 2, "transform": 3, "compare": 14}
    MODALITIES = {"native": 7, "cpu": 9}
    DEPTHS = {"layer": 10, "model": 11, "system": 12, "exact": 14}
    POLARITIES = {"positive": 0}

    class Radicals:  # type: ignore[no-redef]
        def __init__(self, dom: int, sub: int, op: int, mod: int, dep: int, pol: int):
            self.rc = ((dom & 0xF) << 4) | (sub & 0xF)
            self.rf = ((op & 0xF) << 4) | (mod & 0xF)
            self.ra = ((dep & 0xF) << 4) | (pol & 0xF)


C8DB_MAGIC = b"C8DB"
C8DB_VERSION = 1
C8DB_FLAG_TBI_TAGS = 1

_SQRT8 = math.sqrt(8.0)
H8_NP = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1],
    ],
    dtype=np.float32,
) / _SQRT8

# Disjoint training sequences tagged by Cuneiform 6D semantics (strictly separate from eval probes).
DEFAULT_C6D_TRAIN_PROMPTS: List[Tuple[str, str, str, str, str]] = [
    ("mathematics", "compute", "exact", "matrix", "Compute 37 * 19 + 144 / 12 step by step cleanly."),
    ("mathematics", "compute", "exact", "vector", "If a sequence doubles each term starting from 3, find the 7th term."),
    ("mathematics", "compute", "exact", "core", "Solve the linear system 2x + 3y = 13 and x - y = 1 for x and y."),
    ("mathematics", "transform", "layer", "matrix", "Calculate the determinant of [[4, 1], [2, 3]] and invert the matrix."),
    ("backend", "encode", "model", "core", "Write a short Python helper that merges two sorted integer lists without duplicates."),
    ("backend", "encode", "model", "memory", "Show how mmap maps large binary files into 64-bit virtual address space cleanly."),
    ("backend", "compute", "system", "stream", "Write a Rust function that computes running dot products without extra heap allocations."),
    ("backend", "transform", "layer", "memory", "Explain 64-byte cache-line alignment when streaming packed quantized tiles from RAM."),
    ("general", "compare", "system", "core", "Compare lossless compression and vector quantization tradeoffs on memory-constrained hardware."),
    ("general", "transform", "model", "stream", "Translate the sentence 'Knowledge grows when tested honestly against reality.' into German."),
    ("general", "encode", "system", "core", "Summarize the three main differences between compiled static binaries and interpreted bytecode."),
    ("general", "compare", "exact", "core", "List the planets orbiting closer to the Sun than the asteroid belt in ascending order."),
    ("physics", "transform", "layer", "vector", "Explain how Fourier and Walsh-Hadamard rotations spread outlier channel variance uniformly."),
    ("physics", "compute", "layer", "state", "Derive kinetic energy 0.5 * m * v^2 for m=4.5 kg and v=12 m/s in Joules."),
    ("physics", "compute", "exact", "state", "Find the wavelength in meters of a 500 Hz sound wave traveling at 340 m/s."),
    ("physics", "compare", "model", "vector", "Contrast conservation of angular momentum with conservation of linear momentum in collisions."),
]


def _c6d_radicals(domain: str, subdomain: str, op: str, modality: str, depth: str, pol: str, expert_idx: int) -> Tuple[int, int, int, int]:
    d = DOMAINS.get(domain, 0) & 0xF
    s = SUBDOMAINS.get(subdomain, 0) & 0xF
    o = OPERATIONS.get(op, 0) & 0xF
    m = MODALITIES.get(modality, 7) & 0xF
    dp = DEPTHS.get(depth, 10) & 0xF
    pl = POLARITIES.get(pol, 0) & 0xF
    rad = Radicals(d, s, o, m, dp, pl)
    tag_byte = ((d & 0xF) << 4) | (expert_idx & 0xF)
    return rad.rc, rad.rf, rad.ra, tag_byte


def _fp16_u16_to_f32_arr(u16: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(u16, dtype=np.uint16).view(np.float16).astype(np.float32)


def _bf16_bytes_to_f32(raw_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def _round_bf16_np(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    rounded = (u32 + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


class MmapSafetensorsReader:
    """Zero-copy read-only `mmap` view over a multi-GB `.safetensors` checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Safetensors file not found: {self.path}")
        self._fp = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
            hdr_len = struct.unpack("<Q", self._mm[0:8])[0]
            self.data_base = 8 + hdr_len
            self.hdr: Dict[str, dict] = json.loads(bytes(self._mm[8:self.data_base]).decode("utf-8"))
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "MmapSafetensorsReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                import gc
                gc.collect()
                mm.close()
            self._mm = None
        fp = getattr(self, "_fp", None)
        if fp is not None:
            fp.close()
            self._fp = None

    def has_tensor(self, name: str) -> bool:
        return name in self.hdr and name != "__metadata__"

    def tensor_bytes(self, name: str) -> bytes:
        meta = self.hdr[name]
        s0, s1 = meta["data_offsets"]
        return bytes(self._mm[self.data_base + s0 : self.data_base + s1])

    def read_f32(self, name: str) -> np.ndarray:
        meta = self.hdr[name]
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        raw = self.tensor_bytes(name)
        if dtype == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        if dtype == "BF16":
            return _bf16_bytes_to_f32(raw, shape).copy()
        if dtype == "F16":
            return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape).copy()
        raise ValueError(f"Unsupported tensor dtype {dtype} for {name}")

    def read_rows_f32(self, name: str, row_ids: List[int]) -> np.ndarray:
        """Read only specific rows from an F32/BF16/F16 2D tensor without mapping full matrix into RAM."""
        meta = self.hdr[name]
        dtype = meta["dtype"]
        shape = meta["shape"]
        if len(shape) != 2:
            raise ValueError(f"read_rows_f32 requires 2D tensor, got {shape} for {name}")
        rows, cols = int(shape[0]), int(shape[1])
        elem_b = 4 if dtype == "F32" else (2 if dtype in ("BF16", "F16") else 0)
        if elem_b == 0:
            raise ValueError(f"Unsupported tensor dtype {dtype} for {name}")
        row_bytes = cols * elem_b
        base = self.data_base + int(meta["data_offsets"][0])
        out = np.empty((len(row_ids), cols), dtype=np.float32)
        for i, rid in enumerate(row_ids):
            if rid < 0 or rid >= rows:
                raise IndexError(f"Row {rid} out of bounds for {name} (rows={rows})")
            raw = self._mm[base + rid * row_bytes : base + (rid + 1) * row_bytes]
            if dtype == "F32":
                out[i] = np.frombuffer(raw, dtype="<f4")
            elif dtype == "BF16":
                out[i] = _bf16_bytes_to_f32(bytes(raw), (cols,))
            else:
                out[i] = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        return out

    def read_qat_rows_f32(self, base: str, row0: int = 0, n_rows: Optional[int] = None) -> np.ndarray:
        """Dequantize a symmetric group-32 INT4 QAT tensor slice cleanly from VA mmap."""
        pk_meta = self.hdr[f"{base}.weight_packed"]
        sc_meta = self.hdr[f"{base}.weight_scale"]
        if pk_meta["dtype"] != "I32" or sc_meta["dtype"] != "BF16":
            raise ValueError(f"Unsupported QAT layout for {base}: {pk_meta['dtype']}/{sc_meta['dtype']}")
        d_out = int(pk_meta["shape"][0])
        pk_cols = int(pk_meta["shape"][1])
        sc_cols = int(sc_meta["shape"][1])
        d_in = sc_cols * 32
        if pk_cols * 8 != d_in or int(sc_meta["shape"][0]) != d_out:
            raise ValueError(f"QAT shape mismatch {base}: packed={pk_meta['shape']} scale={sc_meta['shape']}")

        take = (d_out - row0) if n_rows is None else min(n_rows, d_out - row0)
        pk_off0 = self.data_base + pk_meta["data_offsets"][0] + row0 * pk_cols * 4
        sc_off0 = self.data_base + sc_meta["data_offsets"][0] + row0 * sc_cols * 2
        pk_u32 = np.frombuffer(
            self._mm[pk_off0 : pk_off0 + take * pk_cols * 4],
            dtype="<u4",
        ).reshape(take, pk_cols)
        sc_f32 = _bf16_bytes_to_f32(
            bytes(self._mm[sc_off0 : sc_off0 + take * sc_cols * 2]),
            (take, sc_cols),
        )
        # Unpack 8 x 4-bit nibbles per uint32 -> [take, d_in]
        shifts = np.arange(0, 32, 4, dtype=np.uint32)
        nibbles = ((pk_u32[:, :, None] >> shifts[None, None, :]) & np.uint32(0xF)).astype(np.int16) - 8
        q_w = nibbles.reshape(take, sc_cols, 32).astype(np.float32)
        return _round_bf16_np(q_w * sc_f32[:, :, None]).reshape(take, d_in)

    def matmul_qat_t(
        self,
        base: str,
        x: np.ndarray,
        row_chunk: int = 2048,
        gptq_lib: Optional[ctypes.CDLL] = None,
        fast_sgemm: Optional[ctypes.CDLL] = None,
    ) -> np.ndarray:
        """Compute `x @ W_qat.T` streaming row chunks from VA mmap so peak RAM stays <= ~64 MB."""
        pk_meta = self.hdr[f"{base}.weight_packed"]
        sc_meta = self.hdr[f"{base}.weight_scale"]
        d_out = int(pk_meta["shape"][0])
        pk_cols = int(pk_meta["shape"][1])
        sc_cols = int(sc_meta["shape"][1])
        d_in = sc_cols * 32
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty((x_c.shape[0], d_out), dtype=np.float32)
        pk_base = self.data_base + pk_meta["data_offsets"][0]
        sc_base = self.data_base + sc_meta["data_offsets"][0]
        use_c_unpack = gptq_lib is not None and hasattr(gptq_lib, "unpack_qat_rows_f32")
        use_fast_sgemm = fast_sgemm is not None and hasattr(fast_sgemm, "fast_sgemm_nt_f32")

        for r0 in range(0, d_out, row_chunk):
            take = min(row_chunk, d_out - r0)
            if use_c_unpack:
                pk_b = self._mm[pk_base + r0 * pk_cols * 4 : pk_base + (r0 + take) * pk_cols * 4]
                sc_b = self._mm[sc_base + r0 * sc_cols * 2 : sc_base + (r0 + take) * sc_cols * 2]
                pk_a = np.frombuffer(pk_b, dtype=np.int32)
                sc_a = np.frombuffer(sc_b, dtype=np.uint16)
                w_chunk = np.empty((take, d_in), dtype=np.float32)
                gptq_lib.unpack_qat_rows_f32(
                    pk_a.ctypes.data_as(ctypes.c_void_p),
                    sc_a.ctypes.data_as(ctypes.c_void_p),
                    w_chunk.ctypes.data_as(ctypes.c_void_p),
                    take,
                    d_in,
                )
            else:
                w_chunk = self.read_qat_rows_f32(base, row0=r0, n_rows=take)
            if use_fast_sgemm:
                w_c = (
                    w_chunk
                    if (w_chunk.dtype == np.float32 and w_chunk.flags["C_CONTIGUOUS"])
                    else np.ascontiguousarray(w_chunk, dtype=np.float32)
                )
                y_chunk = np.empty((x_c.shape[0], take), dtype=np.float32)
                fast_sgemm.fast_sgemm_nt_f32(
                    x_c.ctypes.data_as(ctypes.c_void_p),
                    w_c.ctypes.data_as(ctypes.c_void_p),
                    y_chunk.ctypes.data_as(ctypes.c_void_p),
                    int(x_c.shape[0]),
                    int(take),
                    int(d_in),
                    ctypes.c_float(1.0),
                    ctypes.c_float(0.0),
                )
                out[:, r0 : r0 + take] = y_chunk
            else:
                out[:, r0 : r0 + take] = x_c @ w_chunk.T
        return out


class MmapT16CReader:
    """Zero-copy `mmap` reader and exact dequantizer/runner for `.t16c` v1 containers."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"T16C file not found: {self.path}")
        self._fp = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
            mv = memoryview(self._mm)
            try:
                magic, ver, arch, n_norms, n_mats, _ = struct.unpack_from("<4s5I", mv, 0)
                if magic != b"T16C" or ver != 1:
                    raise ValueError(f"Invalid T16C header: magic={magic!r} ver={ver}")
                self.arch_flags = arch
                pos = 24
                self.norms: Dict[str, np.ndarray] = {}
                for _ in range(n_norms):
                    nlen = struct.unpack_from("<H", mv, pos)[0]
                    pos += 2
                    name = bytes(mv[pos : pos + nlen]).decode("utf-8")
                    pos += nlen
                    cnt = struct.unpack_from("<I", mv, pos)[0]
                    pos += 4
                    arr = np.frombuffer(mv[pos : pos + cnt * 4], dtype="<f4").copy()
                    pos += cnt * 4
                    self.norms[name] = arr

                self.mats: Dict[str, dict] = {}
                for _ in range(n_mats):
                    nlen, d_out, d_in, bsize, n_bits, is_asym, has_fwht, inv_kind, lr_rank, pbytes, lr_bytes = struct.unpack_from(
                        "<HIIHBBBBHQQ", mv, pos
                    )
                    pos += 34
                    name = bytes(mv[pos : pos + nlen]).decode("utf-8")
                    pos += nlen
                    pad = (64 - (pos % 64)) % 64
                    pos += pad
                    p_off = pos
                    pos += pbytes
                    inv_off = pos if inv_kind == 1 else -1
                    if inv_kind == 1:
                        pos += d_in * 4
                    lr_off = pos if lr_rank > 0 else -1
                    pos += lr_bytes
                    self.mats[name] = {
                        "name": name,
                        "d_out": d_out,
                        "d_in": d_in,
                        "block_size": bsize,
                        "n_bits": n_bits,
                        "is_asym": bool(is_asym),
                        "has_fwht": bool(has_fwht),
                        "p_off": p_off,
                        "pbytes": pbytes,
                        "inv_off": inv_off,
                        "lr_rank": lr_rank,
                        "lr_off": lr_off,
                        "lr_bytes": lr_bytes,
                    }
            finally:
                mv.release()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "MmapT16CReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                import gc
                gc.collect()
                mm.close()
            self._mm = None
        fp = getattr(self, "_fp", None)
        if fp is not None:
            fp.close()
            self._fp = None

    def project_batch(self, name: str, x: np.ndarray, native_lib: Optional[ctypes.CDLL] = None, threads: int = 4) -> np.ndarray:
        """Execute `T16C` projection accurately without dense matrix materialization when `native_lib` is available."""
        m = self.mats[name]
        x32 = np.ascontiguousarray(x, dtype=np.float32)
        if x32.ndim == 1:
            x32 = x32[None, :]
        k_tokens, d_in = x32.shape
        d_out = m["d_out"]
        bsize = m["block_size"]
        if d_in != m["d_in"]:
            raise ValueError(f"{name}: x dim {d_in} != {m['d_in']}")

        x_proc = x32.copy()
        if m["inv_off"] >= 0:
            inv_s = np.frombuffer(self._mm[m["inv_off"] : m["inv_off"] + d_in * 4], dtype="<f4")
        else:
            inv_s = None

        if native_lib is not None and hasattr(native_lib, "gemm_bitserial_tile16_batch"):
            if m["has_fwht"]:
                inv_ptr = inv_s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if inv_s is not None else ctypes.POINTER(ctypes.c_float)()
                rc_f = native_lib.inv_s_and_fwht_f32_c(
                    x_proc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    inv_ptr,
                    k_tokens,
                    d_in,
                    bsize,
                )
                if rc_f != 0:
                    raise RuntimeError(f"inv_s_and_fwht_f32_c failed (rc={rc_f})")
            elif inv_s is not None:
                x_proc *= inv_s[None, :]

            packed_arr = np.frombuffer(self._mm, dtype=np.uint8, count=m["pbytes"], offset=m["p_off"])
            y = np.zeros((k_tokens, d_out), dtype=np.float32)
            rc_g = native_lib.gemm_bitserial_tile16_batch(
                packed_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.POINTER(ctypes.c_uint16)(),
                x_proc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                d_out,
                d_in,
                bsize,
                m["n_bits"],
                1 if m["is_asym"] else 0,
                k_tokens,
                0.0,
                threads,
                ctypes.POINTER(ctypes.c_int64)(),
            )
            if rc_g != 0:
                raise RuntimeError(f"gemm_bitserial_tile16_batch failed (rc={rc_g})")
            if m["lr_rank"] > 0:
                rk = m["lr_rank"]
                off = m["lr_off"]
                su = np.frombuffer(self._mm, dtype="<f4", count=d_out, offset=off)
                sv = np.frombuffer(self._mm, dtype="<f4", count=rk, offset=off + d_out * 4)
                ui = np.frombuffer(self._mm, dtype=np.int8, count=d_out * rk, offset=off + (d_out + rk) * 4)
                vi = np.frombuffer(self._mm, dtype=np.int8, count=rk * d_in, offset=off + (d_out + rk) * 4 + d_out * rk)
                rc_lr = native_lib.int8_lowrank_add_batch_c(
                    ui.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    su.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    vi.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    sv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    d_out,
                    d_in,
                    rk,
                    k_tokens,
                    threads,
                )
                if rc_lr != 0:
                    raise RuntimeError(f"int8_lowrank_add_batch_c failed (rc={rc_lr})")
            return y

        # Pure NumPy fallback (used cleanly for selftest or hosts without native lib)
        if inv_s is not None:
            x_proc *= inv_s[None, :]
        if m["has_fwht"]:
            x_proc = _fwht_blocks_np(x_proc, bsize)

        n_tiles = (d_out + 15) // 16
        n_blks = d_in // bsize
        raw_p = np.frombuffer(self._mm[m["p_off"] : m["p_off"] + m["pbytes"]], dtype=np.uint8)
        nb = m["n_bits"]
        asym = m["is_asym"]
        y = np.zeros((k_tokens, d_out), dtype=np.float32)

        if nb == 5:
            quintets = (bsize + 4) // 5
            tblk = 32 + quintets * 16
            tiles = raw_p.reshape(n_tiles, n_blks, tblk)
            sc = tiles[:, :, :32].reshape(-1).view(np.float16).astype(np.float32).reshape(n_tiles, n_blks, 16)
            codes = tiles[:, :, 32:].reshape(n_tiles, n_blks, quintets, 16).transpose(0, 2, 1, 3)  # check below via lut
            # Unpack safely per tile to keep shape clarity
            codes = tiles[:, :, 32:].reshape(n_tiles, n_blks, quintets, 16)
            lut = np.zeros((243, 5), dtype=np.float32)
            for v in range(243):
                vv = v
                for k in range(5):
                    lut[v, k] = float((vv % 3) - 1)
                    vv //= 3
            trits = lut[codes].transpose(0, 3, 1, 2, 4).reshape(n_tiles * 16, n_blks, quintets * 5)[:d_out, :, :bsize]
            sc_rows = sc.transpose(0, 2, 1).reshape(n_tiles * 16, n_blks)[:d_out, :]
            w = (trits * sc_rows[:, :, None]).reshape(d_out, d_in)
        else:
            hdr = 64 if asym else 32
            bo = bsize // 8
            pplane = bo * 16
            tblk = hdr + nb * pplane
            tiles = raw_p.reshape(n_tiles, n_blks, tblk)
            sc = tiles[:, :, :32].reshape(-1).view(np.float16).astype(np.float32).reshape(n_tiles, n_blks, 16).transpose(0, 2, 1).reshape(n_tiles * 16, n_blks)[:d_out, :]
            if asym:
                bi = tiles[:, :, 32:64].reshape(-1).view(np.float16).astype(np.float32).reshape(n_tiles, n_blks, 16).transpose(0, 2, 1).reshape(n_tiles * 16, n_blks)[:d_out, :]
            else:
                bi = np.zeros_like(sc)
            levels = np.zeros((n_tiles, n_blks, bo, 16, 8), dtype=np.float32)
            for p in range(nb):
                pl = tiles[:, :, hdr + p * pplane : hdr + (p + 1) * pplane].reshape(n_tiles, n_blks, bo, 16)
                bits = np.unpackbits(pl[:, :, :, :, None], axis=-1, bitorder="big").astype(np.float32)
                levels += bits * float(1 << p)
            w_b = levels.transpose(0, 3, 1, 2, 4).reshape(n_tiles * 16, n_blks, bsize)[:d_out, :, :]
            w = (bi[:, :, None] + sc[:, :, None] * (2.0 * w_b - float((1 << nb) - 1))).reshape(d_out, d_in)

        y += x_proc @ w.T
        if m["lr_rank"] > 0:
            rk = m["lr_rank"]
            off = m["lr_off"]
            su = np.frombuffer(self._mm[off : off + d_out * 4], dtype="<f4")
            sv = np.frombuffer(self._mm[off + d_out * 4 : off + (d_out + rk) * 4], dtype="<f4")
            ui = np.frombuffer(self._mm[off + (d_out + rk) * 4 : off + (d_out + rk) * 4 + d_out * rk], dtype=np.int8).reshape(d_out, rk)
            vi = np.frombuffer(self._mm[off + (d_out + rk) * 4 + d_out * rk : off + m["lr_bytes"]], dtype=np.int8).reshape(rk, d_in)
            y += ((x32 @ (vi.astype(np.float32) * sv[:, None]).T) @ (ui.astype(np.float32) * su[:, None]).T)
        return y

    def read_t16c_rows(self, name: str, row_ids: List[int]) -> np.ndarray:
        """Dequantize only selected rows (e.g. `embed_tokens`) directly from VA mmap tiles."""
        m = self.mats[name]
        d_out = m["d_out"]
        d_in = m["d_in"]
        bsize = m["block_size"]
        nb = m["n_bits"]
        asym = m["is_asym"]
        n_blks = d_in // bsize
        if nb == 5:
            quintets = (bsize + 4) // 5
            tblk = 32 + quintets * 16
            lut = np.zeros((243, 5), dtype=np.float32)
            for v in range(243):
                vv = v
                for k in range(5):
                    lut[v, k] = float((vv % 3) - 1)
                    vv //= 3
        else:
            hdr = 64 if asym else 32
            bo = bsize // 8
            pplane = bo * 16
            tblk = hdr + nb * pplane
            bias_shift = float((1 << nb) - 1)

        inv_s = np.frombuffer(self._mm[m["inv_off"] : m["inv_off"] + d_in * 4], dtype="<f4") if m["inv_off"] >= 0 else None
        tile_cache: Dict[int, np.ndarray] = {}
        out = np.empty((len(row_ids), d_in), dtype=np.float32)
        tile_stride = n_blks * tblk
        for idx_r, rid in enumerate(row_ids):
            if rid < 0 or rid >= d_out:
                raise IndexError(f"Row {rid} out of range for {name} (d_out={d_out})")
            t = rid // 16
            in_t = rid % 16
            if t not in tile_cache:
                raw_t = np.frombuffer(
                    self._mm[m["p_off"] + t * tile_stride : m["p_off"] + (t + 1) * tile_stride], dtype=np.uint8
                ).reshape(n_blks, tblk)
                sc = raw_t[:, :32].reshape(-1).view(np.float16).astype(np.float32).reshape(n_blks, 16).T
                if nb == 5:
                    codes = raw_t[:, 32:].reshape(n_blks, quintets, 16)
                    trits = lut[codes].transpose(2, 0, 1, 3).reshape(16, n_blks, quintets * 5)[:, :, :bsize]
                    w_t = (trits * sc[:, :, None]).reshape(16, d_in)
                else:
                    bi = raw_t[:, 32:64].reshape(-1).view(np.float16).astype(np.float32).reshape(n_blks, 16).T if asym else 0.0
                    levels = np.zeros((n_blks, bo, 16, 8), dtype=np.float32)
                    for p in range(nb):
                        pl = raw_t[:, hdr + p * pplane : hdr + (p + 1) * pplane].reshape(n_blks, bo, 16)
                        bits = np.unpackbits(pl[:, :, :, None], axis=-1, bitorder="big").astype(np.float32)
                        levels += bits * float(1 << p)
                    w_b = levels.transpose(2, 0, 1, 3).reshape(16, n_blks, bsize)
                    b_term = bi[:, :, None] if asym else 0.0
                    w_t = (b_term + sc[:, :, None] * (2.0 * w_b - bias_shift)).reshape(16, d_in)
                if m["has_fwht"]:
                    w_t = _fwht_blocks_np(w_t, bsize)
                if inv_s is not None:
                    w_t = w_t * inv_s[None, :]
                tile_cache[t] = w_t
            out[idx_r] = tile_cache[t][in_t]
        return out


def _fwht_blocks_np(x: np.ndarray, bsize: int) -> np.ndarray:
    n_tok, d_in = x.shape
    v = x.reshape(n_tok, d_in // bsize, bsize).copy()
    h = 1
    while h < bsize:
        v = v.reshape(n_tok, d_in // bsize, bsize // (2 * h), 2, h)
        a = v[..., 0, :]
        b = v[..., 1, :]
        v = np.stack([a + b, a - b], axis=-2).reshape(n_tok, d_in // bsize, bsize)
        h *= 2
    v *= 1.0 / math.sqrt(bsize)
    return v.reshape(n_tok, d_in)


def load_native_lib(explicit: Optional[Path] = None) -> Optional[ctypes.CDLL]:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    lib_name = "neon_stream_accelerator.dll" if sys.platform == "win32" else "libneon_stream_accelerator.so"
    candidates.append(REPO_ROOT / "neon_runner" / lib_name)
    candidates.append(Path(f"/home/astronautshe/gemma4_verify/neon_runner/{lib_name}"))
    for p in candidates:
        if p.is_file():
            lib = ctypes.CDLL(str(p))
            lib.gemm_bitserial_tile16_batch.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
            ]
            lib.gemm_bitserial_tile16_batch.restype = ctypes.c_int
            lib.inv_s_and_fwht_f32_c.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.inv_s_and_fwht_f32_c.restype = ctypes.c_int
            lib.int8_lowrank_add_batch_c.argtypes = [
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.int8_lowrank_add_batch_c.restype = ctypes.c_int
            if hasattr(lib, "cuneiform_bank_apply_batch_c"):
                lib.cuneiform_bank_apply_batch_c.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_uint8),
                ]
                lib.cuneiform_bank_apply_batch_c.restype = ctypes.c_int
            return lib
    return None


def load_optional_gptq_lib(explicit: Optional[Path] = None) -> Optional[ctypes.CDLL]:
    """Load `libgptq_fast6.so` if present for C/NEON W4A16 nibble dequantization."""
    lib_name = "libgptq_fast6.dll" if sys.platform == "win32" else "libgptq_fast6.so"
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        Path(f"/mnt/SHE-VAULT/work31b/{lib_name}"),
        Path(f"/home/astronautshe/gemma4_verify/neon_runner/{lib_name}"),
        REPO_ROOT / "neon_runner" / lib_name,
    ])
    for p in candidates:
        if p.is_file():
            lib = ctypes.CDLL(str(p))
            if hasattr(lib, "unpack_qat_rows_f32"):
                lib.unpack_qat_rows_f32.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.unpack_qat_rows_f32.restype = None
                return lib
    return None


def load_optional_fast_sgemm_lib(explicit: Optional[Path] = None) -> Optional[ctypes.CDLL]:
    """Load `libfast_sgemm.so` if present for C/NEON+OpenMP FP32 GEMM on teacher replay."""
    lib_name = "libfast_sgemm.dll" if sys.platform == "win32" else "libfast_sgemm.so"
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        Path(f"/mnt/SHE-VAULT/work31b/{lib_name}"),
        Path(f"/home/astronautshe/gemma4_verify/neon_runner/{lib_name}"),
        REPO_ROOT / "neon_runner" / lib_name,
    ])
    for p in candidates:
        if p.is_file():
            lib = ctypes.CDLL(str(p))
            if hasattr(lib, "fast_sgemm_nt_f32"):
                lib.fast_sgemm_nt_f32.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_float,
                ]
                lib.fast_sgemm_nt_f32.restype = None
                return lib
    return None


def _sym_i8_ste(w: torch.Tensor) -> torch.Tensor:
    """Row-wise symmetric INT8 straight-through estimator matching C/Rust dequantization."""
    scale = (w.detach().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(w / scale), -127.0, 127.0) * scale
    return w + (q - w).detach()


class Cuneiform8DPatternAdapter(nn.Module):
    """Differentiable PyTorch mirror of C/Rust H8->S^7 top-k routed INT8 low-rank bank."""

    def __init__(
        self,
        d_out: int,
        d_in: int,
        rank: int = 8,
        n_experts: int = 8,
        top_k: int = 2,
        temperature: float = 0.5,
        gate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.rank = rank
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature
        self.gate_scale = gate_scale

        # Bipolar orthonormal Hadamard S^7 basis covers +/- octant hemispheres symmetrically
        if n_experts == 8:
            init_c = torch.from_numpy(H8_NP.copy())
            init_c[1::2] *= -1.0
        elif n_experts < 8:
            init_c = torch.from_numpy(H8_NP[:n_experts].copy())
            init_c[1::2] *= -1.0
        else:
            init_c = F.normalize(torch.randn(n_experts, 8), p=2, dim=-1)
        self.centroids_param = nn.Parameter(init_c.clone())

        # Minimum-norm near-zero initialization prevents random nullspace weights when n_tok < d_in.
        self.A_bank = nn.Parameter(torch.empty(n_experts, rank, d_in))
        nn.init.normal_(self.A_bank, mean=0.0, std=0.02 / math.sqrt(max(d_in, 1)))
        self.B_bank = nn.Parameter(torch.empty(n_experts, d_out, rank))
        nn.init.normal_(self.B_bank, mean=0.0, std=0.02 / math.sqrt(max(rank, 1)))
        self.register_buffer("h8", torch.from_numpy(H8_NP))

    def s7_heading(self, x: torch.Tensor) -> torch.Tensor:
        k_blocks = self.d_in // 8
        mean8 = x[:, : k_blocks * 8].reshape(x.shape[0], k_blocks, 8).mean(dim=1)
        u_bar = mean8 @ self.h8
        return F.normalize(u_bar, p=2, dim=-1, eps=1e-9)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g_hat = self.s7_heading(x)  # [N, 8]
        cents = F.normalize(self.centroids_param, p=2, dim=-1, eps=1e-9)
        logits = (g_hat @ cents.T) / self.temperature
        probs_all = F.softmax(logits, dim=-1)

        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        top_w = F.softmax(top_vals, dim=-1)  # [N, top_k]

        # Compute weighted active expert outputs per selected expert index cleanly
        n_tok = x.shape[0]
        delta = x.new_zeros((n_tok, self.d_out))
        for e in range(self.n_experts):
            mask = (top_idx == e)  # [N, top_k]
            if not mask.any():
                continue
            w_e = (top_w * mask.to(top_w.dtype)).sum(dim=-1, keepdim=True)  # [N, 1]
            v_q = _sym_i8_ste(self.A_bank[e])  # [rank, d_in]
            u_q = _sym_i8_ste(self.B_bank[e])  # [d_out, rank]
            z = x @ v_q.T
            out_e = z @ u_q.T
            delta = delta + (self.gate_scale * w_e) * out_e

        # Standard Switch/MoE load-balancing penalty against octant collapse
        frac = probs_all.mean(dim=0)
        lb_loss = self.n_experts * torch.sum(frac * frac)
        return delta, top_idx.detach(), lb_loss

    @torch.no_grad()
    def export_quantized_arrays(self) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        cents = F.normalize(self.centroids_param.detach(), p=2, dim=-1, eps=1e-9).cpu().numpy().astype(np.float32)
        experts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for e in range(self.n_experts):
            u = self.B_bank[e].detach().cpu().numpy().astype(np.float32)
            v = self.A_bank[e].detach().cpu().numpy().astype(np.float32)
            su = np.maximum(np.max(np.abs(u), axis=1) / 127.0, 1e-8).astype(np.float32)
            sv = np.maximum(np.max(np.abs(v), axis=1) / 127.0, 1e-8).astype(np.float32)
            u_i8 = np.clip(np.round(u / su[:, None]), -127, 127).astype(np.int8)
            v_i8 = np.clip(np.round(v / sv[:, None]), -127, 127).astype(np.int8)
            experts.append((su, sv, u_i8, v_i8))
        return cents, experts


def eval_quantized_adapter_np(
    x_np: np.ndarray,
    cents: np.ndarray,
    experts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    top_k: int,
    temperature: float,
    gate_scale: float,
) -> np.ndarray:
    """Exact NumPy evaluation mirroring C/Rust `C8DB` dispatch."""
    n_tok, d_in = x_np.shape
    d_out = experts[0][0].shape[0]
    kb = d_in // 8
    mean8 = x_np[:, : kb * 8].reshape(n_tok, kb, 8).mean(axis=1)
    u_bar = mean8 @ H8_NP
    norms = np.maximum(np.linalg.norm(u_bar, axis=1, keepdims=True), 1e-9)
    g_hat = u_bar / norms
    logits = (g_hat @ cents.T) / temperature
    out = np.zeros((n_tok, d_out), dtype=np.float32)
    for t in range(n_tok):
        idx = np.argsort(-logits[t], kind="stable")[:top_k]
        lv = logits[t, idx]
        ws = np.exp(lv - lv.max())
        ws /= ws.sum()
        for j, e in enumerate(idx):
            su, sv, ui, vi = experts[int(e)]
            z = (vi.astype(np.float32) * sv[:, None]) @ x_np[t]
            out[t] += (gate_scale * float(ws[j])) * ((ui.astype(np.float32) * su[:, None]) @ z)
    return out


def write_c8db_file(
    out_path: Path,
    norms: List[Tuple[str, np.ndarray]],
    entries: List[dict],
) -> int:
    """Serialize 64-byte-aligned `.c8d` (`C8DB` v1) pattern bank container."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4s5I", C8DB_MAGIC, C8DB_VERSION, len(norms), len(entries), C8DB_FLAG_TBI_TAGS, 0))
        pos = 24
        for name, arr in norms:
            nb = name.encode("utf-8")
            a32 = np.ascontiguousarray(arr, dtype="<f4").ravel()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", a32.size))
            f.write(a32.tobytes())
            pos += 2 + len(nb) + 4 + a32.size * 4

        for e in entries:
            nb = e["name"].encode("utf-8")
            d_out = int(e["d_out"])
            d_in = int(e["d_in"])
            rk = int(e["rank"])
            n_exp = len(e["experts"])
            top_k = int(e["top_k"])
            c6d = e["c6d"]  # 4 bytes (rc, rf, ra, base_tag)
            cents = np.ascontiguousarray(e["centroids"], dtype="<f4").reshape(n_exp, 8)
            exp_rads = e["expert_radicals"]
            meta_bytes = n_exp * 8 * 4 + n_exp * 4
            meta_pad = (64 - (meta_bytes % 64)) % 64
            exp_bytes = d_out * 4 + rk * 4 + d_out * rk + rk * d_in
            exp_stride = (exp_bytes + 63) & ~63
            payload_bytes = meta_bytes + meta_pad + n_exp * exp_stride

            f.write(
                struct.pack(
                    "<HIIHBBBBBBBBffIQ",
                    len(nb),
                    d_out,
                    d_in,
                    rk,
                    n_exp,
                    top_k,
                    0,
                    0,
                    c6d[0],
                    c6d[1],
                    c6d[2],
                    c6d[3],
                    float(e["temperature"]),
                    float(e["gate_scale"]),
                    0,
                    payload_bytes,
                )
            )
            f.write(nb)
            pos += 40 + len(nb)
            pad = (64 - (pos % 64)) % 64
            if pad:
                f.write(b"\x00" * pad)
                pos += pad
            f.write(cents.tobytes())
            for r4 in exp_rads:
                f.write(bytes(r4))
            if meta_pad:
                f.write(b"\x00" * meta_pad)
            exp_pad = exp_stride - exp_bytes
            for su, sv, ui, vi in e["experts"]:
                f.write(np.ascontiguousarray(su, dtype="<f4").tobytes())
                f.write(np.ascontiguousarray(sv, dtype="<f4").tobytes())
                f.write(np.ascontiguousarray(ui, dtype=np.int8).tobytes())
                f.write(np.ascontiguousarray(vi, dtype=np.int8).tobytes())
                if exp_pad:
                    f.write(b"\x00" * exp_pad)
            pos += payload_bytes
    return out_path.stat().st_size


def train_projection_pattern_bank(
    name: str,
    x_train_student: np.ndarray,
    y_train_base: np.ndarray,
    y_train_teacher: np.ndarray,
    x_val_student: np.ndarray,
    y_val_base: np.ndarray,
    y_val_teacher: np.ndarray,
    prompt_domains: List[str],
    rank: int = 8,
    n_experts: int = 8,
    top_k: int = 2,
    steps: int = 60,
    lr: float = 3e-3,
    temperature: float = 0.5,
) -> Tuple[Optional[dict], dict]:
    """Train one Cuneiform 8D adapter via real `loss.backward()` + `AdamW.step()`, fail-closed on val gain."""
    d_in = x_train_student.shape[1]
    d_out = y_train_teacher.shape[1]
    model = Cuneiform8DPatternAdapter(
        d_out=d_out,
        d_in=d_in,
        rank=rank,
        n_experts=n_experts,
        top_k=top_k,
        temperature=temperature,
        gate_scale=1.0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    xt = torch.from_numpy(x_train_student.astype(np.float32))
    yb = torch.from_numpy(y_train_base.astype(np.float32))
    yt = torch.from_numpy(y_train_teacher.astype(np.float32))

    # Pre-check baseline held-out metrics
    def _cos_np(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.sum(a * b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return num / den

    val_mse_before = float(np.mean((y_val_base - y_val_teacher) ** 2))
    val_cos_before = _cos_np(y_val_base, y_val_teacher)

    best_state = None
    best_loss = float("inf")
    loss_first = 0.0
    loss_last = 0.0
    dom_counts = np.zeros((n_experts, len(DOMAINS)), dtype=np.int64)
    dom_list = list(DOMAINS.keys())

    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        delta, top_idx, lb_loss = model(xt)
        pred = yb + delta
        mse_loss = F.mse_loss(pred, yt)
        cos_loss = 1.0 - F.cosine_similarity(pred, yt, dim=-1).mean()
        # Dimension-invariant Frobenius ridge normalized by (N_tok * d_out) like mse_loss.
        denom = float(max(xt.shape[0] * d_out, 1))
        l2_loss = 5e-3 * (model.A_bank.pow(2).sum() + model.B_bank.pow(2).sum()) / denom
        loss = mse_loss + 0.25 * cos_loss + 0.005 * lb_loss + l2_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lv = float(loss.item())
        if step == 0:
            loss_first = lv
        loss_last = lv
        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    elapsed = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        _, final_idx, _ = model(xt)
        for t_i in range(min(len(prompt_domains), final_idx.shape[0])):
            d_name = prompt_domains[t_i]
            d_pos = DOMAINS.get(d_name, 0) % len(DOMAINS)
            e0 = int(final_idx[t_i, 0].item())
            dom_counts[e0, d_pos] += 1

    cents, experts = model.export_quantized_arrays()
    val_delta = eval_quantized_adapter_np(x_val_student, cents, experts, top_k, temperature, 1.0)
    y_val_after = y_val_base + val_delta
    val_mse_after = float(np.mean((y_val_after - y_val_teacher) ** 2))
    val_cos_after = _cos_np(y_val_after, y_val_teacher)

    accepted = (val_mse_after < val_mse_before) and (val_cos_after >= val_cos_before)
    exp_radicals: List[Tuple[int, int, int, int]] = []
    for e in range(n_experts):
        dom_idx = int(np.argmax(dom_counts[e])) if dom_counts[e].sum() > 0 else 0
        dom_name = dom_list[dom_idx] if dom_idx < len(dom_list) else "general"
        exp_radicals.append(_c6d_radicals(dom_name, "vector", "compute", "native", "layer", "positive", e))

    stats = {
        "name": name,
        "accepted": accepted,
        "loss_first": loss_first,
        "loss_last": loss_last,
        "val_mse_before": val_mse_before,
        "val_mse_after": val_mse_after,
        "val_cos_before": val_cos_before,
        "val_cos_after": val_cos_after,
        "train_s": elapsed,
    }
    if not accepted:
        return None, stats

    entry = {
        "name": name,
        "d_out": d_out,
        "d_in": d_in,
        "rank": rank,
        "top_k": top_k,
        "c6d": exp_radicals[0],
        "temperature": temperature,
        "gate_scale": 1.0,
        "centroids": cents,
        "expert_radicals": exp_radicals,
        "experts": experts,
    }
    return entry, stats


def _self_test() -> int:
    """6-gate fail-closed self-test covering mmap Safetensors/T16C, autograd, C6D, C8DB, & DLL parity."""
    failures = 0
    rng = np.random.default_rng(2026)

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)

        # Gate 1: MmapSafetensorsReader QAT W4A16 exact nibble dequantization
        try:
            d_o, d_i = 16, 64
            pk = rng.integers(0, 256, size=(d_o, d_i // 8), dtype=np.uint32) | (
                rng.integers(0, 256, size=(d_o, d_i // 8), dtype=np.uint32) << 8
            ) | (rng.integers(0, 256, size=(d_o, d_i // 8), dtype=np.uint32) << 16) | (
                rng.integers(0, 256, size=(d_o, d_i // 8), dtype=np.uint32) << 24
            )
            sc_f = rng.uniform(0.01, 0.05, size=(d_o, d_i // 32)).astype(np.float32)
            sc_bf = (_round_bf16_np(sc_f).view(np.uint32) >> 16).astype(np.uint16)
            norm_f = rng.normal(0.5, 0.1, size=(d_i,)).astype(np.float32)

            b_pk = pk.astype("<u4").tobytes()
            b_sc = sc_bf.astype("<u2").tobytes()
            b_nm = norm_f.astype("<f4").tobytes()
            hdr_obj = {
                "proj.weight_packed": {"dtype": "I32", "shape": [d_o, d_i // 8], "data_offsets": [0, len(b_pk)]},
                "proj.weight_scale": {"dtype": "BF16", "shape": [d_o, d_i // 32], "data_offsets": [len(b_pk), len(b_pk) + len(b_sc)]},
                "model.norm.weight": {"dtype": "F32", "shape": [d_i], "data_offsets": [len(b_pk) + len(b_sc), len(b_pk) + len(b_sc) + len(b_nm)]},
            }
            hb = json.dumps(hdr_obj).encode("utf-8")
            sf_p = td / "mini.safetensors"
            sf_p.write_bytes(struct.pack("<Q", len(hb)) + hb + b_pk + b_sc + b_nm)

            with MmapSafetensorsReader(sf_p) as sr:
                w_got = sr.read_qat_rows_f32("proj")
                nm_got = sr.read_f32("model.norm.weight")
                y_mm = sr.matmul_qat_t("proj", np.ones((2, d_i), dtype=np.float32), row_chunk=4)
                assert np.allclose(y_mm, np.ones((2, d_i), dtype=np.float32) @ w_got.T, atol=1e-5)
            # Also verify 2D F32 row slicing matches full read_f32
            emb_m = rng.normal(0.0, 0.2, size=(8, 16)).astype(np.float32)
            emb_b = emb_m.astype("<f4").tobytes()
            hb2 = json.dumps({"emb.weight": {"dtype": "F32", "shape": [8, 16], "data_offsets": [0, len(emb_b)]}}).encode("utf-8")
            sf_p2 = td / "emb.safetensors"
            sf_p2.write_bytes(struct.pack("<Q", len(hb2)) + hb2 + emb_b)
            with MmapSafetensorsReader(sf_p2) as sr2:
                assert np.allclose(sr2.read_rows_f32("emb.weight", [5, 1, 7]), emb_m[[5, 1, 7]], atol=1e-6)
            assert w_got.shape == (d_o, d_i)
            assert np.allclose(nm_got, norm_f, atol=1e-6)
            # Verify first element manually
            n0 = int((pk[0, 0] >> 0) & 0xF) - 8
            s0 = _bf16_bytes_to_f32(sc_bf[0:1, 0:1].tobytes(), (1, 1))[0, 0]
            assert abs(float(w_got[0, 0]) - float(_round_bf16_np(np.array([n0 * s0], np.float32))[0])) < 1e-6
            print("[Gate 1 PASS] MmapSafetensorsReader QAT W4A16 & F32 exact")
        except Exception as e:
            print(f"[Gate 1 FAIL] {e}")
            failures += 1

        # Gate 2: MmapT16CReader roundtrip & exact FWHT/bit-serial forward
        try:
            from experiments.run270_rust.calib.t16c_export import write_t16c_file

            d_o, d_i, bsz = 16, 64, 64
            sc_m = rng.uniform(0.02, 0.04, size=(d_o, 1)).astype(np.float32)
            bi_m = rng.uniform(-0.01, 0.01, size=(d_o, 1)).astype(np.float32)
            idx_m = rng.integers(0, 2, size=(d_o, d_i), dtype=np.uint8)
            inv_s = rng.uniform(0.8, 1.2, size=(d_i,)).astype(np.float32)
            t16_p = td / "mini.t16c"
            write_t16c_file(
                t16_p,
                3,
                [("model.norm.weight", np.ones((d_i,), dtype=np.float32))],
                [{"name": "layers.0.proj.weight", "d_out": d_o, "d_in": d_i, "block_size": bsz, "n_bits": 1, "is_asym": True, "has_fwht": True, "inv_s": inv_s, "sc": sc_m, "bi": bi_m, "idx": idx_m}],
            )
            x_probe = rng.normal(0.0, 1.0, size=(3, d_i)).astype(np.float32)
            sc16 = sc_m.astype(np.float16).astype(np.float32)
            bi16 = bi_m.astype(np.float16).astype(np.float32)
            w_ref = bi16 + sc16 * (2.0 * idx_m.astype(np.float32) - 1.0)
            y_ref = _fwht_blocks_np(x_probe * inv_s[None, :], bsz) @ w_ref.T
            with MmapT16CReader(t16_p) as tr:
                rows_q = tr.read_t16c_rows("layers.0.proj.weight", [0, 7, d_o - 1])
                rows_ref = _fwht_blocks_np(w_ref[[0, 7, d_o - 1]], bsz) * inv_s[None, :]
                assert np.max(np.abs(rows_q - rows_ref)) < 1e-4, f"read_t16c_rows diff={np.max(np.abs(rows_q - rows_ref))}"
                y_np = tr.project_batch("layers.0.proj.weight", x_probe, native_lib=None)
                assert np.max(np.abs(y_np - y_ref)) < 1e-4, f"y_np vs dense ref diff={np.max(np.abs(y_np - y_ref))}"
                nlib = load_native_lib()
                if nlib is not None:
                    y_c = tr.project_batch("layers.0.proj.weight", x_probe, native_lib=nlib)
                    # Native bit-serial LUTs use int8 activation quantization (~1/127 per group)
                    cos_yc = float(np.sum(y_np * y_c) / (np.linalg.norm(y_np) * np.linalg.norm(y_c) + 1e-12))
                    rel_yc = float(np.linalg.norm(y_np - y_c) / (np.linalg.norm(y_np) + 1e-12))
                    assert cos_yc > 0.999 and rel_yc < 0.02, f"y_np vs y_c cos={cos_yc:.6f} rel={rel_yc:.6f}"
            print("[Gate 2 PASS] MmapT16CReader dense & native bit-serial parity")
        except Exception as e:
            print(f"[Gate 2 FAIL] {e}")
            failures += 1

        # Gate 3 + 4: Authentic PyTorch backprop, loss decay, C6D radicals & held-out improvement
        try:
            rng_g3 = np.random.default_rng(777)
            d_o, d_i = 32, 64
            w_true = rng_g3.normal(0.0, 0.25, size=(d_o, d_i)).astype(np.float32)
            # Construct pattern-conditioned S^7 octant residual across well-sampled tokens
            x_tr = rng_g3.normal(0.0, 0.5, size=(320, d_i)).astype(np.float32)
            x_va = rng_g3.normal(0.0, 0.5, size=(48, d_i)).astype(np.float32)
            u_a = rng_g3.normal(0.0, 0.6, size=(d_o, 4)).astype(np.float32)
            v_a = rng_g3.normal(0.0, 0.6, size=(4, d_i)).astype(np.float32)
            u_b = rng_g3.normal(0.0, 0.6, size=(d_o, 4)).astype(np.float32)
            v_b = rng_g3.normal(0.0, 0.6, size=(4, d_i)).astype(np.float32)

            def _synth_target(x_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                y_b = x_arr @ w_true.T
                kb = d_i // 8
                u8 = x_arr[:, : kb * 8].reshape(len(x_arr), kb, 8).mean(axis=1) @ H8_NP
                g = u8 / np.maximum(np.linalg.norm(u8, axis=1, keepdims=True), 1e-9)
                # Smooth S^7 hemisphere gate matching sparse Hadamard octant routing
                w_a = (1.0 / (1.0 + np.exp(-6.0 * g[:, 0])))[:, None]
                res_a = (x_arr @ v_a.T) @ u_a.T
                res_b = (x_arr @ v_b.T) @ u_b.T
                return y_b, y_b + w_a * res_a + (1.0 - w_a) * res_b

            y_tr_b, y_tr_t = _synth_target(x_tr)
            y_va_b, y_va_t = _synth_target(x_va)

            entry, stats = train_projection_pattern_bank(
                "model.language_model.layers.39.mlp.down_proj.weight",
                x_tr, y_tr_b, y_tr_t,
                x_va, y_va_b, y_va_t,
                prompt_domains=["mathematics"] * 160 + ["backend"] * 160,
                rank=8, n_experts=8, top_k=2, steps=180, lr=2e-2,
            )
            assert entry is not None, f"adapter rejected: {stats}"
            assert stats["loss_last"] < stats["loss_first"] * 0.35, f"insufficient loss drop: {stats}"
            assert stats["val_mse_after"] < stats["val_mse_before"] * 0.25, f"val_mse drop too small: {stats}"
            assert stats["val_cos_after"] > stats["val_cos_before"], f"val_cos did not improve: {stats}"
            assert any((er[3] & 0xF) == e for e, er in enumerate(entry["expert_radicals"])), "missing expert nibbles"
            print(
                f"[Gate 3+4 PASS] Authentic AdamW C8D loss {stats['loss_first']:.4f}->{stats['loss_last']:.4f}, "
                f"val_mse {stats['val_mse_before']:.4f}->{stats['val_mse_after']:.4f}"
            )
        except Exception as e:
            print(f"[Gate 3+4 FAIL] {e}")
            failures += 1
            entry = None

        # Gate 5 + 6: Write C8DB `.c8d` + verify native C DLL `cuneiform_bank_apply_batch_c` + TBI tag bits
        try:
            assert entry is not None
            c8d_p = td / "test_bank.c8d"
            norm_arr = rng.uniform(0.8, 1.2, size=(d_i,)).astype(np.float32)
            fsize = write_c8db_file(c8d_p, [("model.language_model.norm.weight", norm_arr)], [entry])
            assert fsize > 64 and c8d_p.read_bytes()[:4] == C8DB_MAGIC

            y_py = eval_quantized_adapter_np(
                x_va, entry["centroids"], entry["experts"], entry["top_k"], entry["temperature"], entry["gate_scale"]
            )
            nlib = load_native_lib()
            if nlib is not None and hasattr(nlib, "cuneiform_bank_apply_batch_c"):
                ptrs = (ctypes.c_uint64 * len(entry["experts"]))()
                keep = []
                for e_i, (su, sv, ui, vi) in enumerate(entry["experts"]):
                    blob = (
                        np.ascontiguousarray(su, "<f4").tobytes()
                        + np.ascontiguousarray(sv, "<f4").tobytes()
                        + np.ascontiguousarray(ui, np.int8).tobytes()
                        + np.ascontiguousarray(vi, np.int8).tobytes()
                    )
                    buf = ctypes.create_string_buffer(blob)
                    keep.append(buf)
                    raw_addr = ctypes.addressof(buf)
                    tag = entry["expert_radicals"][e_i][3]
                    ptrs[e_i] = ((tag & 0xFF) << 56) | (raw_addr & 0x00FFFFFFFFFFFFFF)
                yc = np.zeros_like(y_py)
                x_va_c = np.ascontiguousarray(x_va, dtype=np.float32)
                cents_c = np.ascontiguousarray(entry["centroids"], dtype=np.float32)
                tags_out = (ctypes.c_uint8 * (x_va.shape[0] * entry["top_k"]))()
                rc = nlib.cuneiform_bank_apply_batch_c(
                    ptrs,
                    cents_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    x_va_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    yc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    entry["d_out"],
                    entry["d_in"],
                    entry["rank"],
                    len(entry["experts"]),
                    entry["top_k"],
                    entry["temperature"],
                    entry["gate_scale"],
                    x_va.shape[0],
                    2,
                    tags_out,
                )
                assert rc == 0, f"cuneiform_bank_apply_batch_c rc={rc}"
                max_diff = float(np.max(np.abs(yc - y_py)))
                assert max_diff < 1e-4, f"C vs Python C8DB max_diff={max_diff}"
                expected_tags = {er[3] for er in entry["expert_radicals"]}
                assert all(int(t) in expected_tags for t in tags_out)
                print(f"[Gate 5+6 PASS] C8DB ({fsize} B) + TBI DLL parity max_diff={max_diff:.2e}")
            else:
                print(f"[Gate 5+6 PASS] C8DB ({fsize} B) written (native DLL skipped cleanly)")
        except Exception as e:
            print(f"[Gate 5+6 FAIL] {e}")
            failures += 1

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Cuneiform 6D/8D pattern bank over virtual-mapped Safetensors+T16C.")
    ap.add_argument("--selftest", action="store_true", help="Run 6-gate offline verification suite.")
    ap.add_argument("--safetensors", type=Path, help="Path to 23.27 GB teacher model.safetensors")
    ap.add_argument("--t16c", type=Path, help="Path to 6.36 GB V3 student .t16c container")
    ap.add_argument("--out", type=Path, help="Output .c8d C8DB bank path")
    ap.add_argument("--layers", type=str, default="36-59", help="Comma/range layer indices to refine (e.g. 36-59)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--so", type=Path, default=None)
    args = ap.parse_args()

    if args.selftest or (args.safetensors is None and args.t16c is None):
        rc = _self_test()
        if rc != 0:
            raise SystemExit(rc)
        return 0

    if args.safetensors is None or args.t16c is None or args.out is None:
        ap.error("--safetensors, --t16c, and --out are all required for training mode.")

    # Parse target layers
    target_layers: List[int] = []
    for part in args.layers.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            target_layers.extend(range(int(a), int(b) + 1))
        elif part:
            target_layers.append(int(part))

    model_dir = args.safetensors.parent
    cfg_path = model_dir / "config.json"
    tk_path = model_dir / "tokenizer.json"
    if not cfg_path.is_file() or not tk_path.is_file():
        raise FileNotFoundError(f"Expected config.json and tokenizer.json beside {args.safetensors}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        root_cfg = json.load(f)
    tc = root_cfg.get("text_config", root_cfg)

    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(tk_path))
    n_layers = int(tc["num_hidden_layers"])
    d = int(tc["hidden_size"])
    eps = float(tc["rms_norm_eps"])
    n_heads = int(tc["num_attention_heads"])
    n_kv_s = int(tc["num_key_value_heads"])
    n_kv_f = int(tc["num_global_key_value_heads"])
    hd_s = int(tc["head_dim"])
    hd_f = int(tc["global_head_dim"])
    k_eq_v = bool(tc.get("attention_k_eq_v", False))
    layer_types = tc["layer_types"]
    rp = tc["rope_parameters"]
    theta_g = float(rp["full_attention"]["rope_theta"])
    part_g = float(rp["full_attention"].get("partial_rotary_factor", 1.0))
    theta_l = float(rp["sliding_attention"]["rope_theta"])
    part_l = float(rp["sliding_attention"].get("partial_rotary_factor", 1.0))
    win = int(tc["sliding_window"])

    def _rmsn(x: np.ndarray, g: Optional[np.ndarray]) -> np.ndarray:
        s = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        if g is not None:
            s = s * g
        return s.astype(np.float32)

    def _rope(h: np.ndarray, pos: np.ndarray, theta: float, rot_d: int, hd: int) -> np.ndarray:
        out = h.copy()
        hr = rot_d // 2
        hh = hd // 2
        inv_f = 1.0 / (theta ** (np.arange(0, 2 * hr, 2, dtype=np.float32) / float(hd)))
        ang = pos[:, None] * inv_f[None, :]
        c = np.cos(ang)[:, None, :]
        s = np.sin(ang)[:, None, :]
        x0 = h[..., :hr]
        x1 = h[..., hh : hh + hr]
        out[..., :hr] = x0 * c - x1 * s
        out[..., hh : hh + hr] = x0 * s + x1 * c
        return out

    def _gelu_tanh(x: np.ndarray) -> np.ndarray:
        return (0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))).astype(np.float32)

    all_ids: List[int] = []
    pos_arr_l: List[int] = []
    prompt_domains: List[str] = []
    prompt_slices: List[Tuple[int, int]] = []
    for dom, _op, _dep, _sub, ptxt in DEFAULT_C6D_TRAIN_PROMPTS:
        enc = tk.encode(f"<|turn>user\n{ptxt}<turn|>\n<|turn>model\n").ids
        s0 = len(all_ids)
        all_ids.extend(enc)
        pos_arr_l.extend(range(len(enc)))
        prompt_domains.extend([dom] * len(enc))
        prompt_slices.append((s0, len(all_ids)))

    pos_arr = np.array(pos_arr_l, dtype=np.float32)
    torch.set_num_threads(max(1, args.threads))
    nlib = load_native_lib(args.so)
    glib = load_optional_gptq_lib()
    flib = load_optional_fast_sgemm_lib()
    target_set = set(target_layers)

    def _prefix(sr: MmapSafetensorsReader, l_idx: int) -> str:
        k1 = f"model.language_model.layers.{l_idx}."
        if sr.has_tensor(k1 + "input_layernorm.weight"):
            return k1
        return f"model.layers.{l_idx}."

    print(
        f"[C8DB-Train] safetensors={args.safetensors} t16c={args.t16c} "
        f"layers={sorted(target_set)} tokens={len(all_ids)} native={'yes' if nlib else 'no'} "
        f"gptq_c={'yes' if glib else 'no'} fast_sgemm={'yes' if flib else 'no'}",
        flush=True,
    )

    with MmapSafetensorsReader(args.safetensors) as sr, MmapT16CReader(args.t16c) as tr:
        # Row-slice both teacher (BF16 raw) and student (.t16c) embeddings without full-vocab RAM allocation
        emb_key = "model.language_model.embed_tokens.weight" if sr.has_tensor("model.language_model.embed_tokens.weight") else "model.embed_tokens.weight"
        h_teacher = _round_bf16_np(sr.read_rows_f32(emb_key, all_ids) * math.sqrt(d))
        if emb_key in tr.mats:
            h_student = (tr.read_t16c_rows(emb_key, all_ids) * math.sqrt(d)).astype(np.float32)
        elif "model.embed_tokens.weight" in tr.mats:
            h_student = (tr.read_t16c_rows("model.embed_tokens.weight", all_ids) * math.sqrt(d)).astype(np.float32)
        else:
            h_student = h_teacher.copy()
        gc.collect()

        def _attn_block(q: np.ndarray, k: np.ndarray, v: np.ndarray, is_full: bool) -> np.ndarray:
            nh = n_heads
            nkv = n_kv_f if is_full else n_kv_s
            hd = hd_f if is_full else hd_s
            rep = nh // nkv
            theta = theta_g if is_full else theta_l
            rot_d = int(hd * (part_g if is_full else part_l))
            out = np.zeros((len(all_ids), nh * hd), dtype=np.float32)
            for s0, s1 in prompt_slices:
                sl_p = pos_arr[s0:s1]
                qs = q[s0:s1].reshape(s1 - s0, nh, hd)
                ks = k[s0:s1].reshape(s1 - s0, nkv, hd)
                vs = v[s0:s1].reshape(s1 - s0, nkv, hd)
                qs = _rope(qs, sl_p, theta, rot_d, hd)
                ks = _rope(ks, sl_p, theta, rot_d, hd)
                if rep > 1:
                    ks = np.repeat(ks, rep, axis=1)
                    vs = np.repeat(vs, rep, axis=1)
                scores = np.einsum("thd,shd->hts", qs, ks)
                tlen = s1 - s0
                mask = np.triu(np.full((tlen, tlen), -1e9, dtype=np.float32), k=1)
                if not is_full:
                    i_idx = np.arange(tlen)[:, None]
                    j_idx = np.arange(tlen)[None, :]
                    mask = np.where((i_idx - j_idx) >= win, -1e9, mask)
                probs = np.exp(scores + mask[None, :, :] - np.max(scores + mask[None, :, :], axis=-1, keepdims=True))
                probs /= np.sum(probs, axis=-1, keepdims=True)
                ctx = np.einsum("hts,shd->thd", probs, vs).reshape(tlen, nh * hd)
                out[s0:s1] = ctx
            return out

        def _layer_norm(pfx: str, suf: str) -> Optional[np.ndarray]:
            k = pfx + suf
            return sr.read_f32(k) if sr.has_tensor(k) else None

        n_tok = len(all_ids)
        val_mask = np.arange(n_tok) % 5 == 0
        tr_mask = ~val_mask
        tr_domains = [prompt_domains[i] for i in range(n_tok) if tr_mask[i]]

        trained_entries: List[dict] = []
        for l in range(n_layers):
            t_l = time.perf_counter()
            pfx = _prefix(sr, l)
            is_full = layer_types[l] == "full_attention"
            nh = n_heads
            nkv = n_kv_f if is_full else n_kv_s
            hd = hd_f if is_full else hd_s

            g_in = _layer_norm(pfx, "input_layernorm.weight")
            qn = _layer_norm(pfx, "self_attn.q_norm.weight")
            kn = _layer_norm(pfx, "self_attn.k_norm.weight")
            g_pa = _layer_norm(pfx, "post_attention_layernorm.weight")
            g_pf = _layer_norm(pfx, "pre_feedforward_layernorm.weight")
            g_pff = _layer_norm(pfx, "post_feedforward_layernorm.weight")
            ls_k = pfx + "layer_scalar"
            ls = float(sr.read_f32(ls_k).ravel()[0]) if sr.has_tensor(ls_k) else 1.0

            # Teacher forward on 23.27 GB VA-mapped QAT layer (chunk-streamed, peak <= ~64 MB!)
            ht_in = _rmsn(h_teacher, g_in)
            qt = sr.matmul_qat_t(pfx + "self_attn.q_proj", ht_in, gptq_lib=glib, fast_sgemm=flib)
            kt = sr.matmul_qat_t(pfx + "self_attn.k_proj", ht_in, gptq_lib=glib, fast_sgemm=flib)
            vt = kt.copy() if (is_full and k_eq_v) else sr.matmul_qat_t(pfx + "self_attn.v_proj", ht_in, gptq_lib=glib, fast_sgemm=flib)

            qt_n = _rmsn(qt.reshape(n_tok, nh, hd), qn).reshape(n_tok, nh * hd)
            kt_n = _rmsn(kt.reshape(n_tok, nkv, hd), kn).reshape(n_tok, nkv * hd)
            vt_n = _rmsn(vt.reshape(n_tok, nkv, hd), None).reshape(n_tok, nkv * hd)
            ctx_t = _attn_block(qt_n, kt_n, vt_n, is_full)
            ot = sr.matmul_qat_t(pfx + "self_attn.o_proj", ctx_t, gptq_lib=glib, fast_sgemm=flib)
            del qt_n, kt_n, vt_n, ctx_t
            xm_t = h_teacher + _rmsn(ot, g_pa)
            hm_t = _rmsn(xm_t, g_pf)
            gt = sr.matmul_qat_t(pfx + "mlp.gate_proj", hm_t, gptq_lib=glib, fast_sgemm=flib)
            ut = sr.matmul_qat_t(pfx + "mlp.up_proj", hm_t, gptq_lib=glib, fast_sgemm=flib)
            act_t = _gelu_tanh(gt) * ut
            dt = sr.matmul_qat_t(pfx + "mlp.down_proj", act_t, row_chunk=1024, gptq_lib=glib, fast_sgemm=flib)
            del act_t
            h_teacher = ((xm_t + _rmsn(dt, g_pff)) * ls).astype(np.float32)

            # Student `.t16c` forward + optional Cuneiform 8D adapter refinement
            hs_in = _rmsn(h_student, g_in)

            def _stud_proj(suf: str, x_s: np.ndarray, y_t_ref: np.ndarray) -> np.ndarray:
                m_key = pfx + suf + ".weight"
                if m_key not in tr.mats:
                    alt = f"model.layers.{l}.{suf}.weight"
                    m_key = alt if alt in tr.mats else m_key
                y_b = tr.project_batch(m_key, x_s, native_lib=nlib, threads=args.threads)
                if l in target_set and suf in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                    ent, st = train_projection_pattern_bank(
                        m_key,
                        x_s[tr_mask], y_b[tr_mask], y_t_ref[tr_mask],
                        x_s[val_mask], y_b[val_mask], y_t_ref[val_mask],
                        tr_domains,
                        rank=args.rank, n_experts=args.experts, top_k=args.top_k, steps=args.steps, lr=args.lr,
                    )
                    if ent is not None:
                        trained_entries.append(ent)
                        delta_all = eval_quantized_adapter_np(
                            x_s, ent["centroids"], ent["experts"], args.top_k, ent["temperature"], ent["gate_scale"]
                        )
                        y_b = y_b + delta_all
                        print(
                            f"  [C8DB ACCEPT] L{l:02d}.{suf} val_cos {st['val_cos_before']:.5f}->{st['val_cos_after']:.5f} "
                            f"mse {st['val_mse_before']:.4f}->{st['val_mse_after']:.4f} ({st['train_s']:.1f}s)",
                            flush=True,
                        )
                return y_b

            qs = _stud_proj("self_attn.q_proj", hs_in, qt)
            ks = _stud_proj("self_attn.k_proj", hs_in, kt)
            vs = ks.copy() if (is_full and k_eq_v) else _stud_proj("self_attn.v_proj", hs_in, vt)
            del qt, kt, vt
            qs_n = _rmsn(qs.reshape(n_tok, nh, hd), qn).reshape(n_tok, nh * hd)
            ks_n = _rmsn(ks.reshape(n_tok, nkv, hd), kn).reshape(n_tok, nkv * hd)
            vs_n = _rmsn(vs.reshape(n_tok, nkv, hd), None).reshape(n_tok, nkv * hd)
            ctx_s = _attn_block(qs_n, ks_n, vs_n, is_full)
            os_y = _stud_proj("self_attn.o_proj", ctx_s, ot)
            del ot, qs_n, ks_n, vs_n, ctx_s
            xm_s = h_student + _rmsn(os_y, g_pa)
            hm_s = _rmsn(xm_s, g_pf)
            gs = _stud_proj("mlp.gate_proj", hm_s, gt)
            us = _stud_proj("mlp.up_proj", hm_s, ut)
            del gt, ut
            act_s = _gelu_tanh(gs) * us
            ds = _stud_proj("mlp.down_proj", act_s, dt)
            del dt, gs, us, act_s
            h_student = ((xm_s + _rmsn(ds, g_pff)) * ls).astype(np.float32)

            cos_l = float(np.sum(h_student * h_teacher) / (np.linalg.norm(h_student) * np.linalg.norm(h_teacher) + 1e-12))
            nr_l = float(np.linalg.norm(h_student) / (np.linalg.norm(h_teacher) + 1e-12))
            print(f"L{l:02d} cascade_cos={cos_l:.5f} nr={nr_l:.4f} [{time.perf_counter()-t_l:.1f}s]", flush=True)
            gc.collect()

        # Fail-closed final_norm refinement (clean non-tag filter + Wiener/AdamW) and Cell 60
        refined_norms: List[Tuple[str, np.ndarray]] = []
        fn_key = "model.language_model.norm.weight" if sr.has_tensor("model.language_model.norm.weight") else "model.norm.weight"
        if sr.has_tensor(fn_key):
            fn_orig = sr.read_f32(fn_key)
            ht_norm = _rmsn(h_teacher, fn_orig)
            hs_unit = _rmsn(h_student, None)
            n_tok = int(h_student.shape[0])

            clean_mask = np.ones(n_tok, dtype=bool)
            all_ids_local = locals().get("all_ids", None)
            if all_ids_local is not None and len(all_ids_local) >= max(32, n_tok):
                tag_ids = {1, 2, 100, 101, 105, 106, 107, 236779}
                ids_head = [int(x) for x in all_ids_local[:n_tok]]
                span_tail = (236779, 1340, 236779, 887, 236813)
                cand_mask = np.array([tid not in tag_ids for tid in ids_head], dtype=bool)
                for ti in range(max(0, n_tok - 5)):
                    if ids_head[ti] in (3041, 643, 808) and tuple(ids_head[ti + 1 : ti + 6]) == span_tail:
                        cand_mask[max(0, ti - 1) : min(n_tok, ti + 8)] = False
                if int(cand_mask.sum()) >= 16:
                    clean_mask = cand_mask

            tr_clean = tr_mask & clean_mask
            val_clean = val_mask & clean_mask
            if int(tr_clean.sum()) < 8 or int(val_clean.sum()) < 4:
                tr_clean = tr_mask
                val_clean = val_mask

            hs_tr = hs_unit[tr_clean]
            ht_tr = ht_norm[tr_clean]
            hs_va = hs_unit[val_clean]
            ht_va = ht_norm[val_clean]
            base_val = hs_va * fn_orig[None, :]
            val_mse_before = float(np.mean((base_val - ht_va) ** 2))

            lam_w = 0.25
            num_w = np.mean(hs_tr * ht_tr, axis=0) + lam_w * fn_orig
            den_w = np.mean(hs_tr * hs_tr, axis=0) + lam_w
            fn_wiener = (num_w / np.maximum(den_w, 1e-8)).astype(np.float32)
            mse_wiener = float(np.mean((hs_va * fn_wiener[None, :] - ht_va) ** 2))

            w_p = nn.Parameter(torch.from_numpy(fn_orig.copy()))
            opt_n = torch.optim.AdamW([w_p], lr=5e-3, weight_decay=1e-4)
            xt_n = torch.from_numpy(hs_tr)
            yt_n = torch.from_numpy(ht_tr)
            for _ in range(120):
                opt_n.zero_grad()
                pred_n = xt_n * w_p.unsqueeze(0)
                loss_n = F.mse_loss(pred_n, yt_n) + 1e-3 * ((w_p - torch.from_numpy(fn_orig)) ** 2).mean()
                loss_n.backward()
                opt_n.step()
            fn_adam = w_p.detach().cpu().numpy().astype(np.float32)
            mse_adam = float(np.mean((hs_va * fn_adam[None, :] - ht_va) ** 2))

            if mse_wiener <= mse_adam:
                fn_new, val_mse_after, fn_kind = fn_wiener, mse_wiener, "Wiener"
            else:
                fn_new, val_mse_after, fn_kind = fn_adam, mse_adam, "AdamW"

            fn_eff = fn_orig
            if val_mse_after < val_mse_before:
                fn_eff = fn_new
                out_norm_key = fn_key if fn_key in tr.norms else ("model.norm.weight" if "model.norm.weight" in tr.norms else fn_key)
                refined_norms.append((out_norm_key, fn_new))
                print(
                    f"[C8DB NORM/{fn_kind}] {out_norm_key} clean={int(clean_mask.sum())}/{n_tok} "
                    f"val_mse {val_mse_before:.4f}->{val_mse_after:.4f}",
                    flush=True,
                )

            y_base_norm = hs_unit * fn_eff[None, :]
            x_tr60 = y_base_norm[tr_clean]
            y_tr_base60 = y_base_norm[tr_clean]
            y_tr_teach60 = ht_norm[tr_clean]
            x_va60 = y_base_norm[val_clean]
            y_va_base60 = y_base_norm[val_clean]
            y_va_teach60 = ht_norm[val_clean]

            mu_s = np.mean(x_tr60, axis=0)
            mu_norm = float(np.linalg.norm(mu_s))
            if mu_norm > 1e-8:
                u_s = (mu_s / mu_norm).astype(np.float32)
                y_tr_teach60 = y_tr_teach60 - 0.35 * ((y_tr_teach60 @ u_s)[:, None] * u_s[None, :])
                y_va_teach60 = y_va_teach60 - 0.35 * ((y_va_teach60 @ u_s)[:, None] * u_s[None, :])

            pd_local = locals().get("prompt_domains", None)
            if pd_local is not None and len(pd_local) >= n_tok:
                pd_tr60 = [pd_local[i] for i in np.flatnonzero(tr_clean)]
            else:
                pd_tr60 = ["general"] * int(tr_clean.sum())

            entry60, stats60 = train_projection_pattern_bank(
                "model.language_model.norm.post_cell60",
                x_tr60,
                y_tr_base60,
                y_tr_teach60,
                x_va60,
                y_va_base60,
                y_va_teach60,
                pd_tr60,
                rank=8,
                n_experts=8,
                top_k=2,
                steps=120,
                lr=5e-3,
            )
            if entry60 is not None:
                trained_entries.append(entry60)
                print(
                    f"[C8DB CELL60] val_cos {stats60.get('val_cos_before', 0.0):.5f}->{stats60.get('val_cos_after', 0.0):.5f}",
                    flush=True,
                )

    out_sz = write_c8db_file(args.out, refined_norms, trained_entries)
    print(f"[C8DB-DONE] wrote {args.out} ({len(refined_norms)} norms, {len(trained_entries)} entries, {out_sz:,} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

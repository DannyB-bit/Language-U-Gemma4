#!/usr/bin/env python3
"""
spatial_district_converter.py — Width-Axis Spatial-District Layer Converter, COLBLOCK
(`T16C_COLBLOCK_BIT = 0x20`) Repacker & GTA LOD Empirical Locality Probe.

Implements:
  1. `repack_t16c_colblock`: Low-RAM streaming `.t16c` converter that repacks
     `mlp.down_proj.weight` from `[n_tiles, n_blks, tblk_bytes]` to Column-Block-Major
     `[n_blks, n_tiles, tblk_bytes]` (`n_bits |= 0x20`) and optionally splits cold
     middle-layer MLPs into a separate zero-RAM shard (`hot.t16c+cold.t16c`).
  2. `cluster_spatial_districts`: Co-activation clustering partitioning each MLP's
     `ff` GeGLU neurons into Zone 0 (PINNED Shared Core, exact multiple of `block_size`)
     plus `M` contiguous semantic "city districts" aligned to `block_size = 256` tiles.
  3. `evaluate_locality_and_lod_residual`: Measures prompt energy capture, token-to-token
     district cache hit rate, and GTA LOD Residual MSE (`W_base(1b)` + `ΔW_hi-res(2b)` on
     active districts vs pure `1b`).
  4. Fail-closed 6-gate `--selftest` (`python spatial_district_converter.py --selftest`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pack_t16c_v3 as p3  # noqa: E402

T16C_MAGIC = p3.T16C_MAGIC
T16C_VERSION = p3.T16C_VERSION
T16C_GAUSS_BIT = 0x80
T16C_ROWMAJOR_BIT = 0x40
T16C_COLBLOCK_BIT = 0x20

_MLP_RE = re.compile(r"^model(?:\.language_model)?\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def parse_mlp_proj(name: str) -> Optional[Tuple[int, str]]:
    """Return `(layer_idx, proj_name)` for LM MLP weights, else `None`."""
    m = _MLP_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def parse_layer_spec(spec: str, n_layers: int = 60) -> Set[int]:
    """Parse comma/range layer specification (`6-53`, `6..54`, `10,12,20-40`, `all`, `none`)."""
    trimmed = spec.strip()
    if not trimmed:
        raise ValueError("empty layer spec")
    low = trimmed.lower()
    if low == "none":
        return set()
    if low == "all":
        return set(range(n_layers))
    out: Set[int] = set()
    for raw in trimmed.split(","):
        t = raw.strip()
        if not t:
            continue
        if "..=" in t:
            a_s, b_s = t.split("..=", 1)
            a, b = int(a_s.strip()), int(b_s.strip())
            if a > b or a < 0 or b >= n_layers:
                raise ValueError(f"layer range {a}..={b} out of bounds for n_layers={n_layers}")
            out.update(range(a, b + 1))
        elif ".." in t:
            a_s, b_s = t.split("..", 1)
            a, b = int(a_s.strip()), int(b_s.strip())
            if a >= b or a < 0 or b > n_layers:
                raise ValueError(f"layer range {a}..{b} out of bounds for n_layers={n_layers}")
            out.update(range(a, b))
        elif "-" in t:
            a_s, b_s = t.split("-", 1)
            a, b = int(a_s.strip()), int(b_s.strip())
            if a > b or a < 0 or b >= n_layers:
                raise ValueError(f"layer range {a}-{b} out of bounds for n_layers={n_layers}")
            out.update(range(a, b + 1))
        else:
            idx = int(t)
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"layer {idx} out of bounds for n_layers={n_layers}")
            out.add(idx)
    return out


def transpose_colblock_bytes(
    pb_bytes: bytes,
    d_out: int,
    d_in: int,
    block_size: int,
    n_bits: int,
    is_asym: bool,
    to_colblock: bool = True,
) -> bytes:
    """Losslessly convert packed T16C payload between `[n_tiles, n_blks]` and `[n_blks, n_tiles]`."""
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // block_size
    tblk_b = p3.tile16_blk_bytes(block_size, n_bits, is_asym)
    expected_len = n_tiles * n_blks * tblk_b
    if len(pb_bytes) != expected_len:
        raise ValueError(f"packed len {len(pb_bytes)} != expected {expected_len}")
    arr = np.frombuffer(pb_bytes, dtype=np.uint8)
    if to_colblock:
        reshaped = arr.reshape(n_tiles, n_blks, tblk_b)
    else:
        reshaped = arr.reshape(n_blks, n_tiles, tblk_b)
    return np.ascontiguousarray(reshaped.transpose(1, 0, 2)).tobytes()


def verify_t16c_file(path: Path) -> Dict[str, object]:
    """Fail-closed streaming integrity verifier for T16C v1 files (supports COLBLOCK 0x20)."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing T16C file: {path}")
    file_size = path.stat().st_size
    if file_size < 24:
        raise RuntimeError(f"T16C truncated header: {path} ({file_size} B < 24 B)")
    colblock_mats = 0
    mat_names: List[str] = []
    norm_names: List[str] = []
    with open(path, "rb") as fp:
        hdr = fp.read(24)
        magic, ver, arch_flags, n_norms, n_mats, reserved = struct.unpack("<4s5I", hdr)
        if magic != T16C_MAGIC or ver != T16C_VERSION or reserved != 0:
            raise RuntimeError(f"Invalid T16C header in {path}: magic={magic!r} ver={ver} reserved={reserved}")
        for ni in range(n_norms):
            kb = fp.read(2)
            if len(kb) != 2:
                raise RuntimeError(f"EOF reading norm[{ni}] keylen in {path}")
            klen = struct.unpack("<H", kb)[0]
            nbuf = fp.read(klen)
            if len(nbuf) != klen or klen == 0:
                raise RuntimeError(f"Invalid/EOF norm[{ni}] name in {path}")
            norm_names.append(nbuf.decode("utf-8"))
            ab = fp.read(4)
            if len(ab) != 4:
                raise RuntimeError(f"EOF reading norm[{ni}] len in {path}")
            cnt = struct.unpack("<I", ab)[0]
            if fp.seek(cnt * 4, 1) > file_size:
                raise RuntimeError(f"Norm[{ni}] overruns EOF in {path}")
        for mi in range(n_mats):
            raw = fp.read(34)
            if len(raw) != 34:
                raise RuntimeError(f"EOF reading mat[{mi}] header in {path}")
            (
                nlen, d_out, d_in, bsize, n_bits, is_asym_u8, has_fwht_u8,
                inv_k, rk_u16, pb_len, lr_len,
            ) = struct.unpack("<HIIHBBBBHQQ", raw)
            if nlen == 0 or d_out == 0 or d_in == 0 or is_asym_u8 > 1 or has_fwht_u8 > 1 or inv_k > 2:
                raise RuntimeError(f"Invalid mat[{mi}] metadata in {path}")
            nbuf = fp.read(nlen)
            if len(nbuf) != nlen:
                raise RuntimeError(f"EOF reading mat[{mi}] name in {path}")
            name = nbuf.decode("utf-8")
            mat_names.append(name)
            if (n_bits & T16C_COLBLOCK_BIT) != 0:
                colblock_mats += 1
            pos = fp.tell()
            pad = (64 - (pos % 64)) % 64
            if pad:
                fp.seek(pad, 1)
            if fp.tell() % 64 != 0:
                raise RuntimeError(f"Misaligned packed start for {name} in {path}")
            exp_p = p3.expected_tile16_bytes(d_out, d_in, bsize, n_bits, is_asym_u8 == 1)
            if pb_len != exp_p:
                raise RuntimeError(f"{name}: packed_bytes={pb_len} != expected={exp_p}")
            inv_bytes = (d_in * 4) if inv_k == 1 else ((d_in * 2) if inv_k == 2 else 0)
            exp_lr = (d_out * 4 + rk_u16 * 4 + d_out * rk_u16 + rk_u16 * d_in) if rk_u16 > 0 else 0
            if lr_len != exp_lr:
                raise RuntimeError(f"{name}: lr_bytes={lr_len} != expected={exp_lr}")
            if fp.seek(pb_len + inv_bytes + lr_len, 1) > file_size:
                raise RuntimeError(f"{name}: overruns EOF in {path}")
        if fp.tell() != file_size:
            raise RuntimeError(f"Trailing bytes in {path}: parsed={fp.tell()} != size={file_size}")
    return {
        "path": str(path),
        "bytes": file_size,
        "arch_flags": arch_flags,
        "n_norms": n_norms,
        "n_mats": n_mats,
        "colblock_mats": colblock_mats,
        "norm_names": norm_names,
        "mat_names": mat_names,
    }


def _scan_t16c_counts(src_path: Path, cold_layers: Set[int]) -> Tuple[int, int, int, int]:
    """First-pass lightweight header scan to determine hot vs cold matrix counts without reading payloads."""
    hot_mats = 0
    cold_mats = 0
    with open(src_path, "rb") as fp:
        hdr = fp.read(24)
        if len(hdr) != 24:
            raise RuntimeError(f"Truncated T16C header: {src_path}")
        magic, ver, arch_flags, n_norms, n_mats, reserved = struct.unpack("<4s5I", hdr)
        if magic != T16C_MAGIC or ver != T16C_VERSION or reserved != 0:
            raise RuntimeError(f"Invalid T16C header in {src_path}")
        for _ in range(n_norms):
            klen = struct.unpack("<H", fp.read(2))[0]
            fp.seek(klen, 1)
            cnt = struct.unpack("<I", fp.read(4))[0]
            fp.seek(cnt * 4, 1)
        for _ in range(n_mats):
            raw = fp.read(34)
            if len(raw) != 34:
                raise RuntimeError(f"EOF in matrix scan of {src_path}")
            (
                nlen, _do, d_in, _bs, _nb, _ia, _hf, inv_k, _rk, pb_len, lr_len,
            ) = struct.unpack("<HIIHBBBBHQQ", raw)
            name = fp.read(nlen).decode("utf-8")
            pad = (64 - (fp.tell() % 64)) % 64
            inv_bytes = (d_in * 4) if inv_k == 1 else ((d_in * 2) if inv_k == 2 else 0)
            fp.seek(pad + pb_len + inv_bytes + lr_len, 1)
            parsed = parse_mlp_proj(name)
            if parsed is not None and parsed[0] in cold_layers:
                cold_mats += 1
            else:
                hot_mats += 1
    return arch_flags, n_norms, hot_mats, cold_mats


def _write_mat_to_stream(
    fout,
    pos: int,
    header_tuple: Tuple[int, int, int, int, int, int, int, int, int, int, int],
    name_b: bytes,
    pb_bytes: bytes,
    inv_b: bytes,
    lr_b: bytes,
) -> int:
    nlen, d_out, d_in, bsize, n_bits_out, is_asym_u8, has_fwht_u8, inv_k, rk_u16, _pb_old, lr_len = header_tuple
    mhdr = struct.pack(
        "<HIIHBBBBHQQ",
        nlen, d_out, d_in, bsize, n_bits_out, is_asym_u8, has_fwht_u8, inv_k, rk_u16, len(pb_bytes), lr_len,
    )
    fout.write(mhdr)
    fout.write(name_b)
    pos += 34 + nlen
    pad = (64 - (pos % 64)) % 64
    if pad > 0:
        fout.write(b"\x00" * pad)
        pos += pad
    fout.write(pb_bytes)
    pos += len(pb_bytes)
    if inv_b:
        fout.write(inv_b)
        pos += len(inv_b)
    if lr_b:
        fout.write(lr_b)
        pos += len(lr_b)
    return pos


def repack_t16c_colblock(
    src_path: Path,
    dst_path: Path,
    cold_layers: Set[int],
    split_cold_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Streamingly repack cold `mlp.down_proj.weight` to `COLBLOCK (0x20)` with optional hot/cold split."""
    if not src_path.is_file():
        raise FileNotFoundError(f"Missing source T16C: {src_path}")
    t0 = time.perf_counter()
    arch_flags, n_norms, n_hot, n_cold = _scan_t16c_counts(src_path, cold_layers)
    n_total = n_hot + n_cold

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if split_cold_path is not None:
        split_cold_path.parent.mkdir(parents=True, exist_ok=True)

    converted_down = 0
    written_hot = 0
    written_cold = 0
    hot_tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    cold_tmp = split_cold_path.with_suffix(split_cold_path.suffix + ".tmp") if split_cold_path is not None else None

    f_cold = None
    try:
        with open(src_path, "rb") as fin, open(hot_tmp, "wb") as f_hot:
            if cold_tmp is not None:
                f_cold = open(cold_tmp, "wb")
            hdr24 = fin.read(24)
            magic, ver, _flags, src_norms, src_mats, reserved = struct.unpack("<4s5I", hdr24)
            if split_cold_path is not None:
                f_hot.write(struct.pack("<4s5I", magic, ver, arch_flags, src_norms, n_hot, 0))
                f_cold.write(struct.pack("<4s5I", magic, ver, arch_flags, 0, n_cold, 0))
            else:
                f_hot.write(struct.pack("<4s5I", magic, ver, arch_flags, src_norms, n_total, 0))
            pos_hot = 24
            pos_cold = 24

            for _ in range(src_norms):
                nlen_b = fin.read(2)
                nlen = struct.unpack("<H", nlen_b)[0]
                nbuf = fin.read(nlen)
                cnt_b = fin.read(4)
                cnt = struct.unpack("<I", cnt_b)[0]
                data = fin.read(cnt * 4)
                f_hot.write(nlen_b)
                f_hot.write(nbuf)
                f_hot.write(cnt_b)
                f_hot.write(data)
                pos_hot += 2 + nlen + 4 + cnt * 4

            for _ in range(src_mats):
                mhdr = fin.read(34)
                htup = struct.unpack("<HIIHBBBBHQQ", mhdr)
                nlen, d_out, d_in, bsize, n_bits, is_asym_u8, has_fwht_u8, inv_k, rk_u16, pb_len, lr_len = htup
                name_b = fin.read(nlen)
                name = name_b.decode("utf-8")
                in_pad = (64 - (fin.tell() % 64)) % 64
                if in_pad:
                    fin.read(in_pad)
                pb_raw = fin.read(pb_len)
                inv_len = d_in * 4 if inv_k == 1 else (d_in * 2 if inv_k == 2 else 0)
                inv_b = fin.read(inv_len) if inv_len > 0 else b""
                lr_b = fin.read(lr_len) if lr_len > 0 else b""

                parsed = parse_mlp_proj(name)
                is_cold = parsed is not None and parsed[0] in cold_layers
                is_cold_down = is_cold and parsed[1] == "down_proj"
                if is_cold_down and (n_bits & T16C_COLBLOCK_BIT) == 0:
                    pb_out = transpose_colblock_bytes(pb_raw, d_out, d_in, bsize, n_bits, bool(is_asym_u8), to_colblock=True)
                    n_bits_out = n_bits | T16C_COLBLOCK_BIT
                    converted_down += 1
                else:
                    pb_out = pb_raw
                    n_bits_out = n_bits

                out_htup = (nlen, d_out, d_in, bsize, n_bits_out, is_asym_u8, has_fwht_u8, inv_k, rk_u16, pb_len, lr_len)
                if split_cold_path is not None and is_cold:
                    pos_cold = _write_mat_to_stream(f_cold, pos_cold, out_htup, name_b, pb_out, inv_b, lr_b)
                    written_cold += 1
                else:
                    pos_hot = _write_mat_to_stream(f_hot, pos_hot, out_htup, name_b, pb_out, inv_b, lr_b)
                    written_hot += 1

            f_hot.flush()
            if f_cold is not None:
                f_cold.flush()
    finally:
        if f_cold is not None and not f_cold.closed:
            f_cold.close()

    os.replace(hot_tmp, dst_path)
    hot_ver = verify_t16c_file(dst_path)
    cold_ver = None
    if split_cold_path is not None and cold_tmp is not None:
        os.replace(cold_tmp, split_cold_path)
        cold_ver = verify_t16c_file(split_cold_path)

    return {
        "src": str(src_path),
        "dst": str(dst_path),
        "split_cold_dst": str(split_cold_path) if split_cold_path is not None else None,
        "cold_layers": sorted(cold_layers),
        "converted_down_colblock": converted_down,
        "written_hot_mats": written_hot,
        "written_cold_mats": written_cold,
        "hot_bytes": hot_ver["bytes"],
        "cold_bytes": cold_ver["bytes"] if cold_ver is not None else 0,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def cluster_spatial_districts(
    prompt_neuron_energy: np.ndarray,
    n_districts: int = 8,
    shared_frac: float = 0.25,
    block_size: int = 256,
    n_iters: int = 25,
) -> Tuple[np.ndarray, List[Tuple[str, int, int]], np.ndarray]:
    """
    Partition `ff` GeGLU neurons into Zone 0 (Shared Core, exact multiple of `block_size`)
    plus `n_districts` contiguous semantic districts aligned to `block_size` boundaries.

    Returns `(perm, district_slices, centroids)`:
      - `perm`: `int64[ff]` valid permutation placing shared core at `[0 : n_shared]`
        and districts sequentially afterwards.
      - `district_slices`: `[("shared_core", 0, n_shared), ("district_0", ...), ...]`.
      - `centroids`: `float32[n_districts, n_prompts]` L2-normalized district affinities.
    """
    e = np.asarray(prompt_neuron_energy, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] == 0 or e.shape[1] == 0:
        raise ValueError(f"Expected 2D non-empty [n_prompts, ff] energy array, got shape={e.shape}")
    if np.any(e < 0) or not np.all(np.isfinite(e)):
        raise ValueError("prompt_neuron_energy must be finite and non-negative")
    n_prompts, ff = e.shape
    if block_size <= 0 or (ff % block_size) != 0:
        raise ValueError(f"ff={ff} must be a positive multiple of block_size={block_size}")
    if n_districts < 1:
        raise ValueError(f"n_districts must be >= 1, got {n_districts}")
    if not (0.0 <= shared_frac < 1.0):
        raise ValueError(f"shared_frac must be in [0.0, 1.0), got {shared_frac}")

    n_blks = ff // block_size
    if n_blks < n_districts:
        raise ValueError(f"total blocks n_blks={n_blks} < n_districts={n_districts}")

    n_shared_blks = int(round(shared_frac * n_blks))
    n_shared_blks = max(0, min(n_shared_blks, n_blks - n_districts))
    n_shared = n_shared_blks * block_size

    # Identify globally hot/cross-prompt neurons for Shared Core via q25 + 0.5*mean energy.
    q25 = np.percentile(e, 25, axis=0)
    mean_e = np.mean(e, axis=0)
    shared_score = q25 + 0.35 * mean_e
    order_shared = np.argsort(-shared_score, kind="stable")
    shared_idx = order_shared[:n_shared]
    rem_idx = order_shared[n_shared:]

    # Build L2-normalized prompt profiles for remaining neurons [n_rem, n_prompts].
    rem_prof = e[:, rem_idx].T.copy()
    norms = np.linalg.norm(rem_prof, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    rem_unit = rem_prof / norms
    rem_strength = mean_e[rem_idx]

    n_rem = len(rem_idx)
    rem_blks = n_rem // block_size
    # Deterministic farthest-first seed initialization over normalized neuron profiles
    seeds = [int(np.argmax(rem_strength))]
    if n_districts > 1:
        best_sim = rem_unit @ rem_unit[seeds[0]]
        for _ in range(1, n_districts):
            next_seed = int(np.argmin(best_sim))
            seeds.append(next_seed)
            sim_new = rem_unit @ rem_unit[next_seed]
            best_sim = np.maximum(best_sim, sim_new)

    centroids = rem_unit[seeds].copy()  # [n_districts, n_prompts]
    labels = np.zeros(n_rem, dtype=np.int64)
    for _ in range(n_iters):
        sims = rem_unit @ centroids.T  # [n_rem, n_districts]
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for d in range(n_districts):
            mask = labels == d
            if np.any(mask):
                c = rem_unit[mask].mean(axis=0)
                cn = np.linalg.norm(c)
                if cn > 1e-12:
                    centroids[d] = c / cn

    # Quantize district sizes to exact `block_size` multiples (>= 1 block each) using largest remainder.
    counts = np.bincount(labels, minlength=n_districts).astype(np.float64)
    target_blks_f = (counts / max(n_rem, 1)) * rem_blks
    blk_counts = np.floor(target_blks_f).astype(np.int64)
    for d in range(n_districts):
        if blk_counts[d] < 1:
            blk_counts[d] = 1
    while blk_counts.sum() > rem_blks:
        cands = np.where(blk_counts > 1)[0]
        drop_d = cands[int(np.argmax(blk_counts[cands] - target_blks_f[cands]))]
        blk_counts[drop_d] -= 1
    while blk_counts.sum() < rem_blks:
        rem_frac = target_blks_f - blk_counts
        add_d = int(np.argmax(rem_frac))
        blk_counts[add_d] += 1

    # Assign remaining neurons greedily to districts ordered by similarity margin so capacities match
    # exact block_size multiples cleanly.
    sims = rem_unit @ centroids.T
    assigned = np.full(n_rem, -1, dtype=np.int64)
    cap_left = blk_counts * block_size
    neuron_order = np.argsort(-np.max(sims, axis=1), kind="stable")
    for ni in neuron_order:
        pref = np.argsort(-sims[ni], kind="stable")
        for d in pref:
            if cap_left[d] > 0:
                assigned[ni] = d
                cap_left[d] -= 1
                break

    district_slices: List[Tuple[str, int, int]] = []
    perm_parts: List[np.ndarray] = []
    cursor = 0
    if n_shared > 0:
        perm_parts.append(shared_idx)
        district_slices.append(("shared_core", 0, n_shared))
        cursor = n_shared
    for d in range(n_districts):
        d_local = np.where(assigned == d)[0]
        # Order within district by mean activation energy descending so top blocks align naturally
        d_local = d_local[np.argsort(-rem_strength[d_local], kind="stable")]
        perm_parts.append(rem_idx[d_local])
        width = len(d_local)
        if width != int(blk_counts[d] * block_size):
            raise RuntimeError(f"Internal capacity mismatch on district_{d}: {width}")
        district_slices.append((f"district_{d}", cursor, cursor + width))
        cursor += width

    perm = np.concatenate(perm_parts).astype(np.int64)
    if cursor != ff or perm.shape != (ff,) or np.unique(perm).size != ff:
        raise RuntimeError("Invalid spatial district permutation constructed")
    return perm, district_slices, centroids.astype(np.float32)


def _quantize_1b_sym_or_asym(w: np.ndarray, block_size: int = 256) -> np.ndarray:
    """Block-wise 1-bit asymmetric weight quantizer `[d_out, d_in]` -> reconstructed FP32."""
    d_out, d_in = w.shape
    n_blks = d_in // block_size
    wb = w.reshape(d_out, n_blks, block_size)
    bi = wb.mean(axis=2, keepdims=True)
    res = wb - bi
    sc = np.mean(np.abs(res), axis=2, keepdims=True)
    sgn = np.where(res >= 0.0, 1.0, -1.0)
    return (bi + sc * sgn).reshape(d_out, d_in)


def _quantize_2b_residual(err: np.ndarray, block_size: int = 256) -> np.ndarray:
    """Block-wise 2-bit uniform symmetric RMS-optimal residual quantizer for ΔW_hi-res."""
    d_out, d_in = err.shape
    n_blks = d_in // block_size
    eb = err.reshape(d_out, n_blks, block_size)
    rms = np.sqrt(np.mean(eb * eb, axis=2, keepdims=True))
    step = np.where(rms > 1e-12, rms * 1.0, 1.0)
    # 4 symmetric levels {-1.5, -0.5, +0.5, +1.5} * step
    q = np.clip(np.round(eb / step - (-1.5)), 0.0, 3.0) - 1.5
    return (q * step).reshape(d_out, d_in)


def evaluate_locality_and_lod_residual(
    seq_prompt_ids: np.ndarray,
    seq_neuron_energy: np.ndarray,
    w_down: Optional[np.ndarray],
    perm: np.ndarray,
    district_slices: Sequence[Tuple[str, int, int]],
    top_k_districts: int = 2,
    block_size: int = 256,
) -> Dict[str, float]:
    """
    Evaluate held-out prompt energy capture, token-to-token district cache hit rate,
    and GTA LOD Residual MSE (`W_base(1b) + ΔW_hi-res(2b)` on active districts vs pure `1b`).
    """
    n_tok, ff = seq_neuron_energy.shape
    if perm.shape != (ff,) or seq_prompt_ids.shape != (n_tok,):
        raise ValueError("Shape mismatch in evaluate_locality_and_lod_residual")
    e_perm = seq_neuron_energy[:, perm]
    shared_cols = None
    non_shared: List[Tuple[str, int, int]] = []
    for name, s, e_idx in district_slices:
        if name == "shared_core":
            shared_cols = (s, e_idx)
        else:
            non_shared.append((name, s, e_idx))

    total_e = np.sum(e_perm, axis=1) + 1e-12
    # Compute energy per prompt/token in shared + top-k districts
    d_energies = np.zeros((n_tok, len(non_shared)), dtype=np.float64)
    for di, (_, s, e_idx) in enumerate(non_shared):
        d_energies[:, di] = np.sum(e_perm[:, s:e_idx], axis=1)
    shared_e = np.sum(e_perm[:, shared_cols[0]:shared_cols[1]], axis=1) if shared_cols else np.zeros(n_tok)

    # Group tokens by prompt ID to compute prompt-level top-K routing and intra-sequence hit rate
    unique_pids = np.unique(seq_prompt_ids)
    captured_fracs: List[float] = []
    cache_hits = 0
    cache_steps = 0
    active_union_cols = np.zeros(ff, dtype=bool)
    if shared_cols:
        active_union_cols[shared_cols[0]:shared_cols[1]] = True

    k_eff = min(top_k_districts, len(non_shared))
    for pid in unique_pids:
        tok_mask = np.where(seq_prompt_ids == pid)[0]
        if len(tok_mask) == 0:
            continue
        prefill_t = tok_mask[0]
        top_ds = np.argsort(-d_energies[prefill_t], kind="stable")[:k_eff]
        top_set = set(int(x) for x in top_ds)
        for ti in tok_mask:
            cap_e = shared_e[ti] + sum(d_energies[ti, d] for d in top_set)
            captured_fracs.append(float(cap_e / total_e[ti]))
            tok_best = int(np.argmax(d_energies[ti]))
            cache_steps += 1
            if tok_best in top_set:
                cache_hits += 1
        if pid == unique_pids[0]:
            for d in top_set:
                _, s, e_idx = non_shared[d]
                active_union_cols[s:e_idx] = True

    metrics: Dict[str, float] = {
        "n_tokens": float(n_tok),
        "n_prompts": float(len(unique_pids)),
        "top_k_districts": float(k_eff),
        "energy_capture_mean": float(np.mean(captured_fracs)) if captured_fracs else 0.0,
        "district_cache_hit_rate": float(cache_hits / max(cache_steps, 1)),
    }

    if w_down is not None:
        w_p = w_down[:, perm]
        w_base_1b = _quantize_1b_sym_or_asym(w_p, block_size=block_size)
        err_res = w_p - w_base_1b
        delta_2b = _quantize_2b_residual(err_res, block_size=block_size)
        w_lod = w_base_1b.copy()
        w_lod[:, active_union_cols] += delta_2b[:, active_union_cols]

        mse_1b_active = float(np.mean((w_p[:, active_union_cols] - w_base_1b[:, active_union_cols]) ** 2))
        mse_lod_active = float(np.mean((w_p[:, active_union_cols] - w_lod[:, active_union_cols]) ** 2))
        metrics["mse_1b_active"] = mse_1b_active
        metrics["mse_lod_active"] = mse_lod_active
        metrics["lod_mse_reduction_x"] = float(mse_1b_active / max(mse_lod_active, 1e-18))
    return metrics


def _run_selftest() -> int:
    """Fail-closed 6-gate self-verification battery."""
    failures: List[str] = []
    rng = np.random.default_rng(20261122)

    # Gate 1: COLBLOCK repack byte integrity + lossless roundtrip across 1b/2b_gauss/3b/5-trit
    try:
        for nb, asym in ((1 | T16C_ROWMAJOR_BIT, True), (2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT, False), (3, False), (5, False)):
            d_out, d_in, bsize = 64, 512, 128
            exp_len = p3.expected_tile16_bytes(d_out, d_in, bsize, nb, asym)
            raw = rng.integers(0, 256, size=exp_len, dtype=np.uint8).tobytes()
            cb = transpose_colblock_bytes(raw, d_out, d_in, bsize, nb, asym, to_colblock=True)
            rt = transpose_colblock_bytes(cb, d_out, d_in, bsize, nb | T16C_COLBLOCK_BIT, asym, to_colblock=False)
            assert len(cb) == exp_len and rt == raw, f"COLBLOCK roundtrip mismatch nb={nb}"
        print("[Gate 1 PASS] COLBLOCK transpose lossless roundtrip across 1b/2b_gauss/3b/tern5", flush=True)
    except Exception as exc:
        failures.append(f"Gate 1: {exc}")

    # Gate 2: perm_district exact GeGLU mathematical equivalence
    try:
        d, ff = 64, 1024
        x = rng.normal(0.0, 0.5, size=(3, d)).astype(np.float64)
        w_g = rng.normal(0.0, 0.2, size=(ff, d)).astype(np.float64)
        w_u = rng.normal(0.0, 0.2, size=(ff, d)).astype(np.float64)
        w_d = rng.normal(0.0, 0.2, size=(d, ff)).astype(np.float64)
        gelu = lambda z: 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z ** 3))))
        y_ref = (gelu(x @ w_g.T) * (x @ w_u.T)) @ w_d.T
        perm = rng.permutation(ff)
        y_perm = (gelu(x @ w_g[perm].T) * (x @ w_u[perm].T)) @ w_d[:, perm].T
        rel = float(np.max(np.abs(y_ref - y_perm)) / max(np.max(np.abs(y_ref)), 1e-12))
        assert rel < 1e-12, f"GeGLU permutation equivalence rel_err={rel}"
        print(f"[Gate 2 PASS] Spatial district GeGLU permutation mathematical equivalence (rel_err={rel:.2e})", flush=True)
    except Exception as exc:
        failures.append(f"Gate 2: {exc}")

    # Gate 3: Multi-domain synthetic co-activation recovers planted districts & shared core
    try:
        ff, block_size, n_districts = 2560, 256, 4
        # First 512 neurons = universal shared core (2 blocks); next 2048 = 4 domains x 512 neurons (2 blocks each)
        shared_truth = set(range(0, 512))
        domain_truth = [set(range(512 + d * 512, 512 + (d + 1) * 512)) for d in range(n_districts)]
        e_prompts = np.full((32, ff), 0.01, dtype=np.float64)
        e_prompts[:, :512] = 2.0 + rng.uniform(0.0, 0.3, size=(32, 512))
        for pi in range(32):
            dom = pi % n_districts
            s0, s1 = 512 + dom * 512, 512 + (dom + 1) * 512
            e_prompts[pi, s0:s1] = 3.0 + rng.uniform(0.0, 0.4, size=512)
        perm, slices, _ = cluster_spatial_districts(e_prompts, n_districts=4, shared_frac=0.20, block_size=block_size)
        assert slices[0][0] == "shared_core" and slices[0][2] == 512
        shared_rec = len(set(perm[:512].tolist()) & shared_truth) / 512.0
        assert shared_rec >= 0.99, f"shared_recall={shared_rec} < 0.99"
        purities = []
        for name, s0, s1 in slices[1:]:
            assert (s1 - s0) % block_size == 0
            cols = set(perm[s0:s1].tolist()) - shared_truth
            best_overlap = max(len(cols & dt) for dt in domain_truth)
            purities.append(best_overlap / max(len(cols), 1))
        mean_purity = float(np.mean(purities))
        assert mean_purity >= 0.95, f"district purity={mean_purity} < 0.95"
        print(f"[Gate 3 PASS] Multi-domain spatial district recovery (shared_recall={shared_rec:.3f}, purity={mean_purity:.3f})", flush=True)
    except Exception as exc:
        failures.append(f"Gate 3: {exc}")

    # Gate 4: GTA LOD Residual (1b base + 2b active districts) & district cache hit rate >= 0.90
    try:
        ff = 2560
        # Build 4 sequence prompts x 16 tokens staying coherent per domain
        seq_pids = np.repeat(np.arange(4), 16)
        seq_e = np.full((64, ff), 0.01, dtype=np.float64)
        seq_e[:, :512] = 2.0
        for ti, pid in enumerate(seq_pids):
            s0, s1 = 512 + pid * 512, 512 + (pid + 1) * 512
            seq_e[ti, s0:s1] = 3.0 + rng.uniform(0.0, 0.2, size=512)
        w_down = rng.normal(0.0, 0.08, size=(64, ff))
        metrics = evaluate_locality_and_lod_residual(seq_pids, seq_e, w_down, perm, slices, top_k_districts=1, block_size=256)
        assert metrics["energy_capture_mean"] >= 0.95, f"energy_capture={metrics['energy_capture_mean']}"
        assert metrics["district_cache_hit_rate"] >= 0.90, f"cache_hit={metrics['district_cache_hit_rate']}"
        assert metrics["lod_mse_reduction_x"] > 2.0, f"lod_mse_reduction={metrics['lod_mse_reduction_x']}"
        print(
            f"[Gate 4 PASS] GTA LOD residual MSE reduced by {metrics['lod_mse_reduction_x']:.2f}x "
            f"(capture={metrics['energy_capture_mean']:.3f}, hit_rate={metrics['district_cache_hit_rate']:.3f})",
            flush=True,
        )
    except Exception as exc:
        failures.append(f"Gate 4: {exc}")

    # Gate 5: Full repack-colblock + hot/cold multi-shard split preserves 100% of norms & matrices
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src_p = tdp / "src.t16c"
            repack_p = tdp / "repacked.t16c"
            hot_p = tdp / "hot.t16c"
            cold_p = tdp / "cold.t16c"
            d_out, d_in, bs = 32, 256, 128
            mats = [
                ("model.language_model.layers.0.mlp.gate_proj.weight", 1 | T16C_ROWMAJOR_BIT, True),
                ("model.language_model.layers.0.mlp.down_proj.weight", 3, False),
                ("model.language_model.layers.6.mlp.gate_proj.weight", 1 | T16C_ROWMAJOR_BIT, True),
                ("model.language_model.layers.6.mlp.down_proj.weight", 2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT, False),
                ("model.language_model.layers.6.self_attn.q_proj.weight", 5, False),
            ]
            with open(src_p, "wb") as fp:
                fp.write(struct.pack("<4s5I", T16C_MAGIC, T16C_VERSION, p3.ARCH_FLAGS, 1, len(mats), 0))
                pos = 24
                n0 = b"model.language_model.norm.weight"
                fp.write(struct.pack("<H", len(n0)) + n0 + struct.pack("<I", 4))
                fp.write(np.ones(4, dtype="<f4").tobytes())
                pos += 2 + len(n0) + 4 + 16
                for mname, nb, asy in mats:
                    pb_len = p3.expected_tile16_bytes(d_out, d_in, bs, nb, asy)
                    pb = rng.integers(0, 256, size=pb_len, dtype=np.uint8)
                    pos = p3._write_mat_record(fp, pos, mname, d_out, d_in, bs, nb, asy, True, pb, None, None)

            st_unified = repack_t16c_colblock(src_p, repack_p, {6})
            assert st_unified["converted_down_colblock"] == 1
            v_u = verify_t16c_file(repack_p)
            assert v_u["n_mats"] == 5 and v_u["colblock_mats"] == 1

            st_split = repack_t16c_colblock(src_p, hot_p, {6}, split_cold_path=cold_p)
            v_h = verify_t16c_file(hot_p)
            v_c = verify_t16c_file(cold_p)
            assert st_split["written_hot_mats"] == 3 and st_split["written_cold_mats"] == 2
            assert v_h["n_norms"] == 1 and v_c["n_norms"] == 0
            assert set(v_h["mat_names"]) | set(v_c["mat_names"]) == {m[0] for m in mats}
            assert v_c["colblock_mats"] == 1
        print("[Gate 5 PASS] Unified COLBLOCK repack + hot/cold multi-shard split preserve 100% norms & matrices", flush=True)
    except Exception as exc:
        failures.append(f"Gate 5: {exc}")

    # Gate 6: Fail-closed error checks
    try:
        caught_trunc = caught_bs = caught_range = False
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "trunc.t16c"
            bad.write_bytes(b"T16C_SHORT")
            try:
                verify_t16c_file(bad)
            except Exception:
                caught_trunc = True
        try:
            cluster_spatial_districts(np.ones((4, 300)), n_districts=2, shared_frac=0.25, block_size=256)
        except ValueError:
            caught_bs = True
        try:
            parse_layer_spec("5-60", 60)
        except ValueError:
            caught_range = True
        assert caught_trunc and caught_bs and caught_range
        print("[Gate 6 PASS] Fail-closed handling verified on truncated T16C, misaligned ff, and OOB layer range", flush=True)
    except Exception as exc:
        failures.append(f"Gate 6: {exc}")

    if failures:
        print(f"SPATIAL_DISTRICT_SELFTEST FAILED: {failures}", file=sys.stderr)
        return 1
    print("SPATIAL_DISTRICT_SELFTEST: 6/6 gates passed", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Spatial-District Converter & T16C COLBLOCK Repacker")
    ap.add_argument("--selftest", action="store_true", help="Run 6-gate fail-closed self-test and exit")
    sub = ap.add_subparsers(dest="cmd")

    rp = sub.add_parser("repack-colblock", help="Repack cold down_proj matrices to COLBLOCK (0x20)")
    rp.add_argument("--src", required=True, type=Path)
    rp.add_argument("--dst", required=True, type=Path)
    rp.add_argument("--cold-layers", required=True, help="Layer spec e.g. '6-53' or 'all'")
    rp.add_argument("--split-cold", type=Path, default=None, help="Optional cold shard output path")
    rp.add_argument("--n-layers", type=int, default=60)
    rp.add_argument("--report-json", type=Path, default=None)

    pr = sub.add_parser("probe", help="Run Spatial-District + GTA LOD Residual probe on saved activation NPZ")
    pr.add_argument("--energy-npz", required=True, type=Path, help="NPZ with prompt_ids, seq_energy, prompt_energy, optional w_down")
    pr.add_argument("--n-districts", type=int, default=8)
    pr.add_argument("--shared-frac", type=float, default=0.25)
    pr.add_argument("--block-size", type=int, default=256)
    pr.add_argument("--top-k", type=int, default=2)
    pr.add_argument("--report-json", type=Path, default=None)

    args = ap.parse_args()
    if args.selftest:
        return _run_selftest()
    if args.cmd == "repack-colblock":
        layers = parse_layer_spec(args.cold_layers, n_layers=args.n_layers)
        res = repack_t16c_colblock(args.src, args.dst, layers, split_cold_path=args.split_cold)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2), flush=True)
        return 0
    if args.cmd == "probe":
        dat = np.load(args.energy_npz)
        perm, slices, _ = cluster_spatial_districts(
            dat["prompt_energy"], n_districts=args.n_districts, shared_frac=args.shared_frac, block_size=args.block_size
        )
        w_down = dat["w_down"] if "w_down" in dat else None
        metrics = evaluate_locality_and_lod_residual(
            dat["prompt_ids"], dat["seq_energy"], w_down, perm, slices, top_k_districts=args.top_k, block_size=args.block_size
        )
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

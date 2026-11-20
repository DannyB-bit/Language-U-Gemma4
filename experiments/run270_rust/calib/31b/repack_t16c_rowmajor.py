#!/usr/bin/env python3
"""
repack_t16c_rowmajor.py — Convert Column-Major T16C bit-serial matrices to Row-Major
(T16C_ROWMAJOR_BIT = 0x40) for ARMv8 NEON Horner AND+VCNT acceleration.

Identical byte size per tile-block (zero container bloat, 100% T16C v1 compatible).
Supports selective conversion (e.g., --modes 1b to convert only 1-bit matrices while
keeping 2-bit/3-bit matrices in column-major vqtbl1q_s8 layout, or --modes 1b,2b).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path
import numpy as np

T16C_MAGIC = b"T16C"
T16C_VERSION = 1
T16C_GAUSS_BIT = 0x80
T16C_ROWMAJOR_BIT = 0x40


def tile16_blk_bytes(block_size: int, n_bits: int, is_asym: bool) -> int:
    eff = n_bits & 0x3F
    if eff == 5:
        return 32 + ((block_size + 4) // 5) * 16
    if eff == 8:
        return 32 + block_size * 16
    hdr = 64 if is_asym else 32
    return hdr + eff * (block_size // 8) * 16


def transpose_planes_col_to_rowmajor(
    pb: np.ndarray, d_out: int, d_in: int, block_size: int, eff_bits: int, is_asym: bool
) -> np.ndarray:
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
        # Col-major [n_tiles, n_blks, bpr, 16] -> Row-major [n_tiles, n_blks, 16, bpr]
        pl = arr[:, :, s0:s1].reshape(n_tiles, n_blks, bpr, 16)
        arr[:, :, s0:s1] = np.ascontiguousarray(pl.transpose(0, 1, 3, 2)).reshape(n_tiles, n_blks, plane_b)
    return arr.reshape(-1)


def repack_file(src_path: Path, dst_path: Path, convert_eff_bits: set[int], skip_embed: bool = True) -> dict:
    t0 = time.perf_counter()
    converted = 0
    kept = 0
    total_bytes = 0
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        hdr24 = fin.read(24)
        magic, ver, flags, n_norms, n_mats, reserved = struct.unpack("<4s5I", hdr24)
        if magic != T16C_MAGIC or ver != T16C_VERSION:
            raise ValueError(f"Unsupported T16C header: magic={magic!r} ver={ver}")
        fout.write(hdr24)
        pos = 24
        for _ in range(n_norms):
            nlen_b = fin.read(2)
            nlen = struct.unpack("<H", nlen_b)[0]
            nbuf = fin.read(nlen)
            cnt_b = fin.read(4)
            cnt = struct.unpack("<I", cnt_b)[0]
            data = fin.read(cnt * 4)
            fout.write(nlen_b)
            fout.write(nbuf)
            fout.write(cnt_b)
            fout.write(data)
            pos += 2 + nlen + 4 + cnt * 4

        for _ in range(n_mats):
            mhdr = fin.read(34)
            nlen, d_out, d_in, bsize, n_bits, is_asym_u8, has_fwht_u8, inv_k, rk_u16, pb_len, lr_len = struct.unpack(
                "<HIIHBBBBHQQ", mhdr
            )
            name_b = fin.read(nlen)
            name = name_b.decode("utf-8")
            in_pos = fin.tell()
            pad = (64 - (in_pos % 64)) % 64
            pad_b = fin.read(pad) if pad > 0 else b""
            pb_raw = fin.read(pb_len)
            inv_len = d_in * 4 if inv_k == 1 else (d_in * 2 if inv_k == 2 else 0)
            inv_b = fin.read(inv_len) if inv_len > 0 else b""
            lr_b = fin.read(lr_len) if lr_len > 0 else b""

            eff = n_bits & 0x3F
            already_rm = (n_bits & T16C_ROWMAJOR_BIT) != 0
            is_embed = ("embed_tokens" in name) or ("lm_head" in name)
            do_conv = (
                (not already_rm)
                and (eff in convert_eff_bits)
                and (eff in (1, 2, 3, 4))
                and (not (skip_embed and is_embed))
            )
            if do_conv:
                pb_arr = np.frombuffer(pb_raw, dtype=np.uint8)
                pb_out = transpose_planes_col_to_rowmajor(
                    pb_arr, d_out, d_in, bsize, eff, bool(is_asym_u8)
                ).tobytes()
                n_bits_out = n_bits | T16C_ROWMAJOR_BIT
                converted += 1
            else:
                pb_out = pb_raw
                n_bits_out = n_bits
                kept += 1

            out_mhdr = struct.pack(
                "<HIIHBBBBHQQ",
                nlen, d_out, d_in, bsize, n_bits_out, is_asym_u8, has_fwht_u8, inv_k, rk_u16, len(pb_out), lr_len
            )
            fout.write(out_mhdr)
            fout.write(name_b)
            pos += 34 + nlen
            out_pad = (64 - (pos % 64)) % 64
            if out_pad > 0:
                fout.write(b"\x00" * out_pad)
                pos += out_pad
            fout.write(pb_out)
            pos += len(pb_out)
            if inv_b:
                fout.write(inv_b)
                pos += len(inv_b)
            if lr_b:
                fout.write(lr_b)
                pos += len(lr_b)
        fout.flush()
        total_bytes = pos
    dt = time.perf_counter() - t0
    return {"converted": converted, "kept": kept, "bytes": total_bytes, "elapsed_s": dt}


def main() -> int:
    ap = argparse.ArgumentParser(description="Repack T16C bit-serial matrices to Row-Major AND+VCNT layout.")
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--eff-bits", default="1", help="Comma-separated eff_bits to convert (e.g. '1' or '1,2').")
    ap.add_argument("--include-embed", action="store_true", help="Also convert embed_tokens (default: skip for CS sieve).")
    args = ap.parse_args()
    bits_set = {int(x.strip()) for x in args.eff_bits.split(",") if x.strip()}
    if not bits_set or not bits_set.issubset({1, 2, 3, 4}):
        raise SystemExit(f"Invalid --eff-bits {bits_set} (must be subset of 1,2,3,4)")
    stats = repack_file(args.src, args.dst, bits_set, skip_embed=not args.include_embed)
    print(
        f"[repack_t16c_rowmajor] src={args.src} -> dst={args.dst} "
        f"converted={stats['converted']} kept={stats['kept']} "
        f"bytes={stats['bytes']:,} elapsed={stats['elapsed_s']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

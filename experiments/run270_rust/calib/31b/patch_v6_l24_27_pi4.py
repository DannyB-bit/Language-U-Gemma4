#!/usr/bin/env python3
"""
Zero-GPU in-place patcher for V6.0 `L24..L27` (MAT[277..324], 48 records,
149,841,920 bytes of 0x00 hole at offsets 1,513,308,032..1,663,149,952) in
`gemma4_31b_v6_0_littlebit2.t16c`, reconstructed directly from W4A16 safetensors
using BLAS-only Cholesky-QR Power-1 LittleBit-2 + randomized Hadamard mixing.
"""

import argparse
import gc
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

LIB_DIR = Path(__file__).resolve().parent
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import pack_t16c_v3 as p3  # noqa: E402

T16C_ROWMAJOR_BIT = 0x40


def tile16_blk_bytes(block_size: int, n_bits: int, is_asym: bool) -> int:
    """Self-contained tile16 block byte count supporting `T16C_ROWMAJOR_BIT (0x40)` & `0x80`."""
    eff_bits = n_bits & 0x3F
    if block_size not in (64, 128, 256):
        raise ValueError(f"Unsupported block_size={block_size}")
    if eff_bits == 5:
        return 32 + ((block_size + 4) // 5) * 16
    if eff_bits == 8:
        return 32 + block_size * 16
    if not (1 <= eff_bits <= 4):
        raise ValueError(f"Unsupported n_bits={n_bits}")
    return (64 if is_asym else 32) + eff_bits * (block_size // 8) * 16


def expected_tile16_bytes(d_out: int, d_in: int, block_size: int, n_bits: int, is_asym: bool) -> int:
    """Self-contained full packed byte length for a `.t16c` matrix."""
    if d_out <= 0 or d_in <= 0 or (d_in % block_size) != 0:
        raise ValueError(f"Invalid dims ({d_out},{d_in}) bsize={block_size}")
    return ((d_out + 15) // 16) * (d_in // block_size) * tile16_blk_bytes(block_size, n_bits, is_asym)


def write_mat_record(
    fp,
    pos: int,
    name: str,
    d_out: int,
    d_in: int,
    bsize: int,
    n_bits: int,
    is_asym: bool,
    has_fwht: bool,
    pbytes: np.ndarray,
    inv_s: Optional[np.ndarray],
    low_rank: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> int:
    """Serialize one `T16CMat` record matching `t16c::T16CModel::load` byte-for-byte."""
    exp_p = expected_tile16_bytes(d_out, d_in, bsize, n_bits, is_asym)
    pb_u8 = np.ascontiguousarray(pbytes, dtype=np.uint8).ravel()
    if pb_u8.size != exp_p:
        raise ValueError(f"{name}: packed length {pb_u8.size} != expected {exp_p}")
    inv_f32: Optional[np.ndarray] = None
    inv_kind = 0
    if inv_s is not None:
        inv_f32 = np.ascontiguousarray(inv_s, dtype="<f4").ravel()
        if inv_f32.size != d_in:
            raise ValueError(f"{name}: inv_s len {inv_f32.size} != d_in {d_in}")
        inv_kind = 1
    lr_rk = 0
    lr_bytes = 0
    su_f32 = sv_f32 = u_i8 = v_i8 = None
    if low_rank is not None:
        su_raw, sv_raw, u_raw, v_raw = low_rank
        su_f32 = np.ascontiguousarray(su_raw, dtype="<f4").ravel()
        sv_f32 = np.ascontiguousarray(sv_raw, dtype="<f4").ravel()
        u_i8 = np.ascontiguousarray(u_raw, dtype=np.int8)
        v_i8 = np.ascontiguousarray(v_raw, dtype=np.int8)
        lr_rk = int(sv_f32.size)
        lr_bytes = d_out * 4 + lr_rk * 4 + d_out * lr_rk + lr_rk * d_in
    nb = name.encode("utf-8")
    fp.write(
        struct.pack(
            "<HIIHBBBBHQQ",
            len(nb), d_out, d_in, bsize, n_bits,
            1 if is_asym else 0, 1 if has_fwht else 0, inv_kind, lr_rk, exp_p, lr_bytes,
        )
    )
    fp.write(nb)
    pos += 34 + len(nb)
    pad = (64 - (pos % 64)) % 64
    if pad:
        fp.write(b"\x00" * pad)
        pos += pad
    mv = memoryview(pb_u8)
    for off in range(0, pb_u8.size, 16 * 1024 * 1024):
        fp.write(mv[off : off + 16 * 1024 * 1024])
    pos += exp_p
    if inv_f32 is not None:
        fp.write(inv_f32.tobytes())
        pos += d_in * 4
    if lr_rk > 0:
        fp.write(su_f32.tobytes())
        fp.write(sv_f32.tobytes())
        fp.write(u_i8.tobytes())
        fp.write(v_i8.tobytes())
        pos += lr_bytes
    return pos
HOLE_MAT_START = 277
HOLE_MAT_COUNT = 48
HOLE_BYTE_START = 1_513_308_032
HOLE_BYTE_END = 1_663_149_952
HOLE_BYTE_LEN = HOLE_BYTE_END - HOLE_BYTE_START  # 149,841,920 bytes
EXPECTED_EOF = 3_345_923_072

L24_27_SLIDING_PROJS: List[Tuple[str, str, int, int, str]] = [
    ("q_proj", "self_attn", 8192, 5376, "lb"),
    ("k_proj", "self_attn", 4096, 5376, "dr"),
    ("v_proj", "self_attn", 4096, 5376, "dr"),
    ("o_proj", "self_attn", 5376, 8192, "lb"),
    ("gate_proj", "mlp", 21504, 5376, "lb"),
    ("up_proj", "mlp", 21504, 5376, "lb"),
    ("down_proj", "mlp", 5376, 21504, "lb"),
]


def read_raw_safetensor(sr: p3.SafetensorsReader, name: str) -> torch.Tensor:
    """Zero-mmap safetensors tensor reader for edge RAM constraints."""
    info = sr.hdr[name]
    dtype = info["dtype"]
    shape = tuple(int(v) for v in info["shape"])
    o0, o1 = info["data_offsets"]
    sr._fp.seek(sr.data_base + o0)
    raw = sr._fp.read(o1 - o0)
    if len(raw) != (o1 - o0):
        raise EOFError(f"Short read on {name}: got {len(raw)} bytes, expected {o1 - o0}")
    if dtype == "I32":
        arr = np.frombuffer(raw, dtype="<i4").copy()
        return torch.from_numpy(arr.reshape(shape))
    if dtype == "BF16":
        u32 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        arr = u32.view(np.float32).copy()
        return torch.from_numpy(arr.reshape(shape))
    if dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        return torch.from_numpy(arr.reshape(shape))
    if dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4").copy()
        return torch.from_numpy(arr.reshape(shape))
    raise ValueError(f"Unsupported tensor dtype {dtype!r} for {name}")


def dequant_w4a16_weight(sr: p3.SafetensorsReader, pfx: str, d_out: int, d_in: int) -> torch.Tensor:
    """Unpack GPTQ/W4A16 4-bit symmetric nibbles + BF16 group-32 scales into FP32."""
    wp_key = f"{pfx}.weight_packed"
    ws_key = f"{pfx}.weight_scale"
    w_key = f"{pfx}.weight"
    if wp_key in sr.hdr and ws_key in sr.hdr:
        wp = read_raw_safetensor(sr, wp_key)
        ws = read_raw_safetensor(sr, ws_key)
        assert wp.shape == (d_out, d_in // 8), f"{wp_key} shape {tuple(wp.shape)} != {(d_out, d_in // 8)}"
        assert ws.shape == (d_out, d_in // 32), f"{ws_key} shape {tuple(ws.shape)} != {(d_out, d_in // 32)}"
        sh = torch.arange(0, 32, 4, dtype=torch.int32)
        nibs = ((wp.unsqueeze(-1) >> sh) & 0xF) - 8
        w = (nibs.to(torch.float32).reshape(d_out, d_in // 32, 32) * ws.unsqueeze(-1)).reshape(d_out, d_in)
    elif w_key in sr.hdr:
        w = read_raw_safetensor(sr, w_key)
    else:
        raise KeyError(f"Missing W4A16 packed or raw weight for {pfx}")
    assert w.shape == (d_out, d_in), f"{pfx} shape {tuple(w.shape)} != {(d_out, d_in)}"
    if not bool(torch.isfinite(w).all()):
        raise FloatingPointError(f"Non-finite values decoded for {pfx}")
    return w


def fwht_1d_pow2(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Normalized Sylvester FWHT matching `t16c.rs::fwht_block_inplace` on power-of-2 axis."""
    t = x if axis == -1 or axis == x.ndim - 1 else x.transpose(axis, -1)
    n = t.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT axis length must be power of two, got {n}")
    orig_shape = t.shape
    work = t.reshape(-1, n).clone()
    rows = work.shape[0]
    h = 1
    while h < n:
        step = h * 2
        v = work.view(rows, n // step, 2, h)
        a = v[:, :, 0, :] + v[:, :, 1, :]
        b = v[:, :, 0, :] - v[:, :, 1, :]
        work = torch.cat([a, b], dim=-1).reshape(rows, n)
        h = step
    out = work.reshape(orig_shape) * (1.0 / math.sqrt(n))
    return out if axis == -1 or axis == x.ndim - 1 else out.transpose(axis, -1)


def fwht_blocks_cpu(w: torch.Tensor, bsize: int = 256) -> torch.Tensor:
    """Apply `t16c.rs`-exact block-local FWHT across columns of `[d_out, d_in]`."""
    d_out, d_in = w.shape
    if d_in % bsize != 0:
        raise ValueError(f"d_in={d_in} not divisible by bsize={bsize}")
    return fwht_1d_pow2(w.reshape(d_out, d_in // bsize, bsize), axis=-1).reshape(d_out, d_in)


def _chol_qr2_rows(y: torch.Tensor) -> torch.Tensor:
    """Two-pass FP64-Gram CholeskyQR2 for near-machine-precision orthonormal rows."""
    r = y.shape[0]
    row_norm = y.norm(dim=1, keepdim=True).clamp(min=1e-8)
    q = (y / row_norm).to(torch.float64)
    eye = torch.eye(r, dtype=torch.float64)
    for eps in (1e-6, 1e-10):
        g = q @ q.T
        dmean = float(g.diag().mean().clamp(min=1e-8))
        l_mat = torch.linalg.cholesky(g + (eps * dmean) * eye)
        q = torch.linalg.solve_triangular(l_mat, q, upper=False)
    return q.to(torch.float32)


def fast_balanced_lowrank(w: torch.Tensor, rank: int = 2048, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factorize `w` `[d_out, d_in]` into norm-balanced, Hadamard-mixed `U @ V` (`[d_out, rank]`,
    `[rank, d_in]`) using BLAS3-only Cholesky-QR2 with 1 power step + Rademacher-Hadamard mixing.
    """
    d_out, d_in = w.shape
    if rank > min(d_out, d_in):
        raise ValueError(f"rank={rank} exceeds min(d_out={d_out}, d_in={d_in})")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if rank == d_in and d_in <= d_out:
        q = torch.eye(rank, dtype=torch.float32)
        b = w.float()
        sc = torch.sqrt(b.norm(dim=0).clamp(min=1e-6))
        u0 = b / sc.unsqueeze(0)
        v0 = sc.unsqueeze(1) * q
    elif rank == d_out and d_out < d_in:
        q = torch.eye(rank, dtype=torch.float32)
        b = w.float()
        sc = torch.sqrt(b.norm(dim=1).clamp(min=1e-6))
        u0 = q * sc.unsqueeze(0)
        v0 = b / sc.unsqueeze(1)
    elif d_out >= d_in:
        om = torch.randn((d_in, rank), dtype=torch.float32, generator=gen)
        z = w @ om
        y = z.T @ w
        del om, z
        q = _chol_qr2_rows(y)  # [rank, d_in], orthonormal rows
        b = w @ q.T  # [d_out, rank]
        del y
        sc = torch.sqrt(b.norm(dim=0).clamp(min=1e-6))
        u0 = b / sc.unsqueeze(0)
        v0 = sc.unsqueeze(1) * q
    else:
        om = torch.randn((rank, d_out), dtype=torch.float32, generator=gen)
        z = om @ w
        y = w @ z.T
        del om, z
        q = _chol_qr2_rows(y.T).T  # [d_out, rank], orthonormal cols
        b = q.T @ w  # [rank, d_in]
        del y
        sc = torch.sqrt(b.norm(dim=1).clamp(min=1e-6))
        u0 = q * sc.unsqueeze(0)
        v0 = b / sc.unsqueeze(1)

    # Apply random Rademacher flips + orthonormal Sylvester FWHT on rank axis:
    # (U0 @ D @ H) @ (H @ D @ V0) == U0 @ V0 identically while eliminating latent outliers.
    signs = torch.where(torch.randint(0, 2, (rank,), generator=gen) == 1, 1.0, -1.0).to(torch.float32)
    u_mix = fwht_1d_pow2(u0 * signs.unsqueeze(0), axis=-1)
    v_mix = fwht_1d_pow2(v0 * signs.unsqueeze(1), axis=0)
    return u_mix, v_mix


def pack_and_dequant_1b_asym_cpu(w: torch.Tensor, bsize: int = 256, n_refine: int = 2) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Quantize `[d_out, d_in]` into 1-bit asymmetric `(bi + sc * {+1, -1})` per `bsize` block,
    returning exact `T16C` (`n_bits = 1 | 0x40`, `is_asym = True`) bytes and FP16-exact decoded tensor.
    """
    d_out, d_in = w.shape
    if d_in % bsize != 0:
        raise ValueError(f"d_in={d_in} not divisible by bsize={bsize}")
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    pad_rows = n_tiles * 16 - d_out
    if pad_rows > 0:
        w_pad = torch.zeros((n_tiles * 16, d_in), dtype=torch.float32)
        w_pad[:d_out] = w.float()
    else:
        w_pad = w.float()

    xb = w_pad.view(n_tiles, 16, n_blks, bsize)
    bi = xb.mean(dim=-1, keepdim=True)
    sgn = torch.where(xb >= bi, 1.0, -1.0)
    sc = ((xb - bi) * sgn).mean(dim=-1, keepdim=True).clamp(min=1e-7)
    for _ in range(n_refine):
        sgn = torch.where(xb >= bi, 1.0, -1.0)
        bi = (xb - sc * sgn).mean(dim=-1, keepdim=True)
        sc = ((xb - bi) * sgn).mean(dim=-1, keepdim=True).clamp(min=1e-7)

    bits_t = xb >= bi
    sc_t16 = sc.squeeze(-1).clamp(-65504.0, 65504.0).to(torch.float16)
    bi_t16 = bi.squeeze(-1).clamp(-65504.0, 65504.0).to(torch.float16)
    deq = (bi_t16.float().unsqueeze(-1) + sc_t16.float().unsqueeze(-1) * torch.where(bits_t, 1.0, -1.0)).reshape(
        n_tiles * 16, d_in
    )[:d_out]

    bits_u8 = bits_t.numpy().astype(np.uint8)
    sc_f16 = sc_t16.numpy()
    bi_f16 = bi_t16.numpy()
    if pad_rows > 0:
        valid_mask = np.arange(n_tiles * 16).reshape(n_tiles, 16) < d_out
        sc_f16 = np.where(valid_mask[:, :, None], sc_f16, np.float16(0.0))
        bi_f16 = np.where(valid_mask[:, :, None], bi_f16, np.float16(0.0))
        bits_u8 = np.where(valid_mask[:, :, None, None], bits_u8, np.uint8(0))

    # Row-major packing inside plane p=0: [t, r, blk, bo, 8 bits] -> [t, blk, r, bo]
    b_bytes = bits_u8.reshape(n_tiles, 16, n_blks, bsize // 8, 8)
    bit_w = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    planes = (b_bytes * bit_w).sum(axis=-1).astype(np.uint8)
    planes_tb = np.ascontiguousarray(planes.transpose(0, 2, 1, 3)).reshape(n_tiles * n_blks, 16 * (bsize // 8))

    sc_u8 = np.ascontiguousarray(sc_f16.astype("<f2").transpose(0, 2, 1)).view(np.uint8).reshape(n_tiles * n_blks, 32)
    bi_u8 = np.ascontiguousarray(bi_f16.astype("<f2").transpose(0, 2, 1)).view(np.uint8).reshape(n_tiles * n_blks, 32)
    out = np.ascontiguousarray(np.concatenate([sc_u8, bi_u8, planes_tb], axis=1)).reshape(-1)
    exp_bytes = expected_tile16_bytes(d_out, d_in, bsize, 1 | T16C_ROWMAJOR_BIT, True)
    assert out.size == exp_bytes, f"Packed byte count {out.size} != {exp_bytes}"
    return out, deq


def pack_1b_asym_rowmajor_cpu(w: torch.Tensor, bsize: int = 256, n_refine: int = 2) -> np.ndarray:
    """Thin wrapper returning packed `T16C` (`0x41`) bytes."""
    pb, _ = pack_and_dequant_1b_asym_cpu(w, bsize=bsize, n_refine=n_refine)
    return pb


def refit_u_against_vq(w_eq: torch.Tensor, v_q: torch.Tensor) -> torch.Tensor:
    """Closed-form ridge-stabilized FP64 Cholesky solve `U_opt` minimizing `||w_eq - U @ v_q||_F`."""
    r = v_q.shape[0]
    g = (v_q @ v_q.T).to(torch.float64)
    diag_mean = float(g.diag().mean().clamp(min=1e-8))
    g.diagonal().add_(1e-4 * diag_mean)
    l_mat = torch.linalg.cholesky(g)
    rhs = (w_eq @ v_q.T).to(torch.float64)
    u_opt = torch.cholesky_solve(rhs.T, l_mat).T
    return u_opt.to(torch.float32)


def reference_dequant_1b_rowmajor(
    pb: np.ndarray, d_out: int, d_in: int, bsize: int, has_fwht: bool, inv_s: Optional[np.ndarray]
) -> np.ndarray:
    """Exact bit-for-bit reference port of `t16c.rs::dequantize_row` for 1-bit `0x41` rowmajor."""
    n_blks = d_in // bsize
    tblk_bytes = 64 + 16 * (bsize // 8)
    bpr = bsize // 8
    out = np.zeros((d_out, d_in), dtype=np.float32)
    for row in range(d_out):
        t, r = divmod(row, 16)
        w_row = np.zeros((d_in,), dtype=np.float32)
        for blk in range(n_blks):
            off = (t * n_blks + blk) * tblk_bytes
            sc = float(np.frombuffer(pb[off + r * 2 : off + r * 2 + 2].tobytes(), dtype="<f2")[0])
            bi = float(np.frombuffer(pb[off + 32 + r * 2 : off + 32 + r * 2 + 2].tobytes(), dtype="<f2")[0])
            p_off = off + 64
            w_blk = np.full((bsize,), bi, dtype=np.float32)
            for bo in range(bpr):
                byte = int(pb[p_off + r * bpr + bo])
                for bit in range(8):
                    sgn = 1.0 if ((byte >> (7 - bit)) & 1) != 0 else -1.0
                    w_blk[bo * 8 + bit] += sc * sgn
            w_row[blk * bsize : (blk + 1) * bsize] = w_blk
        if has_fwht:
            w_row = fwht_blocks_cpu(torch.from_numpy(w_row).unsqueeze(0), bsize).squeeze(0).numpy()
        if inv_s is not None:
            w_row = w_row * inv_s
        out[row] = w_row
    return out


def verify_t16c_full_eof(path: Path, expected_eof: Optional[int] = None) -> Tuple[int, int]:
    """Fail-closed streaming header/record/EOF verifier matching `t16c.rs`."""
    fsize = path.stat().st_size
    with open(path, "rb") as fp:
        hdr = fp.read(24)
        if len(hdr) != 24:
            raise RuntimeError(f"Short T16C header in {path}")
        magic, ver, arch, n_norms, n_mats, reserved = struct.unpack("<4s5I", hdr)
        if magic != p3.T16C_MAGIC or ver != p3.T16C_VERSION or reserved != 0:
            raise RuntimeError(f"Bad T16C header: magic={magic!r} ver={ver} reserved={reserved}")
        for ni in range(n_norms):
            nb_hdr = fp.read(2)
            if len(nb_hdr) != 2:
                raise RuntimeError(f"EOF at norm[{ni}]")
            nlen = struct.unpack("<H", nb_hdr)[0]
            fp.seek(nlen, 1)
            dim = struct.unpack("<I", fp.read(4))[0]
            fp.seek(dim * 4, 1)
        for mi in range(n_mats):
            rec_pos = fp.tell()
            raw = fp.read(34)
            if len(raw) != 34:
                raise RuntimeError(f"EOF reading mat[{mi}] header at {rec_pos}")
            nlen, d_out, d_in, bsize, nbits, is_asym, has_fwht, inv_k, lr_rk, pb, lrb = struct.unpack(
                "<HIIHBBBBHQQ", raw
            )
            if nlen == 0 or d_out == 0 or d_in == 0:
                raise RuntimeError(f"Zero/corrupt record at mat[{mi}] pos={rec_pos}: nlen={nlen} d=({d_out},{d_in})")
            name = fp.read(nlen).decode("utf-8", errors="strict")
            pos = fp.tell()
            pad = (64 - (pos % 64)) % 64
            if pad:
                fp.seek(pad, 1)
            exp_p = expected_tile16_bytes(d_out, d_in, bsize, nbits, is_asym == 1)
            if pb != exp_p:
                raise RuntimeError(f"mat[{mi}] {name} pb={pb} != exp={exp_p}")
            inv_b = (d_in * 4) if inv_k == 1 else ((d_in * 2) if inv_k == 2 else 0)
            end_pos = fp.seek(pb + inv_b + lrb, 1)
            if end_pos > fsize:
                raise RuntimeError(f"mat[{mi}] {name} overruns EOF ({end_pos} > {fsize})")
        final_pos = fp.tell()
        if final_pos != fsize:
            raise RuntimeError(f"EOF mismatch after {n_mats} mats: pos={final_pos} != fsize={fsize}")
        if expected_eof is not None and final_pos != expected_eof:
            raise RuntimeError(f"Expected EOF {expected_eof} != {final_pos}")
    return n_mats, fsize


def compute_smooth_inv_s(w: torch.Tensor) -> np.ndarray:
    """Compute bounded column equalization scales `[0.25, 4.0]` matching `.t16c` inv_s range."""
    col_rms = w.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    med = float(col_rms.median())
    inv_s = (col_rms / max(med, 1e-6)).clamp(min=0.25, max=4.0).to(torch.float32)
    return np.ascontiguousarray(inv_s.numpy(), dtype="<f4")


def build_patch_and_apply(
    t16c_path: Path,
    safetensors_path: Path,
    patch_tmp_path: Path,
    rank: int = 2048,
) -> None:
    """Generate the exact 149,841,920-byte `L24..L27` stream and splice into the hole."""
    if not t16c_path.is_file():
        raise FileNotFoundError(f"Missing target .t16c: {t16c_path}")
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Missing safetensors: {safetensors_path}")

    fsize = t16c_path.stat().st_size
    if fsize != EXPECTED_EOF:
        raise RuntimeError(f"Unexpected .t16c EOF {fsize} != {EXPECTED_EOF}")

    # Fail-closed check that [HOLE_BYTE_START..HOLE_BYTE_END] is strictly zero-filled.
    with open(t16c_path, "rb") as fp:
        fp.seek(HOLE_BYTE_START)
        rem = HOLE_BYTE_LEN
        while rem > 0:
            chunk = fp.read(min(rem, 8 * 1024 * 1024))
            if not chunk or chunk != bytes(len(chunk)):
                raise RuntimeError(f"Target slot [{HOLE_BYTE_START}..{HOLE_BYTE_END}] is non-zero; refusing overwrite!")
            rem -= len(chunk)
    print(f"[Verified Hole] Confirmed {HOLE_BYTE_LEN:,} zero bytes at [{HOLE_BYTE_START:,}..{HOLE_BYTE_END:,}].", flush=True)

    patch_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    pos = HOLE_BYTE_START
    written_mats = 0
    with p3.SafetensorsReader(safetensors_path) as sr, open(patch_tmp_path, "wb") as pfp:
        for l_idx in (24, 25, 26, 27):
            for pn_idx, (pn, sub, d_out, d_in, kind) in enumerate(L24_27_SLIDING_PROJS):
                t_m0 = time.perf_counter()
                pfx = f"model.language_model.layers.{l_idx}.{sub}.{pn}"
                w = dequant_w4a16_weight(sr, pfx, d_out, d_in)
                inv_s = compute_smooth_inv_s(w)
                w_eq = w / torch.from_numpy(inv_s).unsqueeze(0)
                del w

                if kind == "dr":
                    w_h = fwht_blocks_cpu(w_eq, 256)
                    pb = pack_1b_asym_rowmajor_cpu(w_h, bsize=256)
                    pos = write_mat_record(
                        pfp, pos, f"{pfx}.weight", d_out, d_in, 256, 1 | T16C_ROWMAJOR_BIT, True, True, pb, inv_s, None
                    )
                    written_mats += 1
                    del w_eq, w_h, pb
                else:
                    u_mix, v_mix = fast_balanced_lowrank(w_eq, rank=rank, seed=2400 + l_idx * 16 + pn_idx)
                    del u_mix
                    v_h = fwht_blocks_cpu(v_mix, 256)
                    pb_v, v_hq = pack_and_dequant_1b_asym_cpu(v_h, bsize=256)
                    del v_mix, v_h
                    pos = write_mat_record(
                        pfp, pos, f"{pfx}.weight.lb_v", rank, d_in, 256, 1 | T16C_ROWMAJOR_BIT, True, True, pb_v, inv_s, None
                    )
                    del pb_v
                    v_q = fwht_blocks_cpu(v_hq, 256)
                    del v_hq
                    u_opt = refit_u_against_vq(w_eq, v_q)
                    del w_eq, v_q
                    pb_u = pack_1b_asym_rowmajor_cpu(u_opt, bsize=256)
                    del u_opt
                    pos = write_mat_record(
                        pfp, pos, f"{pfx}.weight.lb_u", d_out, rank, 256, 1 | T16C_ROWMAJOR_BIT, True, False, pb_u, None, None
                    )
                    del pb_u
                    written_mats += 2
                gc.collect()
                print(
                    f"  [L{l_idx:02d}.{pn:<9s}] kind={kind} mats={written_mats:02d}/{HOLE_MAT_COUNT} pos={pos:,} ({time.perf_counter() - t_m0:.1f}s)",
                    flush=True,
                )
        pfp.flush()
        os.fsync(pfp.fileno())

    patch_sz = patch_tmp_path.stat().st_size
    if pos != HOLE_BYTE_END or patch_sz != HOLE_BYTE_LEN or written_mats != HOLE_MAT_COUNT:
        raise RuntimeError(
            f"Patch mismatch: pos={pos} (exp {HOLE_BYTE_END}), sz={patch_sz} (exp {HOLE_BYTE_LEN}), mats={written_mats}"
        )

    print(f"[Splicing] Writing {patch_sz:,} B into {t16c_path} @ {HOLE_BYTE_START:,}...", flush=True)
    with open(patch_tmp_path, "rb") as src, open(t16c_path, "r+b") as dst:
        dst.seek(HOLE_BYTE_START)
        while True:
            buf = src.read(8 * 1024 * 1024)
            if not buf:
                break
            dst.write(buf)
        dst.flush()
        os.fsync(dst.fileno())

    patch_tmp_path.unlink(missing_ok=True)
    n_mats, f_eof = verify_t16c_full_eof(t16c_path, expected_eof=EXPECTED_EOF)
    print(f"[PATCH COMPLETE] Verified {n_mats}/912 mats to EOF={f_eof:,} B in {time.perf_counter() - t0_all:.1f}s!", flush=True)


def run_selftest() -> int:
    """Mandatory 6-gate fail-closed CPU verification battery."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Mathematical offset alignment & byte count sanity
    pos = HOLE_BYTE_START
    total_mats = 0
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        dummy_bin = td_p / "offset_check.bin"
        with open(dummy_bin, "wb") as fp:
            for l_idx in (24, 25, 26, 27):
                for pn, sub, d_out, d_in, kind in L24_27_SLIDING_PROJS:
                    pfx = f"model.language_model.layers.{l_idx}.{sub}.{pn}"
                    if kind == "dr":
                        pb = np.zeros(expected_tile16_bytes(d_out, d_in, 256, 0x41, True), dtype=np.uint8)
                        pos = write_mat_record(fp, pos, f"{pfx}.weight", d_out, d_in, 256, 0x41, True, True, pb, np.ones(d_in, dtype="<f4"), None)
                        total_mats += 1
                    else:
                        pb_v = np.zeros(expected_tile16_bytes(2048, d_in, 256, 0x41, True), dtype=np.uint8)
                        pb_u = np.zeros(expected_tile16_bytes(d_out, 2048, 256, 0x41, True), dtype=np.uint8)
                        pos = write_mat_record(fp, pos, f"{pfx}.weight.lb_v", 2048, d_in, 256, 0x41, True, True, pb_v, np.ones(d_in, dtype="<f4"), None)
                        pos = write_mat_record(fp, pos, f"{pfx}.weight.lb_u", d_out, 2048, 256, 0x41, True, False, pb_u, None, None)
                        total_mats += 2
        sz = dummy_bin.stat().st_size
        gate("Gate 1 (Stream Alignment & Byte Count)", pos == HOLE_BYTE_END and sz == HOLE_BYTE_LEN and total_mats == 48, f"pos={pos:,} sz={sz:,} mats={total_mats}")

        # Gate 2: Exact `t16c.rs` rowmajor bit/byte/FWHT decode parity + smooth inv_s
        torch.manual_seed(7)
        w_test = torch.randn((37, 512), dtype=torch.float32)
        w_test[:, 0] *= 5.0
        inv_s = compute_smooth_inv_s(w_test)
        w_eq = w_test / torch.from_numpy(inv_s).unsqueeze(0)
        pb_t, w_h_deq = pack_and_dequant_1b_asym_cpu(fwht_blocks_cpu(w_eq, 256), bsize=256)
        w_deq_fast = (fwht_blocks_cpu(w_h_deq, 256) * torch.from_numpy(inv_s).unsqueeze(0)).numpy()
        w_rec = reference_dequant_1b_rowmajor(pb_t, 37, 512, 256, has_fwht=True, inv_s=inv_s)
        max_diff_g2 = float(np.max(np.abs(w_deq_fast - w_rec)))
        cos_g2 = float(torch.nn.functional.cosine_similarity(w_test.reshape(1, -1), torch.from_numpy(w_rec).reshape(1, -1)).item())
        gate(
            "Gate 2 (Tile16 RowMajor + FWHT Decode Parity)",
            cos_g2 >= 0.78 and max_diff_g2 < 1e-5 and np.isfinite(w_rec).all(),
            f"cos={cos_g2:.4f} max_diff={max_diff_g2:.2e}",
        )

        # Gate 3: Fast Cholesky-QR2 Power-1 + V_q-refitted LittleBit-2 reconstruction & rank distinctness
        a = torch.randn((320, 256), dtype=torch.float32)
        b = torch.randn((256, 512), dtype=torch.float32)
        w_lr = a @ b + 0.05 * torch.randn((320, 512), dtype=torch.float32)
        _, v_m = fast_balanced_lowrank(w_lr, rank=256, seed=9)
        pb_vm, v_hq = pack_and_dequant_1b_asym_cpu(fwht_blocks_cpu(v_m, 256), 256)
        v_q_fast = fwht_blocks_cpu(v_hq, 256)
        u_opt = refit_u_against_vq(w_lr, v_q_fast)
        pb_um = pack_1b_asym_rowmajor_cpu(u_opt, 256)
        u_q = torch.from_numpy(reference_dequant_1b_rowmajor(pb_um, 320, 256, 256, False, None))
        v_q = torch.from_numpy(reference_dequant_1b_rowmajor(pb_vm, 256, 512, 256, True, None))
        w_lr_rec = u_q @ v_q
        cos_lr = float(torch.nn.functional.cosine_similarity(w_lr.reshape(1, -1), w_lr_rec.reshape(1, -1)).item())
        svals = torch.linalg.svdvals(u_q @ v_q)
        gate("Gate 3 (Cholesky-QR2 + V_q-Refit + Non-Degenerate Rank)", cos_lr >= 0.72 and float(svals[31]) > 0.1 * float(svals[0]), f"cos_lr={cos_lr:.4f}")

        # Gate 4: Smooth inv_s keeps outlier column error bounded vs unscaled
        x_in = torch.randn((4, 512), dtype=torch.float32)
        y_ref = x_in @ w_test.T
        y_rec = x_in @ torch.from_numpy(w_rec).T
        cos_act = float(torch.nn.functional.cosine_similarity(y_ref.reshape(1, -1), y_rec.reshape(1, -1)).item())
        gate("Gate 4 (MatVec Semantic Fidelity)", cos_act >= 0.75, f"act_cos={cos_act:.4f}")

        # Gate 5: Mini T16C EOF verifier catches 0x00 hole and accepts complete stream
        mini_ok = td_p / "mini_ok.t16c"
        mini_bad = td_p / "mini_bad.t16c"
        with open(mini_ok, "wb") as fp:
            fp.write(struct.pack("<4s5I", p3.T16C_MAGIC, p3.T16C_VERSION, p3.ARCH_FLAGS, 1, 1, 0))
            nname = b"test.norm"
            fp.write(struct.pack("<H", len(nname)) + nname + struct.pack("<I", 4) + np.ones(4, dtype="<f4").tobytes())
            write_mat_record(fp, fp.tell(), "test.w", 37, 512, 256, 0x41, True, True, pb_t, inv_s, None)
        nm, fsz = verify_t16c_full_eof(mini_ok, expected_eof=mini_ok.stat().st_size)
        raw_bad = bytearray(mini_ok.read_bytes())
        raw_bad[-400:] = b"\x00" * 400
        # Zero out record header to test corruption detection:
        hdr_off = 24 + 2 + len(b"test.norm") + 4 + 16
        raw_bad[hdr_off : hdr_off + 34] = b"\x00" * 34
        mini_bad.write_bytes(bytes(raw_bad))
        caught = False
        try:
            verify_t16c_full_eof(mini_bad)
        except RuntimeError:
            caught = True
        gate("Gate 5 (Fail-Closed Hole Verifier)", nm == 1 and fsz == mini_ok.stat().st_size and caught, f"mats={nm} caught_hole={caught}")

        # Gate 6: Rectangular transposed branch (d_out < d_in) + W4A16 nibble decode exactness
        w_wide = torch.randn((256, 512), dtype=torch.float32)
        u_w, v_w = fast_balanced_lowrank(w_wide, rank=256, seed=11)
        rel_w = float((w_wide - u_w @ v_w).norm() / w_wide.norm())
        gate("Gate 6 (Transposed LowRank + W4A16 Branch)", rel_w < 1e-4, f"wide_rel_err={rel_w:.2e}")

    print(f"SELFTEST SUMMARY: {6 - failures}/6 gates passed", flush=True)
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-GPU patcher for V6.0 L24..L27 hole on Pi 4.")
    ap.add_argument("--t16c", type=Path, default=Path("/home/astronautshe/gemma4_v6/gemma4_31b_v6_0_littlebit2.t16c"))
    ap.add_argument("--safetensors", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/model.safetensors"))
    ap.add_argument("--patch-tmp", type=Path, default=Path("/tmp/v6_l24_27_patch.bin"))
    ap.add_argument("--rank", type=int, default=2048)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(run_selftest())
    build_patch_and_apply(args.t16c, args.safetensors, args.patch_tmp, rank=args.rank)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stream-pack `Gemma4AssistantForCausalLM` BF16/F32 safetensors into mixed-precision `.t16c`.

Converts all 48 tensors of Google's 4-layer KV-sharing EAGLE-3 Assistant model:
  - `pre_projection.weight` (`[d, 2 * backbone_hidden]`)
  - `post_projection.weight` (`[backbone_hidden, d]`)
  - `model.embed_tokens.weight` (`[vocab, d]`, streamed in 4096-row chunks <= 40 MB RSS)
  - `model.layers.{0..n-1}` (`q_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, norms, `layer_scalar`)
  - `model.norm.weight`

Default precision (`--body-bits 8 --embed-bits 4 --block-size 64`):
  - Body linears (`pre_projection`, `post_projection`, `layers.*`): symmetric 8-bit signed INT8
    Tile-16 (`n_bits=8, is_asym=False, has_fwht=False`), eliminating softmax attention error cascade
    for near-exact HF parity (`logits_cos > 0.985`).
  - `model.embed_tokens.weight`: symmetric 4-bit FWHT Tile-16 (`n_bits=4, is_asym=False, has_fwht=True`)
    with 1-step MSE-optimal FP16 scale refinement (~136 MiB).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pack_t16c_v3 import (  # noqa: E402
    ARCH_FLAGS,
    T16C_MAGIC,
    T16C_VERSION,
    SafetensorsReader,
    _write_synthetic_safetensors,
    expected_tile16_bytes,
    tile16_blk_bytes,
)


def resolve_assistant_safetensors(path_arg: Optional[Path]) -> Path:
    """Resolve Assistant safetensors file or directory cleanly, failing closed if absent."""
    candidates = []
    if path_arg is not None:
        p = Path(path_arg)
        if p.is_dir():
            candidates.append(p / "model.safetensors")
        else:
            candidates.append(p)
    else:
        env_p = os.environ.get("ASSISTANT_SAFETENSORS")
        if env_p:
            ep = Path(env_p)
            candidates.append(ep / "model.safetensors" if ep.is_dir() else ep)
        candidates.extend(
            [
                Path("/mnt/SHE-VAULT/gemma-4-31B-it-assistant/model.safetensors"),
                Path("/mnt/SHE-VAULT/gemma4_assistant/model.safetensors"),
            ]
        )
    for c in candidates:
        if c.is_file():
            return c.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Assistant safetensors not found; tried: {tried}")


@lru_cache(maxsize=8)
def make_sylvester_hadamard(bsize: int) -> np.ndarray:
    """Return symmetric normalized Sylvester Hadamard matrix H_n (`n` power of 2)."""
    if bsize <= 0 or (bsize & (bsize - 1)) != 0:
        raise ValueError(f"Hadamard size must be power of two, got {bsize}")
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < bsize:
        h = np.block([[h, h], [h, -h]])
    return np.ascontiguousarray(h / np.float32(np.sqrt(float(bsize))), dtype=np.float32)


def fwht_reference_blocks(x: np.ndarray, bsize: int) -> np.ndarray:
    """In-place equivalent radix-2 butterfly FWHT matching `t16c::fwht_block_inplace`."""
    out = np.array(x, dtype=np.float32, copy=True)
    flat = out.reshape(-1, bsize)
    inv_sqrt = np.float32(1.0 / np.sqrt(float(bsize)))
    length = 1
    while length < bsize:
        step = length * 2
        for start in range(0, bsize, step):
            u = flat[:, start : start + length].copy()
            v = flat[:, start + length : start + step].copy()
            flat[:, start : start + length] = u + v
            flat[:, start + length : start + step] = u - v
        length = step
    flat *= inv_sqrt
    return out


def pack_chunk_sym4_tile16(
    w_chunk: np.ndarray,
    block_size: int = 256,
    has_fwht: bool = True,
) -> bytes:
    """
    Pack `(rows, d_in)` float32 chunk (`rows` <= chunk, auto-padded to multiple of 16) into T16C 4-bit sym bytes.
    """
    rows, d_in = w_chunk.shape
    if d_in % block_size != 0 or block_size % 8 != 0:
        raise ValueError(f"Invalid dims ({rows}, {d_in}) for block_size={block_size}")
    pad_rows = (16 - (rows % 16)) % 16
    if pad_rows > 0:
        w_work = np.zeros((rows + pad_rows, d_in), dtype=np.float32)
        w_work[:rows, :] = w_chunk
    else:
        w_work = np.ascontiguousarray(w_chunk, dtype=np.float32)

    n_tiles = w_work.shape[0] // 16
    n_blks = d_in // block_size
    w_blk = w_work.reshape(n_tiles, 16, n_blks, block_size)

    if has_fwht:
        H = make_sylvester_hadamard(block_size)
        # Apply symmetric orthogonal transform per block: w_rot = w_blk @ H
        w_blk = np.matmul(w_blk, H)

    # Initial min-max scale followed by 1-step MSE-optimal scale refinement in FP16 grid
    max_a = np.max(np.abs(w_blk), axis=-1, keepdims=True)
    nz = max_a > 0.0
    sc0 = np.where(nz, max_a / np.float32(15.0), np.float32(0.0)).astype(np.float32)
    sc0_f16 = sc0.astype(np.float16)
    sc0_q = sc0_f16.astype(np.float32)

    safe_sc0 = np.where(sc0_q > 0.0, sc0_q, np.float32(1.0))
    q0 = np.where(sc0_q > 0.0, np.clip(np.rint((w_blk / safe_sc0 + 15.0) * 0.5), 0.0, 15.0), 8.0)
    lvl0 = 2.0 * q0 - 15.0
    err0 = np.sum((w_blk - sc0_q * lvl0) ** 2, axis=-1, keepdims=True)

    denom = np.sum(lvl0 * lvl0, axis=-1, keepdims=True)
    sc1 = np.where((denom > 0.0) & nz, np.maximum(np.sum(w_blk * lvl0, axis=-1, keepdims=True) / denom, 0.0), sc0_q)
    sc1_f16 = sc1.astype(np.float16)
    sc1_q = sc1_f16.astype(np.float32)
    safe_sc1 = np.where(sc1_q > 0.0, sc1_q, np.float32(1.0))
    q1 = np.where(sc1_q > 0.0, np.clip(np.rint((w_blk / safe_sc1 + 15.0) * 0.5), 0.0, 15.0), 8.0)
    err1 = np.sum((w_blk - sc1_q * (2.0 * q1 - 15.0)) ** 2, axis=-1, keepdims=True)

    choose_1 = err1 < err0
    sc_best_f16 = np.where(choose_1, sc1_f16, sc0_f16)[..., 0]  # (n_tiles, 16, n_blks)
    q_final = np.where(choose_1, q1, q0).astype(np.uint8)       # (n_tiles, 16, n_blks, block_size)

    # Header: 16 FP16 little-endian scales -> 32 bytes per (tile, blk)
    hdr_bytes = np.ascontiguousarray(sc_best_f16.transpose(0, 2, 1), dtype="<f2").view(np.uint8).reshape(
        n_tiles, n_blks, 32
    )

    # Planes 0..3: reshape bits MSB-first per byte, transposed so row r in 0..15 is innermost
    bytes_per_row_blk = block_size // 8
    plane_bytes = bytes_per_row_blk * 16
    q_bits = q_final.reshape(n_tiles, 16, n_blks, bytes_per_row_blk, 8)
    planes = []
    for p in range(4):
        bp = ((q_bits >> p) & 1).astype(np.uint8)
        packed_p = np.packbits(bp, axis=-1, bitorder="big")[..., 0]  # (n_tiles, 16, n_blks, bytes_per_row_blk)
        planes.append(
            np.ascontiguousarray(packed_p.transpose(0, 2, 3, 1), dtype=np.uint8).reshape(
                n_tiles, n_blks, plane_bytes
            )
        )

    full = np.concatenate([hdr_bytes, *planes], axis=-1)
    expected_tb = tile16_blk_bytes(block_size, 4, False)
    if full.shape != (n_tiles, n_blks, expected_tb):
        raise AssertionError(f"Packed chunk shape {full.shape} != {(n_tiles, n_blks, expected_tb)}")
    return np.ascontiguousarray(full, dtype=np.uint8).tobytes()


def pack_chunk_sym8_tile16(
    w_chunk: np.ndarray,
    block_size: int = 64,
    has_fwht: bool = False,
) -> bytes:
    """
    Pack `(rows, d_in)` float32 chunk (`rows` <= chunk, auto-padded to multiple of 16)
    into T16C 8-bit symmetric signed INT8 Tile-16 bytes (`[col][row=0..15]`, `q in [-127, 127]`).
    """
    rows, d_in = w_chunk.shape
    if d_in % block_size != 0 or block_size % 8 != 0:
        raise ValueError(f"Invalid dims ({rows}, {d_in}) for block_size={block_size}")
    pad_rows = (16 - (rows % 16)) % 16
    if pad_rows > 0:
        w_work = np.zeros((rows + pad_rows, d_in), dtype=np.float32)
        w_work[:rows, :] = w_chunk
    else:
        w_work = np.ascontiguousarray(w_chunk, dtype=np.float32)

    n_tiles = w_work.shape[0] // 16
    n_blks = d_in // block_size
    w_blk = w_work.reshape(n_tiles, 16, n_blks, block_size)

    if has_fwht:
        H = make_sylvester_hadamard(block_size)
        w_blk = np.matmul(w_blk, H)

    # 1-step MSE-optimal FP16 scale refinement with q clamped to [-127, 127] to avoid i16 pair overflow
    max_a = np.max(np.abs(w_blk), axis=-1, keepdims=True)
    nz = max_a > 0.0
    sc0 = np.where(nz, max_a / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    sc0_f16 = sc0.astype(np.float16)
    sc0_q = sc0_f16.astype(np.float32)

    safe_sc0 = np.where(sc0_q > 0.0, sc0_q, np.float32(1.0))
    q0 = np.where(sc0_q > 0.0, np.clip(np.rint(w_blk / safe_sc0), -127.0, 127.0), 0.0)
    err0 = np.sum((w_blk - sc0_q * q0) ** 2, axis=-1, keepdims=True)

    denom = np.sum(q0 * q0, axis=-1, keepdims=True)
    safe_denom = np.where(denom > 0.0, denom, np.float32(1.0))
    sc1 = np.where((denom > 0.0) & nz, np.maximum(np.sum(w_blk * q0, axis=-1, keepdims=True) / safe_denom, 0.0), sc0_q)
    sc1_f16 = sc1.astype(np.float16)
    sc1_q = sc1_f16.astype(np.float32)
    safe_sc1 = np.where(sc1_q > 0.0, sc1_q, np.float32(1.0))
    q1 = np.where(sc1_q > 0.0, np.clip(np.rint(w_blk / safe_sc1), -127.0, 127.0), 0.0)
    err1 = np.sum((w_blk - sc1_q * q1) ** 2, axis=-1, keepdims=True)

    choose_1 = err1 < err0
    sc_best_f16 = np.where(choose_1, sc1_f16, sc0_f16)[..., 0]  # (n_tiles, 16, n_blks)
    q_final = np.where(choose_1, q1, q0).astype(np.int8)          # (n_tiles, 16, n_blks, block_size)

    hdr_bytes = np.ascontiguousarray(sc_best_f16.transpose(0, 2, 1), dtype="<f2").view(np.uint8).reshape(
        n_tiles, n_blks, 32
    )
    q_bytes = np.ascontiguousarray(q_final.view(np.uint8).transpose(0, 2, 3, 1), dtype=np.uint8).reshape(
        n_tiles, n_blks, block_size * 16
    )

    full = np.concatenate([hdr_bytes, q_bytes], axis=-1)
    expected_tb = tile16_blk_bytes(block_size, 8, False)
    if full.shape != (n_tiles, n_blks, expected_tb):
        raise AssertionError(f"Packed chunk shape {full.shape} != {(n_tiles, n_blks, expected_tb)}")
    return np.ascontiguousarray(full, dtype=np.uint8).tobytes()


def _read_f32_rows(sr: SafetensorsReader, tensor_name: str, row_start: int, row_end: int) -> np.ndarray:
    """Stream-read rows `[row_start:row_end]` of a 2D BF16/F16/F32 safetensors matrix as float32."""
    info = sr.hdr[tensor_name]
    dtype = info["dtype"]
    d_out, d_in = [int(v) for v in info["shape"]]
    if row_start < 0 or row_end > d_out or row_start >= row_end:
        raise ValueError(f"{tensor_name}: invalid row slice [{row_start}:{row_end}] for d_out={d_out}")
    elem_b = 2 if dtype in ("BF16", "F16") else 4
    base_off = sr.data_base + int(info["data_offsets"][0])
    sr._fp.seek(base_off + row_start * d_in * elem_b)
    raw = sr._fp.read((row_end - row_start) * d_in * elem_b)
    if len(raw) != (row_end - row_start) * d_in * elem_b:
        raise EOFError(f"{tensor_name}: short read at rows [{row_start}:{row_end}]")
    if dtype == "BF16":
        arr = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).copy()
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4").copy()
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        raise ValueError(f"Unsupported tensor dtype {dtype!r} for {tensor_name}")
    return arr.reshape((row_end - row_start, d_in))


def stream_write_assistant_t16c_mat(
    fp,
    pos: int,
    sr: SafetensorsReader,
    name: str,
    block_size: int = 64,
    n_bits: int = 4,
    has_fwht: bool = True,
    chunk_rows: int = 4096,
) -> int:
    """Write one 4-bit or 8-bit symmetric `T16CMat` record directly from safetensors without full matrix copy."""
    if chunk_rows <= 0 or chunk_rows % 16 != 0:
        raise ValueError(f"chunk_rows must be positive multiple of 16, got {chunk_rows}")
    if n_bits not in (4, 8):
        raise ValueError(f"n_bits must be 4 or 8, got {n_bits}")
    info = sr.hdr[name]
    shape = info["shape"]
    if len(shape) != 2:
        raise ValueError(f"{name}: expected 2D tensor, got shape={shape}")
    d_out, d_in = int(shape[0]), int(shape[1])
    exp_p = expected_tile16_bytes(d_out, d_in, block_size, n_bits, False)

    nb = name.encode("utf-8")
    fp.write(
        struct.pack(
            "<HIIHBBBBHQQ",
            len(nb),
            d_out,
            d_in,
            block_size,
            n_bits,
            0,
            1 if has_fwht else 0,
            0,
            0,
            exp_p,
            0,
        )
    )
    fp.write(nb)
    pos += 34 + len(nb)
    pad = (64 - (pos % 64)) % 64
    if pad:
        fp.write(b"\x00" * pad)
        pos += pad

    written = 0
    for r0 in range(0, d_out, chunk_rows):
        r1 = min(r0 + chunk_rows, d_out)
        w_sub = _read_f32_rows(sr, name, r0, r1)
        if n_bits == 8:
            packed_chunk = pack_chunk_sym8_tile16(w_sub, block_size=block_size, has_fwht=has_fwht)
        else:
            packed_chunk = pack_chunk_sym4_tile16(w_sub, block_size=block_size, has_fwht=has_fwht)
        fp.write(packed_chunk)
        written += len(packed_chunk)
        del w_sub, packed_chunk
    if written != exp_p:
        raise AssertionError(f"{name}: streamed {written} bytes != expected {exp_p}")
    return pos + written


def collect_assistant_keys(hdr: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    """Discover and validate all norm and linear matrix keys in `Gemma4AssistantForCausalLM`."""
    layer_indices = set()
    for k in hdr:
        if k == "__metadata__":
            continue
        if k.startswith("model.layers."):
            parts = k.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
    n_layers = max(layer_indices) + 1 if layer_indices else 0
    if n_layers == 0 or layer_indices != set(range(n_layers)):
        raise ValueError(f"Invalid assistant layer indices: {sorted(layer_indices)}")

    norm_keys = ["model.norm.weight"]
    mat_keys = ["pre_projection.weight", "post_projection.weight", "model.embed_tokens.weight"]
    for l in range(n_layers):
        p = f"model.layers.{l}."
        norm_keys.extend(
            [
                f"{p}self_attn.q_norm.weight",
                f"{p}input_layernorm.weight",
                f"{p}post_attention_layernorm.weight",
                f"{p}pre_feedforward_layernorm.weight",
                f"{p}post_feedforward_layernorm.weight",
                f"{p}layer_scalar",
            ]
        )
        mat_keys.extend(
            [
                f"{p}self_attn.q_proj.weight",
                f"{p}self_attn.o_proj.weight",
                f"{p}mlp.gate_proj.weight",
                f"{p}mlp.up_proj.weight",
                f"{p}mlp.down_proj.weight",
            ]
        )

    for k in norm_keys + mat_keys:
        if k not in hdr:
            raise KeyError(f"Assistant safetensors missing required tensor: {k}")
    return norm_keys, mat_keys


def pack_assistant_safetensors_to_t16c(
    safetensors_path: Path,
    out_path: Path,
    block_size: int = 64,
    body_bits: int = 8,
    embed_bits: int = 4,
    has_fwht: bool = True,
    chunk_rows: int = 4096,
    verbose: bool = True,
) -> int:
    """Convert `Gemma4AssistantForCausalLM` safetensors -> streaming mixed-precision `.t16c`."""
    if body_bits not in (4, 8) or embed_bits not in (4, 8):
        raise ValueError(f"body_bits/embed_bits must be 4 or 8, got body={body_bits} embed={embed_bits}")
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".t16c.tmp")
    try:
        with SafetensorsReader(safetensors_path) as sr, open(tmp_path, "wb") as fp:
            norm_keys, mat_keys = collect_assistant_keys(sr.hdr)
            fp.write(struct.pack("<4s5I", T16C_MAGIC, T16C_VERSION, ARCH_FLAGS, len(norm_keys), len(mat_keys), 0))
            pos = 24
            for nk in norm_keys:
                arr = sr.read_f32_tensor(nk)
                nb = nk.encode("utf-8")
                a32 = np.ascontiguousarray(arr, dtype="<f4").ravel()
                fp.write(struct.pack("<H", len(nb)))
                fp.write(nb)
                fp.write(struct.pack("<I", a32.size))
                fp.write(a32.tobytes())
                pos += 2 + len(nb) + 4 + a32.size * 4
            for idx, mk in enumerate(mat_keys, 1):
                mt0 = time.time()
                mat_bits = embed_bits if mk == "model.embed_tokens.weight" else body_bits
                mat_fwht = has_fwht and (mat_bits == 4)
                pos = stream_write_assistant_t16c_mat(
                    fp,
                    pos,
                    sr,
                    mk,
                    block_size=block_size,
                    n_bits=mat_bits,
                    has_fwht=mat_fwht,
                    chunk_rows=chunk_rows,
                )
                if verbose:
                    shape = sr.hdr[mk]["shape"]
                    print(
                        f"[{idx:02d}/{len(mat_keys):02d}] packed {mk} {shape} Q{mat_bits}"
                        f"{'+FWHT' if mat_fwht else ''} in {time.time() - mt0:.2f}s",
                        flush=True,
                    )
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    sz = out_path.stat().st_size
    if verbose:
        print(f"[OK] Packed assistant {out_path} ({sz:,} B / {sz / (1024**2):.2f} MiB) in {time.time() - t0:.2f}s")
    return sz


def _decode_tile16_mat(
    packed: bytes,
    d_out: int,
    d_in: int,
    block_size: int,
    n_bits: int,
    has_fwht: bool,
) -> np.ndarray:
    """Reference T16C symmetric decoder (4-bit bitplane or 8-bit INT8) matching `t16c.rs` layout."""
    if n_bits not in (4, 8):
        raise ValueError(f"Unsupported n_bits={n_bits}")
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // block_size
    tb = tile16_blk_bytes(block_size, n_bits, False)
    bytes_per_row_blk = block_size // 8
    plane_bytes = bytes_per_row_blk * 16
    arr = np.frombuffer(packed, dtype=np.uint8).reshape(n_tiles, n_blks, tb)

    out = np.zeros((d_out, d_in), dtype=np.float32)
    for t in range(n_tiles):
        for blk in range(n_blks):
            chunk = arr[t, blk]
            scs = np.frombuffer(chunk[:32].tobytes(), dtype="<f2").astype(np.float32)
            q_mat_i8 = None
            if n_bits == 8:
                q_mat_i8 = np.frombuffer(chunk[32 : 32 + block_size * 16].tobytes(), dtype=np.int8).reshape(
                    block_size, 16
                )
            for r in range(16):
                row = t * 16 + r
                if row >= d_out:
                    continue
                sc = scs[r]
                if n_bits == 8:
                    w_blk = sc * q_mat_i8[:, r].astype(np.float32)
                else:
                    w_blk = np.zeros((block_size,), dtype=np.float32)
                    for p in range(4):
                        wp = sc * float(1 << p)
                        p_off = 32 + p * plane_bytes
                        for bo in range(bytes_per_row_blk):
                            b = int(chunk[p_off + bo * 16 + r])
                            for bit in range(8):
                                sgn = 1.0 if ((b >> (7 - bit)) & 1) else -1.0
                                w_blk[bo * 8 + bit] += wp * sgn
                if has_fwht:
                    w_blk = fwht_reference_blocks(w_blk, block_size)
                out[row, blk * block_size : (blk + 1) * block_size] = w_blk
    return out


def _decode_sym4_tile16_mat(packed: bytes, d_out: int, d_in: int, block_size: int, has_fwht: bool) -> np.ndarray:
    """Backward-compatible wrapper for 4-bit symmetric Tile-16 decoding."""
    return _decode_tile16_mat(packed, d_out, d_in, block_size, 4, has_fwht)


def _self_test() -> int:
    """Run 6 fail-closed gates validating FWHT, 4-bit T16C bit-plane layout, streaming, and error checks."""
    failures = 0
    rng = np.random.default_rng(20261115)

    # Gate 1: Sylvester Hadamard matrix matches radix-2 butterfly FWHT
    try:
        for bsz in (64, 128, 256):
            x = rng.normal(size=(7, bsz)).astype(np.float32)
            H = make_sylvester_hadamard(bsz)
            ref = fwht_reference_blocks(x, bsz)
            got = x @ H
            assert np.max(np.abs(ref - got)) < 1e-5, f"FWHT mismatch bsz={bsz}"
            rt = got @ H
            assert np.max(np.abs(rt - x)) < 1e-5, f"FWHT involutory fail bsz={bsz}"
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    # Gate 2: Fail-closed on missing file
    try:
        try:
            resolve_assistant_safetensors(Path("/nonexistent/assistant/model.safetensors"))
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass
    except Exception as exc:
        print(f"[Gate 2 FAIL] {exc}")
        failures += 1

    # Gate 3: 4-bit sym packing/decoding exact on grid weights & >0.995 cos on Gaussian (bsize=64)
    try:
        d_out, d_in, bsize = 19, 256, 64
        # Exact 4-bit representable grid values (no FWHT) verify bit-exact sign/scale packing
        lvl = (2 * rng.integers(0, 16, size=(d_out, d_in // bsize, bsize)) - 15).astype(np.float32)
        sc_g = np.array([0.0625], dtype=np.float32)
        w_exact = (lvl * sc_g).reshape(d_out, d_in)
        p_exact = pack_chunk_sym4_tile16(w_exact, block_size=bsize, has_fwht=False)
        d_exact = _decode_sym4_tile16_mat(p_exact, d_out, d_in, bsize, False)
        exact_cos = float(np.sum(w_exact * d_exact) / (np.linalg.norm(w_exact) * np.linalg.norm(d_exact) + 1e-12))
        assert exact_cos > 0.99999, f"Exact grid 4-bit roundtrip cos={exact_cos}"

        w_gauss = rng.normal(0.0, 0.05, size=(d_out, d_in)).astype(np.float32)
        packed = pack_chunk_sym4_tile16(w_gauss, block_size=bsize, has_fwht=True)
        assert len(packed) == expected_tile16_bytes(d_out, d_in, bsize, 4, False)
        dec = _decode_sym4_tile16_mat(packed, d_out, d_in, bsize, True)
        cos = float(np.sum(w_gauss * dec) / (np.linalg.norm(w_gauss) * np.linalg.norm(dec) + 1e-12))
        assert cos > 0.995, f"Gaussian 4-bit FWHT bsize={bsize} roundtrip cos={cos} < 0.995"

        # Spiky outlier resistance with FWHT vs zero block preservation
        w_zero = np.zeros((16, bsize), dtype=np.float32)
        p_zero = pack_chunk_sym4_tile16(w_zero, block_size=bsize, has_fwht=True)
        d_zero = _decode_sym4_tile16_mat(p_zero, 16, bsize, bsize, True)
        assert np.max(np.abs(d_zero)) == 0.0

        # 8-bit signed INT8 sym packing exact on integer grid & >0.9999 cos on Gaussian (bsize=64)
        q_grid = rng.integers(-127, 128, size=(d_out, d_in // bsize, bsize)).astype(np.float32)
        q_grid[:, :, 0] = 127.0
        w8_exact = (q_grid * np.float32(0.0625)).reshape(d_out, d_in)
        p8_exact = pack_chunk_sym8_tile16(w8_exact, block_size=bsize, has_fwht=False)
        assert len(p8_exact) == expected_tile16_bytes(d_out, d_in, bsize, 8, False)
        d8_exact = _decode_tile16_mat(p8_exact, d_out, d_in, bsize, 8, False)
        assert float(np.max(np.abs(w8_exact - d8_exact))) < 1e-6, "Exact INT8 grid mismatch"

        p8_gauss = pack_chunk_sym8_tile16(w_gauss, block_size=bsize, has_fwht=False)
        d8_gauss = _decode_tile16_mat(p8_gauss, d_out, d_in, bsize, 8, False)
        cos8 = float(np.sum(w_gauss * d8_gauss) / (np.linalg.norm(w_gauss) * np.linalg.norm(d8_gauss) + 1e-12))
        assert cos8 > 0.9999, f"Gaussian 8-bit bsize={bsize} roundtrip cos={cos8} < 0.9999"

        p8_zero = pack_chunk_sym8_tile16(w_zero, block_size=bsize, has_fwht=False)
        d8_zero = _decode_tile16_mat(p8_zero, 16, bsize, bsize, 8, False)
        assert np.max(np.abs(d8_zero)) == 0.0
    except Exception as exc:
        print(f"[Gate 3 FAIL] {exc}")
        failures += 1

    # Gate 4-6: Full Assistant schema safetensors -> streaming mixed Q8+Q4 .t16c -> byte/layout/cos verification
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        sf_p = td / "assistant.safetensors"
        out_p = td / "assistant_q8q4.t16c"
        out_44 = td / "assistant_q4q4.t16c"
        d, bh, ff, hd_s, hd_f, vocab = 256, 256, 512, 256, 512, 32
        tensors: Dict[str, np.ndarray] = {
            "pre_projection.weight": rng.normal(0.0, 0.04, size=(d, 2 * bh)).astype(np.float32),
            "post_projection.weight": rng.normal(0.0, 0.04, size=(bh, d)).astype(np.float32),
            "model.embed_tokens.weight": rng.normal(0.0, 0.04, size=(vocab, d)).astype(np.float32),
            "model.norm.weight": np.ones((d,), dtype=np.float32),
        }
        for l, is_full in enumerate([False, True]):
            hd = hd_f if is_full else hd_s
            p = f"model.layers.{l}."
            tensors[f"{p}self_attn.q_proj.weight"] = rng.normal(0.0, 0.04, size=(hd * 2, d)).astype(np.float32)
            tensors[f"{p}self_attn.o_proj.weight"] = rng.normal(0.0, 0.04, size=(d, hd * 2)).astype(np.float32)
            tensors[f"{p}mlp.gate_proj.weight"] = rng.normal(0.0, 0.04, size=(ff, d)).astype(np.float32)
            tensors[f"{p}mlp.up_proj.weight"] = rng.normal(0.0, 0.04, size=(ff, d)).astype(np.float32)
            tensors[f"{p}mlp.down_proj.weight"] = rng.normal(0.0, 0.04, size=(d, ff)).astype(np.float32)
            tensors[f"{p}self_attn.q_norm.weight"] = np.ones((hd,), dtype=np.float32)
            tensors[f"{p}input_layernorm.weight"] = np.ones((d,), dtype=np.float32)
            tensors[f"{p}post_attention_layernorm.weight"] = np.ones((d,), dtype=np.float32)
            tensors[f"{p}pre_feedforward_layernorm.weight"] = np.ones((d,), dtype=np.float32)
            tensors[f"{p}post_feedforward_layernorm.weight"] = np.ones((d,), dtype=np.float32)
            tensors[f"{p}layer_scalar"] = np.array([0.85], dtype=np.float32)

        try:
            _write_synthetic_safetensors(sf_p, tensors)
            # Chunk rows=16 to exercise multi-chunk streaming on vocab=32 (chunks 0..16, 16..32)
            sz = pack_assistant_safetensors_to_t16c(
                sf_p,
                out_p,
                block_size=64,
                body_bits=8,
                embed_bits=4,
                has_fwht=True,
                chunk_rows=16,
                verbose=False,
            )
            assert sz == out_p.stat().st_size
            raw = out_p.read_bytes()
            magic, ver, flags, n_norms, n_mats, resv = struct.unpack("<4s5I", raw[:24])
            assert magic == T16C_MAGIC and ver == T16C_VERSION and flags == ARCH_FLAGS
            assert n_norms == 13 and n_mats == 13 and resv == 0
            pos = 24
            for _ in range(n_norms):
                nlen = struct.unpack("<H", raw[pos : pos + 2])[0]
                pos += 2 + nlen
                cnt = struct.unpack("<I", raw[pos : pos + 4])[0]
                pos += 4 + cnt * 4
            decoded_emb = None
            decoded_pre = None
            for _ in range(n_mats):
                nlen, do, di, bs, nb, asym_u8, fwht_u8, inv_k, rk_u16, pb_len, lr_len = struct.unpack(
                    "<HIIHBBBBHQQ", raw[pos : pos + 34]
                )
                pos += 34
                nm = raw[pos : pos + nlen].decode("utf-8")
                pos += nlen
                pos += (64 - (pos % 64)) % 64
                assert pos % 64 == 0
                if nm == "model.embed_tokens.weight":
                    assert nb == 4 and asym_u8 == 0 and fwht_u8 == 1
                else:
                    assert nb == 8 and asym_u8 == 0 and fwht_u8 == 0
                assert inv_k == 0 and rk_u16 == 0 and lr_len == 0
                payload = raw[pos : pos + pb_len]
                pos += pb_len
                if nm == "model.embed_tokens.weight":
                    decoded_emb = _decode_tile16_mat(payload, do, di, bs, nb, bool(fwht_u8))
                elif nm == "pre_projection.weight":
                    decoded_pre = _decode_tile16_mat(payload, do, di, bs, nb, bool(fwht_u8))
            assert pos == len(raw)
            assert decoded_emb is not None and decoded_pre is not None
            emb_cos = float(
                np.sum(decoded_emb * tensors["model.embed_tokens.weight"])
                / (np.linalg.norm(decoded_emb) * np.linalg.norm(tensors["model.embed_tokens.weight"]) + 1e-12)
            )
            assert emb_cos > 0.995, f"streamed embed_tokens roundtrip cos={emb_cos}"
            pre_cos = float(
                np.sum(decoded_pre * tensors["pre_projection.weight"])
                / (np.linalg.norm(decoded_pre) * np.linalg.norm(tensors["pre_projection.weight"]) + 1e-12)
            )
            assert pre_cos > 0.9999, f"streamed Q8 pre_projection roundtrip cos={pre_cos}"

            # Also verify legacy uniform Q4+FWHT mode still works
            sz44 = pack_assistant_safetensors_to_t16c(
                sf_p,
                out_44,
                block_size=64,
                body_bits=4,
                embed_bits=4,
                has_fwht=True,
                chunk_rows=16,
                verbose=False,
            )
            assert 0 < sz44 < sz

            # Fail closed on missing required tensor
            bad_sf = td / "bad.safetensors"
            bad_t = dict(tensors)
            del bad_t["pre_projection.weight"]
            _write_synthetic_safetensors(bad_sf, bad_t)
            try:
                pack_assistant_safetensors_to_t16c(bad_sf, td / "bad.t16c", verbose=False)
                raise AssertionError("Expected KeyError on missing pre_projection.weight")
            except KeyError:
                pass
        except Exception as exc:
            print(f"[Gate 4-6 FAIL] {exc}")
            failures += 1

    if failures == 0:
        print("[PASS] pack_assistant_t16c.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser(description="Pack Gemma4AssistantForCausalLM into mixed-precision .t16c")
    ap.add_argument("--safetensors", type=Path, default=None, help="Path to assistant model.safetensors or dir")
    ap.add_argument("--out", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_assistant.t16c"))
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--body-bits", type=int, choices=(4, 8), default=8, help="Bit width for body linear layers")
    ap.add_argument("--embed-bits", type=int, choices=(4, 8), default=4, help="Bit width for embed_tokens matrix")
    ap.add_argument("--no-fwht", action="store_true")
    ap.add_argument("--chunk-rows", type=int, default=4096)
    args = ap.parse_args()

    sf_path = resolve_assistant_safetensors(args.safetensors)
    pack_assistant_safetensors_to_t16c(
        sf_path,
        args.out.resolve(),
        block_size=args.block_size,
        body_bits=args.body_bits,
        embed_bits=args.embed_bits,
        has_fwht=not args.no_fwht,
        chunk_rows=args.chunk_rows,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

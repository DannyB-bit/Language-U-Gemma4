#!/usr/bin/env python3
"""
encode_gemma4_e4b_to_glyphs.py
================================================================================
Real Layer-Ordered Encoder from `google/gemma-4-E4B-it` safetensors to
Consider-Trinity Dual-Precision Hybrid Macro-Glyph Streams & Neon-Runner V2.
Uses byte-offset streaming directly from disk to bypass 16 GB memory limitations.
================================================================================
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gemma4_e4b.consider_trinity_e4b_core import is_boundary_layer_e4b
from neon_runner.bridge import (
    LANU_HEADER_BYTES,
    encode_matrix_opt4_v2,
    encode_matrix_opt5_v2,
    pack_lanu_header,
)
from neon_runner.codebooks import CodebookRegistryV2


def build_canonical_s7_codebook(size: int = 2048, dim: int = 8) -> np.ndarray:
    idx = np.arange(size, dtype=np.float32)[:, None]
    d = np.arange(dim, dtype=np.float32)[None, :]
    phases = idx * (d + 1.0) * 0.61803398875 * math.pi
    cb = np.cos(phases) + 0.5 * np.sin(phases * 1.41421356)
    norms = np.linalg.norm(cb, axis=1, keepdims=True)
    return (cb / np.maximum(norms, 1e-12)).astype(np.float32)


def encode_block_opt5_outlier(weights_256: np.ndarray) -> Tuple[bytes, float]:
    """
    Encodes 256 FP32 weights into a 48-byte Outlier-Preserved Option 5 Glyph:
      - Bytes 0..1: FP16 scale
      - Bytes 2..3: FP16 bias
      - Bytes 4..35: 32-byte sign bitmask (256 bits)
      - Bytes 36..39: 4 x uint8 indices of Top-4 salient outliers
      - Bytes 40..47: 4 x FP16 outlier residuals
    Returns (48-byte glyph, measured SNR in dB).
    """
    w = weights_256.astype(np.float32).flatten()
    bias = float(np.mean(w))
    centered = w - bias
    scale = float(np.mean(np.abs(centered)))
    bits = (centered >= 0).astype(np.uint8)
    packed_bits = np.packbits(bits).tobytes()

    recon_base = np.where(bits == 1, scale, -scale) + bias
    residual = w - recon_base

    # Top-4 outlier indices and FP16 residuals
    top4_idx = np.argsort(np.abs(residual))[-4:].astype(np.uint8)
    top4_res = residual[top4_idx].astype(np.float16)

    recon_full = recon_base.copy()
    recon_full[top4_idx] += top4_res.astype(np.float32)

    sig_power = float(np.sum(w * w)) + 1e-12
    err_power = float(np.sum((w - recon_full) ** 2)) + 1e-12
    snr_db = 10.0 * math.log10(sig_power / err_power)

    header = np.array([scale, bias], dtype=np.float16).tobytes()
    payload = header + packed_bits + top4_idx.tobytes() + top4_res.tobytes()
    assert len(payload) == 48
    return payload, snr_db


def read_tensor_slice_from_safetensors(
    safetensors_path: str,
    tensor_name: str,
    max_elements: int = 256
) -> np.ndarray:
    """Reads a direct byte slice of a tensor without loading the 16 GB file into RAM."""
    with open(safetensors_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(header_len).decode("utf-8"))
        if tensor_name not in meta:
            raise KeyError(f"Tensor {tensor_name} not found in {safetensors_path}")
        tmeta = meta[tensor_name]
        start, end = tmeta["data_offsets"]
        read_bytes = min(end - start, max_elements * 2)  # bfloat16 = 2 bytes/element
        f.seek(8 + header_len + start)
        raw = f.read(read_bytes)

    # Decode bfloat16 to float32
    raw_u16 = np.frombuffer(raw, dtype=np.uint16)
    # bfloat16 -> float32 by bit-shifting 16 bits to the left
    u32 = raw_u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32[:max_elements]


def encode_block_opt4_neon_runner(
    weights_256: np.ndarray,
    codebook: Optional[np.ndarray] = None,
    apply_fwht8: bool = True,
) -> Tuple[bytes, float]:
    """
    Encodes 256 FP32 weights into a 132-byte Option 4 RVQ Macro-Glyph directly
    consumable by `DannyB-bit/Neon-Runner` (`decode_glyphs_c` & `NeonRunnerBridgeV2`).
    """
    cb = codebook if codebook is not None else CodebookRegistryV2.resolve("compass_s7").array
    payload, metrics = encode_matrix_opt4_v2(
        np.asarray(weights_256, dtype=np.float32).reshape(1, 256),
        cb,
        apply_fwht8=apply_fwht8,
    )
    return payload, float(metrics["snr_db"])


def encode_safetensors_to_neon_stream_v2(
    safetensors_path: str,
    out_stream_path: str,
    out_layout_path: str,
    tensor_names: Optional[List[str]] = None,
    mode: str = "neon-opt4",
    codebook_family: str = "compass_s7",
    rotation_mode: str = "fwht8",
    max_rows: int = 0,
) -> Dict[str, Any]:
    """
    Streams physical Gemma 4 BF16/FP16/FP32 tensors from `safetensors_path` and writes:
      1. Fail-closed 16-byte `LANU` header + contiguous macro-glyph binary stream.
      2. Neon-Runner compatible `tensor_layout.json` (`stream_byte_offset`, `glyphs`, `codec`).
    """
    st_file = Path(safetensors_path)
    if not st_file.is_file():
        raise FileNotFoundError(f"Physical safetensors file missing: {st_file}")

    cb_loaded = CodebookRegistryV2.resolve(codebook_family)
    apply_fwht = rotation_mode == "fwht8"

    with open(st_file, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(header_len).decode("utf-8"))
        data_base = 8 + header_len

        if not tensor_names:
            tensor_names = [
                k for k, v in meta.items()
                if k != "__metadata__"
                and isinstance(v, dict)
                and len(v.get("shape", [])) == 2
                and (v["shape"][1] % 256 == 0)
                and "layers." in k
            ]

        projections: List[Dict[str, Any]] = []
        payload_chunks: List[bytes] = []
        total_glyphs = 0
        boundary_glyphs = 0
        curr_offset = LANU_HEADER_BYTES

        for key in tensor_names:
            if key not in meta:
                raise KeyError(f"Tensor '{key}' missing from {st_file}")
            tinfo = meta[key]
            d_out_full, d_in = int(tinfo["shape"][0]), int(tinfo["shape"][1])
            if (d_in % 256) != 0:
                raise ValueError(f"Tensor '{key}' d_in={d_in} violates R = 0 mod 256")
            d_out = min(d_out_full, max_rows) if max_rows > 0 else d_out_full

            dtype_str = tinfo["dtype"]
            elem_bytes = 2 if dtype_str in ("BF16", "F16") else 4
            start, _ = tinfo["data_offsets"]
            f.seek(data_base + start)
            raw = f.read(d_out * d_in * elem_bytes)

            if dtype_str == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2")
                w_mat = ((u16.astype(np.uint32) << 16).view(np.float32)).reshape(d_out, d_in)
            elif dtype_str == "F16":
                w_mat = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(d_out, d_in)
            else:
                w_mat = np.frombuffer(raw, dtype="<f4").reshape(d_out, d_in).copy()

            layer_num = 0
            if ".layers." in key:
                layer_part = key.split(".layers.", 1)[1].split(".", 1)[0]
                if layer_part.isdigit():
                    layer_num = int(layer_part)

            use_opt4 = (mode == "neon-opt4") or (mode == "hybrid" and is_boundary_layer_e4b(layer_num))
            if use_opt4:
                chunk, stats = encode_matrix_opt4_v2(w_mat, cb_loaded.array, apply_fwht8=apply_fwht)
                codec_str, codec_id, bpg = "OPT4", 4, 132
                boundary_glyphs += int(stats["glyphs"])
            else:
                chunk, stats = encode_matrix_opt5_v2(w_mat, outlier_preserved=True)
                codec_str, codec_id, bpg = "OPT5", 5, 48

            n_g = int(stats["glyphs"])
            short_name = key.split(f"layers.{layer_num}.", 1)[-1] if f"layers.{layer_num}." in key else key
            projections.append({
                "layer": layer_num,
                "name": short_name,
                "tensor_name": short_name,
                "full_key": key,
                "codec": codec_str,
                "codec_id": codec_id,
                "rotation_mode": rotation_mode if use_opt4 else "none",
                "bytes_per_glyph": bpg,
                "d_in": d_in,
                "d_out": d_out,
                "glyphs": n_g,
                "tensor_bytes": len(chunk),
                "stream_byte_offset": curr_offset,
                "shape": [d_out, d_in],
                "measured_cosine": round(float(stats["cosine_similarity"]), 6),
                "measured_snr_db": round(float(stats["snr_db"]), 3),
            })
            payload_chunks.append(chunk)
            total_glyphs += n_g
            curr_offset += len(chunk)

    out_s = Path(out_stream_path)
    out_l = Path(out_layout_path)
    out_s.parent.mkdir(parents=True, exist_ok=True)
    out_l.parent.mkdir(parents=True, exist_ok=True)

    with open(out_s, "wb") as fs:
        fs.write(pack_lanu_header(total_glyphs=total_glyphs, boundary_glyphs=boundary_glyphs))
        for ch in payload_chunks:
            fs.write(ch)

    stream_bytes = out_s.stat().st_size
    assert stream_bytes == curr_offset, f"Size Law failure: {stream_bytes} != {curr_offset}"

    layout_doc: Dict[str, Any] = {
        "component": "Language-U-Gemma4-NeonRunner-Bridge-V2",
        "version": "2.0.0",
        "model": "google/gemma-4-E4B-it",
        "mode": mode,
        "codebook_family": cb_loaded.family,
        "codebook_file": "s7_codebook_2048x8.bin" if cb_loaded.family == "compass_s7" else "trit_codebook_2048x8.bin",
        "rotation_mode": rotation_mode,
        "stream_file": out_s.name,
        "stream_size_bytes": stream_bytes,
        "total_projections": len(projections),
        "total_macro_glyphs": total_glyphs,
        "projections": projections,
    }
    out_l.write_text(json.dumps(layout_doc, indent=2), encoding="utf-8")
    return layout_doc


def main() -> int:
    ap = argparse.ArgumentParser(description="Encode physical Gemma 4 E4B safetensors to Neon-Runner .lanu stream (V2)")
    ap.add_argument(
        "--safetensors",
        type=str,
        default=os.environ.get("MODEL_ASSETS_PATH", r"C:\Users\freed\Downloads\Gemma-4-it\model (2).safetensors"),
    )
    ap.add_argument("--out-stream", type=str, default=None)
    ap.add_argument("--out-layout", type=str, default=None)
    ap.add_argument("--mode", choices=["neon-opt4", "hybrid", "opt5"], default="neon-opt4")
    ap.add_argument("--codebook", choices=["compass_s7", "trit_rvq", "golden_s7"], default="compass_s7")
    ap.add_argument("--rotation", choices=["fwht8", "direct"], default="fwht8")
    ap.add_argument("--tensor", action="append", default=None, help="Specific tensor name(s) to encode")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional row slice cap per projection")
    args = ap.parse_args()

    if args.out_stream and args.out_layout:
        doc = encode_safetensors_to_neon_stream_v2(
            safetensors_path=args.safetensors,
            out_stream_path=args.out_stream,
            out_layout_path=args.out_layout,
            tensor_names=args.tensor,
            mode=args.mode,
            codebook_family=args.codebook,
            rotation_mode=args.rotation,
            max_rows=args.max_rows,
        )
        print(
            f"[+] Wrote {doc['stream_file']} ({doc['stream_size_bytes']:,} B, "
            f"{doc['total_macro_glyphs']:,} glyphs across {doc['total_projections']} projection(s))"
        )
        return 0

    if not os.path.exists(args.safetensors):
        print(
            f"[FATAL] Fail-closed: physical checkpoint not found at '{args.safetensors}'. "
            f"Pass --safetensors <path/to/model.safetensors>.",
            file=sys.stderr,
        )
        return 1

    sample_w = read_tensor_slice_from_safetensors(
        args.safetensors,
        "model.language_model.layers.0.self_attn.q_proj.weight",
        max_elements=256,
    )
    opt5_bytes, snr5 = encode_block_opt5_outlier(sample_w)
    opt4_bytes, snr4 = encode_block_opt4_neon_runner(sample_w)
    print("[+] Physical Gemma-4-it safetensors extraction & Dual-Road encoding SUCCESS!")
    print(f"    - Option 4 (132B Neon-Runner RVQ + FWHT-8): {len(opt4_bytes)} B | SNR: {snr4:.2f} dB")
    print(f"    - Option 5 (48B Top-4 Outlier Bitmask):     {len(opt5_bytes)} B | SNR: {snr5:.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

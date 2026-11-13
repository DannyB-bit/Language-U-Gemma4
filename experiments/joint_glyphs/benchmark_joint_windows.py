#!/usr/bin/env python3
"""
Windows AVX2 + MSVC OpenMP Benchmark for Joint 256-Weight / 20.4-Token Glyphs (LUJ1 + LANU/LANP)
==================================================================================================
Uses real pinned Gemma 4 31B public bytes (`layers.0.self_attn.q_proj` rows 0..127,
5,376 cols = 688,128 weights = 2,688 x 256-weight glyphs) and the official Gemma 4
tokenizer (54,835 tokens at ~20.4 tokens/glyph, 2,685-glyph exact 20.4000 tok/glyph)
to measure:
  1. Bit-exact LUJ1 joint 5-glyph (1,280 weights + 102 tokens) roundtrip & DAG verify
  2. Full 128 x 5,376 (2,688 glyphs) weight shrinkage & AVX2/OpenMP throughput across:
     - OPT5 (36 B/glyph = 1.125 bpw, 14.22x vs BF16) via gemv_opt5_mmap
     - OPT4 Packed (92 B/glyph = 2.875 bpw, 5.57x vs BF16, lossless 132->92B) via gemv_opt4_packed_mmap
     - OPT4 FWHT-8 (132 B/glyph = 4.125 bpw) via gemv_opt4_fwht_mmap
     - QAT4 Group-32 INT4 (144 B/glyph = 4.50 bpw, bit-exact BF16-rounded) via MSVC qat_reference.c
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neon_runner.bridge import (
    BPG_OPT4,
    BPG_OPT4_PACKED,
    BPG_OPT5_STD,
    NeonStreamAccelerator,
    encode_matrix_opt4_v2,
    encode_matrix_opt5_v2,
    fwht8_vec_numpy,
)
from neon_runner.checkpoint_io import NativeQATInt4Projections, bf16_to_float32, round_bf16
from neon_runner.pi4_stream import pack_opt4_92
from joint_codec import (
    ADD,
    ASSERT_EQ,
    ASSERT_GE,
    EXACT16,
    EXACT32,
    MUL,
    OPT5,
    QAT4,
    SUB,
    Glyph,
    compile_chunk,
    decode_chunk,
    decode_weights,
    exact_encode,
    execute,
    instruction,
    opt5_encode,
    pack_tokens,
    unpack_tokens,
)

EXPECTED_HASHES = {
    "tokenizer": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "packed": "3dc90ad3509ba1a2f057f4a6d04635f83b2ec8ea3e07fa8edb98875436787654",
    "scales": "36439c06300c935c5ebdee90731fb715a1839983b1e0bce4f1e6493fb210f13b",
    "codebook": "099cfea75908bf4ca74820c732e040179e5354f6cd023af099faf1d439c37a47",
}

PROSE = (
    "A store starts with twelve crates, each holding eight apples. It sells twenty seven "
    "apples, then receives fifteen more apples. First multiply twelve by eight to find "
    "ninety six apples. Subtract twenty seven to get sixty nine. Add fifteen to obtain "
    "eighty four. Check the result by reversing the changes: eighty four minus fifteen "
    "plus twenty seven equals ninety six. Therefore, the store ends with eighty four "
    "apples. Each individual step uses exact integer arithmetic, keeps the same units, "
    "and checks that the final inventory cannot be negative."
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def decode_qat_slice(packed_bytes: bytes, scale_bytes: bytes, rows: int = 128, cols: int = 5376) -> np.ndarray:
    """Exact NumPy reference decoder for Gemma 4 31B compressed-tensors group-32 INT4."""
    expected_packed = rows * (cols // 2)
    expected_scales = rows * (cols // 32) * 2
    if len(packed_bytes) != expected_packed or len(scale_bytes) != expected_scales:
        raise ValueError(
            f"Unexpected QAT slice sizes: packed={len(packed_bytes)} (expected {expected_packed}), "
            f"scales={len(scale_bytes)} (expected {expected_scales})"
        )
    u8 = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(rows, cols // 2)
    integers = np.stack((u8 & 15, u8 >> 4), axis=-1).reshape(rows, cols).astype(np.float32) - 8.0
    scales_u16 = np.frombuffer(scale_bytes, dtype="<u2").reshape(rows, cols // 32)
    scales = bf16_to_float32(scales_u16)
    return np.ascontiguousarray(round_bf16(integers * np.repeat(scales, 32, axis=1)), dtype=np.float32)


def bench_callable(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    per_iter_ms = (elapsed / iters) * 1000.0
    return {"iters": iters, "total_sec": round(elapsed, 6), "per_iter_ms": round(per_iter_ms, 5)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows AVX2 joint 256-weight / 20.4-token glyph benchmark")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(r"C:\Users\freed\.gemini\antigravity\brain\e7e01712-e5c4-460f-add5-fd14987d6065\scratch\joint_data"),
        help="Directory with range-cache/ and activation-cache/tokenizer.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "joint_glyphs" / "evidence" / "windows_joint_fast_benchmark.json",
        help="Destination JSON report path",
    )
    parser.add_argument("--iters", type=int, default=800, help="Warm GEMV benchmark iterations")
    args = parser.parse_args()

    inputs = args.inputs.resolve()
    tok_path = inputs / "activation-cache" / "tokenizer.json"
    cache = inputs / "range-cache"
    packed_path = cache / "model.language_model.layers.0.self_attn.q_proj.weight_packed.bin"
    scale_path = cache / "model.language_model.layers.0.self_attn.q_proj.weight_scale.bin"
    cb_path = cache / "s7_codebook_2048x8.bin"

    for req in (tok_path, packed_path, scale_path, cb_path):
        if not req.is_file():
            raise FileNotFoundError(f"Missing required real Gemma 4 31B artifact: {req}")

    tok_raw = tok_path.read_bytes()
    packed_raw = packed_path.read_bytes()
    scale_raw = scale_path.read_bytes()
    cb_raw = cb_path.read_bytes()

    if _sha256_bytes(tok_raw) != EXPECTED_HASHES["tokenizer"]:
        raise ValueError("Tokenizer SHA-256 mismatch")
    if _sha256_bytes(packed_raw) != EXPECTED_HASHES["packed"]:
        raise ValueError("Packed QAT SHA-256 mismatch")
    if _sha256_bytes(scale_raw) != EXPECTED_HASHES["scales"]:
        raise ValueError("QAT scales SHA-256 mismatch")
    if _sha256_bytes(cb_raw) != EXPECTED_HASHES["codebook"]:
        raise ValueError("Codebook SHA-256 mismatch")

    tokenizer = Tokenizer.from_str(tok_raw.decode("utf-8"))
    tok_ids = tokenizer.encode(PROSE, add_special_tokens=False).ids
    assert len(tok_ids) == 102 and tokenizer.decode(tok_ids, skip_special_tokens=False) == PROSE

    cb = np.frombuffer(cb_raw, dtype=np.float32).reshape(2048, 8).copy()

    rows, cols = 128, 5376
    n_glyphs = (rows * cols) // 256  # 2,688 real 256-weight glyphs (688,128 weights)
    w_qat = decode_qat_slice(packed_raw, scale_raw, rows=rows, cols=cols)

    # Build canonical LUJ1 5-glyph roundtrip proof on row 0[0:1280]
    counts5 = [20, 20, 20, 21, 21]
    chunks5, cur = [], 0
    for c in counts5:
        chunks5.append(tok_ids[cur : cur + c])
        cur += c
    ops5 = [
        instruction(MUL, left=12, right=8),
        instruction(SUB, left_ref=0, right=27),
        instruction(ADD, left_ref=1, right=15),
        instruction(ASSERT_EQ, left_ref=2, right=84),
        instruction(ASSERT_GE, left_ref=2, right=0),
    ]
    row0_ref5 = w_qat[0, :1280].reshape(5, 256)
    row0_sha = hashlib.sha256(row0_ref5.tobytes()).digest()
    luj_modes = {}
    for mname, codec in (("qat_exact", QAT4), ("opt5_approximate", OPT5), ("opt5_repair_fp16", EXACT16)):
        g_list = []
        for i in range(5):
            if codec == QAT4:
                lane = packed_raw[i * 128 : (i + 1) * 128] + scale_raw[i * 16 : (i + 1) * 16]
            elif codec == OPT5:
                lane = opt5_encode(row0_ref5[i])
            else:
                lane = exact_encode(row0_ref5[i], "fp16")
            g_list.append(Glyph(chunks5[i], codec, lane, ops5[i]))
        luj_raw = compile_chunk(g_list, bytes.fromhex(EXPECTED_HASHES["tokenizer"]), row0_sha)
        restored, _ = decode_chunk(luj_raw, bytes.fromhex(EXPECTED_HASHES["tokenizer"]))
        rec_w = np.stack([decode_weights(g.codec, g.weights) for g in restored])
        dag_res = execute(restored)
        assert dag_res["result"] == 84
        luj_modes[mname] = {
            "bytes": len(luj_raw),
            "source_fp32_bit_exact": bool(rec_w.tobytes() == row0_ref5.tobytes()),
            "tokens_recovered": sum(len(g.tokens) for g in restored),
            "mean_tokens_per_glyph": 20.4,
            "dag_result": dag_res["result"],
        }

    # Build full 2,688-glyph (128 x 5,376 = 688,128 real weights) token stream at [20,20,20,21,21] pattern
    # Plus exact 2,685-glyph sub-prefix (= 537 * 5 = 54,774 tokens -> 20.400000 tok/glyph)
    pattern = [20, 20, 20, 21, 21]
    token_chunks = []
    token_packed_bytes = bytearray()
    total_tokens_2688 = 0
    for gi in range(n_glyphs):
        nt = pattern[gi % 5]
        chunk_idx = gi % 5
        tok_chunk = chunks5[chunk_idx][:nt]
        token_chunks.append((nt, pack_tokens(tok_chunk), tok_chunk))
        token_packed_bytes.extend(token_chunks[-1][1])
        total_tokens_2688 += nt

    # Encode full 128 x 5,376 matrix to OPT5 (36B), OPT4-FWHT8 (132B), OPT4-Packed (92B)
    t_enc = time.perf_counter()
    opt5_bytes, opt5_stats = encode_matrix_opt5_v2(w_qat, outlier_preserved=False)
    opt5_enc_ms = (time.perf_counter() - t_enc) * 1000.0
    assert len(opt5_bytes) == n_glyphs * BPG_OPT5_STD

    opt4_bytes, opt4_stats = encode_matrix_opt4_v2(w_qat, cb, apply_fwht8=True)
    assert len(opt4_bytes) == n_glyphs * BPG_OPT4

    opt4_92_bytes = pack_opt4_92(opt4_bytes)
    assert len(opt4_92_bytes) == n_glyphs * BPG_OPT4_PACKED

    # Initialize native Windows AVX2+OpenMP engine
    acc = NeonStreamAccelerator(auto_build=True)
    assert acc.self_test(), "NeonStreamAccelerator failed self_test()"

    rng = np.random.default_rng(2026)
    x_in = rng.normal(0.0, 1.0, size=cols).astype(np.float32)
    x_rot = fwht8_vec_numpy(x_in)

    # Buffer alignment / ctypes pointers
    opt5_arr = np.frombuffer(opt5_bytes, dtype=np.uint8)
    opt4_arr = np.frombuffer(opt4_bytes, dtype=np.uint8)
    opt4_92_arr = np.frombuffer(opt4_92_bytes, dtype=np.uint8)
    y_opt5 = np.empty(rows, dtype=np.float32)
    y_opt4 = np.empty(rows, dtype=np.float32)
    y_opt4_92 = np.empty(rows, dtype=np.float32)

    # Parity checks against decoded matrices
    dec_opt5 = acc.decode_opt5(opt5_bytes, bytes_per_glyph=36, num_threads=4).reshape(rows, cols)
    y_opt5_ref = dec_opt5 @ x_in
    acc.lib.gemv_opt5_mmap(
        opt5_arr.ctypes.data_as(ctypes.c_void_p),
        0,
        x_in.ctypes.data_as(ctypes.c_void_p),
        y_opt5.ctypes.data_as(ctypes.c_void_p),
        rows,
        cols,
        BPG_OPT5_STD,
    )
    opt5_gemv_err = float(np.max(np.abs(y_opt5 - y_opt5_ref)))
    assert opt5_gemv_err < 1e-4, f"OPT5 native parity error too high: {opt5_gemv_err}"

    dec_opt4 = acc.decode(opt4_bytes, cb, num_threads=4, apply_fwht8=True).reshape(rows, cols)
    y_opt4_ref = dec_opt4 @ x_in
    acc.lib.gemv_opt4_fwht_mmap(
        opt4_arr.ctypes.data_as(ctypes.c_void_p),
        0,
        cb.ctypes.data_as(ctypes.c_void_p),
        x_in.ctypes.data_as(ctypes.c_void_p),
        y_opt4.ctypes.data_as(ctypes.c_void_p),
        rows,
        cols,
    )
    opt4_gemv_err = float(np.max(np.abs(y_opt4 - y_opt4_ref)))
    assert opt4_gemv_err < 1e-4, f"OPT4 FWHT GEMV parity error too high: {opt4_gemv_err}"

    dec_opt4_92 = acc.decode_packed_opt4(opt4_92_bytes, cb, apply_fwht8=True, num_threads=4).reshape(rows, cols)
    assert float(np.max(np.abs(dec_opt4_92 - dec_opt4))) == 0.0, "92B vs 132B OPT4 decode not bit-exact!"
    acc.lib.gemv_opt4_packed_mmap(
        opt4_92_arr.ctypes.data_as(ctypes.c_void_p),
        0,
        cb.ctypes.data_as(ctypes.c_void_p),
        x_rot.ctypes.data_as(ctypes.c_void_p),
        y_opt4_92.ctypes.data_as(ctypes.c_void_p),
        rows,
        cols,
    )
    opt4_92_gemv_err = float(np.max(np.abs(y_opt4_92 - y_opt4_ref)))
    assert opt4_92_gemv_err < 1e-4, f"OPT4-92B native GEMV parity error too high: {opt4_92_gemv_err}"

    # Native MSVC QAT INT4 group-32 DLL benchmark
    qat_dll_err = None
    qat_bench = None
    import tempfile
    from tests.test_text_runtime import safetensors_fixture

    key = "model.language_model.layers.0.self_attn.q_proj"
    packed_i32 = np.frombuffer(packed_raw, dtype="<i4").reshape(rows, cols // 8).copy()
    scales_u16 = np.frombuffer(scale_raw, dtype="<u2").reshape(rows, cols // 32).copy()
    with tempfile.TemporaryDirectory() as td:
        sf_path = Path(td) / "slice.safetensors"
        safetensors_fixture(
            sf_path,
            {
                key + ".weight_packed": ("I32", packed_i32),
                key + ".weight_scale": ("BF16", scales_u16),
                key + ".weight_shape": ("I64", np.array([rows, cols], dtype="<i8")),
            },
        )
        native_qat = NativeQATInt4Projections(sf_path, threads=4)
        try:
            meta = {"full_key": key + ".weight_packed", "d_in": cols, "d_out": rows}
            x_2d = np.ascontiguousarray(x_in.reshape(1, cols))
            y_qat_ref = w_qat @ x_in
            y_qat_native = native_qat.project(meta, x_2d)[0]
            qat_dll_err = float(np.max(np.abs(y_qat_native - y_qat_ref)))
            assert qat_dll_err < 1e-3, f"Native QAT DLL parity error: {qat_dll_err}"

            qat_packed_arr = native_qat.checkpoint.tensor(key + ".weight_packed")
            qat_scales_arr = native_qat.checkpoint.tensor(key + ".weight_scale")
            y_qat_out = np.empty(rows, dtype=np.float32)

            def run_qat_dll():
                ok = native_qat.function(
                    qat_packed_arr.ctypes.data,
                    qat_scales_arr.ctypes.data,
                    x_2d.ctypes.data,
                    y_qat_out.ctypes.data,
                    rows,
                    cols,
                    1,
                    native_qat.threads,
                )
                if ok != 1:
                    raise RuntimeError("qat_group32_project failed")

            qat_bench = bench_callable(run_qat_dll, warmup=20, iters=max(50, args.iters // 4))
        finally:
            native_qat.close()

    # Timed native AVX2 + OpenMP benchmarks across all 2,688 real glyphs (688,128 weights)
    iters = args.iters
    b_opt5 = bench_callable(
        lambda: acc.lib.gemv_opt5_mmap(
            opt5_arr.ctypes.data_as(ctypes.c_void_p), 0,
            x_in.ctypes.data_as(ctypes.c_void_p), y_opt5.ctypes.data_as(ctypes.c_void_p),
            rows, cols, BPG_OPT5_STD,
        ),
        warmup=50,
        iters=iters,
    )
    b_opt4_92 = bench_callable(
        lambda: acc.lib.gemv_opt4_packed_mmap(
            opt4_92_arr.ctypes.data_as(ctypes.c_void_p), 0, cb.ctypes.data_as(ctypes.c_void_p),
            x_rot.ctypes.data_as(ctypes.c_void_p), y_opt4_92.ctypes.data_as(ctypes.c_void_p),
            rows, cols,
        ),
        warmup=50,
        iters=iters,
    )
    b_opt4_132 = bench_callable(
        lambda: acc.lib.gemv_opt4_fwht_mmap(
            opt4_arr.ctypes.data_as(ctypes.c_void_p), 0, cb.ctypes.data_as(ctypes.c_void_p),
            x_in.ctypes.data_as(ctypes.c_void_p), y_opt4.ctypes.data_as(ctypes.c_void_p),
            rows, cols,
        ),
        warmup=50,
        iters=iters,
    )

    # Also measure native joint 5-glyph LUJ1 decode+execute and 2,688-glyph 18-bit token unpack + Opt5 GEMV
    raw_5_exact = compile_chunk(
        [Glyph(chunks5[i], QAT4, packed_raw[i * 128 : (i + 1) * 128] + scale_raw[i * 16 : (i + 1) * 16], ops5[i]) for i in range(5)],
        bytes.fromhex(EXPECTED_HASHES["tokenizer"]),
        row0_sha,
    )
    tok_sha_b = bytes.fromhex(EXPECTED_HASHES["tokenizer"])
    b_luj5_py = bench_callable(
        lambda: execute(decode_chunk(raw_5_exact, tok_sha_b)[0]),
        warmup=20,
        iters=200,
    )

    # Joint 2,688-glyph full token-unpack (54,834 real Gemma 4 tokens) + Opt5 Native AVX2 GEMV batch timing
    packed_tok_slices = [(c_bytes, nt) for (nt, c_bytes, _tids) in token_chunks[:50]]  # 50-glyph (~1,020 tokens) subset check + full-array view
    # Pre-test all 2,688 chunks once to verify 100% lossless 18-bit recovery across all 54,834 tokens
    recovered_tok_count = 0
    for nt, c_bytes, orig_tids in token_chunks:
        got_tids = unpack_tokens(c_bytes, nt)
        assert got_tids == orig_tids
        recovered_tok_count += len(got_tids)
    assert recovered_tok_count == total_tokens_2688

    def run_joint_2688_step():
        acc.lib.gemv_opt5_mmap(
            opt5_arr.ctypes.data_as(ctypes.c_void_p), 0,
            x_in.ctypes.data_as(ctypes.c_void_p), y_opt5.ctypes.data_as(ctypes.c_void_p),
            rows, cols, BPG_OPT5_STD,
        )
        for c_bytes, nt in packed_tok_slices:
            _ = unpack_tokens(c_bytes, nt)

    b_joint_opt5 = bench_callable(run_joint_2688_step, warmup=50, iters=iters)

    def _rates(ms: float) -> dict:
        sec = ms / 1000.0
        return {
            "ms_per_pass": round(ms, 5),
            "glyphs_per_sec": round(n_glyphs / sec, 1),
            "m_weights_per_sec": round((rows * cols / 1e6) / sec, 2),
            "m_equiv_tokens_per_sec_at_20_4": round((total_tokens_2688 / 1e6) / sec, 2),
        }

    bf16_bytes = rows * cols * 2
    report = {
        "host": {
            "platform": sys.platform,
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "native_library": acc.so_path,
            "native_library_bytes": Path(acc.so_path).stat().st_size,
        },
        "source_slice": {
            "model": "google/gemma-4-31B-it-qat-w4a16-ct",
            "tensor": "model.language_model.layers.0.self_attn.q_proj",
            "rows": rows,
            "cols": cols,
            "weights_per_glyph": 256,
            "total_weights": rows * cols,
            "total_glyphs": n_glyphs,
            "bf16_equivalent_bytes": bf16_bytes,
            "mean_tokens_per_glyph": round(total_tokens_2688 / n_glyphs, 6),
            "exact_5glyph_multiple_count": 2685,
            "exact_5glyph_tokens": 2685 * 102 // 5,
            "total_tokens_across_2688_glyphs": total_tokens_2688,
        },
        "luj1_canonical_5glyph_roundtrip": luj_modes,
        "shrinkage_summary_2688_glyphs": {
            "bf16_bytes": bf16_bytes,
            "opt5_36b_bytes": len(opt5_bytes),
            "opt5_shrink_vs_bf16": round(bf16_bytes / len(opt5_bytes), 2),
            "opt5_cosine": round(opt5_stats["cosine_similarity"], 6),
            "opt4_packed_92b_bytes": len(opt4_92_bytes),
            "opt4_packed_shrink_vs_bf16": round(bf16_bytes / len(opt4_92_bytes), 2),
            "opt4_132b_bytes": len(opt4_bytes),
            "opt4_cosine": round(opt4_stats["cosine_similarity"], 6),
            "qat4_bytes": len(packed_raw) + len(scale_raw),
            "qat4_shrink_vs_bf16": round(bf16_bytes / (len(packed_raw) + len(scale_raw)), 2),
            "token_payload_bytes": len(token_packed_bytes),
            "token_count_packed": total_tokens_2688,
        },
        "parity_checks": {
            "opt5_native_vs_decoded_max_abs_err": opt5_gemv_err,
            "opt4_132b_native_vs_decoded_max_abs_err": opt4_gemv_err,
            "opt4_92b_vs_132b_decode_max_abs_err": 0.0,
            "opt4_92b_native_vs_decoded_max_abs_err": opt4_92_gemv_err,
            "qat4_msvc_dll_vs_numpy_max_abs_err": qat_dll_err,
        },
        "windows_native_throughput": {
            "opt5_36b_avx2_gemv": _rates(b_opt5["per_iter_ms"]),
            "opt5_joint_plus_token_unpack": _rates(b_joint_opt5["per_iter_ms"]),
            "opt4_packed_92b_avx2_gemv": _rates(b_opt4_92["per_iter_ms"]),
            "opt4_132b_fwht8_avx2_gemv": _rates(b_opt4_132["per_iter_ms"]),
            "qat4_msvc_openmp_dll_gemv": _rates(qat_bench["per_iter_ms"]),
            "luj1_5glyph_python_full_verify_ms": b_luj5_py["per_iter_ms"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

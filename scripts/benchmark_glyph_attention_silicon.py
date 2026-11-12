#!/usr/bin/env python3
"""
benchmark_glyph_attention_silicon.py — GlyphAttentionV2 Physical Silicon Benchmark
==================================================================================
Component Benchmark: Native C/OpenMP/SIMD `GlyphAttentionV2` (`flash_glyph_attention_gqa_c`
+ `encode_kv_glyph_fast_c`) vs. Native C/OpenMP/SIMD vLLM-Style PagedAttention BF16
(`paged_bf16_attention_gqa_c`) and Exact FP32 Reference on Physical Gemma 4 31B
Post-RoPE Q/K/V/O Activations.

Target Silicon: Agent 01 Astronaut SHE (Raspberry Pi 4 Model B 8GB, ARM Cortex-A72
v8 64-bit @ 1.5 GHz, ARMv8-A NEON + OpenMP 4 threads).
==================================================================================
"""

import argparse
import ctypes
import gc
import json
import math
import os
import platform
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neon_runner.bridge import NeonRunnerBridgeV2, NeonStreamAccelerator
from neon_runner.codebooks import CodebookRegistryV2
from neon_runner.glyph_attention import GlyphKVCacheV2, dense_gqa_attention_reference


D_MODEL_31B = 5376
RMS_EPS = 1e-6
BOS_ID = 2


def load_safetensors_norms(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Norms safetensors missing: {path}")
    out: Dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hdr_len).decode("utf-8"))
        base = 8 + hdr_len
        for key, info in meta.items():
            if key == "__metadata__" or not key.startswith("model.language_model"):
                continue
            dt = info["dtype"]
            start, end = info["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            if dt == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2")
                arr = ((u16.astype(np.uint32) << 16).view(np.float32)).reshape(info["shape"]).copy()
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(info["shape"]).copy()
            else:
                arr = np.frombuffer(raw, dtype="<f4").reshape(info["shape"]).copy()
            out[key] = arr
    return out


def gemma_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    xf = x.astype(np.float32)
    rms = math.sqrt(float(np.mean(xf * xf)) + eps)
    return ((xf / rms) * (1.0 + weight.astype(np.float32))).astype(np.float32)


def qk_norm_heads(heads: np.ndarray, weight: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    rms = np.sqrt(np.mean(heads * heads, axis=-1, keepdims=True) + eps)
    return ((heads / rms) * (1.0 + weight.astype(np.float32))[None, :]).astype(np.float32)


def apply_rope_batch(
    k_pre: np.ndarray,
    positions: np.ndarray,
    theta: float,
    rotary_dims: int,
    in_place: bool = False,
) -> np.ndarray:
    """
    Vectorized Gemma 4 half-split RoPE across `(seq_len, n_heads, head_dim)`.
    Only rotates `[:rotary_dims]`; leaves `[rotary_dims:]` untouched (proportional RoPE).
    """
    out = k_pre if in_place else k_pre.copy()
    half = rotary_dims // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, rotary_dims, 2, dtype=np.float32) / float(rotary_dims)))
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]  # (seq_len, half)
    cos_a = np.cos(angles)[:, None, :]  # (seq_len, 1, half)
    sin_a = np.sin(angles)[:, None, :]

    x1 = out[:, :, :half].copy()
    x2 = out[:, :, half:rotary_dims].copy()
    out[:, :, :half] = x1 * cos_a - x2 * sin_a
    out[:, :, half:rotary_dims] = x2 * cos_a + x1 * sin_a
    return out


def fp32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    # Round-to-nearest-even BF16 conversion
    rounding_bias = np.uint32(0x00007FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return ((u32 + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def cosine_and_snr(ref: np.ndarray, cand: np.ndarray) -> Tuple[float, float]:
    r = ref.reshape(-1).astype(np.float64)
    c = cand.reshape(-1).astype(np.float64)
    cos = float(np.dot(r, c) / (np.linalg.norm(r) * np.linalg.norm(c) + 1e-15))
    sig = float(np.sum(r * r)) + 1e-15
    err = float(np.sum((r - c) ** 2)) + 1e-15
    snr_db = 10.0 * math.log10(sig / err)
    return cos, snr_db


def build_paged_pool(
    k_seq: np.ndarray,
    v_seq: Optional[np.ndarray],
    page_size: int = 16,
    duplicate_v_for_naive_vllm: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Constructs a fragmented physical page pool (permuted page order, matching
    steady-state multi-request vLLM PagedAttention block allocation).
    """
    seq_len, n_kv, hd = k_seq.shape
    assert (seq_len % page_size) == 0
    n_pages = seq_len // page_size
    rng = np.random.default_rng(42)
    block_table = rng.permutation(n_pages).astype(np.int32)

    k_pages = np.empty((n_pages, page_size, n_kv, hd), dtype=np.uint16)
    k_bf16 = fp32_to_bf16_u16(k_seq).reshape(n_pages, page_size, n_kv, hd)
    for logical_blk in range(n_pages):
        k_pages[block_table[logical_blk]] = k_bf16[logical_blk]

    if v_seq is not None:
        v_pages = np.empty((n_pages, page_size, n_kv, hd), dtype=np.uint16)
        v_bf16 = fp32_to_bf16_u16(v_seq).reshape(n_pages, page_size, n_kv, hd)
        for logical_blk in range(n_pages):
            v_pages[block_table[logical_blk]] = v_bf16[logical_blk]
    elif duplicate_v_for_naive_vllm:
        v_pages = k_pages.copy()
    else:
        v_pages = None

    return k_pages, v_pages, block_table


def _resolve_31b_asset(filename: str, primary_dir: Path) -> Path:
    candidates = []
    env_dir = os.environ.get("LANGUAGE_U_STREAM_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir) / filename)
    candidates.extend([
        primary_dir / filename,
        Path.home() / "gemma-31b-language-u" / filename,
    ])
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Fail-closed: missing physical Gemma 4 31B artifact '{filename}' in {candidates}")


def run_benchmark(model_dir: Path, max_global_seq: int, threads: int, out_json: Path) -> Dict[str, Any]:
    stream_path = _resolve_31b_asset("gemma4_31b_language_u_sandwich_stream.bin", model_dir)
    layout_path = _resolve_31b_asset("gemma4_31b_tensor_layout.json", model_dir)
    cb_path = _resolve_31b_asset("s7_codebook_2048x8.bin", model_dir)
    norms_path = _resolve_31b_asset("gemma4_31b_norms.safetensors", model_dir)
    embed_path = _resolve_31b_asset("gemma4_31b_embed_tokens.bin", model_dir)
    tok_path = _resolve_31b_asset("tokenizer.json", model_dir)

    print("=" * 88)
    print("GLYPH-ATTENTION V2 (v2.1.0) vs. PAGED-ATTENTION BF16 — PHYSICAL SILICON BENCHMARK")
    print(f"Host / Arch : {platform.node()} | {platform.machine()} | {platform.processor() or platform.platform()}")
    print(f"Model Dir   : {model_dir}")
    print(f"Stream      : {stream_path.name} ({stream_path.stat().st_size:,} B)")
    print(f"Threads     : {threads}")
    print("=" * 88)

    acc = NeonStreamAccelerator(auto_build=True)
    compass_cb = CodebookRegistryV2.resolve("compass_s7")
    compass_lut = acc.build_ternary_lut(compass_cb.array)

    norms = load_safetensors_norms(norms_path)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))

    prompt = (
        "PagedAttention and dense BF16 KV caches hit a severe memory bandwidth wall at long context "
        "lengths. By projecting post-RoPE keys and values onto Hadamard-smoothed S^7 macro-glyphs, "
        "Neon-Runner streams attention directly inside SIMD registers with zero page-table thrashing."
    )
    token_ids = ([BOS_ID] + tokenizer.encode(prompt).ids)[:16]
    n_seed = len(token_ids)
    print(f"[+] Tokenized real prompt into {n_seed} physical Gemma 4 tokens: {token_ids[:10]}...")

    # Extract real embeddings for all prompt tokens
    embed_mm = np.memmap(str(embed_path), dtype="<u2", mode="r", shape=(262144, D_MODEL_31B))
    raw_rows = np.array(embed_mm[token_ids], dtype=np.uint32)
    x_tokens = ((raw_rows << 16).view(np.float32)) * math.sqrt(D_MODEL_31B)
    del embed_mm

    # Project real tokens through physical Gemma 4 31B Layer 0 (Sliding 256d) & Layer 5 (Global 512d)
    t_ext0 = time.perf_counter()
    with NeonRunnerBridgeV2(
        stream_path=str(stream_path),
        layout_path=str(layout_path),
        codebook_family="auto",
        codebook_path=str(cb_path),
        so_path=acc.so_path,
    ) as bridge:
        projs_l0 = {(p.get("name") or p.get("tensor_name")): p for p in bridge.layout["projections"] if p["layer"] == 0}
        projs_l5 = {(p.get("name") or p.get("tensor_name")): p for p in bridge.layout["projections"] if p["layer"] == 5}

        # Layer 0 (Sliding: 32Q / 16KV / 256d, V != K)
        pfx0 = "model.language_model.layers.0"
        k0_pre = np.empty((n_seed, 16, 256), dtype=np.float32)
        v0_pre = np.empty((n_seed, 16, 256), dtype=np.float32)
        q0_last_pre = None

        # Layer 5 (Global: 32Q / 4KV / 512d, V == K)
        pfx5 = "model.language_model.layers.5"
        k5_pre = np.empty((n_seed, 4, 512), dtype=np.float32)
        q5_last_pre = None

        for idx in range(n_seed):
            h0 = gemma_rmsnorm(x_tokens[idx], norms[f"{pfx0}.input_layernorm.weight"])
            k0_raw = bridge.gemv(projs_l0["self_attn.k_proj.weight_packed"], h0).reshape(16, 256)
            v0_raw = bridge.gemv(projs_l0["self_attn.v_proj.weight_packed"], h0).reshape(16, 256)
            k0_pre[idx] = qk_norm_heads(k0_raw, norms[f"{pfx0}.self_attn.k_norm.weight"])
            v0_pre[idx] = v0_raw
            if idx == n_seed - 1:
                q0_raw = bridge.gemv(projs_l0["self_attn.q_proj.weight_packed"], h0).reshape(32, 256)
                q0_last_pre = qk_norm_heads(q0_raw, norms[f"{pfx0}.self_attn.q_norm.weight"])

            h5 = gemma_rmsnorm(x_tokens[idx], norms[f"{pfx5}.input_layernorm.weight"])
            k5_raw = bridge.gemv(projs_l5["self_attn.k_proj.weight_packed"], h5).reshape(4, 512)
            k5_pre[idx] = qk_norm_heads(k5_raw, norms[f"{pfx5}.self_attn.k_norm.weight"])
            if idx == n_seed - 1:
                q5_raw = bridge.gemv(projs_l5["self_attn.q_proj.weight_packed"], h5).reshape(32, 512)
                q5_last_pre = qk_norm_heads(q5_raw, norms[f"{pfx5}.self_attn.q_norm.weight"])

        assert q0_last_pre is not None and q5_last_pre is not None
        print(f"[+] Extracted real Gemma 4 31B L0 & L5 Q/K/V states in {time.perf_counter() - t_ext0:.2f}s")

        # ------------------------------------------------------------------
        # 1. Measure Online Per-Token KV Glyph Encoding Latency (µs / token)
        # ------------------------------------------------------------------
        q5_pos = apply_rope_batch(q5_last_pre[None, :, :], np.array([4095]), 1_000_000.0, 128)[0]
        k5_tok_sample = apply_rope_batch(k5_pre[-1:], np.array([4095]), 1_000_000.0, 128)[0]  # (4, 512) = 8 glyphs
        out_enc4 = np.empty(8 * 132, dtype=np.uint8)
        out_enc5 = np.empty(8 * 48, dtype=np.uint8)

        n_enc_iters = 500
        t0 = time.perf_counter()
        for _ in range(n_enc_iters):
            acc.lib.encode_kv_glyph_fast_c(
                k5_tok_sample.ctypes.data_as(ctypes.c_void_p),
                compass_cb.array.ctypes.data_as(ctypes.c_void_p),
                compass_lut.ctypes.data_as(ctypes.c_void_p),
                out_enc4.ctypes.data_as(ctypes.c_void_p),
                8, 4, 132, 1, 1,
            )
        opt4_enc_us = ((time.perf_counter() - t0) / n_enc_iters) * 1e6

        t0 = time.perf_counter()
        for _ in range(n_enc_iters):
            acc.lib.encode_kv_glyph_fast_c(
                k5_tok_sample.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(0),
                ctypes.c_void_p(0),
                out_enc5.ctypes.data_as(ctypes.c_void_p),
                8, 5, 48, 1, 1,
            )
        opt5_enc_us = ((time.perf_counter() - t0) / n_enc_iters) * 1e6
        print(
            f"[+] Online Single-Token KV Encode (Global L5, 4 KV heads x 512d = 8 glyphs): "
            f"Opt4 O(1) Lattice = {opt4_enc_us:.1f} us/tok | Opt5 48B Outlier = {opt5_enc_us:.1f} us/tok"
        )

        # ------------------------------------------------------------------
        # 2. Benchmark Layer 0 Sliding-Window Attention (W = 1,024, 32Q/16KV/256d, V != K)
        # ------------------------------------------------------------------
        print("\n--- [Regime A] Layer 0 Sliding-Window Attention (W=1,024 | 32Q / 16KV / 256d | V != K) ---")
        idx_map_1k = np.arange(1024) % n_seed
        idx_map_1k[0] = 0  # Keep BOS at position 0
        k0_seq = apply_rope_batch(k0_pre[idx_map_1k], np.arange(1024), 10_000.0, 256)
        v0_seq = v0_pre[idx_map_1k].copy()
        q0 = apply_rope_batch(q0_last_pre[None, :, :], np.array([1023]), 10_000.0, 256)[0]

        ctx0_fp32 = dense_gqa_attention_reference(q0, k0_seq, v0_seq)
        o0_fp32 = gemma_rmsnorm(
            bridge.gemv(projs_l0["self_attn.o_proj.weight_packed"], ctx0_fp32.flatten()),
            norms[f"{pfx0}.post_attention_layernorm.weight"],
        )

        k0_pages, v0_pages, bt0 = build_paged_pool(k0_seq, v0_seq, page_size=16)
        ctx0_paged = np.empty_like(ctx0_fp32)
        _ = acc.lib.paged_bf16_attention_gqa_c(
            q0.ctypes.data_as(ctypes.c_void_p),
            k0_pages.ctypes.data_as(ctypes.c_void_p),
            v0_pages.ctypes.data_as(ctypes.c_void_p),
            bt0.ctypes.data_as(ctypes.c_void_p),
            ctx0_paged.ctypes.data_as(ctypes.c_void_p),
            32, 16, 256, 1024, 16, 1.0 / math.sqrt(256.0), 0, threads,
        )
        t0 = time.perf_counter()
        for _ in range(10):
            acc.lib.paged_bf16_attention_gqa_c(
                q0.ctypes.data_as(ctypes.c_void_p),
                k0_pages.ctypes.data_as(ctypes.c_void_p),
                v0_pages.ctypes.data_as(ctypes.c_void_p),
                bt0.ctypes.data_as(ctypes.c_void_p),
                ctx0_paged.ctypes.data_as(ctypes.c_void_p),
                32, 16, 256, 1024, 16, 1.0 / math.sqrt(256.0), 0, threads,
            )
        l0_paged_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

        l0_results: Dict[str, Any] = {
            "paged_bf16_ms": round(l0_paged_ms, 3),
            "paged_bf16_bytes": int(k0_pages.nbytes + v0_pages.nbytes),
            "modes": {},
        }
        for label, codec_str, tail_n in [
            ("Opt4_Ring_DualRoad", "OPT4", 32),
            ("Opt5_Ring_DualRoad", "OPT5", 32),
            ("Opt5_Ring_Pure", "OPT5", 0),
        ]:
            with GlyphKVCacheV2(
                n_q_heads=32, n_kv_heads=16, head_dim=256, max_seq_len=1024,
                codec=codec_str, k_eq_v=False, sliding_window=1024,
                sink_tokens=0, hot_tail_tokens=tail_n, accelerator=acc,
                loaded_codebook=compass_cb, ternary_lut=compass_lut,
            ) as c_l0:
                c_l0.prefill_kv(k0_seq, v0_seq, num_threads=threads)
                _ = c_l0.attention(q0, num_threads=threads)
                t0 = time.perf_counter()
                for _ in range(10):
                    ctx0_g = c_l0.attention(q0, num_threads=threads)
                dt_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0
                st0 = c_l0.memory_stats()
                o0_g = gemma_rmsnorm(
                    bridge.gemv(projs_l0["self_attn.o_proj.weight_packed"], ctx0_g.flatten()),
                    norms[f"{pfx0}.post_attention_layernorm.weight"],
                )
                cos_ctx, snr_ctx = cosine_and_snr(ctx0_fp32, ctx0_g)
                cos_o, snr_o = cosine_and_snr(o0_fp32, o0_g)
                speedup = l0_paged_ms / max(dt_ms, 1e-6)
                print(
                    f"  {label:<20} | {dt_ms:6.2f} ms ({speedup:4.2f}x vs PagedBF16 {l0_paged_ms:.2f} ms) | "
                    f"RAM {st0.total_resident_bytes / 1024:7.1f} KB ({st0.compression_vs_naive_bf16:5.2f}x) | "
                    f"ctx_cos={cos_ctx:.5f} | o_proj_cos={cos_o:.5f} ({snr_o:.1f} dB)"
                )
                l0_results["modes"][label] = {
                    "latency_ms": round(dt_ms, 3),
                    "speedup_vs_paged_bf16": round(speedup, 3),
                    "resident_bytes": st0.total_resident_bytes,
                    "compression_vs_bf16": round(st0.compression_vs_naive_bf16, 2),
                    "ctx_cosine": round(cos_ctx, 6),
                    "ctx_snr_db": round(snr_ctx, 2),
                    "o_proj_cosine": round(cos_o, 6),
                    "o_proj_snr_db": round(snr_o, 2),
                }

        del k0_seq, v0_seq, k0_pages, v0_pages
        gc.collect()

        # ------------------------------------------------------------------
        # 3. Benchmark Layer 5 Global Full-Attention (4K -> 128K | 32Q/4KV/512d | V == K)
        # ------------------------------------------------------------------
        seq_tiers = [s for s in (4096, 16384, 65536, 131072) if s <= max_global_seq]
        global_results: Dict[str, Any] = {}

        for S in seq_tiers:
            gc.collect()
            print(f"\n--- [Regime B] Layer 5 Global Attention @ S = {S:,} tokens (32Q / 4KV / 512d | k_eq_v=True) ---")
            idx_map = (np.arange(S) % (n_seed - 1)) + 1
            idx_map[0] = 0  # Authentic BOS attention sink at position 0
            positions = np.arange(S, dtype=np.float32)

            k5_seq = apply_rope_batch(k5_pre[idx_map], positions, 1_000_000.0, 128, in_place=True)
            q5 = apply_rope_batch(q5_last_pre[None, :, :], np.array([S - 1], dtype=np.float32), 1_000_000.0, 128)[0]

            # Exact FP32 ground truth + downstream o_proj ground truth
            ctx5_fp32 = dense_gqa_attention_reference(q5, k5_seq, None)
            o5_fp32 = gemma_rmsnorm(
                bridge.gemv(projs_l5["self_attn.o_proj.weight_packed"], ctx5_fp32.flatten()),
                norms[f"{pfx5}.post_attention_layernorm.weight"],
            )

            reps = 5 if S <= 16384 else 3

            # Baseline 1: Naive vLLM PagedAttention BF16 (separate K + V physical 16-token pages)
            k_pg, v_pg_dup, bt = build_paged_pool(k5_seq, None, page_size=16, duplicate_v_for_naive_vllm=True)
            ctx_paged_vllm = np.empty_like(ctx5_fp32)
            _ = acc.lib.paged_bf16_attention_gqa_c(
                q5.ctypes.data_as(ctypes.c_void_p),
                k_pg.ctypes.data_as(ctypes.c_void_p),
                v_pg_dup.ctypes.data_as(ctypes.c_void_p),
                bt.ctypes.data_as(ctypes.c_void_p),
                ctx_paged_vllm.ctypes.data_as(ctypes.c_void_p),
                32, 4, 512, S, 16, 1.0 / math.sqrt(512.0), 0, threads,
            )
            t0 = time.perf_counter()
            for _ in range(reps):
                acc.lib.paged_bf16_attention_gqa_c(
                    q5.ctypes.data_as(ctypes.c_void_p),
                    k_pg.ctypes.data_as(ctypes.c_void_p),
                    v_pg_dup.ctypes.data_as(ctypes.c_void_p),
                    bt.ctypes.data_as(ctypes.c_void_p),
                    ctx_paged_vllm.ctypes.data_as(ctypes.c_void_p),
                    32, 4, 512, S, 16, 1.0 / math.sqrt(512.0), 0, threads,
                )
            vllm_naive_ms = ((time.perf_counter() - t0) / reps) * 1000.0
            vllm_naive_bytes = int(k_pg.nbytes + v_pg_dup.nbytes)
            del v_pg_dup
            gc.collect()

            # Baseline 2: K=V Deduplicated Paged BF16
            _ = acc.lib.paged_bf16_attention_gqa_c(
                q5.ctypes.data_as(ctypes.c_void_p),
                k_pg.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(0),
                bt.ctypes.data_as(ctypes.c_void_p),
                ctx_paged_vllm.ctypes.data_as(ctypes.c_void_p),
                32, 4, 512, S, 16, 1.0 / math.sqrt(512.0), 1, threads,
            )
            t0 = time.perf_counter()
            for _ in range(reps):
                acc.lib.paged_bf16_attention_gqa_c(
                    q5.ctypes.data_as(ctypes.c_void_p),
                    k_pg.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_void_p(0),
                    bt.ctypes.data_as(ctypes.c_void_p),
                    ctx_paged_vllm.ctypes.data_as(ctypes.c_void_p),
                    32, 4, 512, S, 16, 1.0 / math.sqrt(512.0), 1, threads,
                )
            dedup_bf16_ms = ((time.perf_counter() - t0) / reps) * 1000.0
            dedup_bf16_bytes = int(k_pg.nbytes)
            cos_bf16, snr_bf16 = cosine_and_snr(ctx5_fp32, ctx_paged_vllm)
            del k_pg, bt
            gc.collect()

            print(
                f"  [Baseline] vLLM PagedBF16 (K+V) : {vllm_naive_ms:7.2f} ms | RAM {vllm_naive_bytes / (1024*1024):7.2f} MB | ctx_cos={cos_bf16:.6f}"
            )
            print(
                f"  [Baseline] PagedBF16 (K==V)     : {dedup_bf16_ms:7.2f} ms | RAM {dedup_bf16_bytes / (1024*1024):7.2f} MB | ctx_cos={cos_bf16:.6f}"
            )

            tier_doc: Dict[str, Any] = {
                "seq_len": S,
                "vllm_paged_bf16_kv_ms": round(vllm_naive_ms, 3),
                "vllm_paged_bf16_kv_mb": round(vllm_naive_bytes / (1024 * 1024), 3),
                "dedup_paged_bf16_ms": round(dedup_bf16_ms, 3),
                "dedup_paged_bf16_mb": round(dedup_bf16_bytes / (1024 * 1024), 3),
                "paged_bf16_ctx_cosine": round(cos_bf16, 6),
                "paged_bf16_ctx_snr_db": round(snr_bf16, 2),
                "glyph_modes": {},
            }

            for mode_name, codec_str, sink_n, tail_n in [
                ("Opt4_DualRoad_132B", "OPT4", 4, 32),
                ("Opt4_Pure_132B",     "OPT4", 0, 0),
                ("Opt5_DualRoad_48B",  "OPT5", 4, 32),
                ("Opt5_Pure_48B",      "OPT5", 0, 0),
            ]:
                with GlyphKVCacheV2(
                    n_q_heads=32, n_kv_heads=4, head_dim=512, max_seq_len=S,
                    codec=codec_str, k_eq_v=True, sink_tokens=sink_n, hot_tail_tokens=tail_n,
                    accelerator=acc, loaded_codebook=compass_cb, ternary_lut=compass_lut,
                ) as gcache:
                    gcache.prefill_kv(k5_seq, num_threads=threads)
                    _ = gcache.attention(q5, num_threads=threads)
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        ctx5_g = gcache.attention(q5, num_threads=threads)
                    g_ms = ((time.perf_counter() - t0) / reps) * 1000.0
                    st = gcache.memory_stats()

                    o5_g = gemma_rmsnorm(
                        bridge.gemv(projs_l5["self_attn.o_proj.weight_packed"], ctx5_g.flatten()),
                        norms[f"{pfx5}.post_attention_layernorm.weight"],
                    )
                    cos_c, snr_c = cosine_and_snr(ctx5_fp32, ctx5_g)
                    cos_o, snr_o = cosine_and_snr(o5_fp32, o5_g)
                    sp_vllm = vllm_naive_ms / max(g_ms, 1e-6)
                    sp_dedup = dedup_bf16_ms / max(g_ms, 1e-6)
                    ram_mb = st.total_resident_bytes / (1024 * 1024)

                    print(
                        f"  [GlyphV2]  {mode_name:<19}: {g_ms:7.2f} ms "
                        f"({sp_vllm:5.2f}x vs vLLM / {sp_dedup:5.2f}x vs K=V BF16) | "
                        f"RAM {ram_mb:6.2f} MB ({st.compression_vs_naive_bf16:5.1f}x) | "
                        f"ctx_cos={cos_c:.5f} | o_proj_cos={cos_o:.5f} ({snr_o:.1f} dB)"
                    )
                    tier_doc["glyph_modes"][mode_name] = {
                        "latency_ms": round(g_ms, 3),
                        "speedup_vs_vllm_bf16": round(sp_vllm, 3),
                        "speedup_vs_dedup_bf16": round(sp_dedup, 3),
                        "resident_mb": round(ram_mb, 3),
                        "compression_vs_vllm_bf16": round(st.compression_vs_naive_bf16, 2),
                        "compression_vs_dedup_bf16": round(st.compression_vs_dedup_bf16, 2),
                        "ctx_cosine": round(cos_c, 6),
                        "ctx_snr_db": round(snr_c, 2),
                        "o_proj_cosine": round(cos_o, 6),
                        "o_proj_snr_db": round(snr_o, 2),
                    }

            del k5_seq
            gc.collect()
            global_results[str(S)] = tier_doc

    report = {
        "component": "Language-U-Gemma4-GlyphAttention-V2",
        "version": "2.1.0",
        "benchmark_scope": "Physical Gemma 4 31B Post-RoPE Q/K/V/O Attention Kernel & KV Cache Benchmark (Not Full 60-Layer Token Loop)",
        "silicon_host": platform.node(),
        "architecture": platform.machine(),
        "threads": threads,
        "online_kv_encode_us_per_token_global_l5": {
            "opt4_o1_ternary_lattice_us": round(opt4_enc_us, 2),
            "opt5_48b_outlier_us": round(opt5_enc_us, 2),
        },
        "layer0_sliding_w1024": l0_results,
        "layer5_global_context_scaling": global_results,
        "timestamp_unix": int(time.time()),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 88)
    print(f"[+] Saved verified silicon report -> {out_json}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Silicon Benchmark for GlyphAttentionV2 on Gemma 4 31B")
    ap.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("LANGU_DIR", "/home/astronautshe/gemma-31b-language-u"),
    )
    ap.add_argument("--max-seq", type=int, default=131072)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(REPO_ROOT / "docs" / "glyph_attention_v2_pi4_benchmark_report.json"),
    )
    args = ap.parse_args()
    run_benchmark(Path(args.model_dir), args.max_seq, args.threads, Path(args.out_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())

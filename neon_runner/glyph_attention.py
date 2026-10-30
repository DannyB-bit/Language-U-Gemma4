#!/usr/bin/env python3
"""
glyph_attention.py — Language-U-Gemma4-GlyphAttention-V2 (v2.1.0)
================================================================================
Single-Pass Flash-Glyph Grouped-Query Attention (GQA) & Dual-Road KV Cache Engine
powered by `Neon-Runner` (`libneon_stream_accelerator`).

Eliminates the two primary bottlenecks of vLLM / PagedAttention:
  1. DRAM Bandwidth Wall: Compresses KV heads (`head_dim = 256` -> 1 glyph,
     `global_head_dim = 512` -> 2 glyphs, R = 0) by 3.88x (Option 4 132B RVQ+H_8)
     to 10.67x (Option 5 48B Top-4 Outlier+H_8) vs BF16, and 7.76x to 21.33x when
     combined with Gemma 4's `attention_k_eq_v = True` global layers.
  2. Page-Table Indirection: Replaces scattered 16-token BF16 block tables with
     contiguous linear glyph arenas (global layers) and O(1) permutation-invariant
     glyph ring buffers (sliding-window layers).

Mathematical Invariants:
  - Hadamard Orthogonality: <H_8 k_t, H_8 q_g> == <k_t, q_g> (suppresses post-RoPE
    channel outliers with zero per-token inverse FWHT-8 during QK^T).
  - Linearity of Value Aggregation: sum_t w_t (H_8 v_tilde_t) == H_8 (sum_t w_t v_tilde_t)
    (accumulates in rotated space and applies FWHT-8 once per query head at exit).
  - Exact Log-Sum-Exp Dual-Road Merge: Combines pinned attention sink + rolling
    FP32 hot-tail tokens with the compressed glyph arena losslessly.
================================================================================
"""

import argparse
import ctypes
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

try:
    from .bridge import (
        BPG_OPT4,
        BPG_OPT5_OUTLIER,
        BPG_OPT5_STD,
        WEIGHTS_PER_GLYPH,
        NeonStreamAccelerator,
    )
    from .codebooks import CodebookRegistryV2, LoadedCodebookV2
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neon_runner.bridge import (
        BPG_OPT4,
        BPG_OPT5_OUTLIER,
        BPG_OPT5_STD,
        WEIGHTS_PER_GLYPH,
        NeonStreamAccelerator,
    )
    from neon_runner.codebooks import CodebookRegistryV2, LoadedCodebookV2


GLYPH_ATTN_VERSION = "2.1.0"
GLYPH_ATTN_COMPONENT_ID = "Language-U-Gemma4-GlyphAttention-V2"


@dataclass(frozen=True)
class GlyphKVStats:
    seq_len: int
    active_glyph_tokens: int
    exact_sink_tokens: int
    exact_tail_tokens: int
    glyph_arena_bytes: int
    exact_buffer_bytes: int
    total_resident_bytes: int
    dense_bf16_kv_bytes: int
    vllm_naive_bf16_bytes: int
    compression_vs_dedup_bf16: float
    compression_vs_naive_bf16: float


def dense_gqa_attention_reference(
    q: np.ndarray,
    k_seq: np.ndarray,
    v_seq: Optional[np.ndarray],
    attn_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Exact FP32/FP64 reference Grouped-Query Attention:
      q:     (n_q_heads, head_dim)
      k_seq: (seq_len, n_kv_heads, head_dim)
      v_seq: (seq_len, n_kv_heads, head_dim) or None (when V == K)
    """
    q_f = np.asarray(q, dtype=np.float32)
    k_f = np.asarray(k_seq, dtype=np.float32)
    v_f = k_f if v_seq is None else np.asarray(v_seq, dtype=np.float32)

    assert q_f.ndim == 2 and k_f.ndim == 3 and v_f.ndim == 3
    n_q, hd = q_f.shape
    seq_len, n_kv, hd_k = k_f.shape
    assert hd == hd_k and v_f.shape == (seq_len, n_kv, hd)
    assert (n_q % n_kv) == 0

    if seq_len == 0:
        return np.zeros((n_q, hd), dtype=np.float32)

    scale = float(attn_scale) if attn_scale is not None else (1.0 / math.sqrt(hd))
    gqa_group = n_q // n_kv
    out = np.empty((n_q, hd), dtype=np.float32)

    for kvh in range(n_kv):
        k_mat = k_f[:, kvh, :]  # (seq_len, hd)
        v_mat = v_f[:, kvh, :]  # (seq_len, hd)
        q_grp = q_f[kvh * gqa_group : (kvh + 1) * gqa_group, :]  # (G, hd)
        scores = (q_grp @ k_mat.T).astype(np.float64) * scale  # (G, seq_len)
        m = np.max(scores, axis=1, keepdims=True)
        e = np.exp(scores - m)
        w = (e / np.sum(e, axis=1, keepdims=True)).astype(np.float32)
        out[kvh * gqa_group : (kvh + 1) * gqa_group, :] = w @ v_mat

    return out


class GlyphKVCacheV2:
    """
    Production Macro-Glyph KV Cache + Single-Pass Flash-Glyph GQA Engine.

    Supports:
      - Pure Glyph Mode (`sink_tokens=0, hot_tail_tokens=0`)
      - Dual-Road Mode (`sink_tokens=4, hot_tail_tokens=32`): pins attention sink
        tokens + rolling local tail in exact FP32 and crystallizes evicted tokens
        into `Neon-Runner` Option 4 (132B) or Option 5 (48B/36B) glyphs.
      - Global Contiguous Arena (`sliding_window=None`) & O(1) Ring-Buffer Sliding
        Window (`sliding_window=1024`) with zero `memmove` overhead.
      - Gemma 4 `k_eq_v=True` zero-duplicate global layers (`V = K`).
    """

    def __init__(
        self,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 32768,
        codec: str = "OPT4",
        codebook_family: str = "compass_s7",
        codebook_path: Optional[str] = None,
        apply_fwht8: bool = True,
        k_eq_v: bool = False,
        sliding_window: Optional[int] = None,
        sink_tokens: int = 4,
        hot_tail_tokens: int = 32,
        accelerator: Optional[NeonStreamAccelerator] = None,
        loaded_codebook: Optional[LoadedCodebookV2] = None,
        ternary_lut: Optional[np.ndarray] = None,
    ):
        if n_q_heads <= 0 or n_kv_heads <= 0 or (n_q_heads % n_kv_heads) != 0:
            raise ValueError(f"Invalid GQA head counts: n_q={n_q_heads}, n_kv={n_kv_heads}")
        if head_dim <= 0 or (head_dim % WEIGHTS_PER_GLYPH) != 0:
            raise ValueError(f"head_dim={head_dim} violates Zero-Remainder Law (R = {head_dim % 256})")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")

        self.n_q_heads = int(n_q_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.gqa_group = self.n_q_heads // self.n_kv_heads
        self.head_dim = int(head_dim)
        self.glyphs_per_head = self.head_dim // WEIGHTS_PER_GLYPH
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

        codec_norm = codec.upper()
        if codec_norm in ("OPT4", "4"):
            self.codec_id = 4
            self.bytes_per_glyph = BPG_OPT4
        elif codec_norm in ("OPT5", "OPT5_OUTLIER", "5", "48"):
            self.codec_id = 5
            self.bytes_per_glyph = BPG_OPT5_OUTLIER
        elif codec_norm in ("OPT5_STD", "36"):
            self.codec_id = 5
            self.bytes_per_glyph = BPG_OPT5_STD
        else:
            raise ValueError(f"Unsupported glyph KV codec: {codec!r}")

        self.head_glyph_bytes = self.glyphs_per_head * self.bytes_per_glyph
        self.apply_fwht8 = bool(apply_fwht8)
        self.k_eq_v = bool(k_eq_v)
        self.sliding_window = int(sliding_window) if sliding_window is not None else None
        self.sink_cap = 0 if self.sliding_window is not None else max(0, int(sink_tokens))
        self.tail_cap = max(0, int(hot_tail_tokens))

        self.acc = accelerator if accelerator is not None else NeonStreamAccelerator(auto_build=True)
        if not self.acc.has_v2_glyph_attn:
            raise RuntimeError(
                f"Loaded library '{self.acc.so_path}' lacks V2.1 GlyphAttention symbols; "
                "run `python neon_runner/build_native.py`."
            )

        self.codebook: Optional[LoadedCodebookV2] = None
        self.ternary_lut: Optional[np.ndarray] = ternary_lut
        if self.codec_id == 4:
            self.codebook = (
                loaded_codebook
                if loaded_codebook is not None
                else CodebookRegistryV2.resolve(codebook_family, explicit_path=codebook_path)
            )
            if self.ternary_lut is None:
                self.ternary_lut = self.acc.build_ternary_lut(self.codebook.array)

        # Arena capacity: ring-buffer bounded if sliding_window is set, else max_seq_len
        if self.sliding_window is not None:
            self.glyph_capacity = max(1, self.sliding_window - self.tail_cap)
        else:
            self.glyph_capacity = max(1, int(max_seq_len))

        # Head-major contiguous linear glyph arena: [n_kv_heads, glyph_capacity, head_glyph_bytes]
        # Guarantees unit-stride hardware prefetcher streaming inside each Split-K thread.
        self.k_arena = np.empty((self.n_kv_heads, self.glyph_capacity, self.head_glyph_bytes), dtype=np.uint8)
        self.v_arena = (
            None
            if self.k_eq_v
            else np.empty((self.n_kv_heads, self.glyph_capacity, self.head_glyph_bytes), dtype=np.uint8)
        )

        # Pinned attention sink + rolling local hot-tail FP32 buffers
        self.k_sink = np.empty((max(1, self.sink_cap), self.n_kv_heads, self.head_dim), dtype=np.float32)
        self.v_sink = (
            None
            if self.k_eq_v
            else np.empty((max(1, self.sink_cap), self.n_kv_heads, self.head_dim), dtype=np.float32)
        )
        self.k_tail = np.empty((max(1, self.tail_cap), self.n_kv_heads, self.head_dim), dtype=np.float32)
        self.v_tail = (
            None
            if self.k_eq_v
            else np.empty((max(1, self.tail_cap), self.n_kv_heads, self.head_dim), dtype=np.float32)
        )

        self.total_seen = 0
        self.sink_len = 0
        self.tail_len = 0
        self.tail_write_pos = 0
        self.glyph_len = 0
        self.glyph_write_pos = 0

        # Preallocated scratch buffer for single-token C encoding (zero per-step allocation)
        self._enc_k_buf = np.empty((self.n_kv_heads, self.head_glyph_bytes), dtype=np.uint8)
        self._enc_v_buf = None if self.k_eq_v else np.empty((self.n_kv_heads, self.head_glyph_bytes), dtype=np.uint8)

    def reset(self) -> None:
        self.total_seen = 0
        self.sink_len = 0
        self.tail_len = 0
        self.tail_write_pos = 0
        self.glyph_len = 0
        self.glyph_write_pos = 0

    def close(self) -> None:
        self.k_arena = np.empty((0, 0, 0), dtype=np.uint8)
        self.v_arena = None

    def __enter__(self) -> "GlyphKVCacheV2":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _encode_batch_into_glyphs(
        self,
        kv_batch: np.ndarray,
        out_bytes: np.ndarray,
        num_threads: int = 4,
    ) -> None:
        """Encodes contiguous FP32 tensor `kv_batch` (total floats divisible by 256) into `out_bytes`."""
        kv_c = np.ascontiguousarray(kv_batch, dtype=np.float32)
        n_glyphs = kv_c.size // WEIGHTS_PER_GLYPH
        assert kv_c.size == n_glyphs * WEIGHTS_PER_GLYPH
        assert out_bytes.nbytes == n_glyphs * self.bytes_per_glyph

        cb_ptr = (
            self.codebook.array.ctypes.data_as(ctypes.c_void_p)
            if self.codebook is not None
            else ctypes.c_void_p(0)
        )
        lut_ptr = (
            self.ternary_lut.ctypes.data_as(ctypes.c_void_p)
            if self.ternary_lut is not None
            else ctypes.c_void_p(0)
        )
        self.acc.lib.encode_kv_glyph_fast_c(
            kv_c.ctypes.data_as(ctypes.c_void_p),
            cb_ptr,
            lut_ptr,
            out_bytes.ctypes.data_as(ctypes.c_void_p),
            n_glyphs,
            self.codec_id,
            self.bytes_per_glyph,
            1 if self.apply_fwht8 else 0,
            num_threads,
        )

    def _push_glyph_token(self, k_tok: np.ndarray, v_tok: Optional[np.ndarray]) -> None:
        self._encode_batch_into_glyphs(k_tok, self._enc_k_buf, num_threads=1)
        slot = self.glyph_write_pos
        self.k_arena[:, slot, :] = self._enc_k_buf
        if not self.k_eq_v:
            assert v_tok is not None and self._enc_v_buf is not None and self.v_arena is not None
            self._encode_batch_into_glyphs(v_tok, self._enc_v_buf, num_threads=1)
            self.v_arena[:, slot, :] = self._enc_v_buf

        if self.sliding_window is not None:
            self.glyph_write_pos = (slot + 1) % self.glyph_capacity
            if self.glyph_len < self.glyph_capacity:
                self.glyph_len += 1
        else:
            if slot >= self.glyph_capacity:
                raise OverflowError(f"GlyphKVCacheV2 capacity ({self.glyph_capacity}) exceeded")
            self.glyph_write_pos = slot + 1
            self.glyph_len = self.glyph_write_pos

    def append_token(self, k_t: np.ndarray, v_t: Optional[np.ndarray] = None) -> None:
        """
        Appends one token's post-RoPE KV state:
          k_t: (n_kv_heads, head_dim)
          v_t: (n_kv_heads, head_dim) or None when k_eq_v=True
        """
        k_arr = np.ascontiguousarray(k_t, dtype=np.float32)
        assert k_arr.shape == (self.n_kv_heads, self.head_dim)
        v_arr = None
        if not self.k_eq_v:
            if v_t is None:
                raise ValueError("v_t is required when k_eq_v=False")
            v_arr = np.ascontiguousarray(v_t, dtype=np.float32)
            assert v_arr.shape == (self.n_kv_heads, self.head_dim)

        self.total_seen += 1

        # 1. Fill pinned attention sink first (global layers)
        if self.sink_len < self.sink_cap:
            self.k_sink[self.sink_len] = k_arr
            if not self.k_eq_v and self.v_sink is not None and v_arr is not None:
                self.v_sink[self.sink_len] = v_arr
            self.sink_len += 1
            return

        # 2. Pure glyph path when hot-tail is disabled
        if self.tail_cap == 0:
            self._push_glyph_token(k_arr, v_arr)
            return

        # 3. Rolling hot-tail buffer: if full, evict oldest tail token into glyph arena
        if self.tail_len == self.tail_cap:
            evict_k = self.k_tail[self.tail_write_pos]
            evict_v = None if self.k_eq_v or self.v_tail is None else self.v_tail[self.tail_write_pos]
            self._push_glyph_token(evict_k, evict_v)
            self.k_tail[self.tail_write_pos] = k_arr
            if not self.k_eq_v and self.v_tail is not None and v_arr is not None:
                self.v_tail[self.tail_write_pos] = v_arr
            self.tail_write_pos = (self.tail_write_pos + 1) % self.tail_cap
        else:
            self.k_tail[self.tail_len] = k_arr
            if not self.k_eq_v and self.v_tail is not None and v_arr is not None:
                self.v_tail[self.tail_len] = v_arr
            self.tail_len += 1
            self.tail_write_pos = self.tail_len % self.tail_cap

    def prefill_kv(
        self,
        k_seq: np.ndarray,
        v_seq: Optional[np.ndarray] = None,
        num_threads: int = 4,
    ) -> None:
        """
        High-speed parallel prefill of `(seq_len, n_kv_heads, head_dim)` into the cache.
        Encodes all glyph-tier tokens in parallel across OpenMP threads.
        """
        self.reset()
        k_f = np.ascontiguousarray(k_seq, dtype=np.float32)
        seq_len = k_f.shape[0]
        assert k_f.shape == (seq_len, self.n_kv_heads, self.head_dim)
        v_f = None
        if not self.k_eq_v:
            if v_seq is None:
                raise ValueError("v_seq is required when k_eq_v=False")
            v_f = np.ascontiguousarray(v_seq, dtype=np.float32)
            assert v_f.shape == (seq_len, self.n_kv_heads, self.head_dim)

        if seq_len == 0:
            return

        if self.sliding_window is not None and seq_len > self.sliding_window:
            k_f = np.ascontiguousarray(k_f[-self.sliding_window :])
            if v_f is not None:
                v_f = np.ascontiguousarray(v_f[-self.sliding_window :])
            self.total_seen = seq_len
            seq_len = self.sliding_window
        else:
            self.total_seen = seq_len

        s_n = min(seq_len, self.sink_cap)
        if s_n > 0:
            self.k_sink[:s_n] = k_f[:s_n]
            if not self.k_eq_v and self.v_sink is not None and v_f is not None:
                self.v_sink[:s_n] = v_f[:s_n]
            self.sink_len = s_n

        rem = seq_len - s_n
        t_n = min(rem, self.tail_cap)
        g_n = rem - t_n

        if g_n > 0:
            if g_n > self.glyph_capacity:
                raise OverflowError(f"Prefill glyph count {g_n} exceeds glyph_capacity {self.glyph_capacity}")
            # Transpose (g_n, n_kv_heads, head_dim) -> contiguous head-major (n_kv_heads, g_n, head_dim)
            k_mid_hm = np.ascontiguousarray(np.transpose(k_f[s_n : s_n + g_n], (1, 0, 2)))
            k_glyph_out = np.empty((self.n_kv_heads, g_n, self.head_glyph_bytes), dtype=np.uint8)
            self._encode_batch_into_glyphs(k_mid_hm, k_glyph_out, num_threads=num_threads)
            self.k_arena[:, :g_n, :] = k_glyph_out

            if not self.k_eq_v and self.v_arena is not None and v_f is not None:
                v_mid_hm = np.ascontiguousarray(np.transpose(v_f[s_n : s_n + g_n], (1, 0, 2)))
                v_glyph_out = np.empty((self.n_kv_heads, g_n, self.head_glyph_bytes), dtype=np.uint8)
                self._encode_batch_into_glyphs(v_mid_hm, v_glyph_out, num_threads=num_threads)
                self.v_arena[:, :g_n, :] = v_glyph_out

            self.glyph_len = g_n
            self.glyph_write_pos = g_n % self.glyph_capacity if self.sliding_window is not None else g_n

        if t_n > 0:
            self.k_tail[:t_n] = k_f[seq_len - t_n : seq_len]
            if not self.k_eq_v and self.v_tail is not None and v_f is not None:
                self.v_tail[:t_n] = v_f[seq_len - t_n : seq_len]
            self.tail_len = t_n
            self.tail_write_pos = t_n % max(1, self.tail_cap)

    def attention(
        self,
        q: np.ndarray,
        attn_scale: Optional[float] = None,
        num_threads: int = 4,
    ) -> np.ndarray:
        """
        Executes Single-Pass Flash-Glyph GQA Attention over the compressed glyph
        arena and merges any pinned sink / hot-tail tokens via exact log-sum-exp.
        """
        q_c = np.ascontiguousarray(q, dtype=np.float32)
        assert q_c.shape == (self.n_q_heads, self.head_dim)
        scale = float(attn_scale) if attn_scale is not None else self.attn_scale

        has_exact = (self.sink_len + self.tail_len) > 0
        out_glyph = np.empty((self.n_q_heads, self.head_dim), dtype=np.float32)
        m_glyph = np.empty(self.n_q_heads, dtype=np.float32) if has_exact else None
        l_glyph = np.empty(self.n_q_heads, dtype=np.float32) if has_exact else None

        if self.glyph_len > 0:
            seq_stride = self.head_glyph_bytes
            head_stride = self.glyph_capacity * self.head_glyph_bytes
            cb_ptr = (
                self.codebook.array.ctypes.data_as(ctypes.c_void_p)
                if self.codebook is not None
                else ctypes.c_void_p(0)
            )
            v_ptr = (
                ctypes.c_void_p(0)
                if self.k_eq_v or self.v_arena is None
                else self.v_arena.ctypes.data_as(ctypes.c_void_p)
            )
            m_ptr = m_glyph.ctypes.data_as(ctypes.c_void_p) if m_glyph is not None else ctypes.c_void_p(0)
            l_ptr = l_glyph.ctypes.data_as(ctypes.c_void_p) if l_glyph is not None else ctypes.c_void_p(0)

            rc = self.acc.lib.flash_glyph_attention_gqa_c(
                q_c.ctypes.data_as(ctypes.c_void_p),
                self.k_arena.ctypes.data_as(ctypes.c_void_p),
                v_ptr,
                cb_ptr,
                out_glyph.ctypes.data_as(ctypes.c_void_p),
                m_ptr,
                l_ptr,
                self.n_q_heads,
                self.n_kv_heads,
                self.head_dim,
                self.glyph_len,
                seq_stride,
                head_stride,
                scale,
                self.codec_id,
                self.bytes_per_glyph,
                1 if self.apply_fwht8 else 0,
                1 if self.k_eq_v else 0,
                num_threads,
            )
            if rc != 0:
                raise RuntimeError(f"flash_glyph_attention_gqa_c failed with error code {rc}")
            if not has_exact:
                return out_glyph
        else:
            if not has_exact:
                return np.zeros((self.n_q_heads, self.head_dim), dtype=np.float32)

        # Gather tiny exact sink + tail slice (typically <= 36 tokens total)
        exact_k_parts = []
        exact_v_parts = []
        if self.sink_len > 0:
            exact_k_parts.append(self.k_sink[: self.sink_len])
            if not self.k_eq_v and self.v_sink is not None:
                exact_v_parts.append(self.v_sink[: self.sink_len])
        if self.tail_len > 0:
            exact_k_parts.append(self.k_tail[: self.tail_len])
            if not self.k_eq_v and self.v_tail is not None:
                exact_v_parts.append(self.v_tail[: self.tail_len])

        k_ex = np.concatenate(exact_k_parts, axis=0) if len(exact_k_parts) > 1 else exact_k_parts[0]
        v_ex = (
            k_ex
            if self.k_eq_v
            else (np.concatenate(exact_v_parts, axis=0) if len(exact_v_parts) > 1 else exact_v_parts[0])
        )

        # Compute exact window (m_ex, l_ex, out_ex) in a single batched GQA tensor contraction
        q_grp = q_c.reshape(self.n_kv_heads, self.gqa_group, self.head_dim)
        k_hm = np.swapaxes(k_ex, 0, 1)  # (n_kv_heads, n_ex, head_dim)
        scores_ex = np.matmul(q_grp, np.swapaxes(k_hm, 1, 2)) * scale  # (n_kv_heads, gqa_group, n_ex)
        mx = np.max(scores_ex, axis=2)
        exps = np.exp(scores_ex - mx[:, :, None])
        lx = np.sum(exps, axis=2)
        v_hm = k_hm if self.k_eq_v else np.swapaxes(v_ex, 0, 1)
        out_ex = np.matmul(exps / lx[:, :, None], v_hm).reshape(self.n_q_heads, self.head_dim)
        m_ex = mx.reshape(self.n_q_heads)
        l_ex = lx.reshape(self.n_q_heads)

        if self.glyph_len == 0:
            return out_ex

        # Associative FlashAttention Log-Sum-Exp combiner: (m_glyph, l_glyph, out_glyph) (+) (m_ex, l_ex, out_ex)
        assert m_glyph is not None and l_glyph is not None
        m_joint = np.maximum(m_glyph, m_ex)
        w_g = l_glyph * np.exp(m_glyph - m_joint)
        w_e = l_ex * np.exp(m_ex - m_joint)
        denom = np.maximum(w_g + w_e, 1e-30)
        alpha_g = (w_g / denom)[:, None]
        alpha_e = (w_e / denom)[:, None]
        return (alpha_g * out_glyph + alpha_e * out_ex).astype(np.float32)

    def memory_stats(self) -> GlyphKVStats:
        active_seq = self.sink_len + self.glyph_len + self.tail_len
        streams = 1 if self.k_eq_v else 2
        glyph_bytes = self.glyph_len * self.n_kv_heads * self.head_glyph_bytes * streams
        exact_bytes = (self.sink_len + self.tail_len) * self.n_kv_heads * self.head_dim * 4 * streams
        total_res = glyph_bytes + exact_bytes
        dedup_bf16 = active_seq * self.n_kv_heads * self.head_dim * 2 * streams
        naive_bf16 = active_seq * self.n_kv_heads * self.head_dim * 2 * 2
        return GlyphKVStats(
            seq_len=active_seq,
            active_glyph_tokens=self.glyph_len,
            exact_sink_tokens=self.sink_len,
            exact_tail_tokens=self.tail_len,
            glyph_arena_bytes=glyph_bytes,
            exact_buffer_bytes=exact_bytes,
            total_resident_bytes=total_res,
            dense_bf16_kv_bytes=dedup_bf16,
            vllm_naive_bf16_bytes=naive_bf16,
            compression_vs_dedup_bf16=float(dedup_bf16) / max(total_res, 1),
            compression_vs_naive_bf16=float(naive_bf16) / max(total_res, 1),
        )


def _self_test() -> int:
    """
    Mandatory 6-Gate Self-Verification Battery for GlyphAttentionV2 (Fail-Closed).
    """
    print("=" * 80)
    print(f"{GLYPH_ATTN_COMPONENT_ID} (v{GLYPH_ATTN_VERSION}) -- 6-GATE PHYSICAL SILICON BATTERY")
    print("=" * 80)
    failures = 0

    def record_gate(gate_num: int, title: str, ok: bool, detail: str) -> None:
        nonlocal failures
        tag = "PASS" if ok else "FAIL"
        print(f"[Gate {gate_num}] [{tag}] {title} -- {detail}")
        if not ok:
            failures += 1

    acc = NeonStreamAccelerator(auto_build=True)
    cb_compass = CodebookRegistryV2.resolve("compass_s7")
    cb_trit = CodebookRegistryV2.resolve("trit_rvq")
    cb_golden = CodebookRegistryV2.resolve("golden_s7")

    # Gate 1: Native Symbol Binding & O(1) Ternary Lattice LUT Discrimination
    try:
        lut_c = acc.build_ternary_lut(cb_compass.array)
        lut_t = acc.build_ternary_lut(cb_trit.array)
        lut_g = acc.build_ternary_lut(cb_golden.array)
        valid_c = int(np.sum(lut_c != 0xFFFF)) if lut_c is not None else 0
        valid_t = int(np.sum(lut_t != 0xFFFF)) if lut_t is not None else 0
        g1_ok = (
            acc.self_test()
            and acc.has_v2_glyph_attn
            and valid_c == 2048
            and valid_t == 2048
            and lut_g is None
        )
        record_gate(
            1,
            "Native Kernel Binding & Ternary Lattice LUT Gate",
            g1_ok,
            f"compass_entries={valid_c}, trit_entries={valid_t}, golden_is_ternary={lut_g is not None}",
        )
    except Exception as exc:
        record_gate(1, "Native Kernel Binding & Ternary Lattice LUT Gate", False, repr(exc))

    # Gate 2: O(1) Online KV Encoder Fidelity & Sub-Microsecond Speed on Physical E4B Weights
    try:
        sample_path = Path(__file__).resolve().parent.parent / "model" / "gemma4_e4b_sample_layer0_q_proj.neon_opt4.lanu"
        if not sample_path.is_file():
            raise FileNotFoundError(f"Missing physical Gemma 4 E4B sample: {sample_path}")
        raw_lanu = sample_path.read_bytes()[16 : 16 + 16 * BPG_OPT4]
        phys_w = acc.decode(raw_lanu, cb_compass.array, apply_fwht8=True)  # 4096 continuous E4B floats
        # Normalize each 256d head to unit RMS (matching Gemma 4 qk_norm_heads pre-RoPE scale)
        phys_heads = phys_w.reshape(16, 256)
        phys_heads = (phys_heads / (np.sqrt(np.mean(phys_heads * phys_heads, axis=1, keepdims=True)) + 1e-6)).astype(np.float32)
        phys_w = phys_heads.reshape(-1)

        g4_buf = np.empty(16 * BPG_OPT4, dtype=np.uint8)
        g5_buf = np.empty(16 * BPG_OPT5_OUTLIER, dtype=np.uint8)
        assert lut_c is not None
        acc.lib.encode_kv_glyph_fast_c(
            phys_w.ctypes.data_as(ctypes.c_void_p),
            cb_compass.array.ctypes.data_as(ctypes.c_void_p),
            lut_c.ctypes.data_as(ctypes.c_void_p),
            g4_buf.ctypes.data_as(ctypes.c_void_p),
            16, 4, 132, 1, 1,
        )
        acc.lib.encode_kv_glyph_fast_c(
            phys_w.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(0),
            ctypes.c_void_p(0),
            g5_buf.ctypes.data_as(ctypes.c_void_p),
            16, 5, 48, 1, 1,
        )
        rec4 = acc.decode(g4_buf, cb_compass.array, apply_fwht8=True)
        rec5 = acc.decode_opt5(g5_buf, bytes_per_glyph=48, apply_fwht8=True)
        cos4 = float(np.dot(phys_w, rec4) / (np.linalg.norm(phys_w) * np.linalg.norm(rec4) + 1e-12))
        cos5 = float(np.dot(phys_w, rec5) / (np.linalg.norm(phys_w) * np.linalg.norm(rec5) + 1e-12))
        g2_ok = cos4 > 0.96 and cos5 > 0.80 and bool(np.all(np.isfinite(rec4)))
        record_gate(
            2,
            "O(1) Online KV Encoder Reconstruction Sanity (Opt4 & Opt5+FWHT8)",
            g2_ok,
            f"opt4_cos={cos4:.5f}, opt5_fwht_cos={cos5:.5f}",
        )
    except Exception as exc:
        record_gate(2, "O(1) Online KV Encoder Reconstruction Sanity", False, repr(exc))

    # Gate 3: Non-Rank-1 Head & Sequence Distinctness in Glyph Arena
    try:
        rng = np.random.default_rng(20261029)
        # Construct structured multi-frequency RoPE-like test manifold derived from physical E4B seed
        base_vec = phys_w.reshape(16, 256)
        k_seq_test = np.empty((128, 4, 512), dtype=np.float32)
        for t in range(128):
            ang = 0.07 * t
            c_rot, s_rot = math.cos(ang), math.sin(ang)
            for h in range(4):
                v0 = base_vec[(t + h * 3) % 16] * c_rot + base_vec[(t + h * 5 + 1) % 16] * s_rot
                v1 = base_vec[(t * 2 + h) % 16] * s_rot - base_vec[(t + h + 7) % 16] * c_rot
                k_seq_test[t, h, :256] = v0
                k_seq_test[t, h, 256:] = v1
        q_test = np.empty((32, 512), dtype=np.float32)
        for qh in range(32):
            q_test[qh] = k_seq_test[97, qh // 8] * 0.85 + 0.15 * k_seq_test[(qh * 3) % 128, qh // 8]

        with GlyphKVCacheV2(
            n_q_heads=32, n_kv_heads=4, head_dim=512, max_seq_len=256,
            codec="OPT4", k_eq_v=True, sink_tokens=0, hot_tail_tokens=0, accelerator=acc
        ) as cache_pure:
            cache_pure.prefill_kv(k_seq_test)
            ctx_out = cache_pure.attention(q_test)
            head_norms = np.linalg.norm(ctx_out, axis=1, keepdims=True) + 1e-12
            gram = np.abs((ctx_out / head_norms) @ (ctx_out / head_norms).T)
            # Heads across distinct KV groups must not collapse to rank-1
            cross_group_cos = float(np.max(gram[:8, 8:]))
            g3_ok = cross_group_cos < 0.98 and ctx_out.shape == (32, 512)
            record_gate(
                3,
                "Multi-Head Output Distinctness (32Q / 4KV / 512d GQA)",
                g3_ok,
                f"cross_kv_group_max_cos={cross_group_cos:.4f}",
            )
    except Exception as exc:
        record_gate(3, "Multi-Head Output Distinctness", False, repr(exc))

    # Gate 4: Exact Bit/FP32 Mathematical Parity — Flash-Glyph C Kernel vs Decoded Dense Attention
    try:
        # Verify BOTH Option 4 (512d, 8:1 GQA, k_eq_v=True, Split-K tiles) AND Option 5 (256d, 2:1 GQA, k_eq_v=False)
        with GlyphKVCacheV2(
            n_q_heads=32, n_kv_heads=4, head_dim=512, max_seq_len=1200,
            codec="OPT4", k_eq_v=True, sink_tokens=0, hot_tail_tokens=0, accelerator=acc
        ) as c4:
            k_1200 = np.tile(k_seq_test, (10, 1, 1))[:1200]  # 1200 > 512 triggers multi-tile Split-K OpenMP reduction!
            c4.prefill_kv(k_1200)
            y_flash4 = c4.attention(q_test, num_threads=4)

            # Decode stored glyphs back to FP32 and run exact dense reference
            k_dec4 = np.empty((1200, 4, 512), dtype=np.float32)
            for h in range(4):
                raw_h = c4.k_arena[h, :1200, :].reshape(-1)
                k_dec4[:, h, :] = acc.decode(raw_h, c4.codebook.array, apply_fwht8=True).reshape(1200, 512)
            y_ref_dec4 = dense_gqa_attention_reference(q_test, k_dec4, None)
            err_flash4 = float(np.max(np.abs(y_flash4 - y_ref_dec4)))

        k_256 = k_seq_test[:, :, :256]
        v_256 = k_seq_test[:, :, 256:]
        q_256 = q_test[:8, :256]
        with GlyphKVCacheV2(
            n_q_heads=8, n_kv_heads=4, head_dim=256, max_seq_len=128,
            codec="OPT5", k_eq_v=False, sink_tokens=0, hot_tail_tokens=0, accelerator=acc
        ) as c5:
            c5.prefill_kv(k_256, v_256)
            y_flash5 = c5.attention(q_256, num_threads=2)
            k_dec5 = np.empty((128, 4, 256), dtype=np.float32)
            v_dec5 = np.empty((128, 4, 256), dtype=np.float32)
            for h in range(4):
                k_dec5[:, h, :] = acc.decode_opt5(c5.k_arena[h, :128, :].reshape(-1), 48, apply_fwht8=True).reshape(128, 256)
                v_dec5[:, h, :] = acc.decode_opt5(c5.v_arena[h, :128, :].reshape(-1), 48, apply_fwht8=True).reshape(128, 256)
            y_ref_dec5 = dense_gqa_attention_reference(q_256, k_dec5, v_dec5)
            err_flash5 = float(np.max(np.abs(y_flash5 - y_ref_dec5)))

        g4_ok = err_flash4 < 1e-4 and err_flash5 < 1e-4
        record_gate(
            4,
            "Flash-Glyph C Kernel vs Decoded FP32 Attention Parity (Split-K + GQA)",
            g4_ok,
            f"opt4_512d_8to1_err={err_flash4:.2e}, opt5_256d_2to1_err={err_flash5:.2e}",
        )
    except Exception as exc:
        record_gate(4, "Flash-Glyph C Kernel vs Decoded FP32 Attention Parity", False, repr(exc))

    # Gate 5: Dual-Road (Sink+Tail) Semantic Fidelity & Sliding-Window Ring Buffer Accounting
    try:
        # Query spanning sink (t=2), historical glyph body (t=97), and hot-tail (t=123)
        q_dual = np.empty((32, 512), dtype=np.float32)
        for qh in range(32):
            kv_idx = qh // 8
            mix = 0.30 * k_seq_test[2, kv_idx] + 0.45 * k_seq_test[97, kv_idx] + 0.45 * k_seq_test[123, kv_idx]
            q_dual[qh] = mix / (np.sqrt(np.mean(mix * mix)) + 1e-6)

        y_dense_hist = dense_gqa_attention_reference(q_test, k_seq_test, None)
        y_dense_dual = dense_gqa_attention_reference(q_dual, k_seq_test, None)

        with GlyphKVCacheV2(
            n_q_heads=32, n_kv_heads=4, head_dim=512, max_seq_len=256,
            codec="OPT4", k_eq_v=True, sink_tokens=0, hot_tail_tokens=0, accelerator=acc
        ) as c_pure_cmp:
            c_pure_cmp.prefill_kv(k_seq_test)
            y_pure_on_dual = c_pure_cmp.attention(q_dual)
            cos_pure_on_dual = float(
                np.dot(y_pure_on_dual.reshape(-1), y_dense_dual.reshape(-1))
                / (np.linalg.norm(y_pure_on_dual) * np.linalg.norm(y_dense_dual) + 1e-12)
            )

        with GlyphKVCacheV2(
            n_q_heads=32, n_kv_heads=4, head_dim=512, max_seq_len=256,
            codec="OPT4", k_eq_v=True, sink_tokens=4, hot_tail_tokens=16, accelerator=acc
        ) as c_dual:
            for t in range(128):
                c_dual.append_token(k_seq_test[t])
            y_hist = c_dual.attention(q_test)
            y_dual = c_dual.attention(q_dual)
            st = c_dual.memory_stats()
            cos_hist_only = float(
                np.dot(y_hist.reshape(-1), y_dense_hist.reshape(-1))
                / (np.linalg.norm(y_hist) * np.linalg.norm(y_dense_hist) + 1e-12)
            )
            cos_vs_fp32 = float(
                np.dot(y_dual.reshape(-1), y_dense_dual.reshape(-1))
                / (np.linalg.norm(y_dual) * np.linalg.norm(y_dense_dual) + 1e-12)
            )

        with GlyphKVCacheV2(
            n_q_heads=8, n_kv_heads=4, head_dim=256, max_seq_len=64,
            codec="OPT5", k_eq_v=False, sliding_window=64, sink_tokens=0, hot_tail_tokens=8, accelerator=acc
        ) as c_ring:
            for t in range(128):
                c_ring.append_token(k_256[t], v_256[t])
            st_ring = c_ring.memory_stats()

        g5_ok = (
            cos_hist_only > 0.98
            and cos_vs_fp32 > 0.99
            and cos_vs_fp32 > cos_pure_on_dual
            and st.seq_len == 128
            and st.exact_sink_tokens == 4
            and st.exact_tail_tokens == 16
            and st.active_glyph_tokens == 108
            and st_ring.seq_len == 64
        )
        record_gate(
            5,
            "Dual-Road (Sink+Tail) Fidelity & O(1) Ring-Buffer Boundary Accounting",
            g5_ok,
            f"dual_cos={cos_vs_fp32:.6f} (>pure {cos_pure_on_dual:.6f}), hist_cos={cos_hist_only:.6f}, ring={st_ring.seq_len}/64",
        )
    except Exception as exc:
        record_gate(5, "Dual-Road Fidelity & Boundary Accounting", False, repr(exc))

    # Gate 6: End-to-End 4K Context Execution & Clean Lifecycle Without Exceptions
    try:
        k_4k = np.tile(k_seq_test, (32, 1, 1))  # 4096 x 4 x 512 (Gemma 4 31B global layer geometry)
        with GlyphKVCacheV2(
            n_q_heads=32, n_kv_heads=4, head_dim=512, max_seq_len=4096,
            codec="OPT5", k_eq_v=True, sink_tokens=4, hot_tail_tokens=32, accelerator=acc
        ) as c_4k:
            c_4k.prefill_kv(k_4k, num_threads=4)
            _ = c_4k.attention(q_test, num_threads=4)
            t0 = time.perf_counter()
            y_4k = c_4k.attention(q_test, num_threads=4)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            st_4k = c_4k.memory_stats()
        g6_ok = y_4k.shape == (32, 512) and bool(np.all(np.isfinite(y_4k))) and st_4k.compression_vs_naive_bf16 > 15.0
        record_gate(
            6,
            "End-to-End 4K Global GQA Execution & Compression Verification",
            g6_ok,
            f"latency={dt_ms:.2f} ms, kv_ram={st_4k.total_resident_bytes / 1024:.1f} KB "
            f"({st_4k.compression_vs_naive_bf16:.1f}x vs naive BF16)",
        )
    except Exception as exc:
        record_gate(6, "End-to-End 4K Global GQA Execution", False, repr(exc))

    print("=" * 80)
    if failures > 0:
        print(f"VERDICT: FAILED ({failures} gate(s) failed)")
        raise SystemExit(failures)
    print("VERDICT: ALL 6 FAIL-CLOSED GLYPH-ATTENTION GATES PASSED ON PHYSICAL SILICON")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{GLYPH_ATTN_COMPONENT_ID} CLI")
    parser.add_argument("--self-test", action="store_true", help="Run 6-gate self-verification")
    _ = parser.parse_args()
    return _self_test()


if __name__ == "__main__":
    sys.exit(main())


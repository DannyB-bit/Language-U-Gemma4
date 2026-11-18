#!/usr/bin/env python3
"""
Clean single-pass QAT teacher state capture for Gemma 4 31B V4 calibration & C8DB bridge.

Streams L00..L59 via bounded open/seek/read on `model.safetensors` (< 500 MB RSS,
zero mmap page-cache thrashing) and saves:
  1. `<out-dir>/meta_v4.json`
  2. `<out-dir>/snaps_L00..L59.npz` (`x_cal` fp16, `x_t2` / `x_tf` f32)
  3. `<out-dir>/snaps_FINAL.npz` (`x_t2`, `x_tf`, `fn`)
  4. `<out-dir>/teacher_bridge_v4.npz` (L58/L59 KV + L60 post-norm targets for Step 3)

Run with no arguments or `--selftest` to execute the 6-gate offline suite.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import ctypes
import gc
import json
import math
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pack_t16c_v3 import SafetensorsReader, resolve_safetensors_path
from v4_corpus import assert_no_contamination, build_tokenized_v4_corpus


def _rss_mb() -> float:
    """Return current Linux VmRSS in MiB, or 0.0 on non-procfs hosts."""
    p = Path("/proc/self/status")
    if not p.is_file():
        return 0.0
    for line in p.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def rmsn(x: np.ndarray, w: Optional[np.ndarray], eps: float = 1e-6) -> np.ndarray:
    """Apply Gemma 4 unshifted RMSNorm (`weight` directly, no `1 + weight`)."""
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (y * w).astype(np.float32) if w is not None else y.astype(np.float32)


def rope_with_pos(
    x: np.ndarray,
    pos: np.ndarray,
    theta: float,
    rot_dims: int,
    hd: int,
) -> np.ndarray:
    """Apply Gemma 4 proportional RoPE (`rot_dims` pairs split at `hd // 2`)."""
    hr = rot_dims // 2
    hh = hd // 2
    inv_f = 1.0 / (theta ** (np.arange(0, 2 * hr, 2, dtype=np.float32) / float(hd)))
    ang = pos[:, None].astype(np.float32) * inv_f[None, :]
    c = np.cos(ang)[:, None, :]
    s = np.sin(ang)[:, None, :]
    out = x.copy()
    x0 = x[:, :, :hr]
    x1 = x[:, :, hh : hh + hr]
    out[:, :, :hr] = x0 * c - x1 * s
    out[:, :, hh : hh + hr] = x0 * s + x1 * c
    return out


def gelu_tanh(g: np.ndarray) -> np.ndarray:
    """Compute Gemma GELU-tanh approximation cleanly in float32."""
    c0 = math.sqrt(2.0 / math.pi)
    return (0.5 * g * (1.0 + np.tanh(c0 * (g + 0.044715 * g * g * g)))).astype(np.float32)


def causal_gqa_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    pos: np.ndarray,
    prompt_slices: List[Tuple[int, int]],
    *,
    nh: int,
    nkv: int,
    hd: int,
    rot_dims: int,
    theta: float,
    is_full: bool,
    window: int,
) -> np.ndarray:
    """Compute exact Gemma 4 per-sequence causal GQA attention (scale=1.0, window on sliding)."""
    n_tok = len(q)
    gqa = nh // nkv
    out = np.empty((n_tok, nh, hd), dtype=np.float32)
    for s0, s1 in prompt_slices:
        slen = s1 - s0
        sl_pos = pos[s0:s1]
        qs = rope_with_pos(q[s0:s1].reshape(slen, nh, hd), sl_pos, theta, rot_dims, hd)
        ks = rope_with_pos(k[s0:s1].reshape(slen, nkv, hd), sl_pos, theta, rot_dims, hd)
        vs = v[s0:s1].reshape(slen, nkv, hd)
        mask = np.triu(np.full((slen, slen), -1e9, dtype=np.float32), 1)
        if not is_full and slen > window:
            idx = np.arange(slen)
            mask = np.where((idx[:, None] - idx[None, :]) >= window, -1e9, mask)
        for h in range(nh):
            kv_h = h // gqa
            sc = qs[:, h, :] @ ks[:, kv_h, :].T + mask
            sc -= np.max(sc, axis=-1, keepdims=True)
            pr = np.exp(sc)
            pr /= np.sum(pr, axis=-1, keepdims=True)
            out[s0:s1, h, :] = pr @ vs[:, kv_h, :]
    return out.reshape(n_tok, nh * hd)


class _FastTeacherOps:
    """Optional C-accelerated INT4 QAT unpacker and SGEMM with NumPy fallback for offline tests."""

    def __init__(self, work_dir: Optional[Path], require_native: bool = True) -> None:
        self.lib_g = None
        self.lib_sg = None
        if work_dir is not None:
            gp = work_dir / "libgptq_fast6.so"
            sp = work_dir / "libfast_sgemm.so"
            if gp.is_file():
                self.lib_g = ctypes.CDLL(str(gp))
                self.lib_g.unpack_qat_rows_f32.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
            if sp.is_file():
                self.lib_sg = ctypes.CDLL(str(sp))
                self.lib_sg.fast_sgemm_nt_f32.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_float,
                ]
        if require_native and (self.lib_g is None or self.lib_sg is None):
            raise FileNotFoundError(
                f"Missing required native capture libraries in {work_dir} (libgptq_fast6.so / libfast_sgemm.so)"
            )

    def mm_xt(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        xc = np.ascontiguousarray(x, dtype=np.float32)
        wc = np.ascontiguousarray(w, dtype=np.float32)
        if self.lib_sg is not None:
            y = np.empty((xc.shape[0], wc.shape[0]), dtype=np.float32)
            self.lib_sg.fast_sgemm_nt_f32(
                xc.ctypes.data_as(ctypes.c_void_p),
                wc.ctypes.data_as(ctypes.c_void_p),
                y.ctypes.data_as(ctypes.c_void_p),
                xc.shape[0],
                wc.shape[0],
                xc.shape[1],
                1.0,
                0.0,
            )
            return y
        return (xc @ wc.T).astype(np.float32)

    def read_qat_mm(self, sr: SafetensorsReader, pfx: str, x: np.ndarray) -> np.ndarray:
        ip = sr.hdr[pfx + ".weight_packed"]
        isc = sr.hdr[pfx + ".weight_scale"]
        sr._fp.seek(sr.data_base + int(ip["data_offsets"][0]))
        pk_raw = sr._fp.read(int(ip["data_offsets"][1]) - int(ip["data_offsets"][0]))
        sr._fp.seek(sr.data_base + int(isc["data_offsets"][0]))
        sc_raw = sr._fp.read(int(isc["data_offsets"][1]) - int(isc["data_offsets"][0]))
        rows = int(ip["shape"][0])
        d_in = int(ip["shape"][1]) * 8
        pk = np.frombuffer(pk_raw, dtype=np.int32)
        sc = np.frombuffer(sc_raw, dtype=np.uint16)
        if self.lib_g is not None:
            w = np.empty((rows, d_in), dtype=np.float32)
            self.lib_g.unpack_qat_rows_f32(
                pk.ctypes.data_as(ctypes.c_void_p),
                sc.ctypes.data_as(ctypes.c_void_p),
                w.ctypes.data_as(ctypes.c_void_p),
                rows,
                d_in,
            )
        else:
            shifts = np.arange(0, 32, 4, dtype=np.uint32)
            nibs = ((pk.view(np.uint32).reshape(rows, -1)[:, :, None] >> shifts[None, None, :]) & 0xF).astype(np.int16) - 8
            sc_f32 = (sc.astype(np.uint32) << 16).view(np.float32).reshape(rows, -1)
            w = (nibs.reshape(rows, -1, 32).astype(np.float32) * sc_f32[:, :, None]).reshape(rows, d_in)
        y = self.mm_xt(x, w)
        del w, pk, sc
        return y


def _embed_lookup(sr: SafetensorsReader, all_ids: List[int], d_model: int, escale: float) -> np.ndarray:
    """Seek/read exact token embedding rows and scale by `sqrt(d_model)`."""
    info = sr.hdr["model.language_model.embed_tokens.weight"]
    dtype = info["dtype"]
    elem_b = 2 if dtype in ("BF16", "F16") else 4
    base_off = sr.data_base + int(info["data_offsets"][0])
    rb = d_model * elem_b
    out = np.empty((len(all_ids), d_model), dtype=np.float32)
    for idx, tid in enumerate(all_ids):
        sr._fp.seek(base_off + tid * rb)
        raw = sr._fp.read(rb)
        if dtype == "BF16":
            vec = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F32":
            vec = np.frombuffer(raw, dtype="<f4")
        else:
            vec = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        out[idx] = vec * escale
    return out


def run_capture(
    sf_path: Path,
    cfg_path: Path,
    tok_path: Path,
    out_dir: Path,
    work_dir: Path,
    *,
    require_native: bool = True,
) -> Path:
    """Capture L00..L59 QAT hidden states and Step-3 L58/L59/L60 teacher targets."""
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["text_config"]
    ops = _FastTeacherOps(work_dir, require_native=require_native)

    corp = build_tokenized_v4_corpus(tok_path, include_eval_probes=True)
    all_ids: List[int] = corp["all_ids"]  # type: ignore[assignment]
    next_token_ids = np.array(corp["next_token_ids"], dtype=np.int32)
    pos_arr = np.array(corp["pos_arr"], dtype=np.float32)
    ans_mask = np.array(corp["ans_mask"], dtype=bool)
    prompt_domains: List[str] = corp["prompt_domains"]  # type: ignore[assignment]
    prompt_slices: List[Tuple[int, int]] = corp["prompt_slices"]  # type: ignore[assignment]
    cal_end = int(corp["cal_end"])  # type: ignore[arg-type]
    t2_s0, t2_s1 = [int(v) for v in corp["test2_bounds"]]  # type: ignore[misc]
    tf_s0, tf_s1 = [int(v) for v in corp["testf_bounds"]]  # type: ignore[misc]
    n_tok = len(all_ids)

    # Assert strict domain isolation: calibration contains zero holdout probes.
    assert_no_contamination(
        [d for d in prompt_domains[:cal_end]],
        "tokenized_domains",
    )
    assert all(d != "heldout" for d in prompt_domains[:cal_end])

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = {
        "cal_end": cal_end,
        "cal_pairs_count": int(corp["cal_pairs_count"]),  # type: ignore[arg-type]
        "ans_tokens": int(ans_mask[:cal_end].sum()),
        "test2": [t2_s0, t2_s1],
        "testf": [tf_s0, tf_s1],
        "prompt_slices": [list(s) for s in prompt_slices],
        "prompt_domains": prompt_domains,
        "all_ids": all_ids,
        "next_token_ids": next_token_ids.tolist(),
        "pos_arr": [int(p) for p in pos_arr],
        "ans_mask": ans_mask.tolist(),
    }
    (out_dir / "meta_v4.json").write_text(json.dumps(meta_out), encoding="utf-8")
    print(
        f"[CaptureV4] n_tok={n_tok} cal_end={cal_end} ans_tokens={meta_out['ans_tokens']} "
        f"test2={t2_s0}:{t2_s1} testf={tf_s0}:{tf_s1}",
        flush=True,
    )

    n_layers = int(cfg["num_hidden_layers"])
    d_model = int(cfg["hidden_size"])
    eps = float(cfg["rms_norm_eps"])
    escale = math.sqrt(float(d_model))
    nh = int(cfg["num_attention_heads"])
    nkv_s = int(cfg["num_key_value_heads"])
    nkv_f = int(cfg["num_global_key_value_heads"])
    hd_s = int(cfg["head_dim"])
    hd_f = int(cfg["global_head_dim"])
    k_eq_v = bool(cfg.get("attention_k_eq_v", False))
    layer_types = cfg["layer_types"]
    rp = cfg["rope_parameters"]
    theta_g = float(rp["full_attention"]["rope_theta"])
    part_g = float(rp["full_attention"].get("partial_rotary_factor", 1.0))
    theta_l = float(rp["sliding_attention"]["rope_theta"])
    part_l = float(rp["sliding_attention"].get("partial_rotary_factor", 1.0))
    win = int(cfg["sliding_window"])

    bridge_blobs: Dict[str, np.ndarray] = {}
    bridge_partial = out_dir / "bridge_l58_l59_partial.npz"

    with SafetensorsReader(sf_path) as sr:
        x = _embed_lookup(sr, all_ids, d_model, escale)
        start_l = 0
        for l in range(n_layers - 1, -1, -1):
            snap_p = out_dir / f"snaps_L{l:02d}.npz"
            x_p = out_dir / f"x_after_L{l:02d}.npy"
            if snap_p.is_file() and x_p.is_file():
                x = np.load(x_p).astype(np.float32)
                start_l = l + 1
                if start_l > 58 and bridge_partial.is_file():
                    with np.load(bridge_partial) as bp:
                        for k in bp.files:
                            bridge_blobs[k] = bp[k]
                print(f"[CaptureV4 RESUME] starting at L{start_l:02d}", flush=True)
                break

        ts = time.perf_counter()
        for l in range(start_l, n_layers):
            tl = time.perf_counter()
            p = f"model.language_model.layers.{l}."
            is_full = layer_types[l] == "full_attention"
            nkv = nkv_f if is_full else nkv_s
            hd = hd_f if is_full else hd_s
            theta = theta_g if is_full else theta_l
            rot_d = int(round(hd * (part_g if is_full else part_l)))

            g_in = sr.read_f32_tensor(p + "input_layernorm.weight")
            qn = sr.read_f32_tensor(p + "self_attn.q_norm.weight")
            kn = sr.read_f32_tensor(p + "self_attn.k_norm.weight")
            g_pa = sr.read_f32_tensor(p + "post_attention_layernorm.weight")
            g_pf = sr.read_f32_tensor(p + "pre_feedforward_layernorm.weight")
            g_pff = sr.read_f32_tensor(p + "post_feedforward_layernorm.weight")
            lsc = float(np.asarray(sr.read_f32_tensor(p + "layer_scalar")).reshape(-1)[0])

            tmp_snap = out_dir / f"snaps_L{l:02d}.tmp.npz"
            np.savez(
                tmp_snap,
                x_cal=x[:cal_end].astype(np.float16),
                x_t2=x[t2_s0:t2_s1],
                x_tf=x[tf_s0:tf_s1],
            )
            os.replace(tmp_snap, out_dir / f"snaps_L{l:02d}.npz")

            if l == 58:
                bridge_blobs["h_teach_before_L58"] = x.astype(np.float32).copy()

            inp = rmsn(x, g_in, eps)
            qt = ops.read_qat_mm(sr, p + "self_attn.q_proj", inp)
            kt = ops.read_qat_mm(sr, p + "self_attn.k_proj", inp)
            vt = kt.copy() if (is_full and k_eq_v) else ops.read_qat_mm(sr, p + "self_attn.v_proj", inp)

            if l == 58:
                bridge_blobs["ht_in_58"] = inp.copy()
                bridge_blobs["kt_58_raw"] = kt.copy()
                bridge_blobs["vt_58_raw"] = vt.copy()
                np.savez(bridge_partial, **bridge_blobs)
            elif l == 59:
                bridge_blobs["ht_in_59"] = inp.copy()
                bridge_blobs["kt_59_raw"] = kt.copy()

            qt_n = rmsn(qt.reshape(n_tok, nh, hd), qn, eps).reshape(n_tok, nh * hd)
            kt_n = rmsn(kt.reshape(n_tok, nkv, hd), kn, eps).reshape(n_tok, nkv * hd)
            vt_n = rmsn(vt.reshape(n_tok, nkv, hd), None, eps).reshape(n_tok, nkv * hd)
            del qt, kt, vt, inp
            ctx = causal_gqa_attention(
                qt_n,
                kt_n,
                vt_n,
                pos_arr,
                prompt_slices,
                nh=nh,
                nkv=nkv,
                hd=hd,
                rot_dims=rot_d,
                theta=theta,
                is_full=is_full,
                window=win,
            )
            del qt_n, kt_n, vt_n
            ot = ops.read_qat_mm(sr, p + "self_attn.o_proj", ctx)
            del ctx
            xm = x + rmsn(ot, g_pa, eps)
            del ot

            hm = rmsn(xm, g_pf, eps)
            gt = ops.read_qat_mm(sr, p + "mlp.gate_proj", hm)
            ut = ops.read_qat_mm(sr, p + "mlp.up_proj", hm)
            act = gelu_tanh(gt) * ut
            del gt, ut, hm
            dt_out = ops.read_qat_mm(sr, p + "mlp.down_proj", act)
            del act
            x = ((xm + rmsn(dt_out, g_pff, eps)) * lsc).astype(np.float32)
            del xm, dt_out
            gc.collect()

            tmp_x = out_dir / f"x_after_L{l:02d}.tmp.npy"
            np.save(tmp_x, x)
            os.replace(tmp_x, out_dir / f"x_after_L{l:02d}.npy")
            if l > 0:
                prev_x = out_dir / f"x_after_L{l - 1:02d}.npy"
                if prev_x.exists():
                    prev_x.unlink()
            print(
                f"L{l:02d}/{n_layers - 1:02d} ({'F' if is_full else 'S'}) "
                f"dt={time.perf_counter() - tl:5.1f}s elap={time.perf_counter() - ts:6.1f}s "
                f"rss={_rss_mb():.0f}MB",
                flush=True,
            )

        fn_orig = sr.read_f32_tensor("model.language_model.norm.weight")
        tmp_fin = out_dir / "snaps_FINAL.tmp.npz"
        np.savez(tmp_fin, x_t2=x[t2_s0:t2_s1], x_tf=x[tf_s0:tf_s1], fn=fn_orig)
        os.replace(tmp_fin, out_dir / "snaps_FINAL.npz")

        bridge_path = out_dir / "teacher_bridge_v4.npz"
        if n_layers >= 60:
            tmp_br = out_dir / "teacher_bridge_v4.tmp.npz"
            np.savez(
                tmp_br,
                h_teach_L59=x.astype(np.float32),
                fn_orig=fn_orig.astype(np.float32),
                all_ids=np.array(all_ids, dtype=np.int32),
                next_token_ids=next_token_ids,
                pos_arr=pos_arr,
                ans_mask=ans_mask,
                cal_end=np.array(cal_end, dtype=np.int32),
                test2_bounds=np.array([t2_s0, t2_s1], dtype=np.int32),
                testf_bounds=np.array([tf_s0, tf_s1], dtype=np.int32),
                prompt_slices=np.array(prompt_slices, dtype=np.int32),
                prompt_domains=np.array(prompt_domains),
                **bridge_blobs,
            )
            os.replace(tmp_br, bridge_path)
            if bridge_partial.exists():
                bridge_partial.unlink()
        print("CAPTURE_V4_COMPLETE OK", flush=True)
    return out_dir


def _self_test() -> int:
    """Run 6 fail-closed self-test gates on pure capture primitives."""
    failures = 0
    rng = np.random.default_rng(19)

    # Gate 1: RMSNorm exactness without 1+w shift
    try:
        x = rng.standard_normal((5, 32)).astype(np.float32) * 3.0
        w = np.full((32,), 0.5, dtype=np.float32)
        y = rmsn(x, w, 1e-6)
        assert y.shape == (5, 32)
        assert np.allclose(np.sqrt(np.mean(y * y, axis=-1)), 0.5, atol=1e-4)
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    # Gate 2: Proportional RoPE preserves norm and rotates only first rot_dims/2 pairs per half
    try:
        v = rng.standard_normal((4, 2, 16)).astype(np.float32)
        pos = np.array([0, 3, 7, 15], dtype=np.float32)
        vr = rope_with_pos(v, pos, 1e6, rot_dims=4, hd=16)
        assert np.allclose(np.linalg.norm(v, axis=-1), np.linalg.norm(vr, axis=-1), atol=1e-5)
        assert np.allclose(v[:, :, 2:8], vr[:, :, 2:8], atol=1e-6)
        assert np.allclose(v[:, :, 10:16], vr[:, :, 10:16], atol=1e-6)
    except Exception as exc:
        print(f"[Gate 2 FAIL] {exc}")
        failures += 1

    # Gate 3: Causal GQA isolation across prompt_slices and sliding window masking
    try:
        pos = np.array([0, 1, 2, 0, 1], dtype=np.float32)
        q = rng.standard_normal((5, 4, 8)).astype(np.float32)
        k = rng.standard_normal((5, 2, 8)).astype(np.float32)
        v = rng.standard_normal((5, 2, 8)).astype(np.float32)
        out1 = causal_gqa_attention(
            q, k, v, pos, [(0, 3), (3, 5)], nh=4, nkv=2, hd=8, rot_dims=4, theta=1e4, is_full=False, window=2
        )
        # Mutating slice 1 token 4 must not change slice 0 or slice 1 token 3;
        # Mutating token 0 in slice 0 must not change token 2 under window=2.
        v_mut = v.copy()
        v_mut[0] += 100.0
        v_mut[4] -= 50.0
        out2 = causal_gqa_attention(
            q, k, v_mut, pos, [(0, 3), (3, 5)], nh=4, nkv=2, hd=8, rot_dims=4, theta=1e4, is_full=False, window=2
        )
        assert np.allclose(out1[2], out2[2], atol=1e-5), "sliding window=2 leaked token 0 into token 2"
        assert np.allclose(out1[3], out2[3], atol=1e-5), "causal slice 1 leaked future/prior slice"
    except Exception as exc:
        print(f"[Gate 3 FAIL] {exc}")
        failures += 1

    # Gate 4: GELU-tanh monotonic & zero identity
    try:
        g = np.array([-3.0, 0.0, 2.5], dtype=np.float32)
        yo = gelu_tanh(g)
        assert abs(float(yo[1])) < 1e-6 and float(yo[2]) > 2.4
    except Exception as exc:
        print(f"[Gate 4 FAIL] {exc}")
        failures += 1

    # Gate 5: Fail closed on missing safetensors/config
    try:
        caught = False
        try:
            with tempfile.TemporaryDirectory() as td:
                run_capture(
                    Path(td) / "missing.safetensors",
                    Path(td) / "missing_config.json",
                    Path(td) / "missing_tok.json",
                    Path(td) / "out",
                    Path(td) / "work",
                    require_native=False,
                )
        except FileNotFoundError:
            caught = True
        assert caught
    except Exception as exc:
        print(f"[Gate 5 FAIL] {exc}")
        failures += 1

    # Gate 6: _FastTeacherOps offline fallback parity
    try:
        ops = _FastTeacherOps(None, require_native=False)
        xm = rng.standard_normal((6, 16)).astype(np.float32)
        wm = rng.standard_normal((12, 16)).astype(np.float32)
        assert np.allclose(ops.mm_xt(xm, wm), xm @ wm.T, atol=1e-4)
    except Exception as exc:
        print(f"[Gate 6 FAIL] {exc}")
        failures += 1

    if failures == 0:
        print("[PASS] capture_v4_clean.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser(description="Capture clean Gemma 4 31B QAT teacher states for V4 calibration.")
    ap.add_argument("--safetensors", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b/snaps_v4"))
    args = ap.parse_args()

    sf_path = resolve_safetensors_path(args.safetensors)
    run_capture(sf_path, args.config.resolve(), args.tokenizer.resolve(), args.out_dir.resolve(), args.work_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

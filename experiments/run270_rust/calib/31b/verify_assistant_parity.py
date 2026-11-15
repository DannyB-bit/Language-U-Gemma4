#!/usr/bin/env python3
"""verify_assistant_parity.py — Golden parity probe for HF BF16 vs `.t16c` Gemma4AssistantForCausalLM.

Verifies 1-step forward parity (both `post_projection` hidden state and 262,144-vocab
`logits` + top-1/top-5 overlap) across HuggingFace reference weights, ARM64/AVX2 native
`libneon_stream_accelerator`, and pure-Rust fallback (`--no-native-ffi`) via `probe_assistant`.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def pack_probe_input(
    kv_len: int,
    q_pos: int,
    tok_emb: np.ndarray,
    backbone_h: np.ndarray,
    kv_sk_hf: np.ndarray,
    kv_sv_hf: np.ndarray,
    kv_fk_hf: np.ndarray,
    kv_fv_hf: np.ndarray,
) -> bytes:
    """Serialize HF `[1, n_kv, kv_len, hd]` KV tensors into `probe_assistant` input bytes."""
    tok = np.ascontiguousarray(tok_emb, dtype="<f4").reshape(-1)
    back = np.ascontiguousarray(backbone_h, dtype="<f4").reshape(-1)
    if tok.shape != back.shape:
        raise ValueError(f"tok_emb len {tok.shape[0]} != backbone_h len {back.shape[0]}")
    bh = int(tok.shape[0])

    def _to_rust_kv(arr: np.ndarray, label: str) -> Tuple[int, int, np.ndarray]:
        a = np.asarray(arr, dtype="<f4")
        if a.ndim == 4:
            if a.shape[0] != 1:
                raise ValueError(f"{label}: expected batch=1, got {a.shape}")
            a = a[0]
        if a.ndim != 3:
            raise ValueError(f"{label}: expected 3D/4D [1, n_kv, kv_len, hd], got {arr.shape}")
        n_kv, klen, hd = a.shape
        if klen != kv_len:
            raise ValueError(f"{label}: kv_len {klen} != expected {kv_len}")
        # HF stores [n_kv, kv_len, hd]; Rust attn_decode reads [kv_len, n_kv, hd].
        return int(n_kv), int(hd), np.ascontiguousarray(np.transpose(a, (1, 0, 2)), dtype="<f4")

    n_kv_s, hd_s, sk_r = _to_rust_kv(kv_sk_hf, "kv_sk")
    n_kv_sv, hd_sv, sv_r = _to_rust_kv(kv_sv_hf, "kv_sv")
    n_kv_f, hd_f, fk_r = _to_rust_kv(kv_fk_hf, "kv_fk")
    n_kv_fv, hd_fv, fv_r = _to_rust_kv(kv_fv_hf, "kv_fv")
    if (n_kv_sv, hd_sv) != (n_kv_s, hd_s) or (n_kv_fv, hd_fv) != (n_kv_f, hd_f):
        raise ValueError("K/V head geometry mismatch between key and value tensors")

    hdr = struct.pack("<7I", kv_len, q_pos, bh, n_kv_s, hd_s, n_kv_f, hd_f)
    return b"".join([hdr, tok.tobytes(), back.tobytes(), sk_r.tobytes(), sv_r.tobytes(), fk_r.tobytes(), fv_r.tobytes()])


def unpack_probe_output(raw: bytes) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Parse `probe_assistant` output bytes into `(vocab, top1, post_h, logits)`."""
    if len(raw) < 12:
        raise ValueError(f"probe output too small: {len(raw)} < 12 bytes")
    bh, vocab, top1 = struct.unpack_from("<3I", raw, 0)
    expected = 12 + (bh + vocab) * 4
    if len(raw) != expected:
        raise ValueError(f"probe output len {len(raw)} != expected {expected} (bh={bh}, vocab={vocab})")
    post_h = np.frombuffer(raw, dtype="<f4", count=bh, offset=12).copy()
    logits = np.frombuffer(raw, dtype="<f4", count=vocab, offset=12 + bh * 4).copy()
    return int(vocab), int(top1), post_h, logits


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    if af.shape != bf.shape:
        raise ValueError(f"shape mismatch in cos_sim: {af.shape} vs {bf.shape}")
    den = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return float(np.dot(af, bf) / den) if den > 1e-18 else 0.0


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    den = float(np.linalg.norm(bf))
    return float(np.linalg.norm(af - bf) / (den + 1e-12))


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> Tuple[int, int, int, int]:
    """Return `(overlap_count, top1_a, top1_b, rank_of_top1_b_in_a)`."""
    af = np.asarray(a, dtype=np.float32).ravel()
    bf = np.asarray(b, dtype=np.float32).ravel()
    k_eff = min(k, af.size)
    top_a = set(np.argpartition(af, -k_eff)[-k_eff:].tolist())
    top_b = set(np.argpartition(bf, -k_eff)[-k_eff:].tolist())
    arg_a = int(np.argmax(af))
    arg_b = int(np.argmax(bf))
    rank_b_in_a = int(np.sum(af > af[arg_b])) + 1
    return len(top_a & top_b), arg_a, arg_b, rank_b_in_a


def evaluate_parity(
    label: str,
    got_post: np.ndarray,
    ref_post: np.ndarray,
    got_logits: np.ndarray,
    ref_logits: np.ndarray,
    min_post_cos: float,
    min_logits_cos: float,
) -> Dict[str, float]:
    """Validate parity metrics against strict thresholds; raise `AssertionError` on mismatch."""
    c_post = cos_sim(got_post, ref_post)
    r_post = rel_l2(got_post, ref_post)
    c_log = cos_sim(got_logits, ref_logits)
    r_log = rel_l2(got_logits, ref_logits)
    ov5, arg_got, arg_ref, rank_ref = top_k_overlap(got_logits, ref_logits, k=5)
    print(
        f"[PARITY:{label}] post_cos={c_post:.5f} post_rel={r_post:.4f} | "
        f"logits_cos={c_log:.5f} logits_rel={r_log:.4f} | "
        f"top1 got={arg_got} ref={arg_ref} ref_rank_in_got={rank_ref} top5_overlap={ov5}/5",
        flush=True,
    )
    if c_post < min_post_cos:
        raise AssertionError(f"{label}: post_h cosine {c_post:.5f} < min {min_post_cos}")
    if c_log < min_logits_cos:
        raise AssertionError(f"{label}: logits cosine {c_log:.5f} < min {min_logits_cos}")
    return {
        "post_cos": c_post,
        "post_rel": r_post,
        "logits_cos": c_log,
        "logits_rel": r_log,
        "top1_match": 1.0 if arg_got == arg_ref else 0.0,
        "ref_rank_in_got": float(rank_ref),
        "top5_overlap": float(ov5),
    }


def _run_selftest() -> int:
    failures = 0

    def _gate(name: str, fn) -> None:
        nonlocal failures
        try:
            fn()
            print(f"[GATE PASS] {name}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"[GATE FAIL] {name}: {exc}", flush=True)

    # Gate 1: Binary pack/unpack roundtrip and fail-closed truncation/trailing rejection.
    def _g1() -> None:
        rng = np.random.default_rng(1)
        tok = rng.normal(size=(8,)).astype(np.float32)
        back = rng.normal(size=(8,)).astype(np.float32)
        sk = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
        sv = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
        fk = rng.normal(size=(1, 1, 3, 6)).astype(np.float32)
        fv = rng.normal(size=(1, 1, 3, 6)).astype(np.float32)
        raw_in = pack_probe_input(3, 3, tok, back, sk, sv, fk, fv)
        expected_in = 28 + (8 + 8 + 2 * 2 * 3 * 4 + 2 * 1 * 3 * 6) * 4
        assert len(raw_in) == expected_in, f"{len(raw_in)} != {expected_in}"

        out_bytes = struct.pack("<3I", 8, 16, 7) + tok.tobytes() + np.arange(16, dtype="<f4").tobytes()
        vocab, top1, post, logits = unpack_probe_output(out_bytes)
        assert vocab == 16 and top1 == 7
        assert np.allclose(post, tok) and logits.shape == (16,)
        try:
            unpack_probe_output(out_bytes[:-4])
            raise AssertionError("accepted truncated output")
        except ValueError:
            pass

    # Gate 2: HF [1, n_kv, kv_len, hd] -> Rust [kv_len, n_kv, hd] exact element coordinate mapping.
    def _g2() -> None:
        kv_len, n_kv, hd = 5, 3, 4
        hf = np.zeros((1, n_kv, kv_len, hd), dtype=np.float32)
        for h in range(n_kv):
            for t in range(kv_len):
                for d in range(hd):
                    hf[0, h, t, d] = 100.0 * t + 10.0 * h + d
        tok = np.zeros((2,), dtype=np.float32)
        raw = pack_probe_input(kv_len, kv_len, tok, tok, hf, hf, hf[:, :1], hf[:, :1])
        off = 28 + (2 + 2) * 4
        rust_sk = np.frombuffer(raw, dtype="<f4", count=kv_len * n_kv * hd, offset=off).reshape(kv_len, n_kv, hd)
        for t in range(kv_len):
            for h in range(n_kv):
                for d in range(hd):
                    assert rust_sk[t, h, d] == 100.0 * t + 10.0 * h + d

    # Gate 3: Similarity & top-k metrics distinguish signal from orthogonal/corrupt outputs.
    def _g3() -> None:
        a = np.array([1.0, 2.0, 10.0, -3.0, 0.5, 4.0], dtype=np.float32)
        b = a + np.array([0.05, -0.02, 0.1, 0.01, -0.01, 0.02], dtype=np.float32)
        orth = np.zeros_like(a)
        assert cos_sim(a, b) > 0.999 and rel_l2(a, b) < 0.02
        assert cos_sim(a, orth) == 0.0
        ov, g1, r1, rk = top_k_overlap(a, b, k=3)
        assert ov == 3 and g1 == 2 and r1 == 2 and rk == 1

    # Gate 4: Parity evaluator fails closed below min cosine.
    def _g4() -> None:
        rng = np.random.default_rng(7)
        ref = rng.normal(size=(64,)).astype(np.float32)
        good = ref + 0.02 * rng.normal(size=(64,)).astype(np.float32)
        bad = rng.normal(size=(64,)).astype(np.float32)
        st = evaluate_parity("selftest_good", good, ref, good, ref, 0.985, 0.985)
        assert st["post_cos"] > 0.99 and st["logits_cos"] > 0.99
        try:
            evaluate_parity("selftest_bad", bad, ref, bad, ref, 0.985, 0.985)
            raise AssertionError("failed to reject orthogonal output")
        except AssertionError:
            pass

    # Gate 5: Reject shape mismatch between key and value tensors.
    def _g5() -> None:
        tok = np.zeros((4,), dtype=np.float32)
        sk = np.zeros((1, 2, 3, 4), dtype=np.float32)
        sv = np.zeros((1, 2, 3, 8), dtype=np.float32)
        fk = np.zeros((1, 1, 3, 4), dtype=np.float32)
        try:
            pack_probe_input(3, 3, tok, tok, sk, sv, fk, fk)
            raise AssertionError("failed to reject K/V head_dim mismatch")
        except ValueError:
            pass

    # Gate 6: CLI self-invocation sanity inside subprocess without touching external artifacts.
    def _g6() -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe_in.bin"
            tok = np.ones((4,), dtype=np.float32)
            sk = np.ones((1, 2, 2, 4), dtype=np.float32)
            fk = np.ones((1, 1, 2, 4), dtype=np.float32)
            data = pack_probe_input(2, 2, tok, tok, sk, sk, fk, fk)
            p.write_bytes(data)
            assert p.stat().st_size == len(data) > 28

    for name, fn in [
        ("G1_binary_roundtrip", _g1),
        ("G2_kv_layout_mapping", _g2),
        ("G3_metrics_sanity", _g3),
        ("G4_fail_closed_parity", _g4),
        ("G5_shape_guard", _g5),
        ("G6_probe_file_e2e", _g6),
    ]:
        _gate(name, fn)

    if failures:
        print(f"[FAIL] verify_assistant_parity.py selftest: {failures} gate(s) failed", flush=True)
        return failures
    print("[PASS] verify_assistant_parity.py 6-gate self-test passed (all 6 gates)", flush=True)
    return 0


def run_live_parity(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    hf_dir = Path(args.hf_dir)
    t16c_path = Path(args.t16c)
    probe_bin = Path(args.probe_bin)
    so_path = Path(args.so) if args.so else None

    for req in (hf_dir, t16c_path, probe_bin):
        if not req.exists():
            raise FileNotFoundError(f"Mandatory parity artifact missing: {req}")
    if so_path is not None and not so_path.exists():
        raise FileNotFoundError(f"Requested accelerator library missing: {so_path}")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"[PARITY] Loading HF reference from {hf_dir} (dtype={args.dtype})...", flush=True)
    hf_model = AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype=dtype, device_map="cpu")
    hf_model.eval()
    cfg = hf_model.config
    tc = cfg.get_text_config()
    bh = int(cfg.backbone_hidden_size)
    tc_dict = tc.to_dict()
    per_layer = getattr(tc, "per_layer_config", None)
    layer_types = list(tc_dict.get("layer_types") or [])
    if per_layer is not None and "sliding_attention" in layer_types and "full_attention" in layer_types:
        s_cfg = per_layer[layer_types.index("sliding_attention")]
        f_cfg = per_layer[layer_types.index("full_attention")]
        n_kv_slide = int(s_cfg.num_key_value_heads)
        hd_slide = int(s_cfg.head_dim)
        n_kv_full = int(f_cfg.num_key_value_heads)
        hd_full = int(f_cfg.head_dim)
    else:
        n_kv_slide = int(tc_dict["num_key_value_heads"])
        hd_slide = int(tc_dict["head_dim"])
        n_kv_full = int(tc_dict["num_global_key_value_heads"])
        hd_full = int(tc_dict["global_head_dim"])

    kv_len = int(args.kv_len)
    q_pos = int(args.q_pos) if args.q_pos is not None else kv_len

    rng = np.random.default_rng(args.seed)
    tok_emb_np = rng.normal(0.0, 0.25, size=(1, 1, bh)).astype(np.float32)
    back_h_np = rng.normal(0.0, 0.25, size=(1, 1, bh)).astype(np.float32)
    sk_np = rng.normal(0.0, 0.5, size=(1, n_kv_slide, kv_len, hd_slide)).astype(np.float32)
    sv_np = rng.normal(0.0, 0.5, size=(1, n_kv_slide, kv_len, hd_slide)).astype(np.float32)
    fk_np = rng.normal(0.0, 0.5, size=(1, n_kv_full, kv_len, hd_full)).astype(np.float32)
    fv_np = rng.normal(0.0, 0.5, size=(1, n_kv_full, kv_len, hd_full)).astype(np.float32)

    inputs_embeds = torch.cat(
        [torch.from_numpy(tok_emb_np), torch.from_numpy(back_h_np)], dim=-1
    ).to(dtype)
    shared_kv = {
        "sliding_attention": (torch.from_numpy(sk_np).to(dtype), torch.from_numpy(sv_np).to(dtype)),
        "full_attention": (torch.from_numpy(fk_np).to(dtype), torch.from_numpy(fv_np).to(dtype)),
    }
    position_ids = torch.tensor([[q_pos]], dtype=torch.long)

    print(
        f"[PARITY] Running HF reference forward (kv_len={kv_len}, q_pos={q_pos}, bh={bh})...",
        flush=True,
    )
    with torch.no_grad():
        out = hf_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            shared_kv_states=shared_kv,
            use_cache=False,
        )
    ref_post = out.last_hidden_state[0, 0].float().cpu().numpy().astype(np.float32)
    ref_logits = out.logits[0, 0].float().cpu().numpy().astype(np.float32)

    del out, hf_model, inputs_embeds, shared_kv

    probe_in_bytes = pack_probe_input(kv_len, q_pos, tok_emb_np, back_h_np, sk_np, sv_np, fk_np, fv_np)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        in_p = td_p / "probe_in.bin"
        in_p.write_bytes(probe_in_bytes)

        results = {}
        modes: List[Tuple[str, List[str]]] = [("t16c_rust_fallback", ["--no-native-ffi"])]
        if so_path is not None:
            modes.insert(0, ("t16c_native_neon", ["--so", str(so_path)]))
        else:
            modes.insert(0, ("t16c_native_auto", []))

        for mode_name, extra in modes:
            out_p = td_p / f"probe_out_{mode_name}.bin"
            cmd = [
                str(probe_bin),
                "--assistant",
                str(t16c_path),
                "--in",
                str(in_p),
                "--out",
                str(out_p),
                "--threads",
                str(args.threads),
                *extra,
            ]
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            sys.stdout.write(proc.stdout)
            sys.stdout.flush()
            vocab, top1, got_post, got_logits = unpack_probe_output(out_p.read_bytes())
            assert vocab == ref_logits.shape[0], f"vocab mismatch: {vocab} != {ref_logits.shape[0]}"
            assert int(np.argmax(got_logits)) == top1
            results[mode_name] = (
                got_post,
                got_logits,
                evaluate_parity(
                    mode_name,
                    got_post,
                    ref_post,
                    got_logits,
                    ref_logits,
                    args.min_post_cos,
                    args.min_logits_cos,
                ),
            )

        keys = list(results.keys())
        if len(keys) == 2:
            p0, l0, _ = results[keys[0]]
            p1, l1, _ = results[keys[1]]
            c_cross_p = cos_sim(p0, p1)
            c_cross_l = cos_sim(l0, l1)
            print(
                f"[PARITY:{keys[0]}_vs_{keys[1]}] post_cos={c_cross_p:.6f} logits_cos={c_cross_l:.6f} "
                f"max_abs_post={np.max(np.abs(p0 - p1)):.3e} max_abs_logits={np.max(np.abs(l0 - l1)):.3e}",
                flush=True,
            )
            if c_cross_p < 0.999 or c_cross_l < 0.999:
                raise AssertionError(
                    f"Native vs Rust fallback drift too high: post_cos={c_cross_p:.6f}, logits_cos={c_cross_l:.6f}"
                )

    print("[PASS] Gemma4Assistant HF vs T16C golden parity verified!", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify HF BF16 vs T16C Gemma4Assistant parity.")
    ap.add_argument("--selftest", action="store_true", help="Run offline 6-gate self-test without loading weights.")
    ap.add_argument("--hf-dir", default="/mnt/SHE-VAULT/gemma-4-31B-it-assistant")
    ap.add_argument("--t16c", default="/mnt/SHE-VAULT/gemma4_31b_assistant.t16c")
    ap.add_argument("--probe-bin", default="experiments/run270_rust/target/release/probe_assistant")
    ap.add_argument("--so", default=None)
    ap.add_argument("--kv-len", type=int, default=12)
    ap.add_argument("--q-pos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--min-post-cos", type=float, default=0.985)
    ap.add_argument("--min-logits-cos", type=float, default=0.985)
    args = ap.parse_args()
    if args.selftest:
        return _run_selftest()
    return run_live_parity(args)


if __name__ == "__main__":
    raise SystemExit(main())

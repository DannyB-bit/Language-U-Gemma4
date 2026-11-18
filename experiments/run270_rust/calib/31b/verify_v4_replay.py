#!/usr/bin/env python3
"""
2x2 Speculative Assistant Replay & Shared-KV Verifier for Gemma 4 31B V4.

Replays `L58..L59` through `gemma4_31b_v4.t16c` (+ optional `.c8d`) via bounded
open/seek/read (`_SeekT16CReader`) on the two clean replay slices (`test_2p2` and
`test_fr` stored in `teacher_bridge_v4.npz`), compares Teacher (`RefH`, `RefKV`) vs
Student (`V4H`, `V4KV`) across the 2x2 grid (`RefH_RefKV`, `V4H_RefKV`, `RefH_V4KV`,
`V4H_V4KV`) using `probe_assistant`, and checks step-by-step greedy draft parity.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import ctypes
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capture_v4_clean import causal_gqa_attention, gelu_tanh, rmsn, rope_with_pos
from pack_t16c_v3 import SafetensorsReader, resolve_safetensors_path
import train_c8d_pattern_bank as c8d
from train_v4_bridge import _SeekT16CReader
from verify_assistant_parity import pack_probe_input, unpack_probe_output


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute vector/tensor cosine similarity across all elements."""
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def score_replay_agreement(ref_tokens: Sequence[int], cand_tokens: Sequence[int]) -> Tuple[bool, int, int]:
    """Return `(step0_match, longest_prefix_match, exact_slot_matches)`."""
    n = min(len(ref_tokens), len(cand_tokens))
    if n == 0:
        return (False, 0, 0)
    prefix = 0
    for r, c in zip(ref_tokens, cand_tokens):
        if r == c:
            prefix += 1
        else:
            break
    exact = sum(1 for r, c in zip(ref_tokens, cand_tokens) if r == c)
    return (ref_tokens[0] == cand_tokens[0], prefix, exact)


def top1_bf16_logit_token(sr: SafetensorsReader, embed_key: str, h_vec: np.ndarray) -> int:
    """Compute exact BF16 `embed_tokens` matrix-vector top-1 argmax via streaming seek/read."""
    info = sr.hdr[embed_key]
    vocab, d = int(info["shape"][0]), int(info["shape"][1])
    base_off = sr.data_base + int(info["data_offsets"][0])
    h32 = np.ascontiguousarray(h_vec, dtype=np.float32).reshape(d)
    best_id = 0
    best_val = -1.0e30
    chunk = 16384
    for r0 in range(0, vocab, chunk):
        take = min(chunk, vocab - r0)
        sr._fp.seek(base_off + r0 * d * 2)
        raw = sr._fp.read(take * d * 2)
        w = ((np.frombuffer(raw, np.uint16).reshape(take, d).astype(np.uint32) << 16).view(np.float32)).copy()
        logits = 30.0 * np.tanh((w @ h32) / 30.0)
        loc_i = int(np.argmax(logits))
        loc_v = float(logits[loc_i])
        if loc_v > best_val:
            best_val = loc_v
            best_id = r0 + loc_i
    return best_id


def run_probe_draft(
    probe_bin: Path,
    ast_path: Path,
    so_path: Path,
    sk: _SeekT16CReader,
    esc: float,
    Ls: int,
    h0: np.ndarray,
    k58: np.ndarray,
    v58: np.ndarray,
    k59: np.ndarray,
    v59: np.ndarray,
    y0: int,
    n_steps: int,
    tk,
) -> List[Tuple[int, str]]:
    """Run `n_steps` autoregressive Assistant draft steps via `probe_assistant`."""
    sk_hf = np.transpose(k58, (1, 0, 2))[None, ...]
    sv_hf = np.transpose(v58, (1, 0, 2))[None, ...]
    fk_hf = np.transpose(k59, (1, 0, 2))[None, ...]
    fv_hf = np.transpose(v59, (1, 0, 2))[None, ...]
    cur_tok = y0
    cur_h = h0.copy()
    drafts: List[Tuple[int, str]] = []
    for _ in range(n_steps):
        temb = (sk.read_t16c_rows("model.language_model.embed_tokens.weight", [cur_tok])[0] * esc).astype(np.float32)
        blob = pack_probe_input(Ls, Ls, temb, cur_h, sk_hf, sv_hf, fk_hf, fv_hf)
        with tempfile.NamedTemporaryFile(suffix=".bin") as fin, tempfile.NamedTemporaryFile(suffix=".bin") as fout:
            fin.write(blob)
            fin.flush()
            subprocess.run(
                [str(probe_bin), "--assistant", str(ast_path), "--so", str(so_path), "--in", fin.name, "--out", fout.name, "--threads", "4"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _, _, post_h, logits = unpack_probe_output(Path(fout.name).read_bytes())
        for btid in (100, 101, 45518):
            logits[btid] = -1e9
        if not (236770 <= cur_tok <= 236820):
            logits[cur_tok] = -1e9
        pred = int(np.argmax(logits))
        drafts.append((pred, tk.decode([pred])))
        cur_tok = pred
        cur_h = post_h
    return drafts


def run_2x2_replay(
    t16c_path: Path,
    c8d_path: Optional[Path],
    bridge_npz: Path,
    student_npz: Path,
    sf_path: Path,
    cfg_path: Path,
    tok_path: Path,
    ast_path: Path,
    so_path: Path,
    probe_bin: Path,
    n_steps: int = 5,
) -> dict:
    """Execute 2x2 replay verification on `test_2p2` and `test_fr` slices from `teacher_bridge_v4.npz`."""
    for p in (t16c_path, bridge_npz, student_npz, sf_path, cfg_path, tok_path, ast_path, so_path, probe_bin):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required V4 replay artifact: {p}")
    if c8d_path is not None and not c8d_path.is_file():
        raise FileNotFoundError(f"Missing C8DB artifact: {c8d_path}")

    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(tok_path))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["text_config"]
    d = int(cfg["hidden_size"])
    eps = float(cfg["rms_norm_eps"])
    esc = math.sqrt(float(d))
    nh = int(cfg["num_attention_heads"])
    n_kv_s = int(cfg["num_key_value_heads"])
    n_kv_f = int(cfg["num_global_key_value_heads"])
    hd_s = int(cfg["head_dim"])
    hd_f = int(cfg["global_head_dim"])
    k_eq_v = bool(cfg.get("attention_k_eq_v", False))
    layer_types = cfg["layer_types"]
    rp = cfg["rope_parameters"]
    theta_g, part_g = float(rp["full_attention"]["rope_theta"]), float(rp["full_attention"].get("partial_rotary_factor", 1.0))
    theta_l, part_l = float(rp["sliding_attention"]["rope_theta"]), float(rp["sliding_attention"].get("partial_rotary_factor", 1.0))
    win = int(cfg["sliding_window"])

    with np.load(bridge_npz, allow_pickle=False) as bz:
        bridge = {k: bz[k] for k in bz.files}
    with np.load(student_npz, allow_pickle=False) as sz:
        h_stud_before_58 = sz["h_stud_before_L58"].astype(np.float32)

    fn_orig = bridge["fn_orig"].astype(np.float32)
    c8d_norms, c8d_entries = c8d.read_c8db_file(c8d_path) if c8d_path is not None else ({}, {})
    fn_stud = c8d_norms.get("model.language_model.norm.weight", fn_orig)
    nlib = c8d.load_native_lib(so_path)
    if nlib is None:
        raise FileNotFoundError(f"Could not load {so_path}")

    results: Dict[str, dict] = {}
    with SafetensorsReader(sf_path) as sr, _SeekT16CReader(t16c_path, nlib) as sk:
        slices = [
            ("test_2p2", tuple(int(v) for v in bridge["test2_bounds"].tolist())),
            ("test_fr", tuple(int(v) for v in bridge["testf_bounds"].tolist())),
        ]
        for tag, (s0, s1) in slices:
            Ls = s1 - s0
            pos_s = np.arange(Ls, dtype=np.float32)

            kt58_raw = bridge["kt_58_raw"][s0:s1].reshape(Ls, n_kv_s, hd_s)
            vt58_raw = bridge["vt_58_raw"][s0:s1].reshape(Ls, n_kv_s, hd_s)
            kn58 = sr.read_f32_tensor("model.language_model.layers.58.self_attn.k_norm.weight")
            kt58 = rope_with_pos(rmsn(kt58_raw, kn58, eps), pos_s, theta_l, int(round(hd_s * part_l)), hd_s)
            vt58 = rmsn(vt58_raw, None, eps)

            kt59_raw = bridge["kt_59_raw"][s0:s1].reshape(Ls, n_kv_f, hd_f)
            kn59 = sr.read_f32_tensor("model.language_model.layers.59.self_attn.k_norm.weight")
            kt59 = rope_with_pos(rmsn(kt59_raw, kn59, eps), pos_s, theta_g, int(round(hd_f * part_g)), hd_f)
            vt59 = rmsn(kt59_raw, None, eps)
            ht_f = rmsn(bridge["h_teach_L59"][s0:s1], fn_orig, eps)[-1]

            h_s = h_stud_before_58[s0:s1].copy()
            ks58_out = vs58_out = ks59_out = vs59_out = None
            for l in (58, 59):
                pfx = f"model.language_model.layers.{l}."
                is_full = layer_types[l] == "full_attention"
                nkv = n_kv_f if is_full else n_kv_s
                hd = hd_f if is_full else hd_s
                theta = theta_g if is_full else theta_l
                rot_d = int(round(hd * (part_g if is_full else part_l)))
                g_in = sr.read_f32_tensor(pfx + "input_layernorm.weight")
                qn = sr.read_f32_tensor(pfx + "self_attn.q_norm.weight")
                kn = sr.read_f32_tensor(pfx + "self_attn.k_norm.weight")
                g_pa = sr.read_f32_tensor(pfx + "post_attention_layernorm.weight")
                g_pf = sr.read_f32_tensor(pfx + "pre_feedforward_layernorm.weight")
                g_pff = sr.read_f32_tensor(pfx + "post_feedforward_layernorm.weight")
                ls = float(np.asarray(sr.read_f32_tensor(pfx + "layer_scalar")).reshape(-1)[0])
                hs_in = rmsn(h_s, g_in, eps)

                def _p(suf: str, xa: np.ndarray) -> np.ndarray:
                    mk = pfx + suf + ".weight"
                    yb = sk.project_batch(mk, xa)
                    if mk in c8d_entries:
                        ent = c8d_entries[mk]
                        yb = yb + c8d.eval_quantized_adapter_np(
                            xa, ent["centroids"], ent["experts"], ent["top_k"], ent["temperature"], ent["gate_scale"]
                        )
                    return yb

                qs = rmsn(_p("self_attn.q_proj", hs_in).reshape(Ls, nh, hd), qn, eps)
                ks_raw = _p("self_attn.k_proj", hs_in)
                ks = rmsn(ks_raw.reshape(Ls, nkv, hd), kn, eps)
                vs_raw = ks_raw if (is_full and k_eq_v) else _p("self_attn.v_proj", hs_in)
                vs = rmsn(vs_raw.reshape(Ls, nkv, hd), None, eps)
                ks_r = rope_with_pos(ks, pos_s, theta, rot_d, hd)
                if l == 58:
                    ks58_out, vs58_out = ks_r.copy(), vs.copy()
                else:
                    ks59_out, vs59_out = ks_r.copy(), vs.copy()
                ctx = causal_gqa_attention(
                    qs.reshape(Ls, nh * hd), ks.reshape(Ls, nkv * hd), vs.reshape(Ls, nkv * hd),
                    pos_s, [(0, Ls)], nh=nh, nkv=nkv, hd=hd, rot_dims=rot_d, theta=theta, is_full=is_full, window=win,
                )
                xm = h_s + rmsn(_p("self_attn.o_proj", ctx), g_pa, eps)
                hm = rmsn(xm, g_pf, eps)
                h_s = ((xm + rmsn(_p("mlp.down_proj", gelu_tanh(_p("mlp.gate_proj", hm)) * _p("mlp.up_proj", hm)), g_pff, eps)) * ls).astype(np.float32)

            assert ks58_out is not None and vs58_out is not None and ks59_out is not None and vs59_out is not None
            hs_norm = rmsn(h_s, fn_stud, eps)
            if "model.language_model.norm.post_cell60" in c8d_entries:
                e60 = c8d_entries["model.language_model.norm.post_cell60"]
                hs_norm = hs_norm + c8d.eval_quantized_adapter_np(
                    hs_norm, e60["centroids"], e60["experts"], e60["top_k"], e60["temperature"], e60["gate_scale"]
                )
            hs_f = hs_norm[-1]

            y0_t = top1_bf16_logit_token(sr, "model.language_model.embed_tokens.weight", ht_f)
            y0_s = top1_bf16_logit_token(sr, "model.language_model.embed_tokens.weight", hs_f)
            c_h = cos_sim(ht_f, hs_f)
            c_k58 = cos_sim(kt58, ks58_out)
            c_v58 = cos_sim(vt58, vs58_out)
            c_k59 = cos_sim(kt59, ks59_out)
            print(
                f"=== {tag} (Ls={Ls}) y0_T={y0_t}:{tk.decode([y0_t])!r} y0_S={y0_s}:{tk.decode([y0_s])!r} "
                f"cos_h={c_h:.5f} cos_k58={c_k58:.5f} cos_v58={c_v58:.5f} cos_k59={c_k59:.5f} ===",
                flush=True,
            )

            mode_drafts: Dict[str, List[Tuple[int, str]]] = {}
            for mname, h0, k58, v58, k59, v59 in [
                ("1_RefH_RefKV", ht_f, kt58, vt58, kt59, vt59),
                ("2_V4H_RefKV", hs_f, kt58, vt58, kt59, vt59),
                ("3_RefH_V4KV", ht_f, ks58_out, vs58_out, ks59_out, vs59_out),
                ("4_V4H_V4KV", hs_f, ks58_out, vs58_out, ks59_out, vs59_out),
            ]:
                dr = run_probe_draft(probe_bin, ast_path, so_path, sk, esc, Ls, h0, k58, v58, k59, v59, y0_t, n_steps, tk)
                mode_drafts[mname] = dr
                ref_ids = [t for t, _ in mode_drafts["1_RefH_RefKV"]]
                s0_ok, pref_m, ex_m = score_replay_agreement(ref_ids, [t for t, _ in dr])
                print(f"  {mname}: step0_match={s0_ok} prefix={pref_m}/{n_steps} exact={ex_m}/{n_steps} {dr}", flush=True)
            results[tag] = {
                "cos_h": c_h,
                "cos_k58": c_k58,
                "cos_v58": c_v58,
                "cos_k59": c_k59,
                "y0_teach": y0_t,
                "y0_stud": y0_s,
                "drafts": mode_drafts,
            }
    return results


def _self_test() -> int:
    """Run 6 fail-closed unit gates on C8DB roundtrip, agreement scoring, and error checks."""
    failures = 0
    rng = np.random.default_rng(7)
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rt.c8d"
            norm_a = rng.standard_normal(16).astype(np.float32)
            cents = rng.standard_normal((2, 8)).astype(np.float32)
            experts = [
                (
                    rng.standard_normal(16).astype(np.float32),
                    rng.standard_normal(4).astype(np.float32),
                    rng.integers(-64, 64, size=(16, 4), dtype=np.int8),
                    rng.integers(-64, 64, size=(4, 16), dtype=np.int8),
                )
                for _ in range(2)
            ]
            entries = [{
                "name": "test.entry", "d_out": 16, "d_in": 16, "rank": 4, "top_k": 2,
                "c6d": (1, 2, 3, 4), "temperature": 0.5, "gate_scale": 1.0,
                "centroids": cents, "expert_radicals": [(1, 2, 3, 4), (2, 3, 4, 5)], "experts": experts,
            }]
            c8d.write_c8db_file(p, [("test.norm", norm_a)], entries)
            rn, re = c8d.read_c8db_file(p)
            assert np.allclose(rn["test.norm"], norm_a)
            assert np.allclose(re["test.entry"]["centroids"], cents)
            x = rng.standard_normal((3, 16)).astype(np.float32)
            y0 = c8d.eval_quantized_adapter_np(x, cents, experts, 2, 0.5, 1.0)
            y1 = c8d.eval_quantized_adapter_np(x, re["test.entry"]["centroids"], re["test.entry"]["experts"], 2, 0.5, 1.0)
            assert np.allclose(y0, y1)
    except Exception as exc:
        print(f"[Gate 1-2 FAIL] {exc}")
        failures += 1

    try:
        a = rng.standard_normal((4, 32)).astype(np.float32)
        assert abs(cos_sim(a, a) - 1.0) < 1e-6
        s0, pref, ex = score_replay_agreement([10, 20, 30, 40], [10, 20, 99, 40])
        assert s0 is True and pref == 2 and ex == 3
        s0b, prefb, exb = score_replay_agreement([10, 20], [11, 20])
        assert s0b is False and prefb == 0 and exb == 1
    except Exception as exc:
        print(f"[Gate 3-4 FAIL] {exc}")
        failures += 1

    try:
        blob = pack_probe_input(2, 2, np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32), np.zeros((1, 2, 2, 4), dtype=np.float32), np.zeros((1, 2, 2, 4), dtype=np.float32), np.zeros((1, 1, 2, 8), dtype=np.float32), np.zeros((1, 1, 2, 8), dtype=np.float32))
        assert len(blob) > 0
        caught = False
        try:
            run_2x2_replay(Path("/no/v4.t16c"), None, Path("/no/b.npz"), Path("/no/s.npz"), Path("/no/m.sf"), Path("/no/c.json"), Path("/no/t.json"), Path("/no/a.t16c"), Path("/no/lib.so"), Path("/no/probe"))
        except FileNotFoundError:
            caught = True
        assert caught
    except Exception as exc:
        print(f"[Gate 5-6 FAIL] {exc}")
        failures += 1

    if failures == 0:
        print("[PASS] verify_v4_replay.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()
    ap = argparse.ArgumentParser(description="Verify V4 2x2 speculative replay & shared-KV recovery.")
    ap.add_argument("--t16c", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v4.t16c"))
    ap.add_argument("--c8d", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v4_c8d.c8d"))
    ap.add_argument("--no-c8d", action="store_true")
    ap.add_argument("--bridge-npz", type=Path, default=Path("/mnt/SHE-VAULT/work31b/snaps_v4/teacher_bridge_v4.npz"))
    ap.add_argument("--student-npz", type=Path, default=Path("/mnt/SHE-VAULT/work31b/student_state_v4.npz"))
    ap.add_argument("--safetensors", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--assistant", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_assistant.t16c"))
    ap.add_argument("--so", type=Path, default=Path("/home/astronautshe/gemma4_verify/neon_runner/libneon_stream_accelerator.so"))
    ap.add_argument("--probe-bin", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/target/release/probe_assistant"))
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    run_2x2_replay(
        args.t16c,
        None if args.no_c8d else args.c8d,
        args.bridge_npz,
        args.student_npz,
        resolve_safetensors_path(args.safetensors),
        args.config,
        args.tokenizer,
        args.assistant,
        args.so,
        args.probe_bin,
        n_steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

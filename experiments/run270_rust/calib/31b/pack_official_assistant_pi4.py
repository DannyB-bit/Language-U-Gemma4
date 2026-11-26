#!/usr/bin/env python3
"""Pack official `google/gemma-4-31B-it-assistant` body without embed_tokens on Pi 4 (Target-Embed Sharing).

Allows immediate Pi 4 `--cortical-cascade` & Target-Embed Sharing validation (~56 MB resident assistant)
using existing V9.2 `.t16c` (with identity or loaded `perm_pi` folded into pre/post projections).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_assistant_v7_byte_head as bhmod  # noqa: E402
from pack_t16c_v3 import SafetensorsReader, _write_synthetic_safetensors  # noqa: E402


def permute_assistant_safetensors(src_sf: Path, dst_sf: Path, perm_pi: np.ndarray, d_mod: int = 5376) -> None:
    """Rewrite `pre_projection.weight` and `post_projection.weight` under `perm_pi` into F32 `dst_sf`."""
    p2 = np.concatenate([perm_pi, d_mod + perm_pi])
    tensors: Dict[str, np.ndarray] = {}
    with SafetensorsReader(src_sf) as sr:
        for k in sr.hdr.keys():
            if k in ("__metadata__", "model.embed_tokens.weight"):
                continue
            arr = sr.read_f32_tensor(k)
            if k == "pre_projection.weight" and arr.shape[1] == 2 * d_mod:
                arr = np.ascontiguousarray(arr[:, p2])
            elif k == "post_projection.weight" and arr.shape[0] == d_mod:
                arr = np.ascontiguousarray(arr[perm_pi, :])
            tensors[k] = arr
    _write_synthetic_safetensors(dst_sf, tensors)


def run_selftest() -> int:
    """6-gate CPU verification of embed-free Q2/Q4 packing + perm_pi folding."""
    import pack_t16c_v3 as p3
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gates 1-6 on synthetic 2-layer assistant body
    with tempfile.TemporaryDirectory() as td:
        from safetensors.numpy import save_file

        d_mod, d_ast, ff = 64, 64, 64
        src = Path(td) / "ast.safetensors"
        perm_sf = Path(td) / "ast_perm.safetensors"
        out_q2 = Path(td) / "ast_q2.t16c"
        rng = np.random.default_rng(7)
        sd = {
            "pre_projection.weight": rng.normal(0, 0.05, (d_ast, 2 * d_mod)).astype(np.float32),
            "post_projection.weight": rng.normal(0, 0.05, (d_mod, d_ast)).astype(np.float32),
            "model.embed_tokens.weight": rng.normal(0, 0.05, (128, d_ast)).astype(np.float32),
            "model.norm.weight": np.ones((d_ast,), dtype=np.float32),
        }
        for l in range(2):
            p = f"model.layers.{l}."
            sd[p + "self_attn.q_proj.weight"] = rng.normal(0, 0.05, (256, d_ast)).astype(np.float32)
            sd[p + "self_attn.o_proj.weight"] = rng.normal(0, 0.05, (d_ast, 256)).astype(np.float32)
            sd[p + "mlp.gate_proj.weight"] = rng.normal(0, 0.05, (ff, d_ast)).astype(np.float32)
            sd[p + "mlp.up_proj.weight"] = rng.normal(0, 0.05, (ff, d_ast)).astype(np.float32)
            sd[p + "mlp.down_proj.weight"] = rng.normal(0, 0.05, (d_ast, ff)).astype(np.float32)
            sd[p + "self_attn.q_norm.weight"] = np.ones((64 if l == 0 else 64,), dtype=np.float32)
            for nk in ("input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"):
                sd[p + nk] = np.ones((d_ast,), dtype=np.float32)
            sd[p + "layer_scalar"] = np.array([1.0], dtype=np.float32)
        save_file(sd, str(src))
        gate("Gate 1: Synthetic Safetensors Written", src.is_file())

        perm = rng.permutation(d_mod).astype(np.int64)
        permute_assistant_safetensors(src, perm_sf, perm, d_mod=d_mod)
        gate("Gate 2: Permuted Embed-Free Safetensors Created", perm_sf.is_file())

        with SafetensorsReader(perm_sf) as sr:
            keys = {k for k in sr.hdr.keys() if k != "__metadata__"}
            x = rng.normal(0, 1.0, (2 * d_mod,)).astype(np.float32)
            p2 = np.concatenate([perm, d_mod + perm])
            y_orig = sd["post_projection.weight"] @ (sd["pre_projection.weight"] @ x)
            y_perm = sr.read_f32_tensor("post_projection.weight") @ (sr.read_f32_tensor("pre_projection.weight") @ x[p2])
            perm_err = float(np.max(np.abs(y_orig[perm] - y_perm)))
        gate("Gate 3: Perm-Pi Linear Equivariance & Embed Omitted", "model.embed_tokens.weight" not in keys and perm_err < 1e-5, f"err={perm_err:.2e}")

        sz = bhmod.pack_assistant_body_from_safetensors(src, out_q2, body_bits=2, block_size=64, has_fwht=False, verbose=False)
        gate("Gate 4: Q2 Embed-Free Body Packed", out_q2.is_file() and sz > 0, f"bytes={sz}")

        hdr = out_q2.read_bytes()[:24]
        magic = hdr[:4]
        gate("Gate 5: T16C Magic Header Valid", magic == p3.T16C_MAGIC)

        out_q4 = Path(td) / "ast_q4.t16c"
        sz4 = bhmod.pack_assistant_body_from_safetensors(src, out_q4, body_bits=4, block_size=64, has_fwht=True, verbose=False)
        gate("Gate 6: Q4 Embed-Free Body Packed", out_q4.is_file() and sz4 > sz)

    print(f"Pack Official Assistant Self-Test: {6 - failures}/6 passed", flush=True)
    return failures


def recover_perm_pi_from_t16c(target_t16c: Path, orig_safetensors: Path) -> np.ndarray:
    """Recover exact global coordinate permutation `perm_pi` from target `.t16c` norms vs `model.safetensors`."""
    import struct

    need_keys = {f"model.language_model.layers.{l}.input_layernorm.weight" for l in range(60)}
    t16c_norms: Dict[str, np.ndarray] = {}
    with open(target_t16c, "rb") as fp:
        magic, _, _, n_norms, _, _ = struct.unpack("<4s5I", fp.read(24))
        assert magic == b"T16C", f"Bad T16C magic {magic!r}"
        for _ in range(n_norms):
            nlen = struct.unpack("<H", fp.read(2))[0]
            name = fp.read(nlen).decode("utf-8")
            nval = struct.unpack("<I", fp.read(4))[0]
            raw = fp.read(nval * 4)
            if name in need_keys:
                t16c_norms[name] = np.frombuffer(raw, dtype="<f4").copy()

    orig_norms: Dict[str, np.ndarray] = {}
    with SafetensorsReader(orig_safetensors) as sr:
        for k in (
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.1.input_layernorm.weight",
            "model.language_model.layers.2.input_layernorm.weight",
            "model.language_model.layers.59.input_layernorm.weight",
        ):
            orig_norms[k] = sr.read_f32_tensor(k)

    # Match 4-layer tuple keys for collision-free exact perm_pi recovery
    sig_orig: Dict[Tuple[int, int, int, int], List[int]] = {}
    keys4 = (
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.1.input_layernorm.weight",
        "model.language_model.layers.2.input_layernorm.weight",
        "model.language_model.layers.59.input_layernorm.weight",
    )
    o0, o1, o2, o59 = [orig_norms[k] for k in keys4]
    t0, t1, t2, t59 = [t16c_norms[k] for k in keys4]
    for i in range(len(o0)):
        key = (int(round(float(o0[i]) * 1e5)), int(round(float(o1[i]) * 1e5)), int(round(float(o2[i]) * 1e5)), int(round(float(o59[i]) * 1e5)))
        sig_orig.setdefault(key, []).append(i)

    perm = np.empty((len(t0),), dtype=np.int64)
    for j in range(len(t0)):
        key = (int(round(float(t0[j]) * 1e5)), int(round(float(t1[j]) * 1e5)), int(round(float(t2[j]) * 1e5)), int(round(float(t59[j]) * 1e5)))
        cands = sig_orig.get(key)
        if not cands:
            raise KeyError(f"Channel {j} signature mismatch while recovering perm_pi")
        perm[j] = cands.pop(0)
    max_diff = float(np.max(np.abs(o0[perm] - t0)))
    assert max_diff < 1e-6, f"perm_pi verification failed: max_diff={max_diff:.2e}"
    print(f"[perm_pi] Recovered exact 5376-ch permutation from {target_t16c.name} (max_diff={max_diff:.2e})", flush=True)
    return perm


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack official Gemma 4 31B Assistant body (embed-free Target-Embed Sharing)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--src", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31B-it-assistant/model.safetensors"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_official_ast_q2_body.t16c"))
    ap.add_argument("--target-t16c", type=Path, default=None)
    ap.add_argument("--orig-safetensors", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/model.safetensors"))
    ap.add_argument("--body-bits", type=int, default=2)
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()
    if not args.src.is_file():
        raise FileNotFoundError(f"Missing official assistant safetensors: {args.src}")
    src_pack = args.src
    tmp_dir_obj = None
    try:
        if args.target_t16c is not None:
            perm_pi = recover_perm_pi_from_t16c(args.target_t16c, args.orig_safetensors)
            tmp_dir_obj = tempfile.TemporaryDirectory()
            src_pack = Path(tmp_dir_obj.name) / "permuted_ast.safetensors"
            permute_assistant_safetensors(args.src, src_pack, perm_pi, d_mod=len(perm_pi))
        bhmod.pack_assistant_body_from_safetensors(
            src_pack,
            args.out,
            body_bits=args.body_bits,
            block_size=64,
            has_fwht=(args.body_bits == 4),
        )
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

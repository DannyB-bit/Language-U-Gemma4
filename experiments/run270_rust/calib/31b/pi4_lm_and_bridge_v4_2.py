#!/usr/bin/env python3
"""V4.2 Phase B on Raspberry Pi 4: full 60-layer LM + causal C8DB bridge + unified LM+Vision `.t16c`.

Reuses the proven, resumable V4 pipelines with explicit V4.2 hooks (no forked math):
  Stage 3  `capture_v4_clean.run_capture` on the FULL clean V4.2 corpus (`full_corpus_v4_2.npz`,
           4,096 cal tokens + test2/testf probes) -> `work31b/snaps_v4_2/` (+ teacher bridge).
  Stage 4  `calibrate_31b_v4.run_calibration_v4` with the V4.2 allocation (zero 1-bit attention,
           all full-attention MLPs >= tern5, L58/L59 3b_u0) and the Pi-fast Cholesky ActALS
           (`fit_act_als_i8_fast`, parity_cos=1.0 vs the inv+eigh path) on ALL calibration tokens.
  Stage 5  `train_v4_bridge.run_bridge_v4` with 300-step C8DB training (KV + post_cell60) and the
           kept Vision `embed_vision` adapter merged into `gemma4_31b_v4_2_c8d.c8d`.
  Stage 6  Unified `.t16c` = LM norms + Vision norms + embed + 60 LM layers + Vision records,
           hard-guarded at `<= 6,360,000,000 B`.
  Stage 7  `verify_v4_replay.run_2x2_replay` on the unified file.

Run with no arguments or `--selftest` for the 6-gate offline suite; `--run` executes on the Pi.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import shutil
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

import calibrate_31b_v4 as c4
import capture_v4_clean as cap
import pack_t16c_v3 as p3
import train_c8d_pattern_bank as c8d
import train_v4_bridge as vb
import verify_v4_replay as vr
from calibrate_31b_v3 import _NativeCalOps
from calibrate_31b_v4 import _mat_size
from pack_t16c_v3 import (
    ARCH_FLAGS,
    LAYER_PROJS,
    MODE_TO_T16C,
    REQUIRED_PROJS,
    T16C_MAGIC,
    T16C_VERSION,
    SafetensorsReader,
    _write_mat_record,
    expected_tile16_bytes,
    tile16_blk_bytes,
)
from pi4_stage1_vision_v4_2 import fit_act_als_i8_fast

TOTAL_BUDGET_BYTES = 6_360_000_000
VISION_C8D_NAME = "model.embed_vision.embedding_projection.weight"
# 36 sliding-attention MLPs kept at 1-bit asym (d r=16, g/u r=8). Every full-attention layer is excluded.
MLP_1B_LAYERS = frozenset(
    {8, 9, 10, 12, 13, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34,
     36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52}
)
_PINV_SENTINEL = np.zeros((1,), dtype=np.float32)


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fp:
            for ln in fp:
                if ln.startswith("VmRSS:"):
                    return float(ln.split()[1]) / 1024.0
    except FileNotFoundError:
        return -1.0
    return -1.0


# ----------------------------------------------------------------------------------------------
# V4.2 allocation (mode codes: 0 = 1-bit asym, 1 = tern5, 4 = 3b_u0, 5 = 2b_gauss)
# ----------------------------------------------------------------------------------------------
def _proj_spec_v4_2(l: int, stem: str, is_full: bool) -> Tuple[int, int, int]:
    if l in (58, 59):
        return (4, 256, 32 if stem in ("k", "v") else 16)
    if stem in ("q", "o"):
        if l in (2, 4):
            return (4, 256, 16)
        if l in (10, 14, 19):
            return (5, 256, 16)
        return (1, 256, 16 if (is_full or l < 12 or l >= 50) else 8)
    if stem in ("k", "v"):
        return (1, 256, 0)
    if l in MLP_1B_LAYERS:
        return (0, 256, 16 if stem == "d" else 8)
    return (1, 256, 16 if stem == "d" else (8 if (l in (0, 1, 56, 57) or is_full) else 0))


def lm_t16c_upper_bytes(cfg: dict, alloc: Dict[int, dict], norms_bytes: int) -> int:
    """Upper bound of LM `.t16c` bytes (record headers + 64 B worst-case alignment per matrix)."""
    d = int(cfg["hidden_size"])
    ff = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    layer_types = cfg["layer_types"]
    emb_name = "model.language_model.embed_tokens.weight"
    tot = 24 + norms_bytes + 34 + len(emb_name) + 64 + ((vocab + 15) // 16) * (d // 256) * tile16_blk_bytes(256, 3, False)
    for l in range(int(cfg["num_hidden_layers"])):
        is_full = layer_types[l] == "full_attention"
        q_out = 16384 if is_full else 8192
        kv_out = 2048 if is_full else 4096
        a = alloc[l]
        p = f"model.language_model.layers.{l}."
        shapes = {
            "q": ("self_attn.q_proj.weight", q_out, d), "k": ("self_attn.k_proj.weight", kv_out, d),
            "v": ("self_attn.v_proj.weight", kv_out, d), "o": ("self_attn.o_proj.weight", d, q_out),
            "g": ("mlp.gate_proj.weight", ff, d), "u": ("mlp.up_proj.weight", ff, d),
            "d": ("mlp.down_proj.weight", d, ff),
        }
        for key, (suf, do, di) in shapes.items():
            if a[key] is None:
                continue
            tot += _mat_size(p + suf, do, di, *a[key]) + 64
    return tot


def build_v4_2_allocation(cfg: dict, budget_bytes: int, norms_bytes: int) -> Tuple[Dict[int, dict], int]:
    """Build the fixed V4.2 per-projection allocation; fail closed if it exceeds `budget_bytes`."""
    layer_types = cfg["layer_types"]
    k_eq_v = bool(cfg.get("attention_k_eq_v", False))
    alloc: Dict[int, dict] = {}
    for l in range(int(cfg["num_hidden_layers"])):
        is_full = layer_types[l] == "full_attention"
        if is_full and l in MLP_1B_LAYERS:
            raise ValueError(f"L{l:02d} is full_attention but listed in MLP_1B_LAYERS")
        alloc[l] = {s: _proj_spec_v4_2(l, s, is_full) for s in ("q", "k", "v", "o", "g", "u", "d")}
        if is_full and k_eq_v:
            alloc[l]["v"] = None
    tot = lm_t16c_upper_bytes(cfg, alloc, norms_bytes)
    if tot > budget_bytes:
        raise ValueError(f"V4.2 LM allocation {tot:,} B exceeds LM budget {budget_bytes:,} B")
    return alloc, tot


def safetensors_lm_norm_bytes(sf_path: Path) -> int:
    with SafetensorsReader(sf_path) as sr:
        keys = [k for k in sr.hdr if k.startswith("model.language_model.") and ("norm.weight" in k or "layer_scalar" in k)]
        return sum(2 + len(k.encode("utf-8")) + 4 + 4 * int(np.prod(sr.hdr[k]["shape"] or [1])) for k in keys)


def vision_npz_upper_bytes(vision_npz: Path) -> int:
    """Upper bound of the Vision records + norms bytes inside the unified `.t16c`."""
    with np.load(vision_npz, allow_pickle=False) as vz:
        rec_meta = json.loads(str(vz["rec_meta_json"]))
        tot = 0
        for n in vz["norm_names"].tolist():
            tot += 2 + len(str(n).encode("utf-8")) + 4 + vz[f"norm__{n}"].size * 4
        for i, rec in enumerate(rec_meta):
            tot += 34 + len(str(rec[0]).encode("utf-8")) + 64
            for suf in ("pb", "inv", "su", "sv", "ui8", "vi8"):
                key = f"m_{i}_{suf}"
                if key in vz.files:
                    tot += int(vz[key].nbytes)
    return tot


# ----------------------------------------------------------------------------------------------
# Stage 3: teacher capture on the full V4.2 corpus
# ----------------------------------------------------------------------------------------------
def load_v4_2_corpus(corpus_npz: Path) -> dict:
    """Adapt `full_corpus_v4_2.npz` to the dict contract of `build_tokenized_v4_corpus`."""
    if not corpus_npz.is_file():
        raise FileNotFoundError(f"Missing V4.2 corpus: {corpus_npz}")
    with np.load(corpus_npz, allow_pickle=False) as z:
        segs = [(int(a), int(b)) for a, b in z["cal_segs"].tolist()]
        cal_end = int(z["cal_end"])
        domains = [str(v) for v in z["prompt_domains"].tolist()]
        stats = json.loads(str(z["stats_json"])) if "stats_json" in z.files else {}
        all_ids = [int(v) for v in z["all_ids"].tolist()]
        corp = {
            "all_ids": all_ids,
            "next_token_ids": z["next_token_ids"].astype(np.int32).tolist(),
            "pos_arr": z["pos_arr"].astype(np.int64).tolist(),
            "ans_mask": z["ans_mask"].astype(bool).tolist(),
            "prompt_domains": domains,
            "prompt_slices": segs,
            "cal_end": cal_end,
        }
    probes = [s for s in segs if s[0] >= cal_end]
    if len(probes) != 2:
        raise ValueError(f"Expected exactly 2 eval probe segments after cal_end={cal_end}, got {probes}")
    if domains[probes[0][0]] != "test2" or domains[probes[1][0]] != "testf":
        raise ValueError(f"Probe domains mismatch: {domains[probes[0][0]]!r}, {domains[probes[1][0]]!r}")
    if any(d in ("test2", "testf", "heldout") for d in domains[:cal_end]):
        raise ValueError("Calibration region contains eval/heldout probe tokens (contamination)")
    if any(all_ids[s0] != 2 for s0, _ in segs):
        raise ValueError("Every segment must start with BOS id 2")
    corp["test2_bounds"] = list(probes[0])
    corp["testf_bounds"] = list(probes[1])
    corp["cal_pairs_count"] = int(stats.get("turns_packed", len(segs) - 2))
    return corp


def stage3_capture(sf_path: Path, cfg_path: Path, tok_path: Path, corpus_npz: Path, snaps_dir: Path, work_dir: Path) -> Path:
    bridge = snaps_dir / "teacher_bridge_v4.npz"
    if bridge.is_file() and (snaps_dir / "snaps_FINAL.npz").is_file():
        print(f"[V4.2 Stage3] teacher capture already complete -> {bridge}", flush=True)
        return bridge
    corp = load_v4_2_corpus(corpus_npz)
    orig = cap.build_tokenized_v4_corpus
    cap.build_tokenized_v4_corpus = lambda _tok_path, include_eval_probes=True: corp
    try:
        t0 = time.perf_counter()
        cap.run_capture(sf_path, cfg_path, tok_path, snaps_dir, work_dir)
        print(f"[V4.2 Stage3] teacher capture done in {time.perf_counter() - t0:.1f}s rss={_rss_mb():.0f}MB", flush=True)
    finally:
        cap.build_tokenized_v4_corpus = orig
    if not bridge.is_file():
        raise FileNotFoundError(f"Teacher capture finished without {bridge}")
    return bridge


# ----------------------------------------------------------------------------------------------
# Stage 4: causal V4.2 calibration cascade (proven V4 loop + V4.2 hooks)
# ----------------------------------------------------------------------------------------------
def _fast_act_als(y_err: np.ndarray, x_bal: np.ndarray, _x_pinv_t, rk: int, ops: _NativeCalOps):
    """Drop-in for `fit_act_als_i8`: exact Cholesky ActALS on ALL calibration tokens (no subsampling)."""
    return fit_act_als_i8_fast(y_err, x_bal, rk, ops)


def _skip_pinv(_x_bal: np.ndarray, _ops: _NativeCalOps) -> np.ndarray:
    """The fast ActALS never reads `x_pinv_t`; skip the O(N^3) inv+eigh precompute entirely."""
    return _PINV_SENTINEL


def stage4_calibrate(
    sf_path: Path, cfg_path: Path, snaps_dir: Path, work_dir: Path, layers_dir: Path, lm_budget: int
) -> int:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["text_config"]
    norms_b = safetensors_lm_norm_bytes(sf_path)
    alloc, tot = build_v4_2_allocation(cfg, lm_budget, norms_b)
    n_layers = int(cfg["num_hidden_layers"])
    if (layers_dir / "embed_3b256.npz").is_file() and all((layers_dir / f"L{l:02d}.npz").is_file() for l in range(n_layers)):
        print(f"[V4.2 Stage4] calibration already complete -> {layers_dir}", flush=True)
        return tot
    print(f"[V4.2 Stage4] LM allocation upper bound {tot:,} B (LM budget {lm_budget:,} B)", flush=True)
    (work_dir / "alloc_v4_2.json").write_text(json.dumps({str(k): v for k, v in alloc.items()}, indent=2))

    saved = (c4.build_v4_allocation, c4.fit_act_als_i8, c4.precompute_x_pinv_t_fast)
    c4.build_v4_allocation = lambda _cfg, budget_bytes=lm_budget: (alloc, tot)
    c4.fit_act_als_i8 = _fast_act_als
    c4.precompute_x_pinv_t_fast = _skip_pinv
    try:
        t0 = time.perf_counter()
        c4.run_calibration_v4(sf_path, cfg_path, snaps_dir, work_dir, layers_dir, None, budget_bytes=lm_budget)
        print(f"[V4.2 Stage4] calibration done in {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        c4.build_v4_allocation, c4.fit_act_als_i8, c4.precompute_x_pinv_t_fast = saved
    missing = [l for l in range(int(cfg["num_hidden_layers"])) if not (layers_dir / f"L{l:02d}.npz").is_file()]
    if missing or not (layers_dir / "embed_3b256.npz").is_file():
        raise FileNotFoundError(f"Calibration incomplete: missing layers {missing} or embed in {layers_dir}")
    return tot


# ----------------------------------------------------------------------------------------------
# Stage 6: unified LM + Vision `.t16c` (single file, hard byte guard)
# ----------------------------------------------------------------------------------------------
def assemble_unified_t16c(
    sf_path: Path, layers_dir: Path, vision_npz: Path, out_path: Path, n_layers: int, budget: int = TOTAL_BUDGET_BYTES
) -> int:
    """Write LM norms + Vision norms, embed, 60 LM layers, then all Vision records into one `.t16c`."""
    if out_path.is_file():
        existing_sz = out_path.stat().st_size
        if 6_000_000_000 <= existing_sz <= budget:
            print(f"[V4.2 Stage6] unified {out_path} already complete = {existing_sz:,} B", flush=True)
            return existing_sz
    for p in [layers_dir / "embed_3b256.npz", vision_npz] + [layers_dir / f"L{l:02d}.npz" for l in range(n_layers)]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing unified-assembly input: {p}")

    with SafetensorsReader(sf_path) as sr:
        lm_norm_keys = sorted(
            k for k in sr.hdr
            if k.startswith("model.language_model.") and ("norm.weight" in k or "layer_scalar" in k)
        )
        emb_shape = sr.hdr["model.language_model.embed_tokens.weight"]["shape"]
        vocab_sz, d_model = int(emb_shape[0]), int(emb_shape[1])
        norms: List[Tuple[str, np.ndarray]] = [(k, sr.read_f32_tensor(k)) for k in lm_norm_keys]

    vz = np.load(vision_npz, allow_pickle=False)
    try:
        vis_meta = json.loads(str(vz["rec_meta_json"]))
        vis_norm_names = [str(n) for n in vz["norm_names"].tolist()]
        seen = {k for k, _ in norms}
        for n in vis_norm_names:
            if n in seen:
                raise ValueError(f"Duplicate norm name across LM/Vision: {n}")
            norms.append((n, vz[f"norm__{n}"].astype(np.float32)))

        layer_present: List[List[Tuple[str, str]]] = []
        for l in range(n_layers):
            with np.load(layers_dir / f"L{l:02d}.npz") as d_npz:
                fs = set(d_npz.files)
            pres = []
            for stem, suffix in LAYER_PROJS:
                if f"{stem}__meta" in fs:
                    pres.append((stem, suffix))
                elif stem in REQUIRED_PROJS:
                    raise KeyError(f"L{l:02d}.npz missing required projection {stem}__meta")
            layer_present.append(pres)
        n_mats = 1 + sum(len(p) for p in layer_present) + len(vis_meta)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".t16c.tmp")
        with open(tmp_path, "wb") as fp:
            fp.write(struct.pack("<4s5I", T16C_MAGIC, T16C_VERSION, ARCH_FLAGS, len(norms), n_mats, 0))
            pos = 24
            for name, arr in norms:
                nb = name.encode("utf-8")
                a32 = np.ascontiguousarray(arr, dtype="<f4").ravel()
                fp.write(struct.pack("<H", len(nb)))
                fp.write(nb)
                fp.write(struct.pack("<I", a32.size))
                fp.write(a32.tobytes())
                pos += 2 + len(nb) + 4 + a32.size * 4
            with np.load(layers_dir / "embed_3b256.npz") as ed:
                pos = _write_mat_record(
                    fp, pos, "model.language_model.embed_tokens.weight", vocab_sz, d_model, 256, 3, False, True,
                    ed["pbytes"], None, None,
                )
            for l in range(n_layers):
                with np.load(layers_dir / f"L{l:02d}.npz") as d_npz:
                    for stem, suffix in layer_present[l]:
                        d_out, d_in, bsize, mode_i, rk = [int(v) for v in d_npz[f"{stem}__meta"]]
                        n_bits, is_asym = MODE_TO_T16C[mode_i]
                        lr = None
                        if rk > 0:
                            lr = tuple(d_npz[f"{stem}__{s}"] for s in ("su", "sv", "ui8", "vi8"))
                        pos = _write_mat_record(
                            fp, pos, f"model.language_model.layers.{l}.{suffix}", d_out, d_in, bsize, n_bits,
                            is_asym, True, d_npz[f"{stem}__pbytes"], d_npz[f"{stem}__inv_s"], lr,
                        )
            for i, (name, do, di, bs, nb_v, asym_v) in enumerate(vis_meta):
                inv = vz[f"m_{i}_inv"] if f"m_{i}_inv" in vz.files else None
                lr = None
                if f"m_{i}_su" in vz.files:
                    lr = tuple(vz[f"m_{i}_{s}"] for s in ("su", "sv", "ui8", "vi8"))
                pos = _write_mat_record(
                    fp, pos, str(name), int(do), int(di), int(bs), int(nb_v), bool(asym_v), True,
                    vz[f"m_{i}_pb"], inv, lr,
                )
    finally:
        vz.close()

    sz = tmp_path.stat().st_size
    if sz > budget:
        tmp_path.unlink()
        raise ValueError(f"Unified .t16c {sz:,} B exceeds hard budget {budget:,} B (file removed)")
    os.replace(tmp_path, out_path)
    print(f"[V4.2 Stage6] unified {out_path} = {sz:,} B ({sz / 1e9:.4f} GB, {sz / 1024**3:.3f} GiB) "
          f"norms={len(norms)} mats={n_mats}", flush=True)
    return sz


# ----------------------------------------------------------------------------------------------
# Stage 5: 300-step C8DB bridge on the unified file + Vision adapter merge; Stage 7 replay
# ----------------------------------------------------------------------------------------------
def load_vision_c8d_entry(vis_c8d_npz: Path) -> Tuple[Optional[dict], dict]:
    """Return the kept `embed_vision` C8DB entry (or None when Vision validation did not improve)."""
    if not vis_c8d_npz.is_file():
        raise FileNotFoundError(f"Missing Vision C8DB result: {vis_c8d_npz}")
    with np.load(vis_c8d_npz, allow_pickle=False) as z:
        stats = {k: float(np.asarray(z[k]).reshape(-1)[0]) for k in ("keep", "cos_before", "cos_after") if k in z.files}
        if int(stats.get("keep", 0)) != 1:
            return None, stats
        cents = z["centroids"].astype(np.float32)
        n_exp = cents.shape[0]
        experts = [
            (z[f"e{e}_su"].astype(np.float32), z[f"e{e}_sv"].astype(np.float32),
             z[f"e{e}_ui"].astype(np.int8), z[f"e{e}_vi"].astype(np.int8))
            for e in range(n_exp)
        ]
    rank = int(experts[0][1].shape[0])
    d_out = int(experts[0][0].shape[0])
    d_in = int(experts[0][3].shape[1])
    rads = [c8d._c6d_radicals("general", "vector", "compute", "native", "layer", "positive", e) for e in range(n_exp)]
    entry = {
        "name": VISION_C8D_NAME, "d_out": d_out, "d_in": d_in, "rank": rank, "top_k": 2, "c6d": rads[0],
        "temperature": 0.5, "gate_scale": 1.0, "centroids": cents, "expert_radicals": rads, "experts": experts,
    }
    return entry, stats


def stage5_bridge(
    sf_path: Path, cfg_path: Path, t16c_path: Path, bridge_npz: Path, work_dir: Path,
    out_c8d: Path, state_cache: Path, vis_c8d_npz: Path, steps: int = 300,
) -> Path:
    vis_entry, vis_stats = load_vision_c8d_entry(vis_c8d_npz)
    print(f"[V4.2 Stage5] Vision C8DB stats={vis_stats} merge={'yes' if vis_entry else 'no'}", flush=True)
    orig_train, orig_write = c8d.train_projection_pattern_bank, c8d.write_c8db_file

    def _train_steps(*a, **kw):
        kw["steps"] = steps
        return orig_train(*a, **kw)

    def _write_with_vision(path, norms_l, entries_l):
        ents = list(entries_l) + ([vis_entry] if vis_entry is not None else [])
        return orig_write(path, norms_l, ents)

    c8d.train_projection_pattern_bank = _train_steps
    c8d.write_c8db_file = _write_with_vision
    try:
        vb.run_bridge_v4(
            sf_path, cfg_path, t16c_path, bridge_npz, work_dir, out_c8d,
            rank_cell60=16, rank_kv=32, state_cache=state_cache, tau=1.5, gamma=2.0, steps=steps,
        )
    finally:
        c8d.train_projection_pattern_bank, c8d.write_c8db_file = orig_train, orig_write
    _, ents = c8d.read_c8db_file(out_c8d)
    if vis_entry is not None and VISION_C8D_NAME not in ents:
        raise ValueError(f"Vision adapter missing from {out_c8d}")
    print(f"[V4.2 Stage5] {out_c8d} entries={sorted(ents)}", flush=True)
    return out_c8d


def stage7_replay(t16c: Path, c8d_path: Path, bridge_npz: Path, state_cache: Path, sf_path: Path,
                  cfg_path: Path, tok_path: Path, ast_path: Path, so_path: Path, probe_bin: Path) -> dict:
    res = vr.run_2x2_replay(t16c, c8d_path, bridge_npz, state_cache, sf_path, cfg_path, tok_path,
                            ast_path, so_path, probe_bin, n_steps=5)
    print("[V4.2 Stage7] 2x2 replay complete", flush=True)
    return res


# ----------------------------------------------------------------------------------------------
# 6-gate self-test
# ----------------------------------------------------------------------------------------------
def _gemma4_31b_cfg_stub() -> dict:
    return {
        "num_hidden_layers": 60, "hidden_size": 5376, "intermediate_size": 21504, "vocab_size": 262144,
        "attention_k_eq_v": True,
        "layer_types": ["full_attention" if (l % 6 == 5) else "sliding_attention" for l in range(60)],
    }


def _self_test() -> int:
    failures = 0
    rng = np.random.default_rng(42)

    # Gate 1: V4.2 allocation invariants + byte budget
    try:
        cfg = _gemma4_31b_cfg_stub()
        alloc, tot = build_v4_2_allocation(cfg, 6_200_000_000, 1_900_000)
        for l, a in alloc.items():
            full = cfg["layer_types"][l] == "full_attention"
            for s in ("q", "k", "v", "o"):
                assert a[s] is None or a[s][0] != 0, f"1-bit attention at L{l} {s}"
            if full:
                assert a["v"] is None and all(a[s][0] != 0 for s in ("g", "u", "d")), f"full L{l} MLP 1-bit"
        assert all(alloc[l][s][0] == 4 for l in (58, 59) for s in ("q", "k", "o", "g", "u", "d"))
        assert alloc[58]["k"][2] == 32 and alloc[58]["v"][2] == 32
        assert sum(1 for l in range(60) if alloc[l]["g"][0] == 0) == 36
        assert 6_000_000_000 < tot <= 6_200_000_000, tot
        try:
            build_v4_2_allocation(cfg, 5_000_000_000, 1_900_000)
            raise AssertionError("expected budget ValueError")
        except ValueError:
            pass
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc!r}")
        failures += 1

    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        # Gate 2: corpus adapter contract + contamination fail-closed
        try:
            ids = np.array([2, 5, 6, 7, 2, 8, 9, 2, 3, 2, 4], dtype=np.int32)
            segs = np.array([[0, 4], [4, 7], [7, 9], [9, 11]], dtype=np.int32)
            dom = np.array(["math"] * 4 + ["code"] * 3 + ["test2"] * 2 + ["testf"] * 2)
            base = dict(all_ids=ids, pos_arr=np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 1], dtype=np.int32),
                        cal_segs=segs, cal_end=np.array(7, dtype=np.int32), ans_mask=np.ones(11, dtype=bool),
                        next_token_ids=np.roll(ids, -1), prompt_domains=dom,
                        stats_json=np.array(json.dumps({"turns_packed": 2})))
            np.savez(td / "c.npz", **base)
            corp = load_v4_2_corpus(td / "c.npz")
            assert corp["test2_bounds"] == [7, 9] and corp["testf_bounds"] == [9, 11]
            assert corp["cal_end"] == 7 and corp["cal_pairs_count"] == 2 and len(corp["all_ids"]) == 11
            bad = dict(base)
            bad["prompt_domains"] = np.array(["math", "test2"] + ["math"] * 2 + ["code"] * 3 + ["test2"] * 2 + ["testf"] * 2)
            np.savez(td / "bad.npz", **bad)
            try:
                load_v4_2_corpus(td / "bad.npz")
                raise AssertionError("expected contamination ValueError")
            except ValueError:
                pass
        except Exception as exc:
            print(f"[Gate 2 FAIL] {exc!r}")
            failures += 1

        # Gate 3: Stage-4 hooks are exact drop-ins and restore cleanly
        try:
            ops = _NativeCalOps(None, require_native=False)
            x = rng.standard_normal((96, 48)).astype(np.float32)
            w_lr = (rng.standard_normal((40, 4)) @ rng.standard_normal((4, 48))).astype(np.float32)
            y_err = np.ascontiguousarray((x @ w_lr.T).T, dtype=np.float32)
            out_fast = _fast_act_als(y_err, x, _skip_pinv(x, ops), 4, ops)
            out_ref = c4.fit_act_als_i8(y_err, x, c4.precompute_x_pinv_t_fast(x, ops), 4, ops)
            yf = x @ out_fast[1].T @ out_fast[0].T
            yr = x @ out_ref[1].T @ out_ref[0].T
            cos = float(np.sum(yf * yr) / (np.linalg.norm(yf) * np.linalg.norm(yr) + 1e-12))
            assert cos > 0.999, f"fast ActALS parity cos={cos}"
            assert len(out_fast) == 6 and out_fast[4].dtype == np.int8 and out_fast[5].dtype == np.int8
        except Exception as exc:
            print(f"[Gate 3 FAIL] {exc!r}")
            failures += 1

        # Gate 4: unified assembler writes LM + Vision records, parses back, enforces budget
        try:
            vocab, d = 32, 256
            sf_p = td / "model.safetensors"
            p3._write_synthetic_safetensors(sf_p, {
                "model.language_model.embed_tokens.weight": np.zeros((vocab, d), dtype=np.float32),
                "model.language_model.layers.0.input_layernorm.weight": np.ones((d,), dtype=np.float32),
                "model.language_model.layers.0.layer_scalar": np.array([0.5], dtype=np.float32),
                "model.language_model.norm.weight": np.ones((d,), dtype=np.float32),
            })
            ld = td / "layers"
            ld.mkdir()
            np.savez(ld / "embed_3b256.npz", pbytes=np.full(expected_tile16_bytes(vocab, d, 256, 3, False), 0x3C, np.uint8))
            l0 = {}
            for stem, (do, di, md, rk) in {"q_proj": (16, d, 1, 4), "k_proj": (16, d, 1, 0), "o_proj": (d, d, 4, 4),
                                           "gate_proj": (32, d, 0, 4), "up_proj": (32, d, 0, 4), "down_proj": (d, d, 1, 4)}.items():
                nb, asym = MODE_TO_T16C[md]
                l0[f"{stem}__meta"] = np.array([do, di, 256, md, rk], dtype=np.int32)
                l0[f"{stem}__pbytes"] = np.full(expected_tile16_bytes(do, di, 256, nb, asym), 42 if nb == 5 else 0x55, np.uint8)
                l0[f"{stem}__inv_s"] = np.ones((di,), dtype=np.float32)
                if rk:
                    l0.update({f"{stem}__su": np.full((do,), 0.01, np.float32), f"{stem}__sv": np.full((rk,), 0.02, np.float32),
                               f"{stem}__ui8": np.full((do, rk), 3, np.int8), f"{stem}__vi8": np.full((rk, di), -2, np.int8)})
            np.savez(ld / "L00.npz", **l0)
            vname = "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
            nname = "model.vision_tower.encoder.layers.0.input_layernorm.weight"
            np.savez(td / "vis.npz", rec_meta_json=np.array(json.dumps([[vname, 16, 64, 64, 5, 0]])),
                     norm_names=np.array([nname]), **{f"norm__{nname}": np.ones((64,), np.float32)},
                     m_0_pb=np.full(expected_tile16_bytes(16, 64, 64, 5, False), 42, np.uint8),
                     m_0_inv=np.ones((64,), np.float32), m_0_su=np.full((16,), 0.01, np.float32),
                     m_0_sv=np.full((2,), 0.02, np.float32), m_0_ui8=np.ones((16, 2), np.int8), m_0_vi8=np.ones((2, 64), np.int8))
            out = td / "u.t16c"
            sz = assemble_unified_t16c(sf_p, ld, td / "vis.npz", out, n_layers=1)
            assert sz == out.stat().st_size and sz <= vision_npz_upper_bytes(td / "vis.npz") + 10_000_000
            with vb._SeekT16CReader(out, None) as rd:
                assert vname in rd.mats and nname in rd.norms
                assert "model.language_model.layers.0.self_attn.v_proj.weight" not in rd.mats
                assert len(rd.mats) == 1 + 6 + 1 and rd.mats[vname]["lr_rank"] == 2
            try:
                assemble_unified_t16c(sf_p, ld, td / "vis.npz", td / "small.t16c", n_layers=1, budget=1000)
                raise AssertionError("expected budget ValueError")
            except ValueError:
                assert not (td / "small.t16c").exists() and not (td / "small.t16c.tmp").exists()
        except Exception as exc:
            print(f"[Gate 4 FAIL] {exc!r}")
            failures += 1

        # Gate 5: Vision C8DB entry loads, round-trips through C8DB, keep=0 is skipped
        try:
            ve = {"keep": np.array(1), "cos_before": np.array(0.9), "cos_after": np.array(0.95),
                  "centroids": rng.standard_normal((8, 8)).astype(np.float32)}  # routing space = 2 * rank
            for e in range(8):
                ve.update({f"e{e}_su": np.full((48,), 0.01, np.float32), f"e{e}_sv": np.full((4,), 0.02, np.float32),
                           f"e{e}_ui": rng.integers(-60, 60, (48, 4), dtype=np.int8),
                           f"e{e}_vi": rng.integers(-60, 60, (4, 64), dtype=np.int8)})
            np.savez(td / "vc.npz", **ve)
            ent, st = load_vision_c8d_entry(td / "vc.npz")
            assert ent is not None and ent["d_out"] == 48 and ent["d_in"] == 64 and ent["rank"] == 4 and st["keep"] == 1.0
            c8d.write_c8db_file(td / "x.c8d", [("model.language_model.norm.weight", np.ones(64, np.float32))], [ent])
            _, ents = c8d.read_c8db_file(td / "x.c8d")
            xq = rng.standard_normal((3, 64)).astype(np.float32)
            y0 = c8d.eval_quantized_adapter_np(xq, ent["centroids"], ent["experts"], 2, 0.5, 1.0)
            r = ents[VISION_C8D_NAME]
            y1 = c8d.eval_quantized_adapter_np(xq, r["centroids"], r["experts"], r["top_k"], r["temperature"], r["gate_scale"])
            assert np.allclose(y0, y1, atol=1e-5)
            ve["keep"] = np.array(0)
            np.savez(td / "vc0.npz", **ve)
            assert load_vision_c8d_entry(td / "vc0.npz")[0] is None
        except Exception as exc:
            print(f"[Gate 5 FAIL] {exc!r}")
            failures += 1

        # Gate 6: fail closed on missing artifacts
        try:
            for fn in (lambda: load_vision_c8d_entry(td / "nope.npz"),
                       lambda: load_v4_2_corpus(td / "nope.npz"),
                       lambda: assemble_unified_t16c(td / "no.sf", td / "nol", td / "nov.npz", td / "o.t16c", 1)):
                try:
                    fn()
                    raise AssertionError("expected FileNotFoundError")
                except FileNotFoundError:
                    pass
        except Exception as exc:
            print(f"[Gate 6 FAIL] {exc!r}")
            failures += 1

    if failures == 0:
        print("[PASS] pi4_lm_and_bridge_v4_2.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()
    ap = argparse.ArgumentParser(description="V4.2 Phase B (LM + bridge + unified .t16c) on Pi 4.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--safetensors", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct"))
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b"))
    ap.add_argument("--out-t16c", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v4_2_full.t16c"))
    ap.add_argument("--out-c8d", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v4_2_c8d.c8d"))
    ap.add_argument("--assistant", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_assistant.t16c"))
    ap.add_argument("--so", type=Path, default=Path("/home/astronautshe/gemma4_verify/neon_runner/libneon_stream_accelerator.so"))
    ap.add_argument("--probe-bin", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/target/release/probe_assistant"))
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    if not args.run:
        ap.error("pass --run to execute (or --selftest)")

    wd = args.work_dir
    cfg_path, tok_path = args.model_dir / "config.json", args.model_dir / "tokenizer.json"
    sf_path = p3.resolve_safetensors_path(args.safetensors if args.safetensors else args.model_dir)
    corpus_npz, vision_npz, vis_c8d_npz = wd / "full_corpus_v4_2.npz", wd / "vision_v4_2.npz", wd / "vis_c8d_v4_2.npz"
    for p in (cfg_path, tok_path, corpus_npz, vision_npz, vis_c8d_npz, args.assistant, args.so, args.probe_bin):
        if not p.is_file():
            raise FileNotFoundError(f"Missing Phase B prerequisite: {p}")
    snaps_dir, layers_dir = wd / "snaps_v4_2", wd / "layers_v4_2"
    state_cache = wd / "student_state_v4_2.npz"
    n_layers = int(json.loads(cfg_path.read_text(encoding="utf-8"))["text_config"]["num_hidden_layers"])

    vis_upper = vision_npz_upper_bytes(vision_npz)
    lm_budget = TOTAL_BUDGET_BYTES - vis_upper
    print(f"[V4.2] Vision upper bound {vis_upper:,} B -> LM budget {lm_budget:,} B (total {TOTAL_BUDGET_BYTES:,})", flush=True)
    free_b = shutil.disk_usage(str(wd)).free
    req_free = 1_000_000_000 if args.out_t16c.is_file() else 12_000_000_000
    if free_b < req_free:
        raise OSError(f"Need >= {req_free:,} B free on {wd} for resume/assembly, have {free_b:,} B")

    bridge_npz = stage3_capture(sf_path, cfg_path, tok_path, corpus_npz, snaps_dir, wd)
    stage4_calibrate(sf_path, cfg_path, snaps_dir, wd, layers_dir, lm_budget)
    assemble_unified_t16c(sf_path, layers_dir, vision_npz, args.out_t16c, n_layers)
    stage5_bridge(sf_path, cfg_path, args.out_t16c, bridge_npz, wd, args.out_c8d, state_cache, vis_c8d_npz, args.steps)
    stage7_replay(args.out_t16c, args.out_c8d, bridge_npz, state_cache, sf_path, cfg_path, tok_path,
                  args.assistant, args.so, args.probe_bin)
    print("V4_2_PHASE_B_COMPLETE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

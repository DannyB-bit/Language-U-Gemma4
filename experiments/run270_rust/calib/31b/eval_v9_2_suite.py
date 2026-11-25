#!/usr/bin/env python3
"""
V9.2-CliffFix-VLM Physical Raspberry Pi 4 Proof & Telemetry Suite.

Builds cleanly on `eval_v6_1_skeptic_proof` (exact disjoint held-out passages,
exact 15-prompt deterministic battery checkers, and strict fail-closed guards)
while exposing `--andvcnt-bits 6`, `--head-router`, `--head-top-p`,
`--shared-mlp-blocks`, `--kv-quant glyph-lod|fp8`, and `PROFILE31B`/`ATTN_SPARSE`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_v6_1_skeptic_proof as e61  # noqa: E402


def parse_pareto_spec(spec: str) -> List[Tuple[str, float, float, float]]:
    """Parse comma-separated `name:mlp_top_p:mod_thresh:head_top_p` configs."""
    out: List[Tuple[str, float, float, float]] = []
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 4:
            raise ValueError(f"Expected name:mlp_top_p:mod_thresh:head_top_p, got {tok!r}")
        name, mp_s, mt_s, hp_s = parts[0].strip(), float(parts[1]), float(parts[2]), float(parts[3])
        if not name or not (0.0 < mp_s <= 1.0) or not (0.0 <= mt_s <= 1.0) or not (0.0 < hp_s <= 1.0):
            raise ValueError(f"Invalid Pareto config bounds in {tok!r}")
        out.append((name, mp_s, mt_s, hp_s))
    if not out:
        raise ValueError("Empty pareto config list")
    return out


def parse_extended_telemetry(stdout_text: str) -> Dict[str, object]:
    """Extend `eval_v6_1_skeptic_proof.parse_run31b_telemetry` with V9.2 fields."""
    res = dict(e61.parse_run31b_telemetry(stdout_text))
    m_ids = re.search(r"GEN31B ids=\[([0-9,\s]*)\]", stdout_text)
    if m_ids:
        raw_ids = [x.strip() for x in m_ids.group(1).split(",") if x.strip()]
        res["n_gen_tokens"] = len(raw_ids)
        if "s_per_tok" not in res and res.get("decode_s") is not None and len(raw_ids) > 1:
            res["s_per_tok"] = round(float(res["decode_s"]) / float(len(raw_ids) - 1), 3)
    m_lmcs = re.search(r"lmhead_cs_skipped_blocks=(\d+)", stdout_text)
    if m_lmcs:
        res["lmhead_cs_skipped_blocks"] = int(m_lmcs.group(1))
    m_att = re.search(r"ATTN_SPARSE skipped_heads=(\d+) total_heads=(\d+) saved_pct=([0-9.]+)%", stdout_text)
    if m_att:
        res["head_skipped"] = int(m_att.group(1))
        res["head_total"] = int(m_att.group(2))
        res["head_saved_pct"] = float(m_att.group(3))
    m_prof = re.search(
        r"PROFILE31B steps=(\d+) attn_s=([0-9.]+) gate_s=([0-9.]+) up_s=([0-9.]+) "
        r"down_s=([0-9.]+) norm_kv_s=([0-9.]+) lmhead_s=([0-9.]+) avg_step_s=([0-9.]+)",
        stdout_text,
    )
    if m_prof:
        res["profile"] = {
            "steps": int(m_prof.group(1)),
            "attn_s": float(m_prof.group(2)),
            "gate_s": float(m_prof.group(3)),
            "up_s": float(m_prof.group(4)),
            "down_s": float(m_prof.group(5)),
            "norm_kv_s": float(m_prof.group(6)),
            "lmhead_s": float(m_prof.group(7)),
            "avg_step_s": float(m_prof.group(8)),
        }
    return res


def validate_required_paths(args: argparse.Namespace) -> None:
    """Fail closed immediately when required binary or model assets are absent."""
    req = [("run31b", args.run31b), ("t16c", args.t16c), ("config", args.config), ("tokenizer", args.tokenizer)]
    for label, p in req:
        if p is None or not Path(p).is_file():
            raise FileNotFoundError(f"Missing required --{label} file: {p}")


def build_base_cmd(
    args: argparse.Namespace,
    kv_quant: Optional[str] = None,
    mlp_top_p: float = 1.0,
    mod_thresh: float = 0.0,
    head_top_p: float = 1.0,
) -> List[str]:
    """Construct exact `run31b` command vector cleanly for any density/routing regime."""
    cmd = [
        str(args.run31b),
        "--t16c",
        str(args.t16c),
        "--config",
        str(args.config),
        "--tokenizer",
        str(args.tokenizer),
        "--threads",
        str(int(args.threads)),
        "--kv-quant",
        kv_quant if kv_quant is not None else args.kv_quant,
        "--lmhead-cs",
        args.lmhead_cs,
        "--andvcnt-bits",
        str(int(args.andvcnt_bits)),
    ]
    if args.c8d is not None:
        cmd.extend(["--cuneiform-bank", str(args.c8d)])
    if args.so is not None:
        cmd.extend(["--so", str(args.so)])
    if args.profile_components:
        cmd.append("--profile-components")
    if args.tri_modal_gate:
        cmd.append("--tri-modal-gate")
    if mlp_top_p < 0.999 or mod_thresh > 0.0:
        if args.router is None:
            raise FileNotFoundError("--router required when mlp_top_p < 1.0 or mod_thresh > 0")
        cmd.extend(["--cortical-router", str(args.router), "--mlp-top-p", f"{mlp_top_p:.4f}"])
        if args.shared_mlp_blocks > 0:
            cmd.extend(["--shared-mlp-blocks", str(int(args.shared_mlp_blocks))])
        if mod_thresh > 0.0:
            cmd.extend(["--mod-thresh", f"{mod_thresh:.4f}"])
    if head_top_p < 0.999:
        if args.head_router is None:
            raise FileNotFoundError("--head-router required when head_top_p < 1.0")
        cmd.extend(["--head-router", str(args.head_router), "--head-top-p", f"{head_top_p:.4f}"])
    return cmd


def _run_cmd(cmd: Sequence[str], log_fp=None) -> Tuple[str, Dict[str, object]]:
    proc = subprocess.run(list(cmd), capture_output=True, text=True, check=False)
    combined = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr else "")
    if log_fp is not None:
        log_fp.write("$ " + " ".join(cmd) + "\n")
        log_fp.write(combined + "\n")
        log_fp.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"run31b failed rc={proc.returncode}:\n{combined[-2000:]}")
    return combined, parse_extended_telemetry(combined)


def _aggregate_passages(rows: List[dict], cfg_name: str, kv_q: str, mp: float, mt: float, hp: float) -> dict:
    tot_pos = sum(int(r.get("eval_pos", 0)) for r in rows)
    tot_tok = sum(int(r.get("eval_tokens", 0)) for r in rows)
    def _wavg(key: str) -> Optional[float]:
        vals = [(int(r.get("eval_pos", 0)), r.get(key)) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        w = sum(p for p, _ in vals)
        return round(sum(p * float(v) for p, v in vals) / max(w, 1), 5)

    mean_nll = _wavg("mean_nll") or 0.0
    t_nll = _wavg("teacher_nll")
    return {
        "config": cfg_name,
        "kv_quant": kv_q,
        "mlp_top_p": mp,
        "mod_thresh": mt,
        "head_top_p": hp,
        "eval_tokens": tot_tok,
        "eval_pos": tot_pos,
        "mean_nll": mean_nll,
        "ppl": round(math.exp(min(mean_nll, 30.0)), 3),
        "top1_acc": _wavg("top1_acc"),
        "mean_entropy": _wavg("mean_entropy"),
        "kl_t_to_s": _wavg("kl_t_to_s"),
        "kl_s_to_t": _wavg("kl_s_to_t"),
        "teacher_top1_agree": _wavg("teacher_top1_agree"),
        "teacher_top5_agree": _wavg("teacher_top5_agree"),
        "teacher_nll": t_nll,
        "gate_saved_pct": _wavg("gate_saved_pct"),
        "head_saved_pct": _wavg("head_saved_pct"),
        "eval_s": round(sum(float(r.get("eval_s", 0.0)) for r in rows), 2),
        "vm_hwm_kb": max((int(r.get("vm_hwm_kb", 0)) for r in rows), default=0),
        "ssd_read_bytes": sum(int(r.get("ssd_read_bytes", 0)) for r in rows),
        "passages": rows,
    }


def run_nll_pareto(args: argparse.Namespace, out_dir: Path) -> List[dict]:
    """Run held-out 3-passage Teacher NLL/KL Pareto across KV modes and sparse configs."""
    kv_modes = [k.strip() for k in args.pareto_kv_modes.split(",") if k.strip()]
    cfgs = parse_pareto_spec(args.pareto_configs)
    all_summaries: List[dict] = []
    with open(out_dir / "nll_pareto_v9_2.log", "w", encoding="utf-8") as lfp:
        for kv_q in kv_modes:
            for name, mp, mt, hp in cfgs:
                p_rows: List[dict] = []
                for p_idx, (ptag, passage) in enumerate(e61.HELDOUT_NLL_PASSAGES):
                    lgt_p = args.teacher_logits / f"{ptag}.lgt1" if args.teacher_logits else None
                    q_txt = e61.HELDOUT_CHAT_QUESTIONS[p_idx] if p_idx < len(e61.HELDOUT_CHAT_QUESTIONS) else "Explain briefly."
                    prompt_s = e61.format_gemma4_chat_nll_turn(q_txt, passage)
                    cmd = build_base_cmd(args, kv_quant=kv_q, mlp_top_p=mp, mod_thresh=mt, head_top_p=hp)
                    cmd.extend(["--eval-nll", "--eval-nll-after-turn", "--raw-prompt", "--prompt-text", prompt_s])
                    if lgt_p is not None and lgt_p.is_file():
                        cmd.extend(["--eval-teacher-logits", str(lgt_p)])
                    _, parsed = _run_cmd(cmd, log_fp=lfp)
                    parsed["passage"] = ptag
                    p_rows.append(parsed)
                summary = _aggregate_passages(p_rows, f"{name}_{kv_q}", kv_q, mp, mt, hp)
                all_summaries.append(summary)
                print(
                    f"[PARETO] {summary['config']:22s} nll={summary['mean_nll']:.5f} ppl={summary['ppl']:.2f} "
                    f"kl_t2s={summary['kl_t_to_s']} agree1={summary['teacher_top1_agree']} "
                    f"mlp_save={summary['gate_saved_pct']} head_save={summary['head_saved_pct']} "
                    f"hwm={summary['vm_hwm_kb']}",
                    flush=True,
                )
    return all_summaries


def run_battery(args: argparse.Namespace, out_dir: Path) -> Dict[str, dict]:
    """Run all 15 deterministic prompts under each requested dense/sparse config."""
    cfgs = parse_pareto_spec(args.battery_configs)
    suites: Dict[str, dict] = {}
    with open(out_dir / "battery_15_v9_2.log", "w", encoding="utf-8") as lfp:
        for name, mp, mt, hp in cfgs:
            rows: List[dict] = []
            for tag, prompt, max_toks, checker in e61.BATTERY_PROMPTS:
                lfp.write(f"\n=== [{name}] {tag}: {prompt} ===\n")
                cmd = build_base_cmd(args, kv_quant=args.kv_quant, mlp_top_p=mp, mod_thresh=mt, head_top_p=hp)
                cmd.extend(["--prompt-text", prompt, "--max-tokens", str(int(max_toks))])
                _, parsed = _run_cmd(cmd, log_fp=lfp)
                gen_txt = str(parsed.get("gen_text", ""))
                ok = bool(checker(gen_txt))
                parsed.update({"tag": tag, "prompt": prompt, "passed": ok})
                rows.append(parsed)
                print(
                    f"[BAT:{name}] {tag:15s} pass={int(ok)} pre={parsed.get('prefill_s')}s "
                    f"dec={parsed.get('decode_s')}s text={gen_txt!r}",
                    flush=True,
                )
            passed = sum(1 for r in rows if r["passed"])
            suites[name] = {
                "passed": passed,
                "total": len(rows),
                "vm_hwm_kb": max((int(r.get("vm_hwm_kb", 0)) for r in rows), default=0),
                "ssd_read_bytes": sum(int(r.get("ssd_read_bytes", 0)) for r in rows),
                "rows": rows,
            }
            print(f"[BATTERY SUMMARY:{name}] {passed}/{len(rows)} passed hwm={suites[name]['vm_hwm_kb']}", flush=True)
    return suites


def run_vision(args: argparse.Namespace, out_dir: Path) -> dict:
    """Run multimodal COCO Vision QA cleanly on physical image."""
    if args.image is None or not args.image.is_file():
        raise FileNotFoundError(f"Missing COCO test image --image: {args.image}")
    with open(out_dir / "vision_v9_2.log", "w", encoding="utf-8") as lfp:
        cmd = build_base_cmd(args, kv_quant=args.kv_quant, mlp_top_p=1.0, mod_thresh=0.0, head_top_p=1.0)
        cmd.extend([
            "--image",
            str(args.image),
            "--prompt-text",
            "Describe what you see in this image in one short sentence.",
            "--max-tokens",
            "28",
        ])
        _, parsed = _run_cmd(cmd, log_fp=lfp)
    print(
        f"[VISION] text={parsed.get('gen_text')!r} pre={parsed.get('prefill_s')}s "
        f"dec={parsed.get('decode_s')}s hwm={parsed.get('vm_hwm_kb')} ssd={parsed.get('ssd_read_bytes')}",
        flush=True,
    )
    return parsed


def run_spec_ab(args: argparse.Namespace, out_dir: Path) -> List[dict]:
    """Compare Dense k=1 vs Assistant-V7 BHDR speculative decoding under abits=6 + exact CS."""
    if args.assistant_v7_body is None or not args.assistant_v7_body.is_file():
        raise FileNotFoundError(f"Missing --assistant-v7-body: {args.assistant_v7_body}")
    if args.byte_head is None or not args.byte_head.is_file():
        raise FileNotFoundError(f"Missing --byte-head: {args.byte_head}")
    prompt_tags = {"P1_arith", "P2_france", "P5_speed", "B5_multistep"}
    selected = [p for p in e61.BATTERY_PROMPTS if p[0] in prompt_tags]
    regimes = [
        ("dense_k1", False, False),
        ("bhdr_spec_k4", True, False),
        ("bhdr_spec_k4_tmg", True, True),
    ]
    rows: List[dict] = []
    with open(out_dir / "spec_bhdr_v9_2.log", "w", encoding="utf-8") as lfp:
        for rname, is_spec, use_tmg in regimes:
            for tag, prompt, max_toks, checker in selected:
                lfp.write(f"\n=== [{rname}] {tag} ===\n")
                cmd = build_base_cmd(args, kv_quant=args.kv_quant, mlp_top_p=1.0, mod_thresh=0.0, head_top_p=1.0)
                if use_tmg and "--tri-modal-gate" not in cmd:
                    cmd.append("--tri-modal-gate")
                elif (not use_tmg) and "--tri-modal-gate" in cmd:
                    cmd.remove("--tri-modal-gate")
                if is_spec:
                    cmd.extend([
                        "--speculative",
                        "--batch-k",
                        "4",
                        "--spec-assistant",
                        str(args.assistant_v7_body),
                        "--spec-byte-head",
                        str(args.byte_head),
                    ])
                cmd.extend(["--prompt-text", prompt, "--max-tokens", str(int(max_toks))])
                _, parsed = _run_cmd(cmd, log_fp=lfp)
                parsed.update({"regime": rname, "tag": tag, "passed": bool(checker(str(parsed.get("gen_text", ""))))})
                rows.append(parsed)
                print(
                    f"[SPEC:{rname:16s}] {tag:12s} pass={int(parsed['passed'])} s_tok={parsed.get('s_per_tok')} "
                    f"acc={parsed.get('accept_rate')} dec={parsed.get('decode_s')}s hwm={parsed.get('vm_hwm_kb')} "
                    f"text={parsed.get('gen_text')!r}",
                    flush=True,
                )
    return rows


def _self_test() -> int:
    """Mandatory 6-gate fail-closed verification battery."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Contamination detector verifies zero overlap on held-out passages & battery prompts
    try:
        dj = e61.assert_holdouts_disjoint()
        gate("Gate 1: Disjoint held-out benchmark integrity", dj["overlap_4grams"] == 0 and dj["battery_prompts"] == 15, f"p={dj['passages']} q={dj['battery_prompts']}")
    except Exception as exc:
        gate("Gate 1: Disjoint held-out benchmark integrity", False, str(exc))

    # Gate 2: Pareto spec parser validates bounds and builds exact V9.2 flags
    try:
        cfgs = parse_pareto_spec("dense:1.0:0.0:1.0,crtr75_hr85:0.75:0.0:0.85")
        assert len(cfgs) == 2 and cfgs[1][1] == 0.75 and cfgs[1][3] == 0.85
        bad_raised = False
        try:
            parse_pareto_spec("broken:1.5:0.0")
        except ValueError:
            bad_raised = True
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            for fn in ("run31b", "m.t16c", "cfg.json", "tok.json", "r.crtr", "hr.crtr"):
                (td / fn).write_text("x", encoding="utf-8")
            ns = argparse.Namespace(
                run31b=td / "run31b", t16c=td / "m.t16c", config=td / "cfg.json", tokenizer=td / "tok.json",
                c8d=None, router=td / "r.crtr", head_router=td / "hr.crtr", so=None, threads=4,
                andvcnt_bits=6, kv_quant="glyph-lod", lmhead_cs="exact", shared_mlp_blocks=8,
                profile_components=True, tri_modal_gate=False,
            )
            cmd = build_base_cmd(ns, mlp_top_p=0.75, mod_thresh=0.0, head_top_p=0.85)
            ok_flags = ("--andvcnt-bits" in cmd and "6" in cmd and "--head-router" in cmd and "--shared-mlp-blocks" in cmd)
        gate("Gate 2: Pareto parser & CLI flag builder", bad_raised and ok_flags, f"flags={len(cmd)}")
    except Exception as exc:
        gate("Gate 2: Pareto parser & CLI flag builder", False, str(exc))

    # Gate 3: Extended telemetry parser extracts PROFILE31B + ATTN_SPARSE + SPEC31B + NLL
    try:
        sample = (
            "EVAL_NLL tokens=92 eval_start=28 eval_pos=63 mean_nll=6.21000 ppl=497.700 top1_acc=0.3500 mean_entropy=1.9000 eval_s=110.50\n"
            "EVAL_TEACHER_KL pos=63 kl_t_to_s=3.10000 kl_s_to_t=4.20000 top1_agree=0.4100 top5_agree=0.5900 teacher_nll=5.26875 teacher_ppl=194.17\n"
            "CORTICAL_ROUTER gate_skipped_blks=120 gate_total_blks=480 gate_saved_pct=25.0%\n"
            "ATTN_SPARSE skipped_heads=48 total_heads=320 saved_pct=15.0%\n"
            "PROFILE31B steps=10 attn_s=5.100 gate_s=3.200 up_s=3.100 down_s=4.000 norm_kv_s=0.400 lmhead_s=0.350 avg_step_s=1.615\n"
            "SPEC31B max_batch_k=4 ngram=4 min_count=1 seed_entries=0 assistant_mb=66.98 byte_head=true step_pos=true min_margin=0.25 margin_stopped=1 ast_lm_skipped=0 drafted=6 accepted=3 batches=3 fallback_k1=2 accept_rate=0.500 generated=8 s_per_tok=1.850\n"
            "GEN31B ids=[818, 1354, 563, 236812, 106]\n"
            "GEN31B text=\"The result is 4.<turn|>\"\n"
            "MEM31B pinned_bytes=6000000000 embed_bytes=539492352 ssd_read_bytes=0 cs_skipped_blocks=0 lmhead_cs_skipped_blocks=900 prefill_s=20.00 decode_s=14.80 vm_hwm_kb=Some(7100000) vm_rss_kb=Some(7050000)\n"
        )
        p = parse_extended_telemetry(sample)
        ok_p = (
            abs(float(p["mean_nll"]) - 6.21) < 1e-4
            and abs(float(p["head_saved_pct"]) - 15.0) < 1e-4
            and p["profile"]["steps"] == 10
            and p["n_gen_tokens"] == 5
            and p["lmhead_cs_skipped_blocks"] == 900
            and abs(float(p["s_per_tok"]) - 1.85) < 1e-4
            and int(p["vm_hwm_kb"]) == 7100000
        )
        gate("Gate 3: Extended telemetry regex & field types", ok_p, f"head_save={p.get('head_saved_pct')}")
    except Exception as exc:
        gate("Gate 3: Extended telemetry regex & field types", False, str(exc))

    # Gate 4: Deterministic checkers handle P8_romeo, B5_multistep, U4_syllogism accurately
    try:
        by_tag = {t: chk for t, _, _, chk in e61.BATTERY_PROMPTS}
        ok_chk = (
            by_tag["P8_romeo"]("William Shakespeare wrote Romeo and Juliet.<turn|>")
            and not by_tag["P8_romeo"]("It seems you are referring to Romeo")
            and by_tag["B5_multistep"]("12 * 8 = 96. 96 - 27 = 69.<turn|>")
            and not by_tag["B5_multistep"]("12 * 8 = 96")
            and by_tag["U4_syllogism"]("No, whales are warm-blooded mammals.")
            and not by_tag["U4_syllogism"]("Yes, whales are cold-blooded.")
        )
        gate("Gate 4: Deterministic battery checker calibration", ok_chk)
    except Exception as exc:
        gate("Gate 4: Deterministic battery checker calibration", False, str(exc))

    # Gate 5: Fail closed on missing required artifacts
    try:
        ns_bad = argparse.Namespace(
            run31b=Path("/nonexistent/run31b"), t16c=Path("/nonexistent/m.t16c"),
            config=Path("/nonexistent/cfg.json"), tokenizer=Path("/nonexistent/tok.json"),
        )
        raised = False
        try:
            validate_required_paths(ns_bad)
        except FileNotFoundError:
            raised = True
        gate("Gate 5: Fail-closed FileNotFoundError on missing assets", raised)
    except Exception as exc:
        gate("Gate 5: Fail-closed FileNotFoundError on missing assets", False, str(exc))

    # Gate 6: End-to-end aggregation & JSON serialization roundtrip
    try:
        with tempfile.TemporaryDirectory() as td_s:
            rp = Path(td_s) / "report.json"
            agg = _aggregate_passages(
                [{"eval_tokens": 90, "eval_pos": 60, "mean_nll": 6.0, "top1_acc": 0.4, "kl_t_to_s": 3.0, "teacher_top1_agree": 0.45, "teacher_nll": 5.2, "eval_s": 100.0, "vm_hwm_kb": 7000000, "ssd_read_bytes": 0}],
                "dense_glyph-lod", "glyph-lod", 1.0, 0.0, 1.0,
            )
            rp.write_text(json.dumps({"v9_2": agg}, indent=2), encoding="utf-8")
            back = json.loads(rp.read_text(encoding="utf-8"))
            gate("Gate 6: Passage aggregation & JSON roundtrip", back["v9_2"]["mean_nll"] == 6.0 and back["v9_2"]["ssd_read_bytes"] == 0)
    except Exception as exc:
        gate("Gate 6: Passage aggregation & JSON roundtrip", False, str(exc))

    print(f"V9_2_EVAL_SELFTEST: {6 - failures}/6", flush=True)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="V9.2-CliffFix-VLM Raspberry Pi 4 Proof Suite")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=("nll-pareto", "battery", "vision", "spec-ab", "all"), default="all")
    ap.add_argument("--run31b", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/target/release/run31b"))
    ap.add_argument("--t16c", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_cliff_fix_vlm.t16c"))
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--c8d", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_c8d.c8d"))
    ap.add_argument("--router", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_router.crtr"))
    ap.add_argument("--head-router", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_head_router.crtr"))
    ap.add_argument("--so", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/libneon_stream_accelerator.so"))
    ap.add_argument("--teacher-logits", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v6_1/teacher_lgt1"))
    ap.add_argument("--assistant-v7-body", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v6_1/gemma4_31b_assistant_v7_body_q2.t16c"))
    ap.add_argument("--byte-head", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v6_1/gemma4_31b_assistant_v7.bhdr"))
    ap.add_argument("--image", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_1/frame_01_coco.png"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_2/eval_out"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--andvcnt-bits", type=int, default=6)
    ap.add_argument("--kv-quant", type=str, default="glyph-lod")
    ap.add_argument("--lmhead-cs", type=str, default="exact")
    ap.add_argument("--shared-mlp-blocks", type=int, default=8)
    ap.add_argument("--profile-components", action="store_true", default=True)
    ap.add_argument("--tri-modal-gate", action="store_true")
    ap.add_argument("--pareto-kv-modes", type=str, default="glyph-lod,fp8")
    ap.add_argument("--pareto-configs", type=str, default="dense:1.0:0.0:1.0,crtr_p80:0.80:0.0:1.0,crtr_p80_hr85:0.80:0.0:0.85")
    ap.add_argument("--battery-configs", type=str, default="dense:1.0:0.0:1.0")
    args = ap.parse_args()

    if len(sys.argv) == 1 or args.selftest:
        return _self_test()

    validate_required_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "andvcnt_bits": args.andvcnt_bits,
        "lmhead_cs": args.lmhead_cs,
        "kv_quant": args.kv_quant,
        "t16c": str(args.t16c),
    }
    if args.mode in ("nll-pareto", "all"):
        report["nll_pareto"] = run_nll_pareto(args, args.out_dir)
    if args.mode in ("battery", "all"):
        report["battery"] = run_battery(args, args.out_dir)
    if args.mode in ("vision", "all"):
        report["vision"] = run_vision(args, args.out_dir)
    if args.mode in ("spec-ab", "all"):
        report["spec_ab"] = run_spec_ab(args, args.out_dir)

    rep_path = args.out_dir / f"v9_2_{args.mode.replace('-', '_')}_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[REPORT WRITTEN] {rep_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

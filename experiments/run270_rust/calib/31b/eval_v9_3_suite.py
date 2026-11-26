#!/usr/bin/env python3
"""
V9.3-E2E-KL Physical Raspberry Pi 4 Proof & Telemetry Suite.

Extends `eval_v9_2_suite` and `eval_v6_1_skeptic_proof` with:
  - V9.3 artifact paths (`/mnt/SHE-VAULT/gemma4_v9_3/`)
  - Exact `--attn-head-top-p` flag (`run31b.rs`) + `--pin-layers` memory guard
  - V9.3 trained sparse routing point `crtr_p50_hr625` (`42/84` MLP, `20/32` heads)
  - Multi-window `k=2` / `k=4` speculative A/B with strict greedy token-parity checks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_v6_1_skeptic_proof as e61  # noqa: E402
import eval_v9_2_suite as e92  # noqa: E402


DEFAULT_V93_DIR = Path("/mnt/SHE-VAULT/gemma4_v9_3")
DEFAULT_PARETO_CONFIGS = (
    "dense:1.0:0.0:1.0,"
    "crtr_p80:0.80:0.0:1.0,"
    "crtr_p75_hr85:0.75:0.0:0.85,"
    "crtr_p50_hr625:0.50:0.0:0.625"
)
DEFAULT_BATTERY_CONFIGS = (
    "dense:1.0:0.0:1.0,"
    "crtr80_hr85:0.80:0.0:0.85,"
    "crtr50_hr625:0.50:0.0:0.625"
)


def read_mem_available_kb(meminfo_path: Path = Path("/proc/meminfo")) -> Optional[int]:
    """Read Linux `/proc/meminfo` MemAvailable in kB when available."""
    if not meminfo_path.is_file():
        return None
    text = meminfo_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, flags=re.MULTILINE)
    return int(m.group(1)) if m else None


def parse_batch_ks(spec: str) -> List[int]:
    """Parse comma-separated speculative `--batch-k` values (`k >= 2`)."""
    out: List[int] = []
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        k_val = int(tok)
        if k_val < 2:
            raise ValueError(f"Speculative batch-k must be >= 2, got {k_val}")
        if k_val not in out:
            out.append(k_val)
    if not out:
        raise ValueError("Empty --spec-batch-ks list")
    return out


def validate_required_paths(args: argparse.Namespace) -> None:
    """Fail closed immediately when required binary or V9.3 assets are absent."""
    e92.validate_required_paths(args)


def build_v93_cmd(
    args: argparse.Namespace,
    kv_quant: Optional[str] = None,
    mlp_top_p: float = 1.0,
    mod_thresh: float = 0.0,
    head_top_p: float = 1.0,
) -> List[str]:
    """Build exact `run31b` command vector with `--attn-head-top-p` and `--pin-layers`."""
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
    if getattr(args, "pin_layers", None):
        cmd.extend(["--pin-layers", str(int(args.pin_layers))])
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
        cmd.extend([
            "--head-router",
            str(args.head_router),
            "--attn-head-top-p",
            f"{head_top_p:.4f}",
            "--attn-shared-heads",
            str(int(getattr(args, "attn_shared_heads", 4))),
        ])
    return cmd


def run_nll_pareto(args: argparse.Namespace, out_dir: Path) -> List[dict]:
    """Run held-out 3-passage Teacher NLL/KL Pareto across KV modes and V9.3 sparse configs."""
    kv_modes = [k.strip() for k in args.pareto_kv_modes.split(",") if k.strip()]
    cfgs = e92.parse_pareto_spec(args.pareto_configs)
    all_summaries: List[dict] = []
    with open(out_dir / "nll_pareto_v9_3.log", "w", encoding="utf-8") as lfp:
        for kv_q in kv_modes:
            for name, mp, mt, hp in cfgs:
                p_rows: List[dict] = []
                for p_idx, (ptag, passage) in enumerate(e61.HELDOUT_NLL_PASSAGES):
                    lgt_p = args.teacher_logits / f"{ptag}.lgt1" if args.teacher_logits else None
                    q_txt = e61.HELDOUT_CHAT_QUESTIONS[p_idx] if p_idx < len(e61.HELDOUT_CHAT_QUESTIONS) else "Explain briefly."
                    prompt_s = e61.format_gemma4_chat_nll_turn(q_txt, passage)
                    cmd = build_v93_cmd(args, kv_quant=kv_q, mlp_top_p=mp, mod_thresh=mt, head_top_p=hp)
                    cmd.extend(["--eval-nll", "--eval-nll-after-turn", "--raw-prompt", "--prompt-text", prompt_s])
                    if lgt_p is not None and lgt_p.is_file():
                        cmd.extend(["--eval-teacher-logits", str(lgt_p)])
                    _, parsed = e92._run_cmd(cmd, log_fp=lfp)
                    parsed["passage"] = ptag
                    p_rows.append(parsed)
                summary = e92._aggregate_passages(p_rows, f"{name}_{kv_q}", kv_q, mp, mt, hp)
                all_summaries.append(summary)
                print(
                    f"[PARETO] {summary['config']:24s} nll={summary['mean_nll']:.5f} ppl={summary['ppl']:.2f} "
                    f"kl_t2s={summary['kl_t_to_s']} agree1={summary['teacher_top1_agree']} "
                    f"mlp_save={summary['gate_saved_pct']} head_save={summary['head_saved_pct']} "
                    f"hwm={summary['vm_hwm_kb']}",
                    flush=True,
                )
    return all_summaries


def run_battery(args: argparse.Namespace, out_dir: Path) -> Dict[str, dict]:
    """Run all 15 deterministic prompts under each V9.3 dense/sparse configuration."""
    cfgs = e92.parse_pareto_spec(args.battery_configs)
    suites: Dict[str, dict] = {}
    with open(out_dir / "battery_15_v9_3.log", "w", encoding="utf-8") as lfp:
        for name, mp, mt, hp in cfgs:
            rows: List[dict] = []
            for tag, prompt, max_toks, checker in e61.BATTERY_PROMPTS:
                lfp.write(f"\n=== [{name}] {tag}: {prompt} ===\n")
                cmd = build_v93_cmd(args, kv_quant=args.kv_quant, mlp_top_p=mp, mod_thresh=mt, head_top_p=hp)
                cmd.extend(["--prompt-text", prompt, "--max-tokens", str(int(max_toks))])
                _, parsed = e92._run_cmd(cmd, log_fp=lfp)
                gen_txt = str(parsed.get("gen_text", ""))
                ok = bool(checker(gen_txt))
                parsed.update({"tag": tag, "prompt": prompt, "passed": ok})
                rows.append(parsed)
                print(
                    f"[BAT:{name}] {tag:15s} pass={int(ok)} pre={parsed.get('prefill_s')}s "
                    f"dec={parsed.get('decode_s')}s s_tok={parsed.get('s_per_tok')} text={gen_txt!r}",
                    flush=True,
                )
            passed = sum(1 for r in rows if r["passed"])
            s_toks = [float(r["s_per_tok"]) for r in rows if r.get("s_per_tok") is not None]
            suites[name] = {
                "passed": passed,
                "total": len(rows),
                "mean_s_per_tok": round(sum(s_toks) / max(len(s_toks), 1), 3) if s_toks else None,
                "vm_hwm_kb": max((int(r.get("vm_hwm_kb", 0)) for r in rows), default=0),
                "ssd_read_bytes": sum(int(r.get("ssd_read_bytes", 0)) for r in rows),
                "rows": rows,
            }
            print(
                f"[BATTERY SUMMARY:{name}] {passed}/{len(rows)} passed "
                f"mean_s_tok={suites[name]['mean_s_per_tok']} hwm={suites[name]['vm_hwm_kb']}",
                flush=True,
            )
    return suites


def run_vision(args: argparse.Namespace, out_dir: Path) -> dict:
    """Run multimodal COCO Vision QA on held-out image."""
    if args.image is None or not args.image.is_file():
        raise FileNotFoundError(f"Missing COCO test image --image: {args.image}")
    with open(out_dir / "vision_v9_3.log", "w", encoding="utf-8") as lfp:
        cmd = build_v93_cmd(args, kv_quant=args.kv_quant, mlp_top_p=1.0, mod_thresh=0.0, head_top_p=1.0)
        cmd.extend([
            "--image",
            str(args.image),
            "--prompt-text",
            "Describe what you see in this image in one short sentence.",
            "--max-tokens",
            "28",
        ])
        _, parsed = e92._run_cmd(cmd, log_fp=lfp)
    print(
        f"[VISION] text={parsed.get('gen_text')!r} pre={parsed.get('prefill_s')}s "
        f"dec={parsed.get('decode_s')}s hwm={parsed.get('vm_hwm_kb')} ssd={parsed.get('ssd_read_bytes')}",
        flush=True,
    )
    return parsed


def run_spec_ab(args: argparse.Namespace, out_dir: Path) -> List[dict]:
    """Compare Dense k=1 vs matched Assistant-V9 (`Q2` / optional `Q4`) speculative `k=2,4` with greedy parity."""
    body_specs: List[Tuple[str, Path]] = []
    if args.assistant_body is None or not args.assistant_body.is_file():
        raise FileNotFoundError(f"Missing --assistant-body: {args.assistant_body}")
    body_specs.append(("q2", args.assistant_body))
    if getattr(args, "include_q4_spec", False):
        if args.assistant_body_q4 is None or not args.assistant_body_q4.is_file():
            raise FileNotFoundError(f"Missing --assistant-body-q4: {args.assistant_body_q4}")
        body_specs.append(("q4", args.assistant_body_q4))
    if args.byte_head is None or not args.byte_head.is_file():
        raise FileNotFoundError(f"Missing --byte-head: {args.byte_head}")

    batch_ks = parse_batch_ks(args.spec_batch_ks)
    prompt_tags = {"P1_arith", "P2_france", "P5_speed", "B5_multistep"}
    selected = [p for p in e61.BATTERY_PROMPTS if p[0] in prompt_tags]

    # `(regime_name, is_spec, batch_k, use_tmg, body_path)`
    regimes: List[Tuple[str, bool, int, bool, Optional[Path]]] = [("dense_k1", False, 1, False, None)]
    for btag, bpath in body_specs:
        suf = "" if btag == "q2" else f"_{btag}"
        for bk in batch_ks:
            regimes.append((f"bhdr{suf}_spec_k{bk}", True, bk, False, bpath))
            regimes.append((f"bhdr{suf}_spec_k{bk}_tmg", True, bk, True, bpath))

    dense_outputs: Dict[str, str] = {}
    rows: List[dict] = []
    with open(out_dir / "spec_bhdr_v9_3.log", "w", encoding="utf-8") as lfp:
        for rname, is_spec, bk, use_tmg, bpath in regimes:
            for tag, prompt, max_toks, checker in selected:
                lfp.write(f"\n=== [{rname}] {tag} ===\n")
                cmd = build_v93_cmd(args, kv_quant=args.kv_quant, mlp_top_p=1.0, mod_thresh=0.0, head_top_p=1.0)
                if use_tmg and "--tri-modal-gate" not in cmd:
                    cmd.append("--tri-modal-gate")
                elif (not use_tmg) and "--tri-modal-gate" in cmd:
                    cmd.remove("--tri-modal-gate")
                if is_spec and bpath is not None:
                    cmd.extend([
                        "--speculative",
                        "--batch-k",
                        str(int(bk)),
                        "--spec-assistant",
                        str(bpath),
                        "--spec-byte-head",
                        str(args.byte_head),
                    ])
                cmd.extend(["--prompt-text", prompt, "--max-tokens", str(int(max_toks))])
                _, parsed = e92._run_cmd(cmd, log_fp=lfp)
                gen_txt = str(parsed.get("gen_text", ""))
                if not is_spec:
                    dense_outputs[tag] = gen_txt
                exact_vs_dense = (gen_txt == dense_outputs.get(tag, gen_txt))
                parsed.update({
                    "regime": rname,
                    "batch_k": bk,
                    "tag": tag,
                    "passed": bool(checker(gen_txt)),
                    "exact_match_vs_dense_k1": exact_vs_dense,
                })
                rows.append(parsed)
                print(
                    f"[SPEC:{rname:22s}] {tag:12s} pass={int(parsed['passed'])} parity={int(exact_vs_dense)} "
                    f"s_tok={parsed.get('s_per_tok')} acc={parsed.get('accept_rate')} "
                    f"dec={parsed.get('decode_s')}s hwm={parsed.get('vm_hwm_kb')} text={gen_txt!r}",
                    flush=True,
                )
    return rows


def _self_test() -> int:
    """Mandatory 6-gate fail-closed verification battery for V9.3 evaluation harness."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Contamination detector verifies zero 4-gram overlap on held-out passages & 15 battery prompts
    try:
        dj = e61.assert_holdouts_disjoint()
        gate(
            "Gate 1: Disjoint held-out benchmark integrity",
            dj["overlap_4grams"] == 0 and dj["battery_prompts"] == 15,
            f"p={dj['passages']} q={dj['battery_prompts']}",
        )
    except Exception as exc:
        gate("Gate 1: Disjoint held-out benchmark integrity", False, str(exc))

    # Gate 2: V9.3 Pareto parser & exact `run31b` CLI flags (`--attn-head-top-p`, `--pin-layers`)
    try:
        cfgs = e92.parse_pareto_spec(DEFAULT_PARETO_CONFIGS)
        assert len(cfgs) == 4 and cfgs[3] == ("crtr_p50_hr625", 0.50, 0.0, 0.625)
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            for fn in ("run31b", "m.t16c", "cfg.json", "tok.json", "r.crtr", "hr.crtr"):
                (td / fn).write_text("x", encoding="utf-8")
            ns = argparse.Namespace(
                run31b=td / "run31b",
                t16c=td / "m.t16c",
                config=td / "cfg.json",
                tokenizer=td / "tok.json",
                c8d=None,
                router=td / "r.crtr",
                head_router=td / "hr.crtr",
                so=None,
                threads=4,
                andvcnt_bits=6,
                kv_quant="glyph-lod",
                lmhead_cs="exact",
                shared_mlp_blocks=8,
                attn_shared_heads=4,
                pin_layers=58,
                profile_components=True,
                tri_modal_gate=False,
            )
            cmd = build_v93_cmd(ns, mlp_top_p=0.50, mod_thresh=0.0, head_top_p=0.625)
            ok_flags = (
                "--attn-head-top-p" in cmd
                and "0.6250" in cmd
                and "--head-top-p" not in cmd
                and "--attn-shared-heads" in cmd
                and "--pin-layers" in cmd
                and "58" in cmd
                and "--shared-mlp-blocks" in cmd
            )
        gate("Gate 2: V9.3 Pareto & exact run31b CLI flags", ok_flags, f"flags={len(cmd)}")
    except Exception as exc:
        gate("Gate 2: V9.3 Pareto & exact run31b CLI flags", False, str(exc))

    # Gate 3: Speculative batch-k parser & invalid value rejection
    try:
        ks = parse_batch_ks("2,4,2")
        bad_k = False
        try:
            parse_batch_ks("1,4")
        except ValueError:
            bad_k = True
        gate("Gate 3: Speculative multi-window batch-k validation", ks == [2, 4] and bad_k, f"ks={ks}")
    except Exception as exc:
        gate("Gate 3: Speculative multi-window batch-k validation", False, str(exc))

    # Gate 4: Extended telemetry parser extracts PROFILE31B + ATTN_SPARSE + SPEC31B + NLL
    try:
        sample = (
            "EVAL_NLL tokens=92 eval_start=28 eval_pos=63 mean_nll=6.12345 ppl=456.43 top1_acc=0.3800 mean_entropy=1.8000 eval_s=98.40\n"
            "EVAL_TEACHER_KL pos=63 kl_t_to_s=2.95000 kl_s_to_t=4.05000 top1_agree=0.4400 top5_agree=0.6200 teacher_nll=5.26875 teacher_ppl=194.17\n"
            "CORTICAL_ROUTER gate_skipped_blks=240 gate_total_blks=480 gate_saved_pct=50.0%\n"
            "ATTN_SPARSE skipped_heads=120 total_heads=320 saved_pct=37.5%\n"
            "PROFILE31B steps=10 attn_s=4.500 gate_s=2.800 up_s=2.700 down_s=3.600 norm_kv_s=0.380 lmhead_s=0.320 avg_step_s=1.430\n"
            "SPEC31B max_batch_k=2 ngram=4 min_count=1 seed_entries=0 assistant_mb=66.98 byte_head=true step_pos=true min_margin=0.25 margin_stopped=0 ast_lm_skipped=0 drafted=4 accepted=2 batches=2 fallback_k1=0 accept_rate=0.500 generated=5 s_per_tok=1.720\n"
            "GEN31B ids=[818, 1354, 563, 236812, 106]\n"
            'GEN31B text="The result is 4.<turn|>"\n'
            "MEM31B pinned_bytes=6440574208 embed_bytes=539492352 ssd_read_bytes=0 cs_skipped_blocks=0 lmhead_cs_skipped_blocks=900 prefill_s=18.00 decode_s=12.40 vm_hwm_kb=Some(7225000) vm_rss_kb=Some(7180000)\n"
        )
        p = e92.parse_extended_telemetry(sample)
        ok_p = (
            abs(float(p["mean_nll"]) - 6.12345) < 1e-4
            and abs(float(p["head_saved_pct"]) - 37.5) < 1e-4
            and abs(float(p["gate_saved_pct"]) - 50.0) < 1e-4
            and p["profile"]["avg_step_s"] == 1.43
            and abs(float(p["accept_rate"]) - 0.5) < 1e-4
            and int(p["vm_hwm_kb"]) == 7225000
        )
        gate("Gate 4: Extended telemetry extraction (CRTR+HeadRouter+Spec)", ok_p, f"head_save={p.get('head_saved_pct')}%")
    except Exception as exc:
        gate("Gate 4: Extended telemetry extraction (CRTR+HeadRouter+Spec)", False, str(exc))

    # Gate 5: Fail closed on missing required assets
    try:
        ns_bad = argparse.Namespace(
            run31b=Path("/nonexistent/run31b"),
            t16c=Path("/nonexistent/gemma4_31b_v9_3_e2e.t16c"),
            config=Path("/nonexistent/cfg.json"),
            tokenizer=Path("/nonexistent/tok.json"),
        )
        raised = False
        try:
            validate_required_paths(ns_bad)
        except FileNotFoundError:
            raised = True
        gate("Gate 5: Fail-closed FileNotFoundError on missing assets", raised)
    except Exception as exc:
        gate("Gate 5: Fail-closed FileNotFoundError on missing assets", False, str(exc))

    # Gate 6: Passage aggregation & strict finite-JSON roundtrip
    try:
        with tempfile.TemporaryDirectory() as td_s:
            rp = Path(td_s) / "v9_3_report.json"
            agg = e92._aggregate_passages(
                [
                    {
                        "eval_tokens": 92,
                        "eval_pos": 63,
                        "mean_nll": 6.25,
                        "top1_acc": 0.41,
                        "kl_t_to_s": 3.05,
                        "teacher_top1_agree": 0.46,
                        "teacher_nll": 5.26875,
                        "head_saved_pct": 37.5,
                        "gate_saved_pct": 50.0,
                        "eval_s": 98.4,
                        "vm_hwm_kb": 7225000,
                        "ssd_read_bytes": 0,
                    }
                ],
                "crtr_p50_hr625_glyph-lod",
                "glyph-lod",
                0.50,
                0.0,
                0.625,
            )
            rp.write_text(json.dumps({"v9_3": agg}, indent=2), encoding="utf-8")

            def _reject_nonfinite(tok: str) -> None:
                raise ValueError(f"Non-finite JSON token: {tok}")

            back = json.loads(rp.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
            ok_r = (
                back["v9_3"]["mean_nll"] == 6.25
                and back["v9_3"]["head_saved_pct"] == 37.5
                and back["v9_3"]["ssd_read_bytes"] == 0
            )
            gate("Gate 6: Passage aggregation & strict finite JSON roundtrip", ok_r)
    except Exception as exc:
        gate("Gate 6: Passage aggregation & strict finite JSON roundtrip", False, str(exc))

    print(f"V9_3_EVAL_SELFTEST: {6 - failures}/6", flush=True)
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="V9.3-E2E-KL Raspberry Pi 4 Proof Suite")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=("nll-pareto", "battery", "vision", "spec-ab", "all"), default="all")
    ap.add_argument("--run31b", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/target/release/run31b"))
    ap.add_argument("--t16c", type=Path, default=DEFAULT_V93_DIR / "gemma4_31b_v9_3_e2e.t16c")
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--c8d", type=Path, default=DEFAULT_V93_DIR / "gemma4_31b_v9_3_c8d.c8d")
    ap.add_argument("--router", type=Path, default=DEFAULT_V93_DIR / "gemma4_31b_v9_3_router.crtr")
    ap.add_argument("--head-router", type=Path, default=DEFAULT_V93_DIR / "gemma4_31b_v9_3_head_router.crtr")
    ap.add_argument("--so", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/libneon_stream_accelerator.so"))
    ap.add_argument("--teacher-logits", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v6_1/teacher_lgt1"))
    ap.add_argument("--assistant-body", type=Path, default=DEFAULT_V93_DIR / "gemma4_31b_v9_3_assistant_body_q2.t16c")
    ap.add_argument("--assistant-body-q4", type=Path, default=DEFAULT_V93_DIR / "gemma4_31b_v9_3_assistant_body_q4.t16c")
    ap.add_argument("--include-q4-spec", action="store_true")
    ap.add_argument("--byte-head", type=Path, default=DEFAULT_V93_DIR / "gemma4_31b_v9_3_assistant.bhdr")
    ap.add_argument("--image", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_1/frame_01_coco.png"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_V93_DIR / "eval_out")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--andvcnt-bits", type=int, default=6)
    ap.add_argument("--kv-quant", type=str, default="glyph-lod")
    ap.add_argument("--lmhead-cs", type=str, default="exact")
    ap.add_argument("--shared-mlp-blocks", type=int, default=8)
    ap.add_argument("--attn-shared-heads", type=int, default=4)
    ap.add_argument("--pin-layers", type=int, default=None)
    ap.add_argument("--profile-components", action="store_true", default=True)
    ap.add_argument("--tri-modal-gate", action="store_true")
    ap.add_argument("--pareto-kv-modes", type=str, default="glyph-lod,fp8")
    ap.add_argument("--pareto-configs", type=str, default=DEFAULT_PARETO_CONFIGS)
    ap.add_argument("--battery-configs", type=str, default=DEFAULT_BATTERY_CONFIGS)
    ap.add_argument("--spec-batch-ks", type=str, default="2,4")
    args = ap.parse_args(argv)

    if len(sys.argv) == 1 or args.selftest:
        return _self_test()

    validate_required_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mem_avail_kb = read_mem_available_kb()
    report: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "andvcnt_bits": args.andvcnt_bits,
        "lmhead_cs": args.lmhead_cs,
        "kv_quant": args.kv_quant,
        "pin_layers": args.pin_layers,
        "host_mem_available_kb": mem_avail_kb,
        "t16c": str(args.t16c),
    }
    if mem_avail_kb is not None:
        print(f"[PREFLIGHT] MemAvailable={mem_avail_kb} kB pin_layers={args.pin_layers}", flush=True)

    if args.mode in ("nll-pareto", "all"):
        report["nll_pareto"] = run_nll_pareto(args, args.out_dir)
    if args.mode in ("battery", "all"):
        report["battery"] = run_battery(args, args.out_dir)
    if args.mode in ("vision", "all"):
        report["vision"] = run_vision(args, args.out_dir)
    if args.mode in ("spec-ab", "all"):
        report["spec_ab"] = run_spec_ab(args, args.out_dir)

    rep_path = args.out_dir / f"v9_3_{args.mode.replace('-', '_')}_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[REPORT WRITTEN] {rep_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

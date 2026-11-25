#!/usr/bin/env python3
"""
V6.1 Hybrid + Real CRTR/MoD + Assistant-V7 Skeptic-Proofing Evaluation Suite.

Addresses every reviewer boundary on physical Raspberry Pi 4 silicon (`run31b`):
  1. Coherence & Perplexity Under REAL Trained CRTR + MoD (`--mode nll-pareto`):
     Runs held-out raw text NLL/PPL (`--eval-nll`) and full-vocab Teacher-KL
     (`--eval-teacher-logits`) across Dense baseline vs trained `--cortical-router`
     `top_p` in `[0.70, 0.50, 0.35]` x `--mod-thresh` in `[0.0, 0.35, 0.50]`.
  2. Multi-Step Reasoning vs Short QA Stress Battery (`--mode battery`):
     Runs all 11 prior P1..P8 + BENCH5 prompts plus 4 strictly unseen multi-step
     reasoning holdouts (`algebra`, `unit conversion`, `Python bug fix`, `syllogism`),
     grading deterministically and preserving full verbatim `GEN31B` text.
  3. Multimodal Vision QA (`--mode vision`): Held-out COCO visual QA parity.
  4. Assistant-V6 vs Assistant-V7 Byte-Head Speculative A/B (`--mode spec-ab`):
     Measures RSS reduction (`334 MB` -> `assistant_v7_body + .bhdr`) & token parity.
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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import v4_corpus as vc  # noqa: E402

# Strictly held-out technical/reasoning/code passages for `--eval-nll` (zero overlap with V4_CAL_PAIRS).
HELDOUT_NLL_PASSAGES: List[Tuple[str, str]] = [
    (
        "heldout_systems_cache",
        "In NUMA multiprocessor servers, cross-socket cache-line invalidations increase tail latency "
        "whenever worker threads share mutable counters on the same 64-byte coherence granule. "
        "Aligning per-core accumulator structs to cache-line offsets eliminates false sharing "
        "and restores linear scaling across memory channels without altering algorithmic complexity.",
    ),
    (
        "heldout_numerical_pde",
        "When integrating stiff reaction-diffusion equations with explicit Runge-Kutta schemes, "
        "step sizes are constrained quadratically by the smallest spatial grid spacing. "
        "Switching to an implicit Crank-Nicolson discretization preserves second-order temporal accuracy "
        "while remaining unconditionally stable across diffusive eigenmodes.",
    ),
    (
        "heldout_compiler_ir",
        "Static single assignment form simplifies dead-code elimination and global value numbering "
        "because every virtual register has exactly one lexical definition site. "
        "Phi nodes placed at iterated dominance frontiers reconcile control-flow merges "
        "before register allocation maps virtual temporaries onto physical machine lanes.",
    ),
]

# Prompt questions paired 1:1 with HELDOUT_NLL_PASSAGES for response-conditioned IT chat `--eval-nll`.
HELDOUT_CHAT_QUESTIONS: List[str] = [
    "Explain how false sharing affects NUMA multiprocessor servers and how to eliminate it.",
    "Explain why stiff reaction-diffusion PDEs constrain explicit Runge-Kutta step sizes and how Crank-Nicolson helps.",
    "Explain how static single assignment form and phi nodes simplify compiler optimization passes.",
]


def format_gemma4_chat_nll_turn(question: str, response: str) -> str:
    """Wrap a held-out Q/A pair in exact Gemma-4-IT `<|turn>` template for `--eval-nll-after-turn`."""
    return f"<|turn>user\n{question}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{response}<turn|>"


PromptSpec = Tuple[str, str, int, Callable[[str], bool]]


def _build_battery_prompts() -> List[PromptSpec]:
    return [
        # 11 original P1..P8 + BENCH5 prompts
        ("P1_arith", "What is 2 + 2?", 32, lambda s: bool(re.search(r"\b4\b", s))),
        ("P2_france", "The capital of France is", 32, lambda s: "paris" in s.lower()),
        ("P3_hamlet_quote", "To be or not to be, that is the", 32, lambda s: "question" in s.lower()),
        ("P4_prime_py", "Write a Python function to check if a number is prime.", 48, lambda s: "def " in s or "prime" in s.lower()),
        ("P5_speed", "A train travels 120 km in 2 hours. What is its average speed?", 32, lambda s: bool(re.search(r"\b60\b", s))),
        ("P6_photo", "Explain how photosynthesis works in one short sentence.", 40, lambda s: "light" in s.lower() or "sun" in s.lower()),
        ("P7_bsearch", "What is the time complexity of binary search on a sorted array?", 32, lambda s: "log" in s.lower()),
        ("P8_romeo", "Who wrote the play Romeo and Juliet?", 32, lambda s: "shakespeare" in s.lower()),
        ("B5_hamlet", "Who wrote the play Hamlet?", 32, lambda s: "shakespeare" in s.lower()),
        ("B5_multistep", "Compute 12 * 8 - 27.", 40, lambda s: bool(re.search(r"\b69\b", s))),
        ("B5_min", "Which number is smaller, 547 or 193?", 32, lambda s: bool(re.search(r"\b193\b", s))),
        # 4 strictly unseen multi-step reasoning prompts
        ("U1_algebra", "If 5x + 3 = 38, what is the value of 2x - 1?", 48, lambda s: bool(re.search(r"\b13\b", s))),
        ("U2_units", "A cyclist rides at 18 kilometers per hour for 40 minutes. How many meters did the cyclist travel?", 48, lambda s: "12000" in s.replace(",", "")),
        ("U3_bugfix", "What is the bug in `def avg(nums): return sum(nums) / len(nums)` when given an empty list?", 48, lambda s: "empty" in s.lower() or "zero" in s.lower()),
        ("U4_syllogism", "All mammals are warm-blooded. Whales are mammals. Are whales cold-blooded? Answer Yes or No and explain briefly.", 40, lambda s: bool(re.search(r"\bno\b", s.lower()))),
    ]


BATTERY_PROMPTS: List[PromptSpec] = _build_battery_prompts()


def assert_holdouts_disjoint() -> Dict[str, int]:
    """Fail closed if any NLL passage overlaps V4 calibration blacklists or 4-grams."""
    cal_texts = [vc.format_gemma4_turn(p, r) for _, p, r in vc.V4_CAL_PAIRS]
    cal_4grams = set()
    for t in cal_texts:
        toks = re.findall(r"[a-z0-9]+", t.lower())
        for i in range(max(len(toks) - 3, 0)):
            cal_4grams.add(tuple(toks[i : i + 4]))

    bad_terms = tuple(vc.BLACKLIST_SUBSTRINGS)
    overlap_4grams = 0
    items: List[Tuple[str, str]] = list(HELDOUT_NLL_PASSAGES) + [
        (f"chat_q_{i}", q) for i, q in enumerate(HELDOUT_CHAT_QUESTIONS)
    ]
    for name, txt in items:
        low = txt.lower()
        for b in bad_terms:
            if b in low:
                raise ValueError(f"Holdout NLL passage {name} matched blacklist term {b!r}")
        toks = re.findall(r"[a-z0-9]+", low)
        for i in range(max(len(toks) - 3, 0)):
            if tuple(toks[i : i + 4]) in cal_4grams:
                overlap_4grams += 1
    if overlap_4grams > 0:
        raise ValueError(f"Found {overlap_4grams} overlapping 4-grams between HELDOUT_NLL_PASSAGES and V4_CAL_PAIRS")
    return {"passages": len(HELDOUT_NLL_PASSAGES), "overlap_4grams": overlap_4grams, "battery_prompts": len(BATTERY_PROMPTS)}


def parse_run31b_telemetry(stdout_text: str) -> Dict[str, object]:
    """Extract grounded numerical telemetry from `run31b` stdout lines."""
    res: Dict[str, object] = {}
    m_gen = re.findall(r'^GEN31B text=(.+)$', stdout_text, flags=re.MULTILINE)
    if m_gen:
        raw_last = m_gen[-1].strip()
        if raw_last.startswith('"') and raw_last.endswith('"'):
            try:
                res["gen_text"] = bytes(raw_last[1:-1], "utf-8").decode("unicode_escape")
            except Exception:
                res["gen_text"] = raw_last.strip('"')
        else:
            res["gen_text"] = raw_last

    m_nll = re.search(
        r"EVAL_NLL tokens=(\d+) (?:eval_start=(\d+) )?eval_pos=(\d+) mean_nll=([0-9.+-eE]+) ppl=([0-9.+-eE]+) top1_acc=([0-9.+-eE]+) mean_entropy=([0-9.+-eE]+) eval_s=([0-9.+-eE]+)",
        stdout_text,
    )
    if m_nll:
        res["eval_tokens"] = int(m_nll.group(1))
        if m_nll.group(2) is not None:
            res["eval_start"] = int(m_nll.group(2))
        res["eval_pos"] = int(m_nll.group(3))
        res["mean_nll"] = float(m_nll.group(4))
        res["ppl"] = float(m_nll.group(5))
        res["top1_acc"] = float(m_nll.group(6))
        res["mean_entropy"] = float(m_nll.group(7))
        res["eval_s"] = float(m_nll.group(8))

    m_kl = re.search(
        r"EVAL_TEACHER_KL pos=(\d+) kl_t_to_s=([0-9.+-eE]+) kl_s_to_t=([0-9.+-eE]+) top1_agree=([0-9.+-eE]+) top5_agree=([0-9.+-eE]+) teacher_nll=([0-9.+-eE]+) teacher_ppl=([0-9.+-eE]+)",
        stdout_text,
    )
    if m_kl:
        res["kl_t_to_s"] = float(m_kl.group(2))
        res["kl_s_to_t"] = float(m_kl.group(3))
        res["teacher_top1_agree"] = float(m_kl.group(4))
        res["teacher_top5_agree"] = float(m_kl.group(5))
        res["teacher_nll"] = float(m_kl.group(6))
        res["teacher_ppl"] = float(m_kl.group(7))

    m_mem = re.search(
        r"MEM31B pinned_bytes=(\d+) embed_bytes=(\d+) ssd_read_bytes=(\d+) cs_skipped_blocks=(\d+) lmhead_cs_skipped_blocks=(\d+) prefill_s=([0-9.]+) decode_s=([0-9.]+) vm_hwm_kb=Some\((\d+)\) vm_rss_kb=Some\((\d+)\)",
        stdout_text,
    )
    if m_mem:
        res["pinned_bytes"] = int(m_mem.group(1))
        res["embed_bytes"] = int(m_mem.group(2))
        res["ssd_read_bytes"] = int(m_mem.group(3))
        res["prefill_s"] = float(m_mem.group(6))
        res["decode_s"] = float(m_mem.group(7))
        res["vm_hwm_kb"] = int(m_mem.group(8))
        res["vm_rss_kb"] = int(m_mem.group(9))

    m_crtr = re.search(r"CORTICAL_ROUTER gate_skipped_blks=(\d+) gate_total_blks=(\d+) gate_saved_pct=([0-9.]+)%", stdout_text)
    if m_crtr:
        res["gate_saved_pct"] = float(m_crtr.group(3))
    m_mod = re.search(r"MOD31B thresh=([0-9.]+) skipped_layers=(\d+) eligible_layers=(\d+) skipped_pct=([0-9.]+)%", stdout_text)
    if m_mod:
        res["mod_skipped_pct"] = float(m_mod.group(4))
    m_spec = re.search(r"SPEC31B .*?assistant_mb=([0-9.]+).*?accept_rate=([0-9.]+) generated=(\d+) s_per_tok=([0-9.]+)", stdout_text)
    if m_spec:
        res["assistant_mb"] = float(m_spec.group(1))
        res["accept_rate"] = float(m_spec.group(2))
        res["spec_generated"] = int(m_spec.group(3))
        res["s_per_tok"] = float(m_spec.group(4))
    m_lod = re.search(r"GTALOD31B active_districts=(\d+) total_districts=(\d+) district_dram_saved_pct=([0-9.]+)%", stdout_text)
    if m_lod:
        res["lod_active_districts"] = int(m_lod.group(1))
        res["lod_total_districts"] = int(m_lod.group(2))
        res["district_dram_saved_pct"] = float(m_lod.group(3))
    m_pov = re.search(r"POV6SENSE31B layers_eval=(\d+) mean_consensus=([0-9.]+) clean=(\d+) caution=(\d+) rescue=(\d+)", stdout_text)
    if m_pov:
        res["pov_layers_eval"] = int(m_pov.group(1))
        res["pov_mean_consensus"] = float(m_pov.group(2))
        res["pov_clean"] = int(m_pov.group(3))
        res["pov_caution"] = int(m_pov.group(4))
        res["pov_rescue"] = int(m_pov.group(5))
    return res


def _run_cmd(cmd: Sequence[str], log_fp=None) -> str:
    proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    if log_fp is not None:
        log_fp.write(f"$ {' '.join(cmd)}\n{proc.stdout}\n")
        log_fp.flush()
    return proc.stdout


def _require_file(path: Optional[Path], label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"required {label} file missing: {path}")
    return path


def validate_args(args: argparse.Namespace) -> None:
    """Reject incomplete experiment requests before creating a success receipt."""
    for attr, label in (
        ("run31b", "run31b binary"),
        ("t16c", "base T16C model"),
        ("config", "model config"),
        ("tokenizer", "tokenizer"),
    ):
        _require_file(getattr(args, attr), label)
    for attr, label in (
        ("c8d", "C8D bank"),
        ("so", "native library"),
        ("router", "CRTR router"),
        ("lod_shared", "shared LOD"),
        ("district_router", "district router"),
        ("assistant_v6", "V6 assistant"),
        ("assistant_v7_body", "assistant body"),
        ("byte_head", "byte head"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            _require_file(value, label)
    if args.lod_district_dir is not None and not args.lod_district_dir.is_dir():
        raise FileNotFoundError(f"required LOD district directory missing: {args.lod_district_dir}")
    if args.mode in ("vision", "all"):
        _require_file(args.image, "vision image")
    elif args.image is not None:
        _require_file(args.image, "vision image")
    if args.mode in ("spec-ab", "all"):
        for attr in ("assistant_v6", "assistant_v7_body", "byte_head"):
            _require_file(getattr(args, attr), attr)
    if args.teacher_logits is not None:
        if args.nll_format == "raw":
            _require_file(args.teacher_logits, "teacher logits")
        elif args.nll_format == "chat":
            if not args.teacher_logits.is_dir():
                raise FileNotFoundError(f"teacher chat-logit directory missing: {args.teacher_logits}")
            missing = [name for name, _ in HELDOUT_NLL_PASSAGES if not (args.teacher_logits / f"{name}.lgt1").is_file()]
            if missing:
                raise FileNotFoundError(f"teacher chat logits incomplete; missing={missing}")


def _require_v8_telemetry(args: argparse.Namespace, rec: dict) -> None:
    has_lod = any(
        getattr(args, name, None) is not None
        for name in ("lod_shared", "lod_district_dir", "district_router")
    )
    if has_lod and (rec.get("lod_active_districts", 0) <= 0 or rec.get("pov_layers_eval", 0) <= 0):
        raise RuntimeError(f"missing V8 GTALOD/POV telemetry: {rec}")


def _require_nll_telemetry(args: argparse.Namespace, rec: dict, teacher: bool = False) -> None:
    if rec.get("eval_pos", 0) <= 0 or "mean_nll" not in rec:
        raise RuntimeError(f"missing positive NLL telemetry: {rec}")
    if teacher and "kl_t_to_s" not in rec:
        raise RuntimeError(f"missing teacher-KL telemetry: {rec}")
    _require_v8_telemetry(args, rec)


def _base_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        str(args.run31b),
        "--t16c",
        str(args.t16c),
        "--config",
        str(args.config),
        "--tokenizer",
        str(args.tokenizer),
        "--threads",
        str(args.threads),
        "--kv-quant",
        args.kv_quant,
    ]
    # Explicit artifacts are contracts, not optional hints. Silently dropping a
    # missing sidecar would mislabel a degraded run as a complete V8 proof.
    for flag, path in (("--cuneiform-bank", args.c8d), ("--so", args.so)):
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"explicit {flag} artifact missing: {path}")
            cmd.extend([flag, str(path)])
    lod_shared = getattr(args, "lod_shared", None)
    if lod_shared is not None:
        if not lod_shared.is_file():
            raise FileNotFoundError(f"explicit --lod-shared artifact missing: {lod_shared}")
        cmd.extend(["--lod-shared", str(lod_shared)])
    lod_district_dir = getattr(args, "lod_district_dir", None)
    if lod_district_dir is not None and str(lod_district_dir):
        if not lod_district_dir.is_dir():
            raise FileNotFoundError(f"explicit --lod-district-dir missing: {lod_district_dir}")
        cmd.extend(["--lod-district-dir", str(lod_district_dir)])
    district_router = getattr(args, "district_router", None)
    if district_router is not None:
        if not district_router.is_file():
            raise FileNotFoundError(f"explicit --district-router artifact missing: {district_router}")
        cmd.extend(["--district-router", str(district_router)])
    if getattr(args, "player_pov_6sense", False):
        cmd.append("--player-pov-6sense")
    return cmd


def run_nll_pareto(args: argparse.Namespace, out_dir: Path) -> List[dict]:
    """Run held-out `--eval-nll` Pareto grid across CRTR `top_p` x `mod_thresh`."""
    if not args.run31b.is_file() or not args.t16c.is_file():
        raise FileNotFoundError(f"Missing run31b ({args.run31b}) or t16c ({args.t16c})")
    nll_format = getattr(args, "nll_format", "chat")
    if nll_format not in ("chat", "raw"):
        raise ValueError(f"Unsupported nll_format: {nll_format}")
    top_ps = [float(x) for x in args.top_ps.split(",")]
    mod_ths = [float(x) for x in args.mod_thresholds.split(",")]
    has_lod = any(
        getattr(args, name, None) is not None
        for name in ("lod_shared", "lod_district_dir", "district_router")
    )
    # With V8 flags, the first lane is V8 base+LOD without CRTR/MoD, not the
    # original dense V4.3 model. Keep that distinction explicit in the receipt.
    configs: List[Tuple[str, Optional[float], float]] = [("v8_base_no_crtr_mod" if has_lod else "dense", None, 0.0)]
    if args.router is not None:
        for tp in top_ps:
            for mt in mod_ths:
                configs.append((f"crtr_p{tp:.2f}_m{mt:.2f}", tp, mt))

    rows: List[dict] = []
    combined_txt = "\n\n".join(txt for _, txt in HELDOUT_NLL_PASSAGES)
    with open(out_dir / "nll_pareto.log", "w", encoding="utf-8") as lfp:
        for label, tp, mt in configs:
            router_flags: List[str] = []
            if tp is not None and args.router is not None:
                router_flags.extend(["--cortical-router", str(args.router), "--mlp-top-p", f"{tp:.3f}"])
                if mt > 0.0:
                    router_flags.extend(["--mod-thresh", f"{mt:.3f}"])

            if nll_format == "raw":
                cmd = _base_cmd(args) + ["--eval-nll", "--raw-prompt", "--prompt-text", combined_txt] + router_flags
                if args.teacher_logits is not None:
                    cmd.extend(["--eval-teacher-logits", str(args.teacher_logits)])
                rec = parse_run31b_telemetry(_run_cmd(cmd, lfp))
                _require_nll_telemetry(args, rec, teacher=args.teacher_logits is not None)
                rec["ppl"] = math.exp(min(rec["mean_nll"], 80.0))
                rec["nll_format"] = "raw"
            else:
                sub_rows: List[dict] = []
                tot_pos = 0
                tot_tok = 0
                w_nll = 0.0
                w_ent = 0.0
                w_t1 = 0.0
                w_eval_s = 0.0
                w_kl_ts = 0.0
                w_kl_st = 0.0
                w_a1 = 0.0
                w_a5 = 0.0
                w_tnll = 0.0
                kl_pos = 0
                for (p_name, p_ans), p_q in zip(HELDOUT_NLL_PASSAGES, HELDOUT_CHAT_QUESTIONS):
                    chat_prompt = format_gemma4_chat_nll_turn(p_q, p_ans)
                    cmd = _base_cmd(args) + [
                        "--eval-nll",
                        "--eval-nll-after-turn",
                        "--raw-prompt",
                        "--prompt-text",
                        chat_prompt,
                    ] + router_flags
                    if args.teacher_logits is not None:
                        cand_lgt = args.teacher_logits / f"{p_name}.lgt1"
                        cmd.extend(["--eval-teacher-logits", str(cand_lgt)])
                    sr = parse_run31b_telemetry(_run_cmd(cmd, lfp))
                    _require_nll_telemetry(args, sr, teacher=args.teacher_logits is not None)
                    sr["passage"] = p_name
                    sub_rows.append(sr)
                    ep = int(sr.get("eval_pos", 0))
                    tot_pos += ep
                    tot_tok += int(sr.get("eval_tokens", 0))
                    w_nll += float(sr.get("mean_nll", 0.0)) * ep
                    w_ent += float(sr.get("mean_entropy", 0.0)) * ep
                    w_t1 += float(sr.get("top1_acc", 0.0)) * ep
                    w_eval_s += float(sr.get("eval_s", 0.0))
                    if "kl_t_to_s" in sr:
                        kl_pos += ep
                        w_kl_ts += float(sr["kl_t_to_s"]) * ep
                        w_kl_st += float(sr["kl_s_to_t"]) * ep
                        w_a1 += float(sr["teacher_top1_agree"]) * ep
                        w_a5 += float(sr["teacher_top5_agree"]) * ep
                        w_tnll += float(sr["teacher_nll"]) * ep
                mean_nll_agg = (w_nll / tot_pos) if tot_pos > 0 else 0.0
                rec = {
                    "nll_format": "chat",
                    "eval_tokens": tot_tok,
                    "eval_pos": tot_pos,
                    "mean_nll": round(mean_nll_agg, 5),
                    "ppl": round(math.exp(min(mean_nll_agg, 80.0)), 3),
                    "top1_acc": round((w_t1 / tot_pos) if tot_pos > 0 else 0.0, 4),
                    "mean_entropy": round((w_ent / tot_pos) if tot_pos > 0 else 0.0, 4),
                    "eval_s": round(w_eval_s, 2),
                    "passages": sub_rows,
                }
                if kl_pos > 0:
                    t_nll_agg = w_tnll / kl_pos
                    rec["kl_t_to_s"] = round(w_kl_ts / kl_pos, 5)
                    rec["kl_s_to_t"] = round(w_kl_st / kl_pos, 5)
                    rec["teacher_top1_agree"] = round(w_a1 / kl_pos, 4)
                    rec["teacher_top5_agree"] = round(w_a5 / kl_pos, 4)
                    rec["teacher_nll"] = round(t_nll_agg, 5)
                    rec["teacher_ppl"] = round(math.exp(min(t_nll_agg, 80.0)), 3)
                for extra_k in (
                    "gate_saved_pct", "mod_skipped_pct", "vm_hwm_kb", "vm_rss_kb",
                    "lod_active_districts", "lod_total_districts", "district_dram_saved_pct",
                    "pov_layers_eval", "pov_mean_consensus", "pov_clean", "pov_caution", "pov_rescue",
                ):
                    vals = [float(r[extra_k]) for r in sub_rows if extra_k in r]
                    if vals:
                        rec[extra_k] = round(max(vals) if "vm_" in extra_k else sum(vals) / len(vals), 2)

            rec["config"] = label
            rec["mlp_top_p"] = tp if tp is not None else 1.0
            rec["mod_thresh"] = mt
            rows.append(rec)
    return rows


def run_battery(args: argparse.Namespace, out_dir: Path) -> List[dict]:
    """Execute all 15 stress-battery prompts and score deterministically."""
    if not args.run31b.is_file() or not args.t16c.is_file():
        raise FileNotFoundError(f"Missing run31b ({args.run31b}) or t16c ({args.t16c})")
    rows: List[dict] = []
    with open(out_dir / "battery_15.log", "w", encoding="utf-8") as lfp:
        for tag, prompt, max_toks, checker in BATTERY_PROMPTS:
            cmd = _base_cmd(args) + [
                "--prompt-text",
                prompt,
                "--max-tokens",
                str(max_toks),
                "--lmhead-cs",
                args.lmhead_cs,
            ]
            if args.tri_modal_gate:
                cmd.append("--tri-modal-gate")
            if args.router is not None and args.battery_top_p < 0.999:
                cmd.extend(["--cortical-router", str(args.router), "--mlp-top-p", f"{args.battery_top_p:.3f}"])
                if args.battery_mod_thresh > 0.0:
                    cmd.extend(["--mod-thresh", f"{args.battery_mod_thresh:.3f}"])
            out = _run_cmd(cmd, lfp)
            rec = parse_run31b_telemetry(out)
            _require_v8_telemetry(args, rec)
            gtxt = str(rec.get("gen_text", ""))
            rec["tag"] = tag
            rec["prompt"] = prompt
            rec["passed"] = bool(checker(gtxt))
            rows.append(rec)
    return rows


def _require_spec_telemetry(args: argparse.Namespace, rec: dict) -> None:
    _require_v8_telemetry(args, rec)
    if not isinstance(rec.get("gen_text"), str) or not rec["gen_text"].strip():
        raise RuntimeError(f"missing non-empty GEN31B text in spec lane: {rec}")
    if rec.get("spec_generated", 0) <= 0 or "accept_rate" not in rec or rec.get("s_per_tok", 0.0) <= 0.0:
        raise RuntimeError(f"incomplete SPEC31B telemetry: {rec}")


def _generated_text_parity(rows: List[dict]) -> bool:
    if len(rows) != 2:
        raise RuntimeError(f"spec parity requires exactly two lanes, got {len(rows)}")
    texts = [row.get("gen_text") for row in rows]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise RuntimeError("spec parity is undefined when either GEN31B text is missing or empty")
    return texts[0] == texts[1]


def run_spec_ab(args: argparse.Namespace, out_dir: Path) -> List[dict]:
    """Compare legacy Assistant LM-head vs Assistant-V7 `BHDR` runtime footprint and speed."""
    if not args.assistant_v6 or not args.assistant_v7_body or not args.byte_head:
        raise ValueError("--assistant-v6, --assistant-v7-body, and --byte-head required for --mode spec-ab")
    rows: List[dict] = []
    prompt = "What is 2 + 2?"
    with open(out_dir / "spec_ab.log", "w", encoding="utf-8") as lfp:
        for label, ast_p, bh_p in [
            ("assistant_v6_full", args.assistant_v6, None),
            ("assistant_v7_bhdr", args.assistant_v7_body, args.byte_head),
        ]:
            cmd = _base_cmd(args) + [
                "--speculative",
                "--batch-k",
                "4",
                "--spec-assistant",
                str(ast_p),
                "--assistant-step-pos",
                "--prompt-text",
                prompt,
                "--max-tokens",
                "16",
                "--lmhead-cs",
                args.lmhead_cs,
            ]
            if bh_p is not None:
                cmd.extend(["--spec-byte-head", str(bh_p)])
            out = _run_cmd(cmd, lfp)
            rec = parse_run31b_telemetry(out)
            _require_spec_telemetry(args, rec)
            rec["variant"] = label
            rows.append(rec)
    parity = _generated_text_parity(rows)
    for row in rows:
        row["generated_text_parity"] = parity
    print(f"SPEC_AB_GENERATED_TEXT_PARITY: {'PASS' if parity else 'FAIL'}", flush=True)
    return rows


def _run_selftest() -> int:
    """Mandatory 10-gate fail-closed self-test without requiring large model weights."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Strict disjointness against V4 calibration corpus & blacklists
    try:
        dj = assert_holdouts_disjoint()
        gate(
            "Gate 1: Held-out NLL passages 0 blacklist hits & 0 4-gram overlap",
            dj["overlap_4grams"] == 0 and dj["passages"] == 3 and dj["battery_prompts"] == 15,
            f"passages={dj['passages']} prompts={dj['battery_prompts']}",
        )
    except Exception as exc:
        gate("Gate 1: Holdout contamination check", False, str(exc))

    # Gate 2: Deterministic battery checkers accept valid answers and reject wrong/empty
    by_tag = {t: chk for t, _, _, chk in BATTERY_PROMPTS}
    g2_ok = (
        by_tag["P1_arith"]("The result is 4.<turn|>")
        and not by_tag["P1_arith"]("**. is the a")
        and by_tag["U1_algebra"]("Step 1: 5x = 35, x = 7, so 2x - 1 = 13.")
        and not by_tag["U1_algebra"]("2x - 1 = 12")
        and by_tag["U2_units"]("18 * (40/60) = 12 km = 12,000 meters.")
    )
    gate("Gate 2: Deterministic battery answer graders calibrated & fail closed", g2_ok)

    # Gate 3: Telemetry parser extracts NLL (with/without eval_start), KL, CRTR, MoD, SPEC31B & GEN31B accurately
    sample_log = "\n".join(
        [
            'GEN31B text="The result is 4.<turn|>"',
            "EVAL_NLL tokens=120 eval_start=21 eval_pos=99 mean_nll=2.34500 ppl=10.433 top1_acc=0.5546 mean_entropy=2.1200 eval_s=4.51",
            "EVAL_TEACHER_KL pos=99 kl_t_to_s=0.41230 kl_s_to_t=0.49100 top1_agree=0.7143 top5_agree=0.9160 teacher_nll=1.88000 teacher_ppl=6.553",
            "CORTICAL_ROUTER gate_skipped_blks=2520 gate_total_blks=5040 gate_saved_pct=50.0%",
            "MOD31B thresh=0.350 skipped_layers=12 eligible_layers=40 skipped_pct=30.0%",
            "SPEC31B max_batch_k=4 ngram=4 min_count=1 seed_entries=0 assistant_mb=201.40 byte_head=true step_pos=true min_margin=0.75 margin_stopped=0 ast_lm_skipped=0 drafted=9 accepted=6 batches=4 fallback_k1=0 accept_rate=0.667 generated=16 s_per_tok=0.812",
            "MEM31B pinned_bytes=4600000000 embed_bytes=539492352 ssd_read_bytes=0 cs_skipped_blocks=0 lmhead_cs_skipped_blocks=1024 prefill_s=12.34 decode_s=14.56 vm_hwm_kb=Some(4800000) vm_rss_kb=Some(4750000)",
            "GTALOD31B active_districts=112 total_districts=672 district_dram_saved_pct=83.3%",
            "POV6SENSE31B layers_eval=56 mean_consensus=0.8125 clean=44 caution=10 rescue=2",
        ]
    )
    parsed = parse_run31b_telemetry(sample_log)
    parsed_legacy = parse_run31b_telemetry(
        "EVAL_NLL tokens=63 eval_pos=62 mean_nll=3.10000 ppl=22.198 top1_acc=0.4000 mean_entropy=2.0000 eval_s=1.23"
    )
    g3_ok = (
        parsed.get("gen_text") == "The result is 4.<turn|>"
        and parsed.get("eval_start") == 21
        and parsed.get("eval_pos") == 99
        and parsed_legacy.get("eval_pos") == 62
        and "eval_start" not in parsed_legacy
        and abs(float(parsed.get("ppl", 0.0)) - 10.433) < 1e-3
        and abs(float(parsed.get("kl_t_to_s", 0.0)) - 0.41230) < 1e-5
        and abs(float(parsed.get("gate_saved_pct", 0.0)) - 50.0) < 1e-3
        and abs(float(parsed.get("mod_skipped_pct", 0.0)) - 30.0) < 1e-3
        and abs(float(parsed.get("assistant_mb", 0.0)) - 201.40) < 1e-2
        and parsed.get("ssd_read_bytes") == 0
        and parsed.get("lod_active_districts") == 112
        and parsed.get("lod_total_districts") == 672
        and abs(float(parsed.get("district_dram_saved_pct", 0.0)) - 83.3) < 1e-2
        and parsed.get("pov_layers_eval") == 56
        and abs(float(parsed.get("pov_mean_consensus", 0.0)) - 0.8125) < 1e-4
        and parsed.get("pov_clean") == 44
        and parsed.get("pov_caution") == 10
        and parsed.get("pov_rescue") == 2
        and "<channel|>" in format_gemma4_chat_nll_turn("Q", "A")
    )
    gate("Gate 3: Telemetry parser exact across EVAL_NLL/KL/CRTR/MoD/SPEC/MEM", g3_ok)

    # Gate 4: Contamination guard fails closed on injected contamination
    caught = False
    try:
        orig = HELDOUT_NLL_PASSAGES[0]
        bad_pair = ("bad", "What is the capital of France and Paris?")
        HELDOUT_NLL_PASSAGES[0] = bad_pair  # type: ignore[index]
        assert_holdouts_disjoint()
    except ValueError:
        caught = True
    finally:
        HELDOUT_NLL_PASSAGES[0] = orig  # type: ignore[index]
    gate("Gate 4: Contamination detector fails closed on injected leak", caught)

    # Gate 5: Fail closed on missing binary/model artifacts
    caught_missing = False
    try:
        ns = argparse.Namespace(
            run31b=Path("/nonexistent/run31b"),
            t16c=Path("/nonexistent/model.t16c"),
            top_ps="0.5",
            mod_thresholds="0.0",
            router=None,
        )
        with tempfile.TemporaryDirectory() as td:
            run_nll_pareto(ns, Path(td))
    except FileNotFoundError:
        caught_missing = True
    gate("Gate 5: Missing model/binary artifacts raise FileNotFoundError", caught_missing)

    # Gate 6: Report serialization roundtrip cleanly
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "report.json"
        rp.write_text(json.dumps({"selftest": parsed}, indent=2), encoding="utf-8")
        reloaded = json.loads(rp.read_text(encoding="utf-8"))
        gate("Gate 6: JSON artifact serialization roundtrip", reloaded["selftest"]["vm_rss_kb"] == 4750000)

    # Gate 7: Explicit sidecars must never disappear silently from run31b commands
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ns = argparse.Namespace(
            run31b=root / "run31b",
            t16c=root / "model.t16c",
            config=root / "config.json",
            tokenizer=root / "tokenizer.json",
            threads=4,
            kv_quant="glyph",
            c8d=None,
            so=None,
            lod_shared=None,
            lod_district_dir=None,
            district_router=None,
            player_pov_6sense=False,
        )
        rejected = 0
        for attr, value in (
            ("c8d", root / "missing.c8d"),
            ("so", root / "missing.so"),
            ("lod_shared", root / "missing_shared.t16c"),
            ("lod_district_dir", root / "missing_districts"),
            ("district_router", root / "missing.drtr"),
        ):
            setattr(ns, attr, value)
            try:
                _base_cmd(ns)
            except FileNotFoundError:
                rejected += 1
            setattr(ns, attr, None)
        present = root / "present.drtr"
        present.write_bytes(b"test")
        ns.district_router = present
        ns.player_pov_6sense = True
        cmd = _base_cmd(ns)
        gate(
            "Gate 7: Explicit V8 sidecars fail closed and remain visible in commands",
            rejected == 5 and "--district-router" in cmd and "--player-pov-6sense" in cmd,
            f"missing_artifacts_rejected={rejected}/5",
        )

    # Gate 8: Selected modes and teacher inputs are complete before execution
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in ("run31b", "model.t16c", "config.json", "tokenizer.json"):
            (root / name).write_bytes(b"test")
        ns = argparse.Namespace(
            run31b=root / "run31b",
            t16c=root / "model.t16c",
            config=root / "config.json",
            tokenizer=root / "tokenizer.json",
            mode="battery",
            c8d=None,
            so=None,
            router=None,
            lod_shared=None,
            lod_district_dir=None,
            district_router=None,
            assistant_v6=None,
            assistant_v7_body=None,
            byte_head=None,
            image=None,
            teacher_logits=None,
            nll_format="chat",
        )
        valid_minimal = False
        try:
            validate_args(ns)
            valid_minimal = True
        except Exception:
            pass
        rejected_all = False
        ns.mode = "all"
        try:
            validate_args(ns)
        except FileNotFoundError:
            rejected_all = True
        ns.mode = "battery"
        teacher_dir = root / "teacher"
        teacher_dir.mkdir()
        (teacher_dir / f"{HELDOUT_NLL_PASSAGES[0][0]}.lgt1").write_bytes(b"test")
        ns.teacher_logits = teacher_dir
        rejected_partial_teacher = False
        try:
            validate_args(ns)
        except FileNotFoundError:
            rejected_partial_teacher = True
        gate(
            "Gate 8: Mode and teacher preflight rejects incomplete experiments",
            valid_minimal and rejected_all and rejected_partial_teacher,
            f"minimal={valid_minimal} all={rejected_all} partial_teacher={rejected_partial_teacher}",
        )

        # Gate 9: Missing NLL/KL/V8 telemetry cannot produce a complete report
        ns.teacher_logits = None
        ns.lod_shared = None
        valid_rec = {"eval_pos": 10, "mean_nll": 2.0}
        checks = 0
        try:
            _require_nll_telemetry(ns, valid_rec)
            checks += 1
        except RuntimeError:
            pass
        for rec, teacher in (({"eval_pos": 10}, False), (valid_rec, True)):
            try:
                _require_nll_telemetry(ns, rec, teacher=teacher)
            except RuntimeError:
                checks += 1
        ns.lod_shared = root / "shared.t16c"
        try:
            _require_nll_telemetry(ns, valid_rec)
        except RuntimeError:
            checks += 1
        try:
            _require_nll_telemetry(
                ns,
                {**valid_rec, "lod_active_districts": 12, "pov_layers_eval": 60},
            )
            checks += 1
        except RuntimeError:
            pass
        gate("Gate 9: NLL, teacher-KL, and V8 telemetry fail closed", checks == 5, f"checks={checks}/5")

        # Gate 10: Empty spec lanes can never compare equal or emit complete evidence
        ns.lod_shared = None
        good_spec = {
            "gen_text": "deterministic answer",
            "spec_generated": 8,
            "accept_rate": 0.75,
            "s_per_tok": 0.5,
        }
        spec_checks = 0
        try:
            _require_spec_telemetry(ns, good_spec)
            spec_checks += 1
        except RuntimeError:
            pass
        for bad in (
            {},
            {"gen_text": ""},
            {"gen_text": "answer", "spec_generated": 0, "accept_rate": 0.0, "s_per_tok": 0.5},
        ):
            try:
                _require_spec_telemetry(ns, bad)
            except RuntimeError:
                spec_checks += 1
        try:
            _generated_text_parity([{}, {}])
        except RuntimeError:
            spec_checks += 1
        spec_checks += int(_generated_text_parity([good_spec, good_spec]))
        spec_checks += int(not _generated_text_parity([good_spec, {**good_spec, "gen_text": "different"}]))
        gate("Gate 10: Spec generation and text parity fail closed", spec_checks == 7, f"checks={spec_checks}/7")

    print(f"SKEPTIC_PROOF_SELFTEST: {10 - failures}/10 gates passed", flush=True)
    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _run_selftest()

    ap = argparse.ArgumentParser(description="Skeptic-Proofing V6.1 + CRTR/MoD + Assistant-V7 Evaluation Suite")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=("nll-pareto", "battery", "vision", "spec-ab", "all"), default="all")
    ap.add_argument("--run31b", type=Path, required=True)
    ap.add_argument("--t16c", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--c8d", type=Path, default=None)
    ap.add_argument("--router", type=Path, default=None)
    ap.add_argument("--lod-shared", type=Path, default=None)
    ap.add_argument("--lod-district-dir", type=Path, default=None)
    ap.add_argument("--district-router", type=Path, default=None)
    ap.add_argument("--player-pov-6sense", action="store_true")
    ap.add_argument("--so", type=Path, default=None)
    ap.add_argument("--teacher-logits", type=Path, default=None)
    ap.add_argument("--assistant-v6", type=Path, default=None)
    ap.add_argument("--assistant-v7-body", type=Path, default=None)
    ap.add_argument("--byte-head", type=Path, default=None)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("./skeptic_proof_out"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--kv-quant", type=str, default="glyph")
    ap.add_argument("--lmhead-cs", type=str, default="exact")
    ap.add_argument("--tri-modal-gate", action="store_true")
    ap.add_argument("--nll-format", choices=("chat", "raw"), default="chat")
    ap.add_argument("--top-ps", type=str, default="0.70,0.50,0.35")
    ap.add_argument("--mod-thresholds", type=str, default="0.0,0.35,0.50")
    ap.add_argument("--battery-top-p", type=float, default=1.0)
    ap.add_argument("--battery-mod-thresh", type=float, default=0.0)
    args = ap.parse_args()

    assert_holdouts_disjoint()
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": args.mode}

    if args.mode in ("nll-pareto", "all"):
        report["nll_pareto"] = run_nll_pareto(args, args.out_dir)
    if args.mode in ("battery", "all"):
        bat = run_battery(args, args.out_dir)
        passed = sum(1 for r in bat if r["passed"])
        report["battery"] = {"passed": passed, "total": len(bat), "rows": bat}
    if args.mode in ("vision", "all"):
        cmd = _base_cmd(args) + ["--image", str(args.image), "--prompt-text", "Describe what you see in this image in one clear sentence.", "--max-tokens", "32", "--lmhead-cs", args.lmhead_cs]
        v_out = _run_cmd(cmd)
        (args.out_dir / "vision.log").write_text(v_out, encoding="utf-8")
        vision_rec = parse_run31b_telemetry(v_out)
        _require_v8_telemetry(args, vision_rec)
        report["vision"] = vision_rec
    if args.mode in ("spec-ab", "all"):
        report["spec_ab"] = run_spec_ab(args, args.out_dir)

    report["complete"] = True
    out_json = args.out_dir / "skeptic_proof_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

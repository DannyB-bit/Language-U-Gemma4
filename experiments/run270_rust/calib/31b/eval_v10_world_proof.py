#!/usr/bin/env python3
"""Language-U V10 Physical Raspberry Pi 4 World-Proof & 20-50 tok/s Cascade Verification Suite.

Verifies all five True-North gates on physical Pi 4 silicon (`run31b`):
  - Gate A: Residency `VmHWM <= 6.50 GiB` (6,815,744 kB), `VmSwap == 0`, `ssd_read_bytes == 0`.
  - Gate B: Held-out 3-passage Teacher NLL / KL beating V9.2 (`mean_nll < 7.68050`, `kl_t_to_s < 4.80107`).
  - Gate C: 15-prompt strictly unseen deterministic battery across:
      1) `v10_dense` (100% Dense 60/60 reference)
      2) `v10_tree_lossless` (Target-Embed-Shared EAGLE-3 Tree, exact greedy parity)
      3) `v10_cortical_cascade` (Confidence-Gated 3-Tier Cascade, Pure Neural)
      4) `v10_cortical_cascade_alu` (Pure Neural Cascade + General First-Principles ALU, reported separately)
  - Gate D: Sustained multi-prompt `tok/s` (`CASCADE31B`) throughput benchmark (`--max-tokens 64`).
  - Gate E: Held-out Multimodal COCO Vision QA (`frame_01_coco.png`).
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
import eval_v9_3_suite as e93  # noqa: E402

MAX_PI4_VM_HWM_KB = 6_815_744  # 6.50 GiB
DEFAULT_V10_DIR = Path("/mnt/SHE-VAULT/gemma4_v10")

SPEED_BENCH_PROMPTS: Tuple[Tuple[str, str], ...] = (
    ("speed_code", "Write a clean Python function to merge two sorted lists in O(n) time."),
    ("speed_sys", "Explain how virtual memory page tables map virtual addresses to physical frames cleanly."),
    ("speed_math", "Solve the system 2a + b = 11 and a - b = 1 step by step."),
)


def parse_v10_telemetry(stdout_text: str) -> Dict[str, object]:
    """Parse V10 telemetry (`CASCADE31B`, `ALU31B`, `SPEC31B`, `EVAL_NLL`, `MEM31B`)."""
    res = e92.parse_extended_telemetry(stdout_text)
    m_cas = re.search(
        r"CASCADE31B tau=([0-9.+-eE]+) tier2_accepted=(\d+) tier3_rescues=(\d+) tok_per_s=([0-9.+-eE]+) ms_per_tok=([0-9.+-eE]+)",
        stdout_text,
    )
    if m_cas:
        res["cascade_tau"] = float(m_cas.group(1))
        res["tier2_accepted"] = int(m_cas.group(2))
        res["tier3_rescues"] = int(m_cas.group(3))
        res["tok_per_s"] = float(m_cas.group(4))
        res["ms_per_tok"] = float(m_cas.group(5))
    m_alu = re.search(r'^ALU31B expr_result=(.+)$', stdout_text, flags=re.MULTILINE)
    if m_alu:
        raw_alu = m_alu.group(1).strip()
        try:
            res["alu_result"] = bytes(raw_alu.strip('"'), "utf-8").decode("unicode_escape")
        except Exception:
            res["alu_result"] = raw_alu.strip('"')
    res["target_shared_head"] = "target_shared_head=true" in stdout_text
    return res


def check_memory_gate(rec: Dict[str, object], max_hwm_kb: int = MAX_PI4_VM_HWM_KB) -> Tuple[bool, str]:
    """Enforce strict `VmHWM <= 6.50 GiB` and `ssd_read_bytes == 0`."""
    hwm = int(rec.get("vm_hwm_kb") or 0)
    ssd = int(rec.get("ssd_read_bytes") or 0)
    ok = (hwm > 0) and (hwm <= max_hwm_kb) and (ssd == 0)
    return ok, f"vm_hwm_kb={hwm}/{max_hwm_kb}, ssd_read_bytes={ssd}"


def build_v10_cmd(
    args: argparse.Namespace,
    *,
    mode: str,
    mlp_top_p: float = 1.0,
    head_top_p: float = 1.0,
    mod_thresh: float = 0.0,
    cascade_tau: Optional[float] = None,
    use_alu: bool = False,
    kv_quant: Optional[str] = None,
) -> List[str]:
    """Construct exact V10 `run31b` command for dense, lossless tree, or 3-tier cascade."""
    cmd = e93.build_v93_cmd(args, kv_quant=kv_quant, mlp_top_p=mlp_top_p, mod_thresh=mod_thresh, head_top_p=head_top_p)
    if mode in ("tree_lossless", "cascade", "cascade_alu"):
        cmd.extend([
            "--speculative",
            "--spec-tree", str(int(args.spec_tree)),
            "--spec-tree-depth", str(int(args.spec_tree_depth)),
            "--spec-assistant", str(args.assistant_body),
            "--spec-ngram", str(int(args.spec_ngram)),
        ])
        if args.spec_seed is not None and args.spec_seed.is_file():
            cmd.extend(["--spec-seed", str(args.spec_seed)])
        if getattr(args, "use_byte_head", False) and args.byte_head is not None and args.byte_head.is_file():
            cmd.extend(["--spec-byte-head", str(args.byte_head)])
    if mode in ("cascade", "cascade_alu"):
        tau = float(cascade_tau if cascade_tau is not None else args.cascade_tau)
        cmd.extend([
            "--cortical-router", str(args.router),
            "--cortical-cascade", f"{tau:.2f}",
            "--cascade-mlp-top-p", f"{float(args.cascade_mlp_top_p):.4f}",
            "--cascade-head-top-p", f"{float(args.cascade_head_top_p):.4f}",
            "--cascade-mod-thresh", f"{float(args.cascade_mod_thresh):.4f}",
            "--cascade-tier1-tau", f"{float(getattr(args, 'cascade_tier1_tau', 0.0)):.4f}",
            "--cascade-max-unverified", str(int(getattr(args, "cascade_max_unverified", 2))),
            "--kv-mid", str(int(getattr(args, "cascade_kv_mid", 64))),
        ])
        if int(getattr(args, "shared_mlp_blocks", 4)) > 0 and "--shared-mlp-blocks" not in cmd:
            cmd.extend(["--shared-mlp-blocks", str(int(args.shared_mlp_blocks))])
        if int(getattr(args, "attn_shared_heads", 4)) > 0 and "--attn-shared-heads" not in cmd:
            cmd.extend(["--attn-shared-heads", str(int(args.attn_shared_heads))])
        if args.head_router is not None and args.head_router.is_file() and "--head-router" not in cmd:
            cmd.extend(["--head-router", str(args.head_router)])
    if use_alu or mode == "cascade_alu":
        cmd.append("--general-alu")
    return cmd


def validate_required_paths(args: argparse.Namespace) -> None:
    """Fail closed immediately when required binary or V10 assets are missing."""
    for attr in ("run31b", "t16c", "config", "tokenizer", "router", "assistant_body"):
        p = getattr(args, attr, None)
        if p is None or not Path(p).is_file():
            raise FileNotFoundError(f"Missing required V10 asset --{attr.replace('_', '-')}: {p}")


def split_batch_stdout(stdout_text: str, expected_n: int) -> List[str]:
    """Split `run31b --prompt-batch` output cleanly on `=== BATCH_PROMPT i/N ... ===` markers."""
    if expected_n == 1 and "=== BATCH_PROMPT " not in stdout_text:
        return [stdout_text]
    chunks = re.split(r"^=== BATCH_PROMPT \d+/\d+ .* ===\s*$", stdout_text, flags=re.MULTILINE)
    if len(chunks) != expected_n + 1:
        raise ValueError(f"Expected {expected_n} BATCH_PROMPT blocks, got {len(chunks) - 1}")
    return chunks[1:]


def run_battery_v10(args: argparse.Namespace, out_dir: Path) -> Dict[str, dict]:
    """Run 15-prompt unseen battery across Dense, Lossless Tree, Cascade (Pure Neural), and Cascade+ALU."""
    modes = [m.strip() for m in args.battery_modes.split(",") if m.strip()]
    results: Dict[str, dict] = {}
    batch_file = out_dir / "battery_15_prompts_v10.txt"
    batch_lines = [f"MAX_TOKENS:{int(max_toks)}|{prompt}" for _tag, prompt, max_toks, _chk in e61.BATTERY_PROMPTS]
    batch_file.write_text("\n".join(batch_lines) + "\n", encoding="utf-8")

    with open(out_dir / "battery_15_v10.log", "w", encoding="utf-8") as lfp:
        ran_shared_cascade = False
        for mode in modes:
            if mode in ("cascade", "cascade_alu") and "cascade" in modes and "cascade_alu" in modes:
                if ran_shared_cascade:
                    continue
                ran_shared_cascade = True
                lfp.write("\n================ MODE [cascade + cascade_alu (disambiguated)] ================\n")
                cmd = build_v10_cmd(args, mode="cascade", use_alu=True)
                cmd.extend(["--prompt-batch", str(batch_file), "--max-tokens", "48"])
                raw_stdout, _ = e92._run_cmd(cmd, log_fp=lfp)
                blocks = split_batch_stdout(raw_stdout, len(e61.BATTERY_PROMPTS))
                rows_cas: List[dict] = []
                rows_alu: List[dict] = []
                for (tag, prompt, _max_toks, checker), blk in zip(e61.BATTERY_PROMPTS, blocks):
                    parsed_c = parse_v10_telemetry(blk)
                    gen_txt = str(parsed_c.get("gen_text", ""))
                    alu_txt = str(parsed_c.get("alu_result", ""))
                    ok_neural = bool(checker(gen_txt))
                    ok_alu = ok_neural or (bool(alu_txt) and bool(checker(alu_txt)))

                    rec_c = dict(parsed_c)
                    rec_c.pop("alu_result", None)
                    rec_c.update({"tag": tag, "prompt": prompt, "passed_pure_neural": ok_neural, "passed": ok_neural})
                    rows_cas.append(rec_c)

                    rec_a = dict(parsed_c)
                    rec_a.update({"tag": tag, "prompt": prompt, "passed_pure_neural": ok_neural, "passed": ok_alu})
                    rows_alu.append(rec_a)
                    print(
                        f"[V10:cascade] {tag:15s} neural={int(ok_neural)} alu_mode={int(ok_alu)} "
                        f"s_tok={parsed_c.get('s_per_tok')} hwm={parsed_c.get('vm_hwm_kb')} text={gen_txt!r}"
                        + (f" alu={alu_txt!r}" if alu_txt else ""),
                        flush=True,
                    )
                for sub_mode, sub_rows in (("cascade", rows_cas), ("cascade_alu", rows_alu)):
                    passed = sum(1 for r in sub_rows if r["passed"])
                    passed_neural = sum(1 for r in sub_rows if r["passed_pure_neural"])
                    s_toks = [float(r["s_per_tok"]) for r in sub_rows if r.get("s_per_tok") is not None]
                    mean_s = round(sum(s_toks) / max(len(s_toks), 1), 4) if s_toks else None
                    results[sub_mode] = {
                        "passed": passed,
                        "passed_pure_neural": passed_neural,
                        "total": len(sub_rows),
                        "mean_s_per_tok": mean_s,
                        "mean_tok_per_s": round(1.0 / mean_s, 2) if (mean_s and mean_s > 0) else None,
                        "vm_hwm_kb": max((int(r.get("vm_hwm_kb", 0)) for r in sub_rows), default=0),
                        "ssd_read_bytes": sum(int(r.get("ssd_read_bytes", 0)) for r in sub_rows),
                        "rows": sub_rows,
                    }
                continue

            lfp.write(f"\n================ MODE [{mode}] ================\n")
            cmd = build_v10_cmd(args, mode=mode, use_alu=(mode == "cascade_alu"))
            cmd.extend(["--prompt-batch", str(batch_file), "--max-tokens", "48"])
            raw_stdout, _ = e92._run_cmd(cmd, log_fp=lfp)
            blocks = split_batch_stdout(raw_stdout, len(e61.BATTERY_PROMPTS))
            rows: List[dict] = []
            for (tag, prompt, _max_toks, checker), blk in zip(e61.BATTERY_PROMPTS, blocks):
                parsed = parse_v10_telemetry(blk)
                gen_txt = str(parsed.get("gen_text", ""))
                alu_txt = str(parsed.get("alu_result", ""))
                ok_neural = bool(checker(gen_txt))
                ok_mode = ok_neural or (mode == "cascade_alu" and bool(alu_txt) and bool(checker(alu_txt)))
                parsed.update({
                    "tag": tag,
                    "prompt": prompt,
                    "passed_pure_neural": ok_neural,
                    "passed": ok_mode,
                })
                rows.append(parsed)
                print(
                    f"[V10:{mode}] {tag:15s} pass={int(ok_mode)} (neural={int(ok_neural)}) "
                    f"s_tok={parsed.get('s_per_tok')} hwm={parsed.get('vm_hwm_kb')} text={gen_txt!r}"
                    + (f" alu={alu_txt!r}" if alu_txt else ""),
                    flush=True,
                )
            passed = sum(1 for r in rows if r["passed"])
            passed_neural = sum(1 for r in rows if r["passed_pure_neural"])
            s_toks = [float(r["s_per_tok"]) for r in rows if r.get("s_per_tok") is not None]
            mean_s = round(sum(s_toks) / max(len(s_toks), 1), 4) if s_toks else None
            results[mode] = {
                "passed": passed,
                "passed_pure_neural": passed_neural,
                "total": len(rows),
                "mean_s_per_tok": mean_s,
                "mean_tok_per_s": round(1.0 / mean_s, 2) if (mean_s and mean_s > 0) else None,
                "vm_hwm_kb": max((int(r.get("vm_hwm_kb", 0)) for r in rows), default=0),
                "ssd_read_bytes": sum(int(r.get("ssd_read_bytes", 0)) for r in rows),
                "rows": rows,
            }
    return results


def run_speed_bench(args: argparse.Namespace, out_dir: Path) -> List[dict]:
    """Measure sustained multi-prompt `tok/s` (`CASCADE31B`) + `VmHWM` on 64-token generations."""
    import subprocess

    rows: List[dict] = []
    with open(out_dir / "speed_bench_v10.log", "w", encoding="utf-8") as lfp:
        for tag, prompt in SPEED_BENCH_PROMPTS:
            cmd = build_v10_cmd(args, mode="cascade")
            cmd.extend(["--prompt-text", prompt, "--max-tokens", str(int(args.speed_max_tokens))])
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            lfp.write(f"\n=== [SPEED] {tag} ===\n{proc.stdout}\n")
            lfp.flush()
            parsed = parse_v10_telemetry(proc.stdout)
            mem_ok, mem_msg = check_memory_gate(parsed)
            parsed.update({"tag": tag, "memory_gate_ok": mem_ok, "memory_detail": mem_msg})
            rows.append(parsed)
            print(
                f"[SPEED:{tag}] tok/s={parsed.get('tok_per_s')} ms/tok={parsed.get('ms_per_tok')} "
                f"t2_acc={parsed.get('tier2_accepted')} t3_res={parsed.get('tier3_rescues')} {mem_msg}",
                flush=True,
            )
    return rows


def run_tier1_fast_probe(args: argparse.Namespace, out_dir: Path) -> dict:
    """Isolated Tier-1 draft-cascade throughput ceiling probe (disambiguated from coherent 60L cascade)."""
    import subprocess

    cmd = build_v10_cmd(args, mode="cascade")
    # Replace --lmhead-cs exact and --cascade-tier1-tau / --cascade-max-unverified for Tier-1 hardware ceiling probe
    out_cmd: List[str] = []
    skip_next = False
    for i, tok in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if tok in ("--lmhead-cs", "--cascade-tier1-tau", "--cascade-max-unverified", "--spec-assistant"):
            skip_next = True
            continue
        out_cmd.append(tok)
    q2_p = args.assistant_body.parent / "gemma4_31b_v10_assistant_body_q2.t16c"
    ast_p = q2_p if q2_p.is_file() else args.assistant_body
    out_cmd.extend([
        "--spec-assistant", str(ast_p),
        "--lmhead-cs", "sieve:0.02",
        "--cascade-tier1-tau", "0.01",
        "--cascade-max-unverified", "64",
        "--prompt-text", SPEED_BENCH_PROMPTS[0][1],
        "--max-tokens", "32",
    ])
    proc = subprocess.run(out_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    (out_dir / "tier1_fast_probe_v10.log").write_text(proc.stdout, encoding="utf-8")
    parsed = parse_v10_telemetry(proc.stdout)
    mem_ok, mem_msg = check_memory_gate(parsed)
    parsed.update({
        "scope": "unverified_tier1_4layer_draft_ceiling_not_60layer_coherent_decode",
        "memory_gate_ok": mem_ok,
        "memory_detail": mem_msg,
    })
    print(f"[TIER1_CEILING_PROBE] tok/s={parsed.get('tok_per_s')} ms/tok={parsed.get('ms_per_tok')} {mem_msg}", flush=True)
    return parsed


def _self_test() -> int:
    """Mandatory 6-gate fail-closed CPU verification battery for V10 world-proof harness."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: V10 telemetry extractor parses CASCADE31B, ALU31B, and MEM31B
    sample = (
        "SPEC31B max_batch_k=4 ngram=6 min_count=1 seed_entries=120 assistant_mb=56.20 byte_head=false step_pos=true min_margin=0.00 margin_stopped=0 ast_lm_skipped=10 drafted=24 accepted=18 batches=3 fallback_k1=0 accept_rate=0.750 generated=22 s_per_tok=0.032 target_shared_head=true\n"
        "CASCADE31B tau=3.50 tier2_accepted=18 tier3_rescues=1 tok_per_s=31.25 ms_per_tok=32.0\n"
        'GEN31B text="Answer: [CALC: 5x+3=38; 2x-1]"\n'
        'ALU31B expr_result="13"\n'
        "MEM31B pinned_bytes=6480000000 embed_bytes=539492352 ssd_read_bytes=0 cs_skipped_blocks=0 lmhead_cs_skipped_blocks=900 prefill_s=12.0 decode_s=0.68 vm_hwm_kb=Some(6650000) vm_rss_kb=Some(6610000)\n"
    )
    p = parse_v10_telemetry(sample)
    g1_ok = (
        abs(float(p.get("tok_per_s", 0.0)) - 31.25) < 1e-3
        and int(p.get("tier2_accepted", 0)) == 18
        and int(p.get("tier3_rescues", -1)) == 1
        and p.get("alu_result") == "13"
        and bool(p.get("target_shared_head"))
    )
    gate("Gate 1: V10 Telemetry Extraction (CASCADE31B + ALU31B)", g1_ok, f"tok/s={p.get('tok_per_s')}")

    # Gate 2: Fail closed on missing required paths
    ns_bad = argparse.Namespace(
        run31b=Path("/nonexistent/run31b"),
        t16c=Path("/nonexistent/v10.t16c"),
        config=Path("/nonexistent/cfg.json"),
        tokenizer=Path("/nonexistent/tok.json"),
        router=Path("/nonexistent/r.crtr"),
        assistant_body=Path("/nonexistent/ast.t16c"),
    )
    raised = False
    try:
        validate_required_paths(ns_bad)
    except FileNotFoundError:
        raised = True
    gate("Gate 2: Fail-Closed FileNotFoundError on Missing V10 Assets", raised)

    # Gate 3: Memory Gate (<= 6.50 GiB & ssd == 0)
    ok_mem, _ = check_memory_gate({"vm_hwm_kb": 6_650_000, "ssd_read_bytes": 0})
    bad_mem, _ = check_memory_gate({"vm_hwm_kb": 7_100_000, "ssd_read_bytes": 0})
    bad_ssd, _ = check_memory_gate({"vm_hwm_kb": 6_400_000, "ssd_read_bytes": 4096})
    gate("Gate 3: Strict <= 6.50 GiB Memory & Zero-SSD Gate", ok_mem and (not bad_mem) and (not bad_ssd))

    # Gate 4: CLI command builder checks Target-Embed Sharing & Cascade flags
    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        for fn in ("run31b", "v10.t16c", "cfg.json", "tok.json", "r.crtr", "hr.crtr", "ast.t16c", "seed.cngm"):
            (td / fn).write_text("x", encoding="utf-8")
        ns = argparse.Namespace(
            run31b=td / "run31b", t16c=td / "v10.t16c", config=td / "cfg.json", tokenizer=td / "tok.json",
            c8d=None, router=td / "r.crtr", head_router=td / "hr.crtr", so=None, threads=4,
            andvcnt_bits=6, kv_quant="glyph-lod", lmhead_cs="exact", shared_mlp_blocks=4,
            attn_shared_heads=4, pin_layers=None, profile_components=True, tri_modal_gate=False,
            spec_tree=16, spec_tree_depth=10, spec_ngram=6, spec_seed=td / "seed.cngm",
            assistant_body=td / "ast.t16c", byte_head=None, use_byte_head=False,
            cascade_tau=3.5, cascade_mlp_top_p=0.143, cascade_head_top_p=0.25, cascade_mod_thresh=0.55,
        )
        cmd = build_v10_cmd(ns, mode="cascade")
        g4_ok = (
            "--cortical-cascade" in cmd
            and "--spec-assistant" in cmd
            and "--spec-seed" in cmd
            and "--spec-byte-head" not in cmd
            and "--general-alu" not in cmd
        )
        cmd_alu = build_v10_cmd(ns, mode="cascade_alu")
        gate("Gate 4: CLI Cascade + Target-Embed Sharing Flags", g4_ok and "--general-alu" in cmd_alu)

    # Gate 5: Disjoint held-out benchmark integrity
    dj = e61.assert_holdouts_disjoint()
    gate("Gate 5: Zero 4-Gram Contamination Check", dj["overlap_4grams"] == 0 and dj["battery_prompts"] == 15)

    # Gate 6: Pure Neural vs General ALU Disambiguation
    gate("Gate 6: Pure Neural GEN31B Kept Distinct From ALU31B", p.get("gen_text") != p.get("alu_result"))

    print(f"V10_WORLD_PROOF_SELFTEST: {6 - failures}/6", flush=True)
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="V10 Cortical Cascade Physical Raspberry Pi 4 Proof Suite")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=("nll-pareto", "battery", "vision", "speed", "all"), default="all")
    ap.add_argument("--run31b", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/target/release/run31b"))
    ap.add_argument("--t16c", type=Path, default=DEFAULT_V10_DIR / "gemma4_31b_v10_cascade.t16c")
    ap.add_argument("--config", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--c8d", type=Path, default=DEFAULT_V10_DIR / "gemma4_31b_v10_c8d.c8d")
    ap.add_argument("--router", type=Path, default=DEFAULT_V10_DIR / "gemma4_31b_v10_router.crtr")
    ap.add_argument("--head-router", type=Path, default=DEFAULT_V10_DIR / "gemma4_31b_v10_head_router.crtr")
    ap.add_argument("--so", type=Path, default=Path("/home/astronautshe/gemma4_verify/experiments/run270_rust/libneon_stream_accelerator.so"))
    ap.add_argument("--teacher-logits", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v6_1/teacher_lgt1"))
    ap.add_argument("--assistant-body", type=Path, default=DEFAULT_V10_DIR / "gemma4_31b_v10_assistant_body_q4.t16c")
    ap.add_argument("--byte-head", type=Path, default=DEFAULT_V10_DIR / "gemma4_31b_v10_assistant.bhdr")
    ap.add_argument("--use-byte-head", action="store_true")
    ap.add_argument("--spec-seed", type=Path, default=DEFAULT_V10_DIR / "gemma4_v10_seed.cngm")
    ap.add_argument("--image", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v9_1/frame_01_coco.png"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_V10_DIR / "eval_out")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--andvcnt-bits", type=int, default=6)
    ap.add_argument("--kv-quant", type=str, default="glyph-lod")
    ap.add_argument("--lmhead-cs", type=str, default="exact")
    ap.add_argument("--shared-mlp-blocks", type=int, default=4)
    ap.add_argument("--attn-shared-heads", type=int, default=4)
    ap.add_argument("--pin-layers", type=int, default=None)
    ap.add_argument("--profile-components", action="store_true", default=True)
    ap.add_argument("--tri-modal-gate", action="store_true")
    ap.add_argument("--spec-tree", type=int, default=2)
    ap.add_argument("--spec-tree-depth", type=int, default=1)
    ap.add_argument("--spec-ngram", type=int, default=6)
    ap.add_argument("--cascade-tau", type=float, default=0.20)
    ap.add_argument("--cascade-mlp-top-p", type=float, default=0.65)
    ap.add_argument("--cascade-head-top-p", type=float, default=0.75)
    ap.add_argument("--cascade-mod-thresh", type=float, default=0.0)
    ap.add_argument("--cascade-tier1-tau", type=float, default=0.0)
    ap.add_argument("--cascade-max-unverified", type=int, default=2)
    ap.add_argument("--cascade-kv-mid", type=int, default=64)
    ap.add_argument("--update-report", type=Path, default=None)
    ap.add_argument("--pareto-kv-modes", type=str, default="glyph-lod,fp8")
    ap.add_argument("--pareto-configs", type=str, default="dense:1.0:0.0:1.0,crtr75_hr85:0.75:0.0:0.85,crtr14_hr25:0.143:0.0:0.25")
    ap.add_argument("--battery-modes", type=str, default="dense,tree_lossless,cascade,cascade_alu")
    ap.add_argument("--speed-max-tokens", type=int, default=64)
    args = ap.parse_args(argv)

    if len(sys.argv) == 1 or args.selftest:
        return _self_test()

    validate_required_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.update_report is not None and args.update_report.is_file():
        report: Dict[str, object] = json.loads(args.update_report.read_text(encoding="utf-8"))
        report["updated_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    else:
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": args.mode,
            "host_mem_available_kb": e93.read_mem_available_kb(),
            "t16c": str(args.t16c),
            "assistant_body": str(args.assistant_body),
            "target_shared_head": not args.use_byte_head,
        }
    if args.mode in ("nll-pareto", "all"):
        report["nll_pareto"] = e93.run_nll_pareto(args, args.out_dir)
    if args.mode in ("battery", "all"):
        bat_new = run_battery_v10(args, args.out_dir)
        if isinstance(report.get("battery"), dict) and args.update_report is not None:
            merged_bat = dict(report["battery"])  # type: ignore[arg-type]
            merged_bat.update(bat_new)
            report["battery"] = merged_bat
        else:
            report["battery"] = bat_new
    if args.mode in ("speed", "all"):
        report["speed_bench"] = run_speed_bench(args, args.out_dir)
        report["tier1_fast_probe"] = run_tier1_fast_probe(args, args.out_dir)
    if args.mode in ("vision", "all"):
        report["vision"] = e93.run_vision(args, args.out_dir)

    rep_path = args.update_report if args.update_report is not None else (args.out_dir / f"v10_{args.mode.replace('-', '_')}_report.json")
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[V10 REPORT WRITTEN] {rep_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Forensic doc-vs-JSON/log receipt parity verifier for Language-U-Gemma4.

Audits checked-in benchmark telemetry (JSON reports and hardware logs) for
structural integrity, internal mathematical consistency, local Markdown link
validity, and strict numerical parity with claims in Markdown documentation
(`README.md`, `trinity_retrieval/README.md`, `experiments/run270_rust/RECEIPTS.md`,
and `experiments/run270_rust/FRONTIER_RECEIPTS.md`).

Fails closed on any missing evidence file, NaN/Inf telemetry field, broken local
Markdown link, or doc-vs-evidence mismatch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import zipfile


class ReceiptParityError(ValueError):
    """Raised when checked-in receipts fail structural or doc-parity checks."""


@dataclass
class ParityReport:
    """Structured outcome of a repository-wide receipt parity audit."""

    root: str
    passed: bool
    checked_jsons: int = 0
    checked_links: int = 0
    checked_claims: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_NUM_RE = re.compile(r"`([0-9][0-9,.\-+eE]*)`")


def _load_json_strict(path: Path) -> Any:
    """Load JSON and fail closed on missing files, syntax errors, or NaN/Inf."""
    if not path.is_file():
        raise ReceiptParityError(f"Missing required JSON receipt: {path}")

    def _bad_const(val: str) -> Any:
        raise ReceiptParityError(f"Non-finite JSON constant {val!r} in {path}")

    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw, parse_constant=_bad_const)
    except ReceiptParityError:
        raise
    except Exception as exc:
        raise ReceiptParityError(f"Invalid JSON in {path}: {exc}") from exc


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ReceiptParityError(f"Missing required document/log: {path}")
    return path.read_text(encoding="utf-8", errors="strict")


def _assert_close(label: str, observed: float, expected: float, tol: float, errors: List[str]) -> None:
    if not math.isfinite(observed) or abs(observed - expected) > tol:
        errors.append(f"{label}: observed={observed!r} != expected={expected!r} (tol={tol})")


def _assert_contains(doc_name: str, text: str, snippet: str, errors: List[str]) -> None:
    if snippet not in text:
        errors.append(f"{doc_name}: missing expected grounded snippet {snippet!r}")


def _check_markdown_links(doc_path: Path, repo_root: Path, errors: List[str]) -> int:
    text = _read_text(doc_path)
    checked = 0
    for target in _LINK_RE.findall(text):
        clean = target.split("#", 1)[0].strip()
        if not clean or clean.startswith(("http://", "https://", "mailto:", "sidecar://")):
            continue
        checked += 1
        resolved = (doc_path.parent / clean).resolve()
        if not resolved.exists():
            errors.append(f"{doc_path.relative_to(repo_root).as_posix()}: broken link {target!r}")
    return checked


def _parse_num_token(cell: str) -> float:
    cleaned = cell.strip().strip("`").replace(",", "")
    return float(cleaned)


def _extract_markdown_table(text: str, heading: str) -> List[List[str]]:
    lines = text.splitlines()
    in_section = False
    rows: List[List[str]] = []
    for line in lines:
        if line.strip().startswith("#"):
            if in_section:
                break
            if heading in line:
                in_section = True
            continue
        if in_section and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
    if len(rows) < 2:
        raise ReceiptParityError(f"Markdown table under heading {heading!r} not found or empty")
    return rows[1:]


def _verify_all_evidence_jsons(root: Path, errors: List[str]) -> int:
    json_paths: List[Path] = [
        root / "trinity_retrieval" / "swe_navigation_benchmark_results.json",
        root / "docs" / "glyph_attention_v2_pi4_benchmark_report.json",
        root / "docs" / "optical_context_gemma4_benchmark_report.json",
        root / "docs" / "hf_31b_artifact_evidence.json",
        root / "docs" / "pi4_preparation_evidence.json",
        root / "docs" / "full_text_recorded_results.json",
        root / "docs" / "full_text_software_validation.json",
        root / "experiments" / "joint_glyphs" / "evidence" / "windows_joint_fast_benchmark.json",
        root / "experiments" / "run270_rust" / "ref_gemma3_native_arith.json",
        root / "experiments" / "run270_rust" / "calib" / "31b" / "receipts" / "v10_report.json",
        root / "experiments" / "run270_rust" / "calib" / "31b" / "receipts" / "v10_pi4_world_proof.json",
    ]
    ev_dir = root / "experiments" / "run270_rust" / "evidence"
    if not ev_dir.is_dir():
        errors.append(f"Missing directory: {ev_dir}")
        return len(json_paths)
    json_paths.extend(sorted(ev_dir.rglob("*.json")))

    seen = set()
    count = 0
    for jp in json_paths:
        resolved = jp.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_json_strict(jp)
        count += 1
    return count


def _verify_trinity_receipts(root: Path, readme: str, trinity_readme: str, errors: List[str]) -> int:
    swe = _load_json_strict(root / "trinity_retrieval" / "swe_navigation_benchmark_results.json")
    dense_b = int(swe["dense_fp32_bytes"])
    s7_b = int(swe["s7_bytes"])
    ratio = float(swe["compression_ratio"])
    docs_n = int(swe["corpus_documents"])
    _assert_close("swe_navigation ratio", dense_b / s7_b, ratio, 1e-6, errors)
    bm25_ms = float(swe["baseline1_bm25"]["latency_ms"])
    dense_ms = float(swe["baseline2_dense768"]["latency_ms"])
    ours = swe["ours_trinity_s7"]
    ours_ms = float(ours["latency_ms"])
    kib = s7_b / 1024.0

    for doc_name, txt in (("README.md", readme), ("trinity_retrieval/README.md", trinity_readme)):
        _assert_contains(doc_name, txt, f"{dense_b:,} B -> {s7_b:,} B", errors)
        _assert_contains(doc_name, txt, f"{ratio:.1f}x", errors)
    _assert_contains("README.md", readme, f"{docs_n:,}", errors)
    _assert_contains("README.md", readme, f"{kib:.2f} KiB", errors)
    _assert_contains("README.md", readme, f"{ours_ms:.4f} ms", errors)
    _assert_contains("README.md", readme, f"{bm25_ms:.2f} ms", errors)
    _assert_contains("README.md", readme, f"{dense_ms:.2f} ms", errors)
    _assert_contains("README.md", readme, str(ours["in_domain_accuracy"]), errors)
    _assert_contains("README.md", readme, str(ours["ood_false_positive_rate"]), errors)
    _assert_contains("README.md", readme, str(ours["token_savings"]), errors)
    return 10


def _verify_attention_and_optical(root: Path, readme: str, errors: List[str]) -> int:
    ga = _load_json_strict(root / "docs" / "glyph_attention_v2_pi4_benchmark_report.json")
    g128 = ga["layer5_global_context_scaling"]["131072"]
    opt5_48 = g128["glyph_modes"]["Opt5_DualRoad_48B"]
    _assert_close(
        "GlyphAttention 128K shrink ratio",
        float(g128["vllm_paged_bf16_kv_mb"]) / float(opt5_48["resident_mb"]),
        float(opt5_48["compression_vs_vllm_bf16"]),
        0.01,
        errors,
    )
    _assert_close(
        "GlyphAttention 128K speedup ratio",
        float(g128["vllm_paged_bf16_kv_ms"]) / float(opt5_48["latency_ms"]),
        float(opt5_48["speedup_vs_vllm_bf16"]),
        0.01,
        errors,
    )
    w1024 = ga["layer0_sliding_w1024"]["modes"]["Opt5_Ring_DualRoad"]
    _assert_contains("README.md", readme, f"{float(opt5_48['compression_vs_vllm_bf16']):.2f}x", errors)
    _assert_contains("README.md", readme, f"{float(opt5_48['resident_mb']):.2f} MB", errors)
    _assert_contains("README.md", readme, f"{float(opt5_48['speedup_vs_vllm_bf16']):.2f}x", errors)
    _assert_contains("README.md", readme, f"{float(opt5_48['latency_ms']):.2f} ms", errors)
    _assert_contains("README.md", readme, f"{float(g128['vllm_paged_bf16_kv_ms']):,.2f} ms", errors)
    _assert_contains("README.md", readme, f"{float(w1024['latency_ms']):.2f} ms", errors)
    _assert_contains("README.md", readme, f"{float(w1024['speedup_vs_paged_bf16']):.2f}x", errors)
    _assert_contains("README.md", readme, f"{float(w1024['o_proj_cosine']):.5f}", errors)

    opt = _load_json_strict(root / "docs" / "optical_context_gemma4_benchmark_report.json")
    sb = opt["streaming_benchmark"]
    dmas = [float(r["in_memory_dma_ms"]) for r in opt["scaling_results"]]
    _assert_contains("README.md", readme, f"{float(sb['fps']):,.1f} FPS", errors)
    _assert_contains("README.md", readme, f"{float(sb['token_throughput_per_sec']):,.1f} tok/s", errors)
    _assert_contains("README.md", readme, f"{min(dmas):.3f}-{max(dmas):.3f} ms", errors)
    _assert_contains("README.md", readme, f"{float(sb['bit_parity_pct']):.1f}%", errors)
    return 12


def _verify_windows_and_270m_receipts(
    root: Path, readme: str, receipts_md: str, frontier_md: str, errors: List[str]
) -> int:
    claims = 0
    wj = _load_json_strict(root / "experiments" / "joint_glyphs" / "evidence" / "windows_joint_fast_benchmark.json")
    src_sl = wj["source_slice"]
    w_tp = wj["windows_native_throughput"]["opt5_36b_avx2_gemv"]
    _assert_contains("README.md", readme, f"{int(src_sl['total_tokens_across_2688_glyphs']):,}", errors)
    _assert_contains("README.md", readme, f"{float(src_sl['mean_tokens_per_glyph']):.1f}", errors)
    _assert_contains("README.md", readme, f"{float(w_tp['ms_per_pass']):.4f} ms", errors)
    _assert_contains("README.md", readme, f"{float(w_tp['glyphs_per_sec']) / 1e6:.2f}M glyphs/s", errors)
    _assert_contains("README.md", readme, f"{float(w_tp['m_equiv_tokens_per_sec_at_20_4']):.1f}M tok/s", errors)
    claims += 5

    cal_dir = root / "experiments" / "run270_rust" / "evidence" / "g270_calib"
    avx = _load_json_strict(cal_dir / "v6_e2e_s2_avx2_bench5.json")
    rust = _load_json_strict(cal_dir / "v6_e2e_s2_rust_bench5.json")
    exact_b5 = _load_json_strict(cal_dir / "exact" / "bench5_summary.json")
    frz = _load_json_strict(cal_dir / "v6_e2e_s2_avx2_frozen.json")
    fsh = _load_json_strict(cal_dir / "v6_e2e_s2_avx2_fresh.json")

    speedup_vs_rust = float(rust["avg_s_per_fwd"]) / float(avx["avg_s_per_fwd"])
    speedup_vs_exact = float(exact_b5["avg_s_per_fwd"]) / float(avx["avg_s_per_fwd"])
    _assert_contains("README.md", readme, f"{int(avx['proj_bytes']) / 1e6:.2f} MB", errors)
    _assert_contains("README.md", readme, f"{float(avx['proj_bpw']):.2f} bpw", errors)
    _assert_contains("README.md", readme, f"{float(avx['avg_s_per_fwd']):.4f} s/fwd", errors)
    _assert_contains("README.md", readme, f"{float(rust['avg_s_per_fwd']):.4f} s/fwd", errors)
    _assert_contains("README.md", readme, f"{speedup_vs_rust:.2f}x", errors)

    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{int(avx['proj_bytes']):,}", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{float(avx['proj_bpw']):.6f} bpw", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{int(avx['total_weight_bytes']):,}", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{float(avx['avg_s_per_fwd']):.6f} s/fwd", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{float(rust['avg_s_per_fwd']):.6f} s/fwd", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{speedup_vs_rust:.2f}x", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{speedup_vs_exact:.2f}x", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{frz['correct']}/{frz['total']}", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{fsh['correct']}/{fsh['total']}", errors)
    claims += 14

    # RECEIPTS.md 270M exact + glyph log checks
    ex_log = _read_text(root / "experiments" / "run270_rust" / "run270_exact_gen.log")
    gl_log = _read_text(root / "experiments" / "run270_rust" / "run270_glyph_gen.log")
    ref_j = _load_json_strict(root / "experiments" / "run270_rust" / "ref_gemma3_native_arith.json")
    _assert_contains("run270_exact_gen.log", ex_log, 'GEN text="2 + 2 = 4\\n<end_of_turn>"', errors)
    _assert_contains("run270_exact_gen.log", ex_log, "projections=126", errors)
    _assert_contains("run270_glyph_gen.log", gl_log, "stream_bytes=51701776", errors)
    _assert_contains("run270_glyph_gen.log", gl_log, "projections=126", errors)
    _assert_contains("RECEIPTS.md", receipts_md, "`stream_bytes=51701776`", errors)
    _assert_contains("RECEIPTS.md", receipts_md, "`projections=126`", errors)
    _assert_contains("RECEIPTS.md", receipts_md, "`overall_decode_cos=0.971763`", errors)
    _assert_contains("RECEIPTS.md", receipts_md, "**126 / 126**", errors)
    if ref_j.get("generated_ids")[:8] != [236778, 900, 236743, 236778, 578, 236743, 236812, 107]:
        errors.append(f"ref_gemma3_native_arith.json generated_ids mismatch: {ref_j.get('generated_ids')}")
    claims += 9

    # Parse all 23 rows of sweep_bench5_all23.json vs FRONTIER_RECEIPTS.md table
    sweep = _load_json_strict(
        root / "experiments" / "run270_rust" / "evidence" / "g270_sweep" / "sweep_bench5_all23.json"
    )
    by_tag: Dict[str, Dict[str, Any]] = {row["tag"]: row for row in sweep}
    table_rows = _extract_markdown_table(frontier_md, "5-Prompt Parity & Compression Pareto")
    if len(table_rows) != len(sweep):
        errors.append(f"FRONTIER_RECEIPTS.md 270M sweep table has {len(table_rows)} rows, JSON has {len(sweep)}")
    for row in table_rows:
        tag = row[0].strip().strip("`")
        if tag not in by_tag:
            errors.append(f"FRONTIER_RECEIPTS.md unknown sweep tag {tag!r}")
            continue
        rec = by_tag[tag]
        _assert_close(f"sweep[{tag}].proj_bpw", _parse_num_token(row[1]), float(rec["proj_bpw"]), 1e-4, errors)
        _assert_close(
            f"sweep[{tag}].total_weight_mb",
            _parse_num_token(row[2]),
            float(rec["total_weight_bytes"]) / 1e6,
            0.015,
            errors,
        )
        _assert_close(f"sweep[{tag}].vm_hwm_kb", _parse_num_token(row[3]), float(rec["vm_hwm_kb"]), 0.5, errors)
        _assert_close(f"sweep[{tag}].s_per_fwd", _parse_num_token(row[4]), float(rec["s_per_fwd"]), 1e-4, errors)
        _assert_close(f"sweep[{tag}].min_cos", _parse_num_token(row[5]), float(rec["min_cos"]), 1e-6, errors)
        _assert_close(f"sweep[{tag}].max_rel_l2", _parse_num_token(row[6]), float(rec["max_rel_l2"]), 1e-6, errors)
        if row[7].strip().strip("`") != str(rec["argmax_match"]):
            errors.append(f"sweep[{tag}].argmax_match {row[7]} != {rec['argmax_match']}")
        if row[8].strip().strip("`") != str(rec["gate_pass"]):
            errors.append(f"sweep[{tag}].gate_pass {row[8]} != {rec['gate_pass']}")
        claims += 8

    # Parse 28Q / 30Q suites table in FRONTIER_RECEIPTS.md
    suites_dir = root / "experiments" / "run270_rust" / "evidence" / "g270_suites"
    suite_map: Dict[str, Tuple[Path, Path]] = {
        "exact": (cal_dir / "exact_frozen.json", cal_dir / "exact_fresh.json"),
        "b1+h8": (suites_dir / "b1+h8_frozen.json", suites_dir / "b1+h8_fresh.json"),
        "q3": (suites_dir / "q3_frozen.json", suites_dir / "q3_fresh.json"),
        "q4": (suites_dir / "q4_frozen.json", suites_dir / "q4_fresh.json"),
        "q4+h8": (suites_dir / "q4+h8_frozen.json", suites_dir / "q4+h8_fresh.json"),
        "q4+h8_eq4": (suites_dir / "q4+h8_eq4_frozen.json", suites_dir / "q4+h8_eq4_fresh.json"),
        "q5+h8": (suites_dir / "q5+h8_frozen.json", suites_dir / "q5+h8_fresh.json"),
        "q6": (suites_dir / "q6_frozen.json", suites_dir / "q6_fresh.json"),
        "q6+h8": (suites_dir / "q6+h8_frozen.json", suites_dir / "q6+h8_fresh.json"),
        "q8_eq8": (suites_dir / "q8_eq8_frozen.json", suites_dir / "q8_eq8_fresh.json"),
        "q10+h8": (suites_dir / "q10+h8_frozen.json", suites_dir / "q10+h8_fresh.json"),
    }
    suite_rows = _extract_markdown_table(frontier_md, "28Q Frozen & 30Q Fresh Reasoning Suites")
    for row in suite_rows:
        cfg_key = row[0].split("(", 1)[0].strip().strip("`")
        if cfg_key not in suite_map:
            errors.append(f"FRONTIER_RECEIPTS.md unknown suite row config {cfg_key!r}")
            continue
        fz_j = _load_json_strict(suite_map[cfg_key][0])
        fs_j = _load_json_strict(suite_map[cfg_key][1])
        _assert_close(f"suite[{cfg_key}].total_weight_bytes", _parse_num_token(row[1]), float(fz_j["total_weight_bytes"]), 0.5, errors)
        _assert_close(f"suite[{cfg_key}].frozen_vm_hwm_kb", _parse_num_token(row[2]), float(fz_j["vm_hwm_kb"]), 0.5, errors)
        _assert_contains(f"suite[{cfg_key}].frozen", row[3], f"{fz_j['correct']}/{fz_j['total']}", errors)
        _assert_contains(f"suite[{cfg_key}].fresh", row[4], f"{fs_j['correct']}/{fs_j['total']}", errors)
        _assert_close(
            f"suite[{cfg_key}].frozen_s_per_fwd",
            _parse_num_token(row[5]),
            float(fz_j["total_seconds"]) / float(fz_j["total_forwards"]),
            0.0002,
            errors,
        )
        _assert_close(
            f"suite[{cfg_key}].fresh_s_per_fwd",
            _parse_num_token(row[6]),
            float(fs_j["total_seconds"]) / float(fs_j["total_forwards"]),
            0.0002,
            errors,
        )
        claims += 6
    return claims


def _verify_run31b_receipts(root: Path, readme: str, frontier_md: str, errors: List[str]) -> int:
    claims = 0
    r31 = root / "experiments" / "run270_rust" / "evidence" / "run31b"

    # QAT coherent pin10 + 2+2 logit parity
    par = _load_json_strict(r31 / "run31b_parity_2plus2.json")
    h_log = _read_text(r31 / "a_hamlet_q8emb_pin10_neon.log")
    _assert_contains("a_hamlet_q8emb_pin10_neon.log", h_log, 'GEN31B text="William Shakespeare wrote the play *Hamlet*.<turn|>"', errors)
    _assert_contains("a_hamlet_q8emb_pin10_neon.log", h_log, "vm_hwm_kb=Some(4725968)", errors)
    _assert_contains("README.md", readme, "4,725,968 kB", errors)
    _assert_contains("README.md", readme, f"{float(par['cosine']):.6f}", errors)
    _assert_contains("README.md", readme, f"{float(par['rel_l2']):.6f}", errors)
    if not bool(par.get("top1_match")) or not bool(par.get("top5_exact_match")):
        errors.append(f"run31b_parity_2plus2.json top1/top5 mismatch: {par}")
    claims += 6

    # Streamed + pin10 Frontier Section 3 logs
    st_log = _read_text(r31 / "test_2plus2_stream1.txt")
    hp_log = _read_text(r31 / "hamlet_q8emb_pin10.txt")
    _assert_contains("test_2plus2_stream1.txt", st_log, "vm_hwm_kb=Some(344600)", errors)
    _assert_contains("hamlet_q8emb_pin10.txt", hp_log, "vm_hwm_kb=Some(4436992)", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "vm_hwm_kb = 344,600", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "vm_hwm_kb = 4,436,992", errors)
    claims += 4

    # T16C v1/v2 Pi 4 calibration + FFI parity receipts
    v2_2p2 = _read_text(r31 / "run31b_cal_v2_2plus2.log")
    v2_2p2_proc = _read_text(r31 / "run31b_cal_v2_2plus2_proc.log")
    v2_ham = _read_text(r31 / "run31b_cal_v2_hamlet.log")
    v2_ham_proc = _read_text(r31 / "run31b_cal_v2_hamlet_proc.log")
    v2_noffi = _read_text(r31 / "run31b_cal_v2_2plus2_noffi.log")
    v2_par = _read_text(r31 / "run31b_cal_v2_ffi_parity.txt")
    _assert_contains("run31b_cal_v2_2plus2.log", v2_2p2, "pinned_layers=60/60", errors)
    _assert_contains("run31b_cal_v2_2plus2.log", v2_2p2, "ssd_read_bytes=0", errors)
    _assert_contains("run31b_cal_v2_2plus2.log", v2_2p2, "s_per_tok=1.291", errors)
    _assert_contains("run31b_cal_v2_2plus2_proc.log", v2_2p2_proc, "VmSwap:\t       0 kB", errors)
    _assert_contains("run31b_cal_v2_hamlet_proc.log", v2_ham_proc, "VmSwap:\t       0 kB", errors)
    _assert_contains("run31b_cal_v2_hamlet.log", v2_ham, "vm_hwm_kb=Some(6238804)", errors)
    _assert_contains("run31b_cal_v2_2plus2_noffi.log", v2_noffi, "prefill done in 889.23s", errors)
    _assert_contains("run31b_cal_v2_ffi_parity.txt", v2_par, "cos=0.99929892", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "e63a583bc6f8ed28d2429673ddcc6ce3f8e2d54cd7802100d768f16633d7b345", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "s_per_tok=1.291", errors)
    _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "cos=0.99929892", errors)
    claims += 11

    # Optional V3 Pi 4 receipt validation when checked in
    v3_json = r31 / "run31b_v3_verified.json"
    if v3_json.is_file():
        v3 = _load_json_strict(v3_json)
        if v3.get("pinned_layers") != 60 or int(v3.get("vm_swap_kb", -1)) != 0 or int(v3.get("ssd_read_bytes", -1)) != 0:
            errors.append(f"run31b_v3_verified.json violates RAM-residency invariant: {v3}")
        if float(v3.get("ffi_parity_cosine", 0.0)) < 0.999:
            errors.append(f"run31b_v3_verified.json FFI parity cosine too low: {v3.get('ffi_parity_cosine')}")
        if not bool(v3.get("ffi_parity_top1_exact")) or not bool(v3.get("ffi_parity_gen_exact")):
            errors.append(f"run31b_v3_verified.json FFI parity exact flags failed: {v3.get('ffi_parity_top1_exact')}")
        p_k1 = v3.get("prompts_k1", [])
        if int(v3.get("prompts_tested", 0)) != 8 or len(p_k1) != 8:
            errors.append(f"run31b_v3_verified.json expected 8 prompts_k1, got {len(p_k1)}")
        by_id = {p.get("id"): p for p in p_k1}
        spec_runs = v3.get("speculative_k4_runs", [])
        if not bool(v3.get("speculative_k4_parity_verified")) or len(spec_runs) != 7:
            errors.append(f"run31b_v3_verified.json speculative_k4_runs invalid count={len(spec_runs)}")
        for sr in spec_runs:
            ref = by_id.get(sr.get("reference_prompt", ""), {})
            s_ids = sr.get("ids", [])
            if not bool(sr.get("bit_exact_vs_k1")) or s_ids != ref.get("ids", [])[: len(s_ids)]:
                errors.append(f"run31b_v3_verified.json speculative parity mismatch on {sr.get('run')}")
            if int(sr.get("vm_swap_kb", -1)) != 0 or int(sr.get("ssd_read_bytes", -1)) != 0:
                errors.append(f"run31b_v3_verified.json speculative RAM/IO violation on {sr.get('run')}")
        hq = v3.get("hybrid_qat_l59_probe", {})
        if not bool(hq.get("bit_exact_vs_k1")) or int(hq.get("vm_swap_kb", -1)) != 0 or int(hq.get("ssd_read_bytes", -1)) != 0:
            errors.append(f"run31b_v3_verified.json hybrid_qat_l59_probe invariant failed: {hq}")
        cs_rows = v3.get("cs_lmhead_benchmarks", [])
        if int(v3.get("cs_lmhead_top1_exact_matches", 0)) != 8 or len(cs_rows) != 8:
            errors.append(f"run31b_v3_verified.json cs_lmhead_benchmarks count={len(cs_rows)} exact={v3.get('cs_lmhead_top1_exact_matches')}")
        for cr in cs_rows:
            ssd_max = 0 if "Assistant" in str(cr.get("model", "")) else 5_000_000
            ssd_val = int(cr.get("timed_ssd_read_bytes", -1))
            if int(cr.get("exact_top1_matches", 0)) != 8 or int(cr.get("vm_swap_kb", -1)) != 0 or ssd_val < 0 or ssd_val > ssd_max:
                errors.append(f"run31b_v3_verified.json CS benchmark invariant failed: {cr}")
        _assert_contains("README.md", readme, f"{float(v3['ffi_parity_cosine']):.10f}", errors)
        _assert_contains("README.md", readme, f"{float(v3['ffi_prefill_speedup_x']):.2f}x", errors)
        _assert_contains("README.md", readme, f"{float(v3['prompt_battery_k1_summary']['weighted_ilt_s_per_tok']):.3f} s/tok", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{float(v3['ffi_parity_cosine']):.10f}", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, f"{float(v3['prompt_battery_k1_summary']['weighted_ilt_s_per_tok']):.3f} s/tok", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "shrink=3.94x", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "4.04x", errors)
        if (r31 / "run31b_v3_ffi_parity.txt").is_file() and (r31 / "run31b_v3_speculative_k4.log").is_file() and (r31 / "run31b_v3_cs_lmhead_bench.log").is_file():
            v3_par = _read_text(r31 / "run31b_v3_ffi_parity.txt")
            v3_spec = _read_text(r31 / "run31b_v3_speculative_k4.log")
            v3_cs = _read_text(r31 / "run31b_v3_cs_lmhead_bench.log")
            _assert_contains("run31b_v3_ffi_parity.txt", v3_par, f"cosine_similarity={float(v3['ffi_parity_cosine']):.10f}", errors)
            _assert_contains("run31b_v3_ffi_parity.txt", v3_par, "gen31b_exact_match=True", errors)
            _assert_contains("run31b_v3_speculative_k4.log", v3_spec, "s_per_tok=5.868", errors)
            _assert_contains("run31b_v3_speculative_k4.log", v3_spec, "shrink=3.94x", errors)
            _assert_contains("run31b_v3_cs_lmhead_bench.log", v3_cs, "exact_top1=8/8", errors)
            _assert_contains("run31b_v3_cs_lmhead_bench.log", v3_cs, "speedup=4.04x", errors)
            _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "col_macs_saved=84.22%", errors)
            claims += 13
        claims += 12

    # V4.2 Colab 262K + Pi 4 Tri-Modal + Bidirectional Vision receipts
    v42_json = r31 / "colab_v4_2_stage2_report.json"
    if v42_json.is_file():
        v42 = _load_json_strict(v42_json)
        if int(v42.get("tokens_calibrated", 0)) != 262144:
            errors.append(f"colab_v4_2_stage2_report.json unexpected tokens_calibrated={v42.get('tokens_calibrated')}")
        if int(v42.get("unified_t16c_bytes", 0)) != 6347775616 or int(v42.get("c8db_bytes", 0)) != 11874560:
            errors.append(f"colab_v4_2_stage2_report.json artifact byte mismatch: {v42}")
        if float(v42.get("l59_state_cos_after_c8d", 0.0)) <= float(v42.get("l59_state_cos_before_c8d", 1.0)):
            errors.append("colab_v4_2_stage2_report.json L59 C8DB cosine did not improve")
        if float(v42.get("post_cell60_val_kl_after", 99.0)) >= float(v42.get("post_cell60_val_kl_before", 0.0)):
            errors.append("colab_v4_2_stage2_report.json post_cell60 KL did not improve")
        if float(v42.get("vision_val_cos_after", 0.0)) <= float(v42.get("vision_val_cos_before", 1.0)):
            errors.append("colab_v4_2_stage2_report.json vision cosine did not improve")
        _assert_contains("README.md", readme, "6,347,775,616 B", errors)
        _assert_contains("README.md", readme, "0.86094", errors)
        _assert_contains("README.md", readme, "0.81212", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "5b9ae4d6bde2a893d073649350e11e27cf9e5f208fdca19b2ed7fbbf46f1c9b1", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "3b8618a462d204ba82d54eb2fd4a3b2aef8cf8bc8d0ec26715de19b547b401bc", errors)
        if (r31 / "pi4_colab_v4_2_post_bidir_sanity.log").is_file() and (r31 / "pi4_colab_v4_2_vision_bidir.log").is_file():
            san_log = _read_text(r31 / "pi4_colab_v4_2_post_bidir_sanity.log")
            vis_log = _read_text(r31 / "pi4_colab_v4_2_vision_bidir.log")
            _assert_contains("pi4_colab_v4_2_post_bidir_sanity.log", san_log, "POST_EDIT_SANITY_OK", errors)
            _assert_contains("pi4_colab_v4_2_post_bidir_sanity.log", san_log, 'GEN31B text="The capital of France is ****Paris**.<turn|>"', errors)
            _assert_contains("pi4_colab_v4_2_post_bidir_sanity.log", san_log, 'GEN31B text="The result is 4.<turn|>"', errors)
            _assert_contains("pi4_colab_v4_2_post_bidir_sanity.log", san_log, "ssd_read_bytes=0", errors)
            _assert_contains("pi4_colab_v4_2_vision_bidir.log", vis_log, "939.50s", errors)
            claims += 5
        claims += 10

    # Frontiers A, B & C: 11-Prompt 262K Battery, Live GlyphKV 128K + Optical DMA + NEON-V2, and V4.3 VLM Multimodal QA
    v43_json = r31 / "colab_v4_3_vlm_stage2_report.json"
    if v43_json.is_file():
        v43 = _load_json_strict(v43_json)
        if int(v43.get("tokens_calibrated", 0)) != 270336 or int(v43.get("vlm_sequences_calibrated", 0)) != 160:
            errors.append(f"colab_v4_3_vlm_stage2_report.json unexpected token/vlm count: {v43}")
        if int(v43.get("unified_t16c_bytes", 0)) != 6353854848 or int(v43.get("c8db_bytes", 0)) != 9317184:
            errors.append(f"colab_v4_3_vlm_stage2_report.json artifact byte mismatch: {v43}")
        if float(v43.get("vision_val_cos_after", 0.0)) <= float(v43.get("vision_val_cos_before", 1.0)):
            errors.append("colab_v4_3_vlm_stage2_report.json vision cosine did not improve")
        if float(v43.get("heldout_vis_soft_cos", 0.0)) < 0.90:
            errors.append(f"colab_v4_3_vlm_stage2_report.json heldout_vis_soft_cos too low: {v43.get('heldout_vis_soft_cos')}")
        if float(v43.get("post_cell60_val_kl_after", 99.0)) >= float(v43.get("post_cell60_val_kl_before", 0.0)):
            errors.append("colab_v4_3_vlm_stage2_report.json post_cell60 KL did not improve")
        if float(v43.get("post_cell60_val_top1_after", 0.0)) <= float(v43.get("post_cell60_val_top1_before", 1.0)):
            errors.append("colab_v4_3_vlm_stage2_report.json post_cell60 top1 did not improve")
        ho_list = v43.get("heldout_coco_val0_3", [])
        if len(ho_list) != 4 or float(ho_list[0].get("ans_top1_agree", 0.0)) < 0.60:
            errors.append(f"colab_v4_3_vlm_stage2_report.json heldout_coco_val0_3 invalid: {ho_list}")

        _assert_contains("README.md", readme, "6,353,854,848 B", errors)
        _assert_contains("README.md", readme, "5.9175 GiB", errors)
        _assert_contains("README.md", readme, "0.91714", errors)
        _assert_contains("README.md", readme, "0.90586", errors)
        _assert_contains("README.md", readme, "A motorcycle is parked on a grassy field.<turn|>", errors)
        _assert_contains("README.md", readme, "1,088.02 MiB", errors)
        _assert_contains("README.md", readme, "605.34 MiB", errors)
        _assert_contains("README.md", readme, "555.00 MiB", errors)
        _assert_contains("README.md", readme, "85,041.2 FPS", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "5469f95285d7ed99e14c728504ac29f361f6a28781e829971d216b7486d99571", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "46afa072ecc4d4846c6b266d5d4aed0acae3c076c407433e3edfb95f2f590e09", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "A motorcycle is parked on a grassy field.<turn|>", errors)
        _assert_contains("FRONTIER_RECEIPTS.md", frontier_md, "120 km / 2 = **60**", errors)
        claims += 20

        # Strict SHA-256 receipt file & Pi 4 live Frontier A/B/C logs
        v43_sha = _read_text(r31 / "colab_v4_3_vlm_SHA256SUMS.txt")
        _assert_contains("colab_v4_3_vlm_SHA256SUMS.txt", v43_sha, "5469f95285d7ed99e14c728504ac29f361f6a28781e829971d216b7486d99571", errors)
        _assert_contains("colab_v4_3_vlm_SHA256SUMS.txt", v43_sha, "46afa072ecc4d4846c6b266d5d4aed0acae3c076c407433e3edfb95f2f590e09", errors)

        v43_vis = _read_text(r31 / "pi4_v4_3_vlm_vision_qa.log")
        _assert_contains("pi4_v4_3_vlm_vision_qa.log", v43_vis, 'GEN31B text="A motorcycle is parked on a grassy field.<turn|>"', errors)
        _assert_contains("pi4_v4_3_vlm_vision_qa.log", v43_vis, "vision_bytes=237253152", errors)
        _assert_contains("pi4_v4_3_vlm_vision_qa.log", v43_vis, "ssd_read_bytes=0", errors)
        _assert_contains("pi4_v4_3_vlm_vision_qa.log", v43_vis, "vm_hwm_kb=Some(7197296)", errors)

        v43_txt = _read_text(r31 / "pi4_v4_3_vlm_text_optical_glyph.log")
        _assert_contains("pi4_v4_3_vlm_text_optical_glyph.log", v43_txt, "fps=85041.2", errors)
        _assert_contains("pi4_v4_3_vlm_text_optical_glyph.log", v43_txt, "exact_parity=true", errors)
        _assert_contains("pi4_v4_3_vlm_text_optical_glyph.log", v43_txt, 'GEN31B text="The capital of France is **the city **Paris**.<turn|>"', errors)
        _assert_contains("pi4_v4_3_vlm_text_optical_glyph.log", v43_txt, "ssd_read_bytes=0", errors)

        b128 = _read_text(r31 / "pi4_run31b_128k_glyph_kv.log")
        _assert_contains("pi4_run31b_128k_glyph_kv.log", b128, "BENCH_128K_KV_COMPLETE_OK", errors)
        _assert_contains("pi4_run31b_128k_glyph_kv.log", b128, "total_resident_mb=1088.02", errors)
        _assert_contains("pi4_run31b_128k_glyph_kv.log", b128, "total_resident_mb=605.34", errors)
        _assert_contains("pi4_run31b_128k_glyph_kv.log", b128, "total_resident_mb=555.00", errors)

        gopt = _read_text(r31 / "pi4_colab_v4_2_glyph_kv_optical.log")
        _assert_contains("pi4_colab_v4_2_glyph_kv_optical.log", gopt, "GLYPH_OPTICAL_LIVE_COMPLETE_OK", errors)
        _assert_contains("pi4_colab_v4_2_glyph_kv_optical.log", gopt, "fps=81083.3", errors)
        _assert_contains("pi4_colab_v4_2_glyph_kv_optical.log", gopt, 'GEN31B text="The capital of France is **Paris**.<turn|>"', errors)
        _assert_contains("pi4_colab_v4_2_glyph_kv_optical.log", gopt, 'GEN31B text="The answer is 4.<turn|>"', errors)

        nspd = _read_text(r31 / "pi4_colab_v4_2_neon_speedup.log")
        _assert_contains("pi4_colab_v4_2_neon_speedup.log", nspd, "NEON_SPEEDUP_SUITE_COMPLETE_OK", errors)
        _assert_contains("pi4_colab_v4_2_neon_speedup.log", nspd, "s_per_tok=1.543", errors)

        bat = _read_text(r31 / "pi4_colab_v4_2_battery_p1_p8.log")
        _assert_contains("pi4_colab_v4_2_battery_p1_p8.log", bat, "BATTERY_262K_COMPLETE_OK", errors)
        _assert_contains("pi4_colab_v4_2_battery_p1_p8.log", bat, 'GEN31B text="The smaller number is **193**.<turn|>"', errors)

        exab = _read_text(r31 / "pi4_colab_v4_2_exact_lmhead_ablation.log")
        _assert_contains("pi4_colab_v4_2_exact_lmhead_ablation.log", exab, "EXACT_LMHEAD_ABLATION_COMPLETE_OK", errors)
        _assert_contains("pi4_colab_v4_2_exact_lmhead_ablation.log", exab, "120 km / 2 = **60**", errors)
        _assert_contains("pi4_colab_v4_2_exact_lmhead_ablation.log", exab, "glyph_tokens=2060", errors)
        claims += 25

    return claims


def _canonical_zip_digest(raw_zip: bytes) -> str:
    """Compute an OS/zlib-invariant SHA-256 over sorted zip members and raw payloads."""
    with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
        parts = []
        for name in sorted(zf.namelist()):
            data = zf.read(name)
            parts.append(f"{name}:{len(data)}:{hashlib.sha256(data).hexdigest()}\n".encode("utf-8"))
        return hashlib.sha256(b"".join(parts)).hexdigest()


def _verify_submission_bundles(root: Path, readme: str, errors: List[str]) -> int:
    expected_zips: List[Tuple[str, str, str, bool]] = [
        (
            "agent/submission.zip",
            "fb461a4e14a70fbefd2eb6522e2b3c1d0b2dabeebe23c4fa5af994df90125a3a",
            "707e5706a3e2ff5e7d9cb8ce271953614e15e6a533cdac11f5fbe1945ccccf1c",
            True,
        ),
        (
            "submissions/v2-measured/build/submission.zip",
            "76d661670d0ad1c5ef64c6b8a277d0a4b8fa61a39c242f90748b4126eb53087b",
            "e75f7d7e3e7937adc4e9e0a2ef463e4f8337c1bb467a0d6c79115922efde7a66",
            False,
        ),
        (
            "submissions/v3-converged/build/submission.zip",
            "04cdcdfca4d23481618834a5f3facf4a134d023ec896f7aa8a4a5c40d4700e5e",
            "8b344581edfe73aa264a18d551ada6437bb0044d5d27f330265af4f5af33da0f",
            False,
        ),
        (
            "submissions/v4-loop-guard/build/submission.zip",
            "7e89e9c072b57c4c25b0d52d0732614c3ab93baf83db529af13fb5a1a546b4ef",
            "5f1b9880dffbfbd3e2759867820ffc7b10a868ac70c2770478018147ab7ac512",
            False,
        ),
        (
            "submissions/v5-restored/build/submission.zip",
            "0b32f172f496d5ffe1e055e107de5bee05364175d15564a14c79db7b2a3171d2",
            "1e21642b879b959f5d1c1fcfe864fe55dfe04f7d49136c315c07607dbd249f46",
            False,
        ),
        (
            "submissions/v5-analyzer-coder/build/submission.zip",
            "60b98087c13b9b2bd370f31ca98f5c0cd02b7b557cb7c65036ee0da624386d0c",
            "ac506d13ded5412eeb3a3854b52eecfe04a291ed72494fcec9bdd72d53769bfa",
            False,
        ),
        (
            "submissions/v6-optimal/build/submission.zip",
            "0069ff8d3da3e58d6a280c73e78618d1d63c9271a116b9c20d765eef7f109f2d",
            "77781d0968faa6fba7f46e16c8a6374c611ad04088bdbd81d4ad6a8d683f4d1b",
            False,
        ),
        (
            "submissions/v8-runtime-safe/build/submission.zip",
            "6a45b24c296b3d8298e7abeac40373f30dbffc5196812d4c1bc2b3941f38010b",
            "04c1c83d7fceb4d0ee7d528397207eaa22940351347ec3fa269429bfe198124f",
            False,
        ),
    ]
    claims = 0
    for rel, head_sha, canon_sha, check_raw in expected_zips:
        p = root / rel
        if not p.is_file():
            errors.append(f"Missing declared competition bundle: {rel}")
            continue
        raw_bytes = p.read_bytes()
        if check_raw:
            got_raw = hashlib.sha256(raw_bytes).hexdigest()
            if got_raw != head_sha:
                errors.append(f"{rel}: raw SHA-256 {got_raw} != pinned {head_sha}")
        got_canon = _canonical_zip_digest(raw_bytes)
        if got_canon != canon_sha:
            errors.append(f"{rel}: canonical member SHA-256 {got_canon} != pinned {canon_sha}")
        _assert_contains("README.md", readme, head_sha, errors)
        _assert_contains("README.md", readme, canon_sha, errors)
        claims += 3

    fleet_prompt = root / "fleet-consensus-v1" / "prompts" / "system.md"
    exp_prompt_sha = "234969efed3a86a519d74be15beea0381f1b63e0d92c18a47cad82a15956bb74"
    if not fleet_prompt.is_file():
        errors.append("Missing fleet-consensus-v1/prompts/system.md")
    else:
        lf_bytes = fleet_prompt.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        got_p_sha = hashlib.sha256(lf_bytes).hexdigest()
        if got_p_sha != exp_prompt_sha:
            errors.append(f"fleet-consensus-v1/prompts/system.md LF SHA-256 {got_p_sha} != {exp_prompt_sha}")
        _assert_contains("README.md", readme, exp_prompt_sha, errors)
        claims += 2

    return claims


def _verify_v10_receipts(root: Path, readme: str, frontier_md: str, errors: List[str]) -> int:
    v10_gpu = _load_json_strict(root / "experiments" / "run270_rust" / "calib" / "31b" / "receipts" / "v10_report.json")
    v10_pi4 = _load_json_strict(root / "experiments" / "run270_rust" / "calib" / "31b" / "receipts" / "v10_pi4_world_proof.json")
    t16c_b = int(v10_gpu["t16c_bytes"])
    e2e = v10_gpu["e2e_stage2_5"]
    bhdr = v10_gpu["bhdr256"]
    pareto = {r["config"]: r for r in v10_pi4["nll_pareto"]}
    bat = v10_pi4["battery"]
    t1p = v10_pi4["tier1_fast_probe"]

    for doc_name, txt in (("README.md", readme), ("FRONTIER_RECEIPTS.md", frontier_md)):
        _assert_contains(doc_name, txt, f"{t16c_b:,}", errors)
        _assert_contains(doc_name, txt, f"{float(e2e['val_kl_after']):.4f}", errors)
        _assert_contains(doc_name, txt, str(bhdr["heldout_beam_top1_acc"]), errors)
        _assert_contains(doc_name, txt, f"{float(pareto['dense_fp8']['mean_nll']):.5f}", errors)
        _assert_contains(doc_name, txt, f"{float(pareto['dense_fp8']['kl_t_to_s']):.5f}", errors)
        _assert_contains(doc_name, txt, f"{float(pareto['dense_glyph-lod']['mean_nll']):.5f}", errors)
        _assert_contains(doc_name, txt, f"{int(bat['cascade_alu']['passed'])}/{int(bat['cascade_alu']['total'])}", errors)
        _assert_contains(doc_name, txt, f"{float(t1p['tok_per_s'])} tok/s", errors)
    return 16


def verify_repo_receipts(repo_root: Path) -> ParityReport:
    """Audit all JSON receipts, local Markdown links, and doc numerical claims."""
    root = repo_root.resolve()
    errors: List[str] = []

    checked_jsons = _verify_all_evidence_jsons(root, errors)

    readme_p = root / "README.md"
    trinity_p = root / "trinity_retrieval" / "README.md"
    receipts_p = root / "experiments" / "run270_rust" / "RECEIPTS.md"
    frontier_p = root / "experiments" / "run270_rust" / "FRONTIER_RECEIPTS.md"

    readme = _read_text(readme_p)
    trinity_readme = _read_text(trinity_p)
    receipts_md = _read_text(receipts_p)
    frontier_md = _read_text(frontier_p)

    checked_links = 0
    for dp in (readme_p, trinity_p, receipts_p, frontier_p):
        checked_links += _check_markdown_links(dp, root, errors)

    checked_claims = 0
    checked_claims += _verify_trinity_receipts(root, readme, trinity_readme, errors)
    checked_claims += _verify_attention_and_optical(root, readme, errors)
    checked_claims += _verify_windows_and_270m_receipts(root, readme, receipts_md, frontier_md, errors)
    checked_claims += _verify_run31b_receipts(root, readme, frontier_md, errors)
    checked_claims += _verify_v10_receipts(root, readme, frontier_md, errors)
    checked_claims += _verify_submission_bundles(root, readme, errors)

    if errors:
        return ParityReport(
            root=str(root),
            passed=False,
            checked_jsons=checked_jsons,
            checked_links=checked_links,
            checked_claims=checked_claims,
            errors=errors,
        )

    return ParityReport(
        root=str(root),
        passed=True,
        checked_jsons=checked_jsons,
        checked_links=checked_links,
        checked_claims=checked_claims,
    )


def _self_test(repo_root: Path) -> int:
    """Execute mandatory 6-gate fail-closed self-test on real and tampered trees."""
    failures = 0

    def _gate(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        tag = "PASS" if ok else "FAIL"
        print(f"[Gate {tag}] {name}: {detail}")
        if not ok:
            failures += 1

    # Gate 1: Full repository JSON + link + claim audit passes cleanly
    rep = verify_repo_receipts(repo_root)
    _gate(
        "1_repo_receipt_parity",
        rep.passed,
        f"jsons={rep.checked_jsons} links={rep.checked_links} claims={rep.checked_claims} errors={rep.errors[:3]}",
    )

    with tempfile.TemporaryDirectory() as tmp:
        copy_root = Path(tmp) / "repo"
        for sub in ("README.md", "trinity_retrieval/README.md", "trinity_retrieval/swe_navigation_benchmark_results.json"):
            dst = copy_root / sub
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_root / sub, dst)

        # Gate 2: Fail closed on NaN JSON constant
        swe_copy = copy_root / "trinity_retrieval" / "swe_navigation_benchmark_results.json"
        swe_copy.write_text(swe_copy.read_text(encoding="utf-8").replace("96.0", "NaN"), encoding="utf-8")
        try:
            _load_json_strict(swe_copy)
            nan_rejected = False
        except ReceiptParityError:
            nan_rejected = True
        _gate("2_nan_json_rejected", nan_rejected, "NaN rejected by _load_json_strict")

        # Gate 3: Fail closed on broken markdown link
        doc_copy = copy_root / "trinity_retrieval" / "README.md"
        doc_copy.write_text("[broken](does_not_exist_12345.json)\n", encoding="utf-8")
        link_errs: List[str] = []
        _check_markdown_links(doc_copy, copy_root, link_errs)
        _gate("3_broken_link_detected", len(link_errs) == 1, f"errs={link_errs}")

        # Gate 4: Fail closed on numeric drift in README claim
        t_errs: List[str] = []
        _verify_trinity_receipts(repo_root, readme="no metrics here", trinity_readme="", errors=t_errs)
        _gate("4_numeric_drift_detected", len(t_errs) >= 5, f"detected {len(t_errs)} mismatches")

        # Gate 5: Fail closed on missing root directory
        try:
            verify_repo_receipts(Path(tmp) / "missing_repo")
            missing_failed = False
        except ReceiptParityError:
            missing_failed = True
        _gate("5_missing_root_fails_closed", missing_failed, "missing JSON raised ReceiptParityError")

    # Gate 6: CLI exit-code check
    rc_bad = main(["--root", str(repo_root / "__definitely_missing__")])
    _gate("6_cli_nonzero_on_failure", rc_bad != 0, f"rc_bad={rc_bad}")

    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify doc-vs-JSON/log receipt parity across Language-U-Gemma4.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Repository root directory")
    parser.add_argument("--selftest", action="store_true", help="Run 6-gate fail-closed self-test")
    parser.add_argument("--json", action="store_true", help="Print JSON summary report")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if args.selftest:
        rc = _self_test(root)
        if rc != 0:
            raise SystemExit(rc)
        return 0

    try:
        rep = verify_repo_receipts(root)
    except ReceiptParityError as exc:
        if args.json:
            print(json.dumps({"root": str(root), "passed": False, "errors": [str(exc)]}, indent=2))
        else:
            print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "root": rep.root,
                    "passed": rep.passed,
                    "checked_jsons": rep.checked_jsons,
                    "checked_links": rep.checked_links,
                    "checked_claims": rep.checked_claims,
                    "errors": rep.errors,
                },
                indent=2,
            )
        )
    elif rep.passed:
        print(
            f"[PASS] Receipt parity verified: {rep.checked_jsons} JSON files, "
            f"{rep.checked_links} local links, {rep.checked_claims} claim checks (0 errors)"
        )
    else:
        for err in rep.errors:
            print(f"[FAIL] {err}", file=sys.stderr)
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

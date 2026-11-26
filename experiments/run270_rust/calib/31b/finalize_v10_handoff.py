#!/usr/bin/env python3
"""Turnkey V10 Post-Colab Handoff Orchestrator (Drive Check -> Sequential Pi 4 Pull -> Report Render).

Designed so a lightweight execution model (e.g. Gemini Hydrogen) can finish the V10 release
purely via deterministic CLI calls without authoring new scripts:
  1. `--mode check-drive`: Queries shared Drive folder `1Yp_bp3KZSospNVtRTnjIulv8ev14_dpn` via bare `gdrive`.
  2. `--mode pull-to-pi4`: Sequentially downloads each V10 file to temp staging, `scp`s to Pi 4
     `/mnt/SHE-VAULT/gemma4_v10/`, deletes multi-GB staging files immediately (Rule 13), archives
     `v10_report.json` + `SHA256SUMS_V10.txt` in `receipts/`, and runs `sha256sum -c` on Pi 4.
  3. `--mode render-report`: Renders `benchmarks/v10_final_proof/V10_CORTICAL_CASCADE_REPORT.md`
     from grounded `v10_report.json` + Pi 4 `v10_pi4_world_proof.json` with V9.2 vs V10 parity matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
RECEIPTS_DIR = THIS_DIR / "receipts"
BENCH_REPORT = REPO_ROOT / "experiments" / "run270_rust" / "benchmarks" / "v10_final_proof" / "V10_CORTICAL_CASCADE_REPORT.md"

DEFAULT_DRIVE_FOLDER_ID = "1Yp_bp3KZSospNVtRTnjIulv8ev14_dpn"
DEFAULT_SSH_KEY = Path(r"C:\Users\freed\.ssh\rpi_key")
DEFAULT_PI4_HOST = "astronautshe@192.168.1.207"
DEFAULT_PI4_DIR = "/mnt/SHE-VAULT/gemma4_v10"

EXPECTED_V10_FILES: Tuple[str, ...] = (
    "gemma4_31b_v10_cascade.t16c",
    "gemma4_31b_v10_router.crtr",
    "gemma4_31b_v10_head_router.crtr",
    "gemma4_31b_v10_c8d.c8d",
    "gemma4_31b_v10_assistant_body_q2.t16c",
    "gemma4_31b_v10_assistant_body_q4.t16c",
    "gemma4_31b_v10_assistant.bhdr",
    "v10_report.json",
    "SHA256SUMS_V10.txt",
)


def parse_gdrive_ls_json(raw_json: str) -> Dict[str, dict]:
    """Parse `gdrive readonly ls <folder_id> --json` output into `{filename: item_dict}`."""
    if not raw_json.strip():
        return {}
    data = json.loads(raw_json)
    items = data.get("files", data) if isinstance(data, dict) else data
    out: Dict[str, dict] = {}
    if isinstance(items, list):
        for entry in items:
            if isinstance(entry, dict) and "name" in entry and "id" in entry:
                out[str(entry["name"])] = entry
    return out


def parse_sha256sums(text: str) -> Dict[str, str]:
    """Parse standard `SHA256SUMS` lines (`<64hex>  <filename>`) fail-closed."""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed SHA256SUMS line: {raw!r}")
        out[parts[-1].lstrip("*")] = parts[0].lower()
    if not out:
        raise ValueError("Empty SHA256SUMS manifest")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _gdrive_bin() -> str:
    """Resolve `gdrive` / `gdrive.cmd` executable on Windows & POSIX for `subprocess.run`."""
    found = shutil.which("gdrive")
    if found:
        return found
    if os.name == "nt":
        cands = sorted(
            Path(tempfile.gettempdir()).glob("agy-plugin-cli-*/gdrive.CMD"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            return str(cands[0])
    raise FileNotFoundError(
        "gdrive not found on PATH; run `gdrive --version` once in the session first."
    )


def list_drive_folder(folder_id: str) -> Dict[str, dict]:
    proc = subprocess.run(
        [_gdrive_bin(), "readonly", "ls", folder_id, "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return parse_gdrive_ls_json(proc.stdout)


def cmd_check_drive(folder_id: str) -> int:
    present = list_drive_folder(folder_id)
    found = [f for f in EXPECTED_V10_FILES if f in present]
    missing = [f for f in EXPECTED_V10_FILES if f not in present]
    ready = ("v10_report.json" in present) and ("SHA256SUMS_V10.txt" in present) and (len(missing) == 0)
    status = {
        "folder_id": folder_id,
        "ready_for_download": ready,
        "found_count": len(found),
        "expected_count": len(EXPECTED_V10_FILES),
        "found": found,
        "missing": missing,
        "extra_files": sorted(k for k in present if k not in EXPECTED_V10_FILES),
    }
    print(json.dumps(status, indent=2), flush=True)
    return 0 if ready else 2


def cmd_pull_to_pi4(folder_id: str, ssh_key: Path, pi4_host: str, pi4_dir: str, staging_dir: Path) -> int:
    if not ssh_key.is_file():
        raise FileNotFoundError(f"Missing SSH key: {ssh_key}")
    present = list_drive_folder(folder_id)
    missing = [f for f in EXPECTED_V10_FILES if f not in present]
    if missing:
        raise FileNotFoundError(f"V10 Drive folder {folder_id} still missing artifacts: {missing}")

    staging_dir.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ssh", "-i", str(ssh_key), "-o", "StrictHostKeyChecking=no", pi4_host, f"mkdir -p {pi4_dir}"],
        check=True,
    )

    # Pull manifest & report first so we can verify every file locally BEFORE scp
    for meta_name in ("SHA256SUMS_V10.txt", "v10_report.json"):
        fid = str(present[meta_name]["id"])
        dst_local = RECEIPTS_DIR / meta_name
        if dst_local.exists():
            dst_local.unlink()
        subprocess.run([_gdrive_bin(), "readonly", "download", fid, "--destination", str(RECEIPTS_DIR)], check=True)
        subprocess.run(
            ["scp", "-i", str(ssh_key), "-o", "StrictHostKeyChecking=no", str(dst_local), f"{pi4_host}:{pi4_dir}/{meta_name}"],
            check=True,
        )

    expected_sha = parse_sha256sums((RECEIPTS_DIR / "SHA256SUMS_V10.txt").read_text(encoding="utf-8"))

    # Sequential single-file transfer + immediate local cleanup on C: (Rule 13)
    for fname in EXPECTED_V10_FILES:
        if fname in ("SHA256SUMS_V10.txt", "v10_report.json"):
            continue
        fid = str(present[fname]["id"])
        local_f = staging_dir / fname
        if local_f.exists():
            local_f.unlink()
        print(f"[PULL 1/3] Downloading {fname} ({fid}) from Drive...", flush=True)
        subprocess.run([_gdrive_bin(), "readonly", "download", fid, "--destination", str(staging_dir)], check=True)
        got_sha = sha256_file(local_f)
        if fname in expected_sha and got_sha != expected_sha[fname]:
            local_f.unlink(missing_ok=True)
            raise IOError(f"Local SHA-256 mismatch on {fname}: got {got_sha}, expected {expected_sha[fname]}")
        print(f"[PULL 2/3] Verified local SHA-256={got_sha[:16]}... -> SCP to {pi4_host}:{pi4_dir}/{fname}", flush=True)
        try:
            subprocess.run(
                ["scp", "-i", str(ssh_key), "-o", "StrictHostKeyChecking=no", str(local_f), f"{pi4_host}:{pi4_dir}/{fname}"],
                check=True,
            )
        finally:
            # Rule 13: Always free multi-GB staging file on Windows C: right away
            local_f.unlink(missing_ok=True)
        print(f"[PULL 3/3] Cleaned local staging copy of {fname}.", flush=True)

    print("[VERIFY PI4] Running sha256sum -c SHA256SUMS_V10.txt on Raspberry Pi 4...", flush=True)
    subprocess.run(
        ["ssh", "-i", str(ssh_key), "-o", "StrictHostKeyChecking=no", pi4_host, f"cd {pi4_dir} && sha256sum -c SHA256SUMS_V10.txt"],
        check=True,
    )
    print("[OK] All V10 artifacts verified bit-exact on Raspberry Pi 4!", flush=True)
    return 0


def render_v10_markdown_report(gpu_rpt: dict, pi4_rpt: dict) -> str:
    """Render grounded V9.2 vs V10 comparison report strictly from JSON receipts."""
    def _reject_nf(tok: str) -> None:
        raise ValueError(f"Non-finite token {tok}")

    json.loads(json.dumps(gpu_rpt), parse_constant=_reject_nf)
    json.loads(json.dumps(pi4_rpt), parse_constant=_reject_nf)

    e2e = gpu_rpt.get("e2e_stage2_5", {})
    ast = gpu_rpt.get("assistant_target_shared", {})
    pareto = pi4_rpt.get("nll_pareto", [])
    dense_p = next((r for r in pareto if str(r.get("config", "")).startswith("dense_glyph-lod")), pareto[0] if pareto else {})
    bat = pi4_rpt.get("battery", {})
    b_dense = bat.get("dense", {})
    b_cas = bat.get("cascade", {})
    b_alu = bat.get("cascade_alu", {})
    spd = pi4_rpt.get("speed_bench", [])
    vis = pi4_rpt.get("vision", {})

    spd_toks = [float(r["tok_per_s"]) for r in spd if r.get("tok_per_s") is not None]
    peak_tok_s = max(spd_toks) if spd_toks else 0.0
    mean_tok_s = (sum(spd_toks) / len(spd_toks)) if spd_toks else 0.0
    max_hwm_kb = max(
        [int(dense_p.get("vm_hwm_kb") or 0), int(b_cas.get("vm_hwm_kb") or 0)]
        + [int(r.get("vm_hwm_kb") or 0) for r in spd]
    )
    hwm_gib = max_hwm_kb / (1024.0 * 1024.0)

    lines = [
        "# Language-U V10 Cortical Cascade — Physical Raspberry Pi 4 Verification Report",
        "",
        "## 1. Executive Summary & V9.2 vs. V10 Comparison Matrix",
        "",
        "| Metric | V9.2 Cliff-Fix VLM | V10 Cortical Cascade | Target |",
        "|---|---|---|---|",
        f"| `.t16c` File Size | `6.3816 GiB` | `{gpu_rpt.get('t16c_gib')} GiB` (`{gpu_rpt.get('t16c_bytes'):,} B`) | `<= 6.30 GiB` |",
        f"| Peak Pi 4 `VmHWM` | `6.48 GiB` | `{hwm_gib:.4f} GiB` (`{max_hwm_kb:,} kB`) | `<= 6.50 GiB` |",
        f"| `ssd_read_bytes` | `0` | `{int(b_cas.get('ssd_read_bytes', 0))}` | `0` |",
        f"| Held-out 3-Passage `mean_nll` | `7.68050` | `{dense_p.get('mean_nll')}` | `< 7.68050` |",
        f"| Held-out Teacher `kl_t_to_s` | `4.80107` | `{dense_p.get('kl_t_to_s')}` | `< 4.80107` |",
        f"| 15-Prompt Battery (Pure Neural Dense) | `8/15` | `{b_dense.get('passed')}/{b_dense.get('total', 15)}` | `>= 13/15` |",
        f"| 15-Prompt Battery (Pure Neural Cascade) | `N/A` | `{b_cas.get('passed')}/{b_cas.get('total', 15)}` | `>= 13/15` |",
        f"| 15-Prompt Battery (Cascade + General ALU) | `N/A` | `{b_alu.get('passed')}/{b_alu.get('total', 15)}` | Disambiguated |",
        f"| Pi 4 Sustained / Peak `tok/s` | `0.37 tok/s` | `{mean_tok_s:.2f} / {peak_tok_s:.2f} tok/s` | `20–50 tok/s` |",
        f"| Held-Out COCO Vision (`frame_01_coco.png`) | `\"A motorcycle is parked on a grassy area.<turn|>\"` | `{vis.get('gen_text')!r}` | Clean grounded caption |",
        "",
        "## 2. L4 High-RAM Dual-Regime E2E KL QAT & Target-Shared EAGLE-3 Telemetry",
        "",
        f"- **GPU:** `{gpu_rpt.get('gpu')}` (`elapsed_s={gpu_rpt.get('elapsed_s')}`)",
        f"- **Stage 2.5 Dual E2E KL:** Dense `val_kl` `{e2e.get('val_kl_before')} -> {e2e.get('val_kl_after')}`, Sparse-12/84 `val_kl` `{e2e.get('val_sp12_kl_before')} -> {e2e.get('val_sp12_kl_after')}`, Top-1 Agreement `{e2e.get('top1_agree_before')} -> {e2e.get('top1_agree_after')}`",
        f"- **Target-Embed-Shared Assistant (`0 MB` extra LM-head):** `val_kl` `{ast.get('kl_before')} -> {ast.get('kl_after')}`, Top-1 `{ast.get('top1_before')} -> {ast.get('top1_after')}`",
        "",
        "## 3. Pi 4 Sustained Throughput Breakdown (`--cortical-cascade`)",
        "",
        "| Tag | `tok/s` | `ms/tok` | `tier2_accepted` | `tier3_rescues` | `vm_hwm_kb` |",
        "|---|---|---|---|---|---|",
    ]
    for r in spd:
        lines.append(
            f"| `{r.get('tag')}` | `{r.get('tok_per_s')}` | `{r.get('ms_per_tok')}` | `{r.get('tier2_accepted')}` | `{r.get('tier3_rescues')}` | `{r.get('vm_hwm_kb')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def cmd_render_report(gpu_json: Path, pi4_json: Path, out_md: Path) -> int:
    if not gpu_json.is_file():
        raise FileNotFoundError(f"Missing GPU receipt JSON: {gpu_json}")
    if not pi4_json.is_file():
        raise FileNotFoundError(f"Missing Pi 4 proof receipt JSON: {pi4_json}")
    gpu_rpt = json.loads(gpu_json.read_text(encoding="utf-8"))
    pi4_rpt = json.loads(pi4_json.read_text(encoding="utf-8"))
    md = render_v10_markdown_report(gpu_rpt, pi4_rpt)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote grounded V10 report: {out_md}", flush=True)
    return 0


def run_selftest() -> int:
    """6-gate CPU verification of V10 handoff automation."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: gdrive JSON parser handles both list and {"files": [...]} schemas
    m1 = parse_gdrive_ls_json('{"files": [{"id": "abc", "name": "v10_report.json"}]}')
    m2 = parse_gdrive_ls_json('[{"id": "def", "name": "SHA256SUMS_V10.txt"}]')
    gate("Gate 1: gdrive JSON Schema Parser", "v10_report.json" in m1 and "SHA256SUMS_V10.txt" in m2)

    # Gate 2: SHA256SUMS parser & fail-closed malformed check
    dig = "a" * 64
    sums = parse_sha256sums(f"{dig}  gemma4_31b_v10_cascade.t16c\n")
    bad_sums = False
    try:
        parse_sha256sums("not_a_valid_sha  bad.bin\n")
    except ValueError:
        bad_sums = True
    gate("Gate 2: SHA256SUMS Parser & Corruption Trap", sums.get("gemma4_31b_v10_cascade.t16c") == dig and bad_sums)

    # Gate 3: Streaming SHA-256 file helper exactness
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        p.write_bytes(b"gemma4-v10")
        got = sha256_file(p)
        exp = hashlib.sha256(b"gemma4-v10").hexdigest()
        gate("Gate 3: Streaming File SHA-256", got == exp)

    # Gate 4: Grounded Markdown report renderer & V9.2 vs V10 matrix
    gpu_mock = {
        "gpu": "NVIDIA L4", "elapsed_s": 4200.0, "t16c_gib": 6.2675, "t16c_bytes": 6729678848,
        "e2e_stage2_5": {"val_kl_before": 2.4, "val_kl_after": 1.8, "val_sp12_kl_before": 3.1, "val_sp12_kl_after": 2.2, "top1_agree_before": 0.61, "top1_agree_after": 0.71},
        "assistant_target_shared": {"kl_before": 1.9, "kl_after": 0.7, "top1_before": 0.58, "top1_after": 0.81},
    }
    pi4_mock = {
        "nll_pareto": [{"config": "dense_glyph-lod", "mean_nll": 7.12, "kl_t_to_s": 3.95, "vm_hwm_kb": 6610000}],
        "battery": {"dense": {"passed": 14, "total": 15}, "cascade": {"passed": 14, "total": 15, "vm_hwm_kb": 6680000, "ssd_read_bytes": 0}, "cascade_alu": {"passed": 15, "total": 15}},
        "speed_bench": [{"tag": "speed_code", "tok_per_s": 28.4, "ms_per_tok": 35.2, "tier2_accepted": 54, "tier3_rescues": 2, "vm_hwm_kb": 6685000}],
        "vision": {"gen_text": "A motorcycle is parked on grass.<turn|>"},
    }
    md = render_v10_markdown_report(gpu_mock, pi4_mock)
    gate("Gate 4: Grounded V9.2 vs V10 Markdown Renderer", "28.40 / 28.40 tok/s" in md and "14/15" in md)

    # Gate 5: Non-finite JSON trap fails closed
    nf_caught = False
    try:
        render_v10_markdown_report({"t16c_bytes": 0, "bad": float("nan")}, pi4_mock)
    except ValueError:
        nf_caught = True
    gate("Gate 5: Non-Finite JSON Trap Fails Closed", nf_caught)

    # Gate 6: Missing receipt file fails closed
    miss_caught = False
    try:
        cmd_render_report(Path("/nonexistent/gpu.json"), Path("/nonexistent/pi4.json"), Path("/tmp/out.md"))
    except FileNotFoundError:
        miss_caught = True
    gate("Gate 6: Missing Receipt File Fails Closed", miss_caught)

    print(f"V10_HANDOFF_ORCHESTRATOR_SELFTEST: {6 - failures}/6", flush=True)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Turnkey V10 Post-Colab Handoff Orchestrator")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=("check-drive", "pull-to-pi4", "render-report"), default="check-drive")
    ap.add_argument("--folder-id", type=str, default=DEFAULT_DRIVE_FOLDER_ID)
    ap.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    ap.add_argument("--pi4-host", type=str, default=DEFAULT_PI4_HOST)
    ap.add_argument("--pi4-dir", type=str, default=DEFAULT_PI4_DIR)
    ap.add_argument("--staging-dir", type=Path, default=Path(tempfile.gettempdir()) / "v10_staging")
    ap.add_argument("--gpu-report", type=Path, default=RECEIPTS_DIR / "v10_report.json")
    ap.add_argument("--pi4-report", type=Path, default=RECEIPTS_DIR / "v10_pi4_world_proof.json")
    ap.add_argument("--out-md", type=Path, default=BENCH_REPORT)
    args = ap.parse_args()

    if len(sys.argv) == 1 or args.selftest:
        return run_selftest()
    if args.mode == "check-drive":
        return cmd_check_drive(args.folder_id)
    if args.mode == "pull-to-pi4":
        return cmd_pull_to_pi4(args.folder_id, args.ssh_key, args.pi4_host, args.pi4_dir, args.staging_dir)
    return cmd_render_report(args.gpu_report, args.pi4_report, args.out_md)


if __name__ == "__main__":
    raise SystemExit(main())

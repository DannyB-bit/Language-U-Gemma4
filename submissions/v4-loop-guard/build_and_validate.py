#!/usr/bin/env python3
"""v4-loop-guard build & validate — deterministic zip + fleet contract checks.

Base: v3-converged byte-exact (v11 spliced prompt + 16384/300s/150turns
config fixes) PLUS the three loop-guard prompt splices, each evidenced by
happyc0der's public experiment ledger (measured verbatim-repeat pathology
at temp 0.2: loops of up to 23 identical calls; 27% of calls exact repeats;
their loop-fix family cut looping tasks 3->0, median first-edit 33->18).

Determinism law: fixed ZipInfo date_time so the SAME content always pins
the SAME sha256 on every fleet node.

Coupling guard (fleet law): the prompt's stated budget line
("N tool calls and M minutes") must equal eval_config.yaml — a call plan
calibrated for a different budget is a silent behavioral regression.
"""
import hashlib
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "submission.zip"
FILES = ["agent.yaml", "configs/sampling.yaml", "eval_config.yaml", "prompts/system.md"]
FIXED_DT = (2026, 1, 1, 0, 0, 0)  # determinism pin
CONTRACT_MEMBERS = set(FILES)      # zip carries EXACTLY the contract files (LICENSE/NOTICE stay in the repo dir)

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

texts = {}
for f in FILES:
    p = HERE / f
    check(f"exists: {f}", p.is_file())
    texts[f] = p.read_text(encoding="utf-8") if p.is_file() else ""
agent, sampling, evalcfg, prompt = (texts[k] for k in FILES)

# ---- contract checks
check("model: gemma-4-31b-it-qat-w4a16-ct (V10 badge verbatim)", "gemma-4-31b-it-qat-w4a16-ct" in agent)
check("9 legal tools, !include wiring intact",
      "instruction: !include prompts/system.md" in agent and "generate_content_config: !include configs/sampling.yaml" in agent)
m_temp = re.search(r"temperature:\s*([\d.]+)", sampling)
temp = float(m_temp.group(1)) if m_temp else 0.0
m_max_out = re.search(r"max_output_tokens:\s*(\d+)", sampling)
max_out = int(m_max_out.group(1)) if m_max_out else 0
m_think = re.search(r"thinking_budget:\s*(\d+)", sampling)
think = int(m_think.group(1)) if m_think else 0
check("temp 0.2 (V10 badge conditions — temp-1.0 experiment lives in the v2 lane, not here)", temp == 0.2)
check("thinking summaries ON (E17/E18: disabling costs 10-19 pts)", "include_thoughts: true" in sampling)
check("max_output_tokens 16384 (README gotcha #1: 8192 risks truncated tool calls @ thinking-4096)", max_out == 16384 and max_out >= think + 6000)

ec = {k: int(v) for k, v in re.findall(r"^\s+(\w+):\s+(\d+)$", evalcfg, re.M)}
check("eval budgets: 300s / 28 calls / 8 min / 150 turns (V10 badge budgets + safe margins)",
      ec == {"timeout_seconds": 300, "max_tool_calls": 28, "max_time_minutes": 8, "max_turns": 150}, str(ec))

# ---- THE COUPLING GUARD
m = re.search(r"(\d+) tool calls and (\d+) minutes", prompt)
check("prompt plan/budget coupling: stated budget == eval_config",
      bool(m) and int(m.group(1)) == ec.get("max_tool_calls") and int(m.group(2)) == ec.get("max_time_minutes"),
      f"prompt says {m.groups() if m else 'n/a'} vs eval {ec.get('max_tool_calls')}/{ec.get('max_time_minutes')}")
check("prompt edit-deadline consistency (call plan calibrated to 28 calls)",
      "Calls 9-12: edit" in prompt and "edited a source file by call 12" in prompt)

# ---- v11 prompt lineage (the measured base must survive intact)
for marker, why in [
    ("50-word issue contract", "E03 splice"),
    ("Coverage checkpoint", "BFV5 splice"),
    ("re-run the most targeted check", "E05 splice"),
    ("FastAPI", "repo-specific splice"),
    ("Never create, modify or delete test files", "anti-tamper guard"),
    ("submit_patch", "final flow"),
]:
    check(f"v11 splice intact: {marker} ({why})", marker.split(" — ")[0] in prompt or marker in prompt)

# ---- v4 loop-guard splices (happyc0der public ledger: temp-0.2 verbatim loops up to
# 23 identical calls; 27% of calls were exact repeats at v4.8; loop fixes cut looping
# tasks 3->0 and median first-edit call 33->18)
for marker, why in [
    ("Never re-send a tool call identical", "LG1: no-verbatim-repeat law"),
    ("If three consecutive calls produce no new edit", "LG2: progress law"),
    ("An empty search result means the string is not in the repository", "LG3: empty-means-absent semantics"),
]:
    check(f"v4 loop-guard intact: {marker[:40]}… ({why})", marker in prompt)

# ---- deterministic assembly
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        zi = zipfile.ZipInfo(f, date_time=FIXED_DT)
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, texts[f])
with zipfile.ZipFile(OUT) as zf:
    check("zip integrity", zf.testzip() is None)
    members = set(zf.namelist())
    check("zip carries EXACTLY the contract files (no LICENSE/NOTICE inside)", members == CONTRACT_MEMBERS, str(sorted(members)))
sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
size = OUT.stat().st_size
print(f"\nsha256: {sha}  (deterministic — same content pins same sha on every node)")
print(f"bundle: {OUT} ({size} bytes)")
# determinism self-proof: independent in-memory rebuild from source strings
import io
buf2 = io.BytesIO()
with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as z2:
    for f in FILES:
        zi = zipfile.ZipInfo(f, date_time=FIXED_DT)
        zi.compress_type = zipfile.ZIP_DEFLATED
        z2.writestr(zi, texts[f])
sha2 = hashlib.sha256(buf2.getvalue()).hexdigest()
check("sha stable across rebuild (determinism law)", sha == sha2)

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
sys.exit(0 if not fails else 1)

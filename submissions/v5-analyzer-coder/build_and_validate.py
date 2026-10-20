#!/usr/bin/env python3
"""v5-analyzer-coder build & validate — deterministic zip + fleet contract checks.

The v5 candidate: Roman Rozen's TOP-1 `analyzer+coder` two-agent architecture
(public 0.24 badge lineage), verbatim prompts, architecture-ISOLATED against
the v3 control (same 28-call / 8-min / 300s / 150-turn budget, same 16384
output fix) so the paired A/B measures the architecture variable alone.

Known deviations from Roman's published defaults (both flagged in
docs/fleet/SCORING_POSTMORTEM_015.md, both to be settled by the A/B):
  - max_output_tokens 16384 (his 8192): the harness README's gotcha #1 —
    thinking_budget 4096 + a big edit can truncate the tool_call tag at 8192.
  - include_thoughts true (his false): public ablation E17/E18 measured
    summaries-off as catastrophic; contradiction noted for the evaluator.

Determinism law: fixed ZipInfo date_time (2026-01-01) so the same content
always pins the same sha256 on every fleet node.
"""
import hashlib
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "submission.zip"
FILES = [
    "agent.yaml",
    "configs/sampling.yaml",
    "eval_config.yaml",
    "prompts/system.md",
    "prompts/analyzer.md",
    "sub_agents/code_analyzer.yaml",
]
FIXED_DT = (2026, 1, 1, 0, 0, 0)  # determinism pin
CONTRACT_MEMBERS = set(FILES)

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

texts = {}
for f in FILES:
    p = HERE / f
    check(f"exists: {f}", p.is_file())
    texts[f] = p.read_text() if p.is_file() else ""
agent, sampling, evalcfg, prompt, analyzer, subagent = (texts[k] for k in FILES)

# ---- chassis + wiring
check("model: gemma-4-31b-it-qat-w4a16-ct (0.24-badge chassis)", "gemma-4-31b-it-qat-w4a16-ct" in agent)
check("main-agent !include wiring intact",
      "instruction: !include prompts/system.md" in agent and "generate_content_config: !include configs/sampling.yaml" in agent)
check("analyzer wired as agent_tool (skip_summarization: true)",
      "agent_tool:" in agent and "config_path: sub_agents/code_analyzer.yaml" in agent and "skip_summarization: true" in agent)
check("sub-agent !include wiring intact", "instruction: !include ../prompts/analyzer.md" in subagent)

# ---- tool split (Roman's CODER_TOOLS vs ANALYZER_TOOLS)
coder_ok = all(t in agent for t in ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"])
check("coder carries exactly Roman's 6 tools (nav tools moved to analyzer)", coder_ok)
check("coder does NOT carry nav tools (they live in the analyzer — context isolation)",
      "search_similar_code" not in agent.split("agent_tool:")[0] and "get_code_neighbors" not in agent.split("agent_tool:")[0])
check("analyzer carries the 5 read-only nav tools",
      all(t in subagent for t in ["run_command", "read_file", "search_similar_code", "get_code_neighbors", "get_code_subgraph"]))

# ---- sampling contract
temp = float(re.search(r"temperature:\s*([\d.]+)", sampling).group(1))
max_out = int(re.search(r"max_output_tokens:\s*(\d+)", sampling).group(1))
think = int(re.search(r"thinking_budget:\s*(\d+)", sampling).group(1))
check("temp 0.2 (Roman's 0.24-badge condition — isolation vs v3 control)", temp == 0.2)
check("thinking summaries ON (E17/E18; Roman-false contradiction flagged for the A/B)", "include_thoughts: true" in sampling)
check("max_output_tokens 16384 (README gotcha #1; deviation from Roman's 8192 flagged)", max_out == 16384 and max_out >= think + 6000)

# ---- eval budget: MUST equal the v3 control (the isolation law)
ec = {k: int(v) for k, v in re.findall(r"^\s+(\w+):\s+(\d+)$", evalcfg, re.M)}
check("eval budgets == v3 control (300s / 28 calls / 8 min / 150 turns) — architecture-isolated A/B",
      ec == {"timeout_seconds": 300, "max_tool_calls": 28, "max_time_minutes": 8, "max_turns": 150}, str(ec))

# ---- coupling guard (v5 form): the prompt states no numeric budget that could
# contradict eval_config (Roman's prompt carries no numbers — the harness injects them)
m = re.search(r"(\d+) tool calls and (\d+) minutes", prompt)
check("prompt carries no conflicting numeric budget (numbers live in eval_config only)",
      m is None or (int(m.group(1)) == ec.get("max_tool_calls") and int(m.group(2)) == ec.get("max_time_minutes")))

# ---- architecture lineage (the 0.24 recipe must survive verbatim)
for marker, why in [
    ("Call the `code_analyzer` tool with the full issue text first", "two-agent architecture entry"),
    ("LOCATION / ROOT CAUSE / FIX PLAN", "analyzer handoff contract"),
    ("Verify its claim by reading those exact lines", "coder verification duty"),
    ("Answer format (at most 250 words, nothing else)", "analyzer bounded report"),
    ("get_status` every ~8 tool calls", "budget discipline"),
    ("Never edit, add or delete tests", "anti-tamper guard"),
    ("submit_patch", "final flow"),
    ("Scratch files go to /tmp only", "patch hygiene"),
]:
    check(f"Roman lineage intact: {marker[:48]}… ({why})", marker in prompt or marker in analyzer)

# ---- deterministic assembly
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        zi = zipfile.ZipInfo(f, date_time=FIXED_DT)
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, texts[f])
zf = zipfile.ZipFile(OUT)
check("zip integrity", zf.testzip() is None)
members = set(zf.namelist())
check("zip carries EXACTLY the v5 contract files", members == CONTRACT_MEMBERS, str(sorted(members)))
sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
size = OUT.stat().st_size
print(f"\nsha256: {sha}  (deterministic — same content pins same sha on every node)")
print(f"bundle: {OUT} ({size} bytes)")
sha2 = hashlib.sha256(OUT.read_bytes()).hexdigest()
check("sha stable across rebuild (determinism law)", sha == sha2)

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
sys.exit(0 if not fails else 1)

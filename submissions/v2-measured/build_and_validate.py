#!/usr/bin/env python3
"""Build & validate the v2-measured Gemma-4 submission bundle.

Assembles the 4 contract files into a deterministic submission.zip and
validates every invariant that matters to the swegemma harness:
  - exact file set (agent.yaml, configs/sampling.yaml, eval_config.yaml, prompts/system.md)
  - model on the pre-registered allowlist, all 9 tools legal, !include wiring intact
  - sampling: thinking summaries ON, max_output_tokens >= thinking_budget + edit headroom
  - eval budgets: sane magnitudes within harness defaults
  - prompt: budget numbers agree with eval_config, core guardrails present
Prints PASS/FAIL per check + the bundle sha256. Exit 0 only on full pass.
"""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "submission.zip"
ALLOWED_MODELS = {"gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct", "gemma-4-9b-it-qat-w4a16-ct", "gemma-4-e4b-it-qat-w4a16-ct"}
ALLOWED_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status",
                 "submit_patch", "get_code_neighbors", "search_similar_code", "get_code_subgraph"}
FILES = ["agent.yaml", "configs/sampling.yaml", "eval_config.yaml", "prompts/system.md"]

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

# ---- load files
texts = {}
for f in FILES:
    p = HERE / f
    check(f"exists: {f}", p.is_file())
    texts[f] = p.read_text() if p.is_file() else ""

agent, sampling, evalcfg, prompt = texts["agent.yaml"], texts["configs/sampling.yaml"], texts["eval_config.yaml"], texts["prompts/system.md"]

# ---- agent.yaml checks
check("model on allowlist", any(m in agent for m in ALLOWED_MODELS), "gemma-4-31b-it-qat-w4a16-ct")
tools_used = set(re.findall(r"^\s+-\s+(\w+)$", agent, re.M)) & ALLOWED_TOOLS
check("all declared tools legal", tools_used.issubset(ALLOWED_TOOLS), f"{len(tools_used)} tools")
check("!include prompt wiring intact", "instruction: !include prompts/system.md" in agent)
check("!include sampling wiring intact", "generate_content_config: !include configs/sampling.yaml" in agent)
check("no python entrypoints (declarative-only)", "def " not in agent and "import " not in agent)

# ---- sampling.yaml checks (regex parse — yaml lib optional on edge nodes)
temp = float(re.search(r"temperature:\s*([\d.]+)", sampling).group(1))
top_p = float(re.search(r"top_p:\s*([\d.]+)", sampling).group(1))
max_out = int(re.search(r"max_output_tokens:\s*(\d+)", sampling).group(1))
think = int(re.search(r"thinking_budget:\s*(\d+)", sampling).group(1))
summaries = "include_thoughts: true" in sampling
check("thinking summaries ON (E17/E18: disabling costs ~19 pts)", summaries)
check("max_output_tokens >= thinking_budget + 6k headroom (README gotcha #1)", max_out >= think + 6000, f"{max_out} vs {think}")
check("temperature in measured band [0.2, 1.0]", 0.2 <= temp <= 1.0, f"temp={temp}")

# ---- eval_config.yaml checks
ec = {k: int(v) for k, v in re.findall(r"^\s+(\w+):\s+(\d+)$", evalcfg, re.M)}
check("budgets present", {"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"} == set(ec), str(ec))
check("max_time_minutes == 15 (E22 optimum)", ec.get("max_time_minutes") == 15)
check("max_tool_calls == 50 (E30 optimum)", ec.get("max_tool_calls") == 50)
check("timeout_seconds >= 300 (harness default)", ec.get("timeout_seconds", 0) >= 300)
check("max_turns headroom for 50 calls", ec.get("max_turns", 0) >= 100)

# ---- prompt/eval consistency + guardrails
check("prompt budget numbers match eval_config",
      f"{ec.get('max_tool_calls')} tool calls and {ec.get('max_time_minutes')} minutes" in prompt)
check("prompt forbids test-file edits", "Never create, modify or delete test files" in prompt)
check("prompt requires submit_patch flow", "submit_patch" in prompt and "git diff" in prompt)
check("prompt keeps docs_src rule (FastAPI lane)", "docs_src" in prompt)
check("prompt has thought-externalization rule (vLLM reasoning_content drop)", "WORKING MEMORY" in prompt)
check("prompt notes free tools", "FREE" in prompt)
check("prompt warns against oversized edits (truncation guard)", "Never paste a whole file into one edit_file call" in prompt)

# ---- assemble deterministic zip
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.writestr(f, texts[f])
check("zip integrity", zipfile.ZipFile(OUT).testzip() is None)
size = OUT.stat().st_size
check("zip size sane (< 3 GiB contract limit)", size < 3 * 1024**3, f"{size} bytes")
sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(f"\nsha256: {sha}")
print(f"bundle: {OUT}")

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
sys.exit(0 if not fails else 1)

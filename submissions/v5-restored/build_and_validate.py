#!/usr/bin/env python3
"""Build & validate the v5-restored Gemma-4 submission bundle.

v5 = the prompt that actually drew our 0.17 (honghanhhh 0.24-badge lineage,
byte-exact) + ONLY the three zero-coupling config fixes (output 16384,
timeout 300s, turns 150). Temp 0.2 / 28 calls / 8 min stay verbatim — the
badge conditions of our best draw.

Guards enforced (the lessons of the 0.15 regression, made structural):
  - LINEAGE GUARD: the prompt sha must equal the banked 0.17 prompt
    (ff760906…) — a prompt-family swap between draws can never ship silently.
  - COUPLING GUARD: the prompt's stated budget must equal eval_config.
  - DETERMINISM LAW: pinned ZipInfo timestamps — same content, same sha,
    every node (the fleet law from PR #4).
Exit 0 only on full pass. Prints the bundle sha256.
"""
import hashlib
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "submission.zip"
FILES = ["agent.yaml", "configs/sampling.yaml", "eval_config.yaml", "prompts/system.md"]
ALLOWED_MODELS = {"gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                  "gemma-4-9b-it-qat-w4a16-ct", "gemma-4-e4b-it-qat-w4a16-ct"}
ALLOWED_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status",
                 "submit_patch", "get_code_neighbors", "search_similar_code", "get_code_subgraph"}
LINEAGE_SHA = "234969efed3a86a519d74be15beea0381f1b63e0d92c18a47cad82a15956bb74"  # the TRUE 0.17-drawing prompt (fleet-consensus-v1, 64 lines, 7876 B, ref 57366201)

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

texts = {}
for f in FILES:
    p = HERE / f
    check(f"exists: {f}", p.is_file())
    texts[f] = p.read_text(encoding="utf-8") if p.is_file() else ""

agent, sampling, evalcfg, prompt = (texts["agent.yaml"], texts["configs/sampling.yaml"],
                                    texts["eval_config.yaml"], texts["prompts/system.md"])

# ---- LINEAGE GUARD (the 0.15 lesson): prompt must be the 0.17-drawing bytes
psha = hashlib.sha256(prompt.encode()).hexdigest()
check("LINEAGE: prompt is byte-exact 0.17 lineage (no family swap)", psha == LINEAGE_SHA, psha[:16])

# ---- agent.yaml
check("model on allowlist", any(m in agent for m in ALLOWED_MODELS))
tools_used = set(re.findall(r"^\s+-\s+(\w+)$", agent, re.M)) & ALLOWED_TOOLS
check("9 legal tools", tools_used.issubset(ALLOWED_TOOLS), f"{len(tools_used)} tools")
check("!include wiring intact (prompt + sampling)",
      "instruction: !include prompts/system.md" in agent
      and "generate_content_config: !include configs/sampling.yaml" in agent)
check("declarative-only (no python)", "def " not in agent and "import " not in agent)

# ---- sampling: badge conditions verbatim + the one fix
m_temp = re.search(r"temperature:\s*([\d.]+)", sampling)
temp = float(m_temp.group(1)) if m_temp else 0.0
m_max_out = re.search(r"max_output_tokens:\s*(\d+)", sampling)
max_out = int(m_max_out.group(1)) if m_max_out else 0
m_think = re.search(r"thinking_budget:\s*(\d+)", sampling)
think = int(m_think.group(1)) if m_think else 0
check("temp 0.2 (badge conditions of the 0.17 draw)", temp == 0.2)
check("thinking summaries ON", "include_thoughts: true" in sampling)
check("max_output_tokens 16384 (README gotcha #1 fix)", max_out == 16384)
check("headroom law (output >= thinking + 6k)", max_out >= think + 6000, f"{max_out} vs {think}")

# ---- eval_config: badge budgets + zero-coupling margins
ec = {k: int(v) for k, v in re.findall(r"^\s+(\w+):\s+(\d+)$", evalcfg, re.M)}
check("budgets present", {"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"} == set(ec), str(ec))
check("28 calls / 8 min verbatim (badge conditions)", ec.get("max_tool_calls") == 28 and ec.get("max_time_minutes") == 8)
check("timeout 300s (harness default)", ec.get("timeout_seconds") == 300)
check("turns 150 (pure margin)", ec.get("max_turns") == 150)

# ---- COUPLING GUARD: prompt-stated budget == eval_config
m = re.search(r"(\d+) tool calls and (\d+) minutes", prompt)
check("prompt/eval budget coupling", m and (int(m.group(1)), int(m.group(2))) == (ec["max_tool_calls"], ec["max_time_minutes"]),
      f"prompt says {m.groups() if m else 'n/a'} vs eval {ec.get('max_tool_calls')}/{ec.get('max_time_minutes')}")

# ---- deterministic zip (fleet law: pinned timestamps)
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        zi = zipfile.ZipInfo(f, date_time=(2026, 1, 1, 0, 0, 0))
        zi.external_attr = 0o644 << 16
        z.writestr(zi, texts[f])
with zipfile.ZipFile(OUT) as _zf:
    check("zip integrity + exactly-4 contract members",
          _zf.testzip() is None and _zf.namelist() == FILES)
sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(f"\nsha256: {sha}")
print(f"bundle: {OUT} ({OUT.stat().st_size} bytes)")

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
sys.exit(0 if not fails else 1)

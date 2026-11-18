#!/usr/bin/env python3
"""v6-optimal build & full schema/harness contract validator (deterministic zip)."""

from __future__ import annotations

import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "submission.zip"
FILES = [
    "agent.yaml",
    "configs/sampling.yaml",
    "eval_config.yaml",
    "prompts/system.md",
]
FIXED_DT = (2026, 1, 1, 0, 0, 0)
CONTRACT_MEMBERS = set(FILES)
ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}
ALLOWED_TOOLS = {
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "get_status",
    "submit_patch",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
}
FORBIDDEN_GCONF = {"tools", "system_instruction", "http_options", "safety_settings", "response_schema"}

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


texts = {}
for f in FILES:
    p = HERE / f
    check(f"exists: {f}", p.is_file())
    texts[f] = p.read_text(encoding="utf-8") if p.is_file() else ""


class _IncludeLoader(yaml.SafeLoader):
    pass


def _include_ctor(loader: yaml.SafeLoader, node: yaml.Node):
    rel = loader.construct_scalar(node)
    assert not rel.startswith("/") and ".." not in Path(rel).parts, f"Unsafe !include path: {rel}"
    assert rel in texts, f"Missing !include target: {rel}"
    return texts[rel] if rel.endswith((".md", ".txt")) else yaml.safe_load(texts[rel])


_IncludeLoader.add_constructor("!include", _include_ctor)

agent = yaml.load(texts["agent.yaml"], Loader=_IncludeLoader)
sampling = yaml.safe_load(texts["configs/sampling.yaml"])
eval_full = yaml.safe_load(texts["eval_config.yaml"])
ev = eval_full["evaluation"]
prompt = texts["prompts/system.md"]

check("model in ALLOWED_MODEL_NAMES", agent.get("model") in ALLOWED_MODELS, agent.get("model", ""))
check("single root agent (no sub_agents / zero missing-model risk)", not agent.get("sub_agents"))
check("no dead adapter reference", agent.get("adapter") is None)
tools = agent.get("tools", [])
check("exact 6 core tools valid", tools == ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"] and set(tools) <= ALLOWED_TOOLS)

gconf = agent.get("generate_content_config", {})
check("no forbidden generate_content_config keys", not (set(gconf) & FORBIDDEN_GCONF))
check("Gemma 4 optimal sampling (T=1.0, p=0.95, k=64)", gconf.get("temperature") == 1.0 and gconf.get("top_p") == 0.95 and gconf.get("top_k") == 64)
check("max_output_tokens=12288 within [1,32768] & >= thinking+8192", gconf.get("max_output_tokens") == 12288)
tconf = gconf.get("thinking_config", {})
check("thinking_budget=4096 & include_thoughts=true & no thinking_level", tconf.get("thinking_budget") == 4096 and tconf.get("include_thoughts") is True and "thinking_level" not in tconf)

check(
    "eval budgets runtime-safe (90s / 34 calls / 8 min / 100 turns) & turns > calls",
    ev == {"timeout_seconds": 90, "max_tool_calls": 34, "max_time_minutes": 8, "max_turns": 100} and ev["max_turns"] > ev["max_tool_calls"],
    str(ev),
)

check("prompt coupling == eval_config (34 tool calls, 8 minutes)", "(34 tool calls, 8 minutes)" in prompt and "34 paid tool calls" in prompt)
check("prompt contains ZERO '{' or '}' ADK template-break braces", "{" not in prompt and "}" not in prompt)
check("prompt concise (< 4500 chars)", len(prompt) < 4500, f"{len(prompt)} chars")
for marker in ("by **call 12**", "one short line", "python3 -m py_compile", "submit_patch"):
    check(f"prompt rule present: {marker}", marker in prompt)

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(FILES):
        zi = zipfile.ZipInfo(f, date_time=FIXED_DT)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o100644 << 16
        z.writestr(zi, texts[f].encode("utf-8"))

with zipfile.ZipFile(OUT) as zf:
    check("zip integrity", zf.testzip() is None)
    check("zip exact 4 members", set(zf.namelist()) == CONTRACT_MEMBERS, str(zf.namelist()))
    total_unpacked = sum(i.file_size for i in zf.infolist())
    check("unpacked size < 3 GiB", total_unpacked < 3 * 1024**3, f"{total_unpacked} B")

sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
buf2 = io.BytesIO()
with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as z2:
    for f in sorted(FILES):
        zi = zipfile.ZipInfo(f, date_time=FIXED_DT)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o100644 << 16
        z2.writestr(zi, texts[f].encode("utf-8"))
check("deterministic rebuild sha parity", sha == hashlib.sha256(buf2.getvalue()).hexdigest())

fails = [r for r in results if not r[1]]
print(f"\nsha256: {sha}\nbundle: {OUT} ({OUT.stat().st_size} B)\n{len(results)-len(fails)}/{len(results)} checks passed")
raise SystemExit(0 if not fails else 1)

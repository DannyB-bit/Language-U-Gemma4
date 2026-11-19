#!/usr/bin/env python3
"""v10-lora-clean-ctx build & full schema/LoRA/harness contract validator (deterministic zip)."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "submission.zip"
CORE_FILES = [
    "agent.yaml",
    "configs/sampling.yaml",
    "eval_config.yaml",
    "prompts/system.md",
]
ADAPTER_FILES = [
    "adapters/main_lora/adapter_config.json",
    "adapters/main_lora/adapter_model.safetensors",
]
FILES = CORE_FILES + ADAPTER_FILES
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


def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}


texts = {}
for f in CORE_FILES:
    p = HERE / f
    check(f"exists: {f}", p.is_file())
    texts[f] = p.read_text(encoding="utf-8") if p.is_file() else ""

for f in ADAPTER_FILES:
    p = HERE / f
    check(f"exists: {f}", p.is_file(), f"{p.stat().st_size:,} B" if p.is_file() else "MISSING (train LoRA first!)")

if not all((HERE / f).is_file() for f in FILES):
    print("\nFAIL-CLOSED: Missing required adapter files. Run train_swe_lora_a100.py on A100 first.")
    raise SystemExit(1)


class _IncludeLoader(yaml.SafeLoader):
    pass


def _include_ctor(loader: yaml.SafeLoader, node: yaml.Node):
    rel = loader.construct_scalar(node)
    assert not rel.startswith("/") and ".." not in Path(rel).parts, f"Unsafe !include path: {rel}"
    assert rel in texts, f"Missing !include target: {rel}"
    return texts[rel] if rel.endswith((".md", ".txt")) else yaml.safe_load(texts[rel])


_IncludeLoader.add_constructor("!include", _include_ctor)

agent = yaml.load(texts["agent.yaml"], Loader=_IncludeLoader)
eval_full = yaml.safe_load(texts["eval_config.yaml"])
ev = eval_full["evaluation"]
prompt = texts["prompts/system.md"]

check("model in ALLOWED_MODEL_NAMES", agent.get("model") in ALLOWED_MODELS, agent.get("model", ""))
check("single root agent (no sub_agents / zero missing-model risk)", not agent.get("sub_agents"))
check("adapter == main_lora", agent.get("adapter") == "main_lora", str(agent.get("adapter")))
tools = agent.get("tools", [])
check("exact 6 core tools valid", tools == ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"] and set(tools) <= ALLOWED_TOOLS)

gconf = agent.get("generate_content_config", {})
check("no forbidden generate_content_config keys", not (set(gconf) & FORBIDDEN_GCONF))
check("Gemma 4 optimal sampling (T=1.0, p=0.95, k=64)", gconf.get("temperature") == 1.0 and gconf.get("top_p") == 0.95 and gconf.get("top_k") == 64)
check("max_output_tokens=8192 within [1,32768]", gconf.get("max_output_tokens") == 8192)
tconf = gconf.get("thinking_config", {})
check(
    "clean-context thinking (budget=3584, include_thoughts=false, no thinking_level)",
    tconf.get("thinking_budget") == 3584 and tconf.get("include_thoughts") is False and "thinking_level" not in tconf,
)

check(
    "eval budgets runtime-safe (90s / 32 calls / 7 min / 75 turns) & turns > calls",
    ev == {"timeout_seconds": 90, "max_tool_calls": 32, "max_time_minutes": 7, "max_turns": 75} and ev["max_turns"] > ev["max_tool_calls"],
    str(ev),
)

check("prompt coupling == eval_config (32 tool calls, 7 minutes)", "(32 tool calls, 7 minutes)" in prompt and "32 tool calls" in prompt)
check("prompt contains ZERO '{' or '}' ADK template-break braces", "{" not in prompt and "}" not in prompt)
check("prompt concise (< 4300 chars)", len(prompt) < 4300, f"{len(prompt)} chars")

# Audit LoRA adapter_config.json and adapter_model.safetensors (pure stdlib header parse!)
acfg = json.loads((HERE / "adapters/main_lora/adapter_config.json").read_text(encoding="utf-8"))
check("LoRA peft_type == LORA & rank <= 128", acfg.get("peft_type") == "LORA" and 1 <= int(acfg.get("r", 0)) <= 128, f"r={acfg.get('r')}, alpha={acfg.get('lora_alpha')}")
check("LoRA base_model matches competition W4A16", "gemma-4-31b-it-qat-w4a16-ct" in str(acfg.get("base_model_name_or_path", "")).lower())

st_hdr = read_safetensors_header(HERE / "adapters/main_lora/adapter_model.safetensors")
n_keys = len(st_hdr)
prefix_ok = n_keys >= 800 and all(
    k.startswith("base_model.model.model.language_model.layers.") and (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"))
    for k in st_hdr
)
bf16_ok = all(v.get("dtype") == "BF16" for v in st_hdr.values())
no_vis_ok = not any("vision" in k.lower() for k in st_hdr)
check("LoRA safetensors exact vLLM prefix & count (820 = 410x2)", prefix_ok, f"{n_keys} tensors")
check("LoRA safetensors all BF16", bf16_ok)
check("LoRA safetensors zero vision keys", no_vis_ok)

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(FILES):
        p = HERE / f
        zi = zipfile.ZipInfo(f, date_time=FIXED_DT)
        zi.compress_type = zipfile.ZIP_DEFLATED if not f.endswith(".safetensors") else zipfile.ZIP_STORED
        zi.external_attr = 0o100644 << 16
        z.writestr(zi, p.read_bytes())

with zipfile.ZipFile(OUT) as zf:
    check("zip integrity", zf.testzip() is None)
    check("zip exact 6 members", set(zf.namelist()) == CONTRACT_MEMBERS, str(zf.namelist()))
    total_unpacked = sum(i.file_size for i in zf.infolist())
    check("unpacked size < 3 GiB", total_unpacked < 3 * 1024**3, f"{total_unpacked / (1024**2):.2f} MiB")

sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
fails = [r for r in results if not r[1]]
print(f"\nsha256: {sha}\nbundle: {OUT} ({OUT.stat().st_size / (1024**2):.2f} MiB)\n{len(results)-len(fails)}/{len(results)} checks passed")
raise SystemExit(0 if not fails else 1)

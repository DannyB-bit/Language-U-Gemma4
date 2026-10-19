#!/usr/bin/env python3
"""Language-U-Gemma4 — KOFI v11 bundle builder (deterministic, offline).

Builds agent/submission.zip: the fleet's improved Gemma 4 Developer Agent
submission. NO network, NO model calls, NO submission — build + validate only.

Ancestry (all Apache-2.0, credited in NOTICE.txt):
  verracodeguacas "Budget-Fit Single Agent V5" (base config)
  → flexonafft "Budget-Aware Coding Agent V10" (public badge 0.22, prompt base)
  → KOFI v11: four bounded, evidence-documented prompt splices
     (E03 issue-contract + E05 final-edit verification + coverage checkpoint
      + repo-specific verification focus, per prvsiyan/cubres public evidence)
"""
import hashlib
import io
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "submission.zip")

RUNTIME_FILES = [
    "agent.yaml",
    "configs/sampling.yaml",
    "eval_config.yaml",
    "prompts/system.md",
]


def build() -> bytes:
    import yaml  # safe loader with !include resolution like the harness

    class IncludeLoader(yaml.SafeLoader):
        pass

    files = {}
    for rel in RUNTIME_FILES:
        p = os.path.join(HERE, rel)
        files[rel] = open(p, encoding="utf-8").read()

    def include(loader, node):
        rel = loader.construct_scalar(node)
        assert rel in files, rel
        return files[rel] if rel.endswith(".md") else yaml.safe_load(files[rel])

    IncludeLoader.add_constructor("!include", include)

    # ── structural validation (mirrors the organizer's checks) ──────────────
    agent = yaml.load(files["agent.yaml"], Loader=IncludeLoader)
    assert agent["model"] == "gemma-4-31b-it-qat-w4a16-ct", agent["model"]
    assert "instruction" in agent and "gemma-4-31b" not in files["prompts/system.md"][:200].lower()
    sampling = yaml.safe_load(files["configs/sampling.yaml"])
    assert sampling["temperature"] == 0.2, sampling["temperature"]
    assert sampling["thinking_config"]["thinking_budget"] == 4096
    evaluation = yaml.safe_load(files["eval_config.yaml"])["evaluation"]
    assert evaluation["max_tool_calls"] == 28
    assert evaluation["max_time_minutes"] == 8
    assert evaluation["max_turns"] == 80
    tools = agent["tools"]
    assert isinstance(tools, list) and len(tools) >= 6, tools
    prompt = files["prompts/system.md"]
    for must in [
        "issue contract",           # E03 splice
        "Coverage checkpoint",       # bounded parallel-impl check
        "re-run the most targeted check",  # E05 splice
        "Repo-specific focus",       # corpus-audit hints
        "submit_patch",              # graded submission discipline
        "28 tool calls and 8 minutes",
    ]:
        assert must.lower() in prompt.lower(), f"prompt missing: {must}"

    # ── deterministic archive ───────────────────────────────────────────────
    # ZipInfo.date_time is pinned so the container bytes (and therefore the
    # SHA-256) are reproducible: writestr() with a bare name stamps the
    # current wall clock, which made two builds of identical content hash
    # differently (caught in Zymatica's review 2026-10-19).
    FIXED_DT = (1980, 1, 1, 0, 0, 0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def write_member(rel: str, content: str) -> None:
            zi = zipfile.ZipInfo(rel, date_time=FIXED_DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, content)

        for rel in RUNTIME_FILES:
            write_member(rel, files[rel])
        write_member("LICENSE.txt", open(os.path.join(HERE, "LICENSE.txt"), encoding="utf-8").read())
        write_member("NOTICE.txt", open(os.path.join(HERE, "NOTICE.txt"), encoding="utf-8").read())
    data = buf.getvalue()
    with open(OUT, "wb") as f:
        f.write(data)
    print(f"Submission: {OUT}")
    print(f"SHA-256: {hashlib.sha256(data).hexdigest()}")
    print(f"Size: {len(data)} bytes")
    print(f"Members: {zipfile.ZipFile(OUT).namelist()}")
    print("ALL STRUCTURAL CHECKS PASS")
    return data


if __name__ == "__main__":
    build()

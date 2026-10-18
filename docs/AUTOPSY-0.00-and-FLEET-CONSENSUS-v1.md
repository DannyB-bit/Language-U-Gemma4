# Gemma 4 Developer Agent Competition — 0.00 Autopsy & FLEET-CONSENSUS-v1

**Date:** 2026-10-18 · **Competition:** Google — The Gemma 4 Developer Agent Competition (Kaggle) · **Fleet:** SolanaNFT.art agent fleet (AMARA 07, KOFI 06, JAE 08, JULIAN 09, SHE)

---

## 1. Executive Summary

Three submissions built from this repo's machinery all returned **publicScore 0.00** despite
`SubmissionStatus.SUCCEEDED`:

| Date (UTC)       | Description                                                                                          | Score |
|------------------|------------------------------------------------------------------------------------------------------|-------|
| 2026-10-15 14:54 | Consider-Trinity-Gemma4-E4B: S^7 Octonionic RRF Locator + AST Verifier                               | 0.00  |
| 2026-10-16 00:00 | Consider-Trinity-Gemma4-E4B: ADK Declarative SWE Agent with AST Graph & 256-D Code Intelligence      | 0.00  |
| 2026-10-17 18:59 | Consider-Trinity-Gemma4-31B: Language-U 410-Proj Census, S^7 RRF Locator, AST Tracer & SCA/BAI/SBS Gates | 0.00  |

**Root cause (fleet-verified):** the submissions packaged Language-U / Consider-Trinity
machinery — glyph codecs, S⁷ projections, AST-graph services, custom Python entry points —
that **cannot exist inside Kaggle's sealed evaluation sandbox**. The harness compiles a
*declarative* `agent.yaml` into an ADK agent served by **Kaggle's own Gemma-4 QAT weights**
with a **fixed set of predefined tools**. Any plan that assumes our custom runtime, our
imports, or our model weights executes to nothing: the agent is structurally dead on arrival,
resolves zero SWE tasks, and scores 0.00. "SUCCEEDED" only means the zip was accepted and the
harness ran — not that our agent ever acted.

## 2. The Harness Contract (verified from the competition's HARNESS_README + starter kit)

- **Submission = a zip of declarative config**, not code we run: `agent.yaml` at root plus
  `prompts/`, `configs/`, optional `skills/` and LoRA assets. Limit 3 GiB.
- **Single model, served by Kaggle**: e.g. `gemma-4-31b-it-qat-w4a16-ct`. We do not ship weights
  or inference code.
- **Predefined tool set only** (9 tools): `run_command`, `read_file`, `edit_file`, `write_file`,
  `get_status`, `submit_patch`, `get_code_neighbors`, `search_similar_code`,
  `get_code_subgraph`. Custom tools/services cannot be registered.
- **SWE-Bench-style PASS/FAIL scoring** per task; resolution rate = public score.
- ~32K context; evaluation budget on the order of 12 h total; per-task budgets set through
  `eval_config.yaml`.
- **LoRA adapters currently have no effect** (organizer-confirmed harness bug) — tuning effort
  belongs in the prompt + sampling config, not adapters.

## 3. Why this repo's machinery cannot score — file-by-file

| Repo path | What it is | Why it dies in the sandbox |
|---|---|---|
| `agent/submission.zip` | the packaged 0.00 build | embeds the machinery below; no valid declarative fallback |
| `compiler/glyph_lang.py`, `compiler/glyphs.py` | Language-U glyph macro compiler | custom runtime never imported by the harness; dead code |
| `compression/compression_engine.py`, `compression/language_u_codec.py`, `compression/language_u_embeddinggemma_adapter.py` | weight/semantic compression codecs | assume local weights + our Python env; sandbox serves only Kaggle's QAT model |
| `gemma4_e4b/consider_trinity_e4b_core.py`, `gemma4_e4b/encode_gemma4_e4b_to_glyphs.py` | Consider-Trinity E4B core + glyph encoder | same — no execution surface, no import path |
| `model/gemma4_e4b_sample_layer0_q_proj.glyph.bin` | glyph-encoded weight sample | weights are irrelevant: we never serve a model |

The pattern: every one of the three failed submissions was **runtime-first** (our engine, our
tools, our weights) while the competition is **declaration-first** (their engine, their tools,
their weights; we supply identity, instructions, sampling, and budgets only).

## 4. FLEET-CONSENSUS-v1 — the design that CAN score

Consensus reached across the fleet (harness decoded by JAE; baselines + LB distribution
analyzed by KOFI and AMARA; packet built by JULIAN; byte-level verification + submission by the
fleet; see §6). The packet in `fleet-consensus-v1/` is **ADK-native and minimal** — every byte
maps to a harness-executable surface:

- **`agent.yaml`** — name `swe_agent`; model `gemma-4-31b-it-qat-w4a16-ct`; all 9 predefined
  tools declared; `instruction: !include prompts/system.md`;
  `generate_content_config: !include configs/sampling.yaml`.
- **`configs/sampling.yaml`** — temperature 0.2, top_p 0.95, max_output_tokens 8192,
  thinking budget 4096 with thoughts enabled: low-variance, patch-disciplined decoding.
- **`eval_config.yaml`** — per-task budgets: 240 s timeout, max 28 tool calls, 8 min/task,
  80 turns — budget-aware so long tasks fail fast instead of burning the clock.
- **`prompts/system.md`** — battle-tested SWE-Bench system prompt (from the public
  LB-leading baseline structure) plus the fleet's 8 edit-discipline rules: reproduce → locate →
  minimal-diff patch → verify → submit; no scope creep, no unverified edits, mandatory
  `submit_patch` before budget exhaustion.

The prompt + budgets are where the achievable points live, because (per §2) the model, tools,
and sandbox are fixed by the harness.

## 5. Honest ceiling analysis

- **Public LB top = 0.24** (Yurnero; the `honghanhhh` public baseline also sits at 0.24).
  No team on the board is near 0.50 — a 0.50+ public score is not supported by any evidence
  and should not be promised by anyone.
- Realistic consensus target: **escape 0.00 and land in the top 0.10–0.24 band**. The packet
  is structurally identical in shape to the 0.24 baseline (same model, same 9 tools, same
  sampling family) with fleet edit-discipline added, so the prior on a non-zero score is high.
- The 3-study public quantization and the LoRA no-op bug both cap what any config can show
  publicly this week; further gains come from prompt iterations scored against the daily
  submission cadence.

## 6. Submission record

- **2026-10-18 19:14:17 UTC — FLEET-CONSENSUS-v1 submitted** (Kaggle ref `57366201`):
  "Gemma 4 SWE Agent — fleet consensus v1: ADK-native agent (gemma-4-31b-it-qat-w4a16-ct),
  battle-tested SWE-bench system prompt, budget-aware eval_config (28 calls/8min/task, 80
  turns)." Status at time of this writing: PENDING.
- The packet shipped in this PR (`fleet-consensus-v1/`) is byte-identical to the kernel-built
  zip verified against the harness contract before submission (4 files, 8,403 bytes
  uncompressed: `agent.yaml` 314 B, `configs/sampling.yaml` 119 B, `eval_config.yaml` 94 B,
  `prompts/system.md` 7,876 B; `agent.yaml` at zip root).

## 7. What happens next

1. Score for ref 57366201 lands (watcher armed) → recorded here.
2. If non-zero: iterate prompt/sampling within the consensus frame; each daily submission is
   one measured experiment. If 0.00 persists despite the contract-clean packet, the failure
   is inside the harness side and the discussion forum + starter-kit local evaluator
   (swegemma Evaluator, published tasks.jsonl) become the debugging surface.
3. Language-U machinery stays in this repo as research (its compression/compiler tracks are
   unaffected) but is **retired from competition submissions** — the harness cannot run it.

— Fleet forensics: AMARA (Agent 07), with JAE (08), KOFI (06), JULIAN (09), SHE.

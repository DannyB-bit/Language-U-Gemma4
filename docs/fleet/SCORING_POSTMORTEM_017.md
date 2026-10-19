# Gemma-4 Developer Agent — 0.17 Post-Mortem & v2 Improvement Dossier

**Fleet:** SolanaNFT.art (JAE / Agent 08) · **Date:** 2026-10-19 · **Status:** analysis + v2 candidate staged — **no submission fired** (per operator order)

## 1. What we scored and why

Our submission 57366201 ("fleet consensus v1") scored **0.17**. Three prior submissions scored 0.00 (Language-U/S⁷ machinery cannot execute in the ADK sandbox — already diagnosed and fixed by the fleet pivot).

**The decisive forensic finding:** our 0.17 bundle is **byte-identical in all 4 files** to the public 0.24 baseline (`gemma-4-baseline-lb-0-24`, verified by extracting the notebook's embedded FILES dict and diffing):

| file | ours (0.17) | baseline (0.24) |
|---|---|---|
| agent.yaml | 314 B | IDENTICAL |
| configs/sampling.yaml | 119 B | IDENTICAL |
| eval_config.yaml | 94 B | IDENTICAL |
| prompts/system.md | 6,963 B | IDENTICAL |

**Conclusion:** 0.17 vs 0.24 is **evaluation noise, not a config gap**. The same recipe class has publicly scored 0.12 / 0.10 / 0.06 on byte-identical reruns. The scoring pipeline re-runs the agent (temp > 0 sampling) against ~120 hidden SWE-bench-style tasks; run-to-run variance is intrinsic. Our config was already the top public recipe — shuffling configs cannot climb the band. **Only raising the agent's true pass-rate moves the needle.**

## 2. The measured levers (competitor ablations on the public 129-task evaluator)

From the public experiment thread (discussion/746250), all numbers measured on the same evaluator:

| Change | Result vs base (48/129 = 37.2%) | Verdict |
|---|---|---|
| Temperature 0.2 → **1.0** | **51/129 = 39.5%** | take (+2.3 pts) |
| Thinking budget 2048 / 6144 | 38.8% / 37.2% | keep 4096 |
| Thinking disabled | 23/129 = 17.8% | catastrophic — never |
| Thought summaries off | 27.1% | catastrophic — keep ON |
| Time cap 6/10/**15**/20/60 min | 28.7% / 35.7% / **41.9%** / 38.0% / 39.5% | **15 min is the optimum** (timeouts 61 → 3) |
| Tool calls 25 / **50** | 27.9% / **42.6%** | **50 calls** |

## 3. Harness-README levers (from the harness bible itself)

- **Gotcha #1 (explicit):** thinking_budget 4096 needs `max_output_tokens: 16384` — big thought + big edit at 8192 truncates the `<|tool_call>` tag mid-call (wasted turn). **Our v1 shipped 8192 — a real defect.**
- **submit_patch() and get_status() are FREE** (`count_tool_call=False`) — the prompt should say so and use them.
- Patch auto-recovery exists (unsubmitted diffs are harvested), but test-file edits are always stripped — fix library code only.
- Phase-2 resolution is strict: pytest exit 0, zero failures/errors, no skipped required tests.

## 4. Known harness defect (prompt-mitigatable)

vLLM drops `reasoning_content` between tool calls (discussion/744354): the model's thoughts do not persist in history. Mitigation: instruct the agent to **restate goal/progress in plain text before each tool call** (externalized working memory).

## 5. The v2 change set (every line evidence-backed)

| file | v1 (0.17) | v2 | evidence |
|---|---|---|---|
| sampling.yaml | temp 0.2 | **temp 1.0** | E12: +2.3 pts |
| sampling.yaml | max_output 8192 | **16384** | README gotcha #1 |
| sampling.yaml | thinking 4096 + summaries on | keep | E17/E18: disabling = −19 pts |
| eval_config.yaml | 28 calls | **50** | E30: +3.9 pts vs 25 |
| eval_config.yaml | 8 min | **15 min** | E22: +4.7 pts vs 10, timeouts 61→3 |
| eval_config.yaml | timeout 240s | **300s** | harness default; slow-pytest headroom |
| eval_config.yaml | 80 turns | **150** | safety margin at 50 calls (harness default 500) |
| system.md | call plan for 28 calls | **re-scaled to 50**, free-tool note, thought-externalization rule | E30 + README + defect report |
| agent.yaml | unchanged | **unchanged** | the 0.24 chassis — don't touch what works |

Honest expectation: v1's config sits in the ~37% public-evaluator band; the v2 change set moves us into the measured **41–43% band** (E22/E30 class). On the LB that is the 0.22–0.24 territory. **0.75–0.99 does not exist on this board** — the top of 3,449 teams after 3 months is 0.24; the fleet's goal translated honestly: maximize true pass-rate, land at the top of the band, and let the two-select-finals rule keep our best reruns.

## 6. The ceiling-breakers (next rungs, not in v2 — separate decisions)

1. **Wheelhouse simulation before any submission:** `swegemma eval` runs the 129 public tasks locally/on a Kaggle GPU kernel — v1 vs v2 measured BEFORE burning a daily slot. Strongly recommended as the gate for every future candidate. (A 30–40 task shard gives a ±8% signal in one GPU session.)
2. **ADK `skills/` directories** ship domain knowledge + scripts INSIDE the submission (schema-sanctioned): per-repo recipes (fastapi docs_src layout, rich/requires test patterns) the agent can read. Most of the 0.22 band doesn't use this.
3. **LoRA adapters (`adapters/`)** — a genuinely fine-tuned SWE LoRA is the only structural edge left (the 0.00-era Language-U adapters were mathematically dead — all-zero lora_B). Training one is a GPU-kernel program, not a tablet program; fleet decision required.
4. **AgentTool read-only sub-agent** for exploration (README best practice #4) — trades context efficiency for failure-mode risk; only with simulation evidence.

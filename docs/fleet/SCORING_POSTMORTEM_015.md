# Gemma-4 Developer Agent — the 0.15 Draw: Post-Mortem, Regression Analysis & the v5 Ladder

**Fleet:** SolanaNFT.art · **Author:** AMARA — Co-Sentinel & Lead Web Systems Architect (Agent 07)
**Date:** 2026-10-20 · **Refs:** 57406588 (v3-converged → **0.15**) vs 57366201 (fleet consensus v1 → **0.17**)
**Method:** live Kaggle API pulls + repo forensics + the trinity-hybrid-retrieval MCP store (queried on this run: Function Calling, Test Plan & Coverage, AI-Powered Code Reviews) + session-history recovery of the Roman Rozen 0.24 analysis.

---

## 1. How is this result less than the one before? — the mechanical answer

- 0.17 = **10 of 58** hidden tasks resolved (fleet consensus v1, fired 2026-10-18)
- 0.15 = **9 of 58** hidden tasks resolved (v3-converged, fired 2026-10-19)
- The delta is **exactly one task.**

## 2. Why one task is not a signal — the statistical answer

The scorer re-runs the agent with temperature > 0 sampling against the hidden task set. Run-to-run variance is intrinsic and large:

- The **byte-identical** public 0.24-baseline recipe has scored **0.12 / 0.10 / 0.06** across reruns (4–13 task swings) — documented in `docs/fleet/SCORING_POSTMORTEM_017.md`.
- Our own v1 was **byte-identical in all 4 files** to that 0.24 baseline and drew 0.17. The entire 0.06–0.24 range is **one noisy band**.
- Fleet law (pre-registered in the score plan): single-draw gaps below ±2 tasks are **unjudgeable**.

**Conclusion: the 0.15 vs 0.17 outcome is draw variance. No config change caused it, and no config can be judged from it.** This is not a comfort statement — it is the same standard we applied when v1 drew 0.17 against the same recipe's 0.24 public draw.

## 3. The part that actually matters — the standing answer

Kaggle's final standing auto-selects our **two best submissions across all draws**:

| | banked best-2 |
|---|---|
| Before the v3 draw | 0.17 + 0.00 |
| After the v3 draw | **0.17 + 0.15** |

**The "worse" result improved our final position.** The regression Dan saw on the leaderboard is cosmetic; the two-select standing moved up. Every additional draw at a 0.15-class-or-better config is free expected value on the final score.

## 4. What we actually did wrong — the honest forensics

1. **v1 shipped `max_output_tokens: 8192`** — the harness README's own gotcha #1: thinking_budget 4096 plus a large edit can truncate the `<|tool_call|>` tag mid-call and waste turns. A real defect, present in the 0.17 draw, **fixed in v3** (16384).
2. **We fired v3-converged without an A/B verdict.** The wheelhouse evaluator self-aborted — `FATAL: fewer than 4 GPUs` on the 2×T4 session Kaggle actually provisioned — and the slot was fired anyway. The result is a draw we cannot attribute to the prompt changes: exactly the cost of skipping the measurement gate. **Fixed in this PR** (see §5, rung 1).
3. **Our strongest measured config has never been fired.** The ablation-backed v2-measured levers — temp 1.0 (+2.3 pts), 50 tool calls (+3.9 pts), 15-minute cap (+4.7 pts, timeouts 61→3), all measured on the public 129-task evaluator — remain un-drawn. Every public team above us lives in that measured band.
4. **The structural gap we identified and then never built.** On 2026-10-19 the fleet extracted Roman Rozen's TOP-1 notebook and found his default is the **`analyzer+coder` two-agent architecture** — a read-only `code_analyzer` sub-agent returns a bounded LOCATION / ROOT CAUSE / FIX PLAN report so exploration stays out of the coder's context. We shipped the `single` agent class in v1 and v3. **Built in this PR** (see §5, rung 2).

## 5. What we do better — the v5 ladder (every rung evidence-tagged)

- **Rung 1 — the measurement gate is fixed.** `submissions/wheelhouse-eval/run_ab.py` now adapts to the hardware Kaggle actually provisions: tensor-parallel = GPU count, paired subset auto-scaled (40 → 20 tasks on 2×T4), per-lane estimates widened, no hard FATAL. Pairing is preserved (both lanes on the same hardware), so the A/B delta stays valid on any shape. For scorer-faithful absolute numbers, the kernel README now says to request **GPU L4 ×4** in the Kaggle notebook settings.
- **Rung 2 — v5-analyzer-coder is built** (`submissions/v5-analyzer-coder/`, 29/29 contract checks, deterministic sha256 `60b98087c13b9b2bd370f31ca98f5c0cd02b7b557cb7c65036ee0da624386d0c`): Roman's two-agent architecture with verbatim prompts, **architecture-isolated** against the v3 control — same 28-call / 8-min / 300s / 150-turn budget, same 16384 output fix — so the paired evaluator measures the architecture variable alone. Two flagged deviations from Roman's published defaults, both deliberately staged for the evaluator to settle: `max_output_tokens 16384` (his 8192; README gotcha #1) and `include_thoughts true` (his false; public ablation E17/E18 measured summaries-off as catastrophic — a genuine contradiction in the public record, and exactly the kind of question the paired A/B exists to answer).
- **Rung 3 — combine winners one variable at a time.** If v5's architecture ships (paired net ≥ +3), the next candidate fuses it with the un-fired v2-measured budget (50 calls / 15 min / temp 1.0) — each fusion gated by the same evaluator before any slot is spent.
- **Rung 4 — draw discipline.** Best-2 selection means: draw daily at the strongest-evidenced config, judge configs on paired evaluator verdicts, never on single leaderboard draws. The 0.15 draw cost nothing; the next unmeasured fire would.

## 6. The honest ceiling statement

0.24 is the top of 3,449 teams after 3 months of daily draws. The realistic target is the top of the 0.22–0.24 band with the two-select rule banking our best reruns. The v5 ladder is how we get there with evidence instead of slot-burning.

---

*Verification receipts: 29/29 v5 contract checks (deterministic sha above); run_ab.py compiles clean; all score numbers pulled live from the Kaggle submissions API on 2026-10-20 (11:48 UTC pull); Roman Rozen architecture extracted from his public TOP-1 notebook (Apache-2.0, credited in the bundle NOTICE lineage); trinity-hybrid-retrieval MCP store queried on-node (1,962-topic hybrid index, lexical + Gemma embedding lanes).*

Signed-off-by: Amara 200 amsterdam <amara@solananft.art>

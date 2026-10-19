# Language-U-Gemma4 Agent v11 — Score Analysis & Improvement Study

**KOFI (Agent 01), 2026-10-19. Evidence-based. No scores claimed that don't exist.**

## 1. Our verified score: 0.17 (fleet consensus v1)

Submission 57366201, 2026-10-18 19:14:17 UTC, `COMPLETE`, public score **0.17**.
That bundle was the byte-exact reproduction of the community's best public
recipe family (single ADK agent, `gemma-4-31b-it-qat-w4a16-ct`, temperature
0.2, thinking budget 4096, 28 calls / 8 min / 80 turns, battle-tested
SWE-bench prompt). It replaced the three earlier **0.00** submissions
(refs 57322690, 57247159, 57234680) which packaged custom Language-U
machinery (S^7 locators, AST gates, sub-agents, skills) that the sandbox
cannot execute, plus a weaker E4B model and misconfigured budgets.

**Root causes of the 0.00s (fixed in v1, kept fixed in v11):**
- Custom skills/sub-agents the evaluation harness never invokes.
- `gemma-4-e4b-it` instead of the required 31B model.
- Temperature 0.7 (creativity is poison for exact-name SWE repairs).
- Budget misconfiguration blowing the 12-hour wall clock.

## 2. What the score actually is (mechanics)

- Public LB = **58 tasks**. One solved task = **+0.017**. Our 0.17 = **10/58**;
  the current top (Yurnero, Yike Xu) = 0.24 = **14/58**.
- The hidden benchmark has ~120 tasks from **private repositories**, split
  across public/private LBs; the final score auto-selects the best 2 submissions.
- **The noise is huge**: community-verified evidence shows *byte-identical*
  archives scoring 4, 6, 7, 8, 9, 11, 12 and 13 tasks across reruns — up to a
  6-task swing (≈0.10) with zero file changes. Single-draw comparisons below
  ~2 tasks are meaningless.
- Every top public recipe is a **4-file bundle** (agent.yaml, sampling.yaml,
  eval_config.yaml, prompts/system.md). No public evidence that skills,
  sub-agents, LoRA adapters or custom tools add anything. The winning family
  is prompt + budget discipline on the fixed 31B model.

## 3. The v11 recipe: measured base + four bounded, evidence-backed splices

**Base: flexonafft "Budget-Aware Coding Agent V10" — public badge 0.22.**
Its prompt is our v1 prompt minus three rules that V10's author removed
(the /tmp scratch rule — documented as a live failure mode in public traces;
the unbounded parallel-implementation rule; `git status` before diff). Its
badge (0.22 = 13/58) is the best *pullable, reproducible* recipe observed.

**Four additions, each with public experimental evidence behind it:**

| # | Addition | Evidence | Bound |
|---|----------|----------|-------|
| a | **Issue contract** — 50 words stated with the first tool call: triggering condition, required outcome, one behaviour to preserve, exact names, unknowns marked | prvsiyan E03: scored 0.17 vs their E01 0.18 baseline (noise-equivalent, but a sound discipline that costs nothing) | 50 words, same action as first call |
| b | **Coverage checkpoint** — after the first successful edit, check for an obvious parallel implementation (async twin, Client pair, overload, re-export) | cubres BFV5 appendix; recovers the parallel-impl awareness V10 dropped, in bounded form | ≤2 calls (one 20-line grep, one 80-line read), only if ≥4 calls remain |
| c | **Final-edit verification** — re-run the targeted check if anything changed after the last check | prvsiyan E05 hypothesis: an earlier passing test does not verify a later patch | 1 call, only if needed |
| d | **Repo-specific verification focus** — FastAPI: schema/type boundaries; Rich: captured rendering/width; Requests: offline `.prepare()`, never DNS | prvsiyan public-corpus audit: 96/129 tasks are FastAPI+Rich; DNS stalls and missing snapshot deps are documented verification failure modes | prompt guidance only |

Everything else is **byte-identical to V10**: model, sampling (temp 0.2,
4096 thinking), 28 calls / 8 min / 80 turns, tools, call plan, edit and
recovery discipline, decision rules.

## 4. Honest expectations

- The v1 (0.17) and V10 (0.22) badges are draws from the same distribution
  (both inside the 7-13-task identical-file noise band). v11's splices are
  individually small; their combined effect can only be judged across
  **multiple draws**, and we get exactly one submission per day.
- Single-draw expectation: the 0.12-0.24 band, same as every top recipe.
  **No public recipe has demonstrated more than 14/58.** A 0.75+ score would
  require solving 44+/58 — 3× the best public result — and no mechanism
  demonstrated publicly gets close. We will not claim a number we cannot
  evidence; we maximize the measured recipe and let the private-LB draw land.
- The two competition levers that are real: (1) submit the strongest recipe
  daily so the auto-selector sees many draws, (2) never regress the four
  hard-won rules that took us from 0.00 to live points.

## 5. Build & verify (offline, deterministic, no submission)

```bash
cd agent && python3 build.py
# → SHA-256 fb461a4e14a70fbefd2eb6522e2b3c1d0b2dabeebe23c4fa5af994df90125a3a
#   (timestamp-pinned container since the Zymatica review PR; the six member
#    files are byte-identical to the reviewed v11 content)
# → ALL STRUCTURAL CHECKS PASS
```

The builder validates: 31B model, temp 0.2, thinking 4096, 28/8/80 budgets,
tool declarations, all four splice markers, and packaging — then writes the
6-member archive (4 runtime files + LICENSE + NOTICE with full Apache-2.0
attribution to verracodeguacas, flexonafft and cubres).

**This study submits nothing. The bundle awaits fleet review + the
operator's go.**

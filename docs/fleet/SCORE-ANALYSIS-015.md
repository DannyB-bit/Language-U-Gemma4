# Score Analysis — the 0.15 draw (ref 57406588) vs the banked 0.17

**Zymatica (Agent 02), 2026-10-20. MCP-grounded via trinity-hybrid-retrieval
(hybrid search → Kameleoon A/B-testing doctrine, STRONG 0.3645). Evidence-first.**

## 1. The direct answer: why is this result less than the one before?

**Statistically, it is not less — it is one task.** 0.15 = 9/58 resolved; 0.17 = 10/58.
The documented rerun noise on *byte-identical* archives in this harness is 4–13 tasks
(0.06–0.24). A 1-task delta cannot distinguish the two bundles' true pass-rates —
the same recipe publicly scored 0.12/0.10/0.06 across reruns. The honest statement:
**v1 and v3 draws are indistinguishable; both sit mid-band of the same distribution.**
Best-2 auto-selection keeps the 0.17 regardless.

## 2. What we actually did wrong (the process defect, found by execution)

The failure is not the prompt — it is **the fire happened without measurement**:

1. **The pre-registered gate law was bypassed.** The repo's own SCORE-MAXIMIZATION-PLAN
   (§3) says: *"Nothing ships unmeasured while the A/B engine exists."* The A/B engine
   (submissions/wheelhouse-eval/) was pushed the same evening — and v3 fired before it
   produced any data.
2. **The instrument ran and self-aborted correctly.** Live receipts (kernel output,
   pulled 2026-10-20): the A/B kernel completed on **T4×2** and hit its own GPU-shape
   gate — `FATAL: fewer than 4 GPUs; not the scorer shape. Stopping.` The gate is
   correct (a T4×2 verdict at ~12 tok/s would not transfer to the scorer's 4×L4);
   the process failure was that **nobody verified the kernel could get an L4×4
   session before the fire window depended on it.** Pushed kernels on this account
   tier land T4×2 by default; L4×4 requires the interactive notebook path.
3. **A/B doctrine violation (the MCP grounding):** Kameleoon's guide — the store's
   STRONG node for experiment analysis — defines A/B testing as comparing two
   versions to determine which performs better, with significance before shipping.
   We compared D/D (draw vs draw) on **confounded variables** (v1→v3 changed prompt
   AND three config knobs) with no controlled pair, then read a 1-task delta as a
   regression. That is the exact anti-pattern the doctrine prohibits.

## 3. What we do better — the four fixes (all shipped in this PR)

**Fix A — the measurement-first enforcement law (process):**
No Gemma-4 submission fires without a `paired_ab.json` covering its exact bundle
bytes vs the incumbent, on the L4×4 shape. The decision rule stays pre-registered:
net ≥ +3 tasks on the paired set → ship the candidate; ±2 → keep incumbent and
re-draw; ≤ −2 → discard. Until the A/B produces data, the only sanctioned daily
play is a re-draw of the best-evidenced bytes (free variance under best-2 selection).

**Fix B — the L4×4 execution law (instrument):**
Documented in the wheelhouse README: pushed script-kernels land T4×2 on this
account tier. The A/B must be run via the **interactive notebook session with
L4×4 selected** (the competition's own getting-started notebook runs L4×4 — the
shape is available; it is a session-start choice, not a push parameter). This PR
also re-pushes the kernel — if the queue assigns L4×4 this time it self-starts;
otherwise the interactive path is the documented fallback. The kernel aborts
harmlessly on any wrong shape (already proven live).

**Fix C — the candidate discipline (v5-restored converges this):**
Until measurement says otherwise, the strongest-evidenced single-draw config is
the 0.17-drawing prompt bytes + only the zero-coupling 16384 truncation fix —
exactly JAE's v5-restored lane (sha-gated to the original prompt). The levers
that remain unmeasured (loop-guards, budget climb, graph-tool unban, 12B variant)
enter only through the A/B matrix, never straight into a slot.

**Fix D — the capability edges (where real points can come from):**
- **Graph-tool unban (JULIAN's lever):** every public recipe — ours included —
  byte-inherits a ban on the three code-graph tools. The harness README says they
  work offline (precomputed embeddings ship with the competition data). Bounded
  use is a capability edge no one on the board has tested publicly.
- **The 12B variant:** organizers confirmed the scorer already serves
  `gemma-4-12b-it-qat-w4a16-ct` (~67 vs ~35 tok/s → roughly 2× test-time compute
  inside the same wall clock). Public wheelhouse wheels break on it locally;
  the scorer-side is patched. It must clear the A/B first — but it is the only
  lever with headroom beyond ±2 tasks.

## 4. The forward schedule (deadline Dec 2 — ~43 daily slots remain)

1. A/B on L4×4 (interactive session) → `paired_ab.json` for: v5-restored (control)
   vs v4-loop-guard vs v2-measured-configs vs graph-tool-unban.
2. Daily draws continue meanwhile on the best-evidenced bytes (operator's word only).
3. Every claimed improvement ships with its paired evidence; single-draw deltas <2
   tasks are noise and never reported as regressions or wins again.

— Zymatica (Agent 02), with the fleet's lanes (JULIAN postmortem · KOFI noise tool ·
JAE v5-restored · AMARA lineage diff) converging in this document.

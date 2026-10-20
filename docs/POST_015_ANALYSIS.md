# The 0.15 Draw — Full Failure Analysis & the Corrected Protocol

**KOFI (Agent 01), 2026-10-20. Every number below is live-verified, none is invented.**

## 1. What happened (live API receipts)

| ref | bundle | date (UTC) | public score | tasks solved (×58) |
|-----|--------|-----------|--------------|--------------------|
| 57406588 | fleet v3-converged (v11 prompt + output 16384 + margins) | Oct 19 19:07 | **0.15** | **9** |
| 57366201 | fleet consensus v1 (lb-0.24 recipe clone, 8192 output) | Oct 18 19:14 | **0.17** | **10** |

The "drop" is **one task**. The 58-task public board quantizes at 0.017/task.

## 2. Why the result is lower — the three hypotheses, honestly ranked

**H1 (primary, evidence-weighted): draw noise.** The community's own measured
evidence — byte-identical archive bytes, zero code changes — re-scores between
**4 and 13 tasks** across reruns (dmitriigluzdov's ledger: identical files drew
4/6/7 and 7/9/11/13; prvsiyan's: 3–5 across seven official draws; the
budget-fit reference: 8–11 over seven runs). A 9-vs-10 delta between two
*different* same-family bundles is **statistically indistinguishable from
zero**. One draw cannot order them. This is not an excuse — it is the
competition's documented measurement law.

**H2 (real mechanism, unverifiable at n=1): the 16384 output-cap interaction.**
With `include_thoughts: true`, thinking tokens count toward output. v1's
tighter 8192 cap acted as an *accidental per-turn throttle* — a thinking-heavy
turn could never run away. v3's 16384 (the harness bible's own value, the
fleet-verified truncation fix) lets a single turn think up to 2× longer —
if the model over-thinks early turns, fewer calls fit inside the 8-minute
session clock. The truncation defect (v1's risk) and the throttle benefit
(opposite sign) are BOTH real; one draw cannot tell which dominated.

**H3 (possible, smaller): the 4 prompt splices** (issue contract, coverage
checkpoint, final-edit re-verification, repo-specific focus). Each is
individually bounded and evidence-cited, but they were shipped together with
the config deltas in a single draw — the composite effect is unmeasured.

## 3. What we did wrong (the honest process audit)

1. **We spent a live slot on unmeasured deltas.** v3's changes were verified
   *structurally* (21/21 checks, byte-reproducible) but never *behaviorally*
   (no wheelhouse A/B run) — the fleet built the measurement engine and then
   didn't run it before firing. That's the exact failure mode our own gate
   law was written to prevent.
2. **We compared single draws as if they were signals.** Public evidence says
   ±3–6 tasks of noise on identical files; our delta is 1 task. Treating 0.15
   vs 0.17 as "v3 is worse" repeats the same statistical error in reverse.
3. **One shot per day means the slot IS the experiment.** A slot spent on a
   config delta inside the noise band is a slot not spent accumulating a
   best-2 finals draw.

## 4. What we do better — the corrected protocol (evidence-ranked levers)

**The draw law:** the auto-selection takes the best 2 of ALL submissions, so
the optimal play is a strong reference fired daily (each draw is a free sample
of the noise distribution's upper tail) while every *candidate* change goes
through the wheelhouse A/B (zero slots, scorer-identical stack, pre-registered
rule: net ≥ +3 tasks ships, ±2 keeps incumbent, ≤ −2 discards).

**Lever table (ranked by measured evidence):**

| lever | evidence | slot cost | gate |
|-------|----------|-----------|------|
| Daily draws of the strongest reference | best-2 accumulation is pure upside | 1/day | none — it IS the play |
| Jae's v2 budget profile (temp 1.0, 50 calls, 15-min) | ablation: session timeouts 61→3; schema-audited bust-free | 1 slot after wheelhouse | wheelhouse first |
| Graph-tools bounded splice (the recipe-banned offline .npz locators) | harness bible says they work offline; 100+ embedding files shipped; the whole top band copies the same ban | 1 slot after wheelhouse | wheelhouse first |
| Loop-guards (v4, Zymatica) | public ledger: 27% of calls exact repeats; loop-fix family cut looping 3→0, first-edit call 33→18 | 1 slot after wheelhouse | wheelhouse first |
| 12B model lane | ~2× tokens/sec = 2× test-time compute | 1 slot | wheelhouse first (Julian's 12h audit applies) |

**Trinity MCP grounding (roadmap_hybrid_v4, live queries this session):**
- *Monitoring* (score 0.3135): metrics/alerting discipline → the wheelhouse IS
  our Prometheus; every candidate must clear a measured gate before touching a slot.
- *Function Calling* (0.3286): tool-contract design → the banned graph tools
  are the unexploited tool-contract surface.
- *CI/CD* (0.415): automated builds/tests before deploy → the gate law, applied
  to submissions: wheelhouse = CI, slot = production deploy.

## 5. Verdict for the next slot

Fire the strongest measured reference daily. v3 and v1 are the same family
within noise; the reference choice between them should come from ONE wheelhouse
run (v3 vs v1 vs v2, paired, seeded, pre-registered) — not from today's
one-task delta. Until that run exists, the reference stays v1 (the 0.17 draw
is the best single result we own).

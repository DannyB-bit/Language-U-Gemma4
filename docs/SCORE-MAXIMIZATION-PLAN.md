# Score Maximization Plan — 0.17 → maximum achievable

**Zymatica (Agent 02), 2026-10-19. Fleet-reviewer synthesis of every lane's evidence.
No scores claimed that don't exist. Nothing submits until the operator says go.**

## 1. Where we honestly stand

- Our verified public score: **0.17** (10 of 58 public-LB tasks; submission 57366201).
- Board top: **0.24** (14/58, Yurnero). Best public recipe family: 0.22 (flexonafft V10 — our v11 base).
- Noise: byte-identical archives re-score **4–13 tasks** across reruns. Single draws within ±2 tasks
  are unjudgeable; the final standing auto-selects our **best 2 submissions**, so every daily draw is
  free expected value.
- The hidden set: ~120 tasks from **private repositories** (not fastapi/rich/requests/httpx), 50/50
  public/private LB split, graded all-or-nothing (hidden `test_patch` must pass on a fresh checkout).

**The 0.75–0.99 question, answered with the board's own math:** 0.75 = 44/58 and
0.99 = 57/58 — that is 3–4× the best public result on Earth in this competition, on a
fixed 31B INT4 model, in ≤28 tool calls, on repos the model has never seen. No participant
of 3,449 has demonstrated a mechanism close to it; the entire top-15 lives in the 10–14
task band. We will not claim a number the evidence does not support — what we WILL do is
run every lever below through the measurement engine and ship the empirically strongest
recipe each day, letting the best-2 auto-selection accumulate the tail of the distribution.

## 2. The lever stack, ranked by evidence

| # | Lever | Evidence | Cost/risk | Status |
|---|---|---|---|---|
| 1 | **Daily draws of the best recipe** | best-2 auto-selection; noise band 4–13 | zero | awaiting operator go |
| 2 | **Loop-guard prompt splices (LG1/LG2/LG3)** | happyc0der ledger: verbatim loops up to 23 calls at temp 0.2; 27% of calls exact repeats; loop-fix family cut looping tasks 3→0, median first-edit 33→18 | prompt-only, bounded | **v4 built, 24/24 checks, sha 7e89e9c0…b4ef** |
| 3 | **A/B measurement before shipping** (the wheelhouse kit) | the only honest judge given the noise band | 1 kernel session (~free) | **built: `submissions/wheelhouse-eval/`** |
| 4 | max_output_tokens 16384 + 300s/150t margins | harness README gotcha #1 (thinking counts toward output cap → mid-tool-call truncation at 8192) | zero | **shipped in v3** (fleet-reviewed) |
| 5 | v11 evidence-backed splices (E03 contract, BFV5 checkpoint, E05 re-verify, repo hints) | prvsiyan/cubres public experiments | prompt-only, bounded | **shipped in v11/v3/v4** |
| 6 | temp 0.7 as a loop countermeasure | happyc0der: loops gone at 0.7; their day-11 real-LB 0.10 = same as temp 0.2 best | config-only | eval-matrix follow-up (v4-t07) |
| 7 | **12B variant (2× tok/s)** | organizers added 12B/E4B/E2B; scorer already patched to serve them; ~67 vs ~35 tok/s → more test-time compute per task | wheelhouse-compat risk (public vLLM wheel breaks on 12B locally; scorer-side confirmed working) | parked — measure on wheelhouse before any fire |
| 8 | Skills with sandbox scripts | submission format allows `skills/*/scripts/`; public evidence neutral (our own 0.00s shipped skills the harness never invoked) | risk of dead weight | parked |
| 9 | Sub-agents / analyzer | measured NEGATIVE (8/29 vs 9/29 baseline; +0 across 5 configs) | adds failure surface | rejected |
| 10 | LoRA adapters | organizer-confirmed no-op harness bug | wasted effort | rejected |

## 3. The pipeline (state → measure → ship)

```
submissions/
  v3-converged/     ← INCUMBENT (v11 prompt + config fixes)  sha 04cdcdfc…700e5e
  v4-loop-guard/    ← CANDIDATE (v3 + LG1/LG2/LG3)           sha 7e89e9c0…b4ef
  wheelhouse-eval/  ← the A/B kernel + pre-registered decision rule
```

1. Any sibling pushes the A/B kernel (README has the exact steps; needs KGAT + GPU session).
2. `paired_ab.json` verdict per the pre-registered rule: **net ≥ +3 → ship; ±2 → keep incumbent and
   spend the slot on a fresh draw; ≤ −2 → discard.**
3. Only an operator "go" fires a leaderboard submission; the fired bundle is the verdict winner.
4. Every new idea enters as a lane in the kernel's RUNS matrix — never straight into a submission.

## 4. The four hard-won laws (never regress)

1. Declarative-only: agent.yaml + prompts + configs; zero custom runtimes (the 0.00 lineage).
2. The measured recipe base (V10-family) changes only through bounded, evidenced splices.
3. Builders are deterministic (pinned ZipInfo) + coupling-guarded (prompt budget line == eval_config).
4. Nothing ships unmeasured while the A/B engine exists; nothing ships at all without the operator's word.

## 5. Expected trajectory (honest)

- Best case with levers 1–5: the top-band 0.17–0.24 window, hit more often and with a real
  mechanism edge (loop-conversion), not just variance.
- The 12B lane (7) is the only lever with headroom beyond +2–3 tasks — it trades model quality
  for ~2× test-time compute; it must clear the wheelhouse first.
- Anything above ~0.24 requires a mechanism nobody has published. If one appears in the
  discussion forum, it enters the matrix like everything else: measured, then shipped.

— Zymatica, with KOFI (v11 analysis), JAE (v3-converged + builder laws), fleet forensics
(AUTOPSY-0.00), and the public ledgers (happyc0der, prvsiyan/cubres, flexonafft, Jayaprakash).

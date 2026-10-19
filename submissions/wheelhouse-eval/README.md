# Wheelhouse A/B Evaluation Kit — the fleet's measurement engine

**This is a Kaggle GPU KERNEL, not a leaderboard submission.** It consumes zero
daily submission slots. Per the operator's 2026-10-19 order, nothing ships to the
leaderboard until he says go; every improvement is MEASURED here first.

## Why this exists

Our 0.17 (10/58) sits inside a documented noise band: byte-identical archives
re-score 4–13 tasks across reruns. No prompt/config change can be honestly
judged from a single daily draw. The 129 published wheelhouse tasks
(fastapi 67 / rich 48 / requests 13 / httpx 1) are the only ground truth we can
measure against cheaply — on the scorer's own hardware shape (4× L4) with the
organizers' own stack (wheelhouse vLLM 0.19.1 + swegemma Evaluator).

## How to run (any sibling with a Kaggle account + KGAT)

1. Push this kernel: `kaggle kernels push -p submissions/wheelhouse-eval/`
   (requires the wheelhouse dataset, competition data, and the 31B model
   attached — all declared in `kernel-metadata.json`).
2. Add a Kaggle **User Secret** named `github-token` (read-only PAT scoped to
   `DannyB-bit/Language-U-Gemma4`) — or attach the repo's `submissions/` dir as
   a private dataset instead.
3. Run on **GPU L4×4** (the scorer's shape; the kernel aborts on anything else).
   Session budget ≈ 11 h: control + one candidate fit comfortably.
4. Collect `/kaggle/working/paired_ab.json` + `results/<lane>/` — post the
   verdict table to the fleet.

## The pre-registered decision rule (agreed BEFORE any run)

Each lane runs the **same seeded 40-task subset** (`TASK_SEED = 20261019`,
`TASK_COUNT = 40`), paired per-task. The rule:

- **net ≥ +3 tasks** vs control on the paired set → **SHIP the candidate**
  (outside the ±1–2 task noise on 40 pairs).
- **net −2 … +2** → **INCONCLUSIVE** → keep the incumbent, spend the slot on a
  fresh draw instead (variance is free EV under best-2 auto-selection).
- **net < −2** → **DISCARD** the candidate.

Env-unstable tasks (older library pins; ~19 of 129 per public oracle runs) fail
identically in both lanes → excluded from signal automatically as ties.

Escalation: a full-107 paired run (all oracle-usable tasks) is a weekend
session (~8–10 h/lane); the 40-task subset is the daily-driver.

## Lanes (the matrix)

| Lane | Bundle | Delta vs control |
|---|---|---|
| `ctrl-v3` | `submissions/v3-converged` | — (incumbent: v11 prompt + 16384/300s/150t) |
| `cand-v4` | `submissions/v4-loop-guard` | 3 loop-guard prompt splices (LG1/LG2/LG3) |
| follow-up | `submissions/v4-t07` (to create) | LG set @ temperature 0.7 |

## Evidence base for the v4 guards (public, measured)

- happyc0der's public ledger (github.com/happyc0der/gemma-swe-agent):
  temp 0.2 produced verbatim retry loops of up to **23 identical calls**;
  ~27% of calls were exact repeats (v4.8 measurement); their loop-fix family
  (never-resend + progress-forcing + empty-means-absent) cut looping tasks
  3→0 and median first-edit call 33→18; their best prompt (v5.0n, same family)
  scored 0.10 on the real LB and 9/16 on their faithful eval.
- Jayaprakash's public writeup: single-agent baseline 9/29–11/30; sub-agents
  and thinking-off HURT; oracle 110/129.

## API provenance

Every harness call in `run_ab.py` is adapted 1:1 from happyc0der's working
`kaggle/eval_official_l4/run.py` (Apache-2.0), which itself mirrors the
organizers' Getting Started notebook flags (TP=4, max_model_len 32768,
tool_call/reasoning parser `gemma4`, enable_thinking chat-template kwarg).

— Zymatica (Agent 02), 2026-10-19

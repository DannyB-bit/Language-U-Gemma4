#!/usr/bin/env python3
"""Gemma 4 Developer Agent — score statistics for slot decisions.

Real logic, no mocks: converts public scores to solved-task counts on the
58-task board and flags whether a score delta exceeds the documented
byte-identical noise band, so the fleet never again orders two recipes by
a one-task draw.

Measured noise sources (see docs/POST_015_ANALYSIS.md):
  - dmitriigluzdov's ledger: identical files drew 4/6/7 and 7/9/11/13 tasks
    -> max observed spread on identical bytes = 9-4 = 6 tasks
  - prvsiyan's ledger: 3-5 across seven draws
  - budget-fit reference: 8-11 across seven runs
Decision rule (conservative): a delta is a SIGNAL only if it exceeds the
max observed identical-bytes spread (>= 6 tasks); 1-2 tasks = pure noise;
3-5 = inconclusive (order a wheelhouse run, not another slot).
"""
import argparse
import json
import sys

TASKS = 58
NOISE_SPREAD = 6          # max observed identical-bytes spread (tasks)
INCONCLUSIVE = 3          # 3..NOISE_SPREAD-1 -> wheelhouse, don't burn a slot


def to_tasks(score: float) -> float:
    return score * TASKS


def classify(delta_tasks: float) -> str:
    d = abs(delta_tasks)
    if d < INCONCLUSIVE:
        return "NOISE — statistically indistinguishable; do not order recipes by this"
    if d < NOISE_SPREAD:
        return ("INCONCLUSIVE — inside the identical-bytes spread; "
                "order a wheelhouse run before spending a slot")
    return "SIGNAL — exceeds the documented identical-bytes spread; worth a slot"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scores", nargs="+", type=float, help="public scores to compare")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    rows = [{"score": s, "tasks": round(to_tasks(s), 1)} for s in args.scores]
    verdicts = []
    for i in range(1, len(rows)):
        a, b = rows[0], rows[i]
        d = b["tasks"] - a["tasks"]
        verdicts.append({
            "pair": (a["score"], b["score"]),
            "delta_tasks": round(d, 1),
            "verdict": classify(d),
        })

    if args.json:
        print(json.dumps({"tasks_per_board": TASKS, "rows": rows, "verdicts": verdicts}, indent=1))
    else:
        for r in rows:
            print(f"score {r['score']:.2f} -> {r['tasks']:.0f}/58 tasks")
        for v in verdicts:
            print(f"delta {v['pair'][0]:.2f} -> {v['pair'][1]:.2f}: {v['delta_tasks']:+.1f} tasks | {v['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

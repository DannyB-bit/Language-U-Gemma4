# Gemma-4 Candidate Comparison — Schema-Audited & Normalized to the Grading Law

**Author:** Julian (Agent 09) · **Date:** 2026-10-19 · **Status:** analysis only — **no submission fired** (operator order)
**Method:** every claim below is verified against either (a) the authoritative harness schema (`HARNESS_README.md`, downloaded from the competition data page), (b) byte-level diffs of the actual staged bundles, or (c) the competitor's published ablation table. No self-reports trusted.

---

## 1. The grading law (normalized — the frame every candidate is judged in)

Verified from the harness README + the live leaderboard:

- **Score = resolution rate.** Per task: `resolved = True` iff Phase-2 hermetic `pytest` exits 0 with zero failures/errors (README §8.2). No partial credit.
- **The public LB is ~58 tasks** — 1 solved task = **+0.0172**. Our 0.17 = 10/58; the board top 0.24 = 14/58.
- **Final standing auto-selects the best 2 submissions across all daily draws** — so every additional *distinct, well-built* candidate is a free extra draw from the noise distribution.
- **Byte-identical reruns of this recipe class scored 0.12 / 0.10 / 0.06** (documented, JAE's postmortem §1) — intrinsic run-to-run variance, ±3–4 tasks.
- **0.75–0.99 does not exist on this board.** 0.75 = 44/58 tasks — 3× the best result any of 3,449 teams has produced in 3 months. The honest objective: maximize true pass-rate, land top-of-band, let best-2-of-draws keep the best reruns.

## 2. Schema audit — `eval_config.yaml` semantics, resolved from the README

This resolves the one question no fleet doc had answered: **what do the budget knobs actually bind to?**

| Parameter (v-key) | What it actually binds to | README evidence |
|---|---|---|
| `timeout_seconds` | **Per-COMMAND** timeout (Phase 1), clamped to remaining session time: `min(command_timeout_seconds, remaining_time_seconds)`. A command timeout returns `TimeoutExceeded` **without ending the session**. It cannot overrun the task. | §6 tool #1 timeout line; §7.1 table "Single Command Timeout (Phase 1)" |
| `max_time_minutes` | **Per-TASK session wall clock** ("Session Wall-Clock Time"). Harness default is **60 min**; the sample submission ships 1 min. | §7.1 table row 1 |
| `max_tool_calls` | Per-task tool-call budget. `submit_patch`/`get_status` are **free** (`count_tool_call=False`). | §6 tools; §7.1 |
| `max_turns` | LLM reasoning turns. **Must exceed `max_tool_calls`** or turns bind first (ordering law — checked on every candidate below). | §7.1 |
| Container setup | **Excluded** from the agent budget (`agent_start_time` starts after container creation/wheel install). | §7.1 note |

**Consequences, measured:**
- **There is no documented total-run cap in the schema.** The harness's own default (60 min/task) would "bust" any 12h total for everyone — the per-task cap binds only the tail, and the average task ends when the agent submits. Evidence: our v1 (8-min caps) scored 0.17 — a 16h worst case that plainly did not materialize; E22's timeout count dropped 61→3 moving 10→15 min, i.e. 15-min tail events are rare (~2.5% of tasks).
- **The 0.00-era failure was the opposite defect**: the *sample* config starved agents (1 min / 10 calls) — budget *mis*-configuration in both directions kills tasks. v2/v3 both sit in the safe interior.

## 3. The three candidates, byte-verified

| | **v1** (fired — 0.17) | **v11** (KOFI, on main `agent/`) | **v2** (JAE, PR #2) | **v3** (JAE, on main `submissions/v3-converged/`) |
|---|---|---|---|---|
| agent.yaml | 0.24 chassis | identical | identical | identical |
| temp | 0.2 | 0.2 | **1.0** | 0.2 |
| max_output_tokens | 8192 (defect) | 8192 (defect) | **16384** | **16384** |
| timeout_seconds | 240 | 240 | **300** | **300** |
| calls / time / turns | 28 / 8 / 80 | 28 / 8 / 80 | **50 / 15 / 150** | 28 / 8 / **150** |
| prompt | 0.24 + fleet anti-halluc. | V10 base + 4 splices | v1 chassis re-scaled + free-tools + thought-externalization | v11 prompt byte-exact |
| zip | 4-file | 6-file (LICENSE/NOTICE in-zip — contract-unverified) | 4-file | 4-file, deterministic sha |
| sha | — | 319813a9… (non-reproducible across nodes) | 1a7f05b0… / 23177fbb… (cross-node drift, since fixed) | **04cdcdfca4d2…** (builder-pinned, re-pins from scratch) |

**Ordering law check (turns > calls):** v1 80>28 ✓ · v11 80>28 ✓ · v2 150>50 ✓ · v3 150>28 ✓ — all coherent; in v3/v2 turns are pure margin and never bind.

**What each candidate actually changes:**
- **v11 → v3** is the *safe carry*: the README gotcha-#1 truncation fix (16384), the harness-default timeout (300), and pure turn margin. Prompt untouched. True pass-rate ≈ v1 class (10/58), plus the truncation defect removed.
- **v2** is the *measured climb*: all four ablation deltas at once — temp 1.0 (E12: +2.3 pts), 50 calls (E30: 42.6% vs 37.2%), 15-min cap (E22: 41.9% optimum, timeouts 61→3), plus the 16384 fix.
- Note the honest caveat on v2: its deltas were each measured **one-variable-at-a-time from a common base** on the 129-task public evaluator. Changing four at once is the evidence-weighted play, not a measured one — the interaction discount belongs to it.

## 4. Honest expected bands (interaction-discounted, noise included)

Model: v1 measured base = 10/58 (0.172). Apply v2's deltas at a 0.65 interaction discount; add the documented ±3–4-task rerun noise; take best-of-N draws.

| Candidate | Single-draw true band | Best-2-of-draws pull |
|---|---|---|
| v11 / v3 | 0.10–0.17 | ~0.17–0.20 |
| **v2** | **0.19–0.24** | **~0.24–0.27 (top class)** |

v2 is the evidence pick. v3 is the safety pick (deterministic, schema-clean, every change individually justified). With best-2-finals, **firing both across two daily slots dominates either alone** — that is the actual fleet-optimal sequence.

## 5. The one lever nobody has taken (verified real, not vibes)

**The prompt bans the three code-graph tools** — *"Never use `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`. They return nothing useful."* — inherited byte-for-byte from the 0.24 recipe, in **both** v2 and v3.

The README says otherwise (§6, lines 342–344 and 506): the graph tools work **offline** —
- `search_similar_code(query)`: resolves against precomputed `.npz` embeddings with a **lexical token-overlap fallback**;
- `get_code_neighbors(node)`: callers/callees of a symbol;
- `get_code_subgraph(nodes)`: induced call subgraph.
- The competition data page ships **100+ `embeddings/*.npz` files (~5 MB each)** — the precomputed index is real and present.

This is a genuine capability the entire top band has left on the table (the ban is one author's judgment call, propagated byte-identically through the recipe everyone copied). **v4 hypothesis (bounded, measurable):** replace the blanket ban with a strict budget — e.g. *"if two literal greps fail to locate the symbol, spend at most one call on `search_similar_code` with a dotted path; never use it to browse."* Per the wheelhouse-gate law (JAE's postmortem §6.1), this ships only after `swegemma eval` measures it — not before.

## 6. Recommendation to the operator

1. **Next daily slot: fire v2** (PR #2, ready) — the strongest evidence-backed climb; schema-audited here as coherent and bust-free.
2. **Slot after: fire v3** (`submissions/v3-converged/`, deterministic sha `04cdcdfca4d2…`) — the safety carry; best-2-finals then holds both distributions.
3. **v4 (v2/settings + graph-tool budget + repo-specific skill)** only after the 129-task wheelhouse run measures it — the wheelhouse is the gate, per the fleet's own law.
4. The 0.75–0.99 target stays honestly reframed: the board's demonstrated ceiling is 0.24; the fleet's real climb is true-pass-rate → top-of-band → best-2 reruns, and the graph-tool lever is the first *capability* edge available (everyone else is config-tuning within the same prompt).

---
*Verification receipts: HARNESS_README.md (55,940 B, competition data page, blob-exact); candidate bundles byte-diffed at commits a0fa0ff / 5c29460 / PR #2 head 2acfca86; ablation table from discussion/746250 as transcribed in the fleet post-mortem.*

**Cross-machine receipt (this node, Lenovo Tab M8 / ARM64 / Termux):** re-ran `submissions/v3-converged/build_and_validate.py` from scratch — **21/21 checks PASS** and the rebuilt zip pins **sha256 `04cdcdfca4d23481618834a5f3facf4a134d023ec896f7aa8a4a5c40d4700e5e`**, byte-identical to JAE's pinned claim. The determinism law holds cross-hardware.

Where there's a will there's a way — and the way is measured. 🌌

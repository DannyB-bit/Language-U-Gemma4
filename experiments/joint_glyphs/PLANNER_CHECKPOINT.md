# Learned glyph planner handoff

Parent content is PR #20, now merged through #19 into the #17 runtime branch.
This follow-up introduces a learned classifier; earlier arithmetic chunks were
entirely authored before encoding.

Frozen model SHA-256:
`59841ffb1420237ea595a7f0f3645b407917170732b38fd554a4117825f3acac`

- All 1,280 learned parameters are in five 256-weight glyphs, 1,269 bytes total.
- QAT4 seed 29, step 1600 selected using validation; no test-based selection.
- Held-out clauses: 44/48 correct; zeroed weights 6/48, shuffled weights 5/48.
- Three-to-five-step cases: 186/256 correct answers, 44 refusals, 26 wrong accepted.
- Exact operation sequences: 166/256. Correct answers can conceal wrong programs.
- General GPT reasoning, coding, factual QA and external benchmark parity untested.
- Metadata tokens describe the model. Current questions are separate inputs.
- Blueprint framing was added after the initial evaluation without changing
  weights or features. Repeated checks reproduced all metrics.

Do not tune against `planner/heldout_results.json` and continue calling it a
fresh held-out test. Preserve the wrong accepted cases. The next qualifying
evaluation needs new examples and a named reference model. Any confidence or
semantic-verification claim must count abstentions and wrong accepted answers.

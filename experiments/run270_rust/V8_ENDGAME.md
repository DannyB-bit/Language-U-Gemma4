# V8.0 endgame: from sealed GPU artifacts to a proven Pi 4 runtime

## Honest status

`origin/main` at `86ce1dc` proves V8.0 GPU calibration and artifact sealing. Its
`COMPLETED_AND_SEALED` receipt is **not** a Pi 4 execution receipt. Step 11 of
`PROJECT_STATE_AND_TODO.md` says this explicitly: the V8 Pi run had not started.

This branch closes the missing code-side ending. It does **not** fabricate the
remaining hardware evidence. Final completion requires the Pi procedure below
and a committed `skeptic_proof_report.json` produced by that run.

## What was incomplete in the code

- `adaptive_mod_q` and `dyn_draft_k` were computed by the six-sense consensus
  gate but never consumed by `run31b`.
- POV decisions were evaluated after the MoD skip branch, so an adaptive MoD
  threshold could not have affected the current layer anyway.
- Missing explicitly supplied C8D, native, shared-LOD, or district-router files
  were silently omitted from evaluator commands.
- `--mode all` could silently skip vision and spec-AB; teacher logits and NLL
  telemetry could be partial while the driver still printed `[DONE]`.
- A lane named `dense` still included V8 LOD/POV controls when those were in the
  base command. It is now labeled `v8_base_no_crtr_mod`; a true dense baseline
  must be a separate V4.3 invocation without V8 controls.

## What this branch changes

1. `run31b` evaluates POV once before MoD and reuses the same decision for MoD,
   CRTR top-p, and LOD top-k.
2. Ordinary causal decode uses `adaptive_mod_q`. The minimum `dyn_draft_k`
   across evaluated layers becomes an upper bound on the next draft budget.
3. Speculative decode freezes target-side adaptive mod/top-p/top-k values and
   uses POV only as a draft cap. Existing KV append/truncate and EWMA acceptance
   logic is unchanged.
4. `verify_v8_0_artifacts.py` verifies 22 runtime/spec files against
   `SHA256SUMS.txt` with streamed SHA-256; it never loads tensors into RAM.
5. The evaluator fails closed on missing files, incomplete modes, missing or
   partial teacher logits, missing NLL/KL telemetry, and missing V8
   GTALOD/POV telemetry. `complete: true` is emitted only after all selected
   lanes finish.
6. Spec-AB reports generated-text parity. This is evidence, not a substitute
   for a separate sequential-vs-speculative token-ID parity test.

## Stage 1 — local and Pi static gates

```bash
python3 experiments/run270_rust/calib/31b/verify_v8_0_artifacts.py --selftest
python3 experiments/run270_rust/calib/31b/eval_v6_1_skeptic_proof.py --selftest
RUSTFLAGS='-D warnings' cargo test --release \
  --manifest-path experiments/run270_rust/Cargo.toml
cargo build --release --bin run31b \
  --manifest-path experiments/run270_rust/Cargo.toml
```

The whole repository has pre-existing rustfmt drift. Do not reformat unrelated
files as part of the V8 proof.

## Stage 2 — verify the actual external V8 artifact set

On the Pi, with the V8 Drive directory mounted at `$V8`:

```bash
export V8=/mnt/SHE-VAULT/gemma4_v8_0_player_pov
python3 experiments/run270_rust/calib/31b/verify_v8_0_artifacts.py \
  --artifact-dir "$V8" | tee "$V8/v8_0_pi4_artifact_preflight.json"
```

The command must print `status: PASS`, `files_verified: 22`, and
`sha256_verified: true`. A missing district, checksum mismatch, partial BHDR, or
missing Q2/Q4/Q8 assistant body is a hard stop.

## Stage 3 — short live smoke before the long battery

```bash
export HF=/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct
export RUN=experiments/run270_rust/target/release/run31b

"$RUN" \
  --t16c "$V8/gemma4_31b_v8_0_base.t16c" \
  --config "$HF/config.json" --tokenizer "$HF/tokenizer.json" \
  --cuneiform-bank "$V8/gemma4_31b_v8_0_c8d.c8d" \
  --cortical-router "$V8/gemma4_31b_v8_0_router.crtr" \
  --lod-shared "$V8/gemma4_31b_v8_0_lod_shared.t16c" \
  --lod-district-dir "$V8" \
  --district-router "$V8/gemma4_31b_v8_0_pov_6sense_router.drtr" \
  --player-pov-6sense --kv-quant glyph --mlp-top-p 0.35 --mod-thresh 0.35 \
  --lmhead-cs exact --max-tokens 8 \
  --prompt-text 'In one sentence, explain why evidence must match a claim.'
```

Required log evidence: nonzero `GTALOD31B`, `POV6SENSE31B`, and `MEM31B`
telemetry; no panic; no KV-desync message. Record peak RSS. Do not infer a speed
claim from an eight-token smoke.

## Stage 4 — full V8 Pi evaluation

```bash
export OUT=/mnt/SHE-VAULT/work31b/v8_0_pi4_final
python3 experiments/run270_rust/calib/31b/eval_v6_1_skeptic_proof.py \
  --mode all \
  --run31b "$RUN" --t16c "$V8/gemma4_31b_v8_0_base.t16c" \
  --config "$HF/config.json" --tokenizer "$HF/tokenizer.json" \
  --c8d "$V8/gemma4_31b_v8_0_c8d.c8d" \
  --router "$V8/gemma4_31b_v8_0_router.crtr" \
  --lod-shared "$V8/gemma4_31b_v8_0_lod_shared.t16c" \
  --lod-district-dir "$V8" \
  --district-router "$V8/gemma4_31b_v8_0_pov_6sense_router.drtr" \
  --player-pov-6sense \
  --assistant-v6 /home/astronautshe/gemma4_v6/gemma4_31b_assistant_v6.t16c \
  --assistant-v7-body "$V8/gemma4_31b_v8_0_assistant_body_q8.t16c" \
  --byte-head "$V8/gemma4_31b_v8_0_assistant.bhdr" \
  --image /mnt/SHE-VAULT/work31b/frame_01_coco.png \
  --battery-top-p 0.50 --battery-mod-thresh 0.35 --out-dir "$OUT"
```

`--mode all` now requires every lane. The Q8 assistant is intentional here; the
artifact preflight verifies Q2/Q4/Q8 so lower-RSS spec lanes remain available.

## Stage 5 — independent true-dense baseline

Run `nll-pareto` separately against the V4.3 base artifact with no `--lod-*`,
`--district-router`, or `--player-pov-6sense` options. Do not compare V8 to the
`v8_base_no_crtr_mod` row and call it a dense-V4.3 result.

Teacher-KL is optional and separately gated. The V8 GPU receipt contains no V8
`.lgt1` teacher-logit artifacts. Until matching logits are generated for all
three held-out passages, report NLL and battery evidence only—never V8 KL.

## Acceptance contract for final completion

- Artifact preflight: 22/22 hashes pass.
- Native and Python gates pass on the Pi architecture actually being claimed.
- Full report has `complete: true`, positive `eval_pos` for every NLL row, 15/15
  battery rows executed (not necessarily correct), vision, and both spec lanes.
- Every V8 lane has positive GTALOD and POV telemetry; memory is measured by
  `VmHWM`, not estimated from artifact sizes.
- Quality and speed are reported from the resulting JSON. There is no preset
  "must win" adjustment to CRTR, MoD, LOD, or draft parameters.
- Sequential-vs-speculative token-ID parity is a separate explicit experiment.
  Spec-AB text parity alone does not prove it.
- Commit the Pi report and logs, then change project state from
  `verification pending` to `verified` with commit hashes and artifact SHA-256.

## Tablet verification already performed

On Lenovo Tab M8 / Android / aarch64, this branch compiled warning-free in
release mode and passed 94 Rust tests. The artifact and evaluator self-tests
passed, including rejection of missing and corrupted V8 files. The external V8
weights and Pi 4 were not available in that session, so no Pi execution or
quality result is claimed.

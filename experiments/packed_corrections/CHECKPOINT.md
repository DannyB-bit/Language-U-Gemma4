# Packed correction checkpoint

PR #18 is stacked on #17. No production runtime defaults are changed.

## Completed

- Public-only reproduction scripts and attribution published in the first PR commit.
- Reproduced rank-8 direct correction after relocating scripts into the repo.
- Added RCF1 FP16/INT8 correction storage and chunked FP32 reference inference.
- Validation chose rank-15 INT8 at 83,196 bytes over rank-8 FP16 at 88,128 bytes.
- Reused development test error: 0.0687073346 -> 0.0669766775; 2.52% reduction.
- Nine offline tests pass locally, including malformed files and independent arithmetic.
- Precision receipts retain input/base digests and all validation candidates.

## Reproduction and interpretation

Run the README commands in a fresh data directory. Factor writers intentionally
refuse to overwrite an earlier run. The experiment uses pinned public inputs
and requires no private code. Source Q excerpt: 128x5376; input splits: 112/85/80
disjoint token types. Both corrections are evaluated with stored FP32 operators.
The selected INT8 rank-15 sidecar SHA-256 is
`8631afe273a3333280eb4e32c16400eaf568d77a2f6c7880cab123b0542471b9`.

These are development projection results. The model still makes approximation
errors; there is no full-model coherence, lossless recovery, 5 GB residency or
Pi speed proof. INT8 rank 15 increases compute versus rank 8. Its FP32 latent
occupies 60 bytes; file-byte savings do not establish peak-memory savings.

## Next qualifying work

1. Freeze candidates and evaluate unseen token types before further tuning.
2. Expand to actual intermediate activations and sensitive projections across layers.
3. Allocate correction rank and base precision under an explicit global byte budget.
4. Build and evaluate a complete source-correct candidate with held-out language tasks.
5. Measure native ARM execution, actual KV/runtime memory, and physical Pi I/O/latency.

Preserve negative results and selection provenance. Any later test-set reuse
turns that set into development data. Update this checkpoint when advancing it.

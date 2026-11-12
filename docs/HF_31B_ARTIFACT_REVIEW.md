# Full 31B Hugging Face artifact verification

This extends PR #17 from E4B sample checks to the actual published 31B stream.
The source is pinned to
[`TheAiCollectiveART/Glyph-Stream-31b@bd9c48f1da41b51521570674ba2696657eb7358e`](https://huggingface.co/TheAiCollectiveART/Glyph-Stream-31b/tree/bd9c48f1da41b51521570674ba2696657eb7358e).
All new measurements below concern artifact bytes and host arithmetic. They do
not report physical Pi 4 generation throughput or a new quantized model's quality.

## Verified full-stream result

| Item | Result |
| --- | ---: |
| Downloaded original stream | 7,080,837,136 bytes |
| Original header | `LANU`, version **1**, flags 45 |
| Projections | 410, contiguous, with consistent shapes and offsets |
| Total 256-weight glyphs | 114,401,280 |
| Opt4, 132 bytes | 18,923,520 glyphs |
| Opt5, 48 bytes | 95,477,760 glyphs |
| Losslessly packed stream | **6,323,896,336 bytes** |
| Exact byte reduction | **756,940,800 bytes** |
| Independent reverse reconstruction | Original full-file SHA-256 reproduced |
| Sampled record checks across all projections | 1,230 |
| Representative full-matrix GEMV comparisons | 6 |

The original download matches the Hub's LFS SHA-256:

```text
9465174f3bd782eb549457d51f62605e2ecffc4bdc5fd9b5aad8ea89861cf4c8
```

The packed `LANP` version 3 file has SHA-256:

```text
cf664e5db891e7e666e8b1114644003e13bd9832fcafded96b4649ecdf8bc164
```

Every Opt4 batch was round-trip checked during packing. The verification then
independently read the **entire packed file**, expanded its 92-byte Opt4 records,
retained the Opt5 records and restored the original 16-byte header into a streaming
SHA-256 digest. That digest matches the original full-file hash above. No second
7 GB reconstructed file is required. This proves lossless container conversion
of the complete linear-weight stream; it does not claim lossless conversion of
the original pre-quantization model.

Sampled native Opt4 decodes match bit-for-bit under both direct and H8
interpretations. The six native GEMV comparisons cover Q and down projections
in layers 0, 5 and 59. Their worst relative L2 difference is approximately
`2.875e-7`; the largest absolute difference is `0.001953125`, reflecting changed
FP32 accumulation order. These probes are decoded weight rows, not captured
model activations or text-generation tests.

Raw evidence and source hashes are in
[`hf_31b_artifact_evidence.json`](hf_31b_artifact_evidence.json).

## Full redundancy scan

The native scanner processed every record, sorting 64-bit fingerprints and
resolving equal fingerprints with actual metadata/payload comparisons. It does
not assume that hash equality establishes duplicate glyphs. Unit tests force
one-bit fingerprints to exercise collision handling.

| Measure | Exact result |
| --- | ---: |
| All glyph records | 114,401,280 |
| Unique glyph records | **114,400,946** |
| Duplicate occurrences | 334 |
| Repeated payload bytes that could be removed before indexing | 16,200 |
| Sequential zlib level-6 payload size, header excluded | 6,007,540,883 bytes |
| Original payload / zlib payload | 1.1786581661 |
| Ideal unique payload plus packed record indices, excluding metadata | 7,466,925,240 bytes |
| C working allocation allowance, excluding mmap pages and Python | 3,661,889,536 bytes |

A dictionary is larger than the original payload even under an idealized index
ledger. Zlib also remains above the 4.8 GB linear-stream target. These results
rule out those two measured approaches as sufficient solutions for this artifact;
they do not rule out every possible lossless compressor.

`stream_census.c` uses two bounded 16-byte-per-record arrays for radix sorting
and in-place comparison sorting only within fingerprint collision groups. It
does not materialize a full decoded weight matrix or a Python record set.
The allocation figure is **not** a total RSS measurement: file-backed pages,
the Python process and operating-system caching are separate. This utility
requires 64-bit Linux, a C compiler and zlib development headers. The existing
SQLite scanner remains available for other hosts and small files.

## Compatibility findings and fixes

1. The published stream uses **LANU v1**. The initial PR's restrictive header
   validation accepted only v2/v3; this is corrected. A regression test uses the
   actual published header and exercises v1-to-v3 repacking and native execution.
2. The uploaded codebook is `golden_s7`, with SHA-256
   `099cfea75908bf4ca74820c732e040179e5354f6cd023af099faf1d439c37a47`.
   It differs from the repository's E4B sample codebook. Full-stream checks use
   the uploaded table explicitly; substituting the sample table would change
   decoded weights.
3. All 410 uploaded projection records omit `rotation_mode`. Container identity
   checks test both interpretations, but representative GEMVs use the bridge's
   direct-mode default. The absent metadata does not independently establish the
   encoder's intended rotation or full-model semantic correctness.
4. Packing/unpacking now vectorizes over a bounded batch. This removes the
   per-index Python loop from full-model conversion while retaining the same
   bit layout. It is an offline format operation, not the inference LUT-kernel
   optimization that the spec defers until Gate 1.

## Corrected RAM ledger

The upload contains a separate 2,818,572,288-byte token embedding table and a
2,990,082-byte norms file. Their published file sizes are included below; these
two companion binaries were not locally downloaded/hash-verified in this audit.
The published configuration also sets `tie_word_embeddings=true`, so accounting
for input token lookups alone does not account for the tied output head.

| Arrangement | Published/projected bytes |
| --- | ---: |
| Measured losslessly packed linear stream alone | 6,323,896,336 |
| Hypothetical all-36-byte linear stream alone, including header | 4,118,446,096 |
| Hypothetical all-36-byte stream + unchanged embedding and norm files | **6,940,008,466** |
| Above + hypothetical 200 MB runtime reserve | 7,140,008,466 |
| Spec's 10% at 92 bytes / 90% at 36 bytes, linear stream alone | 4,759,093,264 |
| Spec's 10% split + unchanged embedding/norm files + 200 MB reserve | **7,780,655,634** |

The all-36-byte and 10% arrangements are arithmetic design points, not measured
quality-qualified candidate models. The 200 MB reserve is editable and
hypothetical; it is not a measured KV-cache/runtime footprint.

These unchanged companions and the hypothetical all-36-byte linear stream
cannot all be resident inside the specified 5,000,000,000-byte total envelope.
The plan must explicitly include verified embedding/output-head compression,
deliberate streaming with measured I/O, or a different total memory envelope.
Norm headers, codebooks, KV state, scale maps, allocations and runtime buffers
also need accounting. No inference latency is derived from these file sizes.

## Meaning of the uploaded Pi report

`PHYSICAL_PI4_31B_VERIFIED_REPORT.json` reports kernel data-path verification
with a synthetic probe vector. It explicitly excludes token generation,
embedding, a complete 60-layer forward pass and logits. This is useful existing
kernel evidence; it does not by itself pass the supplied spec's new generation,
held-out quality or RAM-residency gates. Its Pi results were not independently
rerun by this audit.

The original full linear-weight artifact is now available and independently
hash-verified. **Gate 1 remains unpassed**: a RAM-qualified candidate, actual
calibration/held-out inputs and logits, and physical Pi full-generation memory,
read-I/O and latency measurements are still required. LUT inference optimization,
dynamic skipping and speculative generation remain deferred.

## Reproduce on downloaded artifacts

Use the pinned source snapshot and a new output directory. Do not overwrite the
original stream. Substitute the actual local model directory below.

```bash
python3 -m pip install -r requirements-pi4.txt
python3 neon_runner/build_native.py

python3 scripts/pi4_barrier_break.py scan-native \
  --stream /model/gemma4_31b_language_u_sandwich_stream.bin \
  --layout /model/gemma4_31b_tensor_layout.json \
  --expected-sha256 9465174f3bd782eb549457d51f62605e2ecffc4bdc5fd9b5aad8ea89861cf4c8 \
  --report /new/full-stream-census.json

python3 scripts/pi4_barrier_break.py pack \
  --stream /model/gemma4_31b_language_u_sandwich_stream.bin \
  --layout /model/gemma4_31b_tensor_layout.json \
  --chunk-glyphs 16384 --output-directory /new/lossless-packed

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python3 scripts/verify_31b_artifact.py \
  --model-directory /model --packed-directory /new/lossless-packed \
  --census /new/full-stream-census.json --report /new/31b-artifact-audit.json
```

The audit command requires the original stream, layout, uploaded codebook,
configuration and uploaded Pi report from the exact pinned snapshot. Its input
hash checks intentionally reject mismatched artifacts. It does not download or
execute remote Python model code.

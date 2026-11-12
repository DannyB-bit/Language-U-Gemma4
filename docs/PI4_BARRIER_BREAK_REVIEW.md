# Pi 4 31B barrier: implementation and evidence review

Reviewed against repository baseline `33db3e354ee59d06761f58fb46f177542fc9b73c`
and the supplied `PI4_31B_BARRIER_BREAK_SPEC.md`, November 12, 2026.

The 31B RAM-resident inference target remains **unverified**. This change supplies
working format support, calibration primitives, fixes, and reproducible software
evidence. It does not report a Pi 4 speedup or a successful 31B quality gate.

**Full-artifact update:** the subsequently supplied Hugging Face stream is now
downloaded, hash-verified, scanned in full and losslessly repacked. See the
[31B artifact review](HF_31B_ARTIFACT_REVIEW.md) for the 756,940,800-byte saving,
LANU v1 compatibility fix, exact redundancy census and revised embedding-inclusive
memory ledger. The sample-only results below remain a separate software check.

## Measured contribution

The committed E4B weight samples can be repacked without changing their Opt4
scales or codebook indices. The new native decoder produces bit-identical FP32
weights with either direct or H8 interpretation. GEMV accumulation order can
differ, so its result is checked with a numerical tolerance.

| Supplied sample | Original file | Packed file | Exact bytes saved |
| --- | ---: | ---: | ---: |
| Layer 0 Opt4, 160 glyphs | 21,136 | 14,736 | 6,400 |
| Hybrid, 160 Opt4 + 160 Opt5 | 28,816 | 22,416 | 6,400 |

Each Opt4 glyph falls from 132 to 92 bytes: **30.303% fewer payload bytes**.
The Opt5 records are unchanged. The recorded host GEMV difference between the
original and packed Opt4 containers is `2.384185791015625e-7` on a probe derived
from the supplied weight sample. This is a software parity check, not a model
activation or a generation-quality measurement.

Exact redundancy scans found 160/160 and 320/320 unique glyph records in these
two samples. Streaming zlib compression ratios were approximately 1.226 and
1.147 respectively. Ideal dictionary storage, even before dictionary metadata,
would be larger than the original sample payloads. These counts say nothing
about the uniqueness of the unavailable full 31B stream.

Raw results, file hashes, source hashes and host architecture are in
[`pi4_preparation_evidence.json`](pi4_preparation_evidence.json).

## Changes

* `neon_runner/pi4_stream.py`: exact 11-bit packing/unpacking, bounded-memory
  stream rewrite, layout/range validation, collision-free disk-backed SQLite
  uniqueness scan and sequential zlib measurement. Source files are preserved;
  outputs must use new paths. Streaming hashes detect source changes.
* Native C: additive `decode_glyphs_opt4_packed_c` and
  `gemv_opt4_packed_mmap` symbols. Decode uses a single 132-byte stack record per
  worker; GEMV reads the packed records directly. This is a correctness-first
  implementation, without a speed claim. Existing 132-byte symbols retain their
  ABI. The build must be updated before using a 92-byte stream.
* Bridge: explicit 92-byte dispatch, input/range/format rejection before native
  reads, and a fix for Opt5 `rotation_mode="fwht8"`. Previously, GEMV ignored that
  rotation in both native dispatch and the legacy fallback.
* Native H8: removed the conflicting `restrict` promise from a function that the
  attention kernel calls in place. Its documented API now permits identical
  input/output pointers. SIMD-only variable declarations are guarded so the
  portable scalar library builds with `-Wall -Wextra -Werror`.
* `neon_runner/pi4_quantization.py`: incremental channel maxima/second moments,
  shared-consumer scaling and norm folding, optional `1 + gamma` storage
  convention, bias-free GeGLU up/down folding, covariance basis transformation,
  a reference intra-tile second-order quantizer, corrected skip-bound helpers,
  and exact integer RAM accounting.
* `scripts/pi4_barrier_break.py`: `scan`, `pack`, `calibrate`, and `budget` commands.
  Calibration consumes a real exported floating-point `[tokens, channels]` `.npy`
  array. Optional full tile covariances use additional memory.
* GitHub Actions: sample correctness runs on Linux x86-64 and AArch64, using
  SHA-pinned checkout/setup actions and a strict portable C build. Hosted AArch64
  execution exercises NEON but does not constitute a physical Pi 4 benchmark.

The calibration API does not discover model graph consumers. The caller must
identify every consumer of a shared norm, supply the correct stored-gamma
convention, collect inputs from the actual inference path, and integrate the
folded parameters with the external model runtime. There is no full-model
mixed-precision encoder, automatic sensitivity allocator, Rust runtime, or
generation engine added by this change. The budget command is a ledger, not a
quality-qualified precision allocation. The intra-tile quantizer is a reference
implementation, not a throughput-optimized full-model GPTQ implementation.

## Corrections required before the supplied spec can pass its gates

1. **Shared norms require shared scaling.** If Q/K/V consume the same norm
   output, use one channel scale across all three. Deriving separate scales and
   folding each into that one gamma cannot preserve all consumers. The same
   rule applies to gate/up branches. Stored `1 + gamma` gains must be folded as
   `(1 + gamma) / s - 1`. SmoothQuant equivalence before quantization does not
   establish improved weight-only 1-bit accuracy; sweep alpha and keep an
   unscaled control.
2. **Diagonal statistics are not a full GPTQ Hessian.** Off-diagonal error
   feedback needs actual within-block cross correlations. Collect them from real
   projection inputs. When weights become `W D_s H8^T`, the calibration input
   covariance must become `H8 D_s^-1 H D_s^-1 H8^T`. The reference quantizer fixes
   fp16 scale/bias, applies inverse-Cholesky error feedback within a tile, then
   retains the better candidate under the actual calibration objective. It does
   not compensate across tiles or establish held-out model quality.
3. **36-byte Opt5 is an affine binary format.** There are 256 sign bits and two
   fp16 values. A third independent zero level is not encoded. The existing ABI
   is `[scale:2][bias:2][mask:32]`, with MSB-first sign bits within each byte.
   LUT bit ordering must follow that convention. H8-encoded weights require
   H8-transformed activations, including the input sum used by the bias term.
4. **The skipping bound needs bias, rows and accumulation.** For one 256-weight
   glyph with signs `s`, scale `a`, and bias `b`, the real-arithmetic norm obeys
   `||a*s + b||_2^2 = 256*(a^2+b^2) + 2*a*b*sum(s)`.
   `16*abs(a)` is false when bias is nonzero; `a=0, b=1/8` is a direct
   counterexample. Across output rows use the full block Frobenius norm, then
   enforce `sum(skipped ||W_block||_F * ||x_block||_2) <= epsilon`. A separate
   threshold on each block does not impose one total layer budget. Supplied
   helpers include conservative float64 rounding allowances and reject overflow.
   The guarantee applies to the linear projection; it does not guarantee later
   logits or unchanged sampling. Skipping is not enabled in the runtime.
5. **Count all bytes.** At 10% high precision, the glyph payload is exactly
   `4,759,093,248` bytes; with the LANU header it is `4,759,093,264`. A hypothetical
   200,000,000-byte runtime reserve brings the ledger to `4,959,093,264` bytes.
   KV cache, embeddings, norms, codebooks, scale vectors, precision maps,
   offsets/alignment, allocator overhead and any draft model must be measured
   and included. With zero sidecar bytes and that reserve, at most 12,170,605
   glyphs fit in the 92-byte tier (about 10.6385%). This is integer arithmetic,
   not evidence of sufficient runtime headroom or residency. Packing every
   glyph at 92 bytes would still take 10,524,917,760 payload bytes.
6. **Warm mmap does not prove RAM residency.** Record resident pages, RSS/PSS,
   process I/O and major faults around the actual complete warm decode.
   Zero major faults alone is insufficient: readahead and other activity can
   still perform disk I/O. A 4.8 GB file alone does not establish zero reads.
7. **A LUT is small per input tile, not necessarily per layer.** A 256-input tile
   has 64 tables. A `d_in=21504` layer has 84 such tiles. Caching all int8 tables
   would take 84 KiB (int16: 168 KiB), exceeding a 32 KiB L1. Tile the input/LUT
   working set and output-row groups deliberately. Four full-range int8 inputs
   also need a wider signed-sum table or smaller activation range to prevent
   overflow. Optimizing this is deferred until Gate 1 passes.
8. **Speculative exactness is conditional.** Rejection sampling can preserve a
   defined target model distribution when proposal probabilities, target
   probabilities, causal verification, cache rollback, rejection correction and
   tokenization are correct. SmolLM2 and Gemma cannot be assumed to share token
   IDs. It does not recover the original model distribution after lossy
   quantization or approximate skipping. A batched forward pass over supplied
   tokens measures teacher-forced verification/prefill, not sequential generation
   throughput without an acceptance ledger.

The existing optical benchmark also needs that scope distinction:
[`benchmark_optical_gemma31b_live.py`](../scripts/benchmark_optical_gemma31b_live.py)
constructs random input vectors and runs layer-0 Q/K/V GEMVs. Its carrier CRC and
projection measurements do not establish full-model text generation or the
four-prompt quality gate in the supplied Pi 4 spec. The supplied Hamlet and
latency results remain reported baseline observations that were not independently
reproduced here.

## Reproduce this change

NumPy, Pillow, Python 3 and a C compiler are required for the new preparation
suite. Pillow is loaded by the package's existing optical-carrier import, even
when invoking stream-only tools. Pinned dependencies are supplied in
[`requirements-pi4.txt`](../requirements-pi4.txt).
The commands below generate software evidence on the current host; running them
on a Pi does not by itself add a full-model quality benchmark.

```bash
python3 -m pip install -r requirements-pi4.txt
python3 neon_runner/build_native.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python3 -m unittest discover -s tests -p test_pi4_barrier.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python3 scripts/verify_pi4_preparation.py --report /tmp/pi4-preparation-new.json
python3 scripts/pi4_barrier_break.py budget --high-glyphs 11440128
```

Run the actual full-stream redundancy scan on the machine holding those files:

```bash
mkdir -p pi4-runs
python3 scripts/pi4_barrier_break.py scan \
  --stream /actual/path/to/31b.lanu \
  --layout /actual/path/to/31b.layout.json \
  --database pi4-runs/full31b-unique.sqlite
```

The SQLite database stores exact payloads: provide disk space at least comparable
to the stream plus database overhead. The zlib ratio and ideal dictionary figure
are diagnostics; no dictionary execution format or deduplication speedup is
claimed. Paths above are arguments to replace with actual files, not bundled
weights.

Capture calibration from the actual post-norm inputs to a shared attention/MLP
consumer group, or the actual post-GeGLU down-projection input:

```bash
python3 scripts/pi4_barrier_break.py calibrate \
  --activations /actual/path/to/projection-inputs.npy \
  --output pi4-runs/projection-calibration.npz --tile-hessians
```

Collect an independent held-out prompt set. Apply graph-consistent folding
before H8 encoding; use the supplied functions from the external model encoder.
Do not label weight-derived unit probes as calibration. Tile Hessians are larger
than the spec's diagonal-only statistic budget and should be processed one
projection at a time.

To create a losslessly packed copy of a compatible LANU stream:

```bash
python3 scripts/pi4_barrier_break.py pack \
  --stream /actual/path/to/model.lanu \
  --layout /actual/path/to/model.layout.json \
  --output-directory pi4-runs/packed-model
```

The new container uses magic `LANP`, header version 3 and per-projection
`bytes_per_glyph=92`. Distinct magic is necessary because the old LANU parser
did not check version; it makes old consumers reject this file before reading
packed records with the wrong stride.
Use the upgraded bridge and native library. Do not point an older consumer at it
or replace an existing 132-byte stream under the same filename. This command
does not perform lossy 132-to-36 conversion or produce a RAM-qualified model.

## Validation and remaining gate

* New software suite: **21 tests passed** on Linux x86-64, both the host SIMD
  build and the strict scalar C build.
* Existing bridge and glyph-attention self-tests: **6/6 + 6/6 passed** on this
  host. These legacy tests contain constructed inputs and their labels/timings
  must not be represented as the Pi 4 31B proof requested by the supplied spec.
* Existing retrieval math suite: 3 tests passed. Three other retrieval modules
  cannot import because this environment does not have PyTorch; they are
  recorded as dependency-blocked, not passed.
* AArch64 compilation and physical Cortex-A72 execution have not been verified
  in this x86-64 environment.

**Gate 1 is not passed.** Required next evidence is the complete 31B candidate
stream/ledger, calibration provenance, exact Hamlet tokens plus at least three
held-out prompts and raw per-step logits, and real Pi warm-run memory/I/O data.
Only after it passes should the int8/int16 `vqtbl1q_u8` kernel, skipping or
speculative generation be optimized or advertised as a speedup.

## Technical sources

* [SmoothQuant paper](https://arxiv.org/abs/2211.10438) and
  [official implementation](https://github.com/mit-han-lab/smoothquant):
  equivalent activation/weight scaling; the original paper studies W8A8,
  rather than proving this proposed 1-bit weight-only target.
* [GPTQ paper](https://arxiv.org/abs/2210.17323) and
  [official implementation](https://github.com/IST-DASLab/gptq): second-order
  post-training quantization and compensation.
* [T-MAC paper](https://arxiv.org/abs/2407.00088) and
  [official implementation](https://github.com/microsoft/T-MAC): LUT inference
  precedent; its reported performance does not establish a Pi 4 31B result.

The format savings, corrected bounds and gate review above are derived from the
supplied spec and this repository's source and samples.

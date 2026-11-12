# Packed weights with learned correction

This development experiment compares binary-factor weight storage, calibration
of its existing BF16 scales, and a learned input-dependent correction:

`y = PackedFactors(x) + B @ (A @ x)`

It covers the first 128 rows of the source Gemma 4 31B layer-0 query projection,
with real source embedding/RMSNorm inputs. It is not a full model, lossless
weight reconstruction, English-coherence benchmark or physical Pi measurement.

## Recorded results

| Method | Relative squared projection error | Output cosine |
| --- | ---: | ---: |
| Calibrated binary factors | 0.07579274 | 0.96139902 |
| Four-byte 8D codebook guide | 0.07089567 | 0.96390538 |
| Direct rank-8 correction | 0.06870733 | 0.96504121 |

The direct correction reduces the previous guide's error by 3.09%. The numbers
are projection statistics, not percentages of correct answers. Validation
preferred one rank-8 correction over two rank-4 or four rank-2 experts. All
three have 88,064 bytes of FP16 factor payload; routers add metadata. Direct
correction retains eight coordinates (16 temporary bytes if stored FP16),
instead of the four-byte quantized guide, and needs no 65,536-byte codebook.
Recorded direct-correction arithmetic is float64 with FP16-stored factors.

Training, validation and development test sets contain 112, 85 and 80 disjoint
token IDs from locally authored English passages. The test set has been reused
across research iterations; it is not an independent final acceptance set.
Candidate selection within each script uses validation only. Input samples
here contain no preceding attention context because this is layer-0 pre-attention.

## Reproduce

Requires the checkpoint/runtime helpers introduced in PR #17. From repository
root, use a fresh data directory (writers refuse to replace factor binaries):

```sh
python3 -m pip install -r experiments/packed_corrections/requirements.txt
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export LANGUAGE_U_EXPERIMENT_DIR="$PWD/experiments/packed_corrections/data"
python3 experiments/packed_corrections/prepare_public_inputs.py
python3 experiments/packed_corrections/probe_real_activations.py
python3 experiments/packed_corrections/binary_factor_probe.py
python3 experiments/packed_corrections/tune_factor_scales.py
python3 experiments/packed_corrections/probe_glyph_guidance.py
python3 experiments/packed_corrections/probe_routed_corrections.py
python3 experiments/packed_corrections/budget_full_model.py
```

Public model and tokenizer downloads use pinned revisions. Range fetches check
status, Content-Range and length, with fixed digests for the initial projection
excerpts. The entire 23 GB reference checkpoint is not downloaded or rehashed.
Recorded JSON files under `evidence/` are receipts from the original scratch
experiment; their historical paths are not instructions or current outputs.
New results and selected factors go to the configured data directory.

BFP1 uses a 64-byte header, BF16 input/output scales, and packed sign matrices.
Its chunked projector decodes factor rows rather than materializing the full
effective weight matrix. Calibration and independent oracles do materialize
small dense matrices; this is not a peak-RAM or latency claim.

## Attribution and boundaries

The NumPy LB-ADMM initializer is adapted from SamsungLabs/NanoQuant at
`a9e0a430881ff80d83b622c3129e330dc33c04f5`. Apache-2.0 license, NOTICE and original
source are in `third_party/NanoQuant/`. This is only its initialization component;
it omits gradient KFAC, block STE and full-model KL reconstruction. Other public
artifacts retain their own terms.

The full-model budget JSON is dimension arithmetic for unbuilt candidates.
Its runtime reserve is assumed. It does not establish that the successful
1.08854-bit excerpt generalizes to a 0.9-bit full model or fits a physical Pi.
The earlier full 31B low-bit candidate failed its quality gate. A full
source-correct model, independent generation tests and measured memory/I/O are
still required. No private repository review or source corpus is included here.

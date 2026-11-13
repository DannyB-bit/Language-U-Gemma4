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
python3 experiments/packed_corrections/probe_correction_precision.py
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


## Stored correction with FP32 execution

The follow-up freezes the calibrated BFP1 base and compares corrections within
an 88,128-byte file budget (rank-8 FP16 factors plus a 64-byte header). Candidate
ranks/storage and ridge are selected using validation only. Both the base and
correction run through chunked FP32 operators on the evaluation inputs.

| Correction | Stored bytes, including header/scales | Development error | Output cosine |
| --- | ---: | ---: | ---: |
| FP16 rank 8 | 88,128 | 0.06870733 | 0.96504120 |
| INT8 rank 15 | 83,196 | 0.06697668 | 0.96593449 |

Validation selected INT8 rank 15, ridge 0.01. Compared with the FP16 control,
this saves 4,932 bytes (5.60%) and lowers development error by 2.52%. Stored FP32
correction output agrees with an independent float64 oracle using the same
stored coefficients to 2.67e-7 relative error on validation inputs. This parity
checks execution, not recovery of the original uncompressed model.

`correction_codec.py` writes RCF1: a 64-byte `<4s7I4Q` header, optional FP32
per-row scales, then row-major A and B. Fields are magic, version, output width,
input width, rank, type (1=FP16, 2=INT8), zero flags, zero reserved, scale bytes,
A bytes, B bytes and total payload bytes. INT8 uses symmetric [-127,127] values
and one FP32 scale for each A/B row. A maps input to latent; B maps latent to
output. Files reject malformed dimensions, lengths, flags and nonfinite values.
Writers use exclusive creation. The reader does not materialize B@A.

The selected rank-15 latent uses 60 bytes in the tested FP32 operator. Increasing
rank increases arithmetic as well as correction capacity; smaller files alone
do not prove faster inference or lower total peak RAM. Native ARM kernels,
full-model calibration and new independent language evaluations are still open.

Run the nine offline format/arithmetic tests with:

```sh
python3 -m unittest discover -s tests -p test_packed_corrections.py -v
```

The precision receipt records input/base SHA-256, every candidate, exact file
sizes, selected file SHA-256 and independent parity results. The same small
development test is reused; no full-model quality claim is made.

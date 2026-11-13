# Full text control and RAM-fit gate

**The complete Pi 4 objective is not solved.** The actual all-36B candidate with
a three-bit tied embedding/head fits the file budget and fails every quality
check. Physical Pi 4 residency, read bytes/token and throughput are unverified.
LUT inference optimization, dynamic skipping and speculation stay disabled.

## Evidence boundary

The large-checkpoint experiments below completed earlier in this same session.
A workspace restart subsequently cleared their unpushed binary caches, raw
logit arrays and detailed JSON files. The numerical results in
[`full_text_recorded_results.json`](full_text_recorded_results.json) are
transcribed from completed tool outputs retained in the conversation. They are
explicitly not replacement raw receipts or fresh generation measurements.

The four core source files and the gate CLI were restored **byte-for-byte**
against the SHA-256 fingerprints printed by the completed candidate run.
Software checks were rerun after restoration; their separate results are in
[`full_text_software_validation.json`](full_text_software_validation.json).
The CLI tools permit a fresh full-data reproduction from the pinned sources.

## Source checkpoint and missing parameters

The glyph package is pinned to
[`TheAiCollectiveART/Glyph-Stream-31b@bd9c48f1da41b51521570674ba2696657eb7358e`](https://huggingface.co/TheAiCollectiveART/Glyph-Stream-31b/tree/bd9c48f1da41b51521570674ba2696657eb7358e).
The upstream checkpoint is
[`google/gemma-4-31B-it-qat-w4a16-ct@52f3f65bc7a02d555763bc923bd1d9094898219d`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/tree/52f3f65bc7a02d555763bc923bd1d9094898219d).
Its entire 23,265,352,448-byte safetensors file passed SHA-256 verification:
`1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20`.

All 361 supplied language norm tensors matched upstream dtype, shape and BF16
payload exactly. Both complete upstream embedding/head tensors matched the
published embedding file's SHA-256. The norm package omits all 60 layer scalars.
Actual recovered values include `0.0986328125`, `0.0732421875` for the first two
layers and `0.03173828125` for the last. Substituting ones changes inference.
`recover_31b_parameters.py` validates the full source hash and all norms before
writing the real scalar sidecar with exclusive creation.

The supplied tokenizer decodes `[715,100]` to `//<\|channel>`, not Shakespeare.
Encoding `Shakespeare` without special tokens gives `[157092]`. The gate
records actual control tokens; it does not hardcode the desired answer.

## Projection lineage and reference inference

The first four real rows of each of the 410 projections were compared with
properly dequantized upstream group-32 INT4 weights. This was sampling per
matrix, not a full elementwise comparison.

| Codec | Matrices | Median direct cosine | Median H8 cosine |
| --- | ---: | ---: | ---: |
| Opt4 boundary | 69 | -0.001246 | -0.001466 |
| Opt5 middle | 341 | 0.815254 | -0.001574 |

Neither supported rotation gives source fidelity for the sampled boundary
rows under the uploaded codebook. All **44,596 sampled Opt5 records** have
scale, bias and sign-mask bytes identical to re-encoding the **unscaled integer
codes**. The source's per-group scales are omitted from that encoding. The final
12 outlier bytes were not asserted equal. Applying source group scales before
glyph encoding is necessary for a correct rebuild.

`GlyphTextRuntime` implements all 60 dense text layers, hybrid causal GQA,
raw-gain RMSNorm, half-split partial RoPE, checkpoint scalars, GeGLU, FP32 KV and
the full 262,144-token tied head. There is no shortlist or answer lookup.
Architecture references are the public
[Gemma 4 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
and [compressed-tensors INT4 format](https://github.com/vllm-project/compressed-tensors/tree/main/src/compressed_tensors/compressors/pack_quantized).
Official Transformers BF16 arithmetic equivalence is not claimed.

Independent NumPy dequantization/projection and fused C source projection,
through the same transformer architecture, produced full Hamlet logits with
cosine `0.9999999999996179`, relative L2 difference `8.88212e-7` and identical
argmax. With the supplied chat template, eight source-control steps generated
**William Shakespeare wrote the play *Hamlet*.** France produced
`The capital of France is Paris.<turn|>`; arithmetic produced
`2 + 2 = 4<turn|>`. Fixed-length probes retain post-turn tokens, if any.

The uploaded mixed glyph stream instead generated `kumb kumb` on all four raw
prompts using the actual recovered scalars and direct-mode metadata default.
Sampled batched versus single-token projection differences were below `4.37e-7`
relative L2. These results concern the uploaded mixed stream, not a separate
unavailable all-132-byte control.

## The built candidate fails quality

This is an **uncalibrated ablation**, rebuilt from actual dequantized source
weights using H8. Every linear weight is represented in 36-byte Opt5 records.
The whole embedding/head uses explicit `LUE3` three-bit records. Scaling, GPTQ
and mixed high-tier allocation are not applied; the failure does not establish
that those techniques cannot help.

| Stored component | Bytes |
| --- | ---: |
| Linear stream with header | 4,118,446,096 |
| Complete three-bit embedding/head with header | 539,492,384 |
| Norms and real layer scalars | 2,990,450 |
| Configuration, layout and loaded codebook | 273,543 |
| **Total** | **4,661,202,473** |

Identical raw prompt tokens and full final-softcapped first-position logits were
used for comparison; each path then generated two greedy steps.

| Prompt | Source IDs | Candidate IDs | Cosine | Result |
| --- | --- | --- | ---: | --- |
| Hamlet | `[108,20470]` | `[237456,1638]` | 0.067004 | Fail |
| France | `[107,100]` | `[1067,872]` | 0.267696 | Fail |
| 2 + 2 | `[107,100]` | `[715,715]` | 0.094296 | Fail |
| Opposite of hot | `[7445,236761]` | `[1638,51940]` | 0.447110 | Fail |

The gate requires at least 0.95 cosine and exact control-token agreement. It
wrote the failed report and exited **1**. Raw and chat-formatted prompts are
different contexts and must be compared separately.

Peak host RSS was 2,351,316,992 bytes at a maximum of 11 context tokens, with
file-backed pages only partially resident under concurrent source evaluation.
This is **not** full-model RAM residency evidence. A 200 MB illustrative runtime
reserve leaves room for only about 2.17% of glyphs at 92 bytes with these tables.
That reserve is an assumption; sustained KV/runtime memory needs measurement.
The host is x86-64, with no connected physical Pi 4.

## Reproduce on the pinned downloads

All outputs use new paths. The source checkpoint is needed offline for recovery
and the control, not as part of the proposed Pi candidate.

```bash
python3 -m pip install -r requirements-pi4-inference.txt
python3 neon_runner/build_native.py
python3 scripts/recover_31b_parameters.py \
  --checkpoint /qat/model.safetensors --model-dir /glyph \
  --output /glyph/layer_scalars_qat.npy --report /new/scalars.json
python3 scripts/verify_31b_weight_lineage.py \
  --checkpoint /qat/model.safetensors --model-dir /glyph --report /new/lineage.json
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
python3 scripts/run_31b_text_gate.py --model-dir /glyph --backend qat \
  --qat-native --checkpoint /qat/model.safetensors --output /new/control --tokens 2
python3 scripts/build_31b_lowbit_candidate.py --model-dir /glyph \
  --checkpoint /qat/model.safetensors --output /new/candidate --rotation fwht8
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
python3 scripts/run_31b_text_gate.py --model-dir /glyph \
  --stream /new/candidate/candidate.lanu --layout /new/candidate/candidate.layout.json \
  --embedding-3bit /new/candidate/embedding.lue3 --reference /new/control \
  --output /new/gate --tokens 2
# Expected for the measured uncalibrated ablation: exit 1 (quality failure).
```

For separate chat probes add `--chat --tokens 8` to the source control command.
The NumPy source backend is the default; `--qat-native` requires a Linux C
compiler with OpenMP. This source-model reference is not a candidate LUT kernel.
CI covers software correctness on x86-64 and ARM64; neither is a Pi benchmark.

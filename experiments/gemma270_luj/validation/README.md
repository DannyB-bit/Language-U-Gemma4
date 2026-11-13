# Gemma 3 270M: higher-precision glyphs, external parity and reasoning gates

This experiment builds on [PR #25](../../../../pull/25) and the strict
logit checker in PR #26. It uses the user's *actual* BF16 checkpoint, pinned to
SHA-256 `700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab`.

**Never confuse these independent results:**
1. **Format correctness (PASS):** one LUJ1 glyph contains 256 BF16-restorable
   weights, 20/21 *metadata* token IDs and one checked instruction. Five glyphs
   carry 1,280 weights, 102 token IDs and five instructions. The full model has
   1,047,259 glyphs / 1,023 shards.
2. **Model execution (PASS):** C++17 Gemma 3 runs all 18 transformer layers on
   the reconstructed checkpoint; no Python is used for native inference.
3. **Native-vs-independent PyTorch mathematical parity (PASS on 3 development
   prompts):** full 262,144-logit relative L2 below 4.6e-6; argmax matches.
   This separate validation reference is *not* Hugging Face Transformers.
4. **Official Transformers parity (NOT RUN):** the official 4.55.0 package was
   not available in this container and outbound Python package downloads failed.
   A pinned, local-file-only official validation runner is provided.
5. **General GPT-level reasoning (NOT PROVEN):** 10/28 strict normalized exact
   matches on a new multi-domain local test; no external GPT comparison ran.

## Full-model grouped quantization (native C++)

```sh
make -C experiments/gemma270_luj -B all inference compare_logits
experiments/gemma270_luj/luj_gemma270 selftest
experiments/gemma270_luj/luj_gemma270 quantize-q12 \
  /models/gemma270m.gbf /models/tokenizer.json /new/q12
experiments/gemma270_luj/luj_gemma270 quantize-q14 \
  /models/gemma270m.gbf /models/tokenizer.json /new/q14
experiments/gemma270_luj/luj_gemma270 decode /new/q14 /new/q14.safetensors
```

Available modes: `quantize-q10`, `quantize-q12`, `quantize-q14` and
matching `resume-q*` modes. The old Q8 mode remains the default and unchanged.
Group size is 32; symmetric signed integer grid per group, with BF16
round-to-nearest-even reconstruction. The LUJ1 codec still stores these
**dequantized BF16 bits** via EXACT32, not packed native Q14 codes. This
experiment thus **does not establish compression**; the original GBF1
file is smaller. The suffix is a quantization *source precision*, not a
storage-bit accounting promise.

| Model | Relative weight L2 | Same greedy sequences | LUJ1 bytes |
|---|---:|---:|---:|
| Exact BF16 | 0 | 5/5 | 637,454,658 |
| Group32 Q8 | 0.00531198 | 2/5 | 630,373,072 |
| Group32 Q12 | 0.000185647 | 4/5 | 637,448,527 |
| Group32 Q14 | 0.0000278652 | 5/5 | 637,484,603 |

Q14 round-trip checkpoint SHA-256:
`996fa320819d77b8ee1217fa6ef5088373a0f1645c0b3ec04cdf8111a722311e`
(decoder output SHA matches its own encoded checkpoint).
The five developmental prompts are arithmetic, France, Hamlet, multistep and
minimum. **Q14 still FAILS the strict declared logit gate for France:** source
vs Q14 relative L2 = 0.01994360785, above the example threshold 0.01;
argmax and greedy sequence nonetheless match. Five prompts are not a broad
accuracy demonstration.

## Independent validation-only parity

The C++ runner produces a 262,144-way float32 logit file:

```sh
./gemma3_native /new/exact102.safetensors '2,105,2364,107,3689,563,236743,236778,900,236743,236778,236881,106,107,105,4368,107' 0 \
  --logits-out /new/native.bin
```

With a **separate validation environment** having PyTorch and Safetensors,
run `validation/independent_torch_reference.py --checkpoint
/new/exact102.safetensors --ids <same-ids> --logits-out /new/torch.bin`.
Then run `./compare_logits /new/native.bin /new/torch.bin --min-cosine
0.999 --max-relative-l2 0.01 --require-argmax`.

Three real FP32 source-weight tests (arithmetic/France/Hamlet) passed with
cosine >0.99999999999 and relative L2 below 4.6e-6. This reference
independently reproduces the HF-published Gemma3 mathematical operations;
it is **not the official Transformers library**.

For the **actual official Transformers gate**, install
`transformers==4.55.0` in a validation-only environment, and execute
`validation/official_transformers_reference.py --checkpoint
/new/exact102.safetensors --config /models/config.json --ids <same-ids>
--logits-out /new/official.bin --dtype fp32`. That tool requires SHA-256
identity before local loading and refuses to silently use a different
Transformers version. Compare its logits using the same C++ parity checker.
Then repeat under BF16 with a justified, separately recorded BF16 tolerance.
**Do not mark official parity PASS without a completed official run.**

## Frozen local-language reasoning test

The 28 tasks were authored and their SHA frozen *before first inference*:
`5bc26faf9f100080ea561b20b9f437dc197a13931aab3cf3ce578ca9d11a4256`.
Strict answer-format exact-match result on **unquantized** glyph-restored
Gemma: **10/28 (35.7%)**. Domain results: arithmetic 1/8, factual 2/4,
language 1/4, coding 1/4, logic 3/6, instruction 2/2. The score includes
formatting mistakes as misses, and the recorded output includes substantive
reasoning mistakes. This suite is **now development data**, and is small,
locally authored and not standardized.

```sh
python3 validation/run_unseen_eval.py \
  --suite validation/frozen_reasoning_set.json \
  --tokenizer /models/tokenizer.model \
  --model /new/exact102.safetensors \
  --native ./gemma3_native \
  --sha 700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab \
  --out /new/frozen_results.json
```

The C++ model receives **only tokenized prompts**—never the expected labels.
Python/SentencePiece are used for validation-time token IDs and scoring, not
native inference. An externally collected named GPT baseline can be provided
via `--external-baseline-json` with provenance fields `model`,
`timestamp`, `capture_method`, and exactly all 28 `responses`
(`id`, `answer` pairs). No GPT baseline has been executed here. For any
new training/tuning decisions, design a **new untouched test set** and repeat
the GPT comparison under identical prompt/decoding conditions.

All numerical and scope records live in
`evidence/q12_q14_native_evidence.json`. The model weights and massive glyph
shards are intentionally not stored in Git; they are reproduced from
user-owned, hash-pinned assets.

**Conclusion:** exact glyphs reproduce Gemma 3 270M's native behavior; Q14
shows materially better preservation than Q8 in the short development suite.
Neither encoding nor a more accurate quantizer increases the intrinsic
reasoning capacity of this checkpoint.

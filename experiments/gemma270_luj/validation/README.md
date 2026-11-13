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
3. **Native-vs-official Transformers 4.55.0 parity (PASS in FP32; verified on Raspberry Pi aarch64):**
   `official_transformers_reference.py --dtype fp32` matches native source on all 5 development prompts (`cos >= 0.999999999991`, `relative_l2 <= 4.26e-6`, 5/5 argmax).
   `quantize-q14-norms` (`--preserve-norms`) and `quantize-q12-norms` fix the 1D norm outlier scaling bug discovered on the capital prompt: Q14-norms passes **5/5** (`max relative_l2 = 0.000393` vs HF FP32, reducing weight error to `1.246e-6`), and Q12-norms also passes **5/5** (`max relative_l2 = 0.001883`).
4. **General GPT-level reasoning (NOT PROVEN):** 10/28 strict exact match on `frozen_reasoning_set.json` and 6/30 on `fresh_blind_reasoning_set.json` (identical across Source BF16, Q14, and Q14-norms); no external GPT completions executed yet without `OPENAI_API_KEY`.

## Full-model grouped quantization (native C++)

```sh
make -C experiments/gemma270_luj -B all inference compare_logits
experiments/gemma270_luj/luj_gemma270 selftest
experiments/gemma270_luj/luj_gemma270 quantize-q12-norms \
  /models/gemma270m.gbf /models/tokenizer.json /new/q12_norms
experiments/gemma270_luj/luj_gemma270 quantize-q14-norms \
  /models/gemma270m.gbf /models/tokenizer.json /new/q14_norms
experiments/gemma270_luj/luj_gemma270 decode /new/q14_norms /new/q14_norms.safetensors
```

Available modes: `quantize-q8`, `quantize-q10`, `quantize-q12`, `quantize-q14`, `-norms` variants (or `--preserve-norms` / `--preserve-embedding`), and matching `resume-*` modes. Group size is 32; `--preserve-norms` parses 1D tensors from the pinned SafeTensors header and preserves all `55,936` norm weights bitwise in BF16 (preventing RMSNorm gain outliers such as 1648 from corrupting neighbor group32 scales). The LUJ1 codec still stores **dequantized BF16 bits** via EXACT32, not packed native Q14 codes.

| Model | Relative weight L2 | Strict gate vs HF FP32 | Same greedy sequences | LUJ1 bytes |
|---|---:|---:|---:|---:|
| Exact BF16 | 0 | 5/5 (`rel_l2 <= 4.26e-6`) | 5/5 | 637,454,658 |
| Group32 Q8 | 0.00531198 | — | 2/5 | 630,373,072 |
| Group32 Q12 | 0.000185647 | — | 4/5 | 637,448,527 |
| **Group32 Q12-norms** | **0.00000781547** | **5/5 (`rel_l2 <= 0.001883`)** | **5/5** | **637,448,637** |
| Group32 Q14 | 0.0000278652 | 4/5 (`capital rel_l2=0.01995`) | 5/5 | 637,484,603 |
| **Group32 Q14-norms** | **0.00000124623** | **5/5 (`rel_l2 <= 0.000393`)** | **5/5** | **637,484,624** |

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
`timestamp`, `capture_method`, and all suite `responses`
(`id`, `answer` pairs). You can generate either a blank template or a real
OpenAI completion file (fail-closed on missing `OPENAI_API_KEY`) using:

```sh
python3 validation/collect_openai_baseline.py \
  --suite validation/fresh_blind_reasoning_set.json \
  --model gpt-4o \
  --out /new/gpt_baseline.json
python3 validation/run_unseen_eval.py \
  --suite validation/fresh_blind_reasoning_set.json \
  --tokenizer /models/tokenizer.model \
  --model /new/q14_norms.safetensors \
  --native ./gemma3_native \
  --sha 2b64ad286431271106f87e7f94fb41269a8c890f2ec16aaf38a5f8ba3ece2986 \
  --external-baseline-json /new/gpt_baseline.json \
  --out /new/fresh_with_gpt.json
```

No GPT baseline has been executed here (`6/30` strict exact match on `fresh_blind_reasoning_set.json` for source BF16, Q14, and Q14-norms). For any
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

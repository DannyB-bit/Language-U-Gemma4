# Gemma 3 270M: real model → LUJ1 glyphs → native inference

**Status:** Full-lossless checkpoint round-trip PASS; Q8/Q10/Q12/Q14 and 1D-norm-preserved Q12-norms/Q14-norms round-trip PASS; official `transformers==4.55.0` FP32 numerical parity PASS (5/5 strict gate on Q14-norms, Q12-norms, and exact source on Raspberry Pi). **GPT-level reasoning is not established** (10/28 frozen suite, 6/30 fresh blind suite; GPT baseline collection helper ready in `validation/collect_openai_baseline.py`).

This is a **C++17-only inference and conversion implementation**, using the actual Gemma 3 270M BF16 weights supplied by the user (`TheAiCollectiveART/gemma-3-270m-it-bucket`). It is a separate model from the Gemma 4 31B experiments elsewhere in this repository. The source checkpoint, tokenizer assets and full glyph shards are intentionally NOT included in Git; obtain them from the rightsholder/source and check their hashes.

## Layout and model binding

- 268,098,176 model weights, 236 BF16 tensors, 18 decoder layers, 262,144-token vocabulary.
- 1,047,259 glyphs × at most 256 weights (last glyph has 128); 1,023 shards of at most 1,024 glyphs each.
- 20/20/20/21/21 **valid 18-bit token IDs** per consecutive five glyphs (102 total). Tokens currently bind *metadata*, not five requests, an independent tokenizer or 102 free language semantics.
- Each glyph carries one 22-byte LUJ1 instruction for flat offset + valid count. These instructions are validated layout descriptors; they do **not** perform the neural transformer attention or magically create reasoning. The C++ transformer runtime executes the actual neural layers after decoding.
- The weight lane uses existing **LUJ1 EXACT32** framing and built-in SHA-256 integrity; full-lossless reconstruction preserves the *original BF16 bits*. Tokenizer identity is SHA-256-bound separately.
- All five-glyph groups carry five instructions, one per glyph. One five-glyph unit is **1,280 weights**, not a 270M checkpoint.

Source checkpoint SHA-256: `700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab`
Tokenization JSON SHA-256: `7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c`

## Build (GCC/Clang; no Python)

```sh
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libzstd-dev
make -B all inference compare_logits
./luj_gemma270 selftest
```

The tools use only standard C++, OpenMP for faster CPU projection, zlib/zstd for storage, and OpenSSL for SHA-256.

## Reproduce exact checkpoint recovery

`model.gbf` is the user's genuine GBF1 source model, not a dummy archive.

```sh
./luj_gemma270 encode /models/gemma270m.gbf /models/tokenizer.json /new/exact_glyphs
./luj_gemma270 verify /new/exact_glyphs
./luj_gemma270 decode /new/exact_glyphs /new/restored.safetensors
sha256sum /new/restored.safetensors   # expected: 700b710a9a99c295...

# Required 102-token / five-glyph format; retains every BF16 weight bit:
./luj_gemma270 upgrade102 /new/exact_glyphs /new/exact102_glyphs
./luj_gemma270 verify /new/exact102_glyphs
./luj_gemma270 decode /new/exact102_glyphs /new/exact102.safetensors
sha256sum /new/exact102.safetensors # SAME source SHA256
```

The encoder supports `resume` for complete, already-written shards. Legacy full exact mode stores 20 token IDs per glyph. **Use `upgrade102`** to transform a verified legacy exact stream into the requested 20/20/20/21/21 mode without changing any weight bits. The experimental Q8 modes below already use that distribution. No source checkpoint is needed for the decoder; it verifies each shard, token metadata, per-shard decoded FP32 weight digest, and the complete output SHA.

## Quantize while retaining current LUJ1 EXACT32 codec

```sh
./luj_gemma270 quantize-q8 /models/gemma270m.gbf /models/tokenizer.json /new/q8
# or protect exact tied embedding/head weights:
./luj_gemma270 mixed-q8 /models/gemma270m.gbf /models/tokenizer.json /new/mixed_q8

# If a run stopped, retain completed shards and resume the matching mode:
./luj_gemma270 resume-q8 /models/gemma270m.gbf /models/tokenizer.json /new/q8
./luj_gemma270 resume-mixed-q8 /models/gemma270m.gbf /models/tokenizer.json /new/mixed_q8

./luj_gemma270 decode /new/q8 /new/q8-restored.safetensors
./luj_gemma270 verify /new/q8
```

This **Q8 group32** experiment applies symmetric INT8 quantization with one source-derived FP32 scale per group of 32, dequantizes values to BF16, then packs those BF16 bytes with the existing LUJ1 EXACT32 codec. The output is not a native low-bit inference kernel. Quantized checkpoint hashes differ from the original, as expected. The Q8 error is real, not papered over by using original weights at inference.

Use `run_from_glyphs.sh /new/q8 '<token IDs>' 16` to independently decode in a new temporary directory and execute the restored model using only C++ and a shell script (intermediate disk file is explicit).

## Verified result — development probes only

| Representation | LUJ shard bytes | Weight relative L2 error | First-token agreement (5 prompts) | Whole greedy token sequence agreement (5 prompts) |
|---|---:|---:|---:|---:|
| Exact BF16 reconstructed (legacy 20-token metadata) | 636,197,949 | 0.0 | 5/5, by byte-identical source weights | 5/5, by byte-identical source weights |
| **Exact BF16 20/20/20/21/21** | **637,454,658** | **0.0** | **5/5, by byte-identical source weights** | **5/5, by byte-identical source weights** |
| Q8 group32 | 630,373,072 | 0.00531198 | 3/5 | 2/5 |
| Mixed embedding/head BF16 + Q8 elsewhere | 635,366,421 | 0.00530948 | 3/5 | 2/5 |

The original GBF1 archive itself is **504,874,104 bytes**. Thus Q8-in-EXACT32 is **not** a better compressed representation than the original source container; the explicit token and instruction lanes add nontrivial overhead.

Quantized checkpoint SHA-256:
- Q8: `98290a484bec5501444a3e966521062ca53cb13f55143d3bfc7883a1d7a4cb14`
- Mixed: `cddf1949fae6b778462e11f532259006138cdaa41a03cbf1852251a0acf6f610`

Native `gemma3_native` verifies the SHA256 of its entire input file before inference and uses all 18 transformer decoder layers, hybrid GQA, RoPE, raw-gain RMSNorm, GeGLU/FFN and a full 262,144-way tied output head. The reference has 32-bit accumulation and an exact BF16 weight source; it does **not** claim bit-identical math with CUDA/PyTorch/Transformers. The CLI accepts `--expected-sha HASH` for each explicit candidate identity and `--logits-out new-file` for comparison, and stops on EOS 1 or 106. The model decoding path is not streaming glyph-native computation: it reconstructs an intermediate SafeTensors file before inference.

The following five chat-prefilled prompts were run: `What is 2+2?`, `What is the capital of France?`, `Who wrote Hamlet?`, a multi-step arithmetic instruction, and a min-of-two problem. Native full-vocabulary *first-position* cosine between source and Q8 ranges from 0.83719 to 0.99124; importantly, even relatively high cosine does not imply equal generated tokens. The math/France generations matched, while Hamlet, multi-step, and min differed in some or all tokens. These five cases are **development cases**, not a statistically meaningful general-language benchmark and not independent GPT assessment. See `evidence/` for raw numerical output.

### Strict model-fidelity gates for future work

1. Freeze source hashes, config, tokenizer and chat templates. Never use hidden oracle hints or prompt-specific answer rules.
2. Prove format correctness with SHA-256, corruption tests, declared size and valid token lane boundaries.
3. Check end-to-end native first-token full-vocabulary logits and multi-token generation vs lossless source with the same prefill, context and decoding.
4. Run actual official Transformers BF16 reference under a controlled environment **only as an independent comparison**; it has not been run here. Match logits within a declared tolerance and account for BF16 vs FP32 arithmetic.
5. Run a new unseen multi-domain evaluation with a named, fixed GPT baseline, count all errors and abstentions, and publish confidence intervals and control ablations. This is necessary before making any GPT-comparable reasoning claim.

**Non-claims:** No universal compression, no automatic GPT intelligence from glyph count, no exact official Transformers parity, no 36B packing of arbitrary 256 full-precision weights plus tokens, no physical Raspberry Pi 4 residency proof.

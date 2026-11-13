# LUJ1 native five-glyph neural planner — semantic repair experiment

This is an **independent native C++17** implementation of the checked-in five-glyph
QAT4 bank from [Language-U-Gemma4 PR #22](https://github.com/DannyB-bit/Language-U-Gemma4/pull/22).
It retains the exact **5 glyphs = 1,280 weights + 102 token IDs + 5 instructions**,
1,269 bytes, SHA-256 `59841ffb1420237ea595a7f0f3645b407917170732b38fd554a4117825f3acac`.

Build and run (Linux, OpenSSL development headers, GCC or Clang):

```sh
make
make test
./luj_native_planner --bank ../planner/planner.luj --question \
  'Start with 547; take the lower of this amount and 193; divide the running total by 1; take 449 away from the total' \
  --chunk-out /tmp/plan.luj
```

No Python, NumPy or internet is required by the native executable. Internally it:
1. Validates the original LUJ1 body, tokenizer fingerprint and frozen model digest.
2. Recovers every INT4/group32-BF16 weight and authenticates the full FP32 decoded tensor hash.
3. Decodes and checks all five 18-bit token-ID lanes and the architecture blueprint.
4. Recreates PR #22's signed BLAKE2s word/character features and 32→tanh(32)→8 neural scoring.
5. Applies explicit semantic constraints only to unambiguous min/max and remainder vocabulary, rather than accepting the model's confident misread of “take the lower of”. Ambiguous input is rejected.
6. Executes bounded int64 arithmetic; serializes an updated executable LUJ1 plan retaining the same weights and token metadata, with renewed integrity checksum.

**Evidence boundary:** deterministic semantic constraints are additional code, *not* newly learned language skill. The output remains a grammar-restricted arithmetic interpreter and does not prove GPT-level reasoning, arbitrary English comprehension, or a general-purpose LLM. A fresh unseen-language evaluation is necessary even if previously recorded arithmetic prompts improve.

**Known limitations:** only the frozen five-glyph QAT4 bank is accepted (not the Gemma 3 270M or Gemma 4 31B full checkpoints); compilation needs OpenSSL libcrypto; `__int128` requires GCC/Clang. On its own this does not implement full transformer inference. The program blob does not authenticate the external user question and is not a signature.

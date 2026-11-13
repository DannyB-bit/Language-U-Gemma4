# Native glyph integration checkpoint

`glyph_bridge.cpp` connects the C++ closed grammar to LUJ1. It validates the
five-record QAT4 bank, checks body and recovered-FP32 source hashes, decodes all
1,280 weights, preserves token/weight lanes, replaces blueprint instructions
with parsed operations, decodes and executes those serialized instructions and
compares the result before emitting a rehashed program chunk. Tokens remain
model metadata; the question is external. The parser does not use decoded
weights to infer meaning. This is not neural Gemma inference.

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror glyph_bridge.cpp -lcrypto -o glyph_bridge
./glyph_bridge ../joint_glyphs/planner/planner.luj 'Start with 12; subtract the total from 27' /tmp/plan.luj
```

Requires OpenSSL development headers/library. Supported bank codec is QAT4
only; other codecs and non-blueprint inputs are rejected. Output is an executable
program, not a reloadable model blueprint. SHA-256 is integrity, not authenticity.

## Attached Gemma checkpoint

Read the actual safetensors header and hashed the complete attachment:
- 236 BF16 tensors; 268,098,176 stored parameters.
- Model SHA-256: 700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab
- Config: Gemma3ForCausalLM / gemma3_text; 18 layers; hidden width 640;
  intermediate width 2048; four attention heads, one KV head; vocabulary 262144.
- Attachment names alone do not authenticate remote model provenance.

The attached Gemma checkpoint has NOT been converted into glyphs or run in this
component. All 268M parameters are distinct from the tiny 1,280-weight planner.
No quality, compression ratio, native generation or GPT equivalence is claimed
for the attachment. No attached weights were uploaded into the code repository.

Remaining work: native Gemma inference/tokenization and checkpoint conversion;
BF16-preserving glyph codec or measured quantization; original-versus-restored
logit/token parity; independent language/reasoning evaluations against a named
reference model. No implementation or benchmark evidence yet closes those gates.

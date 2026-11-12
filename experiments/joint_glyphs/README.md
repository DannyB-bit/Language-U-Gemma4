# Joint token/weight glyph round-trip proof

The fixed target is **256 weights per glyph** and **20.4 tokens per glyph on
average**. This experiment carries 102 actual Gemma tokenizer IDs in five
records containing 20, 20, 20, 21 and 21 tokens, plus 1,280 actual source weights.

LUJ1 is a new combined record envelope, **not** a claim that tokens and exact
weights now fit inside the existing 36-byte or 132-byte weight-only ABI.

## Measured result

| Weight lane | Bytes for all five complete glyphs | Tokens exact | Source weights exact |
| --- | ---: | --- | --- |
| Source QAT INT4 codes + BF16 group scales | 1,269 | Yes | Yes, FP16 and FP32 |
| Standard 36B OPT5 | 729 | Yes | No |
| Standard OPT5 + exact FP16 repair | 2,520 | Yes | Yes |
| Standard OPT5 + exact FP32 repair | 2,819 | Yes | Yes |

The source is pinned `google/gemma-4-31B-it-qat-w4a16-ct` revision
`52f3f65bc7a02d555763bc923bd1d9094898219d`: row 0, columns 0..1279 of layer-0
query weights. Exactness means the dequantized **QAT checkpoint's** BF16-rounded
values, represented in FP16/FP32. It does not recover the earlier pre-QAT
training checkpoint. Original arbitrary FP32 values are additionally covered
by the exact-repair codec's randomized unit test, with no promised size ratio.

The QAT lane stores 128 bytes of packed INT4 codes and eight BF16 group scales
(16 bytes) per 256-weight block. It uses the source quantization structure;
no semantic phrase is assumed to predict the missing weight bits.

For a 20-token glyph the explicit byte ledger is:

| Part | Bytes |
| --- | ---: |
| Exact source QAT weights | 144 |
| Twenty 18-bit token IDs | 45 |
| Executable integer instruction | 22 |
| Record framing | 20 |
| Record subtotal | 231 |

A 21-token record is 234 bytes. The five-record stream additionally has a
108-byte shared header, yielding `3*231 + 2*234 + 108 = 1269` bytes. A standalone
20-token record with its own stream header would be 339 bytes. Each record
always decodes exactly 256 weights. No phrase dictionary is needed in this
literal-token mode. Text decoding requires the shared tokenizer; its full
file size and SHA-256 are disclosed in the evidence, separately from the stream.

Replacing each record with a short dictionary ID would still require storing
or transmitting its dictionary. For example, five uint16 IDs take 10 bytes
but referring to these five exact records needs their 1,269-byte dictionary,
so cold storage is at least 1,279 bytes, before any additional lookup framing.
This is accounting, not a separately implemented dictionary codec.

## Linked reasoning chunk

The proof joins the recovered tokens into an authored English arithmetic
explanation and executes an explicitly authored integer dependency graph:

1. `12 * 8 = 96`
2. `result[0] - 27 = 69`
3. `result[1] + 15 = 84`
4. Assert `result[2] == 84`
5. Assert `result[2] >= 0`

The decoder returns `The verified result is 84.` Source and restored weights
also produce identical FP32 GEMV outputs in all exact modes. The arithmetic
answer is computed by the typed interpreter, **not by the 1,280 model weights**.
The prose is supplied before encoding; this is not a language-model generation
test. Numeric agreement does not establish that arbitrary prose entails the
bytecode. A trained full-model consumer and independent language/reasoning
evaluation are still required for any GPT-level claim.

## Reproduction

From the repository root, prepare the same public source data as the correction
experiments, or reuse a verified existing data directory:

```sh
python3 -m pip install -r experiments/packed_corrections/requirements.txt
export LANGUAGE_U_EXPERIMENT_DIR="$PWD/experiments/packed_corrections/data"
python3 experiments/packed_corrections/prepare_public_inputs.py
LANGU_CALIB_PREPARE_ONLY=1 python3 experiments/packed_corrections/probe_real_activations.py
python3 experiments/joint_glyphs/run_joint_proof.py \
  --inputs "$LANGUAGE_U_EXPERIMENT_DIR" \
  --output "$PWD/experiments/joint_glyphs/proof"
python3 -m unittest discover -s tests -p test_joint_glyphs.py -v
```

The output directory must be new. Four `.luj` files, JSON receipts and the
recovered English text are written. No downloaded model weights or private
repository sources are committed. Public inputs retain their upstream terms.

## Format and rejection rules

The little-endian stream header is `<4sBBHI32s32s32s`: magic LUJ1, version 1,
18 token bits, zero flags, record count, tokenizer SHA-256, source FP32-weight
SHA-256, and body SHA-256. Record headers are `<IBBHIII`: sequential index,
weight codec, token count, zero reserved, weight-lane length, token-lane length,
instruction length. Token bits are packed least-significant first with zero
padding. SHA-256 provides integrity checks, not sender authentication.

Weight codecs: 1=source QAT4/group32; 2=approximate standard OPT5; 3=OPT5 plus
exact FP16 repair; 4=OPT5 plus exact FP32 repair. Repairs XOR the baseline's
target-dtype bit patterns with the original, byte-plane shuffle the XOR bytes,
and optionally zlib-compress them. A raw repair fallback bounds expansion and
keeps incompressible finite inputs representable. Exact source hashes are
verified after decoding all-exact streams. Approximate streams retain their
reference source hash as metadata, without claiming to match it.

Instructions are `<BBhhqq`: opcode, zero flags, two signed reference indices,
two signed 64-bit immediates. Reference -1 selects an immediate; other references
must point backward and have a zero unused immediate. Opcodes are ADD, SUB,
MUL, ASSERT_EQ and ASSERT_GE. Overflow, failed constraints, malformed lengths,
unknown formats, bad padding, corrupted bodies and lossy requested output
casts are rejected. This is a small executable extension, not native codegen.

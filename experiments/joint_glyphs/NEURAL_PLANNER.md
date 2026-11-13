# Learned planning from the five-glyph bank

This adds a real trained neural component to the previous storage proof. All
1,280 learned weights are recovered from five 256-weight glyphs and used to
choose arithmetic operations. No full LLM or hidden neural checkpoint is used.
**GPT-level reasoning is not established.** This is an eight-class operator
classifier plus a constrained integer interpreter, not a generative transformer.

## Architecture and what the glyphs contain

The neural network is `32 -> tanh(32) -> 8`, with 1,024 first-layer and 256
second-layer weights. Inputs are fixed signed word/character-hash features;
numeric operands are masked for classification and parsed separately. Training
uses NumPy Adam. The compact model uses group-32 INT4/BF16 quantization with a
straight-through gradient estimator during training. Source/master FP32 weights
are not recovered from INT4; the exported quantized checkpoint is recovered
exactly into FP32 for execution.

The complete bank is **1,269 bytes**: five weight payloads of 144 bytes, 102
metadata tokens in 20/20/20/21/21 groups, five 22-byte instructions, record
framing and the stream header. The instructions encode a checked model
blueprint `[32,32,8,1,2]`: input/hidden/output dimensions, tanh activation ID and
feature-version ID. The loader rejects an incompatible blueprint.

**The 102 tokens describe the model; they are not the current user question.**
Questions are separate inputs and their UTF-8 size is counted separately in
evaluation records. The metadata English does not itself generate intelligence.
The learned weights choose operations; numbers, clause boundaries and program
execution use explicit code. Python, NumPy, the featurizer and interpreter are
not included in the 1,269-byte model count. Inference does not need a tokenizer
file or network; recreating the metadata tokens during training uses the shared
Gemma tokenizer whose size/hash is in the manifest. No Pi RAM/speed claim is made.

## Fixed evaluation result

| Measurement | Result |
| --- | ---: |
| Training / validation / test clause templates | 80 / 32 / 48 |
| Selected validation operation accuracy | 31/32 (96.875%) |
| Held-out operation accuracy | 44/48 (91.67%) |
| Zero-weight control accuracy | 6/48 (12.5%) |
| Shuffled-weight control accuracy | 5/48 (10.42%) |
| Generated three-to-five-step test problems | 256 |
| Correct final answers, counting refusals as misses | 186/256 (72.66%) |
| Exact correct operation sequences | 166/256 (64.84%) |
| Answered / refused | 212 / 44 |
| Wrong answers accepted by arithmetic checks | 26 |
| Accuracy among answered cases | 186/212 (87.74%) |
| Rejected examples in small unsupported-input suite | 13/13 |

These are locally authored, template-disjoint development examples with a
fixed random composition seed, not an external benchmark. Operation selection
is neural, but grammar is restricted to `Start with <integer>; <clause>; ...`.
Arithmetic validity is not semantic correctness. For example, the model
misinterprets "take the lower of this amount and 193" as subtraction. The
resulting integer program can execute correctly and still answer the question
incorrectly. This failure remains in the checked-in evidence.

The first feature variant obtained only 23/32 validation accuracy. Removing
generic function words before feature hashing improved validation to 31/32;
both choices occurred before the first held-out evaluation. Three initialization
seeds, two precision modes and four saved training steps were compared using
validation. Selection prioritizes validation accuracy, then compact QAT4 mode,
then validation cross-entropy. The fixed 0.55 softmax threshold is not a calibrated
probability that a complete answer is correct.

No test answers or operation labels enter `solve`. Labels are used only by the
external evaluator after inference. After the first evaluation, the bank's
no-op instructions were replaced with architecture descriptors; no weight,
feature rule, selection or threshold changed. Re-running the same suite after
that packaging change produced identical metrics. Both receipts are retained.
This test set is now development data; later learning changes need fresh tests.

## Run the stored model

```sh
python3 experiments/joint_glyphs/solve_with_glyphs.py \
  --model experiments/joint_glyphs/planner/planner.luj \
  --question 'Start with 12; multiply by 8; subtract 27; add 15' \
  --chunk-out /tmp/learned-plan.luj
```

The restored network chooses multiply/subtract/add. The compiler writes those
instructions into the five-glyph structure; decoding and executing the result
returns 84. The output explicitly sets `semantic_correctness_certified=false`.
Program glyphs cannot be loaded as model-blueprint glyphs. The input question's
SHA-256 is returned separately; the `.luj` program alone does not contain or
authenticate the input question. Integrity hashes are not signatures.

Supported operations: add, subtract, multiply, exact integer divide, minimum,
maximum, nonnegative integer power (exponent <=63), and remainder. Unsupported
grammar/negation, low confidence, inexact division, zero divisors and int64
overflow are rejected. These checks do not guarantee rejection of every
unsupported interpretation.

## Reproduce training and evaluation

```sh
export OPENBLAS_NUM_THREADS=2
python3 experiments/joint_glyphs/train_planner.py \
  --tokenizer /path/to/verified/tokenizer.json --output /tmp/new-planner
python3 experiments/joint_glyphs/evaluate_planner.py \
  --model-dir /tmp/new-planner --output /tmp/new-planner/evaluation.json
python3 -m unittest discover -s tests -p 'test_*glyph*.py' -v
```

Use new output paths. Training reads training and validation examples only;
the test templates are evaluated in the separate script. Candidate receipts,
source hashes, final bank and all 256 outcomes are under `planner/`. Small
floating-point differences across NumPy/BLAS platforms may alter training;
the checked-in model is the frozen object used for the recorded evaluation.

## Research context and next acceptance gate

Separating learned program proposal from execution follows established
[program-aided language modeling](https://proceedings.mlr.press/v202/gao23f.html).
This experiment's particular contribution is a complete tiny trained planner
carried and executed through the existing token/weight glyph envelope. It does
not claim the broader method is new or reproduce PAL's large-model results.

The unresolved obstacle is reliable interpretation and generalization, not
whether the bytes can be decoded. Next work must use new language forms and
independent tasks, compare against a named reference model under the same
conditions, and expand learned parsing/generation beyond this grammar. Do not
claim GPT equivalence from these arithmetic results or from format correctness.

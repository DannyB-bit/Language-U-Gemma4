# Gemma 3 270M: fail-closed logit parity gate

A five-glyph LUJ1 group holds 1,280 weights, 102 metadata token IDs and five instructions. This layout preserves model information; it does not itself produce GPT-level intelligence.

This native C++ gate compares **all 262,144 next-token FP32 logits** from separately generated reference and candidate binary files. It emits JSON for cosine, relative L2, maximum absolute error, KL divergence and argmax agreement. Exit code 0 means gate pass, 2 means numerical mismatch, 1 means malformed input/arguments.

Run:

```sh
make compare_logits
./compare_logits original_logits.bin glyph_logits.bin --min-cosine 0.999 --max-relative-l2 0.01 --require-argmax
bash tests/test_strict_parity.sh
```

The tolerances are deliberately illustrative strict experimental gates, **not** a claim of official-Transformers equivalence.

**Real checkpoint negative control:** The existing original-vs-INT8 Q8 arithmetic first-position logits have cosine 0.991237867877 and relative L2 0.177260801991. The gate fails even though both runs share the same first greedy token. Thus greedy agreement on one question cannot mask poor logit parity.

This gate checks quantization fidelity within the native implementation. Full equivalence with Hugging Face Transformers still needs independently generated official model outputs using the same source checkpoint, input IDs, chat template, attention implementation, dtype and decoding rules. GPT-level reasoning requires a separate named GPT baseline and fresh multi-domain tests, not mere encoding.

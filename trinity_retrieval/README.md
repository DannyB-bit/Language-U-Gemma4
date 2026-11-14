# Trinity Retrieval (`trinity_retrieval/`)

Consolidated **Nature 2** retrieval, octonionic $S^7$ compression, and edge-streaming modules for **Language-U & Gemma 4** (extracted cleanly without SQLite databases, external weight binaries, or host-specific paths).

## Core Modules

- `language_u_octonion.py`: Normalized Sylvester-Hadamard $H_8$ rotation and `768-D -> 8-D` $S^7$ projection (`6,027,264 B -> 62,784 B`, exact `96.0x` index compression at 1,962 chunks).
- `affine_hadamard_h64.py`: Order-64 $H_{64}$ Walsh-Hadamard transform and 1-bit (`uint64`) hypercube packing.
- `noise_stability_sieve.py`: 3-scale query stability sieve (`PASS` / `FORCE_WEAK` / `REJECT_MISS`) against dense OOD false positives.
- `roadmap_hybrid_v5_gemma.py`, `battery_contract.py`, `test_35_hardware_suite.py`: 6-channel Hybrid RRF engine (`get_engine()`, `store_revision()`, `clear_engine_cache()`, lazy `_ensure_gemma_model()` loading) and **Two-Tier Trust Law** (`STRONG` / `WEAK` / `MISS` with `STRONG_GATE = 0.302`, `WEAK_GATE = 0.280`, `LEX_STRONG_FLOOR = 0.080`). Note: querying a pre-built SQLite FTS5 index requires `$TRINITY_RETRIEVAL_DATA` or local index artifacts (`roadmap_fts5.sqlite`, `roadmap_v2.npz`, `roadmap_v2_meta.json`); `swe_navigation_benchmark_results.json` records verified summary metrics.
- `formal_verification/TrinityGates.lean`: Machine-checkable Lean 4 contract specifying `evaluateTrustGate` and proving `zero_confabulation_guarantee`, `noise_refusal_guarantee`, and `weak_requires_opt_in`.
- `gemma2_paths.py`, `gemma2_embedding.py`: Cross-platform resolver and mean-pooled PyTorch embedder for `google/embeddinggemma-2` (resolves `$GEMMA2_WEIGHTS_DIR`, `$GEMMA2_MODEL_PATH`, or Hugging Face Hub).
- `air_trinity_streamer.py`, `edge_lora_streamer.py`, `sparse_octonion_streamer.py`, `edge_self_test.py`: Memory-constrained layer streaming, LoRA gradient flow, and sparse octonion routing.

## Verification

```bash
python trinity_retrieval/edge_self_test.py
python trinity_retrieval/language_u_octonion.py
python trinity_retrieval/affine_hadamard_h64.py
python trinity_retrieval/noise_stability_sieve.py
python -m unittest discover -s tests/retrieval -v
```


# run270 — native Rust (zero-Python) Gemma 3 270M glyph-stream inference + parity receipts

Branch: `amara/run270-rust-parity-270m`

## What this adds
- `encode_model.rs` — safetensors → 132 B OPT4 dual-stage RVQ glyph stream (126 projections, 391,680 glyphs, 100,270,080 weights; embed kept as raw bf16 sidecar, norms as fp32 sidecar). 4-thread `std::thread::scope`.
- `run270.rs` — config-driven inference runtime: greedy + sampled, sliding-window GQA, per-layer RoPE base (10k local / 1M global), Gemma `(1+w)` RMSNorm, embed×√d, tied head. Two weight modes:
  - default: glyph stream decoded to RAM (52 MB stream → ~401 MB f32) + native 4-thread matvec — the NEON `gemv_opt4` assumes d_in % 256 == 0, which 270m's d=640 violates, so the stream path is RAM-decoded.
  - `--exact`: bf16 weights straight from safetensors (parity harness mode).
- `probe270.rs` — per-projection decode-vs-source verifier (re-encode byte-diff + per-glyph cos histogram).

## Measured receipts (Pi 4, this machine, 2026-11-13)
- Checkpoint: `/mnt/SHE-VAULT/gemma-3-270m-it/model.safetensors` sha256 `700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab` (matches PR #26 runner pin; 64/64 curl chunks, 0 zero-blocks).
- Encode: stream sha256 `16c0d11e56eee64106e14dc0fe261a9fbf8522303c5f779583aca1d86db7fda7`, overall cos 0.985318, min proj cos 0.985036, census_R=0.
- **Parity (exact mode vs independent C++ fp32 reference, 17-token arithmetic prompt, full 262,144 logits):** cos = 0.999999999999, relL2 = 0.000107%, max|d| = 1.16e-4, top-1 match (236778). Greedy ids identical to reference: `[236778,900,236743,236778,578,236743,236812,107]` → "2 + 2 = 4". 0.64 s/tok.
- **Fidelity boundary (honest):** the 132 B RVQ stream (per-tensor decode cos ~0.89–0.95) does NOT preserve generation on 270m (glyph-mode run diverges). Consistent with Q8 failing and LUJ1/Q14 passing: lossless tiers preserve behavior; OPT4-RVQ is a storage tier, not an inference-faithful one at 270m.

Logs: `run270_exact_gen.log`, `run270_glyph_gen.log`, `ref_gemma3_native_arith.json`.

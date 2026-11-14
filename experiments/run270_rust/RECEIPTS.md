# run270 — standalone zero-dependency Rust Gemma 3 270M runtime & codec receipts

Branch: `amara/run270-rust-parity-270m`

## Package layout & build

Standalone Cargo crate (`experiments/run270_rust/Cargo.toml`, zero external crate dependencies; Rust stdlib only):

- `src/codec.rs` — OPT4 132 B glyph codec (coherent glyph-wide `fp16` `(s1, s2)` with joint 2x2 least-squares solve and strict decode-verified cosine/SNR), IEEE `binary16` <-> `fp32`, and `bf16` -> `fp32`.
- `src/json.rs` — iterative depth-bounded JSON parser with exact `i64`/`u64` integer preservation (avoiding `f64` precision loss on large SafeTensors offsets).
- `src/safetensors.rs` — cross-platform SafeTensors reader (`pread` via `FileExt` on Unix, cloned seek+read fallback elsewhere) with header/offset bounds checks.
- `src/sha256.rs` — streaming SHA-256 helper for byte-verified artifact receipts.
- `src/tokenizer.rs` — Gemma SentencePiece/BPE tokenizer loaded from `tokenizer.json`: longest-match `added_tokens` scanning (`<bos>=2`, `<eos>=1`, `<start_of_turn>=105`, `<end_of_turn>=106`, `\n=107`), `Replace(' ' -> '▁')`, `<0xNN>` byte fallback, and `decode()` restoring `'▁'` -> `' '`.
- `src/math.rs` — Gemma `(1 + w)` RMSNorm, Q/K norm, dual-base RoPE (`10k` local sliding window / `1M` global), GQA attention decode, tanh-approx GELU, SplitMix64 RNG, and top-k/top-p sampler with repetition penalty.
- `src/bin/encode_model.rs` — `model.safetensors` -> `stream_132b.bin` (126 projections, 391,680 OPT4 glyphs) + `embed_tokens.bin` (BF16 sidecar) + `norms.f32bin` / `norms.json` + `layout.json` + `receipts.json`.
- `src/bin/probe270.rs` — decodes every glyph in `stream_132b.bin` back to F32, checks global `tensor_cos`, per-glyph cosine distribution, `g0_reencode_byte_diffs`, and exits non-zero if any projection falls below `min_cos`.
- `src/bin/run270.rs` — standalone inference runtime (`--exact` reading `model.safetensors` directly with only `--model --config --tokenizer`, or OPT4 glyph stream + sidecars decoded to RAM and executed with 4-thread FP64-accumulated matvec).

Build & test:

```sh
RUSTFLAGS="-D warnings" cargo test --manifest-path experiments/run270_rust/Cargo.toml --release
RUSTFLAGS="-D warnings" cargo build --manifest-path experiments/run270_rust/Cargo.toml --release --bins
```

## Review defects found & corrected

1. **Unbuildable standalone scripts -> Cargo crate + CI:** Added `Cargo.toml`, `src/lib.rs`, shared modules (`codec`, `json`, `math`, `safetensors`, `sha256`, `tokenizer`), `src/bin/*.rs`, removed redundant uncompilable root `.rs` files, and added `.github/workflows/run270-rust.yml` (`ubuntu-24.04` & `ubuntu-24.04-arm`).
2. **Silent OPT4 `(s1, s2)` encoder/RAM mismatch:** Prior `encode_model` solved separate `(s1, s2)` scales per octant while overwriting glyph header `[0..4)` across octants `0..31` (leaving octant 31's scales on disk, degrading real disk cosine to ~`0.888–0.915` despite logging `~0.985` from in-loop RAM estimates). Fixed to solve one coherent glyph-wide `(s1, s2)` via joint 2x2 least squares and compute metrics solely from `decode_opt4_glyph(payload, cb)`.
3. **Tokenizer mismatch vs Gemma 3:** Replaced GPT-2 `bytes_to_unicode` + missing `added_tokens` logic with `tokenizer.json`-faithful handling (`<bos>`, special tokens, `'▁'`, `<0xNN>` byte fallback), enabling `--prompt-text` to match HuggingFace token IDs verbatim (`17/16/13/34/24` tokens on the 5 parity prompts).
4. **Misleading CLI & runtime behavior:** `--exact` no longer requires `--stream`/`--layout`/`--codebook`/`--embed`/`--norms`; removed dead `--so` dlopen stub and misleading `[run270] NEON accel loaded` log line; removed unnecessary `unsafe` FFI and `transmute` casts.
5. **Cross-platform I/O & JSON numeric precision:** Replaced Unix-only `FileExt` requirement with `#[cfg(unix)]` / `#[cfg(not(unix))]` abstraction, enforced exact `i64`/`u64` offset parsing in `json.rs`, and replaced panicking `.unwrap()` on `--ids` / tensor lookups with clean `Result` errors.

## Measured verification receipts (Raspberry Pi 4 ARM64, `rustc 1.98.1`, 2026-11-13)

- **Checkpoint:** `/mnt/SHE-VAULT/gemma-3-270m-it/model.safetensors` SHA-256 `700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab`; codebook `s7_codebook_2048x8.bin` (`2048 x 8` f32 unit-norm antipodal).
- **Unit tests & release build:** `cargo test --release` under `RUSTFLAGS="-D warnings"`: **11 / 11 passed**, `0` warnings.
- **Encode (`encode_model`):** `projections=126`, `glyphs=391680`, `weights=100270080`, `census_R=0`, `stream_bytes=51701776`, `overall_decode_cos=0.971763`, `min_cos=0.961642`, stream SHA-256 `93bdf60fa18d234367c482025a69da79b4e310deef31c340fd2ef478df5d4e49`.
- **Decode verifier (`probe270 ... all 0.94`):** **126 / 126** projections checked (`PROBE270_PASS checked=126`), `g0_reencode_byte_diffs=0/132` across every projection, global `tensor_cos` range `0.948984..0.974944` (mean `0.971205`; worst global cosine `model.layers.5.self_attn.k_proj.weight` `0.948984` fails a `0.965` per-tensor gate and passes `0.94`).
- **Exact standalone parity (`run270 --exact --prompt-text`, 262,144 FP32 logits, strict gate `--min-cosine 0.999 --max-relative-l2 0.01 --require-argmax`):** **5 / 5 PASS** vs official HuggingFace `transformers==4.55.0` FP32 AND C++ `gemma3_native`:

| Prompt | Tokens | Generated prefix | vs `HF 4.55.0 FP32` cos / relL2 | vs `gemma3_native` cos / relL2 | Argmax |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `arithmetic` | 17 | `2 + 2 = 4\n<end_of_turn>` | `0.999999999999` / `2.5028e-06` | `0.999999999999` / `1.0655e-06` | `236778` (PASS) |
| `capital` | 16 | `The capital of France is Paris.\n<end_of_turn>` | `0.999999999975` / `8.0428e-06` | `0.99999999999` / `5.6330e-06` | `818` (PASS) |
| `hamlet` | 13 | `Shakespeare wrote Hamlet.\n<end_of_turn>` | `0.999999999993` / `4.9390e-06` | `0.999999999997` / `3.0986e-06` | `157092` (PASS) |
| `multistep` | 34 | `12 * 8 - 27 = 96 - 2` | `1` / `2.2029e-06` | `1` / `1.2033e-06` | `236770` (PASS) |
| `minimum` | 24 | `The smaller of 547 and 193 is 19` | `0.999999999998` / `2.0716e-06` | `0.999999999998` / `2.5935e-06` | `818` (PASS) |

- **OPT4 glyph-stream fidelity boundary (arithmetic prompt vs `HF 4.55.0 FP32`):** Strict gate **FAIL** (`exit 2`, `gate_pass=false`): `cosine=0.921860172696`, `relative_l2=0.551065110577`, `max_abs_error=41.3553767204`, `kl=16.1745965801`, `source_argmax=236778` vs `candidate_argmax=568` (`argmax_match=false`; generation `" ( up]   and still..."`). Zero-training 2-stage 8D spherical RVQ at `~0.9718` cosine / `~12.7 dB` SNR (~23% per-matrix relative L2) remains too lossy for inference at 270M without higher precision or QAT; see `experiments/gemma270_luj` (`Q14-norms`) for strict-parity glyph storage.

Logs: `run270_exact_gen.log`, `run270_glyph_gen.log`, `ref_gemma3_native_arith.json`.

---

## Sub-Byte Frontier, `T16C` Calibration & `run31b` Pi 4 Receipts

Extended `run270` (`23`-config sub-byte sweep, 28Q/30Q suites, Windows AVX2 `T16C` `v6_e2e_s2`) and `run31b` Gemma 4 31B Raspberry Pi 4 receipts (4.51 GiB QAT `pin10` coherent generation, `packed92` coherent `2+2=4`, and `60/60` RAM-resident `5.71 GiB` `gemma4_31b_cal.t16c` with `VmSwap=0` and NEON FFI acceleration vs pure Rust) are recorded in [FRONTIER_RECEIPTS.md](FRONTIER_RECEIPTS.md) and backed by logs under `evidence/`. Portable calibration scripts live under `calib/`.

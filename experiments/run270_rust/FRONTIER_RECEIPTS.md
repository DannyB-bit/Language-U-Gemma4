# Packed Sub-Byte Frontier & Gemma 4 31B Pi 4 Execution Receipts

Branch: `packed-subbyte-frontier`
Hardware: Physical Raspberry Pi 4 ARM64 (`astronautshe@192.168.1.207`, `7.6 GiB` RAM, Cortex-A72 4 cores, `/mnt/SHE-VAULT` SSD).
Toolchain: `rustc` release profile (`RUSTFLAGS="-D warnings"`, zero external crate dependencies; PR #21 `ramfit` C tools built with `gcc -O3 -march=armv8-a+simd -mcpu=cortex-a72 -fopenmp -include stdint.h`).

## 1. New Capabilities in `experiments/run270_rust`

- `src/packed.rs`: Multi-threaded row-wise symmetric group-32 quantization (`b1` 2-means 1-bit + `q2`/`q3`/`q4`/`q5`/`q6`/`q8`/`q10`/`q12` bit-packed uniform group-32), optional orthonormal FWHT-8 (`+h8`) pre-rotation before quantization, and direct dequant-in-matvec (`PackedMatrix::matvec`, `decode_row`) without unpacking dense weight matrices.
- `src/safetensors.rs`: Bounded-RAM chunked reading (`read_raw_slice`, `read_tensor_bf16_u16`, `read_tensor_rows_f32`).
- `src/math.rs`: Raw Gemma 4 RMSNorm / QK-norm without `(1 + w)` offset (`rmsn_raw`, `qk_norm_raw`).
- `src/bin/run270.rs`: Added `--quantize`, `--h8`, `--embed-mode`, `--embed-h8`, `--bench-5`, and `--eval-suite` / `--suite-out` for direct 270M parity & reasoning Pareto sweeps.
- `src/bin/run31b.rs`: Bounded-RAM zero-Python native Gemma 4 31B QAT W4A16 inference runtime supporting batched prefill, proportional RoPE (`partial_rotary_factor=0.25`), `attention_k_eq_v`, `v_norm` (`qk_norm_raw(..., None)`), per-layer `layer_scalar`, `final_logit_softcapping=30.0`, streamed/BF16/`PackedMatrix` embedding/head modes (`--embed-mode stream|bf16|q4|q6|q8`), and partial layer pinning (`--pin-layers N`).

## 2. Gemma 3 270M Packed Sub-Byte Frontier (23 Configs, Physical Pi 4)

Reference calibration (`--exact --embed-mode f32`, `evidence/g270_calib/exact/bench5_summary.json`):
- **5-prompt strict HF `transformers==4.55.0` FP32 parity (`min_cos >= 0.999`, `max_relL2 <= 0.01`, `require_argmax`):** `5/5` PASS (`rel_l2` `2.07e-6..8.04e-6`, `166` forwards, `38.23s`, `vm_hwm_kb=1388968`).
- **Frozen reasoning (`frozen_reasoning_set.json`):** `10/28` (`35.71%`), `total_forwards=865`, `total_seconds=160.29`, `s/fwd=0.1853`.
- **Fresh blind reasoning (`fresh_blind_reasoning_set.json`):** `6/30` (`20.00%`), `total_forwards=890`, `total_seconds=156.16`, `s/fwd=0.1755`.

### 5-Prompt Parity & Compression Pareto (`evidence/g270_sweep/sweep_bench5_all23.json`)

| Config | Proj BPW | Total Weights (MB) | Peak `VmHWM` (KB) | `s/fwd` | Min Cosine | Max Rel-L2 | Argmax | Strict Gate | `hamlet_play` Completion Prefix |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|
| `exact_bf16` | `32.0000` | `736.85` | `1,033,244` | `0.1996` | `1.000000` | `0.000008` | `5/5` | `5/5` | `The play Hamlet was written by William Shakespeare.` |
| `b1` | `1.2706` | `351.69` | `656,816` | `0.1641` | `-0.500715` | `1.484452` | `0/5` | `0/5` | Non-coherent multilingual tokens |
| `b1+h8` | `1.2706` | `351.69` | `656,860` | `0.1633` | `-0.015118` | `1.033796` | `0/5` | `0/5` | Non-coherent multilingual tokens |
| `q2` | `2.5000` | `367.10` | `671,796` | `0.2010` | `0.577558` | `1.060201` | `0/5` | `0/5` | Word salad / hallucinated |
| `q2+h8` | `2.5000` | `367.10` | `671,884` | `0.2013` | `0.306943` | `1.308254` | `0/5` | `0/5` | Fragmented symbol loops |
| `q3` | `3.5000` | `379.64` | `683,996` | `0.2622` | `0.207504` | `0.986480` | `0/5` | `0/5` | Fluent grammatical drift |
| `q3+h8` | `3.5000` | `379.64` | `684,224` | `0.2630` | `0.649394` | `1.350004` | `0/5` | `0/5` | Bracketed CJK template drift |
| `q4` | `4.5000` | `392.17` | `696,284` | `0.1598` | `0.874029` | `1.034042` | `4/5` | `0/5` | `The play "Hamlet" by William Shakespeare is written by **Hamlet Polonio**.` |
| `q4+h8` | `4.5000` | `392.17` | `696,296` | `0.1518` | `0.890427` | `0.465145` | `3/5` | `0/5` | `The play Hamlet is attributed to **William Shakespeare**.` |
| `q4+h8_eq4` | `4.5000` | `151.00` | `460,844` | `0.1565` | `0.890325` | `0.536594` | `2/5` | `0/5` | `The play Hamlet is attributed to **William Shakespeare**.` (unstopped) |
| `q4+h8_eq6` | `4.5000` | `192.94` | `501,696` | `0.2489` | `0.890845` | `0.473828` | `3/5` | `0/5` | `The play Hamlet is attributed to **William Shakespeare**.` |
| `q5` | `5.5000` | `404.70` | `708,624` | `0.2955` | `0.874941` | `0.821380` | `4/5` | `0/5` | `The play Hamlet was written by William Shakespeare.` |
| `q5+h8` | `5.5000` | `404.70` | `708,496` | `0.2882` | `0.951566` | `0.330993` | `4/5` | `0/5` | `The play Hamlet was written by William Shakespeare.` |
| `q6` | `6.5000` | `417.24` | `720,820` | `0.3311` | `0.972592` | `0.247567` | `5/5` | `0/5` | `The play Hamlet was written by William Shakespeare.` |
| `q6+h8` | `6.5000` | `417.24` | `720,940` | `0.3180` | `0.991824` | `0.165464` | `4/5` | `0/5` | `The play Hamlet was written by William Shakespeare.` |
| `q8` | `8.5000` | `442.31` | `745,216` | `0.1508` | `0.998939` | `0.071482` | `5/5` | `0/5` | `The play Hamlet was written by William Shakespeare.` |
| `q8+h8` | `8.5000` | `442.31` | `745,356` | `0.1558` | `0.999052` | `0.047611` | `5/5` | `0/5` | `The play Hamlet was written by William Shakespeare.` |
| `q8_eq8` | `8.5000` | `285.02` | `591,828` | `0.1607` | `0.999191` | `0.061950` | `5/5` | `0/5` | `The play Hamlet was written by William Shakespeare.` |
| `q8+h8_eq8` | `8.5000` | `285.02` | `591,664` | `0.1570` | `0.998315` | `0.063065` | `5/5` | `1/5` | `The play Hamlet was written by William Shakespeare.` |
| `q10` | `11.0000` | `473.64` | `776,028` | `0.3774` | `0.999952` | `0.010063` | `5/5` | `3/5` | `The play Hamlet was written by William Shakespeare.` |
| `q10+h8` | `11.0000` | `473.64` | `775,808` | `0.3782` | `0.999973` | `0.007393` | `5/5` | `5/5` | `The play Hamlet was written by William Shakespeare.` |
| `q12` | `13.0000` | `498.71` | `800,444` | `0.3924` | `0.999998` | `0.002328` | `5/5` | `5/5` | `The play Hamlet was written by William Shakespeare.` |
| `q12_eq8` | `13.0000` | `341.42` | `646,752` | `0.3983` | `0.999882` | `0.033842` | `5/5` | `2/5` | `The play Hamlet was written by William Shakespeare.` |

Key sub-byte findings:
- **Coherence onset (`Q4+H8`, `4.50 bpw`):** H8 reduces worst `rel_l2` by 55% vs `Q4` (`0.4651` vs `1.0340`) and eliminates factual hallucination on Hamlet (`William Shakespeare` vs `Hamlet Polonio`).
- **Full reasoning recovery onset (`Q5+H8`, `5.50 bpw`):** Jumps from `1/28` (`Q4+H8`) to `8/28` frozen / `10/30` fresh.
- **Efficiency-accuracy sweet spot (`Q8_eq8`, `8.50 bpw` projections + `Q8` embedding):** `285.02 MB` total weights (`-61.3%` vs BF16, `-73.4%` vs F32/BF16 exact calibration), `591,660 KB` peak `VmHWM`, `0.1233-0.1304 s/fwd` (~30% faster than `exact`), exact `5/5` argmax match, and `10/28` frozen / `6/30` fresh parity with unquantized BF16.
- **Lowest strict HF FP32 logit-gate pass (`Q10+H8`, `11.00 bpw`):** `H8` drops strict parity (`rel_l2 <= 0.01`, `min_cos >= 0.999`, `5/5` argmax) from `Q12` (`13.00 bpw`) down to `Q10+H8` (`11.00 bpw`, `max_rel_l2 = 0.007393`, `min_cos = 0.999973`, `5/5` pass).

### 28Q Frozen & 30Q Fresh Reasoning Suites (`evidence/g270_suites/`)

| Config | Total Weights (B) | Frozen `VmHWM` (KB) | Frozen (28Q) | Fresh (30Q) | Frozen `s/fwd` | Fresh `s/fwd` |
|---|---:|---:|---:|---:|---:|---:|
| `exact` (`emb f32`) | `1,072,392,704` | `1,388,968` | `10/28` (`35.71%`) | `6/30` (`20.00%`) | `0.1853` | `0.1755` |
| `b1+h8` (`emb bf16`) | `351,693,312` | `656,720` | `0/28` (`0.00%`) | `0/30` (`0.00%`) | `0.1575` | `0.1656` |
| `q3` (`emb bf16`) | `379,636,224` | `684,080` | `0/28` (`0.00%`) | `0/30` (`0.00%`) | `0.2552` | `0.2550` |
| `q4` (`emb bf16`) | `392,169,984` | `696,176` | `0/28` (`0.00%`) | `0/30` (`0.00%`) | `0.1547` | `0.1487` |
| `q4+h8` (`emb bf16`) | `392,169,984` | `696,320` | `1/28` (`3.57%`) | `2/30` (`6.67%`) | `0.1324` | `0.1329` |
| `q4+h8_eq4` (`emb q4+h8`) | `150,997,504` | `460,860` | `0/28` (`0.00%`) | `1/30` (`3.33%`) | `0.1370` | `0.1427` |
| `q5+h8` (`emb bf16`) | `404,703,744` | `708,552` | `8/28` (`28.57%`) | `10/30` (`33.33%`) | `0.2663` | `0.2629` |
| `q6` (`emb bf16`) | `417,237,504` | `720,828` | `10/28` (`35.71%`) | `7/30` (`23.33%`) | `0.2866` | `0.2834` |
| `q6+h8` (`emb bf16`) | `417,237,504` | `720,832` | `9/28` (`32.14%`) | `7/30` (`23.33%`) | `0.2906` | `0.2846` |
| `q8_eq8` (`emb q8`) | `285,018,624` | `591,660` | `10/28` (`35.71%`) | `6/30` (`20.00%`) | `0.1304` | `0.1233` |
| `q10+h8` (`emb bf16`) | `473,639,424` | `775,784` | `10/28` (`35.71%`) | `6/30` (`20.00%`) | `0.3529` | `0.3474` |

## 3. Full 60-Layer Gemma 4 31B Native Rust Execution on Pi 4 (`run31b`)

Checkpoint: `/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct` (`model.safetensors`, `config.json`, `tokenizer.json`; `d=5376, layers=60, ff=21504, heads=32/16, global_kv=4, hd=256/512, window=1024, softcap=30.0`).
`/mnt/SHE-VAULT/gemma-4-31B-it-assistant/model.safetensors` was separately verified to be a 4-layer `Gemma4AssistantForCausalLM` draft module without `k_proj`/`v_proj`/`k_norm`, unsuitable for standalone LM inference.

1. **Streamed 60-layer prefill (`evidence/run31b/test_2plus2_stream1.txt`):**
   - Command flags: `--threads 4 --embed-mode stream --pin-layers 0 --prompt-text "What is 2 + 2? Answer with only the number." --max-tokens 1` (`27` prompt tokens matching HF `AutoTokenizer` bit-exact).
   - **Peak RAM:** `vm_hwm_kb = 344,600` (`336.5 MiB`), `vm_rss_kb = 258,784` (`252.7 MiB`).
   - **I/O & timing:** `ssd_read_bytes = 19,292,646,912`, `prefill_s = 511.13`.
   - **`PREFILL_TOP5`:** `[236812:"4"(26.483), 236819:"9"(14.989), 236832:"7"(14.259), 236825:"6"(14.164), 236810:"5"(13.849)]`.
   - **Generated output:** `ids=[236812]`, `text="4"` (`+11.49` logit margin over 2nd place).

2. **Hybrid RAM-pinned + packed-embedding 60-layer multi-token decode (`evidence/run31b/hamlet_q8emb_pin10.txt`):**
   - Command flags: `--threads 4 --embed-mode q8 --pin-layers 10 --prompt-text "Who wrote the play Hamlet?" --max-tokens 12` (`19` prompt tokens matching HF `AutoTokenizer` bit-exact).
   - **RAM budget (`< 5 GB` envelope):** `pinned_bytes = 2,730,312,704` (`10/60` layers), `embed_bytes = 1,497,366,528` (`q8`), `vm_hwm_kb = 4,436,992` (`4.23 GiB`), `vm_rss_kb = 4,344,084` (`4.14 GiB`).
   - **I/O & timing:** `ssd_read_bytes = 123,739,176,960`, `prefill_s = 431.36`, `decode_s = 614.32` (`76.2-78.9 s/tok` across 8 decode steps).
   - **`PREFILL_TOP5`:** `[34848:"William"(28.013), 818:"The"(21.836), 1018:"**"(21.649), 8966:" William"(19.445), 246061:"윌"(17.434)]`.
   - **Generated output:** `ids=[34848, 36951, 7971, 506, 1441, 808, 201313, 22429, 106]`, `text="William Shakespeare wrote the play *Hamlet*.<turn|>"` (clean EOS `<turn|>` termination).

## 4. PR #21 (`amara/ramfit-gate1`) Physical Pi 4 Findings (`evidence/ramfit_logs/`)

- **Build quirk:** `neon_runner/ramfit/reencode31b.c` omits `#include <stdint.h>`, failing gcc compilation on ARM64 Linux (`unknown type name 'uint16_t'`) unless compiled with `-include stdint.h`.
- **`selftest` (`selftest.txt`):** `132c-vs-92 decode parity on same weights: max|d| = 0.000e+00`, while round-tripping `/mnt/SHE-VAULT/gemma-31b-language-u/gemma4_31b_language_u_sandwich_stream.bin` gave `92B round-trip vs 132B decode: max RMS diff 9.339e-03`.
- **On-disk Pi sandwich stream state (`cosprobe.txt`):** `/mnt/SHE-VAULT/gemma-31b-language-u/gemma4_31b_language_u_sandwich_stream.bin` is only `33 MB` (stopped at projection `[1/410]` per `reencode.log`), and `/home/astronautshe/gemma-31b-language-u/gemma4_31b_tensor_layout.json` lists all `410` projections as `OPT4` (`bytes_per_glyph: 132`), so `cosprobe` reported `MEAN over 0 projections: median-law nan | 2-means nan`.
- **`embed_encode` (`embed_encode.txt`, `embed_verify.txt`):** Encoded `/home/astronautshe/gemma-31b-language-u/gemma4_31b_embed_tokens.bin` (`2,818,572,288` B) -> `198,180,864` B (`198.2 MB`), measuring `verify 512 rows: SNR = 3.49 dB (36B binary vs fp16 embed)`. Combined with the `270M` `b1+h8` collapse (`1.27 bpw`, `0/28` / `0/30`), ~`1.125-1.27 bpw` 1-bit quantization cannot sustain coherent multi-layer inference without higher bit-width or training.

## 5. Pi 4 `T16C v1` 60/60 RAM-Resident 31B (`gemma4_31b_cal.t16c`) (`evidence/run31b/`, `calib/31b/`)

- **Artifact & Assembly (`cal31b_v2.log`):** Assembled via `calib/31b/cal31b_streaming.py` -> `/mnt/SHE-VAULT/gemma4_31b_cal.t16c` (`6,134,689,856` B / `5.7134 GiB`, SHA-256 `e63a583bc6f8ed28d2429673ddcc6ce3f8e2d54cd7802100d768f16633d7b345`; `421` norms `5,326,064` B + `411` mats [`410` projections `5,413,632,000` B + `4b` tied embed `715,653,120` B], `dt=67.6s`). Loads into `pinned_layers=60/60` (`pinned_bytes=5,418,958,064` B) inside 8 GB Raspberry Pi 4 RAM.
- **Run 1 — `2 + 2` 27-token chat, NEON FFI (`run31b_cal_v2_2plus2.log`, `run31b_cal_v2_2plus2_proc.log`):** `loaded in 36.03s`, prefill `33.78s` (`ssd_read_mb=0.0`), 8 greedy steps `decode_s=14.65s` (`2.08-2.13s/step`), `BATCH_K k=4 body_s=4.810 head_s=0.355 total_s=5.165 s_per_tok=1.291 head_ms_per_tok=88.8`. Process `/proc` telemetry proves **`vm_hwm_kb=6238800`** (`5.95 GiB`), **`VmSwap: 0 kB`**, **`ssd_read_bytes=0`**, and kernel `read_bytes` frozen at `7925960704` from prefill exit across all decode steps `0..7`.
- **Run 2 — `Hamlet` prompt, NEON FFI (`run31b_cal_v2_hamlet.log`, `run31b_cal_v2_hamlet_proc.log`):** `loaded in 26.42s`, prefill `22.69s` (`ssd_read_mb=0.0`), 16 greedy steps `decode_s=31.27s` (`2.06-2.17s/step`), `PREFILL_TOP5 [715:"//"(28.548), 100:"<|channel>"(27.403), 1018:"**"(27.028), 818:"The"(26.473), 1638:" same"(26.393)]`, **`vm_hwm_kb=6238804`**, **`VmSwap: 0 kB`**, kernel `read_bytes` frozen at `6633926656` across steps `0..15`.
- **Run 3 — `--no-native-ffi` Pure Rust vs NEON FFI Parity (`run31b_cal_v2_2plus2_noffi.log`, `run31b_cal_v2_ffi_parity.txt`):** `loaded in 25.95s`, prefill **`889.23s`** (`ssd_read_mb=0.0`, `vm_hwm_kb=6219076`, layers `00/05/14/29/44/59` in `15.38s/16.58s/14.06s/16.66s/14.42s/17.15s` vs `~0.52-0.68s/layer` with NEON FFI = **`26.32x` prefill speedup** on Cortex-A72). Logit check against Run 1: `FFI_VS_NOFFI: cos=0.99929892 rel_l2=3.75e-02 max_abs=2.35e+00 top1=(715, 715) match=True`.
- **Honest fidelity boundary:** While `T16C` breaks the 8 GB Pi 4 RAM-residency (`60/60`, `VmSwap=0`, `ssd_read_bytes=0`) and throughput (`1.291 s/tok k=4`, `26.32x` NEON FFI prefill acceleration) barriers, greedy decoding on `gemma4_31b_cal.t16c` collapses to `GEN31B text="////////////////"` on both prompts (`cal31b_v2.log` records mid-depth gain/RMS shifts `L28 g:0.9540`, `L29 g:0.9508`, `L32 g:0.9254`, `x_rms 4.05 -> 0.14`), showing sub-2-bit PTQ still requires QAT or selective higher-bit allocation on sensitive layers (in contrast to coherent `run31b` QAT `pin10` `cos=0.999995` and `packed92` `"2 + 2 = 4<turn|>"`).

## 6. Windows `T16C` `270M v6_e2e_s2` Calibration & AVX2 Parity (`evidence/g270_calib/`, `calib/`)

- **Container (`gemma3_270m_v6_e2e_s2.t16c` via `solve_270m_v6_e2e.py` + `solve_270m_v6_e2e_stage2.py`):** `proj_bytes=19,349,040` (`1.543754 bpw` projections), `embed_bytes=335,544,320` (`BF16`), `norm_bytes=223,744`, `total_weight_bytes=355,117,104` (`10.598840 eff_bpw`).
- **Windows MSVC AVX2 DLL vs Pure-Rust Speed (`v6_e2e_s2_avx2_bench5.json` vs `v6_e2e_s2_rust_bench5.json`):** `neon_stream_accelerator.dll` AVX2 FFI measured `0.090499 s/fwd` (`15.4753s` / `171` fwds) vs pure Rust `0.505491 s/fwd` (`85.4280s` / `169` fwds) = **`5.59x` speedup** (and `2.54x` faster than exact `0.230274 s/fwd` in `exact/bench5_summary.json`).
- **Honest suite boundary:** `v6_e2e_s2_avx2_frozen.json` **`0/28`** (`110.3619s`) and `v6_e2e_s2_avx2_fresh.json` **`0/30`** (`128.9111s`) establish the `1.54 bpw` PTQ accuracy bound against `q8_eq8` (`285.02 MB`, `10/28` frozen & `6/30` fresh) and `exact` (`10/28` frozen & `6/30` fresh).

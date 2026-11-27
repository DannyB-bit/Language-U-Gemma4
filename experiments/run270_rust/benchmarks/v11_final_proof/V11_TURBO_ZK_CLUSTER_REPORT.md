# Language-U V11 Turbo + ZK-Cortical-Cluster + Cognitive-Core VA Mirror Report (Raspberry Pi 4 Physical Silicon)

> **Scope & Truthfulness Note:** All metrics below were measured physically on a single **Raspberry Pi 4 Model B (4x Cortex-A72 ARM64, 7.6 GiB RAM)** running `experiments/run270_rust/target/release/run31b` compiled with `RUSTFLAGS=-Dwarnings` (`93/93` Rust unit tests + `10/10` `Cognitive-Core-Atlas` unit tests passing, `0` warnings). Unverified Tier-1 draft probes and repeat-session VA-mirror throughput vs. cold unseen accuracy are reported separately below.

---

## 1. Executive Summary vs. V10 Baseline

| Subsystem / Verification Gate | V10 Baseline (`v10_pi4_world_proof.json`) | V11 Measured on Pi 4 Silicon | Status & Source |
| :--- | :---: | :---: | :--- |
| **Strict Text Memory Gate (`VmHWM <= 6.50 GiB`, `ssd=0`)** | `6,822,916 kB` (**FAIL**, `+7.0 MiB` over) | **`6,519,172 – 6,811,868 kB`** (`<= 6,815,744 kB`), `ssd=0` | **PASS** (pre-router Vision eviction) |
| **Intra-Chip VA Parking (`--cca-va-plan`, `[32, 46]` parked)** | N/A | **`VmHWM = 6,596,072 kB` (`6.290 GiB`)**, `0.40 tok/s`, fluent code | **PASS** (`task-9256`, `179.2 MB` VA-parked) |
| **32 GiB `MAP_NORESERVE` VA Mirror (`L=8`, repeat session)** | N/A | **`0.957 tok/s`** (**`2.03x`** vs `0.471` 60L dense), **`16/16` exact match**, **`3,512 kB`** faulted RAM | **PASS** (`v11_va_mirror_exp.json`) |
| **Standard Single-Device Coherent Cascade (`v11_parity_k2` / `v11_sparse_mod_k3`)** | `0.39 – 0.43 tok/s` | `0.39 – 0.40 tok/s` (`VmHWM = 6.46 GiB` PASS) | **PASS** (memory fixed; ~same speed) |
| **Aggressive Unverified Cascade (`v11_fused_cascade_turbo`)** | N/A | `0.54 – 0.70 tok/s` (`tier1_fast=33`), **degraded output** | **COHERENCE BOUNDARY** (too sparse at `mlp=0.143`) |
| **`255 B` `ZK-LoRaWAN` Single-Chirp E2E (`ZLU1` + ALU)** | N/A | `255 B` in -> `510 B` out, **`alu='13'`**, `XOR-FEC` + ZK seal verified | **PASS** (`v11_pi4_turbo_zk_proof.json`) |
| **`ZK-Cortical-Cluster` Shard Worker (`--layers 30..60`)** | N/A (`ggml-rpc` raw pointers) | **`3.136 GiB` RSS (`3,288,204 kB`)**, `LUZKCL11` mutual SHA-256 attest OK | **PASS** (fits `4 GB` RAK Miner v2 / phone) |
| **`Cognitive-Core-Atlas` Bounded `.t16c` Census + 31B Probe** | N/A (`bfloat16` needs ~62 GB) | `1,190` tensors / `31.27B` params in **`283.16 ms`**; live 31B depth probe | **PASS** (`Cognitive-Core-Atlas` `d2a21c5`) |

---

## 2. Dual-End Input-to-Output VA Mirror Experiment (`v11_va_mirror_exp.json`)

- **Arena:** `32.00 GiB` reserved via `mmap(MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE)` (`hardware_mmap=true`, `hardware_tbi=true` using ARM64 Top-Byte-Ignore `[63:56]`).
- **Physical DRAM Footprint:** `0 kB` at reservation; only **`878` 4 KB pages (`3,512 KiB`)** faulted after `407` mirror writes across 4 depths (`L = 0, 8, 20, 32`).
- **Input Conversion:** Incoming sequences are converted before Layer `0` into causal multi-scale mirrored vectors + `FWHT-64` orthogonal 64-bit mirror codes + Cuneiform-6D TBI tags (`src/va_mirror.rs`).

### Hit Rate & Exact 60-Layer Token Agreement by Depth & Hamming Radius

| Evaluation Split | Depth | `max_h=0` Hits / Agree | `max_h=4` Hits / Agree | `max_h=8` Hits / Agree | `max_h=12` Hits / Agree |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `strict_unseen` (`n=24`, incl. chat header) | `0` (Input Mirror) | 45.8% / 90.9% | 54.2% / 84.6% | 54.2% / 84.6% | 58.3% / 85.7% |
| `strict_unseen` (`n=24`) | `8` (13% depth) | 16.7% / 100.0% | **58.3% / 85.7%** | 70.8% / 76.5% | 91.7% / 68.2% |
| `strict_unseen` (`n=24`) | `20` (33% depth) | 16.7% / 75.0% | 58.3% / 64.3% | 91.7% / 50.0% | 95.8% / 47.8% |
| `strict_unseen` (`n=24`) | `32` (53% depth) | 12.5% / 100.0% | 37.5% / 55.6% | 91.7% / 31.8% | 100.0% / 33.3% |
| `shifted_domain` (`n=36`, `2a+b=13` vs `11`) | `0` (Input Mirror) | 91.7% / 93.9% | 94.4% / 94.1% | 97.2% / 94.3% | 97.2% / 94.3% |
| `shifted_domain` (`n=36`) | `8` (13% depth) | 91.7% / 100.0% | **97.2% / 100.0%** | 97.2% / 100.0% | 97.2% / 100.0% |
| `shifted_domain` (`n=36`) | `20` / `32` | 77.8% / 100% | 97.2–100% / 100% | 100% / 100% | 100% / 100% |

### Live Autoregressive Decode (`16` tokens, repeat warm session query, exact 60L KV flush on miss)

- **Dense 60L Baseline:** `0.471 tok/s` (`33.95 s`), output `"To solve the linear system, you can use a step-by-step approach"`
- **VA Mirror `L=0` (Input Mirror):** `0.597 tok/s` (**`1.27x`**), `5/15` hits, `5/16` exact token matches (diverged after shared prefix due to L0 lexical collisions)
- **VA Mirror `L=8` (13% depth + batched 8..60 KV sync):** **`0.957 tok/s` (`2.03x`)**, `12/15` hits, **`16/16` (100% exact token match)**, `VmHWM = 6,519,172 kB`, `ssd_read_bytes = 0`

---

## 3. Coherent vs. Aggressive Cascade Telemetry (`v11_pi4_turbo_zk_proof.json`)

- **Coherent Cascade (`v11_parity_k2` / `v11_sparse_mod_k3`):** `0.39 – 0.40 tok/s`, fluent Python/system answers, `4/4` on spot battery with ALU (`2/4` pure neural), `VmHWM = 6,726,408 – 6,775,416 kB` (**100% memory gate pass**).
- **Coherent Q4 + CCA Plan (`cca_coherent_verified_q4_runs`):** `0.29 – 0.33 tok/s`, `VmHWM = 6,810,956 – 6,811,868 kB` (`<= 6,815,744 kB`).
- **Over-Pruned Cascade Boundary (`v11_fused_cascade_turbo`):** `0.54 – 0.70 tok/s` (`mlp_top_p = 0.143`, `head_top_p = 0.25`, `mod_thresh = 0.50`, `tier1_tau = 2.20`), but `accept_rate = 0.0` and garbled multilingual/repeated `"that"` output — confirming `mlp_top_p = 0.143` is below the coherent Pareto knee on 31B without task-specific retraining.
- **Disambiguated Tier-1 4-Layer Draft Ceiling Probe:** `21.17 tok/s` (`47.2 ms/tok`), unverified 4-layer draft ceiling only (`VmHWM = 6,710,708 kB`).

---

## 4. `ZK-LoRaWAN` (`255 B`), `ZK-Cortical-Cluster`, & `Cognitive-Core-Atlas`

- **`ZK-LoRaWAN` Single-Chirp Ingress -> 31B -> Egress:** Packed `"If 5x + 3 = 38, what is the value of 2x - 1?"` into a single `255 B` `ZLU1` chirp (`seq=77`, `coords_6d=[-89, 6, 72, 51, 11, 19]`, `nullifier=861dec00...`), executed live Gemma 4 31B + General ALU (`alu='13'`), and emitted `2` `ZLU1` egress chirps (`510 B`) verified with `20 B` `XOR-FEC` and SHA-256 commitment seals.
- **`ZK-Cortical-Cluster` Worker (`--layers 30..60`):** `LUZKCL11` mutual SHA-256 attestation passed (`attestation_sha256_verified = true`), **`worker_vm_rss_kb = 3,288,204` (`3.136 GiB`)**, confirming `<= 4 GB` compatibility for paired RAK Hotspot Miner v2 or spare Android phone deployments (`src/zk_cluster.rs`).
- **`Cognitive-Core-Atlas` Bridge:** Paired `v11-zk-cortical-cluster` branch in `DannyB-bit/Cognitive-Core-Atlas` (`d2a21c5`) plans `2x 4 GB` shards (`0..35` @ `3.762 GiB` + `35..60` @ `3.043 GiB`) bound by `cca_manifest_sha256` (`c442701c...`) and drives `run31b --cca-plan` / `--cca-va-plan` natively (`src/cca_bridge.rs`).

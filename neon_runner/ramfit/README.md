# RAM-Fit Mixed-Container Encoder (Gate 1 for the Pi 4 31B barrier)

Pure-C tools that turn the published **7.08 GB Gemma-4-31B Language-U sandwich
stream** into a **RAM-resident mixed 92 B / 36 B stream** that fits the Pi 4's
5 GB envelope — and prove what each container costs.

No Python. No new dependencies. The 132-byte doctrine container is read, never written.

## Tools

| Tool | What it does |
|---|---|
| `reencode31b.c` | `encode` — re-quantize the published stream into per-projection 92 B (lossless packing of the 132 B law) or 36 B (median-law binary) under a byte budget; `verify` — per-projection cosine of the new stream against the original; `selftest` — prove the 92 B container is bit-exact against the 132 B law |
| `shuffle_stream.c` | null control: permute glyph order inside each projection (same bytes, same codec, same offsets) to destroy the learned row→weight mapping while leaving everything else intact |
| `embed_encode.c` | encode the 2.82 GB fp16 token-embedding table into 36 B glyphs (198 MB) so the embedding fits the same ledger |

## Container laws (implemented exactly as the published runtime)

- **132 B (Opt4, boundary, read-only):** `[s1 f16][s2 f16][32×u16 idx1][32×u16 idx2]`,
  per octonion `w = fwht8(s1·cb[idx1] + s2·cb[idx2])`
- **48 B (Opt5, middle):** `[scale f16][bias f16][32 B sign mask][4×u8 outlier idx][4×f16 outlier residual]`,
  `w[i] = bias + (bit ? +scale : -scale) + Σ res_k·[i == idx_k]`
- **92 B (new high tier):** the 132 B law with its 64 indices packed to 11 bits
  (`4 + 64×11/8 = 92 B`). **Lossless vs 132 B** — the selftest prints
  `132c-vs-92 decode parity on same weights: max|d| = 0.000e+00`.
- **36 B (new low tier):** `[scale f16][bias f16][32 B sign mask]`,
  `w[i] = bias + (bit ? +scale : -scale)`. Two variants, same 36 bytes:
  - *median law* — bias = median, scale = max|centered|;
  - **2-means law (default)** — bias/scale encode the two k-means centroids
    (`c0 = bias − scale/2`, `c1 = bias + scale/2`), bits assigned to the nearer
    centroid. Measured on the real 48 B middle lane (5 projections, 32 rows each):

    | law | mean cosine vs the published decode |
    |---|---|
    | median law | **0.660** |
    | **2-means law** | **0.893** |

    Same byte cost, +0.23 cosine — this is the single biggest Gate 1 win measured so far.

Sign bits are **MSB-first within each byte** in every container, matching the
existing Opt5 ABI.

## Budget arithmetic (integer, verified)

`N = 114,401,280` glyphs. All-36 B baseline = `16 + N×36 = 4,118,446,116` B.
Upgrading a glyph to 92 B costs 56 B more, so a 4.30 GB budget gives
`p = (4.30e9 − 4.1184e9) / (N×56) = 2.83 %` high-tier glyphs.

Full 500-turn resident ledger (Pi 4, 8 GB):

| Item | Bytes |
|---|---|
| mixed stream (4.298 GB measured, p = 2.83 % @ 92 B) | 4,298,000,000 |
| token embedding as 36 B glyphs (5,505,024 glyphs) | 198,180,864 |
| KV cache, 500 turns, int8 (840 KB/token fp16 → 420 KB int8) | 215,000,000 |
| norms + codebook | 3,055,618 |
| runtime reserve | 200,000,000 |
| **total** | **≈ 4.91 GB** |

With an fp16 KV cache the same ledger is ≈ 5.13 GB — **over budget**. Either
int8 KV or a smaller stream budget is required for 500 turns.

## Run on the Pi 4 (the body that holds the 7 GB stream)

```bash
# build (no Python, no repo deps)
gcc -O3 -march=armv8-a+simd -mcpu=cortex-a72 -fopenmp -o reencode31b reencode31b.c -lm
gcc -O3 -march=armv8-a+simd -mcpu=cortex-a72 -o embed_encode embed_encode.c -lm
gcc -O3 -march=armv8-a+simd -mcpu=cortex-a72 -o shuffle_stream shuffle_stream.c -lm

D=/home/astronautshe/gemma-31b-language-u

# 0) prove the 92 B container is lossless before you trust it
./reencode31b selftest \
  $D/gemma4_31b_language_u_sandwich_stream.bin \
  $D/gemma4_31b_tensor_layout.json \
  $D/s7_codebook_2048x8.bin 16

# 1) encode the RAM-fit stream (checkpointed: pass 1 resumes if killed)
OMP_NUM_THREADS=4 ./reencode31b encode \
  $D/gemma4_31b_language_u_sandwich_stream.bin \
  $D/gemma4_31b_tensor_layout.json \
  $D/s7_codebook_2048x8.bin \
  $D/ramfit500 4300000000

# 2) embedding -> glyphs
./embed_encode encode $D/gemma4_31b_embed_tokens.bin 262144 5376 $D/embed_glyphs36.bin

# 2b) A/B the two 36B laws (median vs 2-means) on the real middle lane
./reencode31b cosprobe \
  $D/gemma4_31b_language_u_sandwich_stream.bin \
  $D/gemma4_31b_tensor_layout.json \
  $D/s7_codebook_2048x8.bin 5

# 3) fidelity: cosine of the new stream against the published one
./reencode31b verify \
  $D/gemma4_31b_language_u_sandwich_stream.bin \
  $D/gemma4_31b_tensor_layout.json \
  $D/ramfit500.lanu \
  $D/ramfit500.layout.json 20

# 4) null control
./shuffle_stream $D/ramfit500.lanu $D/ramfit500.layout.json $D/ramfit500-shuf.lanu 12345
```

## Evidence required to call Gate 1 passed

1. `selftest` prints `max|d| = 0` for the 92 B parity.
2. `verify` prints per-projection cosine for every projection (not a sample only).
3. **Multi-prompt logit parity**: exact Hamlet tokens plus ≥3 held-out prompts, raw
   per-step logits, new stream vs published stream.
4. **Real residency**: resident pages, RSS/PSS, process I/O and major faults around a
   complete warm decode. A 4.3 GB file alone does not prove RAM residency —
   readahead can still hit the disk.
5. The embedding lane must be included in the ledger (198 MB as glyphs, not 2.82 GB fp16).

## Known honest limits

- The 36 B lane is **lossy by construction** (1 bit/weight + median). It is not a
  lossless re-encoding of the published stream; `verify` quantifies the loss per projection.
- The published stream's own fidelity against the source checkpoint is already
  ≈ 0.964 output cosine (measured on layer-0 Q in the correction experiments); the
  36 B lane stacks on top of that. Judge the candidate against the published stream,
  and the published stream against the checkpoint.
- This tool does **not** produce generation quality, a Pi throughput number, or a
  reasoning claim. It produces the RAM-fit candidate and its fidelity receipts.

## Provenance

The container laws are taken from this repository's `compiler/glyphs.py` and
`langut/runtime/cortex.py`; the 132 B decode follows `fwht8` as in the doctrine
accelerator. The encoder here is original work written for this gate.

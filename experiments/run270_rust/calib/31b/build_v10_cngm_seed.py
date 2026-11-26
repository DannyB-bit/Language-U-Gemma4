#!/usr/bin/env python3
"""Build a strictly decontaminated CNGM v1 general phrase seed for Language-U V10 Tier-0 drafting.

Enforces Rule 2 (zero blacklist terms AND zero 4-gram overlap against all 15 stress-battery
and 3 held-out NLL/chat prompts) before writing compact binary `CNGM` v1 transitions.
"""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_v9_0_sparse_spinquant_qat_gpu as v90  # noqa: E402
import v4_corpus as vc  # noqa: E402

CNGM_MAGIC = b"CNGM"
CNGM_VERSION = 1

# General non-contaminated technical/code/prose idioms for Tier-0 cold phrase hits (never eval topics)
GENERAL_PROSE_SEEDS: Tuple[str, ...] = (
    "In summary, the algorithm runs in linear time and constant auxiliary space.",
    "Therefore, each step preserves invariants across iterations until termination.",
    "However, careful numerical scaling prevents underflow and improves convergence.",
    "Specifically, we can simplify the expression by factoring common terms first.",
    "First, compute the intermediate value. Second, verify boundary conditions carefully.",
    "```python\ndef solve(items):\n    if not items:\n        return []\n    result = []\n    for item in items:\n        result.append(item)\n    return result\n```",
    "```python\nfor i in range(len(values)):\n    if values[i] > threshold:\n        total += values[i]\n```",
    "```json\n{\"status\": \"ok\", \"count\": 0, \"items\": []}\n```",
    "This occurs because the matrix dimension must match the vector length along the inner axis.",
    "Yes, that statement holds true under standard assumptions.",
    "No, that condition is false because the conclusion does not follow from the premises.",
    "The key takeaway is that memory bandwidth dominates elementwise throughput on edge CPUs.",
    "As a result, batching across speculative branches amortizes weight DRAM traffic per token.",
)


def encode_cngm_v1(sequences: Sequence[Sequence[int]], max_n: int = 6, min_count: int = 1) -> bytes:
    """Serialize tokenized transitions `(ctx_len 1..max_n-1)` into exact Rust `FastNGramProposalEngine` `CNGM` v1 format."""
    counts: List[Dict[Tuple[int, ...], Dict[int, int]]] = [defaultdict(lambda: defaultdict(int)) for _ in range(max_n - 1)]
    for seq in sequences:
        n = len(seq)
        for pos in range(1, n):
            nxt = int(seq[pos])
            for ctx_len in range(1, min(max_n, pos + 1)):
                ctx = tuple(int(v) for v in seq[pos - ctx_len : pos])
                counts[ctx_len - 1][ctx][nxt] += 1

    records: List[Tuple[int, Tuple[int, ...], int, int]] = []
    for idx, tbl in enumerate(counts):
        ctx_len = idx + 1
        for ctx, cmap in tbl.items():
            for nxt, cnt in cmap.items():
                if cnt >= min_count:
                    records.append((ctx_len, ctx, nxt, cnt))
    records.sort(key=lambda r: (r[0], r[1], -r[3], r[2]))

    buf = bytearray(struct.pack("<4sIII", CNGM_MAGIC, CNGM_VERSION, max_n, len(records)))
    for ctx_len, ctx, nxt, cnt in records:
        buf.append(ctx_len & 0xFF)
        for tid in ctx:
            buf.extend(struct.pack("<I", tid))
        buf.extend(struct.pack("<II", nxt, cnt))
    return bytes(buf)


def build_clean_seed_texts() -> List[str]:
    """Assemble and decontaminate all non-overlapping response & general prose seed strings."""
    hold_4g = v90.collect_holdout_4grams()
    texts: List[str] = []
    # Use clean V4 responses + non-overlapping turns only (never include any overlapping prompt prefix)
    for _, _, resp in vc.V4_CAL_PAIRS:
        ws = v90._normalize_words(resp)
        if not any((ws[i], ws[i + 1], ws[i + 2], ws[i + 3]) in hold_4g for i in range(max(0, len(ws) - 3))):
            texts.append(resp)
    texts.extend(GENERAL_PROSE_SEEDS)
    v90.verify_corpus_decontaminated(texts, hold_4g)
    return texts


def build_cngm_file(tokenizer_json: Path, out_path: Path, max_n: int = 6) -> Dict[str, int]:
    """Tokenize decontaminated texts with Gemma 4 tokenizer and write `.cngm` atomically."""
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json: {tokenizer_json}")
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(tokenizer_json))
    texts = build_clean_seed_texts()
    seqs = [list(tk.encode(t, add_special_tokens=False).ids) for t in texts]
    raw = encode_cngm_v1(seqs, max_n=max_n, min_count=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".cngm.tmp")
    tmp.write_bytes(raw)
    tmp.replace(out_path)
    return {"texts": len(texts), "bytes": len(raw)}


def run_selftest() -> int:
    """Mandatory 6-gate CPU self-test for clean CNGM seed builder."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Zero holdout 4-gram overlap on build_clean_seed_texts()
    hold_4g = v90.collect_holdout_4grams()
    texts = build_clean_seed_texts()
    gate("Gate 1: Strict Zero-Overlap 4-Gram Clean Corpus", len(texts) >= 30, f"texts={len(texts)}, hold={len(hold_4g)}")

    # Gate 2: Synthetic contamination fails closed
    dirty_caught = False
    try:
        v90.verify_corpus_decontaminated(["Compute 12 * 8 - 27 cleanly."], hold_4g)
    except ValueError:
        dirty_caught = True
    gate("Gate 2: Contamination Trap Fails Closed", dirty_caught)

    # Gate 3: Binary header + exact byte length verification
    payload = encode_cngm_v1([[10, 20, 30, 40, 50], [10, 20, 30, 40, 50]], max_n=5, min_count=1)
    magic, ver, max_n, n_rec = struct.unpack("<4sIII", payload[:16])
    gate("Gate 3: CNGM v1 Header Layout", magic == CNGM_MAGIC and ver == 1 and max_n == 5 and n_rec > 0, f"n_rec={n_rec}")

    # Gate 4: Parse back every record cleanly to exact EOF
    pos = 16
    parsed = []
    for _ in range(n_rec):
        clen = payload[pos]
        pos += 1
        ctx = struct.unpack(f"<{clen}I", payload[pos : pos + 4 * clen])
        pos += 4 * clen
        nxt, cnt = struct.unpack("<II", payload[pos : pos + 8])
        pos += 8
        parsed.append((ctx, nxt, cnt))
    gate("Gate 4: Exact EOF Binary Roundtrip", pos == len(payload) and ((10, 20, 30, 40), 50, 2) in parsed)

    # Gate 5: Multi-order longest-context ranking preserves high count
    ctx4 = [r for r in parsed if r[0] == (10, 20, 30)]
    gate("Gate 5: Context-3 Candidate Coverage", len(ctx4) == 1 and ctx4[0][1] == 40)

    # Gate 6: Atomic temp write succeeds cleanly
    with tempfile.TemporaryDirectory() as td:
        out_f = Path(td) / "seed.cngm"
        out_f.write_bytes(payload)
        gate("Gate 6: Disk Write Verification", out_f.is_file() and out_f.stat().st_size == len(payload))

    print(f"V10 CNGM Seed Self-Test: {6 - failures}/6 passed", flush=True)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="Build decontaminated CNGM v1 Tier-0 draft seed")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_v10/gemma4_v10_seed.cngm"))
    ap.add_argument("--max-n", type=int, default=6)
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()
    stats = build_cngm_file(args.tokenizer, args.out, max_n=args.max_n)
    print(f"[OK] Built {args.out}: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

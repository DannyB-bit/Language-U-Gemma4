#!/usr/bin/env python3
"""
Build a static `CNGM` v1 n-gram seed table from math/code/prose corpora for cold-start speculation.

Mines text files (e.g., `/mnt/SHE-VAULT/openai-math-ingest` LaTeX/Lean/Markdown traces and/or
repository source files), tokenizes them cleanly using HuggingFace `tokenizers` / `transformers`,
collects `1..max_n-1`-token context transitions, keeps the highest-frequency candidates meeting
`--min-count`, and writes compact binary `CNGM` v1 records matching `run270_rust::speculative`.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

CNGM_MAGIC = b"CNGM"
CNGM_VERSION = 1
DEFAULT_EXTENSIONS = (".tex", ".md", ".lean", ".py", ".rs", ".c", ".h", ".txt")


def load_text_encoder(tokenizer_path: Path) -> Callable[[str], List[int]]:
    """Load fast tokenizer encoder from `tokenizer.json` or HF model directory."""
    tok_p = Path(tokenizer_path)
    if tok_p.is_dir():
        cand = tok_p / "tokenizer.json"
        if cand.is_file():
            tok_p = cand
    if not tok_p.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    if tok_p.is_file() and tok_p.name.endswith(".json"):
        try:
            from tokenizers import Tokenizer  # type: ignore

            tok = Tokenizer.from_file(str(tok_p))
            return lambda text: [int(t) for t in tok.encode(text, add_special_tokens=False).ids]
        except ImportError:
            pass

    from transformers import AutoTokenizer  # type: ignore

    tok_dir = tok_p.parent if tok_p.is_file() else tok_p
    hf_tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    return lambda text: [int(t) for t in hf_tok.encode(text, add_special_tokens=False)]


def iter_corpus_files(
    roots: Sequence[Path],
    extensions: Tuple[str, ...] = DEFAULT_EXTENSIONS,
    max_file_bytes: int = 2 * 1024 * 1024,
) -> List[Path]:
    """Deterministically enumerate readable text corpus files under `roots`."""
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    found: List[Path] = []
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            raise FileNotFoundError(f"Corpus path not found: {rp}")
        if rp.is_file():
            if rp.stat().st_size <= max_file_bytes:
                found.append(rp.resolve())
            continue
        for dirpath, dirnames, filenames in os.walk(rp):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "target" and d != "__pycache__")
            for fn in sorted(filenames):
                fp = Path(dirpath) / fn
                if fp.suffix.lower() in ext_set and fp.is_file() and fp.stat().st_size <= max_file_bytes:
                    found.append(fp.resolve())
    return found


class SeedNGramBuilder:
    """Bounded n-gram counter that exports deterministic `CNGM` v1 binary payloads."""

    def __init__(self, max_n: int = 4) -> None:
        if max_n < 2 or max_n > 255:
            raise ValueError(f"max_n must be in 2..255, got {max_n}")
        self.max_n = int(max_n)
        self.counts: List[Dict[Tuple[int, ...], Counter]] = [defaultdict(Counter) for _ in range(self.max_n - 1)]
        self.total_tokens = 0

    def train_tokens(self, tokens: Sequence[int]) -> None:
        """Observe one contiguous token stream without crossing document boundaries."""
        if len(tokens) < 2:
            return
        ids = [int(t) for t in tokens]
        self.total_tokens += len(ids)
        max_ctx = self.max_n - 1
        for idx in range(1, len(ids)):
            nxt = ids[idx]
            lim = min(idx, max_ctx)
            for ctx_len in range(1, lim + 1):
                ctx = tuple(ids[idx - ctx_len : idx])
                self.counts[ctx_len - 1][ctx][nxt] += 1

    def export_records(
        self,
        min_count: int = 2,
        top_k_per_ctx: int = 1,
        max_records: int = 200_000,
    ) -> List[Tuple[int, Tuple[int, ...], int, int]]:
        """Select top-`min_count` transitions ranked cleanly by frequency and longer contexts."""
        if min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {min_count}")
        if top_k_per_ctx < 1:
            raise ValueError(f"top_k_per_ctx must be >= 1, got {top_k_per_ctx}")

        raw_recs: List[Tuple[int, Tuple[int, ...], int, int]] = []
        for idx, tbl in enumerate(self.counts):
            ctx_len = idx + 1
            for ctx, counter in tbl.items():
                kept = 0
                for nxt, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                    if cnt < min_count:
                        break
                    raw_recs.append((ctx_len, ctx, int(nxt), int(cnt)))
                    kept += 1
                    if kept >= top_k_per_ctx:
                        break

        if max_records > 0 and len(raw_recs) > max_records:
            raw_recs.sort(key=lambda r: (-r[3], -r[0], r[1], r[2]))
            raw_recs = raw_recs[:max_records]

        raw_recs.sort(key=lambda r: (r[0], r[1], -r[3], r[2]))
        return raw_recs

    def to_cngm_bytes(
        self,
        min_count: int = 2,
        top_k_per_ctx: int = 1,
        max_records: int = 200_000,
    ) -> bytes:
        """Serialize filtered records in exact `CNGM` v1 wire format."""
        recs = self.export_records(min_count=min_count, top_k_per_ctx=top_k_per_ctx, max_records=max_records)
        buf = bytearray(struct.pack("<4sIII", CNGM_MAGIC, CNGM_VERSION, self.max_n, len(recs)))
        for ctx_len, ctx, nxt, cnt in recs:
            buf.append(ctx_len & 0xFF)
            buf.extend(struct.pack(f"<{ctx_len}I", *ctx))
            buf.extend(struct.pack("<II", nxt, cnt))
        return bytes(buf)


def parse_cngm_bytes(data: bytes) -> Tuple[int, List[Tuple[int, Tuple[int, ...], int, int]]]:
    """Parse and strictly validate `CNGM` v1 binary seed bytes."""
    if len(data) < 16:
        raise ValueError(f"CNGM truncated: {len(data)} < 16")
    magic, ver, max_n, rec_cnt = struct.unpack("<4sIII", data[:16])
    if magic != CNGM_MAGIC:
        raise ValueError(f"CNGM bad magic: {magic!r}")
    if ver != CNGM_VERSION:
        raise ValueError(f"CNGM unsupported version: {ver}")
    pos = 16
    recs = []
    for idx in range(rec_cnt):
        if pos >= len(data):
            raise ValueError(f"CNGM truncated at record {idx}/{rec_cnt}")
        ctx_len = data[pos]
        pos += 1
        need = ctx_len * 4 + 8
        if ctx_len == 0 or pos + need > len(data):
            raise ValueError(f"CNGM invalid record {idx}: ctx_len={ctx_len}")
        ctx = struct.unpack(f"<{ctx_len}I", data[pos : pos + ctx_len * 4])
        pos += ctx_len * 4
        nxt, cnt = struct.unpack("<II", data[pos : pos + 8])
        pos += 8
        recs.append((ctx_len, ctx, nxt, cnt))
    if pos != len(data):
        raise ValueError(f"CNGM trailing bytes: {pos} != {len(data)}")
    return max_n, recs


def build_seed_file(
    corpus_roots: Sequence[Path],
    out_path: Path,
    encode_fn: Callable[[str], List[int]],
    max_n: int = 4,
    min_count: int = 2,
    top_k_per_ctx: int = 1,
    max_records: int = 200_000,
    max_files: int = 2000,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """Mine corpora and write validated atomic `.cngm` file. Returns `(n_files, total_tokens, bytes)`."""
    files = iter_corpus_files(corpus_roots)
    if not files:
        raise ValueError(f"Zero matching corpus files found under {[str(r) for r in corpus_roots]}")
    files = files[:max_files]
    builder = SeedNGramBuilder(max_n=max_n)
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if text:
            ids = encode_fn(text)
            builder.train_tokens(ids)

    payload = builder.to_cngm_bytes(
        min_count=min_count,
        top_k_per_ctx=top_k_per_ctx,
        max_records=max_records,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".cngm.tmp")
    try:
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    sz = out_path.stat().st_size
    if verbose:
        _, recs = parse_cngm_bytes(payload)
        print(
            f"[OK] Wrote {out_path} ({sz:,} B, {len(recs):,} seed transitions from {len(files)} files, {builder.total_tokens:,} tokens)"
        )
    return len(files), builder.total_tokens, sz


def _self_test() -> int:
    """Run 6 fail-closed self-test gates validating CNGM serialization, mining, capping, and errors."""
    failures = 0

    # Gate 1: Empty CNGM header & roundtrip
    try:
        b0 = SeedNGramBuilder(max_n=4).to_cngm_bytes(min_count=1)
        assert len(b0) == 16
        mx, recs = parse_cngm_bytes(b0)
        assert mx == 4 and recs == []
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    # Gate 2: Fail-closed on corrupt/truncated headers & missing corpus
    try:
        for bad in (b"BAD!", b"CNGM\x02\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00", b"CNGM\x01\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00"):
            try:
                parse_cngm_bytes(bad)
                raise AssertionError(f"Expected ValueError for {bad!r}")
            except ValueError:
                pass
        try:
            iter_corpus_files([Path("/nonexistent/corpus/dir")])
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass
    except Exception as exc:
        print(f"[Gate 2 FAIL] {exc}")
        failures += 1

    # Gate 3: Frequency ordering, min_count filter, and max_records cap
    try:
        sb = SeedNGramBuilder(max_n=3)
        sb.train_tokens([10, 20, 30, 10, 20, 30, 10, 20, 99, 10, 40, 50])
        recs_min2 = sb.export_records(min_count=2, top_k_per_ctx=1)
        assert (2, (10, 20), 30, 2) in recs_min2
        assert all(r[1] != (10, 40) for r in recs_min2)
        recs_cap1 = sb.export_records(min_count=2, top_k_per_ctx=1, max_records=1)
        assert len(recs_cap1) == 1
    except Exception as exc:
        print(f"[Gate 3 FAIL] {exc}")
        failures += 1

    # Gate 4-6: End-to-end directory scan + deterministic encoding + strict binary verification
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        cdir = td / "corpus"
        cdir.mkdir()
        (cdir / "proof.tex").write_text(
            "\\begin{theorem} For all $n$, $\\sum_{k=1}^n k = n(n+1)/2$. \\end{theorem}\n" * 3,
            encoding="utf-8",
        )
        (cdir / "skip.bin").write_bytes(b"\x00\xff" * 64)
        out_p = td / "seed.cngm"

        def simple_tok(s: str) -> List[int]:
            return [sum(ord(ch) for ch in w) % 5000 + 10 for w in s.split()]

        try:
            nf, ntok, sz = build_seed_file(
                [cdir],
                out_p,
                encode_fn=simple_tok,
                max_n=4,
                min_count=2,
                top_k_per_ctx=1,
                max_records=1000,
                verbose=False,
            )
            assert nf == 1 and ntok > 15 and sz == out_p.stat().st_size
            mx, recs = parse_cngm_bytes(out_p.read_bytes())
            assert mx == 4 and len(recs) > 0
            assert all(r[3] >= 2 for r in recs)
        except Exception as exc:
            print(f"[Gate 4-6 FAIL] {exc}")
            failures += 1

    if failures == 0:
        print("[PASS] build_spec_seed.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser(description="Build CNGM static n-gram speculative seed table")
    ap.add_argument("--corpus", type=Path, action="append", required=True, help="Corpus directory or file (repeatable)")
    ap.add_argument("--tokenizer", type=Path, required=True, help="Path to tokenizer.json or HF model dir")
    ap.add_argument("--out", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_math_seed.cngm"))
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=200_000)
    ap.add_argument("--max-files", type=int, default=2000)
    args = ap.parse_args()

    enc = load_text_encoder(args.tokenizer)
    build_seed_file(
        args.corpus,
        args.out.resolve(),
        encode_fn=enc,
        max_n=args.max_n,
        min_count=args.min_count,
        top_k_per_ctx=args.top_k,
        max_records=args.max_records,
        max_files=args.max_files,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

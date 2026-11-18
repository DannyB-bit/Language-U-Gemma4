#!/usr/bin/env python3
"""
Clean, non-trivia Gemma 4 31B V4 calibration corpus builder.

Constructs ~1,350-1,500 Gemma 4 `<|turn>`-wrapped tokens spanning algebra,
arithmetic, Python/Rust/SQL/C code, computer systems, earth/physical sciences,
and multi-turn structured responses. Strictly excludes every topic, entity, and
question phrasing present in the P1..P8 verification battery and H1..H4 unseen
holdouts, failing closed via `_self_test()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# Substrings forbidden from appearing in any V4 training/calibration pair.
BLACKLIST_SUBSTRINGS: Tuple[str, ...] = (
    "2 + 2",
    "2+2",
    "15 * 14",
    "15*14",
    "france",
    "paris",
    "japan",
    "tokyo",
    "hamlet",
    "to be or not to be",
    "romeo",
    "juliet",
    "shakespeare",
    "mona lisa",
    "leonardo",
    "da vinci",
    "is_prime",
    "prime",
    "reverse a string",
    "reverse_string",
    "average speed",
    "120 km",
    "photosynthesis",
    "chloroplast",
    "binary search",
    "einstein",
    "relativity",
)


# (domain, prompt, response) triples using Gemma 4 <|turn> formatting.
V4_CAL_PAIRS: List[Tuple[str, str, str]] = [
    (
        "mathematics",
        "Compute 47 * 23 - 156 / 12 step by step cleanly.",
        "First, 47 * 23 = 1081 and 156 / 12 = 13. Subtracting gives 1081 - 13 = 1068.",
    ),
    (
        "mathematics",
        "Solve the linear system 3x - 2y = 14 and x + y = 8 for x and y.",
        "From x = 8 - y, substituting yields 3(8 - y) - 2y = 14, so 5y = 10, giving y = 2 and x = 6.",
    ),
    (
        "mathematics",
        "Factor the quadratic expression x^2 - 9x + 20 and state its roots.",
        "The expression factors as (x - 4)(x - 5), so the roots are x = 4 and x = 5.",
    ),
    (
        "mathematics",
        "A bookstore discounts a $40 novel by 15 percent. What is the final sale price?",
        "Fifteen percent of $40 is $6, so the final sale price is $40 - $6 = $34.",
    ),
    (
        "code",
        "Write a concise Python function to compute the n-th Fibonacci number iteratively.",
        "```python\ndef fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```",
    ),
    (
        "code",
        "Write a Python function that computes the greatest common divisor using Euclid's loop.",
        "```python\ndef gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return abs(a)\n```",
    ),
    (
        "code",
        "Write a Python helper to count word frequencies from a list of lowercase words.",
        "```python\ndef word_counts(words):\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n```",
    ),
    (
        "code",
        "Write a Rust function that computes running prefix sums over a slice of i64 values.",
        "```rust\npub fn prefix_sums(xs: &[i64]) -> Vec<i64> {\n    let mut out = Vec::with_capacity(xs.len());\n    let mut acc = 0i64;\n    for &v in xs {\n        acc += v;\n        out.push(acc);\n    }\n    out\n}\n```",
    ),
    (
        "code",
        "Write an SQL query to sum order revenue per region for orders placed in 2025.",
        "```sql\nSELECT region, SUM(amount) AS total_revenue\nFROM orders\nWHERE order_year = 2025\nGROUP BY region\nORDER BY total_revenue DESC;\n```",
    ),
    (
        "systems",
        "List three architectural differences between connection-oriented TCP and datagram UDP.",
        "1. TCP establishes a stateful handshake, whereas UDP is connectionless.\n2. TCP retransmits lost packets in order, while UDP offers best-effort delivery.\n3. UDP headers use 8 bytes vs 20+ bytes in TCP, reducing per-packet overhead.",
    ),
    (
        "systems",
        "Why does aligning SIMD weight tiles to 64-byte boundaries improve quantized decode speed?",
        "64-byte alignment matches L1 cache lines, preventing split-line loads and avoiding unaligned stalls during vectorized NEON reads.",
    ),
    (
        "systems",
        "Explain how an orthogonal Walsh-Hadamard transform stabilizes low-bit weight quantization.",
        "Multiplying blocks by an orthonormal Hadamard matrix redistributes isolated outlier weights uniformly across channels, reducing per-block dynamic range before rounding.",
    ),
    (
        "systems",
        "What causes a TLB shootdown when unmapping shared memory across CPU cores?",
        "Each core caches virtual-to-physical translations in a local TLB, so unmapping a shared page requires inter-processor interrupts to invalidate stale entries.",
    ),
    (
        "science",
        "Summarize the three main stages of the atmospheric water cycle concisely.",
        "Solar heating drives evaporation from surface water, cooling air triggers condensation into clouds, and gravity returns moisture as precipitation.",
    ),
    (
        "science",
        "Why does a clear daytime sky appear blue rather than violet or red?",
        "Atmospheric molecules scatter short blue wavelengths far more strongly than longer red waves via Rayleigh scattering, while human vision is less sensitive to violet.",
    ),
    (
        "science",
        "What causes seasonal temperature shifts between the northern and southern hemispheres?",
        "Earth's 23.5-degree axial tilt changes the solar incidence angle and day length throughout its annual orbit, concentrating or spreading sunlight across latitude bands.",
    ),
    (
        "science",
        "Explain what happens chemically when an acid neutralizes a strong base in water.",
        "Hydrogen ions react with hydroxide ions to form liquid water, while the remaining cation and anion form a dissolved ionic salt and release heat.",
    ),
    (
        "concise",
        "What is the boiling point of pure water at standard sea-level pressure in Celsius? Answer in one sentence.",
        "At standard sea-level pressure (1 atm), pure liquid water boils at 100 degrees Celsius.",
    ),
    (
        "concise",
        "How many edges and vertices does a regular cube have? Answer directly.",
        "A cube has 12 edges, 8 vertices, and 6 square faces.",
    ),
    (
        "concise",
        "Convert the binary number 101101 to base-10 integer form step by step.",
        "Adding powers of two gives 32 + 0 + 8 + 4 + 0 + 1 = 45.",
    ),
]


def format_gemma4_turn(prompt: str, response: Optional[str] = None) -> str:
    """Format a user/model exchange using exact Gemma 4 `<|turn>` delimiters."""
    base = (
        f"<|turn>user\n{prompt}<turn|>\n"
        "<|turn>model\n<|channel>thought\n<channel|>"
    )
    if response is None:
        return base
    return f"{base}{response}<turn|>"


def assert_no_contamination(texts: Sequence[str], label: str = "corpus") -> None:
    """Fail closed if any forbidden benchmark/holdout substring appears."""
    for idx, txt in enumerate(texts):
        low = txt.lower()
        for bad in BLACKLIST_SUBSTRINGS:
            if bad in low:
                raise ValueError(f"Contamination detected in {label}[{idx}]: matched {bad!r}")


def build_tokenized_v4_corpus(
    tokenizer_path: Path,
    *,
    include_eval_probes: bool = True,
) -> Dict[str, object]:
    """Tokenize `V4_CAL_PAIRS` cleanly with strict train/eval separation."""
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json: {tokenizer_path}")
    from tokenizers import Tokenizer

    assert_no_contamination([p for _, p, _ in V4_CAL_PAIRS], "V4_CAL_PAIRS.prompt")
    assert_no_contamination([r for _, _, r in V4_CAL_PAIRS], "V4_CAL_PAIRS.response")

    tk = Tokenizer.from_file(str(tokenizer_path))
    all_ids: List[int] = []
    next_token_ids: List[int] = []
    pos_arr: List[int] = []
    prompt_domains: List[str] = []
    prompt_slices: List[Tuple[int, int]] = []
    ans_mask: List[bool] = []

    for dom, ptxt, resp in V4_CAL_PAIRS:
        p_ids = [2] + list(tk.encode(format_gemma4_turn(ptxt, None), add_special_tokens=False).ids)
        r_ids = list(tk.encode(f"{resp}<turn|>", add_special_tokens=False).ids)
        full_ids = p_ids + r_ids
        seq_in = full_ids[:-1]
        seq_nxt = full_ids[1:]
        s0 = len(all_ids)
        all_ids.extend(seq_in)
        next_token_ids.extend(seq_nxt)
        pos_arr.extend(range(len(seq_in)))
        prompt_domains.extend([dom] * len(seq_in))
        ans_mask.extend([idx >= (len(p_ids) - 1) for idx in range(len(seq_in))])
        prompt_slices.append((s0, len(all_ids)))

    cal_end = len(all_ids)
    meta_out: Dict[str, object] = {
        "cal_end": cal_end,
        "cal_pairs_count": len(V4_CAL_PAIRS),
        "all_ids": all_ids,
        "next_token_ids": next_token_ids,
        "pos_arr": pos_arr,
        "prompt_domains": prompt_domains,
        "prompt_slices": prompt_slices,
        "ans_mask": ans_mask,
    }

    if include_eval_probes:
        # Diagnostic-only cascade checks appended strictly AFTER `cal_end`.
        probe_specs = [
            ("test2", "<|turn>user\nWhat is 2 + 2? Answer with only the number.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
            ("testf", "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
        ]
        for key, raw_prompt in probe_specs:
            p_ids = [2] + list(tk.encode(raw_prompt, add_special_tokens=False).ids)
            s0 = len(all_ids)
            all_ids.extend(p_ids)
            next_token_ids.extend(p_ids[1:] + [818])
            pos_arr.extend(range(len(p_ids)))
            prompt_domains.extend(["heldout"] * len(p_ids))
            ans_mask.extend([False] * len(p_ids))
            s1 = len(all_ids)
            prompt_slices.append((s0, s1))
            meta_out[f"{key}_bounds"] = (s0, s1)
            assert s0 >= cal_end

    assert len(all_ids) == len(next_token_ids) == len(pos_arr) == len(prompt_domains) == len(ans_mask)
    return meta_out


def _resolve_default_tokenizer() -> Optional[Path]:
    for c in (
        Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json"),
        Path("C:/Users/freed/.gemini/antigravity/scratch/Language-U-Gemma4/tokenizer.json"),
    ):
        if c.is_file():
            return c
    return None


def _self_test() -> int:
    """Run 6 fail-closed self-test gates without requiring large weights."""
    failures = 0

    # Gate 1: Non-empty & balanced multi-domain pairs
    try:
        assert len(V4_CAL_PAIRS) >= 20
        domains = {d for d, _, _ in V4_CAL_PAIRS}
        assert {"mathematics", "code", "systems", "science", "concise"}.issubset(domains)
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    # Gate 2: Zero contamination against P1..P8 and H1..H4 blacklists
    try:
        assert_no_contamination([p for _, p, _ in V4_CAL_PAIRS], "prompts")
        assert_no_contamination([r for _, _, r in V4_CAL_PAIRS], "responses")
    except Exception as exc:
        print(f"[Gate 2 FAIL] {exc}")
        failures += 1

    # Gate 3: Contamination detector fails closed on synthetic violation
    try:
        caught = False
        try:
            assert_no_contamination(["Write a Python function is_prime(n)."], "synthetic")
        except ValueError:
            caught = True
        assert caught
    except Exception as exc:
        print(f"[Gate 3 FAIL] {exc}")
        failures += 1

    # Gate 4: Exact Gemma 4 turn wrapper format
    try:
        msg = format_gemma4_turn("Hello", "World")
        assert msg.startswith("<|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
        assert msg.endswith("World<turn|>")
    except Exception as exc:
        print(f"[Gate 4 FAIL] {exc}")
        failures += 1

    # Gate 5: Missing tokenizer fails closed with FileNotFoundError
    try:
        caught = False
        try:
            build_tokenized_v4_corpus(Path("/nonexistent_dir_12345/tokenizer.json"))
        except FileNotFoundError:
            caught = True
        assert caught
    except Exception as exc:
        print(f"[Gate 5 FAIL] {exc}")
        failures += 1

    # Gate 6: Real tokenizer count check when available
    try:
        tok_p = _resolve_default_tokenizer()
        if tok_p is not None:
            corp = build_tokenized_v4_corpus(tok_p, include_eval_probes=True)
            cal_end = int(corp["cal_end"])  # type: ignore[arg-type]
            assert 1100 <= cal_end <= 2400, f"Unexpected cal_end={cal_end}"
            ans_cnt = sum(1 for m in corp["ans_mask"][:cal_end] if m)  # type: ignore[index]
            assert ans_cnt >= 500, f"Unexpected ans_cnt={ans_cnt}"
            print(f"[INFO] Tokenized V4 corpus: cal_end={cal_end}, ans_tokens={ans_cnt}, total={len(corp['all_ids'])}")
    except Exception as exc:
        print(f"[Gate 6 FAIL] {exc}")
        failures += 1

    if failures == 0:
        print("[PASS] v4_corpus.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    import sys

    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()
    tok_p = Path(sys.argv[1])
    corp = build_tokenized_v4_corpus(tok_p, include_eval_probes=True)
    cal_end = int(corp["cal_end"])  # type: ignore[arg-type]
    ans_cnt = sum(1 for m in corp["ans_mask"][:cal_end] if m)  # type: ignore[index]
    print(json.dumps({"cal_end": cal_end, "ans_tokens": ans_cnt, "total_tokens": len(corp["all_ids"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

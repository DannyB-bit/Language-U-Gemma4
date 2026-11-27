#!/usr/bin/env python3
"""Unit and negative fail-closed tests for `scripts/verify_receipt_parity.py`."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_receipt_parity import main, verify_repo_receipts


class TestReceiptParity(unittest.TestCase):
    def test_repo_receipts_and_links_pass(self):
        report = verify_repo_receipts(REPO_ROOT)
        self.assertTrue(report.passed, f"Receipt parity errors: {report.errors}")
        self.assertGreaterEqual(report.checked_jsons, 35)
        self.assertGreaterEqual(report.checked_links, 30)
        self.assertGreaterEqual(report.checked_claims, 150)

    def test_tampered_sweep_metric_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_root = Path(tmp) / "repo"
            for item in (
                "LICENSE",
                "NOTICE",
                "README.md",
                "agent",
                "fleet-consensus-v1",
                "submissions",
                "scripts",
                "compiler",
                "neon_runner",
                "trinity_retrieval/README.md",
                "trinity_retrieval/formal_verification",
                "trinity_retrieval/swe_navigation_benchmark_results.json",
                "docs",
                "experiments/gemma270_luj",
                "experiments/joint_glyphs",
                "experiments/native_language",
                "experiments/packed_corrections",
                "experiments/run270_rust/RECEIPTS.md",
                "experiments/run270_rust/FRONTIER_RECEIPTS.md",
                "experiments/run270_rust/run270_exact_gen.log",
                "experiments/run270_rust/run270_glyph_gen.log",
                "experiments/run270_rust/ref_gemma3_native_arith.json",
                "experiments/run270_rust/calib/31b/bench_cs_lmhead.c",
                "experiments/run270_rust/calib/31b/receipts",
                "experiments/run270_rust/benchmarks",
                "experiments/run270_rust/src/t16c.rs",
                "experiments/run270_rust/evidence",
            ):
                src = REPO_ROOT / item
                dst = copy_root / item
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

            sweep_p = copy_root / "experiments" / "run270_rust" / "evidence" / "g270_sweep" / "sweep_bench5_all23.json"
            data = json.loads(sweep_p.read_text(encoding="utf-8"))
            data[0]["min_cos"] = 0.123456
            sweep_p.write_text(json.dumps(data), encoding="utf-8")

            report = verify_repo_receipts(copy_root)
            self.assertFalse(report.passed)
            self.assertTrue(any("sweep[exact_bf16].min_cos" in err for err in report.errors), report.errors)

    def test_selftest_six_gates(self):
        self.assertEqual(main(["--selftest"]), 0)


if __name__ == "__main__":
    unittest.main()

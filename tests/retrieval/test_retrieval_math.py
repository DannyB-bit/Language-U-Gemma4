#!/usr/bin/env python3
"""Standalone deterministic unit tests for Trinity Retrieval S^7 math and Trust Law."""

import sys
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trinity_retrieval.language_u_octonion import (
    H8_NORMALIZED,
    TRUE_NORTH_GLYPH,
    batch_geodesic_resonance,
    geodesic_resonance,
    project_batch_to_s7,
    project_mrl_to_s7,
)
from trinity_retrieval.affine_hadamard_h64 import (
    H64_NORMALIZED,
    h64_resonance,
    hamming_distance64,
    project_mrl_to_h64,
    to_bitpack64,
)
from trinity_retrieval.noise_stability_sieve import NoiseStabilitySieve
from trinity_retrieval.battery_contract import assess_case


class TestTrinityRetrievalMath(unittest.TestCase):
    def test_h8_orthonormality_and_s7_projection(self):
        err = float(np.linalg.norm(H8_NORMALIZED @ H8_NORMALIZED.T - np.eye(8, dtype=np.float32)))
        self.assertLess(err, 1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(TRUE_NORTH_GLYPH)), 1.0, places=5)

        rng = np.random.default_rng(2026)
        v = rng.standard_normal(768).astype(np.float32)
        g = project_mrl_to_s7(v)
        self.assertEqual(g.shape, (8,))
        self.assertAlmostEqual(float(np.linalg.norm(g)), 1.0, places=5)
        self.assertAlmostEqual(geodesic_resonance(g, g), 1.0, places=5)

        # Verify exact 96.0x byte compression ratio for 1,962 x 768 FP32 -> 1,962 x 8 FP32
        mat = rng.standard_normal((1962, 768)).astype(np.float32)
        s7 = project_batch_to_s7(mat)
        self.assertEqual(mat.nbytes, 6_027_264)
        self.assertEqual(s7.nbytes, 62_784)
        self.assertEqual(mat.nbytes / s7.nbytes, 96.0)
        res = batch_geodesic_resonance(s7, g)
        self.assertEqual(res.shape, (1962,))
        self.assertTrue(np.all((res >= 0.0) & (res <= 1.0)))

    def test_h64_orthonormality_and_1bit_pack(self):
        err = float(np.linalg.norm(H64_NORMALIZED @ H64_NORMALIZED.T - np.eye(64, dtype=np.float32)))
        self.assertLess(err, 1e-5)
        rng = np.random.default_rng(77)
        v = rng.standard_normal(768).astype(np.float32)
        h = project_mrl_to_h64(v)
        self.assertEqual(h.shape, (64,))
        self.assertAlmostEqual(float(np.linalg.norm(h)), 1.0, places=5)
        packed = to_bitpack64(h)
        self.assertEqual(hamming_distance64(packed, packed), 0)
        self.assertAlmostEqual(h64_resonance(h, h), 1.0, places=5)

    def test_noise_sieve_and_battery_contract(self):
        sieve = NoiseStabilitySieve(vocab={"sqlite", "fts5", "index", "search", "gemma"})
        ok = sieve.evaluate("sqlite fts5 search index")
        self.assertEqual(ok.verdict, "PASS")
        bad = sieve.evaluate("xqzjflkw pvbmzkqw")
        self.assertEqual(bad.verdict, "REJECT_MISS")

        bucket_s, _ = assess_case("kafka", "exact", {"slug": "kafka", "roadmap": "backend"}, None)
        self.assertEqual(bucket_s, "strong")
        bucket_fp, _ = assess_case(None, "miss", {"slug": "oops", "roadmap": "backend"}, None)
        self.assertEqual(bucket_fp, "fail")
        bucket_miss, _ = assess_case(None, "miss", None, None)
        self.assertEqual(bucket_miss, "miss")


if __name__ == "__main__":
    unittest.main()

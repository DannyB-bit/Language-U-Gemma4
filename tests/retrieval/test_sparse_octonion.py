#!/usr/bin/env python3
"""Unit tests for Sparse Octonion Directional Router & Expert Streaming Engine."""

import os
import shutil
import sys
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PKG_DIR = REPO_ROOT / "trinity_retrieval"
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(REPO_ROOT))

from sparse_octonion_streamer import SparseOctonionRouter, SparseOctonionStreamer, CLIFFORD_BASES

TEST_EXPERTS_DIR = HERE / "_tmp_octonion_experts" / "test_octonion_experts"


class TestSparseOctonionStreamer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TEST_EXPERTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.streamer = SparseOctonionStreamer(
            experts_dir=TEST_EXPERTS_DIR,
            in_dim=512,
            out_dim=512,
            top_k=2
        )

    @classmethod
    def tearDownClass(cls):
        if TEST_EXPERTS_DIR.exists():
            shutil.rmtree(TEST_EXPERTS_DIR, ignore_errors=True)

    def test_directional_routing_bounds(self):
        router = SparseOctonionRouter(top_k=2)
        x = torch.randn(4, 512)
        indices, weights = router.route(x)
        
        self.assertEqual(indices.shape, (4, 2))
        self.assertEqual(weights.shape, (4, 2))
        
        # Softmax weights must sum to 1.0 per sample
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        
        # Indices must be valid 0..7 octants
        self.assertTrue((indices >= 0).all() and (indices < 8).all())

    def test_selective_expert_loading(self):
        x = torch.randn(1, 512)
        out, active = self.streamer.forward_sparse(x)
        
        self.assertEqual(out.shape, (1, 512))
        self.assertEqual(len(active), 2, "Only top-2 experts should be loaded for batch size 1")
        self.assertFalse(torch.isnan(out).any())

    def test_batch_execution(self):
        x = torch.randn(5, 512)
        out, active = self.streamer.forward_sparse(x)
        self.assertEqual(out.shape, (5, 512))
        # At most 5 * 2 = 10 (capped by 8 total)
        self.assertLessEqual(len(active), 8)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for Air-Trinity Layer-Streaming Inference Engine."""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PKG_DIR = REPO_ROOT / "trinity_retrieval"
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(REPO_ROOT))

from air_trinity_streamer import AirTrinityLayerStreamer, LayerPrefetcher, rms_norm, apply_rotary_emb
from gemma2_paths import weights_file

WEIGHTS_PATH = weights_file()


class TestAirTrinityStreamer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(WEIGHTS_PATH):
            raise unittest.SkipTest(f"Weights file not found at {WEIGHTS_PATH}")
        cls.streamer = AirTrinityLayerStreamer(WEIGHTS_PATH, num_layers=24)

    def test_rms_norm(self):
        x = torch.randn(2, 4, 512)
        w = torch.zeros(512)
        out = rms_norm(x, w)
        self.assertEqual(out.shape, x.shape)
        # Variance of output should be close to 1
        var = out.pow(2).mean(-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

    def test_rope_embedding(self):
        x = torch.randn(1, 4, 256)
        out = apply_rotary_emb(x, dim=256, seq_len=4)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_prefetcher_lifecycle(self):
        pf = LayerPrefetcher(WEIGHTS_PATH, num_layers=2)
        pf.start(start_layer=0)
        idx0, w0 = pf.get_next_layer()
        self.assertEqual(idx0, 0)
        self.assertIn("self_attn.q_proj.weight", w0)
        idx1, w1 = pf.get_next_layer()
        self.assertEqual(idx1, 1)
        self.assertIn("self_attn.q_proj.weight", w1)
        pf.stop()

    def test_layer_streaming_inference(self):
        tokens = torch.tensor([[101, 2054, 2003, 102]], dtype=torch.long)
        vec = self.streamer.encode(tokens, normalize=True)
        self.assertEqual(vec.shape[0], 1)
        norm = np.linalg.norm(vec[0])
        self.assertAlmostEqual(norm, 1.0, places=3)
        self.assertFalse(np.isnan(vec).any())

    def test_semantic_directionality(self):
        # Two identical tokens should produce identical embeddings
        t1 = torch.tensor([[101, 2000, 102]], dtype=torch.long)
        t2 = torch.tensor([[101, 2000, 102]], dtype=torch.long)
        v1 = self.streamer.encode(t1, normalize=True)[0]
        v2 = self.streamer.encode(t2, normalize=True)[0]
        sim = float(np.dot(v1, v2))
        self.assertAlmostEqual(sim, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

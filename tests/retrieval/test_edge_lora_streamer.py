#!/usr/bin/env python3
"""Unit tests for Edge LoRA Layer-Streaming Training Engine."""

import os
import sys
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PKG_DIR = REPO_ROOT / "trinity_retrieval"
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(REPO_ROOT))

from edge_lora_streamer import EdgeLoRAStreamer, StreamingLoRALinear
from gemma2_paths import weights_file

WEIGHTS_PATH = weights_file()


class TestEdgeLoRAStreamer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(WEIGHTS_PATH):
            raise unittest.SkipTest(f"Weights file not found at {WEIGHTS_PATH}")
        cls.streamer = EdgeLoRAStreamer(WEIGHTS_PATH, num_layers=2, r=4, lr=1e-3)

    def test_lora_linear_init(self):
        adapter = StreamingLoRALinear(in_features=512, out_features=1024, r=4)
        # B should be all zeros initially
        self.assertTrue(torch.all(adapter.lora_B == 0))
        # When B is zero, adapter output should exactly equal base output
        x = torch.randn(1, 4, 512)
        base_w = torch.randn(1024, 512)
        base_out = torch.nn.functional.linear(x, base_w)
        combined_out = adapter(x, base_w)
        self.assertTrue(torch.allclose(base_out, combined_out, atol=1e-5))

    def test_authentic_gradient_flow(self):
        q_ids = torch.tensor([[101, 2054, 102]], dtype=torch.long)
        t_ids = torch.tensor([[101, 2054, 102]], dtype=torch.long)
        
        # Ensure gradients are zeroed initially
        self.streamer.optimizer.zero_grad()
        loss = self.streamer.train_step(q_ids, t_ids)
        self.assertFalse(torch.isnan(torch.tensor(loss)))
        
        # Verify gradients were calculated on active adapter parameters
        adapter_0_q = self.streamer.adapters["layer_0_q"]
        self.assertIsNotNone(adapter_0_q.lora_A.grad)
        self.assertIsNotNone(adapter_0_q.lora_B.grad)

    def test_adamw_weight_update(self):
        q_ids = torch.tensor([[101, 2054, 102]], dtype=torch.long)
        t_ids = torch.tensor([[101, 2054, 102]], dtype=torch.long)
        
        b_before = self.streamer.adapters["layer_0_q"].lora_B.clone().detach()
        self.streamer.train_step(q_ids, t_ids)
        b_after = self.streamer.adapters["layer_0_q"].lora_B.clone().detach()
        
        delta = torch.norm(b_after - b_before).item()
        self.assertGreater(delta, 0.0, "Weights must update after AdamW step")


if __name__ == "__main__":
    unittest.main()

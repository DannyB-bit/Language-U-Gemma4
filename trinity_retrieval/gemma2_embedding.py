#!/usr/bin/env python3
"""Gemma2 Embedding Engine — Physical Neural Inference for Trinity Hybrid Retrieval.

Replaces BAAI/bge-small-en-v1.5 with google/embeddinggemma-2 (740M params, 768-D MRL).
Features:
  - Authentic PyTorch forward passes on physical silicon (CUDA/MPS/CPU).
  - Native Matryoshka Representation Learning (MRL) dimension slicing (e.g. 768, 512, 256).
  - Asymmetric prompt prefixing: "task: search result | query: " for queries.
  - Zero synthetic simulation loops: all vectors computed from genuine model weights.
  - Embedded self-verification battery (_self_test) conforming to Standard 6.
"""

import sys
import time
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def _resolve_gemma2_model_path(default_id: str = "google/embeddinggemma-2") -> str:
    """Resolve the physical model directory cross-platform.

    Priority: $GEMMA2_WEIGHTS_DIR -> repo models/embeddinggemma-2 (shards
    auto-reassembled + SHA256-verified) -> $GEMMA2_MODEL_PATH -> HF hub id.
    gemma2_paths.py is the single source of truth for this resolution — add
    tiers there, never machine-specific hardcoded paths here.
    """
    try:
        from . import gemma2_paths
    except ImportError:
        import gemma2_paths

    return gemma2_paths.model_dir() or default_id


class Gemma2EmbeddingModel:
    """Authentic Google EmbeddingGemma-2 Model Wrapper for Corpus and Query Inference."""

    DEFAULT_MODEL_ID = "google/embeddinggemma-2"
    DEFAULT_DIM = 768
    QUERY_PREFIX = "task: search result | query: "

    def __init__(
        self,
        model_id: Optional[str] = None,
        target_dim: int = DEFAULT_DIM,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id or _resolve_gemma2_model_path(self.DEFAULT_MODEL_ID)
        self.target_dim = target_dim
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if dtype is None:
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = dtype

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.load_latency_s = time.perf_counter() - t0

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query with mandatory asymmetric task prefix."""
        prompt = f"{self.QUERY_PREFIX}{query}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # POOLING FIX (Researcher A, 2026-10-17; restored 2026-10-18 after a
            # WIP edit reverted it): Gemma-2's model card mandates MEAN pooling
            # over real tokens (attention-mask weighted); last-token pooling is
            # the known Gemma-2 trap. The shipped corpus vectors
            # (roadmap_gemma_emb.npz) are mean-pooled — queries MUST pool
            # identically or every query-doc cosine is broken.
            mask = inputs.attention_mask.unsqueeze(-1).bool()
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = inputs.attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (summed / counts)[0].float()
            vec = pooled[:self.target_dim].cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            return vec

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        is_query: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """Embed a list of documents or queries on physical silicon in batched forward passes."""
        if not texts:
            return np.empty((0, self.target_dim), dtype=np.float32)

        all_vectors = []
        n_total = len(texts)

        for i in range(0, n_total, batch_size):
            batch_texts = texts[i : i + batch_size]
            if is_query:
                batch_prompts = [f"{self.QUERY_PREFIX}{t}" for t in batch_texts]
            else:
                batch_prompts = batch_texts

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # POOLING FIX (Researcher A, 2026-10-17; restored 2026-10-18):
                # mean pooling over real tokens per Gemma-2 model card; replaces
                # last-token extraction (the known Gemma-2 trap). Same change as
                # embed_query — queries and documents must pool identically.
                mask = inputs.attention_mask.unsqueeze(-1).bool()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = inputs.attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                reps = (summed / counts)[:, :self.target_dim].float()

                batch_tensor = reps.cpu().numpy().astype(np.float32)
                # L2 normalize each row
                norms = np.linalg.norm(batch_tensor, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                batch_tensor /= norms
                all_vectors.append(batch_tensor)

            if verbose and (i + batch_size < n_total):
                print(f"   Embedded {min(i + batch_size, n_total)}/{n_total} documents...", flush=True)

        return np.vstack(all_vectors)

    @classmethod
    def _self_test(cls) -> bool:
        """Standard 6 verification: executes genuine forward passes and asserts semantic margin."""
        print(f"[Self-Test] Initializing {cls.DEFAULT_MODEL_ID}...", flush=True)
        engine = cls(target_dim=768)
        print(f"[Self-Test] Weights loaded in {engine.load_latency_s:.2f}s on {engine.device}.", flush=True)

        v_zk = engine.embed_query("zero knowledge proofs snark pairing")
        v_zt = engine.embed_query("zero trust firewall perimeter access")
        v_crypto = engine.embed_query("cryptographic pairing snark precompile")

        sim_crypto = float(np.dot(v_zk, v_crypto))
        sim_zt = float(np.dot(v_zk, v_zt))
        margin = sim_crypto - sim_zt

        print(f"[Self-Test] ZK <-> Crypto Cosine: {sim_crypto:.4f}")
        print(f"[Self-Test] ZK <-> Zero-Trust Cosine: {sim_zt:.4f}")
        print(f"[Self-Test] Semantic Disambiguation Margin: {margin:+.4f}")

        assert margin > 0.0, f"Disambiguation margin failed: {margin} <= 0"
        assert v_zk.shape == (768,), f"Unexpected embedding shape: {v_zk.shape}"
        print("[Self-Test] SUCCESS: Physical neural forward pass verified.", flush=True)
        return True


if __name__ == "__main__":
    success = Gemma2EmbeddingModel._self_test()
    sys.exit(0 if success else 1)

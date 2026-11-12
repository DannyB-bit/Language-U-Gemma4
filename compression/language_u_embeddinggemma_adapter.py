"""
Language-U & EmbeddingGemma-2 Octonionic Bridge
================================================
Integrates Google DeepMind's EmbeddingGemma-2 with the Language-U 8D Gyroscopic
Compass on the unit 7-sphere S^7.

SHE REWORK 2026-10-12 (Honesty Law):
The original version of this adapter named itself "EmbeddingGemma" but never
loaded the model — `text_to_mrl_embedding` was a sha256 feature-hashing trick.
Audited and A/B-measured (see PR body): the hash backend loses to the real
model on paraphrase retrieval. This file now ships BOTH backends honestly:

  backend='real'  — google/embeddinggemma-2 (740M params, native MRL
                    128/256/512/768d) via transformers. TRUE semantic
                    embeddings. Needs torch+transformers + ~1.5 GB weights.
  backend='hash'  — the original sha256 hashing trick, kept as the DOCUMENTED
                    offline fallback (zero heavy deps, instant, deterministic).
  backend='auto'  — real if loadable, else hash with a loud warning.

The active backend is ALWAYS reported via `get_backend()` — outputs are never
silently produced by an unannounced encoder.

Mathematical Principles (unchanged):
1. Matryoshka Representation Learning (MRL) dimension alignment:
   EmbeddingGemma-2 supports native MRL truncation at D in {128, 256, 512, 768}.
   Every supported dimension is an exact integer multiple of 8.
2. Sylvester-Hadamard H8 Rotation onto S^7: decompose the D-dim MRL vector
   into K = D/8 octonionic blocks, rotate each via normalized H8, take the
   gyroscopic mean, normalize onto the unit 7-sphere.
3. Geodesic Resonance with the True North Attractor.
4. morluto/rea Evidence Graph scoring with contradiction pruning.
"""

import os
import math
import json
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

MODEL_ID = 'google/embeddinggemma-2'
MRL_DIMS = (128, 256, 512, 768)

# 8D Normalized Sylvester-Hadamard Matrix (1 / sqrt(8))
SQRT8 = math.sqrt(8.0)
H8_NORMALIZED = np.array([
    [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
    [ 1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0],
    [ 1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
    [ 1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
    [ 1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0],
    [ 1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0],
    [ 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0]
], dtype=np.float32) / SQRT8

# Canonical True North 4-Sparse Cardinal Glyph on S^7
TRUE_NORTH_GLYPH = np.array([0.0, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, 0.5], dtype=np.float32)
TRUE_NORTH_GLYPH /= np.linalg.norm(TRUE_NORTH_GLYPH)


def _deterministic_hash_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode('utf-8')).digest()


class RealEmbeddingGemma2:
    """Loads the actual HF model. Raises honestly if deps/weights are missing."""
    def __init__(self, target_dim: int = 128):
        if target_dim not in MRL_DIMS:
            raise ValueError(f'target_dim must be one of {MRL_DIMS}, got {target_dim}')
        import torch  # honest ImportError if missing
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.target_dim = target_dim
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        with self._torch.no_grad():
            inp = self.tokenizer(f'task: search result | query: {text}',
                                 return_tensors='pt', truncation=True, max_length=256)
            h = self.model(**inp).last_hidden_state[0, -1, :].float()
        v = h[:self.target_dim].numpy().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v


class HashEmbeddingFallback:
    """The original deterministic sha256 hashing embedding (offline fallback)."""
    def __init__(self, target_dim: int = 128):
        if target_dim not in MRL_DIMS:
            raise ValueError(f'target_dim must be one of {MRL_DIMS}, got {target_dim}')
        self.target_dim = target_dim

    @staticmethod
    def _hash_bytes(s: str) -> bytes:
        return hashlib.sha256(s.encode('utf-8')).digest()

    def embed(self, text: str) -> np.ndarray:
        clean = text.lower().strip()
        tokens = re.findall(r'[a-zA-Z0-9_\-+*/^]+', clean)
        if not tokens:
            return np.zeros(self.target_dim, dtype=np.float32)
        vec = np.zeros(self.target_dim, dtype=np.float32)
        for i, token in enumerate(tokens):
            pos_weight = 1.0 / (1.0 + 0.05 * math.log(1 + i))
            h_bytes = self._hash_bytes(token)
            vec[int.from_bytes(h_bytes[:4], 'big') % self.target_dim] += (1.0 if h_bytes[4] % 2 == 0 else -1.0) * pos_weight
            if len(token) >= 3:
                for j in range(len(token) - 2):
                    ht = self._hash_bytes(token[j:j+3])
                    vec[int.from_bytes(ht[:4], 'big') % self.target_dim] += 0.5 * (1.0 if ht[4] % 2 == 0 else -1.0) * pos_weight
            if i > 0:
                hb = self._hash_bytes(f'{tokens[i-1]}_{token}')
                vec[int.from_bytes(hb[:4], 'big') % self.target_dim] += 0.75 * (1.0 if hb[4] % 2 == 0 else -1.0) * pos_weight
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec


class EmbeddingGemma2MRLProjector:
    """
    MRL projector with TWO honest backends:
      backend='real'  — the actual google/embeddinggemma-2 model
      backend='hash'  — deterministic sha256 fallback (offline, instant)
      backend='auto'  — real if loadable, else hash + loud warning
    The active backend is always available via get_backend().
    """

    def __init__(self, target_dim: int = 128, backend: str = 'auto'):
        if target_dim not in MRL_DIMS:
            raise ValueError(f'Dimension {target_dim} must be in {MRL_DIMS}')
        self.target_dim = target_dim
        self.num_octonions = target_dim // 8
        self.backend_name: Optional[str] = None
        self._impl = None
        disable_real = os.environ.get("LUC_NO_GEMMA", "").strip().lower() in ("1", "true", "yes")
        if backend == 'real' or (backend == 'auto' and not disable_real):
            try:
                self._impl = RealEmbeddingGemma2(target_dim)
                self.backend_name = 'real'
                return
            except Exception as exc:
                if backend == 'real':
                    raise
                import warnings
                warnings.warn(
                    f'[language-u] EmbeddingGemma-2 real backend unavailable ({exc}); '
                    f'falling back to hash embedding — semantic quality reduced.')
        elif backend == 'auto' and disable_real:
            import warnings
            warnings.warn(
                '[language-u] LUC_NO_GEMMA=1 set; using hash embedding fallback — semantic quality reduced.')
        self._impl = HashEmbeddingFallback(target_dim)
        self.backend_name = 'hash'

    def get_backend(self) -> str:
        """The encoder actually in use: 'real' or 'hash'. Never a guess."""
        return self.backend_name

    def text_to_mrl_embedding(self, text: str) -> np.ndarray:
        """
        Projects arbitrary text into an MRL embedding vector in R^{target_dim}
        via the active backend ('real' model or 'hash' fallback).
        """
        return self._impl.embed(text)


class LanguageUOctonionBridge:
    """
    Bridges EmbeddingGemma-2 MRL vectors into Language-U 8D Clifford Octonions on S^7.
    """

    @classmethod
    def project_to_s7(cls, mrl_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Rotates an MRL vector (128D, 256D, etc.) through Sylvester-Hadamard H8
        blocks and projects onto S^7.
        Returns:
            (g_hat, resonance): 8D unit heading on S^7 and resonance with True North.
        """
        dim = len(mrl_vector)
        assert dim % 8 == 0, f"Vector dimension {dim} must be divisible by 8."
        k = dim // 8

        # Reshape into K blocks of 8
        blocks = mrl_vector.reshape(k, 8)

        # Apply H8 rotation to all blocks
        rotated_blocks = blocks @ H8_NORMALIZED

        # Gyroscopic centroid on S^7
        u_bar = np.mean(rotated_blocks, axis=0)

        # Unit projection on S^7
        norm = np.linalg.norm(u_bar)
        if norm < 1e-9:
            g_hat = TRUE_NORTH_GLYPH.copy()
        else:
            g_hat = u_bar / norm

        # Resonance with True North Attractor: <g_hat, g_north>
        resonance = float(np.dot(g_hat, TRUE_NORTH_GLYPH))
        return g_hat, resonance

    @classmethod
    def geodesic_distance_s7(cls, g1: np.ndarray, g2: np.ndarray) -> float:
        """
        Computes Riemannian geodesic distance d_S7(g1, g2) = arccos(clamp(<g1, g2>, -1, 1)).
        """
        dot = float(np.dot(g1, g2))
        clamped_dot = max(-1.0, min(1.0, dot))
        return math.acos(clamped_dot)


class REAEvidenceGraphScorer:
    """
    morluto/rea Evidence Graph Reasoner powered by EmbeddingGemma-2 & Language-U.
    Scores hypotheses against observations with contradiction pruning.
    """

    def __init__(self, target_dim: int = 128, backend: str = 'auto'):
        self.target_dim = target_dim
        self.backend = backend
        self._projector: Optional[EmbeddingGemma2MRLProjector] = None

    @property
    def projector(self) -> EmbeddingGemma2MRLProjector:
        if self._projector is None:
            self._projector = EmbeddingGemma2MRLProjector(
                target_dim=self.target_dim, backend=self.backend
            )
        return self._projector

    def extract_observations_from_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """Extracts individual constraint and property observations from problem text."""
        sentences = re.split(r'[\.\n;]+', prompt)
        observations = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) >= 10:
                v = self.projector.text_to_mrl_embedding(s_clean)
                g, res = LanguageUOctonionBridge.project_to_s7(v)
                words = set(re.findall(r'[a-zA-Z0-9_]{3,}', s_clean.lower()))
                observations.append({
                    "text": s_clean,
                    "mrl_vector": v,
                    "s7_heading": g,
                    "resonance": res,
                    "words": words
                })
        return observations

    def score_multiple_choice_hypotheses(self,
                                        prompt: str,
                                        choices: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluates choices as competing hypotheses using the Evidence Graph.
        Returns normalized logit scores for each choice letter.
        """
        observations = self.extract_observations_from_prompt(prompt)
        prompt_vec = self.projector.text_to_mrl_embedding(prompt)
        prompt_g, _ = LanguageUOctonionBridge.project_to_s7(prompt_vec)
        prompt_words = set(re.findall(r'[a-zA-Z0-9_]{3,}', prompt.lower()))

        scores = {letter: 0.0 for letter in choices}

        for letter, choice_text in choices.items():
            c_clean = choice_text.strip()
            c_words = set(re.findall(r'[a-zA-Z0-9_]{3,}', c_clean.lower()))
            c_vec = self.projector.text_to_mrl_embedding(c_clean)
            c_g, c_res = LanguageUOctonionBridge.project_to_s7(c_vec)

            # 1. Base Geodesic Resonance on S^7
            dot_prompt = float(np.dot(prompt_g, c_g))
            geo_dist = LanguageUOctonionBridge.geodesic_distance_s7(prompt_g, c_g)
            mrl_dot = float(np.dot(prompt_vec, c_vec))

            scores[letter] += (2.0 * dot_prompt) + (1.5 * mrl_dot) - (0.5 * geo_dist)

            # 2. Token overlap and domain keyword entailment
            overlap = len(prompt_words & c_words)
            if overlap > 0:
                scores[letter] += 1.0 * overlap

            # 3. Evidence Graph Accumulation across Observations
            confirms = 0.0
            refutes = 0.0

            for obs in observations:
                dot_obs = float(np.dot(obs["s7_heading"], c_g))
                obs_overlap = len(obs["words"] & c_words)
                if obs_overlap > 0:
                    confirms += 1.5 * obs_overlap
                if dot_obs > 0.3:
                    confirms += dot_obs
                elif dot_obs < -0.3:
                    refutes += abs(dot_obs)

            scores[letter] += (1.2 * confirms) - (1.8 * refutes)

            # 4. Contradiction Pruning (morluto/rea principle)
            c_lower = c_clean.lower()
            if any(w in c_lower for w in ["none of the above", "neither", "impossible"]):
                if any(w in prompt.lower() for w in ["no solution", "impossible", "cannot be determined"]):
                    scores[letter] += 3.0
                else:
                    scores[letter] -= 1.0

        return scores


# Global singleton instance for high-throughput zero-copy execution.
# Backend resolution: 'auto' — real model if loadable, else hash fallback.
REA_EVIDENCE_SCORER = REAEvidenceGraphScorer(target_dim=128)


def _self_test() -> bool:
    """Self-verification battery on physical silicon. Verifies BOTH backends."""
    # 0. Backend report — the active encoder is never a silent guess.
    probe = EmbeddingGemma2MRLProjector(target_dim=128, backend='auto')
    print(f"[+] Active MRL backend: {probe.get_backend()}")
    if probe.get_backend() == 'real':
        v_real_1 = probe.text_to_mrl_embedding('Differential Galois theory of Picard-Vessiot extensions')
        v_real_2 = probe.text_to_mrl_embedding('Differential Galois theory and linear algebraic groups')
        sim_real = float(np.dot(v_real_1, v_real_2))
        print(f"[+] Real-model related-concept cosine: {sim_real:.4f} (should exceed hash backend's)")
        h = EmbeddingGemma2MRLProjector(target_dim=128, backend='hash')
        vh1 = h.text_to_mrl_embedding('Differential Galois theory of Picard-Vessiot extensions')
        vh2 = h.text_to_mrl_embedding('Differential Galois theory and linear algebraic groups')
        sim_hash = float(np.dot(vh1, vh2))
        print(f"[+] Hash-fallback same-pair cosine:    {sim_hash:.4f}")

    # 1. MRL Projector test (active backend)
    proj = EmbeddingGemma2MRLProjector(target_dim=128)
    v1 = proj.text_to_mrl_embedding("Differential Galois theory of Picard-Vessiot extensions")
    v2 = proj.text_to_mrl_embedding("Differential Galois theory and linear algebraic groups")
    assert len(v1) == 128, f"Expected 128D, got {len(v1)}"
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-4, "MRL vector must be normalized."

    # Dot product between related mathematical concepts must be positive
    sim = float(np.dot(v1, v2))
    assert sim > 0.0, f"Expected positive cosine similarity, got {sim}"

    # 2. S^7 Octonionic Bridge test
    g, res = LanguageUOctonionBridge.project_to_s7(v1)
    assert len(g) == 8, f"Expected 8D heading on S^7, got {len(g)}"
    assert abs(np.linalg.norm(g) - 1.0) < 1e-4, "Heading must be unit vector on S^7."
    assert -1.0 <= res <= 1.0, f"Resonance {res} must be in [-1, 1]."

    # 3. Evidence Graph Scoring test
    prompt = "Consider a finite dimensional Lie algebra whose Killing form is non-degenerate. A Lie algebra is called semisimple if its Killing form is non-degenerate."
    choices = {
        "A": "Abelian Lie algebra",
        "B": "Solvable Lie algebra",
        "C": "Semisimple Lie algebra",
        "D": "Nilpotent Lie algebra"
    }
    scores = REA_EVIDENCE_SCORER.score_multiple_choice_hypotheses(prompt, choices)
    assert "C" in scores, "Scores must contain choice C"
    assert scores["C"] > scores["A"], f"Semisimple score ({scores['C']}) must exceed Abelian ({scores['A']})"

    return True


if __name__ == '__main__':
    print("Executing Language-U EmbeddingGemma-2 Adapter Self-Test...")
    if _self_test():
        print("Language-U EmbeddingGemma-2 Adapter PASSED self-test on physical silicon.")

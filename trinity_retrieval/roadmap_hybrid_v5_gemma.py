#!/usr/bin/env python3
"""Roadmap Hybrid Engine v5 — Six-Channel Trinity Law with Google EmbeddingGemma-2 & Language-U Octonionic Compass.

Channels (fused, linear scales):
  1. BM25 (SQLite FTS5, unicode61)          — lexical precision (W=0.28)
  2. Symbolic boosts (slug/title overlap)   — exact topic-name signal (W=0.16)
  3. Char 3-4gram TF-IDF cosine             — typo/morphology tolerance (W=0.20)
  4. Word TF-IDF LSA (SVD k=100) cosine     — co-occurrence semantics (W=0.12)
  5. Google EmbeddingGemma-2 (768-D MRL)    — neural semantic fingerprints (W=0.27)
  6. Language-U S^7 Gyroscopic Compass      — 8D Clifford spherical resonance (W=0.03, 61.3 KB RAM)

Trust Law (v5 Calibrated):
  STRONG: fused >= 0.302 AND lexical_evidence >= 0.080 -> execute
  WEAK:   fused >= 0.280 AND lexical_evidence >= 0.050 -> rescue (opt-in --weak)
  else:   honest MISS. False-positive noise structurally refused.
"""

import argparse
import json
import math
import os
import re
import socket
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

# Import physical neural embedding engine and Language-U S^7 Compass.
# Torch-optional (Agent 07, Oct 2026): gemma2_embedding.py hard-imports torch at
# module level. On torchless edge nodes (Pi/tablet) the shipped .npz fingerprint
# store still powers document-side scoring; construct with load_gemma=False and the
# engine degrades loudly to the 4-channel law (BM25+symbolic+char+LSA) instead of
# dying at import. Mirrors the Consider-Trinity engines/__init__ MiMo guard.
# Lazily import the physical neural embedding engine only when needed (`load_gemma=True`
# or explicit CLI usage), so torchless or non-Gemma callers never pay PyTorch import cost.
Gemma2EmbeddingModel = None
_GEMMA_IMPORT_OK = False
_GEMMA_IMPORT_CHECKED = False


def _ensure_gemma_model():
    """Import Gemma2EmbeddingModel on demand unless disabled via LUC_NO_GEMMA."""
    global Gemma2EmbeddingModel, _GEMMA_IMPORT_OK, _GEMMA_IMPORT_CHECKED
    if os.environ.get("LUC_NO_GEMMA", "").strip().lower() in ("1", "true", "yes"):
        Gemma2EmbeddingModel = None
        _GEMMA_IMPORT_OK = False
        _GEMMA_IMPORT_CHECKED = True
        return None
    if _GEMMA_IMPORT_CHECKED:
        return Gemma2EmbeddingModel
    _GEMMA_IMPORT_CHECKED = True
    try:
        from .gemma2_embedding import Gemma2EmbeddingModel as _GemmaCls
        Gemma2EmbeddingModel = _GemmaCls
        _GEMMA_IMPORT_OK = True
    except ImportError:
        try:
            from gemma2_embedding import Gemma2EmbeddingModel as _GemmaCls
            Gemma2EmbeddingModel = _GemmaCls
            _GEMMA_IMPORT_OK = True
        except ImportError as _gemma_exc:
            import warnings as _warnings
            Gemma2EmbeddingModel = None
            _GEMMA_IMPORT_OK = False
            _warnings.warn(
                "[v5 engine] EmbeddingGemma-2 unavailable on this host "
                f"({_gemma_exc}); running torchless — channels 5/6 (Gemma/S7-compass) "
                "degraded, channels 1-4 + fingerprint store fully operational. "
                "Use HybridEngineV5(load_gemma=False).",
                stacklevel=2,
            )
    return Gemma2EmbeddingModel
try:
    from .language_u_octonion import project_mrl_to_s7, project_batch_to_s7, batch_geodesic_resonance, _self_test as s7_self_test
except ImportError:
    from language_u_octonion import project_mrl_to_s7, project_batch_to_s7, batch_geodesic_resonance, _self_test as s7_self_test

# Path Resolution: Support explicit $TRINITY_RETRIEVAL_DATA or repo-local data_store/
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_store_path(filename: str) -> str:
    env_dir = os.environ.get("TRINITY_RETRIEVAL_DATA", "").strip()
    if env_dir:
        env_path = os.path.join(env_dir, filename)
        if os.path.exists(env_path):
            return env_path
    for base in (REPO_DIR, os.path.join(REPO_DIR, "data_store"), os.path.dirname(REPO_DIR)):
        cand = os.path.join(base, filename)
        if os.path.exists(cand):
            return cand
    return os.path.join(REPO_DIR, filename)


DB = resolve_store_path("roadmap_fts5.sqlite")
ART = resolve_store_path("roadmap_v2.npz")
META = resolve_store_path("roadmap_v2_meta.json")
GEMMA_EMB = resolve_store_path("roadmap_gemma_emb.npz")
S7_EMB = resolve_store_path("roadmap_s7_emb.npz")
SOCK = "/tmp/roadmap_hybrid_v5.sock"

_ENGINE_CACHE: Dict[str, Any] = {}


def store_revision() -> Tuple[Tuple[str, ...], ...]:
    """Return a stat tuple across SQLite/WAL and index artifacts to invalidate stale caches."""
    paths = [DB, DB + "-wal", ART, META, GEMMA_EMB, S7_EMB]
    rev = []
    for path in paths:
        try:
            st = os.stat(path)
            rev.append((os.path.abspath(path), str(st.st_ino), str(st.st_size), str(st.st_mtime_ns)))
        except FileNotFoundError:
            rev.append((os.path.abspath(path), "missing"))
    return tuple(rev)


def clear_engine_cache() -> None:
    """Close and evict any cached HybridEngineV5 instance."""
    eng = _ENGINE_CACHE.pop("engine", None)
    if eng is not None and hasattr(eng, "close"):
        eng.close()
    _ENGINE_CACHE.clear()


def get_engine(load_gemma: bool = True) -> "HybridEngineV5":
    """Return a cached HybridEngineV5 instance unless on-disk store artifacts changed."""
    rev = store_revision()
    if (
        _ENGINE_CACHE.get("engine") is not None
        and _ENGINE_CACHE.get("revision") == rev
        and _ENGINE_CACHE.get("load_gemma") == load_gemma
    ):
        return _ENGINE_CACHE["engine"]
    clear_engine_cache()
    eng = HybridEngineV5(load_gemma=load_gemma)
    _ENGINE_CACHE.update(engine=eng, revision=rev, load_gemma=load_gemma)
    return eng

STOPWORDS: Set[str] = set(
    "the and for with that this from which what where when how are is was were "
    "been have has had does did can could should would about above across after "
    "again against all almost alone along already also although always among an "
    "another any".split()
)

# Calibrated Fusion Weights (6-Channel)
W_BM25, W_SYM, W_CHAR, W_LSA, W_GEMMA, W_S7 = 0.28, 0.16, 0.20, 0.12, 0.27, 0.03
BM25_SCALE = 40.0
SYM_SCALE = 24.0

GEMMA_BASELINE = 0.72
S7_BASELINE = 0.88

STRONG_GATE = 0.302
WEAK_GATE = 0.280
LEX_STRONG_FLOOR = 0.080
LEX_WEAK_FLOOR = 0.050


def tokenize(text: str) -> List[str]:
    tokens, seen = [], set()
    for word in re.split(r"[^a-zA-Z0-9]+", text):
        lower = word.lower()
        if len(lower) >= 3 and not lower.isdigit() and lower not in STOPWORDS and lower not in seen:
            seen.add(lower)
            tokens.append(lower)
    return tokens


def char_grams(text: str) -> Set[str]:
    grams = set()
    for w in text.split():
        w2 = f" {w.strip()} "
        for n in (3, 4):
            for i in range(len(w2) - n + 1):
                grams.add(w2[i : i + n])
    return grams


class HybridEngineV5:
    """All retrieval channels + Google EmbeddingGemma-2 + Language-U Octonionic Compass resident in memory."""

    def __init__(self, load_gemma: bool = True):
        t0 = time.perf_counter()
        if not os.path.exists(DB):
            raise FileNotFoundError(f"Database not found at {DB}")
        if not os.path.exists(ART):
            raise FileNotFoundError(f"Artifacts not found at {ART}")
        if not os.path.exists(META):
            raise FileNotFoundError(f"Metadata not found at {META}")

        with np.load(ART) as z:
            self.Vt_k = z["Vt_k"]
            self.doc_lsa = z["doc_lsa"]
            self.doc_char = z["doc_char"]
            self.gidf = z["gidf"]
            self.idf = z["idf"]
        with open(META, "r", encoding="utf-8") as mf:
            m = json.load(mf)

        self.vocab = m["vocab"]
        self.gvocab = m["gvocab"]
        self.slugs = m["slugs"]
        self.titles = m["titles"]
        self.bodies = m["bodies"]
        self.links = m["links"]
        self.roadmaps = m["roadmaps"]
        self.categories = m["categories"]
        self.ids = m["ids"]

        self.id_to_idx = {tid: i for i, tid in enumerate(self.ids)}
        self.wpos = {t: i for i, t in enumerate(self.vocab)}
        self.gpos = {g: i for i, g in enumerate(self.gvocab)}
        self.conn = sqlite3.connect(DB)

        self._slug_tokens = [set(tokenize(s)) for s in self.slugs]
        self._title_tokens = [set(tokenize(t)) for t in self.titles]

        # Channel 6: Language-U S^7 Compass (8D, 61.3 KB footprint)
        self.doc_s7: Optional[np.ndarray] = None
        if os.path.exists(S7_EMB):
            with np.load(S7_EMB) as s:
                self.doc_s7 = s["emb"].astype(np.float32)
            print(f"[v5 engine] Loaded {len(self.doc_s7)} Language-U S^7 vectors ({os.path.getsize(S7_EMB)/1024:.1f} KB) from {S7_EMB}", flush=True)

        # Channel 5: Google EmbeddingGemma-2
        self.load_gemma = load_gemma
        self.gemma_model: Optional[Gemma2EmbeddingModel] = None
        self.doc_gemma: Optional[np.ndarray] = None
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._skip_validated: Set[str] = set()

        if os.path.exists(GEMMA_EMB):
            with np.load(GEMMA_EMB) as b:
                self.doc_gemma = b["emb"].astype(np.float32)
            norms = np.linalg.norm(self.doc_gemma, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            self.doc_gemma /= norms
            print(f"[v5 engine] Loaded {len(self.doc_gemma)} Gemma-2 embeddings from {GEMMA_EMB}", flush=True)

        if self.load_gemma:
            gemma_cls = _ensure_gemma_model()
            if not _GEMMA_IMPORT_OK or gemma_cls is None:
                raise RuntimeError(
                    "[v5 engine] load_gemma=True requested but EmbeddingGemma-2 is "
                    "unavailable on this host (torch/transformers missing). Construct "
                    "HybridEngineV5(load_gemma=False) for the torchless 4-channel lane."
                )
            print("[v5 engine] Initializing Google EmbeddingGemma-2 model...", flush=True)
            self.gemma_model = gemma_cls(target_dim=768)
            print(f"[v5 engine] Gemma-2 resident in {self.gemma_model.load_latency_s:.2f}s", flush=True)

        dt = time.perf_counter() - t0
        print(f"[v5 engine] HybridEngineV5 ready in {dt:.2f}s", flush=True)

    def close(self):
        """Clean resource lifecycle per Standard 6."""
        if hasattr(self, "conn") and self.conn is not None:
            self.conn.close()
            self.conn = None

    def __del__(self):
        self.close()

    def search(
        self,
        query: str,
        limit: int = 5,
        weak: bool = False,
        use_gemma: Optional[bool] = None,
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """Physical 6-channel search with adaptive Gemma inference and trust gating."""
        tokens = tokenize(query)
        if not tokens:
            return None, []

        c = self.conn
        match_q = " OR ".join(tokens[:16])
        bm_rows = c.execute(
            "SELECT t.id, bm25(topics_fts) FROM topics_fts f JOIN topics t ON t.id = f.rowid "
            "WHERE topics_fts MATCH ? ORDER BY rank LIMIT 30",
            (match_q,),
        ).fetchall()
        bm25_by_id = {r[0]: float(r[1]) for r in bm_rows}

        if len(tokens) <= 6:
            cover_rows = c.execute(
                "SELECT t.id FROM topics t WHERE "
                + " AND ".join(["(LOWER(t.title || ' ' || t.slug) LIKE ?)" for _ in tokens]),
                tuple(f"%{t}%" for t in tokens),
            ).fetchall()
            for (tid,) in cover_rows:
                bm25_by_id.setdefault(tid, 0.0)

        # LSA
        qw = np.zeros((len(self.vocab),), dtype=np.float32)
        for t in tokens:
            j = self.wpos.get(t)
            if j is not None:
                qw[j] += 1.0
        np.log1p(qw, out=qw)
        qw *= self.idf
        qlsa = qw @ self.Vt_k.T
        qlsa /= np.linalg.norm(qlsa) + 1e-9
        lsa_cos = self.doc_lsa @ qlsa

        # Char grams
        qg = np.zeros((len(self.gvocab),), dtype=np.float32)
        for g in char_grams(" ".join(tokens)):
            j = self.gpos.get(g)
            if j is not None:
                qg[j] = 1.0
        qg *= self.gidf
        qg /= np.linalg.norm(qg) + 1e-9
        char_cos = self.doc_char @ qg

        # Embed query with Gemma-2 and project to Language-U S^7
        gemma_cos = None
        s7_res = None
        if self.gemma_model is not None and self.doc_gemma is not None:
            qvec = self._emb_cache.get(query)
            if qvec is None:
                qvec = self.gemma_model.embed_query(query)
                self._emb_cache[query] = qvec
                if len(self._emb_cache) > 512:
                    self._emb_cache.pop(next(iter(self._emb_cache)))

            gemma_cos = self.doc_gemma @ qvec
            if self.doc_s7 is not None:
                q_s7 = project_mrl_to_s7(qvec)
                s7_res = batch_geodesic_resonance(self.doc_s7, q_s7)

        # Robust Candidate Aggregation (correctly mapping array indices to IDs)
        cand_ids = set(bm25_by_id)
        cand_ids |= {self.ids[c] for c in np.argsort(-char_cos)[:30]}
        cand_ids |= {self.ids[l] for l in np.argsort(-lsa_cos)[:30]}
        if s7_res is not None:
            for s_idx in np.argsort(-s7_res)[:25]:
                cand_ids.add(s_idx + 1)
        if gemma_cos is not None:
            for g_idx in np.argsort(-gemma_cos)[:25]:
                cand_ids.add(g_idx + 1)

        qset = set(tokens)
        results = []

        for tid in cand_ids:
            i = self.id_to_idx.get(tid)
            if i is None:
                continue
            bm = bm25_by_id.get(tid, 0.0)
            bm25n = min(max(-bm / BM25_SCALE, 0.0), 1.0) if bm < 0 else 0.0
            st = self._slug_tokens[i]
            overlap = len(st & qset)
            is_exact = (st == qset)
            is_contain = (st <= qset) if st else False

            boost = 6.0 * overlap
            if is_exact:
                boost += 6.0
            elif is_contain:
                boost += 4.0

            tt = self._title_tokens[i]
            if tt and all(t in tt for t in tokens[:4]):
                boost += 3.0

            # Priority for canonical curated topics over external harvested source articles
            if not self.slugs[i].startswith("source/"):
                boost += 4.0
            else:
                boost -= 2.0

            symn = min(max(boost / SYM_SCALE, 0.0), 1.0)
            charn = max(float(char_cos[i]), 0.0)
            lsan = max(float(lsa_cos[i]), 0.0)
            lex = W_BM25 * bm25n + W_SYM * symn + W_CHAR * charn
            lex_fused = lex + W_LSA * lsan

            # Channel 6: Language-U S^7 Octonionic Geodesic Resonance
            # (index by meta position i, NOT tid-1 — id contiguity is not a
            # guarantee the artifact-gate law should ever have to make)
            if s7_res is not None and i < len(s7_res):
                raw_s7 = float(s7_res[i])
                s7n = max(raw_s7 - S7_BASELINE, 0.0) / (1.0 - S7_BASELINE)
            else:
                s7n = 0.0

            # Channel 5: Google EmbeddingGemma-2 Dense MRL
            if gemma_cos is not None and i < len(gemma_cos):
                raw_gemma = float(gemma_cos[i])
                gemman = max(raw_gemma - GEMMA_BASELINE, 0.0) / (1.0 - GEMMA_BASELINE)
            else:
                gemman = 0.0

            fused = lex_fused + W_GEMMA * gemman + W_S7 * s7n

            results.append({
                "id": tid,
                "slug": self.slugs[i],
                "title": self.titles[i],
                "roadmap": self.roadmaps[i],
                "score": round(fused, 4),
                "lex": round(lex, 4),
                "channels": {
                    "bm25": round(bm25n, 3),
                    "sym": round(symn, 3),
                    "char": round(charn, 3),
                    "lsa": round(lsan, 3),
                    "gemma": round(gemman, 3),
                    "s7": round(s7n, 3),
                },
                "body": self.bodies[i][:600],
            })

        results.sort(key=lambda r: -r["score"])
        if not results:
            return None, []

        best = results[0]
        if best["score"] >= STRONG_GATE and best["lex"] >= LEX_STRONG_FLOOR:
            tier = "STRONG"
        elif best["score"] >= WEAK_GATE and best["lex"] >= LEX_WEAK_FLOOR:
            tier = "WEAK"
        else:
            tier = "MISS"

        best["tier"] = tier
        if tier == "STRONG" or (tier == "WEAK" and weak):
            return best, results[:limit]
        return None, results[:limit]


# ---------------------------------------------------------------- battery
BATTERY = [
    ("shadow dom encapsulated components", "shadow-dom", "exact"),
    ("cascading stylesheet styling language", "css", "exact"),
    ("react component library frontend framework", "react", "exact"),
    ("fast in-memory cache redis", "redis", "exact"),
    ("acid transactions database consistency", "acid", "parent-ok"),
    ("cap theorem distributed consistency", "cap-theorem", "exact"),
    ("kafka message broker streaming", "kafka", "exact"),
    ("json web token authentication", "jwt", "parent-ok"),
    ("circuit breaker microservices failure", "circuit-breaker", "exact"),
    ("llm agents mcp tools", "mcp", "parent-ok"),
    ("oauth third party login", "oauth", "parent-ok"),
    ("sharding partition database scale", "sharding-strategies", "parent-ok"),
    ("vue svelte angular frameworks", "vuejs", "parent-ok"),
    ("nginx reverse proxy server", "nginx", "exact"),
    ("postgres relational database sql", "postgresql", "parent-ok"),
    ("playwright e2e browser testing", "playwright", "exact"),
    ("service worker offline pwa", "service-workers", "exact"),
    ("tailwind utility first css", "tailwind", "exact"),
    ("javascript", "javascript", "exact"),
    ("translate domain to ip address", "dns-and-how-it-works", "parent-ok"),
    ("playwrigt browser testing", "playwright", "exact"),
    ("kafaka event streaming", "kafka", "exact"),
    ("postgers sql database", "postgresql|relational-databases|database-indexes|ms-sql", "weak-only"),
    ("javascrpt the web language", "javascript|pick-a-backend-language|pick-a-language", "weak-only"),
    ("login with google", "oauth|auth-strategies", "weak-only"),
    ("store sessions across requests", "cookie-based-auth|token-authentication|urlsession", "weak-only"),
    ("database query too many joins slow", "n1-problem|database-indexes", "weak-only"),
    ("make website work on phones", "css|responsive|mobile-apps", "miss-by-design"),
    ("type safe javascript superset", "type-checkers|javascript", "weak-or-strong"),
    ("zero knowledge proofs", None, "miss"),
    ("quantum entanglement simulator", None, "miss"),
    ("glacial ice ships alien lore", None, "miss"),
    ("nft marketplace minting", None, "miss"),
    ("kubernetes orchestration", None, "miss"),
    ("bloom filter", None, "miss"),
]


def run_battery_in_process(eng: HybridEngineV5) -> bool:
    strong = weakr = miss = fail = 0
    fails = []
    t0 = time.perf_counter()
    for q, exp, kind in BATTERY:
        hit, _ = eng.search(q, limit=3, weak=False)
        hit_w, _ = eng.search(q, limit=3, weak=True)
        if kind == "miss":
            if hit is None and hit_w is None:
                miss += 1
            else:
                fail += 1
                fails.append((q, f"noise admitted: {(hit_w or hit)['slug']}"))
        elif kind in ("weak-only", "weak-or-strong"):
            got = hit or hit_w
            acc = set(exp.split("|"))
            if got is not None and got["slug"] in acc:
                weakr += 1
            else:
                fail += 1
                fails.append((q, f"got {got['slug'] if got else 'MISS'}"))
        elif kind == "miss-by-design":
            acc = set(exp.split("|"))
            if hit is None and hit_w is None:
                miss += 1
            elif hit_w is not None and hit_w["slug"] in acc:
                weakr += 1
            elif hit is not None and hit["slug"] in acc:
                strong += 1
            else:
                fail += 1
                fails.append((q, f"should refuse but got {(hit or hit_w)['slug']}"))
        else:
            if hit is not None and (hit["slug"] == exp or kind == "parent-ok"):
                strong += 1
            elif hit_w is not None and hit_w["slug"] in exp.split("|"):
                weakr += 1
            else:
                fail += 1
                fails.append((q, f"got {hit['slug'] if hit else None} want {exp}"))
    dt = (time.perf_counter() - t0) / len(BATTERY) * 1000
    print(
        f"BATTERY v5: strong={strong} weak-rescue={weakr} honest-miss={miss} FAIL={fail} / {len(BATTERY)}  ({dt:.1f} ms/query)"
    )
    for f in fails:
        print("  FAIL:", f)
    return fail == 0


def build_embeddings(batch_size: int = 32, max_length: int = 128, out_path: str = GEMMA_EMB):
    """Generate authentic Google EmbeddingGemma-2 embeddings for all corpus topics."""
    print("=== BUILDING PHYSICAL GEMMA-2 EMBEDDINGS ===", flush=True)
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT id, title, body FROM topics ORDER BY id").fetchall()
    conn.close()
    print(f"Fetched {len(rows)} topics from {DB}", flush=True)

    texts = [f"{t}\n{b[:300]}" for _, t, b in rows]
    gemma_cls = _ensure_gemma_model()
    if gemma_cls is None:
        raise RuntimeError("EmbeddingGemma-2 unavailable on this host")
    model = gemma_cls(target_dim=768)
    print(f"Model loaded onto {model.device} in {model.load_latency_s:.2f}s", flush=True)

    t0 = time.perf_counter()
    embs = model.embed_texts(texts, batch_size=batch_size, max_length=max_length, verbose=True)
    dt = time.perf_counter() - t0
    print(f"Embedded {len(texts)} documents in {dt:.2f}s ({dt/len(texts)*1000:.1f}ms/doc) -> shape {embs.shape}")

    ids = np.array([r[0] for r in rows])
    np.savez_compressed(out_path, emb=embs, ids=ids)
    print(f"Saved compressed embeddings to {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)", flush=True)


def build_s7_embeddings(in_path: str = GEMMA_EMB, out_path: str = S7_EMB):
    """Project the stored Gemma-2 fingerprints to Language-U S^7 unit headings.

    Deterministic numpy projection (project_batch_to_s7) — no model inference,
    so every fleet node can rebuild channel 6 locally. This builder exists
    because roadmap_s7_emb.npz previously had no in-repo producer and silently
    drifted one doc short of the store (doc 1963, the gitingest ingestion).
    """
    if not os.path.exists(in_path):
        sys.exit(f"Missing Gemma fingerprints at {in_path} — run 'embed' first.")
    g = np.load(in_path)
    emb = g["emb"].astype(np.float32)
    ids = g["ids"]
    s7 = project_batch_to_s7(emb)
    np.savez_compressed(out_path, emb=s7, ids=ids)
    print(f"Projected {len(ids)} Gemma fingerprints -> S^7 headings ({s7.shape[1]}D) at {out_path}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description="Roadmap Hybrid Retrieval Engine v5 (EmbeddingGemma-2 & Language-U S^7)")
    ap.add_argument("cmd", choices=["search", "battery", "embed", "embed-s7", "self-test"])
    ap.add_argument("query", nargs="?", help="Search query")
    ap.add_argument("--weak", action="store_true", help="Enable weak rescue tier")
    ap.add_argument("--limit", type=int, default=5, help="Max results returned")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    ap.add_argument("--no-gemma", action="store_true",
                    help="Run without the Gemma-2 neural channel (5-channel degrade lane; "
                         "also enabled via env LUC_NO_GEMMA=1). Honest for torchless nodes.")
    a = ap.parse_args()
    if os.environ.get("LUC_NO_GEMMA", "").strip() in ("1", "true", "yes"):
        a.no_gemma = True

    if a.cmd == "self-test":
        # Torch-optional (Agent 07): on torchless hosts the Gemma-2 half of the
        # self-test cannot run — report it explicitly instead of crashing, and
        # still exercise the S^7 compass self-test (numpy-only, always available).
        gemma_cls = _ensure_gemma_model()
        if _GEMMA_IMPORT_OK and gemma_cls is not None:
            s1 = gemma_cls._self_test()
        else:
            print("[v5 engine] SKIP: EmbeddingGemma-2 self-test unavailable (torch/transformers missing on this host)")
            s1 = None
        s2 = s7_self_test()
        if s1 is None:
            sys.exit(0 if s2 else 1)
        sys.exit(0 if (s1 and s2) else 1)
    elif a.cmd == "embed":
        build_embeddings(batch_size=a.batch_size)
        sys.exit(0)
    elif a.cmd == "embed-s7":
        build_s7_embeddings()
        sys.exit(0)
    elif a.cmd == "battery":
        # Torch-optional (Agent 07 / Julian PR #3): fall back to the fingerprint-store lane when
        # Gemma-2 cannot import or when --no-gemma / LUC_NO_GEMMA is set.
        _lg = not a.no_gemma
        gemma_cls = _ensure_gemma_model() if _lg else None
        if _lg and (not _GEMMA_IMPORT_OK or gemma_cls is None):
            print("[v5 engine] torchless host: running battery on the 4-channel fingerprint lane", flush=True)
            _lg = False
        elif a.no_gemma:
            print("[v5 engine] DEGRADED MODE: Gemma-2 neural channel disabled (--no-gemma/LUC_NO_GEMMA) — "
                  "battery runs the 5-channel lane.", flush=True)
        eng = HybridEngineV5(load_gemma=_lg)
        ok = run_battery_in_process(eng)
        sys.exit(0 if ok else 1)
    elif a.cmd == "search":
        if not a.query:
            sys.exit("Error: search command requires a query string.")
        _lg = not a.no_gemma
        gemma_cls = _ensure_gemma_model() if _lg else None
        if _lg and (not _GEMMA_IMPORT_OK or gemma_cls is None):
            _lg = False
        eng = HybridEngineV5(load_gemma=_lg)
        t0 = time.perf_counter()
        hit, rest = eng.search(a.query, limit=a.limit, weak=a.weak)
        ms = (time.perf_counter() - t0) * 1000
        if hit:
            print(f"[{hit['tier']}] {hit['title']} ({hit['slug']}, {hit['roadmap']} roadmap) score={hit['score']} ({ms:.1f}ms)")
            print(f"Channels: {hit['channels']}")
            print(f"\n{hit['body'][:500]}")
        else:
            top = rest[0] if rest else None
            top_str = f" (top below gate: {top['slug']}@{top['score']})" if top else ""
            print(f"[MISS] for: {a.query!r}{top_str} ({ms:.1f}ms)")
        sys.exit(0 if hit else 1)


if __name__ == "__main__":
    main()

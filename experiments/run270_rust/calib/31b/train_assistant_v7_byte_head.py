#!/usr/bin/env python3
"""
Assistant-V7 Byte-Head Speculative Drafter (`BHDR` v1) Trainer, Trie Builder & Repacker.

Implements Google Gemma-4 BPE prefix trie construction over a canonical 261-symbol
alphabet (`0..=255` raw bytes, `256=<eot>`, `257='▁'`, `258=<eos>`, `259=<turn|>`,
`260=<0xNN> escape`) and exact single-pass Appendix F.1 `<eot>` knowledge distillation
from arXiv:2609.12303 into a compact ~1.6 MB INT8 GRU byte+`<eot>` micro-decoder.

Also strips `model.embed_tokens.weight` (~136 MB..268 MB) from existing assistant
`.t16c` checkpoints or packs 2-bit/4-bit/8-bit embed-free `assistant_v7_body.t16c`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pack_t16c_v3 import (  # noqa: E402
    ARCH_FLAGS,
    T16C_MAGIC,
    T16C_VERSION,
    SafetensorsReader,
    expected_tile16_bytes,
    tile16_blk_bytes,
)
from train_cortical_router import _quantize_rows_i8  # noqa: E402
import pack_assistant_t16c as pa  # noqa: E402

T16C_GAUSS_BIT = 0x80
T16C_ROWMAJOR_BIT = 0x40
_GAUSS2_LV = np.array([-1.510417, -0.452780, 0.452780, 1.510417], dtype=np.float32)
_GAUSS2_MID = np.array([-0.981599, 0.0, 0.981599], dtype=np.float32)

BHDR_MAGIC = b"BHDR"
BHDR_VERSION = 1
BHDR_HEADER_BYTES = 32

SYM_EOT = 256
SYM_SP = 257
SYM_EOS = 258
SYM_TURN_END = 259
SYM_BYTE_ESC = 260
BHDR_VOCAB_SYM = 261


def token_to_symbols(token_id: int, tok_str: str) -> List[int]:
    """Mirror `byte_head::token_to_symbols` cleanly across 261-symbol alphabet."""
    if not tok_str:
        return []
    if token_id == 1 and tok_str == "<eos>":
        return [SYM_EOS]
    if token_id == 106 and tok_str in ("<turn|>", "<end_of_turn>"):
        return [SYM_TURN_END]
    if (
        len(tok_str) == 6
        and tok_str.startswith("<0x")
        and tok_str.endswith(">")
        and all(c in "0123456789abcdefABCDEF" for c in tok_str[3:5])
    ):
        return [SYM_BYTE_ESC, int(tok_str[3:5], 16)]
    out: List[int] = []
    for ch in tok_str:
        if ch == "\u2581":
            out.append(SYM_SP)
        else:
            out.extend(ch.encode("utf-8"))
    return out


def load_id_to_token_from_json(tok_path: Path) -> List[str]:
    """Load exact `id_to_token` vector matching `tokenizer.rs::Tokenizer::load`."""
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json: {tok_path}")
    data = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab_map: Dict[str, int] = data["model"]["vocab"]
    max_id = max(vocab_map.values()) if vocab_map else 0
    added = data.get("added_tokens") or []
    for item in added:
        if "id" in item:
            max_id = max(max_id, int(item["id"]))
    id_to_tok = [""] * (max_id + 1)
    for k, tid in vocab_map.items():
        ti = int(tid)
        if not id_to_tok[ti]:
            id_to_tok[ti] = k
    for item in added:
        if "id" in item and "content" in item:
            id_to_tok[int(item["id"])] = str(item["content"])
    return id_to_tok


class BpePrefixTrie:
    """Exact prefix trie over Gemma BPE pieces for Appendix F.1 `<eot>` distillation."""

    def __init__(self, id_to_token: Sequence[str]) -> None:
        self.id_to_token = list(id_to_token)
        self.node_edges: List[Dict[int, int]] = [{}]
        self.node_term: List[int] = [-1]
        self.tok_syms: List[List[int]] = [[] for _ in range(len(self.id_to_token))]
        self.tok_nodes: List[List[int]] = [[] for _ in range(len(self.id_to_token))]
        self.covered_tokens = 0
        self.collision_count = 0
        self.collisions_sample: List[Tuple[int, int, str, str]] = []
        self.max_depth = 0
        self.total_edges = 0

        for tid, tstr in enumerate(self.id_to_token):
            syms = token_to_symbols(tid, tstr)
            if not syms:
                continue
            self.tok_syms[tid] = syms
            self.max_depth = max(self.max_depth, len(syms))
            cur = 0
            path = [0]
            for s in syms:
                nxt = self.node_edges[cur].get(s)
                if nxt is None:
                    nxt = len(self.node_edges)
                    self.node_edges[cur][s] = nxt
                    self.node_edges.append({})
                    self.node_term.append(-1)
                    self.total_edges += 1
                cur = nxt
                path.append(cur)
            self.tok_nodes[tid] = path
            if self.node_term[cur] == -1:
                self.node_term[cur] = tid
                self.covered_tokens += 1
            else:
                self.collision_count += 1
                if len(self.collisions_sample) < 10:
                    prev_id = self.node_term[cur]
                    self.collisions_sample.append((prev_id, tid, self.id_to_token[prev_id], tstr))

    @classmethod
    def from_tokenizer_json(cls, tok_path: Path) -> "BpePrefixTrie":
        return cls(load_id_to_token_from_json(tok_path))

    def valid_syms_at(self, node: int) -> List[int]:
        syms = sorted(self.node_edges[node].keys())
        if self.node_term[node] >= 0:
            return [SYM_EOT] + syms
        return syms

    def approx_rust_csr_bytes(self) -> int:
        # Mirrors Rust TrieNode (12B w/ alignment) + TrieEdge (8B)
        return len(self.node_edges) * 12 + self.total_edges * 8


def compute_eot_byte_targets(
    trie: BpePrefixTrie,
    support_ids: Sequence[int],
    support_probs: Sequence[float],
    gold_tok_id: int,
) -> List[Tuple[int, List[int], np.ndarray, int]]:
    """
    Exact arXiv:2609.12303 Appendix F.1 single-pass `<eot>` target conversion.

    For each step k=0..L along `gold_tok_id`'s trie path `[u_0, ..., u_L]`:
      - Returns `(node_id, valid_syms, q_over_valid, gold_sym)`
      - Guarantee: sum(q_over_valid) == 1.0, and prod_{k=0..L} q_k(gold_sym_k) == P_T(gold_tok_id).
    """
    gold_syms = trie.tok_syms[gold_tok_id]
    gold_nodes = trie.tok_nodes[gold_tok_id]
    if not gold_syms:
        raise ValueError(f"gold_tok_id={gold_tok_id} maps to empty symbol sequence")
    L = len(gold_syms)

    mass_at_step: List[Dict[int, float]] = [defaultdict(float) for _ in range(L + 1)]
    node_mass: List[float] = [0.0 for _ in range(L + 1)]

    for wid_raw, p_raw in zip(support_ids, support_probs):
        p = float(p_raw)
        if p <= 0.0:
            continue
        wsyms = trie.tok_syms[int(wid_raw)]
        if not wsyms:
            continue
        k = 0
        lim = min(len(wsyms), L)
        while k < lim and wsyms[k] == gold_syms[k]:
            mass_at_step[k][wsyms[k]] += p
            node_mass[k] += p
            k += 1
        if k == len(wsyms):
            mass_at_step[k][SYM_EOT] += p
            node_mass[k] += p
        else:
            mass_at_step[k][wsyms[k]] += p
            node_mass[k] += p

    out: List[Tuple[int, List[int], np.ndarray, int]] = []
    for k in range(L + 1):
        node = gold_nodes[k]
        valid = trie.valid_syms_at(node)
        z = node_mass[k]
        if z <= 0.0:
            raise ValueError(f"Zero subtree probability at step k={k} for gold_tok_id={gold_tok_id}")
        q = np.array([mass_at_step[k].get(s, 0.0) / z for s in valid], dtype=np.float32)
        gold_sym = SYM_EOT if k == L else gold_syms[k]
        out.append((node, valid, q, gold_sym))
    return out


class ByteHeadGRU(nn.Module):
    """Exact PyTorch equivalent of Rust `byte_head::ByteHeadDecoder`."""

    def __init__(self, d_in: int = 1024, d_byte: int = 256, max_bytes: int = 32) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_byte = int(d_byte)
        self.vocab_sym = BHDR_VOCAB_SYM
        self.max_bytes = int(max_bytes)

        self.w_init = nn.Linear(self.d_in, self.d_byte, bias=True)
        self.w_ctx = nn.Linear(self.d_in, 3 * self.d_byte, bias=False)
        self.e_sym = nn.Embedding(self.vocab_sym, self.d_byte)
        self.w_ih = nn.Linear(self.d_byte, 3 * self.d_byte, bias=True)
        self.w_hh = nn.Linear(self.d_byte, 3 * self.d_byte, bias=True)
        self.w_out = nn.Linear(self.d_byte, self.vocab_sym, bias=True)

    def init_state_and_ctx(self, h_final: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s0 = torch.tanh(self.w_init(h_final))
        ctx = self.w_ctx(h_final)
        return s0, ctx

    def gru_step(self, state: torch.Tensor, sym_ids: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x_emb = self.e_sym(sym_ids)
        ih = self.w_ih(x_emb) + ctx
        hh = self.w_hh(state)
        ih_r, ih_z, ih_n = ih.chunk(3, dim=-1)
        hh_r, hh_z, hh_n = hh.chunk(3, dim=-1)
        r = torch.sigmoid(ih_r + hh_r)
        z = torch.sigmoid(ih_z + hh_z)
        n = torch.tanh(ih_n + r * hh_n)
        return (1.0 - z) * n + z * state

    def forward_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.w_out(state)

    def forward_path(
        self, h_final: torch.Tensor, sym_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate logits `[B, T, 261]` along teacher-forced symbol sequence `sym_seq [B, T-1]`.
        Position 0 evaluates `s0`; position k>0 evaluates state after feeding `sym_seq[:, k-1]`.
        """
        state, ctx = self.init_state_and_ctx(h_final)
        logits_steps = [self.forward_logits(state)]
        for k in range(sym_seq.shape[1]):
            state = self.gru_step(state, sym_seq[:, k], ctx)
            logits_steps.append(self.forward_logits(state))
        return torch.stack(logits_steps, dim=1)


def _quantize_mat_tensor(t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    return _quantize_rows_i8(t.detach().cpu().numpy().astype(np.float32))


def quantize_byte_head_to_dict(model: ByteHeadGRU) -> dict:
    """Symmetrically quantize `ByteHeadGRU` matrices into `BHDR` v1 INT8 arrays."""
    wi_q, wi_s = _quantize_mat_tensor(model.w_init.weight)
    wc_q, wc_s = _quantize_mat_tensor(model.w_ctx.weight)
    es_q, es_s = _quantize_mat_tensor(model.e_sym.weight)
    wih_q, wih_s = _quantize_mat_tensor(model.w_ih.weight)
    whh_q, whh_s = _quantize_mat_tensor(model.w_hh.weight)
    wo_q, wo_s = _quantize_mat_tensor(model.w_out.weight)
    return {
        "d_in": model.d_in,
        "d_byte": model.d_byte,
        "vocab_sym": model.vocab_sym,
        "max_bytes": model.max_bytes,
        "flags": 0,
        "w_init": wi_q,
        "s_init": wi_s,
        "b_init": model.w_init.bias.detach().cpu().numpy().astype("<f4"),
        "w_ctx": wc_q,
        "s_ctx": wc_s,
        "e_sym": es_q,
        "s_sym": es_s,
        "w_ih": wih_q,
        "s_ih": wih_s,
        "b_ih": model.w_ih.bias.detach().cpu().numpy().astype("<f4"),
        "w_hh": whh_q,
        "s_hh": whh_s,
        "b_hh": model.w_hh.bias.detach().cpu().numpy().astype("<f4"),
        "w_out": wo_q,
        "s_out": wo_s,
        "b_out": model.w_out.bias.detach().cpu().numpy().astype("<f4"),
    }


def export_bhdr_v1(model: ByteHeadGRU, path: Path) -> int:
    """Serialize `ByteHeadGRU` cleanly into binary `BHDR` v1 container matching Rust."""
    rec = quantize_byte_head_to_dict(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(
            struct.pack(
                "<4s7I",
                BHDR_MAGIC,
                BHDR_VERSION,
                rec["d_in"],
                rec["d_byte"],
                rec["vocab_sym"],
                rec["max_bytes"],
                rec["flags"],
                0,
            )
        )
        for arr_key, sc_key, b_key in [
            ("w_init", "s_init", "b_init"),
            ("w_ctx", "s_ctx", None),
            ("e_sym", "s_sym", None),
            ("w_ih", "s_ih", "b_ih"),
            ("w_hh", "s_hh", "b_hh"),
            ("w_out", "s_out", "b_out"),
        ]:
            fp.write(rec[arr_key].astype(np.int8).tobytes(order="C"))
            fp.write(rec[sc_key].astype("<f4").tobytes(order="C"))
            if b_key is not None:
                fp.write(rec[b_key].astype("<f4").tobytes(order="C"))
    return path.stat().st_size


def load_bhdr_v1(path: Path) -> dict:
    """Fail-closed binary reader for `BHDR` v1 files."""
    data = path.read_bytes()
    if len(data) < BHDR_HEADER_BYTES:
        raise ValueError(f"BHDR file truncated: {len(data)} < {BHDR_HEADER_BYTES}")
    magic, ver, d_in, d_byte, vsym, max_b, flags, resv = struct.unpack("<4s7I", data[:BHDR_HEADER_BYTES])
    if magic != BHDR_MAGIC or ver != BHDR_VERSION or resv != 0 or flags != 0 or vsym != BHDR_VOCAB_SYM:
        raise ValueError(f"BHDR bad header: magic={magic!r} ver={ver} vsym={vsym} flags={flags}")
    pos = BHDR_HEADER_BYTES

    def rd_i8(rows: int, cols: int) -> np.ndarray:
        nonlocal pos
        n = rows * cols
        arr = np.frombuffer(data, dtype=np.int8, count=n, offset=pos).reshape(rows, cols).copy()
        pos += n
        return arr

    def rd_f32(n: int) -> np.ndarray:
        nonlocal pos
        arr = np.frombuffer(data, dtype="<f4", count=n, offset=pos).astype(np.float32).copy()
        pos += n * 4
        return arr

    out = {
        "d_in": d_in,
        "d_byte": d_byte,
        "vocab_sym": vsym,
        "max_bytes": max_b,
        "flags": flags,
        "w_init": rd_i8(d_byte, d_in),
        "s_init": rd_f32(d_byte),
        "b_init": rd_f32(d_byte),
        "w_ctx": rd_i8(3 * d_byte, d_in),
        "s_ctx": rd_f32(3 * d_byte),
        "e_sym": rd_i8(vsym, d_byte),
        "s_sym": rd_f32(vsym),
        "w_ih": rd_i8(3 * d_byte, d_byte),
        "s_ih": rd_f32(3 * d_byte),
        "b_ih": rd_f32(3 * d_byte),
        "w_hh": rd_i8(3 * d_byte, d_byte),
        "s_hh": rd_f32(3 * d_byte),
        "b_hh": rd_f32(3 * d_byte),
        "w_out": rd_i8(vsym, d_byte),
        "s_out": rd_f32(vsym),
        "b_out": rd_f32(vsym),
    }
    if pos != len(data):
        raise ValueError(f"BHDR trailing bytes: parsed {pos} != {len(data)}")
    return out


def eval_bhdr_np_init_and_logprobs(
    bh: dict, trie: BpePrefixTrie, h_final: np.ndarray, node: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray]:
    """Reference quantized NumPy inference matching `byte_head.rs` step-0."""
    wi = bh["w_init"].astype(np.float32) * bh["s_init"][:, None]
    wc = bh["w_ctx"].astype(np.float32) * bh["s_ctx"][:, None]
    wo = bh["w_out"].astype(np.float32) * bh["s_out"][:, None]
    s0 = np.tanh(wi @ h_final.astype(np.float32) + bh["b_init"])
    ctx = wc @ h_final.astype(np.float32)
    valid = trie.valid_syms_at(node)
    logits = np.array([float(np.dot(wo[s], s0) + bh["b_out"][s]) for s in valid], dtype=np.float32)
    m = float(np.max(logits))
    ex = np.exp((logits - m).astype(np.float64))
    lps = (logits.astype(np.float64) - m - math.log(float(np.sum(ex)))).astype(np.float32)
    return s0, ctx, valid, lps


def bhdr_np_gru_step(bh: dict, state: np.ndarray, sym: int, ctx: np.ndarray) -> np.ndarray:
    db = int(bh["d_byte"])
    es = bh["e_sym"][sym].astype(np.float32) * float(bh["s_sym"][sym])
    wih = bh["w_ih"].astype(np.float32) * bh["s_ih"][:, None]
    whh = bh["w_hh"].astype(np.float32) * bh["s_hh"][:, None]
    ih = wih @ es + bh["b_ih"] + ctx
    hh = whh @ state + bh["b_hh"]
    r = 1.0 / (1.0 + np.exp(-np.clip(ih[:db] + hh[:db], -60.0, 60.0)))
    z = 1.0 / (1.0 + np.exp(-np.clip(ih[db : 2 * db] + hh[db : 2 * db], -60.0, 60.0)))
    n = np.tanh(ih[2 * db :] + r * hh[2 * db :])
    return ((1.0 - z) * n + z * state).astype(np.float32)


def decode_bhdr_np_beam(
    bh: dict, trie: BpePrefixTrie, h_final: np.ndarray, beam_width: int = 2
) -> Tuple[int, float, float, int]:
    """Reference quantized NumPy beam decoder matching `byte_head.rs::decode_token_beam`."""
    wo = bh["w_out"].astype(np.float32) * bh["s_out"][:, None]
    s0, ctx, _, _ = eval_bhdr_np_init_and_logprobs(bh, trie, h_final, 0)
    beams = [(0.0, 0, 0, s0)]
    completed: List[Tuple[float, int, int]] = []
    pruned_alt = float("-inf")
    max_b = int(bh["max_bytes"])
    bw = max(beam_width, 1)

    for step_idx in range(max_b + 1):
        allow_expand = step_idx < max_b
        next_cands: List[Tuple[float, int, int, int, int]] = []
        for bi, (cl, node, depth, st) in enumerate(beams):
            valid = trie.valid_syms_at(node)
            if not valid:
                continue
            logits = np.array([float(np.dot(wo[s], st) + bh["b_out"][s]) for s in valid], dtype=np.float64)
            m = float(np.max(logits))
            ex = np.exp(logits - m)
            lps = logits - m - math.log(float(np.sum(ex)))
            for sym, lp in zip(valid, lps):
                tot = float(cl + lp)
                if sym == SYM_EOT:
                    tid = trie.node_term[node]
                    if tid >= 0:
                        completed.append((tot, tid, depth))
                elif allow_expand:
                    child = trie.node_edges[node].get(sym)
                    if child is not None:
                        next_cands.append((tot, child, depth + 1, bi, sym))
                else:
                    pruned_alt = max(pruned_alt, tot)
        if not allow_expand or not next_cands:
            break
        next_cands.sort(key=lambda x: x[0], reverse=True)
        if len(next_cands) > bw:
            pruned_alt = max(pruned_alt, next_cands[bw][0])
            next_cands = next_cands[:bw]
        completed.sort(key=lambda x: x[0], reverse=True)
        if len(completed) >= 2 and next_cands[0][0] <= completed[1][0]:
            break
        beams = [
            (tot, child, depth, bhdr_np_gru_step(bh, beams[bi][3], sym, ctx))
            for tot, child, depth, bi, sym in next_cands
        ]
    if not completed:
        raise RuntimeError("No valid token reached <eot>")
    completed.sort(key=lambda x: x[0], reverse=True)
    top1_lp, best_tok, depth = completed[0]
    top2_lp = max(completed[1][0], pruned_alt) if len(completed) >= 2 else (pruned_alt if math.isfinite(pruned_alt) else top1_lp - 30.0)
    return int(best_tok), float(top1_lp), float(top2_lp), int(depth)


def _build_training_step_tensors(
    trie: BpePrefixTrie,
    batch_support_ids: Sequence[Sequence[int]],
    batch_support_probs: Sequence[Sequence[float]],
    batch_gold_ids: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded `[B, max_steps, 261]` exact `<eot>` targets & active valid masks."""
    per_ex = [
        compute_eot_byte_targets(trie, sids, sps, int(gid))
        for sids, sps, gid in zip(batch_support_ids, batch_support_probs, batch_gold_ids)
    ]
    bsz = len(per_ex)
    max_t = max(len(item) for item in per_ex)
    sym_in = torch.zeros((bsz, max(max_t - 1, 1)), dtype=torch.long)
    valid_mask = torch.zeros((bsz, max_t, BHDR_VOCAB_SYM), dtype=torch.bool)
    tgt_probs = torch.zeros((bsz, max_t, BHDR_VOCAB_SYM), dtype=torch.float32)
    step_active = torch.zeros((bsz, max_t), dtype=torch.float32)

    for b, steps in enumerate(per_ex):
        for k, (_node, valid, q, gsym) in enumerate(steps):
            valid_mask[b, k, valid] = True
            tgt_probs[b, k, valid] = torch.from_numpy(q)
            step_active[b, k] = 1.0
            if k + 1 < len(steps):
                sym_in[b, k] = int(gsym)

    return (
        sym_in.to(device),
        valid_mask.to(device),
        tgt_probs.to(device),
        step_active.to(device),
    )


def train_byte_head_eot_distill(
    trie: BpePrefixTrie,
    h_train: torch.Tensor,
    supp_ids_train: Sequence[Sequence[int]],
    supp_probs_train: Sequence[Sequence[float]],
    gold_ids_train: Sequence[int],
    h_val: torch.Tensor,
    supp_ids_val: Sequence[Sequence[int]],
    supp_probs_val: Sequence[Sequence[float]],
    gold_ids_val: Sequence[int],
    *,
    d_byte: int = 256,
    max_bytes: int = 32,
    steps: int = 200,
    batch_size: int = 32,
    lr: float = 3e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[ByteHeadGRU, Dict[str, float]]:
    """Train `ByteHeadGRU` via exact single-pass `<eot>` KL divergence + real `loss.backward()`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    d_in = int(h_train.shape[1])
    model = ByteHeadGRU(d_in=d_in, d_byte=d_byte, max_bytes=max_bytes).to(dev)

    hv = h_val.float().to(dev)
    sym_v, mask_v, tgt_v, act_v = _build_training_step_tensors(
        trie, supp_ids_val, supp_probs_val, gold_ids_val, dev
    )

    def eval_val() -> Tuple[float, float]:
        with torch.no_grad():
            lg = model.forward_path(hv, sym_v[:, : mask_v.shape[1] - 1] if mask_v.shape[1] > 1 else sym_v[:, :0])
            lg_m = lg.masked_fill(~mask_v, -1e9)
            lp = F.log_softmax(lg_m, dim=-1)
            kl_step = (tgt_v * (torch.log(tgt_v.clamp_min(1e-12)) - lp)).sum(dim=-1)
            kl = float((kl_step * act_v).sum().item() / act_v.sum().clamp_min(1.0).item())
            sym_tgt = tgt_v.argmax(dim=-1)
            sym_pred = lg_m.argmax(dim=-1)
            tok_ok = (((sym_pred == sym_tgt) | (act_v == 0)).all(dim=-1)).float().mean().item()
            return kl, float(tok_ok)

    kl0, acc0 = eval_val()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.05)
    best_kl = kl0
    best_acc = acc0
    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    n_tr = int(h_train.shape[0])
    for step in range(steps):
        idx = np.random.choice(n_tr, size=min(batch_size, n_tr), replace=False)
        hb = h_train[idx].float().to(dev)
        sids = [supp_ids_train[i] for i in idx]
        sps = [supp_probs_train[i] for i in idx]
        gids = [gold_ids_train[i] for i in idx]
        sym_b, mask_b, tgt_b, act_b = _build_training_step_tensors(trie, sids, sps, gids, dev)

        opt.zero_grad()
        in_seq = sym_b[:, : mask_b.shape[1] - 1] if mask_b.shape[1] > 1 else sym_b[:, :0]
        lg = model.forward_path(hb, in_seq)
        lg_m = lg.masked_fill(~mask_b, -1e9)
        lp = F.log_softmax(lg_m, dim=-1)
        kl_step = (tgt_b * (torch.log(tgt_b.clamp_min(1e-12)) - lp)).sum(dim=-1)
        loss = (kl_step * act_b).sum() / act_b.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if (step + 1) % max(steps // 5, 1) == 0 or (step + 1) == steps:
            kl_cur, acc_cur = eval_val()
            if kl_cur < best_kl:
                best_kl = kl_cur
                best_acc = acc_cur
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    kl1, acc1 = eval_val()
    return model.cpu(), {
        "val_kl_before": round(kl0, 5),
        "val_kl_after": round(min(kl1, best_kl), 5),
        "val_tok_acc_before": round(acc0, 4),
        "val_tok_acc_after": round(max(acc1, best_acc), 4),
    }


def repack_assistant_t16c_strip_embed(src_t16c: Path, dst_t16c: Path) -> Tuple[int, int]:
    """
    Create an embed-free `assistant_v7_body.t16c` by copying all norms and all linear
    records EXCEPT `model.embed_tokens.weight`, maintaining exact 64-byte record alignment.
    """
    if not src_t16c.is_file():
        raise FileNotFoundError(f"Missing source assistant .t16c: {src_t16c}")
    file_len = src_t16c.stat().st_size
    if file_len < 24:
        raise ValueError(f"Truncated T16C header in {src_t16c}")

    dst_t16c.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst_t16c.with_suffix(".t16c.tmp")
    kept_mats = 0
    dropped = 0
    try:
        with open(src_t16c, "rb") as fin:
            hdr = fin.read(24)
            magic, ver, flags, n_norms, n_mats, reserved = struct.unpack("<4s5I", hdr)
            if magic != T16C_MAGIC or ver != T16C_VERSION or reserved != 0:
                raise ValueError(f"Invalid T16C header in {src_t16c}")

            norm_chunks: List[bytes] = []
            for _ in range(n_norms):
                klen_b = fin.read(2)
                klen = struct.unpack("<H", klen_b)[0]
                k_b = fin.read(klen)
                alen_b = fin.read(4)
                alen = struct.unpack("<I", alen_b)[0]
                vec_b = fin.read(alen * 4)
                norm_chunks.append(klen_b + k_b + alen_b + vec_b)

            mat_meta: List[Tuple[bytes, bytes, int, int]] = []
            for _ in range(n_mats):
                rec_hdr = fin.read(34)
                (
                    nlen,
                    do,
                    di,
                    bs,
                    nb,
                    asym,
                    fwht,
                    invk,
                    rk,
                    pb_len,
                    lr_len,
                ) = struct.unpack("<HIIHBBBBHQQ", rec_hdr)
                name_b = fin.read(nlen)
                name = name_b.decode("utf-8")
                in_pos = fin.tell()
                in_pad = (64 - (in_pos % 64)) % 64
                if in_pad:
                    fin.seek(in_pad, 1)
                inv_b = (di * 4) if invk == 1 else ((di * 2) if invk == 2 else 0)
                payload_off = fin.tell()
                payload_total = pb_len + inv_b + lr_len
                fin.seek(payload_total, 1)
                if name == "model.embed_tokens.weight":
                    dropped += 1
                    continue
                mat_meta.append((rec_hdr, name_b, payload_off, payload_total))

            if dropped != 1:
                raise RuntimeError(
                    f"Expected exactly 1 model.embed_tokens.weight record in {src_t16c}, found {dropped}"
                )
            kept_mats = len(mat_meta)
            with open(tmp_dst, "wb") as fout:
                fout.write(struct.pack("<4s5I", magic, ver, flags, n_norms, kept_mats, 0))
                pos = 24
                for nc in norm_chunks:
                    fout.write(nc)
                    pos += len(nc)
                for rec_hdr, name_b, p_off, p_len in mat_meta:
                    fout.write(rec_hdr)
                    fout.write(name_b)
                    pos += 34 + len(name_b)
                    pad = (64 - (pos % 64)) % 64
                    if pad:
                        fout.write(b"\x00" * pad)
                        pos += pad
                    fin.seek(p_off)
                    rem = p_len
                    while rem > 0:
                        blk = fin.read(min(rem, 8 * 1024 * 1024))
                        fout.write(blk)
                        rem -= len(blk)
                    pos += p_len
        os.replace(tmp_dst, dst_t16c)
    finally:
        if tmp_dst.exists():
            try:
                tmp_dst.unlink()
            except OSError:
                pass
    return kept_mats, dst_t16c.stat().st_size


def pack_chunk_rowmajor_2b_gaus(w_chunk: np.ndarray, block_size: int = 64) -> bytes:
    """
    Pack `(rows, d_in)` float32 chunk into Row-Major 2-bit non-uniform Gaussian T16C
    (`n_bits = 2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT`) matching `t16c.rs`.
    """
    rows, d_in = w_chunk.shape
    if d_in % block_size != 0 or block_size % 8 != 0:
        raise ValueError(f"Invalid dims ({rows}, {d_in}) for block_size={block_size}")
    pad_rows = (16 - (rows % 16)) % 16
    if pad_rows > 0:
        w_work = np.zeros((rows + pad_rows, d_in), dtype=np.float32)
        w_work[:rows, :] = w_chunk
    else:
        w_work = np.ascontiguousarray(w_chunk, dtype=np.float32)

    n_tiles = w_work.shape[0] // 16
    n_blks = d_in // block_size
    blk = w_work.reshape(n_tiles, 16, n_blks, block_size)

    sc = np.maximum(0.470 * np.sqrt(np.mean(blk * blk, axis=-1)), 1e-6).astype(np.float32)
    idx = np.zeros_like(blk, dtype=np.uint8)
    for _ in range(3):
        inv_s = np.where(sc > 1e-12, 1.0 / np.maximum(sc, 1e-12), 0.0)[..., None]
        idx = np.digitize(blk * inv_s, _GAUSS2_MID).astype(np.uint8)
        lv = _GAUSS2_LV[idx]
        den = np.sum(lv * lv, axis=-1)
        sc = np.where(den > 0.0, np.sum(lv * blk, axis=-1) / np.maximum(den, 1e-12), sc).astype(np.float32)
    sc_f16 = np.maximum(sc.astype(np.float16).astype(np.float32), 1e-6).astype("<f2")

    bpr = block_size // 8
    plane_b = 16 * bpr
    tblk_b = 32 + 2 * plane_b
    out = np.zeros((n_tiles, n_blks, tblk_b), dtype=np.uint8)
    sc_t = np.ascontiguousarray(sc_f16.transpose(0, 2, 1))
    out[:, :, 0:32] = sc_t.view(np.uint8).reshape(n_tiles, n_blks, 32)

    idx_t = np.ascontiguousarray(idx.transpose(0, 2, 1, 3))
    out[:, :, 32 : 32 + plane_b] = np.packbits((idx_t & 1).astype(np.uint8), axis=-1).reshape(n_tiles, n_blks, plane_b)
    out[:, :, 32 + plane_b : 32 + 2 * plane_b] = np.packbits(((idx_t >> 1) & 1).astype(np.uint8), axis=-1).reshape(
        n_tiles, n_blks, plane_b
    )
    return out.tobytes()


def pack_assistant_body_from_safetensors(
    safetensors_path: Path,
    out_path: Path,
    body_bits: int = 4,
    block_size: int = 64,
    has_fwht: bool = True,
    chunk_rows: int = 4096,
    verbose: bool = True,
) -> int:
    """
    Pack an embed-free `Gemma4AssistantForCausalLM` body `.t16c` directly from safetensors
    (`body_bits` in {2, 4, 8}), omitting `model.embed_tokens.weight`.
    """
    if body_bits not in (2, 4, 8):
        raise ValueError(f"body_bits must be 2, 4, or 8, got {body_bits}")
    if chunk_rows <= 0 or chunk_rows % 16 != 0:
        raise ValueError(f"chunk_rows must be positive multiple of 16, got {chunk_rows}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".t16c.tmp")
    try:
        with SafetensorsReader(safetensors_path) as sr, open(tmp_path, "wb") as fp:
            norm_keys, all_mat_keys = pa.collect_assistant_keys(sr.hdr)
            mat_keys = [mk for mk in all_mat_keys if mk != "model.embed_tokens.weight"]
            fp.write(struct.pack("<4s5I", T16C_MAGIC, T16C_VERSION, ARCH_FLAGS, len(norm_keys), len(mat_keys), 0))
            pos = 24
            for nk in norm_keys:
                arr = sr.read_f32_tensor(nk)
                nb = nk.encode("utf-8")
                a32 = np.ascontiguousarray(arr, dtype="<f4").ravel()
                fp.write(struct.pack("<H", len(nb)))
                fp.write(nb)
                fp.write(struct.pack("<I", a32.size))
                fp.write(a32.tobytes())
                pos += 2 + len(nb) + 4 + a32.size * 4
            for idx, mk in enumerate(mat_keys, 1):
                if body_bits in (4, 8):
                    pos = pa.stream_write_assistant_t16c_mat(
                        fp,
                        pos,
                        sr,
                        mk,
                        block_size=block_size,
                        n_bits=body_bits,
                        has_fwht=has_fwht and (body_bits == 4),
                        chunk_rows=chunk_rows,
                    )
                else:
                    info = sr.hdr[mk]
                    d_out, d_in = int(info["shape"][0]), int(info["shape"][1])
                    n_bits_rm = 2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT
                    exp_p = expected_tile16_bytes(d_out, d_in, block_size, n_bits_rm, False)
                    nb_b = mk.encode("utf-8")
                    fp.write(struct.pack("<HIIHBBBBHQQ", len(nb_b), d_out, d_in, block_size, n_bits_rm, 0, 0, 0, 0, exp_p, 0))
                    fp.write(nb_b)
                    pos += 34 + len(nb_b)
                    pad = (64 - (pos % 64)) % 64
                    if pad:
                        fp.write(b"\x00" * pad)
                        pos += pad
                    written = 0
                    for r0 in range(0, d_out, chunk_rows):
                        r1 = min(r0 + chunk_rows, d_out)
                        w_sub = pa._read_f32_rows(sr, mk, r0, r1)
                        chk = pack_chunk_rowmajor_2b_gaus(w_sub, block_size=block_size)
                        fp.write(chk)
                        written += len(chk)
                        del w_sub, chk
                    if written != exp_p:
                        raise AssertionError(f"{mk}: streamed {written} != {exp_p}")
                    pos += written
                if verbose:
                    print(f"[{idx:02d}/{len(mat_keys):02d}] packed Q{body_bits} body {mk}", flush=True)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    sz = out_path.stat().st_size
    if verbose:
        print(f"[OK] Packed embed-free assistant body {out_path} ({sz:,} B / {sz / (1024**2):.2f} MiB)")
    return sz


def train_from_activations(
    act_path: Path,
    tok_path: Path,
    out_bhdr: Path,
    *,
    d_byte: int = 256,
    max_bytes: int = 32,
    steps: int = 1200,
    batch_size: int = 128,
    lr: float = 3e-3,
    top_k: int = 64,
    device: str = "cpu",
) -> Dict[str, object]:
    """Train `ByteHeadGRU` from saved `(h_final, top_ids/logits, gold_ids)` `.pt` activations."""
    if not act_path.is_file():
        raise FileNotFoundError(f"Missing activations bundle: {act_path}")
    trie = BpePrefixTrie.from_tokenizer_json(tok_path)
    blob = torch.load(act_path, map_location="cpu", weights_only=False)
    h_all = blob["h_final"].float()
    gold_ids = [int(v) for v in blob["gold_ids"].tolist()]
    if "top_ids" in blob and "top_probs" in blob:
        sids = blob["top_ids"].tolist()
        sps = blob["top_probs"].tolist()
    elif "logits" in blob:
        lg = blob["logits"].float()
        tv, ti = torch.topk(lg, k=min(top_k, lg.shape[-1]), dim=-1)
        tp = F.softmax(tv, dim=-1)
        sids = ti.tolist()
        sps = tp.tolist()
    else:
        raise KeyError("Activations bundle must contain ('top_ids', 'top_probs') or 'logits'")

    keep = [i for i, gid in enumerate(gold_ids) if trie.tok_syms[gid]]
    h_all = h_all[keep]
    gold_ids = [gold_ids[i] for i in keep]
    sids = [sids[i] for i in keep]
    sps = [sps[i] for i in keep]
    # Guarantee gold token mass inclusion for clean Appendix F.1 denominator stability
    for i, gid in enumerate(gold_ids):
        if gid not in sids[i]:
            sids[i].append(gid)
            sps[i].append(1e-4)
            tot = sum(sps[i])
            sps[i] = [v / tot for v in sps[i]]

    n_val = max(32, len(gold_ids) // 10)
    model, metrics = train_byte_head_eot_distill(
        trie,
        h_all[:-n_val],
        sids[:-n_val],
        sps[:-n_val],
        gold_ids[:-n_val],
        h_all[-n_val:],
        sids[-n_val:],
        sps[-n_val:],
        gold_ids[-n_val:],
        d_byte=d_byte,
        max_bytes=max_bytes,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    sz = export_bhdr_v1(model, out_bhdr)
    rpt = {"bytes": sz, "mb": round(sz / (1024**2), 3), "metrics": metrics, "path": str(out_bhdr)}
    out_bhdr.with_suffix(".report.json").write_text(json.dumps(rpt, indent=2), encoding="utf-8")
    return rpt


def export_parity_fixture(out_dir: Path) -> None:
    """Export deterministic mini tokenizer, BHDR, h_final, and expected step0 + beam outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_json = {
        "added_tokens": [
            {"id": 0, "content": "<pad>"},
            {"id": 1, "content": "<eos>"},
            {"id": 2, "content": "<bos>"},
            {"id": 106, "content": "<turn|>"},
        ],
        "model": {
            "vocab": {
                "\u2581t": 10,
                "\u2581th": 11,
                "\u2581the": 12,
                "4": 13,
                "<0xF0>": 14,
            },
            "merges": [],
        },
    }
    tok_p = out_dir / "tokenizer.json"
    tok_p.write_text(json.dumps(tok_json), encoding="utf-8")
    trie = BpePrefixTrie.from_tokenizer_json(tok_p)

    torch.manual_seed(77)
    d_in, d_byte, max_b = 32, 32, 16
    model = ByteHeadGRU(d_in=d_in, d_byte=d_byte, max_bytes=max_b)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.25, 0.25)

    bhdr_p = out_dir / "test.bhdr"
    export_bhdr_v1(model, bhdr_p)
    bh = load_bhdr_v1(bhdr_p)

    h_np = np.linspace(-0.5, 0.5, d_in, dtype=np.float32)
    (out_dir / "h_final.f32").write_bytes(h_np.astype("<f4").tobytes())

    _, _, syms0, lps0 = eval_bhdr_np_init_and_logprobs(bh, trie, h_np, node=0)
    best_tok, lp1, lp2, depth = decode_bhdr_np_beam(bh, trie, h_np, beam_width=2)
    expected = {
        "best_tok": best_tok,
        "bytes_unrolled": depth,
        "top1_logp": lp1,
        "top2_logp": lp2,
        "step0_syms": [int(s) for s in syms0],
        "step0_log_probs": [float(v) for v in lps0],
    }
    (out_dir / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"[PARITY FIXTURE] wrote {out_dir} best_tok={best_tok} depth={depth}")


def _run_selftest() -> int:
    """Mandatory 6-gate fail-closed verification battery."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Mini vocab mirroring real Gemma special/SP/UTF8/byte-escape structure
    id_to_tok = [""] * 120
    id_to_tok[0] = "<pad>"
    id_to_tok[1] = "<eos>"
    id_to_tok[2] = "<bos>"
    id_to_tok[10] = "\u2581o"
    id_to_tok[11] = "\u2581ok"
    id_to_tok[12] = "\u2581cat"
    id_to_tok[13] = "4"
    id_to_tok[14] = "\u258142"
    id_to_tok[15] = "<0xF0>"
    id_to_tok[16] = "é"
    id_to_tok[106] = "<turn|>"
    trie = BpePrefixTrie(id_to_tok)
    active_ids = [tid for tid, s in enumerate(id_to_tok) if s]

    # Disjoint synthetic subspace classification problem for real loss.backward()
    rng = np.random.default_rng(2026)
    d_in = 48
    n_toks = len(active_ids)
    q_orth, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(d_in, n_toks)).astype(np.float32))
    protos = q_orth.T

    def make_examples(n: int) -> Tuple[torch.Tensor, List[List[int]], List[List[float]], List[int]]:
        hs = np.zeros((n, d_in), dtype=np.float32)
        sids_all, sps_all, gids_all = [], [], []
        for i in range(n):
            c = i % n_toks
            gid = active_ids[c]
            hs[i] = 2.5 * protos[c] + rng.normal(0.0, 0.1, size=(d_in,)).astype(np.float32)
            # Soft teacher distribution peaked on c, with non-zero sibling mass (testing prefix marginalization)
            probs = np.full((n_toks,), 0.01 / max(n_toks - 1, 1), dtype=np.float32)
            probs[c] = 0.90
            sib = (c + 1) % n_toks
            probs[sib] += 0.09
            probs /= float(np.sum(probs))
            hs[i] += 0.35 * protos[sib]
            sids_all.append(list(active_ids))
            sps_all.append([float(x) for x in probs])
            gids_all.append(gid)
        return torch.from_numpy(hs), sids_all, sps_all, gids_all

    h_tr, sid_tr, sp_tr, g_tr = make_examples(220)
    h_va, sid_va, sp_va, g_va = make_examples(55)

    model, metrics = train_byte_head_eot_distill(
        trie,
        h_tr,
        sid_tr,
        sp_tr,
        g_tr,
        h_va,
        sid_va,
        sp_va,
        g_va,
        d_byte=64,
        max_bytes=16,
        steps=180,
        batch_size=32,
        lr=6e-3,
        seed=11,
    )

    with tempfile.TemporaryDirectory() as td:
        tmpd = Path(td)
        bhdr_p = tmpd / "selftest.bhdr"
        written = export_bhdr_v1(model, bhdr_p)
        bh = load_bhdr_v1(bhdr_p)

        # Gate 1: Stream header & exact byte length
        gate(
            "Gate 1: BHDR header & byte count integrity",
            written == bhdr_p.stat().st_size and bh["d_in"] == d_in and bh["vocab_sym"] == BHDR_VOCAB_SYM,
            f"bytes={written}",
        )

        # Gate 2: Quantized decode variance & finite bounds
        wi_var = float(np.var(bh["w_init"].astype(np.float32)))
        wo_var = float(np.var(bh["w_out"].astype(np.float32)))
        gate(
            "Gate 2: INT8 weight decode variance & finite scales",
            wi_var > 50.0 and wo_var > 50.0 and bool(np.all(np.isfinite(bh["s_out"]))),
            f"var_winit={wi_var:.1f} var_wout={wo_var:.1f}",
        )

        # Gate 3: Non-rank-1 distinctness across inputs
        _, _, _, lp0 = eval_bhdr_np_init_and_logprobs(bh, trie, h_va[0].numpy(), 0)
        _, _, _, lp1 = eval_bhdr_np_init_and_logprobs(bh, trie, h_va[5].numpy(), 0)
        diff = float(np.sum(np.abs(lp0 - lp1)))
        gate("Gate 3: Distinct inputs produce distinct step-0 distributions", diff > 0.5, f"l1_diff={diff:.3f}")

        # Gate 4: Exact Appendix F.1 telescoping identity + real KL drop + beam decode accuracy >= 0.90
        telescoping_max_err = 0.0
        p_test = np.array([float(v) for v in sp_va[0]], dtype=np.float64)
        for idx_t, tid in enumerate(active_ids):
            steps_tgt = compute_eot_byte_targets(trie, active_ids, p_test, tid)
            prod = 1.0
            for _, valid, q, gsym in steps_tgt:
                pos_s = valid.index(gsym)
                prod *= float(q[pos_s])
                telescoping_max_err = max(telescoping_max_err, abs(float(np.sum(q)) - 1.0))
            telescoping_max_err = max(telescoping_max_err, abs(prod - p_test[idx_t]))

        q_hits = 0
        for vi in range(len(g_va)):
            ptok, _, _, _ = decode_bhdr_np_beam(bh, trie, h_va[vi].numpy(), beam_width=2)
            if ptok == g_va[vi]:
                q_hits += 1
        q_acc = q_hits / max(len(g_va), 1)
        gate(
            "Gate 4: Exact F.1 <eot> chain-rule identity & real backward() KL + beam acc >= 0.90",
            telescoping_max_err < 1e-6
            and metrics["val_kl_after"] < metrics["val_kl_before"] * 0.25
            and q_acc >= 0.90,
            f"f1_err={telescoping_max_err:.2e} kl={metrics['val_kl_before']:.4f}->{metrics['val_kl_after']:.4f} beam_acc={q_acc:.3f}",
        )

        # Gate 5: Bijection coverage & T16C embed-stripping repacker
        synth_src = tmpd / "ast_full.t16c"
        synth_dst = tmpd / "ast_body.t16c"
        with open(synth_src, "wb") as fp:
            fp.write(struct.pack("<4s5I", T16C_MAGIC, T16C_VERSION, ARCH_FLAGS, 1, 2, 0))
            nk = b"model.norm.weight"
            fp.write(struct.pack("<H", len(nk)) + nk + struct.pack("<I", 4) + np.ones((4,), dtype="<f4").tobytes())
            for mname, nbits in [(b"model.embed_tokens.weight", 4), (b"pre_projection.weight", 8)]:
                payload_b = b"\x55" * 16
                fp.write(struct.pack("<HIIHBBBBHQQ", len(mname), 16, 64, 64, nbits, 0, 0, 0, 0, len(payload_b), 0))
                fp.write(mname)
                pos = fp.tell()
                pad = (64 - (pos % 64)) % 64
                fp.write(b"\x00" * pad + payload_b)
        kept, out_sz = repack_assistant_t16c_strip_embed(synth_src, synth_dst)
        w2_rm = np.random.default_rng(9).normal(0.0, 0.05, size=(16, 128)).astype(np.float32)
        p2_rm = pack_chunk_rowmajor_2b_gaus(w2_rm, block_size=64)
        exp2 = expected_tile16_bytes(16, 128, 64, 2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT, False)
        gate(
            "Gate 5: Trie 0 collisions + T16C embed-strip & Q2 RowMajor align",
            trie.collision_count == 0
            and trie.covered_tokens == len(active_ids)
            and kept == 1
            and 0 < out_sz < synth_src.stat().st_size
            and len(p2_rm) == exp2,
            f"covered={trie.covered_tokens}/{len(active_ids)} stripped_bytes={synth_src.stat().st_size}->{out_sz} q2_bytes={len(p2_rm)}",
        )

        # Gate 6: End-to-end FP32 vs exported BHDR INT8 step-0 logit cosine > 0.995
        with torch.no_grad():
            s0_fp, _ = model.init_state_and_ctx(h_va[:1])
            fp_logits = model.forward_logits(s0_fp)[0, trie.valid_syms_at(0)].numpy()
        _, _, _, q_lps = eval_bhdr_np_init_and_logprobs(bh, trie, h_va[0].numpy(), 0)
        fp_lps = fp_logits - float(np.max(fp_logits)) - math.log(float(np.sum(np.exp(fp_logits - float(np.max(fp_logits))))))
        cos = float(np.dot(fp_lps, q_lps) / ((np.linalg.norm(fp_lps) * np.linalg.norm(q_lps)) + 1e-12))
        gate("Gate 6: FP32 vs BHDR-INT8 step-0 log-prob cosine >= 0.995", cos >= 0.995, f"cos={cos:.6f}")

    print(f"ASSISTANT_V7_BHDR_SELFTEST: {6 - failures}/6 gates passed", flush=True)
    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _run_selftest()

    ap = argparse.ArgumentParser(description="Assistant-V7 BHDR v1 Byte-Head trainer, trie verifier & repacker")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-tokenizer", type=Path, default=None)
    ap.add_argument("--export-parity-fixture", type=Path, default=None)
    ap.add_argument("--repack-body-from-t16c", type=Path, default=None)
    ap.add_argument("--pack-body-safetensors", type=Path, default=None)
    ap.add_argument("--body-bits", type=int, choices=(2, 4, 8), default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--out-body", type=Path, default=None)
    ap.add_argument("--train-pilot", action="store_true", help="Train small architecture-validation .bhdr pilot (not converged weights)")
    ap.add_argument("--train-from-activations", action="store_true", help="Distill BHDR v1 from saved .pt activations")
    ap.add_argument("--activations", type=Path, default=None)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--out-bhdr", type=Path, default=None)
    ap.add_argument("--d-in", type=int, default=1024)
    ap.add_argument("--d-byte", type=int, default=256)
    ap.add_argument("--max-bytes", type=int, default=32)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if args.verify_tokenizer is not None:
        trie = BpePrefixTrie.from_tokenizer_json(args.verify_tokenizer)
        avg_sym = sum(len(s) for s in trie.tok_syms if s) / max(trie.covered_tokens, 1)
        print(
            f"TRIE_VERIFIED covered={trie.covered_tokens}/{len(trie.id_to_token)} "
            f"collisions={trie.collision_count} nodes={len(trie.node_edges)} "
            f"edges={trie.total_edges} max_depth={trie.max_depth} "
            f"avg_syms={avg_sym:.2f} csr_mb={trie.approx_rust_csr_bytes() / (1024**2):.2f}"
        )
        if trie.collisions_sample:
            print(f"COLLISIONS_SAMPLE: {trie.collisions_sample}")
        return 0 if trie.collision_count == 0 else 1

    if args.export_parity_fixture is not None:
        export_parity_fixture(args.export_parity_fixture)
        return 0

    if args.repack_body_from_t16c is not None:
        if args.out_body is None:
            raise ValueError("--out-body required with --repack-body-from-t16c")
        kept, sz = repack_assistant_t16c_strip_embed(args.repack_body_from_t16c, args.out_body)
        print(f"REPACKED_BODY kept_mats={kept} bytes={sz} mb={sz / (1024**2):.2f} path={args.out_body}")
        return 0

    if args.pack_body_safetensors is not None:
        if args.out_body is None:
            raise ValueError("--out-body required with --pack-body-safetensors")
        sz = pack_assistant_body_from_safetensors(
            args.pack_body_safetensors, args.out_body, body_bits=args.body_bits, block_size=args.block_size
        )
        print(f"PACKED_BODY bits={args.body_bits} bytes={sz} mb={sz / (1024**2):.2f} path={args.out_body}")
        return 0

    if args.train_from_activations:
        if args.activations is None or args.tokenizer is None or args.out_bhdr is None:
            raise ValueError("--activations, --tokenizer, and --out-bhdr required with --train-from-activations")
        rpt = train_from_activations(
            args.activations,
            args.tokenizer,
            args.out_bhdr,
            d_byte=args.d_byte,
            max_bytes=args.max_bytes,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        print(json.dumps(rpt, indent=2))
        return 0

    if args.train_pilot:
        if args.tokenizer is None or args.out_bhdr is None:
            raise ValueError("--tokenizer and --out-bhdr required with --train-pilot")
        trie = BpePrefixTrie.from_tokenizer_json(args.tokenizer)
        # Deterministic architecture-validation pilot over a compact common-token subset
        sample_strs = ["\u2581the", "\u2581is", "\u2581of", "\u2581and", "\u2581result", "4", ".", "<eos>", "<turn|>", "a", "\u2581a", "\u2581in", "0", "1", "2", " "]
        tok_ids = [i for i, s in enumerate(trie.id_to_token) if s in sample_strs]
        if not tok_ids:
            tok_ids = [i for i, s in enumerate(trie.tok_syms) if s][:32]
        rng = np.random.default_rng(7)
        h_tr = torch.from_numpy(rng.normal(0.0, 0.5, size=(64, args.d_in)).astype(np.float32))
        h_va = torch.from_numpy(rng.normal(0.0, 0.5, size=(16, args.d_in)).astype(np.float32))
        g_tr = [tok_ids[i % len(tok_ids)] for i in range(64)]
        g_va = [tok_ids[i % len(tok_ids)] for i in range(16)]
        probs = [1.0 / len(tok_ids)] * len(tok_ids)
        model, m = train_byte_head_eot_distill(
            trie,
            h_tr,
            [tok_ids] * 64,
            [probs] * 64,
            g_tr,
            h_va,
            [tok_ids] * 16,
            [probs] * 16,
            g_va,
            d_byte=args.d_byte,
            max_bytes=args.max_bytes,
            steps=args.steps,
            batch_size=16,
            lr=3e-3,
            device=args.device,
        )
        sz = export_bhdr_v1(model, args.out_bhdr)
        print(f"BHDR_PILOT_SAVED bytes={sz} mb={sz / (1024**2):.2f} kl={m['val_kl_before']}->{m['val_kl_after']} path={args.out_bhdr}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

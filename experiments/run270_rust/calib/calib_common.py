#!/usr/bin/env python3
"""Shared path resolution, Gemma-3 270M PyTorch helpers, FWHT, and parity evaluation."""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_DIR = REPO_ROOT / "experiments" / "gemma270_luj" / "validation"
FROZEN_SUITE = VALIDATION_DIR / "frozen_reasoning_set.json"
FRESH_SUITE = VALIDATION_DIR / "fresh_blind_reasoning_set.json"

if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))
from run_unseen_eval import normalize as eval_normalize  # noqa: E402

HF5 = [
    ("arithmetic", [2, 105, 2364, 107, 3689, 563, 236743, 236778, 900, 236743, 236778, 236881, 106, 107, 105, 4368, 107]),
    ("capital", [2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107]),
    ("hamlet", [2, 105, 2364, 107, 15938, 7971, 124600, 236881, 106, 107, 105, 4368, 107]),
    ("multistep", [2, 105, 2364, 107, 50702, 236743, 236770, 236778, 2782, 236743, 236828, 236764, 30579, 236743, 236778, 236832, 236764, 1299, 1138, 236743, 236770, 236810, 236761, 24181, 607, 1186, 506, 3890, 236761, 106, 107, 105, 4368, 107]),
    ("minimum", [2, 105, 2364, 107, 3689, 563, 506, 7100, 529, 236743, 236810, 236812, 236832, 532, 236743, 236770, 236819, 236800, 236881, 106, 107, 105, 4368, 107]),
]


def resolve_model_dir(explicit: str | Path | None = None) -> Path:
    """Resolve Gemma 3 270M IT checkpoint directory from CLI, env, or repo-relative paths."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    for key in ("GEMMA3_270M_DIR", "GEMMA_270M_DIR"):
        val = os.environ.get(key, "").strip()
        if val:
            return Path(val).expanduser().resolve()
    for cand in (REPO_ROOT / "models" / "gemma-3-270m-it", REPO_ROOT.parent / "gemma-3-270m-it"):
        if (cand / "model.safetensors").is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "Gemma 3 270M model directory not configured; pass --model-dir or set GEMMA3_270M_DIR."
    )


def rmsnorm_gemma(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    inv = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return x * inv * (1.0 + weight.float())


def _rope_slice(x: torch.Tensor, base_freq: float, start_pos: int) -> torch.Tensor:
    t_len, _, hd = x.shape
    half = hd // 2
    pos = torch.arange(start_pos, start_pos + t_len, dtype=torch.float32, device=x.device)[:, None]
    inv_freq = 1.0 / (base_freq ** (torch.arange(0, hd, 2, dtype=torch.float32, device=x.device) / hd))
    angles = pos * inv_freq[None, :]
    cos = torch.cos(angles)[:, None, :]
    sin = torch.sin(angles)[:, None, :]
    return torch.cat([x[..., :half] * cos - x[..., half:] * sin, x[..., half:] * cos + x[..., :half] * sin], dim=-1)


def rope_rotate(x: torch.Tensor, base_freq: float, seq_len: int) -> torch.Tensor:
    del seq_len
    return _rope_slice(x, base_freq, 0)


def fwht_block(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    if block <= 1:
        return x
    orig_shape = x.shape
    d = orig_shape[-1]
    assert d % block == 0, f"{d} not divisible by {block}"
    y = x.reshape(*orig_shape[:-1], d // block, block).clone()
    h = 1
    while h < block:
        y = y.reshape(*orig_shape[:-1], d // block, block // (2 * h), 2, h)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        h *= 2
    return y.reshape(*orig_shape) * (1.0 / math.sqrt(block))


def forward_270m_layer_ov(x: torch.Tensor, layer_idx: int, cfg: dict, tensors: dict, ov: dict) -> torch.Tensor:
    prefix = f"model.layers.{layer_idx}."
    eps = cfg["rms_norm_eps"]
    hd = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    n_rep = n_heads // n_kv
    seq_len = x.shape[0]
    scale = 1.0 / math.sqrt(float(cfg["query_pre_attn_scalar"]))
    is_full = cfg["layer_types"][layer_idx] == "full_attention"
    rope_base = float(cfg["rope_theta"] if is_full else cfg["rope_local_base_freq"])

    def get_t(sub: str) -> torch.Tensor:
        name = prefix + sub
        return ov[name] if name in ov else tensors[name].float()

    h_attn_in = rmsnorm_gemma(x, get_t("input_layernorm.weight"), eps)
    q = F.linear(h_attn_in, get_t("self_attn.q_proj.weight")).view(seq_len, n_heads, hd)
    k = F.linear(h_attn_in, get_t("self_attn.k_proj.weight")).view(seq_len, n_kv, hd)
    v = F.linear(h_attn_in, get_t("self_attn.v_proj.weight")).view(seq_len, n_kv, hd)

    q = rmsnorm_gemma(q, get_t("self_attn.q_norm.weight"), eps)
    k = rmsnorm_gemma(k, get_t("self_attn.k_norm.weight"), eps)
    q = rope_rotate(q, rope_base, seq_len)
    k = rope_rotate(k, rope_base, seq_len)

    if n_rep > 1:
        k_exp = k.repeat_interleave(n_rep, dim=1)
        v_exp = v.repeat_interleave(n_rep, dim=1)
    else:
        k_exp, v_exp = k, v

    scores = torch.bmm(q.permute(1, 0, 2), k_exp.permute(1, 0, 2).transpose(1, 2)) * scale
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    if not is_full and seq_len > cfg["sliding_window"]:
        causal = causal + torch.tril(torch.full((seq_len, seq_len), float("-inf")), diagonal=-cfg["sliding_window"])
    probs = F.softmax(scores + causal[None, :, :], dim=-1)
    ctx = torch.bmm(probs, v_exp.permute(1, 0, 2)).permute(1, 0, 2).contiguous().view(seq_len, n_heads * hd)

    attn_out = F.linear(ctx, get_t("self_attn.o_proj.weight"))
    h = x + rmsnorm_gemma(attn_out, get_t("post_attention_layernorm.weight"), eps)

    h_mlp_in = rmsnorm_gemma(h, get_t("pre_feedforward_layernorm.weight"), eps)
    gate = F.linear(h_mlp_in, get_t("mlp.gate_proj.weight"))
    up = F.linear(h_mlp_in, get_t("mlp.up_proj.weight"))
    act = F.gelu(gate, approximate="tanh") * up
    mlp_out = F.linear(act, get_t("mlp.down_proj.weight"))
    return h + rmsnorm_gemma(mlp_out, get_t("post_feedforward_layernorm.weight"), eps)


def forward_270m_logits_ov(ids, cfg: dict, tensors: dict, ov: dict, lm_head_w: torch.Tensor = None) -> torch.Tensor:
    emb = tensors["model.embed_tokens.weight"]
    idx = torch.tensor(ids, dtype=torch.long)
    x = emb[idx].float() * math.sqrt(float(cfg["hidden_size"]))
    for l in range(cfg["num_hidden_layers"]):
        x = forward_270m_layer_ov(x, l, cfg, tensors, ov)
    last = rmsnorm_gemma(x[-1:], ov.get("model.norm.weight", tensors["model.norm.weight"].float()), cfg["rms_norm_eps"])
    head = lm_head_w if lm_head_w is not None else emb.float()
    return F.linear(last, head).squeeze(0)


def generate_greedy_kv(ids, cfg: dict, tensors: dict, ov: dict, max_tokens: int = 24, lm_head_w: torch.Tensor = None):
    emb = tensors["model.embed_tokens.weight"]
    head = lm_head_w if lm_head_w is not None else emb.float()
    eps = cfg["rms_norm_eps"]
    hd = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    n_rep = n_heads // n_kv
    n_layers = cfg["num_hidden_layers"]
    sw = cfg["sliding_window"]
    scale = 1.0 / math.sqrt(float(cfg["query_pre_attn_scalar"]))

    def gt(name: str) -> torch.Tensor:
        return ov[name] if name in ov else tensors[name].float()

    layers_w = []
    for l in range(n_layers):
        p = f"model.layers.{l}."
        is_full = cfg["layer_types"][l] == "full_attention"
        rope_base = float(cfg["rope_theta"] if is_full else cfg["rope_local_base_freq"])
        layers_w.append((
            is_full, rope_base,
            gt(p + "input_layernorm.weight"),
            gt(p + "self_attn.q_proj.weight"),
            gt(p + "self_attn.k_proj.weight"),
            gt(p + "self_attn.v_proj.weight"),
            gt(p + "self_attn.q_norm.weight"),
            gt(p + "self_attn.k_norm.weight"),
            gt(p + "self_attn.o_proj.weight"),
            gt(p + "post_attention_layernorm.weight"),
            gt(p + "pre_feedforward_layernorm.weight"),
            gt(p + "mlp.gate_proj.weight"),
            gt(p + "mlp.up_proj.weight"),
            gt(p + "mlp.down_proj.weight"),
            gt(p + "post_feedforward_layernorm.weight"),
        ))
    norm_f = gt("model.norm.weight")
    kv_cache = [None] * n_layers

    def step_tokens(tok_ids, start_pos: int):
        x = emb[torch.tensor(tok_ids, dtype=torch.long)].float() * math.sqrt(float(cfg["hidden_size"]))
        t_cur = x.shape[0]
        for l, lw in enumerate(layers_w):
            (is_full, rope_base, w_in, wq, wk, wv, qn, kn, wo, w_pa, w_pf, wg, wu, wd, w_pff) = lw
            hi = rmsnorm_gemma(x, w_in, eps)
            q = _rope_slice(rmsnorm_gemma(F.linear(hi, wq).view(t_cur, n_heads, hd), qn, eps), rope_base, start_pos).permute(1, 0, 2)
            k = _rope_slice(rmsnorm_gemma(F.linear(hi, wk).view(t_cur, n_kv, hd), kn, eps), rope_base, start_pos).permute(1, 0, 2)
            v = F.linear(hi, wv).view(t_cur, n_kv, hd).permute(1, 0, 2)
            if kv_cache[l] is not None:
                kc, vc = kv_cache[l]
                k = torch.cat([kc, k], dim=1)
                v = torch.cat([vc, v], dim=1)
            kv_cache[l] = (k, v)
            s_tot = k.shape[1]
            k_e = k.repeat_interleave(n_rep, dim=0) if n_rep > 1 else k
            v_e = v.repeat_interleave(n_rep, dim=0) if n_rep > 1 else v
            scores = torch.bmm(q, k_e.transpose(1, 2)) * scale
            if t_cur > 1:
                pos_q = torch.arange(start_pos, start_pos + t_cur)[:, None]
                pos_k = torch.arange(s_tot)[None, :]
                mask = torch.where(pos_k > pos_q, float("-inf"), 0.0)
                if not is_full:
                    mask = torch.where((pos_q - pos_k) >= sw, float("-inf"), mask)
                scores = scores + mask[None, :, :]
            elif not is_full and s_tot > sw:
                pos_k = torch.arange(s_tot)
                mask = torch.where((start_pos - pos_k) >= sw, float("-inf"), 0.0)
                scores = scores + mask[None, None, :]
            probs = F.softmax(scores, dim=-1)
            ctx = torch.bmm(probs, v_e).permute(1, 0, 2).contiguous().view(t_cur, n_heads * hd)
            h = x + rmsnorm_gemma(F.linear(ctx, wo), w_pa, eps)
            hm = rmsnorm_gemma(h, w_pf, eps)
            act = F.gelu(F.linear(hm, wg), approximate="tanh") * F.linear(hm, wu)
            x = h + rmsnorm_gemma(F.linear(act, wd), w_pff, eps)
        last = rmsnorm_gemma(x[-1:], norm_f, eps)
        return F.linear(last, head).squeeze(0)

    logits = step_tokens(ids, 0)
    pos = len(ids)
    out = []
    for _ in range(max_tokens):
        nxt = int(torch.argmax(logits).item())
        if nxt in (1, 106):
            break
        out.append(nxt)
        if len(out) == max_tokens:
            break
        logits = step_tokens([nxt], pos)
        pos += 1
    return out


def eval_suite(cfg, tensors, ov, tk: Tokenizer, lm_head_w=None, eval_reasoning: str = "auto", ref_dir: Path | None = None):
    if ref_dir is None:
        env_ref = os.environ.get("GEMMA3_270M_REF_DIR", "").strip()
        ref_dir = Path(env_ref).expanduser().resolve() if env_ref else (REPO_ROOT / "ref270")
    per_prompt = {}
    coss = []
    top1 = 0
    for name, ids in HF5:
        logits = forward_270m_logits_ov(ids, cfg, tensors, ov, lm_head_w=lm_head_w).numpy()
        ref_path = ref_dir / f"{name}.f32"
        if not ref_path.is_file():
            raise FileNotFoundError(f"Missing exact reference logits: {ref_path}")
        ref = np.fromfile(ref_path, dtype=np.float32)
        dot = float(np.dot(logits.astype(np.float64), ref.astype(np.float64)))
        cos = dot / (np.linalg.norm(logits.astype(np.float64)) * np.linalg.norm(ref.astype(np.float64)))
        coss.append(cos)
        top1 += int(int(np.argmax(logits)) == int(np.argmax(ref)))
        gen_ids = generate_greedy_kv(ids, cfg, tensors, ov, max_tokens=16, lm_head_w=lm_head_w)
        per_prompt[name] = (cos, tk.decode(gen_ids).strip().replace("\n", "\\n"))

    mean_cos = float(np.mean(coss))
    suite_scores = {"frozen": (0, 28), "fresh": (0, 30)}
    run_reason = (eval_reasoning == "always") or (eval_reasoning == "auto" and mean_cos >= 0.94)
    if run_reason:
        for label, path in (("frozen", FROZEN_SUITE), ("fresh", FRESH_SUITE)):
            data = json.loads(path.read_text(encoding="utf-8"))
            passed = 0
            total = len(data["tasks"])
            for item in data["tasks"]:
                text = "<start_of_turn>user\n" + item["question"] + "<end_of_turn>\n<start_of_turn>model\n"
                ids = tk.encode(text).ids
                gen = generate_greedy_kv(ids, cfg, tensors, ov, max_tokens=24, lm_head_w=lm_head_w)
                decoded = tk.decode(gen)
                if eval_normalize(decoded) == eval_normalize(item["gold"]):
                    passed += 1
            suite_scores[label] = (passed, total)
    return mean_cos, min(coss), top1, per_prompt, suite_scores, run_reason

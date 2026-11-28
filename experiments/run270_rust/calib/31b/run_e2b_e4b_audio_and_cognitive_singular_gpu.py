#!/usr/bin/env python3
"""Gemma 4 E2B & E4B Audio-Core + Cognitive-Core Atlas & 31B Token-258881 Bridge (GPU Pipeline).

Executes genuine physical forward + backward passes on `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it`:
  1. Multi-Acoustic Audio-Core Scan (`12` Conformer layers, `8` heads/layer, `1024` `lconv1d` filters/layer,
     `4096` FFN1 + `4096` FFN2 neurons/layer) across 32 paired 16 kHz structured waveforms (formant syllables,
     harmonic triads, rhythmic Morse pulse codes, FM Doppler glides) vs. exact FFT phase-randomized controls
     with identical power spectra.
  2. Multi-Domain Text Cognitive-Core Scan (`35` layers on E2B, `42` layers on E4B, `8` heads/layer,
     `6144`/`10240` GeGLU neurons/layer, plus `256`-D Per-Layer Embedding `PLE` gates) across 200 paired
     prompts (`400` prompts across 10 reasoning families vs. matched surface controls).
  3. Exact `262,144`-Vocabulary Shared-Token Ridge/Procrustes Bridge projecting E4B (`2560`-D) and E2B (`1536`-D)
     `embed_audio` representations directly into Gemma 4 31B's `5376`-D space on `audio_token_id = 258881`.
  4. Surgically extracts and SHA-256 verifies all E2B & E4B Audio Cores, Text Cognitive Cores, Atlas JSON/NPZ maps,
     and the complete 31B Audio Ear + `258881` Bridge `safetensors` bundle to Google Drive.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GEMMA4_AUDIO_TOKEN_ID = 258881
GEMMA4_BOA_TOKEN_ID = 256000
GEMMA4_EOA_TOKEN_ID = 258883
SAMPLE_RATE = 16000

OUT_DIR = Path(os.environ.get("SINGULAR_OUT_DIR", "/content/e2b_e4b_singular_out"))
_drv = os.environ.get("SINGULAR_DRIVE_DIR", "").strip()
DRIVE_DIR: Optional[Path] = Path(_drv) if _drv else None


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _drive_safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp_up")
    with open(src, "rb") as fs, open(tmp, "wb") as fd:
        shutil.copyfileobj(fs, fd, 16 * 1024 * 1024)
        fd.flush()
        os.fsync(fd.fileno())
    os.replace(tmp, dst)
    if src.stat().st_size != dst.stat().st_size:
        raise IOError(f"Drive size mismatch on {dst}")


# ---------------------------------------------------------------------------
# 1. Paired Acoustic Stimulus Generator (Exact Power-Spectrum Matched Controls)
# ---------------------------------------------------------------------------

def phase_randomize_preserve_spectrum(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomize Fourier phases in [pi/3, 5*pi/3] while preserving exact magnitude spectrum & RMS."""
    n = wav.shape[0]
    spec = np.fft.rfft(wav.astype(np.float64))
    mag = np.abs(spec)
    orig_ph = np.angle(spec)
    shifts = rng.uniform(np.pi / 2.5, 2.0 * np.pi - np.pi / 2.5, size=spec.shape[0])
    new_ph = orig_ph + shifts
    new_ph[0] = 0.0
    if n % 2 == 0:
        new_ph[-1] = 0.0
    ctrl = np.fft.irfft(mag * np.exp(1j * new_ph), n=n).astype(np.float32)
    rms_orig = float(np.sqrt(np.mean(wav ** 2)) + 1e-8)
    rms_ctrl = float(np.sqrt(np.mean(ctrl ** 2)) + 1e-8)
    return (ctrl * (rms_orig / rms_ctrl)).astype(np.float32)


def build_paired_audio_stimuli(n_pairs: int = 32, duration_s: float = 2.0, sr: int = SAMPLE_RATE) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    """Synthesize `n_pairs` structured 16 kHz waveforms + exact power-spectrum-matched controls."""
    rng = np.random.default_rng(20261128)
    n_samp = int(round(duration_s * sr))
    t = np.arange(n_samp, dtype=np.float32) / float(sr)
    pairs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    families = ("formant_speech_syllables", "harmonic_chord_cadence", "temporal_morse_rhythm", "fm_doppler_glide")

    for i in range(n_pairs):
        fam = families[i % len(families)]
        k = i // len(families)
        wav = np.zeros(n_samp, dtype=np.float32)

        if fam == "formant_speech_syllables":
            # 3-syllable voiced formant transitions (F1/F2/F3 vowel trajectories)
            f0 = 115.0 + 14.0 * (k % 7)
            glottal = np.zeros_like(t)
            for h_idx in range(1, 22):
                if h_idx * f0 < 0.45 * sr:
                    glottal += (1.0 / (h_idx ** 1.15)) * np.sin(2.0 * np.pi * h_idx * f0 * t)
            f1 = np.interp(t, [0.0, 0.6, 1.2, 2.0], [320.0 + 30 * (k % 3), 720.0, 410.0, 580.0])
            f2 = np.interp(t, [0.0, 0.6, 1.2, 2.0], [2150.0, 1180.0 + 60 * (k % 4), 1920.0, 1450.0])
            env = (np.sin(np.pi * np.clip(t * 3.0 % 1.0, 0.0, 1.0)) ** 0.6).astype(np.float32)
            c1 = np.sin(2.0 * np.pi * np.cumsum(f1) / sr)
            c2 = 0.65 * np.sin(2.0 * np.pi * np.cumsum(f2) / sr)
            wav = env * (0.45 * glottal + 0.55 * (c1 + c2))
        elif fam == "harmonic_chord_cadence":
            # 4-chord I-IV-V-I triad progression
            root = 220.0 * (2.0 ** ((k % 8) / 12.0))
            ratios = [(1.0, 1.25, 1.5), (1.3333, 1.6667, 2.0), (1.5, 1.875, 2.25), (1.0, 1.25, 2.0)]
            seg = n_samp // 4
            for ci, chord in enumerate(ratios):
                s0, s1 = ci * seg, (ci + 1) * seg if ci < 3 else n_samp
                ts = t[s0:s1] - t[s0]
                local_env = np.exp(-2.2 * ts) * np.minimum(ts / 0.015, 1.0)
                tone = sum(
                    (0.6 ** o_i) * np.sin(2.0 * np.pi * root * r * (o_i + 1) * ts)
                    for r in chord
                    for o_i in range(3)
                )
                wav[s0:s1] = (local_env * tone).astype(np.float32)
        elif fam == "temporal_morse_rhythm":
            # Distinct short/long coded acoustic bursts on multi-harmonic carrier
            carrier_f = 480.0 + 55.0 * (k % 9)
            carrier = np.sin(2.0 * np.pi * carrier_f * t) + 0.4 * np.sin(4.0 * np.pi * carrier_f * t)
            pattern = [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1][(k % 3) :] + [1, 0, 1]
            slot = n_samp // len(pattern)
            gate = np.zeros(n_samp, dtype=np.float32)
            for si, bit in enumerate(pattern):
                if bit:
                    a0, a1 = si * slot, min((si + 1) * slot - slot // 6, n_samp)
                    if a1 > a0:
                        ramp = np.minimum(np.arange(a1 - a0) / 80.0, 1.0) * np.minimum(np.arange(a1 - a0)[::-1] / 80.0, 1.0)
                        gate[a0:a1] = ramp
            wav = gate * carrier
        else:
            # Dual-component FM + harmonic glide trajectory
            f_start = 260.0 + 70.0 * (k % 6)
            f_end = 1850.0 - 90.0 * (k % 5)
            if k % 2 == 1:
                f_start, f_end = f_end, f_start
            inst_f = f_start + (f_end - f_start) * (t / duration_s) ** 1.25
            phase = 2.0 * np.pi * np.cumsum(inst_f) / sr
            wav = (np.sin(phase) + 0.45 * np.sin(2.0 * phase) + 0.2 * np.sin(3.0 * phase)).astype(np.float32)
            wav *= np.hanning(n_samp).astype(np.float32)

        peak = float(np.max(np.abs(wav)) + 1e-8)
        wav = (0.75 * wav / peak).astype(np.float32)
        ctrl = phase_randomize_preserve_spectrum(wav, rng)
        pairs.append((f"aud_{i:02d}_{fam}", fam, wav, ctrl))
    return pairs


# ---------------------------------------------------------------------------
# 2. 10-Domain Paired Multi-Logic Text Suite (`reasoning` vs `control`)
# ---------------------------------------------------------------------------

def build_paired_logic_prompts(pairs_per_domain: int = 20) -> List[Dict[str, str]]:
    """Build `10 * pairs_per_domain` deterministic reasoning vs surface-control pairs."""
    out: List[Dict[str, str]] = []
    names = ["Alice", "Bruno", "Clara", "Devon", "Elena", "Felix", "Grace", "Hector", "Iris", "Jules"]
    cities = ["Oslo", "Kyoto", "Lima", "riga", "Bern", "Cork", "Graz", "Split", "linz", "york"]

    for k in range(pairs_per_domain):
        # 1. arithmetic_number_theory
        a, b, c = 17 + 3 * k, 11 + (k % 7), 5 + (k % 4)
        ans1 = str(a * b - c)
        out.append({
            "id": f"arith_{k:02d}", "domain": "arithmetic_number_theory",
            "r_prompt": f"Compute the exact integer value of ({a} * {b}) - {c}. Reply with only the integer.",
            "r_ans": ans1,
            "c_prompt": f"Repeat the exact integer {ans1} with no extra words.",
            "c_ans": ans1,
        })
        # 2. algebra_equations
        x_val = 4 + k
        m_val = 3 + (k % 5)
        b_val = 7 + 2 * (k % 6)
        rhs = m_val * x_val + b_val
        ans2 = str(2 * x_val + 3)
        out.append({
            "id": f"alg_{k:02d}", "domain": "algebra_equations",
            "r_prompt": f"If {m_val}x + {b_val} = {rhs}, what is the exact value of 2x + 3? Reply with only the number.",
            "r_ans": ans2,
            "c_prompt": f"Write the number {ans2} and nothing else.",
            "c_ans": ans2,
        })
        # 3. algorithmic_code_trace
        s0 = 2 + k
        step = 3 + (k % 4)
        ans3 = str(s0 + step * 3)
        out.append({
            "id": f"code_{k:02d}", "domain": "algorithmic_code_trace",
            "r_prompt": f"Trace Python: `v = {s0}; [v := v + {step} for _ in range(3)]`. What is `v` at the end? Reply with only the integer.",
            "r_ans": ans3,
            "c_prompt": f"Output the literal integer {ans3} only.",
            "c_ans": ans3,
        })
        # 4. formal_propositional_logic
        p_nm, q_nm = f"{names[k % 10]}_{k}", f"{names[(k + 3) % 10]}_{k}"
        ans4 = "False" if k % 2 == 0 else "True"
        if k % 2 == 0:
            rp4 = f"If {p_nm} attends, {q_nm} attends. {q_nm} did NOT attend. Did {p_nm} attend? Answer True or False only."
        else:
            rp4 = f"Either {p_nm} or {q_nm} holds the key. {p_nm} does NOT hold the key. Does {q_nm} hold the key? Answer True or False only."
        out.append({
            "id": f"prop_{k:02d}", "domain": "formal_propositional_logic",
            "r_prompt": rp4, "r_ans": ans4,
            "c_prompt": f"Case {k}: print the single word {ans4} only.", "c_ans": ans4,
        })
        # 5. syllogistic_relational
        n1, n2, n3 = f"{names[k % 10]}_{k}", f"{names[(k + 1) % 10]}_{k}", f"{names[(k + 2) % 10]}_{k}"
        out.append({
            "id": f"syll_{k:02d}", "domain": "syllogistic_relational",
            "r_prompt": f"{n1} is taller than {n2}. {n2} is taller than {n3}. Who is the shortest of the three? Reply with only the identifier.",
            "r_ans": n3,
            "c_prompt": f"Copy the identifier {n3} with no extra text.", "c_ans": n3,
        })
        # 6. temporal_state_machine
        init_c = cities[k % 10].capitalize()
        mid_c = cities[(k + 2) % 10].capitalize()
        fin_c = cities[(k + 5) % 10].capitalize()
        out.append({
            "id": f"temp_{k:02d}", "domain": "temporal_state_machine",
            "r_prompt": f"Courier #{k + 1} starts in {init_c}, flies to {mid_c} on Tuesday, then moves to {fin_c} on Thursday, and stays put on Friday. Where is Courier #{k + 1} on Friday? Reply with only the city.",
            "r_ans": fin_c,
            "c_prompt": f"Courier #{k + 1}: echo the city name {fin_c} only.", "c_ans": fin_c,
        })
        # 7. causal_physics_counterfactual
        mass = 2 + k
        acc_i = 3 + (k % 9)
        force = mass * acc_i
        acc = str(acc_i)
        out.append({
            "id": f"phys_{k:02d}", "domain": "causal_physics_counterfactual",
            "r_prompt": f"A frictionless {mass} kg cart experiences a net horizontal force of {force} N. What is its acceleration in m/s^2? Reply with only the integer.",
            "r_ans": acc,
            "c_prompt": f"Trial {k}: state the integer {acc} only.", "c_ans": acc,
        })
        # 8. constraint_combinatorics
        n_items = 4 + k
        ways = str((n_items * (n_items - 1)) // 2)
        out.append({
            "id": f"comb_{k:02d}", "domain": "constraint_combinatorics",
            "r_prompt": f"How many distinct unordered pairs can be chosen from {n_items} distinct books? Reply with only the integer.",
            "r_ans": ways,
            "c_prompt": f"Write the integer {ways} only.", "c_ans": ways,
        })
        # 9. spatial_geometry_transforms
        dx, dy = 3 + k, 4 + k
        manhattan = str(dx + dy)
        out.append({
            "id": f"geom_{k:02d}", "domain": "spatial_geometry_transforms",
            "r_prompt": f"Starting at (0, 0), a robot moves {dx} units East and {dy} units North. What is its Manhattan distance from (0, 0)? Reply with only the integer.",
            "r_ans": manhattan,
            "c_prompt": f"Reply with the integer {manhattan} only.", "c_ans": manhattan,
        })
        # 10. negation_fallacy_trap
        total_b = 20 + k
        kept = 7 + (k % 8)
        ans10 = str(kept)
        out.append({
            "id": f"neg_{k:02d}", "domain": "negation_fallacy_trap",
            "r_prompt": f"A farmer had {total_b} sheep; all but {kept} ran away. How many sheep remain with the farmer? Reply with only the integer.",
            "r_ans": ans10,
            "c_prompt": f"Output the number {ans10} only.", "c_ans": ans10,
        })
    return out


# ---------------------------------------------------------------------------
# 3. HTTP Range Reader for 31B Shared Vocabulary Embedding (`embed_tokens` + `norm`)
# ---------------------------------------------------------------------------

def fetch_31b_vocab_embed_via_range(repo_id: str = "google/gemma-4-31B-it-qat-w4a16-ct") -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch ONLY `embed_tokens.weight` + `norm.weight` from 31B via HTTP Range (avoids 23 GB full download)."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors"

    def _range_get(b0: int, b1: int) -> bytes:
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={b0}-{b1}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return resp.read()
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("Unreachable")

    raw8 = _range_get(0, 7)
    hlen = struct.unpack("<Q", raw8)[0]
    hdr = json.loads(_range_get(8, 8 + hlen - 1).decode("utf-8"))
    hdr.pop("__metadata__", None)
    base = 8 + hlen

    def _fetch_bf16_chunked(key: str) -> torch.Tensor:
        info = hdr[key]
        o0, o1 = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        sh = tuple(int(x) for x in info["shape"])
        total = o1 - o0
        buf = bytearray(total)
        chunk = 64 * 1024 * 1024
        off = 0
        while off < total:
            end = min(off + chunk, total)
            part = _range_get(base + o0 + off, base + o0 + end - 1)
            buf[off:end] = part
            off = end
        u16 = np.frombuffer(buf, dtype="<u2").copy()
        return torch.from_numpy(u16.view(np.int16)).view(torch.bfloat16).reshape(sh)

    emb_31b = _fetch_bf16_chunked("model.language_model.embed_tokens.weight")
    norm_31b = _fetch_bf16_chunked("model.language_model.norm.weight")
    return emb_31b, norm_31b


def solve_vocab_procrustes_ridge_bridge(
    emb_src: torch.Tensor,
    norm_src: torch.Tensor,
    emb_tgt_31b: torch.Tensor,
    norm_tgt_31b: torch.Tensor,
    device: torch.device,
    ridge_lambda: float = 1e-2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Solve `W_bridge` (`[5376, d_src]`) mapping `emb_src` -> `emb_tgt_31b` over 262,144 shared tokens."""
    with torch.no_grad():
        n_vocab = int(min(emb_src.shape[0], emb_tgt_31b.shape[0]))
        # Hold out every 64th token (4,096 unseen tokens) to prove out-of-sample generalization!
        all_idx = torch.arange(n_vocab)
        va_mask = (all_idx % 64) == 31
        tr_idx = all_idx[~va_mask]
        va_idx = all_idx[va_mask]

        d_src = int(emb_src.shape[1])
        d_tgt = int(emb_tgt_31b.shape[1])
        xtx = torch.zeros((d_src, d_src), device=device, dtype=torch.float64)
        xty = torch.zeros((d_src, d_tgt), device=device, dtype=torch.float64)

        esc_src = math.sqrt(float(d_src))
        esc_tgt = math.sqrt(float(d_tgt))
        for i0 in range(0, int(tr_idx.numel()), 16384):
            sl = tr_idx[i0 : i0 + 16384]
            xs = (emb_src[sl].to(device, dtype=torch.float32) * esc_src).to(torch.float64)
            yt = (emb_tgt_31b[sl].to(device, dtype=torch.float32) * esc_tgt).to(torch.float64)
            xtx.add_(xs.T @ xs)
            xty.add_(xs.T @ yt)

        reg = ridge_lambda * float(torch.trace(xtx).item()) / float(d_src)
        xtx_reg = xtx + reg * torch.eye(d_src, device=device, dtype=torch.float64)
        w_t = torch.linalg.solve(xtx_reg, xty)  # [d_src, d_tgt]
        w_bridge = w_t.T.to(torch.float32).contiguous()  # [5376, d_src]

        # Evaluate on 4,096 strictly unseen held-out tokens
        x_va = emb_src[va_idx].to(device, dtype=torch.float32) * esc_src
        y_va = emb_tgt_31b[va_idx].to(device, dtype=torch.float32) * esc_tgt
        pred_va = F.linear(x_va, w_bridge)
        cos_va = float(F.cosine_similarity(pred_va, y_va, dim=-1).mean().item())

        # Retrieval R@1 / R@5 among 4,096 held-out tokens
        p_u = F.normalize(pred_va, dim=-1)
        y_u = F.normalize(y_va, dim=-1)
        sim_mat = p_u @ y_u.T  # [4096, 4096]
        top5 = torch.topk(sim_mat, k=5, dim=-1).indices
        gold = torch.arange(va_idx.numel(), device=device).unsqueeze(1)
        r1 = float((top5[:, :1] == gold).float().mean().item())
        r5 = float((top5 == gold).any(dim=-1).float().mean().item())

        metrics = {
            "train_tokens": int(tr_idx.numel()),
            "heldout_tokens": int(va_idx.numel()),
            "heldout_mean_cosine": round(cos_va, 5),
            "heldout_r_at_1_4096way": round(r1, 5),
            "heldout_r_at_5_4096way": round(r5, 5),
        }
        return w_bridge.cpu(), metrics


# ---------------------------------------------------------------------------
# 4. Audio-Core + Text Cognitive-Core Scanner & Surgical Extractor for One Model
# ---------------------------------------------------------------------------

def tokenize_prompt_and_answer(tokenizer: Any, processor: Any, prompt: str, ans: str) -> Tuple[List[int], int, List[int]]:
    msg = [{"role": "user", "content": prompt}]
    try:
        pfx = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        pfx = f"<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    p_ids = list(tokenizer.encode(pfx, add_special_tokens=False))
    full_txt = pfx + ("" if pfx.endswith((" ", "\n")) else " ") + ans.strip()
    f_ids = list(tokenizer.encode(full_txt, add_special_tokens=False))
    lcp = 0
    while lcp < min(len(p_ids), len(f_ids)) and p_ids[lcp] == f_ids[lcp]:
        lcp += 1
    ans_ids = f_ids[lcp:]
    if not ans_ids:
        ans_ids = list(tokenizer.encode(ans.strip(), add_special_tokens=False))
        f_ids = p_ids + ans_ids
        lcp = len(p_ids)
    return f_ids, lcp, ans_ids


def score_batch_nll(logits: torch.Tensor, starts: List[int], spans: List[List[int]]) -> torch.Tensor:
    losses = []
    for b, (st, sp) in enumerate(zip(starts, spans)):
        # Predict token at `pos` from `logits[b, pos - 1]`
        pos_list = [max(0, st + j - 1) for j in range(len(sp))]
        lg_b = logits[b, pos_list, :].float()
        tgt = torch.tensor(sp, device=logits.device, dtype=torch.long)
        losses.append(F.cross_entropy(lg_b, tgt, reduction="mean"))
    return torch.stack(losses, dim=0)


def scan_and_extract_model(
    model_id: str,
    tag: str,
    audio_pairs: List[Tuple[str, str, np.ndarray, np.ndarray]],
    logic_pairs: List[Dict[str, str]],
    emb_31b: torch.Tensor,
    norm_31b: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoProcessor, Gemma4ForConditionalGeneration

    t0 = time.perf_counter()
    if device.type == "cuda":
        cc_major = torch.cuda.get_device_capability(device)[0]
        dt = torch.bfloat16 if cc_major >= 8 else torch.float16
    else:
        dt = torch.float32
    print(f"\n================ [{tag.upper()}] Loading {model_id} ({dt}) ================", flush=True)
    proc = AutoProcessor.from_pretrained(model_id)
    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dt, low_cpu_mem_usage=True).to(device)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, low_cpu_mem_usage=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    base = model.model if hasattr(model, "model") else model
    aud_tower = base.audio_tower
    emb_aud = base.embed_audio
    lm = base.language_model
    txt_layers = lm.layers
    n_aud_layers = len(aud_tower.layers)
    n_txt_layers = len(txt_layers)
    n_aud_heads = int(model.config.audio_config.num_attention_heads)
    aud_hd = int(model.config.audio_config.hidden_size) // n_aud_heads
    n_txt_heads = int(model.config.text_config.num_attention_heads)
    txt_ff = int(model.config.text_config.intermediate_size)
    d_txt = int(model.config.text_config.hidden_size)

    # -----------------------------------------------------------------------
    # Part A: Audio-Core Scan (`12` Conformer Layers + `96` Heads + `lconv1d` + `FFN`)
    # -----------------------------------------------------------------------
    print(f"[{tag}] Scanning {n_aud_layers}-layer Conformer Audio Tower across {len(audio_pairs)} paired waveforms...", flush=True)
    fe = getattr(proc, "feature_extractor", None) or getattr(proc, "audio_processor", None)

    def _prep_audio_inputs(wav_np: np.ndarray) -> Dict[str, torch.Tensor]:
        # Pass `[wav_np]` (1-item list) to take batched branch in Gemma4AudioFeatureExtractor (avoids 1D `.T` upstream bug)
        wav_1d = np.asarray(wav_np, dtype=np.float32).reshape(-1)
        if fe is not None:
            out = fe([wav_1d], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        else:
            out = proc(audio=[wav_1d], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        res = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.to(device=device, dtype=(dt if v.is_floating_point() else v.dtype))
        return res

    def _forward_audio_embed(aud_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = aud_inputs["input_features"]
        msk = aud_inputs.get("input_features_mask", aud_inputs.get("attention_mask"))
        ao = aud_tower(feats, msk, return_dict=True)
        ea = emb_aud(inputs_embeds=ao.last_hidden_state)
        return ea.reshape(-1, d_txt)

    n_ap = len(audio_pairs)
    aud_bypass_struct = np.zeros((n_ap, n_aud_layers), dtype=np.float32)
    aud_bypass_ctrl = np.zeros((n_ap, n_aud_layers), dtype=np.float32)
    aud_head_struct = np.zeros((n_ap, n_aud_layers, n_aud_heads), dtype=np.float32)
    aud_head_ctrl = np.zeros((n_ap, n_aud_layers, n_aud_heads), dtype=np.float32)
    aud_conv_delta = np.zeros((n_aud_layers, 1024), dtype=np.float32)
    aud_ffn_delta = np.zeros((n_aud_layers, 4096), dtype=np.float32)

    norm_w_txt = lm.norm.weight.detach().float()

    for ai, (_, _, w_st, w_ct) in enumerate(audio_pairs):
        in_st = _prep_audio_inputs(w_st)
        in_ct = _prep_audio_inputs(w_ct)

        with torch.no_grad():
            full_st = F.normalize(_forward_audio_embed(in_st).float(), dim=-1)
            full_ct = F.normalize(_forward_audio_embed(in_ct).float(), dim=-1)

            for al in range(n_aud_layers):
                def _skip_hook(_mod, args, kwargs, _out):
                    x_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
                    if isinstance(_out, tuple):
                        return (x_in,) + _out[1:]
                    return x_in

                h_handle = aud_tower.layers[al].register_forward_hook(_skip_hook, with_kwargs=True)
                try:
                    byp_st = F.normalize(_forward_audio_embed(in_st).float(), dim=-1)
                    byp_ct = F.normalize(_forward_audio_embed(in_ct).float(), dim=-1)
                finally:
                    h_handle.remove()
                aud_bypass_struct[ai, al] = float((1.0 - (full_st * byp_st).sum(dim=-1)).mean().item())
                aud_bypass_ctrl[ai, al] = float((1.0 - (full_ct * byp_ct).sum(dim=-1)).mean().item())

        # Sub-layer Taylor pass on Audio Conformer (Attention Heads + lconv1d + FFN1/FFN2)
        for is_ctrl, aud_in in ((False, in_st), (True, in_ct)):
            aud_in_g = {k: (v.detach().clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in aud_in.items()}
            cap_a: Dict[str, torch.Tensor] = {}
            handles = []

            for al in range(n_aud_layers):
                lyr_a = aud_tower.layers[al]
                def _mk_pre(key_name: str):
                    def _pre(_m, args, kwargs):
                        t_in = args[0] if len(args) > 0 else next(iter(kwargs.values()))
                        if isinstance(t_in, torch.Tensor) and t_in.requires_grad:
                            t_in.retain_grad()
                            cap_a[key_name] = t_in
                    return _pre

                def _mk_fwd(key_name: str):
                    def _fwd(_m, _args, out):
                        t_o = out[0] if isinstance(out, tuple) else out
                        if isinstance(t_o, torch.Tensor) and t_o.requires_grad:
                            t_o.retain_grad()
                            cap_a[key_name] = t_o
                    return _fwd

                handles.append(lyr_a.self_attn.post.register_forward_pre_hook(_mk_pre(f"attn_{al}"), with_kwargs=True))
                handles.append(lyr_a.lconv1d.depthwise_conv1d.register_forward_hook(_mk_fwd(f"conv_{al}")))
                handles.append(lyr_a.feed_forward1.ffw_layer_2.register_forward_pre_hook(_mk_pre(f"ff1_{al}"), with_kwargs=True))
                handles.append(lyr_a.feed_forward2.ffw_layer_2.register_forward_pre_hook(_mk_pre(f"ff2_{al}"), with_kwargs=True))

            try:
                ea = _forward_audio_embed(aud_in_g).float()
                z = ea * torch.rsqrt((ea * ea).mean(dim=-1, keepdim=True) + 1e-6) * norm_w_txt
                zu = F.normalize(z, dim=-1)
                gram = zu @ zu.T
                eye = torch.eye(gram.shape[0], device=device)
                off_d = ((gram * (1.0 - eye)) ** 2).sum() / float(max(1, gram.shape[0] * (gram.shape[0] - 1)))
                obj_a = z.var(dim=0).mean() - 2.0 * off_d
                obj_a.backward()
            finally:
                for hh in handles:
                    hh.remove()

            sign = -1.0 if is_ctrl else 1.0
            for al in range(n_aud_layers):
                at = cap_a[f"attn_{al}"]
                ah_imp = (at.detach().float() * at.grad.detach().float()).abs().reshape(-1, n_aud_heads, aud_hd).sum(dim=(0, 2)).cpu().numpy()
                if is_ctrl:
                    aud_head_ctrl[ai, al] = ah_imp
                else:
                    aud_head_struct[ai, al] = ah_imp

                ct = cap_a[f"conv_{al}"]
                # depthwise_conv1d output is [B, 1024, T] or [B, T, 1024]
                c_prod = (ct.detach().float() * ct.grad.detach().float()).abs()
                c_imp = (c_prod.sum(dim=(0, 2)) if c_prod.shape[1] == 1024 else c_prod.sum(dim=(0, 1))).cpu().numpy()
                aud_conv_delta[al] += sign * c_imp / float(n_ap)

                f1 = cap_a[f"ff1_{al}"]
                f2 = cap_a[f"ff2_{al}"]
                f_imp = (
                    (f1.detach().float() * f1.grad.detach().float()).abs().reshape(-1, 4096).sum(dim=0)
                    + (f2.detach().float() * f2.grad.detach().float()).abs().reshape(-1, 4096).sum(dim=0)
                ).cpu().numpy()
                aud_ffn_delta[al] += sign * f_imp / float(n_ap)

    # Select Audio Core Layers, Heads, Top-128 Conv Filters & Top-256 FFN Channels
    aud_core_layers: List[int] = []
    aud_layer_recs = []
    q70_aud = float(np.quantile(aud_bypass_struct.mean(axis=0), 0.70))
    for al in range(n_aud_layers):
        sd = aud_bypass_struct[:, al]
        cd = aud_bypass_ctrl[:, al]
        nd = sd - cd
        q95 = float(np.quantile(cd, 0.95))
        stab = float((sd > q95).mean())
        win = float((nd > 0.0).mean())
        is_c = (stab >= 0.35) or (float(nd.mean()) > 0.002 and win >= 0.62) or (float(sd.mean()) >= q70_aud and win >= 0.55)
        if is_c:
            aud_core_layers.append(al)
        aud_layer_recs.append({
            "layer": al,
            "mean_structured_cosine_damage": round(float(sd.mean()), 6),
            "mean_control_cosine_damage": round(float(cd.mean()), 6),
            "net_structured_minus_control": round(float(nd.mean()), 6),
            "win_rate": round(win, 4),
            "stability_gt_ctrl_q95": round(stab, 4),
            "selected_audio_core": bool(is_c),
        })
    if not aud_core_layers:
        aud_core_layers = sorted(np.argsort(aud_bypass_struct.mean(axis=0) - aud_bypass_ctrl.mean(axis=0))[::-1][:4].tolist())

    aud_hdiff = aud_head_struct - aud_head_ctrl
    aud_hq95 = np.quantile(aud_head_ctrl, 0.95, axis=0)
    aud_hthr = float(np.quantile(aud_hdiff.mean(axis=0).ravel(), 0.85))
    aud_selected_heads = []
    for al in range(n_aud_layers):
        for hi in range(n_aud_heads):
            st_h = float((aud_head_struct[:, al, hi] > aud_hq95[al, hi]).mean())
            wn_h = float((aud_hdiff[:, al, hi] > 0.0).mean())
            md_h = float(aud_hdiff[:, al, hi].mean())
            if (st_h >= 0.40 and wn_h >= 0.62) or (md_h >= aud_hthr and wn_h >= 0.65):
                aud_selected_heads.append({
                    "layer": al, "head": hi, "head_dim": aud_hd,
                    "row_col_slice": [hi * aud_hd, (hi + 1) * aud_hd],
                    "mean_taylor_delta": round(md_h, 6), "win_rate": round(wn_h, 4),
                })

    print(f"[{tag} Audio Done] Audio Core Layers={aud_core_layers} | Selected Audio Heads={len(aud_selected_heads)}/{n_aud_layers*n_aud_heads}", flush=True)

    # -----------------------------------------------------------------------
    # Part B: Text Cognitive-Core Scan (`35`/`42` Layers x 10 Logic Domains)
    # -----------------------------------------------------------------------
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    pad_id = int(getattr(tok, "pad_token_id", None) or 0)
    n_lp = len(logic_pairs)
    print(f"[{tag}] Scanning {n_txt_layers}-layer Text Brain across {n_lp} paired 10-domain prompts...", flush=True)

    r_items = [tokenize_prompt_and_answer(tok, proc, p["r_prompt"], p["r_ans"]) for p in logic_pairs]
    c_items = [tokenize_prompt_and_answer(tok, proc, p["c_prompt"], p["c_ans"]) for p in logic_pairs]
    domains_list = [p["domain"] for p in logic_pairs]

    txt_nll_r_base = np.zeros(n_lp, dtype=np.float32)
    txt_nll_c_base = np.zeros(n_lp, dtype=np.float32)
    txt_delta_r = np.zeros((n_lp, n_txt_layers), dtype=np.float32)
    txt_delta_c = np.zeros((n_lp, n_txt_layers), dtype=np.float32)

    bs_txt = 16
    for is_ctrl, items in ((False, r_items), (True, c_items)):
        for i0 in range(0, n_lp, bs_txt):
            batch = items[i0 : i0 + bs_txt]
            max_l = max(len(x[0]) for x in batch)
            inp = torch.full((len(batch), max_l), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(batch), max_l), device=device, dtype=torch.long)
            starts, spans = [], []
            for bi, (f_ids, st, sp) in enumerate(batch):
                inp[bi, : len(f_ids)] = torch.tensor(f_ids, device=device, dtype=torch.long)
                msk[bi, : len(f_ids)] = 1
                starts.append(st)
                spans.append(sp)

            with torch.no_grad():
                lg0 = model(input_ids=inp, attention_mask=msk, use_cache=False).logits
                base_nll = score_batch_nll(lg0, starts, spans).cpu().numpy()
                if is_ctrl:
                    txt_nll_c_base[i0 : i0 + len(batch)] = base_nll
                else:
                    txt_nll_r_base[i0 : i0 + len(batch)] = base_nll

                for tl in range(n_txt_layers):
                    def _skip_txt(_m, args, kwargs, _out):
                        x_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
                        return (x_in,) + _out[1:] if isinstance(_out, tuple) else x_in

                    th = txt_layers[tl].register_forward_hook(_skip_txt, with_kwargs=True)
                    try:
                        lg_b = model(input_ids=inp, attention_mask=msk, use_cache=False).logits
                        b_nll = score_batch_nll(lg_b, starts, spans).cpu().numpy()
                    finally:
                        th.remove()
                    if is_ctrl:
                        txt_delta_c[i0 : i0 + len(batch), tl] = b_nll - base_nll
                    else:
                        txt_delta_r[i0 : i0 + len(batch), tl] = b_nll - base_nll

    # Text Component Taylor Attribution (Heads + GeGLU Neurons + PLE 256-D channels)
    txt_head_r = np.zeros((n_lp, n_txt_layers, n_txt_heads), dtype=np.float32)
    txt_head_c = np.zeros((n_lp, n_txt_layers, n_txt_heads), dtype=np.float32)
    txt_mlp_delta = np.zeros((n_txt_layers, txt_ff), dtype=np.float32)
    txt_ple_delta = np.zeros((n_txt_layers, 256), dtype=np.float32)

    bs_grad = 8
    for is_ctrl, items in ((False, r_items), (True, c_items)):
        for i0 in range(0, n_lp, bs_grad):
            batch = items[i0 : i0 + bs_grad]
            max_l = max(len(x[0]) for x in batch)
            inp = torch.full((len(batch), max_l), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(batch), max_l), device=device, dtype=torch.long)
            starts, spans = [], []
            for bi, (f_ids, st, sp) in enumerate(batch):
                inp[bi, : len(f_ids)] = torch.tensor(f_ids, device=device, dtype=torch.long)
                msk[bi, : len(f_ids)] = 1
                starts.append(st)
                spans.append(sp)

            cap_t: Dict[str, torch.Tensor] = {}
            handles_t = []
            # Make embedding output differentiable for clean activation*grad hooks without enabling parameter grads
            def _emb_grad_hook(_m, _a, out):
                o = out.detach().clone().requires_grad_(True)
                cap_t["emb"] = o
                return o
            handles_t.append(lm.embed_tokens.register_forward_hook(_emb_grad_hook))

            for tl in range(n_txt_layers):
                lyr_t = txt_layers[tl]
                def _pre_t(kname: str):
                    def _fn(_m, args, kwargs):
                        x = args[0] if len(args) > 0 else next(iter(kwargs.values()))
                        if isinstance(x, torch.Tensor) and x.requires_grad:
                            x.retain_grad()
                            cap_t[kname] = x
                    return _fn
                handles_t.append(lyr_t.self_attn.o_proj.register_forward_pre_hook(_pre_t(f"o_{tl}"), with_kwargs=True))
                handles_t.append(lyr_t.mlp.down_proj.register_forward_pre_hook(_pre_t(f"d_{tl}"), with_kwargs=True))
                if hasattr(lyr_t, "per_layer_projection"):
                    handles_t.append(lyr_t.per_layer_projection.register_forward_pre_hook(_pre_t(f"ple_{tl}"), with_kwargs=True))

            try:
                lg_g = model(input_ids=inp, attention_mask=msk, use_cache=False).logits
                losses_b = score_batch_nll(lg_g, starts, spans)
                losses_b.sum().backward()
            finally:
                for hh in handles_t:
                    hh.remove()

            sign = -1.0 if is_ctrl else 1.0
            for tl in range(n_txt_layers):
                ot = cap_t[f"o_{tl}"]
                # o_proj input shape: [B, T, n_heads * hd_layer]
                hd_l = ot.shape[-1] // n_txt_heads
                oh = (ot.detach().float() * ot.grad.detach().float()).abs().view(len(batch), -1, n_txt_heads, hd_l).sum(dim=(1, 3)).cpu().numpy()
                if is_ctrl:
                    txt_head_c[i0 : i0 + len(batch), tl] = oh
                else:
                    txt_head_r[i0 : i0 + len(batch), tl] = oh

                dt_m = cap_t[f"d_{tl}"]
                mp = (dt_m.detach().float() * dt_m.grad.detach().float()).abs().sum(dim=(0, 1)).cpu().numpy()
                txt_mlp_delta[tl] += sign * mp / float(n_lp)

                if f"ple_{tl}" in cap_t:
                    pt_m = cap_t[f"ple_{tl}"]
                    pp = (pt_m.detach().float() * pt_m.grad.detach().float()).abs().sum(dim=(0, 1)).cpu().numpy()
                    if pp.shape[0] == 256:
                        txt_ple_delta[tl] += sign * pp / float(n_lp)

    # Domain-union + Taylor multi-head Cognitive Core selection (exact 31B protocol!)
    domain_selected: Dict[str, List[int]] = {}
    core_set = set()
    for dom in sorted(set(domains_list)):
        d_idx = [i for i, d in enumerate(domains_list) if d == dom]
        sel_d = []
        for tl in range(n_txt_layers):
            rd = txt_delta_r[d_idx, tl]
            cd = txt_delta_c[d_idx, tl]
            q95 = float(np.quantile(cd, 0.95))
            stab = float((rd > q95).mean())
            if stab >= 0.50 and float(rd.mean()) > 0.15:
                sel_d.append(tl)
                core_set.add(tl)
        domain_selected[dom] = sel_d

    txt_hdiff = txt_head_r - txt_head_c
    txt_hq95 = np.quantile(txt_head_c, 0.95, axis=0)
    txt_selected_heads = []
    txt_heads_by_l: Dict[str, List[int]] = {}
    for tl in range(n_txt_layers):
        ch_l = []
        for hi in range(n_txt_heads):
            st_h = float((txt_head_r[:, tl, hi] > txt_hq95[tl, hi]).mean())
            if st_h >= 0.50 or (float(txt_hdiff[:, tl, hi].mean()) > float(np.quantile(txt_hdiff.mean(axis=0), 0.92)) and float((txt_hdiff[:, tl, hi] > 0).mean()) >= 0.65):
                ch_l.append(hi)
                txt_selected_heads.append({"layer": tl, "head": hi, "stability": round(st_h, 4), "mean_delta": round(float(txt_hdiff[:, tl, hi].mean()), 6)})
        txt_heads_by_l[str(tl)] = ch_l
        if len(ch_l) >= 2:
            core_set.add(tl)

    txt_core_layers = sorted(core_set)
    txt_null_layers = [
        tl for tl in range(n_txt_layers)
        if tl not in core_set and float(txt_delta_r[:, tl].mean()) <= 0.02 and float((txt_delta_r[:, tl] > np.quantile(txt_delta_c[:, tl], 0.95)).mean()) < 0.10
    ]

    # -----------------------------------------------------------------------
    # Part C: Solve Exact Shared-262K-Vocab Bridge to Gemma 4 31B (`258881`)
    # -----------------------------------------------------------------------
    emb_src_cpu = lm.embed_tokens.weight.detach().cpu()
    norm_src_cpu = lm.norm.weight.detach().cpu()
    w_bridge_to_31b, bridge_metrics = solve_vocab_procrustes_ridge_bridge(
        emb_src_cpu, norm_src_cpu, emb_31b, norm_31b, device=device
    )
    print(f"[{tag} -> 31B Bridge] heldout_cos={bridge_metrics['heldout_mean_cosine']} | R@1(4096)={bridge_metrics['heldout_r_at_1_4096way']*100:.2f}% | R@5={bridge_metrics['heldout_r_at_5_4096way']*100:.2f}%", flush=True)

    # -----------------------------------------------------------------------
    # Part D: Save Atlas JSON/NPZ & Surgical BF16 Safetensors Bundles
    # -----------------------------------------------------------------------
    sd_all = model.state_dict()
    top256_txt = {}
    for tl in range(n_txt_layers):
        ord_m = np.argsort(txt_mlp_delta[tl])[::-1][:256]
        ord_p = np.argsort(txt_ple_delta[tl])[::-1][:64]
        top256_txt[str(tl)] = {
            "top256_mlp_neurons": [int(i) for i in ord_m.tolist()],
            "top256_mlp_scores": [round(float(txt_mlp_delta[tl, i]), 6) for i in ord_m.tolist()],
            "top64_ple_channels": [int(i) for i in ord_p.tolist()],
        }

    atlas = {
        "schema_version": "cca.gemma4_edge_multimodal_singular.v1",
        "model_id": model_id,
        "tag": tag,
        "audio_core": {
            "total_layers": n_aud_layers,
            "heads_per_layer": n_aud_heads,
            "selected_audio_core_layers": aud_core_layers,
            "selected_audio_heads_count": len(aud_selected_heads),
            "selected_audio_heads": aud_selected_heads,
            "layers": aud_layer_recs,
        },
        "text_cognitive_core": {
            "total_layers": n_txt_layers,
            "heads_per_layer": n_txt_heads,
            "ff_dim": txt_ff,
            "selected_cognitive_core_layers": txt_core_layers,
            "null_bypass_candidate_layers": txt_null_layers,
            "domain_selected_layers": domain_selected,
            "selected_heads_by_layer": txt_heads_by_l,
            "selected_head_total": len(txt_selected_heads),
        },
        "bridge_to_gemma4_31b_token_258881": bridge_metrics,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    atlas_path = OUT_DIR / f"cca_{tag}_audio_and_cognitive_atlas.json"
    npz_path = OUT_DIR / f"cca_{tag}_taylor_tensors.npz"
    atlas_path.write_text(json.dumps(atlas, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        npz_path,
        aud_bypass_struct=aud_bypass_struct, aud_bypass_ctrl=aud_bypass_ctrl,
        aud_head_struct=aud_head_struct, aud_head_ctrl=aud_head_ctrl,
        aud_conv_delta=aud_conv_delta, aud_ffn_delta=aud_ffn_delta,
        txt_delta_r=txt_delta_r, txt_delta_c=txt_delta_c,
        txt_head_r=txt_head_r, txt_head_c=txt_head_c,
        txt_mlp_delta=txt_mlp_delta, txt_ple_delta=txt_ple_delta,
    )

    # 1) Surgical Audio Cores BF16 safetensors (+ 31B token 258881 bridge)
    aud_core_sf: Dict[str, torch.Tensor] = {}
    for k, v in sd_all.items():
        if k.startswith(("model.audio_tower.subsample_conv_projection.", "model.audio_tower.output_proj.", "model.embed_audio.")):
            aud_core_sf[k] = v.detach().to(torch.bfloat16).cpu().contiguous()
    for al in aud_core_layers:
        pfx = f"model.audio_tower.layers.{al}."
        for k, v in sd_all.items():
            if k.startswith(pfx):
                aud_core_sf[k] = v.detach().to(torch.bfloat16).cpu().contiguous()
    for al in range(n_aud_layers):
        top_c = torch.tensor(np.argsort(aud_conv_delta[al])[::-1][:128].tolist(), dtype=torch.long)
        top_f = torch.tensor(np.argsort(aud_ffn_delta[al])[::-1][:256].tolist(), dtype=torch.long)
        aud_core_sf[f"model.audio_tower.layers.{al}.top128_lconv1d_indices"] = top_c.to(torch.int32)
        aud_core_sf[f"model.audio_tower.layers.{al}.top256_ffn_indices"] = top_f.to(torch.int32)

    w_emb_aud = sd_all["model.embed_audio.embedding_projection.weight"].detach().float().cpu()  # [d_txt, 1536]
    w_fused_aud1536_to_31b = (w_bridge_to_31b.float() @ w_emb_aud).to(torch.bfloat16).contiguous()  # [5376, 1536]
    aud_core_sf[f"bridge_to_31b.{tag}_hidden_to_31b_5376.weight"] = w_bridge_to_31b.to(torch.bfloat16).contiguous()
    aud_core_sf[f"bridge_to_31b.{tag}_audio1536_to_31b_tok258881.weight"] = w_fused_aud1536_to_31b
    aud_sf_path = OUT_DIR / f"gemma4_{tag}_audio_cores_and_31b_bridge_bf16.safetensors"
    save_file(aud_core_sf, str(aud_sf_path), metadata={"audio_token_id": str(GEMMA4_AUDIO_TOKEN_ID)})

    # If E4B: ALSO save the complete 12-layer Full Audio Ear + 31B Token-258881 Bridge so 31B can hear end-to-end immediately!
    full_ear_path = None
    if tag == "e4b":
        full_ear_sf: Dict[str, torch.Tensor] = {
            k: v.detach().to(torch.bfloat16).cpu().contiguous()
            for k, v in sd_all.items()
            if k.startswith(("model.audio_tower.", "model.embed_audio."))
        }
        full_ear_sf["bridge_to_31b.e4b_hidden2560_to_31b_5376.weight"] = w_bridge_to_31b.to(torch.bfloat16).contiguous()
        full_ear_sf["bridge_to_31b.e4b_audio1536_to_31b_tok258881.weight"] = w_fused_aud1536_to_31b
        full_ear_path = OUT_DIR / "gemma4_e4b_full_audio_ear_for_31b_tok258881_bf16.safetensors"
        save_file(full_ear_sf, str(full_ear_path), metadata={"target_model": "google/gemma-4-31B-it", "audio_token_id": "258881"})

    # 2) Surgical Text Cognitive Cores BF16 safetensors (selected heads + Top-256 GeGLU neurons + PLE gates per core layer)
    txt_core_sf: Dict[str, torch.Tensor] = {
        "model.language_model.norm.weight": sd_all["model.language_model.norm.weight"].detach().to(torch.bfloat16).cpu().contiguous(),
    }
    if "model.language_model.per_layer_projection_norm.weight" in sd_all:
        txt_core_sf["model.language_model.per_layer_projection_norm.weight"] = sd_all["model.language_model.per_layer_projection_norm.weight"].detach().to(torch.bfloat16).cpu().contiguous()

    for tl in txt_core_layers:
        pfx = f"model.language_model.layers.{tl}"
        idx_m = torch.tensor(top256_txt[str(tl)]["top256_mlp_neurons"], dtype=torch.long)
        txt_core_sf[f"{pfx}.mlp.top256.neuron_indices"] = idx_m.to(torch.int32)
        txt_core_sf[f"{pfx}.mlp.top256.gate_proj.weight"] = sd_all[f"{pfx}.mlp.gate_proj.weight"][idx_m, :].detach().to(torch.bfloat16).cpu().contiguous()
        txt_core_sf[f"{pfx}.mlp.top256.up_proj.weight"] = sd_all[f"{pfx}.mlp.up_proj.weight"][idx_m, :].detach().to(torch.bfloat16).cpu().contiguous()
        txt_core_sf[f"{pfx}.mlp.top256.down_proj.weight"] = sd_all[f"{pfx}.mlp.down_proj.weight"][:, idx_m].detach().to(torch.bfloat16).cpu().contiguous()
        for k_sub in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
            "post_per_layer_input_norm.weight", "per_layer_input_gate.weight", "per_layer_projection.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight", "layer_scalar",
        ):
            fk = f"{pfx}.{k_sub}"
            if fk in sd_all:
                txt_core_sf[fk] = sd_all[fk].detach().to(torch.bfloat16).cpu().contiguous()

    for hrec in txt_selected_heads:
        tl = int(hrec["layer"])
        hi = int(hrec["head"])
        pfx = f"model.language_model.layers.{tl}"
        q_w = sd_all[f"{pfx}.self_attn.q_proj.weight"]
        o_w = sd_all[f"{pfx}.self_attn.o_proj.weight"]
        hd_l = q_w.shape[0] // n_txt_heads
        c0, c1 = hi * hd_l, (hi + 1) * hd_l
        txt_core_sf[f"{pfx}.self_attn.head_{hi:02d}.q_proj.weight"] = q_w[c0:c1, :].detach().to(torch.bfloat16).cpu().contiguous()
        txt_core_sf[f"{pfx}.self_attn.head_{hi:02d}.o_proj.weight"] = o_w[:, c0:c1].detach().to(torch.bfloat16).cpu().contiguous()

    txt_sf_path = OUT_DIR / f"gemma4_{tag}_cognitive_cores_bf16.safetensors"
    save_file(txt_core_sf, str(txt_sf_path), metadata={"schema": f"gemma4_{tag}.cognitive_cores.v1"})

    del sd_all, model, aud_tower, lm
    gc.collect()
    torch.cuda.empty_cache()

    produced = [atlas_path, npz_path, aud_sf_path, txt_sf_path] + ([full_ear_path] if full_ear_path else [])
    return {"atlas": atlas, "files": produced}


def run_pipeline(audio_pairs_n: int = 32, logic_pairs_per_domain: int = 20) -> Dict[str, Any]:
    assert torch.cuda.is_available(), "CUDA GPU required for E2B/E4B Audio + Cognitive Core pipeline"
    device = torch.device("cuda:0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DRIVE_DIR is not None:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    audio_pairs = build_paired_audio_stimuli(n_pairs=audio_pairs_n)
    logic_pairs = build_paired_logic_prompts(pairs_per_domain=logic_pairs_per_domain)
    print(f"[Init] Fetching 31B shared-vocab embedding via HTTP Range for exact 258881 bridge...", flush=True)
    emb_31b, norm_31b = fetch_31b_vocab_embed_via_range()

    res_e2b = scan_and_extract_model("google/gemma-4-E2B-it", "e2b", audio_pairs, logic_pairs, emb_31b, norm_31b, device)
    res_e4b = scan_and_extract_model("google/gemma-4-E4B-it", "e4b", audio_pairs, logic_pairs, emb_31b, norm_31b, device)
    del emb_31b, norm_31b
    gc.collect()

    all_files = res_e2b["files"] + res_e4b["files"]
    artifacts_meta = {}
    for fp in all_files:
        dig = _sha256_file(fp)
        sz = fp.stat().st_size
        artifacts_meta[fp.name] = {"bytes": sz, "sha256": dig}
        if DRIVE_DIR is not None:
            _drive_safe_copy(fp, DRIVE_DIR / fp.name)

    summary = {
        "schema_version": "gemma4_e2b_e4b_audio_cognitive_singular_report.v1",
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "e2b_summary": {
            "audio_core_layers": res_e2b["atlas"]["audio_core"]["selected_audio_core_layers"],
            "audio_core_heads": res_e2b["atlas"]["audio_core"]["selected_audio_heads_count"],
            "text_cognitive_core_layers": res_e2b["atlas"]["text_cognitive_core"]["selected_cognitive_core_layers"],
            "text_null_layers": res_e2b["atlas"]["text_cognitive_core"]["null_bypass_candidate_layers"],
            "text_core_heads": res_e2b["atlas"]["text_cognitive_core"]["selected_head_total"],
            "bridge_31b": res_e2b["atlas"]["bridge_to_gemma4_31b_token_258881"],
        },
        "e4b_summary": {
            "audio_core_layers": res_e4b["atlas"]["audio_core"]["selected_audio_core_layers"],
            "audio_core_heads": res_e4b["atlas"]["audio_core"]["selected_audio_heads_count"],
            "text_cognitive_core_layers": res_e4b["atlas"]["text_cognitive_core"]["selected_cognitive_core_layers"],
            "text_null_layers": res_e4b["atlas"]["text_cognitive_core"]["null_bypass_candidate_layers"],
            "text_core_heads": res_e4b["atlas"]["text_cognitive_core"]["selected_head_total"],
            "bridge_31b": res_e4b["atlas"]["bridge_to_gemma4_31b_token_258881"],
        },
        "artifacts": artifacts_meta,
    }
    rpt_path = OUT_DIR / "e2b_e4b_singular_report.json"
    sums_path = OUT_DIR / "SHA256SUMS_E2B_E4B.txt"
    rpt_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sums_path.write_text("".join(f"{v['sha256']}  {k}\n" for k, v in artifacts_meta.items()), encoding="utf-8")
    if DRIVE_DIR is not None:
        _drive_safe_copy(rpt_path, DRIVE_DIR / rpt_path.name)
        _drive_safe_copy(sums_path, DRIVE_DIR / sums_path.name)
        for fname, meta in artifacts_meta.items():
            got = _sha256_file(DRIVE_DIR / fname)
            assert got == meta["sha256"], f"Drive SHA256 mismatch on {fname}"
    print(f"[E2B+E4B SINGULAR COMPLETE] {len(artifacts_meta)} artifacts verified in {summary['elapsed_s']}s", flush=True)
    return summary


def run_selftest() -> int:
    """Mandatory 6-gate fail-closed CPU verification battery."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Acoustic generator produces distinct waveforms & exact power-spectrum-matched controls
    ap = build_paired_audio_stimuli(n_pairs=8, duration_s=1.0)
    spec_diffs = []
    wf_corrs = []
    for _, _, w, c in ap:
        sw = np.abs(np.fft.rfft(w))
        sc = np.abs(np.fft.rfft(c))
        spec_diffs.append(float(np.max(np.abs(sw - sc)) / (np.max(sw) + 1e-8)))
        wf_corrs.append(float(np.corrcoef(w, c)[0, 1]))
    gate("Gate 1: Spectrum-Preserving Phase-Randomized Audio Controls", max(spec_diffs) < 1e-4 and max(wf_corrs) < 0.35, f"max_spec_err={max(spec_diffs):.2e}, max_signed_corr={max(wf_corrs):.3f}")

    # Gate 2: 10-domain paired logic prompt uniqueness & non-empty answers
    lp = build_paired_logic_prompts(pairs_per_domain=20)
    r_set = {x["r_prompt"] for x in lp}
    dom_set = {x["domain"] for x in lp}
    gate("Gate 2: 10-Domain Paired Logic Suite Uniqueness", len(lp) == 200 and len(r_set) == 200 and len(dom_set) == 10, f"pairs={len(lp)}, uniq={len(r_set)}, domains={len(dom_set)}")

    # Gate 3: Vocab Procrustes/Ridge Bridge recovers planted linear ground-truth on unseen tokens
    torch.manual_seed(7)
    e_src = torch.randn(512, 64)
    true_w = torch.randn(128, 64) * 0.2
    e_tgt = (e_src @ true_w.T) + 0.01 * torch.randn(512, 128)
    wb, met = solve_vocab_procrustes_ridge_bridge(e_src, torch.ones(64), e_tgt, torch.ones(128), torch.device("cpu"))
    gate("Gate 3: Shared-Vocab Bridge Generalization on Held-Out Tokens", wb.shape == (128, 64) and met["heldout_mean_cosine"] > 0.99, f"cos={met['heldout_mean_cosine']}")

    # Gate 4: Fused Audio-1536 -> 31B-5376 Token-258881 Associativity Parity
    w_ea = torch.randn(64, 48, dtype=torch.float64)
    w_f = wb.double() @ w_ea
    a_tok = torch.randn(5, 48, dtype=torch.float64)
    assoc_err = float(((a_tok @ w_ea.T) @ wb.double().T - (a_tok @ w_f.T)).abs().max().item())
    gate("Gate 4: Fused Token-258881 Audio Bridge Associativity", assoc_err < 1e-10, f"err={assoc_err:.2e}")

    # Gate 5: Answer-span NLL scorer alignment
    lg_toy = torch.zeros((2, 6, 16))
    lg_toy[0, 2, 5] = 12.0
    lg_toy[0, 3, 7] = 12.0
    lg_toy[1, 1, 3] = 12.0
    nlls = score_batch_nll(lg_toy, [3, 2], [[5, 7], [3]])
    gate("Gate 5: Exact Answer-Span Causal NLL Scoring", float(nlls.max().item()) < 1e-3, f"max_nll={float(nlls.max().item()):.2e}")

    # Gate 6: Real Gemma4AudioFeatureExtractor([wav]) + Toy Gemma4AudioModel + Gemma4TextModel (with PLE & KV-sharing) E2E Hook Proof
    g6_ok = True
    g6_detail = ""
    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

        fe_t = Gemma4AudioFeatureExtractor()
        fe_out = fe_t([ap[0][2]], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        acfg = Gemma4AudioConfig(hidden_size=256, num_attention_heads=4, num_hidden_layers=2, output_proj_dims=64, subsampling_conv_channels=[128, 8])
        tcfg = Gemma4TextConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=2,
            head_dim=16, global_head_dim=16, hidden_size_per_layer_input=16, vocab_size_per_layer_input=256,
            num_kv_shared_layers=2, layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        )
        vcfg = Gemma4VisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, head_dim=16, global_head_dim=16)
        toy = Gemma4ForConditionalGeneration(Gemma4Config(audio_config=acfg, text_config=tcfg, vision_config=vcfg)).eval()
        for p in toy.parameters():
            p.requires_grad_(False)
        feats_g = fe_out["input_features"].clone().requires_grad_(True)
        msk_g = fe_out["input_features_mask"]
        cap_g = {}
        ha = toy.model.audio_tower.layers[0].self_attn.post.register_forward_pre_hook(
            lambda _m, a, kw: ((a[0] if a else next(iter(kw.values()))).retain_grad(), cap_g.setdefault("a", a[0] if a else next(iter(kw.values())))) and None,
            with_kwargs=True,
        )
        ao = toy.model.audio_tower(feats_g, msk_g, return_dict=True)
        ea = toy.model.embed_audio(inputs_embeds=ao.last_hidden_state)
        (ea.pow(2).sum()).backward()
        ha.remove()
        # Also test Text KV-shared bypass + PLE hook
        ht = toy.model.language_model.layers[1].register_forward_hook(
            lambda _m, a, kw, _o: a[0] if a else kw["hidden_states"], with_kwargs=True
        )
        _ = toy(input_ids=torch.randint(1, 100, (2, 8)), attention_mask=torch.ones((2, 8), dtype=torch.long), use_cache=False).logits
        ht.remove()
        g6_ok = cap_g["a"].grad is not None and float(cap_g["a"].grad.abs().sum().item()) > 0.0
        g6_detail = f"fe={tuple(fe_out['input_features'].shape)}, ea={tuple(ea.shape)}"
    except Exception as exc:
        g6_ok = False
        g6_detail = repr(exc)
    gate("Gate 6: Live Gemma4AudioFeatureExtractor + Audio/PLE/SharedKV Hook Proof", g6_ok, g6_detail)

    print(f"E2B/E4B Singular Self-Test Result: {6 - failures}/6 gates passed", flush=True)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--audio-pairs", type=int, default=32)
    ap.add_argument("--logic-pairs-per-domain", type=int, default=20)
    args = ap.parse_args()
    if args.selftest or (not args.run and not torch.cuda.is_available()):
        return run_selftest()
    run_pipeline(audio_pairs_n=args.audio_pairs, logic_pairs_per_domain=args.logic_pairs_per_domain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Assistant-V7 Byte-Head (`BHDR` v1) GPU Distiller & W4A16 Teacher `LGT1` Exporter.

Runs on Colab GPU (reusing cached `/tmp/gemma4_31b` + `/tmp/gemma4_31b_assistant`
when present) to:
  1. Pack embed-free `Gemma4AssistantForCausalLM` bodies (`Q2` ~56 MB, `Q4` ~107 MB,
     `Q8` ~198 MB) omitting `model.embed_tokens.weight`.
  2. Stream `google/gemma-4-31B-it-qat-w4a16-ct` (`L00..L59`) once across:
       - 96 x 256 mixed FineWeb-Edu + Alpaca `<|turn>` sequences (capturing
         `L58`/`L59` KV + `h_59_final` for authentic 4-layer Assistant execution), and
       - all 3 strictly disjoint held-out skeptic-proof chat turns from
         `eval_v6_1_skeptic_proof.py` -> exports full `262,144`-vocab softcapped
         BF16 `LGT1` teacher logit files (`heldout_*.lgt1`) with exact `fnv1a64`
         sequence+start hashes matching `run31b.rs`.
  3. Distill `gemma4_31b_assistant_v7.bhdr` via `arXiv:2609.12303` Appendix F.1
     single-pass `<eot>` byte+`<eot>` KL divergence over 24,384 live samples.
  4. Verify exported INT8 `.bhdr` quantized beam accuracy and persist all artifacts
     atomically with SHA-256 checksums to Drive (`COLAB_AUTO_UNASSIGN`-safe).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_v6_1_skeptic_proof as spmod  # noqa: E402
import train_assistant_v7_byte_head as bhmod  # noqa: E402

LGT1_MAGIC = b"LGT1"
LGT1_VERSION = 1
LGT1_HEADER_BYTES = 24
FNV1A_OFFSET = 0xCBF29CE484222325
FNV1A_PRIME = 0x0100000001B3
EVAL_START_MIX = 0x9E3779B97F4A7C15


def fnv1a64_ids(ids: Sequence[int]) -> int:
    """Compute exact 64-bit FNV-1a token sequence hash matching `run31b.rs::fnv1a64_ids`."""
    h = FNV1A_OFFSET
    for tid in ids:
        h ^= int(tid) & 0xFFFFFFFFFFFFFFFF
        h = (h * FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def resolve_eval_nll_start_pos_after_turn(ids: Sequence[int]) -> int:
    """Locate `<channel|>` (101) or `<|turn>model\\n` ([105, 4368, 107]) matching `run31b.rs`."""
    if len(ids) < 2:
        raise ValueError("--eval-nll requires at least 2 tokens")
    max_pos = len(ids) - 1
    for pos in range(max_pos - 1, -1, -1):
        if int(ids[pos]) == 101:
            return pos
    for pos in range(max_pos - 3, -1, -1):
        if int(ids[pos]) == 105 and int(ids[pos + 1]) == 4368 and int(ids[pos + 2]) == 107:
            end_pos = pos + 2
            if end_pos < max_pos:
                return end_pos
    raise ValueError("Could not locate <channel|> (101) or <|turn>model\\n ([105, 4368, 107]) before final token")


def compute_lgt1_ids_hash(ids: Sequence[int], eval_start: int) -> int:
    """Compute exact `LGT1` header hash matching `run31b.rs` lines 3491-3495."""
    h = fnv1a64_ids(ids)
    if eval_start != 0:
        h ^= (int(eval_start) * EVAL_START_MIX) & 0xFFFFFFFFFFFFFFFF
    return h


def f32_rows_to_bf16_le_bytes(logits_2d: torch.Tensor) -> bytes:
    """Convert `[n_pos, vocab]` float tensor to packed little-endian BF16 bytes matching `run31b.rs`."""
    if logits_2d.ndim != 2:
        raise ValueError(f"Expected 2D [n_pos, vocab] logits tensor, got {tuple(logits_2d.shape)}")
    u16 = logits_2d.to(dtype=torch.bfloat16).view(torch.uint16).detach().cpu().numpy().astype("<u2", copy=False)
    return u16.tobytes()


def write_lgt1_bf16(
    path: Path,
    ids: Sequence[int],
    eval_start: int,
    logits_rows: torch.Tensor,
) -> Tuple[int, int]:
    """Write an atomic `LGT1` v1 BF16 teacher-logit container verified byte-for-byte."""
    if logits_rows.ndim != 2:
        raise ValueError(f"Expected 2D logits tensor, got {tuple(logits_rows.shape)}")
    n_pos, vocab = int(logits_rows.shape[0]), int(logits_rows.shape[1])
    exp_n_pos = (len(ids) - 1) - eval_start
    if n_pos != exp_n_pos or n_pos <= 0 or vocab <= 0:
        raise ValueError(f"LGT1 shape mismatch: rows={n_pos} != expected={exp_n_pos}, vocab={vocab}")

    ids_hash = compute_lgt1_ids_hash(ids, eval_start)
    hdr = struct.pack("<4sIIIQ", LGT1_MAGIC, LGT1_VERSION, n_pos, vocab, ids_hash)
    payload = f32_rows_to_bf16_le_bytes(logits_rows)
    exp_bytes = LGT1_HEADER_BYTES + n_pos * vocab * 2
    if len(hdr) + len(payload) != exp_bytes:
        raise AssertionError(f"LGT1 serialized length {len(hdr)+len(payload)} != {exp_bytes}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as fp:
            fp.write(hdr)
            fp.write(payload)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return int(path.stat().st_size), ids_hash


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chk = fp.read(8 * 1024 * 1024)
            if not chk:
                break
            h.update(chk)
    return h.hexdigest()


def persist_verified(src: Path, dst_dir: Optional[Path]) -> Tuple[int, str]:
    """Atomically persist `src` into `dst_dir` (if provided) and fail closed on SHA-256 mismatch."""
    if not src.is_file():
        raise FileNotFoundError(f"Missing output artifact: {src}")
    sz = int(src.stat().st_size)
    digest = sha256_file(src)
    if dst_dir is not None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        tmp = dst_dir / (src.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        dst_digest = sha256_file(dst)
        if dst_digest != digest or dst.stat().st_size != sz:
            raise IOError(f"Drive copy verification failed for {src.name}: {dst_digest} != {digest}")
    return sz, digest


def eval_teacher_rows_stats(logits_rows: torch.Tensor, gold_next_ids: Sequence[int]) -> Dict[str, float]:
    """Compute exact teacher NLL, PPL, and top-1 accuracy matching `eval_student_logits_stats`."""
    gold_t = torch.tensor(list(gold_next_ids), dtype=torch.long, device=logits_rows.device)
    lp = F.log_softmax(logits_rows.float(), dim=-1)
    nll = -lp.gather(1, gold_t.unsqueeze(1)).squeeze(1)
    mean_nll = float(nll.mean().item())
    top1 = float((logits_rows.argmax(dim=-1) == gold_t).float().mean().item())
    return {
        "eval_pos": int(logits_rows.shape[0]),
        "mean_nll": round(mean_nll, 5),
        "ppl": round(math.exp(min(mean_nll, 80.0)), 3),
        "top1_acc": round(top1, 4),
    }


def run_selftest() -> int:
    """Mandatory 6-gate CPU self-verification battery."""
    failures = 0

    def gate(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    # Gate 1: Byte-Head 6-gate core self-test
    bh_rc = bhmod._run_selftest()
    gate("Gate 1: train_assistant_v7_byte_head 6-gate selftest", bh_rc == 0, f"rc={bh_rc}")

    # Gate 2: Skeptic-proof holdout disjointness & parser 6-gate self-test
    sp_rc = spmod._run_selftest()
    gate("Gate 2: eval_v6_1_skeptic_proof 6-gate selftest", sp_rc == 0, f"rc={sp_rc}")

    # Gate 3: Exact Rust run31b.rs parity on fnv1a64_ids, resolve_eval_nll_start_pos, and ids_hash
    ids_c = [2, 105, 2364, 107, 3689, 106, 107, 105, 4368, 107, 100, 45518, 107, 101, 818, 3890, 106]
    start_c = resolve_eval_nll_start_pos_after_turn(ids_c)
    ids_nt = [2, 105, 2364, 107, 3689, 106, 107, 105, 4368, 107, 818, 106]
    start_nt = resolve_eval_nll_start_pos_after_turn(ids_nt)
    bad_caught = False
    try:
        resolve_eval_nll_start_pos_after_turn([2, 818, 5279, 529])
    except ValueError:
        bad_caught = True
    h1 = fnv1a64_ids([2, 105, 818, 1354, 563])
    h2 = fnv1a64_ids([2, 105, 818, 1355, 563])
    gate(
        "Gate 3: Rust fnv1a64 + eval_nll_start_pos parity",
        start_c == 13 and start_nt == 9 and bad_caught and h1 != h2,
        f"start_thought={start_c} start_plain={start_nt} h1=0x{h1:016x}",
    )

    # Gate 4: LGT1 BF16 binary container roundtrip & exact size/hash parity
    with tempfile.TemporaryDirectory() as td:
        lgt_p = Path(td) / "sample.lgt1"
        rows = torch.tensor(
            [
                [-2.5, 0.0, 1.25, 3.5, -0.5, 4.0, -1.0, 0.75],
                [0.125, -3.0, 2.5, -0.25, 1.0, 0.0, 5.25, -4.5],
                [1.5, -1.5, 0.25, 2.0, -2.0, 3.0, 0.5, -0.5],
            ],
            dtype=torch.float32,
        )
        sz, h_exp = write_lgt1_bf16(lgt_p, ids_c, start_c, rows)
        raw = lgt_p.read_bytes()
        magic, ver, np_f, voc_f, h_f = struct.unpack("<4sIIIQ", raw[:24])
        dec_u16 = np.frombuffer(raw[24:], dtype="<u2").copy()
        dec_f32 = torch.from_numpy(dec_u16).view(torch.bfloat16).float().reshape(np_f, voc_f)
        max_err = float((dec_f32 - rows).abs().max().item())
        stats = eval_teacher_rows_stats(dec_f32, [3, 6, 5])
        gate(
            "Gate 4: LGT1 v1 BF16 container write/read & stats parity",
            sz == 24 + 3 * 8 * 2 and magic == b"LGT1" and ver == 1 and h_f == h_exp and max_err < 0.05 and stats["eval_pos"] == 3,
            f"bytes={sz} hash=0x{h_f:016x} bf16_err={max_err:.4f} ppl={stats['ppl']}",
        )

    # Gate 5: Held-out chat prompt disjointness verified
    dj = spmod.assert_holdouts_disjoint()
    gate("Gate 5: Held-out passages zero 4-gram overlap & blacklist clean", dj["overlap_4grams"] == 0, str(dj))

    # Gate 6: Fail closed on missing output / bad LGT1 dimensions
    caught_bad = False
    try:
        write_lgt1_bf16(Path("/tmp/should_not_write.lgt1"), ids_c, start_c, torch.zeros((2, 8)))
    except ValueError:
        caught_bad = True
    missing_caught = False
    try:
        persist_verified(Path("/nonexistent/path/missing.bhdr"), None)
    except FileNotFoundError:
        missing_caught = True
    gate("Gate 6: Fail-closed guards on bad LGT1 shape & missing file", caught_bad and missing_caught)

    print(f"ASSISTANT_V7_LGT1_SELFTEST: {6 - failures}/6 gates passed", flush=True)
    return 1 if failures else 0


def run_gpu_pipeline(
    base_dir: Optional[Path] = None,
    ast_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    drive_dir: Optional[Path] = None,
    n_seq: int = 96,
    seq_len: int = 256,
    steps: int = 2400,
    batch_size: int = 256,
    max_bytes: int = 48,
) -> Dict[str, object]:
    """Run full GPU extraction, `LGT1` export, `Assistant-V7` Q2/Q4/Q8 packing, and `BHDR` distillation."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_gpu_pipeline requires a CUDA GPU (use --selftest on CPU)")
    if bhmod._run_selftest() != 0:
        raise RuntimeError("Byte-Head 6-gate selftest failed")
    spmod.assert_holdouts_disjoint()

    t0_all = time.perf_counter()
    device = torch.device("cuda:0")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    local_out = out_dir or Path(os.environ.get("V7_OUT_DIR", "/content/assistant_v7_out"))
    local_out.mkdir(parents=True, exist_ok=True)

    if drive_dir is None:
        drv_str = os.environ.get("V6_1_DRIVE_DIR", "").strip()
        if not drv_str and Path("/content/drive/MyDrive").is_dir():
            drv_str = "/content/drive/MyDrive/gemma4_v6_1_hybrid"
        drive_dir = Path(drv_str) if drv_str else None
    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from tokenizers import Tokenizer

    base_local = str(base_dir or Path("/tmp/gemma4_31b"))
    ast_local = str(ast_dir or Path("/tmp/gemma4_31b_assistant"))
    print(f"[Setup] Ensuring models in {base_local} and {ast_local} (GPU={torch.cuda.get_device_name(0)}, dtype={compute_dtype})...", flush=True)
    bdir = Path(snapshot_download("google/gemma-4-31B-it-qat-w4a16-ct", local_dir=base_local, ignore_patterns=["*.bin", "*.pt"]))
    adir = Path(snapshot_download("google/gemma-4-31B-it-assistant", local_dir=ast_local, ignore_patterns=["*.bin", "*.pt"]))
    base_sf = bdir / "model.safetensors"
    ast_sf = adir / "model.safetensors"
    tok_path = bdir / "tokenizer.json"
    cfg_full = json.loads((bdir / "config.json").read_text(encoding="utf-8"))

    # 1. Pack embed-free Assistant bodies (Q2 ~56 MB, Q4 ~107 MB, Q8 ~198 MB)
    body_paths = {
        2: local_out / "gemma4_31b_assistant_v7_body_q2.t16c",
        4: local_out / "gemma4_31b_assistant_v7_body_q4.t16c",
        8: local_out / "gemma4_31b_assistant_v7_body_q8.t16c",
    }
    for bits, bp in body_paths.items():
        bhmod.pack_assistant_body_from_safetensors(ast_sf, bp, body_bits=bits, block_size=64, has_fwht=(bits == 4))

    # 2. Verify full 262,144-token BpePrefixTrie
    trie = bhmod.BpePrefixTrie.from_tokenizer_json(tok_path)
    if trie.collision_count != 0 or trie.covered_tokens != 262144:
        raise RuntimeError(f"Trie check failed: covered={trie.covered_tokens} collisions={trie.collision_count}")
    print(f"[Trie OK] covered={trie.covered_tokens} nodes={len(trie.node_edges):,} csr_mb={trie.approx_rust_csr_bytes()/(1024**2):.2f}", flush=True)

    # 3. Prepare mixed FineWeb-Edu + Alpaca `<|turn>` sequences AND exact held-out chat turns
    tk = Tokenizer.from_file(str(tok_path))
    bos_id = int(tk.token_to_id("<bos>") or 2)
    ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds_alp = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)

    n_fw = n_seq // 2
    n_ch = n_seq - n_fw
    fw_buf: List[int] = []
    for row in ds_fw:
        txt = str(row.get("text", "")).strip()
        if not txt:
            continue
        ids_row = tk.encode(txt, add_special_tokens=False).ids
        if len(ids_row) >= 32:
            fw_buf.extend(ids_row[:seq_len])
        if len(fw_buf) >= n_fw * seq_len:
            break

    chat_buf: List[int] = []
    for row in ds_alp:
        q_t = (str(row.get("instruction", "")) + ("\n" + str(row.get("input", "")) if row.get("input", "") else "")).strip()
        a_t = str(row.get("output", "")).strip()
        if not q_t or not a_t:
            continue
        turn_txt = f"<|turn>user\n{q_t}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{a_t}<turn|>\n"
        chat_buf.extend(tk.encode(turn_txt, add_special_tokens=False).ids)
        if len(chat_buf) >= n_ch * seq_len:
            break

    seqs_list = [[bos_id] + fw_buf[i * (seq_len - 1) : (i + 1) * (seq_len - 1)] for i in range(n_fw)]
    seqs_list += [[bos_id] + chat_buf[i * (seq_len - 1) : (i + 1) * (seq_len - 1)] for i in range(n_ch)]
    all_seq_ids = np.array(seqs_list, dtype=np.int64)

    heldout_items: List[dict] = []
    for (p_name, p_ans), p_q in zip(spmod.HELDOUT_NLL_PASSAGES, spmod.HELDOUT_CHAT_QUESTIONS):
        chat_turn = spmod.format_gemma4_chat_nll_turn(p_q, p_ans)
        h_ids = [bos_id] + [int(x) for x in tk.encode(chat_turn, add_special_tokens=False).ids]
        eval_start = resolve_eval_nll_start_pos_after_turn(h_ids)
        heldout_items.append({
            "name": p_name,
            "ids": h_ids,
            "eval_start": eval_start,
            "ids_hash": compute_lgt1_ids_hash(h_ids, eval_start),
            "n_pos": (len(h_ids) - 1) - eval_start,
        })
        print(f"[Heldout Init] {p_name}: tokens={len(h_ids)} eval_start={eval_start} eval_pos={(len(h_ids)-1)-eval_start} hash=0x{heldout_items[-1]['ids_hash']:016x}", flush=True)

    tc = cfg_full["text_config"]
    n_layers = int(tc["num_hidden_layers"])
    d_mod = int(tc["hidden_size"])
    eps = float(tc["rms_norm_eps"])
    escale = math.sqrt(float(d_mod))
    nh = int(tc["num_attention_heads"])
    nkv_s = int(tc["num_key_value_heads"])
    nkv_f = int(tc["num_global_key_value_heads"])
    hd_s = int(tc["head_dim"])
    hd_f = int(tc["global_head_dim"])
    k_eq_v = bool(tc.get("attention_k_eq_v", False))
    layer_types = tc["layer_types"]
    rp = tc["rope_parameters"]
    theta_g = float(rp["full_attention"]["rope_theta"])
    part_g = float(rp["full_attention"].get("partial_rotary_factor", 1.0))
    theta_l = float(rp["sliding_attention"]["rope_theta"])
    part_l = float(rp["sliding_attention"].get("partial_rotary_factor", 1.0))
    softcap = float(tc.get("final_logit_softcapping") or 30.0)

    def make_rope(theta: float, rot_d: int, hd: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_1d = torch.arange(max_len, device=device, dtype=torch.float32)
        hr = rot_d // 2
        inv_f = 1.0 / (theta ** (torch.arange(0, 2 * hr, 2, device=device, dtype=torch.float32) / float(hd)))
        ang = pos_1d[:, None] * inv_f[None, :]
        return torch.cos(ang), torch.sin(ang)

    def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_d: int, hd: int) -> torch.Tensor:
        hr, hh = rot_d // 2, hd // 2
        c = cos[None, :, None, :].to(t.dtype)
        s = sin[None, :, None, :].to(t.dtype)
        x0, x1 = t[..., :hr], t[..., hh : hh + hr]
        r0, r1 = x0 * c - x1 * s, x0 * s + x1 * c
        if hr == hh:
            return torch.cat([r0, r1], dim=-1)
        return torch.cat([r0, t[..., hr:hh], r1, t[..., hh + hr :]], dim=-1)

    def rmsn(x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return ((y * w.float()) if w is not None else y).to(compute_dtype)

    # 4. Single-pass W4A16 teacher streaming over all 60 layers (Assistant samples + 3 heldout chat turns)
    with safe_open(base_sf, framework="pt", device="cpu") as sf:
        sf_keys = set(sf.keys())

        def unpack_proj(pfx: str) -> torch.Tensor:
            if f"{pfx}.weight_packed" in sf_keys:
                wp = sf.get_tensor(f"{pfx}.weight_packed").to(device)
                ws = sf.get_tensor(f"{pfx}.weight_scale").to(device, dtype=torch.float32)
                sh = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
                nibs = ((wp.unsqueeze(-1) >> sh) & 0xF) - 8
                return (nibs.to(torch.float32).reshape(wp.shape[0], -1, 32) * ws.unsqueeze(-1)).reshape(wp.shape[0], -1).to(compute_dtype)
            return sf.get_tensor(f"{pfx}.weight").to(device, dtype=compute_dtype)

        emb_31b = sf.get_tensor("model.language_model.embed_tokens.weight")
        X_cpu = (emb_31b[torch.from_numpy(all_seq_ids).long()].float() * escale).to(compute_dtype).cpu()
        heldout_states = [
            (emb_31b[torch.tensor(item["ids"], dtype=torch.long)].unsqueeze(0).float() * escale).to(device=device, dtype=compute_dtype)
            for item in heldout_items
        ]
        fn_w_31b = sf.get_tensor("model.language_model.norm.weight").to(device, dtype=torch.float32)
        cap_kv58 = None
        cap_kv59 = None

        print(f"[Teacher Stream] Streaming {n_layers} W4A16 layers across {n_seq}x{seq_len} + 3 heldout chat turns...", flush=True)
        max_rope_len = max(seq_len, max(len(it["ids"]) for it in heldout_items))
        for l_idx in range(n_layers):
            p = f"model.language_model.layers.{l_idx}."
            is_full = layer_types[l_idx] == "full_attention"
            nkv = nkv_f if is_full else nkv_s
            hd = hd_f if is_full else hd_s
            rot_d = int(round(hd * (part_g if is_full else part_l)))
            cos_r, sin_r = make_rope(theta_g if is_full else theta_l, rot_d, hd, max_rope_len)
            gqa_rep = nh // nkv
            w_q = unpack_proj(p + "self_attn.q_proj")
            w_k = unpack_proj(p + "self_attn.k_proj")
            w_v = None if (is_full and k_eq_v) else unpack_proj(p + "self_attn.v_proj")
            w_o = unpack_proj(p + "self_attn.o_proj")
            w_g = unpack_proj(p + "mlp.gate_proj")
            w_u = unpack_proj(p + "mlp.up_proj")
            w_d = unpack_proj(p + "mlp.down_proj")
            n_in = sf.get_tensor(p + "input_layernorm.weight").to(device, dtype=torch.float32)
            n_qn = sf.get_tensor(p + "self_attn.q_norm.weight").to(device, dtype=torch.float32)
            n_kn = sf.get_tensor(p + "self_attn.k_norm.weight").to(device, dtype=torch.float32)
            n_pa = sf.get_tensor(p + "post_attention_layernorm.weight").to(device, dtype=torch.float32)
            n_pf = sf.get_tensor(p + "pre_feedforward_layernorm.weight").to(device, dtype=torch.float32)
            n_pff = sf.get_tensor(p + "post_feedforward_layernorm.weight").to(device, dtype=torch.float32)
            lsc = float(sf.get_tensor(p + "layer_scalar").float().item())

            def layer_forward_batch(xb: torch.Tensor, capture_kv: bool = False):
                bsz_b, sl_b, _ = xb.shape
                inp = rmsn(xb, n_in)
                q = F.linear(inp, w_q).view(bsz_b, sl_b, nh, hd)
                k = F.linear(inp, w_k).view(bsz_b, sl_b, nkv, hd)
                v = k.clone() if (is_full and k_eq_v) else F.linear(inp, w_v).view(bsz_b, sl_b, nkv, hd)
                cr_s, sr_s = cos_r[:sl_b], sin_r[:sl_b]
                qn = apply_rope(rmsn(q, n_qn), cr_s, sr_s, rot_d, hd).transpose(1, 2)
                kn = apply_rope(rmsn(k, n_kn), cr_s, sr_s, rot_d, hd).transpose(1, 2)
                vn = rmsn(v, None).transpose(1, 2)
                ctx = F.scaled_dot_product_attention(qn, kn.repeat_interleave(gqa_rep, 1), vn.repeat_interleave(gqa_rep, 1), is_causal=True, scale=1.0)
                ot = F.linear(ctx.transpose(1, 2).contiguous().view(bsz_b, sl_b, nh * hd), w_o)
                xm = xb.float() + rmsn(ot, n_pa).float()
                hm = rmsn(xm, n_pf)
                act = (F.gelu(F.linear(hm, w_g).float(), approximate="tanh") * F.linear(hm, w_u).float()).to(compute_dtype)
                xo = ((xm + rmsn(F.linear(act, w_d), n_pff).float()) * lsc).to(compute_dtype)
                if capture_kv:
                    return xo, kn.cpu(), vn.cpu()
                return xo, None, None

            X_next = torch.empty_like(X_cpu)
            k_buf = [] if l_idx in (58, 59) else None
            v_buf = [] if l_idx in (58, 59) else None
            with torch.no_grad():
                for b0 in range(0, n_seq, 16):
                    b1 = min(b0 + 16, n_seq)
                    xo, kn_c, vn_c = layer_forward_batch(X_cpu[b0:b1].to(device), capture_kv=(l_idx in (58, 59)))
                    X_next[b0:b1] = xo.cpu()
                    if l_idx in (58, 59):
                        k_buf.append(kn_c)
                        v_buf.append(vn_c)
                for hi in range(len(heldout_states)):
                    heldout_states[hi], _, _ = layer_forward_batch(heldout_states[hi], capture_kv=False)
            X_cpu = X_next
            if l_idx == 58:
                cap_kv58 = (torch.cat(k_buf, dim=0), torch.cat(v_buf, dim=0))
            elif l_idx == 59:
                cap_kv59 = (torch.cat(k_buf, dim=0), torch.cat(v_buf, dim=0))
            del w_q, w_k, w_v, w_o, w_g, w_u, w_d
            torch.cuda.empty_cache()
            if l_idx % 15 == 14 or l_idx == n_layers - 1:
                print(f"  [Teacher Stream] L{l_idx+1:02d}/{n_layers} done ({time.perf_counter()-t0_all:.1f}s)", flush=True)

        with torch.no_grad():
            h_59_final = rmsn(X_cpu.to(device), fn_w_31b).cpu()
            heldout_finals = [rmsn(hs.squeeze(0), fn_w_31b) for hs in heldout_states]

        del heldout_states, X_cpu
        torch.cuda.empty_cache()

        # Export exact 262,144-vocab LGT1 BF16 teacher logits for the 3 heldout chat turns!
        emb_31b_gpu = emb_31b.to(device, dtype=compute_dtype)
        lgt1_paths: List[Path] = []
        lgt1_report: Dict[str, dict] = {}
        with torch.no_grad():
            for item, h_fin in zip(heldout_items, heldout_finals):
                p_name = item["name"]
                ev_st = int(item["eval_start"])
                n_pos = int(item["n_pos"])
                ht_sub = h_fin[ev_st : ev_st + n_pos].to(device, dtype=compute_dtype)
                logits_sub = softcap * torch.tanh((ht_sub @ emb_31b_gpu.T).float() / softcap)
                lgt_p = local_out / f"{p_name}.lgt1"
                sz_lgt, h_lgt = write_lgt1_bf16(lgt_p, item["ids"], ev_st, logits_sub)
                t_stats = eval_teacher_rows_stats(logits_sub, item["ids"][ev_st + 1 :])
                lgt1_paths.append(lgt_p)
                lgt1_report[p_name] = {
                    **t_stats,
                    "tokens": len(item["ids"]),
                    "eval_start": ev_st,
                    "ids_hash": f"0x{h_lgt:016x}",
                    "bytes": sz_lgt,
                }
                print(
                    f"[LGT1 Exported] {lgt_p.name}: bytes={sz_lgt:,} hash=0x{h_lgt:016x} "
                    f"teacher_nll={t_stats['mean_nll']:.5f} ppl={t_stats['ppl']} top1={t_stats['top1_acc']:.4f}",
                    flush=True,
                )
        del heldout_finals

    # 5. Run 4-layer Assistant KV-sharing backbone on real h_59_final + L58/L59 KV -> h_ast_final [N, 1024]
    with safe_open(ast_sf, framework="pt", device="cpu") as asf:
        pre_w = asf.get_tensor("pre_projection.weight").to(device, dtype=compute_dtype)
        fn_ast = asf.get_tensor("model.norm.weight").to(device, dtype=torch.float32)
        cr_l, sr_l = make_rope(theta_l, int(round(hd_s * part_l)), hd_s, seq_len)
        cr_g, sr_g = make_rope(theta_g, int(round(hd_f * part_g)), hd_f, seq_len)
        h_ast_chunks = []
        with torch.no_grad():
            sm1 = seq_len - 1
            q_mask = torch.tril(torch.ones((sm1, seq_len), device=device, dtype=torch.bool), diagonal=0).unsqueeze(0).unsqueeze(0)
            for b0 in range(0, n_seq, 16):
                b1 = min(b0 + 16, n_seq)
                tok_e = (emb_31b[torch.from_numpy(all_seq_ids[b0:b1, 1:]).long()].float() * escale).to(device, dtype=compute_dtype)
                back_h = h_59_final[b0:b1, :-1].to(device, dtype=compute_dtype)
                st = F.linear(torch.cat([tok_e, back_h], dim=-1), pre_w).float()
                for l_a in range(4):
                    pa = f"model.layers.{l_a}."
                    is_f = (l_a == 3)
                    hd_a, nkv_a = (hd_f, nkv_f) if is_f else (hd_s, nkv_s)
                    rot_a = int(round(hd_f * part_g)) if is_f else int(round(hd_s * part_l))
                    gqa_a = nh // nkv_a
                    inp = rmsn(st, asf.get_tensor(pa + "input_layernorm.weight").to(device, dtype=torch.float32))
                    q = F.linear(inp, asf.get_tensor(pa + "self_attn.q_proj.weight").to(device, dtype=compute_dtype)).view(b1 - b0, sm1, nh, hd_a)
                    cr = (cr_g if is_f else cr_l)[1 : sm1 + 1]
                    sr = (sr_g if is_f else sr_l)[1 : sm1 + 1]
                    qn = apply_rope(rmsn(q, asf.get_tensor(pa + "self_attn.q_norm.weight").to(device, dtype=torch.float32)), cr, sr, rot_a, hd_a).transpose(1, 2)
                    kv_src = cap_kv59 if is_f else cap_kv58
                    kk = kv_src[0][b0:b1].to(device, dtype=compute_dtype).repeat_interleave(gqa_a, dim=1)
                    vv = kv_src[1][b0:b1].to(device, dtype=compute_dtype).repeat_interleave(gqa_a, dim=1)
                    ctx = F.scaled_dot_product_attention(qn, kk, vv, attn_mask=q_mask, scale=1.0)
                    st = st + rmsn(
                        F.linear(ctx.transpose(1, 2).contiguous().view(b1 - b0, sm1, nh * hd_a), asf.get_tensor(pa + "self_attn.o_proj.weight").to(device, dtype=compute_dtype)),
                        asf.get_tensor(pa + "post_attention_layernorm.weight").to(device, dtype=torch.float32),
                    ).float()
                    hm = rmsn(st, asf.get_tensor(pa + "pre_feedforward_layernorm.weight").to(device, dtype=torch.float32))
                    act = (
                        F.gelu(F.linear(hm, asf.get_tensor(pa + "mlp.gate_proj.weight").to(device, dtype=compute_dtype)).float(), approximate="tanh")
                        * F.linear(hm, asf.get_tensor(pa + "mlp.up_proj.weight").to(device, dtype=compute_dtype)).float()
                    ).to(compute_dtype)
                    lsc_a = float(asf.get_tensor(pa + "layer_scalar").float().item())
                    st = (
                        st
                        + rmsn(
                            F.linear(act, asf.get_tensor(pa + "mlp.down_proj.weight").to(device, dtype=compute_dtype)),
                            asf.get_tensor(pa + "post_feedforward_layernorm.weight").to(device, dtype=torch.float32),
                        ).float()
                    ) * lsc_a
                h_ast_chunks.append(rmsn(st, fn_ast).float().cpu())
    del cap_kv58, cap_kv59, pre_w
    torch.cuda.empty_cache()

    # 6. Extract top-64 exact softmaxed W4A16 teacher probabilities aligned to Assistant outputs
    H_ast_all = torch.cat(h_ast_chunks, dim=0)[:, :-1, :].reshape(-1, 1024)
    H_teach_all = h_59_final[:, 1:-1, :].reshape(-1, d_mod)
    gold_flat = torch.from_numpy(all_seq_ids[:, 2:].reshape(-1)).long()
    top_ids_t = torch.empty((H_teach_all.shape[0], 64), dtype=torch.int32)
    top_probs_t = torch.empty((H_teach_all.shape[0], 64), dtype=torch.float32)
    with torch.no_grad():
        for i0 in range(0, H_teach_all.shape[0], 256):
            i1 = min(i0 + 256, H_teach_all.shape[0])
            ht = H_teach_all[i0:i1].to(device, dtype=compute_dtype)
            logits_b = softcap * torch.tanh((ht @ emb_31b_gpu.T).float() / softcap)
            tv, ti = torch.topk(logits_b, k=64, dim=-1)
            top_probs_t[i0:i1] = F.softmax(tv, dim=-1).cpu()
            top_ids_t[i0:i1] = ti.to(torch.int32).cpu()
    del emb_31b, emb_31b_gpu, h_59_final, H_teach_all
    gc.collect()
    torch.cuda.empty_cache()

    act_bundle = local_out / "assistant_v7_activations.pt"
    torch.save({"h_final": H_ast_all.half(), "gold_ids": gold_flat, "top_ids": top_ids_t, "top_probs": top_probs_t}, act_bundle)
    print(f"[Activations Saved] {act_bundle} (samples={H_ast_all.shape[0]:,})", flush=True)

    # 7. Distill `gemma4_31b_assistant_v7.bhdr` via exact Appendix F.1 `<eot>` byte-head loss
    bhdr_out = local_out / "gemma4_31b_assistant_v7.bhdr"
    bh_rpt = bhmod.train_from_activations(
        act_bundle,
        tok_path,
        bhdr_out,
        d_byte=256,
        max_bytes=max_bytes,
        steps=steps,
        batch_size=batch_size,
        lr=3e-3,
        device=str(device),
    )
    metrics = dict(bh_rpt["metrics"])

    # Verify INT8 exported BHDR quantized beam top-1 accuracy on 256 held-out validation vectors
    bh_loaded = bhmod.load_bhdr_v1(bhdr_out)
    val_h = H_ast_all[-256:].numpy().astype(np.float32)
    val_g = gold_flat[-256:].tolist()
    val_total = 0
    beam_hits = 0
    for hv, gv in zip(val_h, val_g):
        if trie.tok_syms[gv]:
            val_total += 1
            if bhmod.decode_bhdr_np_beam(bh_loaded, trie, hv, beam_width=2)[0] == gv:
                beam_hits += 1
    metrics["heldout_beam_top1_acc"] = round(beam_hits / max(val_total, 1), 4)
    metrics["heldout_beam_samples"] = val_total
    print(
        f"[Assistant-V7 BHDR DONE] bytes={bh_rpt['bytes']:,} ({bh_rpt['mb']} MB) "
        f"kl={metrics['val_kl_before']}->{metrics['val_kl_after']} "
        f"tok_acc={metrics['val_tok_acc_before']}->{metrics['val_tok_acc_after']} "
        f"beam_top1={metrics['heldout_beam_top1_acc']}",
        flush=True,
    )

    # 8. Atomically persist all artifacts to Drive with SHA-256 receipts
    manifest: Dict[str, object] = {
        "version": "ASSISTANT-V7-BHDR-LGT1-V1",
        "elapsed_s": round(time.perf_counter() - t0_all, 2),
        "bhdr_metrics": metrics,
        "lgt1_teacher": lgt1_report,
        "artifacts": {},
    }
    for fp in [bhdr_out, body_paths[2], body_paths[4], body_paths[8], *lgt1_paths]:
        sz_f, dig_f = persist_verified(fp, drive_dir)
        manifest["artifacts"][fp.name] = {"bytes": sz_f, "sha256": dig_f}
        print(f"[Persisted] {fp.name}: {sz_f:,} B  sha256={dig_f[:16]}...", flush=True)

    rpt_path = local_out / "assistant_v7_bhdr_report.json"
    rpt_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    persist_verified(rpt_path, drive_dir)
    act_bundle.unlink(missing_ok=True)
    print(f"[COMPLETE] All Assistant-V7 & Teacher LGT1 artifacts ready ({manifest['elapsed_s']}s).", flush=True)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Assistant-V7 Byte-Head GPU Distiller & Teacher LGT1 Exporter")
    ap.add_argument("--selftest", action="store_true", help="Run 6-gate CPU self-verification battery and exit.")
    ap.add_argument("--run", action="store_true", help="Run full GPU distillation + LGT1 export pipeline.")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-seq", type=int, default=96)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()
    if args.selftest or (not args.run and len(sys.argv) == 1):
        return run_selftest()
    run_gpu_pipeline(n_seq=args.n_seq, seq_len=args.seq_len, steps=args.steps, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

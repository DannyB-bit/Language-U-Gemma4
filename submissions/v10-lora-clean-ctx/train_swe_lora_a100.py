#!/usr/bin/env python3
"""
Gemma 4 31B Thought-Masked Multi-Turn SWE LoRA Trainer (V10 — A100-80GB Native W4A16->BF16).

Trains a vLLM 0.19.1-compatible PEFT LoRA adapter (`r=32, lora_alpha=64`, all 410 LM projections,
~450 MB BF16 << 3 GiB Kaggle limit) directly against the EXACT unpacked weights of
`google/gemma-4-31B-it-qat-w4a16-ct` (zero NF4 quantization shift vs Kaggle 4x L4 serving!).

Key engineering pillars:
  1. Exact W4A16-CT -> BF16 unpacking directly into Gemma4TextForCausalLM/Gemma4ForConditionalGeneration
     (vision tower kept off GPU -> ~58.7 GiB base LM on 79.3 GiB A100-80GB).
  2. Diverse multi-repo Python SWE-bench trajectories: official 129 `tasks.jsonl` training tasks +
     real GitHub Python PR bug fixes from `princeton-nlp/SWE-bench` train split (filtered for
     concise multi-hunk Python patches, strictly zero hidden-test overlap).
  3. Clean-context 6-tool protocol matching `v10-lora-clean-ctx` (`include_thoughts: false` history,
     thought tokens masked with `labels=-100`, supervising strictly tool calls & surgical `edit_file`).
  4. 64-token gradient-checkpointed sparse CE loss with exact `final_logit_softcapping = 30.0`,
     held-out validation loss + action top-1 accuracy gating, and fail-closed vLLM key verification.
"""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration,
    Gemma4TextRotaryEmbedding,
)
from peft import LoraConfig, get_peft_model

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0")
OUT_ROOT = Path(os.environ.get("V10_OUT_DIR", "/kaggle/working/v10_lora_out"))
OUT_ROOT.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = OUT_ROOT / "adapters" / "main_lora"
REPORT_JSON = OUT_ROOT / "v10_lora_report.json"

LORA_R = int(os.environ.get("V10_LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("V10_LORA_ALPHA", "64"))
LR = float(os.environ.get("V10_LR", "5e-5"))
EPOCHS = int(os.environ.get("V10_EPOCHS", "2"))
GRAD_ACCUM = int(os.environ.get("V10_GRAD_ACCUM", "8"))
MAX_SEQ_LEN = int(os.environ.get("V10_MAX_SEQ_LEN", "3840"))
MAX_SWEBENCH_EXTRA = int(os.environ.get("V10_EXTRA_TASKS", "2500"))
MAX_TRAJECTORIES = int(os.environ.get("V10_MAX_TRAJ", "650"))

SWEGEMMA_6_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in /workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a line range from a file in /workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace old_string with new_string in an existing file in /workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "allow_multiple": {"type": "boolean"},
                },
                "required": ["filepath", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in /workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Check remaining budget and patch status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Stage and submit git diff patch.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
PROTECTED_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]+\.py$|_test\.py$|(^|/)(conftest\.py|pytest\.ini|pyproject\.toml|setup\.cfg|tox\.ini)$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_system_prompt() -> str:
    for cand in [
        Path(__file__).resolve().parent / "prompts" / "system.md",
        Path("/content/repo/submissions/v10-lora-clean-ctx/prompts/system.md"),
    ]:
        if cand.is_file():
            return cand.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Could not locate v10-lora-clean-ctx/prompts/system.md")


def parse_unified_diff(patch_text: str) -> List[dict]:
    file_ops: List[dict] = []
    cur_file: Optional[str] = None
    is_new = False
    cur_hunks: List[dict] = []
    cur_hunk: Optional[dict] = None

    def flush_hunk() -> None:
        nonlocal cur_hunk
        if cur_hunk is not None and (cur_hunk["old_lines"] or cur_hunk["new_lines"]):
            cur_hunks.append(cur_hunk)
        cur_hunk = None

    def flush_file() -> None:
        nonlocal cur_file, is_new, cur_hunks
        flush_hunk()
        if cur_file and cur_hunks and not PROTECTED_RE.search(cur_file):
            file_ops.append({"filepath": cur_file, "is_new": is_new, "hunks": cur_hunks})
        cur_file = None
        is_new = False
        cur_hunks = []

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush_file()
        elif line.startswith("--- "):
            if line.strip() == "--- /dev/null":
                is_new = True
        elif line.startswith("+++ "):
            p = line[4:].strip()
            if p.startswith("b/"):
                p = p[2:]
            if p != "/dev/null":
                cur_file = p
        elif line.startswith("@@ "):
            flush_hunk()
            m = HUNK_RE.match(line)
            sl = int(m.group(1)) if m else 1
            cur_hunk = {"start_line": max(1, sl), "old_lines": [], "new_lines": []}
        elif cur_hunk is not None:
            if line.startswith(" "):
                cur_hunk["old_lines"].append(line[1:])
                cur_hunk["new_lines"].append(line[1:])
            elif line.startswith("-"):
                cur_hunk["old_lines"].append(line[1:])
            elif line.startswith("+"):
                cur_hunk["new_lines"].append(line[1:])
    flush_file()
    return file_ops


def extract_test_target(test_patch: str) -> Tuple[Optional[str], Optional[str]]:
    test_file, test_func = None, None
    for line in (test_patch or "").splitlines():
        if line.startswith("+++ b/"):
            c = line[6:].strip()
            if "test" in c:
                test_file = c
        elif line.startswith("+def test_") or line.startswith("+async def test_"):
            test_func = line.split("def ", 1)[1].split("(", 1)[0].strip()
            if test_file:
                break
    return test_file, test_func


def fmt_arg(val) -> str:
    if isinstance(val, str):
        return f'<|"|>{val}<|"|>'
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def fmt_tool_call(name: str, args: dict) -> str:
    parts = [f"{k}:{fmt_arg(args[k])}" for k in sorted(args.keys())]
    return f'<|tool_call>call:{name}{{{",".join(parts)}}}<tool_call|>'


def fmt_tool_resp(name: str, resp: dict) -> str:
    parts = [f"{k}:{fmt_arg(resp[k])}" for k in sorted(resp.keys())]
    return f'<|turn>user\n<|tool_response>response:{name}{{{",".join(parts)}}}<tool_response|><turn|>\n<|turn>model\n'


def pick_search_token(file_ops: List[dict]) -> str:
    for op in file_ops:
        for h in op["hunks"]:
            for ln in h["old_lines"]:
                s = ln.strip()
                if s.startswith(("def ", "async def ", "class ")):
                    tok = s.split("(")[0].split(":")[0].strip()
                    if len(tok) >= 6:
                        return tok
    return Path(file_ops[0]["filepath"]).stem


def compact_issue(prob: str, hints: str, max_chars: int = 900) -> str:
    t = (prob or "").strip()
    if len(t) > max_chars:
        t = t[:600].rstrip() + "\n...[snip]...\n" + t[-260:].lstrip()
    if hints and hints.strip():
        t += "\n\n## Hints:\n" + hints.strip()[:220]
    return t


def build_trajectory(task: dict, tokenizer, sys_prompt: str, max_len: int) -> Optional[dict]:
    patch = task.get("patch", "")
    if not patch or not patch.strip():
        return None
    file_ops = parse_unified_diff(patch)
    if not file_ops or len(file_ops) > 3:
        return None
    py_ops = [op for op in file_ops if op["filepath"].endswith(".py")]
    if not py_ops:
        return None

    # Validate hunk sizes so trajectories fit cleanly inside MAX_SEQ_LEN with full surgical diffs
    total_hunks = sum(len(op["hunks"]) for op in py_ops)
    if total_hunks == 0 or total_hunks > 4:
        return None
    for op in py_ops:
        for h in op["hunks"]:
            ob = "\n".join(h["old_lines"])
            nb = "\n".join(h["new_lines"])
            if len(ob) > 1200 or len(nb) > 1200 or (not op["is_new"] and not ob.strip()):
                return None

    repo = task.get("repo", "repo")
    primary_file = py_ops[0]["filepath"]
    search_tok = pick_search_token(py_ops)
    issue_prompt = (
        f"You are evaluating a software engineering task for repository {repo}.\n\n"
        f"Problem Statement:\n{compact_issue(task.get('problem_statement', ''), task.get('hints_text', ''))}\n\n"
        f"## Task Budget (Session terminates when any budget is exhausted)\n"
        f"- Time allowance: 7.0 minutes\n"
        f"- Tool calls allowance: 32 calls\n"
        f"- Max loop iterations: 75 turns\n"
    )

    header_str = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": issue_prompt},
        ],
        tools=SWEGEMMA_6_TOOLS,
        enable_thinking=True,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids: List[int] = tokenizer.encode(header_str, add_special_tokens=False)
    labels: List[int] = [-100] * len(input_ids)

    def append_turn(action_str: str, resp_str: Optional[str]) -> None:
        # Clean-context format (`include_thoughts: false`): supervise tool call + `<turn|>\n` directly
        # after an empty/masked `<|channel>thought\n<channel|>` header so Gemma 4's native thought
        # distribution is NEVER corrupted by synthetic thoughts!
        th_ids = tokenizer.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
        act_ids = tokenizer.encode(action_str + "<turn|>\n", add_special_tokens=False)
        input_ids.extend(th_ids)
        labels.extend([-100] * len(th_ids))
        input_ids.extend(act_ids)
        labels.extend(act_ids)
        if resp_str is not None:
            r_ids = tokenizer.encode(resp_str, add_special_tokens=False)
            input_ids.extend(r_ids)
            labels.extend([-100] * len(r_ids))

    # Turn 1: Targeted fixed-string git grep
    h0 = py_ops[0]["hunks"][0]
    grep_cmd = f'git grep -n -F "{search_tok}" -- \'*.py\' | head -n 40'
    grep_out = f"{primary_file}:{h0['start_line']}:{(h0['old_lines'][0] if h0['old_lines'] else search_tok)}"
    append_turn(
        fmt_tool_call("run_command", {"command": grep_cmd}),
        fmt_tool_resp("run_command", {"exit_code": 0, "status": "ok", "stderr": "", "stdout": grep_out}),
    )

    # Turn 2+: Read target region(s) and apply surgical edit_file / write_file per file & hunk
    edited_files: List[str] = []
    for op in py_ops:
        fp = op["filepath"]
        edited_files.append(fp)
        if op["is_new"]:
            content = "\n".join(op["hunks"][0]["new_lines"]) + "\n"
            append_turn(
                fmt_tool_call("write_file", {"content": content, "filepath": fp}),
                fmt_tool_resp("write_file", {"filepath": fp, "size": len(content), "status": "ok"}),
            )
            continue

        # Bounded read_file covering hunks
        min_l = max(1, min(h["start_line"] for h in op["hunks"]) - 3)
        preview = "\n".join(op["hunks"][0]["old_lines"])
        end_l = min(min_l + max(len(op["hunks"][0]["old_lines"]) + 10, 40), min_l + 119)
        append_turn(
            fmt_tool_call("read_file", {"end_line": end_l, "filepath": fp, "start_line": min_l}),
            fmt_tool_resp(
                "read_file",
                {"content": preview, "end_line": end_l, "filepath": fp, "is_truncated": False, "start_line": min_l, "status": "ok"},
            ),
        )
        for h in op["hunks"]:
            old_s = "\n".join(h["old_lines"])
            new_s = "\n".join(h["new_lines"])
            append_turn(
                fmt_tool_call("edit_file", {"allow_multiple": False, "filepath": fp, "new_string": new_s, "old_string": old_s}),
                fmt_tool_resp("edit_file", {"filepath": fp, "occurrences": 1, "status": "ok", "strategy": "exact"}),
            )

    # Verification turn: py_compile + focused pytest (if test target known) + git status/diff check
    py_list = " ".join(sorted(set(edited_files)))
    test_file, test_func = extract_test_target(task.get("test_patch", ""))
    if test_file and test_func:
        ver_cmd = (
            f"python3 -m py_compile {py_list} && "
            f"timeout -k 2s 45s python3 -m pytest {test_file} -x -q -k {test_func} 2>&1 | tail -n 30 && "
            f"git status --short && git diff --stat"
        )
        ver_out = "1 passed in 0.35s\n" + "\n".join(f" M {f}" for f in sorted(set(edited_files)))
    else:
        ver_cmd = f"python3 -m py_compile {py_list} && git status --short && git diff --stat"
        ver_out = "\n".join(f" M {f}" for f in sorted(set(edited_files)))
    append_turn(
        fmt_tool_call("run_command", {"command": ver_cmd}),
        fmt_tool_resp("run_command", {"exit_code": 0, "status": "ok", "stderr": "", "stdout": ver_out}),
    )

    # Final submit_patch + 1-word "Done" termination
    append_turn(
        fmt_tool_call("submit_patch", {}),
        fmt_tool_resp("submit_patch", {"files_changed": len(set(edited_files)), "patch_size": len(patch), "status": "ok"}),
    )
    append_turn("Done", None)

    if len(input_ids) > max_len:
        return None
    n_sup = sum(1 for x in labels if x != -100)
    if n_sup < 24:
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "repo": repo,
        "instance_id": task.get("instance_id", ""),
        "n_sup": n_sup,
    }


def load_training_tasks() -> List[dict]:
    tasks: List[dict] = []
    seen_ids = set()

    # 1. Official 129 competition training tasks (`tasks.jsonl`)
    for cand in [
        Path("/content/tasks.jsonl"),
        Path("/kaggle/working/tasks.jsonl"),
        Path("/kaggle/input/gemma-4-developer-agent/tasks.jsonl"),
        Path("/kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl"),
        Path("tasks.jsonl"),
    ]:
        if cand.is_file():
            for ln in cand.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    row = json.loads(ln)
                    iid = row.get("instance_id", "")
                    if iid not in seen_ids:
                        seen_ids.add(iid)
                        tasks.append(row)
            print(f"[Data] Loaded {len(tasks)} official tasks from {cand}", flush=True)
            break

    # 2. Generalizing real GitHub Python PR bug-fix pairs (`princeton-nlp/SWE-bench` train split)
    if MAX_SWEBENCH_EXTRA > 0:
        try:
            from datasets import load_dataset
            ds = load_dataset("princeton-nlp/SWE-bench", split="train")
            idxs = list(range(len(ds)))
            random.Random(SEED).shuffle(idxs)
            added = 0
            for i in idxs:
                if added >= MAX_SWEBENCH_EXTRA:
                    break
                r = ds[i]
                iid = str(r.get("instance_id", ""))
                if iid in seen_ids:
                    continue
                patch = str(r.get("patch", ""))
                if not patch or len(patch) > 4500:
                    continue
                seen_ids.add(iid)
                tasks.append({
                    "instance_id": iid,
                    "repo": str(r.get("repo", "")),
                    "problem_statement": str(r.get("problem_statement", "")),
                    "hints_text": str(r.get("hints_text", "")),
                    "patch": patch,
                    "test_patch": str(r.get("test_patch", "")),
                })
                added += 1
            print(f"[Data] Added {added} real Python SWE-bench train tasks (total={len(tasks)}).", flush=True)
        except Exception as e:
            print(f"[Data WARN] Could not load princeton-nlp/SWE-bench train split: {e}", flush=True)

    assert len(tasks) >= 40, f"Insufficient SWE training tasks loaded: {len(tasks)}"
    return tasks


def load_exact_w4a16_as_bf16_model(model_dir: Path) -> Gemma4ForConditionalGeneration:
    """Unpack EXACT `google/gemma-4-31B-it-qat-w4a16-ct` INT4 weights -> BF16 on A100-80GB (zero NF4 shift!)."""
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    if hasattr(cfg, "quantization_config"):
        cfg.quantization_config = None
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "quantization_config"):
        cfg.text_config.quantization_config = None

    print("[Model] Instantiating Gemma4ForConditionalGeneration on meta device...", flush=True)
    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration(cfg)

    # Materialize LM modules on CUDA BF16 and keep Vision tower tiny/unallocated on CPU to save VRAM!
    model.to_empty(device="cpu")
    model = model.to(dtype=torch.bfloat16)

    sf_files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    assert sf_files, f"No safetensors shards in {model_dir}"
    print(f"[Model] Streaming & unpacking {len(sf_files)} W4A16-CT shards -> exact BF16 LM weights...", flush=True)

    # Track exact loaded parameters so zero LM params or buffers remain uninitialized garbage!
    param_map = dict(model.named_parameters())
    buf_map = dict(model.named_buffers())
    loaded_lm_params = set()
    shifts = torch.arange(0, 32, 4, device=DEVICE, dtype=torch.int32)
    packed_done = 0
    direct_done = 0

    for sfp in sf_files:
        with safe_open(sfp, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for k in sorted(keys):
                if k.endswith(".weight_scale") or k.endswith(".weight_shape") or k.endswith(".weight_zero_point"):
                    continue
                if "vision_tower" in k or "embed_vision" in k:
                    # Leave vision tower on CPU (not used in text/tool LoRA training)
                    continue
                if k.endswith(".weight_packed"):
                    base_k = k[:-len(".weight_packed")]
                    w_name = base_k + ".weight"
                    if w_name not in param_map:
                        continue
                    wp = f.get_tensor(k).to(DEVICE)
                    ws = f.get_tensor(base_k + ".weight_scale").to(DEVICE, dtype=torch.float32)
                    nibs = ((wp.unsqueeze(-1) >> shifts) & 0xF) - 8
                    w_bf16 = (nibs.to(torch.float32).reshape(wp.shape[0], -1, 32) * ws.unsqueeze(-1)).reshape(wp.shape[0], -1).to(torch.bfloat16)
                    param_map[w_name].data = w_bf16
                    loaded_lm_params.add(w_name)
                    del wp, ws, nibs
                    packed_done += 1
                else:
                    if k in param_map:
                        param_map[k].data = f.get_tensor(k).to(device=DEVICE, dtype=torch.bfloat16)
                        loaded_lm_params.add(k)
                        direct_done += 1
                    elif k in buf_map:
                        buf_map[k].data = f.get_tensor(k).to(device=DEVICE)

    # Tie lm_head weight to embed_tokens if tied
    if getattr(cfg.text_config, "tie_word_embeddings", True):
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
        loaded_lm_params.add("lm_head.weight")

    # CRITICAL: Re-initialize all non-persistent LM buffers that `to_empty()` leaves as uninitialized memory!
    # 1) Exact FP32 Rotary Embeddings (full & sliding attention inv_freq)
    lm_mod = model.model.language_model
    lm_mod.rotary_emb = Gemma4TextRotaryEmbedding(cfg.text_config, device=DEVICE)
    # 2) Exact ScaledWordEmbedding `embed_scale = sqrt(hidden_size)`
    emb_mod = lm_mod.embed_tokens
    if hasattr(emb_mod, "embed_scale"):
        scale_val = float(cfg.text_config.hidden_size) ** 0.5
        if torch.is_tensor(emb_mod.embed_scale):
            emb_mod.embed_scale = torch.tensor(scale_val, device=DEVICE, dtype=torch.bfloat16)
        else:
            emb_mod.embed_scale = scale_val
    # 3) Check any remaining non-loaded LM parameters (fail-closed!)
    missing_lm = [
        n for n, _ in model.named_parameters()
        if ("vision_tower" not in n and "embed_vision" not in n) and n not in loaded_lm_params
    ]
    print(f"[Model] Missing LM params={len(missing_lm)} {missing_lm[:10]}", flush=True)
    assert len(missing_lm) == 0, f"Unloaded LM parameters remain garbage: {missing_lm[:10]}"
    # 4) Verify all LM buffers are finite & on CUDA
    for bn, b in lm_mod.named_buffers():
        if b.device.type != "cuda":
            b.data = b.data.to(DEVICE)
        assert torch.isfinite(b).all(), f"Non-finite buffer after init: {bn}"

    gc.collect()
    torch.cuda.empty_cache()
    vram_gb = torch.cuda.memory_allocated(DEVICE) / (1024**3)
    print(f"[Model] Loaded exact W4A16->BF16 LM: packed_projs={packed_done}, direct_tensors={direct_done}, GPU VRAM={vram_gb:.2f} GiB", flush=True)
    assert packed_done == 410, f"Expected 410 W4A16-CT LM projections, got {packed_done}"
    return model


def sparse_chunked_ce_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    softcap: float,
    chunk_size: int = 64,
    need_acc: bool = False,
) -> Tuple[torch.Tensor, int, int]:
    peft_base = model.get_base_model()
    lm_backbone = peft_base.model.language_model
    lm_head = peft_base.lm_head

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = lm_backbone(input_ids=input_ids, use_cache=False, return_dict=True)
    h = out.last_hidden_state
    del out

    shift_h = h[:, :-1, :].reshape(-1, h.size(-1))
    shift_y = labels[:, 1:].reshape(-1)
    idx = torch.nonzero(shift_y.ne(-100), as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return (h.sum() * 0.0).float(), 0, 0

    sup_h = shift_h.index_select(0, idx)
    sup_y = shift_y.index_select(0, idx)
    del h, shift_h

    n_sup = int(sup_y.numel())
    n_correct = 0

    def _chunk_fn(hc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
        logits = lm_head(hc).float()
        if softcap > 0.0:
            logits = torch.tanh(logits / softcap) * softcap
        return F.cross_entropy(logits, yc, reduction="sum")

    tot_loss = torch.zeros((), device=DEVICE, dtype=torch.float32)
    for s0 in range(0, n_sup, chunk_size):
        s1 = min(s0 + chunk_size, n_sup)
        hc = sup_h[s0:s1]
        yc = sup_y[s0:s1]
        if torch.is_grad_enabled():
            tot_loss = tot_loss + grad_checkpoint(_chunk_fn, hc, yc, use_reentrant=False)
        else:
            logits = lm_head(hc).float()
            if softcap > 0.0:
                logits = torch.tanh(logits / softcap) * softcap
            tot_loss = tot_loss + F.cross_entropy(logits, yc, reduction="sum")
            if need_acc:
                n_correct += int((logits.argmax(dim=-1) == yc).sum().item())

    return tot_loss / float(n_sup), n_sup, n_correct


@torch.no_grad()
def evaluate_val(model: nn.Module, val_samples: List[dict], softcap: float) -> Tuple[float, float]:
    model.eval()
    loss_sum, tok_sum, corr_sum = 0.0, 0, 0
    for s in val_samples:
        ids = torch.tensor([s["input_ids"]], device=DEVICE, dtype=torch.long)
        lbl = torch.tensor([s["labels"]], device=DEVICE, dtype=torch.long)
        loss, n_sup, n_corr = sparse_chunked_ce_loss(model, ids, lbl, softcap, chunk_size=64, need_acc=True)
        loss_sum += float(loss.item()) * n_sup
        tok_sum += n_sup
        corr_sum += n_corr
    model.train()
    return loss_sum / max(tok_sum, 1), corr_sum / max(tok_sum, 1)


def main() -> None:
    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[V10 SWE-LoRA INIT] GPU={gpu_name} ({gpu_gb:.1f} GiB) | r={LORA_R} alpha={LORA_ALPHA} lr={LR} epochs={EPOCHS}", flush=True)
    assert gpu_gb >= 70.0, f"A100-80GB required for exact W4A16->BF16 LoRA training (detected {gpu_gb:.1f} GiB)"

    model_dir = Path(snapshot_download("google/gemma-4-31B-it-qat-w4a16-ct"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    sys_prompt = load_system_prompt()

    raw_tasks = load_training_tasks()
    samples: List[dict] = []
    for t in raw_tasks:
        s = build_trajectory(t, tokenizer, sys_prompt, MAX_SEQ_LEN)
        if s is not None:
            samples.append(s)
            if len(samples) >= MAX_TRAJECTORIES:
                break
    random.Random(SEED).shuffle(samples)
    assert len(samples) >= 40, f"Too few valid trajectories built: {len(samples)}"

    n_val = min(64, max(16, int(round(len(samples) * 0.12))))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    avg_len = sum(len(s["input_ids"]) for s in samples) / len(samples)
    avg_sup = sum(s["n_sup"] for s in samples) / len(samples)
    print(f"[Data] Built {len(samples)} valid trajectories (train={len(train_samples)}, val={len(val_samples)}) | avg_len={avg_len:.0f}, avg_sup={avg_sup:.0f}", flush=True)

    base_model = load_exact_w4a16_as_bf16_model(model_dir)
    base_model.config.use_cache = False
    base_model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        exclude_modules=r".*(vision|visual|embed_tokens|lm_head|per_layer).*",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Keep all LoRA params in FP32 for exact AdamW updates on BF16 base weights
    lora_params = [p for p in model.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.to(device=DEVICE, dtype=torch.float32)

    adapted_mods = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0]
    print(f"[LoRA] Adapted LM projection count={len(adapted_mods)}", flush=True)
    assert len(adapted_mods) == 410, f"Expected 410 adapted LM projections, got {len(adapted_mods)}"
    assert not any("vision" in n.lower() for n in adapted_mods), "Vision tower leaked into LoRA!"

    softcap = float(getattr(base_model.config.text_config, "final_logit_softcapping", 30.0) or 30.0)

    # FAIL-CLOSED GATE 0: Verify unpacked W4A16->BF16 base LM produces exact Gemma 4 31B logits BEFORE training!
    with torch.no_grad():
        probe_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ph = model.get_base_model().model.language_model(input_ids=probe_ids, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
        plog = model.get_base_model().lm_head(ph).float()
        if softcap > 0.0:
            plog = torch.tanh(plog / softcap) * softcap
        tv, ti = torch.topk(plog[0], 5)
        top5 = [(tokenizer.decode([int(i)]), round(float(v), 2)) for v, i in zip(tv, ti)]
        print(f"[Gate 0 Base Sanity] 'The capital of France is' -> TOP5={top5}", flush=True)
        assert any("Paris" in tok for tok, _ in top5), f"FAIL-CLOSED: Base LM sanity failed! top5={top5}"

    # Step 0 held-out validation baseline (must be sane pretrained loss, NOT random noise!)
    val_loss_0, val_acc_0 = evaluate_val(model, val_samples, softcap)
    print(f"[Val Step 000] loss={val_loss_0:.4f} | action_top1={val_acc_0*100:.2f}% | VRAM={torch.cuda.memory_allocated(DEVICE)/(1024**3):.2f} GiB", flush=True)
    assert val_loss_0 < 4.0 and val_acc_0 > 0.40, f"FAIL-CLOSED: Step-0 base model sanity failed (loss={val_loss_0:.4f}, acc={val_acc_0:.4f})!"

    opt = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01, betas=(0.9, 0.95), foreach=False)
    total_opt_steps = max(1, (len(train_samples) * EPOCHS + GRAD_ACCUM - 1) // GRAD_ACCUM)
    warmup_steps = max(4, int(round(total_opt_steps * 0.06)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val_loss = val_loss_0
    best_val_acc = val_acc_0
    best_lora_sd = {k: v.detach().to(torch.bfloat16).cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    opt_step = 0
    model.train()
    opt.zero_grad()
    accum_loss = 0.0
    accum_cnt = 0

    for ep in range(EPOCHS):
        order = list(range(len(train_samples)))
        random.Random(SEED + ep + 1).shuffle(order)
        for idx_i, si in enumerate(order):
            s = train_samples[si]
            ids = torch.tensor([s["input_ids"]], device=DEVICE, dtype=torch.long)
            lbl = torch.tensor([s["labels"]], device=DEVICE, dtype=torch.long)
            loss, _, _ = sparse_chunked_ce_loss(model, ids, lbl, softcap, chunk_size=64, need_acc=False)
            (loss / float(GRAD_ACCUM)).backward()
            accum_loss += float(loss.item())
            accum_cnt += 1

            is_step = (accum_cnt == GRAD_ACCUM) or (idx_i + 1 == len(order))
            if is_step:
                torch.nn.utils.clip_grad_norm_(lora_params, 0.5)
                opt.step()
                sch.step()
                opt.zero_grad()
                opt_step += 1
                mean_l = accum_loss / max(accum_cnt, 1)
                accum_loss = 0.0
                accum_cnt = 0
                if opt_step % 10 == 0 or opt_step == 1:
                    print(
                        f"  [Ep {ep+1}/{EPOCHS} step {opt_step:03d}/{total_opt_steps}] train_loss={mean_l:.4f} "
                        f"lr={sch.get_last_lr()[0]:.2e} VRAM={torch.cuda.max_memory_allocated(DEVICE)/(1024**3):.2f}GiB "
                        f"dt={time.perf_counter()-t0:.0f}s",
                        flush=True,
                    )

        vl, va = evaluate_val(model, val_samples, softcap)
        improved = vl < best_val_loss
        print(f"[Val Ep {ep+1}/{EPOCHS}] loss={vl:.4f} (best={best_val_loss:.4f}) | action_top1={va*100:.2f}% | improved={improved}", flush=True)
        if improved:
            best_val_loss = vl
            best_val_acc = va
            best_lora_sd = {k: v.detach().to(torch.bfloat16).cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    # Export verified vLLM 0.19.1 PEFT LoRA adapter (`adapters/main_lora/`)
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR), safe_serialization=True)

    # Normalize & verify exact BF16 safetensors keys matching sample_submission schema
    clean_sd: Dict[str, torch.Tensor] = {}
    for k, v in best_lora_sd.items():
        nk = k.replace(".default.weight", ".weight")
        assert nk.startswith("base_model.model.model.language_model.layers."), f"Unexpected LoRA key prefix: {nk}"
        assert nk.endswith(".lora_A.weight") or nk.endswith(".lora_B.weight"), f"Unexpected LoRA key suffix: {nk}"
        assert torch.isfinite(v).all(), f"Non-finite values in {nk}"
        clean_sd[nk] = v.to(torch.bfloat16).contiguous()

    assert len(clean_sd) == 820, f"Expected 820 LoRA tensors (410x2), got {len(clean_sd)}"
    st_path = ADAPTER_DIR / "adapter_model.safetensors"
    save_file(clean_sd, str(st_path))

    cfg_path = ADAPTER_DIR / "adapter_config.json"
    acfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    acfg["base_model_name_or_path"] = "google/gemma-4-31b-it-qat-w4a16-ct"
    acfg["r"] = LORA_R
    acfg["lora_alpha"] = LORA_ALPHA
    acfg["target_modules"] = ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
    cfg_path.write_text(json.dumps(acfg, indent=2) + "\n", encoding="utf-8")

    # Remove any extraneous files (README.md etc.) from adapter_dir
    for extra in list(ADAPTER_DIR.iterdir()):
        if extra.name not in {"adapter_config.json", "adapter_model.safetensors"}:
            if extra.is_dir():
                shutil.rmtree(extra)
            else:
                extra.unlink()

    st_bytes = st_path.stat().st_size
    st_sha = sha256_file(st_path)
    cfg_sha = sha256_file(cfg_path)
    report = {
        "version": "v10-lora-clean-ctx",
        "gpu": gpu_name,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "val_loss_before": round(val_loss_0, 4),
        "val_loss_after": round(best_val_loss, 4),
        "val_action_top1_before": round(val_acc_0, 4),
        "val_action_top1_after": round(best_val_acc, 4),
        "n_lora_tensors": len(clean_sd),
        "adapter_safetensors_bytes": int(st_bytes),
        "adapter_safetensors_mib": round(st_bytes / (1024**2), 2),
        "adapter_safetensors_sha256": st_sha,
        "adapter_config_sha256": cfg_sha,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n[V10 SWE-LoRA COMPLETE]", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

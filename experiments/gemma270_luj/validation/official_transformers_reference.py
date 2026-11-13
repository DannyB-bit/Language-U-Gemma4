#!/usr/bin/env python3
"""OFFICIAL HF Transformers validation ONLY. C++ remains the production runtime.
Requires transformers==4.55.0 plus torch; no network/model download.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

DEFAULT_SHA = "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--ids", required=True, help="Literal comma-separated Gemma token IDs")
    p.add_argument("--logits-out", required=True, type=Path)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--expected-sha", default=DEFAULT_SHA)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.logits_out.exists():
        p.error("refusing to overwrite logits")
    ids = [int(v) for v in args.ids.split(",")]
    if not ids or len(ids) > 64 or any(x < 0 or x >= 262144 for x in ids):
        p.error("invalid model input IDs")
    with args.checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != args.expected_sha:
        p.error("checkpoint SHA256 mismatch")
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise SystemExit("OFFICIAL_TRANSFORMERS_UNAVAILABLE: validation-only "
                         "environment needs transformers==4.55.0 and torch") from e
    if transformers.__version__ != "4.55.0":
        raise SystemExit("unreviewed Transformers version: " + transformers.__version__)
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    with tempfile.TemporaryDirectory(prefix="gemma270_official_") as temp:
        for name, source in (("model.safetensors", args.checkpoint), ("config.json", args.config)):
            (Path(temp) / name).symlink_to(source.resolve())
        model = AutoModelForCausalLM.from_pretrained(temp, torch_dtype=dtype,
                 attn_implementation="eager", local_files_only=True).eval()
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids],dtype=torch.long),
                           logits_to_keep=1, use_cache=False,
                           return_dict=True).logits[0,-1].float().cpu().contiguous()
        if logits.numel() != 262144 or not torch.isfinite(logits).all():
            raise SystemExit("invalid official reference logits")
        logits.numpy().astype("<f4",copy=False).tofile(args.logits_out)
    print(json.dumps({"reference":"official_HF_Transformers","version":transformers.__version__,
                      "weight_sha256":digest,"dtype":args.dtype,"input_token_count":len(ids),
                      "argmax":int(logits.argmax()),"output":str(args.logits_out)}))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run real full-vocabulary text inference and a fail-closed host quality gate.

Reference and candidate runs are separate. Comparisons require identical prompt
tokens and tokenizer hashes. Timing is explicitly host timing, never Pi proof.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neon_runner.checkpoint_io import QATInt4Projections, NativeQATInt4Projections
from neon_runner.lowbit_embedding import ThreeBitTable
from neon_runner.pi4_stream import sha256_file
from neon_runner.text_runtime import GlyphTextRuntime

PROMPTS = ["Who wrote the play Hamlet?", "What is the capital of France?",
           "What is 2 + 2?", "Complete the phrase: The opposite of hot is"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--backend", choices=("glyph", "qat"), default="glyph")
    parser.add_argument("--checkpoint")
    parser.add_argument("--qat-native", action="store_true", help="Use the fused source-model reference (NumPy decoder is the oracle)")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--stream")
    parser.add_argument("--layout")
    parser.add_argument("--embedding-3bit")
    parser.add_argument("--scalars")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", help="Directory produced by an earlier run")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--prompts-json", help="JSON list of prompts; defaults to Hamlet + three checks")
    parser.add_argument("--chat", action="store_true", help="Render the supplied chat template in a sandbox")
    args = parser.parse_args()
    from tokenizers import Tokenizer
    root, out = Path(args.model_dir), Path(args.output)
    if args.tokens < 1 or (args.backend == "qat" and not args.checkpoint):
        parser.error("Positive token count and an explicit QAT checkpoint are required")
    out.mkdir(parents=True, exist_ok=False)
    tokenizer_path = root/"tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cfg = json.loads((root/"config.json").read_text()).get("text_config")
    prompts = PROMPTS if not args.prompts_json else json.loads(Path(args.prompts_json).read_text())
    if not isinstance(prompts, list) or len(prompts) < 4 or not all(isinstance(p, str) and p for p in prompts):
        raise ValueError("At least four nonempty real prompts are required")
    backend_class = NativeQATInt4Projections if args.qat_native else QATInt4Projections
    backend = backend_class(args.checkpoint) if args.backend == "qat" else None
    embedding = ThreeBitTable(args.embedding_3bit) if args.embedding_3bit else None
    runtime = GlyphTextRuntime(root, stream=args.stream, layout=args.layout, scalars=args.scalars,
                               projection_backend=backend, embedding_table=embedding)
    reference = json.loads((Path(args.reference)/"report.json").read_text()) if args.reference else None
    tokenizer_sha = sha256_file(tokenizer_path)
    if reference and (reference["tokenizer_sha256"] != tokenizer_sha or reference["prompt_style"] != ("chat" if args.chat else "raw")):
        raise ValueError("Reference tokenizer/style mismatch")
    reports = []
    try:
        for index, prompt in enumerate(prompts):
            rendered = prompt
            if args.chat:
                from jinja2.sandbox import ImmutableSandboxedEnvironment
                env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
                def template_error(message): raise ValueError(message)
                env.globals["raise_exception"] = template_error
                rendered = env.from_string((root/"chat_template.jinja").read_text()).render(
                    messages=[dict(role="user", content=prompt)], add_generation_prompt=True,
                    bos_token=tokenizer.id_to_token(cfg["bos_token_id"]), enable_thinking=False, tools=None)
            ids = tokenizer.encode(rendered, add_special_tokens=False).ids
            bos = cfg["bos_token_id"]
            if not ids or ids[0] != bos: ids = [bos]+ids
            runtime.reset()
            start = time.perf_counter()
            def progress(layer):
                if args.progress and (layer % 10 == 9 or layer == 0): print(f"Prompt {index} layer {layer}",flush=True)
            hidden = runtime.forward(ids,progress=progress)
            first = runtime.logits(hidden)
            np.save(out/f"prompt-{index:02d}-logits.npy", first, allow_pickle=False)
            generated, logits = [], first
            for token_number in range(args.tokens):
                token = int(np.argmax(logits)); generated.append(token)
                if token_number+1 < args.tokens:
                    logits = runtime.logits(runtime.forward([token]))
            record = dict(prompt=prompt, input_ids=ids, generated_ids=generated,
                          decoded=tokenizer.decode(generated, skip_special_tokens=False),
                          first_top10=np.argsort(first)[-10:][::-1].tolist(),
                          elapsed_host_seconds=time.perf_counter()-start)
            if reference:
                ctrl = reference["prompts"][index]
                if ctrl["prompt"] != prompt or ctrl["input_ids"] != ids:
                    raise ValueError("Reference prompt/token mismatch")
                ref = np.load(Path(args.reference)/f"prompt-{index:02d}-logits.npy", allow_pickle=False).astype(np.float64)
                candidate = first.astype(np.float64)
                if ref.shape != candidate.shape or not np.isfinite(ref).all():
                    raise ValueError("Invalid reference full-vocabulary logits")
                cosine = float(np.dot(ref, candidate)/(np.linalg.norm(ref)*np.linalg.norm(candidate)))
                record.update(first_logit_cosine=cosine, exact_greedy_match=generated == ctrl["generated_ids"],
                              quality_pass=cosine >= .95 and generated == ctrl["generated_ids"])
            reports.append(record)
            print(json.dumps(record), flush=True)
    finally:
        runtime.close()
    # Byte accounting is independent of virtual address space or mmap laziness.
    stream = Path(args.stream or root/"gemma4_31b_language_u_sandwich_stream.bin")
    embedding_path = Path(args.embedding_3bit or root/"gemma4_31b_embed_tokens.bin")
    files = [embedding_path, root/"gemma4_31b_norms.safetensors", Path(args.scalars or root/"layer_scalars_qat.npy"), root/"config.json"]
    if args.backend == "glyph": files.extend([stream, Path(args.layout or root/"gemma4_31b_tensor_layout.json"), root/"s7_codebook_2048x8.bin"])
    else: files.append(Path(args.checkpoint))
    total = sum(path.stat().st_size for path in files)
    result = dict(schema="language-u-full-text-host-gate-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  host_architecture=platform.machine(), scope="host inference; physical Pi 4 gate NOT verified",
                  backend=args.backend, tokenizer_sha256=tokenizer_sha, prompt_style="chat" if args.chat else "raw",
                  qat_projection_reference="native" if args.qat_native else "numpy",
                  prompts=reports, file_bytes={path.name:path.stat().st_size for path in files},
                  file_sha256={path.name:sha256_file(path) for path in files}, total_weight_file_bytes=total,
                  peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  supplied_spec_tokens=dict(ids=[715,100],decoded=tokenizer.decode([715,100],skip_special_tokens=False),
                                            shakespeare_ids=tokenizer.encode("Shakespeare",add_special_tokens=False).ids),
                  full_vocabulary=True, physical_pi4_verified=False,
                  max_tested_context_tokens=max(len(r["input_ids"])+len(r["generated_ids"])-1 for r in reports),
                  host_quality_pass=all(r.get("quality_pass",False) for r in reports) if reference else None,
                  linear_stream_size_pass=stream.stat().st_size <= 4800000000 if args.backend == "glyph" else False,
                  total_weight_size_pass=total < 5000000000)
    result["source_sha256"] = {name:sha256_file(ROOT/name) for name in
        ("neon_runner/checkpoint_io.py","neon_runner/qat_reference.c","neon_runner/text_runtime.py",
         "neon_runner/lowbit_embedding.py","scripts/run_31b_text_gate.py")}
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    if reference and not result["host_quality_pass"]: return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

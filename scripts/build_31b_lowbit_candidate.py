#!/usr/bin/env python3
"""Build a lossy size-feasible candidate from the actual dequantized QAT model.

This is the uncalibrated all-36B ablation, not the proposed scaled/GPTQ solution.
It retains every linear weight through an explicit binary codec and compresses
the entire tied embedding/head. Always run the text quality gate afterwards.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neon_runner.bridge import encode_matrix_opt5_v2, fwht8_vec_numpy, pack_lanu_header
from neon_runner.checkpoint_io import BF16Table, QATInt4Projections
from neon_runner.lowbit_embedding import encode_table
from neon_runner.pi4_stream import checked_layout, sha256_file
from scripts.recover_31b_parameters import UPSTREAM_REVISION, UPSTREAM_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rotation", choices=("fwht8", "direct"), default="fwht8")
    args = parser.parse_args()
    root, out = Path(args.model_dir), Path(args.output)
    if sha256_file(args.checkpoint) != UPSTREAM_SHA256:
        raise ValueError("Input is not the pinned upstream checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    layout = deepcopy(json.loads((root/"gemma4_31b_tensor_layout.json").read_text()))
    projections = QATInt4Projections(args.checkpoint)
    offset, evidence = 16, []
    stream = out/"candidate.lanu"
    try:
        with (out/"candidate.lanu.partial").open("xb") as destination:
            destination.write(pack_lanu_header(layout["total_macro_glyphs"], 0))
            for p in layout["projections"]:
                count, similarity = 0, 0.
                p.update(stream_byte_offset=offset, bytes_per_glyph=36, codec="OPT5", codec_id=5,
                         tensor_bytes=p["glyphs"]*36, rotation_mode=args.rotation)
                for start in range(0, p["d_out"], 128):
                    values = projections.rows(p, start, min(start+128, p["d_out"]))
                    if args.rotation == "fwht8":
                        values = fwht8_vec_numpy(values.reshape(-1)).reshape(values.shape)
                    payload, stats = encode_matrix_opt5_v2(values, outlier_preserved=False)
                    destination.write(payload)
                    similarity += stats["cosine_similarity"]*values.size; count += values.size
                evidence.append(dict(key=p["full_key"], weights=count,
                                     weighted_chunk_cosine=similarity/count))
                offset += p["tensor_bytes"]
                if len(evidence) % 35 == 0: print(f"Encoded {len(evidence)}/410 projections", flush=True)
        (out/"candidate.lanu.partial").rename(stream)
    finally:
        projections.close()
    layout.update(stream_file=stream.name, stream_size_bytes=offset, rotation_mode=args.rotation,
                  source_checkpoint_sha256=UPSTREAM_SHA256, source_checkpoint_revision=UPSTREAM_REVISION,
                  stream_sha256=sha256_file(stream))
    layout_path = out/"candidate.layout.json"
    layout_path.write_text(json.dumps(layout, indent=2)+"\n")
    checked_layout(stream, layout_path)
    embedding = BF16Table(root/"gemma4_31b_embed_tokens.bin", 262144, 5376)
    try: embed_bytes = encode_table(embedding, out/"embedding.lue3")
    finally: embedding.close()
    report = dict(schema="language-u-lowbit-ablation-v1", source_checkpoint_sha256=UPSTREAM_SHA256,
                  source_checkpoint_revision=UPSTREAM_REVISION, linear_stream_bytes=offset,
                  embedding_bytes=embed_bytes, projections=evidence,
                  calibration=False, gptq=False, quality_verified=False, physical_pi4_verified=False,
                  scope="uncalibrated size-feasible ablation; run quality gate before optimization")
    (out/"build-report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key != "projections"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare real glyph rows with the source QAT weights, including their scales.

This tests weight lineage, not text quality. Both supported rotation modes are
measured; it never selects a mode by the desired answer to a prompt.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neon_runner.bridge import NeonRunnerBridgeV2, encode_matrix_opt5_v2
from neon_runner.checkpoint_io import QATInt4Projections
from neon_runner.pi4_stream import sha256_file
from scripts.recover_31b_parameters import UPSTREAM_REPOSITORY, UPSTREAM_REVISION, UPSTREAM_SHA256


def compare(checkpoint, directory):
    directory = Path(directory)
    if sha256_file(checkpoint) != UPSTREAM_SHA256:
        raise ValueError("Upstream checkpoint hash mismatch")
    original = QATInt4Projections(checkpoint)
    stream = directory/"gemma4_31b_language_u_sandwich_stream.bin"
    layout = directory/"gemma4_31b_tensor_layout.json"
    codebook = directory/"s7_codebook_2048x8.bin"
    results = []
    try:
        with NeonRunnerBridgeV2(str(stream), str(layout), codebook_family="auto", codebook_path=str(codebook)) as bridge:
            for p in bridge.layout["projections"]:
                rows = min(4, p["d_out"])
                record = dict(key=p["full_key"], codec=p["codec"], sampled_rows=rows, row_start=0)
                source = original.rows(p, 0, rows).astype(np.float64).reshape(-1)
                count = rows*p["d_in"]//256
                offset = p["stream_byte_offset"]
                raw = bytes(bridge._memmap[offset:offset+count*p["bytes_per_glyph"]])
                if p["codec_id"] == 5:
                    packed = original.checkpoint.tensor(p["full_key"])[:rows].view(np.uint8)
                    codes = np.stack((packed & 15, packed >> 4),axis=-1).reshape(rows,p["d_in"]).astype(np.float32)-8
                    unscaled, _ = encode_matrix_opt5_v2(codes, outlier_preserved=p["bytes_per_glyph"] == 48)
                    a = np.frombuffer(raw,dtype=np.uint8).reshape(-1,p["bytes_per_glyph"])
                    b = np.frombuffer(unscaled,dtype=np.uint8).reshape(a.shape)
                    record["unscaled_integer_reencode"] = dict(sampled_records=len(a),
                        exact_scale_bias_mask_records=int(np.all(a[:,:36] == b[:,:36],axis=1).sum()),
                        note="Only the first 36 bytes are compared; outlier selection is not asserted equal")
                for rotation in (False, True):
                    if p["codec_id"] == 4:
                        decoded = bridge.accelerator.decode(raw, bridge.codebook.array, apply_fwht8=rotation)
                    else:
                        decoded = bridge.accelerator.decode_opt5(raw, p["bytes_per_glyph"], apply_fwht8=rotation)
                    decoded = decoded.astype(np.float64)
                    ns, nd = np.linalg.norm(source), np.linalg.norm(decoded)
                    cosine = float(np.dot(source, decoded)/(ns*nd)) if ns*nd else None
                    record["fwht8" if rotation else "direct"] = dict(cosine=cosine,
                        relative_l2_error=float(np.linalg.norm(source-decoded)/ns) if ns else None,
                        norm_ratio=float(nd/ns) if ns else None)
                results.append(record)
    finally:
        original.close()
    summaries = {}
    for codec in ("OPT4", "OPT5"):
        subset = [r for r in results if r["codec"] == codec]
        summaries[codec] = dict(projections=len(subset))
        if codec == "OPT5":
            summaries[codec]["unscaled_integer_reencode"] = dict(
                sampled_records=sum(r["unscaled_integer_reencode"]["sampled_records"] for r in subset),
                exact_scale_bias_mask_records=sum(r["unscaled_integer_reencode"]["exact_scale_bias_mask_records"] for r in subset))
        for rotation in ("direct", "fwht8"):
            cosine = [r[rotation]["cosine"] for r in subset if r[rotation]["cosine"] is not None]
            summaries[codec][rotation] = dict(min_cosine=min(cosine), median_cosine=float(np.median(cosine)), max_cosine=max(cosine))
    return dict(schema="language-u-source-weight-lineage-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        upstream_repository=UPSTREAM_REPOSITORY, upstream_revision=UPSTREAM_REVISION,
        upstream_full_sha256=UPSTREAM_SHA256, glyph_stream_sha256=sha256_file(stream),
        layout_sha256=sha256_file(layout), codebook_sha256=sha256_file(codebook),
        scope="first four real rows of each projection; not a full elementwise comparison or generation proof",
        summary=summaries, comparisons=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    report = compare(args.checkpoint, args.model_dir)
    Path(args.report).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["summary"],indent=2))


if __name__ == "__main__": main()

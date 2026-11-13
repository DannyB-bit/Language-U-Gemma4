#!/usr/bin/env python3
"""Recover actual layer scalars from the pinned public QAT checkpoint.

The glyph package omits these 60 parameters. Ones are not an equivalent
substitute. Check the entire upstream file SHA and all supplied language norms
before writing a sidecar. The source checkpoint is opened read-only.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neon_runner.checkpoint_io import SafetensorsFile, bf16_to_float32
from neon_runner.pi4_stream import sha256_file

UPSTREAM_REPOSITORY = "google/gemma-4-31B-it-qat-w4a16-ct"
UPSTREAM_REVISION = "52f3f65bc7a02d555763bc923bd1d9094898219d"
UPSTREAM_SHA256 = "1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20"


def recover(checkpoint, directory, output):
    checkpoint, directory, output = Path(checkpoint), Path(directory), Path(output)
    actual = sha256_file(checkpoint)
    if actual != UPSTREAM_SHA256:
        raise ValueError("Upstream checkpoint SHA256 does not match the pinned reference")
    source = SafetensorsFile(checkpoint)
    sidecar = SafetensorsFile(directory/"gemma4_31b_norms.safetensors")
    try:
        matched = 0
        for name, item in sidecar.header.items():
            if not name.startswith("model.language_model."):
                continue
            upstream = source.header.get(name)
            if (upstream is None or upstream["dtype"] != item["dtype"] or upstream["shape"] != item["shape"]
                    or not np.array_equal(source.tensor(name), sidecar.tensor(name))):
                raise ValueError(f"Checkpoint/sidecar lineage mismatch: {name}")
            matched += 1
        if matched != 361:
            raise ValueError("Expected all 361 language norm tensors")
        scalars = []
        for layer in range(60):
            name = f"model.language_model.layers.{layer}.layer_scalar"
            if source.header[name]["dtype"] != "BF16" or source.header[name]["shape"] != [1]:
                raise ValueError("Unexpected layer scalar format")
            scalars.append(bf16_to_float32(source.tensor(name))[0])
        values = np.asarray(scalars, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite layer scalars")
        # Exclusive creation avoids silently replacing a different checkpoint.
        with output.open("xb") as destination:
            np.save(destination, values, allow_pickle=False)
    finally:
        sidecar.close(); source.close()
    return {"schema": "language-u-recovered-qat-scalars-v1",
            "upstream_repository": UPSTREAM_REPOSITORY, "upstream_revision": UPSTREAM_REVISION,
            "upstream_full_sha256": actual, "language_norms_matched": matched,
            "scalar_count": len(values), "scalar_values": values.tolist(),
            "output_sha256": sha256_file(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = recover(args.checkpoint, args.model_dir, args.output)
    Path(args.report).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

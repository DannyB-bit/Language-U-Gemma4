#!/usr/bin/env python3
"""Reproduce sample-only packing evidence, without publishing benchmark timings."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neon_runner.bridge import NeonRunnerBridgeV2, NeonStreamAccelerator
from neon_runner.codebooks import CodebookRegistryV2
from neon_runner.pi4_quantization import mixed_budget
from neon_runner.pi4_stream import (checked_layout, pack_opt4_92, unpack_opt4_92,
                                    repack_stream, redundancy_scan, sha256_file)


def verify():
    acc = NeonStreamAccelerator(auto_build=True)
    if not acc.self_test():
        raise RuntimeError("Native library self-test failed")
    cb = CodebookRegistryV2.resolve("compass_s7").array
    samples = []
    for stem in ("gemma4_e4b_sample_layer0_q_proj.neon_opt4", "gemma4_e4b_sample_neon_hybrid_v2"):
        stream = ROOT / "model" / (stem + ".lanu")
        layout_path = ROOT / "model" / (stem + ".layout.json")
        layout, _ = checked_layout(stream, layout_path)
        with tempfile.TemporaryDirectory() as tmp:
            # Run the requested redundancy scan BEFORE changing the container.
            scan = redundancy_scan(stream, layout_path, Path(tmp) / "unique.db")
            packed = repack_stream(stream, layout_path, Path(tmp) / "packed")
            projections = []
            with NeonRunnerBridgeV2(str(stream), str(layout_path), so_path=acc.so_path) as old, \
                    NeonRunnerBridgeV2(packed["stream"], packed["layout"], so_path=acc.so_path) as new:
                for source_meta, target_meta in zip(layout["projections"], new.layout["projections"]):
                    width = source_meta["bytes_per_glyph"]
                    start = source_meta["stream_byte_offset"]
                    raw = bytes(old._memmap[start:start + source_meta["tensor_bytes"]])
                    rotate = source_meta.get("rotation_mode", layout.get("rotation_mode")) == "fwht8"
                    bitwise = True
                    if source_meta["codec_id"] == 4:
                        encoded = pack_opt4_92(raw)
                        bitwise = unpack_opt4_92(encoded) == raw
                        weights = acc.decode(raw, cb, apply_fwht8=rotate, num_threads=1)
                        decoded = acc.decode_packed_opt4(encoded, cb, apply_fwht8=rotate, num_threads=1)
                        bitwise = bitwise and bool(np.array_equal(weights, decoded))
                    else:
                        weights = acc.decode_opt5(raw, bytes_per_glyph=width, apply_fwht8=rotate)
                    if not bitwise:
                        raise RuntimeError("Native decode bitwise parity failed")
                    # This is a weight-derived software probe, NOT a captured activation.
                    probe = weights.reshape(source_meta["d_out"], source_meta["d_in"])[1].copy()
                    baseline, candidate = old.gemv(source_meta, probe), new.gemv(target_meta, probe)
                    if not np.allclose(baseline, candidate, rtol=2e-5, atol=1e-5):
                        raise RuntimeError("Native GEMV sample parity failed")
                    reference = weights.reshape(source_meta["d_out"], source_meta["d_in"]).astype(np.float64) @ probe
                    reference_error = float(np.max(np.abs(candidate.astype(np.float64) - reference)))
                    projections.append({"key": source_meta["full_key"],
                                        "probe_kind": "decoded sample weight row 1; not model activations",
                                        "native_decode_bitwise_equal": bitwise,
                                        "gemv_max_abs_difference_vs_original_container":
                                            float(np.max(np.abs(baseline - candidate))),
                                        "gemv_max_abs_difference_vs_decoded_fp64": reference_error,
                                        "source_bytes_per_glyph": width,
                                        "packed_bytes_per_glyph": target_meta["bytes_per_glyph"]})
            packed.pop("stream")
            packed.pop("layout")
            samples.append({"source": str(stream.relative_to(ROOT)), "redundancy": scan,
                            "packing": packed, "projection_checks": projections})
    tracked_sources = ["neon_runner/bridge.py", "neon_runner/pi4_stream.py",
                       "neon_runner/pi4_quantization.py", "neon_runner/neon_stream_accelerator_v2.c",
                       "neon_runner/neon_stream_accelerator_v2.h", "scripts/pi4_barrier_break.py",
                       "scripts/verify_pi4_preparation.py", "tests/test_pi4_barrier.py",
                       "neon_runner/native_census.py", "neon_runner/stream_census.c",
                       "scripts/verify_31b_artifact.py"]
    return {"schema": "language-u-pi4-preparation-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Host software correctness on committed E4B sample records only",
            "host_architecture": platform.machine(), "host_os": platform.system(),
            "python": platform.python_version(), "numpy": np.__version__,
            "native_library_sha256": sha256_file(acc.so_path),
            "source_sha256": {path: sha256_file(ROOT / path) for path in tracked_sources},
            "samples": samples, "specified_31b_10_percent_budget": mixed_budget(high_glyphs=11_440_128),
            "gate_1": {"status": "UNVERIFIED",
                       "missing_evidence": ["Validated RAM-fit 31B candidate and complete byte/reserve ledger (full original now audited separately)",
                                            "Captured calibration and held-out model activations",
                                            "Tokenizer-consistent control tokens for Hamlet plus >=3 held-out prompt outputs and logits",
                                            "Pi 4 warm-run resident pages, major faults, read bytes/token, RSS"]},
            "gate_2": {"status": "NOT_RUN_GATE_1_REQUIRED", "pi4_speedup_claim": None},
            "full_model_distribution_preservation": "NOT_ESTABLISHED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="New JSON file; existing evidence is never overwritten")
    args = parser.parse_args()
    report = verify()
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        with Path(args.report).open("x", encoding="utf-8") as f:
            f.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

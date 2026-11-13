#!/usr/bin/env python3
"""Pi 4 preparation tools. Software results are not Pi 4 performance evidence."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neon_runner.pi4_quantization import ChannelStats, mixed_budget
from neon_runner.pi4_stream import repack_stream, redundancy_scan, sha256_file


def calibrate(path, output, tile_hessians=False):
    """Consume an mmap .npy export of actual post-norm/projection inputs."""
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or x.dtype.kind != "f":
        raise ValueError("Expected nonempty floating-point [tokens, channels] .npy")
    if Path(output).suffix != ".npz":
        raise ValueError("Calibration output must end with .npz")
    if tile_hessians and (x.shape[1] % 256 or x.shape[1] > 32768):
        raise ValueError("Tile Hessians require <=32768 channels divisible by 256")
    digest = sha256_file(path)
    stats = ChannelStats.empty(x.shape[1])
    hsum = np.zeros((x.shape[1] // 256, 256, 256)) if tile_hessians else None
    for begin in range(0, x.shape[0], 64):
        batch = np.asarray(x[begin:begin + 64], dtype=np.float64)
        stats.update(batch)
        if hsum is not None:
            for g in range(hsum.shape[0]):
                block = batch[:, g * 256:(g + 1) * 256]
                hsum[g] += block.T @ block
    if sha256_file(path) != digest:
        raise ValueError("Activation dump changed during calibration")
    report = {"source": str(Path(path).resolve()), "source_sha256": digest,
              "samples": stats.samples, "channels": x.shape[1],
              "basis": "as captured; scale and rotate covariance before GPTQ",
              "tile_hessians": tile_hessians, "model_quality_gate": "UNVERIFIED"}
    arrays = {"absmax": stats.absmax, "h_diag": stats.h_diag,
              "samples": np.array(stats.samples), "metadata": np.array(json.dumps(report))}
    if hsum is not None:
        arrays["h_blocks"] = hsum / stats.samples
    with Path(output).open("xb") as f:
        np.savez(f, **arrays)
    report["output"] = str(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    budget = commands.add_parser("budget", help="Exact 36/92-byte integer ledger")
    for option, default in (("glyphs", 114401280), ("stream-budget", 4800000000),
                            ("available-ram", 5000000000), ("runtime-reserve", 200000000),
                            ("sidecar-bytes", 0)):
        budget.add_argument("--" + option, type=int, default=default)
    budget.add_argument("--high-glyphs", type=int)
    scan = commands.add_parser("scan", help="Exact uniqueness and streaming zlib; SQLite uses disk space")
    pack = commands.add_parser("pack", help="Losslessly repack Opt4 while retaining Opt5 records")
    native = commands.add_parser("scan-native", help="Exact 64-bit Linux census with bounded sorting workspace")
    for command in (scan, pack, native):
        command.add_argument("--stream", required=True)
        command.add_argument("--layout", required=True)
    scan.add_argument("--database", required=True, help="New SQLite file; allow stream size plus overhead")
    native.add_argument("--max-workspace", type=int, default=4_000_000_000,
                        help="C working-allocation budget; excludes Python and mmap page residency")
    native.add_argument("--expected-sha256")
    native.add_argument("--report", help="New JSON report; existing evidence is never overwritten")
    pack.add_argument("--output-directory", required=True, help="New directory, never an existing stream")
    pack.add_argument("--chunk-glyphs", type=int, default=1024,
                      help="Bounded offline packing batch; 16384 was used for the full 31B artifact")
    calibration = commands.add_parser("calibrate", help="Summarize captured projection-input .npy files")
    calibration.add_argument("--activations", required=True)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--tile-hessians", action="store_true",
                             help="Also collect full within-tile covariances; larger than diagonal stats")
    args = parser.parse_args()
    if args.command == "budget":
        values = vars(args).copy()
        values.pop("command")
        result = mixed_budget(**values)
    elif args.command == "scan":
        result = redundancy_scan(args.stream, args.layout, args.database)
    elif args.command == "pack":
        result = repack_stream(args.stream, args.layout, args.output_directory, chunk_glyphs=args.chunk_glyphs)
    elif args.command == "scan-native":
        from neon_runner.native_census import native_redundancy_scan
        result = native_redundancy_scan(args.stream, args.layout, max_workspace=args.max_workspace,
                                        expected_sha256=args.expected_sha256)
    else:
        result = calibrate(args.activations, args.output, args.tile_hessians)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if getattr(args, "report", None):
        with Path(args.report).open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))

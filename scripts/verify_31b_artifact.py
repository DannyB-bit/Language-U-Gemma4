#!/usr/bin/env python3
"""Verify the pinned HF 31B artifact and a lossless packed copy on this host.

This does not generate tokens or measure physical Raspberry Pi performance.
It requires already downloaded files; it never executes remote model code.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neon_runner.bridge import NeonRunnerBridgeV2, NeonStreamAccelerator, parse_lanu_header
from neon_runner.codebooks import CodebookRegistryV2
from neon_runner.pi4_stream import checked_layout, sha256_file, unpack_opt4_92

REVISION = "bd9c48f1da41b51521570674ba2696657eb7358e"
REPOSITORY = "TheAiCollectiveART/Glyph-Stream-31b"
EXPECTED = {
    "gemma4_31b_language_u_sandwich_stream.bin": "9465174f3bd782eb549457d51f62605e2ecffc4bdc5fd9b5aad8ea89861cf4c8",
    "gemma4_31b_tensor_layout.json": "5229403e3d93dde3f7719eef16f735b5c1a76611afd9423c75b98af214306bc8",
    "s7_codebook_2048x8.bin": "099cfea75908bf4ca74820c732e040179e5354f6cd023af099faf1d439c37a47",
    "config.json": "b100d85e571c25b688e82919d819253063a1defb0e6cee3b9b1c1fbad99bd0c2",
    "PHYSICAL_PI4_31B_VERIFIED_REPORT.json": "12b79e3923c7941f918a5e2d7fcca4499d11a35b653d1a5d50bd206f3e7d0f5b",
}


def verify(directory, packed_directory, census_path):
    directory, packed_directory = Path(directory), Path(packed_directory)
    hashes = {}
    for name, expected in EXPECTED.items():
        actual = sha256_file(directory / name)
        if actual != expected:
            raise ValueError(f"Pinned artifact hash mismatch: {name}")
        hashes[name] = actual
    stream = directory / "gemma4_31b_language_u_sandwich_stream.bin"
    layout_path = directory / "gemma4_31b_tensor_layout.json"
    layout, header = checked_layout(stream, layout_path)
    packed_stream, packed_layout = packed_directory / "packed.lanu", packed_directory / "packed.layout.json"
    new_layout, new_header = checked_layout(packed_stream, packed_layout)
    parsed = parse_lanu_header(header)
    census = json.loads(Path(census_path).read_text())
    if (census["stream_sha256"] != hashes[stream.name] or census["glyphs"] != parsed.total_glyphs
            or census.get("fingerprint_bits") != 64):
        raise ValueError("Census does not cover the exact pinned full stream")
    by_width = Counter()
    for p in layout["projections"]:
        by_width[p["bytes_per_glyph"]] += p["glyphs"]
    if (len(layout["projections"]) != 410 or parsed.total_glyphs != 114401280
            or parsed.boundary_glyphs != by_width[132]):
        raise ValueError("Full 31B header and projection census disagree")
    predicted = 16 + by_width[132]*92 + by_width[48]*48
    if packed_stream.stat().st_size != predicted:
        raise ValueError("Packed full-model stream size does not match exact ledger")
    packed_sha = sha256_file(packed_stream)
    if (new_layout.get("source_sha256") != hashes[stream.name]
            or new_layout.get("stream_sha256") != packed_sha):
        raise ValueError("Packed artifact provenance or SHA256 mismatch")
    if len(new_layout["projections"]) != len(layout["projections"]):
        raise ValueError("Packed projection count changed")

    # Independently reverse the WHOLE packed stream into a streaming digest.
    # This proves all preserved records as well as all 11-bit containers, without
    # writing a second 7GB reconstructed file or relying on sampled equality.
    reconstruction = hashlib.sha256(header)
    reconstructed_bytes = 16
    with packed_stream.open("rb") as packed_input:
        if packed_input.read(16) != new_header:
            raise ValueError("Packed header changed during verification")
        for p, q in zip(layout["projections"], new_layout["projections"]):
            if p["glyphs"] != q["glyphs"] or p["codec_id"] != q["codec_id"]:
                raise ValueError("Packed glyph/codec census changed")
            for begin in range(0, p["glyphs"], 16384):
                take = min(16384, p["glyphs"]-begin)
                raw = packed_input.read(take*q["bytes_per_glyph"])
                if len(raw) != take*q["bytes_per_glyph"]:
                    raise ValueError("Packed stream truncated during reverse reconstruction")
                if p["codec_id"] == 4 and q["bytes_per_glyph"] == 92:
                    restored = unpack_opt4_92(raw)
                elif p["codec_id"] == 5 and p["bytes_per_glyph"] == q["bytes_per_glyph"]:
                    restored = raw
                else:
                    raise ValueError("Unexpected packed format in pinned 31B reconstruction")
                reconstruction.update(restored)
                reconstructed_bytes += len(restored)
        if packed_input.read(1):
            raise ValueError("Packed stream has unexpected trailing bytes")
    if reconstructed_bytes != stream.stat().st_size or reconstruction.hexdigest() != hashes[stream.name]:
        raise ValueError("Full packed stream does not reconstruct the exact original SHA256")

    codebook_path = directory / "s7_codebook_2048x8.bin"
    codebook = CodebookRegistryV2.resolve("auto", explicit_path=str(codebook_path))
    accelerator = NeonStreamAccelerator(auto_build=True)
    record_checks, glyph_checks, numerical = 0, 0, []
    with NeonRunnerBridgeV2(str(stream), str(layout_path), codebook_family="auto",
                            codebook_path=str(codebook_path), so_path=accelerator.so_path) as old, \
            NeonRunnerBridgeV2(str(packed_stream), str(packed_layout), codebook_family="auto",
                               codebook_path=str(codebook_path), so_path=accelerator.so_path) as new:
        for p, q in zip(layout["projections"], new_layout["projections"]):
            for field in ("full_key", "d_in", "d_out", "glyphs", "codec_id"):
                if p[field] != q[field]:
                    raise ValueError(f"Packed projection metadata changed: {field}")
            width_old, width_new = p["bytes_per_glyph"], q["bytes_per_glyph"]
            for i in sorted({0, p["glyphs"]//2, p["glyphs"]-1}):
                a = p["stream_byte_offset"] + i*width_old
                b = q["stream_byte_offset"] + i*width_new
                original = bytes(old._memmap[a:a+width_old])
                candidate = bytes(new._memmap[b:b+width_new])
                if p["codec_id"] == 4:
                    for rotate in (False, True):
                        reference = accelerator.decode(original, codebook.array, num_threads=1, apply_fwht8=rotate)
                        decoded = accelerator.decode_packed_opt4(candidate, codebook.array,
                                                                num_threads=1, apply_fwht8=rotate)
                        if not np.array_equal(reference, decoded):
                            raise ValueError("Sampled native packed decode differs from original")
                elif original != candidate:
                    raise ValueError("Retained Opt5 record changed")
                glyph_checks += 1
            record_checks += 1

            selected = p["layer"] in (0, 5, 59) and p["tensor_name"] in (
                "self_attn.q_proj.weight_packed", "mlp.down_proj.weight_packed")
            if not selected:
                continue
            # A weight-derived probe provides arithmetic coverage only; it is not
            # a captured inference activation or a semantic generation test.
            row_bytes = (p["d_in"] // 256) * width_old
            begin = p["stream_byte_offset"] + row_bytes
            raw = bytes(old._memmap[begin:begin+row_bytes])
            if p["codec_id"] == 4:
                probe = accelerator.decode(raw, codebook.array, num_threads=1, apply_fwht8=False)
            else:
                probe = accelerator.decode_opt5(raw, bytes_per_glyph=width_old, apply_fwht8=False)
            a, b = old.gemv(p, probe).astype(np.float64), new.gemv(q, probe).astype(np.float64)
            norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or min(norm_a,norm_b) <= 0:
                raise ValueError("Non-finite or empty representative projection")
            relative = float(np.linalg.norm(a-b)) / norm_a
            cosine = float(np.dot(a,b)) / (norm_a*norm_b)
            if relative > 1e-4 or cosine < 0.999999:
                raise ValueError("Representative native GEMV parity gate failed")
            numerical.append({"key":p["full_key"],"relative_l2_difference":relative,
                              "cosine":min(1.0,cosine),"max_abs_difference":float(np.max(np.abs(a-b))),
                              "probe_kind":"decoded weight row 1; not model activations",
                              "rotation_assumption":"direct; upload layout omits rotation_mode"})

    # Embedding and norms sizes come from the pinned Hub LFS metadata/model card.
    # They are explicitly not reported as locally downloaded/hash-verified here.
    embeddings, norms, reserve = 2818572288, 2990082, 200000000
    all36 = 16 + parsed.total_glyphs*36
    ten_percent = all36 + 11440128*(92-36)
    physical = json.loads((directory/"PHYSICAL_PI4_31B_VERIFIED_REPORT.json").read_text())
    sources = ["neon_runner/bridge.py","neon_runner/pi4_stream.py","neon_runner/native_census.py",
               "neon_runner/stream_census.c","neon_runner/neon_stream_accelerator_v2.c",
               "neon_runner/neon_stream_accelerator_v2.h","scripts/verify_31b_artifact.py",
               "tests/test_pi4_barrier.py"]
    return {"schema":"language-u-hf-31b-artifact-audit-v1",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "repository":REPOSITORY,"revision":REVISION,
            "source_url":f"https://huggingface.co/{REPOSITORY}/tree/{REVISION}",
            "host_architecture":platform.machine(),"scope":"artifact integrity and host arithmetic; no token generation",
            "native_library_sha256":sha256_file(accelerator.so_path),
            "verified_local_file_sha256":hashes,
            "source_sha256":{name:sha256_file(ROOT/name) for name in sources},
            "original_header":{"magic":parsed.magic.decode(),"version":parsed.version,
                               "glyphs":parsed.total_glyphs,"boundary_glyphs":parsed.boundary_glyphs},
            "projection_count":record_checks,"sampled_record_checks":glyph_checks,
            "original_stream_bytes":stream.stat().st_size,"packed_stream_bytes":predicted,
            "packed_stream_sha256":packed_sha,"saved_bytes":stream.stat().st_size-predicted,
            "full_reverse_roundtrip":"PASS","full_reverse_reconstructed_bytes":reconstructed_bytes,
            "full_reverse_reconstruction_sha256":reconstruction.hexdigest(),
            "packed_header":{"magic":new_header[:4].decode(),"version":parse_lanu_header(new_header).version},
            "codebook":{"family":codebook.family,"sha256":hashes[codebook_path.name],
                        "repository_sample_codebook_sha256":sha256_file(ROOT/"neon_runner/codebooks/s7_codebook_2048x8.bin")},
            "census":census,"representative_gemv_checks":numerical,
            "memory_ledger":{"existing_embedding_bytes_from_hub_metadata":embeddings,
                             "existing_norm_bytes_from_hub_metadata":norms,
                             "companion_large_files_locally_hash_verified":False,
                             "hypothetical_runtime_reserve_bytes":reserve,
                             "all36_linear_bytes":all36,
                             "all36_plus_existing_embeddings_and_norms_bytes":all36+embeddings+norms,
                             "all36_plus_companions_and_reserve_bytes":all36+embeddings+norms+reserve,
                             "ten_percent_92_linear_bytes":ten_percent,
                             "ten_percent_92_plus_companions_and_reserve_bytes":ten_percent+embeddings+norms+reserve,
                             "target_total_available_bytes":5000000000},
            "uploaded_pi_report":{"status":physical["status"],"timestamp":physical["timestamp"],
                                  "scope":physical["verification_scope_note"],"independently_rerun_on_pi":False},
            "gate_1":{"status":"NOT_PASSED","full_linear_artifact_now_available":True,
                      "blockers":["No validated <=4.8 GB mixed-precision candidate or complete <=5 GB resident runtime",
                                  "Existing embedding table must be included, compressed with quality verification, or explicitly streamed",
                                  "No captured calibration/held-out activations or multi-prompt raw logits",
                                  "No independently measured physical Pi full-generation residency, I/O or latency"]},
            "pi4_speedup_claim":None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory",required=True)
    parser.add_argument("--packed-directory",required=True)
    parser.add_argument("--census",required=True)
    parser.add_argument("--report",required=True,help="New JSON file")
    args=parser.parse_args()
    report=verify(args.model_directory,args.packed_directory,args.census)
    with Path(args.report).open("x",encoding="utf-8") as output:
        json.dump(report,output,indent=2,allow_nan=False); output.write("\n")
    print(json.dumps({"report":args.report,"projection_count":report["projection_count"],
                      "sampled_records":report["sampled_record_checks"],"saved_bytes":report["saved_bytes"],
                      "gate_1":report["gate_1"]["status"]},indent=2))


if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed V8.0 Pi 4 artifact preflight.

The V8 GPU receipt proves artifact production, not that a Pi received a complete
and uncorrupted runtime set. This verifier closes that gap before run31b starts.
It uses only Python's standard library and never loads model tensors into RAM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

BASE = "gemma4_31b_v8_0_base.t16c"
SHARED = "gemma4_31b_v8_0_lod_shared.t16c"
DISTRICTS = [f"gemma4_31b_v8_0_lod_d{i}.t16c" for i in range(12)]
RUNTIME_REQUIRED = [
    BASE,
    SHARED,
    *DISTRICTS,
    "gemma4_31b_v8_0_router.crtr",
    "gemma4_31b_v8_0_c8d.c8d",
    "gemma4_31b_v8_0_pov_6sense_router.drtr",
    "gemma4_31b_v8_0_pov_6sense_manifest.json",
]
SPEC_REQUIRED = [
    "gemma4_31b_v8_0_assistant.bhdr",
    "gemma4_31b_v8_0_assistant_body_q2.t16c",
    "gemma4_31b_v8_0_assistant_body_q4.t16c",
    "gemma4_31b_v8_0_assistant_body_q8.t16c",
]
SHA256_RE = re.compile(r"^([0-9a-f]{64})  ([^\n]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256_manifest(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing V8 checksum manifest: {path}")
    entries: Dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = SHA256_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS line {line_no}: {line!r}")
        digest, name = match.groups()
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name in ("", "."):
            raise ValueError(f"unsafe artifact path on line {line_no}: {name!r}")
        if name in entries:
            raise ValueError(f"duplicate checksum entry: {name}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"empty V8 checksum manifest: {path}")
    return entries


def verify_v8_artifacts(artifact_dir: Path, include_spec: bool = True) -> dict:
    artifact_dir = artifact_dir.resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"V8 artifact directory not found: {artifact_dir}")
    entries = read_sha256_manifest(artifact_dir / "SHA256SUMS.txt")
    required = list(RUNTIME_REQUIRED) + (list(SPEC_REQUIRED) if include_spec else [])
    missing_checksums = [name for name in required if name not in entries]
    if missing_checksums:
        raise ValueError(f"V8 checksum manifest incomplete; missing={missing_checksums}")
    missing_files = [name for name in required if not (artifact_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"V8 runtime set incomplete; missing={missing_files}")

    total_bytes = 0
    verified: List[str] = []
    for name in required:
        path = artifact_dir / name
        actual = sha256_file(path)
        if actual != entries[name]:
            raise ValueError(f"V8 SHA-256 mismatch for {name}: expected={entries[name]} actual={actual}")
        total_bytes += path.stat().st_size
        verified.append(name)
    return {
        "status": "PASS",
        "artifact_dir": str(artifact_dir),
        "files_verified": len(verified),
        "total_bytes": total_bytes,
        "sha256_verified": True,
        "includes_all_spec_bodies": include_spec,
        "files": verified,
    }


def _write_manifest(root: Path, names: Iterable[str]) -> None:
    lines = [f"{sha256_file(root / name)}  {name}" for name in names]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selftest() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        required = RUNTIME_REQUIRED + SPEC_REQUIRED
        for index, name in enumerate(required):
            (root / name).write_bytes(f"v8-test-{index}:{name}\n".encode("utf-8"))
        _write_manifest(root, required)
        report = verify_v8_artifacts(root)
        assert report["files_verified"] == len(required)

        original = (root / DISTRICTS[0]).read_bytes()
        (root / DISTRICTS[0]).write_bytes(original + b"corruption")
        try:
            verify_v8_artifacts(root)
        except ValueError as exc:
            assert "SHA-256 mismatch" in str(exc)
        else:
            raise AssertionError("corrupted district shard was not rejected")
        (root / DISTRICTS[0]).write_bytes(original)

        (root / DISTRICTS[-1]).unlink()
        try:
            verify_v8_artifacts(root)
        except FileNotFoundError as exc:
            assert "incomplete" in str(exc)
        else:
            raise AssertionError("missing district shard was not rejected")

        (root / "SHA256SUMS.txt").write_text("../outside\n", encoding="utf-8")
        try:
            verify_v8_artifacts(root)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed checksum line was not rejected")
    print(f"V8_0_ARTIFACT_SELFTEST: PASS ({len(RUNTIME_REQUIRED) + len(SPEC_REQUIRED)} required files)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--skip-spec", action="store_true", help="validate runtime only; excludes BHDR and Q2/Q4/Q8 assistants")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    if args.artifact_dir is None:
        parser.error("--artifact-dir is required unless --selftest is used")
    report = verify_v8_artifacts(args.artifact_dir, include_spec=not args.skip_spec)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Unified structural precheck (`sniff_container`) for Language-U binary containers.

Supports:
  - `T16C` v1 (`.t16c`): Tile-16 bit-serial & 5-trit ternary containers
  - `LANU` v1/v2 & `LANP` v3 (`.lanu`): Macro-glyph hybrid/packed streams (+ optional `.layout.json`)
  - `LUJ1` (`.luj`): Joint semantic/weight glyph banks (+ optional body SHA-256 verification)
  - Raw headerless glyph blobs (`.glyph.bin`): Option-5 (36/48 B) or Option-4 (92/132 B) records

Streams headers without loading multi-gigabyte payloads into RAM and fails closed
on truncation, overflow, invalid alignment, lying counts, or trailing garbage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union


class ContainerSniffError(ValueError):
    """Raised when a binary container violates format magic, size-law, or bounds checks."""


T16C_MAGIC = b"T16C"
T16C_HEADER_FMT = "<4s5I"
T16C_HEADER_BYTES = struct.calcsize(T16C_HEADER_FMT)
T16C_MAT_HEADER_FMT = "<HIIHBBBBHQQ"
T16C_MAT_HEADER_BYTES = struct.calcsize(T16C_MAT_HEADER_FMT)

LANU_MAGIC = b"LANU"
LANP_MAGIC = b"LANP"
LANU_HEADER_FMT = "<4sHHII"
LANU_HEADER_BYTES = struct.calcsize(LANU_HEADER_FMT)

LUJ1_MAGIC = b"LUJ1"
LUJ1_HEADER_BYTES = 108

_RAW_GLYPH_WIDTHS: Tuple[int, ...] = (48, 36, 132, 92)


@dataclass(frozen=True)
class ContainerSniff:
    """Structured audit summary returned by `sniff_container`."""

    path: str
    format_id: str
    magic: str
    version: int
    file_size_bytes: int
    expected_size_bytes: int
    valid: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# Canonical T16C packing mode table: mode_id -> (n_bits, is_asym)
MODE_TO_T16C = {
    0: (1, True),
    1: (5, False),
    2: (2, False),
    3: (2, True),
    4: (3, False),
}


def tile16_blk_bytes(block_size: int, n_bits: int, is_asym: bool) -> int:
    """Compute exact per-tile-block byte size for Tile-16 packed weights."""
    if block_size not in (64, 128, 256):
        raise ContainerSniffError(f"Unsupported T16C block_size={block_size}")
    if n_bits == 5 and not is_asym:
        n_groups = (block_size + 4) // 5
        return 32 + n_groups * 16
    if 1 <= n_bits <= 4:
        meta_bytes = 64 if is_asym else 32
        return meta_bytes + n_bits * (block_size // 8) * 16
    raise ContainerSniffError(f"Unsupported T16C mode n_bits={n_bits}, is_asym={is_asym}")


def _read_exact(f, n: int, what: str) -> bytes:
    """Read exactly `n` bytes or fail closed with `ContainerSniffError`."""
    data = f.read(n)
    if len(data) != n:
        raise ContainerSniffError(f"Truncated read for {what}: got {len(data)} B, expected {n} B")
    return data


def _sniff_t16c(f, file_size: int, path_str: str) -> ContainerSniff:
    if file_size < T16C_HEADER_BYTES:
        raise ContainerSniffError(f"T16C header truncated ({file_size} B < {T16C_HEADER_BYTES} B)")
    f.seek(0)
    magic, ver, arch_flags, n_norms, n_mats, reserved = struct.unpack(
        T16C_HEADER_FMT, _read_exact(f, T16C_HEADER_BYTES, "T16C header")
    )
    if magic != T16C_MAGIC or ver != 1 or reserved != 0:
        raise ContainerSniffError(f"Invalid T16C header magic={magic!r}, version={ver}, reserved={reserved}")

    pos = T16C_HEADER_BYTES
    for i in range(n_norms):
        if pos + 2 > file_size:
            raise ContainerSniffError(f"Truncated norm[{i}] length prefix at offset {pos}")
        (nlen,) = struct.unpack("<H", _read_exact(f, 2, f"norm[{i}].nlen"))
        pos += 2
        if nlen == 0 or pos + nlen + 4 > file_size:
            raise ContainerSniffError(f"Corrupt/truncated norm[{i}] name (nlen={nlen}) at offset {pos}")
        _ = _read_exact(f, nlen, f"norm[{i}].name").decode("utf-8")
        pos += nlen
        (vlen,) = struct.unpack("<I", _read_exact(f, 4, f"norm[{i}].vlen"))
        pos += 4
        vec_bytes = vlen * 4
        if pos + vec_bytes > file_size:
            raise ContainerSniffError(f"Norm[{i}] vector ({vec_bytes} B) exceeds EOF {file_size}")
        pos += vec_bytes
        f.seek(pos)

    mode_counts: Dict[str, int] = {}
    for i in range(n_mats):
        if pos + T16C_MAT_HEADER_BYTES > file_size:
            raise ContainerSniffError(f"Truncated matrix[{i}] header at offset {pos}")
        nlen, d_out, d_in, bsize, nbits, is_asym, has_fwht, inv_kind, lr_rank, packed_bytes, lr_bytes = (
            struct.unpack(T16C_MAT_HEADER_FMT, _read_exact(f, T16C_MAT_HEADER_BYTES, f"mat[{i}] header"))
        )
        pos += T16C_MAT_HEADER_BYTES
        if nlen == 0 or pos + nlen > file_size:
            raise ContainerSniffError(f"Corrupt/truncated matrix[{i}] name at offset {pos}")
        mname = _read_exact(f, nlen, f"mat[{i}].name").decode("utf-8")
        pos += nlen
        pos = (pos + 63) & ~63
        if pos > file_size:
            raise ContainerSniffError(f"Matrix[{i}] ({mname}) 64B alignment pad exceeds EOF {file_size}")

        if d_out == 0 or d_in == 0 or bsize not in (64, 128, 256) or (d_in % bsize) != 0:
            raise ContainerSniffError(f"Matrix[{i}] ({mname}) invalid geometry d_out={d_out}, d_in={d_in}, bsize={bsize}")
        expected_packed = ((d_out + 15) // 16) * (d_in // bsize) * tile16_blk_bytes(bsize, nbits, bool(is_asym))
        if packed_bytes != expected_packed:
            raise ContainerSniffError(
                f"Matrix[{i}] ({mname}) packed_bytes={packed_bytes} != expected {expected_packed}"
            )
        if inv_kind not in (0, 1) or has_fwht not in (0, 1):
            raise ContainerSniffError(f"Matrix[{i}] ({mname}) invalid flags has_fwht={has_fwht}, inv_kind={inv_kind}")
        inv_bytes = (d_in * 4) if inv_kind != 0 else 0
        expected_lr = (d_out * 4 + lr_rank * 4 + d_out * lr_rank + lr_rank * d_in) if lr_rank > 0 else 0
        if lr_bytes != expected_lr:
            raise ContainerSniffError(f"Matrix[{i}] ({mname}) lr_bytes={lr_bytes} != expected {expected_lr}")

        total_payload = packed_bytes + inv_bytes + lr_bytes
        if pos + total_payload > file_size:
            raise ContainerSniffError(
                f"Matrix[{i}] ({mname}) payload ({pos + total_payload} B) exceeds EOF {file_size}"
            )
        pos += total_payload
        f.seek(pos)
        mode_key = f"b{nbits}_{'asym' if is_asym else 'sym'}_bs{bsize}"
        mode_counts[mode_key] = mode_counts.get(mode_key, 0) + 1

    if pos != file_size:
        raise ContainerSniffError(f"Trailing bytes in T16C file: parsed={pos}, file_size={file_size}")

    return ContainerSniff(
        path=path_str,
        format_id="t16c_v1",
        magic="T16C",
        version=ver,
        file_size_bytes=file_size,
        expected_size_bytes=pos,
        valid=True,
        details={
            "arch_flags": arch_flags,
            "n_norms": n_norms,
            "n_mats": n_mats,
            "mode_counts": mode_counts,
        },
    )


def _infer_layout_path(stream_path: Path, explicit_layout: Optional[Union[str, Path]]) -> Optional[Path]:
    if explicit_layout is not None:
        lp = Path(explicit_layout)
        if not lp.is_file():
            raise ContainerSniffError(f"Explicit layout file not found: {lp}")
        return lp
    for cand in (stream_path.with_suffix(".layout.json"), Path(str(stream_path) + ".layout.json")):
        if cand.is_file():
            return cand
    return None


def _sniff_lanu(
    f,
    file_size: int,
    path_str: str,
    layout_path: Optional[Path] = None,
) -> ContainerSniff:
    if file_size < LANU_HEADER_BYTES:
        raise ContainerSniffError(f"LANU/LANP stream truncated ({file_size} B < {LANU_HEADER_BYTES} B)")
    f.seek(0)
    magic, ver, flags, total_glyphs, boundary_glyphs = struct.unpack(
        LANU_HEADER_FMT, _read_exact(f, LANU_HEADER_BYTES, "LANU/LANP header")
    )
    if (magic, ver) not in ((LANU_MAGIC, 1), (LANU_MAGIC, 2), (LANP_MAGIC, 3)):
        raise ContainerSniffError(f"Invalid stream magic/version: {magic!r}/{ver}")
    if boundary_glyphs > total_glyphs:
        raise ContainerSniffError(f"Lying LANU header: boundary_glyphs ({boundary_glyphs}) > total_glyphs ({total_glyphs})")

    b_width = 92 if magic == LANP_MAGIC else 132
    non_boundary = total_glyphs - boundary_glyphs
    payload_bytes = file_size - LANU_HEADER_BYTES
    expected_boundary_bytes = boundary_glyphs * b_width
    rem_bytes = payload_bytes - expected_boundary_bytes
    if rem_bytes < 0:
        raise ContainerSniffError(
            f"Stream truncated inside boundary region: payload={payload_bytes} B < {expected_boundary_bytes} B"
        )

    if non_boundary == 0:
        if rem_bytes != 0:
            raise ContainerSniffError(
                f"Stream size mismatch for pure boundary container: actual={file_size} B, expected={LANU_HEADER_BYTES + expected_boundary_bytes} B"
            )
    else:
        min_nb = non_boundary * 36
        max_nb = non_boundary * 48
        if rem_bytes < min_nb or rem_bytes > max_nb or ((rem_bytes - min_nb) % 12) != 0:
            raise ContainerSniffError(
                f"Stream size violates LANU glyph size law: rem={rem_bytes} B cannot partition "
                f"{non_boundary} non-boundary glyphs into 36B/48B Opt5 records"
            )

    details: Dict[str, Any] = {
        "flags": flags,
        "total_glyphs": total_glyphs,
        "boundary_glyphs": boundary_glyphs,
        "non_boundary_glyphs": non_boundary,
        "boundary_bytes_per_glyph": b_width,
        "layout_verified": False,
    }

    if layout_path is not None:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        if layout.get("stream_size_bytes", file_size) != file_size:
            raise ContainerSniffError(
                f"Layout stream_size_bytes ({layout.get('stream_size_bytes')}) != file_size ({file_size})"
            )
        projs = layout.get("projections", [])
        if not projs:
            raise ContainerSniffError(f"Layout {layout_path} contains zero projections")
        cursor = LANU_HEADER_BYTES
        counted = 0
        for idx_p, proj in enumerate(projs):
            d_out = proj.get("d_out")
            d_in = proj.get("d_in")
            if type(d_out) is not int or type(d_in) is not int or d_out <= 0 or d_in <= 0 or (d_in % 256) != 0:
                raise ContainerSniffError(f"Layout projection[{idx_p}] has invalid matrix dimensions {d_out}x{d_in}")
            codec = proj.get("codec_id", 4 if proj.get("codec", "OPT4") == "OPT4" else 5)
            width = proj.get("bytes_per_glyph", 132 if codec == 4 else 48)
            if codec not in (4, 5) or width not in ({4: (92, 132), 5: (36, 48)}[codec]):
                raise ContainerSniffError(f"Layout projection[{idx_p}] has invalid codec={codec}/width={width}")
            g = d_out * (d_in // 256)
            off = proj.get("stream_byte_offset", proj.get("byte_offset", 16))
            if proj.get("glyphs", g) != g or proj.get("tensor_bytes", g * width) != g * width or off != cursor:
                raise ContainerSniffError(
                    f"Layout projection[{idx_p}] offset/tensor_bytes mismatch at cursor={cursor}"
                )
            cursor += g * width
            counted += g
        if cursor != file_size or counted != total_glyphs:
            raise ContainerSniffError(
                f"Layout cursor/glyph census mismatch: cursor={cursor} (file={file_size}), "
                f"counted={counted} (header={total_glyphs})"
            )
        details["layout_verified"] = True
        details["layout_projections"] = len(projs)

    fmt_id = "lanp_v3" if magic == LANP_MAGIC else f"lanu_v{ver}"
    return ContainerSniff(
        path=path_str,
        format_id=fmt_id,
        magic=magic.decode("ascii"),
        version=ver,
        file_size_bytes=file_size,
        expected_size_bytes=file_size,
        valid=True,
        details=details,
    )


def _sniff_luj1(f, file_size: int, path_str: str, verify_hash: bool = False) -> ContainerSniff:
    if file_size < LUJ1_HEADER_BYTES:
        raise ContainerSniffError(f"LUJ1 bank truncated ({file_size} B < {LUJ1_HEADER_BYTES} B)")
    f.seek(0)
    hdr = _read_exact(f, LUJ1_HEADER_BYTES, "LUJ1 header")
    magic = hdr[:4]
    ver, bits_flag, reserved, n_glyphs = struct.unpack_from("<BBHI", hdr, 4)
    if magic != LUJ1_MAGIC or ver != 1 or reserved != 0 or n_glyphs == 0:
        raise ContainerSniffError(
            f"Invalid LUJ1 header: magic={magic!r}, ver={ver}, reserved={reserved}, n_glyphs={n_glyphs}"
        )
    tok_sha = hdr[12:44].hex()
    body_sha_hdr = hdr[76:108].hex()
    body_verified = False
    if verify_hash:
        h = hashlib.sha256()
        f.seek(LUJ1_HEADER_BYTES)
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        got = h.hexdigest()
        if got != body_sha_hdr:
            raise ContainerSniffError(f"LUJ1 body SHA-256 mismatch: header={body_sha_hdr}, actual={got}")
        body_verified = True
    return ContainerSniff(
        path=path_str,
        format_id="luj1_v1",
        magic="LUJ1",
        version=ver,
        file_size_bytes=file_size,
        expected_size_bytes=file_size,
        valid=True,
        details={
            "flags_byte": bits_flag,
            "n_glyphs": n_glyphs,
            "tokenizer_sha256": tok_sha,
            "body_sha256_verified": body_verified,
        },
    )


def _sniff_raw_glyph_bin(
    f,
    file_size: int,
    path_str: str,
    bytes_per_glyph: Optional[int] = None,
) -> ContainerSniff:
    if file_size == 0:
        raise ContainerSniffError("Empty .glyph.bin file (0 bytes)")
    candidates = [bytes_per_glyph] if bytes_per_glyph is not None else [w for w in _RAW_GLYPH_WIDTHS if file_size % w == 0]
    if not candidates or (bytes_per_glyph is not None and file_size % bytes_per_glyph != 0):
        raise ContainerSniffError(
            f"Raw glyph file size {file_size} B is not an exact multiple of valid record widths {_RAW_GLYPH_WIDTHS}"
        )
    bpg = candidates[0]
    if bpg not in _RAW_GLYPH_WIDTHS:
        raise ContainerSniffError(f"Unsupported bytes_per_glyph={bpg}; expected one of {_RAW_GLYPH_WIDTHS}")

    n_glyphs = file_size // bpg
    f.seek(0)
    # Validate scale/bias fp16 pairs across the header of each sampled record.
    sample_cap = min(n_glyphs, 256)
    for g in range(sample_cap):
        f.seek(g * bpg)
        rec_hdr = _read_exact(f, 4, f"glyph[{g}] fp16 header")
        s1, s2 = struct.unpack("<ee", rec_hdr)
        if not (math.isfinite(s1) and math.isfinite(s2)):
            raise ContainerSniffError(f"Non-finite fp16 scale/bias at glyph[{g}] ({s1}, {s2})")

    return ContainerSniff(
        path=path_str,
        format_id=f"raw_glyph_{bpg}b",
        magic="NONE",
        version=0,
        file_size_bytes=file_size,
        expected_size_bytes=n_glyphs * bpg,
        valid=True,
        details={
            "bytes_per_glyph": bpg,
            "glyphs": n_glyphs,
            "weights_covered": n_glyphs * 256,
            "header_samples_checked": sample_cap,
        },
    )


def sniff_container(
    path: Union[str, Path],
    *,
    layout_path: Optional[Union[str, Path]] = None,
    raw_bytes_per_glyph: Optional[int] = None,
    verify_hash: bool = False,
    raise_on_error: bool = True,
) -> ContainerSniff:
    """
    Inspect and validate a Language-U binary container before reading or mmapping.

    Dispatches automatically across `T16C`, `LANU`, `LANP`, `LUJ1`, and `.glyph.bin`.
    """
    p = Path(path)
    if not p.is_file():
        err = f"Container file missing: {p}"
        if raise_on_error:
            raise FileNotFoundError(err)
        return ContainerSniff(str(p), "missing", "", 0, 0, 0, False, error=err)

    file_size = os.path.getsize(p)
    try:
        if file_size == 0:
            raise ContainerSniffError(f"Container file is empty (0 bytes): {p}")
        with p.open("rb") as f:
            head4 = f.read(4)
            f.seek(0)
            if head4 == T16C_MAGIC:
                return _sniff_t16c(f, file_size, str(p))
            if head4 in (LANU_MAGIC, LANP_MAGIC):
                lp = _infer_layout_path(p, layout_path)
                return _sniff_lanu(f, file_size, str(p), layout_path=lp)
            if head4 == LUJ1_MAGIC:
                return _sniff_luj1(f, file_size, str(p), verify_hash=verify_hash)
            if p.name.endswith(".glyph.bin") or raw_bytes_per_glyph is not None:
                return _sniff_raw_glyph_bin(f, file_size, str(p), bytes_per_glyph=raw_bytes_per_glyph)
            raise ContainerSniffError(f"Unrecognized container magic {head4!r} on '{p.name}' ({file_size} B)")
    except ContainerSniffError as exc:
        if raise_on_error:
            raise
        return ContainerSniff(str(p), "invalid", "", 0, file_size, 0, False, error=str(exc))


def _self_test() -> int:
    """Execute 6 fail-closed unit gates on container_sniff primitives."""
    failures = 0
    repo_root = Path(__file__).resolve().parents[1]

    # Gate 1: Valid repository sample containers (LANU + Opt4 + raw .glyph.bin)
    try:
        s1 = sniff_container(repo_root / "model" / "gemma4_e4b_sample_layer0_q_proj.neon_opt4.lanu")
        s2 = sniff_container(repo_root / "model" / "gemma4_e4b_sample_neon_hybrid_v2.lanu")
        s3 = sniff_container(repo_root / "model" / "gemma4_e4b_sample_layer0_q_proj.glyph.bin")
        assert s1.valid and s1.details["layout_verified"] and s1.details["total_glyphs"] == 160
        assert s2.valid and s2.details["layout_verified"] and s2.details["total_glyphs"] == 320
        assert s3.valid and s3.details["glyphs"] == 16 and s3.details["bytes_per_glyph"] == 48
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    # Gate 2: Truncation rejection on LANU (must fail closed without silent acceptance)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "trunc.lanu"
            raw = (repo_root / "model" / "gemma4_e4b_sample_layer0_q_proj.neon_opt4.lanu").read_bytes()
            bad.write_bytes(raw[:-1])
            caught = False
            try:
                sniff_container(bad)
            except ContainerSniffError:
                caught = True
            assert caught
    except Exception as exc:
        print(f"[Gate 2 FAIL] {exc}")
        failures += 1

    # Gate 3: Synthetic T16C v1 (including 5-trit ternary mode n_bits=5) roundtrip & trailing-byte fail
    try:
        assert tile16_blk_bytes(256, 5, False) == 864
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.t16c"
            mname = b"test.w"
            hdr = struct.pack(T16C_HEADER_FMT, T16C_MAGIC, 1, 3, 1, 1, 0)
            n_entry = struct.pack("<H", 1) + b"n" + struct.pack("<I", 4) + (b"\x00" * 16)
            m_hdr = struct.pack(T16C_MAT_HEADER_FMT, len(mname), 16, 256, 256, 5, 0, 1, 0, 0, 864, 0)
            prefix = hdr + n_entry + m_hdr + mname
            pad = ((len(prefix) + 63) & ~63) - len(prefix)
            good_bytes = prefix + (b"\x00" * pad) + (b"\x00" * 864)
            p.write_bytes(good_bytes)
            res = sniff_container(p)
            assert res.valid and res.details["mode_counts"]["b5_sym_bs256"] == 1
            p.write_bytes(good_bytes + b"\xff")
            res_bad = sniff_container(p, raise_on_error=False)
            assert not res_bad.valid
    except Exception as exc:
        print(f"[Gate 3 FAIL] {exc}")
        failures += 1

    # Gate 4: Lying LANU header (`boundary > total`) & invalid magic rejection
    try:
        with tempfile.TemporaryDirectory() as tmp:
            lying = Path(tmp) / "lying.lanu"
            lying.write_bytes(struct.pack(LANU_HEADER_FMT, LANU_MAGIC, 2, 45, 10, 50) + b"\x00" * (10 * 132))
            bad_magic = Path(tmp) / "fake.bin"
            bad_magic.write_bytes(b"NOPE" + b"\x00" * 32)
            assert not sniff_container(lying, raise_on_error=False).valid
            assert not sniff_container(bad_magic, raise_on_error=False).valid
    except Exception as exc:
        print(f"[Gate 4 FAIL] {exc}")
        failures += 1

    # Gate 5: Non-finite NaN scale/bias rejection in .glyph.bin
    try:
        with tempfile.TemporaryDirectory() as tmp:
            gbad = Path(tmp) / "nan.glyph.bin"
            gbad.write_bytes(struct.pack("<ee", float("nan"), 1.0) + b"\x00" * 44)
            assert not sniff_container(gbad, raise_on_error=False).valid
    except Exception as exc:
        print(f"[Gate 5 FAIL] {exc}")
        failures += 1

    # Gate 6: LUJ1 container header + SHA-256 verification
    try:
        with tempfile.TemporaryDirectory() as tmp:
            luj = Path(tmp) / "bank.luj"
            body = b"payload_weights_1234"
            bhash = hashlib.sha256(body).digest()
            hdr = bytearray(LUJ1_HEADER_BYTES)
            hdr[0:4] = LUJ1_MAGIC
            struct.pack_into("<BBHI", hdr, 4, 1, 18, 0, 5)
            hdr[76:108] = bhash
            luj.write_bytes(bytes(hdr) + body)
            ok_luj = sniff_container(luj, verify_hash=True)
            assert ok_luj.valid and ok_luj.details["body_sha256_verified"]
            luj.write_bytes(bytes(hdr) + b"corrupted_body")
            assert not sniff_container(luj, verify_hash=True, raise_on_error=False).valid
    except Exception as exc:
        print(f"[Gate 6 FAIL] {exc}")
        failures += 1

    if failures == 0:
        print("[PASS] container_sniff.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate Language-U binary containers.")
    parser.add_argument("paths", nargs="*", help="Container file paths to inspect (.t16c, .lanu, .luj, .glyph.bin)")
    parser.add_argument("--layout", default=None, help="Optional explicit .layout.json for .lanu/.lanp checks")
    parser.add_argument("--verify-hash", action="store_true", help="Verify LUJ1 body SHA-256")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args()

    if not args.paths:
        return _self_test()

    results = []
    any_bad = False
    for raw_p in args.paths:
        res = sniff_container(
            raw_p,
            layout_path=args.layout,
            verify_hash=args.verify_hash,
            raise_on_error=False,
        )
        results.append(asdict(res))
        if not res.valid:
            any_bad = True
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            status = "PASS" if r["valid"] else "FAIL"
            extra = r["error"] or str(r["details"])
            print(f"[{status}] {r['path']} format={r['format_id']} size={r['file_size_bytes']} B | {extra}")
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

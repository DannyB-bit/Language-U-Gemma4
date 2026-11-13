"""Linux native exact census; fingerprints are resolved with full byte equality."""
import ctypes
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile

from .pi4_stream import checked_layout, sha256_file, validate_projection


class Projection(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_uint64), ("count", ctypes.c_uint64),
                ("width", ctypes.c_uint32), ("kind", ctypes.c_uint32)]


class Result(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("records", "unique_records", "payload_bytes", "unique_payload_bytes",
                 "zlib_bytes", "workspace_bytes", "largest_fingerprint_group")]
    _fields_ += [("unique_by_kind", ctypes.c_uint64 * 8)]


def native_redundancy_scan(stream_path, layout_path, *, max_workspace=4_000_000_000,
                           hash_bits=64, expected_sha256=None):
    """Sort fingerprints with deterministic two-array radix workspace.

    Full-record comparisons resolve every fingerprint collision. `hash_bits`
    can be reduced in SOFTWARE TESTS to force collision groups; published real
    scans use 64. Workspace excludes file-backed pages and is not an RSS bound.
    Hashes are checked before and after scanning. This is host data analysis,
    never a physical Pi inference benchmark.
    """
    if type(max_workspace) is not int or max_workspace < 0 or max_workspace > 2**64 - 1:
        raise ValueError("Invalid workspace budget")
    if type(hash_bits) is not int or not 1 <= hash_bits <= 64:
        raise ValueError("hash_bits must be in 1..64")
    if platform.system() != "Linux" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("Native census requires 64-bit Linux; use the SQLite scanner elsewhere")
    compiler = shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        raise RuntimeError("Native census requires a C compiler and zlib development headers")
    layout, _ = checked_layout(stream_path, layout_path)
    original_sha = sha256_file(stream_path)
    if expected_sha256 is not None and original_sha != expected_sha256:
        raise ValueError("Physical stream SHA256 does not match the expected artifact")
    size = Path(stream_path).stat().st_size
    entries = []
    for p in layout["projections"]:
        _, width, count, offset = validate_projection(p, size)
        kind = {36: 0, 48: 1, 92: 2, 132: 3}[width]
        rotation = p.get("rotation_mode", layout.get("rotation_mode", "direct"))
        if rotation == "fwht8":
            kind += 4
        entries.append(Projection(offset, count, width, kind))
    source = Path(__file__).with_name("stream_census.c")
    with tempfile.TemporaryDirectory(prefix="langu-census-") as temporary:
        library_path = Path(temporary) / "libcensus.so"
        subprocess.run([compiler, "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", str(source), "-lz", "-o", str(library_path)], check=True)
        library = ctypes.CDLL(str(library_path))
        function = library.census_glyphs_c
        function.argtypes = [ctypes.c_char_p, ctypes.POINTER(Projection), ctypes.c_size_t,
                            ctypes.c_uint64, ctypes.c_uint, ctypes.POINTER(Result),
                            ctypes.c_char_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        projections = (Projection * len(entries))(*entries)
        output, error = Result(), ctypes.create_string_buffer(512)
        status = function(os.fsencode(Path(stream_path).resolve()), projections, len(entries),
                          max_workspace, hash_bits, ctypes.byref(output), error, len(error))
        if status:
            raise ValueError(error.value.decode("utf-8", errors="replace"))
    if sha256_file(stream_path) != original_sha:
        raise ValueError("Physical stream changed during census")
    index_bits = (output.unique_records - 1).bit_length()
    return {"stream_sha256": original_sha, "glyphs": output.records,
            "unique_glyphs": output.unique_records, "payload_bytes": output.payload_bytes,
            "zlib_bytes": output.zlib_bytes, "zlib_ratio": output.payload_bytes / output.zlib_bytes,
            "unique_payload_bytes": output.unique_payload_bytes,
            "ideal_dictionary_bytes_excluding_metadata": output.unique_payload_bytes +
                (output.records * index_bits + 7) // 8,
            "unique_by_kind": list(output.unique_by_kind),
            "workspace_bytes_excluding_mmap_pages": output.workspace_bytes,
            "largest_fingerprint_group": output.largest_fingerprint_group,
            "fingerprint_bits": hash_bits, "collision_resolution": "full metadata and payload comparison",
            "scope": "supplied stream only", "pi4_benchmark": False}

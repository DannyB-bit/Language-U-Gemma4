"""Bounded-memory, lossless Opt4 packing and exact stream redundancy scans."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import zlib

def validate_projection(projection, stream_size):
    def integer(key, default=None):
        value = projection.get(key, default)
        if type(value) is not int:
            raise ValueError(f"{key} must be an integer")
        return value

    codec = integer("codec_id", 4 if projection.get("codec", "OPT4") == "OPT4" else 5)
    width = integer("bytes_per_glyph", 132 if codec == 4 else 48)
    rows, channels = integer("d_out"), integer("d_in")
    if codec not in (4, 5) or width not in ({4: (92, 132), 5: (36, 48)}[codec]):
        raise ValueError("Unsupported codec/bytes_per_glyph combination")
    if not 0 < rows <= 2**31 - 1 or not 0 < channels <= 2**31 - 1 or channels % 256:
        raise ValueError("Invalid native matrix dimensions or incomplete glyph row")
    glyphs = rows * (channels // 256)
    if integer("glyphs", glyphs) != glyphs or integer("tensor_bytes", glyphs * width) != glyphs * width:
        raise ValueError("Projection shape/glyph/byte counts disagree")
    offset = integer("stream_byte_offset", projection.get("byte_offset", 16))
    if offset < 16 or offset + glyphs * width > stream_size:
        raise ValueError("Projection range extends outside the stream")
    rotation = projection.get("rotation_mode")
    if rotation is not None and rotation not in ("fwht8", "none", "direct"):
        raise ValueError("Unsupported rotation_mode")
    return codec, width, glyphs, offset


def checked_layout(stream_path, layout_path):
    path = Path(stream_path)
    layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(16)
    if len(raw) != 16:
        raise ValueError("Truncated LANU header")
    magic, version, flags, total, boundary = struct.unpack("<4sHHII", raw)
    if (magic, version) not in ((b"LANU", 2), (b"LANP", 3)) or boundary > total:
        raise ValueError("Invalid or unsupported LANU header")
    if layout.get("stream_size_bytes", size) != size:
        raise ValueError("Layout stream_size_bytes disagrees with physical file")
    if layout.get("rotation_mode", "direct") not in ("direct", "none", "fwht8"):
        raise ValueError("Unsupported default rotation mode")
    cursor, counted = 16, 0
    projections = layout.get("projections", [])
    if not projections:
        raise ValueError("A projection layout is required")
    for p in projections:
        _, width, count, offset = validate_projection(p, size)
        if offset != cursor:
            raise ValueError("Projection ranges must be ordered, contiguous and nonoverlapping")
        cursor += count * width
        counted += count
    if cursor != size or counted != total or layout.get("total_macro_glyphs", counted) != counted:
        raise ValueError("Header/layout/file size or glyph census mismatch")
    return layout, raw


def pack_opt4_92(raw):
    """132 -> 92, lossless for canonical indices in [0, 2047].

    Preserve the first four fp16 bytes verbatim. Serialize 64 indices in stage
    order as one LSB-first 704-bit integer; reject noncanonical high bits.
    """
    if len(raw) % 132:
        raise ValueError("Opt4 source is not a whole number of 132-byte glyphs")
    out = bytearray()
    for i in range(0, len(raw), 132):
        record = raw[i:i + 132]
        indices = struct.unpack("<64H", record[4:])
        packed = 0
        for j, index in enumerate(indices):
            if index >= 2048:
                raise ValueError("Opt4 index exceeds 11 bits; refusing lossy masking")
            packed |= index << (11 * j)
        out.extend(record[:4])
        out.extend(packed.to_bytes(88, "little"))
    return bytes(out)


def unpack_opt4_92(raw):
    if len(raw) % 92:
        raise ValueError("Packed Opt4 is not a whole number of 92-byte glyphs")
    out = bytearray()
    for i in range(0, len(raw), 92):
        record = raw[i:i + 92]
        bits = int.from_bytes(record[4:], "little")
        out.extend(record[:4])
        out.extend(struct.pack("<64H", *[(bits >> (11 * j)) & 2047 for j in range(64)]))
    return bytes(out)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repack_stream(stream_path, layout_path, output_directory, *, chunk_glyphs=1024):
    """Create a new two-file stream; source is never overwritten.

    The destination directory must not exist. A failed attempt remains without
    a complete layout and is never presented as a valid stream.
    """
    if type(chunk_glyphs) is not int or chunk_glyphs <= 0:
        raise ValueError("chunk_glyphs must be positive")
    layout, header = checked_layout(stream_path, layout_path)
    result = deepcopy(layout)
    source_hash = hashlib.sha256()
    target_hash = hashlib.sha256()
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    stream_out = destination / "packed.lanu"
    layout_out = destination / "packed.layout.json"
    partial = destination / "packed.lanu.partial"
    offset = 16
    with Path(stream_path).open("rb") as src, partial.open("xb") as dst:
        if src.read(16) != header:
            raise ValueError("Source changed after validation")
        source_hash.update(header)
        # A distinct magic makes existing LANU-only consumers reject this file;
        # the old parser did not check version and would misread 92B as 132B.
        new_header = b"LANP" + struct.pack("<H", 3) + header[6:]
        dst.write(new_header)
        target_hash.update(new_header)
        for p, original in zip(result["projections"], layout["projections"]):
            codec, old_width, count, _ = validate_projection(original, Path(stream_path).stat().st_size)
            new_width = 92 if codec == 4 else old_width
            p.update(bytes_per_glyph=new_width, stream_byte_offset=offset,
                     tensor_bytes=count * new_width)
            p.pop("byte_offset", None)
            left = count
            while left:
                take = min(left, chunk_glyphs)
                raw = src.read(take * old_width)
                if len(raw) != take * old_width:
                    raise ValueError("Source truncated during repack")
                source_hash.update(raw)
                packed = pack_opt4_92(raw) if codec == 4 and old_width == 132 else raw
                if codec == 4 and old_width == 132 and unpack_opt4_92(packed) != raw:
                    raise RuntimeError("Lossless round-trip gate failed")
                dst.write(packed)
                target_hash.update(packed)
                left -= take
            offset += p["tensor_bytes"]
        if src.read(1):
            raise ValueError("Source grew during repack")
    # A second streaming hash detects concurrent source edits, including old pages.
    if sha256_file(stream_path) != source_hash.hexdigest():
        raise ValueError("Source changed during repack")
    result.update(stream_file=stream_out.name, stream_size_bytes=offset,
                  container_version=3, source_sha256=source_hash.hexdigest(),
                  stream_sha256=target_hash.hexdigest())
    partial.rename(stream_out)
    layout_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    checked_layout(stream_out, layout_out)
    return {"stream": str(stream_out), "layout": str(layout_out),
            "source_bytes": Path(stream_path).stat().st_size, "packed_bytes": offset,
            "saved_bytes": Path(stream_path).stat().st_size - offset,
            "source_sha256": source_hash.hexdigest(), "packed_sha256": target_hash.hexdigest(),
            "roundtrip": "PASS", "model_quality_gate": "UNVERIFIED",
            "pi4_residency_gate": "UNVERIFIED"}


def redundancy_scan(stream_path, layout_path, database_path):
    """Exact unique records in a disk-backed SQLite table; no hash collisions.

    zlib sees the sequential glyph payload (header excluded). Unique counts
    include codec, width and rotation because identical bytes can decode
    differently. The caller must provide space for a database roughly as large
    as the uncompressed stream plus SQLite overhead; no RAM-sized Python set.
    """
    layout, header = checked_layout(stream_path, layout_path)
    database = Path(database_path)
    with database.open("xb"):
        pass
    compressor = zlib.compressobj(level=6)
    compressed, payload_bytes, records = 0, 0, 0
    digest = hashlib.sha256(header)
    try:
        with sqlite3.connect(database) as connection, Path(stream_path).open("rb") as src:
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("CREATE TABLE glyphs (codec INTEGER, width INTEGER, rotation TEXT, payload BLOB, "
                               "PRIMARY KEY(codec,width,rotation,payload)) WITHOUT ROWID")
            if src.read(16) != header:
                raise ValueError("Source changed after validation")
            for p in layout["projections"]:
                codec, width, count, _ = validate_projection(p, Path(stream_path).stat().st_size)
                rotation = p.get("rotation_mode", layout.get("rotation_mode", "direct"))
                for start in range(0, count, 4096):
                    take = min(4096, count - start)
                    raw = src.read(width * take)
                    if len(raw) != width * take:
                        raise ValueError("Source truncated during scan")
                    digest.update(raw)
                    compressed += len(compressor.compress(raw))
                    payload_bytes += len(raw)
                    records += take
                    connection.executemany("INSERT OR IGNORE INTO glyphs VALUES (?,?,?,?)",
                                           ((codec, width, rotation, raw[i:i+width])
                                            for i in range(0, len(raw), width)))
                connection.commit()
            if src.read(1):
                raise ValueError("Source grew during scan")
            unique, unique_bytes = connection.execute("SELECT COUNT(*), SUM(width) FROM glyphs").fetchone()
        compressed += len(compressor.flush())
        if sha256_file(stream_path) != digest.hexdigest():
            raise ValueError("Source changed during scan")
        # Ideal address bits only; a real dictionary still needs widths/offsets,
        # alignment and decoder workspace. This is an estimate, not a new format.
        index_bits = (unique - 1).bit_length()
        ideal = unique_bytes + (records * index_bits + 7) // 8
        return {"stream_sha256": digest.hexdigest(), "glyphs": records,
                "unique_glyphs": unique, "payload_bytes": payload_bytes,
                "zlib_bytes": compressed, "zlib_ratio": payload_bytes / compressed,
                "unique_payload_bytes": unique_bytes,
                "ideal_dictionary_bytes_excluding_metadata": ideal,
                "database_bytes": database.stat().st_size,
                "scope": "supplied stream only", "pi4_benchmark": False}
    except Exception:
        # Do not report a partial uniqueness scan as complete.
        raise

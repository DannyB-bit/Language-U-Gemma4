"""Experimental 3-bit tied embedding/head; quality must be measured separately.

LUE3 v1 is an explicit new format: a 32-byte header, then 98-byte records
(one little-endian FP16 scale and 256 signed 3-bit codes, offset by four).
This is lossy. A smaller file is not evidence of generation fidelity.
"""
from pathlib import Path
import struct

import numpy as np

HEADER = struct.Struct("<4s7I")
BLOCK, RECORD = 256, 98


def pack_codes(codes):
    values = np.asarray(codes)
    if values.ndim != 2 or values.shape[1] != BLOCK or np.any(values < -4) or np.any(values > 3):
        raise ValueError("Expected 256 signed 3-bit codes per row")
    unsigned = (values.astype(np.int16)+4).astype(np.uint32).reshape(-1, 32, 8)
    words = np.sum(unsigned << (np.arange(8, dtype=np.uint32)*3), axis=-1, dtype=np.uint32)
    return ((words[..., None] >> np.array([0, 8, 16], dtype=np.uint32)) & 255).astype(np.uint8).reshape(-1, 96)


def unpack_codes(payload):
    data = np.asarray(payload, dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != 96:
        raise ValueError("Expected 96 payload bytes per 3-bit block")
    data = data.reshape(-1, 32, 3).astype(np.uint32)
    words = data[..., 0] | (data[..., 1] << 8) | (data[..., 2] << 16)
    return (((words[..., None] >> (np.arange(8, dtype=np.uint32)*3)) & 7).astype(np.int8)-4).reshape(-1, BLOCK)


def encode_blocks(weights):
    x = np.asarray(weights, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != BLOCK or not np.isfinite(x).all():
        raise ValueError("Expected finite 256-element embedding blocks")
    scale = np.max(np.abs(x), axis=1)/np.float32(3.5)
    for _ in range(4):
        safe = np.where(scale > 0, scale, 1)
        codes = np.clip(np.rint(x/safe[:, None]), -4, 3)
        denom = np.sum(codes*codes, axis=1)
        scale = np.divide(np.sum(x*codes, axis=1), denom, out=np.zeros_like(scale), where=denom > 0)
    stored = scale.astype("<f2")
    if not np.isfinite(stored).all():
        raise ValueError("Embedding scales exceed the FP16 format")
    safe = np.where(stored > 0, stored.astype(np.float32), 1)
    codes = np.clip(np.rint(x/safe[:, None]), -4, 3).astype(np.int8)
    records = np.empty((len(x), RECORD), dtype=np.uint8)
    records[:, :2] = stored[:, None].view(np.uint8)
    records[:, 2:] = pack_codes(codes)
    return records


def decode_blocks(records):
    data = np.asarray(records, dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != RECORD:
        raise ValueError("Expected 98-byte 3-bit records")
    scale = data[:, :2].copy().view("<f2").astype(np.float32)
    return unpack_codes(data[:, 2:]).astype(np.float32)*scale


def encode_table(source, destination, chunk_rows=512):
    """Quantize every source row; do not retain an uncompressed head for reranking."""
    if source.columns % BLOCK or chunk_rows <= 0:
        raise ValueError("Invalid embedding geometry/chunk size")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with destination.open("xb") as output:
        output.write(HEADER.pack(b"LUE3", 1, source.rows, source.columns, BLOCK, RECORD, 0, 0))
        for begin in range(0, source.rows, chunk_rows):
            values = np.stack([source.row(token) for token in range(begin, min(begin+chunk_rows, source.rows))])
            output.write(encode_blocks(values.reshape(-1, BLOCK)).tobytes())
    return destination.stat().st_size


class ThreeBitTable:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as source:
            raw = source.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("Truncated 3-bit embedding header")
        magic, version, self.rows, self.columns, block, record, reserved1, reserved2 = HEADER.unpack(raw)
        if (magic != b"LUE3" or version != 1 or block != BLOCK or record != RECORD
                or reserved1 or reserved2 or self.rows <= 0 or self.columns <= 0
                or self.columns % BLOCK or self.path.stat().st_size != HEADER.size+self.rows*self.columns//BLOCK*RECORD):
            raise ValueError("Invalid 3-bit embedding format/geometry")
        self.data = np.memmap(self.path, dtype=np.uint8, mode="r", offset=HEADER.size,
                              shape=(self.rows, self.columns//BLOCK, RECORD))

    def row(self, token):
        if not 0 <= token < self.rows:
            raise ValueError("Embedding token is outside the vocabulary")
        return decode_blocks(self.data[token]).reshape(self.columns)

    def logits(self, hidden, chunk_rows=512):
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        if hidden.shape != (self.columns,) or not np.isfinite(hidden).all() or chunk_rows <= 0:
            raise ValueError("Invalid output-head input/chunk size")
        result = np.empty(self.rows, dtype=np.float32)
        for start in range(0, self.rows, chunk_rows):
            stop = min(start+chunk_rows, self.rows)
            weights = decode_blocks(self.data[start:stop].reshape(-1, RECORD)).reshape(stop-start, self.columns)
            result[start:stop] = weights @ hidden
        return result

    def close(self):
        if self.data is not None:
            self.data._mmap.close()
            self.data = None

"""Experimental RCF1 low-rank correction, stored FP16 or row-scaled INT8.

Inference uses FP32 arithmetic and bounded row decoding. This is a NumPy
reference operator, not a native ARM kernel or a throughput claim.
"""
from pathlib import Path
import struct

import numpy as np

HEADER = struct.Struct("<4s7I4Q")


def quantize_rows(matrix):
    matrix = np.asarray(matrix, dtype="f4")
    peak = np.max(np.abs(matrix), axis=1)
    scales = np.where(peak > 0, peak / 127, 1).astype("<f4")
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Unrepresentable INT8 scales")
    values = np.clip(np.rint(matrix / scales[:, None]), -127, 127).astype("i1")
    return values, scales


def save_correction(path, a, b, storage="int8"):
    """Write B(Ax); exclusive creation protects earlier experiment outputs."""
    a, b = np.asarray(a), np.asarray(b)
    if (a.ndim != 2 or b.ndim != 2 or min(*a.shape, *b.shape) < 1
            or b.shape[1] != a.shape[0]
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Invalid correction factors")
    r, n = a.shape
    m = b.shape[0]
    if storage == "int8":
        av, sa = quantize_rows(a)
        bv, sb = quantize_rows(b)
        scales = np.concatenate((sa, sb)).astype("<f4").tobytes()
        dtype = 2
    elif storage == "fp16":
        with np.errstate(over="ignore"):
            av, bv = a.astype("<f2"), b.astype("<f2")
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError("Unrepresentable FP16 factors")
        scales, dtype = b"", 1
    else:
        raise ValueError("Unknown correction storage")
    ap, bp = av.tobytes(), bv.tobytes()
    header = HEADER.pack(b"RCF1", 1, m, n, r, dtype, 0, 0,
                         len(scales), len(ap), len(bp), len(scales) + len(ap) + len(bp))
    with Path(path).open("xb") as stream:
        stream.write(header + scales + ap + bp)
    return Path(path).stat().st_size


class Correction:
    def __init__(self, path):
        path = Path(path)
        with path.open("rb") as stream:
            raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("Truncated correction header")
        magic, version, m, n, r, dtype, flags, reserved, ss, aa, bb, total = HEADER.unpack(raw)
        itemsize = 2 if dtype == 1 else 1
        if (magic != b"RCF1" or version != 1 or dtype not in (1, 2)
                or flags or reserved or min(m, n, r) < 1
                or ss != (0 if dtype == 1 else 4 * (r + m))
                or aa != r * n * itemsize or bb != m * r * itemsize
                or total != ss + aa + bb or path.stat().st_size != HEADER.size + total):
            raise ValueError("Invalid correction format")
        self.shape, self.storage = (m, n, r), "fp16" if dtype == 1 else "int8"
        self.data = np.memmap(path, mode="r", dtype="u1")
        if dtype == 2:
            scales = self.data[64:64 + ss].view("<f4")
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError("Invalid correction scales")
            self.sa, self.sb = scales[:r], scales[r:]
        else:
            self.sa, self.sb = np.ones(r, "f4"), np.ones(m, "f4")
        value_type = "<f2" if dtype == 1 else "i1"
        self.a = self.data[64 + ss:64 + ss + aa].view(value_type).reshape(r, n)
        self.b = self.data[64 + ss + aa:].view(value_type).reshape(m, r)
        if not np.isfinite(self.a).all() or not np.isfinite(self.b).all():
            raise ValueError("Nonfinite correction values")

    @staticmethod
    def _matvec(values, scales, x, chunk):
        out = np.empty(len(values), "f4")
        for start in range(0, len(values), chunk):
            stop = start + chunk
            decoded = values[start:stop].astype("f4") * scales[start:stop, None]
            out[start:stop] = decoded @ x
        return out

    def project(self, x, chunk=8):
        x = np.asarray(x, dtype="f4")
        if x.shape != (self.shape[1],) or not np.isfinite(x).all():
            raise ValueError("Expected a finite input vector of the stored width")
        if not isinstance(chunk, int) or chunk < 1:
            raise ValueError("Chunk size must be positive")
        h = self._matvec(self.a, self.sa, x, chunk)
        return self._matvec(self.b, self.sb, h, chunk)

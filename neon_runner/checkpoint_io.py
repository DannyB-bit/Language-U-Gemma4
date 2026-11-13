"""Small floating-point checkpoint sidecars and bounded BF16 table access.

Uses the published safetensors container format. No PyTorch, executable model
code, procedural weight generation, or implicit defaults for missing tensors.
"""
import json
import math
import ctypes
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

from .build_native import build_c_shared_library


def bf16_to_float32(values):
    bits = np.asarray(values, dtype="<u2").astype(np.uint32)
    bits <<= 16
    return bits.view(np.float32)


def load_float_sidecar(path, prefix="model.language_model."):
    path = Path(path)
    with path.open("rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors prefix")
        length = struct.unpack("<Q", raw)[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("Safetensors header exceeds the sidecar limit")
        encoded = source.read(length)
        if len(encoded) != length:
            raise ValueError("Truncated safetensors header")
        metadata = json.loads(encoded)
        base, size = 8 + length, path.stat().st_size
        tensors = {}
        for name, item in metadata.items():
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            shape, dtype = item["shape"], item["dtype"]
            if any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("Invalid tensor shape")
            elements = math.prod(shape)
            widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
            if dtype not in widths or elements > 1_000_000:
                raise ValueError("Sidecar loader accepts small floating tensors only")
            begin, end = item["data_offsets"]
            if not 0 <= begin <= end <= size-base or end-begin != elements*widths[dtype]:
                raise ValueError("Invalid safetensors data range")
            source.seek(base+begin)
            raw = source.read(end-begin)
            if len(raw) != end-begin:
                raise ValueError("Sidecar changed while reading")
            if dtype == "BF16":
                value = bf16_to_float32(np.frombuffer(raw, dtype="<u2"))
            else:
                value = np.frombuffer(raw, dtype={"F16":"<f2", "F32":"<f4", "F64":"<f8"}[dtype]).astype(np.float32)
            value = value.reshape(shape)
            if not np.isfinite(value).all():
                raise ValueError("Non-finite sidecar parameter")
            tensors[name] = value
    return tensors


class BF16Table:
    """Exact source table. Full-vocabulary logits; no shortlist or re-ranking."""
    def __init__(self, path, rows, columns):
        self.path = Path(path)
        self.rows, self.columns = rows, columns
        if self.path.stat().st_size != rows*columns*2:
            raise ValueError("BF16 table size does not match the full vocabulary")
        self.data = np.memmap(self.path, dtype="<u2", mode="r", shape=(rows, columns))

    def row(self, token):
        if not 0 <= token < self.rows:
            raise ValueError("Embedding token is outside the vocabulary")
        return bf16_to_float32(self.data[token])

    def logits(self, hidden, chunk_rows=512):
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        if hidden.shape != (self.columns,) or not np.isfinite(hidden).all():
            raise ValueError("Invalid output-head input")
        result = np.empty(self.rows, dtype=np.float32)
        for start in range(0, self.rows, chunk_rows):
            stop = min(start+chunk_rows, self.rows)
            result[start:stop] = bf16_to_float32(self.data[start:stop]) @ hidden
        return result

    def close(self):
        if self.data is not None:
            self.data._mmap.close()
            self.data = None


class SafetensorsFile:
    """Read-only, bounded views of an ordinary safetensors checkpoint.

    Only explicitly supported numeric types are exposed. Loading a checkpoint
    never imports or executes code from its repository.
    """
    DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "I32": "<i4", "I64": "<i8"}

    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        with self.path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors prefix")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= min(16*1024*1024, self.size-8):
                raise ValueError("Invalid safetensors header length")
            self.header = json.loads(source.read(length))
        self.base = 8+length
        self.data = np.memmap(self.path, dtype=np.uint8, mode="r")

    def tensor(self, name):
        item = self.header[name]
        shape, dtype = item["shape"], item["dtype"]
        if dtype not in self.DTYPES or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Unsupported safetensors shape/dtype")
        begin, end = item["data_offsets"]
        data_type = np.dtype(self.DTYPES[dtype])
        if (type(begin) is not int or type(end) is not int
                or not 0 <= begin <= end <= self.size-self.base
                or end-begin != math.prod(shape)*data_type.itemsize):
            raise ValueError("Invalid safetensors data range")
        return np.ndarray(shape, dtype=data_type, buffer=self.data, offset=self.base+begin)

    def close(self):
        if self.data is not None:
            self.data._mmap.close()
            self.data = None


def round_bf16(values):
    """Round finite FP32 values to BF16 (ties to even), returning FP32."""
    x = np.ascontiguousarray(values, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


class QATInt4Projections:
    """Offline FP32-accumulation reference for symmetric group-32 QAT weights.

    The public compressed-tensors format packs low nibble first, with an offset
    of eight, and stores a separate BF16 scale for each group of 32 columns.
    Dequantized weights are rounded to BF16, as in compressed-tensors' default
    dequantization. Activations and accumulation here remain FP32. This is not
    a physical-device or BF16-arithmetic performance reference.
    """
    def __init__(self, path, chunk_rows=128):
        if type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError("Invalid projection chunk size")
        self.checkpoint = SafetensorsFile(path)
        self.chunk_rows = chunk_rows

    def rows(self, metadata, start, stop):
        key = metadata["full_key"].removesuffix(".weight_packed")
        packed = self.checkpoint.tensor(key+".weight_packed")
        scale_bits = self.checkpoint.tensor(key+".weight_scale")
        rows, columns = metadata["d_out"], metadata["d_in"]
        shape = self.checkpoint.tensor(key+".weight_shape")
        if (packed.dtype != np.dtype("<i4") or packed.shape != (rows, columns//8)
                or scale_bits.shape != (rows, columns//32) or columns % 32
                or self.checkpoint.header[key+".weight_scale"]["dtype"] != "BF16"
                or shape.tolist() != [rows, columns] or not 0 <= start < stop <= rows):
            raise ValueError("Projection is not the supported symmetric group-32 format")
        data = packed[start:stop].view(np.uint8)
        integer = np.stack((data & 15, data >> 4), axis=-1).reshape(stop-start, columns).astype(np.float32)-8
        scales = bf16_to_float32(scale_bits[start:stop])
        return round_bf16(integer*np.repeat(scales, 32, axis=1))

    def project(self, metadata, inputs):
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != metadata["d_in"] or not np.isfinite(x).all():
            raise ValueError("Invalid projection inputs")
        output = np.empty((len(x), metadata["d_out"]), dtype=np.float32)
        for start in range(0, metadata["d_out"], self.chunk_rows):
            stop = min(start+self.chunk_rows, metadata["d_out"])
            output[:, start:stop] = x @ self.rows(metadata, start, stop).T
        return output

    def close(self):
        self.checkpoint.close()


class NativeQATInt4Projections(QATInt4Projections):
    """Fused source-model reference; independent NumPy decoder remains the oracle.

    Does not materialize large FP32 matrices. This reads the much larger source
    checkpoint, so its memory/I/O behavior is not the Pi 4 candidate's behavior.
    """
    def __init__(self, path, threads=4, library=None):
        super().__init__(path)
        try:
            if threads <= 0:
                raise ValueError("Invalid reference thread count")
            source = Path(__file__).with_name("qat_reference.c")
            selected = library or os.environ.get("LANGU_QAT_SO")
            if selected is None:
                key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
                directory = Path(tempfile.gettempdir())/("language-u-qat-"+key)
                directory.mkdir(exist_ok=True)
                ext = ".dll" if sys.platform == "win32" else ".so"
                selected = directory/("reference"+ext)
                if not selected.exists():
                    temporary = directory/(f"reference-{os.getpid()}{ext}")
                    if sys.platform == "win32":
                        build_c_shared_library(source, temporary, use_avx2=False)
                    else:
                        subprocess.run([os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC", "-fopenmp",
                                        str(source), "-o", str(temporary)], check=True, capture_output=True)
                    temporary.replace(selected)
            self.library = ctypes.CDLL(str(selected))
            self.function = self.library.qat_group32_project
            self.function.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*4
            self.function.restype = ctypes.c_int
            self.threads = threads
        except Exception:
            self.close()
            raise

    def project(self, metadata, inputs):
        # The independent row decoder performs the shape/dtype checks first.
        self.rows(metadata, 0, 1)
        key = metadata["full_key"].removesuffix(".weight_packed")
        packed = self.checkpoint.tensor(key+".weight_packed")
        scales = self.checkpoint.tensor(key+".weight_scale")
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != metadata["d_in"] or not np.isfinite(x).all():
            raise ValueError("Invalid projection inputs")
        result = np.empty((len(x), metadata["d_out"]), dtype=np.float32)
        for begin in range(0, len(x), 64):
            stop = min(begin+64, len(x))
            status = self.function(packed.ctypes.data, scales.ctypes.data, x[begin:stop].ctypes.data,
                                   result[begin:stop].ctypes.data, metadata["d_out"], metadata["d_in"],
                                   stop-begin, self.threads)
            if status != 1:
                raise RuntimeError("Source-model reference projection failed")
        return result

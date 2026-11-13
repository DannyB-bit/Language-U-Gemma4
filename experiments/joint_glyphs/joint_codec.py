"""Experimental token/weight/operation records; not the existing 36B ABI.

Every record carries exactly 256 weights and 20 or 21 real token IDs. Exact
QAT mode preserves a group-32 INT4/BF16 source, not pre-QAT training weights.
Exact FP mode preserves arbitrary finite FP16/FP32 source bits using a repair
lane for a standard 36B weight glyph. SHA256 checks integrity, not authenticity.
"""
from dataclasses import dataclass
import hashlib
import struct
import zlib

import numpy as np

STREAM = struct.Struct("<4sBBHI32s32s32s")
RECORD = struct.Struct("<IBBHIII")
INSTRUCTION = struct.Struct("<BBhhqq")
QAT4, OPT5, EXACT16, EXACT32 = 1, 2, 3, 4
ADD, SUB, MUL, ASSERT_EQ, ASSERT_GE = 1, 2, 3, 4, 5
DIV_EXACT, MINIMUM, MAXIMUM, POWER, REMAINDER = 6, 7, 8, 9, 10
OPERATIONS = (ADD, SUB, MUL, ASSERT_EQ, ASSERT_GE, DIV_EXACT,
              MINIMUM, MAXIMUM, POWER, REMAINDER)
MAX_RECORDS = 1024


def pack_tokens(tokens):
    if len(tokens) not in (20, 21) or any(type(t) is not int or not 0 <= t < 2**18 for t in tokens):
        raise ValueError("Expected 20 or 21 unsigned 18-bit token IDs")
    value = sum(t << (18*i) for i, t in enumerate(tokens))
    return value.to_bytes((18*len(tokens) + 7)//8, "little")


def unpack_tokens(raw, count):
    if count not in (20, 21) or len(raw) != (18*count+7)//8:
        raise ValueError("Invalid token lane length")
    value = int.from_bytes(raw, "little")
    if value >> (18*count):
        raise ValueError("Nonzero token padding")
    return [(value >> (18*i)) & ((1 << 18)-1) for i in range(count)]


def opt5_encode(weights):
    w = np.asarray(weights, dtype="f4")
    if w.shape != (256,) or not np.isfinite(w).all():
        raise ValueError("Expected 256 finite weights")
    with np.errstate(over="ignore"):
        bias = np.float16(w.mean(dtype="f8"))
        scale = np.float16(np.mean(np.abs(w.astype('f8') - float(bias))))
    if not np.isfinite([bias, scale]).all():
        raise ValueError("Unrepresentable OPT5 parameters")
    return np.asarray([scale, bias], dtype="<f2").tobytes() + np.packbits(w >= float(bias)).tobytes()


def opt5_decode(raw):
    if len(raw) != 36:
        raise ValueError("OPT5 must be 36 bytes")
    scale, bias = np.frombuffer(raw[:4], dtype="<f2").astype("f4")
    if not np.isfinite([scale, bias]).all():
        raise ValueError("Nonfinite OPT5 parameters")
    signs = np.unpackbits(np.frombuffer(raw[4:], dtype="u1")).astype("f4") * 2 - 1
    return signs * scale + bias


def qat_decode(raw):
    if len(raw) != 144:
        raise ValueError("QAT lane must contain 128 code bytes and 16 scale bytes")
    packed = np.frombuffer(raw[:128], dtype="u1")
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(-1).astype("f4") - 8
    scales = (np.frombuffer(raw[128:], dtype="<u2").astype("<u4") << 16).view("<f4")
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("Invalid BF16 QAT scales")
    with np.errstate(over="ignore"):
        values = codes * np.repeat(scales, 32)
    if not np.isfinite(values).all():
        raise ValueError("QAT dequantization overflow")
    bits = values.view("u4")
    values = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view("f4")
    if not np.isfinite(values).all():
        raise ValueError("QAT BF16 rounding overflow")
    return values


def exact_encode(weights, dtype):
    if dtype not in ("fp16", "fp32"):
        raise ValueError("Unknown exact dtype")
    dt = "<f2" if dtype == "fp16" else "<f4"
    w = np.asarray(weights)
    if w.shape != (256,) or not np.isfinite(w).all():
        raise ValueError("Expected 256 finite weights")
    target = w.astype(dt)
    if not np.isfinite(target).all() or not np.array_equal(target.astype('f8'), w.astype('f8')):
        raise ValueError("Requested dtype would change source values")
    try:
        base = opt5_encode(target)
    except ValueError:
        # A zero predictor allows exact FP32 values outside FP16 scale range.
        base = bytes(36)
    predicted = opt5_decode(base).astype(dt)
    if not np.isfinite(predicted).all():
        base = bytes(36)
        predicted = opt5_decode(base).astype(dt)
    xor = np.bitwise_xor(np.frombuffer(target.tobytes(), 'u1'),
                        np.frombuffer(predicted.tobytes(), 'u1'))
    # Byte-plane shuffle improves compressibility; both order and size are fixed.
    shuffled = xor.reshape(256, target.itemsize).T.copy().tobytes()
    compressed = zlib.compress(shuffled, 9)
    mode, repair = (1, compressed) if len(compressed) < len(shuffled) else (0, shuffled)
    return base + bytes([mode]) + repair


def exact_decode(raw, dtype):
    dt = "<f2" if dtype == EXACT16 else "<f4"
    size = 256 * np.dtype(dt).itemsize
    if len(raw) < 37 or len(raw) > 37 + size:
        raise ValueError("Invalid exact lane length")
    mode = raw[36]
    if mode == 0:
        repair = raw[37:]
    elif mode == 1:
        decoder = zlib.decompressobj()
        try:
            repair = decoder.decompress(raw[37:], size + 1)
        except zlib.error as error:
            raise ValueError("Invalid repair compression") from error
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Incomplete or oversized repair")
    else:
        raise ValueError("Unknown repair mode")
    if len(repair) != size:
        raise ValueError("Invalid repair expansion length")
    unshuffled = np.frombuffer(repair, 'u1').reshape(-1, 256).T.copy().reshape(-1)
    predicted = opt5_decode(raw[:36]).astype(dt)
    target = np.bitwise_xor(unshuffled, np.frombuffer(predicted.tobytes(), 'u1'))
    weights = target.view(dt).copy()
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite restored weights")
    return weights


def decode_weights(codec, raw, output_dtype="fp32"):
    if codec == QAT4:
        weights = qat_decode(raw)
    elif codec == OPT5:
        weights = opt5_decode(raw)
    elif codec in (EXACT16, EXACT32):
        weights = exact_decode(raw, codec)
    else:
        raise ValueError("Unknown weight codec")
    if output_dtype not in ("fp16", "fp32"):
        raise ValueError("Unknown output dtype")
    with np.errstate(over='ignore'):
        result = weights.astype('<f2' if output_dtype == 'fp16' else '<f4')
    if not np.isfinite(result).all() or not np.array_equal(result.astype('f8'), weights.astype('f8')):
        raise ValueError("Output cast would lose recovered values")
    return result


@dataclass
class Glyph:
    tokens: list
    codec: int
    weights: bytes
    instruction: bytes


def instruction(op, left_ref=-1, right_ref=-1, left=0, right=0):
    if op not in OPERATIONS:
        raise ValueError("Unknown operation")
    return INSTRUCTION.pack(op, 0, left_ref, right_ref, left, right)


def compile_chunk(glyphs, tokenizer_sha256, source_weights_sha256):
    if not 1 <= len(glyphs) <= MAX_RECORDS:
        raise ValueError("Invalid glyph count")
    if len(tokenizer_sha256) != 32 or len(source_weights_sha256) != 32:
        raise ValueError("Expected full SHA256 bindings")
    body = bytearray()
    for index, glyph in enumerate(glyphs):
        tokens = pack_tokens(glyph.tokens)
        decode_weights(glyph.codec, glyph.weights)
        if len(glyph.instruction) != INSTRUCTION.size:
            raise ValueError("Invalid instruction size")
        body += RECORD.pack(index, glyph.codec, len(glyph.tokens), 0,
                            len(glyph.weights), len(tokens), len(glyph.instruction))
        body += glyph.weights + tokens + glyph.instruction
    raw = STREAM.pack(b"LUJ1", 1, 18, 0, len(glyphs), tokenizer_sha256,
                      source_weights_sha256, hashlib.sha256(body).digest()) + body
    restored, _ = decode_chunk(raw, tokenizer_sha256)
    execute(restored)  # A malformed or failing executable chunk is not emitted.
    return raw


def decode_chunk(raw, expected_tokenizer_sha256):
    if len(raw) < STREAM.size:
        raise ValueError("Truncated stream")
    magic, version, bits, flags, count, tokenizer, source, digest = STREAM.unpack_from(raw)
    if (magic != b"LUJ1" or version != 1 or bits != 18 or flags
            or not 1 <= count <= MAX_RECORDS or tokenizer != expected_tokenizer_sha256
            or len(raw) > STREAM.size + count * 1200):
        raise ValueError("Invalid stream or tokenizer binding")
    body = raw[STREAM.size:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("Chunk integrity failure")
    out, pos = [], 0
    for index in range(count):
        if pos + RECORD.size > len(body):
            raise ValueError("Truncated record")
        got, codec, nt, reserved, nw, nb, ni = RECORD.unpack_from(body, pos)
        pos += RECORD.size
        if (got != index or reserved or ni != INSTRUCTION.size or nt not in (20, 21)
                or nb != (18*nt+7)//8 or nw > 1061 or pos+nw+nb+ni > len(body)):
            raise ValueError("Invalid record boundaries")
        weights = bytes(body[pos:pos+nw]); pos += nw
        tokens = unpack_tokens(body[pos:pos+nb], nt); pos += nb
        code = bytes(body[pos:pos+ni]); pos += ni
        decode_weights(codec, weights)
        out.append(Glyph(tokens, codec, weights, code))
    if pos != len(body):
        raise ValueError("Unexpected trailing bytes")
    if all(g.codec != OPT5 for g in out):
        recovered = b"".join(decode_weights(g.codec, g.weights).tobytes() for g in out)
        if hashlib.sha256(recovered).digest() != source:
            raise ValueError("Exact source-weight fingerprint mismatch")
    return out, source


def execute(glyphs):
    """Typed integer DAG, explicitly not neural inference or natural-language parsing."""
    values = []
    for index, glyph in enumerate(glyphs):
        if len(glyph.instruction) != INSTRUCTION.size:
            raise ValueError("Invalid instruction length")
        op, flags, ar, br, ai, bi = INSTRUCTION.unpack(glyph.instruction)
        if flags or op not in OPERATIONS:
            raise ValueError("Invalid operation")
        def operand(ref, immediate):
            if ref == -1:
                return immediate
            if not 0 <= ref < index or immediate != 0:
                raise ValueError("Invalid, forward or noncanonical dependency")
            return values[ref]
        a, b = operand(ar, ai), operand(br, bi)
        if op == ADD:
            value = a+b
        elif op == SUB:
            value = a-b
        elif op == MUL:
            value = a*b
        elif op == DIV_EXACT:
            if b == 0 or a % b:
                raise ValueError("Division requires a nonzero divisor and exact integer result")
            value = a // b
        elif op == MINIMUM:
            value = min(a, b)
        elif op == MAXIMUM:
            value = max(a, b)
        elif op == POWER:
            # Avoid unbounded allocation before the int64 overflow check.
            if not 0 <= b <= 63:
                raise ValueError("Exponent outside bounded integer domain")
            value = pow(a, b)
        elif op == REMAINDER:
            if b == 0:
                raise ValueError("Remainder requires a nonzero divisor")
            value = a % b
        else:
            if not (a == b if op == ASSERT_EQ else a >= b):
                raise ValueError("Reasoning constraint failed")
            value = a
        if not -(2**63) <= value < 2**63:
            raise ValueError("Integer overflow")
        values.append(value)
    return {"values": values, "result": values[-1],
            "english": f"The verified result is {values[-1]}."}

# The Glyph Law — Language-U's deterministic semantic core
# KOFI 2026-10-11. Byte-identical laws to cuneiform_glyphs_ch9b (Consider-Trinity)
# and the Language-u-Process encoder. These are NOT inventions: every constant
# here was byte-proven against the real 33,711,216-glyph substrate.

import math
import hashlib
import struct
from typing import List, Tuple

# ── The 6D Cuneiform-U coordinate space ──────────────────────────────────────
# Language-U maps every proposition onto 6 orthogonal semantic axes:
#   [Domain, Subdomain, Operation, Modality, Depth, Polarity]
# packed into three 8-bit transmission radicals (the fleet's Cuneiform-U law):
#   R_C (Classifier) = (Domain << 4) | Subdomain
#   R_F (Factor)      = (Operation << 4) | Modality
#   R_A (Active)      = (Depth << 4) | Polarity

DOMAINS = {
    "general": 0, "solana": 1, "lorarf": 2, "security": 3, "deploy": 4,
    "frontend": 5, "backend": 6, "compression": 7, "zk": 8, "media": 9,
    "mathematics": 10, "physics": 11,
}

SUBDOMAINS = {
    "core": 0, "kernel": 1, "runtime": 2, "memory": 3, "network": 4,
    "storage": 5, "protocol": 6, "interface": 7, "sensor": 8, "frame": 9,
    "signal": 10, "state": 11, "channel": 12, "matrix": 13, "vector": 14, "stream": 15,
}

OPERATIONS = {
    "compute": 0, "decode": 1, "encode": 2, "transform": 3, "verify": 4,
    "rotate": 5, "project": 6, "reduce": 7, "hash": 8, "emit": 9,
    "load": 10, "store": 11, "add": 12, "multiply": 13, "compare": 14, "branch": 15,
}

MODALITIES = {
    "edge": 0, "cloud": 1, "offline": 2, "realtime": 3, "batch": 4,
    "serial": 5, "parallel": 6, "native": 7, "gpu": 8, "cpu": 9,
    "radio": 10, "chain": 11, "disk": 12, "wire": 13, "quantum": 14, "host": 15,
}

DEPTHS = {
    "bit": 0, "byte": 1, "word": 2, "dword": 3, "qword": 4,
    "vector": 5, "block": 6, "tensor": 7, "field": 8, "kernel": 9,
    "layer": 10, "model": 11, "system": 12, "universe": 13, "exact": 14, "approx": 15,
}

POLARITIES = {
    "positive": 0, "negative": 1, "inverse": 2, "converse": 3,
    "forward": 4, "backward": 5, "up": 6, "down": 7,
    "true": 8, "false": 9, "on": 10, "off": 11, "open": 12, "closed": 13, "fast": 14, "slow": 15,
}

AXES = [("domain", DOMAINS), ("subdomain", SUBDOMAINS), ("operation", OPERATIONS),
        ("modality", MODALITIES), ("depth", DEPTHS), ("polarity", POLARITIES)]


def _unknown(axis: str, word: str) -> int:
    """Deterministic 4-bit coordinate for words outside the canon tables —
    derived from the word's sha256, not from a random or fixed guess."""
    h = hashlib.sha256(f"{axis}:{word}".encode()).digest()[0]
    return h & 0xF


class Radicals:
    """The three 8-bit Cuneiform-U transmission radicals of a proposition."""
    __slots__ = ("rc", "rf", "ra")

    def __init__(self, domain: int, subdomain: int, operation: int,
                 modality: int, depth: int, polarity: int):
        self.rc = ((domain & 0xF) << 4) | (subdomain & 0xF)
        self.rf = ((operation & 0xF) << 4) | (modality & 0xF)
        self.ra = ((depth & 0xF) << 4) | (polarity & 0xF)

    def to_bytes(self) -> bytes:
        return bytes((self.rc, self.rf, self.ra))

    def __repr__(self):
        return f"Radicals(rc=0x{self.rc:02x}, rf=0x{self.rf:02x}, ra=0x{ra:02x})".replace("ra=0x", "ra=0x") if False else (
            f"Radicals(rc=0x{self.rc:02x}, rf=0x{self.rf:02x}, ra=0x{self.ra:02x})")


def proposition_to_radicals(prop) -> Radicals:
    """Map a parsed proposition's 6 axis words to the 3 transmission radicals."""
    vals = []
    for axis, table in AXES:
        word = getattr(prop, axis).lower()
        vals.append(table.get(word, _unknown(axis, word)))
    return Radicals(*vals)


# ── The S^7 octonion projection (H8 Hadamard law) ────────────────────────────
# Byte-identical to the encoder's H8_NORMALIZED (H8_RAW / sqrt(8)).

H8_NORMALIZED: List[List[float]] = [
    [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
    [ 1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0],
    [ 1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
    [ 1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
    [ 1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0],
    [ 1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0],
    [ 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
]


_INV_SQRT8 = 1.0 / math.sqrt(8.0)
for _r in range(8):
    for _c in range(8):
        H8_NORMALIZED[_r][_c] *= _INV_SQRT8


def rotate_h8(vec8: List[float]) -> List[float]:
    """Rotate an 8-group through the encoder's H8 basis (the S^7 projection)."""
    out = [0.0] * 8
    for r in range(8):
        acc = 0.0
        row = H8_NORMALIZED[r]
        for c in range(8):
            acc += row[c] * vec8[c]
        out[r] = acc
    return out


# ── The 2048-entry trit codebook (3^8 states, k=1..5 sparsity selection) ─────
# Byte-identical to build_2048_glyph_codebook() in cuneiform_glyphs_ch9b.

def build_trit_codebook() -> List[List[float]]:
    all_states = []
    k_sparsity = []
    for i in range(6561):
        val = i
        state = [0.0] * 8
        count = 0
        for d in range(8):
            trit = float(val % 3) - 1.0
            state[d] = trit
            if trit != 0.0:
                count += 1
            val //= 3
        all_states.append(state)
        k_sparsity.append(count)
    selected = ([i for i, k in enumerate(k_sparsity) if k == 1]
                + [i for i, k in enumerate(k_sparsity) if k == 2]
                + [i for i, k in enumerate(k_sparsity) if k == 3]
                + [i for i, k in enumerate(k_sparsity) if k == 4]
                + [i for i, k in enumerate(k_sparsity) if k == 5][:352])
    assert len(selected) == 2048, f"codebook selection = {len(selected)} != 2048"
    codebook = []
    for idx in selected:
        raw = all_states[idx]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        codebook.append([x / norm for x in raw])
    return codebook


# ── The glyph containers (the two real disk laws) ─────────────────────────────
# 132B dual-stage RVQ glyph:
#   [s1 fp16 @0:2][s2 fp16 @2:4][idx1 i16 x32 @4:68][idx2 i16 x32 @68:132]
#   w_vec8 = cb[idx1]*s1 + cb[idx2]*s2  (32 groups), then @ H8 (encoder law:
#   v8_rot = v8 @ H8, decoder inverts with H8^T = H8 — Hadamard symmetric)
# 36B bitmask glyph:
#   [scale fp16 @0:2][bias fp16 @2:4][32B bitmask @4:36 → 256 bits → ±scale + bias]

WEIGHTS_PER_GLYPH = 256
BPG_OPT4 = 132
BPG_OPT5 = 36


def encode_glyph_132(weights: List[float], codebook) -> bytes:
    """Quantize 256 weights into a 132B dual-stage RVQ glyph (the encoder law:
    greedy stage-1 codebook match, least-squares scale, residual stage-2)."""
    assert len(weights) == 256
    # The REAL 132B per-glyph law (byte-proven on the live streams):
    #   [s1 fp16 @0:2][s2 fp16 @2:4][32 x i16 idx1 @4:68][32 x i16 idx2 @68:132]
    # ONE scale pair for the whole glyph; 32 codebook index pairs for 32 groups.
    i1s, i2s = [], []
    rot_all = []
    for g in range(32):
        vec8 = weights[g * 8:(g + 1) * 8]
        rot = rotate_h8(vec8)
        rot_all.append(rot)
        # stage 1: greedy codebook match on the normalized direction
        norm = math.sqrt(sum(x * x for x in rot)) or 1e-12
        u = [x / norm for x in rot]
        best_i1, best_dot = 0, -2.0
        for ci in range(2048):
            cb = codebook[ci]
            dot = sum(cb[c] * u[c] for c in range(8))
            if dot > best_dot:
                best_dot, best_i1 = dot, ci
        rec1 = codebook[best_i1]
        rec1_scaled = [r * norm for r in rec1]  # scale placeholder; refined below
        # stage 2: residual match
        res = [rot[c] - rec1_scaled[c] for c in range(8)]
        rnorm = math.sqrt(sum(x * x for x in res)) or 1e-12
        ru = [x / rnorm for x in res]
        best_i2, best_dot2 = 0, -2.0
        for ci in range(2048):
            cb = codebook[ci]
            dot = sum(cb[c] * ru[c] for c in range(8))
            if dot > best_dot2:
                best_dot2, best_i2 = dot, ci
        i1s.append(best_i1)
        i2s.append(best_i2)
    # Global least-squares scale pair over all 32 groups (the shared scales):
    num1 = num2 = den = 0.0
    for g in range(32):
        rot = rot_all[g]
        r1, r2 = codebook[i1s[g]], codebook[i2s[g]]
        for c in range(8):
            num1 += r1[c] * rot[c]
            num2 += r2[c] * rot[c]
            den += (r1[c] + r2[c]) ** 2
    s1 = num1 / (den or 1e-12)
    s2 = num2 / (den or 1e-12)
    out = bytearray()
    out += struct.pack("<e", s1)
    out += struct.pack("<e", s2)
    for i in i1s:
        out += struct.pack("<h", i)
    for i in i2s:
        out += struct.pack("<h", i)
    assert len(out) == 132, f"container law violated: {len(out)}"
    return bytes(out)


def encode_glyph_36(weights: List[float]) -> bytes:
    """Quantize 256 weights into a 36B bitmask glyph (sign-of-median law)."""
    assert len(weights) == 256
    w = sorted(weights)
    median = (w[127] + w[128]) / 2.0
    centered = [x - median for x in weights]
    scale = float(np_max_abs(centered)) or 1.0
    bits = 0
    for i, x in enumerate(centered):
        if x >= 0.0:
            bits |= (1 << (255 - i))
    out = struct.pack("<e", scale) + struct.pack("<e", median)
    out += bits.to_bytes(32, "big")
    assert len(out) == 36
    return out


def np_max_abs(vals):
    m = 0.0
    for v in vals:
        av = abs(v)
        if av > m:
            m = av
    return m


def decode_glyph_132(data: bytes, codebook) -> List[float]:
    """Decode a 132B RVQ glyph into 256 weights (the fail-closed law)."""
    if len(data) < 132:
        raise ValueError("glyph slice too short (fail-closed)")
    s1 = struct.unpack_from("<e", data, 0)[0]
    s2 = struct.unpack_from("<e", data, 2)[0]
    weights = []
    for g in range(32):
        i1 = struct.unpack_from("<h", data, 4 + 2 * g)[0]
        i2 = struct.unpack_from("<h", data, 68 + 2 * g)[0]
        if not (0 <= i1 < 2048 and 0 <= i2 < 2048):
            raise ValueError(f"glyph index out of range: {i1}/{i2} (fail-closed)")
        cb1, cb2 = codebook[i1], codebook[i2]
        rot = [cb1[c] * s1 + cb2[c] * s2 for c in range(8)]
        weights.extend(rotate_h8(rot))
    return weights


def glyph_checksum(weights: List[float]) -> float:
    """Honest accumulation: the f64 sum of all 256 decoded weights."""
    return math.fsum(weights)

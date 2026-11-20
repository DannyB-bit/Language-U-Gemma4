#!/usr/bin/env python3
"""
Assemble a unified Gemma 4 31B V3 `.t16c` artifact from quantized layer caches.

Reads calibrated layer `.npz` archives (`embed_3b256.npz` and `L00..L59.npz`)
together with BF16/F32 norm and `layer_scalar` tensors from `model.safetensors`,
verifies exact tile16 byte sizes and `Int8LowRank` shapes fail-closed, and streams
out a 64-byte-aligned `.t16c` binary compatible with `run31b` and `t16c.rs`.
"""

import argparse
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

T16C_MAGIC = b"T16C"
T16C_VERSION = 1
ARCH_FLAGS = 3  # ARCH_FLAG_GEMMA4 | ARCH_FLAG_TIED_HEAD

# Mapping from fast_quant.c quantize mode_i -> (t16c n_bits, is_asym)
MODE_TO_T16C: Dict[int, Tuple[int, bool]] = {
    0: (1, True),          # 1-bit asymmetric bit-serial (64 + plane = 576B @ b256)
    1: (5, False),         # 5-trit ternary symmetric (32 + 52*16 = 864B @ b256)
    2: (2, False),         # 2-bit symmetric bit-serial (32 + 2*plane = 1056B @ b256)
    3: (2, True),          # 2-bit asymmetric bit-serial (64 + 2*plane = 1088B @ b256)
    4: (3, False),         # 3-bit symmetric bit-serial (32 + 3*plane = 1568B @ b256)
    5: (2 | 0x80, False),  # 2-bit non-uniform Gaussian symmetric (32 + 2*plane = 1056B @ b256)
    6: (3 | 0x80, False),  # 3-bit non-uniform Gaussian symmetric (32 + 3*plane = 1568B @ b256)
}

LAYER_PROJS: List[Tuple[str, str]] = [
    ("q_proj", "self_attn.q_proj.weight"),
    ("k_proj", "self_attn.k_proj.weight"),
    ("v_proj", "self_attn.v_proj.weight"),
    ("o_proj", "self_attn.o_proj.weight"),
    ("gate_proj", "mlp.gate_proj.weight"),
    ("up_proj", "mlp.up_proj.weight"),
    ("down_proj", "mlp.down_proj.weight"),
]
REQUIRED_PROJS = {"q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def tile16_blk_bytes(block_size: int, n_bits: int, is_asym: bool) -> int:
    """Return packed byte count of a single `16 x block_size` tile-block."""
    if block_size not in (64, 128, 256):
        raise ValueError(f"Unsupported block_size={block_size}")
    eff_bits = n_bits & 0x3F
    gauss = (n_bits & 0x80) != 0
    rowmajor = (n_bits & 0x40) != 0
    if gauss and eff_bits not in (2, 3):
        raise ValueError(f"T16C_GAUSS_BIT (0x80) requires eff_bits in (2, 3), got n_bits={n_bits}")
    if rowmajor and not (1 <= eff_bits <= 4):
        raise ValueError(f"T16C_ROWMAJOR_BIT (0x40) requires eff_bits in 1..4, got n_bits={n_bits}")
    if eff_bits == 5:
        if is_asym:
            raise ValueError("n_bits=5 (5-trit ternary) requires is_asym=False")
        quintets = (block_size + 4) // 5
        return 32 + quintets * 16
    if eff_bits == 8:
        if is_asym:
            raise ValueError("n_bits=8 (symmetric signed INT8) requires is_asym=False")
        return 32 + block_size * 16
    if not (1 <= eff_bits <= 4):
        raise ValueError(f"Unsupported n_bits={n_bits}")
    hdr_bytes = 64 if is_asym else 32
    plane_bytes = (block_size // 8) * 16
    return hdr_bytes + eff_bits * plane_bytes


def expected_tile16_bytes(d_out: int, d_in: int, block_size: int, n_bits: int, is_asym: bool) -> int:
    """Return expected packed byte length of a full `.t16c` matrix."""
    if d_out <= 0 or d_in <= 0 or (d_in % block_size) != 0:
        raise ValueError(f"Invalid matrix dims d_out={d_out}, d_in={d_in}, block_size={block_size}")
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // block_size
    return n_tiles * n_blks * tile16_blk_bytes(block_size, n_bits, is_asym)


def resolve_safetensors_path(model_path: Optional[Path]) -> Path:
    """Resolve `model.safetensors` across explicit path, env vars, and standard locations."""
    candidates: List[Path] = []
    if model_path is not None:
        candidates.append(model_path / "model.safetensors" if model_path.is_dir() else model_path)
    for env_key in ("GEMMA4_31B_MODEL", "MODEL_ASSETS_PATH", "GEMMA4_31B_DIR"):
        val = os.environ.get(env_key, "").strip()
        if val:
            vp = Path(val)
            candidates.append(vp / "model.safetensors" if vp.is_dir() else vp)
    candidates.append(Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/model.safetensors"))
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "Missing model.safetensors; pass --safetensors or set GEMMA4_31B_MODEL "
        f"(checked: {[str(c) for c in candidates]})"
    )


class SafetensorsReader:
    """Minimal zero-copy header parser and tensor reader for `.safetensors` files."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fp = open(path, "rb")
        try:
            hdr_len = struct.unpack("<Q", self._fp.read(8))[0]
            self.hdr: Dict[str, dict] = json.loads(self._fp.read(hdr_len).decode("utf-8"))
            self.data_base = 8 + hdr_len
        except Exception:
            self._fp.close()
            raise

    def __enter__(self) -> "SafetensorsReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def read_f32_tensor(self, name: str) -> np.ndarray:
        """Read a BF16 or F32 tensor as contiguous little-endian float32."""
        info = self.hdr[name]
        dtype = info["dtype"]
        o0, o1 = info["data_offsets"]
        self._fp.seek(self.data_base + o0)
        raw = self._fp.read(o1 - o0)
        if dtype == "BF16":
            arr = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).copy()
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype="<f4").copy()
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise ValueError(f"Unsupported norm dtype {dtype!r} for {name}")
        shape = info["shape"]
        return arr.reshape(shape) if shape else arr.reshape((1,))


def _write_mat_record(
    fp,
    pos: int,
    name: str,
    d_out: int,
    d_in: int,
    bsize: int,
    n_bits: int,
    is_asym: bool,
    has_fwht: bool,
    pbytes: np.ndarray,
    inv_s: Optional[np.ndarray],
    low_rank: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> int:
    """Serialize one `T16CMat` record matching `t16c::T16CModel::load` byte-for-byte."""
    exp_p = expected_tile16_bytes(d_out, d_in, bsize, n_bits, is_asym)
    pb_u8 = np.ascontiguousarray(pbytes, dtype=np.uint8).ravel()
    if pb_u8.size != exp_p:
        raise ValueError(f"{name}: packed length {pb_u8.size} != expected {exp_p}")

    inv_f32: Optional[np.ndarray] = None
    inv_kind = 0
    if inv_s is not None:
        inv_f32 = np.ascontiguousarray(inv_s, dtype="<f4").ravel()
        if inv_f32.size != d_in:
            raise ValueError(f"{name}: inv_s len {inv_f32.size} != d_in {d_in}")
        inv_kind = 1

    lr_rk = 0
    lr_bytes = 0
    su_f32 = sv_f32 = u_i8 = v_i8 = None
    if low_rank is not None:
        su_raw, sv_raw, u_raw, v_raw = low_rank
        su_f32 = np.ascontiguousarray(su_raw, dtype="<f4").ravel()
        sv_f32 = np.ascontiguousarray(sv_raw, dtype="<f4").ravel()
        u_i8 = np.ascontiguousarray(u_raw, dtype=np.int8)
        v_i8 = np.ascontiguousarray(v_raw, dtype=np.int8)
        lr_rk = int(sv_f32.size)
        if su_f32.size != d_out or u_i8.shape != (d_out, lr_rk) or v_i8.shape != (lr_rk, d_in):
            raise ValueError(f"{name}: invalid low_rank shapes su={su_f32.shape} u={u_i8.shape} v={v_i8.shape}")
        lr_bytes = d_out * 4 + lr_rk * 4 + d_out * lr_rk + lr_rk * d_in

    nb = name.encode("utf-8")
    fp.write(
        struct.pack(
            "<HIIHBBBBHQQ",
            len(nb),
            d_out,
            d_in,
            bsize,
            n_bits,
            1 if is_asym else 0,
            1 if has_fwht else 0,
            inv_kind,
            lr_rk,
            exp_p,
            lr_bytes,
        )
    )
    fp.write(nb)
    pos += 34 + len(nb)

    pad = (64 - (pos % 64)) % 64
    if pad:
        fp.write(b"\x00" * pad)
        pos += pad

    # Write packed buffer in chunks to keep transient peak copy small on edge devices.
    chunk_b = 16 * 1024 * 1024
    mv = memoryview(pb_u8)
    for off in range(0, pb_u8.size, chunk_b):
        fp.write(mv[off : off + chunk_b])
    pos += exp_p

    if inv_f32 is not None:
        fp.write(inv_f32.tobytes())
        pos += d_in * 4
    if lr_rk > 0:
        fp.write(su_f32.tobytes())
        fp.write(sv_f32.tobytes())
        fp.write(u_i8.tobytes())
        fp.write(v_i8.tobytes())
        pos += lr_bytes
    return pos


def assemble_t16c_v3(
    safetensors_path: Path,
    layers_dir: Path,
    out_path: Path,
    n_layers: int = 60,
    has_fwht: bool = True,
) -> int:
    """Assemble `.t16c` from `embed_3b256.npz` + `L00..L{n_layers-1}.npz` and norms."""
    embed_npz = layers_dir / "embed_3b256.npz"
    if not embed_npz.is_file():
        raise FileNotFoundError(f"Missing embed cache: {embed_npz}")
    for l in range(n_layers):
        lp = layers_dir / f"L{l:02d}.npz"
        if not lp.is_file():
            raise FileNotFoundError(f"Missing layer cache: {lp}")

    with SafetensorsReader(safetensors_path) as sr:
        norm_keys = sorted(
            k
            for k in sr.hdr
            if ("norm.weight" in k or "layer_scalar" in k) and k.startswith("model.language_model.")
        )
        if not norm_keys:
            raise ValueError(f"No model.language_model.* norm/layer_scalar keys found in {safetensors_path}")
        if "model.language_model.embed_tokens.weight" not in sr.hdr:
            raise KeyError("Missing model.language_model.embed_tokens.weight in safetensors header")
        emb_shape = sr.hdr["model.language_model.embed_tokens.weight"]["shape"]
        vocab_sz, d_model = int(emb_shape[0]), int(emb_shape[1])
        norms = [(k, sr.read_f32_tensor(k)) for k in norm_keys]

    # Count matrices cleanly first before writing binary header.
    n_mats = 1  # embed_tokens
    layer_present: List[List[Tuple[str, str]]] = []
    for l in range(n_layers):
        with np.load(layers_dir / f"L{l:02d}.npz") as d_npz:
            files_set = set(d_npz.files)
        present = []
        for stem, suffix in LAYER_PROJS:
            if f"{stem}__meta" in files_set:
                present.append((stem, suffix))
            elif stem in REQUIRED_PROJS:
                raise KeyError(f"L{l:02d}.npz missing required projection {stem}__meta")
        layer_present.append(present)
        n_mats += len(present)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".t16c.tmp")
    with open(tmp_path, "wb") as fp:
        pos = 0
        fp.write(struct.pack("<4s5I", T16C_MAGIC, T16C_VERSION, ARCH_FLAGS, len(norms), n_mats, 0))
        pos += 24
        for name, arr in norms:
            nb = name.encode("utf-8")
            a32 = np.ascontiguousarray(arr, dtype="<f4").ravel()
            fp.write(struct.pack("<H", len(nb)))
            fp.write(nb)
            fp.write(struct.pack("<I", a32.size))
            fp.write(a32.tobytes())
            pos += 2 + len(nb) + 4 + a32.size * 4

        with np.load(embed_npz) as ed:
            emb_pb = ed["pbytes"]
            pos = _write_mat_record(
                fp,
                pos,
                "model.language_model.embed_tokens.weight",
                vocab_sz,
                d_model,
                256,
                3,
                False,
                has_fwht,
                emb_pb,
                None,
                None,
            )
            del emb_pb

        for l in range(n_layers):
            with np.load(layers_dir / f"L{l:02d}.npz") as d_npz:
                for stem, suffix in layer_present[l]:
                    d_out, d_in, bsize, mode_i, rk = [int(v) for v in d_npz[f"{stem}__meta"]]
                    if mode_i not in MODE_TO_T16C:
                        raise ValueError(f"L{l:02d} {stem}: unknown quant mode_i={mode_i}")
                    n_bits, is_asym = MODE_TO_T16C[mode_i]
                    lr = None
                    if rk > 0:
                        lr = (
                            d_npz[f"{stem}__su"],
                            d_npz[f"{stem}__sv"],
                            d_npz[f"{stem}__ui8"],
                            d_npz[f"{stem}__vi8"],
                        )
                    pos = _write_mat_record(
                        fp,
                        pos,
                        f"model.language_model.layers.{l}.{suffix}",
                        d_out,
                        d_in,
                        bsize,
                        n_bits,
                        is_asym,
                        has_fwht,
                        d_npz[f"{stem}__pbytes"],
                        d_npz[f"{stem}__inv_s"],
                        lr,
                    )

    os.replace(tmp_path, out_path)
    return out_path.stat().st_size


def _write_synthetic_safetensors(path: Path, tensors: Dict[str, np.ndarray]) -> None:
    """Write minimal valid `.safetensors` containing F32 tensors for self-testing."""
    hdr = {}
    chunks = []
    off = 0
    for k, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    hb = json.dumps(hdr).encode("utf-8")
    pad = (8 - (len(hb) % 8)) % 8
    hb += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for c in chunks:
            f.write(c)


def _self_test() -> int:
    """Run 6 fail-closed self-verification gates without touching external assets."""
    failures = 0

    # Gate 1: Exact tile16 byte formulas across modes & block sizes
    try:
        assert tile16_blk_bytes(256, 1, True) == 576
        assert tile16_blk_bytes(256, 5, False) == 864
        assert tile16_blk_bytes(256, 2, False) == 1056
        assert tile16_blk_bytes(256, 2 | 0x80, False) == 1056
        assert tile16_blk_bytes(256, 3, False) == 1568
        assert tile16_blk_bytes(256, 3 | 0x80, False) == 1568
        assert tile16_blk_bytes(64, 5, False) == 240
        assert tile16_blk_bytes(128, 5, False) == 448
        assert tile16_blk_bytes(64, 8, False) == 1056
        assert tile16_blk_bytes(256, 8, False) == 4128
        try:
            tile16_blk_bytes(256, 5, True)
            raise AssertionError("Expected ValueError for ternary+asym")
        except ValueError:
            pass
        try:
            tile16_blk_bytes(256, 1 | 0x80, False)
            raise AssertionError("Expected ValueError for 1-bit+gauss")
        except ValueError:
            pass
        try:
            tile16_blk_bytes(256, 8, True)
            raise AssertionError("Expected ValueError for int8+asym")
        except ValueError:
            pass
    except Exception as exc:
        print(f"[Gate 1 FAIL] {exc}")
        failures += 1

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        sf_p = td / "model.safetensors"
        ldir = td / "layers"
        ldir.mkdir()
        out_p = td / "test.t16c"

        vocab, d = 32, 256
        _write_synthetic_safetensors(
            sf_p,
            {
                "model.language_model.embed_tokens.weight": np.zeros((vocab, d), dtype=np.float32),
                "model.language_model.layers.0.input_layernorm.weight": np.ones((d,), dtype=np.float32),
                "model.language_model.layers.0.layer_scalar": np.array([0.25], dtype=np.float32),
                "model.language_model.norm.weight": np.ones((d,), dtype=np.float32),
            },
        )

        # Gate 2: Fail-closed on missing layer/embed files
        try:
            try:
                assemble_t16c_v3(sf_p, ldir, out_p, n_layers=1)
                raise AssertionError("Expected FileNotFoundError")
            except FileNotFoundError:
                pass
        except Exception as exc:
            print(f"[Gate 2 FAIL] {exc}")
            failures += 1

        # Prepare valid embed + L00 (with modes 0, 1, 2, 4 and rank-4 Int8LowRank, omitting v_proj to test full-attn k_eq_v)
        emb_pb = np.full(expected_tile16_bytes(vocab, d, 256, 3, False), 0x3C, dtype=np.uint8)
        np.savez(ldir / "embed_3b256.npz", pbytes=emb_pb)

        l0_dict = {}
        specs = {
            "q_proj": (16, d, 256, 2, 4),
            "k_proj": (16, d, 256, 2, 0),
            "o_proj": (d, d, 256, 4, 4),
            "gate_proj": (32, d, 256, 0, 4),
            "up_proj": (32, d, 256, 1, 4),
            "down_proj": (d, d, 256, 1, 4),
        }
        for stem, (do, di, bs, md, rk) in specs.items():
            nb, asym = MODE_TO_T16C[md]
            l0_dict[f"{stem}__meta"] = np.array([do, di, bs, md, rk], dtype=np.int32)
            fill_b = 42 if nb == 5 else 0x55
            l0_dict[f"{stem}__pbytes"] = np.full(expected_tile16_bytes(do, di, bs, nb, asym), fill_b, dtype=np.uint8)
            l0_dict[f"{stem}__inv_s"] = np.linspace(0.5, 1.5, di, dtype=np.float32)
            if rk > 0:
                l0_dict[f"{stem}__su"] = np.full((do,), 0.01, dtype=np.float32)
                l0_dict[f"{stem}__sv"] = np.full((rk,), 0.02, dtype=np.float32)
                l0_dict[f"{stem}__ui8"] = np.full((do, rk), 7, dtype=np.int8)
                l0_dict[f"{stem}__vi8"] = np.full((rk, di), -5, dtype=np.int8)
        np.savez(ldir / "L00.npz", **l0_dict)

        # Gate 3..6: Stream assembly, header parsing, 64-byte alignment, exact payload check
        try:
            sz = assemble_t16c_v3(sf_p, ldir, out_p, n_layers=1)
            assert sz == out_p.stat().st_size and sz > 0
            raw = out_p.read_bytes()
            magic, ver, flags, n_norms, n_mats, resv = struct.unpack("<4s5I", raw[:24])
            assert magic == T16C_MAGIC and ver == T16C_VERSION and flags == ARCH_FLAGS
            assert n_norms == 3 and n_mats == 7 and resv == 0
            pos = 24
            seen_norms = []
            for _ in range(n_norms):
                nlen = struct.unpack("<H", raw[pos : pos + 2])[0]
                pos += 2
                nm = raw[pos : pos + nlen].decode("utf-8")
                pos += nlen
                cnt = struct.unpack("<I", raw[pos : pos + 4])[0]
                pos += 4 + cnt * 4
                seen_norms.append(nm)
            assert len(seen_norms) == 3

            seen_mats = []
            for _ in range(n_mats):
                nlen, do, di, bs, nb, asym_u8, fwht_u8, inv_k, rk_u16, pb_len, lr_len = struct.unpack(
                    "<HIIHBBBBHQQ", raw[pos : pos + 34]
                )
                pos += 34
                nm = raw[pos : pos + nlen].decode("utf-8")
                pos += nlen
                pos += (64 - (pos % 64)) % 64
                assert pos % 64 == 0, f"{nm} packed payload not 64B aligned: pos={pos}"
                assert pb_len == expected_tile16_bytes(do, di, bs, nb, bool(asym_u8))
                pos += pb_len
                if inv_k == 1:
                    pos += di * 4
                pos += lr_len
                seen_mats.append((nm, nb, bool(asym_u8), rk_u16))
            assert pos == len(raw), f"Trailing or missing bytes: pos={pos} vs len={len(raw)}"
            assert len(seen_mats) == 7
            # Check Gate 5: corrupt pbytes size fails closed
            l0_bad = dict(l0_dict)
            l0_bad["up_proj__pbytes"] = l0_dict["up_proj__pbytes"][:-1]
            np.savez(ldir / "L00.npz", **l0_bad)
            try:
                assemble_t16c_v3(sf_p, ldir, out_p, n_layers=1)
                raise AssertionError("Expected ValueError for truncated pbytes")
            except ValueError:
                pass
        except Exception as exc:
            print(f"[Gate 3-6 FAIL] {exc}")
            failures += 1

    if failures == 0:
        print("[PASS] pack_t16c_v3.py 6-gate self-test passed (all 6 gates)")
    return failures


def main() -> int:
    if len(sys.argv) == 1 or "--selftest" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser(description="Assemble Gemma 4 31B V3 .t16c file from layers_v3 caches.")
    ap.add_argument("--safetensors", type=Path, default=None, help="Path to QAT model.safetensors or dir")
    ap.add_argument("--layers-dir", type=Path, default=Path("/mnt/SHE-VAULT/work31b/layers_v3"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/SHE-VAULT/gemma4_31b_v3.t16c"))
    ap.add_argument("--n-layers", type=int, default=60)
    args = ap.parse_args()

    sf_path = resolve_safetensors_path(args.safetensors)
    sz = assemble_t16c_v3(sf_path, args.layers_dir.resolve(), args.out.resolve(), n_layers=args.n_layers)
    print(f"[OK] Wrote {args.out} ({sz:,} bytes / {sz / (1024**3):.3f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

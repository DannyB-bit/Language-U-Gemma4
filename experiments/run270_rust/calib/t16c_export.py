import struct
import time
from pathlib import Path
import numpy as np


T16C_MAGIC = b"T16C"
T16C_VERSION = 1
ARCH_FLAG_GEMMA4 = 1
ARCH_FLAG_TIED_HEAD = 2


def pack_tile16_bytes(
    d_out: int,
    d_in: int,
    block_size: int,
    n_bits: int,
    is_asym: bool,
    sc: np.ndarray,
    bi: np.ndarray | None,
    idx: np.ndarray,
) -> bytes:
    """Vectorized Tile-16 layout packer matching run270_rust::t16c::T16CMat bit-for-bit."""
    assert block_size in (64, 128, 256) and d_in % block_size == 0
    assert 1 <= n_bits <= 4
    n_tiles = (d_out + 15) // 16
    row_pad = n_tiles * 16
    n_blks = d_in // block_size
    bo_cnt = block_size // 8

    sc_f = np.asarray(sc, dtype=np.float32).reshape(d_out, n_blks)
    sc_pad = np.zeros((row_pad, n_blks), dtype=np.float16)
    sc_pad[:d_out, :] = sc_f.astype(np.float16)
    # [n_tiles, n_blks, 16] uint16 -> bytes [n_tiles, n_blks, 32]
    sc_u8 = (
        np.ascontiguousarray(sc_pad.reshape(n_tiles, 16, n_blks).transpose(0, 2, 1))
        .view(np.uint8)
        .reshape(n_tiles, n_blks, 32)
    )

    chunks = [sc_u8]
    if is_asym:
        assert bi is not None
        bi_f = np.asarray(bi, dtype=np.float32).reshape(d_out, n_blks)
        bi_pad = np.zeros((row_pad, n_blks), dtype=np.float16)
        bi_pad[:d_out, :] = bi_f.astype(np.float16)
        bi_u8 = (
            np.ascontiguousarray(bi_pad.reshape(n_tiles, 16, n_blks).transpose(0, 2, 1))
            .view(np.uint8)
            .reshape(n_tiles, n_blks, 32)
        )
        chunks.append(bi_u8)

    idx_u8 = np.asarray(idx, dtype=np.uint8).reshape(d_out, d_in)
    max_lvl = (1 << n_bits) - 1
    if int(idx_u8.max()) > max_lvl:
        raise ValueError(f"idx max {int(idx_u8.max())} > max_level {max_lvl}")
    idx_pad = np.zeros((row_pad, n_blks, bo_cnt, 8), dtype=np.uint8)
    idx_pad[:d_out] = idx_u8.reshape(d_out, n_blks, bo_cnt, 8)
    idx_5d = idx_pad.reshape(n_tiles, 16, n_blks, bo_cnt, 8)

    for p in range(n_bits):
        bits = (idx_5d >> p) & 1
        packed_p = np.packbits(bits, axis=-1, bitorder="big").reshape(n_tiles, 16, n_blks, bo_cnt)
        # [n_tiles, n_blks, bo_cnt, 16] -> [n_tiles, n_blks, bo_cnt * 16]
        plane_u8 = packed_p.transpose(0, 2, 3, 1).reshape(n_tiles, n_blks, bo_cnt * 16)
        chunks.append(plane_u8)

    combined = np.concatenate(chunks, axis=-1)
    return np.ascontiguousarray(combined).tobytes()


def write_t16c_file(
    out_path: Path,
    arch_flags: int,
    norms: list[tuple[str, np.ndarray]],
    mats: list[dict],
) -> None:
    """
    mats entries: dict(
      name, d_out, d_in, block_size, n_bits, is_asym, has_fwht,
      packed_bytes (or sc, bi, idx),
      inv_s (optional 1D float32),
      low_rank (optional dict(su, sv, u_i8, v_i8)),
    )
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pos = 0
        f.write(struct.pack("<4s5I", T16C_MAGIC, T16C_VERSION, arch_flags, len(norms), len(mats), 0))
        pos += 24
        for name, arr in norms:
            nb = name.encode("utf-8")
            a32 = np.ascontiguousarray(arr, dtype="<f4").ravel()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", a32.size))
            f.write(a32.tobytes())
            pos += 2 + len(nb) + 4 + a32.size * 4

        for m in mats:
            nb = m["name"].encode("utf-8")
            d_out = int(m["d_out"])
            d_in = int(m["d_in"])
            bsize = int(m["block_size"])
            nbits = int(m["n_bits"])
            is_asym = bool(m["is_asym"])
            has_fwht = bool(m["has_fwht"])
            inv_s = m.get("inv_s")
            inv_kind = 1 if inv_s is not None else 0
            lr = m.get("low_rank")
            lr_rank = int(lr["rank"]) if lr is not None else 0
            lr_bytes = (
                (d_out * 4 + lr_rank * 4 + d_out * lr_rank + lr_rank * d_in)
                if lr is not None
                else 0
            )
            pbytes = m.get("packed_bytes")
            if pbytes is None:
                pbytes = pack_tile16_bytes(d_out, d_in, bsize, nbits, is_asym, m["sc"], m.get("bi"), m["idx"])
            f.write(
                struct.pack(
                    "<HIIHBBBBHQQ",
                    len(nb),
                    d_out,
                    d_in,
                    bsize,
                    nbits,
                    1 if is_asym else 0,
                    1 if has_fwht else 0,
                    inv_kind,
                    lr_rank,
                    len(pbytes),
                    lr_bytes,
                )
            )
            f.write(nb)
            pos += 34 + len(nb)
            pad = (64 - (pos % 64)) % 64
            if pad:
                f.write(b"\x00" * pad)
                pos += pad
            f.write(pbytes)
            pos += len(pbytes)
            if inv_s is not None:
                iv = np.ascontiguousarray(inv_s, dtype="<f4").ravel()
                assert iv.size == d_in
                f.write(iv.tobytes())
                pos += iv.size * 4
            if lr is not None:
                su = np.ascontiguousarray(lr["su"], dtype="<f4").ravel()
                sv = np.ascontiguousarray(lr["sv"], dtype="<f4").ravel()
                ui = np.ascontiguousarray(lr["u_i8"], dtype=np.int8).ravel()
                vi = np.ascontiguousarray(lr["v_i8"], dtype=np.int8).ravel()
                assert su.size == d_out and sv.size == lr_rank
                assert ui.size == d_out * lr_rank and vi.size == lr_rank * d_in
                f.write(su.tobytes())
                f.write(sv.tobytes())
                f.write(ui.tobytes())
                f.write(vi.tobytes())
                pos += lr_bytes


def export_270m_ckpt(ckpt_path: Path, model_safetensors: Path, out_t16c: Path) -> None:
    import torch
    from safetensors import safe_open

    t0 = time.perf_counter()
    ck = torch.load(str(ckpt_path), weights_only=False)
    ov = ck["ov"]
    mats = []

    with safe_open(str(model_safetensors), framework="pt", device="cpu") as f:
        st_keys = list(f.keys())
        norms = []
        for k in sorted(st_keys):
            if "norm.weight" in k:
                arr = ov.get(k, f.get_tensor(k)).float().cpu().numpy()
                norms.append((k, arr))

    if "projs" in ck:
        for name, d in sorted(ck["projs"].items()):
            idx = d["idx"].cpu().numpy()
            sc = d["sc16"].float().cpu().numpy()
            bi = d["bi16"].float().cpu().numpy()
            inv_s = d["inv_s16"].float().cpu().numpy()
            bsize = int(d["bsize"])
            nbits = int(d["nbits"])
            d_out, n_blks = sc.shape[:2]
            d_in = n_blks * bsize
            mats.append(
                dict(
                    name=name,
                    d_out=d_out,
                    d_in=d_in,
                    block_size=bsize,
                    n_bits=nbits,
                    is_asym=True,
                    has_fwht=True,
                    sc=sc,
                    bi=bi,
                    idx=idx.reshape(d_out, d_in),
                    inv_s=inv_s,
                    low_rank=None,
                )
            )
    elif "qparams" in ck:
        bsize = 128
        for name, (Qr, scx, bix, isx, nbx) in sorted(ck["qparams"].items()):
            d_out, d_in = Qr.shape
            qb = Qr.view(d_out, d_in // bsize, bsize)
            sc0 = scx.detach().clamp(min=1e-6)
            bi0 = bix.detach()
            wn = (qb - bi0) / sc0
            nbits = int(nbx)
            if nbits == 1:
                idx = (wn >= 0.0).to(torch.uint8).cpu().numpy()
            else:
                ml = float((1 << nbits) - 1)
                idx = torch.round(0.5 * (wn + ml)).clamp(0.0, ml).to(torch.uint8).cpu().numpy()
            mats.append(
                dict(
                    name=name,
                    d_out=d_out,
                    d_in=d_in,
                    block_size=bsize,
                    n_bits=nbits,
                    is_asym=True,
                    has_fwht=True,
                    sc=sc0.squeeze(-1).cpu().numpy(),
                    bi=bi0.squeeze(-1).cpu().numpy(),
                    idx=idx.reshape(d_out, d_in),
                    inv_s=isx.float().cpu().numpy(),
                    low_rank=None,
                )
            )
    else:
        raise KeyError("checkpoint has neither 'projs' nor 'qparams'")

    write_t16c_file(out_t16c, arch_flags=0, norms=norms, mats=mats)
    sz = out_t16c.stat().st_size
    print(f"Exported {len(mats)} matrices + {len(norms)} norms to {out_t16c} ({sz/1e6:.2f} MB) in {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export_270m_ckpt(Path(args.ckpt), Path(args.model), Path(args.out))

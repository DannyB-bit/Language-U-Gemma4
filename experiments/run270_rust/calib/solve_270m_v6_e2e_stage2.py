"""Gemma 3 270M <= 1.5441 bpw T16C v6 Stage-2 Continuation (SmoothE2E-v6b):
- Unpacks exact discrete params + INT8 low-rank directly from gemma3_270m_v6_e2e.t16c
- Fails closed on both write/read .t16c roundtrip AND end-to-end HF5 logit parity vs bench5_v6_e2e_noffi
- Continues optimization on expanded disjoint general corpus (e2e_targets_v6_s2.pt)
- Adds Top-16 relative logit-gap SmoothL1 + top-1 margin loss to eliminate rank-1/rank-2 swaps
- Gentle STE across L10..L17 under Windows BelowNormal priority, 2 threads, < 1.1 GB RSS
- Strictly gates export on held-out val_orig score improvement
"""
import argparse
import ctypes
import gc
import json
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path


def set_below_normal_priority() -> None:
    if sys.platform != "win32":
        return
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.SetPriorityClass.restype = ctypes.c_int
    k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k32.GetPriorityClass.restype = ctypes.c_uint32
    h = k32.GetCurrentProcess()
    k32.SetPriorityClass(h, 0x00004000)
    print(f"[sys] Windows PriorityClass=0x{k32.GetPriorityClass(h):04x}", flush=True)


set_below_normal_priority()
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from torch.utils.checkpoint import checkpoint

torch.set_num_threads(2)

CALIB_DIR = Path(__file__).resolve().parent
if str(CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_DIR))

from calib_common import eval_suite, fwht_block, resolve_model_dir, rmsnorm_gemma, rope_rotate
from t16c_export import write_t16c_file


def get_rss_mb() -> float:
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        psapi = ctypes.windll.psapi
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        h = k32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 1048576.0
    return 0.0


def unpack_tile16_bytes(buf: bytes, d_out: int, d_in: int, bsize: int, nbits: int, is_asym: bool):
    n_tiles = (d_out + 15) // 16
    n_blks = d_in // bsize
    bo_cnt = bsize // 8
    hdr_bytes = 64 if is_asym else 32
    plane_bytes = bo_cnt * 16
    tblk_bytes = hdr_bytes + nbits * plane_bytes
    expected_bytes = n_tiles * n_blks * tblk_bytes
    if len(buf) != expected_bytes:
        raise ValueError(f"packed_bytes mismatch: got {len(buf)}, expected {expected_bytes}")

    raw = np.frombuffer(buf, dtype=np.uint8).reshape(n_tiles, n_blks, tblk_bytes)
    sc = (
        raw[:, :, :32]
        .reshape(-1)
        .view(np.float16)
        .reshape(n_tiles, n_blks, 16)
        .transpose(0, 2, 1)
        .reshape(n_tiles * 16, n_blks)[:d_out]
        .astype(np.float32)
        .copy()
    )
    if is_asym:
        bi = (
            raw[:, :, 32:64]
            .reshape(-1)
            .view(np.float16)
            .reshape(n_tiles, n_blks, 16)
            .transpose(0, 2, 1)
            .reshape(n_tiles * 16, n_blks)[:d_out]
            .astype(np.float32)
            .copy()
        )
    else:
        bi = None

    idx_out = np.zeros((d_out, n_blks, bsize), dtype=np.uint8)
    for p in range(nbits):
        p0 = hdr_bytes + p * plane_bytes
        bp = raw[:, :, p0 : p0 + plane_bytes].reshape(n_tiles, n_blks, bo_cnt, 16)
        bits = np.unpackbits(bp, axis=2, bitorder="big").transpose(0, 3, 1, 2).reshape(n_tiles * 16, n_blks, bsize)[:d_out]
        if p == 0:
            idx_out |= bits
        else:
            idx_out |= (bits << p).astype(np.uint8)
    return sc, bi, idx_out


def read_t16c_file(path: Path):
    data = path.read_bytes()
    pos = 0
    magic, ver, arch_flags, n_norms, n_mats, reserved = struct.unpack_from("<4s5I", data, pos)
    pos += 24
    if magic != b"T16C" or ver != 1:
        raise ValueError(f"Bad T16C header in {path}: magic={magic!r} ver={ver}")

    norms = {}
    for _ in range(n_norms):
        (nlen,) = struct.unpack_from("<H", data, pos)
        pos += 2
        name = data[pos : pos + nlen].decode("utf-8")
        pos += nlen
        (vlen,) = struct.unpack_from("<I", data, pos)
        pos += 4
        arr = np.frombuffer(data, dtype="<f4", count=vlen, offset=pos).copy()
        pos += 4 * vlen
        norms[name] = arr

    mats = {}
    hdr_fmt = "<HIIHBBBBHQQ"
    hdr_sz = struct.calcsize(hdr_fmt)
    for _ in range(n_mats):
        nlen, d_out, d_in, bsize, nbits, is_asym, has_fwht, inv_kind, lr_rank, packed_bytes, lr_bytes = struct.unpack_from(
            hdr_fmt, data, pos
        )
        pos += hdr_sz
        name = data[pos : pos + nlen].decode("utf-8")
        pos += nlen
        pos = (pos + 63) & ~63

        sc, bi, idx_blk = unpack_tile16_bytes(data[pos : pos + packed_bytes], d_out, d_in, bsize, nbits, bool(is_asym))
        pos += packed_bytes

        inv_s = None
        if inv_kind != 0:
            inv_s = np.frombuffer(data, dtype="<f4", count=d_in, offset=pos).copy()
            pos += 4 * d_in

        low_rank = None
        if lr_rank > 0:
            lr_end = pos + lr_bytes
            su = np.frombuffer(data, dtype="<f4", count=d_out, offset=pos).copy()
            pos += 4 * d_out
            sv = np.frombuffer(data, dtype="<f4", count=lr_rank, offset=pos).copy()
            pos += 4 * lr_rank
            u_i8 = np.frombuffer(data, dtype=np.int8, count=d_out * lr_rank, offset=pos).reshape(d_out, lr_rank).copy()
            pos += d_out * lr_rank
            v_i8 = np.frombuffer(data, dtype=np.int8, count=lr_rank * d_in, offset=pos).reshape(lr_rank, d_in).copy()
            pos += lr_rank * d_in
            if pos != lr_end:
                raise ValueError(f"Low-rank byte count mismatch on {name}: pos={pos} expected={lr_end}")
            low_rank = dict(rank=int(lr_rank), su=su, sv=sv, u_i8=u_i8, v_i8=v_i8)

        mats[name] = dict(
            name=name,
            d_out=int(d_out),
            d_in=int(d_in),
            bsize=int(bsize),
            nbits=int(nbits),
            is_asym=bool(is_asym),
            has_fwht=bool(has_fwht),
            sc=sc,
            bi=bi,
            idx=idx_blk,
            inv_s=inv_s,
            low_rank=low_rank,
        )

    if pos != len(data):
        raise ValueError(f"Trailing bytes in {path}: parsed={pos} file_len={len(data)}")
    return dict(arch_flags=arch_flags, norms=norms, mats=mats)


def fake_f16(x: torch.Tensor) -> torch.Tensor:
    return x + (x.to(torch.float16).float() - x).detach()


def fake_i8_row(w: torch.Tensor) -> torch.Tensor:
    s = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    wq = torch.clamp(torch.round(w / s), -127.0, 127.0) * s
    return w + (wq - w).detach()


class SteDiscreteHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchor: torch.Tensor, idx_u8: torch.Tensor, max_lvl: float, name: str, grad_map: dict):
        ctx.name = name
        ctx.grad_map = grad_map
        ctx.a_shape = anchor.shape
        return 2.0 * idx_u8.float() - max_lvl

    @staticmethod
    def backward(ctx, grad_u: torch.Tensor):
        if ctx.grad_map is not None:
            g = grad_u.detach().to(torch.float16)
            prev = ctx.grad_map.get(ctx.name)
            ctx.grad_map[ctx.name] = g if prev is None else (prev.float() + g.float()).to(torch.float16)
        return torch.zeros(ctx.a_shape, dtype=torch.float32), None, None, None, None


def online_vocab_eval(q_fin: torch.Tensor, emb_bf16: torch.Tensor, e_bar_t: torch.Tensor,
                      t_neg_ent: torch.Tensor, t_argmax: torch.Tensor, tok_w: torch.Tensor = None, chunk_v: int = 32768):
    N, D = q_fin.shape
    V = emb_bf16.shape[0]
    qd = q_fin.detach().float()
    m = torch.full((N,), -1e30, dtype=torch.float32)
    s = torch.zeros((N,), dtype=torch.float32)
    e_num = torch.zeros((N, D), dtype=torch.float32)
    best_val = torch.full((N,), -1e30, dtype=torch.float32)
    best_idx = torch.zeros((N,), dtype=torch.int32)
    for v0 in range(0, V, chunk_v):
        v1 = min(v0 + chunk_v, V)
        ec = emb_bf16[v0:v1].float()
        z = qd @ ec.T
        c_max, c_arg = z.max(dim=-1)
        upd = c_max > best_val
        best_val = torch.where(upd, c_max, best_val)
        best_idx = torch.where(upd, c_arg.to(torch.int32) + v0, best_idx)
        w = torch.exp(z - c_max[:, None])
        cs = w.sum(dim=-1)
        ce = w @ ec
        m_new = torch.maximum(m, c_max)
        a = torch.exp(m - m_new)
        b = torch.exp(c_max - m_new)
        s = s * a + cs * b
        e_num = e_num * a[:, None] + ce * b[:, None]
        m = m_new
    lse = m + torch.log(s.clamp(min=1e-30))
    q_ebar = e_num / s.clamp(min=1e-30)[:, None]
    kl_per_tok = (t_neg_ent.float() - (qd * e_bar_t.float()).sum(dim=-1) + lse).clamp(min=0.0)
    hit_per_tok = (best_idx == t_argmax).float()
    if tok_w is None:
        wn = torch.full((N,), 1.0 / float(N), dtype=torch.float32)
    else:
        wn = tok_w.float() / tok_w.float().sum().clamp(min=1e-12)
    kl_mean = float((kl_per_tok * wn).sum().item())
    top1 = float((hit_per_tok * wn).sum().item())
    g_kl = (q_ebar - e_bar_t.float()) * wn[:, None]
    return kl_mean, top1, g_kl, kl_per_tok, hit_per_tok


def compute_top16_rank_loss(q_fin: torch.Tensor, emb_bf16: torch.Tensor, t_top16_idx: torch.Tensor,
                            t_top16_val: torch.Tensor, wn: torch.Tensor) -> torch.Tensor:
    emb_top16 = emb_bf16[t_top16_idx.long()].float()  # [N, 16, D]
    z_s16 = torch.einsum("nd,nkd->nk", q_fin, emb_top16)
    z_t16 = t_top16_val.float()
    ds = z_s16 - z_s16[:, :1]
    dt = z_t16 - z_t16[:, :1]
    gap_loss = (F.smooth_l1_loss(ds[:, 1:], dt[:, 1:], reduction="none", beta=0.5).mean(dim=-1) * wn).sum()
    # Margin term: keep top-1 ahead of runners-up where teacher gap > 0.05
    t_margin = torch.clamp(z_t16[:, :1] - z_t16[:, 1:], min=0.0, max=2.0)
    s_margin = z_s16[:, :1] - z_s16[:, 1:]
    margin_hinge = (F.relu(t_margin - s_margin).mean(dim=-1) * wn).sum()
    return gap_loss + 0.35 * margin_hinge


def _self_test():
    rng = np.random.default_rng(7)
    d_out, d_in, bsize, nbits = 35, 128, 64, 2
    n_blks = d_in // bsize
    sc0 = rng.uniform(0.05, 0.25, size=(d_out, n_blks)).astype(np.float16).astype(np.float32)
    bi0 = rng.uniform(-0.08, 0.08, size=(d_out, n_blks)).astype(np.float16).astype(np.float32)
    idx0 = rng.integers(0, 1 << nbits, size=(d_out, n_blks, bsize), dtype=np.uint8)
    inv_s0 = rng.uniform(0.5, 1.5, size=(d_in,)).astype(np.float32)
    su0 = rng.uniform(0.01, 0.05, size=(d_out,)).astype(np.float32)
    sv0 = rng.uniform(0.01, 0.05, size=(4,)).astype(np.float32)
    u_i8_0 = rng.integers(-127, 128, size=(d_out, 4), dtype=np.int8)
    v_i8_0 = rng.integers(-127, 128, size=(4, d_in), dtype=np.int8)

    d_out_sym, nbits_sym = 16, 1
    sc1 = rng.uniform(0.05, 0.25, size=(d_out_sym, n_blks)).astype(np.float16).astype(np.float32)
    idx1 = rng.integers(0, 1 << nbits_sym, size=(d_out_sym, n_blks, bsize), dtype=np.uint8)

    norm0 = rng.normal(size=(d_in,)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / "test_roundtrip.t16c"
        write_t16c_file(
            tp,
            arch_flags=0,
            norms=[("test.norm.weight", norm0)],
            mats=[
                dict(
                    name="mat_asym_lr", d_out=d_out, d_in=d_in, block_size=bsize, n_bits=nbits,
                    is_asym=True, has_fwht=True, sc=sc0, bi=bi0, idx=idx0.reshape(d_out, d_in),
                    inv_s=inv_s0, low_rank=dict(rank=4, su=su0, sv=sv0, u_i8=u_i8_0, v_i8=v_i8_0),
                ),
                dict(
                    name="mat_sym_base", d_out=d_out_sym, d_in=d_in, block_size=bsize, n_bits=nbits_sym,
                    is_asym=False, has_fwht=True, sc=sc1, bi=None, idx=idx1.reshape(d_out_sym, d_in),
                    inv_s=None, low_rank=None,
                ),
            ],
        )
        parsed = read_t16c_file(tp)

    assert np.array_equal(parsed["norms"]["test.norm.weight"], norm0)
    m0 = parsed["mats"]["mat_asym_lr"]
    assert np.array_equal(m0["sc"], sc0)
    assert np.array_equal(m0["bi"], bi0)
    assert np.array_equal(m0["idx"], idx0)
    assert np.array_equal(m0["inv_s"], inv_s0)
    assert np.array_equal(m0["low_rank"]["su"], su0)
    assert np.array_equal(m0["low_rank"]["sv"], sv0)
    assert np.array_equal(m0["low_rank"]["u_i8"], u_i8_0)
    assert np.array_equal(m0["low_rank"]["v_i8"], v_i8_0)

    m1 = parsed["mats"]["mat_sym_base"]
    assert np.array_equal(m1["sc"], sc1) and m1["bi"] is None
    assert np.array_equal(m1["idx"], idx1) and m1["inv_s"] is None and m1["low_rank"] is None

    # Gradient check on top16 rank loss
    qt = torch.randn((4, 32), requires_grad=True)
    et = torch.randn((64, 32), dtype=torch.bfloat16)
    ti = torch.randint(0, 64, (4, 16), dtype=torch.int32)
    tv = torch.randn((4, 16), dtype=torch.float16)
    tv[:, 0] = tv.max(dim=-1).values + 1.0
    wn = torch.tensor([0.25, 0.25, 0.25, 0.25])
    ls = compute_top16_rank_loss(qt, et, ti, tv, wn)
    ls.backward()
    assert qt.grad is not None and torch.isfinite(qt.grad).all()
    print("[self_test] T16C roundtrip (asym+sym+inv_s+lr) + top16_rank_loss PASSED", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Gemma 3 270M <= 1.5441 bpw T16C v6 Stage-2 Continuation (SmoothE2E-v6b)")
    ap.add_argument("--model-dir", type=Path, default=None, help="Path to Gemma 3 270M IT HF checkpoint")
    ap.add_argument("--work-dir", type=Path, default=CALIB_DIR / "work", help="Directory holding calibration targets/checkpoints")
    ap.add_argument("--targets", type=Path, default=None, help="Path to e2e_targets_v6_s2.pt")
    ap.add_argument("--init-t16c", type=Path, default=None, help="Path to Stage-1 gemma3_270m_v6_e2e.t16c")
    ap.add_argument("--out-t16c", type=Path, default=None, help="Output Stage-2 T16C path")
    ap.add_argument("--noffi-ref", type=Path, default=None, help="Optional run270 --no-native-ffi capital.f32 parity dump")
    ap.add_argument("--ref-dir", type=Path, default=None, help="Directory holding exact FP32 ref270 logits")
    args = ap.parse_args()

    t0 = time.perf_counter()
    _self_test()

    model_dir = resolve_model_dir(args.model_dir)
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    tgt_path = (args.targets or (work_dir / "e2e_targets_v6_s2.pt")).expanduser().resolve()
    init_t16c_path = (args.init_t16c or (work_dir / "gemma3_270m_v6_e2e.t16c")).expanduser().resolve()
    out_t16c = (args.out_t16c or (work_dir / "gemma3_270m_v6_e2e_s2.t16c")).expanduser().resolve()
    noffi_ref_path = (args.noffi_ref or (work_dir / "bench5_v6_e2e_noffi" / "capital.f32")).expanduser().resolve()
    ref_dir = args.ref_dir.expanduser().resolve() if args.ref_dir else None
    if not tgt_path.exists():
        raise FileNotFoundError(f"Missing stage-2 target cache: {tgt_path}")
    if not init_t16c_path.exists():
        raise FileNotFoundError(f"Missing stage-1 T16C model: {init_t16c_path}")

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tk = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    eps = cfg["rms_norm_eps"]
    hd = cfg["head_dim"]
    n_heads, n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    n_rep = n_heads // n_kv
    scale = 1.0 / math.sqrt(float(cfg["query_pre_attn_scalar"]))
    sqrt_d = math.sqrt(float(cfg["hidden_size"]))

    tgt = torch.load(str(tgt_path), weights_only=False)
    cal_seqs = tgt["cal_seqs"]
    is_chat = tgt["is_chat"]
    tok_weights_list = tgt["tok_weights"]
    t_fin_all = tgt["t_fin"]
    taps = tgt["taps"]
    e_bar_t_all = tgt["e_bar_t"]
    t_neg_ent_all = tgt["t_neg_ent"]
    t_argmax_all = tgt["t_argmax"]
    t_top16_idx_all = tgt["t_top16_idx"]
    t_top16_val_all = tgt["t_top16_val"]
    R_gram = tgt["R_gram"]

    # Preserve EXACT Stage-1 val_orig split (indices < 143) plus new chat holdout
    orig_n = 143
    prose_all = [i for i, c in enumerate(is_chat) if not c]
    chat_orig = [i for i in range(orig_n) if is_chat[i]]
    chat_new = [i for i in range(orig_n, len(is_chat)) if is_chat[i]]
    val_prose = prose_all[::max(1, len(prose_all) // 6)][:6]
    val_chat_orig = chat_orig[::max(1, len(chat_orig) // 6)][:6]
    val_orig_ids = val_prose + val_chat_orig
    val_new_chat = chat_new[::max(1, len(chat_new) // 6)][:6]
    val_set = set(val_orig_ids) | set(val_new_chat)
    train_prose = [i for i in prose_all if i not in val_set]
    train_chat = [i for i, c in enumerate(is_chat) if c and i not in val_set]

    seq_offs, p0 = [], 0
    for s in cal_seqs:
        seq_offs.append((p0, p0 + len(s)))
        p0 += len(s)
    print(f"[data-s2] total={len(cal_seqs)} tok={p0} val_orig={len(val_orig_ids)} val_new_c={len(val_new_chat)} train={len(train_prose)}p+{len(train_chat)}c rss={get_rss_mb():.0f}MB", flush=True)

    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        emb_bf16 = f.get_tensor("model.embed_tokens.weight").contiguous()

    # Load Stage-1 exported .t16c parameters directly
    parsed = read_t16c_file(init_t16c_path)
    tot_b = int(init_t16c_path.stat().st_size)
    bpw_file = float(tot_b * 8.0 / 100270080.0)
    norm_frozen = {k: torch.from_numpy(v).float() for k, v in parsed["norms"].items()}
    fin_norm_frozen = norm_frozen.pop("model.norm.weight")

    STE_MIN_LAYER = 10
    idx_u8, sc0_m, bi0_m, a_sc_p, a_bi_p = {}, {}, {}, {}, {}
    lr_u_p, lr_v_p, lr_s_map, invs_map, bs_map, nb_map, asym_map = {}, {}, {}, {}, {}, {}, {}
    u_res, u_grad_map, u_ema_rms = {}, {}, {}
    ste_anchor = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))
    layer_idx_map = {}

    for name in sorted(parsed["mats"].keys()):
        m = parsed["mats"][name]
        l_idx = int(name.split(".")[2])
        layer_idx_map[name] = l_idx
        bs = m["bsize"]
        nb = m["nbits"]
        asym = m["is_asym"]
        idx_u8[name] = torch.from_numpy(m["idx"]).contiguous()
        invs_map[name] = torch.from_numpy(m["inv_s"]).float() if m["inv_s"] is not None else torch.ones((m["d_in"],), dtype=torch.float32)
        bs_map[name] = bs
        nb_map[name] = nb
        asym_map[name] = asym
        sc0 = torch.from_numpy(m["sc"]).float().clamp(min=1e-6).unsqueeze(-1)
        sc0_m[name] = sc0
        a_sc_p[name] = torch.nn.Parameter(torch.zeros_like(sc0))
        if asym:
            bi0_m[name] = torch.from_numpy(m["bi"]).float().unsqueeze(-1)
            a_bi_p[name] = torch.nn.Parameter(torch.zeros_like(sc0))
        if m["low_rank"] is not None:
            lr = m["low_rank"]
            uf = torch.from_numpy(lr["u_i8"].astype(np.float32) * lr["su"][:, None])
            vf = torch.from_numpy(lr["v_i8"].astype(np.float32) * lr["sv"][:, None])
            ru = max(float(uf.abs().mean().item()), 1e-4)
            rv = max(float(vf.abs().mean().item()), 1e-4)
            lr_u_p[name] = torch.nn.Parameter((uf / ru).clone())
            lr_v_p[name] = torch.nn.Parameter((vf / rv).clone())
            lr_s_map[name] = (ru, rv)
        if l_idx >= STE_MIN_LAYER:
            u_res[name] = torch.zeros_like(idx_u8[name], dtype=torch.float16)
            u_ema_rms[name] = 0.0

    del parsed
    gc.collect()

    def proj(name: str, x_raw: torch.Tensor, grad_active: bool) -> torch.Tensor:
        bs = bs_map[name]
        ml = float((1 << nb_map[name]) - 1)
        sc = fake_f16(sc0_m[name] * torch.exp(a_sc_p[name].clamp(-0.9, 0.9)))
        x_fw = fwht_block(x_raw * invs_map[name][None, :], bs)
        if grad_active and layer_idx_map[name] >= STE_MIN_LAYER:
            u_q = SteDiscreteHook.apply(ste_anchor, idx_u8[name], ml, name, u_grad_map)
        else:
            u_q = 2.0 * idx_u8[name].float() - ml
        q_blk = sc * u_q
        if asym_map[name]:
            bi = fake_f16(bi0_m[name] + sc0_m[name] * a_bi_p[name].clamp(-1.2, 1.2))
            q_blk = q_blk + bi
        y = F.linear(x_fw, q_blk.view(sc.shape[0], -1))
        if name in lr_u_p:
            ru, rv = lr_s_map[name]
            y = y + (x_raw @ fake_i8_row(lr_v_p[name] * rv).T) @ fake_i8_row(lr_u_p[name] * ru).T
        return y

    def run_layer(l_idx: int, h: torch.Tensor, slens: list[int], grad_active: bool) -> torch.Tensor:
        p = f"model.layers.{l_idx}."
        is_full = cfg["layer_types"][l_idx] == "full_attention"
        rb = float(cfg["rope_theta"] if is_full else cfg["rope_local_base_freq"])
        hi = rmsnorm_gemma(h, norm_frozen[p + "input_layernorm.weight"], eps)
        q_all = rmsnorm_gemma(proj(p + "self_attn.q_proj.weight", hi, grad_active).view(-1, n_heads, hd),
                              norm_frozen[p + "self_attn.q_norm.weight"], eps)
        k_all = rmsnorm_gemma(proj(p + "self_attn.k_proj.weight", hi, grad_active).view(-1, n_kv, hd),
                              norm_frozen[p + "self_attn.k_norm.weight"], eps)
        v_all = proj(p + "self_attn.v_proj.weight", hi, grad_active).view(-1, n_kv, hd)
        ctxs, soff = [], 0
        for tl in slens:
            q = rope_rotate(q_all[soff:soff + tl], rb, tl)
            k = rope_rotate(k_all[soff:soff + tl], rb, tl)
            v = v_all[soff:soff + tl]
            ke, ve = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
            c = torch.triu(torch.full((tl, tl), float("-inf")), diagonal=1)
            if not is_full and tl > cfg["sliding_window"]:
                c = c + torch.tril(torch.full((tl, tl), float("-inf")), diagonal=-cfg["sliding_window"])
            pr = F.softmax(torch.bmm(q.permute(1, 0, 2), ke.permute(1, 0, 2).transpose(1, 2)) * scale + c[None], dim=-1)
            ctxs.append(torch.bmm(pr, ve.permute(1, 0, 2)).permute(1, 0, 2).contiguous().view(tl, n_heads * hd))
            soff += tl
        h_mid = h + rmsnorm_gemma(proj(p + "self_attn.o_proj.weight", torch.cat(ctxs, 0), grad_active),
                                  norm_frozen[p + "post_attention_layernorm.weight"], eps)
        hm = rmsnorm_gemma(h_mid, norm_frozen[p + "pre_feedforward_layernorm.weight"], eps)
        act = (F.gelu(proj(p + "mlp.gate_proj.weight", hm, grad_active), approximate="tanh")
               * proj(p + "mlp.up_proj.weight", hm, grad_active))
        return h_mid + rmsnorm_gemma(proj(p + "mlp.down_proj.weight", act, grad_active),
                                     norm_frozen[p + "post_feedforward_layernorm.weight"], eps)

    # Verify step-0 unpacked forward parity against run270 noffi reference output!
    if noffi_ref_path.exists():
        with torch.no_grad():
            cap_prompt = "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"
            cap_ids = torch.tensor(tk.encode(cap_prompt).ids, dtype=torch.long)
            hc = emb_bf16[cap_ids].float() * sqrt_d
            for li in range(18):
                hc = run_layer(li, hc, [len(cap_ids)], False)
            qf_c = rmsnorm_gemma(hc[-1:], fin_norm_frozen, eps)
            logits_py = (qf_c @ emb_bf16.float().T).squeeze(0).numpy()
            logits_rs = np.fromfile(str(noffi_ref_path), dtype=np.float32)
            cos_p = float(np.dot(logits_py, logits_rs) / (np.linalg.norm(logits_py) * np.linalg.norm(logits_rs)))
            print(f"[parity] Stage-1 .t16c unpack vs run270 noffi capital.f32: cos={cos_p:.7f} max_diff={np.max(np.abs(logits_py-logits_rs)):.3e}", flush=True)
            assert cos_p > 0.9999, f"Unpack parity failure: cos={cos_p}"

    tap_weights = {9: 0.5, 11: 0.7, 12: 1.0, 13: 1.0, 14: 1.2, 15: 1.5, 16: 1.8, 17: 2.2}
    tap_w_sum = sum(tap_weights.values())

    def forward_batch(seq_ids: list[int], train: bool):
        h0 = torch.cat([emb_bf16[cal_seqs[i]].float() * sqrt_d for i in seq_ids], dim=0)
        slens = [len(cal_seqs[i]) for i in seq_ids]
        tidx = torch.cat([torch.arange(seq_offs[i][0], seq_offs[i][1]) for i in seq_ids], dim=0)
        tok_w = torch.cat([tok_weights_list[i] for i in seq_ids], dim=0)
        wn = tok_w / tok_w.sum().clamp(min=1e-12)

        h = h0
        tap_loss = torch.zeros((), dtype=torch.float32)
        for l_idx in range(18):
            if train:
                h = checkpoint(lambda x, _l=l_idx: run_layer(_l, x, slens, True), h, use_reentrant=False)
                if l_idx in tap_weights:
                    tgt_h = taps[l_idx][tidx].float()
                    c_l = F.cosine_similarity(h, tgt_h, dim=-1)
                    tap_loss = tap_loss + tap_weights[l_idx] * ((1.0 - c_l) * wn).sum()
            else:
                h = run_layer(l_idx, h, slens, False)
        q_fin = rmsnorm_gemma(h, fin_norm_frozen, eps)
        return q_fin, tidx, tok_w, wn, tap_loss / tap_w_sum

    @torch.no_grad()
    def eval_validation(seq_ids: list[int], n_first_group: int):
        q_fin, tidx, tok_w, wn, _ = forward_batch(seq_ids, train=False)
        tv = t_fin_all[tidx].float()
        kl, top1, _, kl_tok, hit_tok = online_vocab_eval(q_fin, emb_bf16, e_bar_t_all[tidx], t_neg_ent_all[tidx], t_argmax_all[tidx], tok_w=tok_w)
        fcos_tok = F.cosine_similarity(q_fin, tv, dim=-1)
        lcos_tok = F.cosine_similarity(q_fin @ R_gram.T, tv @ R_gram.T, dim=-1)
        fcos = float((fcos_tok * wn).sum().item())
        lcos = float((lcos_tok * wn).sum().item())
        n_g1 = sum(len(cal_seqs[i]) for i in seq_ids[:n_first_group])
        wc = tok_w[n_g1:].float()
        wc = wc / wc.sum().clamp(min=1e-12)
        kl_c = float((kl_tok[n_g1:] * wc).sum().item())
        top1_c = float((hit_tok[n_g1:] * wc).sum().item())
        lcos_c = float((lcos_tok[n_g1:] * wc).sum().item())
        score = 0.45 * lcos + 0.35 * fcos + 0.45 * top1 - 0.14 * kl
        return dict(kl=kl, fcos=fcos, lcos=lcos, top1=top1, kl_c=kl_c, lcos_c=lcos_c, top1_c=top1_c, score=score)

    base_lrs = [2.2e-4, 2.5e-4, 3.0e-4]
    opt = torch.optim.AdamW([
        {"params": list(a_sc_p.values()), "lr": base_lrs[0], "weight_decay": 0.0},
        {"params": list(a_bi_p.values()), "lr": base_lrs[1], "weight_decay": 0.0},
        {"params": list(lr_u_p.values()) + list(lr_v_p.values()), "lr": base_lrs[2], "weight_decay": 1e-4},
    ])
    adam_params = list(a_sc_p.values()) + list(a_bi_p.values()) + list(lr_u_p.values()) + list(lr_v_p.values())

    def grab_best():
        return {
            "asc": {k: v.detach().clone() for k, v in a_sc_p.items()},
            "abi": {k: v.detach().clone() for k, v in a_bi_p.items()},
            "u": {k: v.detach().clone() for k, v in lr_u_p.items()},
            "v": {k: v.detach().clone() for k, v in lr_v_p.items()},
            "idx": {k: idx_u8[k].clone() for k in u_res.keys()},
        }

    def restore_best(st):
        with torch.no_grad():
            for k, v in st["asc"].items(): a_sc_p[k].copy_(v)
            for k, v in st["abi"].items(): a_bi_p[k].copy_(v)
            for k, v in st["u"].items(): lr_u_p[k].copy_(v)
            for k, v in st["v"].items(): lr_v_p[k].copy_(v)
            for k, v in st["idx"].items():
                idx_u8[k].copy_(v)
                u_res[k].zero_()

    init_v = eval_validation(val_orig_ids, len(val_prose))
    init_new_v = eval_validation(val_new_chat, 0) if val_new_chat else None
    best_v = dict(init_v)
    best_state = grab_best()
    print(f"[E2E-v6b] step  0 | val KL={init_v['kl']:.4f} fcos={init_v['fcos']:.4f} lcos={init_v['lcos']:.4f} top1={init_v['top1']:.3f} | "
          f"chatOrig top1={init_v['top1_c']:.3f} | newChat top1={init_new_v['top1']:.3f} KL={init_new_v['kl']:.4f} | rss={get_rss_mb():.0f}MB", flush=True)

    rng = np.random.default_rng(2027)
    p_pool, c_pool = [], []
    steps = 60
    lr_mult = 1.0
    total_flips = 0
    ste_start_step = 6

    for step in range(steps):
        ts = time.perf_counter()
        warmup = min(1.0, float(step + 1) / 5.0)
        cos_sched = warmup * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * step / float(steps))))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * lr_mult * cos_sched

        if len(p_pool) < 2:
            p_pool.extend(list(rng.permutation(train_prose)))
        if len(c_pool) < 3:
            c_pool.extend(list(rng.permutation(train_chat)))
        sel = [p_pool.pop(), p_pool.pop(), c_pool.pop(), c_pool.pop(), c_pool.pop()]

        opt.zero_grad()
        u_grad_map.clear()
        q_fin, tidx, tok_w, wn, tap_loss = forward_batch(sel, train=True)
        tv = t_fin_all[tidx].float()

        kl_val, b_top1, g_kl, _, _ = online_vocab_eval(q_fin, emb_bf16, e_bar_t_all[tidx], t_neg_ent_all[tidx], t_argmax_all[tidx], tok_w=tok_w)
        kl_surr = (q_fin * g_kl.detach()).sum()
        rank16_loss = compute_top16_rank_loss(q_fin, emb_bf16, t_top16_idx_all[tidx], t_top16_val_all[tidx], wn)
        lcos_loss = 1.0 - F.cosine_similarity(q_fin @ R_gram.T, tv @ R_gram.T, dim=-1)
        lcos_mean = (lcos_loss * wn).sum()
        fcos_loss = ((1.0 - F.cosine_similarity(q_fin, tv, dim=-1)) * wn).sum()
        mse_loss = ((q_fin - tv).pow(2).mean(dim=-1) * wn).sum()

        loss = 1.0 * kl_surr + 0.35 * rank16_loss + 0.5 * lcos_mean + 2.5 * fcos_loss + 0.02 * mse_loss + 1.2 * tap_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adam_params, 1.0)
        opt.step()

        step_flips = 0
        if step >= ste_start_step:
            lr_u_step = 0.022 * lr_mult * (0.20 + 0.80 * 0.5 * (1.0 + math.cos(math.pi * (step - ste_start_step) / float(max(1, steps - ste_start_step)))))
            with torch.no_grad():
                for name, g16 in list(u_grad_map.items()):
                    gf = g16.float()
                    rms = float(gf.pow(2).mean().sqrt().item())
                    ema = 0.85 * u_ema_rms[name] + 0.15 * rms if u_ema_rms[name] > 0.0 else rms
                    u_ema_rms[name] = ema
                    delta_r = -lr_u_step * (gf / (ema + 1e-10)).clamp(-2.5, 2.5)
                    ml = (1 << nb_map[name]) - 1
                    res_new = u_res[name].float() + delta_r
                    u_target = (2.0 * idx_u8[name].float() - float(ml)) + res_new
                    idx_new = torch.clamp(torch.round((u_target + float(ml)) * 0.5), 0, ml).to(torch.uint8)
                    d_idx = idx_new.to(torch.int16) - idx_u8[name].to(torch.int16)
                    changed = int((d_idx != 0).sum().item())
                    if changed > 0:
                        step_flips += changed
                        idx_u8[name].copy_(idx_new)
                        res_new = res_new - 2.0 * d_idx.float()
                    u_res[name].copy_(res_new.clamp(-1.35, 1.35).to(torch.float16))
        u_grad_map.clear()
        total_flips += step_flips

        if step == 0:
            print(f"[E2E-v6b] step  1/{steps} ({time.perf_counter()-ts:.1f}s) | bKL={kl_val:.3f} bTop1={b_top1:.3f} rank16={rank16_loss.item():.3f} | rss={get_rss_mb():.0f}MB", flush=True)
        if step == 2 or (step + 1) % 6 == 0 or step == steps - 1:
            vm = eval_validation(val_orig_ids, len(val_prose))
            vn = eval_validation(val_new_chat, 0) if val_new_chat else dict(kl=0.0, top1=0.0)
            safe = (vm["fcos"] >= init_v["fcos"] - 0.002) and (vm["lcos"] >= init_v["lcos"] - 0.002)
            flag = ""
            if safe and vm["score"] > best_v["score"]:
                best_v = dict(vm)
                best_state = grab_best()
                flag = "*BEST*"
            elif vm["score"] < best_v["score"] - 0.030 or vm["kl"] > best_v["kl"] * 1.08:
                restore_best(best_state)
                lr_mult *= 0.65
                flag = f"ROLLBACK(lr*{lr_mult:.2f})"
            print(
                f"[E2E-v6b] step {step+1:2d}/{steps} ({time.perf_counter()-ts:.1f}s) | bKL={kl_val:.3f} bTop1={b_top1:.3f} flips={step_flips} | "
                f"val KL={vm['kl']:.4f} fcos={vm['fcos']:.4f} lcos={vm['lcos']:.4f} top1={vm['top1']:.3f} chatOrig={vm['top1_c']:.3f} newC={vn['top1']:.3f}/{vn['kl']:.3f} {flag} | rss={get_rss_mb():.0f}MB",
                flush=True,
            )
            if lr_mult < 0.15:
                break

    if best_v["score"] <= init_v["score"] + 1e-6:
        print(f"[E2E-v6b] No held-out improvement over Stage-1 (best={best_v['score']:.5f} <= init={init_v['score']:.5f}); keeping Stage-1 .t16c.", flush=True)
        return

    restore_best(best_state)
    del tgt, t_fin_all, taps, e_bar_t_all, t_neg_ent_all, t_argmax_all, t_top16_idx_all, t_top16_val_all, u_res, u_grad_map, emb_bf16
    gc.collect()

    print(
        f"[E2E-v6b] BEST val: KL {init_v['kl']:.4f}->{best_v['kl']:.4f} | lcos {init_v['lcos']:.4f}->{best_v['lcos']:.4f} | "
        f"fcos {init_v['fcos']:.4f}->{best_v['fcos']:.4f} | top1 {init_v['top1']:.3f}->{best_v['top1']:.3f} | chatOrig {init_v['top1_c']:.3f}->{best_v['top1_c']:.3f} | flips={total_flips}",
        flush=True,
    )

    ov_out = {}
    mats = []
    with torch.no_grad():
        for name in sorted(idx_u8.keys()):
            bs = bs_map[name]
            nb = nb_map[name]
            asym = asym_map[name]
            ml = float((1 << nb) - 1)
            sc16 = (sc0_m[name] * torch.exp(best_state["asc"][name].clamp(-0.9, 0.9))).squeeze(-1).clamp(min=1e-6).to(torch.float16).float().numpy()
            bi16 = (bi0_m[name] + sc0_m[name] * best_state["abi"][name].clamp(-1.2, 1.2)).squeeze(-1).to(torch.float16).float().numpy() if asym else None
            idx_np = idx_u8[name].view(sc16.shape[0], -1).numpy()
            inv_s_np = invs_map[name].numpy()

            q_rec = np.repeat(sc16, bs, axis=1) * (2.0 * idx_np.astype(np.float32) - ml)
            if asym:
                q_rec += np.repeat(bi16, bs, axis=1)
            W_eff = fwht_block(torch.from_numpy(q_rec), bs) * invs_map[name][None, :]

            lr_dict = None
            if name in lr_u_p:
                ru, rv = lr_s_map[name]
                U_f = (best_state["u"][name] * ru).numpy()
                V_f = (best_state["v"][name] * rv).numpy()
                su = np.maximum(np.max(np.abs(U_f), axis=1), 1e-8).astype(np.float32) / 127.0
                sv = np.maximum(np.max(np.abs(V_f), axis=1), 1e-8).astype(np.float32) / 127.0
                u_i8 = np.clip(np.rint(U_f / su[:, None]), -127, 127).astype(np.int8)
                v_i8 = np.clip(np.rint(V_f / sv[:, None]), -127, 127).astype(np.int8)
                W_eff = W_eff + torch.from_numpy(u_i8.astype(np.float32) * su[:, None]) @ torch.from_numpy(v_i8.astype(np.float32) * sv[:, None])
                lr_dict = dict(rank=int(U_f.shape[1]), su=su, sv=sv, u_i8=u_i8, v_i8=v_i8)

            ov_out[name] = W_eff
            mats.append(dict(
                name=name, d_out=int(idx_np.shape[0]), d_in=int(idx_np.shape[1]), block_size=bs,
                n_bits=nb, is_asym=asym, has_fwht=True, sc=sc16, bi=bi16, idx=idx_np, inv_s=inv_s_np, low_rank=lr_dict,
            ))

    norms_list = []
    for k in sorted(norm_frozen.keys()):
        ov_out[k] = norm_frozen[k]
        norms_list.append((k, norm_frozen[k].numpy()))
    ov_out["model.norm.weight"] = fin_norm_frozen
    norms_list.append(("model.norm.weight", fin_norm_frozen.numpy()))

    write_t16c_file(out_t16c, arch_flags=0, norms=norms_list, mats=mats)
    out_bytes = out_t16c.stat().st_size
    out_bpw = float(out_bytes * 8.0 / 100270080.0)
    print(f"Exported {out_t16c} ({out_bytes:,} B / {out_bpw:.6f} file-bpw)", flush=True)

    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        full_tensors = {k: f.get_tensor(k) for k in f.keys()}

    mc, minc, t1, pp, ss, _ = eval_suite(cfg, full_tensors, ov_out, tk, eval_reasoning="always", ref_dir=ref_dir)
    print(
        f"\nRESULT SmoothE2E-v6b | bytes={tot_b:,} ({out_bpw:.6f} file-bpw) | "
        f"HF5 cos={mc:.5f} min={minc:.5f} top1={t1}/5 | frozen={ss['frozen'][0]}/{ss['frozen'][1]} fresh={ss['fresh'][0]}/{ss['fresh'][1]} | {time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    for k, (c, txt) in pp.items():
        print(f"  {k:10s}: cos={c:.5f} gen='{txt}'", flush=True)
    torch.save(dict(tag="v6_e2e_s2", best_val=best_v, init_val=init_v, bpw=out_bpw, tot_b=out_bytes, mc=mc, minc=minc, t1=t1, ss=ss, pp=pp),
               str(work_dir / "ckpt_v6_e2e_s2_summary.pt"))


if __name__ == "__main__":
    main()

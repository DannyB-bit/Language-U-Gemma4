import argparse
import ctypes, gc, json, math, os, pickle, platform, struct, subprocess, time
from pathlib import Path
import numpy as np
import scipy.linalg
from tokenizers import Tokenizer

try:
    import resource
except ImportError:  # Windows
    resource = None

SCRIPT_DIR = Path(__file__).resolve().parent
T16C_MAGIC = b"T16C"
T16C_VERSION = 1
ARCH_FLAGS = 3  # ARCH_FLAG_GEMMA4 | ARCH_FLAG_TIED_HEAD

lib = None
lfast = None


def _resolve_model_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env = os.environ.get("GEMMA4_31B_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    default_pi4 = Path("/mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct")
    if default_pi4.exists():
        return default_pi4
    raise FileNotFoundError("Gemma 4 31B model directory not configured; pass --model-dir or set GEMMA4_31B_DIR.")


def _init_native_libs(compile_dir: Path, mcpu: str = "cortex-a72") -> None:
    global lib, lfast
    if lib is not None and lfast is not None:
        return
    compile_dir.mkdir(parents=True, exist_ok=True)
    lib_path = compile_dir / "libgptq_fast6.so"
    fast_path = compile_dir / "libfast_sgemm.so"
    arch_flags = [f"-mcpu={mcpu}"] if (mcpu and platform.machine().lower() in ("aarch64", "arm64")) else ["-march=native"]
    if not lib_path.exists():
        subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", *arch_flags, "-fopenmp", str(SCRIPT_DIR / "gptq_fast6.c"), "-o", str(lib_path), "-lm"])
    if not fast_path.exists():
        subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", *arch_flags, "-fopenmp", str(SCRIPT_DIR / "fast_sgemm.c"), "-o", str(fast_path), "-lm"])
    lib = ctypes.CDLL(str(lib_path))
    lfast = ctypes.CDLL(str(fast_path))
    lfast.fast_sgemm_nt_f32.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3 + [ctypes.c_float]*2
    lib.fwht_rows_f32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.quant_init_blocks_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.gptq_intrablock_neon.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.trailing_update_neon.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.dequant_blocks_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.unpack_qat_rows_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.sgemm_nt_f32.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3 + [ctypes.c_float]*2
    lib.sgemm_tn_f32.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3 + [ctypes.c_float]*2
    lib.build_gptq_u_f32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p]
    lib.pack_tile16_bytes_c.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*5
    lib.sub_inplace_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


def rss_mb():
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class SafeSeekReader:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        hlen = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(hlen).decode("utf-8"))
        self.base = 8 + hlen

    def read_raw(self, key: str):
        info = self.hdr[key]
        o0, o1 = info["data_offsets"]
        self.f.seek(self.base + o0)
        return info["dtype"], info["shape"], self.f.read(o1 - o0)

    def read_bf16_f32(self, key: str) -> np.ndarray:
        dt, shp, raw = self.read_raw(key)
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shp).copy()
        assert dt == "BF16"
        return (np.frombuffer(raw, dtype=np.uint16).reshape(shp).astype(np.uint32) << 16).view(np.float32).copy()

    def read_bf16_rows(self, key: str, r0: int, r1: int) -> np.ndarray:
        info = self.hdr[key]
        shp = info["shape"]
        o0, _ = info["data_offsets"]
        rb = shp[1] * 2
        self.f.seek(self.base + o0 + r0 * rb)
        raw = self.f.read((r1 - r0) * rb)
        return (np.frombuffer(raw, dtype=np.uint16).reshape(r1 - r0, shp[1]).astype(np.uint32) << 16).view(np.float32).copy()

    def read_qat_f32(self, pfx: str) -> np.ndarray:
        _, shp_p, raw_p = self.read_raw(pfx + ".weight_packed")
        _, _, raw_s = self.read_raw(pfx + ".weight_scale")
        pk = np.frombuffer(raw_p, dtype=np.int32)
        sc = np.frombuffer(raw_s, dtype=np.uint16)
        rows, d_in = shp_p[0], shp_p[1] * 8
        out = np.empty((rows, d_in), dtype=np.float32)
        lib.unpack_qat_rows_f32(pk.ctypes.data, sc.ctypes.data, out.ctypes.data, rows, d_in)
        return out

    def close(self):
        self.f.close()


def fast_nt(A, B):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.empty((A.shape[0], B.shape[0]), dtype=np.float32)
    lfast.fast_sgemm_nt_f32(A.ctypes.data, B.ctypes.data, C.ctypes.data, A.shape[0], B.shape[0], A.shape[1], 1.0, 0.0)
    return C


def fast_tn(A, B):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.empty((A.shape[1], B.shape[1]), dtype=np.float32)
    lib.sgemm_tn_f32(A.ctypes.data, B.ctypes.data, C.ctypes.data, A.shape[1], B.shape[1], A.shape[0], 1.0, 0.0)
    return C


def fast_matmul_xt(X, W):
    return fast_nt(X, W)


def chunked_cos_nr(A: np.ndarray, B: np.ndarray):
    dot = na = nb = 0.0
    rows = A.shape[0]
    for r0 in range(0, rows, 512):
        r1 = min(r0 + 512, rows)
        a = A[r0:r1].astype(np.float64)
        b = B[r0:r1].astype(np.float64)
        dot += float(np.sum(a * b))
        na += float(np.sum(a * a))
        nb += float(np.sum(b * b))
    cos_v = dot / (math.sqrt(na * nb) + 1e-18)
    nr_v = math.sqrt(nb / max(na, 1e-18))
    return cos_v, nr_v


def chol_qr2(Y: np.ndarray) -> np.ndarray:
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    _, q = Y.shape
    Q = Y
    for _ in range(2):
        G = fast_tn(Q, Q).astype(np.float64)
        G = 0.5 * (G + G.T)
        G[np.diag_indices(q)] += max(1e-7 * float(np.mean(np.diag(G))), 1e-12)
        R_inv_T = scipy.linalg.lapack.dtrtri(np.linalg.cholesky(G), lower=1)[0].astype(np.float32)
        Q = fast_nt(Q, R_inv_T)
    return Q


def fast_thin_svd_short_fat(A: np.ndarray):
    A32 = np.ascontiguousarray(A, dtype=np.float32)
    G = fast_nt(A32, A32).astype(np.float64)
    G = 0.5 * (G + G.T)
    vals, vecs = np.linalg.eigh(G)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(vals[idx], 0.0)
    U = vecs[:, idx]
    S = np.sqrt(vals)
    tol = max(1e-7 * (S[0] if len(S) else 1.0), 1e-12)
    keep = S > tol
    Uk = np.ascontiguousarray(U[:, keep], dtype=np.float32)
    Sk = S[keep].astype(np.float32)
    Vt = fast_tn(Uk / Sk[None, :], A32)
    return Uk, Sk, Vt


def svd_llm_lowrank_stacked(W_res, X_raw, rank, svd_x_cache, seed=0):
    """Single-pass Halko SVD-LLM(L_x) with stacked W_res @ [Vt_x; Omega_T]^T in 1 DRAM scan."""
    d_out, d_in = W_res.shape
    _, S_x, Vt_x = svd_x_cache
    m_x = len(S_x)
    tr = float(np.sum(S_x * S_x))
    lam = max(0.01 * tr / float(d_in), 1e-6)
    s_lam = math.sqrt(lam)
    alpha = (np.sqrt(S_x * S_x + lam) - s_lam).astype(np.float32)
    beta = (1.0 / np.sqrt(S_x * S_x + lam) - 1.0 / s_lam).astype(np.float32)

    q = min(rank + 24, d_out, d_in)
    rng = np.random.default_rng(seed)
    Omega_T = rng.standard_normal((q, d_in), dtype=np.float32)
    Stack = np.ascontiguousarray(np.vstack([Vt_x, Omega_T]), dtype=np.float32)
    # 1 DRAM scan of W_res produces BOTH W_res @ Vt_x^T and W_res @ Omega_T^T!
    Both = fast_nt(W_res, Stack)
    del Stack
    WR_V = Both[:, :m_x] * alpha[None, :]
    Y = s_lam * Both[:, m_x:] + WR_V @ fast_nt(Vt_x, Omega_T)
    del Both, Omega_T

    Q = chol_qr2(Y)
    C_T = s_lam * fast_tn(W_res, Q) + fast_tn(Vt_x, WR_V.T @ Q)
    C = np.ascontiguousarray(C_T.T)
    del C_T, WR_V
    Uc, Sc, Vhc = fast_thin_svd_short_fat(C)
    del C
    k = min(rank, len(Sc))
    U_b = Q @ Uc[:, :k]
    sr = np.sqrt(Sc[:k]).astype(np.float32)
    U_f = U_b * sr[None, :]
    V_mid = sr[:, None] * Vhc[:k, :]
    V_f = (1.0 / s_lam) * V_mid + fast_nt((V_mid @ Vt_x.T) * beta[None, :], np.ascontiguousarray(Vt_x.T))
    if k < rank:
        U_f = np.pad(U_f, ((0, 0), (0, rank - k)))
        V_f = np.pad(V_f, ((0, rank - k), (0, 0)))

    su = (np.max(np.abs(U_f), axis=1) / 127.0).clip(min=1e-8).astype(np.float32)
    u_i8 = np.clip(np.rint(U_f / su[:, None]), -127, 127).astype(np.int8)
    sv = (np.max(np.abs(V_f), axis=1) / 127.0).clip(min=1e-8).astype(np.float32)
    v_i8 = np.clip(np.rint(V_f / sv[:, None]), -127, 127).astype(np.int8)
    return dict(rank=rank, su=su, sv=sv, u_i8=u_i8, v_i8=v_i8)


def quantize_matrix_t16c(
    name: str, W_orig: np.ndarray, X_raw: np.ndarray, nbits: int, is_asym: bool,
    bsize: int, hblk: int, lr_rank: int = 0, svd_x_cache=None, seed: int = 0,
    smooth_alpha: float = 0.0, s_min: float = 1.0, s_max: float = 1.0,
    damp: float = 0.25, apply_ls: bool = True,
):
    d_out, d_in = W_orig.shape
    N = X_raw.shape[0]
    if smooth_alpha > 0.0:
        x_rms = np.sqrt(np.mean(X_raw * X_raw, axis=0) + 1e-12)
        w_rms = np.sqrt(np.mean(W_orig * W_orig, axis=0) + 1e-12)
        s = np.power(x_rms / w_rms, smooth_alpha).astype(np.float32)
        s = s / np.exp(np.mean(np.log(np.clip(s, 1e-8, 1e8))))
        s = np.clip(s, s_min, s_max).astype(np.float32)
        inv_s = (1.0 / s).astype(np.float32)
        W_rot = np.ascontiguousarray(W_orig * s[None, :], dtype=np.float32)
        X_rot = np.ascontiguousarray(X_raw * inv_s[None, :], dtype=np.float32)
    else:
        inv_s = None
        W_rot = np.ascontiguousarray(W_orig, dtype=np.float32).copy()
        X_rot = np.ascontiguousarray(X_raw, dtype=np.float32).copy()

    lib.fwht_rows_f32(W_rot.ctypes.data, d_out, d_in, bsize)
    lib.fwht_rows_f32(X_rot.ctypes.data, N, d_in, bsize)

    n_blks = d_in // bsize
    sc = np.empty((d_out, n_blks), dtype=np.float32)
    bi = np.empty((d_out, n_blks), dtype=np.float32)
    lib.quant_init_blocks_f32(W_rot.ctypes.data, sc.ctypes.data, bi.ctypes.data, d_out, d_in, bsize, nbits, 1 if is_asym else 0)

    idx = np.empty((d_out, d_in), dtype=np.uint8)
    work = np.empty(2 * hblk * hblk, dtype=np.float64)
    U_buf = np.empty((hblk, hblk), dtype=np.float32)
    E_buf = np.empty((d_out, bsize), dtype=np.float32)
    idx_blk = np.empty((d_out, bsize), dtype=np.uint8)
    for c0 in range(0, d_in, hblk):
        c1 = min(c0 + hblk, d_in)
        cw = c1 - c0
        Xc = np.ascontiguousarray(X_rot[:, c0:c1], dtype=np.float32)
        U = U_buf if cw == hblk else np.empty((cw, cw), dtype=np.float32)
        lib.build_gptq_u_f32(Xc.ctypes.data, N, cw, float(damp), U.ctypes.data, work.ctypes.data)
        W_sub = np.ascontiguousarray(W_rot[:, c0:c1], dtype=np.float32)
        for b0 in range(0, cw, bsize):
            g_blk = (c0 + b0) // bsize
            sc_c = np.ascontiguousarray(sc[:, g_blk], dtype=np.float32)
            bi_c = np.ascontiguousarray(bi[:, g_blk], dtype=np.float32)
            lib.gptq_intrablock_neon(W_sub.ctypes.data, U.ctypes.data, sc_c.ctypes.data, bi_c.ctypes.data, E_buf.ctypes.data, idx_blk.ctypes.data, d_out, cw, b0, bsize, nbits, 1 if is_asym else 0)
            idx[:, c0 + b0 : c0 + b0 + bsize] = idx_blk
            if b0 + bsize < cw:
                lib.trailing_update_neon(W_sub.ctypes.data, E_buf.ctypes.data, U.ctypes.data, d_out, cw, b0, bsize)
        del W_sub, Xc
    del X_rot, work, U_buf, E_buf, idx_blk

    lib.dequant_blocks_f32(idx.ctypes.data, sc.ctypes.data, bi.ctypes.data, W_rot.ctypes.data, d_out, d_in, bsize, nbits, 1 if is_asym else 0)
    lib.fwht_rows_f32(W_rot.ctypes.data, d_out, d_in, bsize)
    if inv_s is not None:
        W_rot *= inv_s[None, :]

    Y_q = fast_matmul_xt(X_raw, W_rot)
    Y_ref = fast_matmul_xt(X_raw, W_orig)
    if apply_ls:
        num = np.sum(Y_ref * Y_q, axis=0, dtype=np.float64)
        den = np.sum(Y_q * Y_q, axis=0, dtype=np.float64) + 1e-12
        ar = np.clip(num / den, 0.70, 1.40).astype(np.float32)
        sc *= ar[:, None]
        if is_asym:
            bi *= ar[:, None]
        sc_q = sc.astype(np.float16).astype(np.float32)
        bi_q = bi.astype(np.float16).astype(np.float32)
        lib.dequant_blocks_f32(idx.ctypes.data, sc_q.ctypes.data, bi_q.ctypes.data, W_rot.ctypes.data, d_out, d_in, bsize, nbits, 1 if is_asym else 0)
        lib.fwht_rows_f32(W_rot.ctypes.data, d_out, d_in, bsize)
        if inv_s is not None:
            W_rot *= inv_s[None, :]
        Y_q = fast_matmul_xt(X_raw, W_rot)
        del sc_q, bi_q

    w_cos, _ = chunked_cos_nr(W_orig, W_rot)
    y_cos, y_nr = chunked_cos_nr(Y_ref, Y_q)
    del Y_ref

    lr = None
    if lr_rank > 0:
        lib.sub_inplace_f32(W_orig.ctypes.data, W_rot.ctypes.data, W_orig.size)
        del W_rot
        lr = svd_llm_lowrank_stacked(W_orig, X_raw, lr_rank, svd_x_cache=svd_x_cache, seed=seed)
        U_deq = lr["u_i8"].astype(np.float32) * lr["su"][:, None]
        V_deq = lr["v_i8"].astype(np.float32) * lr["sv"][:, None]
        Y_q = Y_q + (X_raw @ V_deq.T) @ U_deq.T
    else:
        del W_rot

    n_tiles = (d_out + 15) // 16
    per_blk = (64 if is_asym else 32) + nbits * (bsize // 8) * 16
    packed = np.empty(n_tiles * n_blks * per_blk, dtype=np.uint8)
    lib.pack_tile16_bytes_c(sc.ctypes.data, (bi if is_asym else sc).ctypes.data, idx.ctypes.data, packed.ctypes.data, d_out, d_in, bsize, nbits, 1 if is_asym else 0)
    del idx, sc, bi

    mat = dict(
        name=name, d_out=d_out, d_in=d_in, block_size=bsize, n_bits=nbits,
        is_asym=is_asym, has_fwht=True, packed_bytes=packed.tobytes(),
        inv_s=inv_s, low_rank=lr,
    )
    del packed
    return Y_q, mat, (w_cos, y_cos, y_nr)


def pack_embed_streaming(sr: SafeSeekReader, out_path: Path):
    if out_path.exists():
        print(f"[embed] reusing cached {out_path} ({out_path.stat().st_size:,} B)", flush=True)
        return
    t0 = time.perf_counter()
    key = "model.language_model.embed_tokens.weight"
    d_out, d_in = sr.hdr[key]["shape"]  # [262144, 5376]
    bsize, nbits, is_asym = 256, 4, False
    n_blks = d_in // bsize
    per_blk = 32 + nbits * (bsize // 8) * 16
    chunk = 16384
    tmp_path = out_path.with_suffix(".tmp")
    err_num = err_den = 0.0
    with open(tmp_path, "wb") as f:
        for r0 in range(0, d_out, chunk):
            r1 = min(r0 + chunk, d_out)
            rows = r1 - r0
            assert rows % 16 == 0
            E = sr.read_bf16_rows(key, r0, r1)
            norm_e = float(np.sum(E * E, dtype=np.float64))
            err_den += norm_e
            E_rot = E.copy()
            lib.fwht_rows_f32(E_rot.ctypes.data, rows, d_in, bsize)
            sc = np.empty((rows, n_blks), dtype=np.float32)
            bi = np.empty((rows, n_blks), dtype=np.float32)
            lib.quant_init_blocks_f32(E_rot.ctypes.data, sc.ctypes.data, bi.ctypes.data, rows, d_in, bsize, nbits, 0)
            sc_b = np.repeat(sc, bsize, axis=1)
            idx = np.clip(np.rint(E_rot / (2.0 * sc_b) + 7.5), 0, 15).astype(np.uint8)
            E_q = np.empty_like(E)
            lib.dequant_blocks_f32(idx.ctypes.data, sc.ctypes.data, bi.ctypes.data, E_q.ctypes.data, rows, d_in, bsize, nbits, 0)
            lib.fwht_rows_f32(E_q.ctypes.data, rows, d_in, bsize)
            diff = E - E_q
            err_num += float(np.sum(diff * diff, dtype=np.float64))
            packed = np.empty((rows // 16) * n_blks * per_blk, dtype=np.uint8)
            lib.pack_tile16_bytes_c(sc.ctypes.data, sc.ctypes.data, idx.ctypes.data, packed.ctypes.data, rows, d_in, bsize, nbits, 0)
            f.write(packed.tobytes())
            del E, E_rot, E_q, diff, sc_b, sc, bi, idx, packed
    os.replace(tmp_path, out_path)
    rse = math.sqrt(err_num / max(err_den, 1e-12))
    print(f"[embed] packed 4b b256 sym FWHT in {time.perf_counter()-t0:.1f}s | bytes={out_path.stat().st_size:,} | rse={rse:.5f} | rss={rss_mb():.0f}MB", flush=True)


def rmsn_raw(x, w, eps):
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * inv
    return y * w if w is not None else y


def rope_rotate(x, theta, rotary_dims, hd):
    half_rot = rotary_dims // 2
    if half_rot == 0:
        return x
    half_h = hd // 2
    pos = np.arange(x.shape[0], dtype=np.float32)[:, None]
    j = np.arange(0, half_rot * 2, 2, dtype=np.float32)
    ang = pos * (1.0 / (theta ** (j / float(hd))))[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[:, :, :half_rot].copy(), x[:, :, half_h:half_h + half_rot].copy()
    out = x.copy()
    out[:, :, :half_rot] = x1 * c - x2 * s
    out[:, :, half_h:half_h + half_rot] = x1 * s + x2 * c
    return out


def assemble_t16c(sr: SafeSeekReader, state_dir: Path, embed_bin: Path, n_layers: int, out_path: Path):
    t0 = time.perf_counter()
    norm_keys = sorted(k for k in sr.hdr if ("norm.weight" in k or "layer_scalar" in k) and k.startswith("model.language_model."))
    norms = [(k, sr.read_bf16_f32(k)) for k in norm_keys]
    n_mats = 1
    layer_mats = []
    for l in range(n_layers):
        with open(state_dir / f"L{l:02d}.bin", "rb") as lf:
            mats = pickle.load(lf)
        layer_mats.append(len(mats))
        n_mats += len(mats)

    tmp_out = out_path.with_suffix(".tmp")
    with open(tmp_out, "wb") as f:
        pos = 0
        f.write(struct.pack("<4s5I", T16C_MAGIC, T16C_VERSION, ARCH_FLAGS, len(norms), n_mats, 0))
        pos += 24
        for name, arr in norms:
            nb = name.encode("utf-8")
            a32 = np.ascontiguousarray(arr, dtype="<f4").ravel()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", a32.size))
            f.write(a32.tobytes())
            pos += 2 + len(nb) + 4 + a32.size * 4

        # 1) Streaming embed_tokens [262144, 5376] 4b b256 sym FWHT
        emb_name = "model.language_model.embed_tokens.weight".encode("utf-8")
        emb_pbytes = embed_bin.stat().st_size
        f.write(struct.pack("<HIIHBBBBHQQ", len(emb_name), 262144, 5376, 256, 4, 0, 1, 0, 0, emb_pbytes, 0))
        f.write(emb_name)
        pos += 34 + len(emb_name)
        pad = (64 - (pos % 64)) % 64
        if pad:
            f.write(b"\x00" * pad)
            pos += pad
        with open(embed_bin, "rb") as ef:
            while True:
                buf = ef.read(16 * 1024 * 1024)
                if not buf:
                    break
                f.write(buf)
                pos += len(buf)

        # 2) Layer projections from L{nn}.bin in exact layer order
        for l in range(n_layers):
            with open(state_dir / f"L{l:02d}.bin", "rb") as lf:
                mats = pickle.load(lf)
            for m in mats:
                nb = m["name"].encode("utf-8")
                d_out, d_in, bsize, nbits = int(m["d_out"]), int(m["d_in"]), int(m["block_size"]), int(m["n_bits"])
                is_asym, has_fwht = bool(m["is_asym"]), bool(m["has_fwht"])
                inv_s = m["inv_s"]
                inv_kind = 1 if inv_s is not None else 0
                lr = m["low_rank"]
                lr_rk = int(lr["rank"]) if lr is not None else 0
                lr_b = (d_out * 4 + lr_rk * 4 + d_out * lr_rk + lr_rk * d_in) if lr is not None else 0
                pbytes = m["packed_bytes"]
                f.write(struct.pack("<HIIHBBBBHQQ", len(nb), d_out, d_in, bsize, nbits, 1 if is_asym else 0, 1 if has_fwht else 0, inv_kind, lr_rk, len(pbytes), lr_b))
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
                    f.write(iv.tobytes())
                    pos += iv.size * 4
                if lr is not None:
                    f.write(np.ascontiguousarray(lr["su"], dtype="<f4").ravel().tobytes())
                    f.write(np.ascontiguousarray(lr["sv"], dtype="<f4").ravel().tobytes())
                    f.write(np.ascontiguousarray(lr["u_i8"], dtype=np.int8).ravel().tobytes())
                    f.write(np.ascontiguousarray(lr["v_i8"], dtype=np.int8).ravel().tobytes())
                    pos += lr_b
            del mats
    os.replace(tmp_out, out_path)
    sz = out_path.stat().st_size
    print(f"ASSEMBLED {out_path} | {sz:,} B ({sz/(1024**3):.4f} GiB) | {len(norms)} norms + {n_mats} mats | dt={time.perf_counter()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Streaming layer-by-layer Gemma 4 31B -> T16C v1 calibrator")
    ap.add_argument("--model-dir", type=Path, default=None, help="Path to gemma-4-31b-it-qat-w4a16-ct directory (or set GEMMA4_31B_DIR)")
    ap.add_argument("--state-dir", type=Path, default=SCRIPT_DIR / "state_31b", help="Checkpoint directory for embed_tokens.bin and L{nn}.bin")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "gemma4_31b_cal.t16c", help="Output .t16c path")
    ap.add_argument("--compile-dir", type=Path, default=None, help="Directory for compiled C helper shared libraries")
    ap.add_argument("--mcpu", default="cortex-a72", help="GCC -mcpu target on AArch64 hosts")
    args = ap.parse_args()

    model_dir = _resolve_model_dir(args.model_dir)
    model_path = model_dir / "model.safetensors"
    cfg_path = model_dir / "config.json"
    tok_path = model_dir / "tokenizer.json"
    state_dir = args.state_dir.expanduser().resolve()
    out_t16c = args.out.expanduser().resolve()
    compile_dir = (args.compile_dir or (state_dir / "lib")).expanduser().resolve()
    _init_native_libs(compile_dir, mcpu=args.mcpu)

    state_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(tok_path))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["text_config"]
    n_layers = cfg["num_hidden_layers"]
    sr = SafeSeekReader(model_path)

    pack_embed_streaming(sr, state_dir / "embed_tokens.bin")

    state_file = state_dir / "x_state.npz"
    if state_file.exists():
        z = np.load(state_file)
        start_l = int(z["next_l"][0])
        lengths = [int(v) for v in z["lengths"]]
        x_seqs = list(np.split(z["x_cat"], np.cumsum(lengths)[:-1]))
        print(f"[resume] starting from L{start_l:02d}/{n_layers} ({len(lengths)} prompts, {sum(lengths)} tokens)", flush=True)
    else:
        prompts = [
            "<|turn>user\nExplain how photosynthesis converts light energy into chemical energy in plants.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Photosynthesis occurs in chloroplasts through light-dependent reactions on thylakoid membranes and the Calvin cycle in the stroma, producing ATP, NADPH, and carbohydrates from carbon dioxide and water.",
            "<|turn>user\nWrite a Python function that computes the greatest common divisor using Euclid's algorithm.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)",
            "<|turn>user\nWhat were the main technological drivers of the Industrial Revolution in Britain?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Mechanized cotton spinning, Watt's separate-condenser steam engine, coke-smelted iron, precision machine tools, and canal and railway networks transformed industrial output.",
            "<|turn>user\nSimplify 3(x + 4) - 2(x - 5) and solve for x when the expression equals 30.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Expanding gives 3x + 12 - 2x + 10 = x + 22. Setting x + 22 = 30 yields x = 8.",
            "<|turn>user\nState the second law of thermodynamics and explain what entropy measures.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>The total entropy of an isolated system never decreases over time; statistical mechanics relates entropy to the logarithm of accessible microstates corresponding to a macrostate.",
            "<|turn>user\nSummarize how Ohm's law relates voltage, current, and resistance in DC circuits.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>In a linear resistor at constant temperature, voltage equals current multiplied by resistance (V = I * R), and electrical power dissipated as heat equals I^2 * R.",
            "<|turn>user\nDescribe the major steps of the biological nitrogen cycle.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Nitrogen-fixing bacteria convert atmospheric N2 into ammonia, nitrifying microbes oxidize ammonia into nitrites and nitrates absorbed by plants, and denitrifiers return N2 to air.",
            "<|turn>user\nCompare merge sort and quicksort in terms of time and memory complexity.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Merge sort guarantees O(n log n) worst-case time with O(n) auxiliary space, whereas randomized quicksort runs in O(n log n) expected time in place with O(log n) stack depth.",
            "<|turn>user\nHow did Roman engineers maintain steady water flow across long aqueducts?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Surveyors used surveying levels like the chorobates to lay masonry channels at uniform shallow downhill gradients, routing water through tunnels, arcades, and inverted siphons.",
            "<|turn>user\nExplain the geological difference between divergent plate boundaries and subduction zones.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Divergent boundaries pull lithospheric plates apart so rising magma forms new oceanic crust, while subduction zones sink dense oceanic plates into the mantle beneath volcanic arcs.",
        ]
        seqs = [tok.encode(pr).ids for pr in prompts]
        esc = math.sqrt(float(cfg["hidden_size"]))
        x_seqs = [np.stack([sr.read_bf16_rows("model.language_model.embed_tokens.weight", tid, tid + 1)[0] * esc for tid in ids], 0) for ids in seqs]
        lengths = [len(x) for x in x_seqs]
        start_l = 0
        print(f"[init] {len(lengths)} general calibration prompts ({sum(lengths)} tokens), start L00..L{n_layers-1}", flush=True)

    DOWN_2B_LAYERS = {0, 1, 2, 3, 4, 5, 6, 10, 11, 17, 23, 29, 35, 41, 47, 53, 54, 55, 56, 57, 58, 59}
    t_run = time.perf_counter()
    c0 = math.sqrt(2.0 / math.pi)
    eps = cfg["rms_norm_eps"]
    nh = cfg["num_attention_heads"]

    for l in range(start_l, n_layers):
        tl0 = time.perf_counter()
        p = f"model.language_model.layers.{l}."
        lt = cfg["layer_types"][l]
        is_full = lt == "full_attention"
        if is_full:
            hd = cfg["global_head_dim"]
            nkv = cfg["num_global_key_value_heads"]
            rp = cfg["rope_parameters"]["full_attention"]
            theta = rp["rope_theta"]
            rotary_dims = int(hd * rp["partial_rotary_factor"])
            sw = 100000
        else:
            hd = cfg["head_dim"]
            nkv = cfg["num_key_value_heads"]
            theta = cfg["rope_parameters"]["sliding_attention"]["rope_theta"]
            rotary_dims = hd
            sw = cfg["sliding_window"]
        gqa = nh // nkv
        mats = []

        g_in = sr.read_bf16_f32(p + "input_layernorm.weight")
        X_in = rmsn_raw(np.concatenate(x_seqs, 0), g_in, eps)

        Wq = sr.read_qat_f32(p + "self_attn.q_proj")
        Yq_cat, mq, st_q = quantize_matrix_t16c(p + "self_attn.q_proj.weight", Wq, X_in, 2, False, 256, 256, 0, smooth_alpha=0.25, s_min=0.5, s_max=2.0, damp=0.15, apply_ls=True); del Wq; mats.append(mq)
        Wk = sr.read_qat_f32(p + "self_attn.k_proj")
        Yk_cat, mk, _    = quantize_matrix_t16c(p + "self_attn.k_proj.weight", Wk, X_in, 2, False, 256, 256, 0, smooth_alpha=0.25, s_min=0.5, s_max=2.0, damp=0.15, apply_ls=True); del Wk; mats.append(mk)
        if not is_full:
            Wv = sr.read_qat_f32(p + "self_attn.v_proj")
            Yv_cat, mv, _ = quantize_matrix_t16c(p + "self_attn.v_proj.weight", Wv, X_in, 2, False, 256, 256, 0, smooth_alpha=0.25, s_min=0.5, s_max=2.0, damp=0.15, apply_ls=True); del Wv; mats.append(mv)
        else:
            Yv_cat = None
        del X_in

        qn = sr.read_bf16_f32(p + "self_attn.q_norm.weight")
        kn = sr.read_bf16_f32(p + "self_attn.k_norm.weight")
        o_list, off = [], 0
        for tl in lengths:
            q = rope_rotate(rmsn_raw(Yq_cat[off:off+tl].reshape(tl, nh, hd), qn, eps), theta, rotary_dims, hd)
            k = rope_rotate(rmsn_raw(Yk_cat[off:off+tl].reshape(tl, nkv, hd), kn, eps), theta, rotary_dims, hd)
            v_src = Yk_cat[off:off+tl] if is_full else Yv_cat[off:off+tl]
            v = rmsn_raw(v_src.reshape(tl, nkv, hd), None, eps)
            ctx = np.empty((tl, nh, hd), dtype=np.float32)
            ii, jj = np.arange(tl)[:, None], np.arange(tl)[None, :]
            mask = np.where((jj > ii) | ((ii - jj) >= sw if not is_full else False), -1e9, 0.0).astype(np.float32)
            for h in range(nh):
                scm = q[:, h, :] @ k[:, h // gqa, :].T + mask
                scm -= np.max(scm, axis=-1, keepdims=True)
                pr = np.exp(scm); pr /= np.sum(pr, axis=-1, keepdims=True)
                ctx[:, h, :] = pr @ v[:, h // gqa, :]
            o_list.append(ctx.reshape(tl, nh * hd))
            off += tl
        del Yq_cat, Yk_cat, Yv_cat
        X_o = np.concatenate(o_list, 0); del o_list

        Wo = sr.read_qat_f32(p + "self_attn.o_proj")
        q_yo, mo, _ = quantize_matrix_t16c(p + "self_attn.o_proj.weight", Wo, X_o, 2, False, 256, 256, 0, smooth_alpha=0.0, damp=0.15, apply_ls=True); del Wo, X_o; mats.append(mo)
        g_pa = sr.read_bf16_f32(p + "post_attention_layernorm.weight")
        x_cat = np.concatenate(x_seqs, 0)
        q_mid = x_cat + rmsn_raw(q_yo, g_pa, eps); del q_yo, x_cat

        g_pf = sr.read_bf16_f32(p + "pre_feedforward_layernorm.weight")
        q_hm = rmsn_raw(q_mid, g_pf, eps)

        Wg = sr.read_qat_f32(p + "mlp.gate_proj")
        q_g, mg, st_g = quantize_matrix_t16c(p + "mlp.gate_proj.weight", Wg, q_hm, 1, True, 256, 256, 0, smooth_alpha=0.0, damp=0.25, apply_ls=True); del Wg; mats.append(mg)
        Wu = sr.read_qat_f32(p + "mlp.up_proj")
        q_u, mu, _    = quantize_matrix_t16c(p + "mlp.up_proj.weight",   Wu, q_hm, 1, True, 256, 256, 0, smooth_alpha=0.0, damp=0.25, apply_ls=True); del Wu, q_hm; mats.append(mu)

        q_act = 0.5 * q_g * (1.0 + np.tanh(c0 * (q_g + 0.044715 * (q_g * q_g * q_g)))) * q_u
        del q_g, q_u

        d_nb = 2 if l in DOWN_2B_LAYERS else 1
        d_asym = (d_nb == 1)
        d_damp = 0.10 if d_nb == 2 else 0.25
        Wd = sr.read_qat_f32(p + "mlp.down_proj")
        q_yd, md, st_d = quantize_matrix_t16c(p + "mlp.down_proj.weight", Wd, q_act, d_nb, d_asym, 256, 256, 0, smooth_alpha=0.0, damp=d_damp, apply_ls=True); del Wd, q_act; mats.append(md)

        g_pff = sr.read_bf16_f32(p + "post_feedforward_layernorm.weight")
        ls_val = float(sr.read_bf16_f32(p + "layer_scalar").ravel()[0])
        x_next_cat = (q_mid + rmsn_raw(q_yd, g_pff, eps)) * ls_val
        del q_mid, q_yd
        x_rms = float(np.sqrt(np.mean(x_next_cat * x_next_cat)))
        x_seqs = list(np.split(x_next_cat, np.cumsum(lengths)[:-1]))

        tmp_l = state_dir / f"L{l:02d}.bin.tmp"
        with open(tmp_l, "wb") as lf:
            pickle.dump(mats, lf, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_l, state_dir / f"L{l:02d}.bin")
        del mats

        tmp_s = state_dir / "x_state.tmp.npz"
        np.savez(tmp_s, x_cat=x_next_cat, lengths=np.array(lengths, dtype=np.int32), next_l=np.array([l + 1], dtype=np.int32))
        os.replace(tmp_s, state_file)
        del x_next_cat
        gc.collect()

        dt_l = time.perf_counter() - tl0
        done_n = l - start_l + 1
        avg = (time.perf_counter() - t_run) / done_n
        eta_m = (n_layers - 1 - l) * avg / 60.0
        print(
            f"L{l:02d}/{n_layers-1} ({lt[:4]}) {dt_l:.1f}s | avg={avg:.1f}s/L ETA={eta_m:.1f}m | "
            f"x_rms={x_rms:.2f} | q:{st_q[1]:.4f}/{st_q[2]:.2f}(w={st_q[0]:.3f}) "
            f"g:{st_g[1]:.4f}/{st_g[2]:.2f}(w={st_g[0]:.3f}) "
            f"d({d_nb}b):{st_d[1]:.4f}/{st_d[2]:.2f}(w={st_d[0]:.3f}) | rss={rss_mb():.0f}MB",
            flush=True,
        )

    assemble_t16c(sr, state_dir, state_dir / "embed_tokens.bin", n_layers, out_t16c)
    sr.close()


if __name__ == "__main__":
    main()

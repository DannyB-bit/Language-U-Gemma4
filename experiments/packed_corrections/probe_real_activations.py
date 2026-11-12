"""Small, held-out real-input compression experiment, not a 31B model run.

Fits on actual layer-0 post-norm token embeddings. Token types, not merely
sentences, are disjoint between calibration, validation and final evaluation.
Uses independently written batched quantization and residual fitting code.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np
from tokenizers import Tokenizer

from paths import REPO as ROOT, DATA as AUDIT
CACHE = AUDIT / "activation-cache"
CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from neon_runner.bridge import fwht8_vec_numpy, encode_matrix_opt5_v2
from neon_runner.checkpoint_io import bf16_to_float32, round_bf16
from neon_runner.pi4_quantization import decode_opt5_36
from neon_runner.text_runtime import rmsnorm

REVISION = "52f3f65bc7a02d555763bc923bd1d9094898219d"
PACKAGE = "bd9c48f1da41b51521570674ba2696657eb7358e"
SOURCE = f"https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/resolve/{REVISION}/model.safetensors?download=true"
SIZE = 23265352448
receipts = []
header = json.loads((AUDIT / "range-cache/header.bin").read_bytes()[8:])
base = 266888


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch_range(start, end, label):
    path = CACHE / label
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        data = path.read_bytes()
        if (receipt["start"] == start and receipt["end_inclusive"] == end
                and len(data) == end - start + 1 and sha(data) == receipt["sha256"]):
            receipts.append(receipt)
            return data
        raise ValueError("Cached range failed verification: " + label)
    req = Request(SOURCE + f"&range_start={start}",
                  headers={"Range": f"bytes={start}-{end}"})
    with urlopen(req, timeout=90) as response:
        if (response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}"):
            raise ValueError("Wrong range for " + label)
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("Truncated range: " + label)
    receipt = dict(label=label, start=start, end_inclusive=end, sha256=sha(data))
    path.write_bytes(data)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipts.append(receipt)
    return data


def download_small(name, expected=None):
    path = CACHE / name
    if path.exists():
        data = path.read_bytes()
    else:
        url = f"https://huggingface.co/TheAiCollectiveART/Glyph-Stream-31b/resolve/{PACKAGE}/{name}"
        data = urlopen(url, timeout=90).read()
        path.write_bytes(data)
    if expected and sha(data) != expected:
        raise ValueError("Hash mismatch: " + name)
    return data


tokenizer_raw = download_small(
    "tokenizer.json", "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f")
tokenizer = Tokenizer.from_str(tokenizer_raw.decode())
config = json.loads(download_small("config.json"))["text_config"]

# Locally authored passages; no benchmark-answer data, external/private text or
# target-model answers enter calibration. Vocabulary types are filtered below.
passages = [
    "A small library stores books by subject and author. Readers compare a story "
    "with a newspaper article, write a summary, and explain the difference. "
    "The teacher asks why the character changed her mind after the storm. "
    "A programmer sorts numbers, checks empty input, and returns a dictionary. "
    "The database keeps an index for names and uses a transaction to update balances. "
    "An engineer measures voltage, resistance, current, temperature, speed and time. "
    "The garden has red flowers, green leaves, wet soil and two tall trees. "
    "Write a careful answer in English, show the calculation, and check every unit. "
    "Explain how a function handles missing values, duplicate keys, and invalid types. "
    "A train leaves the station at noon and arrives before sunset. "
    "Compare multiplication and division using fractions and whole numbers. "
    "The scientist records each observation and repeats the experiment.",
    "Volcanic ash alters atmospheric chemistry while ocean currents redistribute "
    "thermal energy between continents. Geologists examine sedimentary layers and "
    "measure isotope ratios to estimate the age of mineral deposits. "
    "A compiler transforms expressions into instructions; concurrent threads require "
    "synchronization when modifying shared resources. "
    "Describe photosynthesis, cellular respiration, evaporation, condensation and "
    "gravitational acceleration. A violinist adjusts pitch while an orchestra rehearses. "
    "A municipality repairs bridges and schedules waste collection during winter. "
    "The detective interviews witnesses and distinguishes evidence from speculation. "
    "A graph algorithm visits vertices and records predecessor relationships. "
    "Recursive descent parsers recognize nested parentheses and operator precedence.",
    "Archaeologists catalog ceramic fragments discovered beneath a medieval courtyard. "
    "Migratory birds navigate coastlines, wetlands, mountains and seasonal weather. "
    "Astronomers estimate stellar luminosity using spectroscopic measurements. "
    "The bakery kneads dough, regulates fermentation, and monitors oven humidity. "
    "Cryptographic authentication prevents undetected modification of transmitted messages. "
    "A distributed queue retries interrupted jobs and deduplicates acknowledgements. "
    "Explain the relationship between biodiversity, pollination, agriculture and habitat. "
    "Analyze a sonnet's imagery, rhythm, metaphors and narrative perspective. "
    "Numerical integration approximates the area beneath a continuous curve. "
    "An electric motor converts electromagnetic forces into rotational motion."
]
seen = set()
splits = []
for text in passages:
    ids = []
    for value in tokenizer.encode(text, add_special_tokens=False).ids:
        if value not in seen:
            ids.append(value)
            seen.add(value)
    splits.append(ids)
assert min(map(len, splits)) >= 60, list(map(len, splits))
all_ids = sorted(seen)
print(json.dumps({"stage": "fetch_inputs", "disjoint_token_counts": list(map(len, splits))}), flush=True)

emb = header["model.language_model.embed_tokens.weight"]
assert emb["dtype"] == "BF16" and emb["shape"] == [262144, 5376]
groups = []
for token in all_ids:
    if groups and token - groups[-1][-1] <= 16:
        groups[-1].append(token)
    else:
        groups.append([token])


def fetch_embeddings(group):
    lo, hi = group[0], group[-1]
    offset = base + emb["data_offsets"][0] + lo * 5376 * 2
    data = fetch_range(offset, offset + (hi-lo+1)*5376*2-1,
                       f"embeddings-{lo}-{hi}.bin")
    decoded = bf16_to_float32(np.frombuffer(data, dtype="<u2")).reshape(-1, 5376)
    return {value: decoded[value-lo].copy() for value in group}


embeddings = {}
with ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(fetch_embeddings, groups):
        embeddings.update(result)
norm_item = header["model.language_model.layers.0.input_layernorm.weight"]
norm_start = base + norm_item["data_offsets"][0]
norm_raw = fetch_range(norm_start, norm_start+5376*2-1, "input-norm.bin")
gain = bf16_to_float32(np.frombuffer(norm_raw, dtype="<u2"))
inputs = [rmsnorm(np.stack([embeddings[i] for i in ids]) * np.float32(math.sqrt(5376)),
                  gain, config.get("rms_norm_eps", 1e-6)).astype("f8") for ids in splits]
data = np.frombuffer((AUDIT / "range-cache/model.language_model.layers.0.self_attn.q_proj.weight_packed.bin").read_bytes(), dtype="u1").reshape(128, -1)
integers = np.stack((data & 15, data >> 4), axis=-1).reshape(128, -1).astype("f4")-8
scales = bf16_to_float32(np.frombuffer((AUDIT / "range-cache/model.language_model.layers.0.self_attn.q_proj.weight_scale.bin").read_bytes(), dtype="<u2")).reshape(128, -1)
w = round_bf16(integers * np.repeat(scales, 32, axis=1)).astype("f8")
xtrain, xval, xtest = inputs
ytrain, yval, ytest = [x @ w.T for x in inputs]
np.savez(AUDIT / "real_activation_inputs.npz", w=w, xtrain=xtrain, xval=xval, xtest=xtest)
if __import__("os").environ.get("LANGU_CALIB_PREPARE_ONLY") == "1":
    print("Saved real activation inputs", flush=True)
    raise SystemExit(0)


def rotate(a):
    return fwht8_vec_numpy(a.astype("f4").ravel()).reshape(a.shape).astype("f8")


def to_records(levels, bits):
    rec = np.empty((len(bits), 36), dtype="u1")
    rec[:, :4] = levels.astype("<f2").view("u1").reshape(-1, 4)
    rec[:, 4:] = np.packbits(bits, axis=1)
    return rec.tobytes()


def gptq_matrix(weight, activation, damping):
    """Batched per-tile GPTQ; reuse one inverse-Cholesky per input tile.

    Fixed fp16 scale/bias, with an optional H-weighted two-level refit.
    Candidates are selected using calibration data only. All returned values
    are decoded from the actual serialized 36-byte record format.
    """
    result = np.empty_like(weight)
    all_records = []
    changes = 0
    for start in range(0, weight.shape[1], 256):
        orig = weight[:, start:start+256]
        a = activation[:, start:start+256]
        h = a.T @ a / len(a)
        diag = max(float(np.diag(h).mean()), 1e-12)
        upper = np.linalg.cholesky(np.linalg.solve(
            h + np.eye(256)*diag*damping, np.eye(256))).T
        bias = orig.mean(1).astype("f2").astype("f8")
        scale = np.abs(orig-bias[:, None]).mean(1).astype("f2").astype("f8")
        bits = orig >= bias[:, None]
        work = orig.copy()
        proposed = np.empty_like(bits)
        for j in range(256):
            proposed[:, j] = work[:, j] >= bias
            decoded = bias + scale * (2.0*proposed[:, j]-1)
            error = (work[:, j]-decoded)/upper[j, j]
            work[:, j:] -= error[:, None]*upper[j, j:]

        def error_of(candidate, lev):
            delta = orig - (lev[:, :1]*(2.0*candidate-1)+lev[:, 1:2])
            return np.einsum("ri,ij,rj->r", delta, h, delta)

        levels = np.column_stack((scale, bias))
        best_error = error_of(bits, levels)
        for candidate in (proposed, bits.copy()):
            candidate_levels = levels.copy()
            s = 2.0*candidate-1
            one_h = h.sum(0)
            ss = np.einsum("ri,ij,rj->r", s, h, s)
            so = s @ one_h
            oo = float(one_h.sum())
            sw = np.einsum("ri,ij,rj->r", s, h, orig)
            ow = orig @ one_h
            determinant = ss*oo-so*so
            safe = determinant > 1e-12*np.maximum(ss*oo, 1e-30)
            fitted_scale = np.divide(sw*oo-ow*so, determinant,
                                     out=scale.copy(), where=safe)
            fitted_bias = np.divide(ow*ss-sw*so, determinant,
                                    out=bias.copy(), where=safe)
            for lev in (candidate_levels, np.column_stack((fitted_scale, fitted_bias))):
                stored = lev.astype("f2").astype("f8")
                finite = np.isfinite(stored).all(1)
                loss = error_of(candidate, np.nan_to_num(stored))
                better = finite & (loss < best_error)
                changes += int(better.sum())
                bits[better] = candidate[better]
                levels[better] = stored[better]
                best_error[better] = loss[better]
        raw = to_records(levels, bits)
        result[:, start:start+256] = decode_opt5_36(raw)
        all_records.append(raw)
    return result, changes


def metric(weight, x, target):
    pred = x @ weight.T
    err = np.sum((target-pred)**2)
    return dict(relative_squared_error=float(err/np.sum(target**2)),
                output_cosine=float(np.sum(target*pred)/np.linalg.norm(target)/np.linalg.norm(pred)))


variants = []
weights = {}
def record(name, q, extra=None):
    item = dict(name=name, train=metric(q,xtrain,ytrain), validation=metric(q,xval,yval),
                stored_linear_bytes=128*5376//256*36)
    if extra:
        item.update(extra)
    variants.append(item)
    weights[name] = q
    print(json.dumps({"stage":"candidate", **item}), flush=True)


baseline_raw, _ = encode_matrix_opt5_v2(rotate(w).astype("f4"), outlier_preserved=False)
baseline = rotate(decode_opt5_36(baseline_raw).reshape(w.shape))
record("h8_binary_baseline", baseline)

for alpha in (0.0, 0.5, 1.0):
    # This is one projection experiment, not a folded whole-model graph.
    am = np.maximum(np.max(np.abs(xtrain),axis=0),1e-8)
    wm = np.maximum(np.max(np.abs(w),axis=0),1e-8)
    diagonal = am**alpha / wm**(1-alpha)
    diagonal = np.clip(diagonal/np.median(diagonal),0.01,100).astype("f2").astype("f8")
    transformed = rotate(w*diagonal)
    xt = rotate(xtrain/diagonal)
    # Exact-basis sanity check; evaluated independently of compression.
    equiv = np.linalg.norm(xt@transformed.T-ytrain)/np.linalg.norm(ytrain)
    if equiv > 1e-5:
        raise ValueError("Basis equivalence failed")
    raw,_ = encode_matrix_opt5_v2(transformed.astype("f4"),outlier_preserved=False)
    q = rotate(decode_opt5_36(raw).reshape(w.shape))/diagonal
    record(f"diagonal_{alpha}_binary",q,
           dict(diagonal_bytes=diagonal.size*2,basis_relative_error=float(equiv)))
    for damping in (0.01,0.1):
        decoded,changes = gptq_matrix(transformed,xt,damping)
        q = rotate(decoded)/diagonal
        record(f"diagonal_{alpha}_gptq_{damping}",q,
               dict(diagonal_bytes=diagonal.size*2,accepted_tile_row_updates=changes))

# Choose quantizer on validation; test data has not entered selection.
best_quant = min(variants,key=lambda a:a["validation"]["relative_squared_error"])
best_weight = weights[best_quant["name"]]
residual = w-best_weight
u,s,vt = np.linalg.svd(residual,full_matrices=False)
rank=8
b=(u[:,:rank]*np.sqrt(s[:rank])).astype("f2").astype("f8")
a=(np.sqrt(s[:rank])[:,None]*vt[:rank]).astype("f2").astype("f8")
record(best_quant["name"]+"_weight_svd8",best_weight+b@a,
       dict(adapter_bytes=(a.size+b.size)*2))
error_target=ytrain-xtrain@best_weight.T
u,s,vt=np.linalg.svd(error_target,full_matrices=False)
gram=xtrain@xtrain.T
for ridge in (0.01,0.1,1.0):
    regularized=gram+np.eye(len(gram))*ridge*np.diag(gram).mean()
    projection=np.linalg.solve(regularized,xtrain)
    b=(vt[:rank].T*np.sqrt(s[:rank])).astype("f2").astype("f8")
    a=((np.sqrt(s[:rank])[:,None]*u[:,:rank].T)@projection).astype("f2").astype("f8")
    record(best_quant["name"]+f"_activation_svd8_ridge{ridge}",best_weight+b@a,
           dict(adapter_bytes=(a.size+b.size)*2))

selected=min(variants,key=lambda a:a["validation"]["relative_squared_error"])
baseline_result=metric(baseline,xtest,ytest)
selected_result=metric(weights[selected["name"]],xtest,ytest)
report=dict(scope="One real first-layer Q projection excerpt (128x5376), token-type-disjoint inputs. Not whole-model generation or Pi proof.",
            checkpoint_revision=REVISION,tokenizer_sha256=sha(tokenizer_raw),
            split_ids=splits,split_counts=list(map(len,splits)),
            calibration_texts=passages,selection_metric="validation relative squared projection error",
            selected=selected["name"],baseline_test=baseline_result,selected_test=selected_result,
            test_error_reduction_fraction=1-selected_result["relative_squared_error"]/baseline_result["relative_squared_error"],
            variants=variants,range_receipts=sorted(receipts,key=lambda r:r["start"]))
destination=AUDIT/"real_activation_probe.json"
destination.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("selected","baseline_test","selected_test","test_error_reduction_fraction")}),flush=True)

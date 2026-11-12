"""Validation-selected correction capacity at a fixed stored byte budget."""
import hashlib
import json

import numpy as np

from paths import DATA
from binary_factor_probe import PackedFactors
from correction_codec import Correction, save_correction


def metric(prediction, target):
    p, t = prediction.astype("f8"), target.astype("f8")
    return {"relative_squared_error": float(np.sum((p - t)**2) / np.sum(t*t)),
            "output_cosine": float(np.sum(p*t) / np.linalg.norm(p) / np.linalg.norm(t))}


def main():
    data = np.load(DATA / "real_activation_inputs.npz", allow_pickle=False)
    w = data["w"].astype("f8")
    train, val, test = [data[k].astype("f4") for k in ("xtrain", "xval", "xtest")]
    previous = json.loads((DATA / "scale_calibration_probe.json").read_text())
    strength = previous["variants"][previous["selected_variant"]]["ridge_strength"]
    base_path = DATA / f"factors-scale-fit-{strength}.bin"
    base = PackedFactors(base_path)
    # Fit against the deployed packed FP32 operator, not an ideal dense base.
    base_train = np.stack([base.project(x) for x in train])
    base_val = np.stack([base.project(x) for x in val])
    residual = train.astype("f8") @ w.T - base_train
    u, singular, v = np.linalg.svd(residual, full_matrices=False)
    x = train.astype("f8")
    gram = x @ x.T
    budget = 64 + 2 * 8 * (w.shape[0] + w.shape[1])
    candidates, stored_models = [], []
    for ridge in (.01, .1, 1., 10.):
        inverse_projection = np.linalg.solve(gram + ridge * np.diag(gram).mean() * np.eye(len(x)), x)
        for storage, rank in (("fp16", 8), ("int8", 8), ("int8", 12), ("int8", 15)):
            a = (np.sqrt(singular[:rank])[:, None] * u[:, :rank].T) @ inverse_projection
            b = v[:rank].T * np.sqrt(singular[:rank])
            path = DATA / f"correction-{storage}-r{rank}-ridge{ridge}.rcf"
            size = save_correction(path, a, b, storage)
            if size > budget:
                raise ValueError("Candidate exceeded fixed budget")
            stored = Correction(path)
            projected = np.stack([stored.project(z) for z in val])
            # Independent float64 oracle for the coefficients actually stored.
            da = stored.a.astype("f8") * stored.sa[:, None]
            db = stored.b.astype("f8") * stored.sb[:, None]
            oracle = (val.astype("f8") @ da.T) @ db.T
            parity = float(np.linalg.norm(projected - oracle) / np.linalg.norm(oracle))
            if parity > 1e-5:
                raise ValueError("Stored correction parity failed")
            rec = {"storage": storage, "rank": rank, "ridge": ridge, "file": path.name,
                   "bytes_including_header_and_scales": size,
                   "validation": metric(base_val + projected, val.astype("f8") @ w.T),
                   "stored_fp32_vs_float64_relative_error": parity,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            candidates.append(rec)
            stored_models.append(stored)
            print(json.dumps(rec), flush=True)
    selected = min(range(len(candidates)), key=lambda i: candidates[i]["validation"]["relative_squared_error"])
    controls = [i for i, c in enumerate(candidates) if c["storage"] == "fp16"]
    control = min(controls, key=lambda i: candidates[i]["validation"]["relative_squared_error"])
    base_test = np.stack([base.project(z) for z in test])
    target = test.astype("f8") @ w.T
    def evaluate(i):
        return metric(base_test + np.stack([stored_models[i].project(z) for z in test]), target)
    report = {"scope": "128 rows of L0 Q, reused development test; not language accuracy or Pi memory/speed.",
              "selection": "Minimum validation error within FP16 rank-8 file-byte budget; no test selection.",
              "execution": "Stored binary base and stored correction, both chunked FP32 NumPy operators.",
              "data_sha256": hashlib.sha256((DATA / "real_activation_inputs.npz").read_bytes()).hexdigest(),
              "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
              "correction_budget_bytes": budget, "candidates": candidates,
              "selected": candidates[selected], "fp16_control": candidates[control],
              "base_test": metric(base_test, target), "fp16_control_test": evaluate(control),
              "selected_test": evaluate(selected)}
    (DATA / "correction_precision_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("selected", "fp16_control_test", "selected_test")}), flush=True)


if __name__ == "__main__":
    main()

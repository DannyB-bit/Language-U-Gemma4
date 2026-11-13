"""Calibration and reference quantization for the Pi 4 RAM-fit experiment.

This module does not claim model quality or Pi throughput. Feed actual captured
projection inputs, never token IDs or unrelated vectors, into calibration.
"""
from dataclasses import dataclass
import math
import numpy as np


def _finite(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


@dataclass
class ChannelStats:
    """Streaming channel maxima and diagonal E[xx^T], in the supplied basis."""
    absmax: np.ndarray
    sumsq: np.ndarray
    samples: int = 0

    @classmethod
    def empty(cls, channels):
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        return cls(np.zeros(channels), np.zeros(channels))

    def update(self, activations):
        x = _finite(activations, "activations")
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.absmax.size or x.shape[0] == 0:
            raise ValueError("Expected nonempty [tokens, channels] activations")
        self.absmax = np.maximum(self.absmax, np.max(np.abs(x), axis=0))
        self.sumsq += np.sum(x * x, axis=0)
        self.samples += x.shape[0]

    @property
    def h_diag(self):
        if not self.samples:
            raise ValueError("No calibration samples")
        return self.sumsq / self.samples


def shared_channel_scale(absmax, consumer_weights, alpha=0.5):
    """One scale for ALL consumers of the same norm output (e.g. Q/K/V).

    SmoothQuant equivalence is exact before quantization; improved 1-bit weight
    quality is a hypothesis. Evaluate alpha, including an unscaled control.
    """
    a = _finite(absmax, "absmax")
    if a.ndim != 1 or np.any(a < 0) or not 0 <= alpha <= 1:
        raise ValueError("Invalid channel maxima or alpha")
    maxima = np.zeros_like(a)
    found = False
    for weight in consumer_weights:
        w = _finite(weight, "consumer weight")
        if w.ndim != 2 or w.shape[1] != a.size or w.shape[0] == 0:
            raise ValueError("Consumers must share the same input channels")
        maxima = np.maximum(maxima, np.max(np.abs(w), axis=0))
        found = True
    if not found:
        raise ValueError("At least one consumer is required")
    scale = np.maximum(a, 1e-12) ** alpha / np.maximum(maxima, 1e-12) ** (1 - alpha)
    # A dead input or all-zero weight column needs no reparameterization.
    scale[(a == 0) | (maxima == 0)] = 1.0
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Scale is not representable")
    return scale


def fold_shared_rmsnorm(gamma_parameter, consumer_weights, scale, *, unit_offset=False):
    """Return folded gamma and all consumer matrices, without mutating inputs.

    Set unit_offset=True for a stored parameter whose effective gain is 1+gamma.
    Explicitly supply every consumer; this API cannot discover a model graph.
    Residual branches that bypass the norm are not modified.
    """
    gamma = _finite(gamma_parameter, "gamma")
    s = _finite(scale, "scale")
    if gamma.ndim != 1 or s.shape != gamma.shape or np.any(s <= 0):
        raise ValueError("Gamma/scale dimensions or values are invalid")
    folded = []
    for weight in consumer_weights:
        w = _finite(weight, "consumer weight")
        if w.ndim != 2 or w.shape[1] != s.size:
            raise ValueError("Every consumer must use the shared scale")
        folded.append(w * s)
    if not folded:
        raise ValueError("At least one consumer is required")
    offset = float(unit_offset)
    return (gamma + offset) / s - offset, folded


def fold_geglu_up(up_weight, down_weight, scale):
    """Bias-free GeGLU: (up/s) * gelu(gate), followed by down*diag(s)."""
    up = _finite(up_weight, "up")
    down = _finite(down_weight, "down")
    s = _finite(scale, "scale")
    if (up.ndim != 2 or down.ndim != 2 or s.ndim != 1 or np.any(s <= 0)
            or up.shape[0] != s.size or down.shape[1] != s.size):
        raise ValueError("GeGLU scale dimensions are invalid")
    return up / s[:, None], down * s


def transform_tile_hessian(hessian, scale, *, rotate_h8=False):
    """Transform REAL full E[xx^T] to x/scale and optional H8 coordinates.

    A diagonal channel statistic does not supply missing cross correlations.
    Collect tile covariances in a second calibration pass when GPTQ is enabled.
    """
    h = _finite(hessian, "H")
    s = _finite(scale, "scale")
    if h.shape != (256, 256) or s.shape != (256,) or np.any(s <= 0):
        raise ValueError("Expected a 256x256 Hessian and 256 positive scales")
    if not np.allclose(h, h.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Hessian must be symmetric")
    result = h / s[:, None] / s[None, :]
    if rotate_h8:
        from .bridge import H8_MATRIX
        rotation = np.kron(np.eye(32), H8_MATRIX.astype(np.float64))
        result = rotation @ result @ rotation.T
    return result


def decode_opt5_36(payload):
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size == 0 or raw.size % 36:
        raise ValueError("Expected whole 36-byte glyphs")
    records = raw.reshape(-1, 36)
    scalebias = records[:, :4].copy().view("<f2").astype(np.float64)
    signs = np.unpackbits(records[:, 4:], axis=1).astype(np.float64) * 2 - 1
    return scalebias[:, :1] * signs + scalebias[:, 1:2]


def gptq_opt5_tile(weights, hessian, *, damping=0.01):
    """Reference intra-tile second-order error feedback with fixed fp16 levels.

    Returns a standard MSB-first 36-byte glyph. The candidate is retained only
    when its REAL, undamped calibration objective beats the unscaled sign
    baseline. No compensation across tile boundaries, no model-quality claim.
    Reuse a factorization in a production encoder; this reference solves one tile.
    """
    w = _finite(weights, "weights")
    h = _finite(hessian, "H")
    if w.shape != (256,) or h.shape != (256, 256) or not math.isfinite(damping) or damping <= 0:
        raise ValueError("Expected a 256-weight tile, full Hessian and positive damping")
    if not np.allclose(h, h.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Hessian must be symmetric")
    if np.min(np.linalg.eigvalsh(h)) < -1e-9 * max(1.0, float(np.max(np.abs(h)))):
        raise ValueError("Hessian must be positive semidefinite")
    bias = np.float16(np.mean(w))
    scale = np.float16(np.mean(np.abs(w - float(bias))))
    if not np.isfinite(bias) or not np.isfinite(scale):
        raise ValueError("fp16 glyph parameters overflow")
    h_damped = h + np.eye(256) * max(float(np.mean(np.diag(h))), 1e-12) * damping
    upper = np.linalg.cholesky(np.linalg.solve(h_damped, np.eye(256))).T
    work = w.copy()
    signs = np.empty(256, dtype=np.uint8)
    for j in range(256):
        signs[j] = work[j] >= float(bias)
        q = float(bias) + float(scale) * (1 if signs[j] else -1)
        error = (work[j] - q) / upper[j, j]
        work[j:] -= error * upper[j, j:]
    baseline = (w >= float(bias)).astype(np.uint8)

    def objective(bits):
        error = w - (float(bias) + float(scale) * (2.0 * bits - 1))
        return float(error @ h @ error)

    loss_baseline = objective(baseline)
    loss_gptq = objective(signs)
    chosen = signs if loss_gptq < loss_baseline else baseline
    payload = np.array([scale, bias], dtype="<f2").tobytes() + np.packbits(chosen).tobytes()
    return payload, {"baseline_loss": loss_baseline, "gptq_loss": loss_gptq,
                     "selected_loss": min(loss_gptq, loss_baseline),
                     "selected": "gptq" if loss_gptq < loss_baseline else "baseline"}


def opt5_block_frobenius(payload, rows, input_blocks):
    """True decoded norm per input block across ALL output rows, bias included."""
    if rows <= 0 or input_blocks <= 0:
        raise ValueError("Positive matrix dimensions required")
    weights = decode_opt5_36(payload).reshape(rows, input_blocks, 256)
    squared = np.sum(weights * weights, axis=(0, 2))
    # Conservative accumulation allowance; fp16-derived values cannot underflow
    # their squares in float64. Include rounding of multiplication and sqrt.
    error = np.finfo(np.float64).eps * (rows * 256 + 3)
    if error >= 1:
        raise ValueError("Too many terms for the norm's rounding allowance")
    upper = np.nextafter(np.sqrt(squared / (1 - error)), np.inf)
    return np.where(squared == 0, 0.0, upper)


def bounded_skip_mask(input_vector, block_frobenius, epsilon):
    """Reference skip policy: sum of skipped-block L2 bounds <= epsilon.

    This certifies only this linear projection's absolute output error. It is
    not a bound on subsequent nonlinear layers, logits, or the distribution.
    """
    x = _finite(input_vector, "input")
    norms = _finite(block_frobenius, "block norms")
    if (x.ndim != 1 or norms.ndim != 1 or x.size != norms.size * 256
            or np.any(norms < 0) or not math.isfinite(epsilon) or epsilon < 0):
        raise ValueError("Invalid block norms, input or error budget")
    blocks = x.reshape(-1, 256)
    amplitude = np.max(np.abs(blocks), axis=1)
    scaled = blocks / np.where(amplitude == 0, 1.0, amplitude)[:, None]
    allowance = 1 - np.finfo(np.float64).eps * 1024
    input_norms = np.nextafter(amplitude * np.sqrt(np.sum(scaled * scaled, axis=1) / allowance), np.inf)
    input_norms[amplitude == 0] = 0
    bounds = np.nextafter(input_norms * norms, np.inf)
    bounds[(input_norms == 0) | (norms == 0)] = 0
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Norm bound overflow; refusing to skip")
    skipped = np.zeros(norms.size, dtype=bool)
    total = 0.0
    for i in np.argsort(bounds, kind="stable"):
        # Round upward so the selection cannot consume a rounded-down bound.
        next_total = np.nextafter(total + float(bounds[i]), np.inf) if bounds[i] else total
        if next_total <= epsilon:
            skipped[i], total = True, next_total
    return skipped, total


def mixed_budget(glyphs=114_401_280, *, stream_budget=4_800_000_000,
                 available_ram=5_000_000_000, runtime_reserve=200_000_000,
                 sidecar_bytes=0, high_glyphs=None):
    """Exact integer ledger, including header and explicit non-weight reserves."""
    values = (glyphs, stream_budget, available_ram, runtime_reserve, sidecar_bytes)
    if any(type(v) is not int or v < 0 for v in values) or glyphs == 0:
        raise ValueError("Budgets and counts must be nonnegative integers; glyphs > 0")
    cap = min(stream_budget, available_ram - runtime_reserve - sidecar_bytes)
    base = 16 + glyphs * 36
    if cap < base:
        raise ValueError("Even all-36-byte glyphs exceed the available weight budget")
    maximum = min(glyphs, (cap - base) // (92 - 36))
    if high_glyphs is None:
        high_glyphs = maximum
    if type(high_glyphs) is not int or not 0 <= high_glyphs <= maximum:
        raise ValueError(f"High-tier allocation exceeds maximum {maximum}")
    stream = base + high_glyphs * 56
    return {"glyphs": glyphs, "high_glyphs": high_glyphs,
            "low_glyphs": glyphs - high_glyphs, "max_high_glyphs": maximum,
            "high_fraction": high_glyphs / glyphs, "stream_bytes": stream,
            "sidecar_bytes": sidecar_bytes, "runtime_reserve_bytes": runtime_reserve,
            "total_reserved_bytes": stream + sidecar_bytes + runtime_reserve,
            "available_ram_bytes": available_ram,
            "quality_gate": "UNVERIFIED", "residency_gate": "UNVERIFIED"}

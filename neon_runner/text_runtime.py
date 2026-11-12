"""FP32 text-only Gemma 4 dense reference runtime over stored glyph weights.

Architecture source: Hugging Face transformers' public Gemma 4 modeling and
RoPE implementations (Apache-2.0). This independent NumPy implementation uses
the repository's native projection ABI. It is not a substitute checkpoint and
does not assert equivalence to unquantized weights or physical Pi performance.
"""
import ctypes
import json
import math
from pathlib import Path

import numpy as np

from .bridge import NeonRunnerBridgeV2, fwht8_vec_numpy
from .checkpoint_io import BF16Table, load_float_sidecar


def rmsnorm(values, gain=None, epsilon=1e-6):
    x = np.asarray(values, dtype=np.float32)
    normalized = x * np.power(np.mean(x*x, axis=-1, keepdims=True)+epsilon, -0.5)
    return normalized if gain is None else normalized*gain


def rotary(values, positions, theta, fraction=1.0):
    x = np.asarray(values, dtype=np.float32)
    dimension = x.shape[-1]
    pairs = int(dimension*fraction)//2
    inverse = np.power(theta, -np.arange(0, pairs*2, 2, dtype=np.float32)/dimension)
    angles = np.asarray(positions, dtype=np.float32)[:, None, None]*inverse[None, None, :]
    c, s = np.cos(angles), np.sin(angles)
    out = x.copy()
    first, second = x[..., :pairs], x[..., dimension//2:dimension//2+pairs]
    out[..., :pairs] = first*c-second*s
    out[..., dimension//2:dimension//2+pairs] = second*c+first*s
    return out


def causal_gqa(q, k, v, query_positions, key_positions, window=None):
    tokens, heads, dimension = q.shape
    kv_heads = k.shape[1]
    if heads % kv_heads or k.shape != v.shape:
        raise ValueError("Invalid grouped-query attention geometry")
    grouped = q.reshape(tokens, kv_heads, heads//kv_heads, dimension)
    scores = np.einsum("thgd,shd->hgts", grouped, k, optimize=True)
    distance = np.asarray(query_positions)[:, None]-np.asarray(key_positions)[None, :]
    allowed = distance >= 0
    if window is not None:
        allowed &= distance < window
    scores = np.where(allowed[None, None, :, :], scores, -np.inf)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probability = np.exp(scores)
    probability /= probability.sum(axis=-1, keepdims=True)
    return np.einsum("hgts,shd->thgd", probability, v, optimize=True).reshape(tokens, heads*dimension)


class GlyphTextRuntime:
    """Complete text forward pass with exact FP32 KV and the full output head.

    Missing norms/scalars are errors. This deliberately avoids sampling tricks,
    vocabulary shortlists, token hardcoding, tile dropping and optional models.
    The reference cache has an explicit sequence limit; production cache work
    must be quality-tested independently.
    """
    def __init__(self, directory, *, stream=None, layout=None, embedding_table=None,
                 scalars=None, max_sequence=2048, library=None, threads=4,
                 projection_backend=None):
        root = Path(directory)
        config = json.loads((root/"config.json").read_text())
        self.config = config.get("text_config", config)
        c = self.config
        if (c.get("enable_moe_block", False) or c.get("hidden_size_per_layer_input", 0)
                or c.get("num_kv_shared_layers", 0) or c.get("attention_bias", False)):
            raise ValueError("This reference supports the published dense text configuration only")
        self.dimension, self.layers, self.vocabulary = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
        self.heads, self.epsilon = c["num_attention_heads"], c.get("rms_norm_eps", 1e-6)
        self.max_sequence = max_sequence
        self.norms = load_float_sidecar(root/"gemma4_31b_norms.safetensors")
        self.scalars = np.load(scalars or root/"layer_scalars_qat.npy", allow_pickle=False)
        if self.scalars.shape != (self.layers,) or not np.isfinite(self.scalars).all():
            raise ValueError("Missing/invalid checkpoint layer scalars")
        required = ["model.language_model.norm.weight"]
        for layer in range(self.layers):
            prefix = f"model.language_model.layers.{layer}"
            required.extend(prefix+"."+name+".weight" for name in
                            ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                             "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm"))
        if any(name not in self.norms for name in required):
            raise ValueError("Checkpoint norm sidecar is incomplete")
        for name in required:
            expected = self.dimension
            if ".self_attn." in name:
                layer = int(name.split(".")[3])
                expected = c["global_head_dim"] if c["layer_types"][layer] == "full_attention" else c["head_dim"]
            if self.norms[name].shape != (expected,):
                raise ValueError(f"Checkpoint norm shape mismatch: {name}")
        self.projection_backend = projection_backend
        self.bridge = None
        layout_path = layout or root/"gemma4_31b_tensor_layout.json"
        if projection_backend is None:
            self.bridge = NeonRunnerBridgeV2(str(stream or root/"gemma4_31b_language_u_sandwich_stream.bin"),
                                            str(layout_path), codebook_family="auto",
                                            codebook_path=str(root/"s7_codebook_2048x8.bin"), so_path=library)
        metadata = self.bridge.layout if self.bridge is not None else json.loads(Path(layout_path).read_text())
        # The legacy native self-test sets the OpenMP thread count to one.
        # Restore the explicit runtime setting after that correctness check.
        if self.bridge is not None and hasattr(self.bridge.accelerator.lib, "omp_set_num_threads"):
            configure = self.bridge.accelerator.lib.omp_set_num_threads
            configure.argtypes, configure.restype = [ctypes.c_int], None
            configure(threads)
        self.projections = {}
        for p in metadata["projections"]:
            self.projections.setdefault(p["layer"], {})[p.get("tensor_name", p.get("name"))] = p
        self.embedding = embedding_table or BF16Table(root/"gemma4_31b_embed_tokens.bin", self.vocabulary, self.dimension)
        if self.embedding.rows != self.vocabulary or self.embedding.columns != self.dimension:
            raise ValueError("Embedding vocabulary/hidden dimensions do not match the model")
        self.reset()

    def reset(self):
        self.position = 0
        self.cache = [None for _ in range(self.layers)]

    def project(self, layer, name, inputs):
        metadata = self.projections[layer][name+".weight_packed"]
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != metadata["d_in"]:
            raise ValueError("Projection activation shape mismatch")
        if self.projection_backend is not None:
            return self.projection_backend.project(metadata, x)
        count = x.shape[0]
        codec, width = metadata["codec_id"], metadata["bytes_per_glyph"]
        acc = self.bridge.accelerator
        multi = acc.has_opt4_multi if codec == 4 else acc.has_v2_opt5_multi
        if count == 1 or not multi or (codec == 4 and width == 92):
            return np.stack([self.bridge.gemv(metadata, row) for row in x])
        if metadata.get("rotation_mode", self.bridge.layout.get("rotation_mode", "direct")) == "fwht8":
            x = np.stack([fwht8_vec_numpy(row) for row in x])
        out = np.empty((count, metadata["d_out"]), dtype=np.float32)
        pointer_array = ctypes.c_void_p*count
        xp = pointer_array(*(x[i].ctypes.data for i in range(count)))
        yp = pointer_array(*(out[i].ctypes.data for i in range(count)))
        common = [ctypes.c_void_p(self.bridge._mmap_ptr), metadata["stream_byte_offset"]]
        if codec == 4:
            acc.lib.gemv_opt4_mmap_multi(*common, self.bridge.codebook.array.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.cast(xp, ctypes.c_void_p), ctypes.cast(yp, ctypes.c_void_p), count, metadata["d_out"], metadata["d_in"])
        else:
            acc.lib.gemv_opt5_mmap_multi(*common, ctypes.cast(xp, ctypes.c_void_p), ctypes.cast(yp, ctypes.c_void_p), count, metadata["d_out"], metadata["d_in"], width)
        return out

    def forward(self, token_ids, capture=None, progress=None):
        ids = list(token_ids)
        if not ids or self.position+len(ids) > self.max_sequence:
            raise ValueError("Empty input or reference sequence capacity exceeded")
        positions = np.arange(self.position, self.position+len(ids))
        c = self.config
        states = np.stack([self.embedding.row(int(token)) for token in ids])*np.float32(math.sqrt(self.dimension))
        for layer in range(self.layers):
            full = c["layer_types"][layer] == "full_attention"
            dimension = c["global_head_dim"] if full else c["head_dim"]
            kv = c["num_global_key_value_heads"] if full else c["num_key_value_heads"]
            prefix = f"model.language_model.layers.{layer}"
            gain = lambda name: self.norms[prefix+"."+name+".weight"]
            inputs = rmsnorm(states, gain("input_layernorm"), self.epsilon)
            if capture is not None: capture[f"L{layer:02d}.attn"] = inputs.copy()
            q = self.project(layer, "self_attn.q_proj", inputs).reshape(len(ids), self.heads, dimension)
            raw_k = self.project(layer, "self_attn.k_proj", inputs).reshape(len(ids), kv, dimension)
            raw_v = raw_k if full and c["attention_k_eq_v"] else self.project(layer, "self_attn.v_proj", inputs).reshape(len(ids), kv, dimension)
            q = rmsnorm(q, gain("self_attn.q_norm"), self.epsilon)
            k = rmsnorm(raw_k, gain("self_attn.k_norm"), self.epsilon)
            v = rmsnorm(raw_v, epsilon=self.epsilon)
            rope = c["rope_parameters"]["full_attention" if full else "sliding_attention"]
            q = rotary(q, positions, rope["rope_theta"], rope.get("partial_rotary_factor", 1.0))
            k = rotary(k, positions, rope["rope_theta"], rope.get("partial_rotary_factor", 1.0))
            previous = self.cache[layer]
            if previous is not None:
                k = np.concatenate((previous[0], k)); v = np.concatenate((previous[1], v))
                key_positions = np.concatenate((previous[2], positions))
            else:
                key_positions = positions
            window = None if full else c["sliding_window"]
            context = causal_gqa(q, k, v, positions, key_positions, window)
            keep = len(k) if window is None else min(window, len(k))
            self.cache[layer] = (k[-keep:].copy(), v[-keep:].copy(), key_positions[-keep:].copy())
            if capture is not None: capture[f"L{layer:02d}.out"] = context.copy()
            attention = self.project(layer, "self_attn.o_proj", context)
            states = states+rmsnorm(attention, gain("post_attention_layernorm"), self.epsilon)
            inputs = rmsnorm(states, gain("pre_feedforward_layernorm"), self.epsilon)
            if capture is not None: capture[f"L{layer:02d}.mlp"] = inputs.copy()
            gate = self.project(layer, "mlp.gate_proj", inputs)
            up = self.project(layer, "mlp.up_proj", inputs)
            activated = (0.5*gate*(1+np.tanh(np.float32(math.sqrt(2/math.pi))*(gate+np.float32(0.044715)*gate**3))))*up
            if capture is not None: capture[f"L{layer:02d}.down"] = activated.copy()
            down = self.project(layer, "mlp.down_proj", activated)
            states = (states+rmsnorm(down, gain("post_feedforward_layernorm"), self.epsilon))*self.scalars[layer]
            if not np.isfinite(states).all():
                raise ValueError(f"Non-finite activation after layer {layer}")
            if progress is not None: progress(layer)
        self.position += len(ids)
        return rmsnorm(states[-1], self.norms["model.language_model.norm.weight"], self.epsilon)

    def logits(self, final_hidden):
        raw = self.embedding.logits(final_hidden)
        if raw.shape != (self.vocabulary,) or not np.isfinite(raw).all():
            raise ValueError("Invalid full-vocabulary logits")
        cap = self.config.get("final_logit_softcapping")
        return raw if not cap else np.tanh(raw/np.float32(cap))*np.float32(cap)

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
        if self.projection_backend is not None:
            self.projection_backend.close()
        self.embedding.close()

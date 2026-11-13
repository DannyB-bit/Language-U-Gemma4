"""Small arithmetic/container tests; not model quality or Pi performance."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from neon_runner.checkpoint_io import (BF16Table, QATInt4Projections, NativeQATInt4Projections,
    SafetensorsFile, bf16_to_float32, load_float_sidecar, round_bf16)
from neon_runner.text_runtime import causal_gqa, rmsnorm, rotary
from neon_runner.lowbit_embedding import (ThreeBitTable, encode_blocks, decode_blocks,
                                          pack_codes, unpack_codes, encode_table)


def safetensors_fixture(path, arrays):
    header, blobs, offset = {}, [], 0
    for name, (dtype, value) in arrays.items():
        data = value.tobytes()
        header[name] = dict(dtype=dtype, shape=list(value.shape), data_offsets=[offset, offset+len(data)])
        blobs.append(data); offset += len(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded))+encoded+b''.join(blobs))


class TextRuntimeTests(unittest.TestCase):
    def test_three_bit_codes_and_complete_tied_head(self):
        codes = np.tile(np.arange(-4, 4, dtype=np.int8), 64).reshape(2, 256)
        np.testing.assert_array_equal(unpack_codes(pack_codes(codes)), codes)
        values = np.stack([np.zeros(256), np.linspace(-1, 1, 256), np.sin(np.arange(256))]).astype('f4')
        records = encode_blocks(values)
        self.assertEqual(records.shape, (3, 98))
        decoded = decode_blocks(records)
        np.testing.assert_array_equal(decoded[0], 0)
        self.assertLess(float(np.linalg.norm(decoded-values)/np.linalg.norm(values)), .25)
        with tempfile.TemporaryDirectory() as d:
            # Container/complete-head unit fixture, not a checkpoint quality test.
            source_path = Path(d)/'bf16.bin'
            source_path.write_bytes((round_bf16(values).view('u4')>>16).astype('<u2').tobytes())
            source = BF16Table(source_path, 3, 256)
            packed_path = Path(d)/'embedding.lue3'
            try: self.assertEqual(encode_table(source, packed_path, chunk_rows=1), 32+3*98)
            finally: source.close()
            table = ThreeBitTable(packed_path)
            try:
                x = np.cos(np.arange(256,dtype='f4'))
                expected = np.stack([table.row(i) for i in range(3)])@x
                np.testing.assert_allclose(table.logits(x, chunk_rows=1),expected,atol=1e-5)
                with self.assertRaises(ValueError): table.row(-1)
            finally: table.close()

    def test_bf16_bits_and_full_vocabulary_head(self):
        bits = np.array([[0x3f80, 0x4000], [0xbf80, 0x3f00], [0, 0xc000]], dtype='<u2')
        expected = np.array([[1, 2], [-1, .5], [0, -2]], dtype='f4')
        np.testing.assert_array_equal(bf16_to_float32(bits), expected)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'embedding.bin'; path.write_bytes(bits.tobytes())
            table = BF16Table(path, 3, 2)
            try:
                np.testing.assert_array_equal(table.row(1), expected[1])
                np.testing.assert_array_equal(table.logits([2, -3], chunk_rows=1), expected@np.array([2, -3]))
                with self.assertRaises(ValueError): table.row(3)
            finally: table.close()

    def test_norm_raw_gain_and_partial_rope(self):
        x = np.arange(1, 17, dtype='f4').reshape(1, 2, 8)
        gain = np.linspace(.25, 2, 8).astype('f4')
        np.testing.assert_allclose(rmsnorm(x, gain), x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-6)*gain, rtol=1e-6)
        out = rotary(x, [7], 10000., .5)
        # Partial RoPE pairs the full head's two halves; unused coordinates stay intact.
        np.testing.assert_array_equal(out[..., [2, 3, 6, 7]], x[..., [2, 3, 6, 7]])
        angle = 7/10000.**(2/8)
        np.testing.assert_allclose(out[..., 1], x[..., 1]*np.cos(angle)-x[..., 5]*np.sin(angle), rtol=1e-5)
        np.testing.assert_allclose(np.sum(out*out, axis=-1), np.sum(x*x, axis=-1), rtol=2e-6)

    def test_batched_gqa_matches_explicit_causal_and_incremental(self):
        q = np.sin(np.arange(4*4*8, dtype='f4')).reshape(4, 4, 8)
        k = np.cos(np.arange(4*2*8, dtype='f4')).reshape(4, 2, 8)
        v = np.sin(k*2)
        positions = np.arange(4)
        for window in (None, 2):
            out = causal_gqa(q, k, v, positions, positions, window).reshape(4, 4, 8)
            for t in range(4):
                begin = 0 if window is None else max(0, t-window+1)
                for h in range(4):
                    scores = k[begin:t+1, h//2]@q[t, h]
                    p = np.exp(scores-scores.max()); p /= p.sum()
                    np.testing.assert_allclose(out[t, h], p@v[begin:t+1, h//2], atol=1e-6)
                incremental = causal_gqa(q[t:t+1], k[:t+1], v[:t+1], [t], positions[:t+1], window)
                np.testing.assert_allclose(incremental.reshape(4, 8), out[t], atol=1e-6)

    def test_float_sidecar_and_reject_bad_ranges(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'norms.safetensors'
            safetensors_fixture(path, {'model.language_model.norm.weight':('BF16', np.array([0x3f80, 0x4000],dtype='<u2'))})
            np.testing.assert_array_equal(load_float_sidecar(path)['model.language_model.norm.weight'], [1, 2])
            header = {'model.language_model.norm.weight':dict(dtype='F32', shape=[2**63, 2**63],data_offsets=[0,0])}
            encoded = json.dumps(header).encode(); path.write_bytes(struct.pack('<Q',len(encoded))+encoded)
            with self.assertRaises(ValueError): load_float_sidecar(path)
            with self.assertRaises(ValueError):
                file = SafetensorsFile(path)
                try: file.tensor('model.language_model.norm.weight')
                finally: file.close()

    def test_group32_int4_source_scales_are_applied(self):
        # Exhaust all signed codes with different real scale bit patterns per group.
        integers = np.tile(np.arange(-8, 8, dtype=np.int8), 8).reshape(2,64)
        codes = (integers.astype(np.int16)+8).astype('u1')
        packed_bytes = codes[:,0::2] | (codes[:,1::2]<<4)
        packed = packed_bytes.copy().view('<i4')
        scale_bits = np.array([[0x3dcd, 0x3e4d], [0x3e99, 0x3f00]],dtype='<u2')
        prefix = 'model.language_model.layers.0.self_attn.q_proj'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.safetensors'
            safetensors_fixture(path, {prefix+'.weight_packed':('I32',packed), prefix+'.weight_scale':('BF16',scale_bits), prefix+'.weight_shape':('I64',np.array([2,64],dtype='<i8'))})
            backend=QATInt4Projections(path,chunk_rows=1)
            meta=dict(full_key=prefix+'.weight_packed', d_in=64,d_out=2)
            try:
                weights=round_bf16(integers*np.repeat(bf16_to_float32(scale_bits),32,axis=1))
                np.testing.assert_array_equal(backend.rows(meta,0,2),weights)
                x=np.cos(np.arange(128,dtype='f4')).reshape(2,64)
                np.testing.assert_allclose(backend.project(meta,x),x@weights.T,rtol=1e-5,atol=1e-5)
                native=NativeQATInt4Projections(path,threads=2)
                try:
                    np.testing.assert_allclose(native.project(meta,x),backend.project(meta,x),rtol=1e-5,atol=1e-5)
                    # Exercise native multi-call batching across its 64-token bound.
                    many=np.tile(x,(33,1))[:65]
                    np.testing.assert_allclose(native.project(meta,many),backend.project(meta,many),rtol=1e-5,atol=1e-5)
                finally: native.close()
            finally: backend.close()


if __name__ == '__main__': unittest.main()

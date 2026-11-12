"""Joint glyph representation and integer-program integrity, without downloads."""
import hashlib
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/joint_glyphs'))
from joint_codec import (Glyph, QAT4, OPT5, EXACT16, EXACT32, ADD, MUL, ASSERT_EQ,
                         STREAM, RECORD, pack_tokens, unpack_tokens, exact_encode,
                         decode_weights, opt5_encode, instruction, compile_chunk,
                         decode_chunk, execute)


class JointGlyphTests(unittest.TestCase):
    def setUp(self):
        self.tok = hashlib.sha256(b'test tokenizer').digest()
        self.w = np.linspace(-.2,.3,256).astype('<f2')
        self.source = hashlib.sha256(self.w.astype('<f4').tobytes()).digest()
        self.glyph = Glyph(list(range(20)), EXACT16, exact_encode(self.w,'fp16'),
                           instruction(MUL,left=12,right=8))

    def raw(self):
        return compile_chunk([self.glyph], self.tok, self.source)

    def test_token_18bit_boundaries_and_padding(self):
        for n in (20,21):
            ids = [0,262143] + list(range(n-2))
            raw = pack_tokens(ids)
            self.assertEqual(len(raw),(18*n+7)//8)
            self.assertEqual(unpack_tokens(raw,n),ids)
        bad = bytearray(pack_tokens(list(range(21))));bad[-1] |= 0x80
        with self.assertRaises(ValueError):unpack_tokens(bad,21)
        for ids in (list(range(19)),list(range(22)),[-1]+[0]*19,[262144]+[0]*19):
            with self.assertRaises(ValueError):pack_tokens(ids)

    def test_exact_fp16_and_fp32_bits(self):
        rng=np.random.default_rng(781)
        for dtype,codec,dt in [('fp16',EXACT16,'<f2'),('fp32',EXACT32,'<f4')]:
            w=rng.normal(size=256).astype(dt);w[0]=0.;w[1]=-0.
            encoded=exact_encode(w,dtype)
            self.assertEqual(decode_weights(codec,encoded,dtype).tobytes(),w.tobytes())
            self.assertEqual(decode_weights(codec,encoded).tobytes(),w.astype('<f4').tobytes())

    def test_fp32_outside_fp16_range_and_lossy_cast_rejected(self):
        w=np.linspace(1e20,2e20,256).astype('<f4')
        lane=exact_encode(w,'fp32')
        self.assertEqual(decode_weights(EXACT32,lane).tobytes(),w.tobytes())
        with self.assertRaises(ValueError):decode_weights(EXACT32,lane,'fp16')

    def test_qat_all_codes_and_groups(self):
        codes=np.tile(np.arange(16,dtype='u1'),16)
        packed=(codes[::2] | (codes[1::2]<<4)).tobytes()
        scales=(np.arange(1,9,dtype='f4')/32)
        bf16=(scales.view('u4')>>16).astype('<u2').tobytes()
        expected=(codes.astype('f4')-8)*np.repeat(scales,32)
        got=decode_weights(QAT4,packed+bf16)
        np.testing.assert_array_equal(got,expected)
        with self.assertRaises(ValueError):decode_weights(QAT4,(packed+bf16)[:-1])

    def test_roundtrip_and_instruction_execution(self):
        restored,source=decode_chunk(self.raw(),self.tok)
        self.assertEqual(restored[0].tokens,self.glyph.tokens)
        self.assertEqual(source,self.source)
        self.assertEqual(execute(restored)['result'],96)

    def test_full_20_point_4_average(self):
        counts=[20,20,20,21,21]
        glyphs=[Glyph(list(range(n)),EXACT16,self.glyph.weights,
                      instruction(ADD,left=i,right=1)) for i,n in enumerate(counts)]
        source=hashlib.sha256(self.w.astype('<f4').tobytes()*5).digest()
        restored,_=decode_chunk(compile_chunk(glyphs,self.tok,source),self.tok)
        self.assertEqual(sum(len(g.tokens) for g in restored),102)
        self.assertEqual(sum(len(decode_weights(g.codec,g.weights)) for g in restored),1280)

    def test_corruption_truncation_append_and_tokenizer(self):
        raw=self.raw()
        for bad in [raw[:-1],raw+b'X',raw[:10],raw[:120]+bytes([raw[120]^1])+raw[121:]]:
            with self.assertRaises(ValueError):decode_chunk(bad,self.tok)
        with self.assertRaises(ValueError):decode_chunk(raw,b'X'*32)
        with self.assertRaises(ValueError):compile_chunk([self.glyph],self.tok,b'X'*32)

    def test_record_boundary_and_order_after_valid_digest(self):
        raw=self.raw()
        for offset,value,fmt in [(0,3,'<I'),(6,1,'<H'),(8,4000,'<I')]:
            body=bytearray(raw[STREAM.size:]);struct.pack_into(fmt,body,offset,value)
            header=bytearray(raw[:STREAM.size]);header[-32:]=hashlib.sha256(body).digest()
            with self.assertRaises(ValueError):decode_chunk(header+body,self.tok)

    def test_reasoning_dependency_constraint_overflow(self):
        for op in [instruction(ADD,left_ref=0,right=1),instruction(ASSERT_EQ,left=3,right=4),
                   instruction(MUL,left=2**62,right=4)]:
            g=Glyph(self.glyph.tokens,EXACT16,self.glyph.weights,op)
            with self.assertRaises(ValueError):compile_chunk([g],self.tok,self.source)

    def test_malformed_repair_and_weight_count(self):
        for raw in [bytes(36),bytes(36)+b'\x02',bytes(36)+b'\x01broken']:
            with self.assertRaises(ValueError):decode_weights(EXACT16,raw)
        with self.assertRaises(ValueError):exact_encode(self.w[:-1],'fp16')

    def test_approximate_control_does_not_restore_source(self):
        q=decode_weights(OPT5,opt5_encode(self.w))
        self.assertFalse(np.array_equal(q,self.w.astype('f4')))


if __name__=='__main__':
    unittest.main()

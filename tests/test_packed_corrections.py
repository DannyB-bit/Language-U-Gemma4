"""Offline format and independent arithmetic checks; no model downloads."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/packed_corrections"))
from correction_codec import Correction, save_correction
from binary_factor_probe import PackedFactors, save_packed


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "correction.rcf"
        rng = np.random.default_rng(472)
        self.a = rng.normal(size=(5, 24))
        self.b = rng.normal(size=(11, 5))
        self.x = rng.normal(size=24).astype("f4")

    def test_stored_coefficients_match_independent_oracle(self):
        for dtype in ("fp16", "int8"):
            with self.subTest(dtype=dtype):
                path = self.path.with_suffix('.' + dtype)
                size = save_correction(path, self.a, self.b, dtype)
                c = Correction(path)
                a = c.a.astype("f8") * c.sa[:, None]
                b = c.b.astype("f8") * c.sb[:, None]
                expected = b @ (a @ self.x.astype("f8"))
                for chunk in (1, 3, 32):
                    np.testing.assert_allclose(c.project(self.x, chunk), expected, rtol=2e-5, atol=2e-6)
                wanted = 64 + (2 * (5*24 + 11*5) if dtype == "fp16" else 5*24 + 11*5 + 4*(5+11))
                self.assertEqual(size, wanted)

    def test_zero_rows_remain_zero(self):
        save_correction(self.path, np.zeros_like(self.a), self.b)
        np.testing.assert_array_equal(Correction(self.path).project(self.x), np.zeros(11))

    def test_invalid_shapes_and_nonfinite_factors(self):
        for a, b in ((self.a, self.b[:, :4]), (np.full_like(self.a, np.nan), self.b),
                     (self.a, np.full_like(self.b, np.inf)), (self.a[:0], self.b[:, :0])):
            with self.assertRaises(ValueError):
                save_correction(self.path, a, b)
        self.assertFalse(self.path.exists())

    def test_fp16_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            save_correction(self.path, self.a * 1e10, self.b, "fp16")

    def test_truncated_extended_and_bad_header_rejected(self):
        save_correction(self.path, self.a, self.b)
        raw = self.path.read_bytes()
        variants = [raw[:20], raw[:-1], raw + b'\x00', b'BAD!' + raw[4:]]
        for offset, value in ((4, 9), (8, 0), (20, 3), (24, 1), (28, 1)):
            bad = bytearray(raw)
            struct.pack_into('<I', bad, offset, value)
            variants.append(bad)
        for i, bad in enumerate(variants):
            path = Path(self.temp.name) / f'bad{i}'
            path.write_bytes(bad)
            with self.assertRaises(ValueError):
                Correction(path)

    def test_invalid_scale_and_nonfinite_stored_fp16(self):
        save_correction(self.path, self.a, self.b)
        raw = bytearray(self.path.read_bytes())
        for value in (float('nan'), float('inf'), 0., -1.):
            struct.pack_into('<f', raw, 64, value)
            p = Path(self.temp.name) / 'bad-scale'
            p.write_bytes(raw)
            with self.assertRaises(ValueError):
                Correction(p)
        p = Path(self.temp.name) / 'fp16'
        save_correction(p, self.a, self.b, 'fp16')
        raw = bytearray(p.read_bytes())
        struct.pack_into('<H', raw, 64, 0x7c00)
        p.write_bytes(raw)
        with self.assertRaises(ValueError):
            Correction(p)

    def test_invalid_input_and_chunk(self):
        save_correction(self.path, self.a, self.b)
        c = Correction(self.path)
        for x in (self.x[:-1], self.x[None, :], np.full_like(self.x, np.nan)):
            with self.assertRaises(ValueError):
                c.project(x)
        for chunk in (0, -1, 1.2):
            with self.assertRaises(ValueError):
                c.project(self.x, chunk)

    def test_exclusive_write_protects_existing_result(self):
        save_correction(self.path, self.a, self.b)
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_correction(self.path, self.a * 2, self.b)
        self.assertEqual(self.path.read_bytes(), before)

    def test_combined_packed_base_and_correction(self):
        rng = np.random.default_rng(123)
        u = np.where(rng.random((11, 8)) > .5, 1., -1.)
        v = np.where(rng.random((8, 24)) > .5, 1., -1.)
        basepath = Path(self.temp.name) / 'base.bfp'
        save_packed(basepath, u, v, np.ones(11)*.5, np.ones(24)*.25)
        base = PackedFactors(basepath)
        save_correction(self.path, self.a, self.b)
        c = Correction(self.path)
        a = c.a.astype('f8') * c.sa[:, None]
        b = c.b.astype('f8') * c.sb[:, None]
        expected = ((.5 * u @ v * .25) + b @ a) @ self.x.astype('f8')
        actual = base.project(self.x) + c.project(self.x)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-6)


if __name__ == '__main__':
    unittest.main()

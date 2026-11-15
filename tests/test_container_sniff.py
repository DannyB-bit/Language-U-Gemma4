"""Unit tests for neon_runner.container_sniff across T16C, LANU/LANP, LUJ1, and raw .glyph.bin."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from neon_runner.container_sniff import (
    ContainerSniffError,
    LANU_HEADER_FMT,
    LANU_MAGIC,
    LUJ1_HEADER_BYTES,
    LUJ1_MAGIC,
    T16C_HEADER_FMT,
    T16C_MAGIC,
    T16C_MAT_HEADER_FMT,
    sniff_container,
    tile16_blk_bytes,
)

ROOT = Path(__file__).resolve().parents[1]


class ContainerSniffTests(unittest.TestCase):
    def test_committed_samples_and_luj1_bank(self):
        opt4 = sniff_container(ROOT / "model" / "gemma4_e4b_sample_layer0_q_proj.neon_opt4.lanu")
        self.assertTrue(opt4.valid)
        self.assertEqual(opt4.format_id, "lanu_v2")
        self.assertTrue(opt4.details["layout_verified"])
        self.assertEqual(opt4.details["total_glyphs"], 160)

        hybrid = sniff_container(ROOT / "model" / "gemma4_e4b_sample_neon_hybrid_v2.lanu")
        self.assertTrue(hybrid.valid)
        self.assertTrue(hybrid.details["layout_verified"])
        self.assertEqual(hybrid.details["total_glyphs"], 320)

        raw = sniff_container(ROOT / "model" / "gemma4_e4b_sample_layer0_q_proj.glyph.bin")
        self.assertTrue(raw.valid)
        self.assertEqual(raw.details["bytes_per_glyph"], 48)
        self.assertEqual(raw.details["glyphs"], 16)

        luj = sniff_container(ROOT / "experiments" / "joint_glyphs" / "planner" / "planner.luj", verify_hash=True)
        self.assertTrue(luj.valid)
        self.assertTrue(luj.details["body_sha256_verified"])

    def test_lanu_truncation_and_header_lies_fail_closed(self):
        orig = (ROOT / "model" / "gemma4_e4b_sample_layer0_q_proj.neon_opt4.lanu").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            trunc = tdir / "trunc.lanu"
            trunc.write_bytes(orig[:-1])
            with self.assertRaises(ContainerSniffError):
                sniff_container(trunc)

            # Lying boundary count even when total size looks plausible
            lying = tdir / "lying.lanu"
            lying.write_bytes(struct.pack(LANU_HEADER_FMT, LANU_MAGIC, 2, 45, 5, 10) + b"\x00" * (5 * 132))
            with self.assertRaises(ContainerSniffError):
                sniff_container(lying)

            # Paired layout discrepancy caught immediately
            good = tdir / "good.lanu"
            good.write_bytes(orig)
            bad_layout = tdir / "good.layout.json"
            bad_layout.write_text(json.dumps({"stream_size_bytes": len(orig) + 16, "projections": []}), encoding="utf-8")
            with self.assertRaises(ContainerSniffError):
                sniff_container(good, layout_path=bad_layout)

    def test_t16c_v1_valid_modes_and_corruption(self):
        self.assertEqual(tile16_blk_bytes(256, 0 + 1, True), 576)
        self.assertEqual(tile16_blk_bytes(256, 5, False), 864)
        self.assertEqual(tile16_blk_bytes(256, 2, False), 1056)
        self.assertEqual(tile16_blk_bytes(256, 3, False), 1568)
        with self.assertRaises(ContainerSniffError):
            tile16_blk_bytes(256, 5, True)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tiny.t16c"
            mname = b"m0.weight"
            hdr = struct.pack(T16C_HEADER_FMT, T16C_MAGIC, 1, 3, 1, 1, 0)
            n_entry = struct.pack("<H", 2) + b"n0" + struct.pack("<I", 4) + (b"\x00" * 16)
            m_hdr = struct.pack(T16C_MAT_HEADER_FMT, len(mname), 16, 256, 256, 5, 0, 1, 1, 0, 864, 0)
            prefix = hdr + n_entry + m_hdr + mname
            pad = ((len(prefix) + 63) & ~63) - len(prefix)
            blob = prefix + (b"\x00" * pad) + (b"\x00" * 864) + (b"\x00" * (256 * 4))
            p.write_bytes(blob)

            sniffed = sniff_container(p)
            self.assertTrue(sniffed.valid)
            self.assertEqual(sniffed.details["mode_counts"]["b5_sym_bs256"], 1)

            # Truncated by 1 byte must fail
            p.write_bytes(blob[:-1])
            with self.assertRaises(ContainerSniffError):
                sniff_container(p)

    def test_raw_glyph_bin_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            odd = tdir / "odd.glyph.bin"
            odd.write_bytes(b"\x00" * 47)
            with self.assertRaises(ContainerSniffError):
                sniff_container(odd)

            inf_rec = tdir / "inf.glyph.bin"
            inf_rec.write_bytes(struct.pack("<ee", float("inf"), 0.0) + b"\x00" * 44)
            with self.assertRaises(ContainerSniffError):
                sniff_container(inf_rec)

            missing = tdir / "does_not_exist.lanu"
            res = sniff_container(missing, raise_on_error=False)
            self.assertFalse(res.valid)
            self.assertEqual(res.format_id, "missing")


if __name__ == "__main__":
    unittest.main()

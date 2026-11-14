"""Software correctness tests, not activation calibration or Pi 4 benchmarks.

Numeric probes use decoded committed Gemma sample weights. Hand-built records
exercise format boundaries and the missing-bias counterexample only.
"""
from copy import deepcopy
import ctypes
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

from neon_runner.bridge import NeonRunnerBridgeV2, NeonStreamAccelerator, pack_lanu_header, parse_lanu_header, H8_MATRIX
from neon_runner.codebooks import CodebookRegistryV2
from neon_runner.pi4_quantization import (
    ChannelStats, shared_channel_scale, fold_shared_rmsnorm, fold_geglu_up,
    transform_tile_hessian, gptq_opt5_tile, decode_opt5_36, mixed_budget,
    opt5_block_frobenius, bounded_skip_mask,
)
from neon_runner.pi4_stream import (
    checked_layout, pack_opt4_92, unpack_opt4_92, repack_stream, redundancy_scan,
)
from scripts.pi4_barrier_break import calibrate
from neon_runner.native_census import native_redundancy_scan

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
SOURCE = MODEL / "gemma4_e4b_sample_layer0_q_proj.neon_opt4.lanu"
LAYOUT = MODEL / "gemma4_e4b_sample_layer0_q_proj.neon_opt4.layout.json"
HYBRID = MODEL / "gemma4_e4b_sample_neon_hybrid_v2.lanu"
HYBRID_LAYOUT = MODEL / "gemma4_e4b_sample_neon_hybrid_v2.layout.json"


class Pi4BarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acc = NeonStreamAccelerator(auto_build=True)
        cls.cb = CodebookRegistryV2.resolve("compass_s7").array
        cls.raw = SOURCE.read_bytes()[16:]
        cls.weights = cls.acc.decode(cls.raw, cls.cb, num_threads=1).reshape(16, 2560)
        cls.probe = cls.weights[1].copy()

    def test_real_sample_exact_roundtrip(self):
        packed = pack_opt4_92(self.raw)
        self.assertEqual(len(packed), 160 * 92)
        self.assertEqual(unpack_opt4_92(packed), self.raw)

    def test_uploaded_31b_v1_header_and_legacy_stream_compatibility(self):
        # Actual 16-byte header read from the pinned HF 31B artifact.
        header = bytes.fromhex("4c414e5501002d0000a0d10600c02001")
        parsed = parse_lanu_header(header)
        self.assertEqual((parsed.version, parsed.total_glyphs, parsed.boundary_glyphs),
                         (1, 114401280, 18923520))
        self.assertEqual(pack_lanu_header(114401280, 18923520, version=1), header)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy-v1.lanu"
            old_header = parse_lanu_header(SOURCE.read_bytes()[:16])
            source.write_bytes(pack_lanu_header(old_header.total_glyphs, old_header.boundary_glyphs, version=1) + self.raw)
            checked_layout(source, LAYOUT)
            report = repack_stream(source, LAYOUT, Path(tmp) / "packed")
            with NeonRunnerBridgeV2(str(source), str(LAYOUT)) as legacy, \
                    NeonRunnerBridgeV2(report["stream"], report["layout"]) as packed:
                np.testing.assert_allclose(legacy.gemv(legacy.layout["projections"][0], self.probe),
                                           packed.gemv(packed.layout["projections"][0], self.probe),
                                           rtol=2e-5, atol=1e-5)

    def test_all_2048_indices_and_record_boundaries(self):
        # Unit fixtures only: cover every legal index and every bit alignment.
        indices = list(range(2048))
        raw = b"".join(struct.pack("<2e64H", 0.125, -0.25, *indices[i:i+64])
                       for i in range(0, len(indices), 64))
        self.assertEqual(unpack_opt4_92(pack_opt4_92(raw)), raw)
        for rotation in (False, True):
            expected = self.acc.decode(raw, self.cb, apply_fwht8=rotation, num_threads=1)
            actual = self.acc.decode_packed_opt4(pack_opt4_92(raw), self.cb,
                                                 apply_fwht8=rotation, num_threads=1)
            np.testing.assert_array_equal(actual, expected)

    def test_reject_noncanonical_indices_and_partial_records(self):
        corrupt = bytearray(self.raw)
        corrupt[4:6] = struct.pack("<H", 2048)
        with self.assertRaises(ValueError):
            pack_opt4_92(corrupt)
        with self.assertRaises(ValueError):
            pack_opt4_92(self.raw[:-1])
        with self.assertRaises(ValueError):
            unpack_opt4_92(b"x" * 91)

    def test_native_decode_bitwise_identical(self):
        for rotation in (False, True):
            old = self.acc.decode(self.raw, self.cb, apply_fwht8=rotation, num_threads=1)
            packed = self.acc.decode_packed_opt4(pack_opt4_92(self.raw), self.cb,
                                                 apply_fwht8=rotation, num_threads=1)
            np.testing.assert_array_equal(packed, old)

    def test_native_fwht_supports_in_place_attention_call(self):
        values = self.weights[0].copy()
        expected = (values.reshape(-1, 8) @ H8_MATRIX.T).reshape(-1)
        function = self.acc.lib.fwht8_vec_c
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        function.restype = None
        pointer = values.ctypes.data_as(ctypes.c_void_p)
        function(pointer, pointer, values.size)
        np.testing.assert_allclose(values, expected, atol=1e-7, rtol=1e-6)

    def test_hybrid_repack_and_native_gemv_both_rotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = repack_stream(HYBRID, HYBRID_LAYOUT, Path(tmp) / "packed", chunk_glyphs=7)
            self.assertEqual(report["saved_bytes"], 160 * 40)
            self.assertEqual(Path(report["stream"]).read_bytes()[:4], b"LANP")
            with NeonRunnerBridgeV2(str(HYBRID), str(HYBRID_LAYOUT)) as old, NeonRunnerBridgeV2(
                    report["stream"], report["layout"]) as new:
                for p_old, p_new in zip(old.layout["projections"], new.layout["projections"]):
                    for rotation in ("direct", "fwht8"):
                        po, pn = deepcopy(p_old), deepcopy(p_new)
                        po["rotation_mode"] = pn["rotation_mode"] = rotation
                        np.testing.assert_allclose(new.gemv(pn, self.probe), old.gemv(po, self.probe),
                                                   rtol=2e-5, atol=1e-5)
                p_old, p_new = old.layout["projections"][1], new.layout["projections"][1]
                self.assertEqual(bytes(old._memmap[p_old["stream_byte_offset"]:]),
                                 bytes(new._memmap[p_new["stream_byte_offset"]:]))

    def test_opt5_fwht_gemv_and_legacy_fallback(self):
        with NeonRunnerBridgeV2(str(HYBRID), str(HYBRID_LAYOUT)) as bridge:
            p = deepcopy(bridge.layout["projections"][1])
            p["rotation_mode"] = "fwht8"
            raw = bytes(bridge._memmap[p["stream_byte_offset"]:])
            decoded = self.acc.decode_opt5(raw, bytes_per_glyph=48, apply_fwht8=True).reshape(16, 2560)
            expected = decoded @ self.probe
            np.testing.assert_allclose(bridge.gemv(p, self.probe), expected, rtol=3e-5, atol=1e-5)
            bridge.accelerator.has_v2_opt5 = False
            np.testing.assert_allclose(bridge.gemv(p, self.probe), expected, rtol=3e-5, atol=1e-5)

    def test_native_call_rejects_invalid_ranges_and_formats(self):
        with NeonRunnerBridgeV2(str(SOURCE), str(LAYOUT)) as bridge:
            for change in ({"bytes_per_glyph": 91}, {"codec_id": 9}, {"d_in": 255},
                           {"stream_byte_offset": -1}, {"stream_byte_offset": 32},
                           {"tensor_bytes": 1}, {"rotation_mode": "unknown"}):
                bad = dict(bridge.layout["projections"][0], **change)
                with self.assertRaises(ValueError):
                    bridge.gemv(bad, self.probe)

    def test_layout_rejects_overlap_and_header_census(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            layout = json.loads(HYBRID_LAYOUT.read_text())
            layout["projections"][1]["stream_byte_offset"] = 16
            path.write_text(json.dumps(layout))
            with self.assertRaises(ValueError):
                checked_layout(HYBRID, path)
            corrupt = Path(tmp) / "bad.lanu"
            raw = SOURCE.read_bytes()
            corrupt.write_bytes(pack_lanu_header(159, 159) + raw[16:])
            with self.assertRaises(ValueError):
                checked_layout(corrupt, LAYOUT)

    def test_exact_scan_uses_full_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = redundancy_scan(HYBRID, HYBRID_LAYOUT, Path(tmp) / "unique.db")
            expected = set()
            layout = json.loads(HYBRID_LAYOUT.read_text())
            raw = HYBRID.read_bytes()
            for p in layout["projections"]:
                width = p["bytes_per_glyph"]
                start = p["stream_byte_offset"]
                for i in range(p["glyphs"]):
                    expected.add((p["codec_id"], width, p["rotation_mode"], raw[start+i*width:start+(i+1)*width]))
            self.assertEqual(report["unique_glyphs"], len(expected))
            self.assertEqual(report["payload_bytes"], len(raw) - 16)

    @unittest.skipUnless(sys.platform.startswith("linux") and ctypes.sizeof(ctypes.c_void_p) == 8, "64-bit Linux census")
    def test_native_census_matches_sqlite_and_resolves_forced_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = redundancy_scan(HYBRID, HYBRID_LAYOUT, Path(tmp) / "exact.db")
            for bits in (64, 1):
                actual = native_redundancy_scan(HYBRID, HYBRID_LAYOUT, hash_bits=bits)
                for field in ("glyphs", "unique_glyphs", "payload_bytes", "unique_payload_bytes", "zlib_bytes"):
                    self.assertEqual(actual[field], expected[field])
                if bits == 1:
                    self.assertGreater(actual["largest_fingerprint_group"], 2)

    @unittest.skipUnless(sys.platform.startswith("linux") and ctypes.sizeof(ctypes.c_void_p) == 8, "64-bit Linux census")
    def test_native_census_detects_exact_duplicate_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream, metadata = Path(tmp) / "duplicates.lanu", Path(tmp) / "duplicates.json"
            layout = json.loads(LAYOUT.read_text())
            count = 2 * (len(self.raw) // 132)
            stream.write_bytes(pack_lanu_header(count, count, version=1) + self.raw + self.raw)
            p = layout["projections"][0]
            p.update(d_out=32, glyphs=count, tensor_bytes=count*132, shape=[32, 2560])
            layout.update(total_macro_glyphs=count, stream_size_bytes=16+count*132)
            metadata.write_text(json.dumps(layout))
            result = native_redundancy_scan(stream, metadata, hash_bits=1)
            self.assertEqual(result["glyphs"], count)
            self.assertEqual(result["unique_glyphs"], count//2)

    @unittest.skipUnless(sys.platform.startswith("linux") and ctypes.sizeof(ctypes.c_void_p) == 8, "64-bit Linux census")
    def test_native_census_rejects_budget_and_wrong_artifact(self):
        with self.assertRaisesRegex(ValueError, "workspace budget"):
            native_redundancy_scan(SOURCE, LAYOUT, max_workspace=0)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            native_redundancy_scan(SOURCE, LAYOUT, expected_sha256="0"*64)

    def test_calibration_streaming_and_cli_export(self):
        stats = ChannelStats.empty(2560)
        for row in self.weights:
            stats.update(row)
        np.testing.assert_allclose(stats.h_diag, np.mean(self.weights.astype(np.float64)**2, axis=0))
        np.testing.assert_allclose(stats.absmax, np.max(np.abs(self.weights), axis=0))
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "probes.npy", Path(tmp) / "stats.npz"
            # Software test data from weights; explicitly NOT real activation calibration.
            np.save(source, self.weights)
            result = calibrate(source, dest, tile_hessians=True)
            self.assertEqual(result["samples"], 16)
            with np.load(dest, allow_pickle=False) as loaded:
                block = self.weights[:, :256].astype(np.float64)
                np.testing.assert_allclose(loaded["h_blocks"][0], block.T @ block / 16)

    def test_shared_norm_fold_including_unit_offset(self):
        stats = ChannelStats.empty(2560)
        stats.update(self.weights)
        consumers = [self.weights[:7], self.weights[7:11], self.weights[11:]]
        scale = shared_channel_scale(stats.absmax, consumers)
        gamma = self.weights[2].astype(np.float64)
        x = self.probe.astype(np.float64)
        for unit_offset in (False, True):
            gamma_new, folded = fold_shared_rmsnorm(gamma, consumers, scale, unit_offset=unit_offset)
            for original, rescaled in zip(consumers, folded):
                expected = original @ ((gamma + float(unit_offset)) * x)
                actual = rescaled @ ((gamma_new + float(unit_offset)) * x)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)

    def test_geglu_fold(self):
        up = self.weights.astype(np.float64)
        down = up[:, :16]
        scale = np.maximum(np.max(np.abs(up), axis=1), 1e-6)
        folded_up, folded_down = fold_geglu_up(up, down, scale)
        x = self.probe.astype(np.float64)
        gate = np.tanh(up @ x)  # Arbitrary fixed gate is enough to test the product identity.
        np.testing.assert_allclose(folded_down @ ((folded_up @ x) * gate),
                                   down @ ((up @ x) * gate), rtol=1e-12, atol=1e-10)

    def test_hessian_basis_transform(self):
        x = self.weights[:, :256].astype(np.float64)
        h = x.T @ x / len(x)
        scale = np.maximum(np.max(np.abs(x), axis=0), 1e-6)
        rotated = ((x / scale).reshape(-1, 8) @ H8_MATRIX.T).reshape(x.shape)
        actual = transform_tile_hessian(h, scale, rotate_h8=True)
        np.testing.assert_allclose(actual, rotated.T @ rotated / len(x), atol=1e-10, rtol=1e-10)

    def test_gptq_reference_cannot_worsen_measured_tile_objective(self):
        x = self.weights[:, :256].astype(np.float64)
        h = x.T @ x / len(x)
        payload, report = gptq_opt5_tile(x[2], h)
        decoded = decode_opt5_36(payload)[0]
        loss = (x[2] - decoded) @ h @ (x[2] - decoded)
        self.assertEqual(len(payload), 36)
        self.assertLessEqual(report["selected_loss"], report["baseline_loss"])
        self.assertAlmostEqual(loss, report["selected_loss"], places=8)
        with self.assertRaises(ValueError):
            gptq_opt5_tile(x[2], -np.eye(256))

    def test_bias_counterexample_and_total_skip_budget(self):
        # If alpha=0 and beta!=0, the spec's 16*abs(alpha) bound is zero and false.
        glyph = struct.pack("<2e", 0.0, 0.125) + bytes(32)
        payload = glyph * 4
        norms = opt5_block_frobenius(payload, rows=2, input_blocks=2)
        x = self.weights[1, :512].astype(np.float64)
        self.assertTrue(np.all(norms > 0))
        skip, bound = bounded_skip_mask(x, norms, epsilon=0.0)
        self.assertFalse(np.any(skip))
        epsilon = float(np.max(norms * np.linalg.norm(x.reshape(-1, 256), axis=1))) * 1.1
        skip, bound = bounded_skip_mask(x, norms, epsilon=epsilon)
        matrix = decode_opt5_36(payload).reshape(2, 512)
        removed = np.repeat(skip, 256) * x
        self.assertLessEqual(np.linalg.norm(matrix @ removed), bound + 1e-12)
        self.assertLessEqual(bound, epsilon)

    def test_exact_ram_budget_and_rejected_overflow(self):
        report = mixed_budget(high_glyphs=11_440_128)
        self.assertEqual(report["stream_bytes"], 4_759_093_264)
        self.assertLessEqual(report["total_reserved_bytes"], 5_000_000_000)
        maximum = mixed_budget()["max_high_glyphs"]
        with self.assertRaises(ValueError):
            mixed_budget(high_glyphs=maximum + 1)
        with self.assertRaises(ValueError):
            mixed_budget(runtime_reserve=1_000_000_000)
        # The newly available HF companion tables cannot be hidden in a 200MB
        # reserve: even all-36-byte linear weights plus those tables exceed 5GB.
        with self.assertRaisesRegex(ValueError, "all-36-byte"):
            mixed_budget(sidecar_bytes=2_818_572_288 + 2_990_082)

    def test_tile16_bitserial_gemm_and_cs_skip(self):
        self.assertTrue(self.acc.has_v2_tile16)
        self.assertTrue(self.acc.self_test())
        rng = np.random.default_rng(20261113)
        for d_out, d_in, bsize, n_bits, is_asym, k_tok in [
            (37, 256, 128, 1, True, 1),
            (32, 192, 64, 2, False, 4),
            (37, 256, 256, 3, True, 5),
            (16, 128, 128, 4, False, 1),
        ]:
            n_tiles = (d_out + 15) // 16
            n_blks = d_in // bsize
            hdr = 64 if is_asym else 32
            plane_b = (bsize // 8) * 16
            tblk = hdr + n_bits * plane_b
            packed = np.zeros(n_tiles * n_blks * tblk, dtype=np.uint8)
            norms = np.zeros(n_tiles * n_blks, dtype=np.uint16)
            w_ref = np.zeros((d_out, d_in), dtype=np.float32)

            for t in range(n_tiles):
                for blk in range(n_blks):
                    off = (t * n_blks + blk) * tblk
                    max_wn2 = 0.0
                    for r in range(16):
                        gr = t * 16 + r
                        if gr >= d_out:
                            continue
                        sc_h = np.float16(0.02 + 0.03 * float(rng.random()))
                        bi_h = np.float16(0.01 * (2.0 * float(rng.random()) - 1.0)) if is_asym else np.float16(0.0)
                        packed[off + r * 2 : off + r * 2 + 2] = np.frombuffer(sc_h.tobytes(), dtype=np.uint8)
                        if is_asym:
                            packed[off + 32 + r * 2 : off + 32 + r * 2 + 2] = np.frombuffer(bi_h.tobytes(), dtype=np.uint8)
                        wv = np.full(bsize, float(bi_h) if is_asym else 0.0, dtype=np.float32)
                        sc_f = float(sc_h)
                        for p in range(n_bits):
                            p_off = off + hdr + p * plane_b
                            b_row = rng.integers(0, 256, size=bsize // 8, dtype=np.uint8)
                            for bo in range(bsize // 8):
                                packed[p_off + bo * 16 + r] = b_row[bo]
                            bits = np.unpackbits(b_row).astype(np.float32) * 2.0 - 1.0
                            wv += sc_f * float(1 << p) * bits
                        w_ref[gr, blk * bsize : (blk + 1) * bsize] = wv
                        max_wn2 = max(max_wn2, float(np.sum(wv * wv)))
                    norms[t * n_blks + blk] = np.frombuffer(np.float16(np.sqrt(max_wn2)).tobytes(), dtype=np.uint16)[0]

            x = rng.normal(size=(k_tok, d_in)).astype(np.float32)
            x_arg = x[0] if k_tok == 1 else x
            y_got, sk0 = self.acc.gemm_bitserial_tile16(packed, x_arg, d_out, d_in, bsize, n_bits, is_asym, norms, 0.0, 2)
            self.assertEqual(sk0, 0)
            y_ref = x @ w_ref.T
            if k_tok == 1:
                y_mmap = self.acc.gemv_bitserial_tile16(packed.ctypes.data, x_arg, d_out, d_in, bsize, n_bits, is_asym, norms.ctypes.data, 0.0, 2)
                np.testing.assert_allclose(y_mmap, y_got, rtol=1e-5, atol=1e-5)
                y_ref_eval = y_ref[0]
            else:
                y_ref_eval = y_ref
            cos = float(np.dot(y_got.ravel(), y_ref_eval.ravel()) / (np.linalg.norm(y_got) * np.linalg.norm(y_ref_eval) + 1e-12))
            self.assertGreater(cos, 0.999)

            y_skip, sk_all = self.acc.gemm_bitserial_tile16(packed, x_arg, d_out, d_in, bsize, n_bits, is_asym, norms, 1e9, 2)
            n_k_chunks = (k_tok + 3) // 4
            self.assertEqual(sk_all, n_k_chunks * n_tiles * n_blks)
            self.assertEqual(float(np.max(np.abs(y_skip))), 0.0)

        with self.assertRaises(RuntimeError):
            self.acc.gemm_bitserial_tile16(np.zeros(64, dtype=np.uint8), np.zeros(128, dtype=np.float32), 16, 128, block_size=32)


if __name__ == "__main__":
    unittest.main()

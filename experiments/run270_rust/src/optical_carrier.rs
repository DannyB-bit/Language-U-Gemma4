//! OpticalCarrierV1 (`AIQC` 2D Physical Macro-Glyph & Token Bus) native Rust codec & DMA reader.
//!
//! Implements the canonical 20-byte big-endian `AIQC` protocol (`docs/OPTICAL_CONTEXT_INTEGRATION_V2_2.md`):
//!   - `[0..4)`   Magic `"AIQC"`
//!   - `[4]`      Version `0x01`
//!   - `[5]`      Payload type (`0x02` Token IDs, `0x04` Glyph + Token Propositions)
//!   - `[6..8)`   Reserved (`0x0000`)
//!   - `[8..12)`  Uncompressed payload length (`u32` BE)
//!   - `[12..16)` Stored payload length (`u32` BE; equals uncompressed for zero-dep raw store)
//!   - `[16..20)` IEEE 802.3 CRC32 over the raw payload (`u32` BE)
//!
//! Encodes bits into a dense `S x S` binary module lattice (`scale` pixels/module, default 4)
//! and ingests via sub-millisecond center-module DMA sampling with fail-closed CRC32 verification.

use crate::vision::RgbImage;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Write an `RgbImage` as a binary PPM (`P6`) file.
pub fn save_ppm(img: &RgbImage, path: &Path) -> Result<(), String> {
    let header = format!("P6\n{} {}\n255\n", img.width, img.height);
    let mut bytes = Vec::with_capacity(header.len() + img.data.len());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&img.data);
    fs::write(path, bytes).map_err(|e| format!("write PPM {}: {e}", path.display()))
}

pub const AIQC_MAGIC: &[u8; 4] = b"AIQC";
pub const AIQC_VERSION: u8 = 0x01;
pub const AIQC_TYPE_TOKEN_IDS: u8 = 0x02;
pub const AIQC_TYPE_GLYPH_PROPOSITIONS: u8 = 0x04;
pub const AIQC_HEADER_BYTES: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AiqcHeader {
    pub version: u8,
    pub payload_type: u8,
    pub uncompressed_len: u32,
    pub stored_len: u32,
    pub crc32: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct OpticalDmaStats {
    pub width: usize,
    pub height: usize,
    pub grid_side: usize,
    pub scale: usize,
    pub payload_bytes: usize,
    pub n_tokens: usize,
    pub crc32: u32,
    pub encode_ms: f64,
    pub dma_ms: f64,
    pub fps: f64,
    pub tok_per_s: f64,
}

pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

pub fn calculate_grid_side(total_bytes: usize) -> usize {
    let total_bits = total_bytes * 8;
    let mut side = (total_bits as f64).sqrt().floor() as usize;
    while side * side < total_bits {
        side += 1;
    }
    side.max(1)
}

impl AiqcHeader {
    pub fn pack(&self) -> [u8; AIQC_HEADER_BYTES] {
        let mut out = [0u8; AIQC_HEADER_BYTES];
        out[0..4].copy_from_slice(AIQC_MAGIC);
        out[4] = self.version;
        out[5] = self.payload_type;
        out[6] = 0;
        out[7] = 0;
        out[8..12].copy_from_slice(&self.uncompressed_len.to_be_bytes());
        out[12..16].copy_from_slice(&self.stored_len.to_be_bytes());
        out[16..20].copy_from_slice(&self.crc32.to_be_bytes());
        out
    }

    pub fn unpack(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < AIQC_HEADER_BYTES {
            return Err(format!("AIQC header underflow: {} < {AIQC_HEADER_BYTES}", buf.len()));
        }
        if &buf[0..4] != AIQC_MAGIC {
            return Err(format!("AIQC invalid magic: {:?}", &buf[0..4]));
        }
        let version = buf[4];
        if version != AIQC_VERSION {
            return Err(format!("AIQC unsupported version: 0x{version:02x}"));
        }
        let payload_type = buf[5];
        let uncompressed_len = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let stored_len = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]);
        let crc32 = u32::from_be_bytes([buf[16], buf[17], buf[18], buf[19]]);
        Ok(Self {
            version,
            payload_type,
            uncompressed_len,
            stored_len,
            crc32,
        })
    }
}

/// Encode Gemma 4 token IDs into an `AIQC` 2D optical carrier frame (`RgbImage`).
pub fn encode_tokens_to_optical_frame(
    token_ids: &[u32],
    scale: usize,
) -> Result<(RgbImage, AiqcHeader, f64), String> {
    if scale == 0 {
        return Err("AIQC scale must be >= 1".into());
    }
    let t0 = Instant::now();
    let mut payload = Vec::with_capacity(4 + token_ids.len() * 4);
    payload.extend_from_slice(&(token_ids.len() as u32).to_be_bytes());
    for &tid in token_ids {
        payload.extend_from_slice(&tid.to_be_bytes());
    }
    let crc = crc32_ieee(&payload);
    let hdr = AiqcHeader {
        version: AIQC_VERSION,
        payload_type: AIQC_TYPE_TOKEN_IDS,
        uncompressed_len: payload.len() as u32,
        stored_len: payload.len() as u32,
        crc32: crc,
    };

    let mut frame_bytes = Vec::with_capacity(AIQC_HEADER_BYTES + payload.len());
    frame_bytes.extend_from_slice(&hdr.pack());
    frame_bytes.extend_from_slice(&payload);

    let side = calculate_grid_side(frame_bytes.len());
    let width = side * scale;
    let height = side * scale;
    let mut data = vec![0u8; width * height * 3];

    for (byte_idx, &b) in frame_bytes.iter().enumerate() {
        for bit in 0..8 {
            let bit_val = (b >> (7 - bit)) & 1;
            if bit_val != 0 {
                let linear = byte_idx * 8 + bit;
                let gy = linear / side;
                let gx = linear % side;
                let py0 = gy * scale;
                let px0 = gx * scale;
                for dy in 0..scale {
                    let row_off = (py0 + dy) * width;
                    for dx in 0..scale {
                        let off = (row_off + px0 + dx) * 3;
                        data[off] = 255;
                        data[off + 1] = 255;
                        data[off + 2] = 255;
                    }
                }
            }
        }
    }

    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok((RgbImage { width, height, data }, hdr, encode_ms))
}

/// Sub-millisecond center-module DMA decode of an `AIQC` 2D optical frame back into token IDs.
pub fn decode_tokens_from_optical_frame(
    img: &RgbImage,
    scale: usize,
) -> Result<(Vec<u32>, AiqcHeader, f64), String> {
    if scale == 0 || img.width == 0 || img.height == 0 || img.width != img.height || img.width % scale != 0 {
        return Err(format!(
            "AIQC frame dimensions {}x{} incompatible with scale={scale}",
            img.width, img.height
        ));
    }
    let t0 = Instant::now();
    let side = img.width / scale;
    let total_bits = side * side;
    let total_bytes = total_bits / 8;
    if total_bytes < AIQC_HEADER_BYTES {
        return Err(format!("AIQC frame too small: {total_bytes} bytes < {AIQC_HEADER_BYTES}"));
    }

    let center = scale / 2;
    let mut raw = vec![0u8; total_bytes];
    for byte_idx in 0..total_bytes {
        let mut acc = 0u8;
        let base_bit = byte_idx * 8;
        for bit in 0..8 {
            let linear = base_bit + bit;
            let gy = linear / side;
            let gx = linear % side;
            let py = gy * scale + center;
            let px = gx * scale + center;
            let r = img.data[(py * img.width + px) * 3];
            acc = (acc << 1) | u8::from(r >= 128);
        }
        raw[byte_idx] = acc;
    }

    let hdr = AiqcHeader::unpack(&raw[..AIQC_HEADER_BYTES])?;
    if hdr.payload_type != AIQC_TYPE_TOKEN_IDS {
        return Err(format!("AIQC unexpected payload_type 0x{:02x}", hdr.payload_type));
    }
    if hdr.stored_len != hdr.uncompressed_len {
        return Err(format!(
            "AIQC compressed payload (stored={} != uncomp={}) unsupported in zero-dep Rust DMA",
            hdr.stored_len, hdr.uncompressed_len
        ));
    }
    let end = AIQC_HEADER_BYTES
        .checked_add(hdr.stored_len as usize)
        .ok_or("AIQC length overflow")?;
    if end > raw.len() {
        return Err(format!("AIQC payload truncated: need {end} B, frame holds {} B", raw.len()));
    }
    let payload = &raw[AIQC_HEADER_BYTES..end];
    let got_crc = crc32_ieee(payload);
    if got_crc != hdr.crc32 {
        return Err(format!(
            "AIQC CRC32 integrity mismatch: got 0x{got_crc:08x} != header 0x{:08x}",
            hdr.crc32
        ));
    }
    if payload.len() < 4 {
        return Err("AIQC token payload missing count prefix".into());
    }
    let n_toks = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if payload.len() != 4 + n_toks * 4 {
        return Err(format!(
            "AIQC token payload size {} != 4 + {}*4",
            payload.len(),
            n_toks
        ));
    }
    let mut ids = Vec::with_capacity(n_toks);
    for i in 0..n_toks {
        let off = 4 + i * 4;
        ids.push(u32::from_be_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]));
    }
    let dma_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok((ids, hdr, dma_ms))
}

/// Encode tokens into an `AIQC` optical frame and immediately DMA-decode back, verifying
/// 100% bit-exact token parity and measuring live DMA throughput.
pub fn optical_context_dma_roundtrip(
    token_ids: &[u32],
    scale: usize,
) -> Result<(RgbImage, Vec<u32>, OpticalDmaStats), String> {
    let (img, hdr, encode_ms) = encode_tokens_to_optical_frame(token_ids, scale)?;
    let (decoded, dec_hdr, dma_ms) = decode_tokens_from_optical_frame(&img, scale)?;
    if decoded != token_ids || dec_hdr.crc32 != hdr.crc32 {
        return Err("AIQC optical DMA roundtrip parity violation".into());
    }
    let dma_s = (dma_ms / 1000.0).max(1e-9);
    let stats = OpticalDmaStats {
        width: img.width,
        height: img.height,
        grid_side: img.width / scale,
        scale,
        payload_bytes: hdr.stored_len as usize,
        n_tokens: decoded.len(),
        crc32: hdr.crc32,
        encode_ms,
        dma_ms,
        fps: 1.0 / dma_s,
        tok_per_s: (decoded.len() as f64) / dma_s,
    };
    Ok((img, decoded, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aiqc_optical_token_roundtrip_is_bit_exact() {
        let ids: Vec<u32> = vec![2, 105, 2364, 107, 3689, 563, 236743, 236778, 900, 236743, 236778, 236881, 106];
        let (img, dec, stats) = optical_context_dma_roundtrip(&ids, 4).unwrap();
        assert_eq!(dec, ids);
        assert_eq!(img.width, stats.width);
        assert_eq!(stats.n_tokens, ids.len());
        assert!(stats.fps > 0.0 && stats.tok_per_s > 0.0);
    }

    #[test]
    fn aiqc_optical_dma_rejects_bit_flip_fail_closed() {
        let ids: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563, 50429, 106];
        let (mut img, _, _) = encode_tokens_to_optical_frame(&ids, 4).unwrap();
        // Flip a payload module center pixel past the 20-byte (160-bit) header
        let side = img.width / 4;
        let bit_idx = 168usize;
        let gy = bit_idx / side;
        let gx = bit_idx % side;
        let py = gy * 4 + 2;
        let px = gx * 4 + 2;
        let off = (py * img.width + px) * 3;
        img.data[off] = 255 - img.data[off];
        let err = decode_tokens_from_optical_frame(&img, 4).unwrap_err();
        assert!(err.contains("CRC32"), "expected CRC32 failure, got {err}");
    }

    #[test]
    fn aiqc_ppm_file_roundtrip_preserves_tokens() {
        let ids: Vec<u32> = (0..128).map(|i| 1000 + i * 17).collect();
        let (img, _, _) = encode_tokens_to_optical_frame(&ids, 4).unwrap();
        let tmp = std::env::temp_dir().join("test_aiqc_optical_carrier.ppm");
        save_ppm(&img, &tmp).unwrap();
        let loaded = crate::vision::load_rgb_image(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let (dec, _, _) = decode_tokens_from_optical_frame(&loaded, 4).unwrap();
        assert_eq!(dec, ids);
    }
}

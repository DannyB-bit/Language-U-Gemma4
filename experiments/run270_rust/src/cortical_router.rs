//! V6.1 Predictive Cortical Column Router + V7.0 Mixture-of-Depths (`CRTR` v1)
//!
//! Compact INT8 2-layer GELU router per transformer layer that predicts which
//! `ff / block_size` (84 x 256) GeGLU cortical columns will fire *before* reading
//! `gate_proj` from Pi 4 LPDDR4 DRAM, enabling sparse masking across all three
//! MLP projections (`gate_proj`, `up_proj`, `down_proj`). Optionally packs an
//! INT8 linear Mixture-of-Depths (`MoD`) residual-highway gate per layer.

use crate::math::gelu_tanh_fast_f32;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const CRTR_MAGIC: &[u8; 4] = b"CRTR";
pub const CRTR_VERSION: u32 = 1;
pub const CRTR_HEADER_BYTES: usize = 36;
pub const CRTR_FLAG_MOD_GATE: u32 = 1 << 0;

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x.clamp(-60.0f32, 60.0f32)).exp())
}

#[inline]
fn dot_i8_f32_4way(w: &[i8], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let n = w.len();
    let chunks = n / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = 0usize;
    for _ in 0..chunks {
        s0 += (w[i] as f32) * x[i];
        s1 += (w[i + 1] as f32) * x[i + 1];
        s2 += (w[i + 2] as f32) * x[i + 2];
        s3 += (w[i + 3] as f32) * x[i + 3];
        i += 4;
    }
    let mut s = (s0 + s1) + (s2 + s3);
    while i < n {
        s += (w[i] as f32) * x[i];
        i += 1;
    }
    s
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("CRTR read u32: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32_vec<R: Read>(r: &mut R, len: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut raw = vec![0u8; len * 4];
    r.read_exact(&mut raw).map_err(|e| format!("CRTR read {label}: {e}"))?;
    let mut out = Vec::with_capacity(len);
    for ch in raw.chunks_exact(4) {
        let v = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
        if !v.is_finite() {
            return Err(format!("CRTR non-finite float in {label}: {v}"));
        }
        out.push(v);
    }
    Ok(out)
}

fn read_i8_vec<R: Read>(r: &mut R, len: usize, label: &str) -> Result<Vec<i8>, String> {
    let mut raw = vec![0u8; len];
    r.read_exact(&mut raw).map_err(|e| format!("CRTR read {label}: {e}"))?;
    Ok(raw.into_iter().map(|b| b as i8).collect())
}

fn write_f32_slice<W: Write>(w: &mut W, vals: &[f32]) -> Result<(), String> {
    for &v in vals {
        if !v.is_finite() {
            return Err(format!("CRTR refusing to serialize non-finite float {v}"));
        }
        w.write_all(&v.to_le_bytes())
            .map_err(|e| format!("CRTR write f32: {e}"))?;
    }
    Ok(())
}

fn write_i8_slice<W: Write>(w: &mut W, vals: &[i8]) -> Result<(), String> {
    let bytes: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
    w.write_all(&bytes).map_err(|e| format!("CRTR write i8: {e}"))
}

#[derive(Clone, Debug)]
pub struct CorticalRouterLayer {
    pub d_model: usize,
    pub ff: usize,
    pub block_size: usize,
    pub n_blocks: usize,
    pub rank: usize,
    pub v_i8: Vec<i8>,
    pub sv: Vec<f32>,
    pub b1: Vec<f32>,
    pub u_i8: Vec<i8>,
    pub su: Vec<f32>,
    pub b2: Vec<f32>,
    pub mod_w_i8: Option<Vec<i8>>,
    pub mod_scale: f32,
    pub mod_bias: f32,
}

impl CorticalRouterLayer {
    pub fn byte_len(&self, has_mod_gate: bool) -> usize {
        let base = self.rank * self.d_model
            + self.rank * 4
            + self.rank * 4
            + self.n_blocks * self.rank
            + self.n_blocks * 4
            + self.n_blocks * 4;
        if has_mod_gate {
            base + self.d_model + 8
        } else {
            base
        }
    }

    pub fn validate(&self, expect_mod_gate: bool) -> Result<(), String> {
        if self.d_model == 0 || self.ff == 0 || self.block_size == 0 || self.rank == 0 {
            return Err("CRTR layer has zero dimension".into());
        }
        if self.ff % self.block_size != 0 || self.block_size % 16 != 0 {
            return Err(format!(
                "CRTR invalid tiling: ff={} block_size={}",
                self.ff, self.block_size
            ));
        }
        let n_blocks = self.ff / self.block_size;
        if self.n_blocks != n_blocks {
            return Err(format!("CRTR n_blocks {} != {n_blocks}", self.n_blocks));
        }
        if self.v_i8.len() != self.rank * self.d_model {
            return Err(format!(
                "CRTR v_i8 len {} != {}",
                self.v_i8.len(),
                self.rank * self.d_model
            ));
        }
        if self.sv.len() != self.rank || self.b1.len() != self.rank {
            return Err("CRTR sv/b1 length mismatch".into());
        }
        if self.u_i8.len() != n_blocks * self.rank {
            return Err("CRTR u_i8 length mismatch".into());
        }
        if self.su.len() != n_blocks || self.b2.len() != n_blocks {
            return Err("CRTR su/b2 length mismatch".into());
        }
        if self
            .sv
            .iter()
            .chain(self.b1.iter())
            .chain(self.su.iter())
            .chain(self.b2.iter())
            .any(|v| !v.is_finite())
        {
            return Err("CRTR layer contains non-finite scale or bias".into());
        }
        if expect_mod_gate {
            let mw = self
                .mod_w_i8
                .as_ref()
                .ok_or_else(|| "CRTR missing mod_w_i8 despite CRTR_FLAG_MOD_GATE".to_string())?;
            if mw.len() != self.d_model {
                return Err(format!("CRTR mod_w_i8 len {} != {}", mw.len(), self.d_model));
            }
            if !self.mod_scale.is_finite() || !self.mod_bias.is_finite() {
                return Err("CRTR non-finite mod_scale/mod_bias".into());
            }
        } else if self.mod_w_i8.is_some() {
            return Err("CRTR has mod_w_i8 without CRTR_FLAG_MOD_GATE".into());
        }
        Ok(())
    }

    /// Predict per-block firing probabilities `[n_blocks]`, max-aggregated across `0..seq_len`.
    pub fn predict_block_scores(&self, x_batch: &[f32], seq_len: usize) -> Result<Vec<f32>, String> {
        if seq_len == 0 {
            return Err("predict_block_scores: seq_len == 0".into());
        }
        let expected = seq_len * self.d_model;
        if x_batch.len() != expected {
            return Err(format!(
                "predict_block_scores: x_batch len {} != expected {expected}",
                x_batch.len()
            ));
        }
        let mut max_probs = vec![0.0f32; self.n_blocks];
        let mut h = vec![0.0f32; self.rank];
        for t in 0..seq_len {
            let xt = &x_batch[t * self.d_model..(t + 1) * self.d_model];
            for k in 0..self.rank {
                let vrow = &self.v_i8[k * self.d_model..(k + 1) * self.d_model];
                let pre = self.sv[k] * dot_i8_f32_4way(vrow, xt) + self.b1[k];
                h[k] = gelu_tanh_fast_f32(pre);
            }
            for b in 0..self.n_blocks {
                let urow = &self.u_i8[b * self.rank..(b + 1) * self.rank];
                let logit = self.su[b] * dot_i8_f32_4way(urow, &h) + self.b2[b];
                let p = sigmoid_f32(logit);
                if p > max_probs[b] {
                    max_probs[b] = p;
                }
            }
        }
        Ok(max_probs)
    }

    /// Predict `(row_tile_mask, col_block_mask, block_size)` before running `gate_proj`.
    pub fn predict_masks(
        &self,
        x_batch: &[f32],
        seq_len: usize,
        top_p: f32,
    ) -> Result<(Vec<u8>, Vec<u8>, usize), String> {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(format!("predict_masks: top_p must be finite in (0, 1], got {top_p}"));
        }
        let scores = self.predict_block_scores(x_batch, seq_len)?;
        let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let keep = ((self.n_blocks as f32 * top_p).ceil() as usize).clamp(1, self.n_blocks);
        let tiles_per_blk = self.block_size / 16;
        let mut col_mask = vec![0u8; self.n_blocks];
        let mut row_mask = vec![0u8; self.ff.div_ceil(16)];
        for &(b, _) in &ranked[..keep] {
            col_mask[b] = 1;
            let t0 = b * tiles_per_blk;
            for dt in 0..tiles_per_blk {
                if t0 + dt < row_mask.len() {
                    row_mask[t0 + dt] = 1;
                }
            }
        }
        Ok((row_mask, col_mask, self.block_size))
    }

    /// Predict Mixture-of-Depths layer execution probability, max-aggregated across `0..seq_len`.
    pub fn predict_mod_exec_prob(&self, x_batch: &[f32], seq_len: usize) -> Result<Option<f32>, String> {
        if seq_len == 0 {
            return Err("predict_mod_exec_prob: seq_len == 0".into());
        }
        let expected = seq_len * self.d_model;
        if x_batch.len() != expected {
            return Err(format!(
                "predict_mod_exec_prob: x_batch len {} != expected {expected}",
                x_batch.len()
            ));
        }
        let Some(ref mw) = self.mod_w_i8 else {
            return Ok(None);
        };
        let mut max_prob = 0.0f32;
        for t in 0..seq_len {
            let xt = &x_batch[t * self.d_model..(t + 1) * self.d_model];
            let logit = self.mod_scale * dot_i8_f32_4way(mw, xt) + self.mod_bias;
            let p = sigmoid_f32(logit);
            if p > max_prob {
                max_prob = p;
            }
        }
        Ok(Some(max_prob))
    }
}

#[derive(Clone, Debug)]
pub struct CorticalRouterBank {
    pub n_layers: usize,
    pub d_model: usize,
    pub ff: usize,
    pub block_size: usize,
    pub n_blocks: usize,
    pub rank: usize,
    pub flags: u32,
    pub layers: Vec<CorticalRouterLayer>,
}

impl CorticalRouterBank {
    pub fn has_mod_gate(&self) -> bool {
        (self.flags & CRTR_FLAG_MOD_GATE) != 0
    }

    pub fn byte_len(&self) -> usize {
        let has_mod = self.has_mod_gate();
        CRTR_HEADER_BYTES + self.layers.iter().map(|l| l.byte_len(has_mod)).sum::<usize>()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_layers == 0 || self.d_model == 0 || self.ff == 0 || self.block_size == 0 || self.rank == 0 {
            return Err("CRTR header contains zero dimension".into());
        }
        if self.ff % self.block_size != 0 || self.block_size % 16 != 0 {
            return Err(format!(
                "CRTR invalid header tiling: ff={} block_size={}",
                self.ff, self.block_size
            ));
        }
        let n_blocks = self.ff / self.block_size;
        if self.n_blocks != n_blocks {
            return Err(format!("CRTR bank n_blocks {} != {n_blocks}", self.n_blocks));
        }
        if (self.flags & !CRTR_FLAG_MOD_GATE) != 0 {
            return Err(format!("CRTR unknown header flags 0x{:x}", self.flags));
        }
        if self.layers.len() != self.n_layers {
            return Err(format!(
                "CRTR layer count {} != n_layers {}",
                self.layers.len(),
                self.n_layers
            ));
        }
        let has_mod = self.has_mod_gate();
        for (idx, layer) in self.layers.iter().enumerate() {
            if layer.d_model != self.d_model
                || layer.ff != self.ff
                || layer.block_size != self.block_size
                || layer.rank != self.rank
            {
                return Err(format!("CRTR layer {idx} dimension mismatch vs header"));
            }
            layer
                .validate(has_mod)
                .map_err(|e| format!("CRTR layer {idx}: {e}"))?;
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<usize, String> {
        self.validate()?;
        let f = File::create(path).map_err(|e| format!("CRTR create {}: {e}", path.display()))?;
        let mut w = BufWriter::new(f);
        w.write_all(CRTR_MAGIC)
            .map_err(|e| format!("CRTR write magic: {e}"))?;
        for &val in &[
            CRTR_VERSION,
            self.n_layers as u32,
            self.d_model as u32,
            self.ff as u32,
            self.block_size as u32,
            self.rank as u32,
            self.flags,
            0u32,
        ] {
            w.write_all(&val.to_le_bytes())
                .map_err(|e| format!("CRTR write header: {e}"))?;
        }
        let has_mod = self.has_mod_gate();
        for layer in &self.layers {
            write_i8_slice(&mut w, &layer.v_i8)?;
            write_f32_slice(&mut w, &layer.sv)?;
            write_f32_slice(&mut w, &layer.b1)?;
            write_i8_slice(&mut w, &layer.u_i8)?;
            write_f32_slice(&mut w, &layer.su)?;
            write_f32_slice(&mut w, &layer.b2)?;
            if has_mod {
                let mw = layer.mod_w_i8.as_ref().unwrap();
                write_i8_slice(&mut w, mw)?;
                write_f32_slice(&mut w, &[layer.mod_scale, layer.mod_bias])?;
            }
        }
        w.flush().map_err(|e| format!("CRTR flush {}: {e}", path.display()))?;
        Ok(self.byte_len())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("CRTR open {}: {e}", path.display()))?;
        let file_len = f
            .metadata()
            .map_err(|e| format!("CRTR metadata {}: {e}", path.display()))?
            .len() as usize;
        if file_len < CRTR_HEADER_BYTES {
            return Err(format!(
                "CRTR file {} truncated: {file_len} < {CRTR_HEADER_BYTES}",
                path.display()
            ));
        }
        let mut r = BufReader::new(f);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)
            .map_err(|e| format!("CRTR read magic {}: {e}", path.display()))?;
        if &magic != CRTR_MAGIC {
            return Err(format!("CRTR invalid magic {:?} in {}", magic, path.display()));
        }
        let version = read_u32_le(&mut r)?;
        if version != CRTR_VERSION {
            return Err(format!("CRTR unsupported version {version} != {CRTR_VERSION}"));
        }
        let n_layers = read_u32_le(&mut r)? as usize;
        let d_model = read_u32_le(&mut r)? as usize;
        let ff = read_u32_le(&mut r)? as usize;
        let block_size = read_u32_le(&mut r)? as usize;
        let rank = read_u32_le(&mut r)? as usize;
        let flags = read_u32_le(&mut r)?;
        let reserved = read_u32_le(&mut r)?;
        if reserved != 0 {
            return Err(format!("CRTR unexpected non-zero reserved field {reserved}"));
        }
        if n_layers == 0 || d_model == 0 || ff == 0 || block_size == 0 || rank == 0 {
            return Err("CRTR invalid zero dimension in header".into());
        }
        if ff % block_size != 0 || block_size % 16 != 0 {
            return Err(format!("CRTR invalid header tiling: ff={ff} block_size={block_size}"));
        }
        if (flags & !CRTR_FLAG_MOD_GATE) != 0 {
            return Err(format!("CRTR unknown header flags 0x{flags:x}"));
        }
        let n_blocks = ff / block_size;
        let has_mod = (flags & CRTR_FLAG_MOD_GATE) != 0;
        let per_layer_bytes = rank * d_model
            + rank * 4
            + rank * 4
            + n_blocks * rank
            + n_blocks * 4
            + n_blocks * 4
            + if has_mod { d_model + 8 } else { 0 };
        let expected_len = CRTR_HEADER_BYTES + n_layers * per_layer_bytes;
        if file_len != expected_len {
            return Err(format!(
                "CRTR byte length mismatch in {}: actual={file_len} != expected={expected_len}",
                path.display()
            ));
        }

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let v_i8 = read_i8_vec(&mut r, rank * d_model, &format!("L{l}.v_i8"))?;
            let sv = read_f32_vec(&mut r, rank, &format!("L{l}.sv"))?;
            let b1 = read_f32_vec(&mut r, rank, &format!("L{l}.b1"))?;
            let u_i8 = read_i8_vec(&mut r, n_blocks * rank, &format!("L{l}.u_i8"))?;
            let su = read_f32_vec(&mut r, n_blocks, &format!("L{l}.su"))?;
            let b2 = read_f32_vec(&mut r, n_blocks, &format!("L{l}.b2"))?;
            let (mod_w_i8, mod_scale, mod_bias) = if has_mod {
                let mw = read_i8_vec(&mut r, d_model, &format!("L{l}.mod_w_i8"))?;
                let msb = read_f32_vec(&mut r, 2, &format!("L{l}.mod_sb"))?;
                (Some(mw), msb[0], msb[1])
            } else {
                (None, 0.0, 0.0)
            };
            layers.push(CorticalRouterLayer {
                d_model,
                ff,
                block_size,
                n_blocks,
                rank,
                v_i8,
                sv,
                b1,
                u_i8,
                su,
                b2,
                mod_w_i8,
                mod_scale,
                mod_bias,
            });
        }
        let bank = Self {
            n_layers,
            d_model,
            ff,
            block_size,
            n_blocks,
            rank,
            flags,
            layers,
        };
        bank.validate()?;
        Ok(bank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::gelu_tanh_scalar;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_crtr_path(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("crtr_{tag}_{}_{nanos}.crtr", std::process::id()))
    }

    fn build_planted_bank(with_mod: bool) -> CorticalRouterBank {
        // d_model=64, ff=256, block_size=32 -> n_blocks=8, rank=8
        // Block b fires strongly when group b (dims b*8..(b+1)*8) is positive.
        let d_model = 64usize;
        let ff = 256usize;
        let block_size = 32usize;
        let n_blocks = ff / block_size;
        let rank = 8usize;
        let mut layers = Vec::new();
        for l in 0..2 {
            let mut v_i8 = vec![0i8; rank * d_model];
            let mut u_i8 = vec![0i8; n_blocks * rank];
            for k in 0..rank {
                for j in 0..8 {
                    let sign = if l == 0 || (j + k) % 2 == 0 { 100i8 } else { 80i8 };
                    v_i8[k * d_model + k * 8 + j] = sign;
                }
                u_i8[k * rank + k] = 100i8;
            }
            let sv = vec![0.02f32; rank];
            let b1 = vec![-0.2f32; rank];
            let su = vec![0.15f32; n_blocks];
            let b2 = vec![-2.5f32; n_blocks];
            let (mod_w_i8, mod_scale, mod_bias) = if with_mod {
                let mut mw = vec![0i8; d_model];
                for j in 0..16 {
                    mw[j] = 100i8;
                }
                (Some(mw), 0.01f32, -1.0f32)
            } else {
                (None, 0.0, 0.0)
            };
            layers.push(CorticalRouterLayer {
                d_model,
                ff,
                block_size,
                n_blocks,
                rank,
                v_i8,
                sv,
                b1,
                u_i8,
                su,
                b2,
                mod_w_i8,
                mod_scale,
                mod_bias,
            });
        }
        CorticalRouterBank {
            n_layers: 2,
            d_model,
            ff,
            block_size,
            n_blocks,
            rank,
            flags: if with_mod { CRTR_FLAG_MOD_GATE } else { 0 },
            layers,
        }
    }

    #[test]
    fn gate1_crtr_roundtrip_header_and_fail_closed_corruption() {
        let bank = build_planted_bank(true);
        let p = temp_crtr_path("gate1");
        let n_written = bank.save(&p).unwrap();
        assert_eq!(n_written, bank.byte_len());
        let meta_len = std::fs::metadata(&p).unwrap().len() as usize;
        assert_eq!(meta_len, n_written);

        let loaded = CorticalRouterBank::load(&p).unwrap();
        assert_eq!(loaded.byte_len(), n_written);
        assert!(loaded.has_mod_gate());
        assert_eq!(loaded.layers[0].v_i8, bank.layers[0].v_i8);
        assert_eq!(loaded.layers[1].mod_w_i8, bank.layers[1].mod_w_i8);

        // Corrupt magic
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0] = b'X';
        let p_bad_magic = temp_crtr_path("gate1_bad_magic");
        std::fs::write(&p_bad_magic, &bytes).unwrap();
        assert!(CorticalRouterBank::load(&p_bad_magic).is_err());

        // Truncated file
        let p_trunc = temp_crtr_path("gate1_trunc");
        std::fs::write(&p_trunc, &std::fs::read(&p).unwrap()[..n_written - 3]).unwrap();
        assert!(CorticalRouterBank::load(&p_trunc).is_err());

        // Unsupported version
        let mut bytes_ver = std::fs::read(&p).unwrap();
        bytes_ver[4..8].copy_from_slice(&99u32.to_le_bytes());
        let p_ver = temp_crtr_path("gate1_bad_ver");
        std::fs::write(&p_ver, &bytes_ver).unwrap();
        assert!(CorticalRouterBank::load(&p_ver).is_err());

        for f in [&p, &p_bad_magic, &p_trunc, &p_ver] {
            let _ = std::fs::remove_file(f);
        }
    }

    #[test]
    fn gate2_crtr_rejects_nonfinite_and_checks_variance() {
        let mut bank = build_planted_bank(false);
        let sum: f64 = bank.layers[0].v_i8.iter().map(|&v| v as f64).sum();
        let mean = sum / bank.layers[0].v_i8.len() as f64;
        let var: f64 = bank.layers[0]
            .v_i8
            .iter()
            .map(|&v| {
                let d = (v as f64) - mean;
                d * d
            })
            .sum::<f64>()
            / bank.layers[0].v_i8.len() as f64;
        assert!(var > 100.0, "expected non-trivial weight variance, got {var}");

        bank.layers[0].sv[0] = f32::NAN;
        assert!(bank.validate().is_err());
        let p = temp_crtr_path("gate2_nan");
        assert!(bank.save(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn gate3_crtr_blocks_and_tokens_are_non_rank1_distinct() {
        let bank = build_planted_bank(false);
        let layer = &bank.layers[0];
        let mut x0 = vec![0.0f32; layer.d_model];
        x0[0..8].fill(2.0);
        let mut x1 = vec![0.0f32; layer.d_model];
        x1[40..48].fill(2.0);
        let s0 = layer.predict_block_scores(&x0, 1).unwrap();
        let s1 = layer.predict_block_scores(&x1, 1).unwrap();
        let diff: f32 = s0.iter().zip(s1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0, "distinct queries produced near-identical block scores: diff={diff}");
    }

    #[test]
    fn gate4_semantic_topk_recall_and_batch_union_and_tile_masks() {
        let bank = build_planted_bank(false);
        let layer = &bank.layers[0];

        // Query triggering blocks 2 and 6 strongly
        let mut x_single = vec![0.0f32; layer.d_model];
        x_single[16..24].fill(3.0);
        x_single[48..56].fill(3.0);
        let (rmask, cmask, bsize) = layer.predict_masks(&x_single, 1, 0.25).unwrap();
        assert_eq!(bsize, 32);
        assert_eq!(cmask.len(), 8);
        assert_eq!(cmask[2], 1);
        assert_eq!(cmask[6], 1);
        assert_eq!(cmask.iter().filter(|&&v| v == 1).count(), 2);
        // Verify exact tile-mapping (32 / 16 = 2 tiles per block)
        assert_eq!(rmask.len(), 256 / 16);
        assert_eq!(rmask[4], 1);
        assert_eq!(rmask[5], 1);
        assert_eq!(rmask[12], 1);
        assert_eq!(rmask[13], 1);
        assert_eq!(rmask.iter().filter(|&&v| v == 1).count(), 4);

        // Speculative batch (seq_len=4) where token 0 fires block 1 and token 3 fires block 7
        let mut x_batch = vec![0.0f32; 4 * layer.d_model];
        x_batch[8..16].fill(3.0);
        x_batch[3 * layer.d_model + 56..3 * layer.d_model + 64].fill(3.0);
        let (_rm_b, cm_b, _) = layer.predict_masks(&x_batch, 4, 0.25).unwrap();
        assert_eq!(cm_b[1], 1);
        assert_eq!(cm_b[7], 1);
    }

    #[test]
    fn gate5_mod_gate_separates_salience_and_returns_none_when_disabled() {
        let bank_no_mod = build_planted_bank(false);
        let x = vec![0.5f32; 64];
        assert_eq!(
            bank_no_mod.layers[0].predict_mod_exec_prob(&x, 1).unwrap(),
            None
        );

        let bank_mod = build_planted_bank(true);
        let mut x_high = vec![0.0f32; 64];
        x_high[0..16].fill(1.0); // dot = 1600 -> scale 0.01 -> 16 - 1 = +15 -> p ~ 1.0
        let x_low = vec![0.0f32; 64]; // dot = 0 -> logit = -1.0 -> p ~ 0.2689
        let p_hi = bank_mod.layers[0].predict_mod_exec_prob(&x_high, 1).unwrap().unwrap();
        let p_lo = bank_mod.layers[0].predict_mod_exec_prob(&x_low, 1).unwrap().unwrap();
        assert!(p_hi > 0.99, "expected p_hi > 0.99, got {p_hi}");
        assert!(p_lo < 0.30, "expected p_lo < 0.30, got {p_lo}");

        // Batch max-aggregation keeps MoD active if any token in batch needs deep compute
        let mut x_batch = vec![0.0f32; 2 * 64];
        x_batch[64..80].fill(1.0);
        let p_batch = bank_mod.layers[0].predict_mod_exec_prob(&x_batch, 2).unwrap().unwrap();
        assert!(p_batch > 0.99);
    }

    #[test]
    fn gate6_bit_exact_parity_vs_naive_fp32_scalar_reference() {
        let bank = build_planted_bank(true);
        let layer = &bank.layers[1];
        let mut x = vec![0.0f32; 3 * layer.d_model];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (((i * 37) % 23) as f32 - 11.0) * 0.08;
        }
        let fast = layer.predict_block_scores(&x, 3).unwrap();

        let mut ref_max = vec![0.0f32; layer.n_blocks];
        for t in 0..3 {
            let xt = &x[t * layer.d_model..(t + 1) * layer.d_model];
            let mut h = vec![0.0f32; layer.rank];
            for k in 0..layer.rank {
                let mut acc = 0.0f32;
                for j in 0..layer.d_model {
                    acc += (layer.v_i8[k * layer.d_model + j] as f32) * xt[j];
                }
                h[k] = gelu_tanh_scalar(layer.sv[k] * acc + layer.b1[k]);
            }
            for b in 0..layer.n_blocks {
                let mut acc = 0.0f32;
                for k in 0..layer.rank {
                    acc += (layer.u_i8[b * layer.rank + k] as f32) * h[k];
                }
                let logit = layer.su[b] * acc + layer.b2[b];
                let p = 1.0f32 / (1.0f32 + (-logit).exp());
                if p > ref_max[b] {
                    ref_max[b] = p;
                }
            }
        }
        let mut max_err = 0.0f32;
        for (&a, &b) in fast.iter().zip(ref_max.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-5, "Gate 6 FP32 parity max_err={max_err}");
    }
}

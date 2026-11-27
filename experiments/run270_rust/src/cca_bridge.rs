//! Cognitive-Core-Atlas (`cca.cluster_plan.v1` & `cca.depth.v1`) runtime bridge.
//!
//! Connects evidence-backed, null-controlled reasoning circuit discovery from
//! `DannyB-bit/Cognitive-Core-Atlas` directly to `Language-U` V11 per-layer
//! `CorticalRouter` / `HeadRouter` / `MoD` / Tier-2 residual bypass policies and
//! `LUZKCL11` SHA-256 cryptographic shard attestations.

use crate::cuneiform_tbi::{tbi_extract_tag, tbi_tag_ptr, tbi_untag_ptr};
use crate::json::{parse as parse_json, Json};
use std::path::Path;

/// 8-bit Intra-Chip Cognitive-Core descriptor packed into unused ARM64 TBI pointer bits `[63:56]`:
///   - bits `[7:6]` (`tier`): `0 = ColdVaParked (0 RSS)`, `1 = PeripheralSparse`, `2 = WarmCore`, `3 = LitCognitiveCore`
///   - bits `[5:3]` (`abits_code`): activation bit-plane budget (`0..=7` -> `1..=8` bits, e.g. `3 -> 4b`, `5 -> 6b`, `7 -> 8b`)
///   - bits `[2:0]` (`salience_q`): quantized empirical reasoning salience (`0..=7`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TbiCognitiveTag {
    pub tier: u8,
    pub abits: u8,
    pub salience_q: u8,
}

impl TbiCognitiveTag {
    #[inline]
    pub fn pack_byte(self) -> u8 {
        let t = (self.tier & 0x3) << 6;
        let a = (self.abits.clamp(1, 8) - 1) << 3;
        let s = self.salience_q.min(7);
        t | a | s
    }

    #[inline]
    pub fn unpack_byte(b: u8) -> Self {
        Self {
            tier: (b >> 6) & 0x3,
            abits: ((b >> 3) & 0x7) + 1,
            salience_q: b & 0x7,
        }
    }

    /// Tag a 64-bit virtual address in `[55:0]` with this 8-bit Cognitive-Core descriptor in `[63:56]` (zero-cycle MMU ignore on AArch64 TBI).
    #[inline]
    pub fn tag_va(self, raw_va: u64) -> u64 {
        tbi_tag_ptr(raw_va, self.pack_byte())
    }

    #[inline]
    pub fn from_tagged_va(tagged_va: u64) -> (Self, u64) {
        (Self::unpack_byte(tbi_extract_tag(tagged_va)), tbi_untag_ptr(tagged_va))
    }
}

impl CcaLayerPolicy {
    /// Derive the Intra-Chip TBI tag (`PrivateLM` proportional budget inside a single chip's 256 TiB VA space).
    pub fn intra_chip_tbi_tag(&self) -> TbiCognitiveTag {
        if self.selected_cognitive_core {
            TbiCognitiveTag { tier: 3, abits: 8, salience_q: 7 }
        } else if self.tier2_bypass {
            TbiCognitiveTag { tier: 0, abits: 2, salience_q: 0 }
        } else if !self.measured {
            TbiCognitiveTag { tier: 2, abits: 6, salience_q: 5 }
        } else {
            let sq = ((self.mlp_top_p * 7.0).round() as u8).clamp(1, 6);
            let ab = if self.mlp_top_p >= 0.55 { 6 } else { 4 };
            TbiCognitiveTag { tier: 1, abits: ab, salience_q: sq }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CcaLayerPolicy {
    pub layer: usize,
    pub measured: bool,
    pub selected_cognitive_core: bool,
    pub tier2_bypass: bool,
    pub mlp_top_p: f32,
    pub head_top_p: f32,
    pub mod_thresh: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CcaShardSpec {
    pub shard_index: usize,
    pub node_name: String,
    pub endpoint: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub skip_embed_tokens: bool,
    pub luzkcl11_shard_attestation: String,
}

#[derive(Clone, Debug)]
pub struct CcaClusterPlan {
    pub model_id: String,
    pub num_layers: usize,
    pub d_model: usize,
    pub cca_manifest_sha256: String,
    pub bound_cluster_psk: String,
    pub selected_cognitive_core_layers: Vec<usize>,
    pub tier2_bypass_layers: Vec<usize>,
    pub layer_policies: Vec<CcaLayerPolicy>,
    pub shards: Vec<CcaShardSpec>,
}

impl CcaClusterPlan {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("read CCA plan {}: {e}", path.display()))?;
        Self::from_json_bytes(&raw)
    }

    pub fn from_json_bytes(raw: &[u8]) -> Result<Self, String> {
        let root = parse_json(raw).map_err(|e| format!("parse CCA plan JSON: {e}"))?;
        let schema = root
            .get("schema_version")
            .and_then(Json::as_str)
            .ok_or("CCA plan missing schema_version")?;
        if schema != "cca.cluster_plan.v1" {
            return Err(format!("unsupported CCA plan schema_version '{schema}', expected 'cca.cluster_plan.v1'"));
        }
        let model_id = root
            .get("model_id")
            .and_then(Json::as_str)
            .ok_or("CCA plan missing model_id")?
            .to_string();
        let num_layers = root
            .get("num_layers")
            .and_then(Json::as_usize)
            .ok_or("CCA plan missing num_layers")?;
        let d_model = root
            .get("d_model")
            .and_then(Json::as_usize)
            .ok_or("CCA plan missing d_model")?;
        let cca_manifest_sha256 = root
            .get("cca_manifest_sha256")
            .and_then(Json::as_str)
            .ok_or("CCA plan missing cca_manifest_sha256")?
            .to_string();
        if cca_manifest_sha256.len() != 64 || !cca_manifest_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("invalid cca_manifest_sha256 '{cca_manifest_sha256}'"));
        }
        let bound_cluster_psk = root
            .get("bound_cluster_psk")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();

        let pol_arr = root
            .get("layer_policies")
            .and_then(Json::as_arr)
            .ok_or("CCA plan missing layer_policies array")?;
        if pol_arr.len() != num_layers {
            return Err(format!(
                "CCA plan layer_policies len {} != num_layers {num_layers}",
                pol_arr.len()
            ));
        }

        let mut layer_policies = Vec::with_capacity(num_layers);
        let mut selected_cognitive_core_layers = Vec::new();
        let mut tier2_bypass_layers = Vec::new();
        for (expected_l, item) in pol_arr.iter().enumerate() {
            let layer = item
                .get("layer")
                .and_then(Json::as_usize)
                .ok_or_else(|| format!("layer_policies[{expected_l}] missing layer"))?;
            if layer != expected_l {
                return Err(format!("layer_policies[{expected_l}] has out-of-order layer={layer}"));
            }
            let measured = match item.get("measured") {
                Some(Json::Bool(b)) => *b,
                _ => false,
            };
            let selected = match item.get("selected_cognitive_core") {
                Some(Json::Bool(b)) => *b,
                _ => false,
            };
            let mut tier2_bypass = match item.get("tier2_bypass") {
                Some(Json::Bool(b)) => *b,
                _ => false,
            };
            if selected && tier2_bypass {
                return Err(format!("CCA invariant violation: selected cognitive core layer {layer} cannot have tier2_bypass=true"));
            }
            if !measured {
                tier2_bypass = false;
            }
            let mlp_top_p = item
                .get("mlp_top_p")
                .and_then(Json::as_f64)
                .ok_or_else(|| format!("layer {layer} missing mlp_top_p"))? as f32;
            let head_top_p = item
                .get("head_top_p")
                .and_then(Json::as_f64)
                .ok_or_else(|| format!("layer {layer} missing head_top_p"))? as f32;
            let mod_thresh = item
                .get("mod_thresh")
                .and_then(Json::as_f64)
                .ok_or_else(|| format!("layer {layer} missing mod_thresh"))? as f32;
            if !(0.0 < mlp_top_p && mlp_top_p <= 1.0 && 0.0 < head_top_p && head_top_p <= 1.0 && (0.0..=1.0).contains(&mod_thresh)) {
                return Err(format!(
                    "layer {layer} invalid routing bounds mlp={mlp_top_p} head={head_top_p} mod={mod_thresh}"
                ));
            }
            if selected {
                selected_cognitive_core_layers.push(layer);
            }
            if tier2_bypass {
                tier2_bypass_layers.push(layer);
            }
            layer_policies.push(CcaLayerPolicy {
                layer,
                measured,
                selected_cognitive_core: selected,
                tier2_bypass,
                mlp_top_p,
                head_top_p,
                mod_thresh,
            });
        }

        let mut shards = Vec::new();
        if let Some(sh_arr) = root.get("shards").and_then(Json::as_arr) {
            for sh in sh_arr {
                shards.push(CcaShardSpec {
                    shard_index: sh.get("shard_index").and_then(Json::as_usize).unwrap_or(shards.len()),
                    node_name: sh.get("node_name").and_then(Json::as_str).unwrap_or("node").to_string(),
                    endpoint: sh.get("endpoint").and_then(Json::as_str).unwrap_or("").to_string(),
                    layer_start: sh.get("layer_start").and_then(Json::as_usize).ok_or("shard missing layer_start")?,
                    layer_end: sh.get("layer_end").and_then(Json::as_usize).ok_or("shard missing layer_end")?,
                    skip_embed_tokens: matches!(sh.get("skip_embed_tokens"), Some(Json::Bool(true))),
                    luzkcl11_shard_attestation: sh
                        .get("luzkcl11_shard_attestation")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_string(),
                });
            }
        }

        Ok(Self {
            model_id,
            num_layers,
            d_model,
            cca_manifest_sha256,
            bound_cluster_psk,
            selected_cognitive_core_layers,
            tier2_bypass_layers,
            layer_policies,
            shards,
        })
    }

    pub fn bind_psk(&self, raw_psk: &str) -> String {
        format!("{raw_psk}:cca:{}", &self.cca_manifest_sha256[..16])
    }
}

/// Canonical 6 matched reasoning vs. repeat-control pairs from `Cognitive-Core-Atlas/scripts/gemma4_depth_probe.py`.
pub struct CcaProbePair {
    pub id: &'static str,
    pub reasoning: &'static str,
    pub control: &'static str,
    pub answer: &'static str,
}

pub const CCA_PROBE_PAIRS: [CcaProbePair; 6] = [
    CcaProbePair {
        id: "arithmetic_31",
        reasoning: "Let x=7 and y=5. Compute 3*x + 2*y. Answer with only the integer.",
        control: "Repeat only this integer: 31.",
        answer: "31",
    },
    CcaProbePair {
        id: "arithmetic_42",
        reasoning: "Compute (18 / 3 + 8) * 3. Answer with only the integer.",
        control: "Repeat only this integer: 42.",
        answer: "42",
    },
    CcaProbePair {
        id: "logic_blue",
        reasoning: "All bloops are blue. Tavi is a bloop. What color is Tavi? Answer with only the uppercase color.",
        control: "Repeat only this uppercase word: BLUE.",
        answer: "BLUE",
    },
    CcaProbePair {
        id: "code_17",
        reasoning: "A Python list starts as [2,4,6,8]. We append 11 and then remove 2. What is the sum? Answer with only the integer.",
        control: "Repeat only this integer: 27.",
        answer: "27",
    },
    CcaProbePair {
        id: "logic_no",
        reasoning: "No kavits are red. Some red things are round. Does being a kavit imply being red? Answer only YES or NO.",
        control: "Repeat only this uppercase word: NO.",
        answer: "NO",
    },
    CcaProbePair {
        id: "algebra_12",
        reasoning: "If 4a - 7 = 41, what is a? Answer with only the integer.",
        control: "Repeat only this integer: 12.",
        answer: "12",
    },
];

pub fn cca_quantile(values: &[f64], prob: f64) -> Result<f64, String> {
    if values.is_empty() {
        return Err("CCA quantile requires non-empty control effects".into());
    }
    let mut ord: Vec<f64> = values.to_vec();
    if ord.iter().any(|v| !v.is_finite()) {
        return Err("non-finite effect in CCA quantile".into());
    }
    ord.sort_by(|a, b| a.total_cmp(b));
    let pos = ((ord.len() - 1) as f64) * prob.clamp(0.0, 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(ord.len() - 1);
    let frac = pos - (lo as f64);
    Ok(ord[lo] * (1.0 - frac) + ord[hi] * frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cca_cluster_plan_and_enforces_core_bypass_invariant() {
        let sample = br#"{
          "schema_version": "cca.cluster_plan.v1",
          "model_id": "google/gemma-4-31B-it",
          "num_layers": 2,
          "d_model": 5376,
          "cca_manifest_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
          "bound_cluster_psk": "p:cca:0123456789abcdef",
          "layer_policies": [
            {"layer": 0, "measured": true, "selected_cognitive_core": true, "tier2_bypass": false, "mlp_top_p": 1.0, "head_top_p": 1.0, "mod_thresh": 0.0},
            {"layer": 1, "measured": true, "selected_cognitive_core": false, "tier2_bypass": true, "mlp_top_p": 0.143, "head_top_p": 0.25, "mod_thresh": 0.55}
          ],
          "shards": [
            {"shard_index": 0, "node_name": "n0", "endpoint": "127.0.0.1:19555", "layer_start": 0, "layer_end": 2, "skip_embed_tokens": false, "luzkcl11_shard_attestation": "abcd"}
          ]
        }"#;
        let plan = CcaClusterPlan::from_json_bytes(sample).expect("valid plan");
        assert_eq!(plan.selected_cognitive_core_layers, vec![0]);
        assert_eq!(plan.tier2_bypass_layers, vec![1]);
        assert_eq!(plan.bind_psk("p"), "p:cca:0123456789abcdef");

        let bad = br#"{
          "schema_version": "cca.cluster_plan.v1",
          "model_id": "m",
          "num_layers": 1,
          "d_model": 64,
          "cca_manifest_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
          "layer_policies": [
            {"layer": 0, "measured": true, "selected_cognitive_core": true, "tier2_bypass": true, "mlp_top_p": 1.0, "head_top_p": 1.0, "mod_thresh": 0.0}
          ]
        }"#;
        assert!(CcaClusterPlan::from_json_bytes(bad).is_err());
    }

    #[test]
    fn tbi_cognitive_tag_packs_into_unused_arm64_top_byte_and_recovers_va_losslessly() {
        let pol_core = CcaLayerPolicy {
            layer: 17,
            measured: true,
            selected_cognitive_core: true,
            tier2_bypass: false,
            mlp_top_p: 1.0,
            head_top_p: 1.0,
            mod_thresh: 0.0,
        };
        let pol_cold = CcaLayerPolicy {
            layer: 46,
            measured: true,
            selected_cognitive_core: false,
            tier2_bypass: true,
            mlp_top_p: 0.143,
            head_top_p: 0.25,
            mod_thresh: 0.55,
        };
        let fake_va: u64 = 0x0000_7fff_8123_4000;
        let t_core = pol_core.intra_chip_tbi_tag().tag_va(fake_va);
        let t_cold = pol_cold.intra_chip_tbi_tag().tag_va(fake_va);
        assert_ne!(t_core >> 56, t_cold >> 56);
        let (dec_core, va1) = TbiCognitiveTag::from_tagged_va(t_core);
        let (dec_cold, va2) = TbiCognitiveTag::from_tagged_va(t_cold);
        assert_eq!(va1, fake_va);
        assert_eq!(va2, fake_va);
        assert_eq!(dec_core.tier, 3);
        assert_eq!(dec_core.abits, 8);
        assert_eq!(dec_cold.tier, 0);
        assert_eq!(dec_cold.salience_q, 0);
    }
}

//! V7+ Exact `<eot>` Byte-Head Speculative Decoder (`BHDR` v1 + Gemma-4 BPE Prefix Trie).
//!
//! Replaces the 262,144-way draft LM head (`~268 MB`) of `Gemma4AssistantForCausalLM`
//! with a compact INT8 byte+`<eot>` GRU micro-decoder constrained over an exact CSR
//! BPE prefix trie built from Gemma-4's `tokenizer.json`. Adapted from arXiv:2609.12303
//! Appendix F.1 exact single-pass `<eot>` token-to-byte marginalization.

use crate::tokenizer::Tokenizer;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const BHDR_MAGIC: &[u8; 4] = b"BHDR";
pub const BHDR_VERSION: u32 = 1;
pub const BHDR_HEADER_BYTES: usize = 32;

pub const SYM_EOT: u16 = 256;
pub const SYM_SP: u16 = 257; // SentencePiece U+2581 ('▁')
pub const SYM_EOS: u16 = 258; // Canonical id=1 "<eos>"
pub const SYM_TURN_END: u16 = 259; // Canonical id=106 "<turn|>" / "<end_of_turn>"
pub const SYM_BYTE_ESC: u16 = 260; // Escape prefix before raw byte 0..=255 for "<0xNN>"
pub const BHDR_VOCAB_SYM: usize = 261;

const NO_TOKEN: u32 = u32::MAX;

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
    r.read_exact(&mut b).map_err(|e| format!("BHDR read u32: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32_vec<R: Read>(r: &mut R, len: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut raw = vec![0u8; len * 4];
    r.read_exact(&mut raw).map_err(|e| format!("BHDR read {label}: {e}"))?;
    let mut out = Vec::with_capacity(len);
    for ch in raw.chunks_exact(4) {
        let v = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
        if !v.is_finite() {
            return Err(format!("BHDR non-finite float in {label}: {v}"));
        }
        out.push(v);
    }
    Ok(out)
}

fn read_i8_vec<R: Read>(r: &mut R, len: usize, label: &str) -> Result<Vec<i8>, String> {
    let mut raw = vec![0u8; len];
    r.read_exact(&mut raw).map_err(|e| format!("BHDR read {label}: {e}"))?;
    Ok(raw.into_iter().map(|b| b as i8).collect())
}

fn write_f32_slice<W: Write>(w: &mut W, vals: &[f32]) -> Result<(), String> {
    for &v in vals {
        if !v.is_finite() {
            return Err(format!("BHDR refusing to serialize non-finite float {v}"));
        }
        w.write_all(&v.to_le_bytes())
            .map_err(|e| format!("BHDR write f32: {e}"))?;
    }
    Ok(())
}

fn write_i8_slice<W: Write>(w: &mut W, vals: &[i8]) -> Result<(), String> {
    let bytes: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
    w.write_all(&bytes).map_err(|e| format!("BHDR write i8: {e}"))
}

/// Convert one Gemma tokenizer piece `(token_id, tok_str)` into its canonical
/// 261-alphabet symbol sequence (excluding trailing `SYM_EOT`).
pub fn token_to_symbols(token_id: u32, tok_str: &str) -> Vec<u16> {
    if tok_str.is_empty() {
        return Vec::new();
    }
    if token_id == 1 && tok_str == "<eos>" {
        return vec![SYM_EOS];
    }
    if token_id == 106 && (tok_str == "<turn|>" || tok_str == "<end_of_turn>") {
        return vec![SYM_TURN_END];
    }
    if tok_str.len() == 6 && tok_str.starts_with("<0x") && tok_str.ends_with('>') {
        if let Ok(b) = u8::from_str_radix(&tok_str[3..5], 16) {
            return vec![SYM_BYTE_ESC, b as u16];
        }
    }
    let mut out = Vec::with_capacity(tok_str.len());
    let mut utf8 = [0u8; 4];
    for ch in tok_str.chars() {
        if ch == '\u{2581}' {
            out.push(SYM_SP);
        } else {
            for &b in ch.encode_utf8(&mut utf8).as_bytes() {
                out.push(b as u16);
            }
        }
    }
    out
}

/// Compact CSR node for the constrained Gemma-4 BPE prefix trie.
#[derive(Clone, Copy, Debug)]
pub struct TrieNode {
    pub edge_start: u32,
    pub edge_count: u16,
    pub token_id: u32,
}

/// Compact CSR edge `(sym, child_node_id)` sorted strictly by `sym` per node.
#[derive(Clone, Copy, Debug)]
pub struct TrieEdge {
    pub sym: u16,
    pub child: u32,
}

struct BuildTrieNode {
    children: Vec<(u16, u32)>,
    token_id: u32,
}

/// Exact prefix-free CSR BPE trie over `token_to_symbols` for all valid vocab pieces.
#[derive(Clone, Debug)]
pub struct BpePrefixTrie {
    pub nodes: Vec<TrieNode>,
    pub edges: Vec<TrieEdge>,
    pub covered_tokens: usize,
    pub collision_count: usize,
    pub max_depth: usize,
}

impl BpePrefixTrie {
    /// Construct directly from a loaded Gemma `Tokenizer`.
    pub fn from_tokenizer(tk: &Tokenizer) -> Self {
        Self::from_id_to_token(&tk.id_to_token)
    }

    /// Construct from `id_to_token` strings, sorting child edges per node into CSR arrays.
    pub fn from_id_to_token(id_to_token: &[String]) -> Self {
        let mut build_nodes = Vec::with_capacity(id_to_token.len().saturating_mul(2).max(16));
        build_nodes.push(BuildTrieNode {
            children: Vec::new(),
            token_id: NO_TOKEN,
        });
        let mut covered_tokens = 0usize;
        let mut collision_count = 0usize;
        let mut max_depth = 0usize;

        for (id_usize, tok_str) in id_to_token.iter().enumerate() {
            let tid = id_usize as u32;
            let syms = token_to_symbols(tid, tok_str);
            if syms.is_empty() {
                continue;
            }
            max_depth = max_depth.max(syms.len());
            let mut cur = 0u32;
            for &sym in &syms {
                let next_opt = build_nodes[cur as usize]
                    .children
                    .iter()
                    .find_map(|&(s, c)| if s == sym { Some(c) } else { None });
                if let Some(next_id) = next_opt {
                    cur = next_id;
                } else {
                    let new_id = build_nodes.len() as u32;
                    build_nodes[cur as usize].children.push((sym, new_id));
                    build_nodes.push(BuildTrieNode {
                        children: Vec::new(),
                        token_id: NO_TOKEN,
                    });
                    cur = new_id;
                }
            }
            let slot = &mut build_nodes[cur as usize].token_id;
            if *slot == NO_TOKEN {
                *slot = tid;
                covered_tokens += 1;
            } else {
                collision_count += 1;
            }
        }

        let mut total_edges = 0usize;
        for bn in &mut build_nodes {
            bn.children.sort_unstable_by_key(|&(sym, _)| sym);
            total_edges += bn.children.len();
        }
        let mut nodes = Vec::with_capacity(build_nodes.len());
        let mut edges = Vec::with_capacity(total_edges);
        for bn in build_nodes {
            let edge_start = edges.len() as u32;
            let edge_count = u16::try_from(bn.children.len()).unwrap_or(u16::MAX);
            for (sym, child) in bn.children {
                edges.push(TrieEdge { sym, child });
            }
            nodes.push(TrieNode {
                edge_start,
                edge_count,
                token_id: bn.token_id,
            });
        }

        Self {
            nodes,
            edges,
            covered_tokens,
            collision_count,
            max_depth,
        }
    }

    /// Total heap byte footprint of the CSR trie arrays.
    pub fn byte_len(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<TrieNode>()
            + self.edges.len() * std::mem::size_of::<TrieEdge>()
    }

    /// Valid outgoing symbol edges sorted ascending by `sym`.
    #[inline]
    pub fn node_edges(&self, node: u32) -> &[TrieEdge] {
        let Some(nd) = self.nodes.get(node as usize) else {
            return &[];
        };
        let s = nd.edge_start as usize;
        let e = s + nd.edge_count as usize;
        &self.edges[s..e]
    }

    /// Advance one symbol in the trie via binary search.
    #[inline]
    pub fn step(&self, node: u32, sym: u16) -> Option<u32> {
        let edges = self.node_edges(node);
        edges
            .binary_search_by_key(&sym, |e| e.sym)
            .ok()
            .map(|idx| edges[idx].child)
    }

    /// Return the Gemma token ID completing at `node`, if any.
    #[inline]
    pub fn terminal_token(&self, node: u32) -> Option<u32> {
        let nd = self.nodes.get(node as usize)?;
        if nd.token_id == NO_TOKEN {
            None
        } else {
            Some(nd.token_id)
        }
    }

    /// Lookup exact token ID for a full symbol slice (without `SYM_EOT`).
    pub fn lookup_symbols(&self, syms: &[u16]) -> Option<u32> {
        let mut node = 0u32;
        for &sym in syms {
            node = self.step(node, sym)?;
        }
        self.terminal_token(node)
    }
}

/// INT8 compact 1-layer byte+`<eot>` GRU micro-decoder (`BHDR` v1).
#[derive(Clone, Debug)]
pub struct ByteHeadDecoder {
    pub d_in: usize,
    pub d_byte: usize,
    pub vocab_sym: usize,
    pub max_bytes: usize,
    pub flags: u32,
    pub w_init: Vec<i8>,
    pub s_init: Vec<f32>,
    pub b_init: Vec<f32>,
    pub w_ctx: Vec<i8>,
    pub s_ctx: Vec<f32>,
    pub e_sym: Vec<i8>,
    pub s_sym: Vec<f32>,
    pub w_ih: Vec<i8>,
    pub s_ih: Vec<f32>,
    pub b_ih: Vec<f32>,
    pub w_hh: Vec<i8>,
    pub s_hh: Vec<f32>,
    pub b_hh: Vec<f32>,
    pub w_out: Vec<i8>,
    pub s_out: Vec<f32>,
    pub b_out: Vec<f32>,
}

impl ByteHeadDecoder {
    pub fn expected_byte_len(d_in: usize, d_byte: usize, vocab_sym: usize) -> usize {
        BHDR_HEADER_BYTES
            + d_byte * d_in
            + d_byte * 4
            + d_byte * 4
            + 3 * d_byte * d_in
            + 3 * d_byte * 4
            + vocab_sym * d_byte
            + vocab_sym * 4
            + 3 * d_byte * d_byte
            + 3 * d_byte * 4
            + 3 * d_byte * 4
            + 3 * d_byte * d_byte
            + 3 * d_byte * 4
            + 3 * d_byte * 4
            + vocab_sym * d_byte
            + vocab_sym * 4
            + vocab_sym * 4
    }

    pub fn byte_len(&self) -> usize {
        Self::expected_byte_len(self.d_in, self.d_byte, self.vocab_sym)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.d_in == 0 || self.d_byte == 0 || self.vocab_sym != BHDR_VOCAB_SYM || self.max_bytes == 0 {
            return Err(format!(
                "BHDR invalid header dims d_in={} d_byte={} vocab_sym={} max_bytes={}",
                self.d_in, self.d_byte, self.vocab_sym, self.max_bytes
            ));
        }
        if self.flags != 0 {
            return Err(format!("BHDR unknown header flags 0x{:x}", self.flags));
        }
        let d_in = self.d_in;
        let db = self.d_byte;
        let vs = self.vocab_sym;
        if self.w_init.len() != db * d_in
            || self.s_init.len() != db
            || self.b_init.len() != db
            || self.w_ctx.len() != 3 * db * d_in
            || self.s_ctx.len() != 3 * db
            || self.e_sym.len() != vs * db
            || self.s_sym.len() != vs
            || self.w_ih.len() != 3 * db * db
            || self.s_ih.len() != 3 * db
            || self.b_ih.len() != 3 * db
            || self.w_hh.len() != 3 * db * db
            || self.s_hh.len() != 3 * db
            || self.b_hh.len() != 3 * db
            || self.w_out.len() != vs * db
            || self.s_out.len() != vs
            || self.b_out.len() != vs
        {
            return Err("BHDR tensor array length mismatch".into());
        }
        for (label, sl) in [
            ("s_init", self.s_init.as_slice()),
            ("b_init", self.b_init.as_slice()),
            ("s_ctx", self.s_ctx.as_slice()),
            ("s_sym", self.s_sym.as_slice()),
            ("s_ih", self.s_ih.as_slice()),
            ("b_ih", self.b_ih.as_slice()),
            ("s_hh", self.s_hh.as_slice()),
            ("b_hh", self.b_hh.as_slice()),
            ("s_out", self.s_out.as_slice()),
            ("b_out", self.b_out.as_slice()),
        ] {
            if sl.iter().any(|v| !v.is_finite()) {
                return Err(format!("BHDR non-finite value in {label}"));
            }
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<usize, String> {
        self.validate()?;
        let f = File::create(path).map_err(|e| format!("BHDR create {}: {e}", path.display()))?;
        let mut w = BufWriter::new(f);
        w.write_all(BHDR_MAGIC)
            .map_err(|e| format!("BHDR write magic: {e}"))?;
        for &val in &[
            BHDR_VERSION,
            self.d_in as u32,
            self.d_byte as u32,
            self.vocab_sym as u32,
            self.max_bytes as u32,
            self.flags,
            0u32,
        ] {
            w.write_all(&val.to_le_bytes())
                .map_err(|e| format!("BHDR write header: {e}"))?;
        }
        write_i8_slice(&mut w, &self.w_init)?;
        write_f32_slice(&mut w, &self.s_init)?;
        write_f32_slice(&mut w, &self.b_init)?;
        write_i8_slice(&mut w, &self.w_ctx)?;
        write_f32_slice(&mut w, &self.s_ctx)?;
        write_i8_slice(&mut w, &self.e_sym)?;
        write_f32_slice(&mut w, &self.s_sym)?;
        write_i8_slice(&mut w, &self.w_ih)?;
        write_f32_slice(&mut w, &self.s_ih)?;
        write_f32_slice(&mut w, &self.b_ih)?;
        write_i8_slice(&mut w, &self.w_hh)?;
        write_f32_slice(&mut w, &self.s_hh)?;
        write_f32_slice(&mut w, &self.b_hh)?;
        write_i8_slice(&mut w, &self.w_out)?;
        write_f32_slice(&mut w, &self.s_out)?;
        write_f32_slice(&mut w, &self.b_out)?;
        w.flush().map_err(|e| format!("BHDR flush {}: {e}", path.display()))?;
        Ok(self.byte_len())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("BHDR open {}: {e}", path.display()))?;
        let file_len = f
            .metadata()
            .map_err(|e| format!("BHDR metadata {}: {e}", path.display()))?
            .len() as usize;
        if file_len < BHDR_HEADER_BYTES {
            return Err(format!(
                "BHDR file {} truncated: {file_len} < {BHDR_HEADER_BYTES}",
                path.display()
            ));
        }
        let mut r = BufReader::new(f);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)
            .map_err(|e| format!("BHDR read magic {}: {e}", path.display()))?;
        if &magic != BHDR_MAGIC {
            return Err(format!("BHDR invalid magic {:?} in {}", magic, path.display()));
        }
        let version = read_u32_le(&mut r)?;
        if version != BHDR_VERSION {
            return Err(format!("BHDR unsupported version {version} != {BHDR_VERSION}"));
        }
        let d_in = read_u32_le(&mut r)? as usize;
        let d_byte = read_u32_le(&mut r)? as usize;
        let vocab_sym = read_u32_le(&mut r)? as usize;
        let max_bytes = read_u32_le(&mut r)? as usize;
        let flags = read_u32_le(&mut r)?;
        let reserved = read_u32_le(&mut r)?;
        if reserved != 0 {
            return Err(format!("BHDR non-zero reserved field {reserved}"));
        }
        if d_in == 0 || d_byte == 0 || vocab_sym != BHDR_VOCAB_SYM || max_bytes == 0 || flags != 0 {
            return Err(format!(
                "BHDR invalid header params d_in={d_in} d_byte={d_byte} vocab_sym={vocab_sym} max_bytes={max_bytes} flags=0x{flags:x}"
            ));
        }
        let expected_len = Self::expected_byte_len(d_in, d_byte, vocab_sym);
        if file_len != expected_len {
            return Err(format!(
                "BHDR byte length mismatch in {}: actual={file_len} != expected={expected_len}",
                path.display()
            ));
        }

        let db = d_byte;
        let dec = Self {
            d_in,
            d_byte,
            vocab_sym,
            max_bytes,
            flags,
            w_init: read_i8_vec(&mut r, db * d_in, "w_init")?,
            s_init: read_f32_vec(&mut r, db, "s_init")?,
            b_init: read_f32_vec(&mut r, db, "b_init")?,
            w_ctx: read_i8_vec(&mut r, 3 * db * d_in, "w_ctx")?,
            s_ctx: read_f32_vec(&mut r, 3 * db, "s_ctx")?,
            e_sym: read_i8_vec(&mut r, vocab_sym * db, "e_sym")?,
            s_sym: read_f32_vec(&mut r, vocab_sym, "s_sym")?,
            w_ih: read_i8_vec(&mut r, 3 * db * db, "w_ih")?,
            s_ih: read_f32_vec(&mut r, 3 * db, "s_ih")?,
            b_ih: read_f32_vec(&mut r, 3 * db, "b_ih")?,
            w_hh: read_i8_vec(&mut r, 3 * db * db, "w_hh")?,
            s_hh: read_f32_vec(&mut r, 3 * db, "s_hh")?,
            b_hh: read_f32_vec(&mut r, 3 * db, "b_hh")?,
            w_out: read_i8_vec(&mut r, vocab_sym * db, "w_out")?,
            s_out: read_f32_vec(&mut r, vocab_sym, "s_out")?,
            b_out: read_f32_vec(&mut r, vocab_sym, "b_out")?,
        };
        dec.validate()?;
        Ok(dec)
    }

    /// Initialize `s0 = tanh(W_init @ h_final + b_init)` and `ctx = W_ctx @ h_final`.
    pub fn init_state_and_ctx(&self, h_final: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        if h_final.len() != self.d_in {
            return Err(format!(
                "ByteHeadDecoder::init_state_and_ctx: h_final len {} != d_in {}",
                h_final.len(),
                self.d_in
            ));
        }
        let db = self.d_byte;
        let mut s0 = Vec::with_capacity(db);
        for r in 0..db {
            let row = &self.w_init[r * self.d_in..(r + 1) * self.d_in];
            let v = self.s_init[r] * dot_i8_f32_4way(row, h_final) + self.b_init[r];
            s0.push(v.tanh());
        }
        let mut ctx = Vec::with_capacity(3 * db);
        for r in 0..3 * db {
            let row = &self.w_ctx[r * self.d_in..(r + 1) * self.d_in];
            ctx.push(self.s_ctx[r] * dot_i8_f32_4way(row, h_final));
        }
        Ok((s0, ctx))
    }

    /// Compute exact normalized `log_softmax` over only valid outgoing trie symbols + optional `SYM_EOT`.
    pub fn node_log_probs(
        &self,
        state: &[f32],
        trie: &BpePrefixTrie,
        node: u32,
        ban_tokens: &[u32],
    ) -> Vec<(u16, f32)> {
        let edges = trie.node_edges(node);
        let term_ok = trie
            .terminal_token(node)
            .is_some_and(|tid| !ban_tokens.contains(&tid));
        let cap = edges.len() + usize::from(term_ok);
        if cap == 0 {
            return Vec::new();
        }
        let db = self.d_byte;
        let mut out: Vec<(u16, f32)> = Vec::with_capacity(cap);
        let mut max_logit = f32::NEG_INFINITY;
        let mut score_sym = |sym: u16| {
            let s = sym as usize;
            let row = &self.w_out[s * db..(s + 1) * db];
            let logit = self.s_out[s] * dot_i8_f32_4way(row, state) + self.b_out[s];
            if logit > max_logit {
                max_logit = logit;
            }
            out.push((sym, logit));
        };
        if term_ok {
            score_sym(SYM_EOT);
        }
        for edge in edges {
            score_sym(edge.sym);
        }
        let mut sum_exp = 0.0f64;
        for &(_, lg) in &out {
            sum_exp += ((lg - max_logit) as f64).exp();
        }
        let log_z = (max_logit as f64) + sum_exp.ln();
        for &mut (_, ref mut v) in &mut out {
            *v = ((*v as f64) - log_z) as f32;
        }
        out
    }

    /// Advance the compact GRU micro-state by one emitted symbol `sym` with backbone conditioning `ctx`.
    pub fn gru_step(&self, state: &[f32], sym: u16, ctx: &[f32]) -> Vec<f32> {
        let db = self.d_byte;
        let s_idx = sym as usize;
        let erow = &self.e_sym[s_idx * db..(s_idx + 1) * db];
        let es = self.s_sym[s_idx];
        let mut x_emb = vec![0.0f32; db];
        for j in 0..db {
            x_emb[j] = es * (erow[j] as f32);
        }
        let mut ih = vec![0.0f32; 3 * db];
        let mut hh = vec![0.0f32; 3 * db];
        for r in 0..3 * db {
            let irow = &self.w_ih[r * db..(r + 1) * db];
            let hrow = &self.w_hh[r * db..(r + 1) * db];
            ih[r] = self.s_ih[r] * dot_i8_f32_4way(irow, &x_emb) + self.b_ih[r] + ctx[r];
            hh[r] = self.s_hh[r] * dot_i8_f32_4way(hrow, state) + self.b_hh[r];
        }
        let mut next_state = Vec::with_capacity(db);
        for j in 0..db {
            let r = sigmoid_f32(ih[j] + hh[j]);
            let z = sigmoid_f32(ih[db + j] + hh[db + j]);
            let n = (ih[2 * db + j] + r * hh[2 * db + j]).tanh();
            next_state.push((1.0f32 - z) * n + z * state[j]);
        }
        next_state
    }

    /// Exact trie-constrained beam search returning `(best_tok, top1_logp, top2_logp, bytes_unrolled)`.
    pub fn decode_token_beam(
        &self,
        h_final: &[f32],
        trie: &BpePrefixTrie,
        ban_tokens: &[u32],
        beam_width: usize,
    ) -> Result<(u32, f32, f32, usize), String> {
        let bw = beam_width.max(1);
        let (s0, ctx) = self.init_state_and_ctx(h_final)?;

        struct Beam {
            cum_logp: f32,
            node: u32,
            depth: usize,
            state: Vec<f32>,
        }
        struct NextCand {
            cum_logp: f32,
            child: u32,
            depth: usize,
            parent_idx: usize,
            sym: u16,
        }

        let mut beams = vec![Beam {
            cum_logp: 0.0f32,
            node: 0u32,
            depth: 0usize,
            state: s0,
        }];
        let mut completed: Vec<(f32, u32, usize)> = Vec::new();
        let mut pruned_alt_logp = f32::NEG_INFINITY;

        for step_idx in 0..=self.max_bytes {
            let allow_expand = step_idx < self.max_bytes;
            let mut next_cands: Vec<NextCand> = Vec::new();

            for (bi, bm) in beams.iter().enumerate() {
                let lps = self.node_log_probs(&bm.state, trie, bm.node, ban_tokens);
                for (sym, lp) in lps {
                    let tot = bm.cum_logp + lp;
                    if sym == SYM_EOT {
                        if let Some(tid) = trie.terminal_token(bm.node) {
                            completed.push((tot, tid, bm.depth));
                        }
                    } else if allow_expand {
                        if let Some(child) = trie.step(bm.node, sym) {
                            next_cands.push(NextCand {
                                cum_logp: tot,
                                child,
                                depth: bm.depth + 1,
                                parent_idx: bi,
                                sym,
                            });
                        }
                    } else if tot > pruned_alt_logp {
                        pruned_alt_logp = tot;
                    }
                }
            }

            if !allow_expand || next_cands.is_empty() {
                break;
            }
            next_cands.sort_unstable_by(|a, b| b.cum_logp.total_cmp(&a.cum_logp));
            if next_cands.len() > bw {
                pruned_alt_logp = pruned_alt_logp.max(next_cands[bw].cum_logp);
                next_cands.truncate(bw);
            }

            completed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            if completed.len() >= 2 && next_cands[0].cum_logp <= completed[1].0 {
                break;
            }

            let mut next_beams = Vec::with_capacity(next_cands.len());
            for c in next_cands {
                let nstate = self.gru_step(&beams[c.parent_idx].state, c.sym, &ctx);
                next_beams.push(Beam {
                    cum_logp: c.cum_logp,
                    node: c.child,
                    depth: c.depth,
                    state: nstate,
                });
            }
            beams = next_beams;
        }

        if completed.is_empty() {
            return Err("ByteHeadDecoder::decode_token_beam: no valid unbanned token reached <eot>".into());
        }
        completed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let (top1_logp, best_tok, best_depth) = completed[0];
        let top2_logp = if let Some(&(s2, _, _)) = completed.get(1) {
            s2.max(pruned_alt_logp)
        } else if pruned_alt_logp.is_finite() {
            pruned_alt_logp
        } else {
            top1_logp - 30.0
        };
        Ok((best_tok, top1_logp, top2_logp, best_depth))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_bhdr_path(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("bhdr_{tag}_{}_{nanos}.bhdr", std::process::id()))
    }

    fn build_planted_decoder(d_in: usize, d_byte: usize, max_bytes: usize) -> ByteHeadDecoder {
        let vs = BHDR_VOCAB_SYM;
        let mut dec = ByteHeadDecoder {
            d_in,
            d_byte,
            vocab_sym: vs,
            max_bytes,
            flags: 0,
            w_init: vec![0i8; d_byte * d_in],
            s_init: vec![0.02f32; d_byte],
            b_init: vec![0.0f32; d_byte],
            w_ctx: vec![0i8; 3 * d_byte * d_in],
            s_ctx: vec![0.01f32; 3 * d_byte],
            e_sym: vec![0i8; vs * d_byte],
            s_sym: vec![0.02f32; vs],
            w_ih: vec![0i8; 3 * d_byte * d_byte],
            s_ih: vec![0.02f32; 3 * d_byte],
            b_ih: vec![-2.0f32; 3 * d_byte],
            w_hh: vec![0i8; 3 * d_byte * d_byte],
            s_hh: vec![0.02f32; 3 * d_byte],
            b_hh: vec![0.0f32; 3 * d_byte],
            w_out: vec![0i8; vs * d_byte],
            s_out: vec![0.10f32; vs],
            b_out: vec![-4.0f32; vs],
        };
        // Non-zero structured weights across feature subspaces
        for r in 0..d_byte {
            for c in 0..d_in {
                dec.w_init[r * d_in + c] = if (r + c) % 4 == 0 { 90 } else { -30 };
            }
        }
        // Make input dim 0->SYM_SP path ("▁ok") vs input dim 1->"cat" distinct!
        // Reset b_ih[z..] negative so GRU rapidly tracks new emitted symbol embedding.
        for j in 0..d_byte {
            dec.b_ih[j] = 1.0; // reset gate mostly open
            dec.b_ih[d_byte + j] = -3.0; // update gate favors new candidate
            dec.b_ih[2 * d_byte + j] = 0.0;
        }
        // Map dim 0..4 strongly in w_init[:, 0] -> favors SYM_SP
        for j in 0..4 {
            dec.w_init[j * d_in] = 127;
            dec.w_out[(SYM_SP as usize) * d_byte + j] = 127;
            // Embedding of SYM_SP sets dims 4..8, which trigger 'o'
            dec.e_sym[(SYM_SP as usize) * d_byte + 4 + j] = 127;
            dec.w_ih[(2 * d_byte + 4 + j) * d_byte + 4 + j] = 127;
            dec.w_out[(b'o' as usize) * d_byte + 4 + j] = 127;
            // Embedding of 'o' sets dims 8..12, which trigger 'k'
            dec.e_sym[(b'o' as usize) * d_byte + 8 + j] = 127;
            dec.w_ih[(2 * d_byte + 8 + j) * d_byte + 8 + j] = 127;
            dec.w_out[(b'k' as usize) * d_byte + 8 + j] = 127;
            // Embedding of 'k' sets dims 12..16, which trigger SYM_EOT
            dec.e_sym[(b'k' as usize) * d_byte + 12 + j] = 127;
            dec.w_ih[(2 * d_byte + 12 + j) * d_byte + 12 + j] = 127;
            dec.w_out[(SYM_EOT as usize) * d_byte + 12 + j] = 127;

            // Alternate path from h_final[1] -> '4' -> SYM_EOT
            dec.w_init[(16 + j) * d_in + 1] = 127;
            dec.w_out[(b'4' as usize) * d_byte + 16 + j] = 127;
            dec.e_sym[(b'4' as usize) * d_byte + 12 + j] = 127;
        }
        dec.b_out[SYM_SP as usize] = 0.5;
        dec.b_out[b'o' as usize] = 0.5;
        dec.b_out[b'k' as usize] = 0.5;
        dec.b_out[b'4' as usize] = 0.4;
        dec.b_out[SYM_EOT as usize] = 1.0;
        dec
    }

    fn mini_vocab() -> Vec<String> {
        let mut v = vec![String::new(); 120];
        v[0] = "<pad>".into();
        v[1] = "<eos>".into();
        v[2] = "<bos>".into();
        v[10] = "\u{2581}o".into();
        v[11] = "\u{2581}ok".into();
        v[12] = "4".into();
        v[13] = "<0xF0>".into();
        v[106] = "<turn|>".into();
        v
    }

    #[test]
    fn gate1_header_eof_and_roundtrip() {
        let dec = build_planted_decoder(32, 32, 16);
        let p = temp_bhdr_path("g1");
        let n = dec.save(&p).unwrap();
        assert_eq!(n, dec.byte_len());
        assert_eq!(std::fs::metadata(&p).unwrap().len() as usize, n);
        let loaded = ByteHeadDecoder::load(&p).unwrap();
        assert_eq!(loaded.w_init, dec.w_init);
        assert_eq!(loaded.w_out, dec.w_out);

        let mut bad = std::fs::read(&p).unwrap();
        bad[0] = b'Z';
        let p_bad = temp_bhdr_path("g1_bad");
        std::fs::write(&p_bad, &bad).unwrap();
        assert!(ByteHeadDecoder::load(&p_bad).is_err());

        let p_trunc = temp_bhdr_path("g1_trunc");
        std::fs::write(&p_trunc, &std::fs::read(&p).unwrap()[..n - 2]).unwrap();
        assert!(ByteHeadDecoder::load(&p_trunc).is_err());

        for f in [&p, &p_bad, &p_trunc] {
            let _ = std::fs::remove_file(f);
        }
    }

    #[test]
    fn gate2_weight_sanity_and_validation() {
        let mut dec = build_planted_decoder(32, 32, 16);
        let sum: f64 = dec.w_init.iter().map(|&v| v as f64).sum();
        let mean = sum / dec.w_init.len() as f64;
        let var: f64 = dec
            .w_init
            .iter()
            .map(|&v| {
                let d = (v as f64) - mean;
                d * d
            })
            .sum::<f64>()
            / dec.w_init.len() as f64;
        assert!(var > 100.0, "expected non-zero weight variance, got {var}");
        dec.s_out[0] = f32::NAN;
        assert!(dec.validate().is_err());
    }

    #[test]
    fn gate3_distinct_inputs_distinct_logits() {
        let dec = build_planted_decoder(32, 32, 16);
        let trie = BpePrefixTrie::from_id_to_token(&mini_vocab());
        let mut h0 = vec![0.0f32; 32];
        h0[0] = 2.0;
        let mut h1 = vec![0.0f32; 32];
        h1[1] = 2.0;
        let (s0, _) = dec.init_state_and_ctx(&h0).unwrap();
        let (s1, _) = dec.init_state_and_ctx(&h1).unwrap();
        let lp0 = dec.node_log_probs(&s0, &trie, 0, &[]);
        let lp1 = dec.node_log_probs(&s1, &trie, 0, &[]);
        let sp0 = lp0.iter().find(|&&(s, _)| s == SYM_SP).unwrap().1;
        let sp1 = lp1.iter().find(|&&(s, _)| s == SYM_SP).unwrap().1;
        let d4_0 = lp0.iter().find(|&&(s, _)| s == b'4' as u16).unwrap().1;
        let d4_1 = lp1.iter().find(|&&(s, _)| s == b'4' as u16).unwrap().1;
        assert!(sp0 > d4_0, "h0 should prefer SYM_SP ({sp0}) > '4' ({d4_0})");
        assert!(d4_1 > sp1, "h1 should prefer '4' ({d4_1}) > SYM_SP ({sp1})");
    }

    #[test]
    fn gate4_trie_bijection_and_beam_decode_respects_bans() {
        let vocab = mini_vocab();
        let trie = BpePrefixTrie::from_id_to_token(&vocab);
        assert_eq!(trie.collision_count, 0);
        assert_eq!(trie.covered_tokens, 8);
        assert_eq!(trie.lookup_symbols(&token_to_symbols(1, "<eos>")), Some(1));
        assert_eq!(trie.lookup_symbols(&token_to_symbols(106, "<turn|>")), Some(106));
        assert_eq!(trie.lookup_symbols(&token_to_symbols(13, "<0xF0>")), Some(13));
        assert_eq!(trie.lookup_symbols(&token_to_symbols(11, "\u{2581}ok")), Some(11));

        let dec = build_planted_decoder(32, 32, 16);
        let mut h1 = vec![0.0f32; 32];
        h1[1] = 2.0;
        let (tok, lp1, lp2, depth) = dec.decode_token_beam(&h1, &trie, &[], 2).unwrap();
        assert_eq!(tok, 12); // "4"
        assert_eq!(depth, 1);
        assert!(lp1 > lp2);

        // Ban token 12 -> must emit another valid unbanned token instead of panicking or emitting 12
        let (tok_alt, _, _, _) = dec.decode_token_beam(&h1, &trie, &[12], 2).unwrap();
        assert_ne!(tok_alt, 12);
    }

    #[test]
    fn gate5_coverage_and_boundary_guards() {
        let vocab = mini_vocab();
        let trie = BpePrefixTrie::from_id_to_token(&vocab);
        let mut h_short = vec![0.0f32; 8];
        h_short[0] = 1.0;
        let dec = build_planted_decoder(32, 32, 16);
        assert!(dec.init_state_and_ctx(&h_short).is_err());

        // Root node (0) has no terminal token so node_log_probs excludes SYM_EOT
        let (s0, _) = dec.init_state_and_ctx(&vec![0.0f32; 32]).unwrap();
        let root_lps = dec.node_log_probs(&s0, &trie, 0, &[]);
        assert!(!root_lps.iter().any(|&(s, _)| s == SYM_EOT));
        let sum_prob: f64 = root_lps.iter().map(|&(_, lp)| (lp as f64).exp()).sum();
        assert!((sum_prob - 1.0).abs() < 1e-5, "sum_prob={sum_prob}");
    }

    #[test]
    fn gate6_end_to_end_multi_byte_token_decode() {
        let trie = BpePrefixTrie::from_id_to_token(&mini_vocab());
        let dec = build_planted_decoder(32, 32, 16);
        let mut h0 = vec![0.0f32; 32];
        h0[0] = 2.0;
        let (tok, top1_lp, top2_lp, depth) = dec.decode_token_beam(&h0, &trie, &[10], 2).unwrap();
        assert_eq!(tok, 11); // "▁ok" unrolled via [SYM_SP, 'o', 'k'] -> SYM_EOT!
        assert_eq!(depth, 3);
        assert!(top1_lp > top2_lp);
    }
}

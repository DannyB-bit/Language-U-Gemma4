//! Adaptive online n-gram speculative decoding primitives (adapted from Zymatica-Engine).
//!
//! Provides a lossless draft-and-verify state helper that guarantees greedy argmax bit-parity
//! with sequential `k = 1` autoregressive decoding via exact prefix verification and KV rollback.

use std::collections::HashMap;
use std::path::Path;

pub const CNGM_MAGIC: &[u8; 4] = b"CNGM";
pub const CNGM_VERSION: u32 = 1;

/// Adaptive draft window controller matching Zymatica-Engine's EWMA schedule.
#[derive(Debug, Clone)]
pub struct AdaptiveDraftController {
    min_k: usize,
    max_k: usize,
    current_k: usize,
    acceptance_ewma: f32,
}

impl AdaptiveDraftController {
    pub fn new(min_k: usize, max_k: usize) -> Self {
        let min_k = min_k.max(1);
        let max_k = max_k.max(min_k);
        Self {
            min_k,
            max_k,
            current_k: max_k,
            acceptance_ewma: 1.0,
        }
    }

    pub fn current_k(&self) -> usize {
        self.current_k
    }

    pub fn acceptance_ewma(&self) -> f32 {
        self.acceptance_ewma
    }

    pub fn observe(&mut self, accepted: usize, drafted: usize) -> usize {
        if drafted == 0 {
            return self.current_k;
        }
        let rate = accepted as f32 / drafted as f32;
        self.acceptance_ewma = 0.8 * self.acceptance_ewma + 0.2 * rate;
        if self.acceptance_ewma < 0.45 && self.current_k > self.min_k {
            self.current_k -= 1;
        } else if self.acceptance_ewma > 0.80 && self.current_k < self.max_k {
            self.current_k += 1;
        }
        self.current_k
    }
}

/// Online multi-order n-gram transition table with longest-context backoff (`max_n` down to `2`)
/// and recency tie-breaking.
#[derive(Debug, Clone)]
pub struct FastNGramProposalEngine {
    /// Separate transition maps indexed by context length `1..max_n`.
    /// Each entry stores `(next_token, count, last_tick)`.
    tables: Vec<HashMap<Vec<u32>, Vec<(u32, u32, u64)>>>,
    max_n: usize,
    min_count: u32,
    tick: u64,
}

impl FastNGramProposalEngine {
    pub fn new(max_n: usize) -> Self {
        let max_n = max_n.max(2);
        let tables = (0..max_n.saturating_sub(1)).map(|_| HashMap::new()).collect();
        Self {
            tables,
            max_n,
            min_count: 1,
            tick: 0,
        }
    }

    pub fn with_min_count(mut self, min_count: u32) -> Self {
        self.min_count = min_count.max(1);
        self
    }

    pub fn max_context_len(&self) -> usize {
        self.max_n - 1
    }

    /// Seed the n-gram tables from a complete prefix (e.g., prompt tokens).
    pub fn train_tokens(&mut self, tokens: &[u32]) {
        for pos in 1..tokens.len() {
            self.observe_transition(&tokens[..pos], tokens[pos]);
        }
    }

    /// Record a new `next_token` following `context` across all supported context lengths.
    pub fn observe_transition(&mut self, context: &[u32], next_token: u32) {
        if context.is_empty() {
            return;
        }
        self.tick = self.tick.saturating_add(1);
        let tick = self.tick;
        let max_ctx = self.max_context_len().min(context.len());
        for ctx_len in 1..=max_ctx {
            let key = context[context.len() - ctx_len..].to_vec();
            let entry = self.tables[ctx_len - 1].entry(key).or_default();
            if let Some(pair) = entry.iter_mut().find(|(tok, _, _)| *tok == next_token) {
                pair.1 = pair.1.saturating_add(1);
                pair.2 = tick;
            } else {
                entry.push((next_token, 1, tick));
            }
        }
    }

    /// Lookup highest-count (recency tie-broken) next token using longest-context backoff.
    fn lookup_next(&self, context: &[u32]) -> Option<u32> {
        let max_ctx = self.max_context_len().min(context.len());
        for ctx_len in (1..=max_ctx).rev() {
            // Require at least 2 occurrences for unigram (ctx_len == 1) backoff unless higher-order matches,
            // preventing noisy speculative rollbacks on common single tokens like punctuation/whitespace.
            let threshold = if ctx_len == 1 {
                self.min_count.max(2)
            } else {
                self.min_count
            };
            let key = &context[context.len() - ctx_len..];
            if let Some(cands) = self.tables[ctx_len - 1].get(key) {
                if let Some(&(tok, count, _)) = cands.iter().max_by_key(|&&(_, cnt, rec)| (cnt, rec)) {
                    if count >= threshold {
                        return Some(tok);
                    }
                }
            }
        }
        None
    }

    /// Autoregressively draft up to `draft_len` speculative tokens from `prefix`.
    pub fn propose_sequence(&self, prefix: &[u32], draft_len: usize) -> Vec<u32> {
        if draft_len == 0 || prefix.is_empty() {
            return Vec::new();
        }
        let max_ctx = self.max_context_len();
        let start = prefix.len().saturating_sub(max_ctx);
        let mut ctx = prefix[start..].to_vec();
        let mut result = Vec::with_capacity(draft_len);
        for _ in 0..draft_len {
            let Some(next_tok) = self.lookup_next(&ctx) else {
                break;
            };
            result.push(next_tok);
            ctx.push(next_tok);
            if ctx.len() > max_ctx {
                ctx.remove(0);
            }
        }
        result
    }

    /// Propose up to `max_branches` candidate continuation sequences (primary + alternate high-count branches).
    pub fn propose_multi_branches(&self, prefix: &[u32], draft_len: usize, max_branches: usize) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        if draft_len == 0 || prefix.is_empty() || max_branches == 0 {
            return out;
        }
        let primary = self.propose_sequence(prefix, draft_len);
        if primary.is_empty() {
            return out;
        }
        out.push(primary.clone());
        if max_branches == 1 {
            return out;
        }
        let max_ctx = self.max_context_len().min(prefix.len());
        for ctx_len in (2..=max_ctx).rev() {
            let key = &prefix[prefix.len() - ctx_len..];
            if let Some(cands) = self.tables[ctx_len - 1].get(key) {
                let mut sorted: Vec<(u32, u32, u64)> = cands.iter().copied().filter(|c| c.1 >= self.min_count).collect();
                sorted.sort_unstable_by(|a, b| (b.1, b.2).cmp(&(a.1, a.2)));
                for &(tok, _, _) in &sorted {
                    if tok == primary[0] {
                        continue;
                    }
                    let mut p_ext = prefix.to_vec();
                    p_ext.push(tok);
                    let mut seq = Vec::with_capacity(draft_len);
                    seq.push(tok);
                    seq.extend(self.propose_sequence(&p_ext, draft_len - 1));
                    out.push(seq);
                    if out.len() >= max_branches {
                        return out;
                    }
                }
            }
        }
        out
    }

    /// Return total stored `(context -> candidate)` transitions across all orders.
    pub fn entry_count(&self) -> usize {
        self.tables.iter().map(|t| t.values().map(|v| v.len()).sum::<usize>()).sum()
    }

    /// Serialize transitions meeting `min_count` into compact `CNGM` v1 binary format.
    pub fn to_seed_bytes(&self, min_count: u32) -> Vec<u8> {
        let mut records: Vec<(u8, &[u32], u32, u32)> = Vec::new();
        for (idx, tbl) in self.tables.iter().enumerate() {
            let ctx_len = (idx + 1) as u8;
            for (ctx, cands) in tbl {
                for &(next_tok, cnt, _) in cands {
                    if cnt >= min_count {
                        records.push((ctx_len, ctx.as_slice(), next_tok, cnt));
                    }
                }
            }
        }
        records.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)).then_with(|| b.3.cmp(&a.3)).then_with(|| a.2.cmp(&b.2)));

        let mut out = Vec::with_capacity(16 + records.len() * 20);
        out.extend_from_slice(CNGM_MAGIC);
        out.extend_from_slice(&CNGM_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.max_n as u32).to_le_bytes());
        out.extend_from_slice(&(records.len() as u32).to_le_bytes());
        for (ctx_len, ctx, next_tok, cnt) in records {
            out.push(ctx_len);
            for &id in ctx {
                out.extend_from_slice(&id.to_le_bytes());
            }
            out.extend_from_slice(&next_tok.to_le_bytes());
            out.extend_from_slice(&cnt.to_le_bytes());
        }
        out
    }

    /// Load static corpus seed transitions (`tick = 0`) from `CNGM` v1 bytes.
    ///
    /// Seed entries enter with weight `1` at `tick = 0` so cold turn-0 drafting works
    /// immediately, while live conversation repetitions (`count >= 2`, `tick >= 1`) naturally
    /// supersede generic prior transitions.
    pub fn load_seed_bytes(&mut self, data: &[u8]) -> Result<usize, String> {
        if data.len() < 16 {
            return Err(format!("CNGM seed truncated ({} bytes < 16)", data.len()));
        }
        if &data[0..4] != CNGM_MAGIC {
            return Err("CNGM bad magic header".into());
        }
        let ver = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if ver != CNGM_VERSION {
            return Err(format!("CNGM unsupported version {ver} != {CNGM_VERSION}"));
        }
        let _seed_max_n = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let record_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        let mut pos = 16usize;
        let mut loaded = 0usize;
        let max_ctx = self.max_context_len();
        for rec_idx in 0..record_count {
            if pos >= data.len() {
                return Err(format!("CNGM truncated at record {rec_idx}/{record_count}"));
            }
            let ctx_len = data[pos] as usize;
            pos += 1;
            let need = ctx_len.checked_mul(4).and_then(|v| v.checked_add(8)).ok_or("CNGM size overflow")?;
            if ctx_len == 0 || pos + need > data.len() {
                return Err(format!("CNGM invalid or truncated record {rec_idx} (ctx_len={ctx_len})"));
            }
            let mut ctx = Vec::with_capacity(ctx_len);
            for _ in 0..ctx_len {
                ctx.push(u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]));
                pos += 4;
            }
            let next_tok = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            let _seed_cnt = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
            pos += 8;

            if ctx_len <= max_ctx {
                let entry = self.tables[ctx_len - 1].entry(ctx).or_default();
                if !entry.iter().any(|(tok, _, _)| *tok == next_tok) {
                    entry.push((next_tok, 1, 0));
                    loaded += 1;
                }
            }
        }
        if pos != data.len() {
            return Err(format!("CNGM trailing bytes: consumed {pos} != len {}", data.len()));
        }
        Ok(loaded)
    }

    /// Load static corpus seed transitions from a `.cngm` file on disk.
    pub fn load_seed_file(&mut self, path: &Path) -> Result<usize, String> {
        let data = std::fs::read(path).map_err(|e| format!("read CNGM seed {}: {e}", path.display()))?;
        self.load_seed_bytes(&data)
    }
}

#[inline]
pub fn argmax_f32(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}

/// Outcome of verifying a zero-waste batch `[y_0, d_1, ..., d_r]` (`batch_logits` has `r + 1` rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeStepOutcome {
    /// Draft tokens that matched target greedy predictions (0..=draft_tokens.len()).
    pub accepted_drafts: Vec<u32>,
    /// Next target greedy token after `y_0 + accepted_drafts`, if generation should continue.
    pub next_pending: Option<u32>,
    /// True if an EOS token matched among `accepted_drafts` or remaining budget hit 0.
    pub hit_stop: bool,
}

/// Lossless greedy verification over `[y_0, d_1, ..., d_r]`.
///
/// Preconditions:
/// - `batch_logits.len() == (draft_tokens.len() + 1) * vocab`
/// - `remaining_after_y0 >= 1`
pub fn verify_greedy_draft_batch(
    draft_tokens: &[u32],
    batch_logits: &[f32],
    vocab: usize,
    eos: &[u32],
    remaining_after_y0: usize,
) -> Result<SpeculativeStepOutcome, String> {
    if vocab == 0 {
        return Err("vocab must be > 0".into());
    }
    let expected = (draft_tokens.len() + 1) * vocab;
    if batch_logits.len() != expected {
        return Err(format!(
            "batch_logits len {} != expected {expected} ((1 + {}) * {vocab})",
            batch_logits.len(),
            draft_tokens.len()
        ));
    }
    if remaining_after_y0 == 0 {
        return Err("remaining_after_y0 must be >= 1".into());
    }

    let mut accepted_drafts = Vec::with_capacity(draft_tokens.len());
    for (i, &draft_tok) in draft_tokens.iter().enumerate() {
        let slot_logits = &batch_logits[i * vocab..(i + 1) * vocab];
        let target_tok = argmax_f32(slot_logits);
        if target_tok != draft_tok {
            return Ok(SpeculativeStepOutcome {
                accepted_drafts,
                next_pending: Some(target_tok),
                hit_stop: false,
            });
        }
        accepted_drafts.push(target_tok);
        if eos.contains(&target_tok) || accepted_drafts.len() >= remaining_after_y0 {
            return Ok(SpeculativeStepOutcome {
                accepted_drafts,
                next_pending: None,
                hit_stop: true,
            });
        }
    }

    // All draft tokens accepted without hitting EOS or budget -> claim bonus token from last slot!
    let bonus_slot = &batch_logits[draft_tokens.len() * vocab..(draft_tokens.len() + 1) * vocab];
    let bonus_tok = argmax_f32(bonus_slot);
    Ok(SpeculativeStepOutcome {
        accepted_drafts,
        next_pending: Some(bonus_tok),
        hit_stop: false,
    })
}

/// Node in a flat speculative prefix tree (`nodes.len() <= 32`, `nodes[0]` is always root `y_0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecTreeNode {
    pub token_id: u32,
    /// Parent slot index (`-1` for root `0`, `0..i as i32` for descendants).
    pub parent: i32,
    /// Sequence offset relative to root (`0` for root `0`, `parent.depth + 1` for children).
    pub depth: usize,
    /// Bitmask over `0..32` where bit `j` is `1` iff slot `j` is on the ancestor path `0..=self`.
    pub ancestor_mask: u32,
}

/// Deduplicated speculative prefix-tree builder (`<= 24..32` nodes) fusing EAGLE-3 / BHDR beams
/// and n-gram / suffix-table candidate sequences into a single batched target verification pass.
#[derive(Debug, Clone)]
pub struct SpecTreeBuilder {
    /// Graft multiple candidate branches and record confidence margins cleanly.
    /// Parallel vector `draft_margins[slot]` stores `(top1_logit - top2_logit)` from drafter (`100.0` for exact suffix lookup).
    pub nodes: Vec<SpecTreeNode>,
    pub draft_margins: Vec<f32>,
    pub max_nodes: usize,
}

impl SpecTreeBuilder {
    pub fn new(root_token: u32, max_nodes: usize) -> Self {
        let cap = max_nodes.clamp(1, 24);
        let mut nodes = Vec::with_capacity(cap);
        let mut draft_margins = Vec::with_capacity(cap);
        nodes.push(SpecTreeNode {
            token_id: root_token,
            parent: -1,
            depth: 0,
            ancestor_mask: 1u32,
        });
        draft_margins.push(0.0);
        Self { nodes, draft_margins, max_nodes: cap }
    }

    pub fn set_slot_margin(&mut self, slot: usize, margin: f32) {
        if slot < self.draft_margins.len() {
            self.draft_margins[slot] = margin;
        }
    }

    pub fn slot_margin(&self, slot: usize) -> f32 {
        self.draft_margins.get(slot).copied().unwrap_or(0.0)
    }

    /// Insert a child `token_id` under `parent_slot`, deduplicating if that exact child already exists.
    /// Returns `Some(slot_idx)` if present or inserted within `max_nodes`, else `None`.
    pub fn insert_child(&mut self, parent_slot: usize, token_id: u32) -> Option<usize> {
        if parent_slot >= self.nodes.len() {
            return None;
        }
        for (idx, n) in self.nodes.iter().enumerate() {
            if n.parent == parent_slot as i32 && n.token_id == token_id {
                return Some(idx);
            }
        }
        if self.nodes.len() >= self.max_nodes {
            return None;
        }
        let idx = self.nodes.len();
        let p = self.nodes[parent_slot];
        self.nodes.push(SpecTreeNode {
            token_id,
            parent: parent_slot as i32,
            depth: p.depth + 1,
            ancestor_mask: p.ancestor_mask | (1u32 << idx),
        });
        self.draft_margins.push(0.0);
        Some(idx)
    }

    /// Graft an entire candidate sequence `[d_1, d_2, ...]` starting under root `0`, stopping at `eos` or `max_nodes`.
    pub fn graft_sequence(&mut self, seq: &[u32], eos: &[u32]) -> usize {
        self.graft_sequence_with_margin(seq, eos, 0.0)
    }

    /// Graft candidate sequence and tag inserted nodes with a drafter confidence margin (max-merged).
    pub fn graft_sequence_with_margin(&mut self, seq: &[u32], eos: &[u32], margin: f32) -> usize {
        let mut cur = 0usize;
        let mut added = 0usize;
        for &tok in seq {
            let prev_len = self.nodes.len();
            match self.insert_child(cur, tok) {
                Some(next_slot) => {
                    if self.nodes.len() > prev_len {
                        added += 1;
                    }
                    if margin > self.slot_margin(next_slot) {
                        self.set_slot_margin(next_slot, margin);
                    }
                    cur = next_slot;
                    if eos.contains(&tok) {
                        break;
                    }
                }
                None => break,
            }
        }
        added
    }

    pub fn depths_and_masks(&self) -> Vec<(usize, u32)> {
        self.nodes.iter().map(|n| (n.depth, n.ancestor_mask)).collect()
    }
}

/// Zero-DRAM O(1) longest-suffix matcher (PLD-style) over active token history.
/// Searches backwards for the longest suffix match of length `2..=6` inside `history[..n-1]`
/// and returns up to `max_draft` continuation tokens immediately following that match.
pub fn prompt_suffix_lookup(history: &[u32], min_match: usize, max_draft: usize) -> Vec<u32> {
    let n = history.len();
    let min_m = min_match.max(2);
    if max_draft == 0 || n < min_m + 1 {
        return Vec::new();
    }
    let max_m = 6usize.min(n - 1);
    for m in (min_m..=max_m).rev() {
        let needle = &history[n - m..n];
        // Scan backwards for most recent prior occurrence of `needle` with at least 1 follower token
        let search_limit = n - m;
        for pos in (0..search_limit).rev() {
            if &history[pos..pos + m] == needle && pos + m < n - m {
                let start = pos + m;
                let end = (start + max_draft).min(n - m);
                if end > start {
                    return history[start..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// Outcome of lossless greedy verification over a flat speculative prefix tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecTreeOutcome {
    pub accepted_drafts: Vec<u32>,
    /// Slot indices `[0, s_1, ..., s_acc]` along the accepted path (to commit into KV cache).
    pub accepted_slots: Vec<usize>,
    pub next_pending: Option<u32>,
    pub hit_stop: bool,
    /// Maximum possible chain depth present in the tree (for acceptance telemetry).
    pub max_tree_depth: usize,
}

/// Lossless greedy verification over a flat speculative prefix tree `nodes` (`nodes[0]` = `y_0`).
/// Applies optional `guard_fn(slot_logits, accepted_so_far)` ONLY along the active winning path
/// so token output is 100% bit-exact with sequential `k=1` greedy decoding!
pub fn verify_greedy_tree_batch<G>(
    nodes: &[SpecTreeNode],
    batch_logits: &mut [f32],
    vocab: usize,
    eos: &[u32],
    remaining_after_y0: usize,
    mut guard_fn: G,
) -> Result<SpecTreeOutcome, String>
where
    G: FnMut(&mut [f32], &[u32]),
{
    if nodes.is_empty() || nodes[0].parent != -1 || nodes[0].depth != 0 {
        return Err("verify_greedy_tree_batch requires non-empty tree rooted at slot 0".into());
    }
    if vocab == 0 || batch_logits.len() != nodes.len() * vocab {
        return Err(format!(
            "verify_greedy_tree_batch logits len {} != {} * {vocab}",
            batch_logits.len(),
            nodes.len()
        ));
    }
    if remaining_after_y0 == 0 {
        return Err("remaining_after_y0 must be >= 1".into());
    }
    let max_tree_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut accepted_drafts: Vec<u32> = Vec::with_capacity(max_tree_depth);
    let mut accepted_slots: Vec<usize> = Vec::with_capacity(1 + max_tree_depth);
    accepted_slots.push(0);
    let mut cur_slot = 0usize;

    loop {
        let slot_logits = &mut batch_logits[cur_slot * vocab..(cur_slot + 1) * vocab];
        guard_fn(slot_logits, &accepted_drafts);
        let target_tok = argmax_f32(slot_logits);

        // Check if any child of `cur_slot` predicted `target_tok`.
        let matched_child = nodes
            .iter()
            .enumerate()
            .skip(cur_slot + 1)
            .find(|&(_, n)| n.parent == cur_slot as i32 && n.token_id == target_tok)
            .map(|(idx, _)| idx);

        match matched_child {
            Some(child_slot) => {
                accepted_drafts.push(target_tok);
                accepted_slots.push(child_slot);
                if eos.contains(&target_tok) || accepted_drafts.len() >= remaining_after_y0 {
                    return Ok(SpecTreeOutcome {
                        accepted_drafts,
                        accepted_slots,
                        next_pending: None,
                        hit_stop: true,
                        max_tree_depth,
                    });
                }
                cur_slot = child_slot;
            }
            None => {
                return Ok(SpecTreeOutcome {
                    accepted_drafts,
                    accepted_slots,
                    next_pending: Some(target_tok),
                    hit_stop: false,
                    max_tree_depth,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot_vocab(vocab: usize, winner: u32) -> Vec<f32> {
        let mut v = vec![-1.0f32; vocab];
        v[winner as usize] = 10.0;
        v
    }

    #[test]
    fn adaptive_draft_controller_scales_with_acceptance() {
        let mut ctrl = AdaptiveDraftController::new(1, 6);
        assert_eq!(ctrl.current_k(), 6);
        for _ in 0..8 {
            ctrl.observe(0, 6);
        }
        assert!(ctrl.current_k() < 6);
        for _ in 0..12 {
            ctrl.observe(ctrl.current_k(), ctrl.current_k());
        }
        assert_eq!(ctrl.current_k(), 6);
    }

    #[test]
    fn fast_ngram_trains_backoff_and_proposes() {
        let mut ngram = FastNGramProposalEngine::new(4);
        ngram.train_tokens(&[1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4]);
        // 3-gram context [1, 2, 3] prefers most frequent/recent 4, then transitions [2,3,4]->1, [3,4,1]->2
        assert_eq!(ngram.propose_sequence(&[9, 1, 2, 3], 3), vec![4, 1, 2]);
        // 2-gram backoff when only [1, 2] matches -> predicts 3 then 4
        assert_eq!(ngram.propose_sequence(&[100, 1, 2], 2), vec![3, 4]);
        // Unseen sequence returns empty
        assert!(ngram.propose_sequence(&[77, 88], 3).is_empty());
        // Online learning works across orders
        ngram.observe_transition(&[7, 9, 8], 6);
        ngram.observe_transition(&[9, 8, 6], 5);
        assert_eq!(ngram.propose_sequence(&[7, 9, 8], 2), vec![6, 5]);
    }

    #[test]
    fn verify_greedy_draft_all_accepted_emits_bonus() {
        let vocab = 8;
        let mut logits = Vec::new();
        logits.extend_from_slice(&one_hot_vocab(vocab, 3));
        logits.extend_from_slice(&one_hot_vocab(vocab, 5));
        logits.extend_from_slice(&one_hot_vocab(vocab, 7)); // bonus
        let res = verify_greedy_draft_batch(&[3, 5], &logits, vocab, &[1], 6).unwrap();
        assert_eq!(res.accepted_drafts, vec![3, 5]);
        assert_eq!(res.next_pending, Some(7));
        assert!(!res.hit_stop);
    }

    #[test]
    fn verify_greedy_draft_partial_accept_rolls_back_and_corrects() {
        let vocab = 8;
        let mut logits = Vec::new();
        logits.extend_from_slice(&one_hot_vocab(vocab, 3)); // matches draft 0
        logits.extend_from_slice(&one_hot_vocab(vocab, 6)); // rejects draft 1 (=5), corrects to 6
        logits.extend_from_slice(&one_hot_vocab(vocab, 2)); // ignored
        let res = verify_greedy_draft_batch(&[3, 5], &logits, vocab, &[1], 6).unwrap();
        assert_eq!(res.accepted_drafts, vec![3]);
        assert_eq!(res.next_pending, Some(6));
        assert!(!res.hit_stop);
    }

    #[test]
    fn verify_greedy_draft_zero_accepted_keeps_base_and_emits_l0() {
        let vocab = 8;
        let mut logits = Vec::new();
        logits.extend_from_slice(&one_hot_vocab(vocab, 4)); // rejects draft 0 (=3)
        logits.extend_from_slice(&one_hot_vocab(vocab, 5));
        let res = verify_greedy_draft_batch(&[3], &logits, vocab, &[1], 6).unwrap();
        assert!(res.accepted_drafts.is_empty());
        assert_eq!(res.next_pending, Some(4));
        assert!(!res.hit_stop);
    }

    #[test]
    fn speculative_toy_kv_loop_matches_sequential_greedy_bit_exact() {
        // Deterministic stateful KV oracle whose logits depend on EVERY KV entry seen so far.
        // If speculative KV rollback is off by even 1 token, the hash diverges permanently!
        const VOCAB: usize = 16;
        const EOS: [u32; 2] = [1, 15];
        fn step_logits(kv: &[u32], next_tok: u32) -> (Vec<u32>, Vec<f32>) {
            let mut new_kv = kv.to_vec();
            new_kv.push(next_tok);
            // Periodic/semi-structured pattern mixed with state hash so n-grams hit AND miss.
            let win = new_kv.len();
            let winner = if win % 7 == 0 {
                // Deliberate surprise break to force speculative rollback + truncation!
                ((new_kv.iter().fold(17u32, |acc, &x| acc.wrapping_mul(31).wrapping_add(x))) % 11) + 2
            } else {
                // Repeatable 4-cycle [2, 3, 4, 5] offset slightly by length
                match next_tok {
                    2 => 3,
                    3 => 4,
                    4 => 5,
                    _ => 2,
                }
            };
            (new_kv, one_hot_vocab(VOCAB, winner))
        }

        let prompt = vec![2u32, 3, 4, 5, 2, 3, 4, 5];
        let max_tokens = 24usize;

        // 1. Sequential greedy baseline (k = 1)
        let mut seq_kv = prompt.clone();
        let (k0, l0) = step_logits(&seq_kv[..seq_kv.len() - 1], *seq_kv.last().unwrap());
        seq_kv = k0;
        let mut seq_logits = l0;
        let mut seq_gen = Vec::new();
        for _ in 0..max_tokens {
            let nxt = argmax_f32(&seq_logits);
            seq_gen.push(nxt);
            if EOS.contains(&nxt) || seq_gen.len() >= max_tokens {
                break;
            }
            let (nk, nl) = step_logits(&seq_kv, nxt);
            seq_kv = nk;
            seq_logits = nl;
        }

        // 2. Speculative draft + verify + KV rollback loop
        let mut spec_kv = prompt.clone();
        let (sk0, sl0) = step_logits(&spec_kv[..spec_kv.len() - 1], *spec_kv.last().unwrap());
        spec_kv = sk0;
        let mut history = prompt.clone();
        let mut ngram = FastNGramProposalEngine::new(4);
        ngram.train_tokens(&history);
        let mut ctrl = AdaptiveDraftController::new(1, 3);
        let mut pending = Some(argmax_f32(&sl0));
        let mut spec_gen = Vec::new();
        let mut total_accepted = 0usize;

        while let Some(y0) = pending.take() {
            spec_gen.push(y0);
            ngram.observe_transition(&history, y0);
            history.push(y0);
            if EOS.contains(&y0) || spec_gen.len() >= max_tokens {
                break;
            }
            let rem = max_tokens - spec_gen.len();
            let draft_budget = ctrl.current_k().min(rem.saturating_sub(1));
            let drafts = if draft_budget > 0 {
                ngram.propose_sequence(&history, draft_budget)
            } else {
                Vec::new()
            };

            let cur_kv_len = spec_kv.len();
            let mut batch = Vec::with_capacity(1 + drafts.len());
            batch.push(y0);
            batch.extend_from_slice(&drafts);

            let mut batch_logits = Vec::with_capacity(batch.len() * VOCAB);
            for &tok in &batch {
                let (nk, nl) = step_logits(&spec_kv, tok);
                spec_kv = nk;
                batch_logits.extend_from_slice(&nl);
            }

            if drafts.is_empty() {
                pending = Some(argmax_f32(&batch_logits));
            } else {
                let outcome = verify_greedy_draft_batch(&drafts, &batch_logits, VOCAB, &EOS, rem).unwrap();
                let acc = outcome.accepted_drafts.len();
                total_accepted += acc;
                ctrl.observe(acc, drafts.len());
                spec_kv.truncate(cur_kv_len + 1 + acc);
                for &t in &outcome.accepted_drafts {
                    spec_gen.push(t);
                    ngram.observe_transition(&history, t);
                    history.push(t);
                }
                pending = if outcome.hit_stop { None } else { outcome.next_pending };
            }
        }

        assert!(total_accepted > 0, "expected n-gram drafter to accept tokens on repeating cycle");
        assert_eq!(spec_gen, seq_gen, "speculative decode diverged from k=1 greedy");
        assert_eq!(spec_kv, seq_kv, "KV history diverged after rollback/truncation");
    }

    #[test]
    fn cngm_seed_roundtrip_cold_draft_and_live_precedence() {
        let mut corpus = FastNGramProposalEngine::new(4);
        corpus.train_tokens(&[10, 20, 30, 40, 50]);
        let bytes = corpus.to_seed_bytes(1);

        let mut live = FastNGramProposalEngine::new(4);
        let n_loaded = live.load_seed_bytes(&bytes).unwrap();
        assert!(n_loaded >= 4);
        // Cold turn-0 proposal immediately succeeds from seed transitions.
        assert_eq!(live.propose_sequence(&[10, 20], 3), vec![30, 40, 50]);

        // Live conversation repetition on [10, 20] -> 99 supersedes static seed (-> 30).
        live.observe_transition(&[10, 20], 99);
        live.observe_transition(&[10, 20], 99);
        assert_eq!(live.propose_sequence(&[10, 20], 1), vec![99]);
    }

    #[test]
    fn cngm_seed_rejects_corrupt_header_and_truncation() {
        let mut engine = FastNGramProposalEngine::new(4);
        assert!(engine.load_seed_bytes(b"BAD!").is_err());
        let valid = FastNGramProposalEngine::new(4).to_seed_bytes(1);
        assert!(engine.load_seed_bytes(&valid).unwrap() == 0);
        let mut bad_magic = valid.clone();
        bad_magic[0] = b'X';
        assert!(engine.load_seed_bytes(&bad_magic).is_err());
        let mut extra = valid;
        extra.push(0);
        assert!(engine.load_seed_bytes(&extra).is_err());
    }

    #[test]
    fn verify_greedy_tree_batch_accepts_alternate_sibling_branch_exact() {
        let mut tb = SpecTreeBuilder::new(100, 12);
        // Primary chain: 100 -> 11 -> 12
        let s1 = tb.insert_child(0, 11).unwrap();
        let _s2 = tb.insert_child(s1, 12).unwrap();
        // Alternate sibling branch under root: 100 -> 21 -> 22 -> 23
        let a1 = tb.insert_child(0, 21).unwrap();
        let a2 = tb.insert_child(a1, 22).unwrap();
        let a3 = tb.insert_child(a2, 23).unwrap();
        // Deduplication check: inserting 21 under root returns existing slot `a1`
        assert_eq!(tb.insert_child(0, 21), Some(a1));

        let vocab = 64usize;
        let n = tb.nodes.len();
        let mut logits = vec![-5.0f32; n * vocab];
        // Target picks alternate branch: slot 0 -> 21, slot a1 -> 22, slot a2 -> 23, slot a3 -> 55 (bonus token!)
        logits[0 * vocab + 21] = 9.0;
        logits[s1 * vocab + 9] = 9.0; // rejected branch shouldn't even be touched by guard_fn!
        logits[a1 * vocab + 22] = 9.0;
        logits[a2 * vocab + 23] = 9.0;
        logits[a3 * vocab + 55] = 9.0;

        let mut guard_calls = Vec::new();
        let out = verify_greedy_tree_batch(&tb.nodes, &mut logits, vocab, &[1], 10, |_lg, acc| {
            guard_calls.push(acc.to_vec());
        })
        .unwrap();

        assert_eq!(out.accepted_drafts, vec![21, 22, 23]);
        assert_eq!(out.accepted_slots, vec![0, a1, a2, a3]);
        assert_eq!(out.next_pending, Some(55));
        assert!(!out.hit_stop);
        assert_eq!(
            guard_calls,
            vec![vec![], vec![21], vec![21, 22], vec![21, 22, 23]],
            "guard_fn must execute strictly along the winning ancestor path"
        );
    }

    #[test]
    fn prompt_suffix_lookup_and_multi_branches_work() {
        let hist = vec![10, 20, 30, 40, 50, 99, 20, 30, 40];
        let hit = prompt_suffix_lookup(&hist, 3, 4);
        assert_eq!(hit, vec![50, 99]);

        let mut ngram = FastNGramProposalEngine::new(4);
        ngram.train_tokens(&[1, 2, 3, 10, 11, 1, 2, 3, 10, 11, 1, 2, 3, 20, 21]);
        let branches = ngram.propose_multi_branches(&[1, 2, 3], 2, 2);
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0], vec![10, 11]);
        assert_eq!(branches[1], vec![20, 21]);

        let mut tb = SpecTreeBuilder::new(9, 12);
        tb.graft_sequence_with_margin(&hit, &[1], 88.0);
        assert_eq!(tb.nodes.len(), 3);
        assert!((tb.slot_margin(1) - 88.0).abs() < 1e-6);
    }
}

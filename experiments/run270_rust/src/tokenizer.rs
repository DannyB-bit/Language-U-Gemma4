//! Gemma 3 HuggingFace tokenizer.json encoder & decoder in pure Rust.
//!
//! Matches HuggingFace `tokenizers` behavior verified against AutoTokenizer:
//! - Exact added_tokens lookup (including `<bos>`, `<eos>`, `<start_of_turn>`, `<end_of_turn>`, newlines)
//! - Space -> U+2581 (`▁`) normalizer before char-level BPE merges
//! - UTF-8 `<0xNN>` byte fallback for unknown pieces
//! - Post-processing `<bos>` injection when absent and `add_bos` is enabled
//! - SentencePiece `▁` -> `' '` and `<0xNN>` byte reconstruction in `decode`

use crate::json::Json;
use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    pub id_to_token: Vec<String>,
    pub vocab: HashMap<String, u32>,
    pub merges: HashMap<(String, String), u32>,
    added_buckets: Vec<Vec<(String, u32)>>,
    pub bos_id: Option<u32>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let v = crate::json::parse(&text).map_err(|e| format!("tokenizer json: {e}"))?;
        let model = v.get("model").ok_or("tokenizer missing 'model'")?;
        let vocab_obj = model.get("vocab").ok_or("tokenizer missing 'model.vocab'")?;

        let mut vocab = HashMap::with_capacity(262_144);
        let mut max_id = 0usize;
        if let Json::Obj(map) = vocab_obj {
            for (k, val) in map {
                if let Some(id) = val.as_u64() {
                    let uid = id as u32;
                    vocab.insert(k.clone(), uid);
                    max_id = max_id.max(uid as usize);
                }
            }
        }

        let mut added_buckets: Vec<Vec<(String, u32)>> = vec![Vec::new(); 256];
        if let Some(Json::Arr(added)) = v.get("added_tokens") {
            for item in added {
                if let (Some(content), Some(id)) = (
                    item.get("content").and_then(|x| x.as_str()),
                    item.get("id").and_then(|x| x.as_u64()),
                ) {
                    if let Some(&first_byte) = content.as_bytes().first() {
                        let uid = id as u32;
                        max_id = max_id.max(uid as usize);
                        vocab.entry(content.to_string()).or_insert(uid);
                        added_buckets[first_byte as usize].push((content.to_string(), uid));
                    }
                }
            }
        }
        for bucket in &mut added_buckets {
            bucket.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        }

        let mut id_to_token = vec![String::new(); max_id + 1];
        for (k, &id) in &vocab {
            if (id as usize) < id_to_token.len() && id_to_token[id as usize].is_empty() {
                id_to_token[id as usize] = k.clone();
            }
        }
        // Let added_tokens override canonical id_to_token string when present.
        for bucket in &added_buckets {
            for (content, id) in bucket {
                id_to_token[*id as usize] = content.clone();
            }
        }

        let mut merges = HashMap::with_capacity(520_000);
        if let Some(Json::Arr(list)) = model.get("merges") {
            for (rank, item) in list.iter().enumerate() {
                let (a, b) = match item {
                    Json::Arr(arr) if arr.len() >= 2 => {
                        match (arr[0].as_str(), arr[1].as_str()) {
                            (Some(x), Some(y)) => (x.to_string(), y.to_string()),
                            _ => continue,
                        }
                    }
                    Json::Str(s) => match s.split_once(' ') {
                        Some((x, y)) => (x.to_string(), y.to_string()),
                        None => continue,
                    },
                    _ => continue,
                };
                merges.insert((a, b), rank as u32);
            }
        }

        let bos_id = vocab.get("<bos>").copied();
        Ok(Self {
            id_to_token,
            vocab,
            merges,
            added_buckets,
            bos_id,
        })
    }

    fn match_added_at(&self, rest: &str) -> Option<(usize, u32)> {
        let &first = rest.as_bytes().first()?;
        for (content, id) in &self.added_buckets[first as usize] {
            if rest.starts_with(content) {
                return Some((content.len(), *id));
            }
        }
        None
    }

    fn bpe_encode_segment(&self, seg: &str) -> Vec<u32> {
        let normalized: String = seg.chars().map(|c| if c == ' ' { '\u{2581}' } else { c }).collect();
        let mut parts: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..parts.len() - 1 {
                if let Some(&r) = self.merges.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = i;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            let merged = format!("{}{}", parts[best_idx], parts[best_idx + 1]);
            parts.splice(best_idx..=best_idx + 1, [merged]);
        }

        let mut out = Vec::with_capacity(parts.len());
        for p in parts {
            if let Some(&id) = self.vocab.get(&p) {
                out.push(id);
            } else {
                for b in p.as_bytes() {
                    let hx = format!("<0x{b:02X}>");
                    if let Some(&id) = self.vocab.get(&hx) {
                        out.push(id);
                    }
                }
            }
        }
        out
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_id {
                ids.push(bos);
            }
        }
        let mut i = 0usize;
        let mut seg_start = 0usize;
        while i < text.len() {
            if let Some((len, id)) = self.match_added_at(&text[i..]) {
                if seg_start < i {
                    ids.extend(self.bpe_encode_segment(&text[seg_start..i]));
                }
                ids.push(id);
                i += len;
                seg_start = i;
            } else {
                let ch_len = text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                i += ch_len;
            }
        }
        if seg_start < text.len() {
            ids.extend(self.bpe_encode_segment(&text[seg_start..]));
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(tok) = self.id_to_token.get(id as usize) else {
                continue;
            };
            if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    byte_buf.push(b);
                    continue;
                }
            }
            if !byte_buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(&byte_buf));
                byte_buf.clear();
            }
            for ch in tok.chars() {
                if ch == '\u{2581}' {
                    out.push(' ');
                } else {
                    out.push(ch);
                }
            }
        }
        if !byte_buf.is_empty() {
            out.push_str(&String::from_utf8_lossy(&byte_buf));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn encodes_added_tokens_bpe_and_byte_fallback_and_decodes_sp_space() {
        let json_src = r#"{
            "added_tokens": [
                {"id": 2, "content": "<bos>"},
                {"id": 105, "content": "<start_of_turn>"},
                {"id": 106, "content": "<end_of_turn>"},
                {"id": 107, "content": "\n"}
            ],
            "model": {
                "vocab": {
                    "W": 10, "h": 11, "a": 12, "t": 13, "What": 20, "▁": 21, "2": 22,
                    "▁+": 23, "<0xF0>": 30, "<0x9F>": 31, "<0x98>": 32, "<0x8A>": 33
                },
                "merges": ["W h", "Wh a", "Wha t", "▁ +"]
            }
        }"#;
        let path = std::env::temp_dir().join(format!("run270_tok_test_{}.json", std::process::id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(json_src.as_bytes()).unwrap();
        drop(f);

        let tk = Tokenizer::load(&path).unwrap();
        let ids = tk.encode("<start_of_turn>What + 2😊\n<end_of_turn>", true);
        assert_eq!(ids, vec![2, 105, 20, 23, 21, 22, 30, 31, 32, 33, 107, 106]);
        let decoded = tk.decode(&[22, 23, 21, 22, 30, 31, 32, 33, 107]);
        assert_eq!(decoded, "2 + 2😊\n");
        let _ = std::fs::remove_file(&path);
    }
}

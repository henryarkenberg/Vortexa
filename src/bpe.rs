//! Byte-level BPE tokenizer, implemented from scratch (GPT-2 style).
//!
//! The base vocabulary is the 256 byte values (each byte is a token id in
//! `0..255`). `train` learns `num_merges` most-frequent adjacent-pair merges;
//! each merge gets a new id `256 + i`. `encode` maps bytes to ids via the
//! merges (applied lowest-rank first), `decode` maps ids back to bytes.
//!
//! This keeps a matching pair of methods so the existing raw-byte fallback
//! stays available: use vocab size 256 when no tokenizer is trained.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The tokenizer: just the ordered list of learned merges. `merges[i]` is
/// `(left, right, new_id)` with `new_id == 256 + i`. Everything else is
/// derived and cached in [`BpeData`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeTokenizer {
    pub(crate) merges: Vec<(u32, u32, u32)>,
}

/// Precomputed lookups derived from the merges (ranks, reverse map, and a
/// byte-expansion table for decoding).
pub struct BpeData {
    ranks: HashMap<(u32, u32), usize>,
    expand: Vec<Vec<u8>>,
}

/// Round-trip exact separators: whitespace bytes are never merged.
fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// Greedy BPE encoding of a single (short) non-whitespace word.
fn encode_word(word: &[u8], data: &BpeData, out: &mut Vec<u32>) {
    let mut seq: Vec<u32> = word.iter().map(|&b| b as u32).collect();
    loop {
        if seq.len() < 2 {
            break;
        }
        let mut best_pos = usize::MAX;
        let mut best_rank = usize::MAX;
        for i in 0..seq.len() - 1 {
            if let Some(&rank) = data.ranks.get(&(seq[i], seq[i + 1])) {
                if rank < best_rank {
                    best_rank = rank;
                    best_pos = i;
                }
            }
        }
        if best_pos == usize::MAX {
            break;
        }
        seq.splice(best_pos..best_pos + 2, [256 + best_rank as u32]);
    }
    out.extend(seq);
}

impl BpeTokenizer {
    /// Learn `num_merges` merges from `bytes`. Fewer may be learned if the
    /// corpus is too small for more merges.
    pub fn train(bytes: &[u8], num_merges: usize) -> Self {
        let mut seq: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
        let mut merges: Vec<(u32, u32, u32)> = Vec::with_capacity(num_merges);
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();

        for _ in 0..num_merges {
            counts.clear();
            for i in 0..seq.len().saturating_sub(1) {
                *counts.entry((seq[i], seq[i + 1])).or_insert(0) += 1;
            }
            let Some((&pair, &cnt)) = counts.iter().max_by_key(|(_, c)| **c) else {
                break;
            };
            if cnt < 2 {
                break;
            }
            let new_id = 256 + merges.len() as u32;
            merges.push((pair.0, pair.1, new_id));

            let mut out = Vec::with_capacity(seq.len());
            let mut i = 0;
            while i < seq.len() {
                if i + 1 < seq.len() && seq[i] == pair.0 && seq[i + 1] == pair.1 {
                    out.push(new_id);
                    i += 2;
                } else {
                    out.push(seq[i]);
                    i += 1;
                }
            }
            seq = out;
        }

        Self { merges }
    }

    /// Number of byte-base tokens plus learned merges.
    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    pub fn data(&self) -> BpeData {
        let mut ranks: HashMap<(u32, u32), usize> = HashMap::with_capacity(self.merges.len());
        let mut expand: Vec<Vec<u8>> = Vec::with_capacity(self.vocab_size());
        for i in 0..256u32 {
            expand.push(vec![i as u8]);
        }
        for (i, (a, b, _)) in self.merges.iter().enumerate() {
            ranks.insert((*a, *b), i);
            let mut e = expand[*a as usize].clone();
            e.extend_from_slice(&expand[*b as usize]);
            expand.push(e);
        }
        BpeData { ranks, expand }
    }

    /// Encode raw bytes into token ids.
    ///
    /// Linear in the input: text is split into whitespace-delimited "words"
    /// and each word is encoded greedily by lowest-rank merge. Whitespace
    /// bytes pass through as their own (byte-valued) ids, so decoding is
    /// exact.
    pub fn encode(&self, bytes: &[u8], data: &BpeData) -> Vec<u32> {
        let mut out = Vec::with_capacity(bytes.len());
        let mut word: Vec<u8> = Vec::new();
        for &b in bytes {
            if is_whitespace(b) {
                if !word.is_empty() {
                    encode_word(&word, data, &mut out);
                    word.clear();
                }
                out.push(b as u32);
            } else {
                word.push(b);
            }
        }
        if !word.is_empty() {
            encode_word(&word, data, &mut out);
        }
        out
    }

    /// Decode token ids back into bytes.
    pub fn decode(&self, ids: &[u32], data: &BpeData) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(b) = data.expand.get(id as usize) {
                out.extend_from_slice(b);
            }
        }
        out
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serializing BPE tokenizer")
    }

    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("parsing BPE tokenizer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_merges_and_roundtrip() {
        // Repeated text so merges are learned.
        let text = b"the quick brown fox jumps over the lazy dog the quick brown fox"
            .repeat(200);
        let tok = BpeTokenizer::train(&text, 512);
        let data = tok.data();

        assert!(tok.num_merges() > 0, "should learn merges");
        assert_eq!(tok.vocab_size(), 256 + tok.num_merges());

        // Encode then decode must round-trip to the original bytes.
        let ids = tok.encode(&text, &data);
        assert!(ids.len() < text.len(), "BPE should compress");
        let decoded = tok.decode(&ids, &data);
        assert_eq!(decoded, text, "decode(encode(x)) != x");
    }

    #[test]
    fn serde_roundtrip_preserves_vocab() {
        let text = b"aaaa bbbb cccc aaaa bbbb cccc".repeat(50);
        let tok = BpeTokenizer::train(&text, 64);
        let json = tok.to_json().unwrap();
        let tok2 = BpeTokenizer::from_json(&json).unwrap();
        assert_eq!(tok.num_merges(), tok2.num_merges());
        assert_eq!(tok.vocab_size(), tok2.vocab_size());
    }
}

//! Dataset of token ids. In byte mode these ids are raw bytes (`0..255`); in
//! BPE mode they are the 256 byte tokens plus learned merge ids. Everything
//! downstream samples windows of token ids and predicts next tokens.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use rand::Rng;

pub struct ByteDataset {
    tokens: Vec<u32>,
}

impl ByteDataset {
    pub fn from_tokens(tokens: Vec<u32>) -> Result<Self> {
        anyhow::ensure!(
            tokens.len() > 1024,
            "dataset too small ({} tokens); need at least a few thousand",
            tokens.len()
        );
        Ok(Self { tokens })
    }

    /// Byte mode: read a file, tokens are just the bytes.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = Self::read_bytes(path)?;
        Self::from_tokens(bytes.iter().map(|&b| b as u32).collect())
    }

    pub fn read_bytes(path: &std::path::Path) -> Result<Vec<u8>> {
        std::fs::read(path).with_context(|| format!("reading dataset {}", path.display()))
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Split off a contiguous tail for validation. Returns `(train, val)`.
    pub fn split_tail(&self, val_frac: f64, min_val_tokens: usize) -> (Self, Self) {
        let val_len = (((self.tokens.len() as f64) * val_frac) as usize).max(min_val_tokens);
        let cut = self.tokens.len().saturating_sub(val_len);
        (
            Self {
                tokens: self.tokens[..cut].to_vec(),
            },
            Self {
                tokens: self.tokens[cut..].to_vec(),
            },
        )
    }

    /// Sample a batch of random windows: `[batch, seq_len]` u32 token ids.
    pub fn random_batch(
        &self,
        rng: &mut impl Rng,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        assert!(
            self.tokens.len() > seq_len,
            "dataset too small for seq_len {}",
            seq_len
        );
        let max_start = (self.tokens.len() - seq_len - 1) as u64;
        let mut xs = Vec::with_capacity(batch_size * seq_len);
        let mut ys = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            let start = rng.gen_range(0..=max_start) as usize;
            let chunk = &self.tokens[start..start + seq_len + 1];
            xs.extend_from_slice(&chunk[..seq_len]);
            ys.extend_from_slice(&chunk[1..]);
        }
        Ok((
            Tensor::from_vec(xs, (batch_size, seq_len), device)?,
            Tensor::from_vec(ys, (batch_size, seq_len), device)?,
        ))
    }
}

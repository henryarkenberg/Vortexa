//! Architecture configuration for Vortexa.

use serde::{Deserialize, Serialize};

/// Architecture hyperparameters of the model.
///
/// The retention decay rates are derived from `decay_min`/`decay_max`:
/// head `i` of `n` gets a per-step survival gap that is geometrically
/// interpolated between the two, so short-memory and long-memory heads
/// coexist (multi-scale retention).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    /// Context length used during training. Recurrent generation is not
    /// limited by this value.
    pub max_seq_len: usize,
    /// Decay (per-step survival) of the shortest-memory head.
    #[serde(default = "default_decay_min")]
    pub decay_min: f64,
    /// Decay of the longest-memory head.
    #[serde(default = "default_decay_max")]
    pub decay_max: f64,
    /// Chunk size for chunkwise recurrent training. Growth is O(T·chunk_len)
    /// instead of the parallel form's O(T²); 0 disables and uses the parallel
    /// form directly.
    #[serde(default = "default_chunk_len")]
    pub chunk_len: usize,
    /// Tokenization mode: `"bytes"` (vocab 256) or `"bpe"`.
    #[serde(default = "default_tokenizer")]
    pub tokenizer: String,
    /// Number of BPE merges to learn (used when `tokenizer == "bpe"`).
    #[serde(default = "default_num_merges")]
    pub num_merges: usize,
}

fn default_decay_min() -> f64 {
    0.90
}

fn default_decay_max() -> f64 {
    0.995
}

fn default_chunk_len() -> usize {
    64
}

fn default_tokenizer() -> String {
    "bytes".to_string()
}

fn default_num_merges() -> usize {
    512
}

impl Default for Config {
    fn default() -> Self {
        Self::tiny()
    }
}

impl Config {
    /// The recommended first configuration (~460k parameters).
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            head_dim: 32,
            ffn_dim: 512,
            max_seq_len: 256,
            decay_min: default_decay_min(),
            decay_max: default_decay_max(),
            chunk_len: default_chunk_len(),
            tokenizer: default_tokenizer(),
            num_merges: default_num_merges(),
        }
    }

    /// A larger configuration that still fits a laptop CPU (~2.8M params).
    /// 4 layers, d_model 256, ffn 512 SwiGLU.
    pub fn larger() -> Self {
        Self {
            vocab_size: 256,
            d_model: 256,
            num_layers: 4,
            num_heads: 8,
            head_dim: 32,
            ffn_dim: 512,
            max_seq_len: 256,
            decay_min: default_decay_min(),
            decay_max: default_decay_max(),
            chunk_len: default_chunk_len(),
            tokenizer: default_tokenizer(),
            num_merges: default_num_merges(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.vocab_size >= 256,
            "vocab_size must be at least the 256 byte tokens (got {})",
            self.vocab_size
        );
        anyhow::ensure!(
            self.vocab_size == 256 || (self.tokenizer == "bpe"),
            "vocab_size > 256 requires the BPE tokenizer"
        );
        anyhow::ensure!(self.num_layers >= 1, "need at least one layer");
        anyhow::ensure!(self.num_heads >= 1, "need at least one head");
        anyhow::ensure!(
            self.d_model == self.num_heads * self.head_dim,
            "d_model ({}) must equal num_heads ({}) * head_dim ({})",
            self.d_model,
            self.num_heads,
            self.head_dim
        );
        anyhow::ensure!(self.ffn_dim >= 1 && self.max_seq_len >= 2, "ffn_dim and max_seq_len must be sane");
        anyhow::ensure!(
            self.decay_min > 0.0
                && self.decay_min <= self.decay_max
                && self.decay_max < 1.0,
            "decays must satisfy 0 < decay_min <= decay_max < 1"
        );
        Ok(())
    }

    /// Per-head *initial* decay rates. Head 0 forbets fastest, the last head
    /// keeps information longest; these seed the learnable per-head decays.
    /// The interpolation is geometric over the "forgetting gap" `(1 - gamma)`
    /// so heads spread across time scales. After construction each head's
    /// decay is a trainable scalar in `(0, 1)`.
    pub fn decays(&self) -> Vec<f64> {
        let n = self.num_heads;
        let hi_gap = 1.0 - self.decay_min;
        let lo_gap = 1.0 - self.decay_max;
        (0..n)
            .map(|i| {
                if n == 1 {
                    return self.decay_max;
                }
                let frac = i as f64 / (n - 1) as f64;
                let gap = hi_gap * (lo_gap / hi_gap).powf(frac);
                1.0 - gap
            })
            .collect()
    }
}

//! The full Vortexa model: byte embedding -> stacked RetNet blocks ->
//! final RMSNorm -> LM head. There is no attention anywhere.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use candle_nn::RmsNorm;

use crate::config::Config;
use crate::retention::{MultiScaleRetention, RetentionState};

/// SwiGLU feed-forward network: `out = W_o(gate(x) * silu(up(x)))`.
///
/// Two parallel projections `gate`/`up` (both `d_model -> ffn_dim`) are
/// combined with a SiLU gate before a single output projection back down.
#[derive(Debug)]
pub struct FeedForward {
    fc_gate: Linear,
    fc_up: Linear,
    fc_out: Linear,
}

impl FeedForward {
    pub fn new(vs: VarBuilder, d_model: usize, ffn_dim: usize) -> Result<Self> {
        Ok(Self {
            fc_gate: candle_nn::linear_no_bias(d_model, ffn_dim, vs.pp("fc_gate"))?,
            fc_up: candle_nn::linear_no_bias(d_model, ffn_dim, vs.pp("fc_up"))?,
            fc_out: candle_nn::linear_no_bias(ffn_dim, d_model, vs.pp("fc_out"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up = self.fc_up.forward(x)?.silu()?;
        let gated = self.fc_gate.forward(x)?.mul(&up)?;
        self.fc_out.forward(&gated)
    }
}

/// Pre-normalized residual RetNet block:
///
/// ```text
/// x + Retention(RMSNorm(x))
/// x + FFN(RMSNorm(x))
/// ```
#[derive(Debug)]
pub struct RetNetBlock {
    norm1: RmsNorm,
    retention: MultiScaleRetention,
    norm2: RmsNorm,
    ffn: FeedForward,
    chunk_len: usize,
}

impl RetNetBlock {
    pub fn new(cfg: &Config, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: candle_nn::rms_norm(cfg.d_model, 1e-5, vs.pp("norm1"))?,
            retention: MultiScaleRetention::new(
                vs.pp("retention"),
                cfg.d_model,
                cfg.num_heads,
                cfg.head_dim,
                &cfg.decays(),
            )?,
            norm2: candle_nn::rms_norm(cfg.d_model, 1e-5, vs.pp("norm2"))?,
            ffn: FeedForward::new(vs.pp("ffn"), cfg.d_model, cfg.ffn_dim)?,
            chunk_len: cfg.chunk_len,
        })
    }

    /// Training path over a whole sequence (chunkwise recurrent retention).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.retention.forward_chunkwise(&self.norm1.forward(x)?, self.chunk_len)?)?;
        let x = x.add(&self.ffn.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
    /// Recurrent generation path for a single token.
    pub fn forward_step(
        &self,
        x: &Tensor,
        states: &mut [RetentionState],
    ) -> Result<Tensor> {
        let x = x.add(&self.retention.forward_step(&self.norm1.forward(x)?, states)?)?;
        let x = x.add(&self.ffn.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

/// Retention states of every layer (each holding one state per head).
#[derive(Debug)]
pub struct ModelStates {
    pub layers: Vec<Vec<RetentionState>>,
}

impl ModelStates {
    pub fn zeros(cfg: &Config, batch: usize, device: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let mut heads = Vec::with_capacity(cfg.num_heads);
            for _ in 0..cfg.num_heads {
                heads.push(RetentionState::zeros(batch, cfg.head_dim, device)?);
            }
            layers.push(heads);
        }
        Ok(Self { layers })
    }
}

/// Vortexa itself.
#[derive(Debug)]
pub struct Vortexa {
    cfg: Config,
    embedding: Embedding,
    blocks: Vec<RetNetBlock>,
    final_norm: RmsNorm,
    lm_head: Linear,
}

impl Vortexa {
    pub fn new(cfg: Config, vs: VarBuilder) -> Result<Self> {
        let embedding =
            candle_nn::embedding(cfg.vocab_size, cfg.d_model, vs.pp("embedding"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(RetNetBlock::new(&cfg, vs.pp(format!("blocks.{i}")))?);
        }
        let final_norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vs.pp("final_norm"))?;
        let lm_head = candle_nn::linear_no_bias(
            cfg.d_model,
            cfg.vocab_size,
            vs.pp("lm_head"),
        )?;
        Ok(Self {
            cfg,
            embedding,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Full-sequence pass. `ids`: `[B, T]` (u32) -> logits `[B, T, vocab]`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let mut x = self.embedding.forward(ids)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.lm_head.forward(&self.final_norm.forward(&x)?)
    }

    /// Recurrent pass for one token. `ids`: `[B, 1]` -> logits `[B, 1, vocab]`.
    pub fn forward_step(&self, ids: &Tensor, states: &mut ModelStates) -> Result<Tensor> {
        let mut x = self.embedding.forward(ids)?;
        for (block, layer_states) in self.blocks.iter().zip(states.layers.iter_mut()) {
            x = block.forward_step(&x, layer_states)?;
        }
        self.lm_head.forward(&self.final_norm.forward(&x)?)
    }

    pub fn new_states(&self, batch: usize, device: &Device) -> Result<ModelStates> {
        ModelStates::zeros(&self.cfg, batch, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use candle_core::DType;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Guide test 4 at the whole-model level: the training-style full
    /// sequence pass and the token-by-token recurrent pass must agree.
    #[test]
    fn full_model_sequence_matches_recurrent() {
        let cfg = Config {
            vocab_size: 256,
            d_model: 16,
            num_layers: 2,
            num_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            max_seq_len: 16,
            decay_min: 0.90,
            decay_max: 0.995,
            chunk_len: 4,
            tokenizer: "bytes".into(),
            num_merges: 512,
        };
        // A fresh VarMap auto-initializes weights randomly, so this test
        // exercises real (nonzero) weights.
        let varmap = candle_nn::VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &cpu());
        let model = Vortexa::new(cfg, vs).unwrap();

        let ids: Vec<u32> = (0..2 * 10).map(|i| (i * 37 % 256) as u32).collect();
        let ids = Tensor::from_vec(ids, (2, 10), &cpu()).unwrap();

        let seq = model.forward(&ids).unwrap();

        let mut states = model.new_states(2, &cpu()).unwrap();
        let mut steps: Vec<Tensor> = Vec::with_capacity(10);
        for t in 0..10 {
            steps.push(
                model
                    .forward_step(&ids.narrow(1, t, 1).unwrap(), &mut states)
                    .unwrap(),
            );
        }
        let refs: Vec<&Tensor> = steps.iter().collect();
        let rec = Tensor::cat(&refs, 1).unwrap();

        assert_eq!(seq.dims3().unwrap(), (2, 10, 256));
        assert_eq!(rec.dims3().unwrap(), (2, 10, 256));

        let a = seq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = rec.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-3, "full-model modes diverged: max diff {diff}");
    }

    /// The initial loss of an untrained byte-level model should be close
    /// to ln(256) ~= 5.545.
    #[test]
    fn untrained_loss_is_near_ln_256() {
        let cfg = Config {
            d_model: 16,
            num_layers: 1,
            num_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            ..Config::default()
        };
        let varmap = candle_nn::VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &cpu());
        let model = Vortexa::new(cfg, vs).unwrap();

        let ids = Tensor::from_vec(vec![7u32; 8], (2, 4), &cpu()).unwrap();
        let logits = model.forward(&ids).unwrap();
        let mean_logit: f32 = logits
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        // Random tiny model => logits should be small but not exactly zero.
        assert!(mean_logit < 20.0, "logits exploded: {mean_logit}");
    }
}

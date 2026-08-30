//! Deterministic evaluation harness: computes per-token cross-entropy and
//! perplexity on a held-out slice, so different configs / tokenizers can be
//! compared objectively (independent of machine noise while training).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::bpe::{BpeData, BpeTokenizer};
use crate::config::Config;
use crate::data::ByteDataset;
use crate::model::Vortexa;
use crate::train::{cross_entropy, resolve_checkpoint_paths};

pub struct EvalArgs {
    /// Checkpoint directory or model.safetensors file.
    pub checkpoint: PathBuf,
    /// Corpus to evaluate (the same file used for training).
    pub data: PathBuf,
    /// Window length for the sequential eval pass.
    pub seq_len: usize,
    /// Fraction of the corpus held out as the validation slice.
    pub val_frac: f64,
    /// Backend device preference (`auto`, `cpu`, `cuda`, `metal`).
    pub device: String,
}

pub fn run(args: EvalArgs) -> Result<()> {
    let device = crate::device::pick_device(&args.device)?;
    println!("device: {}", crate::device::describe(&device));
    let (weights_path, cfg_path) = resolve_checkpoint_paths(&args.checkpoint);
    let cfg: Config = serde_json::from_str(
        &std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?,
    )
    .with_context(|| format!("parsing {}", cfg_path.display()))?;
    cfg.validate()?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Vortexa::new(cfg.clone(), vs)?;
    varmap.load(weights_path.as_path())?;

    // Load the tokenizer (byte mode if the checkpoint has none).
    let cfg_dir = cfg_path.parent().unwrap_or_else(|| Path::new("."));
    let tokenizer = std::fs::read_to_string(cfg_dir.join("bpe.json"))
        .ok()
        .and_then(|s| BpeTokenizer::from_json(&s).ok());
    let tok_data: Option<BpeData> = tokenizer.as_ref().map(|t| t.data());

    let raw = ByteDataset::read_bytes(&args.data)?;
    let tokens: Vec<u32> = match (&tokenizer, &tok_data) {
        (Some(t), Some(d)) => t.encode(&raw, d),
        _ => raw.iter().map(|&b| b as u32).collect(),
    };

    let n = tokens.len();
    let seq_len = args.seq_len.max(1).min(n.saturating_sub(1));
    let val_len = ((n as f64 * args.val_frac) as usize)
        .clamp(seq_len + 1, n.saturating_sub(seq_len + 1));
    let cut = n - val_len;
    let train_slice = &tokens[..cut];
    let val_slice = &tokens[cut..];

    let (train_sum, train_cnt) = eval_slice(&model, train_slice, seq_len, &device)?;
    let (val_sum, val_cnt) = eval_slice(&model, val_slice, seq_len, &device)?;

    println!(
        "Vortexa: {} layers x {} heads@{}d, d_model {}, ffn {}, vocab {} | {}",
        cfg.num_layers,
        cfg.num_heads,
        cfg.head_dim,
        cfg.d_model,
        cfg.ffn_dim,
        cfg.vocab_size,
        if tokenizer.is_some() { "BPE" } else { "bytes" }
    );
    println!(
        "tokens {} (train slice {} | val slice {}) | window {}",
        n,
        train_cnt,
        val_cnt,
        seq_len
    );
    if train_cnt >= 2 {
        let nll = train_sum / train_cnt as f64;
        println!("train  nats/token {nll:.4}   perplexity {:.3}", nll.exp());
    }
    if val_cnt >= 2 {
        let nll = val_sum / val_cnt as f64;
        println!("val    nats/token {nll:.4}   perplexity {:.3}", nll.exp());
    }
    Ok(())
}

/// Sequential, non-overlapping evaluation pass. Returns `(nll_sum, token_count)`.
fn eval_slice(
    model: &Vortexa,
    tokens: &[u32],
    seq_len: usize,
    device: &Device,
) -> Result<(f64, usize)> {
    let mut nll_sum = 0.0f64;
    let mut count = 0usize;
    let mut start = 0;
    while start + seq_len < tokens.len() {
        let x = &tokens[start..start + seq_len];
        let y = &tokens[start + 1..start + seq_len + 1];
        let xt = Tensor::from_vec(x.to_vec(), (1, seq_len), device)?;
        let yt = Tensor::from_vec(y.to_vec(), (1, seq_len), device)?;
        let logits = model.forward(&xt)?;
        let loss = cross_entropy(&logits, &yt)?;
        nll_sum += loss.to_scalar::<f32>()? as f64 * seq_len as f64;
        count += seq_len;
        start += seq_len;
    }
    Ok((nll_sum, count))
}

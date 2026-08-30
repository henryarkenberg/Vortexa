//! Training loop: random byte batches -> full-sequence forward ->
//! cross-entropy -> AdamW step, with periodic validation + checkpoints.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use rand::{rngs::StdRng, SeedableRng};

use crate::bpe::BpeTokenizer;

/// Live progress reported during training, shared with a TUI front end.
#[derive(Clone, Copy, Debug)]
pub struct TrainProgress {
    pub step: usize,
    pub total: usize,
    pub loss: f32,
    pub tok_s: f64,
    pub done: bool,
}

impl Default for TrainProgress {
    fn default() -> Self {
        Self { step: 0, total: 0, loss: 0.0, tok_s: 0.0, done: false }
    }
}
use crate::config::Config;
use crate::data::ByteDataset;
use crate::model::Vortexa;

pub struct TrainArgs {
    pub data: PathBuf,
    pub out_dir: PathBuf,
    pub steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub lr: f64,
    pub log_every: usize,
    pub val_every: usize,
    pub val_batches: usize,
    pub save_every: usize,
    pub val_frac: f64,
    pub seed: u64,
    /// Optional checkpoint (dir or safetensors file) to continue from.
    pub resume: Option<PathBuf>,
    /// Linear LR warmup steps before cosine decay starts.
    pub warmup_steps: usize,
    /// Global gradient-norm clip (0 disables).
    pub grad_clip: f64,
    /// Backend device preference (`auto`, `cpu`, `cuda`, `metal`).
    pub device: String,
    pub config: Config,
}

/// Linear warmup to `peak`, then cosine decay down to 10% of peak.
fn scheduled_lr(step: usize, total: usize, warmup: usize, peak: f64) -> f64 {
    let warmup = warmup.min(total.saturating_sub(1)).max(1);
    if step <= warmup {
        return peak * step as f64 / warmup as f64;
    }
    let progress = ((step - warmup) as f64 / (total - warmup).max(1) as f64).min(1.0);
    let floor = peak * 0.1;
    floor + 0.5 * (peak - floor) * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Global-norm gradient clipping + optimizer step.
///
/// Equivalent to `backward_step` when the norm is already within `clip`.
fn clipped_step(
    opt: &mut candle_nn::AdamW,
    loss: &Tensor,
    clip: f64,
) -> candle_core::Result<()> {
    let mut grads = loss.backward()?;
    let ids: Vec<_> = grads.get_ids().copied().collect();

    let mut total_sq = 0.0f64;
    for id in &ids {
        if let Some(g) = grads.get_id(*id) {
            let s: f32 = g.sqr()?.sum_all()?.to_scalar()?;
            total_sq += s as f64;
        }
    }
    let norm = total_sq.sqrt();
    if norm.is_finite() && norm > clip {
        let scale = (clip / norm) as f32;
        for id in ids {
            if let Some(g) = grads.get_id(id) {
                let scaled = g.affine(scale as f64, 0.0)?;
                grads.insert_id(id, scaled);
            }
        }
    }
    opt.step(&grads)
}

/// Map a checkpoint argument to `(weights, config_json)` paths.
pub fn resolve_checkpoint_paths(p: &Path) -> (PathBuf, PathBuf) {
    if p.is_dir() {
        (
            p.join("model.safetensors"),
            p.join("model_config.json"),
        )
    } else {
        let dir = p.parent().unwrap_or_else(|| Path::new("."));
        (p.to_path_buf(), dir.join("model_config.json"))
    }
}

/// A progress bar styled with indicatif, driven by `step` position and a
/// `loss / tok/s` message. Log lines are emitted through [`ProgressBar::log`]
/// so they don't collide with the bar, and a tidy `training.log` is written
/// to stdout as a plain side channel.
pub struct TrainBar {
    bar: indicatif::ProgressBar,
    window: Instant,
    window_tokens: u64,
    ema_tps: f64,
}

impl TrainBar {
    pub fn new(total: usize) -> Self {
        let bar = indicatif::ProgressBar::new(total as u64);
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "  {bar:24.cyan/blue} {pos:>6}/{len} ({percent:>3}%) {msg} eta {eta}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        Self {
            bar,
            window: Instant::now(),
            window_tokens: 0,
            ema_tps: 0.0,
        }
    }

    /// Advance to `step`; `bar_msg` is the live right-hand status text.
    pub fn update(&mut self, step: usize, tokens_this_step: u64, bar_msg: &str) {
        self.window_tokens += tokens_this_step;
        let dt = self.window.elapsed().as_secs_f64();
        if dt > 0.5 {
            let inst = self.window_tokens as f64 / dt;
            self.ema_tps = if self.ema_tps == 0.0 { inst } else { self.ema_tps * 0.6 + inst * 0.4 };
            self.window = Instant::now();
            self.window_tokens = 0;
        }
        self.bar.set_position(step as u64);
        self.bar.set_message(bar_msg.to_string());
    }

    /// Current tok/s (for composing status text).
    pub fn tps(&self) -> f64 {
        self.ema_tps
    }

    /// Print a log line above the bar (indicatif-managed, goes to stderr).
    pub fn log(&self, msg: impl AsRef<str>) {
        self.bar.println(msg.as_ref());
    }

    pub fn finish(&self) {
        self.bar.finish_and_clear();
    }
}

/// Run a full training loop, drawing the indicatif bar to the terminal.
pub fn run(args: TrainArgs) -> Result<()> {
    run_inner(args, None, Arc::new(AtomicBool::new(false)))
}

/// Run a full training loop while reporting progress to the TUI. The loop
/// stops early if `cancel` is set (checked each step).
pub fn run_with_progress(
    args: TrainArgs,
    progress: Arc<Mutex<TrainProgress>>,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    run_inner(args, Some(progress), cancel)
}

fn run_inner(
    args: TrainArgs,
    progress: Option<Arc<Mutex<TrainProgress>>>,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    args.config.validate()?;
    anyhow::ensure!(
        args.seq_len <= args.config.max_seq_len,
        "seq_len {} exceeds max_seq_len {}",
        args.seq_len,
        args.config.max_seq_len
    );

    let device = crate::device::pick_device(&args.device)?;
    let device_name = crate::device::describe(&device);
    // When a TUI is drawing the progress itself, keep stdout silent so the
    // alternate screen is not corrupted by stray prints.
    let verbose = progress.is_none();
    if verbose {
        println!("device: {device_name}");
    }

    // Build the tokenizer if requested, then convert raw bytes to token ids.
    let mut config = args.config.clone();
    let tokenizer = if config.tokenizer == "bpe" {
        // Prefer a tokenizer already saved with the checkpoint, so resuming
        // is byte-for-byte consistent with the weights (re-training BPE can
        // tie-break differently on equal-frequency pairs). Otherwise train a
        // fresh one.
        let cached = args.resume.as_ref().and_then(|r| {
            let (_, cfg_path) = resolve_checkpoint_paths(r);
            let dir = cfg_path.parent().unwrap_or_else(|| Path::new("."));
            std::fs::read_to_string(dir.join("bpe.json"))
                .ok()
                .and_then(|s| BpeTokenizer::from_json(&s).ok())
        });
        let tok = match cached {
            Some(tok) => {
                if verbose {
                    println!(
                        "reusing BPE tokenizer: {} merges -> vocab {}",
                        tok.num_merges(),
                        tok.vocab_size()
                    );
                }
                tok
            }
            None => {
                let raw = ByteDataset::read_bytes(&args.data)?;
                let tok = BpeTokenizer::train(&raw, config.num_merges);
                if verbose {
                    println!(
                        "BPE tokenizer: {} merges -> vocab {}",
                        tok.num_merges(),
                        tok.vocab_size()
                    );
                }
                tok
            }
        };
        config.vocab_size = tok.vocab_size();
        Some(tok)
    } else {
        None
    };

    let dataset = match &tokenizer {
        Some(tok) => {
            let raw = ByteDataset::read_bytes(&args.data)?;
            let data = tok.data();
            ByteDataset::from_tokens(tok.encode(&raw, &data))?
        }
        None => ByteDataset::load(&args.data)?,
    };

    let (train_ds, val_ds) = dataset.split_tail(args.val_frac, 4096);
    anyhow::ensure!(
        train_ds.len() > args.seq_len + 1 && val_ds.len() > args.seq_len + 1,
        "dataset too small: train {} / val {} tokens for seq_len {}",
        train_ds.len(),
        val_ds.len(),
        args.seq_len
    );
    if verbose {
        println!(
            "dataset: {} total | train {} | val {} tokens{}",
            train_ds.len() + val_ds.len(),
            train_ds.len(),
            val_ds.len(),
            if config.tokenizer == "bpe" { " (BPE)" } else { " (bytes)" }
        );
    }

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Vortexa::new(config.clone(), vs)?;

    if let Some(resume) = &args.resume {
        let (weights, _) = resolve_checkpoint_paths(resume);
        varmap
            .load(weights.as_path())
            .with_context(|| format!("resuming weights from {}", weights.display()))?;
        if verbose {
            println!("resumed weights from {}", weights.display());
        }
    }

    let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    if verbose {
        println!(
            "Vortexa: {} layers x {} heads@{}d, d_model {}, ffn {}, vocab {}",
            config.num_layers,
            config.num_heads,
            config.head_dim,
            config.d_model,
            config.ffn_dim,
            config.vocab_size
        );
    }
    if verbose {
        println!(
            "parameters: {n_params} ({:.2}M) | peak lr {} | batch {} x seq {}",
            n_params as f64 / 1e6,
            args.lr,
            args.batch_size,
            args.seq_len
        );
        println!(
            "schedule: {} warmup steps, cosine decay to {:.0}%, grad clip {}",
            args.warmup_steps,
            10.0,
            if args.grad_clip > 0.0 {
                format!("{:.2}", args.grad_clip)
            } else {
                "off".into()
            }
        );
    }

    let mut opt = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: args.lr,
            ..Default::default()
        },
    )?;

    let mut rng = StdRng::seed_from_u64(args.seed);
    let start = Instant::now();
    let tokens_per_step = (args.batch_size * args.seq_len) as u64;
    let mut window_start = Instant::now();
    let mut window_tokens: u64 = 0;
    let mut ema_tok_s: f64 = 0.0;
    let mut bar = (progress.is_none()).then(|| TrainBar::new(args.steps));

    for step in 1..=args.steps {
        if cancel.load(Ordering::Relaxed) {
            if verbose {
                println!("training cancelled");
            }
            break;
        }

        let (x, y) =
            train_ds.random_batch(&mut rng, args.batch_size, args.seq_len, &device)?;
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;

        opt.set_learning_rate(scheduled_lr(step, args.steps, args.warmup_steps, args.lr));
        if args.grad_clip > 0.0 {
            clipped_step(&mut opt, &loss, args.grad_clip)?;
        } else {
            opt.backward_step(&loss)?;
        }

        let loss_now: f32 = loss.to_scalar()?;

        // Tokens/s (windowed EMA), shared by both the bar and the TUI.
        window_tokens += tokens_per_step;
        let dt = window_start.elapsed().as_secs_f64();
        if dt >= 1.0 {
            let window_tps = window_tokens as f64 / dt;
            ema_tok_s = if ema_tok_s == 0.0 {
                window_tps
            } else {
                ema_tok_s * 0.6 + window_tps * 0.4
            };
            window_start = Instant::now();
            window_tokens = 0;
        }
        let tok_s = ema_tok_s;

        if let Some(p) = &progress {
            let mut g = p.lock().unwrap();
            g.step = step;
            g.total = args.steps;
            g.loss = loss_now;
            g.tok_s = tok_s;
            g.done = false;
        }
        if let Some(b) = bar.as_mut() {
            b.update(
                step,
                tokens_per_step,
                &format!("loss {loss_now:.3}  {:.1}k tok/s", tok_s / 1e3),
            );
            if step % args.log_every == 0 || step == args.steps {
                b.log(format!(
                    "step {step:>6}  loss {loss_now:7.4}  tok/s {:8.0}  elapsed {:6.1}s",
                    tok_s,
                    start.elapsed().as_secs_f64()
                ));
            }
        }

        if args.val_every > 0 && step % args.val_every == 0 {
            let vl = validation_loss(
                &model,
                &val_ds,
                &mut rng,
                args.val_batches,
                args.batch_size,
                args.seq_len,
                &device,
            )?;
            if let Some(b) = bar.as_ref() {
                b.log(format!("         val loss {vl:7.4}"));
            }
        }

        if args.save_every > 0 && (step % args.save_every == 0 || step == args.steps) {
            save_checkpoint(&varmap, &config, &args.out_dir, tokenizer.as_ref())
                .with_context(|| format!("checkpointing at step {step}"))?;
        }
    }

    if let Some(p) = &progress {
        let mut g = p.lock().unwrap();
        g.done = true;
    }
    if let Some(b) = bar.as_ref() {
        b.finish();
    }

    if verbose {
        println!(
            "done. checkpoint saved to {}",
            args.out_dir.join("model.safetensors").display()
        );
    }
    Ok(())
}

fn validation_loss(
    model: &Vortexa,
    ds: &ByteDataset,
    rng: &mut StdRng,
    batches: usize,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> Result<f64> {
    let mut total = 0.0f64;
    for _ in 0..batches {
        // Note: no backprop is ever called on these losses, so the
        // autograd graph built here is simply dropped.
        let (x, y) = ds.random_batch(rng, batch_size, seq_len, device)?;
        let loss = cross_entropy(&model.forward(&x)?, &y)?;
        total += loss.to_scalar::<f32>()? as f64;
    }
    Ok(total / batches as f64)
}

/// Next-token cross entropy: `[B, T, V]` logits vs `[B, T]` byte targets.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let logits = logits.reshape((b * t, v))?;
    let targets = targets.reshape((b * t,))?;
    let log_probs = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
    candle_nn::loss::nll(&log_probs, &targets)
}

pub fn save_checkpoint(
    varmap: &VarMap,
    config: &Config,
    dir: &Path,
    tokenizer: Option<&BpeTokenizer>,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    varmap.save(dir.join("model.safetensors"))?;
    std::fs::write(
        dir.join("model_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    if let Some(tok) = tokenizer {
        std::fs::write(dir.join("bpe.json"), tok.to_json()?)?;
    }
    Ok(())
}

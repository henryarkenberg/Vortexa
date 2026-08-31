//! Vortexa CLI: train, chat, and evaluate a tiny RetNet language model.
//!
//!   cargo run --release --           # interactive menu
//!   cargo run --release -- train --data data/input.txt --steps 20000
//!   cargo run --release -- generate --checkpoint checkpoints --prompt "Q: hello\nA:"

#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use vortexa::config::Config;

#[derive(Parser)]
#[command(
    name = "vortexa",
    version,
    about = "Vortexa: a tiny byte-level RetNet language model (CPU, Candle).\nRun without a subcommand for the interactive menu."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
// CLI subcommand variants differ a lot in size (Train holds many flags); the
// enum is parsed once at startup so the memory cost is irrelevant.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Train a model on raw text (byte-level tokens).
    Train {
        /// Path to the training text file.
        #[arg(long)]
        data: PathBuf,
        /// Optional JSON file defining the model architecture. Takes full
        /// precedence over the --d-model/--layers/... flags.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Output directory for checkpoints.
        #[arg(long, default_value = "checkpoints")]
        out: PathBuf,
        #[arg(long, default_value_t = 20000)]
        steps: usize,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long, default_value_t = 256)]
        seq_len: usize,
        #[arg(long, default_value_t = 6e-4)]
        lr: f64,
        #[arg(long, default_value_t = 50)]
        log_every: usize,
        #[arg(long, default_value_t = 500)]
        val_every: usize,
        #[arg(long, default_value_t = 4)]
        val_batches: usize,
        #[arg(long, default_value_t = 1000)]
        save_every: usize,
        /// Fraction of the data held out for validation.
        #[arg(long, default_value_t = 0.05)]
        val_frac: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Continue training from this checkpoint (dir or file).
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Linear LR warmup steps before cosine decay.
        #[arg(long, default_value_t = 200)]
        warmup: usize,
        /// Global gradient-norm clip (0 disables).
        #[arg(long, default_value_t = 1.0)]
        clip: f64,

        // --- architecture overrides ---
        #[arg(long, default_value_t = 256)]
        d_model: usize,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 8)]
        heads: usize,
        #[arg(long, default_value_t = 32)]
        head_dim: usize,
        #[arg(long, default_value_t = 512)]
        ffn: usize,
        /// Decay of the shortest-memory retention head.
        #[arg(long, default_value_t = 0.90)]
        decay_min: f64,
        /// Decay of the longest-memory retention head.
        #[arg(long, default_value_t = 0.995)]
        decay_max: f64,
        /// Chunk size for chunkwise retention (0 = full parallel).
        #[arg(long, default_value_t = 64)]
        chunk: usize,
        /// Tokenizer mode: "bytes" or "bpe".
        #[arg(long, default_value = "bytes")]
        tokenizer: String,
        /// Number of BPE merges to learn (when --tokenizer bpe).
        #[arg(long, default_value_t = 512)]
        num_merges: usize,
        /// Backend device: auto|cpu|cuda|metal.
        #[arg(long, default_value = "auto")]
        device: String,
    },
    /// Generate text from a checkpoint using recurrent retention.
    Generate {
        /// Checkpoint directory or model.safetensors path.
        #[arg(long)]
        checkpoint: PathBuf,
        /// Prompt text (bytes); empty seeds with a newline.
        #[arg(long, default_value = "")]
        prompt: String,
        /// Wrap the prompt. Use "{prompt}" as the placeholder, e.g.
        /// "Q: {prompt}\nA:". Empty = use the prompt verbatim.
        #[arg(long, default_value = "")]
        template: String,
        #[arg(long, default_value_t = 200)]
        tokens: usize,
        /// Sampling temperature; 0 means greedy.
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        /// Keep only the k most likely tokens (0 = off).
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Backend device: auto|cpu|cuda|metal.
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Evaluate a checkpoint: deterministic perplexity on the corpus held-out.
    Eval {
        /// Checkpoint directory or model.safetensors file.
        #[arg(long)]
        checkpoint: PathBuf,
        /// Corpus to evaluate (same file used for training).
        #[arg(long, default_value = "data/input.txt")]
        data: PathBuf,
        /// Window length for the sequential eval pass.
        #[arg(long, default_value_t = 256)]
        seq_len: usize,
        /// Fraction of the corpus held out as the validation slice.
        #[arg(long, default_value_t = 0.05)]
        val_frac: f64,
        /// Backend device: auto|cpu|cuda|metal.
        #[arg(long, default_value = "auto")]
        device: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => vortexa::tui::run()?,
        Some(Command::Eval {
            checkpoint,
            data,
            seq_len,
            val_frac,
            device,
        }) => {
            vortexa::eval::run(vortexa::eval::EvalArgs {
                checkpoint,
                data,
                seq_len,
                val_frac,
                device,
            })?;
        }
        Some(Command::Train {
            data,
            out,
            steps,
            batch_size,
            seq_len,
            lr,
            log_every,
            val_every,
            val_batches,
            save_every,
            val_frac,
            seed,
            resume,
            warmup,
            clip,
            config: config_path,
            d_model,
            layers,
            heads,
            head_dim,
            ffn,
            decay_min,
            decay_max,
            chunk,
            tokenizer,
            num_merges,
            device,
        }) => {
            // A --config JSON file takes full precedence over the individual
            // --d-model/--layers/... flags, so users can define the whole
            // network by hand.
            let config = match &config_path {
                Some(path) => {
                    let text = std::fs::read_to_string(path).with_context(|| {
                        format!("reading config file {}", path.display())
                    })?;
                    let cfg: Config =
                        serde_json::from_str(&text).with_context(|| {
                            format!("parsing config file {}", path.display())
                        })?;
                    cfg.validate()?;
                    cfg
                }
                None => Config {
                    vocab_size: 256,
                    d_model,
                    num_layers: layers,
                    num_heads: heads,
                    head_dim,
                    ffn_dim: ffn,
                    max_seq_len: seq_len,
                    decay_min,
                    decay_max,
                    chunk_len: chunk,
                    tokenizer,
                    num_merges,
                },
            };
            vortexa::train::run(vortexa::train::TrainArgs {
                data,
                out_dir: out,
                steps,
                batch_size,
                seq_len,
                lr,
                log_every,
                val_every,
                val_batches,
                save_every,
                val_frac,
                seed,
                resume,
                warmup_steps: warmup,
                grad_clip: clip,
                device,
                config,
            })?;
        }
        Some(Command::Generate {
            checkpoint,
            prompt,
            template,
            tokens,
            temperature,
            top_k,
            device,
            seed,
        }) => {
            vortexa::generate::run(vortexa::generate::GenerateArgs {
                checkpoint,
                prompt,
                template,
                tokens,
                temperature,
                top_k,
                device,
                seed,
            })?;
        }
    }
    Ok(())
}

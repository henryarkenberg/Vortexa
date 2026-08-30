//! Interactive console UI: a friendly menu with ASCII branding, device
//! selection, train / chat / eval on any data file.

use std::path::{Path, PathBuf};

use anyhow::Result;
use rand::{rngs::StdRng, SeedableRng};

use crate::config::Config;
use crate::device;
use crate::eval::{self, EvalArgs};
use crate::generate::{self, Generator};
use crate::train::{self, TrainArgs};

pub fn run() -> Result<()> {
    let mut device_pref = "auto".to_string();
    banner();

    loop {
        let device = device::pick_device(&device_pref)?;
        let dev_name = device::describe(&device);
        menu(&dev_name);

        match ask("choose").as_str() {
            "1" => train_flow(&device_pref),
            "2" => continue_flow(&device_pref),
            "3" => chat_flow(&device_pref),
            "4" => eval_flow(&device_pref),
            "5" => device_flow(&mut device_pref),
            "6" => about(),
            "0" | "q" | "quit" | "exit" => {
                println!("\nbye 👋");
                return Ok(());
            }
            other => println!("unknown option '{other}'"),
        }
    }
}

/// Render the menu as a right-aligned box whose width adapts to its content
/// (so the `Device : <name>` row never breaks the borders).
fn menu(dev_name: &str) {
    let labels: Vec<String> = [
        "[1] Train".to_string(),
        "[2] Continue".to_string(),
        "[3] Chat / Ask".to_string(),
        "[4] Evaluate".to_string(),
        format!("[5] Device : {dev_name}"),
        "[6] About".to_string(),
        "[0] Exit".to_string(),
    ]
    .to_vec();

    let w = labels.iter().map(|l| l.chars().count()).max().unwrap_or(1);
    let inner = w + 2; // a space either side of the label

    let title = " VORTEXA ";
    let left = (inner - title.chars().count()) / 2;
    let top = format!(
        "{}{}{}",
        "─".repeat(left),
        title,
        "─".repeat(inner - left - title.chars().count())
    );

    println!();
    println!("  ╭{top}╮");
    for l in &labels {
        println!("  │ {:<w$} │", l, w = w);
    }
    println!("  ╰{}╯", "─".repeat(inner));
}

fn banner() {
    // "VORTEXA" in a uniform 6-wide × 5-tall block font. Every glyph is the
    // same width and height so the letters align and the word reads cleanly.
    const LETTERS: [[&str; 5]; 7] = [
        ["##  ##", "##  ##", "##  ##", " ## ##", "  ### "], // V
        [" #### ", "#    #", "#    #", "#    #", " #### "], // O
        ["####  ", "#   # ", "####  ", "#  #  ", "#   # "], // R
        ["######", "  ##  ", "  ##  ", "  ##  ", "  ##  "], // T
        ["######", "#     ", "##### ", "#     ", "######"], // E
        ["##  ##", "##  ##", "  ##  ", "##  ##", "##  ##"], // X
        [" ###  ", "#    #", "######", "#    #", "#    #"], // A
    ];
    const INDENT: &str = "      ";
    println!();

    // Print the word, joining each glyph's row with a single space.
    for row in 0..5 {
        let line: String = LETTERS.iter().map(|l| l[row]).collect::<Vec<_>>().join(" ");
        println!("{INDENT}{line}");
    }

    // Center the version and tagline underneath the block.
    let version = format!("v{}", env!("CARGO_PKG_VERSION"));
    let block_w = 6 * LETTERS.len() + (LETTERS.len() - 1);
    let left = (block_w / 2).saturating_sub(version.len() / 2);
    println!("{INDENT}{}{}", " ".repeat(left), version);
    println!("{INDENT}a tiny RetNet language model · CPU / GPU · Rust");
    println!();
}

fn train_flow(device_pref: &str) {
    let data = PathBuf::from(ask_default("data file (any .txt)", "data/input.txt"));
    if !data.exists() {
        println!("no file at {} — put your training text there", data.display());
        return;
    }
    let steps = ask_parse("training steps", 20000);
    let tokenizer = ask_default("tokenizer", "bpe");
    let out = PathBuf::from(ask_default("checkpoint directory", "checkpoints"));

    let config = Config {
        tokenizer: sanitize_tokenizer(&tokenizer),
        max_seq_len: 256,
        ..Config::larger()
    };

    run_training(TrainArgs {
        data,
        out_dir: out,
        steps,
        batch_size: 16,
        seq_len: 256,
        lr: 1e-3,
        log_every: 50,
        val_every: 500,
        val_batches: 4,
        save_every: 1000,
        val_frac: 0.05,
        seed: 42,
        resume: None,
        warmup_steps: 200,
        grad_clip: 1.0,
        device: device_pref.to_string(),
        config,
    });
}

fn continue_flow(device_pref: &str) {
    let ckpt = PathBuf::from(ask_default("checkpoint directory", "checkpoints"));
    let extra = ask_parse("additional training steps", 5000);
    let data = PathBuf::from(ask_default("data file (any .txt)", "data/input.txt"));

    let config = config_for_checkpoint(&ckpt);
    println!(
        "continuing: {} layer(s) x {} head(s)@{}d, vocab {}, {}",
        config.num_layers,
        config.num_heads,
        config.head_dim,
        config.vocab_size,
        if config.tokenizer == "bpe" { "BPE" } else { "bytes" }
    );

    run_training(TrainArgs {
        data,
        out_dir: ckpt.clone(),
        steps: extra,
        batch_size: 16,
        seq_len: 128,
        lr: 3e-4,
        log_every: 50,
        val_every: 500,
        val_batches: 4,
        save_every: 1000,
        val_frac: 0.05,
        seed: 42,
        resume: Some(ckpt),
        warmup_steps: 100,
        grad_clip: 1.0,
        device: device_pref.to_string(),
        config,
    });
}

fn chat_flow(device_pref: &str) {
    let checkpoint = PathBuf::from(ask_default("checkpoint directory", "checkpoints"));
    let device = device::pick_device(device_pref).unwrap_or(candle_core::Device::Cpu);
    let mut generator = match Generator::load(&checkpoint, &device) {
        Ok(g) => g,
        Err(e) => {
            println!("could not load model: {e:#}\n(try [1] Train first)");
            return;
        }
    };

    let cfg = generator.config().clone();
    println!(
        "loaded Vortexa | {} layers x {} heads @{}d | vocab {}",
        cfg.num_layers, cfg.num_heads, cfg.head_dim, cfg.vocab_size
    );
    println!("(train on Q/A data like \"Q: ...\\nA: ...\" to make it answer questions)");

    let template = ask_default(
        "answer template ({prompt}=question)",
        "Q: {prompt}\nA:",
    );
    let temperature = ask_parse("temperature (0 = greedy)", 0.4f64);
    let top_k = ask_parse("top-k (0 = off)", 40usize);
    let tokens = ask_parse("tokens per answer", 120usize);

    let mut rng = StdRng::seed_from_u64(42);
    println!("\nask a question; empty input returns to the menu");
    loop {
        let prompt = ask("question");
        if prompt.is_empty() {
            break;
        }
        let full = generate::apply_template(&template, &prompt);
        match generator.complete(&full, tokens, temperature, top_k, &mut rng) {
            Ok(text) => println!("\n{text}\n"),
            Err(e) => println!("generation failed: {e:#}"),
        }
    }
}

fn eval_flow(device_pref: &str) {
    let checkpoint = PathBuf::from(ask_default("checkpoint directory", "checkpoints"));
    let data = PathBuf::from(ask_default("data file", "data/input.txt"));
    let seq_len = ask_parse("window length", 256usize);
    if let Err(e) = eval::run(EvalArgs {
        checkpoint,
        data,
        seq_len,
        val_frac: 0.05,
        device: device_pref.to_string(),
    }) {
        println!("eval failed: {e:#}");
    }
}

fn device_flow(pref: &mut String) {
    let choice = ask_default("device (auto/cpu/cuda/metal)", pref);
    match device::pick_device(&choice) {
        Ok(d) => {
            println!(
                "device set: {} ({})",
                device::describe(&d),
                choice
            );
            *pref = choice;
        }
        Err(e) => println!("{e:#}"),
    }
}

fn about() {
    let dev = device::pick_device("auto").unwrap();
    println!("\n  Vortexa v{}", env!("CARGO_PKG_VERSION"));
    println!("  RetNet-based language model (Rust + Candle)");
    println!("  backend: CPU/GPU | auto-detected device: {}", device::describe(&dev));
    println!("  checkpoint present: {}", Path::new("checkpoints/model.safetensors").exists());
}

fn sanitize_tokenizer(s: &str) -> String {
    if s.eq_ignore_ascii_case("bytes") || s.eq_ignore_ascii_case("byte") {
        "bytes".into()
    } else {
        "bpe".into()
    }
}

fn config_for_checkpoint(ckpt: &Path) -> Config {
    let (_, cfg_path) = train::resolve_checkpoint_paths(ckpt);
    match std::fs::read_to_string(&cfg_path).ok() {
        Some(json) => serde_json::from_str(&json).unwrap_or_else(|_| {
            println!(
                "could not parse {} — using default config",
                cfg_path.display()
            );
            Config::larger()
        }),
        None => {
            println!(
                "no config at {} — using default config",
                cfg_path.display()
            );
            Config::larger()
        }
    }
}

fn run_training(args: TrainArgs) {
    if let Err(e) = train::run(args) {
        println!("training failed: {e:#}");
    }
}

// --- tiny stdin helpers ---

fn ask(label: &str) -> String {
    use std::io::{BufRead, Write};
    print!("{label}> ");
    let _ = std::io::stdout().flush();
    let mut buf = String::new();
    std::io::stdin().lock().read_line(&mut buf).unwrap_or(0);
    buf.trim().to_string()
}

fn ask_default(label: &str, default: &str) -> String {
    let answer = ask(&format!("{label} [{default}]"));
    if answer.is_empty() {
        default.to_string()
    } else {
        answer
    }
}

fn ask_parse<T: std::str::FromStr + std::fmt::Debug>(label: &str, default: T) -> T {
    let answer = ask(label);
    answer.parse().unwrap_or_else(|_| {
        println!("  (using default {default:?})");
        default
    })
}

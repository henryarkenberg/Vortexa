<img width="1280" height="640" alt="VORTEXA" src="https://github.com/user-attachments/assets/1ba0e617-b03c-4727-be94-a2e6871b8d35" />
# Vortexa

Created By TheArkenberg (in collaboration with DeepseekV4)

Vortexa is a small language model you can train on your own text. It is
built on a RetNet, which is a modern attention-free architecture for
sequence models. Instead of computing attention, a RetNet keeps a compact
recurrent state, which makes training long contexts cheaper and generation
constant time, with no key-value cache.

The whole model is around 3M parameters. It fits comfortably on a normal
laptop and trains at thousands of tokens per second on CPU. It is a research
lab rather than a production chatbot, and that is the point. You can read
every line of the math in `src/retention.rs` and change it.

## What you can do with it

- Train on any plain text file you have (any `.txt`).
- Pick bytes or BPE tokenization.
- Choose the backend: CPU, CUDA or Metal, with automatic detection.
- Chat with a trained model in a Q&A style ("Q: ... A:...").
- Measure quality with a deterministic perplexity evaluator.
- Use it as a library from your own Rust project.

## How it looks

Run it and you get a full-screen terminal app (built with Ratatui). The main
menu shows the VORTEXA banner, and you move with the arrow keys or j/k and
press Enter to choose:

- **Train** - pick a `.txt` from the `data/` folder and your step count, then
  watch a live gauge
- **Continue** - resume training from a checkpoint
- **Chat / Ask** - type a question, get an answer in the Q&A format you trained
- **Datasets** - list, download (with a progress bar) and manage text corpora
  into `data/`. Includes plain-text sources and HuggingFace dataset artifacts
  like `codelion/finewiki-10M` (a Wikipedia corpus, decoded from Parquet
  automatically)
- **Evaluate** - links out to the deterministic perplexity command
- **Settings** - edit the architecture, tokenizer, training and chat options
- **Device** - cycles auto / cpu / cuda / metal
- **About** - version and a short summary

On first launch the app creates the folders it needs (an empty `data/` and a
default `settings.json`), so a fresh download works immediately. The training
screen draws a real-time progress gauge with step, loss and tokens per second,
and you stop it with Esc. The chat view auto-scrolls to the newest message.
On terminals that cannot do a full-screen TUI it falls back to a plain
line-based menu.

## Install

### Option A: download a prebuilt binary

For Windows, Linux and macOS binaries, check the Releases page of this
repository. Each release contains a self-contained executable plus the
license. Unzip it and run it. Nothing else is needed.

- Releases: https://github.com/henryarkenberg/Vortexa/releases

If your operating system is not covered by a built file, use Option B.

### Option B: build from source

You only need Rust (1.75 or newer, for `div_ceil` and friends).

- Rust: https://rustup.rs

```bash
git clone https://github.com/henryarkenberg/Vortexa.git
cd vortexa
cargo build --release
```

The binary is at `target/release/vortexa`. To install it into your PATH:

```bash
cargo install --path .
```

On Windows the same commands work in PowerShell, just append `.exe` when you
run the binary by hand.

## Train on your own data

### The easy way

```bash
cargo run --release
```

Then pick `[1] Train`, type the path of your text file (like
`data/input.txt`), follow the prompts and watch the progress bar.

### The CLI way

```bash
cargo run --release -- train --data books.txt --tokenizer bpe --steps 10000
```

Useful options:

| Flag | Meaning | Default |
|---|---|---|
| `--data` | your text file | `data/input.txt` |
| `--steps` | training steps | 20000 |
| `--tokenizer` | `bpe` or `bytes` | `bpe` |
| `--num-merges` | BPE merges (vocab = 256 + merges) | 512 |
| `--seq-len` | context length | 256 |
| `--lr` | peak learning rate | 1e-3 |
| `--device` | `auto`, `cpu`, `cuda`, `metal` | `auto` |
| `--config` | a JSON file defining the whole architecture | *(none)* |

Checkpoints are saved to the directory you choose (default `checkpoints`) as
`model.safetensors` plus `model_config.json` and `bpe.json`. Training
continues from a checkpoint when you pass `--resume <dir>` or use the menu
option `[2] Continue`.

The included `data/input.txt` is a copy of Tiny Shakespeare so the example
works immediately. Any text file is fine, and the menu accepts a path to
anywhere.

## Choose your own architecture

The network is not fixed to one size. You can make it as big or as small as
you want, either with individual flags or with a config file.

With flags:

```bash
cargo run --release -- train --data books.txt \
  --d-model 512 --layers 6 --heads 16 --head-dim 32 --ffn 1024 \
  --num-merges 1024 --steps 10000
```

Or with a JSON config file that defines every number:

```bash
cargo run --release -- train --data books.txt --config examples/large.json
```

The file takes full precedence over the flags. Two ready-made examples live
in `examples/`:

- `examples/small.json` — about 0.5M parameters, quick on any laptop
- `examples/large.json` — about 17M parameters, needs more time and memory

The whole `Config` shape is:

```json
{
  "vocab_size": 256,
  "d_model": 512,
  "num_layers": 6,
  "num_heads": 16,
  "head_dim": 32,
  "ffn_dim": 1024,
  "max_seq_len": 512,
  "decay_min": 0.9,
  "decay_max": 0.995,
  "chunk_len": 64,
  "tokenizer": "bpe",
  "num_merges": 1024
}
```

Notes on the fields:

- `d_model` must equal `num_heads * head_dim`.
- `tokenizer` is `bytes` (vocab stays 256) or `bpe` (vocab becomes
  `256 + num_merges` when you train).
- `chunk_len` controls chunkwise retention. Leave it around 64.
- `decay_min` / `decay_max` set the starting decay range; each head learns
  its own decay after that.

Whatever you choose is stored in `model_config.json`, so `generate` and
`eval` load the exact same architecture later.

## Q&A: making it answer questions

A 3M parameter model cannot reason, but it can learn the pattern of
question and answer. Format your data that way and train:

```text
Q: What is the capital of France?
A: Paris.

Q: How many hours in a day?
A: 24.
```

Then ask it with the default chat mode, which wraps your prompt as
`Q: {prompt}\nA:`. You can change the template from the menu or with
`--template "Q: {prompt}\nA:"`.

Expect answers that look right for simple, common patterns, and stay away
from anything a 3M model could not have learned.

## Use it in your own project

Vortexa ships both a CLI and a Rust library. Add it as a dependency in your
`Cargo.toml`:

```toml
[dependencies]
vortexa = { path = "/path/to/vortexa" }
```

Train a model from your code:

```rust
use vortexa::{config::Config, train::{self, TrainArgs}};

let config = Config {
    tokenizer: "bpe".into(),
    num_merges: 512,
    max_seq_len: 256,
    ..Config::larger()
};

train::run(TrainArgs {
    data: "data.txt".into(),
    out_dir: "checkpoints".into(),
    steps: 10000,
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
    device: "auto".into(),
    config,
})?;
```

Then load a checkpoint and generate:

```rust
use rand::{rngs::StdRng, SeedableRng};
use vortexa::{device, generate::Generator};

let dev = device::pick_device("auto")?;
let mut gen = Generator::load(std::path::Path::new("checkpoints"), &dev)?;
let mut rng = StdRng::seed_from_u64(42);

let answer = gen.complete("Q: What is 2+2?\nA:", 100, 0.4, 40, &mut rng)?;
println!("{answer}");
```

The important public pieces are `Config`, `BpeTokenizer`, `ByteDataset`,
`Vortexa`, `Generator`, `TrainArgs` and the `retention` module.

## Evaluate

A quick, reproducible number beats eyeballing:

```bash
cargo run --release -- eval --checkpoint checkpoints
```

It reports nats per token and perplexity on a held-out slice of the corpus.
Use it to compare configs: BPE vs bytes, depth, chunk size, and so on. It is
deterministic, so your numbers will match across machines.

## What it uses

Vortexa is built on small, fine pieces:

- Candle, the tensor framework: https://github.com/huggingface/candle
  - Candle examples and docs: https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training
  - Candle VarBuilder docs: https://github.com/huggingface/candle/blob/main/candle-nn/src/var_builder.rs
- The RetNet paper, "Retentive Network: A Successor to Transformer for Large
  Language Models": https://arxiv.org/abs/2307.08621
- Tiny Shakespeare dataset: https://github.com/karpathy/char-rnn (used only
  as the sample `data/input.txt`)
- indicatif for progress bars: https://github.com/console-rs/indicatif
- mimalloc as the allocator: https://github.com/microsoft/mimalloc

For the design story and the full build plan, read
[`docs/design-guide.md`](docs/design-guide.md).

## Development

- Tests: `cargo test --release`
- Lint: `cargo clippy --release -- -D warnings`
- CI runs both on every push and pull request.
- Package a Windows release zip: `powershell scripts\package-release.ps1`

Releases with prebuilt binaries are built automatically by GitHub Actions
when you push a tag like `v0.3.0`. See `.github/workflows/release.yml`.

## License

MIT. See [LICENSE](LICENSE). Use it however you like, commercially included,
with attribution.

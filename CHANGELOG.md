# Changelog

All notable changes to Vortexa are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/). This
project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-08 | Publishing release

### Documentation
- Rewrote `README.md`: VORTEXA ASCII art heading; simple, humanized copy;
  no em dashes.
- Added install options: prebuilt release binaries (Windows/Linux/macOS) or
  build from source with Rust.
- Added a train guide (menu + CLI), a Q&A data format guide, and a
  "use it in your own project" section with copy-paste examples.
- Added an Evaluate section and a "What it uses" section with links to
  Candle, the RetNet paper, Tiny Shakespeare, indicatif, and mimalloc.
- Moved the design story to `docs/design-guide.md` and linked it.

### Licensing
- Added MIT `LICENSE`.

### Automation & distribution
- Added `.github/workflows/ci.yml`: clippy + tests on every push and pull
  request.
- Added `.github/workflows/release.yml`: on a tag push like `v0.2.0`, builds
  Windows/Linux/macOS release binaries and attaches them to the GitHub
  Release.
- Added `scripts/package-release.ps1` for building a local ready-to-run
  Windows zip.
- Removed stale `training.log` / `training.err`; updated `.gitignore`
  (`/dist`, logs, `.DS_Store`).

### API
- Split the project into a library + binary crate (`src/lib.rs` + thin CLI in
  `main.rs`), so Vortexa can be embedded in other Rust projects.
- Exposed a clean public API: `Config`, `BpeTokenizer`, `ByteDataset`,
  `Vortexa`, `Generator`, `TrainArgs`, `eval`, `retention`.

### Flexible architecture
- Added a `--config <file.json>` flag so users can define the entire network
  by hand (d_model, layers, heads, head_dim, ffn, decays, chunk, tokenizer).
  Full precedence over the `--d-model`/`--layers`/... flags.
- Added ready-made presets `examples/small.json` (~0.5M) and
  `examples/large.json` (~17M).
- Added model-size **templates** to the Settings panel: tiny ~0.5M, small
  ~2.8M, medium ~7M, large ~17M. Cycling a template applies the architecture
  at once; hand-editing an architecture field switches the label to "custom".

### UX
- Redesigned the menu box so borders align regardless of the device name;
  redesigned the VORTEXA ASCII banner as a uniform block font.
- Added a full-screen terminal UI with Ratatui: main menu, live training
  gauge, chat with an input box, device selector, and an about panel. The
  older line-based menu remains as a fallback.
- Added a Settings panel: edit the model architecture, tokenizer, training
  defaults, device, and chat options in the UI. Persisted to `settings.json`.
  The first row is a **Model template** selector that cycles size presets
  (tiny ~0.5M, small ~2.8M, medium ~7M, large ~17M) and applies them at once.
- Chat now auto-scrolls to the latest message on new output.

## [0.1.0] - earlier | Core research build

### Architecture
- Initial byte-level RetNet: embeddings, RetNet blocks with RMSNorm and GELU
  FFN, LM head; about 460k parameters.
- Multi-scale retention: per-head `Wq/Wk/Wv`, recurrent state `S`, read-out
  `(q·S)/√d`.
- RetNet fidelity: learnable per-head decay (sigmoid parameterized in (0,1)),
  SwiGLU feed-forward network, per-head value RMSNorm.
- Chunkwise recurrent retention for training: small `chunk_len × chunk_len`
  score matrices, recurrent state carried across chunks; cost grows
  O(T·chunk) instead of O(T²). Made context 256 and 512 affordable.
- Parallel and full-recurrent retention kept as tested reference
  implementations.

### Training
- Warmup + cosine learning-rate schedule.
- Global-norm gradient clipping.
- AdamW optimizer.
- Defaults tuned by measurement: batch 16 × seq 256, chunk 64. Benchmarks
  showed the workload is memory-bandwidth bound on laptop CPUs.
- Cached causal-index matrices and arange vectors to cut per-step allocation
  churn.
- `mimalloc` global allocator for cheaper tensor allocation.

### Tokenization
- From-scratch byte-level BPE tokenizer: learns `num_merges` pair merges,
  per-word linear encoding, exact byte round-trip, serde persistence
  (`bpe.json`).
- `vocab_size` grows to `256 + merges`; bytes mode still available.
- Resume reuses the checkpoint's `bpe.json` instead of retraining, fixing a
  nondeterministic tie-break risk that could silently shift tokenization.

### Evaluation
- `vortexa eval`: deterministic perplexity and nats/token on a held-out
  slice, comparable across configs.

### CLI & UI
- Interactive menu: Train / Continue / Chat / Evaluate / Device / About /
  Exit, with version display.
- Chat ("Ask") mode with Q&A template (default `Q: {prompt}\nA:`),
  configurable via `--template`.
- Progress bars moved to `indicatif` with ETA, step position, loss and tok/s
  status.
- Device selection with automatic detection: `auto` → CUDA → Metal → CPU,
  plus forced `cpu` / `cuda` / `metal`.
- Checkpoint resume (`--resume`), temperature + top-k sampling, greedy mode.

### Data
- Byte/token datasets built from any plain-text file, with deterministic
  train/val split and fixed-seed batching.

### Engineering
- `cargo clippy` clean.
- 9 tests covering the four guide correctness tests, chunkwise/parallel/
  recurrent equivalence, BPE round-trip, and serde round-trip.
- Included `data/input.txt` (Tiny Shakespeare) as a sample corpus.

## Notes

- Work before 0.2.0 was developed iteratively under `0.1.0` and is grouped by
  theme above rather than by exact release.

## Links

- Repository: https://github.com/henryarkenberg/Vortexa
- Releases: https://github.com/henryarkenberg/Vortexa/releases

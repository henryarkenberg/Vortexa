# Changelog

All notable changes to Vortexa are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/). This
project uses [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-08 | Correctness fix

### Architecture (retention)
- Fixed the Multi-Scale Retention to match the RetNet paper. Previously the
  value was RMS-normalized on the *input* and the output passed through only
  the output projection, which suppressed the retention signal and made the
  model behave like an FFN (early loss plateau). Now the head outputs are
  concatenated, **GroupNorm'd per head**, multiplied by a **`swish(g_proj(x))`
  gate**, then projected out — exactly the paper's MSR.
- Removed the extra `1/sqrt(head_dim)` read-out scale (the output GroupNorm
  handles stability) and the value RMSNorm.
- Moved the default decay schedule to the paper's range
  (`0.969` → `0.9995`, was `0.90` → `0.995`), so long-range memory is kept.

### Training
- Lowered the default learning rate to `6e-4` (matches the reference RetNet
  mini model; `1e-3` oscillated and settled higher).
- Training now prints a plain stdout loss line, so redirected / non-TTY runs
  (and `training.log`) stay readable even though indicatif hides its bar.

### Result
- With the fix, a 600-step bytes run reached 2.98 nats and was still
  descending (baseline ln 256 = 5.55), instead of plateauing near 4.6.

## [0.3.0] - 2026-08 | Terminal UI era

### User experience
- Added a full-screen Ratatui terminal UI as the default `cargo run --release`
  experience: animated main menu, live training gauge, a chat input with
  scrolling history, a settings panel, device selector, and an about screen.
  The older line-based menu in `ui.rs` is kept as a fallback.
- Training now runs on a background thread and reports progress to the UI
  through a shared `TrainProgress`, so the gauge updates in real time and you
  can stop it with Esc (or Ctrl+C anywhere).
- Chat auto-scrolls to the newest message; PageUp/PageDown or j/k scroll back.

### Data & onboarding
- Added a **Datasets** menu with a catalog of downloadable corpora (name, size,
  capability) and a streaming download gauge into `data/`. Downloads run on a
  background thread and can be cancelled with Esc.
- Added support for HuggingFace dataset artifacts stored as Parquet
  (e.g. `codelion/finewiki-10M`): the file is fetched once and a named text
  column is decoded into a plain `.txt`, without datasets-server page limits.
- The training screen now lists the `.txt` files in `data/` and lets you pick
  one (cycle with j/k). An empty `data/` folder is shipped in release zips.
- First-run setup creates the `data/` folder and a default `settings.json`
  automatically, so a fresh install or release zip works out of the box.

### Settings
- Added persistent user settings (`settings.json`), editable from the UI and
  reloaded on startup. Covers device, data file, training hyperparameters,
  tokenizer, architecture, and chat options.
- Added model-size **templates** to the Settings panel: tiny (~0.5M), small
  (~2.8M), medium (~7M), and large (~17M). Cycling a template applies the
  architecture at once. Editing an architecture row by hand switches the
  label to "custom".

### Flexible architecture
- Added `--config <file.json>` to define the whole network by hand, with full
  precedence over the individual `--d-model`/`--layers`/... flags.
- Added ready-made presets `examples/small.json` (~0.5M) and
  `examples/large.json` (~17M).

### Engineering
- Split the crate into a library + binary so Vortexa can be embedded in other
  Rust projects; exposed a public API (`Config`, `BpeTokenizer`,
  `ByteDataset`, `Vortexa`, `Generator`, `TrainArgs`, `eval`, `retention`).
- Training stdout is silenced when the TUI is drawing, so prints never corrupt
  the alternate screen.
- Tests now cover TUI rendering (via TestBackend) and settings round-trip.

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

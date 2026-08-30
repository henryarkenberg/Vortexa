# Vortexa

A tiny byte-level **RetNet** language model in Rust + [Candle](https://github.com/huggingface/candle).
CPU-only, ~0.5M parameters by default, no attention anywhere — the only
sequence-mixing mechanism is retention:

```text
S_t = gamma * S_{t-1} + outer(k_t, v_t)   // recurrent state with decayed memory
y_t = (q_t . S_t) / sqrt(head_dim)        // read-out
```

- **Training**: retention unrolled over the whole sequence (`forward_sequence`)
- **Generation**: one token at a time, carrying `layer x head` states forward
  (`forward_step`) — no KV cache, no prefix recomputation, and no context-length
  limit at inference

## Layout

```text
src/
├── main.rs        CLI (train / generate)
├── config.rs      architecture config (serde-serializable)
├── data.rs        byte-level dataset + batch sampler
├── retention.rs   RetentionHead, MultiScaleRetention, RetentionState (+ tests)
├── model.rs       RMSNorm blocks, FeedForward, RetNetBlock, Vortexa (+ tests)
├── train.rs       AdamW training loop, cross-entropy, checkpointing
└── generate.rs    recurrent generation, greedy/temperature sampling
```

## Usage

Run without arguments for the **interactive menu** (train / continue /
chat-style generation with repeated prompts / checkpoint info / progress):

```bash
cargo run --release
```

Or use the CLI directly:

```bash
cargo run --release -- train --data data/input.txt --steps 20000
cargo run --release -- train --resume checkpoints --steps 5000        # continue training
cargo run --release -- generate --checkpoint checkpoints --prompt "ROMEO:" --tokens 200 --temperature 0.8 --top-k 40
cargo run --release -- eval --checkpoint checkpoints                  # deterministic perplexity
```

Useful flags: `--seq-len`, `--batch-size`, `--lr`, `--val-every`,
`--save-every`, and architecture overrides `--d-model/--layers/--heads/
--head-dim/--ffn/--decay-min/--decay-max`. Temperature `0` = greedy.
Checkpoints are written as `model.safetensors` + `model_config.json`
(the config is reloaded automatically for generation).

## Evaluation

`vortexa eval --checkpoint <dir>` reports deterministic per-token
cross-entropy and perplexity on a held-out slice of the corpus (default 5%),
using the checkpoint's own tokenizer. Because it uses fixed, non-overlapping
windows (no RNG), results are reproducible and directly comparable across
configs — e.g. BPE vs bytes, model depth, chunk size, or a RetNet vs
Transformer run. A val perplexity close to train means the model is
underfitting; a large gap means it is overfitting.

## Training speed

Training uses **parallel retention** — `Y = ((Q K^T) ⊙ D_mask) V / √d`,
where `D_mask[i][j] = γ^(i-j)` for `i ≥ j` — which is mathematically
identical to unrolling the recurrence but runs as three batched matmuls.
Together with the recurrent step used at inference, both forms are tested
for equivalence (`src/retention.rs`). A global `mimalloc` allocator keeps
CPU tensor churn cheap. Training shows a live progress bar on stderr
(clean log lines still go to stdout / `training.log`).

Best measured throughput on a laptop CPU is around **~25k tok/s** at
batch 32 × seq 128 (~4× the naive per-step loop); larger batches throttle.
Defaults are tuned accordingly.

## Tokenization

Two modes, chosen with `--tokenizer`:

- `bytes` (default) — vocab 256, each byte is a token. Simplest, lossless.
- `bpe` — a from-scratch byte-level BPE over `--num-merges` (default 512) merges
  learned from the corpus (vocab grows to `256 + merges`). Common subwords
  become single tokens, so the model packs more meaning per token and trains
  faster per sequence. The tokenizer is trained at train time and saved as
  `bpe.json` next to the checkpoint, so generation reloads it automatically.

## Defaults (~2.8M params)

| vocab | d_model | layers | heads | head_dim | ffn (SwiGLU) | context |
|-------|---------|--------|-------|----------|--------------|---------|
| 256   | 256     | 4      | 8     | 32       | 512          | 256     |

A smaller `~0.5M` config also ships for quick experiments. Head decays are
**learnable** per head (sigmoid-parameterized in `(0,1)`), seeded from the
geometric range over the forgetting gap `(1 - gamma)` between `--decay-min`
(0.90) and `--decay-max` (0.995). The feed-forward network is SwiGLU and each
retention head's value is RMS-normalized per token.

## Tests

The four correctness tests from the design guide live in
`src/retention.rs` / `src/model.rs`:

1. shape preservation through multi-head retention
2. zero-state: first output depends only on the first K/V pair
3. exponential state decay when K/V are zero
4. **parallel == sequence == recurrent** (head level and full-model level)

```bash
cargo test --release
```

## Notes & expectations

- Untrained loss starts near ln(vocab); for bytes that is ln(256) ≈ 5.55.
  A default 2.8M model reaches ~3.0–3.5 nats/token on Tiny Shakespeare after
  a few thousand steps (measured ~4–9k tok/s at batch 32 × seq 256 on a laptop
  CPU; the ~0.5M model is several times faster).
- Learnable multi-scale decays, chunkwise recurrent retention, SwiGLU FFN,
  value RMSNorm, warmup + cosine LR, gradient clipping and byte/BPE
  tokenization are implemented; planned items include weight tying and a
  RetNet-vs-Transformer comparison benchmark.

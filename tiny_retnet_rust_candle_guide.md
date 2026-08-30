# Tiny RetNet LLM in Rust + Candle

## Goal

This project builds a **small autoregressive language model based on RetNet/retention**, entirely in Rust using Hugging Face's Candle framework.

The goal is experimentation, not useful chatbot quality.

The model is deliberately small enough to train on a normal CPU-only laptop. It does **not** use a conventional Transformer attention layer.

The first version should use:

- Rust
- Candle
- CPU
- byte-level tokenization
- causal next-token prediction
- 2–4 RetNet blocks
- roughly 1–5M parameters
- short context during training
- recurrent retention during generation

Candle provides tensors, automatic differentiation, neural-network building blocks, optimizers, CPU execution, and model training support. Its official examples use `VarMap`, `VarBuilder`, and `AdamW` for training. [Candle](https://github.com/huggingface/candle) is therefore a good fit for this experiment.

---

# 1. What we are building

The complete model is:

```text
UTF-8 text
   │
   ▼
bytes 0..255
   │
   ▼
token IDs
   │
   ▼
Embedding
   │
   ▼
┌───────────────────────────────┐
│ RetNet Block                  │
│                               │
│ LayerNorm                     │
│    │                          │
│    ▼                          │
│ Multi-scale Retention         │
│    │                          │
│    ▼                          │
│ Residual                      │
│    │                          │
│ LayerNorm                     │
│    │                          │
│    ▼                          │
│ Feed Forward Network          │
│    │                          │
│    ▼                          │
│ Residual                      │
└───────────────────────────────┘
             × N
   │
   ▼
Final RMSNorm
   │
   ▼
Linear vocabulary head
   │
   ▼
logits[256]
```

There is **no normal self-attention layer** anywhere in the model.

---

# 2. Why RetNet?

A normal autoregressive Transformer repeatedly computes relationships between the current token and previous tokens.

RetNet instead maintains a compressed recurrent state.

A simplified retention recurrence is:

\[
S_t = \gamma S_{t-1} + K_t^T V_t
\]

followed by:

\[
Y_t = Q_t S_t
\]

where:

- \(Q_t\) is the query for the current token
- \(K_t\) is its key
- \(V_t\) is its value
- \(S_t\) is the recurrent retention state
- \(\gamma\) controls decay of old information

This gives us two useful ways to run the model:

### Training

Process a sequence in parallel.

### Generation

Process one token at a time while carrying the retention state forward.

That distinction is one of the main reasons to experiment with RetNet.

---

# 3. Important caveat

This project is a **minimal educational RetNet**, not a claim that it reproduces every detail of the Microsoft Research RetNet implementation.

The original RetNet paper contains several important details, including:

- multi-scale retention
- specific decay schedules
- parallel retention
- recurrent retention
- chunkwise recurrent computation
- normalization and scaling choices

We will implement the core ideas first, then add those details incrementally.

That makes the implementation easier to understand and debug.

---

# 4. Project structure

Create:

```text
tiny-retnet/
├── Cargo.toml
├── data/
│   └── input.txt
└── src/
    ├── main.rs
    ├── config.rs
    ├── data.rs
    ├── model.rs
    ├── retention.rs
    └── train.rs
```

Later, add:

```text
    ├── generate.rs
    ├── checkpoint.rs
    └── benchmark.rs
```

---

# 5. Install Rust

Install Rust using rustup.

Then verify:

```bash
rustc --version
cargo --version
```

Create the project:

```bash
cargo new tiny-retnet
cd tiny-retnet
```

---

# 6. Cargo dependencies

Use Candle's CPU backend initially.

A starting `Cargo.toml` can look like:

```toml
[package]
name = "tiny-retnet"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
rand = "0.8"

candle-core = "0.9"
candle-nn = "0.9"
```

Candle's API changes over time, so if the current release has moved beyond `0.9`, use the current compatible Candle versions from the official repository rather than mixing versions.

The official Candle repository contains CPU support, training support, and examples using `VarMap`, `VarBuilder`, and `AdamW`.

---

# 7. Dataset: start with raw bytes

Do not begin with BPE.

For the first experiment, every byte is a token.

The vocabulary is therefore:

```text
0..255
```

Read a file:

```rust
let data = std::fs::read("data/input.txt")?;
```

For example:

```text
hello world
hello rust
hello neural networks
```

becomes:

```text
104 101 108 108 111 32 119 ...
```

This is inefficient compared with modern tokenizers, but it removes a huge amount of complexity.

You are trying to learn RetNet, not tokenizer engineering.

---

# 8. Training samples

Choose:

```text
SEQ_LEN = 256
```

Take a chunk:

```text
x₀ x₁ x₂ x₃ ... x₂₅₅
```

and train the model to predict:

```text
x₁ x₂ x₃ x₄ ... x₂₅₆
```

Therefore:

```text
input  = data[i .. i + SEQ_LEN]
target = data[i + 1 .. i + SEQ_LEN + 1]
```

The loss is next-token cross entropy.

---

# 9. Configuration

Start with:

```rust
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
}
```

Recommended first configuration:

```text
vocab_size   = 256
d_model      = 128
num_layers   = 2
num_heads    = 4
head_dim     = 32
ffn_dim      = 512
max_seq_len  = 256
```

Because:

```text
4 heads × 32 dimensions = 128
```

matches `d_model`.

This model is tiny.

---

# 10. Embedding

The embedding matrix has shape:

```text
[vocab_size, d_model]
```

For the initial model:

```text
[256, 128]
```

Every input byte becomes a 128-dimensional vector.

In Candle this can be represented using `candle_nn::Embedding`.

Conceptually:

```rust
let embedding = candle_nn::embedding(
    config.vocab_size,
    config.d_model,
    vs.pp("embedding"),
)?;
```

---

# 11. Retention

This is the important part.

For every retention head, create:

```text
Wq: [d_model, head_dim]
Wk: [d_model, head_dim]
Wv: [d_model, head_dim]
```

For input:

```text
X: [batch, sequence, d_model]
```

calculate:

```text
Q = X Wq
K = X Wk
V = X Wv
```

giving:

```text
Q: [batch, sequence, head_dim]
K: [batch, sequence, head_dim]
V: [batch, sequence, head_dim]
```

---

# 12. The recurrent state

For a single head, maintain:

```text
S: [head_dim, head_dim]
```

At timestep `t`:

```text
S = decay * S + outer(K[t], V[t])
```

where:

```text
outer(K, V)
```

is:

```text
K[:, None] × V[None, :]
```

Then:

```text
Y[t] = Q[t] × S
```

giving:

```text
Y[t]: [head_dim]
```

This is the core mechanism.

---

# 13. Multi-head retention

With four heads:

```text
Head 0 → 32 dimensions
Head 1 → 32 dimensions
Head 2 → 32 dimensions
Head 3 → 32 dimensions
```

Concatenate:

```text
32 + 32 + 32 + 32 = 128
```

Then project back into model space:

```text
W_o: [128, 128]
```

So:

```text
Retention
   │
   ├── head 0
   ├── head 1
   ├── head 2
   └── head 3
   │
   ▼
concatenate
   │
   ▼
output projection
```

---

# 14. Multi-scale decay

The point of multiple retention heads is not merely parallelism.

Give different heads different decay rates.

For example:

```text
head 0: γ = 0.90
head 1: γ = 0.95
head 2: γ = 0.99
head 3: γ = 0.995
```

Conceptually:

```text
head 0 → remembers very recent information
head 1 → somewhat longer memory
head 2 → long memory
head 3 → very long memory
```

Later, replace these manually chosen values with the decay schedule used by the full RetNet formulation.

For the first implementation, fixed values are easier to debug.

---

# 15. Important numerical detail

The recurrence:

\[
S_t = \gamma S_{t-1} + K_t^T V_t
\]

can become unstable if dimensions and initialization are poorly chosen.

Use a scale factor.

A useful starting point is:

\[
Y_t = \frac{Q_t S_t}{\sqrt{d}}
\]

where:

```text
d = head_dim
```

This keeps activations from growing too aggressively.

Monitor the loss and tensor magnitudes while debugging.

---

# 16. RetNet block

Use a pre-normalized residual block:

```text
x
│
├───────────────┐
│               │
▼               │
RMSNorm         │
│               │
▼               │
Retention       │
│               │
▼               │
+ <─────────────┘
│
├───────────────┐
│               │
▼               │
RMSNorm         │
│               │
▼               │
FFN             │
│               │
▼               │
+ <─────────────┘
│
▼
output
```

This is not Transformer attention. The only sequence-mixing mechanism is retention.

---

# 17. Feed-forward network

Use a simple two-layer FFN:

\[
FFN(x) = W_2 \sigma(W_1x)
\]

Start with:

```text
d_model = 128
ffn_dim = 512
```

So:

```text
128 → 512 → 128
```

Use GELU initially.

This is simply the per-token nonlinear processing part of the block.

---

# 18. RMSNorm

Use RMSNorm before retention and before the FFN.

For vector \(x\):

\[
RMS(x)=\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}
\]

and:

\[
y_i = \frac{x_i}{RMS(x)}g_i
\]

where `g` is a learned scale vector.

Candle provides neural-network building blocks, but implementing a tiny RMSNorm yourself is also reasonable if you want to understand the operation.

---

# 19. Final model

The model becomes:

```text
token IDs
    │
    ▼
Embedding
    │
    ▼
RetNetBlock 1
    │
    ▼
RetNetBlock 2
    │
    ▼
Final RMSNorm
    │
    ▼
Linear(128 → 256)
    │
    ▼
logits
```

The output shape is:

```text
[batch, sequence, vocab_size]
```

or:

```text
[B, T, 256]
```

---

# 20. Weight tying

A useful optimization for a tiny model is tying the output projection to the embedding matrix.

Instead of having:

```text
Embedding: [256, 128]

Output:
[128, 256]
```

as two independent matrices, reuse the embedding weights for the output projection.

This saves parameters.

However, for the first implementation, independent weights make the code easier.

Add weight tying after everything works.

---

# 21. Parallel training vs recurrent inference

This distinction is extremely important.

## Training

For the first version, it is acceptable to implement the recurrence explicitly:

```text
for t in 0..T {
    state = decay * state
          + outer(k[t], v[t]);

    y[t] = q[t] * state;
}
```

This is simple and correct conceptually.

It will not be the fastest possible implementation.

For a tiny CPU experiment, that is fine.

## Generation

Use exactly the same recurrence.

After generating one token:

```text
state
```

is retained.

Then the next token only requires the new:

```text
Q
K
V
```

and the existing state.

You do not need to recompute the entire prefix.

---

# 22. Training loop

Candle's training pattern is roughly:

```rust
let mut varmap = VarMap::new();

let vs = VarBuilder::from_varmap(
    &varmap,
    DType::F32,
    &device,
);

let model = RetNet::new(config, vs)?;

let params = candle_nn::ParamsAdamW {
    lr: learning_rate,
    ..Default::default()
};

let mut optimizer =
    candle_nn::AdamW::new(
        varmap.all_vars(),
        params,
    )?;
```

Then:

```rust
let logits = model.forward(&input)?;

let loss = cross_entropy(
    &logits,
    &targets,
)?;

optimizer.backward_step(&loss)?;
```

Candle's official training examples use this same general `VarMap` → `VarBuilder` → `AdamW` pattern.

---

# 23. Cross entropy

Flatten:

```text
[B, T, V]
```

into:

```text
[B*T, V]
```

and flatten targets:

```text
[B, T]
```

into:

```text
[B*T]
```

Then calculate log-softmax and negative log likelihood.

Conceptually:

```rust
let logits = logits.reshape((batch * seq, vocab))?;
let targets = targets.reshape((batch * seq,))?;

let log_probs = candle_nn::ops::log_softmax(
    &logits,
    D::Minus1,
)?;

let loss = candle_nn::loss::nll(
    &log_probs,
    &targets,
)?;
```

Check the current Candle API if an operation has moved between releases.

---

# 24. Batch size

CPU-only:

```text
batch_size = 8
```

is a reasonable starting point.

If your laptop has plenty of RAM and the model is fast enough:

```text
batch_size = 16
```

or:

```text
batch_size = 32
```

Try not to optimize this before the model works.

---

# 25. Learning rate

A starting point:

```text
learning_rate = 1e-3
```

For a tiny randomly initialized model this may work.

If training is unstable:

```text
3e-4
```

is a reasonable next experiment.

The correct value depends on the exact normalization, initialization, batch size, and dataset.

---

# 26. Gradient clipping

Add gradient clipping once the basic training loop works.

This is particularly useful for recurrent/state-based architectures.

A typical target is:

```text
global gradient norm <= 1.0
```

If the current Candle version makes this inconvenient, initially monitor the loss and reduce the learning rate instead.

---

# 27. Checkpointing

Use Candle's `VarMap`.

During training:

```rust
varmap.save("checkpoints/model.safetensors")?;
```

To resume:

```rust
varmap.load("checkpoints/model.safetensors")?;
```

Candle's `VarBuilder`/`VarMap` infrastructure supports initializing trainable variables and saving/loading them.

Use `safetensors` for checkpoints.

---

# 28. Generation

Generation should maintain one retention state per:

```text
layer × head
```

Conceptually:

```text
states[layer][head]
```

At the beginning:

```text
S = zeros(head_dim, head_dim)
```

For every generated token:

```text
embedding
   ↓
layer 0
   ├── retention state 0
   ├── retention state 1
   ├── retention state 2
   └── retention state 3
   ↓
layer 1
   ├── retention state 0
   ├── ...
   ↓
logits
   ↓
sampling
   ↓
next token
```

The state is updated in-place conceptually, although Candle tensors are immutable values and you will assign the new tensor back to the state variable.

---

# 29. Sampling

Start with greedy generation:

```text
next_token = argmax(logits)
```

This is easiest for debugging.

Then implement temperature:

\[
p_i =
softmax(logits_i/T)
\]

Try:

```text
T = 0.7
T = 1.0
T = 1.2
```

Later add:

- top-k
- top-p
- repetition penalty

Do not add these until greedy generation works.

---

# 30. Byte decoding

Because the model produces bytes, generation is straightforward.

For each predicted token:

```rust
let byte = token as u8;
```

append it to an output buffer.

Then convert the accumulated bytes using UTF-8 handling.

For arbitrary byte-level models, use a loss-tolerant approach while experimenting because the model may produce invalid UTF-8 during early training.

---

# 31. Dataset recommendation

For the first run use a tiny text corpus.

Good options:

```text
Tiny Shakespeare
```

or a text file you already have.

You do not need millions of documents.

Your objective is:

```text
Does the model learn?
Does loss decrease?
Does generation become structured?
Does recurrent inference work?
```

not:

```text
Does it know the world?
```

---

# 32. First experiment

Train:

```text
d_model      = 128
layers       = 2
heads        = 4
head_dim     = 32
ffn_dim      = 512
sequence     = 128
batch        = 8
```

Start with 10,000–50,000 training steps depending on dataset size and CPU speed.

Log:

```text
step
training loss
validation loss
tokens/sec
elapsed time
```

Example:

```text
step 1000   loss 4.21   tok/s 8200
step 2000   loss 3.72   tok/s 8170
step 3000   loss 3.41   tok/s 8150
...
```

The exact numbers are not important.

A consistently falling validation loss is what you want.

---

# 33. What success looks like

At the beginning:

```text
loss ≈ ln(256)
```

which is about:

```text
5.55
```

for a completely uncertain 256-way prediction.

As the model learns, loss should fall.

Eventually generated text should progress from:

```text
xq7��aZp...
```

to:

```text
ROMEO:
What is the...
```

and eventually develop local syntax and recurring structures.

Do not expect semantic intelligence from a 1–5M parameter model.

---

# 34. Tests you should write

Before training, test the retention implementation.

## Test 1: shape test

Input:

```text
[2, 16, 128]
```

Output:

```text
[2, 16, 128]
```

## Test 2: zero-state test

With:

```text
S₀ = 0
```

the first recurrence should depend only on the first K/V pair.

## Test 3: decay test

Set:

```text
K = 0
V = 0
```

Then:

\[
S_t = \gamma S_{t-1}
\]

The state should decay exponentially.

## Test 4: recurrent vs sequence mode

Process:

```text
A B C D
```

as one sequence.

Then process:

```text
A
B
C
D
```

one token at a time while preserving state.

The resulting outputs should be approximately equal.

This is one of the most important tests in the entire project.

---

# 35. Do not accidentally implement Transformer attention

Avoid code that calculates:

```text
Q.matmul(K.transpose(...))
softmax(...)
matmul(V)
```

That is conventional attention.

Your core operation should instead resemble:

```text
state = state * decay
      + outer(k, v)

output = q * state
```

The recurrent state is the defining feature of the experiment.

---

# 36. After the minimal version works

Then progressively add the real RetNet features.

### Version 2

Multi-scale learned decay.

### Version 3

Proper parallel retention.

### Version 4

Chunkwise recurrent retention.

### Version 5

Relative positional encoding / decay formulation from the paper.

### Version 6

Weight tying.

### Version 7

Better initialization.

### Version 8

BPE/tokenizer.

### Version 9

Long-context benchmark.

This keeps every architectural change measurable.

---

# 37. Experiments worth running

Once the model works, don't immediately make it bigger.

Instead vary one thing at a time.

## Experiment A — decay

```text
0.90
0.95
0.99
0.995
0.999
```

Measure validation loss.

## Experiment B — number of heads

```text
1
2
4
8
```

## Experiment C — number of layers

```text
1
2
4
6
```

## Experiment D — sequence length

```text
64
128
256
512
1024
```

## Experiment E — recurrent inference

Compare:

```text
full sequence processing
```

against:

```text
one-token recurrent processing
```

Measure:

```text
memory
latency
tokens/sec
```

---

# 38. A particularly interesting experiment

Create an artificial dataset where information has to be remembered over long distances.

For example:

```text
KEY=ABCD1234

[10,000 random characters]

What was KEY?
```

Train the model to predict:

```text
ABCD1234
```

This lets you directly test long-term memory.

Increase the distance:

```text
100 tokens
1,000 tokens
10,000 tokens
50,000 tokens
```

This is much more informative about recurrent architectures than simply generating Shakespeare.

---

# 39. Expected limitations

A CPU-only tiny RetNet will be slow.

The first implementation will also be intentionally inefficient because the simplest recurrence loops over sequence positions.

That is okay.

The project is about:

```text
correctness
      ↓
understanding
      ↓
experimentation
      ↓
optimization
```

not maximum tokens/sec.

Once it works, you can optimize the Candle tensor operations.

---

# 40. Suggested final project architecture

Eventually aim for:

```text
tiny-retnet/
│
├── Cargo.toml
│
├── data/
│   └── input.txt
│
├── checkpoints/
│
├── src/
│   ├── main.rs
│   │
│   ├── config.rs
│   │
│   ├── data.rs
│   │   ├── byte_dataset
│   │   └── batch_sampler
│   │
│   ├── model.rs
│   │   ├── RetNet
│   │   ├── RetNetBlock
│   │   ├── FeedForward
│   │   └── RMSNorm
│   │
│   ├── retention.rs
│   │   ├── RetentionHead
│   │   ├── MultiScaleRetention
│   │   └── recurrent_state
│   │
│   ├── train.rs
│   │   ├── training_loop
│   │   ├── validation
│   │   └── checkpointing
│   │
│   └── generate.rs
│       ├── recurrent_generation
│       └── sampling
│
└── README.md
```

---

# 41. Recommended implementation order

Do **not** write all of this at once.

Use this order:

```text
1. Cargo project
       ↓
2. Candle CPU test
       ↓
3. Load bytes
       ↓
4. Create batches
       ↓
5. Embedding
       ↓
6. RMSNorm
       ↓
7. Feed-forward network
       ↓
8. Single retention head
       ↓
9. Multi-head retention
       ↓
10. RetNet block
       ↓
11. Full model
       ↓
12. Cross-entropy loss
       ↓
13. AdamW training
       ↓
14. Checkpoint
       ↓
15. Greedy generation
       ↓
16. Recurrent generation
       ↓
17. Tests
       ↓
18. Multi-scale retention
       ↓
19. Benchmarks
```

This order makes debugging dramatically easier.

---

# 42. The most important implementation rule

Keep **sequence-mode retention** and **recurrent-mode retention** as separate functions.

For example:

```rust
fn forward_sequence(
    &self,
    x: &Tensor,
) -> Result<Tensor>
```

and:

```rust
fn forward_step(
    &self,
    x: &Tensor,
    state: &mut RetentionState,
) -> Result<Tensor>
```

Then explicitly test that:

```text
forward_sequence([A,B,C,D])
```

matches:

```text
forward_step(A)
forward_step(B)
forward_step(C)
forward_step(D)
```

within floating-point tolerance.

That gives you a clean architectural boundary and makes the project much easier to extend.

---

# 43. What you should NOT add initially

Avoid:

- Hugging Face tokenizers
- pretrained weights
- CUDA
- quantization
- BPE
- MoE
- retrieval
- instruction tuning
- chat templates
- distributed training
- sophisticated sampling
- FlashAttention
- Transformer blocks

The entire point is to create a **small, understandable RetNet laboratory**.

---

# 44. End state

At the end of the first milestone you should be able to run something like:

```bash
cargo run --release -- train \
    --data data/input.txt \
    --steps 20000
```

then:

```bash
cargo run --release -- generate \
    --checkpoint checkpoints/model.safetensors \
    --prompt "ROMEO:"
```

and get generated text.

More importantly, the architecture should be understandable enough that you can open:

```text
src/retention.rs
```

and change:

```text
decay
state update
head dimensions
number of heads
normalization
projection
```

without needing to understand a giant framework.

---

# 45. References

The primary architectural reference is:

**Retentive Network: A Successor to Transformer for Large Language Models**

Read the paper alongside the implementation. The purpose of this project is not to blindly copy the paper but to build from the core mathematical idea and progressively reproduce the more complete architecture.

For Candle, use the official repository and examples. Candle's documentation and examples demonstrate CPU execution, model construction, `VarBuilder`, `VarMap`, optimizers, and training workflows.

- Candle: https://github.com/huggingface/candle
- Candle training example: https://github.com/huggingface/candle/tree/main/candle-examples/examples/mnist-training
- Candle `VarBuilder`: https://github.com/huggingface/candle/blob/main/candle-nn/src/var_builder.rs

---

# 46. Practical target

For the first working version, aim for:

```text
             Tiny RetNet

Vocabulary:       256
Embedding:        128
Layers:           2
Heads:            4
Head dimension:   32
FFN:              512
Context:          128–256
Parameters:       ~1–3M
Precision:        FP32
Device:           CPU
Tokenizer:        raw bytes
Optimizer:        AdamW
Objective:        next-token prediction
```

Once that works, **stop increasing the model size**.

The next milestone should be making the architecture more faithful and running controlled experiments.

The most valuable experiment is:

```text
             SAME PARAMETERS
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
       RetNet             Transformer
          │                   │
          ▼                   ▼
     same dataset       same dataset
          │                   │
          └─────────┬─────────┘
                    ▼
            compare loss,
            memory and
            inference speed
```

But the Transformer should exist only as a **benchmark**, not as part of the RetNet model itself. The actual LLM you build remains purely RetNet-based.

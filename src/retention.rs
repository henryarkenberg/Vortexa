//! Retention: the sequence-mixing mechanism of Vortexa.
//!
//! This is *not* attention (no `Q K^T -> softmax -> V`). Instead each head
//! keeps a compressed recurrent state `S: [head_dim, head_dim]` and applies:
//!
//! ```text
//! S_t = decay * S_{t-1} + outer(k_t, v_t)     // write, with exponential forgetting
//! y_t = (q_t . S_t) * scale                   // read-out, scale = 1/sqrt(head_dim)
//! ```
//!
//! The per-head decay is **learnable** (a scalar in `(0,1)` via a sigmoid),
//! and each head's value is RMS-normalized per token (a stand-in for the
//! paper's per-head GroupNorm on the value that stays sequence-length
//! independent). Two entry points share the same per-step recurrence:
//!
//! * [`RetentionHead::forward_sequence`] unrolls over a whole `[B, T, D]`
//!   batch — the recurrent reference, used by tests.
//! * [`MultiScaleRetention::forward_parallel`] runs the batched-matmul
//!   parallel form used for training.
//! * [`RetentionHead::forward_step`] / [`MultiScaleRetention::forward_step`]
//!   consume one token at a time for generation.
//!
//! Tests verify all three agree.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_nn::RmsNorm;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn stack_dim0(ts: &[Tensor]) -> Result<Tensor> {
    let refs: Vec<&Tensor> = ts.iter().collect();
    Tensor::stack(&refs, 0)
}

fn stack_dim1(ts: &[Tensor]) -> Result<Tensor> {
    let refs: Vec<&Tensor> = ts.iter().collect();
    Tensor::stack(&refs, 1)
}

/// `[0, 1, ..., c-1]` as f32 — a per-length constant, cached for reuse.
fn arange_f32(c: usize, device: &Device) -> Result<Tensor> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Tensor>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(t) = guard.get(&c) {
        return Ok(t.clone());
    }
    let idx = Tensor::arange(0u32, c as u32, device)?.to_dtype(DType::F32)?;
    guard.insert(c, idx.clone());
    Ok(idx)
}

/// Build `[T, T]` causal exponent matrix of `(i - j)` clamped `>= 0` and a
/// lower-triangular mask. Constant per length; see [`causal_indices`].
fn compute_causal_indices(t: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let pos = Tensor::arange(0u32, t as u32, device)?;
    let pos_i = pos.to_dtype(DType::I64)?; // signed so i-j can go negative
    let diff = pos_i.unsqueeze(1)?.broadcast_sub(&pos_i.unsqueeze(0)?)?; // [T, T]
    let zeros = Tensor::zeros((t, t), DType::I64, device)?;
    let tril = diff.ge(&zeros)?.to_dtype(DType::F32)?; // [T, T]
    let expo = diff.maximum(&zeros)?.to_dtype(DType::F32)?; // [T, T], >= 0
    Ok((expo, tril))
}

/// Cached per-length causal indices. These only depend on `t` (not the data or
/// learnable decays), so building them fresh on every head/chunk/layer of
/// every step is pure waste and allocation churn in a memory-bound loop.
fn causal_indices(t: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    static CACHE: OnceLock<Mutex<HashMap<usize, (Tensor, Tensor)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(v) = guard.get(&t) {
        return Ok((v.0.clone(), v.1.clone()));
    }
    let (expo, tril) = compute_causal_indices(t, device)?;
    guard.insert(t, (expo.clone(), tril.clone()));
    Ok((expo, tril))
}

/// `γ^p` for a single integer `p`, differentiable in `log_gamma`.
fn gamma_power(log_gamma: &Tensor, p: f64) -> Result<Tensor> {
    log_gamma.affine(p, 0.0)?.exp()
}

/// Vector `[c]` of `γ^0, γ^1, ..., γ^{c-1}`, differentiable in `log_gamma`.
fn gamma_powers(log_gamma: &Tensor, c: usize, device: &Device) -> Result<Tensor> {
    arange_f32(c, device)?.broadcast_mul(log_gamma)?.exp()
}

/// Vector `[c]` of `γ^{c-1}, γ^{c-2}, ..., γ^0`, differentiable in `log_gamma`.
fn gamma_reverse(log_gamma: &Tensor, c: usize, device: &Device) -> Result<Tensor> {
    arange_f32(c, device)?
        .affine(-1.0, (c - 1) as f64)? // c-1-i for i in 0..c
        .broadcast_mul(log_gamma)?
        .exp()
}

/// Within-chunk parallel retention (no read-out scale): `[B, C, d] -> [B, C, d]`.
fn chunk_intra(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_gamma: &Tensor,
    c: usize,
    device: &Device,
) -> Result<Tensor> {
    let scores = q.matmul(&k.transpose(1, 2)?)?; // [B, C, C]
    let (expo, tril) = causal_indices(c, device)?; // [C, C]
    let mask = expo.broadcast_mul(log_gamma)?.exp()?.broadcast_mul(&tril)?;
    scores
        .broadcast_mul(&mask.unsqueeze(0)?)?
        .matmul(v)
}

/// Chunkwise sequence over one head: `[B, T, d] -> [B, T, d]`, carrying the
/// recurrent state across chunks. Equivalent to unrolled retention.
fn chunk_sequence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_gamma: &Tensor,
    chunk_len: usize,
    scale: f64,
    device: &Device,
) -> Result<Tensor> {
    let (b, t, d) = q.dims3()?;
    let mut state = Tensor::zeros((b, d, d), DType::F32, device)?;
    let mut outs: Vec<Tensor> = Vec::with_capacity(t.div_ceil(chunk_len));

    let mut start = 0;
    while start < t {
        let c = chunk_len.min(t - start);
        let qc = q.narrow(1, start, c)?;
        let kc = k.narrow(1, start, c)?;
        let vc = v.narrow(1, start, c)?;

        // Output for each token r within the chunk:
        //   y_r = γ^{r+1} (q_r . S_in)  +  within-chunk intra-product
        let intra = chunk_intra(&qc, &kc, &vc, log_gamma, c, device)?; // [B, c, d]
        let gamma = gamma_power(log_gamma, 1.0)?; // γ
        let row_scale = gamma_powers(log_gamma, c, device)?
            .broadcast_mul(&gamma)?; // γ^1 .. γ^c
        let prefix = qc
            .matmul(&state)? // q_r . S_in -> [B, c, d]
            .broadcast_mul(&row_scale.unsqueeze(0)?.unsqueeze(2)?)?;
        let y = intra.add(&prefix)?.affine(scale, 0.0)?; // [B, c, d]
        outs.push(y);

        // State update:
        //   S_out = γ^c S_in + Σ_r γ^{c-1-r} k_r v_r^T
        let k_scaled = kc.broadcast_mul(
            &gamma_reverse(log_gamma, c, device)?.unsqueeze(0)?.unsqueeze(2)?,
        )?; // [B, c, d]
        let update = k_scaled.transpose(1, 2)?.matmul(&vc)?; // [B, d, d]
        let gamma_c = gamma_power(log_gamma, c as f64)?; // γ^c scalar
        state = state.broadcast_mul(&gamma_c)?.add(&update)?;

        start += c;
    }

    let refs: Vec<&Tensor> = outs.iter().collect();
    Tensor::cat(&refs, 1)
}

/// Read-out scale that keeps activation magnitudes stable.
pub fn retention_scale(head_dim: usize) -> f64 {
    1.0 / (head_dim as f64).sqrt()
}

/// logit(x) = ln(x / (1 - x)) — maps a decay in (0,1) to a raw head param.
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// Elementwise sigmoid `1 / (1 + exp(-x))` (candle 0.9 has no `sigmoid`).
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.neg()?.exp()?.affine(1.0, 1.0)?.recip()
}

/// One step of the shared retention recurrence.
///
/// * `q`, `k`, `v`: `[B, H]`
/// * `state`: `[B, H, H]`, rebound to the updated tensor
/// * `decay`: scalar tensor in `(0, 1)`
/// * returns `y`: `[B, H]`
pub fn retention_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    state: &mut Tensor,
    decay: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    // Rank-1 write: outer(k, v) -> [B, H, H]
    let kv = k.unsqueeze(2)?.broadcast_mul(&v.unsqueeze(1)?)?;
    let updated = state.broadcast_mul(decay)?.add(&kv)?;
    *state = updated;
    // Read-out: q . S -> [B, H]
    q.unsqueeze(1)?
        .matmul(state)?
        .squeeze(1)?
        .affine(scale, 0.0)
}

/// Recurrent state of a single head: `[batch, head_dim, head_dim]`.
#[derive(Debug, Clone)]
pub struct RetentionState {
    pub s: Tensor,
}

impl RetentionState {
    pub fn zeros(batch: usize, head_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            s: Tensor::zeros((batch, head_dim, head_dim), DType::F32, device)?,
        })
    }
}

/// A single retention head: projections, a per-token value norm and a
/// learnable decay.
#[derive(Debug)]
pub struct RetentionHead {
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) log_gamma: Tensor,
    pub(crate) vnorm: RmsNorm,
    pub(crate) scale: f64,
}

impl RetentionHead {
    /// `init_decay` is only the starting value; the decay becomes learnable.
    pub fn new(
        vs: VarBuilder,
        d_model: usize,
        head_dim: usize,
        init_decay: f64,
    ) -> Result<Self> {
        Ok(Self {
            wq: candle_nn::linear_no_bias(d_model, head_dim, vs.pp("wq"))?,
            wk: candle_nn::linear_no_bias(d_model, head_dim, vs.pp("wk"))?,
            wv: candle_nn::linear_no_bias(d_model, head_dim, vs.pp("wv"))?,
            log_gamma: vs.get_with_hints((), "log_gamma", candle_nn::Init::Const(logit(init_decay)))?,
            vnorm: candle_nn::rms_norm(head_dim, 1e-5, vs.pp("vnorm"))?,
            scale: retention_scale(head_dim),
        })
    }

    /// Current learnable decay, clamped to `(0, 1)` by construction.
    pub fn decay(&self) -> Result<Tensor> {
        sigmoid(&self.log_gamma)
    }

    /// Projected + value-normalized tensor for `x: [B, T, d_model]`.
    fn values(&self, x: &Tensor) -> Result<Tensor> {
        self.vnorm.forward(&self.wv.forward(x)?)
    }

    /// Unrolled processing of a full sequence: `[B, T, d_model] -> [B, T, head_dim]`.
    pub fn forward_sequence(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.values(x)?;
        let gamma = self.decay()?;
        let mut state =
            Tensor::zeros((b, q.dim(2)?, k.dim(2)?), DType::F32, x.device())?;
        let mut outs: Vec<Tensor> = Vec::with_capacity(t);
        for ti in 0..t {
            let y = retention_step(
                &q.narrow(1, ti, 1)?.squeeze(1)?,
                &k.narrow(1, ti, 1)?.squeeze(1)?,
                &v.narrow(1, ti, 1)?.squeeze(1)?,
                &mut state,
                &gamma,
                self.scale,
            )?;
            outs.push(y.unsqueeze(1)?);
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Recurrent single-token processing: `[B, 1, d_model] -> [B, 1, head_dim]`,
    /// advancing `state` by one step.
    pub fn forward_step(&self, x: &Tensor, state: &mut RetentionState) -> Result<Tensor> {
        let q = self.wq.forward(x)?.squeeze(1)?;
        let k = self.wk.forward(x)?.squeeze(1)?;
        let v = self.values(x)?.squeeze(1)?;
        let gamma = self.decay()?;
        let y = retention_step(&q, &k, &v, &mut state.s, &gamma, self.scale)?;
        y.unsqueeze(1)
    }
}

/// Multi-scale retention: heads with learnable decays + output projection.
#[derive(Debug)]
pub struct MultiScaleRetention {
    heads: Vec<RetentionHead>,
    wo: Linear,
    head_dim: usize,
}

impl MultiScaleRetention {
    /// `init_decays` provides the starting value of each head's learnable
    /// decay (one per head).
    pub fn new(
        vs: VarBuilder,
        d_model: usize,
        num_heads: usize,
        head_dim: usize,
        init_decays: &[f64],
    ) -> Result<Self> {
        assert_eq!(init_decays.len(), num_heads, "need exactly one decay per head");
        let heads_vs = vs.pp("heads");
        let mut heads = Vec::with_capacity(num_heads);
        for (i, decay) in init_decays.iter().enumerate() {
            heads.push(RetentionHead::new(
                heads_vs.pp(i.to_string()),
                d_model,
                head_dim,
                *decay,
            )?);
        }
        Ok(Self {
            heads,
            wo: candle_nn::linear_no_bias(num_heads * head_dim, d_model, vs.pp("wo"))?,
            head_dim,
        })
    }

    /// Parallel retention (used as the training fast path when `chunk_len`
    /// covers the whole sequence; kept as a reference otherwise):
    ///
    /// ```text
    /// Y = ((Q K^T) ⊙ D_mask) V / sqrt(head_dim)
    /// ```
    ///
    /// where `D_mask[i][j] = γ_h^(i-j)` for `i >= j`, built per head from its
    /// learnable decay. Mathematically identical to unrolling `retention_step`,
    /// but computed with three batched matmuls.
    #[allow(dead_code)]
    pub fn forward_parallel(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let n_heads = self.heads.len();
        let device = x.device();

        let mut qs = Vec::with_capacity(n_heads);
        let mut ks = Vec::with_capacity(n_heads);
        let mut vs = Vec::with_capacity(n_heads);
        for head in &self.heads {
            qs.push(head.wq.forward(x)?);
            ks.push(head.wk.forward(x)?);
            vs.push(head.values(x)?);
        }
        let q = stack_dim1(&qs)?; // [B, h, T, d]
        let k = stack_dim1(&ks)?;
        let v = stack_dim1(&vs)?;

        let scores = q.matmul(&k.transpose(2, 3)?)?; // [B, h, T, T]

        // Per-head decay mask D[i][j] = γ^(i-j) for i >= j.
        let (expo, tril) = causal_indices(t, device)?;
        let mut masks: Vec<Tensor> = Vec::with_capacity(n_heads);
        for head in &self.heads {
            let log_gamma = head.decay()?.log()?; // ln gamma (negative)
            let m = expo.broadcast_mul(&log_gamma)?.exp()?;
            masks.push(m.broadcast_mul(&tril)?);
        }
        let masks = stack_dim0(&masks)?; // [h, T, T]

        let y = scores
            .broadcast_mul(&masks.unsqueeze(0)?)? // decay-weighted causal scores
            .matmul(&v)? // [B, h, T, d]
            .affine(self.heads[0].scale, 0.0)?;
        // Merge heads: [B, h, T, d] -> [B, T, h*d]
        let merged = y
            .transpose(1, 2)?
            .reshape((b, t, n_heads * self.head_dim))?;
        self.wo.forward(&merged)
    }

    /// Chunkwise recurrent retention (`[B, T, d_model] -> [B, T, d_model]`).
    ///
    /// Splits the sequence into chunks of `chunk_len`; within a chunk it uses
    /// the parallel form (a small `chunk_len × chunk_len` score matrix) and
    /// between chunks it carries the recurrent state forward. This grows like
    /// `O(T·chunk_len)` instead of the full parallel form's `O(T²)`, making
    /// longer contexts affordable. Mathematically identical to the recurrent
    /// path.
    pub fn forward_chunkwise(&self, x: &Tensor, chunk_len: usize) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let n_heads = self.heads.len();
        let device = x.device();
        let scale = self.heads[0].scale;
        let chunk_len = chunk_len.max(1).min(t);

        let mut per_head: Vec<Tensor> = Vec::with_capacity(n_heads);
        for head in &self.heads {
            let q = head.wq.forward(x)?;
            let k = head.wk.forward(x)?;
            let v = head.values(x)?;
            let log_gamma = head.decay()?.log()?;
            per_head.push(chunk_sequence(&q, &k, &v, &log_gamma, chunk_len, scale, device)?);
        }

        let merged = stack_dim1(&per_head)? // [B, h, T, d]
            .transpose(1, 2)?
            .reshape((b, t, n_heads * self.head_dim))?; // [B, T, h*d]
        self.wo.forward(&merged)
    }

    /// Reference recurrent implementation (kept for testing/equivalence):
    /// `[B, T, d_model] -> [B, T, d_model]`.
    #[allow(dead_code)]
    pub fn forward_sequence(&self, x: &Tensor) -> Result<Tensor> {
        let mut outs: Vec<Tensor> = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            outs.push(head.forward_sequence(x)?);
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        self.wo.forward(&Tensor::cat(&refs, candle_core::D::Minus1)?)
    }

    /// `[B, 1, d_model] -> [B, 1, d_model]`, advancing one state per head.
    pub fn forward_step(
        &self,
        x: &Tensor,
        states: &mut [RetentionState],
    ) -> Result<Tensor> {
        assert_eq!(states.len(), self.heads.len(), "state/head mismatch");
        let mut outs: Vec<Tensor> = Vec::with_capacity(self.heads.len());
        for (head, state) in self.heads.iter().zip(states.iter_mut()) {
            outs.push(head.forward_step(x, state)?);
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        self.wo.forward(&Tensor::cat(&refs, candle_core::D::Minus1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn randn(shape: (usize, usize, usize)) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, &cpu()).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    fn random_linear(in_d: usize, out_d: usize) -> Linear {
        let w = Tensor::randn(0f32, 0.05f32, (out_d, in_d), &cpu()).unwrap();
        Linear::new(w, None)
    }

    /// RMSNorm whose weight initializes to 1 (unlike `VarBuilder::zeros`,
    /// which would zero it and collapse the values to nothing).
    fn vnorm(head_dim: usize) -> RmsNorm {
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, DType::F32, &cpu());
        candle_nn::rms_norm(head_dim, 1e-5, vs).unwrap()
    }
    /// Guide test 4 (core): sequence-mode, recurrent-mode and parallel form
    /// all agree.
    #[test]
    fn sequence_parallel_and_recurrent_agree() {
        let (d_model, head_dim, batch, seq_len) = (16usize, 8usize, 2usize, 32usize);
        let mk = |g| RetentionHead {
            wq: random_linear(d_model, head_dim),
            wk: random_linear(d_model, head_dim),
            wv: random_linear(d_model, head_dim),
            log_gamma: Tensor::new(logit(g) as f32, &cpu()).unwrap(),
            vnorm: vnorm(head_dim),
            scale: retention_scale(head_dim),
        };
        let decays = [0.90f64, 0.95, 0.99];
        let msr = MultiScaleRetention {
            heads: decays.iter().map(|g| mk(*g)).collect(),
            wo: random_linear(head_dim * decays.len(), d_model),
            head_dim,
        };
        let x = randn((batch, seq_len, d_model));

        let par = msr.forward_parallel(&x).unwrap();
        let seq = msr.forward_sequence(&x).unwrap();
        assert_eq!(par.dims3().unwrap(), (batch, seq_len, d_model));
        assert!(
            max_abs_diff(&par, &seq) < 1e-4,
            "parallel vs sequence diverged"
        );

        // Recurrent reference on head 0 only.
        let head0 = &msr.heads[0];
        let seq0 = head0.forward_sequence(&x).unwrap();
        let mut state = RetentionState::zeros(batch, head_dim, &cpu()).unwrap();
        let mut steps: Vec<Tensor> = Vec::with_capacity(seq_len);
        for ti in 0..seq_len {
            steps.push(
                head0
                    .forward_step(&x.narrow(1, ti, 1).unwrap(), &mut state)
                    .unwrap(),
            );
        }
        let refs: Vec<&Tensor> = steps.iter().collect();
        let rec0 = Tensor::cat(&refs, 1).unwrap();
        assert!(max_abs_diff(&seq0, &rec0) < 1e-4);
    }

    /// Chunkwise, parallel and recurrent modes must all agree (with a chunk
    /// size smaller than the sequence, to exercise the state carry).
    #[test]
    fn chunkwise_matches_parallel_and_recurrent() {
        let (d_model, head_dim, batch, seq_len) = (16usize, 8usize, 2usize, 24usize);
        let mk = |g| RetentionHead {
            wq: random_linear(d_model, head_dim),
            wk: random_linear(d_model, head_dim),
            wv: random_linear(d_model, head_dim),
            log_gamma: Tensor::new(logit(g) as f32, &cpu()).unwrap(),
            vnorm: vnorm(head_dim),
            scale: retention_scale(head_dim),
        };
        let decays = [0.90f64, 0.95, 0.99];
        let msr = MultiScaleRetention {
            heads: decays.iter().map(|g| mk(*g)).collect(),
            wo: random_linear(head_dim * decays.len(), d_model),
            head_dim,
        };
        let x = randn((batch, seq_len, d_model));

        let par = msr.forward_parallel(&x).unwrap();
        let chunk = msr.forward_chunkwise(&x, 5).unwrap(); // 5 chunks of seq 24
        let seq = msr.forward_sequence(&x).unwrap();

        assert!(max_abs_diff(&chunk, &par) < 1e-3, "chunkwise vs parallel");
        assert!(max_abs_diff(&chunk, &seq) < 1e-3, "chunkwise vs recurrent");
    }

    /// Guide test 1: shapes preserved through multi-head retention (parallel
    /// and recurrent paths).
    #[test]
    fn multi_scale_shapes_are_preserved() {
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, DType::F32, &cpu());
        let msr =
            MultiScaleRetention::new(vs.pp("ret"), 16, 4, 4, &[0.90, 0.95, 0.99, 0.995])
                .unwrap();
        let x = randn((2, 16, 16));

        let y = msr.forward_parallel(&x).unwrap();
        assert_eq!(y.dims3().unwrap(), (2, 16, 16));

        let mut states: Vec<RetentionState> = (0..4)
            .map(|_| RetentionState::zeros(2, 4, &cpu()))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let ys = msr.forward_step(&x.narrow(1, 0, 1).unwrap(), &mut states).unwrap();
        assert_eq!(ys.dims3().unwrap(), (2, 1, 16));
    }

    /// Guide test 2: with S_0 = 0, the first output depends only on the
    /// first K/V pair.
    #[test]
    fn first_output_depends_only_on_first_kv() {
        let (d_model, head_dim, t) = (12usize, 6usize, 8usize);
        let mk = |g| RetentionHead {
            wq: random_linear(d_model, head_dim),
            wk: random_linear(d_model, head_dim),
            wv: random_linear(d_model, head_dim),
            log_gamma: Tensor::new(logit(g) as f32, &cpu()).unwrap(),
            vnorm: vnorm(head_dim),
            scale: retention_scale(head_dim),
        };
        let head = mk(0.9);
        let first = randn((1, 1, d_model));
        let rest_a = randn((1, t - 1, d_model));
        let rest_b = randn((1, t - 1, d_model));
        let xa = Tensor::cat(&[&first, &rest_a], 1).unwrap();
        let xb = Tensor::cat(&[&first, &rest_b], 1).unwrap();

        let ya = head.forward_sequence(&xa).unwrap();
        let yb = head.forward_sequence(&xb).unwrap();

        let ya0 = ya.narrow(1, 0, 1).unwrap();
        let yb0 = yb.narrow(1, 0, 1).unwrap();
        assert!(max_abs_diff(&ya0, &yb0) < 1e-6);

        // Sanity: the differing suffixes *must* change the later positions
        // much more than the (exactly equal) first position.
        let ya_last = ya.narrow(1, t - 1, 1).unwrap();
        let yb_last = yb.narrow(1, t - 1, 1).unwrap();
        let first_diff = max_abs_diff(&ya0, &yb0);
        let last_diff = max_abs_diff(&ya_last, &yb_last);
        assert!(
            last_diff > first_diff * 10.0 && last_diff > 1e-7,
            "later outputs should differ more than the first: {first_diff} vs {last_diff}"
        );
    }

    /// Guide test 3: with K = V = 0 the state never gains, so read-outs
    /// decay by exactly `gamma` per step.
    #[test]
    fn state_decays_exponentially() {
        let h = 4usize;
        let decay = Tensor::new(0.5f32, &cpu()).unwrap();
        let scale = retention_scale(h);
        let device = cpu();
        let ones = Tensor::ones((1, h), DType::F32, &device).unwrap();
        let zeros = Tensor::zeros((1, h), DType::F32, &device).unwrap();
        let mut st = Tensor::zeros((1, h, h), DType::F32, &device).unwrap();

        let y1 = retention_step(&ones, &ones, &ones, &mut st, &decay, scale).unwrap();
        let y2 = retention_step(&ones, &zeros, &zeros, &mut st, &decay, scale).unwrap();
        let y3 = retention_step(&ones, &zeros, &zeros, &mut st, &decay, scale).unwrap();

        let to_scalar = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let (a, b, c) = (to_scalar(&y1), to_scalar(&y2), to_scalar(&y3));
        assert!((a - h as f32 * scale as f32).abs() < 1e-5);
        assert!((b / a - 0.5).abs() < 1e-5);
        assert!((c / b - 0.5).abs() < 1e-5);
    }
}

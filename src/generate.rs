//! Recurrent generation: feed the prompt token by token to warm up the
//! retention states, then keep emitting bytes one at a time. No KV cache,
//! no recomputation of the prefix — that is the whole point of retention.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::bpe::{BpeData, BpeTokenizer};
use crate::config::Config;
use crate::model::{ModelStates, Vortexa};
use crate::train::resolve_checkpoint_paths;

pub struct GenerateArgs {
    /// Checkpoint directory (containing `model.safetensors`) or the
    /// safetensors file itself.
    pub checkpoint: PathBuf,
    pub prompt: String,
    /// Wrap the prompt around a template; `{prompt}` is the placeholder.
    /// Empty = use the prompt verbatim.
    pub template: String,
    pub tokens: usize,
    /// Sampling temperature; `<= 0` means greedy argmax.
    pub temperature: f64,
    /// Keep only the k most likely tokens (0 = off).
    pub top_k: usize,
    /// Backend device preference (`auto`, `cpu`, `cuda`, `metal`).
    pub device: String,
    pub seed: u64,
}

/// A loaded model ready to complete prompts. Load once, generate many.
pub struct Generator {
    model: Vortexa,
    states: ModelStates,
    device: Device,
    tokenizer: Option<BpeTokenizer>,
    tok_data: Option<BpeData>,
}

impl Generator {
    pub fn load(checkpoint: &Path, device: &Device) -> Result<Self> {
        let (weights_path, cfg_path) = resolve_checkpoint_paths(checkpoint);
        let cfg: Config = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path)
                .with_context(|| format!("reading {}", cfg_path.display()))?,
        )
        .with_context(|| format!("parsing {}", cfg_path.display()))?;
        cfg.validate()?;

        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = Vortexa::new(cfg.clone(), vs)?;
        varmap.load(weights_path.as_path())?;

        // Load the BPE tokenizer if the checkpoint has one.
        let (_, cfg_dir) = resolve_checkpoint_paths(checkpoint);
        let tokenizer = std::fs::read_to_string(cfg_dir.join("bpe.json"))
            .ok()
            .and_then(|s| BpeTokenizer::from_json(&s).ok());
        let tok_data = tokenizer.as_ref().map(|t| t.data());

        Ok(Self {
            states: model.new_states(1, device)?,
            model,
            device: device.clone(),
            tokenizer,
            tok_data,
        })
    }

    pub fn config(&self) -> &Config {
        self.model.config()
    }

    /// Encode a byte prompt into token ids (falling back to raw bytes).
    fn encode_prompt(&self, bytes: &[u8]) -> Vec<u32> {
        match (&self.tokenizer, &self.tok_data) {
            (Some(t), Some(d)) => t.encode(bytes, d),
            _ => bytes.iter().map(|&b| b as u32).collect(),
        }
    }

    /// Decode generated token ids back to bytes.
    fn decode_ids(&self, ids: &[u32]) -> Vec<u8> {
        match (&self.tokenizer, &self.tok_data) {
            (Some(t), Some(d)) => t.decode(ids, d),
            _ => ids.iter().map(|&id| id as u8).collect(),
        }
    }

    /// Complete `prompt` with `tokens` new tokens. Retention states are reset
    /// per call; the returned string includes the prompt.
    pub fn complete(
        &mut self,
        prompt: &str,
        tokens: usize,
        temperature: f64,
        top_k: usize,
        rng: &mut StdRng,
    ) -> Result<String> {
        // Prompt bytes (empty seeds with a newline).
        let mut prompt_bytes = prompt.as_bytes().to_vec();
        if prompt_bytes.is_empty() {
            prompt_bytes.push(b'\n');
        }
        let prompt_ids = self.encode_prompt(&prompt_bytes);

        // Fresh retention state per completion.
        self.states = self.model.new_states(1, &self.device)?;

        let mut logits = None;
        for &id in &prompt_ids {
            let ids = Tensor::from_vec(vec![id], (1, 1), &self.device)?;
            logits = Some(self.model.forward_step(&ids, &mut self.states)?);
        }
        let mut logits = logits.expect("prompt must not be empty");

        let mut out_ids: Vec<u32> = Vec::with_capacity(tokens);
        for _ in 0..tokens {
            let next = sample_token(&logits, temperature, top_k, rng)?;
            out_ids.push(next);
            let ids = Tensor::from_vec(vec![next], (1, 1), &self.device)?;
            logits = self.model.forward_step(&ids, &mut self.states)?;
        }

        let mut produced = prompt_bytes;
        produced.extend(self.decode_ids(&out_ids));
        Ok(String::from_utf8_lossy(&produced).into_owned())
    }
}

pub fn run(args: GenerateArgs) -> Result<()> {
    let device = crate::device::pick_device(&args.device)?;
    println!("device: {}", crate::device::describe(&device));
    let mut generator = Generator::load(&args.checkpoint, &device)?;
    println!(
        "loaded Vortexa ({} layers, {} heads@{}d, d_model {})",
        generator.config().num_layers,
        generator.config().num_heads,
        generator.config().head_dim,
        generator.config().d_model
    );
    let mut rng = StdRng::seed_from_u64(args.seed);
    let full_prompt = apply_template(&args.template, &args.prompt);
    let text = generator.complete(
        &full_prompt,
        args.tokens,
        args.temperature,
        args.top_k,
        &mut rng,
    )?;
    println!("{text}");
    Ok(())
}

/// Substitute a prompt into a template (e.g. `"Q: {prompt}\nA:"`). An empty
/// template returns the prompt verbatim, so plain text continuation still
/// works.
pub fn apply_template(template: &str, prompt: &str) -> String {
    if template.is_empty() {
        return prompt.to_string();
    }
    template.replace("{prompt}", prompt)
}

fn sample_token(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    rng: &mut StdRng,
) -> Result<u32> {
    if temperature <= 0.0 {
        return greedy_token(logits);
    }

    let (_, _, v) = logits.dims3()?;
    let raw = logits.reshape(v)?.to_vec1::<f32>()?;

    // Optional top-k filtering: mask everything below the k-th largest logit.
    let mut filtered = raw.clone();
    if top_k > 0 && top_k < filtered.len() {
        let mut sorted = filtered.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[top_k - 1];
        for l in filtered.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    let t = temperature as f32;
    let max = filtered.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = filtered.iter().map(|l| ((l - max) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let pick = rng.gen_range(0.0f32..sum);
    let mut acc = 0.0f32;
    for (i, e) in exps.iter().enumerate() {
        acc += e;
        if pick <= acc {
            return Ok(i as u32);
        }
    }
    Ok(exps.len() as u32 - 1)
}

fn greedy_token(logits: &Tensor) -> Result<u32> {
    let (_, _, v) = logits.dims3()?;
    let flat = logits.reshape(v)?;
    Ok(flat.argmax(0)?.to_scalar::<u32>()?)
}

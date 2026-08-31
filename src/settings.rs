//! Persistent user settings for the model and training.
//!
//! Stored as `settings.json` next to the project so choices survive across
//! runs. Everything here has sane defaults that match the shipped behavior.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

fn default_model_template() -> String {
    "small".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// Named size preset: tiny / small / medium / large / custom.
    #[serde(default = "default_model_template")]
    pub model_template: String,
    pub device: String,
    pub data_file: String,
    pub steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub lr: f64,
    pub tokenizer: String,
    pub num_merges: usize,
    pub chunk_len: usize,
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub decay_min: f64,
    pub decay_max: f64,
    pub chat_template: String,
    pub temperature: f64,
    pub top_k: usize,
    pub tokens: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_template: "small".to_string(),
            device: "auto".to_string(),
            data_file: "data/input.txt".to_string(),
            steps: 20000,
            batch_size: 16,
            seq_len: 256,
            lr: 6e-4,
            tokenizer: "bpe".to_string(),
            num_merges: 512,
            chunk_len: 64,
            d_model: 256,
            layers: 4,
            heads: 8,
            head_dim: 32,
            ffn: 512,
            decay_min: 0.90,
            decay_max: 0.995,
            chat_template: "Q: {prompt}\nA:".to_string(),
            temperature: 0.4,
            top_k: 40,
            tokens: 120,
        }
    }
}

impl Settings {
    /// Load settings from `dir/settings.json`, falling back to defaults.
    pub fn load(dir: &Path) -> Self {
        std::fs::read_to_string(dir.join("settings.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::write(
            dir.join("settings.json"),
            serde_json::to_string_pretty(self)?,
        )?;
        Ok(())
    }
}

/// Create the folders a first install needs and a default `settings.json`.
///
/// Called on app startup so a fresh download (or the release zip, which ships
/// an empty `data/`) works immediately without the user hunting for files.
pub fn init_workspace() -> Result<()> {
    // These are relative to the current working directory (project root).
    std::fs::create_dir_all("data")?;
    if !Path::new("settings.json").exists() {
        Settings::default().save(Path::new("."))?;
    }
    Ok(())
}

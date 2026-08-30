//! Vortexa: a tiny research-grade RetNet language model in Rust + Candle.
//!
//! The crate exposes the pieces you need to build on top of it:
//!
//! * [`config::Config`] - architecture + tokenizer settings
//! * [`bpe::BpeTokenizer`] - byte-level BPE tokenizer
//! * [`data::ByteDataset`] - token datasets and batch samplers
//! * [`retention`] - the retention mechanism (parallel / chunkwise / recurrent)
//! * [`model::Vortexa`] - the full model
//! * [`generate::Generator`] - load a checkpoint and complete prompts
//! * [`train::TrainArgs`] and [`train::run`] - run a training loop
//! * [`eval::run`] - deterministic perplexity evaluation
//!
//! The CLI lives in the `vortexa` binary; the library is what you embed in
//! your own project.

pub mod bpe;
pub mod config;
pub mod data;
pub mod device;
pub mod eval;
pub mod generate;
pub mod model;
pub mod retention;
pub mod train;
pub mod ui;

pub use crate::config::Config;
pub use crate::generate::Generator;
pub use crate::model::Vortexa;

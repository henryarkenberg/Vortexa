//! A small catalog of downloadable datasets and a streaming downloader.
//!
//! The UI lists curated corpora with their size and capability, letting anyone
//! grab a dataset into the local `data/` folder without leaving the app. Two
//! source kinds are supported:
//!
//! * `Plain` - a direct URL to a `.txt` (or any text) file, streamed to disk.
//! * `Parquet` - a Parquet file (e.g. a HuggingFace dataset artifact) that is
//!   downloaded once, then converted by extracting a named string column into
//!   plain text. This avoids datasets-server rate limits and works for
//!   Parquet-backed corpora like `codelion/finewiki-10M`.
//!
//! Downloads run in a background thread and report progress through a shared
//! [`DownloadProgress`], so a TUI can draw a live gauge.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use parquet::file::reader::FileReader;

/// How a dataset's text is fetched and converted.
#[derive(Clone, Copy)]
pub enum DatasetKind {
    /// A single URL streamed straight into a `.txt` file.
    Plain { url: &'static str },
    /// A Parquet file downloaded once, then a named column written as text.
    Parquet { url: &'static str, column: &'static str },
}

/// A downloadable dataset available in the app.
#[derive(Clone, Copy)]
pub struct Dataset {
    pub id: &'static str,
    pub name: &'static str,
    pub size_hint: &'static str,
    pub desc: &'static str,
    pub kind: DatasetKind,
    pub file: &'static str,
}

/// Curated, stable text sources. Sizes and descriptions are for the picker.
pub const DATASETS: [Dataset; 4] = [
    Dataset {
        id: "shakespeare",
        name: "Tiny Shakespeare",
        size_hint: "1.1 MB",
        desc: "Classic character-level corpus. Great for learning English patterns.",
        kind: DatasetKind::Plain {
            url: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        },
        file: "shakespeare.txt",
    },
    Dataset {
        id: "alice",
        name: "Alice in Wonderland",
        size_hint: "170 KB",
        desc: "Public-domain prose. Small and fast to train.",
        kind: DatasetKind::Plain {
            url: "https://www.gutenberg.org/files/11/11-0.txt",
        },
        file: "alice.txt",
    },
    Dataset {
        id: "frankenstein",
        name: "Frankenstein",
        size_hint: "190 KB",
        desc: "Public-domain prose. Small and fast to train.",
        kind: DatasetKind::Plain {
            url: "https://www.gutenberg.org/files/84/84-0.txt",
        },
        file: "frankenstein.txt",
    },
    Dataset {
        id: "finewiki",
        name: "finewiki-10M",
        size_hint: "25 MB",
        desc: "Wikipedia article text (markdown). Broad coverage, good general English.",
        kind: DatasetKind::Parquet {
            url: "https://huggingface.co/datasets/codelion/finewiki-10M/resolve/main/data/train-00000-of-00001.parquet",
            column: "text",
        },
        file: "finewiki.txt",
    },
];

/// Shareable download progress, polled by the UI. Units are bytes for both
/// [`DatasetKind::Plain`] and [`DatasetKind::Parquet`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DownloadProgress {
    pub total: u64,
    pub done: u64,
    pub finished: bool,
    pub error: bool,
}

impl DownloadProgress {
    pub fn percent(&self) -> u16 {
        if self.total == 0 {
            0
        } else {
            ((self.done as f64 / self.total as f64) * 100.0).min(100.0) as u16
        }
    }
}

/// Stream `dataset` into `dest_dir/<dataset.file>`, reporting progress.
/// Honors `cancel` between chunks. Returns the saved path.
pub fn download(
    dataset: &Dataset,
    dest_dir: &Path,
    progress: Arc<Mutex<DownloadProgress>>,
    cancel: Arc<AtomicBool>,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("creating {}", dest_dir.display()))?;
    {
        let mut p = progress.lock().unwrap();
        *p = DownloadProgress::default();
    }

    let path = dest_dir.join(dataset.file);
    match dataset.kind {
        DatasetKind::Plain { url } => download_plain(url, &path, &progress, &cancel)?,
        DatasetKind::Parquet { url, column } => {
            download_parquet(url, column, &path, &progress, &cancel)?
        }
    }

    {
        let mut p = progress.lock().unwrap();
        p.finished = true;
    }
    Ok(path)
}

fn download_plain(
    url: &str,
    path: &Path,
    progress: &Arc<Mutex<DownloadProgress>>,
    cancel: &Arc<AtomicBool>,
) -> Result<()> {
    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    progress.lock().unwrap().total = total;

    let mut reader = resp.into_reader();
    let mut out = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut buf = vec![0u8; 64 * 1024];
    let mut done = 0u64;

    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = std::fs::remove_file(path);
            bail!("download cancelled");
        }
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        progress.lock().unwrap().done = done;
    }
    Ok(())
}

/// Fetch a Parquet file into a temporary path, then pull the named string
/// column into plain text at `path` and delete the temporary file.
fn download_parquet(
    url: &str,
    column: &str,
    path: &Path,
    progress: &Arc<Mutex<DownloadProgress>>,
    cancel: &Arc<AtomicBool>,
) -> Result<()> {
    let temp = path.with_extension("parquet.tmp");
    download_plain(url, &temp, progress, cancel)?;

    // Conversion: read rows (generic string fields) and write the column.
    let file = File::open(&temp).with_context(|| format!("opening {}", temp.display()))?;
    let reader =
        parquet::file::reader::SerializedFileReader::new(file).context("opening parquet")?;
    let mut out = File::create(path).with_context(|| format!("creating {}", path.display()))?;

    for row in reader.get_row_iter(None).context("iterating parquet")? {
        let row = row?;
        for (name, value) in row.get_column_iter() {
            if name.as_str() != column {
                continue;
            }
            let text = match value {
                parquet::record::Field::Bytes(ba) => std::str::from_utf8(ba.data()).ok(),
                parquet::record::Field::Str(s) => Some(s.as_str()),
                _ => None,
            };
            if let Some(s) = text {
                if !s.is_empty() {
                    writeln!(out, "{s}\n")?;
                }
            }
        }
    }

    let _ = std::fs::remove_file(&temp);
    Ok(())
}

/// List `.txt` files present in `data/`, for the training file picker.
pub fn data_files() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir("data") {
        for entry in rd.flatten() {
            if let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) {
                if name.ends_with(".txt") {
                    out.push(name);
                }
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Network test: downloads the finewiki Parquet dataset to a temp dir and
    /// checks the text is non-empty. Ignored by default so `cargo test` stays
    /// fast and offline; run with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn downloads_parquet_ok() {
        let mut temp = std::env::temp_dir();
        temp.push("vortexa-ds-test");
        let _ = std::fs::remove_dir_all(&temp);

        let ds = DATASETS
            .iter()
            .find(|d| matches!(d.kind, DatasetKind::Parquet { .. }))
            .copied()
            .unwrap_or_else(|| panic!("no parquet dataset in catalog"));
        let progress = Arc::new(Mutex::new(DownloadProgress::default()));
        let cancel = Arc::new(AtomicBool::new(false));
        let path = download(&ds, &temp, progress, cancel).expect("download failed");
        let text = std::fs::read_to_string(&path).expect("read result");
        assert!(text.len() > 1000, "text too short: {}", text.len());
        assert!(text.contains("\n"), "expected line breaks between articles");
        let _ = std::fs::remove_dir_all(&temp);
    }
}

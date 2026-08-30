//! Backend device selection: CPU, CUDA, or Metal, with automatic detection.

use anyhow::{Result, bail};
use candle_core::Device;

/// Resolve a device preference to a concrete Candle `Device`.
///
/// * `auto` (or empty) — pick the best available: CUDA, then Metal, then CPU.
/// * `cpu`, `cuda`, `metal` — force a specific backend (errors if missing).
pub fn pick_device(pref: &str) -> Result<Device> {
    match pref.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => {
            if candle_core::utils::cuda_is_available() {
                Ok(Device::cuda_if_available(0)?)
            } else if candle_core::utils::metal_is_available() {
                Ok(Device::metal_if_available(0)?)
            } else {
                Ok(Device::Cpu)
            }
        }
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::cuda_if_available(0)?),
        "metal" => Ok(Device::metal_if_available(0)?),
        other => bail!(
            "unknown device '{other}' — use one of: auto, cpu, cuda, metal"
        ),
    }
}

/// Human-readable device name for display.
pub fn describe(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA".to_string(),
        Device::Metal(_) => "Metal".to_string(),
    }
}

/// Backends actually usable on this machine, for display.
pub fn backend_report() -> String {
    let mut v = vec!["CPU"];
    if candle_core::utils::cuda_is_available() {
        v.push("CUDA");
    }
    if candle_core::utils::metal_is_available() {
        v.push("Metal");
    }
    v.join(" / ")
}

/// True when any GPU backend is present (CUDA or Metal).
pub fn is_gpu_available() -> bool {
    candle_core::utils::cuda_is_available() || candle_core::utils::metal_is_available()
}

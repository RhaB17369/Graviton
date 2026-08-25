//! Resource-aware capacity hints for running more than one model at once.
//!
//! Ollama itself already manages which models are actually resident in RAM
//! (it loads on demand and evicts least-recently-used ones under pressure) —
//! this module doesn't second-guess that. What it does is give an honest
//! answer to "can this machine actually run N models/agents at once?" so
//! `grv swarm` picks a sane default concurrency instead of either refusing
//! to help or firing off more concurrent generations than the machine can
//! hold, which would just thrash instead of running faster.
//!
//! Two different constraints, both real, neither fake-solved here:
//! - **RAM** bounds how many distinct models can be resident *at all* —
//!   this we estimate, from `/api/tags` sizes vs. total system memory.
//! - **CPU** bounds how fast concurrent generations run *once resident* —
//!   on a 6C/12T laptop with no big GPU, three agents generating at once
//!   split the same cores three ways; that's still a net win when the
//!   agents are independent (wall-clock for the batch drops), but each
//!   individual answer streams slower than it would alone. We don't hide
//!   that trade-off; `grv swarm` states it up front.

use graviton_llm::OllamaClient;

/// Fraction of total system RAM we're willing to assume is available for
/// resident models, leaving headroom for the OS, GRAVITON itself, and the
/// KV cache growth that happens as a conversation gets longer.
const RAM_SAFETY_FRACTION: f64 = 0.7;

pub struct Capacity {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub cpu_threads: usize,
}

pub fn detect() -> Capacity {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    Capacity {
        total_ram_mb: sys.total_memory() / 1024 / 1024,
        available_ram_mb: sys.available_memory() / 1024 / 1024,
        cpu_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
    }
}

/// How many of `models` (deduplicated model tags) could realistically be
/// resident at once, given `cap`. Models not yet pulled (no size known)
/// are assumed to cost `fallback_mb` each — better to under-promise than
/// to recommend a concurrency Ollama will just have to evict its way out of.
pub async fn safe_concurrency(client: &OllamaClient, models: &[&str], cap: &Capacity) -> (usize, String) {
    let sizes = model_sizes_mb(client).await;
    let fallback_mb: u64 = 5000; // ~an 8B Q4 model, the documented sweet spot for this class of hardware
    let mut costs: Vec<u64> = models
        .iter()
        .map(|m| sizes.get(*m).copied().unwrap_or(fallback_mb))
        .collect();
    costs.sort_unstable();

    let budget = (cap.total_ram_mb as f64 * RAM_SAFETY_FRACTION) as u64;
    let mut resident = 0usize;
    let mut used = 0u64;
    for c in &costs {
        if used + c > budget {
            break;
        }
        used += c;
        resident += 1;
    }
    resident = resident.max(1).min(models.len().max(1));

    let note = format!(
        "{} MB total RAM ({} MB available now), {} CPU threads → \
         ~{resident}/{} model(s) can realistically stay resident (RAM-bound); \
         CPU threads are shared across whatever runs concurrently, so wall-clock \
         improves for a *batch* of independent agents, not the speed of any one of them.",
        cap.total_ram_mb, cap.available_ram_mb, cap.cpu_threads, models.len(),
    );
    (resident, note)
}

/// Best-effort model-size lookup via `/api/tags` (already-pulled models
/// only — Ollama doesn't expose remote size without downloading).
async fn model_sizes_mb(client: &OllamaClient) -> std::collections::HashMap<String, u64> {
    client.model_sizes_mb().await.unwrap_or_default()
}

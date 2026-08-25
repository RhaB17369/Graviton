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
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Fraction of total system RAM we're willing to assume is available for
/// resident models, leaving headroom for the OS, GRAVITON itself, and the
/// KV cache growth that happens as a conversation gets longer.
const RAM_SAFETY_FRACTION: f64 = 0.7;

/// ~an 8B Q4 model, the documented sweet spot for this class of hardware —
/// used as the size estimate for a model that hasn't been pulled yet, so
/// there's still a number to plan around (better to under-promise than to
/// recommend a concurrency Ollama has to evict its way out of).
const FALLBACK_MODEL_MB: u64 = 5000;

/// Concurrent requests Ollama will actually serve against *one* resident
/// model before queueing the rest (its own `OLLAMA_NUM_PARALLEL`, which
/// defaults in this range on unconfigured installs). This is why
/// concurrency isn't capped at "number of distinct models fitting in RAM":
/// five agents that all happen to share one resident model can still run
/// several requests at once against it — RAM decides how many *different*
/// models can be resident, this decides how many callers can share one.
const ASSUMED_PARALLEL_PER_MODEL: usize = 3;

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

/// The heaviest processes on the machine right now by resident memory —
/// "what's actually eating the RAM headroom decisions above are based on",
/// surfaced so a human (or a log) can see it, not just trust a number.
pub fn top_memory_consumers(n: usize) -> Vec<(String, u64)> {
    use sysinfo::{ProcessesToUpdate, System};
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let mut procs: Vec<(String, u64)> = sys
        .processes()
        .values()
        .map(|p| (p.name().to_string_lossy().into_owned(), p.memory() / 1024 / 1024))
        .filter(|(_, mb)| *mb > 0)
        .collect();
    procs.sort_by(|a, b| b.1.cmp(&a.1));
    procs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    procs.truncate(n);
    procs
}

/// Best-effort model-size lookup via `/api/tags` (already-pulled models
/// only — Ollama doesn't expose remote size without downloading).
pub async fn model_sizes_mb(client: &OllamaClient) -> HashMap<String, u64> {
    client.model_sizes_mb().await.unwrap_or_default()
}

/// Pure computation, no I/O: how many of `models` (deduplicated model tags)
/// fit resident at once given `cap` and known/estimated `sizes`. Split out
/// from the network fetch so a live scheduler can recompute this every tick
/// against fresh `Capacity` without re-querying Ollama each time.
pub fn pick_concurrency(cap: &Capacity, models: &[&str], sizes: &HashMap<String, u64>) -> (usize, String) {
    let mut costs: Vec<u64> = models.iter().map(|m| sizes.get(*m).copied().unwrap_or(FALLBACK_MODEL_MB)).collect();
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
    resident = resident.clamp(1, models.len().max(1));
    // RAM bounds how many *distinct* models can be resident at once;
    // ASSUMED_PARALLEL_PER_MODEL bounds how many callers can share one
    // already-resident model. Multiple agents on the same tier/model is
    // the common case (see the tier design), so concurrency shouldn't
    // collapse to 1 just because they're all one model.
    let concurrency = (resident * ASSUMED_PARALLEL_PER_MODEL).max(1);

    let note = format!(
        "{} MB total RAM ({} MB available now), {} CPU threads → ~{resident}/{} distinct \
         model(s) can stay resident (RAM-bound) × ~{ASSUMED_PARALLEL_PER_MODEL} concurrent \
         requests/model ≈ {concurrency} concurrent call(s); CPU threads are shared across \
         whatever runs concurrently, so wall-clock improves for a *batch* of independent \
         agents, not the speed of any one of them.",
        cap.total_ram_mb, cap.available_ram_mb, cap.cpu_threads, models.len(),
    );
    (concurrency, note)
}

/// One-shot version of `pick_concurrency` for callers that just want an
/// answer now (`grv status`, `grv swarm`'s startup estimate) and don't need
/// to keep adjusting — fetches model sizes fresh, computes once.
pub async fn safe_concurrency(client: &OllamaClient, models: &[&str], cap: &Capacity) -> (usize, String) {
    let sizes = model_sizes_mb(client).await;
    pick_concurrency(cap, models, &sizes)
}

/// A concurrency gate whose pool size is continuously resampled from live
/// system headroom instead of fixed once at startup — so a long-running
/// `grv mission` (which can spawn agents recursively, unpredictable in
/// count and timing) never holds more concurrent model calls than the
/// machine can actually take, and can *grow* back into headroom that frees
/// up as earlier agents finish, without a human tuning `--max-parallel`.
///
/// Model sizes are fetched once at spawn time (they don't change mid-run);
/// only the cheap, local `Capacity::detect()` (sysinfo, no network) is
/// re-sampled each tick. This is deliberately *not* reimplementing Ollama's
/// own model residency/eviction — it exists only to decide how many
/// concurrent requests to have in flight before asking Ollama to do
/// anything, using the same RAM-based estimate `grv status`/`swarm` show.
pub struct LiveScheduler {
    sem: Arc<tokio::sync::Semaphore>,
    current_target: AtomicUsize,
    hard_cap: usize,
}

impl LiveScheduler {
    /// Spawn the background resampler and return the scheduler. `hard_cap`
    /// bounds the pool regardless of headroom (e.g. don't hold more permits
    /// than there is pending work, or a user-supplied `--max-parallel`).
    pub fn spawn(models: Vec<String>, sizes: HashMap<String, u64>, hard_cap: usize) -> Arc<Self> {
        let hard_cap = hard_cap.max(1);
        let initial = pick_concurrency(&detect(), &models.iter().map(String::as_str).collect::<Vec<_>>(), &sizes)
            .0
            .min(hard_cap);
        let sem = Arc::new(tokio::sync::Semaphore::new(initial));
        let this = Arc::new(Self { sem: sem.clone(), current_target: AtomicUsize::new(initial), hard_cap });

        let bg_sem = sem.clone();
        let bg_this = this.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(3)).await;
                let cap = detect();
                let model_refs: Vec<&str> = models.iter().map(String::as_str).collect();
                let (target, _) = pick_concurrency(&cap, &model_refs, &sizes);
                let target = target.min(bg_this.hard_cap);
                let current = bg_this.current_target.load(Ordering::Relaxed);
                if target > current {
                    bg_sem.add_permits(target - current);
                } else if target < current {
                    bg_sem.forget_permits(current - target);
                }
                bg_this.current_target.store(target, Ordering::Relaxed);
            }
        });

        this
    }

    /// Wait for a permit to run one agent/model call. Never blocks the
    /// caller from *scheduling* other work — it just gates how many run
    /// concurrently, which is the actual point: cheap work queues up
    /// instantly, expensive work waits its turn instead of piling onto RAM.
    pub async fn acquire(self: &Arc<Self>) -> tokio::sync::OwnedSemaphorePermit {
        self.sem.clone().acquire_owned().await.expect("scheduler semaphore is never closed")
    }

    pub fn current_target(&self) -> usize {
        self.current_target.load(Ordering::Relaxed)
    }
}

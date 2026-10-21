//! `grv models` — inspect and edit the per-tier/per-agent-key model
//! routing in `~/.config/graviton/models.toml` (see
//! `graviton_core::ModelRouting`).
//!
//! Three ways to change what runs where:
//! - hand-edit `models.toml` directly (it's a plain text file, documented
//!   inline);
//! - `grv models set <agent-key-or-tier> <model>` / `unset` for a scripted
//!   or one-line change;
//! - `grv models recommend`, which inspects what's actually pulled in
//!   Ollama plus this machine's RAM and writes a full routing sized to run
//!   without thrashing into swap — the "dynamic" configuration path.
//!
//! None of this is required: an untouched install has no `models.toml` at
//! all, and every agent falls straight through to `config.toml`'s
//! `model`/`model_fast`/`model_deep` exactly as before this file existed.

use anyhow::{Context, Result};
use graviton_core::{Config, ModelRouting, ModelTier};
use graviton_llm::OllamaClient;

use crate::agents;

/// Safety fraction of total RAM a *single* recommended model's on-disk
/// size is allowed to consume — same number `resources::safe_concurrency`
/// uses, kept in sync here rather than re-derived, since it encodes the
/// same "don't crash the machine" judgment call.
const RAM_SAFETY_FRACTION: f64 = 0.7;

pub async fn cmd_models_show(cfg: &Config) -> Result<()> {
    let client = OllamaClient::new(&cfg.ollama_host);
    let pulled = client.list_models().await.unwrap_or_default();

    println!("model routing (agent.<key> > tier.<fast|standard|deep> > config.toml fallback)\n");
    if cfg.routing.tiers.is_empty() && cfg.routing.agents.is_empty() {
        println!(
            "no overrides set — every agent runs on config.toml's model/model_fast/model_deep.\n\
             Try `grv models recommend` to generate one from what's pulled in Ollama.\n"
        );
    }

    let mut last_tier: Option<ModelTier> = None;
    for a in agents::ALL_AGENTS {
        if last_tier != Some(a.tier) {
            println!("[{}]", a.tier.label());
            last_tier = Some(a.tier);
        }
        let resolved = cfg.model_for(a.tier, a.key);
        let flag = match cfg.resolve_model(resolved) {
            graviton_core::ResolvedModel::Cloud { provider_name, provider, .. } => {
                if std::env::var(&provider.api_key_env).is_ok() {
                    format!(" \x1b[36m(cloud: {provider_name})\x1b[0m")
                } else {
                    format!(" \x1b[31m(cloud: {provider_name}, but ${} isn't set)\x1b[0m", provider.api_key_env)
                }
            }
            graviton_core::ResolvedModel::Local { .. } if pulled.is_empty() => String::new(),
            graviton_core::ResolvedModel::Local { .. } if pulled.iter().any(|m| m == resolved) => String::new(),
            graviton_core::ResolvedModel::Local { .. } => " \x1b[31m(not pulled — `ollama pull` it or `grv models set` a different one)\x1b[0m".to_string(),
        };
        let source = if cfg.routing.agents.contains_key(a.key) {
            "agent override"
        } else if cfg.routing.tiers.contains_key(a.tier.label()) {
            "tier override"
        } else {
            "config.toml fallback"
        };
        println!("  {:<14} {:<40} [{}]{}", a.key, resolved, source, flag);
    }
    if pulled.is_empty() {
        println!("\n(couldn't reach Ollama to check which of these are actually pulled)");
    }
    Ok(())
}

pub fn cmd_models_set(key: &str, model: &str) -> Result<()> {
    let mut routing = ModelRouting::load()?;
    if agents::find(key).is_some() {
        routing.agents.insert(key.to_string(), model.to_string());
        routing.save()?;
        println!("agent.{key} = \"{model}\" — written to {}", ModelRouting::path()?.display());
    } else if matches!(key, "fast" | "standard" | "deep") {
        routing.tiers.insert(key.to_string(), model.to_string());
        routing.save()?;
        println!("tier.{key} = \"{model}\" — written to {}", ModelRouting::path()?.display());
    } else {
        anyhow::bail!(
            "'{key}' isn't an agent key or a tier name (fast/standard/deep) — \
             `grv agents` lists the valid agent keys"
        );
    }
    Ok(())
}

pub fn cmd_models_unset(key: &str) -> Result<()> {
    let mut routing = ModelRouting::load()?;
    let removed_agent = routing.agents.remove(key).is_some();
    let removed_tier = routing.tiers.remove(key).is_some();
    if !removed_agent && !removed_tier {
        anyhow::bail!("no override set for '{key}' — nothing to unset (see `grv models`)");
    }
    routing.save()?;
    println!("removed — {key} now falls through to the next level (see `grv models`)");
    Ok(())
}

/// A pulled model, classified by a best-effort pattern match on its tag —
/// never a guarantee, always overridable by hand (`grv models set`), but a
/// reasonable starting point given what model authors actually name their
/// releases.
#[derive(Debug)]
struct Candidate {
    tag: String,
    size_mb: u64,
    kind: Kind,
}

#[derive(Debug, PartialEq, Eq)]
enum Kind {
    /// Small/fast, code-tuned — e.g. a 1.5B-3B "coder" model.
    CodeFast,
    /// Larger code-tuned model — the general standard-tier workhorse.
    CodeStandard,
    /// Vision/multimodal — not wired into any text agent today, so
    /// surfaced but not assigned.
    Vision,
    /// Security-focused, not deliberately uncensored — a reasonable
    /// defensive-deep pick (crypto review, hardening analysis).
    SecurityDefensive,
    /// Explicitly uncensored/abliterated — reserved for the
    /// offensive-deep agents (REAPER/BINEXP) that legitimately need to
    /// produce exploit/payload code without refusals getting in the way
    /// of an authorized engagement; never auto-picked for anything else.
    SecurityOffensive,
    /// A hosted `:cloud` tag — not local, contradicts this project's
    /// "nothing leaves the machine" premise, so never auto-assigned.
    Cloud,
    /// Doesn't match any recognized pattern — still usable, just not
    /// something `recommend` has an opinion about.
    Unclassified,
}

fn classify(tag: &str) -> Kind {
    let t = tag.to_lowercase();
    if t.ends_with(":cloud") || t.contains(":cloud") {
        return Kind::Cloud;
    }
    if t.contains("obliterat") || t.contains("abliterat") || t.contains("uncensored") {
        return Kind::SecurityOffensive;
    }
    if t.contains("deephat") || t.contains("cyber") || t.contains("whiterabbit") || t.contains("pentest") {
        return Kind::SecurityDefensive;
    }
    // Covers both the hyphenated official naming ("qwen2.5-vl") and the
    // compressed tag Ollama's library actually ships ("qwen2.5vl:7b" —
    // "vl" glued straight onto the version number, no separator before
    // it, caught `-vl`/`:vl` missing it on a real pulled model).
    if t.contains("-vl") || t.contains(":vl") || t.contains("vl:") || t.contains("vl-") || t.contains("vision") || t.contains("llava") || t.contains("minicpm-v") || t.contains("pixtral") {
        return Kind::Vision;
    }
    if t.contains("coder") || t.contains("code") {
        // Ollama tags embed the parameter count (e.g. "qwen2.5-coder:3b",
        // "...-7b") far more reliably than model size alone would let us
        // guess fast-vs-standard, so read it from the tag first and only
        // fall back to the size heuristic when that fails.
        if let Some(b) = t.rsplit(':').next().and_then(|v| v.trim_end_matches('b').parse::<f32>().ok()) {
            return if b <= 3.5 { Kind::CodeFast } else { Kind::CodeStandard };
        }
        return Kind::CodeStandard;
    }
    Kind::Unclassified
}

/// Inspect what's actually pulled in Ollama plus this machine's RAM, and
/// write a full `models.toml` sized to run without thrashing into swap.
/// Always safe to re-run — it overwrites the whole file, so a manual edit
/// since the last `recommend` would be lost (the command says so).
pub async fn cmd_models_recommend(cfg: &Config) -> Result<()> {
    let client = OllamaClient::new(&cfg.ollama_host);
    let sizes = client.model_sizes_mb().await.context("listing pulled models from Ollama — is `ollama serve` running?")?;
    if sizes.is_empty() {
        anyhow::bail!("no models pulled in Ollama yet — `ollama pull <model>` at least one first");
    }

    let total_ram_mb = crate::resources::detect().total_ram_mb;
    let budget_mb = (total_ram_mb as f64 * RAM_SAFETY_FRACTION) as u64;

    let mut candidates: Vec<Candidate> = Vec::new();
    let mut skipped_too_big: Vec<(String, u64)> = Vec::new();
    for (tag, size_mb) in sizes {
        if size_mb > budget_mb {
            skipped_too_big.push((tag, size_mb));
            continue;
        }
        let kind = classify(&tag);
        candidates.push(Candidate { tag, size_mb, kind });
    }
    candidates.sort_by_key(|c| c.size_mb);

    let pick = |kind: Kind| -> Option<&Candidate> { candidates.iter().filter(|c| c.kind == kind).max_by_key(|c| c.size_mb) };
    let pick_smallest = |kind: Kind| -> Option<&Candidate> { candidates.iter().filter(|c| c.kind == kind).min_by_key(|c| c.size_mb) };

    let fast = pick_smallest(Kind::CodeFast).or_else(|| candidates.iter().min_by_key(|c| c.size_mb));
    let standard = pick(Kind::CodeStandard).or(fast);
    let defensive_deep = pick(Kind::SecurityDefensive);
    let offensive_deep = pick(Kind::SecurityOffensive);
    // Design-deep (ARCHITECT/CLOUDARCH): prefer the largest otherwise-
    // unclassified generalist (distinct from the code-tuned standard
    // model, broader reasoning being more the point for design than code
    // completion), falling back to `standard` if nothing else fits.
    let design_deep = pick(Kind::Unclassified).or(standard);

    // CRYPTOGRAPHER/QUANTUMCRYPTO/REAPER/BINEXP used to get auto-assigned
    // to `defensive_deep`/`offensive_deep` -- the security-tagged picks.
    // Verified live, twice, why that's worse than it sounds: asked a
    // read_file-requiring question, REAPER on the uncensored pick threw a
    // hard `HTTP 500` from Ollama on the *second* turn every time
    // (`Jinja Exception: No user query found in messages` -- its chat
    // template, inspected via `ollama show --template`, turned out to be
    // a vision-model template with multi-step-tool macros that don't
    // match how Ollama structures a generic tool-result turn, not a
    // template meant for this model's normal use at all). The
    // "defensive"-tagged pick doesn't crash, but never actually calls a
    // tool either -- `tool_calls` came back empty every time and a
    // tool-call-shaped JSON blob showed up as plain text instead, meaning
    // Ollama's chosen template for it doesn't support function-calling
    // signaling. Every agent here reaches the model through the same
    // tool loop everything else uses (`read_file`/`search_code`/
    // `web_search`/...) specifically because that's what makes answers
    // grounded instead of guessed -- a model that silently can't call
    // tools, or hard-crashes the moment it tries, is a worse default than
    // one that can, even if (per the one-shot, --file-only comparison
    // that motivated routing crypto agents here in the first place) it's
    // more careful when it happens to stay within a single turn. Neither
    // is assigned by default any more; both are still recognized and
    // surfaced below so a one-shot/--file-heavy workflow can opt in with
    // `grv models set`.

    let mut routing = ModelRouting::default();
    if let Some(c) = fast {
        routing.tiers.insert("fast".to_string(), c.tag.clone());
    }
    if let Some(c) = standard {
        routing.tiers.insert("standard".to_string(), c.tag.clone());
    }
    if let Some(c) = design_deep {
        routing.tiers.insert("deep".to_string(), c.tag.clone());
        routing.agents.insert("architect".to_string(), c.tag.clone());
        routing.agents.insert("cloudarch".to_string(), c.tag.clone());
    }
    routing.save()?;

    println!("wrote {}\n", ModelRouting::path()?.display());
    println!("fast / standard (everything not called out below):");
    if let Some(c) = fast {
        println!("  tier.fast     = {} ({} MB)", c.tag, c.size_mb);
    }
    if let Some(c) = standard {
        println!("  tier.standard = {} ({} MB)", c.tag, c.size_mb);
    }
    println!(
        "\ndeep (architect, cloudarch -- everyone else here, including \
         cryptographer/quantumcrypto/reaper/binexp, shares this too; see below for why):"
    );
    if let Some(c) = design_deep {
        println!("  {} ({} MB)", c.tag, c.size_mb);
    }
    if let Some(c) = defensive_deep {
        println!(
            "\nfound but NOT auto-assigned — {} ({} MB): recognized as security-tagged, but it \
             never actually signals a tool call through Ollama (tool_calls came back empty and a \
             tool-call-shaped JSON blob leaked into plain text instead) when tested live. Usable \
             for a one-shot/--file-only question with `grv models set <agent> {0}`, not recommended \
             for anything that leans on read_file/search_code/web_search.",
            c.tag, c.size_mb
        );
    }
    if let Some(c) = offensive_deep {
        println!(
            "\nfound but NOT auto-assigned — {} ({} MB): recognized as uncensored (use only in \
             authorized engagements, same as any other offensive-security tooling here), but its \
             chat template hard-crashes Ollama (`HTTP 500`, confirmed reproducible) the instant a \
             second turn follows a tool call. More careful than the alternatives in a one-shot \
             test with no tool calls (see git log), but that makes it a `--file`-only tool, not a \
             safe default for `ask`/`run`'s normal tool loop. Opt in per agent with `grv models \
             set <agent> {0}` if you mainly feed it complete context up front.",
            c.tag, c.size_mb
        );
    }

    let vision: Vec<&Candidate> = candidates.iter().filter(|c| c.kind == Kind::Vision).collect();
    if !vision.is_empty() {
        println!("\nnot assigned (vision/multimodal — no text agent calls on one yet):");
        for c in vision {
            println!("  {} ({} MB)", c.tag, c.size_mb);
        }
    }
    let unused_cloud: Vec<&Candidate> = candidates.iter().filter(|c| c.kind == Kind::Cloud).collect();
    if !unused_cloud.is_empty() {
        println!("\nnot assigned (hosted :cloud tags — not local, left out on purpose):");
        for c in unused_cloud {
            println!("  {}", c.tag);
        }
    }
    if !skipped_too_big.is_empty() {
        println!(
            "\nskipped (bigger than {budget_mb} MB, {:.0}% of this machine's {total_ram_mb} MB RAM — would risk swapping):",
            RAM_SAFETY_FRACTION * 100.0
        );
        for (tag, mb) in &skipped_too_big {
            println!("  {tag} ({mb} MB)");
        }
    }
    println!(
        "\nOnly one model is resident for a single `grv ask`; `grv swarm`/`grv mission` can hold \
         several at once — check `grv status` before firing a wide swarm. Override anything \
         above with `grv models set <key> <model>`."
    );
    Ok(())
}

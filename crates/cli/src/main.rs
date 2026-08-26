mod agentic;
mod agents;
mod browser;
mod checkpoint;
mod context;
mod custom_tools;
mod mission;
mod resources;
mod tools;
mod web;

use agents::AgentSpec;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use graviton_core::Config;
use graviton_llm::{ChatMessage, OllamaClient};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "grv", version, about = "GRAVITON — local multi-agent framework for high-level programming, infrastructure, and offensive & defensive security, powered by Ollama")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// (Re)index the current repo into .graviton/index.db
    Index {
        /// Repo root to index (defaults to the current directory)
        path: Option<PathBuf>,
        /// Wipe the existing index before indexing
        #[arg(long)]
        force: bool,
    },
    /// Full-text search over indexed code chunks
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Look up a symbol (function/struct/class/...) by name
    Symbol {
        name: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },
    /// Ask one agent a question, with retrieved code as context
    Ask {
        question: String,
        /// Explicit files to include in full (in addition to retrieval)
        #[arg(long = "file")]
        files: Vec<PathBuf>,
        /// Which agent answers: architect (default), sentinel, reaper, singularity
        #[arg(long, default_value = "architect")]
        agent: String,
    },
    /// Like `ask`, but with a structured multi-step analysis prompt
    Investigate {
        question: String,
        #[arg(long = "file")]
        files: Vec<PathBuf>,
        /// Which agent investigates: reaper (default), architect, sentinel, singularity
        #[arg(long, default_value = "reaper")]
        agent: String,
    },
    /// Run the full crew on one question: architect -> reaper -> sentinel -> singularity,
    /// each stage reading the previous agents' actual output.
    Crew {
        question: String,
        #[arg(long = "file")]
        files: Vec<PathBuf>,
        /// Comma-separated pipeline override, e.g. "architect,reaper"
        #[arg(long, default_value = "architect,reaper,sentinel,singularity")]
        agents: String,
    },
    /// List GRAVITON's agent roster
    Agents,
    /// Recursively decompose a large task across the agent roster: a
    /// planner call splits it into subtasks assigned to specialists, each
    /// subtask can decompose further up to --max-depth, and results are
    /// synthesized back up the tree. Every model call anywhere in the tree
    /// shares one live, RAM-resampled concurrency gate (see `grv status`).
    Mission {
        task: String,
        #[arg(long = "file")]
        files: Vec<PathBuf>,
        /// How many levels a subtask may recurse (default 2, hard ceiling 4)
        #[arg(long)]
        max_depth: Option<usize>,
        /// Cap on concurrent model calls in flight at once (default: auto-detected from RAM)
        #[arg(long)]
        max_parallel: Option<usize>,
    },
    /// Run several independent agents concurrently (no hand-off between
    /// them, unlike `crew`) — each on its own tier's model. Concurrency
    /// defaults to what this machine's RAM can hold resident at once;
    /// override with --max-parallel.
    Swarm {
        question: String,
        #[arg(long = "file")]
        files: Vec<PathBuf>,
        /// Comma-separated agent list, e.g. "sentinel,reaper,cryptographer"
        #[arg(long)]
        agents: String,
        /// Cap on concurrent agents (default: auto-detected from RAM)
        #[arg(long)]
        max_parallel: Option<usize>,
    },
    /// Autonomous agentic loop: the agent reads/writes/edits files, runs
    /// shell commands, and (with --browser) drives a headless browser,
    /// until the task is done. Writes/edits/deletes/shell calls are
    /// confirmed unless --yolo is set; file changes are checkpointed
    /// (`grv rollback` undoes them).
    Run {
        /// Required unless --continue is given a session with existing history
        /// (in which case this is treated as an additional instruction, or
        /// omitted entirely to just resume where it left off)
        task: Option<String>,
        #[arg(long = "file")]
        files: Vec<PathBuf>,
        #[arg(long, default_value = "architect")]
        agent: String,
        /// Skip confirmation prompts — full autonomy
        #[arg(long)]
        yolo: bool,
        /// Offer browser_navigate/eval/screenshot/console tools (launches headless Chromium on first use)
        #[arg(long)]
        browser: bool,
        /// Resume a previous session's conversation instead of starting fresh
        /// (most recent session unless --session is also given)
        #[arg(long = "continue")]
        resume: bool,
        /// Session id to resume (with --continue) — defaults to the most recent
        #[arg(long)]
        session: Option<String>,
    },
    /// List `grv run` checkpoint sessions
    Checkpoints,
    /// Show a session's latest self-reported plan (from `update_plan`)
    Plan {
        /// Session id from `grv checkpoints` (defaults to the most recent one)
        session: Option<String>,
    },
    /// Undo a `grv run` session's file changes (all of it, or back to one step)
    Rollback {
        /// Session id from `grv checkpoints` (defaults to the most recent one)
        session: Option<String>,
        /// Undo only steps after this one, instead of the whole session
        #[arg(long)]
        to: Option<u64>,
    },
    /// Show index stats and Ollama connectivity
    Status,
    /// Show or update the config file (~/.config/graviton/config.toml)
    Config {
        #[arg(long)]
        model: Option<String>,
        /// Smaller/faster model for ModelTier::Fast agents (unset = use `model` for everything)
        #[arg(long)]
        model_fast: Option<String>,
        /// Larger/stronger model for ModelTier::Deep agents (unset = use `model` for everything)
        #[arg(long)]
        model_deep: Option<String>,
        #[arg(long)]
        num_ctx: Option<usize>,
        #[arg(long)]
        host: Option<String>,
    },
    /// Run recon/security tools and index their output for `ask`/`search`
    Tool {
        #[command(subcommand)]
        action: ToolCommand,
    },
    /// Manage `grv run`'s user-defined tools (TOML files, no recompiling —
    /// see ARCHITECTURE.md for the format)
    Custom {
        #[command(subcommand)]
        action: CustomCommand,
    },
}

#[derive(Subcommand)]
enum CustomCommand {
    /// List every loaded custom tool (global + this project) and where it came from
    List,
    /// Scaffold a new custom tool at .graviton/tools/<name>.toml, ready to edit
    New { name: String },
    /// Print one custom tool's parsed definition (schema the model would see)
    Show { name: String },
}

#[derive(Subcommand)]
enum ToolCommand {
    /// Run a whitelisted tool (e.g. `grv tool run nmap -- -sV 10.10.10.5`),
    /// streaming its output live and indexing it under `tool://<name>#<id>`
    Run {
        tool: String,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Index output you already captured elsewhere (e.g. `cat scan.txt | grv tool ingest nmap "initial scan"`)
    Ingest {
        tool: String,
        #[arg(default_value = "")]
        label: String,
    },
    /// List whitelisted tools and recent runs
    List {
        #[arg(long, default_value_t = 15)]
        limit: usize,
    },
    /// Print a stored run's full output (from `grv tool run`/`ingest`)
    Show { id: i64 },
}

fn chrono_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn resolve_agent(key: &str) -> Result<&'static AgentSpec> {
    agents::find(key).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown agent '{key}' — run `grv agents` to see the roster ({})",
            agents::ALL_AGENTS.iter().map(|a| a.key).collect::<Vec<_>>().join(", ")
        )
    })
}

fn repo_root() -> Result<PathBuf> {
    std::env::current_dir().context("resolving current directory")
}

fn open_repo_db(cfg: &Config, root: &std::path::Path) -> Result<rusqlite::Connection> {
    let db_path = graviton_core::db_path_for(root, &cfg.index_dir)?;
    if !db_path.exists() {
        anyhow::bail!(
            "no index found at {} — run `grv index` first",
            db_path.display()
        );
    }
    graviton_core::open_db(&db_path)
}

/// Like `open_repo_db`, but creates an empty index if none exists yet —
/// used by `grv tool`, which is useful even before you've run `grv index`.
fn open_or_init_repo_db(cfg: &Config, root: &std::path::Path) -> Result<rusqlite::Connection> {
    let db_path = graviton_core::db_path_for(root, &cfg.index_dir)?;
    graviton_core::open_db(&db_path)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .without_time()
        .init();

    let cli = Cli::parse();
    let cfg = Config::load_or_init()?;

    match cli.cmd {
        Command::Index { path, force } => cmd_index(&cfg, path, force),
        Command::Search { query, limit } => cmd_search(&cfg, &query, limit),
        Command::Symbol { name, limit } => cmd_symbol(&cfg, &name, limit),
        Command::Ask { question, files, agent } => {
            let spec = resolve_agent(&agent)?;
            cmd_ask(&cfg, &question, files, spec, false).await
        }
        Command::Investigate { question, files, agent } => {
            let spec = resolve_agent(&agent)?;
            cmd_ask(&cfg, &question, files, spec, true).await
        }
        Command::Crew { question, files, agents } => cmd_crew(&cfg, &question, files, &agents).await,
        Command::Agents => {
            println!("{}", agents::list_text());
            Ok(())
        }
        Command::Swarm { question, files, agents, max_parallel } => {
            cmd_swarm(&cfg, &question, files, &agents, max_parallel).await
        }
        Command::Mission { task, files, max_depth, max_parallel } => {
            cmd_mission(&cfg, &task, files, max_depth, max_parallel).await
        }
        Command::Run { task, files, agent, yolo, browser, resume, session } => {
            let spec = resolve_agent(&agent)?;
            let resume_session = if resume {
                Some(match session {
                    Some(id) => id,
                    None => checkpoint::most_recent_session(&repo_root()?)?
                        .ok_or_else(|| anyhow::anyhow!("no checkpoint sessions to continue — run `grv run` once first"))?,
                })
            } else {
                None
            };
            if resume_session.is_none() && task.as_deref().unwrap_or("").trim().is_empty() {
                anyhow::bail!("a task is required unless --continue is given");
            }
            cmd_run(&cfg, &task.unwrap_or_default(), files, spec, yolo, browser, resume_session).await
        }
        Command::Checkpoints => cmd_checkpoints(),
        Command::Plan { session } => cmd_plan(session),
        Command::Rollback { session, to } => cmd_rollback(session, to),
        Command::Status => cmd_status(&cfg).await,
        Command::Config { model, model_fast, model_deep, num_ctx, host } => {
            cmd_config(model, model_fast, model_deep, num_ctx, host)
        }
        Command::Tool { action } => cmd_tool(&cfg, action),
        Command::Custom { action } => cmd_custom(action),
    }
}

fn cmd_index(cfg: &Config, path: Option<PathBuf>, force: bool) -> Result<()> {
    let root = path.map(Ok).unwrap_or_else(repo_root)?;
    let root = root.canonicalize().context("resolving repo path")?;
    let db_path = graviton_core::db_path_for(&root, &cfg.index_dir)?;
    let mut conn = graviton_core::open_db(&db_path)?;
    if force {
        graviton_core::clear_index(&conn)?;
    }
    println!("Indexing {} ...", root.display());
    let stats = graviton_indexer::index_repo(&mut conn, &root)?;
    println!(
        "done: {} files scanned, {} indexed, {} unchanged, {} symbols, {} chunks",
        stats.files_scanned,
        stats.files_indexed,
        stats.files_skipped_unchanged,
        stats.symbols_extracted,
        stats.chunks_written
    );
    Ok(())
}

fn cmd_search(cfg: &Config, query: &str, limit: usize) -> Result<()> {
    let root = repo_root()?;
    let conn = open_repo_db(cfg, &root)?;
    let blocks = context::search_chunks(&conn, query, limit)?;
    if blocks.is_empty() {
        println!("no matches");
        return Ok(());
    }
    for b in blocks {
        println!("\x1b[1;36m{}\x1b[0m", b.header);
        for line in b.body.lines().take(6) {
            println!("  {line}");
        }
        println!();
    }
    Ok(())
}

fn cmd_symbol(cfg: &Config, name: &str, limit: usize) -> Result<()> {
    let root = repo_root()?;
    let conn = open_repo_db(cfg, &root)?;
    let blocks = context::search_symbols(&conn, &root, name, limit)?;
    if blocks.is_empty() {
        println!("no symbol matching '{name}'");
        return Ok(());
    }
    for b in blocks {
        println!("\x1b[1;33m{}\x1b[0m", b.header);
        println!("{}\n", b.body);
    }
    Ok(())
}

/// Retrieve context for `question` (explicit files + symbol hits + FTS
/// chunks, budgeted to `cfg`'s context window) — shared by `ask`/
/// `investigate`/`crew` so every agent reasons over the same evidence.
fn build_context(cfg: &Config, root: &std::path::Path, conn: &rusqlite::Connection, question: &str, files: &[PathBuf]) -> Result<String> {
    let explicit: Vec<context::ContextBlock> = files
        .iter()
        .filter_map(|f| context::read_whole_file(root, f))
        .collect();
    let symbol_hits = context::search_symbols(conn, root, question, 8)?;
    let chunk_hits = context::search_chunks(conn, question, 12)?;
    let budget = cfg.context_char_budget();
    Ok(context::assemble(budget, vec![explicit, symbol_hits, chunk_hits]))
}

fn agent_system_prompt(agent: &AgentSpec, investigate: bool) -> String {
    if investigate {
        format!("{}{}", agent.system_prompt, agents::INVESTIGATE_FORMAT)
    } else {
        agent.system_prompt.to_string()
    }
}

/// Stream one agent's reply to stdout, returning the full text. This path
/// never offers tools — `ask`/`investigate`/`crew` are read-only analysis
/// over retrieved context. `grv run` (see `agentic.rs`) is the tool-using
/// agentic loop.
async fn run_agent(client: &OllamaClient, cfg: &Config, model: &str, system: &str, user_msg: &str) -> Result<String> {
    let messages = vec![ChatMessage::system(system), ChatMessage::user(user_msg)];
    let stdout = std::io::stdout();
    let result = client
        .chat_stream(model, &messages, cfg.num_ctx, &[], |tok| {
            let mut lock = stdout.lock();
            let _ = lock.write_all(tok.as_bytes());
            let _ = lock.flush();
        })
        .await?;
    println!();
    Ok(result.content)
}

async fn cmd_ask(cfg: &Config, question: &str, files: Vec<PathBuf>, agent: &AgentSpec, investigate: bool) -> Result<()> {
    let root = repo_root()?;
    let conn = open_repo_db(cfg, &root)?;
    let context_text = build_context(cfg, &root, &conn, question, &files)?;

    let system = agent_system_prompt(agent, investigate);
    let user_msg = if context_text.is_empty() {
        format!(
            "{question}\n\n(No indexed context matched — either run `grv index` first, \
             or this question doesn't map to specific code.)"
        )
    } else {
        format!("Question: {question}\n\nRetrieved context:\n{context_text}")
    };

    println!("\x1b[1;35m═══ {} ═══\x1b[0m", agent.display);
    let client = OllamaClient::new(&cfg.ollama_host);
    run_agent(&client, cfg, cfg.model_for_tier(agent.tier), &system, &user_msg).await.map(|_| ())
}

async fn cmd_crew(cfg: &Config, question: &str, files: Vec<PathBuf>, pipeline: &str) -> Result<()> {
    let root = repo_root()?;
    let conn = open_repo_db(cfg, &root)?;
    let context_text = build_context(cfg, &root, &conn, question, &files)?;
    let context_block = if context_text.is_empty() {
        "(No indexed context matched — either run `grv index` first, or this question doesn't map to specific code.)".to_string()
    } else {
        format!("Retrieved context:\n{context_text}")
    };

    let keys: Vec<&str> = pipeline.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if keys.is_empty() {
        anyhow::bail!("empty agent pipeline");
    }
    let mut specs = Vec::with_capacity(keys.len());
    for k in &keys {
        specs.push(resolve_agent(k)?);
    }

    let client = OllamaClient::new(&cfg.ollama_host);
    let mut prior = String::new(); // other agents' findings so far, fed to each subsequent stage

    for spec in specs {
        println!("\x1b[1;35m═══ {} — {} ═══\x1b[0m", spec.display, spec.tagline);
        let user_msg = if prior.is_empty() {
            format!("Question: {question}\n\n{context_block}")
        } else {
            format!(
                "Question: {question}\n\n{context_block}\n\n\
                 Findings from the rest of the crew so far:\n{prior}"
            )
        };
        let output = run_agent(&client, cfg, cfg.model_for_tier(spec.tier), spec.system_prompt, &user_msg).await?;
        prior.push_str(&format!("\n--- {} ---\n{}\n", spec.display, output.trim()));
        // Cap accrued findings so a long crew doesn't blow past num_ctx by
        // the final stage — keep the most recent agents' output, since
        // that's what the next stage (and the coordinator) most needs.
        let cap = cfg.context_char_budget();
        if prior.len() > cap {
            let mut cut = prior.len() - cap;
            while !prior.is_char_boundary(cut) {
                cut += 1;
            }
            prior = format!("[...earlier agents truncated...]\n{}", &prior[cut..]);
        }
        println!();
    }
    Ok(())
}

/// Run several agents concurrently on the same question, no hand-off
/// between them (unlike `crew`, where each stage reads the previous one's
/// output). Each agent calls its own tier's model. Concurrency is capped
/// by what `resources::safe_concurrency` estimates this machine's RAM can
/// hold resident at once, unless the user overrides it — running more
/// agents at once than RAM can hold just means Ollama thrashes, evicting
/// and reloading models between requests, which is slower than sequential.
async fn cmd_swarm(cfg: &Config, question: &str, files: Vec<PathBuf>, roster: &str, max_parallel: Option<usize>) -> Result<()> {
    let root = repo_root()?;
    let conn = open_repo_db(cfg, &root)?;
    let context_text = build_context(cfg, &root, &conn, question, &files)?;
    let context_block = if context_text.is_empty() {
        "(No indexed context matched — either run `grv index` first, or this question doesn't map to specific code.)".to_string()
    } else {
        format!("Retrieved context:\n{context_text}")
    };

    let keys: Vec<&str> = roster.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if keys.is_empty() {
        anyhow::bail!("empty agent roster — pass --agents a,b,c");
    }
    let mut specs = Vec::with_capacity(keys.len());
    for k in &keys {
        specs.push(resolve_agent(k)?);
    }

    let client = OllamaClient::new(&cfg.ollama_host);
    let models: Vec<String> = {
        let mut m: Vec<String> = specs.iter().map(|s| cfg.model_for_tier(s.tier).to_string()).collect();
        m.dedup();
        m
    };
    let sizes = resources::model_sizes_mb(&client).await;
    let hard_cap = max_parallel.unwrap_or(specs.len()).max(1);
    let scheduler = resources::LiveScheduler::spawn(models, sizes, hard_cap);
    if max_parallel.is_none() {
        println!(
            "\x1b[2mstarting concurrency ~{} (auto, re-sampled live from RAM as the swarm runs — see `grv status`)\x1b[0m\n",
            scheduler.current_target()
        );
    }

    let mut tasks = tokio::task::JoinSet::new();
    for spec in specs {
        let scheduler = scheduler.clone();
        let client = OllamaClient::new(&cfg.ollama_host);
        let model = cfg.model_for_tier(spec.tier).to_string();
        let num_ctx = cfg.num_ctx;
        let system = spec.system_prompt.to_string();
        let user_msg = format!("Question: {question}\n\n{context_block}");
        let display = spec.display;
        let tagline = spec.tagline;
        tasks.spawn(async move {
            let _permit = scheduler.acquire().await;
            let started = std::time::Instant::now();
            let result = client.chat_stream(&model, &[ChatMessage::system(system), ChatMessage::user(user_msg)], num_ctx, &[], |_| {}).await;
            (display, tagline, model, started.elapsed(), result)
        });
    }

    while let Some(joined) = tasks.join_next().await {
        let (display, tagline, model, elapsed, result) = joined.context("swarm task panicked")?;
        println!("\x1b[1;35m═══ {display} — {tagline} [{model}, {:.1}s] ═══\x1b[0m", elapsed.as_secs_f32());
        match result {
            Ok(r) => println!("{}\n", r.content.trim()),
            Err(e) => println!("(failed: {e})\n"),
        }
    }
    Ok(())
}

async fn cmd_mission(cfg: &Config, task: &str, files: Vec<PathBuf>, max_depth: Option<usize>, max_parallel: Option<usize>) -> Result<()> {
    let root = repo_root()?;
    let conn = open_repo_db(cfg, &root)?;
    let context_text = build_context(cfg, &root, &conn, task, &files)?;
    let context_block = if context_text.is_empty() {
        "(No indexed context matched — either run `grv index` first, or this task doesn't map to specific code.)".to_string()
    } else {
        format!("Retrieved context:\n{context_text}")
    };
    mission::run(cfg, &root, context_block, task, max_depth, max_parallel).await
}

async fn cmd_run(cfg: &Config, task: &str, files: Vec<PathBuf>, agent: &AgentSpec, yolo: bool, browser: bool, resume_session: Option<String>) -> Result<()> {
    let root = repo_root()?;
    let conn = open_or_init_repo_db(cfg, &root)?;
    let context_text = if resume_session.is_none() {
        build_context(cfg, &root, &conn, task, &files).unwrap_or_default()
    } else {
        String::new() // unused when resuming — the restored transcript already has the original context
    };
    let loop_cfg = agentic::AgentLoopConfig { auto_approve: yolo, enable_browser: browser };
    agentic::run(cfg, &conn, &root, agent, task, Some(context_text), loop_cfg, resume_session).await
}

fn cmd_checkpoints() -> Result<()> {
    let root = repo_root()?;
    let sessions = checkpoint::list_sessions(&root)?;
    if sessions.is_empty() {
        println!("no checkpoint sessions yet — they're created by `grv run`");
        return Ok(());
    }
    for s in sessions {
        println!("{:<20} {} step(s), {} file(s) touched", s.id, s.steps, s.files_touched);
    }
    Ok(())
}

fn cmd_plan(session: Option<String>) -> Result<()> {
    let root = repo_root()?;
    let session_id = match session {
        Some(s) => s,
        None => checkpoint::most_recent_session(&root)?
            .ok_or_else(|| anyhow::anyhow!("no checkpoint sessions yet — they're created by `grv run`"))?,
    };
    match checkpoint::load_plan(&root, &session_id)? {
        Some(plan) => println!("{}", agentic::format_plan(plan.get("steps").unwrap_or(&plan))),
        None => println!("session {session_id} has no saved plan (the agent never called update_plan)"),
    }
    Ok(())
}

fn cmd_rollback(session: Option<String>, to: Option<u64>) -> Result<()> {
    let root = repo_root()?;
    let session_id = match session {
        Some(s) => s,
        None => checkpoint::list_sessions(&root)?
            .into_iter()
            .last()
            .map(|s| s.id)
            .ok_or_else(|| anyhow::anyhow!("no checkpoint sessions to roll back"))?,
    };
    let undone = checkpoint::rollback(&root, &session_id, to)?;
    println!("rolled back {undone} change(s) from session {session_id}");
    Ok(())
}

async fn cmd_status(cfg: &Config) -> Result<()> {
    println!("config: {}", Config::config_path()?.display());
    println!("  ollama_host = {}", cfg.ollama_host);
    println!("  model       = {} (standard tier)", cfg.model);
    if let Some(m) = &cfg.model_fast {
        println!("  model_fast  = {m} (fast tier)");
    }
    if let Some(m) = &cfg.model_deep {
        println!("  model_deep  = {m} (deep tier)");
    }
    println!("  num_ctx     = {}", cfg.num_ctx);
    println!(
        "  context budget ≈ {} chars (~{} tokens)",
        cfg.context_char_budget(),
        cfg.context_char_budget() / 4
    );

    match repo_root().and_then(|r| open_repo_db(cfg, &r)) {
        Ok(conn) => {
            let files: i64 = conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
            let symbols: i64 = conn.query_row("SELECT COUNT(*) FROM symbols", [], |r| r.get(0))?;
            let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM content_fts", [], |r| r.get(0))?;
            println!("index: {files} files, {symbols} symbols, {chunks} chunks");
        }
        Err(e) => println!("index: {e}"),
    }

    let client = OllamaClient::new(&cfg.ollama_host);
    match client.list_models().await {
        Ok(models) => {
            println!("ollama: reachable, {} model(s) pulled", models.len());
            for m in models {
                let marker = if m == cfg.model { " <- selected" } else { "" };
                println!("  - {m}{marker}");
            }
        }
        Err(e) => println!("ollama: unreachable ({e})"),
    }

    let cap = resources::detect();
    let distinct = cfg.distinct_models();
    let (_, note) = resources::safe_concurrency(&client, &distinct, &cap).await;
    println!("swarm capacity: {note}");
    println!("heaviest processes on this machine right now (what that estimate is actually competing with):");
    for (name, mb) in resources::top_memory_consumers(5) {
        println!("  {name:<24} {mb} MB");
    }
    Ok(())
}

fn cmd_tool(cfg: &Config, action: ToolCommand) -> Result<()> {
    let root = repo_root()?;
    match action {
        ToolCommand::Run { tool, args } => {
            let conn = open_or_init_repo_db(cfg, &root)?;
            let id = tools::run_and_index(&conn, &tool, &args)?;
            println!(
                "\n[stored as tool run #{id} — try `grv ask \"analyze tool run #{id}\"` or `grv search '{tool}'`]"
            );
        }
        ToolCommand::Ingest { tool, label } => {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
                .context("reading tool output from stdin")?;
            if buf.trim().is_empty() {
                anyhow::bail!("no input on stdin — pipe tool output in, e.g. `cat scan.txt | grv tool ingest nmap`");
            }
            let conn = open_or_init_repo_db(cfg, &root)?;
            let id = tools::ingest(&conn, &tool, &label, &buf)?;
            println!("stored as tool run #{id} ({} lines)", buf.lines().count());
        }
        ToolCommand::List { limit } => {
            println!("whitelisted tools: {}", tools::ALLOWED_TOOLS.join(", "));
            let conn = open_or_init_repo_db(cfg, &root)?;
            let runs = tools::recent_runs(&conn, limit)?;
            if runs.is_empty() {
                println!("\nno tool runs recorded yet in this repo's index");
            } else {
                println!("\nrecent runs:");
                for r in runs {
                    let status = r
                        .exit_code
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "?".into());
                    let secs_ago = (chrono_now() - r.ran_at).max(0);
                    println!(
                        "  #{:<4} {:<12} exit={:<4} {}s ago  {}",
                        r.id, r.tool, status, secs_ago, r.args
                    );
                }
            }
        }
        ToolCommand::Show { id } => {
            let conn = open_or_init_repo_db(cfg, &root)?;
            let output: String = conn.query_row(
                "SELECT output FROM tool_runs WHERE id = ?1",
                [id],
                |r| r.get(0),
            ).with_context(|| format!("no tool run #{id}"))?;
            print!("{output}");
        }
    }
    Ok(())
}

fn cmd_custom(action: CustomCommand) -> Result<()> {
    let root = repo_root()?;
    match action {
        CustomCommand::List => {
            let tools = custom_tools::load_all(&root);
            if tools.is_empty() {
                println!(
                    "no custom tools loaded — put a .toml file in ~/.config/graviton/tools/ \
                     or .graviton/tools/ (try `grv custom new <name>`)"
                );
            }
            for t in tools {
                println!("{:<20} {}\n  {}\n  ({})", t.name, t.description, t.command, t.source.display());
            }
        }
        CustomCommand::New { name } => {
            let dir = root.join(".graviton").join("tools");
            std::fs::create_dir_all(&dir)?;
            let path = dir.join(format!("{name}.toml"));
            if path.exists() {
                anyhow::bail!("{} already exists", path.display());
            }
            std::fs::write(&path, custom_tools::scaffold(&name))?;
            println!("wrote {} — edit it, then `grv custom show {name}` to check it loads", path.display());
        }
        CustomCommand::Show { name } => {
            let tools = custom_tools::load_all(&root);
            match custom_tools::find(&tools, &name) {
                Some(t) => println!("{:#?}", t.to_tool_def()),
                None => anyhow::bail!("no loaded custom tool named '{name}' — `grv custom list` to see what's loaded"),
            }
        }
    }
    Ok(())
}

fn cmd_config(
    model: Option<String>,
    model_fast: Option<String>,
    model_deep: Option<String>,
    num_ctx: Option<usize>,
    host: Option<String>,
) -> Result<()> {
    let mut cfg = Config::load_or_init()?;
    let mut changed = false;
    if let Some(m) = model {
        cfg.model = m;
        changed = true;
    }
    if let Some(m) = model_fast {
        cfg.model_fast = (!m.is_empty()).then_some(m);
        changed = true;
    }
    if let Some(m) = model_deep {
        cfg.model_deep = (!m.is_empty()).then_some(m);
        changed = true;
    }
    if let Some(n) = num_ctx {
        cfg.num_ctx = n;
        changed = true;
    }
    if let Some(h) = host {
        cfg.ollama_host = h;
        changed = true;
    }
    if changed {
        cfg.save()?;
        println!("saved {}", Config::config_path()?.display());
    }
    println!("{cfg:#?}");
    Ok(())
}

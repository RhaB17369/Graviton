mod agentic;
mod agents;
mod browser;
mod checkpoint;
mod context;
mod tools;

use agents::AgentSpec;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use graviton_core::Config;
use graviton_llm::{ChatMessage, OllamaClient};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "grv", version, about = "GRAVITON — local multi-agent framework for high-level programming, defensive & offensive security, powered by Ollama")]
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
    /// Autonomous agentic loop: the agent reads/writes/edits files, runs
    /// shell commands, and (with --browser) drives a headless browser,
    /// until the task is done. Writes/edits/deletes/shell calls are
    /// confirmed unless --yolo is set; file changes are checkpointed
    /// (`grv rollback` undoes them).
    Run {
        task: String,
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
    },
    /// List `grv run` checkpoint sessions
    Checkpoints,
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
        Command::Run { task, files, agent, yolo, browser } => {
            let spec = resolve_agent(&agent)?;
            cmd_run(&cfg, &task, files, spec, yolo, browser).await
        }
        Command::Checkpoints => cmd_checkpoints(),
        Command::Rollback { session, to } => cmd_rollback(session, to),
        Command::Status => cmd_status(&cfg).await,
        Command::Config { model, num_ctx, host } => cmd_config(model, num_ctx, host),
        Command::Tool { action } => cmd_tool(&cfg, action),
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
async fn run_agent(client: &OllamaClient, cfg: &Config, system: &str, user_msg: &str) -> Result<String> {
    let messages = vec![ChatMessage::system(system), ChatMessage::user(user_msg)];
    let stdout = std::io::stdout();
    let result = client
        .chat_stream(&cfg.model, &messages, cfg.num_ctx, &[], |tok| {
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
    run_agent(&client, cfg, &system, &user_msg).await.map(|_| ())
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
        let output = run_agent(&client, cfg, spec.system_prompt, &user_msg).await?;
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

async fn cmd_run(cfg: &Config, task: &str, files: Vec<PathBuf>, agent: &AgentSpec, yolo: bool, browser: bool) -> Result<()> {
    let root = repo_root()?;
    let conn = open_or_init_repo_db(cfg, &root)?;
    let context_text = build_context(cfg, &root, &conn, task, &files).unwrap_or_default();
    let loop_cfg = agentic::AgentLoopConfig { auto_approve: yolo, enable_browser: browser };
    agentic::run(cfg, &conn, &root, agent, task, Some(context_text), loop_cfg).await
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
    println!("  model       = {}", cfg.model);
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

fn cmd_config(model: Option<String>, num_ctx: Option<usize>, host: Option<String>) -> Result<()> {
    let mut cfg = Config::load_or_init()?;
    let mut changed = false;
    if let Some(m) = model {
        cfg.model = m;
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

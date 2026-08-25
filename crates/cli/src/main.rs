mod context;
mod prompts;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use graviton_core::Config;
use graviton_llm::{ChatMessage, OllamaClient};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "grv", version, about = "GRAVITON — local code-intelligence & offensive-security copilot, powered by Ollama")]
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
    /// Ask the model a question, with retrieved code as context
    Ask {
        question: String,
        /// Explicit files to include in full (in addition to retrieval)
        #[arg(long = "file")]
        files: Vec<PathBuf>,
    },
    /// Like `ask`, but with a structured multi-step analysis prompt
    Investigate {
        question: String,
        #[arg(long = "file")]
        files: Vec<PathBuf>,
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
        Command::Ask { question, files } => cmd_ask(&cfg, &question, files, false).await,
        Command::Investigate { question, files } => cmd_ask(&cfg, &question, files, true).await,
        Command::Status => cmd_status(&cfg).await,
        Command::Config { model, num_ctx, host } => cmd_config(model, num_ctx, host),
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

async fn cmd_ask(cfg: &Config, question: &str, files: Vec<PathBuf>, investigate: bool) -> Result<()> {
    let root = repo_root()?;
    let conn = open_repo_db(cfg, &root)?;

    let explicit: Vec<context::ContextBlock> = files
        .iter()
        .filter_map(|f| context::read_whole_file(&root, f))
        .collect();
    let symbol_hits = context::search_symbols(&conn, &root, question, 8)?;
    let chunk_hits = context::search_chunks(&conn, question, 12)?;

    let budget = cfg.context_char_budget();
    let context_text = context::assemble(budget, vec![explicit, symbol_hits, chunk_hits]);

    let system = if investigate { prompts::SYSTEM_INVESTIGATE } else { prompts::SYSTEM_ASK };
    let user_msg = if context_text.is_empty() {
        format!(
            "{question}\n\n(No indexed context matched — either run `grv index` first, \
             or this question doesn't map to specific code.)"
        )
    } else {
        format!("Question: {question}\n\nRetrieved context:\n{context_text}")
    };

    let client = OllamaClient::new(&cfg.ollama_host);
    let messages = vec![ChatMessage::system(system), ChatMessage::user(user_msg)];

    let stdout = std::io::stdout();
    let result = client
        .chat_stream(&cfg.model, &messages, cfg.num_ctx, |tok| {
            let mut lock = stdout.lock();
            let _ = lock.write_all(tok.as_bytes());
            let _ = lock.flush();
        })
        .await;
    println!();
    result.map(|_| ())
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

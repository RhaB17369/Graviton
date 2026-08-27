//! graviton-core: config, database schema, and shared types for GRAVITON.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Which model an agent should run on. `Standard` is always `Config::model`
/// — the other two are optional overrides (`None` falls back to `model`),
/// so a config file with none of this set behaves exactly as before: one
/// model for everything. Set `model_fast`/`model_deep` to actually spread
/// work across differently-sized models (see `Config::model_for_tier`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    /// Cheap, bounded reasoning (pattern-matching, mechanical rewrites) —
    /// a small model (1.5B-3B) is usually enough and frees RAM for others.
    Fast,
    /// Default: whatever `Config::model` is.
    Standard,
    /// Reasoning that benefits from the biggest model that still fits —
    /// exploit dev, crypto correctness, architecture decisions.
    Deep,
}

/// Runtime configuration, loaded from `~/.config/graviton/config.toml`
/// (created with sane defaults on first run).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Base URL of the local Ollama daemon.
    pub ollama_host: String,
    /// Model tag to use for chat/generation, e.g. "qwen3:8b". Also the
    /// `Standard` tier and the fallback for `Fast`/`Deep` when unset.
    pub model: String,
    /// Optional smaller/faster model for `ModelTier::Fast` agents. Running
    /// this alongside `model` means two models resident at once — only set
    /// it if there's RAM to spare (see `Config::model_for_tier`).
    #[serde(default)]
    pub model_fast: Option<String>,
    /// Optional larger/stronger model for `ModelTier::Deep` agents.
    #[serde(default)]
    pub model_deep: Option<String>,
    /// Optional embedding model (e.g. "nomic-embed-text", "all-minilm") for
    /// semantic search. Unset means semantic search is simply unavailable —
    /// `grv search`/`grv ask`/etc. fall back to lexical FTS exactly as
    /// before, so this is opt-in and never a behavior change by itself.
    #[serde(default)]
    pub embed_model: Option<String>,
    /// Context window requested from the model (must fit in RAM as KV cache).
    pub num_ctx: usize,
    /// Fraction of num_ctx reserved for injected code context (0.0-1.0).
    /// The rest is left for the system prompt, the question, and the answer.
    pub context_budget_fraction: f32,
    /// Directory name (relative to a repo root) holding the SQLite index.
    pub index_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ollama_host: "http://127.0.0.1:11434".to_string(),
            model: "qwen3:8b".to_string(),
            model_fast: None,
            model_deep: None,
            embed_model: None,
            num_ctx: 8192,
            context_budget_fraction: 0.55,
            index_dir: ".graviton".to_string(),
        }
    }
}

impl Config {
    pub fn config_path() -> Result<PathBuf> {
        let base = dirs::config_dir().context("no config dir on this platform")?;
        Ok(base.join("graviton").join("config.toml"))
    }

    /// Load config from disk, writing defaults if none exists yet.
    pub fn load_or_init() -> Result<Self> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            let cfg = Self::default();
            cfg.save()?;
            return Ok(cfg);
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: Config = toml_from_str(&raw)?;
        Ok(cfg)
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let raw = toml_to_string(self)?;
        std::fs::write(&path, raw)?;
        Ok(())
    }

    /// Approximate token budget (chars/4 heuristic) available for injected
    /// code context, leaving headroom for system prompt + question + answer.
    pub fn context_char_budget(&self) -> usize {
        let tokens = (self.num_ctx as f32 * self.context_budget_fraction) as usize;
        tokens * 4
    }

    /// Resolve which model tag an agent in the given tier should call.
    /// `Fast`/`Deep` fall back to `model` when no override is configured,
    /// so an untouched config runs everything on one model exactly as v0.4
    /// did — tiering is opt-in.
    pub fn model_for_tier(&self, tier: ModelTier) -> &str {
        match tier {
            ModelTier::Fast => self.model_fast.as_deref().unwrap_or(&self.model),
            ModelTier::Standard => &self.model,
            ModelTier::Deep => self.model_deep.as_deref().unwrap_or(&self.model),
        }
    }

    /// The distinct model tags this config would actually call across all
    /// three tiers (deduplicated) — used to size concurrent-capacity
    /// estimates without guessing at agent assignments.
    pub fn distinct_models(&self) -> Vec<&str> {
        let mut out = vec![self.model.as_str()];
        for m in [self.model_fast.as_deref(), self.model_deep.as_deref()].into_iter().flatten() {
            if !out.contains(&m) {
                out.push(m);
            }
        }
        out
    }
}

// Minimal hand-rolled TOML (de)serialization so we don't pull in the `toml`
// crate just for a handful of scalar fields.
fn toml_to_string(cfg: &Config) -> Result<String> {
    let mut out = format!(
        "ollama_host = \"{}\"\nmodel = \"{}\"\n",
        cfg.ollama_host, cfg.model
    );
    if let Some(m) = &cfg.model_fast {
        out.push_str(&format!("model_fast = \"{m}\"\n"));
    }
    if let Some(m) = &cfg.model_deep {
        out.push_str(&format!("model_deep = \"{m}\"\n"));
    }
    if let Some(m) = &cfg.embed_model {
        out.push_str(&format!("embed_model = \"{m}\"\n"));
    }
    out.push_str(&format!(
        "num_ctx = {}\ncontext_budget_fraction = {}\nindex_dir = \"{}\"\n",
        cfg.num_ctx, cfg.context_budget_fraction, cfg.index_dir
    ));
    Ok(out)
}

fn toml_from_str(raw: &str) -> Result<Config> {
    let mut cfg = Config::default();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"');
        match key {
            "ollama_host" => cfg.ollama_host = value.to_string(),
            "model" => cfg.model = value.to_string(),
            "model_fast" => cfg.model_fast = (!value.is_empty()).then(|| value.to_string()),
            "model_deep" => cfg.model_deep = (!value.is_empty()).then(|| value.to_string()),
            "embed_model" => cfg.embed_model = (!value.is_empty()).then(|| value.to_string()),
            "num_ctx" => cfg.num_ctx = value.parse().unwrap_or(cfg.num_ctx),
            "context_budget_fraction" => {
                cfg.context_budget_fraction = value.parse().unwrap_or(cfg.context_budget_fraction)
            }
            "index_dir" => cfg.index_dir = value.to_string(),
            _ => {}
        }
    }
    Ok(cfg)
}

/// A source file tracked in the index.
#[derive(Debug, Clone)]
pub struct FileRecord {
    pub id: i64,
    pub path: String,
    pub lang: String,
    pub size: i64,
    pub mtime: i64,
    pub hash: String,
}

/// A symbol (function, struct, class, ...) extracted from a file.
#[derive(Debug, Clone, Serialize)]
pub struct Symbol {
    pub id: i64,
    pub file_path: String,
    pub kind: String,
    pub name: String,
    pub start_line: i64,
    pub end_line: i64,
    pub parent: Option<String>,
}

/// Locate (or create) the `.graviton` index directory for a repo rooted at
/// `root`, and return the path to its SQLite database file.
pub fn db_path_for(root: &Path, index_dir: &str) -> Result<PathBuf> {
    let dir = root.join(index_dir);
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join("index.db"))
}

/// Open the SQLite database at `path`, creating and migrating the schema if
/// necessary.
pub fn open_db(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS files (
            id      INTEGER PRIMARY KEY,
            path    TEXT UNIQUE NOT NULL,
            lang    TEXT NOT NULL,
            size    INTEGER NOT NULL,
            mtime   INTEGER NOT NULL,
            hash    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS symbols (
            id          INTEGER PRIMARY KEY,
            file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            kind        TEXT NOT NULL,
            name        TEXT NOT NULL,
            start_line  INTEGER NOT NULL,
            end_line    INTEGER NOT NULL,
            parent      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);

        -- Name-based (not type-resolved) call graph: `callee_name` is
        -- matched textually, same simplification `symbols.name` LIKE
        -- lookups already make. `caller_symbol_id` is NULL when a call
        -- site isn't inside any extracted symbol (e.g. module-level code).
        CREATE TABLE IF NOT EXISTS calls (
            id                INTEGER PRIMARY KEY,
            file_id           INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            caller_symbol_id  INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
            callee_name       TEXT NOT NULL,
            line              INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_name);
        CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_symbol_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
            path UNINDEXED,
            start_line UNINDEXED,
            end_line UNINDEXED,
            kind UNINDEXED,
            name UNINDEXED,
            body
        );

        -- One row per embedded content_fts chunk (chunk_id = its rowid).
        -- Not foreign-keyed to content_fts (a virtual fts5 table) -- callers
        -- that delete/replace chunk rows (indexer re-indexing a changed
        -- file, clear_index) are responsible for deleting the matching
        -- embeddings rows too, so this table never silently points at rows
        -- that no longer exist.
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id    INTEGER PRIMARY KEY,
            model       TEXT NOT NULL,
            dims        INTEGER NOT NULL,
            vector      BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

        CREATE TABLE IF NOT EXISTS tool_runs (
            id          INTEGER PRIMARY KEY,
            tool        TEXT NOT NULL,
            args        TEXT NOT NULL,
            ran_at      INTEGER NOT NULL,
            exit_code   INTEGER,
            output      TEXT NOT NULL
        );
        "#,
    )?;
    Ok(conn)
}

/// Wipe indexed *code* for a fresh re-index (schema is kept). Tool-run
/// history (`tool_runs` and its `content_fts` rows) is untouched — it's
/// recon log, not derived from the repo tree, so re-indexing code shouldn't
/// discard it.
pub fn clear_index(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        DELETE FROM embeddings WHERE chunk_id IN (SELECT rowid FROM content_fts WHERE kind != 'tool_output');
        DELETE FROM content_fts WHERE kind != 'tool_output';
        DELETE FROM calls;
        DELETE FROM symbols;
        DELETE FROM files;
        "#,
    )?;
    Ok(())
}

//! graviton-core: config, database schema, and shared types for GRAVITON.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Runtime configuration, loaded from `~/.config/graviton/config.toml`
/// (created with sane defaults on first run).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Base URL of the local Ollama daemon.
    pub ollama_host: String,
    /// Model tag to use for chat/generation, e.g. "qwen3:8b".
    pub model: String,
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
}

// Minimal hand-rolled TOML (de)serialization so we don't pull in the `toml`
// crate just for a handful of scalar fields.
fn toml_to_string(cfg: &Config) -> Result<String> {
    Ok(format!(
        "ollama_host = \"{}\"\nmodel = \"{}\"\nnum_ctx = {}\ncontext_budget_fraction = {}\nindex_dir = \"{}\"\n",
        cfg.ollama_host, cfg.model, cfg.num_ctx, cfg.context_budget_fraction, cfg.index_dir
    ))
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

        CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
            path UNINDEXED,
            start_line UNINDEXED,
            end_line UNINDEXED,
            kind UNINDEXED,
            name UNINDEXED,
            body
        );
        "#,
    )?;
    Ok(conn)
}

/// Wipe all indexed data for a fresh re-index (schema is kept).
pub fn clear_index(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        DELETE FROM content_fts;
        DELETE FROM symbols;
        DELETE FROM files;
        "#,
    )?;
    Ok(())
}

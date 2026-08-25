mod lang;

pub use lang::Lang;

use anyhow::Result;
use ignore::WalkBuilder;
use rusqlite::Connection;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::UNIX_EPOCH;
use tree_sitter::{Parser, QueryCursor, StreamingIterator};

const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024; // skip anything bigger than 2MB
const CHUNK_LINES: usize = 150;
const CHUNK_OVERLAP: usize = 30;

/// Directories that are always skipped regardless of .gitignore, because
/// indexing them is either pointless or actively harmful to relevance.
const SKIP_DIRS: &[&str] = &[
    ".git", ".graviton", "target", "node_modules", "dist", "build", "venv",
    ".venv", "__pycache__", ".mypy_cache", ".idea", ".vscode",
];

#[derive(Default, Debug)]
pub struct IndexStats {
    pub files_scanned: usize,
    pub files_indexed: usize,
    pub files_skipped_unchanged: usize,
    pub symbols_extracted: usize,
    pub chunks_written: usize,
}

/// Walk `root`, parse/chunk every text file under it, and (re)populate the
/// SQLite index. Unchanged files (same content hash) are skipped, so
/// re-running this after small edits is cheap.
pub fn index_repo(conn: &mut Connection, root: &Path) -> Result<IndexStats> {
    let mut stats = IndexStats::default();

    let walker = WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_exclude(true)
        .filter_entry(|entry| {
            if let Some(name) = entry.file_name().to_str() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                    && SKIP_DIRS.contains(&name)
                {
                    return false;
                }
            }
            true
        })
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        let meta = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.len() > MAX_FILE_BYTES {
            continue;
        }
        stats.files_scanned += 1;

        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        if bytes.iter().take(4096).any(|b| *b == 0) {
            continue; // binary file
        }
        let content = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let rel_path = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        let hash = content_hash(&content);
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let unchanged: Option<String> = conn
            .query_row(
                "SELECT hash FROM files WHERE path = ?1",
                [&rel_path],
                |r| r.get(0),
            )
            .ok();
        if unchanged.as_deref() == Some(hash.as_str()) {
            stats.files_skipped_unchanged += 1;
            continue;
        }

        let language = Lang::from_path(path);
        let tx = conn.transaction()?;
        // Replace any previous rows for this file (cascades to symbols).
        tx.execute("DELETE FROM files WHERE path = ?1", [&rel_path])?;
        tx.execute("DELETE FROM content_fts WHERE path = ?1", [&rel_path])?;
        tx.execute(
            "INSERT INTO files (path, lang, size, mtime, hash) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![rel_path, language.name(), meta.len() as i64, mtime, hash],
        )?;
        let file_id = tx.last_insert_rowid();

        let symbols = extract_symbols(&content, language);
        for sym in &symbols {
            tx.execute(
                "INSERT INTO symbols (file_id, kind, name, start_line, end_line, parent) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![file_id, sym.kind, sym.name, sym.start_line, sym.end_line, sym.parent],
            )?;
        }
        stats.symbols_extracted += symbols.len();

        let chunks = chunk_lines(&content, CHUNK_LINES, CHUNK_OVERLAP);
        for (start, end, body) in &chunks {
            tx.execute(
                "INSERT INTO content_fts (path, start_line, end_line, kind, name, body) VALUES (?1, ?2, ?3, 'chunk', NULL, ?4)",
                rusqlite::params![rel_path, *start as i64, *end as i64, body],
            )?;
        }
        stats.chunks_written += chunks.len();

        tx.commit()?;
        stats.files_indexed += 1;
    }

    Ok(stats)
}

fn content_hash(content: &str) -> String {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Split `content` into overlapping line windows so every line ends up in at
/// least one chunk small enough to cite verbatim in an LLM prompt.
fn chunk_lines(content: &str, window: usize, overlap: usize) -> Vec<(usize, usize, String)> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let step = window.saturating_sub(overlap).max(1);
    let mut start = 0usize;
    loop {
        let end = (start + window).min(lines.len());
        let body = lines[start..end].join("\n");
        out.push((start + 1, end, body)); // 1-indexed lines for humans
        if end >= lines.len() {
            break;
        }
        start += step;
    }
    out
}

pub struct ExtractedSymbol {
    pub kind: String,
    pub name: String,
    pub start_line: i64,
    pub end_line: i64,
    pub parent: Option<String>,
}

/// Best-effort tree-sitter symbol extraction. Returns an empty vec (never an
/// error) for unsupported languages or if the grammar/query fails.
pub fn extract_symbols(content: &str, language: Lang) -> Vec<ExtractedSymbol> {
    let Some(ts_lang) = language.ts_language() else {
        return Vec::new();
    };
    let Some(query) = language.compile_def_query() else {
        return Vec::new();
    };

    let mut parser = Parser::new();
    if parser.set_language(&ts_lang).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };

    let name_idx = query.capture_index_for_name("name");
    let def_idx = query.capture_index_for_name("def");
    let (Some(name_idx), Some(def_idx)) = (name_idx, def_idx) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut cursor = QueryCursor::new();
    let bytes = content.as_bytes();
    let mut matches = cursor.matches(&query, tree.root_node(), bytes);
    while let Some(m) = matches.next() {
        let mut name_text = None;
        let mut def_node = None;
        for cap in m.captures {
            if cap.index == name_idx {
                name_text = cap.node.utf8_text(bytes).ok().map(|s| s.to_string());
            } else if cap.index == def_idx {
                def_node = Some(cap.node);
            }
        }
        let (Some(name), Some(node)) = (name_text, def_node) else {
            continue;
        };
        let kind = node.kind().to_string();
        out.push(ExtractedSymbol {
            kind,
            name,
            start_line: node.start_position().row as i64 + 1,
            end_line: node.end_position().row as i64 + 1,
            parent: None,
        });
    }
    out
}

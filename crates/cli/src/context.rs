//! Retrieval + context assembly: turns a natural-language question into a
//! bounded block of cited code the model can actually reason over.

use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;

pub struct ContextBlock {
    pub header: String,
    pub body: String,
}

impl ContextBlock {
    fn rendered(&self) -> String {
        format!("--- {} ---\n{}\n", self.header, self.body)
    }
}

/// Build a fts5 MATCH expression from free text: keep alnum/underscore
/// tokens of length >= 2, quote each (so punctuation/keywords in the
/// question can't break FTS5 query syntax), OR them together.
pub fn fts_query(text: &str) -> Option<String> {
    let tokens: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 2)
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" OR "))
    }
}

pub fn search_chunks(conn: &Connection, query: &str, limit: usize) -> Result<Vec<ContextBlock>> {
    let Some(match_expr) = fts_query(query) else {
        return Ok(Vec::new());
    };
    let mut stmt = conn.prepare(
        "SELECT path, start_line, end_line, body FROM content_fts \
         WHERE content_fts MATCH ?1 ORDER BY rank LIMIT ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![match_expr, limit as i64], |r| {
        let path: String = r.get(0)?;
        let start: i64 = r.get(1)?;
        let end: i64 = r.get(2)?;
        let body: String = r.get(3)?;
        Ok(ContextBlock {
            header: format!("{path}:{start}-{end}"),
            body,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn search_symbols(
    conn: &Connection,
    root: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<ContextBlock>> {
    let tokens: Vec<&str> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 3)
        .collect();
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stmt = conn.prepare(
        "SELECT f.path, s.kind, s.name, s.start_line, s.end_line \
         FROM symbols s JOIN files f ON f.id = s.file_id \
         WHERE s.name LIKE ?1 LIMIT ?2",
    )?;
    for tok in tokens {
        if out.len() >= limit {
            break;
        }
        let pattern = format!("%{tok}%");
        let rows = stmt.query_map(rusqlite::params![pattern, (limit - out.len()) as i64], |r| {
            let path: String = r.get(0)?;
            let kind: String = r.get(1)?;
            let name: String = r.get(2)?;
            let start: i64 = r.get(3)?;
            let end: i64 = r.get(4)?;
            Ok((path, kind, name, start, end))
        })?;
        for row in rows.filter_map(|r| r.ok()) {
            let (path, kind, name, start, end) = row;
            // The question can split into multiple tokens (e.g. "Invoke-Recon"
            // -> "Invoke", "Recon") that both hit the same symbol; keep one.
            if !seen.insert((path.clone(), start, end)) {
                continue;
            }
            let full_path = root.join(&path);
            let body = read_lines(&full_path, start, end).unwrap_or_default();
            if body.is_empty() {
                continue;
            }
            out.push(ContextBlock {
                header: format!("{path}:{start}-{end} ({kind} {name})"),
                body,
            });
        }
    }
    Ok(out)
}

fn read_lines(path: &Path, start: i64, end: i64) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = content.lines().collect();
    let start = (start.max(1) as usize) - 1;
    let end = (end as usize).min(lines.len());
    if start >= end {
        return None;
    }
    Some(lines[start..end].join("\n"))
}

pub fn read_whole_file(root: &Path, rel: &Path) -> Option<ContextBlock> {
    let path = if rel.is_absolute() { rel.to_path_buf() } else { root.join(rel) };
    let content = std::fs::read_to_string(&path).ok()?;
    const CAP: usize = 20_000;
    let truncated = content.len() > CAP;
    let body = if truncated {
        format!("{}\n... [truncated, {} bytes total]", &content[..CAP], content.len())
    } else {
        content
    };
    Some(ContextBlock {
        header: format!("{} (explicit)", rel.display()),
        body,
    })
}

/// Greedily assemble blocks into a single string under `char_budget`,
/// explicit files first (highest priority), then symbol hits, then FTS
/// chunks. Stops as soon as the budget would be exceeded.
pub fn assemble(char_budget: usize, groups: Vec<Vec<ContextBlock>>) -> String {
    let mut out = String::new();
    for group in groups {
        for block in group {
            let rendered = block.rendered();
            if out.len() + rendered.len() > char_budget {
                continue;
            }
            out.push_str(&rendered);
        }
    }
    out
}

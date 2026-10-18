//! Retrieval + context assembly: turns a natural-language question into a
//! bounded block of cited code the model can actually reason over.

use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;

pub struct ContextBlock {
    pub header: String,
    pub body: String,
    /// Repo-relative path this block came from, if any — used by
    /// `promote_full_files` to tell which hits share a file so a small
    /// snippet of it can be upgraded to the whole thing.
    pub path: Option<String>,
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
            path: Some(path),
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
                path: Some(path),
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
        path: Some(rel.display().to_string()),
    })
}

/// Same whole-file read, for a path already known relative to `root`
/// (`promote_full_files` knows that from the hits it's upgrading) —
/// distinct header so the model can tell "the user explicitly pointed me
/// at this" apart from "the index's top hits happened to land here".
fn read_whole_file_promoted(root: &Path, rel: &str) -> Option<ContextBlock> {
    let content = std::fs::read_to_string(root.join(rel)).ok()?;
    const CAP: usize = 20_000;
    if content.len() > CAP {
        // A hit landed in this file, but it's too big to safely promote
        // without risking truncating past the part search actually
        // matched (which a plain prefix cut could do) -- leave the
        // original snippet(s) in place instead of replacing them with a
        // worse partial view.
        return None;
    }
    Some(ContextBlock {
        header: format!("{rel} (whole file — auto-expanded from a top search hit)"),
        body: content,
        path: Some(rel.to_string()),
    })
}

/// A snippet (one FTS chunk, one symbol's lines) is one slice of a file —
/// real-world testing on an actual repo caught a security agent citing a
/// controller's one `@Roles` decorator from a chunk hit, concluding from
/// it alone, and inventing a second guard's implementation wholesale
/// because the file that actually holds it (a different, stricter,
/// per-operation check) was never retrieved. Telling the model to
/// `read_file` itself when a snippet looks incomplete did not reliably
/// help — a 7B-class local model skipped the extra tool call outright in
/// that same test and hallucinated anyway. This is the mechanical fix
/// instead of a behavioral one: for every distinct file any symbol/chunk
/// hit landed in, replace all of that file's snippets with the file in
/// full (bounded by `max_files` and the same size cap `--file` already
/// uses), in hit-rank order, so the model sees the real mechanism instead
/// of reasoning from a fragment of it regardless of whether it bothers to
/// ask for more.
pub fn promote_full_files(root: &Path, groups: &mut [&mut Vec<ContextBlock>], max_files: usize) -> Vec<ContextBlock> {
    let mut seen_paths: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut promoted = Vec::new();
    'outer: for group in groups.iter() {
        for block in group.iter() {
            let Some(path) = &block.path else { continue };
            if !seen_paths.insert(path.clone()) {
                continue;
            }
            if let Some(whole) = read_whole_file_promoted(root, path) {
                promoted.push(whole);
            }
            if promoted.len() >= max_files {
                break 'outer;
            }
        }
    }
    if promoted.is_empty() {
        return promoted;
    }
    let promoted_paths: std::collections::HashSet<&str> = promoted.iter().filter_map(|b| b.path.as_deref()).collect();
    for group in groups.iter_mut() {
        group.retain(|b| !matches!(&b.path, Some(p) if promoted_paths.contains(p.as_str())));
    }
    promoted
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

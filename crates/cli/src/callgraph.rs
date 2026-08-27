//! `grv callers`/`grv callees`: a name-based call graph built from the
//! `calls` table (`crates/indexer` extracts call sites via a second
//! tree-sitter query per language — see `Lang::call_query_src`).
//!
//! Deliberately not type-resolved: `callee_name` is matched as text, so
//! `grv callers run` matches every call site anywhere literally named
//! `run(...)`, regardless of which `run` it actually is at that scope.
//! That's the same simplification `grv symbol`'s `LIKE`-based name lookup
//! already makes for definitions — a second, real tree-sitter pass
//! resolving call targets against scope/type information would be a much
//! larger undertaking for a marginal precision gain at this tool's scale
//! (a single developer reading the results, not an automated refactoring
//! engine that needs to be exactly right).

use anyhow::Result;
use rusqlite::Connection;

pub struct CallerHit {
    pub caller_path: String,
    pub caller_symbol: Option<(String, String)>, // (kind, name)
    pub line: i64,
}

pub struct CalleeHit {
    pub callee_name: String,
    pub line: i64,
}

/// Every call site anywhere in the index literally naming `callee_name`,
/// with which symbol (if any) contains that call site.
pub fn find_callers(conn: &Connection, callee_name: &str, limit: usize) -> Result<Vec<CallerHit>> {
    let mut stmt = conn.prepare(
        "SELECT f.path, s.kind, s.name, c.line \
         FROM calls c \
         JOIN files f ON f.id = c.file_id \
         LEFT JOIN symbols s ON s.id = c.caller_symbol_id \
         WHERE c.callee_name = ?1 \
         ORDER BY f.path, c.line \
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![callee_name, limit as i64], |r| {
        let path: String = r.get(0)?;
        let kind: Option<String> = r.get(1)?;
        let name: Option<String> = r.get(2)?;
        let line: i64 = r.get(3)?;
        Ok(CallerHit { caller_path: path, caller_symbol: kind.zip(name), line })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Every call site made from within the symbol(s) named `symbol_name`
/// (there can be more than one — an overload, or the same method name in
/// different types/files; each is shown separately with its own file).
pub fn find_callees(conn: &Connection, symbol_name: &str, limit: usize) -> Result<Vec<(String, Vec<CalleeHit>)>> {
    let mut stmt = conn.prepare(
        "SELECT f.path, c.callee_name, c.line \
         FROM calls c \
         JOIN symbols s ON s.id = c.caller_symbol_id \
         JOIN files f ON f.id = c.file_id \
         WHERE s.name = ?1 \
         ORDER BY f.path, c.line \
         LIMIT ?2",
    )?;
    let rows: Vec<(String, String, i64)> = stmt
        .query_map(rusqlite::params![symbol_name, limit as i64], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .filter_map(|r| r.ok())
        .collect();

    let mut grouped: Vec<(String, Vec<CalleeHit>)> = Vec::new();
    for (path, callee_name, line) in rows {
        match grouped.iter_mut().find(|(p, _)| *p == path) {
            Some((_, hits)) => hits.push(CalleeHit { callee_name, line }),
            None => grouped.push((path, vec![CalleeHit { callee_name, line }])),
        }
    }
    Ok(grouped)
}

//! `grv callers`/`grv callees`: a name-based call graph built from the
//! `calls` table (`crates/indexer` extracts call sites via a second
//! tree-sitter query per language — see `Lang::call_query_src`).
//!
//! Deliberately not type-resolved: `callee_name` is matched as text, so
//! `grv callers run` matches every call site anywhere literally named
//! `run(...)`, regardless of which `run` it actually is at that scope.
//! That's the same simplification `grv symbol`'s `LIKE`-based name lookup
//! already makes for definitions — a real per-language scope/import
//! resolver (knowing exactly which `run` a given call site can see) is a
//! different order of engineering effort, on par with what a language
//! server spends its whole existence on, not a query tweak.
//!
//! What *is* a query tweak, and worth doing: `ResolutionHint` (below) adds
//! the cheapest real signal available without one — whether a same-named
//! definition exists in the call site's own file. In real code that's
//! true overwhelmingly often (shadowing a name across modules on purpose
//! is the rare case, not the common one), so it's a genuine, honest
//! disambiguation hint layered on top of the name-based match, not a claim
//! of true resolution.
//!
//! Three limits of that hint, named plainly rather than left implicit:
//!
//! 1. "Same file" isn't "same scope" — two `impl` blocks in one file can
//!    each define `new`, and a bare file-level check can't tell them
//!    apart. Fixed by carrying every matching definition (with its
//!    `parent` — the enclosing `impl`/`class`, see
//!    `ExtractedSymbol::parent`) instead of a single confident-sounding
//!    label, so a same-file collision is *shown*, not hidden behind
//!    `LikelySameFile`.
//! 2. No notion of imports/visibility: GRAVITON can't know that a call
//!    site's file specifically imports the `foo` from `b.rs` and not the
//!    one in `c.rs`. Not resolvable without real per-language import
//!    resolution — but `Ambiguous`'s candidate list at least hands over
//!    every real definition's file and enclosing scope, so a human's own
//!    knowledge of the codebase's imports can finish what the name match
//!    couldn't.
//! 3. `NoDefinitionIndexed` never meant "this doesn't exist" — only "not
//!    in what `grv index` walked" — but a bare unlabeled hit used to say
//!    nothing at all, leaving that reading available by omission. Now
//!    stated explicitly wherever it's shown (see `main.rs`'s formatter).

use anyhow::Result;
use rusqlite::Connection;

/// One real definition matching a callee name — enough context (file,
/// kind, enclosing scope, line) for a human to actually tell two
/// same-named candidates apart, which a bare `Vec<String>` of file paths
/// couldn't.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DefinitionRef {
    pub path: String,
    pub kind: String,
    pub name: String,
    /// The enclosing `impl`/`class`/module, if any — see
    /// `ExtractedSymbol::parent`. `None` for a top-level definition.
    pub parent: Option<String>,
    pub line: i64,
}

/// How confidently a `CallerHit`'s call site can be matched to a specific
/// definition, given only file-level proximity — not scope, not imports,
/// not types. Computed once per `find_callers` call from a single extra
/// query, not per-hit. See the module doc for the three concrete limits
/// this does (and doesn't) address.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolutionHint {
    /// One or more definitions named `callee_name` exist in this call
    /// site's own file — the strongest signal available without real
    /// scope resolution, and right far more often than not in real code.
    /// More than one entry here means the same-file check *itself* is
    /// ambiguous (e.g. two `impl` blocks each defining the same method
    /// name) — surfaced, not silently resolved to "the first one".
    LikelySameFile(Vec<DefinitionRef>),
    /// Exactly one definition named `callee_name` exists anywhere in the
    /// index, and it isn't in this file — still unambiguous (there's only
    /// one candidate), just not local.
    UniqueElsewhere(DefinitionRef),
    /// Multiple same-named definitions exist, none in this call site's
    /// file — genuinely ambiguous without real resolution. Every
    /// candidate is listed (capped — see `find_callers`) rather than just
    /// counted, so a human's own knowledge of the codebase's imports can
    /// narrow it down GRAVITON can't.
    Ambiguous(Vec<DefinitionRef>),
    /// No definition named `callee_name` was found anywhere in the index
    /// — a stdlib/external call, a trait method on a type defined
    /// elsewhere, dynamic dispatch, or simply unindexed code. Not "this
    /// function doesn't exist".
    NoDefinitionIndexed,
}

pub struct CallerHit {
    pub caller_path: String,
    pub caller_symbol: Option<(String, String)>, // (kind, name)
    pub line: i64,
    pub resolution: ResolutionHint,
}

pub struct CalleeHit {
    pub callee_name: String,
    pub line: i64,
}

/// Definitions considered for one `find_callers` call are capped here —
/// past this many same-named matches, listing every candidate stops being
/// useful to a human reading the output anyway (a name like `new` or
/// `get` in a huge repo could otherwise return hundreds).
const MAX_DEFINITIONS_CONSIDERED: usize = 50;

/// Every call site anywhere in the index literally naming `callee_name`,
/// with which symbol (if any) contains that call site, and a
/// `ResolutionHint` for how confidently that call site can be tied to a
/// specific definition (see the module doc and `ResolutionHint` itself).
pub fn find_callers(conn: &Connection, callee_name: &str, limit: usize) -> Result<Vec<CallerHit>> {
    // One query, up front, for every definition named `callee_name` --
    // cheap (an indexed name lookup, capped), reused for every hit below
    // instead of re-querying per row.
    let mut def_stmt = conn.prepare(
        "SELECT f.path, s.kind, s.name, s.parent, s.start_line \
         FROM symbols s JOIN files f ON f.id = s.file_id \
         WHERE s.name = ?1 \
         LIMIT ?2",
    )?;
    let definitions: Vec<DefinitionRef> = def_stmt
        .query_map(rusqlite::params![callee_name, MAX_DEFINITIONS_CONSIDERED as i64], |r| {
            Ok(DefinitionRef { path: r.get(0)?, kind: r.get(1)?, name: r.get(2)?, parent: r.get(3)?, line: r.get(4)? })
        })?
        .filter_map(|r| r.ok())
        .collect();

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
        Ok((path, kind, name, line))
    })?;

    Ok(rows
        .filter_map(|r| r.ok())
        .map(|(path, kind, name, line)| {
            let same_file: Vec<DefinitionRef> = definitions.iter().filter(|d| d.path == path).cloned().collect();
            let resolution = if definitions.is_empty() {
                ResolutionHint::NoDefinitionIndexed
            } else if !same_file.is_empty() {
                ResolutionHint::LikelySameFile(same_file)
            } else if definitions.len() == 1 {
                ResolutionHint::UniqueElsewhere(definitions[0].clone())
            } else {
                ResolutionHint::Ambiguous(definitions.clone())
            };
            CallerHit { caller_path: path, caller_symbol: kind.zip(name), line, resolution }
        })
        .collect())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Connection {
        let conn = graviton_core::open_db(std::path::Path::new(":memory:")).expect("open in-memory db");
        conn.execute("INSERT INTO files (id, path, lang, size, mtime, hash) VALUES (1, 'a.rs', 'rust', 0, 0, '')", []).unwrap();
        conn.execute("INSERT INTO files (id, path, lang, size, mtime, hash) VALUES (2, 'b.rs', 'rust', 0, 0, '')", []).unwrap();
        conn.execute("INSERT INTO files (id, path, lang, size, mtime, hash) VALUES (3, 'c.rs', 'rust', 0, 0, '')", []).unwrap();
        conn
    }

    #[test]
    fn likely_same_file_when_a_definition_shares_the_call_sites_file() {
        let conn = test_db();
        // `helper` defined in a.rs, called from a.rs and b.rs.
        conn.execute("INSERT INTO symbols (id, file_id, kind, name, start_line, end_line) VALUES (1, 1, 'function', 'helper', 1, 3)", []).unwrap();
        conn.execute("INSERT INTO calls (file_id, caller_symbol_id, callee_name, line) VALUES (1, NULL, 'helper', 10)", []).unwrap();
        conn.execute("INSERT INTO calls (file_id, caller_symbol_id, callee_name, line) VALUES (2, NULL, 'helper', 5)", []).unwrap();

        let hits = find_callers(&conn, "helper", 10).unwrap();
        assert_eq!(hits.len(), 2);
        let a = hits.iter().find(|h| h.caller_path == "a.rs").unwrap();
        let b = hits.iter().find(|h| h.caller_path == "b.rs").unwrap();
        match &a.resolution {
            ResolutionHint::LikelySameFile(defs) => {
                assert_eq!(defs.len(), 1);
                assert_eq!(defs[0].path, "a.rs");
            }
            other => panic!("expected LikelySameFile, got {other:?}"),
        }
        match &b.resolution {
            ResolutionHint::UniqueElsewhere(def) => assert_eq!(def.path, "a.rs"),
            other => panic!("expected UniqueElsewhere, got {other:?}"),
        }
    }

    /// Insufficiency #1, named directly: two `impl` blocks in the SAME
    /// file each defining a method called `new` used to collapse into one
    /// undifferentiated `LikelySameFile` label -- as confident-sounding as
    /// the single-candidate case, even though the "same file" check is
    /// itself ambiguous here. `symbols.parent` (the enclosing `impl`) now
    /// lets both candidates be told apart in the output.
    #[test]
    fn likely_same_file_lists_every_candidate_when_the_same_file_defines_it_twice() {
        let conn = test_db();
        // Point::new and Circle::new, both in a.rs; called from a.rs.
        conn.execute("INSERT INTO symbols (id, file_id, kind, name, start_line, end_line, parent) VALUES (1, 1, 'function', 'new', 2, 4, 'Point')", []).unwrap();
        conn.execute("INSERT INTO symbols (id, file_id, kind, name, start_line, end_line, parent) VALUES (2, 1, 'function', 'new', 8, 10, 'Circle')", []).unwrap();
        conn.execute("INSERT INTO calls (file_id, caller_symbol_id, callee_name, line) VALUES (1, NULL, 'new', 20)", []).unwrap();

        let hits = find_callers(&conn, "new", 10).unwrap();
        assert_eq!(hits.len(), 1);
        match &hits[0].resolution {
            ResolutionHint::LikelySameFile(defs) => {
                assert_eq!(defs.len(), 2, "both same-file candidates must be surfaced, not collapsed: {defs:?}");
                let parents: std::collections::HashSet<_> = defs.iter().map(|d| d.parent.clone()).collect();
                assert_eq!(parents, std::collections::HashSet::from([Some("Point".to_string()), Some("Circle".to_string())]));
            }
            other => panic!("expected LikelySameFile with 2 candidates, got {other:?}"),
        }
    }

    #[test]
    fn ambiguous_when_multiple_definitions_exist_and_none_match_the_call_site() {
        let conn = test_db();
        // Two different `run`s, defined in a.rs and b.rs; called from c.rs
        // (neither).
        conn.execute("INSERT INTO symbols (id, file_id, kind, name, start_line, end_line) VALUES (1, 1, 'function', 'run', 1, 3)", []).unwrap();
        conn.execute("INSERT INTO symbols (id, file_id, kind, name, start_line, end_line) VALUES (2, 2, 'function', 'run', 1, 3)", []).unwrap();
        conn.execute("INSERT INTO calls (file_id, caller_symbol_id, callee_name, line) VALUES (3, NULL, 'run', 1)", []).unwrap();

        let hits = find_callers(&conn, "run", 10).unwrap();
        assert_eq!(hits.len(), 1);
        match &hits[0].resolution {
            ResolutionHint::Ambiguous(defs) => {
                let paths: std::collections::HashSet<_> = defs.iter().map(|d| d.path.clone()).collect();
                assert_eq!(paths, std::collections::HashSet::from(["a.rs".to_string(), "b.rs".to_string()]));
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn no_definition_indexed_when_nothing_defines_the_name() {
        let conn = test_db();
        // Calling something from an external/stdlib crate -- never
        // indexed as a symbol anywhere.
        conn.execute("INSERT INTO calls (file_id, caller_symbol_id, callee_name, line) VALUES (1, NULL, 'println', 1)", []).unwrap();

        let hits = find_callers(&conn, "println", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].resolution, ResolutionHint::NoDefinitionIndexed);
    }
}

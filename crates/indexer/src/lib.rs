mod imports;
mod lang;
mod resolve;

pub use imports::{ImportEdge, extract_imports};
pub use lang::{ALL_LANGS, Lang};
pub use resolve::resolve_all_imports;

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
pub const SKIP_DIRS: &[&str] = &[
    ".git", ".graviton", "target", "node_modules", "dist", "build", "venv",
    ".venv", "__pycache__", ".mypy_cache", ".idea", ".vscode",
];

#[derive(Default, Debug)]
pub struct IndexStats {
    pub files_scanned: usize,
    pub files_indexed: usize,
    pub files_skipped_unchanged: usize,
    pub files_removed: usize,
    pub symbols_extracted: usize,
    pub calls_extracted: usize,
    pub imports_extracted: usize,
    pub imports_resolved: usize,
    pub chunks_written: usize,
}

/// Walk `root`, parse/chunk every text file under it, and (re)populate the
/// SQLite index. Unchanged files (same content hash) are skipped, so
/// re-running this after small edits is cheap.
pub fn index_repo(conn: &mut Connection, root: &Path) -> Result<IndexStats> {
    let mut stats = IndexStats::default();
    // Every path actually seen on disk this pass -- anything in `files`
    // that *isn't* in here by the end has been deleted/moved/renamed since
    // the last index and gets cleaned up below, instead of lingering in
    // search/symbol/call-graph results forever (this matters more once
    // `grv index --watch` is running unattended).
    let mut seen_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

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
        seen_paths.insert(rel_path.clone());

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
        // Any embeddings for this file's old chunks are about to be
        // orphaned (content_fts rowids aren't stable across a re-index) --
        // drop them now rather than leave `embeddings` pointing at rows
        // that no longer exist.
        tx.execute(
            "DELETE FROM embeddings WHERE chunk_id IN (SELECT rowid FROM content_fts WHERE path = ?1)",
            [&rel_path],
        )?;
        // Replace any previous rows for this file (cascades to symbols).
        tx.execute("DELETE FROM files WHERE path = ?1", [&rel_path])?;
        tx.execute("DELETE FROM content_fts WHERE path = ?1", [&rel_path])?;
        tx.execute(
            "INSERT INTO files (path, lang, size, mtime, hash) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![rel_path, language.name(), meta.len() as i64, mtime, hash],
        )?;
        let file_id = tx.last_insert_rowid();

        let mut symbols = extract_symbols(&content, language);
        // Each symbol's innermost enclosing symbol -- e.g. a method's
        // containing impl/class -- via the same smallest-enclosing-span
        // trick used below for call sites, applied to symbols themselves
        // (`ExtractedSymbol.parent` existed as a field long before
        // anything ever set it to `Some`). This is what lets
        // `grv symbol`'s output, and `ResolutionHint::LikelySameFile`'s
        // candidate list (`crates/cli/src/callgraph.rs`), distinguish two
        // same-named methods in two different `impl`/`class` blocks
        // within one file instead of only knowing "a `new` exists
        // somewhere in this file" -- a real, if still file-local (not
        // full-scope), precision gain.
        for i in 0..symbols.len() {
            let (start_i, end_i) = (symbols[i].start_line, symbols[i].end_line);
            let parent_idx = symbols
                .iter()
                .enumerate()
                .filter(|(j, s)| *j != i && s.start_line <= start_i && end_i <= s.end_line)
                .min_by_key(|(_, s)| s.end_line - s.start_line)
                .map(|(j, _)| j);
            symbols[i].parent = parent_idx.map(|j| symbols[j].name.clone());
        }
        // (symbol row id, start_line, end_line) for every symbol just
        // inserted -- used below to find each call site's innermost
        // enclosing symbol without a second query.
        let mut symbol_rows: Vec<(i64, i64, i64)> = Vec::with_capacity(symbols.len());
        for sym in &symbols {
            tx.execute(
                "INSERT INTO symbols (file_id, kind, name, start_line, end_line, parent) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![file_id, sym.kind, sym.name, sym.start_line, sym.end_line, sym.parent],
            )?;
            symbol_rows.push((tx.last_insert_rowid(), sym.start_line, sym.end_line));
        }
        stats.symbols_extracted += symbols.len();

        let calls = extract_calls(&content, language);
        for c in &calls {
            // Innermost enclosing symbol = the containing one with the
            // smallest line span (nested functions/methods all contain
            // the call site; the smallest span is the most specific).
            let caller_symbol_id = symbol_rows
                .iter()
                .filter(|(_, start, end)| *start <= c.line && c.line <= *end)
                .min_by_key(|(_, start, end)| end - start)
                .map(|(id, _, _)| *id);
            tx.execute(
                "INSERT INTO calls (file_id, caller_symbol_id, callee_name, line) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![file_id, caller_symbol_id, c.callee_name, c.line],
            )?;
        }
        stats.calls_extracted += calls.len();

        let file_imports = extract_imports(&content, language);
        for imp in &file_imports {
            tx.execute(
                "INSERT INTO imports (file_id, raw_path, imported_name, is_wildcard, module_prefix, line) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    file_id,
                    imp.raw_path,
                    imp.imported_name,
                    imp.is_wildcard,
                    (!imp.module_prefix.is_empty()).then(|| imp.module_prefix.join("::")),
                    imp.line
                ],
            )?;
        }
        stats.imports_extracted += file_imports.len();

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

    let previously_indexed: Vec<String> = {
        let mut stmt = conn.prepare("SELECT path FROM files")?;
        let out: Vec<String> = stmt.query_map([], |r| r.get(0))?.filter_map(|r| r.ok()).collect();
        out
    };
    let gone: Vec<&String> = previously_indexed.iter().filter(|p| !seen_paths.contains(*p)).collect();
    if !gone.is_empty() {
        let tx = conn.transaction()?;
        for path in &gone {
            tx.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT rowid FROM content_fts WHERE path = ?1)",
                [path.as_str()],
            )?;
            tx.execute("DELETE FROM content_fts WHERE path = ?1", [path.as_str()])?;
            tx.execute("DELETE FROM files WHERE path = ?1", [path.as_str()])?; // cascades symbols/calls
        }
        tx.commit()?;
        stats.files_removed = gone.len();
    }

    // A separate, repo-wide pass (see `resolve.rs`'s module doc): turning
    // a raw `use`/`import`/`require` edge into an actual file needs the
    // *complete* current file set (a target added/removed elsewhere this
    // same run must be reflected), not just the one file it was extracted
    // from -- so this always runs after every file above has been
    // processed, never per-file.
    stats.imports_resolved = resolve_all_imports(conn, root)?;

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
    // Scratch buffers for `satisfies_text_predicates` -- reused across
    // matches to avoid reallocating per match. Needed by languages whose
    // grammar has no dedicated node for "this is a definition" (Elixir's
    // `def`/`defmodule`, Racket/Scheme's `define` are all just generic
    // `call`/`list` nodes at the tree-sitter level) -- their queries use
    // `#eq?`/`#match?` predicates on a captured node's *text* to tell a
    // real definition apart from an ordinary call/form with the same
    // shape. Every other language's query has no predicates, so this is a
    // no-op there (`satisfies_text_predicates` returns `true` when a
    // pattern declares none).
    let mut pred_buf1 = Vec::new();
    let mut pred_buf2 = Vec::new();
    let mut bytes_provider = bytes;
    let mut matches = cursor.matches(&query, tree.root_node(), bytes);
    while let Some(m) = matches.next() {
        if !m.satisfies_text_predicates(&query, &mut pred_buf1, &mut pred_buf2, &mut bytes_provider) {
            continue;
        }
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

pub struct CallSite {
    pub line: i64,
    pub callee_name: String,
}

/// Best-effort call-site extraction (see `Lang::call_query_src` for the
/// name-based-not-type-resolved caveat). Empty vec, never an error, for a
/// language with no call query or a grammar/query mismatch — same
/// contract as `extract_symbols`.
pub fn extract_calls(content: &str, language: Lang) -> Vec<CallSite> {
    let Some(ts_lang) = language.ts_language() else {
        return Vec::new();
    };
    let Some(query) = language.compile_call_query() else {
        return Vec::new();
    };

    let mut parser = Parser::new();
    if parser.set_language(&ts_lang).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };

    let callee_idx = query.capture_index_for_name("callee");
    let call_idx = query.capture_index_for_name("call");
    let (Some(callee_idx), Some(call_idx)) = (callee_idx, call_idx) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut cursor = QueryCursor::new();
    let bytes = content.as_bytes();
    // Same predicate-enforcement need as `extract_symbols` -- see its
    // comment. Elixir/Racket/Scheme/Asm's call queries use `#not-any-of?`/
    // `#any-of?` to exclude definition keywords or restrict to actual
    // control-flow mnemonics, which is inert without this.
    let mut pred_buf1 = Vec::new();
    let mut pred_buf2 = Vec::new();
    let mut bytes_provider = bytes;
    let mut matches = cursor.matches(&query, tree.root_node(), bytes);
    while let Some(m) = matches.next() {
        if !m.satisfies_text_predicates(&query, &mut pred_buf1, &mut pred_buf2, &mut bytes_provider) {
            continue;
        }
        let mut callee_text = None;
        let mut call_node = None;
        for cap in m.captures {
            if cap.index == callee_idx {
                callee_text = cap.node.utf8_text(bytes).ok().map(|s| s.to_string());
            } else if cap.index == call_idx {
                call_node = Some(cap.node);
            }
        }
        let (Some(callee_name), Some(node)) = (callee_text, call_node) else {
            continue;
        };
        out.push(CallSite { line: node.start_position().row as i64 + 1, callee_name });
    }
    out
}

/// `index_repo` had never had a real integration test at all before this
/// batch — every other test in this crate exercises `extract_symbols`/
/// `extract_calls` directly on an in-memory string, never the actual
/// filesystem-walking, SQLite-writing entry point every `grv index`
/// invocation goes through.
#[cfg(test)]
mod index_repo_tests {
    use super::*;

    /// `ExtractedSymbol.parent` existed as a field since this project's
    /// first version but was hardcoded to `None` -- nothing ever computed
    /// it. Populated now via the same smallest-enclosing-span technique
    /// already used for call-site-to-symbol resolution, so that e.g. two
    /// same-named methods in two different `impl` blocks in one file are
    /// distinguishable (`grv symbol`'s output, and
    /// `callgraph::ResolutionHint`'s candidate list, both read this).
    #[test]
    fn parent_distinguishes_same_named_methods_in_different_impls() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("shapes.rs"),
            "struct Point { x: i32 }\nimpl Point {\n    fn new() -> Point { Point { x: 0 } }\n}\n\nstruct Circle { r: i32 }\nimpl Circle {\n    fn new() -> Circle { Circle { r: 0 } }\n}\n",
        )
        .unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut stmt = conn.prepare("SELECT name, parent FROM symbols WHERE name = 'new' ORDER BY parent").unwrap();
        let rows: Vec<(String, Option<String>)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().filter_map(|r| r.ok()).collect();

        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!(rows[0], ("new".to_string(), Some("Circle".to_string())));
        assert_eq!(rows[1], ("new".to_string(), Some("Point".to_string())));
    }

    #[test]
    fn top_level_symbols_have_no_parent() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "fn free_function() {}\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let parent: Option<String> = conn.query_row("SELECT parent FROM symbols WHERE name = 'free_function'", [], |r| r.get(0)).unwrap();
        assert_eq!(parent, None);
    }

    /// A real per-language import resolver (`resolve.rs`), exercised
    /// end-to-end through `index_repo` rather than just its internal
    /// path-matching functions -- these are the concrete scenarios
    /// `ResolutionHint::ImportResolved` (`crates/cli/src/callgraph.rs`)
    /// depends on to actually narrow an ambiguous call site down to the
    /// one definition the call site's own file really imports.
    #[test]
    fn rust_use_crate_resolves_across_modules_in_one_crate() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "mod helpers;\nuse crate::helpers::run;\nfn go() { run(); }\n").unwrap();
        std::fs::write(dir.path().join("src/helpers.rs"), "pub fn run() {}\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        let stats = index_repo(&mut conn, dir.path()).unwrap();
        assert!(stats.imports_resolved >= 1, "{stats:?}");

        let resolved_path: String = conn
            .query_row(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'crate::helpers::run'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(resolved_path, "src/helpers.rs");
    }

    #[test]
    fn rust_cross_crate_use_resolves_via_cargo_toml_package_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("crates/core/src")).unwrap();
        std::fs::write(dir.path().join("crates/core/Cargo.toml"), "[package]\nname = \"my-core\"\n").unwrap();
        std::fs::write(dir.path().join("crates/core/src/lib.rs"), "pub fn shared() {}\n").unwrap();
        std::fs::create_dir_all(dir.path().join("crates/app/src")).unwrap();
        std::fs::write(dir.path().join("crates/app/Cargo.toml"), "[package]\nname = \"app\"\n").unwrap();
        std::fs::write(dir.path().join("crates/app/src/main.rs"), "use my_core::shared;\nfn main() { shared(); }\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let resolved_path: String = conn
            .query_row(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'my_core::shared'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(resolved_path, "crates/core/src/lib.rs");
    }

    /// A real bug this project's own dogfooding caught: `use super::x;`
    /// inside an *inline* `#[cfg(test)] mod tests { ... }` block was
    /// resolved as if the whole file were one flat module, jumping
    /// `super` straight past the file's own top level to the crate root
    /// -- a confidently WRONG cross-file answer, not an honest "don't
    /// know" (this project holds its heuristics to a stricter bar than
    /// that -- see `resolve.rs`'s module doc). Fixed via
    /// `ImportEdge::module_prefix` tracking inline `mod` nesting.
    #[test]
    fn super_inside_an_inline_test_module_resolves_back_to_its_own_file_not_the_crate_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(
            dir.path().join("src/permissions.rs"),
            "fn glob_match(_p: &str, _t: &str) -> bool { true }\n\n#[cfg(test)]\nmod tests {\n    use super::glob_match;\n\n    #[test]\n    fn it_works() { assert!(glob_match(\"*\", \"x\")); }\n}\n",
        )
        .unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut stmt = conn
            .prepare(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'super::glob_match'",
            )
            .unwrap();
        let paths: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(paths, vec!["src/permissions.rs".to_string()], "must resolve back to its own file, never main.rs");
    }

    #[test]
    fn rust_external_crate_use_stays_unresolved() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "use serde::Deserialize;\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM imports i JOIN import_resolutions r ON r.import_id = i.id WHERE i.raw_path = 'serde::Deserialize'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0, "an external crate must never be guessed as resolved");
    }

    #[test]
    fn python_relative_import_resolves_to_sibling_module() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("pkg")).unwrap();
        std::fs::write(dir.path().join("pkg/__init__.py"), "").unwrap();
        std::fs::write(dir.path().join("pkg/models.py"), "class User: pass\n").unwrap();
        std::fs::write(dir.path().join("pkg/views.py"), "from . import models\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        // `from . import models` is genuinely ambiguous between "models is
        // an attribute of `pkg/__init__.py` itself" and "models is the
        // submodule `pkg/models.py`" from a pure path-based heuristic with
        // no access to `__init__.py`'s actual contents -- both are real,
        // honestly recorded candidates (the same multi-candidate honesty
        // as Go's whole-package case below), not a single guess. The
        // submodule file must be among them.
        let mut stmt = conn
            .prepare(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = '.' ORDER BY f2.path",
            )
            .unwrap();
        let paths: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert!(paths.contains(&"pkg/models.py".to_string()), "{paths:?}");
    }

    #[test]
    fn js_relative_import_resolves_with_extension_inference() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("utils.ts"), "export function helper() {}\n").unwrap();
        std::fs::write(dir.path().join("main.ts"), "import { helper } from './utils';\nhelper();\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let resolved_path: String = conn
            .query_row(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = './utils'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(resolved_path, "utils.ts");
    }

    #[test]
    fn go_import_resolves_to_every_file_in_the_target_package_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("go.mod"), "module myproject\n\ngo 1.22\n").unwrap();
        std::fs::create_dir_all(dir.path().join("pkg/utils")).unwrap();
        std::fs::write(dir.path().join("pkg/utils/a.go"), "package utils\nfunc Run() {}\n").unwrap();
        std::fs::write(dir.path().join("pkg/utils/b.go"), "package utils\nfunc Helper() {}\n").unwrap();
        std::fs::write(dir.path().join("main.go"), "package main\n\nimport \"myproject/pkg/utils\"\n\nfunc main() { utils.Run() }\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut stmt = conn
            .prepare(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'myproject/pkg/utils' ORDER BY f2.path",
            )
            .unwrap();
        let paths: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(paths, vec!["pkg/utils/a.go".to_string(), "pkg/utils/b.go".to_string()]);
    }
}

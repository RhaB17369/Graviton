mod imports;
mod lang;
mod resolve;

pub use imports::{ImportEdge, extract_imports, has_import_resolver};
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

    /// Proves the generic `resolve_relative_literal` mechanism (shared by
    /// C/C++/ObjC/GLSL/HLSL/Vim/Proto/Solidity/Verilog/Nix/Bash/Fish/Ruby/
    /// R/Racket/CMake) actually works end-to-end for a real case, not just
    /// in extraction isolation -- and specifically that a *quoted*
    /// `#include "x.h"` with no leading `./` still resolves against the
    /// including file's own directory (real C search-path semantics),
    /// while `#include <...>` never does.
    #[test]
    fn resolve_c_include_finds_local_header_but_not_system_one() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("helper.h"), "void helper(void);\n").unwrap();
        std::fs::write(dir.path().join("main.c"), "#include \"helper.h\"\n#include <stdio.h>\nint main(void) { helper(); return 0; }\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut stmt = conn.prepare("SELECT i.raw_path, f2.path FROM imports i LEFT JOIN import_resolutions r ON r.import_id = i.id LEFT JOIN files f2 ON f2.id = r.file_id").unwrap();
        let rows: Vec<(String, Option<String>)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().filter_map(|r| r.ok()).collect();
        assert!(rows.contains(&("helper.h".to_string(), Some("helper.h".to_string()))), "{rows:?}");
        // A `<...>` system include is skipped entirely at extraction time
        // (see `imports.rs::query_based::preproc_include`) -- stronger
        // than merely "unresolved", so it shouldn't appear as a row at all.
        assert!(!rows.iter().any(|(p, _)| p.contains("stdio")), "a system header must never even be recorded: {rows:?}");
    }

    /// Proves the generic `resolve_dotted_module` mechanism (shared by
    /// Java/Kotlin/Groovy/Scala/C#) resolves a real cross-file class
    /// import against a conventional Maven-style source root -- not just
    /// the repo root itself.
    #[test]
    fn java_import_resolves_via_conventional_maven_source_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/main/java/com/example/util")).unwrap();
        std::fs::write(dir.path().join("src/main/java/com/example/util/Helper.java"), "package com.example.util;\npublic class Helper { public static void run() {} }\n").unwrap();
        std::fs::create_dir_all(dir.path().join("src/main/java/com/example")).unwrap();
        std::fs::write(
            dir.path().join("src/main/java/com/example/Main.java"),
            "package com.example;\nimport com.example.util.Helper;\npublic class Main { public static void main(String[] a) { Helper.run(); } }\n",
        )
        .unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let resolved_path: String = conn
            .query_row(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'com.example.util.Helper'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(resolved_path, "src/main/java/com/example/util/Helper.java");
    }

    /// Proves a JVM-family wildcard import (`import a.b.*;`) resolves to
    /// *every* file in the target package directory, the same
    /// multi-candidate honesty Go's package-level resolution already has.
    #[test]
    fn java_wildcard_import_resolves_to_every_file_in_the_package() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("com/example/util")).unwrap();
        std::fs::write(dir.path().join("com/example/util/A.java"), "package com.example.util;\npublic class A {}\n").unwrap();
        std::fs::write(dir.path().join("com/example/util/B.java"), "package com.example.util;\npublic class B {}\n").unwrap();
        std::fs::write(dir.path().join("com/example/util/Main.java"), "package com.example.util;\nimport com.example.util.*;\npublic class Main {}\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut stmt = conn
            .prepare(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'com.example.util.*' ORDER BY f2.path",
            )
            .unwrap();
        let paths: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(paths, vec!["com/example/util/A.java".to_string(), "com/example/util/B.java".to_string(), "com/example/util/Main.java".to_string()]);
    }

    /// The exact case the user asked to be fixed: `import Data.List`
    /// exposes all of `Data.List`'s own names unqualified, but that must
    /// resolve to the ONE file `Data/List.hs` -- not a directory listing.
    /// A sibling module (`Data/Map.hs`) living in the SAME directory is
    /// not part of what this wildcard exposes, and must NOT appear in the
    /// resolution (the exact mistake this project's own `resolve_dotted_module`
    /// made for Elm before this fix -- see its module doc).
    #[test]
    fn haskell_wildcard_import_resolves_to_its_own_file_only_not_the_whole_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/Data")).unwrap();
        std::fs::write(dir.path().join("src/Data/List.hs"), "module Data.List (map) where\nmap :: (a -> b) -> [a] -> [b]\nmap _ _ = []\n").unwrap();
        std::fs::write(dir.path().join("src/Data/Map.hs"), "module Data.Map (lookup) where\nlookup :: a -> b\nlookup _ = undefined\n").unwrap();
        std::fs::write(dir.path().join("src/Main.hs"), "module Main where\nimport Data.List\nmain :: IO ()\nmain = print (map id [1])\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut stmt = conn
            .prepare(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'Data.List.*'",
            )
            .unwrap();
        let paths: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(paths, vec!["src/Data/List.hs".to_string()], "must resolve to exactly its own file, never a sibling module or a directory listing: {paths:?}");
    }

    /// `import qualified Data.Map as M` never brings names into unqualified
    /// scope (`M.lookup`, not bare `lookup`) -- so it must not be flagged
    /// as a wildcard, even though it still resolves to a real file (the
    /// distinction matters for `ResolutionHint`'s corroboration logic in
    /// `crates/cli/src/callgraph.rs`, not for whether the file is found).
    #[test]
    fn haskell_qualified_import_still_resolves_but_is_not_a_wildcard() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/Data")).unwrap();
        std::fs::write(dir.path().join("src/Data/Map.hs"), "module Data.Map (lookup) where\nlookup :: a -> b\nlookup _ = undefined\n").unwrap();
        std::fs::write(dir.path().join("src/Main.hs"), "module Main where\nimport qualified Data.Map as M\nmain :: IO ()\nmain = return ()\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let (raw_path, is_wildcard, resolved_path): (String, bool, String) = conn
            .query_row(
                "SELECT i.raw_path, i.is_wildcard, f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(raw_path, "Data.Map");
        assert!(!is_wildcard, "a qualified import must not be flagged as a wildcard");
        assert_eq!(resolved_path, "src/Data/Map.hs");
    }

    /// D's `import std.stdio;` (no selection, no `static`, no alias) --
    /// same single-file-not-directory correctness as the Haskell case
    /// above, for the completely different grammar shape found via a real
    /// parse dump (see `imports.rs::query_based::d_import`'s doc for the
    /// wrong first guess that dump corrected).
    #[test]
    fn d_wildcard_import_resolves_to_its_own_file_only() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("source/std")).unwrap();
        std::fs::write(dir.path().join("source/std/stdio.d"), "module std.stdio;\nvoid writeln(string s) {}\n").unwrap();
        std::fs::write(dir.path().join("source/std/algorithm.d"), "module std.algorithm;\n").unwrap();
        std::fs::write(dir.path().join("source/app.d"), "import std.stdio;\nvoid main() { writeln(\"hi\"); }\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut stmt = conn
            .prepare(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'std.stdio.*'",
            )
            .unwrap();
        let paths: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(paths, vec!["source/std/stdio.d".to_string()]);
    }

    /// Julia's `using Base: sin` -- a real selective import resolving to
    /// its module's file with the specific name captured, distinct from
    /// `using Base` alone (whole-module wildcard) and `import Base` alone
    /// (binds only `Base` itself, not a wildcard).
    #[test]
    fn julia_selective_using_resolves_to_the_modules_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/Helpers.jl"), "module Helpers\nfoo() = 1\nend\n").unwrap();
        std::fs::write(dir.path().join("src/Main.jl"), "using Helpers: foo\nfoo()\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let (imported_name, resolved_path): (String, String) = conn
            .query_row(
                "SELECT i.imported_name, f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'Helpers'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(imported_name, "foo");
        assert_eq!(resolved_path, "src/Helpers.jl");
    }

    /// The regression this whole fix started from: Elm's `exposing (..)`
    /// must resolve to the ONE file the imported module names, never a
    /// directory listing -- `resolve_dotted_module` originally treated
    /// every wildcard as a Java-style package directory, which is simply
    /// wrong for a language where a module is always exactly one file.
    #[test]
    fn elm_exposing_all_resolves_to_its_own_file_only_not_a_directory_listing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/Html")).unwrap();
        std::fs::write(dir.path().join("src/Html/Events.elm"), "module Html.Events exposing (onClick)\nonClick = 1\n").unwrap();
        std::fs::write(dir.path().join("src/Html/Attributes.elm"), "module Html.Attributes exposing (class)\nclass = 1\n").unwrap();
        std::fs::write(dir.path().join("src/Main.elm"), "module Main exposing (main)\nimport Html.Events exposing (..)\nmain = onClick\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut stmt = conn
            .prepare(
                "SELECT f2.path FROM imports i \
                 JOIN import_resolutions r ON r.import_id = i.id \
                 JOIN files f2 ON f2.id = r.file_id \
                 WHERE i.raw_path = 'Html.Events.*'",
            )
            .unwrap();
        let paths: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(paths, vec!["src/Html/Events.elm".to_string()], "must not also list Html/Attributes.elm: {paths:?}");
    }

    /// Ada's `with My_Pkg.Child;` -- GNAT's dash-joined-lowercase flat
    /// naming convention, not a slash-nested directory the way Java's is.
    #[test]
    fn ada_with_clause_resolves_via_dash_joined_lowercase_filename() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/my_pkg-child.ads"), "package My_Pkg.Child is\nend My_Pkg.Child;\n").unwrap();
        std::fs::write(dir.path().join("src/main.adb"), "with My_Pkg.Child;\nprocedure Main is\nbegin\n   null;\nend Main;\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'My_Pkg.Child'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["src/my_pkg-child.ads".to_string()]);
    }

    /// OCaml's `open Helper` -- lowercased outermost segment only.
    #[test]
    fn ocaml_open_resolves_to_lowercase_file_for_outermost_segment() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/helper.ml"), "let f x = x\n").unwrap();
        std::fs::write(dir.path().join("src/main.ml"), "open Helper\nlet () = ignore (f 1)\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'Helper.*'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["src/helper.ml".to_string()]);
    }

    /// Perl's `use Foo::Bar;` -- `::` -> `/`, rooted at `lib/`.
    #[test]
    fn perl_use_resolves_via_double_colon_to_lib_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("lib/Foo")).unwrap();
        std::fs::write(dir.path().join("lib/Foo/Bar.pm"), "package Foo::Bar;\n1;\n").unwrap();
        std::fs::write(dir.path().join("script.pl"), "use Foo::Bar;\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'Foo::Bar.*'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["lib/Foo/Bar.pm".to_string()]);
    }

    /// Fortran's `use mod_utils` -- no naming convention exists, so it's a
    /// flat filename guess against a handful of real extensions.
    #[test]
    fn fortran_use_module_resolves_via_flat_filename_guess() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/mod_utils.f90"), "module mod_utils\ncontains\nend module mod_utils\n").unwrap();
        std::fs::write(dir.path().join("src/main.f90"), "program p\n  use mod_utils\nend program p\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'mod_utils.*'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["src/mod_utils.f90".to_string()]);
    }

    /// Elixir's `alias MyApp.Helper` -- Mix's CamelCase-per-segment ->
    /// snake_case-per-segment convention under `lib/`. Also the regression
    /// test for the `find_nodes` -> `find_nodes_nested` extraction fix:
    /// without it, nothing nested inside `defmodule ... do ... end`'s
    /// `do_block` was ever found in the first place.
    #[test]
    fn elixir_alias_resolves_via_camelcase_to_snake_case_convention() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("lib/my_app")).unwrap();
        std::fs::write(dir.path().join("lib/my_app/helper.ex"), "defmodule MyApp.Helper do\n  def foo, do: 1\nend\n").unwrap();
        std::fs::write(dir.path().join("lib/my_app.ex"), "defmodule MyApp do\n  alias MyApp.Helper\nend\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'MyApp.Helper.*'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["lib/my_app/helper.ex".to_string()]);
    }

    /// Lua's `require("a.b")` -- dotted-to-slash, `package.path`-style.
    #[test]
    fn lua_require_resolves_dotted_path_to_slash_convention() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/a")).unwrap();
        std::fs::write(dir.path().join("src/a/b.lua"), "return {}\n").unwrap();
        std::fs::write(dir.path().join("main.lua"), "local m = require(\"a.b\")\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'a.b'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["src/a/b.lua".to_string()]);
    }

    /// Dart's relative `import 'helper.dart';` -- no leading `./` needed,
    /// same as C's quoted `#include`, via `resolve_relative_literal`.
    #[test]
    fn dart_relative_import_resolves_to_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("lib")).unwrap();
        std::fs::write(dir.path().join("lib/helper.dart"), "int foo() => 1;\n").unwrap();
        std::fs::write(dir.path().join("lib/main.dart"), "import 'helper.dart';\nvoid main() { foo(); }\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'helper.dart'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["lib/helper.dart".to_string()]);
    }

    /// Scheme's `(include "helper.scm")` -- a plain relative-literal path,
    /// same shape as Racket's.
    #[test]
    fn scheme_include_resolves_to_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("helper.scm"), "(define (f x) x)\n").unwrap();
        std::fs::write(dir.path().join("main.scm"), "(include \"helper.scm\")\n(f 1)\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'helper.scm'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["helper.scm".to_string()]);
    }

    /// PowerShell's dot-source `. .\helper.ps1` -- also the regression test
    /// for the `command_argument_sep`-vs-`generic_token` extraction fix
    /// (the dot-source form worked before the fix; `Import-Module` didn't).
    #[test]
    fn powershell_import_module_resolves_relative_module_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Other.psm1"), "function F { 1 }\n").unwrap();
        std::fs::write(dir.path().join("main.ps1"), "Import-Module ./Other.psm1\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = './Other.psm1'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["Other.psm1".to_string()]);
    }

    /// Assembly's GAS-style `.include "helper.inc"` -- a real relative
    /// path, resolved the same way as C's `#include "x"`.
    #[test]
    fn asm_include_resolves_to_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("helper.inc"), "MACRO_X equ 1\n").unwrap();
        std::fs::write(dir.path().join("main.asm"), ".include \"helper.inc\"\nmov eax, 1\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'helper.inc'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["helper.inc".to_string()]);
    }

    /// Swift Package Manager's `import CoreModule` -- resolves to every
    /// `.swift` file under `Sources/CoreModule/`, recursively; a bare
    /// `import Foundation` (no matching directory) must stay unresolved.
    #[test]
    fn swift_import_resolves_to_every_file_under_the_target_sources_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Sources/CoreModule/Nested")).unwrap();
        std::fs::create_dir_all(dir.path().join("Sources/App")).unwrap();
        std::fs::write(dir.path().join("Sources/CoreModule/Widget.swift"), "struct Widget {}\n").unwrap();
        std::fs::write(dir.path().join("Sources/CoreModule/Nested/Helper.swift"), "struct Helper {}\n").unwrap();
        std::fs::write(dir.path().join("Sources/App/main.swift"), "import Foundation\nimport CoreModule\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'CoreModule.*'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        paths.sort();
        assert_eq!(paths, vec!["Sources/CoreModule/Nested/Helper.swift".to_string(), "Sources/CoreModule/Widget.swift".to_string()]);

        let foundation_resolved: i64 = conn
            .query_row("SELECT COUNT(*) FROM imports i JOIN import_resolutions r ON r.import_id = i.id WHERE i.raw_path = 'Foundation.*'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(foundation_resolved, 0, "an external framework must never resolve to a real file");
    }

    /// Terraform's `module "vpc" { source = "./modules/vpc" }` -- resolves
    /// to every `.tf` file in the referenced directory.
    #[test]
    fn hcl_module_source_resolves_to_every_tf_file_in_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("modules/vpc")).unwrap();
        std::fs::write(dir.path().join("modules/vpc/main.tf"), "resource \"aws_vpc\" \"this\" {}\n").unwrap();
        std::fs::write(dir.path().join("modules/vpc/outputs.tf"), "output \"id\" {}\n").unwrap();
        std::fs::write(dir.path().join("main.tf"), "module \"vpc\" {\n  source = \"./modules/vpc\"\n}\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let mut paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = './modules/vpc'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        paths.sort();
        assert_eq!(paths, vec!["modules/vpc/main.tf".to_string(), "modules/vpc/outputs.tf".to_string()]);
    }

    /// Nim's `import pkg/helper` -- absolute-style, resolved against a
    /// bounded source root (the same convention Python's resolver uses),
    /// since it isn't relative to the importing file's own directory.
    #[test]
    fn nim_import_resolves_via_bounded_source_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("pkg")).unwrap();
        std::fs::write(dir.path().join("pkg/helper.nim"), "proc f*(x: int): int = x\n").unwrap();
        std::fs::write(dir.path().join("main.nim"), "import pkg/helper\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'pkg/helper'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["pkg/helper.nim".to_string()]);
    }

    /// VHDL's `use work.my_pkg.all` -- flat filename guess, `work` library
    /// only.
    #[test]
    fn vhdl_use_work_resolves_via_flat_filename_guess() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/my_pkg.vhd"), "package my_pkg is\nend package my_pkg;\n").unwrap();
        std::fs::write(dir.path().join("src/top.vhd"), "use work.my_pkg.all;\nentity top is\nend entity;\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'my_pkg'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["src/my_pkg.vhd".to_string()]);
    }

    /// Prolog's `:- consult('helper.pro').` -- a real relative path.
    /// Uses the `.pro` extension deliberately, not `.pl` -- this
    /// project's own `Lang::from_path` already resolves that ambiguous
    /// extension to Perl (a documented, pre-existing tradeoff, not
    /// something this batch changes).
    #[test]
    fn prolog_consult_resolves_to_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("helper.pro"), ":- module(helper, [f/1]).\nf(X) :- X = 1.\n").unwrap();
        std::fs::write(dir.path().join("main.pro"), ":- consult('helper.pro').\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = 'helper.pro'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["helper.pro".to_string()]);
    }

    /// The `graphql-import` convention's `# import Foo from './other.graphql'`.
    #[test]
    fn graphql_import_comment_resolves_to_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("other.graphql"), "type Foo { id: ID }\n").unwrap();
        std::fs::write(dir.path().join("main.graphql"), "# import Foo from './other.graphql'\ntype Bar { foo: Foo }\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = './other.graphql'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["other.graphql".to_string()]);
    }

    /// Svelte's `<script>` block is real JS underneath -- the injected
    /// second-parse-pass regression test: a relative import inside a
    /// `.svelte` file's script block must resolve exactly like a plain
    /// `.js` file's would.
    #[test]
    fn svelte_script_block_import_resolves_via_injected_js_parse() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("helper.js"), "export function f() { return 1; }\n").unwrap();
        std::fs::write(dir.path().join("App.svelte"), "<script>\nimport { f } from './helper';\n</script>\n<div>{f()}</div>\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = './helper'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["helper.js".to_string()]);
    }

    /// Crystal's `require "./helper"` -- a real relative path, resolved
    /// through the vendored fork's grammar (see
    /// vendor/tree-sitter-crystal/NOTICE.md).
    #[test]
    fn crystal_require_resolves_to_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("helper.cr"), "def helper\n  1\nend\n").unwrap();
        std::fs::write(dir.path().join("main.cr"), "require \"./helper\"\nputs helper\n").unwrap();

        let db_path = dir.path().join("index.db");
        let mut conn = graviton_core::open_db(&db_path).unwrap();
        index_repo(&mut conn, dir.path()).unwrap();

        let paths: Vec<String> = conn
            .prepare("SELECT f2.path FROM imports i JOIN import_resolutions r ON r.import_id = i.id JOIN files f2 ON f2.id = r.file_id WHERE i.raw_path = './helper'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(paths, vec!["helper.cr".to_string()]);
    }
}

//! Turning a raw `imports.raw_path`/`imported_name` edge into an actual
//! file (or files) in the repo -- a real, per-language, path-based
//! resolver, not a symbol table or type checker. Run as one pass over the
//! whole repo (`resolve_all_imports`, called at the end of `index_repo`
//! once every file's final path set is known), separate from per-file
//! extraction (`imports.rs`) which only needs one file's own content.
//!
//! What this can and can't do, stated plainly rather than left implicit
//! (same standard this project holds every other heuristic to -- see
//! `callgraph.rs`'s module doc):
//!
//! - **Rust**: resolves `crate::`/`self::`/`super::` against the owning
//!   crate's module tree (discovered from every `Cargo.toml` in the repo,
//!   `[package] name` mapped to its `src/` directory), and a leading
//!   segment matching another discovered crate's name for cross-crate
//!   imports. Assumes the standard `cargo new` layout (module path
//!   segments = directory/file path segments, one file per module) --
//!   `#[path = "..."]` attributes or other non-standard layouts aren't
//!   modeled. Anything else (a first segment matching neither `crate`/
//!   `self`/`super` nor a known local crate) is treated as external
//!   (std/a crates.io dependency) and left unresolved -- correctly, not
//!   as a failure.
//! - **Python**: relative imports (`from . import x`, `from ..pkg import
//!   y`) resolve unambiguously against the importing file's own directory
//!   -- no guessing needed. Absolute imports (`import a.b.c`) are tried
//!   against the repo root and every one of its immediate subdirectories
//!   as candidate "source roots" (covers both flat layouts and a single
//!   `src/`-style layout) -- a bounded heuristic, not a real `sys.path`.
//! - **JavaScript/TypeScript/TSX**: only relative imports (`./x`, `../x`)
//!   are resolved, directly against the importing file's directory, tried
//!   against real extensions and `index.*`. Bare specifiers (`import x
//!   from 'react'`) are always external packages in practice and are
//!   correctly left unresolved; `tsconfig.json` path aliases are a known,
//!   named gap (not modeled).
//! - **Go**: resolves an import path against the owning module's declared
//!   path (`go.mod`'s `module` line) plus the repo's directory tree.
//!   Unlike the others, a Go import names a *package* (a directory), so
//!   resolution can legitimately produce several files, not one.
//!
//! An import that doesn't resolve to anything is exactly as informative as
//! one that does: it means "not indexed" (an external dependency, the
//! stdlib, or a shape this heuristic doesn't cover) -- never a wrong
//! guess presented as a real one.

use anyhow::Result;
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};
use std::path::Path;

fn segs_of(path: &str) -> Vec<String> {
    path.split('/').filter(|s| !s.is_empty()).map(String::from).collect()
}

fn segs_to_path(segs: &[String]) -> String {
    segs.join("/")
}

fn parent_segs(segs: &[String]) -> Vec<String> {
    let mut s = segs.to_vec();
    s.pop();
    s
}

fn is_prefix(prefix: &[String], full: &[String]) -> bool {
    prefix.len() <= full.len() && prefix.iter().zip(full.iter()).all(|(a, b)| a == b)
}

struct RustCrate {
    name: String, // normalized: `-` -> `_`, matching how `use` paths spell it
    crate_dir: Vec<String>,
    src_root: Vec<String>,
}

fn discover_rust_crates(root: &Path, all_paths: &HashSet<String>) -> Vec<RustCrate> {
    let mut out = Vec::new();
    for path in all_paths {
        if !path.ends_with("Cargo.toml") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(root.join(path)) else { continue };
        let Ok(value) = raw.parse::<toml::Value>() else { continue };
        let Some(name) = value.get("package").and_then(|p| p.get("name")).and_then(|n| n.as_str()) else { continue };
        let crate_dir = parent_segs(&segs_of(path));
        let mut src_root = crate_dir.clone();
        src_root.push("src".to_string());
        let has_src_dir = all_paths.iter().any(|p| p.starts_with(&format!("{}/", segs_to_path(&src_root))));
        if !has_src_dir {
            src_root = crate_dir.clone();
        }
        out.push(RustCrate { name: name.replace('-', "_"), crate_dir, src_root });
    }
    out
}

fn owning_rust_crate<'a>(file_segs: &[String], crates: &'a [RustCrate]) -> Option<&'a RustCrate> {
    crates.iter().filter(|c| is_prefix(&c.crate_dir, file_segs)).max_by_key(|c| c.crate_dir.len())
}

/// A file's own module path, relative to its crate's `src_root`:
/// `src/foo/bar.rs` -> `["foo","bar"]`, `src/foo/mod.rs` -> `["foo"]`,
/// `src/lib.rs`/`src/main.rs` -> `[]` (crate root).
fn rust_module_path_of_file(file_segs: &[String], src_root: &[String]) -> Vec<String> {
    if !is_prefix(src_root, file_segs) {
        return Vec::new();
    }
    let mut rel: Vec<String> = file_segs[src_root.len()..].to_vec();
    match rel.last().map(|s| s.as_str()) {
        Some("mod.rs") | Some("lib.rs") | Some("main.rs") => {
            rel.pop();
        }
        Some(last) => {
            if let Some(stem) = last.strip_suffix(".rs") {
                let stem = stem.to_string();
                *rel.last_mut().unwrap() = stem;
            }
        }
        None => {}
    }
    rel
}

fn try_rust_module_file(src_root: &[String], modpath: &[String], all_paths: &HashSet<String>, out: &mut Vec<String>) {
    if modpath.is_empty() {
        for candidate in ["lib.rs", "main.rs"] {
            let mut segs = src_root.to_vec();
            segs.push(candidate.to_string());
            let p = segs_to_path(&segs);
            if all_paths.contains(&p) {
                out.push(p);
            }
        }
        return;
    }
    let mut base = src_root.to_vec();
    base.extend(modpath.iter().cloned());
    let as_file = format!("{}.rs", segs_to_path(&base));
    if all_paths.contains(&as_file) {
        out.push(as_file);
    }
    let mut mod_segs = base;
    mod_segs.push("mod.rs".to_string());
    let as_mod = segs_to_path(&mod_segs);
    if all_paths.contains(&as_mod) {
        out.push(as_mod);
    }
}

fn resolve_rust(raw_path: &str, module_prefix: &[String], importer_segs: &[String], crates: &[RustCrate], all_paths: &HashSet<String>) -> Vec<String> {
    let clean = raw_path.trim_end_matches("::*");
    let segs: Vec<&str> = clean.split("::").filter(|s| !s.is_empty()).collect();
    let Some(&first) = segs.first() else { return Vec::new() };
    let Some(owner) = owning_rust_crate(importer_segs, crates) else { return Vec::new() };
    // The module path *at the point the `use` actually sits*, not just
    // the file's own top-level module -- an inline `mod tests { ... }`
    // (extremely common for `#[cfg(test)]`) is one level deeper than the
    // file itself, and `self`/`super` need to resolve from there, not
    // from the file as a whole (see `ImportEdge::module_prefix`'s doc for
    // the wrong-answer this fixes).
    let mut effective_path = rust_module_path_of_file(importer_segs, &owner.src_root);
    effective_path.extend(module_prefix.iter().cloned());

    let (src_root, target): (&[String], Vec<String>) = match first {
        "crate" => (&owner.src_root, segs[1..].iter().map(|s| s.to_string()).collect()),
        "self" => {
            let mut p = effective_path.clone();
            p.extend(segs[1..].iter().map(|s| s.to_string()));
            (&owner.src_root, p)
        }
        "super" => {
            let mut p = effective_path.clone();
            let mut i = 0;
            while i < segs.len() && segs[i] == "super" {
                if p.is_empty() {
                    return Vec::new();
                }
                p.pop();
                i += 1;
            }
            p.extend(segs[i..].iter().map(|s| s.to_string()));
            (&owner.src_root, p)
        }
        name => match crates.iter().find(|c| c.name == name) {
            Some(c) => (&c.src_root, segs[1..].iter().map(|s| s.to_string()).collect()),
            None => return Vec::new(), // external crate/std -- honestly unresolved
        },
    };

    let mut out = Vec::new();
    try_rust_module_file(src_root, &target, all_paths, &mut out);
    if !target.is_empty() {
        try_rust_module_file(src_root, &target[..target.len() - 1], all_paths, &mut out);
    }
    out.sort();
    out.dedup();
    out
}

fn discover_python_roots(all_paths: &HashSet<String>) -> Vec<Vec<String>> {
    let mut roots = vec![Vec::new()]; // the repo root itself
    let mut children: HashSet<String> = HashSet::new();
    for p in all_paths {
        if let Some((first, rest)) = p.split_once('/') {
            if !rest.is_empty() {
                children.insert(first.to_string());
            }
        }
    }
    for c in children {
        roots.push(vec![c]);
    }
    roots
}

fn try_python_module(segs: &[String], all_paths: &HashSet<String>, out: &mut Vec<String>) {
    if segs.is_empty() {
        return;
    }
    let base = segs_to_path(segs);
    let as_file = format!("{base}.py");
    if all_paths.contains(&as_file) {
        out.push(as_file);
    }
    let as_pkg = format!("{base}/__init__.py");
    if all_paths.contains(&as_pkg) {
        out.push(as_pkg);
    }
}

fn resolve_python(raw_path: &str, imported_name: Option<&str>, importer_segs: &[String], roots: &[Vec<String>], all_paths: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(dots) = leading_dots(raw_path) {
        let after = &raw_path[dots..];
        let mut base = parent_segs(importer_segs);
        for _ in 1..dots {
            if base.is_empty() {
                return out;
            }
            base.pop();
        }
        if !after.is_empty() {
            base.extend(after.split('.').map(String::from));
        }
        try_python_module(&base, all_paths, &mut out);
        if let Some(name) = imported_name {
            let mut sub = base.clone();
            sub.push(name.to_string());
            try_python_module(&sub, all_paths, &mut out);
        }
    } else {
        for root in roots {
            let mut base = root.clone();
            base.extend(raw_path.split('.').map(String::from));
            try_python_module(&base, all_paths, &mut out);
            if let Some(name) = imported_name {
                let mut sub = base.clone();
                sub.push(name.to_string());
                try_python_module(&sub, all_paths, &mut out);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn leading_dots(s: &str) -> Option<usize> {
    let n = s.chars().take_while(|c| *c == '.').count();
    if n > 0 { Some(n) } else { None }
}

fn resolve_js(raw_path: &str, importer_segs: &[String], all_paths: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    if !(raw_path.starts_with("./") || raw_path.starts_with("../")) {
        return out; // bare specifier -- external package/stdlib
    }
    let mut base = parent_segs(importer_segs);
    for part in raw_path.split('/') {
        match part {
            "." | "" => {}
            ".." => {
                if base.is_empty() {
                    return out;
                }
                base.pop();
            }
            seg => base.push(seg.to_string()),
        }
    }
    let base_path = segs_to_path(&base);
    let mut candidates = vec![base_path.clone()];
    for ext in [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"] {
        candidates.push(format!("{base_path}{ext}"));
    }
    for ext in ["ts", "tsx", "js", "jsx"] {
        candidates.push(format!("{base_path}/index.{ext}"));
    }
    for c in candidates {
        if all_paths.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// Resolves a literal quoted/relative path (`#include "x.h"`, `source
/// ./helper.sh`, `` `include "x.vh" ``, ...) against the importing file's
/// own directory -- regardless of whether it has a leading `./`, since
/// several of these languages' real search semantics check the including
/// file's own directory first even without one (C's quoted `#include`
/// being the canonical case). `extra_extensions` lets a language whose
/// import statement can omit the file extension (Bash's `source helper`
/// meaning `helper.sh`) still resolve; pass `&[]` for languages that
/// always spell the extension out.
fn resolve_relative_literal(raw_path: &str, importer_segs: &[String], all_paths: &HashSet<String>, extra_extensions: &[&str]) -> Vec<String> {
    if raw_path.is_empty() {
        return Vec::new();
    }
    let mut base = parent_segs(importer_segs);
    for part in raw_path.split('/') {
        match part {
            "." | "" => {}
            ".." => {
                if base.is_empty() {
                    return Vec::new();
                }
                base.pop();
            }
            seg => base.push(seg.to_string()),
        }
    }
    let base_path = segs_to_path(&base);
    let mut out = Vec::new();
    if all_paths.contains(&base_path) {
        out.push(base_path.clone());
    }
    for ext in extra_extensions {
        let c = format!("{base_path}.{ext}");
        if all_paths.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// Every file with extension `ext`, grouped by its containing directory
/// -- the same "an import can name a whole directory of files" shape Go's
/// package resolution needs, reused here for JVM-family wildcard imports
/// (`import a.b.*;`) and Elm's `exposing (..)`.
fn files_by_dir_for_ext(all_paths: &HashSet<String>, ext: &str) -> HashMap<Vec<String>, Vec<String>> {
    let mut out: HashMap<Vec<String>, Vec<String>> = HashMap::new();
    let suffix = format!(".{ext}");
    for p in all_paths {
        if p.ends_with(&suffix) {
            out.entry(parent_segs(&segs_of(p))).or_default().push(p.clone());
        }
    }
    out
}

/// Candidate "source roots" for a hierarchical-module-name language: the
/// repo root, every one of a handful of well-known per-ecosystem
/// conventional paths (e.g. Maven/Gradle's `src/main/java`) that actually
/// exist in this repo, plus every immediate subdirectory of the repo root
/// as a generic fallback (the same bounded heuristic `discover_python_roots`
/// uses) -- not a real build-tool-aware source-root resolver.
fn conventional_roots(all_paths: &HashSet<String>, conventional: &[&str]) -> Vec<Vec<String>> {
    let mut roots = vec![Vec::new()];
    for c in conventional {
        let segs = segs_of(c);
        let prefix = format!("{}/", segs_to_path(&segs));
        if all_paths.iter().any(|p| p.starts_with(&prefix)) {
            roots.push(segs);
        }
    }
    let mut children: HashSet<String> = HashSet::new();
    for p in all_paths {
        if let Some((first, rest)) = p.split_once('/') {
            if !rest.is_empty() {
                children.insert(first.to_string());
            }
        }
    }
    for c in children {
        roots.push(vec![c]);
    }
    roots
}

/// Generic resolver for a hierarchical/dotted module-name language where
/// each import names one specific class/type as its last segment (Java,
/// Kotlin, Groovy, Scala, C#) -- `a.b.C` tried as `<root>/a/b/C.<ext>`
/// against each candidate root. `a.b.*` (or `a.b._` for Scala, via
/// `wildcard_char`) is a whole-package wildcard, resolved to every
/// same-extension file directly in that directory instead (`files_by_dir`)
/// -- the same multi-file honesty as Go's package-level resolution.
/// `wildcard_char` of `'\0'` disables wildcard detection for a language
/// that has none (C#) or whose "whole module" semantics this project
/// deliberately doesn't model as a wildcard (see `extract_imports`'s
/// D/Haskell/Julia omission).
fn resolve_dotted_module(raw_path: &str, wildcard_char: char, ext: &str, roots: &[Vec<String>], files_by_dir: &HashMap<Vec<String>, Vec<String>>, all_paths: &HashSet<String>) -> Vec<String> {
    let wildcard_suffix = if wildcard_char == '\0' { None } else { Some(format!(".{wildcard_char}")) };
    let (clean, is_wildcard) = match &wildcard_suffix {
        Some(suffix) => match raw_path.strip_suffix(suffix.as_str()) {
            Some(c) => (c, true),
            None => (raw_path, false),
        },
        None => (raw_path, false),
    };
    let segs: Vec<&str> = clean.split('.').filter(|s| !s.is_empty()).collect();
    if segs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for root in roots {
        let mut dir = root.clone();
        dir.extend(segs.iter().map(|s| s.to_string()));
        if is_wildcard {
            if let Some(files) = files_by_dir.get(&dir) {
                out.extend(files.iter().cloned());
            }
        } else {
            let file = format!("{}.{}", segs_to_path(&dir), ext);
            if all_paths.contains(&file) {
                out.push(file);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

struct GoModule {
    dir_segs: Vec<String>,
    module_path: String,
}

fn discover_go_modules(root: &Path, all_paths: &HashSet<String>) -> Vec<GoModule> {
    let mut out = Vec::new();
    for path in all_paths {
        if !path.ends_with("go.mod") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(root.join(path)) else { continue };
        let Some(line) = raw.lines().find(|l| l.trim_start().starts_with("module ")) else { continue };
        let module_path = line.trim_start().trim_start_matches("module ").trim().to_string();
        out.push(GoModule { dir_segs: parent_segs(&segs_of(path)), module_path });
    }
    out
}

fn owning_go_module<'a>(importer_segs: &[String], mods: &'a [GoModule]) -> Option<&'a GoModule> {
    mods.iter().filter(|m| is_prefix(&m.dir_segs, importer_segs)).max_by_key(|m| m.dir_segs.len())
}

fn resolve_go(raw_path: &str, importer_segs: &[String], mods: &[GoModule], files_by_dir: &HashMap<Vec<String>, Vec<String>>) -> Vec<String> {
    let Some(m) = owning_go_module(importer_segs, mods) else { return Vec::new() };
    let rel: Option<String> = if raw_path == m.module_path {
        Some(String::new())
    } else {
        raw_path.strip_prefix(&format!("{}/", m.module_path)).map(String::from)
    };
    let Some(rel) = rel else { return Vec::new() }; // a different module -- external, honestly unresolved
    let mut dir_segs = m.dir_segs.clone();
    if !rel.is_empty() {
        dir_segs.extend(rel.split('/').map(String::from));
    }
    files_by_dir.get(&dir_segs).cloned().unwrap_or_default()
}

/// Recomputes `import_resolutions` for every row currently in `imports`,
/// using the repo's complete current file set. Cheap enough to always run
/// in full (no incremental bookkeeping needed): it's pure string/path
/// matching over already-loaded rows, no re-parsing.
pub fn resolve_all_imports(conn: &Connection, root: &Path) -> Result<usize> {
    let all_paths: HashSet<String> = {
        let mut stmt = conn.prepare("SELECT path FROM files")?;
        stmt.query_map([], |r| r.get::<_, String>(0))?.filter_map(|r| r.ok()).collect()
    };
    // `lang`, not the raw file extension, decides which resolver runs --
    // extension guessing breaks for filename-based languages (`CMakeLists.txt`
    // has extension "txt", not "cmake"; same issue would hit `Dockerfile`/
    // `Makefile` for any language added here later), and `Lang::from_path`
    // has already done this correctly once at index time.
    let path_and_lang_by_file_id: HashMap<i64, (String, String)> = {
        let mut stmt = conn.prepare("SELECT id, path, lang FROM files")?;
        stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, (r.get::<_, String>(1)?, r.get::<_, String>(2)?))))?.filter_map(|r| r.ok()).collect()
    };
    let path_to_id: HashMap<String, i64> = path_and_lang_by_file_id.iter().map(|(id, (p, _))| (p.clone(), *id)).collect();

    let rust_crates = discover_rust_crates(root, &all_paths);
    let python_roots = discover_python_roots(&all_paths);
    let go_mods = discover_go_modules(root, &all_paths);
    let go_files_by_dir = files_by_dir_for_ext(&all_paths, "go");
    let java_files_by_dir = files_by_dir_for_ext(&all_paths, "java");
    let kotlin_files_by_dir = files_by_dir_for_ext(&all_paths, "kt");
    let groovy_files_by_dir = files_by_dir_for_ext(&all_paths, "groovy");
    let scala_files_by_dir = files_by_dir_for_ext(&all_paths, "scala");
    let jvm_roots = conventional_roots(&all_paths, &["src/main/java", "src/test/java", "src/main/kotlin", "src/test/kotlin", "src/main/scala", "src/main/groovy", "src"]);
    let elm_roots = conventional_roots(&all_paths, &["src"]);
    let elm_files_by_dir = files_by_dir_for_ext(&all_paths, "elm");
    let csharp_roots = conventional_roots(&all_paths, &["src"]);

    let imports: Vec<(i64, i64, String, Option<String>, Option<String>)> = {
        let mut stmt = conn.prepare("SELECT id, file_id, raw_path, imported_name, module_prefix FROM imports")?;
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?.filter_map(|r| r.ok()).collect()
    };

    conn.execute("DELETE FROM import_resolutions", [])?;
    let mut resolved_count = 0usize;
    for (import_id, file_id, raw_path, imported_name, module_prefix_raw) in imports {
        let Some((importer_path, lang)) = path_and_lang_by_file_id.get(&file_id) else { continue };
        let importer_segs = segs_of(importer_path);
        let module_prefix: Vec<String> = module_prefix_raw.map(|s| s.split("::").map(String::from).collect()).unwrap_or_default();

        let resolved_paths: Vec<String> = match lang.as_str() {
            "rust" => resolve_rust(&raw_path, &module_prefix, &importer_segs, &rust_crates, &all_paths),
            "python" => resolve_python(&raw_path, imported_name.as_deref(), &importer_segs, &python_roots, &all_paths),
            "javascript" | "typescript" | "tsx" => resolve_js(&raw_path, &importer_segs, &all_paths),
            "go" => resolve_go(&raw_path, &importer_segs, &go_mods, &go_files_by_dir),
            // Quoted-include-style languages: resolved relative to the
            // importing file's own directory regardless of a leading
            // `./` (correct C/C++ `#include "x"` search-path semantics --
            // the same directory is searched first -- and harmless
            // elsewhere: a bare name that doesn't happen to match a real
            // repo-relative file just stays unresolved, no false positive
            // risk).
            "c" | "cpp" | "objc" | "glsl" | "hlsl" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &[]),
            "vim" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["vim"]),
            "proto" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &[]),
            "solidity" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["sol"]),
            "verilog" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &[]),
            "nix" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["nix"]),
            "bash" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["sh", "bash"]),
            "fish" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["fish"]),
            "ruby" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["rb"]),
            "r" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["R", "r"]),
            "racket" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["rkt"]),
            // CMake's `include(x.cmake)` is a relative-literal-path edge
            // like the others above; `add_subdirectory(dir)` names a
            // *directory* (containing its own CMakeLists.txt), which this
            // resolver can't usefully turn into "a file" -- it just stays
            // unresolved, correctly, rather than guessed at.
            "cmake" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &[]),
            // JVM-family: one specific class/type per import, resolved
            // against conventional Maven/Gradle source roots plus the
            // generic repo-root/immediate-subdirectory fallback every
            // hierarchical resolver here uses.
            "java" => resolve_dotted_module(&raw_path, '*', "java", &jvm_roots, &java_files_by_dir, &all_paths),
            "kotlin" => resolve_dotted_module(&raw_path, '*', "kt", &jvm_roots, &kotlin_files_by_dir, &all_paths),
            "groovy" => resolve_dotted_module(&raw_path, '*', "groovy", &jvm_roots, &groovy_files_by_dir, &all_paths),
            "scala" => resolve_dotted_module(&raw_path, '_', "scala", &jvm_roots, &scala_files_by_dir, &all_paths),
            // C# has no wildcard `using` and namespaces don't reliably
            // mirror directory structure the way Java's package
            // convention does -- still attempted, just lower-confidence,
            // hence no dedicated `files_by_dir` (a C# wildcard can't occur
            // anyway) and the generic root list rather than a JVM-specific
            // one.
            "csharp" => resolve_dotted_module(&raw_path, '\0', "cs", &csharp_roots, &HashMap::new(), &all_paths),
            "elm" => resolve_dotted_module(&raw_path, '*', "elm", &elm_roots, &elm_files_by_dir, &all_paths),
            _ => Vec::new(),
        };
        for p in resolved_paths {
            if let Some(&resolved_file_id) = path_to_id.get(p.as_str()) {
                conn.execute("INSERT OR IGNORE INTO import_resolutions (import_id, file_id) VALUES (?1, ?2)", rusqlite::params![import_id, resolved_file_id])?;
                resolved_count += 1;
            }
        }
    }
    Ok(resolved_count)
}

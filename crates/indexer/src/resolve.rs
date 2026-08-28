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
//! - **C/C++/ObjC/GLSL/HLSL/Vim/Proto/Solidity/Verilog/Nix/Bash/Fish/Ruby/
//!   R/Racket/CMake/Erlang/Zig/PHP/LaTeX/Dart/Scheme/PowerShell**: quoted/
//!   relative-literal paths, resolved against the importing file's own
//!   directory via `resolve_relative_literal` -- the same shape as C's
//!   `#include "x"`.
//! - **Java/Kotlin/Groovy/Scala/C#/Elm/Haskell/D/Julia**: hierarchical
//!   dotted module names, resolved via `resolve_dotted_module` against
//!   conventional source roots. The JVM family (plus a wildcard-disabled
//!   C#) are real *package* languages, where `a.b.*` genuinely means
//!   "every file in directory a/b" (`wildcard_is_package_directory:
//!   true`). Elm/Haskell/D/Julia map exactly one module to one file --
//!   there is no package directory at all, so their wildcard/unqualified/
//!   `exposing (..)`/`hiding`/aliased forms all resolve to that SAME one
//!   file a plain import would (`wildcard_is_package_directory: false`);
//!   getting this backwards for Elm once produced a confidently wrong
//!   directory listing, which is the mistake this flag exists to prevent
//!   from recurring for any future module-per-file language.
//! - **Lua**: `require("a.b")` -> `a/b.lua`, the same dotted-to-slash shape
//!   as the module languages above (via `resolve_dotted_module` with its
//!   wildcard form disabled, since Lua has none).
//! - **Ada, OCaml, Perl, Fortran, Elixir**: each has its own genuinely
//!   different file-naming convention that doesn't fit the two generic
//!   resolvers above, so each gets a small dedicated function --
//!   `resolve_ada` (GNAT's dash-joined-lowercase flat naming), `resolve_ocaml`
//!   (single-lowercase-segment, sub-modules-in-the-same-file left
//!   unresolved-past-the-first-segment rather than guessed at),
//!   `resolve_perl` (`::`-separated, `lib`-rooted, same shape as
//!   `resolve_dotted_module` but keyed on a two-character separator),
//!   `resolve_fortran_module` (a flat filename guess -- Fortran has no
//!   real module-to-file convention at all), `resolve_elixir` (Mix's
//!   CamelCase-per-segment -> snake_case-per-segment convention under
//!   `lib/`). See each function's own doc for the specific reasoning.
//! - **Not resolved, with reasons rather than silence**: Nim (the grammar
//!   has no import-related node at all to extract from), VHDL (no
//!   reliable package-to-file naming convention exists to resolve
//!   against), Prolog (directive shape too uncertain to encode safely),
//!   and Crystal (confirmed via a real parse-tree dump that
//!   tree-sitter-crystal 0.1.0 doesn't parse `require "..."` as a call/
//!   macro-invocation node at all, so there's no extraction to resolve in
//!   the first place).
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

/// Generic resolver for a hierarchical/dotted module-name language --
/// `a.b.C` tried as `<root>/a/b/C.<ext>` against each candidate root.
/// `wildcard_char` of `'\0'` disables wildcard detection entirely (C#,
/// which has no wildcard `using`).
///
/// `wildcard_is_package_directory` is the real distinction this project
/// initially got wrong for Elm and had to fix: for a *package* language
/// (Java/Kotlin/Groovy/Scala, where `a.b` genuinely names a directory that
/// can hold many files), a wildcard (`a.b.*`, `a.b._`) resolves to every
/// same-extension file in that directory (`files_by_dir`) -- the same
/// multi-file honesty as Go. For a *module* language (Haskell/D/Julia/Elm,
/// where the dotted path names exactly ONE file, one module per file,
/// full stop -- there is no "package directory" at all), a wildcard
/// resolves to that SAME single file a plain import of it would, via the
/// exact same `<root>/a/b/C.<ext>` lookup -- looking up `a/b` as a
/// directory would be wrong here (it usually doesn't even exist; `a.b`'s
/// sibling modules living in the same directory, like `Data/Map.hs` next
/// to `Data/List.hs`, are NOT part of what a `Data.List` wildcard exposes).
/// The wildcard marker only still matters for `is_wildcard` itself (read
/// separately from `imports.is_wildcard` by `callgraph::find_callers` to
/// decide whether this import corroborates an arbitrary call) -- it does
/// not change which file gets resolved when this flag is false.
fn resolve_dotted_module(raw_path: &str, wildcard_char: char, ext: &str, roots: &[Vec<String>], files_by_dir: &HashMap<Vec<String>, Vec<String>>, wildcard_is_package_directory: bool, all_paths: &HashSet<String>) -> Vec<String> {
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
        if is_wildcard && wildcard_is_package_directory {
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

/// GNAT's Ada file-naming convention: a `with`-ed unit `Parent.Child` maps
/// to `parent-child.ads`/`.adb` -- lowercased, and (unlike every
/// slash-nested hierarchical resolver above) the `.` separator becomes a
/// literal `-` in the FILENAME itself, not a directory nesting level; Ada
/// units are conventionally flat inside one source directory regardless of
/// how deep their dotted name is. Existing underscores inside a segment
/// (`Text_IO`) are real identifier characters and stay untouched.
fn resolve_ada(raw_path: &str, roots: &[Vec<String>], all_paths: &HashSet<String>) -> Vec<String> {
    if raw_path.is_empty() {
        return Vec::new();
    }
    let dashed = raw_path.to_lowercase().replace('.', "-");
    let mut out = Vec::new();
    for root in roots {
        let mut dir = root.clone();
        dir.push(dashed.clone());
        let base = segs_to_path(&dir);
        for ext in ["ads", "adb"] {
            let file = format!("{base}.{ext}");
            if all_paths.contains(&file) {
                out.push(file);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// OCaml's `open Module`/`open Module.Sub`: only the FIRST (outermost)
/// dotted segment is a real compilation unit that maps to a file
/// (`module_name.ml`/`.mli`, lowercased -- OCaml's own compiler-enforced
/// convention, not a guess); anything after the first dot names a
/// sub-module declared *inside* that same file (or re-exported through a
/// wrapping library, e.g. `Base.List`), which this path-based resolver has
/// no way to look inside a file for -- so a multi-segment `open` still
/// resolves to its outermost unit's file, honestly approximate rather than
/// silently wrong about which file, at least.
fn resolve_ocaml(raw_path: &str, roots: &[Vec<String>], all_paths: &HashSet<String>) -> Vec<String> {
    let clean = raw_path.strip_suffix(".*").unwrap_or(raw_path);
    let Some(first) = clean.split('.').next().filter(|s| !s.is_empty()) else { return Vec::new() };
    let lower = first.to_lowercase();
    let mut out = Vec::new();
    for root in roots {
        let mut dir = root.clone();
        dir.push(lower.clone());
        let base = segs_to_path(&dir);
        for ext in ["ml", "mli"] {
            let file = format!("{base}.{ext}");
            if all_paths.contains(&file) {
                out.push(file);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Perl's `use Foo::Bar;` -- CPAN's own `::` -> `/` directory convention
/// (`Foo::Bar` lives at `lib/Foo/Bar.pm`), the same shape as
/// `resolve_dotted_module` but keyed on a two-character separator that
/// function's single-`char` `split('.')` can't express, so it gets its own
/// small, honest duplicate rather than a generalized separator parameter
/// bent to fit one outlier.
fn resolve_perl(raw_path: &str, roots: &[Vec<String>], all_paths: &HashSet<String>) -> Vec<String> {
    let clean = raw_path.strip_suffix(".*").unwrap_or(raw_path);
    let segs: Vec<&str> = clean.split("::").filter(|s| !s.is_empty()).collect();
    if segs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for root in roots {
        let mut dir = root.clone();
        dir.extend(segs.iter().map(|s| s.to_string()));
        let file = format!("{}.pm", segs_to_path(&dir));
        if all_paths.contains(&file) {
            out.push(file);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Fortran's `use module_name` names a *module*, not a file, and unlike
/// every other module-name language here there is no standardized
/// module-to-filename convention at all (no classpath, no GNAT naming
/// rule) -- so this is an honest flat guess: try the module name directly
/// as a filename (both as written and lowercased, since `gfortran`
/// doesn't care about case but real repos are inconsistent about it) against
/// a handful of real Fortran source extensions, no directory nesting.
fn resolve_fortran_module(module_name: &str, roots: &[Vec<String>], all_paths: &HashSet<String>) -> Vec<String> {
    if module_name.is_empty() {
        return Vec::new();
    }
    let lower = module_name.to_lowercase();
    let mut out = Vec::new();
    for root in roots {
        for name in [module_name, lower.as_str()] {
            let mut dir = root.clone();
            dir.push(name.to_string());
            let base = segs_to_path(&dir);
            for ext in ["f90", "f95", "f03", "f08", "f", "for", "F90", "F95", "F"] {
                let file = format!("{base}.{ext}");
                if all_paths.contains(&file) {
                    out.push(file);
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `MyModule` -> `my_module`, Elixir/Mix's own convention for mapping a
/// CamelCase module segment to its snake_case file/directory name. Inserts
/// an underscore before an uppercase letter only when it sits at a real
/// word boundary (preceded or followed by a lowercase letter) so a run of
/// capitals from an acronym (`HTTPServer`) still splits sensibly
/// (`http_server`) instead of `h_t_t_p_server`.
fn camel_to_snake(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_uppercase() {
            let prev_lower = i > 0 && chars[i - 1].is_lowercase();
            let next_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();
            if i > 0 && (prev_lower || next_lower) {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Elixir/Mix's `alias`/`import`/`require`/`use Foo.Bar` -> `lib/foo/bar.ex`:
/// each dotted CamelCase segment becomes its own snake_case path segment
/// (via `camel_to_snake`), joined by `/` under the conventional `lib`
/// source root -- Mix's real, standardized module-to-file convention (not
/// a guess the way Fortran's has to be).
fn resolve_elixir(raw_path: &str, roots: &[Vec<String>], all_paths: &HashSet<String>) -> Vec<String> {
    let clean = raw_path.strip_suffix(".*").unwrap_or(raw_path);
    let segs: Vec<String> = clean.split('.').filter(|s| !s.is_empty()).map(camel_to_snake).collect();
    if segs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for root in roots {
        let mut dir = root.clone();
        dir.extend(segs.iter().cloned());
        let base = segs_to_path(&dir);
        for ext in ["ex", "exs"] {
            let file = format!("{base}.{ext}");
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
    let csharp_roots = conventional_roots(&all_paths, &["src"]);
    let haskell_roots = conventional_roots(&all_paths, &["src", "app", "lib"]);
    let d_roots = conventional_roots(&all_paths, &["source", "src"]);
    let julia_roots = conventional_roots(&all_paths, &["src"]);
    let ada_roots = conventional_roots(&all_paths, &["src", "source"]);
    let ocaml_roots = conventional_roots(&all_paths, &["src", "lib"]);
    let perl_roots = conventional_roots(&all_paths, &["lib"]);
    let fortran_roots = conventional_roots(&all_paths, &["src"]);
    let elixir_roots = conventional_roots(&all_paths, &["lib"]);
    let lua_roots = conventional_roots(&all_paths, &["lua", "src"]);

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
            // JVM-family: real multi-file *packages* -- `a.b` is a
            // directory that can hold many files, so a wildcard resolves
            // to every file in it (`wildcard_is_package_directory: true`).
            // Resolved against conventional Maven/Gradle source roots plus
            // the generic repo-root/immediate-subdirectory fallback every
            // hierarchical resolver here uses.
            "java" => resolve_dotted_module(&raw_path, '*', "java", &jvm_roots, &java_files_by_dir, true, &all_paths),
            "kotlin" => resolve_dotted_module(&raw_path, '*', "kt", &jvm_roots, &kotlin_files_by_dir, true, &all_paths),
            "groovy" => resolve_dotted_module(&raw_path, '*', "groovy", &jvm_roots, &groovy_files_by_dir, true, &all_paths),
            "scala" => resolve_dotted_module(&raw_path, '_', "scala", &jvm_roots, &scala_files_by_dir, true, &all_paths),
            // C# has no wildcard `using` and namespaces don't reliably
            // mirror directory structure the way Java's package
            // convention does -- still attempted, just lower-confidence.
            "csharp" => resolve_dotted_module(&raw_path, '\0', "cs", &csharp_roots, &HashMap::new(), false, &all_paths),
            // Elm/Haskell/D/Julia: one MODULE per file, not a package
            // directory -- `wildcard_is_package_directory: false` means a
            // wildcard/`exposing (..)`/unqualified-whole-module import
            // still resolves to that one file, never a directory listing
            // (an earlier version of this resolver got Elm's case wrong
            // this same way -- see `resolve_dotted_module`'s doc).
            "elm" => resolve_dotted_module(&raw_path, '*', "elm", &elm_roots, &HashMap::new(), false, &all_paths),
            "haskell" => resolve_dotted_module(&raw_path, '*', "hs", &haskell_roots, &HashMap::new(), false, &all_paths),
            "d" => resolve_dotted_module(&raw_path, '*', "d", &d_roots, &HashMap::new(), false, &all_paths),
            "julia" => resolve_dotted_module(&raw_path, '*', "jl", &julia_roots, &HashMap::new(), false, &all_paths),
            // Quoted/relative-literal-path languages, same family as the
            // C/vim/bash group above.
            "erlang" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["hrl"]),
            "zig" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["zig"]),
            "php" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["php"]),
            "latex" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["tex"]),
            "dart" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["dart"]),
            "scheme" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["scm", "ss"]),
            "powershell" => resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["ps1", "psm1"]),
            // Dedicated per-language naming conventions -- see each
            // resolver's own doc for why it can't just be another
            // `resolve_dotted_module`/`resolve_relative_literal` call.
            "ada" => resolve_ada(&raw_path, &ada_roots, &all_paths),
            "ocaml" => resolve_ocaml(&raw_path, &ocaml_roots, &all_paths),
            "perl" => resolve_perl(&raw_path, &perl_roots, &all_paths),
            "elixir" => resolve_elixir(&raw_path, &elixir_roots, &all_paths),
            // Fortran's `include 'x.inc'` is a real relative-literal path;
            // its `use module_name` is a bare module name with no
            // standardized file convention at all, so it's tried as a
            // relative-literal path first (cheap, and correct on the rare
            // repo that names files after modules verbatim) and only
            // falls back to the flat module-name guess when that comes up
            // empty.
            "fortran" => {
                let literal = resolve_relative_literal(&raw_path, &importer_segs, &all_paths, &["inc"]);
                if literal.is_empty() {
                    // A plain `use module_name` (no `only:` clause) carries
                    // a trailing `.*` wildcard marker, same convention as
                    // Haskell/D/Elixir -- strip it before treating the rest
                    // as the module's own name.
                    let module_name = raw_path.strip_suffix(".*").unwrap_or(&raw_path);
                    resolve_fortran_module(module_name, &fortran_roots, &all_paths)
                } else {
                    literal
                }
            }
            // Lua's `require("a.b")` -> `a/b.lua` via `package.path`'s dot
            // -> slash convention; no wildcard form exists, so the
            // wildcard char is disabled the same way C#'s is.
            "lua" => resolve_dotted_module(&raw_path, '\0', "lua", &lua_roots, &HashMap::new(), false, &all_paths),
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

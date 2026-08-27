//! Real per-file import-statement extraction, one bespoke AST walker per
//! language rather than the flat single-capture query style `Lang::def_query_src`/
//! `call_query_src` use.
//!
//! Why not a query, when everything else in this crate is one: import
//! syntax nests and aliases far more than a definition or call site does --
//! `use a::{b::{c, d as e}, f::*}` is a recursive tree of brace lists,
//! renames, and glob markers, not a single flat shape a `tree-sitter` query
//! pattern can capture in one pass. A dedicated walker over the real AST
//! handles that recursion directly; this is the same reason real IDE
//! tooling (rust-analyzer, ts-morph) hand-writes import resolution instead
//! of pattern-matching it.
//!
//! This module only extracts the *raw* text of each import edge (the
//! module path as written, plus which specific name(s) it binds, if any).
//! Turning that into an actual resolved file in the repo is a separate,
//! deliberately later step -- see `resolve.rs` -- since that needs the
//! whole repo's file set and per-language project metadata (`Cargo.toml`/
//! `go.mod`), not just one file's content.

use tree_sitter::{Node, Parser, StreamingIterator, Tree};

use crate::Lang;

/// One raw import edge exactly as written in the source -- not yet
/// resolved to a file. `imported_name` is the specific name bound by this
/// edge, when there is one; `None` means a whole-module import (`import
/// foo`) or a glob (`use foo::*`, `from foo import *`), which don't name
/// a specific symbol to match a call site's `callee_name` against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportEdge {
    pub raw_path: String,
    pub imported_name: Option<String>,
    /// True for an import that brings an *unknown set* of names into reach
    /// as a whole -- a real glob (`use foo::*`, `from foo import *`) or a
    /// Go import (Go has no per-name import syntax; a package import makes
    /// every exported name reachable via `pkg.Name()`, which this
    /// project's call queries do capture, selector name only). False for
    /// a plain whole-module bind (Python `import os`, a JS default/
    /// namespace import) -- those need attribute access to reach a name
    /// inside, unlike a bare-identifier call match, so they deliberately
    /// don't count as "could be anything" (see `callgraph::find_callers`).
    pub is_wildcard: bool,
    pub line: i64,
    /// Names of every *inline* `mod name { ... }` block (as opposed to a
    /// file-declaring `mod name;`) this import statement is textually
    /// nested inside, outermost first. Rust-only, and usually empty (a
    /// top-level `use`) -- but a `#[cfg(test)] mod tests { use super::*; }`
    /// is extremely common, and without this, `super`/`self` inside it
    /// would be resolved as if the whole file were one flat module,
    /// silently jumping a level too far (see `resolve.rs`'s Rust
    /// resolver, which folds this into the effective module path before
    /// applying `super`/`self`).
    pub module_prefix: Vec<String>,
}

/// Best-effort import extraction. Empty vec (never an error) for a
/// language with no import walker yet, or a grammar/parse failure -- same
/// graceful-degradation contract as `extract_symbols`/`extract_calls`: a
/// missing resolver never breaks indexing, it just means call sites in
/// that language don't get the extra `ResolutionHint::ImportResolved`
/// signal yet.
pub fn extract_imports(content: &str, language: Lang) -> Vec<ImportEdge> {
    let Some(ts_lang) = language.ts_language() else {
        return Vec::new();
    };
    let mut parser = Parser::new();
    if parser.set_language(&ts_lang).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };
    let bytes = content.as_bytes();

    match language {
        Lang::Rust => rust::imports(&tree, bytes),
        Lang::Python => python::imports(&tree, bytes),
        Lang::JavaScript | Lang::TypeScript | Lang::Tsx => js::imports(&tree, bytes),
        Lang::Go => go::imports(&tree, bytes),
        Lang::C | Lang::Cpp | Lang::ObjC | Lang::Glsl | Lang::Hlsl => query_based::preproc_include(&tree, bytes),
        Lang::Vim => query_based::vim_source(&tree, bytes),
        Lang::Proto => query_based::proto_import(&tree, bytes),
        Lang::Solidity => query_based::solidity_import(&tree, bytes),
        Lang::Verilog => query_based::verilog_include(&tree, bytes),
        Lang::Nix => query_based::nix_import(&tree, bytes),
        // Java/Kotlin/Groovy/C# all import one specific class/type per
        // statement (the last dotted segment IS the thing actually bound
        // into scope) -- `dotted_module_imports`'s `last_segment_is_name:
        // true` reflects that directly.
        Lang::Java | Lang::Groovy => query_based::dotted_module_imports(&tree, bytes, "(import_declaration [(scoped_identifier) (identifier)] @target) @import", Some('*'), true),
        Lang::Kotlin => query_based::dotted_module_imports(&tree, bytes, "(import [(qualified_identifier) (identifier)] @target) @import", Some('*'), true),
        Lang::CSharp => query_based::dotted_module_imports(&tree, bytes, "(using_directive [(qualified_name) (identifier)] @target) @import", None, true),
        Lang::Scala => query_based::scala_import(&tree, bytes),
        Lang::Elm => query_based::elm_import(&tree, bytes),
        // D/Haskell/Julia deliberately NOT handled: their plain (non-
        // `qualified`) import brings a whole module's exports into
        // *unqualified* scope, unlike Java-style "import one class" --
        // treating the module's last dotted segment as "the imported
        // name" the way Java's resolver does would be a confidently WRONG
        // signal (the module name itself is rarely a callable), and
        // guessing "wildcard" risks false corroboration for the
        // `qualified`/aliased form this project can't yet tell apart from
        // the unqualified one. Left unresolved rather than guessed.
        Lang::Bash => command_style::bash_source(&tree, bytes),
        Lang::Fish => command_style::fish_source(&tree, bytes),
        Lang::Ruby => command_style::ruby_require(&tree, bytes),
        Lang::R => command_style::r_source(&tree, bytes),
        Lang::Racket => command_style::racket_require(&tree, bytes),
        Lang::CMake => command_style::cmake_include(&tree, bytes),
        _ => Vec::new(),
    }
}

fn text(node: Node, bytes: &[u8]) -> String {
    node.utf8_text(bytes).unwrap_or("").to_string()
}

/// Depth-first collection of every node of `kind` under `root`. Does not
/// recurse *into* a matched node -- import statements never nest inside
/// each other, so once one is found there's nothing further to find below
/// it worth walking into separately.
fn find_nodes<'a>(root: Node<'a>, kind: &str, out: &mut Vec<Node<'a>>) {
    if root.kind() == kind {
        out.push(root);
        return;
    }
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        find_nodes(child, kind, out);
    }
}

mod rust {
    use super::*;

    pub(super) fn imports(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut decls = Vec::new();
        find_nodes(tree.root_node(), "use_declaration", &mut decls);
        let mut out = Vec::new();
        for decl in decls {
            let line = decl.start_position().row as i64 + 1;
            let module_prefix = inline_module_prefix(decl, bytes);
            if let Some(arg) = decl.child_by_field_name("argument") {
                walk_use_tree(arg, &[], bytes, line, &module_prefix, &mut out);
            }
        }
        out
    }

    /// Names of every *inline* `mod name { ... }` block `node` is nested
    /// inside, outermost first -- e.g. `["tests"]` for a `use` inside
    /// `#[cfg(test)] mod tests { ... }`. A `mod name;` (file-declaring, no
    /// body) can never be an ancestor of anything, since it has no body
    /// for anything to nest inside -- so every `mod_item` ancestor found
    /// here is inherently inline.
    fn inline_module_prefix(node: Node, bytes: &[u8]) -> Vec<String> {
        let mut mods = Vec::new();
        let mut current = node.parent();
        while let Some(n) = current {
            if n.kind() == "mod_item" {
                if let Some(name) = n.child_by_field_name("name") {
                    mods.push(text(name, bytes));
                }
            }
            current = n.parent();
        }
        mods.reverse();
        mods
    }

    /// Flattens any path-shaped node (`identifier`/`crate`/`self`/`super`/
    /// `scoped_identifier`/`metavariable`) into `prefix`, in order.
    fn push_path_segment(node: Node, bytes: &[u8], prefix: &mut Vec<String>) {
        match node.kind() {
            "scoped_identifier" => {
                if let Some(path) = node.child_by_field_name("path") {
                    push_path_segment(path, bytes, prefix);
                }
                if let Some(name) = node.child_by_field_name("name") {
                    prefix.push(text(name, bytes));
                }
            }
            "identifier" | "crate" | "self" | "super" | "metavariable" => prefix.push(text(node, bytes)),
            _ => {}
        }
    }

    fn walk_use_tree(node: Node, prefix: &[String], bytes: &[u8], line: i64, module_prefix: &[String], out: &mut Vec<ImportEdge>) {
        match node.kind() {
            "identifier" | "crate" | "self" | "super" | "metavariable" => {
                let name = text(node, bytes);
                let mut path = prefix.to_vec();
                path.push(name.clone());
                out.push(ImportEdge { raw_path: path.join("::"), imported_name: Some(name), is_wildcard: false, line, module_prefix: module_prefix.to_vec() });
            }
            "scoped_identifier" => {
                let mut path = prefix.to_vec();
                push_path_segment(node, bytes, &mut path);
                let name = path.last().cloned().unwrap_or_default();
                out.push(ImportEdge { raw_path: path.join("::"), imported_name: Some(name), is_wildcard: false, line, module_prefix: module_prefix.to_vec() });
            }
            "scoped_use_list" => {
                let mut new_prefix = prefix.to_vec();
                if let Some(path) = node.child_by_field_name("path") {
                    push_path_segment(path, bytes, &mut new_prefix);
                }
                if let Some(list) = node.child_by_field_name("list") {
                    let mut cursor = list.walk();
                    for child in list.named_children(&mut cursor) {
                        walk_use_tree(child, &new_prefix, bytes, line, module_prefix, out);
                    }
                }
            }
            "use_list" => {
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    walk_use_tree(child, prefix, bytes, line, module_prefix, out);
                }
            }
            "use_as_clause" => {
                if let Some(path) = node.child_by_field_name("path") {
                    let mut new_prefix = prefix.to_vec();
                    push_path_segment(path, bytes, &mut new_prefix);
                    let name = new_prefix.last().cloned().unwrap_or_default();
                    out.push(ImportEdge { raw_path: new_prefix.join("::"), imported_name: Some(name), is_wildcard: false, line, module_prefix: module_prefix.to_vec() });
                }
            }
            "use_wildcard" => {
                let mut new_prefix = prefix.to_vec();
                let mut cursor = node.walk();
                if let Some(path) = node.named_children(&mut cursor).next() {
                    push_path_segment(path, bytes, &mut new_prefix);
                }
                out.push(ImportEdge { raw_path: format!("{}::*", new_prefix.join("::")), imported_name: None, is_wildcard: true, line, module_prefix: module_prefix.to_vec() });
            }
            _ => {}
        }
    }
}

mod python {
    use super::*;

    pub(super) fn imports(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut plain = Vec::new();
        find_nodes(tree.root_node(), "import_statement", &mut plain);
        for node in plain {
            let line = node.start_position().row as i64 + 1;
            let mut cursor = node.walk();
            for name_node in node.children_by_field_name("name", &mut cursor) {
                match name_node.kind() {
                    "dotted_name" => out.push(ImportEdge { raw_path: dotted_name_text(name_node, bytes), imported_name: None, is_wildcard: false, line, module_prefix: Vec::new() }),
                    "aliased_import" => {
                        if let Some(dotted) = name_node.child_by_field_name("name") {
                            out.push(ImportEdge { raw_path: dotted_name_text(dotted, bytes), imported_name: None, is_wildcard: false, line, module_prefix: Vec::new() });
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut from_stmts = Vec::new();
        find_nodes(tree.root_node(), "import_from_statement", &mut from_stmts);
        for node in from_stmts {
            let line = node.start_position().row as i64 + 1;
            let Some(module) = node.child_by_field_name("module_name") else { continue };
            let module_path = module_path_text(module, bytes);

            let has_wildcard = {
                let mut cursor = node.walk();
                node.children(&mut cursor).any(|c| c.kind() == "wildcard_import")
            };
            if has_wildcard {
                out.push(ImportEdge { raw_path: format!("{module_path}.*"), imported_name: None, is_wildcard: true, line, module_prefix: Vec::new() });
                continue;
            }
            let mut cursor = node.walk();
            for name_node in node.children_by_field_name("name", &mut cursor) {
                match name_node.kind() {
                    "dotted_name" => {
                        // A single bare identifier here (`from a.b import c`)
                        // -- dotted_name can technically hold more than one
                        // segment if the grammar allows it, but in practice
                        // this field only ever holds a plain name.
                        let name = dotted_name_text(name_node, bytes);
                        out.push(ImportEdge { raw_path: module_path.clone(), imported_name: Some(name), is_wildcard: false, line, module_prefix: Vec::new() });
                    }
                    "aliased_import" => {
                        if let Some(dotted) = name_node.child_by_field_name("name") {
                            let name = dotted_name_text(dotted, bytes);
                            out.push(ImportEdge { raw_path: module_path.clone(), imported_name: Some(name), is_wildcard: false, line, module_prefix: Vec::new() });
                        }
                    }
                    _ => {}
                }
            }
        }
        out
    }

    fn dotted_name_text(node: Node, bytes: &[u8]) -> String {
        let mut cursor = node.walk();
        node.named_children(&mut cursor).map(|c| text(c, bytes)).collect::<Vec<_>>().join(".")
    }

    /// `a.b` for a plain `dotted_name`, or `.`/`..pkg` for a
    /// `relative_import` (leading dots preserved literally so the resolver
    /// can tell "same package" from "two packages up" apart).
    fn module_path_text(node: Node, bytes: &[u8]) -> String {
        match node.kind() {
            "dotted_name" => dotted_name_text(node, bytes),
            "relative_import" => {
                let mut dots = String::new();
                let mut rest = String::new();
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "import_prefix" => dots.push_str(&text(child, bytes)),
                        "dotted_name" => rest = dotted_name_text(child, bytes),
                        _ => {}
                    }
                }
                format!("{dots}{rest}")
            }
            _ => String::new(),
        }
    }
}

mod js {
    use super::*;

    pub(super) fn imports(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut stmts = Vec::new();
        find_nodes(tree.root_node(), "import_statement", &mut stmts);
        for node in stmts {
            let line = node.start_position().row as i64 + 1;
            let Some(source) = node.child_by_field_name("source") else { continue };
            let raw_path = string_text(source, bytes);

            let mut cursor = node.walk();
            let clause = node.children(&mut cursor).find(|c| c.kind() == "import_clause");
            let Some(clause) = clause else {
                // Side-effect-only import (`import './styles.css'`) --
                // still a real edge worth recording, just no bound name.
                out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, line, module_prefix: Vec::new() });
                continue;
            };
            let mut ccursor = clause.walk();
            let mut bound_any = false;
            for child in clause.children(&mut ccursor) {
                match child.kind() {
                    "identifier" => {
                        // default import: `import Foo from './x'` -- the
                        // bound local name doesn't tell us the *exported*
                        // name in the target module (commonly `default`
                        // itself), so this is recorded as a whole-module
                        // edge rather than a specific name match.
                        bound_any = true;
                        out.push(ImportEdge { raw_path: raw_path.clone(), imported_name: None, is_wildcard: false, line, module_prefix: Vec::new() });
                    }
                    "namespace_import" => {
                        bound_any = true;
                        out.push(ImportEdge { raw_path: raw_path.clone(), imported_name: None, is_wildcard: false, line, module_prefix: Vec::new() });
                    }
                    "named_imports" => {
                        let mut ncursor = child.walk();
                        for spec in child.children(&mut ncursor) {
                            if spec.kind() != "import_specifier" {
                                continue;
                            }
                            if let Some(name_node) = spec.child_by_field_name("name") {
                                if name_node.kind() == "identifier" {
                                    bound_any = true;
                                    out.push(ImportEdge { raw_path: raw_path.clone(), imported_name: Some(text(name_node, bytes)), is_wildcard: false, line, module_prefix: Vec::new() });
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !bound_any {
                out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, line, module_prefix: Vec::new() });
            }
        }
        out
    }

    fn string_text(node: Node, bytes: &[u8]) -> String {
        let mut cursor = node.walk();
        node.children(&mut cursor)
            .find(|c| c.kind() == "string_fragment")
            .map(|c| text(c, bytes))
            .unwrap_or_default()
    }
}

mod go {
    use super::*;

    pub(super) fn imports(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut specs = Vec::new();
        find_nodes(tree.root_node(), "import_spec", &mut specs);
        for spec in specs {
            let line = spec.start_position().row as i64 + 1;
            // Blank (`_`) and dot (`.`) imports are still real edges
            // (side-effect-only, or "everything exported" respectively) --
            // recorded as whole-package edges either way.
            let Some(path_node) = spec.child_by_field_name("path") else { continue };
            let raw = text(path_node, bytes);
            let raw_path = raw.trim_matches(|c| c == '"' || c == '`').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: true, line, module_prefix: Vec::new() });
        }
        out
    }
}

/// Shared machinery for languages whose import syntax has a single
/// dedicated grammar node (no ambiguity with an ordinary call/command, so
/// no keyword-filtering predicate is ever needed) -- a `tree-sitter`
/// query with `@import` (the whole statement, for its line) and `@target`
/// (the path/module text) captures is enough, unlike Rust's nested `use`
/// lists.
mod query_based {
    use super::*;
    use std::collections::HashMap;

    /// Runs `query_src` against `tree` and returns each match as a
    /// name -> node map, so a per-language function can just look up
    /// `"target"`/`"import"`(/anything else it declared) by name instead
    /// of re-deriving capture indices by hand every time.
    fn query_matches<'a>(tree: &'a Tree, bytes: &'a [u8], query_src: &str) -> Vec<HashMap<String, Node<'a>>> {
        let Ok(query) = tree_sitter::Query::new(&tree.language(), query_src) else { return Vec::new() };
        let names = query.capture_names();
        let mut out = Vec::new();
        let mut cursor = tree_sitter::QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), bytes);
        while let Some(m) = matches.next() {
            let mut map = HashMap::new();
            for cap in m.captures {
                map.insert(names[cap.index as usize].to_string(), cap.node);
            }
            out.push(map);
        }
        out
    }

    /// `#include "foo.h"` (C/C++/Objective-C/GLSL/HLSL all share this
    /// grammar node). `#include <foo.h>` (`system_lib_string`) is always a
    /// system header, never a repo-local file -- skipped at extraction
    /// time rather than recorded as a doomed-to-fail edge.
    pub(super) fn preproc_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(preproc_include path: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            if target.kind() == "system_lib_string" {
                continue;
            }
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Vim's `:source file.vim` has its own dedicated grammar node --
    /// unlike Bash/Fish's `source`, which is an ordinary command.
    pub(super) fn vim_source(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(source_statement file: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    pub(super) fn proto_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(import path: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    pub(super) fn solidity_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(import_directive source: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `` `include "file.v" `` -- SystemVerilog/Verilog's compiler
    /// directive. The double-quoted string is a direct (unnamed) child,
    /// not a field.
    pub(super) fn verilog_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(include_compiler_directive (double_quoted_string) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `import ./file.nix` -- Nix's `import` is an ordinary builtin
    /// function, not special grammar (hence the `#_fn` text check rather
    /// than a dedicated node kind), but its argument is a real
    /// `path_expression` node whose text is already the literal relative
    /// path -- nothing to strip.
    pub(super) fn nix_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(apply_expression function: (_) @_fn argument: (path_expression) @target) @import") {
            let (Some(&fn_node), Some(&target), Some(&import_node)) = (m.get("_fn"), m.get("target"), m.get("import")) else { continue };
            if text(fn_node, bytes) != "import" {
                continue;
            }
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes);
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Shared by Java/Kotlin/Groovy/C# (via `extract_imports`'s dispatch):
    /// a dotted/qualified path node whose whole span is already the
    /// literal `a.b.C` text (no brace-list nesting to walk, unlike Rust).
    /// `wildcard_char` is the language's own glob marker (`*` for Java/
    /// Kotlin/Groovy, `None` for C# which has none); `last_segment_is_name`
    /// is true for these four because each import statement always names
    /// one specific class/type as its last segment -- that segment really
    /// is the thing bound into scope, unlike a language where a plain
    /// import exposes a whole module's names unqualified (see the
    /// deliberate D/Haskell/Julia omission in `extract_imports`).
    pub(super) fn dotted_module_imports(tree: &Tree, bytes: &[u8], query_src: &str, wildcard_char: Option<char>, last_segment_is_name: bool) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, query_src) {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes);
            let stmt_text = text(import_node, bytes);
            let is_wildcard = wildcard_char.is_some_and(|c| stmt_text.trim_end_matches(';').trim_end().ends_with(c));
            let raw_path = if is_wildcard { format!("{raw}.{}", wildcard_char.unwrap()) } else { raw.clone() };
            let imported_name = if is_wildcard { None } else if last_segment_is_name { raw.rsplit('.').next().map(String::from) } else { None };
            out.push(ImportEdge { raw_path, imported_name, is_wildcard, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Scala's `import_declaration` spreads its dotted path across
    /// several sibling tokens (`.`/`identifier`/`operator_identifier`)
    /// rather than one contiguous node, and uses `_` (not `*`) for a
    /// wildcard -- simplest to just read the whole statement's own text
    /// and trim the `import ` keyword off the front, rather than
    /// reconstructing the path from scattered children.
    pub(super) fn scala_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(import_declaration) @import") {
            let Some(&import_node) = m.get("import") else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let stmt = text(import_node, bytes);
            let Some(rest) = stmt.trim_end_matches(';').trim().strip_prefix("import").map(str::trim) else { continue };
            // Only the first comma-separated path (`import a.B, c.D` is
            // rare in real Scala code; handling just the common single-
            // path case here, same trade-off this project makes
            // elsewhere for genuinely rare syntax forms).
            let first = rest.split(',').next().unwrap_or(rest).trim();
            // Strip a trailing `{...}` selector clause (`import a.b.{C, D}`)
            // -- not walked into for individual names, same simplification
            // as HCL's first-label-only choice elsewhere in this project.
            let raw = first.split('{').next().unwrap_or(first).trim().trim_end_matches('.').to_string();
            if raw.is_empty() {
                continue;
            }
            let is_wildcard = raw.ends_with('_') || first.contains('{');
            let raw_path = if is_wildcard { format!("{}.{}", raw.trim_end_matches('_').trim_end_matches('.'), '_') } else { raw.clone() };
            let imported_name = if is_wildcard { None } else { raw.rsplit('.').next().map(String::from) };
            out.push(ImportEdge { raw_path, imported_name, is_wildcard, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Elm's `exposing (...)` list is a real, precisely capturable
    /// analogue of Python's `from X import a, b` -- `exposing (..)`
    /// (double-dot) is an explicit "expose everything" wildcard, distinct
    /// from a plain `import Html` (which binds only the qualified module
    /// name, not its contents, into scope -- so gets no `imported_name`
    /// at all, never a guess).
    pub(super) fn elm_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(import_clause moduleName: (upper_case_qid) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes);
            let exposing = import_node.child_by_field_name("exposing").map(|n| text(n, bytes));
            match exposing {
                None => out.push(ImportEdge { raw_path: raw, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line }),
                Some(list) if list.contains("..") => {
                    out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
                }
                Some(list) => {
                    // `exposing`'s captured field text includes the
                    // `exposing` keyword itself, not just the parenthesized
                    // list (e.g. "exposing (class, id)") -- slicing between
                    // the first `(` and last `)` gets just the names,
                    // regardless of exactly what precedes them.
                    let inner = list.split_once('(').map(|(_, rest)| rest).unwrap_or(&list);
                    let inner = inner.strip_suffix(')').unwrap_or(inner);
                    for name in inner.split(',') {
                        let name = name.trim();
                        if name.is_empty() {
                            continue;
                        }
                        out.push(ImportEdge { raw_path: raw.clone(), imported_name: Some(name.to_string()), is_wildcard: false, module_prefix: Vec::new(), line });
                    }
                }
            }
        }
        out
    }
}

/// Languages whose `source`/`require`-style import is an *ordinary*
/// command/function call at the grammar level, not a dedicated node --
/// filtering to the right command name is done here, in plain Rust code
/// after the query runs, rather than via a `tree-sitter` text predicate.
/// This project has been burned once already by a predicate silently not
/// applying (see `lang.rs`'s query-safety-net doc); for these lower-
/// traffic languages, a straightforward Rust `==`/`contains` check is the
/// safer bet, not just a shorter one.
mod command_style {
    use super::*;

    fn strip_quotes(s: &str) -> String {
        s.trim_matches(|c| c == '"' || c == '\'').to_string()
    }

    /// `source file.sh` / `. file.sh`.
    pub(super) fn bash_source(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut cmds = Vec::new();
        find_nodes(tree.root_node(), "command", &mut cmds);
        for cmd in cmds {
            let Some(name_node) = cmd.child_by_field_name("name") else { continue };
            let name = text(name_node, bytes);
            if name != "source" && name != "." {
                continue;
            }
            let mut cursor = cmd.walk();
            let Some(arg) = cmd.children_by_field_name("argument", &mut cursor).next() else { continue };
            let line = cmd.start_position().row as i64 + 1;
            out.push(ImportEdge { raw_path: strip_quotes(&text(arg, bytes)), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `source file.fish`.
    pub(super) fn fish_source(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut cmds = Vec::new();
        find_nodes(tree.root_node(), "command", &mut cmds);
        for cmd in cmds {
            let Some(name_node) = cmd.child_by_field_name("name") else { continue };
            if text(name_node, bytes) != "source" {
                continue;
            }
            let mut cursor = cmd.walk();
            let Some(arg) = cmd.children_by_field_name("argument", &mut cursor).next() else { continue };
            let line = cmd.start_position().row as i64 + 1;
            out.push(ImportEdge { raw_path: strip_quotes(&text(arg, bytes)), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `require 'x'` / `require_relative './x'` / `load 'x.rb'`.
    pub(super) fn ruby_require(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut calls = Vec::new();
        find_nodes(tree.root_node(), "call", &mut calls);
        for call in calls {
            let Some(method) = call.child_by_field_name("method") else { continue };
            let name = text(method, bytes);
            if !matches!(name.as_str(), "require" | "require_relative" | "load") {
                continue;
            }
            let Some(args) = call.child_by_field_name("arguments") else { continue };
            let mut cursor = args.walk();
            let Some(first_str) = args.named_children(&mut cursor).find(|c| c.kind() == "string") else { continue };
            let line = call.start_position().row as i64 + 1;
            out.push(ImportEdge { raw_path: strip_quotes(&text(first_str, bytes)), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `source("file.R")`.
    pub(super) fn r_source(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut calls = Vec::new();
        find_nodes(tree.root_node(), "call", &mut calls);
        for call in calls {
            let Some(func) = call.child_by_field_name("function") else { continue };
            if text(func, bytes) != "source" {
                continue;
            }
            let Some(args) = call.child_by_field_name("arguments") else { continue };
            let mut cursor = args.walk();
            let first_str = args
                .named_children(&mut cursor)
                .filter(|c| c.kind() == "argument")
                .find_map(|a| a.named_child(0).filter(|v| v.kind() == "string"));
            let Some(first_str) = first_str else { continue };
            let line = call.start_position().row as i64 + 1;
            out.push(ImportEdge { raw_path: strip_quotes(&text(first_str, bytes)), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `(require "file.rkt")` -- a plain relative-path require, as
    /// opposed to `(require racket/base)` naming a library collection
    /// (bare symbol, not a repo-local file) -- only the string form is
    /// recorded, the symbol form is correctly left unextracted since it
    /// virtually never names a file in *this* repo.
    pub(super) fn racket_require(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut lists = Vec::new();
        find_nodes(tree.root_node(), "list", &mut lists);
        for list in lists {
            let mut cursor = list.walk();
            let mut children = list.named_children(&mut cursor);
            let Some(head) = children.next() else { continue };
            if head.kind() != "symbol" || text(head, bytes) != "require" {
                continue;
            }
            let Some(arg) = children.next() else { continue };
            if !arg.kind().contains("string") {
                continue;
            }
            let line = list.start_position().row as i64 + 1;
            out.push(ImportEdge { raw_path: strip_quotes(&text(arg, bytes)), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `include(file.cmake)` / `add_subdirectory(dir)` -- CMake command
    /// names are case-insensitive by language spec, so the check is
    /// case-folded rather than a literal `==`.
    pub(super) fn cmake_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut cmds = Vec::new();
        find_nodes(tree.root_node(), "normal_command", &mut cmds);
        for cmd in cmds {
            let mut cursor = cmd.walk();
            let Some(name_node) = cmd.children(&mut cursor).find(|c| c.kind() == "identifier") else { continue };
            let name = text(name_node, bytes).to_ascii_lowercase();
            if name != "include" && name != "add_subdirectory" {
                continue;
            }
            let mut cursor2 = cmd.walk();
            let Some(arg_list) = cmd.children(&mut cursor2).find(|c| c.kind() == "argument_list") else { continue };
            let mut cursor3 = arg_list.walk();
            let Some(first_arg) = arg_list.named_children(&mut cursor3).next() else { continue };
            let line = cmd.start_position().row as i64 + 1;
            out.push(ImportEdge { raw_path: strip_quotes(&text(first_arg, bytes)), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges(content: &str, lang: Lang) -> Vec<(String, Option<String>)> {
        extract_imports(content, lang).into_iter().map(|e| (e.raw_path, e.imported_name)).collect()
    }

    #[test]
    fn rust_plain_and_nested_and_aliased_and_glob() {
        let src = "use crate::foo::Bar;\nuse super::helper;\nuse std::collections::{HashMap, HashSet as Set};\nuse serde::{Deserialize, Serialize};\nuse tokio::*;\n";
        let got = edges(src, Lang::Rust);
        assert!(got.contains(&("crate::foo::Bar".to_string(), Some("Bar".to_string()))), "{got:?}");
        assert!(got.contains(&("super::helper".to_string(), Some("helper".to_string()))), "{got:?}");
        assert!(got.contains(&("std::collections::HashMap".to_string(), Some("HashMap".to_string()))), "{got:?}");
        assert!(got.contains(&("std::collections::HashSet".to_string(), Some("HashSet".to_string()))), "{got:?}", );
        assert!(got.contains(&("serde::Deserialize".to_string(), Some("Deserialize".to_string()))), "{got:?}");
        assert!(got.contains(&("tokio::*".to_string(), None)), "{got:?}");
    }

    #[test]
    fn python_plain_from_relative_and_wildcard() {
        let src = "import os\nfrom collections import OrderedDict\nfrom . import models\nfrom ..pkg import util as u\nfrom foo import *\n";
        let got = edges(src, Lang::Python);
        assert!(got.contains(&("os".to_string(), None)), "{got:?}");
        assert!(got.contains(&("collections".to_string(), Some("OrderedDict".to_string()))), "{got:?}");
        assert!(got.contains(&(".".to_string(), Some("models".to_string()))), "{got:?}");
        assert!(got.contains(&("..pkg".to_string(), Some("util".to_string()))), "{got:?}");
        assert!(got.contains(&("foo.*".to_string(), None)), "{got:?}");
    }

    #[test]
    fn js_relative_named_default_and_bare_specifier() {
        let src = "import { helper, other as o } from './utils';\nimport Foo from '../foo';\nimport React from 'react';\nimport './styles.css';\n";
        let got = edges(src, Lang::JavaScript);
        assert!(got.contains(&("./utils".to_string(), Some("helper".to_string()))), "{got:?}");
        assert!(got.contains(&("./utils".to_string(), Some("other".to_string()))), "{got:?}");
        assert!(got.contains(&("../foo".to_string(), None)), "{got:?}"); // default import: whole-module edge
        assert!(got.contains(&("react".to_string(), None)), "{got:?}"); // bare specifier -- still recorded, resolver will skip it
        assert!(got.contains(&("./styles.css".to_string(), None)), "{got:?}");
    }

    #[test]
    fn go_plain_and_aliased_imports() {
        let src = "package main\n\nimport (\n    \"fmt\"\n    u \"myproject/pkg/utils\"\n)\n";
        let got = edges(src, Lang::Go);
        assert!(got.contains(&("fmt".to_string(), None)), "{got:?}");
        assert!(got.contains(&("myproject/pkg/utils".to_string(), None)), "{got:?}");
    }

    #[test]
    fn c_family_include_skips_system_headers() {
        for lang in [Lang::C, Lang::Cpp, Lang::ObjC, Lang::Glsl, Lang::Hlsl] {
            let src = "#include \"helper.h\"\n#include <stdio.h>\n";
            let got = edges(src, lang);
            assert!(got.contains(&("helper.h".to_string(), None)), "{lang:?}: {got:?}");
            assert!(!got.iter().any(|(p, _)| p.contains("stdio")), "{lang:?}: system header must stay unresolved: {got:?}");
        }
    }

    #[test]
    fn vim_source_statement() {
        let got = edges(":source helper.vim\n", Lang::Vim);
        assert!(got.iter().any(|(p, _)| p.contains("helper.vim")), "{got:?}");
    }

    #[test]
    fn proto_import() {
        let got = edges("syntax = \"proto3\";\nimport \"other.proto\";\n", Lang::Proto);
        assert!(got.contains(&("other.proto".to_string(), None)), "{got:?}");
    }

    #[test]
    fn solidity_import() {
        let got = edges("import \"./Other.sol\";\n", Lang::Solidity);
        assert!(got.contains(&("./Other.sol".to_string(), None)), "{got:?}");
    }

    #[test]
    fn verilog_include_directive() {
        let got = edges("`include \"other.vh\"\nmodule m; endmodule\n", Lang::Verilog);
        assert!(got.contains(&("other.vh".to_string(), None)), "{got:?}");
    }

    #[test]
    fn nix_import_expression() {
        let got = edges("import ./other.nix\n", Lang::Nix);
        assert!(got.contains(&("./other.nix".to_string(), None)), "{got:?}");
    }

    #[test]
    fn java_plain_and_wildcard_import() {
        let got = edges("import a.b.Helper;\nimport a.c.*;\n", Lang::Java);
        assert!(got.contains(&("a.b.Helper".to_string(), Some("Helper".to_string()))), "{got:?}");
        assert!(got.contains(&("a.c.*".to_string(), None)), "{got:?}");
    }

    #[test]
    fn groovy_plain_import() {
        let got = edges("import a.b.Helper\n", Lang::Groovy);
        assert!(got.contains(&("a.b.Helper".to_string(), Some("Helper".to_string()))), "{got:?}");
    }

    #[test]
    fn kotlin_plain_and_wildcard_import() {
        let got = edges("import a.b.Helper\nimport a.c.*\n", Lang::Kotlin);
        assert!(got.contains(&("a.b.Helper".to_string(), Some("Helper".to_string()))), "{got:?}");
        assert!(got.contains(&("a.c.*".to_string(), None)), "{got:?}");
    }

    #[test]
    fn csharp_using_directive() {
        let got = edges("using System.Collections.Generic;\n", Lang::CSharp);
        assert!(got.contains(&("System.Collections.Generic".to_string(), Some("Generic".to_string()))), "{got:?}");
    }

    #[test]
    fn scala_plain_and_wildcard_import() {
        let got = edges("import a.b.Helper\nimport a.c._\n", Lang::Scala);
        assert!(got.contains(&("a.b.Helper".to_string(), Some("Helper".to_string()))), "{got:?}");
        assert!(got.iter().any(|(p, n)| p == "a.c._" && n.is_none()), "{got:?}");
    }

    #[test]
    fn elm_plain_and_exposing_and_exposing_all() {
        let src = "import Html\nimport Html.Attributes exposing (class, id)\nimport Html.Events exposing (..)\n";
        let got = edges(src, Lang::Elm);
        assert!(got.contains(&("Html".to_string(), None)), "{got:?}");
        assert!(got.contains(&("Html.Attributes".to_string(), Some("class".to_string()))), "{got:?}");
        assert!(got.contains(&("Html.Attributes".to_string(), Some("id".to_string()))), "{got:?}");
        assert!(got.contains(&("Html.Events.*".to_string(), None)), "{got:?}");
    }

    #[test]
    fn bash_source_and_dot() {
        let got = edges("source ./helper.sh\n. ./other.sh\n", Lang::Bash);
        assert!(got.contains(&("./helper.sh".to_string(), None)), "{got:?}");
        assert!(got.contains(&("./other.sh".to_string(), None)), "{got:?}");
    }

    #[test]
    fn fish_source() {
        let got = edges("source ./helper.fish\n", Lang::Fish);
        assert!(got.contains(&("./helper.fish".to_string(), None)), "{got:?}");
    }

    #[test]
    fn ruby_require_variants() {
        let src = "require 'json'\nrequire_relative './helper'\nload 'other.rb'\n";
        let got = edges(src, Lang::Ruby);
        assert!(got.contains(&("json".to_string(), None)), "{got:?}");
        assert!(got.contains(&("./helper".to_string(), None)), "{got:?}");
        assert!(got.contains(&("other.rb".to_string(), None)), "{got:?}");
    }

    #[test]
    fn r_source_call() {
        let got = edges("source(\"helper.R\")\n", Lang::R);
        assert!(got.contains(&("helper.R".to_string(), None)), "{got:?}");
    }

    #[test]
    fn racket_require_string_only() {
        let src = "(require \"helper.rkt\")\n(require racket/base)\n";
        let got = edges(src, Lang::Racket);
        assert!(got.contains(&("helper.rkt".to_string(), None)), "{got:?}");
        assert!(!got.iter().any(|(p, _)| p.contains("racket/base")), "bare collection symbol must not be extracted: {got:?}");
    }

    #[test]
    fn cmake_include_and_add_subdirectory_case_insensitive() {
        let src = "include(helper.cmake)\nADD_SUBDIRECTORY(sub)\n";
        let got = edges(src, Lang::CMake);
        assert!(got.contains(&("helper.cmake".to_string(), None)), "{got:?}");
        assert!(got.contains(&("sub".to_string(), None)), "{got:?}");
    }
}

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
        // D/Haskell/Julia: their plain (non-`qualified`) import brings a
        // whole module's exports into *unqualified* scope, unlike Java-
        // style "import one class" -- so each gets its own function that
        // tells `qualified`/`hiding`/an explicit selection/`import` vs
        // `using` apart, rather than being forced through
        // `dotted_module_imports`'s "last segment is the name" assumption
        // (which would be a confidently wrong signal here). See each
        // function's doc for the exact per-language rules.
        Lang::Haskell => query_based::haskell_import(&tree, bytes),
        Lang::D => query_based::d_import(&tree, bytes),
        Lang::Julia => query_based::julia_import(&tree, bytes),
        Lang::Bash => command_style::bash_source(&tree, bytes),
        Lang::Fish => command_style::fish_source(&tree, bytes),
        Lang::Ruby => command_style::ruby_require(&tree, bytes),
        Lang::R => command_style::r_source(&tree, bytes),
        Lang::Racket => command_style::racket_require(&tree, bytes),
        Lang::CMake => command_style::cmake_include(&tree, bytes),
        Lang::Erlang => query_based::erlang_include(&tree, bytes),
        Lang::Zig => query_based::zig_import(&tree, bytes),
        Lang::Php => query_based::php_include(&tree, bytes),
        Lang::Latex => query_based::latex_include(&tree, bytes),
        Lang::Dart => query_based::dart_import(&tree, bytes),
        Lang::Ada => query_based::ada_with(&tree, bytes),
        Lang::OCaml => query_based::ocaml_open(&tree, bytes),
        Lang::Perl => query_based::perl_use(&tree, bytes),
        Lang::Fortran => query_based::fortran_include(&tree, bytes),
        Lang::Elixir => query_based::elixir_alias(&tree, bytes),
        Lang::Lua => command_style::lua_require(&tree, bytes),
        Lang::Scheme => command_style::scheme_include(&tree, bytes),
        Lang::PowerShell => command_style::powershell_dotsource(&tree, bytes),
        Lang::Asm => query_based::asm_include(&tree, bytes),
        Lang::Swift => query_based::swift_import(&tree, bytes),
        Lang::Hcl => query_based::hcl_module_source(&tree, bytes),
        // Nim (immature 0.1.0 grammar with no import-related node found in
        // a real check -- rather than guess at an unverified shape), VHDL
        // (no reliable package-to-file naming convention exists to
        // resolve against at all, unlike every language above), Prolog
        // (directive shape too uncertain to encode safely from this
        // batch's investigation), and Crystal (confirmed via a real debug
        // dump that tree-sitter-crystal 0.1.0 does not parse `require
        // "..."` -- with or without parens -- as a call/macro-invocation
        // node at all; it splits into two unrelated `expression_statement`
        // nodes, so there is no node shape to hook a resolver onto yet)
        // are deliberately left unhandled. GraphQL and WGSL have no
        // import/include concept in their own spec at all (checked
        // against their real node-types.json -- no node type name even
        // contains "import"/"include") -- correctly nothing to extract,
        // not a gap. Svelte/Vue's real imports live inside a `<script>`
        // block that these grammars parse as one opaque `raw_text` node
        // (same limitation already documented for `def_query_src` --
        // recovering them needs a second, language-injection parse pass
        // this project doesn't do). See ARCHITECTURE.md's "Import
        // resolution" section for the full accounting.
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

/// Like `find_nodes`, but keeps descending *into* a match too --
/// needed wherever a node of the target kind can genuinely nest inside
/// another one of the same kind (Elixir's `call` node: `defmodule Foo do
/// ... end` is itself a `call`, whose `do_block` can contain more `call`s
/// for `alias`/`import`/`require`/`use`; stopping at the outer match would
/// silently miss every one of those).
fn find_nodes_nested<'a>(root: Node<'a>, kind: &str, out: &mut Vec<Node<'a>>) {
    if root.kind() == kind {
        out.push(root);
    }
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        find_nodes_nested(child, kind, out);
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

    /// Haskell's `import [qualified] Module [as Alias] [hiding] (names)`.
    /// Unlike Java-style imports, a plain (non-`qualified`) `import
    /// Data.List` brings *all* of `Data.List`'s exports into unqualified
    /// scope -- much closer to a wildcard than to "import one class" --
    /// so this is handled on its own rather than forced through
    /// `dotted_module_imports`'s "last segment is the name" assumption
    /// (which was the whole reason D/Haskell/Julia were left unresolved
    /// in the first place; see `resolve.rs`'s `resolve_dotted_module` doc
    /// for the matching fix on the resolution side). An explicit `(names)`
    /// list is captured precisely, one `imported_name` per name -- the
    /// same real signal Python's `from X import a, b` and Elm's
    /// `exposing (a, b)` already get. `qualified` (with or without `as`)
    /// means names are never brought in unqualified, so it's treated like
    /// a plain whole-module bind, not a wildcard -- and `hiding` still
    /// exposes everything else unqualified, so it's treated as a wildcard
    /// too (the handful of hidden names being counted as "maybe from
    /// here" is a minor imprecision, not a wrong *file* resolution).
    pub(super) fn haskell_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(import module: (module) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes);
            let before_module = std::str::from_utf8(&bytes[import_node.start_byte()..target.start_byte()]).unwrap_or("");
            let is_qualified = import_node.child_by_field_name("alias").is_some() || before_module.contains("qualified");

            match import_node.child_by_field_name("names") {
                Some(names_node) => {
                    let before_names = std::str::from_utf8(&bytes[import_node.start_byte()..names_node.start_byte()]).unwrap_or("");
                    if before_names.contains("hiding") {
                        if is_qualified {
                            out.push(ImportEdge { raw_path: raw.clone(), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
                        } else {
                            out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
                        }
                    } else {
                        let mut cursor = names_node.walk();
                        for name_node in names_node.named_children(&mut cursor) {
                            if name_node.kind() != "import_name" {
                                continue;
                            }
                            out.push(ImportEdge { raw_path: raw.clone(), imported_name: Some(text(name_node, bytes)), is_wildcard: false, module_prefix: Vec::new(), line });
                        }
                    }
                }
                None if is_qualified => {
                    out.push(ImportEdge { raw_path: raw.clone(), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
                }
                None => {
                    out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
                }
            }
        }
        out
    }

    /// D's `[static] import Module [: names] [= alias];`. Plain `import
    /// std.stdio;` brings all of `stdio`'s exports into unqualified
    /// scope -- a wildcard, same reasoning as Haskell's plain import.
    /// `static import` and a renamed `import io = std.stdio;` both
    /// require explicit qualification, so neither counts as a wildcard;
    /// a selective `: writeln, write` list is captured precisely.
    pub(super) fn d_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        // `module_fqn` sits inside an `imported` wrapper node -- always
        // present (not just for the `import x = y;` alias form, as a
        // first guess assumed; that guess was wrong and a real sample
        // parse caught it immediately, same discipline as everywhere else
        // in this crate). The alias itself, when present, is `imported`'s
        // own `alias` FIELD, not a sibling of `imported`. Each selective
        // name (`: a, b`) is its own separate `import_bind` sibling of
        // `imported` -- not one `import_bind` holding a list.
        for m in query_matches(tree, bytes, "(import_declaration (imported (module_fqn) @target)) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes);
            let mut cursor = import_node.walk();
            let children: Vec<Node> = import_node.children(&mut cursor).collect();
            let is_static = children.iter().any(|c| c.kind() == "static");
            let is_aliased = children
                .iter()
                .find(|c| c.kind() == "imported")
                .is_some_and(|n| n.child_by_field_name("alias").is_some());
            let binds: Vec<Node> = children.iter().filter(|c| c.kind() == "import_bind").copied().collect();

            if !binds.is_empty() {
                for bind in binds {
                    if let Some(id) = bind.named_child(0) {
                        out.push(ImportEdge { raw_path: raw.clone(), imported_name: Some(text(id, bytes)), is_wildcard: false, module_prefix: Vec::new(), line });
                    }
                }
            } else if is_static || is_aliased {
                out.push(ImportEdge { raw_path: raw.clone(), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
            } else {
                out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
            }
        }
        out
    }

    /// Julia's `using X` / `import X` / `using X: a, b` / `import X: a, b`.
    /// `using X` (no selection) brings all of `X`'s exports into
    /// unqualified scope -- a wildcard. `import X` alone binds only `X`
    /// itself (`X.foo()` required) -- not a wildcard. Either form with an
    /// explicit `: a, b` selection is captured precisely regardless of
    /// `using`/`import`, since both forms bring exactly those names in
    /// unqualified either way.
    pub(super) fn julia_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "[(import_statement) (using_statement)] @import") {
            let Some(&import_node) = m.get("import") else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let is_using = import_node.kind() == "using_statement";
            let mut cursor = import_node.walk();
            let children: Vec<Node> = import_node.named_children(&mut cursor).collect();

            if let Some(sel) = children.iter().find(|c| c.kind() == "selected_import") {
                let mut scursor = sel.walk();
                let mut sel_children = sel.named_children(&mut scursor);
                let Some(module_node) = sel_children.next() else { continue };
                let module_text = text(module_node, bytes);
                for name_node in sel_children {
                    if name_node.kind() == "import_alias" {
                        continue; // a renamed selection -- ambiguous which side is the real target name, skipped rather than guessed
                    }
                    out.push(ImportEdge { raw_path: module_text.clone(), imported_name: Some(text(name_node, bytes)), is_wildcard: false, module_prefix: Vec::new(), line });
                }
            } else if let Some(module_node) = children.iter().find(|c| matches!(c.kind(), "identifier" | "scoped_identifier" | "import_path")) {
                let module_text = text(*module_node, bytes);
                if is_using {
                    out.push(ImportEdge { raw_path: format!("{module_text}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
                } else {
                    out.push(ImportEdge { raw_path: module_text, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
                }
            }
        }
        out
    }

    /// `-include("file.hrl").` -- Erlang's own dedicated preprocessor
    /// node, a real relative path. `-include_lib("kernel/include/x.hrl")`
    /// is deliberately NOT extracted: its path is rooted at an OTP
    /// *library* name, not this repo, so treating it as repo-relative
    /// would risk a coincidental wrong match rather than an honest
    /// "unresolved".
    pub(super) fn erlang_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(pp_include file: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `@import("std")` / `@import("./helper.zig")` -- Zig's builtin
    /// import function, checked by text since (like Nix) it's an
    /// ordinary builtin call, not special grammar.
    pub(super) fn zig_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(builtin_function (builtin_identifier) @_fn (arguments (string) @target)) @import") {
            let (Some(&fn_node), Some(&target), Some(&import_node)) = (m.get("_fn"), m.get("target"), m.get("import")) else { continue };
            if text(fn_node, bytes) != "@import" {
                continue;
            }
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// PHP's `require`/`require_once`/`include`/`include_once` -- only
    /// extracted when the argument is a literal string; a computed path
    /// (a variable, a concatenation) can't be resolved without evaluating
    /// PHP, so it's correctly left unextracted rather than guessed at.
    /// `use Namespace\Class;` (PSR-4 autoloading) is deliberately NOT
    /// handled -- real PSR-4 resolution needs `composer.json`'s autoload
    /// map, which this project doesn't parse; guessing a directory
    /// convention instead risks a confidently wrong signal the way the
    /// original D/Haskell/Julia mistake did.
    pub(super) fn php_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut nodes = Vec::new();
        for kind in ["require_expression", "require_once_expression", "include_expression", "include_once_expression"] {
            find_nodes(tree.root_node(), kind, &mut nodes);
        }
        for node in nodes {
            let line = node.start_position().row as i64 + 1;
            let Some(arg) = node.named_child(0) else { continue };
            if !arg.kind().contains("string") {
                continue;
            }
            let raw_path = text(arg, bytes).trim_matches(|c| c == '"' || c == '\'').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `\input{file}` / `\include{file}` -- only the two most common forms
    /// (LaTeX has a dozen include-like commands for bibliographies,
    /// graphics, SVGs, etc. -- see the module doc's grammar survey; the
    /// rest are lower-value and not attempted this batch).
    pub(super) fn latex_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(latex_include path: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches(|c| c == '{' || c == '}').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Dart's `import 'uri';`. `package:x/y.dart` and `dart:core`-style
    /// URIs are always external (a pub package or the SDK), never repo-
    /// relative, and are correctly skipped rather than guessed at; a bare
    /// relative URI resolves the same way JS's relative imports do (via
    /// `resolve_relative_literal` in `resolve.rs`), including without a
    /// leading `./` -- Dart's own relative-import resolution checks the
    /// importing file's own directory the same way C's quoted `#include`
    /// does.
    pub(super) fn dart_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(import_specification uri: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes).trim_matches(|c| c == '"' || c == '\'').to_string();
            if raw.starts_with("package:") || raw.starts_with("dart:") {
                continue;
            }
            out.push(ImportEdge { raw_path: raw, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Ada's `with Package.Name;` -- can name several packages
    /// comma-separated (`with A, B;`), each captured as its own edge.
    /// `use`-clauses (bringing a `with`-ed package's names into
    /// unqualified scope) aren't tracked separately here; `with` alone is
    /// enough to know which file a call might come from.
    pub(super) fn ada_with(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(with_clause [(identifier) (selected_component)] @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes);
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
        }
        out
    }

    /// OCaml's `open Module` / `open Module.Sub` -- brings the opened
    /// module's names into unqualified scope, always (there's no
    /// "selective open" or "qualified open" form the way Haskell/D have),
    /// so this is always a wildcard.
    pub(super) fn ocaml_open(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(open_module module: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes);
            out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Perl's `use Module::Name;` / `use Module::Name qw(a b);` / `no
    /// Module::Name;` (`use_no_statement` covers both `use` and `no`).
    /// Perl's `Exporter`-based convention means a plain `use Module;`
    /// typically brings a default set of names into unqualified scope --
    /// treated as a wildcard, same reasoning as Haskell's plain import.
    /// An explicit `qw(...)` import list is captured precisely when
    /// present.
    pub(super) fn perl_use(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(use_no_statement package_name: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes);
            let mut cursor = import_node.walk();
            let qw_list = import_node.children(&mut cursor).find(|c| c.kind().starts_with("word_list"));
            match qw_list {
                Some(list) => {
                    let mut lcursor = list.walk();
                    for word in list.named_children(&mut lcursor) {
                        out.push(ImportEdge { raw_path: raw.clone(), imported_name: Some(text(word, bytes)), is_wildcard: false, module_prefix: Vec::new(), line });
                    }
                }
                None => out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line }),
            }
        }
        out
    }

    /// Fortran's `include 'file.inc'` (a real relative path) and `use
    /// module_name` / `use module_name, only: a, b` (module-name-based --
    /// Fortran has no standardized module-to-file naming convention the
    /// way Java's classpath does, so this is a plain filename guess: the
    /// module name itself, tried directly as a filename against a few
    /// real extensions, in `resolve.rs`).
    pub(super) fn fortran_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(include_statement path: (_) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('\'').trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        for m in query_matches(tree, bytes, "(use_statement (module_name) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes);
            let has_only = {
                let mut cursor = import_node.walk();
                import_node.children(&mut cursor).any(|c| c.kind() == "included_items")
            };
            if has_only {
                let included = {
                    let mut cursor = import_node.walk();
                    import_node.children(&mut cursor).find(|c| c.kind() == "included_items")
                };
                if let Some(included) = included {
                    let mut icursor = included.walk();
                    for name in included.named_children(&mut icursor) {
                        out.push(ImportEdge { raw_path: raw.clone(), imported_name: Some(text(name, bytes)), is_wildcard: false, module_prefix: Vec::new(), line });
                    }
                }
            } else {
                out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
            }
        }
        out
    }

    /// Elixir's `alias`/`import`/`require`/`use Foo.Bar`. No file-based
    /// import exists at the language level -- module-to-file mapping is
    /// purely a Mix build-tool *convention* (CamelCase segments become
    /// snake_case directories under `lib/`), not enforced by the compiler,
    /// but real and near-universal in practice (Phoenix and virtually
    /// every published Hex package follow it). Always treated as a
    /// wildcard: `import`/`use` bring the target module's functions into
    /// unqualified scope by design; `alias`/`require` don't, but
    /// distinguishing them isn't worth the extra complexity when all four
    /// share one grammar shape and the file resolved is identical either
    /// way.
    pub(super) fn elixir_alias(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut calls = Vec::new();
        find_nodes_nested(tree.root_node(), "call", &mut calls);
        for call in calls {
            let Some(target_node) = call.child_by_field_name("target") else { continue };
            if target_node.kind() != "identifier" {
                continue;
            }
            if !matches!(text(target_node, bytes).as_str(), "alias" | "import" | "require" | "use") {
                continue;
            }
            let mut mod_nodes = Vec::new();
            find_nodes(call, "alias", &mut mod_nodes);
            let Some(&module_node) = mod_nodes.first() else { continue };
            let raw = text(module_node, bytes);
            let line = call.start_position().row as i64 + 1;
            out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Assembly's `include` directive, across the real spellings this
    /// project's tree-sitter-asm grammar actually parses differently
    /// (confirmed via a real debug dump, not assumed from one sample):
    /// GAS/MASM's `.include "x"` is a dedicated `meta` node (`kind` field
    /// text `.include`) -- extracted precisely. NASM's `%include "x"`
    /// isn't understood by this grammar at all -- the leading `%` becomes
    /// an `ERROR` node -- but the `include "x"` that follows it still
    /// parses as an ordinary `instruction` node with `kind` text
    /// `"include"` and a single `string` operand, the exact same shape a
    /// bare `INCLUDE "x"` (no `%`, some assemblers' own spelling) parses
    /// as too. Matched by `kind` text alone (case-insensitive, for MASM):
    /// "include" is not a real instruction mnemonic in any assembly
    /// dialect, so this can't collide with a genuine instruction the way
    /// a looser heuristic might.
    pub(super) fn asm_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(meta kind: (meta_ident) @kind (string) @target) @import") {
            let (Some(&kind), Some(&target), Some(&import_node)) = (m.get("kind"), m.get("target"), m.get("import")) else { continue };
            if text(kind, bytes) != ".include" {
                continue;
            }
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        for m in query_matches(tree, bytes, "(instruction kind: (word) @kind (string) @target) @import") {
            let (Some(&kind), Some(&target), Some(&import_node)) = (m.get("kind"), m.get("target"), m.get("import")) else { continue };
            if !text(kind, bytes).eq_ignore_ascii_case("include") {
                continue;
            }
            let line = import_node.start_position().row as i64 + 1;
            let raw_path = text(target, bytes).trim_matches('"').to_string();
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Swift's `import Foundation` / `import MyTarget.Something`. Swift
    /// has no per-file module concept the way Java/Python do -- a whole
    /// compiled target/framework is one module -- so this is always
    /// treated as a wildcard the same way a Go package import is: the
    /// resolvable case (a Swift Package Manager multi-target package
    /// importing a sibling target, e.g. `import CoreModule`) names a
    /// whole `Sources/<Target>/` directory, not one file (see
    /// `resolve_swift_module` in `resolve.rs`); an external framework
    /// (`Foundation`, `UIKit`, ...) simply has no matching directory and
    /// stays honestly unresolved, the same pattern every other language's
    /// external-dependency imports already use.
    pub(super) fn swift_import(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        for m in query_matches(tree, bytes, "(import_declaration (identifier) @target) @import") {
            let (Some(&target), Some(&import_node)) = (m.get("target"), m.get("import")) else { continue };
            let line = import_node.start_position().row as i64 + 1;
            let raw = text(target, bytes);
            out.push(ImportEdge { raw_path: format!("{raw}.*"), imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
        }
        out
    }

    /// Terraform/HCL's `module "name" { source = "./path" }` -- the one
    /// real cross-file(-directory) reference this grammar has, and a
    /// genuinely different shape from every other language here: `block`/
    /// `attribute` are fully generic/positional nodes (verified against
    /// real node-types.json -- no dedicated `module_block`/`source`
    /// field exists at all), so this walks them by hand rather than a
    /// single declarative query. Only a literal string `source` (no
    /// interpolation/variable) starting with `./`/`../` is extracted --
    /// a registry reference (`terraform-aws-modules/vpc/aws`) or a git
    /// URL is unambiguously external and correctly left unextracted
    /// rather than guessed at. Always a wildcard: a Terraform module is a
    /// whole directory of `.tf` files, resolved the same multi-file way
    /// Go's package imports are (see `resolve_hcl_module`).
    pub(super) fn hcl_module_source(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut blocks = Vec::new();
        find_nodes(tree.root_node(), "block", &mut blocks);
        for block in blocks {
            let mut cursor = block.walk();
            let Some(block_kind) = block.children(&mut cursor).find(|c| c.kind() == "identifier") else { continue };
            if text(block_kind, bytes) != "module" {
                continue;
            }
            let Some(body) = block.children(&mut block.walk()).find(|c| c.kind() == "body") else { continue };
            let mut attrs = Vec::new();
            find_nodes(body, "attribute", &mut attrs);
            for attr in attrs {
                let Some(attr_name) = attr.children(&mut attr.walk()).find(|c| c.kind() == "identifier") else { continue };
                if text(attr_name, bytes) != "source" {
                    continue;
                }
                let mut strings = Vec::new();
                find_nodes(attr, "string_lit", &mut strings);
                let Some(&string_node) = strings.first() else { continue };
                let raw = text(string_node, bytes).trim_matches('"').to_string();
                if !(raw.starts_with("./") || raw.starts_with("../")) {
                    continue; // a registry/git reference -- unambiguously external
                }
                let line = attr.start_position().row as i64 + 1;
                out.push(ImportEdge { raw_path: raw, imported_name: None, is_wildcard: true, module_prefix: Vec::new(), line });
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

    /// `require("module")` / `require "module"` -- Lua's `package.path`
    /// convention converts dots to directory separators (`require("a.b")`
    /// looks for `a/b.lua`), so the raw path is kept dotted here and
    /// converted in `resolve.rs`, the same split-then-join approach
    /// Python's absolute imports already use.
    pub(super) fn lua_require(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut calls = Vec::new();
        find_nodes(tree.root_node(), "function_call", &mut calls);
        for call in calls {
            let Some(name_node) = call.child_by_field_name("name") else { continue };
            if text(name_node, bytes) != "require" {
                continue;
            }
            let Some(args) = call.child_by_field_name("arguments") else { continue };
            let mut cursor = args.walk();
            let Some(first_str) = args.named_children(&mut cursor).find(|c| c.kind().contains("string")) else { continue };
            let line = call.start_position().row as i64 + 1;
            out.push(ImportEdge { raw_path: strip_quotes(&text(first_str, bytes)), imported_name: None, is_wildcard: false, module_prefix: Vec::new(), line });
        }
        out
    }

    /// `(include "file.scm")` / `(load "file.scm")` -- same shape as
    /// Racket's `require`, different head symbols (Scheme has no
    /// standardized module system across implementations; `include`/
    /// `load` with a literal path are the closest thing to a portable,
    /// resolvable form).
    pub(super) fn scheme_include(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut lists = Vec::new();
        find_nodes(tree.root_node(), "list", &mut lists);
        for list in lists {
            let mut cursor = list.walk();
            let mut children = list.named_children(&mut cursor);
            let Some(head) = children.next() else { continue };
            if !matches!(head.kind(), "symbol" | "identifier") || !matches!(text(head, bytes).as_str(), "include" | "load") {
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

    /// `. .\helper.ps1` (dot-sourcing -- brings the sourced script's
    /// functions into the current scope, a real wildcard) and
    /// `Import-Module ./Helper.psm1` (only the relative-path form; a
    /// bare module *name*, e.g. `Import-Module ActiveDirectory`, names an
    /// installed module, not a repo file, and is correctly left
    /// unextracted).
    pub(super) fn powershell_dotsource(tree: &Tree, bytes: &[u8]) -> Vec<ImportEdge> {
        let mut out = Vec::new();
        let mut cmds = Vec::new();
        find_nodes(tree.root_node(), "command", &mut cmds);
        for cmd in cmds {
            let Some(name_node) = cmd.child_by_field_name("command_name") else { continue };
            let name = text(name_node, bytes);
            let mut cursor = cmd.walk();
            let has_dot_operator = cmd.children(&mut cursor).any(|c| c.kind() == "command_invokation_operator" && text(c, bytes) == ".");
            let is_wildcard = has_dot_operator;
            let is_import_module = name.eq_ignore_ascii_case("Import-Module");
            if !has_dot_operator && !is_import_module {
                continue;
            }
            let line = cmd.start_position().row as i64 + 1;
            let raw_path = if has_dot_operator {
                strip_quotes(&name)
            } else {
                let mut ccursor = cmd.walk();
                let Some(elements) = cmd.children_by_field_name("command_elements", &mut ccursor).next() else { continue };
                let mut ecursor = elements.walk();
                let Some(first) = elements.named_children(&mut ecursor).find(|c| c.kind() != "command_argument_sep") else { continue };
                let p = strip_quotes(&text(first, bytes));
                if !(p.starts_with("./") || p.starts_with("../") || p.starts_with(".\\") || p.starts_with("..\\")) {
                    continue; // a bare module name, not a repo file
                }
                p
            };
            out.push(ImportEdge { raw_path, imported_name: None, is_wildcard, module_prefix: Vec::new(), line });
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
    fn haskell_plain_qualified_selective_and_hiding() {
        let src = "import Data.List\nimport qualified Data.Map as M\nimport Data.Set (member, insert)\nimport Data.Text hiding (map)\n";
        let got = edges(src, Lang::Haskell);
        assert!(got.contains(&("Data.List.*".to_string(), None)), "plain import must be a wildcard: {got:?}");
        assert!(got.contains(&("Data.Map".to_string(), None)) && !got.iter().any(|(p, _)| p == "Data.Map.*"), "qualified import must NOT be a wildcard: {got:?}");
        assert!(got.contains(&("Data.Set".to_string(), Some("member".to_string()))), "{got:?}");
        assert!(got.contains(&("Data.Set".to_string(), Some("insert".to_string()))), "{got:?}");
        assert!(got.contains(&("Data.Text.*".to_string(), None)), "hiding still exposes the rest as a wildcard: {got:?}");
    }

    #[test]
    fn d_plain_selective_static_and_aliased() {
        let src = "import std.stdio;\nimport std.algorithm : map, filter;\nstatic import std.conv;\nimport io = std.stdio;\n";
        let got = edges(src, Lang::D);
        assert!(got.contains(&("std.stdio.*".to_string(), None)), "plain import must be a wildcard: {got:?}");
        assert!(got.contains(&("std.algorithm".to_string(), Some("map".to_string()))), "{got:?}");
        assert!(got.contains(&("std.algorithm".to_string(), Some("filter".to_string()))), "{got:?}");
        assert!(got.contains(&("std.conv".to_string(), None)) && !got.iter().any(|(p, _)| p == "std.conv.*"), "static import must NOT be a wildcard: {got:?}");
    }

    #[test]
    fn julia_using_import_and_selective() {
        let src = "using Base\nimport Base\nusing Base: sin, cos\nimport LinearAlgebra: dot\n";
        let got = edges(src, Lang::Julia);
        assert!(got.contains(&("Base.*".to_string(), None)), "plain `using` must be a wildcard: {got:?}");
        assert!(got.contains(&("Base".to_string(), None)), "plain `import` must NOT be a wildcard: {got:?}");
        assert!(got.contains(&("Base".to_string(), Some("sin".to_string()))), "{got:?}");
        assert!(got.contains(&("Base".to_string(), Some("cos".to_string()))), "{got:?}");
        assert!(got.contains(&("LinearAlgebra".to_string(), Some("dot".to_string()))), "{got:?}");
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

    #[test]
    fn erlang_include_skips_include_lib() {
        let src = "-module(foo).\n-include(\"foo.hrl\").\n-include_lib(\"kernel/include/file.hrl\").\n";
        let got = edges(src, Lang::Erlang);
        assert!(got.contains(&("foo.hrl".to_string(), None)), "{got:?}");
        assert!(!got.iter().any(|(p, _)| p.contains("kernel")), "include_lib must not be treated as repo-relative: {got:?}");
    }

    #[test]
    fn zig_import_builtin() {
        let src = "const std = @import(\"std\");\nconst helper = @import(\"./helper.zig\");\n";
        let got = edges(src, Lang::Zig);
        assert!(got.contains(&("std".to_string(), None)), "{got:?}");
        assert!(got.contains(&("./helper.zig".to_string(), None)), "{got:?}");
    }

    #[test]
    fn php_require_and_include_variants() {
        let src = "<?php\nrequire 'a.php';\nrequire_once 'b.php';\ninclude 'c.php';\ninclude_once 'd.php';\nrequire $dynamic;\n";
        let got = edges(src, Lang::Php);
        assert!(got.contains(&("a.php".to_string(), None)), "{got:?}");
        assert!(got.contains(&("b.php".to_string(), None)), "{got:?}");
        assert!(got.contains(&("c.php".to_string(), None)), "{got:?}");
        assert!(got.contains(&("d.php".to_string(), None)), "{got:?}");
        assert_eq!(got.len(), 4, "a dynamic (non-literal) require must not be extracted: {got:?}");
    }

    #[test]
    fn latex_input_and_include() {
        let src = "\\input{chapter1}\n\\include{chapter2}\n";
        let got = edges(src, Lang::Latex);
        assert!(got.contains(&("chapter1".to_string(), None)), "{got:?}");
        assert!(got.contains(&("chapter2".to_string(), None)), "{got:?}");
    }

    #[test]
    fn dart_import_skips_package_and_dart_schemes() {
        let src = "import 'dart:core';\nimport 'package:foo/foo.dart';\nimport './helper.dart';\n";
        let got = edges(src, Lang::Dart);
        assert!(got.contains(&("./helper.dart".to_string(), None)), "{got:?}");
        assert!(!got.iter().any(|(p, _)| p.contains("dart:") || p.contains("package:")), "{got:?}");
    }

    #[test]
    fn ada_with_clause_multiple_packages() {
        let src = "with Ada.Text_IO;\nwith Ada.Text_IO, Ada.Integer_Text_IO;\n";
        let got = edges(src, Lang::Ada);
        assert!(got.iter().any(|(p, _)| p == "Ada.Text_IO"), "{got:?}");
        assert!(got.iter().any(|(p, _)| p == "Ada.Integer_Text_IO"), "{got:?}");
    }

    #[test]
    fn ocaml_open_module() {
        let got = edges("open List\nopen Core.Std\n", Lang::OCaml);
        assert!(got.contains(&("List.*".to_string(), None)), "{got:?}");
        assert!(got.contains(&("Core.Std.*".to_string(), None)), "{got:?}");
    }

    #[test]
    fn perl_use_plain_and_qw_list() {
        let src = "use Data::Dumper;\nuse List::Util qw(sum max);\n";
        let got = edges(src, Lang::Perl);
        assert!(got.contains(&("Data::Dumper.*".to_string(), None)), "{got:?}");
        assert!(got.contains(&("List::Util".to_string(), Some("sum".to_string()))), "{got:?}");
        assert!(got.contains(&("List::Util".to_string(), Some("max".to_string()))), "{got:?}");
    }

    #[test]
    fn fortran_include_and_use_with_only() {
        let src = "include 'helper.inc'\nuse mymod\nuse othermod, only: foo, bar\n";
        let got = edges(src, Lang::Fortran);
        assert!(got.contains(&("helper.inc".to_string(), None)), "{got:?}");
        assert!(got.contains(&("mymod.*".to_string(), None)), "{got:?}");
        assert!(got.contains(&("othermod".to_string(), Some("foo".to_string()))), "{got:?}");
        assert!(got.contains(&("othermod".to_string(), Some("bar".to_string()))), "{got:?}");
    }

    #[test]
    fn elixir_alias_import_require_use() {
        let src = "defmodule Foo do\n  alias MyApp.Helper\n  import MyApp.Util\n  require Logger\n  use GenServer\nend\n";
        let got = edges(src, Lang::Elixir);
        assert!(got.contains(&("MyApp.Helper.*".to_string(), None)), "{got:?}");
        assert!(got.contains(&("MyApp.Util.*".to_string(), None)), "{got:?}");
        assert!(got.contains(&("Logger.*".to_string(), None)), "{got:?}");
        assert!(got.contains(&("GenServer.*".to_string(), None)), "{got:?}");
    }

    #[test]
    fn lua_require_dotted() {
        let got = edges("local m = require(\"a.b\")\n", Lang::Lua);
        assert!(got.contains(&("a.b".to_string(), None)), "{got:?}");
    }

    #[test]
    fn scheme_include_and_load() {
        let src = "(include \"helper.scm\")\n(load \"other.scm\")\n";
        let got = edges(src, Lang::Scheme);
        assert!(got.contains(&("helper.scm".to_string(), None)), "{got:?}");
        assert!(got.contains(&("other.scm".to_string(), None)), "{got:?}");
    }

    #[test]
    fn powershell_dot_source_and_import_module() {
        let src = ". .\\helper.ps1\nImport-Module ./Other.psm1\nImport-Module ActiveDirectory\n";
        let got = edges(src, Lang::PowerShell);
        assert!(got.iter().any(|(p, _)| p.contains("helper.ps1")), "{got:?}");
        assert!(got.iter().any(|(p, _)| p.contains("Other.psm1")), "{got:?}");
        assert!(!got.iter().any(|(p, _)| p.contains("ActiveDirectory")), "a bare installed-module name must not be extracted: {got:?}");
    }

    #[test]
    fn asm_include_gas_and_nasm_style() {
        let src = ".include \"helper.inc\"\n%include \"other.inc\"\nINCLUDE thirdparty.inc\nmov eax, 1\n";
        let got = edges(src, Lang::Asm);
        assert!(got.contains(&("helper.inc".to_string(), None)), "{got:?}");
        assert!(got.contains(&("other.inc".to_string(), None)), "{got:?}");
        assert!(!got.iter().any(|(p, _)| p.contains("mov")), "a real instruction must never be mistaken for an include: {got:?}");
    }

    #[test]
    fn swift_import_plain_and_dotted() {
        let src = "import Foundation\nimport CoreModule.Helper\n";
        let got = edges(src, Lang::Swift);
        assert!(got.contains(&("Foundation.*".to_string(), None)), "{got:?}");
        assert!(got.contains(&("CoreModule.Helper.*".to_string(), None)), "{got:?}");
    }

    #[test]
    fn hcl_module_source_local_only() {
        let src = "module \"vpc\" {\n  source = \"./modules/vpc\"\n}\nmodule \"remote\" {\n  source = \"terraform-aws-modules/vpc/aws\"\n}\n";
        let got = edges(src, Lang::Hcl);
        assert!(got.contains(&("./modules/vpc".to_string(), None)), "{got:?}");
        assert!(!got.iter().any(|(p, _)| p.contains("terraform-aws-modules")), "a registry reference must stay unextracted: {got:?}");
    }
}

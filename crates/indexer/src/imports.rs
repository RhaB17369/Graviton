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

use tree_sitter::{Node, Parser, Tree};

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
}

//! Language detection and best-effort tree-sitter definition queries.
//!
//! Query coverage is intentionally best-effort: full-text search (via the
//! generic line-chunker in `lib.rs`) never depends on it, so an imperfect or
//! outdated query for a given grammar version degrades symbol lookup only,
//! never search recall.

use tree_sitter::{Language, Query};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    C,
    Cpp,
    Go,
    Other,
}

impl Lang {
    pub fn from_path(path: &std::path::Path) -> Lang {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "rs" => Lang::Rust,
            "py" | "pyw" => Lang::Python,
            "js" | "jsx" | "mjs" | "cjs" => Lang::JavaScript,
            "ts" | "mts" | "cts" => Lang::TypeScript,
            "tsx" => Lang::Tsx,
            "c" | "h" => Lang::C,
            "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => Lang::Cpp,
            "go" => Lang::Go,
            _ => Lang::Other,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Lang::Rust => "rust",
            Lang::Python => "python",
            Lang::JavaScript => "javascript",
            Lang::TypeScript => "typescript",
            Lang::Tsx => "tsx",
            Lang::C => "c",
            Lang::Cpp => "cpp",
            Lang::Go => "go",
            Lang::Other => "text",
        }
    }

    pub fn ts_language(&self) -> Option<Language> {
        Some(match self {
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Lang::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Lang::C => tree_sitter_c::LANGUAGE.into(),
            Lang::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Lang::Go => tree_sitter_go::LANGUAGE.into(),
            Lang::Other => return None,
        })
    }

    /// Definition query: every match must expose a `@def` capture (the whole
    /// definition node, for line-range extraction) and a `@name` capture.
    pub fn def_query_src(&self) -> Option<&'static str> {
        Some(match self {
            Lang::Rust => {
                r#"
                (function_item name: (identifier) @name) @def
                (struct_item name: (type_identifier) @name) @def
                (enum_item name: (type_identifier) @name) @def
                (trait_item name: (type_identifier) @name) @def
                (impl_item type: (type_identifier) @name) @def
                (mod_item name: (identifier) @name) @def
                "#
            }
            Lang::Python => {
                r#"
                (function_definition name: (identifier) @name) @def
                (class_definition name: (identifier) @name) @def
                "#
            }
            Lang::JavaScript => {
                r#"
                (function_declaration name: (identifier) @name) @def
                (class_declaration name: (identifier) @name) @def
                (method_definition name: (property_identifier) @name) @def
                "#
            }
            Lang::TypeScript | Lang::Tsx => {
                r#"
                (function_declaration name: (identifier) @name) @def
                (class_declaration name: (identifier) @name) @def
                (method_definition name: (property_identifier) @name) @def
                (interface_declaration name: (type_identifier) @name) @def
                "#
            }
            Lang::C => {
                r#"
                (function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
                (struct_specifier name: (type_identifier) @name) @def
                "#
            }
            Lang::Cpp => {
                r#"
                (function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
                (struct_specifier name: (type_identifier) @name) @def
                (class_specifier name: (type_identifier) @name) @def
                "#
            }
            Lang::Go => {
                r#"
                (function_declaration name: (identifier) @name) @def
                (method_declaration name: (field_identifier) @name) @def
                (type_spec name: (type_identifier) @name) @def
                "#
            }
            Lang::Other => return None,
        })
    }

    pub fn compile_def_query(&self) -> Option<Query> {
        let lang = self.ts_language()?;
        let src = self.def_query_src()?;
        match Query::new(&lang, src) {
            Ok(q) => Some(q),
            Err(e) => {
                tracing::warn!(lang = self.name(), error = %e, "definition query failed to compile, symbol extraction disabled for this language");
                None
            }
        }
    }
}

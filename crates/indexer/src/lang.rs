//! Language detection and best-effort tree-sitter definition queries.
//!
//! Query coverage is intentionally best-effort: full-text search (via the
//! generic line-chunker in `lib.rs`) never depends on it, so an imperfect or
//! outdated query for a given grammar version degrades symbol lookup only,
//! never search recall.
//!
//! Languages split into two tiers:
//! - **Parsed**: a tree-sitter grammar + definition query extracts real
//!   symbols (functions/classes/...) into the `symbols` table.
//! - **Tagged**: no grammar wired up (usually because there's no meaningful
//!   "symbol" concept — markup/config/data formats), but the file is still
//!   labeled with its real language instead of falling into the generic
//!   `Other`/"text" bucket, and is always fully searchable via the
//!   line-chunker regardless.

use std::path::Path;
use tree_sitter::{Language, Query};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    // --- parsed: tree-sitter symbol extraction ---
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    C,
    Cpp,
    Go,
    Java,
    CSharp,
    Php,
    Ruby,
    Bash,
    Lua,
    Solidity,
    PowerShell,
    // --- tagged only: labeled + always full-text searchable ---
    // (Kotlin lives here, not in the parsed tier: the only maintained
    // tree-sitter-kotlin release compatible with crates.io is pinned to
    // tree-sitter 0.21/0.22, which conflicts with the 0.26 native lib every
    // other grammar here links against — cargo refuses two `links =
    // "tree-sitter"` versions in one binary. Revisit if a grammar update
    // rebases onto a current tree-sitter core.)
    Kotlin,
    Html,
    Css,
    Json,
    Yaml,
    Toml,
    Xml,
    Markdown,
    Sql,
    Dockerfile,
    Ini,
    Makefile,
    Other,
}

impl Lang {
    pub fn from_path(path: &Path) -> Lang {
        // Filename-based detection first, for extensionless conventions.
        if let Some(fname) = path.file_name().and_then(|f| f.to_str()) {
            match fname {
                "Dockerfile" | "Containerfile" => return Lang::Dockerfile,
                "Makefile" | "makefile" | "GNUmakefile" => return Lang::Makefile,
                _ if fname.starts_with("Dockerfile.") => return Lang::Dockerfile,
                _ => {}
            }
        }
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
            "java" => Lang::Java,
            "cs" => Lang::CSharp,
            "php" | "php3" | "php4" | "php5" | "phtml" => Lang::Php,
            "rb" | "rake" | "gemspec" => Lang::Ruby,
            "sh" | "bash" | "zsh" => Lang::Bash,
            "lua" => Lang::Lua,
            "sol" => Lang::Solidity,
            "kt" | "kts" => Lang::Kotlin,
            "ps1" | "psm1" | "psd1" => Lang::PowerShell,
            "html" | "htm" => Lang::Html,
            "css" | "scss" | "sass" | "less" => Lang::Css,
            "json" | "jsonc" => Lang::Json,
            "yml" | "yaml" => Lang::Yaml,
            "toml" => Lang::Toml,
            "xml" | "xsd" | "xsl" => Lang::Xml,
            "md" | "markdown" => Lang::Markdown,
            "sql" => Lang::Sql,
            "ini" | "cfg" | "conf" => Lang::Ini,
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
            Lang::Java => "java",
            Lang::CSharp => "csharp",
            Lang::Php => "php",
            Lang::Ruby => "ruby",
            Lang::Bash => "bash",
            Lang::Lua => "lua",
            Lang::Solidity => "solidity",
            Lang::PowerShell => "powershell",
            Lang::Kotlin => "kotlin",
            Lang::Html => "html",
            Lang::Css => "css",
            Lang::Json => "json",
            Lang::Yaml => "yaml",
            Lang::Toml => "toml",
            Lang::Xml => "xml",
            Lang::Markdown => "markdown",
            Lang::Sql => "sql",
            Lang::Dockerfile => "dockerfile",
            Lang::Ini => "ini",
            Lang::Makefile => "makefile",
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
            Lang::Java => tree_sitter_java::LANGUAGE.into(),
            Lang::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            Lang::Php => tree_sitter_php::LANGUAGE_PHP.into(),
            Lang::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Lang::Bash => tree_sitter_bash::LANGUAGE.into(),
            Lang::Lua => tree_sitter_lua::LANGUAGE.into(),
            Lang::Solidity => tree_sitter_solidity::LANGUAGE.into(),
            Lang::PowerShell => tree_sitter_powershell::LANGUAGE.into(),
            _ => return None,
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
            Lang::Java => {
                r#"
                (class_declaration name: (identifier) @name) @def
                (interface_declaration name: (identifier) @name) @def
                (method_declaration name: (identifier) @name) @def
                (constructor_declaration name: (identifier) @name) @def
                (enum_declaration name: (identifier) @name) @def
                (record_declaration name: (identifier) @name) @def
                "#
            }
            Lang::CSharp => {
                r#"
                (class_declaration name: (identifier) @name) @def
                (interface_declaration name: (identifier) @name) @def
                (method_declaration name: (identifier) @name) @def
                (constructor_declaration name: (identifier) @name) @def
                (enum_declaration name: (identifier) @name) @def
                (struct_declaration name: (identifier) @name) @def
                (record_declaration name: (identifier) @name) @def
                "#
            }
            Lang::Php => {
                r#"
                (class_declaration name: (name) @name) @def
                (interface_declaration name: (name) @name) @def
                (trait_declaration name: (name) @name) @def
                (function_definition name: (name) @name) @def
                (method_declaration name: (name) @name) @def
                (enum_declaration name: (name) @name) @def
                "#
            }
            Lang::Ruby => {
                r#"
                (class name: (constant) @name) @def
                (module name: (constant) @name) @def
                (method name: (identifier) @name) @def
                (singleton_method name: (identifier) @name) @def
                "#
            }
            Lang::Bash => {
                r#"
                (function_definition name: (word) @name) @def
                "#
            }
            Lang::Lua => {
                r#"
                (function_declaration name: [(identifier) (dot_index_expression) (method_index_expression)] @name) @def
                "#
            }
            Lang::Solidity => {
                r#"
                (contract_declaration name: (identifier) @name) @def
                (interface_declaration name: (identifier) @name) @def
                (library_declaration name: (identifier) @name) @def
                (function_definition name: (identifier) @name) @def
                (struct_declaration name: (identifier) @name) @def
                (enum_declaration name: (identifier) @name) @def
                (modifier_definition name: (identifier) @name) @def
                "#
            }
            Lang::PowerShell => {
                r#"
                (function_statement (function_name) @name) @def
                (class_statement (simple_name) @name) @def
                (class_method_definition (simple_name) @name) @def
                "#
            }
            _ => return None,
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

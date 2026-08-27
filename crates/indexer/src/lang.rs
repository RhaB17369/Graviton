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
    // The following all link a real tree-sitter grammar (verified: this
    // whole batch compiles and links together with everything above, no
    // `links = "tree-sitter"` conflicts) but only some have a *verified*
    // `def_query_src` (checked against a real sample file, same bar as
    // every language above) -- the rest return `None` there for now, which
    // degrades to "recognized + fully searchable, no `grv symbol`" exactly
    // like the tagged tier below, but with the hard part (a linkable
    // grammar) already done. `grv callers`/`callees` needs its own
    // `call_query_src` regardless of `def_query_src` -- currently written
    // for none of these (see ARCHITECTURE.md's call graph section for the
    // full list of what has one).
    Haskell,   // def query verified
    Fish,      // def query verified
    Dart,      // def query verified
    Zig,       // def query verified
    Julia,     // def query verified
    Groovy,    // def query verified
    GraphQL,   // def query verified
    Crystal,   // def query verified
    D,         // def query verified
    Asm,       // def query verified (label-based)
    Elixir,
    Scala,
    Swift,
    Perl,
    R,
    OCaml,
    Elm,
    Nim,
    Erlang,
    Vim,
    Latex,
    Nix,
    Hcl,
    CMake,
    Verilog,
    Vhdl,
    Fortran,
    Prolog,
    Racket,
    Scheme,
    Proto,
    Svelte,
    Vue,
    ObjC,
    Glsl,
    Hlsl,
    Wgsl,
    Ada,
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
                "CMakeLists.txt" => return Lang::CMake,
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
            "hs" | "lhs" => Lang::Haskell,
            "fish" => Lang::Fish,
            "dart" => Lang::Dart,
            "zig" => Lang::Zig,
            "jl" => Lang::Julia,
            "groovy" | "gvy" | "gradle" => Lang::Groovy,
            "graphql" | "gql" => Lang::GraphQL,
            "cr" => Lang::Crystal,
            "d" | "di" => Lang::D,
            // .s/.asm are the common assembly extensions; the conventional
            // capital-S variant (preprocessed-with-cpp assembly) lands here
            // too since this whole match already lowercases first.
            "asm" | "s" => Lang::Asm,
            "ex" | "exs" => Lang::Elixir,
            "scala" | "sc" => Lang::Scala,
            "swift" => Lang::Swift,
            // .pl is genuinely ambiguous between Perl and Prolog; Perl gets
            // it as the far more common modern usage. Prolog files mostly
            // show up as .pro/.p in practice.
            "pl" | "pm" | "t" => Lang::Perl,
            "r" => Lang::R,
            "ml" | "mli" => Lang::OCaml,
            "elm" => Lang::Elm,
            "nim" | "nims" => Lang::Nim,
            "erl" | "hrl" => Lang::Erlang,
            "vim" => Lang::Vim,
            "tex" | "latex" | "sty" | "cls" => Lang::Latex,
            "nix" => Lang::Nix,
            "hcl" | "tf" | "tfvars" => Lang::Hcl,
            "cmake" => Lang::CMake,
            "v" | "vh" => Lang::Verilog,
            "vhd" | "vhdl" => Lang::Vhdl,
            "f90" | "f95" | "f03" | "f08" | "f" | "for" => Lang::Fortran,
            "pro" => Lang::Prolog,
            "rkt" => Lang::Racket,
            "scm" | "ss" => Lang::Scheme,
            "proto" => Lang::Proto,
            "svelte" => Lang::Svelte,
            "vue" => Lang::Vue,
            "m" | "mm" => Lang::ObjC,
            "glsl" | "vert" | "frag" | "geom" | "comp" | "tesc" | "tese" => Lang::Glsl,
            "hlsl" | "fx" => Lang::Hlsl,
            "wgsl" => Lang::Wgsl,
            "adb" | "ads" => Lang::Ada,
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
            Lang::Haskell => "haskell",
            Lang::Fish => "fish",
            Lang::Dart => "dart",
            Lang::Zig => "zig",
            Lang::Julia => "julia",
            Lang::Groovy => "groovy",
            Lang::GraphQL => "graphql",
            Lang::Crystal => "crystal",
            Lang::D => "d",
            Lang::Asm => "asm",
            Lang::Elixir => "elixir",
            Lang::Scala => "scala",
            Lang::Swift => "swift",
            Lang::Perl => "perl",
            Lang::R => "r",
            Lang::OCaml => "ocaml",
            Lang::Elm => "elm",
            Lang::Nim => "nim",
            Lang::Erlang => "erlang",
            Lang::Vim => "vim",
            Lang::Latex => "latex",
            Lang::Nix => "nix",
            Lang::Hcl => "hcl",
            Lang::CMake => "cmake",
            Lang::Verilog => "verilog",
            Lang::Vhdl => "vhdl",
            Lang::Fortran => "fortran",
            Lang::Prolog => "prolog",
            Lang::Racket => "racket",
            Lang::Scheme => "scheme",
            Lang::Proto => "proto",
            Lang::Svelte => "svelte",
            Lang::Vue => "vue",
            Lang::ObjC => "objc",
            Lang::Glsl => "glsl",
            Lang::Hlsl => "hlsl",
            Lang::Wgsl => "wgsl",
            Lang::Ada => "ada",
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
            Lang::Haskell => tree_sitter_haskell::LANGUAGE.into(),
            Lang::Fish => tree_sitter_fish::language(),
            Lang::Dart => tree_sitter_dart::LANGUAGE.into(),
            Lang::Zig => tree_sitter_zig::LANGUAGE.into(),
            Lang::Julia => tree_sitter_julia::LANGUAGE.into(),
            Lang::Groovy => tree_sitter_groovy::LANGUAGE.into(),
            Lang::GraphQL => tree_sitter_graphql::LANGUAGE.into(),
            Lang::Crystal => tree_sitter_crystal::LANGUAGE.into(),
            Lang::D => tree_sitter_d::LANGUAGE.into(),
            Lang::Asm => tree_sitter_asm::LANGUAGE.into(),
            Lang::Elixir => tree_sitter_elixir::LANGUAGE.into(),
            Lang::Scala => tree_sitter_scala::LANGUAGE.into(),
            Lang::Swift => tree_sitter_swift::LANGUAGE.into(),
            Lang::Perl => tree_sitter_perl::LANGUAGE.into(),
            Lang::R => tree_sitter_r::LANGUAGE.into(),
            Lang::OCaml => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
            Lang::Elm => tree_sitter_elm::LANGUAGE.into(),
            Lang::Nim => tree_sitter_nim::LANGUAGE.into(),
            Lang::Erlang => tree_sitter_erlang::LANGUAGE.into(),
            Lang::Vim => tree_sitter_vim::language(),
            Lang::Nix => tree_sitter_nix::LANGUAGE.into(),
            Lang::Hcl => tree_sitter_hcl::LANGUAGE.into(),
            Lang::CMake => tree_sitter_cmake::LANGUAGE.into(),
            Lang::Verilog => tree_sitter_verilog::LANGUAGE.into(),
            Lang::Vhdl => tree_sitter_vhdl::LANGUAGE.into(),
            Lang::Fortran => tree_sitter_fortran::LANGUAGE.into(),
            Lang::Prolog => tree_sitter_prolog::LANGUAGE.into(),
            Lang::Racket => tree_sitter_racket::LANGUAGE.into(),
            Lang::Scheme => tree_sitter_scheme::LANGUAGE.into(),
            Lang::Proto => tree_sitter_proto::LANGUAGE.into(),
            Lang::ObjC => tree_sitter_objc::LANGUAGE.into(),
            Lang::Glsl => tree_sitter_glsl::LANGUAGE_GLSL.into(),
            Lang::Hlsl => tree_sitter_hlsl::LANGUAGE_HLSL.into(),
            Lang::Ada => tree_sitter_ada::LANGUAGE.into(),
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
            // Verified against a real parse (see `dump_tree_scratch` below,
            // or just re-run it) for each of the ten languages that follow
            // — same bar as every language above, not a guess left
            // unchecked because the grammar happened to link.
            Lang::Haskell => {
                r#"
                (function name: (variable) @name) @def
                (data_type name: (name) @name) @def
                (class name: (name) @name) @def
                "#
            }
            Lang::Fish => {
                r#"
                (function_definition name: (word) @name) @def
                "#
            }
            Lang::Dart => {
                r#"
                (function_declaration signature: (function_signature name: (identifier) @name)) @def
                (class_declaration name: (identifier) @name) @def
                (method_declaration signature: (method_signature (function_signature name: (identifier) @name))) @def
                "#
            }
            Lang::Zig => {
                r#"
                (function_declaration name: (identifier) @name) @def
                (variable_declaration (identifier) @name (struct_declaration)) @def
                "#
            }
            Lang::Julia => {
                r#"
                (function_definition (signature (call_expression (identifier) @name))) @def
                (macro_definition (signature (call_expression (identifier) @name))) @def
                (struct_definition (type_head (identifier) @name)) @def
                "#
            }
            Lang::Groovy => {
                r#"
                (function_definition name: (identifier) @name) @def
                (class_declaration name: (identifier) @name) @def
                (method_declaration name: (identifier) @name) @def
                "#
            }
            Lang::GraphQL => {
                r#"
                (object_type_definition (name) @name) @def
                (interface_type_definition (name) @name) @def
                (field_definition (name) @name) @def
                (input_object_type_definition (name) @name) @def
                "#
            }
            Lang::Crystal => {
                r#"
                (method_definition name: (identifier) @name) @def
                (class_declaration name: (identifier) @name) @def
                "#
            }
            Lang::D => {
                r#"
                (function_declaration (identifier) @name) @def
                (class_declaration (identifier) @name) @def
                (struct_declaration (identifier) @name) @def
                "#
            }
            // Assembly's closest analog to a "symbol": a `label:` marking a
            // jump/call target or data location.
            Lang::Asm => {
                r#"
                (label (ident) @name) @def
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

    /// Call-site query: every match must expose a `@call` capture (the
    /// whole call expression, for its line number) and a `@callee`
    /// capture (just the called name's text). This resolves *names*, not
    /// symbols — no type inference, so `grv callers foo` matches every
    /// call site textually named `foo` regardless of which `foo` it is at
    /// a given scope. That's the same honest simplification `grv symbol`
    /// already makes (LIKE-based name matching, not full resolution), not
    /// a new one.
    ///
    /// Covers the languages exercised/verified so far (Rust, Python,
    /// JS/TS/TSX, Go) — not yet every parsed language; an uncovered
    /// language just yields no call edges, same graceful-degradation
    /// contract as `def_query_src` for a query that fails to compile.
    pub fn call_query_src(&self) -> Option<&'static str> {
        Some(match self {
            Lang::Rust => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (call_expression function: (field_expression field: (field_identifier) @callee)) @call
                (call_expression function: (scoped_identifier name: (identifier) @callee)) @call
                (macro_invocation macro: (identifier) @callee) @call
                "#
            }
            Lang::Python => {
                r#"
                (call function: (identifier) @callee) @call
                (call function: (attribute attribute: (identifier) @callee)) @call
                "#
            }
            Lang::JavaScript => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (call_expression function: (member_expression property: (property_identifier) @callee)) @call
                "#
            }
            Lang::TypeScript | Lang::Tsx => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (call_expression function: (member_expression property: (property_identifier) @callee)) @call
                "#
            }
            Lang::Go => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (call_expression function: (selector_expression field: (field_identifier) @callee)) @call
                "#
            }
            _ => return None,
        })
    }

    pub fn compile_call_query(&self) -> Option<Query> {
        let lang = self.ts_language()?;
        let src = self.call_query_src()?;
        match Query::new(&lang, src) {
            Ok(q) => Some(q),
            Err(e) => {
                tracing::warn!(lang = self.name(), error = %e, "call query failed to compile, call-graph extraction disabled for this language");
                None
            }
        }
    }
}

#[cfg(test)]
mod new_language_queries {
    // Each `def_query_src` addition verified against a real parse of real
    // sample code, not just "it compiles" -- same bar every existing
    // language here was held to (see the module doc comment).
    use super::*;

    fn names(lang: Lang, src: &str) -> Vec<String> {
        crate::extract_symbols(src, lang).into_iter().map(|s| s.name).collect()
    }

    #[test]
    fn haskell_function_data_class() {
        let src = "module M where\n\nfoo :: Int -> Int\nfoo x = x + 1\n\ndata Color = Red | Blue\n\nclass Show2 a where\n  show2 :: a -> String\n";
        let found = names(Lang::Haskell, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Color".to_string()), "{found:?}");
        assert!(found.contains(&"Show2".to_string()), "{found:?}");
    }

    #[test]
    fn fish_function() {
        let found = names(Lang::Fish, "function foo\n    echo hi\nend\n");
        assert_eq!(found, vec!["foo"]);
    }

    #[test]
    fn dart_function_class_method() {
        let src = "int foo(int x) {\n  return x + 1;\n}\n\nclass Bar {\n  void baz() {}\n}\n";
        let found = names(Lang::Dart, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Bar".to_string()), "{found:?}");
        assert!(found.contains(&"baz".to_string()), "{found:?}");
    }

    #[test]
    fn zig_function_and_struct_const() {
        let src = "fn foo(x: i32) i32 {\n    return x + 1;\n}\n\nconst Bar = struct {\n    dummy: i32,\n};\n";
        let found = names(Lang::Zig, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Bar".to_string()), "{found:?}");
    }

    #[test]
    fn julia_function_struct_macro() {
        let src = "function foo(x)\n    return x + 1\nend\n\nstruct Point\n    x\n    y\nend\n\nmacro mymacro(x)\n    x\nend\n";
        let found = names(Lang::Julia, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Point".to_string()), "{found:?}");
        assert!(found.contains(&"mymacro".to_string()), "{found:?}");
    }

    #[test]
    fn groovy_function_class_method() {
        let src = "int foo(int x) {\n    return x + 1\n}\n\nclass Bar {\n    def baz() {}\n}\n";
        let found = names(Lang::Groovy, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Bar".to_string()), "{found:?}");
        assert!(found.contains(&"baz".to_string()), "{found:?}");
    }

    #[test]
    fn graphql_object_and_field() {
        let found = names(Lang::GraphQL, "type Query {\n  hello: String\n}\n");
        assert!(found.contains(&"Query".to_string()), "{found:?}");
        assert!(found.contains(&"hello".to_string()), "{found:?}");
    }

    #[test]
    fn crystal_method_and_class() {
        let src = "def foo(x)\n  x + 1\nend\n\nclass Bar\n  def baz\n  end\nend\n";
        let found = names(Lang::Crystal, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Bar".to_string()), "{found:?}");
        assert!(found.contains(&"baz".to_string()), "{found:?}");
    }

    #[test]
    fn d_function_class_struct() {
        let src = "int foo(int x) {\n    return x + 1;\n}\n\nclass Bar {\n}\n\nstruct Baz {\n}\n";
        let found = names(Lang::D, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Bar".to_string()), "{found:?}");
        assert!(found.contains(&"Baz".to_string()), "{found:?}");
    }

    #[test]
    fn asm_label() {
        let found = names(Lang::Asm, "foo:\n    mov eax, 1\n    ret\n");
        assert_eq!(found, vec!["foo"]);
    }
}

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

/// Every `Lang` variant, once, in one place — used by the query-safety-net
/// tests below (and anything else that wants "every language GRAVITON
/// knows about" without re-deriving the list). Deliberately hand-maintained
/// rather than a derive macro (no `strum` dependency for one array); a
/// `match` with every arm spelled out below would fail to compile if a
/// variant were ever added here and forgotten there, so this list itself
/// can't silently drift — see `all_langs_is_exhaustive` below.
pub const ALL_LANGS: &[Lang] = &[
    Lang::Rust,
    Lang::Python,
    Lang::JavaScript,
    Lang::TypeScript,
    Lang::Tsx,
    Lang::C,
    Lang::Cpp,
    Lang::Go,
    Lang::Java,
    Lang::CSharp,
    Lang::Php,
    Lang::Ruby,
    Lang::Bash,
    Lang::Lua,
    Lang::Solidity,
    Lang::PowerShell,
    Lang::Haskell,
    Lang::Fish,
    Lang::Dart,
    Lang::Zig,
    Lang::Julia,
    Lang::Groovy,
    Lang::GraphQL,
    Lang::Crystal,
    Lang::D,
    Lang::Asm,
    Lang::Elixir,
    Lang::Scala,
    Lang::Swift,
    Lang::Perl,
    Lang::R,
    Lang::OCaml,
    Lang::Elm,
    Lang::Nim,
    Lang::Erlang,
    Lang::Vim,
    Lang::Latex,
    Lang::Nix,
    Lang::Hcl,
    Lang::CMake,
    Lang::Verilog,
    Lang::Vhdl,
    Lang::Fortran,
    Lang::Prolog,
    Lang::Racket,
    Lang::Scheme,
    Lang::Proto,
    Lang::Svelte,
    Lang::Vue,
    Lang::ObjC,
    Lang::Glsl,
    Lang::Hlsl,
    Lang::Wgsl,
    Lang::Ada,
    Lang::Kotlin,
    Lang::Html,
    Lang::Css,
    Lang::Json,
    Lang::Yaml,
    Lang::Toml,
    Lang::Xml,
    Lang::Markdown,
    Lang::Sql,
    Lang::Dockerfile,
    Lang::Ini,
    Lang::Makefile,
    Lang::Other,
];

/// Compile-time guard, not a runtime check: if a new `Lang` variant is ever
/// added to the enum without also being added to `ALL_LANGS` above, the
/// `match` in `all_langs_covers_every_variant` (below, in `#[cfg(test)]`)
/// becomes non-exhaustive and the crate fails to *build* -- catching a
/// forgotten variant immediately, at the next `cargo build`, rather than
/// leaving `ALL_LANGS` (and everything that trusts it, like the query
/// safety net below) silently incomplete.
#[cfg(test)]
fn _all_langs_exhaustive_match_guard(l: Lang) {
    match l {
        Lang::Rust
        | Lang::Python
        | Lang::JavaScript
        | Lang::TypeScript
        | Lang::Tsx
        | Lang::C
        | Lang::Cpp
        | Lang::Go
        | Lang::Java
        | Lang::CSharp
        | Lang::Php
        | Lang::Ruby
        | Lang::Bash
        | Lang::Lua
        | Lang::Solidity
        | Lang::PowerShell
        | Lang::Haskell
        | Lang::Fish
        | Lang::Dart
        | Lang::Zig
        | Lang::Julia
        | Lang::Groovy
        | Lang::GraphQL
        | Lang::Crystal
        | Lang::D
        | Lang::Asm
        | Lang::Elixir
        | Lang::Scala
        | Lang::Swift
        | Lang::Perl
        | Lang::R
        | Lang::OCaml
        | Lang::Elm
        | Lang::Nim
        | Lang::Erlang
        | Lang::Vim
        | Lang::Latex
        | Lang::Nix
        | Lang::Hcl
        | Lang::CMake
        | Lang::Verilog
        | Lang::Vhdl
        | Lang::Fortran
        | Lang::Prolog
        | Lang::Racket
        | Lang::Scheme
        | Lang::Proto
        | Lang::Svelte
        | Lang::Vue
        | Lang::ObjC
        | Lang::Glsl
        | Lang::Hlsl
        | Lang::Wgsl
        | Lang::Ada
        | Lang::Kotlin
        | Lang::Html
        | Lang::Css
        | Lang::Json
        | Lang::Yaml
        | Lang::Toml
        | Lang::Xml
        | Lang::Markdown
        | Lang::Sql
        | Lang::Dockerfile
        | Lang::Ini
        | Lang::Makefile
        | Lang::Other => {}
    }
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
            Lang::Nim => tree_sitter_nim::language(),
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
            Lang::Latex => codebook_tree_sitter_latex::LANGUAGE.into(),
            Lang::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
            Lang::Svelte => tree_sitter_svelte_ng::LANGUAGE.into(),
            Lang::Vue => tree_sitter_vuejs::LANGUAGE.into(),
            Lang::Wgsl => tree_sitter_wgsl_bevy::LANGUAGE.into(),
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
            // `class_declaration`'s `name:` field is a `type_identifier`
            // here, unlike plain JavaScript's `identifier` -- a real,
            // previously-undetected mismatch this query shipped with since
            // this project's very first version (TypeScript/TSX had no
            // dedicated test, unlike every language added since; caught by
            // `query_predicate_safety_net`'s pattern-count check, which
            // found the whole query failing to *compile* -- not just
            // matching the wrong thing -- since `Query::new` rejects a
            // field/type mismatch as an "impossible pattern" at compile
            // time. That meant every TypeScript/TSX symbol -- functions
            // and interfaces included, not just classes -- silently
            // extracted nothing, because one bad pattern fails the whole
            // multi-pattern query string.
            Lang::TypeScript | Lang::Tsx => {
                r#"
                (function_declaration name: (identifier) @name) @def
                (class_declaration name: (type_identifier) @name) @def
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
            // `method_def`/`class_def`, not `method_definition`/
            // `class_declaration` -- this project's vendored fork (see
            // vendor/tree-sitter-crystal/NOTICE.md) uses different node
            // names than the stale crates.io grammar this query was
            // originally written against; verified via a real parse-tree
            // dump. A class name is its own `constant` node kind (Crystal
            // lexes capitalized names distinctly from `identifier`).
            Lang::Crystal => {
                r#"
                (method_def name: (identifier) @name) @def
                (class_def name: (constant) @name) @def
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
            // Elixir's macro system means `defmodule`/`def`/`defp`/etc. are
            // all plain `call` nodes at the grammar level -- there's no
            // dedicated node for "this is a definition" the way `def
            // foo` is a `function_item` in Rust. Same is true of `if`,
            // `case`, `receive`, and every other control-flow macro, all
            // of which are *also* `call`s with a `do_block` -- so a query
            // that just matched "any call with a do_block" would wrongly
            // treat every `if ... do ... end` as a symbol definition. The
            // `#eq?`/`#any-of?` predicates below (checked in Rust via
            // `QueryMatch::satisfies_text_predicates` -- see
            // `extract_symbols` in lib.rs) are what make this reliable
            // instead of a guess: only a call whose target identifier's
            // *text* is literally one of the def-like keywords counts.
            Lang::Elixir => {
                r#"
                ((call target: (identifier) @kw (arguments (alias) @name) (do_block)) @def
                 (#eq? @kw "defmodule"))

                ((call target: (identifier) @kw (arguments (call target: (identifier) @name (arguments _))) (do_block)) @def
                 (#any-of? @kw "def" "defp" "defmacro" "defmacrop"))

                ((call target: (identifier) @kw (arguments (call target: (identifier) @name (arguments _)) (keywords))) @def
                 (#any-of? @kw "def" "defp" "defmacro" "defmacrop"))

                ((call target: (identifier) @kw (arguments (identifier) @name) (do_block)) @def
                 (#any-of? @kw "def" "defp" "defmacro" "defmacrop"))
                "#
            }
            Lang::Scala => {
                r#"
                (function_definition name: (identifier) @name) @def
                (function_declaration name: (identifier) @name) @def
                (class_definition name: (identifier) @name) @def
                (object_definition name: (identifier) @name) @def
                (trait_definition name: (identifier) @name) @def
                "#
            }
            // Swift's grammar has one `class_declaration` node for
            // `class`/`struct`/`protocol`/`actor` alike (the keyword
            // itself is an anonymous token, invisible to a query) -- so a
            // struct shows up here with `kind = "class_declaration"` too.
            // Harmless for a symbol *list*; just not a place to expect
            // struct-vs-class precision.
            Lang::Swift => {
                r#"
                (function_declaration name: (simple_identifier) @name) @def
                (class_declaration name: (type_identifier) @name) @def
                "#
            }
            Lang::Perl => {
                r#"
                (function_definition name: (identifier) @name) @def
                (package_statement (package_name (identifier) @name)) @def
                "#
            }
            // R has no dedicated "function definition" node -- `foo <-
            // function(x) ...` is just an assignment whose right-hand side
            // happens to be a `function_definition`. The name is the
            // assignment target, one level up.
            Lang::R => {
                r#"
                (binary_operator lhs: (identifier) @name rhs: (function_definition)) @def
                "#
            }
            Lang::OCaml => {
                r#"
                (let_binding pattern: (value_name) @name) @def
                (module_binding (module_name) @name) @def
                (type_binding name: (type_constructor) @name) @def
                "#
            }
            Lang::Elm => {
                r#"
                (value_declaration functionDeclarationLeft: (function_declaration_left (lower_case_identifier) @name)) @def
                (type_alias_declaration name: (upper_case_identifier) @name) @def
                "#
            }
            Lang::Nim => {
                r#"
                (proc_declaration name: (identifier) @name) @def
                (func_declaration name: (identifier) @name) @def
                "#
            }
            Lang::Erlang => {
                r#"
                (function_clause name: (atom) @name) @def
                (module_attribute name: (atom) @name) @def
                "#
            }
            Lang::Vim => {
                r#"
                (function_definition (function_declaration name: (identifier) @name)) @def
                "#
            }
            // Same shape as R: `foo = x: x + 1` is an attribute binding
            // whose value happens to be a lambda -- the structural
            // distinguisher is the binding's `expression:` field being a
            // `function_expression`, no predicate needed.
            Lang::Nix => {
                r#"
                (binding attrpath: (attrpath attr: (identifier) @name) expression: (function_expression)) @def
                "#
            }
            // HCL blocks (`resource "type" "name" { ... }`, `variable
            // "name" { ... }`) carry 0-2 string labels depending on block
            // kind, in a fixed but kind-dependent order with no field name
            // distinguishing "the label that names *this* declaration"
            // from "the label that names its *type*". This captures
            // whichever label comes first -- exactly the name for a
            // single-labeled block (`variable`/`output`/...), the type
            // rather than the specific instance name for a two-labeled
            // block (`resource`/`data`) -- good enough to make every block
            // searchable by `grv symbol`, not a precise per-resource name.
            Lang::Hcl => {
                r#"
                (block (identifier) (string_lit (template_literal) @name)) @def
                "#
            }
            Lang::CMake => {
                r#"
                (function_def (function_command (argument_list . (argument (unquoted_argument) @name)))) @def
                (macro_def (macro_command (argument_list . (argument (unquoted_argument) @name)))) @def
                "#
            }
            Lang::Verilog => {
                r#"
                (module_header (simple_identifier) @name) @def
                (function_declaration (function_body_declaration (function_identifier (function_identifier (simple_identifier) @name)))) @def
                "#
            }
            Lang::Vhdl => {
                r#"
                (entity_declaration entity: (identifier) @name) @def
                (architecture_definition architecture: (identifier) @name) @def
                "#
            }
            Lang::Fortran => {
                r#"
                (function (function_statement name: (name) @name)) @def
                (subroutine (subroutine_statement name: (name) @name)) @def
                (module (module_statement (name) @name)) @def
                "#
            }
            // Prolog has no separate "function definition" node either --
            // every fact/rule is just a `clause`, whose head is either the
            // clause's whole term (a fact, e.g. `foo(a).`) or the left
            // side of a `:-` rule (parsed generically as a
            // `binary_operation`, since `:-` is lexed as an operator, not
            // special-cased). Either way, the clause's *functor* name is a
            // real, structural definition point -- no predicates needed.
            Lang::Prolog => {
                r#"
                (clause term: (compound_term functor: (atom) @name)) @def
                (clause term: (binary_operation left: (compound_term functor: (atom) @name))) @def
                "#
            }
            // Racket and Scheme are generic S-expression grammars -- `define`
            // is not a distinct node type, just a `list` whose first symbol
            // happens to be the text "define" (indistinguishable at the
            // grammar level from a `list` that's an ordinary function call
            // like `(+ x 1)`). Same predicate technique as Elixir above:
            // `#eq?`/`#any-of?` on the leading symbol's text, checked by
            // `satisfies_text_predicates` in `extract_symbols`. `.` anchors
            // pin each captured child to an exact position so e.g. `(define
            // (foo x) (+ x 1))`'s *body* `(+ x 1)` can't also match the
            // "name is the second child" shape.
            Lang::Racket => {
                r#"
                ((list . (symbol) @kw . (symbol) @name) @def
                 (#any-of? @kw "define" "struct" "define-struct"))

                ((list . (symbol) @kw . (list . (symbol) @name)) @def
                 (#eq? @kw "define"))
                "#
            }
            Lang::Scheme => {
                r#"
                ((list . (symbol) @kw . (symbol) @name) @def
                 (#eq? @kw "define"))

                ((list . (symbol) @kw . (list . (symbol) @name)) @def
                 (#eq? @kw "define"))
                "#
            }
            Lang::Proto => {
                r#"
                (message (message_name (identifier) @name)) @def
                (service (service_name (identifier) @name)) @def
                (rpc (rpc_name (identifier) @name)) @def
                "#
            }
            // `class_interface`/`class_implementation`'s own name is a
            // bare (unlabeled) leading child, immediately before the
            // (also unlabeled but position-2) `superclass:`-field sibling
            // -- `.` anchors it to specifically the first child so this
            // doesn't also match the superclass name.
            Lang::ObjC => {
                r#"
                (class_interface . (identifier) @name) @def
                (class_implementation . (identifier) @name) @def
                (method_declaration (identifier) @name) @def
                (method_definition (identifier) @name) @def
                (function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
                "#
            }
            // GLSL/HLSL are C-family grammars for shader code -- same
            // node shapes as `Lang::C` (verified against a real parse,
            // not assumed from the family resemblance alone).
            Lang::Glsl => {
                r#"
                (function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
                (struct_specifier name: (type_identifier) @name) @def
                "#
            }
            Lang::Hlsl => {
                r#"
                (function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
                (struct_specifier name: (type_identifier) @name) @def
                "#
            }
            Lang::Ada => {
                r#"
                (package_declaration name: (identifier) @name) @def
                (package_body name: (identifier) @name) @def
                (subprogram_body (function_specification name: (identifier) @name)) @def
                (subprogram_body (procedure_specification name: (identifier) @name)) @def
                (subprogram_declaration (function_specification name: (identifier) @name)) @def
                (subprogram_declaration (procedure_specification name: (identifier) @name)) @def
                "#
            }
            Lang::Kotlin => {
                r#"
                (function_declaration name: (identifier) @name) @def
                (class_declaration name: (identifier) @name) @def
                "#
            }
            Lang::Wgsl => {
                r#"
                (function_declaration name: (identifier) @name) @def
                (struct_declaration name: (identifier) @name) @def
                "#
            }
            // LaTeX's `\script`/component content isn't its own concern
            // here -- `\section{...}`/`\label{...}` are the structural
            // "definitions" LaTeX's own grammar actually exposes; the
            // title's full text (which can be multiple words) lives in the
            // inner `text` node, not a single `word` -- capturing `word`
            // alone would only grab a title's first word.
            Lang::Latex => {
                r#"
                (section text: (curly_group (text) @name)) @def
                (label_definition name: (curly_group_label label: (label) @name)) @def
                "#
            }
            // Svelte/Vue deliberately have NO def_query_src: their own
            // grammars parse a `<script>` block's entire body as one
            // opaque `raw_text` node -- the actual function/variable
            // definitions inside it are real JS/TS, but recovering them
            // would mean a second parse pass (tree-sitter's "language
            // injection", which real editors do via a separate .scm query
            // an editor's own host application drives) that this
            // project's one-query-per-language design doesn't do. Still a
            // real win over the tagged tier: both are linked, parsed
            // grammars now (so e.g. future call-graph/injection-based work
            // has something to build on), just not ones with a
            // `grv symbol`-shaped answer today.
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
            Lang::C | Lang::Glsl | Lang::Hlsl => {
                r#"
                (call_expression function: (identifier) @callee) @call
                "#
            }
            Lang::Cpp => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (call_expression function: (field_expression field: (field_identifier) @callee)) @call
                (call_expression function: (qualified_identifier name: (identifier) @callee)) @call
                "#
            }
            Lang::Java | Lang::Groovy => {
                r#"
                (method_invocation name: (identifier) @callee) @call
                "#
            }
            Lang::CSharp => {
                r#"
                (invocation_expression function: (identifier) @callee) @call
                (invocation_expression function: (member_access_expression name: (identifier) @callee)) @call
                "#
            }
            Lang::Php => {
                r#"
                (function_call_expression function: (name) @callee) @call
                (member_call_expression name: (name) @callee) @call
                (scoped_call_expression name: (name) @callee) @call
                "#
            }
            Lang::Ruby => {
                r#"
                (call method: (identifier) @callee) @call
                "#
            }
            Lang::Bash => {
                r#"
                (command name: (command_name (word) @callee)) @call
                "#
            }
            Lang::Fish => {
                r#"
                (command name: (word) @callee) @call
                "#
            }
            Lang::Lua => {
                r#"
                (function_call name: (identifier) @callee) @call
                (function_call name: (method_index_expression method: (identifier) @callee)) @call
                (function_call name: (dot_index_expression field: (identifier) @callee)) @call
                "#
            }
            Lang::Solidity => {
                r#"
                (call_expression function: (expression (identifier) @callee)) @call
                (call_expression function: (expression (member_expression property: (identifier) @callee))) @call
                "#
            }
            // PowerShell cmdlet names (`Get-Process`, `Write-Host`) are
            // captured as the whole `command_name` node's text -- there's
            // no further identifier field to drill into (it's a leaf).
            Lang::PowerShell => {
                r#"
                (command command_name: (command_name) @callee) @call
                "#
            }
            Lang::Haskell => {
                r#"
                (apply function: (variable) @callee) @call
                "#
            }
            Lang::Dart => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (call_expression function: (member_expression property: (identifier) @callee)) @call
                "#
            }
            Lang::Zig => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (call_expression function: (field_expression . (field_expression) . (identifier) @callee)) @call
                (call_expression function: (field_expression member: (identifier) @callee)) @call
                "#
            }
            Lang::Julia => {
                r#"
                (call_expression (identifier) @callee) @call
                (call_expression (field_expression value: (identifier) . (identifier) @callee)) @call
                "#
            }
            // `method` field, not `name` -- see the def_query_src comment
            // on `Lang::Crystal` above for why (vendored fork, verified
            // via a real parse-tree dump).
            Lang::Crystal => {
                r#"
                (call method: (identifier) @callee) @call
                "#
            }
            Lang::D => {
                r#"
                (call_expression . (identifier) @callee) @call
                (call_expression (type (identifier) . (identifier) @callee)) @call
                "#
            }
            // Assembly has no dedicated "call" node -- a `call`/`jmp`/`jNN`
            // instruction's operand is structurally identical to any other
            // instruction's operand (both are just `(ident (reg (word)))`
            // per this grammar). The `#any-of?` predicate on the mnemonic
            // is what makes this a *call/jump-target* graph instead of
            // "every operand of every instruction" -- deliberately
            // includes conditional jumps alongside `call`, since a control-
            // flow-target graph is the closest analog assembly has to
            // "what does this call".
            Lang::Asm => {
                r#"
                ((instruction kind: (word) @mnemonic (ident (reg (word)) @callee)) @call
                 (#any-of? @mnemonic "call" "callq" "jmp" "je" "jne" "jz" "jnz" "jl" "jle" "jg" "jge" "ja" "jae" "jb" "jbe" "loop"))
                "#
            }
            // See the def_query_src comment on `Lang::Elixir`: `def`/
            // `defmodule`/control-flow macros are all plain `call` nodes.
            // `#not-any-of?` keeps this a *call graph*, not a "definition
            // graph" wearing a callee's clothes -- without it, every `def
            // foo` would also show up as if something called `def`.
            Lang::Elixir => {
                r#"
                ((call target: (identifier) @callee (arguments)) @call
                 (#not-any-of? @callee "def" "defp" "defmacro" "defmacrop" "defmodule" "if" "unless" "case" "cond" "receive" "try" "with" "for"))

                (call target: (dot right: (identifier) @callee)) @call
                "#
            }
            Lang::Scala => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (call_expression function: (field_expression field: (identifier) @callee)) @call
                "#
            }
            Lang::Swift => {
                r#"
                (call_expression . (simple_identifier) @callee) @call
                (call_expression (navigation_expression suffix: (navigation_suffix suffix: (simple_identifier) @callee))) @call
                "#
            }
            Lang::Perl => {
                r#"
                (call_expression_with_bareword function_name: (identifier) @callee) @call
                (method_invocation function_name: (identifier) @callee) @call
                "#
            }
            Lang::R => {
                r#"
                (call function: (identifier) @callee) @call
                "#
            }
            Lang::OCaml => {
                r#"
                (application_expression function: (value_path (value_name) @callee)) @call
                "#
            }
            Lang::Elm => {
                r#"
                (function_call_expr target: (value_expr name: (value_qid (lower_case_identifier) @callee))) @call
                "#
            }
            // `function` field, not `name` -- this project's vendored fork
            // (see vendor/tree-sitter-nim/NOTICE.md) renamed it relative to
            // the stale crates.io grammar this query was originally written
            // against; verified via a real parse-tree dump.
            Lang::Nim => {
                r#"
                (call function: (identifier) @callee) @call
                "#
            }
            Lang::Erlang => {
                r#"
                (call expr: (atom) @callee) @call
                "#
            }
            Lang::Vim => {
                r#"
                (call_expression function: (identifier) @callee) @call
                "#
            }
            Lang::Nix => {
                r#"
                (apply_expression function: (variable_expression name: (identifier) @callee)) @call
                "#
            }
            Lang::Hcl => {
                r#"
                (function_call (identifier) @callee) @call
                "#
            }
            Lang::CMake => {
                r#"
                (normal_command (identifier) @callee) @call
                "#
            }
            // `tf_call` (task/function call) covers a plain user
            // function/task call like `y = bar(1);` -- found on a second
            // real sample after the first attempt's sample tripped an
            // unrelated grammar quirk; `system_tf_call` is the separate
            // builtin `$display`/`$finish`/... form.
            Lang::Verilog => {
                r#"
                (system_tf_call (system_tf_identifier) @callee) @call
                (tf_call (simple_identifier) @callee) @call
                "#
            }
            Lang::Vhdl => {
                r#"
                (procedure_call_statement (name (identifier) @callee)) @call
                "#
            }
            Lang::Fortran => {
                r#"
                (subroutine_call subroutine: (identifier) @callee) @call
                (call_expression . (identifier) @callee) @call
                "#
            }
            // No dedicated "call" node in Prolog's grammar -- every
            // `compound_term` (a predicate applied to arguments) is
            // structurally the same whether it's a clause's head or a goal
            // in its body. Matching every one, unscoped, is simpler and
            // more robust across arbitrarily deep comma-conjunction chains
            // than trying to exclude just the head -- the head then also
            // shows up as a "call" to itself, a minor, documented
            // over-approximation consistent with this call graph's
            // existing name-based (not resolved) nature.
            Lang::Prolog => {
                r#"
                (compound_term functor: (atom) @callee) @call
                "#
            }
            // Same generic-S-expression reasoning as the def_query_src
            // comment on `Lang::Racket`/`Lang::Scheme`: everything is a
            // `list`, so "is this a call" needs the same `.`-anchored
            // leading-symbol capture, this time with `#not-any-of?` to
            // exclude the special forms/binders that aren't really calls
            // (including `define` itself, already covered separately).
            Lang::Racket => {
                r#"
                ((list . (symbol) @callee) @call
                 (#not-any-of? @callee "define" "struct" "define-struct" "lambda" "if" "cond" "let" "let*" "letrec" "begin" "when" "unless" "set!" "quote" "quasiquote" "unquote"))
                "#
            }
            Lang::Scheme => {
                r#"
                ((list . (symbol) @callee) @call
                 (#not-any-of? @callee "define" "lambda" "if" "cond" "let" "let*" "letrec" "begin" "when" "unless" "set!" "quote" "quasiquote" "unquote"))
                "#
            }
            Lang::ObjC => {
                r#"
                (call_expression function: (identifier) @callee) @call
                (message_expression method: (identifier) @callee) @call
                "#
            }
            Lang::Ada => {
                r#"
                (procedure_call_statement name: (identifier) @callee) @call
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

    #[test]
    fn elixir_defmodule_def_defp() {
        let src = "defmodule MyModule do\n  def foo(x) do\n    x + 1\n  end\n\n  defp bar(y), do: y * 2\nend\n";
        let found = names(Lang::Elixir, src);
        assert!(found.contains(&"MyModule".to_string()), "{found:?}");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn scala_function_class_object_trait() {
        let src = "object Main {\n  def foo(x: Int): Int = x + 1\n}\n\nclass Greeter(name: String) {\n  def greet(): String = name\n}\n\ntrait Shape {\n  def area(): Double\n}\n";
        let found = names(Lang::Scala, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Main".to_string()), "{found:?}");
        assert!(found.contains(&"Greeter".to_string()), "{found:?}");
        assert!(found.contains(&"greet".to_string()), "{found:?}");
        assert!(found.contains(&"Shape".to_string()), "{found:?}");
        assert!(found.contains(&"area".to_string()), "{found:?}");
    }

    #[test]
    fn swift_function_and_class() {
        let src = "func foo(x: Int) -> Int {\n    return x + 1\n}\n\nclass Greeter {\n    func greet() -> String {\n        return \"hi\"\n    }\n}\n\nstruct Point {\n    var x: Int\n}\n";
        let found = names(Lang::Swift, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Greeter".to_string()), "{found:?}");
        assert!(found.contains(&"greet".to_string()), "{found:?}");
        assert!(found.contains(&"Point".to_string()), "{found:?}");
    }

    #[test]
    fn perl_sub_and_package() {
        let src = "package MyPackage;\n\nsub foo {\n    my ($x) = @_;\n    return $x + 1;\n}\n\n1;\n";
        let found = names(Lang::Perl, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"MyPackage".to_string()), "{found:?}");
    }

    #[test]
    fn r_function_assignment() {
        let found = names(Lang::R, "foo <- function(x) {\n  x + 1\n}\n\nbar = function(y) y * 2\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn ocaml_let_module_type() {
        let src = "let foo x = x + 1\n\nmodule MyModule = struct\n  let bar y = y * 2\nend\n\ntype point = { x : int; y : int }\n";
        let found = names(Lang::OCaml, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"MyModule".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
        assert!(found.contains(&"point".to_string()), "{found:?}");
    }

    #[test]
    fn elm_function_and_type_alias() {
        let src = "module Main exposing (..)\n\nfoo : Int -> Int\nfoo x = x + 1\n\ntype alias Point = { x : Int, y : Int }\n";
        let found = names(Lang::Elm, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Point".to_string()), "{found:?}");
    }

    #[test]
    fn nim_proc_and_func() {
        let found = names(Lang::Nim, "proc foo(x: int): int =\n  x + 1\n\nfunc bar(y: int): int =\n  y * 2\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn erlang_function_clause_and_module() {
        let src = "-module(my_module).\n\nfoo(X) ->\n    X + 1.\n\nbar(Y) ->\n    Y * 2.\n";
        let found = names(Lang::Erlang, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
        assert!(found.contains(&"my_module".to_string()), "{found:?}");
    }

    #[test]
    fn vim_function() {
        let found = names(Lang::Vim, "function! Foo(x)\n  return a:x + 1\nendfunction\n");
        assert!(found.contains(&"Foo".to_string()), "{found:?}");
    }

    #[test]
    fn nix_function_bindings() {
        let found = names(Lang::Nix, "{\n  foo = x: x + 1;\n  bar = { a, b }: a + b;\n}\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn hcl_blocks() {
        let src = "resource \"aws_instance\" \"web\" {\n  ami = \"abc\"\n}\n\nvariable \"name\" {\n  default = \"x\"\n}\n";
        let found = names(Lang::Hcl, src);
        // Documented simplification (see Lang::Hcl's def_query_src comment):
        // the *first* label is captured -- the resource's type, not its
        // specific instance name -- and the variable's own name (its only
        // label) for the single-labeled block.
        assert!(found.contains(&"aws_instance".to_string()), "{found:?}");
        assert!(found.contains(&"name".to_string()), "{found:?}");
    }

    #[test]
    fn cmake_function_and_macro() {
        let src = "function(my_func arg1)\n  message(${arg1})\nendfunction()\n\nmacro(my_macro arg1)\n  message(${arg1})\nendmacro()\n";
        let found = names(Lang::CMake, src);
        assert!(found.contains(&"my_func".to_string()), "{found:?}");
        assert!(found.contains(&"my_macro".to_string()), "{found:?}");
    }

    #[test]
    fn verilog_module_and_function() {
        let src = "module counter(input clk, output reg [3:0] q);\n  always @(posedge clk) begin\n    q <= q + 1;\n  end\nendmodule\n\nfunction [3:0] add_one;\n  input [3:0] a;\n  add_one = a + 1;\nendfunction\n";
        let found = names(Lang::Verilog, src);
        assert!(found.contains(&"counter".to_string()), "{found:?}");
        assert!(found.contains(&"add_one".to_string()), "{found:?}");
    }

    #[test]
    fn vhdl_entity_and_architecture() {
        let src = "entity counter is\n  port (clk : in std_logic);\nend entity counter;\n\narchitecture behavior of counter is\nbegin\nend architecture behavior;\n";
        let found = names(Lang::Vhdl, src);
        assert!(found.contains(&"counter".to_string()), "{found:?}");
        assert!(found.contains(&"behavior".to_string()), "{found:?}");
    }

    #[test]
    fn fortran_function_subroutine_module() {
        let src = "module my_module\ncontains\n  function foo(x) result(y)\n    integer :: x, y\n    y = x + 1\n  end function foo\n\n  subroutine bar(z)\n    integer :: z\n    z = z * 2\n  end subroutine bar\nend module my_module\n";
        let found = names(Lang::Fortran, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
        assert!(found.contains(&"my_module".to_string()), "{found:?}");
    }

    #[test]
    fn prolog_facts_and_rules() {
        let found = names(Lang::Prolog, "foo(X, Y) :- Y is X + 1.\n\nbar(X) :- foo(X, _).\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn racket_define_and_struct() {
        let src = "#lang racket\n(define (foo x) (+ x 1))\n(define bar 42)\n(struct point (x y))\n";
        let found = names(Lang::Racket, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
        assert!(found.contains(&"point".to_string()), "{found:?}");
        // The real bar this predicate machinery has to clear: an ordinary
        // call like `(+ x 1)` must NOT be mistaken for a definition.
        assert!(!found.contains(&"+".to_string()), "{found:?}");
    }

    #[test]
    fn scheme_define() {
        let found = names(Lang::Scheme, "(define (foo x) (+ x 1))\n(define bar 42)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
        assert!(!found.contains(&"+".to_string()), "{found:?}");
    }

    #[test]
    fn proto_message_service_rpc() {
        let src = "syntax = \"proto3\";\n\nmessage Person {\n  string name = 1;\n  int32 id = 2;\n}\n\nservice Greeter {\n  rpc SayHello (Person) returns (Person);\n}\n";
        let found = names(Lang::Proto, src);
        assert!(found.contains(&"Person".to_string()), "{found:?}");
        assert!(found.contains(&"Greeter".to_string()), "{found:?}");
        assert!(found.contains(&"SayHello".to_string()), "{found:?}");
    }

    #[test]
    fn objc_interface_impl_function() {
        let src = "@interface Greeter : NSObject\n- (NSString *)greet;\n@end\n\n@implementation Greeter\n- (NSString *)greet {\n    return @\"hi\";\n}\n@end\n\nint foo(int x) {\n    return x + 1;\n}\n";
        let found = names(Lang::ObjC, src);
        assert!(found.contains(&"Greeter".to_string()), "{found:?}");
        assert!(found.contains(&"greet".to_string()), "{found:?}");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
    }

    #[test]
    fn glsl_function_and_struct() {
        let src = "float foo(float x) {\n    return x + 1.0;\n}\n\nstruct Point {\n    float x;\n    float y;\n};\n\nvoid main() {\n    foo(1.0);\n}\n";
        let found = names(Lang::Glsl, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Point".to_string()), "{found:?}");
        assert!(found.contains(&"main".to_string()), "{found:?}");
    }

    #[test]
    fn hlsl_function_and_struct() {
        let src = "float foo(float x) {\n    return x + 1.0;\n}\n\nstruct Point {\n    float x;\n    float y;\n};\n";
        let found = names(Lang::Hlsl, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Point".to_string()), "{found:?}");
    }

    #[test]
    fn ada_package_function_procedure() {
        let src = "package My_Package is\n   function Foo (X : Integer) return Integer;\n   procedure Bar (Y : Integer);\nend My_Package;\n\npackage body My_Package is\n   function Foo (X : Integer) return Integer is\n   begin\n      return X + 1;\n   end Foo;\n\n   procedure Bar (Y : Integer) is\n   begin\n      null;\n   end Bar;\nend My_Package;\n";
        let found = names(Lang::Ada, src);
        assert!(found.contains(&"My_Package".to_string()), "{found:?}");
        assert!(found.contains(&"Foo".to_string()), "{found:?}");
        assert!(found.contains(&"Bar".to_string()), "{found:?}");
    }
}

#[cfg(test)]
mod call_queries {
    // Each `call_query_src` addition verified against a real parse of real
    // sample code, same discipline as `new_language_queries` above --
    // `crate::extract_calls`'s output checked against a hand-written
    // sample containing real call sites, not just "the query compiles".
    use super::*;

    fn callees(lang: Lang, src: &str) -> Vec<String> {
        crate::extract_calls(src, lang).into_iter().map(|c| c.callee_name).collect()
    }

    #[test]
    fn c_call() {
        let found = callees(Lang::C, "int main() {\n    foo(1);\n    return bar(2);\n}\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn cpp_call_field_and_qualified() {
        let src = "void main() {\n    foo(1);\n    obj.method(2);\n    obj->method(3);\n    Namespace::func(4);\n}\n";
        let found = callees(Lang::Cpp, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.iter().filter(|n| *n == "method").count() >= 2, "{found:?}");
        assert!(found.contains(&"func".to_string()), "{found:?}");
    }

    #[test]
    fn java_method_invocation() {
        let src = "class A {\n  void m() {\n    foo(1);\n    obj.method(2);\n    this.method(3);\n  }\n}\n";
        let found = callees(Lang::Java, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.iter().filter(|n| *n == "method").count() >= 2, "{found:?}");
    }

    #[test]
    fn csharp_invocation() {
        let src = "class A {\n  void M() {\n    Foo(1);\n    obj.Method(2);\n    this.Method(3);\n  }\n}\n";
        let found = callees(Lang::CSharp, src);
        assert!(found.contains(&"Foo".to_string()), "{found:?}");
        assert!(found.iter().filter(|n| *n == "Method").count() >= 2, "{found:?}");
    }

    #[test]
    fn php_function_member_scoped_call() {
        let src = "<?php\nfoo(1);\n$obj->method(2);\nKlass::method(3);\n";
        let found = callees(Lang::Php, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.iter().filter(|n| *n == "method").count() >= 2, "{found:?}");
    }

    #[test]
    fn ruby_call() {
        let found = callees(Lang::Ruby, "foo(1)\nobj.method(2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn bash_command() {
        let found = callees(Lang::Bash, "foo bar\nls -la\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"ls".to_string()), "{found:?}");
    }

    #[test]
    fn fish_command() {
        let found = callees(Lang::Fish, "foo bar\necho hi\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"echo".to_string()), "{found:?}");
    }

    #[test]
    fn lua_function_call_and_methods() {
        let src = "foo(1)\nobj:method(2)\nobj.method(3)\n";
        let found = callees(Lang::Lua, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.iter().filter(|n| *n == "method").count() >= 2, "{found:?}");
    }

    #[test]
    fn solidity_call() {
        let src = "contract A {\n  function m() public {\n    foo(1);\n    this.bar(2);\n  }\n}\n";
        let found = callees(Lang::Solidity, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn powershell_command() {
        let found = callees(Lang::PowerShell, "Get-Process\nWrite-Host \"hi\"\n");
        assert!(found.contains(&"Get-Process".to_string()), "{found:?}");
        assert!(found.contains(&"Write-Host".to_string()), "{found:?}");
    }

    #[test]
    fn haskell_apply() {
        let found = callees(Lang::Haskell, "main = do\n  foo 1\n  bar 2\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn dart_call_and_method() {
        let src = "void f() {\n  foo(1);\n  obj.method(2);\n}\n";
        let found = callees(Lang::Dart, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn zig_call_and_field_call() {
        let src = "fn f() void {\n    foo(1);\n    std.debug.print(\"hi\", .{});\n}\n";
        let found = callees(Lang::Zig, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"print".to_string()), "{found:?}");
    }

    #[test]
    fn julia_call_and_field_call() {
        let found = callees(Lang::Julia, "foo(1)\nbar.method(2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn groovy_method_invocation() {
        let found = callees(Lang::Groovy, "foo(1)\nobj.method(2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn crystal_call() {
        let found = callees(Lang::Crystal, "foo(1)\nobj.method(2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn d_call_and_qualified_call() {
        let src = "void f() {\n    foo(1);\n    obj.method(2);\n}\n";
        let found = callees(Lang::D, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn asm_call_and_jump_targets() {
        let found = callees(Lang::Asm, "call foo\njmp bar\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn elixir_call_excludes_definition_keywords() {
        let found = callees(Lang::Elixir, "foo(1)\nMod.bar(2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
        // The real bar the #not-any-of? predicate has to clear.
        let found2 = callees(Lang::Elixir, "defmodule M do\n  def foo(x) do\n    x\n  end\nend\n");
        assert!(!found2.contains(&"def".to_string()), "{found2:?}");
        assert!(!found2.contains(&"defmodule".to_string()), "{found2:?}");
    }

    #[test]
    fn scala_call_and_field_call() {
        let src = "object A {\n  def m() = {\n    foo(1)\n    obj.method(2)\n  }\n}\n";
        let found = callees(Lang::Scala, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn swift_call_and_navigation_call() {
        let src = "func f() {\n    foo(1)\n    obj.method(2)\n}\n";
        let found = callees(Lang::Swift, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn perl_call_and_method() {
        let found = callees(Lang::Perl, "foo(1);\n$obj->method(2);\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn r_call() {
        let found = callees(Lang::R, "foo(1)\nbar(2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn ocaml_application() {
        let found = callees(Lang::OCaml, "let () =\n  foo 1;\n  bar 2\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn elm_function_call() {
        let found = callees(Lang::Elm, "f = foo 1\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
    }

    #[test]
    fn nim_call() {
        let found = callees(Lang::Nim, "foo(1)\nbar(2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn erlang_call() {
        let found = callees(Lang::Erlang, "f() ->\n    foo(1),\n    bar(2).\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn vim_call_expression() {
        let found = callees(Lang::Vim, "call Foo(1)\necho Bar()\n");
        assert!(found.contains(&"Foo".to_string()), "{found:?}");
        assert!(found.contains(&"Bar".to_string()), "{found:?}");
    }

    #[test]
    fn nix_apply_expression() {
        let found = callees(Lang::Nix, "foo 1\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
    }

    #[test]
    fn hcl_function_call() {
        let found = callees(Lang::Hcl, "x = foo(1)\ny = bar(2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn cmake_normal_command() {
        let found = callees(Lang::CMake, "message(hi)\nfoo(bar)\n");
        assert!(found.contains(&"message".to_string()), "{found:?}");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
    }

    #[test]
    fn verilog_system_tf_call() {
        let src = "module m;\n  initial begin\n    $display(\"hi\");\n  end\nendmodule\n";
        let found = callees(Lang::Verilog, src);
        assert!(found.contains(&"$display".to_string()), "{found:?}");
    }

    #[test]
    fn verilog_plain_task_function_call() {
        let src = "module m;\n  function integer bar;\n    input integer x;\n    begin\n      bar = x;\n    end\n  endfunction\n\n  initial begin\n    reg [31:0] y;\n    y = bar(1);\n  end\nendmodule\n";
        let found = callees(Lang::Verilog, src);
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn vhdl_procedure_call() {
        let src = "architecture behavior of counter is\nbegin\n  process is\n  begin\n    foo(1);\n  end process;\nend architecture behavior;\n";
        let found = callees(Lang::Vhdl, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
    }

    #[test]
    fn fortran_call_and_call_expression() {
        let found = callees(Lang::Fortran, "program p\n  call foo(1)\n  x = bar(2)\nend program p\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn prolog_compound_term_calls() {
        let found = callees(Lang::Prolog, "f(X) :- foo(X), bar(X).\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
    }

    #[test]
    fn racket_call_excludes_special_forms() {
        let found = callees(Lang::Racket, "(foo 1)\n(bar 2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
        let found2 = callees(Lang::Racket, "(define (foo x) (if x 1 2))\n");
        assert!(!found2.contains(&"define".to_string()), "{found2:?}");
        assert!(!found2.contains(&"if".to_string()), "{found2:?}");
    }

    #[test]
    fn scheme_call_excludes_special_forms() {
        let found = callees(Lang::Scheme, "(foo 1)\n(bar 2)\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"bar".to_string()), "{found:?}");
        let found2 = callees(Lang::Scheme, "(define (foo x) (let ((y 1)) y))\n");
        assert!(!found2.contains(&"define".to_string()), "{found2:?}");
        assert!(!found2.contains(&"let".to_string()), "{found2:?}");
    }

    #[test]
    fn objc_call_and_message() {
        let src = "void f() {\n    foo(1);\n    [obj method:2];\n}\n";
        let found = callees(Lang::ObjC, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"method".to_string()), "{found:?}");
    }

    #[test]
    fn glsl_call() {
        let found = callees(Lang::Glsl, "void f() {\n    foo(1.0);\n}\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
    }

    #[test]
    fn hlsl_call() {
        let found = callees(Lang::Hlsl, "void f() {\n    foo(1.0);\n}\n");
        assert!(found.contains(&"foo".to_string()), "{found:?}");
    }

    #[test]
    fn ada_procedure_call() {
        let found = callees(Lang::Ada, "procedure P is\nbegin\n   Foo (1);\n   Bar (2);\nend P;\n");
        assert!(found.contains(&"Foo".to_string()), "{found:?}");
        assert!(found.contains(&"Bar".to_string()), "{found:?}");
    }
}

/// The automated version of the bug hunt that found Elixir's `if x do y
/// end` being wrongly extracted as a definition named "x": a
/// `(#eq?/#any-of?/#not-any-of?/#match?/...)` predicate written as a
/// sibling top-level form after `(pattern) @capture` — instead of nested
/// *inside* the same outer parens, `((pattern) @capture (#pred? ...))` —
/// silently compiles into an extra, content-less pattern of its own,
/// leaving the real pattern's predicate list empty (and therefore
/// unfiltered). That bug shipped, undetected, in three languages' `def_query_src`
/// for an entire session, because every sample those queries were tested
/// against happened not to need the predicate to get the right answer.
///
/// Rather than trust "someone will remember to write an adversarial test
/// for every future predicate", this checks a structural invariant that's
/// true for every query in this file regardless of what it matches: every
/// intended top-level pattern ends with exactly one `@def` (or `@call`)
/// capture, so a compiled query's `pattern_count()` must equal how many
/// times that capture name appears in the source. If a predicate ever
/// gets mis-nested again — in a language that exists today, or one added
/// next year — `pattern_count()` silently grows past that number, and
/// this test fails immediately, for every language, without needing a
/// human to think up the specific adversarial input that would expose it.
#[cfg(test)]
mod query_predicate_safety_net {
    use super::*;

    /// Exact-token count of `capture` (e.g. `@call`) in `src` -- a plain
    /// `src.matches(capture).count()` would also count it as a substring
    /// of a longer capture name (`@call` inside `@callee`, which every
    /// call query also has), so this requires the character right after
    /// the match to NOT be a capture-name continuation character.
    fn count_capture_token(src: &str, capture: &str) -> usize {
        let is_capture_char = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-';
        src.match_indices(capture)
            .filter(|(i, m)| src[i + m.len()..].chars().next().is_none_or(|c| !is_capture_char(c)))
            .count()
    }

    fn assert_pattern_count_matches_captures(lang: Lang, src: &str, capture: &str, query_kind: &str) {
        let Some(ts_lang) = lang.ts_language() else { return };
        let query = match Query::new(&ts_lang, src) {
            Ok(q) => q,
            Err(e) => panic!("{lang:?}: {query_kind} failed to compile: {e}"),
        };
        let expected = count_capture_token(src, capture);
        assert_eq!(
            query.pattern_count(),
            expected,
            "{lang:?}: {query_kind} compiled to {} pattern(s) but has {expected} `{capture}` capture(s) -- \
             a predicate is very likely written as a sibling top-level form instead of nested inside \
             the pattern's own parens. Wrong:\n  (pattern) @{cap}\n  (#eq? ...)\nRight:\n  ((pattern) @{cap}\n   (#eq? ...))",
            query.pattern_count(),
            cap = capture.trim_start_matches('@'),
        );
    }

    #[test]
    fn every_def_query_pattern_count_matches_its_def_captures() {
        for &lang in ALL_LANGS {
            if let Some(src) = lang.def_query_src() {
                assert_pattern_count_matches_captures(lang, src, "@def", "def_query_src");
            }
        }
    }

    #[test]
    fn every_call_query_pattern_count_matches_its_call_captures() {
        for &lang in ALL_LANGS {
            if let Some(src) = lang.call_query_src() {
                assert_pattern_count_matches_captures(lang, src, "@call", "call_query_src");
            }
        }
    }

    /// Ties `ALL_LANGS` to the actual enum: `_all_langs_exhaustive_match_guard`
    /// fails to *compile* the moment a new `Lang` variant exists without an
    /// arm for it, and this count is the second half -- it fails at *test*
    /// time if that new variant's arm was added there but never added to
    /// `ALL_LANGS` itself (the two tests above, and anything else built on
    /// `ALL_LANGS`, would otherwise silently skip it forever). Update this
    /// number (and `ALL_LANGS`) together when adding a language.
    #[test]
    fn all_langs_has_every_known_variant() {
        assert_eq!(ALL_LANGS.len(), 67, "a Lang variant was added/removed without updating ALL_LANGS to match");
    }
}

/// The original six languages (Rust/Python/JS/TS/TSX/Go) had never had a
/// single real-sample assertion test of their own, unlike every language
/// added since -- `query_predicate_safety_net` catching TypeScript's
/// `class_declaration` field-type mismatch (a compile-time "impossible
/// pattern" that silently zeroed out ALL TypeScript/TSX symbol extraction,
/// not just classes) is exactly what that gap let through undetected.
/// This closes it: real def + call extraction, asserted against real
/// samples, for the languages this whole tool was originally built around.
#[cfg(test)]
mod original_six_queries {
    use super::*;

    fn names(lang: Lang, src: &str) -> Vec<String> {
        crate::extract_symbols(src, lang).into_iter().map(|s| s.name).collect()
    }

    fn callees(lang: Lang, src: &str) -> Vec<String> {
        crate::extract_calls(src, lang).into_iter().map(|c| c.callee_name).collect()
    }

    #[test]
    fn rust_def_and_call() {
        let src = "struct Point { x: i32 }\nenum Color { Red }\ntrait Shape {}\nimpl Point {}\nmod util {}\n\nfn foo(x: i32) -> i32 {\n    bar(x);\n    x.method();\n    Point::new();\n    println!(\"{}\", x);\n    x\n}\n";
        let n = names(Lang::Rust, src);
        for want in ["Point", "Color", "Shape", "util", "foo"] {
            assert!(n.contains(&want.to_string()), "names={n:?}");
        }
        let c = callees(Lang::Rust, src);
        for want in ["bar", "method", "new", "println"] {
            assert!(c.contains(&want.to_string()), "callees={c:?}");
        }
    }

    #[test]
    fn python_def_and_call() {
        let src = "class Greeter:\n    def greet(self):\n        foo(1)\n        self.helper()\n\ndef foo(x):\n    return x\n";
        let n = names(Lang::Python, src);
        for want in ["Greeter", "greet", "foo"] {
            assert!(n.contains(&want.to_string()), "names={n:?}");
        }
        let c = callees(Lang::Python, src);
        for want in ["foo", "helper"] {
            assert!(c.contains(&want.to_string()), "callees={c:?}");
        }
    }

    #[test]
    fn javascript_def_and_call() {
        let src = "class Greeter {\n  greet() {\n    foo(1);\n    this.helper();\n  }\n}\n\nfunction foo(x) {\n  return x;\n}\n";
        let n = names(Lang::JavaScript, src);
        for want in ["Greeter", "greet", "foo"] {
            assert!(n.contains(&want.to_string()), "names={n:?}");
        }
        let c = callees(Lang::JavaScript, src);
        for want in ["foo", "helper"] {
            assert!(c.contains(&want.to_string()), "callees={c:?}");
        }
    }

    #[test]
    fn typescript_class_is_not_silently_dropped() {
        // The exact regression: class_declaration's name field is a
        // type_identifier in TypeScript, not a plain identifier -- get
        // this wrong and Query::new rejects the WHOLE multi-pattern query
        // as an "impossible pattern", so this asserts every def kind in
        // one query still comes back, not just the class.
        let src = "interface Shape {\n  area(): number;\n}\n\nclass Greeter {\n  greet(): string {\n    foo(1);\n    this.helper();\n    return \"hi\";\n  }\n}\n\nfunction foo(x: number): number {\n  return x;\n}\n";
        let n = names(Lang::TypeScript, src);
        for want in ["Shape", "Greeter", "greet", "foo"] {
            assert!(n.contains(&want.to_string()), "names={n:?}");
        }
        let c = callees(Lang::TypeScript, src);
        for want in ["foo", "helper"] {
            assert!(c.contains(&want.to_string()), "callees={c:?}");
        }
    }

    #[test]
    fn tsx_class_is_not_silently_dropped() {
        let src = "class Greeter {\n  greet(): string {\n    foo(1);\n    return \"hi\";\n  }\n}\n\nfunction foo(x: number): number {\n  return x;\n}\n";
        let n = names(Lang::Tsx, src);
        for want in ["Greeter", "greet", "foo"] {
            assert!(n.contains(&want.to_string()), "names={n:?}");
        }
    }

    #[test]
    fn go_def_and_call() {
        let src = "package main\n\ntype Point struct {\n\tX int\n}\n\nfunc foo(x int) int {\n\tbar(x)\n\tp := Point{}\n\tp.Method()\n\treturn x\n}\n";
        let n = names(Lang::Go, src);
        for want in ["Point", "foo"] {
            assert!(n.contains(&want.to_string()), "names={n:?}");
        }
        let c = callees(Lang::Go, src);
        for want in ["bar", "Method"] {
            assert!(c.contains(&want.to_string()), "callees={c:?}");
        }
    }
}

/// Kotlin/Svelte/Vue/WGSL/LaTeX all used to be "genuinely unlinkable"
/// (see git history / ARCHITECTURE.md for the real type-mismatch and
/// missing-scanner-file failures that earned them that classification).
/// Each is now backed by a real, actively-maintained fork that was
/// individually verified to link *and* parse before being trusted --
/// these tests are that verification, permanently, not just "it compiled
/// today".
#[cfg(test)]
mod newly_unblocked_grammars {
    use super::*;

    fn names(lang: Lang, src: &str) -> Vec<String> {
        crate::extract_symbols(src, lang).into_iter().map(|s| s.name).collect()
    }

    fn parses_without_error(lang: Lang, src: &str) {
        let ts_lang = lang.ts_language().unwrap_or_else(|| panic!("{lang:?}: grammar not linked"));
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&ts_lang).unwrap_or_else(|e| panic!("{lang:?}: set_language failed: {e}"));
        let tree = parser.parse(src, None).unwrap_or_else(|| panic!("{lang:?}: parse returned None"));
        assert!(!tree.root_node().has_error(), "{lang:?}: parse tree has an ERROR node for real sample code:\n{}", tree.root_node().to_sexp());
    }

    #[test]
    fn kotlin_function_and_class() {
        let src = "fun foo(x: Int): Int {\n    return bar(x)\n}\n\nclass Greeter(val name: String) {\n    fun greet(): String {\n        return name\n    }\n}\n";
        parses_without_error(Lang::Kotlin, src);
        let found = names(Lang::Kotlin, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Greeter".to_string()), "{found:?}");
        assert!(found.contains(&"greet".to_string()), "{found:?}");
    }

    #[test]
    fn wgsl_function_and_struct() {
        let src = "fn foo(x: f32) -> f32 {\n    return bar(x);\n}\n\nstruct Point {\n    x: f32,\n    y: f32,\n}\n";
        parses_without_error(Lang::Wgsl, src);
        let found = names(Lang::Wgsl, src);
        assert!(found.contains(&"foo".to_string()), "{found:?}");
        assert!(found.contains(&"Point".to_string()), "{found:?}");
    }

    #[test]
    fn latex_section_and_label() {
        let src = "\\section{Introduction}\n\\label{sec:intro}\n\nSome text \\cite{foo} here.\n\n\\begin{equation}\n  x = y\n\\end{equation}\n";
        parses_without_error(Lang::Latex, src);
        let found = names(Lang::Latex, src);
        assert!(found.contains(&"Introduction".to_string()), "{found:?}");
        assert!(found.contains(&"sec:intro".to_string()), "{found:?}");
    }

    #[test]
    fn svelte_parses_a_real_component() {
        // No def_query_src (see its doc comment) -- a real parse with no
        // ERROR node is the actual claim being verified here.
        parses_without_error(
            Lang::Svelte,
            "<script>\n  function foo(x) {\n    return bar(x);\n  }\n  let count = 0;\n</script>\n\n<button on:click={foo}>{count}</button>\n",
        );
    }

    #[test]
    fn vue_parses_a_real_component() {
        parses_without_error(
            Lang::Vue,
            "<template>\n  <button @click=\"foo\">{{ count }}</button>\n</template>\n\n<script>\nexport default {\n  methods: {\n    foo() {\n      return bar();\n    }\n  }\n}\n</script>\n",
        );
    }
}


# Provenance

Vendored from https://github.com/crystal-lang-tools/tree-sitter-crystal,
commit `50ca9e6fcfb16a2cbcad59203cfd8ad650e25c49` (fetched 2026-08-28).

## Why vendored instead of a normal crates.io dependency

The only `tree-sitter-crystal` published on crates.io (version `0.1.0`,
`https://github.com/undivisible/tree-sitter-crystal`) **cannot parse
`require "..."` as a call/macro-invocation node at all** -- confirmed via
a real parse-tree dump: it splits into two unrelated `expression_statement`
nodes, so there is no node shape for GRAVITON's real import resolver to
hook into. It was also published by a brand-new, zero-star account
(`undivisible`) that batch-published several other language grammars
(including, separately, `tree-sitter-nim`) on the same day -- the same
batch-publishing pattern this project's own `crates/indexer/Cargo.toml`
already treats with suspicion for the Vue ecosystem (see the note there
from v0.15).

`crystal-lang-tools` is Crystal's own official tooling GitHub org (the
`crystalline` language server lives there too), and their grammar has a
real, dedicated `require` node (`require: $ => seq('require', $.string)`)
-- verified directly by parsing real Crystal source and inspecting the
resulting tree, not assumed from the grammar source alone.

## Why vendored instead of a `git` Cargo dependency

Unlike the Nim fork, this grammar's own Rust binding already uses the
correct `tree-sitter-language` ABI shim (the same shape every other
grammar in this workspace uses) -- no manifest patch was needed, a plain
`git` dependency pinned to the commit above should just work. In
practice, `cargo build` against
`git = "https://github.com/crystal-lang-tools/tree-sitter-crystal"`
repeatedly failed in this sandbox with a transient SSL/network error
(`SSL error: unknown error; class=Ssl (16)`), even after enabling
`net.git-fetch-with-cli` -- while a plain `git clone` of the exact same
commit succeeded reliably in well under a minute. Rather than keep
fighting this specific sandbox's flaky connectivity to this one host (a
known recurring issue for this project -- see the memory note on
`cargo-audit`'s own yanked-check timeouts), the already-successfully-
cloned source was vendored directly.

## Exactly what was changed from the upstream commit above

Nothing. Every vendored file (`grammar.js`, `src/{parser.c,scanner.c,
unicode.c,node-types.json,grammar.json,tree_sitter/*.h}`,
`bindings/rust/{lib.rs,build.rs}`, `LICENSE`) is copied byte-for-byte from
the upstream commit. Only `Cargo.toml` is new (a minimal manifest for
this vendored copy, `publish = false` since it's never meant to be
published under this name) and files unrelated to building the Rust
binding (samples/, test/, other-language bindings, CI config, editor
config) were left out -- they aren't needed to compile this crate.

## License

MIT (see `LICENSE`), same as upstream.

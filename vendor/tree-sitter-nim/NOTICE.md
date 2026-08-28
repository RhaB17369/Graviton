# Provenance

Vendored from https://github.com/alaviss/tree-sitter-nim, commit
`ac72ba30d16edf0be021588a9301ede4accd6cf4` (fetched 2026-08-28).

## Why vendored instead of a normal crates.io dependency

The only `tree-sitter-nim` published on crates.io (version `0.1.0`,
`https://github.com/undivisible/tree-sitter-nim`) has **no import/include
node type in its grammar at all** -- confirmed by inspecting its real
`node-types.json` -- so GRAVITON's real import resolver has nothing to hook
into via that crate. It was also published by a brand-new, zero-star
account (`undivisible`) that published several other language grammars
(including this project's previously-pinned `tree-sitter-crystal`) on the
same day -- the same batch-publishing pattern this project's own
`crates/indexer/Cargo.toml` comment already treats with suspicion for the
Vue ecosystem (see the note there from v0.15).

`alaviss/tree-sitter-nim` is an actively maintained, richer grammar (61
stars, pushed within the last month at the time of writing) that DOES have
real `import_statement`/`import_from_statement`/`include_statement` nodes
-- verified directly by parsing real Nim source and inspecting the
resulting tree, not assumed from its README. It isn't published to
crates.io at all, so a normal `git`/registry dependency isn't available.

## Why vendored instead of a `git` dependency to a fork

The fix needed is exactly one line: this grammar's own Rust binding pins
`tree-sitter = "~0.25"`, which conflicts (`links = "tree-sitter"`, only one
version allowed in the whole dependency graph) with this workspace's
`tree-sitter = "0.26"`. Relaxing that one line to `tree-sitter = "0.26"` was
verified to work correctly with zero other changes (a real parse of
`import`/`from ... import`/`include` all produced correct, error-free
trees against tree-sitter 0.26's actual runtime -- the generated parser's
C-level ABI is unaffected by which Rust crate version generated the
binding glue).

Hosting that one-line patch as a `git` dependency would normally be the
preferred, non-vendored approach (as already done for `tree-sitter-vuejs`
in this same project) -- but doing so needs a place to host the patched
fork, and this session had no GitHub API/`gh` CLI credentials available to
create a new repository. The user was asked and explicitly chose to vendor
directly rather than wait on that.

## Exactly what was changed from the upstream commit above

- `Cargo.toml`: `tree-sitter = "~0.25"` -> `tree-sitter = "0.26"`. Nothing
  else. `publish = false` added since this is a local vendored copy, not
  meant to ever be published under this name.
- Everything else (`grammar.js`, `src/parser.c`, `src/scanner.c`,
  `src/node-types.json`, `src/grammar.json`, `src/tree_sitter/*.h`,
  `bindings/rust/{lib.rs,build.rs}`, `LICENSE.txt`, `LICENSES/`) is copied
  verbatim from the upstream commit, unmodified. Files unrelated to
  building the Rust binding (samples/, test/, other-language bindings,
  CI config, editor config) were left out of this vendored copy -- they
  aren't needed to compile this crate and would just add more weight to
  this repo for no functional benefit.

## License

MPL-2.0 (see `LICENSE.txt`, `LICENSES/`), same as upstream. The one-line
change above is disclosed here per MPL-2.0 section 3.3's modification-notice
requirement.

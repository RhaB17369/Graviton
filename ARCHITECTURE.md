# Architecture

## Hardware this was designed for

- CPU: Intel i7-10850H, 6C/12T
- RAM: 16GB (usable ~9-10GB once desktop environment + browser are up)
- GPU: Quadro P620, 4GB VRAM, Pascal — CUDA 13 capable but too little VRAM to
  fully offload anything past a small model
- Swap: 31GB, but swap thrashing turns "slow" into "unusable" — the design
  goal throughout is to *never* need it

Conclusion driving every decision below: **treat VRAM as a small accelerator,
not the primary memory budget.** RAM is the real constraint, and it's
generous enough for an 8B model at Q4 quantization (~5-6GB resident) with
room left for the OS and GRAVITON's own index.

## Why not just increase the context window?

A model's context window (e.g. Qwen3 8B's 32,768 tokens) bounds how much
text it can hold *at once* — instructions + history + injected code +
output. A large real-world codebase is easily 10-100x that. Two bad options:

1. Truncate/summarize the repo to fit — loses precision exactly where it
   matters (the one function with the bug).
2. Buy a model with a bigger window and feed it more raw code anyway — this
   still doesn't scale to multi-million-token repos, costs proportionally
   more KV-cache RAM (which *does* scale with context length, independent of
   model size), and dumps irrelevant code in front of the model, which
   measurably hurts reasoning quality ("needle in a haystack" degradation).

The fix isn't a bigger window, it's not needing one: retrieve only what's
relevant, keep the window small and dense. That's what `grv index` /
`context.rs` do.

## Component breakdown

```
graviton/
├── crates/
│   ├── core/       Config (~/.config/graviton/config.toml), SQLite schema, shared types
│   ├── indexer/    tree-sitter symbol extraction + line-window chunking + repo walker
│   ├── llm/        Ollama HTTP client (streaming /api/chat)
│   └── cli/        `grv` binary: subcommands, retrieval/context assembly, prompts
```

Rust edition 2024 (bumped from 2021 once the toolchain running this project
comfortably supported it — no code changes were needed; the workspace
already avoided the patterns 2024 changes the defaults for).

### Why Rust over C/C++ for this

- Memory safety matters more than usual here: this walks and parses
  *untrusted, adversarial* input by design (CTF binaries, obfuscated
  challenge code, malware samples you're reverse-engineering) — a
  memory-safe indexer means a crash-worthy input doesn't become an exploit
  primitive against your own tooling.
- `cargo` gives single-binary distribution, no separate build system, and
  first-class tree-sitter/SQLite/async-HTTP bindings.
- Performance is within a few percent of C/C++ for this workload (I/O- and
  parse-bound, not tight numeric loops), so there's no real cost to the
  safety.

C/C++ would only earn its keep for a hypothetical future component doing
something tree-sitter/existing crates can't do at all (e.g. a custom
disassembler) — not for the indexing/retrieval/CLI layer itself.

### Storage: SQLite + FTS5, not tantivy, not a vector DB

Earlier sketches of this idea reached for tantivy (full-text) plus an
embeddings/vector store. For v1 that's over-engineering:

- SQLite FTS5 (bm25 ranking) ships in `rusqlite`'s `bundled` feature — zero
  extra native dependencies, one file on disk, `grv index` and `grv search`
  read/write the same `.graviton/index.db`.
- Symbol name lookup and full-text chunk search cover the two retrieval
  patterns that matter most for "explain/trace/find X in this codebase"
  and "give me the source of function Y" — the bulk of real usage.
- No embedding model to keep loaded alongside the chat model on a
  16GB-RAM box, *unless you ask for one* — semantic search (below) is opt-in
  and off by default, so this constraint is respected until a user
  deliberately trades some RAM/setup time for better retrieval.

Semantic (embedding-based) search shipped later (`crates/cli/src/semantic.rs`,
see "Semantic search" below) as exactly that: opt-in, not a replacement for
FTS5/symbol lookup — both still run, and semantic hits are added as a third
source when configured. A vector database was still skipped even then:
cosine similarity over an in-process `Vec<f32>` scan is fast enough at
single-repo scale (see below) and is one less moving part to run locally.

### Schema

```sql
files(id, path UNIQUE, lang, size, mtime, hash)
symbols(id, file_id -> files, kind, name, start_line, end_line, parent)
content_fts(path, start_line, end_line, kind, name, body)   -- FTS5 virtual table
embeddings(chunk_id -> content_fts.rowid, model, dims, vector BLOB)  -- optional, see "Semantic search"
```

`hash` (a fast non-cryptographic hash of file content) makes `grv index`
incremental: unchanged files are skipped entirely, so re-indexing after a
small edit only touches the files that changed.

### Symbol extraction is best-effort, search is not

`indexer::extract_symbols` runs a tree-sitter query per language (function/
struct/class/impl/interface definitions) and silently returns nothing if the
grammar version's node names don't match the query — logged as a warning,
never a hard failure. This matters because tree-sitter grammar crates version
independently of `tree-sitter` core and node-kind names do shift between
grammar releases.

Deliberately, **full-text search never depends on this working.**
`chunk_lines` in `indexer/src/lib.rs` always splits every text file into
150-line/30-line-overlap windows and indexes those into `content_fts`,
regardless of language or parse success. So a query grammar mismatch
degrades `grv symbol` precision, never `grv search`/`grv ask` recall.

### Call graph (`grv callers`/`grv callees`, `crates/cli/src/callgraph.rs`)

A second tree-sitter query per language, same graceful-degradation contract
as the definition query above (`Lang::call_query_src`/`compile_call_query`,
`indexer::extract_calls`, `CallSite { line, callee_name }`) — written and
verified for 48 of the 53 parsed languages (all except GraphQL and
Protobuf, which have no "function call" concept at all — schema
definition languages, nothing to extract); a language with no call query
just yields no call edges, never a hard failure. Every match must expose a
`@call` capture (the whole call expression, for its line) and a `@callee`
capture (the called name's text) — e.g. Rust's query also matches
`scoped_identifier` calls (`Type::method()`) and macro invocations, not
just bare identifiers.

A few languages' call queries lean on the same text-predicate technique as
their def queries (see "Text-predicate-based queries" below) to stay a
*call* graph rather than accidentally re-surfacing definitions or special
forms as if something "called" them:

- **Elixir**: excludes `def`/`defmodule`/`if`/`case`/etc. via
  `#not-any-of?` — without it, `def foo do ... end` would show up as a
  call to something named `def`, since it's structurally the same `call`
  node a real function call is.
- **Racket/Scheme**: excludes `define`/`lambda`/`if`/`let`/etc. the same
  way, for the same generic-S-expression reason as their def queries.
- **Assembly**: has no dedicated "call" node — a `call`/`jmp`/`jNN`
  instruction's operand looks identical to any other instruction's operand
  at the grammar level. `#any-of?` on the mnemonic is what turns "every
  instruction operand" into "control-flow targets" — deliberately
  including conditional jumps alongside `call`, since a jump-target graph
  is the closest thing assembly has to "what does this call".

Verilog's call query covers both `system_tf_call` (builtin
`$display`/`$finish`/...) and `tf_call` (a plain user-defined task/function
call, e.g. `y = bar(1);`) — the latter was initially left out after a
hand-written sample tripped an unrelated grammar quirk (a `task`
declaration's own syntax, not the call site) and produced an `ERROR` node;
a second, narrower sample isolating just the call site parsed cleanly and
revealed the real shape (`(tf_call (simple_identifier) @callee)`).

Deliberately name-based, not type-resolved: a new `calls` table
(`file_id`, `caller_symbol_id` nullable, `callee_name`, `line`) stores the
callee as plain text, so `grv callers run` matches every call site
anywhere literally named `run(...)`, regardless of which `run` it actually
is at that scope. `caller_symbol_id` (which symbol's body the call site
falls inside, or `NULL` for module-level code) is resolved at index time by
finding the *smallest* still-open symbol span containing the call's line —
capturing already-inserted `(id, start_line, end_line)` triples for the
file being indexed lets `index_repo` do this with one linear scan per file,
no extra query. Full type/scope resolution would need real semantic
analysis per language (a much larger undertaking, on par with what a
language server spends its whole existence on) for a precision gain that
matters more to an automated refactoring tool than to a developer reading
`grv callers`' output — the same tradeoff `grv symbol`'s `LIKE`-based name
matching already makes.

**`ResolutionHint`** is the honest middle ground: not real resolution, but
a real signal beyond the bare name match, computed from one extra query
(`SELECT path, kind, name, parent, start_line FROM symbols JOIN files
WHERE name = ?`, capped at `MAX_DEFINITIONS_CONSIDERED` = 50, once per
`find_callers` call, not once per hit) rather than per-row lookups.
Comparing that list of real definitions against each hit's own file gives
four outcomes, and — unlike the first version of this design — each
variant now *carries the actual candidate data* (`DefinitionRef { path,
kind, name, parent, line }`) instead of being a bare label:

- `LikelySameFile(Vec<DefinitionRef>)` — one or more same-named
  definitions live in the call site's own file. When there's exactly one,
  that's still the strongest signal available without real scope
  resolution (deliberately shadowing a name across modules is the rare
  case, not the common one). When there's more than one — e.g. two
  `impl` blocks each defining `new` in the same file — both are listed,
  distinguished by `parent` (the enclosing `impl`/`class`, populated at
  index time from `ExtractedSymbol::parent` via the same
  smallest-enclosing-span technique `caller_symbol_id` uses), instead of
  collapsing into one confident-sounding label that would have hidden the
  very ambiguity it was supposed to flag.
- `UniqueElsewhere(DefinitionRef)` — exactly one definition exists
  anywhere, just not in this file — still unambiguous, only not local.
- `ImportResolved(DefinitionRef)` — the call site's own file has a real,
  resolved `use`/`import` naming exactly this definition's file (see
  "Import resolution" below) — genuine resolution via an actual import
  statement, not a co-location heuristic. Only produced for the languages
  with an import resolver (54 languages as of this writing -- see "Import
  resolution" below for the full list).
- `Ambiguous(Vec<DefinitionRef>)` — multiple same-named definitions exist,
  none local, and either this language has no import resolver yet or none
  of its resolved imports narrowed things down to one candidate. Rather
  than stop at "ambiguous" and leave a human to grep the name back up
  themselves, every remaining real candidate's file and enclosing scope is
  listed (narrowed to just the import-corroborated ones when at least one
  matched, even without narrowing all the way to a single answer), so a
  human's own knowledge of the codebase's imports has real material to
  finish the disambiguation with.
- `NoDefinitionIndexed` — no definition found anywhere in the index. This
  does *not* mean "this function doesn't exist" — only "not in what `grv
  index` walked" (a stdlib/external-crate call, a trait method whose
  implementor lives outside the repo, or dynamic dispatch). The CLI
  renderer (`format_resolution_hint` in `crates/cli/src/main.rs`) states
  this explicitly (`[not indexed anywhere -- external/stdlib call,
  dynamic dispatch, or just not part of this repo]`) rather than printing
  nothing, which the first version of this design did — a bare
  unlabeled hit could otherwise be misread as "resolved cleanly."

Covered by real unit tests against a synthetic in-memory index
(`crates/cli/src/callgraph.rs`'s `tests` module) exercising all four
outcomes, including one written specifically to prove the same-file
scope fix (`likely_same_file_lists_every_candidate_when_the_same_file_defines_it_twice`:
two `impl` blocks' `new()` in one file, asserting both surface as distinct
candidates rather than one blanket label) — plus dogfooded against this
repo's own index: `grv callers new` shows `crates/cli/src/checkpoint.rs`'s
real `Session::new`/`MissionCheckpoint::new` collision as `[this file
defines it 2x, still ambiguous locally: Session::new (line 53),
MissionCheckpoint::new (line 200)]` instead of a blanket "likely" label,
and `grv callers println` shows `[not indexed anywhere -- external/stdlib
call, dynamic dispatch, or just not part of this repo]` for every call
site of the (external, `std`) `println!` macro.

### Import resolution (`crates/indexer/src/imports.rs`/`resolve.rs`)

A real, per-language import resolver — not a heuristic guess dressed up as
one, and not a symbol table or type checker either. It runs in two
deliberately separate passes:

1. **Extraction** (`imports.rs`, per file, during the same walk that
   extracts symbols/calls): a bespoke AST walker per language, not a
   `def`/`call`-style flat query — import syntax nests and aliases too
   much for one capture pattern (`use a::{b::{c, d as e}, f::*}` is a
   recursive tree of brace lists, renames, and glob markers). Produces
   `ImportEdge { raw_path, imported_name, is_wildcard, module_prefix,
   line }` rows into a new `imports` table. `imported_name` is `None` for
   a whole-module bind (Python `import os`, a JS default/namespace
   import); `is_wildcard` is true only for a *real* glob (`use foo::*`,
   `from foo import *`) or any Go import (Go has no per-name import syntax
   at all — a package import makes every exported name reachable via
   `pkg.Name()`, which this project's call queries do capture, selector
   name only) — the distinction matters for step 3 below.
2. **Resolution** (`resolve.rs`, one pass over the whole repo, run once
   after every file has been (re)indexed): turns a raw edge into an actual
   file — or files, for Go, since an import there names a whole package
   directory, which can span several — via a real per-language algorithm,
   never a fuzzy match:
   - **Rust**: `crate::`/`self::`/`super::` resolve against the owning
     crate's module tree (crates discovered from every `Cargo.toml` in the
     repo, `[package] name` mapped to its `src/` directory); a leading
     segment matching another discovered crate's name resolves
     cross-crate. Assumes the standard `cargo new` layout (module path
     segments = directory/file segments) — `#[path = "..."]` isn't
     modeled. A first segment matching neither `crate`/`self`/`super` nor
     a known local crate is external (std/a dependency) and stays
     unresolved, correctly.
   - **Python**: relative imports (`from . import x`, `from ..pkg import
     y`) resolve unambiguously against the importing file's own directory.
     Absolute imports (`import a.b.c`) are tried against the repo root and
     every one of its immediate subdirectories as candidate source roots —
     a bounded heuristic, not a real `sys.path`.
   - **JavaScript/TypeScript/TSX**: only relative imports (`./x`, `../x`)
     resolve, directly against the importing file's directory, tried
     against real extensions and `index.*`. Bare specifiers are external
     packages in practice and stay unresolved; `tsconfig.json` path
     aliases are a named, real gap, not modeled.
   - **Go**: resolves against the owning module's declared path (`go.mod`'s
     `module` line) plus the directory tree; an import can legitimately
     resolve to several files (every `.go` file in the target package
     directory), unlike the single-file resolution the other three give.

   An import that resolves to nothing means exactly that: external
   dependency, stdlib, or a shape this heuristic doesn't cover — never a
   wrong guess presented as a real one.

**25 more languages** got real import extraction + resolution in a
follow-up batch, after the first six proved the two-pass design out. Import
syntax for these is structurally simpler than Rust's brace-list nesting —
almost always one dedicated grammar node, no aliasing tree to walk — so
extraction here is `tree-sitter` queries (`@import`+`@target` captures,
mirroring `def_query_src`/`call_query_src`'s own style) rather than bespoke
walkers, grouped by mechanism instead of duplicated 25 times:

- **A dedicated grammar node exists, unambiguous, no predicate needed**
  (`crates/indexer/src/imports.rs`'s `query_based` module): C/C++/
  Objective-C/GLSL/HLSL (`preproc_include` — `#include "x"` resolved,
  `#include <x>` skipped at extraction time since a system header is
  never repo-local, never even recorded as a doomed edge), Vim
  (`source_statement`), Proto (`import`), Solidity (`import_directive`),
  Verilog (`` `include ``), Nix (`import ./x` — an ordinary builtin call,
  checked by text since Nix has no dedicated import node, but the
  argument's `path_expression` node is unambiguous), and the hierarchical/
  dotted-module languages Java, Kotlin, Groovy, Scala, C#, Elm (see below).
- **An ordinary command/function call at the grammar level** (`imports.rs`'s
  `command_style` module): Bash (`source`/`.`), Fish (`source`), Ruby
  (`require`/`require_relative`/`load`), R (`source(...)`), Racket
  (`(require "path")` — only the string form; `(require racket/base)`'s
  bare-symbol form names a library collection, not a repo file, and is
  correctly left unextracted), CMake (`include()`/`add_subdirectory()`,
  case-folded since CMake command names are case-insensitive). Filtering
  to the right command name happens as a plain Rust string comparison
  *after* the query runs, deliberately not a `#eq?`/`#any-of?` text
  predicate — this project has already been burned once by a predicate
  silently not applying (see the query-safety-net story in "Language
  coverage" below); for these lower-traffic languages, a straightforward
  `==`/`matches!` check in Rust is the safer bet, not just a shorter one.

Resolution reuses two generic functions rather than one per language:

- **`resolve_relative_literal`**: used by every quoted/relative-path
  language above (C-family, Vim, Proto, Solidity, Verilog, Nix, Bash,
  Fish, Ruby, R, Racket, CMake). Resolves against the importing file's own
  directory *regardless of a leading `./`* — correct C/C++ `#include "x"`
  search-path semantics (the including file's own directory is checked
  first even without one), and harmless for the others: a bare name that
  doesn't happen to match a real repo-relative file just stays unresolved.
  An optional extension list lets a language that can omit the extension
  (Bash's `source helper` meaning `helper.sh`) still resolve.
- **`resolve_dotted_module`**: used by Java, Kotlin, Groovy, Scala (wildcard
  marker `_` instead of `*`), and C# (no wildcard, and namespaces don't
  reliably mirror directories the way Java's package convention does, so
  lower-confidence). `a.b.C` is tried as `<root>/a/b/C.<ext>` against a
  candidate root list — repo root, a handful of real, *present-in-this-repo*
  conventional Maven/Gradle source roots (`src/main/java`, `src/main/kotlin`,
  ...), and every immediate subdirectory as a generic fallback. `a.b.*`
  (`a.b._` for Scala) is a whole-package wildcard, resolved to every
  same-extension file in that directory — the same multi-file honesty as
  Go's package resolution. Elm's `exposing (...)` list is captured
  precisely (each exposed name becomes its own `imported_name`, same as
  Python's `from X import a, b`), with `exposing (..)` treated as the
  wildcard case.

**D/Haskell/Julia's whole-module-unqualified-exposure gap, since solved**:
a plain (non-`qualified`) import in these three brings a whole module's
exports into *unqualified* scope — semantically closer to a wildcard than
to Java's "import one class". The original mistake would have been reusing
`resolve_dotted_module`'s "last segment is the imported name" assumption
as-is, which — for a plain, unqualified import — produces a confidently
wrong signal (the module's own name is rarely a callable). The actual fix
was a `wildcard_is_package_directory: bool` parameter on
`resolve_dotted_module` itself, not a bespoke resolver per language: for a
*package* language (Java/Kotlin/Groovy/Scala, where `a.b` genuinely names
a directory that can hold many files), a wildcard resolves to every
same-extension file in that directory via `files_by_dir` — the existing
behavior. For a *module* language (Haskell/D/Julia/Elm — one module maps
to exactly one file, full stop, there is no package directory), a
wildcard instead resolves to that SAME single file a plain import of it
would, via the identical `<root>/a/b/C.<ext>` lookup — looking up `a/b` as
a directory would be wrong here (it usually doesn't even exist, and a
module's sibling files in the same directory are never part of what its
wildcard exposes). Extraction still had to distinguish each language's own
qualified/selective/hiding forms precisely (Haskell's `qualified` keyword
and `hiding` clause, D's `import_bind`/`imported`-wrapper shape verified
via a real parse-tree dump, Julia's `using` vs `import` and `selected_import`
clause) rather than guess — see `haskell_import`/`d_import`/`julia_import`'s
own doc comments in `imports.rs` for the exact per-language rules.

**A retroactively-caught regression, found while designing the fix
above, not user-reported**: Elm's own `exposing (..)` had been wired as a
Java-style package wildcard (a directory listing) before this batch,
which is exactly the same mistake D/Haskell/Julia would have made —
Elm modules are one-per-file too. Fixed the same way, by passing
`wildcard_is_package_directory: false` for Elm as well and removing the
now-unneeded `elm_files_by_dir` directory index; locked in by
`elm_exposing_all_resolves_to_its_own_file_only_not_a_directory_listing`.

**14 more languages** got real import extraction + resolution in the
follow-up batch after that: Erlang, Zig, PHP, LaTeX, Dart, Scheme, and
PowerShell are quoted/relative-literal paths, resolved the same way as the
C-family group above via `resolve_relative_literal` (PowerShell's fix
doubled as a real extraction bug fix — see below). Lua's `require("a.b")`
reuses `resolve_dotted_module` with its wildcard form disabled (Lua's
`package.path` convention has no wildcard). Ada, OCaml, Perl, Fortran, and
Elixir each needed a genuinely different small dedicated resolver, since
none of their naming conventions fit either generic function:
`resolve_ada` (GNAT's convention: the *whole* dotted name is dash-joined
and lowercased into one flat filename — `My_Pkg.Child` → `my_pkg-child.ads`
— not slash-nested into a directory the way Java's is), `resolve_ocaml`
(only the outermost dotted segment is a real compilation unit —
`open Base.List` resolves via `base.ml`'s lowercased name only; a
sub-module named after the first dot lives inside that same file and is
honestly left unresolved-past-the-first-segment rather than guessed at),
`resolve_perl` (CPAN's `::` → `/` convention under `lib/`, structurally
identical to `resolve_dotted_module` but keyed on a two-character
separator that function's `split('.')` can't express), `resolve_fortran_module`
(a flat filename guess against a handful of real extensions — Fortran, unlike
every other language here, has no standardized module-to-file convention
at all, so `include`'s real relative-literal path is tried first and this
is only a fallback), and `resolve_elixir` (Mix's CamelCase-per-segment →
snake_case-per-segment convention under `lib/`, via a small `camel_to_snake`
helper that only inserts an underscore at a real word boundary so an
acronym like `HTTPServer` still splits as `http_server`, not
`h_t_t_p_server`).

Two real extraction bugs surfaced and were fixed during this batch, both
found by checking a real parse-tree dump rather than trusting the first
plausible-looking grammar shape:

- **Elixir**: `find_nodes` stops descending once it finds a match of the
  target kind — correct for `use_declaration`/`import_statement`, which
  never nest inside themselves, but wrong for Elixir's `call` node, since
  `defmodule Foo do ... end` is *itself* a `call` (target `defmodule`)
  whose `do_block` can contain more `call`s for `alias`/`import`/`require`/
  `use`. The outer match was found and recursion stopped right there,
  silently missing every import inside every module. Fixed with a new
  `find_nodes_nested` helper that keeps descending into a match instead of
  stopping at it, used only where nesting is real (Elixir).
- **PowerShell**: the `Import-Module ./x` extraction path picked the FIRST
  named child of `command_elements` as the argument — which is actually
  a `command_argument_sep` separator token, not the `generic_token` path
  that follows it (confirmed via a real parse-tree dump). Fixed to skip
  separator tokens when looking for the argument. The dot-source form
  (`. .\file.ps1`) was unaffected — it reads a different field.

**3 languages got real extraction + resolution** via straightforward
resolvers: Assembly's `.include "x"` (GAS/MASM's own dedicated `meta`
grammar node) and NASM's `%include "x"` / bare `INCLUDE "x"` (this
grammar's `%`-preprocessor support is broken — a real parse-tree dump
confirmed the leading `%` becomes an `ERROR` node — but the `include "x"`
that follows still parses as an ordinary `instruction` node with kind text
`"include"` and one string operand, matched by that text alone since
"include" is never a real instruction mnemonic in any assembly dialect);
both resolved via `resolve_relative_literal`, same as C's `#include`.
Swift's `import CoreModule` (verified via `node-types.json`: a bare
`identifier` child, no `path` field at all) is always treated as a
wildcard, resolved via a new `resolve_swift_module` to every `.swift` file
under `Sources/CoreModule/` — Swift Package Manager's own convention for a
sibling target in the same multi-target package; unlike every other
module language here a Swift target is a whole *subtree*, not one file or
one flat directory, so this scans by path prefix rather than reusing
`resolve_dotted_module`/`files_by_dir`. An external framework
(`Foundation`, `UIKit`, ...) simply has no matching directory and stays
honestly unresolved, same as every other language's external-dependency
imports. Terraform/HCL's `module "x" { source = "./y" }` needed its own
hand-walked extraction (`hcl_module_source`) since `block`/`attribute` are
fully generic, field-less, positional nodes in this grammar (verified
against real node-types.json) — matched by walking the block's first
`identifier` child for the `module` keyword, then its `body` for an
`attribute` named `source`; only a literal string starting with `./`/
`../` is extracted, a registry reference (`terraform-aws-modules/vpc/aws`)
or git URL being unambiguously external and left unextracted. Resolved via
a new `resolve_hcl_module` to every `.tf` file in the referenced
directory — the same multi-file honesty Go's package imports use.

**The remaining 8 languages this project had previously accepted as
gaps — Nim, VHDL, Prolog, Crystal, GraphQL, WGSL, Svelte, Vue — got a
second, harder look, and 7 of the 8 turned out to be genuinely fixable,
not fundamental limitations:**

- **Crystal**: the crates.io `0.1.0` grammar (owned by a brand-new,
  zero-star account, `undivisible`, that batch-published several other
  grammar crates the same day — the same dependency-confusion pattern
  this project already treats with suspicion for the Vue ecosystem)
  confirmed via a real parse-tree dump to not parse `require "..."` as a
  call/macro-invocation node at all. `crystal-lang-tools` (Crystal's own
  official tooling org — the `crystalline` LSP lives there too) maintains
  a real grammar with a dedicated `require` node. Its Cargo dependency
  fetch repeatedly hit a transient SSL/network error in this sandbox
  specifically (a plain `git clone` of the same commit worked fine in
  well under a minute) — rather than fight that, the already-cloned
  source was vendored directly (`vendor/tree-sitter-crystal/`, see its
  `NOTICE.md` for the full provenance chain). Swapping the grammar broke
  the *existing* `def_query_src`/`call_query_src` (`method_definition`/
  `class_declaration`/`(call name: ...)` → `method_def`/`class_def`/
  `(call method: ...)`, verified via another real dump) — a real,
  expected cost of a grammar swap, fixed the same turn.
- **Nim**: same root cause as Crystal — the only crates.io release has no
  import-related node type in its grammar at all. `alaviss/tree-sitter-nim`
  (an actively maintained, richer community grammar) DOES have real
  `import_statement`/`import_from_statement`/`include_statement` nodes,
  verified directly. Its Rust binding pins `tree-sitter = "~0.25"`, which
  conflicts (`links = "tree-sitter"`, only one version allowed per binary)
  with this workspace's `tree-sitter = "0.26"` — confirmed as a real,
  hard build error, not assumed. Relaxing that one version constraint
  (nothing else) was verified to parse real Nim source correctly against
  0.26's actual runtime — the C-level ABI doesn't care which Rust crate
  version generated the binding glue. The user was asked how to host this
  one-line-patched fork (a `git` dependency to a pushed copy, matching the
  `tree-sitter-vuejs` precedent, needs a repository this session had no
  credentials to create) and chose to vendor it directly
  (`vendor/tree-sitter-nim/`, ~41MB — see its `NOTICE.md`). Its call query
  needed the same kind of fix as Crystal's (`name` field → `function`
  field, verified via a dump).
- **VHDL**: previously declined as "no reliable package-to-file naming
  convention exists" — true in general, but `use work.my_pkg.all` is a
  real exception: `work` is VHDL's own unambiguous "compiled from this
  repo" marker (as opposed to `ieee.std_logic_1164` or any other named
  library, always external). `vhdl_use` extracts only the `work`-scoped
  case; `resolve_vhdl_unit` is a flat filename guess against `.vhd`/`.vhdl`
  extensions, the same honest category of guess Fortran's resolver
  already makes.
- **Prolog**: previously declined as "directive shape too uncertain" — a
  real parse-tree dump resolved that uncertainty directly: `clause`'s
  grammar has no dedicated "directive" node at all, but `:-` (as opposed
  to `?-`, a query) is simply the first, anonymous child of a
  `unary_operation`, whose own `kind()` is literally the operator's text
  — checkable with a plain string comparison. `prolog_directive` extracts
  `use_module`/`consult`/`ensure_loaded`'s quoted-atom argument and the
  `[File]` shorthand; `library(Name)` arguments are correctly left
  unextracted (external, not a repo file).
- **GraphQL**: the base spec genuinely has no import syntax (confirmed:
  no node type name in its real `node-types.json` contains "import" or
  "include") — but the `graphql-import`/GraphQL Modules community
  convention (`# import Foo from './other.graphql'`, a specially-shaped
  line *comment*) is real and in active use. `graphql_import_comment`
  matches it via plain text pattern matching over `comment` nodes, since
  there is no grammar shape to hook a query onto — opt-in by construction:
  a GraphQL file that doesn't use this convention just produces nothing.
- **WGSL**: same "no import in the base spec" fact, but this project's
  already-linked grammar isn't the base spec's — `tree-sitter-wgsl-bevy`
  was chosen specifically because it's Bevy-flavored, and Bevy's naga_oil
  shader preprocessor extends WGSL with a REAL `#import a::b` directive
  that this grammar models as a dedicated `preproc_import` node,
  discovered by actually checking rather than assuming the base spec's
  answer still applied to a fork. Extracted via `wgsl_preproc_import` --
  but deliberately given NO resolver: naga_oil modules are registered
  under an arbitrary Rust-side name (`load_internal_asset!`/asset
  registration), not a fixed file-per-module convention, so there's
  nothing reliable to resolve against. This is the one case, out of all
  8, that stayed a genuine gap on the resolution side even after the
  investigation — but the extraction itself (real, useful signal even
  unresolved) was still worth doing, the same way Rust's external-crate
  imports are extracted and correctly never resolved.
- **Svelte/Vue**: the real imports inside a `<script>` block are ordinary
  JS/TS, but both grammars parse that block's entire body as one opaque
  `raw_text` node (the injection limitation already known from their
  missing `def_query_src`). `script_injected_imports` does a real second
  parse pass: slices out the `<script>` element's own source text,
  re-parses it with the JS or TS grammar (`lang="ts"`/`lang="typescript"`
  picks TS, checked against the real `attribute` shape both grammars
  share), reuses the EXISTING `js::imports` walker on that second tree (no
  duplicated logic), and shifts every resulting line number by the script
  block's own starting line.

Verified per-language against real samples (`imports.rs`'s `tests` module
— 46 tests, one or more per language/family), plus end-to-end resolution
integration tests through `index_repo` for a representative case of each
resolver mechanism (`lib.rs`'s `index_repo_tests`, 173 tests total in that
crate: a real C `#include` resolving to a local header while a system
header stays unrecorded, a real cross-file Java import resolving via a
conventional Maven source root, a Java wildcard import resolving to every
file in its package directory, the D/Haskell/Julia/Elm
`wildcard_is_package_directory` fix, one representative case each for Ada,
OCaml, Perl, Fortran, Elixir, Lua, Dart, Scheme, PowerShell, Assembly,
Swift, and Terraform, and a final case each for Nim, VHDL, Prolog,
GraphQL, Crystal, and Svelte's script injection) — plus a live dogfood
against a real synthetic multi-language scratch repo covering all 8
newly-fixed languages at once through the actual `grv` binary (Nim's
`import pkg/helper`, Crystal's `require "./helper"`, VHDL's `use
work.my_pkg.all`, Prolog's `consult`, GraphQL's import-comment
convention, and Svelte's script-injected import all resolved correctly;
WGSL's naga_oil import extracted and correctly stayed unresolved).

A real bug this project's own dogfooding against its own repo caught
(and a useful example of why "run it on yourself" stays part of this
project's verification discipline, not just unit tests in isolation):
`use super::glob_match;` inside `crates/cli/src/permissions.rs`'s
`#[cfg(test)] mod tests { ... }` block — an *inline* module, not a
separate file — was resolved as if the whole file were one flat module,
so `super` jumped straight past `permissions.rs`'s own top level to the
crate root (`main.rs`), a confidently WRONG cross-file answer, not an
honest "don't know". Fixed by tracking `module_prefix` — the names of
every inline `mod name { ... }` block an import is textually nested
inside (found by walking up the AST from the `use_declaration`, collecting
`mod_item` ancestors — a `mod name;` with no body can never be an
ancestor of anything, so every `mod_item` found this way is inherently
inline) — and folding it into the effective module path before applying
`self`/`super`. Verified live: `grv index .` on this repo, then a direct
query, now shows `super::glob_match` resolving to
`crates/cli/src/permissions.rs` itself; a dedicated regression test
(`super_inside_an_inline_test_module_resolves_back_to_its_own_file_not_the_crate_root`
in `crates/indexer/src/lib.rs`) locks this in.

`calls` rows are cleaned up the same way `symbols`/`embeddings` rows are:
`ON DELETE CASCADE` from `files`, so a file re-index (which deletes and
re-inserts that file's `files` row) or `clear_index` never leaves stale
call edges pointing at chunks/symbols that no longer exist.

### Watch mode (`grv index --watch`, `crates/cli/src/watch.rs`)

Real filesystem events via the `notify` crate's recommended backend
(inotify on Linux, kqueue on BSD/macOS, ReadDirectoryChangesW on Windows) —
not polling. `notify` is the one new runtime dependency added for this:
reimplementing cross-platform FS event watching by hand would mean three
different syscall interfaces to get right, not a reasonable "just hand-roll
it" case the way e.g. `content_hash`'s `DefaultHasher` use was.

Events are debounced (600ms): a single save fires several raw events
(write, rename, metadata-change) in quick succession, and a `git checkout`/
branch switch touches many files at once — `watch::watch` drains whatever
else arrives within the debounce window after the first relevant event, so
either becomes one `index_repo` call, not one per raw event. `index_repo`
is already incremental (unchanged-file hash skip), so re-running it on the
whole tree after a burst is correct and cheap, not a full rebuild.

Events under a skipped directory (`indexer::SKIP_DIRS`, now `pub` so
`watch.rs` can reuse the exact list `index_repo`'s walker filters on — a
second hand-copied list would drift) or the configured `index_dir` are
filtered out before triggering anything; without this, the index.db's own
WAL writes during re-indexing would trigger re-indexing itself in a loop.

**A real, general bug fix that fell out of building this**: `index_repo`
never previously removed rows for files deleted since the last index —
`grv search`/`grv symbol`/`grv callers` could keep surfacing a file that no
longer exists on disk, forever, until a full `--force` re-index. This
mattered more once unattended watch-mode re-indexing was going to run for
arbitrary lengths of time, so it's fixed at the source: `index_repo` now
tracks every path actually seen on disk during its walk (`seen_paths`),
and after the walk, deletes `files`/`content_fts`/`embeddings` rows (cascading to `symbols`/`calls`) for any
previously-indexed path *not* in that set. Reported as `IndexStats::files_removed`,
verified with a real `rm` under a live `grv index --watch` process (not
just at compile time) during development.

### Context assembly (`crates/cli/src/context.rs`)

For `ask`/`investigate`, three retrieval passes feed one budgeted assembly:

1. `--file` arguments — read in full (capped at 20KB each), highest priority.
2. Symbol name matches against tokens extracted from the question (`LIKE
   '%token%'` on `symbols.name`, ≥3 chars) — precise, pulls exact function/
   class source via stored line ranges.
3. FTS5 `MATCH` over the question's tokens, top 12 chunks by bm25 rank —
   broader recall for anything symbol matching missed.

`Config::context_char_budget()` reserves `context_budget_fraction` (default
55%) of `num_ctx` for this injected context — chars/4 as a token-count
approximation — leaving the rest of the window for the system prompt, the
question, and the model's own output. `context::assemble` adds blocks
greedily in priority order and simply stops once the budget would be
exceeded, rather than truncating mid-block.

This budget is deliberately conservative and adjustable: raise
`context_budget_fraction` or `num_ctx` via `grv config` once you've watched
`free -h` during a session and confirmed headroom.

### Ollama client (`crates/llm`)

Plain `reqwest` POST to `{host}/api/chat` with `"stream": true`; Ollama
returns newline-delimited JSON objects, each with an incremental
`message.content` piece and a `done` flag. `chat_stream` parses these
line-by-line off a `bytes_stream()` and invokes a caller-supplied closure
per token — that's what makes `grv ask` print live instead of blocking
until the full answer is generated, important on CPU-bound inference where
a full 8B response can take tens of seconds.

`options.num_ctx` is passed explicitly per-request rather than relying on
whatever the model's default is, so `grv config --num-ctx N` actually takes
effect.

## Sizing guidance (why qwen3:8b / Q4)

| Model | Approx RAM (Q4_K_M) | Fits in 16GB alongside OS+GRAVITON? |
|---|---|---|
| 3-4B | ~2.5-3GB | Yes, easily — pick this if you want speed over depth |
| 8B | ~5-6GB | Yes, comfortably — best reasoning/code quality for this RAM budget |
| 14B | ~9-10GB | Marginal — workable if nothing else heavy is running, watch swap |
| 30B+ | ~18GB+ | No — will swap, unusable latency |

The P620's 4GB VRAM is used by Ollama as a partial offload accelerator when
available; it is not a constraint you need to size the model against, RAM
is. Verify actual GPU usage during inference with `watch -n1 nvidia-smi` —
if `GPU-Util` stays at 0% while a model runs, Ollama fell back to full-CPU
for that model (common for the P620's older Pascal arch depending on the
Ollama build's compute-capability floor); this doesn't change which model
size to pick, since RAM was always the binding constraint here anyway.

## Language coverage (v0.22)

Each parsed language is one tree-sitter grammar crate + one `def_query_src`
entry in `crates/indexer/src/lang.rs`. Adding a language is: find its
grammar on crates.io, pull its `node-types.json` to find the right node/field
names for function/class/method definitions, write the query, and let the
existing "best-effort, never fatal" machinery handle version drift.
`ALL_LANGS` (a hand-maintained `&[Lang]` const, tied to the enum by a
compile-time-exhaustive match — see "The query safety net" below) is the
canonical list of every recognized language; `grv languages` prints it
split into the same tiers as here, plus a separate "import resolution"
line driven by `imports::has_import_resolver` — its own independent
exhaustive match (same discipline as `ALL_LANGS` itself, for the same
reason: a language can gain/lose import support on a schedule completely
unrelated to its symbol-extraction tier, and a hand-maintained list here
could silently drift from `extract_imports`/`resolve_all_imports`'s real
dispatch otherwise).

GRAVITON recognizes **66 languages total**, split into three honest tiers —
"honest" meaning the tier a language sits in reflects what was actually
verified, not what merely compiles:

- **53 with a verified `def_query_src`** — the grammar is linked AND its
  query was checked against a real sample file's `extract_symbols()` output,
  not just compiled: the original 16 (Rust, Python, JavaScript, TypeScript,
  TSX, C, C++, Go, Java, C#, PHP, Ruby, Bash, Lua, Solidity, PowerShell),
  10 added for the "at least 60 languages" push (Haskell, Fish, Dart, Zig,
  Julia, Groovy, GraphQL, Crystal, D, and assembly — assembly's "symbol" is
  a label, the closest thing the format has to one), 24 more added in a
  follow-up pass (Elixir, Scala, Swift, Perl, R, OCaml, Elm, Nim, Erlang,
  Vim, Nix, HCL/Terraform, CMake, Verilog, VHDL, Fortran, Prolog, Racket,
  Scheme, Protobuf, Objective-C, GLSL, HLSL, Ada — see "Text-predicate-based
  queries" below for the subset of these that needed more than a
  structural query), and finally Kotlin, LaTeX, and WGSL once each got a
  real working grammar (see "Previously-blocked grammars" below).
- **2 with a real, linked, parseable grammar but no `def_query_src`** —
  Svelte and Vue. Both grammars parse a `<script>` block's entire body as
  one opaque `raw_text` node; the actual function/variable definitions
  inside it are real JS/TS, but recovering them needs a second,
  "language injection" parse pass (what a real editor's own host
  application drives via a separate `.scm` query) that this project's
  one-query-per-language *symbol-extraction* design doesn't do. Import
  extraction is a narrower problem than full symbol extraction (one
  `<script>`-block substring, one JS/TS parse, one existing walker reused
  unchanged, versus reconciling injected spans back into `extract_symbols`'
  whole-file line/scope bookkeeping) and DID get exactly this treatment —
  see "Import resolution" above's `script_injected_imports` — without
  this gap being closed for symbols too; the two are genuinely
  different-sized undertakings, not an inconsistency. Still real progress over
  having no grammar at all — both are now genuinely parsed, just not in a
  `grv symbol`-shaped way.
- **11 tagged only, no grammar** — HTML, CSS, JSON, YAML, TOML, XML,
  Markdown, SQL, Dockerfile, INI, Makefile — markup/data/config formats
  with no tree-sitter grammar attempted (not "blocked", just never
  applicable the way a programming language's function/class concept is).

Two real constraints surfaced while adding the original v0.2 batch (Java,
C#, PHP, Ruby, Bash, Lua, Solidity, PowerShell), and both recurred, harder,
in the later, much bigger language pushes:

- **Node/field names must be looked up per grammar, not assumed.** They
  don't follow one convention: Ruby's `class`/`module` nodes *do* expose a
  `name:` field (unlike what a naive guess suggests), while PowerShell's
  grammar exposes the name as a bare positional child with no field label
  at all (`(class_declaration (type_identifier) @name)` instead of
  `(class_declaration name: (type_identifier) @name)`). Get this wrong and
  the query still compiles — it just silently matches nothing, which is
  why every language in the 53-language tier was verified two ways: real
  `node-types.json` inspection plus a real parse (`to_sexp()` dump) of a
  hand-written sample file in each language, then a permanent test
  (`crates/indexer/src/lang.rs`'s `new_language_queries` module) asserting
  `extract_symbols()` returns the expected names from that sample — not
  "it compiled". "Compiles and matches nothing" isn't even the worst
  failure mode a wrong field guess can produce — see "The query safety
  net" below for a *different* mismatch (a field of the wrong *type*) that
  fails to compile the entire query, silently zeroing out every def kind
  in it, not just the wrong one.
- **A `links` conflict, or a subtler type/symbol mismatch, can make a
  grammar crate unusable however new the code around it is** — see
  "Previously-blocked grammars" for the specifics and how each was
  eventually resolved (all five were, in the end — none of the languages
  named in past versions of this doc as "blocked" still are).

### Previously-blocked grammars — Kotlin, Svelte, Vue, WGSL, LaTeX

All five used to be flatly unlinkable. All five now have a real, linked,
tested grammar, via actively-maintained forks rather than the original
crates.io releases — the *reasons* they were blocked are still worth
recording, because the failure modes are exactly what to expect from any
future language whose ecosystem hasn't caught up to a newer tree-sitter
core:

- **The real fix, once found, generalizes**: modern tree-sitter grammar
  crates (roughly 2023+) don't depend on the full `tree-sitter` crate at
  runtime at all — they depend on a tiny, ABI-stable `tree-sitter-language`
  shim crate (currently "0.1", rarely needing to bump) for the `LanguageFn`
  type their `LANGUAGE` const is built from, and only pull in the real
  `tree-sitter` crate as a *dev-dependency* for their own internal tests.
  That shim is what lets every one of this project's 60+ grammar crates —
  pinning wildly different tree-sitter versions in their own `Cargo.toml`
  metadata — link together against this workspace's actual tree-sitter
  0.26 with zero conflicts: none of them actually depend on it as a real
  dependency. A grammar crate that still depends on the *full* `tree-sitter`
  crate directly is a pre-shim, older-generation crate, and that's the
  actual, generalizable tell for "this one might not link":
  - `tree-sitter-kotlin` (0.3.8, crates.io's only release) pins tree-sitter
    0.21/0.22 as a real dependency — Cargo's `links = "tree-sitter"`
    uniqueness rule refuses two versions of that native library in one
    binary, so it couldn't coexist with anything else here at all.
  - `tree-sitter-svelte`/`-vue`/`-wgsl`'s official crates.io releases pin
    tree-sitter 0.20.10 the same pre-shim way — but *without* a `links`
    conflict being raised (Cargo resolves and compiles each in isolation
    fine). The failure only shows up when their `language()` fn's return
    value is used somewhere expecting `tree_sitter::Language`: their
    bindings return that type from that *old* core, a structurally
    different Rust type from this workspace's 0.26 despite an identical
    name. `cargo build -p graviton-indexer` (lib-only) won't catch this if
    the mismatched arm is never reached in that crate's own code — it only
    surfaces as a `match`-arm type error once `ts_language()` actually
    tries to return one.
  - `tree-sitter-latex`'s official crate compiles and links its *grammar*
    fine, but is missing `scanner.c` from its packaged file list entirely
    — a real packaging bug, not a version issue — so its external scanner
    (used for verbatim/raw-environment lexing) references
    `tree_sitter_latex_external_scanner_{create,destroy,scan,serialize,
    deserialize}` symbols with no definition anywhere, undefined at
    **executable** link time only. A plain `cargo build -p graviton-indexer`
    (a library build, where not every symbol needs resolving yet) won't
    catch this — it only appears once something links the real `grv`
    binary or a test binary, which is why every new-language batch is
    verified with a full `cargo build --workspace`, never a per-crate
    build alone.
- **The fix**: each of the five now uses a real, actively-maintained fork
  instead — checked before adding, not just swapped in blindly: real
  crates.io download/update history, a `tree-sitter-language` dependency
  (confirming it's the modern, shim-based generation), and for Latex,
  confirming `scanner.c` actually ships this time.
  - `tree-sitter-kotlin-ng`, `tree-sitter-svelte-ng`, `tree-sitter-wgsl-bevy`
    are all published, current releases from the `tree-sitter-grammars`
    GitHub org — the same community maintenance org Zed/Neovim/Helix lean
    on for grammars whose original author has moved on.
  - `codebook-tree-sitter-latex` republishes `latex-lsp/tree-sitter-latex`
    (the real, actively developed upstream — `latex-lsp` also maintains
    `texlab`, an established LaTeX language server) under a crates.io name
    that isn't already squatted by the broken original.
  - Vue had no equivalent published release anywhere — only an unpublished
    `update` branch on the `tree-sitter-grammars` fork. Added as a `git`
    dependency pinned to that branch's exact commit (not a floating
    branch reference, so this doesn't silently change under us), since
    crates.io simply has no better option yet.
  - **Explicitly rejected**: several *other* recently-published
    `tree-sitter-vue-*` crates on crates.io also "fix" the same symptom.
    Not used, on purpose — a cluster of near-identical, recently-published
    crates from unaffiliated, unfamiliar authors clustered around one
    popular missing package name (one had 100k+ downloads within months of
    a 0.1.0 release) is a real dependency-confusion/typosquatting pattern
    worth being paranoid about, not just an abundance of choice. This
    matters more, not less, for a tool whose own purpose includes security
    tooling.
- Kotlin/WGSL/Latex all gained a real, verified `def_query_src` once
  linked (Kotlin/WGSL both expose clean `name:` fields; Latex's
  `\section{...}`/`\label{...}` are the structural "definitions" its
  grammar actually has). Svelte/Vue did not — see the tier breakdown above
  for why that's a real, different limitation, not an oversight.

### Vendored grammars — Nim, Crystal (`vendor/`)

The next escalation from the fork-replacement pattern above, for the two
cases where even a `git` dependency to a better fork wasn't enough:

- **Crystal**: `crystal-lang-tools/tree-sitter-crystal` (the fix -- see
  "Import resolution" above for why the crates.io release needed
  replacing) already uses the modern `tree-sitter-language` shim, so a
  plain `git` dependency should have worked with zero other changes. In
  practice, `cargo build` against it repeatedly hit a transient SSL/
  network error specific to this sandbox (`SSL error: unknown error;
  class=Ssl (16)`), even with `net.git-fetch-with-cli` enabled (added to
  `~/.cargo/config.toml` for this reason, and harmless to leave on
  regardless) -- while a plain `git clone` of the exact same commit
  succeeded reliably in well under a minute. Rather than keep fighting
  one host's flaky connectivity in this specific environment, the
  already-cloned source was vendored directly into `vendor/tree-sitter-crystal/`.
- **Nim**: a harder case than Crystal's -- `alaviss/tree-sitter-nim`'s own
  Rust binding still depends on the pre-shim, full `tree-sitter` crate
  directly (pinned `~0.25`), which is a real `links = "tree-sitter"`
  conflict with this workspace's `0.26` (confirmed via an actual failed
  build, not assumed from the version numbers alone). The fix needed is
  exactly one line in that crate's own `Cargo.toml` (`tree-sitter =
  "0.26"` instead of `"~0.25"`) -- verified to parse real Nim source
  correctly with zero other changes, since the C-level ABI a generated
  parser exposes doesn't depend on which Rust crate version wrote the
  binding glue around it. Patching a dependency's manifest can't be done
  through a plain `git`/registry dependency at all (Cargo always honors
  the depended-on crate's own declared version requirements) -- normally
  this project's move would be to fork the upstream repo, apply the
  patch, and depend on that fork via `git` (the same shape as the Vue
  dependency above), but this session had no GitHub API/`gh` CLI
  credentials available to create a new repository for it. Asked
  directly, the user chose to vendor the (one-line-patched) source
  locally instead of waiting on that -- `vendor/tree-sitter-nim/`, ~41MB,
  by far the largest thing in this repository, a real and known tradeoff
  the user made explicitly aware of the size.

Both `vendor/*/NOTICE.md` files record the exact upstream commit, exactly
what (if anything) was changed from it, and why vendoring was the right
call over every other option tried first -- read those before touching
either directory again, especially before "helpfully" upgrading either
grammar to a newer upstream commit without re-deriving whether the same
patch (or no patch at all, if upstream fixes it first) is still needed.
Swapping either grammar broke that language's *existing*
`def_query_src`/`call_query_src` (different field/node names between the
old and new grammars, e.g. Crystal's `method_definition`/`(call name:
...)` → `method_def`/`(call method: ...)`) -- an expected, real cost of a
grammar swap, not a regression that slipped through; both were re-verified
against real samples and fixed the same turn, not left broken.

### Text-predicate-based queries (Elixir, Racket, Scheme)

A tree-sitter grammar node-type mapping is enough for most languages
because a function/class/struct is a genuinely distinct node type. Three
grammars break that assumption in a way no amount of query cleverness on
node *shape* alone can fix:

- **Elixir** compiles `def`/`defp`/`defmacro`/`defmodule` — and, just as
  importantly, `if`/`case`/`unless`/`receive`/every other control-flow
  form — down to the exact same `call` node with a `do_block` child. The
  grammar has no "this call is special" bit.
- **Racket and Scheme** are generic S-expression grammars: `(define (foo
  x) ...)` and an ordinary function call like `(+ x 1)` are both just a
  `list` of children. `define` isn't a keyword to the grammar at all.

A query that matches on shape alone (`(call ... (do_block))`, `(list
(symbol) (list ...))`) would therefore also match every `if` block or
every function call — not a minor imprecision, a query that flags most of
an ordinary source file as "definitions". Tree-sitter's query language has
an answer for exactly this: `#eq?`/`#any-of?` predicates that constrain a
capture's *text*, not just its shape (e.g. `(#eq? @kw "defmodule")`).

The catch: predicates are parsed by `Query::new` but **not enforced by
`QueryCursor` on its own** — the tree-sitter Rust crate deliberately leaves
evaluating them to the caller (`QueryMatch::satisfies_text_predicates`),
since it doesn't want to force a specific text-lookup strategy on every
consumer. Nothing in this codebase called that method before this batch,
so a predicate in a query string would previously have been silently
inert — parsed, never checked, every match kept regardless. Fixed once, in
`extract_symbols` (`crates/indexer/src/lib.rs`), for every language at
once:

```rust
let mut pred_buf1 = Vec::new();
let mut pred_buf2 = Vec::new();
let mut bytes_provider = bytes;
let mut matches = cursor.matches(&query, tree.root_node(), bytes);
while let Some(m) = matches.next() {
    if !m.satisfies_text_predicates(&query, &mut pred_buf1, &mut pred_buf2, &mut bytes_provider) {
        continue;
    }
    // ... existing capture-extraction logic, unchanged
}
```

This is a no-op for every parsed language whose queries declare no
predicate — `satisfies_text_predicates` returns `true` when a pattern has
none. Elixir's def query combines this with `.` anchors (`(list . (symbol)
@kw . (symbol) @name)`, in the Racket/Scheme case) to pin a capture to an
exact child position — needed because an unanchored pattern can match a
node's *later* children just as validly as its first ones, which for
`(define (foo x) (+ x 1))` could otherwise capture `+` (from the body) as
if it were the name being defined.

**A second, easy-to-miss requirement, found the hard way after the fix
above still didn't work**: a predicate must be nested *inside the same
outer parentheses* as the pattern it modifies, not written as a sibling
top-level form after it. The natural-looking

```scheme
(call target: (identifier) @kw (arguments (alias) @name) (do_block)) @def
(#eq? @kw "defmodule")
```

silently compiles to **two separate patterns** — the `(#eq? ...)` line
becomes its own pattern with no node content, matching independently (and
uselessly) elsewhere in the tree, while the real pattern keeps its empty
predicate list and stays completely unfiltered. `is_predicate_actually_enforced_on_real_def_query`,
a debug test added while chasing this (`(if x do y end)` in Elixir
returning `["x"]` as a "definition") is what caught it — every earlier
"passing" predicate test up to that point (Elixir def, Racket/Scheme def)
had happened to pass anyway, because their sample inputs were never
actually structurally ambiguous enough to need the predicate — the `.`
anchors alone were doing all the real filtering, silently masking that the
predicates themselves were inert. The fix is one extra pair of parens
wrapping pattern *and* predicate together:

```scheme
((call target: (identifier) @kw (arguments (alias) @name) (do_block)) @def
 (#eq? @kw "defmodule"))
```

Every predicate-bearing pattern in `lang.rs` (Elixir/Racket/Scheme's
`def_query_src`, and Elixir/Racket/Scheme/Asm's `call_query_src`) uses this
wrapped form. The lesson generalizes: a predicate-based query is only
proven correct once a test exercises an input that would produce the
*wrong* answer if the predicate were silently inert — `racket_define_and_struct`'s
`assert!(!found.contains(&"+".to_string()))` and
`elixir_call_excludes_definition_keywords`'s check against a real
`defmodule`/`def` sample are exactly that kind of test, not just assertions
that the real names are found.

### The query safety net (`lang::query_predicate_safety_net`)

Hand-crafting one adversarial sample per predicate-bearing query (the
previous section) proves *that specific* query correct — it doesn't
prevent the *next* predicate, in the next language added a year from now,
from repeating the exact same mis-nesting mistake. Rather than trust "the
next person will remember to write an adversarial test", this checks a
structural invariant that holds for every query in this file regardless of
what it matches: every intended top-level pattern ends with exactly one
`@def` (or `@call`) capture, so a correctly-compiled query's
`Query::pattern_count()` must equal how many times that capture name
appears in the source text. A mis-nested predicate makes `pattern_count()`
grow past that number — silently, for any language, present or future —
and this test catches it immediately:

```rust
for &lang in ALL_LANGS {
    if let Some(src) = lang.def_query_src() {
        let query = Query::new(&lang.ts_language().unwrap(), src).unwrap();
        let expected = count_capture_token(src, "@def"); // exact-token count, not a substring match
        assert_eq!(query.pattern_count(), expected, "...");
    }
}
```

(`count_capture_token`, not a plain `src.matches("@call").count()`,
because every call query also has `@callee` — a plain substring search
would double-count it, since `"@call"` is itself a substring of
`"@callee"`. A real bug this test's own *first* draft hit before catching
anything real, worth remembering: a checker's checker still needs to be
correct.)

**Immediately found a second, unrelated, real bug on its first real run**:
TypeScript/TSX's `def_query_src` failed to *compile* outright —
`(class_declaration name: (identifier) @name)` — because in this
grammar, `class_declaration`'s `name:` field is a `type_identifier`, not a
plain `identifier` (verified against the real `node-types.json`). Tree-sitter
statically type-checks a query's field/type pairs against the grammar at
`Query::new` time and rejects a mismatch as an "impossible pattern" — for
the *entire multi-pattern query string*, not just the one bad pattern. That
means every TypeScript/TSX symbol — functions and interfaces included, not
just classes — had been silently extracting nothing, for as long as this
bug existed, which could be as far back as this project's very first
version: TypeScript and TSX, unlike every language added since, had never
had a dedicated real-sample test (`lang::original_six_queries` closes that
gap now, for all six of the original languages). Fixed by using the
correct field type; a permanent test (`typescript_class_is_not_silently_dropped`)
now asserts a real sample's function *and* interface *and* class all come
back together, specifically because "the whole query silently produces
nothing" is a failure mode where testing only one def kind wouldn't have
caught the others going missing too.

`ALL_LANGS` (`crates/indexer/src/lang.rs`) is what lets both safety-net
tests iterate "every language" without a separate enumeration to keep in
sync by hand: a hand-written `&[Lang]` const, but tied to the actual enum
by `_all_langs_exhaustive_match_guard`, a `match` with every arm spelled
out and no wildcard — add a `Lang` variant without adding an arm there and
the crate fails to *compile*, not just silently skips the new language in
these tests.

## Tool execution (`grv tool`)

`crates/cli/src/tools.rs` runs a whitelisted binary via `std::process::Command`
with piped stdout/stderr, tees each line to the terminal live (two reader
threads feeding one `mpsc::channel`, since `Command`'s piped streams aren't
`Read` from two threads without separate handles) while also accumulating
it, then writes one `tool_runs` row plus one `content_fts` row (`kind =
'tool_output'`, `path = 'tool://<tool>#<id>'`) on exit.

That second insert is the entire integration point with retrieval: `ask`'s
`search_chunks` pass already queries `content_fts` with no `kind` filter, so
a scan's output becomes part of the context budget for the very next `grv
ask` with no additional plumbing. `grv index --force` deliberately spares
`content_fts` rows with `kind = 'tool_output'` (see `clear_index` in
`crates/core/src/lib.rs`) — recon history isn't derived from the repo tree,
so re-indexing code shouldn't discard it.

The whitelist (`ALLOWED_TOOLS`) is a scoping choice, not a sandboxing one:
`grv tool run` executes exactly the command you typed with your own
process permissions, identical to running it in the shell directly. Its
purpose is keeping the subcommand a recon-tool launcher instead of turning
into a second, worse shell — it is not a defense against a malicious
argument to a whitelisted tool, which was never the threat model here (you
already have a shell).

## Multi-agent design (`grv ask --agent`, `grv crew`)

`crates/cli/src/agents.rs` defines 22 `AgentSpec`s (key, display name,
tagline, system prompt, model tier) across five categories — programming,
infrastructure & engineering, defensive security, offensive security, and
SINGULARITY as a coordinator with no first-hand analysis of its own (it
only synthesizes whichever other agents' output it's given). `grv agents`
prints the live, data-driven roster; this section covers the mechanics
that make it more than a system-prompt picker, using the original four
(ARCHITECT/SENTINEL/REAPER/SINGULARITY) as the running example.

Two important scoping decisions:

- **Personas by default, real multi-model when configured.** Out of the box
  every agent still calls the one model in `grv config`, because that's
  what fits this hardware without any setup. But `AgentSpec` carries a
  `ModelTier` (`Fast`/`Standard`/`Deep`) and `Config::model_for_tier`
  resolves it against optional `model_fast`/`model_deep` overrides — set
  those and, e.g., TESTER/SUPPLYCHAIN/CLOUDSEC/SINGULARITY (mechanical,
  pattern-matching work) actually run on a small 1.5B-3B model while
  ARCHITECT/CRYPTOGRAPHER/REAPER/BINEXP (design and exploit-dev reasoning)
  stay on the 8B one. See "Model tiers and `grv swarm`" below for the
  concurrency story once more than one model is configured.
- **What makes `crew` a pipeline instead of asking the same question four
  times**: each stage's prompt includes the *actual text* of every prior
  stage's output (`prior` in `cmd_crew`, capped to `context_char_budget()`
  and truncated from the front — i.e. it keeps the most recent agents'
  findings — if a long crew run would otherwise blow past `num_ctx`), not
  just the same retrieved code again. REAPER reads what ARCHITECT actually
  said about the code before finding what's exploitable in it; SINGULARITY
  reads all three real outputs before writing a decision brief.

`grv investigate` is orthogonal to the agent roster: it's a structured
*output format* (`agents::INVESTIGATE_FORMAT`, appended to whichever
agent's base prompt) rather than its own persona, so
`grv investigate --agent sentinel "..."` is meaningful (structured
defensive audit) distinct from plain `grv ask --agent sentinel` (free-form).

### Per-agent retrieval: a bounded read-only tool loop, not one fixed pass

`build_context` (initial retrieval — explicit files, symbol matches, FTS
chunks, semantic hits if configured) is still computed once per `ask`/
`crew`/`swarm`/`review` invocation and handed to every agent as a starting
point, but it's no longer the *only* evidence an agent gets to work with.
`agentic::run_read_only_loop`/`run_read_only_loop_with` wrap every model
call in `ask`/`crew`/`review`/`swarm` (`run_pipeline` for the first two, a
per-agent spawn for the third) in the same bounded tool loop mission's
leaves and `grv run` already used: `read_only_tool_defs()`'s tools
(`search_code`/`semantic_search`/`read_file`/`list_dir`/`web_search`/
`web_fetch`/`git_status`/`git_diff`/`git_log`), dispatched via the existing
`dispatch_read_only`, capped at `MAX_READONLY_TOOL_STEPS` (6) with a final
no-tools call to force a text answer if that budget runs out without one.

This is what makes "per-agent retrieval" real without needing a
per-specialty retrieval *query*: REAPER doesn't need a differently-shaped
initial FTS query than ARCHITECT's — it can just call `search_code`/
`semantic_search` itself, with whatever question its own reasoning
actually needs answered, mid-turn. `run_read_only_loop` is a thin
convenience wrapper (stdout streaming + `println!` tool-call announcements)
around the callback-based `run_read_only_loop_with`, which takes plain
`on_token`/`on_tool_call` closures instead — the same split `grv serve`'s
streaming `ask`/`review` reuse (see below), so there's one tool-loop
implementation, not a terminal one and a socket one drifting apart.

**Why `swarm` doesn't stream tokens live even though it now runs this
loop too**: several agents run concurrently there (`tokio::JoinSet`);
character-level interleaving from more than one at once would be
unreadable, so `swarm` passes `stream_final_answer: false` and prints each
agent's full answer as one block once it finishes. Tool-call announcement
lines can still interleave under concurrency — the same tradeoff `grv
mission`'s concurrent leaves already accept, kept consistent rather than
solved differently in two places.

## Model tiers, `grv swarm`, and `grv mission`: real multi-model, resource-aware

The 4-agent (later 22-agent) design above runs everything through one
model — accurate for the default config, but the tiering added from v0.3
onward makes GRAVITON capable of genuinely running more than one model,
concurrently, sized continuously to what the machine can actually hold —
this section covers `model_fast`/`model_deep`, `grv swarm`, and the
recursive `grv mission`, in the order they build on each other.

- `Config.model` is the `Standard` tier and always the fallback.
  `model_fast`/`model_deep` are optional overrides (`grv config
  --model-fast qwen2.5:1.5b --model-deep qwen3:14b`); leave them unset and
  nothing changes from v0.2's one-model behavior. Each `AgentSpec` declares
  which tier it wants — see the table `grv agents` prints.
- **`grv crew`** stays sequential by design (§ above) — its whole point is
  real hand-off, so parallelizing it would just make each stage read a
  half-finished prior stage. Multi-model there means *different-sized*
  models per stage, not concurrent ones.
- **`grv swarm --agents a,b,c "question"`** is the flat, one-level
  concurrent mode: independent agents (no hand-off — for questions that
  don't need one, e.g. running SENTINEL/REAPER/CRYPTOGRAPHER over the same
  question at once) fired via `tokio::task::JoinSet`, each against its own
  tier's model.
- **`grv mission "big task" --max-depth N`** (`crates/cli/src/mission.rs`)
  is the recursive mode: a planner call (`ModelTier::Fast`, cheap — it's
  called once per tree node) proposes a JSON list of subtasks, each
  assigned to whichever specialist fits; each subtask recurses through the
  same planner step up to `--max-depth` (default 2, hard ceiling 4), and a
  subtask the planner judges already atomic returns a single
  self-referential entry, which short-circuits straight to a leaf — so a
  simple task doesn't get artificially split just because depth budget is
  available. Children of one node run concurrently
  (`tokio::task::JoinSet`, recursed via a hand-rolled `Pin<Box<dyn Future
  + Send>>` — Rust needs that indirection for a function that calls
  itself in async code, since the naive recursive-`async fn` type would be
  infinitely sized), and their results are combined by one more model call
  (a synthesis step, also `Fast` tier) before returning up to the parent.
  JSON extraction from the planner's response is bracket-counted by hand
  (string-escape aware) rather than assumed well-formed, because a small
  local model asked for "ONLY a JSON array, no prose" often adds prose or
  a code fence anyway.

### The concurrency number is computed and continuously re-sampled, not assumed once

`crates/cli/src/resources.rs` separates two different constraints that a
single "how many models fit" number was conflating in earlier drafts of
this design:

- **RAM bounds how many *distinct* models can be resident at once.**
  `Capacity::detect()` reads total/available system RAM (`sysinfo`);
  `model_sizes_mb` reads each configured model's on-disk size
  (`OllamaClient::model_sizes_mb`, via `/api/tags`); `pick_concurrency`
  reserves a 30% safety margin and packs as many distinct model tags as
  fit in the rest.
- **A resident model's own parallel-serving capacity bounds how many
  callers can share it.** Five agents on the same `Standard`-tier model is
  the common case (see the tier assignments in `grv agents`) — capping
  concurrency at "number of distinct models" would wrongly serialize them
  even though Ollama happily serves several concurrent requests against
  one already-loaded model. `ASSUMED_PARALLEL_PER_MODEL` (3) accounts for
  that: final concurrency = distinct-models-resident ×
  `ASSUMED_PARALLEL_PER_MODEL`, capped by however many agents/subtasks are
  actually pending. This was caught empirically, not assumed correct up
  front — an early version capped concurrency at 1 whenever only one
  distinct model was configured, which a mock-server test then visibly
  serialized three independent `swarm` agents that should have run
  together; fixed once the flaw showed up in real output, not left as a
  known gap.
- **`grv swarm`** computes this once at startup (`safe_concurrency`, a
  thin wrapper: fetch sizes, then `pick_concurrency` once).
- **`grv mission`** needs more than a one-shot number, because a mission's
  shape (how many subtasks exist, how deep they recurse) isn't known until
  the planner starts producing it — `resources::LiveScheduler` fetches
  model sizes once at spawn (they don't change mid-run) then re-samples
  the cheap, local part (`Capacity::detect()`, no network call) every 3
  seconds for the whole mission, growing or shrinking a
  `tokio::sync::Semaphore`'s permit count via `add_permits`/
  `forget_permits` to match. Every single model call anywhere in a
  mission's tree — leaf work, every planner call, every synthesis call —
  acquires one permit from the *same* scheduler instance, so a mission
  that fans out into 15 subtasks can never put more concurrent model calls
  on the machine than its RAM can hold *at that moment*, and the gate
  grows back into headroom that frees up as earlier subtasks finish. This
  is the direct answer to "don't crash Ollama or the OS no matter how much
  the task decomposes" — a hard structural guarantee (one shared gate,
  every call site goes through it), not a best-effort convention agents
  are trusted to follow.
- `--max-parallel` (both `swarm` and `mission`) overrides the estimate for
  a user who has tuned `OLLAMA_NUM_PARALLEL` differently or otherwise
  knows their real headroom better than the heuristic does.
- **`grv status`** surfaces the same estimate plus
  `resources::top_memory_consumers` — the actual heaviest processes on the
  machine right now, via `sysinfo`'s process list — so "why did it pick
  this number" has a concrete answer instead of a black box.
- **What this doesn't pretend to solve**: CPU threads (6C/12T on the
  reference hardware) are shared across whatever runs concurrently —
  three agents generating at once each get roughly a third of the cores,
  so each individual stream is slower than it would be alone. The win is
  wall-clock for the *batch*: independent work that would otherwise run
  one piece after another finishes closer to together. Both `swarm` and
  `mission` state this up front rather than hiding it.
- Ollama itself owns actual model residency (loading on first request,
  evicting LRU under memory pressure) — GRAVITON's estimate exists only to
  pick a sane concurrency *before* asking Ollama to do anything, not to
  duplicate Ollama's own memory management.

### Real-time web access (`web_search`/`web_fetch`, `crates/cli/src/web.rs`)

Added so agents don't answer version-specific or time-sensitive questions
from a frozen training snapshot. Deliberately no API key: a keyed search
API (Brave/Tavily) returns better results but requires the user to
provision and manage a key before the tool works at all, which fails the
"works out of the box" bar this project holds itself to elsewhere (no
cloud, no signup, `ollama pull` and go). The cost of that choice is
fragility — `web.rs` parses DuckDuckGo's actual result HTML with
hand-rolled string scanning (no HTML-parser crate, same philosophy as
`graviton_core`'s hand-rolled TOML), so a markup redesign on their end
breaks it until updated. Two DuckDuckGo endpoints (`lite` and `html` —
same engine, different endpoints, so not real backend diversity but
resilient to one being throttled) are used with a fallback. Mojeek and
Startpage were tried and rejected during development, not assumed to
work: Mojeek serves a captcha to scripted requests, Startpage refused the
connection outright — verified with a live `curl` before writing a parser
for either. `web_fetch` strips `<script>`/`<style>` blocks and tags,
unescapes entities, and collapses whitespace, also hand-rolled. Both
tools are scoped to `grv run` (and `grv mission`'s underlying agent
calls) — `ask`/`investigate`/`crew`/`swarm` stay single-pass and
tool-free by design (§ above); reach for `run`/`mission` when an answer
needs to be checked against the live internet, not assumed from memory.

## The agentic loop (`grv run`, `crates/cli/src/agentic.rs`)

Everything above (`ask`/`investigate`/`crew`) is one retrieval pass and one
generation: the model reads context and answers in text. `grv run` is a
different interaction model entirely — a real tool-use loop, matching
Ollama's `/api/chat` `tools` parameter (OpenAI-style function calling):

1. Send the conversation so far + the tool schemas (`agentic::tool_defs`).
2. If the model's response carries `tool_calls` instead of (or alongside)
   text, execute each one (`dispatch_inner`), append an assistant message
   carrying the raw tool calls and a `role: "tool"` message per result, and
   loop.
3. If the response is plain text with no tool calls, that's the final
   answer — stream it and stop. Hard cap at `MAX_STEPS` (40) so a
   model that won't converge doesn't loop forever.

This required extending `graviton-llm` beyond the original text-only
`chat_stream`: `ChatMessage` gained `tool_calls`/`tool_name` fields (so
assistant-with-tool-calls and tool-result messages serialize correctly),
and `ChatResult` carries both accumulated text and accumulated tool calls
out of the streamed response.

### Tools

`read_file`, `list_dir`, `write_file`, `edit_file` (unique
old_string/new_string replace — same shape as this codebase's own edit
tool, chosen over unified-diff patches because a small local model
produces "replace this exact block" far more reliably than a correctly
context-lined diff), `delete_file`, `run_shell`, `recon_tool` (the
`grv tool` whitelist, exposed as a tool so the agent can run nmap/ffuf/etc.
itself instead of the user doing it out-of-band), `search_code`/
`semantic_search` (the indexed repo, mid-task — see "Semantic search"
below for why the latter needs an embedding model configured and errors
clearly when it isn't, rather than silently falling back), and `ask_user`
(a fixed-option clarifying question — the agent's own version of
Claude Code's own `AskUserQuestion`: present a short, known set of sane
answers instead of guessing or asking in free text the agent then has to
parse back out of a reply; not for yes/no about an action, which the
existing write/edit/delete/shell confirmation already covers). `--browser` adds
`browser_navigate`/`browser_eval`/`browser_screenshot`/`browser_console`,
backed by `chromiumoxide` driving the system's Chromium over CDP
(`crates/cli/src/browser.rs`) — headless, one page kept alive for the
session, console output captured via a `Runtime.consoleAPICalled` event
listener. Browser tools are opt-in (`--browser`) rather than always-on
because launching Chromium has real startup cost and most coding tasks
don't need it.

`resolve_rel` gives every file tool a cheap containment check (rejects a
path that normalizes outside the repo root) — not a hardened sandbox
(symlinks aren't specially handled), just a guard against the model
wandering outside the project by constructing a `../../` path.

### Safety model: confirm by default, checkpoint always

This is the one place in GRAVITON where the *model* decides to write files
or run commands, not the user typing them — a materially different trust
situation from `grv tool run`, where the user names the exact tool and
args. Two independent mechanisms, chosen deliberately not to overlap:

- **Confirmation** (`state.io.confirm`, via the `RunIo` trait — see "A
  pluggable confirm/output sink" below) gates `write_file`, `edit_file`,
  `delete_file`, `run_shell`, and `recon_tool` — the user sees the exact
  diff (`similar`-generated, via `diff_preview`) or command before it runs,
  and can decline. `--yolo` skips this for a fully autonomous run.
- **Checkpointing** (`checkpoint.rs`) is unconditional — it runs even under
  `--yolo` — and is scoped to file state only: before any write/edit/delete,
  the pre-change bytes (or the fact that the file didn't exist yet) are
  saved to `.graviton/checkpoints/<session>/<seq>.bak` plus a
  `manifest.jsonl` line. `grv rollback` replays that manifest backwards.

  **Why raw file snapshots instead of git commits/stashes for this**: `grv
  run` operates inside *the user's* repo, which already has its own git
  state — staged changes, a mid-rebase, a detached HEAD, or simply no `.git`
  at all (plenty of quick scratch directories aren't repos, and `grv run`
  works in them fine). Driving git for internal bookkeeping would mean
  either polluting the user's actual history/reflog with tool-generated
  commits, or juggling stashes that can collide with whatever they already
  have stashed — and it would require a repo to exist in the first place.
  A plain byte-snapshot-per-tool-call sidesteps all of that: it works
  identically in a git repo or a bare directory, gives per-*step* rollback
  granularity (`--to N`) without needing an interactive-rebase-style commit
  edit, and never touches `.git` at all — no risk of a tool-driven commit,
  stash, or signing step interacting badly with however the user already
  has git configured.

Deliberately *not* checkpointed: `run_shell`. A shell command's side
effects are unbounded (it can touch anything on the filesystem, start a
process, hit the network) and there's no generic way to snapshot "the
world" before running one. That's exactly why confirmation is the primary
safety net for `run_shell` rather than promising a rollback the system
can't actually guarantee — being honest about that boundary matters more
than making the feature look more complete than it is.

### Session resumability, a visible plan, and mid-task steering

Three related gaps closed together, since they share the same
infrastructure — a checkpoint session already existed per `grv run`
invocation to track file changes, so it's the natural place to also store
the conversation itself:

- **`grv run --continue [--session <id>] ["additional instruction"]`**
  (`checkpoint::Session::open_existing`, `append_message`/`load_transcript`)
  restores a session's full message history — every system/user/assistant/
  tool-result message, not just a summary — instead of starting cold.
  `push_and_record` (a thin wrapper the loop calls instead of
  `messages.push` directly) writes every message to `.graviton/
  checkpoints/<id>/transcript.jsonl` as it happens, so *every* `grv run` is
  resumable by default, not just ones a user thought to flag in advance.
  `open_existing` also recomputes `next_seq` from the existing manifest
  (max seq + 1) rather than restarting at 0, so file-change step numbers
  never collide with the previous invocation's — `--to N` rollback stays
  correct across a resume. `ChatMessage` gained `Deserialize` (previously
  serialize-only, since it only ever went out over HTTP to Ollama) to make
  this round-trip possible.
- **A visible, persisted plan** (`update_plan` tool, `crates/cli/src/
  agentic.rs`'s `format_plan`): the agent reports a list of `{text,
  status}` steps whenever it forms or updates a plan; GRAVITON prints it as
  a `[ ]`/`[~]`/`[x]` checklist immediately and saves the latest snapshot
  to `plan.json` in the session dir. `grv plan [session]` shows it outside
  a live run, and `--continue` prints it up front so resuming shows where
  things stood before asking the model anything. This is deliberately a
  self-reported tool call, not a structure GRAVITON infers from tool-call
  history — inferring "the plan" from a stream of `write_file`/`run_shell`
  calls after the fact would be guesswork; asking the model to state it
  costs one extra tool definition and is unambiguous.
- **Mid-task steering through existing confirmation gates**
  (`agentic::Decision`/`require_allowed`): a confirmation prompt no longer
  just parses y/n — anything else typed is a `Decision::Redirect(text)`
  that declines the action *and* carries the typed text into the tool
  result the model sees next ("user declined this write — the user says:
  <text>"), so "no, do X instead" reaches the next turn as real guidance
  instead of only being expressible as a blind refusal. This is
  deliberately scoped to existing pause points, not a general interrupt: a
  `--yolo` run has no confirmation gates to redirect at, so it remains
  Ctrl-C-only, and there's no mechanism to interject while a model call is
  actually in flight (mid-generation). Solving that would need racing
  stdin against the HTTP response (`tokio::select!`) and is a reasonable
  next step, not something this claims to already do.

`grv mission --continue` now exists too (see "Mission session resume"
below), but it's a genuinely different mechanism from `grv run`'s — a
tree-shaped, concurrent execution has no single linear transcript to
append to, so it needed its own checkpoint format rather than reusing
`Session`'s.

### Mission session resume (`checkpoint::MissionCheckpoint`, `grv mission --continue`)

`grv run`'s resumability (above) works because its execution is linear —
one transcript, appended to in order. `grv mission`'s is a tree, executed
*concurrently* (siblings run in parallel `tokio::spawn`ed tasks, finishing
in whatever order the scheduler gets to them), so there's no single
sequence to log. `MissionCheckpoint` instead keys a flat map by tree
position — `"0"` for the root, `"0.0"`/`"0.1"` for its children, `"0.1.0"`
for a grandchild, and so on — rewritten to `mission_tree.json` as a whole
each time any node's status changes (`Pending`/`Done`/`Failed`, plus its
result once `Done`). Cheap in practice: even a wide, deep mission produces
at most a few hundred nodes, and a write only happens at node boundaries
(once per subtask/decompose/synthesize call), never per token.

Resuming (`execute_node` checks `checkpoint.get(&node_path)` before doing
anything) has two distinct cases, both load-bearing:

- **A node already `Done`** short-circuits immediately, returning its
  cached result — no model call at all. This is what makes resume actually
  useful: a mission that got 8 of 10 subtasks done before something failed
  doesn't re-run those 8.
- **An internal node that was already decomposed** (it has a recorded
  `subtasks` list, whether or not it finished synthesizing) replays that
  *exact* split instead of calling the planner again. This matters because
  the planner is a live model call — asking it again on resume could
  return a differently-shaped decomposition, which would silently orphan
  any already-completed children's cached results under node paths that no
  longer correspond to anything in the new split.

**A real correctness bug caught before it shipped, not after**: the
initial implementation let a resume default `--max-depth` independently of
what the original run used. A mission started with `--max-depth 1` (so its
depth-1 children are leaves) that got resumed with no `--max-depth` flag
picked up `DEFAULT_MAX_DEPTH` (2) instead — meaning a node the original run
treated as terminal would suddenly try to decompose *again* on resume,
producing a wrong tree shape live and verified via a real failing-then-
succeeding mock run before the fix (a resumed leaf spuriously re-decomposed
into two fresh subtasks instead of just running). Fixed by persisting the
resolved depth once, at the start of a fresh mission (`save_max_depth`,
`mission_meta.json` alongside the tree file), and reusing it on resume
unless `--max-depth` is passed explicitly (which always wins, letting a
user deliberately widen or narrow a resumed mission).

`--max-parallel`/the `LiveScheduler` aren't part of what's persisted or
replayed — they're a live resource estimate re-sampled from the machine's
*current* state, and always should be, resume or not.

### A pluggable confirm/output sink (`crates/cli/src/run_io.rs`) — one loop, two front ends

`agentic::run` originally talked to a terminal directly: `println!` for
every tool-call announcement/plan update/checkpoint summary, a blocking
stdin read for confirmation. Once `grv serve`'s `run_start` needed to drive
the *same* loop from a socket instead (see below), that coupling had to
come out — not by writing a second loop, which would drift from the first
one's behavior the moment either changed, but by making the loop talk to a
small trait instead:

```rust
pub trait RunIo: Send + Sync {
    fn emit(&self, line: String);                    // one line of output
    fn on_token(&self, tok: &str);                    // one streamed answer token
    fn note_checkpoint_id(&self, _id: &str) {}         // default no-op
    fn confirm(&self, auto_approve: bool, action: String)
        -> Pin<Box<dyn Future<Output = Decision> + Send + '_>>;
    fn ask_choice(&self, question: String, options: Vec<String>, multi_select: bool)
        -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>>;
}
```

`confirm`/`ask_choice` are boxed by hand (`Pin<Box<dyn Future<...>>>`)
rather than plain `async fn`s in the trait: `dyn RunIo` needs object safety
(`agentic::run` doesn't know at compile time which implementation it has),
and native async-fn-in-traits doesn't support dynamic dispatch without
either this or the `async-trait` crate — hand-rolling two boxed methods was
less than pulling in a dependency for it.

Two implementations:
- **`TerminalIo`** (`run_io.rs`) — the original behavior, byte for byte:
  `emit`/`on_token` print to stdout, `confirm` does the blocking stdin
  read, moved onto `tokio::task::spawn_blocking` so it's not literally
  blocking the async runtime (it always effectively blocked all other work
  in `grv run`'s single-agent CLI process anyway — this just stops relying
  on that as an accident of scheduling). `ask_choice` (the `ask_user` tool
  — see "Tools" above) is the same pattern: prints a numbered option list,
  reads a comma-separated pick (or `all` for a multi-select question).
- **`ChannelIo`** (`daemon.rs`) — output/tokens/`ask_choice` questions
  become `RunEvent`s on a `tokio::sync::broadcast` channel any number of
  `run_attach`ed connections can subscribe to; `confirm`/`ask_choice` block
  on their own per-session `mpsc` channel that `run_confirm`/
  `run_answer_choice` (from any connection, not necessarily the one that
  called `run_start`) feeds a `Decision`/chosen-option-list into.

This is also why `agentic::run`/`dispatch`/`dispatch_inner` no longer take
`&rusqlite::Connection` as a parameter at all (they used to) — `grv serve`
spawns `run_start` sessions via `tokio::spawn`, which requires the future
to be `Send`, and (per the `Send`-poisoning behavior documented under
"Semantic search" below) an async fn with `&Connection` anywhere in its
signature can't satisfy that regardless of how carefully it's used
internally. The two dispatch arms that need a connection (`recon_tool`,
`search_code`/`semantic_search`) each open their own short-lived one now,
the same pattern `dispatch_read_only` already used for `mission`'s leaves.

### Custom tools (`crates/cli/src/custom_tools.rs`) — extensibility without recompiling

`tool_defs` in `agentic.rs` is a fixed Rust function — real extensibility
needs a way to add a tool without touching that function or rebuilding
the binary. The design lands on the simplest thing that's actually useful:
a custom tool is a TOML file describing a named, described, parameterized
**shell command template** — `{{param}}` placeholders substituted with
that argument's value at call time.

- **Why a command template instead of a richer plugin API** (WASM module,
  dynamically loaded `.so`, subprocess-as-tool-server): a shell command is
  the one execution primitive GRAVITON already fully trusts and has a
  safety story for (`run_shell` already lets the model run arbitrary
  shell, gated by `confirm`/`--yolo`) — a custom tool is a friendlier,
  named, schema'd wrapper around exactly that trust boundary, not a new
  one. A WASM/plugin ABI would be more powerful (real logic, not just
  shell composition) and is a reasonable v2, but it's a materially bigger
  surface (a sandboxing story of its own) for a feature whose actual ask
  was "add a tool without recompiling," which a text file already solves.
- **Loading**: `custom_tools::load_all(root)` scans `~/.config/graviton/
  tools/*.toml` (every project) then `<root>/.graviton/tools/*.toml` (this
  project, shareable via the repo), parsing each with the real `toml`
  crate (added as a dependency for this specifically — unlike
  `graviton_core::Config`'s hand-rolled flat TOML, user-authored files
  with nested `[[params]]` tables deserve a real parser instead of a
  hand-rolled one silently mis-parsing an edge case). A bad file is a
  stderr warning naming the file, not a reason to abort `grv run` — one
  broken tool definition shouldn't take down an otherwise-working session.
  A project-local tool overrides a global one of the same name (loaded
  second into the same by-name map).
- **Argument substitution is shell-quoted, not string-interpolated**:
  `render_command` wraps every substituted value in single quotes,
  escaping embedded `'` — verified during development that a value like
  `it's; rm -rf /` lands as one inert literal argument to the underlying
  command instead of terminating it and injecting a second one. This
  matters because the *value* comes from the model's tool-call arguments,
  which is a less trusted source than the *command template* itself
  (author-written, checked into the repo).
- **Dispatch**: `agentic::dispatch_inner`'s final `other =>` arm (previously
  just "unknown tool") now checks the loaded custom-tool registry before
  giving up; a match renders the command and runs it through the same
  `confirm`/`require_allowed` gate and subprocess execution as
  `run_shell`, so it's confirmed before running unless `--yolo`, same as
  everything else that acts.
- **Discoverability**: `grv custom list` shows every loaded tool and which
  file it came from; `grv custom new <name>` scaffolds a working example
  (not a blank file) at `.graviton/tools/<name>.toml`; `grv custom show
  <name>` prints the exact `ToolDef`/JSON-schema the model would be given,
  so "will the model understand this argument" is checkable before
  running anything.
- **Scope**: custom tools are `grv run`-only, like the rest of the acting
  tool roster — `mission`'s leaves stay on the read-only subset
  (`read_only_tool_defs`) by design (§ above), and a shell-command tool is
  definitionally not read-only.

## Git-native tools, `grv review`, structured tests, and fine-grained permissions

Four gaps closed together (from a "what's missing vs. Claude Code"
conversation) — each is a small, self-contained addition, grouped here
because they share the same shape: a new tool or gate slotted into the
existing `agentic.rs`/`dispatch_inner` machinery, not a new subsystem.

- **`git_status`/`git_diff`/`git_log`** (read-only, shared with `mission`'s
  leaves via `dispatch_git_readonly`) and **`git_commit`** (`grv run`-only,
  gated like every other action) shell out to `git` with fixed argument
  arrays — never a model-supplied arbitrary git command string — so the
  agent gets real repository state instead of reconstructing it from
  separate `read_file` calls. `git_commit` is deliberately *not*
  checkpointed the way file writes are: a commit is already its own undo
  point (`git reset`/`git revert`), and there's no interactive-rebase
  machinery here to justify duplicating that.
  - **Caught during testing, not assumed correct**: these three read-only
    tools were initially added only to `read_only_tool_defs`/
    `dispatch_read_only` (the subset `mission` uses) and not to `grv run`'s
    own `dispatch_inner` — a mock-server test that had the model call
    `git_status` surfaced "unknown tool" immediately. Fixed by extracting
    the three tools' logic into `dispatch_git_readonly`, called from both
    dispatch paths, so the same bug class (a tool defined in the schema
    `grv run` sends but not wired into the dispatcher it uses) can't
    recur silently for these three.
- **`run_tests`**: auto-detects a test command from repo markers
  (`Cargo.toml` → `cargo test`, `go.mod` → `go test ./...`,
  `pyproject.toml`/`setup.py`/`pytest.ini` → `pytest`, `package.json` →
  `npm test`, `Gemfile` → `bundle exec rspec`; a model-supplied `command`
  overrides detection), runs it, and returns `summarize_test_output`'s
  parsed result — pass/fail, and for failures the specific failing test
  names/error lines — instead of raw stdout+stderr. The "loop" in
  "test→failure→fix loop" isn't new orchestration code: `grv run`'s
  existing step loop already lets the model call `run_tests`, see what
  failed, `edit_file`, and call `run_tests` again — what was missing was a
  clean enough failure summary for the model to act on without wasting a
  step misreading noisy raw output. The parser is explicitly heuristic
  (substring/prefix markers across cargo/pytest/jest/go/rspec's typical
  output shapes, unit-tested against a representative cargo-test failure
  and a no-recognizable-marker case), with a tail-of-raw-output fallback
  when nothing matches rather than returning an empty summary.
- **`grv review [range] [--staged] [--agents ...]`**: reuses the exact
  sequential hand-off loop `grv crew` uses (extracted into a shared
  `run_pipeline` helper during this change, so both stay in sync rather
  than drifting) but sources `context_block` from a real `git diff`
  instead of FTS retrieval — the distinction the README draws between
  `ask`/`crew` (indexed-chunk-based) and this. No range/`--staged` means
  `git diff HEAD` (staged + unstaged together), since "review what I
  haven't committed yet" is the common case, not just one half of it.
- **`.graviton/permissions.toml`** (`crates/cli/src/permissions.rs`):
  rules layered *underneath* confirm/`--yolo`, not replacing it — every
  side-effecting dispatch arm now calls `gate(state, tool, primary_arg,
  ...)` instead of `confirm` directly; `gate` checks permission rules
  first (first match in file order wins) and only falls through to the
  existing confirm/`--yolo` behavior when nothing matches. This is why a
  `deny` rule is *stronger* than `--yolo` (it's checked before
  `auto_approve` is ever consulted) and an `allow` rule is *weaker* than
  the default (skips the prompt even without `--yolo`) — two independent
  axes the old binary confirm-or-yolo model couldn't express. Pattern
  matching is a small hand-rolled `*`-wildcard glob (unit-tested), not a
  crate — the realistic cases (`"rm -rf*"`, `"*.env"`, `"*password*"`)
  don't need one.
  - Verified end-to-end together with the git tools: a
    `.graviton/permissions.toml` denying `run_shell` matching `"rm -rf*"`
    correctly blocked a mock model's attempt at that command *while
    running under `--yolo`*, with the exact rule cited back to the model
    as the reason — confirming the "stronger than yolo" property actually
    holds, not just reads that way in the rule file.

## Semantic search (`crates/cli/src/semantic.rs`)

Opt-in embedding-based retrieval, additive to (never a replacement for)
FTS5/symbol lookup. Set `grv config --embed-model <tag>` (any
embedding-capable model already pulled in Ollama — `nomic-embed-text` and
`all-minilm` are the common small ones), run `grv embed`, and every
retrieval path (`build_context`, used by `ask`/`investigate`/`crew`/
`swarm`/`mission`/`run`; `grv search --semantic`; `search_code`/
`semantic_search` as tools inside `grv run`/`mission` leaves) picks it up
automatically. Nothing here changes behavior for a repo that never opts in.

- **Storage**: one `embeddings` row per `content_fts` chunk (`chunk_id` =
  its `rowid`), vector as a little-endian `f32` BLOB. `graviton_indexer`
  deletes a file's stale embedding rows in the same transaction it replaces
  that file's `content_fts` rows on re-index (rowids aren't stable across a
  re-index, so without this an embedding would silently end up pointing at
  whatever chunk happens to land on that old rowid next); `clear_index`
  wipes `embeddings` for the same reason. `grv embed` is otherwise
  incremental — it only embeds chunks missing a *current-model* embedding,
  so switching `--embed-model` doesn't require `--force` to notice (the old
  model's rows just sit unused, matched by a `WHERE model = ?` on read).
- **Ranking**: cosine similarity — either an exact linear scan over every
  stored vector for the query's model, or, once `grv embed` has built one,
  an ANN (approximate nearest neighbor) index lookup. See "ANN index"
  below for when each path is taken and why both still exist.
- **Concurrency**: `grv embed`'s requests run concurrently, bounded by CPU
  thread count (2-8) via a plain `tokio::Semaphore` — embedding a short
  chunk is light but still contends for one resident model in Ollama, so
  this pipelines latency without pretending it's independent CPU-bound work
  (same honesty `resources.rs` applies to `swarm`/`mission`).

### ANN index (`crates/cli/src/ann.rs`)

The linear cosine scan is fine at the chunk counts a normal repo reaches —
single-digit milliseconds — but this tool is explicitly meant to hold up on
much larger ones, where an O(n) SQL load of every stored embedding's full
body text *and* an O(n) exact distance computation both stop being free.
`grv embed` now also builds a real ANN index: an HNSW graph via
[`instant-distance`](https://docs.rs/instant-distance), a pure-Rust
implementation with no FFI — deliberately chosen after this session's
tree-sitter grammar-linking pain (see "Language coverage"): one fewer
native library that can mismatch or fail to link later. It's serialized
(`postcard`) to `<index_dir>/ann_<model>.bin`, one file per embedding
model. Originally `bincode`, swapped after `cargo audit` flagged it
unmaintained (RUSTSEC-2025-0141 — the bincode team stopped all
development after a doxxing/harassment incident, and explicitly call
1.3.3 "complete" rather than defective; not exploitable as actually used
here either, since this file is only ever written and read by GRAVITON
itself, never untrusted input) — moved off it anyway while the swap was
still cheap (one file, one format, both sides already `serde`-based, no
other code changes needed) rather than waiting for a real bug nobody
would ever fix. `postcard` (56M+ downloads, actively maintained) is
pulled in with `default-features = false` — its default `heapless-cas`
feature (relevant only to embedded/no-atomics targets) would have quietly
added its own now-unmaintained transitive dependency
(`atomic-polyfill`, RUSTSEC-2023-0089) right back in.

- **The index is purely an accelerator, never a source of truth.** Every
  code path that reads it (`semantic::rank_by_query`) falls back to the
  exact linear scan on *any* problem — file missing, wrong dimensions
  (stale model switch), corrupt/unreadable encoding — via `ann::search`
  returning `Ok(None)` rather than an error. Nothing here can make a
  search *wrong*, only faster when it's available.
  `crates/cli/src/ann.rs`'s own unit tests build a real 3-vector index and
  assert the correct nearest match comes back, plus that a missing file
  and a dimension mismatch both degrade to `None` instead of panicking;
  re-verified live end-to-end after the `postcard` migration (a real mock
  embedding server, `grv embed`, `grv search --semantic`) since a
  serialization format swap is exactly the kind of change a passing unit
  test suite alone doesn't fully vouch for.
- **Rebuilt fully, every time `grv embed` runs — not incrementally.**
  `instant-distance` has no incremental-insert API; building means handing
  it the complete point set at once. Rather than track a separate
  version/hash to decide when a partial rebuild is safe, the invariant is
  kept deliberately simple: `ann::rebuild` runs at the end of every
  `embed_index` call over *every* currently-stored embedding for that
  model (not just what was newly embedded), so the file on disk, if
  present, is always an exact snapshot of `embeddings` as of the last
  successful `grv embed` — a query never needs to ask "is this stale?".
  Written to a sibling `.bin.tmp` then renamed into place, so a crash or a
  concurrent reader never sees a half-written index.
- **Only vectors + `chunk_id`s live in the index file — no duplicated body
  text.** This is the actual memory win for a huge repo: without it,
  ranking has to hold every chunk's full text in RAM to score it; with it,
  the HNSW walk only touches compact `(chunk_id, vector)` pairs, and a
  final `SELECT ... WHERE rowid IN (...)` hydrates just the handful of
  winning chunks' text — see `semantic::hydrate`.
- **`instant_distance::Point` is implemented on a small `CosinePoint`
  wrapper** using `1.0 - cosine_similarity` as the distance (HNSW is
  defined in terms of "smaller = nearer"), converted back to a familiar
  score when reporting hits.

### A real `Send` constraint shaped this module's (and `ann.rs`'s) API

`rusqlite::Connection` is `Send` but not `Sync`. Empirically (regardless of
*where* inside a function body the reference is actually used — before or
after an `.await`), an `async fn` with `&Connection` anywhere in its
signature has its returned future's `Send`-ness poisoned. That's fatal for
two real call paths here: `grv mission`'s leaves run inside
`Box::pin(... + Send)` (self-referential recursion needs that indirection —
see "Model tiers, `grv swarm`, and `grv mission`" above), and `grv serve`
spawns one task per connection via `tokio::spawn`, which requires `Send`
too.

The fix, applied consistently: split any operation that needs both a DB
read and a model call into a plain **sync** function that does the DB read
and returns owned data, and a separate **async** function that takes that
owned data and does the model call, with no `Connection`-shaped type
anywhere in its signature. Adding the ANN fast path meant widening what
counts as "prepared, owned data" from just `Vec<EmbeddedChunk>` to a small
`QuerySource` enum (`Ann { root, index_dir, model }` — cheap enough to
build without touching the DB at all, since it only needs a file-exists
check — or `Linear { model, chunks }`, the pre-ANN behavior unchanged):

```rust
pub fn prepare_query_source(conn: &Connection, root: &Path, index_dir: &str, model: &str) -> Result<QuerySource>  // sync
pub async fn rank_by_query(ollama_host: &str, query: &str,
                            source: QuerySource, limit: usize) -> Result<Vec<SemanticHit>>  // no Connection
```

`rank_by_query`'s `Ann` arm still needs a `Connection` to hydrate the
winning chunk ids' text — it opens its own short-lived one (via
`graviton_core::open_db`) *after* the one `.await` in the function (the
query-embedding call) has already completed, never holding it across an
`.await`. This is the same already-established idiom used by `agentic.rs`'s
`search_code`/`semantic_search` tool arms, not a new pattern.

The same split appears in `agentic.rs` (`prepare_search_tool` sync /
`finish_search_outcome` async, backing the `search_code`/`semantic_search`
tools) and `main.rs` (`build_context_sync` / `finish_context`, backing
`ask`/`crew`/`swarm`/`mission`/`run`'s context retrieval and reused as-is by
`grv serve`'s `ask` handler). A thin `async fn search(...)` convenience
wrapper that *does* take `&Connection` still exists in `semantic.rs` for
callers with no `Send` constraint (`grv search --semantic` is a plain
top-level `.await`, never spawned) — the poisoned signature is harmless
there, so callers aren't forced into the two-step split unless they
actually need it.

## `grv serve` — a daemon for editor/IDE integrations (`crates/cli/src/daemon.rs`)

A background process (foreground by default, like `ollama serve`) so an
editor/IDE integration gets code intelligence and agent answers without
paying a fresh `grv` process's startup cost — re-resolving config,
re-opening the index, losing `swarm`/`mission`'s warmed-up scheduler state —
on every request.

### Wire protocol: JSON-RPC 2.0, newline-delimited

One JSON object per line, in both directions — not LSP's `Content-Length`
header framing. JSON-RPC itself doesn't mandate a framing, and NDJSON needs
no header parser on either end: `nc -U .graviton/grv.sock` and a few lines
of Python/Node can already speak this.

```
--> {"jsonrpc":"2.0","id":1,"method":"status","params":{}}
<-- {"jsonrpc":"2.0","id":1,"result":{"root":"...","model":"qwen3:8b","embed_model":null,"index":{"files":42,"symbols":310,"chunks":198,"embedded":0},"ollama_reachable":true,"ollama_models":[...],"scheduler_target":3}}

--> {"jsonrpc":"2.0","id":2,"method":"ask","params":{"question":"what does check_password do?","agent":"architect"}}
<-- {"jsonrpc":"2.0","id":2,"result":{"agent":"architect","model":"qwen3:8b","answer":"..."}}

--> {"jsonrpc":"2.0","id":3,"method":"shutdown"}
<-- {"jsonrpc":"2.0","id":3,"result":"ok"}
```

An error is `{"jsonrpc":"2.0","id":...,"error":{"code":-32000,"message":"..."}}`
— one generic code for now (an unknown method, a missing required param, no
index yet, no embedding model configured); there's no need for a finer
error taxonomy until a real client shows up wanting to branch on one.

**Methods**: `status` (index stats + Ollama reachability + scheduler
target), `agents` (roster: key/display/tagline/tier), `search` (FTS,
`{query, limit}`), `symbol` (`{name, limit}`), `semantic_search` (`{query,
limit}`, errors clearly if no embed model/embeddings), `ask` (`{question,
agent?, files?, stream?}`, through the same `prepare_ask` +
`run_read_only_loop_with` every other agentic-retrieval command uses —
`search_code`/`semantic_search`/etc. mid-answer included), `review`
(`{range?, staged?, agent?, stream?}`, single-agent over a real `git diff`,
not the full `grv review` crew pipeline — kept fast for an editor round
trip), `run_start`/`run_attach`/`run_confirm`/`run_status` (a full
checkpointed `grv run` agentic session over the socket — see below),
`shutdown` (acks then exits; also removes its own socket file so a clean
shutdown never needs the stale-socket recovery path below).

### Streaming (`ask`/`review` with `"stream": true`)

Ordinarily `ask`/`review` return one `{"id":...,"result":{...}}` line once
the whole answer is ready. With `"stream": true` in `params`, the same
request instead gets zero or more notifications first:

```
--> {"jsonrpc":"2.0","id":5,"method":"ask","params":{"question":"...","stream":true}}
<-- {"jsonrpc":"2.0","method":"tool_call","params":{"id":5,"name":"semantic_search","arguments":{...}}}
<-- {"jsonrpc":"2.0","method":"token","params":{"id":5,"text":"Password"}}
<-- {"jsonrpc":"2.0","method":"token","params":{"id":5,"text":" verification"}}
<-- {"jsonrpc":"2.0","id":5,"result":{"agent":"architect","model":"qwen3:8b","answer":"Password verification ..."}}
```

...ending in the exact same final `result` line the non-streaming path
sends, so a client that ignores notification-shaped lines (no `id`,
instead a `method` naming the notification) still gets the complete
answer either way. Implemented in `handle_streaming`: an `mpsc::
unbounded_channel` carries `Token`/`ToolCall` events out of the plain sync
closures `run_read_only_loop_with` calls (`on_token`/`on_tool_call` can't
themselves be `async`, so they can't write to the socket directly), and
`tokio::select!` interleaves draining that channel with polling the loop's
future to completion.

### Driving a full `grv run` session over the socket

`run_start {task, agent?, yolo?, browser?, files?}` returns
`{"session_id": "..."}` immediately — the actual `agentic::run` loop runs
as an independent `tokio::spawn`ed task inside the daemon, *not* tied to
the connection that started it, using a `ChannelIo` (see "A pluggable
confirm/output sink" above) instead of a terminal:

- **`run_attach {session_id}`** (on any connection, including the one that
  called `run_start`) acks once, then replays that session's **entire**
  event history from `run_start` onward — not just what happens after
  attaching — as `run_event` notifications (`{"session_id","kind":
  "output"|"token"|"confirm_request"|"ask_choice"|"done", ...}`), then
  continues live until `done` or the connection drops. Multiple
  connections can attach to the same session independently, each getting
  its own full replay regardless of when it attaches.
- **`run_confirm {session_id, decision}`** (`"yes"`/`"no"`/anything else =
  a `Decision::Redirect` — same three-way semantics the terminal's y/n/
  free-text prompt has) feeds a decision into the session's confirm
  channel from *any* connection, unblocking `ChannelIo::confirm`.
- **`run_answer_choice {session_id, selected: [...]}`** is the same idea
  for an `ask_user` tool call (an `ask_choice` event) — `selected` is the
  chosen option string(s), fed into `ChannelIo::ask_choice`'s channel.
- **`run_status {session_id}`** returns a point-in-time snapshot (running?,
  a pending confirm's text or pending `ask_user` question if any, the
  checkpoint id once known, how it finished) without needing to attach —
  for a quick poll that doesn't want the full event log.

**Full replay, not just a race-safety patch.** `RunHandle` keeps an
authoritative, ordered, append-only `history: Arc<Mutex<Vec<RunEvent>>>`
for the session — every `Output`/`Token`/`ConfirmRequest`/`AskChoice`/
`Done` a run ever produces, not only the latest pending one. The broadcast
channel (`events_tx`) carries **no payload at all**, just a `()` wake-up
ping; `handle_run_attach` subscribes to it first (for the same
race-safety reason as before — nothing pushed after this point can be
missed), then loops: drain `history[next_idx..]` and send each as a
notification, and when a ping arrives (or the channel reports `Lagged`,
meaning some pings coalesced), just re-drain from `next_idx` again. This
is what actually closes the "only replays from attach time forward" gap,
and it subsumes the previous per-field catch-up hack (which separately
special-cased an already-pending confirm, an already-pending `ask_user`
question, and an already-finished run) with one mechanism: a `Lagged` or
even entirely missed ping costs nothing, because the next drain always
re-reads the authoritative log from exactly where this attach left off,
never from the channel's payload. `history` is bounded by one run's own
output (a fresh `run_start` gets a fresh, empty log), not daemon uptime.

Verified against the actual gap, not just reasoned about: a mock chat
server streaming ten words with a 300ms gap between each, `run_start`ed,
then deliberately left with **nobody attached** for 1.5 seconds (several
tokens streaming into the void), then attached from a brand-new
connection — the full original sentence came back from the very first
word, not just whatever streamed after the late attach. The pre-existing
confirm-race test (a mock run reaching its confirm point faster than a
second process could attach, `run_confirm`ed from a third, independent
connection) still passes under the new mechanism too.

### Framing/lifecycle details

- **Concurrency**: each accepted connection is its own `tokio::spawn`ed
  task (`handle_conn`), generic over `AsyncRead + AsyncWrite` so the same
  handler serves both the Unix listener and the optional `--tcp` one.
  Model-calling methods acquire a permit from a `LiveScheduler` sized from
  `cfg.distinct_models()` (same mechanism as `grv swarm`/`mission`, capped
  at `DAEMON_HARD_CAP = 6` regardless of RAM headroom — an editor is one
  human's queries, not a swarm, so there's no reason to let it queue more
  than a handful at once).
- **Stale socket recovery**: a crashed daemon leaves its socket file
  behind, which makes `bind` fail with "address in use" even though nothing
  is listening. `remove_stale_socket` tries `UnixStream::connect` first (a
  live daemon accepts) and only unlinks the file if that fails — so it
  never removes a socket a real running daemon still owns.
- **Unix socket permissions, explicitly**: `bind_unix_socket_locked_down`
  chmods the socket to `0600` right after binding, rather than trusting
  `bind`'s default (mode `0777` masked by the calling process's *umask*,
  not a fixed value). Every method reachable over this socket includes
  `write_file`/`run_shell`/`recon_tool` via `run_start` — so "filesystem
  permissions are the trust boundary" (asserted throughout this doc) is
  only actually true if that boundary is unconditional. A typical umask
  (022) happens to leave the socket non-writable — and therefore
  non-connectable, under Linux's Unix-socket permission model — for
  group/other anyway, which is exactly why this had gone unnoticed: the
  common case was already accidentally safe. An unusual umask (0, or a
  shared directory's inherited ACL) would not be, silently, with no
  symptom until someone actually tried connecting from another local
  account. Verified directly, not just reasoned about:
  `unix_socket_is_locked_down_even_under_a_permissive_umask` binds under
  a real `umask(0)` (via a small direct `libc` `extern "C"` declaration —
  not worth a whole crate for one syscall) and asserts the resulting mode
  is `0600` regardless; a live daemon started under `umask 000` was
  additionally confirmed to produce a `0600` socket on disk.
- **Unix socket path length**: `sockaddr_un.sun_path` is a real OS limit
  (~108 bytes on Linux, ~104 on macOS/BSD), not a GRAVITON one — this
  surfaced during testing when a scratch path under a long tmp directory
  hit it directly. Handled two ways rather than left as a thing to hit and
  work around: (1) the **default** socket path (no `--socket` given) is no
  longer under the repo at all — `default_socket_path` hashes the
  canonicalized repo root + index dir into a short, stable name under
  `$XDG_RUNTIME_DIR/grv/` (falling back to the system temp dir), so a
  deeply nested repo path never reaches the limit in the common case; (2)
  `check_socket_path_len` rejects an explicit `--socket` that's still too
  long up front, with a message suggesting a short path (e.g. `/tmp/
  grv.sock`), instead of letting `bind` fail with the OS's bare "path must
  be shorter than SUN_LEN".
- **`--tcp` token auth**: `serve()` always requires one once `--tcp` is
  given — `generate_token()` draws 32 bytes from the OS CSPRNG
  (`getrandom`, hex-encoded to a 64-char/256-bit token) unless
  `--tcp-token` sets one explicitly, printed once at startup. This used to
  be a plain `DefaultHasher` over wall clock + pid + a per-process
  counter — not actually a secret, since `DefaultHasher` uses a fixed,
  known key and every input feeding it was either guessable (pid, a
  counter starting at 0) or coarse enough to narrow down (a daemon-start
  timestamp); someone who could bound the startup window and pid range
  could search the *real* keyspace directly instead of the nominal 128
  bits it looked like. `handle_conn` checks `params.token` against the
  real token via `constant_time_eq` (a fixed-time byte comparison — `==`
  on `&str` short-circuits on the first differing byte, which leaks how
  many leading bytes were right through response timing) for every
  request on a TCP-accepted connection (`is_tcp`, threaded through from
  which listener accepted it) before dispatching — including `shutdown`,
  so an unauthenticated TCP client can't stop the daemon either. A wrong
  or missing token also costs the caller a flat 250ms before the error
  comes back, cheap insurance against a scripted guesser hammering the
  port.
- **`--tcp` transport encryption** (`crates/cli/src/tls.rs`): the token
  above used to travel in cleartext — encrypted now, every `--tcp`
  connection required to be TLS 1.3, via `rustls`/`tokio-rustls` (both
  already in the dependency tree through `reqwest`, so this added no new
  crypto backend to the build, just a server-side use of one already
  there). `ephemeral_server_tls()` generates a fresh self-signed
  certificate (via `rcgen`) on every `grv serve --tcp` start and prints
  its SHA-256 fingerprint alongside the token — since a private-IP
  certificate has no CA to validate against, this is trust-on-first-use,
  the same model an SSH host key uses: the operator hands the fingerprint
  to whoever connects, out of band, for them to pin. `serve()`'s TCP
  accept loop wraps each accepted `TcpStream` in `TlsAcceptor::accept`
  before it ever reaches `handle_conn` — a failed handshake (including a
  plain garbage/plaintext connection attempt) never reaches the JSON-RPC
  layer at all, logged and dropped. Verified live, not just unit-tested:
  a real Python TLS client connecting to a running daemon negotiated
  TLS 1.3, and the fingerprint read from the actual handshake matched
  what the daemon printed at startup exactly; a plaintext connection
  attempt to the same port got a TLS alert (`InvalidContentType`) instead
  of ever reaching request handling. Unix socket connections still never
  need a token or TLS (filesystem permissions on the socket file are that
  boundary instead, same trust model `ollama serve`'s own socket uses) —
  this is additive to `--tcp` specifically, not a new requirement
  elsewhere.

## What's explicitly *not* built yet (see README roadmap)

- Call-graph *full* type/scope resolution — still deliberately name-based
  for the call match itself; a real import resolver exists now for Rust/
  Python/JS/TS/TSX/Go (see "Import resolution" above,
  `ResolutionHint::ImportResolved`), but it's path-based, not a symbol
  table or type checker — it can't model shadowing, visibility, or
  re-exports, and 40+ other parsed languages have no import resolver yet.
  True type resolution (which overload/trait impl a call targets) remains
  on par with what a language server spends its whole existence on, not a
  query tweak.
- Svelte/Vue symbol extraction — both are now genuinely parsed (see
  "Previously-blocked grammars" above), but their own grammars expose a
  `<script>` block as one opaque text node; recovering real definitions
  from it needs a second, injection-based parse this project's
  one-query-per-language design doesn't do.

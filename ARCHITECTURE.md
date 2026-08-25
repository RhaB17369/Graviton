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
  16GB-RAM box. Every extra resident model is RAM you don't have for qwen3.

Semantic (embedding-based) search is a legitimate v2 addition once lexical +
symbol retrieval proves insufficient in practice — it's called out in the
README roadmap rather than built speculatively now.

### Schema

```sql
files(id, path UNIQUE, lang, size, mtime, hash)
symbols(id, file_id -> files, kind, name, start_line, end_line, parent)
content_fts(path, start_line, end_line, kind, name, body)   -- FTS5 virtual table
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

## What's explicitly *not* built yet (see README roadmap)

- Call-graph edges (`callers`/`callees` beyond a name-match placeholder) —
  needs a second tree-sitter pass resolving call-expression targets against
  the symbol table, including cross-file resolution. Nontrivial, deferred.
- Agentic loop where the model can issue its own `search`/`symbol` calls
  mid-answer instead of getting one fixed context injection. This is the
  highest-value next step for `investigate` on large repos, structured as
  an explicit roadmap item rather than half-implemented now.

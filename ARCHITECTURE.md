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

## Language coverage (v0.2)

Each parsed language is one tree-sitter grammar crate + one `def_query_src`
entry in `crates/indexer/src/lang.rs`. Adding a language is: find its
grammar on crates.io, pull its `node-types.json` to find the right node/field
names for function/class/method definitions, write the query, and let the
existing "best-effort, never fatal" machinery handle version drift.

Two real constraints surfaced while adding the v0.2 batch (Java, C#, PHP,
Ruby, Bash, Lua, Solidity, PowerShell):

- **Node/field names must be looked up per grammar, not assumed.** They
  don't follow one convention: Ruby's `class`/`module` nodes *do* expose a
  `name:` field (unlike what a naive guess suggests), while Kotlin and
  PowerShell's grammars expose the name as a bare positional child with no
  field label at all (`(class_declaration (type_identifier) @name)` instead
  of `(class_declaration name: (type_identifier) @name)`). Get this wrong
  and the query still compiles — it just silently matches nothing, which is
  why every language here was verified against a real sample file, not just
  compiled.
- **Cargo's `links` uniqueness bites grammar crates that lag upstream.**
  `tree-sitter-kotlin` (0.3.8, the latest release on crates.io) depends on
  tree-sitter 0.21/0.22; every other grammar here depends on 0.26. Cargo
  will not link two versions of a native library that both declare `links =
  "tree-sitter"` into one binary, so Kotlin can't be added as a parsed
  language without vendoring a patched grammar crate — not worth it for one
  language when the file is still fully searchable either way. This is the
  actual failure mode to expect when adding more languages later, not a
  one-off: check what tree-sitter core version a candidate grammar crate
  pins before writing its query.

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

`crates/cli/src/agents.rs` defines four `AgentSpec`s (key, display name,
tagline, system prompt): ARCHITECT (high-level programming), SENTINEL
(defensive security), REAPER (offensive security), and SINGULARITY (a
coordinator with no first-hand analysis of its own — it only synthesizes
the other three's actual output).

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
  reads all three real outputs before writing a decision brief. This is the
  one part of the system where output literally becomes another call's
  input — everywhere else in GRAVITON, retrieval and generation are one
  fixed pass (see the next section for why, and the roadmap for what a
  fuller agentic loop would add).

`grv investigate` is orthogonal to the agent roster: it's a structured
*output format* (`agents::INVESTIGATE_FORMAT`, appended to whichever
agent's base prompt) rather than its own persona, so
`grv investigate --agent sentinel "..."` is meaningful (structured
defensive audit) distinct from plain `grv ask --agent sentinel` (free-form).

Context retrieval (`build_context`) is shared and computed once per
`ask`/`investigate`/`crew` invocation — every agent in a `crew` run
currently sees identical retrieved chunks, not a query tailored to its own
specialty. That's a deliberate v1 simplification (documented in the README
roadmap), not an oversight: giving each agent its own retrieval query is
straightforward to add once it's clear the shared-context version is
actually leaving relevant code unfound in practice.

## Model tiers and `grv swarm`: real multi-model, resource-aware

The 4-agent (later 14-agent) design above runs everything through one
model — accurate for the default config, but the tiering added in v0.3
of this document makes GRAVITON capable of genuinely running more than
one model, sized to what the machine can actually hold:

- `Config.model` is the `Standard` tier and always the fallback.
  `model_fast`/`model_deep` are optional overrides (`grv config
  --model-fast qwen2.5:1.5b --model-deep qwen3:14b`); leave them unset and
  nothing changes from v0.2's one-model behavior. Each `AgentSpec` declares
  which tier it wants — see the table `grv agents` prints.
- **`grv crew`** stays sequential by design (§ above) — its whole point is
  real hand-off, so parallelizing it would just make each stage read a
  half-finished prior stage. Multi-model there means *different-sized*
  models per stage, not concurrent ones.
- **`grv swarm --agents a,b,c "question"`** is the actually-concurrent
  mode: independent agents (no hand-off — this is for questions that don't
  need one, e.g. running SENTINEL/REAPER/CRYPTOGRAPHER over the same
  question at once) fired via `tokio::task::JoinSet`, each against its own
  tier's model, gated by a `tokio::sync::Semaphore` sized by
  `resources::safe_concurrency`.
- **The concurrency number is computed, not assumed.** `crates/cli/src/
  resources.rs` reads total system RAM (`sysinfo`) and each configured
  model's on-disk size (`OllamaClient::model_sizes_mb`, via `/api/tags`),
  reserves a 30% safety margin for the OS/KV-cache growth, and packs as
  many of the *distinct* configured models as fit — e.g. 16GB RAM with a
  1.5GB fast model and a 5GB standard model comfortably fits both (2-way
  concurrency); three 8B models would not, so `swarm` would cap at 1-2
  instead of pretending otherwise. `--max-parallel` overrides the estimate
  for a user who knows their actual headroom better than a heuristic does.
- **What this doesn't pretend to solve**: CPU threads (6C/12T on the
  reference hardware) are shared across whatever runs concurrently — three
  agents generating at once each get roughly a third of the cores, so each
  individual stream is slower than it would be alone. The win is wall-clock
  for the *batch*: three independent questions that would otherwise run one
  after another finish closer to together. `grv swarm` prints the capacity
  note up front so this trade-off isn't hidden.
- Ollama itself owns actual model residency (loading on first request,
  evicting LRU under memory pressure) — GRAVITON's estimate exists only to
  pick a sane concurrency *before* asking Ollama to do anything, not to
  duplicate Ollama's own memory management.

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
context-lined diff), `delete_file`, `run_shell`, and `recon_tool` (the
`grv tool` whitelist, exposed as a tool so the agent can run nmap/ffuf/etc.
itself instead of the user doing it out-of-band). `--browser` adds
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

- **Confirmation** (`agentic::confirm`) gates `write_file`, `edit_file`,
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

## What's explicitly *not* built yet (see README roadmap)

- Call-graph edges (`callers`/`callees` beyond a name-match placeholder) —
  needs a second tree-sitter pass resolving call-expression targets against
  the symbol table, including cross-file resolution. Nontrivial, deferred.
- Agentic loop where the model can issue its own `search`/`symbol` calls
  mid-answer instead of getting one fixed context injection. This is the
  highest-value next step for `investigate` on large repos, structured as
  an explicit roadmap item rather than half-implemented now.

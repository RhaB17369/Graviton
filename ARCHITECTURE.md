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

`grv mission` does not (yet) share this — its tree-shaped, concurrent
execution doesn't map onto one linear transcript the way `grv run`'s
single-agent loop does, so resuming a partially-completed mission is left
as future work rather than half-implemented.

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

## What's explicitly *not* built yet (see README roadmap)

- Call-graph edges (`callers`/`callees` beyond a name-match placeholder) —
  needs a second tree-sitter pass resolving call-expression targets against
  the symbol table, including cross-file resolution. Nontrivial, deferred.
- Agentic loop where the model can issue its own `search`/`symbol` calls
  mid-answer instead of getting one fixed context injection. This is the
  highest-value next step for `investigate` on large repos, structured as
  an explicit roadmap item rather than half-implemented now.

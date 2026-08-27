# GRAVITON

> A graviton is theoretically the particle that carries gravity — vanishingly
> small, disproportionate force. That's the bet here: an 8B local model on a
> 4GB-VRAM laptop, made to punch above its weight with a Rust engine that
> does the deterministic work (parsing, indexing, retrieval) so the model
> only has to reason.

`grv` is a local, offline multi-agent framework specialized in high-level
programming, defensive security, and offensive security — for huge
codebases and CTF/HTB/THM/OSCP-style lab work, built for machines that
can't run a 30B+ model comfortably. It talks to [Ollama](https://ollama.com)
— no cloud, your code never leaves the machine.

"Multi-agent" here means real specialist personas that hand off actual
work product to each other, not a handful of names for one prompt.
22 agents across five categories (`grv agents` prints the live roster with
taglines):

| Category | Agents |
|---|---|
| Programming | ARCHITECT (design/refactoring), TESTER, DEBUGGER, PERFORMANCE |
| Infrastructure & engineering | NETARCH, DEVOPS, CLOUDARCH, DEPLOY, AUTOMATOR, DOCUMENTOR |
| Defensive security | SENTINEL (general audit), CRYPTOGRAPHER, QUANTUMCRYPTO, SUPPLYCHAIN, CLOUDSEC, IDENTITY |
| Offensive security | REAPER (general exploit dev), WEBHUNTER, BINEXP, ADVERSARY, OSINT |
| Coordinator | SINGULARITY — no analysis of its own, synthesizes whichever agents ran |

That's a curated set, not padding — each one covers a genuinely distinct
failure mode or skill (crypto misuse is a different read than IDOR is
different from a binary overflow; network architecture is a different
question than cloud-cost architecture), and the roster is data-driven
(`crates/cli/src/agents.rs`) so adding one more is copy-paste-edit, not a
redesign. Depth over headcount: a roster of near-duplicate personas would
just make `grv crew`/`swarm`/`mission` slower without finding anything more.

Ask one directly (`grv ask --agent webhunter "..."`), or run a crew — the
default pipeline is ARCHITECT → REAPER → SENTINEL → SINGULARITY, or pick
your own: `grv crew --agents cryptographer,identity,singularity "audit the auth module"`.
Every stage reads the *actual output* of the ones before it, not just the
same raw context re-served — that hand-off is what makes it a pipeline
instead of running the same question N times.

By default they all run on the one local model configured in `grv config`
— the specialization comes from distinct system prompts and (in `crew`) a
real sequential hand-off, not separate brains. But each agent also carries
a `ModelTier` (`grv agents` shows it in brackets: `fast`/`standard`/`deep`),
and if you configure `model_fast`/`model_deep` overrides, GRAVITON actually
runs more than one model — sized to what your RAM can hold, not just
assumed:

```sh
grv config --model qwen3:8b --model-fast qwen2.5:1.5b --model-deep qwen3:14b
grv swarm --agents sentinel,reaper,cryptographer "audit auth.rs"
```

`swarm` (unlike `crew`) has no hand-off — it fires independent agents
*concurrently*, each on its own tier's model, gated by a concurrency
number `grv` actually computes (RAM ÷ model size × Ollama's per-model
parallel-request capacity, not a fixed guess) and keeps *re-computing
live* for the whole run — `grv status` shows the same estimate, plus
what else on your machine is currently eating that RAM. `--max-parallel`
overrides it.

For a task too big to hand a fixed agent list, **`grv mission`** lets a
planner call decide the breakdown itself, recursively:

```sh
grv mission "harden this API service end to end" --max-depth 2
```

A planner call splits the task into subtasks assigned to whichever
specialists fit, each subtask can decompose *again* up to `--max-depth`
(a subtask the planner judges already atomic short-circuits straight to a
leaf, so depth adapts to the task instead of always maxing out), and
results synthesize back up the tree. One rule holds no matter how wide or
deep it gets: every model call anywhere in the tree — leaves, every
planner call, every synthesis step — shares one live-resampled concurrency
gate, so a mission that fans out into 15 subtasks can't put more
concurrent model calls on the machine than its RAM can actually take at
that moment; the gate grows back into headroom as earlier subtasks finish.

This is the honest version of "multiple agents at once, as many as the
machine can take": real concurrent requests, bounded by what a 16GB laptop
can hold resident and re-checked continuously rather than assumed once at
startup — with CPU threads still shared across whatever's running (see
ARCHITECTURE.md for the trade-off spelled out).

**`grv mission` is resumable too**, the same idea as `grv run --continue`
but tracking the whole task tree instead of one flat conversation — every
node's status (pending/done/failed), its result, and the exact subtask
split a planner call chose are checkpointed as they happen, keyed by tree
position:

```sh
grv mission --continue                          # resume the most recent mission, no new instruction
grv mission --continue --session <id>           # resume a specific (not the latest) mission
```

A resume never re-asks the planner for nodes that already finished — a
`Done` leaf or synthesis short-circuits straight to its cached result, and
a node that was already split into subtasks reuses that exact split rather
than re-decomposing (re-decomposing could produce a differently-shaped
tree and orphan children that already finished). Only a genuinely
unfinished node — e.g. one that failed because a model call errored, or
because you killed the process mid-run — actually does new work on
resume. `--max-depth` is remembered from the original run too, so a resume
without an explicit `--max-depth` doesn't silently fall back to the
default and cause an already-terminal node to spuriously decompose again.

### `ask`/`crew` (read-only) vs. `run` (acts on your project)

`ask`, `investigate`, and `crew` are analysis only: they retrieve context
and answer in text, never touching disk. **`grv run` is the mode that
actually acts** — the agent gets real tools and a loop, like Claude Code:

```sh
grv run "add input validation to the login form and fix any bug you find"
grv run --browser "add a /health endpoint to the Express app, then curl it and confirm it returns 200"
grv run --agent reaper --yolo "fuzz the login endpoint for SQLi and write a PoC script"
```

Tools available to the agent: `read_file`, `list_dir`, `write_file`,
`edit_file`, `delete_file`, `run_shell`, `run_tests` (auto-detects
`cargo test`/`npm test`/`pytest`/`go test`/`bundle exec rspec` from the
repo, or takes an explicit `command`, and returns a parsed pass/fail
summary with the specific failing tests instead of raw noise — the agent
is told to run this after a change, before declaring the task done),
`git_status`/`git_diff`/`git_log` (real git state, not guessed from
separate file reads) and `git_commit`, `recon_tool` (the same whitelisted
nmap/ffuf/etc. as `grv tool`), `web_search`/`web_fetch` (DuckDuckGo-backed,
no API key — so the agent checks a current API/CVE/best-practice instead of
answering from a possibly-stale training snapshot), and — with `--browser`
— `browser_navigate`, `browser_eval`, `browser_screenshot`, `browser_console`,
driving the system's headless Chromium via CDP so the agent can actually
load a page, run JS in it, and see what broke, not just guess from source.

Every `write_file`/`edit_file`/`delete_file`/`run_shell`/`run_tests`/
`git_commit`/`recon_tool` call is shown to you and confirmed before it
happens — you see the exact diff or command — unless you pass `--yolo` for
a fully autonomous run. `git_commit` isn't checkpointed the way file writes
are: a commit is already its own undo point (`git reset`/`git revert`), so
GRAVITON doesn't duplicate that. File changes are checkpointed regardless
of `--yolo`:

```sh
grv checkpoints                # list grv run sessions and what they touched
grv rollback                   # undo the most recent session's file changes entirely
grv rollback <session-id> --to 3   # undo everything after step 3 in that session
```

Rollback covers file writes/edits/deletes only — a `run_shell` call's side
effects (installing a package, starting a server) aren't generically
undoable, which is exactly why it's confirmed up front instead of promising
an undo it can't deliver.

**A confirmation prompt isn't just yes/no** — type anything else instead
and it's read as a redirect: the write/edit/shell call is declined *and*
what you typed is fed back to the model as the reason, so "no, use a
different approach" actually reaches the next turn instead of only being
expressible as a blind refusal.

**The agent can ask *you* a question, too.** Every `ask`/`run`/`crew`/
`review`/`swarm`/`mission` agent has an `ask_user` tool: instead of
guessing at an ambiguous instruction, it can stop and present you with a
question plus a short list of options (single- or multi-select) — "which
of these three auth flows should I harden first?" — and get a real answer
back into the same turn before continuing, rather than silently picking
one or asking you to re-run with more detail. On the terminal this is a
numbered prompt (comma-separated picks, or `all`); over `grv serve` it's
an `ask_choice` event a client answers via `run_answer_choice` (see
"`grv serve`" below) — so an editor integration can render it as an actual
checkbox/radio UI instead of a text prompt.

**Every `grv run` is resumable**, not just the ones you remembered to plan
for — the full conversation (including every tool call and result) is
saved to the session automatically:

```sh
grv run --continue                              # resume the most recent session, no new instruction
grv run --continue "also add a test for this"   # resume + give it one more thing to do
grv run --continue --session <id> "..."         # resume a specific (not the latest) session
```

For anything beyond a one-shot task, the agent is told to maintain a
visible plan (`update_plan`, shown live as `[ ]`/`[~]`/`[x]` and saved with
the session):

```sh
grv plan                # show the most recent session's current plan
grv plan <session-id>   # show a specific one
```

**Fine-grained permissions** go under `--yolo`/confirm, not instead of it —
drop a `.graviton/permissions.toml` in the repo to make specific tool
calls *stronger* than `--yolo` (never allowed, no override) or *weaker*
than the default (never prompted, even without `--yolo`):

```toml
[[rule]]
tool = "run_shell"
pattern = "rm -rf*"
action = "deny"      # blocked even under --yolo

[[rule]]
tool = "web_search"
action = "allow"     # never prompts, even without --yolo
```

Rules are checked in file order, first match wins (`tool = "*"` matches
anything, `pattern` is a small `*`-wildcard glob against the call's path/
command); a call that matches nothing behaves exactly as before.

### `grv review` — real git diffs, not retrieved context

`ask`/`crew` answer from *indexed* chunks that happen to match a question.
For "review what I just changed," that's the wrong source — you want the
actual diff:

```sh
grv review                       # every uncommitted change (staged + unstaged) vs HEAD
grv review --staged              # just what's staged
grv review main..HEAD            # a specific range
grv review --agents cryptographer,singularity   # custom pipeline, same as crew
```

Runs the same sequential hand-off pipeline as `crew` (default:
`sentinel,architect,singularity`) over the real diff text instead of FTS
retrieval — each stage cites the actual changed lines, not a symbol that
happened to match a keyword.

Instead of stuffing an entire repository into a context window (impossible
past a few hundred KLOC anyway), GRAVITON indexes the repo once with
tree-sitter + SQLite/FTS5, then retrieves only the symbols and chunks
relevant to each question and hands the model a bounded, cited context.

```
   your repo                 grv index                  grv ask/investigate
 ┌───────────┐         ┌──────────────────────┐        ┌─────────────────────┐
 │  100K+     │  ──▶   │ tree-sitter symbols   │  ──▶   │ retrieval           │
 │  files     │        │ + FTS5 line chunks    │        │ + budgeted context  │
 └───────────┘         │ (.graviton/index.db)  │        │ + Ollama (qwen3:8b) │
                        └──────────────────────┘        └─────────────────────┘
```

## Install

```sh
cd graviton
cargo build --release
sudo install -m755 target/release/grv /usr/local/bin/grv   # or add target/release to PATH
```

Requires a running Ollama daemon (`ollama serve`) and a pulled model
(`ollama pull qwen3:8b` is the recommended default for 16GB RAM / 4GB VRAM
machines — see ARCHITECTURE.md for the sizing rationale).

## Usage

```sh
cd /path/to/some/huge/repo

grv index                          # build/update the local index (incremental)
grv index --watch                  # ...then keep re-indexing on file changes
grv search "jwt verify"            # full-text search over indexed chunks
grv symbol validate_token          # jump straight to a symbol's source
grv callers validate_token         # every call site named validate_token
grv callees handle_login           # every call made from within handle_login

grv agents                         # show the roster
grv ask "trace user input into os.system in vuln.py"              # ARCHITECT by default
grv ask --agent sentinel "audit auth.rb for auth bypass"
grv ask --agent reaper --file exploit.c "write a working PoC for this overflow"
grv investigate "is this deserialization path exploitable?"       # REAPER by default, structured output
grv crew "is Vault.sol safe to deploy?"                            # full pipeline, all four agents
grv swarm --agents sentinel,reaper,cryptographer "audit auth.rs"  # independent agents, concurrent
grv mission "harden this API service end to end" --max-depth 2    # recursive planner + execution
grv review                                                        # crew review of your actual uncommitted diff

grv status                         # index stats + Ollama connectivity + live capacity/top-consumers
grv config --model qwen3:8b --num-ctx 8192 --host http://127.0.0.1:11434
grv config --model-fast qwen2.5:1.5b --model-deep qwen3:14b        # opt into real multi-model
grv config --embed-model nomic-embed-text && grv embed             # opt into semantic search
grv serve                                                          # daemon for editor/IDE integrations
```

`ask`/`investigate` take `--agent <architect|sentinel|reaper|singularity>`
(defaults: architect / reaper respectively) and stream the answer token by
token. `investigate` additionally structures the output as a fixed format
(symbols found → data flow → analysis → concrete next step/PoC) regardless
of which agent runs it. `crew` runs several agents in sequence — pass
`--agents architect,reaper` to run a subset/reorder — printing each stage
as it streams; expect a full four-stage crew run to take several minutes on
an 8B CPU-bound model, since it's four full generations, not one.

### Recon/security tools

`grv tool` runs a whitelisted recon tool, streams its output live like
running it directly, and indexes it so it's immediately part of the
codebase's searchable context — the "nmap → LLM → next step" loop:

```sh
grv tool run nmap -- -sV -p- 10.10.10.5
grv tool run ffuf -- -u https://target/FUZZ -w wordlist.txt
grv ask "based on the nmap scan, what should I try first?"

grv tool list                      # whitelist + recent runs in this repo
grv tool show 3                    # full output of run #3

# already have output from somewhere else (Burp export, a tool not on
# the whitelist, whatever)? index it without re-running anything:
cat scan.txt | grv tool ingest nmap "initial external scan"
```

Whitelist (`grv tool list` always shows the current one): nmap, masscan,
rustscan, ffuf, gobuster, dirb, wfuzz, feroxbuster, nikto, whatweb, wpscan,
nuclei, httpx, sqlmap, hydra, medusa, john, hashcat, subfinder, amass, dnsx,
dig, whois, curl, nc/ncat/netcat, enum4linux, smbclient, smbmap,
crackmapexec/netexec, searchsploit. This is a launcher + logger, not a
sandbox — it runs exactly what you typed with your own permissions, same as
typing it in the shell; the whitelist just keeps `tool run` scoped to recon
tools rather than becoming a second shell.

### Custom tools — extend `grv run` without recompiling

`agentic.rs`'s tool roster (read/write/shell/recon/web/browser) is fixed in
the binary, but you're not limited to it: drop a TOML file in
`~/.config/graviton/tools/` (every project) or `.graviton/tools/` (this
project only, shareable via the repo) and it's a tool the very next
`grv run` invocation:

```sh
grv custom new docker_ps          # scaffolds .graviton/tools/docker_ps.toml
grv custom list                   # every loaded custom tool + where it's from
grv custom show docker_ps         # the exact schema the model would see
```

A custom tool is a named, described, parameterized shell command
template — `{{param}}` gets substituted with that argument's value
(shell-quoted for you, so a value like `it's; rm -rf /` lands as one inert
literal argument, not a second command):

```toml
name = "docker_ps"
description = "List running Docker containers, optionally filtered by name"
command = "docker ps {{flags}}"

[[params]]
name = "flags"
description = "extra docker ps flags, e.g. '-a' for stopped containers too"
required = false
default = ""
```

Under the hood it's still a shell command, so it goes through the exact
same confirm-before-running gate as `run_shell` (or runs straight through
under `--yolo`) — a custom tool is a friendlier, self-documenting name and
argument schema for the model, not a new trust boundary.

### Semantic search — optional, opt-in, additive

FTS5 (`grv search`/`grv ask`/etc. by default) matches tokens, not meaning —
a question about "credential verification" won't find a function called
`check_password` unless the words actually overlap. Semantic search closes
that gap with embeddings, entirely opt-in:

```sh
grv config --embed-model nomic-embed-text   # any embedding-capable model you've pulled
grv embed                                   # embed every indexed chunk (incremental — re-run after `grv index`)
grv search "credential verification logic" --semantic
```

Once both are set up, `ask`/`investigate`/`crew`/`swarm`/`mission`/`run`
automatically add a semantic retrieval pass alongside FTS/symbol hits — no
flag needed, and silently skipped (not an error) if you never ran `grv
embed`. `grv run`'s agent also gets `search_code`/`semantic_search` as
tools it can call mid-task instead of only being handed context up front.

Vectors are stored as BLOBs in `.graviton/index.db` (one `embeddings` row
per FTS chunk) and ranked by cosine similarity — a linear scan for a
freshly-configured repo, or a real ANN index once one exists (see below).
`grv embed --force` recomputes everything (e.g. after switching embedding
models); re-indexing a changed file automatically drops its stale
embeddings so they never point at chunks that no longer exist.

**`grv embed` also builds/refreshes a real ANN index** (`.graviton/ann_<model>.bin`
— an HNSW graph, no FFI, no vector database, no server to run — see
ARCHITECTURE.md for the `instant-distance` crate and design). This is the
actual answer to "does this scale to a huge repo": without it, a semantic
query loads every stored embedding's full text into memory and scores it
one by one; with it, ranking is a compact on-disk graph walk that only
touches full text for the handful of winning chunks. It's purely an
accelerator — every retrieval path falls back to the exact linear scan
automatically if the index is missing, stale (a model/dimension mismatch),
or unreadable, so this can never make a search *wrong*, only faster once
it's there.

### Call graph — `grv callers`/`grv callees`

```sh
grv callers check_password        # every call site anywhere in the index literally named check_password
grv callees dispatch_inner        # every call made from within dispatch_inner
```

Text-based, not type-resolved: `callee_name` is matched literally, the same
simplification `grv symbol`'s `LIKE`-based name lookup already makes for
definitions. Built from a second tree-sitter query per language
(`Lang::call_query_src`) — currently covers Rust/Python/JS/TS/TSX/Go; other
parsed languages just yield no call edges yet (never a hard failure, same
graceful-degradation contract as symbol extraction). `grv index` reports
call sites found alongside symbols/chunks.

### Watch mode — `grv index --watch`

```sh
grv index --watch
```

Indexes once, then keeps re-indexing on real filesystem events (`notify`'s
inotify/kqueue/FSEvents backend, not polling), debounced so a save (which
fires several raw events) or a `git checkout`/branch switch (which touches
many files at once) becomes one re-index instead of many. Deleted files are
now also cleaned out of the index (search/symbol/call-graph results no
longer linger for files that no longer exist) — a real fix, not just a
watch-mode feature, since `grv index` re-runs benefit from it too.

### Per-agent retrieval, everywhere

`ask`/`investigate`/`crew`/`review`/`swarm` no longer just reason over one
fixed context block handed to them up front — each stage/agent gets its own
bounded tool loop (`search_code`/`semantic_search`/`read_file`/`list_dir`/
`web_search`/`web_fetch`/`git_*`), so a `crew` stage that needs different
evidence than the one before it can go get it, instead of every stage being
stuck with the same retrieval pass. `grv run`'s own tool loop already had
this; the read-only commands now share the exact same mechanism.

### `grv serve` — a daemon for editor/IDE integrations

```sh
grv serve                              # unix socket at a short, repo-hashed path (avoids the ~100-byte socket path limit)
grv serve --tcp 127.0.0.1:7420         # also listen on TCP -- prints a required token; see below
```

Runs in the foreground (like `ollama serve`) speaking one JSON object per
line — JSON-RPC 2.0 over a plain newline-delimited socket, not
LSP-style `Content-Length` framing, so a three-line script in anything can
talk to it (`nc -U .graviton/grv.sock` works for manual testing). This
keeps the index connection and Ollama's warmed-up state alive across
requests, instead of every editor query paying a fresh `grv` process's
startup cost.

Methods: `status`, `agents`, `search`, `symbol`, `semantic_search`, `ask`,
`review` (add `"stream": true` to either of the last two for live
`token`/`tool_call` notifications instead of one final answer), `run_start`/
`run_attach`/`run_confirm`/`run_answer_choice`/`run_status` (drive a full
checkpointed `grv run` agentic session — confirm prompts and `ask_user`
questions included — over the socket instead of only from a terminal), and
`shutdown`. `run_attach` streams `run_event` notifications, one of which is
`"kind": "ask_choice"` (the agent's `ask_user` tool call, with its question
and options) — reply with `run_answer_choice {session_id, selected: [...]}`
from any connection. Full request/response shapes and an example session
(including the confirm and ask_choice round trips) are in ARCHITECTURE.md.
Model-calling methods share a `LiveScheduler` (same design as `swarm`/
`mission`), so a chatty editor firing several requests at once still can't
out-run this machine's RAM.

`--tcp` always requires a token (auto-generated and printed once at
startup unless `--tcp-token` sets one) that every request over it must echo
back in `params.token`; the Unix socket never needs one (filesystem
permissions are that boundary instead, same as `ollama serve`'s own
socket). Not a hardened secret — enough to stop an opportunistic hit on the
port from doing anything, for a tool meant to bind `127.0.0.1`/a trusted
LAN, not to stand in for real auth on an internet-facing service.

## Current scope (v0.13)

- **Languages with verified symbol extraction (50):** Rust, Python,
  JavaScript, TypeScript, TSX, C, C++, Go, Java, C#, PHP, Ruby, Bash, Lua,
  Solidity, PowerShell, Haskell, Fish, Dart, Zig, Julia, Groovy, GraphQL,
  Crystal, D, assembly (label-based — the closest thing assembly has to a
  "symbol"), Elixir, Scala, Swift, Perl, R, OCaml, Elm, Nim, Erlang, Vim,
  Nix, HCL/Terraform, CMake, Verilog, VHDL, Fortran, Prolog, Racket,
  Scheme, Protobuf, Objective-C, GLSL, HLSL, and Ada. Every one of these
  was checked against a real sample file's actual `grv symbol` output, not
  just compiled — see ARCHITECTURE.md's "Language coverage" for the method,
  including the handful (Elixir, Racket, Scheme) whose grammar has no
  dedicated "this is a definition" node at all and needed a text-predicate
  check on top of the tree shape to avoid mistaking an ordinary function
  call for one.
- **Recognized + fully searchable, no grammar at all (16):** Kotlin,
  Svelte, Vue, WGSL, LaTeX (all five have a real reason a grammar can't be
  linked here — version conflicts or a broken external scanner, see
  ARCHITECTURE.md), HTML, CSS, JSON, YAML, TOML, XML, Markdown, SQL,
  Dockerfile, INI, Makefile.
- That's 66 recognized languages total, plus: any other text file is still
  fully searchable (line-window chunking never depends on tree-sitter), it
  just won't show up in `grv symbol`/get its own name.
- **Retrieval:** SQLite FTS5 (bm25) for text, LIKE-based symbol name matching
  for precise jumps, plus optional embedding-based semantic search (see
  above) once `grv config --embed-model` + `grv embed` are set up — off by
  default, and every command behaves exactly as before if you never opt in.
- **Agentic retrieval, not single-shot:** `ask`/`investigate`/`crew`/
  `review`/`swarm` (bounded, read-only tool loop) and `grv run`/`grv
  mission` (full tool loop) can all pull in more than their initial
  retrieval pass via `search_code`/`semantic_search`/`read_file`/etc.
  mid-answer. If a question still turns out to be underspecified, the
  model is instructed to say exactly what it's missing so you can re-run
  with `--file` or a narrower question.

## Roadmap

- Kotlin/Svelte/Vue/WGSL/LaTeX symbol extraction — each blocked on a real external issue (old tree-sitter core pinned by the crate, or a broken external-scanner link), not an oversight. See ARCHITECTURE.md.
- Call-graph coverage beyond Rust/Python/JS/TS/TSX/Go — the other 44+ parsed languages have no call queries yet.
- `grv serve --tcp`'s token is a stopgap, not hardened auth — fine bound to 127.0.0.1/a trusted LAN, not something to expose more broadly as-is
- `grv serve run_attach` only streams events from attach time forward (plus a synthesized catch-up for a pending confirm, a pending `ask_user` question, or an already-finished run) — not a full replay of everything that happened before you attached; `run_status` fills the rest of the gap

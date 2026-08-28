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
grv languages                      # every recognized language + what grv can do with it
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
(`Lang::call_query_src`) — covers 48 of the 53 parsed languages (everything
except GraphQL and Protobuf, which have no function-call concept to
extract); a language with no call query just yields no call edges (never a
hard failure, same graceful-degradation contract as symbol extraction).
`grv index` reports call sites found alongside symbols/chunks.

Each `grv callers` hit also carries a `ResolutionHint`, shown inline —
listing real candidates (file, enclosing `impl`/`class`, line), not just a
label: `[likely: Foo::bar in this file]` when one same-named definition
lives in that call site's own file (true far more often than not in real
code); `[this file defines it Nx, still ambiguous locally: ...]` when the
same file defines it more than once (e.g. two `impl` blocks each with
`new()`) — surfaced explicitly rather than papered over with one
confident label; `[unique: path:line Foo::bar]` when there's only one
candidate anywhere even though it's not local; `[import-resolved:
path:line Foo::bar]` when the call site's own file has a **real, resolved
`use`/`import` statement** naming that exact definition (50 languages —
see "Import resolution" below) — genuine resolution via an actual import,
not a heuristic; `[ambiguous -- N candidates, none in this
file: ...]` listing every remaining real candidate's file and scope when
several same-named definitions exist, none are local, and no resolved
import narrows it to one (narrowed to just the import-corroborated
candidates when at least one matched, even without narrowing all the way);
and `[not indexed anywhere -- external/stdlib call, dynamic dispatch, or
just not part of this repo]` — never silence — when nothing defines the
name anywhere in the index (which does *not* mean the function doesn't
exist, only that it wasn't indexed). Not full type resolution — but a real
import resolver, not just a heuristic, for the languages it covers.

### Import resolution

`grv index` also extracts every `use`/`import`/`require`/`#include`
statement (`crates/indexer/src/imports.rs`) and resolves it to an actual
file in the repo where possible (`crates/indexer/src/resolve.rs`), for
**50 languages**: Rust (crate/module tree, discovered from every
`Cargo.toml`'s package name), Python (relative-import directory resolution
plus a bounded source-root guess), JavaScript/TypeScript/TSX (relative-path
+ extension resolution), Go (`go.mod`'s module path — an import can
legitimately resolve to several files, since it names a whole package),
C/C++/Objective-C/GLSL/HLSL/Verilog/Vim/Proto/Solidity/Nix/Bash/Fish/Ruby/
R/Racket/CMake/Erlang/Zig/PHP/LaTeX/Dart/Scheme/PowerShell/Assembly (quoted/
relative-literal path resolution — `#include "x"` resolved against the
including file's own directory, real search-path semantics, while
`#include <x>` is always a system header, correctly never even recorded),
Java/Kotlin/Groovy/Scala/C# (hierarchical module-name resolution against
conventional Maven/Gradle-style source roots, with `import a.b.*`-style
wildcards resolving to every file in the target package directory — same
multi-file honesty as Go), Elm/Haskell/D/Julia (the same hierarchical
resolution, but a wildcard/unqualified/`exposing (..)`/`hiding` import
resolves to that one module's own file, never a directory listing — these
are one-module-per-file languages, not package languages; an earlier
version of this resolver got Elm's case wrong exactly this way before the
distinction was made explicit), Lua (dotted-to-slash `package.path`
convention), Ada/OCaml/Perl/Fortran/Elixir, each via its own genuinely
different naming convention (GNAT's dash-joined-lowercase flat naming;
OCaml's lowercase-outermost-segment; Perl's `::`-to-`/` CPAN convention;
a flat filename guess for Fortran, which has no real convention at all;
Mix's CamelCase-to-snake_case under `lib/` for Elixir), Swift (Swift
Package Manager's own target-to-directory convention: `import CoreModule`
resolves to every file under `Sources/CoreModule/`, a whole subtree, for
a sibling target in the same multi-target package — an external
framework has no matching directory and stays honestly unresolved), and
Terraform/HCL (`module "x" { source = "./y" }` resolves to every `.tf`
file in the referenced directory — same multi-file honesty as Go; a
registry/git reference is unambiguously external and never even
extracted). An import that can't be resolved (an external crate/package/
system header, the stdlib, a `tsconfig.json` path alias, a non-standard
Rust `#[path]` layout, PSR-4 autoloading for PHP `use`, a multi-segment
OCaml `open` naming a sub-module rather than a compilation unit, a Swift
`import` of an external framework) is left unresolved — never a wrong
guess. This is what powers `ResolutionHint::ImportResolved` above; see
ARCHITECTURE.md's "Import resolution" section for exactly what each
language's resolver can and can't do, and for the 8 parsed languages
(Nim, VHDL, Prolog, Crystal, GraphQL, WGSL, Svelte, Vue) that still don't
have one, with the specific reason each was left out.

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
`shutdown`. `run_attach` replays a session's **entire** event history from
`run_start` onward, not just what happens after you attach, then continues
live — attach a minute late and you still get everything from the
beginning. One `run_event` kind is `"ask_choice"` (the agent's `ask_user`
tool call, with its question and options) — reply with `run_answer_choice
{session_id, selected: [...]}` from any connection. Full request/response
shapes and an example session (including the confirm and ask_choice round
trips) are in ARCHITECTURE.md.
Model-calling methods share a `LiveScheduler` (same design as `swarm`/
`mission`), so a chatty editor firing several requests at once still can't
out-run this machine's RAM.

`--tcp` is TLS-only (TLS 1.3, via `rustls`) — every connection is
encrypted, using a self-signed certificate generated fresh each time `grv
serve --tcp` starts. Its SHA-256 fingerprint prints at startup right next
to the token; since there's no CA to validate a private-IP certificate
against, hand that fingerprint to whoever connects out of band and have
them pin it, the same trust-on-first-use model an SSH host key uses. On
top of that (not instead of it), `--tcp` still always requires a token — a
real 256-bit value from the OS's CSPRNG (auto-generated and printed once
at startup unless `--tcp-token` sets one), checked with a fixed-time
comparison (not `==`, which leaks timing info byte-by-byte) and a flat
delay on a wrong guess — that every request over it must echo back in
`params.token`. The Unix socket never needs a token or TLS — filesystem
permissions are that boundary instead (mode `0600`, set explicitly right
after `bind` rather than left to whatever the process's umask happens to
produce — verified this actually holds under a permissive `umask 000`,
not just assumed), same trust model `ollama serve`'s own socket uses.
This is still a `127.0.0.1`/trusted-LAN mechanism, not hardened auth for
an internet-facing service — but a passive listener on the wire can no
longer read the token off it, which is the specific gap this
closes.

## Current scope (v0.21)

Run `grv languages` any time for the live version of this list.

- **Languages with verified symbol extraction (53):** Rust, Python,
  JavaScript, TypeScript, TSX, C, C++, Go, Java, C#, PHP, Ruby, Bash, Lua,
  Solidity, PowerShell, Haskell, Fish, Dart, Zig, Julia, Groovy, GraphQL,
  Crystal, D, assembly (label-based — the closest thing assembly has to a
  "symbol"), Elixir, Scala, Swift, Perl, R, OCaml, Elm, Nim, Erlang, Vim,
  Nix, HCL/Terraform, CMake, Verilog, VHDL, Fortran, Prolog, Racket,
  Scheme, Protobuf, Objective-C, GLSL, HLSL, Ada, Kotlin, LaTeX, and WGSL.
  Every one of these was checked against a real sample file's actual `grv
  symbol` output, not just compiled — see ARCHITECTURE.md's "Language
  coverage" for the method, including the handful (Elixir, Racket, Scheme)
  whose grammar has no dedicated "this is a definition" node at all and
  needed a text-predicate check on top of the tree shape, and the
  automated regression test (`query_predicate_safety_net`) that check
  needed once a subtle tree-sitter query syntax mistake shipped
  undetected for a whole session.
- **Call-graph coverage (48 of the 53):** everything above except
  GraphQL/Protobuf (no function-call concept to extract). Purely
  name-based — `grv callers run` matches every call site literally named
  `run(...)`, whichever `run` it actually is — but each hit now carries a
  `ResolutionHint` (same-file definition found / unique definition
  elsewhere / genuinely ambiguous / nothing indexed), a real signal beyond
  plain name matching without pretending to be full type resolution.
- **Grammar linked and parseable, no `grv symbol` support (2):** Svelte
  and Vue — their own grammars parse a `<script>` block as one opaque
  blob of text, so there's no def query to write without a second,
  injection-based parse pass this project doesn't do.
- **Recognized + fully searchable, no grammar (11):** HTML, CSS, JSON,
  YAML, TOML, XML, Markdown, SQL, Dockerfile, INI, Makefile.
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

- Call-graph *type/full scope* resolution — still name-based by design; a real import resolver now exists for 50 languages (see "Import resolution" above, and `ResolutionHint::ImportResolved`), but true type resolution (knowing exactly which overload/trait impl a call targets) is a different order of engineering effort, on par with what a language server spends its whole existence on. 8 parsed languages still have no import resolver at all — Nim (its grammar has no import-related node to extract from), VHDL (no reliable package-to-file naming convention exists), Prolog (directive shape too uncertain to encode safely), Crystal (confirmed via a real parse-tree dump that its grammar doesn't parse `require "..."` as a call node at all), GraphQL and WGSL (no import/include concept in their own spec at all), Svelte and Vue (their real imports live inside a `<script>` block both grammars parse as one opaque `raw_text` node) — extending to any of them needs either a grammar upgrade, a language-injection parse pass, or genuinely new research, not just more of the same pattern.
- Svelte/Vue symbol extraction would need a second, injection-based parse of their `<script>` block's embedded JS/TS — the grammars themselves only expose it as opaque text.

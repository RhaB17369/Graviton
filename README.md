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
`edit_file`, `delete_file`, `run_shell`, `recon_tool` (the same whitelisted
nmap/ffuf/etc. as `grv tool`), `web_search`/`web_fetch` (DuckDuckGo-backed,
no API key — so the agent checks a current API/CVE/best-practice instead of
answering from a possibly-stale training snapshot), and — with `--browser`
— `browser_navigate`, `browser_eval`, `browser_screenshot`, `browser_console`,
driving the system's headless Chromium via CDP so the agent can actually
load a page, run JS in it, and see what broke, not just guess from source.

Every `write_file`/`edit_file`/`delete_file`/`run_shell`/`recon_tool` call
is shown to you and confirmed before it happens — you see the exact diff or
command — unless you pass `--yolo` for a fully autonomous run. File changes
are checkpointed regardless of `--yolo`:

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
grv search "jwt verify"            # full-text search over indexed chunks
grv symbol validate_token          # jump straight to a symbol's source

grv agents                         # show the roster
grv ask "trace user input into os.system in vuln.py"              # ARCHITECT by default
grv ask --agent sentinel "audit auth.rb for auth bypass"
grv ask --agent reaper --file exploit.c "write a working PoC for this overflow"
grv investigate "is this deserialization path exploitable?"       # REAPER by default, structured output
grv crew "is Vault.sol safe to deploy?"                            # full pipeline, all four agents
grv swarm --agents sentinel,reaper,cryptographer "audit auth.rs"  # independent agents, concurrent
grv mission "harden this API service end to end" --max-depth 2    # recursive planner + execution

grv status                         # index stats + Ollama connectivity + live capacity/top-consumers
grv config --model qwen3:8b --num-ctx 8192 --host http://127.0.0.1:11434
grv config --model-fast qwen2.5:1.5b --model-deep qwen3:14b        # opt into real multi-model
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

## Current scope (v0.8)

- **Languages with symbol extraction (17):** Rust, Python, JavaScript,
  TypeScript, TSX, C, C++, Go, Java, C#, PHP, Ruby, Bash, Lua, Solidity,
  PowerShell.
- **Languages recognized + fully searchable, no symbol extraction (11):**
  Kotlin (see ARCHITECTURE.md for why — tree-sitter-kotlin's crates.io
  release is stuck on an incompatible tree-sitter core), HTML, CSS, JSON,
  YAML, TOML, XML, Markdown, SQL, Dockerfile, INI, Makefile.
- Any other text file is still fully searchable (line-window chunking never
  depends on tree-sitter), it just won't show up in `grv symbol`.
- **Retrieval:** SQLite FTS5 (bm25) for text, LIKE-based symbol name matching
  for precise jumps. No embeddings/vector store in v1 — see ARCHITECTURE.md
  for why that's a deliberate simplification, not an oversight.
- **Single-shot context building:** `ask`/`investigate` run one retrieval
  pass before calling the model. They don't yet let the model ask for more
  context mid-answer (that's the natural v2: give the model a `search`/
  `symbol` tool call and loop). Today, if context is insufficient, the model
  is instructed to say exactly what it's missing so you can re-run with
  `--file` or a narrower question.

## Roadmap

- Let agents request more context mid-run instead of one fixed retrieval pass — partially done: `grv mission`'s leaves can already call `web_search`/`web_fetch`/`read_file`/`list_dir` mid-answer (see ARCHITECTURE.md); `ask`/`investigate`/`crew`/`swarm` still don't, and issuing their own `search`/`symbol` calls against the local index (not just the web) is still open
- Call-graph edges (`grv callers`/`grv callees`) from tree-sitter reference queries
- Incremental re-index on file save (watch mode)
- Optional local embeddings for semantic (not just lexical) search
- Kotlin symbol extraction once a crates.io grammar release supports current tree-sitter
- Per-agent retrieval (each crew stage currently reasons over the same shared context; REAPER asking a differently-shaped question than ARCHITECT could pull in more relevant chunks for each)

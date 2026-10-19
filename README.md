# GRAVITON

`grv` is a local, offline, multi-agent code-intelligence and security
framework. It talks only to [Ollama](https://ollama.com) — your code,
your questions, and the model's answers never leave your machine. There
is no cloud mode, no telemetry, no API key.

The name is the premise: a graviton is the (hypothetical) particle that
carries the weakest fundamental force, yet still shapes galaxies. The bet
here is the same — a small local model (4B–8B, quantized) on ordinary
hardware (no datacenter GPU, often no GPU at all), made to punch above
its weight by a Rust engine that does every deterministic part of the
job — parsing, indexing, retrieval, resource management, tool execution
— so the model is only ever asked to do the one thing it's actually
needed for: reason about what the engine already found.

That division of labor is the whole design. The model is the least
reliable, most expensive component in this system, measured and treated
that way throughout — see [How the model is kept honest](#how-the-model-is-kept-honest-and-what-still-gets-past-it)
for the specific failures that shaped it.

## What it's actually for

- **Exploring and understanding a codebase** you didn't write, too big to
  read — real symbol/call-graph/import-graph indexing, not just grep with
  extra steps.
- **A standing panel of specialists** — 22 agents across programming,
  infrastructure, defensive security, and offensive security — that can
  each be asked directly, chained into a pipeline where each one reads
  the previous one's actual output, or fired off independently and
  compared.
- **An autonomous coding/ops agent** (`grv run`) that reads, writes, and
  edits files, runs shell commands, runs your test suite, drives a real
  git repo, and (optionally) drives a headless browser — checkpointed,
  confirmable, resumable.
- **CTF/HTB/THM/OSCP-style security work** against code or infrastructure
  you own or are authorized to test — the offensive agents are written
  for that explicitly, with no refusal theater layered on top of a
  context that's already established once.
- **IDE/editor integration** via a long-lived daemon (`grv serve`) that
  keeps the index warm instead of re-opening it per keystroke.

What it is *not*: a hosted service, a wrapper that calls a frontier model
over the network, or a tool that assumes you have a 4090 sitting idle.
Every design decision below exists because of a real constraint measured
on real hardware — see [ARCHITECTURE.md](ARCHITECTURE.md) for the
specifics and the dead ends.

## Install

```sh
# 1. Ollama, running, with at least one chat-capable model pulled
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b

# 2. Build grv (Rust, edition 2024 — a recent stable toolchain)
git clone https://github.com/RhaB17369/Graviton.git
cd Graviton
cargo build --release
./target/release/grv --version
```

Put `target/release/grv` on your `PATH`, or invoke it by full path.
First run writes `~/.config/graviton/config.toml` with sane defaults
(`model = "qwen2.5-coder:7b"`, `num_ctx = 8192`) — nothing else to set up
before `grv index` works.

## Five minutes to a working answer

```sh
cd your-project
grv index                              # build the local index (incremental after this)
grv status                             # confirms Ollama is reachable + shows capacity
grv search "jwt verify"                # full-text search, no model call
grv symbol validate_token               # jump straight to a symbol's source
grv ask "trace user input into os.system in app.py"   # ARCHITECT by default
grv ask --agent sentinel "audit auth.rb for an auth bypass"
grv crew "is Vault.sol safe to deploy?"                # architect -> reaper -> sentinel -> singularity
grv run "add input validation to the login form"       # acts on your repo, confirms writes
```

`ask`/`investigate`/`crew`/`swarm`/`review` are **read-only** — the model
can read files, search the index, and fetch the web, but never write.
`run`/`mission` are the acting modes — every write/edit/delete/shell call
is confirmed unless you pass `--yolo`, and every file change is
checkpointed (`grv rollback` undoes it, entirely or step-by-step).

## The agent roster

22 specialists, each a distinct system prompt plus a model tier — not 22
names for one prompt. `grv agents` prints the live roster with taglines;
the table below is a snapshot.

| Category | Agents |
|---|---|
| Programming | `architect` (design/refactoring/correctness), `tester`, `debugger`, `performance` |
| Infrastructure | `netarch`, `devops`, `cloudarch`, `deploy`, `automator`, `documentor` |
| Defensive security | `sentinel` (general audit), `cryptographer`, `quantumcrypto`, `supplychain`, `cloudsec`, `identity` |
| Offensive security | `reaper` (general exploit dev), `webhunter`, `binexp`, `adversary`, `osint` |
| Coordinator | `singularity` — no analysis of its own, synthesizes whichever agents ran |

Four ways to use them:

```sh
grv ask --agent webhunter "..."                              # one agent, read-only
grv run --agent reaper --yolo "fuzz the login endpoint..."    # one agent, acting
grv crew "..."                                                 # sequential hand-off, default pipeline
grv crew --agents cryptographer,identity,singularity "..."     # sequential hand-off, your pipeline
grv swarm --agents sentinel,reaper,cryptographer "..."          # independent, concurrent, RAM-aware
grv mission "harden this API end to end" --max-depth 2          # recursive planner across the whole roster
```

`crew` and `mission` are real pipelines, not the same prompt asked twice:
each stage is handed the *actual text* the previous stage produced.
`swarm` has no hand-off — it fires agents concurrently, each on its own
tier's model, at a concurrency the engine computes from live RAM
headroom (not a guess, not a fixed number — see `grv status`).

## Model routing — one knob, or twenty-two

By default every agent runs on whatever `grv config --model` points at.
Two more slots (`--model-fast`, `--model-deep`) split work across tiers
if you have RAM for more than one resident model. For finer control than
three slots allow — e.g. giving `architect`'s design work a different
model than `reaper`'s exploit dev, even though both are tier `Deep` — a
separate file, `~/.config/graviton/models.toml`, maps individual agent
keys to specific model tags, resolved `agent override > tier override >
config.toml fallback`.

```sh
grv config --model qwen2.5-coder:7b --model-fast qwen2.5-coder:3b --model-deep qwen3.5:4b
grv models                              # show the resolved model for all 22 agents
grv models set architect qwen3.5:4b     # pin one agent
grv models recommend                    # inspect what's pulled in Ollama + this machine's
                                         # RAM, write a full routing sized not to swap
```

`grv models recommend` is the "figure it out for me" path: it classifies
every model you've pulled (coder-tuned → fast/standard, security-tagged →
flagged, uncensored → flagged, vision-tagged → left alone, `:cloud` tags
→ left alone since they're not local), checks each candidate's size
against a safety fraction of *available* RAM (not total — see
ARCHITECTURE.md for why that distinction is load-bearing), and writes the
result. It will tell you plainly when a model it found isn't being
auto-assigned and why (see the tool-calling note below).

## How the model is kept honest, and what still gets past it

This project does not pretend a 4B–9B local model is as reliable as a
frontier one, and it's built around that rather than around the hope
that a better prompt fixes it. Concretely, testing this against real
production code (not toy examples) found and fixed three distinct
failure modes, each with its own fix because prompting alone didn't
close all of them:

1. **A question with no relevant indexed context got a refusal instead
   of an answer.** Retrieval returns whatever ranked highest, with no
   relevance floor — asking an agent to say hello in French pulled back
   unrelated Rust code, and it concluded it *couldn't* answer. Fixed with
   a shared prompt guard telling every agent retrieved context isn't
   guaranteed relevant.
2. **A multi-layered check got audited from one layer and the agent
   answered anyway.** A security review saw a controller-level guard,
   never looked at the stricter service-level check underneath it, and
   both *invented a reimplementation of a mechanism that already existed*
   and *fabricated an exploit scenario the real control flow ruled out* —
   across four different agent/model combinations. A prompt guard telling
   agents to verify before asserting did not reliably change this
   behavior; what did was a mechanical fix, not a behavioral one:
   **`promote_full_files`** now upgrades a search hit from a snippet to
   the whole file (bounded, ~20KB) for every file a hit landed in, so the
   model sees the real mechanism whether or not it bothers to ask for
   more.
3. **An enumerated checklist in an agent's own prompt got read as a
   quota.** Several security agents' prompts list concrete things to
   check for (`weak primitives (...ECB mode)...`, `missing authentication
   (no MAC/AEAD)...`). Given code with neither encryption nor ECB mode
   anywhere, the model still reported "Use of ECB Mode" against plain
   HMAC-signed JWTs — an HMAC already *is* a MAC — because the phrase was
   an example in its own prompt. Reworded every checklist-style agent
   prompt to say explicitly: most categories won't apply to most code,
   and reporting one that isn't present is a false positive like any
   other.

What's still an open, honest limitation: even after all three fixes, a
small (3B–9B) local model given already-correctly-guarded code will
sometimes still assert a false positive, confidently, with a specific
(wrong) line citation. Verified repeatedly across qwen2.5-coder, qwen3.5,
DeepHat, and an uncensored fine-tune, against a Dockerfile with an
explicit, correctly placed `USER node` — more than one model still
reported "running as root," once citing specific lines that do neither.
**Treat every finding from a small local model as a lead to verify,
never as a verdict.**

That specific failure turned out to be a model-*capacity* problem, not a
prompt or architecture problem — confirmed by swapping just the model.
The exact same question, same agent, same three guards, pointed at a
31B model (`gemma4:31b-cloud`, free-tier on Ollama's cloud) instead of a
local 3B–9B one: correct verdict ("not running as root") on the first
try, the right lines quoted, and it went on to `read_file` a second,
*unprompted* file on its own and surface a real, verified-accurate
finding (default secrets with a weak fallback value in a
`docker-compose.yml` never mentioned in the question) that no local
model had found. The guards work; local models at this size don't
reliably have the capacity to follow them. If accuracy matters more than
keeping every byte on this machine for a given question, pin that one
agent to a cloud model (`grv models set <agent> gemma4:31b-cloud`, free
at the time of writing) — but know what that trade is: the code in that
question, and whatever it reads on its own, leaves the machine. Nothing
here does that by default, and nothing should without you choosing it
per-agent, deliberately — which is exactly why `grv models recommend`
never auto-assigns a `:cloud` tag.

Two more things learned the hard way, not from a code bug but from how
*models themselves* behave through Ollama, worth knowing before you pick
one for `grv models set`:

- **A "thinking"-capable model's turn can go silently empty.** A model
  with reasoning on by default (several do) streams it in a separate API
  field GRAVITON never read — a turn that spent its whole budget
  reasoning completed with no error and no visible output at all. Fixed
  by disabling thinking mode on every request GRAVITON makes; no agent
  here consumes a reasoning trace, so there's no reason to pay for one.
- **Not every model you can `ollama pull` actually supports tool calling
  through Ollama**, even when the model itself is capable of it. One
  custom GGUF import hard-crashed Ollama (`HTTP 500`) the instant a
  second turn followed a tool call — its bundled chat template turned out
  to be a vision-model template with no generic tool-result handling at
  all. Another's tool calls silently never registered — the JSON showed
  up as plain text instead of a structured call. Since every agent here
  relies on the same tool loop (`read_file`/`search_code`/`web_search`/
  ...) for grounding, `grv models recommend` now only auto-assigns models
  it has verified actually signal tool calls correctly; anything else is
  surfaced, flagged, and left to manual opt-in.

## `grv run` — the acting mode

```sh
grv run "add a /health endpoint, then curl it and confirm it returns 200" --browser
grv run --continue                              # resume the most recent session
grv checkpoints                                  # list sessions and what they touched
grv rollback                                     # undo the most recent session entirely
grv rollback <session-id> --to 3                 # undo back to step 3
grv plan                                         # show the session's current self-reported plan
```

Every write/edit/delete/shell call is confirmed before it happens unless
`--yolo` is set. Every file change is checkpointed regardless — rollback
works whether or not you used `--yolo`. A session's conversation and plan
persist to disk, so `--continue` picks up a multi-hour task exactly where
it left off, including after a crash.

Fine-grained permission rules (`.graviton/permissions.toml`) sit
underneath that default, able to make a specific tool call stronger than
`--yolo` (a `deny` rule blocks it even with `--yolo` set) or weaker than
the default (an `allow` rule skips confirmation without `--yolo`).

Custom tools (`grv custom new <name>`, a TOML file, no recompiling) add a
named, schema'd shell-command template to the roster `grv run` can call —
useful for a project-specific action (`docker ps`, a deploy script) the
model shouldn't have to construct from a raw shell command every time.

## `grv review` — real diffs, real tests, real recon

```sh
grv review                        # crew review of every uncommitted change vs HEAD
grv review --staged
grv review main..HEAD
grv review --agents cryptographer,singularity
```

`review` feeds the crew a real `git diff`, not retrieved/indexed context
— each agent audits what actually changed. `grv run` separately drives
git natively (status/diff/log/commit as real tool calls) and runs your
actual test suite as part of its own test→fail→fix loop, not a simulated
one.

```sh
grv tool run nmap -- -sV -p- 10.10.10.5     # whitelisted recon tools, indexed for later ask/search
grv tool list                                # whitelist + recent runs in this repo
```

## Retrieval: what the model actually sees

- **Lexical** (SQLite FTS5/bm25) and **symbol-name** lookups, always on.
- **Semantic** search (embedding cosine similarity) is opt-in:
  `grv config --embed-model nomic-embed-text && grv embed`, then
  `grv search --semantic` or it's folded automatically into agent
  retrieval. Off by default; every command behaves exactly as before if
  you never touch it.
- **Call graph** (`grv callers`/`grv callees`): name-based (not full type
  resolution), but every hit carries a `ResolutionHint` — same-file
  definition found, unique definition elsewhere, genuinely ambiguous, or
  nothing indexed — a real signal past plain name matching.
- **Import resolution**: a real per-language resolver (not a guess) for
  54 of the 66 recognized languages, turning a raw `use`/`import`
  statement into the actual file(s) it points at.
- **Watch mode** (`grv index --watch`): keeps the index current on file
  changes, debounced, for a long-running `grv serve` or an active editing
  session.

## `grv serve` — a daemon for editors

```sh
grv serve                              # unix socket at a short, repo-hashed path
grv serve --tcp 127.0.0.1:7420          # also listen on TCP -- prints a required token
```

Newline-delimited JSON-RPC 2.0 over the socket — `status`/`search`/
`symbol`/`callers`/`callees`/`ask`/`review`/a full `grv run` session
(`run_attach`, with `ask_user`/approval round-trips)/`shutdown`. Lets an
IDE extension query without spawning a fresh process (and re-opening the
index) per keystroke.

## Current scope (v0.20)

Run `grv languages`/`grv agents` for the live, authoritative version of
this — it's generated from the same code that runs the commands, so it
can't drift from reality the way a hand-maintained list can.

- **66 recognized languages**: 48 with full symbol extraction + call
  graph, 5 more with symbol extraction only (no call-graph concept —
  GraphQL/Kotlin/LaTeX/Protobuf/WGSL), 13 searchable with no `grv symbol`
  support (CSS/HTML/JSON/YAML/TOML/XML/SQL/Dockerfile/etc., plus Svelte/
  Vue, whose grammars expose `<script>` as opaque text). Any other text
  file is still fully searchable via FTS — it just won't show up in
  `grv symbol`.
- **54 languages** have real import resolution; only WGSL's import
  convention has no reliable file-to-resolve-against mapping.
- **22 agents**, 3 model tiers, per-agent model routing via
  `models.toml`.
- Multi-model, resource-aware concurrency for `swarm`/`mission`, computed
  from live *available* RAM (re-sampled continuously during a run, not
  assumed once at startup).

## Roadmap

- Call-graph *type* resolution (knowing exactly which overload/trait impl
  a call targets) is a different order of engineering effort than this
  project's name-based-plus-hints approach — on par with what a language
  server spends its whole existence on. Not attempted; `ResolutionHint`
  is the deliberate stopping point short of it.
- Svelte/Vue symbol extraction needs a second, injection-based parse of
  their `<script>` block's embedded JS/TS — their grammars only expose it
  as opaque text today.
- The false-positive rate on already-correct code (see above) is a real,
  open problem. The three fixes so far reduced it; none eliminated it,
  and the honest next step is more mechanical grounding (à la
  `promote_full_files`), not another round of prompt wording.

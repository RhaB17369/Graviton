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
work product to each other, not four names for one prompt:

| Agent | Specialty |
|---|---|
| **ARCHITECT** | high-level programming: design, refactoring, correctness, performance |
| **SENTINEL** | defensive security: vulnerability auditing, hardening, blue-team |
| **REAPER** | offensive security: exploit dev, payloads, red-team/CTF |
| **SINGULARITY** | coordinator: synthesizes the other three into one decision brief |

Ask one directly (`grv ask --agent sentinel "..."`), or run the whole crew
(`grv crew "..."`) — ARCHITECT explains the code, REAPER finds what's
exploitable in it, SENTINEL proposes fixes, SINGULARITY reads all three
outputs and converges them into a prioritized action plan. Every stage
reads the *actual output* of the ones before it, not just the same raw
context re-served — that hand-off is what makes it a pipeline instead of
running the same question three times.

They all run on the one local model configured in `grv config` (there's no
second GPU to run four models on) — the specialization comes from four
different system prompts and a real sequential hand-off, not four separate
brains.

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

grv status                         # index stats + Ollama connectivity
grv config --model qwen3:8b --num-ctx 8192 --host http://127.0.0.1:11434
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

## Current scope (v0.2)

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

- Let agents request more context mid-run instead of one fixed retrieval pass (model issues its own `search`/`symbol` calls, or triggers `grv tool run` itself, before answering)
- Call-graph edges (`grv callers`/`grv callees`) from tree-sitter reference queries
- Incremental re-index on file save (watch mode)
- Optional local embeddings for semantic (not just lexical) search
- Kotlin symbol extraction once a crates.io grammar release supports current tree-sitter
- Per-agent retrieval (each crew stage currently reasons over the same shared context; REAPER asking a differently-shaped question than ARCHITECT could pull in more relevant chunks for each)

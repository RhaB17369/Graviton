# GRAVITON

> A graviton is theoretically the particle that carries gravity — vanishingly
> small, disproportionate force. That's the bet here: an 8B local model on a
> 4GB-VRAM laptop, made to punch above its weight with a Rust engine that
> does the deterministic work (parsing, indexing, retrieval) so the model
> only has to reason.

`grv` is a local, offline CLI copilot for huge codebases and offensive
security work (CTF, HTB/THM, OSCP/OSCE/CPTS-style labs, auditing code you
own or are authorized to test), built for machines that can't run a 30B+
model comfortably. It talks to [Ollama](https://ollama.com) — no cloud, your
code never leaves the machine.

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
grv ask "trace user input into os.system in vuln.py"
grv investigate "is this deserialization path exploitable?"
grv ask --file exploit.c "write a working PoC for this overflow"

grv status                         # index stats + Ollama connectivity
grv config --model qwen3:8b --num-ctx 8192 --host http://127.0.0.1:11434
```

`ask` streams the model's answer token by token. `investigate` uses a
structured system prompt (symbols found → data flow → analysis → concrete
next step/PoC) instead of a free-form answer.

## Current scope (v0.1)

- **Languages with symbol extraction:** Rust, Python, JavaScript, TypeScript/TSX, C, C++, Go.
  Any other text file is still fully searchable (line-window chunking never
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

- Agentic tool loop for `investigate` (model can request more symbols/chunks mid-run)
- Call-graph edges (`grv callers`/`grv callees`) from tree-sitter reference queries
- Incremental re-index on file save (watch mode)
- Optional local embeddings for semantic (not just lexical) search

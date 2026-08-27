//! The agentic tool-use loop behind `grv run`: an agent that doesn't just
//! answer from retrieved context, but reads/writes files, runs shell
//! commands, and drives a browser, in a loop, until it's done.
//!
//! `ask`/`investigate`/`crew` are read-only; this is the side-effecting
//! mode, so it gets its own safety net: every write/edit/delete is
//! checkpointed (`checkpoint.rs`, undoable with `grv rollback`), and every
//! write/edit/delete/shell call is confirmed with the user unless `--yolo`
//! is set.

use crate::agents::AgentSpec;
use crate::browser::BrowserSession;
use crate::checkpoint;
use crate::custom_tools::{self, CustomTool};
use crate::permissions;
use crate::run_io::RunIo;
use crate::semantic;
use crate::tools as recon_tools;
use crate::web;
use anyhow::{Context, Result};
use graviton_core::Config;
use graviton_llm::{ChatMessage, OllamaClient, ToolCall, ToolDef};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_STEPS: usize = 40;
const MAX_TOOL_OUTPUT: usize = 8_000;

pub struct AgentLoopConfig {
    pub auto_approve: bool,
    pub enable_browser: bool,
}

struct State {
    root: PathBuf,
    checkpoint: checkpoint::Session,
    auto_approve: bool,
    browser: Option<BrowserSession>,
    custom_tools: Vec<CustomTool>,
    permissions: Vec<permissions::Rule>,
    index_dir: String,
    ollama_host: String,
    embed_model: Option<String>,
    io: Arc<dyn RunIo>,
}

/// The read-only subset of the tool roster — no file writes/shell/recon,
/// just reading the repo and the live web. Shared with `mission.rs`, whose
/// leaves stay analysis-only (like `ask`/`crew`/`swarm`) rather than
/// gaining `grv run`'s full acting/checkpointed tool set.
pub(crate) fn read_only_tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "read_file",
            "Read a text file's contents, given a path relative to the repo root.",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        ),
        ToolDef::new(
            "list_dir",
            "List files and directories under a path relative to the repo root (skips .git/target/node_modules/etc).",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "defaults to the repo root" },
                    "recursive": { "type": "boolean", "description": "defaults to false" }
                }
            }),
        ),
        ToolDef::new(
            "web_search",
            "Search the web (DuckDuckGo) for current information and get titles + URLs + snippets back. \
             Use this whenever a technique, API, CVE, or best practice might have changed since training — \
             don't answer from memory alone for anything version-specific or time-sensitive.",
            json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] }),
        ),
        ToolDef::new(
            "web_fetch",
            "Fetch a URL and return its page text (HTML stripped) — read a specific page, e.g. one found via web_search, official docs, or an advisory.",
            json!({ "type": "object", "properties": { "url": { "type": "string" } }, "required": ["url"] }),
        ),
        ToolDef::new(
            "git_status",
            "Show `git status` (branch, staged/unstaged/untracked files) for the repo.",
            json!({ "type": "object", "properties": {} }),
        ),
        ToolDef::new(
            "git_diff",
            "Show a real git diff — staged changes, or unstaged if staged=false (default), optionally restricted to one path. Use this instead of guessing what changed from separate read_file calls.",
            json!({
                "type": "object",
                "properties": {
                    "staged": { "type": "boolean", "description": "diff the index instead of the working tree; defaults to false" },
                    "path": { "type": "string", "description": "restrict the diff to this path; omit for the whole repo" }
                }
            }),
        ),
        ToolDef::new(
            "git_log",
            "Show recent commit history (oneline), optionally restricted to one path.",
            json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "description": "defaults to 10" },
                    "path": { "type": "string" }
                }
            }),
        ),
        ToolDef::new(
            "search_code",
            "Full-text search over the indexed codebase (lexical, exact-token matching) — \
             good for a known identifier, error string, or exact API name. Requires \
             `grv index` to have been run in this repo.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "description": "defaults to 8" }
                },
                "required": ["query"]
            }),
        ),
        ToolDef::new(
            "semantic_search",
            "Semantic (meaning-based) search over the indexed codebase using embeddings — \
             finds conceptually related code even when it shares no keywords with the query. \
             Only works if an embedding model is configured (`grv config --embed-model ...`) \
             and `grv embed` has been run; returns a clear error otherwise, so try search_code \
             instead if this one errors.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "description": "defaults to 8" }
                },
                "required": ["query"]
            }),
        ),
    ]
}

/// Everything `search_code`/`semantic_search` need after the (synchronous)
/// index read: either a ready answer, or the loaded embeddings + query text
/// still needing an embedding call + ranking.
///
/// Split this way — instead of one `async fn` taking `&rusqlite::Connection`
/// — because `Connection` isn't `Sync`, so `&Connection` isn't `Send`, and
/// (per rustc, regardless of whether it's actually touched after an await)
/// an `async fn` with `&Connection` anywhere in its signature has its
/// returned future's `Send`-ness poisoned by that alone. `search_code`/
/// `semantic_search` run from inside mission's `Box::pin(... + Send)`
/// leaves and `grv serve`'s `tokio::spawn`ed connections, so no async fn in
/// that path may take `&Connection` — see `semantic::EmbeddedChunk`.
enum SearchOutcome {
    Done(Result<String>),
    Rank { model: String, query: String, limit: usize, chunks: Vec<semantic::EmbeddedChunk> },
}

/// `search_code`/`semantic_search` logic, shared between `dispatch_inner`
/// (which already has an open index connection) and `dispatch_read_only`
/// (which opens a short-lived one per call — fine under WAL, and mission's
/// leaves calling this concurrently is exactly the case WAL mode is for).
/// Plain sync fn — see `SearchOutcome`.
fn prepare_search_tool(conn: &rusqlite::Connection, embed_model: Option<&str>, name: &str, args: &Value) -> Option<SearchOutcome> {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
    match name {
        "search_code" => Some(SearchOutcome::Done((|| {
            let hits = crate::context::search_chunks(conn, &query, limit)?;
            if hits.is_empty() {
                return Ok("no matches".to_string());
            }
            Ok(hits.iter().map(|h| format!("--- {} ---\n{}", h.header, h.body)).collect::<Vec<_>>().join("\n\n"))
        })())),
        "semantic_search" => {
            let Some(model) = embed_model else {
                return Some(SearchOutcome::Done(Err(anyhow::anyhow!(
                    "no embedding model configured — set one with `grv config --embed-model <tag>` and run `grv embed` first"
                ))));
            };
            if !semantic::has_embeddings(conn) {
                return Some(SearchOutcome::Done(Err(anyhow::anyhow!("no embeddings computed yet — run `grv embed` first"))));
            }
            match semantic::load_embeddings(conn, model) {
                Ok(chunks) => Some(SearchOutcome::Rank { model: model.to_string(), query, limit, chunks }),
                Err(e) => Some(SearchOutcome::Done(Err(e))),
            }
        }
        _ => None,
    }
}

/// The (`Connection`-free — see `SearchOutcome`) async half: embed the
/// query and rank, if needed.
async fn finish_search_outcome(ollama_host: &str, outcome: SearchOutcome) -> Result<String> {
    match outcome {
        SearchOutcome::Done(r) => r,
        SearchOutcome::Rank { model, query, limit, chunks } => {
            let hits = semantic::rank_by_query(ollama_host, &model, &query, chunks, limit).await?;
            if hits.is_empty() {
                return Ok("no matches".to_string());
            }
            Ok(hits
                .iter()
                .map(|h| format!("--- {}:{}-{} (score {:.2}) ---\n{}", h.path, h.start_line, h.end_line, h.score, h.body))
                .collect::<Vec<_>>()
                .join("\n\n"))
        }
    }
}

/// Run a git subcommand with fixed, safe arguments (never a model-supplied
/// arbitrary argument string) and return its combined output — used by
/// both the read-only git tools and `grv review`.
pub(crate) fn run_git(root: &Path, args: &[&str]) -> Result<String> {
    let output = std::process::Command::new("git").args(args).current_dir(root).output().context("running git")?;
    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(combined)
}

/// The three read-only git tools' logic, shared between `dispatch_read_only`
/// (used by `mission`'s leaves) and `dispatch_inner` (used by `grv run`) —
/// `None` if `name` isn't one of them.
fn dispatch_git_readonly(root: &Path, name: &str, args: &Value) -> Option<Result<String>> {
    let arg_str = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string();
    match name {
        "git_status" => Some(run_git(root, &["status"])),
        "git_diff" => {
            let staged = args.get("staged").and_then(|v| v.as_bool()).unwrap_or(false);
            let path = arg_str("path");
            let mut a = vec!["diff"];
            if staged {
                a.push("--staged");
            }
            if !path.is_empty() {
                a.push("--");
                a.push(&path);
            }
            Some(run_git(root, &a))
        }
        "git_log" => {
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10).to_string();
            let path = arg_str("path");
            let mut a = vec!["log", "--oneline", "-n", &limit];
            if !path.is_empty() {
                a.push("--");
                a.push(&path);
            }
            Some(run_git(root, &a))
        }
        _ => None,
    }
}

/// Dispatch one of `read_only_tool_defs`' tools outside the full agentic
/// loop's `State`/checkpoint machinery — used by `mission.rs`, which has no
/// write/edit/delete tools to checkpoint in the first place.
pub(crate) async fn dispatch_read_only(cfg: &Config, root: &Path, name: &str, args: &Value) -> String {
    let arg_str = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let result: Result<String> = async {
        match name {
            "read_file" => {
                let rel = arg_str("path");
                let full = resolve_rel(root, &rel)?;
                Ok(std::fs::read_to_string(&full).with_context(|| format!("reading {rel}"))?)
            }
            "list_dir" => {
                let rel = arg_str("path");
                let rel = if rel.is_empty() { "." } else { &rel };
                let full = resolve_rel(root, rel)?;
                list_dir(&full, args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false))
            }
            "web_search" => web::search(&arg_str("query")).await,
            "web_fetch" => web::fetch(&arg_str("url")).await,
            "search_code" | "semantic_search" => {
                // A short-lived read connection per call — cheap, and safe
                // under WAL for mission's concurrently-running leaves to
                // each open their own against the same index.db.
                let db_path = graviton_core::db_path_for(root, &cfg.index_dir)?;
                if !db_path.exists() {
                    anyhow::bail!("no index found — run `grv index` first");
                }
                let conn = graviton_core::open_db(&db_path)?;
                let outcome = prepare_search_tool(&conn, cfg.embed_model.as_deref(), name, args);
                drop(conn); // done with the connection before the async ranking step below
                match outcome {
                    Some(outcome) => finish_search_outcome(&cfg.ollama_host, outcome).await,
                    None => unreachable!(),
                }
            }
            other => match dispatch_git_readonly(root, other, args) {
                Some(r) => r,
                None => anyhow::bail!("unknown read-only tool '{other}'"),
            },
        }
    }
    .await;
    match result {
        Ok(s) => truncate(s),
        Err(e) => format!("error: {e:#}"),
    }
}

/// Bounded read-only tool loop shared by `ask`/`investigate`/`crew`/
/// `review`/`swarm`: instead of one fixed retrieval pass handed to a
/// single completion call (the old `run_agent`), the model can call
/// `read_only_tool_defs()`'s tools along the way — `search_code`/
/// `semantic_search` to pull in more than the initial retrieval surfaced,
/// `read_file`/`list_dir` to look at a specific file it was only shown a
/// chunk of, `web_search`/`web_fetch`/`git_*` same as everywhere else.
/// This is what makes "per-agent retrieval" real: each crew stage (or
/// swarm agent) can now go find its *own* evidence instead of every stage
/// reasoning over one shared context block.
///
/// Not the same loop as `grv run`'s `run()` below — no write/edit/delete/
/// shell, no checkpoints, no confirm gate; this can never act, only look
/// harder before answering.
pub const MAX_READONLY_TOOL_STEPS: usize = 6;

pub async fn run_read_only_loop(
    client: &OllamaClient,
    cfg: &Config,
    root: &Path,
    model: &str,
    system: &str,
    user_msg: &str,
    stream_final_answer: bool,
) -> Result<String> {
    let stdout = std::io::stdout();
    let on_token = |tok: &str| {
        if stream_final_answer {
            let mut lock = stdout.lock();
            let _ = lock.write_all(tok.as_bytes());
            let _ = lock.flush();
        }
    };
    let on_tool_call = |name: &str, args: &Value| {
        println!("\x1b[2m  → {name}({args})\x1b[0m");
    };
    let out = run_read_only_loop_with(client, cfg, root, model, system, user_msg, on_token, on_tool_call).await?;
    if stream_final_answer {
        println!();
    }
    Ok(out)
}

/// The generic core `run_read_only_loop` wraps for the CLI's stdout/
/// println behavior — pulled out so `grv serve` (`daemon.rs`) can reuse the
/// exact same tool loop with its own callbacks (NDJSON notifications over a
/// socket instead of terminal output), rather than a second hand-copied
/// implementation drifting from this one over time.
pub async fn run_read_only_loop_with(
    client: &OllamaClient,
    cfg: &Config,
    root: &Path,
    model: &str,
    system: &str,
    user_msg: &str,
    mut on_token: impl FnMut(&str),
    mut on_tool_call: impl FnMut(&str, &Value),
) -> Result<String> {
    let tools = read_only_tool_defs();
    let mut messages = vec![ChatMessage::system(system), ChatMessage::user(user_msg)];

    for _ in 0..MAX_READONLY_TOOL_STEPS {
        let result = client.chat_stream(model, &messages, cfg.num_ctx, &tools, &mut on_token).await?;
        if result.tool_calls.is_empty() {
            return Ok(result.content);
        }
        messages.push(ChatMessage::assistant_tool_calls(result.tool_calls.clone()));
        for tc in &result.tool_calls {
            on_tool_call(&tc.function.name, &tc.function.arguments);
            let out = dispatch_read_only(cfg, root, &tc.function.name, &tc.function.arguments).await;
            messages.push(ChatMessage::tool_result(&tc.function.name, out));
        }
    }
    // Used up the tool-call budget without a final answer -- ask once
    // more with no tools offered, forcing a text response instead of
    // silently returning nothing.
    let result = client.chat_stream(model, &messages, cfg.num_ctx, &[], &mut on_token).await?;
    Ok(result.content)
}

fn tool_defs(enable_browser: bool, custom: &[CustomTool]) -> Vec<ToolDef> {
    let mut tools = read_only_tool_defs();
    tools.extend(vec![
        ToolDef::new(
            "update_plan",
            "Report your current step-by-step plan for this task, with a status per step \
             (pending/in_progress/done). Call this when you form a plan and again whenever it \
             changes — a step starts, finishes, or the plan itself changes — so progress is \
             visible to the user watching and saved for `grv run --continue`. Skip it for a \
             trivial single-step task.",
            json!({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": { "type": "string" },
                                "status": { "type": "string", "enum": ["pending", "in_progress", "done"] }
                            },
                            "required": ["text", "status"]
                        }
                    }
                },
                "required": ["steps"]
            }),
        ),
        ToolDef::new(
            "write_file",
            "Create a file or overwrite it entirely with new content. Checkpointed — undoable via `grv rollback`.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        ),
        ToolDef::new(
            "edit_file",
            "Replace one exact, unique occurrence of old_string with new_string in an existing file. Fails if old_string appears zero or multiple times — include enough surrounding context to make it unique. Checkpointed — undoable via `grv rollback`.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        ),
        ToolDef::new(
            "delete_file",
            "Delete a file. Checkpointed — undoable via `grv rollback`.",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        ),
        ToolDef::new(
            "run_shell",
            "Run a shell command in the repo root and return its stdout+stderr. Not checkpointed (arbitrary side effects can't be generically undone) — confirmed before running unless auto-approve is on.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "why": { "type": "string", "description": "one short sentence on what this command is for" }
                },
                "required": ["command"]
            }),
        ),
        ToolDef::new(
            "run_tests",
            "Run this project's test suite and get back a structured summary (pass/fail, and for \
             failures the specific failing test names and error output) instead of raw noise. The \
             command is auto-detected from the repo (cargo test / npm test / pytest / go test / \
             bundle exec rspec) — pass `command` to override if detection is wrong or the task \
             needs a narrower run (one test file, one test name). Prefer this over run_shell for \
             tests. After a fix, run this again before declaring the task done.",
            json!({
                "type": "object",
                "properties": { "command": { "type": "string", "description": "override the auto-detected test command" } }
            }),
        ),
        ToolDef::new(
            "git_commit",
            "Stage and commit changes. Not checkpointed the way file writes are — a commit is \
             already its own undo point (`git reset`/`git revert`), so this doesn't duplicate that. \
             Confirmed before running unless auto-approve is on.",
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" },
                    "paths": { "type": "array", "items": { "type": "string" }, "description": "specific paths to stage; omit to stage all changes" }
                },
                "required": ["message"]
            }),
        ),
        ToolDef::new(
            "recon_tool",
            &format!(
                "Run a whitelisted recon/security tool ({}) and get its output.",
                recon_tools::ALLOWED_TOOLS.join(", ")
            ),
            json!({
                "type": "object",
                "properties": {
                    "tool": { "type": "string" },
                    "args": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["tool", "args"]
            }),
        ),
    ]);

    if enable_browser {
        tools.push(ToolDef::new(
            "browser_navigate",
            "Navigate the headless browser to a URL and return its page title.",
            json!({ "type": "object", "properties": { "url": { "type": "string" } }, "required": ["url"] }),
        ));
        tools.push(ToolDef::new(
            "browser_eval",
            "Evaluate a JavaScript expression in the current page and return its value.",
            json!({ "type": "object", "properties": { "script": { "type": "string" } }, "required": ["script"] }),
        ));
        tools.push(ToolDef::new(
            "browser_screenshot",
            "Save a full-page PNG screenshot of the current page to a path relative to the repo root.",
            json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
        ));
        tools.push(ToolDef::new(
            "browser_console",
            "Get console.log/warn/error output captured from the current page since navigation.",
            json!({ "type": "object", "properties": {} }),
        ));
    }

    tools.extend(custom.iter().map(CustomTool::to_tool_def));
    tools
}

/// A confirmation gate's outcome. `Redirect` is the mid-task steering
/// mechanism: typing anything other than y/n at a confirmation prompt
/// declines the action *and* carries the typed text back to the model as
/// the tool result, so "no, do X instead" actually reaches the next turn
/// instead of only being expressible as a blind yes/no. This only fires at
/// existing confirmation pauses — a `--yolo` run has none, so it can only
/// be interrupted with Ctrl-C, which this doesn't change.
pub(crate) enum Decision {
    Allow,
    Deny,
    Redirect(String),
}

/// Turn a `Decision` into the usual bail-if-not-allowed check, folding a
/// redirect's text into the error the model sees as this tool's result.
fn require_allowed(decision: Decision, declined_msg: &str) -> Result<()> {
    match decision {
        Decision::Allow => Ok(()),
        Decision::Deny => anyhow::bail!("{declined_msg}"),
        Decision::Redirect(note) => anyhow::bail!("{declined_msg} — the user says: {note}"),
    }
}

/// The single gate every side-effecting tool call goes through:
/// `.graviton/permissions.toml` rules are checked first (a `deny` rule
/// blocks even under `--yolo`, an `allow` rule skips the prompt even
/// without it); anything the rules don't cover falls through to
/// `state.io.confirm` (a terminal y/n/redirect prompt for `grv run`, or a
/// round trip over the socket for `grv serve`'s `run_start` — this gate
/// doesn't know or care which).
async fn gate(state: &State, tool: &str, primary_arg: &str, action_desc: &str, declined_msg: &str) -> Result<()> {
    match permissions::check(&state.permissions, tool, primary_arg) {
        permissions::Verdict::Allow => Ok(()),
        permissions::Verdict::Deny(reason) => anyhow::bail!("{declined_msg} — {reason}"),
        permissions::Verdict::Fallback => {
            let decision = state.io.confirm(state.auto_approve, action_desc.to_string()).await;
            require_allowed(decision, declined_msg)
        }
    }
}

fn resolve_rel(root: &Path, rel: &str) -> Result<PathBuf> {
    let p = root.join(rel);
    // Cheap containment check: refuse to touch paths that escape the repo
    // root via `..`. Not a hard sandbox (symlinks aren't checked), just a
    // guard against the common mistake of an absolute-looking path.
    let normalized = path_clean(&p);
    if !normalized.starts_with(root) {
        anyhow::bail!("path '{rel}' resolves outside the repo root, refusing");
    }
    Ok(p)
}

fn path_clean(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn truncate(s: String) -> String {
    if s.len() > MAX_TOOL_OUTPUT {
        format!("{}\n...[truncated, {} bytes total]", &s[..MAX_TOOL_OUTPUT], s.len())
    } else {
        s
    }
}

fn diff_preview(old: &str, new: &str) -> String {
    use similar::{ChangeTag, TextDiff};
    let diff = TextDiff::from_lines(old, new);
    let mut out = String::new();
    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            ChangeTag::Delete => "-",
            ChangeTag::Insert => "+",
            ChangeTag::Equal => " ",
        };
        out.push_str(&format!("{sign}{change}"));
    }
    out
}

// Neither `dispatch` nor `dispatch_inner` takes `&rusqlite::Connection` as
// a parameter -- unlike `dispatch_read_only`'s callers, `grv run`'s own
// loop must itself be spawnable (`grv serve`'s `run_start`), and
// `Connection` isn't `Sync`, so `&Connection` anywhere in an async fn's
// signature poisons its future's `Send`-ness regardless of how it's
// actually used inside (see `SearchOutcome`'s doc comment). The two arms
// that need a connection (`recon_tool`, `search_code`/`semantic_search`)
// each open their own short-lived one instead, same as `dispatch_read_only`.
async fn dispatch(state: &mut State, name: &str, args: &Value) -> String {
    match dispatch_inner(state, name, args).await {
        Ok(s) => s,
        Err(e) => format!("error: {e:#}"),
    }
}

async fn dispatch_inner(state: &mut State, name: &str, args: &Value) -> Result<String> {
    let arg_str = |key: &str| -> Result<String> {
        args.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .with_context(|| format!("missing '{key}' argument"))
    };

    match name {
        "update_plan" => {
            let steps = args.get("steps").cloned().unwrap_or(Value::Array(vec![]));
            state.io.emit(format_plan(&steps));
            state.checkpoint.save_plan(&json!({ "steps": steps })).ok();
            Ok(format!("plan updated ({} step(s))", steps.as_array().map(|a| a.len()).unwrap_or(0)))
        }
        "read_file" => {
            let rel = arg_str("path")?;
            let full = resolve_rel(&state.root, &rel)?;
            let content = std::fs::read_to_string(&full).with_context(|| format!("reading {rel}"))?;
            Ok(truncate(content))
        }
        "list_dir" => {
            let rel = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let recursive = args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false);
            let full = resolve_rel(&state.root, rel)?;
            Ok(truncate(list_dir(&full, recursive)?))
        }
        "write_file" => {
            let rel = arg_str("path")?;
            let content = arg_str("content")?;
            let full = resolve_rel(&state.root, &rel)?;
            let old = std::fs::read_to_string(&full).unwrap_or_default();
            let preview = diff_preview(&old, &content);
            gate(state, "write_file", &rel, &format!("write_file {rel}\n{preview}"), "user declined this write").await?;
            state.checkpoint.snapshot_before_write(&rel)?;
            if let Some(parent) = full.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(&full, &content).with_context(|| format!("writing {rel}"))?;
            Ok(format!("wrote {rel} ({} bytes)", content.len()))
        }
        "edit_file" => {
            let rel = arg_str("path")?;
            let old_string = arg_str("old_string")?;
            let new_string = arg_str("new_string")?;
            let full = resolve_rel(&state.root, &rel)?;
            let content = std::fs::read_to_string(&full).with_context(|| format!("reading {rel}"))?;
            let occurrences = content.matches(&old_string).count();
            if occurrences == 0 {
                anyhow::bail!("old_string not found in {rel} — re-read the file, it may have changed");
            }
            if occurrences > 1 {
                anyhow::bail!("old_string appears {occurrences} times in {rel} — include more context to make it unique");
            }
            let updated = content.replacen(&old_string, &new_string, 1);
            let preview = diff_preview(&content, &updated);
            gate(state, "edit_file", &rel, &format!("edit_file {rel}\n{preview}"), "user declined this edit").await?;
            state.checkpoint.snapshot_before_write(&rel)?;
            std::fs::write(&full, &updated).with_context(|| format!("writing {rel}"))?;
            Ok(format!("edited {rel}"))
        }
        "delete_file" => {
            let rel = arg_str("path")?;
            let full = resolve_rel(&state.root, &rel)?;
            if !full.exists() {
                anyhow::bail!("{rel} doesn't exist");
            }
            gate(state, "delete_file", &rel, &format!("delete_file {rel}"), "user declined this delete").await?;
            state.checkpoint.snapshot_before_delete(&rel)?;
            std::fs::remove_file(&full).with_context(|| format!("deleting {rel}"))?;
            Ok(format!("deleted {rel}"))
        }
        "run_shell" => {
            let command = arg_str("command")?;
            let why = args.get("why").and_then(|v| v.as_str()).unwrap_or("");
            gate(state, "run_shell", &command, &format!("run_shell: {command}\n({why})"), "user declined running this command").await?;
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(&command)
                .current_dir(&state.root)
                .output()
                .context("spawning shell")?;
            let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
            combined.push_str(&String::from_utf8_lossy(&output.stderr));
            combined.push_str(&format!("\n[exit code: {}]", output.status.code().unwrap_or(-1)));
            Ok(truncate(combined))
        }
        "run_tests" => {
            let command = match args.get("command").and_then(|v| v.as_str()) {
                Some(c) if !c.is_empty() => c.to_string(),
                _ => detect_test_command(&state.root)
                    .ok_or_else(|| anyhow::anyhow!("couldn't auto-detect a test command for this project — pass `command` explicitly"))?,
            };
            gate(state, "run_tests", &command, &format!("run_tests: {command}"), "user declined running tests").await?;
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(&command)
                .current_dir(&state.root)
                .output()
                .context("running tests")?;
            let mut raw = String::from_utf8_lossy(&output.stdout).to_string();
            raw.push_str(&String::from_utf8_lossy(&output.stderr));
            Ok(truncate(summarize_test_output(&raw, output.status.success())))
        }
        "git_commit" => {
            let message = arg_str("message")?;
            let paths: Vec<String> = args
                .get("paths")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            let stage_desc = if paths.is_empty() { "all changes".to_string() } else { paths.join(", ") };
            gate(state, "git_commit", &message, &format!("git_commit: \"{message}\" (staging: {stage_desc})"), "user declined this commit").await?;
            let mut add_args = vec!["add".to_string()];
            if paths.is_empty() {
                add_args.push("-A".to_string());
            } else {
                add_args.extend(paths);
            }
            let add_result = run_git(&state.root, &add_args.iter().map(String::as_str).collect::<Vec<_>>())?;
            let commit_result = run_git(&state.root, &["commit", "-m", &message])?;
            Ok(truncate(format!("{add_result}{commit_result}")))
        }
        "web_search" => {
            let query = arg_str("query")?;
            Ok(truncate(web::search(&query).await?))
        }
        "web_fetch" => {
            let url = arg_str("url")?;
            Ok(truncate(web::fetch(&url).await?))
        }
        "recon_tool" => {
            let tool = arg_str("tool")?;
            let args_list: Vec<String> = args
                .get("args")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let recon_invocation = format!("{tool} {}", args_list.join(" "));
            gate(state, "recon_tool", &recon_invocation, &format!("recon_tool: {recon_invocation}"), "user declined running this tool").await?;
            let db_path = graviton_core::db_path_for(&state.root, &state.index_dir)?;
            let conn = graviton_core::open_db(&db_path)?;
            recon_tools::run_and_index(&conn, &tool, &args_list)?;
            let output: String = conn
                .query_row("SELECT output FROM tool_runs ORDER BY id DESC LIMIT 1", [], |r| r.get(0))
                .unwrap_or_default();
            Ok(truncate(output))
        }
        "browser_navigate" | "browser_eval" | "browser_screenshot" | "browser_console" => {
            if state.browser.is_none() {
                state.browser = Some(BrowserSession::launch().await?);
            }
            let browser = state.browser.as_ref().unwrap();
            match name {
                "browser_navigate" => browser.navigate(&arg_str("url")?).await,
                "browser_eval" => browser.eval(&arg_str("script")?).await,
                "browser_screenshot" => {
                    let rel = arg_str("path")?;
                    let full = resolve_rel(&state.root, &rel)?;
                    browser.screenshot(&full).await
                }
                "browser_console" => Ok(browser.console_logs()),
                _ => unreachable!(),
            }
        }
        "search_code" | "semantic_search" => {
            let db_path = graviton_core::db_path_for(&state.root, &state.index_dir)?;
            if !db_path.exists() {
                anyhow::bail!("no index found — run `grv index` first");
            }
            let conn = graviton_core::open_db(&db_path)?;
            let outcome = prepare_search_tool(&conn, state.embed_model.as_deref(), name, args);
            drop(conn); // done with the connection before the async ranking step below
            match outcome {
                Some(outcome) => finish_search_outcome(&state.ollama_host, outcome).await,
                None => unreachable!(),
            }
        }
        other if dispatch_git_readonly(&state.root, other, args).is_some() => {
            dispatch_git_readonly(&state.root, other, args).unwrap()
        }
        other => {
            let Some(tool) = custom_tools::find(&state.custom_tools, other).cloned() else {
                anyhow::bail!("unknown tool '{other}'");
            };
            let command = tool.render_command(args)?;
            gate(
                state,
                &tool.name,
                &command,
                &format!("custom tool '{}': {command}", tool.name),
                &format!("user declined running custom tool '{}'", tool.name),
            ).await?;
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(&command)
                .current_dir(&state.root)
                .output()
                .with_context(|| format!("running custom tool '{}'", tool.name))?;
            let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
            combined.push_str(&String::from_utf8_lossy(&output.stderr));
            combined.push_str(&format!("\n[exit code: {}]", output.status.code().unwrap_or(-1)));
            Ok(truncate(combined))
        }
    }
}

/// Render the agent's self-reported plan as a checklist for the terminal —
/// the visible task-progress view for a long `grv run`.
pub(crate) fn format_plan(steps: &Value) -> String {
    let Some(steps) = steps.as_array() else {
        return String::new();
    };
    let mut out = String::from("\x1b[1;36mplan:\x1b[0m\n");
    for step in steps {
        let text = step.get("text").and_then(|v| v.as_str()).unwrap_or("?");
        let status = step.get("status").and_then(|v| v.as_str()).unwrap_or("pending");
        let mark = match status {
            "done" => "[x]",
            "in_progress" => "[~]",
            _ => "[ ]",
        };
        out.push_str(&format!("  {mark} {text}\n"));
    }
    out
}

/// Best-effort test-command detection from repo markers — a starting
/// point the model can override with an explicit `command`, not a claim
/// this covers every project layout.
fn detect_test_command(root: &Path) -> Option<String> {
    let has = |name: &str| root.join(name).exists();
    if has("Cargo.toml") {
        Some("cargo test".to_string())
    } else if has("go.mod") {
        Some("go test ./...".to_string())
    } else if has("pytest.ini") || has("pyproject.toml") || has("setup.py") {
        Some("pytest".to_string())
    } else if has("package.json") {
        Some("npm test".to_string())
    } else if has("Gemfile") {
        Some("bundle exec rspec".to_string())
    } else {
        None
    }
}

/// Heuristic test-output summarizer across common frameworks (cargo test,
/// pytest, jest/npm test, go test, rspec) — recognizes their usual
/// pass/fail summary lines and failing-test markers by substring, not a
/// real parser per framework. When nothing recognizable is found, falls
/// back to the tail of the raw output (where a test runner's summary
/// almost always ends up) rather than silently returning nothing useful.
fn summarize_test_output(raw: &str, success: bool) -> String {
    let is_failure_marker = |l: &str| {
        l.starts_with("FAILED ")
            || l.starts_with("FAIL ")
            || l.starts_with("--- FAIL:")
            || l.contains("AssertionError")
            || l.contains("panicked at")
            || l.starts_with('✗')
            || l.starts_with('✕')
    };
    let is_summary_marker = |l: &str| {
        let lower = l.to_lowercase();
        lower.contains("passed") || lower.contains("failed") || (lower.contains("tests:") && lower.contains("total"))
    };

    let failing: Vec<&str> = raw.lines().filter(|l| is_failure_marker(l.trim())).take(20).collect();
    let summary = raw.lines().rev().find(|l| is_summary_marker(l.trim()));

    let mut out = String::new();
    out.push_str(if success { "TESTS PASSED\n" } else { "TESTS FAILED\n" });
    if let Some(s) = summary {
        out.push_str(s.trim());
        out.push('\n');
    }
    if !failing.is_empty() {
        out.push_str("\nfailing:\n");
        for l in &failing {
            out.push_str(l.trim());
            out.push('\n');
        }
    }
    if summary.is_none() && failing.is_empty() {
        out.push_str("\n(no recognizable summary line — raw output tail)\n");
        let tail: Vec<&str> = raw.lines().rev().take(40).collect();
        out.push_str(&tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
    }
    out
}

fn list_dir(path: &Path, recursive: bool) -> Result<String> {
    let mut out = String::new();
    let walker = ignore::WalkBuilder::new(path)
        .hidden(true)
        .git_ignore(true)
        .max_depth(if recursive { None } else { Some(1) })
        .build();
    for entry in walker.flatten() {
        if entry.path() == path {
            continue;
        }
        let kind = if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) { "/" } else { "" };
        out.push_str(&format!("{}{}\n", entry.path().display(), kind));
    }
    Ok(out)
}

/// Run the agentic loop: `agent` answers `task`, with tools, until it stops
/// calling them or `MAX_STEPS` is hit.
///
/// `resume_session`: if set, reopen that checkpoint session (so file-change
/// step numbering keeps counting up, not restarting at 0) and restore its
/// saved conversation transcript instead of starting fresh. `task` becomes
/// an *additional* instruction appended after the restored history — pass
/// an empty string to just continue the loop from wherever it left off
/// (e.g. it hit `MAX_STEPS` last time) with no new instruction.
pub async fn run(
    cfg: &Config,
    root: &Path,
    agent: &AgentSpec,
    task: &str,
    initial_context: Option<String>,
    loop_cfg: AgentLoopConfig,
    resume_session: Option<String>,
    io: Arc<dyn RunIo>,
) -> Result<()> {
    let custom = custom_tools::load_all(root);
    if !custom.is_empty() {
        io.emit(format!("\x1b[2mcustom tools loaded: {}\x1b[0m", custom.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", ")));
    }
    let rules = permissions::load(root);
    if !rules.is_empty() {
        io.emit(format!("\x1b[2m{} permission rule(s) loaded from .graviton/permissions.toml\x1b[0m", rules.len()));
    }
    let tools = tool_defs(loop_cfg.enable_browser, &custom);

    let (mut state, mut messages, resumed) = match &resume_session {
        Some(id) => {
            let checkpoint = checkpoint::Session::open_existing(root, id)?;
            let restored = checkpoint::load_transcript(root, id)?;
            (
                State {
                    root: root.to_path_buf(),
                    checkpoint,
                    auto_approve: loop_cfg.auto_approve,
                    browser: None,
                    custom_tools: custom,
                    permissions: rules,
                    index_dir: cfg.index_dir.clone(),
                    ollama_host: cfg.ollama_host.clone(),
                    embed_model: cfg.embed_model.clone(),
                    io: io.clone(),
                },
                restored,
                true,
            )
        }
        None => (
            State {
                root: root.to_path_buf(),
                checkpoint: checkpoint::Session::new(root)?,
                auto_approve: loop_cfg.auto_approve,
                browser: None,
                custom_tools: custom,
                permissions: rules,
                index_dir: cfg.index_dir.clone(),
                ollama_host: cfg.ollama_host.clone(),
                embed_model: cfg.embed_model.clone(),
                io: io.clone(),
            },
            Vec::new(),
            false,
        ),
    };

    io.note_checkpoint_id(&state.checkpoint.id);
    io.emit(format!("\x1b[2mcheckpoint session: {}\x1b[0m", state.checkpoint.id));

    if resumed {
        io.emit(format!("\x1b[2mresumed — {} prior message(s) restored\x1b[0m", messages.len()));
        if let Ok(Some(plan)) = checkpoint::load_plan(root, &state.checkpoint.id) {
            io.emit(format_plan(plan.get("steps").unwrap_or(&plan)));
        }
    }

    if messages.is_empty() {
        // Fresh run (or a resume of a session with no saved transcript,
        // e.g. one from before this feature existed) — build the initial
        // system + user turn as before.
        let system = format!(
            "{}\n\nYou have tools to read/write/edit/delete files, run shell commands, \
             run tests, inspect real git state (status/diff/log) and commit, run recon \
             tools, search the web and fetch pages, search the indexed codebase (search_code \
             for exact identifiers/strings, semantic_search for concepts when it's available), \
             report your plan, and (if offered) drive a headless browser — use them; don't \
             just describe what you'd do. Paths are relative to the repo root. File writes/edits/deletes, commits, \
             and shell commands are confirmed with the user before they happen, so \
             propose them directly rather than asking permission in text first. Use \
             web_search/web_fetch whenever the task depends on something that could have \
             changed since your training — a library's current API, a recent CVE, a best \
             practice — instead of answering from memory and risking an obsolete \
             technique; a wrong answer that looks current is worse than admitting you \
             need to check. Use update_plan for any task with more than one real step, \
             and keep it current. After a code change, run_tests before declaring the \
             task done — a change that hasn't been run isn't verified, it's a guess. \
             When the task is complete, stop calling tools and give a final summary of \
             what you did and the result.",
            agent.system_prompt
        );
        push_and_record(&mut messages, &state, ChatMessage::system(system));
        let user_msg = match initial_context {
            Some(ctx) if !ctx.is_empty() => format!("Task: {task}\n\nRetrieved context:\n{ctx}"),
            _ => format!("Task: {task}"),
        };
        push_and_record(&mut messages, &state, ChatMessage::user(user_msg));
    } else if !task.trim().is_empty() {
        push_and_record(&mut messages, &state, ChatMessage::user(format!("Additional instruction: {task}")));
    }

    let client = OllamaClient::new(&cfg.ollama_host);
    io.emit(format!("\x1b[1;35m═══ {} (agentic) ═══\x1b[0m", agent.display));

    for step in 0..MAX_STEPS {
        let result = client
            .chat_stream(cfg.model_for_tier(agent.tier), &messages, cfg.num_ctx, &tools, |tok| io.on_token(tok))
            .await?;
        io.emit(String::new()); // blank line after the streamed answer, matching the terminal's old unconditional println!()

        if result.tool_calls.is_empty() {
            push_and_record(&mut messages, &state, ChatMessage::assistant(result.content));
            print_checkpoint_summary(&state);
            return Ok(());
        }

        push_and_record(&mut messages, &state, ChatMessage::assistant_tool_calls(clone_tool_calls(&result.tool_calls)));
        for call in &result.tool_calls {
            let name = &call.function.name;
            io.emit(format!("\x1b[2m→ {name}({})\x1b[0m", call.function.arguments));
            let output = dispatch(&mut state, name, &call.function.arguments).await;
            io.emit(format!("\x1b[2m← {}\x1b[0m", truncate_display(&output)));
            push_and_record(&mut messages, &state, ChatMessage::tool_result(name.clone(), output));
        }

        if step == MAX_STEPS - 1 {
            io.emit(format!(
                "\x1b[1;31m[stopped: reached the {MAX_STEPS}-step limit for this run — \
                 `grv run --continue {}` to keep going]\x1b[0m",
                state.checkpoint.id
            ));
        }
    }
    print_checkpoint_summary(&state);
    Ok(())
}

/// Push a message onto the in-memory conversation *and* the session's
/// on-disk transcript in one call, so the two can never drift apart.
fn push_and_record(messages: &mut Vec<ChatMessage>, state: &State, msg: ChatMessage) {
    state.checkpoint.append_message(&msg);
    messages.push(msg);
}

fn print_checkpoint_summary(state: &State) {
    let n = state.checkpoint.step_count();
    if n > 0 {
        state.io.emit(format!(
            "\x1b[2m{n} file change(s) checkpointed under session {} — `grv rollback {}` to undo\x1b[0m",
            state.checkpoint.id, state.checkpoint.id
        ));
    }
}

fn clone_tool_calls(calls: &[ToolCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .map(|c| ToolCall {
            function: graviton_llm::ToolCallFunction {
                name: c.function.name.clone(),
                arguments: c.function.arguments.clone(),
            },
        })
        .collect()
}

fn truncate_display(s: &str) -> String {
    if s.len() > 300 {
        format!("{}...", &s[..300])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_cargo_test_failure() {
        let raw = "running 2 tests\ntest foo::works ... ok\ntest foo::broken ... FAILED\n\nfailures:\n\n---- foo::broken stdout ----\nthread panicked at src/foo.rs:10: assertion failed\n\nfailures:\n    foo::broken\n\ntest result: FAILED. 1 passed; 1 failed; 0 ignored\n";
        let out = summarize_test_output(raw, false);
        assert!(out.starts_with("TESTS FAILED"));
        assert!(out.contains("1 passed; 1 failed"));
        assert!(out.contains("panicked at"));
    }

    #[test]
    fn summarizes_pass_with_no_recognizable_failures() {
        let raw = "running 3 tests\n...\ntest result: ok. 3 passed; 0 failed; 0 ignored\n";
        let out = summarize_test_output(raw, true);
        assert!(out.starts_with("TESTS PASSED"));
        assert!(out.contains("3 passed"));
    }

    #[test]
    fn falls_back_to_tail_when_unrecognizable() {
        let raw = "some completely custom test runner output\nwith no known markers\n";
        let out = summarize_test_output(raw, false);
        assert!(out.contains("raw output tail"));
        assert!(out.contains("some completely custom test runner output"));
    }
}

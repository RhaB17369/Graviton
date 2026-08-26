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
use crate::tools as recon_tools;
use crate::web;
use anyhow::{Context, Result};
use graviton_core::Config;
use graviton_llm::{ChatMessage, OllamaClient, ToolCall, ToolDef};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};

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
    ]
}

/// Dispatch one of `read_only_tool_defs`' tools outside the full agentic
/// loop's `State`/checkpoint machinery — used by `mission.rs`, which has no
/// write/edit/delete tools to checkpoint in the first place.
pub(crate) async fn dispatch_read_only(root: &Path, name: &str, args: &Value) -> String {
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
            other => anyhow::bail!("unknown read-only tool '{other}'"),
        }
    }
    .await;
    match result {
        Ok(s) => truncate(s),
        Err(e) => format!("error: {e:#}"),
    }
}

fn tool_defs(enable_browser: bool) -> Vec<ToolDef> {
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

    tools
}

/// A confirmation gate's outcome. `Redirect` is the mid-task steering
/// mechanism: typing anything other than y/n at a confirmation prompt
/// declines the action *and* carries the typed text back to the model as
/// the tool result, so "no, do X instead" actually reaches the next turn
/// instead of only being expressible as a blind yes/no. This only fires at
/// existing confirmation pauses — a `--yolo` run has none, so it can only
/// be interrupted with Ctrl-C, which this doesn't change.
enum Decision {
    Allow,
    Deny,
    Redirect(String),
}

fn confirm(auto_approve: bool, action: &str) -> Decision {
    if auto_approve {
        return Decision::Allow;
    }
    print!("\x1b[1;33m{action}\nallow? [y/N, or type a note to redirect the agent instead] \x1b[0m");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return Decision::Deny;
    }
    match line.trim() {
        "y" | "Y" | "yes" | "Yes" => Decision::Allow,
        "" | "n" | "N" | "no" | "No" => Decision::Deny,
        other => Decision::Redirect(other.to_string()),
    }
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

async fn dispatch(state: &mut State, conn: &rusqlite::Connection, name: &str, args: &Value) -> String {
    match dispatch_inner(state, conn, name, args).await {
        Ok(s) => s,
        Err(e) => format!("error: {e:#}"),
    }
}

async fn dispatch_inner(state: &mut State, conn: &rusqlite::Connection, name: &str, args: &Value) -> Result<String> {
    let arg_str = |key: &str| -> Result<String> {
        args.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .with_context(|| format!("missing '{key}' argument"))
    };

    match name {
        "update_plan" => {
            let steps = args.get("steps").cloned().unwrap_or(Value::Array(vec![]));
            println!("{}", format_plan(&steps));
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
            require_allowed(confirm(state.auto_approve, &format!("write_file {rel}\n{preview}")), "user declined this write")?;
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
            require_allowed(confirm(state.auto_approve, &format!("edit_file {rel}\n{preview}")), "user declined this edit")?;
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
            require_allowed(confirm(state.auto_approve, &format!("delete_file {rel}")), "user declined this delete")?;
            state.checkpoint.snapshot_before_delete(&rel)?;
            std::fs::remove_file(&full).with_context(|| format!("deleting {rel}"))?;
            Ok(format!("deleted {rel}"))
        }
        "run_shell" => {
            let command = arg_str("command")?;
            let why = args.get("why").and_then(|v| v.as_str()).unwrap_or("");
            require_allowed(confirm(state.auto_approve, &format!("run_shell: {command}\n({why})")), "user declined running this command")?;
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
            require_allowed(confirm(state.auto_approve, &format!("recon_tool: {tool} {}", args_list.join(" "))), "user declined running this tool")?;
            recon_tools::run_and_index(conn, &tool, &args_list)?;
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
        other => anyhow::bail!("unknown tool '{other}'"),
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
    conn: &rusqlite::Connection,
    root: &Path,
    agent: &AgentSpec,
    task: &str,
    initial_context: Option<String>,
    loop_cfg: AgentLoopConfig,
    resume_session: Option<String>,
) -> Result<()> {
    let tools = tool_defs(loop_cfg.enable_browser);

    let (mut state, mut messages, resumed) = match &resume_session {
        Some(id) => {
            let checkpoint = checkpoint::Session::open_existing(root, id)?;
            let restored = checkpoint::load_transcript(root, id)?;
            (
                State { root: root.to_path_buf(), checkpoint, auto_approve: loop_cfg.auto_approve, browser: None },
                restored,
                true,
            )
        }
        None => (
            State { root: root.to_path_buf(), checkpoint: checkpoint::Session::new(root)?, auto_approve: loop_cfg.auto_approve, browser: None },
            Vec::new(),
            false,
        ),
    };

    println!("\x1b[2mcheckpoint session: {}\x1b[0m", state.checkpoint.id);

    if resumed {
        println!("\x1b[2mresumed — {} prior message(s) restored\x1b[0m", messages.len());
        if let Ok(Some(plan)) = checkpoint::load_plan(root, &state.checkpoint.id) {
            println!("{}", format_plan(plan.get("steps").unwrap_or(&plan)));
        }
    }

    if messages.is_empty() {
        // Fresh run (or a resume of a session with no saved transcript,
        // e.g. one from before this feature existed) — build the initial
        // system + user turn as before.
        let system = format!(
            "{}\n\nYou have tools to read/write/edit/delete files, run shell commands, \
             run recon tools, search the web and fetch pages, report your plan, and (if \
             offered) drive a headless browser — use them; don't just describe what you'd \
             do. Paths are relative to the repo root. File writes/edits/deletes and shell \
             commands are confirmed with the user before they happen, so propose them \
             directly rather than asking permission in text first. Use web_search/\
             web_fetch whenever the task depends on something that could have changed \
             since your training — a library's current API, a recent CVE, a best \
             practice — instead of answering from memory and risking an obsolete \
             technique; a wrong answer that looks current is worse than admitting you \
             need to check. Use update_plan for any task with more than one real step, \
             and keep it current. When the task is complete, stop calling tools and give \
             a final summary of what you did and the result.",
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
    println!("\x1b[1;35m═══ {} (agentic) ═══\x1b[0m", agent.display);

    for step in 0..MAX_STEPS {
        let stdout = std::io::stdout();
        let result = client
            .chat_stream(cfg.model_for_tier(agent.tier), &messages, cfg.num_ctx, &tools, |tok| {
                let mut lock = stdout.lock();
                let _ = lock.write_all(tok.as_bytes());
                let _ = lock.flush();
            })
            .await?;
        println!();

        if result.tool_calls.is_empty() {
            push_and_record(&mut messages, &state, ChatMessage::assistant(result.content));
            print_checkpoint_summary(&state);
            return Ok(());
        }

        push_and_record(&mut messages, &state, ChatMessage::assistant_tool_calls(clone_tool_calls(&result.tool_calls)));
        for call in &result.tool_calls {
            let name = &call.function.name;
            println!("\x1b[2m→ {name}({})\x1b[0m", call.function.arguments);
            let output = dispatch(&mut state, conn, name, &call.function.arguments).await;
            println!("\x1b[2m← {}\x1b[0m", truncate_display(&output));
            push_and_record(&mut messages, &state, ChatMessage::tool_result(name.clone(), output));
        }

        if step == MAX_STEPS - 1 {
            println!(
                "\x1b[1;31m[stopped: reached the {MAX_STEPS}-step limit for this run — \
                 `grv run --continue {}` to keep going]\x1b[0m",
                state.checkpoint.id
            );
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
        println!(
            "\x1b[2m{n} file change(s) checkpointed under session {} — `grv rollback {}` to undo\x1b[0m",
            state.checkpoint.id, state.checkpoint.id
        );
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

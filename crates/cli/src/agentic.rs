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

fn tool_defs(enable_browser: bool) -> Vec<ToolDef> {
    let mut tools = vec![
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
    ];

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

fn confirm(auto_approve: bool, action: &str) -> bool {
    if auto_approve {
        return true;
    }
    print!("\x1b[1;33m{action}\nallow? [y/N] \x1b[0m");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
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
            if !confirm(state.auto_approve, &format!("write_file {rel}\n{preview}")) {
                anyhow::bail!("user declined this write");
            }
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
            if !confirm(state.auto_approve, &format!("edit_file {rel}\n{preview}")) {
                anyhow::bail!("user declined this edit");
            }
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
            if !confirm(state.auto_approve, &format!("delete_file {rel}")) {
                anyhow::bail!("user declined this delete");
            }
            state.checkpoint.snapshot_before_delete(&rel)?;
            std::fs::remove_file(&full).with_context(|| format!("deleting {rel}"))?;
            Ok(format!("deleted {rel}"))
        }
        "run_shell" => {
            let command = arg_str("command")?;
            let why = args.get("why").and_then(|v| v.as_str()).unwrap_or("");
            if !confirm(state.auto_approve, &format!("run_shell: {command}\n({why})")) {
                anyhow::bail!("user declined running this command");
            }
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
        "recon_tool" => {
            let tool = arg_str("tool")?;
            let args_list: Vec<String> = args
                .get("args")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            if !confirm(state.auto_approve, &format!("recon_tool: {tool} {}", args_list.join(" "))) {
                anyhow::bail!("user declined running this tool");
            }
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
pub async fn run(
    cfg: &Config,
    conn: &rusqlite::Connection,
    root: &Path,
    agent: &AgentSpec,
    task: &str,
    initial_context: Option<String>,
    loop_cfg: AgentLoopConfig,
) -> Result<()> {
    let tools = tool_defs(loop_cfg.enable_browser);
    let mut state = State {
        root: root.to_path_buf(),
        checkpoint: checkpoint::Session::new(root)?,
        auto_approve: loop_cfg.auto_approve,
        browser: None,
    };

    println!("\x1b[2mcheckpoint session: {}\x1b[0m", state.checkpoint.id);

    let system = format!(
        "{}\n\nYou have tools to read/write/edit/delete files, run shell commands, \
         run recon tools, and (if offered) drive a headless browser — use them; \
         don't just describe what you'd do. Paths are relative to the repo root. \
         File writes/edits/deletes and shell commands are confirmed with the user \
         before they happen, so propose them directly rather than asking permission \
         in text first. When the task is complete, stop calling tools and give a \
         final summary of what you did and the result.",
        agent.system_prompt
    );

    let mut messages = vec![ChatMessage::system(system)];
    let user_msg = match initial_context {
        Some(ctx) if !ctx.is_empty() => format!("Task: {task}\n\nRetrieved context:\n{ctx}"),
        _ => format!("Task: {task}"),
    };
    messages.push(ChatMessage::user(user_msg));

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
            print_checkpoint_summary(&state);
            return Ok(());
        }

        messages.push(ChatMessage::assistant_tool_calls(clone_tool_calls(&result.tool_calls)));
        for call in &result.tool_calls {
            let name = &call.function.name;
            println!("\x1b[2m→ {name}({})\x1b[0m", call.function.arguments);
            let output = dispatch(&mut state, conn, name, &call.function.arguments).await;
            println!("\x1b[2m← {}\x1b[0m", truncate_display(&output));
            messages.push(ChatMessage::tool_result(name.clone(), output));
        }

        if step == MAX_STEPS - 1 {
            println!("\x1b[1;31m[stopped: reached the {MAX_STEPS}-step limit for this run]\x1b[0m");
        }
    }
    print_checkpoint_summary(&state);
    Ok(())
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

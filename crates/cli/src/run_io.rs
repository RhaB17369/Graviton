//! Pluggable output + confirmation sink for `grv run`'s agentic loop
//! (`agentic::run`), so the exact same loop can be driven from a terminal
//! (`grv run` itself, `TerminalIo`) or from `grv serve`'s `run_start`/
//! `run_confirm` RPC methods (`daemon::ChannelIo`) — one loop, two front
//! ends, instead of the daemon needing a second copy to stay in sync with.
//!
//! `confirm` is boxed (`Pin<Box<dyn Future<...>>>`) rather than a plain
//! `async fn` in the trait: `dyn RunIo` needs object safety (the loop
//! doesn't know at compile time which implementation it's running with),
//! and native async-fn-in-traits doesn't support dynamic dispatch without
//! this by hand (or the `async-trait` crate, which this just as easily
//! avoids needing for one method).

use crate::agentic::Decision;
use serde_json::Value;
use std::future::Future;
use std::io::Write;
use std::pin::Pin;

pub trait RunIo: Send + Sync {
    /// One line of output — plan updates, tool-call announcements, the
    /// step-limit warning, the checkpoint summary. Already formatted
    /// (ANSI color codes included where the terminal wants them); a
    /// non-terminal implementation is free to strip or ignore that.
    fn emit(&self, line: String);

    /// One incremental piece of the model's streamed final-answer text
    /// (never called during a tool-calling turn — those turns produce no
    /// content, only tool_calls).
    fn on_token(&self, tok: &str);

    /// One incremental piece of a thinking-capable model's reasoning
    /// trace (`message.thinking`, not `message.content` -- see
    /// `graviton_llm::chat_stream`'s doc comment). Default no-op: a
    /// reasoning trace isn't the final answer `grv run`'s terminal
    /// output is built around, so only callers that want to show it
    /// (the chat TUI) override this.
    fn on_thinking(&self, _tok: &str) {}

    /// A tool call's invocation, before it runs. Default: the same
    /// `emit`-formatted "→ name(args)" line `grv run`'s terminal output
    /// has always had. The chat TUI overrides this to group every call
    /// from one turn into a single collapsible entry instead.
    fn on_tool_call(&self, name: &str, args: &Value) {
        self.emit(format!("\x1b[2m→ {name}({args})\x1b[0m"));
    }

    /// A tool call's result, already truncated to a sane display size by
    /// the caller. Default: the same `emit`-formatted "← result" line.
    fn on_tool_result(&self, _name: &str, result: &str) {
        self.emit(format!("\x1b[2m← {result}\x1b[0m"));
    }

    /// The agent's current plan (`update_plan`'s `steps`, each a
    /// `{text, status}` object -- `status` one of "pending"/
    /// "in_progress"/"done"). Default: `grv run`'s existing text block
    /// (`agentic::format_plan`). The chat TUI overrides this to drive a
    /// standing checklist panel instead of a transcript line, so a
    /// revised plan replaces the old one in place rather than printing
    /// a second copy underneath it.
    fn on_plan(&self, steps: &Value) {
        self.emit(crate::agentic::format_plan(steps));
    }

    /// Once known (right after `checkpoint::Session` is created/reopened),
    /// the session id `grv checkpoints`/`grv rollback`/`grv plan` use.
    /// Default no-op — `TerminalIo` doesn't need it (it's already printed
    /// via `emit`); `ChannelIo` records it for `run_status`.
    fn note_checkpoint_id(&self, _id: &str) {}

    /// Ask for a decision on a gated action (a write/edit/delete/shell/
    /// commit/recon/custom-tool call not covered by a permissions.toml
    /// rule). `auto_approve` is `--yolo`'s value, passed through so an
    /// implementation doesn't need its own copy. `tool` is the gate's own
    /// tool name (`"write_file"`, `"run_shell"`, ...) -- separate from the
    /// formatted `action` text so an implementation that wants to treat
    /// tool types differently (the chat TUI's semi-automatic mode:
    /// auto-accept edits, still ask before a shell command or a delete)
    /// doesn't have to pattern-match the human-readable string to do it.
    fn confirm(&self, auto_approve: bool, tool: &str, action: String) -> Pin<Box<dyn Future<Output = Decision> + Send + '_>>;

    /// Ask a clarifying question with a fixed set of options (the
    /// `ask_user` tool) — the agent's own version of the AskUserQuestion
    /// pattern: for an ambiguous task where the sane answers form a short,
    /// known set, present them instead of guessing or asking in free text
    /// the agent then has to parse back out of a text reply. Returns the
    /// chosen option label(s) verbatim, empty if nothing was picked.
    fn ask_choice(&self, question: String, options: Vec<String>, multi_select: bool) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>>;
}

/// The original `grv run` behavior: everything to stdout, confirmation via
/// a blocking stdin read.
pub struct TerminalIo;

impl RunIo for TerminalIo {
    fn emit(&self, line: String) {
        println!("{line}");
    }

    fn on_token(&self, tok: &str) {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        let _ = lock.write_all(tok.as_bytes());
        let _ = lock.flush();
    }

    fn confirm(&self, auto_approve: bool, _tool: &str, action: String) -> Pin<Box<dyn Future<Output = Decision> + Send + '_>> {
        Box::pin(async move {
            if auto_approve {
                return Decision::Allow;
            }
            // A blocking stdin read inside an async fn would stall the
            // whole runtime; `grv run` was always effectively single
            // threaded/blocking at this exact point anyway (nothing else
            // is happening while waiting on the human), but spawn_blocking
            // keeps it honest rather than relying on that coincidence.
            tokio::task::spawn_blocking(move || {
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
            })
            .await
            .unwrap_or(Decision::Deny)
        })
    }

    fn ask_choice(&self, question: String, options: Vec<String>, multi_select: bool) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                println!("\x1b[1;36m{question}\x1b[0m");
                for (i, opt) in options.iter().enumerate() {
                    println!("  [{}] {opt}", i + 1);
                }
                if multi_select {
                    print!("\x1b[1;33mpick one or more (comma-separated numbers, or 'all'): \x1b[0m");
                } else {
                    print!("\x1b[1;33mpick one (number): \x1b[0m");
                }
                std::io::stdout().flush().ok();
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line).is_err() {
                    return Vec::new();
                }
                let line = line.trim();
                if multi_select && line.eq_ignore_ascii_case("all") {
                    return options;
                }
                let mut picked = Vec::new();
                for tok in line.split(',') {
                    if let Ok(n) = tok.trim().parse::<usize>() {
                        if n >= 1 && n <= options.len() {
                            picked.push(options[n - 1].clone());
                        }
                    }
                    if !multi_select {
                        break;
                    }
                }
                picked
            })
            .await
            .unwrap_or_default()
        })
    }
}

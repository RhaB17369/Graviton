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

    /// Once known (right after `checkpoint::Session` is created/reopened),
    /// the session id `grv checkpoints`/`grv rollback`/`grv plan` use.
    /// Default no-op — `TerminalIo` doesn't need it (it's already printed
    /// via `emit`); `ChannelIo` records it for `run_status`.
    fn note_checkpoint_id(&self, _id: &str) {}

    /// Ask for a decision on a gated action (a write/edit/delete/shell/
    /// commit/recon/custom-tool call not covered by a permissions.toml
    /// rule). `auto_approve` is `--yolo`'s value, passed through so an
    /// implementation doesn't need its own copy.
    fn confirm(&self, auto_approve: bool, action: String) -> Pin<Box<dyn Future<Output = Decision> + Send + '_>>;
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

    fn confirm(&self, auto_approve: bool, action: String) -> Pin<Box<dyn Future<Output = Decision> + Send + '_>> {
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
}

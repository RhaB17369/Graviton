//! Full-screen chat TUI for bare `grv` (no subcommand).
//!
//! Replaces the earlier plain `print!`/`stdin().read_line()` REPL, which
//! looked broken on this project's own (slow, CPU-only, often
//! contended) test hardware: a question could sit for minutes with zero
//! feedback, indistinguishable from a hang. This version has a real
//! input box pinned at the bottom, a scrolling transcript above it, and
//! a live elapsed-time spinner while a request is in flight. Pulls in
//! `ratatui` + `crossterm` + `textwrap`, a deliberate departure from
//! this project's usual zero-extra-dependency stance for scalar config
//! parsing — a real terminal UI needs a real terminal UI crate, and
//! correct line-wrapping needs a real wrapping crate (ratatui's own
//! `Paragraph::wrap` + `.scroll()` don't compose: `.scroll()` counts
//! pre-wrap logical lines, not the rows wrapping actually renders, so
//! a long streamed reply silently scrolled off-screen — see the
//! `visible_rows` doc comment below for the fix).

use crate::agentic::{self, Decision};
use crate::agents::{AgentSpec, ALL_AGENTS};
use crate::checkpoint;
use crate::run_io::RunIo;
use crate::{build_context_sync, finish_context, open_or_init_repo_db, repo_root, resolve_agent};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event, EventStream, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use futures_util::StreamExt;
use graviton_core::Config;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;
use std::future::Future;
use std::io::Stdout;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

/// One rendered row of the transcript. The in-progress agent reply is
/// the last `Agent` entry while `waiting` is true — tokens are appended
/// to its `text` in place, not pushed as new lines, so streaming looks
/// like one answer growing rather than a new line per token.
enum TLine {
    You(String),
    Agent { label: String, text: String },
    /// A thinking-capable model's reasoning trace (`message.thinking`,
    /// streamed live exactly like an answer's tokens -- see
    /// `graviton_llm::chat_stream`), shown dim and separate from the
    /// final answer so the (sometimes long) reasoning phase is visibly
    /// *something happening* instead of silence before the real reply.
    Thinking(String),
    /// Every tool call (and its result, once back) from one agentic
    /// turn, grouped together rather than one flat line per call --
    /// `ChatEvent::ToolCall` appends to this same entry as long as it's
    /// still the last transcript line, so a multi-step tool loop (read
    /// a file, search, read another) ends up as one collapsible group
    /// instead of a wall of `→`/`←` lines.
    Tools(Vec<ToolEntry>),
    /// A line from `agentic::run`'s `io.emit` (checkpoint banners,
    /// plan updates, the step-limit warning, ...) -- ANSI-stripped,
    /// since those calls format for a raw terminal and a literal
    /// escape byte would otherwise show up as garbage in a `Span`.
    Sys(String),
    /// A gated action (write/edit/delete/shell/commit/recon) waiting on
    /// a decision in Manual or Semi-automatic mode -- the action text
    /// (already ANSI-stripped, a diff preview for a write/edit) is
    /// rendered multi-line, never word-wrapped across its own line
    /// breaks (see `push_multiline`), so a diff stays readable.
    Confirm(String),
    /// The agent's `ask_user` tool -- a question with a fixed, numbered
    /// set of options, answered the same way `grv run`'s terminal
    /// prompt always has: type the number(s), or "all", Enter.
    Choice(String),
    Err(String),
}

struct ToolEntry {
    name: String,
    call: String,
    result: Option<String>,
}

enum ChatEvent {
    Token(String),
    Thinking(String),
    ToolCall(String, serde_json::Value),
    ToolResult(String),
    /// `agentic::run`'s `io.emit` -- plan/checkpoint/step-limit lines.
    Emit(String),
    /// `io.note_checkpoint_id` -- the session id this turn is (now)
    /// running under, saved so the *next* turn resumes the same
    /// checkpoint-backed conversation instead of starting fresh.
    CheckpointId(String),
    /// `io.on_plan` -- the agent's current step list, replacing whatever
    /// plan was shown before rather than adding a new copy underneath it.
    Plan(Vec<PlanStep>),
    /// The char length of the retrieved-context dump just attached to a
    /// fresh session's first message -- see `ChatState::context_extra`.
    ContextChars(usize),
    /// A gated action needs a decision -- `(tool, action, reply)`.
    Confirm(String, String, oneshot::Sender<Decision>),
    /// `ask_user` needs an answer -- `(question, options, multi_select, reply)`.
    Choice(String, Vec<String>, bool, oneshot::Sender<Vec<String>>),
    Done,
    Error(String),
}

/// Manual (ask every gated action, like `grv run` without `--yolo`),
/// Auto (approve everything, like `--yolo`), or Semi-automatic
/// (auto-accept `write_file`/`edit_file` -- checkpointed and
/// rollback-able either way -- but still ask before anything that runs
/// a command, deletes a file, or touches version control). Cycled with
/// Ctrl+T or set with `/mode manual|semi-auto|auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermissionMode {
    Manual,
    SemiAuto,
    Auto,
}

impl PermissionMode {
    fn label(self) -> &'static str {
        match self {
            PermissionMode::Manual => "manual",
            PermissionMode::SemiAuto => "semi-auto",
            PermissionMode::Auto => "auto",
        }
    }

    fn next(self) -> Self {
        match self {
            PermissionMode::Manual => PermissionMode::SemiAuto,
            PermissionMode::SemiAuto => PermissionMode::Auto,
            PermissionMode::Auto => PermissionMode::Manual,
        }
    }
}

/// A pending `ask_user` call -- kept until `state.input` (typed the
/// same way a confirm's y/n/redirect is) resolves it.
struct PendingChoice {
    options: Vec<String>,
    multi: bool,
    reply: oneshot::Sender<Vec<String>>,
}

/// One `update_plan` step -- `status` is `"pending"`, `"in_progress"`,
/// or `"done"` (unrecognized values render like `"pending"`).
struct PlanStep {
    text: String,
    status: String,
}

/// Ctrl+R reverse-history-search state, readline/bash-style: `query` is
/// what's been typed to search *for*, `match_idx` the `history` entry it
/// currently resolves to (displayed in the input box in place of the
/// query itself) -- `None` means no entry in `history` contains `query`.
struct HistorySearch {
    query: String,
    match_idx: Option<usize>,
}

struct ChatState {
    agent: &'static AgentSpec,
    cfg_model: String,
    transcript: Vec<TLine>,
    /// Chars, not a `String` -- `cursor` is a char index, and inserting/
    /// removing at an arbitrary position needs `Vec::insert`/`remove`
    /// rather than `String`'s byte-offset-only API (which would panic
    /// mid-UTF8-codepoint on accented input).
    input: Vec<char>,
    cursor: usize,
    waiting: bool,
    waiting_since: Instant,
    /// The in-flight turn's task, so Esc-while-waiting can cancel just
    /// this turn (`.abort()`) instead of the only other option being to
    /// quit the whole app.
    current_task: Option<tokio::task::AbortHandle>,
    /// Rows scrolled back from the live bottom. 0 = pinned to the
    /// newest content (auto-follows as tokens stream in); any other
    /// value freezes the view there until the user scrolls back down
    /// or sends a new message (`submit` resets this to 0 so a fresh
    /// question is never missed because you were reading history).
    scroll: usize,
    /// Global toggle (Tab) for every multi-call `TLine::Tools` group at
    /// once -- simpler than per-group expand state, and there's only
    /// ever one in-flight group to look at at a time anyway.
    tools_expanded: bool,
    /// Highlighted row in the `/agent` autocomplete popup, clamped to
    /// the *current* filtered list every time it's read (see `drive`) --
    /// never reset explicitly, since whatever it pointed at a keystroke
    /// ago may not exist in a narrower filter any more.
    suggestion_idx: usize,
    /// The only two themeable colors (`config.toml`'s `tui_user_color`/
    /// `tui_agent_color`); everything else (borders, errors, the
    /// in-flight spinner) carries semantic meaning a palette shouldn't
    /// override, so it stays fixed.
    user_color: Color,
    agent_color: Color,
    /// A message typed and submitted *while the previous one was still
    /// generating* -- queued here instead of sent immediately (the
    /// request loop isn't reentrant: one spawned task owns the
    /// checkpoint-backed conversation for the duration of a turn), then
    /// auto-submitted the moment `ChatEvent::Done`/`Error` clears
    /// `waiting`. Typing itself was never blocked on `waiting` to begin
    /// with (see `drive`) -- only submitting was, so you can draft,
    /// edit, or even switch agents with `/agent` mid-generation without
    /// losing anything you typed.
    queued: Option<String>,
    /// `agentic::run`'s checkpoint session id -- `None` starts a fresh
    /// one; `Some` resumes it, the same multi-turn-conversation
    /// mechanism `grv run --continue` uses, so the chat's own history
    /// lives in `.graviton`'s checkpoint store, not just in memory.
    /// Seeded from `grv --resume <id>` at startup; cleared by `/clear`
    /// or `/agent` (a real topic/agent change starts a new session).
    checkpoint_session: Option<String>,
    mode: PermissionMode,
    /// A gated action is waiting on y/n/a redirect note, typed into the
    /// ordinary input box and resolved on Enter (Esc denies). Takes
    /// priority over every other Enter/Esc meaning while set.
    pending_confirm: Option<oneshot::Sender<Decision>>,
    /// An `ask_user` call waiting on a number (or "all"), same idea.
    pending_choice: Option<PendingChoice>,
    /// The agent's current plan (`update_plan`), shown as a standing
    /// checklist panel above the transcript -- empty hides the panel
    /// entirely. Cleared by `/clear`/`/agent` (a fresh topic has no
    /// plan yet); *not* cleared on `Done`, so the finished checklist
    /// stays visible as a record of what the turn actually did.
    plan: Vec<PlanStep>,
    should_quit: bool,
    /// Every checkpoint session this process has touched (created via
    /// `submit`, or loaded via `/resume`/`--resume`), most-recent last --
    /// what `/resume`'s autocomplete offers, so picking one never
    /// requires remembering an id from `/checkpoints`. Deliberately not
    /// a live filesystem scan (that would need threading `root` into the
    /// render path for every keystroke); a session from a *different*
    /// `grv` run that this chat never touched simply isn't offered here,
    /// `/checkpoints` (which does hit the filesystem) is still the full
    /// list.
    known_sessions: Vec<String>,
    /// Previously *sent* questions (not slash-commands), oldest first --
    /// what ↑/↓ recall. `None` means "showing the live draft", same
    /// convention a shell's own history uses.
    history: Vec<String>,
    history_cursor: Option<usize>,
    /// The draft that was being typed when history browsing *or* a
    /// Ctrl+R search started -- restored once ↓ walks back past the
    /// newest history entry, or Esc/an empty-match Enter cancels a
    /// search. The two features never run at once, so one field covers
    /// both.
    history_draft: Vec<char>,
    /// `Some` while a Ctrl+R reverse-history-search is in progress --
    /// takes over the input box and most keys (see `drive`) until
    /// Enter/Esc resolves it.
    history_search: Option<HistorySearch>,
    /// Characters attributed to the checkpoint session's one-time
    /// "Retrieved context" dump (sent with the very first message of a
    /// session, then carried in every subsequent turn's history) -- not
    /// visible in the transcript itself, so it's tracked separately
    /// from `estimate_context_chars`, which only sees what's on screen.
    context_extra: usize,
    /// `cfg.num_ctx` at startup, to turn `estimate_context_chars` into a
    /// percentage for the header's rough "how full is the context
    /// window" indicator.
    num_ctx: usize,
}

/// Parses a 6-digit hex color (no `#`, e.g. `"22d3ee"`) into an RGB
/// `Color`, falling back to `default` on anything else -- missing
/// config, wrong length, non-hex characters.
fn theme_color(hex: Option<&str>, default: Color) -> Color {
    let Some(hex) = hex.filter(|h| h.len() == 6) else {
        return default;
    };
    let parse = |s: &str| u8::from_str_radix(s, 16).ok();
    match (parse(&hex[0..2]), parse(&hex[2..4]), parse(&hex[4..6])) {
        (Some(r), Some(g), Some(b)) => Color::Rgb(r, g, b),
        _ => default,
    }
}

const SPINNER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

pub async fn run_chat_tui(cfg: &Config, resume: Option<String>) -> Result<()> {
    let root = repo_root()?;
    let agent = resolve_agent("architect")?;

    let mut state = ChatState {
        agent,
        cfg_model: cfg.model_for(agent.tier, agent.key).to_string(),
        transcript: Vec::new(),
        input: Vec::new(),
        cursor: 0,
        waiting: false,
        waiting_since: Instant::now(),
        current_task: None,
        scroll: 0,
        tools_expanded: false,
        suggestion_idx: 0,
        user_color: theme_color(cfg.tui_user_color.as_deref(), Color::Cyan),
        agent_color: theme_color(cfg.tui_agent_color.as_deref(), Color::Magenta),
        queued: None,
        checkpoint_session: None,
        mode: match cfg.tui_default_mode.as_deref() {
            Some("semi-auto") | Some("semi") => PermissionMode::SemiAuto,
            Some("auto") => PermissionMode::Auto,
            _ => PermissionMode::Manual,
        },
        pending_confirm: None,
        pending_choice: None,
        plan: Vec::new(),
        should_quit: false,
        known_sessions: Vec::new(),
        history: Vec::new(),
        history_cursor: None,
        history_draft: Vec::new(),
        history_search: None,
        context_extra: 0,
        num_ctx: cfg.num_ctx,
    };

    if let Some(id) = resume {
        load_session_into(&mut state, &root, &id);
    }

    enable_raw_mode().context("enabling raw terminal mode")?;
    let mut stdout = std::io::stdout();
    // Mouse capture is kept for wheel scroll, even though it also hands
    // every mouse event -- including click-drag -- to the app instead of
    // the terminal, which is what breaks a terminal's *default*
    // click-drag text selection. Nearly every terminal emulator (xterm,
    // the VTE family -- gnome-terminal, etc. --, alacritty, kitty, foot,
    // wezterm, iTerm2, Windows Terminal) has the same escape hatch for
    // exactly this situation: holding Shift while selecting bypasses the
    // app's mouse capture and forces the terminal's own native selection,
    // which copies normally. Bracketed paste is unconditionally worth
    // having alongside it -- it's what makes a real paste (`Event::Paste`,
    // handled in `drive`) land as the actual pasted text instead of a
    // burst of raw bytes crossterm has to guess at.
    crossterm::execute!(stdout, EnterAlternateScreen, EnableMouseCapture, EnableBracketedPaste).context("entering alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("creating terminal")?;
    terminal.clear().ok();

    let result = drive(&mut terminal, &mut state, cfg, &root).await;
    let session = state.checkpoint_session.clone();

    disable_raw_mode().ok();
    crossterm::execute!(terminal.backend_mut(), DisableMouseCapture, DisableBracketedPaste, LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    // The whole point of the checkpoint-backed session surviving past
    // this process: quitting (Ctrl+C, Esc, /exit) doesn't have to mean
    // losing the conversation. Printed after leaving the alternate
    // screen, to the real terminal scrollback, so it's still visible
    // once the TUI is gone.
    if let Some(id) = session {
        println!("session {id} saved — resume with `grv --resume {id}` (or `grv run --continue {id}` for a one-shot task)");
    }
    result
}

/// Strips `agentic::run`'s own wrapping back off a stored user turn so
/// `/resume`/`--resume` replay shows the question that was actually
/// typed, not its internal framing (see the `format!` calls building
/// `"Task: ..."`/`"Additional instruction: ..."` in `agentic.rs`). Not
/// matching either prefix just returns the content as-is -- a cosmetic
/// heuristic, not a parser that can fail.
fn extract_question(content: &str) -> String {
    if let Some(rest) = content.strip_prefix("Task: ") {
        return rest.split("\n\nRetrieved context:").next().unwrap_or(rest).trim().to_string();
    }
    if let Some(rest) = content.strip_prefix("Additional instruction: ") {
        return rest.trim().to_string();
    }
    content.trim().to_string()
}

/// Switches to a different checkpoint-backed session -- `--resume` at
/// startup, or `/resume <id>` mid-chat -- replacing whatever's on screen
/// with that session's own history instead of silently continuing it
/// off-screen (which is all the old behavior did: the model saw the
/// restored transcript, the chat never showed it).
fn load_session_into(state: &mut ChatState, root: &std::path::Path, id: &str) {
    state.checkpoint_session = Some(id.to_string());
    state.transcript.clear();
    state.plan.clear();
    state.context_extra = 0;
    state.transcript.push(TLine::Sys(format!("(resumed session {id})")));
    match checkpoint::load_transcript(root, id) {
        Ok(messages) => {
            let mut first_user = true;
            for m in messages {
                match m.role.as_str() {
                    "user" => {
                        // The *first* user turn of a session carries the
                        // one-time "Retrieved context" dump that never
                        // shows up in the transcript panel itself --
                        // counted here so the context-usage estimate
                        // isn't silently missing it after a resume.
                        if first_user {
                            state.context_extra += m.content.chars().count();
                            first_user = false;
                        }
                        let text = extract_question(&m.content);
                        if !text.is_empty() {
                            state.transcript.push(TLine::You(text));
                        }
                    }
                    "assistant" if !m.content.trim().is_empty() => {
                        state.transcript.push(TLine::Agent { label: state.agent.display.to_string(), text: m.content });
                    }
                    _ => {}
                }
            }
        }
        Err(e) => state.transcript.push(TLine::Err(format!("{e:#}"))),
    }
    if let Ok(Some(plan)) = checkpoint::load_plan(root, id) {
        let steps = plan.get("steps").cloned().unwrap_or(plan);
        if let Some(arr) = steps.as_array() {
            state.plan = arr
                .iter()
                .map(|s| PlanStep {
                    text: s.get("text").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
                    status: s.get("status").and_then(|v| v.as_str()).unwrap_or("pending").to_string(),
                })
                .collect();
        }
    }
    if !state.known_sessions.iter().any(|s| s == id) {
        state.known_sessions.push(id.to_string());
    }
}

/// ↑ (no Ctrl) with input history to browse: starts by saving the live
/// draft, then walks backward one sent question at a time.
fn history_prev(state: &mut ChatState) {
    if state.history.is_empty() {
        return;
    }
    match state.history_cursor {
        None => {
            state.history_draft = state.input.clone();
            state.history_cursor = Some(state.history.len() - 1);
        }
        Some(i) if i > 0 => state.history_cursor = Some(i - 1),
        Some(_) => return,
    }
    if let Some(i) = state.history_cursor {
        state.input = state.history[i].chars().collect();
        state.cursor = state.input.len();
    }
}

/// ↓ -- walks forward, restoring the saved draft once it passes the
/// newest history entry (the same convention a shell's history uses).
fn history_next(state: &mut ChatState) {
    match state.history_cursor {
        None => {}
        Some(i) if i + 1 < state.history.len() => {
            state.history_cursor = Some(i + 1);
            state.input = state.history[i + 1].chars().collect();
            state.cursor = state.input.len();
        }
        Some(_) => {
            state.history_cursor = None;
            state.input = std::mem::take(&mut state.history_draft);
            state.cursor = state.input.len();
        }
    }
}

/// Re-runs a Ctrl+R search's query against `history`, most recent match
/// first, and mirrors the result into `state.input` for display -- the
/// input box always shows the *matched command*, never the query itself
/// (same convention bash's reverse-i-search uses).
fn history_search_update(state: &mut ChatState) {
    let query = state.history_search.as_ref().map(|s| s.query.clone()).unwrap_or_default();
    if query.is_empty() {
        if let Some(s) = state.history_search.as_mut() {
            s.match_idx = None;
        }
        state.input.clear();
        state.cursor = 0;
        return;
    }
    let q = query.to_lowercase();
    let idx = state.history.iter().enumerate().rev().find(|(_, h)| h.to_lowercase().contains(&q)).map(|(i, _)| i);
    if let Some(s) = state.history_search.as_mut() {
        s.match_idx = idx;
    }
    state.input = idx.map(|i| state.history[i].clone()).unwrap_or_default().chars().collect();
    state.cursor = state.input.len();
}

/// Ctrl+R again -- the next *older* entry still matching the same query.
fn history_search_next_older(state: &mut ChatState) {
    let Some(search) = state.history_search.as_ref() else { return };
    let query = search.query.to_lowercase();
    if query.is_empty() {
        return;
    }
    let start = search.match_idx.unwrap_or(state.history.len());
    let idx = state.history[..start].iter().enumerate().rev().find(|(_, h)| h.to_lowercase().contains(&query)).map(|(i, _)| i);
    if let Some(i) = idx {
        if let Some(s) = state.history_search.as_mut() {
            s.match_idx = Some(i);
        }
        state.input = state.history[i].chars().collect();
        state.cursor = state.input.len();
    }
}

const HELP_TEXT: &str = "keys\n  Enter             send (queues instead, while a reply is still generating)\n  Alt+Enter         insert a newline\n  ↑ / ↓             browse previously sent messages\n  Ctrl+R            search sent messages by substring (Ctrl+R again: older match, Enter: send it, Esc: cancel)\n  Ctrl+↑ / Ctrl+↓   scroll the transcript one line\n  PageUp / PageDown scroll the transcript a page\n  mouse wheel       scroll the transcript (hold Shift while selecting to copy text natively)\n  Tab               expand/collapse grouped tool calls\n  Ctrl+T            cycle permission mode (manual → semi-auto → auto)\n  Esc               cancel the running turn, else dismiss the popup, else quit\n  Ctrl+C            quit\ncommands\n  /agent <key>      switch specialist agent (resets the conversation)\n  /mode [name]      show/cycle/set the permission mode\n  /checkpoints      list checkpoint sessions\n  /rollback [id] [step]  undo file changes from a session\n  /resume <id>      switch to a different checkpoint session\n  /copy             copy the last reply to your system clipboard\n  /export [file]    save the conversation as Markdown\n  /clear            reset the conversation\n  /help             this message\n  /exit             quit\nheader shows [ctx ~N%] -- a rough estimate (chars/4) of how much of num_ctx this session is using";

async fn drive(terminal: &mut Terminal<CrosstermBackend<Stdout>>, state: &mut ChatState, cfg: &Config, root: &std::path::Path) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<ChatEvent>();
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    // How many transcript rows are visible at a time -- kept in sync with
    // the last draw so PageUp/PageDown jump by a sensible amount instead
    // of a hardcoded guess that drifts from the real terminal size.
    let mut page_rows: usize = 10;

    terminal.draw(|f| page_rows = draw(f, state))?;

    loop {
        tokio::select! {
            maybe_ev = events.next() => {
                let Some(Ok(ev)) = maybe_ev else { continue };
                match ev {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        // A gated action or an ask_user call takes over
                        // Enter/Esc completely -- it's answered through
                        // the same input box (y/n/a redirect note; a
                        // number or "all"), but it isn't a *message*, so
                        // it must resolve before anything else (the
                        // autocomplete popup, queuing, quitting) gets a
                        // say in what Enter/Esc do.
                        let confirm_active = state.pending_confirm.is_some();
                        let choice_active = state.pending_choice.is_some();

                        // The autocomplete popup, when showing, steals
                        // Up/Down/Tab/Enter/Esc for its own navigation --
                        // computed fresh from the current input on every
                        // keystroke, never stored, so there's nothing to
                        // go stale. Suppressed while answering a
                        // confirm/choice -- that answer is never a
                        // command.
                        let input_text: String = state.input.iter().collect();
                        let searching = state.history_search.is_some();
                        let suggestions = if confirm_active || choice_active || searching { Vec::new() } else { command_suggestions(&input_text, &state.known_sessions) };
                        let popup_active = !suggestions.is_empty();
                        if popup_active {
                            state.suggestion_idx = state.suggestion_idx.min(suggestions.len() - 1);
                        }

                        if confirm_active && key.code == KeyCode::Enter {
                            let line: String = state.input.iter().collect::<String>().trim().to_string();
                            state.input.clear();
                            state.cursor = 0;
                            if let Some(reply) = state.pending_confirm.take() {
                                let _ = reply.send(parse_confirm_answer(&line));
                            }
                        } else if confirm_active && key.code == KeyCode::Esc {
                            state.input.clear();
                            state.cursor = 0;
                            if let Some(reply) = state.pending_confirm.take() {
                                let _ = reply.send(Decision::Deny);
                            }
                        } else if choice_active && key.code == KeyCode::Enter {
                            let line: String = state.input.iter().collect::<String>().trim().to_string();
                            state.input.clear();
                            state.cursor = 0;
                            if let Some(pending) = state.pending_choice.take() {
                                let picked = parse_choice_answer(&line, &pending.options, pending.multi);
                                let _ = pending.reply.send(picked);
                            }
                        } else if choice_active && key.code == KeyCode::Esc {
                            state.input.clear();
                            state.cursor = 0;
                            if let Some(pending) = state.pending_choice.take() {
                                let _ = pending.reply.send(Vec::new());
                            }
                        } else if searching {
                            // A Ctrl+R search takes over the input box
                            // entirely (it's showing the *matched*
                            // history entry, not a line being typed) --
                            // every key below is about the search itself,
                            // not ordinary editing.
                            match key.code {
                                KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => history_search_next_older(state),
                                KeyCode::Char(c) => {
                                    if let Some(s) = state.history_search.as_mut() {
                                        s.query.push(c);
                                    }
                                    history_search_update(state);
                                }
                                KeyCode::Backspace => {
                                    if let Some(s) = state.history_search.as_mut() {
                                        s.query.pop();
                                    }
                                    history_search_update(state);
                                }
                                KeyCode::Enter => {
                                    state.history_search = None;
                                    if state.input.is_empty() {
                                        state.input = std::mem::take(&mut state.history_draft);
                                        state.cursor = state.input.len();
                                    } else {
                                        submit_or_queue(state, cfg, root, &tx);
                                    }
                                }
                                KeyCode::Esc => {
                                    state.history_search = None;
                                    state.input = std::mem::take(&mut state.history_draft);
                                    state.cursor = state.input.len();
                                }
                                _ => {}
                            }
                        } else if key.code == KeyCode::Char('r') && key.modifiers.contains(KeyModifiers::CONTROL) && !state.waiting {
                            state.history_draft = state.input.clone();
                            state.input.clear();
                            state.cursor = 0;
                            state.history_search = Some(HistorySearch { query: String::new(), match_idx: None });
                        } else if key.code == KeyCode::Char('t') && key.modifiers.contains(KeyModifiers::CONTROL) {
                            state.mode = state.mode.next();
                            state.transcript.push(TLine::Sys(format!("(permission mode: {})", state.mode.label())));
                        } else if key.code == KeyCode::Esc && state.queued.is_some() {
                            state.queued = None;
                        } else if popup_active && key.code == KeyCode::Esc {
                            state.input.clear();
                            state.cursor = 0;
                        } else if key.code == KeyCode::Esc && state.waiting {
                            // Esc while a turn is actually running cancels
                            // *that turn* -- the app and the checkpoint
                            // session underneath it stay alive, same as
                            // Ctrl+C has always meant "stop this agent"
                            // for `grv run`, just without also closing
                            // the chat. Ctrl+C (below) still always means
                            // quit.
                            if let Some(h) = state.current_task.take() {
                                h.abort();
                            }
                            state.pending_confirm = None;
                            state.pending_choice = None;
                            state.queued = None;
                            state.waiting = false;
                            state.transcript.push(TLine::Sys("(cancelled)".into()));
                        } else if key.code == KeyCode::Esc || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)) {
                            if let Some(h) = state.current_task.take() {
                                h.abort();
                            }
                            state.should_quit = true;
                        } else if popup_active && key.code == KeyCode::Up {
                            state.suggestion_idx = state.suggestion_idx.saturating_sub(1);
                        } else if popup_active && key.code == KeyCode::Down {
                            state.suggestion_idx = (state.suggestion_idx + 1).min(suggestions.len() - 1);
                        } else if popup_active && key.code == KeyCode::Tab {
                            state.input = suggestions[state.suggestion_idx].complete.chars().collect();
                            state.cursor = state.input.len();
                        } else if popup_active && key.code == KeyCode::Enter {
                            let complete = suggestions[state.suggestion_idx].complete.clone();
                            state.input = complete.chars().collect();
                            state.cursor = state.input.len();
                            // "/agent " alone (no key yet, just the
                            // generic placeholder entry) isn't a
                            // submittable command -- land the cursor
                            // ready to keep typing instead of queuing/
                            // sending a bare "/agent" question.
                            if complete != "/agent " {
                                submit_or_queue(state, cfg, root, &tx);
                            }
                        } else {
                            match key.code {
                                // Plain ↑/↓ browse input history (shell-
                                // style); Ctrl+↑/↓ scroll the transcript
                                // one line instead, since both can't have
                                // the bare arrow keys. PageUp/PageDown and
                                // the mouse wheel remain the easy way to
                                // scroll casually.
                                KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => state.scroll = state.scroll.saturating_add(1),
                                KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => state.scroll = state.scroll.saturating_sub(1),
                                KeyCode::Up => history_prev(state),
                                KeyCode::Down => history_next(state),
                                KeyCode::PageUp => state.scroll = state.scroll.saturating_add(page_rows),
                                KeyCode::PageDown => state.scroll = state.scroll.saturating_sub(page_rows),
                                KeyCode::Tab => state.tools_expanded = !state.tools_expanded,
                                // Alt, not Shift: most terminals collapse
                                // Shift+Enter to plain Enter in raw mode
                                // (no distinct escape sequence without
                                // extra keyboard-protocol negotiation),
                                // while Alt reliably arrives as its own
                                // modifier bit everywhere since it's just
                                // the ESC-prefix trick terminals have
                                // used for Alt since forever.
                                KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
                                    state.input.insert(state.cursor, '\n');
                                    state.cursor += 1;
                                }
                                KeyCode::Enter => submit_or_queue(state, cfg, root, &tx),
                                KeyCode::Backspace => {
                                    if state.cursor > 0 {
                                        state.cursor -= 1;
                                        state.input.remove(state.cursor);
                                    }
                                }
                                KeyCode::Delete => {
                                    if state.cursor < state.input.len() {
                                        state.input.remove(state.cursor);
                                    }
                                }
                                KeyCode::Left => state.cursor = state.cursor.saturating_sub(1),
                                KeyCode::Right => state.cursor = (state.cursor + 1).min(state.input.len()),
                                KeyCode::Home => state.cursor = 0,
                                KeyCode::End => state.cursor = state.input.len(),
                                KeyCode::Char(c) => {
                                    state.input.insert(state.cursor, c);
                                    state.cursor += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                    Event::Mouse(m) => match m.kind {
                        MouseEventKind::ScrollUp => state.scroll = state.scroll.saturating_add(3),
                        MouseEventKind::ScrollDown => state.scroll = state.scroll.saturating_sub(3),
                        _ => {}
                    },
                    // Bracketed paste delivers the whole clipboard text as
                    // one event instead of a key-by-key burst, so a paste
                    // containing anything unusual (newlines, a lone `\r`,
                    // a character crossterm's plain key parser might not
                    // round-trip) still lands intact. Inserted the same
                    // way typed characters are -- multi-line content just
                    // becomes a multi-line draft, same as Alt+Enter.
                    Event::Paste(text) => {
                        for c in text.chars() {
                            if c == '\r' {
                                continue;
                            }
                            state.input.insert(state.cursor, c);
                            state.cursor += 1;
                        }
                    }
                    _ => {}
                }
            }
            _ = tick.tick() => {
                // Just a redraw pulse for the spinner frame/elapsed time;
                // no state change needed here.
            }
            Some(ev) = rx.recv() => {
                apply(state, ev);
                if !state.waiting {
                    if let Some(line) = state.queued.take() {
                        state.input = line.chars().collect();
                        state.cursor = state.input.len();
                        submit(state, cfg, root, &tx);
                    }
                }
            }
        }

        if state.should_quit {
            break;
        }
        terminal.draw(|f| page_rows = draw(f, state))?;
    }
    Ok(())
}

/// Enter's actual handler: submit right away if nothing's in flight,
/// otherwise queue -- the request loop owns the checkpoint-backed
/// conversation for the duration of one turn, so a second one can't run
/// concurrently, but there's no reason typing or *sending* the next
/// question has to wait for that, any more than Claude Code makes you
/// wait to queue up a follow-up. `drive`'s `ChatEvent::Done`/`Error`
/// handling flushes the queue the instant `waiting` clears.
fn submit_or_queue(state: &mut ChatState, cfg: &Config, root: &std::path::Path, tx: &mpsc::UnboundedSender<ChatEvent>) {
    if !state.waiting {
        submit(state, cfg, root, tx);
        return;
    }
    let line: String = state.input.iter().collect::<String>().trim().to_string();
    state.input.clear();
    state.cursor = 0;
    if !line.is_empty() {
        state.queued = Some(line);
    }
}

fn submit(state: &mut ChatState, cfg: &Config, root: &std::path::Path, tx: &mpsc::UnboundedSender<ChatEvent>) {
    let line: String = state.input.iter().collect::<String>().trim().to_string();
    state.input.clear();
    state.cursor = 0;
    if line.is_empty() {
        return;
    }
    if line == "/exit" || line == "/quit" {
        state.should_quit = true;
        return;
    }
    if line == "/clear" {
        state.transcript.push(TLine::Sys("(conversation cleared)".into()));
        state.checkpoint_session = None;
        state.plan.clear();
        state.context_extra = 0;
        state.scroll = 0;
        return;
    }
    if line == "/help" {
        state.transcript.push(TLine::Sys(HELP_TEXT.to_string()));
        return;
    }
    if line == "/copy" {
        let last_reply = state.transcript.iter().rev().find_map(|l| match l {
            TLine::Agent { text, .. } if !text.trim().is_empty() => Some(text.clone()),
            _ => None,
        });
        match last_reply {
            Some(text) => {
                copy_to_clipboard(&text);
                state.transcript.push(TLine::Sys("(copied last reply to clipboard)".into()));
            }
            None => state.transcript.push(TLine::Err("nothing to copy yet".into())),
        }
        return;
    }
    if let Some(rest) = line.strip_prefix("/export") {
        let path = rest.trim();
        let path = if path.is_empty() {
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            format!("grv-chat-{secs}.md")
        } else {
            path.to_string()
        };
        let mut md = String::from("# GRAVITON chat transcript\n\n");
        for entry in &state.transcript {
            match entry {
                TLine::You(text) => md.push_str(&format!("**You:** {text}\n\n")),
                TLine::Agent { label, text } if !text.trim().is_empty() => md.push_str(&format!("**{label}:** {text}\n\n")),
                _ => {}
            }
        }
        match std::fs::write(&path, md) {
            Ok(()) => state.transcript.push(TLine::Sys(format!("exported to {path}"))),
            Err(e) => state.transcript.push(TLine::Err(format!("couldn't write {path}: {e}"))),
        }
        return;
    }
    if let Some(rest) = line.strip_prefix("/resume") {
        let id = rest.trim();
        if id.is_empty() {
            state.transcript.push(TLine::Err("usage: /resume <session-id> — see /checkpoints for the list".into()));
        } else {
            load_session_into(state, root, id);
            state.scroll = 0;
        }
        return;
    }
    if line == "/checkpoints" {
        match checkpoint::list_sessions(root) {
            Ok(sessions) if sessions.is_empty() => state.transcript.push(TLine::Sys("no checkpoint sessions yet".into())),
            Ok(sessions) => {
                let mut block = String::new();
                for s in sessions {
                    let marker = if state.checkpoint_session.as_deref() == Some(s.id.as_str()) { "* " } else { "  " };
                    block.push_str(&format!("{marker}{:<20} {} step(s), {} file(s) touched\n", s.id, s.steps, s.files_touched));
                }
                state.transcript.push(TLine::Sys(block.trim_end().to_string()));
            }
            Err(e) => state.transcript.push(TLine::Err(format!("{e:#}"))),
        }
        return;
    }
    if let Some(rest) = line.strip_prefix("/rollback") {
        let mut parts = rest.trim().split_whitespace();
        let session_id = parts.next().map(str::to_string).or_else(|| state.checkpoint_session.clone());
        let to = parts.next().and_then(|s| s.parse::<u64>().ok());
        match session_id {
            None => state.transcript.push(TLine::Err("no checkpoint session yet — ask the agent to do something first, or see /checkpoints for an id".into())),
            Some(id) => match checkpoint::rollback(root, &id, to) {
                Ok(n) => state.transcript.push(TLine::Sys(format!("rolled back {n} change(s) from session {id}"))),
                Err(e) => state.transcript.push(TLine::Err(format!("{e:#}"))),
            },
        }
        return;
    }
    if let Some(name) = line.strip_prefix("/mode") {
        let name = name.trim();
        let picked = match name {
            "" => Some(state.mode.next()),
            "manual" => Some(PermissionMode::Manual),
            "semi-auto" | "semi" => Some(PermissionMode::SemiAuto),
            "auto" => Some(PermissionMode::Auto),
            _ => None,
        };
        match picked {
            Some(m) => {
                state.mode = m;
                state.transcript.push(TLine::Sys(format!("(permission mode: {})", state.mode.label())));
            }
            None => state.transcript.push(TLine::Err(format!("unknown mode '{name}' — manual, semi-auto, or auto"))),
        }
        return;
    }
    if let Some(key) = line.strip_prefix("/agent ") {
        match resolve_agent(key.trim()) {
            Ok(a) => {
                state.agent = a;
                state.cfg_model = cfg.model_for(a.tier, a.key).to_string();
                state.checkpoint_session = None;
                state.plan.clear();
                state.context_extra = 0;
                state.transcript.push(TLine::Sys(format!("(switched to {} — conversation reset)", a.display)));
            }
            Err(e) => state.transcript.push(TLine::Err(format!("{e:#}"))),
        }
        state.scroll = 0;
        return;
    }

    if state.history.last().map(String::as_str) != Some(line.as_str()) {
        state.history.push(line.clone());
    }
    state.history_cursor = None;
    state.history_draft.clear();

    state.transcript.push(TLine::You(line.clone()));
    state.waiting = true;
    state.waiting_since = Instant::now();
    // A new question is always worth seeing, even if you'd scrolled up
    // to reread earlier history -- jump back to live.
    state.scroll = 0;

    let cfg = cfg.clone();
    let root = root.to_path_buf();
    let agent = state.agent;
    let mode = state.mode;
    let resume_session = state.checkpoint_session.clone();
    let tx2 = tx.clone();

    let handle = tokio::spawn(async move {
        let io: Arc<dyn RunIo> = Arc::new(TuiIo { tx: tx2.clone(), mode });
        let outcome: Result<()> = async {
            // Mirrors `cmd_run`: context is only built for a fresh
            // session -- a resumed one's restored transcript already
            // has the original question's context in it.
            let initial_context = if resume_session.is_none() {
                let conn = open_or_init_repo_db(&cfg, &root)?;
                let (groups, semantic_src) = build_context_sync(&cfg, &root, &conn, &line, &[])?;
                drop(conn);
                let ctx = finish_context(&cfg, &line, groups, semantic_src).await?;
                let _ = tx2.send(ChatEvent::ContextChars(ctx.chars().count()));
                Some(ctx)
            } else {
                None
            };
            // Browser tools (navigate/eval/screenshot/console) aren't
            // gated by permission mode -- same as `grv run --browser`,
            // they were never behind `gate()`'s write/edit/delete/shell
            // confirmation to begin with, so there's no extra risk tier
            // to wire up here; this just extends an existing capability
            // to the chat's own agentic loop.
            let loop_cfg = agentic::AgentLoopConfig { auto_approve: mode == PermissionMode::Auto, enable_browser: true };
            agentic::run(&cfg, &root, agent, &line, initial_context, loop_cfg, resume_session, io).await
        }
        .await;

        match outcome {
            Ok(()) => {
                let _ = tx2.send(ChatEvent::Done);
            }
            Err(e) => {
                let _ = tx2.send(ChatEvent::Error(format!("{e:#}")));
            }
        }
    });
    state.current_task = Some(handle.abort_handle());
}

/// Strips `ESC [ ... <letter>` CSI sequences -- the only ANSI
/// `agentic::run`'s `io.emit`/action-description text ever produces
/// (color/style codes for a raw terminal) -- so it renders as plain
/// text instead of literal escape bytes once it's a ratatui `Span`.
/// Writes `text` to the *system* clipboard via an OSC 52 escape
/// sequence rather than a clipboard crate or an `xclip`/`wl-copy`
/// subprocess call -- OSC 52 is supported directly by every terminal
/// emulator this project's README already assumes (kitty, alacritty,
/// wezterm, iTerm2, foot, Windows Terminal, xterm with
/// `allowWindowOps`), so this needs no new dependency and works over
/// SSH too, which a clipboard *tool* wouldn't. Mouse-drag copy still
/// works via Shift+select (see `run_chat_tui`'s doc comment) -- this is
/// for `/copy`, a one-keystroke way to grab the last reply without
/// having to select it at all.
fn copy_to_clipboard(text: &str) {
    use std::io::Write;
    print!("\x1b]52;c;{}\x07", base64_encode(text.as_bytes()));
    let _ = std::io::stdout().flush();
}

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[(b2 & 0x3f) as usize] as char } else { '=' });
    }
    out
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            while let Some(c2) = chars.next() {
                if c2.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `y`/`yes` allows, empty/`n`/`no` denies, anything else redirects the
/// agent with that text as a note -- the exact same three-way parse
/// `run_io::TerminalIo::confirm` has always used for `grv run`'s
/// terminal prompt, reused here so the chat TUI's confirm answers mean
/// the same thing.
fn parse_confirm_answer(line: &str) -> Decision {
    match line {
        "y" | "Y" | "yes" | "Yes" => Decision::Allow,
        "" | "n" | "N" | "no" | "No" => Decision::Deny,
        other => Decision::Redirect(other.to_string()),
    }
}

/// `"all"` (multi-select only) picks everything; otherwise a
/// comma-separated list of 1-based option numbers, out-of-range or
/// non-numeric entries silently dropped -- same parse
/// `run_io::TerminalIo::ask_choice` uses for `grv run`'s terminal prompt.
fn parse_choice_answer(line: &str, options: &[String], multi: bool) -> Vec<String> {
    if multi && line.eq_ignore_ascii_case("all") {
        return options.to_vec();
    }
    let mut picked = Vec::new();
    for tok in line.split(',') {
        if let Ok(n) = tok.trim().parse::<usize>() {
            if n >= 1 && n <= options.len() {
                picked.push(options[n - 1].clone());
            }
        }
        if !multi {
            break;
        }
    }
    picked
}

/// Drives `agentic::run` from the chat TUI: `emit`/tool-call/thinking/
/// token output all become `ChatEvent`s on the same channel the model's
/// answer tokens already use; `confirm`/`ask_choice` round-trip through
/// a fresh `oneshot` per call instead of blocking stdin the way
/// `TerminalIo` does, since nothing here can block the UI task.
struct TuiIo {
    tx: mpsc::UnboundedSender<ChatEvent>,
    mode: PermissionMode,
}

impl RunIo for TuiIo {
    fn emit(&self, line: String) {
        let stripped = strip_ansi(&line);
        if !stripped.trim().is_empty() {
            let _ = self.tx.send(ChatEvent::Emit(stripped));
        }
    }

    fn on_token(&self, tok: &str) {
        let _ = self.tx.send(ChatEvent::Token(tok.to_string()));
    }

    fn on_thinking(&self, tok: &str) {
        let _ = self.tx.send(ChatEvent::Thinking(tok.to_string()));
    }

    fn on_tool_call(&self, name: &str, args: &serde_json::Value) {
        let _ = self.tx.send(ChatEvent::ToolCall(name.to_string(), args.clone()));
    }

    fn on_tool_result(&self, _name: &str, result: &str) {
        let _ = self.tx.send(ChatEvent::ToolResult(strip_ansi(result)));
    }

    fn note_checkpoint_id(&self, id: &str) {
        let _ = self.tx.send(ChatEvent::CheckpointId(id.to_string()));
    }

    fn on_plan(&self, steps: &serde_json::Value) {
        let parsed = steps
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|s| PlanStep {
                        text: s.get("text").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
                        status: s.get("status").and_then(|v| v.as_str()).unwrap_or("pending").to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let _ = self.tx.send(ChatEvent::Plan(parsed));
    }

    fn confirm(&self, _auto_approve: bool, tool: &str, action: String) -> Pin<Box<dyn Future<Output = Decision> + Send + '_>> {
        let tool = tool.to_string();
        let tx = self.tx.clone();
        let mode = self.mode;
        Box::pin(async move {
            match mode {
                PermissionMode::Auto => Decision::Allow,
                PermissionMode::SemiAuto if matches!(tool.as_str(), "write_file" | "edit_file") => Decision::Allow,
                _ => {
                    let (reply, rx) = oneshot::channel();
                    let _ = tx.send(ChatEvent::Confirm(tool, strip_ansi(&action), reply));
                    rx.await.unwrap_or(Decision::Deny)
                }
            }
        })
    }

    fn ask_choice(&self, question: String, options: Vec<String>, multi_select: bool) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        let tx = self.tx.clone();
        Box::pin(async move {
            let (reply, rx) = oneshot::channel();
            let _ = tx.send(ChatEvent::Choice(question, options, multi_select, reply));
            rx.await.unwrap_or_default()
        })
    }
}

/// Pretty one-line call signature instead of a raw JSON blob: `read_file(path: "src/main.rs")`.
/// Falls back to compact JSON for anything that isn't a flat object of scalars.
fn format_tool_call(name: &str, args: &serde_json::Value) -> String {
    let Some(obj) = args.as_object() else {
        return format!("{name}({args})");
    };
    let parts: Vec<String> = obj
        .iter()
        .map(|(k, v)| {
            let v = match v {
                serde_json::Value::String(s) if s.chars().count() > 60 => {
                    format!("\"{}…\"", s.chars().take(60).collect::<String>())
                }
                serde_json::Value::String(s) => format!("\"{s}\""),
                other => other.to_string(),
            };
            format!("{k}: {v}")
        })
        .collect();
    format!("{name}({})", parts.join(", "))
}

fn apply(state: &mut ChatState, ev: ChatEvent) {
    match ev {
        ChatEvent::Token(tok) => {
            // Lazily starts the agent's reply line on its first real
            // token -- never on a tool call or an empty placeholder, so
            // a turn with no text output never leaves a stray empty
            // "LABEL›" row, and a multi-step tool loop (several
            // ToolCall/ToolResult pairs in a row) keeps appending into
            // the *same* `Tools` group below instead of splitting into
            // one group per step.
            match state.transcript.last_mut() {
                Some(TLine::Agent { text, .. }) => text.push_str(&tok),
                _ => state.transcript.push(TLine::Agent { label: state.agent.display.to_string(), text: tok }),
            }
        }
        ChatEvent::Thinking(tok) => match state.transcript.last_mut() {
            Some(TLine::Thinking(text)) => text.push_str(&tok),
            _ => state.transcript.push(TLine::Thinking(tok)),
        },
        ChatEvent::ToolCall(name, args) => {
            let call = format_tool_call(&name, &args);
            match state.transcript.last_mut() {
                Some(TLine::Tools(entries)) => entries.push(ToolEntry { name, call, result: None }),
                _ => state.transcript.push(TLine::Tools(vec![ToolEntry { name, call, result: None }])),
            }
        }
        ChatEvent::ToolResult(preview) => {
            if let Some(TLine::Tools(entries)) = state.transcript.last_mut() {
                if let Some(last) = entries.last_mut() {
                    last.result = Some(preview);
                }
            }
        }
        ChatEvent::Emit(line) => state.transcript.push(TLine::Sys(line)),
        ChatEvent::CheckpointId(id) => {
            if !state.known_sessions.iter().any(|s| s == &id) {
                state.known_sessions.push(id.clone());
            }
            state.checkpoint_session = Some(id);
        }
        ChatEvent::Plan(steps) => state.plan = steps,
        ChatEvent::ContextChars(n) => state.context_extra += n,
        ChatEvent::Confirm(_tool, action, reply) => {
            state.transcript.push(TLine::Confirm(action));
            state.pending_confirm = Some(reply);
        }
        ChatEvent::Choice(question, options, multi, reply) => {
            let mut block = question;
            for (i, opt) in options.iter().enumerate() {
                block.push_str(&format!("\n  [{}] {opt}", i + 1));
            }
            state.transcript.push(TLine::Choice(block));
            state.pending_choice = Some(PendingChoice { options, multi, reply });
        }
        ChatEvent::Done => {
            state.waiting = false;
            state.current_task = None;
        }
        ChatEvent::Error(e) => {
            state.transcript.push(TLine::Err(e));
            state.waiting = false;
            state.current_task = None;
        }
    }
}

/// `/agent` autocomplete candidates for the current input, or empty if
/// One row of the autocomplete popup -- an agent to switch to, or one of
/// the plain slash-commands, unified so the popup/Tab/Enter/↑↓ handling
/// in `drive` doesn't need to know which kind it's looking at.
struct Suggestion {
    label: String,
    description: String,
    /// What Tab fills the input with (Enter does the same, then submits).
    complete: String,
}

/// Every chat-REPL command besides `/agent`, which gets its own, richer
/// listing (every agent, with its tagline) below.
/// `/quit` still works in `submit` (undocumented alias, harmless to
/// keep for anyone's muscle memory from other tools) but isn't listed
/// here -- two entries that do the exact same thing was the duplicate
/// the autocomplete popup didn't need to show.
const PLAIN_COMMANDS: &[(&str, &str)] = &[
    ("/clear", "reset the conversation"),
    ("/mode", "cycle manual/semi-auto/auto"),
    ("/checkpoints", "list checkpoint sessions"),
    ("/rollback", "undo file changes from a session"),
    ("/copy", "copy the last reply to your system clipboard"),
    ("/export", "save the conversation as Markdown"),
    ("/help", "keybindings and commands"),
    ("/exit", "quit"),
];

/// Autocomplete candidates for the current input, or empty if it
/// doesn't look like a command at all. Three cases:
/// - `/agent` or `/agent <partial>`: every agent matching on key or
///   display name (bare `/agent` matches everyone).
/// - `/<partial>` with no space yet: the plain commands and `/agent`
///   itself, filtered by prefix -- still typing which command this is.
/// - anything else (including a `/agent ...`-prefixed line that isn't
///   followed by a space, e.g. "/agentic workflows...", which is a
///   message that merely starts with that text, not the command):
///   nothing, so an ordinary question never spuriously triggers a popup.
fn command_suggestions(input: &str, known_sessions: &[String]) -> Vec<Suggestion> {
    if let Some(rest) = input.strip_prefix("/resume") {
        if !rest.is_empty() && !rest.starts_with(' ') {
            return Vec::new();
        }
        let query = rest.trim_start().to_lowercase();
        return known_sessions
            .iter()
            .rev()
            .filter(|id| id.to_lowercase().contains(&query))
            .map(|id| Suggestion { label: id.clone(), description: "checkpoint session".to_string(), complete: format!("/resume {id}") })
            .collect();
    }
    if let Some(rest) = input.strip_prefix("/agent") {
        if !rest.is_empty() && !rest.starts_with(' ') {
            return Vec::new();
        }
        let query = rest.trim_start().to_lowercase();
        return ALL_AGENTS
            .iter()
            .filter(|a| a.key.to_lowercase().contains(&query) || a.display.to_lowercase().contains(&query))
            .map(|a| Suggestion { label: a.key.to_string(), description: a.tagline.to_string(), complete: format!("/agent {} ", a.key) })
            .collect();
    }
    if let Some(rest) = input.strip_prefix("/mode") {
        if !rest.is_empty() && !rest.starts_with(' ') {
            return Vec::new();
        }
        let query = rest.trim_start().to_lowercase();
        const MODES: &[(&str, &str)] = &[
            ("manual", "ask before every write/edit/delete/shell/commit"),
            ("semi-auto", "auto-accept writes/edits, still ask for shell/delete/commit"),
            ("auto", "approve everything automatically"),
        ];
        return MODES
            .iter()
            .filter(|(name, _)| name.starts_with(&query))
            .map(|(name, desc)| Suggestion { label: name.to_string(), description: desc.to_string(), complete: format!("/mode {name}") })
            .collect();
    }
    if input.starts_with('/') && !input.contains(' ') {
        let query = input.to_lowercase();
        let mut out: Vec<Suggestion> = Vec::new();
        if "/agent".starts_with(&query) {
            out.push(Suggestion { label: "/agent <key>".into(), description: "switch to a specialist agent".into(), complete: "/agent ".into() });
        }
        if "/resume".starts_with(&query) {
            out.push(Suggestion { label: "/resume <id>".into(), description: "switch to a different checkpoint session".into(), complete: "/resume ".into() });
        }
        for (cmd, desc) in PLAIN_COMMANDS {
            if cmd.starts_with(&query) {
                out.push(Suggestion { label: cmd.to_string(), description: desc.to_string(), complete: cmd.to_string() });
            }
        }
        return out;
    }
    Vec::new()
}

fn draw(f: &mut ratatui::Frame, state: &ChatState) -> usize {
    let area = f.area();
    let inner_width = area.width.saturating_sub(2).max(5) as usize;
    // The 2 extra columns are the "› "/indent every input row carries
    // (see `draw_input`) -- same accounting `wrapped_rows` does for the
    // transcript's own prefixes.
    let wrap_width = inner_width.saturating_sub(2).max(5);
    let (input_rows, cursor_row, cursor_col) = wrap_input_for_display(&state.input, state.cursor, wrap_width);
    // A gated action or ask_user call is answered through this same box
    // (y/n/a redirect note; a number or "all") -- show the ordinary
    // editable input then, even though `waiting` is still technically
    // true (the turn is paused on this answer, not finished).
    let show_spinner = state.waiting && state.pending_confirm.is_none() && state.pending_choice.is_none();
    // While the spinner's showing, row 1 is always its status; a draft
    // typed for the *next* question (queued on Enter, not sent) gets its
    // own rows below it instead of being hidden until the current reply
    // finishes -- otherwise unlocking typing during `waiting` would let
    // you type into a box that visibly shows nothing back.
    let visible_input_rows = if show_spinner {
        1 + if state.input.is_empty() { 0 } else { input_rows.len().min(5) }
    } else {
        input_rows.len().min(6)
    };
    let input_height = (visible_input_rows + 2) as u16;
    // Hidden (zero height) with no plan yet -- a standing checklist
    // panel only earns its space once there's something to check off.
    let plan_height = if state.plan.is_empty() { 0 } else { (state.plan.len().min(8) + 2) as u16 };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(plan_height), Constraint::Min(3), Constraint::Length(input_height)])
        .split(area);

    draw_header(f, chunks[0], state);
    draw_plan(f, chunks[1], state);
    let rows = draw_transcript(f, chunks[2], state);
    draw_input(f, chunks[3], state, &input_rows, cursor_row, cursor_col, show_spinner);

    // Not gated on `waiting` -- typing (including `/agent ...`) and the
    // popup it drives work the same whether a previous question is
    // still generating or not (see `drive`'s key handling and
    // `ChatState::queued`); gating the *rendering* here while that
    // logic stayed live was exactly the mismatch that made it look
    // broken -- the popup is invisible, but it's using the keys. It's
    // suppressed while answering a confirm/choice, same as in `drive`.
    if state.pending_confirm.is_none() && state.pending_choice.is_none() {
        let input_text: String = state.input.iter().collect();
        let suggestions = command_suggestions(&input_text, &state.known_sessions);
        if !suggestions.is_empty() {
            draw_command_popup(f, chunks[3], state, &suggestions);
        }
    }
    rows
}

/// A rough (chars/4-as-tokens) estimate of how much of `num_ctx` the
/// current session is using -- real tokenization varies by model and
/// language, so this is deliberately presented as "~N%", not an exact
/// figure. Sums what's visibly in the transcript (questions, replies,
/// and tool call/result text, which does count against the model's
/// context same as anything else) plus `context_extra`, the one-time
/// retrieved-context dump that never appears in the transcript itself.
fn estimate_context_chars(state: &ChatState) -> usize {
    let visible: usize = state
        .transcript
        .iter()
        .map(|l| match l {
            TLine::You(t) => t.chars().count(),
            TLine::Agent { text, .. } => text.chars().count(),
            TLine::Tools(entries) => entries.iter().map(|e| e.call.chars().count() + e.result.as_deref().map(str::len).unwrap_or(0)).sum(),
            _ => 0,
        })
        .sum();
    visible + state.context_extra
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    let mode_color = match state.mode {
        PermissionMode::Manual => Color::DarkGray,
        PermissionMode::SemiAuto => Color::Yellow,
        PermissionMode::Auto => Color::Red,
    };
    let pct = ((estimate_context_chars(state) / 4) * 100 / state.num_ctx.max(1)).min(999);
    let ctx_color = if pct >= 85 { Color::Red } else if pct >= 60 { Color::Yellow } else { Color::DarkGray };
    let line = Line::from(vec![
        Span::styled("GRAVITON", Style::default().fg(state.agent_color).add_modifier(Modifier::BOLD)),
        Span::styled(" · ", Style::default().fg(Color::DarkGray)),
        Span::styled(state.agent.display, Style::default().fg(state.agent_color).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" [{}] ", state.cfg_model), Style::default().fg(Color::DarkGray)),
        Span::styled(format!("[{}] ", state.mode.label()), Style::default().fg(mode_color).add_modifier(Modifier::BOLD)),
        Span::styled(format!("[ctx ~{pct}%]"), Style::default().fg(ctx_color)),
        Span::styled("   /help for keys & commands", Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// The standing checklist panel for `update_plan`'s current steps --
/// `[x]` done (green), `[~]` in progress (yellow, bold), `[ ]` pending
/// (dim) -- a no-op when `state.plan` is empty (`draw`'s `plan_height`
/// is 0 then, so this has no area to draw into anyway).
fn draw_plan(f: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    if state.plan.is_empty() {
        return;
    }
    let lines: Vec<Line> = state
        .plan
        .iter()
        .map(|step| {
            let (mark, style) = match step.status.as_str() {
                "done" => ("[x]", Style::default().fg(Color::Green)),
                "in_progress" => ("[~]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                _ => ("[ ]", Style::default().fg(Color::DarkGray)),
            };
            Line::from(vec![Span::styled(format!("{mark} "), style), Span::styled(step.text.clone(), style)])
        })
        .collect();
    let block = Block::default().borders(Borders::ALL).border_type(ratatui::widgets::BorderType::Rounded).border_style(Style::default().fg(Color::DarkGray)).title(" plan ");
    f.render_widget(Paragraph::new(lines).block(block), area);
}

/// A floating autocomplete popup (agents, or the plain commands),
/// drawn after -- so visually on top of -- everything else, anchored
/// just above the input box.
fn draw_command_popup(f: &mut ratatui::Frame, input_area: Rect, state: &ChatState, suggestions: &[Suggestion]) {
    let total = suggestions.len();
    let idx = state.suggestion_idx.min(total.saturating_sub(1));
    // Up to 12 rows, but never more than fit above the input box (the
    // header row at y=0 is the hard ceiling) -- and the window follows
    // `idx` the same way the transcript's own scroll follows the live
    // bottom, so selecting past row 6 (all 22 agents, not just the
    // first page of them) is reachable instead of invisible.
    let max_rows_above = input_area.y.saturating_sub(1).max(1) as usize;
    let cap = total.min(12).min(max_rows_above);
    let start = if total > cap { idx.saturating_sub(cap.saturating_sub(1)).min(total - cap) } else { 0 };
    let visible = &suggestions[start..(start + cap).min(total)];

    let height = (cap + 2) as u16;
    let area = Rect { x: input_area.x, y: input_area.y.saturating_sub(height), width: input_area.width, height };
    let inner_width = area.width.saturating_sub(4) as usize;

    let lines: Vec<Line> = visible
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let style = if start + i == idx { Style::default().fg(Color::Black).bg(state.agent_color) } else { Style::default() };
            let mut label = format!(" {:<14} {}", s.label, s.description);
            if label.chars().count() > inner_width {
                label = label.chars().take(inner_width.max(4)).collect::<String>() + "…";
            }
            Line::from(Span::styled(label, style))
        })
        .collect();

    let title = if total > cap { format!(" ↑↓ pick · Tab/Enter choose ({}/{total}) ", idx + 1) } else { " ↑↓ pick · Tab/Enter choose ".to_string() };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(Style::default().fg(state.agent_color))
        .title(title);
    f.render_widget(Paragraph::new(lines).block(block), area);
}

/// Builds every visual row (label + wrapped continuation rows) up front
/// as plain `(prefix, body, style)` rows, *then* slices the tail that
/// fits -- never ratatui's own `.wrap()` + `.scroll()` combination.
/// `.scroll()` counts the `Line`s it was given, which is the pre-wrap
/// logical line count; `.wrap()` then expands long lines into more
/// *rendered* rows than that at paint time. The two numbers disagree
/// as soon as one reply line is longer than the terminal is wide, so
/// the view undershoots and part (sometimes all) of what the model
/// just generated renders below the visible area -- invisible, not
/// cut off visibly, which is why it looked like nothing was happening
/// on a second question. Wrapping with `textwrap` ourselves first
/// means the row count we scroll by *is* the row count on screen.
fn push_wrapped(out: &mut Vec<Line<'static>>, width: usize, prefix: Option<(String, Style)>, body: &str, body_style: Style) {
    let indent_width = prefix.as_ref().map(|(p, _)| p.chars().count()).unwrap_or(0);
    let avail = width.saturating_sub(indent_width).max(10);
    let wrapped = textwrap::wrap(body, avail);
    let wrapped: Vec<&str> = if wrapped.is_empty() { vec![""] } else { wrapped.iter().map(|c| c.as_ref()).collect() };
    for (i, row) in wrapped.into_iter().enumerate() {
        if i == 0 {
            if let Some((p, style)) = &prefix {
                out.push(Line::from(vec![Span::styled(p.clone(), *style), Span::styled(row.to_string(), body_style)]));
                continue;
            }
        }
        if i > 0 {
            out.push(Line::from(vec![Span::raw(" ".repeat(indent_width)), Span::styled(row.to_string(), body_style)]));
        } else {
            out.push(Line::from(Span::styled(row.to_string(), body_style)));
        }
    }
}

/// Renders text that must keep its own line breaks -- a diff preview,
/// numbered options -- as one `push_wrapped` call per original line
/// instead of one call over the whole string. `push_wrapped` word-wraps
/// through `textwrap`, which treats a literal `\n` as just more
/// whitespace to reflow across, so wrapping the text as a single blob
/// would merge a diff's `+`/`-` lines into paragraphs and make it
/// unreadable; wrapping only *within* each line (still, if a line is
/// wider than the panel) keeps the structure intact.
fn push_multiline(out: &mut Vec<Line<'static>>, width: usize, style: Style, text: &str) {
    for line in text.split('\n') {
        push_wrapped(out, width, None, line, style);
    }
}

/// Same idea as `push_multiline`, but colors each line by `diff_preview`'s
/// own leading sign (`agentic::diff_preview`: `+`/`-`/` ` for insert/
/// delete/unchanged context) instead of one flat color -- the point of
/// showing a diff at all, at exactly the moment a write/edit is being
/// approved or denied, is to see what changed at a glance. A line that
/// doesn't start with one of those (a non-diff action like `run_shell`
/// or `delete_file`, which this same `TLine::Confirm` renders too) just
/// falls back to `default_style`.
fn push_diff_aware(out: &mut Vec<Line<'static>>, width: usize, default_style: Style, text: &str) {
    for line in text.split('\n') {
        let style = if line.starts_with('+') {
            Style::default().fg(Color::Green)
        } else if line.starts_with('-') {
            Style::default().fg(Color::Red)
        } else {
            default_style
        };
        push_wrapped(out, width, None, line, style);
    }
}

/// A paragraph-or-code-fence chunk of a markdown-lite message. Only
/// fenced ` ``` ` blocks are block-level; everything else is prose that
/// still gets inline `**bold**`/`` `code` `` parsed within it.
enum MdBlock<'a> {
    Prose(String),
    Code(Vec<&'a str>),
}

/// Split on ` ``` ` fences. An unterminated trailing fence is still
/// treated as code rather than dropped or shown with its marker — a
/// token budget cutting a reply off mid-block is the likely cause, and
/// losing the formatting is better than losing the content.
fn split_code_fences(text: &str) -> Vec<MdBlock<'_>> {
    let mut blocks = Vec::new();
    let mut in_code = false;
    let mut prose = String::new();
    let mut code: Vec<&str> = Vec::new();
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            if in_code {
                blocks.push(MdBlock::Code(std::mem::take(&mut code)));
            } else if !prose.is_empty() {
                blocks.push(MdBlock::Prose(std::mem::take(&mut prose)));
            }
            in_code = !in_code;
            continue;
        }
        if in_code {
            code.push(line);
        } else {
            if !prose.is_empty() {
                prose.push('\n');
            }
            prose.push_str(line);
        }
    }
    if in_code && !code.is_empty() {
        blocks.push(MdBlock::Code(code));
    } else if !prose.is_empty() {
        blocks.push(MdBlock::Prose(prose));
    }
    blocks
}

/// Pull `**bold**` and `` `code` `` out of one line of prose into styled
/// runs. Deliberately not a real markdown parser (no nesting, no links,
/// no lists) -- just the two inline forms a model's running text
/// actually uses, parsed by hand instead of pulling in a markdown crate
/// for it.
/// Recognizes a Markdown list item's leading marker -- `- `/`* ` (bullet)
/// or `N. `/`N) ` (numbered) -- and splits it into (leading whitespace,
/// the marker to display, the rest of the line). `None` for anything
/// else, including a bare `-`/digit with no following space (so "a - b"
/// or a version number mid-sentence never gets mistaken for a list).
fn parse_list_marker(line: &str) -> Option<(&str, String, &str)> {
    let leading_len = line.len() - line.trim_start().len();
    let leading = &line[..leading_len];
    let trimmed = &line[leading_len..];
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        return Some((leading, "•".to_string(), rest));
    }
    let digits_len = trimmed.find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
    if digits_len > 0 {
        let after_digits = &trimmed[digits_len..];
        if let Some(rest) = after_digits.strip_prefix(". ").or_else(|| after_digits.strip_prefix(") ")) {
            return Some((leading, format!("{}.", &trimmed[..digits_len]), rest));
        }
    }
    None
}

fn parse_inline(line: &str, base: Style, code_style: Style, bold_style: Style) -> Vec<(String, Style)> {
    let mut spans = Vec::new();
    let mut chars = line.chars().peekable();
    let mut buf = String::new();
    while let Some(c) = chars.next() {
        if c == '`' {
            if !buf.is_empty() {
                spans.push((std::mem::take(&mut buf), base));
            }
            let mut inner = String::new();
            while let Some(&c2) = chars.peek() {
                chars.next();
                if c2 == '`' {
                    break;
                }
                inner.push(c2);
            }
            if !inner.is_empty() {
                spans.push((inner, code_style));
            }
        } else if c == '*' && chars.peek() == Some(&'*') {
            chars.next();
            if !buf.is_empty() {
                spans.push((std::mem::take(&mut buf), base));
            }
            let mut inner = String::new();
            loop {
                match chars.next() {
                    Some('*') if chars.peek() == Some(&'*') => {
                        chars.next();
                        break;
                    }
                    Some(c2) => inner.push(c2),
                    None => break,
                }
            }
            if !inner.is_empty() {
                spans.push((inner, bold_style));
            }
        } else {
            buf.push(c);
        }
    }
    if !buf.is_empty() {
        spans.push((buf, base));
    }
    spans
}

/// Greedy word-wrap over *styled* runs -- `textwrap` only wraps plain
/// strings, so once a run of text needs per-word styling (bold/code)
/// this does the wrapping by hand instead. A single word wider than
/// `avail` on its own (a long identifier, a URL) gets its own row and
/// is left to overflow/clip rather than hard-broken mid-word.
fn wrap_styled(spans: Vec<(String, Style)>, avail: usize) -> Vec<Vec<(String, Style)>> {
    let mut rows: Vec<Vec<(String, Style)>> = Vec::new();
    let mut current: Vec<(String, Style)> = Vec::new();
    let mut current_len = 0usize;
    for (text, style) in spans {
        for word in text.split(' ') {
            if word.is_empty() {
                continue;
            }
            let wlen = word.chars().count();
            let sep = if current.is_empty() { 0 } else { 1 };
            if current_len + sep + wlen > avail && !current.is_empty() {
                rows.push(std::mem::take(&mut current));
                current_len = 0;
            }
            if !current.is_empty() {
                current.push((" ".to_string(), Style::default()));
                current_len += 1;
            }
            current.push((word.to_string(), style));
            current_len += wlen;
        }
    }
    if !current.is_empty() {
        rows.push(current);
    }
    if rows.is_empty() {
        rows.push(Vec::new());
    }
    rows
}

/// Renders an agent reply with fenced code blocks pulled out into their
/// own un-wrapped, left-barred rows, and `**bold**`/`` `code` `` parsed
/// within the remaining prose -- the markdown-lite rendering a model's
/// replies actually need, since raw `**`/backtick characters otherwise
/// show up literally in the transcript.
fn push_markdown(out: &mut Vec<Line<'static>>, width: usize, prefix: (String, Style), text: &str) {
    let indent_width = prefix.0.chars().count();
    let avail = width.saturating_sub(indent_width).max(10);
    let base = Style::default();
    let code_style = Style::default().fg(Color::Yellow);
    let bold_style = Style::default().add_modifier(Modifier::BOLD);
    let mut first_row = true;

    let mut emit = |out: &mut Vec<Line<'static>>, row_spans: Vec<Span<'static>>| {
        if first_row {
            let mut spans = vec![Span::styled(prefix.0.clone(), prefix.1)];
            spans.extend(row_spans);
            out.push(Line::from(spans));
            first_row = false;
        } else {
            let mut spans = vec![Span::raw(" ".repeat(indent_width))];
            spans.extend(row_spans);
            out.push(Line::from(spans));
        }
    };

    for block in split_code_fences(text) {
        match block {
            MdBlock::Prose(paragraph) => {
                for line in paragraph.split('\n') {
                    if line.trim().is_empty() {
                        emit(out, vec![]);
                        continue;
                    }
                    if let Some((leading, marker, rest)) = parse_list_marker(line) {
                        // Hanging indent so a wrapped item's continuation
                        // lines align under its *text*, not its marker --
                        // handled entirely here (not through `emit`'s own
                        // first-row-only indent) since it's per list item,
                        // not per message.
                        let hang = leading.chars().count() + marker.chars().count() + 1;
                        let item_avail = avail.saturating_sub(hang).max(10);
                        let spans = parse_inline(rest, base, code_style, bold_style);
                        let mut rows = wrap_styled(spans, item_avail);
                        if rows.is_empty() {
                            rows.push(Vec::new());
                        }
                        for (i, row) in rows.into_iter().enumerate() {
                            let mut row_spans: Vec<Span<'static>> = if i == 0 {
                                vec![Span::raw(leading.to_string()), Span::styled(format!("{marker} "), Style::default().fg(Color::Magenta))]
                            } else {
                                vec![Span::raw(" ".repeat(hang))]
                            };
                            row_spans.extend(row.into_iter().map(|(t, s)| Span::styled(t, s)));
                            emit(out, row_spans);
                        }
                        continue;
                    }
                    let spans = parse_inline(line, base, code_style, bold_style);
                    for row in wrap_styled(spans, avail) {
                        let row_spans = row.into_iter().map(|(t, s)| Span::styled(t, s)).collect();
                        emit(out, row_spans);
                    }
                }
            }
            MdBlock::Code(lines) => {
                for code_line in lines {
                    emit(
                        out,
                        vec![
                            Span::styled("▏", Style::default().fg(Color::DarkGray)),
                            Span::styled(code_line.to_string(), Style::default().fg(Color::Cyan)),
                        ],
                    );
                }
            }
        }
    }
}

fn wrapped_rows(state: &ChatState, width: usize) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();

    for t in &state.transcript {
        match t {
            TLine::You(s) => {
                push_wrapped(&mut out, width, Some(("you   ".into(), Style::default().fg(state.user_color).add_modifier(Modifier::BOLD))), s, Style::default());
            }
            TLine::Agent { label, text } => {
                if text.is_empty() {
                    continue;
                }
                let prefix = format!("{label:<6}");
                push_markdown(&mut out, width, (prefix, Style::default().fg(state.agent_color).add_modifier(Modifier::BOLD)), text);
                out.push(Line::from(""));
            }
            TLine::Thinking(text) => {
                if text.is_empty() {
                    continue;
                }
                let dim = Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC);
                push_wrapped(&mut out, width, Some(("~     ".into(), dim)), text, dim);
                out.push(Line::from(""));
            }
            TLine::Tools(entries) => {
                if entries.len() > 1 && !state.tools_expanded {
                    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
                    let summary = format!("▸ {} tools called ({}) — Tab to expand", entries.len(), names.join(", "));
                    push_wrapped(&mut out, width, None, &summary, Style::default().fg(Color::DarkGray));
                } else {
                    for e in entries {
                        push_wrapped(&mut out, width, Some(("  → ".into(), Style::default().fg(Color::DarkGray))), &e.call, Style::default().fg(Color::DarkGray));
                        let result_text = e.result.as_deref().unwrap_or("…");
                        let result_style = if e.result.is_some() { Style::default().fg(Color::Green) } else { Style::default().fg(Color::DarkGray) };
                        push_wrapped(&mut out, width, Some(("  ← ".into(), result_style)), result_text, Style::default().fg(Color::DarkGray));
                    }
                    if entries.len() > 1 {
                        push_wrapped(&mut out, width, None, "(Tab to collapse)", Style::default().fg(Color::DarkGray));
                    }
                }
                out.push(Line::from(""));
            }
            TLine::Sys(s) => {
                push_wrapped(&mut out, width, None, s, Style::default().fg(Color::DarkGray));
                out.push(Line::from(""));
            }
            TLine::Confirm(action) => {
                push_wrapped(
                    &mut out,
                    width,
                    None,
                    "⚠ confirm — y = allow · n = deny · type a note + Enter = redirect",
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                );
                push_diff_aware(&mut out, width, Style::default().fg(Color::Yellow), action);
                out.push(Line::from(""));
            }
            TLine::Choice(block) => {
                push_wrapped(&mut out, width, None, "? choose — number(s), or \"all\" + Enter", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
                push_multiline(&mut out, width, Style::default().fg(Color::Cyan), block);
                out.push(Line::from(""));
            }
            TLine::Err(s) => {
                push_wrapped(&mut out, width, Some(("error ".into(), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))), s, Style::default().fg(Color::Red));
                out.push(Line::from(""));
            }
        }
    }
    out
}

/// Where the visible window starts, and how much `scroll` actually got
/// clamped to (there may be fewer rows above than the user scrolled
/// back) -- pure function so the scrolling math has a test without
/// standing up a `ratatui::Frame`.
fn scroll_window(total_rows: usize, height: usize, scroll: usize) -> (usize, usize) {
    let max_scroll = total_rows.saturating_sub(height);
    let clamped = scroll.min(max_scroll);
    (max_scroll.saturating_sub(clamped), clamped)
}

fn draw_transcript(f: &mut ratatui::Frame, area: Rect, state: &ChatState) -> usize {
    let inner_width = area.width.saturating_sub(2) as usize;
    let height = area.height.saturating_sub(2) as usize;
    let rows = wrapped_rows(state, inner_width);
    let total = rows.len();
    let (start, scroll) = scroll_window(total, height, state.scroll);
    let visible: Vec<Line> = rows.into_iter().skip(start).take(height).collect();

    let title = if scroll > 0 {
        format!(" conversation — scrolled back, PageDown to catch up ({scroll} row{}) ", if scroll == 1 { "" } else { "s" })
    } else {
        String::new()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(Style::default().fg(if scroll > 0 { Color::Yellow } else { Color::DarkGray }))
        .title(title);
    f.render_widget(Paragraph::new(visible).block(block), area);
    height
}

/// Word-wraps the input for display and reports exactly which wrapped
/// row/column the cursor lands in. Rather than replaying `textwrap`'s
/// internal break/trim decisions by hand to derive that position (the
/// same class of mismatch `wrapped_rows` had with ratatui's own
/// `.scroll()`), a sentinel character is inserted at the cursor index,
/// the whole thing is wrapped once, and whichever output row the
/// sentinel survived into *is* the answer -- `textwrap` only trims
/// whitespace at a break point, it never drops a real character, so
/// the sentinel always reappears in exactly one row except when the
/// cursor sits in trailing whitespace at the very end (the `!found`
/// fallback below: the last row, one past its last character).
fn wrap_input_for_display(input: &[char], cursor: usize, width: usize) -> (Vec<String>, usize, usize) {
    const MARK: char = '\u{0}';
    let width = width.max(1);
    let mut marked: Vec<char> = input.to_vec();
    marked.insert(cursor.min(marked.len()), MARK);
    let text: String = marked.into_iter().collect();

    let mut rows: Vec<String> = Vec::new();
    let mut cursor_row = 0;
    let mut cursor_col = 0;
    let mut found = false;
    for line in text.split('\n') {
        let wrapped = textwrap::wrap(line, width);
        let wrapped: Vec<String> = if wrapped.is_empty() { vec![String::new()] } else { wrapped.iter().map(|c| c.to_string()).collect() };
        for seg in wrapped {
            if !found {
                if let Some(byte_pos) = seg.find(MARK) {
                    cursor_row = rows.len();
                    cursor_col = seg[..byte_pos].chars().count();
                    found = true;
                    rows.push(seg.replace(MARK, ""));
                    continue;
                }
            }
            rows.push(seg);
        }
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    if !found {
        cursor_row = rows.len() - 1;
        cursor_col = rows[cursor_row].chars().count();
    }
    (rows, cursor_row, cursor_col)
}

fn draw_input(f: &mut ratatui::Frame, area: Rect, state: &ChatState, rows: &[String], cursor_row: usize, cursor_col: usize, show_spinner: bool) {
    let border_color = if show_spinner || state.pending_confirm.is_some() || state.pending_choice.is_some() { Color::Yellow } else { Color::DarkGray };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let lines: Vec<Line> = if show_spinner {
        let frame = SPINNER_FRAMES[(state.waiting_since.elapsed().as_millis() / 120) as usize % SPINNER_FRAMES.len()];
        let mut status = vec![
            Span::styled(format!("{frame} "), Style::default().fg(Color::Yellow)),
            Span::styled(format!("{} is thinking… {}s", state.agent.display, state.waiting_since.elapsed().as_secs()), Style::default().fg(Color::DarkGray)),
        ];
        if let Some(q) = &state.queued {
            let preview: String = q.chars().take(40).collect();
            let ellipsis = if q.chars().count() > 40 { "…" } else { "" };
            status.push(Span::styled(format!("  · next: {preview}{ellipsis}"), Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)));
        }
        let mut out = vec![Line::from(status)];
        // A draft for the question *after* this one, typed while still
        // waiting -- shown so typing during `waiting` isn't typing into
        // a box that visibly shows nothing back (see `draw`'s sizing).
        if !state.input.is_empty() {
            for row in rows {
                out.push(Line::from(vec![Span::styled("  ", Style::default()), Span::styled(row.clone(), Style::default().fg(Color::DarkGray))]));
            }
        }
        out
    } else {
        // Cap the visible window and keep the cursor's row inside it --
        // same idea as `scroll_window` for the transcript, just always
        // "following" the cursor instead of a user-controlled offset.
        const CAP: usize = 6;
        let total = rows.len();
        let start = if total > CAP { cursor_row.saturating_sub(CAP - 1).min(total - CAP) } else { 0 };
        let visible = &rows[start..(start + CAP).min(total)];
        let visible_cursor_row = cursor_row.saturating_sub(start);

        visible
            .iter()
            .enumerate()
            .map(|(i, row)| {
                // A real cursor position (Left/Right/Home/End-addressable),
                // drawn as a reversed-video cell rather than relying on the
                // terminal's own cursor -- we never move that (no
                // `f.set_cursor_position`), so this is the only visible
                // insertion point there is.
                let marker = if i != 0 {
                    Span::raw("  ")
                } else if let Some(search) = &state.history_search {
                    let note = if search.match_idx.is_none() && !search.query.is_empty() { " (no match)" } else { "" };
                    Span::styled(format!("(search: {}){note} ", search.query), Style::default().fg(Color::Yellow))
                } else {
                    Span::styled("› ", Style::default().fg(state.user_color))
                };
                if i == visible_cursor_row {
                    let chars: Vec<char> = row.chars().collect();
                    let before: String = chars[..cursor_col.min(chars.len())].iter().collect();
                    let at = chars.get(cursor_col).map(|c| c.to_string()).unwrap_or_else(|| " ".to_string());
                    let after: String = if cursor_col < chars.len() { chars[cursor_col + 1..].iter().collect() } else { String::new() };
                    Line::from(vec![marker, Span::raw(before), Span::styled(at, Style::default().add_modifier(Modifier::REVERSED)), Span::raw(after)])
                } else {
                    Line::from(vec![marker, Span::raw(row.clone())])
                }
            })
            .collect()
    };
    f.render_widget(Paragraph::new(lines).block(block), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_window_pins_to_bottom_when_content_fits() {
        // Fewer rows than the viewport -- nothing to scroll, show from 0.
        assert_eq!(scroll_window(5, 10, 0), (0, 0));
        assert_eq!(scroll_window(5, 10, 3), (0, 0));
    }

    #[test]
    fn scroll_window_follows_the_tail_at_zero_scroll() {
        // 50 rows, a 10-row viewport, not scrolled back: show the last 10.
        assert_eq!(scroll_window(50, 10, 0), (40, 0));
    }

    #[test]
    fn scroll_window_moves_the_window_up_as_scroll_increases() {
        assert_eq!(scroll_window(50, 10, 5), (35, 5));
    }

    #[test]
    fn scroll_window_clamps_past_the_top() {
        // Can't scroll back further than there is history -- clamp at the
        // very first row instead of (silently) going negative.
        assert_eq!(scroll_window(50, 10, 1000), (0, 40));
    }

    #[test]
    fn long_reply_wraps_into_multiple_rows_that_all_fit_the_width() {
        let width = 40;
        let mut out = Vec::new();
        let text = "a ".repeat(100); // far wider than one row at width 40
        push_wrapped(&mut out, width, Some(("AGENT ".into(), Style::default())), &text, Style::default());
        assert!(out.len() > 1, "expected the long reply to wrap into several rows");
        for line in &out {
            let rendered_width: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
            assert!(rendered_width <= width, "row {rendered_width} chars wide exceeds panel width {width}");
        }
    }

    #[test]
    fn empty_agent_placeholder_produces_no_row() {
        // The whole point of skipping empty Agent entries: no stray
        // "LABEL›" row with nothing after it while waiting for tokens.
        let mut out = Vec::new();
        push_wrapped(&mut out, 80, None, "", Style::default());
        // An explicitly empty body still yields exactly one (empty) row
        // from push_wrapped itself -- the *skip* happens one level up,
        // in wrapped_rows's `if text.is_empty() { continue; }" before
        // push_wrapped is ever called for an Agent entry.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].spans.iter().map(|s| s.content.as_ref()).collect::<String>(), "");
    }

    fn line_text(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn split_code_fences_separates_prose_from_code() {
        let blocks = split_code_fences("before\n```\nlet x = 1;\n```\nafter");
        assert_eq!(blocks.len(), 3);
        assert!(matches!(&blocks[0], MdBlock::Prose(p) if p == "before"));
        assert!(matches!(&blocks[1], MdBlock::Code(c) if c == &["let x = 1;"]));
        assert!(matches!(&blocks[2], MdBlock::Prose(p) if p == "after"));
    }

    #[test]
    fn split_code_fences_keeps_an_unterminated_fence_as_code() {
        // A reply cut off mid-block (token budget) shouldn't lose its
        // content just because the closing ``` never arrived.
        let blocks = split_code_fences("```\nfn main() {}");
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], MdBlock::Code(c) if c == &["fn main() {}"]));
    }

    #[test]
    fn parse_inline_extracts_bold_and_code_runs() {
        let base = Style::default();
        let code = Style::default().fg(Color::Yellow);
        let bold = Style::default().add_modifier(Modifier::BOLD);
        let spans = parse_inline("use **bold** and `code`", base, code, bold);
        let texts: Vec<&str> = spans.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(texts, vec!["use ", "bold", " and ", "code"]);
        assert_eq!(spans[1].1, bold);
        assert_eq!(spans[3].1, code);
    }

    #[test]
    fn wrap_styled_never_exceeds_the_available_width() {
        let base = Style::default();
        let spans = vec![("word ".repeat(30), base)];
        let rows = wrap_styled(spans, 20);
        for row in &rows {
            let width: usize = row.iter().map(|(t, _)| t.chars().count()).sum();
            assert!(width <= 20, "row {width} chars exceeds avail 20");
        }
    }

    #[test]
    fn markdown_reply_renders_within_width_and_marks_code_lines() {
        let width = 30;
        let mut out = Vec::new();
        let text = "Explanation with **bold** and `inline`.\n```\nfn long_function_name_here() {}\n```\nmore prose after";
        push_markdown(&mut out, width, ("AGENT ".into(), Style::default()), text);
        for line in &out {
            let rendered: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
            // Code lines are allowed to overflow (never reflowed), prose
            // rows must respect the panel width.
            let is_code_row = line.spans.iter().any(|s| s.content == "▏");
            if !is_code_row {
                assert!(rendered <= width, "prose row {rendered} chars wide exceeds width {width}: {:?}", line_text(line));
            }
        }
        let joined: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(joined.contains("fn long_function_name_here"), "code block content missing: {joined}");
        assert!(joined.contains("more prose after"));
    }

    #[test]
    fn theme_color_parses_valid_hex_and_falls_back_otherwise() {
        assert_eq!(theme_color(Some("22d3ee"), Color::Cyan), Color::Rgb(0x22, 0xd3, 0xee));
        assert_eq!(theme_color(None, Color::Cyan), Color::Cyan);
        assert_eq!(theme_color(Some("not-a-color"), Color::Cyan), Color::Cyan);
        assert_eq!(theme_color(Some("abcd"), Color::Cyan), Color::Cyan); // wrong length
    }

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn wrap_input_cursor_tracks_through_a_wrapped_word() {
        let input = chars("hello world from graviton");
        // Cursor at index 11 sits right after "world" (0-indexed: "hello "
        // is 6 chars, "world" the next 5, so index 11 is the space right
        // after it, before "from"). At width 10 that lands on the row
        // "world from" -- the cursor must land right after "world" on
        // *that* row (col 5), not get lost to wherever textwrap broke
        // the line, and inserting the zero-width sentinel must not have
        // perturbed the wrap decision itself (same rows either way).
        let (rows, row, col) = wrap_input_for_display(&input, 11, 10);
        assert_eq!(rows, vec!["hello".to_string(), "world from".to_string(), "graviton".to_string()]);
        assert_eq!(row, 1);
        assert_eq!(col, 5);
    }

    #[test]
    fn wrap_input_cursor_at_very_end_lands_on_the_last_row() {
        let input = chars("short");
        let (rows, row, col) = wrap_input_for_display(&input, 5, 40);
        assert_eq!(rows.len(), 1);
        assert_eq!(row, 0);
        assert_eq!(col, 5);
    }

    #[test]
    fn wrap_input_cursor_at_zero_is_column_zero_of_first_row() {
        let input = chars("hello");
        let (_, row, col) = wrap_input_for_display(&input, 0, 40);
        assert_eq!((row, col), (0, 0));
    }

    #[test]
    fn wrap_input_respects_explicit_newlines() {
        let input = chars("first line\nsecond line");
        let (rows, row, col) = wrap_input_for_display(&input, input.len(), 80);
        assert_eq!(rows, vec!["first line".to_string(), "second line".to_string()]);
        assert_eq!((row, col), (1, "second line".len()));
    }

    #[test]
    fn command_suggestions_lists_every_agent_on_bare_command() {
        let all = command_suggestions("/agent", &[]);
        assert_eq!(all.len(), ALL_AGENTS.len());
    }

    #[test]
    fn command_suggestions_filters_agents_by_key_substring() {
        let matches = command_suggestions("/agent crypto", &[]);
        let labels: Vec<&str> = matches.iter().map(|s| s.label.as_str()).collect();
        assert!(labels.iter().any(|l| l.contains("crypto")), "expected a crypto-ish agent in {labels:?}");
        for s in &matches {
            assert_eq!(s.complete, format!("/agent {} ", s.label));
        }
    }

    #[test]
    fn command_suggestions_ignores_an_unrelated_message() {
        // "/agentic workflows are great" isn't the /agent command -- no
        // space (or end-of-input) right after "/agent".
        assert!(command_suggestions("/agentic workflows are great", &[]).is_empty());
        assert!(command_suggestions("just a normal question", &[]).is_empty());
    }

    #[test]
    fn command_suggestions_lists_plain_commands_while_typing_the_slash() {
        // "/exi" is a prefix of "/exit" only.
        let matches = command_suggestions("/exi", &[]);
        assert_eq!(matches.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(), vec!["/exit"]);
        assert_eq!(matches[0].complete, "/exit");

        // "/e" is now ambiguous between "/exit" and "/export".
        let matches = command_suggestions("/e", &[]);
        assert_eq!(matches.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(), vec!["/export", "/exit"]);
    }

    #[test]
    fn command_suggestions_bare_slash_offers_agent_and_every_plain_command() {
        let matches = command_suggestions("/", &[]);
        let labels: Vec<&str> = matches.iter().map(|s| s.label.as_str()).collect();
        assert!(labels.contains(&"/agent <key>"));
        for (cmd, _) in PLAIN_COMMANDS {
            assert!(labels.contains(cmd), "missing {cmd} in {labels:?}");
        }
    }

    #[test]
    fn command_suggestions_mode_lists_all_three_and_filters() {
        let all = command_suggestions("/mode", &[]);
        assert_eq!(all.len(), 3);
        let semi = command_suggestions("/mode semi", &[]);
        assert_eq!(semi.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(), vec!["semi-auto"]);
        assert_eq!(semi[0].complete, "/mode semi-auto");
    }

    #[test]
    fn permission_mode_cycles_through_all_three_and_back() {
        assert_eq!(PermissionMode::Manual.next(), PermissionMode::SemiAuto);
        assert_eq!(PermissionMode::SemiAuto.next(), PermissionMode::Auto);
        assert_eq!(PermissionMode::Auto.next(), PermissionMode::Manual);
    }

    #[test]
    fn strip_ansi_removes_csi_sequences_but_keeps_the_text() {
        assert_eq!(strip_ansi("\x1b[1;35mhello\x1b[0m world"), "hello world");
        assert_eq!(strip_ansi("no escapes here"), "no escapes here");
        assert_eq!(strip_ansi("\x1b[2m→ write_file(path: \"a.rs\")\x1b[0m"), "→ write_file(path: \"a.rs\")");
    }

    #[test]
    fn parse_confirm_answer_matches_terminal_ios_three_way_split() {
        assert_eq!(parse_confirm_answer("y"), Decision::Allow);
        assert_eq!(parse_confirm_answer("Yes"), Decision::Allow);
        assert_eq!(parse_confirm_answer(""), Decision::Deny);
        assert_eq!(parse_confirm_answer("n"), Decision::Deny);
        assert!(matches!(parse_confirm_answer("do X instead"), Decision::Redirect(n) if n == "do X instead"));
    }

    #[test]
    fn parse_choice_answer_picks_by_number_and_all() {
        let options = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(parse_choice_answer("2", &options, false), vec!["b".to_string()]);
        assert_eq!(parse_choice_answer("1,3", &options, true), vec!["a".to_string(), "c".to_string()]);
        assert_eq!(parse_choice_answer("all", &options, true), options.clone());
        assert_eq!(parse_choice_answer("9", &options, false), Vec::<String>::new(), "out-of-range number picks nothing");
        assert_eq!(parse_choice_answer("nope", &options, false), Vec::<String>::new());
    }

    #[test]
    fn parse_list_marker_recognizes_bullets_and_numbers() {
        assert_eq!(parse_list_marker("- item one"), Some(("", "•".to_string(), "item one")));
        assert_eq!(parse_list_marker("* item two"), Some(("", "•".to_string(), "item two")));
        assert_eq!(parse_list_marker("  - indented"), Some(("  ", "•".to_string(), "indented")));
        assert_eq!(parse_list_marker("3. third step"), Some(("", "3.".to_string(), "third step")));
        assert_eq!(parse_list_marker("12) twelfth"), Some(("", "12.".to_string(), "twelfth")));
    }

    #[test]
    fn parse_list_marker_ignores_non_list_text() {
        // A bare hyphen or a version number mid-sentence isn't a list.
        assert_eq!(parse_list_marker("a - b"), None);
        assert_eq!(parse_list_marker("released v3.2 today"), None);
        assert_eq!(parse_list_marker("just a normal sentence"), None);
    }

    #[test]
    fn push_diff_aware_colors_plus_and_minus_lines() {
        let mut out = Vec::new();
        push_diff_aware(&mut out, 80, Style::default().fg(Color::Yellow), "write_file a.rs\n-old line\n+new line\n unchanged");
        let color_of = |i: usize| out[i].spans[0].style.fg;
        assert_eq!(color_of(0), Some(Color::Yellow), "non-diff header line keeps the default style");
        assert_eq!(color_of(1), Some(Color::Red));
        assert_eq!(color_of(2), Some(Color::Green));
        assert_eq!(color_of(3), Some(Color::Yellow));
    }

    #[test]
    fn markdown_reply_renders_a_bulleted_list_with_hanging_indent() {
        let width = 40;
        let mut out = Vec::new();
        push_markdown(&mut out, width, ("AGENT ".into(), Style::default()), "- first item\n- a second item that is long enough to wrap onto another row for sure");
        let joined: String = out.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(joined.contains("first item"));
        assert!(joined.contains("second item"));
        for line in &out {
            let rendered: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
            assert!(rendered <= width, "row {rendered} chars wide exceeds width {width}: {:?}", line_text(line));
        }
    }

    #[test]
    fn extract_question_strips_the_task_wrapper_and_any_retrieved_context() {
        assert_eq!(extract_question("Task: fix the bug\n\nRetrieved context:\nsome file contents here"), "fix the bug");
        assert_eq!(extract_question("Task: no context attached"), "no context attached");
    }

    #[test]
    fn extract_question_strips_the_additional_instruction_wrapper() {
        assert_eq!(extract_question("Additional instruction: and now add tests"), "and now add tests");
    }

    #[test]
    fn extract_question_passes_through_anything_unrecognized() {
        assert_eq!(extract_question("  just plain text  "), "just plain text");
    }

    #[test]
    fn command_suggestions_resume_lists_known_sessions_most_recent_first() {
        let sessions = vec!["111-aaa".to_string(), "222-bbb".to_string()];
        let matches = command_suggestions("/resume", &sessions);
        let labels: Vec<&str> = matches.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, vec!["222-bbb", "111-aaa"]);
        assert_eq!(matches[0].complete, "/resume 222-bbb");
    }

    #[test]
    fn command_suggestions_resume_filters_by_substring() {
        let sessions = vec!["111-aaa".to_string(), "222-bbb".to_string()];
        let matches = command_suggestions("/resume bbb", &sessions);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].label, "222-bbb");
    }

    fn test_state() -> ChatState {
        ChatState {
            agent: &ALL_AGENTS[0],
            cfg_model: String::new(),
            transcript: Vec::new(),
            input: Vec::new(),
            cursor: 0,
            waiting: false,
            waiting_since: Instant::now(),
            current_task: None,
            scroll: 0,
            tools_expanded: false,
            suggestion_idx: 0,
            user_color: Color::Cyan,
            agent_color: Color::Magenta,
            queued: None,
            checkpoint_session: None,
            mode: PermissionMode::Manual,
            pending_confirm: None,
            pending_choice: None,
            plan: Vec::new(),
            should_quit: false,
            known_sessions: Vec::new(),
            history: Vec::new(),
            history_cursor: None,
            history_draft: Vec::new(),
            history_search: None,
            context_extra: 0,
            num_ctx: 8192,
        }
    }

    #[test]
    fn history_prev_walks_backward_and_saves_the_live_draft() {
        let mut state = test_state();
        state.history = vec!["first question".to_string(), "second question".to_string()];
        state.input = chars("still typing");
        history_prev(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "second question");
        history_prev(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "first question");
        // Already at the oldest entry -- stays put.
        history_prev(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "first question");
    }

    #[test]
    fn history_next_restores_the_saved_draft_past_the_newest_entry() {
        let mut state = test_state();
        state.history = vec!["only question".to_string()];
        state.input = chars("my draft");
        history_prev(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "only question");
        history_next(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "my draft");
        assert_eq!(state.history_cursor, None);
    }

    #[test]
    fn history_next_is_a_no_op_when_not_browsing() {
        let mut state = test_state();
        state.history = vec!["a question".to_string()];
        state.input = chars("untouched");
        history_next(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "untouched");
    }

    #[test]
    fn base64_encode_matches_known_vectors() {
        // RFC 4648 test vectors, including the padding cases.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn history_search_update_finds_the_most_recent_matching_entry() {
        let mut state = test_state();
        state.history = vec!["fix the parser".to_string(), "add tests".to_string(), "fix the lexer".to_string()];
        state.history_search = Some(HistorySearch { query: "fix".to_string(), match_idx: None });
        history_search_update(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "fix the lexer");
        assert_eq!(state.history_search.unwrap().match_idx, Some(2));
    }

    #[test]
    fn history_search_update_reports_no_match() {
        let mut state = test_state();
        state.history = vec!["add tests".to_string()];
        state.history_search = Some(HistorySearch { query: "zzz".to_string(), match_idx: None });
        history_search_update(&mut state);
        assert!(state.input.is_empty());
        assert_eq!(state.history_search.unwrap().match_idx, None);
    }

    #[test]
    fn history_search_next_older_cycles_to_the_previous_match() {
        let mut state = test_state();
        state.history = vec!["fix the parser".to_string(), "add tests".to_string(), "fix the lexer".to_string()];
        state.history_search = Some(HistorySearch { query: "fix".to_string(), match_idx: None });
        history_search_update(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "fix the lexer");
        history_search_next_older(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "fix the parser");
        // No older match left -- stays put rather than wrapping or panicking.
        history_search_next_older(&mut state);
        assert_eq!(state.input.iter().collect::<String>(), "fix the parser");
    }

    #[test]
    fn estimate_context_chars_sums_transcript_text_plus_the_hidden_dump() {
        let mut state = test_state();
        state.context_extra = 100;
        state.transcript.push(TLine::You("hello".to_string())); // 5
        state.transcript.push(TLine::Agent { label: "A".to_string(), text: "world!".to_string() }); // 6
        state.transcript.push(TLine::Tools(vec![ToolEntry { name: "x".to_string(), call: "abcd".to_string(), result: Some("ef".to_string()) }])); // 4+2
        assert_eq!(estimate_context_chars(&state), 100 + 5 + 6 + 4 + 2);
    }
}

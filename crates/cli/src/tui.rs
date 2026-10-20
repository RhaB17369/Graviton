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

use crate::agentic;
use crate::agents::{AgentSpec, ALL_AGENTS};
use crate::{build_context_sync, finish_context, open_or_init_repo_db, repo_root, resolve_agent};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event, EventStream, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use futures_util::StreamExt;
use graviton_core::Config;
use graviton_llm::{ChatMessage, OllamaClient};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;
use std::io::Stdout;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

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
    Sys(String),
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
    Done(Vec<ChatMessage>),
    Error(String),
}

struct ChatState {
    agent: &'static AgentSpec,
    cfg_model: String,
    messages: Vec<ChatMessage>,
    transcript: Vec<TLine>,
    /// Chars, not a `String` -- `cursor` is a char index, and inserting/
    /// removing at an arbitrary position needs `Vec::insert`/`remove`
    /// rather than `String`'s byte-offset-only API (which would panic
    /// mid-UTF8-codepoint on accented input).
    input: Vec<char>,
    cursor: usize,
    waiting: bool,
    waiting_since: Instant,
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
    /// request loop isn't reentrant: one spawned task owns `messages`
    /// for the duration of a turn), then auto-submitted the moment
    /// `ChatEvent::Done`/`Error` clears `waiting`. Typing itself was
    /// never blocked on `waiting` to begin with (see `drive`) -- only
    /// submitting was, so you can draft, edit, or even switch agents
    /// with `/agent` mid-generation without losing anything you typed.
    queued: Option<String>,
    should_quit: bool,
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

pub async fn run_chat_tui(cfg: &Config) -> Result<()> {
    let root = repo_root()?;
    let client = OllamaClient::new(&cfg.ollama_host);
    let agent = resolve_agent("architect")?;

    let mut state = ChatState {
        agent,
        cfg_model: cfg.model_for(agent.tier, agent.key).to_string(),
        messages: vec![ChatMessage::system(agent.base_prompt())],
        transcript: Vec::new(),
        input: Vec::new(),
        cursor: 0,
        waiting: false,
        waiting_since: Instant::now(),
        scroll: 0,
        tools_expanded: false,
        suggestion_idx: 0,
        user_color: theme_color(cfg.tui_user_color.as_deref(), Color::Cyan),
        agent_color: theme_color(cfg.tui_agent_color.as_deref(), Color::Magenta),
        queued: None,
        should_quit: false,
    };

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

    let result = drive(&mut terminal, &mut state, cfg, &client, &root).await;

    disable_raw_mode().ok();
    crossterm::execute!(terminal.backend_mut(), DisableMouseCapture, DisableBracketedPaste, LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();
    result
}

async fn drive(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    state: &mut ChatState,
    cfg: &Config,
    client: &OllamaClient,
    root: &std::path::Path,
) -> Result<()> {
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
                        // The autocomplete popup, when showing, steals
                        // Up/Down/Tab/Enter/Esc for its own navigation --
                        // computed fresh from the current input on every
                        // keystroke, never stored, so there's nothing to
                        // go stale.
                        // Typing, editing, and the autocomplete popup all
                        // work regardless of `waiting` -- only *sending*
                        // is gated on it (the request loop isn't
                        // reentrant), and that's handled below by
                        // queuing instead of refusing the keystroke.
                        let input_text: String = state.input.iter().collect();
                        let suggestions = command_suggestions(&input_text);
                        let popup_active = !suggestions.is_empty();
                        if popup_active {
                            state.suggestion_idx = state.suggestion_idx.min(suggestions.len() - 1);
                        }

                        if key.code == KeyCode::Esc && state.queued.is_some() {
                            state.queued = None;
                        } else if popup_active && key.code == KeyCode::Esc {
                            state.input.clear();
                            state.cursor = 0;
                        } else if key.code == KeyCode::Esc || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)) {
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
                                submit_or_queue(state, cfg, client, root, &tx);
                            }
                        } else {
                            match key.code {
                                // Vertical motion scrolls the transcript --
                                // kept as the scroll keys even while typing
                                // is otherwise unlocked during `waiting`,
                                // same as before.
                                KeyCode::Up => state.scroll = state.scroll.saturating_add(1),
                                KeyCode::Down => state.scroll = state.scroll.saturating_sub(1),
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
                                KeyCode::Enter => submit_or_queue(state, cfg, client, root, &tx),
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
                        submit(state, cfg, client, root, &tx);
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
/// otherwise queue -- the request loop owns `messages`/the model
/// connection for the duration of one turn, so a second one can't run
/// concurrently, but there's no reason typing or *sending* the next
/// question has to wait for that, any more than Claude Code makes you
/// wait to queue up a follow-up. `drive`'s `ChatEvent::Done`/`Error`
/// handling flushes the queue the instant `waiting` clears.
fn submit_or_queue(state: &mut ChatState, cfg: &Config, client: &OllamaClient, root: &std::path::Path, tx: &mpsc::UnboundedSender<ChatEvent>) {
    if !state.waiting {
        submit(state, cfg, client, root, tx);
        return;
    }
    let line: String = state.input.iter().collect::<String>().trim().to_string();
    state.input.clear();
    state.cursor = 0;
    if !line.is_empty() {
        state.queued = Some(line);
    }
}

fn submit(state: &mut ChatState, cfg: &Config, client: &OllamaClient, root: &std::path::Path, tx: &mpsc::UnboundedSender<ChatEvent>) {
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
        state.messages = vec![ChatMessage::system(state.agent.base_prompt())];
        state.transcript.push(TLine::Sys("(conversation cleared)".into()));
        state.scroll = 0;
        return;
    }
    if let Some(key) = line.strip_prefix("/agent ") {
        match resolve_agent(key.trim()) {
            Ok(a) => {
                state.agent = a;
                state.cfg_model = cfg.model_for(a.tier, a.key).to_string();
                state.messages = vec![ChatMessage::system(a.base_prompt())];
                state.transcript.push(TLine::Sys(format!("(switched to {} — conversation reset)", a.display)));
            }
            Err(e) => state.transcript.push(TLine::Err(format!("{e:#}"))),
        }
        state.scroll = 0;
        return;
    }

    state.transcript.push(TLine::You(line.clone()));
    state.waiting = true;
    state.waiting_since = Instant::now();
    // A new question is always worth seeing, even if you'd scrolled up
    // to reread earlier history -- jump back to live.
    state.scroll = 0;

    let client = client.clone();
    let cfg = cfg.clone();
    let root = root.to_path_buf();
    let model = cfg.model_for(state.agent.tier, state.agent.key).to_string();
    let mut local_messages = state.messages.clone();
    let tx = tx.clone();

    tokio::spawn(async move {
        let outcome: Result<Vec<ChatMessage>> = async {
            let conn = open_or_init_repo_db(&cfg, &root)?;
            let (groups, semantic_src) = build_context_sync(&cfg, &root, &conn, &line, &[])?;
            drop(conn);
            let context_text = finish_context(&cfg, &line, groups, semantic_src).await?;
            let user_msg = if context_text.is_empty() {
                line.clone()
            } else {
                format!("Question: {line}\n\nRetrieved context:\n{context_text}")
            };
            local_messages.push(ChatMessage::user(user_msg));

            let tx_tok = tx.clone();
            let on_token = move |tok: &str| {
                let _ = tx_tok.send(ChatEvent::Token(tok.to_string()));
            };
            let tx_think = tx.clone();
            let on_thinking = move |tok: &str| {
                let _ = tx_think.send(ChatEvent::Thinking(tok.to_string()));
            };
            let tx_tool = tx.clone();
            let on_tool_call = move |name: &str, args: &serde_json::Value| {
                let _ = tx_tool.send(ChatEvent::ToolCall(name.to_string(), args.clone()));
            };
            let tx_res = tx.clone();
            let on_tool_result = move |_name: &str, out: &str| {
                let _ = tx_res.send(ChatEvent::ToolResult(agentic::tool_result_preview(out)));
            };
            agentic::run_read_only_loop_on(&client, &cfg, &root, &model, &mut local_messages, on_token, on_thinking, on_tool_call, on_tool_result).await?;
            Ok(local_messages)
        }
        .await;

        match outcome {
            Ok(msgs) => {
                let _ = tx.send(ChatEvent::Done(msgs));
            }
            Err(e) => {
                let _ = tx.send(ChatEvent::Error(format!("{e:#}")));
            }
        }
    });
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
        ChatEvent::Done(messages) => {
            state.messages = messages;
            state.waiting = false;
        }
        ChatEvent::Error(e) => {
            state.transcript.push(TLine::Err(e));
            state.waiting = false;
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
const PLAIN_COMMANDS: &[(&str, &str)] = &[("/clear", "reset the conversation"), ("/exit", "quit")];

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
fn command_suggestions(input: &str) -> Vec<Suggestion> {
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
    if input.starts_with('/') && !input.contains(' ') {
        let query = input.to_lowercase();
        let mut out: Vec<Suggestion> = Vec::new();
        if "/agent".starts_with(&query) {
            out.push(Suggestion { label: "/agent <key>".into(), description: "switch to a specialist agent".into(), complete: "/agent ".into() });
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
    // While waiting, row 1 is always the spinner status; a draft typed
    // for the *next* question (queued on Enter, not sent) gets its own
    // rows below it instead of being hidden until the current reply
    // finishes -- otherwise unlocking typing during `waiting` would let
    // you type into a box that visibly shows nothing back.
    let visible_input_rows = if state.waiting {
        1 + if state.input.is_empty() { 0 } else { input_rows.len().min(5) }
    } else {
        input_rows.len().min(6)
    };
    let input_height = (visible_input_rows + 2) as u16;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(3), Constraint::Length(input_height)])
        .split(area);

    draw_header(f, chunks[0], state);
    let rows = draw_transcript(f, chunks[1], state);
    draw_input(f, chunks[2], state, &input_rows, cursor_row, cursor_col);

    if !state.waiting {
        let input_text: String = state.input.iter().collect();
        let suggestions = command_suggestions(&input_text);
        if !suggestions.is_empty() {
            draw_command_popup(f, chunks[2], state, &suggestions);
        }
    }
    rows
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    let line = Line::from(vec![
        Span::styled("GRAVITON", Style::default().fg(state.agent_color).add_modifier(Modifier::BOLD)),
        Span::styled(" · ", Style::default().fg(Color::DarkGray)),
        Span::styled(state.agent.display, Style::default().fg(state.agent_color).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" [{}]", state.cfg_model), Style::default().fg(Color::DarkGray)),
        Span::styled(
            "   /agent switch · /clear reset · ↑↓ PgUp/PgDn scroll · Tab tools · Alt+Enter newline · Esc quit",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
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

fn draw_input(f: &mut ratatui::Frame, area: Rect, state: &ChatState, rows: &[String], cursor_row: usize, cursor_col: usize) {
    let border_color = if state.waiting { Color::Yellow } else { Color::DarkGray };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let lines: Vec<Line> = if state.waiting {
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
                let marker = if i == 0 { Span::styled("› ", Style::default().fg(state.user_color)) } else { Span::raw("  ") };
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
        let all = command_suggestions("/agent");
        assert_eq!(all.len(), ALL_AGENTS.len());
    }

    #[test]
    fn command_suggestions_filters_agents_by_key_substring() {
        let matches = command_suggestions("/agent crypto");
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
        assert!(command_suggestions("/agentic workflows are great").is_empty());
        assert!(command_suggestions("just a normal question").is_empty());
    }

    #[test]
    fn command_suggestions_lists_plain_commands_while_typing_the_slash() {
        // "/e" is a prefix of "/exit" only (not "/agent" or "/clear"/"/quit").
        let matches = command_suggestions("/e");
        assert_eq!(matches.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(), vec!["/exit"]);
        assert_eq!(matches[0].complete, "/exit");
    }

    #[test]
    fn command_suggestions_bare_slash_offers_agent_and_every_plain_command() {
        let matches = command_suggestions("/");
        let labels: Vec<&str> = matches.iter().map(|s| s.label.as_str()).collect();
        assert!(labels.contains(&"/agent <key>"));
        for (cmd, _) in PLAIN_COMMANDS {
            assert!(labels.contains(cmd), "missing {cmd} in {labels:?}");
        }
    }
}

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
use crate::agents::AgentSpec;
use crate::{build_context_sync, finish_context, open_or_init_repo_db, repo_root, resolve_agent};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event, EventStream, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind,
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
    should_quit: bool,
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
        should_quit: false,
    };

    enable_raw_mode().context("enabling raw terminal mode")?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen, EnableMouseCapture).context("entering alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("creating terminal")?;
    terminal.clear().ok();

    let result = drive(&mut terminal, &mut state, cfg, &client, &root).await;

    disable_raw_mode().ok();
    crossterm::execute!(terminal.backend_mut(), DisableMouseCapture, LeaveAlternateScreen).ok();
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
                        if key.code == KeyCode::Esc || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)) {
                            state.should_quit = true;
                        } else {
                            match key.code {
                                // Vertical motion scrolls the transcript; it's
                                // always available, even while waiting, so you
                                // can reread earlier context during a slow
                                // generation without it fighting for keys with
                                // text entry (which is blocked while waiting
                                // anyway -- there's nothing to edit).
                                KeyCode::Up => state.scroll = state.scroll.saturating_add(1),
                                KeyCode::Down => state.scroll = state.scroll.saturating_sub(1),
                                KeyCode::PageUp => state.scroll = state.scroll.saturating_add(page_rows),
                                KeyCode::PageDown => state.scroll = state.scroll.saturating_sub(page_rows),
                                KeyCode::Tab => state.tools_expanded = !state.tools_expanded,
                                // Horizontal motion + editing only makes sense
                                // on the input you're typing, which only
                                // exists while not waiting on a reply.
                                _ if !state.waiting => match key.code {
                                    KeyCode::Enter => submit(state, cfg, client, root, &tx),
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
                                },
                                _ => {}
                            }
                        }
                    }
                    Event::Mouse(m) => match m.kind {
                        MouseEventKind::ScrollUp => state.scroll = state.scroll.saturating_add(3),
                        MouseEventKind::ScrollDown => state.scroll = state.scroll.saturating_sub(3),
                        _ => {}
                    },
                    _ => {}
                }
            }
            _ = tick.tick() => {
                // Just a redraw pulse for the spinner frame/elapsed time;
                // no state change needed here.
            }
            Some(ev) = rx.recv() => apply(state, ev),
        }

        if state.should_quit {
            break;
        }
        terminal.draw(|f| page_rows = draw(f, state))?;
    }
    Ok(())
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
            let tx_tool = tx.clone();
            let on_tool_call = move |name: &str, args: &serde_json::Value| {
                let _ = tx_tool.send(ChatEvent::ToolCall(name.to_string(), args.clone()));
            };
            let tx_res = tx.clone();
            let on_tool_result = move |_name: &str, out: &str| {
                let _ = tx_res.send(ChatEvent::ToolResult(agentic::tool_result_preview(out)));
            };
            agentic::run_read_only_loop_on(&client, &cfg, &root, &model, &mut local_messages, on_token, on_tool_call, on_tool_result).await?;
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

fn draw(f: &mut ratatui::Frame, state: &ChatState) -> usize {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(3), Constraint::Length(3)])
        .split(area);

    draw_header(f, chunks[0], state);
    let rows = draw_transcript(f, chunks[1], state);
    draw_input(f, chunks[2], state);
    rows
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    let line = Line::from(vec![
        Span::styled("GRAVITON", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::styled(" · ", Style::default().fg(Color::DarkGray)),
        Span::styled(state.agent.display, Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(format!(" [{}]", state.cfg_model), Style::default().fg(Color::DarkGray)),
        Span::styled("   /agent switch · /clear reset · ↑↓ PgUp/PgDn scroll · Tab tools · Esc quit", Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(Paragraph::new(line), area);
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
                push_wrapped(&mut out, width, Some(("you   ".into(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))), s, Style::default());
            }
            TLine::Agent { label, text } => {
                if text.is_empty() {
                    continue;
                }
                let prefix = format!("{label:<6}");
                push_markdown(&mut out, width, (prefix, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)), text);
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

fn draw_input(f: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    let border_color = if state.waiting { Color::Yellow } else { Color::DarkGray };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let line = if state.waiting {
        let frame = SPINNER_FRAMES[(state.waiting_since.elapsed().as_millis() / 120) as usize % SPINNER_FRAMES.len()];
        Line::from(vec![
            Span::styled(format!("{frame} "), Style::default().fg(Color::Yellow)),
            Span::styled(format!("{} is thinking… {}s", state.agent.display, state.waiting_since.elapsed().as_secs()), Style::default().fg(Color::DarkGray)),
        ])
    } else {
        // A real cursor position (Left/Right/Home/End-addressable), drawn
        // as a reversed-video cell rather than relying on the terminal's
        // own cursor -- we never move that (no `f.set_cursor_position`),
        // so this is the only visible insertion point there is.
        let before: String = state.input[..state.cursor].iter().collect();
        let at: String = state.input.get(state.cursor).map(|c| c.to_string()).unwrap_or_else(|| " ".to_string());
        let after: String = state.input[state.cursor.min(state.input.len())..].iter().skip(1).collect();
        Line::from(vec![
            Span::styled("› ", Style::default().fg(Color::Cyan)),
            Span::raw(before),
            Span::styled(at, Style::default().add_modifier(Modifier::REVERSED)),
            Span::raw(after),
        ])
    };
    f.render_widget(Paragraph::new(line).block(block), area);
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
}

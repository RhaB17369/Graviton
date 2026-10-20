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
    ToolCall(String),
    ToolResult(String),
    Sys(String),
    Err(String),
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
    input: String,
    waiting: bool,
    waiting_since: Instant,
    /// Rows scrolled back from the live bottom. 0 = pinned to the
    /// newest content (auto-follows as tokens stream in); any other
    /// value freezes the view there until the user scrolls back down
    /// or sends a new message (`submit` resets this to 0 so a fresh
    /// question is never missed because you were reading history).
    scroll: usize,
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
        input: String::new(),
        waiting: false,
        waiting_since: Instant::now(),
        scroll: 0,
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
                                KeyCode::Up => state.scroll = state.scroll.saturating_add(1),
                                KeyCode::Down => state.scroll = state.scroll.saturating_sub(1),
                                KeyCode::PageUp => state.scroll = state.scroll.saturating_add(page_rows),
                                KeyCode::PageDown => state.scroll = state.scroll.saturating_sub(page_rows),
                                KeyCode::Home => state.scroll = usize::MAX / 2,
                                KeyCode::End => state.scroll = 0,
                                _ if !state.waiting => match key.code {
                                    KeyCode::Enter => submit(state, cfg, client, root, &tx),
                                    KeyCode::Backspace => { state.input.pop(); }
                                    KeyCode::Char(c) => state.input.push(c),
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
    let line = state.input.trim().to_string();
    state.input.clear();
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
    state.transcript.push(TLine::Agent { label: state.agent.display.to_string(), text: String::new() });
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
            if let Some(TLine::Agent { text, .. }) = state.transcript.last_mut() {
                text.push_str(&tok);
            }
        }
        ChatEvent::ToolCall(name, args) => {
            state.transcript.push(TLine::ToolCall(format_tool_call(&name, &args)));
        }
        ChatEvent::ToolResult(preview) => {
            state.transcript.push(TLine::ToolResult(preview));
            state.transcript.push(TLine::Agent { label: state.agent.display.to_string(), text: String::new() });
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
        Span::styled("   /agent <key> switch · /clear reset · ↑↓ PgUp/PgDn scroll · Esc quit", Style::default().fg(Color::DarkGray)),
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
                push_wrapped(&mut out, width, Some((prefix, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))), text, Style::default());
                out.push(Line::from(""));
            }
            TLine::ToolCall(s) => push_wrapped(&mut out, width, Some(("  → ".into(), Style::default().fg(Color::DarkGray))), s, Style::default().fg(Color::DarkGray)),
            TLine::ToolResult(s) => {
                push_wrapped(&mut out, width, Some(("  ← ".into(), Style::default().fg(Color::Green))), s, Style::default().fg(Color::DarkGray));
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
        format!(" conversation — scrolled back, End to jump to latest ({scroll} row{}) ", if scroll == 1 { "" } else { "s" })
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
        Line::from(vec![Span::styled("› ", Style::default().fg(Color::Cyan)), Span::raw(state.input.clone())])
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
}

//! Full-screen chat TUI for bare `grv` (no subcommand).
//!
//! Replaces the earlier plain `print!`/`stdin().read_line()` REPL, which
//! looked broken on this project's own (slow, CPU-only, often
//! contended) test hardware: a question could sit for minutes with zero
//! feedback, indistinguishable from a hang. This version has a real
//! input box pinned at the bottom, a scrolling transcript above it, and
//! a live elapsed-time spinner while a request is in flight — the two
//! visual "fields" (what you typed, what the agent said) are areas of
//! one screen, not two separate command-line prompts printed in
//! sequence. Pulls in `ratatui` + `crossterm`, a deliberate departure
//! from this project's usual zero-extra-dependency stance for scalar
//! config parsing — a real terminal UI needs a real terminal UI crate.

use crate::agentic;
use crate::agents::AgentSpec;
use crate::{build_context_sync, finish_context, open_or_init_repo_db, repo_root, resolve_agent};
use anyhow::{Context, Result};
use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use futures_util::StreamExt;
use graviton_core::Config;
use graviton_llm::{ChatMessage, OllamaClient};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
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
    Tool(String),
    Sys(String),
    Err(String),
}

enum ChatEvent {
    Token(String),
    ToolCall(String, serde_json::Value),
    Done(Vec<ChatMessage>),
    Error(String),
}

struct ChatState {
    agent: &'static AgentSpec,
    messages: Vec<ChatMessage>,
    transcript: Vec<TLine>,
    input: String,
    waiting: bool,
    waiting_since: Instant,
    should_quit: bool,
}

const SPINNER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

pub async fn run_chat_tui(cfg: &Config) -> Result<()> {
    let root = repo_root()?;
    let client = OllamaClient::new(&cfg.ollama_host);
    let agent = resolve_agent("architect")?;

    let mut state = ChatState {
        agent,
        messages: vec![ChatMessage::system(agent.base_prompt())],
        transcript: vec![TLine::Sys(format!(
            "GRAVITON — chatting with {} [{}]. /agent <key> switch · /clear reset · /exit or Esc quit.",
            agent.display,
            cfg.model_for(agent.tier, agent.key)
        ))],
        input: String::new(),
        waiting: false,
        waiting_since: Instant::now(),
        should_quit: false,
    };

    enable_raw_mode().context("enabling raw terminal mode")?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen).context("entering alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("creating terminal")?;
    terminal.clear().ok();

    let result = drive(&mut terminal, &mut state, cfg, &client, &root).await;

    disable_raw_mode().ok();
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
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

    terminal.draw(|f| draw(f, state))?;

    loop {
        tokio::select! {
            maybe_ev = events.next() => {
                let Some(Ok(ev)) = maybe_ev else { continue };
                if let Event::Key(key) = ev {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    if key.code == KeyCode::Esc || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)) {
                        state.should_quit = true;
                    } else if !state.waiting {
                        match key.code {
                            KeyCode::Enter => submit(state, cfg, client, root, &tx),
                            KeyCode::Backspace => { state.input.pop(); }
                            KeyCode::Char(c) => state.input.push(c),
                            _ => {}
                        }
                    }
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
        terminal.draw(|f| draw(f, state))?;
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
        return;
    }
    if let Some(key) = line.strip_prefix("/agent ") {
        match resolve_agent(key.trim()) {
            Ok(a) => {
                state.agent = a;
                state.messages = vec![ChatMessage::system(a.base_prompt())];
                state.transcript.push(TLine::Sys(format!(
                    "(switched to {} [{}] — conversation reset)",
                    a.display,
                    cfg.model_for(a.tier, a.key)
                )));
            }
            Err(e) => state.transcript.push(TLine::Err(format!("{e:#}"))),
        }
        return;
    }

    state.transcript.push(TLine::You(line.clone()));
    state.transcript.push(TLine::Agent { label: state.agent.display.to_string(), text: String::new() });
    state.waiting = true;
    state.waiting_since = Instant::now();

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
            agentic::run_read_only_loop_on(&client, &cfg, &root, &model, &mut local_messages, on_token, on_tool_call).await?;
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

fn apply(state: &mut ChatState, ev: ChatEvent) {
    match ev {
        ChatEvent::Token(tok) => {
            if let Some(TLine::Agent { text, .. }) = state.transcript.last_mut() {
                text.push_str(&tok);
            }
        }
        ChatEvent::ToolCall(name, args) => {
            state.transcript.push(TLine::Tool(format!("→ {name}({args})")));
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

fn draw(f: &mut ratatui::Frame, state: &ChatState) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(3), Constraint::Length(3)])
        .split(area);

    draw_header(f, chunks[0], state);
    draw_transcript(f, chunks[1], state);
    draw_input(f, chunks[2], state);
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    let text = format!(" GRAVITON · {} ", state.agent.display);
    f.render_widget(
        Paragraph::new(text).style(Style::default().fg(Color::White).bg(Color::Magenta).add_modifier(Modifier::BOLD)),
        area,
    );
}

fn draw_transcript(f: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    let mut lines: Vec<Line> = Vec::new();
    for t in &state.transcript {
        match t {
            TLine::You(s) => {
                lines.push(Line::from(vec![Span::styled("you› ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(s.clone())]));
            }
            TLine::Agent { label, text } => {
                let prefix = Span::styled(format!("{label}› "), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD));
                if text.is_empty() {
                    lines.push(Line::from(vec![prefix]));
                } else {
                    for (i, part) in text.split('\n').enumerate() {
                        if i == 0 {
                            lines.push(Line::from(vec![prefix.clone(), Span::raw(part.to_string())]));
                        } else {
                            lines.push(Line::from(Span::raw(part.to_string())));
                        }
                    }
                }
            }
            TLine::Tool(s) => lines.push(Line::from(Span::styled(format!("  {s}"), Style::default().fg(Color::DarkGray)))),
            TLine::Sys(s) => lines.push(Line::from(Span::styled(s.clone(), Style::default().fg(Color::DarkGray)))),
            TLine::Err(s) => lines.push(Line::from(Span::styled(format!("error: {s}"), Style::default().fg(Color::Red)))),
        }
        lines.push(Line::from(""));
    }

    let height = area.height.saturating_sub(2) as usize;
    let total = lines.len();
    let scroll_off = total.saturating_sub(height) as u16;

    let block = Block::default().borders(Borders::ALL).title(" conversation ");
    f.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: false }).scroll((scroll_off, 0)),
        area,
    );
}

fn draw_input(f: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    let title = if state.waiting {
        let frame = SPINNER_FRAMES[(state.waiting_since.elapsed().as_millis() / 120) as usize % SPINNER_FRAMES.len()];
        format!(" {frame} thinking… {}s — Esc to quit ", state.waiting_since.elapsed().as_secs())
    } else {
        " type your question — Enter to send, Esc to quit ".to_string()
    };
    let style = if state.waiting { Style::default().fg(Color::Yellow) } else { Style::default() };
    let block = Block::default().borders(Borders::ALL).title(title).style(style);
    let content = if state.waiting { String::new() } else { state.input.clone() };
    f.render_widget(Paragraph::new(content).block(block), area);
}

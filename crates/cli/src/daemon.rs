//! `grv serve`: a background daemon speaking newline-delimited JSON-RPC 2.0
//! over a Unix socket (and optionally TCP), so an editor/IDE integration
//! can get code intelligence and agent answers without spawning a fresh
//! `grv` process — and re-opening the index, re-resolving the config, and
//! losing the live concurrency scheduler's warmed-up state — for every
//! keystroke-triggered query.
//!
//! Framing is one JSON object per line rather than LSP-style
//! `Content-Length` headers: JSON-RPC itself doesn't mandate a framing, and
//! NDJSON needs no header parser on either end — `nc -U` and a three-line
//! Python/Node script can already speak this. See ARCHITECTURE.md for the
//! full method list and example session.
//!
//! Model-calling methods (`ask`, `review`, `semantic_search`'s query
//! embedding) acquire a permit from the same `resources::LiveScheduler`
//! design used by `grv swarm`/`mission`, so a chatty editor firing several
//! requests at once still can't put more concurrent Ollama calls on the
//! machine than its RAM can hold.

use crate::agentic::Decision;
use crate::run_io::RunIo;
use crate::{agents, context, resources, semantic};
use anyhow::{Context, Result};
use graviton_core::Config;
use graviton_llm::OllamaClient;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::sync::{broadcast, mpsc, Mutex as AsyncMutex};

/// Concurrent model calls this daemon will allow in flight regardless of
/// RAM headroom -- an editor is one human's queries, not a swarm; there's
/// no reason to let it queue up more than a handful at once.
const DAEMON_HARD_CAP: usize = 6;

/// `sockaddr_un.sun_path` is 108 bytes on Linux including the null
/// terminator (~104 on macOS/BSD); kept well under either rather than
/// cutting it exactly to one platform's limit.
const MAX_SOCKET_PATH_LEN: usize = 100;

/// Where the socket lives when `--socket` isn't given: a short, stable
/// path under `$XDG_RUNTIME_DIR` (falling back to the system temp dir),
/// named by a hash of the repo root + index dir -- not
/// `<repo>/.graviton/grv.sock`, because a repo nested a few directories
/// past a long home/project path routinely exceeds the ~100-byte Unix
/// socket path limit (hit during testing, under a long scratch tmp path).
/// One repo -> one reused name across restarts; `remove_stale_socket`
/// still protects against a leftover file from a crashed daemon.
fn default_socket_path(root: &Path, index_dir: &str) -> PathBuf {
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    index_dir.hash(&mut hasher);
    let base = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    base.join("grv").join(format!("{:016x}.sock", hasher.finish()))
}

/// Reject a too-long socket path up front with a message that says what to
/// do about it, instead of letting `bind` fail with the OS's bare "path
/// must be shorter than SUN_LEN" -- which is exactly what an *explicit*
/// `--socket` under a deep path can still hit even with the short default
/// above.
fn check_socket_path_len(path: &Path) -> Result<()> {
    let len = path.as_os_str().len();
    if len > MAX_SOCKET_PATH_LEN {
        anyhow::bail!(
            "socket path is {len} bytes, too long for a Unix socket (~100-byte OS limit): {}\n\
             pass a shorter one, e.g. `grv serve --socket /tmp/grv.sock`",
            path.display()
        );
    }
    Ok(())
}

#[derive(Deserialize)]
struct RpcRequest {
    #[serde(default)]
    id: Value,
    method: String,
    #[serde(default)]
    params: Value,
}

struct DaemonCtx {
    cfg: Config,
    root: PathBuf,
    client: OllamaClient,
    scheduler: Arc<resources::LiveScheduler>,
    socket_path: PathBuf,
    /// Required in every request's `params.token` for connections accepted
    /// on the `--tcp` listener; `None` means `--tcp` wasn't passed with a
    /// token, which `serve()` never actually allows (see `serve` below) --
    /// this field exists so `handle_conn` has one place to check regardless
    /// of which listener accepted the connection. Never checked for Unix
    /// socket connections (filesystem permissions are the trust boundary
    /// there, same as `ollama serve`'s own socket).
    tcp_token: Option<String>,
    /// Live `run_start` sessions, keyed by the id `run_start` returns.
    /// Entries are never removed (a finished run's handle is cheap to
    /// keep, and `run_status`/a late `run_attach` should still be able to
    /// see how it ended) -- they only go away when the daemon exits.
    runs: AsyncMutex<HashMap<String, RunHandle>>,
}

/// One `run_start` session's live state, shared between the task actually
/// running `agentic::run` (via `ChannelIo`) and every connection that
/// calls `run_confirm`/`run_status`/`run_attach` for it. `Clone` because
/// the map holds these directly (not behind an `Arc`) -- every field here
/// is already cheap to clone (channel handles, an `Arc<Mutex<...>>>`).
#[derive(Clone)]
struct RunHandle {
    events_tx: broadcast::Sender<RunEvent>,
    confirm_tx: mpsc::UnboundedSender<Decision>,
    choice_tx: mpsc::UnboundedSender<Vec<String>>,
    status: Arc<StdMutex<RunStatusSnapshot>>,
}

#[derive(Clone, Default, serde::Serialize)]
struct RunStatusSnapshot {
    running: bool,
    pending_confirm: Option<String>,
    pending_choice: Option<(String, Vec<String>, bool)>,
    checkpoint_id: Option<String>,
    finished_ok: Option<bool>,
    finished_message: Option<String>,
}

/// What a `run_start` session reports as it goes -- `run_attach` streams
/// these as `{"jsonrpc":"2.0","method":"run_event","params":{...}}`
/// notifications (`session_id` plus whatever `to_notification` adds).
#[derive(Clone)]
enum RunEvent {
    Output(String),
    Token(String),
    ConfirmRequest(String),
    AskChoice(String, Vec<String>, bool),
    Done { ok: bool, message: String },
}

impl RunEvent {
    fn to_notification(&self, session_id: &str) -> Value {
        match self {
            RunEvent::Output(line) => json!({"jsonrpc":"2.0","method":"run_event","params":{"session_id":session_id,"kind":"output","line":line}}),
            RunEvent::Token(text) => json!({"jsonrpc":"2.0","method":"run_event","params":{"session_id":session_id,"kind":"token","text":text}}),
            RunEvent::ConfirmRequest(action) => json!({"jsonrpc":"2.0","method":"run_event","params":{"session_id":session_id,"kind":"confirm_request","action":action}}),
            RunEvent::AskChoice(question, options, multi_select) => {
                json!({"jsonrpc":"2.0","method":"run_event","params":{"session_id":session_id,"kind":"ask_choice","question":question,"options":options,"multi_select":multi_select}})
            }
            RunEvent::Done { ok, message } => json!({"jsonrpc":"2.0","method":"run_event","params":{"session_id":session_id,"kind":"done","ok":ok,"message":message}}),
        }
    }
}

/// The `RunIo` a `run_start` session uses: output/tokens become broadcast
/// events any attached connection receives; a confirmation or an
/// `ask_user` choice blocks on its own per-session channel that
/// `run_confirm`/`run_answer_choice` (from any connection) feeds.
struct ChannelIo {
    events_tx: broadcast::Sender<RunEvent>,
    confirm_rx: AsyncMutex<mpsc::UnboundedReceiver<Decision>>,
    choice_rx: AsyncMutex<mpsc::UnboundedReceiver<Vec<String>>>,
    status: Arc<StdMutex<RunStatusSnapshot>>,
}

impl RunIo for ChannelIo {
    fn emit(&self, line: String) {
        let _ = self.events_tx.send(RunEvent::Output(line));
    }

    fn on_token(&self, tok: &str) {
        let _ = self.events_tx.send(RunEvent::Token(tok.to_string()));
    }

    fn note_checkpoint_id(&self, id: &str) {
        self.status.lock().unwrap().checkpoint_id = Some(id.to_string());
    }

    fn confirm(&self, auto_approve: bool, action: String) -> Pin<Box<dyn Future<Output = Decision> + Send + '_>> {
        Box::pin(async move {
            if auto_approve {
                return Decision::Allow;
            }
            self.status.lock().unwrap().pending_confirm = Some(action.clone());
            let _ = self.events_tx.send(RunEvent::ConfirmRequest(action));
            let mut rx = self.confirm_rx.lock().await;
            let decision = rx.recv().await.unwrap_or(Decision::Deny);
            self.status.lock().unwrap().pending_confirm = None;
            decision
        })
    }

    fn ask_choice(&self, question: String, options: Vec<String>, multi_select: bool) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        Box::pin(async move {
            self.status.lock().unwrap().pending_choice = Some((question.clone(), options.clone(), multi_select));
            let _ = self.events_tx.send(RunEvent::AskChoice(question, options, multi_select));
            let mut rx = self.choice_rx.lock().await;
            let picked = rx.recv().await.unwrap_or_default();
            self.status.lock().unwrap().pending_choice = None;
            picked
        })
    }
}

/// Not a cryptographically hardened secret generator -- combines wall
/// clock, process id, and a per-process counter through a plain hasher, no
/// `rand` dependency. Matches this daemon's actual threat model: stopping
/// an opportunistic/accidental hit on a `--tcp` port from doing anything,
/// for a tool meant to bind `127.0.0.1`/a trusted LAN, not stand in for
/// real auth on an internet-facing service.
fn generate_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = DefaultHasher::new();
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    COUNTER.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    // hash twice with a reseed so a 64-bit hash output doesn't feel this thin
    let first = hasher.finish();
    first.hash(&mut hasher);
    format!("{:016x}{:016x}", first, hasher.finish())
}

impl DaemonCtx {
    fn open_conn(&self) -> Result<rusqlite::Connection> {
        let db_path = graviton_core::db_path_for(&self.root, &self.cfg.index_dir)?;
        if !db_path.exists() {
            anyhow::bail!("no index found at {} -- run `grv index` first", db_path.display());
        }
        graviton_core::open_db(&db_path)
    }
}

pub async fn serve(cfg: &Config, root: PathBuf, socket: Option<PathBuf>, tcp: Option<String>, tcp_token: Option<String>) -> Result<()> {
    let socket_path = socket.unwrap_or_else(|| default_socket_path(&root, &cfg.index_dir));
    check_socket_path_len(&socket_path)?;
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    remove_stale_socket(&socket_path).await;

    let client = OllamaClient::new(&cfg.ollama_host);
    let all_models: Vec<String> = cfg.distinct_models().into_iter().map(String::from).collect();
    let sizes = resources::model_sizes_mb(&client).await;
    let scheduler = resources::LiveScheduler::spawn(all_models, sizes, DAEMON_HARD_CAP);

    // --tcp always requires a token -- auto-generated (and printed once,
    // here) unless the caller supplied one explicitly. Every request over
    // TCP must echo it back in params.token; Unix socket connections never
    // need it (filesystem permissions are that boundary instead).
    let tcp_token = tcp.as_ref().map(|_| tcp_token.unwrap_or_else(generate_token));

    let ctx = Arc::new(DaemonCtx {
        cfg: cfg.clone(),
        root: root.clone(),
        client,
        scheduler,
        socket_path: socket_path.clone(),
        tcp_token: tcp_token.clone(),
        runs: AsyncMutex::new(HashMap::new()),
    });

    let unix_listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("binding unix socket {} (another `grv serve` already running here?)", socket_path.display()))?;
    println!("GRAVITON daemon listening on unix:{}", socket_path.display());

    let unix_ctx = ctx.clone();
    let unix_task = tokio::spawn(async move {
        loop {
            match unix_listener.accept().await {
                Ok((stream, _)) => {
                    let ctx = unix_ctx.clone();
                    tokio::spawn(handle_conn(stream, ctx, false));
                }
                Err(e) => eprintln!("\x1b[2maccept error (unix): {e}\x1b[0m"),
            }
        }
    });

    let tcp_task = match &tcp {
        Some(addr) => {
            let listener = TcpListener::bind(addr).await.with_context(|| format!("binding tcp {addr}"))?;
            println!("GRAVITON daemon also listening on tcp:{addr}");
            println!(
                "\x1b[1;33mtoken required for every tcp request (params.token): {}\x1b[0m",
                tcp_token.as_deref().unwrap_or("")
            );
            let tcp_ctx = ctx.clone();
            Some(tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((stream, _)) => {
                            let ctx = tcp_ctx.clone();
                            tokio::spawn(handle_conn(stream, ctx, true));
                        }
                        Err(e) => eprintln!("\x1b[2maccept error (tcp): {e}\x1b[0m"),
                    }
                }
            }))
        }
        None => None,
    };

    println!("\x1b[2mrepo: {} -- Ctrl+C to stop, or send {{\"method\":\"shutdown\"}}\x1b[0m", root.display());
    tokio::signal::ctrl_c().await.ok();
    println!("\nstopping");
    unix_task.abort();
    if let Some(t) = tcp_task {
        t.abort();
    }
    let _ = std::fs::remove_file(&socket_path);
    Ok(())
}

/// A socket file left behind by a crashed daemon makes `bind` fail with
/// "address in use" even though nothing is listening — try connecting
/// first (a live daemon accepts), and only remove the file if that fails.
async fn remove_stale_socket(path: &PathBuf) {
    if !path.exists() {
        return;
    }
    if UnixStream::connect(path).await.is_err() {
        let _ = std::fs::remove_file(path);
    }
}

async fn handle_conn<S>(stream: S, ctx: Arc<DaemonCtx>, is_tcp: bool)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();
    loop {
        let line = match lines.next_line().await {
            Ok(Some(l)) if l.trim().is_empty() => continue,
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(_) => break,
        };

        let req: RpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = json!({"jsonrpc": "2.0", "id": Value::Null, "error": {"code": -32000, "message": format!("invalid JSON-RPC request: {e}")}});
                if writer.write_all(format!("{resp}\n").as_bytes()).await.is_err() || writer.flush().await.is_err() {
                    break;
                }
                continue;
            }
        };

        if is_tcp && !token_ok(&ctx, &req.params) {
            let resp = json!({"jsonrpc": "2.0", "id": req.id, "error": {"code": -32001, "message": "missing or wrong params.token"}});
            if writer.write_all(format!("{resp}\n").as_bytes()).await.is_err() || writer.flush().await.is_err() {
                break;
            }
            continue;
        }

        if req.method == "shutdown" {
            let resp = json!({"jsonrpc": "2.0", "id": req.id, "result": "ok"});
            let _ = writer.write_all(format!("{resp}\n").as_bytes()).await;
            let _ = writer.flush().await;
            println!("shutdown requested over RPC, exiting");
            let _ = std::fs::remove_file(&ctx.socket_path);
            std::process::exit(0);
        }

        let wants_stream = (req.method == "ask" || req.method == "review") && req.params.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
        if wants_stream {
            if handle_streaming(&ctx, &req, &mut writer).await.is_err() {
                break;
            }
            continue;
        }

        if req.method == "run_attach" {
            if handle_run_attach(&ctx, &req, &mut writer).await.is_err() {
                break;
            }
            continue;
        }

        let outcome = handle_method(&ctx, &req.method, &req.params).await;
        let resp = match outcome {
            Ok(result) => json!({"jsonrpc": "2.0", "id": req.id, "result": result}),
            Err(e) => json!({"jsonrpc": "2.0", "id": req.id, "error": {"code": -32000, "message": format!("{e:#}")}}),
        };
        if writer.write_all(format!("{resp}\n").as_bytes()).await.is_err() {
            break;
        }
        if writer.flush().await.is_err() {
            break;
        }
    }
}

/// Server-side visibility into what a non-streaming `ask`/`review` call is
/// doing (a streaming caller gets this as a `tool_call` notification
/// instead — see `handle_streaming`) -- printed to the daemon's own
/// terminal, never sent over the socket.
fn log_tool_call(name: &str, args: &Value) {
    println!("\x1b[2m  → {name}({args})\x1b[0m");
}

fn token_ok(ctx: &DaemonCtx, params: &Value) -> bool {
    match &ctx.tcp_token {
        None => true, // shouldn't happen (serve() always sets one once --tcp is on), fail open only if it's genuinely unset
        Some(expected) => params.get("token").and_then(|v| v.as_str()) == Some(expected.as_str()),
    }
}

/// Shared prep for `ask`: resolve the agent, build retrieval context
/// (identical to every other command's `build_context`), and return what
/// the model call needs -- used by both the streaming and non-streaming
/// paths so they can't drift on what "ask" actually means.
async fn prepare_ask(ctx: &DaemonCtx, params: &Value) -> Result<(&'static crate::agents::AgentSpec, String, String, String)> {
    let question = params.get("question").and_then(|v| v.as_str()).context("missing 'question'")?.to_string();
    let agent_key = params.get("agent").and_then(|v| v.as_str()).unwrap_or("architect").to_string();
    let spec = crate::resolve_agent(&agent_key)?;
    let files: Vec<PathBuf> = params
        .get("files")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(PathBuf::from)).collect())
        .unwrap_or_default();
    let conn = ctx.open_conn()?;
    let (groups, semantic_src) = crate::build_context_sync(&ctx.cfg, &ctx.root, &conn, &question, &files)?;
    drop(conn); // see semantic::EmbeddedChunk's doc comment: no Connection across the await below
    let context_text = crate::finish_context(&ctx.cfg, &question, groups, semantic_src).await?;
    let model = ctx.cfg.model_for_tier(spec.tier).to_string();
    let user_msg = format!("Question: {question}\n\nRetrieved context:\n{context_text}");
    Ok((spec, model, spec.system_prompt.to_string(), user_msg))
}

/// Shared prep for `review`. `Ok((spec, model, system, None))` means there
/// was nothing to diff -- both call sites turn that into the same
/// "nothing to review" answer instead of running a model call over
/// nothing.
fn prepare_review(ctx: &DaemonCtx, params: &Value) -> Result<(&'static crate::agents::AgentSpec, String, String, Option<String>)> {
    let range = params.get("range").and_then(|v| v.as_str()).map(str::to_string);
    let staged = params.get("staged").and_then(|v| v.as_bool()).unwrap_or(false);
    let agent_key = params.get("agent").and_then(|v| v.as_str()).unwrap_or("sentinel").to_string();
    let spec = crate::resolve_agent(&agent_key)?;
    let mut diff_args = vec!["diff".to_string()];
    match &range {
        Some(r) => diff_args.push(r.clone()),
        None if staged => diff_args.push("--staged".to_string()),
        None => diff_args.push("HEAD".to_string()),
    }
    let model = ctx.cfg.model_for_tier(spec.tier).to_string();
    let diff = crate::agentic::run_git(&ctx.root, &diff_args.iter().map(String::as_str).collect::<Vec<_>>())?;
    let user_msg = if diff.trim().is_empty() { None } else { Some(format!("Review this git diff:\n\n{diff}")) };
    Ok((spec, model, spec.system_prompt.to_string(), user_msg))
}

/// `ask`/`review` with `params.stream: true`: writes `{"jsonrpc":"2.0",
/// "method":"token","params":{"id":<request id>,"text":...}}` notifications
/// as the answer streams, and a `"tool_call"`-method notification each time
/// the model calls a tool along the way, before the final normal
/// `{"id":...,"result":{...}}` response line (identical shape to the
/// non-streaming reply, so a client that ignores notifications it doesn't
/// recognize still gets the full answer at the end either way).
async fn handle_streaming<W: AsyncWrite + Unpin>(ctx: &DaemonCtx, req: &RpcRequest, writer: &mut W) -> std::io::Result<()> {
    let prepared = if req.method == "ask" {
        prepare_ask(ctx, &req.params).await.map(|(spec, model, system, msg)| (spec, model, system, Some(msg)))
    } else {
        prepare_review(ctx, &req.params)
    };
    let (spec, model, system, user_msg) = match prepared {
        Ok(v) => v,
        Err(e) => return write_line(writer, &json!({"jsonrpc":"2.0","id":req.id,"error":{"code":-32000,"message":format!("{e:#}")}})).await,
    };
    let Some(user_msg) = user_msg else {
        return write_line(writer, &json!({"jsonrpc":"2.0","id":req.id,"result":{"agent":spec.key,"model":model,"answer":"(no diff -- nothing to review)"}})).await;
    };

    enum Event {
        Token(String),
        ToolCall(String, Value),
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let tx_tok = tx.clone();
    let on_token = move |tok: &str| {
        let _ = tx_tok.send(Event::Token(tok.to_string()));
    };
    let on_tool_call = move |name: &str, args: &Value| {
        let _ = tx.send(Event::ToolCall(name.to_string(), args.clone()));
    };

    let _permit = ctx.scheduler.acquire().await;
    let loop_fut = crate::agentic::run_read_only_loop_with(&ctx.client, &ctx.cfg, &ctx.root, &model, &system, &user_msg, on_token, on_tool_call);
    tokio::pin!(loop_fut);
    let result = loop {
        tokio::select! {
            event = rx.recv() => {
                let Some(event) = event else { continue };
                let notif = match event {
                    Event::Token(text) => json!({"jsonrpc":"2.0","method":"token","params":{"id":req.id,"text":text}}),
                    Event::ToolCall(name, args) => json!({"jsonrpc":"2.0","method":"tool_call","params":{"id":req.id,"name":name,"arguments":args}}),
                };
                write_line(writer, &notif).await?;
            }
            res = &mut loop_fut => break res,
        }
    };
    // Drain anything buffered after the loop finished but before we last polled the channel.
    while let Ok(event) = rx.try_recv() {
        let notif = match event {
            Event::Token(text) => json!({"jsonrpc":"2.0","method":"token","params":{"id":req.id,"text":text}}),
            Event::ToolCall(name, args) => json!({"jsonrpc":"2.0","method":"tool_call","params":{"id":req.id,"name":name,"arguments":args}}),
        };
        write_line(writer, &notif).await?;
    }

    match result {
        Ok(answer) => write_line(writer, &json!({"jsonrpc":"2.0","id":req.id,"result":{"agent":spec.key,"model":model,"answer":answer}})).await,
        Err(e) => write_line(writer, &json!({"jsonrpc":"2.0","id":req.id,"error":{"code":-32000,"message":format!("{e:#}")}})).await,
    }
}

async fn write_line<W: AsyncWrite + Unpin>(writer: &mut W, value: &Value) -> std::io::Result<()> {
    writer.write_all(format!("{value}\n").as_bytes()).await?;
    writer.flush().await
}

/// `run_attach {session_id}`: acks once (so the caller knows it's
/// subscribed), then streams that session's `RunEvent`s as `run_event`
/// notifications until a `Done` event or the connection drops. Multiple
/// connections can attach to the same session (each gets its own
/// `broadcast::Receiver`); a late attach only sees events from then on --
/// `run_status` is how a client finds out what already happened.
async fn handle_run_attach<W: AsyncWrite + Unpin>(ctx: &DaemonCtx, req: &RpcRequest, writer: &mut W) -> std::io::Result<()> {
    let Some(session_id) = req.params.get("session_id").and_then(|v| v.as_str()).map(str::to_string) else {
        return write_line(writer, &json!({"jsonrpc":"2.0","id":req.id,"error":{"code":-32000,"message":"missing 'session_id'"}})).await;
    };
    let handle = ctx.runs.lock().await.get(&session_id).cloned();
    let Some(handle) = handle else {
        return write_line(writer, &json!({"jsonrpc":"2.0","id":req.id,"error":{"code":-32000,"message":"no such run session"}})).await;
    };
    write_line(writer, &json!({"jsonrpc":"2.0","id":req.id,"result":"attached"})).await?;

    // Subscribe *before* checking the snapshot below, so an event that
    // fires in between is still seen live (not missed, not double-sent
    // for real -- the synthesized catch-ups below only fire for state
    // that was already true before this subscribe existed).
    let mut rx = handle.events_tx.subscribe();
    {
        let snapshot = handle.status.lock().unwrap().clone();
        // A confirmation was already pending before we attached -- its
        // `ConfirmRequest` broadcast went out before our receiver existed,
        // so replay it from the snapshot or this attach would wait
        // forever for an event that already happened.
        if let Some(action) = snapshot.pending_confirm {
            write_line(writer, &RunEvent::ConfirmRequest(action).to_notification(&session_id)).await?;
        }
        // Same idea for an ask_user question already waiting on an answer.
        if let Some((question, options, multi_select)) = snapshot.pending_choice {
            write_line(writer, &RunEvent::AskChoice(question, options, multi_select).to_notification(&session_id)).await?;
        }
        // Same idea if the run already finished before we attached.
        if !snapshot.running {
            let ok = snapshot.finished_ok.unwrap_or(false);
            let message = snapshot.finished_message.unwrap_or_default();
            write_line(writer, &RunEvent::Done { ok, message }.to_notification(&session_id)).await?;
            return Ok(());
        }
    }
    loop {
        match rx.recv().await {
            Ok(event) => {
                let done = matches!(event, RunEvent::Done { .. });
                write_line(writer, &event.to_notification(&session_id)).await?;
                if done {
                    return Ok(());
                }
            }
            // Fell behind the broadcast buffer under heavy output -- keep
            // going with whatever's next rather than disconnecting; a
            // `run_status` call fills in anything this attach missed.
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

async fn handle_method(ctx: &DaemonCtx, method: &str, params: &Value) -> Result<Value> {
    let get_str = |k: &str| params.get(k).and_then(|v| v.as_str()).map(str::to_string);
    let limit = |default: u64| params.get("limit").and_then(|v| v.as_u64()).unwrap_or(default) as usize;

    match method {
        "status" => {
            let index = match ctx.open_conn() {
                Ok(conn) => {
                    let files: i64 = conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0)).unwrap_or(0);
                    let symbols: i64 = conn.query_row("SELECT COUNT(*) FROM symbols", [], |r| r.get(0)).unwrap_or(0);
                    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM content_fts", [], |r| r.get(0)).unwrap_or(0);
                    let embedded: i64 = conn.query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0)).unwrap_or(0);
                    json!({"files": files, "symbols": symbols, "chunks": chunks, "embedded": embedded})
                }
                Err(_) => Value::Null,
            };
            let ollama_models = ctx.client.list_models().await.ok();
            Ok(json!({
                "root": ctx.root,
                "ollama_host": ctx.cfg.ollama_host,
                "model": ctx.cfg.model,
                "model_fast": ctx.cfg.model_fast,
                "model_deep": ctx.cfg.model_deep,
                "embed_model": ctx.cfg.embed_model,
                "index": index,
                "ollama_reachable": ollama_models.is_some(),
                "ollama_models": ollama_models,
                "scheduler_target": ctx.scheduler.current_target(),
            }))
        }
        "agents" => Ok(json!(agents::ALL_AGENTS
            .iter()
            .map(|a| json!({"key": a.key, "display": a.display, "tagline": a.tagline, "tier": format!("{:?}", a.tier)}))
            .collect::<Vec<_>>())),
        "search" => {
            let query = get_str("query").context("missing 'query'")?;
            let conn = ctx.open_conn()?;
            let hits = context::search_chunks(&conn, &query, limit(8))?;
            Ok(json!(hits.into_iter().map(|h| json!({"header": h.header, "body": h.body})).collect::<Vec<_>>()))
        }
        "symbol" => {
            let name = get_str("name").context("missing 'name'")?;
            let conn = ctx.open_conn()?;
            let hits = context::search_symbols(&conn, &ctx.root, &name, limit(5))?;
            Ok(json!(hits.into_iter().map(|h| json!({"header": h.header, "body": h.body})).collect::<Vec<_>>()))
        }
        "semantic_search" => {
            let query = get_str("query").context("missing 'query'")?;
            let model = ctx
                .cfg
                .embed_model
                .as_deref()
                .context("no embedding model configured -- `grv config --embed-model <tag>`")?;
            let conn = ctx.open_conn()?;
            if !semantic::has_embeddings(&conn) {
                anyhow::bail!("no embeddings computed yet -- run `grv embed` first");
            }
            // Load synchronously and drop the connection *before* the
            // await below -- see `semantic::EmbeddedChunk`'s doc comment
            // for why `&Connection` can never cross an await on this path
            // (this handler runs inside a `tokio::spawn`ed connection task).
            let chunks = semantic::load_embeddings(&conn, model)?;
            drop(conn);
            let _permit = ctx.scheduler.acquire().await;
            let hits = semantic::rank_by_query(&ctx.cfg.ollama_host, model, &query, chunks, limit(8)).await?;
            Ok(json!(hits
                .into_iter()
                .map(|h| json!({"path": h.path, "start_line": h.start_line, "end_line": h.end_line, "body": h.body, "score": h.score}))
                .collect::<Vec<_>>()))
        }
        "ask" => {
            let (spec, model, system, user_msg) = prepare_ask(ctx, params).await?;
            let _permit = ctx.scheduler.acquire().await;
            let answer = crate::agentic::run_read_only_loop_with(&ctx.client, &ctx.cfg, &ctx.root, &model, &system, &user_msg, |_| {}, log_tool_call).await?;
            Ok(json!({"agent": spec.key, "model": model, "answer": answer}))
        }
        "review" => {
            let (spec, model, system, user_msg) = prepare_review(ctx, params)?;
            let Some(user_msg) = user_msg else {
                return Ok(json!({"agent": spec.key, "model": model, "answer": "(no diff -- nothing to review)"}));
            };
            let _permit = ctx.scheduler.acquire().await;
            let answer = crate::agentic::run_read_only_loop_with(&ctx.client, &ctx.cfg, &ctx.root, &model, &system, &user_msg, |_| {}, log_tool_call).await?;
            Ok(json!({"agent": spec.key, "model": model, "answer": answer}))
        }
        "run_start" => {
            let task = get_str("task").unwrap_or_default();
            if task.trim().is_empty() {
                anyhow::bail!("missing 'task'");
            }
            let agent_key = get_str("agent").unwrap_or_else(|| "architect".to_string());
            let spec = crate::resolve_agent(&agent_key)?;
            let yolo = params.get("yolo").and_then(|v| v.as_bool()).unwrap_or(false);
            let browser = params.get("browser").and_then(|v| v.as_bool()).unwrap_or(false);
            let files: Vec<PathBuf> = params
                .get("files")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(PathBuf::from)).collect())
                .unwrap_or_default();

            let session_id = generate_token()[..12].to_string();
            let (events_tx, _) = broadcast::channel::<RunEvent>(1024);
            let (confirm_tx, confirm_rx) = mpsc::unbounded_channel::<Decision>();
            let (choice_tx, choice_rx) = mpsc::unbounded_channel::<Vec<String>>();
            let status = Arc::new(StdMutex::new(RunStatusSnapshot { running: true, ..Default::default() }));
            let io: Arc<dyn RunIo> = Arc::new(ChannelIo {
                events_tx: events_tx.clone(),
                confirm_rx: AsyncMutex::new(confirm_rx),
                choice_rx: AsyncMutex::new(choice_rx),
                status: status.clone(),
            });

            ctx.runs
                .lock()
                .await
                .insert(session_id.clone(), RunHandle { events_tx: events_tx.clone(), confirm_tx, choice_tx, status: status.clone() });

            let cfg_owned = ctx.cfg.clone();
            let root_owned = ctx.root.clone();
            tokio::spawn(async move {
                // Same build_context_sync/finish_context split every other
                // model-calling method uses -- no `&Connection` can appear
                // in an async fn's signature on a path that's `tokio::spawn`ed
                // (see semantic::EmbeddedChunk's doc comment). Tolerant of a
                // missing/empty index, matching `grv run`'s own CLI behavior
                // (a task with no useful retrieval hit still runs).
                let context_text = graviton_core::db_path_for(&root_owned, &cfg_owned.index_dir)
                    .and_then(|p| graviton_core::open_db(&p))
                    .ok()
                    .and_then(|conn| crate::build_context_sync(&cfg_owned, &root_owned, &conn, &task, &files).ok());
                let context_text = match context_text {
                    Some((groups, semantic_src)) => crate::finish_context(&cfg_owned, &task, groups, semantic_src).await.ok(),
                    None => None,
                };

                let loop_cfg = crate::agentic::AgentLoopConfig { auto_approve: yolo, enable_browser: browser };
                let result = crate::agentic::run(&cfg_owned, &root_owned, spec, &task, context_text, loop_cfg, None, io).await;
                let (ok, message) = match &result {
                    Ok(()) => (true, "done".to_string()),
                    Err(e) => (false, format!("{e:#}")),
                };
                {
                    let mut st = status.lock().unwrap();
                    st.running = false;
                    st.finished_ok = Some(ok);
                    st.finished_message = Some(message.clone());
                }
                let _ = events_tx.send(RunEvent::Done { ok, message });
            });

            Ok(json!({"session_id": session_id}))
        }
        "run_confirm" => {
            let session_id = get_str("session_id").context("missing 'session_id'")?;
            let decision_str = get_str("decision").context("missing 'decision' (\"yes\"/\"no\", or any other text = redirect)")?;
            let decision = match decision_str.as_str() {
                "yes" | "y" | "Yes" | "Y" => Decision::Allow,
                "no" | "n" | "No" | "N" | "" => Decision::Deny,
                other => Decision::Redirect(other.to_string()),
            };
            let runs = ctx.runs.lock().await;
            let handle = runs.get(&session_id).context("no such run session (finished or never started)")?;
            handle
                .confirm_tx
                .send(decision)
                .map_err(|_| anyhow::anyhow!("this run session isn't waiting for a confirmation right now"))?;
            Ok(json!({"ok": true}))
        }
        "run_answer_choice" => {
            let session_id = get_str("session_id").context("missing 'session_id'")?;
            let selected: Vec<String> = params
                .get("selected")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .context("missing 'selected' (array of chosen option strings)")?;
            let runs = ctx.runs.lock().await;
            let handle = runs.get(&session_id).context("no such run session (finished or never started)")?;
            handle
                .choice_tx
                .send(selected)
                .map_err(|_| anyhow::anyhow!("this run session isn't waiting for an ask_user answer right now"))?;
            Ok(json!({"ok": true}))
        }
        "run_status" => {
            let session_id = get_str("session_id").context("missing 'session_id'")?;
            let runs = ctx.runs.lock().await;
            let handle = runs.get(&session_id).context("no such run session")?;
            let snapshot = handle.status.lock().unwrap().clone();
            Ok(json!({
                "session_id": session_id,
                "running": snapshot.running,
                "pending_confirm": snapshot.pending_confirm,
                "pending_choice": snapshot.pending_choice.map(|(q, o, m)| json!({"question": q, "options": o, "multi_select": m})),
                "checkpoint_id": snapshot.checkpoint_id,
                "finished_ok": snapshot.finished_ok,
                "finished_message": snapshot.finished_message,
            }))
        }
        other => anyhow::bail!("unknown method '{other}' -- see ARCHITECTURE.md for the method list"),
    }
}

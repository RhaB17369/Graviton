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

use crate::{agents, context, resources, semantic};
use anyhow::{Context, Result};
use graviton_core::Config;
use graviton_llm::{ChatMessage, OllamaClient};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener, UnixStream};

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

pub async fn serve(cfg: &Config, root: PathBuf, socket: Option<PathBuf>, tcp: Option<String>) -> Result<()> {
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

    let ctx = Arc::new(DaemonCtx { cfg: cfg.clone(), root: root.clone(), client, scheduler, socket_path: socket_path.clone() });

    let unix_listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("binding unix socket {} (another `grv serve` already running here?)", socket_path.display()))?;
    println!("GRAVITON daemon listening on unix:{}", socket_path.display());

    let unix_ctx = ctx.clone();
    let unix_task = tokio::spawn(async move {
        loop {
            match unix_listener.accept().await {
                Ok((stream, _)) => {
                    let ctx = unix_ctx.clone();
                    tokio::spawn(handle_conn(stream, ctx));
                }
                Err(e) => eprintln!("\x1b[2maccept error (unix): {e}\x1b[0m"),
            }
        }
    });

    let tcp_task = match tcp {
        Some(addr) => {
            let listener = TcpListener::bind(&addr).await.with_context(|| format!("binding tcp {addr}"))?;
            println!("GRAVITON daemon also listening on tcp:{addr}");
            let tcp_ctx = ctx.clone();
            Some(tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((stream, _)) => {
                            let ctx = tcp_ctx.clone();
                            tokio::spawn(handle_conn(stream, ctx));
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

async fn handle_conn<S>(stream: S, ctx: Arc<DaemonCtx>)
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

        let (id, outcome) = match serde_json::from_str::<RpcRequest>(&line) {
            Ok(req) => {
                if req.method == "shutdown" {
                    let resp = json!({"jsonrpc": "2.0", "id": req.id, "result": "ok"});
                    let _ = writer.write_all(format!("{resp}\n").as_bytes()).await;
                    let _ = writer.flush().await;
                    println!("shutdown requested over RPC, exiting");
                    let _ = std::fs::remove_file(&ctx.socket_path);
                    std::process::exit(0);
                }
                let outcome = handle_method(&ctx, &req.method, &req.params).await;
                (req.id, outcome)
            }
            Err(e) => (Value::Null, Err(anyhow::anyhow!("invalid JSON-RPC request: {e}"))),
        };

        let resp = match outcome {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32000, "message": format!("{e:#}")}}),
        };
        if writer.write_all(format!("{resp}\n").as_bytes()).await.is_err() {
            break;
        }
        if writer.flush().await.is_err() {
            break;
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
            let question = get_str("question").context("missing 'question'")?;
            let agent_key = get_str("agent").unwrap_or_else(|| "architect".to_string());
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
            let model = ctx.cfg.model_for_tier(spec.tier);
            let messages = vec![
                ChatMessage::system(spec.system_prompt),
                ChatMessage::user(format!("Question: {question}\n\nRetrieved context:\n{context_text}")),
            ];
            let _permit = ctx.scheduler.acquire().await;
            let result = ctx.client.chat_stream(model, &messages, ctx.cfg.num_ctx, &[], |_| {}).await?;
            Ok(json!({"agent": spec.key, "model": model, "answer": result.content}))
        }
        "review" => {
            let range = get_str("range");
            let staged = params.get("staged").and_then(|v| v.as_bool()).unwrap_or(false);
            let agent_key = get_str("agent").unwrap_or_else(|| "sentinel".to_string());
            let spec = crate::resolve_agent(&agent_key)?;
            let mut diff_args = vec!["diff".to_string()];
            match &range {
                Some(r) => diff_args.push(r.clone()),
                None if staged => diff_args.push("--staged".to_string()),
                None => diff_args.push("HEAD".to_string()),
            }
            let model = ctx.cfg.model_for_tier(spec.tier);
            let diff = crate::agentic::run_git(&ctx.root, &diff_args.iter().map(String::as_str).collect::<Vec<_>>())?;
            if diff.trim().is_empty() {
                return Ok(json!({"agent": spec.key, "model": model, "answer": "(no diff -- nothing to review)"}));
            }
            let messages = vec![
                ChatMessage::system(spec.system_prompt),
                ChatMessage::user(format!("Review this git diff:\n\n{diff}")),
            ];
            let _permit = ctx.scheduler.acquire().await;
            let result = ctx.client.chat_stream(model, &messages, ctx.cfg.num_ctx, &[], |_| {}).await?;
            Ok(json!({"agent": spec.key, "model": model, "answer": result.content}))
        }
        other => anyhow::bail!("unknown method '{other}' -- see ARCHITECTURE.md for the method list"),
    }
}

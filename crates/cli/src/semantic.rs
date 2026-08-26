//! Optional local semantic search: embed indexed code chunks with an
//! Ollama embedding model (`nomic-embed-text`, `all-minilm`, ...) and rank
//! by cosine similarity, instead of (well: in addition to — see
//! `main::build_context`) lexical FTS token matching.
//!
//! Purely additive. Nothing here changes behavior until a user sets
//! `grv config --embed-model <tag>` (a model they've pulled) and runs
//! `grv embed`; every other command works exactly as it did before this
//! module existed if that's never done.

use anyhow::{Context, Result};
use graviton_llm::OllamaClient;
use rusqlite::Connection;
use std::sync::Arc;

pub struct SemanticHit {
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub body: String,
    pub score: f32,
}

/// One stored embedding, fully materialized (owned, `Send`) so it can cross
/// an `.await` — unlike `&rusqlite::Connection`, which can't: `Connection`
/// isn't `Sync`, and (per rustc, regardless of where inside the function
/// body it's actually touched) an `async fn` with `&Connection` anywhere in
/// its signature has its returned future's `Send`-ness poisoned by that
/// alone. `search_code`/`semantic_search` run from inside `mission`'s
/// `Box::pin(... + Send)` leaves and `grv serve`'s `tokio::spawn`ed
/// connections, so every async fn in that path (`rank_by_query` below,
/// `finish_context`/`finish_search_outcome` elsewhere) is written to never
/// take `&Connection` — all DB reads happen in a plain sync fn first
/// (`load_embeddings`), and only owned data crosses into async code.
pub struct EmbeddedChunk {
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub body: String,
    pub vector: Vec<f32>,
}

#[derive(Default, Debug)]
pub struct EmbedStats {
    pub embedded: usize,
    pub skipped_existing: usize,
    pub failed: usize,
}

/// Cheap existence check so callers (e.g. `build_context`) can skip
/// semantic retrieval entirely — no wasted embedding call for the query
/// itself — when `grv embed` has never been run.
pub fn has_embeddings(conn: &Connection) -> bool {
    conn.query_row("SELECT EXISTS(SELECT 1 FROM embeddings LIMIT 1)", [], |r| r.get::<_, i64>(0))
        .map(|n| n != 0)
        .unwrap_or(false)
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return -1.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return -1.0;
    }
    dot / (na * nb)
}

/// Embed every 'chunk' row in `content_fts` that doesn't already have a
/// current-model embedding (or, with `force`, every chunk unconditionally),
/// storing vectors as little-endian f32 BLOBs. Requests run concurrently,
/// bounded by CPU thread count (embedding a short text is cheap per-call
/// but still serializes somewhat against one resident model in Ollama, so
/// this is a real-but-modest pipeline rather than either one-at-a-time or
/// an unbounded flood).
pub async fn embed_index(ollama_host: &str, conn: &mut Connection, model: &str, force: bool) -> Result<EmbedStats> {
    let rows: Vec<(i64, String)> = {
        let mut stmt = conn.prepare("SELECT rowid, body FROM content_fts WHERE kind = 'chunk'")?;
        let out: Vec<(i64, String)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.filter_map(|r| r.ok()).collect();
        out
    };

    let existing: std::collections::HashSet<i64> = if force {
        Default::default()
    } else {
        let mut stmt = conn.prepare("SELECT chunk_id FROM embeddings WHERE model = ?1")?;
        let out: std::collections::HashSet<i64> = stmt.query_map([model], |r| r.get(0))?.filter_map(|r| r.ok()).collect();
        out
    };

    let skipped_existing = rows.iter().filter(|(id, _)| existing.contains(id)).count();
    let todo: Vec<(i64, String)> = rows.into_iter().filter(|(id, _)| !existing.contains(id)).collect();
    let total = todo.len();
    if total == 0 {
        return Ok(EmbedStats { embedded: 0, skipped_existing, failed: 0 });
    }

    let client = OllamaClient::new(ollama_host);
    let cap = crate::resources::detect();
    // Embedding calls are lighter than generation but still contend for
    // the one resident embedding model; a handful in flight pipelines
    // network/inference latency without pretending this scales like
    // independent CPU-bound work.
    let concurrency = cap.cpu_threads.clamp(2, 8);
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    println!(
        "\x1b[2membedding {total} chunk(s) with '{model}', ~{concurrency} concurrent request(s) \
         ({} chunk(s) already embedded, skipped)\x1b[0m",
        skipped_existing
    );

    let mut set = tokio::task::JoinSet::new();
    for (id, body) in todo {
        let client = client.clone();
        let model_owned = model.to_string();
        let sem = sem.clone();
        set.spawn(async move {
            let _permit = sem.acquire_owned().await.expect("semaphore is never closed");
            let result = client.embed(&model_owned, &body).await;
            (id, result)
        });
    }

    let mut stats = EmbedStats { embedded: 0, skipped_existing, failed: 0 };
    let mut done = 0usize;
    while let Some(joined) = set.join_next().await {
        let (id, result) = joined.context("embedding task panicked")?;
        done += 1;
        match result {
            Ok(vec) => {
                conn.execute(
                    "INSERT INTO embeddings (chunk_id, model, dims, vector) VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT(chunk_id) DO UPDATE SET model = excluded.model, dims = excluded.dims, vector = excluded.vector",
                    rusqlite::params![id, model, vec.len() as i64, f32_to_bytes(&vec)],
                )?;
                stats.embedded += 1;
            }
            Err(e) => {
                stats.failed += 1;
                eprintln!("\x1b[2m  chunk {id}: embedding failed: {e:#}\x1b[0m");
            }
        }
        if done % 25 == 0 || done == total {
            println!("\x1b[2m  {done}/{total}\x1b[0m");
        }
    }
    Ok(stats)
}

/// Load every stored embedding for `model`, fully materialized. Plain sync
/// fn (see `EmbeddedChunk`'s doc comment for why this matters) — call this
/// first, then hand the result to `rank_by_query`.
pub fn load_embeddings(conn: &Connection, model: &str) -> Result<Vec<EmbeddedChunk>> {
    let mut stmt = conn.prepare(
        "SELECT c.path, c.start_line, c.end_line, c.body, e.vector \
         FROM embeddings e JOIN content_fts c ON c.rowid = e.chunk_id \
         WHERE e.model = ?1",
    )?;
    let mapped = stmt.query_map([model], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?, r.get::<_, Vec<u8>>(4)?))
    })?;
    Ok(mapped
        .filter_map(|r| r.ok())
        .map(|(path, start_line, end_line, body, raw)| EmbeddedChunk { path, start_line, end_line, body, vector: bytes_to_f32(&raw) })
        .collect())
}

/// Embed `query` and rank the given (already-loaded) chunks by cosine
/// similarity. O(n) over however many chunks are embedded — no ANN index —
/// which is plenty fast (single-digit ms) up to the tens-of-thousands-of-
/// chunks range a local single-repo index actually reaches. No `Connection`
/// anywhere in this signature — see `EmbeddedChunk`'s doc comment.
pub async fn rank_by_query(ollama_host: &str, model: &str, query: &str, chunks: Vec<EmbeddedChunk>, limit: usize) -> Result<Vec<SemanticHit>> {
    let client = OllamaClient::new(ollama_host);
    let qvec = client.embed(model, query).await.context("embedding query")?;

    let mut hits: Vec<SemanticHit> = chunks
        .into_iter()
        .map(|c| {
            let score = cosine(&qvec, &c.vector);
            SemanticHit { path: c.path, start_line: c.start_line, end_line: c.end_line, body: c.body, score }
        })
        .collect();
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(limit);
    Ok(hits)
}

/// Convenience wrapper for callers with no `Send` constraint on the
/// resulting future (i.e. not spawned via `tokio::spawn`/`Box::pin(... +
/// Send)`) — `grv search --semantic` is a plain top-level `.await`, not a
/// spawned task, so the `&Connection` in this signature is harmless here.
pub async fn search(ollama_host: &str, conn: &Connection, model: &str, query: &str, limit: usize) -> Result<Vec<SemanticHit>> {
    let chunks = load_embeddings(conn, model)?;
    rank_by_query(ollama_host, model, query, chunks, limit).await
}

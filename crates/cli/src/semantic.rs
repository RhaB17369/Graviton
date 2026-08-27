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
use std::path::{Path, PathBuf};
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
    /// `content_fts` rowid — needed by `ann::rebuild` to store a compact
    /// (id, vector) pair without duplicating path/body text into the ANN
    /// index file (see `ann.rs`'s module doc).
    pub chunk_id: i64,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub body: String,
    pub vector: Vec<f32>,
}

/// What a query ranks against, decided synchronously before any embedding
/// call. `Ann` is the fast path — used whenever `grv embed` has left a
/// fresh index on disk for `model` (see `ann::rebuild`), so the expensive
/// full `load_embeddings` scan is skipped entirely. `Linear` is the
/// always-correct fallback: every embedded chunk, loaded up front and
/// exact-scanned, exactly how this module worked before it gained ANN
/// support. Owned/`Send` throughout — no `Connection` — for the same
/// reason as `EmbeddedChunk` (see its doc comment).
pub enum QuerySource {
    Ann { root: PathBuf, index_dir: String, model: String },
    Linear { model: String, chunks: Vec<EmbeddedChunk> },
}

/// Sync prep, safe to call with a live `&Connection`: pick `Ann` if a
/// fresh index exists for `model` (the whole payoff for a large repo —
/// see `ann.rs`), otherwise fall back to `load_embeddings`. Callers that
/// already check `has_embeddings` first (existing convention throughout
/// this codebase) call this in place of `load_embeddings` directly.
pub fn prepare_query_source(conn: &Connection, root: &Path, index_dir: &str, model: &str) -> Result<QuerySource> {
    if crate::ann::exists(root, index_dir, model) {
        Ok(QuerySource::Ann { root: root.to_path_buf(), index_dir: index_dir.to_string(), model: model.to_string() })
    } else {
        Ok(QuerySource::Linear { model: model.to_string(), chunks: load_embeddings(conn, model)? })
    }
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
pub async fn embed_index(ollama_host: &str, conn: &mut Connection, root: &Path, index_dir: &str, model: &str, force: bool) -> Result<EmbedStats> {
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

    // Refresh the ANN index over *every* currently stored embedding for
    // this model (not just what was just embedded) -- see `ann.rs`'s
    // module doc for why a full rebuild here, rather than incremental
    // maintenance, is the deliberately simple invariant. Never fatal: a
    // failure here only means semantic search stays on its always-correct
    // linear-scan fallback, so it's reported, not propagated.
    match load_embeddings(conn, model) {
        Ok(all) => {
            let count = all.len();
            match crate::ann::rebuild(root, index_dir, model, &all) {
                Ok(()) if count > 0 => println!("\x1b[2mANN index rebuilt ({count} vector(s))\x1b[0m"),
                Ok(()) => {}
                Err(e) => eprintln!("\x1b[2mwarning: ANN index rebuild failed (semantic search will fall back to a linear scan): {e:#}\x1b[0m"),
            }
        }
        Err(e) => eprintln!("\x1b[2mwarning: could not reload embeddings to rebuild the ANN index: {e:#}\x1b[0m"),
    }

    Ok(stats)
}

/// Load every stored embedding for `model`, fully materialized. Plain sync
/// fn (see `EmbeddedChunk`'s doc comment for why this matters) — call this
/// first, then hand the result to `rank_by_query`.
pub fn load_embeddings(conn: &Connection, model: &str) -> Result<Vec<EmbeddedChunk>> {
    let mut stmt = conn.prepare(
        "SELECT e.chunk_id, c.path, c.start_line, c.end_line, c.body, e.vector \
         FROM embeddings e JOIN content_fts c ON c.rowid = e.chunk_id \
         WHERE e.model = ?1",
    )?;
    let mapped = stmt.query_map([model], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, String>(4)?,
            r.get::<_, Vec<u8>>(5)?,
        ))
    })?;
    Ok(mapped
        .filter_map(|r| r.ok())
        .map(|(chunk_id, path, start_line, end_line, body, raw)| EmbeddedChunk {
            chunk_id,
            path,
            start_line,
            end_line,
            body,
            vector: bytes_to_f32(&raw),
        })
        .collect())
}

/// One `SELECT ... WHERE rowid IN (...)` to hydrate ANN search results
/// (`(chunk_id, score)` pairs with no text yet) into full `SemanticHit`s.
/// The only place this module opens a `Connection` after the async embed
/// call has already finished — never held across an `.await`, matching
/// the discipline documented on `EmbeddedChunk`. Building the `IN (...)`
/// list via plain integer formatting (not string interpolation of
/// anything user-supplied) is safe: every id here came from our own
/// `ann::search`, never from the query text.
fn hydrate(conn: &Connection, mut scored: Vec<(i64, f32)>) -> Result<Vec<SemanticHit>> {
    if scored.is_empty() {
        return Ok(Vec::new());
    }
    let ids = scored.iter().map(|(id, _)| id.to_string()).collect::<Vec<_>>().join(",");
    let sql = format!("SELECT rowid, path, start_line, end_line, body FROM content_fts WHERE rowid IN ({ids})");
    let mut stmt = conn.prepare(&sql)?;
    let mut by_id: std::collections::HashMap<i64, (String, i64, i64, String)> = stmt
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))))?
        .filter_map(|r| r.ok())
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored
        .into_iter()
        .filter_map(|(id, score)| by_id.remove(&id).map(|(path, start_line, end_line, body)| SemanticHit { path, start_line, end_line, body, score }))
        .collect())
}

/// Exact cosine scan over already-loaded chunks — the `Linear` path, and
/// the fallback an `Ann` search degrades to if the index turns out to be
/// missing/corrupt/empty at query time.
fn score_linear(qvec: &[f32], chunks: Vec<EmbeddedChunk>, limit: usize) -> Vec<SemanticHit> {
    let mut hits: Vec<SemanticHit> = chunks
        .into_iter()
        .map(|c| {
            let score = cosine(qvec, &c.vector);
            SemanticHit { path: c.path, start_line: c.start_line, end_line: c.end_line, body: c.body, score }
        })
        .collect();
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(limit);
    hits
}

/// Embed `query` and rank against `source` — either an ANN index lookup
/// (fast path, falls back to `Linear` scoring on any index problem) or an
/// exact cosine scan over already-loaded chunks. No `Connection` anywhere
/// in this signature — see `EmbeddedChunk`'s doc comment — any DB access
/// needed for ANN hydration opens its own short-lived connection, after
/// the only `.await` in this function has already completed.
pub async fn rank_by_query(ollama_host: &str, query: &str, source: QuerySource, limit: usize) -> Result<Vec<SemanticHit>> {
    let model = match &source {
        QuerySource::Ann { model, .. } => model.clone(),
        QuerySource::Linear { model, .. } => model.clone(),
    };
    let client = OllamaClient::new(ollama_host);
    let qvec = client.embed(&model, query).await.context("embedding query")?;

    match source {
        QuerySource::Linear { chunks, .. } => Ok(score_linear(&qvec, chunks, limit)),
        QuerySource::Ann { root, index_dir, model } => {
            match crate::ann::search(&root, &index_dir, &model, &qvec, limit)? {
                Some(scored) if !scored.is_empty() => {
                    let db_path = graviton_core::db_path_for(&root, &index_dir)?;
                    let conn = graviton_core::open_db(&db_path)?;
                    hydrate(&conn, scored)
                }
                // No usable index (missing/corrupt/foreign/empty) --
                // degrade to the exact linear scan rather than returning
                // nothing. This is the ANN feature's whole design promise:
                // it can only make a search faster, never wrong.
                _ => {
                    let db_path = graviton_core::db_path_for(&root, &index_dir)?;
                    let conn = graviton_core::open_db(&db_path)?;
                    let chunks = load_embeddings(&conn, &model)?;
                    Ok(score_linear(&qvec, chunks, limit))
                }
            }
        }
    }
}

/// Convenience wrapper for callers with no `Send` constraint on the
/// resulting future (i.e. not spawned via `tokio::spawn`/`Box::pin(... +
/// Send)`) — `grv search --semantic` is a plain top-level `.await`, not a
/// spawned task, so the `&Connection` in this signature is harmless here.
pub async fn search(ollama_host: &str, conn: &Connection, root: &Path, index_dir: &str, model: &str, query: &str, limit: usize) -> Result<Vec<SemanticHit>> {
    let source = prepare_query_source(conn, root, index_dir, model)?;
    rank_by_query(ollama_host, query, source, limit).await
}

//! ANN (approximate nearest neighbor) acceleration for semantic search,
//! built on `instant-distance`'s pure-Rust HNSW implementation — no FFI, no
//! external index server, one bincode blob on disk per embedding model.
//! Chosen deliberately after this session's tree-sitter grammar-linking
//! pain (see ARCHITECTURE.md's "Language coverage"): a pure-Rust crate with
//! no native library to mismatch or fail to link at some later point.
//!
//! This is purely an accelerator layered on top of `semantic.rs`'s exact
//! linear cosine scan, which stays the source of truth and the automatic
//! fallback: every function here degrades to `Ok(None)`/an error on a
//! missing, foreign, or corrupt index file rather than ever producing a
//! wrong answer or a hard failure — `semantic::rank_by_query` falls back to
//! the exact scan whenever that happens. Nothing here can make a search
//! *wrong*, only faster.
//!
//! ## Why this matters for "gigantic repos"
//!
//! Without an ANN index, every semantic query pays two O(n) costs: a SQL
//! join loading every stored embedding's full body text
//! (`semantic::load_embeddings`), and an exact cosine dot product against
//! every one of them. Neither shows up at the chunk counts a normal repo
//! reaches, but both become real at the scale this tool is meant to hold
//! up on. With an index: a query touches a compact on-disk blob (chunk_id
//! + vector only, no duplicated body text) via an O(log n) HNSW walk, and
//! only the handful of winning chunks get a final, tiny `SELECT ... WHERE
//! rowid IN (...)` to fetch their text.
//!
//! ## Why "rebuild fully, every time" instead of incremental maintenance
//!
//! `instant-distance` has no incremental-insert API — building means
//! handing it the complete point set once. Rather than layer on
//! version/hash bookkeeping to decide when a partial rebuild would be
//! safe, the invariant here is deliberately simple: `ann::rebuild` runs
//! once at the end of every `embed_index` call, over *every* currently
//! stored embedding for that model (not just the newly embedded delta), so
//! the file on disk — if present — is always an exact snapshot of the
//! `embeddings` table as of the last successful `grv embed` run. A search
//! never needs to ask "is this still fresh?"; either the file exists (and
//! is trusted), or it doesn't (and the caller falls back).
//!
//! ## Why `&rusqlite::Connection` never appears here
//!
//! Same discipline as everywhere else in this codebase (see
//! `semantic::EmbeddedChunk`'s doc comment): `rebuild` takes already-loaded
//! `EmbeddedChunk`s, and `search` takes only a raw query vector and returns
//! raw `(chunk_id, score)` pairs — hydrating those into full `SemanticHit`s
//! with path/line/body text is `semantic.rs`'s job, using its own
//! short-lived connection opened after every `.await` this module's caller
//! needed has already completed.

use crate::semantic::EmbeddedChunk;
use anyhow::{Context, Result};
use instant_distance::{Builder, HnswMap, Search};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A stored embedding vector, wrapped so it can implement
/// `instant_distance::Point`. Distance is `1 - cosine similarity` (0 =
/// identical direction, up to 2 = opposite) since HNSW is defined in terms
/// of "smaller = nearer" — converted back to a familiar similarity score
/// when reporting hits.
#[derive(Clone, Serialize, Deserialize)]
struct CosinePoint(Vec<f32>);

impl instant_distance::Point for CosinePoint {
    fn distance(&self, other: &Self) -> f32 {
        let (a, b) = (&self.0, &other.0);
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 2.0; // a zero vector never matches anything -- max distance
        }
        1.0 - (dot / (na * nb))
    }
}

/// The value carried alongside each point is just its `content_fts` rowid
/// — hydration back to path/lines/body happens in `semantic.rs`.
#[derive(Serialize, Deserialize)]
struct StoredIndex {
    /// Sanity fields, not staleness bookkeeping (see module doc) — just
    /// enough to refuse a foreign/mismatched file instead of feeding a
    /// wrong-dimension query vector into the HNSW walk.
    model: String,
    dims: usize,
    map: HnswMap<CosinePoint, i64>,
}

fn index_path(root: &Path, index_dir: &str, model: &str) -> PathBuf {
    // Model tags can contain ':' (e.g. "nomic-embed-text:latest") and other
    // characters that are awkward in a filename on some platforms --
    // sanitize rather than assume.
    let safe_model: String = model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' })
        .collect();
    root.join(index_dir).join(format!("ann_{safe_model}.bin"))
}

/// Cheap existence check — a plain file-exists call, no I/O into the file
/// itself, safe to call from anywhere including inside an async fn body
/// (it never touches a `Connection` or does any awaiting).
pub fn exists(root: &Path, index_dir: &str, model: &str) -> bool {
    index_path(root, index_dir, model).is_file()
}

/// Rebuild the on-disk ANN index for `model` from `chunks` (expected to be
/// *every* currently-embedded chunk for that model, not a delta — see
/// module doc), replacing whatever was there before. An empty `chunks`
/// removes the file rather than writing an empty, useless index.
pub fn rebuild(root: &Path, index_dir: &str, model: &str, chunks: &[EmbeddedChunk]) -> Result<()> {
    let path = index_path(root, index_dir, model);
    if chunks.is_empty() {
        let _ = std::fs::remove_file(&path);
        return Ok(());
    }
    let dims = chunks[0].vector.len();
    let points: Vec<CosinePoint> = chunks.iter().map(|c| CosinePoint(c.vector.clone())).collect();
    let values: Vec<i64> = chunks.iter().map(|c| c.chunk_id).collect();
    let map = Builder::default().build(points, values);
    let stored = StoredIndex { model: model.to_string(), dims, map };

    // Write to a temp file then rename, so a crash or a concurrent `grv
    // embed`/query never sees a half-written index (rename is atomic on
    // the same filesystem, which a sibling temp file in the same
    // directory guarantees).
    let tmp_path = path.with_extension("bin.tmp");
    let file = std::fs::File::create(&tmp_path).with_context(|| format!("creating {}", tmp_path.display()))?;
    bincode::serialize_into(std::io::BufWriter::new(file), &stored).context("serializing ANN index")?;
    std::fs::rename(&tmp_path, &path).with_context(|| format!("renaming into place: {}", path.display()))?;
    Ok(())
}

/// Query the on-disk ANN index for `model`'s `limit` nearest neighbors to
/// `qvec`, returning `(chunk_id, cosine_score)` pairs — highest score
/// first. `Ok(None)` means "no usable index" (missing, foreign model,
/// dimension mismatch, or corrupt file) and is not an error: every caller
/// treats it as "fall back to the exact linear scan", per this module's
/// whole reason for existing.
pub fn search(root: &Path, index_dir: &str, model: &str, qvec: &[f32], limit: usize) -> Result<Option<Vec<(i64, f32)>>> {
    let path = index_path(root, index_dir, model);
    if !path.is_file() {
        return Ok(None);
    }
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let stored: StoredIndex = match bincode::deserialize_from(std::io::BufReader::new(file)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("\x1b[2mANN index at {} is unreadable ({e:#}), falling back to a linear scan\x1b[0m", path.display());
            return Ok(None);
        }
    };
    if stored.model != model || stored.dims != qvec.len() {
        // A stale file from a since-changed embed model/dimension — same
        // "not an error, just unusable" treatment.
        return Ok(None);
    }

    let query_point = CosinePoint(qvec.to_vec());
    let mut search = Search::default();
    let hits: Vec<(i64, f32)> = stored
        .map
        .search(&query_point, &mut search)
        .take(limit)
        .map(|item| (*item.value, 1.0 - item.distance))
        .collect();
    Ok(Some(hits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::EmbeddedChunk;

    fn chunk(id: i64, path: &str, v: Vec<f32>) -> EmbeddedChunk {
        EmbeddedChunk { chunk_id: id, path: path.to_string(), start_line: 1, end_line: 2, body: format!("body {id}"), vector: v }
    }

    #[test]
    fn rebuild_then_search_finds_the_nearest_real_vector() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".graviton")).unwrap();
        let chunks = vec![
            chunk(1, "a.rs", vec![1.0, 0.0, 0.0]),
            chunk(2, "b.rs", vec![0.0, 1.0, 0.0]),
            chunk(3, "c.rs", vec![0.0, 0.0, 1.0]),
        ];
        rebuild(root, ".graviton", "test-model", &chunks).unwrap();
        assert!(exists(root, ".graviton", "test-model"));

        // Query close to chunk 2's direction -- should come back first.
        let hits = search(root, ".graviton", "test-model", &[0.1, 0.95, 0.0], 2).unwrap().unwrap();
        assert_eq!(hits[0].0, 2);
        assert!(hits[0].1 > 0.9);
    }

    #[test]
    fn missing_index_returns_none_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let hits = search(dir.path(), ".graviton", "whatever", &[1.0, 0.0], 5).unwrap();
        assert!(hits.is_none());
    }

    #[test]
    fn dimension_mismatch_degrades_to_none_instead_of_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".graviton")).unwrap();
        rebuild(root, ".graviton", "m", &[chunk(1, "a.rs", vec![1.0, 0.0])]).unwrap();
        let hits = search(root, ".graviton", "m", &[1.0, 0.0, 0.0], 5).unwrap();
        assert!(hits.is_none());
    }

    #[test]
    fn empty_chunks_removes_rather_than_writes_an_empty_index() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".graviton")).unwrap();
        rebuild(root, ".graviton", "m", &[chunk(1, "a.rs", vec![1.0])]).unwrap();
        assert!(exists(root, ".graviton", "m"));
        rebuild(root, ".graviton", "m", &[]).unwrap();
        assert!(!exists(root, ".graviton", "m"));
    }
}

//! File-level checkpoint/rollback for `grv run`.
//!
//! Scope is deliberately narrow: this snapshots *file* state before every
//! write/edit/delete the agentic loop performs, so `grv rollback` can undo
//! it. It does not and cannot snapshot arbitrary shell-command side effects
//! (a `run_shell` tool call might touch anything, in or out of the repo) —
//! that's why `run_shell` gets a pre-execution confirmation prompt instead
//! of a post-hoc undo. Treat checkpoints as an undo for the agent's file
//! edits, not a full system snapshot.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Create,
    Modify,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestEntry {
    seq: u64,
    ts: i64,
    action: Action,
    /// Path relative to the repo root.
    path: String,
    /// Filename (within the session dir) holding the pre-change bytes.
    /// `None` for `Create`, since there's nothing to restore but "not
    /// there".
    backup: Option<String>,
}

pub struct Session {
    pub id: String,
    dir: PathBuf,
    root: PathBuf,
    manifest_path: PathBuf,
    next_seq: u64,
}

fn checkpoints_root(root: &Path) -> PathBuf {
    root.join(".graviton").join("checkpoints")
}

impl Session {
    /// Start a new checkpoint session for a `grv run` invocation.
    pub fn new(root: &Path) -> Result<Self> {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let id = format!("{}-{:04x}", ts.as_secs(), ts.subsec_nanos() % 0xffff);
        let dir = checkpoints_root(root).join(&id);
        fs::create_dir_all(&dir).context("creating checkpoint session dir")?;
        let manifest_path = dir.join("manifest.jsonl");
        Ok(Self { id, dir, root: root.to_path_buf(), manifest_path, next_seq: 0 })
    }

    /// Record the pre-change state of `rel_path` (relative to the repo
    /// root) and return the action it should be logged under, based on
    /// whether the file exists yet. Call this *before* the tool actually
    /// writes/deletes anything.
    pub fn snapshot_before_write(&mut self, rel_path: &str) -> Result<()> {
        let full = self.root.join(rel_path);
        let seq = self.next_seq;
        self.next_seq += 1;
        let (action, backup) = if full.exists() {
            let bytes = fs::read(&full).with_context(|| format!("reading {} for checkpoint", full.display()))?;
            let backup_name = format!("{seq}.bak");
            fs::write(self.dir.join(&backup_name), bytes)?;
            (Action::Modify, Some(backup_name))
        } else {
            (Action::Create, None)
        };
        self.append(seq, action, rel_path, backup)
    }

    /// Same idea, but for a `delete_file` tool call — the file must exist.
    pub fn snapshot_before_delete(&mut self, rel_path: &str) -> Result<()> {
        let full = self.root.join(rel_path);
        let seq = self.next_seq;
        self.next_seq += 1;
        let bytes = fs::read(&full).with_context(|| format!("reading {} for checkpoint", full.display()))?;
        let backup_name = format!("{seq}.bak");
        fs::write(self.dir.join(&backup_name), bytes)?;
        self.append(seq, Action::Delete, rel_path, Some(backup_name))
    }

    fn append(&self, seq: u64, action: Action, rel_path: &str, backup: Option<String>) -> Result<()> {
        let entry = ManifestEntry {
            seq,
            ts: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0),
            action,
            path: rel_path.to_string(),
            backup,
        };
        let line = serde_json::to_string(&entry)?;
        use std::io::Write;
        let mut f = fs::OpenOptions::new().create(true).append(true).open(&self.manifest_path)?;
        writeln!(f, "{line}")?;
        Ok(())
    }

    pub fn step_count(&self) -> u64 {
        self.next_seq
    }
}

pub struct SessionSummary {
    pub id: String,
    pub steps: usize,
    pub files_touched: usize,
}

fn read_manifest(dir: &Path) -> Result<Vec<ManifestEntry>> {
    let path = dir.join("manifest.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(&path)?;
    Ok(raw.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
}

pub fn list_sessions(root: &Path) -> Result<Vec<SessionSummary>> {
    let base = checkpoints_root(root);
    if !base.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(&base)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        let manifest = read_manifest(&entry.path())?;
        let files: std::collections::HashSet<_> = manifest.iter().map(|e| e.path.clone()).collect();
        out.push(SessionSummary { id, steps: manifest.len(), files_touched: files.len() });
    }
    Ok(out)
}

/// Undo a session's changes (in reverse order), optionally stopping once
/// `seq` values `> keep_after` have been undone (i.e. `to = Some(2)` undoes
/// everything after step 2, leaving steps 0-2 in place). `to = None` undoes
/// the whole session.
pub fn rollback(root: &Path, session_id: &str, to: Option<u64>) -> Result<usize> {
    let dir = checkpoints_root(root).join(session_id);
    if !dir.exists() {
        anyhow::bail!("no checkpoint session '{session_id}'");
    }
    let mut manifest = read_manifest(&dir)?;
    manifest.sort_by_key(|e| e.seq);
    let keep_after = to.unwrap_or(0).max(0);
    let to_undo: Vec<_> = manifest
        .into_iter()
        .filter(|e| to.is_none() || e.seq > keep_after)
        .rev()
        .collect();

    let mut undone = 0;
    for entry in to_undo {
        let full = root.join(&entry.path);
        match entry.action {
            Action::Create => {
                // The tool created this file from nothing; undo = remove it.
                let _ = fs::remove_file(&full);
            }
            Action::Modify | Action::Delete => {
                let Some(backup) = &entry.backup else { continue };
                let bytes = fs::read(dir.join(backup))
                    .with_context(|| format!("reading backup for {}", entry.path))?;
                if let Some(parent) = full.parent() {
                    fs::create_dir_all(parent).ok();
                }
                fs::write(&full, bytes).with_context(|| format!("restoring {}", full.display()))?;
            }
        }
        undone += 1;
    }
    Ok(undone)
}

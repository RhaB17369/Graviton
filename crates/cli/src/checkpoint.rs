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
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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

    /// Reopen a session by id — for `grv run --continue`, so file-change
    /// step numbering (and thus `--to N` rollback) keeps counting up from
    /// where the previous invocation left off instead of restarting at 0
    /// and risking two different steps sharing a sequence number.
    pub fn open_existing(root: &Path, id: &str) -> Result<Self> {
        let dir = checkpoints_root(root).join(id);
        if !dir.exists() {
            anyhow::bail!("no checkpoint session '{id}' — run `grv checkpoints` to list sessions");
        }
        let manifest_path = dir.join("manifest.jsonl");
        let next_seq = read_manifest(&dir)?.iter().map(|e| e.seq + 1).max().unwrap_or(0);
        Ok(Self { id: id.to_string(), dir, root: root.to_path_buf(), manifest_path, next_seq })
    }

    /// Append one message to this session's conversation transcript — the
    /// full history `grv run --continue` restores. Errors are logged, not
    /// propagated: losing the ability to resume shouldn't abort a run that
    /// is otherwise working.
    pub fn append_message(&self, msg: &graviton_llm::ChatMessage) {
        let path = self.dir.join("transcript.jsonl");
        let line = match serde_json::to_string(msg) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("failed to serialize message for session transcript: {e}");
                return;
            }
        };
        use std::io::Write;
        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{line}");
        }
    }

    /// Persist the agent's latest self-reported plan (`update_plan` tool),
    /// overwriting any previous one — this is a current-state snapshot, not
    /// a log.
    pub fn save_plan(&self, plan: &serde_json::Value) -> Result<()> {
        fs::write(self.dir.join("plan.json"), serde_json::to_string_pretty(plan)?)?;
        Ok(())
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

/// A `grv mission` session's persisted tree state, for `grv mission
/// --continue`. Unlike `Session` (`grv run`'s linear file-change/transcript
/// log), a mission's tree-shaped *concurrent* execution has no single
/// sequence to append to — subtasks at the same depth finish in whatever
/// order the scheduler gets to them. So this is a flat map keyed by tree
/// position (`"0"` = root, `"0.0"`/`"0.1"` = its children, ...), rewritten
/// as a whole to disk each time any node's status changes. Fine in
/// practice: even a wide, deep mission produces at most a few hundred
/// nodes, and writes only happen at node boundaries (once per subtask/
/// decompose/synthesize call), not per token.
#[derive(Clone)]
pub struct MissionCheckpoint {
    pub id: String,
    dir: PathBuf,
    tree: Arc<Mutex<HashMap<String, MissionNodeRecord>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionNodeStatus {
    Pending,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionNodeRecord {
    pub task: String,
    pub agent: String,
    pub status: MissionNodeStatus,
    /// The finished result (leaf answer or synthesis), once `status ==
    /// Done`.
    pub result: Option<String>,
    /// This node's exact `decompose()` output, if it's an internal node —
    /// recorded so a resume replays the same tree shape instead of
    /// re-asking the planner, which could return a different split and
    /// orphan already-completed children's cached results. `(agent, task)`
    /// pairs, mirroring `SubtaskSpec` without a cross-module dependency.
    pub subtasks: Option<Vec<(String, String)>>,
}

impl MissionCheckpoint {
    pub fn new(root: &Path) -> Result<Self> {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let id = format!("mission-{}-{:04x}", ts.as_secs(), ts.subsec_nanos() % 0xffff);
        let dir = checkpoints_root(root).join(&id);
        fs::create_dir_all(&dir).context("creating mission checkpoint session dir")?;
        Ok(Self { id, dir, tree: Arc::new(Mutex::new(HashMap::new())) })
    }

    pub fn open_existing(root: &Path, id: &str) -> Result<Self> {
        let dir = checkpoints_root(root).join(id);
        if !dir.exists() {
            anyhow::bail!("no checkpoint session '{id}' — run `grv checkpoints` to list sessions");
        }
        let tree = fs::read_to_string(dir.join("mission_tree.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Ok(Self { id: id.to_string(), dir, tree: Arc::new(Mutex::new(tree)) })
    }

    /// This node's recorded state, if any previous run (including this
    /// one) already reached it.
    pub fn get(&self, node_path: &str) -> Option<MissionNodeRecord> {
        self.tree.lock().unwrap().get(node_path).cloned()
    }

    /// The tree-wide `--max-depth` this session started with, saved once at
    /// the top so `grv mission --continue` (with no `--max-depth` of its
    /// own) reuses it instead of silently defaulting to a *different*
    /// depth than the original run used -- which would make an
    /// already-terminal node decompose further on resume, or vice versa.
    pub fn save_max_depth(&self, max_depth: usize) {
        if let Err(e) = fs::write(self.dir.join("mission_meta.json"), format!(r#"{{"max_depth":{max_depth}}}"#)) {
            tracing::warn!("failed to persist mission max_depth: {e}");
        }
    }

    pub fn load_max_depth(&self) -> Option<usize> {
        let raw = fs::read_to_string(self.dir.join("mission_meta.json")).ok()?;
        let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
        v.get("max_depth")?.as_u64().map(|n| n as usize)
    }

    /// Record (or overwrite) one node's state and persist the whole tree.
    /// Errors are logged, not propagated — losing the ability to resume
    /// shouldn't abort a mission that's otherwise working, same philosophy
    /// as `Session::append_message`.
    pub fn record(&self, node_path: &str, record: MissionNodeRecord) {
        let snapshot = {
            let mut map = self.tree.lock().unwrap();
            map.insert(node_path.to_string(), record);
            map.clone()
        };
        match serde_json::to_string_pretty(&snapshot) {
            Ok(s) => {
                if let Err(e) = fs::write(self.dir.join("mission_tree.json"), s) {
                    tracing::warn!("failed to persist mission checkpoint: {e}");
                }
            }
            Err(e) => tracing::warn!("failed to serialize mission checkpoint: {e}"),
        }
    }
}

/// Like `most_recent_session`, but only among `grv mission` sessions
/// (id prefix `"mission-"`) — the two session kinds share
/// `.graviton/checkpoints/` but `grv run --continue`/`grv mission
/// --continue` should never accidentally pick up each other's session.
pub fn most_recent_mission_session(root: &Path) -> Result<Option<String>> {
    let base = checkpoints_root(root);
    if !base.exists() {
        return Ok(None);
    }
    let mut ids: Vec<String> = fs::read_dir(&base)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("mission-"))
        .collect();
    ids.sort();
    Ok(ids.into_iter().last())
}

/// The most recently created session id, if any — `list_sessions` already
/// sorts by directory name, which sorts by timestamp since ids are
/// `<unix_secs>-<hex>`.
pub fn most_recent_session(root: &Path) -> Result<Option<String>> {
    Ok(list_sessions(root)?.into_iter().last().map(|s| s.id))
}

/// Restore a session's full conversation history — `grv run --continue`'s
/// entry point. Missing/corrupt lines are skipped rather than failing the
/// whole resume, since a transcript is an append-only best-effort log, not
/// a database with transactional guarantees.
pub fn load_transcript(root: &Path, id: &str) -> Result<Vec<graviton_llm::ChatMessage>> {
    let path = checkpoints_root(root).join(id).join("transcript.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(&path)?;
    Ok(raw.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
}

/// The agent's last self-reported plan for a session, if it ever called
/// `update_plan`.
pub fn load_plan(root: &Path, id: &str) -> Result<Option<serde_json::Value>> {
    let path = checkpoints_root(root).join(id).join("plan.json");
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&fs::read_to_string(&path)?)?))
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

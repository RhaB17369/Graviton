//! `grv index --watch`: keep the index in sync as files change, via real
//! filesystem events (`notify`'s recommended backend -- inotify/kqueue/
//! FSEvents depending on platform), not polling.
//!
//! Events are debounced: a save fires several raw events (write, rename,
//! metadata-change) in quick succession, and a build/`git checkout`/branch
//! switch fires a burst across many files -- each burst collapses into one
//! `index_repo` call rather than one per raw event. `index_repo` itself is
//! already incremental (unchanged-file hash skip, and now file-removal
//! cleanup -- see `IndexStats::files_removed`), so re-running it on the
//! whole tree after a burst is correct and cheap, not a full rebuild.

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const DEBOUNCE: Duration = Duration::from_millis(600);

pub fn watch(root: &Path, index_dir: &str, mut conn: rusqlite::Connection) -> Result<()> {
    let (tx, rx) = mpsc::channel::<notify::Event>();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            let _ = tx.send(event);
        }
    })
    .context("setting up filesystem watcher")?;
    watcher
        .watch(root, RecursiveMode::Recursive)
        .with_context(|| format!("watching {}", root.display()))?;

    println!("\x1b[2mwatching {} for changes (Ctrl+C to stop)...\x1b[0m", root.display());

    loop {
        let Ok(first) = rx.recv() else {
            break; // watcher dropped/channel closed
        };
        let mut relevant = is_relevant(root, index_dir, &first);

        // Drain whatever else arrives within the debounce window so one
        // save/checkout/branch-switch becomes one re-index, not N.
        let deadline = Instant::now() + DEBOUNCE;
        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match rx.recv_timeout(deadline - now) {
                Ok(ev) => relevant |= is_relevant(root, index_dir, &ev),
                Err(_) => break, // timed out: debounce window closed
            }
        }

        if !relevant {
            continue;
        }
        match graviton_indexer::index_repo(&mut conn, root) {
            Ok(stats) if stats.files_indexed > 0 || stats.files_removed > 0 => {
                println!(
                    "\x1b[2mre-indexed: {} changed, {} removed, {} symbols, {} call sites, {} chunks\x1b[0m",
                    stats.files_indexed, stats.files_removed, stats.symbols_extracted, stats.calls_extracted, stats.chunks_written
                );
            }
            Ok(_) => {} // event fired but nothing actually changed content-wise (e.g. a touch)
            Err(e) => eprintln!("\x1b[2mre-index failed: {e:#}\x1b[0m"),
        }
    }
    Ok(())
}

/// Create/modify/remove events matter; access/other metadata-only events
/// don't. Anything under a skipped directory (`.git`, `target`,
/// `node_modules`, ..., or this repo's own index dir) is ignored too --
/// without this, the index.db's own WAL writes would trigger re-indexing
/// itself in a loop.
fn is_relevant(root: &Path, index_dir: &str, event: &notify::Event) -> bool {
    use notify::EventKind::*;
    if !matches!(event.kind, Create(_) | Modify(_) | Remove(_)) {
        return false;
    }
    event.paths.iter().any(|p| {
        let rel = p.strip_prefix(root).unwrap_or(p);
        !rel.components().any(|c| {
            let name = c.as_os_str().to_str().unwrap_or("");
            graviton_indexer::SKIP_DIRS.contains(&name) || name == index_dir
        })
    })
}

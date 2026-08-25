//! Wrappers around common recon/security tools: run one, watch its output
//! live like you normally would, and it also lands in the index so
//! `grv ask`/`grv search` can reason over it afterwards.
//!
//! This is a convenience + logging layer, not a sandbox: it runs exactly
//! the command you typed, with your own permissions, same as typing it in
//! the shell directly. The whitelist exists to keep `grv tool run` a
//! recon-tool launcher rather than a generic command runner (use your
//! shell for that), not as a security boundary.

use anyhow::{bail, Context, Result};
use rusqlite::Connection;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Recon/offensive-security tools `grv tool run` knows about. Anything not
/// on this list can still be run directly in your shell — pipe its output
/// through `grv tool ingest <name>` (stdin) if you want it indexed too.
pub const ALLOWED_TOOLS: &[&str] = &[
    "nmap", "masscan", "rustscan",
    "ffuf", "gobuster", "dirb", "wfuzz", "feroxbuster",
    "nikto", "whatweb", "wpscan", "nuclei", "httpx",
    "sqlmap", "hydra", "medusa", "john", "hashcat",
    "subfinder", "amass", "dnsx", "dig", "whois",
    "curl", "nc", "ncat", "netcat",
    "enum4linux", "smbclient", "smbmap", "crackmapexec", "netexec",
    "searchsploit",
];

pub fn is_allowed(tool: &str) -> bool {
    ALLOWED_TOOLS.contains(&tool)
}

/// Run `tool args...`, streaming stdout/stderr to the terminal live while
/// also capturing everything, then store the run in `tool_runs` +
/// `content_fts` (kind = "tool_output") so it's immediately searchable and
/// available to `grv ask`.
pub fn run_and_index(conn: &Connection, tool: &str, args: &[String]) -> Result<i64> {
    if !is_allowed(tool) {
        bail!(
            "'{tool}' isn't in GRAVITON's tool whitelist.\nKnown tools: {}\n\
             (run it directly in your shell and pipe the output through \
             `grv tool ingest {tool}` if you still want it indexed)",
            ALLOWED_TOOLS.join(", ")
        );
    }

    let mut child = Command::new(tool)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("launching '{tool}' (is it installed and on PATH?)"))?;

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let (tx, rx) = mpsc::channel::<String>();

    let tx_out = tx.clone();
    let out_handle = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            println!("{line}");
            let _ = tx_out.send(line);
        }
    });
    let out_err = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("{line}");
            let _ = tx.send(line);
        }
    });

    let mut captured = String::new();
    // Threads own the only senders once we drop ours; the loop below just
    // needs *a* receiver, and it naturally ends when both threads finish
    // and drop their `tx`/`tx_out` clones.
    for line in rx {
        captured.push_str(&line);
        captured.push('\n');
    }
    let _ = out_handle.join();
    let _ = out_err.join();
    let status = child.wait().context("waiting for tool to exit")?;

    let args_joined = args.join(" ");
    let ran_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    conn.execute(
        "INSERT INTO tool_runs (tool, args, ran_at, exit_code, output) VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![tool, args_joined, ran_at, status.code(), captured],
    )?;
    let run_id = conn.last_insert_rowid();

    let line_count = captured.lines().count() as i64;
    let pseudo_path = format!("tool://{tool}#{run_id}");
    conn.execute(
        "INSERT INTO content_fts (path, start_line, end_line, kind, name, body) VALUES (?1, 0, ?2, 'tool_output', ?3, ?4)",
        rusqlite::params![pseudo_path, line_count, format!("{tool} {args_joined}"), captured],
    )?;

    Ok(run_id)
}

/// Index arbitrary already-captured tool output (e.g. piped in from a tool
/// not on the whitelist, or a Burp/Wireshark export) without running
/// anything.
pub fn ingest(conn: &Connection, tool: &str, label: &str, output: &str) -> Result<i64> {
    let ran_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    conn.execute(
        "INSERT INTO tool_runs (tool, args, ran_at, exit_code, output) VALUES (?1, ?2, ?3, NULL, ?4)",
        rusqlite::params![tool, label, ran_at, output],
    )?;
    let run_id = conn.last_insert_rowid();
    let line_count = output.lines().count() as i64;
    let pseudo_path = format!("tool://{tool}#{run_id}");
    conn.execute(
        "INSERT INTO content_fts (path, start_line, end_line, kind, name, body) VALUES (?1, 0, ?2, 'tool_output', ?3, ?4)",
        rusqlite::params![pseudo_path, line_count, format!("{tool} {label}"), output],
    )?;
    Ok(run_id)
}

pub struct ToolRunSummary {
    pub id: i64,
    pub tool: String,
    pub args: String,
    pub ran_at: i64,
    pub exit_code: Option<i64>,
}

pub fn recent_runs(conn: &Connection, limit: usize) -> Result<Vec<ToolRunSummary>> {
    let mut stmt = conn.prepare(
        "SELECT id, tool, args, ran_at, exit_code FROM tool_runs ORDER BY id DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit as i64], |r| {
        Ok(ToolRunSummary {
            id: r.get(0)?,
            tool: r.get(1)?,
            args: r.get(2)?,
            ran_at: r.get(3)?,
            exit_code: r.get(4)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

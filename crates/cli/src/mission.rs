//! `grv mission`: recursive task decomposition over the agent roster.
//!
//! Unlike `crew` (fixed pipeline, real hand-off) and `swarm` (flat,
//! independent, one level), `mission` lets the model itself decide how to
//! break a task down — a planner call proposes subtasks assigned to
//! specialists, each subtask can recursively decompose further up to
//! `--max-depth`, and results are synthesized back up the tree. A subtask
//! the planner judges atomic short-circuits to a leaf immediately, so
//! recursion depth adapts to the task instead of always hitting the ceiling.
//!
//! The one non-negotiable property, per how this shipped: no matter how
//! wide or deep the tree gets, every single model call anywhere in it —
//! leaf work, every planner call, every synthesis call — acquires a permit
//! from the *same* `resources::LiveScheduler`. The scheduler's pool size is
//! resampled from live system RAM every few seconds for the entire mission
//! run, so a mission that fans out into 20 subtasks can't put more
//! concurrent model calls on the machine than it can actually hold, and it
//! grows back into headroom that frees up as earlier subtasks finish.

use crate::agentic;
use crate::agents;
use crate::checkpoint::{MissionCheckpoint, MissionNodeRecord, MissionNodeStatus};
use anyhow::{Context, Result};
use graviton_core::{Config, ModelTier};
use graviton_llm::{ChatMessage, OllamaClient};
use serde::Deserialize;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

const MAX_SUBTASKS_PER_NODE: usize = 6;
const DEFAULT_MAX_DEPTH: usize = 2;
const HARD_DEPTH_CEILING: usize = 4;
/// Leaf agents get web/read tools (see `agentic::read_only_tool_defs`) in a
/// small bounded loop — not the full `grv run` agentic loop, since mission
/// leaves stay analysis-only (no write/edit/delete/shell, no checkpoints).
const MAX_LEAF_TOOL_STEPS: usize = 5;

#[derive(Deserialize, Clone)]
struct SubtaskSpec {
    agent: String,
    task: String,
}

pub async fn run(
    cfg: &Config,
    root: &Path,
    context_block: String,
    task: Option<&str>,
    max_depth: Option<usize>,
    max_parallel: Option<usize>,
    resume_session: Option<String>,
) -> Result<()> {
    let client = OllamaClient::new(&cfg.ollama_host);

    let checkpoint = match &resume_session {
        Some(id) => {
            let cp = MissionCheckpoint::open_existing(root, id)?;
            println!("\x1b[2mmission checkpoint session: {} (resumed)\x1b[0m", cp.id);
            cp
        }
        None => {
            let cp = MissionCheckpoint::new(root)?;
            println!("\x1b[2mmission checkpoint session: {}\x1b[0m", cp.id);
            cp
        }
    };

    // An explicit --max-depth always wins; otherwise a resume reuses the
    // depth the original run started with (see `save_max_depth`'s doc
    // comment for why silently defaulting to a different one on resume
    // would be wrong), and a fresh mission uses the default.
    let depth = match max_depth {
        Some(d) => d.clamp(1, HARD_DEPTH_CEILING),
        None => resume_session
            .as_ref()
            .and_then(|_| checkpoint.load_max_depth())
            .unwrap_or(DEFAULT_MAX_DEPTH)
            .clamp(1, HARD_DEPTH_CEILING),
    };
    if resume_session.is_none() {
        checkpoint.save_max_depth(depth);
    }

    // A resume with no new task text reuses the root node's originally
    // recorded task -- the whole point of --continue is not having to
    // retype it. A resume that also passes an additional task string
    // treats that as a fresh mission sharing the same session/checkpoint
    // dir instead (each tree position is still keyed by node path, so an
    // unrelated task just starts writing fresh "0", "0.0", ... entries;
    // this is intentionally simple rather than trying to detect "same
    // mission, refined task" vs "different mission" from text alone).
    let task: String = match task {
        Some(t) if !t.trim().is_empty() => t.to_string(),
        _ => checkpoint
            .get("0")
            .map(|r| r.task)
            .ok_or_else(|| anyhow::anyhow!("no task given and no prior task recorded in this session — pass one explicitly"))?,
    };
    let task = task.as_str();

    // Every model the roster could possibly call, regardless of which
    // agents the planner ends up picking — sizes the scheduler once,
    // up front, for the whole run.
    let all_models: Vec<String> = {
        let mut m: Vec<String> = agents::ALL_AGENTS.iter().map(|a| cfg.model_for_tier(a.tier).to_string()).collect();
        m.sort();
        m.dedup();
        m
    };
    let sizes = crate::resources::model_sizes_mb(&client).await;
    let hard_cap = max_parallel.unwrap_or(6).max(1);
    let scheduler = crate::resources::LiveScheduler::spawn(all_models, sizes, hard_cap);
    println!(
        "\x1b[2mmission scheduler: ~{} concurrent model call(s) to start, max depth {depth} \
         (both re-sampled/enforced live — see `grv status`)\x1b[0m\n",
        scheduler.current_target()
    );

    let result = execute_node(
        cfg.clone(),
        client,
        root.to_path_buf(),
        context_block,
        task.to_string(),
        "architect".to_string(),
        depth,
        scheduler,
        checkpoint,
        "0".to_string(),
    )
    .await?;

    println!("\x1b[1;35m═══ MISSION RESULT ═══\x1b[0m\n{}", result.trim());
    Ok(())
}

/// One node of the mission tree. `assigned_agent` is what the parent's
/// decomposition (or the top-level default) picked for this node *if* it
/// turns out to be a leaf; if this node decomposes further, each child
/// gets its own assignment from this node's own planner call instead.
/// `node_path` is this node's position in the tree (`"0"` for the root,
/// `"0.1"` for its second child, ...) — the key `checkpoint` persists
/// results under, so `grv mission --continue` can resume by tree position.
#[allow(clippy::too_many_arguments)]
fn execute_node(
    cfg: Config,
    client: OllamaClient,
    root: PathBuf,
    context_block: String,
    task: String,
    assigned_agent: String,
    depth_remaining: usize,
    scheduler: Arc<crate::resources::LiveScheduler>,
    checkpoint: MissionCheckpoint,
    node_path: String,
) -> Pin<Box<dyn Future<Output = Result<String>> + Send>> {
    Box::pin(async move {
        // Resume short-circuit: this exact tree position already finished
        // on a previous invocation of this session.
        if let Some(record) = checkpoint.get(&node_path) {
            if record.status == MissionNodeStatus::Done {
                if let Some(result) = record.result {
                    println!("\x1b[2m[{node_path}] resumed from checkpoint (already done)\x1b[0m");
                    return Ok(result);
                }
            }
        }

        if depth_remaining == 0 {
            let result = run_leaf(&cfg, &client, &root, &context_block, &task, &assigned_agent, &scheduler).await;
            record_leaf(&checkpoint, &node_path, &task, &assigned_agent, &result);
            return result;
        }

        // If this node was already decomposed on a previous (interrupted)
        // invocation, reuse that exact split instead of re-asking the
        // planner -- a different split now would orphan already-completed
        // children's cached results under stale node paths.
        let subtasks = match checkpoint.get(&node_path).and_then(|r| r.subtasks) {
            Some(recorded) => {
                println!("\x1b[2m[{node_path}] resumed decomposition ({} subtask(s)) from checkpoint\x1b[0m", recorded.len());
                recorded.into_iter().map(|(agent, task)| SubtaskSpec { agent, task }).collect()
            }
            None => {
                let subtasks = decompose(&cfg, &client, &context_block, &task, &scheduler).await?;
                checkpoint.record(
                    &node_path,
                    MissionNodeRecord {
                        task: task.clone(),
                        agent: assigned_agent.clone(),
                        status: MissionNodeStatus::Pending,
                        result: None,
                        subtasks: Some(subtasks.iter().map(|s| (s.agent.clone(), s.task.clone())).collect()),
                    },
                );
                subtasks
            }
        };

        if subtasks.len() <= 1 {
            let agent = subtasks.into_iter().next().map(|s| s.agent).unwrap_or(assigned_agent);
            let result = run_leaf(&cfg, &client, &root, &context_block, &task, &agent, &scheduler).await;
            record_leaf(&checkpoint, &node_path, &task, &agent, &result);
            return result;
        }

        println!(
            "\x1b[2mdecomposed \"{}\" into {} subtask(s):\x1b[0m",
            truncate_line(&task, 70),
            subtasks.len()
        );
        for s in &subtasks {
            println!("\x1b[2m  - [{}] {}\x1b[0m", s.agent, truncate_line(&s.task, 90));
        }

        let mut set = tokio::task::JoinSet::new();
        for (i, sub) in subtasks.into_iter().enumerate() {
            set.spawn(execute_node(
                cfg.clone(),
                client.clone(),
                root.clone(),
                context_block.clone(),
                sub.task,
                sub.agent,
                depth_remaining - 1,
                scheduler.clone(),
                checkpoint.clone(),
                format!("{node_path}.{i}"),
            ));
        }
        let mut children = Vec::new();
        while let Some(joined) = set.join_next().await {
            children.push(joined.context("mission subtask panicked")??);
        }

        let result = synthesize(&cfg, &client, &context_block, &task, &children, &scheduler).await;
        record_synthesis(&checkpoint, &node_path, &task, &assigned_agent, &result);
        result
    })
}

/// Persist a leaf node's outcome — `Done` with its result, or `Failed` (no
/// result cached, so a resume retries it rather than replaying an error).
fn record_leaf(checkpoint: &MissionCheckpoint, node_path: &str, task: &str, agent: &str, result: &Result<String>) {
    checkpoint.record(
        node_path,
        MissionNodeRecord {
            task: task.to_string(),
            agent: agent.to_string(),
            status: if result.is_ok() { MissionNodeStatus::Done } else { MissionNodeStatus::Failed },
            result: result.as_ref().ok().cloned(),
            subtasks: None,
        },
    );
}

/// Same idea for an internal node's synthesis step, preserving the
/// already-recorded `subtasks` split (so a retry after a failed synthesis
/// still reuses the same children instead of re-decomposing).
fn record_synthesis(checkpoint: &MissionCheckpoint, node_path: &str, task: &str, agent: &str, result: &Result<String>) {
    let subtasks = checkpoint.get(node_path).and_then(|r| r.subtasks);
    checkpoint.record(
        node_path,
        MissionNodeRecord {
            task: task.to_string(),
            agent: agent.to_string(),
            status: if result.is_ok() { MissionNodeStatus::Done } else { MissionNodeStatus::Failed },
            result: result.as_ref().ok().cloned(),
            subtasks,
        },
    );
}

/// Every model call in a mission — leaf work, planning, synthesis — funnels
/// through here, so every one of them (not just leaves) is gated by the
/// same live scheduler.
async fn call_model(client: &OllamaClient, cfg: &Config, model: &str, system: &str, user_msg: &str, scheduler: &Arc<crate::resources::LiveScheduler>) -> Result<String> {
    let _permit = scheduler.acquire().await;
    let result = client
        .chat_stream(model, &[ChatMessage::system(system), ChatMessage::user(user_msg)], cfg.num_ctx, &[], |_| {})
        .await?;
    Ok(result.content)
}

/// Run one leaf agent with a small bounded tool loop (web_search, web_fetch,
/// read_file, list_dir — see `agentic::read_only_tool_defs`), so a leaf can
/// check current information instead of answering from memory alone. Each
/// round's model call still goes through the shared scheduler like every
/// other call in the tree.
async fn run_leaf(cfg: &Config, client: &OllamaClient, root: &Path, context_block: &str, task: &str, agent_key: &str, scheduler: &Arc<crate::resources::LiveScheduler>) -> Result<String> {
    let spec = agents::find(agent_key).unwrap_or(&agents::ARCHITECT);
    let model = cfg.model_for_tier(spec.tier);
    let tools = agentic::read_only_tool_defs();
    let mut messages = vec![
        ChatMessage::system(spec.system_prompt),
        ChatMessage::user(format!("Task: {task}\n\nRetrieved context:\n{context_block}")),
    ];
    println!("\x1b[1;35m═══ {} (leaf) ═══\x1b[0m", spec.display);

    for _ in 0..MAX_LEAF_TOOL_STEPS {
        let result = {
            let _permit = scheduler.acquire().await;
            client.chat_stream(model, &messages, cfg.num_ctx, &tools, |_| {}).await?
        };
        if result.tool_calls.is_empty() {
            let out = result.content;
            println!("{}\n", out.trim());
            return Ok(format!("[{}] {}", spec.display, out.trim()));
        }
        messages.push(ChatMessage::assistant_tool_calls(result.tool_calls.clone()));
        for tc in &result.tool_calls {
            println!("\x1b[2m  → {}({})\x1b[0m", tc.function.name, tc.function.arguments);
            let out = agentic::dispatch_read_only(cfg, root, &tc.function.name, &tc.function.arguments).await;
            messages.push(ChatMessage::tool_result(&tc.function.name, out));
        }
    }
    let msg = format!("[{}] (stopped after {MAX_LEAF_TOOL_STEPS} tool round(s) without a final answer)", spec.display);
    println!("{msg}\n");
    Ok(msg)
}

async fn decompose(cfg: &Config, client: &OllamaClient, context_block: &str, task: &str, scheduler: &Arc<crate::resources::LiveScheduler>) -> Result<Vec<SubtaskSpec>> {
    let roster: String = agents::ALL_AGENTS.iter().map(|a| format!("{}: {}", a.key, a.tagline)).collect::<Vec<_>>().join("\n");
    let system = format!(
        "You are GRAVITON's mission planner. Break the given task into independent \
         subtasks, each assigned to the single best-fitting specialist from this roster:\n\
         {roster}\n\n\
         If the task is already small/focused enough for one specialist to do directly, \
         return exactly one subtask that IS the original task, assigned to the best \
         specialist — don't force a split that doesn't help. Otherwise return at most \
         {MAX_SUBTASKS_PER_NODE} independent subtasks (independent means they don't need \
         each other's output — if a real dependency exists between two parts, keep them \
         as one subtask for one specialist instead of splitting them).\n\n\
         Respond with ONLY a JSON array, no prose, no code fences:\n\
         [{{\"agent\": \"<key>\", \"task\": \"<subtask description>\"}}, ...]"
    );
    let user_msg = format!("Task: {task}\n\nRetrieved context:\n{context_block}");
    let model = cfg.model_for_tier(ModelTier::Fast);
    let raw = call_model(client, cfg, model, &system, &user_msg, scheduler).await?;
    parse_subtasks(&raw)
}

async fn synthesize(cfg: &Config, client: &OllamaClient, context_block: &str, task: &str, children: &[String], scheduler: &Arc<crate::resources::LiveScheduler>) -> Result<String> {
    let combined = children.join("\n\n---\n\n");
    let system = "You are GRAVITON's mission synthesizer. Combine the following subtask \
                  results into one coherent, non-redundant answer to the original task. \
                  Don't just concatenate them — resolve overlaps, surface any conflicts \
                  between subtask findings, and end with concrete next actions, most \
                  impactful first.";
    let user_msg = format!("Original task: {task}\n\nRetrieved context:\n{context_block}\n\nSubtask results:\n{combined}");
    let model = cfg.model_for_tier(ModelTier::Fast);
    call_model(client, cfg, model, system, &user_msg, scheduler).await
}

/// Extract a JSON array from the planner's response by bracket-counting
/// (respecting string escapes) rather than assuming the model obeyed
/// "no prose, no code fences" perfectly — small local models often don't.
fn parse_subtasks(raw: &str) -> Result<Vec<SubtaskSpec>> {
    let start = raw.find('[').context("no JSON array found in planner response")?;
    let bytes = raw.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.context("unterminated JSON array in planner response")?;
    let json_str = &raw[start..=end];
    let parsed: Vec<SubtaskSpec> = serde_json::from_str(json_str).with_context(|| format!("parsing planner JSON: {json_str}"))?;

    let mut out: Vec<SubtaskSpec> = parsed.into_iter().take(MAX_SUBTASKS_PER_NODE).collect();
    for s in &mut out {
        if agents::find(&s.agent).is_none() {
            s.agent = "architect".to_string();
        }
    }
    Ok(out)
}

fn truncate_line(s: &str, max: usize) -> String {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.len() > max {
        format!("{}…", &s[..max])
    } else {
        s
    }
}

//! User-defined tools for `grv run`, loaded from TOML files at startup — no
//! recompiling to add a new one. A custom tool is a named, described,
//! parameterized shell command template: the model gets a clean tool name
//! and JSON-schema arguments instead of having to construct raw shell
//! syntax itself, but under the hood it still runs through the same
//! confirm-before-executing path as `run_shell`, because that's exactly
//! what it is — a friendlier, named wrapper around one.
//!
//! Two directories are scanned, both optional:
//! - `~/.config/graviton/tools/*.toml` — available in every project
//! - `<repo>/.graviton/tools/*.toml` — this project only, checked into the
//!   repo if the team wants to share it
//!
//! A project-local tool with the same `name` as a global one wins (loaded
//! second, overwrites in the by-name map) — the project's own definition is
//! more likely to be the intentional, current one.

use anyhow::Result;
use graviton_llm::ToolDef;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct CustomToolParam {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub default: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CustomTool {
    pub name: String,
    pub description: String,
    /// Shell command template — `{{param_name}}` is replaced with that
    /// param's (shell-quoted) value.
    pub command: String,
    #[serde(default)]
    pub params: Vec<CustomToolParam>,
    /// Where this was loaded from — not part of the TOML itself, filled in
    /// after parsing, for `grv custom list`/error messages.
    #[serde(skip)]
    pub source: PathBuf,
}

impl CustomTool {
    pub fn to_tool_def(&self) -> ToolDef {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for p in &self.params {
            properties.insert(p.name.clone(), json!({ "type": "string", "description": p.description }));
            if p.required {
                required.push(Value::String(p.name.clone()));
            }
        }
        ToolDef::new(
            self.name.clone(),
            format!("{} (custom tool from {})", self.description, self.source.display()),
            json!({ "type": "object", "properties": Value::Object(properties), "required": required }),
        )
    }

    /// Substitute `args` into `command`, shell-quoting every value. Errors
    /// if a required param is missing with neither an argument nor a
    /// default — better a clear error than a command missing an argument
    /// silently doing the wrong thing.
    pub fn render_command(&self, args: &Value) -> Result<String> {
        let mut out = self.command.clone();
        for p in &self.params {
            let value = args
                .get(&p.name)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| p.default.clone());
            let value = match value {
                Some(v) => v,
                None if p.required => anyhow::bail!("custom tool '{}': missing required param '{}'", self.name, p.name),
                None => String::new(),
            };
            out = out.replace(&format!("{{{{{}}}}}", p.name), &shell_quote(&value));
        }
        Ok(out)
    }
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn tools_dir_global() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("graviton").join("tools"))
}

fn tools_dir_project(root: &Path) -> PathBuf {
    root.join(".graviton").join("tools")
}

fn load_dir(dir: &Path, out: &mut HashMap<String, CustomTool>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        match std::fs::read_to_string(&path).map_err(anyhow::Error::from).and_then(|raw| Ok(toml::from_str::<CustomTool>(&raw)?)) {
            Ok(mut tool) => {
                tool.source = path.clone();
                out.insert(tool.name.clone(), tool);
            }
            Err(e) => {
                eprintln!("\x1b[1;31mwarning: failed to load custom tool {}: {e:#}\x1b[0m", path.display());
            }
        }
    }
}

/// Load every custom tool from both directories, project overriding global
/// on a name collision. Never fails the caller — a bad tool file is a
/// warning on stderr, not a reason to abort `grv run`.
pub fn load_all(root: &Path) -> Vec<CustomTool> {
    let mut by_name = HashMap::new();
    if let Some(global) = tools_dir_global() {
        load_dir(&global, &mut by_name);
    }
    load_dir(&tools_dir_project(root), &mut by_name);
    let mut tools: Vec<CustomTool> = by_name.into_values().collect();
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    tools
}

pub fn find<'a>(tools: &'a [CustomTool], name: &str) -> Option<&'a CustomTool> {
    tools.iter().find(|t| t.name == name)
}

/// Starter content for `grv custom new <name>` — a working example the
/// user edits rather than a blank file with no shape to copy.
pub fn scaffold(name: &str) -> String {
    format!(
        r#"# GRAVITON custom tool — loaded automatically by `grv run`, no
# recompiling needed. Edit this, save it, and it's available immediately.
# `command` is a shell command template: {{{{param_name}}}} is replaced
# with that param's value (shell-quoted for you). Every param becomes a
# named, described argument in the tool's schema the model sees.

name = "{name}"
description = "TODO: describe what this does and when the agent should use it"
command = "echo {{{{message}}}}"

[[params]]
name = "message"
description = "TODO: describe this argument"
required = true
# default = "fallback value if the model omits this argument"
"#
    )
}

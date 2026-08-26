//! Fine-grained per-tool permission rules (`.graviton/permissions.toml`),
//! layered *underneath* the existing confirm/`--yolo` model rather than
//! replacing it — a rule here can make a tool call stronger than `--yolo`
//! (a `deny` rule blocks it even under `--yolo`) or weaker than the
//! default (an `allow` rule skips confirmation even without `--yolo`), but
//! a call that matches nothing falls through to exactly the behavior
//! GRAVITON already had.
//!
//! ```toml
//! [[rule]]
//! tool = "run_shell"
//! pattern = "rm -rf*"
//! action = "deny"        # blocked even under --yolo
//!
//! [[rule]]
//! tool = "web_search"
//! action = "allow"       # never prompts, even without --yolo
//!
//! [[rule]]
//! tool = "delete_file"
//! action = "ask"         # always confirm, even under --yolo
//! ```
//!
//! Rules are checked in file order, first match wins. `tool = "*"` matches
//! any tool. `pattern` (optional) is matched against the call's one
//! "primary" argument — the path for file tools, the command for
//! `run_shell`/custom tools, the joined invocation for `recon_tool` —
//! using a small hand-rolled glob (`*` wildcard only) rather than a crate,
//! since "prefix*"/"*suffix"/"*contains*" covers the realistic cases.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuleAction {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Rule {
    /// Tool name, or "*" for any tool.
    pub tool: String,
    #[serde(default)]
    pub pattern: Option<String>,
    pub action: RuleAction,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RuleFile {
    #[serde(default)]
    rule: Vec<Rule>,
}

/// What a permission check resolved to. `Fallback` means no rule matched —
/// the caller should use its normal confirm/`--yolo` logic unchanged.
pub enum Verdict {
    Allow,
    Deny(String),
    Fallback,
}

pub fn load(root: &Path) -> Vec<Rule> {
    let path = root.join(".graviton").join("permissions.toml");
    let Ok(raw) = std::fs::read_to_string(&path) else { return Vec::new() };
    match toml::from_str::<RuleFile>(&raw) {
        Ok(f) => f.rule,
        Err(e) => {
            eprintln!("\x1b[1;31mwarning: failed to parse {}: {e:#}\x1b[0m", path.display());
            Vec::new()
        }
    }
}

pub fn check(rules: &[Rule], tool: &str, primary_arg: &str) -> Verdict {
    for rule in rules {
        let tool_matches = rule.tool == "*" || rule.tool == tool;
        let pattern_matches = match &rule.pattern {
            None => true,
            Some(p) => glob_match(p, primary_arg),
        };
        if tool_matches && pattern_matches {
            return match rule.action {
                RuleAction::Allow => Verdict::Allow,
                RuleAction::Deny => Verdict::Deny(format!(
                    "blocked by permission rule in .graviton/permissions.toml (tool = \"{}\"{})",
                    rule.tool,
                    rule.pattern.as_ref().map(|p| format!(", pattern = \"{p}\"")).unwrap_or_default()
                )),
                RuleAction::Ask => Verdict::Fallback,
            };
        }
    }
    Verdict::Fallback
}

/// `*` is the only wildcard. Standard two-pointer/recursive glob match —
/// small enough not to need a crate for one feature.
fn glob_match(pattern: &str, text: &str) -> bool {
    fn go(p: &[u8], t: &[u8]) -> bool {
        match (p.first(), t.first()) {
            (None, None) => true,
            (Some(b'*'), _) => go(&p[1..], t) || (!t.is_empty() && go(p, &t[1..])),
            (Some(pc), Some(tc)) if pc == tc => go(&p[1..], &t[1..]),
            _ => false,
        }
    }
    go(pattern.as_bytes(), text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::glob_match;

    #[test]
    fn glob_cases() {
        assert!(glob_match("rm -rf*", "rm -rf /tmp"));
        assert!(!glob_match("rm -rf*", "echo rm -rf"));
        assert!(glob_match("*password*", "cat secrets/password.txt"));
        assert!(glob_match("*.env", "cat backend/.env"));
        assert!(!glob_match("*.env", "cat backend/.env.example"));
        assert!(glob_match("*", "anything at all"));
    }
}

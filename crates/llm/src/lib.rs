//! Minimal streaming client for the local Ollama HTTP API, including tool
//! (function) calling — the foundation for GRAVITON's agentic loop
//! (`grv run`): a tool-capable model (qwen3 and friends) can ask to read/
//! write files, run shell commands, or drive a browser instead of just
//! answering in text.

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// Ollama sends this as a JSON object (not a string to re-parse), unlike
    /// OpenAI's API.
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system" | "user" | "assistant" | "tool"
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into(), tool_calls: None, tool_name: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into(), tool_calls: None, tool_name: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into(), tool_calls: None, tool_name: None }
    }
    /// An assistant turn that issued tool calls (for replaying history back
    /// to the model on the next request in an agentic loop).
    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self { role: "assistant".into(), content: String::new(), tool_calls: Some(tool_calls), tool_name: None }
    }
    /// The result of executing one tool call, fed back to the model.
    pub fn tool_result(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: "tool".into(), content: content.into(), tool_calls: None, tool_name: Some(tool_name.into()) }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunctionDef {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the arguments.
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunctionDef,
}

impl ToolDef {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            kind: "function".into(),
            function: ToolFunctionDef { name: name.into(), description: description.into(), parameters },
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    num_ctx: usize,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    options: ChatOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDef]>,
}

#[derive(Debug, Deserialize)]
struct ChatChunk {
    message: Option<ChatChunkMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatChunkMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

/// One turn of a chat completion: either free-text content, or one/more
/// tool calls the caller is expected to execute and feed back.
#[derive(Debug, Default)]
pub struct ChatResult {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http: reqwest::Client::new(),
        }
    }

    /// Quick reachability + model listing check, used by `grv status`.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("connecting to Ollama at {url}"))?;
        if !resp.status().is_success() {
            bail!("Ollama returned HTTP {}", resp.status());
        }
        let parsed: TagsResponse = resp.json().await.context("parsing /api/tags response")?;
        Ok(parsed.models.into_iter().map(|m| m.name).collect())
    }

    /// On-disk size (MB) of every locally pulled model, keyed by tag — used
    /// to estimate how many models could realistically be resident at once
    /// on this machine's RAM (see `resources::safe_concurrency`).
    pub async fn model_sizes_mb(&self) -> Result<std::collections::HashMap<String, u64>> {
        let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));
        let resp = self.http.get(&url).send().await.with_context(|| format!("connecting to Ollama at {url}"))?;
        if !resp.status().is_success() {
            bail!("Ollama returned HTTP {}", resp.status());
        }
        let parsed: TagsResponse = resp.json().await.context("parsing /api/tags response")?;
        Ok(parsed.models.into_iter().map(|m| (m.name, m.size / 1024 / 1024)).collect())
    }

    /// Stream a chat completion, invoking `on_token` for every incremental
    /// piece of assistant text. Returns the full text plus any tool calls
    /// the model issued (empty if it just answered in text).
    pub async fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        num_ctx: usize,
        tools: &[ToolDef],
        mut on_token: impl FnMut(&str),
    ) -> Result<ChatResult> {
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let req = ChatRequest {
            model,
            messages,
            stream: true,
            options: ChatOptions { num_ctx },
            tools: if tools.is_empty() { None } else { Some(tools) },
        };

        let resp = self
            .http
            .post(&url)
            .json(&req)
            .send()
            .await
            .with_context(|| format!("connecting to Ollama at {url} (is `ollama serve` running?)"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Ollama returned HTTP {status}: {body}");
        }

        let mut result = ChatResult::default();
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("reading stream chunk from Ollama")?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            // Ollama emits newline-delimited JSON objects.
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf.drain(..=pos);
                if line.is_empty() {
                    continue;
                }
                let parsed: ChatChunk = serde_json::from_str(&line)
                    .with_context(|| format!("parsing Ollama stream line: {line}"))?;
                if let Some(err) = parsed.error {
                    bail!("Ollama error: {err}");
                }
                if let Some(msg) = parsed.message {
                    if !msg.content.is_empty() {
                        on_token(&msg.content);
                        result.content.push_str(&msg.content);
                    }
                    if !msg.tool_calls.is_empty() {
                        result.tool_calls.extend(msg.tool_calls);
                    }
                }
                if parsed.done {
                    return Ok(result);
                }
            }
        }
        Ok(result)
    }
}

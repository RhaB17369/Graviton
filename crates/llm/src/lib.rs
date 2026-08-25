//! Minimal streaming client for the local Ollama HTTP API.

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String, // "system" | "user" | "assistant"
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
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
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

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

    /// Stream a chat completion, invoking `on_token` for every incremental
    /// piece of assistant text. Returns the full concatenated response.
    pub async fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        num_ctx: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let req = ChatRequest {
            model,
            messages,
            stream: true,
            options: ChatOptions { num_ctx },
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

        let mut full = String::new();
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
                        full.push_str(&msg.content);
                    }
                }
                if parsed.done {
                    return Ok(full);
                }
            }
        }
        Ok(full)
    }
}

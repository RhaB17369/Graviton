//! Streaming chat clients for GRAVITON's agentic loop (`grv run`), including
//! tool (function) calling: a tool-capable model can ask to read/write
//! files, run shell commands, or drive a browser instead of just answering
//! in text. Two backends:
//! - [`OllamaClient`]: the local Ollama HTTP API.
//! - [`OpenAiCompatClient`]: any OpenAI-compatible `/chat/completions` API
//!   -- what nearly every open-weight *cloud* provider (Groq, Together.ai,
//!   Fireworks, DeepInfra, OpenRouter, Cerebras, ...) exposes, so one
//!   client covers all of them via a configurable base URL, no
//!   provider-specific code.
//!
//! [`LlmClient`] unifies both behind one `chat_stream` call so the rest of
//! GRAVITON (the agentic loop, the read-only tool loop) doesn't need to
//! know which backend a given model routes to -- see
//! `graviton_core::Config::resolve_model` and `graviton-cli`'s
//! `providers::resolve_client`, which build the right variant from a
//! routed model string like `"groq:llama-3.3-70b-versatile"`.

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
    /// The provider-issued id for this specific call -- `None` for Ollama,
    /// which doesn't use ids at all; populated by `OpenAiCompatClient`,
    /// which must echo the *exact* id a provider issued back as that
    /// message's `tool_call_id` on the next request, not a synthesized
    /// one (a provider is free to reject an id it never issued).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
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
    /// `false` for every call GRAVITON makes. A "thinking"-capable model
    /// (confirmed live: qwen3.5, on by default per `ollama show`) streams
    /// its reasoning in a separate `message.thinking` field, not
    /// `message.content`. The bug this guards against: on top of a model
    /// that ignores this flag and keeps reasoning anyway, `chat_stream`
    /// used to not read `thinking` at all, so a turn that spent its whole
    /// token budget reasoning before it would otherwise have reached a
    /// final answer completed with `done: true` and an empty `content`,
    /// silently, no error at all (confirmed live: 80 generated tokens,
    /// `truncated: false`, and nothing printed). `chat_stream` now
    /// streams `thinking` to its own callback (`on_thinking`) as it
    /// arrives -- the chat TUI shows it as a dim, provisional line so a
    /// long reasoning phase isn't silence, same as a real answer's
    /// tokens -- so disabling it here is a latency/clutter choice for
    /// GRAVITON's one-shot callers, not a correctness requirement.
    think: bool,
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
    // Belt-and-suspenders alongside `think: false` on the request: parsed
    // but deliberately not surfaced as a tool-callable answer anywhere --
    // a reasoning trace isn't a final response, and `chat_stream` not
    // reading this field at all was exactly how a thinking model's turn
    // went silently empty (see `ChatRequest::think`'s doc comment). If a
    // model streams here anyway despite `think: false`, at least it's a
    // field that exists now instead of bytes serde quietly drops.
    #[serde(default)]
    thinking: String,
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

#[derive(Debug, Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    embedding: Vec<f32>,
}

/// One turn of a chat completion: either free-text content, or one/more
/// tool calls the caller is expected to execute and feed back.
#[derive(Debug, Default)]
pub struct ChatResult {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// A thinking-capable model's reasoning trace, if it sent one despite
    /// `think: false` on the request -- not currently surfaced by any
    /// caller, kept only so it's never silently dropped the way it used
    /// to be when `ChatChunkMessage` didn't deserialize this field at
    /// all (see that struct's doc comment for the live failure this
    /// fixed).
    pub thinking: String,
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

    /// Embed one piece of text with an embedding-capable model (e.g.
    /// `nomic-embed-text`, `all-minilm`) via Ollama's `/api/embeddings` —
    /// the foundation for semantic (not just lexical/FTS) search. A model
    /// with no embedding head (a plain chat model) will error here rather
    /// than return nonsense; that error is meant to reach the caller.
    pub async fn embed(&self, model: &str, prompt: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .json(&EmbeddingsRequest { model, prompt })
            .send()
            .await
            .with_context(|| format!("connecting to Ollama at {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Ollama returned HTTP {status} embedding with '{model}': {body}");
        }
        let parsed: EmbeddingsResponse = resp.json().await.context("parsing /api/embeddings response")?;
        if parsed.embedding.is_empty() {
            bail!("Ollama returned an empty embedding for model '{model}' — is it an embedding model?");
        }
        Ok(parsed.embedding)
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
        mut on_thinking: impl FnMut(&str),
    ) -> Result<ChatResult> {
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let req = ChatRequest {
            model,
            messages,
            stream: true,
            options: ChatOptions { num_ctx },
            tools: if tools.is_empty() { None } else { Some(tools) },
            think: false,
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
                    if !msg.thinking.is_empty() {
                        on_thinking(&msg.thinking);
                        result.thinking.push_str(&msg.thinking);
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

// ---------------------------------------------------------------------
// OpenAI-compatible cloud client
// ---------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct OpenAiChatRequest<'a> {
    model: &'a str,
    messages: Vec<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDef]>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChunk {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiChoice {
    #[serde(default)]
    delta: OpenAiDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
    /// DeepSeek-R1 and other reasoning models served through an
    /// OpenAI-compatible endpoint stream their reasoning trace here,
    /// separate from `content` -- Ollama's equivalent is
    /// `ChatChunkMessage::thinking`; both feed the same `on_thinking`.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    /// A fragment of the JSON-encoded arguments string -- OpenAI streams
    /// this incrementally across several deltas sharing the same `index`,
    /// unlike Ollama, which sends one tool call whole in a single chunk.
    /// Concatenated across every delta for this index, then parsed as JSON
    /// once the stream ends (see `chat_stream`'s `accum`).
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Default)]
struct ToolCallAccum {
    id: Option<String>,
    name: String,
    arguments: String,
}

/// Turns GRAVITON's provider-agnostic `ChatMessage` history into the exact
/// JSON shape an OpenAI-compatible `/chat/completions` request expects.
/// Two real schema differences from our own (Ollama-shaped) types:
/// - `tool_calls[].function.arguments` is a JSON-encoded *string* here, not
///   a nested object.
/// - every `role: "tool"` message must carry `tool_call_id`, matching one
///   of the immediately preceding assistant message's `tool_calls[].id` --
///   `ChatMessage::tool_result` doesn't store an id at all (Ollama has no
///   use for one), so this reconstructs the pairing positionally: GRAVITON
///   always pushes one assistant `tool_calls` message followed by exactly
///   that many `tool` result messages, in the same order (see
///   `agentic.rs`'s `push_and_record` call sequence), so a FIFO queue of
///   the ids just-assigned to the preceding assistant message's calls
///   always drains in the right order. A real id from `ToolCall::id`
///   (what a previous `OpenAiCompatClient` response actually issued) is
///   used when present; a synthetic one is only a fallback for history
///   that never came from this backend (e.g. switching an agent from
///   Ollama to a cloud provider mid-session).
fn to_openai_wire_messages(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    let mut pending_ids: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    for (i, m) in messages.iter().enumerate() {
        match m.role.as_str() {
            "tool" => {
                let id = pending_ids.pop_front().unwrap_or_else(|| format!("call_{i}"));
                out.push(serde_json::json!({"role": "tool", "tool_call_id": id, "content": m.content}));
            }
            "assistant" if m.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty()) => {
                let calls = m.tool_calls.as_ref().unwrap();
                let wire_calls: Vec<Value> = calls
                    .iter()
                    .enumerate()
                    .map(|(j, tc)| {
                        let id = tc.id.clone().unwrap_or_else(|| format!("call_{i}_{j}"));
                        pending_ids.push_back(id.clone());
                        serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": serde_json::to_string(&tc.function.arguments).unwrap_or_default(),
                            }
                        })
                    })
                    .collect();
                out.push(serde_json::json!({"role": "assistant", "content": m.content, "tool_calls": wire_calls}));
            }
            _ => out.push(serde_json::json!({"role": m.role, "content": m.content})),
        }
    }
    out
}

#[derive(Clone)]
pub struct OpenAiCompatClient {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
}

impl OpenAiCompatClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self { base_url: base_url.into(), api_key: api_key.into(), http: reqwest::Client::new() }
    }

    /// Same signature and streaming behavior as `OllamaClient::chat_stream`
    /// (one callback per text token, one per reasoning-trace token, tool
    /// calls returned whole once the turn finishes) -- `num_ctx` is
    /// accepted but unused: a cloud model's context window is a property
    /// of the model itself, not a per-request option the way Ollama's
    /// `options.num_ctx` is.
    pub async fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        mut on_token: impl FnMut(&str),
        mut on_thinking: impl FnMut(&str),
    ) -> Result<ChatResult> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let req = OpenAiChatRequest {
            model,
            messages: to_openai_wire_messages(messages),
            stream: true,
            tools: if tools.is_empty() { None } else { Some(tools) },
        };

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await
            .with_context(|| format!("connecting to {url}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("provider returned HTTP {status} from {url}: {body}");
        }

        let mut result = ChatResult::default();
        let mut tool_accum: Vec<Option<ToolCallAccum>> = Vec::new();
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();
        'outer: while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("reading stream chunk")?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf.drain(..=pos);
                let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    break 'outer;
                }
                let parsed: OpenAiChunk = serde_json::from_str(data).with_context(|| format!("parsing stream line: {data}"))?;
                if let Some(err) = parsed.error {
                    bail!("provider error: {err}");
                }
                for choice in parsed.choices {
                    if let Some(content) = choice.delta.content.filter(|c| !c.is_empty()) {
                        on_token(&content);
                        result.content.push_str(&content);
                    }
                    if let Some(reasoning) = choice.delta.reasoning_content.filter(|c| !c.is_empty()) {
                        on_thinking(&reasoning);
                        result.thinking.push_str(&reasoning);
                    }
                    for delta in choice.delta.tool_calls.into_iter().flatten() {
                        if tool_accum.len() <= delta.index {
                            tool_accum.resize_with(delta.index + 1, || None);
                        }
                        let slot = tool_accum[delta.index].get_or_insert_with(ToolCallAccum::default);
                        if let Some(id) = delta.id {
                            slot.id = Some(id);
                        }
                        if let Some(f) = delta.function {
                            if let Some(name) = f.name {
                                slot.name.push_str(&name);
                            }
                            if let Some(args) = f.arguments {
                                slot.arguments.push_str(&args);
                            }
                        }
                    }
                    if choice.finish_reason.is_some() {
                        break 'outer;
                    }
                }
            }
        }

        for slot in tool_accum.into_iter().flatten() {
            let arguments = if slot.arguments.trim().is_empty() {
                Value::Object(Default::default())
            } else {
                serde_json::from_str(&slot.arguments).with_context(|| format!("parsing tool call arguments: {}", slot.arguments))?
            };
            result.tool_calls.push(ToolCall { id: slot.id, function: ToolCallFunction { name: slot.name, arguments } });
        }

        Ok(result)
    }
}

/// Unifies `OllamaClient` and `OpenAiCompatClient` behind one
/// `chat_stream` call so callers (the agentic loop, the read-only tool
/// loop) don't need to know or care which backend a routed model string
/// resolved to -- see `graviton_core::Config::resolve_model`.
#[derive(Clone)]
pub enum LlmClient {
    Ollama(OllamaClient),
    OpenAiCompat(OpenAiCompatClient),
}

impl LlmClient {
    pub async fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        num_ctx: usize,
        tools: &[ToolDef],
        on_token: impl FnMut(&str),
        on_thinking: impl FnMut(&str),
    ) -> Result<ChatResult> {
        match self {
            LlmClient::Ollama(c) => c.chat_stream(model, messages, num_ctx, tools, on_token, on_thinking).await,
            LlmClient::OpenAiCompat(c) => c.chat_stream(model, messages, tools, on_token, on_thinking).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn to_openai_wire_messages_pairs_tool_results_with_real_ids() {
        let messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
            ChatMessage::assistant_tool_calls(vec![
                ToolCall { id: Some("call_abc".into()), function: ToolCallFunction { name: "read_file".into(), arguments: serde_json::json!({"path": "a.rs"}) } },
                ToolCall { id: Some("call_def".into()), function: ToolCallFunction { name: "list_dir".into(), arguments: serde_json::json!({"path": "."}) } },
            ]),
            ChatMessage::tool_result("read_file", "contents of a.rs"),
            ChatMessage::tool_result("list_dir", "a.rs, b.rs"),
        ];
        let wire = to_openai_wire_messages(&messages);
        assert_eq!(wire[3]["tool_call_id"], "call_abc");
        assert_eq!(wire[3]["content"], "contents of a.rs");
        assert_eq!(wire[4]["tool_call_id"], "call_def");
        // Arguments must be a JSON-encoded *string*, not a nested object --
        // the real schema difference from Ollama's wire shape.
        assert_eq!(wire[2]["tool_calls"][0]["function"]["arguments"], "{\"path\":\"a.rs\"}");
    }

    #[test]
    fn to_openai_wire_messages_synthesizes_ids_when_none_were_ever_issued() {
        // History that never went through OpenAiCompatClient (e.g. an
        // Ollama session later switched to a cloud provider) has no
        // `ToolCall::id` at all -- must still produce *some* id, paired
        // positionally, rather than panicking or dropping the pairing.
        let messages = vec![
            ChatMessage::assistant_tool_calls(vec![ToolCall { id: None, function: ToolCallFunction { name: "read_file".into(), arguments: serde_json::json!({}) } }]),
            ChatMessage::tool_result("read_file", "ok"),
        ];
        let wire = to_openai_wire_messages(&messages);
        let assistant_id = wire[0]["tool_calls"][0]["id"].as_str().unwrap().to_string();
        assert_eq!(wire[1]["tool_call_id"], assistant_id);
    }

    /// Minimal hand-rolled HTTP server: writes one canned SSE response to
    /// whatever connects, then closes. Exercises `OpenAiCompatClient`'s
    /// real streaming parse over a real socket, not a hand-extracted
    /// chunk of its logic.
    async fn serve_sse(body: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn chat_stream_reassembles_text_split_across_many_deltas() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let url = serve_sse(body.to_string()).await;
        let client = OpenAiCompatClient::new(url, "test-key");
        let mut seen = String::new();
        let result = client.chat_stream("some-model", &[ChatMessage::user("hi")], &[], |tok| seen.push_str(tok), |_| {}).await.unwrap();
        assert_eq!(result.content, "Hello");
        assert_eq!(seen, "Hello");
    }

    #[tokio::test]
    async fn chat_stream_reassembles_a_tool_call_fragmented_across_deltas() {
        // OpenAI-shaped providers stream a tool call's `arguments` as a
        // JSON string split across several deltas sharing one `index` --
        // the one real behavior this client has to get right that Ollama
        // (whole tool call in one message) never exercises.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_xyz\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let url = serve_sse(body.to_string()).await;
        let client = OpenAiCompatClient::new(url, "test-key");
        let result = client.chat_stream("some-model", &[ChatMessage::user("hi")], &[], |_| {}, |_| {}).await.unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, Some("call_xyz".to_string()));
        assert_eq!(result.tool_calls[0].function.name, "read_file");
        assert_eq!(result.tool_calls[0].function.arguments, serde_json::json!({"path": "a.rs"}));
    }

    #[tokio::test]
    async fn chat_stream_routes_reasoning_content_to_on_thinking() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me think...\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"the answer\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let url = serve_sse(body.to_string()).await;
        let client = OpenAiCompatClient::new(url, "test-key");
        let mut thinking = String::new();
        let result = client.chat_stream("some-model", &[ChatMessage::user("hi")], &[], |_| {}, |tok| thinking.push_str(tok)).await.unwrap();
        assert_eq!(thinking, "let me think...");
        assert_eq!(result.content, "the answer");
    }

    #[tokio::test]
    async fn chat_stream_surfaces_a_provider_error_payload() {
        let body = "data: {\"error\":{\"message\":\"rate limited\"}}\n\n".to_string();
        let url = serve_sse(body).await;
        let client = OpenAiCompatClient::new(url, "test-key");
        let err = client.chat_stream("some-model", &[ChatMessage::user("hi")], &[], |_| {}, |_| {}).await.unwrap_err();
        assert!(format!("{err:#}").contains("rate limited"));
    }
}

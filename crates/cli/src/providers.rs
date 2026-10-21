//! Resolves a *routed* model string (whatever `Config::model_for`/
//! `model_for_tier` returned) into an actual client to call -- the one
//! place that reads a cloud provider's API key out of its configured
//! environment variable. Every caller that used to do
//! `OllamaClient::new(&cfg.ollama_host)` and call it with a routed model
//! string directly should go through `resolve_client` instead, so a tier
//! or agent pointed at a cloud provider (`models.toml`'s
//! `agent.<key> = "groq:llama-3.3-70b-versatile"`, resolved by
//! `graviton_core::Config::resolve_model`) actually reaches it.

use anyhow::{Context, Result};
use graviton_core::{Config, ResolvedModel};
use graviton_llm::{LlmClient, OllamaClient, OpenAiCompatClient};

/// Returns the client to call and the bare model name to pass it --
/// a cloud model string has its `"<provider>:"` prefix stripped; a plain
/// Ollama tag comes back as-is, colons and all.
pub fn resolve_client(cfg: &Config, routed_model: &str) -> Result<(LlmClient, String)> {
    match cfg.resolve_model(routed_model) {
        ResolvedModel::Local { model } => Ok((LlmClient::Ollama(OllamaClient::new(&cfg.ollama_host)), model.to_string())),
        ResolvedModel::Cloud { provider_name, provider, model } => {
            let api_key = std::env::var(&provider.api_key_env).with_context(|| {
                format!(
                    "provider '{provider_name}' needs ${} set in this shell's environment \
                     (configured in providers.toml, never read from disk)",
                    provider.api_key_env
                )
            })?;
            Ok((LlmClient::OpenAiCompat(OpenAiCompatClient::new(&provider.base_url, api_key)), model.to_string()))
        }
    }
}

//! OpenAI-compatible adapter. OpenAI and Mistral share the exact same wire
//! format, so one implementation backs both — only the base URL differs.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::AiError;
use crate::provider::{ChunkStream, Provider};
use crate::types::{Chunk, Message, UnifiedRequest, UnifiedResponse, Usage};

pub struct OpenAiCompatible {
    client: reqwest::Client,
    base: String,
    name: &'static str,
    /// OpenAI accepts `stream_options.include_usage`; Mistral does not.
    stream_usage: bool,
    /// Built-in fallback catalog (used when no key is available).
    catalog: &'static [&'static str],
}

const OPENAI_MODELS: &[&str] = &["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3", "o4-mini"];
const MISTRAL_MODELS: &[&str] = &[
    "mistral-large-latest",
    "mistral-small-latest",
    "open-mistral-nemo",
];

pub fn openai() -> OpenAiCompatible {
    OpenAiCompatible {
        client: reqwest::Client::new(),
        base: "https://api.openai.com/v1".to_string(),
        name: "openai",
        stream_usage: true,
        catalog: OPENAI_MODELS,
    }
}

pub fn mistral() -> OpenAiCompatible {
    OpenAiCompatible {
        client: reqwest::Client::new(),
        base: "https://api.mistral.ai/v1".to_string(),
        name: "mistral",
        stream_usage: false,
        catalog: MISTRAL_MODELS,
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize)]
struct ChatResp {
    model: String,
    choices: Vec<Choice>,
    usage: Option<RawUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMsg,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChoiceMsg {
    content: String,
}

#[derive(Deserialize)]
struct RawUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// ── Streaming wire types ────────────────────────────────────────────────
#[derive(Deserialize)]
struct StreamResp {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    usage: Option<RawUsage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResp {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

impl RawUsage {
    fn into_usage(self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
        }
    }
}

#[async_trait]
impl Provider for OpenAiCompatible {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat(&self, req: &UnifiedRequest, key: &str) -> Result<UnifiedResponse, AiError> {
        let body = ChatReq {
            model: &req.model,
            messages: &req.messages,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            stream: false,
            stream_options: None,
        };

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base))
            .bearer_auth(key)
            .json(&body)
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;
        let parsed: ChatResp = resp.json().await?;

        let choice = parsed.choices.into_iter().next().ok_or(AiError::EmptyResponse)?;
        Ok(UnifiedResponse {
            content: choice.message.content,
            model: parsed.model,
            finish_reason: choice.finish_reason,
            usage: parsed.usage.map(RawUsage::into_usage),
        })
    }

    async fn chat_stream(&self, req: &UnifiedRequest, key: &str) -> Result<ChunkStream, AiError> {
        let body = ChatReq {
            model: &req.model,
            messages: &req.messages,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            stream: true,
            stream_options: self
                .stream_usage
                .then_some(StreamOptions { include_usage: true }),
        };

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base))
            .bearer_auth(key)
            .json(&body)
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;

        let stream = resp
            .bytes_stream()
            .eventsource()
            .map(|event| -> Result<Option<Chunk>, AiError> {
                let event = event.map_err(|e| AiError::Stream(e.to_string()))?;
                if event.data == "[DONE]" {
                    return Ok(None);
                }
                let parsed: StreamResp =
                    serde_json::from_str(&event.data).map_err(|e| AiError::Stream(e.to_string()))?;
                let (delta, finish_reason) = match parsed.choices.into_iter().next() {
                    Some(c) => (c.delta.content.unwrap_or_default(), c.finish_reason),
                    None => (String::new(), None),
                };
                let usage = parsed.usage.map(RawUsage::into_usage);
                if delta.is_empty() && finish_reason.is_none() && usage.is_none() {
                    Ok(None)
                } else {
                    Ok(Some(Chunk {
                        delta,
                        finish_reason,
                        usage,
                    }))
                }
            })
            .filter_map(|r| async move { r.transpose() });

        Ok(stream.boxed())
    }

    fn catalog(&self) -> Vec<String> {
        self.catalog.iter().map(|s| s.to_string()).collect()
    }

    async fn list_models(&self, key: &str) -> Result<Vec<String>, AiError> {
        let resp = self
            .client
            .get(format!("{}/models", self.base))
            .bearer_auth(key)
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;
        let parsed: ModelsResp = resp.json().await?;
        Ok(parsed.data.into_iter().map(|m| m.id).collect())
    }
}

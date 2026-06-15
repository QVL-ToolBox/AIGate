//! Anthropic (Claude) adapter. Claude keeps the system prompt out of the
//! message list and uses `x-api-key` + a version header.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::AiError;
use crate::provider::{ChunkStream, Provider};
use crate::types::{Chunk, Role, UnifiedRequest, UnifiedResponse, Usage};

pub struct Claude {
    client: reqwest::Client,
}

impl Claude {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Msg<'a>>,
    stream: bool,
}

#[derive(Serialize)]
struct Msg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResp {
    model: String,
    content: Vec<Block>,
    stop_reason: Option<String>,
    usage: Option<RawUsage>,
}

#[derive(Deserialize)]
struct Block {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct RawUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ── Streaming wire types ────────────────────────────────────────────────
// Anthropic emits named SSE events; we only care about text deltas and the
// terminal stop reason.
#[derive(Deserialize)]
struct ContentBlockDelta {
    delta: TextDelta,
}

#[derive(Deserialize)]
struct TextDelta {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct MessageDelta {
    delta: StopDelta,
}

#[derive(Deserialize)]
struct StopDelta {
    stop_reason: Option<String>,
}

const CLAUDE_MODELS: &[&str] = &[
    "claude-opus-4-1",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-5-haiku-latest",
];

#[derive(Deserialize)]
struct ModelsResp {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

impl Claude {
    /// Split the unified messages into Anthropic's `system` + `messages` shape.
    fn build<'a>(req: &'a UnifiedRequest, stream: bool) -> ChatReq<'a> {
        let mut system = None;
        let mut messages = Vec::new();
        for m in &req.messages {
            match m.role {
                Role::System => system = Some(m.content.clone()),
                Role::User => messages.push(Msg {
                    role: "user",
                    content: &m.content,
                }),
                Role::Assistant => messages.push(Msg {
                    role: "assistant",
                    content: &m.content,
                }),
            }
        }
        ChatReq {
            model: &req.model,
            // Anthropic requires max_tokens; default to a sane value.
            max_tokens: req.max_tokens.unwrap_or(1024),
            system,
            messages,
            stream,
        }
    }
}

#[async_trait]
impl Provider for Claude {
    fn name(&self) -> &'static str {
        "claude"
    }

    async fn chat(&self, req: &UnifiedRequest, key: &str) -> Result<UnifiedResponse, AiError> {
        let body = Self::build(req, false);

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;
        let parsed: ChatResp = resp.json().await?;

        let content: String = parsed.content.into_iter().map(|b| b.text).collect();
        Ok(UnifiedResponse {
            content,
            model: parsed.model,
            finish_reason: parsed.stop_reason,
            usage: parsed.usage.map(|u| Usage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            }),
        })
    }

    async fn chat_stream(&self, req: &UnifiedRequest, key: &str) -> Result<ChunkStream, AiError> {
        let body = Self::build(req, true);

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;

        let stream = resp
            .bytes_stream()
            .eventsource()
            .map(|event| -> Result<Option<Chunk>, AiError> {
                let event = event.map_err(|e| AiError::Stream(e.to_string()))?;
                match event.event.as_str() {
                    "content_block_delta" => {
                        let parsed: ContentBlockDelta = serde_json::from_str(&event.data)
                            .map_err(|e| AiError::Stream(e.to_string()))?;
                        if parsed.delta.text.is_empty() {
                            Ok(None)
                        } else {
                            Ok(Some(Chunk {
                                delta: parsed.delta.text,
                                ..Default::default()
                            }))
                        }
                    }
                    "message_delta" => {
                        let parsed: MessageDelta = serde_json::from_str(&event.data)
                            .map_err(|e| AiError::Stream(e.to_string()))?;
                        Ok(parsed.delta.stop_reason.map(|r| Chunk {
                            finish_reason: Some(r),
                            ..Default::default()
                        }))
                    }
                    // message_start, content_block_start/stop, ping, message_stop
                    _ => Ok(None),
                }
            })
            .filter_map(|r| async move { r.transpose() });

        Ok(stream.boxed())
    }

    fn catalog(&self) -> Vec<String> {
        CLAUDE_MODELS.iter().map(|s| s.to_string()).collect()
    }

    async fn list_models(&self, key: &str) -> Result<Vec<String>, AiError> {
        let resp = self
            .client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;
        let parsed: ModelsResp = resp.json().await?;
        Ok(parsed.data.into_iter().map(|m| m.id).collect())
    }
}

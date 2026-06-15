//! Anthropic (Claude) adapter. Claude keeps the system prompt out of the
//! message list and uses `x-api-key` + a version header.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::AiError;
use crate::provider::Provider;
use crate::types::{Role, UnifiedRequest, UnifiedResponse, Usage};

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

#[async_trait]
impl Provider for Claude {
    fn name(&self) -> &'static str {
        "claude"
    }

    async fn chat(&self, req: &UnifiedRequest, key: &str) -> Result<UnifiedResponse, AiError> {
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

        let body = ChatReq {
            model: &req.model,
            // Anthropic requires max_tokens; default to a sane value.
            max_tokens: req.max_tokens.unwrap_or(1024),
            system,
            messages,
        };

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
}

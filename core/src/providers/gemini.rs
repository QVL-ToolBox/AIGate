//! Google Gemini adapter. Gemini uses `contents` with `user`/`model` roles,
//! a separate `systemInstruction`, and the API key as a query parameter.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::AiError;
use crate::provider::{ChunkStream, Provider};
use crate::types::{Chunk, Role, UnifiedRequest, UnifiedResponse, Usage};

pub struct Gemini {
    client: reqwest::Client,
}

impl Gemini {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatReq {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<SystemInstruction>,
}

#[derive(Serialize)]
struct Content {
    role: String,
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Serialize, Deserialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatResp {
    #[serde(default)]
    candidates: Vec<Candidate>,
    usage_metadata: Option<UsageMeta>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: OutContent,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OutContent {
    #[serde(default)]
    parts: Vec<Part>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMeta {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    total_token_count: u32,
}

impl UsageMeta {
    fn into_usage(self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_token_count,
            completion_tokens: self.candidates_token_count,
            total_tokens: self.total_token_count,
        }
    }
}

impl ChatResp {
    /// Reduce one response payload (full or streamed) into a `(text, finish, usage)`.
    fn into_parts(self) -> (String, Option<String>, Option<Usage>) {
        let usage = self.usage_metadata.map(UsageMeta::into_usage);
        match self.candidates.into_iter().next() {
            Some(c) => (
                c.content.parts.into_iter().map(|p| p.text).collect(),
                c.finish_reason,
                usage,
            ),
            None => (String::new(), None, usage),
        }
    }
}

impl Gemini {
    fn build(req: &UnifiedRequest) -> ChatReq {
        let mut contents = Vec::new();
        let mut system_instruction = None;
        for m in &req.messages {
            match m.role {
                Role::System => {
                    system_instruction = Some(SystemInstruction {
                        parts: vec![Part {
                            text: m.content.clone(),
                        }],
                    })
                }
                Role::User => contents.push(Content {
                    role: "user".to_string(),
                    parts: vec![Part {
                        text: m.content.clone(),
                    }],
                }),
                Role::Assistant => contents.push(Content {
                    role: "model".to_string(),
                    parts: vec![Part {
                        text: m.content.clone(),
                    }],
                }),
            }
        }
        ChatReq {
            contents,
            system_instruction,
        }
    }
}

#[async_trait]
impl Provider for Gemini {
    fn name(&self) -> &'static str {
        "gemini"
    }

    async fn chat(&self, req: &UnifiedRequest, key: &str) -> Result<UnifiedResponse, AiError> {
        let body = Self::build(req);
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            req.model
        );

        let resp = self
            .client
            .post(url)
            .query(&[("key", key)])
            .json(&body)
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;
        let parsed: ChatResp = resp.json().await?;

        let (content, finish_reason, usage) = parsed.into_parts();
        if content.is_empty() && finish_reason.is_none() {
            return Err(AiError::EmptyResponse);
        }
        Ok(UnifiedResponse {
            content,
            model: req.model.clone(),
            finish_reason,
            usage,
        })
    }

    async fn chat_stream(&self, req: &UnifiedRequest, key: &str) -> Result<ChunkStream, AiError> {
        let body = Self::build(req);
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent",
            req.model
        );

        let resp = self
            .client
            .post(url)
            .query(&[("alt", "sse"), ("key", key)])
            .json(&body)
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;

        let stream = resp
            .bytes_stream()
            .eventsource()
            .map(|event| -> Result<Option<Chunk>, AiError> {
                let event = event.map_err(|e| AiError::Stream(e.to_string()))?;
                let parsed: ChatResp =
                    serde_json::from_str(&event.data).map_err(|e| AiError::Stream(e.to_string()))?;
                let (delta, finish_reason, usage) = parsed.into_parts();
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
}

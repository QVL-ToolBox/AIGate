//! Google Gemini adapter. Gemini uses `contents` with `user`/`model` roles,
//! a separate `systemInstruction`, and the API key as a query parameter.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::AiError;
use crate::provider::Provider;
use crate::types::{Role, UnifiedRequest, UnifiedResponse, Usage};

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

#[async_trait]
impl Provider for Gemini {
    fn name(&self) -> &'static str {
        "gemini"
    }

    async fn chat(&self, req: &UnifiedRequest, key: &str) -> Result<UnifiedResponse, AiError> {
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

        let body = ChatReq {
            contents,
            system_instruction,
        };
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

        let candidate = parsed.candidates.into_iter().next().ok_or(AiError::EmptyResponse)?;
        let content: String = candidate.content.parts.into_iter().map(|p| p.text).collect();
        Ok(UnifiedResponse {
            content,
            model: req.model.clone(),
            finish_reason: candidate.finish_reason,
            usage: parsed.usage_metadata.map(|u| Usage {
                prompt_tokens: u.prompt_token_count,
                completion_tokens: u.candidates_token_count,
                total_tokens: u.total_token_count,
            }),
        })
    }
}

//! OpenAI-compatible adapter. OpenAI and Mistral share the exact same wire
//! format, so one implementation backs both — only the base URL differs.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::AiError;
use crate::provider::Provider;
use crate::types::{Message, UnifiedRequest, UnifiedResponse, Usage};

pub struct OpenAiCompatible {
    client: reqwest::Client,
    base: String,
    name: &'static str,
}

pub fn openai() -> OpenAiCompatible {
    OpenAiCompatible {
        client: reqwest::Client::new(),
        base: "https://api.openai.com/v1".to_string(),
        name: "openai",
    }
}

pub fn mistral() -> OpenAiCompatible {
    OpenAiCompatible {
        client: reqwest::Client::new(),
        base: "https://api.mistral.ai/v1".to_string(),
        name: "mistral",
    }
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
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
            usage: parsed.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        })
    }
}

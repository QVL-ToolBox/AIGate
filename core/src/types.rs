//! Unified request/response types. These mirror the OpenAI chat shape so the
//! HTTP layer can deserialize an OpenAI-compatible body straight into them.

use serde::{Deserialize, Serialize};

/// Role of a chat message, normalized across every engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// A provider-agnostic chat request. `model` may carry a `provider/model`
/// prefix (e.g. `gemini/gemini-2.0-flash`); see [`split_model`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
}

/// Token accounting, normalized across engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A normalized non-streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedResponse {
    pub content: String,
    pub model: String,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

/// One increment of a streamed response, normalized across engines.
///
/// `delta` is the text fragment for this event (may be empty on events that
/// only carry a finish reason or usage). `finish_reason` and `usage` are set
/// on the terminal events that provide them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Chunk {
    pub delta: String,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

/// Split a `provider/model` string into its parts.
/// `"openai/gpt-4o"` -> `(Some("openai"), "gpt-4o")`;
/// `"gpt-4o"` -> `(None, "gpt-4o")`.
pub fn split_model(model: &str) -> (Option<&str>, &str) {
    match model.split_once('/') {
        Some((provider, model)) => (Some(provider), model),
        None => (None, model),
    }
}

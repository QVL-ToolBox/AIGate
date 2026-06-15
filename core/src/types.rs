//! Unified request/response types. These mirror the OpenAI chat shape so the
//! HTTP layer can deserialize an OpenAI-compatible body straight into them.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Role of a chat message, normalized across every engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

fn default_function() -> String {
    "function".to_string()
}

/// A function invocation requested by the model (OpenAI convention: `arguments`
/// is a JSON-encoded string, not a parsed object).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

/// One tool call in an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "default_function")]
    pub kind: String,
    pub function: FunctionCall,
}

/// A single chat message. `content` is optional (an assistant message that only
/// makes tool calls has none); `tool_calls` appears on assistant turns and
/// `tool_call_id` on `tool`-role results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Text content, or empty string when absent.
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}

/// A function the model may call. `parameters` is a JSON Schema object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// A tool declaration (only `function` tools are supported).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "default_function")]
    pub kind: String,
    pub function: FunctionDef,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Raw OpenAI `tool_choice` value, translated per engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    pub model: String,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

/// A streamed tool-call fragment, mirroring OpenAI's `delta.tool_calls` shape.
/// `id`/`name` arrive on the first fragment for a given `index`; `arguments`
/// streams as concatenated partial JSON across fragments.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallChunk {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: String,
}

/// One increment of a streamed response, normalized across engines.
///
/// `delta` is the text fragment for this event (may be empty on events that
/// only carry tool calls, a finish reason, or usage). `tool_calls` carries
/// streamed function-call fragments; `finish_reason` and `usage` are set on the
/// terminal events that provide them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Chunk {
    pub delta: String,
    pub tool_calls: Option<Vec<ToolCallChunk>>,
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

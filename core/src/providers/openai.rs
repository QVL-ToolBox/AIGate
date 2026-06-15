//! OpenAI-compatible adapter. OpenAI and Mistral share the exact same wire
//! format, so one implementation backs both — only the base URL differs.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use serde_json::Value;

use crate::error::AiError;
use crate::provider::{ChunkStream, Provider};
use crate::types::{
    Chunk, Message, Tool, ToolCall, ToolCallChunk, UnifiedRequest, UnifiedResponse, Usage,
};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Tool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a Value>,
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
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
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
    #[serde(default)]
    tool_calls: Option<Vec<ToolDelta>>,
}

#[derive(Deserialize)]
struct ToolDelta {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FnDelta>,
}

#[derive(Deserialize)]
struct FnDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
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

/// Convert one streamed payload into a unified [`Chunk`], or `None` if it
/// carries nothing useful.
fn chunk_from(parsed: StreamResp) -> Option<Chunk> {
    let (delta, finish_reason, tool_calls) = match parsed.choices.into_iter().next() {
        Some(c) => {
            let delta = c.delta.content.unwrap_or_default();
            let tool_calls = c.delta.tool_calls.map(|deltas| {
                deltas
                    .into_iter()
                    .map(|t| ToolCallChunk {
                        index: t.index,
                        id: t.id,
                        name: t.function.as_ref().and_then(|f| f.name.clone()),
                        arguments: t.function.and_then(|f| f.arguments).unwrap_or_default(),
                    })
                    .collect::<Vec<_>>()
            });
            (delta, c.finish_reason, tool_calls)
        }
        None => (String::new(), None, None),
    };
    let usage = parsed.usage.map(RawUsage::into_usage);
    let tool_calls = tool_calls.filter(|v| !v.is_empty());

    if delta.is_empty() && tool_calls.is_none() && finish_reason.is_none() && usage.is_none() {
        None
    } else {
        Some(Chunk {
            delta,
            tool_calls,
            finish_reason,
            usage,
        })
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
            tools: req.tools.as_deref(),
            tool_choice: req.tool_choice.as_ref(),
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
            content: choice.message.content.unwrap_or_default(),
            tool_calls: choice.message.tool_calls,
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
            tools: req.tools.as_deref(),
            tool_choice: req.tool_choice.as_ref(),
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
                Ok(chunk_from(parsed))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_tool_call_delta_maps() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":"}}
        ]}}]}"#;
        let parsed: StreamResp = serde_json::from_str(data).unwrap();
        let chunk = chunk_from(parsed).expect("a chunk");
        let tc = &chunk.tool_calls.expect("tool calls")[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("call_1"));
        assert_eq!(tc.name.as_deref(), Some("f"));
        assert_eq!(tc.arguments, "{\"a\":");
    }

    #[test]
    fn stream_arguments_fragment_has_no_id_or_name() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[
            {"index":0,"function":{"arguments":"1}"}}
        ]}}]}"#;
        let parsed: StreamResp = serde_json::from_str(data).unwrap();
        let tc = chunk_from(parsed).unwrap().tool_calls.unwrap().remove(0);
        assert_eq!(tc.index, 0);
        assert!(tc.id.is_none());
        assert!(tc.name.is_none());
        assert_eq!(tc.arguments, "1}");
    }

    #[test]
    fn empty_keepalive_chunk_is_dropped() {
        let parsed: StreamResp = serde_json::from_str(r#"{"choices":[]}"#).unwrap();
        assert!(chunk_from(parsed).is_none());
    }
}

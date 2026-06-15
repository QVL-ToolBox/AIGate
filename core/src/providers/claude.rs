//! Anthropic (Claude) adapter. Claude keeps the system prompt out of the
//! message list, uses `x-api-key` + a version header, and represents tool use
//! as content blocks (`tool_use` from the model, `tool_result` back to it).

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::AiError;
use crate::provider::{ChunkStream, Provider};
use crate::types::{Chunk, FunctionCall, Role, ToolCall, UnifiedRequest, UnifiedResponse, Usage};

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
struct ChatReq {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResp {
    model: String,
    content: Vec<Block>,
    stop_reason: Option<String>,
    usage: Option<RawUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Block {
    #[serde(rename = "text")]
    Text {
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct RawUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
struct MessageDelta {
    delta: StopDelta,
}

#[derive(Deserialize)]
struct StopDelta {
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ContentBlockDelta {
    delta: TextDelta,
}

#[derive(Deserialize)]
struct TextDelta {
    #[serde(default)]
    text: String,
}

const CLAUDE_MODELS: &[&str] = &[
    "claude-opus-4-1",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-5-haiku-latest",
];

/// Translate an OpenAI `tool_choice` value into Anthropic's shape.
fn map_tool_choice(choice: &Value) -> Value {
    match choice {
        Value::String(s) => match s.as_str() {
            "required" | "any" => json!({ "type": "any" }),
            "none" => json!({ "type": "none" }),
            _ => json!({ "type": "auto" }),
        },
        Value::Object(o) => o
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .map(|name| json!({ "type": "tool", "name": name }))
            .unwrap_or_else(|| json!({ "type": "auto" })),
        _ => json!({ "type": "auto" }),
    }
}

impl Claude {
    fn build(req: &UnifiedRequest, stream: bool) -> ChatReq {
        let mut system = None;
        let mut messages: Vec<Value> = Vec::new();

        for m in &req.messages {
            match m.role {
                Role::System => system = m.content.clone(),
                Role::User => messages.push(json!({ "role": "user", "content": m.text() })),
                Role::Assistant => {
                    let mut blocks: Vec<Value> = Vec::new();
                    if !m.text().is_empty() {
                        blocks.push(json!({ "type": "text", "text": m.text() }));
                    }
                    if let Some(calls) = &m.tool_calls {
                        for call in calls {
                            let input: Value =
                                serde_json::from_str(&call.function.arguments).unwrap_or(json!({}));
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.function.name,
                                "input": input,
                            }));
                        }
                    }
                    if blocks.is_empty() {
                        blocks.push(json!({ "type": "text", "text": "" }));
                    }
                    messages.push(json!({ "role": "assistant", "content": blocks }));
                }
                Role::Tool => messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.text(),
                    }],
                })),
            }
        }

        let tools = req.tools.as_ref().map(|ts| {
            ts.iter()
                .map(|t| {
                    let mut tool = serde_json::Map::new();
                    tool.insert("name".into(), json!(t.function.name));
                    if let Some(d) = &t.function.description {
                        tool.insert("description".into(), json!(d));
                    }
                    tool.insert(
                        "input_schema".into(),
                        t.function
                            .parameters
                            .clone()
                            .unwrap_or_else(|| json!({ "type": "object" })),
                    );
                    Value::Object(tool)
                })
                .collect()
        });

        ChatReq {
            model: req.model.clone(),
            // Anthropic requires max_tokens; default to a sane value.
            max_tokens: req.max_tokens.unwrap_or(1024),
            system,
            messages,
            tools,
            tool_choice: req.tool_choice.as_ref().map(map_tool_choice),
            stream,
        }
    }

    async fn send(&self, body: &ChatReq, key: &str) -> Result<reqwest::Response, AiError> {
        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(body)
            .send()
            .await?;
        super::ensure_ok(resp).await
    }
}

#[async_trait]
impl Provider for Claude {
    fn name(&self) -> &'static str {
        "claude"
    }

    async fn chat(&self, req: &UnifiedRequest, key: &str) -> Result<UnifiedResponse, AiError> {
        let body = Self::build(req, false);
        let parsed: ChatResp = self.send(&body, key).await?.json().await?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        for block in parsed.content {
            match block {
                Block::Text { text } => content.push_str(&text),
                Block::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                    id,
                    kind: "function".to_string(),
                    function: FunctionCall {
                        name,
                        arguments: input.to_string(),
                    },
                }),
                Block::Other => {}
            }
        }

        let tool_calls = (!tool_calls.is_empty()).then_some(tool_calls);
        let finish_reason = if tool_calls.is_some() {
            Some("tool_calls".to_string())
        } else {
            parsed.stop_reason
        };

        Ok(UnifiedResponse {
            content,
            tool_calls,
            model: parsed.model,
            finish_reason,
            usage: parsed.usage.map(|u| Usage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            }),
        })
    }

    async fn chat_stream(&self, req: &UnifiedRequest, key: &str) -> Result<ChunkStream, AiError> {
        let body = Self::build(req, true);
        let resp = self.send(&body, key).await?;

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
        #[derive(Deserialize)]
        struct ModelsResp {
            data: Vec<ModelEntry>,
        }
        #[derive(Deserialize)]
        struct ModelEntry {
            id: String,
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionDef, Message, Tool};

    fn req(messages: Vec<Message>, tools: Option<Vec<Tool>>) -> UnifiedRequest {
        UnifiedRequest {
            model: "claude-sonnet-4-5".into(),
            messages,
            temperature: None,
            max_tokens: None,
            stream: false,
            tools,
            tool_choice: None,
        }
    }

    #[test]
    fn tools_become_input_schema() {
        let tool = Tool {
            kind: "function".into(),
            function: FunctionDef {
                name: "get_weather".into(),
                description: Some("Weather".into()),
                parameters: Some(json!({ "type": "object", "properties": {} })),
            },
        };
        let body = Claude::build(&req(vec![], Some(vec![tool])), false);
        let tools = body.tools.expect("tools present");
        assert_eq!(tools[0]["name"], "get_weather");
        assert!(tools[0].get("input_schema").is_some());
    }

    #[test]
    fn tool_result_becomes_user_block() {
        let msg = Message {
            role: Role::Tool,
            content: Some("{\"temp\":20}".into()),
            tool_calls: None,
            tool_call_id: Some("toolu_1".into()),
            name: None,
        };
        let body = Claude::build(&req(vec![msg], None), false);
        let m = &body.messages[0];
        assert_eq!(m["role"], "user");
        assert_eq!(m["content"][0]["type"], "tool_result");
        assert_eq!(m["content"][0]["tool_use_id"], "toolu_1");
    }
}

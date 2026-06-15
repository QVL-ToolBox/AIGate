//! Google Gemini adapter. Gemini uses `contents` with `user`/`model` roles, a
//! separate `systemInstruction`, the API key as a query parameter, and
//! `functionCall` / `functionResponse` parts for tool use. Gemini does not
//! return tool-call ids, so we synthesize them.

use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::AiError;
use crate::provider::{ChunkStream, Provider};
use crate::types::{Chunk, FunctionCall, Role, ToolCall, UnifiedRequest, UnifiedResponse, Usage};

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

// ── Response wire types ─────────────────────────────────────────────────
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
    content: Option<OutContent>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OutContent {
    #[serde(default)]
    parts: Vec<OutPart>,
}

#[derive(Deserialize)]
struct OutPart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "functionCall")]
    function_call: Option<FnCall>,
}

#[derive(Deserialize)]
struct FnCall {
    name: String,
    #[serde(default)]
    args: Value,
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

const GEMINI_MODELS: &[&str] = &[
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
];

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
    /// Reduce one payload (full or streamed) into text, tool calls, finish, usage.
    fn into_parts(self) -> (String, Option<Vec<ToolCall>>, Option<String>, Option<Usage>) {
        let usage = self.usage_metadata.map(UsageMeta::into_usage);
        let Some(candidate) = self.candidates.into_iter().next() else {
            return (String::new(), None, None, usage);
        };
        let mut text = String::new();
        let mut calls = Vec::new();
        if let Some(content) = candidate.content {
            for (i, part) in content.parts.into_iter().enumerate() {
                if let Some(t) = part.text {
                    text.push_str(&t);
                }
                if let Some(fc) = part.function_call {
                    calls.push(ToolCall {
                        id: format!("call_{i}"),
                        kind: "function".to_string(),
                        function: FunctionCall {
                            name: fc.name,
                            arguments: fc.args.to_string(),
                        },
                    });
                }
            }
        }
        let tool_calls = (!calls.is_empty()).then_some(calls);
        (text, tool_calls, candidate.finish_reason, usage)
    }
}

/// Translate an OpenAI `tool_choice` value into Gemini's `toolConfig`.
fn map_tool_choice(choice: &Value) -> Value {
    let cfg = match choice {
        Value::String(s) => match s.as_str() {
            "required" | "any" => json!({ "mode": "ANY" }),
            "none" => json!({ "mode": "NONE" }),
            _ => json!({ "mode": "AUTO" }),
        },
        Value::Object(o) => o
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .map(|name| json!({ "mode": "ANY", "allowedFunctionNames": [name] }))
            .unwrap_or_else(|| json!({ "mode": "AUTO" })),
        _ => json!({ "mode": "AUTO" }),
    };
    json!({ "functionCallingConfig": cfg })
}

impl Gemini {
    fn build(req: &UnifiedRequest) -> Value {
        // Map tool_call_id -> function name so tool results can name their call.
        let mut id_to_name: HashMap<&str, &str> = HashMap::new();
        for m in &req.messages {
            if let Some(calls) = &m.tool_calls {
                for call in calls {
                    id_to_name.insert(&call.id, &call.function.name);
                }
            }
        }

        let mut contents: Vec<Value> = Vec::new();
        let mut system_instruction = None;
        for m in &req.messages {
            match m.role {
                Role::System => {
                    system_instruction = Some(json!({ "parts": [{ "text": m.text() }] }))
                }
                Role::User => {
                    contents.push(json!({ "role": "user", "parts": [{ "text": m.text() }] }))
                }
                Role::Assistant => {
                    let mut parts: Vec<Value> = Vec::new();
                    if !m.text().is_empty() {
                        parts.push(json!({ "text": m.text() }));
                    }
                    if let Some(calls) = &m.tool_calls {
                        for call in calls {
                            let args: Value =
                                serde_json::from_str(&call.function.arguments).unwrap_or(json!({}));
                            parts.push(json!({
                                "functionCall": { "name": call.function.name, "args": args }
                            }));
                        }
                    }
                    if parts.is_empty() {
                        parts.push(json!({ "text": "" }));
                    }
                    contents.push(json!({ "role": "model", "parts": parts }));
                }
                Role::Tool => {
                    let name = m
                        .tool_call_id
                        .as_deref()
                        .and_then(|id| id_to_name.get(id).copied())
                        .unwrap_or("");
                    // functionResponse.response must be an object.
                    let response = match m.content.as_deref().map(serde_json::from_str::<Value>) {
                        Some(Ok(v)) if v.is_object() => v,
                        _ => json!({ "result": m.text() }),
                    };
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "functionResponse": { "name": name, "response": response } }],
                    }));
                }
            }
        }

        let mut body = serde_json::Map::new();
        body.insert("contents".into(), json!(contents));
        if let Some(sys) = system_instruction {
            body.insert("systemInstruction".into(), sys);
        }
        if let Some(tools) = &req.tools {
            let decls: Vec<Value> = tools
                .iter()
                .map(|t| {
                    let mut d = serde_json::Map::new();
                    d.insert("name".into(), json!(t.function.name));
                    if let Some(desc) = &t.function.description {
                        d.insert("description".into(), json!(desc));
                    }
                    if let Some(params) = &t.function.parameters {
                        d.insert("parameters".into(), params.clone());
                    }
                    Value::Object(d)
                })
                .collect();
            body.insert("tools".into(), json!([{ "functionDeclarations": decls }]));
        }
        if let Some(choice) = &req.tool_choice {
            body.insert("toolConfig".into(), map_tool_choice(choice));
        }
        Value::Object(body)
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

        let (content, tool_calls, finish, usage) = parsed.into_parts();
        if content.is_empty() && tool_calls.is_none() && finish.is_none() {
            return Err(AiError::EmptyResponse);
        }
        let finish_reason = if tool_calls.is_some() {
            Some("tool_calls".to_string())
        } else {
            finish
        };
        Ok(UnifiedResponse {
            content,
            tool_calls,
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
                // Streamed tool calls are not surfaced as chunks yet.
                let (delta, _tool_calls, finish_reason, usage) = parsed.into_parts();
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
        GEMINI_MODELS.iter().map(|s| s.to_string()).collect()
    }

    async fn list_models(&self, key: &str) -> Result<Vec<String>, AiError> {
        #[derive(Deserialize)]
        struct ModelsResp {
            #[serde(default)]
            models: Vec<ModelEntry>,
        }
        #[derive(Deserialize)]
        struct ModelEntry {
            name: String,
        }

        let resp = self
            .client
            .get("https://generativelanguage.googleapis.com/v1beta/models")
            .query(&[("key", key)])
            .send()
            .await?;
        let resp = super::ensure_ok(resp).await?;
        let parsed: ModelsResp = resp.json().await?;
        Ok(parsed
            .models
            .into_iter()
            .map(|m| m.name.strip_prefix("models/").unwrap_or(&m.name).to_string())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn tool_result_resolves_name_from_prior_call() {
        let assistant = Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_0".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: "{\"city\":\"Paris\"}".into(),
                },
            }]),
            tool_call_id: None,
            name: None,
        };
        let tool = Message {
            role: Role::Tool,
            content: Some("{\"temp\":20}".into()),
            tool_calls: None,
            tool_call_id: Some("call_0".into()),
            name: None,
        };
        let req = UnifiedRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![assistant, tool],
            temperature: None,
            max_tokens: None,
            stream: false,
            tools: None,
            tool_choice: None,
        };
        let body = Gemini::build(&req);
        let contents = body["contents"].as_array().unwrap();
        // Assistant turn carries the functionCall.
        assert_eq!(contents[0]["parts"][0]["functionCall"]["name"], "get_weather");
        // Tool turn resolves the name and wraps an object response.
        let fr = &contents[1]["parts"][0]["functionResponse"];
        assert_eq!(fr["name"], "get_weather");
        assert_eq!(fr["response"]["temp"], 20);
    }
}

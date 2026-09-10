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
const OLLAMA_MODELS: &[&str] = &["qwen2.5:3b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"];

const OLLAMA_BASE_ENV: &str = "AIGATE_OLLAMA_BASE_URL";
const DEFAULT_OLLAMA_BASE: &str = "http://127.0.0.1:11434/v1";

pub fn openai() -> OpenAiCompatible {
    OpenAiCompatible {
        client: super::shared_client(),
        base: "https://api.openai.com/v1".to_string(),
        name: "openai",
        stream_usage: true,
        catalog: OPENAI_MODELS,
    }
}

pub fn mistral() -> OpenAiCompatible {
    OpenAiCompatible {
        client: super::shared_client(),
        base: "https://api.mistral.ai/v1".to_string(),
        name: "mistral",
        stream_usage: false,
        catalog: MISTRAL_MODELS,
    }
}

pub fn ollama(base: String) -> OpenAiCompatible {
    OpenAiCompatible {
        client: super::shared_client(),
        base,
        name: "ollama",
        stream_usage: true,
        catalog: OLLAMA_MODELS,
    }
}

pub fn ollama_base_url() -> Result<String, String> {
    parse_ollama_base(std::env::var(OLLAMA_BASE_ENV).ok().as_deref())
}

fn parse_ollama_base(raw: Option<&str>) -> Result<String, String> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_OLLAMA_BASE.to_string());
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(format!(
            "{OLLAMA_BASE_ENV} is set but empty: unset it to use the default {DEFAULT_OLLAMA_BASE}"
        ));
    }
    match reqwest::Url::parse(raw) {
        Ok(url) if is_bare_http_endpoint(&url) => {
            Ok(url.as_str().trim_end_matches('/').to_string())
        }
        _ => Err(format!(
            "invalid {OLLAMA_BASE_ENV}=\"{raw}\": expected an http(s) URL with no query \
             or fragment, e.g. {DEFAULT_OLLAMA_BASE}"
        )),
    }
}

fn is_bare_http_endpoint(url: &reqwest::Url) -> bool {
    matches!(url.scheme(), "http" | "https")
        && url.query().is_none()
        && url.fragment().is_none()
        && url.username().is_empty()
        && url.password().is_none()
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

    fn accepted(raw: &str) -> String {
        parse_ollama_base(Some(raw)).expect("a usable ollama base url")
    }

    fn rejection_of(raw: &str) -> String {
        parse_ollama_base(Some(raw)).expect_err("a rejected ollama base url")
    }

    #[test]
    fn an_unset_variable_uses_the_loopback_default() {
        assert_eq!(parse_ollama_base(None).unwrap(), DEFAULT_OLLAMA_BASE);
    }

    #[test]
    fn an_empty_or_blank_value_is_rejected_rather_than_silently_defaulted() {
        for raw in ["", " ", "\t", " \n "] {
            let message = rejection_of(raw);
            assert!(message.contains(OLLAMA_BASE_ENV), "{raw:?} -> {message}");
            assert!(message.contains("set but empty"), "{raw:?} -> {message}");
            assert!(
                message.contains(DEFAULT_OLLAMA_BASE),
                "{raw:?} -> {message}"
            );
        }
    }

    #[test]
    fn a_value_that_is_not_a_url_is_rejected() {
        for raw in [
            "pas-une-url",
            "127.0.0.1:11434/v1",
            "//127.0.0.1:11434/v1",
            "http://",
        ] {
            assert!(
                rejection_of(raw).contains(OLLAMA_BASE_ENV),
                "{raw:?} should be rejected"
            );
        }
    }

    #[test]
    fn a_non_http_scheme_is_rejected() {
        for raw in [
            "ftp://127.0.0.1:11434/v1",
            "file:///tmp/models",
            "ws://127.0.0.1:11434/v1",
            "wss://127.0.0.1:11434/v1",
        ] {
            assert!(
                rejection_of(raw).contains(OLLAMA_BASE_ENV),
                "{raw:?} should be rejected"
            );
        }
    }

    #[test]
    fn a_query_a_fragment_or_credentials_are_rejected() {
        for raw in [
            "http://127.0.0.1:11434/v1?debug=1",
            "http://127.0.0.1:11434/v1?",
            "http://127.0.0.1:11434/v1#frag",
            "http://127.0.0.1:11434/v1#",
            "http://user:pass@127.0.0.1:11434/v1",
            "http://user@127.0.0.1:11434/v1",
        ] {
            assert!(
                rejection_of(raw).contains(OLLAMA_BASE_ENV),
                "{raw:?} should be rejected"
            );
        }
    }

    #[test]
    fn a_host_absorbed_from_the_path_is_revealed_by_normalisation() {
        assert_eq!(accepted("http:///v1"), "http://v1");
        assert_eq!(accepted("http:////v1"), "http://v1");
        assert_eq!(accepted("http://./v1"), "http://./v1");
    }

    #[test]
    fn the_accepted_value_is_the_validated_url_not_the_raw_input() {
        assert_eq!(
            accepted("http://LOCALHOST:11434/v1"),
            "http://localhost:11434/v1"
        );
        assert_eq!(accepted("http://localhost:80/v1"), "http://localhost/v1");
        assert_eq!(accepted("https://localhost:443/v1"), "https://localhost/v1");
        assert_eq!(
            accepted("http://127.0.0.1:11434/mon chemin/v1"),
            "http://127.0.0.1:11434/mon%20chemin/v1"
        );
    }

    #[test]
    fn an_accepted_base_never_lets_the_request_path_be_swallowed() {
        let base = accepted("http://127.0.0.1:11434/v1");
        assert_eq!(
            format!("{base}/chat/completions"),
            "http://127.0.0.1:11434/v1/chat/completions"
        );
    }

    #[test]
    fn a_valid_endpoint_loses_its_trailing_slashes() {
        assert_eq!(
            accepted("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            accepted("http://127.0.0.1:11434/v1/"),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            accepted("http://127.0.0.1:11434/v1//"),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            accepted("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434"
        );
    }

    #[test]
    fn surrounding_whitespace_is_ignored() {
        assert_eq!(
            accepted("  http://127.0.0.1:11434/v1  "),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            accepted("\thttp://127.0.0.1:11434/v1\n"),
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn https_and_alternative_hosts_are_accepted() {
        assert_eq!(
            accepted("https://127.0.0.1:11434/v1"),
            "https://127.0.0.1:11434/v1"
        );
        assert_eq!(
            accepted("http://localhost:11434/v1"),
            "http://localhost:11434/v1"
        );
        assert_eq!(accepted("http://[::1]:11434/v1"), "http://[::1]:11434/v1");
        assert_eq!(accepted("http://127.0.0.1:11434"), "http://127.0.0.1:11434");
    }

    #[test]
    fn the_configured_endpoint_reaches_the_provider() {
        let provider = ollama(accepted("http://127.0.0.1:11435/v1/"));
        assert_eq!(provider.name(), "ollama");
        assert!(!provider.catalog().is_empty());
    }

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

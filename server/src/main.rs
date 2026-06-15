//! AIGate HTTP daemon.
//!
//! Exposes an OpenAI-compatible surface so any existing SDK can plug in by
//! changing only its base URL. The target engine is selected by a
//! `provider/model` prefix (e.g. `gemini/gemini-2.0-flash`) or an
//! `X-AI-Provider` header.
//!
//! Failover: the request may carry a `fallbacks` array of further
//! `provider/model` specs, tried in order if the primary fails. API keys are
//! forwarded per request and never stored — the `Authorization` bearer is the
//! default key, and a per-engine key can be supplied via `X-AI-Key-<provider>`
//! (e.g. `X-AI-Key-Claude`).

use std::convert::Infallible;

use axum::{
    extract::Json,
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use futures::StreamExt;
use serde_json::{json, Value};

use aigate_core::{
    chat_failover_with, resolve, split_model, stream_failover_with, Chunk, ChunkStream,
    FailoverError, RetryPolicy, Target, UnifiedRequest, UnifiedResponse,
};

type ApiError = (StatusCode, Json<Value>);

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aigate_server=info,tower_http=info".into()),
        )
        .init();

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/chat/completions", post(chat));

    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    tracing::info!("AIGate listening on http://{addr}");
    axum::serve(listener, app).await.expect("serve");
}

async fn chat(headers: HeaderMap, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let req: UnifiedRequest = serde_json::from_value(body.clone())
        .map_err(|e| err(StatusCode::BAD_REQUEST, &format!("invalid request: {e}")))?;

    // Chain = primary (req.model) followed by any `fallbacks` specs.
    let mut chain = vec![req.model.clone()];
    if let Some(fbs) = body.get("fallbacks").and_then(Value::as_array) {
        chain.extend(fbs.iter().filter_map(|v| v.as_str().map(str::to_string)));
    }

    // Build usable targets; record why any spec was skipped.
    let mut targets = Vec::new();
    let mut skipped: Vec<Value> = Vec::new();
    for spec in &chain {
        let (prefix, model) = split_model(spec);
        let provider_name = prefix.map(str::to_string).or_else(|| {
            headers
                .get("x-ai-provider")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        });
        let Some(provider_name) = provider_name else {
            skipped.push(json!({ "target": spec, "error": "no provider: use 'provider/model' or X-AI-Provider" }));
            continue;
        };
        let Some(provider) = resolve(&provider_name) else {
            skipped.push(json!({ "target": spec, "error": format!("unknown provider: {provider_name}") }));
            continue;
        };
        let Some(key) = key_for(&provider_name, &headers) else {
            skipped.push(json!({ "target": spec, "error": format!("no API key for '{provider_name}' (set Authorization bearer or X-AI-Key-{provider_name})") }));
            continue;
        };
        targets.push(Target {
            provider,
            model: model.to_string(),
            key,
        });
    }

    if targets.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": { "message": "no usable target", "attempts": skipped } })),
        ));
    }

    let policy = retry_policy(&headers);

    if req.stream {
        match stream_failover_with(targets, &req, &policy).await {
            Ok((model, stream)) => Ok(sse_response(model, stream)),
            Err(fe) => Err(failover_error(fe, skipped)),
        }
    } else {
        match chat_failover_with(targets, &req, &policy).await {
            Ok(resp) => Ok(completion_response(resp)),
            Err(fe) => Err(failover_error(fe, skipped)),
        }
    }
}

/// Build the retry policy, allowing `X-AI-Retries` to override max attempts.
fn retry_policy(headers: &HeaderMap) -> RetryPolicy {
    let mut policy = RetryPolicy::default();
    if let Some(n) = headers
        .get("x-ai-retries")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
    {
        policy.max_attempts = n.clamp(1, 10);
    }
    policy
}

fn completion_response(resp: UnifiedResponse) -> Response {
    Json(json!({
        "object": "chat.completion",
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": resp.content },
            "finish_reason": resp.finish_reason,
        }],
        "usage": resp.usage.map(|u| json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })),
    }))
    .into_response()
}

fn sse_response(model: String, upstream: ChunkStream) -> Response {
    let events = upstream
        .map(move |item| -> Result<Event, Infallible> {
            match item {
                Ok(chunk) => Ok(Event::default().data(chunk_envelope(&model, &chunk).to_string())),
                Err(e) => Ok(Event::default()
                    .event("error")
                    .data(json!({ "error": { "message": e.to_string() } }).to_string())),
            }
        })
        // OpenAI clients expect a terminal `data: [DONE]`.
        .chain(futures::stream::once(async {
            Ok(Event::default().data("[DONE]"))
        }));

    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

/// An OpenAI-compatible `chat.completion.chunk` envelope.
fn chunk_envelope(model: &str, chunk: &Chunk) -> Value {
    json!({
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "content": chunk.delta },
            "finish_reason": chunk.finish_reason,
        }],
        "usage": chunk.usage.as_ref().map(|u| json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })),
    })
}

/// Combine config-time skips and runtime attempts into one error payload.
/// An aborted chain (non-retriable client error) maps to 400; otherwise 502.
fn failover_error(fe: FailoverError, skipped: Vec<Value>) -> ApiError {
    let (status, message) = if fe.aborted {
        (StatusCode::BAD_REQUEST, "request rejected (not retriable)")
    } else {
        (StatusCode::BAD_GATEWAY, "all providers failed")
    };
    let mut attempts = skipped;
    attempts.extend(fe.attempts.into_iter().map(|a| {
        json!({ "provider": a.provider, "model": a.model, "tries": a.tries, "error": a.error })
    }));
    (
        status,
        Json(json!({ "error": { "message": message, "attempts": attempts } })),
    )
}

/// Resolve the key for a provider: a per-engine `X-AI-Key-<provider>` header
/// if present, otherwise the `Authorization` bearer token.
fn key_for(provider: &str, headers: &HeaderMap) -> Option<String> {
    let header_name = format!("x-ai-key-{}", provider.to_ascii_lowercase());
    if let Some(v) = headers.get(header_name.as_str()).and_then(|v| v.to_str().ok()) {
        return Some(v.to_string());
    }
    bearer(headers)
}

fn bearer(headers: &HeaderMap) -> Option<String> {
    headers
        .get(AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(str::to_string)
}

fn err(code: StatusCode, msg: &str) -> ApiError {
    (code, Json(json!({ "error": { "message": msg } })))
}

//! AIGate HTTP daemon.
//!
//! Exposes an OpenAI-compatible surface so any existing SDK can plug in by
//! changing only its base URL. The target engine is selected by a
//! `provider/model` prefix (e.g. `gemini/gemini-2.0-flash`) or an
//! `X-AI-Provider` header. The caller's API key (`Authorization: Bearer ...`)
//! is forwarded per request and never stored.

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

use aigate_core::{resolve, split_model, Chunk, Provider, UnifiedRequest};

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

async fn chat(headers: HeaderMap, Json(mut req): Json<UnifiedRequest>) -> Result<Response, ApiError> {
    let key = bearer(&headers)
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "missing 'Authorization: Bearer <key>'"))?;

    // Provider from the model prefix, falling back to the X-AI-Provider header.
    let (prefix, model) = split_model(&req.model);
    let provider_name = prefix
        .map(str::to_string)
        .or_else(|| {
            headers
                .get("x-ai-provider")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        })
        .ok_or_else(|| {
            err(
                StatusCode::BAD_REQUEST,
                "no provider: use 'provider/model' or the X-AI-Provider header",
            )
        })?;
    req.model = model.to_string();

    let provider = resolve(&provider_name)
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, &format!("unknown provider: {provider_name}")))?;

    if req.stream {
        stream_chat(provider, req, key).await
    } else {
        complete_chat(provider, req, key).await
    }
}

async fn complete_chat(
    provider: Box<dyn Provider>,
    req: UnifiedRequest,
    key: String,
) -> Result<Response, ApiError> {
    let resp = provider
        .chat(&req, &key)
        .await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, &e.to_string()))?;

    // OpenAI-compatible response envelope.
    Ok(Json(json!({
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
    .into_response())
}

async fn stream_chat(
    provider: Box<dyn Provider>,
    req: UnifiedRequest,
    key: String,
) -> Result<Response, ApiError> {
    let upstream = provider
        .chat_stream(&req, &key)
        .await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, &e.to_string()))?;

    let model = req.model.clone();
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

    Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response())
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

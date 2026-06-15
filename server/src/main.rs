//! AIGate HTTP daemon.
//!
//! Exposes an OpenAI-compatible surface so any existing SDK can plug in by
//! changing only its base URL. The target engine is selected by a
//! `provider/model` prefix (e.g. `gemini/gemini-2.0-flash`) or an
//! `X-AI-Provider` header. The caller's API key (`Authorization: Bearer ...`)
//! is forwarded per request and never stored.

use axum::{
    extract::Json,
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};

use aigate_core::{resolve, split_model, UnifiedRequest};

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

async fn chat(headers: HeaderMap, Json(mut req): Json<UnifiedRequest>) -> Result<Json<Value>, ApiError> {
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
        // TODO(streaming): bridge each engine's SSE into a unified token stream.
        return Err(err(StatusCode::NOT_IMPLEMENTED, "streaming not implemented yet"));
    }

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
    })))
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

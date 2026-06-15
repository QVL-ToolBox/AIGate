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

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Json, Query, State},
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
    chat_failover_with, estimate_cost, resolve, split_model, stream_failover_with, Chunk,
    ChunkStream, FailoverError, RetryPolicy, Target, UnifiedRequest, UnifiedResponse, Usage,
    PROVIDERS,
};

type ApiError = (StatusCode, Json<Value>);

// ── In-memory usage tracking ────────────────────────────────────────────
// Ephemeral: aggregated per (app, provider, model); reset on restart.

#[derive(Clone, Default)]
struct AppState {
    metrics: Arc<Mutex<Metrics>>,
}

#[derive(Default)]
struct Metrics {
    entries: HashMap<StatKey, Stat>,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct StatKey {
    app: String,
    provider: String,
    model: String,
}

#[derive(Default, Clone)]
struct Stat {
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cost_usd: f64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aigate_server=info,tower_http=info".into()),
        )
        .init();

    let state = AppState::default();
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/usage", get(usage))
        .route("/v1/chat/completions", post(chat))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    tracing::info!("AIGate listening on http://{addr}");
    axum::serve(listener, app).await.expect("serve");
}

/// `GET /v1/models` — OpenAI-compatible model listing aggregated across
/// engines. Model ids are prefixed `provider/model` so they can be passed
/// straight back to `/v1/chat/completions`. When a key is available for an
/// engine (bearer or `X-AI-Key-<provider>`) the live list is fetched; otherwise
/// the built-in catalog is returned. Filter to one engine with `?provider=`.
async fn models(
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let names: Vec<&str> = match params.get("provider") {
        Some(p) => {
            let p = p.as_str();
            if resolve(p).is_none() {
                return Err(err(StatusCode::BAD_REQUEST, &format!("unknown provider: {p}")));
            }
            vec![p]
        }
        None => PROVIDERS.to_vec(),
    };

    let mut data = Vec::new();
    for name in names {
        let Some(provider) = resolve(name) else {
            continue;
        };
        let models = match key_for(name, &headers) {
            // Live listing, falling back to the built-in catalog on any error.
            Some(key) => provider
                .list_models(&key)
                .await
                .unwrap_or_else(|_| provider.catalog()),
            None => provider.catalog(),
        };
        for model in models {
            data.push(json!({
                "id": format!("{}/{}", provider.name(), model),
                "object": "model",
                "owned_by": provider.name(),
            }));
        }
    }

    Ok(Json(json!({ "object": "list", "data": data })))
}

async fn chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
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
    let app = app_id(&headers);

    if req.stream {
        match stream_failover_with(targets, &req, &policy).await {
            Ok((provider, model, stream)) => {
                Ok(sse_response(state, app, provider, model, stream))
            }
            Err(fe) => Err(failover_error(fe, skipped)),
        }
    } else {
        match chat_failover_with(targets, &req, &policy).await {
            Ok((provider, resp)) => {
                record(&state, &app, &provider, &resp.model, resp.usage.as_ref());
                Ok(completion_response(resp))
            }
            Err(fe) => Err(failover_error(fe, skipped)),
        }
    }
}

/// The calling app's identity, from `X-AI-App` (defaults to "unknown").
fn app_id(headers: &HeaderMap) -> String {
    headers
        .get("x-ai-app")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
        .to_string()
}

/// Accumulate one request into the in-memory metrics.
fn record(state: &AppState, app: &str, provider: &str, model: &str, usage: Option<&Usage>) {
    let mut metrics = state.metrics.lock().unwrap();
    let stat = metrics
        .entries
        .entry(StatKey {
            app: app.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
        })
        .or_default();
    stat.requests += 1;
    if let Some(u) = usage {
        stat.prompt_tokens += u.prompt_tokens as u64;
        stat.completion_tokens += u.completion_tokens as u64;
        stat.total_tokens += u.total_tokens as u64;
        if let Some(cost) = estimate_cost(provider, model, u) {
            stat.cost_usd += cost;
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

fn sse_response(
    state: AppState,
    app: String,
    provider: String,
    model: String,
    upstream: ChunkStream,
) -> Response {
    // Capture the usage carried by a late chunk so we can record it when the
    // stream completes.
    let captured: Arc<Mutex<Option<Usage>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let chunk_model = model.clone();

    let events = upstream
        .map(move |item| -> Result<Event, Infallible> {
            match item {
                Ok(chunk) => {
                    if let Some(u) = &chunk.usage {
                        *cap.lock().unwrap() = Some(u.clone());
                    }
                    Ok(Event::default().data(chunk_envelope(&chunk_model, &chunk).to_string()))
                }
                Err(e) => Ok(Event::default()
                    .event("error")
                    .data(json!({ "error": { "message": e.to_string() } }).to_string())),
            }
        })
        // Runs after all chunks are consumed: record usage, then emit [DONE].
        .chain(futures::stream::once(async move {
            let usage = captured.lock().unwrap().clone();
            record(&state, &app, &provider, &model, usage.as_ref());
            Ok(Event::default().data("[DONE]"))
        }));

    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

/// `GET /v1/usage` — aggregated token and cost metrics since startup.
async fn usage(State(state): State<AppState>) -> Json<Value> {
    let metrics = state.metrics.lock().unwrap();
    let mut rows: Vec<(&StatKey, &Stat)> = metrics.entries.iter().collect();
    rows.sort_by(|(a, _), (b, _)| {
        (&a.app, &a.provider, &a.model).cmp(&(&b.app, &b.provider, &b.model))
    });

    let (mut t_req, mut t_prompt, mut t_compl, mut t_total, mut t_cost) = (0u64, 0u64, 0u64, 0u64, 0f64);
    let mut by = Vec::new();
    for (k, s) in rows {
        t_req += s.requests;
        t_prompt += s.prompt_tokens;
        t_compl += s.completion_tokens;
        t_total += s.total_tokens;
        t_cost += s.cost_usd;
        by.push(json!({
            "app": k.app,
            "provider": k.provider,
            "model": k.model,
            "requests": s.requests,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "total_tokens": s.total_tokens,
            "cost_usd": round6(s.cost_usd),
        }));
    }

    Json(json!({
        "totals": {
            "requests": t_req,
            "prompt_tokens": t_prompt,
            "completion_tokens": t_compl,
            "total_tokens": t_total,
            "cost_usd": round6(t_cost),
        },
        "by": by,
    }))
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
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

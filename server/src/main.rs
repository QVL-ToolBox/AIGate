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
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Json, Query, State},
    http::{header::AUTHORIZATION, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
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
    cache: Arc<Mutex<Cache>>,
    /// Valid gateway keys -> app name. Empty disables auth.
    keys: Arc<HashMap<String, String>>,
    rate: Arc<Mutex<RateLimiter>>,
}

// ── Per-identity token-bucket rate limiting ──────────────────────────────
#[derive(Default)]
struct RateLimiter {
    /// Requests per minute; 0 disables limiting.
    rpm: u32,
    buckets: HashMap<String, Bucket>,
}

struct Bucket {
    tokens: f64,
    last_ms: u64,
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Default)]
struct Metrics {
    entries: HashMap<StatKey, Stat>,
}

// ── Opt-in, in-memory response cache (non-streaming only) ────────────────
#[derive(Default)]
struct Cache {
    entries: HashMap<String, CacheEntry>,
    hits: u64,
    misses: u64,
    /// Max entries; 0 = unbounded.
    capacity: usize,
    /// Monotonic access counter for LRU ordering.
    tick: u64,
}

struct CacheEntry {
    response: UnifiedResponse,
    /// Unix epoch seconds at which this entry expires (portable across restarts).
    expires_at: u64,
    /// Access tick for LRU eviction (runtime-only, not persisted).
    last_used: u64,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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

    let rpm = load_rate_limit();
    let state = AppState {
        keys: Arc::new(load_keys()),
        rate: Arc::new(Mutex::new(RateLimiter {
            rpm,
            buckets: HashMap::new(),
        })),
        cache: Arc::new(Mutex::new(Cache {
            capacity: load_cache_max(),
            ..Default::default()
        })),
        ..Default::default()
    };
    if state.keys.is_empty() {
        tracing::warn!("gateway auth DISABLED (set AIGATE_KEYS to require X-AIGate-Key)");
    } else {
        tracing::info!("gateway auth enabled: {} key(s)", state.keys.len());
    }
    if rpm == 0 {
        tracing::warn!("rate limiting DISABLED (set AIGATE_RATE_LIMIT to requests/min)");
    } else {
        tracing::info!("rate limiting enabled: {rpm} req/min per identity");
    }

    let path = state_path();
    if let Some(p) = &path {
        if let Some(snap) = load_snapshot(p) {
            apply_snapshot(&state, snap);
            tracing::info!("restored state from {}", p.display());
        }
        spawn_flusher(state.clone(), p.clone());
    }

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/usage", get(usage))
        .route("/v1/chat/completions", post(chat))
        .with_state(state.clone());

    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    tracing::info!("AIGate listening on http://{addr}");

    let serve = axum::serve(listener, app);
    match path {
        Some(p) => serve
            .with_graceful_shutdown(shutdown_signal(state, p))
            .await
            .expect("serve"),
        None => serve.await.expect("serve"),
    }
}

/// `GET /v1/models` — OpenAI-compatible model listing aggregated across
/// engines. Model ids are prefixed `provider/model` so they can be passed
/// straight back to `/v1/chat/completions`. When a key is available for an
/// engine (bearer or `X-AI-Key-<provider>`) the live list is fetched; otherwise
/// the built-in catalog is returned. Filter to one engine with `?provider=`.
async fn models(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let id = authenticate(&state, &headers)?.unwrap_or_else(|| app_id(&headers));
    check_rate(&state, &id)?;

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
    let authed_app = authenticate(&state, &headers)?;
    // Trust the key-derived app when auth is on; otherwise the X-AI-App header.
    let app = authed_app.unwrap_or_else(|| app_id(&headers));
    check_rate(&state, &app)?;

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
        // Streaming responses are not cached.
        match stream_failover_with(targets, &req, &policy).await {
            Ok((provider, model, stream)) => {
                Ok(sse_response(state, app, provider, model, stream))
            }
            Err(fe) => Err(failover_error(fe, skipped)),
        }
    } else {
        // Opt-in response cache, keyed on the resolved chain + request.
        let cache_ttl = cache_ttl(&headers);
        let cache_key = cache_ttl.map(|_| cache_key_of(&targets, &req));
        if let Some(key) = &cache_key {
            if let Some(resp) = cache_get(&state, key) {
                return Ok(completion_response(&resp, true));
            }
        }

        match chat_failover_with(targets, &req, &policy).await {
            Ok((provider, resp)) => {
                record(&state, &app, &provider, &resp.model, resp.usage.as_ref());
                if let (Some(ttl), Some(key)) = (cache_ttl, cache_key) {
                    cache_put(&state, key, &resp, ttl);
                }
                Ok(completion_response(&resp, false))
            }
            Err(fe) => Err(failover_error(fe, skipped)),
        }
    }
}

/// Parse the `X-AI-Cache` header into a TTL. Absent/`off`/`0` disables caching;
/// `on`/`true` uses 300s; a bare integer sets the TTL in seconds.
fn cache_ttl(headers: &HeaderMap) -> Option<Duration> {
    let v = headers.get("x-ai-cache")?.to_str().ok()?.trim().to_string();
    if v.eq_ignore_ascii_case("off") || v == "0" || v.is_empty() {
        return None;
    }
    let secs = if v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true") {
        300
    } else {
        v.parse::<u64>().ok()?
    };
    Some(Duration::from_secs(secs))
}

/// Cache key: the resolved target chain plus the request fields that affect the
/// answer. Images are included verbatim (large but correct).
fn cache_key_of(targets: &[Target], req: &UnifiedRequest) -> String {
    let chain: Vec<String> = targets
        .iter()
        .map(|t| format!("{}/{}", t.provider.name(), t.model))
        .collect();
    format!(
        "{}|{}|{}|{}|{:?}|{:?}",
        chain.join(","),
        serde_json::to_string(&req.messages).unwrap_or_default(),
        serde_json::to_string(&req.tools).unwrap_or_default(),
        serde_json::to_string(&req.tool_choice).unwrap_or_default(),
        req.temperature,
        req.max_tokens,
    )
}

fn cache_get(state: &AppState, key: &str) -> Option<UnifiedResponse> {
    let mut cache = state.cache.lock().unwrap();
    let now = now_unix();
    let tick = cache.tick + 1;
    let hit = match cache.entries.get_mut(key) {
        Some(e) if e.expires_at > now => {
            e.last_used = tick; // LRU touch
            Some(e.response.clone())
        }
        _ => None,
    };
    if hit.is_some() {
        cache.tick = tick;
        cache.hits += 1;
    } else {
        cache.entries.remove(key);
        cache.misses += 1;
    }
    hit
}

fn cache_put(state: &AppState, key: String, resp: &UnifiedResponse, ttl: Duration) {
    let mut cache = state.cache.lock().unwrap();
    cache.tick += 1;
    let last_used = cache.tick;
    cache.entries.insert(
        key,
        CacheEntry {
            response: resp.clone(),
            expires_at: now_unix() + ttl.as_secs(),
            last_used,
        },
    );
    enforce_capacity(&mut cache);
}

/// Drop expired entries, then evict least-recently-used ones until within cap.
fn enforce_capacity(cache: &mut Cache) {
    if cache.capacity == 0 || cache.entries.len() <= cache.capacity {
        return;
    }
    let now = now_unix();
    cache.entries.retain(|_, e| e.expires_at > now);
    while cache.entries.len() > cache.capacity {
        let lru = cache
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone());
        match lru {
            Some(k) => {
                cache.entries.remove(&k);
            }
            None => break,
        }
    }
}

/// Parse `AIGATE_KEYS` (`key:app,key:app,...`) into a key -> app map.
/// An entry without `:app` is labelled `default`.
fn load_keys() -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Ok(raw) = std::env::var("AIGATE_KEYS") {
        for entry in raw.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let (key, app) = match entry.split_once(':') {
                Some((k, a)) => (k.trim(), a.trim()),
                None => (entry, "default"),
            };
            if !key.is_empty() {
                map.insert(key.to_string(), app.to_string());
            }
        }
    }
    map
}

/// Authenticate the caller against the configured gateway keys.
/// - auth disabled (no keys) -> `Ok(None)` (caller falls back to `X-AI-App`)
/// - valid `X-AIGate-Key`     -> `Ok(Some(app))` (trusted app identity)
/// - missing/invalid key      -> `Err(401)`
fn authenticate(state: &AppState, headers: &HeaderMap) -> Result<Option<String>, ApiError> {
    if state.keys.is_empty() {
        return Ok(None);
    }
    let presented = headers.get("x-aigate-key").and_then(|v| v.to_str().ok());
    match presented.and_then(|k| state.keys.get(k)) {
        Some(app) => Ok(Some(app.clone())),
        None => Err(err(StatusCode::UNAUTHORIZED, "invalid or missing X-AIGate-Key")),
    }
}

/// Requests-per-minute limit from `AIGATE_RATE_LIMIT` (0/absent = disabled).
fn load_rate_limit() -> u32 {
    std::env::var("AIGATE_RATE_LIMIT")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

/// Token-bucket check for one identity. `Ok(())` to proceed, `Err(429)` when
/// the bucket is empty (with a `retry_after` hint in the body).
fn check_rate(state: &AppState, identity: &str) -> Result<(), ApiError> {
    let mut rl = state.rate.lock().unwrap();
    if rl.rpm == 0 {
        return Ok(());
    }
    let capacity = rl.rpm as f64;
    let per_ms = capacity / 60_000.0;
    let now = now_millis();
    let bucket = rl.buckets.entry(identity.to_string()).or_insert(Bucket {
        tokens: capacity,
        last_ms: now,
    });
    let elapsed = now.saturating_sub(bucket.last_ms) as f64;
    bucket.tokens = (bucket.tokens + elapsed * per_ms).min(capacity);
    bucket.last_ms = now;

    if bucket.tokens >= 1.0 {
        bucket.tokens -= 1.0;
        Ok(())
    } else {
        let retry_after = ((1.0 - bucket.tokens) / per_ms / 1000.0).ceil() as u64;
        Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({ "error": {
                "message": "rate limit exceeded",
                "retry_after": retry_after.max(1),
            } })),
        ))
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

fn completion_response(resp: &UnifiedResponse, cached: bool) -> Response {
    let mut message = serde_json::Map::new();
    message.insert("role".into(), json!("assistant"));
    // OpenAI sets content to null when the turn is only tool calls.
    if resp.content.is_empty() && resp.tool_calls.is_some() {
        message.insert("content".into(), Value::Null);
    } else {
        message.insert("content".into(), json!(resp.content));
    }
    if let Some(calls) = &resp.tool_calls {
        message.insert(
            "tool_calls".into(),
            serde_json::to_value(calls).unwrap_or(Value::Null),
        );
    }

    let body = json!({
        "object": "chat.completion",
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": resp.finish_reason,
        }],
        "usage": resp.usage.as_ref().map(|u| json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })),
    });

    let mut response = Json(body).into_response();
    response.headers_mut().insert(
        "x-ai-cache",
        HeaderValue::from_static(if cached { "HIT" } else { "MISS" }),
    );
    response
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
async fn usage(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    let id = authenticate(&state, &headers)?.unwrap_or_else(|| app_id(&headers));
    check_rate(&state, &id)?;

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

    let cache = state.cache.lock().unwrap();

    Ok(Json(json!({
        "totals": {
            "requests": t_req,
            "prompt_tokens": t_prompt,
            "completion_tokens": t_compl,
            "total_tokens": t_total,
            "cost_usd": round6(t_cost),
        },
        "cache": {
            "hits": cache.hits,
            "misses": cache.misses,
            "entries": cache.entries.len(),
            "capacity": cache.capacity,
        },
        "by": by,
    })))
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// An OpenAI-compatible `chat.completion.chunk` envelope.
fn chunk_envelope(model: &str, chunk: &Chunk) -> Value {
    let mut delta = serde_json::Map::new();
    if !chunk.delta.is_empty() {
        delta.insert("content".into(), json!(chunk.delta));
    }
    if let Some(calls) = &chunk.tool_calls {
        let arr: Vec<Value> = calls
            .iter()
            .map(|tc| {
                let mut function = serde_json::Map::new();
                if let Some(name) = &tc.name {
                    function.insert("name".into(), json!(name));
                }
                function.insert("arguments".into(), json!(tc.arguments));

                let mut call = serde_json::Map::new();
                call.insert("index".into(), json!(tc.index));
                if let Some(id) = &tc.id {
                    call.insert("id".into(), json!(id));
                    call.insert("type".into(), json!("function"));
                }
                call.insert("function".into(), Value::Object(function));
                Value::Object(call)
            })
            .collect();
        delta.insert("tool_calls".into(), json!(arr));
    }

    json!({
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": Value::Object(delta),
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

// ── Persistence (JSON snapshot of metrics + cache) ───────────────────────

#[derive(Serialize, Deserialize, Default)]
struct Snapshot {
    #[serde(default)]
    metrics: Vec<MetricRow>,
    #[serde(default)]
    cache_hits: u64,
    #[serde(default)]
    cache_misses: u64,
    #[serde(default)]
    cache: Vec<CacheRow>,
}

#[derive(Serialize, Deserialize)]
struct MetricRow {
    app: String,
    provider: String,
    model: String,
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cost_usd: f64,
}

#[derive(Serialize, Deserialize)]
struct CacheRow {
    key: String,
    expires_at: u64,
    response: UnifiedResponse,
}

/// Max cache entries from `AIGATE_CACHE_MAX` (default 1000; 0 = unbounded).
fn load_cache_max() -> usize {
    std::env::var("AIGATE_CACHE_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000)
}

/// Where to persist state. `AIGATE_STATE_FILE` overrides the default; set it to
/// `off`/`none`/empty to disable persistence.
fn state_path() -> Option<PathBuf> {
    match std::env::var("AIGATE_STATE_FILE") {
        Ok(v) if v.is_empty() || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("none") => {
            None
        }
        Ok(v) => Some(v.into()),
        Err(_) => Some("aigate-state.json".into()),
    }
}

fn take_snapshot(state: &AppState) -> Snapshot {
    let metrics = state.metrics.lock().unwrap();
    let cache = state.cache.lock().unwrap();
    Snapshot {
        metrics: metrics
            .entries
            .iter()
            .map(|(k, s)| MetricRow {
                app: k.app.clone(),
                provider: k.provider.clone(),
                model: k.model.clone(),
                requests: s.requests,
                prompt_tokens: s.prompt_tokens,
                completion_tokens: s.completion_tokens,
                total_tokens: s.total_tokens,
                cost_usd: s.cost_usd,
            })
            .collect(),
        cache_hits: cache.hits,
        cache_misses: cache.misses,
        cache: cache
            .entries
            .iter()
            .map(|(key, e)| CacheRow {
                key: key.clone(),
                expires_at: e.expires_at,
                response: e.response.clone(),
            })
            .collect(),
    }
}

fn apply_snapshot(state: &AppState, snap: Snapshot) {
    let now = now_unix();
    {
        let mut metrics = state.metrics.lock().unwrap();
        for r in snap.metrics {
            metrics.entries.insert(
                StatKey {
                    app: r.app,
                    provider: r.provider,
                    model: r.model,
                },
                Stat {
                    requests: r.requests,
                    prompt_tokens: r.prompt_tokens,
                    completion_tokens: r.completion_tokens,
                    total_tokens: r.total_tokens,
                    cost_usd: r.cost_usd,
                },
            );
        }
    }
    {
        let mut cache = state.cache.lock().unwrap();
        cache.hits = snap.cache_hits;
        cache.misses = snap.cache_misses;
        for r in snap.cache {
            // Drop entries that already expired while we were down.
            if r.expires_at > now {
                cache.tick += 1;
                let last_used = cache.tick;
                cache.entries.insert(
                    r.key,
                    CacheEntry {
                        response: r.response,
                        expires_at: r.expires_at,
                        last_used,
                    },
                );
            }
        }
        // Honor the capacity in case the file holds more than the current cap.
        enforce_capacity(&mut cache);
    }
}

fn load_snapshot(path: &Path) -> Option<Snapshot> {
    let data = std::fs::read(path).ok()?;
    serde_json::from_slice(&data).ok()
}

/// Write atomically: serialize to a temp file, then rename over the target.
fn write_snapshot(path: &Path, snap: &Snapshot) -> std::io::Result<()> {
    let json = serde_json::to_vec_pretty(snap)?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)
}

/// Periodically flush state to disk in the background.
fn spawn_flusher(state: AppState, path: PathBuf) {
    let secs = std::env::var("AIGATE_FLUSH_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(15)
        .max(1);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(secs));
        loop {
            interval.tick().await;
            if let Err(e) = write_snapshot(&path, &take_snapshot(&state)) {
                tracing::warn!("state flush failed: {e}");
            }
        }
    });
}

/// Resolve on Ctrl+C; saves a final snapshot before the server stops.
async fn shutdown_signal(state: AppState, path: PathBuf) {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down — saving state to {}", path.display());
    if let Err(e) = write_snapshot(&path, &take_snapshot(&state)) {
        tracing::warn!("final state flush failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aigate_core::{resolve, Content, Message, Role};

    fn headers(value: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(v) = value {
            h.insert("x-ai-cache", HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn cache_ttl_parsing() {
        assert_eq!(cache_ttl(&headers(None)), None);
        assert_eq!(cache_ttl(&headers(Some("off"))), None);
        assert_eq!(cache_ttl(&headers(Some("0"))), None);
        assert_eq!(cache_ttl(&headers(Some("on"))), Some(Duration::from_secs(300)));
        assert_eq!(cache_ttl(&headers(Some("60"))), Some(Duration::from_secs(60)));
    }

    fn target(model: &str) -> Target {
        Target {
            provider: resolve("openai").unwrap(),
            model: model.to_string(),
            key: "k".to_string(),
        }
    }

    fn req(text: &str, temperature: Option<f32>) -> UnifiedRequest {
        UnifiedRequest {
            model: "openai/gpt-4o".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some(Content::Text(text.into())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            temperature,
            max_tokens: None,
            stream: false,
            tools: None,
            tool_choice: None,
        }
    }

    #[test]
    fn cache_key_is_stable_and_sensitive() {
        let t = vec![target("gpt-4o")];
        let a = cache_key_of(&t, &req("hi", None));
        let b = cache_key_of(&t, &req("hi", None));
        assert_eq!(a, b, "identical requests share a key");

        let c = cache_key_of(&t, &req("hi", Some(0.9)));
        assert_ne!(a, c, "temperature changes the key");

        let d = cache_key_of(&t, &req("bye", None));
        assert_ne!(a, d, "different content changes the key");
    }

    #[test]
    fn auth_enforced_only_when_keys_configured() {
        // Auth disabled: no keys -> Ok(None) regardless of header.
        let open = AppState::default();
        assert!(matches!(authenticate(&open, &HeaderMap::new()), Ok(None)));

        // Auth enabled.
        let mut keys = HashMap::new();
        keys.insert("secret".to_string(), "web".to_string());
        let secured = AppState {
            keys: Arc::new(keys),
            ..Default::default()
        };

        // Missing key -> 401.
        let (code, _) = authenticate(&secured, &HeaderMap::new()).unwrap_err();
        assert_eq!(code, StatusCode::UNAUTHORIZED);

        // Wrong key -> 401.
        let mut bad = HeaderMap::new();
        bad.insert("x-aigate-key", HeaderValue::from_static("nope"));
        assert!(authenticate(&secured, &bad).is_err());

        // Valid key -> app identity from the key.
        let mut good = HeaderMap::new();
        good.insert("x-aigate-key", HeaderValue::from_static("secret"));
        assert_eq!(authenticate(&secured, &good).unwrap(), Some("web".to_string()));
    }

    #[test]
    fn rate_limit_allows_burst_then_blocks() {
        let state = AppState {
            rate: Arc::new(Mutex::new(RateLimiter {
                rpm: 2,
                buckets: HashMap::new(),
            })),
            ..Default::default()
        };
        assert!(check_rate(&state, "app").is_ok());
        assert!(check_rate(&state, "app").is_ok());
        let third = check_rate(&state, "app");
        assert_eq!(third.unwrap_err().0, StatusCode::TOO_MANY_REQUESTS);
        // A different identity has its own bucket.
        assert!(check_rate(&state, "other").is_ok());
    }

    #[test]
    fn rate_limit_disabled_when_zero() {
        let state = AppState::default(); // rpm = 0
        for _ in 0..50 {
            assert!(check_rate(&state, "x").is_ok());
        }
    }

    #[test]
    fn cache_evicts_least_recently_used() {
        let state = AppState {
            cache: Arc::new(Mutex::new(Cache {
                capacity: 2,
                ..Default::default()
            })),
            ..Default::default()
        };
        let r = UnifiedResponse {
            content: "x".into(),
            tool_calls: None,
            model: "m".into(),
            finish_reason: None,
            usage: None,
        };
        cache_put(&state, "k1".into(), &r, Duration::from_secs(300));
        cache_put(&state, "k2".into(), &r, Duration::from_secs(300));
        // Touch k1 so k2 becomes the least-recently-used.
        assert!(cache_get(&state, "k1").is_some());
        cache_put(&state, "k3".into(), &r, Duration::from_secs(300));

        let cache = state.cache.lock().unwrap();
        assert_eq!(cache.entries.len(), 2);
        assert!(cache.entries.contains_key("k1"));
        assert!(cache.entries.contains_key("k3"));
        assert!(!cache.entries.contains_key("k2"), "k2 was LRU, evicted");
    }

    #[test]
    fn snapshot_round_trips_metrics_and_cache() {
        use aigate_core::Usage;

        let s = AppState::default();
        record(
            &s,
            "app1",
            "openai",
            "gpt-4o",
            Some(&Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        );
        let resp = UnifiedResponse {
            content: "hi".into(),
            tool_calls: None,
            model: "gpt-4o".into(),
            finish_reason: Some("stop".into()),
            usage: None,
        };
        cache_put(&s, "k1".into(), &resp, Duration::from_secs(300));
        let _ = cache_get(&s, "absent"); // bump misses

        // Serialize and restore into a fresh state.
        let json = serde_json::to_vec(&take_snapshot(&s)).unwrap();
        let restored: Snapshot = serde_json::from_slice(&json).unwrap();
        let s2 = AppState::default();
        apply_snapshot(&s2, restored);

        let metrics = s2.metrics.lock().unwrap();
        let stat = metrics
            .entries
            .get(&StatKey {
                app: "app1".into(),
                provider: "openai".into(),
                model: "gpt-4o".into(),
            })
            .expect("metric restored");
        assert_eq!(stat.requests, 1);
        assert_eq!(stat.total_tokens, 15);
        drop(metrics);

        let cache = s2.cache.lock().unwrap();
        assert!(cache.entries.contains_key("k1"), "cache entry restored");
        assert_eq!(cache.misses, 1, "cache counters restored");
    }
}

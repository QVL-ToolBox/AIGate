# AIGate

A provider-agnostic AI gateway, in Rust. Run one daemon, point any of your apps
at it, and talk to **OpenAI, Gemini, Claude, or Mistral** through a single
OpenAI-compatible API.

- **One wire format.** Apps speak the OpenAI chat API. Any existing SDK works by
  changing only its base URL.
- **Any engine.** The gateway translates to each provider's native format behind
  the scenes. Adding an engine = implementing one trait.
- **Stateless keys.** The caller's API key is forwarded per request and never
  stored by the gateway.

---

## Architecture

```
your app ──(OpenAI-format HTTP)──▶  AIGate daemon  ──▶  OpenAI / Gemini / Claude / Mistral
```

Two crates:

- **`aigate-core`** — the `Provider` trait, the unified request/response types,
  and one adapter per engine. Usable as a library on its own.
- **`aigate-server`** — the axum HTTP daemon built on top of `aigate-core`.

## Run

```bash
cargo run -p aigate-server
# AIGate listening on http://0.0.0.0:8080
```

## Quick reference

The fast, complete contract — for humans skimming and agents integrating. Detail
and examples for each item are in the sections further down.

### Endpoints

| Method & path              | Auth* | Purpose                                          |
|----------------------------|-------|--------------------------------------------------|
| `POST /v1/chat/completions`| yes   | Chat (stream/non-stream, tools, images, failover, cache) |
| `GET  /v1/models`          | yes   | List models; optional `?provider=<name>`         |
| `GET  /v1/usage`           | yes   | Usage/cost metrics + cache stats                 |
| `GET  /health`             | no    | Liveness probe → `ok`                            |

\* Auth is enforced only when `AIGATE_KEYS` is set; otherwise all routes are open.

### Request headers

| Header                         | When            | Meaning                                                        |
|--------------------------------|-----------------|----------------------------------------------------------------|
| `Content-Type: application/json` | POST           | Required for the chat endpoint.                                |
| `Authorization: Bearer <key>`  | usually         | Provider API key; default key for every targeted engine.       |
| `X-AI-Key-<Provider>: <key>`   | optional        | Per-engine provider key (e.g. `X-AI-Key-Claude`). Overrides the bearer for that engine — required for **cross-engine failover**. |
| `X-AIGate-Key: <key>`          | if auth on      | Gateway key (separate from the provider key).                  |
| `X-AI-Provider: <name>`        | optional        | Engine to use when `model` has no `provider/` prefix.          |
| `X-AI-App: <name>`             | optional        | App label for usage metrics (ignored when auth is on — the key decides the identity). |
| `X-AI-Cache: on \| <seconds> \| off` | optional  | Enable the response cache and set its TTL (`on` = 300s).       |
| `X-AI-Retries: <1-10>`         | optional        | Per-target retry attempts (default 3).                         |

A request needs **either** an `Authorization` bearer **or** an `X-AI-Key-<Provider>`
for every engine it targets. Response header `X-AI-Cache: HIT|MISS` is set on
non-streaming chat replies.

### Environment variables (all optional)

| Var                 | Default             | Meaning                                            |
|---------------------|---------------------|----------------------------------------------------|
| `AIGATE_KEYS`       | unset (auth off)    | `key:app,key:app,…` — enables gateway auth.         |
| `AIGATE_RATE_LIMIT` | `0` (off)           | Requests/min per identity (token bucket).           |
| `AIGATE_CACHE_MAX`  | `1000`              | Max cache entries (`0` = unbounded, LRU eviction).  |
| `AIGATE_STATE_FILE` | `aigate-state.json` | Persistence path (`off`/`none` disables).           |
| `AIGATE_FLUSH_SECS` | `15`                | Persistence flush interval (seconds).               |
| `RUST_LOG`          | `aigate_server=info`| Tracing filter.                                     |

### Model id & request body

- **Model id**: `provider/model`, e.g. `openai/gpt-4o-mini`,
  `gemini/gemini-2.0-flash`, `claude/claude-sonnet-4-5`,
  `mistral/mistral-small-latest`. Aliases: `anthropic/`→claude, `google/`→gemini.
- **Body** is the OpenAI chat-completions schema plus one extension:
  `model`, `messages`, `stream`, `temperature`, `max_tokens`, `tools`,
  `tool_choice`, and **`fallbacks`** (array of `provider/model`, AIGate-specific).
- **Response** is the OpenAI `chat.completion` (or `chat.completion.chunk` SSE)
  schema. Errors are `{"error": {"message": ..., "attempts"?: [...]}}`.

### Using an existing OpenAI SDK

Point any OpenAI client at the gateway and prefix the model:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="<your-provider-key>")
client.chat.completions.create(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## Call it

The target engine is chosen by a `provider/model` prefix (or an `X-AI-Provider`
header). The `Authorization` bearer token is your provider API key.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{ "role": "user", "content": "Hello in one word" }]
  }'
```

Switch engine by changing the prefix and the key:

```bash
-d '{ "model": "gemini/gemini-2.0-flash",  "messages": [...] }'   # + Gemini key
-d '{ "model": "claude/claude-sonnet-4-6", "messages": [...] }'   # + Anthropic key
-d '{ "model": "mistral/mistral-large-latest", "messages": [...] }'
```

## List models

```bash
curl http://localhost:8080/v1/models                      # all engines, prefixed ids
curl http://localhost:8080/v1/models?provider=claude      # filter to one engine
```

Returns an OpenAI-compatible `{ "object": "list", "data": [...] }` where each
`id` is `provider/model` — ready to drop straight into `model`. When a key is
available for an engine (bearer or `X-AI-Key-<provider>`), its live catalog is
fetched; otherwise a built-in fallback catalog is returned (no key needed).

## Tool calling

Declare tools in the OpenAI format; AIGate translates them to each engine's
native shape (OpenAI/Mistral passthrough, Claude `tool_use`/`tool_result`
blocks, Gemini `functionDeclarations`/`functionResponse`) and normalizes the
model's tool calls back to OpenAI `tool_calls`. AIGate does **not** execute
tools — your app runs the loop and sends results back as `tool` messages.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{ "role": "user", "content": "Weather in Paris?" }],
    "tools": [{ "type": "function", "function": {
      "name": "get_weather", "description": "Get weather",
      "parameters": { "type": "object",
        "properties": { "city": { "type": "string" } }, "required": ["city"] }
    }}],
    "tool_choice": "auto"
  }'
```

The assistant turn comes back with `tool_calls` and `finish_reason:
"tool_calls"`. Append it plus a `{"role":"tool","tool_call_id":"…","content":"…"}`
message and call again. `tool_choice` accepts `auto` / `required` / `none` / a
specific `{"type":"function","function":{"name":"…"}}`. Gemini doesn't return
call ids, so AIGate synthesizes them.

Tool calling also works over **streaming** (`"stream": true`): tool calls are
emitted as OpenAI-compatible `delta.tool_calls` fragments (id+name on the first
fragment, arguments streamed as partial JSON), so a standard OpenAI streaming
client reassembles them as usual.

## Multimodal (image inputs)

Send images with the OpenAI content-parts shape; AIGate translates them to each
engine (Claude `image` blocks, Gemini `inlineData`/`fileData`). Both `data:` URLs
(base64) and remote `https://` URLs are accepted.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{ "role": "user", "content": [
      { "type": "text", "text": "What is in this image?" },
      { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBORw0..." } }
    ]}]
  }'
```

`data:` URLs work everywhere (the portable path). Remote URLs work natively on
OpenAI and Claude; for **Gemini** they map to `fileData` best-effort and may not
be fetched for arbitrary web URLs — prefer base64 data URLs for Gemini.

## Response cache

Opt in per request with the `X-AI-Cache` header — `on` (300s TTL) or a number of
seconds. Identical requests (same target chain, messages, tools, temperature,
`max_tokens`) then return the stored answer without an upstream call. The
response carries `X-AI-Cache: HIT` or `MISS`.

```bash
curl -i http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "X-AI-Cache: 600" -H "Content-Type: application/json" \
  -d '{ "model": "openai/gpt-4o-mini",
        "messages": [{ "role": "user", "content": "Capital of France?" }] }'
```

Cache hits add **no** tokens or cost to `/v1/usage` (no upstream call) and are
counted separately under `cache` (`hits`/`misses`/`entries`/`capacity`). Caching
is in-memory and applies to **non-streaming** requests only. The entry count is
bounded by `AIGATE_CACHE_MAX` (default 1000; `0` = unbounded) with **LRU
eviction** — expired entries are dropped first, then the least-recently-used.

## Usage & cost tracking

Every **successful** request is aggregated in memory per `(app, provider,
model)`. Identify the calling app with an `X-AI-App: <name>` header (defaults to
`unknown`). Read the running totals:

```bash
curl http://localhost:8080/v1/usage
```

```json
{
  "totals": { "requests": 12, "prompt_tokens": 4200, "completion_tokens": 1800,
              "total_tokens": 6000, "cost_usd": 0.0345 },
  "by": [ { "app": "demo-app", "provider": "openai", "model": "gpt-4o-mini",
            "requests": 12, "prompt_tokens": 4200, "completion_tokens": 1800,
            "total_tokens": 6000, "cost_usd": 0.0345 } ]
}
```

Cost uses a built-in price table (USD per 1M tokens, matched by model prefix);
unpriced models still count tokens with `cost_usd` unchanged. Metrics **survive
restarts** (see [Persistence](#persistence)). For streaming, usage is recorded
from the final chunk that carries it (Claude streaming omits it).

## Gateway authentication

By default AIGate is **open** (anyone who can reach it can use it). Set
`AIGATE_KEYS` to require a gateway key on every `/v1/*` request (`/health` stays
open). Keys are distinct from provider keys: the **gateway** key goes in
`X-AIGate-Key`, the **provider** key stays in `Authorization: Bearer`.

```bash
# key:app pairs, comma-separated
export AIGATE_KEYS="s3cret-web:web-app,s3cret-mob:mobile-app"
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "X-AIGate-Key: s3cret-web" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "model": "openai/gpt-4o-mini", "messages": [{ "role": "user", "content": "hi" }] }'
```

When auth is on, the app identity for `/v1/usage` is taken from the **key**
(trusted), not the self-declared `X-AI-App` header. Keys come from the
environment and are never persisted.

## Rate limiting

Set `AIGATE_RATE_LIMIT` to a requests-per-minute cap, applied **per identity**
(the authenticated app when auth is on, otherwise the `X-AI-App` header). It uses
a token bucket — bursts up to the limit, then sustains `rpm`/min. `0`/absent
disables it.

```bash
export AIGATE_RATE_LIMIT=60   # 60 req/min per app
```

Over the limit, `/v1/*` return `429` with a `retry_after` (seconds) hint.
Buckets are in-memory and reset on restart.

## Persistence

Usage metrics and the response cache are snapshotted to a JSON file so they
survive restarts. State is flushed periodically and on graceful shutdown
(Ctrl+C), and loaded on startup (expired cache entries are dropped).

| Env var             | Default              | Meaning                                  |
|---------------------|----------------------|------------------------------------------|
| `AIGATE_STATE_FILE` | `aigate-state.json`  | State file path; `off`/`none` disables it |
| `AIGATE_FLUSH_SECS` | `15`                 | Background flush interval (seconds)       |

Writes are atomic (temp file + rename). Persistence is best-effort: a hard kill
may lose up to one flush interval.

## Supported engines

| Provider | Prefix              | Auth                         |
|----------|---------------------|------------------------------|
| OpenAI   | `openai/`           | `Authorization: Bearer`      |
| Mistral  | `mistral/`          | `Authorization: Bearer`      |
| Claude   | `claude/`, `anthropic/` | `x-api-key` (handled internally) |
| Gemini   | `gemini/`, `google/`    | API key query param (handled internally) |

## Streaming

Set `"stream": true` to receive an OpenAI-compatible SSE stream
(`chat.completion.chunk` events terminated by `data: [DONE]`). Each engine's
native stream format (OpenAI/Mistral delta chunks, Anthropic named events,
Gemini `streamGenerateContent`) is normalized behind the same surface.

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "model": "openai/gpt-4o-mini", "stream": true,
        "messages": [{ "role": "user", "content": "Count to 5" }] }'
```

## Failover

Add a `fallbacks` array of further `provider/model` targets; they are tried in
order when the primary fails (works for both streaming and non-streaming).
Because each engine has its own key, supply a per-engine key with
`X-AI-Key-<provider>` — the `Authorization` bearer is the default when no
engine-specific header is present.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "X-AI-Key-Claude: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "fallbacks": ["claude/claude-sonnet-4-6"],
    "messages": [{ "role": "user", "content": "Hello" }]
  }'
```

If every target fails, the response is `502` with an `attempts` array detailing
each provider's error and how many `tries` it took. For streaming, failover
covers stream *establishment*; an error after the first bytes are sent surfaces
as-is.

## Retry policy

Errors are classified to avoid both wasted retries and pointless failover:

| Class       | Examples                          | Behaviour                                   |
|-------------|-----------------------------------|---------------------------------------------|
| Transient   | 429, 5xx, 408/409/425, network    | retry **same** target (exp. backoff), then fail over |
| Failover    | 401, 403, 404, empty, unsupported | skip to the **next** target (no same-target retry)   |
| Abort       | 400, 422 (malformed request)      | **stop** the chain → `400` (no target would accept it) |

Per-target attempts default to 3; override with the `X-AI-Retries: <1-10>`
header.

## Roadmap

- [x] **Streaming** (SSE token-by-token) across all four engines
- [x] **Provider failover / fallback** with per-engine keys
- [x] **Smart retry policy** (transient retry + backoff, abort on client errors)
- [x] **`/v1/models`** listing (live with key, built-in catalog without)
- [x] **Token & cost tracking** per app (`/v1/usage`, in-memory)
- [x] **Tool calling** across all four engines (non-streaming and streaming)
- [x] **Multimodal image inputs** (base64 + remote URLs)
- [x] **Response cache** (opt-in, non-streaming)
- [x] **Persistence** of metrics & cache (JSON snapshot, survives restart)
- [x] **Gateway authentication** (per-app keys via `X-AIGate-Key`)
- [x] **Rate limiting** per identity (token bucket, `AIGATE_RATE_LIMIT`)
- [x] **Cache bound + LRU eviction** (`AIGATE_CACHE_MAX`)
- [ ] Auto-fetch remote images to base64 for Gemini
- [ ] Audio/document inputs

## Author

Developed by Martin QUEVAL

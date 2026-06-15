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

## Supported engines

| Provider | Prefix              | Auth                         |
|----------|---------------------|------------------------------|
| OpenAI   | `openai/`           | `Authorization: Bearer`      |
| Mistral  | `mistral/`          | `Authorization: Bearer`      |
| Claude   | `claude/`, `anthropic/` | `x-api-key` (handled internally) |
| Gemini   | `gemini/`, `google/`    | API key query param (handled internally) |

## Roadmap

- [ ] **Streaming** (SSE token-by-token) — the next milestone; the trait and
      endpoint already reserve the seam.
- [ ] Provider failover / fallback
- [ ] `/v1/models` listing
- [ ] Token & cost tracking per app
- [ ] Tool calling and multimodal (lowest-common-denominator mapping)
- [ ] Response caching

## Author

Developed by Martin QUEVAL

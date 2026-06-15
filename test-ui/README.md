# AIGate — Test Console

A single-file browser client to exercise the gateway by hand.

## Use

1. Start the gateway:
   ```bash
   cargo run -p aigate-server
   ```
2. Open `test-ui/index.html` in a browser (double-click, or serve it):
   ```bash
   # optional: serve over http instead of file://
   python -m http.server 5500   # then open http://localhost:5500/test-ui/
   ```
3. Fill in:
   - **Gateway base URL** — `http://localhost:8080` by default.
   - **Engine** — provider + model (the model field auto-fills a default per provider).
   - **Provider API key** — your OpenAI / Gemini / Claude / Mistral key (sent as `Authorization: Bearer`).
   - **Gateway key** — only if the server was started with `AIGATE_KEYS`.
   - Optional: app name (`X-AI-App`), system prompt, **Stream** and **Cache** toggles.
4. Type a message and hit Enter. The buttons also test `/health`, `/v1/models`, and `/v1/usage`.

## Notes

- The gateway enables **permissive CORS** so the browser can call it cross-origin.
  Tighten the allowed origins before any non-local deployment.
- Streaming uses the SSE response and renders tokens as they arrive; tool-call
  deltas are shown raw.
- Keys are only kept in the page (never stored); they travel straight to the
  gateway, which forwards them to the provider without persisting them.

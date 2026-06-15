//! Provider registry and the adapter implementations.

mod claude;
mod gemini;
mod openai;

use crate::provider::Provider;

/// Resolve a provider name into its adapter. Aliases are accepted.
pub fn resolve(name: &str) -> Option<Box<dyn Provider>> {
    match name.to_ascii_lowercase().as_str() {
        "openai" => Some(Box::new(openai::openai())),
        "mistral" => Some(Box::new(openai::mistral())),
        "claude" | "anthropic" => Some(Box::new(claude::Claude::new())),
        "gemini" | "google" => Some(Box::new(gemini::Gemini::new())),
        _ => None,
    }
}

/// Turn a non-success HTTP response into a structured [`AiError::Upstream`].
pub(crate) async fn ensure_ok(
    resp: reqwest::Response,
) -> Result<reqwest::Response, crate::error::AiError> {
    let status = resp.status();
    if status.is_success() {
        Ok(resp)
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(crate::error::AiError::Upstream {
            status: status.as_u16(),
            body,
        })
    }
}

//! Provider registry and the adapter implementations.

mod claude;
mod gemini;
mod openai;

use crate::provider::Provider;

/// Canonical provider names, in a stable order (aliases excluded).
pub const PROVIDERS: &[&str] = &["openai", "mistral", "claude", "gemini"];

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

/// A process-wide reqwest client. `Client` is internally reference-counted, so
/// cloning it shares one connection pool across every request and provider.
pub(crate) fn shared_client() -> reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new).clone()
}

/// Parse model-emitted tool-call arguments (a JSON string) into a value,
/// falling back to an empty object when absent or malformed.
pub(crate) fn parse_tool_args(arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({}))
}

/// Parse a base64 `data:` URL into `(media_type, base64_data)`. Returns `None`
/// for non-data URLs or non-base64 data URLs.
pub(crate) fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if !meta.contains("base64") {
        return None;
    }
    let media_type = meta.split(';').next().unwrap_or("application/octet-stream");
    Some((media_type, data))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_provider_resolves_with_a_catalog() {
        for &name in PROVIDERS {
            let provider = resolve(name).unwrap_or_else(|| panic!("{name} should resolve"));
            assert_eq!(provider.name(), name, "canonical name mismatch for {name}");
            assert!(!provider.catalog().is_empty(), "{name} has an empty catalog");
        }
    }

    #[test]
    fn aliases_resolve_to_canonical_names() {
        assert_eq!(resolve("anthropic").unwrap().name(), "claude");
        assert_eq!(resolve("google").unwrap().name(), "gemini");
        assert!(resolve("nope").is_none());
    }
}

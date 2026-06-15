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

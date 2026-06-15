//! Unified error type shared across every provider adapter.

#[derive(Debug, thiserror::Error)]
pub enum AiError {
    /// Transport-level failure talking to the upstream engine.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// The upstream engine answered with a non-success status.
    #[error("upstream returned {status}: {body}")]
    Upstream { status: u16, body: String },

    /// The upstream engine returned no usable content.
    #[error("empty response from provider")]
    EmptyResponse,

    /// A capability the adapter does not implement yet.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

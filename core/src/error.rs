//! Unified error type shared across every provider adapter.

/// How a failed attempt should be handled by the retry/failover logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    /// Likely transient — retrying the same target may succeed.
    Transient,
    /// Retrying the same target won't help, but another target might.
    Failover,
    /// Client-side problem (e.g. malformed request) — trying anything else is
    /// pointless; stop the whole chain.
    Abort,
}

fn classify_status(status: u16) -> RetryClass {
    match status {
        408 | 409 | 425 | 429 => RetryClass::Transient,
        s if (500..=599).contains(&s) => RetryClass::Transient,
        400 | 422 => RetryClass::Abort,
        // 401 / 403 / 404 and other 4xx: a different key or engine may work.
        _ => RetryClass::Failover,
    }
}

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

    /// A failure while reading or parsing a streamed (SSE) response.
    #[error("stream error: {0}")]
    Stream(String),

    /// A capability the adapter does not implement yet.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl AiError {
    /// Classify this error for the retry/failover logic.
    pub fn retry_class(&self) -> RetryClass {
        match self {
            AiError::Http(_) | AiError::Stream(_) => RetryClass::Transient,
            AiError::EmptyResponse | AiError::Unsupported(_) => RetryClass::Failover,
            AiError::Upstream { status, .. } => classify_status(*status),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upstream(status: u16) -> AiError {
        AiError::Upstream {
            status,
            body: String::new(),
        }
    }

    #[test]
    fn rate_limit_and_server_errors_are_transient() {
        assert_eq!(upstream(429).retry_class(), RetryClass::Transient);
        assert_eq!(upstream(500).retry_class(), RetryClass::Transient);
        assert_eq!(upstream(503).retry_class(), RetryClass::Transient);
        assert_eq!(upstream(408).retry_class(), RetryClass::Transient);
    }

    #[test]
    fn malformed_requests_abort() {
        assert_eq!(upstream(400).retry_class(), RetryClass::Abort);
        assert_eq!(upstream(422).retry_class(), RetryClass::Abort);
    }

    #[test]
    fn auth_and_not_found_fail_over() {
        assert_eq!(upstream(401).retry_class(), RetryClass::Failover);
        assert_eq!(upstream(403).retry_class(), RetryClass::Failover);
        assert_eq!(upstream(404).retry_class(), RetryClass::Failover);
        assert_eq!(AiError::EmptyResponse.retry_class(), RetryClass::Failover);
        assert_eq!(AiError::Unsupported("x".into()).retry_class(), RetryClass::Failover);
    }
}

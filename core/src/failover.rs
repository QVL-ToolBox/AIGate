//! Failover orchestration: try an ordered chain of targets until one succeeds.
//!
//! Each target carries its own provider, model and key, so a chain can span
//! different engines (e.g. OpenAI → Claude). Errors are classified (see
//! [`crate::error::RetryClass`]):
//!
//! - **Transient** (429, 5xx, network): retry the *same* target with
//!   exponential backoff, then fail over.
//! - **Failover** (401/403/404, empty, unsupported): skip to the next target.
//! - **Abort** (400/422 malformed request): stop the whole chain immediately —
//!   no other target would accept it either.
//!
//! For streaming, failover covers only stream *establishment*: once a target
//! returns a live stream we forward it, and a mid-stream error surfaces as-is.

use std::time::Duration;

use crate::error::RetryClass;
use crate::provider::{ChunkStream, Provider};
use crate::types::{UnifiedRequest, UnifiedResponse};

/// One entry in a failover chain.
pub struct Target {
    pub provider: Box<dyn Provider>,
    pub model: String,
    pub key: String,
}

/// Retry behaviour applied to each target before failing over.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Maximum attempts per target (1 = no retry).
    pub max_attempts: u32,
    /// Base backoff; the nth retry waits `base_delay * 2^n`.
    pub base_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 200,
        }
    }
}

impl RetryPolicy {
    fn backoff(&self, retry_index: u32) -> Duration {
        Duration::from_millis(self.base_delay_ms.saturating_mul(2u64.saturating_pow(retry_index)))
    }
}

/// A single failed attempt against one target, kept for diagnostics.
#[derive(Debug, Clone)]
pub struct Attempt {
    pub provider: String,
    pub model: String,
    pub tries: u32,
    pub error: String,
}

/// Returned when the chain failed (every target exhausted, or an Abort error).
#[derive(Debug, Clone)]
pub struct FailoverError {
    pub attempts: Vec<Attempt>,
    /// True when the chain stopped early on a non-retriable client error.
    pub aborted: bool,
}

impl std::fmt::Display for FailoverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.attempts.is_empty() {
            write!(f, "no targets to try")
        } else if self.aborted {
            write!(f, "request rejected (not retriable)")
        } else {
            write!(f, "all {} target(s) failed", self.attempts.len())
        }
    }
}

impl std::error::Error for FailoverError {}

/// What to do after a failed attempt.
enum Action {
    Retry(Duration),
    NextTarget,
    Abort,
}

fn decide(class: RetryClass, attempt: u32, policy: &RetryPolicy) -> Action {
    let is_last = attempt + 1 >= policy.max_attempts;
    match class {
        RetryClass::Abort => Action::Abort,
        RetryClass::Transient if !is_last => Action::Retry(policy.backoff(attempt)),
        _ => Action::NextTarget,
    }
}

/// Non-streaming chat with default retry policy. Returns the winning provider
/// name alongside the response.
pub async fn chat_failover(
    targets: Vec<Target>,
    base: &UnifiedRequest,
) -> Result<(String, UnifiedResponse), FailoverError> {
    chat_failover_with(targets, base, &RetryPolicy::default()).await
}

/// Non-streaming chat with an explicit retry policy. Returns the winning
/// provider name alongside the response.
pub async fn chat_failover_with(
    targets: Vec<Target>,
    base: &UnifiedRequest,
    policy: &RetryPolicy,
) -> Result<(String, UnifiedResponse), FailoverError> {
    let mut attempts = Vec::new();
    for target in targets {
        let mut req = base.clone();
        req.model = target.model.clone();
        let name = target.provider.name();

        let mut tries = 0u32;
        loop {
            match target.provider.chat(&req, &target.key).await {
                Ok(resp) => return Ok((name.to_string(), resp)),
                Err(e) => match decide(e.retry_class(), tries, policy) {
                    Action::Retry(delay) => {
                        tries += 1;
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    Action::Abort => {
                        attempts.push(attempt(name, &target.model, tries + 1, &e));
                        return Err(FailoverError {
                            attempts,
                            aborted: true,
                        });
                    }
                    Action::NextTarget => {
                        attempts.push(attempt(name, &target.model, tries + 1, &e));
                        break;
                    }
                },
            }
        }
    }
    Err(FailoverError {
        attempts,
        aborted: false,
    })
}

/// Streaming chat with default retry policy.
pub async fn stream_failover(
    targets: Vec<Target>,
    base: &UnifiedRequest,
) -> Result<(String, String, ChunkStream), FailoverError> {
    stream_failover_with(targets, base, &RetryPolicy::default()).await
}

/// Streaming chat with an explicit retry policy. Returns the winning provider
/// name and model alongside the first stream that establishes.
pub async fn stream_failover_with(
    targets: Vec<Target>,
    base: &UnifiedRequest,
    policy: &RetryPolicy,
) -> Result<(String, String, ChunkStream), FailoverError> {
    let mut attempts = Vec::new();
    for target in targets {
        let mut req = base.clone();
        req.model = target.model.clone();
        let name = target.provider.name();

        let mut tries = 0u32;
        loop {
            match target.provider.chat_stream(&req, &target.key).await {
                Ok(stream) => return Ok((name.to_string(), target.model, stream)),
                Err(e) => match decide(e.retry_class(), tries, policy) {
                    Action::Retry(delay) => {
                        tries += 1;
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    Action::Abort => {
                        attempts.push(attempt(name, &target.model, tries + 1, &e));
                        return Err(FailoverError {
                            attempts,
                            aborted: true,
                        });
                    }
                    Action::NextTarget => {
                        attempts.push(attempt(name, &target.model, tries + 1, &e));
                        break;
                    }
                },
            }
        }
    }
    Err(FailoverError {
        attempts,
        aborted: false,
    })
}

fn attempt(provider: &str, model: &str, tries: u32, err: &crate::error::AiError) -> Attempt {
    Attempt {
        provider: provider.to_string(),
        model: model.to_string(),
        tries,
        error: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            base_delay_ms: 1,
        }
    }

    #[test]
    fn transient_retries_then_fails_over_on_last() {
        // With max_attempts = 3, indices 0 and 1 retry; index 2 is the last.
        assert!(matches!(
            decide(RetryClass::Transient, 0, &policy(3)),
            Action::Retry(_)
        ));
        assert!(matches!(
            decide(RetryClass::Transient, 1, &policy(3)),
            Action::Retry(_)
        ));
        assert!(matches!(
            decide(RetryClass::Transient, 2, &policy(3)),
            Action::NextTarget
        ));
    }

    #[test]
    fn no_retry_when_max_is_one() {
        assert!(matches!(
            decide(RetryClass::Transient, 0, &policy(1)),
            Action::NextTarget
        ));
    }

    #[test]
    fn failover_never_retries_same_target() {
        assert!(matches!(
            decide(RetryClass::Failover, 0, &policy(5)),
            Action::NextTarget
        ));
    }

    #[test]
    fn abort_stops_immediately() {
        assert!(matches!(decide(RetryClass::Abort, 0, &policy(5)), Action::Abort));
    }

    #[test]
    fn backoff_grows_exponentially() {
        let p = policy(10); // base_delay_ms = 1
        assert_eq!(p.backoff(0), Duration::from_millis(1));
        assert_eq!(p.backoff(1), Duration::from_millis(2));
        assert_eq!(p.backoff(3), Duration::from_millis(8));
    }
}

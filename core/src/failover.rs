//! Failover orchestration: try an ordered chain of targets until one succeeds.
//!
//! Each target carries its own provider, model and key, so a chain can span
//! different engines (e.g. OpenAI → Claude). On any error the next target is
//! tried; the collected errors are returned if every target fails.
//!
//! For streaming, failover covers only stream *establishment*: once a target
//! returns a live stream we forward it, and a mid-stream error surfaces as-is
//! (the client has already received a 200 and partial bytes).

use crate::provider::{ChunkStream, Provider};
use crate::types::{UnifiedRequest, UnifiedResponse};

/// One entry in a failover chain.
pub struct Target {
    pub provider: Box<dyn Provider>,
    pub model: String,
    pub key: String,
}

/// A single failed attempt, kept for diagnostics.
#[derive(Debug, Clone)]
pub struct Attempt {
    pub provider: String,
    pub model: String,
    pub error: String,
}

/// Returned when every target in the chain failed.
#[derive(Debug, Clone)]
pub struct FailoverError {
    pub attempts: Vec<Attempt>,
}

impl std::fmt::Display for FailoverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.attempts.is_empty() {
            write!(f, "no targets to try")
        } else {
            write!(f, "all {} target(s) failed", self.attempts.len())
        }
    }
}

impl std::error::Error for FailoverError {}

/// Try each target's non-streaming chat in order; return the first success.
pub async fn chat_failover(
    targets: Vec<Target>,
    base: &UnifiedRequest,
) -> Result<UnifiedResponse, FailoverError> {
    let mut attempts = Vec::new();
    for target in targets {
        let mut req = base.clone();
        req.model = target.model.clone();
        let name = target.provider.name();
        match target.provider.chat(&req, &target.key).await {
            Ok(resp) => return Ok(resp),
            Err(e) => attempts.push(Attempt {
                provider: name.to_string(),
                model: target.model,
                error: e.to_string(),
            }),
        }
    }
    Err(FailoverError { attempts })
}

/// Try each target's streaming chat in order; return the first stream that
/// establishes, alongside the model that produced it.
pub async fn stream_failover(
    targets: Vec<Target>,
    base: &UnifiedRequest,
) -> Result<(String, ChunkStream), FailoverError> {
    let mut attempts = Vec::new();
    for target in targets {
        let mut req = base.clone();
        req.model = target.model.clone();
        let name = target.provider.name();
        match target.provider.chat_stream(&req, &target.key).await {
            Ok(stream) => return Ok((target.model, stream)),
            Err(e) => attempts.push(Attempt {
                provider: name.to_string(),
                model: target.model,
                error: e.to_string(),
            }),
        }
    }
    Err(FailoverError { attempts })
}

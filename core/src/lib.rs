//! AIGate core — provider-agnostic AI gateway.
//!
//! The gateway speaks one wire format to apps (OpenAI-compatible) and
//! translates to each upstream engine through the [`Provider`] trait.
//! Credentials are never stored: the caller's API key is passed per request.

pub mod error;
pub mod failover;
pub mod provider;
pub mod providers;
pub mod types;

pub use error::AiError;
pub use failover::{chat_failover, stream_failover, Attempt, FailoverError, Target};
pub use provider::{ChunkStream, Provider};
pub use providers::resolve;
pub use types::{split_model, Chunk, Message, Role, UnifiedRequest, UnifiedResponse, Usage};

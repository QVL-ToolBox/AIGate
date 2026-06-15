//! AIGate core — provider-agnostic AI gateway.
//!
//! The gateway speaks one wire format to apps (OpenAI-compatible) and
//! translates to each upstream engine through the [`Provider`] trait.
//! Credentials are never stored: the caller's API key is passed per request.

pub mod error;
pub mod failover;
pub mod pricing;
pub mod provider;
pub mod providers;
pub mod types;

pub use error::{AiError, RetryClass};
pub use failover::{
    chat_failover, chat_failover_with, stream_failover, stream_failover_with, Attempt,
    FailoverError, RetryPolicy, Target,
};
pub use pricing::{estimate_cost, pricing, Pricing};
pub use provider::{ChunkStream, Provider};
pub use providers::{resolve, PROVIDERS};
pub use types::{
    split_model, Chunk, Content, ContentPart, FunctionCall, FunctionDef, ImageUrl, Message, Role,
    Tool, ToolCall, ToolCallChunk, UnifiedRequest, UnifiedResponse, Usage,
};

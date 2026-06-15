//! The adapter contract. Adding a new engine = implementing this trait.

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::AiError;
use crate::types::{Chunk, UnifiedRequest, UnifiedResponse};

/// A normalized stream of response increments.
pub type ChunkStream = BoxStream<'static, Result<Chunk, AiError>>;

#[async_trait]
pub trait Provider: Send + Sync {
    /// Stable identifier, e.g. `"openai"`.
    fn name(&self) -> &'static str;

    /// Non-streaming chat completion.
    async fn chat(&self, req: &UnifiedRequest, key: &str) -> Result<UnifiedResponse, AiError>;

    /// Streaming chat completion. Defaults to unsupported so a new adapter can
    /// ship non-streaming first and add this later.
    async fn chat_stream(&self, _req: &UnifiedRequest, _key: &str) -> Result<ChunkStream, AiError> {
        Err(AiError::Unsupported(format!("{}: streaming", self.name())))
    }
}

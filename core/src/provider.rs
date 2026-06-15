//! The adapter contract. Adding a new engine = implementing this trait.

use async_trait::async_trait;

use crate::error::AiError;
use crate::types::{UnifiedRequest, UnifiedResponse};

#[async_trait]
pub trait Provider: Send + Sync {
    /// Stable identifier, e.g. `"openai"`.
    fn name(&self) -> &'static str;

    /// Non-streaming chat completion.
    async fn chat(&self, req: &UnifiedRequest, key: &str) -> Result<UnifiedResponse, AiError>;

    // TODO(streaming): add
    //   async fn chat_stream(&self, req: &UnifiedRequest, key: &str)
    //       -> Result<BoxStream<'static, Result<Chunk, AiError>>, AiError>;
    // and normalize each engine's SSE format into a shared `Chunk`.
}

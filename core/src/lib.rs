//! AIGate core — provider-agnostic AI gateway.
//!
//! The gateway speaks one wire format to apps (OpenAI-compatible) and
//! translates to each upstream engine through the [`Provider`] trait.
//! Credentials are never stored: the caller's API key is passed per request.

pub mod error;
pub mod provider;
pub mod providers;
pub mod types;

pub use error::AiError;
pub use provider::Provider;
pub use providers::resolve;
pub use types::{split_model, Message, Role, UnifiedRequest, UnifiedResponse, Usage};

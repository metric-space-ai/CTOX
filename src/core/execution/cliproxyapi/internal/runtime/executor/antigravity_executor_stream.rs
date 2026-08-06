// ref: internal/runtime/executor/antigravity_executor_stream.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! Mirrored facade for the active Antigravity streaming vertical. The Rust
//! ownership split keeps wire/body state in `antigravity_executor`, persisted
//! account execution in `antigravity_executor_execute`, and provider payload
//! preparation in `antigravity_executor_request`.

pub use super::antigravity_executor::{
    AntigravityGenerateStreamResponse, AntigravityGenerateStreamingTransport,
    AntigravityResponsesStream,
};
pub use super::antigravity_executor_execute::{
    AntigravityPooledStreamExecutionOutcome, AntigravityStreamExecutionOutcome,
    AntigravityTrackedResponsesStream,
};

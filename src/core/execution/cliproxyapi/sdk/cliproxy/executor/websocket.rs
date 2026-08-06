// ref: sdk/cliproxy/executor/websocket.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;

use super::types::{RequestScopedError, StatusError};

const UPGRADE_REQUIRED_STATUS: u16 = 426;
const REPLAY_REQUIRED_BODY: &str = r#"{"error":{"message":"upstream transport requires full HTTP replay","type":"server_error","code":"upstream_http_replay_required","status":426}}"#;

/// Signals that an incremental request cannot safely continue because its
/// upstream WebSocket is no longer reusable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UpstreamWebsocketReplayRequiredError;

impl UpstreamWebsocketReplayRequiredError {
    #[must_use]
    pub const fn status_code(&self) -> u16 {
        UPGRADE_REQUIRED_STATUS
    }

    #[must_use]
    pub const fn is_request_scoped(&self) -> bool {
        true
    }
}

impl fmt::Display for UpstreamWebsocketReplayRequiredError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(REPLAY_REQUIRED_BODY)
    }
}

impl Error for UpstreamWebsocketReplayRequiredError {}

impl StatusError for UpstreamWebsocketReplayRequiredError {
    fn status_code(&self) -> u16 {
        UpstreamWebsocketReplayRequiredError::status_code(self)
    }
}

impl RequestScopedError for UpstreamWebsocketReplayRequiredError {
    fn is_request_scoped(&self) -> bool {
        UpstreamWebsocketReplayRequiredError::is_request_scoped(self)
    }
}

/// Creates the typed, request-scoped replay signal.
#[must_use]
pub const fn new_upstream_websocket_replay_required_error() -> UpstreamWebsocketReplayRequiredError
{
    UpstreamWebsocketReplayRequiredError
}

/// Reports whether any error in the source chain is the internal replay signal.
#[must_use]
pub fn is_upstream_websocket_replay_required(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(candidate) = current {
        if candidate
            .downcast_ref::<UpstreamWebsocketReplayRequiredError>()
            .is_some()
        {
            return true;
        }
        current = candidate.source();
    }
    false
}

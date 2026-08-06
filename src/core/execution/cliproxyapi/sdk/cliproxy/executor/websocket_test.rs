// ref: sdk/cliproxy/executor/websocket_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;

use super::{
    is_upstream_websocket_replay_required, new_upstream_websocket_replay_required_error,
    RequestScopedError, StatusError,
};

#[derive(Debug)]
struct WrappedReplayError {
    source: super::UpstreamWebsocketReplayRequiredError,
}

impl fmt::Display for WrappedReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("wrapped")
    }
}

impl Error for WrappedReplayError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

#[test]
fn upstream_websocket_replay_required_error_matches_pinned_contract() {
    let error = new_upstream_websocket_replay_required_error();

    assert!(is_upstream_websocket_replay_required(&error));
    assert_eq!(error.status_code(), 426);
    assert!(error.is_request_scoped());
    assert_eq!(StatusError::status_code(&error), 426);
    assert!(RequestScopedError::is_request_scoped(&error));
    assert_eq!(
        error.to_string(),
        r#"{"error":{"message":"upstream transport requires full HTTP replay","type":"server_error","code":"upstream_http_replay_required","status":426}}"#
    );
}

#[test]
fn replay_required_detection_walks_wrapped_error_sources() {
    let wrapped = WrappedReplayError {
        source: new_upstream_websocket_replay_required_error(),
    };
    assert!(is_upstream_websocket_replay_required(&wrapped));

    let unrelated = std::io::Error::other("not a replay signal");
    assert!(!is_upstream_websocket_replay_required(&unrelated));
}

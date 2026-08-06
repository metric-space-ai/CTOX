// ref: internal/runtime/executor/codex_executor_retry_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::{Duration, SystemTime};

use super::codex_executor_terminal::{codex_terminal_status, parse_codex_retry_after};

#[test]
fn retry_after_and_known_statuses_are_classified() {
    let value = serde_json::json!({"retry_after": "1.5"});
    assert_eq!(
        parse_codex_retry_after(429, &value, SystemTime::UNIX_EPOCH),
        Some(Duration::from_millis(1500))
    );
    assert_eq!(codex_terminal_status(Some("rate_limit_error"), None), 429);
    assert_eq!(codex_terminal_status(None, Some("invalid_api_key")), 401);
}

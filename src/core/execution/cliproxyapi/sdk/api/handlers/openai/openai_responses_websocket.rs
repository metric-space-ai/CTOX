// ref: sdk/api/handlers/openai/openai_responses_websocket.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::json;

pub fn websocket_close_payload_for_upstream_error(
    status: u16,
    message: &str,
) -> Option<(u16, Vec<u8>)> {
    let close_code = match status {
        400..=499 => 1008,
        500..=599 => 1011,
        _ => return None,
    };
    let reason = truncate_websocket_close_reason(message, 123);
    Some((
        close_code,
        serde_json::to_vec(&json!({"error": reason})).unwrap_or_default(),
    ))
}

#[must_use]
pub fn truncate_websocket_close_reason(reason: &str, max_bytes: usize) -> String {
    if reason.len() <= max_bytes {
        return reason.to_owned();
    }
    let mut boundary = max_bytes;
    while !reason.is_char_boundary(boundary) {
        boundary -= 1;
    }
    reason[..boundary].to_owned()
}

#[must_use]
pub fn responses_websocket_native_passthrough_allowed(
    upstream_mode: &str,
    use_upstream_websocket: bool,
    pinned_auth_id: &str,
    upstream_auth_id: &str,
) -> bool {
    use_upstream_websocket
        && upstream_mode.trim().eq_ignore_ascii_case("native")
        && (pinned_auth_id.trim().is_empty() || pinned_auth_id.trim() == upstream_auth_id.trim())
}

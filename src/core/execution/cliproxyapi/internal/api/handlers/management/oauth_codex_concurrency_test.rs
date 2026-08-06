// ref: internal/api/handlers/management/oauth_codex_concurrency_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::oauth_callback_test::setup;
use super::ManagementOAuthCallbackRequest;

#[test]
fn codex_callback_completion_keeps_concurrent_session_pending() {
    let (sessions, _, handler) = setup();
    sessions.register_builtin("first", "codex").unwrap();
    sessions.register_builtin("second", "codex").unwrap();
    handler
        .submit(ManagementOAuthCallbackRequest {
            state: "first".to_owned(),
            code: "code".to_owned(),
            ..Default::default()
        })
        .unwrap();
    assert!(sessions.details("first").unwrap().unwrap().completed);
    assert!(sessions.is_pending("second", "codex").unwrap());
}

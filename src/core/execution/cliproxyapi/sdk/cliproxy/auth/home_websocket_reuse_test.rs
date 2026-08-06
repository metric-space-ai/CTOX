// ref: sdk/cliproxy/auth/home_websocket_reuse_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: retained execution selection is reused per session/model and explicitly closed
// License: MIT (upstream); modifications AGPL-3.0-only

use super::home_execution_paths_test::{request, runtime, TestExecutor, TestHomeTransport};

#[tokio::test]
async fn retained_session_reuses_selection_without_another_home_dispatch() {
    let transport = TestHomeTransport::with_auth_ids(&["auth-session"]);
    let executor = TestExecutor::failing(0);
    let (runtime, _) = runtime(transport.clone(), executor);
    runtime
        .execute_home(request("GPT"), "session-1", false)
        .await
        .unwrap();
    runtime
        .execute_home(request("gpt"), "session-1", false)
        .await
        .unwrap();
    assert_eq!(transport.requests().len(), 1);
    assert!(runtime.retained_selection("session-1", "gpt").is_some());
    assert_eq!(runtime.close_execution_session("session-1"), 1);
    assert!(runtime.retained_selection("session-1", "gpt").is_none());
}

#[tokio::test]
async fn close_all_ends_every_retained_route() {
    let transport = TestHomeTransport::with_auth_ids(&["one", "two"]);
    let executor = TestExecutor::failing(0);
    let (runtime, _) = runtime(transport, executor);
    runtime
        .execute_home(request("one"), "session", false)
        .await
        .unwrap();
    runtime
        .execute_home(request("two"), "session", false)
        .await
        .unwrap();
    assert_eq!(
        runtime.close_execution_session(super::CLOSE_ALL_EXECUTION_SESSIONS_ID),
        2
    );
}

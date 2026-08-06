// ref: sdk/cliproxy/auth/home_retry_loop_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: bounded retry uses fresh Home selections and rejects repeated credentials
// License: MIT (upstream); modifications AGPL-3.0-only

use super::home_execution_paths_test::{request, runtime, TestExecutor, TestHomeTransport};

#[tokio::test]
async fn provider_failure_retries_with_next_home_credential() {
    let transport = TestHomeTransport::with_auth_ids(&["auth-1", "auth-2"]);
    let executor = TestExecutor::failing(1);
    let (runtime, _) = runtime(transport.clone(), executor.clone());
    let response = runtime
        .execute_home(request("gpt"), "", false)
        .await
        .unwrap();
    assert_eq!(response.payload, b"payload");
    assert_eq!(transport.requests().len(), 2);
    assert_eq!(
        executor
            .seen()
            .iter()
            .map(|entry| entry.auth_id.as_str())
            .collect::<Vec<_>>(),
        ["auth-1", "auth-2"]
    );
}

#[tokio::test]
async fn repeated_auth_stops_retry_without_a_third_dispatch() {
    let transport = TestHomeTransport::with_auth_ids(&["same", "same", "unused"]);
    let executor = TestExecutor::failing(3);
    let (runtime, _) = runtime(transport.clone(), executor);
    let error = runtime
        .execute_home(request("gpt"), "", false)
        .await
        .unwrap_err();
    assert_eq!(
        format!("{error}"),
        "Home-selected provider execution failed"
    );
    assert_eq!(transport.requests().len(), 2);
}

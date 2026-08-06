// ref: internal/runtime/executor/claude_executor_request_bench_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::remap_claude_oauth_tool_names_with_secret;

#[test]
fn large_remap_contract_remains_linear_and_complete() {
    let tools = (0..256)
        .map(|index| serde_json::json!({"name":format!("tool_{index}")}))
        .collect::<Vec<_>>();
    let body = serde_json::to_vec(&serde_json::json!({"tools":tools})).unwrap();
    let (mapped, reverse) = remap_claude_oauth_tool_names_with_secret(&body, "bench-contract");
    assert_eq!(reverse.len(), 256);
    assert!(mapped.len() > body.len());
}

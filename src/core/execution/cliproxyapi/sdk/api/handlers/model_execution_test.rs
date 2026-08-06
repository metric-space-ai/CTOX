// ref: sdk/api/handlers/model_execution_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

#[test]
fn model_execution_carries_entry_and_exit_protocols() {
    let protocol: ProtocolExecutionRequest = ModelExecutionRequest {
        entry_protocol: "openai".to_owned(),
        exit_protocol: "claude".to_owned(),
        model: "claude-sonnet".to_owned(),
        body: b"{}".to_vec(),
        ..ModelExecutionRequest::default()
    }
    .into();
    assert_eq!(protocol.entry_protocol, "openai");
    assert_eq!(protocol.exit_protocol, "claude");
    assert_eq!(
        response_protocol(&protocol.entry_protocol, &protocol.exit_protocol),
        "claude"
    );
    assert_eq!(response_protocol("openai", ""), "openai");
}

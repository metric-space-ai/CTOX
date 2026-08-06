// ref: sdk/api/handlers/openai/openai_responses_signature_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

#[test]
fn invalid_encrypted_reasoning_is_forwarded_unchanged_to_executor_boundary() {
    let body = r#"{"model":"test-signature-model","stream":false,"input":[{"id":"rs_bad","type":"reasoning","encrypted_content":"gAAAAABq…abc","summary":[]}]}"#
        .as_bytes();
    let request = parse_request(body).unwrap();
    assert_eq!(request.model, "test-signature-model");
    assert!(!request.stream);
    let protocol = crate::sdk::api::handlers::ProtocolExecutionRequest {
        entry_protocol: "openai-response".to_owned(),
        exit_protocol: "openai-response".to_owned(),
        model: request.model,
        body: body.to_vec(),
        ..Default::default()
    };
    let context = crate::sdk::api::handlers::HandlerRequestContext::default();
    let executor = crate::sdk::api::handlers::build_executor_request(
        &context,
        &protocol,
        "codex",
        protocol.body.clone(),
    );
    assert_eq!(executor.payload, body);
    assert_eq!(executor.original_request, body);
}

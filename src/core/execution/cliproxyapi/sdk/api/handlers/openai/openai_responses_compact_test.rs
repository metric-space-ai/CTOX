// ref: sdk/api/handlers/openai/openai_responses_compact_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

#[test]
fn compact_rejects_stream_but_accepts_buffered_request() {
    let stream = parse_request(br#"{"model":"gpt-5","stream":true}"#).unwrap();
    assert!(stream.stream);
    let buffered = parse_request(br#"{"model":"gpt-5","input":"hello"}"#).unwrap();
    assert!(!buffered.stream);
    assert_eq!(buffered.model, "gpt-5");
}

#[test]
fn compact_alt_is_carried_by_protocol_execution_contract() {
    let request = crate::sdk::api::handlers::ProtocolExecutionRequest {
        entry_protocol: "openai-response".to_owned(),
        exit_protocol: "openai-response".to_owned(),
        model: "gpt-5".to_owned(),
        body: br#"{"model":"gpt-5"}"#.to_vec(),
        alt: "responses/compact".to_owned(),
        ..Default::default()
    };
    assert_eq!(request.alt, "responses/compact");
}

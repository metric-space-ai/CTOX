// ref: internal/translator/codex/openai/responses/codex_openai-responses_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_codex_response_to_openai_responses_non_stream;

#[test]
fn incomplete_terminal_returns_response_payload() {
    let output = convert_codex_response_to_openai_responses_non_stream(
        br#"{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}}"#,
    );
    let output: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(output["status"], "incomplete");
    assert_eq!(output["incomplete_details"]["reason"], "max_output_tokens");
}

// ref: internal/translator/claude/openai/responses/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::convert_claude_response_to_openai_responses_non_stream;
use serde_json::Value;

#[test]
fn non_stream_keeps_zero_usage_defaults() {
    let value: Value = serde_json::from_slice(
        &convert_claude_response_to_openai_responses_non_stream(b"{}", b"{}", b""),
    )
    .unwrap();
    assert_eq!(value["usage"]["input_tokens"], 0);
    assert_eq!(value["usage"]["output_tokens"], 0);
    assert_eq!(value["usage"]["total_tokens"], 0);
}

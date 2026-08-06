// ref: internal/translator/claude/gemini/claude_gemini_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    convert_claude_response_to_gemini, convert_claude_response_to_gemini_non_stream,
    ClaudeToGeminiState,
};
use serde_json::Value;

#[test]
fn stream_and_non_stream_preserve_tool_use_id() {
    let events: [&[u8]; 3] = [
        br#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_gateway","name":"lookup"}}

"#,
        br#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"status\"}"}}

"#,
        br#"data: {"type":"content_block_stop","index":0}

"#,
    ];
    let mut state = ClaudeToGeminiState::default();
    assert!(
        convert_claude_response_to_gemini("gemini", &[], &[], events[0], &mut state).is_empty()
    );
    assert!(
        convert_claude_response_to_gemini("gemini", &[], &[], events[1], &mut state).is_empty()
    );
    let out = convert_claude_response_to_gemini("gemini", &[], &[], events[2], &mut state);
    let value: Value = serde_json::from_slice(&out[0]).unwrap();
    assert_eq!(
        value
            .pointer("/candidates/0/content/parts/0/functionCall/id")
            .unwrap(),
        "toolu_gateway"
    );
    let raw = events.concat();
    let value: Value = serde_json::from_slice(&convert_claude_response_to_gemini_non_stream(
        "gemini",
        &[],
        &[],
        &raw,
    ))
    .unwrap();
    assert_eq!(
        value
            .pointer("/candidates/0/content/parts/0/functionCall/id")
            .unwrap(),
        "toolu_gateway"
    );
}

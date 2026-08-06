// ref: internal/translator/codex/gemini/codex_gemini_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::codex_gemini_response::{
    convert_codex_response_to_gemini_non_stream, convert_codex_response_to_gemini_stream,
    CodexToGeminiStreamState,
};

fn stream(raw: &str, state: &mut CodexToGeminiStreamState) -> Vec<Value> {
    convert_codex_response_to_gemini_stream(
        "gemini-2.5-pro",
        br#"{"tools":[]}"#,
        format!("data: {raw}").as_bytes(),
        state,
    )
    .into_iter()
    .map(|bytes| serde_json::from_slice(&bytes).unwrap())
    .collect()
}

#[test]
fn incomplete_terminal_maps_finish_reason_in_both_modes() {
    let raw = r#"{"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":{"input_tokens":1,"output_tokens":2}}}"#;
    let streamed = stream(raw, &mut CodexToGeminiStreamState::default());
    assert_eq!(streamed[0]["candidates"][0]["finishReason"], "MAX_TOKENS");
    let nonstream: Value = serde_json::from_slice(&convert_codex_response_to_gemini_non_stream(
        "gemini-2.5-pro",
        b"{}",
        raw.as_bytes(),
    ))
    .unwrap();
    assert_eq!(nonstream["candidates"][0]["finishReason"], "MAX_TOKENS");
}

#[test]
fn empty_stream_output_uses_done_message_fallback() {
    let mut state = CodexToGeminiStreamState::default();
    let output = stream(
        r#"{"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"ok"}]}}"#,
        &mut state,
    );
    assert_eq!(
        output[0]["candidates"][0]["content"]["parts"][0]["text"],
        "ok"
    );
}

#[test]
fn partial_and_done_images_are_emitted_and_deduplicated() {
    let mut state = CodexToGeminiStreamState::default();
    let partial = r#"{"type":"response.image_generation_call.partial_image","item_id":"ig","output_format":"png","partial_image_b64":"aGVsbG8="}"#;
    let output = stream(partial, &mut state);
    assert_eq!(
        output[0]["candidates"][0]["content"]["parts"][0]["inlineData"]["data"],
        "aGVsbG8="
    );
    assert!(stream(partial, &mut state).is_empty());
    assert!(stream(r#"{"type":"response.output_item.done","item":{"id":"ig","type":"image_generation_call","output_format":"png","result":"aGVsbG8="}}"#, &mut state).is_empty());
    let changed = stream(
        r#"{"type":"response.output_item.done","item":{"id":"ig","type":"image_generation_call","output_format":"jpeg","result":"Ymll"}}"#,
        &mut state,
    );
    assert_eq!(
        changed[0]["candidates"][0]["content"]["parts"][0]["inlineData"]["mimeType"],
        "image/jpeg"
    );
}

#[test]
fn nonstream_adds_image_part() {
    let raw = br#"{"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]},{"type":"image_generation_call","output_format":"png","result":"aGVsbG8="}],"usage":{"input_tokens":1,"output_tokens":1}}}"#;
    let output: Value = serde_json::from_slice(&convert_codex_response_to_gemini_non_stream(
        "gemini-2.5-pro",
        b"{}",
        raw,
    ))
    .unwrap();
    assert_eq!(
        output["candidates"][0]["content"]["parts"][1]["inlineData"]["data"],
        "aGVsbG8="
    );
}

#[test]
fn function_call_ids_survive_stream_and_nonstream() {
    let mut state = CodexToGeminiStreamState::default();
    assert!(stream(r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_gateway","name":"lookup","arguments":"{\"query\":\"status\"}"}}"#, &mut state).is_empty());
    let completed = stream(
        r#"{"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}"#,
        &mut state,
    );
    assert_eq!(
        completed[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["id"],
        "call_gateway"
    );

    let raw = br#"{"type":"response.completed","response":{"output":[{"type":"function_call","call_id":"call_gateway","name":"lookup","arguments":"{\"query\":\"status\"}"}],"usage":{"input_tokens":1,"output_tokens":1}}}"#;
    let output: Value = serde_json::from_slice(&convert_codex_response_to_gemini_non_stream(
        "gemini-2.5-pro",
        b"{}",
        raw,
    ))
    .unwrap();
    assert_eq!(
        output["candidates"][0]["content"]["parts"][0]["functionCall"]["id"],
        "call_gateway"
    );
}

// ref: internal/translator/gemini/openai/responses/gemini_openai-responses_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_gemini_response_to_openai_responses_non_stream,
    convert_gemini_response_to_openai_responses_stream, GeminiToResponsesState,
};

#[test]
fn non_stream_preserves_reasoning_message_function_order_usage_and_name() {
    let output: Value = serde_json::from_slice(
        &convert_gemini_response_to_openai_responses_non_stream(
            br#"{"model":"gemini","instructions":"concise","tools":[{"type":"function","name":"1 read file"}]}"#,
            b"",
            br#"{"responseId":"native","createTime":"2026-08-03T12:34:56Z","candidates":[{"content":{"parts":[{"thought":true,"text":"think","thoughtSignature":"sig-thought"},{"text":"answer"},{"functionCall":{"name":"_1_read_file","args":{"path":"README.md"}},"thoughtSignature":"sig-call"}]}}],"usageMetadata":{"promptTokenCount":5,"cachedContentTokenCount":2,"candidatesTokenCount":3,"thoughtsTokenCount":1,"totalTokenCount":8}}"#,
        ),
    )
    .unwrap();
    assert_eq!(output["id"], "resp_native");
    assert_eq!(output["instructions"], "concise");
    assert_eq!(output["output"][0]["type"], "reasoning");
    assert_eq!(output["output"][1]["type"], "message");
    assert_eq!(output["output"][3]["type"], "function_call");
    assert_eq!(output["output"][3]["name"], "1 read file");
    assert_eq!(output["usage"]["input_tokens_details"]["cached_tokens"], 2);
    assert_eq!(
        output["usage"]["output_tokens_details"]["reasoning_tokens"],
        1
    );
}

#[test]
fn stream_uses_injected_identity_and_completes_exactly_once() {
    let mut state = GeminiToResponsesState::with_identity("resp_injected", 123);
    let first = convert_gemini_response_to_openai_responses_stream(
        br#"{"model":"gemini"}"#,
        b"",
        br#"{"candidates":[{"content":{"parts":[{"text":"hel"}]}}]}"#,
        &mut state,
    );
    let second = convert_gemini_response_to_openai_responses_stream(
        br#"{"model":"gemini"}"#,
        b"",
        br#"{"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#,
        &mut state,
    );
    let after =
        convert_gemini_response_to_openai_responses_stream(b"{}", b"", b"[DONE]", &mut state);
    let wire = [first, second].concat().concat();
    let text = String::from_utf8(wire).unwrap();
    assert!(text.contains("\"id\":\"resp_injected\""));
    assert!(text.contains("\"created_at\":123"));
    assert_eq!(text.matches("event: response.completed").count(), 1);
    assert!(text.contains("\"text\":\"hello\""));
    assert!(after.is_empty());
}

#[test]
fn data_framing_vertex_wrapper_and_done_are_accepted() {
    let mut state = GeminiToResponsesState::with_identity("fallback", 0);
    let events = convert_gemini_response_to_openai_responses_stream(
        b"{}",
        b"",
        br#"data: {"response":{"responseId":"vertex","createTime":"2026-08-03T12:34:56Z","candidates":[{"content":{"parts":[{"text":"wrapped"}]},"finishReason":"STOP"}]}}"#,
        &mut state,
    );
    let text = String::from_utf8(events.concat()).unwrap();
    assert!(text.contains("resp_vertex"));
    assert!(text.contains("wrapped"));
    assert!(text.contains("event: response.completed"));
}

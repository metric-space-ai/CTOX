// ref: internal/translator/codex/interactions/interactions_codex_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::sdk::translator::TranslationContext;

use super::{
    convert_codex_response_to_interactions_non_stream,
    convert_codex_response_to_interactions_stream, convert_interactions_request_to_codex,
    CodexToInteractionsState,
};

#[test]
fn request_maps_system_thinking_tools_media_and_stream() {
    let output: Value = serde_json::from_slice(&convert_interactions_request_to_codex(
        "gpt-5.6",
        br#"{"system_instruction":"be exact","stream":true,"generation_config":{"thinking_level":"high"},"input":[{"type":"user_input","content":[{"type":"text","text":"hi"},{"type":"audio","mime_type":"audio/wav","data":"UklGRg=="}]},{"type":"function_call","name":"lookup","call_id":"c1","arguments":{"q":"x"}},{"type":"function_result","name":"lookup","call_id":"c1","result":{"ok":true}}],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}"#,
        false,
    ))
    .unwrap();
    assert_eq!(output["instructions"], "be exact");
    assert_eq!(output["stream"], true);
    assert_eq!(output["reasoning"]["effort"], "high");
    assert_eq!(output["input"][1]["content"][0]["type"], "input_audio");
    assert_eq!(output["input"][2]["type"], "function_call");
    assert_eq!(output["input"][3]["type"], "function_call_output");
    assert_eq!(output["tools"][0]["name"], "lookup");
}

#[test]
fn aggregate_response_maps_steps_status_and_usage() {
    let output: Value = serde_json::from_slice(&convert_codex_response_to_interactions_non_stream(
        &TranslationContext::default(),
        "gpt-5.6",
        b"{}",
        b"{}",
        br#"{"id":"resp_1","model":"gpt-5.6","status":"incomplete","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"why"}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},{"type":"function_call","name":"lookup","call_id":"c1","arguments":"{}"}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}"#,
    ))
    .unwrap();
    assert_eq!(output["id"], "resp_1");
    assert_eq!(output["status"], "incomplete");
    assert_eq!(output["steps"][0]["type"], "thought");
    assert_eq!(output["steps"][1]["type"], "model_output");
    assert_eq!(output["steps"][2]["type"], "function_call");
    assert_eq!(output["usage"]["total_tokens"], 3);
}

#[test]
fn stream_orders_created_steps_completed_and_done() {
    let context = TranslationContext::default();
    let mut state = CodexToInteractionsState::default();
    let mut events = Vec::new();
    for chunk in [
        br#"data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.6","status":"in_progress"}}"#.as_slice(),
        br#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","name":"lookup","call_id":"c1","arguments":""}}"#.as_slice(),
        br#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"x\"}"}"#.as_slice(),
        br#"data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.6","status":"completed","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}"#.as_slice(),
    ] {
        events.extend(convert_codex_response_to_interactions_stream(
            &context,
            "gpt-5.6",
            b"{}",
            b"{}",
            chunk,
            &mut state,
        ));
    }
    let joined = String::from_utf8(events.concat()).unwrap();
    assert!(joined.contains("interaction.created"));
    assert!(joined.contains("step.start"));
    assert!(joined.contains("arguments_delta"));
    assert!(joined.contains("interaction.completed"));
    assert!(joined.contains("event: done"));
}

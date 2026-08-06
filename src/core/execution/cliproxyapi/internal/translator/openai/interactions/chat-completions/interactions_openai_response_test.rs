// ref: internal/translator/openai/interactions/chat-completions/interactions_openai_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use gjson;
use serde_json::Value;

use super::{
    convert_interactions_response_to_openai, convert_interactions_response_to_openai_non_stream,
    convert_openai_response_to_interactions, convert_openai_response_to_interactions_non_stream,
    InteractionsToOpenAIChatStreamState, OpenAIToInteractionsStreamState,
};
use crate::sdk::translator::{TranslationContext, TranslationState};

fn count_events(events: &[Vec<u8>], event_type: &str) -> usize {
    events
        .iter()
        .filter(|event| interactions_event_name(event) == event_type)
        .count()
}

fn find_event_payload<'a>(events: &'a [Vec<u8>], event_type: &str) -> Option<&'a [u8]> {
    events
        .iter()
        .find(|event| interactions_event_name(event) == event_type)
        .and_then(|event| interactions_sse_payload(event))
}

fn interactions_event_name(event: &[u8]) -> String {
    let text = std::str::from_utf8(event).unwrap_or("");
    if let Some(rest) = text.strip_prefix("event: ") {
        if let Some(line) = rest.lines().next() {
            return line.to_owned();
        }
    }
    if let Some(payload) = interactions_sse_payload(event) {
        let value: Value = serde_json::from_slice(payload).unwrap_or(Value::Null);
        if let Some(name) = value.get("event_type").and_then(Value::as_str) {
            return name.to_owned();
        }
    }
    String::new()
}

fn json_at(bytes: &[u8], pointer: &str) -> Value {
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .and_then(|value| value.pointer(pointer).cloned())
        .unwrap_or(Value::Null)
}

fn interactions_sse_payload(event: &[u8]) -> Option<&[u8]> {
    let marker = b"\ndata: ";
    event
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|index| &event[index + marker.len()..])
}

fn find_openai_chat_chunk<'a>(chunks: &'a [Vec<u8>], path: &str) -> Option<&'a [u8]> {
    chunks
        .iter()
        .find(|chunk| gjson::get(std::str::from_utf8(chunk).unwrap_or(""), path).exists())
        .map(Vec::as_slice)
}

fn find_openai_chat_chunk_value<'a>(
    chunks: &'a [Vec<u8>],
    path: &str,
    want: &str,
) -> Option<&'a [u8]> {
    chunks.iter().find_map(|chunk| {
        let parsed = gjson::get(std::str::from_utf8(chunk).unwrap_or(""), path);
        if parsed.exists() && parsed.str() == want {
            Some(chunk.as_slice())
        } else {
            None
        }
    })
}

#[test]
fn stream_usage_only_terminal_chunk_routes_to_interactions() {
    let context = TranslationContext::default();
    let finish_raw = br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
    let usage_raw = br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-test","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}"#;
    let done_raw = b"data: [DONE]";

    let mut state_box: TranslationState = None;
    let finish_out = convert_openai_response_to_interactions(
        &context,
        "gpt-test",
        b"{}",
        b"{}",
        finish_raw,
        &mut state_box,
    );
    let usage_out = convert_openai_response_to_interactions(
        &context,
        "gpt-test",
        b"{}",
        b"{}",
        usage_raw,
        &mut state_box,
    );
    let done_out = convert_openai_response_to_interactions(
        &context,
        "gpt-test",
        b"{}",
        b"{}",
        done_raw,
        &mut state_box,
    );

    assert_eq!(count_events(&finish_out, "interaction.completed"), 0);
    assert_eq!(count_events(&usage_out, "interaction.completed"), 1);
    assert_eq!(count_events(&done_out, "interaction.completed"), 0);
    assert_eq!(count_events(&done_out, "done"), 1);
    let payload = find_event_payload(&usage_out, "interaction.completed").unwrap();
    assert_eq!(json_at(payload, "/interaction/usage/total_input_tokens"), 3);
    assert_eq!(
        json_at(payload, "/interaction/usage/total_output_tokens"),
        4
    );
    assert_eq!(json_at(payload, "/interaction/usage/total_tokens"), 7);
}

#[test]
fn stream_completes_on_done_without_usage() {
    let context = TranslationContext::default();
    let mut state: TranslationState = None;
    let finish_raw = br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
    let done_raw = b"data: [DONE]";

    let finish_out = convert_openai_response_to_interactions(
        &context, "gpt-test", b"{}", b"{}", finish_raw, &mut state,
    );
    let done_out = convert_openai_response_to_interactions(
        &context, "gpt-test", b"{}", b"{}", done_raw, &mut state,
    );

    assert_eq!(count_events(&finish_out, "interaction.completed"), 0);
    assert_eq!(count_events(&done_out, "interaction.completed"), 1);
    assert_eq!(count_events(&done_out, "done"), 1);
}

#[test]
fn stream_uses_chunk_identity_for_interaction_created() {
    let context = TranslationContext::default();
    let mut state: TranslationState = None;
    let raw = br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-test","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#;
    let out = convert_openai_response_to_interactions(&context, "", b"{}", b"{}", raw, &mut state);
    let payload = find_event_payload(&out, "interaction.created").unwrap();
    assert_eq!(json_at(payload, "/interaction/id"), "chatcmpl_1");
    assert_eq!(json_at(payload, "/interaction/model"), "gpt-test");
}

#[test]
fn non_stream_direct_tool_call_maps_to_function_call_step() {
    let raw = br#"{"id":"chatcmpl_1","model":"gpt-test","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}"#;
    let out = convert_openai_response_to_interactions_non_stream(
        &TranslationContext::default(),
        "gpt-test",
        b"{}",
        b"{}",
        raw,
        &mut TranslationState::default(),
    );
    let value: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(value["steps"][0]["type"], "function_call");
    assert_eq!(value["steps"][0]["call_id"], "call_1");
    assert_eq!(value["steps"][0]["arguments"]["q"], "x");
}

#[test]
fn stream_tool_call_chunks_match_openai_layout() {
    let context = TranslationContext::default();
    let mut state: TranslationState = None;
    // The user message and arguments carry non-ASCII characters. Build the
    // SSE wire bytes through a `String` so the literal can hold UTF-8 without
    // tripping rustfmt's raw-byte-string ASCII restriction.
    let chunks: Vec<Vec<u8>> = vec![
        String::from(
            r#"data: {"event_type":"interaction.created","interaction":{"id":"i1","model":"gemini-3.1-flash-lite"}}"#,
        )
        .into_bytes(),
        String::from(
            r#"data: {"event_type":"step.start","index":0,"step":{"type":"function_call","id":"call_1","name":"get_weather","arguments":{}}}"#,
        )
        .into_bytes(),
        String::from(
            r#"data: {"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"{\"location\":\"\u5317\u4eac\"}"}}"#,
        )
        .into_bytes(),
        String::from(r#"data: {"event_type":"step.stop","index":0}"#).into_bytes(),
        String::from(
            r#"data: {"event_type":"interaction.completed","interaction":{"id":"i1","status":"requires_action","usage":{"total_input_tokens":2,"total_output_tokens":3,"total_tokens":5}}}"#,
        )
        .into_bytes(),
    ];
    let mut out = Vec::new();
    for chunk in &chunks {
        out.extend(convert_interactions_response_to_openai(
            &context,
            "gemini-3.1-flash-lite",
            b"{}",
            b"{}",
            chunk,
            &mut state,
        ));
    }
    let tool_start = find_openai_chat_chunk(&out, "choices.0.delta.tool_calls.0.function.name")
        .expect("tool start chunk");
    assert_eq!(
        json_at(tool_start, "/choices/0/delta/tool_calls/0/id"),
        "call_1"
    );
    assert_eq!(
        json_at(tool_start, "/choices/0/delta/tool_calls/0/function/name"),
        "get_weather"
    );
    let expected_args = String::from(r#"{"location":"北京"}"#);
    let tool_args = find_openai_chat_chunk_value(
        &out,
        "choices.0.delta.tool_calls.0.function.arguments",
        &expected_args,
    )
    .expect("tool args chunk");
    assert_eq!(
        json_at(
            tool_args,
            "/choices/0/delta/tool_calls/0/function/arguments"
        ),
        expected_args
    );
    let completed = find_openai_chat_chunk_value(&out, "choices.0.finish_reason", "tool_calls")
        .expect("completion chunk");
    assert_eq!(json_at(completed, "/choices/0/finish_reason"), "tool_calls");
    assert_eq!(json_at(completed, "/usage/prompt_tokens"), 2);
}

#[test]
fn finish_event_reads_metadata_total_usage() {
    let context = TranslationContext::default();
    let mut state: TranslationState = None;
    let out = convert_interactions_response_to_openai(
        &context,
        "gpt-test",
        b"{}",
        b"{}",
        br#"data: {"event_type":"finish","metadata":{"total_usage":{"total_input_tokens":2,"total_output_tokens":6,"total_thought_tokens":3,"total_cached_tokens":1,"total_tokens":11}}}"#,
        &mut state,
    );
    let completed =
        find_openai_chat_chunk_value(&out, "choices.0.finish_reason", "stop").expect("completion");
    assert_eq!(json_at(completed, "/usage/prompt_tokens"), 2);
    assert_eq!(json_at(completed, "/usage/completion_tokens"), 6);
    assert_eq!(
        json_at(
            completed,
            "/usage/completion_tokens_details/reasoning_tokens"
        ),
        3
    );
    assert_eq!(
        json_at(completed, "/usage/prompt_tokens_details/cached_tokens"),
        1
    );
    assert_eq!(json_at(completed, "/usage/total_tokens"), 11);
}

#[test]
fn non_stream_tool_call_matches_interactions_function_call_step() {
    let raw = String::from(
        r#"{"id":"i1","model":"gemini-3.1-flash-lite","steps":[{"type":"function_call","id":"call_1","name":"get_weather","arguments":{"location":"北京"}}],"usage":{"total_input_tokens":2,"total_output_tokens":3,"total_tokens":5}}"#,
    )
    .into_bytes();
    let out = convert_interactions_response_to_openai_non_stream(
        &TranslationContext::default(),
        "gemini-3.1-flash-lite",
        b"{}",
        b"{}",
        &raw,
        &mut TranslationState::default(),
    );
    let value: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"location\":\"北京\"}"
    );
    assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn stream_state_types_resolve_via_registry_helpers() {
    // Smoke test that the state types used by the leaf are also addressable
    // through the TranslationState downcast helper used by `init.rs`.
    let state: TranslationState = Some(Box::new(OpenAIToInteractionsStreamState::default()));
    assert!(state
        .as_ref()
        .is_some_and(|value| value.is::<OpenAIToInteractionsStreamState>()));
    let state: TranslationState = Some(Box::new(InteractionsToOpenAIChatStreamState::default()));
    assert!(state
        .as_ref()
        .is_some_and(|value| value.is::<InteractionsToOpenAIChatStreamState>()));
}

#[test]
fn synthesized_identity_is_request_local_and_deterministic() {
    let context = TranslationContext::default();
    let chat_raw = br#"{"choices":[]}"#;
    let interactions_raw = br#"{"steps":[]}"#;

    let chat_to_interactions = |request: &[u8]| {
        let mut state = None;
        let output = convert_openai_response_to_interactions_non_stream(
            &context, "gpt-test", request, b"", chat_raw, &mut state,
        );
        serde_json::from_slice::<Value>(&output).unwrap()
    };
    let interactions_to_chat = |request: &[u8]| {
        let mut state = None;
        let output = convert_interactions_response_to_openai_non_stream(
            &context,
            "gpt-test",
            request,
            b"",
            interactions_raw,
            &mut state,
        );
        serde_json::from_slice::<Value>(&output).unwrap()
    };

    let first = chat_to_interactions(br#"{"messages":["one"]}"#);
    let repeated = chat_to_interactions(br#"{"messages":["one"]}"#);
    let other = chat_to_interactions(br#"{"messages":["two"]}"#);
    assert_eq!(first["id"], repeated["id"]);
    assert_ne!(first["id"], other["id"]);

    let first = interactions_to_chat(br#"{"input":"one"}"#);
    let repeated = interactions_to_chat(br#"{"input":"one"}"#);
    let other = interactions_to_chat(br#"{"input":"two"}"#);
    assert_eq!(first["id"], repeated["id"]);
    assert_eq!(first["created"], repeated["created"]);
    assert_ne!(first["id"], other["id"]);
    assert_ne!(first["created"], other["created"]);
}

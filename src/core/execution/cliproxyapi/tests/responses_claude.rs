use ctox_cliproxyapi::internal::translator::claude::openai::responses::{
    convert_claude_response_to_openai_responses,
    convert_claude_response_to_openai_responses_non_stream,
    convert_openai_responses_request_to_claude, ClaudeResponsesStreamDecoder,
    ClaudeToResponsesState,
};
use ctox_cliproxyapi::sdk::translator::TranslationContext;
use serde_json::Value;

fn value(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap()
}
fn event(bytes: &[u8]) -> (&str, Value) {
    let text = std::str::from_utf8(bytes).unwrap();
    let (event, data) = text.split_once("\ndata: ").unwrap();
    (
        event.strip_prefix("event: ").unwrap(),
        serde_json::from_str(data).unwrap(),
    )
}

#[test]
fn request_sanitizes_tool_ids_and_preserves_adjacency() {
    let input = br#"{"input":[{"type":"function_call","call_id":"call.with space:1","name":"Read","arguments":"{\"path\":\"README.md\"}"},{"type":"function_call_output","call_id":"call.with space:1","output":"ok"}]}"#;
    let output = value(&convert_openai_responses_request_to_claude(
        "claude-test",
        input,
        false,
    ));
    assert_eq!(
        output["messages"][0]["content"][0]["id"],
        "call_with_space_1"
    );
    assert_eq!(
        output["messages"][1]["content"][0]["tool_use_id"],
        "call_with_space_1"
    );
    assert_eq!(
        output["messages"][0]["content"][0]["input"]["path"],
        "README.md"
    );
}

#[test]
fn request_preserves_images_cache_control_and_normalizes_tools() {
    let input = br#"{
      "input":[{"role":"user","content":[
        {"type":"input_text","text":"cached","cache_control":{"type":"ephemeral"}},
        {"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}
      ]}],
      "tools":[
        {"type":"custom","name":"apply_patch"},
        {"type":"function","name":"lookup","parameters":{"oneOf":[{"type":"object","properties":{"query":{"type":"string"}}}],"properties":{"id":{"type":"string"}}}}
      ]
    }"#;
    let output = value(&convert_openai_responses_request_to_claude(
        "claude-test",
        input,
        true,
    ));
    assert_eq!(
        output["messages"][0]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(
        output["messages"][0]["content"][1]["source"]["media_type"],
        "image/png"
    );
    assert_eq!(output["tools"].as_array().unwrap().len(), 1);
    assert_eq!(output["tools"][0]["input_schema"]["type"], "object");
    assert!(output["tools"][0]["input_schema"].get("oneOf").is_none());
    assert!(output["tools"][0]["input_schema"]["properties"]
        .get("query")
        .is_some());
    assert_eq!(output["stream"], true);
}

#[test]
fn non_stream_aggregates_text_reasoning_tools_usage_and_namespace() {
    let request = br#"{"model":"gpt-test","tools":[{"type":"namespace","name":"mcp__node_repl","tools":[{"type":"function","name":"js"}]}]}"#;
    let raw = br#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":13,"cache_read_input_tokens":100,"cache_creation_input_tokens":7}}}
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello"}}
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call_1","name":"mcp__node_repl__js","input":{}}}
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"code\":\"1+1\"}"}}
data: {"type":"message_delta","usage":{"output_tokens":4}}"#;
    let output = value(&convert_claude_response_to_openai_responses_non_stream(
        request, b"", raw,
    ));
    assert_eq!(output["output"][0]["encrypted_content"], "sig");
    assert_eq!(output["output"][0]["summary"][0]["text"], "think");
    assert_eq!(output["output"][1]["content"][0]["text"], "hello");
    assert_eq!(output["output"][2]["name"], "js");
    assert_eq!(output["output"][2]["namespace"], "mcp__node_repl");
    assert_eq!(output["usage"]["input_tokens"], 120);
    assert_eq!(output["usage"]["total_tokens"], 124);
}

#[test]
fn non_stream_keeps_zero_usage_defaults() {
    let output = value(&convert_claude_response_to_openai_responses_non_stream(
        b"", b"", br#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
    ));
    assert_eq!(output["usage"]["input_tokens"], 0);
    assert_eq!(output["usage"]["output_tokens"], 0);
    assert_eq!(output["usage"]["total_tokens"], 0);
}

#[test]
fn stream_emits_ordered_responses_events_without_claude_leakage() {
    let chunks: &[&[u8]] = &[
        br#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":1}}}"#,
        br#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        br#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
        br#"data: {"type":"content_block_stop","index":0}"#,
        br#"data: {"type":"message_delta","usage":{"output_tokens":2}}"#,
        br#"data: {"type":"message_stop"}"#,
    ];
    let mut state = ClaudeToResponsesState::default();
    let mut outputs = Vec::new();
    for chunk in chunks {
        outputs.extend(convert_claude_response_to_openai_responses(
            "claude-test",
            b"",
            b"",
            chunk,
            &mut state,
        ));
    }
    let parsed: Vec<_> = outputs.iter().map(|chunk| event(chunk)).collect();
    let names: Vec<_> = parsed.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    let sequences: Vec<_> = parsed
        .iter()
        .map(|(_, data)| data["sequence_number"].as_u64().unwrap())
        .collect();
    assert_eq!(sequences, (1..=9).collect::<Vec<_>>());
    assert_eq!(
        parsed.last().unwrap().1["response"]["usage"]["total_tokens"],
        3
    );
}

#[test]
fn stream_decoder_handles_transport_fragmentation_malformed_events_and_finish() {
    let context = TranslationContext::default();
    let mut decoder = ClaudeResponsesStreamDecoder::new();
    let mut outputs = decoder.push(
        &context,
        "claude-test",
        b"",
        b"",
        b": keepalive\r\ndata: {\"type\":\"message_sta",
    );
    assert!(outputs.is_empty());
    outputs.extend(decoder.push(
        &context,
        "claude-test",
        b"",
        b"",
        b"rt\",\"message\":{\"id\":\"msg_fragmented\",\"usage\":{\"input_tokens\":1}}}\r\n\r\ndata: not-json\r\n\r\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    ));
    outputs.extend(decoder.push(
        &context,
        "claude-test",
        b"",
        b"",
        b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}\n\ndata: {\"type\":\"message_stop\"}",
    ));
    outputs.extend(decoder.finish(&context, "claude-test", b"", b""));

    let parsed: Vec<_> = outputs.iter().map(|chunk| event(chunk)).collect();
    let names: Vec<_> = parsed.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    let sequences: Vec<_> = parsed
        .iter()
        .map(|(_, data)| data["sequence_number"].as_u64().unwrap())
        .collect();
    assert_eq!(sequences, (1..=9).collect::<Vec<_>>());
}

#[test]
fn stream_decoder_stops_consuming_after_cancellation() {
    let context = TranslationContext::default();
    let mut decoder = ClaudeResponsesStreamDecoder::new();
    let started = decoder.push(
        &context,
        "claude-test",
        b"",
        b"",
        b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_cancel\"}}\n\n",
    );
    assert_eq!(started.len(), 2);
    context.cancel();
    assert!(decoder
        .push(
            &context,
            "claude-test",
            b"",
            b"",
            b"data: {\"type\":\"message_stop\"}\n\n",
        )
        .is_empty());
    assert!(decoder.finish(&context, "claude-test", b"", b"").is_empty());
}

#[test]
fn stream_maps_claude_error_to_terminal_response_failed() {
    let mut state = ClaudeToResponsesState::default();
    let started = convert_claude_response_to_openai_responses(
        "claude-test",
        br#"{"model":"gpt-test"}"#,
        b"",
        br#"data: {"type":"message_start","message":{"id":"msg_failed"}}"#,
        &mut state,
    );
    assert_eq!(started.len(), 2);
    let failed = convert_claude_response_to_openai_responses(
        "claude-test",
        br#"{"model":"gpt-test"}"#,
        b"",
        br#"data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        &mut state,
    );
    assert_eq!(failed.len(), 1);
    let (name, payload) = event(&failed[0]);
    assert_eq!(name, "response.failed");
    assert_eq!(payload["sequence_number"], 3);
    assert_eq!(payload["response"]["status"], "failed");
    assert_eq!(payload["response"]["error"]["code"], "overloaded_error");
    assert_eq!(payload["response"]["error"]["message"], "Overloaded");

    assert!(convert_claude_response_to_openai_responses(
        "claude-test",
        b"",
        b"",
        br#"data: {"type":"message_stop"}"#,
        &mut state,
    )
    .is_empty());
}

#[test]
fn non_stream_preserves_interleaved_content_block_order() {
    let sse = b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_order\"}}\n\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"signature\":\"sig\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"why\"}}\n\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"first\"}}\n\n\
data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"lookup\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n\
data: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"text_delta\",\"text\":\"second\"}}\n\n";
    let response = value(&convert_claude_response_to_openai_responses_non_stream(
        br#"{"model":"claude-test"}"#,
        b"",
        sse,
    ));
    let output = response["output"].as_array().unwrap();
    assert_eq!(
        output
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["reasoning", "message", "function_call", "message"]
    );
    assert_eq!(output[1]["content"][0]["text"], "first");
    assert_eq!(output[3]["content"][0]["text"], "second");
}

// ref: internal/translator/openai/claude/openai_claude_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{convert_openai_response_to_claude, OpenAIToClaudeStreamState};
use crate::sdk::translator::TranslationContext;

#[derive(Debug, Clone)]
struct SseEvent {
    event: String,
    payload: String,
}

fn run_stream(original_req: &str, chunks: &[&str]) -> Vec<SseEvent> {
    let mut state = OpenAIToClaudeStreamState::default();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    for chunk in chunks {
        let mut raw = b"data: ".to_vec();
        raw.extend_from_slice(chunk.as_bytes());
        let mut out = convert_openai_response_to_claude(
            &TranslationContext::default(),
            "m",
            original_req.as_bytes(),
            b"{}",
            &raw,
            &mut state,
        );
        emitted.append(&mut out);
    }
    // Drive the [DONE] marker the same way upstream does.
    let mut done = convert_openai_response_to_claude(
        &TranslationContext::default(),
        "m",
        original_req.as_bytes(),
        b"{}",
        b"data: [DONE]",
        &mut state,
    );
    emitted.append(&mut done);

    let mut events: Vec<SseEvent> = Vec::new();
    for raw in &emitted {
        let s = String::from_utf8_lossy(raw).into_owned();
        let mut event: Option<String> = None;
        let mut payload: Option<String> = None;
        for line in s.split('\n') {
            if let Some(rest) = line.strip_prefix("event: ") {
                event = Some(rest.trim().to_owned());
            } else if let Some(rest) = line.strip_prefix("data: ") {
                payload = Some(rest.to_owned());
            }
        }
        if let (Some(event), Some(payload)) = (event, payload) {
            events.push(SseEvent { event, payload });
        }
    }
    events
}

fn count_by_type(events: &[SseEvent], typ: &str) -> usize {
    events.iter().filter(|event| event.event == typ).count()
}

fn tool_use_starts(events: &[SseEvent]) -> Vec<&SseEvent> {
    events
        .iter()
        .filter(|event| {
            event.event == "content_block_start"
                && gjson::get(&event.payload, "content_block.type").str() == "tool_use"
        })
        .collect()
}

fn block_indices(events: &[SseEvent]) -> Vec<i64> {
    events
        .iter()
        .filter(|event| event.event == "content_block_start")
        .map(|event| gjson::get(&event.payload, "index").i64())
        .collect()
}

fn last_stop_reason(events: &[SseEvent]) -> String {
    for event in events.iter().rev() {
        if event.event == "message_delta" {
            return gjson::get(&event.payload, "delta.stop_reason")
                .str()
                .to_owned();
        }
    }
    String::new()
}

const STREAM_REQ: &str = r#"{"stream":true}"#;

#[test]
fn late_usage_only_does_not_emit_after_message_stop() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
            r#"{"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
        ],
    );
    assert_eq!(count_by_type(&events, "message_delta"), 1);
    assert_eq!(count_by_type(&events, "message_stop"), 1);
    assert_eq!(
        events.last().map(|event| event.event.as_str()),
        Some("message_stop")
    );
}

#[test]
fn stream_ignores_null_tool_name_delta() {
    let original_request = STREAM_REQ.as_bytes();
    let mut state = OpenAIToClaudeStreamState::default();
    let mut first = convert_openai_response_to_claude(
        &TranslationContext::default(),
        "m",
        original_request,
        b"{}",
        br#"data: {"id":"chatcmpl_1","model":"m","created":1,"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#,
        &mut state,
    );
    let mut second = convert_openai_response_to_claude(
        &TranslationContext::default(),
        "m",
        original_request,
        b"{}",
        br#"data: {"id":"chatcmpl_1","model":"m","created":1,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":null,"arguments":"{\"path\":\"/tmp/a\"}"}}]},"finish_reason":null}]}"#,
        &mut state,
    );
    first.append(&mut second);
    let combined = first.concat();
    let combined_str = String::from_utf8_lossy(&combined);
    assert!(combined_str.contains("\"name\":\"read_file\""));
}

#[test]
fn empty_name_throughout_suppresses_tool_use_blocks() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","function":{"name":"","arguments":""}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":"{\"x\":1}"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    assert_eq!(tool_use_starts(&events).len(), 0, "events: {events:?}");
    assert_eq!(count_by_type(&events, "content_block_delta"), 0);
    assert_eq!(count_by_type(&events, "content_block_stop"), 0);
    assert_ne!(last_stop_reason(&events), "tool_use");
}

#[test]
fn null_name_suppresses_tool_use_start() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","function":{"name":null,"arguments":""}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    assert_eq!(tool_use_starts(&events).len(), 0);
    assert_eq!(count_by_type(&events, "content_block_stop"), 0);
}

#[test]
fn non_string_name_suppresses_tool_use_start() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","function":{"name":123,"arguments":""}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    assert_eq!(tool_use_starts(&events).len(), 0);
}

#[test]
fn repeated_name_emits_single_tool_use_start() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","function":{"name":"do_it","arguments":""}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"do_it","arguments":"{\"x\""}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"do_it","arguments":":1}"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 1, "events: {events:?}");
    assert_eq!(
        gjson::get(&starts[0].payload, "content_block.name").str(),
        "do_it"
    );
    assert_eq!(count_by_type(&events, "content_block_stop"), 1);
}

#[test]
fn mixed_suppressed_and_valid_tool_calls_yield_single_start() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[
                {"index":0,"id":"call_skip","function":{"name":"","arguments":""}},
                {"index":1,"id":"call_real","function":{"name":"do_it","arguments":""}}
            ]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[
                {"index":1,"function":{"arguments":"{}"}}
            ]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 1);
    assert_eq!(count_by_type(&events, "content_block_stop"), 1);
    let indices = block_indices(&events);
    assert!(!indices.is_empty() && indices[0] == 0);
}

#[test]
fn empty_id_defers_start_until_id_arrives() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"","function":{"name":"do_it","arguments":""}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_real","function":{"arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 1);
    assert_eq!(
        gjson::get(&starts[0].payload, "content_block.id").str(),
        "call_real"
    );
}

#[test]
fn id_in_delta_without_function_still_announces_tool_use() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"name":"do_it"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_real"}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 1);
    assert_eq!(
        gjson::get(&starts[0].payload, "content_block.id").str(),
        "call_real"
    );
    assert_eq!(
        gjson::get(&starts[0].payload, "content_block.name").str(),
        "do_it"
    );
    assert_eq!(count_by_type(&events, "content_block_stop"), 1);
}

#[test]
fn stop_reason_tool_use_is_emitted_when_tool_block_emitted() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","function":{"name":"do_it","arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
        ],
    );
    assert_eq!(last_stop_reason(&events), "tool_use");
}

#[test]
fn stop_reason_tool_use_emitted_with_belated_start_when_id_never_arrives() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"name":"do_it","arguments":""}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 1);
    let id = gjson::get(&starts[0].payload, "content_block.id")
        .str()
        .to_owned();
    assert!(id.starts_with("toolu_"), "synthetic id was {id}");
    assert_eq!(
        gjson::get(&starts[0].payload, "content_block.name").str(),
        "do_it"
    );
    assert_eq!(last_stop_reason(&events), "tool_use");
}

#[test]
fn belated_starts_use_openai_tool_index_order() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[
                {"index":2,"function":{"name":"third_tool","arguments":"{}"}},
                {"index":0,"function":{"name":"first_tool","arguments":"{}"}},
                {"index":1,"function":{"name":"second_tool","arguments":"{}"}}
            ]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 3, "starts: {starts:?}");
    let want_names = ["first_tool", "second_tool", "third_tool"];
    for (index, want_name) in want_names.iter().enumerate() {
        assert_eq!(
            gjson::get(&starts[index].payload, "content_block.name").str(),
            *want_name
        );
        assert_eq!(
            gjson::get(&starts[index].payload, "index").i64(),
            index as i64
        );
    }
}

#[test]
fn late_id_after_finalization_does_not_emit_block_after_message_stop() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"name":"do_it"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_late"}]}}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 1);
    let mut saw_stop = false;
    for event in &events {
        if event.event == "message_stop" {
            saw_stop = true;
            continue;
        }
        if saw_stop {
            assert!(
                !matches!(
                    event.event.as_str(),
                    "content_block_start" | "content_block_delta" | "content_block_stop"
                ),
                "event {} emitted after message_stop: {events:?}",
                event.event
            );
        }
    }
}

#[test]
fn stop_reason_tool_use_when_one_block_announces() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[
                {"index":0,"id":"call_skip","function":{"name":"","arguments":""}},
                {"index":1,"id":"call_real","function":{"name":"do_it","arguments":"{}"}}
            ]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    assert_eq!(last_stop_reason(&events), "tool_use");
}

// -- additional regression coverage for stateful behavior -------------------

#[test]
fn non_stream_emits_thinking_text_and_tool_use_in_order() {
    let mut state = OpenAIToClaudeStreamState::default();
    let raw = br#"{
        "id":"c1",
        "model":"m",
        "choices":[{
            "index":0,
            "message":{
                "role":"assistant",
                "reasoning_content":"think-think",
                "content":"answer",
                "tool_calls":[{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"do_work","arguments":"{\"a\":1}"}
                }]
            },
            "finish_reason":"tool_calls"
        }],
        "usage":{"prompt_tokens":7,"completion_tokens":3}
    }"#;
    let result = super::convert_openai_response_to_claude_non_stream(
        &TranslationContext::default(),
        "m",
        b"{}",
        b"{}",
        raw,
        &mut state,
    );
    let value: Value = serde_json::from_slice(&result).unwrap();
    assert_eq!(value["stop_reason"], "tool_use");
    let blocks = value["content"].as_array().unwrap();
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["thinking"], "think-think");
    assert_eq!(blocks[1]["type"], "text");
    assert_eq!(blocks[1]["text"], "answer");
    assert_eq!(blocks[2]["type"], "tool_use");
    assert_eq!(blocks[2]["id"], "call_1");
    assert_eq!(blocks[2]["name"], "do_work");
    assert_eq!(blocks[2]["input"]["a"], 1);
    assert_eq!(value["usage"]["input_tokens"], 7);
    assert_eq!(value["usage"]["output_tokens"], 3);
}

#[test]
fn non_stream_infers_tool_use_when_finish_reason_missing() {
    let mut state = OpenAIToClaudeStreamState::default();
    let raw = br#"{
        "id":"c1",
        "model":"m",
        "choices":[{
            "index":0,
            "message":{
                "role":"assistant",
                "content":null,
                "tool_calls":[{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"do_work","arguments":"{}"}
                }]
            }
        }],
        "usage":{"prompt_tokens":1,"completion_tokens":1}
    }"#;
    let result = super::convert_openai_response_to_claude_non_stream(
        &TranslationContext::default(),
        "m",
        b"{}",
        b"{}",
        raw,
        &mut state,
    );
    let value: Value = serde_json::from_slice(&result).unwrap();
    assert_eq!(value["stop_reason"], "tool_use");
}

#[test]
fn stream_message_start_only_fires_once_with_role_delta() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
    );
    assert_eq!(count_by_type(&events, "message_start"), 1);
    assert_eq!(last_stop_reason(&events), "end_turn");
    assert_eq!(count_by_type(&events, "message_stop"), 1);
}

#[test]
fn stream_text_block_stops_when_finish_reason_arrives() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
    );
    let text_block_starts = events
        .iter()
        .filter(|event| {
            event.event == "content_block_start"
                && gjson::get(&event.payload, "content_block.type").str() == "text"
        })
        .count();
    assert_eq!(text_block_starts, 1);
    let text_block_stops = events
        .iter()
        .filter(|event| {
            event.event == "content_block_stop" && gjson::get(&event.payload, "index").i64() == 0
        })
        .count();
    assert_eq!(text_block_stops, 1);
}

#[test]
fn stream_cached_usage_subtracts_cached_tokens() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"x"}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":3}}}"#,
        ],
    );
    let message_delta = events
        .iter()
        .find(|event| event.event == "message_delta")
        .expect("message_delta event");
    let payload: Value = serde_json::from_str(&message_delta.payload).unwrap();
    assert_eq!(payload["usage"]["input_tokens"], 7);
    assert_eq!(payload["usage"]["output_tokens"], 2);
    assert_eq!(payload["usage"]["cache_read_input_tokens"], 3);
}

#[test]
fn stream_reasoning_content_uses_thinking_block_index() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"why"}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
    );
    let thinking_starts: Vec<&SseEvent> = events
        .iter()
        .filter(|event| {
            event.event == "content_block_start"
                && gjson::get(&event.payload, "content_block.type").str() == "thinking"
        })
        .collect();
    assert_eq!(thinking_starts.len(), 1);
    let thinking_index = gjson::get(&thinking_starts[0].payload, "index").i64();
    let delta = events
        .iter()
        .find(|event| {
            event.event == "content_block_delta"
                && gjson::get(&event.payload, "delta.type").str() == "thinking_delta"
        })
        .expect("thinking delta");
    assert_eq!(gjson::get(&delta.payload, "index").i64(), thinking_index);
    assert_eq!(gjson::get(&delta.payload, "delta.thinking").str(), "why");
}

#[test]
fn tool_name_map_restores_original_casing() {
    let original_request = r#"{"tools":[{"name":"Read_File"}]}"#;
    let events = run_stream(
        original_request,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 1);
    assert_eq!(
        gjson::get(&starts[0].payload, "content_block.name").str(),
        "Read_File"
    );
}

#[test]
fn stream_with_synthetic_id_uses_sanitize_format() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"name":"do_it","arguments":""}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
    );
    let starts = tool_use_starts(&events);
    assert_eq!(starts.len(), 1);
    let id = gjson::get(&starts[0].payload, "content_block.id")
        .str()
        .to_owned();
    assert!(id.starts_with("toolu_"));
}

#[test]
fn stream_message_start_carries_message_id_and_model() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"chatcmpl_1","model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}"#,
            r#"{"id":"chatcmpl_1","model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
    );
    let start = events
        .iter()
        .find(|event| event.event == "message_start")
        .expect("message_start");
    let payload: Value = serde_json::from_str(&start.payload).unwrap();
    assert_eq!(payload["message"]["id"], "chatcmpl_1");
    assert_eq!(payload["message"]["model"], "gpt-5");
}

#[test]
fn stream_done_marker_closes_text_block_when_no_finish_reason() {
    let events = run_stream(
        STREAM_REQ,
        &[
            r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}"#,
        ],
    );
    let content_block_stops = events
        .iter()
        .filter(|event| event.event == "content_block_stop")
        .count();
    assert_eq!(content_block_stops, 1);
    assert_eq!(count_by_type(&events, "message_stop"), 1);
}

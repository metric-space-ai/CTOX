// ref: test/codex_claude_parallel_function_calls_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::internal::translator::register_all;
use crate::sdk::translator::{claude, codex, Registry, TranslationContext, TranslationState};

#[test]
fn codex_to_claude_parallel_function_calls_have_valid_lifecycle() {
    let chunks: &[&[u8]] = &[
        br#"data: {"type":"response.created","response":{"id":"resp_parallel","model":"gpt-5"}}"#,
        br#"data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_a","name":"Read"},"output_index":1}"#,
        br#"data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_b","name":"Read"},"output_index":2}"#,
        br#"data: {"type":"response.function_call_arguments.delta","delta":"{\"file_path\":\"a\"}","output_index":1}"#,
        br#"data: {"type":"response.function_call_arguments.done","arguments":"{\"file_path\":\"a\"}","output_index":1}"#,
        br#"data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_a","name":"Read","arguments":"{\"file_path\":\"a\"}"},"output_index":1}"#,
        br#"data: {"type":"response.function_call_arguments.delta","delta":"{\"file_path\":\"b\"}","output_index":2}"#,
        br#"data: {"type":"response.function_call_arguments.done","arguments":"{\"file_path\":\"b\"}","output_index":2}"#,
        br#"data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_b","name":"Read","arguments":"{\"file_path\":\"b\"}"},"output_index":2}"#,
        br#"data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1},"output":[{"type":"function_call","call_id":"call_a","name":"Read","arguments":"{\"file_path\":\"a\"}"},{"type":"function_call","call_id":"call_b","name":"Read","arguments":"{\"file_path\":\"b\"}"}]}}"#,
    ];
    let registry = Registry::new();
    register_all(&registry);
    let mut state: TranslationState = None;
    let mut open = BTreeSet::new();
    let mut starts = Vec::new();
    let mut stops = Vec::new();
    let mut tool_ids = BTreeMap::new();
    let mut arguments = BTreeMap::<i64, String>::new();
    let mut message_state = 0;

    for chunk in chunks {
        for output in registry.translate_stream(
            &TranslationContext::default(),
            &codex(),
            &claude(),
            "gpt-5",
            br#"{"stream":true,"tools":[{"name":"Read"}]}"#,
            &[],
            chunk,
            &mut state,
        ) {
            for data in output
                .split(|byte| *byte == b'\n')
                .filter_map(|line| line.strip_prefix(b"data: "))
            {
                let event: Value = serde_json::from_slice(data).unwrap();
                assert_ne!(message_state, 2, "event after message_stop: {event}");
                let index = event["index"].as_i64().unwrap_or_default();
                match event["type"].as_str().unwrap_or_default() {
                    "content_block_start" => {
                        assert!(open.is_empty());
                        assert!(open.insert(index));
                        starts.push(index);
                        tool_ids.insert(
                            index,
                            event["content_block"]["id"].as_str().unwrap().to_owned(),
                        );
                    }
                    "content_block_delta" => {
                        assert!(open.contains(&index));
                        if event["delta"]["type"] == "input_json_delta" {
                            arguments
                                .entry(index)
                                .or_default()
                                .push_str(event["delta"]["partial_json"].as_str().unwrap());
                        }
                    }
                    "content_block_stop" => {
                        assert!(open.remove(&index));
                        stops.push(index);
                    }
                    "message_delta" => {
                        assert!(open.is_empty());
                        assert_eq!(message_state, 0);
                        message_state = 1;
                    }
                    "message_stop" => {
                        assert!(open.is_empty());
                        assert_eq!(message_state, 1);
                        message_state = 2;
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(open.is_empty());
    assert_eq!(message_state, 2);
    assert_eq!(starts, [0, 1]);
    assert_eq!(stops, [0, 1]);
    assert_eq!(tool_ids[&0], "call_a");
    assert_eq!(tool_ids[&1], "call_b");
    assert_eq!(arguments[&0], r#"{"file_path":"a"}"#);
    assert_eq!(arguments[&1], r#"{"file_path":"b"}"#);
}

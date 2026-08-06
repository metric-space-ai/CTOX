// ref: internal/translator/antigravity/claude/antigravity_claude_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

// The 35 upstream tests are dispositioned across executable evidence rather
// than copied one-for-one: 7 aggregate and 5 streaming Web Search differential
// cases; 12 aggregate and 12 normal-stream differential cases; the dedicated
// carrier differential corpus; and the direct state/cache/ordering tests below.
// This keeps every upstream behavior family pinned while testing the CTOX state
// split and terminal-once adaptations explicitly.

use std::sync::Mutex;
use std::time::Duration;

use serde_json::Value;

use super::{
    convert_antigravity_response_to_claude_non_stream,
    convert_antigravity_response_to_claude_stream,
    convert_antigravity_response_to_claude_stream_with_runtime,
    convert_antigravity_web_search_response_to_claude_non_stream,
    convert_antigravity_web_search_response_to_claude_stream, AntigravityClaudeStreamState,
    AntigravityClaudeWebSearchStreamState,
};
use crate::internal::cache::{
    clear_signature_cache, get_cached_signature, signature_cache_test_guard, signature_kv_key,
    SignatureCacheStoreError, SignatureKvStore,
};
use crate::internal::util::is_gemini_claude_tool_use_id;

const ORIGINAL_WEB_SEARCH: &[u8] =
    br#"{"tools":[{"type":"web_search_20250305","name":"web_search"}]}"#;
const TRANSLATED_WEB_SEARCH: &[u8] =
    br#"{"model":"gemini-3.1-flash-lite","request":{"tools":[{"googleSearch":{}}]}}"#;

fn grounding_response() -> &'static [u8] {
    br#"{"response":{"modelVersion":"gemini-3.1-flash-lite","responseId":"resp-search","candidates":[{"content":{"parts":[{"text":"Weather is clear today."}]},"groundingMetadata":{"webSearchQueries":["weather"],"groundingChunks":[{"web":{"uri":"https://example.com/weather","title":"Weather"}},{"web":{"uri":"https://example.com/weather","title":"Duplicate"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":7,"text":"Weather"},"groundingChunkIndices":[0]}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"thoughtsTokenCount":2,"totalTokenCount":16,"cachedContentTokenCount":3}}}"#
}

#[test]
fn native_grounding_builds_tool_result_citation_and_usage() {
    let output = convert_antigravity_web_search_response_to_claude_non_stream(
        ORIGINAL_WEB_SEARCH,
        TRANSLATED_WEB_SEARCH,
        grounding_response(),
        "srvtoolu_fixed",
    )
    .unwrap();
    let output: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(output["content"][0]["type"], "server_tool_use");
    assert_eq!(output["content"][0]["input"]["query"], "weather");
    assert_eq!(output["content"][1]["type"], "web_search_tool_result");
    assert_eq!(output["content"][1]["content"].as_array().unwrap().len(), 1);
    assert_eq!(
        output["content"][2]["citations"][0]["url"],
        "https://example.com/weather"
    );
    assert_eq!(output["usage"]["output_tokens"], 6);
    assert_eq!(output["usage"]["cache_read_input_tokens"], 3);
    assert_eq!(output["usage"]["server_tool_use"]["web_search_requests"], 1);
}

#[test]
fn streaming_grounding_emits_tool_blocks_citations_and_terminal_usage() {
    let mut state = AntigravityClaudeWebSearchStreamState::default();
    let chunk = br#"{"response":{"modelVersion":"gemini-3.1-flash-lite","responseId":"resp-stream","cpaUsageMetadata":{"promptTokenCount":13,"candidatesTokenCount":99},"candidates":[{"content":{"parts":[{"text":"Weather is clear."}]},"groundingMetadata":{"webSearchQueries":["weather"],"groundingChunks":[{"web":{"uri":"https://example.com/weather","title":"Weather"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":7},"groundingChunkIndices":[0]}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":13,"candidatesTokenCount":4,"thoughtsTokenCount":2,"totalTokenCount":19}}}"#;
    let mut events = convert_antigravity_web_search_response_to_claude_stream(
        ORIGINAL_WEB_SEARCH,
        TRANSLATED_WEB_SEARCH,
        chunk,
        &mut state,
        "srvtoolu_fixed",
    );
    events.extend(convert_antigravity_web_search_response_to_claude_stream(
        ORIGINAL_WEB_SEARCH,
        TRANSLATED_WEB_SEARCH,
        b"[DONE]",
        &mut state,
        "srvtoolu_fixed",
    ));
    let events = events
        .iter()
        .map(|event| sse_data(event))
        .collect::<Vec<_>>();

    assert_eq!(events[0]["type"], "message_start");
    assert_eq!(events[0]["message"]["usage"]["input_tokens"], 13);
    assert_eq!(events[0]["message"]["usage"]["output_tokens"], 0);
    assert_eq!(events[1]["content_block"]["type"], "server_tool_use");
    assert_eq!(events[4]["content_block"]["type"], "web_search_tool_result");
    assert!(events
        .iter()
        .any(|event| event["delta"]["type"] == "citations_delta"));
    let message_delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .unwrap();
    assert_eq!(message_delta["usage"]["output_tokens"], 6);
    assert_eq!(
        message_delta["usage"]["server_tool_use"]["web_search_requests"],
        1
    );
    assert_eq!(events.last().unwrap()["type"], "message_stop");
}

#[test]
fn streaming_fallback_flushes_buffer_and_sends_terminal_only_once() {
    let mut state = AntigravityClaudeWebSearchStreamState::default();
    let first =
        br#"{"response":{"candidates":[{"content":{"parts":[{"text":"plain fallback"}]}}]}}"#;
    let finish = br#"{"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":10,"totalTokenCount":16,"cachedContentTokenCount":3}}}"#;
    let _ = convert_antigravity_web_search_response_to_claude_stream(
        ORIGINAL_WEB_SEARCH,
        TRANSLATED_WEB_SEARCH,
        first,
        &mut state,
        "srvtoolu_fixed",
    );
    let mut events = convert_antigravity_web_search_response_to_claude_stream(
        ORIGINAL_WEB_SEARCH,
        TRANSLATED_WEB_SEARCH,
        finish,
        &mut state,
        "srvtoolu_fixed",
    );
    events.extend(convert_antigravity_web_search_response_to_claude_stream(
        ORIGINAL_WEB_SEARCH,
        TRANSLATED_WEB_SEARCH,
        b"[DONE]",
        &mut state,
        "srvtoolu_fixed",
    ));
    assert!(convert_antigravity_web_search_response_to_claude_stream(
        ORIGINAL_WEB_SEARCH,
        TRANSLATED_WEB_SEARCH,
        b"[DONE]",
        &mut state,
        "srvtoolu_fixed",
    )
    .is_empty());
    let events = events
        .iter()
        .map(|event| sse_data(event))
        .collect::<Vec<_>>();
    assert_eq!(events[1]["delta"]["text"], "plain fallback");
    let message_delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .unwrap();
    assert_eq!(message_delta["delta"]["stop_reason"], "max_tokens");
    assert_eq!(message_delta["usage"]["input_tokens"], 7);
    assert_eq!(message_delta["usage"]["output_tokens"], 9);
    assert_eq!(message_delta["usage"]["cache_read_input_tokens"], 3);
}

fn sse_data(event: &[u8]) -> Value {
    let text = std::str::from_utf8(event).unwrap();
    let data = text
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    serde_json::from_str(data).unwrap()
}

#[test]
fn grounding_requires_both_original_typed_tool_and_translated_native_tool() {
    assert!(
        convert_antigravity_web_search_response_to_claude_non_stream(
            br#"{"tools":[{"type":"web_search_20250305"}]}"#,
            br#"{"request":{"contents":[]}}"#,
            grounding_response(),
            "srvtoolu_fixed",
        )
        .is_none()
    );
    assert!(
        convert_antigravity_web_search_response_to_claude_non_stream(
            br#"{"tools":[]}"#,
            br#"{"request":{"tools":[{"googleSearch":{}}]}}"#,
            grounding_response(),
            "srvtoolu_fixed",
        )
        .is_none()
    );
}

#[test]
fn normal_aggregate_preserves_thinking_tool_text_order_and_provenance() {
    let original = br#"{"tools":[{"name":"read file"}]}"#;
    let translated = br#"{"model":"gemini-3.1-pro"}"#;
    let response = br#"{"response":{"responseId":"normal","modelVersion":"gemini-3.1-pro","candidates":[{"content":{"parts":[{"thought":true,"text":"plan"},{"functionCall":{"id":"native-1","name":"read_file","args":{"path":"/tmp/a"}},"thoughtSignature":"tool-signature"},{"text":"done"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"thoughtsTokenCount":1}}}"#;
    let output: Value = serde_json::from_slice(&convert_antigravity_response_to_claude_non_stream(
        original, translated, response, "unused",
    ))
    .unwrap();

    assert_eq!(output["content"][0]["type"], "thinking");
    assert!(output["content"][0]["signature"]
        .as_str()
        .unwrap()
        .starts_with("cpa-gemini-carrier-v1:next:function:"));
    assert_eq!(output["content"][1]["type"], "tool_use");
    assert_eq!(output["content"][1]["name"], "read file");
    assert!(is_gemini_claude_tool_use_id(
        output["content"][1]["id"].as_str().unwrap()
    ));
    assert_eq!(output["content"][2]["text"], "done");
    assert_eq!(output["stop_reason"], "tool_use");
    assert_eq!(output["usage"]["output_tokens"], 3);
}

#[test]
fn normal_aggregate_omits_absent_usage_and_clamps_total_fallback() {
    let no_usage: Value =
        serde_json::from_slice(&convert_antigravity_response_to_claude_non_stream(
            br#"{}"#,
            br#"{"model":"other"}"#,
            br#"{"response":{"candidates":[{"content":{"parts":[]}}]}}"#,
            "unused",
        ))
        .unwrap();
    assert!(no_usage.get("usage").is_none());

    let clamped: Value = serde_json::from_slice(
        &convert_antigravity_response_to_claude_non_stream(
            br#"{}"#,
            br#"{"model":"other"}"#,
            br#"{"response":{"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":10,"totalTokenCount":4}}}"#,
            "unused",
        ),
    )
    .unwrap();
    assert_eq!(clamped["usage"]["output_tokens"], 0);
    assert_eq!(clamped["stop_reason"], "max_tokens");
}

#[test]
fn normal_stream_orders_thinking_signature_text_and_final_usage() {
    let mut state = AntigravityClaudeStreamState::default();
    let original = br#"{}"#;
    let translated = br#"{"model":"gemini-3.1-pro"}"#;
    let chunks: [&[u8]; 2] = [
        br#"{"response":{"responseId":"stream","candidates":[{"content":{"parts":[{"thought":true,"text":"plan","thoughtSignature":"provider-signature"}]}}]}}"#,
        br#"{"response":{"candidates":[{"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":1,"thoughtsTokenCount":1}}}"#,
    ];
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(convert_antigravity_response_to_claude_stream(
            original, translated, chunk, &mut state, "unused",
        ));
    }
    events.extend(convert_antigravity_response_to_claude_stream(
        original, translated, b"[DONE]", &mut state, "unused",
    ));
    let events = events
        .iter()
        .map(|event| sse_data(event))
        .collect::<Vec<_>>();
    assert_eq!(events[0]["type"], "message_start");
    assert!(events
        .iter()
        .any(|event| event["delta"]["type"] == "thinking_delta"));
    assert!(events
        .iter()
        .any(|event| event["delta"]["type"] == "signature_delta"));
    assert!(events
        .iter()
        .any(|event| event["delta"]["text"] == "answer"));
    let final_delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .unwrap();
    assert_eq!(final_delta["usage"]["output_tokens"], 2);
    assert_eq!(events.last().unwrap()["type"], "message_stop");
}

#[test]
fn normal_stream_synthesizes_empty_content_and_terminal_only_once() {
    let mut state = AntigravityClaudeStreamState::default();
    let _ = convert_antigravity_response_to_claude_stream(
        br#"{}"#,
        br#"{"model":"other"}"#,
        br#"{"response":{"candidates":[]}}"#,
        &mut state,
        "unused",
    );
    let terminal = convert_antigravity_response_to_claude_stream(
        br#"{}"#,
        br#"{"model":"other"}"#,
        b"[DONE]",
        &mut state,
        "unused",
    );
    assert_eq!(sse_data(&terminal[0])["type"], "content_block_start");
    assert_eq!(sse_data(terminal.last().unwrap())["type"], "message_stop");
    assert!(convert_antigravity_response_to_claude_stream(
        br#"{}"#,
        br#"{"model":"other"}"#,
        b"[DONE]",
        &mut state,
        "unused",
    )
    .is_empty());
}

#[test]
fn normal_stream_accumulates_thinking_and_caches_signature_only_chunk() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    let mut state = AntigravityClaudeStreamState::default();
    let model = br#"{"model":"claude-sonnet-4-5-thinking"}"#;
    let signature = "signature_12345678901234567890123456789012345678901234567890";
    let chunks = vec![
        br#"{"response":{"candidates":[{"content":{"parts":[{"text":"First part. ","thought":true}]}}]}}"#.to_vec(),
        br#"{"response":{"candidates":[{"content":{"parts":[{"text":"Second part.","thought":true}]}}]}}"#.to_vec(),
        format!(r#"{{"response":{{"candidates":[{{"content":{{"parts":[{{"text":"","thoughtSignature":"{signature}"}}]}}}}]}}}}"#).into_bytes(),
    ];
    for chunk in chunks {
        let _ = convert_antigravity_response_to_claude_stream(
            br#"{}"#, model, &chunk, &mut state, "unused",
        );
    }
    assert_eq!(
        get_cached_signature("claude-sonnet-4-5-thinking", "First part. Second part."),
        signature
    );
    clear_signature_cache("");
}

#[derive(Default)]
struct ResponseSignatureStore {
    fail_set: bool,
    writes: Mutex<Vec<(String, Vec<u8>, Duration)>>,
}

impl SignatureKvStore for ResponseSignatureStore {
    fn get(&self, _key: &str) -> Result<Option<Vec<u8>>, SignatureCacheStoreError> {
        Ok(None)
    }

    fn set(
        &self,
        key: &str,
        value: &[u8],
        ttl: Duration,
    ) -> Result<bool, SignatureCacheStoreError> {
        if self.fail_set {
            return Err(SignatureCacheStoreError::Write);
        }
        self.writes
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push((key.to_owned(), value.to_vec(), ttl));
        Ok(true)
    }

    fn delete(&self, _key: &str) -> Result<bool, SignatureCacheStoreError> {
        Ok(false)
    }

    fn expire(&self, _key: &str, _ttl: Duration) -> Result<bool, SignatureCacheStoreError> {
        Ok(false)
    }
}

#[test]
fn normal_stream_publishes_signature_directly_to_durable_store() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    let store = ResponseSignatureStore::default();
    let model_name = "claude-sonnet-4-5-thinking";
    let translated = format!(r#"{{"model":"{model_name}"}}"#);
    let thinking = "durable thinking";
    let signature = "signature_12345678901234567890123456789012345678901234567890";
    let chunks = [
        format!(
            r#"{{"response":{{"candidates":[{{"content":{{"parts":[{{"text":"{thinking}","thought":true}}]}}}}]}}}}"#
        ),
        format!(
            r#"{{"response":{{"candidates":[{{"content":{{"parts":[{{"text":"","thoughtSignature":"{signature}"}}]}}}}]}}}}"#
        ),
    ];
    let mut state = AntigravityClaudeStreamState::default();
    for chunk in chunks {
        let _ = convert_antigravity_response_to_claude_stream_with_runtime(
            br#"{}"#,
            translated.as_bytes(),
            chunk.as_bytes(),
            &mut state,
            "unused",
            Some(&store),
        );
    }

    let writes = store
        .writes
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0].0, signature_kv_key(model_name, thinking));
    assert_eq!(writes[0].1, signature.as_bytes());
    assert_eq!(writes[0].2, Duration::from_secs(3 * 60 * 60));
    assert!(get_cached_signature(model_name, thinking).is_empty());
}

#[test]
fn durable_publication_failure_is_best_effort_without_local_fallback() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    let store = ResponseSignatureStore {
        fail_set: true,
        ..ResponseSignatureStore::default()
    };
    let model_name = "claude-sonnet-4-5-thinking";
    let translated = format!(r#"{{"model":"{model_name}"}}"#);
    let thinking = br#"{"response":{"candidates":[{"content":{"parts":[{"text":"unpublished thinking","thought":true}]}}]}}"#;
    let signature = br#"{"response":{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"signature_12345678901234567890123456789012345678901234567890"}]}}]}}"#;
    let mut state = AntigravityClaudeStreamState::default();
    let _ = convert_antigravity_response_to_claude_stream_with_runtime(
        br#"{}"#,
        translated.as_bytes(),
        thinking,
        &mut state,
        "unused",
        Some(&store),
    );
    let events = convert_antigravity_response_to_claude_stream_with_runtime(
        br#"{}"#,
        translated.as_bytes(),
        signature,
        &mut state,
        "unused",
        Some(&store),
    );

    assert!(events
        .iter()
        .map(|event| sse_data(event))
        .any(|event| event["delta"]["type"] == "signature_delta"));
    assert!(get_cached_signature(model_name, "unpublished thinking").is_empty());
}

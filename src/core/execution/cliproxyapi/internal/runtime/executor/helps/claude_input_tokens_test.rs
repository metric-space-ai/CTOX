// ref: internal/runtime/executor/helps/claude_input_tokens_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::claude_input_tokens::{
    collect_claude_input_token_segments, count_claude_input_tokens, ClaudeInputTokenError,
    ClaudeInputTokenFailureSink, ClaudeInputTokenState,
};
use crate::sdk::translator::{claude, gemini, openai, Format};

#[test]
fn collects_text_tools_documents_and_lexical_json_in_upstream_order() {
    let payload = br#"{
        "system":[
            {"type":"text","text":"Follow repository rules."},
            {"type":"image","source":{"data":"ignored-system-image"}}
        ],
        "messages":[
            {"role":"user","content":[
                {"type":"text","text":"Review the implementation."},
                {"type":"document","source":{"type":"text","data":"Reference document text."}},
                {"type":"image","source":{"data":"ignored-image"}}
            ]},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"Inspect the relevant files.","signature":"ignored"},
                {"type":"tool_use","id":"toolu_1","name":"read_file","input": { "z": 1e2, "a": "x y" }}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"package main"}]}
            ]}
        ],
        "tools":[{"name":"read_file","description":"Reads a repository file.","input_schema": { "type": "object", "properties": {"path":{"type":"string"}} }}],
        "tool_choice":{"type":"tool","name":"read_file"}
    }"#;

    let segments = collect_claude_input_token_segments(payload).unwrap();

    assert_eq!(
        segments,
        [
            "Follow repository rules.",
            "user",
            "Review the implementation.",
            "Reference document text.",
            "assistant",
            "Inspect the relevant files.",
            "toolu_1",
            "read_file",
            r#"{"z":1e2,"a":"x y"}"#,
            "user",
            "toolu_1",
            "package main",
            "read_file",
            "Reads a repository file.",
            r#"{"type":"object","properties":{"path":{"type":"string"}}}"#,
            "tool",
            "read_file",
        ]
    );
}

#[test]
fn known_tool_results_are_counted_without_encrypted_or_multimedia_payloads() {
    let payload = br#"{
      "messages":[{"role":"user","content":[
        {"type":"web_search_tool_result","tool_use_id":"ws_1","content":[
          {"type":"web_search_result","source":"source","title":"title","url":"https://example","page_age":"1 day","encrypted_content":"secret"}
        ]},
        {"type":"bash_code_execution_tool_result","tool_use_id":"bash_1","content":{
          "type":"bash_code_execution_result","stdout":"out","stderr":"err","return_code":1,
          "content":[{"type":"image","source":{"data":"image-secret"}},{"type":"text","text":"extra"}]
        }}
      ]}]
    }"#;

    let segments = collect_claude_input_token_segments(payload).unwrap();
    let joined = segments.join("\n");

    for expected in [
        "ws_1",
        "source",
        "title",
        "https://example",
        "1 day",
        "bash_1",
        "out",
        "err",
        "1",
        "extra",
    ] {
        assert!(segments.iter().any(|segment| segment == expected));
    }
    assert!(!joined.contains("secret"));
    assert!(!joined.contains("image-secret"));
}

#[test]
fn count_is_deterministic_and_rejects_invalid_json() {
    let payload = r#"{"messages":[{"role":"user","content":"Hello 你好"}]}"#.as_bytes();
    let first = count_claude_input_tokens(payload).unwrap();
    let second = count_claude_input_tokens(payload).unwrap();
    assert!(first > 0);
    assert_eq!(first, second);
    assert_eq!(
        count_claude_input_tokens(br#"{"messages":["#),
        Err(ClaudeInputTokenError::InvalidJson)
    );
    assert_eq!(count_claude_input_tokens(b" \n\t ").unwrap(), 0);
}

fn message_start_tokens(chunks: &[Vec<u8>]) -> i64 {
    chunks
        .iter()
        .flat_map(|chunk| chunk.split(|byte| *byte == b'\n'))
        .find_map(|line| {
            let line = std::str::from_utf8(line).ok()?.trim();
            let payload = line.strip_prefix("data:")?.trim();
            let value: serde_json::Value = serde_json::from_str(payload).ok()?;
            (value.get("type")?.as_str()? == "message_start")
                .then(|| value.pointer("/message/usage/input_tokens")?.as_i64())
                .flatten()
        })
        .unwrap_or_default()
}

#[test]
fn patches_message_start_once_and_preserves_other_events() {
    let original = br#"{"system":"System text.","messages":[{"role":"user","content":"Hello."}]}"#;
    let upstream = Format::from("test-upstream");
    let mut state = ClaudeInputTokenState::new(&claude(), &upstream, &claude(), original);
    let combined = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n".to_vec();
    let output = state.apply(vec![combined]);
    assert!(message_start_tokens(&output) > 0);
    assert!(state.handled());
    assert!(String::from_utf8_lossy(&output.concat()).contains("\"type\":\"ping\""));

    let second = state.apply(vec![
        b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0}}}\n\n"
            .to_vec(),
    ]);
    assert_eq!(message_start_tokens(&second), 0);
}

#[test]
fn preserves_crlf_spacing_and_nonzero_usage() {
    let original = br#"{"messages":[{"role":"user","content":"Hello."}]}"#;
    let mut state = ClaudeInputTokenState::new(&claude(), &openai(), &claude(), original);
    let chunk = b"event: message_start\r\ndata:  {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}  \r\n\r\nevent: ping\r\n".to_vec();
    let output = state.apply(vec![chunk]);
    let tokens = message_start_tokens(&output);
    assert!(tokens > 0);
    let expected = format!("event: message_start\r\ndata:  {{\"type\":\"message_start\",\"message\":{{\"usage\":{{\"input_tokens\":{tokens},\"output_tokens\":0}}}}}}  \r\n\r\nevent: ping\r\n");
    assert_eq!(output.concat(), expected.as_bytes());

    let mut existing = ClaudeInputTokenState::new(&claude(), &openai(), &claude(), b"invalid");
    let output = existing.apply(vec![
        b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":73}}}\n"
            .to_vec(),
    ]);
    assert_eq!(message_start_tokens(&output), 73);
}

#[test]
fn patches_missing_usage_and_skips_unsupported_routes() {
    let original = br#"{"messages":[{"role":"user","content":"Hello."}]}"#;
    let event =
        b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"output_tokens\":0}}}\n"
            .to_vec();
    let mut enabled = ClaudeInputTokenState::new(&claude(), &openai(), &claude(), original);
    assert!(message_start_tokens(&enabled.apply(vec![event.clone()])) > 0);

    for (source, upstream, response) in [
        (openai(), gemini(), claude()),
        (claude(), claude(), claude()),
        (claude(), openai(), openai()),
    ] {
        let mut state = ClaudeInputTokenState::new(&source, &upstream, &response, original);
        assert!(state.handled());
        assert_eq!(state.apply(vec![event.clone()]), vec![event.clone()]);
    }
}

#[test]
fn concurrent_tokenizer_counts_are_positive() {
    let workers = (0..16).map(|worker| {
        std::thread::spawn(move || {
            for iteration in 0..25 {
                let payload = format!(r#"{{"messages":[{{"role":"user","content":"worker {worker} iteration {iteration} 你好"}}]}}"#);
                assert!(count_claude_input_tokens(payload.as_bytes()).unwrap() > 0);
            }
        })
    });
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn invalid_request_reports_typed_failure_without_request_bytes() {
    #[derive(Default)]
    struct Sink(std::sync::Mutex<Vec<(String, String, ClaudeInputTokenError)>>);
    impl ClaudeInputTokenFailureSink for Sink {
        fn estimation_failed(
            &self,
            upstream: &Format,
            response: &Format,
            error: ClaudeInputTokenError,
        ) {
            self.0
                .lock()
                .unwrap()
                .push((upstream.to_string(), response.to_string(), error));
        }
    }
    let sink = std::sync::Arc::new(Sink::default());
    let mut state = ClaudeInputTokenState::new(
        &claude(),
        &openai(),
        &claude(),
        br#"{"messages":["sensitive-secret""#,
    )
    .with_failure_sink(sink.clone());
    let event =
        b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0}}}\n"
            .to_vec();
    assert_eq!(state.apply(vec![event.clone()]), vec![event]);
    assert_eq!(
        *sink.0.lock().unwrap(),
        [(
            "openai".into(),
            "claude".into(),
            ClaudeInputTokenError::InvalidJson
        )]
    );
    assert!(!format!("{state:?}").contains("sensitive-secret"));
}

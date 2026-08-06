// ref: internal/runtime/executor/codex_executor_stream_output_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::SystemTime;

use super::codex_executor_terminal::{CodexTerminalAccumulator, CodexTerminalEvent};

#[test]
fn stream_output_commits_only_at_terminal_event() {
    let mut accumulator = CodexTerminalAccumulator::default();
    assert_eq!(
        accumulator.ingest(
            br#"{"type":"response.output_item.done","output_index":0,"item":{"id":"x"}}"#,
            SystemTime::UNIX_EPOCH,
        ),
        CodexTerminalEvent::Continue
    );
    assert!(!accumulator.committed());
    let CodexTerminalEvent::Completed(body) = accumulator.ingest(
        br#"{"type":"response.completed","response":{"output":[]}}"#,
        SystemTime::UNIX_EPOCH,
    ) else {
        panic!("completion expected")
    };
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["output"][0]["id"],
        "x"
    );
}

#[test]
fn candidate_nonempty_completion_hydrates_only_missing_item_ids() {
    let mut accumulator = CodexTerminalAccumulator::default();
    for event in [
        br#"{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_123","type":"function_call","name":"weather"}}"#.as_slice(),
        br#"{"type":"response.output_item.done","output_index":1,"item":{"id":"fc_done_existing","type":"function_call","name":"other"}}"#.as_slice(),
    ] {
        assert_eq!(
            accumulator.ingest(event, SystemTime::UNIX_EPOCH),
            CodexTerminalEvent::Continue
        );
    }
    let CodexTerminalEvent::Completed(body) = accumulator.ingest(
        br#"{"type":"response.completed","response":{"output":[{"id":null,"type":"function_call","name":"weather-terminal"},{"id":"fc_existing","type":"function_call","name":"preserved"}]}}"#,
        SystemTime::UNIX_EPOCH,
    ) else {
        panic!("completion expected")
    };
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["output"][0]["id"], "fc_123");
    assert_eq!(value["output"][0]["name"], "weather-terminal");
    assert_eq!(value["output"][1]["id"], "fc_existing");
    assert_eq!(value["output"][1]["name"], "preserved");
}

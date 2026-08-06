// ref: internal/translator/openai/openai/chat-completions/openai_openai_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_openai_request_to_openai;

#[test]
fn converter_replaces_model_and_preserves_the_rest() {
    let output = convert_openai_request_to_openai(
        "gpt-5.6",
        br#"{"model":"old","messages":[{"role":"user","content":"hello"}],"temperature":0.2}"#,
        false,
    );
    let output: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(output["model"], "gpt-5.6");
    assert_eq!(output["messages"][0]["content"], "hello");
    assert_eq!(output["temperature"], 0.2);
}

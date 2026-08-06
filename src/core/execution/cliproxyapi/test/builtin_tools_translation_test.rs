// ref: test/builtin_tools_translation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::translator::register_all;
use crate::sdk::translator::{codex, openai, openai_response, Registry, TranslationContext};

#[test]
fn openai_to_codex_preserves_builtin_tools() {
    let input = br#"{
        "model":"gpt-5",
        "messages":[{"role":"user","content":"hi"}],
        "tools":[{"type":"web_search","search_context_size":"high"}],
        "tool_choice":{"type":"web_search"}
    }"#;
    let registry = Registry::new();
    register_all(&registry);
    let output = registry.translate_request(
        &TranslationContext::default(),
        &openai(),
        &codex(),
        "gpt-5",
        input,
        false,
    );
    let output: Value = serde_json::from_slice(&output).unwrap();
    let tools = output["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1, "body={output}");
    assert_eq!(tools[0]["type"], "web_search", "body={output}");
    assert_eq!(tools[0]["search_context_size"], "high", "body={output}");
    assert_eq!(output["tool_choice"]["type"], "web_search", "body={output}");
}

#[test]
fn openai_responses_to_openai_ignores_builtin_tools() {
    let input = br#"{
        "model":"gpt-5",
        "input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
        "tools":[{"type":"web_search","search_context_size":"low"}]
    }"#;
    let registry = Registry::new();
    register_all(&registry);
    let output = registry.translate_request(
        &TranslationContext::default(),
        &openai_response(),
        &openai(),
        "gpt-5",
        input,
        false,
    );
    let output: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        output["tools"].as_array().map_or(0, Vec::len),
        0,
        "body={output}"
    );
}

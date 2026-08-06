// ref: internal/translator/antigravity/claude/antigravity_claude_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Candidate delta evidence: propertyNames regression added by a88197f845c979132c8978ea223c6af05cc81536.
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};
use serde_json::Value;
use std::time::Duration;

use crate::internal::cache::signature_cache::{
    cache_signature, clear_signature_cache, signature_cache_test_guard, SignatureCacheStoreError,
    SignatureKvStore,
};

use super::{
    convert_claude_request_to_antigravity, convert_claude_request_to_antigravity_with_capabilities,
    convert_claude_request_to_antigravity_with_runtime, AntigravityClaudeRequestCapabilities,
    AntigravityClaudeRequestTranslationError,
};

fn convert(model: &str, input: &[u8]) -> Value {
    let _guard = signature_cache_test_guard();
    serde_json::from_slice(&convert_claude_request_to_antigravity(model, input, false)).unwrap()
}

fn convert_with_search(model: &str, input: &[u8], enabled: bool) -> Value {
    let _guard = signature_cache_test_guard();
    serde_json::from_slice(&convert_claude_request_to_antigravity_with_capabilities(
        model,
        input,
        false,
        AntigravityClaudeRequestCapabilities {
            native_google_search: enabled,
        },
    ))
    .unwrap()
}

#[test]
fn maps_system_roles_and_safety_without_attribution() {
    let output = convert(
        "claude-sonnet",
        br#"{"system":[{"type":"text","text":"rules"},{"type":"text","text":" x-anthropic-billing-header: hidden"}],"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":[{"type":"text","text":"answer"}]}]}"#,
    );
    assert_eq!(
        output["request"]["systemInstruction"]["parts"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(output["request"]["contents"][0]["role"], "user");
    assert_eq!(output["request"]["contents"][1]["role"], "model");
    assert_eq!(
        output["request"]["safetySettings"]
            .as_array()
            .unwrap()
            .len(),
        5
    );
}

#[test]
fn cleans_disambiguates_and_selects_tools() {
    let output = convert(
        "claude-thinking-test",
        br#"{"thinking":{"type":"enabled","budget_tokens":512},"tools":[{"name":"read file","input_schema":{"type":"object","properties":{}}},{"name":"read/file","input_schema":{"type":"object","properties":{"path":{"type":"string","format":"uri"}}}}],"tool_choice":{"type":"tool","name":"read file"},"messages":[{"role":"user","content":"go"}]}"#,
    );
    let declarations = output["request"]["tools"][0]["functionDeclarations"]
        .as_array()
        .unwrap();
    assert_eq!(declarations.len(), 2);
    assert_ne!(declarations[0]["name"], declarations[1]["name"]);
    assert!(declarations[0]["parametersJsonSchema"]
        .get("required")
        .is_some());
    assert_eq!(
        output["request"]["toolConfig"]["functionCallingConfig"]["mode"],
        "ANY"
    );
    assert!(output["request"]["systemInstruction"]["parts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|part| part["text"]
            .as_str()
            .is_some_and(|text| text.starts_with("Interleaved thinking"))));
}

#[test]
fn strips_property_names_from_claude_tool_schemas() {
    let output = convert(
        "claude-sonnet-4-5",
        br#"{
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "notion-create-pages",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "records": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                    "propertyNames": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                {
                    "name": "notion-update-page",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "properties": {
                                "type": "object",
                                "propertyNames": {"type": "string"}
                            }
                        }
                    }
                }
            ]
        }"#,
    );

    let declarations = output["request"]["tools"][0]["functionDeclarations"]
        .as_array()
        .expect("function declarations");
    assert_eq!(declarations.len(), 2);
    let encoded = serde_json::to_string(declarations).unwrap();
    assert!(
        !encoded.contains("\"propertyNames\""),
        "propertyNames survived translation: {encoded}"
    );
    assert!(declarations[0]
        .pointer("/parametersJsonSchema/properties/records/items/properties/name")
        .is_some());
    assert!(declarations[1]
        .pointer("/parametersJsonSchema/properties/properties")
        .is_some());
}

#[test]
fn tool_result_keeps_images_inside_function_response() {
    let output = convert(
        "gemini-3-pro",
        br#"{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"paint-1","name":"paint","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"paint-1","content":[{"type":"text","text":"ok"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}]}"#,
    );
    let response = &output["request"]["contents"][1]["parts"][0]["functionResponse"];
    assert_eq!(response["name"], "paint");
    assert_eq!(response["response"]["result"]["text"], "ok");
    assert_eq!(response["parts"][0]["inlineData"]["mimeType"], "image/png");
}

#[test]
fn directional_gemini_carrier_binds_to_following_tool() {
    let output = convert(
        "gemini-3-pro",
        br#"{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"cpa-gemini-carrier-v1:next:function:RWdnS0JnRU1PZGJITkE9PQ"},{"type":"tool_use","id":"tool-1","name":"run","input":{}}]}]}"#,
    );
    let part = &output["request"]["contents"][0]["parts"][0];
    assert_eq!(part["functionCall"]["name"], "run");
    assert_eq!(part["thoughtSignature"], "EggKBgEMOdbHNA==");
}

#[test]
fn model_parts_are_ordered_thinking_text_then_function() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    let native = "EhkKFwgMEAIyEWNsYXVkZS1zb25uZXQtNC02GAE=";
    let cached = general_purpose::STANDARD.encode(native.as_bytes());
    assert!(cache_signature("claude-thinking-test", "why", &cached));
    let output: Value = serde_json::from_slice(&convert_claude_request_to_antigravity(
        "claude-thinking-test",
        br#"{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"run","input":{}},{"type":"text","text":"visible"},{"type":"thinking","thinking":"why","signature":"EhkKFwgMEAIyEWNsYXVkZS1zb25uZXQtNC02GAE="}]}]}"#,
        false,
    ))
    .unwrap();
    let parts = output["request"]["contents"][0]["parts"]
        .as_array()
        .unwrap();
    assert_eq!(parts[0]["thought"], true);
    assert_eq!(parts[1]["text"], "visible");
    assert!(parts[2].get("functionCall").is_some());
    clear_signature_cache("");
}

#[test]
fn adaptive_thinking_and_numeric_generation_controls_are_mapped() {
    let output = convert(
        "claude-sonnet",
        br#"{"thinking":{"type":"adaptive"},"output_config":{"effort":" MEDIUM "},"temperature":0.2,"top_p":0.8,"top_k":12,"max_tokens":99,"messages":[]}"#,
    );
    let generation = &output["request"]["generationConfig"];
    assert_eq!(generation["thinkingConfig"]["thinkingLevel"], "medium");
    assert_eq!(generation["temperature"], 0.2);
    assert_eq!(generation["topP"], 0.8);
    assert_eq!(generation["topK"], 12);
    assert_eq!(generation["maxOutputTokens"], 99);
}

#[test]
fn native_web_search_uses_last_user_query_domains_and_max_uses() {
    let output = convert_with_search(
        "gemini-3.1-flash-lite",
        br#"{"messages":[{"role":"user","content":"old"},{"role":"assistant","content":"ignore"},{"role":"user","content":[{"type":"text","text":" Berlin "},{"type":"text","text":" weather "}]}],"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":8,"allowed_domains":[" example.com ",7,"weather.test"]}]}"#,
        true,
    );
    assert_eq!(output["requestType"], "web_search");
    assert_eq!(
        output["request"]["contents"][0]["parts"][0]["text"],
        "Berlin\nweather"
    );
    assert_eq!(
        output["request"]["tools"][0]["googleSearch"]["enhancedContent"]["imageSearch"]
            ["maxResultCount"],
        8
    );
    assert_eq!(
        output["request"]["tools"][0]["googleSearch"]["includedDomains"],
        serde_json::json!(["example.com", "weather.test"])
    );
    assert_eq!(output["request"]["generationConfig"]["candidateCount"], 1);
}

#[test]
fn native_web_search_defaults_to_five_results_and_accepts_forced_choice() {
    let output = convert_with_search(
        "gemini-3.1-flash-lite",
        br#"{"messages":[{"role":"user","content":"query"}],"tools":[{"type":"web_search_20260209","name":"web_search"}],"tool_choice":{"type":"tool","name":"web_search"}}"#,
        true,
    );
    assert_eq!(output["requestType"], "web_search");
    assert_eq!(
        output["request"]["tools"][0]["googleSearch"]["enhancedContent"]["imageSearch"]
            ["maxResultCount"],
        5
    );
}

#[test]
fn web_search_requires_capability_exclusive_tools_and_allowed_choice() {
    let typed = br#"{"messages":[{"role":"user","content":"query"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}"#;
    assert!(convert_with_search("gemini-3.1-flash-lite", typed, false)
        .get("requestType")
        .is_none());
    let mixed = br#"{"messages":[{"role":"user","content":"query"}],"tools":[{"type":"web_search_20250305","name":"web_search"},{"name":"local","input_schema":{"type":"object"}}]}"#;
    assert!(convert_with_search("gemini-3.1-flash-lite", mixed, true)
        .get("requestType")
        .is_none());
    let disabled = br#"{"messages":[{"role":"user","content":"query"}],"tools":[{"type":"web_search_20250305","name":"web_search"}],"tool_choice":"none"}"#;
    assert!(convert_with_search("gemini-3.1-flash-lite", disabled, true)
        .get("requestType")
        .is_none());
}

struct RequestSignatureStore {
    value: Result<Option<Vec<u8>>, SignatureCacheStoreError>,
}

impl SignatureKvStore for RequestSignatureStore {
    fn get(&self, _key: &str) -> Result<Option<Vec<u8>>, SignatureCacheStoreError> {
        self.value.clone()
    }

    fn set(
        &self,
        _key: &str,
        _value: &[u8],
        _ttl: Duration,
    ) -> Result<bool, SignatureCacheStoreError> {
        unreachable!("request conversion never writes signatures")
    }

    fn delete(&self, _key: &str) -> Result<bool, SignatureCacheStoreError> {
        unreachable!("request conversion never deletes signatures")
    }

    fn expire(&self, _key: &str, _ttl: Duration) -> Result<bool, SignatureCacheStoreError> {
        Ok(true)
    }
}

#[test]
fn fallible_runtime_converter_uses_durable_signature_directly() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    let native = "EhkKFwgMEAIyEWNsYXVkZS1zb25uZXQtNC02GAE=";
    let durable = general_purpose::STANDARD.encode(native.as_bytes());
    let store = RequestSignatureStore {
        value: Ok(Some(durable.clone().into_bytes())),
    };
    let output = convert_claude_request_to_antigravity_with_runtime(
        "claude-thinking-test",
        br#"{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"why","signature":"untrusted-client-value"}]}]}"#,
        false,
        AntigravityClaudeRequestCapabilities::default(),
        Some(&store),
    )
    .unwrap();
    let output: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        output["request"]["contents"][0]["parts"][0]["thoughtSignature"],
        durable
    );
}

#[test]
fn fallible_runtime_converter_propagates_store_failure_before_dispatch() {
    let store = RequestSignatureStore {
        value: Err(SignatureCacheStoreError::Unavailable),
    };
    assert_eq!(
        convert_claude_request_to_antigravity_with_runtime(
            "claude-thinking-test",
            br#"{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"why","signature":"client"}]}]}"#,
            false,
            AntigravityClaudeRequestCapabilities::default(),
            Some(&store),
        ),
        Err(AntigravityClaudeRequestTranslationError::SignatureCache(
            SignatureCacheStoreError::Unavailable
        ))
    );
}

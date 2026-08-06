// ref: internal/translator/antigravity/openai/responses/antigravity_openai-responses_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};
use serde_json::{json, Value};

use super::convert_openai_responses_request_to_antigravity;
use crate::internal::signature::compatible_antigravity_claude_thinking_signature;

fn claude_signature(model: &str) -> String {
    let mut channel = vec![0x08, 0x0c, 0x10, 0x02, 0x32, model.len() as u8];
    channel.extend_from_slice(model.as_bytes());
    let mut container = vec![0x0a, channel.len() as u8];
    container.extend_from_slice(&channel);
    let mut payload = vec![0x12, container.len() as u8];
    payload.extend_from_slice(&container);
    payload.extend_from_slice(&[0x18, 0x01]);
    general_purpose::STANDARD.encode(payload)
}

fn convert(model: &str, input: Value) -> Value {
    serde_json::from_slice(&convert_openai_responses_request_to_antigravity(
        model,
        &serde_json::to_vec(&input).unwrap(),
        false,
    ))
    .unwrap()
}

fn thought_parts(root: &Value) -> Vec<&Value> {
    root.pointer("/request/contents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|content| {
            content
                .get("parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter(|part| part.get("thought").and_then(Value::as_bool) == Some(true))
        .collect()
}

#[test]
fn claude_reasoning_keeps_compatible_native_and_double_layer_signatures() {
    let native = claude_signature("claude-sonnet-4-6");
    let adapted = compatible_antigravity_claude_thinking_signature(&native).unwrap();
    for encrypted in [&native, &adapted] {
        let output = convert(
            "claude-opus-4-6-thinking",
            json!({"input":[
                {"type":"reasoning","encrypted_content":encrypted,"summary":[{"type":"summary_text","text":"internal reasoning"}]},
                {"role":"assistant","content":[{"type":"output_text","text":"visible answer"}]},
                {"role":"user","content":[{"type":"input_text","text":"continue"}]}
            ]}),
        );
        let thoughts = thought_parts(&output);
        assert_eq!(thoughts.len(), 1);
        assert_eq!(thoughts[0]["thoughtSignature"], adapted);
        assert_eq!(thoughts[0]["text"], "internal reasoning");
    }
}

#[test]
fn claude_drops_incompatible_or_empty_reasoning_without_hiding_visible_text() {
    let native = claude_signature("claude-sonnet-4-6");
    let output = convert(
        "claude-opus-4-6-thinking",
        json!({"input":[
            {"type":"reasoning","encrypted_content":"gpt#invalid","summary":[{"type":"summary_text","text":"must not reach Claude"}]},
            {"type":"reasoning","encrypted_content":native,"summary":[]},
            {"role":"assistant","content":[{"type":"output_text","text":"visible answer"}]}
        ]}),
    );
    assert!(thought_parts(&output).is_empty());
    let rendered = serde_json::to_string(&output).unwrap();
    assert!(!rendered.contains("must not reach Claude"));
    assert!(rendered.contains("visible answer"));
}

#[test]
fn empty_reasoning_does_not_shift_a_later_signature_across_message_or_function_boundaries() {
    let first = claude_signature("claude-sonnet-4-6");
    let second = claude_signature("claude-opus-4-6");
    let expected = compatible_antigravity_claude_thinking_signature(&second).unwrap();
    for boundary in [
        vec![json!({"role":"user","content":[{"type":"input_text","text":"boundary"}]})],
        vec![
            json!({"type":"function_call","call_id":"call-1","name":"run","arguments":"{}"}),
            json!({"type":"function_call_output","call_id":"call-1","output":"ok"}),
        ],
    ] {
        let mut input = vec![json!({"type":"reasoning","encrypted_content":first,"summary":[]})];
        input.extend(boundary);
        input.push(json!({"type":"reasoning","encrypted_content":second,"summary":[{"type":"summary_text","text":"second reasoning"}]}));
        input.push(json!({"role":"user","content":[{"type":"input_text","text":"continue"}]}));
        let output = convert("claude-opus-4-6-thinking", json!({"input":input}));
        let thoughts = thought_parts(&output);
        assert_eq!(thoughts.len(), 1);
        assert_eq!(thoughts[0]["text"], "second reasoning");
        assert_eq!(thoughts[0]["thoughtSignature"], expected);
    }
}

#[test]
fn gemini_reasoning_uses_native_thought_signature_placement() {
    let signature = "EjQKMgEMOdbHO0Gd+c9Mxk4ELwPGbpCEcp2mFfYYLix2UVtBH3fL8GECc4+JITVnHF4qZDsA";
    let output = convert(
        "gemini-3-flash-agent",
        json!({"input":[{
            "type":"reasoning","encrypted_content":format!("gemini#{signature}"),
            "summary":[{"type":"summary_text","text":"reasoning summary"}]
        }]}),
    );
    let thoughts = thought_parts(&output);
    assert_eq!(thoughts.len(), 1);
    assert_eq!(thoughts[0]["thoughtSignature"], signature);
}

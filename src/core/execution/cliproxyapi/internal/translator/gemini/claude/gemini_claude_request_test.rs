// ref: internal/translator/gemini/claude/gemini_claude_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_claude_request_to_gemini;

#[test]
fn maps_system_media_specific_tool_and_schema() {
    let output: Value = serde_json::from_slice(&convert_claude_request_to_gemini(
        "gemini-3-flash",
        br#"{"system":"Be concise","messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]}],"tools":[{"name":"json","description":"A JSON tool","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"tool","name":"json"}}"#,
        false,
    ))
    .unwrap();
    assert_eq!(output["model"], "gemini-3-flash");
    assert_eq!(
        output["systemInstruction"]["parts"][0]["text"],
        "Be concise"
    );
    assert_eq!(
        output["contents"][0]["parts"][1]["inline_data"]["mime_type"],
        "image/png"
    );
    assert_eq!(output["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
    assert_eq!(
        output["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
        "json"
    );
}

#[test]
fn maps_structured_tool_result_without_forwarding_empty_parts() {
    let output: Value = serde_json::from_slice(&convert_claude_request_to_gemini(
        "gemini-3-flash",
        br#"{"messages":[{"role":"assistant","content":[{"type":"text","text":""},{"type":"tool_use","id":"json-call-1","name":"json","input":{"ok":true}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"json-call-1","content":[{"type":"text","text":"alpha"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]}]}]}"#,
        false,
    ))
    .unwrap();
    assert!(output["contents"][0]["parts"][0]["functionCall"].is_object());
    assert_eq!(
        output["contents"][1]["parts"][0]["functionResponse"]["name"],
        "json"
    );
    assert_eq!(
        output["contents"][1]["parts"][1]["inline_data"]["data"],
        "aGVsbG8="
    );
}

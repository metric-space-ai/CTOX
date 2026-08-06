// ref: internal/translator/gemini/openai/chat-completions/gemini_openai_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_openai_chat_request_to_gemini;

fn translate(input: &str) -> Value {
    serde_json::from_slice(&convert_openai_chat_request_to_gemini(
        "gemini-test",
        input.as_bytes(),
        false,
    ))
    .expect("translated Gemini request")
}

#[test]
fn strips_prefill_skips_empty_parts_and_preserves_reasoning_order() {
    let output = translate(
        r#"{"messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":[{"type":"text","text":""}],"reasoning_content":"thinking","tool_calls":[{"id":"call-1","type":"function","function":{"name":"read_file","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call-1","content":"ok"},
            {"role":"user","content":"again"},
            {"role":"assistant","content":"trailing prefill"}
        ]}"#,
    );
    assert_eq!(output["contents"].as_array().unwrap().len(), 4);
    assert_eq!(output["contents"][1]["parts"][0]["text"], "thinking");
    assert_eq!(output["contents"][1]["parts"][0]["thought"], true);
    assert_eq!(
        output["contents"][1]["parts"][1]["functionCall"]["name"],
        "read_file"
    );
    assert_eq!(output["contents"][3]["parts"][0]["text"], "again");
}

#[test]
fn preserves_audio_video_and_omits_empty_text_parts() {
    let output = translate(
        r#"{"messages":[{"role":"user","content":[
            {"type":"text","text":""},
            {"type":"input_audio","input_audio":{"data":"SUQzBA==","format":"mp3"}},
            {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}
        ]}]}"#,
    );
    let parts = output["contents"][0]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["inlineData"]["mime_type"], "audio/mpeg");
    assert_eq!(parts[0]["inlineData"]["data"], "SUQzBA==");
    assert_eq!(parts[1]["inlineData"]["mime_type"], "video/mp4");
    assert_eq!(parts[1]["inlineData"]["data"], "AAAA");
}

#[test]
fn maps_token_precedence_schema_cleanup_and_response_formats() {
    let output = translate(
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":30,"max_completion_tokens":40,
        "generationConfig":{"temperature":0.2,"responseSchema":{"type":"string"}},
        "tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object","title":"Root","properties":{"country":{"type":"string"}},"required":["country","stale"]}}}],
        "response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","additionalProperties":false}}}}"#,
    );
    assert_eq!(output["generationConfig"]["maxOutputTokens"], 30);
    assert_eq!(
        output["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert!(output["generationConfig"].get("responseSchema").is_none());
    assert_eq!(
        output["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["required"],
        serde_json::json!(["country"])
    );

    let json_object = translate(
        r#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#,
    );
    assert_eq!(
        json_object["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert!(json_object["generationConfig"]
        .get("responseJsonSchema")
        .is_none());
}

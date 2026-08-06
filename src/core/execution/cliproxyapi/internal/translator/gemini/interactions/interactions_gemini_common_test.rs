// ref: internal/translator/gemini/interactions/interactions_gemini_common_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_gemini_request_to_interactions, convert_gemini_response_to_interactions_non_stream,
    convert_gemini_response_to_interactions_stream, convert_interactions_request_to_gemini,
    convert_interactions_response_to_gemini_non_stream, GeminiToInteractionsState,
};

#[test]
fn interactions_to_gemini_maps_generation_thought_tools_and_ids() {
    let output: Value = serde_json::from_slice(&convert_interactions_request_to_gemini(
        "gemini-3.5-flash",
        br#"{"model":"gemini-3.5-flash","generation_config":{"max_output_tokens":32,"thinking_level":"high","thinking_summaries":"auto","tool_choice":"auto"},"input":[{"type":"thought","content":[{"type":"text","text":"thinking"}]},{"type":"function_call","name":"lookup","call_id":"call_1","arguments":{"q":"x"}}]}"#,
        true,
    ))
    .unwrap();
    assert_eq!(output["generationConfig"]["maxOutputTokens"], 32);
    assert_eq!(
        output["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        "high"
    );
    assert_eq!(
        output["toolConfig"]["functionCallingConfig"]["mode"],
        "AUTO"
    );
    assert_eq!(output["contents"][0]["parts"][0]["thought"], true);
    assert_eq!(
        output["contents"][1]["parts"][0]["functionCall"]["id"],
        "call_1"
    );
}

#[test]
fn gemini_to_interactions_maps_system_config_multimodal_and_function_pairs() {
    let output: Value = serde_json::from_slice(&convert_gemini_request_to_interactions(
        "gemini-3.5-flash",
        br#"{"systemInstruction":{"parts":[{"text":"one"},{"text":"two"}]},"generationConfig":{"maxOutputTokens":32,"thinkingConfig":{"thinkingBudget":1024}},"contents":[{"role":"model","parts":[{"functionCall":{"name":"lookup","id":"call_1","args":{"q":"x"}}}]},{"role":"user","parts":[{"functionResponse":{"name":"lookup","id":"call_1","response":{"ok":true}}},{"inlineData":{"mimeType":"audio/wav","data":"aGVsbG8="}}]}]}"#,
        false,
    ))
    .unwrap();
    assert_eq!(output["system_instruction"], "one\ntwo");
    assert_eq!(output["generation_config"]["max_output_tokens"], 32);
    assert_eq!(output["input"][0]["type"], "function_call");
    assert_eq!(output["input"][0]["call_id"], "call_1");
    assert_eq!(output["input"][1]["type"], "function_result");
    assert_eq!(output["input"][2]["content"][0]["type"], "audio");
}

#[test]
fn gemini_response_maps_nonstream_and_request_local_stream_identity() {
    let raw = br#"{"responseId":"resp_1","usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3},"candidates":[{"finishReason":"STOP","content":{"parts":[{"thought":true,"text":"why"},{"text":"answer"},{"functionCall":{"name":"lookup","id":"call_1","args":{}}}]}}]}"#;
    let output: Value = serde_json::from_slice(
        &convert_gemini_response_to_interactions_non_stream("gemini-3.5-flash", b"{}", b"{}", raw),
    )
    .unwrap();
    assert_eq!(output["id"], "resp_1");
    assert_eq!(output["steps"][0]["type"], "thought");
    assert_eq!(output["steps"][2]["type"], "function_call");
    assert_eq!(output["usage"]["total_tokens"], 3);

    let events = convert_gemini_response_to_interactions_stream(
        "gemini-3.5-flash",
        b"{}",
        b"{}",
        raw,
        &mut GeminiToInteractionsState::with_identity("interaction_fixed", "2026-08-04T00:00:00Z"),
    );
    let joined = String::from_utf8(events.concat()).unwrap();
    assert!(joined.contains("interaction_fixed"));
    assert!(joined.contains("interaction.completed"));
}

#[test]
fn interactions_aggregate_response_maps_back_to_gemini() {
    let output: Value = serde_json::from_slice(
        &convert_interactions_response_to_gemini_non_stream(
            "gemini-3.5-flash",
            br#"{"id":"i1","model":"gemini-3.5-flash","steps":[{"type":"model_output","content":[{"type":"text","text":"answer"}]},{"type":"function_call","name":"lookup","call_id":"c1","arguments":{"q":"x"}}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}"#,
        ),
    )
    .unwrap();
    assert_eq!(output["responseId"], "i1");
    assert_eq!(
        output["candidates"][0]["content"]["parts"][0]["text"],
        "answer"
    );
    assert_eq!(
        output["candidates"][0]["content"]["parts"][1]["functionCall"]["id"],
        "c1"
    );
    assert_eq!(output["usageMetadata"]["totalTokenCount"], 3);
}

// ref: internal/runtime/executor/openai_compat_executor_tool_results_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::openai_compat_executor::{
    normalize_openai_tool_results_text_only, should_normalize_openai_tool_results_for_model,
    OpenAiCompatibility, OpenAiCompatibilityModel,
};

#[test]
fn tool_result_content_follows_input_modalities() {
    let payload = br#"{
        "model":"mapped-model",
        "messages":[{
            "role":"tool",
            "tool_call_id":"call_1",
            "content":[
                {"type":"text","text":"image inspected"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}
            ]
        }]
    }"#;
    for (_stream, modalities, want_string) in [
        (false, vec!["text"], true),
        (true, vec!["text"], true),
        (false, vec!["text", "image"], false),
        (false, vec![], false),
    ] {
        let compat = OpenAiCompatibility {
            name: "compat".into(),
            models: vec![OpenAiCompatibilityModel {
                name: "mapped-model".into(),
                alias: "claude-client".into(),
                input_modalities: modalities.into_iter().map(str::to_owned).collect(),
            }],
            ..OpenAiCompatibility::default()
        };
        let should_normalize = should_normalize_openai_tool_results_for_model(
            &compat,
            "mapped-model",
            "claude-client",
        );
        let output = if should_normalize {
            normalize_openai_tool_results_text_only(payload)
        } else {
            payload.to_vec()
        };
        let value: Value = serde_json::from_slice(&output).unwrap();
        let content = &value["messages"][0]["content"];
        if want_string {
            assert_eq!(
                content,
                "image inspected\n\n[image omitted: unsupported by upstream]"
            );
        } else {
            assert!(content.is_array());
        }
    }
}

#[test]
fn alias_resolution_requires_every_matching_alias_to_exclude_images() {
    let compat = OpenAiCompatibility {
        models: vec![
            OpenAiCompatibilityModel {
                name: "first".into(),
                alias: "shared".into(),
                input_modalities: vec!["text".into()],
            },
            OpenAiCompatibilityModel {
                name: "second".into(),
                alias: "shared".into(),
                input_modalities: vec!["text".into(), "image".into()],
            },
        ],
        ..OpenAiCompatibility::default()
    };
    assert!(!should_normalize_openai_tool_results_for_model(
        &compat, "unknown", "shared"
    ));
}

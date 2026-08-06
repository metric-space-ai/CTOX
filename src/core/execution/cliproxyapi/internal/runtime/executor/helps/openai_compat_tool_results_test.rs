// ref: internal/runtime/executor/helps/openai_compat_tool_results_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::super::openai_compat_executor::{OpenAiCompatibility, OpenAiCompatibilityModel};
use super::openai_compat_tool_results::{
    normalize_openai_tool_results_text_only, should_normalize_openai_tool_results_for_model,
    OPENAI_TOOL_RESULT_IMAGE_OMITTED_TEXT,
};

fn model(name: &str, alias: &str, modalities: &[&str]) -> OpenAiCompatibilityModel {
    OpenAiCompatibilityModel {
        name: name.to_owned(),
        alias: alias.to_owned(),
        input_modalities: modalities.iter().map(|value| (*value).to_owned()).collect(),
    }
}

#[test]
fn normalizes_only_tool_content_and_preserves_existing_strings() {
    let input = br#"{"messages":[
        {"role":"assistant","content":[{"type":"text","text":"before"}]},
        {"role":"tool","content":[{"type":"text","text":"image inspected"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]},
        {"role":"tool","content":"already text"},
        {"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/user.png"}}]}
    ]}"#;
    let output = normalize_openai_tool_results_text_only(input);
    let value: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        value["messages"][1]["content"],
        format!("image inspected\n\n{OPENAI_TOOL_RESULT_IMAGE_OMITTED_TEXT}")
    );
    assert_eq!(value["messages"][2]["content"], "already text");
    assert!(value["messages"][0]["content"].is_array());
    assert!(value["messages"][3]["content"].is_array());
}

#[test]
fn image_and_unknown_content_match_upstream_flattening() {
    for (content, expected) in [
        (
            r#"[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]"#,
            OPENAI_TOOL_RESULT_IMAGE_OMITTED_TEXT,
        ),
        (
            r#"{"type":"image","source":{"type":"base64","data":"AA=="}}"#,
            OPENAI_TOOL_RESULT_IMAGE_OMITTED_TEXT,
        ),
        (
            r#"[{"type":"custom","value":1}]"#,
            r#"{"type":"custom","value":1}"#,
        ),
    ] {
        let input = format!(r#"{{"messages":[{{"role":"tool","content":{content}}}]}}"#);
        let output = normalize_openai_tool_results_text_only(input.as_bytes());
        let value: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["messages"][0]["content"], expected);
    }
}

#[test]
fn model_name_precedes_alias_and_suffixes_are_ignored() {
    let compat = OpenAiCompatibility {
        models: vec![
            model("upstream-text", "alias-text", &["text"]),
            model(
                "upstream-multimodal",
                "alias-multimodal",
                &["text", "image"],
            ),
            model("upstream-unspecified", "alias-unspecified", &[]),
            model("upstream-uppercase", "alias-uppercase", &["TEXT"]),
            model("pool-text", "shared-alias", &["text"]),
            model("pool-image", "shared-alias", &["text", "image"]),
        ],
        ..OpenAiCompatibility::default()
    };
    for (upstream, requested, expected) in [
        ("upstream-text", "", true),
        ("upstream-text(high)", "", true),
        ("unknown", "alias-text", true),
        ("upstream-multimodal", "alias-text", false),
        ("upstream-unspecified", "alias-text", false),
        ("upstream-uppercase", "", true),
        ("unknown", "shared-alias", false),
        ("unknown", "missing", false),
    ] {
        assert_eq!(
            should_normalize_openai_tool_results_for_model(Some(&compat), upstream, requested),
            expected,
            "{upstream}/{requested}"
        );
    }
}

#[test]
fn none_config_and_noop_payloads_are_byte_identical() {
    assert!(!should_normalize_openai_tool_results_for_model(
        None,
        "upstream-text",
        "alias-text"
    ));
    for payload in [
        b"not-json".as_slice(),
        br#" { "messages": {} } "#,
        br#" { "messages": [{"role":"tool","content":"text"}] } "#,
    ] {
        assert_eq!(normalize_openai_tool_results_text_only(payload), payload);
    }
}

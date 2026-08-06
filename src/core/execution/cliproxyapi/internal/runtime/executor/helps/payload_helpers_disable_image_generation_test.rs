// ref: internal/runtime/executor/helps/payload_helpers_disable_image_generation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use super::payload_helpers::{
    apply_disable_image_generation_with_root, apply_payload_config_with_request,
    apply_payload_config_with_root, match_model_pattern, payload_request_path,
    payload_requested_model, PayloadApplyConfig, PayloadFilterRule, PayloadHeaders,
    PayloadModelRule, PayloadRule,
};
use crate::internal::config::DisableImageGenerationMode;
use crate::sdk::cliproxy::executor::Options;

fn apply(payload: &[u8], root: &str, mode: DisableImageGenerationMode, path: &str) -> Vec<u8> {
    apply_disable_image_generation_with_root(payload, root, mode, path)
}

#[test]
fn all_removes_image_tool_and_keeps_other_tool_at_root_or_nested_root() {
    for (root, payload, remaining) in [
        (
            "",
            br#"{"tools":[{"type":"image_generation","output_format":"png"},{"type":"function","name":"f1"}]}"#.as_slice(),
            "function",
        ),
        (
            "request",
            br#"{"request":{"tools":[{"type":"image_generation"},{"type":"web_search"}]}}"#.as_slice(),
            "web_search",
        ),
    ] {
        let output = apply(payload, root, DisableImageGenerationMode::All, "");
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        let tools = if root.is_empty() {
            &value["tools"]
        } else {
            &value["request"]["tools"]
        };
        assert_eq!(tools.as_array().unwrap().len(), 1);
        assert_eq!(tools[0]["type"], remaining);
    }
}

#[test]
fn all_removes_tool_choice_by_type_or_tool_name() {
    for (root, payload) in [
        (
            "",
            br#"{"tools":[],"tool_choice":{"type":"image_generation"}}"#.as_slice(),
        ),
        (
            "request",
            br#"{"request":{"tools":[],"tool_choice":{"type":"tool","name":"image_generation"}}}"#
                .as_slice(),
        ),
    ] {
        let output = apply(payload, root, DisableImageGenerationMode::All, "");
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        let object = if root.is_empty() {
            value.as_object().unwrap()
        } else {
            value["request"].as_object().unwrap()
        };
        assert!(!object.contains_key("tool_choice"));
    }
}

#[test]
fn chat_keeps_images_endpoints_and_strips_non_images_endpoints() {
    let payload = br#"{"tools":[{"type":"image_generation"},{"type":"function"}],"tool_choice":"image_generation"}"#;
    let kept = apply(
        payload,
        "",
        DisableImageGenerationMode::Chat,
        "/v1/images/generations",
    );
    assert_eq!(kept, payload);
    let stripped = apply(
        payload,
        "",
        DisableImageGenerationMode::Chat,
        "/v1/responses",
    );
    let value: serde_json::Value = serde_json::from_slice(&stripped).unwrap();
    assert_eq!(value["tools"].as_array().unwrap().len(), 1);
    assert!(value.get("tool_choice").is_none());
}

#[test]
fn passthrough_and_off_preserve_exact_bytes_on_every_endpoint() {
    let payload = b" { \"tools\" : [ {\"type\":\"image_generation\"} ] } \n";
    for mode in [
        DisableImageGenerationMode::Off,
        DisableImageGenerationMode::Passthrough,
    ] {
        for path in ["", "/v1/responses", "/prefix/v1/images/edits"] {
            assert_eq!(apply(payload, "", mode, path), payload);
        }
    }
}

fn model(name: &str, protocol: &str) -> PayloadModelRule {
    PayloadModelRule {
        name: name.into(),
        protocol: protocol.into(),
        ..PayloadModelRule::default()
    }
}

#[test]
fn raw_override_can_restore_image_generation_after_pre_filter() {
    let mut config = PayloadApplyConfig {
        disable_image_generation: DisableImageGenerationMode::All,
        ..PayloadApplyConfig::default()
    };
    config.rules.override_raw.push(PayloadRule {
        models: vec![model("gpt-5.4", "openai-response")],
        params: BTreeMap::from([
            (
                "tools".into(),
                serde_json::Value::String(
                    r#"[{"type":"image_generation"},{"type":"function","name":"f1"}]"#.into(),
                ),
            ),
            (
                "tool_choice".into(),
                serde_json::Value::String(r#"{"type":"image_generation"}"#.into()),
            ),
        ]),
    });
    let output = apply_payload_config_with_root(
        &config,
        "gpt-5.4",
        "openai-response",
        "",
        br#"{"tools":[{"type":"image_generation"}]}"#,
        None,
        "",
        "",
    );
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(value["tools"].as_array().unwrap().len(), 2);
    assert_eq!(value["tool_choice"]["type"], "image_generation");
}

#[test]
fn header_from_protocol_and_payload_conditions_narrow_rules() {
    let mut matching = model("gpt-*", "openai");
    matching.from_protocol = "responses".into();
    matching
        .headers
        .insert("X-Client-Tier".into(), "tenant-*-region-*".into());
    matching.matches.push(BTreeMap::from([
        ("metadata.client".into(), serde_json::json!("codex")),
        (
            "tools.#(type==\"web_search\").enabled".into(),
            serde_json::json!(true),
        ),
    ]));
    matching.not_matches.push(BTreeMap::from([(
        "metadata.mode".into(),
        serde_json::json!("dev"),
    )]));
    matching
        .exist
        .push("tools.#(type==\"web_search\").type".into());
    matching.not_exist.push("metadata.missing".into());
    matching.not_exist.push("metadata.null_value".into());
    let mut config = PayloadApplyConfig::default();
    config.rules.override_values.push(PayloadRule {
        models: vec![matching],
        params: BTreeMap::from([("metadata.applied".into(), serde_json::json!(true))]),
    });
    let payload = br#"{"metadata":{"client":"codex","mode":"prod","null_value":null},"tools":[{"type":"web_search","enabled":true}]}"#;
    let headers = PayloadHeaders::from([(
        "x-client-tier".into(),
        vec!["tenant-alpha-region-us".into()],
    )]);
    let output = apply_payload_config_with_request(
        &config,
        "gpt-5.4",
        "openai",
        "openai-response",
        "",
        payload,
        None,
        "",
        "",
        &headers,
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output).unwrap()["metadata"]["applied"],
        true
    );

    let mismatched = apply_payload_config_with_request(
        &config, "gpt-5.4", "openai", "openai", "", payload, None, "", "", &headers,
    );
    assert!(
        serde_json::from_slice::<serde_json::Value>(&mismatched).unwrap()["metadata"]
            .get("applied")
            .is_none()
    );
}

#[test]
fn defaults_are_first_write_filter_queries_delete_backwards() {
    let mut config = PayloadApplyConfig::default();
    config.rules.default.push(PayloadRule {
        models: vec![model("gpt-*", "")],
        params: BTreeMap::from([("metadata.default".into(), serde_json::json!(1))]),
    });
    config.rules.default.push(PayloadRule {
        models: vec![model("gpt-*", "")],
        params: BTreeMap::from([("metadata.default".into(), serde_json::json!(2))]),
    });
    config.rules.filter.push(PayloadFilterRule {
        models: vec![model("gpt-*", "")],
        params: vec!["tools.#(type==\"remove\")#.secret".into()],
    });
    let output = apply_payload_config_with_root(
        &config,
        "gpt-5.4",
        "openai",
        "",
        br#"{"tools":[{"type":"remove","secret":1},{"type":"keep","secret":2},{"type":"remove","secret":3}]}"#,
        None,
        "",
        "",
    );
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(value["metadata"]["default"], 1);
    assert!(value["tools"][0].get("secret").is_none());
    assert_eq!(value["tools"][1]["secret"], 2);
    assert!(value["tools"][2].get("secret").is_none());
}

#[test]
fn typed_options_and_star_glob_match_upstream_metadata_helpers() {
    let mut options = Options::default();
    options.metadata.requested_model = Some("  alias-model  ".into());
    options.metadata.request_path = Some(" /v1/responses ".into());
    assert_eq!(payload_requested_model(&options, "fallback"), "alias-model");
    assert_eq!(payload_request_path(&options), "/v1/responses");
    for (pattern, value, expected) in [
        ("*-5", "gpt-5", true),
        ("gpt-*", "gpt-5", true),
        ("gemini-*-pro", "gemini-2.5-pro", true),
        ("gpt-*", "claude", false),
        ("", "anything", false),
    ] {
        assert_eq!(match_model_pattern(pattern, value), expected);
    }
}

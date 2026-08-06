// ref: internal/modelconfig/model_info_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::model_hash::{
    compute_claude_models_hash, compute_codex_models_hash, compute_gemini_models_hash,
    compute_openai_compat_models_hash, compute_vertex_compat_models_hash, ModelAlias,
    OpenAiCompatibilityModel,
};
use super::model_info::{normalize_thinking_support, resolve_model_info, ThinkingSupport};

#[test]
fn resolve_model_info_uses_suffix_free_static_capabilities() {
    let info = resolve_model_info("claude-opus-4-6(high)", "claude", None);
    assert!(info.thinking.is_some(), "{info:?}");
    assert_eq!(info.id, "claude-opus-4-6(high)");
    assert!(!info.user_defined);
}

#[test]
fn resolve_model_info_explicit_thinking_overrides_and_clones() {
    let mut support = ThinkingSupport {
        levels: vec![" XHIGH ".into(), "xhigh".into(), " High ".into()],
        ..ThinkingSupport::default()
    };
    let info = resolve_model_info("custom-model", "codex", Some(&support));
    assert_eq!(info.thinking.as_ref().unwrap().levels, ["xhigh", "high"]);
    support.levels[0] = "low".into();
    assert_eq!(info.thinking.unwrap().levels[0], "xhigh");
}

#[test]
fn normalize_thinking_support_derives_special_level_flags() {
    let support = normalize_thinking_support(&ThinkingSupport {
        levels: vec!["low".into(), "none".into(), "auto".into()],
        ..ThinkingSupport::default()
    });
    assert!(support.zero_allowed);
    assert!(support.dynamic_allowed);
}

#[test]
fn resolve_model_info_unknown_model_keeps_missing_capability() {
    let info = resolve_model_info("unknown-configured-model", "claude", None);
    assert!(info.thinking.is_none(), "{:?}", info.thinking);
    assert!(!info.user_defined);
}

#[test]
fn hashes_match_pinned_go_fixtures_and_preserve_order() {
    let thinking = ThinkingSupport {
        min: 1,
        max: 8,
        levels: vec!["low".into(), "high".into()],
        ..ThinkingSupport::default()
    };
    let aliases = vec![
        ModelAlias {
            name: " Model-A ".into(),
            alias: " ALIAS ".into(),
            display_name: " Display A ".into(),
            force_mapping: true,
            thinking: Some(thinking.clone()),
        },
        ModelAlias {
            name: "model-b".into(),
            alias: String::new(),
            ..ModelAlias::default()
        },
    ];
    let expected = "6ddcb7dd02032c4d37892ab211928b8f87e997a60358132b2d51900a2739e292";
    assert_eq!(compute_vertex_compat_models_hash(&aliases), expected);
    assert_eq!(compute_claude_models_hash(&aliases), expected);
    assert_eq!(compute_codex_models_hash(&aliases), expected);
    assert_eq!(compute_gemini_models_hash(&aliases), expected);

    let mut reversed = aliases.clone();
    reversed.reverse();
    assert_ne!(compute_claude_models_hash(&reversed), expected);
}

#[test]
fn openai_hash_normalizes_modalities_and_ignores_blank_models() {
    let hash = compute_openai_compat_models_hash(&[
        OpenAiCompatibilityModel {
            name: " GPT-Vision ".into(),
            alias: " Vision ".into(),
            display_name: " Visible ".into(),
            force_mapping: true,
            image: true,
            input_modalities: vec![" TEXT ".into(), "image".into(), "text".into(), "".into()],
            output_modalities: vec![" Text ".into(), "IMAGE".into()],
            thinking: None,
        },
        OpenAiCompatibilityModel::default(),
    ]);
    assert_eq!(
        hash,
        "573c185ba51eb60420472ad9a9c2c7d62fb933ec54f4e0f6bbb555c7dae871d5"
    );
    assert!(compute_openai_compat_models_hash(&[OpenAiCompatibilityModel::default()]).is_empty());
}

#[test]
fn thinking_json_matches_go_omitempty_contract() {
    assert_eq!(
        serde_json::to_string(&ThinkingSupport::default()).unwrap(),
        "{}"
    );
    assert_eq!(
        serde_json::to_string(&ThinkingSupport {
            zero_allowed: true,
            dynamic_allowed: true,
            levels: vec!["none".into(), "auto".into()],
            ..ThinkingSupport::default()
        })
        .unwrap(),
        r#"{"zero_allowed":true,"dynamic_allowed":true,"levels":["none","auto"]}"#
    );
}

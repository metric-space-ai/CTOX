// refs: internal/thinking/strip.go, provider/openai/apply.go,
// provider/xai/apply.go @ ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::registry::{ModelInfo, ThinkingSupport};

use super::*;

const NO_LEVELS: &[&str] = &[];
const LEVELS: &[&str] = &["minimal", "low", "medium", "high"];
const NONE_LEVELS: &[&str] = &["none", "low", "high"];
const XAI_LEVELS: &[&str] = &["none", "low", "medium", "high"];

fn model(user_defined: bool, thinking: Option<ThinkingSupport>) -> ModelInfo {
    ModelInfo {
        id: if user_defined {
            "custom-model"
        } else {
            "level-model"
        },
        provider_type: "openai",
        user_defined,
        max_completion_tokens: 0,
        thinking,
    }
}

fn support(zero_allowed: bool, levels: &'static [&'static str]) -> ThinkingSupport {
    ThinkingSupport {
        min: None,
        max: None,
        zero_allowed,
        dynamic_allowed: false,
        levels,
    }
}

fn config(mode: ThinkingMode, budget: isize, level: &str) -> ThinkingConfig {
    ThinkingConfig {
        mode,
        budget,
        level: ThinkingLevel::new(level),
    }
}

fn path(body: &[u8], path: &str) -> Option<Value> {
    let document: Value = serde_json::from_slice(body).ok()?;
    path.split('.')
        .try_fold(&document, |value, segment| value.get(segment))
        .cloned()
}

#[test]
fn strip_preserves_bytes_for_invalid_unknown_and_semantic_noops() {
    for (body, provider) in [
        (b"".as_slice(), "openai"),
        (b"not json".as_slice(), "openai"),
        (br#"{ "model" : "m" }"#.as_slice(), "unknown"),
        (br#"{ "model" : "m" }"#.as_slice(), "OpenAI"),
        (br#"{ "model" : "m" }"#.as_slice(), "openai"),
    ] {
        assert_eq!(strip_thinking_config(body, provider), body);
    }
}

#[test]
fn strip_removes_exact_provider_paths_and_preserves_other_fields() {
    let cases = [
        (
            "claude",
            br#"{"thinking":{"type":"enabled"},"output_config":{"effort":"high","keep":true},"keep":1}"#.as_slice(),
            &["thinking", "output_config.effort"][..],
        ),
        (
            "gemini",
            br#"{"generationConfig":{"thinkingConfig":{"thinkingBudget":10},"keep":true},"keep":1}"#.as_slice(),
            &["generationConfig.thinkingConfig"][..],
        ),
        (
            "antigravity",
            br#"{"request":{"generationConfig":{"thinkingConfig":{"thinkingBudget":10},"keep":true}},"keep":1}"#.as_slice(),
            &["request.generationConfig.thinkingConfig"][..],
        ),
        (
            "openai",
            br#"{"reasoning_effort":"high","reasoning":{"effort":"low","keep":true},"keep":1}"#.as_slice(),
            &["reasoning_effort", "reasoning"][..],
        ),
        (
            "kimi",
            br#"{"reasoning_effort":"high","thinking":{"type":"enabled"},"keep":1}"#.as_slice(),
            &["reasoning_effort", "thinking"][..],
        ),
        (
            "codex",
            br#"{"reasoning":{"effort":"high"},"reasoning_effort":"keep","keep":1}"#.as_slice(),
            &["reasoning"][..],
        ),
        (
            "xai",
            br#"{"reasoning":{"effort":"high"},"reasoning_effort":"keep","keep":1}"#.as_slice(),
            &["reasoning"][..],
        ),
    ];

    for (provider, body, removed_paths) in cases {
        let output = strip_thinking_config(body, provider);
        for removed in removed_paths {
            assert_eq!(path(&output, removed), None, "provider={provider}");
        }
        assert_eq!(path(&output, "keep"), Some(Value::Number(1.into())));
    }
}

#[test]
fn interactions_strip_removes_all_official_and_compatibility_aliases() {
    let body = br#"{"generation_config":{"thinking_level":"high","thinkingLevel":"low","thinking_budget":1,"thinkingBudget":2,"thinking_summaries":"auto","thinkingSummaries":"none","thinking_config":{"include_thoughts":true},"thinkingConfig":{"includeThoughts":true},"keep":true}}"#;
    let output = strip_thinking_config(body, "interactions");
    for field in [
        "thinking_level",
        "thinkingLevel",
        "thinking_budget",
        "thinkingBudget",
        "thinking_summaries",
        "thinkingSummaries",
        "thinking_config",
        "thinkingConfig",
    ] {
        assert_eq!(path(&output, &format!("generation_config.{field}")), None);
    }
    assert_eq!(
        path(&output, "generation_config.keep"),
        Some(Value::Bool(true))
    );
}

#[test]
fn claude_strip_drops_output_config_only_when_it_is_empty() {
    let only_effort = br#"{"output_config":{"effort":"high"},"keep":true}"#;
    let output = strip_thinking_config(only_effort, "claude");
    assert_eq!(path(&output, "output_config"), None);

    let already_empty = br#"{ "output_config" : {}, "keep" : true }"#;
    let output = strip_thinking_config(already_empty, "claude");
    assert_eq!(path(&output, "output_config"), None);

    let with_other = br#"{"output_config":{"effort":"high","keep":"yes"}}"#;
    let output = strip_thinking_config(with_other, "claude");
    assert_eq!(
        path(&output, "output_config.keep"),
        Some(Value::String("yes".into()))
    );
}

#[test]
fn user_defined_detection_matches_nil_and_explicit_registry_flag() {
    let configured = model(false, Some(support(true, LEVELS)));
    let custom = model(true, None);
    assert!(is_user_defined_model(None));
    assert!(is_user_defined_model(Some(&custom)));
    assert!(!is_user_defined_model(Some(&configured)));
}

#[test]
fn openai_registered_model_applies_only_level_and_none_modes() {
    let applier = OpenAiApplier::new();
    let registered = model(false, Some(support(false, LEVELS)));

    let level = applier
        .apply(
            br#"{"model":"level-model"}"#,
            &config(ThinkingMode::Level, 0, "high"),
            Some(&registered),
        )
        .unwrap();
    assert_eq!(
        path(&level, "reasoning_effort"),
        Some(Value::String("high".into()))
    );

    let none = applier
        .apply(
            br#"{}"#,
            &config(ThinkingMode::None, 0, ""),
            Some(&registered),
        )
        .unwrap();
    assert_eq!(
        path(&none, "reasoning_effort"),
        Some(Value::String("minimal".into()))
    );

    for mode in [
        ThinkingMode::Budget,
        ThinkingMode::Auto,
        ThinkingMode::Unknown(8),
    ] {
        let body = br#"{ "model" : "level-model" }"#;
        assert_eq!(
            applier
                .apply(body, &config(mode, 8192, "high"), Some(&registered))
                .unwrap(),
            body
        );
    }
}

#[test]
fn openai_none_prefers_explicit_disable_then_config_then_first_level() {
    let applier = OpenAiApplier::new();
    for (registry_support, request, expected) in [
        (
            support(true, LEVELS),
            config(ThinkingMode::None, 0, "high"),
            "none",
        ),
        (
            support(false, NONE_LEVELS),
            config(ThinkingMode::None, 0, "high"),
            "none",
        ),
        (
            support(false, LEVELS),
            config(ThinkingMode::None, 7, "high"),
            "high",
        ),
        (
            support(false, LEVELS),
            config(ThinkingMode::None, 7, ""),
            "minimal",
        ),
    ] {
        let registered = model(false, Some(registry_support));
        let output = applier
            .apply(br#"{}"#, &request, Some(&registered))
            .unwrap();
        assert_eq!(
            path(&output, "reasoning_effort"),
            Some(Value::String(expected.into()))
        );
    }
}

#[test]
fn normal_model_without_thinking_is_exact_noop_even_for_invalid_body() {
    let applier = OpenAiApplier::new();
    let unsupported = model(false, None);
    let body = b"not json";
    assert_eq!(
        applier
            .apply(
                body,
                &config(ThinkingMode::Level, 0, "high"),
                Some(&unsupported),
            )
            .unwrap(),
        body
    );
}

#[test]
fn compatible_openai_accepts_all_modes_and_normalizes_invalid_bodies() {
    let applier = OpenAiApplier::new();
    let custom = model(true, None);
    let cases = [
        (config(ThinkingMode::Level, 0, "high"), Some("high")),
        (config(ThinkingMode::None, 0, ""), Some("none")),
        (config(ThinkingMode::None, 0, "minimal"), Some("minimal")),
        (config(ThinkingMode::Auto, -1, ""), Some("auto")),
        (config(ThinkingMode::Budget, 8192, ""), Some("medium")),
        (config(ThinkingMode::Budget, 64_000, ""), Some("xhigh")),
        (config(ThinkingMode::Budget, -2, ""), None),
        (config(ThinkingMode::Unknown(9), 0, ""), None),
    ];
    for (request, expected) in cases {
        let output = applier.apply(b"invalid", &request, Some(&custom)).unwrap();
        assert_eq!(
            path(&output, "reasoning_effort"),
            expected.map(|value| Value::String(value.into()))
        );
        assert!(serde_json::from_slice::<Value>(&output).is_ok());
    }

    let through_nil = applier
        .apply(br#"{}"#, &config(ThinkingMode::Budget, 512, ""), None)
        .unwrap();
    assert_eq!(
        path(&through_nil, "reasoning_effort"),
        Some(Value::String("minimal".into()))
    );
}

#[test]
fn xai_delegates_identical_effort_semantics_to_nested_responses_path() {
    let applier = XaiApplier::new();
    let registered = model(false, Some(support(true, LEVELS)));
    let level = applier
        .apply(
            br#"{"model":"xai-level-model"}"#,
            &config(ThinkingMode::Level, 0, "high"),
            Some(&registered),
        )
        .unwrap();
    assert_eq!(
        path(&level, "reasoning.effort"),
        Some(Value::String("high".into()))
    );
    assert_eq!(path(&level, "reasoning_effort"), None);

    let disabled = applier
        .apply(
            br#"{}"#,
            &config(ThinkingMode::None, 0, ""),
            Some(&registered),
        )
        .unwrap();
    assert_eq!(
        path(&disabled, "reasoning.effort"),
        Some(Value::String("none".into()))
    );

    let compatible = applier
        .apply(br#"{}"#, &config(ThinkingMode::Budget, 32_768, ""), None)
        .unwrap();
    assert_eq!(
        path(&compatible, "reasoning.effort"),
        Some(Value::String("xhigh".into()))
    );
}

#[test]
fn xai_matrix_edges_are_validated_before_nested_application() {
    let xai = ModelInfo {
        id: "xai-level-model",
        provider_type: "xai",
        user_defined: false,
        max_completion_tokens: 0,
        thinking: Some(support(true, XAI_LEVELS)),
    };
    let cases = [
        ("openai", config(ThinkingMode::Level, 0, "xhigh"), "high"),
        (
            "openai-response",
            config(ThinkingMode::Level, 0, "minimal"),
            "low",
        ),
        ("gemini", config(ThinkingMode::Budget, 32_768, ""), "high"),
        ("claude", config(ThinkingMode::Budget, 0, ""), "none"),
        ("claude", config(ThinkingMode::Level, 0, "max"), "high"),
    ];
    for (from_format, request, expected) in cases {
        let validated = validate_config(request, Some(&xai), from_format, "xai", false).unwrap();
        let output = XaiApplier::new()
            .apply(br#"{}"#, &validated, Some(&xai))
            .unwrap();
        assert_eq!(
            path(&output, "reasoning.effort"),
            Some(Value::String(expected.into())),
            "from={from_format}"
        );
    }
}

#[test]
fn provider_setter_preserves_byte_identity_when_effort_already_matches() {
    let registered = model(false, Some(support(true, LEVELS)));
    let body = br#"{ "reasoning_effort" : "high", "model" : "m" }"#;
    assert_eq!(
        OpenAiApplier::new()
            .apply(
                body,
                &config(ThinkingMode::Level, 0, "high"),
                Some(&registered),
            )
            .unwrap(),
        body
    );
}

#[test]
fn provider_setter_matches_sjson_for_valid_arrays_and_scalars() {
    let registered = model(false, Some(support(true, LEVELS)));
    let request = config(ThinkingMode::Level, 0, "high");
    let array = br#"[1, 2]"#;
    assert_eq!(
        OpenAiApplier::new()
            .apply(array, &request, Some(&registered))
            .unwrap(),
        array
    );

    let scalar = br#"17"#;
    let output = OpenAiApplier::new()
        .apply(scalar, &request, Some(&registered))
        .unwrap();
    assert_eq!(
        path(&output, "reasoning_effort"),
        Some(Value::String("high".into()))
    );
}

#[test]
fn none_with_empty_capability_returns_normalized_body_without_fake_effort() {
    let empty = model(false, Some(support(false, NO_LEVELS)));
    assert_eq!(
        OpenAiApplier::new()
            .apply(b"invalid", &config(ThinkingMode::None, 7, ""), Some(&empty),)
            .unwrap(),
        br#"{}"#
    );
}

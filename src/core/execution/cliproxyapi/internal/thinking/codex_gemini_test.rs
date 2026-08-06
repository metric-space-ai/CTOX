// refs: internal/thinking/provider/{codex,gemini}/apply.go
// @ ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::registry::{ModelInfo, ThinkingSupport};

use super::*;

const NO_LEVELS: &[&str] = &[];
const LEVELS: &[&str] = &["minimal", "low", "medium", "high"];
const SUBSET_LEVELS: &[&str] = &["low", "high"];

// Keeping the capability columns inline makes the table fixtures below map
// directly to the upstream Go matrix instead of hiding them in named presets.
#[allow(clippy::too_many_arguments)]
fn model(
    id: &'static str,
    provider_type: &'static str,
    user_defined: bool,
    min: Option<u64>,
    max: Option<u64>,
    zero_allowed: bool,
    dynamic_allowed: bool,
    levels: &'static [&'static str],
) -> ModelInfo {
    ModelInfo {
        id,
        provider_type,
        user_defined,
        max_completion_tokens: 0,
        thinking: if user_defined {
            None
        } else {
            Some(ThinkingSupport {
                min,
                max,
                zero_allowed,
                dynamic_allowed,
                levels,
            })
        },
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
fn codex_reuses_level_delegate_at_responses_path() {
    let codex = model(
        "level-model",
        "openai",
        false,
        None,
        None,
        false,
        false,
        LEVELS,
    );
    let applier = CodexApplier::new();
    let level = applier
        .apply(
            br#"{"model":"level-model"}"#,
            &config(ThinkingMode::Level, 0, "medium"),
            Some(&codex),
        )
        .unwrap();
    assert_eq!(
        path(&level, "reasoning.effort"),
        Some(Value::String("medium".into()))
    );
    assert_eq!(path(&level, "reasoning_effort"), None);

    let disabled = applier
        .apply(br#"{}"#, &config(ThinkingMode::None, 0, ""), Some(&codex))
        .unwrap();
    assert_eq!(
        path(&disabled, "reasoning.effort"),
        Some(Value::String("minimal".into()))
    );
}

#[test]
fn codex_e2e_validation_covers_budget_none_auto_and_strict_level() {
    let codex = model(
        "level-model",
        "openai",
        false,
        None,
        None,
        false,
        false,
        LEVELS,
    );
    let cases = [
        ("gemini", config(ThinkingMode::Budget, 8192, ""), "medium"),
        ("gemini", config(ThinkingMode::Budget, 64_000, ""), "high"),
        ("gemini", config(ThinkingMode::Budget, 0, ""), "minimal"),
        ("gemini", config(ThinkingMode::Budget, -1, ""), "medium"),
    ];
    for (from_format, request, expected) in cases {
        let validated =
            validate_config(request, Some(&codex), from_format, "codex", false).unwrap();
        let output = CodexApplier::new()
            .apply(br#"{}"#, &validated, Some(&codex))
            .unwrap();
        assert_eq!(
            path(&output, "reasoning.effort"),
            Some(Value::String(expected.into()))
        );
    }

    let error = validate_config(
        config(ThinkingMode::Level, 0, "xhigh"),
        Some(&codex),
        "openai-response",
        "codex",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::LevelNotSupported);
}

#[test]
fn codex_compatible_and_noop_boundaries_match_openai_delegate() {
    let custom = model(
        "custom", "openai", true, None, None, false, false, NO_LEVELS,
    );
    let compatible = CodexApplier::new()
        .apply(
            b"invalid",
            &config(ThinkingMode::Budget, 8192, ""),
            Some(&custom),
        )
        .unwrap();
    assert_eq!(
        path(&compatible, "reasoning.effort"),
        Some(Value::String("medium".into()))
    );

    let body = br#"{ "keep" : true }"#;
    assert_eq!(
        CodexApplier::new()
            .apply(
                body,
                &config(ThinkingMode::Auto, -1, ""),
                Some(&model(
                    "registered",
                    "openai",
                    false,
                    None,
                    None,
                    true,
                    false,
                    LEVELS,
                )),
            )
            .unwrap(),
        body
    );
}

#[test]
fn gemini_budget_model_emits_budget_for_budget_none_and_auto() {
    let gemini = model(
        "gemini-budget-model",
        "gemini",
        false,
        Some(128),
        Some(20_000),
        false,
        true,
        NO_LEVELS,
    );
    let cases = [
        (config(ThinkingMode::Budget, 8192, ""), 8192),
        (config(ThinkingMode::None, 128, ""), 128),
        (config(ThinkingMode::Auto, -1, ""), -1),
    ];
    for (request, expected) in cases {
        let output = GeminiApplier::new()
            .apply(br#"{}"#, &request, Some(&gemini))
            .unwrap();
        assert_eq!(
            path(&output, "generationConfig.thinkingConfig.thinkingBudget"),
            Some(Value::Number(expected.into()))
        );
        assert_eq!(
            path(&output, "generationConfig.thinkingConfig.thinkingLevel"),
            None
        );
    }
}

#[test]
fn gemini_level_format_normalizes_aliases_and_restores_only_boolean_visibility() {
    let gemini = model(
        "gemini-level-model",
        "gemini",
        false,
        None,
        None,
        false,
        false,
        SUBSET_LEVELS,
    );
    let body = br#"{"generationConfig":{"thinkingConfig":{"thinkingBudget":123,"thinking_budget":456,"thinking_level":"low","includeThoughts":false,"include_thoughts":true,"keep":1}}}"#;
    let output = GeminiApplier::new()
        .apply(body, &config(ThinkingMode::Level, 0, "high"), Some(&gemini))
        .unwrap();
    assert_eq!(
        path(&output, "generationConfig.thinkingConfig.thinkingLevel"),
        Some(Value::String("high".into()))
    );
    for alias in ["thinkingBudget", "thinking_budget", "thinking_level"] {
        assert_eq!(
            path(&output, &format!("generationConfig.thinkingConfig.{alias}")),
            None
        );
    }
    assert_eq!(
        path(&output, "generationConfig.thinkingConfig.includeThoughts"),
        Some(Value::Bool(false))
    );
    assert_eq!(
        path(&output, "generationConfig.thinkingConfig.include_thoughts"),
        None
    );
    assert_eq!(
        path(&output, "generationConfig.thinkingConfig.keep"),
        Some(Value::Number(1.into()))
    );

    let non_boolean = br#"{"generationConfig":{"thinkingConfig":{"includeThoughts":"true"}}}"#;
    let output = GeminiApplier::new()
        .apply(
            non_boolean,
            &config(ThinkingMode::Level, 0, "high"),
            Some(&gemini),
        )
        .unwrap();
    assert_eq!(
        path(&output, "generationConfig.thinkingConfig.includeThoughts"),
        None
    );
}

#[test]
fn gemini_none_level_semantics_remove_or_retain_amount_as_upstream() {
    let toggled = model(
        "gemini-toggle",
        "gemini",
        false,
        Some(128),
        Some(32_768),
        true,
        true,
        SUBSET_LEVELS,
    );
    let body = br#"{"generationConfig":{"thinkingConfig":{"thinkingLevel":"high","includeThoughts":true}},"keep":1}"#;
    let disabled = GeminiApplier::new()
        .apply(body, &config(ThinkingMode::None, 0, ""), Some(&toggled))
        .unwrap();
    assert_eq!(path(&disabled, "generationConfig.thinkingConfig"), None);
    assert_eq!(path(&disabled, "keep"), Some(Value::Number(1.into())));

    let clamped = GeminiApplier::new()
        .apply(
            body,
            &config(ThinkingMode::None, 128, "low"),
            Some(&toggled),
        )
        .unwrap();
    assert_eq!(
        path(&clamped, "generationConfig.thinkingConfig.thinkingLevel"),
        Some(Value::String("low".into()))
    );
    assert_eq!(
        path(&clamped, "generationConfig.thinkingConfig.includeThoughts"),
        Some(Value::Bool(true))
    );
}

#[test]
fn gemini_e2e_capability_validation_selects_level_budget_and_disabled_shapes() {
    let budget = model(
        "gemini-budget",
        "gemini",
        false,
        Some(128),
        Some(20_000),
        false,
        true,
        NO_LEVELS,
    );
    let hybrid = model(
        "gemini-hybrid",
        "gemini",
        false,
        Some(128),
        Some(32_768),
        false,
        true,
        SUBSET_LEVELS,
    );
    let toggle = model(
        "gemini-toggle",
        "gemini",
        false,
        Some(128),
        Some(32_768),
        true,
        true,
        SUBSET_LEVELS,
    );

    let cases = [
        (
            &budget,
            "openai",
            config(ThinkingMode::Level, 0, "medium"),
            "generationConfig.thinkingConfig.thinkingBudget",
            Value::Number(8192.into()),
        ),
        (
            &budget,
            "openai",
            config(ThinkingMode::Level, 0, "none"),
            "generationConfig.thinkingConfig.thinkingBudget",
            Value::Number(128.into()),
        ),
        (
            &budget,
            "openai",
            config(ThinkingMode::Level, 0, "auto"),
            "generationConfig.thinkingConfig.thinkingBudget",
            Value::Number((-1).into()),
        ),
        (
            &hybrid,
            "openai",
            config(ThinkingMode::Level, 0, "xhigh"),
            "generationConfig.thinkingConfig.thinkingLevel",
            Value::String("high".into()),
        ),
        (
            &hybrid,
            "claude",
            config(ThinkingMode::Budget, 8192, ""),
            "generationConfig.thinkingConfig.thinkingBudget",
            Value::Number(8192.into()),
        ),
    ];
    for (target, from, request, output_path, expected) in cases {
        let validated = validate_config(request, Some(target), from, "gemini", false).unwrap();
        let output = GeminiApplier::new()
            .apply(br#"{}"#, &validated, Some(target))
            .unwrap();
        assert_eq!(path(&output, output_path), Some(expected));
    }

    let validated = validate_config(
        config(ThinkingMode::Level, 0, "none"),
        Some(&toggle),
        "openai",
        "gemini",
        false,
    )
    .unwrap();
    let disabled = GeminiApplier::new()
        .apply(
            br#"{"generationConfig":{"thinkingConfig":{"includeThoughts":false}}}"#,
            &validated,
            Some(&toggle),
        )
        .unwrap();
    assert_eq!(path(&disabled, "generationConfig.thinkingConfig"), None);
}

#[test]
fn gemini_compatible_routes_all_modes_without_registry_capability() {
    let custom = model(
        "custom", "gemini", true, None, None, false, false, NO_LEVELS,
    );
    let cases = [
        (
            config(ThinkingMode::Level, 0, "high"),
            "generationConfig.thinkingConfig.thinkingLevel",
            Value::String("high".into()),
        ),
        (
            config(ThinkingMode::None, 0, "low"),
            "generationConfig.thinkingConfig.thinkingLevel",
            Value::String("low".into()),
        ),
        (
            config(ThinkingMode::None, 0, ""),
            "generationConfig.thinkingConfig.thinkingBudget",
            Value::Number(0.into()),
        ),
        (
            config(ThinkingMode::Auto, -1, ""),
            "generationConfig.thinkingConfig.thinkingBudget",
            Value::Number((-1).into()),
        ),
        (
            config(ThinkingMode::Budget, 8192, ""),
            "generationConfig.thinkingConfig.thinkingBudget",
            Value::Number(8192.into()),
        ),
    ];
    for (request, output_path, expected) in cases {
        let output = GeminiApplier::new()
            .apply(b"invalid", &request, Some(&custom))
            .unwrap();
        assert_eq!(path(&output, output_path), Some(expected));
    }
}

#[test]
fn gemini_preserves_bytes_on_unsupported_modes_capabilities_and_sjson_noops() {
    let body = br#"{ "keep" : true }"#;
    let registered = model(
        "gemini",
        "gemini",
        false,
        Some(128),
        Some(20_000),
        false,
        true,
        NO_LEVELS,
    );
    assert_eq!(
        GeminiApplier::new()
            .apply(
                body,
                &config(ThinkingMode::Unknown(99), 0, ""),
                Some(&registered),
            )
            .unwrap(),
        body
    );

    let unsupported = ModelInfo {
        id: "unsupported",
        provider_type: "gemini",
        user_defined: false,
        max_completion_tokens: 0,
        thinking: None,
    };
    assert_eq!(
        GeminiApplier::new()
            .apply(
                b"invalid",
                &config(ThinkingMode::Budget, 8192, ""),
                Some(&unsupported),
            )
            .unwrap(),
        b"invalid"
    );

    let array = br#"[1, 2]"#;
    assert_eq!(
        GeminiApplier::new()
            .apply(
                array,
                &config(ThinkingMode::Budget, 8192, ""),
                Some(&registered),
            )
            .unwrap(),
        array
    );

    let same = br#"{ "generationConfig" : { "thinkingConfig" : { "thinkingBudget" : 8192 } } }"#;
    assert_eq!(
        GeminiApplier::new()
            .apply(
                same,
                &config(ThinkingMode::Budget, 8192, ""),
                Some(&registered),
            )
            .unwrap(),
        same
    );
}

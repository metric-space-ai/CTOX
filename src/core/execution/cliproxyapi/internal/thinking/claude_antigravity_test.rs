// refs: internal/thinking/provider/{claude,antigravity}/apply.go and
// test/thinking_conversion_test.go @ ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::registry::{ModelInfo, ThinkingSupport};

use super::*;

const NO_LEVELS: &[&str] = &[];
const CLAUDE_LEVELS: &[&str] = &["low", "medium", "high", "max"];
const SUBSET_LEVELS: &[&str] = &["low", "high"];

fn support(
    min: Option<u64>,
    max: Option<u64>,
    zero_allowed: bool,
    dynamic_allowed: bool,
    levels: &'static [&'static str],
) -> ThinkingSupport {
    ThinkingSupport {
        min,
        max,
        zero_allowed,
        dynamic_allowed,
        levels,
    }
}

fn model(
    id: &'static str,
    provider_type: &'static str,
    user_defined: bool,
    max_completion_tokens: usize,
    thinking: Option<ThinkingSupport>,
) -> ModelInfo {
    ModelInfo {
        id,
        provider_type,
        user_defined,
        max_completion_tokens,
        thinking,
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
fn claude_applies_manual_adaptive_auto_and_disabled_shapes() {
    let manual = model(
        "claude-manual",
        "claude",
        false,
        0,
        Some(support(Some(1024), Some(128_000), true, false, NO_LEVELS)),
    );
    let adaptive = model(
        "claude-adaptive",
        "claude",
        false,
        0,
        Some(support(None, None, true, true, CLAUDE_LEVELS)),
    );

    let enabled = ClaudeApplier::new()
        .apply(
            br#"{"thinking":{"display":"summarized"},"output_config":{"effort":"low","keep":true}}"#,
            &config(ThinkingMode::Budget, 8192, ""),
            Some(&manual),
        )
        .unwrap();
    assert_eq!(
        path(&enabled, "thinking.type"),
        Some(Value::String("enabled".into()))
    );
    assert_eq!(
        path(&enabled, "thinking.budget_tokens"),
        Some(Value::Number(8192.into()))
    );
    assert_eq!(
        path(&enabled, "thinking.display"),
        Some(Value::String("summarized".into()))
    );
    assert_eq!(path(&enabled, "output_config.effort"), None);
    assert_eq!(
        path(&enabled, "output_config.keep"),
        Some(Value::Bool(true))
    );

    let adapted = ClaudeApplier::new()
        .apply(
            br#"{"thinking":{"budget_tokens":8192,"display":"summarized"}}"#,
            &config(ThinkingMode::Level, 0, "high"),
            Some(&adaptive),
        )
        .unwrap();
    assert_eq!(
        path(&adapted, "thinking.type"),
        Some(Value::String("adaptive".into()))
    );
    assert_eq!(path(&adapted, "thinking.budget_tokens"), None);
    assert_eq!(
        path(&adapted, "output_config.effort"),
        Some(Value::String("high".into()))
    );
    assert_eq!(
        path(&adapted, "thinking.display"),
        Some(Value::String("summarized".into()))
    );

    let adaptive_auto = ClaudeApplier::new()
        .apply(
            br#"{"thinking":{"budget_tokens":1},"output_config":{"effort":"high"}}"#,
            &config(ThinkingMode::Auto, -1, ""),
            Some(&adaptive),
        )
        .unwrap();
    assert_eq!(
        path(&adaptive_auto, "thinking.type"),
        Some(Value::String("adaptive".into()))
    );
    assert_eq!(path(&adaptive_auto, "thinking.budget_tokens"), None);
    assert_eq!(path(&adaptive_auto, "output_config"), None);

    let legacy_auto = ClaudeApplier::new()
        .apply(br#"{}"#, &config(ThinkingMode::Auto, -1, ""), Some(&manual))
        .unwrap();
    assert_eq!(
        path(&legacy_auto, "thinking.type"),
        Some(Value::String("enabled".into()))
    );
    assert_eq!(path(&legacy_auto, "thinking.budget_tokens"), None);

    let disabled = ClaudeApplier::new()
        .apply(
            br#"{"thinking":{"budget_tokens":8192,"display":"summarized"},"output_config":{"effort":"high"}}"#,
            &config(ThinkingMode::None, 0, ""),
            Some(&manual),
        )
        .unwrap();
    assert_eq!(
        path(&disabled, "thinking.type"),
        Some(Value::String("disabled".into()))
    );
    assert_eq!(path(&disabled, "thinking.budget_tokens"), None);
    assert_eq!(path(&disabled, "thinking.display"), None);
    assert_eq!(path(&disabled, "output_config"), None);
}

#[test]
fn claude_enforces_request_and_registry_completion_limits_exactly() {
    let claude = model(
        "claude-limited",
        "claude",
        false,
        4096,
        Some(support(Some(1024), Some(128_000), true, false, NO_LEVELS)),
    );
    let defaulted = ClaudeApplier::new()
        .apply(
            br#"{}"#,
            &config(ThinkingMode::Budget, 8192, ""),
            Some(&claude),
        )
        .unwrap();
    assert_eq!(
        path(&defaulted, "max_tokens"),
        Some(Value::Number(4096.into()))
    );
    assert_eq!(
        path(&defaulted, "thinking.budget_tokens"),
        Some(Value::Number(4095.into()))
    );

    let requested = ClaudeApplier::new()
        .apply(
            br#"{"max_tokens":"4096"}"#,
            &config(ThinkingMode::Budget, 8192, ""),
            Some(&claude),
        )
        .unwrap();
    assert_eq!(
        path(&requested, "max_tokens"),
        Some(Value::String("4096".into()))
    );
    assert_eq!(
        path(&requested, "thinking.budget_tokens"),
        Some(Value::Number(4095.into()))
    );

    let below_min = ClaudeApplier::new()
        .apply(
            br#"{"max_tokens":1024}"#,
            &config(ThinkingMode::Budget, 8192, ""),
            Some(&claude),
        )
        .unwrap();
    assert_eq!(
        path(&below_min, "thinking.budget_tokens"),
        Some(Value::Number(8192.into()))
    );
}

#[test]
fn claude_compatible_and_noop_boundaries_preserve_upstream_bytes() {
    let custom = model("claude-custom", "claude", true, 0, None);
    let budget_zero = ClaudeApplier::new()
        .apply(
            b"invalid",
            &config(ThinkingMode::Budget, 0, ""),
            Some(&custom),
        )
        .unwrap();
    assert_eq!(
        path(&budget_zero, "thinking.type"),
        Some(Value::String("enabled".into()))
    );
    assert_eq!(
        path(&budget_zero, "thinking.budget_tokens"),
        Some(Value::Number(0.into()))
    );

    let auto = ClaudeApplier::new()
        .apply(b"", &config(ThinkingMode::Auto, -1, ""), Some(&custom))
        .unwrap();
    assert_eq!(
        path(&auto, "thinking.type"),
        Some(Value::String("enabled".into()))
    );
    assert_eq!(path(&auto, "thinking.budget_tokens"), None);

    let level = ClaudeApplier::new()
        .apply(
            br#"{}"#,
            &config(ThinkingMode::Level, 0, "max"),
            Some(&custom),
        )
        .unwrap();
    assert_eq!(
        path(&level, "thinking.type"),
        Some(Value::String("adaptive".into()))
    );
    assert_eq!(
        path(&level, "output_config.effort"),
        Some(Value::String("max".into()))
    );

    let body = br#"{ "keep" : true }"#;
    assert_eq!(
        ClaudeApplier::new()
            .apply(
                body,
                &config(ThinkingMode::Unknown(9), 0, ""),
                Some(&custom)
            )
            .unwrap(),
        body
    );
    let unsupported = model("unsupported", "claude", false, 0, None);
    assert_eq!(
        ClaudeApplier::new()
            .apply(
                b"invalid",
                &config(ThinkingMode::Budget, 8192, ""),
                Some(&unsupported)
            )
            .unwrap(),
        b"invalid"
    );
    let same = br#"{ "thinking" : { "type" : "disabled" } }"#;
    assert_eq!(
        ClaudeApplier::new()
            .apply(same, &config(ThinkingMode::None, 0, ""), Some(&custom))
            .unwrap(),
        same
    );
}

#[test]
fn claude_e2e_validation_ports_budget_and_adaptive_matrix_errors() {
    let manual = model(
        "claude-budget-model",
        "claude",
        false,
        0,
        Some(support(Some(1024), Some(128_000), true, false, NO_LEVELS)),
    );
    let cases = [
        ("openai", config(ThinkingMode::Level, 0, "medium"), 8192),
        ("openai", config(ThinkingMode::Level, 0, "xhigh"), 32_768),
        ("openai", config(ThinkingMode::Level, 0, "auto"), 64_512),
        ("gemini", config(ThinkingMode::Budget, 200_000, ""), 128_000),
    ];
    for (from, request, expected) in cases {
        let validated = validate_config(request, Some(&manual), from, "claude", false).unwrap();
        let output = ClaudeApplier::new()
            .apply(br#"{}"#, &validated, Some(&manual))
            .unwrap();
        assert_eq!(
            path(&output, "thinking.budget_tokens"),
            Some(Value::Number(expected.into()))
        );
    }
    let none = validate_config(
        config(ThinkingMode::Level, 0, "none"),
        Some(&manual),
        "openai",
        "claude",
        false,
    )
    .unwrap();
    let disabled = ClaudeApplier::new()
        .apply(br#"{}"#, &none, Some(&manual))
        .unwrap();
    assert_eq!(
        path(&disabled, "thinking.type"),
        Some(Value::String("disabled".into()))
    );

    let error = validate_config(
        config(ThinkingMode::Budget, 200_000, ""),
        Some(&manual),
        "claude",
        "claude",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::BudgetOutOfRange);

    let adaptive = model(
        "claude-adaptive",
        "claude",
        false,
        0,
        Some(support(None, None, true, true, CLAUDE_LEVELS)),
    );
    let validated = validate_config(
        config(ThinkingMode::Level, 0, "max"),
        Some(&adaptive),
        "claude",
        "claude",
        false,
    )
    .unwrap();
    let output = ClaudeApplier::new()
        .apply(br#"{}"#, &validated, Some(&adaptive))
        .unwrap();
    assert_eq!(
        path(&output, "output_config.effort"),
        Some(Value::String("max".into()))
    );
    let error = validate_config(
        config(ThinkingMode::Level, 0, "xhigh"),
        Some(&adaptive),
        "claude",
        "claude",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::LevelNotSupported);
}

#[test]
fn antigravity_routes_level_budget_none_auto_and_normalizes_aliases() {
    let budget_model = model(
        "antigravity-budget",
        "antigravity",
        false,
        0,
        Some(support(Some(128), Some(20_000), true, true, NO_LEVELS)),
    );
    for (request, expected) in [
        (config(ThinkingMode::Budget, 8192, ""), 8192),
        (config(ThinkingMode::None, 0, ""), 0),
        (config(ThinkingMode::Auto, -1, ""), -1),
    ] {
        let output = AntigravityApplier::new()
            .apply(br#"{}"#, &request, Some(&budget_model))
            .unwrap();
        assert_eq!(
            path(
                &output,
                "request.generationConfig.thinkingConfig.thinkingBudget"
            ),
            Some(Value::Number(expected.into()))
        );
    }

    let level_model = model(
        "antigravity-level",
        "antigravity",
        false,
        0,
        Some(support(None, None, true, true, SUBSET_LEVELS)),
    );
    let body = br#"{"request":{"generationConfig":{"thinkingConfig":{"thinkingBudget":1,"thinking_budget":2,"thinking_level":"low","includeThoughts":false,"include_thoughts":true,"keep":1}}}}"#;
    let output = AntigravityApplier::new()
        .apply(
            body,
            &config(ThinkingMode::Level, 0, "high"),
            Some(&level_model),
        )
        .unwrap();
    assert_eq!(
        path(
            &output,
            "request.generationConfig.thinkingConfig.thinkingLevel"
        ),
        Some(Value::String("high".into()))
    );
    for alias in [
        "thinkingBudget",
        "thinking_budget",
        "thinking_level",
        "include_thoughts",
    ] {
        assert_eq!(
            path(
                &output,
                &format!("request.generationConfig.thinkingConfig.{alias}")
            ),
            None
        );
    }
    assert_eq!(
        path(
            &output,
            "request.generationConfig.thinkingConfig.includeThoughts"
        ),
        Some(Value::Bool(false))
    );
    let disabled = AntigravityApplier::new()
        .apply(body, &config(ThinkingMode::None, 0, ""), Some(&level_model))
        .unwrap();
    assert_eq!(
        path(&disabled, "request.generationConfig.thinkingConfig"),
        None
    );
}

#[test]
fn antigravity_enforces_claude_limits_and_summary_visibility_order() {
    let claude = model(
        "claude-antigravity",
        "claude",
        false,
        4096,
        Some(support(Some(1024), Some(128_000), true, false, NO_LEVELS)),
    );
    let defaulted = AntigravityApplier::new()
        .apply(
            br#"{}"#,
            &config(ThinkingMode::Budget, 8192, ""),
            Some(&claude),
        )
        .unwrap();
    assert_eq!(
        path(&defaulted, "request.generationConfig.maxOutputTokens"),
        Some(Value::Number(4096.into()))
    );
    assert_eq!(
        path(
            &defaulted,
            "request.generationConfig.thinkingConfig.thinkingBudget"
        ),
        Some(Value::Number(4095.into()))
    );

    let below_min = AntigravityApplier::new()
        .apply(
            br#"{"request":{"generationConfig":{"thinkingConfig":{"include_thoughts":true,"keep":1}}}}"#,
            &config(ThinkingMode::Budget, 512, ""),
            Some(&claude),
        )
        .unwrap();
    assert_eq!(
        path(
            &below_min,
            "request.generationConfig.thinkingConfig.includeThoughts"
        ),
        Some(Value::Bool(true))
    );
    assert_eq!(
        path(
            &below_min,
            "request.generationConfig.thinkingConfig.thinkingBudget"
        ),
        None
    );
    assert_eq!(
        path(&below_min, "request.generationConfig.thinkingConfig.keep"),
        None
    );
    assert_eq!(
        path(&below_min, "request.generationConfig.maxOutputTokens"),
        None
    );

    let requested_too_low = AntigravityApplier::new()
        .apply(
            br#"{"request":{"generationConfig":{"maxOutputTokens":1024}}}"#,
            &config(ThinkingMode::Budget, 8192, ""),
            Some(&claude),
        )
        .unwrap();
    assert_eq!(
        path(
            &requested_too_low,
            "request.generationConfig.thinkingConfig"
        ),
        None
    );
    assert_eq!(
        path(
            &requested_too_low,
            "request.generationConfig.maxOutputTokens"
        ),
        Some(Value::Number(1024.into()))
    );
}

#[test]
fn antigravity_compatible_and_sjson_noops_are_byte_exact() {
    let custom = model("claude-custom", "antigravity", true, 4096, None);
    let compatible = AntigravityApplier::new()
        .apply(
            b"invalid",
            &config(ThinkingMode::Budget, 8192, ""),
            Some(&custom),
        )
        .unwrap();
    assert_eq!(
        path(&compatible, "request.generationConfig.maxOutputTokens"),
        Some(Value::Number(4096.into()))
    );
    assert_eq!(
        path(
            &compatible,
            "request.generationConfig.thinkingConfig.thinkingBudget"
        ),
        Some(Value::Number(4095.into()))
    );

    let body = br#"{ "keep" : true }"#;
    assert_eq!(
        AntigravityApplier::new()
            .apply(
                body,
                &config(ThinkingMode::Unknown(7), 0, ""),
                Some(&custom)
            )
            .unwrap(),
        body
    );
    let array = br#"[1, 2]"#;
    let known = model(
        "antigravity-budget",
        "antigravity",
        false,
        0,
        Some(support(Some(128), Some(20_000), true, true, NO_LEVELS)),
    );
    assert_eq!(
        AntigravityApplier::new()
            .apply(array, &config(ThinkingMode::Budget, 8192, ""), Some(&known))
            .unwrap(),
        array
    );
    let same = br#"{ "request" : { "generationConfig" : { "thinkingConfig" : { "thinkingBudget" : 8192 } } } }"#;
    assert_eq!(
        AntigravityApplier::new()
            .apply(same, &config(ThinkingMode::Budget, 8192, ""), Some(&known))
            .unwrap(),
        same
    );
}

#[test]
fn antigravity_e2e_validation_ports_budget_matrix_and_typed_error() {
    let model = model(
        "antigravity-budget-model",
        "antigravity",
        false,
        0,
        Some(support(Some(128), Some(20_000), true, true, NO_LEVELS)),
    );
    let cases = [
        ("gemini", config(ThinkingMode::Level, 0, "medium"), 8192),
        ("gemini", config(ThinkingMode::Level, 0, "xhigh"), 20_000),
        ("gemini", config(ThinkingMode::Level, 0, "none"), 0),
        ("gemini", config(ThinkingMode::Level, 0, "auto"), -1),
        ("claude", config(ThinkingMode::Budget, 64_000, ""), 20_000),
    ];
    for (from, request, expected) in cases {
        let validated = validate_config(request, Some(&model), from, "antigravity", true).unwrap();
        let output = AntigravityApplier::new()
            .apply(br#"{}"#, &validated, Some(&model))
            .unwrap();
        assert_eq!(
            path(
                &output,
                "request.generationConfig.thinkingConfig.thinkingBudget"
            ),
            Some(Value::Number(expected.into()))
        );
    }
    let error = validate_config(
        config(ThinkingMode::Budget, 64_000, ""),
        Some(&model),
        "antigravity",
        "antigravity",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::BudgetOutOfRange);
}

// refs: internal/thinking/provider/interactions/apply.go,
// internal/thinking/provider/kimi/apply.go @ ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd
// Port-Status: supplemental
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::registry::{ModelInfo, ThinkingSupport};

use super::*;

const LEVELS: &[&str] = &["low", "medium", "high"];

fn model(user_defined: bool, thinking: Option<ThinkingSupport>) -> ModelInfo {
    ModelInfo {
        id: "thinking-model",
        provider_type: "test",
        user_defined,
        max_completion_tokens: 0,
        thinking,
    }
}

fn support(levels: &'static [&'static str]) -> ThinkingSupport {
    ThinkingSupport {
        min: None,
        max: None,
        zero_allowed: false,
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

fn value(body: &[u8], path: &str) -> Option<Value> {
    let document: Value = serde_json::from_slice(body).ok()?;
    path.split('.')
        .try_fold(&document, |current, segment| current.get(segment))
        .cloned()
}

#[test]
fn interactions_strips_aliases_normalizes_levels_and_restores_summary_intent() {
    let registered = model(false, Some(support(LEVELS)));
    let body = br#"{"generation_config":{"thinkingLevel":"max","thinking_budget":12,"thinkingConfig":{"includeThoughts":true}},"generationConfig":{"thinkingLevel":"old"},"keep":1}"#;
    let output = InteractionsApplier::new()
        .apply(
            body,
            &config(ThinkingMode::Level, 0, "MAX"),
            Some(&registered),
        )
        .unwrap();
    assert_eq!(
        value(&output, "generation_config.thinking_level"),
        Some(Value::String("high".into()))
    );
    assert_eq!(
        value(&output, "generation_config.thinking_summaries"),
        Some(Value::String("auto".into()))
    );
    assert_eq!(value(&output, "generation_config.thinkingLevel"), None);
    assert_eq!(value(&output, "generationConfig.thinkingLevel"), None);
    assert_eq!(value(&output, "keep"), Some(Value::Number(1.into())));
}

#[test]
fn interactions_none_does_not_reactivate_summaries_and_budget_preserves_them() {
    let body = br#"{"generation_config":{"thinking_summaries":"AUTO","thinking_level":"high"}}"#;
    let none = InteractionsApplier::new()
        .apply(body, &config(ThinkingMode::None, 0, ""), None)
        .unwrap();
    assert_eq!(value(&none, "generation_config.thinking_level"), None);
    assert_eq!(value(&none, "generation_config.thinking_summaries"), None);

    let budget = InteractionsApplier::new()
        .apply(body, &config(ThinkingMode::Budget, 1024, ""), None)
        .unwrap();
    assert_eq!(
        value(&budget, "generation_config.thinking_level"),
        Some(Value::String("low".into()))
    );
    assert_eq!(
        value(&budget, "generation_config.thinking_summaries"),
        Some(Value::String("auto".into()))
    );
}

#[test]
fn interactions_invalid_and_unknown_modes_match_upstream_normalization_boundary() {
    assert_eq!(
        InteractionsApplier::new()
            .apply(b"invalid", &config(ThinkingMode::Auto, 0, ""), None)
            .unwrap(),
        b"{}"
    );
    assert_eq!(
        InteractionsApplier::new()
            .apply(b"invalid", &config(ThinkingMode::Unknown(7), 0, ""), None)
            .unwrap(),
        b"invalid"
    );
}

#[test]
fn kimi_registered_and_compatible_paths_emit_native_thinking_objects() {
    let registered = model(false, Some(support(LEVELS)));
    let enabled = KimiApplier::new()
        .apply(
            br#"{"reasoning_effort":"old","thinking":{"keep":true},"x":1}"#,
            &config(ThinkingMode::Level, 0, "high"),
            Some(&registered),
        )
        .unwrap();
    assert_eq!(value(&enabled, "reasoning_effort"), None);
    assert_eq!(
        value(&enabled, "thinking.type"),
        Some(Value::String("enabled".into()))
    );
    assert_eq!(
        value(&enabled, "thinking.effort"),
        Some(Value::String("high".into()))
    );
    assert_eq!(value(&enabled, "thinking.keep"), Some(Value::Bool(true)));

    let compatible = KimiApplier::new()
        .apply(br#"{}"#, &config(ThinkingMode::Budget, 8192, ""), None)
        .unwrap();
    assert_eq!(
        value(&compatible, "thinking.effort"),
        Some(Value::String("medium".into()))
    );
}

#[test]
fn kimi_none_disables_cleanly_and_respects_clamped_fallback_level() {
    let registered = model(false, Some(support(LEVELS)));
    let disabled = KimiApplier::new()
        .apply(
            br#"{"thinking":{"type":"enabled","effort":"high","keep":true},"reasoning_effort":"high"}"#,
            &config(ThinkingMode::None, 0, ""),
            Some(&registered),
        )
        .unwrap();
    assert_eq!(
        value(&disabled, "thinking.type"),
        Some(Value::String("disabled".into()))
    );
    assert_eq!(value(&disabled, "thinking.effort"), None);
    assert_eq!(value(&disabled, "thinking.keep"), None);
    assert_eq!(value(&disabled, "reasoning_effort"), None);

    let fallback = KimiApplier::new()
        .apply(
            br#"{}"#,
            &config(ThinkingMode::None, 0, "low"),
            Some(&registered),
        )
        .unwrap();
    assert_eq!(
        value(&fallback, "thinking.effort"),
        Some(Value::String("low".into()))
    );
}

#[test]
fn kimi_leaves_registered_models_without_thinking_support_byte_identical() {
    let unsupported = model(false, None);
    let body = br#"{ "reasoning_effort" : "keep" }"#;
    assert_eq!(
        KimiApplier::new()
            .apply(
                body,
                &config(ThinkingMode::Level, 0, "high"),
                Some(&unsupported)
            )
            .unwrap(),
        body
    );
}

// ref: internal/thinking/summary_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::registry::{ModelInfo, ThinkingSupport};

use super::summary::{
    apply_summary_config_for_provider, apply_summary_config_for_resolved_model,
    strip_inferred_claude_summary_activation,
};
use super::*;

fn config(mode: SummaryMode, detail: &str) -> SummaryConfig {
    SummaryConfig {
        mode,
        detail: detail.into(),
    }
}

fn json_path(body: &[u8], path: &str) -> Option<Value> {
    let document: Value = serde_json::from_slice(body).ok()?;
    path.split('.')
        .try_fold(&document, |value, segment| value.get(segment))
        .cloned()
}

#[test]
fn extracts_protocol_specific_summary_config() {
    assert_eq!(SummaryMode::from(17).as_i32(), 17);
    let cases = [
        (
            "openai",
            r#"{"reasoning_effort":"high"}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "openai",
            r#"{"reasoning_effort":"none"}"#,
            SummaryMode::Disabled,
            "",
        ),
        ("openai", r#"{}"#, SummaryMode::Unspecified, ""),
        (
            "openai",
            r#"{"reasoning_effort":null}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "openai",
            r#"{"reasoning_effort":17}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "openai",
            r#"{"reasoning_effort":"high","extra_body":{"google":{"thinking_config":{"include_thoughts":false}}}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "openai",
            r#"{"extra_body":{"google":{"thinking_config":{"include_thoughts":true}}}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "openai",
            r#"{"reasoning_effort":"high","reasoning":{"exclude":true}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "openai",
            r#"{"reasoning":{"effort":"high","exclude":false}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "openai",
            r#"{"reasoning_effort":"high","include_reasoning":false}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "openai",
            r#"{"include_reasoning":true}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "openai",
            r#"{"reasoning":{"enabled":false}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "openai",
            r#"{"reasoning":{"enabled":true}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "openai",
            r#"{"reasoning":{"exclude":true},"include_reasoning":true}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "openai",
            r#"{"include_reasoning":"false"}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "openai-response",
            r#"{"reasoning":{"effort":"high"}}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "openai-response",
            r#"{"reasoning":{"effort":"high","summary":"auto"}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "openai-response",
            r#"{"reasoning":{"summary":"concise"}}"#,
            SummaryMode::Enabled,
            "concise",
        ),
        (
            "openai-response",
            r#"{"reasoning":{"summary":null}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "openai-response",
            r#"{"reasoning":{"summary":true}}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "openai-response",
            r#"{"reasoning":{"generate_summary":"detailed"}}"#,
            SummaryMode::Enabled,
            "detailed",
        ),
        (
            "claude",
            r#"{"thinking":{"type":"adaptive","display":"summarized"}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "claude",
            r#"{"thinking":{"type":"enabled","budget_tokens":2048,"display":"omitted"}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "claude",
            r#"{"thinking":{"display":"summarized"}}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "claude",
            r#"{"thinking":{"type":"auto","display":"summarized"}}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "claude",
            r#"{"thinking":{"type":"enabled","display":"summarized"}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "claude",
            r#"{"thinking":{"type":"enabled","budget_tokens":0,"display":"summarized"}}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "claude",
            r#"{"thinking":{"type":"enabled","budget_tokens":-1,"display":"summarized"}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "claude",
            r#"{"thinking":{"type":"enabled","budget_tokens":-1,"display":"omitted"}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "gemini",
            r#"{"generationConfig":{"thinkingConfig":{"includeThoughts":true}}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "gemini",
            r#"{"generationConfig":{"thinkingConfig":{"includeThoughts":false}}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "antigravity",
            r#"{"request":{"generationConfig":{"thinkingConfig":{"includeThoughts":true}}}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_summaries":"auto"}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_summaries":"none"}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_config":{"include_thoughts":false}}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_config":{"includeThoughts":true}}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinkingConfig":{"include_thoughts":true}}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinkingConfig":{"includeThoughts":false}}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_summaries":"none"},"reasoning":{"summary":"auto"}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "interactions",
            r#"{"reasoning":{"summary":"auto"}}"#,
            SummaryMode::Enabled,
            "auto",
        ),
        (
            "interactions",
            r#"{"reasoning":{"summary":"none"}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_summaries":"none","thinking_config":{"include_thoughts":true}}}"#,
            SummaryMode::Disabled,
            "",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_config":{"include_thoughts":"false"}}}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_summaries":"detailed"}}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinking_summaries":true}}"#,
            SummaryMode::Unspecified,
            "",
        ),
        (
            "gemini",
            r#"{"generationConfig":{"thinkingConfig":{"includeThoughts":"true"}}}"#,
            SummaryMode::Unspecified,
            "",
        ),
    ];

    for (format, body, expected_mode, expected_detail) in cases {
        let actual = extract_summary_config(body.as_bytes(), format);
        assert_eq!(actual.mode, expected_mode, "format={format} body={body}");
        assert_eq!(
            actual.detail, expected_detail,
            "format={format} body={body}"
        );
    }
}

#[test]
fn explicit_openai_extractor_does_not_infer_visibility_from_effort() {
    assert_eq!(
        extract_explicit_summary_config(br#"{"reasoning_effort":"high"}"#, "openai").mode,
        SummaryMode::Unspecified
    );
    assert_eq!(
        extract_explicit_summary_config(
            br#"{"reasoning_effort":"high","reasoning":{"exclude":true}}"#,
            "openai"
        )
        .mode,
        SummaryMode::Disabled
    );
}

#[test]
fn applies_and_normalizes_each_target_protocol() {
    let cases = [
        (
            "openai",
            r#"{}"#,
            SummaryMode::Enabled,
            "",
            "reasoning_effort",
            None,
        ),
        (
            "openai",
            r#"{"reasoning_effort":"high"}"#,
            SummaryMode::Disabled,
            "",
            "reasoning_effort",
            Some(Value::String("high".into())),
        ),
        (
            "openai",
            r#"{"reasoning":{"exclude":false}}"#,
            SummaryMode::Disabled,
            "",
            "reasoning.exclude",
            Some(Value::Bool(true)),
        ),
        (
            "openai",
            r#"{"include_reasoning":true}"#,
            SummaryMode::Disabled,
            "",
            "include_reasoning",
            Some(Value::Bool(false)),
        ),
        (
            "claude",
            r#"{"thinking":{"type":"adaptive"}}"#,
            SummaryMode::Enabled,
            "",
            "thinking.display",
            Some(Value::String("summarized".into())),
        ),
        (
            "claude",
            r#"{"thinking":{"type":"enabled","budget_tokens":2048}}"#,
            SummaryMode::Disabled,
            "",
            "thinking.display",
            Some(Value::String("omitted".into())),
        ),
        (
            "gemini",
            r#"{}"#,
            SummaryMode::Enabled,
            "",
            "generationConfig.thinkingConfig.includeThoughts",
            Some(Value::Bool(true)),
        ),
        (
            "gemini",
            r#"{}"#,
            SummaryMode::Disabled,
            "",
            "generationConfig.thinkingConfig.includeThoughts",
            Some(Value::Bool(false)),
        ),
        (
            "antigravity",
            r#"{}"#,
            SummaryMode::Enabled,
            "",
            "request.generationConfig.thinkingConfig.includeThoughts",
            Some(Value::Bool(true)),
        ),
        (
            "interactions",
            r#"{}"#,
            SummaryMode::Enabled,
            "detailed",
            "generation_config.thinking_summaries",
            Some(Value::String("auto".into())),
        ),
        (
            "interactions",
            r#"{}"#,
            SummaryMode::Disabled,
            "",
            "generation_config.thinking_summaries",
            Some(Value::String("none".into())),
        ),
        (
            "openai-response",
            r#"{}"#,
            SummaryMode::Enabled,
            "concise",
            "reasoning.summary",
            Some(Value::String("concise".into())),
        ),
    ];

    for (format, body, mode, detail, path, expected) in cases {
        let output = apply_summary_config(body.as_bytes(), format, &config(mode, detail));
        assert_eq!(
            json_path(&output, path),
            expected,
            "format={format} output={}",
            String::from_utf8_lossy(&output)
        );
    }
}

#[test]
fn openai_chat_provider_dialects_only_write_documented_visibility() {
    let openai = apply_summary_config_for_provider(
        br#"{}"#,
        "openai",
        "model",
        "openai",
        None,
        &config(SummaryMode::Enabled, ""),
    );
    assert_eq!(json_path(&openai, "reasoning.exclude"), None);

    let openrouter = apply_summary_config_for_provider(
        br#"{}"#,
        "openai",
        "model",
        "prod-openrouter",
        None,
        &config(SummaryMode::Disabled, ""),
    );
    assert_eq!(
        json_path(&openrouter, "reasoning.exclude"),
        Some(Value::Bool(true))
    );

    let generic = apply_summary_config_for_provider(
        br#"{"reasoning":{"exclude":false}}"#,
        "openai",
        "model",
        "openai-compatibility",
        None,
        &config(SummaryMode::Disabled, ""),
    );
    assert_eq!(
        json_path(&generic, "reasoning.exclude"),
        Some(Value::Bool(true))
    );
}

#[test]
fn target_aliases_are_removed_after_canonical_write() {
    let cases = [
        (
            "gemini",
            r#"{"generationConfig":{"thinkingConfig":{"include_thoughts":true}}}"#,
            "generationConfig.thinkingConfig.includeThoughts",
            "generationConfig.thinkingConfig.include_thoughts",
        ),
        (
            "antigravity",
            r#"{"request":{"generationConfig":{"thinkingConfig":{"include_thoughts":true}}}}"#,
            "request.generationConfig.thinkingConfig.includeThoughts",
            "request.generationConfig.thinkingConfig.include_thoughts",
        ),
        (
            "interactions",
            r#"{"generation_config":{"thinkingSummaries":"auto"}}"#,
            "generation_config.thinking_summaries",
            "generation_config.thinkingSummaries",
        ),
    ];
    for (format, body, canonical, alias) in cases {
        let output =
            apply_summary_config(body.as_bytes(), format, &config(SummaryMode::Enabled, ""));
        assert!(json_path(&output, canonical).is_some());
        assert!(json_path(&output, alias).is_none());
    }
}

#[test]
fn claude_display_requires_active_thinking_and_preserves_original_bytes() {
    for body in [
        br#"{}"#.as_slice(),
        br#"{"messages":[{"role":"user","content":"hi"}]}"#.as_slice(),
        br#"{"thinking":{"type":"disabled"}}"#.as_slice(),
    ] {
        for mode in [SummaryMode::Enabled, SummaryMode::Disabled] {
            assert_eq!(
                apply_summary_config(body, "claude", &config(mode, "")),
                body
            );
        }
    }
}

#[test]
fn claude_enabled_summary_selects_valid_model_thinking_mode() {
    let adaptive = apply_summary_config_for_model(
        br#"{"model":"claude-opus-5","max_tokens":32000}"#,
        "claude",
        "claude-opus-5",
        &config(SummaryMode::Enabled, ""),
    );
    assert_eq!(
        json_path(&adaptive, "thinking.type"),
        Some(Value::String("adaptive".into()))
    );
    assert_eq!(
        json_path(&adaptive, "thinking.display"),
        Some(Value::String("summarized".into()))
    );

    let manual = apply_summary_config_for_model(
        br#"{"model":"claude-haiku-4-5-20251001","max_tokens":32000}"#,
        "claude",
        "claude-haiku-4-5-20251001",
        &config(SummaryMode::Enabled, ""),
    );
    assert_eq!(
        json_path(&manual, "thinking.type"),
        Some(Value::String("enabled".into()))
    );
    assert_eq!(
        json_path(&manual, "thinking.budget_tokens"),
        Some(Value::Number(1024.into()))
    );
    assert_eq!(
        json_path(&manual, "thinking.display"),
        Some(Value::String("summarized".into()))
    );
}

#[test]
fn resolved_manual_model_respects_max_tokens_and_strip_helper() {
    static NO_LEVELS: &[&str] = &[];
    let model = ModelInfo {
        id: "manual-model",
        provider_type: "claude",
        user_defined: false,
        max_completion_tokens: 0,
        thinking: Some(ThinkingSupport {
            min: Some(2048),
            max: Some(20_000),
            zero_allowed: true,
            dynamic_allowed: false,
            levels: NO_LEVELS,
        }),
    };
    let unchanged = apply_summary_config_for_resolved_model(
        br#"{"max_tokens":2048}"#,
        "claude",
        "manual-model",
        Some(&model),
        &config(SummaryMode::Enabled, ""),
    );
    assert_eq!(json_path(&unchanged, "thinking"), None);

    let stripped = strip_inferred_claude_summary_activation(
        br#"{"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"high"},"model":"manual-model"}"#,
        Some(&model),
    );
    assert_eq!(json_path(&stripped, "thinking"), None);
    assert_eq!(json_path(&stripped, "output_config"), None);
    assert_eq!(
        json_path(&stripped, "model"),
        Some(Value::String("manual-model".into()))
    );
}

#[test]
fn disabled_summary_does_not_enable_claude_thinking() {
    for model in ["claude-opus-5", "claude-haiku-4-5-20251001"] {
        let body = format!(r#"{{"model":"{model}","max_tokens":32000}}"#);
        let output = apply_summary_config_for_model(
            body.as_bytes(),
            "claude",
            model,
            &config(SummaryMode::Disabled, ""),
        );
        assert_eq!(json_path(&output, "thinking"), None);
    }
}

#[test]
fn responses_normalization_and_disable_cleanup_match_upstream() {
    let normalized = apply_summary_config(
        br#"{"reasoning":{"generate_summary":"detailed"}}"#,
        "openai-response",
        &config(SummaryMode::Enabled, "detailed"),
    );
    assert_eq!(
        json_path(&normalized, "reasoning.summary"),
        Some(Value::String("detailed".into()))
    );
    assert_eq!(json_path(&normalized, "reasoning.generate_summary"), None);

    let disabled = apply_summary_config(
        br#"{"reasoning":{"effort":"high","summary":"auto"}}"#,
        "openai-response",
        &config(SummaryMode::Disabled, ""),
    );
    assert_eq!(json_path(&disabled, "reasoning.summary"), None);
    assert_eq!(
        json_path(&disabled, "reasoning.effort"),
        Some(Value::String("high".into()))
    );

    let empty = apply_summary_config(
        br#"{"model":"gpt-5.4","reasoning":{"summary":"auto"}}"#,
        "openai-response",
        &config(SummaryMode::Disabled, ""),
    );
    assert_eq!(json_path(&empty, "reasoning"), None);
}

#[test]
fn invalid_unsupported_and_unspecified_inputs_are_byte_preserving() {
    for (body, format, mode) in [
        (br#"not json"#.as_slice(), "claude", SummaryMode::Enabled),
        (br#"{"x": 1}"#.as_slice(), "unknown", SummaryMode::Enabled),
        (
            br#"{"thinking":{"type":"adaptive"}}"#.as_slice(),
            "claude",
            SummaryMode::Unspecified,
        ),
    ] {
        assert_eq!(apply_summary_config(body, format, &config(mode, "")), body);
    }
}

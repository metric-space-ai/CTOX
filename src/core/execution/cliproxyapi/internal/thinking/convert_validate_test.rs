// ref: internal/thinking/{convert,validate}.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::registry::{ModelInfo, ThinkingSupport};

use super::convert::detect_model_capability;
use super::*;

const NO_LEVELS: &[&str] = &[];
const STANDARD_LEVELS: &[&str] = &["minimal", "low", "medium", "high"];
const SUBSET_LEVELS: &[&str] = &["low", "high"];
const MAX_LEVELS: &[&str] = &["low", "medium", "high", "max"];
const MESSY_LEVELS: &[&str] = &[" LOW ", "HIGH"];

fn model(
    id: &'static str,
    provider_type: &'static str,
    min: Option<u64>,
    max: Option<u64>,
    zero_allowed: bool,
    dynamic_allowed: bool,
    levels: &'static [&'static str],
) -> ModelInfo {
    ModelInfo {
        id,
        provider_type,
        user_defined: false,
        max_completion_tokens: 0,
        thinking: Some(ThinkingSupport {
            min,
            max,
            zero_allowed,
            dynamic_allowed,
            levels,
        }),
    }
}

fn config(mode: ThinkingMode, budget: isize, level: &str) -> ThinkingConfig {
    ThinkingConfig {
        mode,
        budget,
        level: ThinkingLevel::new(level),
    }
}

#[test]
fn level_to_budget_mapping_is_case_insensitive_but_not_trimmed() {
    let cases = [
        ("none", 0),
        ("auto", -1),
        ("minimal", 512),
        ("low", 1024),
        ("medium", 8192),
        ("high", 24_576),
        ("xhigh", 32_768),
        ("max", 128_000),
    ];
    for (level, budget) in cases {
        assert_eq!(convert_level_to_budget(level), Some(budget));
        assert_eq!(
            convert_level_to_budget(&level.to_ascii_uppercase()),
            Some(budget)
        );
    }
    assert_eq!(convert_level_to_budget(" high "), None);
    assert_eq!(convert_level_to_budget("ultra"), None);
}

#[test]
fn budget_to_level_mapping_covers_every_threshold_boundary() {
    let cases = [
        (-2, None),
        (-1, Some("auto")),
        (0, Some("none")),
        (1, Some("minimal")),
        (512, Some("minimal")),
        (513, Some("low")),
        (1024, Some("low")),
        (1025, Some("medium")),
        (8192, Some("medium")),
        (8193, Some("high")),
        (24_576, Some("high")),
        (24_577, Some("xhigh")),
        (isize::MAX, Some("xhigh")),
    ];
    for (budget, expected) in cases {
        assert_eq!(
            convert_budget_to_level(budget)
                .as_ref()
                .map(ThinkingLevel::as_str),
            expected
        );
    }
}

#[test]
fn level_membership_and_claude_effort_match_upstream_normalization() {
    assert_eq!(ModelCapability::from(-1), ModelCapability::Unknown);
    assert_eq!(ModelCapability::from(9).as_i32(), 9);
    assert!(has_level(MESSY_LEVELS, "low"));
    assert!(has_level(MESSY_LEVELS, "HIGH"));
    assert!(!has_level(MESSY_LEVELS, "medium"));

    assert_eq!(map_to_claude_effort(" minimal ", false), Some("low"));
    assert_eq!(map_to_claude_effort("LOW", false), Some("low"));
    assert_eq!(map_to_claude_effort("medium", false), Some("medium"));
    assert_eq!(map_to_claude_effort("high", false), Some("high"));
    assert_eq!(map_to_claude_effort("xhigh", false), Some("high"));
    assert_eq!(map_to_claude_effort("xhigh", true), Some("max"));
    assert_eq!(map_to_claude_effort("max", true), Some("max"));
    assert_eq!(map_to_claude_effort("auto", false), Some("high"));
    assert_eq!(map_to_claude_effort("none", true), None);
    assert_eq!(map_to_claude_effort("", true), None);

    assert!(is_budget_capable_provider("gemini"));
    assert!(is_budget_capable_provider("antigravity"));
    assert!(is_budget_capable_provider("claude"));
    assert!(!is_budget_capable_provider("Claude"));
    assert!(!is_budget_capable_provider("openai"));
}

#[test]
fn capability_detection_distinguishes_nil_empty_budget_level_and_hybrid() {
    let unsupported = ModelInfo {
        id: "unsupported",
        provider_type: "test",
        user_defined: false,
        max_completion_tokens: 0,
        thinking: None,
    };
    let empty = model("empty", "test", None, None, false, false, NO_LEVELS);
    let budget = model(
        "budget",
        "test",
        Some(0),
        Some(20_000),
        false,
        false,
        NO_LEVELS,
    );
    let level = model("level", "test", None, None, false, false, STANDARD_LEVELS);
    let hybrid = model(
        "hybrid",
        "test",
        Some(128),
        Some(20_000),
        false,
        false,
        STANDARD_LEVELS,
    );

    assert_eq!(detect_model_capability(None), ModelCapability::Unknown);
    assert_eq!(
        detect_model_capability(Some(&unsupported)),
        ModelCapability::None
    );
    assert_eq!(detect_model_capability(Some(&empty)), ModelCapability::None);
    assert_eq!(
        detect_model_capability(Some(&budget)),
        ModelCapability::BudgetOnly
    );
    assert_eq!(
        detect_model_capability(Some(&level)),
        ModelCapability::LevelOnly
    );
    assert_eq!(
        detect_model_capability(Some(&hybrid)),
        ModelCapability::Hybrid
    );
}

#[test]
fn model_without_thinking_rejects_every_mode_except_none() {
    let unsupported = ModelInfo {
        id: "no-thinking-model",
        provider_type: "openai",
        user_defined: false,
        max_completion_tokens: 0,
        thinking: None,
    };
    let error = validate_config(
        config(ThinkingMode::Level, 0, "high"),
        Some(&unsupported),
        "openai",
        "openai",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::ThinkingNotSupported);
    assert_eq!(error.message, "thinking not supported for this model");
    assert_eq!(error.model, "no-thinking-model");

    let disabled = config(ThinkingMode::None, 73, "ignored");
    assert_eq!(
        validate_config(disabled.clone(), None, "", "openai", false).unwrap(),
        disabled
    );
}

#[test]
fn budget_only_model_converts_levels_then_clamps_derived_budget() {
    let budget_model = model(
        "budget-model",
        "gemini",
        Some(1024),
        Some(20_000),
        false,
        false,
        NO_LEVELS,
    );
    let actual = validate_config(
        config(ThinkingMode::Level, 0, "high"),
        Some(&budget_model),
        "openai",
        "gemini",
        false,
    )
    .unwrap();
    assert_eq!(actual, config(ThinkingMode::Budget, 20_000, ""));

    let error = validate_config(
        config(ThinkingMode::Level, 0, "ultra"),
        Some(&budget_model),
        "openai",
        "gemini",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::UnknownLevel);
    assert_eq!(error.message, "unknown level: ultra");
}

#[test]
fn level_only_model_converts_budget_and_prefers_lower_level_on_tie() {
    let level_model = model(
        "level-subset-model",
        "openai",
        None,
        None,
        false,
        false,
        SUBSET_LEVELS,
    );
    let actual = validate_config(
        config(ThinkingMode::Budget, 8192, ""),
        Some(&level_model),
        "claude",
        "openai",
        false,
    )
    .unwrap();
    assert_eq!(actual, config(ThinkingMode::Level, 0, "low"));

    let error = validate_config(
        config(ThinkingMode::Budget, -2, ""),
        Some(&level_model),
        "claude",
        "openai",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::UnknownLevel);
    assert_eq!(
        error.message,
        "budget -2 cannot be converted to a valid level"
    );
}

#[test]
fn hybrid_model_preserves_supported_original_representation() {
    let hybrid = model(
        "hybrid-model",
        "gemini",
        Some(128),
        Some(32_768),
        true,
        true,
        SUBSET_LEVELS,
    );
    assert_eq!(
        validate_config(
            config(ThinkingMode::Budget, 8192, ""),
            Some(&hybrid),
            "gemini",
            "gemini",
            false,
        )
        .unwrap(),
        config(ThinkingMode::Budget, 8192, "")
    );
    assert_eq!(
        validate_config(
            config(ThinkingMode::Level, 0, "high"),
            Some(&hybrid),
            "gemini",
            "gemini",
            false,
        )
        .unwrap(),
        config(ThinkingMode::Level, 0, "high")
    );
}

#[test]
fn unsupported_level_is_strict_in_family_and_clamped_across_families() {
    let level_model = model(
        "level-model",
        "openai",
        None,
        None,
        false,
        false,
        STANDARD_LEVELS,
    );
    let error = validate_config(
        config(ThinkingMode::Level, 0, "XHIGH"),
        Some(&level_model),
        "codex",
        "openai-response",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::LevelNotSupported);
    assert_eq!(
        error.message,
        "level \"xhigh\" not supported, valid levels: minimal, low, medium, high"
    );

    assert_eq!(
        validate_config(
            config(ThinkingMode::Level, 0, "xhigh"),
            Some(&level_model),
            "claude",
            "openai",
            false,
        )
        .unwrap(),
        config(ThinkingMode::Level, 0, "high")
    );
}

#[test]
fn wire_protocol_reuse_detects_model_family_mismatch_and_clamps() {
    let kimi_over_claude = model(
        "kimi-level-model",
        "kimi",
        None,
        None,
        false,
        false,
        STANDARD_LEVELS,
    );
    assert_eq!(
        validate_config(
            config(ThinkingMode::Level, 0, "max"),
            Some(&kimi_over_claude),
            "claude",
            "claude",
            false,
        )
        .unwrap(),
        config(ThinkingMode::Level, 0, "high")
    );
}

#[test]
fn strict_body_budget_errors_while_suffix_and_cross_family_budgets_clamp() {
    let budget_model = model(
        "budget-model",
        "gemini",
        Some(128),
        Some(20_000),
        false,
        true,
        NO_LEVELS,
    );
    let error = validate_config(
        config(ThinkingMode::Budget, 64_000, ""),
        Some(&budget_model),
        "antigravity",
        "gemini",
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::BudgetOutOfRange);
    assert_eq!(error.message, "budget 64000 out of range [128,20000]");

    for (from_format, from_suffix) in [("gemini", true), ("openai", false)] {
        assert_eq!(
            validate_config(
                config(ThinkingMode::Budget, 64_000, ""),
                Some(&budget_model),
                from_format,
                "gemini",
                from_suffix,
            )
            .unwrap(),
            config(ThinkingMode::Budget, 20_000, "")
        );
    }
}

#[test]
fn auto_falls_back_to_midrange_or_nearest_supported_level() {
    let budget_model = model(
        "budget-model",
        "claude",
        Some(1024),
        Some(20_000),
        false,
        false,
        NO_LEVELS,
    );
    assert_eq!(
        validate_config(
            config(ThinkingMode::Auto, -1, ""),
            Some(&budget_model),
            "openai",
            "claude",
            false,
        )
        .unwrap(),
        config(ThinkingMode::Budget, 10_512, "")
    );

    let level_subset = model(
        "level-subset",
        "openai",
        None,
        None,
        false,
        false,
        SUBSET_LEVELS,
    );
    assert_eq!(
        validate_config(
            config(ThinkingMode::Auto, -1, ""),
            Some(&level_subset),
            "claude",
            "openai",
            false,
        )
        .unwrap(),
        config(ThinkingMode::Level, 0, "low")
    );

    let dynamic = model(
        "dynamic",
        "gemini",
        Some(128),
        Some(20_000),
        true,
        true,
        NO_LEVELS,
    );
    assert_eq!(
        validate_config(
            config(ThinkingMode::Auto, -1, ""),
            Some(&dynamic),
            "openai",
            "gemini",
            false,
        )
        .unwrap(),
        config(ThinkingMode::Auto, -1, "")
    );
}

#[test]
fn none_is_explicit_for_claude_and_falls_back_for_non_disableable_levels() {
    let level_model = model(
        "level-model",
        "openai",
        None,
        None,
        false,
        false,
        MAX_LEVELS,
    );
    assert_eq!(
        validate_config(
            config(ThinkingMode::None, 0, "ignored"),
            Some(&level_model),
            "openai",
            "claude",
            false,
        )
        .unwrap(),
        config(ThinkingMode::None, 0, "")
    );
    assert_eq!(
        validate_config(
            config(ThinkingMode::Level, 0, "none"),
            Some(&level_model),
            "claude",
            "openai",
            false,
        )
        .unwrap(),
        config(ThinkingMode::None, 0, "low")
    );
}

#[test]
fn supported_level_matching_trims_registry_values_without_rewriting_input() {
    let messy = model(
        "messy-levels",
        "openai",
        None,
        None,
        true,
        false,
        MESSY_LEVELS,
    );
    let actual = validate_config(
        config(ThinkingMode::Level, 0, "LoW"),
        Some(&messy),
        "openai",
        "openai",
        false,
    )
    .unwrap();
    assert_eq!(actual.level.as_str(), "LoW");
}

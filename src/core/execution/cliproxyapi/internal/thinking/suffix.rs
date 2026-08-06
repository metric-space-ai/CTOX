// ref: internal/thinking/suffix.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::types::{
    SuffixResult, ThinkingLevel, ThinkingMode, LEVEL_HIGH, LEVEL_LOW, LEVEL_MAX, LEVEL_MEDIUM,
    LEVEL_MINIMAL, LEVEL_XHIGH,
};

/// Extracts the final parenthesized suffix without interpreting its contents.
pub fn parse_suffix(model: &str) -> SuffixResult {
    let Some(last_open) = model.rfind('(') else {
        return SuffixResult {
            model_name: model.to_owned(),
            ..SuffixResult::default()
        };
    };
    if !model.ends_with(')') {
        return SuffixResult {
            model_name: model.to_owned(),
            ..SuffixResult::default()
        };
    }

    SuffixResult {
        model_name: model[..last_open].to_owned(),
        has_suffix: true,
        raw_suffix: model[last_open + 1..model.len() - 1].to_owned(),
    }
}

/// Parses a non-negative native-width budget. Leading zeros are accepted.
pub fn parse_numeric_suffix(raw_suffix: &str) -> Option<isize> {
    if raw_suffix.is_empty() {
        return None;
    }
    raw_suffix.parse::<isize>().ok().filter(|value| *value >= 0)
}

/// Parses the case-insensitive `none`, `auto`, and `-1` special values.
pub fn parse_special_suffix(raw_suffix: &str) -> Option<ThinkingMode> {
    if raw_suffix.eq_ignore_ascii_case("none") {
        Some(ThinkingMode::None)
    } else if raw_suffix.eq_ignore_ascii_case("auto") || raw_suffix == "-1" {
        Some(ThinkingMode::Auto)
    } else {
        None
    }
}

/// Parses one of the six discrete effort levels, case-insensitively.
pub fn parse_level_suffix(raw_suffix: &str) -> Option<ThinkingLevel> {
    let level = if raw_suffix.eq_ignore_ascii_case(LEVEL_MINIMAL) {
        LEVEL_MINIMAL
    } else if raw_suffix.eq_ignore_ascii_case(LEVEL_LOW) {
        LEVEL_LOW
    } else if raw_suffix.eq_ignore_ascii_case(LEVEL_MEDIUM) {
        LEVEL_MEDIUM
    } else if raw_suffix.eq_ignore_ascii_case(LEVEL_HIGH) {
        LEVEL_HIGH
    } else if raw_suffix.eq_ignore_ascii_case(LEVEL_XHIGH) {
        LEVEL_XHIGH
    } else if raw_suffix.eq_ignore_ascii_case(LEVEL_MAX) {
        LEVEL_MAX
    } else {
        return None;
    };
    Some(ThinkingLevel::new(level))
}

// ref: internal/runtime/executor/helps/thinking.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::thinking::{
    apply_summary_config_for_model, extract_explicit_summary_config, extract_summary_config,
    SummaryConfig, SummaryMode, ThinkingEngine, ThinkingError, ThinkingRequest,
};
use crate::sdk::translator::{Format, Registry};

/// Preserves summary visibility from the original client payload while
/// applying thinking configuration to its translated target payload.
///
/// Upstream reaches the thinking and translator registries through package
/// globals. CTOX injects both instance-owned authorities explicitly.
#[allow(clippy::too_many_arguments)]
pub fn apply_thinking_with_source_payload(
    engine: &ThinkingEngine,
    registry: &Registry,
    body: &[u8],
    current_source_payload: &[u8],
    original_source_payload: &[u8],
    model: &str,
    from_format: &str,
    to_format: &str,
    provider_key: &str,
) -> Result<Vec<u8>, ThinkingError> {
    let summary = translated_request_summary_config(
        registry,
        body,
        current_source_payload,
        original_source_payload,
        model,
        from_format,
        to_format,
    );
    engine.apply_thinking_with_summary(
        ThinkingRequest {
            body,
            model,
            from_format,
            to_format,
            provider_key,
        },
        &summary,
    )
}

/// Determines which summary intent remains authoritative after translation and
/// plugin normalization.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn translated_request_summary_config(
    registry: &Registry,
    body: &[u8],
    current_source_payload: &[u8],
    original_source_payload: &[u8],
    model: &str,
    from_format: &str,
    to_format: &str,
) -> SummaryConfig {
    let from_format = from_format.trim().to_ascii_lowercase();
    let to_format = to_format.trim().to_ascii_lowercase();

    let target_summary = if from_format == to_format {
        extract_summary_config(body, &to_format)
    } else {
        extract_explicit_summary_config(body, &to_format)
    };
    if target_summary.mode != SummaryMode::Unspecified {
        return target_summary;
    }

    let current_summary = extract_summary_config(current_source_payload, &from_format);
    let original_summary = extract_summary_config(original_source_payload, &from_format);
    if current_summary.mode == SummaryMode::Unspecified {
        return original_summary;
    }

    let from = Format::from(from_format.as_str());
    let to = Format::from(to_format.as_str());
    if !registry.has_request_transformer(&from, &to) {
        return SummaryConfig::default();
    }

    let candidate = apply_summary_config_for_model(body, &to_format, model, &current_summary);
    if extract_explicit_summary_config(&candidate, &to_format).mode != SummaryMode::Unspecified {
        return SummaryConfig::default();
    }

    current_summary
}

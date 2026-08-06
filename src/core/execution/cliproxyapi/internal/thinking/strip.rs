// ref: internal/thinking/strip.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::json::{remove_empty_object, remove_path, serialize_if_changed};

/// Removes provider-specific thinking fields, preserving the original bytes on
/// invalid input, unknown providers, and semantic no-ops.
pub fn strip_thinking_config(body: &[u8], provider: &str) -> Vec<u8> {
    if body.is_empty() {
        return body.to_vec();
    }
    let Ok(mut document) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    let original = document.clone();

    let paths: &[&str] = match provider {
        "claude" => &["thinking", "output_config.effort"],
        "gemini" => &["generationConfig.thinkingConfig"],
        "antigravity" => &["request.generationConfig.thinkingConfig"],
        "interactions" => &[
            "generation_config.thinking_level",
            "generation_config.thinkingLevel",
            "generation_config.thinking_budget",
            "generation_config.thinkingBudget",
            "generation_config.thinking_summaries",
            "generation_config.thinkingSummaries",
            "generation_config.thinking_config",
            "generation_config.thinkingConfig",
        ],
        "openai" => &["reasoning_effort", "reasoning"],
        "kimi" => &["reasoning_effort", "thinking"],
        "codex" | "xai" => &["reasoning"],
        _ => return body.to_vec(),
    };

    for path in paths {
        remove_path(&mut document, path);
    }
    if provider == "claude" {
        remove_empty_object(&mut document, "output_config");
    }
    serialize_if_changed(body, &original, &document)
}

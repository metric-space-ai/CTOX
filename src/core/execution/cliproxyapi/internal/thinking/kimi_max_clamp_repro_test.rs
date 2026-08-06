// ref: internal/thinking/kimi_max_clamp_repro_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use serde_json::Value;

use crate::internal::registry::{ModelInfo, ThinkingSupport};

use super::*;

const KIMI_LEVELS: &[&str] = &["low", "medium", "high"];

struct KimiResolver;

impl ModelInfoResolver for KimiResolver {
    fn lookup_model_info(&self, model: &str, _provider: &str) -> Option<ModelInfo> {
        (model == "kimi-k2.5").then_some(ModelInfo {
            id: "kimi-k2.5",
            provider_type: "kimi",
            user_defined: false,
            max_completion_tokens: 0,
            thinking: Some(ThinkingSupport {
                min: None,
                max: None,
                zero_allowed: false,
                dynamic_allowed: false,
                levels: KIMI_LEVELS,
            }),
        })
    }
}

fn path(body: &[u8], candidate: &str) -> Option<Value> {
    let document = serde_json::from_slice::<Value>(body).ok()?;
    candidate
        .split('.')
        .try_fold(&document, |value, segment| value.get(segment))
        .cloned()
}

#[test]
fn kimi_claude_messages_max_clamps_to_high() {
    let engine = ThinkingEngine::new(Arc::new(KimiResolver));
    let output = engine
        .apply_thinking(ThinkingRequest {
            body: br#"{"model":"kimi-k2.5","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"adaptive"},"output_config":{"effort":"max"}}"#,
            model: "kimi-k2.5",
            from_format: "claude",
            to_format: "claude",
            provider_key: "claude",
        })
        .unwrap();
    assert_eq!(
        path(&output, "thinking.type"),
        Some(Value::String("adaptive".into()))
    );
    assert_eq!(
        path(&output, "output_config.effort"),
        Some(Value::String("high".into()))
    );
}

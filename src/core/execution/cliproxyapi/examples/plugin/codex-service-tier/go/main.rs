// ref: examples/plugin/codex-service-tier/go/main.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use super::core::{registration, ExampleRegistration};
use serde_json::Value;
pub fn example() -> ExampleRegistration {
    registration("example-codex-service-tier", &["request_normalizer"])
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ServiceTier {
    fast: bool,
}
impl ServiceTier {
    pub fn configured(fast: bool) -> Self {
        Self { fast }
    }
    pub fn normalize(&self, format: &str, model: &str, mut body: Value) -> Value {
        if self.fast && format.eq_ignore_ascii_case("codex") && model == "gpt-5.5" {
            if let Some(object) = body.as_object_mut() {
                object.insert("service_tier".into(), Value::String("priority".into()));
            }
        }
        body
    }
}
#[test]
fn priority_is_narrowly_scoped() {
    let tier = ServiceTier::configured(true);
    assert_eq!(
        tier.normalize("codex", "gpt-5.5", serde_json::json!({}))["service_tier"],
        "priority"
    );
    assert!(tier
        .normalize("claude", "gpt-5.5", serde_json::json!({}))
        .get("service_tier")
        .is_none());
}

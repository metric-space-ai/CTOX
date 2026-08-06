// ref: sdk/cliproxy/auth/conductor_oauth_alias_suspension_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;
use crate::sdk::cliproxy::executor::Options;
use crate::sdk::cliproxy::usage::UsageContext;
use std::sync::Arc;

struct Store(Vec<CooldownStateRecord>);
impl CooldownStateStore for Store {
    fn load(&self) -> Result<Vec<CooldownStateRecord>, CooldownStoreError> {
        Ok(self.0.clone())
    }
    fn save(&self, _: &[CooldownStateRecord]) -> Result<(), CooldownStoreError> {
        Ok(())
    }
}

#[test]
fn resolved_alias_selection_bypasses_blocked_route_model_but_preserves_usage_alias() {
    let route = "claude-opus-4-6";
    let target = "claude-opus-4-6-thinking";
    let cooldown = CooldownStateRecord {
        provider: "antigravity".into(),
        auth_id: "a".into(),
        model: Some(route.into()),
        status: "cooling".into(),
        next_retry_after_ms: Some(9_000),
        reason: "quota".into(),
        quota: CooldownQuotaState::default(),
        last_error: None,
        updated_at_ms: 1_000,
    };
    let router = AccountRouter::new(Arc::new(Store(vec![cooldown])));
    let candidate = AccountCandidate {
        auth_id: "a".into(),
        provider: "antigravity".into(),
        priority: 0,
        weight: 1,
        websocket_enabled: false,
        supported_models: vec![target.into()],
        disabled: false,
    };
    assert_eq!(
        router
            .select("antigravity", Some(target), 2_000, &[candidate])
            .unwrap()
            .auth_id,
        "a"
    );
    let mut options = Options::default();
    options.metadata.requested_model = Some(route.into());
    let context =
        usage_context_with_requested_model_alias(UsageContext::default(), &options, target);
    assert_eq!(context.requested_model_alias(), route);
}

// ref: internal/api/handlers/management/quota_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::{Arc, Mutex};

use super::{
    management_auth_index_for_id, CooldownManagementQuotaReset, ManagementAuthenticator,
    ManagementQuotaAccount, ManagementQuotaSwitchError, ManagementQuotaSwitchSource,
    ManagementQuotaSwitches, SystemManagementAuthClock,
};
use crate::internal::api::server_management::ManagementHandler;
use crate::sdk::cliproxy::auth::{
    CooldownConductor, CooldownQuotaState, CooldownStateRecord, CooldownStateStore,
    CooldownStoreError,
};

#[derive(Default)]
struct MemoryCooldownStore(Mutex<Vec<CooldownStateRecord>>);

impl MemoryCooldownStore {
    fn records(&self) -> Vec<CooldownStateRecord> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl CooldownStateStore for MemoryCooldownStore {
    fn load(&self) -> Result<Vec<CooldownStateRecord>, CooldownStoreError> {
        Ok(self.records())
    }

    fn save(&self, records: &[CooldownStateRecord]) -> Result<(), CooldownStoreError> {
        *self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = records.to_vec();
        Ok(())
    }
}

fn cooldown(auth_id: &str, model: Option<&str>) -> CooldownStateRecord {
    CooldownStateRecord {
        provider: "claude".to_owned(),
        auth_id: auth_id.to_owned(),
        model: model.map(str::to_owned),
        status: "cooling".to_owned(),
        next_retry_after_ms: Some(10_000),
        reason: "quota".to_owned(),
        quota: CooldownQuotaState {
            exceeded: true,
            reason: "quota".to_owned(),
            next_recover_at_ms: Some(10_000),
            backoff_level: 2,
        },
        last_error: None,
        updated_at_ms: 1_000,
    }
}

fn handler(store: Arc<MemoryCooldownStore>) -> ManagementHandler {
    let conductor = Arc::new(CooldownConductor::new(store));
    let reset = CooldownManagementQuotaReset::new(
        vec![ManagementQuotaAccount {
            auth_id: "reset-auth-id".to_owned(),
        }],
        conductor,
    )
    .unwrap();
    ManagementHandler::new(Arc::new(
        ManagementAuthenticator::new(
            "management-secret",
            false,
            Arc::new(SystemManagementAuthClock),
        )
        .unwrap(),
    ))
    .attach_quota_reset_source(Arc::new(reset))
}

fn headers() -> BTreeMap<String, Vec<String>> {
    BTreeMap::from([
        (
            "X-Management-Key".to_owned(),
            vec!["management-secret".to_owned()],
        ),
        (
            "Content-Type".to_owned(),
            vec!["application/json".to_owned()],
        ),
    ])
}

#[test]
fn reset_quota_uses_public_auth_index_and_clears_account_and_model_state() {
    let store = Arc::new(MemoryCooldownStore(Mutex::new(vec![
        cooldown("reset-auth-id", None),
        cooldown("reset-auth-id", Some("claude-reset-model")),
        cooldown("other-auth-id", Some("claude-other-model")),
    ])));
    let auth_index = management_auth_index_for_id("reset-auth-id").unwrap();
    let body = serde_json::to_vec(&serde_json::json!({"auth_index": auth_index})).unwrap();

    let response = handler(store.clone()).handle(
        "POST",
        "/v0/management/reset-quota",
        &headers(),
        &body,
        IpAddr::V4(Ipv4Addr::LOCALHOST),
    );

    assert_eq!(response.status(), 200);
    let payload: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["auth_index"], auth_index);
    assert_eq!(payload["models"], serde_json::json!(["claude-reset-model"]));
    let remaining = store.records();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].auth_id, "other-auth-id");
}

#[test]
fn reset_quota_does_not_accept_auth_id_or_file_name() {
    let store = Arc::new(MemoryCooldownStore(Mutex::new(vec![cooldown(
        "reset-auth-id",
        Some("claude-reset-model"),
    )])));
    let handler = handler(store.clone());
    for (body, expected_status) in [
        (serde_json::json!({"auth_id": "reset-auth-id"}), 400),
        (serde_json::json!({"id": "reset-auth-id"}), 400),
        (
            serde_json::json!({"auth_index": "reset-auth-file.json"}),
            404,
        ),
        (serde_json::json!({"auth_index": "reset-auth-id"}), 404),
    ] {
        let response = handler.handle(
            "POST",
            "/v0/management/reset-quota",
            &headers(),
            &serde_json::to_vec(&body).unwrap(),
            IpAddr::V4(Ipv4Addr::LOCALHOST),
        );
        assert_eq!(response.status(), expected_status, "body={body}");
    }
    assert_eq!(store.records().len(), 1);
}

#[derive(Default)]
struct MemoryQuotaSwitches(Mutex<ManagementQuotaSwitches>);

impl ManagementQuotaSwitchSource for MemoryQuotaSwitches {
    fn snapshot(&self) -> Result<ManagementQuotaSwitches, ManagementQuotaSwitchError> {
        Ok(*self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()))
    }

    fn set_switch_project(&self, value: bool) -> Result<(), ManagementQuotaSwitchError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .switch_project = value;
        Ok(())
    }

    fn set_switch_preview_model(&self, value: bool) -> Result<(), ManagementQuotaSwitchError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .switch_preview_model = value;
        Ok(())
    }
}

#[test]
fn quota_fallback_switches_use_typed_get_put_and_patch_source() {
    let switches = Arc::new(MemoryQuotaSwitches::default());
    let handler = handler(Arc::new(MemoryCooldownStore::default()))
        .attach_quota_switch_source(switches.clone());
    for (method, path) in [
        ("PUT", "/v0/management/quota-exceeded/switch-project"),
        (
            "PATCH",
            "/v0/management/quota-exceeded/switch-preview-model",
        ),
    ] {
        let response = handler.handle(
            method,
            path,
            &headers(),
            br#"{"value":true}"#,
            IpAddr::V4(Ipv4Addr::LOCALHOST),
        );
        assert_eq!(response.status(), 200);
    }
    assert_eq!(
        *switches
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        ManagementQuotaSwitches {
            switch_project: true,
            switch_preview_model: true,
        }
    );
    let response = handler.handle(
        "GET",
        "/v0/management/quota-exceeded/switch-preview-model",
        &headers(),
        &[],
        IpAddr::V4(Ipv4Addr::LOCALHOST),
    );
    assert_eq!(response.status(), 200);
    let payload: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
    assert_eq!(payload["switch-preview-model"], true);
}

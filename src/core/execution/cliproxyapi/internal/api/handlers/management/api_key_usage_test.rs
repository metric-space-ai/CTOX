// ref: internal/api/handlers/management/api_key_usage_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;

use super::{
    ManagementApiKeyUsageError, ManagementApiKeyUsageRecord, ManagementApiKeyUsageSource,
    ManagementAuthenticator, ManagementRecentRequestBucket, SystemManagementAuthClock,
};
use crate::internal::api::server_management::ManagementHandler;

struct StaticApiKeyUsageSource(Vec<ManagementApiKeyUsageRecord>);

impl ManagementApiKeyUsageSource for StaticApiKeyUsageSource {
    fn snapshot(&self) -> Result<Vec<ManagementApiKeyUsageRecord>, ManagementApiKeyUsageError> {
        Ok(self.0.clone())
    }
}

fn record(
    provider: &str,
    compat_name: Option<&str>,
    account_id: &str,
    success: i64,
    failed: i64,
) -> ManagementApiKeyUsageRecord {
    ManagementApiKeyUsageRecord {
        provider: provider.to_owned(),
        compat_name: compat_name.map(str::to_owned),
        account_id: account_id.to_owned(),
        success,
        failed,
        recent_requests: vec![ManagementRecentRequestBucket {
            time: "12:00-12:15".to_owned(),
            success,
            failed,
        }],
    }
}

fn response(records: Vec<ManagementApiKeyUsageRecord>) -> Vec<u8> {
    let handler = ManagementHandler::new(Arc::new(
        ManagementAuthenticator::new(
            "management-secret",
            false,
            Arc::new(SystemManagementAuthClock),
        )
        .unwrap(),
    ))
    .attach_api_key_usage_source(Arc::new(StaticApiKeyUsageSource(records)));
    let headers = BTreeMap::from([(
        "X-Management-Key".to_owned(),
        vec!["management-secret".to_owned()],
    )]);
    let response = handler.handle(
        "GET",
        "/v0/management/api-key-usage",
        &headers,
        &[],
        IpAddr::V4(Ipv4Addr::LOCALHOST),
    );
    assert_eq!(response.status(), 200);
    response.body().to_vec()
}

#[test]
fn groups_by_provider_and_public_account_id_without_credentials() {
    let body = response(vec![
        record("codex", None, "codex-public", 1, 1),
        record("claude", None, "claude-public", 1, 0),
    ]);
    let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(payload["codex"]["codex-public"]["success"], 1);
    assert_eq!(payload["codex"]["codex-public"]["failed"], 1);
    assert_eq!(
        payload["codex"]["codex-public"]["recent_requests"][0]["failed"],
        1
    );
    assert_eq!(payload["claude"]["claude-public"]["success"], 1);
    let rendered = String::from_utf8(body).unwrap();
    assert!(!rendered.contains("api_key"));
    assert!(!rendered.contains("base_url"));
    assert!(!rendered.contains("https://"));
}

#[test]
fn groups_openai_compatible_accounts_by_normalized_compat_name() {
    let body = response(vec![record(
        "openai-compatible-vast",
        Some(" VAST "),
        "vast-public",
        1,
        0,
    )]);
    let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(payload.get("openai-compatible-vast").is_none());
    assert_eq!(payload["vast"]["vast-public"]["success"], 1);
}

#[test]
fn duplicate_public_account_records_merge_totals_and_recent_buckets() {
    let body = response(vec![
        record("codex", None, "codex-public", 2, 1),
        record("CODEX", None, "codex-public", 3, 4),
    ]);
    let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(payload["codex"]["codex-public"]["success"], 5);
    assert_eq!(payload["codex"]["codex-public"]["failed"], 5);
    assert_eq!(
        payload["codex"]["codex-public"]["recent_requests"][0]["success"],
        5
    );
}

// ref: internal/api/handlers/management/api_key_usage.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagementRecentRequestBucket {
    pub time: String,
    pub success: i64,
    pub failed: i64,
}

/// Secret-free host projection for one API-key-backed account.
///
/// Upstream keys the response by `base_url|api_key`, which serializes a live
/// credential into the HTTP response. CTOX instead requires a stable public
/// account identifier and intentionally has no API-key or base-URL field at
/// this boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagementApiKeyUsageRecord {
    pub provider: String,
    pub compat_name: Option<String>,
    pub account_id: String,
    pub success: i64,
    pub failed: i64,
    pub recent_requests: Vec<ManagementRecentRequestBucket>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementApiKeyUsageError {
    SourceUnavailable,
}

impl fmt::Display for ManagementApiKeyUsageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("API key usage unavailable")
    }
}

impl std::error::Error for ManagementApiKeyUsageError {}

pub trait ManagementApiKeyUsageSource: Send + Sync {
    fn snapshot(&self) -> Result<Vec<ManagementApiKeyUsageRecord>, ManagementApiKeyUsageError>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
struct ApiKeyUsageEntry {
    success: i64,
    failed: i64,
    recent_requests: Vec<ManagementRecentRequestBucket>,
}

pub fn api_key_usage_payload(
    records: Vec<ManagementApiKeyUsageRecord>,
) -> Result<Vec<u8>, ManagementApiKeyUsageError> {
    let mut providers: BTreeMap<String, BTreeMap<String, ApiKeyUsageEntry>> = BTreeMap::new();
    for record in records {
        let account_id = record.account_id.trim();
        if account_id.is_empty() {
            continue;
        }
        let provider = provider_key(&record);
        let entry = providers
            .entry(provider)
            .or_default()
            .entry(account_id.to_owned())
            .or_default();
        entry.success = entry.success.saturating_add(record.success);
        entry.failed = entry.failed.saturating_add(record.failed);
        merge_recent_request_buckets(&mut entry.recent_requests, record.recent_requests);
    }
    serde_json::to_vec(&providers).map_err(|_| ManagementApiKeyUsageError::SourceUnavailable)
}

fn provider_key(record: &ManagementApiKeyUsageRecord) -> String {
    let provider = record
        .compat_name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| record.provider.trim())
        .to_lowercase();
    if provider.is_empty() {
        "unknown".to_owned()
    } else {
        provider
    }
}

fn merge_recent_request_buckets(
    destination: &mut Vec<ManagementRecentRequestBucket>,
    source: Vec<ManagementRecentRequestBucket>,
) {
    if destination.is_empty() {
        *destination = source;
        return;
    }
    for (destination, source) in destination.iter_mut().zip(source) {
        destination.success = destination.success.saturating_add(source.success);
        destination.failed = destination.failed.saturating_add(source.failed);
    }
}

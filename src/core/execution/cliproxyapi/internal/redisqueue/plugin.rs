// ref: internal/redisqueue/plugin.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::internal::logging::{
    get_client_request_metadata, get_endpoint, get_request_id, get_response_status,
};
use crate::sdk::cliproxy::usage::{
    ensure_token_breakdown_for_provider, generate_enabled, Plugin, Record, TokenBreakdown,
    UsageContext, DEFAULT_SERVICE_TIER, TOKEN_ACCOUNTING_SCHEMA_VERSION,
};

use super::{UsageQueue, UsageStatisticsSwitch};

/// Explicitly wired replacement for upstream's `init`-registered global
/// plugin. Each CTOX gateway instance owns its queue and statistics policy.
pub struct UsageQueuePlugin {
    queue: Arc<UsageQueue>,
    statistics: Arc<UsageStatisticsSwitch>,
}

impl UsageQueuePlugin {
    #[must_use]
    pub fn new(queue: Arc<UsageQueue>, statistics: Arc<UsageStatisticsSwitch>) -> Self {
        Self { queue, statistics }
    }
}

impl Plugin for UsageQueuePlugin {
    fn handle_usage(&self, context: &UsageContext, record: &Record) {
        if !self.queue.enabled() || !self.statistics.enabled() {
            return;
        }

        let provider = normalized(&record.provider, "unknown");
        let executor_type = normalized(&record.executor_type, "unknown");
        let model = normalized(&record.model, "unknown");
        let alias = normalized(&record.alias, &model);
        let auth_type = normalized(&record.auth_type, "unknown");
        let reasoning_effort =
            first_non_empty(&[&record.reasoning_effort, context.reasoning_effort()]);
        let service_tier = first_non_empty(&[
            &record.service_tier,
            &record.request_service_tier,
            context.service_tier(),
        ]);
        let request = &context.request;
        let metadata = get_client_request_metadata(Some(request));
        let usage = ensure_token_breakdown_for_provider(
            record.detail.clone(),
            &record.provider,
            &record.executor_type,
        );
        let failed = record.failed || !resolve_success(get_response_status(Some(request)));
        let fail = resolve_fail(record, failed, get_response_status(Some(request)));
        let payload = QueuedUsageDetail {
            timestamp: DateTime::<Utc>::from(record.requested_at.unwrap_or_else(SystemTime::now)),
            latency_ms: duration_millis(record.latency),
            ttft_ms: duration_millis(record.ttft),
            source: record.source.clone(),
            auth_index: record.auth_index.clone(),
            access_token_sha256: record.access_token_sha256.trim().to_owned(),
            client_ip: metadata.client_ip,
            x_forwarded_for: metadata.x_forwarded_for,
            user_agent: metadata.user_agent,
            tokens: TokenStats {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                reasoning_tokens: usage.reasoning_tokens,
                cached_tokens: usage.cached_tokens,
                cache_read_tokens: usage.cache_read_tokens,
                cache_read_tokens_present: true,
                cache_creation_tokens: usage.cache_creation_tokens,
                total_tokens: usage.total_tokens,
            },
            failed,
            generate: generate_enabled(record.generate),
            fail,
            response_headers: record.response_headers.clone(),
            accounting_version: TOKEN_ACCOUNTING_SCHEMA_VERSION,
            token_breakdown: usage.token_breakdown,
            provider,
            executor_type,
            model,
            alias,
            endpoint: get_endpoint(Some(request)).trim().to_owned(),
            auth_type,
            api_key: record.api_key.trim().to_owned(),
            request_id: get_request_id(Some(request)).trim().to_owned(),
            reasoning_effort,
            service_tier: if service_tier.is_empty() {
                DEFAULT_SERVICE_TIER.to_owned()
            } else {
                service_tier
            },
            response_service_tier: record.response_service_tier.trim().to_owned(),
        };
        if let Ok(payload) = serde_json::to_vec(&payload) {
            self.queue.enqueue(&payload);
        }
    }
}

#[derive(Serialize)]
struct QueuedUsageDetail {
    timestamp: DateTime<Utc>,
    latency_ms: i64,
    ttft_ms: i64,
    source: String,
    auth_index: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    access_token_sha256: String,
    client_ip: String,
    x_forwarded_for: String,
    user_agent: String,
    tokens: TokenStats,
    failed: bool,
    generate: bool,
    fail: FailDetail,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    response_headers: BTreeMap<String, Vec<String>>,
    accounting_version: u8,
    token_breakdown: TokenBreakdown,
    provider: String,
    executor_type: String,
    model: String,
    alias: String,
    endpoint: String,
    auth_type: String,
    api_key: String,
    request_id: String,
    reasoning_effort: String,
    service_tier: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    response_service_tier: String,
}

#[derive(Serialize)]
struct TokenStats {
    input_tokens: i64,
    output_tokens: i64,
    reasoning_tokens: i64,
    cached_tokens: i64,
    cache_read_tokens: i64,
    cache_read_tokens_present: bool,
    cache_creation_tokens: i64,
    total_tokens: i64,
}

#[derive(Serialize)]
struct FailDetail {
    status_code: i32,
    body: String,
}

fn resolve_fail(record: &Record, failed: bool, response_status: i32) -> FailDetail {
    if !failed {
        return FailDetail {
            status_code: 200,
            body: String::new(),
        };
    }
    let status_code = if record.fail.status_code > 0 {
        record.fail.status_code
    } else if response_status > 0 {
        response_status
    } else {
        500
    };
    FailDetail {
        status_code,
        body: record.fail.body.trim().to_owned(),
    }
}

fn resolve_success(status: i32) -> bool {
    status == 0 || status < 400
}

fn normalized(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_owned()
    } else {
        value.to_owned()
    }
}

fn first_non_empty(values: &[&str]) -> String {
    values
        .iter()
        .map(|value| value.trim())
        .find(|value| !value.is_empty())
        .unwrap_or_default()
        .to_owned()
}

fn duration_millis(duration: Duration) -> i64 {
    duration.as_millis().min(i64::MAX as u128) as i64
}

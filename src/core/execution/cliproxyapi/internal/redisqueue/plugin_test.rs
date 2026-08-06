// ref: internal/redisqueue/plugin_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use serde_json::Value;

use crate::internal::logging::{
    set_response_status, with_client_request_metadata, with_endpoint, with_request_id,
    with_response_status_holder, ClientRequestMetadata,
};
use crate::sdk::cliproxy::usage::{
    generate_flag, Detail, Failure, Plugin, Record, TokenAccountingQuality, UsageContext,
    AUTO_SERVICE_TIER, TOKEN_ACCOUNTING_SCHEMA_VERSION,
};

use super::{UsageQueue, UsageQueuePlugin, UsageStatisticsSwitch};

fn fixture() -> (Arc<UsageQueue>, UsageQueuePlugin) {
    let queue = Arc::new(UsageQueue::new());
    queue.set_enabled(true);
    let statistics = Arc::new(UsageStatisticsSwitch::new());
    let plugin = UsageQueuePlugin::new(Arc::clone(&queue), statistics);
    (queue, plugin)
}

fn pop(queue: &UsageQueue) -> Value {
    let items = queue.pop_oldest(10);
    assert_eq!(items.len(), 1);
    serde_json::from_slice(&items[0]).unwrap()
}

#[test]
fn stable_success_payload_and_snapshot_headers() {
    let (queue, plugin) = fixture();
    let request = with_response_status_holder(Some(&with_client_request_metadata(
        Some(&with_endpoint(
            Some(&with_request_id(None, "ctx-request-id")),
            "POST /v1/chat/completions",
        )),
        ClientRequestMetadata {
            client_ip: "192.0.2.10".into(),
            x_forwarded_for: "203.0.113.5, 198.51.100.8".into(),
            user_agent: "test-client/1.0".into(),
        },
    )));
    set_response_status(Some(&request), 200);
    let context = UsageContext::from_request(request);
    plugin.handle_usage(
        &context,
        &Record {
            provider: "openai".into(),
            executor_type: "KimiExecutor".into(),
            model: "gpt-5.4".into(),
            alias: "client-gpt".into(),
            api_key: "test-key".into(),
            auth_index: "0".into(),
            access_token_sha256: "token-version-hash".into(),
            auth_type: "apikey".into(),
            source: "user@example.com".into(),
            reasoning_effort: "medium".into(),
            service_tier: "auto".into(),
            response_service_tier: "default".into(),
            generate: generate_flag(true),
            requested_at: Some(UNIX_EPOCH + Duration::from_secs(1_777_075_200)),
            latency: Duration::from_millis(1_500),
            detail: Detail {
                input_tokens: 10,
                output_tokens: 20,
                total_tokens: 30,
                ..Detail::default()
            },
            response_headers: [
                (
                    "X-Upstream-Request-Id".into(),
                    vec!["upstream-req-1".into()],
                ),
                ("Retry-After".into(), vec!["30".into()]),
            ]
            .into(),
            ..Record::default()
        },
    );
    let payload = pop(&queue);
    assert_eq!(payload["provider"], "openai");
    assert_eq!(payload["executor_type"], "KimiExecutor");
    assert_eq!(payload["model"], "gpt-5.4");
    assert_eq!(payload["alias"], "client-gpt");
    assert_eq!(payload["endpoint"], "POST /v1/chat/completions");
    assert_eq!(payload["request_id"], "ctx-request-id");
    assert_eq!(payload["access_token_sha256"], "token-version-hash");
    assert_eq!(payload["client_ip"], "192.0.2.10");
    assert_eq!(payload["reasoning_effort"], "medium");
    assert_eq!(payload["service_tier"], "auto");
    assert_eq!(payload["response_service_tier"], "default");
    assert_eq!(
        payload["accounting_version"],
        TOKEN_ACCOUNTING_SCHEMA_VERSION
    );
    assert_eq!(payload["token_breakdown"]["quality"], "complete");
    assert_eq!(payload["token_breakdown"]["total_tokens"], 30);
    assert_eq!(payload["tokens"]["cache_read_tokens_present"], true);
    assert_eq!(payload["response_headers"]["Retry-After"][0], "30");
    assert_eq!(payload["failed"], false);
    assert_eq!(payload["generate"], true);
    assert_eq!(payload["fail"]["status_code"], 200);
    assert!(payload.get("request_service_tier").is_none());
    assert!(payload.get("user_api_key").is_none());
}

#[test]
fn provider_accounting_generate_and_tier_fallbacks() {
    for (provider, expected_total) in [("openai", 130), ("gemini", 142)] {
        let (queue, plugin) = fixture();
        let context = UsageContext::default().with_service_tier(AUTO_SERVICE_TIER);
        plugin.handle_usage(
            &context,
            &Record {
                provider: provider.into(),
                model: "direct-sdk-model".into(),
                generate: generate_flag(false),
                detail: Detail {
                    input_tokens: 100,
                    output_tokens: 30,
                    reasoning_tokens: 12,
                    ..Detail::default()
                },
                ..Record::default()
            },
        );
        let payload = pop(&queue);
        assert_eq!(payload["tokens"]["total_tokens"], expected_total);
        assert_eq!(payload["token_breakdown"]["total_tokens"], expected_total);
        assert_eq!(payload["token_breakdown"]["quality"], "complete");
        assert_eq!(payload["generate"], false);
        assert_eq!(payload["service_tier"], "auto");
    }
}

#[test]
fn omitted_generate_defaults_true_and_deprecated_tier_is_accepted() {
    let (queue, plugin) = fixture();
    plugin.handle_usage(
        &UsageContext::default(),
        &Record {
            provider: "openai".into(),
            model: "gpt-5.4".into(),
            request_service_tier: "priority".into(),
            detail: Detail {
                input_tokens: 1,
                total_tokens: 1,
                ..Detail::default()
            },
            ..Record::default()
        },
    );
    let payload = pop(&queue);
    assert_eq!(payload["generate"], true);
    assert_eq!(payload["service_tier"], "priority");
    assert!(payload.get("request_service_tier").is_none());
}

#[test]
fn legacy_cached_only_usage_is_preserved() {
    let (queue, plugin) = fixture();
    plugin.handle_usage(
        &UsageContext::default(),
        &Record {
            provider: "openai".into(),
            model: "gpt-5.4".into(),
            detail: Detail {
                cached_tokens: 13,
                ..Detail::default()
            },
            ..Record::default()
        },
    );
    let payload = pop(&queue);
    assert_eq!(payload["tokens"]["cache_read_tokens"], 13);
    assert_eq!(payload["tokens"]["total_tokens"], 13);
    assert_eq!(
        payload["token_breakdown"]["quality"],
        serde_json::to_value(TokenAccountingQuality::Unclassified).unwrap()
    );
}

#[test]
fn response_status_and_record_failure_are_resolved_without_framework_context() {
    let (queue, plugin) = fixture();
    let request = with_response_status_holder(Some(&with_endpoint(None, "GET /v1/responses")));
    set_response_status(Some(&request), 500);
    plugin.handle_usage(
        &UsageContext::from_request(request),
        &Record {
            provider: "openai".into(),
            model: "gpt-5.4-mini".into(),
            fail: Failure {
                status_code: 502,
                body: " bad gateway ".into(),
            },
            ..Record::default()
        },
    );
    let payload = pop(&queue);
    assert_eq!(payload["failed"], true);
    assert_eq!(payload["fail"]["status_code"], 502);
    assert_eq!(payload["fail"]["body"], "bad gateway");
    assert_eq!(payload["endpoint"], "GET /v1/responses");
}

#[test]
fn disabled_queue_or_statistics_discards_usage() {
    let queue = Arc::new(UsageQueue::new());
    let statistics = Arc::new(UsageStatisticsSwitch::new());
    let plugin = UsageQueuePlugin::new(Arc::clone(&queue), Arc::clone(&statistics));
    plugin.handle_usage(&UsageContext::default(), &Record::default());
    assert!(queue.pop_oldest(1).is_empty());

    queue.set_enabled(true);
    statistics.set_enabled(false);
    plugin.handle_usage(&UsageContext::default(), &Record::default());
    assert!(queue.pop_oldest(1).is_empty());
}

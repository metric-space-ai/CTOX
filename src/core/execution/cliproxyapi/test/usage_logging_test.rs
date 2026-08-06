// ref: test/usage_logging_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use serde_json::Value;

use crate::internal::redisqueue::{UsageQueue, UsageQueuePlugin, UsageStatisticsSwitch};
use crate::internal::runtime::executor::gemini_executor::{GeminiExecutor, GeminiExecutorConfig};
use crate::internal::translator::register_all;
use crate::sdk::cliproxy::usage::Manager;
use crate::sdk::pluginapi::{
    ExecutorRequest, HostHttpClient, HttpRequest, HttpResponse, HttpStreamResponse, PluginFuture,
    ProviderExecutor,
};
use crate::sdk::translator::Registry;

struct ZeroUsageClient;

impl HostHttpClient for ZeroUsageClient {
    fn execute<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpResponse> {
        Box::pin(async move {
            assert_eq!(
                request.url,
                "http://upstream/v1beta/models/gemini-zero-usage:generateContent"
            );
            Ok(HttpResponse {
                status_code: 200,
                body: br#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0}}"#.to_vec(),
                ..HttpResponse::default()
            })
        })
    }

    fn execute_stream<'a>(&'a self, _request: HttpRequest) -> PluginFuture<'a, HttpStreamResponse> {
        Box::pin(async { panic!("stream not used") })
    }
}

#[tokio::test]
async fn gemini_executor_records_successful_zero_usage_in_instance_queue() {
    let queue = Arc::new(UsageQueue::new());
    queue.set_enabled(true);
    let statistics = Arc::new(UsageStatisticsSwitch::new());
    let manager = Arc::new(Manager::new(16));
    manager.register(Arc::new(UsageQueuePlugin::new(
        Arc::clone(&queue),
        statistics,
    )));

    let registry = Arc::new(Registry::new());
    register_all(&registry);
    let executor = GeminiExecutor::new(Arc::new(GeminiExecutorConfig::default()), registry)
        .with_usage_manager(Arc::clone(&manager));
    executor
        .execute(ExecutorRequest {
            model: "gemini-zero-usage".into(),
            format: "gemini".into(),
            source_format: "gemini".into(),
            payload: br#"{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}"#.to_vec(),
            auth_attributes: [("base_url".into(), "http://upstream".into())].into(),
            http_client: Some(Arc::new(ZeroUsageClient)),
            ..ExecutorRequest::default()
        })
        .await
        .unwrap();
    manager.stop();

    let payloads = queue.pop_oldest(10);
    assert_eq!(payloads.len(), 1);
    let payload: Value = serde_json::from_slice(&payloads[0]).unwrap();
    assert_eq!(payload["provider"], "gemini");
    assert_eq!(payload["model"], "gemini-zero-usage");
    assert_eq!(payload["failed"], false);
    assert_eq!(payload["tokens"]["total_tokens"], 0);
}

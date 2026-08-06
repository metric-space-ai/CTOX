// ref: internal/runtime/executor/aistudio_executor_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::mpsc;

use crate::sdk::pluginapi::{
    ExecutorRequest, HostHttpClient, HttpRequest, HttpResponse, HttpStreamResponse, PluginFuture,
    ProviderExecutor,
};
use crate::sdk::translator::Registry;

use super::aistudio_executor::{ensure_colon_spaced_json, AiStudioExecutor};

#[derive(Default)]
struct Relay {
    requests: Mutex<Vec<HttpRequest>>,
    response: Mutex<HttpResponse>,
}
impl Relay {
    fn responding(body: &[u8]) -> Arc<Self> {
        Arc::new(Self {
            response: Mutex::new(HttpResponse {
                status_code: 200,
                body: body.to_vec(),
                ..HttpResponse::default()
            }),
            ..Self::default()
        })
    }
}
impl HostHttpClient for Relay {
    fn execute<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpResponse> {
        Box::pin(async move {
            self.requests.lock().unwrap().push(request);
            Ok(self.response.lock().unwrap().clone())
        })
    }
    fn execute_stream<'a>(&'a self, _request: HttpRequest) -> PluginFuture<'a, HttpStreamResponse> {
        Box::pin(async move {
            let (_sender, receiver) = mpsc::channel(1);
            Ok(HttpStreamResponse {
                status_code: 200,
                headers: BTreeMap::new(),
                chunks: receiver,
            })
        })
    }
}

fn request() -> ExecutorRequest {
    ExecutorRequest {
        auth_id: "a1".into(), model: "gemini-2.5-pro".into(), source_format: "gemini".into(),
        payload: br#"{"contents":[],"reasoning":{"summary":"detailed"},"generationConfig":{"maxOutputTokens":100,"responseMimeType":"application/json","responseJsonSchema":{}}}"#.to_vec(),
        auth_attributes: BTreeMap::from([("header:X-Test".into(), "yes".into())]),
        ..ExecutorRequest::default()
    }
}

#[tokio::test]
async fn translate_request_preserves_summary_source_and_removes_studio_unsupported_fields() {
    let relay = Relay::responding(br#"{"candidates":[]}"#);
    let executor = AiStudioExecutor::new("AIStudio", Arc::new(Registry::new()), relay.clone());
    executor.execute(request()).await.unwrap();
    let upstream = relay.requests.lock().unwrap()[0].clone();
    let body: Value = serde_json::from_slice(&upstream.body).unwrap();
    assert_eq!(
        body.pointer("/reasoning/summary").and_then(Value::as_str),
        Some("detailed")
    );
    assert!(body.pointer("/generationConfig/maxOutputTokens").is_none());
    assert!(body.pointer("/generationConfig/responseMimeType").is_none());
    assert_eq!(upstream.headers.get("X-Test").unwrap(), &["yes"]);
}

#[tokio::test]
async fn execute_dispatches_immediately_to_relay_and_returns_response() {
    let relay = Relay::responding(br#"{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}"#);
    let executor = AiStudioExecutor::new("aistudio", Arc::new(Registry::new()), relay.clone());
    let response = executor.execute(request()).await.unwrap();
    assert_eq!(relay.requests.lock().unwrap().len(), 1);
    assert_eq!(
        serde_json::from_slice::<Value>(&response.payload)
            .unwrap()
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(Value::as_str),
        Some("ok")
    );
}

#[test]
fn endpoint_rules_match_generate_stream_count_and_alt() {
    let relay = Relay::responding(b"{}");
    let executor = AiStudioExecutor::new("aistudio", Arc::new(Registry::new()), relay);
    assert!(executor
        .build_endpoint("m", "generateContent", "")
        .ends_with("/v1beta/models/m:generateContent"));
    assert!(executor
        .build_endpoint("m", "streamGenerateContent", "")
        .ends_with("streamGenerateContent?alt=sse"));
    assert!(executor
        .build_endpoint("m", "streamGenerateContent", "json")
        .ends_with("streamGenerateContent?$alt=json"));
    assert!(executor
        .build_endpoint("m", "countTokens", "ignored")
        .ends_with("m:countTokens"));
}

#[test]
fn colon_spacing_is_stable_and_non_json_is_untouched() {
    assert_eq!(
        ensure_colon_spaced_json(br#"{"a":{"b":1},"text":"x:y"}"#),
        br#"{"a": {"b": 1},"text": "x:y"}"#
    );
    assert_eq!(ensure_colon_spaced_json(b"not json"), b"not json");
}

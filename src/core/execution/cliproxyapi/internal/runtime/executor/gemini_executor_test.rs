// ref: internal/runtime/executor/gemini_executor_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use crate::internal::translator::register_all;
use crate::sdk::pluginapi::{
    ExecutorRequest, HostHttpClient, HttpRequest, HttpResponse, HttpStreamChunk,
    HttpStreamResponse, PluginFuture, ProviderExecutor,
};
use crate::sdk::translator::{Format, Registry, TranslationContext};

use super::gemini_executor::{
    apply_interactions_request_headers, apply_interactions_revision_header,
    cap_gemini_max_output_tokens, fix_gemini_image_aspect_ratio, gemini_interactions_sse_done,
    gemini_interactions_sse_payload, native_interactions_source_format, GeminiExecutor,
    GeminiExecutorConfig, GeminiPayloadRule, GEMINI_INTERACTIONS_API_REVISION,
};

#[derive(Default)]
struct MockClient {
    requests: Mutex<Vec<HttpRequest>>,
    response: Mutex<HttpResponse>,
    stream_chunks: Mutex<Vec<Vec<u8>>>,
}

impl MockClient {
    fn with_body(body: &[u8]) -> Arc<Self> {
        Arc::new(Self {
            response: Mutex::new(HttpResponse {
                status_code: 200,
                body: body.to_vec(),
                ..HttpResponse::default()
            }),
            ..Self::default()
        })
    }
    fn request(&self) -> HttpRequest {
        self.requests.lock().unwrap().last().unwrap().clone()
    }
}

impl HostHttpClient for MockClient {
    fn execute<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpResponse> {
        Box::pin(async move {
            self.requests.lock().unwrap().push(request);
            Ok(self.response.lock().unwrap().clone())
        })
    }
    fn execute_stream<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpStreamResponse> {
        Box::pin(async move {
            self.requests.lock().unwrap().push(request);
            let (sender, receiver) = mpsc::channel(16);
            let chunks = std::mem::take(&mut *self.stream_chunks.lock().unwrap());
            for payload in chunks {
                sender
                    .send(HttpStreamChunk {
                        payload,
                        error: None,
                    })
                    .await
                    .unwrap();
            }
            drop(sender);
            Ok(HttpStreamResponse {
                status_code: 200,
                headers: BTreeMap::new(),
                chunks: receiver,
            })
        })
    }
}

fn request(client: Arc<MockClient>) -> ExecutorRequest {
    ExecutorRequest {
        model: "gemini-3.1-pro-preview".into(),
        source_format: "gemini".into(),
        payload:
            br#"{"contents":[],"generationConfig":{"maxOutputTokens":500000,"temperature":0.2}}"#
                .to_vec(),
        auth_attributes: BTreeMap::from([
            ("api_key".into(), "test-key".into()),
            ("base_url".into(), "http://upstream".into()),
        ]),
        http_client: Some(client),
        ..ExecutorRequest::default()
    }
}

#[test]
fn caps_max_output_tokens_using_catalog_limit() {
    let body = br#"{"generationConfig":{"maxOutputTokens":500000,"temperature":0.2}}"#;
    let out: Value = serde_json::from_slice(&cap_gemini_max_output_tokens(
        body,
        "gemini-3.1-pro-preview",
        None,
    ))
    .unwrap();
    assert_eq!(
        out.pointer("/generationConfig/maxOutputTokens"),
        Some(&json!(65_536))
    );
    assert_eq!(
        out.pointer("/generationConfig/temperature"),
        Some(&json!(0.2))
    );
}

#[test]
fn cap_leaves_allowed_unknown_and_non_numeric_values() {
    for (model, input, expected) in [
        ("gemini-3.1-pro-preview", json!(64_000), json!(64_000)),
        ("custom-gemini", json!(500_000), json!(500_000)),
        ("gemini-3.1-pro-preview", json!("many"), json!("many")),
    ] {
        let body =
            serde_json::to_vec(&json!({"generationConfig":{"maxOutputTokens":input}})).unwrap();
        let out: Value =
            serde_json::from_slice(&cap_gemini_max_output_tokens(&body, model, None)).unwrap();
        assert_eq!(
            out.pointer("/generationConfig/maxOutputTokens"),
            Some(&expected)
        );
    }
}

#[test]
fn image_preview_aspect_ratio_injects_baseline_image_and_removes_config() {
    let body = br#"{"contents":[{"role":"user","parts":[{"text":"draw"}]}],"generationConfig":{"imageConfig":{"aspectRatio":"16:9"}}}"#;
    let out: Value = serde_json::from_slice(&fix_gemini_image_aspect_ratio(
        "gemini-2.5-flash-image-preview",
        body,
    ))
    .unwrap();
    assert!(out
        .pointer("/contents/0/parts/1/inlineData/data")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty()));
    assert_eq!(
        out.pointer("/generationConfig/responseModalities/0"),
        Some(&json!("IMAGE"))
    );
    assert!(out.pointer("/generationConfig/imageConfig").is_none());
}

#[tokio::test]
async fn execute_caps_before_upstream_and_uses_gemini_endpoint() {
    let client = MockClient::with_body(br#"{"candidates":[]}"#);
    let executor = GeminiExecutor::new(
        Arc::new(GeminiExecutorConfig::default()),
        Arc::new(Registry::new()),
    );
    executor
        .execute(request(Arc::clone(&client)))
        .await
        .unwrap();
    let upstream = client.request();
    assert_eq!(
        upstream.url,
        "http://upstream/v1beta/models/gemini-3.1-pro-preview:generateContent"
    );
    let body: Value = serde_json::from_slice(&upstream.body).unwrap();
    assert_eq!(
        body.pointer("/generationConfig/maxOutputTokens"),
        Some(&json!(65_536))
    );
}

#[tokio::test]
async fn gemini_api_key_interactions_source_still_uses_generate_content() {
    let client = MockClient::with_body(br#"{"candidates":[]}"#);
    let executor = GeminiExecutor::new(
        Arc::new(GeminiExecutorConfig::default()),
        Arc::new(Registry::new()),
    );
    let mut request = request(Arc::clone(&client));
    request.source_format = "interactions".into();
    request.payload = br#"{"model":"m","input":"hi"}"#.to_vec();
    executor.execute(request).await.unwrap();
    assert!(client.request().url.ends_with(":generateContent"));
    assert!(client
        .request()
        .headers
        .keys()
        .all(|key| !key.eq_ignore_ascii_case("Api-Revision")));
}

#[tokio::test]
async fn native_interactions_uses_endpoint_revision_and_agent_without_model() {
    let client = MockClient::with_body(br#"{"id":"interaction_1"}"#);
    let executor = GeminiExecutor::interactions(
        Arc::new(GeminiExecutorConfig::default()),
        Arc::new(Registry::new()),
    );
    let mut request = request(Arc::clone(&client));
    request.auth_provider = "gemini-interactions".into();
    request.source_format = "interactions".into();
    request.format = "interactions".into();
    request.model = "agents/test-agent".into();
    request.payload = br#"{"agent":"agents/test-agent","input":"hi","model":"remove-me"}"#.to_vec();
    let response = executor.execute(request).await.unwrap();
    let upstream = client.request();
    assert_eq!(upstream.url, "http://upstream/v1beta/interactions");
    assert_eq!(
        header(&upstream.headers, "Api-Revision"),
        Some(GEMINI_INTERACTIONS_API_REVISION)
    );
    assert!(serde_json::from_slice::<Value>(&upstream.body)
        .unwrap()
        .get("model")
        .is_none());
    assert_eq!(
        serde_json::from_slice::<Value>(&response.payload).unwrap()["id"],
        "interaction_1"
    );
}

#[tokio::test]
async fn request_revision_is_preserved_without_overriding_auth() {
    let client = MockClient::with_body(br#"{"id":"i"}"#);
    let executor = GeminiExecutor::interactions(
        Arc::new(GeminiExecutorConfig::default()),
        Arc::new(Registry::new()),
    );
    let mut request = request(Arc::clone(&client));
    request.auth_provider = "gemini-interactions".into();
    request.source_format = "interactions".into();
    request.payload = br#"{"input":"hi"}"#.to_vec();
    request
        .headers
        .insert("api-revision".into(), vec!["custom".into()]);
    request
        .headers
        .insert("Authorization".into(), vec!["request-token".into()]);
    executor.execute(request).await.unwrap();
    let upstream = client.request();
    assert_eq!(header(&upstream.headers, "Api-Revision"), Some("custom"));
    assert_eq!(
        header(&upstream.headers, "x-goog-api-key"),
        Some("test-key")
    );
    assert_eq!(header(&upstream.headers, "Authorization"), None);
}

#[tokio::test]
async fn interactions_payload_rules_match_protocol_and_source() {
    let client = MockClient::with_body(br#"{"id":"i"}"#);
    let rules = vec![GeminiPayloadRule {
        models: vec!["m".into()],
        protocol: "interactions".into(),
        from_protocol: "responses".into(),
        defaults: Map::from_iter([("generation_config.temperature".into(), json!(0.8))]),
        overrides: Map::from_iter([(
            "generation_config.thinking_summaries".into(),
            json!("detailed"),
        )]),
    }];
    let executor = GeminiExecutor::interactions(
        Arc::new(GeminiExecutorConfig {
            payload_rules: rules,
            ..GeminiExecutorConfig::default()
        }),
        Arc::new(Registry::new()),
    );
    let mut request = request(Arc::clone(&client));
    request.auth_provider = "gemini-interactions".into();
    request.source_format = "responses".into();
    request.model = "m".into();
    request.payload = br#"{"input":"hi","generation_config":{"temperature":0.2}}"#.to_vec();
    executor.execute(request).await.unwrap();
    let body: Value = serde_json::from_slice(&client.request().body).unwrap();
    assert_eq!(
        body.pointer("/generation_config/temperature"),
        Some(&json!(0.2))
    );
    assert_eq!(
        body.pointer("/generation_config/thinking_summaries"),
        Some(&json!("detailed"))
    );
}

#[tokio::test]
async fn native_interactions_translates_openai_responses_and_reasoning() {
    let client = MockClient::with_body(
        br#"{"id":"interaction_1","status":"completed","steps":[{"type":"model_output","content":[{"text":"ok"}]}]}"#,
    );
    let registry = Arc::new(Registry::new());
    register_all(&registry);
    let executor =
        GeminiExecutor::interactions(Arc::new(GeminiExecutorConfig::default()), registry);
    let mut request = request(Arc::clone(&client));
    request.auth_provider = "gemini-interactions".into();
    request.source_format = "openai-response".into();
    request.format = "openai-response".into();
    request.model = "gemini-3.1-flash-lite".into();
    request.payload = br#"{"model":"gemini-3.1-flash-lite","instructions":"brief","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"reasoning":{"effort":"high","summary":"auto"}}"#.to_vec();
    let response = executor.execute(request).await.unwrap();
    let upstream: Value = serde_json::from_slice(&client.request().body).unwrap();
    assert_eq!(
        upstream.pointer("/input/0/type"),
        Some(&json!("user_input"))
    );
    assert_eq!(
        upstream.pointer("/generation_config/thinking_level"),
        Some(&json!("high"))
    );
    let output: Value = serde_json::from_slice(&response.payload).unwrap();
    assert_eq!(
        output.pointer("/output/0/content/0/text"),
        Some(&json!("ok"))
    );
}

#[tokio::test]
async fn native_interactions_applies_thinking_suffix_without_rewriting_native_fields() {
    let client = MockClient::with_body(br#"{"id":"i"}"#);
    let executor = GeminiExecutor::interactions(
        Arc::new(GeminiExecutorConfig::default()),
        Arc::new(Registry::new()),
    );
    let mut request = request(Arc::clone(&client));
    request.auth_provider = "gemini-interactions".into();
    request.source_format = "interactions".into();
    request.model = "gemini-3.1-flash-lite(high)".into();
    request.payload = br#"{"model":"gemini-3.1-flash-lite","generation_config":{"tool_choice":"auto","thinking_summaries":"auto"},"input":"hi"}"#.to_vec();
    executor.execute(request).await.unwrap();
    let upstream: Value = serde_json::from_slice(&client.request().body).unwrap();
    assert_eq!(
        upstream.pointer("/generation_config/thinking_level"),
        Some(&json!("high"))
    );
    assert_eq!(
        upstream.pointer("/generation_config/thinking_summaries"),
        Some(&json!("auto"))
    );
    assert!(upstream.get("generationConfig").is_none());
}

#[tokio::test]
async fn stream_parses_interactions_sse_and_stops_at_done() {
    let client = MockClient::with_body(b"{}");
    *client.stream_chunks.lock().unwrap() = vec![
        b"event: step.delta\ndata: {\"event_type\":\"step.delta\"}\n\n".to_vec(),
        b"event: done\ndata: [DONE]\n\n".to_vec(),
        b"data: {\"leaked\":true}\n\n".to_vec(),
    ];
    let executor = GeminiExecutor::interactions(
        Arc::new(GeminiExecutorConfig::default()),
        Arc::new(Registry::new()),
    );
    let mut request = request(client);
    request.auth_provider = "gemini-interactions".into();
    request.source_format = "interactions".into();
    request.payload = b"{}".to_vec();
    let mut stream = executor.execute_stream(request).await.unwrap();
    let first = stream.chunks.recv().await.unwrap();
    let payload = gemini_interactions_sse_payload(&first.payload);
    assert_eq!(
        serde_json::from_slice::<Value>(&payload).unwrap()["event_type"],
        "step.delta"
    );
    let done = stream.chunks.recv().await.unwrap();
    assert!(gemini_interactions_sse_done(&done.payload));
    assert!(stream.chunks.recv().await.is_none());
}

#[tokio::test]
async fn count_tokens_strips_generation_tools_and_safety() {
    let client = MockClient::with_body(br#"{"totalTokens":7}"#);
    let executor = GeminiExecutor::new(
        Arc::new(GeminiExecutorConfig::default()),
        Arc::new(Registry::new()),
    );
    let mut request = request(Arc::clone(&client));
    request.payload =
        br#"{"contents":[],"generationConfig":{},"tools":[],"safetySettings":[]}"#.to_vec();
    let response = executor.count_tokens(request).await.unwrap();
    let body: Value = serde_json::from_slice(&client.request().body).unwrap();
    assert!(
        body.get("generationConfig").is_none()
            && body.get("tools").is_none()
            && body.get("safetySettings").is_none()
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&response.payload).unwrap()["totalTokens"],
        7
    );
}

#[tokio::test]
async fn cancellation_prevents_transport_call() {
    let client = MockClient::with_body(b"{}");
    let context = TranslationContext::default();
    context.cancel();
    let executor = GeminiExecutor::with_context(
        Arc::new(GeminiExecutorConfig::default()),
        Arc::new(Registry::new()),
        context,
    );
    assert!(executor
        .execute(request(Arc::clone(&client)))
        .await
        .is_err());
    assert!(client.requests.lock().unwrap().is_empty());
}

#[test]
fn supported_native_interactions_entry_protocols_match_upstream() {
    for protocol in [
        "interactions",
        "openai",
        "responses",
        "openai-response",
        "claude",
        "gemini",
    ] {
        assert!(native_interactions_source_format(&Format::from(protocol)));
    }
    assert!(!native_interactions_source_format(&Format::from("unknown")));
}

#[test]
fn sse_helpers_accept_raw_json_multiline_data_and_done_variants() {
    assert_eq!(
        gemini_interactions_sse_payload(br#" {"a":1} "#),
        br#"{"a":1}"#
    );
    assert_eq!(
        gemini_interactions_sse_payload(b"event: x\ndata: {\"a\":\ndata: 1}\n"),
        b"{\"a\":\n1}"
    );
    assert!(gemini_interactions_sse_done(b"[DONE]"));
    assert!(gemini_interactions_sse_done(b"event: done\n"));
    assert!(gemini_interactions_sse_done(b"data: [DONE]\n"));
    assert!(!gemini_interactions_sse_done(b"event: step.delta\n"));
}

#[test]
fn revision_helpers_preserve_explicit_request_value() {
    let source = BTreeMap::from([("api-revision".into(), vec!["request".into()])]);
    let mut target = BTreeMap::new();
    apply_interactions_request_headers(&mut target, &source);
    apply_interactions_revision_header(&mut target);
    assert_eq!(header(&target, "Api-Revision"), Some("request"));
}

fn header<'a>(headers: &'a BTreeMap<String, Vec<String>>, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.first())
        .map(String::as_str)
}

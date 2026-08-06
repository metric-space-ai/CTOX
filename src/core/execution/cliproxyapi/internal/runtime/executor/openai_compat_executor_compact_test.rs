// ref: internal/runtime/executor/openai_compat_executor_compact_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use super::openai_compat_executor::{
    prepare_openai_compat_images_payload, OpenAiCompatConfig, OpenAiCompatError,
    OpenAiCompatExecutor, OpenAiCompatPayloadModelRule, OpenAiCompatPayloadRule,
    OpenAiCompatibility,
};
use crate::sdk::pluginapi::{
    ExecutorRequest, HostHttpClient, HttpRequest, HttpResponse, HttpStreamChunk,
    HttpStreamResponse, PluginFuture, ProviderExecutor,
};
use crate::sdk::translator::{Format, Registry};

#[derive(Default)]
struct RecordedClient {
    requests: Mutex<Vec<HttpRequest>>,
    response_body: Mutex<Vec<u8>>,
    stream_chunks: Mutex<Vec<Vec<u8>>>,
}

impl RecordedClient {
    fn with_response(body: &[u8]) -> Arc<Self> {
        Arc::new(Self {
            response_body: Mutex::new(body.to_vec()),
            ..Self::default()
        })
    }

    fn with_stream(chunks: &[&[u8]]) -> Arc<Self> {
        Arc::new(Self {
            stream_chunks: Mutex::new(chunks.iter().map(|chunk| chunk.to_vec()).collect()),
            ..Self::default()
        })
    }

    fn request(&self) -> HttpRequest {
        self.requests.lock().unwrap().last().unwrap().clone()
    }
}

impl HostHttpClient for RecordedClient {
    fn execute<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpResponse> {
        self.requests.lock().unwrap().push(request);
        let body = self.response_body.lock().unwrap().clone();
        Box::pin(async move {
            Ok(HttpResponse {
                status_code: 200,
                body,
                ..HttpResponse::default()
            })
        })
    }

    fn execute_stream<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpStreamResponse> {
        self.requests.lock().unwrap().push(request);
        let chunks = self.stream_chunks.lock().unwrap().clone();
        Box::pin(async move {
            let (sender, receiver) = mpsc::channel(chunks.len().max(1));
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

fn executor(config: OpenAiCompatConfig) -> OpenAiCompatExecutor {
    OpenAiCompatExecutor::new(
        "openai-compatibility",
        Arc::new(config),
        Arc::new(Registry::new()),
    )
}

fn compat_config(support_prompt_cache_key: bool) -> OpenAiCompatConfig {
    OpenAiCompatConfig {
        compatibility: vec![OpenAiCompatibility {
            name: "compat".into(),
            support_prompt_cache_key,
            ..OpenAiCompatibility::default()
        }],
        ..OpenAiCompatConfig::default()
    }
}

fn request(client: Arc<RecordedClient>, payload: &[u8]) -> ExecutorRequest {
    ExecutorRequest {
        auth_provider: "openai-compatibility".into(),
        model: "gpt-5.6".into(),
        format: "openai".into(),
        source_format: "openai".into(),
        payload: payload.to_vec(),
        original_request: payload.to_vec(),
        auth_attributes: BTreeMap::from([
            ("base_url".into(), "https://provider.example/v1".into()),
            ("api_key".into(), "test".into()),
            ("compat_name".into(), "compat".into()),
            ("provider_key".into(), "compat".into()),
        ]),
        http_client: Some(client),
        ..ExecutorRequest::default()
    }
}

fn prompt_key(payload: &[u8]) -> Option<String> {
    serde_json::from_slice::<Value>(payload)
        .ok()?
        .get("prompt_cache_key")?
        .as_str()
        .map(str::to_owned)
}

async fn collect_stream(
    mut response: crate::sdk::pluginapi::ExecutorStreamResponse,
) -> (Vec<u8>, Option<crate::sdk::pluginapi::PluginExecutionError>) {
    let mut output = Vec::new();
    while let Some(chunk) = response.chunks.recv().await {
        if chunk.error.is_some() {
            return (output, chunk.error);
        }
        output.extend_from_slice(&chunk.payload);
    }
    (output, None)
}

#[tokio::test]
async fn compact_passthrough() {
    let upstream = br#"{"id":"resp_1","object":"response.compaction"}"#;
    let client = RecordedClient::with_response(upstream);
    let mut request = request(
        client.clone(),
        br#"{"model":"gpt-5.1-codex-max","input":[{"role":"user","content":"hi"}]}"#,
    );
    request.model = "gpt-5.1-codex-max".into();
    request.source_format = "openai-response".into();
    request.format = "openai-response".into();
    request.alt = "responses/compact".into();
    let response = executor(compat_config(true))
        .execute(request)
        .await
        .unwrap();
    let sent = client.request();
    assert_eq!(sent.url, "https://provider.example/v1/responses/compact");
    assert!(serde_json::from_slice::<Value>(&sent.body).unwrap()["input"].is_array());
    assert!(prompt_key(&sent.body).is_none());
    assert_eq!(response.payload, upstream);
}

#[test]
fn payload_override_wins_over_thinking_suffix() {
    let mut params = Map::new();
    params.insert("reasoning_effort".into(), json!("low"));
    let executor = executor(OpenAiCompatConfig {
        payload_overrides: vec![OpenAiCompatPayloadRule {
            models: vec![OpenAiCompatPayloadModelRule {
                name: "custom-openai".into(),
                protocol: "openai".into(),
            }],
            params,
        }],
        ..OpenAiCompatConfig::default()
    });
    let client = RecordedClient::with_response(b"{}");
    let mut request = request(client, br#"{"model":"custom-openai(high)","messages":[]}"#);
    request.model = "custom-openai(high)".into();
    let output = executor.translate_request(&request, &Format::from("openai"), false);
    assert_eq!(
        serde_json::from_slice::<Value>(&output).unwrap()["reasoning_effort"],
        "low"
    );
}

#[test]
fn apply_prompt_cache_key_matrix() {
    struct Case {
        support: bool,
        source: &'static str,
        payload: &'static [u8],
        session: Option<&'static str>,
        expected: Option<&'static str>,
    }
    let cases = [
        Case { support: false, source: "claude", payload: br#"{"model":"gpt-5.6","metadata":{"user_id":"{\"session_id\":\"cache-session\"}"}}"#, session: None, expected: None },
        Case { support: true, source: "claude", payload: br#"{"model":"gpt-5.6","metadata":{"user_id":"{\"session_id\":\"cache-session\"}"}}"#, session: None, expected: Some("") },
        Case { support: true, source: "claude", payload: br#"{"model":"gpt-5.6","prompt_cache_key":"caller-key","metadata":{"user_id":"{\"session_id\":\"cache-session\"}"}}"#, session: None, expected: Some("caller-key") },
        Case { support: true, source: "openai", payload: br#"{"model":"gpt-5.6","messages":[]}"#, session: None, expected: None },
        Case { support: true, source: "openai", payload: br#"{"model":"gpt-5.6","messages":[]}"#, session: Some("ctx:v1:openai"), expected: Some("") },
        Case { support: true, source: "openai-response", payload: br#"{"model":"gpt-5.6","input":"hello"}"#, session: Some("ctx:v1:responses"), expected: Some("") },
        Case { support: true, source: "gemini", payload: br#"{"model":"gemini-3","contents":[]}"#, session: Some("ctx:v1:gemini"), expected: Some("") },
        Case { support: true, source: "interactions", payload: br#"{"model":"gpt-5.6","input":"hello"}"#, session: Some("ctx:v1:interactions"), expected: Some("") },
        Case { support: true, source: "codex", payload: br#"{"model":"gpt-5.6","input":"hello"}"#, session: Some("ctx:v1:codex"), expected: Some("") },
        Case { support: true, source: "antigravity", payload: br#"{"model":"gpt-5.6","input":"hello"}"#, session: Some("ctx:v1:antigravity"), expected: Some("") },
    ];
    for case in cases {
        let client = RecordedClient::with_response(b"{}");
        let mut request = request(client, case.payload);
        request.source_format = case.source.into();
        if let Some(session) = case.session {
            request
                .metadata
                .insert("derived_session_id".into(), json!(session));
        }
        let output = executor(compat_config(case.support))
            .apply_prompt_cache_key(&request, br#"{"model":"gpt-5.6","messages":[]}"#);
        match case.expected {
            None => assert!(prompt_key(&output).is_none()),
            Some("") => assert!(prompt_key(&output).is_some_and(|key| !key.is_empty())),
            Some(expected) => assert_eq!(prompt_key(&output).as_deref(), Some(expected)),
        }
    }
}

#[test]
fn caller_prompt_cache_key_wins_payload_override() {
    for (payload, original) in [
        (
            br#"{"prompt_cache_key":"caller-key"}"#.as_slice(),
            b"".as_slice(),
        ),
        (
            b"".as_slice(),
            br#"{"prompt_cache_key":"caller-key"}"#.as_slice(),
        ),
    ] {
        let client = RecordedClient::with_response(b"{}");
        let mut request = request(client, payload);
        request.original_request = original.to_vec();
        request
            .metadata
            .insert("derived_session_id".into(), json!("session"));
        let output = executor(compat_config(true)).apply_prompt_cache_key(
            &request,
            br#"{"model":"gpt-5.6","prompt_cache_key":"payload-override"}"#,
        );
        assert_eq!(prompt_key(&output).as_deref(), Some("caller-key"));
    }
}

#[test]
fn prompt_cache_key_is_model_and_protocol_scoped() {
    let derive = |model: &str, source: &str| {
        let client = RecordedClient::with_response(b"{}");
        let mut request = request(client, br#"{"messages":[]}"#);
        request.model = model.into();
        request.source_format = source.into();
        request
            .metadata
            .insert("execution_session_id".into(), json!("execution-session"));
        executor(compat_config(true)).apply_prompt_cache_key(
            &request,
            format!(r#"{{"model":"{model}","messages":[]}}"#).as_bytes(),
        )
    };
    let base = derive("gpt-5.6", "openai");
    assert_eq!(
        prompt_key(&base).as_deref(),
        Some("17a745d7-09eb-5c01-9c79-61bf39a32b0d")
    );
    assert_ne!(prompt_key(&base), prompt_key(&derive("gpt-5.5", "openai")));
    assert_ne!(
        prompt_key(&base),
        prompt_key(&derive("gpt-5.6", "openai-response"))
    );
}

#[test]
fn prompt_cache_key_uses_config_index() {
    let config = OpenAiCompatConfig {
        compatibility: vec![
            OpenAiCompatibility {
                name: "duplicate".into(),
                support_prompt_cache_key: false,
                ..OpenAiCompatibility::default()
            },
            OpenAiCompatibility {
                name: "duplicate".into(),
                support_prompt_cache_key: true,
                ..OpenAiCompatibility::default()
            },
        ],
        ..OpenAiCompatConfig::default()
    };
    for (index, expected) in [("0", false), ("1", true)] {
        let client = RecordedClient::with_response(b"{}");
        let mut request = request(
            client,
            br#"{"metadata":{"user_id":"{\"session_id\":\"cache-session\"}"}}"#,
        );
        request.source_format = "claude".into();
        request
            .auth_attributes
            .insert("compat_name".into(), "duplicate".into());
        request
            .auth_attributes
            .insert("provider_key".into(), "duplicate".into());
        request
            .auth_attributes
            .insert("config_index".into(), index.into());
        request
            .auth_attributes
            .insert("source".into(), "config:duplicate[0]".into());
        let output =
            executor(config.clone()).apply_prompt_cache_key(&request, br#"{"model":"gpt-5.6"}"#);
        assert_eq!(prompt_key(&output).is_some(), expected);
    }
}

#[test]
fn prompt_cache_key_ignores_config_index_for_non_config_auth() {
    let config = OpenAiCompatConfig {
        compatibility: vec![
            OpenAiCompatibility {
                name: "duplicate".into(),
                support_prompt_cache_key: false,
                ..OpenAiCompatibility::default()
            },
            OpenAiCompatibility {
                name: "duplicate".into(),
                support_prompt_cache_key: true,
                ..OpenAiCompatibility::default()
            },
        ],
        ..OpenAiCompatConfig::default()
    };
    let client = RecordedClient::with_response(b"{}");
    let mut request = request(client, br#"{"messages":[]}"#);
    request
        .auth_attributes
        .insert("compat_name".into(), "duplicate".into());
    request
        .auth_attributes
        .insert("provider_key".into(), "duplicate".into());
    request
        .auth_attributes
        .insert("config_index".into(), "1".into());
    request
        .metadata
        .insert("derived_session_id".into(), json!("ctx:v1:non-config"));
    let output = executor(config).apply_prompt_cache_key(&request, br#"{"model":"gpt-5.6"}"#);
    assert!(prompt_key(&output).is_none());
}

#[tokio::test]
async fn prompt_cache_key_execute() {
    let client = RecordedClient::with_response(br#"{"id":"chatcmpl_1"}"#);
    let mut request = request(client.clone(), br#"{"model":"gpt-5.6","messages":[]}"#);
    request
        .metadata
        .insert("derived_session_id".into(), json!("ctx:v1:openai"));
    executor(compat_config(true))
        .execute(request)
        .await
        .unwrap();
    assert!(prompt_key(&client.request().body).is_some());
}

#[tokio::test]
async fn prompt_cache_key_execute_stream() {
    let client = RecordedClient::with_stream(&[b"data: [DONE]\n\n"]);
    let mut request = request(
        client.clone(),
        br#"{"model":"gpt-5.6","messages":[],"stream":true}"#,
    );
    request.stream = true;
    request
        .metadata
        .insert("derived_session_id".into(), json!("ctx:v1:openai-stream"));
    let response = executor(compat_config(true))
        .execute_stream(request)
        .await
        .unwrap();
    let (_, error) = collect_stream(response).await;
    assert!(error.is_none());
    assert!(prompt_key(&client.request().body).is_some());
}

#[tokio::test]
async fn prompt_cache_key_stream_compact_is_skipped() {
    let client = RecordedClient::with_stream(&[b"data: [DONE]\n\n"]);
    let mut request = request(
        client.clone(),
        br#"{"model":"gpt-5.6","messages":[],"stream":true}"#,
    );
    request.stream = true;
    request.alt = "responses/compact".into();
    request
        .metadata
        .insert("derived_session_id".into(), json!("ctx:v1:compact-stream"));
    let response = executor(compat_config(true))
        .execute_stream(request)
        .await
        .unwrap();
    let _ = collect_stream(response).await;
    assert!(prompt_key(&client.request().body).is_none());
}

#[tokio::test]
async fn images_generations_passthrough() {
    let upstream = br#"{"created":123,"data":[{"b64_json":"AA=="}]}"#;
    let client = RecordedClient::with_response(upstream);
    let mut request = request(
        client.clone(),
        br#"{"model":"compat-image","prompt":"draw"}"#,
    );
    request.model = "upstream-image".into();
    request.source_format = "openai-image".into();
    request
        .headers
        .insert("Content-Type".into(), vec!["application/json".into()]);
    request
        .metadata
        .insert("request_path".into(), json!("/v1/images/generations"));
    let response = executor(compat_config(true))
        .execute(request)
        .await
        .unwrap();
    let sent = client.request();
    assert_eq!(sent.url, "https://provider.example/v1/images/generations");
    assert_eq!(
        serde_json::from_slice::<Value>(&sent.body).unwrap()["model"],
        "upstream-image"
    );
    assert!(prompt_key(&sent.body).is_none());
    assert_eq!(response.payload, upstream);
}

#[tokio::test]
async fn images_generations_stream_upstream() {
    let client = RecordedClient::with_stream(&[
        b"event: image_generation.partial\ndata: {\"type\":\"image_generation.partial\"}\n\n",
        b"data: [DONE]\n\n",
    ]);
    let mut request = request(
        client.clone(),
        br#"{"model":"compat-image","prompt":"draw","stream":true}"#,
    );
    request.model = "upstream-image".into();
    request.source_format = "openai-image".into();
    request.stream = true;
    request
        .headers
        .insert("Content-Type".into(), vec!["application/json".into()]);
    request
        .metadata
        .insert("request_path".into(), json!("/v1/images/generations"));
    let response = executor(OpenAiCompatConfig::default())
        .execute_stream(request)
        .await
        .unwrap();
    let (output, error) = collect_stream(response).await;
    assert!(error.is_none());
    assert!(String::from_utf8_lossy(&output).contains("image_generation.partial"));
    let sent = client.request();
    assert_eq!(sent.url, "https://provider.example/v1/images/generations");
    assert_eq!(
        serde_json::from_slice::<Value>(&sent.body).unwrap()["stream"],
        true
    );
}

fn multipart_payload() -> (Vec<u8>, &'static str) {
    let boundary = "upstream-boundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ncompat-image\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"image.webp\"\r\nContent-Type: image/webp\r\n\r\nwebp-data\r\n--{boundary}--\r\n"
    );
    (body.into_bytes(), boundary)
}

#[tokio::test]
async fn images_edits_multipart_rewrites_model() {
    let (body, boundary) = multipart_payload();
    let client = RecordedClient::with_response(br#"{"data":[]}"#);
    let mut request = request(client.clone(), &body);
    request.model = "upstream-image".into();
    request.source_format = "openai-image".into();
    request.headers.insert(
        "Content-Type".into(),
        vec![format!("multipart/form-data; boundary={boundary}")],
    );
    request
        .metadata
        .insert("request_path".into(), json!("/v1/images/edits"));
    executor(OpenAiCompatConfig::default())
        .execute(request)
        .await
        .unwrap();
    let sent = client.request();
    let sent_text = String::from_utf8_lossy(&sent.body);
    assert_eq!(sent.url, "https://provider.example/v1/images/edits");
    assert!(sent_text.contains("upstream-image"));
    assert!(sent_text.contains("edit"));
    assert!(sent_text.contains("webp-data"));
    assert!(sent_text.contains("Content-Type: image/webp"));
    assert!(!sent_text.contains("compat-image"));
}

#[test]
fn multipart_rewrite_preserves_stream_and_file_content_type() {
    let (body, boundary) = multipart_payload();
    let (output, content_type) = prepare_openai_compat_images_payload(
        &body,
        "upstream-image",
        &format!("multipart/form-data; boundary={boundary}"),
        true,
    )
    .unwrap();
    let text = String::from_utf8(output).unwrap();
    assert!(content_type.starts_with("multipart/form-data; boundary=ctox-"));
    assert!(text.contains("upstream-image"));
    assert!(text.contains("name=\"stream\"\r\n\r\ntrue"));
    assert!(text.contains("Content-Type: image/webp"));
    assert!(text.contains("webp-data"));
}

#[tokio::test]
async fn stream_rejects_plain_json_after_blank_lines() {
    let client = RecordedClient::with_stream(&[
        b"\n\n: openrouter processing\n\nevent: error\n",
        br#"{"error":{"message":"upstream failed","type":"server_error"}}
"#,
    ]);
    let mut request = request(
        client,
        br#"{"model":"openrouter-model","messages":[],"stream":true}"#,
    );
    request.model = "openrouter-model".into();
    request.stream = true;
    let response = executor(OpenAiCompatConfig::default())
        .execute_stream(request)
        .await
        .unwrap();
    let (_, error) = collect_stream(response).await;
    let error = error.expect("plain JSON must terminate the SSE stream");
    let error = error.as_ref().downcast_ref::<OpenAiCompatError>().unwrap();
    assert_eq!(error.status_code, 502);
    assert!(error.message.contains("upstream failed"));
}

#[tokio::test]
async fn stream_skips_keepalive_until_data_line() {
    let client = RecordedClient::with_stream(&[
        b"\n\n: openrouter processing\n\nevent: ping\nid: 1\nretry: 1000\n",
        br#"data: {"id":"chatcmpl_1","choices":[{"delta":{"content":"hello"}}]}
"#,
    ]);
    let mut request = request(
        client,
        br#"{"model":"openrouter-model","messages":[],"stream":true}"#,
    );
    request.model = "openrouter-model".into();
    request.stream = true;
    let response = executor(OpenAiCompatConfig::default())
        .execute_stream(request)
        .await
        .unwrap();
    let (output, error) = collect_stream(response).await;
    assert!(error.is_none());
    assert!(String::from_utf8_lossy(&output).contains("hello"));
}

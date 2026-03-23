use anyhow::Context;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;
use tiny_http::Header;
use tiny_http::Method;
use tiny_http::Response;
use tiny_http::Server;
use tiny_http::StatusCode;

use crate::execution_baseline;
use crate::runtime_config;

const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarmonyRelayMode {
    Disabled,
    Json,
    Sse,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProxyConfig {
    pub listen_host: String,
    pub listen_port: u16,
    pub upstream_base_url: String,
    pub active_model: Option<String>,
}

impl ProxyConfig {
    pub fn from_env_with_root(root: &std::path::Path) -> Self {
        Self {
            listen_host: runtime_config::env_or_config(root, "CTOX_PROXY_HOST")
                .unwrap_or_else(|| "127.0.0.1".to_string()),
            listen_port: runtime_config::env_or_config(root, "CTOX_PROXY_PORT")
                .and_then(|value| value.parse().ok())
                .unwrap_or(12434),
            upstream_base_url: runtime_config::env_or_config(root, "CTOX_UPSTREAM_BASE_URL")
                .unwrap_or_else(|| "http://127.0.0.1:1234".to_string()),
            active_model: runtime_config::env_or_config(root, "CTOX_ACTIVE_MODEL")
                .or_else(|| runtime_config::env_or_config(root, "CTOX_CHAT_MODEL")),
        }
    }

    pub fn listen_addr(&self) -> String {
        format!("{}:{}", self.listen_host, self.listen_port)
    }

    pub fn join_url(&self, request_url: &str) -> String {
        format!(
            "{}{}",
            self.upstream_base_url.trim_end_matches('/'),
            request_url
        )
    }
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
pub struct ProxyTelemetry {
    pub active_model: Option<String>,
    pub last_request_path: Option<String>,
    pub last_response_at: Option<String>,
    pub last_latency_ms: Option<u64>,
    pub last_input_tokens: Option<u64>,
    pub last_output_tokens: Option<u64>,
    pub last_total_tokens: Option<u64>,
    pub last_tokens_per_second: Option<f64>,
}

pub fn serve_proxy(config: ProxyConfig) -> anyhow::Result<()> {
    let server = Server::http(config.listen_addr())
        .map_err(|err| anyhow::anyhow!("failed to bind CTOX responses proxy: {err}"))?;
    let shared = Arc::new(config);
    let telemetry = Arc::new(Mutex::new(ProxyTelemetry {
        active_model: shared.active_model.clone(),
        ..ProxyTelemetry::default()
    }));
    let response_state = Arc::new(Mutex::new(HashMap::<String, Value>::new()));

    for request in server.incoming_requests() {
        let config = Arc::clone(&shared);
        let telemetry = Arc::clone(&telemetry);
        let response_state = Arc::clone(&response_state);
        if let Err(err) = handle_request(&config, &telemetry, &response_state, request) {
            eprintln!("ctox proxy error: {err}");
        }
    }

    Ok(())
}

fn handle_request(
    config: &ProxyConfig,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    response_state: &Arc<Mutex<HashMap<String, Value>>>,
    mut request: tiny_http::Request,
) -> anyhow::Result<()> {
    if matches!(request.method(), Method::Get) && request.url() == "/ctox/telemetry" {
        let snapshot = telemetry.lock().expect("proxy telemetry lock poisoned").clone();
        let response = Response::from_string(serde_json::to_string(&snapshot)?)
            .with_status_code(StatusCode(200))
            .with_header(json_header());
        request
            .respond(response)
            .context("failed to write proxy telemetry response")?;
        return Ok(());
    }

    let started = Instant::now();
    let method = request.method().as_str().to_string();
    let url = request.url().to_string();
    let mut body = Vec::new();
    request
        .as_reader()
        .read_to_end(&mut body)
        .context("failed to read proxy request body")?;

    let materialized_request = if matches!(request.method(), Method::Post) && url == "/v1/responses" && !body.is_empty() {
        let previous_conversation = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|payload| {
                payload
                    .get("previous_response_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .and_then(|response_id| {
                response_state
                    .lock()
                    .expect("proxy response state lock poisoned")
                    .get(&response_id)
                    .cloned()
            });
        Some(execution_baseline::materialize_responses_request(
            &body,
            previous_conversation.as_ref(),
        )?)
    } else {
        None
    };

    let effective_body = materialized_request
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()?
        .unwrap_or_else(|| body.clone());

    let use_gpt_oss_harmony_proxy =
        matches!(request.method(), Method::Post) && url == "/v1/responses" && !body.is_empty()
            && execution_baseline::should_use_gpt_oss_harmony_proxy(&effective_body)?;
    let harmony_relay_mode = if use_gpt_oss_harmony_proxy {
        if execution_baseline::responses_request_streams(&effective_body)? {
            HarmonyRelayMode::Sse
        } else {
            HarmonyRelayMode::Json
        }
    } else {
        HarmonyRelayMode::Disabled
    };
    eprintln!(
        "ctox proxy request method={} url={} harmony_proxy={}",
        method, url, use_gpt_oss_harmony_proxy
    );

    let forwarded_body = if use_gpt_oss_harmony_proxy {
        execution_baseline::rewrite_responses_to_gpt_oss_completion(&effective_body)?
    } else if matches!(request.method(), Method::Post) && url == "/v1/responses" && !body.is_empty()
    {
        execution_baseline::rewrite_vllm_serve_responses_request(&effective_body)?
    } else {
        body
    };

    let upstream_path = if use_gpt_oss_harmony_proxy {
        "/v1/completions".to_string()
    } else {
        url.clone()
    };
    eprintln!("ctox proxy upstream_path={upstream_path}");
    if use_gpt_oss_harmony_proxy {
        eprintln!(
            "ctox proxy forwarded harmony request={}",
            String::from_utf8_lossy(&forwarded_body)
        );
    }
    let upstream_url = config.join_url(&upstream_path);
    let agent = ureq::AgentBuilder::new().build();
    let mut upstream = agent.request(&method, &upstream_url);

    for header in request.headers() {
        let field = header.field.as_str().as_str();
        if HOP_BY_HOP_HEADERS
            .iter()
            .any(|candidate| field.eq_ignore_ascii_case(candidate))
        {
            continue;
        }
        upstream = upstream.set(field, header.value.as_str());
    }
    if let Some(active_model) = &config.active_model {
        upstream = upstream.set("x-ctox-active-model", active_model);
    }

    if use_gpt_oss_harmony_proxy {
        return relay_gpt_oss_harmony_response(
            config,
            telemetry,
            response_state,
            request,
            &agent,
            &method,
            &upstream_url,
            upstream,
            materialized_request,
            forwarded_body,
            harmony_relay_mode,
            &url,
            started,
        );
    }

    let upstream_response = if forwarded_body.is_empty() {
        upstream.call()
    } else {
        upstream.send_bytes(&forwarded_body)
    };

    match upstream_response {
        Ok(response) => relay_response(config, telemetry, request, response, harmony_relay_mode, &url, started),
        Err(ureq::Error::Status(_, response)) => {
            relay_response(config, telemetry, request, response, harmony_relay_mode, &url, started)
        }
        Err(err) => {
            let response = Response::from_string(
                serde_json::json!({
                    "error": {
                        "message": err.to_string()
                    }
                })
                .to_string(),
            )
            .with_status_code(StatusCode(502))
            .with_header(json_header());
            request
                .respond(response)
                .context("failed to write proxy error response")?;
            Ok(())
        }
    }
}

fn relay_gpt_oss_harmony_response(
    config: &ProxyConfig,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    response_state: &Arc<Mutex<HashMap<String, Value>>>,
    request: tiny_http::Request,
    agent: &ureq::Agent,
    method: &str,
    upstream_url: &str,
    mut upstream: ureq::Request,
    materialized_request: Option<Value>,
    forwarded_body: Vec<u8>,
    harmony_relay_mode: HarmonyRelayMode,
    request_path: &str,
    started: Instant,
) -> anyhow::Result<()> {
    let first_response = if forwarded_body.is_empty() {
        upstream.call()
    } else {
        upstream.send_bytes(&forwarded_body)
    };
    let mut status_code = 200u16;
    let mut response_headers: Vec<(String, String)> = Vec::new();
    let mut body = match first_response {
        Ok(response) => {
            status_code = response.status();
            response_headers = response
                .headers_names()
                .into_iter()
                .filter(|header_name| {
                    !HOP_BY_HOP_HEADERS
                        .iter()
                        .any(|candidate| header_name.eq_ignore_ascii_case(candidate))
                })
                .flat_map(|header_name| {
                    response
                        .all(&header_name)
                        .into_iter()
                        .map(move |header_value| (header_name.clone(), header_value.to_string()))
                })
                .collect();
            let mut body = Vec::new();
            response
                .into_reader()
                .read_to_end(&mut body)
                .context("failed to read upstream harmony response body")?;
            body
        }
        Err(ureq::Error::Status(code, response)) => {
            status_code = code;
            response_headers = response
                .headers_names()
                .into_iter()
                .filter(|header_name| {
                    !HOP_BY_HOP_HEADERS
                        .iter()
                        .any(|candidate| header_name.eq_ignore_ascii_case(candidate))
                })
                .flat_map(|header_name| {
                    response
                        .all(&header_name)
                        .into_iter()
                        .map(move |header_value| (header_name.clone(), header_value.to_string()))
                })
                .collect();
            let mut body = Vec::new();
            response
                .into_reader()
                .read_to_end(&mut body)
                .context("failed to read upstream harmony error body")?;
            body
        }
        Err(err) => {
            let response = Response::from_string(
                serde_json::json!({
                    "error": { "message": err.to_string() }
                })
                .to_string(),
            )
            .with_status_code(StatusCode(502))
            .with_header(json_header());
            request.respond(response).context("failed to write proxy error response")?;
            return Ok(());
        }
    };

    if status_code < 400 {
        if let Some(followup_body) =
            execution_baseline::build_gpt_oss_followup_completion_request(&forwarded_body, &body)?
        {
            eprintln!("ctox proxy issuing GPT-OSS continuation completion");
            let mut followup = agent.request(method, upstream_url);
            followup = followup.set("content-type", "application/json");
            if let Some(active_model) = &config.active_model {
                followup = followup.set("x-ctox-active-model", active_model);
            }
            match followup.send_bytes(&followup_body) {
                Ok(response) => {
                    status_code = response.status();
                    response_headers = response
                        .headers_names()
                        .into_iter()
                        .filter(|header_name| {
                            !HOP_BY_HOP_HEADERS
                                .iter()
                                .any(|candidate| header_name.eq_ignore_ascii_case(candidate))
                        })
                        .flat_map(|header_name| {
                            response
                                .all(&header_name)
                                .into_iter()
                                .map(move |header_value| (header_name.clone(), header_value.to_string()))
                        })
                        .collect();
                    body.clear();
                    response
                        .into_reader()
                        .read_to_end(&mut body)
                        .context("failed to read GPT-OSS continuation body")?;
                    eprintln!(
                        "ctox proxy continuation harmony body={}",
                        String::from_utf8_lossy(&body)
                    );
                }
                Err(ureq::Error::Status(code, response)) => {
                    status_code = code;
                    response_headers = response
                        .headers_names()
                        .into_iter()
                        .filter(|header_name| {
                            !HOP_BY_HOP_HEADERS
                                .iter()
                                .any(|candidate| header_name.eq_ignore_ascii_case(candidate))
                        })
                        .flat_map(|header_name| {
                            response
                                .all(&header_name)
                                .into_iter()
                                .map(move |header_value| (header_name.clone(), header_value.to_string()))
                        })
                        .collect();
                    body.clear();
                    response
                        .into_reader()
                        .read_to_end(&mut body)
                        .context("failed to read GPT-OSS continuation error body")?;
                }
                Err(err) => {
                    let response = Response::from_string(
                        serde_json::json!({
                            "error": { "message": err.to_string() }
                        })
                        .to_string(),
                    )
                    .with_status_code(StatusCode(502))
                    .with_header(json_header());
                    request.respond(response).context("failed to write proxy error response")?;
                    return Ok(());
                }
            }
        }
    }

    if status_code < 400 {
        if let Some(materialized_request) = materialized_request.as_ref() {
            let response_payload: Value =
                serde_json::from_slice(&execution_baseline::rewrite_gpt_oss_completion_to_responses(&body, None)?)
                    .context("failed to parse rewritten responses payload for proxy state")?;
            if let Some(response_id) = response_payload.get("id").and_then(Value::as_str) {
                let conversation =
                    execution_baseline::extend_conversation_with_response(materialized_request, &response_payload)?;
                response_state
                    .lock()
                    .expect("proxy response state lock poisoned")
                    .insert(response_id.to_string(), conversation);
            }
        }
    }

    relay_response_from_parts(
        config,
        telemetry,
        request,
        status_code,
        response_headers,
        body,
        harmony_relay_mode,
        request_path,
        started,
    )
}

fn relay_response(
    config: &ProxyConfig,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    request: tiny_http::Request,
    response: ureq::Response,
    harmony_relay_mode: HarmonyRelayMode,
    request_path: &str,
    started: Instant,
) -> anyhow::Result<()> {
    let status = StatusCode(response.status());
    let response_headers: Vec<(String, String)> = response
        .headers_names()
        .into_iter()
        .filter(|header_name| {
            !HOP_BY_HOP_HEADERS
                .iter()
                .any(|candidate| header_name.eq_ignore_ascii_case(candidate))
        })
        .flat_map(|header_name| {
            response
                .all(&header_name)
                .into_iter()
                .map(move |header_value| (header_name.clone(), header_value.to_string()))
        })
        .collect();
    let mut body = Vec::new();
    let mut reader = response.into_reader();
    reader
        .read_to_end(&mut body)
        .context("failed to read upstream proxy response body")?;
    relay_response_from_parts(
        config,
        telemetry,
        request,
        status.0,
        response_headers,
        body,
        harmony_relay_mode,
        request_path,
        started,
    )
}

fn relay_response_from_parts(
    config: &ProxyConfig,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    request: tiny_http::Request,
    status_code: u16,
    response_headers: Vec<(String, String)>,
    body: Vec<u8>,
    harmony_relay_mode: HarmonyRelayMode,
    request_path: &str,
    started: Instant,
) -> anyhow::Result<()> {
    let status = StatusCode(status_code);
    let mut content_type_override: Option<&'static str> = None;
    let body = match harmony_relay_mode {
        HarmonyRelayMode::Disabled => body,
        HarmonyRelayMode::Json if status.0 < 400 => {
            eprintln!("ctox proxy rewriting harmony completion response into responses payload");
            eprintln!("ctox proxy raw harmony body={}", String::from_utf8_lossy(&body));
            content_type_override = Some("application/json");
            execution_baseline::rewrite_gpt_oss_completion_to_responses(&body, None)?
        }
        HarmonyRelayMode::Sse if status.0 < 400 => {
            eprintln!("ctox proxy rewriting harmony completion response into SSE payload");
            eprintln!("ctox proxy raw harmony body={}", String::from_utf8_lossy(&body));
            content_type_override = Some("text/event-stream");
            execution_baseline::rewrite_gpt_oss_completion_to_sse(&body, None)?
        }
        _ => body,
    };
    update_proxy_telemetry(
        telemetry,
        config,
        request_path,
        status.0,
        &body,
        started.elapsed().as_millis() as u64,
    );

    let mut tiny_response = Response::from_data(body).with_status_code(status);
    for (header_name, header_value) in response_headers {
        if content_type_override.is_some() && header_name.eq_ignore_ascii_case("content-type") {
            continue;
        }
        if let Ok(header) = Header::from_bytes(header_name.as_bytes(), header_value.as_bytes()) {
            tiny_response = tiny_response.with_header(header);
        }
    }
    if let Some(content_type) = content_type_override {
        if let Ok(header) = Header::from_bytes(b"content-type", content_type.as_bytes()) {
            tiny_response = tiny_response.with_header(header);
        }
    }
    request
        .respond(tiny_response)
        .context("failed to write proxy response")?;
    Ok(())
}

fn json_header() -> Header {
    Header::from_bytes(b"content-type", b"application/json").expect("static content-type header")
}

fn update_proxy_telemetry(
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    config: &ProxyConfig,
    request_path: &str,
    status: u16,
    body: &[u8],
    latency_ms: u64,
) {
    if status >= 400 || !(request_path == "/v1/responses" || request_path == "/v1/completions") {
        return;
    }
    let parsed = extract_usage_telemetry(body);
    let mut state = telemetry.lock().expect("proxy telemetry lock poisoned");
    state.active_model = parsed
        .as_ref()
        .and_then(|usage| usage.model.clone())
        .or_else(|| config.active_model.clone());
    state.last_request_path = Some(request_path.to_string());
    state.last_response_at = Some(iso_now());
    state.last_latency_ms = Some(latency_ms);
    if let Some(usage) = parsed {
        state.last_input_tokens = Some(usage.input_tokens);
        state.last_output_tokens = Some(usage.output_tokens);
        state.last_total_tokens = Some(usage.total_tokens);
        state.last_tokens_per_second = if latency_ms == 0 {
            None
        } else {
            Some((usage.output_tokens as f64) / ((latency_ms as f64) / 1000.0))
        };
    }
}

#[derive(Debug)]
struct UsageTelemetry {
    model: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
}

fn extract_usage_telemetry(body: &[u8]) -> Option<UsageTelemetry> {
    let text = String::from_utf8_lossy(body);
    if text.trim_start().starts_with("data: ") {
        return extract_usage_from_sse(&text);
    }
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    extract_usage_from_json(&value)
}

fn extract_usage_from_sse(sse: &str) -> Option<UsageTelemetry> {
    for line in sse.lines().rev() {
        let trimmed = line.trim();
        if !trimmed.starts_with("data: ") {
            continue;
        }
        let payload = trimmed.trim_start_matches("data: ").trim();
        if payload == "[DONE]" || payload.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(payload).ok()?;
        if value.get("type").and_then(serde_json::Value::as_str) == Some("response.completed") {
            return extract_usage_from_json(value.get("response")?);
        }
    }
    None
}

fn extract_usage_from_json(value: &serde_json::Value) -> Option<UsageTelemetry> {
    let usage = value.get("usage")?;
    let input_tokens = usage
        .get("input_tokens")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| usage.get("prompt_tokens").and_then(serde_json::Value::as_u64))
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| usage.get("completion_tokens").and_then(serde_json::Value::as_u64))
        .unwrap_or(0);
    let total_tokens = usage
        .get("total_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(input_tokens + output_tokens);
    Some(UsageTelemetry {
        model: value
            .get("model")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned),
        input_tokens,
        output_tokens,
        total_tokens,
    })
}

fn iso_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    chrono::DateTime::<chrono::Utc>::from_timestamp(now as i64, 0)
        .map(|timestamp| timestamp.to_rfc3339())
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string())
}

use anyhow::Context;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use tiny_http::Header;
use tiny_http::Method;
use tiny_http::Response;
use tiny_http::Server;
use tiny_http::StatusCode;

use crate::inference::engine;
use crate::inference::runtime_env;
use crate::inference::runtime_plan;
use crate::inference::web_search;

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
const OPENAI_RESPONSES_BASE_URL: &str = "https://api.openai.com";
const DEFAULT_BOOST_MINUTES: u64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarmonyRelayMode {
    Disabled,
    Json,
    Sse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalResponsesRelayMode {
    Disabled,
    NemotronChatJson,
    NemotronChatSse,
    QwenChatJson,
    QwenChatSse,
    GlmChatJson,
    GlmChatSse,
}

struct HarmonyRelayResponse {
    status_code: u16,
    response_headers: Vec<(String, String)>,
    body: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProxyConfig {
    pub root: PathBuf,
    pub listen_host: String,
    pub listen_port: u16,
    pub upstream_base_url: String,
    pub active_model: Option<String>,
    pub embedding_base_url: String,
    pub embedding_model: Option<String>,
    pub transcription_base_url: String,
    pub transcription_model: Option<String>,
    pub speech_base_url: String,
    pub speech_model: Option<String>,
}

impl ProxyConfig {
    pub fn from_env_with_root(root: &std::path::Path) -> Self {
        let active_model = runtime_env::effective_chat_model(root);
        let (embedding_base_url, embedding_model) =
            auxiliary_proxy_target(root, engine::AuxiliaryRole::Embedding, "EMBEDDING");
        let (transcription_base_url, transcription_model) =
            auxiliary_proxy_target(root, engine::AuxiliaryRole::Stt, "STT");
        let (speech_base_url, speech_model) =
            auxiliary_proxy_target(root, engine::AuxiliaryRole::Tts, "TTS");
        Self {
            root: root.to_path_buf(),
            listen_host: runtime_env::env_or_config(root, "CTOX_PROXY_HOST")
                .unwrap_or_else(|| "127.0.0.1".to_string()),
            listen_port: runtime_env::env_or_config(root, "CTOX_PROXY_PORT")
                .and_then(|value| value.parse().ok())
                .unwrap_or(12434),
            upstream_base_url: runtime_env::env_or_config(root, "CTOX_UPSTREAM_BASE_URL")
                .unwrap_or_else(|| match active_model.as_deref() {
                    Some(model) if engine::is_openai_api_chat_model(model) => {
                        OPENAI_RESPONSES_BASE_URL.to_string()
                    }
                    Some(model) => local_chat_upstream_base_url(root, model),
                    None => "http://127.0.0.1:1234".to_string(),
                }),
            active_model,
            embedding_base_url,
            embedding_model,
            transcription_base_url,
            transcription_model,
            speech_base_url,
            speech_model,
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

    pub fn routed_base_url(&self, request_url: &str) -> &str {
        match request_url {
            "/v1/embeddings" => &self.embedding_base_url,
            "/v1/audio/transcriptions" => &self.transcription_base_url,
            "/v1/audio/speech" | "/v1/audio/voices" => &self.speech_base_url,
            _ => &self.upstream_base_url,
        }
    }

    pub fn routed_model(&self, request_url: &str) -> Option<&str> {
        match request_url {
            "/v1/embeddings" => self.embedding_model.as_deref(),
            "/v1/audio/transcriptions" => self.transcription_model.as_deref(),
            "/v1/audio/speech" | "/v1/audio/voices" => self.speech_model.as_deref(),
            _ => self.active_model.as_deref(),
        }
    }

    pub fn join_routed_url(&self, request_url: &str) -> String {
        format!(
            "{}{}",
            self.routed_base_url(request_url).trim_end_matches('/'),
            request_url
        )
    }
}

fn local_chat_upstream_base_url(root: &std::path::Path, model: &str) -> String {
    let runtime_port = runtime_env::env_or_config(root, "CTOX_ENGINE_PORT")
        .and_then(|value| value.parse::<u16>().ok())
        .or_else(|| {
            engine::runtime_config_for_model(model)
                .ok()
                .map(|runtime| runtime.port)
        })
        .unwrap_or(1234);
    format!("http://127.0.0.1:{runtime_port}")
}

fn auxiliary_proxy_target(
    root: &std::path::Path,
    role: engine::AuxiliaryRole,
    role_prefix: &str,
) -> (String, Option<String>) {
    if !runtime_env::auxiliary_backend_enabled(root, role_prefix) {
        return (String::new(), None);
    }
    let selection = engine::auxiliary_model_selection(
        role,
        runtime_env::env_or_config(root, &format!("CTOX_{role_prefix}_MODEL")).as_deref(),
    );
    let base_url_key = format!("CTOX_{role_prefix}_BASE_URL");
    let port_key = format!("CTOX_{role_prefix}_PORT");
    let base_url = runtime_env::env_or_config(root, &base_url_key).unwrap_or_else(|| {
        format!(
            "http://127.0.0.1:{}",
            runtime_env::env_or_config(root, &port_key)
                .and_then(|value| value.parse::<u16>().ok())
                .unwrap_or(selection.default_port)
        )
    });
    (base_url, Some(selection.request_model.to_string()))
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
pub struct ProxyTelemetry {
    pub active_model: Option<String>,
    pub base_model: Option<String>,
    pub boost_model: Option<String>,
    pub boost_active: bool,
    pub boost_active_until_epoch: Option<u64>,
    pub boost_remaining_seconds: Option<u64>,
    pub boost_reason: Option<String>,
    pub upstream_base_url: Option<String>,
    pub last_known_good_model: Option<String>,
    pub backend_healthy: bool,
    pub last_switch_status: Option<String>,
    pub last_switch_error: Option<String>,
    pub recovery_count: u64,
    pub last_request_path: Option<String>,
    pub last_response_at: Option<String>,
    pub last_latency_ms: Option<u64>,
    pub last_input_tokens: Option<u64>,
    pub last_output_tokens: Option<u64>,
    pub last_total_tokens: Option<u64>,
    pub last_tokens_per_second: Option<f64>,
    pub load_observation_path: Option<String>,
    pub load_observation: Option<LoadObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LoadObservationGpu {
    pub gpu_index: usize,
    pub name: String,
    pub total_mb: u64,
    pub baseline_used_mb: u64,
    pub current_used_mb: u64,
    pub peak_used_mb: u64,
    pub final_used_mb: u64,
    pub current_delta_mb: u64,
    pub peak_delta_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LoadObservation {
    pub model: String,
    pub role: String,
    pub port: u16,
    pub startup_healthy: bool,
    pub sample_count: u64,
    pub started_at_epoch: u64,
    pub observed_until_epoch: u64,
    pub healthy_at_epoch: Option<u64>,
    pub gpus: Vec<LoadObservationGpu>,
}

#[derive(Debug, Clone, Deserialize)]
struct ProxySwitchRequest {
    model: String,
    #[serde(default)]
    preset: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoostStatus {
    pub active: bool,
    pub base_model: Option<String>,
    pub boost_model: Option<String>,
    pub active_model: Option<String>,
    pub active_until_epoch: Option<u64>,
    pub remaining_seconds: Option<u64>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProxySwitchResponse {
    ok: bool,
    active_model: String,
    upstream_base_url: String,
    rolled_back: bool,
    message: String,
}

#[derive(Debug, Clone)]
struct ProxyState {
    root: PathBuf,
    config: ProxyConfig,
    last_known_good: Option<ProxyConfig>,
    last_switch_error: Option<String>,
    recovery_count: u64,
}

#[derive(Debug, Clone, Default)]
struct BoostLeaseState {
    active: bool,
    expired: bool,
    base_model: Option<String>,
    boost_model: Option<String>,
    active_until_epoch: Option<u64>,
    remaining_seconds: Option<u64>,
    reason: Option<String>,
}

fn now_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn boost_lease_state_from_env(
    env_map: &BTreeMap<String, String>,
    active_model: Option<&str>,
) -> BoostLeaseState {
    let base_model = runtime_env::configured_chat_model_from_map(env_map);
    let boost_model = env_map
        .get("CTOX_CHAT_MODEL_BOOST")
        .cloned()
        .filter(|value| !value.trim().is_empty());
    let active_until_epoch = env_map
        .get("CTOX_BOOST_ACTIVE_UNTIL_EPOCH")
        .and_then(|value| value.trim().parse::<u64>().ok());
    let reason = env_map
        .get("CTOX_BOOST_REASON")
        .cloned()
        .filter(|value| !value.trim().is_empty());
    let now = now_epoch_seconds();
    let expired = active_until_epoch.is_some_and(|until| until <= now);
    let remaining_seconds = active_until_epoch
        .and_then(|until| until.checked_sub(now))
        .filter(|value| *value > 0);
    let active = remaining_seconds.is_some()
        && boost_model.is_some()
        && active_model
            .zip(boost_model.as_deref())
            .is_some_and(|(active_model, boost_model)| active_model.trim() == boost_model.trim());
    BoostLeaseState {
        active,
        expired,
        base_model,
        boost_model,
        active_until_epoch,
        remaining_seconds,
        reason,
    }
}

fn sync_boost_telemetry_fields(
    telemetry_state: &mut ProxyTelemetry,
    root: &std::path::Path,
    active_model: Option<&str>,
) {
    let env_map = runtime_env::load_runtime_env_map(root).unwrap_or_default();
    let boost = boost_lease_state_from_env(&env_map, active_model);
    telemetry_state.base_model = boost.base_model.clone();
    telemetry_state.boost_model = boost.boost_model.clone();
    telemetry_state.boost_active = boost.active;
    telemetry_state.boost_active_until_epoch = boost.active_until_epoch;
    telemetry_state.boost_remaining_seconds = boost.remaining_seconds;
    telemetry_state.boost_reason = boost.reason.clone();
}

pub fn serve_proxy(config: ProxyConfig) -> anyhow::Result<()> {
    if !config
        .upstream_base_url
        .starts_with(OPENAI_RESPONSES_BASE_URL)
    {
        ensure_backend_ready(&config.root, &config, false).with_context(|| {
            format!(
                "failed to prepare primary backend {} before starting proxy",
                config.active_model.as_deref().unwrap_or("unknown")
            )
        })?;
    }
    let server = Server::http(config.listen_addr())
        .map_err(|err| anyhow::anyhow!("failed to bind CTOX responses proxy: {err}"))?;
    let shared = Arc::new(Mutex::new(ProxyState {
        root: config.root.clone(),
        last_known_good: Some(config.clone()),
        last_switch_error: None,
        recovery_count: 0,
        config,
    }));
    let initial_config = {
        shared
            .lock()
            .expect("proxy state lock poisoned")
            .config
            .clone()
    };
    let telemetry = Arc::new(Mutex::new(ProxyTelemetry {
        active_model: initial_config.active_model.clone(),
        base_model: runtime_env::configured_chat_model(&initial_config.root),
        boost_model: runtime_env::env_or_config(&initial_config.root, "CTOX_CHAT_MODEL_BOOST"),
        boost_active: false,
        boost_active_until_epoch: None,
        boost_remaining_seconds: None,
        boost_reason: runtime_env::env_or_config(&initial_config.root, "CTOX_BOOST_REASON"),
        upstream_base_url: Some(initial_config.upstream_base_url.clone()),
        last_known_good_model: initial_config.active_model.clone(),
        backend_healthy: true,
        last_switch_status: Some("ready".to_string()),
        last_switch_error: None,
        recovery_count: 0,
        ..ProxyTelemetry::default()
    }));
    let response_state = Arc::new(Mutex::new(HashMap::<String, Value>::new()));

    for request in server.incoming_requests() {
        let config = Arc::clone(&shared);
        let telemetry = Arc::clone(&telemetry);
        let response_state = Arc::clone(&response_state);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle_request(&config, &telemetry, &response_state, request)
        })) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => eprintln!("ctox proxy error: {err}"),
            Err(panic_payload) => {
                let panic_message = if let Some(message) = panic_payload.downcast_ref::<&str>() {
                    (*message).to_string()
                } else if let Some(message) = panic_payload.downcast_ref::<String>() {
                    message.clone()
                } else {
                    "unknown panic".to_string()
                };
                eprintln!("ctox proxy request panic: {panic_message}");
            }
        }
    }

    Ok(())
}

fn handle_request(
    state: &Arc<Mutex<ProxyState>>,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    response_state: &Arc<Mutex<HashMap<String, Value>>>,
    mut request: tiny_http::Request,
) -> anyhow::Result<()> {
    maybe_expire_boost_lease(state, telemetry)?;
    if matches!(request.method(), Method::Get) && request.url() == "/ctox/telemetry" {
        let config = state
            .lock()
            .expect("proxy state lock poisoned")
            .config
            .clone();
        let backend_healthy = probe_upstream_health(&config);
        let snapshot = {
            let mut telemetry_state = telemetry.lock().expect("proxy telemetry lock poisoned");
            telemetry_state.active_model = config.active_model.clone();
            telemetry_state.upstream_base_url = Some(config.upstream_base_url.clone());
            telemetry_state.backend_healthy = backend_healthy;
            sync_boost_telemetry_fields(
                &mut telemetry_state,
                &config.root,
                config.active_model.as_deref(),
            );
            if !backend_healthy
                && telemetry_state
                    .last_switch_status
                    .as_deref()
                    .is_some_and(|status| matches!(status, "ready" | "switched" | "recovered"))
            {
                telemetry_state.last_switch_status = Some("backend_unhealthy".to_string());
            }
            telemetry_state.load_observation_path =
                load_observation_path(&config.root, &config.upstream_base_url)
                    .map(|path| path.display().to_string());
            telemetry_state.load_observation =
                read_load_observation(&config.root, &config.upstream_base_url);
            telemetry_state.clone()
        };
        let response = Response::from_string(serde_json::to_string(&snapshot)?)
            .with_status_code(StatusCode(200))
            .with_header(json_header());
        request
            .respond(response)
            .context("failed to write proxy telemetry response")?;
        return Ok(());
    }

    if matches!(request.method(), Method::Post) && request.url() == "/ctox/switch" {
        let mut body = Vec::new();
        request
            .as_reader()
            .read_to_end(&mut body)
            .context("failed to read proxy switch request body")?;
        let payload: ProxySwitchRequest =
            serde_json::from_slice(&body).context("failed to parse proxy switch request")?;
        let response = match switch_active_model(
            state,
            telemetry,
            &payload.model,
            payload.preset.as_deref(),
        ) {
            Ok(result) => Response::from_string(serde_json::to_string(&result)?)
                .with_status_code(StatusCode(200))
                .with_header(json_header()),
            Err(err) => Response::from_string(
                serde_json::json!({
                    "ok": false,
                    "error": { "message": err.to_string() }
                })
                .to_string(),
            )
            .with_status_code(StatusCode(502))
            .with_header(json_header()),
        };
        request
            .respond(response)
            .context("failed to write proxy switch response")?;
        return Ok(());
    }

    let started = Instant::now();
    let method = request.method().as_str().to_string();
    let url = request.url().to_string();
    let config = state
        .lock()
        .expect("proxy state lock poisoned")
        .config
        .clone();
    let mut body = Vec::new();
    request
        .as_reader()
        .read_to_end(&mut body)
        .context("failed to read proxy request body")?;

    let mut materialized_request = if matches!(request.method(), Method::Post)
        && url == "/v1/responses"
        && !body.is_empty()
    {
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
        previous_conversation
            .as_ref()
            .map(|conversation| engine::materialize_responses_request(&body, Some(conversation)))
            .transpose()?
    } else {
        None
    };

    let mut effective_body = materialized_request
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()?
        .unwrap_or_else(|| body.clone());
    effective_body = if matches!(request.method(), Method::Post) && !effective_body.is_empty() {
        rewrite_auxiliary_request_body(&config, &url, &effective_body)
    } else {
        effective_body
    };
    let use_openai_passthrough = matches!(request.method(), Method::Post)
        && url == "/v1/responses"
        && !body.is_empty()
        && config
            .upstream_base_url
            .starts_with(OPENAI_RESPONSES_BASE_URL);
    let mut web_search_augmentation = None;
    if matches!(request.method(), Method::Post)
        && url == "/v1/responses"
        && !effective_body.is_empty()
        && !use_openai_passthrough
    {
        if let Ok(mut payload) = serde_json::from_slice::<Value>(&effective_body) {
            web_search_augmentation =
                web_search::augment_responses_request(&config.root, &mut payload)?;
            if web_search_augmentation.is_some() {
                effective_body = serde_json::to_vec(&payload)
                    .context("failed to encode web-search-augmented request")?;
                materialized_request = Some(payload);
            }
        }
    }

    let use_gpt_oss_harmony_proxy = matches!(request.method(), Method::Post)
        && url == "/v1/responses"
        && !body.is_empty()
        && engine::should_use_gpt_oss_harmony_proxy(&effective_body)?;
    let harmony_relay_mode = if use_gpt_oss_harmony_proxy {
        if engine::responses_request_streams(&effective_body)? {
            HarmonyRelayMode::Sse
        } else {
            HarmonyRelayMode::Json
        }
    } else {
        HarmonyRelayMode::Disabled
    };
    let use_qwen_chat_proxy = matches!(request.method(), Method::Post)
        && url == "/v1/responses"
        && !body.is_empty()
        && config
            .active_model
            .as_deref()
            .map(engine::is_qwen_chat_model_id)
            .unwrap_or(false);
    let use_nemotron_chat_proxy = matches!(request.method(), Method::Post)
        && url == "/v1/responses"
        && !body.is_empty()
        && config
            .active_model
            .as_deref()
            .map(engine::is_nemotron_chat_model_id)
            .unwrap_or(false);
    let use_glm_chat_proxy = matches!(request.method(), Method::Post)
        && url == "/v1/responses"
        && !body.is_empty()
        && config
            .active_model
            .as_deref()
            .map(engine::is_glm_chat_model_id)
            .unwrap_or(false);
    let local_responses_relay_mode = if use_nemotron_chat_proxy {
        if engine::responses_request_streams(&effective_body)? {
            LocalResponsesRelayMode::NemotronChatSse
        } else {
            LocalResponsesRelayMode::NemotronChatJson
        }
    } else if use_qwen_chat_proxy {
        if engine::responses_request_streams(&effective_body)? {
            LocalResponsesRelayMode::QwenChatSse
        } else {
            LocalResponsesRelayMode::QwenChatJson
        }
    } else if use_glm_chat_proxy {
        if engine::responses_request_streams(&effective_body)? {
            LocalResponsesRelayMode::GlmChatSse
        } else {
            LocalResponsesRelayMode::GlmChatJson
        }
    } else {
        LocalResponsesRelayMode::Disabled
    };
    eprintln!(
        "ctox proxy request method={} url={} harmony_proxy={}",
        method, url, use_gpt_oss_harmony_proxy
    );

    if matches!(
        url.as_str(),
        "/v1/embeddings" | "/v1/audio/transcriptions" | "/v1/audio/speech" | "/v1/audio/voices"
    ) {
        if auxiliary_backend_spec(&config, &url).is_none() {
            let response = Response::from_string(
                serde_json::json!({
                    "error": {
                        "message": format!("auxiliary backend for {} is disabled in this runtime", url)
                    }
                })
                .to_string(),
            )
            .with_status_code(StatusCode(503))
            .with_header(json_header());
            request
                .respond(response)
                .context("failed to write auxiliary backend disabled response")?;
            return Ok(());
        }
        if let Err(err) = ensure_auxiliary_backend_ready(&config, &url) {
            let response = Response::from_string(
                serde_json::json!({
                    "error": { "message": err.to_string() }
                })
                .to_string(),
            )
            .with_status_code(StatusCode(502))
            .with_header(json_header());
            request
                .respond(response)
                .context("failed to write auxiliary backend error response")?;
            return Ok(());
        }
    }

    let forwarded_body = if use_gpt_oss_harmony_proxy {
        engine::rewrite_responses_to_gpt_oss_completion(&effective_body)?
    } else if use_nemotron_chat_proxy {
        let rewritten = engine::rewrite_responses_to_nemotron_chat_completions(&effective_body)?;
        if let Ok(value) = serde_json::from_slice::<Value>(&rewritten) {
            if let Some(messages) = value.get("messages").and_then(Value::as_array) {
                let roles = messages
                    .iter()
                    .map(|message| {
                        message
                            .get("role")
                            .and_then(Value::as_str)
                            .unwrap_or("?")
                            .to_string()
                    })
                    .collect::<Vec<_>>();
                eprintln!("ctox proxy nemotron message roles={roles:?}");
            }
        }
        rewritten
    } else if use_qwen_chat_proxy {
        let rewritten = engine::rewrite_responses_to_qwen_chat_completions(&effective_body)?;
        if let Ok(value) = serde_json::from_slice::<Value>(&rewritten) {
            if let Some(messages) = value.get("messages").and_then(Value::as_array) {
                let roles = messages
                    .iter()
                    .map(|message| {
                        message
                            .get("role")
                            .and_then(Value::as_str)
                            .unwrap_or("?")
                            .to_string()
                    })
                    .collect::<Vec<_>>();
                eprintln!("ctox proxy qwen message roles={roles:?}");
            }
        }
        rewritten
    } else if use_glm_chat_proxy {
        let rewritten = engine::rewrite_responses_to_glm_chat_completions(&effective_body)?;
        if let Ok(value) = serde_json::from_slice::<Value>(&rewritten) {
            if let Some(messages) = value.get("messages").and_then(Value::as_array) {
                let roles = messages
                    .iter()
                    .map(|message| {
                        message
                            .get("role")
                            .and_then(Value::as_str)
                            .unwrap_or("?")
                            .to_string()
                    })
                    .collect::<Vec<_>>();
                eprintln!("ctox proxy glm message roles={roles:?}");
            }
        }
        rewritten
    } else if use_openai_passthrough {
        engine::rewrite_openai_responses_request(&body)?
    } else if matches!(request.method(), Method::Post) && url == "/v1/responses" && !body.is_empty()
    {
        engine::rewrite_engine_responses_request(&effective_body)?
    } else if matches!(request.method(), Method::Post) && !body.is_empty() {
        rewrite_auxiliary_request_body(&config, &url, &body)
    } else {
        body
    };

    let upstream_path = if use_gpt_oss_harmony_proxy {
        "/v1/completions".to_string()
    } else if use_nemotron_chat_proxy || use_qwen_chat_proxy || use_glm_chat_proxy {
        "/v1/chat/completions".to_string()
    } else {
        url.clone()
    };
    eprintln!("ctox proxy upstream_path={upstream_path}");
    if use_gpt_oss_harmony_proxy {
        let forwarded_text = String::from_utf8_lossy(&forwarded_body);
        let preview: String = forwarded_text.chars().take(2_000).collect();
        eprintln!(
            "ctox proxy forwarded harmony request bytes={} preview={}",
            forwarded_body.len(),
            preview
        );
    }
    let upstream_url = if use_gpt_oss_harmony_proxy {
        config.join_url(&upstream_path)
    } else {
        config.join_routed_url(&upstream_path)
    };
    let targets_primary_local_chat_backend =
        matches!(request.method(), Method::Post)
            && !use_openai_passthrough
            && !matches!(
                url.as_str(),
                "/v1/embeddings" | "/v1/audio/transcriptions" | "/v1/audio/speech" | "/v1/audio/voices"
            );
    if targets_primary_local_chat_backend {
        if let Err(err) = ensure_backend_ready(&config.root, &config, false) {
            let response = Response::from_string(
                serde_json::json!({
                    "error": { "message": err.to_string() }
                })
                .to_string(),
            )
            .with_status_code(StatusCode(502))
            .with_header(json_header());
            request
                .respond(response)
                .context("failed to write primary backend error response")?;
            return Ok(());
        }
    }
    if matches!(request.method(), Method::Post) && url == "/v1/audio/transcriptions" {
        let content_type = request
            .headers()
            .iter()
            .find(|header| {
                header
                    .field
                    .as_str()
                    .as_str()
                    .eq_ignore_ascii_case("content-type")
            })
            .map(|header| header.value.as_str().to_string())
            .unwrap_or_else(|| "multipart/form-data".to_string());
        return relay_transcription_via_curl(
            telemetry,
            request,
            &upstream_url,
            &content_type,
            forwarded_body,
            &url,
            started,
        );
    }
    if use_gpt_oss_harmony_proxy {
        return relay_gpt_oss_harmony_response(
            state,
            &config,
            telemetry,
            response_state,
            request,
            &method,
            &upstream_url,
            materialized_request,
            forwarded_body,
            harmony_relay_mode,
            web_search_augmentation.as_ref(),
            &url,
            started,
        );
    }

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
        // Local engine backends are sensitive to extra forwarded client headers
        // on /v1/responses. Keep the local bridge minimal and only preserve the
        // content type needed to parse the JSON payload. OpenAI passthrough keeps
        // the broader upstream-compatible header set.
        if !use_openai_passthrough && !field.eq_ignore_ascii_case("content-type") {
            continue;
        }
        upstream = upstream.set(field, header.value.as_str());
    }
    if config
        .upstream_base_url
        .starts_with(OPENAI_RESPONSES_BASE_URL)
        && request.headers().iter().all(|header| {
            !header
                .field
                .as_str()
                .as_str()
                .eq_ignore_ascii_case("authorization")
        })
    {
        if let Some(api_key) = runtime_env::env_or_config(&config.root, "OPENAI_API_KEY") {
            upstream = upstream.set("authorization", &format!("Bearer {api_key}"));
        }
    }

    let upstream_response = if forwarded_body.is_empty() {
        upstream.call()
    } else {
        upstream.send_bytes(&forwarded_body)
    };

    match upstream_response {
        Ok(response) => relay_response(
            &config,
            telemetry,
            request,
            response,
            harmony_relay_mode,
            local_responses_relay_mode,
            web_search_augmentation.as_ref(),
            &url,
            started,
        ),
        Err(ureq::Error::Status(_, response)) => relay_response(
            &config,
            telemetry,
            request,
            response,
            harmony_relay_mode,
            local_responses_relay_mode,
            web_search_augmentation.as_ref(),
            &url,
            started,
        ),
        Err(err) => {
            let _ =
                attempt_auto_recovery(state, telemetry, &format!("upstream request failed: {err}"));
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

fn relay_transcription_via_curl(
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    request: tiny_http::Request,
    upstream_url: &str,
    content_type: &str,
    body: Vec<u8>,
    request_path: &str,
    started: Instant,
) -> anyhow::Result<()> {
    let sentinel = "\n__CTOX_HTTP_STATUS__:";
    let mut child = Command::new("curl")
        .arg("-sS")
        .arg("--max-time")
        .arg("120")
        .arg("-X")
        .arg("POST")
        .arg("-H")
        .arg(format!("content-type: {content_type}"))
        .arg("--data-binary")
        .arg("@-")
        .arg("-w")
        .arg(format!("{sentinel}%{{http_code}}"))
        .arg(upstream_url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to launch curl for transcription relay")?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&body)
            .context("failed to write transcription relay request body")?;
    }

    let output = child
        .wait_with_output()
        .context("failed to wait for transcription relay curl")?;

    if !output.status.success() {
        let response = Response::from_string(
            serde_json::json!({
                "error": {
                    "message": String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            })
            .to_string(),
        )
        .with_status_code(StatusCode(502))
        .with_header(json_header());
        request
            .respond(response)
            .context("failed to write transcription relay error response")?;
        return Ok(());
    }

    let stdout = output.stdout;
    let stdout_text = String::from_utf8_lossy(&stdout);
    let Some(marker_index) = stdout_text.rfind(sentinel) else {
        let response = Response::from_string(
            serde_json::json!({
                "error": {
                    "message": "transcription relay did not return an HTTP status marker"
                }
            })
            .to_string(),
        )
        .with_status_code(StatusCode(502))
        .with_header(json_header());
        request
            .respond(response)
            .context("failed to write transcription relay marker error response")?;
        return Ok(());
    };

    let body_bytes = stdout[..marker_index].to_vec();
    let status_code = stdout_text[marker_index + sentinel.len()..]
        .trim()
        .parse::<u16>()
        .unwrap_or(502);

    update_proxy_telemetry(
        telemetry,
        &ProxyConfig {
            root: PathBuf::new(),
            listen_host: String::new(),
            listen_port: 0,
            upstream_base_url: String::new(),
            active_model: None,
            embedding_base_url: String::new(),
            embedding_model: None,
            transcription_base_url: String::new(),
            transcription_model: None,
            speech_base_url: String::new(),
            speech_model: None,
        },
        request_path,
        status_code,
        &body_bytes,
        started.elapsed().as_millis() as u64,
    );

    let response = Response::from_data(body_bytes)
        .with_status_code(StatusCode(status_code))
        .with_header(json_header());
    request
        .respond(response)
        .context("failed to write transcription relay response")?;
    Ok(())
}

fn relay_gpt_oss_harmony_response(
    state: &Arc<Mutex<ProxyState>>,
    config: &ProxyConfig,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    response_state: &Arc<Mutex<HashMap<String, Value>>>,
    request: tiny_http::Request,
    method: &str,
    upstream_url: &str,
    materialized_request: Option<Value>,
    forwarded_body: Vec<u8>,
    harmony_relay_mode: HarmonyRelayMode,
    web_search_augmentation: Option<&web_search::WebSearchAugmentation>,
    request_path: &str,
    started: Instant,
) -> anyhow::Result<()> {
    eprintln!(
        "ctox proxy harmony relay start mode={:?} request_path={request_path}",
        harmony_relay_mode
    );
    let outcome = complete_gpt_oss_harmony_roundtrip(
        state,
        config,
        telemetry,
        response_state,
        method,
        upstream_url,
        materialized_request,
        forwarded_body,
        web_search_augmentation,
    )?;
    eprintln!(
        "ctox proxy harmony relay roundtrip complete status={} body_bytes={}",
        outcome.status_code,
        outcome.body.len()
    );

    eprintln!("ctox proxy harmony relay about to emit downstream response");
    relay_response_from_parts(
        config,
        telemetry,
        request,
        outcome.status_code,
        outcome.response_headers,
        outcome.body,
        harmony_relay_mode,
        LocalResponsesRelayMode::Disabled,
        web_search_augmentation,
        request_path,
        started,
    )?;
    eprintln!("ctox proxy harmony relay downstream response emitted");
    Ok(())
}

fn complete_gpt_oss_harmony_roundtrip(
    state: &Arc<Mutex<ProxyState>>,
    config: &ProxyConfig,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    response_state: &Arc<Mutex<HashMap<String, Value>>>,
    method: &str,
    upstream_url: &str,
    materialized_request: Option<Value>,
    forwarded_body: Vec<u8>,
    web_search_augmentation: Option<&web_search::WebSearchAugmentation>,
) -> anyhow::Result<HarmonyRelayResponse> {
    eprintln!("ctox proxy harmony roundtrip sending first upstream request");
    let agent = ureq::AgentBuilder::new().build();
    let mut outcome = send_harmony_upstream_request_with_retry(
        &agent,
        config,
        method,
        upstream_url,
        config.active_model.as_deref(),
        &forwarded_body,
    )
    .map_err(|err| {
        let _ = attempt_auto_recovery(state, telemetry, &format!("harmony upstream failed: {err}"));
        anyhow::anyhow!("harmony upstream failed: {err}")
    })?;
    eprintln!(
        "ctox proxy harmony first upstream response status={} body_bytes={}",
        outcome.status_code,
        outcome.body.len()
    );

    if outcome.status_code < 400 {
        if let Some(followup_body) =
            engine::build_gpt_oss_followup_completion_request(&forwarded_body, &outcome.body)?
        {
            eprintln!("ctox proxy issuing GPT-OSS continuation completion");
            outcome = send_harmony_upstream_request_with_retry(
                &agent,
                config,
                method,
                upstream_url,
                config.active_model.as_deref(),
                &followup_body,
            )
            .map_err(|err| {
                let _ = attempt_auto_recovery(
                    state,
                    telemetry,
                    &format!("harmony continuation failed: {err}"),
                );
                anyhow::anyhow!("harmony continuation failed: {err}")
            })?;
            eprintln!(
                "ctox proxy continuation harmony body={}",
                String::from_utf8_lossy(&outcome.body)
            );
            eprintln!(
                "ctox proxy harmony continuation response status={} body_bytes={}",
                outcome.status_code,
                outcome.body.len()
            );
        }
    }

    if outcome.status_code < 400 {
        store_harmony_proxy_response_state(
            response_state,
            materialized_request.as_ref(),
            &outcome.body,
            web_search_augmentation,
        )?;
    }

    Ok(outcome)
}

fn send_harmony_upstream_request_with_retry(
    agent: &ureq::Agent,
    config: &ProxyConfig,
    method: &str,
    upstream_url: &str,
    active_model: Option<&str>,
    body: &[u8],
) -> anyhow::Result<HarmonyRelayResponse> {
    if !config
        .upstream_base_url
        .starts_with(OPENAI_RESPONSES_BASE_URL)
    {
        return send_local_harmony_upstream_request_with_retry(
            config,
            method,
            upstream_url,
            active_model,
            body,
        );
    }
    let retry_deadline =
        Instant::now() + Duration::from_secs(harmony_upstream_startup_retry_secs(config));
    loop {
        match send_harmony_upstream_request(agent, method, upstream_url, active_model, body) {
            Err(err)
                if should_retry_harmony_upstream_connect(&err)
                    && Instant::now() < retry_deadline =>
            {
                thread::sleep(Duration::from_millis(500));
            }
            Ok(response) => {
                return harmony_relay_response_from_ureq(response)
                    .context("failed to read upstream harmony response");
            }
            Err(ureq::Error::Status(_, response)) => {
                return harmony_relay_response_from_ureq(response)
                    .context("failed to read upstream harmony error response");
            }
            Err(err) => return Err(anyhow::anyhow!(err.to_string())),
        }
    }
}

fn send_local_harmony_upstream_request_with_retry(
    config: &ProxyConfig,
    method: &str,
    upstream_url: &str,
    active_model: Option<&str>,
    body: &[u8],
) -> anyhow::Result<HarmonyRelayResponse> {
    let retry_deadline =
        Instant::now() + Duration::from_secs(harmony_upstream_startup_retry_secs(config));
    loop {
        match send_local_harmony_upstream_request(method, upstream_url, active_model, body) {
            Err(err)
                if should_retry_harmony_local_connect(&err)
                    && Instant::now() < retry_deadline =>
            {
                thread::sleep(Duration::from_millis(500));
            }
            other => return other,
        }
    }
}

fn send_local_harmony_upstream_request(
    method: &str,
    upstream_url: &str,
    active_model: Option<&str>,
    body: &[u8],
) -> anyhow::Result<HarmonyRelayResponse> {
    let started = Instant::now();
    let sentinel = "\n__CTOX_HTTP_STATUS__:";
    eprintln!(
        "ctox proxy local harmony upstream spawn method={} url={} body_bytes={}",
        method,
        upstream_url,
        body.len()
    );
    let mut command = Command::new("curl");
    command
        .arg("-sS")
        .arg("--max-time")
        .arg("240")
        .arg("-X")
        .arg(method)
        .arg("-H")
        .arg("content-type: application/json")
        .arg("--data-binary")
        .arg("@-")
        .arg("-w")
        .arg(format!("{sentinel}%{{http_code}}"))
        .arg(upstream_url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(active_model) = active_model {
        command.arg("-H").arg(format!("x-ctox-active-model: {active_model}"));
    }
    let mut child = command
        .spawn()
        .context("failed to launch curl for local harmony upstream")?;
    eprintln!("ctox proxy local harmony upstream curl pid={}", child.id());
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(body)
            .context("failed to write local harmony upstream request body")?;
        stdin
            .flush()
            .context("failed to flush local harmony upstream request body")?;
        drop(stdin);
        eprintln!("ctox proxy local harmony upstream request body sent");
    }
    eprintln!("ctox proxy local harmony upstream waiting for curl output");
    let output = child
        .wait_with_output()
        .context("failed to wait for local harmony upstream curl")?;
    eprintln!(
        "ctox proxy local harmony upstream curl finished rc={:?} elapsed_ms={}",
        output.status.code(),
        started.elapsed().as_millis()
    );
    if !output.status.success() {
        eprintln!(
            "ctox proxy local harmony upstream curl stderr={}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        anyhow::bail!(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let stdout_text = String::from_utf8_lossy(&output.stdout);
    let Some(marker_index) = stdout_text.rfind(sentinel) else {
        anyhow::bail!("local harmony upstream did not return an HTTP status marker");
    };
    let status_code = stdout_text[marker_index + sentinel.len()..]
        .trim()
        .parse::<u16>()
        .unwrap_or(502);
    eprintln!(
        "ctox proxy local harmony upstream parsed status={} response_body_bytes={}",
        status_code,
        marker_index
    );
    Ok(HarmonyRelayResponse {
        status_code,
        response_headers: Vec::new(),
        body: output.stdout[..marker_index].to_vec(),
    })
}

fn should_retry_harmony_local_connect(err: &anyhow::Error) -> bool {
    let text = err.to_string();
    text.contains("Connection refused")
        || text.contains("Failed to connect")
        || text.contains("Could not connect")
        || text.contains("Connection reset by peer")
}

fn harmony_upstream_startup_retry_secs(config: &ProxyConfig) -> u64 {
    if config
        .upstream_base_url
        .starts_with(OPENAI_RESPONSES_BASE_URL)
    {
        return 0;
    }
    backend_startup_wait_secs(config).min(30)
}

fn should_retry_harmony_upstream_connect(err: &ureq::Error) -> bool {
    match err {
        ureq::Error::Transport(transport) => {
            let text = transport.to_string();
            text.contains("Connection refused")
                || text.contains("Connect error")
                || text.contains("Connection reset by peer")
        }
        _ => false,
    }
}

fn send_harmony_upstream_request(
    agent: &ureq::Agent,
    method: &str,
    upstream_url: &str,
    active_model: Option<&str>,
    body: &[u8],
) -> Result<ureq::Response, ureq::Error> {
    let mut request = agent.request(method, upstream_url);
    request = request.set("content-type", "application/json");
    if let Some(active_model) = active_model {
        request = request.set("x-ctox-active-model", active_model);
    }
    if body.is_empty() {
        request.call()
    } else {
        request.send_bytes(body)
    }
}

fn harmony_relay_response_from_ureq(
    response: ureq::Response,
) -> anyhow::Result<HarmonyRelayResponse> {
    let status_code = response.status();
    let response_headers = response
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
        .context("failed to read harmony response body")?;
    Ok(HarmonyRelayResponse {
        status_code,
        response_headers,
        body,
    })
}

fn store_harmony_proxy_response_state(
    response_state: &Arc<Mutex<HashMap<String, Value>>>,
    materialized_request: Option<&Value>,
    completion_body: &[u8],
    web_search_augmentation: Option<&web_search::WebSearchAugmentation>,
) -> anyhow::Result<()> {
    let Some(materialized_request) = materialized_request else {
        return Ok(());
    };
    let mut response_payload: Value = serde_json::from_slice(
        &engine::rewrite_gpt_oss_completion_to_responses(completion_body, None)?,
    )
    .context("failed to parse rewritten responses payload for proxy state")?;
    if let Some(augmentation) = web_search_augmentation {
        response_payload = serde_json::from_slice(&web_search::augment_responses_output(
            &serde_json::to_vec(&response_payload)?,
            augmentation,
        )?)
        .context("failed to parse augmented web-search responses payload for proxy state")?;
    }
    if let Some(response_id) = response_payload.get("id").and_then(Value::as_str) {
        let conversation =
            engine::extend_conversation_with_response(materialized_request, &response_payload)?;
        response_state
            .lock()
            .expect("proxy response state lock poisoned")
            .insert(response_id.to_string(), conversation);
    }
    Ok(())
}

fn build_harmony_terminal_sse(
    completion_body: &[u8],
    web_search_augmentation: Option<&web_search::WebSearchAugmentation>,
    status_code: u16,
) -> anyhow::Result<Vec<u8>> {
    if status_code >= 400 {
        return Ok(sse_error_frame(&extract_harmony_error_message(
            completion_body,
        )));
    }
    let json_payload = engine::rewrite_gpt_oss_completion_to_responses(completion_body, None)?;
    let json_payload = if let Some(augmentation) = web_search_augmentation {
        web_search::augment_responses_output(&json_payload, augmentation)?
    } else {
        json_payload
    };
    engine::rewrite_responses_payload_to_sse(&json_payload)
}

fn extract_harmony_error_message(body: &[u8]) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| String::from_utf8_lossy(body).trim().to_string())
}

fn sse_error_frame(message: &str) -> Vec<u8> {
    format!(
        "event: error\ndata: {}\n\ndata: [DONE]\n\n",
        serde_json::json!({
            "error": {
                "message": message
            }
        })
    )
    .into_bytes()
}

fn maybe_expire_boost_lease(
    state: &Arc<Mutex<ProxyState>>,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
) -> anyhow::Result<()> {
    let (root, active_model) = {
        let guard = state.lock().expect("proxy state lock poisoned");
        (guard.root.clone(), guard.config.active_model.clone())
    };
    let env_map = runtime_env::load_runtime_env_map(&root).unwrap_or_default();
    let boost = boost_lease_state_from_env(&env_map, active_model.as_deref());
    {
        let mut telemetry_state = telemetry.lock().expect("proxy telemetry lock poisoned");
        sync_boost_telemetry_fields(&mut telemetry_state, &root, active_model.as_deref());
    }
    if !boost.expired {
        return Ok(());
    }
    let Some(base_model) = boost.base_model.as_deref() else {
        return Ok(());
    };
    let mut next_env = env_map.clone();
    next_env.remove("CTOX_BOOST_ACTIVE_UNTIL_EPOCH");
    next_env.remove("CTOX_BOOST_REASON");
    runtime_env::save_runtime_env_map(&root, &next_env)?;
    let _ = switch_active_model(state, telemetry, base_model, None)?;
    {
        let mut telemetry_state = telemetry.lock().expect("proxy telemetry lock poisoned");
        telemetry_state.last_switch_status = Some("boost_expired".to_string());
        sync_boost_telemetry_fields(&mut telemetry_state, &root, Some(base_model));
    }
    Ok(())
}

pub fn boost_status(root: &std::path::Path) -> anyhow::Result<BoostStatus> {
    let telemetry = fetch_proxy_telemetry(root)?;
    Ok(BoostStatus {
        active: telemetry.boost_active,
        base_model: telemetry.base_model,
        boost_model: telemetry.boost_model,
        active_model: telemetry.active_model,
        active_until_epoch: telemetry.boost_active_until_epoch,
        remaining_seconds: telemetry.boost_remaining_seconds,
        reason: telemetry.boost_reason,
    })
}

pub fn start_boost_lease(
    root: &std::path::Path,
    model_override: Option<&str>,
    minutes_override: Option<u64>,
    reason: Option<&str>,
) -> anyhow::Result<BoostStatus> {
    let mut env_map = runtime_env::load_runtime_env_map(root).unwrap_or_default();
    let base_model = runtime_env::configured_chat_model_from_map(&env_map)
        .or_else(|| runtime_env::effective_chat_model(root))
        .unwrap_or_else(|| "openai/gpt-oss-20b".to_string());
    let boost_model = model_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            env_map
                .get("CTOX_CHAT_MODEL_BOOST")
                .cloned()
                .filter(|value| !value.trim().is_empty())
        })
        .context("boost start requires CTOX_CHAT_MODEL_BOOST or --model")?;
    let default_minutes = env_map
        .get("CTOX_BOOST_DEFAULT_MINUTES")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_BOOST_MINUTES);
    let minutes = minutes_override
        .filter(|value| *value > 0)
        .unwrap_or(default_minutes);
    env_map.insert("CTOX_CHAT_MODEL_BASE".to_string(), base_model);
    env_map.insert("CTOX_CHAT_MODEL_BOOST".to_string(), boost_model.clone());
    env_map.insert(
        "CTOX_BOOST_DEFAULT_MINUTES".to_string(),
        default_minutes.to_string(),
    );
    env_map.insert(
        "CTOX_BOOST_ACTIVE_UNTIL_EPOCH".to_string(),
        (now_epoch_seconds() + minutes.saturating_mul(60)).to_string(),
    );
    if let Some(reason) = reason.map(str::trim).filter(|value| !value.is_empty()) {
        env_map.insert("CTOX_BOOST_REASON".to_string(), reason.to_string());
    } else {
        env_map.remove("CTOX_BOOST_REASON");
    }
    runtime_env::save_runtime_env_map(root, &env_map)?;
    let _ = request_proxy_switch(root, &boost_model, None)?;
    boost_status(root)
}

pub fn stop_boost_lease(root: &std::path::Path) -> anyhow::Result<BoostStatus> {
    let mut env_map = runtime_env::load_runtime_env_map(root).unwrap_or_default();
    let base_model = runtime_env::configured_chat_model_from_map(&env_map)
        .or_else(|| runtime_env::effective_chat_model(root))
        .unwrap_or_else(|| "openai/gpt-oss-20b".to_string());
    env_map.remove("CTOX_BOOST_ACTIVE_UNTIL_EPOCH");
    env_map.remove("CTOX_BOOST_REASON");
    runtime_env::save_runtime_env_map(root, &env_map)?;
    let _ = request_proxy_switch(root, &base_model, None)?;
    boost_status(root)
}

fn request_proxy_switch(
    root: &std::path::Path,
    model: &str,
    preset: Option<&str>,
) -> anyhow::Result<ProxySwitchResponse> {
    let host = runtime_env::env_or_config(root, "CTOX_PROXY_HOST")
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port =
        runtime_env::env_or_config(root, "CTOX_PROXY_PORT").unwrap_or_else(|| "12434".to_string());
    let url = format!("http://{host}:{port}/ctox/switch");
    let payload = match preset.map(str::trim).filter(|value| !value.is_empty()) {
        Some(preset) => serde_json::json!({ "model": model, "preset": preset }),
        None => serde_json::json!({ "model": model }),
    };
    let response = ureq::post(&url)
        .set("content-type", "application/json")
        .send_string(&payload.to_string())
        .with_context(|| format!("failed to reach proxy switch endpoint at {url}"))?;
    let body = response
        .into_string()
        .context("failed to read proxy switch response body")?;
    serde_json::from_str(&body).context("failed to parse proxy switch response")
}

fn fetch_proxy_telemetry(root: &std::path::Path) -> anyhow::Result<ProxyTelemetry> {
    let host = runtime_env::env_or_config(root, "CTOX_PROXY_HOST")
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port =
        runtime_env::env_or_config(root, "CTOX_PROXY_PORT").unwrap_or_else(|| "12434".to_string());
    let url = format!("http://{host}:{port}/ctox/telemetry");
    let response = ureq::get(&url)
        .call()
        .with_context(|| format!("failed to reach proxy telemetry endpoint at {url}"))?;
    let body = response
        .into_string()
        .context("failed to read proxy telemetry body")?;
    serde_json::from_str(&body).context("failed to parse proxy telemetry body")
}

fn relay_response(
    config: &ProxyConfig,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    request: tiny_http::Request,
    response: ureq::Response,
    harmony_relay_mode: HarmonyRelayMode,
    local_responses_relay_mode: LocalResponsesRelayMode,
    web_search_augmentation: Option<&web_search::WebSearchAugmentation>,
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
        local_responses_relay_mode,
        web_search_augmentation,
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
    local_responses_relay_mode: LocalResponsesRelayMode,
    web_search_augmentation: Option<&web_search::WebSearchAugmentation>,
    request_path: &str,
    started: Instant,
) -> anyhow::Result<()> {
    let status = StatusCode(status_code);
    eprintln!(
        "ctox proxy relay_response_from_parts status={} harmony_mode={:?} local_mode={:?} body_bytes={}",
        status.0,
        harmony_relay_mode,
        local_responses_relay_mode,
        body.len()
    );
    if !matches!(
        local_responses_relay_mode,
        LocalResponsesRelayMode::Disabled
    ) {
        eprintln!(
            "ctox proxy qwen upstream status={} body={}",
            status.0,
            String::from_utf8_lossy(&body)
        );
    }
    let mut content_type_override: Option<&'static str> = None;
    let mut body = match harmony_relay_mode {
        HarmonyRelayMode::Disabled => match local_responses_relay_mode {
            LocalResponsesRelayMode::Disabled => body,
            LocalResponsesRelayMode::NemotronChatJson if status.0 < 400 => {
                eprintln!(
                    "ctox proxy nemotron upstream body={}",
                    String::from_utf8_lossy(&body)
                );
                content_type_override = Some("application/json");
                engine::rewrite_nemotron_chat_completions_to_responses(
                    &body,
                    config.active_model.as_deref(),
                )?
            }
            LocalResponsesRelayMode::NemotronChatSse if status.0 < 400 => {
                eprintln!(
                    "ctox proxy nemotron upstream body={}",
                    String::from_utf8_lossy(&body)
                );
                let json_payload = engine::rewrite_nemotron_chat_completions_to_responses(
                    &body,
                    config.active_model.as_deref(),
                )?;
                let json_payload = if let Some(augmentation) = web_search_augmentation {
                    web_search::augment_responses_output(&json_payload, augmentation)?
                } else {
                    json_payload
                };
                content_type_override = Some("text/event-stream");
                engine::rewrite_responses_payload_to_sse(&json_payload)?
            }
            LocalResponsesRelayMode::QwenChatJson if status.0 < 400 => {
                eprintln!(
                    "ctox proxy qwen upstream body={}",
                    String::from_utf8_lossy(&body)
                );
                content_type_override = Some("application/json");
                engine::rewrite_qwen_chat_completions_to_responses(
                    &body,
                    config.active_model.as_deref(),
                )?
            }
            LocalResponsesRelayMode::QwenChatSse if status.0 < 400 => {
                eprintln!(
                    "ctox proxy qwen upstream body={}",
                    String::from_utf8_lossy(&body)
                );
                let json_payload = engine::rewrite_qwen_chat_completions_to_responses(
                    &body,
                    config.active_model.as_deref(),
                )?;
                let json_payload = if let Some(augmentation) = web_search_augmentation {
                    web_search::augment_responses_output(&json_payload, augmentation)?
                } else {
                    json_payload
                };
                content_type_override = Some("text/event-stream");
                engine::rewrite_responses_payload_to_sse(&json_payload)?
            }
            LocalResponsesRelayMode::GlmChatJson if status.0 < 400 => {
                eprintln!(
                    "ctox proxy glm upstream body={}",
                    String::from_utf8_lossy(&body)
                );
                content_type_override = Some("application/json");
                engine::rewrite_glm_chat_completions_to_responses(
                    &body,
                    config.active_model.as_deref(),
                )?
            }
            LocalResponsesRelayMode::GlmChatSse if status.0 < 400 => {
                eprintln!(
                    "ctox proxy glm upstream body={}",
                    String::from_utf8_lossy(&body)
                );
                let json_payload = engine::rewrite_glm_chat_completions_to_responses(
                    &body,
                    config.active_model.as_deref(),
                )?;
                let json_payload = if let Some(augmentation) = web_search_augmentation {
                    web_search::augment_responses_output(&json_payload, augmentation)?
                } else {
                    json_payload
                };
                content_type_override = Some("text/event-stream");
                engine::rewrite_responses_payload_to_sse(&json_payload)?
            }
            LocalResponsesRelayMode::NemotronChatJson => body,
            LocalResponsesRelayMode::NemotronChatSse => body,
            LocalResponsesRelayMode::QwenChatJson => body,
            LocalResponsesRelayMode::QwenChatSse => body,
            LocalResponsesRelayMode::GlmChatJson => body,
            LocalResponsesRelayMode::GlmChatSse => body,
        },
        HarmonyRelayMode::Json if status.0 < 400 => {
            eprintln!("ctox proxy rewriting harmony completion response into responses payload");
            eprintln!(
                "ctox proxy raw harmony body={}",
                String::from_utf8_lossy(&body)
            );
            content_type_override = Some("application/json");
            engine::rewrite_gpt_oss_completion_to_responses(&body, None)?
        }
        HarmonyRelayMode::Sse if status.0 < 400 => {
            eprintln!("ctox proxy rewriting harmony completion response into SSE payload");
            eprintln!(
                "ctox proxy raw harmony body={}",
                String::from_utf8_lossy(&body)
            );
            let json_payload = engine::rewrite_gpt_oss_completion_to_responses(&body, None)?;
            let json_payload = if let Some(augmentation) = web_search_augmentation {
                web_search::augment_responses_output(&json_payload, augmentation)?
            } else {
                json_payload
            };
            content_type_override = Some("text/event-stream");
            engine::rewrite_responses_payload_to_sse(&json_payload)?
        }
        _ => body,
    };
    if status.0 < 400
        && matches!(
            local_responses_relay_mode,
            LocalResponsesRelayMode::Disabled
                | LocalResponsesRelayMode::NemotronChatJson
                | LocalResponsesRelayMode::QwenChatJson
                | LocalResponsesRelayMode::GlmChatJson
        )
        && !matches!(harmony_relay_mode, HarmonyRelayMode::Sse)
    {
        if let Some(augmentation) = web_search_augmentation {
            body = web_search::augment_responses_output(&body, augmentation)?;
        }
    }
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
    eprintln!(
        "ctox proxy writing downstream response status={} content_type_override={:?}",
        status.0,
        content_type_override
    );
    request
        .respond(tiny_response)
        .context("failed to write proxy response")?;
    eprintln!("ctox proxy downstream response write finished");
    Ok(())
}

fn json_header() -> Header {
    Header::from_bytes(b"content-type", b"application/json").expect("static content-type header")
}

fn rewrite_auxiliary_request_body(
    config: &ProxyConfig,
    request_path: &str,
    body: &[u8],
) -> Vec<u8> {
    let Some(model) = config.routed_model(request_path) else {
        return body.to_vec();
    };
    let Ok(mut value) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    let Some(object) = value.as_object_mut() else {
        return body.to_vec();
    };
    let should_override = match object.get("model") {
        None => true,
        Some(Value::String(existing)) => {
            existing.trim().is_empty()
                || existing == "default"
                || request_path == "/v1/audio/speech"
                || request_path == "/v1/embeddings"
        }
        _ => true,
    };
    if should_override {
        object.insert("model".to_string(), Value::String(model.to_string()));
    }
    serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec())
}

fn switch_active_model(
    state: &Arc<Mutex<ProxyState>>,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    model: &str,
    preset: Option<&str>,
) -> anyhow::Result<ProxySwitchResponse> {
    let requested_model = model.trim();
    if requested_model.is_empty() {
        anyhow::bail!("model switch request must not be empty");
    }

    let (root, previous_config) = {
        let guard = state.lock().expect("proxy state lock poisoned");
        (guard.root.clone(), guard.config.clone())
    };
    let next_config = if engine::is_openai_api_chat_model(requested_model) {
        ProxyConfig {
            root: root.clone(),
            listen_host: previous_config.listen_host.clone(),
            listen_port: previous_config.listen_port,
            upstream_base_url: OPENAI_RESPONSES_BASE_URL.to_string(),
            active_model: Some(requested_model.to_string()),
            embedding_base_url: previous_config.embedding_base_url.clone(),
            embedding_model: previous_config.embedding_model.clone(),
            transcription_base_url: previous_config.transcription_base_url.clone(),
            transcription_model: previous_config.transcription_model.clone(),
            speech_base_url: previous_config.speech_base_url.clone(),
            speech_model: previous_config.speech_model.clone(),
        }
    } else {
        let runtime = engine::runtime_config_for_model(requested_model)?;
        ProxyConfig {
            root: root.clone(),
            listen_host: previous_config.listen_host.clone(),
            listen_port: previous_config.listen_port,
            upstream_base_url: format!("http://127.0.0.1:{}", runtime.port),
            active_model: Some(runtime.model.clone()),
            embedding_base_url: previous_config.embedding_base_url.clone(),
            embedding_model: previous_config.embedding_model.clone(),
            transcription_base_url: previous_config.transcription_base_url.clone(),
            transcription_model: previous_config.transcription_model.clone(),
            speech_base_url: previous_config.speech_base_url.clone(),
            speech_model: previous_config.speech_model.clone(),
        }
    };

    let previous_env = runtime_env::load_runtime_env_map(&root).unwrap_or_default();
    persist_proxy_runtime_config(&root, &next_config, preset)?;
    let next_env = runtime_env::load_runtime_env_map(&root).unwrap_or_default();
    let current_backend_healthy = probe_upstream_health(&previous_config);

    let force_restart = !engine::is_openai_api_chat_model(requested_model)
        && (!current_backend_healthy
            || previous_config.active_model != next_config.active_model
            || previous_env.get("CTOX_CHAT_RUNTIME_PLAN_DIGEST")
                != next_env.get("CTOX_CHAT_RUNTIME_PLAN_DIGEST"));
    if previous_config
        .active_model
        .as_deref()
        .is_some_and(|active_model| !engine::is_openai_api_chat_model(active_model))
        && (previous_config.upstream_base_url != next_config.upstream_base_url
            || previous_config.active_model != next_config.active_model)
    {
        stop_backend_process(&root, &previous_config);
    }
    if let Err(err) = ensure_backend_ready(&root, &next_config, force_restart) {
        let _ = persist_proxy_runtime_config(&root, &previous_config, None);
        let mut telemetry_state = telemetry.lock().expect("proxy telemetry lock poisoned");
        telemetry_state.backend_healthy = false;
        telemetry_state.last_switch_status = Some("switch_failed".to_string());
        telemetry_state.last_switch_error = Some(err.to_string());
        drop(telemetry_state);
        anyhow::bail!("{err}");
    }

    {
        let mut guard = state.lock().expect("proxy state lock poisoned");
        guard.config = next_config.clone();
        guard.last_known_good = Some(next_config.clone());
        guard.last_switch_error = None;
    }

    {
        let mut telemetry_state = telemetry.lock().expect("proxy telemetry lock poisoned");
        telemetry_state.active_model = next_config.active_model.clone();
        telemetry_state.upstream_base_url = Some(next_config.upstream_base_url.clone());
        telemetry_state.last_known_good_model = next_config.active_model.clone();
        telemetry_state.backend_healthy = true;
        telemetry_state.last_switch_status = Some("switched".to_string());
        telemetry_state.last_switch_error = None;
        sync_boost_telemetry_fields(
            &mut telemetry_state,
            &root,
            next_config.active_model.as_deref(),
        );
    }

    Ok(ProxySwitchResponse {
        ok: true,
        active_model: next_config
            .active_model
            .clone()
            .unwrap_or_else(|| requested_model.to_string()),
        upstream_base_url: next_config.upstream_base_url,
        rolled_back: false,
        message: format!("proxy switched to {}", requested_model),
    })
}

fn attempt_auto_recovery(
    state: &Arc<Mutex<ProxyState>>,
    telemetry: &Arc<Mutex<ProxyTelemetry>>,
    reason: &str,
) -> anyhow::Result<()> {
    let (root, fallback_config, active_config, next_recovery_count) = {
        let mut guard = state.lock().expect("proxy state lock poisoned");
        guard.last_switch_error = Some(reason.to_string());
        guard.recovery_count = guard.recovery_count.saturating_add(1);
        (
            guard.root.clone(),
            guard.last_known_good.clone(),
            guard.config.clone(),
            guard.recovery_count,
        )
    };

    let mut telemetry_state = telemetry.lock().expect("proxy telemetry lock poisoned");
    telemetry_state.backend_healthy = false;
    telemetry_state.last_switch_error = Some(reason.to_string());
    telemetry_state.recovery_count = next_recovery_count;

    let Some(fallback_config) = fallback_config else {
        telemetry_state.last_switch_status = Some("recovery_unavailable".to_string());
        return Ok(());
    };

    if same_backend(&active_config, &fallback_config) {
        telemetry_state.last_switch_status = Some("backend_unhealthy".to_string());
        telemetry_state.active_model = active_config.active_model.clone();
        telemetry_state.upstream_base_url = Some(active_config.upstream_base_url.clone());
        return Ok(());
    }

    let ensure_result = persist_proxy_runtime_config(&root, &fallback_config, None)
        .and_then(|_| ensure_backend_ready(&root, &fallback_config, true));
    if ensure_result.is_err() {
        telemetry_state.last_switch_status = Some("recovery_failed".to_string());
        telemetry_state.last_switch_error = ensure_result.err().map(|err| err.to_string());
        return Ok(());
    }
    drop(telemetry_state);

    {
        let mut guard = state.lock().expect("proxy state lock poisoned");
        guard.config = fallback_config.clone();
    }

    let mut telemetry_state = telemetry.lock().expect("proxy telemetry lock poisoned");
    telemetry_state.active_model = fallback_config.active_model.clone();
    telemetry_state.upstream_base_url = Some(fallback_config.upstream_base_url.clone());
    telemetry_state.last_known_good_model = fallback_config.active_model.clone();
    telemetry_state.backend_healthy = true;
    telemetry_state.last_switch_status = Some("recovered".to_string());
    sync_boost_telemetry_fields(
        &mut telemetry_state,
        &root,
        fallback_config.active_model.as_deref(),
    );
    Ok(())
}

fn persist_proxy_runtime_config(
    root: &std::path::Path,
    config: &ProxyConfig,
    preset: Option<&str>,
) -> anyhow::Result<()> {
    let mut env_map = runtime_env::load_runtime_env_map(root).unwrap_or_else(|_| BTreeMap::new());
    if let Some(active_model) = &config.active_model {
        env_map.insert("CTOX_ACTIVE_MODEL".to_string(), active_model.clone());
        env_map.insert("CTOX_CHAT_MODEL".to_string(), active_model.clone());
        env_map.insert(
            "CTOX_CHAT_SOURCE".to_string(),
            if engine::is_openai_api_chat_model(active_model) {
                "api"
            } else {
                "local"
            }
            .to_string(),
        );
        if let Some(preset) = preset.map(str::trim).filter(|value| !value.is_empty()) {
            env_map.insert("CTOX_CHAT_LOCAL_PRESET".to_string(), preset.to_string());
        }
        env_map.remove("CTOX_ENGINE_REALIZED_MAX_SEQ_LEN");
        env_map.remove("CTOX_CHAT_MODEL_REALIZED_CONTEXT");
        env_map.remove("CTOX_ENGINE_REALIZED_MODEL");
        if engine::is_openai_api_chat_model(active_model) {
            env_map.remove("CTOX_ENGINE_MODEL");
            env_map.remove("CTOX_ENGINE_PORT");
            env_map.remove("CTOX_ENGINE_ARCH");
            runtime_plan::clear_chat_plan_env(&mut env_map);
        }
    }
    if !env_map
        .get("CTOX_CHAT_SOURCE")
        .map(|value| value.trim().eq_ignore_ascii_case("api"))
        .unwrap_or(false)
    {
        let _ = runtime_plan::apply_chat_runtime_plan(root, &mut env_map);
    }
    env_map.insert(
        "CTOX_UPSTREAM_BASE_URL".to_string(),
        config.upstream_base_url.clone(),
    );
    env_map.insert("CTOX_PROXY_HOST".to_string(), config.listen_host.clone());
    env_map.insert(
        "CTOX_PROXY_PORT".to_string(),
        config.listen_port.to_string(),
    );
    runtime_env::save_runtime_env_map(root, &env_map)
}

fn probe_upstream_health(config: &ProxyConfig) -> bool {
    if config
        .upstream_base_url
        .starts_with(OPENAI_RESPONSES_BASE_URL)
    {
        return runtime_env::env_or_config(&config.root, "OPENAI_API_KEY")
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false);
    }
    probe_backend_health_url(&config.join_url("/health"))
}

fn probe_backend_health_url(health_url: &str) -> bool {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(1))
        .timeout_read(std::time::Duration::from_secs(2))
        .timeout_write(std::time::Duration::from_secs(2))
        .build();

    match agent.get(health_url).call() {
        Ok(response) => response.status() < 500,
        Err(ureq::Error::Status(code, _)) => code < 500,
        Err(_) => false,
    }
}

fn load_observation_path(root: &std::path::Path, upstream_base_url: &str) -> Option<PathBuf> {
    if upstream_base_url.starts_with(OPENAI_RESPONSES_BASE_URL) {
        return None;
    }
    let port_slug = upstream_base_url.rsplit(':').next()?.trim();
    if port_slug.is_empty() {
        return None;
    }
    Some(
        root.join("runtime")
            .join(format!("load_observation_{port_slug}.json")),
    )
}

fn read_load_observation(
    root: &std::path::Path,
    upstream_base_url: &str,
) -> Option<LoadObservation> {
    let path = load_observation_path(root, upstream_base_url)?;
    let raw = std::fs::read(&path).ok()?;
    serde_json::from_slice(&raw).ok()
}

fn backend_startup_wait_secs_for_model(active_model: Option<&str>) -> u64 {
    match active_model {
        None => 120,
        Some(active_model) => match active_model {
            "openai/gpt-oss-20b" => 240,
            "Qwen/Qwen3.5-4B" | "Qwen/Qwen3.5-9B" => 240,
            // Large local models can spend several minutes in initial HF fetch / shard discovery
            // before they expose /health on a cold host. Keep the proxy patient enough to avoid
            // reporting false startup failures while the backend is still progressing.
            "Qwen/Qwen3.5-27B" => 1_200,
            "Qwen/Qwen3.5-35B-A3B" => 1_500,
            "nvidia/Nemotron-Cascade-2-30B-A3B" => 1_800,
            "zai-org/GLM-4.7-Flash" => 2_400,
            _ => match engine::runtime_config_for_model(active_model) {
                Ok(runtime) if runtime.family == engine::LocalModelFamily::Glm47Flash => 2_400,
                Ok(_) => 120,
                Err(_) => 120,
            },
        },
    }
}

fn backend_startup_wait_secs(config: &ProxyConfig) -> u64 {
    backend_startup_wait_secs_for_model(config.active_model.as_deref())
}

fn ensure_backend_ready(
    root: &std::path::Path,
    config: &ProxyConfig,
    force_restart: bool,
) -> anyhow::Result<()> {
    if config
        .upstream_base_url
        .starts_with(OPENAI_RESPONSES_BASE_URL)
    {
        if runtime_env::env_or_config(root, "OPENAI_API_KEY")
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
        {
            return Ok(());
        }
        anyhow::bail!("OPENAI_API_KEY is required for OpenAI API models");
    }
    if !force_restart && probe_upstream_health(config) {
        return Ok(());
    }

    start_backend_process(root, config)?;
    for _ in 0..backend_startup_wait_secs(config) {
        if probe_upstream_health(config) {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(1));
    }

    anyhow::bail!(
        "backend for model {} is not reachable at {} after startup",
        config.active_model.as_deref().unwrap_or("unknown"),
        config.upstream_base_url
    )
}

fn start_backend_process(root: &std::path::Path, config: &ProxyConfig) -> anyhow::Result<()> {
    start_backend_process_for_role(
        root,
        config.active_model.as_deref().unwrap_or("unknown"),
        &config.upstream_base_url,
        None,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuxiliaryLauncherKind {
    Engine,
    SpeachesCpu,
}

#[derive(Debug, Clone, Copy)]
struct AuxiliaryBackendSpec<'a> {
    model: &'a str,
    base_url: &'a str,
    role: engine::AuxiliaryRole,
    compute_target: engine::ComputeTarget,
    health_path: &'static str,
    launcher_kind: AuxiliaryLauncherKind,
}

impl<'a> AuxiliaryBackendSpec<'a> {
    fn role_env_value(self) -> &'static str {
        match self.role {
            engine::AuxiliaryRole::Embedding => "embedding",
            engine::AuxiliaryRole::Stt => "stt",
            engine::AuxiliaryRole::Tts => "tts",
        }
    }

    fn port_slug(self) -> &'a str {
        self.base_url.rsplit(':').next().unwrap_or("backend")
    }

    fn health_url(self) -> String {
        format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.health_path
        )
    }
}

fn configured_auxiliary_cuda_visible_devices(
    root: &std::path::Path,
    role: engine::AuxiliaryRole,
) -> Option<String> {
    let role_specific = match role {
        engine::AuxiliaryRole::Embedding => "CTOX_EMBEDDING_CUDA_VISIBLE_DEVICES",
        engine::AuxiliaryRole::Stt => "CTOX_STT_CUDA_VISIBLE_DEVICES",
        engine::AuxiliaryRole::Tts => "CTOX_TTS_CUDA_VISIBLE_DEVICES",
    };
    runtime_env::env_or_config(root, role_specific)
        .or_else(|| runtime_env::env_or_config(root, "CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES"))
        .filter(|value| !value.trim().is_empty())
}

fn start_backend_process_for_role(
    root: &std::path::Path,
    active_model: &str,
    upstream_base_url: &str,
    role: Option<&str>,
) -> anyhow::Result<()> {
    let runtime_dir = root.join("runtime");
    std::fs::create_dir_all(&runtime_dir)
        .with_context(|| format!("failed to create runtime dir {}", runtime_dir.display()))?;

    let port_slug = upstream_base_url.rsplit(':').next().unwrap_or("backend");
    let log_path = runtime_dir.join(format!("engine_{port_slug}.log"));
    let load_observation_path = runtime_dir.join(format!("load_observation_{port_slug}.json"));
    let launch_stamp = format!(
        "\n===== CTOX backend launch =====\nmodel={active_model}\nupstream={}\n",
        upstream_base_url
    );
    let _ = std::fs::remove_file(&load_observation_path);
    std::fs::write(&log_path, launch_stamp.as_bytes())
        .with_context(|| format!("failed to reset backend log {}", log_path.display()))?;
    let log_file = File::options()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open backend log {}", log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .with_context(|| format!("failed to clone backend log {}", log_path.display()))?;

    let auxiliary_spec = match role {
        Some("embedding") => Some(AuxiliaryBackendSpec {
            model: active_model,
            base_url: upstream_base_url,
            role: engine::AuxiliaryRole::Embedding,
            compute_target: engine::auxiliary_model_selection(
                engine::AuxiliaryRole::Embedding,
                Some(active_model),
            )
            .compute_target,
            health_path: "/health",
            launcher_kind: AuxiliaryLauncherKind::Engine,
        }),
        Some("stt") => {
            let selection =
                engine::auxiliary_model_selection(engine::AuxiliaryRole::Stt, Some(active_model));
            Some(AuxiliaryBackendSpec {
                model: active_model,
                base_url: upstream_base_url,
                role: engine::AuxiliaryRole::Stt,
                compute_target: selection.compute_target,
                health_path: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches {
                    "/v1/models"
                } else {
                    "/health"
                },
                launcher_kind: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches {
                    AuxiliaryLauncherKind::SpeachesCpu
                } else {
                    AuxiliaryLauncherKind::Engine
                },
            })
        }
        Some("tts") => {
            let selection =
                engine::auxiliary_model_selection(engine::AuxiliaryRole::Tts, Some(active_model));
            Some(AuxiliaryBackendSpec {
                model: active_model,
                base_url: upstream_base_url,
                role: engine::AuxiliaryRole::Tts,
                compute_target: selection.compute_target,
                health_path: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches {
                    "/v1/models"
                } else {
                    "/health"
                },
                launcher_kind: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches {
                    AuxiliaryLauncherKind::SpeachesCpu
                } else {
                    AuxiliaryLauncherKind::Engine
                },
            })
        }
        _ => None,
    };

    let script_path = match auxiliary_spec.map(|spec| spec.launcher_kind) {
        Some(AuxiliaryLauncherKind::SpeachesCpu) => {
            root.join("scripts/run_speaches_cpu_backend.sh")
        }
        _ => root.join("scripts/engine/run_engine.sh"),
    };
    if !script_path.exists() {
        anyhow::bail!("backend launcher missing: {}", script_path.display());
    }

    let _ = Command::new("bash")
        .arg("-lc")
        .arg(format!("fuser -k {port_slug}/tcp 2>/dev/null || true"))
        .current_dir(root)
        .status();

    writeln!(
        &log_file_err,
        "ctox spawn: script={} cwd={} model={} upstream={} port={}",
        script_path.display(),
        root.display(),
        active_model,
        upstream_base_url,
        port_slug,
    )
    .with_context(|| {
        format!(
            "failed to write backend launch preamble {}",
            log_path.display()
        )
    })?;

    let mut command = Command::new("bash");
    command
        .arg(script_path)
        .current_dir(root)
        .env(
            "CTOX_ENGINE_LOAD_OBSERVATION_PATH",
            load_observation_path.display().to_string(),
        )
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err));
    if let Some(spec) = auxiliary_spec {
        let resolved_aux_devices = if spec.compute_target == engine::ComputeTarget::Gpu {
            configured_auxiliary_cuda_visible_devices(root, spec.role)
                .unwrap_or_else(|| "0".to_string())
        } else {
            String::new()
        };
        let mut aux_log_file = File::options()
            .append(true)
            .open(&log_path)
            .with_context(|| format!("failed to reopen backend log {}", log_path.display()))?;
        writeln!(
            &mut aux_log_file,
            "ctox aux spawn env: role={} model_override={} compute_target={} env_file=/dev/null visible_devices={} nm_device={} base_device={}",
            spec.role_env_value(),
            spec.model,
            spec.compute_target.as_env_value(),
            if resolved_aux_devices.is_empty() { "<cpu>" } else { resolved_aux_devices.as_str() },
            if spec.compute_target == engine::ComputeTarget::Gpu { "0" } else { "" },
            if spec.compute_target == engine::ComputeTarget::Gpu { "0" } else { "" },
        )
        .with_context(|| format!("failed to write aux spawn env {}", log_path.display()))?;
        command
            .env("CTOX_ENGINE_ENV_FILE", "/dev/null")
            .env("CTOX_ENGINE_ROLE", spec.role_env_value())
            .env("CTOX_ENGINE_MODEL_OVERRIDE", spec.model)
            .env(
                "CTOX_ENGINE_COMPUTE_TARGET",
                spec.compute_target.as_env_value(),
            );
        match spec.role {
            engine::AuxiliaryRole::Embedding => {
                command
                    .env("CTOX_EMBEDDING_MODEL", spec.model)
                    .env("CTOX_EMBEDDING_PORT", spec.port_slug());
            }
            engine::AuxiliaryRole::Stt => {
                command
                    .env("CTOX_STT_MODEL", spec.model)
                    .env("CTOX_STT_PORT", spec.port_slug());
            }
            engine::AuxiliaryRole::Tts => {
                command
                    .env("CTOX_TTS_MODEL", spec.model)
                    .env("CTOX_TTS_PORT", spec.port_slug());
            }
        }
        if spec.compute_target == engine::ComputeTarget::Gpu {
            command
                .env("CTOX_ENGINE_CUDA_VISIBLE_DEVICES", resolved_aux_devices)
                .env("CTOX_ENGINE_NM_DEVICE_ORDINAL", "0")
                .env("CTOX_ENGINE_BASE_DEVICE_ORDINAL", "0");
        }
        for key in [
            "CTOX_CHAT_RUNTIME_PLAN_ACTIVE",
            "CTOX_ENGINE_DEVICE_LAYERS",
            "CTOX_ENGINE_NUM_DEVICE_LAYERS",
            "CTOX_ENGINE_NM_DEVICE_ORDINAL",
            "CTOX_ENGINE_BASE_DEVICE_ORDINAL",
            "CTOX_ENGINE_TOPOLOGY",
        ] {
            command.env_remove(key);
        }
        if spec.launcher_kind == AuxiliaryLauncherKind::SpeachesCpu {
            command
                .env("CTOX_AUX_CPU_ROLE", spec.role_env_value())
                .env("CTOX_AUX_PORT", spec.port_slug())
                .env("CTOX_AUX_REQUEST_MODEL", spec.model);
        }
    } else if let Some(role) = role {
        command.env("CTOX_ENGINE_ROLE", role);
    }
    command
        .spawn()
        .context("failed to launch backend process")?;

    Ok(())
}

fn auxiliary_backend_spec<'a>(
    config: &'a ProxyConfig,
    request_path: &str,
) -> Option<AuxiliaryBackendSpec<'a>> {
    match request_path {
        "/v1/embeddings" => Some(AuxiliaryBackendSpec {
            model: config.embedding_model.as_deref()?,
            base_url: &config.embedding_base_url,
            role: engine::AuxiliaryRole::Embedding,
            compute_target: engine::auxiliary_model_selection(
                engine::AuxiliaryRole::Embedding,
                config.embedding_model.as_deref(),
            )
            .compute_target,
            health_path: "/health",
            launcher_kind: AuxiliaryLauncherKind::Engine,
        }),
        "/v1/audio/transcriptions" => {
            let selection = engine::auxiliary_model_selection(
                engine::AuxiliaryRole::Stt,
                config.transcription_model.as_deref(),
            );
            Some(AuxiliaryBackendSpec {
                model: config.transcription_model.as_deref()?,
                base_url: &config.transcription_base_url,
                role: engine::AuxiliaryRole::Stt,
                compute_target: selection.compute_target,
                health_path: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches {
                    "/v1/models"
                } else {
                    "/health"
                },
                launcher_kind: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches {
                    AuxiliaryLauncherKind::SpeachesCpu
                } else {
                    AuxiliaryLauncherKind::Engine
                },
            })
        }
        "/v1/audio/speech" | "/v1/audio/voices" => {
            let selection = engine::auxiliary_model_selection(
                engine::AuxiliaryRole::Tts,
                config.speech_model.as_deref(),
            );
            Some(AuxiliaryBackendSpec {
                model: config.speech_model.as_deref()?,
                base_url: &config.speech_base_url,
                role: engine::AuxiliaryRole::Tts,
                compute_target: selection.compute_target,
                health_path: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches {
                    "/v1/models"
                } else {
                    "/health"
                },
                launcher_kind: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches {
                    AuxiliaryLauncherKind::SpeachesCpu
                } else {
                    AuxiliaryLauncherKind::Engine
                },
            })
        }
        _ => None,
    }
}

fn ensure_auxiliary_backend_ready(config: &ProxyConfig, request_path: &str) -> anyhow::Result<()> {
    let Some(spec) = auxiliary_backend_spec(config, request_path) else {
        return Ok(());
    };
    let health_url = spec.health_url();
    if probe_backend_health_url(&health_url) {
        return Ok(());
    }
    start_backend_process_for_role(
        &config.root,
        spec.model,
        spec.base_url,
        Some(spec.role_env_value()),
    )?;
    for _ in 0..backend_startup_wait_secs_for_model(Some(spec.model)) {
        if probe_backend_health_url(&health_url) {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(1));
    }
    anyhow::bail!(
        "auxiliary backend for model {} is not reachable at {} after startup",
        spec.model,
        spec.base_url
    )
}

fn stop_backend_process(root: &std::path::Path, config: &ProxyConfig) {
    if config
        .upstream_base_url
        .starts_with(OPENAI_RESPONSES_BASE_URL)
    {
        return;
    }
    let port_slug = config
        .upstream_base_url
        .rsplit(':')
        .next()
        .unwrap_or("backend");
    let _ = Command::new("bash")
        .arg("-lc")
        .arg(format!("fuser -k {port_slug}/tcp 2>/dev/null || true"))
        .current_dir(root)
        .status();
}

fn same_backend(left: &ProxyConfig, right: &ProxyConfig) -> bool {
    left.upstream_base_url == right.upstream_base_url && left.active_model == right.active_model
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
    state.upstream_base_url = Some(config.upstream_base_url.clone());
    state.backend_healthy = true;
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
        .or_else(|| {
            usage
                .get("prompt_tokens")
                .and_then(serde_json::Value::as_u64)
        })
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            usage
                .get("completion_tokens")
                .and_then(serde_json::Value::as_u64)
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    fn proxy_config(model: &str, upstream_base_url: &str) -> ProxyConfig {
        ProxyConfig {
            root: PathBuf::from("/tmp/ctox"),
            listen_host: "127.0.0.1".to_string(),
            listen_port: 12434,
            upstream_base_url: upstream_base_url.to_string(),
            active_model: Some(model.to_string()),
            embedding_base_url: "http://127.0.0.1:1237".to_string(),
            embedding_model: Some("Qwen/Qwen3-Embedding-0.6B".to_string()),
            transcription_base_url: "http://127.0.0.1:1238".to_string(),
            transcription_model: Some("mistralai/Voxtral-Mini-4B-Realtime-2602".to_string()),
            speech_base_url: "http://127.0.0.1:1239".to_string(),
            speech_model: Some("mistralai/Voxtral-4B-TTS-2603".to_string()),
        }
    }

    #[test]
    fn switch_target_uses_model_family_runtime_port() {
        let gpt_oss = proxy_config("openai/gpt-oss-20b", "http://127.0.0.1:1234");
        let qwen_runtime = engine::runtime_config_for_model("Qwen/Qwen3.5-35B-A3B").unwrap();
        let qwen = ProxyConfig {
            upstream_base_url: format!("http://127.0.0.1:{}", qwen_runtime.port),
            active_model: Some(qwen_runtime.model),
            ..gpt_oss.clone()
        };

        assert_eq!(qwen.upstream_base_url, "http://127.0.0.1:1235");
        assert_ne!(gpt_oss.upstream_base_url, qwen.upstream_base_url);
    }

    #[test]
    fn same_backend_requires_matching_model_and_upstream() {
        let left = proxy_config("openai/gpt-oss-20b", "http://127.0.0.1:1234");
        let same = proxy_config("openai/gpt-oss-20b", "http://127.0.0.1:1234");
        let different_model = ProxyConfig {
            active_model: Some("Qwen/Qwen3.5-35B-A3B".to_string()),
            ..same.clone()
        };

        assert!(same_backend(&left, &same));
        assert!(!same_backend(&left, &different_model));
    }

    #[test]
    fn auxiliary_routes_select_dedicated_upstreams() {
        let config = proxy_config("openai/gpt-oss-20b", "http://127.0.0.1:1234");

        assert_eq!(
            config.join_routed_url("/v1/responses"),
            "http://127.0.0.1:1234/v1/responses"
        );
        assert_eq!(
            config.join_routed_url("/v1/embeddings"),
            "http://127.0.0.1:1237/v1/embeddings"
        );
        assert_eq!(
            config.join_routed_url("/v1/audio/transcriptions"),
            "http://127.0.0.1:1238/v1/audio/transcriptions"
        );
        assert_eq!(
            config.join_routed_url("/v1/audio/speech"),
            "http://127.0.0.1:1239/v1/audio/speech"
        );
        assert_eq!(
            config.join_routed_url("/v1/audio/voices"),
            "http://127.0.0.1:1239/v1/audio/voices"
        );
    }

    #[test]
    fn auxiliary_backend_specs_match_health_paths_and_launchers() {
        let config = proxy_config("openai/gpt-oss-20b", "http://127.0.0.1:1234");

        let embedding = auxiliary_backend_spec(&config, "/v1/embeddings").unwrap();
        assert_eq!(embedding.role_env_value(), "embedding");
        assert_eq!(embedding.health_url(), "http://127.0.0.1:1237/health");
        assert_eq!(embedding.launcher_kind, AuxiliaryLauncherKind::Engine);

        let stt = auxiliary_backend_spec(&config, "/v1/audio/transcriptions").unwrap();
        assert_eq!(stt.role_env_value(), "stt");
        assert_eq!(stt.health_url(), "http://127.0.0.1:1238/health");
        assert_eq!(stt.launcher_kind, AuxiliaryLauncherKind::Engine);

        let tts = auxiliary_backend_spec(&config, "/v1/audio/speech").unwrap();
        assert_eq!(tts.role_env_value(), "tts");
        assert_eq!(tts.health_url(), "http://127.0.0.1:1239/health");
        assert_eq!(tts.launcher_kind, AuxiliaryLauncherKind::Engine);

        let cpu_tts = ProxyConfig {
            speech_model: Some("speaches-ai/piper-en_US-lessac-medium".to_string()),
            ..config
        };
        let cpu_tts_spec = auxiliary_backend_spec(&cpu_tts, "/v1/audio/speech").unwrap();
        assert_eq!(cpu_tts_spec.health_url(), "http://127.0.0.1:1239/v1/models");
        assert_eq!(
            cpu_tts_spec.launcher_kind,
            AuxiliaryLauncherKind::SpeachesCpu
        );
    }

    #[test]
    fn responses_requests_without_model_inherit_active_local_model() {
        let config = proxy_config("Qwen/Qwen3.5-4B", "http://127.0.0.1:1235");
        let body =
            br#"{"input":"Reply with CTOX_MATRIX_OK and nothing else.","max_output_tokens":24}"#;

        let rewritten = rewrite_auxiliary_request_body(&config, "/v1/responses", body);
        let payload: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(
            payload.get("model").and_then(Value::as_str),
            Some("Qwen/Qwen3.5-4B")
        );
    }

    #[test]
    fn api_runtime_config_clears_stale_local_chat_plan_fields() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ctox_gateway_api_runtime_config_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(root.join("runtime")).unwrap();
        std::fs::write(
            root.join("runtime/engine.env"),
            "CTOX_CHAT_SOURCE=local\nCTOX_CHAT_RUNTIME_PLAN_ACTIVE=1\nCTOX_ENGINE_MODEL=Qwen/Qwen3.5-27B\nCTOX_ENGINE_PAGED_ATTN=auto\nCTOX_ENGINE_DEVICE_LAYERS=0:40;1:24\n",
        )
        .unwrap();

        let config = ProxyConfig {
            root: root.clone(),
            listen_host: "127.0.0.1".to_string(),
            listen_port: 12434,
            upstream_base_url: OPENAI_RESPONSES_BASE_URL.to_string(),
            active_model: Some("gpt-5.4".to_string()),
            embedding_base_url: "http://127.0.0.1:1237".to_string(),
            embedding_model: Some("Qwen/Qwen3-Embedding-0.6B".to_string()),
            transcription_base_url: "http://127.0.0.1:1238".to_string(),
            transcription_model: Some("mistralai/Voxtral-Mini-4B-Realtime-2602".to_string()),
            speech_base_url: "http://127.0.0.1:1239".to_string(),
            speech_model: Some("mistralai/Voxtral-4B-TTS-2603".to_string()),
        };

        persist_proxy_runtime_config(&root, &config, None).unwrap();

        let env_map = runtime_env::load_runtime_env_map(&root).unwrap();
        assert_eq!(
            env_map.get("CTOX_CHAT_SOURCE").map(String::as_str),
            Some("api")
        );
        assert_eq!(
            env_map.get("CTOX_ACTIVE_MODEL").map(String::as_str),
            Some("gpt-5.4")
        );
        assert!(!env_map.contains_key("CTOX_CHAT_RUNTIME_PLAN_ACTIVE"));
        assert!(!env_map.contains_key("CTOX_ENGINE_DEVICE_LAYERS"));
        assert!(!env_map.contains_key("CTOX_ENGINE_PAGED_ATTN"));
        assert!(!env_map.contains_key("CTOX_ENGINE_MODEL"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn local_proxy_config_prefers_runtime_port_override() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ctox_gateway_local_runtime_port_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(root.join("runtime")).unwrap();
        std::fs::write(
            root.join("runtime/engine.env"),
            "CTOX_ACTIVE_MODEL=openai/gpt-oss-20b\nCTOX_CHAT_MODEL_BASE=openai/gpt-oss-20b\nCTOX_CHAT_SOURCE=local\nCTOX_ENGINE_PORT=2235\nCTOX_PROXY_PORT=22434\n",
        )
        .unwrap();

        let config = ProxyConfig::from_env_with_root(&root);

        assert_eq!(config.active_model.as_deref(), Some("openai/gpt-oss-20b"));
        assert_eq!(config.listen_port, 22434);
        assert_eq!(config.upstream_base_url, "http://127.0.0.1:2235");

        let _ = std::fs::remove_dir_all(&root);
    }
}

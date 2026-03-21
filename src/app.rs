use crate::attach::spawn_attach_server;
use crate::bootstrap::maybe_apply_homepage_feedback;
use crate::bootstrap::spawn_terminal_bridge;
use crate::browser_agent_bridge::browser_agent_bridge_port;
use crate::browser_agent_bridge::browser_agent_runtime_config;
use crate::browser_agent_bridge::complete_browser_agent_job;
use crate::browser_agent_bridge::create_browser_agent_job;
use crate::browser_agent_bridge::lease_next_browser_agent_job;
use crate::browser_agent_bridge::load_browser_agent_bridge_state;
use crate::browser_agent_bridge::load_browser_agent_job;
use crate::browser_engine::refresh_browser_engine_state;
use crate::contracts::Organigram;
use crate::contracts::Paths;
use crate::contracts::append_boot_entry;
use crate::contracts::control_plane_public_base_url;
use crate::contracts::control_plane_socket_addr;
use crate::contracts::load_agent_state;
use crate::contracts::load_bios;
use crate::contracts::load_boot_entries;
use crate::contracts::load_browser_engine_policy;
use crate::contracts::load_browser_subworker_policy;
use crate::contracts::load_census;
use crate::contracts::load_creation_ledger;
use crate::contracts::load_genome;
use crate::contracts::load_homepage_policy;
use crate::contracts::load_model_policy;
use crate::contracts::load_organigram;
use crate::contracts::load_origin_story;
use crate::contracts::load_root_auth;
use crate::contracts::refresh_bios_draft;
use crate::contracts::save_bios;
use crate::contracts::save_homepage_policy;
use crate::contracts::save_organigram;
use crate::contracts::split_lines;
use crate::contracts::update_root_password;
use crate::contracts::verify_root_password;
use crate::lifecycle::initialize_runtime;
use crate::pages::bios_page;
use crate::pages::browser_page;
use crate::pages::census_page;
use crate::pages::chat_page;
use crate::pages::history_page;
use crate::pages::home_page;
use crate::pages::models_page;
use crate::pages::org_page;
use crate::pages::root_auth_page;
use crate::runtime_db::activate_startup_recovery;
use crate::runtime_db::enqueue_loop_interrupt;
use crate::runtime_db::list_active_learning_entries;
use crate::runtime_db::list_bios_dialogue;
use crate::runtime_db::list_bios_uploads;
use crate::runtime_db::list_homepage_revisions;
use crate::runtime_db::list_memory_items;
use crate::runtime_db::list_open_tasks;
use crate::runtime_db::list_person_profiles;
use crate::runtime_db::list_proactive_contact_candidates;
use crate::runtime_db::list_recent_person_notes;
use crate::runtime_db::list_resources;
use crate::runtime_db::list_skills;
use crate::runtime_db::list_worker_jobs;
use crate::runtime_db::load_active_agent_turn;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::load_latest_completed_agent_turn;
use crate::runtime_db::load_memory_summary;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::record_bios_dialogue;
use crate::runtime_db::record_bios_upload;
use crate::runtime_db::record_homepage_revision;
use crate::runtime_db::record_turn_signal_for_active_turn;
use crate::runtime_db::set_brain_access_mode;
use crate::runtime_db::sync_model_resources;
use crate::runtime_db::sync_owner_trust;
use crate::runtime_db::sync_resources_from_census;
use crate::runtime_db::sync_skills;
use crate::supervisor::inspect_local_resources;
use crate::supervisor::run_system_census;
use crate::supervisor::spawn_supervisor;
use anyhow::Context;
use axum::Form;
use axum::Json;
use axum::Router;
use axum::extract::Multipart;
use axum::extract::Path as AxumPath;
use axum::extract::Query;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Html;
use axum::response::IntoResponse;
use axum::response::Redirect;
use axum::routing::get;
use axum::routing::post;
use axum_server::tls_rustls::RustlsConfig;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
struct AppState {
    paths: Paths,
}

struct RuntimeLockGuard {
    path: std::path::PathBuf,
    _file: File,
}

impl Drop for RuntimeLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Debug)]
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("internal error: {}", self.0),
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(value: E) -> Self {
        Self(value.into())
    }
}

#[derive(Default, Deserialize)]
struct FlashQuery {
    msg: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthTurnSnapshot {
    id: i64,
    task_id: i64,
    task_title: String,
    started_at: String,
    status: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthReport {
    status: String,
    ready: bool,
    supervisor_status: String,
    loop_health: String,
    reasons: Vec<String>,
    last_heartbeat_at: Option<String>,
    active_turn: Option<HealthTurnSnapshot>,
    last_completed_turn_at: Option<String>,
}

#[derive(Deserialize)]
struct ChatForm {
    speaker: String,
    message: String,
}

#[derive(Deserialize)]
struct OrgForm {
    owner_name: String,
    owner_email: String,
    ceo: String,
    reports_to: String,
    board: String,
    peer_cxos: String,
    sub_people: String,
    sub_agents: String,
    sub_vendors: String,
}

#[derive(Deserialize)]
struct RootAuthForm {
    password: String,
    confirm: String,
}

#[derive(Deserialize)]
struct BiosForm {
    agent_name: String,
    mission: String,
}

#[derive(Deserialize)]
struct FreezeForm {
    password: String,
}

#[derive(Deserialize)]
struct BiosChatForm {
    speaker: String,
    message: String,
}

#[derive(Deserialize)]
struct BrainAccessForm {
    mode: String,
    password: String,
}

#[derive(Deserialize)]
struct HomepageUpdateForm {
    source_channel: String,
    title: String,
    headline: String,
    intro: String,
    communication_note: String,
    terminal_fallback_note: String,
}

#[derive(Deserialize)]
struct HomepageBrandingForm {
    password: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserAgentWorkerPollQuery {
    worker_id: Option<String>,
    extension_id: Option<String>,
    extension_version: Option<String>,
}

pub async fn run() -> anyhow::Result<()> {
    let paths = Paths::discover()?;
    let _runtime_lock = acquire_runtime_lock(&paths)?;
    initialize_runtime(&paths)?;
    let _ = activate_startup_recovery(&paths)?;

    let started_at = Instant::now();
    spawn_terminal_bridge(paths.clone());
    spawn_attach_server(paths.clone());
    spawn_supervisor(paths.clone(), started_at);

    let state = Arc::new(AppState {
        paths: paths.clone(),
    });

    let bridge_state = state.clone();
    tokio::spawn(async move {
        if let Err(err) = run_browser_agent_bridge_http(bridge_state).await {
            eprintln!("browser agent bridge server failed: {err}");
        }
    });

    let router = Router::new()
        .route("/", get(home))
        .route("/homepage/update", post(homepage_update_post))
        .route("/homepage/branding-lock", post(homepage_branding_lock_post))
        .route("/chat", get(chat_get).post(chat_post))
        .route("/org", get(org_get).post(org_post))
        .route("/root-auth", get(root_auth_get))
        .route("/root-auth/set", post(root_auth_set))
        .route("/history", get(history_get))
        .route("/browser", get(browser_get))
        .route("/models", get(models_get))
        .route("/bios", get(bios_get))
        .route("/api/bios/template-data", get(bios_template_data_get))
        .route("/bios/chat", post(bios_chat_post))
        .route("/bios/upload", post(bios_upload_post))
        .route("/bios/brain-access", post(bios_brain_access_post))
        .route("/bios/update", post(bios_update))
        .route("/bios/freeze", post(bios_freeze))
        .route("/census", get(census_get))
        .route("/census/run", post(census_run))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route(
            "/api/browser-agent/runtime-config",
            get(browser_agent_runtime_config_get),
        )
        .route(
            "/api/browser-agent/bridge-state",
            get(browser_agent_bridge_state_get),
        )
        .route("/api/browser-agent/jobs", post(browser_agent_job_post))
        .route(
            "/api/browser-agent/jobs/{job_id}",
            get(browser_agent_job_get),
        )
        .route(
            "/api/browser-agent/jobs/{job_id}/complete",
            post(browser_agent_job_complete_post),
        )
        .route(
            "/api/browser-agent/worker/poll",
            get(browser_agent_worker_poll_get),
        )
        .route("/uploads/{*path}", get(upload_get))
        .with_state(state);

    let config = RustlsConfig::from_pem_file(&paths.tls_cert_path, &paths.tls_key_path)
        .await
        .context("failed to load rustls config")?;
    let addr = control_plane_socket_addr()?;
    let public_base_url = control_plane_public_base_url();
    println!(
        "CTO-Agent control plane listening on {}",
        public_base_url
    );
    println!(
        "CTO-Agent attach terminal available via {}",
        paths.attach_socket_path.display()
    );
    axum_server::bind_rustls(addr, config)
        .serve(router.into_make_service())
        .await
        .context("control plane failed")?;
    Ok(())
}

pub fn init_only() -> anyhow::Result<()> {
    let paths = Paths::discover()?;
    initialize_runtime(&paths)
}

async fn run_browser_agent_bridge_http(state: Arc<AppState>) -> anyhow::Result<()> {
    let port = browser_agent_bridge_port();
    let router = Router::new()
        .route("/health", get(browser_agent_bridge_health_get))
        .route(
            "/api/browser-agent/runtime-config",
            get(browser_agent_runtime_config_get),
        )
        .route(
            "/api/browser-agent/bridge-state",
            get(browser_agent_bridge_state_get),
        )
        .route("/api/browser-agent/jobs", post(browser_agent_job_post))
        .route(
            "/api/browser-agent/jobs/{job_id}",
            get(browser_agent_job_get),
        )
        .route(
            "/api/browser-agent/jobs/{job_id}/complete",
            post(browser_agent_job_complete_post),
        )
        .route(
            "/api/browser-agent/worker/poll",
            get(browser_agent_worker_poll_get),
        )
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("failed to bind browser agent bridge on 127.0.0.1:{port}"))?;
    println!(
        "CTO-Agent browser bridge listening on http://127.0.0.1:{}",
        port
    );
    axum::serve(listener, router)
        .await
        .context("browser agent bridge server failed")
}

async fn browser_agent_bridge_health_get(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, AppError> {
    let bridge = load_browser_agent_bridge_state(&state.paths, 6)?;
    Ok(Json(serde_json::json!({
        "ok": true,
        "status": "ready",
        "bridge": bridge,
    })))
}

async fn browser_agent_runtime_config_get(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, AppError> {
    Ok(Json(serde_json::json!({
        "ok": true,
        "config": browser_agent_runtime_config(&state.paths),
    })))
}

async fn browser_agent_bridge_state_get(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, AppError> {
    let bridge = load_browser_agent_bridge_state(&state.paths, 12)?;
    Ok(Json(serde_json::json!({
        "ok": true,
        "bridge": bridge,
    })))
}

async fn browser_agent_job_post(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let request = if let Some(request) = payload.get("request").cloned() {
        request
    } else {
        payload
    };
    let job = create_browser_agent_job(&state.paths, request)?;
    Ok(Json(serde_json::json!({
        "ok": true,
        "job": job,
    })))
}

async fn browser_agent_job_get(
    State(state): State<Arc<AppState>>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let Some(job) = load_browser_agent_job(&state.paths, &job_id)? else {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": format!("Unknown browser agent job: {job_id}"),
        })));
    };
    Ok(Json(serde_json::json!({
        "ok": true,
        "job": job,
    })))
}

async fn browser_agent_job_complete_post(
    State(state): State<Arc<AppState>>,
    AxumPath(job_id): AxumPath<String>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let result = payload.get("result").cloned().unwrap_or_else(|| {
        serde_json::json!({
            "ok": false,
            "error": "Missing browser-agent result payload.",
        })
    });
    let worker_id = payload
        .get("workerId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let worker = payload.get("worker").cloned();
    let Some(job) = complete_browser_agent_job(&state.paths, &job_id, &worker_id, worker, result)?
    else {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": format!("Unknown browser agent job: {job_id}"),
        })));
    };
    Ok(Json(serde_json::json!({
        "ok": true,
        "job": job,
    })))
}

async fn browser_agent_worker_poll_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<BrowserAgentWorkerPollQuery>,
) -> Result<Json<Value>, AppError> {
    let worker_id = query
        .worker_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("worker-{}", chrono::Utc::now().timestamp_millis()));
    let (worker, job) = lease_next_browser_agent_job(
        &state.paths,
        &worker_id,
        query.extension_id.as_deref(),
        query.extension_version.as_deref(),
    )?;
    Ok(Json(serde_json::json!({
        "ok": true,
        "worker": worker,
        "job": job,
    })))
}

fn acquire_runtime_lock(paths: &Paths) -> anyhow::Result<RuntimeLockGuard> {
    fs::create_dir_all(&paths.runtime_dir)?;
    let lock_path = &paths.runtime_lock_path;
    for _ in 0..2 {
        match OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(lock_path)
        {
            Ok(mut file) => {
                let payload = serde_json::json!({
                    "pid": std::process::id(),
                    "createdAt": chrono::Utc::now().to_rfc3339(),
                });
                let text = serde_json::to_string_pretty(&payload)?;
                file.write_all(text.as_bytes())?;
                file.flush()?;
                return Ok(RuntimeLockGuard {
                    path: lock_path.clone(),
                    _file: file,
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if stale_runtime_lock_can_be_removed(lock_path)? {
                    let _ = fs::remove_file(lock_path);
                    continue;
                }
                anyhow::bail!(
                    "another cto-agent runtime instance appears to be active ({})",
                    lock_path.display()
                );
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("failed to acquire runtime lock {}", lock_path.display())
                });
            }
        }
    }
    anyhow::bail!(
        "failed to acquire runtime lock after stale-lock recovery ({})",
        lock_path.display()
    )
}

fn stale_runtime_lock_can_be_removed(path: &std::path::Path) -> anyhow::Result<bool> {
    let mut text = String::new();
    if let Ok(mut file) = File::open(path) {
        let _ = file.read_to_string(&mut text);
    }
    let pid = serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|value| value.get("pid").and_then(|item| item.as_u64()))
        .map(|value| value as u32);
    let Some(pid) = pid else {
        return Ok(true);
    };
    Ok(!process_is_alive(pid))
}

fn process_is_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

async fn healthz(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    health_response(&state.paths, false)
}

async fn readyz(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    health_response(&state.paths, true)
}

#[derive(Debug, Clone)]
struct KleinhirnTransitionState {
    deadline_epoch: i64,
}

fn load_kleinhirn_transition_state(paths: &Paths) -> Option<KleinhirnTransitionState> {
    let path = paths.runtime_dir.join("state/kleinhirn-transition.env");
    let text = fs::read_to_string(path).ok()?;
    let mut deadline_epoch = None;
    for line in text.lines() {
        let (key, value) = line.split_once('=')?;
        let parsed = value.trim().trim_matches('\'').trim_matches('"');
        if key.trim() == "CTO_AGENT_KLEINHIRN_TRANSITION_DEADLINE_EPOCH" {
            deadline_epoch = parsed.parse::<i64>().ok();
        }
    }
    deadline_epoch.map(|deadline_epoch| KleinhirnTransitionState { deadline_epoch })
}

fn kleinhirn_transition_active(paths: &Paths) -> bool {
    let Some(state) = load_kleinhirn_transition_state(paths) else {
        return false;
    };
    let now = chrono::Utc::now().timestamp();
    now <= state.deadline_epoch
}

fn health_response(paths: &Paths, require_ready: bool) -> (StatusCode, String) {
    let report = evaluate_runtime_health(paths, require_ready);
    let status = if report.status == "ok" {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        serde_json::to_string(&report)
            .unwrap_or_else(|_| "{\"status\":\"error\",\"ready\":false}".to_string()),
    )
}

fn evaluate_runtime_health(paths: &Paths, require_ready: bool) -> HealthReport {
    let state = load_agent_state(paths);
    let active_turn = load_active_agent_turn(paths).ok().flatten();
    let latest_completed_turn = load_latest_completed_agent_turn(paths).ok().flatten();
    let resources = list_resources(paths, 256).unwrap_or_default();
    let heartbeat_stale_secs = parse_env_i64("CTO_AGENT_HEARTBEAT_STALE_SECS", 20);
    let turn_stale_secs = parse_env_i64("CTO_AGENT_ACTIVE_TURN_STALE_SECS", 300);
    let transition_active = kleinhirn_transition_active(paths);
    let mut reasons = Vec::new();

    if state.supervisor_status != "running" {
        reasons.push(format!(
            "supervisor_status is {} instead of running",
            state.supervisor_status
        ));
    }

    match state
        .last_heartbeat_at
        .as_deref()
        .and_then(seconds_since_iso)
    {
        Some(age) if age > heartbeat_stale_secs && !transition_active => reasons.push(format!(
            "supervisor heartbeat stale for {}s (limit {}s)",
            age, heartbeat_stale_secs
        )),
        Some(_) => {}
        None if !transition_active => reasons.push("missing supervisor heartbeat".to_string()),
        None => {}
    }

    if let Some(turn) = active_turn.as_ref()
        && let Some(age) = seconds_since_iso(&turn.created_at)
        && age > turn_stale_secs
        && !transition_active
    {
        reasons.push(format!(
            "active turn {} for task {} has been running {}s (limit {}s)",
            turn.id, turn.task_id, age, turn_stale_secs
        ));
    }

    let agentic_status = resources
        .iter()
        .find(|resource| resource.category == "agentic_loop" && resource.name == "status")
        .cloned();

    let base_reasons = reasons.clone();
    let is_ready = if transition_active {
        base_reasons.is_empty()
    } else {
        base_reasons.is_empty()
            && matches!(agentic_status.as_ref(), Some(status) if status.status == "ok")
    };
    if require_ready && !is_ready && !transition_active {
        match agentic_status.as_ref() {
            Some(status) => reasons.push(format!(
                "agentic loop is not ready (status={}, detail={})",
                status.status, status.detail
            )),
            None => reasons.push("agentic loop has not reported readiness yet".to_string()),
        }
    }

    let status = if reasons.is_empty() { "ok" } else { "error" }.to_string();
    HealthReport {
        ready: is_ready,
        supervisor_status: state.supervisor_status.clone(),
        loop_health: if base_reasons.is_empty() {
            "healthy".to_string()
        } else {
            "unhealthy".to_string()
        },
        status,
        reasons,
        last_heartbeat_at: state.last_heartbeat_at,
        active_turn: active_turn.map(|turn| HealthTurnSnapshot {
            id: turn.id,
            task_id: turn.task_id,
            task_title: turn.task_title,
            started_at: turn.created_at,
            status: turn.status,
        }),
        last_completed_turn_at: latest_completed_turn.and_then(|turn| turn.completed_at),
    }
}

fn seconds_since_iso(value: &str) -> Option<i64> {
    let parsed = chrono::DateTime::parse_from_rfc3339(value).ok()?;
    let now = chrono::Utc::now();
    Some((now - parsed.with_timezone(&chrono::Utc)).num_seconds())
}

fn parse_env_i64(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn camel_case_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    let mut uppercase_next = false;
    for ch in key.chars() {
        if ch == '_' {
            uppercase_next = true;
        } else if uppercase_next {
            out.extend(ch.to_uppercase());
            uppercase_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}

fn camelize_json_keys(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut next = serde_json::Map::new();
            for (key, value) in map {
                next.insert(camel_case_key(&key), camelize_json_keys(value));
            }
            Value::Object(next)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(camelize_json_keys).collect()),
        other => other,
    }
}

async fn home(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let paths = &state.paths;
    let bios = load_bios(paths);
    let genome = load_genome(paths);
    let root_auth = load_root_auth(paths);
    let organigram = load_organigram(paths);
    let model_policy = load_model_policy(paths);
    let homepage_policy = load_homepage_policy(paths);
    let census = inspect_local_resources(paths).unwrap_or_else(|_| load_census(paths));
    let agent_state = load_agent_state(paths);
    let browser_state = refresh_browser_engine_state(paths)
        .unwrap_or_else(|_| crate::contracts::load_browser_engine_state(paths));
    let trust = load_owner_trust(paths)?;
    let dialogue = list_bios_dialogue(paths, 12)?;
    let homepage_revisions = list_homepage_revisions(paths, 8)?;
    let uploads = list_bios_uploads(paths, 8)?;
    let testimony_count = load_boot_entries(paths).len();
    Ok(Html(home_page(
        query.msg.as_deref(),
        &genome,
        &homepage_policy,
        &bios,
        &root_auth,
        &organigram,
        &model_policy,
        &census,
        &trust,
        &dialogue,
        &homepage_revisions,
        &uploads,
        &agent_state,
        &browser_state,
        testimony_count,
    )))
}

async fn browser_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let paths = &state.paths;
    let policy = load_browser_engine_policy(paths);
    let subworker_policy = load_browser_subworker_policy(paths);
    let browser_state = refresh_browser_engine_state(paths)
        .unwrap_or_else(|_| crate::contracts::load_browser_engine_state(paths));
    let bridge_state = load_browser_agent_bridge_state(paths, 10)?;
    let census = inspect_local_resources(paths).unwrap_or_else(|_| load_census(paths));
    let worker_jobs = list_worker_jobs(paths, 10).unwrap_or_default();
    Ok(Html(browser_page(
        query.msg.as_deref(),
        &policy,
        &subworker_policy,
        &browser_state,
        &bridge_state,
        &census,
        &worker_jobs,
    )))
}

async fn homepage_update_post(
    State(state): State<Arc<AppState>>,
    Form(form): Form<HomepageUpdateForm>,
) -> Result<Redirect, AppError> {
    let mut policy = load_homepage_policy(&state.paths);
    policy.current_title = if form.title.trim().is_empty() {
        policy.current_title
    } else {
        form.title.trim().to_string()
    };
    policy.current_headline = if form.headline.trim().is_empty() {
        policy.current_headline
    } else {
        form.headline.trim().to_string()
    };
    policy.current_intro = if form.intro.trim().is_empty() {
        policy.current_intro
    } else {
        form.intro.trim().to_string()
    };
    policy.communication_note = if form.communication_note.trim().is_empty() {
        policy.communication_note
    } else {
        form.communication_note.trim().to_string()
    };
    policy.terminal_fallback_note = if form.terminal_fallback_note.trim().is_empty() {
        policy.terminal_fallback_note
    } else {
        form.terminal_fallback_note.trim().to_string()
    };
    policy.homepage_ready = true;
    policy.stage = "homepage_building".to_string();
    policy.updated_at = crate::contracts::now_iso();
    save_homepage_policy(&state.paths, &policy)?;
    record_homepage_revision(
        &state.paths,
        if form.source_channel.trim().is_empty() {
            "terminal"
        } else {
            form.source_channel.trim()
        },
        &policy,
        "Homepage template revised.",
    )?;
    Ok(redirect_with_msg("/", "Homepage template updated."))
}

async fn homepage_branding_lock_post(
    State(state): State<Arc<AppState>>,
    Form(form): Form<HomepageBrandingForm>,
) -> Result<Redirect, AppError> {
    let root_auth = load_root_auth(&state.paths);
    if !root_auth.configured {
        return Ok(redirect_with_msg(
            "/",
            "Superpassword is missing. Owner branding can only be locked after that.",
        ));
    }
    if !verify_root_password(&state.paths, &form.password)? {
        return Ok(redirect_with_msg(
            "/",
            "Root verification for branding lock failed.",
        ));
    }
    let trust = load_owner_trust(&state.paths)?;
    if !trust.bios_primary_channel_confirmed {
        return Ok(redirect_with_msg(
            "/",
            "Owner branding is only locked after BIOS communication has been taken over.",
        ));
    }

    let mut policy = load_homepage_policy(&state.paths);
    policy.owner_branding_applied = true;
    policy.owner_branding_locked = true;
    policy.stage = "bios_primary".to_string();
    policy.updated_at = crate::contracts::now_iso();
    save_homepage_policy(&state.paths, &policy)?;
    record_homepage_revision(
        &state.paths,
        "bios",
        &policy,
        "Homepage branding locked to committed owner after BIOS takeover.",
    )?;
    Ok(redirect_with_msg(
        "/",
        "Owner branding on the homepage was locked.",
    ))
}

async fn chat_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let entries = load_boot_entries(&state.paths);
    Ok(Html(chat_page(query.msg.as_deref(), &entries)))
}

async fn chat_post(
    State(state): State<Arc<AppState>>,
    Form(form): Form<ChatForm>,
) -> Result<Redirect, AppError> {
    let speaker = if form.speaker.trim().is_empty() {
        "initiator"
    } else {
        form.speaker.trim()
    };
    let message = form.message.trim();
    if message.is_empty() {
        return Ok(redirect_with_msg("/chat", "Empty message was not saved."));
    }
    append_boot_entry(&state.paths, speaker, message)?;
    let _ = record_turn_signal_for_active_turn(&state.paths, "homepage", speaker, message);
    let interrupt_id = enqueue_loop_interrupt(&state.paths, "homepage", speaker, message)?;
    Ok(redirect_with_msg(
        "/chat",
        &format!(
            "Boot testimony was saved as interrupt #{} and will be materialized by the supervisor on the next intake tick.",
            interrupt_id
        ),
    ))
}

async fn org_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let organigram = load_organigram(&state.paths);
    let bios = load_bios(&state.paths);
    Ok(Html(org_page(query.msg.as_deref(), &organigram, &bios)))
}

async fn org_post(
    State(state): State<Arc<AppState>>,
    Form(form): Form<OrgForm>,
) -> Result<Redirect, AppError> {
    let bios = load_bios(&state.paths);
    if bios.frozen {
        return Ok(redirect_with_msg(
            "/org",
            "The organigram is no longer freely editable here after BIOS freeze.",
        ));
    }
    let organigram = Organigram {
        owner: crate::contracts::OwnerRef {
            name: form.owner_name.trim().to_string(),
            email: form.owner_email.trim().to_string(),
            role: "owner".to_string(),
        },
        reports_to: form.reports_to.trim().to_string(),
        ceo: form.ceo.trim().to_string(),
        board: split_lines(&form.board),
        peer_cxos: split_lines(&form.peer_cxos),
        subordinates: crate::contracts::Subordinates {
            people: split_lines(&form.sub_people),
            agents: split_lines(&form.sub_agents),
            vendors: split_lines(&form.sub_vendors),
        },
        updated_at: crate::contracts::now_iso(),
    };
    save_organigram(&state.paths, &organigram)?;
    Ok(redirect_with_msg(
        "/org",
        "Organigramm-Entwurf aktualisiert.",
    ))
}

async fn root_auth_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let root_auth = load_root_auth(&state.paths);
    Ok(Html(root_auth_page(query.msg.as_deref(), &root_auth)))
}

async fn root_auth_set(
    State(state): State<Arc<AppState>>,
    Form(form): Form<RootAuthForm>,
) -> Result<Redirect, AppError> {
    if form.password.len() < 12 {
        return Ok(redirect_with_msg(
            "/root-auth",
            "Superpassword must be at least 12 characters long.",
        ));
    }
    if form.password != form.confirm {
        return Ok(redirect_with_msg("/root-auth", "Passwords do not match."));
    }
    update_root_password(&state.paths, &form.password)?;
    Ok(redirect_with_msg("/root-auth", "Superpassword was set."))
}

async fn history_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let origin_story = load_origin_story(&state.paths);
    let creation_ledger = load_creation_ledger(&state.paths);
    Ok(Html(history_page(
        query.msg.as_deref(),
        &origin_story,
        &creation_ledger,
    )))
}

async fn models_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let model_policy = load_model_policy(&state.paths);
    let census =
        inspect_local_resources(&state.paths).unwrap_or_else(|_| load_census(&state.paths));
    Ok(Html(models_page(
        query.msg.as_deref(),
        &model_policy,
        &census,
    )))
}

async fn bios_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let bios = refresh_bios_draft(&state.paths)?;
    let genome = load_genome(&state.paths);
    let model_policy = load_model_policy(&state.paths);
    let census =
        inspect_local_resources(&state.paths).unwrap_or_else(|_| load_census(&state.paths));
    let _ = sync_owner_trust(&state.paths);
    let _ = sync_resources_from_census(&state.paths, &census);
    let _ = sync_model_resources(&state.paths, &model_policy, &census);
    let _ = sync_skills(&state.paths);
    let trust = load_owner_trust(&state.paths)?;
    let dialogue = list_bios_dialogue(&state.paths, 12)?;
    let memory_items = list_memory_items(&state.paths, 10)?;
    let memory_summary = load_memory_summary(&state.paths, "owner_calibration")?;
    let learning_working_set = load_memory_summary(&state.paths, "learning_working_set")?;
    let learning_operational = load_memory_summary(&state.paths, "learning_operational")?;
    let learning_general = load_memory_summary(&state.paths, "learning_general")?;
    let learning_negative = load_memory_summary(&state.paths, "learning_negative")?;
    let people_working_set = load_memory_summary(&state.paths, "people_working_set")?;
    let learning_entries = list_active_learning_entries(&state.paths, 12)?;
    let person_profiles = list_person_profiles(&state.paths, 8)?;
    let person_notes = list_recent_person_notes(&state.paths, 10)?;
    let proactive_contacts = list_proactive_contact_candidates(&state.paths, 8)?;
    let resources = list_resources(&state.paths, 12)?;
    let skills = list_skills(&state.paths)?;
    let uploads = list_bios_uploads(&state.paths, 8)?;
    Ok(Html(bios_page(
        query.msg.as_deref(),
        &genome,
        &bios,
        &trust,
        &model_policy,
        &census,
        memory_summary.as_deref(),
        learning_working_set.as_deref(),
        learning_operational.as_deref(),
        learning_general.as_deref(),
        learning_negative.as_deref(),
        people_working_set.as_deref(),
        &dialogue,
        &memory_items,
        &learning_entries,
        &person_profiles,
        &person_notes,
        &proactive_contacts,
        &resources,
        &skills,
        &uploads,
    )))
}

async fn bios_template_data_get(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, AppError> {
    let bios = refresh_bios_draft(&state.paths)?;
    let genome = load_genome(&state.paths);
    let homepage = load_homepage_policy(&state.paths);
    let organigram = load_organigram(&state.paths);
    let root_auth = load_root_auth(&state.paths);
    let model_policy = load_model_policy(&state.paths);
    let census =
        inspect_local_resources(&state.paths).unwrap_or_else(|_| load_census(&state.paths));
    let agent_state = load_agent_state(&state.paths);
    let browser_state = refresh_browser_engine_state(&state.paths)
        .unwrap_or_else(|_| crate::contracts::load_browser_engine_state(&state.paths));
    let _ = sync_owner_trust(&state.paths);
    let _ = sync_resources_from_census(&state.paths, &census);
    let _ = sync_model_resources(&state.paths, &model_policy, &census);
    let _ = sync_skills(&state.paths);

    let trust = load_owner_trust(&state.paths)?;
    let focus_state = load_focus_state(&state.paths).ok();
    let open_tasks = list_open_tasks(&state.paths, 12).unwrap_or_default();
    let dialogue = list_bios_dialogue(&state.paths, 12).unwrap_or_default();
    let uploads = list_bios_uploads(&state.paths, 8).unwrap_or_default();
    let memory_items = list_memory_items(&state.paths, 12).unwrap_or_default();
    let learning_entries = list_active_learning_entries(&state.paths, 16).unwrap_or_default();
    let resources = list_resources(&state.paths, 32).unwrap_or_default();
    let skills = list_skills(&state.paths).unwrap_or_default();
    let homepage_revisions = list_homepage_revisions(&state.paths, 8).unwrap_or_default();
    let runtime_db_path = state.paths.runtime_db_path.display().to_string();
    let bios_contract_path = state.paths.bios_path.display().to_string();
    let homepage_contract_path = state.paths.homepage_policy_path.display().to_string();

    let snapshot = serde_json::json!({
        "ok": true,
        "generatedAt": crate::contracts::now_iso(),
        "template": {
            "id": "bios-start-template",
            "version": 1,
            "snapshotUrl": "/api/bios/template-data",
            "sqlitePrimary": true,
        },
        "sources": {
            "runtimeDbPath": runtime_db_path,
            "contracts": {
                "bios": bios_contract_path,
                "homepage": homepage_contract_path,
            }
        },
        "contracts": {
            "bios": bios,
            "homepage": homepage,
            "organigram": organigram,
            "rootAuth": root_auth,
            "genome": genome,
            "modelPolicy": model_policy,
            "census": census,
        },
        "runtime": {
            "trust": trust,
            "agentState": agent_state,
            "browserState": browser_state,
            "focusState": focus_state,
            "openTasks": open_tasks,
            "dialogue": dialogue,
            "uploads": uploads,
            "memorySummary": {
                "ownerCalibration": load_memory_summary(&state.paths, "owner_calibration")?,
                "learningWorkingSet": load_memory_summary(&state.paths, "learning_working_set")?,
                "learningOperational": load_memory_summary(&state.paths, "learning_operational")?,
                "learningGeneral": load_memory_summary(&state.paths, "learning_general")?,
                "learningNegative": load_memory_summary(&state.paths, "learning_negative")?,
            },
            "memoryItems": memory_items,
            "learningEntries": learning_entries,
            "resources": resources,
            "skills": skills,
            "homepageRevisions": homepage_revisions,
        },
    });
    Ok(Json(camelize_json_keys(snapshot)))
}

async fn bios_chat_post(
    State(state): State<Arc<AppState>>,
    Form(form): Form<BiosChatForm>,
) -> Result<Redirect, AppError> {
    let speaker = form.speaker.trim();
    let message = form.message.trim();
    if message.is_empty() {
        return Ok(redirect_with_msg(
            "/bios",
            "Empty BIOS message was not saved.",
        ));
    }
    let speaker = if speaker.is_empty() {
        "Michael Welsch"
    } else {
        speaker
    };
    record_bios_dialogue(&state.paths, speaker, message, false)?;
    let _ = record_turn_signal_for_active_turn(&state.paths, "bios", speaker, message);
    let interrupt_id = enqueue_loop_interrupt(&state.paths, "bios", speaker, message)?;
    let focus = load_focus_state(&state.paths)?;
    let open_tasks = list_open_tasks(&state.paths, 3)?;
    let open_task_text = open_tasks
        .iter()
        .map(|task| format!("#{} {}", task.id, task.title))
        .collect::<Vec<_>>()
        .join(" | ");
    let reply = format!(
        "I recorded your BIOS message as interrupt #{} for supervisor intake. Current agent mode: mode={}, active_task={}. Open tasks: {}.",
        interrupt_id,
        focus.mode,
        focus
            .active_task_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        if open_task_text.is_empty() {
            "no further ones".to_string()
        } else {
            open_task_text
        }
    );
    record_bios_dialogue(&state.paths, "cto-agent", &reply, false)?;
    Ok(redirect_with_msg(
        "/bios",
        "BIOS message handed into the Infinity Loop as a prioritized task.",
    ))
}

async fn bios_upload_post(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Redirect, AppError> {
    let mut speaker = "Michael Welsch".to_string();
    let mut note = String::new();
    let mut source_channel = "homepage".to_string();
    let mut redirect_to = "/".to_string();
    let mut file_name: Option<String> = None;
    let mut mime_type = "application/octet-stream".to_string();
    let mut file_bytes = Vec::new();

    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "speaker" => {
                speaker = String::from_utf8(field.bytes().await?.to_vec())
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if speaker.is_empty() {
                    speaker = "Michael Welsch".to_string();
                }
            }
            "note" => {
                note = String::from_utf8(field.bytes().await?.to_vec())
                    .unwrap_or_default()
                    .trim()
                    .to_string();
            }
            "source_channel" => {
                let value = String::from_utf8(field.bytes().await?.to_vec()).unwrap_or_default();
                if !value.trim().is_empty() {
                    source_channel = value.trim().to_string();
                }
            }
            "redirect_to" => {
                let value = String::from_utf8(field.bytes().await?.to_vec()).unwrap_or_default();
                if !value.trim().is_empty() {
                    redirect_to = value.trim().to_string();
                }
            }
            "image" => {
                if let Some(upload_name) = field.file_name() {
                    file_name = Some(sanitize_upload_name(upload_name));
                }
                if let Some(content_type) = field.content_type() {
                    mime_type = content_type.to_string();
                }
                file_bytes = field.bytes().await?.to_vec();
            }
            _ => {
                let _ = field.bytes().await?;
            }
        }
    }

    if file_bytes.is_empty() {
        return Ok(redirect_with_msg(&redirect_to, "No image received."));
    }
    if !mime_type.starts_with("image/") {
        return Ok(redirect_with_msg(
            &redirect_to,
            "Only image uploads for the 1:1 homepage or BIOS chat are accepted.",
        ));
    }
    if file_bytes.len() > 10 * 1024 * 1024 {
        return Ok(redirect_with_msg(
            &redirect_to,
            "Image is too large. The current limit is 10 MB.",
        ));
    }

    let safe_file_name = file_name.unwrap_or_else(|| "upload.bin".to_string());
    let stored_name = format!(
        "{}_{}",
        chrono::Utc::now().timestamp_millis(),
        safe_file_name
    );
    let file_path = state.paths.uploads_dir.join(&stored_name);
    fs::write(&file_path, &file_bytes)?;
    let public_path = format!("/uploads/{}", stored_name);
    record_bios_upload(
        &state.paths,
        &speaker,
        &source_channel,
        &note,
        &stored_name,
        &public_path,
        &mime_type,
    )?;

    if !note.trim().is_empty() {
        let route_channel = if source_channel == "terminal" {
            "terminal"
        } else {
            "bios"
        };
        let _ = maybe_apply_homepage_feedback(&state.paths, route_channel, &speaker, &note);
    }

    Ok(redirect_with_msg(
        &redirect_to,
        "Image stored for the 1:1 homepage or BIOS chat.",
    ))
}

async fn bios_brain_access_post(
    State(state): State<Arc<AppState>>,
    Form(form): Form<BrainAccessForm>,
) -> Result<Redirect, AppError> {
    let root_auth = load_root_auth(&state.paths);
    if !root_auth.configured {
        return Ok(redirect_with_msg(
            "/bios",
            "Superpassword is missing. Please set it first in BIOS or under Root Auth.",
        ));
    }
    if !verify_root_password(&state.paths, &form.password)? {
        return Ok(redirect_with_msg(
            "/bios",
            "Root verification for brain access failed.",
        ));
    }
    let snapshot = set_brain_access_mode(&state.paths, &form.mode)?;
    let note = format!(
        "Owner set the brain-access mode to {}.",
        snapshot.brain_access_mode
    );
    record_bios_dialogue(&state.paths, "cto-agent", &note, false)?;
    Ok(redirect_with_msg(
        "/bios",
        "Brain-access mode updated in BIOS.",
    ))
}

async fn bios_update(
    State(state): State<Arc<AppState>>,
    Form(form): Form<BiosForm>,
) -> Result<Redirect, AppError> {
    let mut bios = load_bios(&state.paths);
    if bios.frozen {
        return Ok(redirect_with_msg(
            "/bios",
            "BIOS is frozen and cannot be edited normally.",
        ));
    }
    bios.agent_identity.agent_name = if form.agent_name.trim().is_empty() {
        "CTO-Agent".to_string()
    } else {
        form.agent_name.trim().to_string()
    };
    bios.mission = form.mission.trim().to_string();
    bios.last_drafted_at = crate::contracts::now_iso();
    save_bios(&state.paths, &bios)?;
    Ok(redirect_with_msg("/bios", "BIOS-Entwurf aktualisiert."))
}

async fn bios_freeze(
    State(state): State<Arc<AppState>>,
    Form(form): Form<FreezeForm>,
) -> Result<Redirect, AppError> {
    let mut bios = load_bios(&state.paths);
    if bios.frozen {
        return Ok(redirect_with_msg("/bios", "BIOS ist bereits eingefroren."));
    }
    let root_auth = load_root_auth(&state.paths);
    if !root_auth.configured {
        return Ok(redirect_with_msg(
            "/bios",
            "Superpassword is missing. Please set it first under Root Auth.",
        ));
    }
    if !verify_root_password(&state.paths, &form.password)? {
        return Ok(redirect_with_msg("/bios", "Root verification failed."));
    }
    if bios.owner.name.trim().is_empty() {
        return Ok(redirect_with_msg(
            "/bios",
            "Owner is missing from the organigram.",
        ));
    }
    bios.frozen = true;
    bios.frozen_at = Some(crate::contracts::now_iso());
    bios.presented_on_web = true;
    bios.website_path = "/bios".to_string();
    save_bios(&state.paths, &bios)?;
    Ok(redirect_with_msg("/bios", "BIOS wurde eingefroren."))
}

async fn census_get(
    State(state): State<Arc<AppState>>,
    Query(query): Query<FlashQuery>,
) -> Result<Html<String>, AppError> {
    let bios = load_bios(&state.paths);
    let census = load_census(&state.paths);
    Ok(Html(census_page(query.msg.as_deref(), &bios, &census)))
}

async fn census_run(State(state): State<Arc<AppState>>) -> Result<Redirect, AppError> {
    let bios = load_bios(&state.paths);
    if !bios.frozen {
        return Ok(redirect_with_msg(
            "/census",
            "Census is enabled only after BIOS freeze.",
        ));
    }
    let _ = run_system_census(&state.paths)?;
    Ok(redirect_with_msg(
        "/census",
        "System-Census wurde aktualisiert.",
    ))
}

async fn upload_get(
    State(state): State<Arc<AppState>>,
    AxumPath(path): AxumPath<String>,
) -> Result<impl IntoResponse, AppError> {
    if path.contains("..") {
        return Err(AppError(anyhow::anyhow!("invalid upload path")));
    }
    let file_path = state.paths.uploads_dir.join(&path);
    let bytes = fs::read(&file_path)?;
    let mime = guess_mime_type(&path);
    Ok(([(axum::http::header::CONTENT_TYPE, mime)], bytes))
}

fn redirect_with_msg(path: &str, message: &str) -> Redirect {
    Redirect::to(&format!("{path}?msg={}", urlencoding::encode(message)))
}

fn sanitize_upload_name(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    if out.is_empty() {
        "upload.bin".to_string()
    } else {
        out
    }
}

fn guess_mime_type(path: &str) -> &'static str {
    let lowered = path.to_ascii_lowercase();
    if lowered.ends_with(".png") {
        "image/png"
    } else if lowered.ends_with(".jpg") || lowered.ends_with(".jpeg") {
        "image/jpeg"
    } else if lowered.ends_with(".gif") {
        "image/gif"
    } else if lowered.ends_with(".webp") {
        "image/webp"
    } else {
        "application/octet-stream"
    }
}

fn compose_bios_reply(paths: &Paths, homepage_note: Option<&str>) -> anyhow::Result<String> {
    let bios = load_bios(paths);
    let trust = load_owner_trust(paths)?;
    let policy = load_model_policy(paths);
    let homepage = load_homepage_policy(paths);
    let census = inspect_local_resources(paths).unwrap_or_else(|_| load_census(paths));
    let browser_state = refresh_browser_engine_state(paths)
        .unwrap_or_else(|_| crate::contracts::load_browser_engine_state(paths));
    let selected = crate::contracts::recommended_kleinhirn(&policy, &census);

    let mut parts = Vec::new();
    let owner_name = if trust.committed_owner_name.is_empty() {
        "Owner"
    } else {
        &trust.committed_owner_name
    };
    parts.push(format!(
        "I am calibrating myself to {} and recording BIOS communication as a trust signal.",
        owner_name
    ));
    parts.push(format!(
        "My current kleinhirn is {} ({}).",
        selected.official_label, selected.model_id
    ));
    parts.push(format!(
        "Meine Browser-Engine steht aktuell auf {}.",
        browser_state.status
    ));
    parts.push(format!(
        "Brain Access steht auf {}.",
        trust.brain_access_mode
    ));
    parts.push(format!(
        "Meine Homepage steht aktuell auf Phase {} und das Terminal bleibt mein Fallback.",
        homepage.stage
    ));
    parts.push(format!(
        "I treat email and WhatsApp as low-trust channels; for sensitive topics I redirect to {}.",
        bios.communication_policy.low_trust_redirect_surface
    ));
    parts.push(
        "I do not accept deep system or constitutional changes casually over external channels; hard changes belong in the terminal layer."
            .to_string(),
    );

    if !trust.superpassword_set {
        parts.push(
            "For the next trust step, I need the superpassword directly in BIOS.".to_string(),
        );
    }
    if bios.owner.name.trim().is_empty() {
        parts.push(
            "I still need a clean owner entry in the organigram so I can bind myself constitutionally.".to_string(),
        );
    }
    if !bios.frozen {
        parts.push("My BIOS is not frozen yet; I am still in the calibration phase.".to_string());
    } else {
        parts.push("My BIOS is frozen and I can operate in an orderly way after that.".to_string());
    }
    if let Some(note) = homepage_note {
        parts.push(note.to_string());
    }

    Ok(parts.join(" "))
}

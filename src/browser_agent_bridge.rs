use crate::contracts::ModelPolicy;
use crate::contracts::Paths;
use crate::contracts::SystemCensus;
use crate::contracts::load_census;
use crate::contracts::load_model_policy;
use crate::contracts::now_iso;
use crate::contracts::recommended_browser_vision_kleinhirn;
use crate::contracts::recommended_kleinhirn;
use crate::storage::save_json;
use anyhow::Context;
use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;

const DEFAULT_BROWSER_AGENT_BRIDGE_PORT: u16 = 8765;
const DEFAULT_BROWSER_AGENT_LEASE_MS: i64 = 120_000;
const DEFAULT_BROWSER_AGENT_WAIT_POLL_MS: u64 = 1_200;

static BRIDGE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn bridge_lock() -> &'static Mutex<()> {
    BRIDGE_LOCK.get_or_init(|| Mutex::new(()))
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAgentWorkerRecord {
    pub worker_id: String,
    #[serde(default)]
    pub extension_id: String,
    #[serde(default)]
    pub extension_version: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAgentJobRecord {
    pub job_id: String,
    pub request: Value,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub leased_at: String,
    #[serde(default)]
    pub completed_at: String,
    #[serde(default)]
    pub result_delivered_at: String,
    #[serde(default)]
    pub worker_id: String,
    #[serde(default)]
    pub worker: Option<Value>,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub result: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAgentBridgeState {
    pub base_url: String,
    pub bridge_port: u16,
    pub extension_workspace: String,
    pub manifest_path: String,
    pub queued_jobs: usize,
    pub leased_jobs: usize,
    pub terminal_jobs: usize,
    pub active_workers: Vec<BrowserAgentWorkerRecord>,
    pub recent_jobs: Vec<BrowserAgentJobRecord>,
    pub runtime_config: Value,
}

pub fn browser_agent_bridge_port() -> u16 {
    std::env::var("CTO_AGENT_BROWSER_AGENT_BRIDGE_PORT")
        .ok()
        .and_then(|raw| raw.parse::<u16>().ok())
        .filter(|port| *port > 0)
        .unwrap_or(DEFAULT_BROWSER_AGENT_BRIDGE_PORT)
}

pub fn browser_agent_bridge_base_url() -> String {
    format!("http://127.0.0.1:{}", browser_agent_bridge_port())
}

pub fn browser_agent_extension_workspace(paths: &Paths) -> PathBuf {
    paths.root.join("browser_agent/extension")
}

pub fn browser_agent_extension_manifest_path(paths: &Paths) -> PathBuf {
    browser_agent_extension_workspace(paths).join("manifest.json")
}

pub fn ensure_browser_agent_bridge(paths: &Paths) -> anyhow::Result<()> {
    for dir in [
        bridge_root(paths),
        bridge_jobs_dir(paths),
        bridge_workers_dir(paths),
    ] {
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    }
    Ok(())
}

pub fn create_browser_agent_job(
    paths: &Paths,
    request: Value,
) -> anyhow::Result<BrowserAgentJobRecord> {
    let _guard = bridge_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("browser agent bridge lock poisoned"))?;
    ensure_browser_agent_bridge(paths)?;
    reclaim_expired_leases_locked(paths)?;
    let job_id = request_job_id(&request)
        .unwrap_or_else(|| format!("browser-job-{}", Utc::now().timestamp_millis()));
    let now = now_iso();
    let job = BrowserAgentJobRecord {
        job_id: job_id.clone(),
        request,
        status: "queued".to_string(),
        created_at: now.clone(),
        updated_at: now,
        leased_at: String::new(),
        completed_at: String::new(),
        result_delivered_at: String::new(),
        worker_id: String::new(),
        worker: None,
        error: String::new(),
        result: None,
    };
    save_json(&bridge_job_path(paths, &job_id), &job)?;
    Ok(job)
}

pub fn load_browser_agent_job(
    paths: &Paths,
    job_id: &str,
) -> anyhow::Result<Option<BrowserAgentJobRecord>> {
    let path = bridge_job_path(paths, job_id);
    if !path.exists() {
        return Ok(None);
    }
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let job = serde_json::from_str::<BrowserAgentJobRecord>(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(job))
}

pub fn load_browser_agent_bridge_state(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<BrowserAgentBridgeState> {
    let _guard = bridge_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("browser agent bridge lock poisoned"))?;
    ensure_browser_agent_bridge(paths)?;
    reclaim_expired_leases_locked(paths)?;
    let workers = load_all_workers_locked(paths)?;
    let jobs = load_all_jobs_locked(paths)?;
    let queued_jobs = jobs.iter().filter(|job| job.status == "queued").count();
    let leased_jobs = jobs.iter().filter(|job| job.status == "leased").count();
    let terminal_jobs = jobs
        .iter()
        .filter(|job| matches!(job.status.as_str(), "completed" | "failed"))
        .count();
    Ok(BrowserAgentBridgeState {
        base_url: browser_agent_bridge_base_url(),
        bridge_port: browser_agent_bridge_port(),
        extension_workspace: browser_agent_extension_workspace(paths)
            .display()
            .to_string(),
        manifest_path: browser_agent_extension_manifest_path(paths)
            .display()
            .to_string(),
        queued_jobs,
        leased_jobs,
        terminal_jobs,
        active_workers: workers.into_iter().take(limit).collect(),
        recent_jobs: jobs.into_iter().take(limit).collect(),
        runtime_config: browser_agent_runtime_config(paths),
    })
}

pub fn lease_next_browser_agent_job(
    paths: &Paths,
    worker_id: &str,
    extension_id: Option<&str>,
    extension_version: Option<&str>,
) -> anyhow::Result<(BrowserAgentWorkerRecord, Option<BrowserAgentJobRecord>)> {
    let _guard = bridge_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("browser agent bridge lock poisoned"))?;
    ensure_browser_agent_bridge(paths)?;
    reclaim_expired_leases_locked(paths)?;
    let worker = BrowserAgentWorkerRecord {
        worker_id: worker_id.trim().to_string(),
        extension_id: extension_id.unwrap_or("").trim().to_string(),
        extension_version: extension_version.unwrap_or("").trim().to_string(),
        updated_at: now_iso(),
    };
    save_json(&bridge_worker_path(paths, &worker.worker_id), &worker)?;

    let mut jobs = load_all_jobs_locked(paths)?;
    jobs.sort_by(|left, right| left.created_at.cmp(&right.created_at));
    for mut job in jobs {
        if job.status != "queued" {
            continue;
        }
        job.status = "leased".to_string();
        job.worker_id = worker.worker_id.clone();
        job.leased_at = now_iso();
        job.updated_at = job.leased_at.clone();
        save_json(&bridge_job_path(paths, &job.job_id), &job)?;
        return Ok((worker, Some(job)));
    }
    Ok((worker, None))
}

pub fn complete_browser_agent_job(
    paths: &Paths,
    job_id: &str,
    worker_id: &str,
    worker: Option<Value>,
    result: Value,
) -> anyhow::Result<Option<BrowserAgentJobRecord>> {
    let _guard = bridge_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("browser agent bridge lock poisoned"))?;
    ensure_browser_agent_bridge(paths)?;
    let Some(mut job) = load_browser_agent_job(paths, job_id)? else {
        return Ok(None);
    };
    let now = now_iso();
    job.worker_id = worker_id.trim().to_string();
    job.worker = worker;
    job.error = result
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    job.status = if result.get("ok").and_then(Value::as_bool).unwrap_or(true) {
        "completed".to_string()
    } else {
        "failed".to_string()
    };
    job.result = Some(result);
    job.completed_at = now.clone();
    job.updated_at = now;
    save_json(&bridge_job_path(paths, &job.job_id), &job)?;
    Ok(Some(job))
}

pub fn wait_for_browser_agent_job(
    paths: &Paths,
    job_id: &str,
    timeout_ms: u64,
) -> anyhow::Result<Option<BrowserAgentJobRecord>> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(500));
    loop {
        let current = load_browser_agent_job(paths, job_id)?;
        if current
            .as_ref()
            .map(|job| matches!(job.status.as_str(), "completed" | "failed"))
            .unwrap_or(false)
        {
            return Ok(current);
        }
        if Instant::now() >= deadline {
            return Ok(current);
        }
        std::thread::sleep(Duration::from_millis(DEFAULT_BROWSER_AGENT_WAIT_POLL_MS));
    }
}

pub fn browser_agent_runtime_config(paths: &Paths) -> Value {
    let runtime_env = load_runtime_env_map(paths).unwrap_or_default();
    let model_policy = load_model_policy(paths);
    let census = load_census(paths);
    browser_agent_runtime_config_from(&runtime_env, &model_policy, &census, paths)
}

fn browser_agent_runtime_config_from(
    runtime_env: &BTreeMap<String, String>,
    model_policy: &ModelPolicy,
    census: &SystemCensus,
    paths: &Paths,
) -> Value {
    let planner_selected = recommended_kleinhirn(model_policy, census);
    let vision_selected =
        recommended_browser_vision_kleinhirn(model_policy, census).unwrap_or(planner_selected);
    let runtime_model = runtime_env
        .get("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL")
        .cloned()
        .or_else(|| planner_selected.runtime_model_id.clone())
        .unwrap_or_else(|| planner_selected.model_id.clone());
    let official_label = runtime_env
        .get("CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL")
        .cloned()
        .unwrap_or_else(|| planner_selected.official_label.clone());
    let preferred_vision_runtime_model = vision_selected
        .runtime_model_id
        .clone()
        .unwrap_or_else(|| vision_selected.model_id.clone());
    let base_url = runtime_env
        .get("CTO_AGENT_KLEINHIRN_BASE_URL")
        .cloned()
        .or_else(|| {
            runtime_env
                .get("CTO_AGENT_KLEINHIRN_PORT")
                .map(|port| format!("http://127.0.0.1:{port}/v1"))
        })
        .unwrap_or_else(|| "http://127.0.0.1:1234/v1".to_string());
    let api_key = runtime_env
        .get("CTO_AGENT_KLEINHIRN_API_KEY")
        .cloned()
        .unwrap_or_else(|| "local-kleinhirn".to_string());
    let provider_id = if runtime_model.to_lowercase().contains("qwen") {
        "local_qwen"
    } else {
        "openai"
    };
    let vision_provider_id = "custom_openai";
    json!({
        "bridgeBaseUrl": browser_agent_bridge_base_url(),
        "bridgePort": browser_agent_bridge_port(),
        "localBaseUrl": base_url,
        "localApiKey": api_key,
        "localModelId": runtime_model,
        "officialLabel": official_label,
        "preferredVisionModelRef": format!("{vision_provider_id}>{preferred_vision_runtime_model}"),
        "visionModelRef": format!("{vision_provider_id}>{preferred_vision_runtime_model}"),
        "visionBaseUrl": base_url,
        "visionApiKey": api_key,
        "preferredVisionBaseUrl": base_url,
        "preferredVisionApiKey": api_key,
        "preferredVisionPolicyModelId": vision_selected.model_id.clone(),
        "preferredVisionOfficialLabel": vision_selected.official_label.clone(),
        "plannerModelRef": format!("{provider_id}>{runtime_model}"),
        "extensionWorkspace": browser_agent_extension_workspace(paths).display().to_string(),
        "manifestPath": browser_agent_extension_manifest_path(paths).display().to_string(),
        "reloadKind": "extension_reload",
    })
}

fn reclaim_expired_leases_locked(paths: &Paths) -> anyhow::Result<()> {
    let now = Utc::now();
    for mut job in load_all_jobs_locked(paths)? {
        if job.status != "leased" {
            continue;
        }
        let leased_at = parse_timestamp(&job.leased_at);
        let expired = leased_at
            .map(|timestamp| {
                now.signed_duration_since(timestamp).num_milliseconds()
                    > DEFAULT_BROWSER_AGENT_LEASE_MS
            })
            .unwrap_or(true);
        if !expired {
            continue;
        }
        job.status = "queued".to_string();
        job.updated_at = now_iso();
        job.leased_at.clear();
        job.worker_id.clear();
        job.worker = None;
        save_json(&bridge_job_path(paths, &job.job_id), &job)?;
    }
    Ok(())
}

fn bridge_root(paths: &Paths) -> PathBuf {
    paths.runtime_dir.join("browser-agent-bridge")
}

fn bridge_jobs_dir(paths: &Paths) -> PathBuf {
    bridge_root(paths).join("jobs")
}

fn bridge_workers_dir(paths: &Paths) -> PathBuf {
    bridge_root(paths).join("workers")
}

fn bridge_job_path(paths: &Paths, job_id: &str) -> PathBuf {
    bridge_jobs_dir(paths).join(format!("{job_id}.json"))
}

fn bridge_worker_path(paths: &Paths, worker_id: &str) -> PathBuf {
    bridge_workers_dir(paths).join(format!("{worker_id}.json"))
}

fn load_all_jobs_locked(paths: &Paths) -> anyhow::Result<Vec<BrowserAgentJobRecord>> {
    let mut jobs = load_records_from_dir::<BrowserAgentJobRecord>(&bridge_jobs_dir(paths))?;
    jobs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    Ok(jobs)
}

fn load_all_workers_locked(paths: &Paths) -> anyhow::Result<Vec<BrowserAgentWorkerRecord>> {
    let mut workers =
        load_records_from_dir::<BrowserAgentWorkerRecord>(&bridge_workers_dir(paths))?;
    workers.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    Ok(workers)
}

fn load_records_from_dir<T>(dir: &Path) -> anyhow::Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut records = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let record = serde_json::from_str::<T>(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        records.push(record);
    }
    Ok(records)
}

fn request_job_id(request: &Value) -> Option<String> {
    request
        .get("requestId")
        .and_then(Value::as_str)
        .or_else(|| request.get("request_id").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn parse_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|timestamp| timestamp.with_timezone(&Utc))
}

fn load_runtime_env_map(paths: &Paths) -> anyhow::Result<BTreeMap<String, String>> {
    let env_path = paths.root.join("runtime/kleinhirn.env");
    let text = fs::read_to_string(&env_path)
        .with_context(|| format!("failed to read {}", env_path.display()))?;
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = trimmed.split_once('=') else {
            continue;
        };
        let value = raw_value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        values.insert(key.trim().to_string(), value);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::ensure_contract_files;
    use std::fs;
    use std::path::PathBuf;

    fn test_paths(root: PathBuf) -> Paths {
        let contracts_dir = root.join("contracts");
        let history_dir = contracts_dir.join("history");
        let models_dir = contracts_dir.join("models");
        let homepage_dir = contracts_dir.join("homepage");
        let bootstrap_dir = contracts_dir.join("bootstrap");
        let context_dir = contracts_dir.join("context");
        let system_dir = contracts_dir.join("system");
        let browser_dir = contracts_dir.join("browser");
        let runtime_dir = root.join("runtime");
        let uploads_dir = runtime_dir.join("uploads");
        let browser_artifacts_dir = runtime_dir.join("browser");
        let recovery_dir = runtime_dir.join("recovery");
        let certs_dir = runtime_dir.join("certs");
        Paths {
            root: root.clone(),
            contracts_dir: contracts_dir.clone(),
            runtime_dir: runtime_dir.clone(),
            uploads_dir,
            browser_artifacts_dir,
            recovery_dir,
            history_dir: history_dir.clone(),
            models_dir: models_dir.clone(),
            homepage_dir: homepage_dir.clone(),
            bootstrap_dir: bootstrap_dir.clone(),
            context_dir: context_dir.clone(),
            system_dir: system_dir.clone(),
            browser_dir: browser_dir.clone(),
            genome_path: contracts_dir.join("genome/genome.json"),
            bios_path: contracts_dir.join("bios/bios.json"),
            org_path: contracts_dir.join("org/organigram.json"),
            root_auth_path: contracts_dir.join("root_auth/root_auth.json"),
            model_policy_path: models_dir.join("model-policy.json"),
            homepage_policy_path: homepage_dir.join("homepage-policy.json"),
            bootstrap_task_pack_path: bootstrap_dir.join("bootstrap-task-pack.json"),
            installation_bootstrap_path: bootstrap_dir.join("installation-bootstrap.json"),
            context_policy_path: context_dir.join("context-policy.json"),
            context_optimization_policy_path: context_dir.join("context-optimization-policy.json"),
            context_query_tool_contract_path: context_dir.join("context-query-tool-contract.json"),
            context_governance_policy_path: context_dir.join("context-governance-policy.json"),
            mode_system_policy_path: system_dir.join("mode-system-policy.json"),
            loop_safety_policy_path: system_dir.join("loop-safety-policy.json"),
            execution_authority_policy_path: system_dir.join("execution-authority-policy.json"),
            browser_engine_policy_path: browser_dir.join("browser-engine-policy.json"),
            browser_capability_policy_path: browser_dir.join("browser-capability-policy.json"),
            browser_subworker_policy_path: browser_dir.join("browser-subworker-policy.json"),
            self_preservation_state_path: system_dir.join("self-preservation-state.json"),
            origin_story_path: history_dir.join("origin-story.md"),
            creation_ledger_path: history_dir.join("creation-ledger.md"),
            boot_log_path: history_dir.join("boot-log.jsonl"),
            agent_state_path: runtime_dir.join("state/agent_state.json"),
            system_census_path: runtime_dir.join("state/system_census.json"),
            browser_engine_state_path: runtime_dir.join("state/browser_engine_state.json"),
            runtime_db_path: runtime_dir.join("cto_agent.db"),
            attach_socket_path: runtime_dir.join("cto-agent-test.sock"),
            runtime_lock_path: runtime_dir.join("cto-agent.lock"),
            pending_hard_reset_report_path: runtime_dir
                .join("recovery/pending-hard-reset-report.json"),
            certs_dir: certs_dir.clone(),
            tls_cert_path: certs_dir.join("tls-cert.pem"),
            tls_key_path: certs_dir.join("tls-key.pem"),
        }
    }

    #[test]
    fn browser_runtime_config_keeps_planner_runtime_but_prefers_qwen35_for_vision() {
        let root = std::env::temp_dir().join(format!(
            "cto-browser-runtime-config-{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let paths = test_paths(root.clone());
        ensure_contract_files(&paths).expect("contracts should initialize");
        let runtime_env = BTreeMap::from([
            (
                "CTO_AGENT_KLEINHIRN_RUNTIME_MODEL".to_string(),
                "openai/gpt-oss-20b".to_string(),
            ),
            (
                "CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL".to_string(),
                "GPT-OSS 20B".to_string(),
            ),
        ]);
        let policy = load_model_policy(&paths);
        let census = SystemCensus {
            cpu_threads: Some(32),
            total_memory_gb: Some(128),
            gpu_count: Some(5),
            total_gpu_memory_gb: Some(100),
            max_single_gpu_memory_gb: Some(20),
            ..SystemCensus::default()
        };
        let config = browser_agent_runtime_config_from(&runtime_env, &policy, &census, &paths);
        assert_eq!(
            config.get("plannerModelRef").and_then(Value::as_str),
            Some("openai>openai/gpt-oss-20b")
        );
        assert_eq!(
            config
                .get("preferredVisionModelRef")
                .and_then(Value::as_str),
            Some("custom_openai>openai/gpt-oss-20b")
        );
        assert_eq!(
            config.get("visionModelRef").and_then(Value::as_str),
            Some("custom_openai>openai/gpt-oss-20b")
        );
        let _ = fs::remove_dir_all(root);
    }
}

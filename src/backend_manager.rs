use anyhow::Context;
use anyhow::Result;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::thread;
use std::time::Duration;

use crate::execution_baseline;
use crate::runtime_config;

const SUPERVISOR_POLL_SECS: u64 = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManagedLauncherKind {
    VllmServe,
    SpeachesCpu,
}

#[derive(Debug, Clone)]
struct ManagedBackendSpec {
    display_model: String,
    request_model: String,
    port: u16,
    health_path: &'static str,
    launcher_kind: ManagedLauncherKind,
    compute_target: Option<execution_baseline::ComputeTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManagedBackendRole {
    Chat,
    Embedding,
    Stt,
    Tts,
}

impl ManagedBackendRole {
    fn as_env_value(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Embedding => "embedding",
            Self::Stt => "stt",
            Self::Tts => "tts",
        }
    }

    fn pid_file_name(self) -> &'static str {
        match self {
            Self::Chat => "ctox_chat_backend.pid",
            Self::Embedding => "ctox_embedding_backend.pid",
            Self::Stt => "ctox_stt_backend.pid",
            Self::Tts => "ctox_tts_backend.pid",
        }
    }

    fn log_file_name(self) -> &'static str {
        match self {
            Self::Chat => "ctox_chat_backend.log",
            Self::Embedding => "ctox_embedding_backend.log",
            Self::Stt => "ctox_stt_backend.log",
            Self::Tts => "ctox_tts_backend.log",
        }
    }

    fn spec(self, root: &Path) -> ManagedBackendSpec {
        match self {
            Self::Chat => {
                let runtime = runtime_config::effective_chat_model(root)
                    .or_else(|| runtime_config::env_or_config(root, "CTOX_VLLM_SERVE_MODEL"))
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| {
                        execution_baseline::default_runtime_config(
                            execution_baseline::LocalModelFamily::GptOss,
                        )
                        .model
                    });
                let fallback_runtime = execution_baseline::runtime_config_for_model(&runtime)
                    .unwrap_or_else(|_| {
                        execution_baseline::default_runtime_config(
                            execution_baseline::LocalModelFamily::GptOss,
                        )
                    });
                let port = runtime_config::env_or_config(root, "CTOX_VLLM_SERVE_PORT")
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(fallback_runtime.port);
                ManagedBackendSpec {
                    display_model: runtime.clone(),
                    request_model: runtime,
                    port,
                    health_path: "/health",
                    launcher_kind: ManagedLauncherKind::VllmServe,
                    compute_target: None,
                }
            }
            Self::Embedding => {
                let configured_model = runtime_config::env_or_config(root, "CTOX_EMBEDDING_MODEL");
                let selection = execution_baseline::auxiliary_model_selection(
                    execution_baseline::AuxiliaryRole::Embedding,
                    configured_model.as_deref(),
                );
                let port = runtime_config::env_or_config(root, "CTOX_EMBEDDING_PORT")
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(selection.default_port);
                ManagedBackendSpec {
                    display_model: selection.choice.to_string(),
                    request_model: selection.request_model.to_string(),
                    port,
                    health_path: "/health",
                    launcher_kind: ManagedLauncherKind::VllmServe,
                    compute_target: Some(selection.compute_target),
                }
            }
            Self::Stt => {
                let configured_model = runtime_config::env_or_config(root, "CTOX_STT_MODEL");
                let selection = execution_baseline::auxiliary_model_selection(
                    execution_baseline::AuxiliaryRole::Stt,
                    configured_model.as_deref(),
                );
                let port = runtime_config::env_or_config(root, "CTOX_STT_PORT")
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(selection.default_port);
                ManagedBackendSpec {
                    display_model: selection.choice.to_string(),
                    request_model: selection.request_model.to_string(),
                    port,
                    health_path: if selection.backend_kind
                        == execution_baseline::AuxiliaryBackendKind::Speaches
                    {
                        "/v1/models"
                    } else {
                        "/health"
                    },
                    launcher_kind: if selection.backend_kind
                        == execution_baseline::AuxiliaryBackendKind::Speaches
                    {
                        ManagedLauncherKind::SpeachesCpu
                    } else {
                        ManagedLauncherKind::VllmServe
                    },
                    compute_target: Some(selection.compute_target),
                }
            }
            Self::Tts => {
                let configured_model = runtime_config::env_or_config(root, "CTOX_TTS_MODEL");
                let selection = execution_baseline::auxiliary_model_selection(
                    execution_baseline::AuxiliaryRole::Tts,
                    configured_model.as_deref(),
                );
                let port = runtime_config::env_or_config(root, "CTOX_TTS_PORT")
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(selection.default_port);
                ManagedBackendSpec {
                    display_model: selection.choice.to_string(),
                    request_model: selection.request_model.to_string(),
                    port,
                    health_path: if selection.backend_kind
                        == execution_baseline::AuxiliaryBackendKind::Speaches
                    {
                        "/v1/models"
                    } else {
                        "/health"
                    },
                    launcher_kind: if selection.backend_kind
                        == execution_baseline::AuxiliaryBackendKind::Speaches
                    {
                        ManagedLauncherKind::SpeachesCpu
                    } else {
                        ManagedLauncherKind::VllmServe
                    },
                    compute_target: Some(selection.compute_target),
                }
            }
        }
    }
}

pub fn start_backend_supervisor(root: PathBuf) {
    thread::spawn(move || loop {
        if let Err(err) = ensure_persistent_backends(&root) {
            eprintln!("ctox backend supervisor error: {err}");
        }
        thread::sleep(Duration::from_secs(SUPERVISOR_POLL_SECS));
    });
}

pub fn ensure_persistent_backends(root: &Path) -> Result<()> {
    ensure_proxy_process(root)?;
    for role in [
        ManagedBackendRole::Chat,
        ManagedBackendRole::Embedding,
        ManagedBackendRole::Stt,
        ManagedBackendRole::Tts,
    ] {
        ensure_backend_process(root, role)?;
    }
    Ok(())
}

pub fn shutdown_persistent_backends(root: &Path) -> Result<()> {
    stop_process(root, proxy_pid_path(root))?;
    for role in [
        ManagedBackendRole::Chat,
        ManagedBackendRole::Embedding,
        ManagedBackendRole::Stt,
        ManagedBackendRole::Tts,
    ] {
        stop_process(root, backend_pid_path(root, role))?;
    }
    Ok(())
}

fn ensure_proxy_process(root: &Path) -> Result<()> {
    let host = runtime_config::env_or_config(root, "CTOX_PROXY_HOST")
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = runtime_config::env_or_config(root, "CTOX_PROXY_PORT")
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(12434);
    let health_url = format!("http://{host}:{port}/ctox/telemetry");
    if health_check(&health_url) {
        return Ok(());
    }

    let pid_path = proxy_pid_path(root);
    stop_process(root, pid_path.clone())?;

    let runtime_dir = root.join("runtime");
    std::fs::create_dir_all(&runtime_dir)
        .with_context(|| format!("failed to create runtime dir {}", runtime_dir.display()))?;
    let log_path = runtime_dir.join("ctox_proxy.log");
    let log_file = open_log_file(&log_path)?;
    let log_file_err = log_file
        .try_clone()
        .with_context(|| format!("failed to clone proxy log {}", log_path.display()))?;
    let exe = std::env::current_exe().context("failed to resolve current CTOX executable")?;
    let child = Command::new(exe)
        .arg("serve-responses-proxy")
        .current_dir(root)
        .env("CTOX_ROOT", root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn CTOX responses proxy")?;
    std::fs::write(&pid_path, format!("{}\n", child.id()))
        .with_context(|| format!("failed to write proxy pid file {}", pid_path.display()))?;
    Ok(())
}

fn ensure_backend_process(root: &Path, role: ManagedBackendRole) -> Result<()> {
    if role == ManagedBackendRole::Chat
        && runtime_config::env_or_config(root, "CTOX_CHAT_SOURCE")
            .map(|value| value.trim().eq_ignore_ascii_case("api"))
            .unwrap_or(false)
    {
        stop_process(root, backend_pid_path(root, role))?;
        return Ok(());
    }

    let spec = role.spec(root);
    if spec.request_model.trim().is_empty() {
        return Ok(());
    }
    let health_url = format!("http://127.0.0.1:{}{}", spec.port, spec.health_path);
    if health_check(&health_url) {
        return Ok(());
    }

    let pid_path = backend_pid_path(root, role);
    stop_process(root, pid_path.clone())?;

    let runtime_dir = root.join("runtime");
    std::fs::create_dir_all(&runtime_dir)
        .with_context(|| format!("failed to create runtime dir {}", runtime_dir.display()))?;
    let log_path = runtime_dir.join(role.log_file_name());
    let log_file = open_log_file(&log_path)?;
    let log_file_err = log_file
        .try_clone()
        .with_context(|| format!("failed to clone backend log {}", log_path.display()))?;
    let script_path = match spec.launcher_kind {
        ManagedLauncherKind::VllmServe => root.join("scripts/run_vllm_serve_backend.sh"),
        ManagedLauncherKind::SpeachesCpu => root.join("scripts/run_speaches_cpu_backend.sh"),
    };
    if !script_path.exists() {
        anyhow::bail!("backend launcher missing: {}", script_path.display());
    }
    let mut command = Command::new("bash");
    command
        .arg(script_path)
        .current_dir(root)
        .env("CTOX_ROOT", root)
        .env("CTOX_VLLM_SERVE_ROLE", role.as_env_value())
        .env("CTOX_VLLM_SERVE_MODEL_OVERRIDE", &spec.request_model);
    if let Some(compute_target) = spec.compute_target {
        command.env(
            "CTOX_VLLM_SERVE_COMPUTE_TARGET",
            compute_target.as_env_value(),
        );
    }
    if spec.launcher_kind == ManagedLauncherKind::SpeachesCpu {
        command
            .env("CTOX_AUX_CPU_ROLE", role.as_env_value())
            .env("CTOX_AUX_PORT", spec.port.to_string())
            .env("CTOX_AUX_REQUEST_MODEL", &spec.request_model);
    }
    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .with_context(|| {
            format!(
                "failed to spawn {} backend for {}",
                role.as_env_value(),
                spec.display_model
            )
        })?;
    std::fs::write(&pid_path, format!("{}\n", child.id()))
        .with_context(|| format!("failed to write backend pid file {}", pid_path.display()))?;
    Ok(())
}

fn health_check(url: &str) -> bool {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(1))
        .timeout_read(Duration::from_secs(2))
        .timeout_write(Duration::from_secs(2))
        .build();
    match agent.get(url).call() {
        Ok(response) => response.status() < 500,
        Err(ureq::Error::Status(code, _)) => code < 500,
        Err(_) => false,
    }
}

fn proxy_pid_path(root: &Path) -> PathBuf {
    root.join("runtime/ctox_proxy.pid")
}

fn backend_pid_path(root: &Path, role: ManagedBackendRole) -> PathBuf {
    root.join("runtime").join(role.pid_file_name())
}

fn open_log_file(path: &Path) -> Result<File> {
    File::options()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open log file {}", path.display()))
}

fn stop_process(root: &Path, pid_path: PathBuf) -> Result<()> {
    let Some(pid) = read_pid(&pid_path) else {
        return Ok(());
    };
    if process_is_alive(pid) {
        let status = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .current_dir(root)
            .status()
            .with_context(|| format!("failed to signal pid {pid}"))?;
        if !status.success() {
            anyhow::bail!("failed to stop pid {pid}");
        }
    }
    let _ = std::fs::remove_file(pid_path);
    Ok(())
}

fn read_pid(path: &Path) -> Option<u32> {
    let raw = std::fs::read_to_string(path).ok()?;
    raw.trim().parse::<u32>().ok()
}

fn process_is_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

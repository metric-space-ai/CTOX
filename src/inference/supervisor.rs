use anyhow::Context;
use anyhow::Result;
#[cfg(unix)]
use libc::getrlimit;
#[cfg(unix)]
use libc::rlimit;
#[cfg(unix)]
use libc::setrlimit;
#[cfg(unix)]
use libc::signal;
#[cfg(unix)]
use libc::RLIMIT_NOFILE;
#[cfg(unix)]
use libc::SIGHUP;
#[cfg(unix)]
use libc::SIGPIPE;
#[cfg(unix)]
use libc::SIG_IGN;
use std::fs::File;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use crate::inference::engine;
use crate::inference::runtime_env;
use crate::inference::runtime_plan;

const SUPERVISOR_POLL_SECS: u64 = 12;

#[cfg(unix)]
use libc::setsid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManagedLauncherKind {
    Engine,
    SpeachesCpu,
}

#[derive(Debug, Clone)]
struct ManagedBackendSpec {
    display_model: String,
    request_model: String,
    port: u16,
    health_path: &'static str,
    launcher_kind: ManagedLauncherKind,
    compute_target: Option<engine::ComputeTarget>,
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
                let runtime = runtime_env::effective_chat_model(root)
                    .or_else(|| runtime_env::env_or_config(root, "CTOX_ENGINE_MODEL"))
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| {
                        engine::default_runtime_config(engine::LocalModelFamily::GptOss).model
                    });
                let fallback_runtime =
                    engine::runtime_config_for_model(&runtime).unwrap_or_else(|_| {
                        engine::default_runtime_config(engine::LocalModelFamily::GptOss)
                    });
                let port = runtime_env::env_or_config(root, "CTOX_ENGINE_PORT")
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(fallback_runtime.port);
                ManagedBackendSpec {
                    display_model: runtime.clone(),
                    request_model: runtime,
                    port,
                    health_path: "/health",
                    launcher_kind: ManagedLauncherKind::Engine,
                    compute_target: None,
                }
            }
            Self::Embedding => {
                let configured_model = runtime_env::env_or_config(root, "CTOX_EMBEDDING_MODEL");
                let selection = engine::auxiliary_model_selection(
                    engine::AuxiliaryRole::Embedding,
                    configured_model.as_deref(),
                );
                let port = runtime_env::env_or_config(root, "CTOX_EMBEDDING_PORT")
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(selection.default_port);
                ManagedBackendSpec {
                    display_model: selection.choice.to_string(),
                    request_model: selection.request_model.to_string(),
                    port,
                    health_path: "/health",
                    launcher_kind: ManagedLauncherKind::Engine,
                    compute_target: Some(selection.compute_target),
                }
            }
            Self::Stt => {
                let configured_model = runtime_env::env_or_config(root, "CTOX_STT_MODEL");
                let selection = engine::auxiliary_model_selection(
                    engine::AuxiliaryRole::Stt,
                    configured_model.as_deref(),
                );
                let port = runtime_env::env_or_config(root, "CTOX_STT_PORT")
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(selection.default_port);
                ManagedBackendSpec {
                    display_model: selection.choice.to_string(),
                    request_model: selection.request_model.to_string(),
                    port,
                    health_path: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches
                    {
                        "/v1/models"
                    } else {
                        "/health"
                    },
                    launcher_kind: if selection.backend_kind
                        == engine::AuxiliaryBackendKind::Speaches
                    {
                        ManagedLauncherKind::SpeachesCpu
                    } else {
                        ManagedLauncherKind::Engine
                    },
                    compute_target: Some(selection.compute_target),
                }
            }
            Self::Tts => {
                let configured_model = runtime_env::env_or_config(root, "CTOX_TTS_MODEL");
                let selection = engine::auxiliary_model_selection(
                    engine::AuxiliaryRole::Tts,
                    configured_model.as_deref(),
                );
                let port = runtime_env::env_or_config(root, "CTOX_TTS_PORT")
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(selection.default_port);
                ManagedBackendSpec {
                    display_model: selection.choice.to_string(),
                    request_model: selection.request_model.to_string(),
                    port,
                    health_path: if selection.backend_kind == engine::AuxiliaryBackendKind::Speaches
                    {
                        "/v1/models"
                    } else {
                        "/health"
                    },
                    launcher_kind: if selection.backend_kind
                        == engine::AuxiliaryBackendKind::Speaches
                    {
                        ManagedLauncherKind::SpeachesCpu
                    } else {
                        ManagedLauncherKind::Engine
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
        if !managed_backend_enabled(root, role) {
            continue;
        }
        ensure_backend_process(root, role)?;
    }
    Ok(())
}

fn managed_backend_enabled(root: &Path, role: ManagedBackendRole) -> bool {
    match role {
        ManagedBackendRole::Chat => true,
        ManagedBackendRole::Embedding => runtime_env::auxiliary_backend_enabled(root, "EMBEDDING"),
        ManagedBackendRole::Stt => runtime_env::auxiliary_backend_enabled(root, "STT"),
        ManagedBackendRole::Tts => runtime_env::auxiliary_backend_enabled(root, "TTS"),
    }
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
    let host = runtime_env::env_or_config(root, "CTOX_PROXY_HOST")
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = runtime_env::env_or_config(root, "CTOX_PROXY_PORT")
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(12434);
    let health_url = format!("http://{host}:{port}/ctox/telemetry");
    if health_check(&health_url) {
        return Ok(());
    }

    let pid_path = proxy_pid_path(root);
    if read_pid(&pid_path)
        .filter(|pid| process_is_alive(*pid))
        .is_some()
    {
        return Ok(());
    }
    if let Some(listener_pid) = listening_pids_for_port(root, port)?.into_iter().next() {
        std::fs::write(&pid_path, format!("{listener_pid}\n")).with_context(|| {
            format!("failed to write proxy pid file {}", pid_path.display())
        })?;
        return Ok(());
    }
    stop_process(root, pid_path.clone())?;

    let runtime_dir = root.join("runtime");
    std::fs::create_dir_all(&runtime_dir)
        .with_context(|| format!("failed to create runtime dir {}", runtime_dir.display()))?;
    let log_path = runtime_dir.join("ctox_proxy.log");
    let log_file = open_log_file(&log_path)?;
    let log_file_err = log_file
        .try_clone()
        .with_context(|| format!("failed to clone proxy log {}", log_path.display()))?;
    let exe = std::env::current_exe()
        .context("failed to resolve current CTOX executable")
        .or_else(|_| {
            let candidate = root.join("target/release/ctox");
            if candidate.is_file() {
                Ok(candidate)
            } else {
                Err(anyhow::anyhow!(
                    "failed to resolve current CTOX executable and no runtime-local ctox binary exists at {}",
                    candidate.display()
                ))
            }
        })?;
    let mut command = Command::new("bash");
    command
        .arg("-lc")
        .arg("ulimit -n 65535; exec \"$1\" serve-responses-proxy")
        .arg("ctox-proxy-spawn")
        .arg(&exe)
        .current_dir(root)
        .env("CTOX_ROOT", root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err));
    configure_detached_process_group(&mut command);
    let child = command
        .spawn()
        .context("failed to spawn CTOX responses proxy")?;
    std::fs::write(&pid_path, format!("{}\n", child.id()))
        .with_context(|| format!("failed to write proxy pid file {}", pid_path.display()))?;
    Ok(())
}

fn ensure_backend_process(root: &Path, role: ManagedBackendRole) -> Result<()> {
    if role == ManagedBackendRole::Chat
        && runtime_env::env_or_config(root, "CTOX_CHAT_SOURCE")
            .map(|value| value.trim().eq_ignore_ascii_case("api"))
            .unwrap_or(false)
    {
        stop_process(root, backend_pid_path(root, role))?;
        return Ok(());
    }

    if role == ManagedBackendRole::Chat {
        let _ = runtime_plan::reconcile_chat_runtime_plan(root)?;
    }

    let spec = role.spec(root);
    if spec.request_model.trim().is_empty() {
        return Ok(());
    }
    let pid_path = backend_pid_path(root, role);
    let health_url = format!("http://127.0.0.1:{}{}", spec.port, spec.health_path);
    if health_check(&health_url) {
        if let Some(matched_pid) = matching_listener_pid_for_backend(root, spec.port, &spec)? {
            std::fs::write(&pid_path, format!("{matched_pid}\n")).with_context(|| {
                format!("failed to write backend pid file {}", pid_path.display())
            })?;
            return Ok(());
        }
        stop_processes_on_port(root, spec.port)?;
    }
    if read_pid(&pid_path)
        .filter(|pid| process_is_alive(*pid))
        .is_some()
    {
        return Ok(());
    }
    if let Some(matched_pid) = matching_listener_pid_for_backend(root, spec.port, &spec)? {
        std::fs::write(&pid_path, format!("{matched_pid}\n")).with_context(|| {
            format!("failed to write backend pid file {}", pid_path.display())
        })?;
        return Ok(());
    }

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
        ManagedLauncherKind::Engine => root.join("scripts/engine/run_engine.sh"),
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
        .env("CTOX_ENGINE_ROLE", role.as_env_value())
        .env("CTOX_ENGINE_MODEL_OVERRIDE", &spec.request_model);
    if let Some(compute_target) = spec.compute_target {
        command.env("CTOX_ENGINE_COMPUTE_TARGET", compute_target.as_env_value());
    }
    if role != ManagedBackendRole::Chat {
        command.env("CTOX_ENGINE_ENV_FILE", "/dev/null");
        match role {
            ManagedBackendRole::Embedding => {
                command
                    .env("CTOX_EMBEDDING_MODEL", &spec.request_model)
                    .env("CTOX_EMBEDDING_PORT", spec.port.to_string());
            }
            ManagedBackendRole::Stt => {
                command
                    .env("CTOX_STT_MODEL", &spec.request_model)
                    .env("CTOX_STT_PORT", spec.port.to_string());
            }
            ManagedBackendRole::Tts => {
                command
                    .env("CTOX_TTS_MODEL", &spec.request_model)
                    .env("CTOX_TTS_PORT", spec.port.to_string());
            }
            ManagedBackendRole::Chat => {}
        }
        if spec.compute_target == Some(engine::ComputeTarget::Gpu) {
            let visible_devices = match role {
                ManagedBackendRole::Embedding => {
                    runtime_env::env_or_config(root, "CTOX_EMBEDDING_CUDA_VISIBLE_DEVICES")
                }
                ManagedBackendRole::Stt => {
                    runtime_env::env_or_config(root, "CTOX_STT_CUDA_VISIBLE_DEVICES")
                }
                ManagedBackendRole::Tts => {
                    runtime_env::env_or_config(root, "CTOX_TTS_CUDA_VISIBLE_DEVICES")
                }
                ManagedBackendRole::Chat => None,
            }
            .or_else(|| runtime_env::env_or_config(root, "CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES"))
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "0".to_string());
            command.env("CTOX_ENGINE_CUDA_VISIBLE_DEVICES", visible_devices);
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
    }
    if spec.launcher_kind == ManagedLauncherKind::SpeachesCpu {
        command
            .env("CTOX_AUX_CPU_ROLE", role.as_env_value())
            .env("CTOX_AUX_PORT", spec.port.to_string())
            .env("CTOX_AUX_REQUEST_MODEL", &spec.request_model);
    }
    configure_detached_process_group(&mut command);
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

fn matching_listener_pid_for_backend(
    root: &Path,
    port: u16,
    spec: &ManagedBackendSpec,
) -> Result<Option<u32>> {
    let expected_port = format!("-p {port}");
    for pid in listening_pids_for_port(root, port)? {
        let Some(command) = process_command(root, pid)? else {
            continue;
        };
        if !command.contains(&expected_port) {
            continue;
        }
        if spec.launcher_kind == ManagedLauncherKind::Engine
            && command.contains(spec.request_model.as_str())
        {
            return Ok(Some(pid));
        }
        if spec.launcher_kind == ManagedLauncherKind::SpeachesCpu {
            return Ok(Some(pid));
        }
    }
    Ok(None)
}

fn listening_pids_for_port(root: &Path, port: u16) -> Result<Vec<u32>> {
    let output = Command::new("bash")
        .arg("-lc")
        .arg(format!("fuser {port}/tcp 2>/dev/null || true"))
        .current_dir(root)
        .output()
        .with_context(|| format!("failed to query listener pid for tcp/{port}"))?;
    let mut pids = output
        .stdout
        .split(|byte| byte.is_ascii_whitespace())
        .filter_map(|chunk| std::str::from_utf8(chunk).ok())
        .filter_map(|chunk| chunk.trim().parse::<u32>().ok())
        .collect::<Vec<_>>();
    pids.sort_unstable();
    pids.dedup();
    Ok(pids)
}

fn process_command(root: &Path, pid: u32) -> Result<Option<String>> {
    let output = Command::new("ps")
        .args(["-ww", "-o", "command=", "-p", &pid.to_string()])
        .current_dir(root)
        .output()
        .with_context(|| format!("failed to inspect command for pid {pid}"))?;
    if !output.status.success() {
        return Ok(None);
    }
    let command = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if command.is_empty() {
        return Ok(None);
    }
    Ok(Some(command))
}

fn stop_processes_on_port(root: &Path, port: u16) -> Result<()> {
    for pid in listening_pids_for_port(root, port)? {
        if pid == std::process::id() {
            continue;
        }
        let status = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .current_dir(root)
            .status()
            .with_context(|| format!("failed to signal pid {pid} on tcp/{port}"))?;
        if !status.success() {
            anyhow::bail!("failed to stop listener pid {pid} on tcp/{port}");
        }
        thread::sleep(Duration::from_millis(150));
        if listening_pids_for_port(root, port)?.contains(&pid) {
            let status = Command::new("kill")
                .arg("-KILL")
                .arg(pid.to_string())
                .current_dir(root)
                .status()
                .with_context(|| format!("failed to force-stop pid {pid} on tcp/{port}"))?;
            if !status.success() {
                anyhow::bail!("failed to force-stop listener pid {pid} on tcp/{port}");
            }
        }
    }
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
        let deadline = Instant::now() + Duration::from_secs(3);
        while process_is_alive(pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(100));
        }
        if process_is_alive(pid) {
            let kill_status = Command::new("kill")
                .arg("-KILL")
                .arg(pid.to_string())
                .current_dir(root)
                .status()
                .with_context(|| format!("failed to force-stop pid {pid}"))?;
            if !kill_status.success() && process_is_alive(pid) {
                anyhow::bail!("failed to force-stop pid {pid}");
            }
        }
    }
    let _ = std::fs::remove_file(pid_path);
    Ok(())
}

#[cfg(unix)]
fn configure_detached_process_group(command: &mut Command) {
    unsafe {
        command.pre_exec(|| {
            if setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            signal(SIGHUP, SIG_IGN);
            signal(SIGPIPE, SIG_IGN);
            let mut current = rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if getrlimit(RLIMIT_NOFILE, &mut current) == 0 {
                let target = 65_535 as libc::rlim_t;
                let raised = rlimit {
                    rlim_cur: std::cmp::min(target, current.rlim_max),
                    rlim_max: current.rlim_max,
                };
                let _ = setrlimit(RLIMIT_NOFILE, &raised);
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_detached_process_group(_command: &mut Command) {}

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

use crate::brain_runtime::load_kleinhirn_runtime_snapshot;
use crate::command_exec::run_one_shot_command;
use crate::command_exec::start_session;
use crate::contracts::BrowserEngineState;
use crate::contracts::Paths;
use crate::contracts::load_browser_engine_policy;
use crate::contracts::load_census;
use crate::contracts::load_model_policy;
use crate::contracts::now_iso;
use crate::contracts::recommended_browser_vision_kleinhirn;
use crate::contracts::write_browser_engine_state;
use crate::desktop_session::detect_desktop_session;
use crate::desktop_session::detect_desktop_session_env;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::record_agent_event;
use anyhow::Context;
use base64::Engine as _;
use codex_app_server_protocol::CommandExecParams;
use codex_app_server_protocol::CommandExecTerminalSize;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserActionDirective {
    pub action: String,
    pub url: Option<String>,
    pub output_path: Option<String>,
    pub wait_ms: Option<u64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub justification: Option<String>,
    pub question: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BrowserActionResult {
    pub status: String,
    pub stdout: String,
    pub stderr: String,
    pub artifact_path: Option<String>,
    pub browser_status: String,
}

pub fn refresh_browser_engine_state(paths: &Paths) -> anyhow::Result<BrowserEngineState> {
    let state = inspect_browser_engine(paths);
    write_browser_engine_state(paths, &state)?;
    Ok(state)
}

pub fn inspect_browser_engine(paths: &Paths) -> BrowserEngineState {
    let install_script = paths
        .root
        .join("scripts/install_browser_engine.sh")
        .display()
        .to_string();
    let chrome_binary = find_chrome_binary();
    let chrome_version = chrome_binary
        .as_ref()
        .and_then(|path| detect_chrome_version(path.as_path()));
    let desktop_available = detect_desktop_session().is_available();
    let headless_ready = chrome_binary.is_some();
    let interactive_ready = headless_ready && desktop_available;
    let status = if chrome_binary.is_none() {
        "missing_chrome"
    } else if interactive_ready {
        "ready_desktop"
    } else {
        "ready_headless_only"
    };

    BrowserEngineState {
        status: status.to_string(),
        chrome_binary: chrome_binary.map(|path| path.display().to_string()),
        chrome_version,
        desktop_available,
        headless_ready,
        interactive_ready,
        artifacts_dir: paths.browser_artifacts_dir.display().to_string(),
        install_script,
        last_checked_at: now_iso(),
        last_install_attempt_at: None,
    }
}

pub fn browser_status_text(paths: &Paths) -> anyhow::Result<String> {
    let policy = load_browser_engine_policy(paths);
    let state = refresh_browser_engine_state(paths)?;
    Ok(format!(
        "browser_engine_status={}; chrome_binary={}; chrome_version={}; desktop_available={}; headless_ready={}; interactive_ready={}; install_via_cli={}; install_script={}",
        state.status,
        state.chrome_binary.as_deref().unwrap_or("none"),
        state.chrome_version.as_deref().unwrap_or("unknown"),
        state.desktop_available,
        state.headless_ready,
        state.interactive_ready,
        policy.runtime_model.install_via_cli_engine,
        state.install_script
    ))
}

pub fn start_browser_install_session(paths: &Paths) -> anyhow::Result<String> {
    let session_id = format!("browser-install-{}", chrono::Utc::now().timestamp_millis());
    start_session(
        paths,
        CommandExecParams {
            command: install_command(paths),
            process_id: Some(session_id),
            tty: true,
            stream_stdin: false,
            stream_stdout_stderr: true,
            output_bytes_cap: None,
            disable_output_cap: false,
            disable_timeout: false,
            timeout_ms: Some(30 * 60 * 1000),
            cwd: Some(paths.root.clone()),
            env: detect_desktop_session_env(),
            size: Some(CommandExecTerminalSize {
                rows: 24,
                cols: 100,
            }),
            sandbox_policy: None,
        },
    )
}

pub fn start_browser_launch_session(paths: &Paths) -> anyhow::Result<String> {
    let session_id = format!("browser-launch-{}", chrono::Utc::now().timestamp_millis());
    start_session(
        paths,
        CommandExecParams {
            command: launch_command(paths),
            process_id: Some(session_id),
            tty: true,
            stream_stdin: false,
            stream_stdout_stderr: true,
            output_bytes_cap: None,
            disable_output_cap: false,
            disable_timeout: false,
            timeout_ms: Some(2 * 60 * 1000),
            cwd: Some(paths.root.clone()),
            env: detect_desktop_session_env(),
            size: Some(CommandExecTerminalSize {
                rows: 24,
                cols: 100,
            }),
            sandbox_policy: None,
        },
    )
}

pub fn run_browser_action(
    paths: &Paths,
    task: &TaskRecord,
    directive: &BrowserActionDirective,
) -> anyhow::Result<BrowserActionResult> {
    let initial_state = refresh_browser_engine_state(paths)?;
    let action = directive.action.trim();
    if action.is_empty() {
        anyhow::bail!("browserAction must not be empty");
    }

    let (command, artifact_path) = match action {
        "install_browser_engine" | "install_chrome" => (install_command(paths), None),
        "launch_browser_agent" | "launch_browser_agent_chrome" => (launch_command(paths), None),
        "dump_dom" => {
            let url = require_url(directive)?;
            let binary = require_chrome_binary(&initial_state)?;
            (
                headless_command(
                    binary,
                    directive.wait_ms,
                    vec!["--dump-dom".to_string(), url],
                ),
                None,
            )
        }
        "screenshot" => {
            let url = require_url(directive)?;
            let binary = require_chrome_binary(&initial_state)?;
            let artifact = resolve_artifact_path(
                paths,
                directive.output_path.as_deref(),
                "screenshot",
                "png",
            )?;
            let width = directive.width.unwrap_or(1440);
            let height = directive.height.unwrap_or(2000);
            let mut args = vec![
                format!("--window-size={},{}", width, height),
                "--hide-scrollbars".to_string(),
                format!("--screenshot={}", artifact.display()),
                url,
            ];
            let command = headless_command(binary, directive.wait_ms, std::mem::take(&mut args));
            (command, Some(artifact.display().to_string()))
        }
        "inspect_visual" => {
            let url = require_url(directive)?;
            let binary = require_chrome_binary(&initial_state)?;
            let artifact = resolve_artifact_path(
                paths,
                directive.output_path.as_deref(),
                "visual-inspect",
                "png",
            )?;
            let width = directive.width.unwrap_or(1440);
            let height = directive.height.unwrap_or(2000);
            let mut args = vec![
                format!("--window-size={},{}", width, height),
                "--hide-scrollbars".to_string(),
                format!("--screenshot={}", artifact.display()),
                url,
            ];
            let command = headless_command(binary, directive.wait_ms, std::mem::take(&mut args));
            (command, Some(artifact.display().to_string()))
        }
        "open_url" => {
            let url = require_url(directive)?;
            if !initial_state.interactive_ready {
                anyhow::bail!(
                    "browser engine interactive open requires a desktop session and installed Chrome"
                );
            }
            (interactive_open_command(&initial_state, &url)?, None)
        }
        other => anyhow::bail!("unsupported browserAction: {other}"),
    };

    let session_id = format!(
        "task-{}-browser-{}",
        task.id,
        chrono::Utc::now().timestamp_millis()
    );
    let timeout_ms = if matches!(
        action,
        "install_browser_engine"
            | "install_chrome"
            | "launch_browser_agent"
            | "launch_browser_agent_chrome"
    ) {
        30 * 60 * 1000
    } else {
        90_000
    };
    let result = run_one_shot_command(
        paths,
        CommandExecParams {
            command: command.clone(),
            process_id: Some(session_id),
            tty: false,
            stream_stdin: false,
            stream_stdout_stderr: false,
            output_bytes_cap: Some(128 * 1024),
            disable_output_cap: false,
            disable_timeout: false,
            timeout_ms: Some(timeout_ms),
            cwd: Some(paths.root.clone()),
            env: detect_desktop_session_env(),
            size: None,
            sandbox_policy: None,
        },
    )?;

    let snapshot = result.snapshot;
    let mut status = if snapshot.exit_code == Some(0) {
        "ok".to_string()
    } else {
        "nonzero_exit".to_string()
    };
    let mut stdout = trim_output(&snapshot.stdout, 16_000);
    let mut stderr = trim_output(&snapshot.stderr, 16_000);
    let mut final_artifact_path = artifact_path.clone();
    if status == "ok" && action == "inspect_visual" {
        let screenshot_path = artifact_path
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("inspect_visual screenshot path missing"))?;
        match inspect_screenshot_with_local_vision(paths, directive, screenshot_path) {
            Ok(report) => {
                stdout = format!(
                    "{}\n\nVision summary:\n{}\nScreenshot: {}\nReport: {}",
                    stdout, report.summary, report.screenshot_path, report.report_path
                )
                .trim()
                .to_string();
                final_artifact_path = Some(report.report_path);
            }
            Err(err) => {
                status = "vision_error".to_string();
                stderr = format!("{}\n\nVision inspection error: {}", stderr, err)
                    .trim()
                    .to_string();
            }
        }
    }
    let refreshed_state = refresh_browser_engine_state(paths).unwrap_or(initial_state);
    let _ = record_agent_event(
        paths,
        "tool/browserAction",
        Some(task.id),
        &task.title,
        &format!("browser action {} finished with status {}", action, status),
        &serde_json::to_string(&serde_json::json!({
            "action": action,
            "command": command,
            "url": directive.url,
            "artifactPath": final_artifact_path,
            "browserStatus": refreshed_state.status,
            "exitCode": snapshot.exit_code,
            "justification": directive.justification,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );

    Ok(BrowserActionResult {
        status,
        stdout,
        stderr,
        artifact_path: final_artifact_path,
        browser_status: refreshed_state.status,
    })
}

fn require_url(directive: &BrowserActionDirective) -> anyhow::Result<String> {
    let url = directive
        .url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .context("browser action requires browserUrl")?;
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        anyhow::bail!("browserUrl must start with http:// or https://");
    }
    Ok(url.to_string())
}

fn require_chrome_binary(state: &BrowserEngineState) -> anyhow::Result<String> {
    state
        .chrome_binary
        .clone()
        .context("browser engine is missing a Chrome binary; run install_browser_engine first")
}

fn install_command(paths: &Paths) -> Vec<String> {
    vec![
        "/bin/sh".to_string(),
        paths
            .root
            .join("scripts/install_browser_engine.sh")
            .display()
            .to_string(),
    ]
}

fn launch_command(paths: &Paths) -> Vec<String> {
    vec![
        "/bin/sh".to_string(),
        paths
            .root
            .join("scripts/launch_browser_agent_chrome.sh")
            .display()
            .to_string(),
    ]
}

fn headless_command(binary: String, wait_ms: Option<u64>, mut args: Vec<String>) -> Vec<String> {
    let mut command = vec![
        binary,
        "--headless=new".to_string(),
        "--disable-gpu".to_string(),
        "--no-first-run".to_string(),
        "--no-default-browser-check".to_string(),
    ];
    if let Some(wait_ms) = wait_ms.filter(|value| *value >= 1) {
        command.push(format!("--virtual-time-budget={wait_ms}"));
    }
    command.append(&mut args);
    command
}

fn interactive_open_command(state: &BrowserEngineState, url: &str) -> anyhow::Result<Vec<String>> {
    #[cfg(target_os = "macos")]
    {
        let _ = state;
        return Ok(vec![
            "open".to_string(),
            "-a".to_string(),
            "Google Chrome".to_string(),
            url.to_string(),
        ]);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let binary = require_chrome_binary(state)?;
        Ok(vec![binary, "--new-window".to_string(), url.to_string()])
    }
}

fn resolve_artifact_path(
    paths: &Paths,
    requested: Option<&str>,
    prefix: &str,
    extension: &str,
) -> anyhow::Result<PathBuf> {
    let candidate = match requested.map(str::trim).filter(|value| !value.is_empty()) {
        Some(raw) => {
            let path = PathBuf::from(raw);
            if path.is_absolute() {
                path
            } else {
                paths.root.join(path)
            }
        }
        None => paths.browser_artifacts_dir.join(format!(
            "{}-{}.{}",
            prefix,
            chrono::Utc::now().timestamp_millis(),
            extension
        )),
    };
    if let Some(parent) = candidate.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(candidate)
}

struct VisualInspectionReport {
    screenshot_path: String,
    report_path: String,
    summary: String,
}

fn inspect_screenshot_with_local_vision(
    paths: &Paths,
    directive: &BrowserActionDirective,
    screenshot_path: &str,
) -> anyhow::Result<VisualInspectionReport> {
    let policy = load_model_policy(paths);
    let census = load_census(paths);
    let selected = recommended_browser_vision_kleinhirn(&policy, &census).ok_or_else(|| {
        anyhow::anyhow!("no supported browser-vision kleinhirn candidate is available")
    })?;
    let selected_runtime = selected
        .runtime_model_id
        .clone()
        .unwrap_or_else(|| selected.model_id.clone());
    let current = load_kleinhirn_runtime_snapshot(paths).unwrap_or_default();
    if current.runtime_model.as_deref() != Some(selected_runtime.as_str())
        || current.policy_model.as_deref() != Some(selected.model_id.as_str())
    {
        anyhow::bail!(
            "Browser vision requires the local Qwen3.5 runtime first; request `brainAction=upgrade_local_browser_vision_kleinhirn` before `inspect_visual` (active runtime: {}, required runtime: {}).",
            current.runtime_model.as_deref().unwrap_or("unknown"),
            selected_runtime
        );
    }

    let runtime_env = load_runtime_env_map(paths)?;
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
    let screenshot_bytes = fs::read(screenshot_path)
        .with_context(|| format!("failed to read screenshot {}", screenshot_path))?;
    let screenshot_data_url = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(screenshot_bytes)
    );
    let question = directive
        .question
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("Describe the visible UI state, name blockers, and assess whether the task is already complete.");
    let prompt = [
        "You are a visual browser inspector for the CTO-Agent.",
        "Analyze the visible screenshot of the current page.",
        "Respond strictly as JSON with the fields `answer`, `completion_assessment`, `observations`, `blockers`, `qa_checks`, and `next_actions`.",
        "completion_assessment.status darf nur not_started, in_progress, blocked oder complete sein.",
        "Kein Markdown. Kein Fliesstext ausserhalb des JSON.",
        &format!("Frage: {}", question.trim()),
        &format!(
            "URL: {}",
            directive.url.as_deref().unwrap_or("about:blank").trim()
        ),
    ]
    .join("\n");
    let request_body = serde_json::json!({
        "model": selected_runtime,
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": screenshot_data_url,
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "stream": false
    });
    let response = Client::new()
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(&request_body)
        .send()
        .context("failed to call local browser-vision runtime")?;
    let response = response
        .error_for_status()
        .context("local browser-vision runtime returned error status")?;
    let payload = response
        .json::<Value>()
        .context("failed to decode browser-vision response as json")?;
    let response_text = extract_chat_response_text(&payload)
        .ok_or_else(|| anyhow::anyhow!("browser-vision runtime returned no textual content"))?;
    let report_path = PathBuf::from(screenshot_path).with_extension("json");
    let report_json = serde_json::json!({
        "modelId": selected.model_id,
        "runtimeModelId": selected_runtime,
        "url": directive.url.as_deref().unwrap_or(""),
        "question": question,
        "screenshotPath": screenshot_path,
        "responseText": response_text,
        "rawResponse": payload,
        "createdAt": now_iso(),
    });
    fs::write(
        &report_path,
        serde_json::to_string_pretty(&report_json)
            .unwrap_or_else(|_| "{\"error\":\"serialize_failed\"}\n".to_string()),
    )
    .with_context(|| format!("failed to write {}", report_path.display()))?;
    Ok(VisualInspectionReport {
        screenshot_path: screenshot_path.to_string(),
        report_path: report_path.display().to_string(),
        summary: response_text,
    })
}

fn extract_chat_response_text(payload: &Value) -> Option<String> {
    let content = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))?;
    if let Some(text) = content.as_str() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    let array = content.as_array()?;
    let joined = array
        .iter()
        .filter_map(|item| {
            item.get("text")
                .and_then(Value::as_str)
                .or_else(|| item.get("content").and_then(Value::as_str))
        })
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

fn load_runtime_env_map(
    paths: &Paths,
) -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    let env_path = paths.root.join("runtime/kleinhirn.env");
    let text = fs::read_to_string(&env_path)
        .with_context(|| format!("failed to read {}", env_path.display()))?;
    let mut values = std::collections::BTreeMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = trimmed.split_once('=') else {
            continue;
        };
        values.insert(
            key.trim().to_string(),
            raw_value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string(),
        );
    }
    Ok(values)
}

fn find_chrome_binary() -> Option<PathBuf> {
    let candidates = vec![
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ];
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }

    for name in [
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
    ] {
        if let Some(found) = which(name) {
            return Some(found);
        }
    }
    None
}

fn which(name: &str) -> Option<PathBuf> {
    let output = Command::new("which").arg(name).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

fn detect_chrome_version(path: &Path) -> Option<String> {
    let output = Command::new(path).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn trim_output(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        value.to_string()
    } else {
        value.chars().take(max_chars).collect::<String>() + "..."
    }
}

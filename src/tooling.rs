use crate::command_exec::run_one_shot_command;
use crate::contracts::HomepagePolicy;
use crate::contracts::Paths;
use crate::contracts::load_homepage_policy;
use crate::contracts::now_iso;
use crate::contracts::save_homepage_policy;
use crate::desktop_session::detect_desktop_session_env;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::record_agent_event;
use crate::runtime_db::record_homepage_revision;
use anyhow::Context;
use codex_app_server_protocol::CommandExecParams;
use codex_shell_command::command_safety::is_dangerous_command::command_might_be_dangerous;
use codex_shell_command::command_safety::is_safe_command::is_known_safe_command;
use codex_shell_command::parse_command::parse_command;
use serde::Deserialize;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecCommandDirective {
    pub command: Vec<String>,
    pub workdir: Option<String>,
    pub timeout_ms: Option<u64>,
    pub justification: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExecCommandResult {
    pub status: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
    pub cwd: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomepageUpdateDirective {
    pub title: Option<String>,
    pub headline: Option<String>,
    pub intro: Option<String>,
    pub communication_note: Option<String>,
    pub terminal_fallback_note: Option<String>,
}

fn command_uses_reviewed_mail_adapter(command: &[String]) -> bool {
    if command.is_empty() {
        return false;
    }
    let joined = command.join(" ").to_lowercase();
    joined.contains("communication_mail_cli.mjs")
}

fn command_looks_like_raw_mail_transport(command: &[String]) -> bool {
    if command.is_empty() {
        return false;
    }
    let joined = command.join(" ").to_lowercase();
    (joined.contains("smtplib")
        || joined.contains("send.one.com")
        || joined.contains("imap.one.com")
        || joined.contains("smtp.")
        || joined.contains("imap.")
        || joined.contains("mail_sent"))
        && !command_uses_reviewed_mail_adapter(command)
}

fn outbound_mail_must_use_reviewed_adapter(task: &TaskRecord, command: &[String]) -> bool {
    matches!(
        task.task_kind.as_str(),
        "owner_interrupt" | "communication_governance" | "proactive_contact_dispatch"
    ) && command_looks_like_raw_mail_transport(command)
}

pub fn run_bounded_command(
    paths: &Paths,
    task: &TaskRecord,
    directive: &ExecCommandDirective,
) -> anyhow::Result<ExecCommandResult> {
    if directive.command.is_empty() {
        anyhow::bail!("execCommand must not be empty");
    }
    if outbound_mail_must_use_reviewed_adapter(task, &directive.command) {
        anyhow::bail!(
            "Outbound mail must use the reviewed JS adapter at scripts/communication_mail_cli.mjs instead of a raw SMTP shell command."
        );
    }

    let timeout_ms = directive.timeout_ms.unwrap_or(15_000).clamp(500, 120_000);
    let cwd = resolve_exec_cwd(paths, directive.workdir.as_deref());
    let cwd_display = cwd.display().to_string();
    let parsed_command = parse_command(&directive.command);
    let known_safe = is_known_safe_command(&directive.command);
    let might_be_dangerous = command_might_be_dangerous(&directive.command);
    let result = run_bounded_command_via_codex(paths, task, directive, &cwd, timeout_ms)
        .with_context(|| {
            format!(
                "failed to execute codex-backed bounded command {:?} in {}",
                directive.command, cwd_display
            )
        })?;

    let _ = record_agent_event(
        paths,
        "tool/commandExec",
        Some(task.id),
        &task.title,
        &format!("bounded exec status={} for task {}", result.status, task.id),
        &serde_json::to_string(&serde_json::json!({
            "command": directive.command,
            "parsedCommand": parsed_command,
            "knownSafeCommand": known_safe,
            "mightBeDangerousCommand": might_be_dangerous,
            "cwd": result.cwd,
            "timeoutMs": timeout_ms,
            "justification": directive.justification,
            "exitCode": result.exit_code,
            "timedOut": result.timed_out,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );

    Ok(result)
}

fn run_bounded_command_via_codex(
    paths: &Paths,
    task: &TaskRecord,
    directive: &ExecCommandDirective,
    cwd: &PathBuf,
    timeout_ms: u64,
) -> anyhow::Result<ExecCommandResult> {
    let session_id = format!(
        "task-{}-bounded-exec-{}",
        task.id,
        now_iso().replace(':', "-")
    );
    let result = run_one_shot_command(
        paths,
        CommandExecParams {
            command: directive.command.clone(),
            process_id: Some(session_id),
            tty: false,
            stream_stdin: false,
            stream_stdout_stderr: false,
            output_bytes_cap: Some(64 * 1024),
            disable_output_cap: false,
            disable_timeout: false,
            timeout_ms: Some(timeout_ms as i64),
            cwd: Some(cwd.clone()),
            env: detect_desktop_session_env(),
            size: None,
            sandbox_policy: None,
        },
    )?;
    let snapshot = result.snapshot;
    let exit_code = snapshot.exit_code.filter(|code| *code >= 0);
    Ok(ExecCommandResult {
        status: if snapshot.status == "timeout" {
            "timeout".to_string()
        } else if exit_code == Some(0) {
            "ok".to_string()
        } else {
            "nonzero_exit".to_string()
        },
        exit_code,
        timed_out: snapshot.status == "timeout",
        stdout: trim_output(&snapshot.stdout, 12_000),
        stderr: trim_output(&snapshot.stderr, 12_000),
        cwd: snapshot.cwd,
    })
}

pub fn apply_homepage_update(
    paths: &Paths,
    task: &TaskRecord,
    directive: &HomepageUpdateDirective,
    source: &str,
) -> anyhow::Result<HomepagePolicy> {
    let mut policy = load_homepage_policy(paths);
    let mut changed = false;

    if let Some(value) = non_empty(directive.title.as_deref()) {
        policy.current_title = value.to_string();
        changed = true;
    }
    if let Some(value) = non_empty(directive.headline.as_deref()) {
        policy.current_headline = value.to_string();
        changed = true;
    }
    if let Some(value) = non_empty(directive.intro.as_deref()) {
        policy.current_intro = value.to_string();
        changed = true;
    }
    if let Some(value) = non_empty(directive.communication_note.as_deref()) {
        policy.communication_note = value.to_string();
        changed = true;
    }
    if let Some(value) = non_empty(directive.terminal_fallback_note.as_deref()) {
        policy.terminal_fallback_note = value.to_string();
        changed = true;
    }

    policy.homepage_ready = true;
    policy.bios_visible = true;
    policy.terminal_fallback_enabled = true;
    policy.updated_at = now_iso();

    if changed {
        save_homepage_policy(paths, &policy)?;
        record_homepage_revision(
            paths,
            source,
            &policy,
            &format!(
                "homepage mutated directly by bounded agent run for task {}",
                task.id
            ),
        )?;
        let _ = record_agent_event(
            paths,
            "tool/homepageUpdate",
            Some(task.id),
            &task.title,
            "homepage policy mutated by bounded agent tool action",
            &serde_json::to_string(&serde_json::json!({
                "title": policy.current_title,
                "headline": policy.current_headline,
                "source": source,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );
    }

    Ok(policy)
}

fn resolve_exec_cwd(paths: &Paths, requested: Option<&str>) -> PathBuf {
    match requested.map(str::trim).filter(|value| !value.is_empty()) {
        Some(raw) => {
            let candidate = PathBuf::from(raw);
            if candidate.is_absolute() {
                candidate
            } else {
                paths.root.join(candidate)
            }
        }
        None => paths.root.clone(),
    }
}

fn trim_output(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        value.to_string()
    } else {
        value.chars().take(max_chars).collect::<String>() + "..."
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_smtp_mail_command_is_detected() {
        let command = vec![
            "python3".to_string(),
            "-c".to_string(),
            "import smtplib; print('MAIL_SENT')".to_string(),
        ];
        assert!(command_looks_like_raw_mail_transport(&command));
        assert!(!command_uses_reviewed_mail_adapter(&command));
    }

    #[test]
    fn reviewed_js_mail_adapter_is_not_treated_as_raw_smtp() {
        let command = vec![
            "node".to_string(),
            "scripts/communication_mail_cli.mjs".to_string(),
            "send".to_string(),
            "--to".to_string(),
            "michael.welsch@metric-space.ai".to_string(),
        ];
        assert!(command_uses_reviewed_mail_adapter(&command));
        assert!(!command_looks_like_raw_mail_transport(&command));
    }

    #[test]
    fn owner_interrupt_mail_task_requires_reviewed_mail_adapter() {
        let task = TaskRecord {
            id: 75,
            created_at: now_iso(),
            updated_at: now_iso(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: Some(1),
            source_channel: "bios".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: "Schreibe eine Test-E-Mail".to_string(),
            detail: "Sende genau eine Mail sichtbar an den Owner.".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "active".to_string(),
            run_count: 1,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        };
        let command = vec![
            "python3".to_string(),
            "-c".to_string(),
            "import smtplib; print('MAIL_SENT')".to_string(),
        ];
        assert!(outbound_mail_must_use_reviewed_adapter(&task, &command));
    }
}

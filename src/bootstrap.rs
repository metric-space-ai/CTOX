use crate::browser_agent_bridge::browser_agent_extension_manifest_path;
use crate::browser_agent_bridge::browser_agent_extension_workspace;
use crate::browser_agent_bridge::load_browser_agent_bridge_state;
use crate::brain_runtime::load_runtime_env_map;
use crate::brain_runtime::save_runtime_env_map;
use crate::browser_engine::BrowserActionDirective;
use crate::browser_engine::browser_status_text;
use crate::browser_engine::run_browser_action;
use crate::browser_engine::start_browser_install_session;
use crate::browser_engine::start_browser_launch_session;
use crate::command_exec::list_sessions;
use crate::command_exec::read_session;
use crate::command_exec::resize_session;
use crate::command_exec::start_session;
use crate::command_exec::terminate_session;
use crate::command_exec::write_session;
use crate::contracts::InstallationBootstrapState;
use crate::contracts::Paths;
use crate::contracts::append_boot_entry;
use crate::contracts::default_homepage_policy;
use crate::contracts::load_bios;
use crate::contracts::load_homepage_policy;
use crate::contracts::load_installation_bootstrap_state;
use crate::contracts::load_model_policy;
use crate::contracts::load_organigram;
use crate::contracts::now_iso;
use crate::contracts::recommended_kleinhirn;
use crate::contracts::save_homepage_policy;
use crate::contracts::save_installation_bootstrap_state;
use crate::contracts::save_organigram;
use crate::context_controller::ContextCompactionTrigger;
use crate::context_controller::prepare_context_package_with_trigger;
use crate::desktop_session::detect_desktop_session_env;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::enqueue_internal_task;
use crate::runtime_db::enqueue_loop_interrupt;
use crate::runtime_db::ingest_pending_loop_interrupts;
use crate::runtime_db::list_open_tasks;
use crate::runtime_db::list_recent_agent_events;
use crate::runtime_db::list_recent_agent_turns;
use crate::runtime_db::list_recent_loop_incidents;
use crate::runtime_db::list_recent_turn_signals;
use crate::runtime_db::list_worker_jobs;
use crate::runtime_db::load_agent_thread;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::load_task_by_id;
use crate::runtime_db::record_homepage_revision;
use crate::runtime_db::record_owner_chat_contact;
use crate::runtime_db::record_memory;
use crate::runtime_db::record_terminal_feedback;
use crate::runtime_db::record_turn_signal_for_active_turn;
use crate::runtime_db::queue_loop_interrupt_as_task;
use crate::supervisor::inspect_local_resources;
use anyhow::Context;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use codex_app_server_protocol::CommandExecParams;
use codex_app_server_protocol::CommandExecResizeParams;
use codex_app_server_protocol::CommandExecTerminalSize;
use codex_app_server_protocol::CommandExecTerminateParams;
use codex_app_server_protocol::CommandExecWriteParams;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::IsTerminal;
use std::io::Write;
use std::path::PathBuf;

pub fn spawn_terminal_bridge(paths: Paths) {
    tokio::task::spawn_blocking(move || {
        eprintln!(
            "CTO-Agent terminal bridge active. Use 'Speaker: Nachricht', '/status' or '/help'."
        );
        let stdin = io::stdin();
        let mut lock = stdin.lock();
        let mut line = String::new();

        loop {
            line.clear();
            match lock.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match handle_input_line(&paths, trimmed, "terminal") {
                        Ok(response) if !response.output.is_empty() => {
                            println!("{}", response.output);
                        }
                        Ok(_) => {}
                        Err(err) => {
                            eprintln!("terminal bridge error: {err}");
                        }
                    }
                }
                Err(err) => {
                    eprintln!("terminal bridge read error: {err}");
                    break;
                }
            }
        }
    });
}

pub fn handle_terminal_line(paths: &Paths, line: &str) -> anyhow::Result<String> {
    handle_input_line(paths, line, "terminal").map(|outcome| outcome.output)
}

pub fn handle_attach_line(paths: &Paths, line: &str) -> anyhow::Result<String> {
    handle_attach_line_detailed(paths, line).map(|outcome| outcome.output)
}

pub fn handle_attach_line_detailed(paths: &Paths, line: &str) -> anyhow::Result<AttachLineOutcome> {
    let message = line.trim();
    if message.is_empty() {
        return Ok(AttachLineOutcome::text(String::new()));
    }

    let speaker = default_attach_chat_speaker(paths);
    let interrupt_id = enqueue_loop_interrupt(paths, "attach_terminal", &speaker, message)?;
    queue_interrupt(paths, interrupt_id, "attach_terminal", &speaker, message)
}

pub struct AttachLineOutcome {
    pub output: String,
    pub queued_task_id: Option<i64>,
    pub queued_task_title: Option<String>,
}

impl AttachLineOutcome {
    fn text(output: String) -> Self {
        Self {
            output,
            queued_task_id: None,
            queued_task_title: None,
        }
    }
}

pub fn run_install_bootstrap_tui(paths: &Paths) -> anyhow::Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        anyhow::bail!("install bootstrap TUI requires an interactive terminal");
    }

    let mut state = load_installation_bootstrap_state(paths);
    print_install_bootstrap_intro(&state);

    let owner_name = prompt_line("Owner Name", state.owner_name.trim())?;
    let owner_contact_email = prompt_line("Owner Contact Email", state.owner_contact_email.trim())?;
    let owner_contact_info = prompt_multiline(
        "Optional: additional owner contact details or notes.\nMultiple lines are allowed. An empty line ends input.",
        &state.owner_contact_info,
    )?;
    let email_mode = prompt_email_assignment_mode(&state.email_assignment_mode)?;
    let email_prompt = match email_mode.as_str() {
        "assigned_now" => {
            "Enter all free-form details for the assigned email mailbox now.\nExamples: address, provider, IMAP/SMTP hosts, access, rules, contacts.\nMultiple lines are allowed. An empty line ends input."
        }
        "self_procure" => {
            "Describe how the CTO-Agent should procure an email mailbox itself.\nExamples: preferred provider, domain, constraints, budget, approvals, contacts.\nMultiple lines are allowed. An empty line ends input."
        }
        _ => {
            "Optional: enter free-form notes about later email setup or communication preferences.\nMultiple lines are allowed. An empty line ends input."
        }
    };
    let email_note = prompt_multiline(email_prompt, &state.email_bootstrap_note)?;
    let installer_free_text = prompt_multiline(
        "Optional: additional startup notes for early communication, the dashboard, or first contacts.\nMultiple lines are allowed. An empty line ends input.",
        &state.installer_free_text,
    )?;

    state.status = "captured".to_string();
    state.owner_name = owner_name;
    state.owner_contact_email = owner_contact_email;
    state.owner_contact_info = owner_contact_info;
    state.email_assignment_mode = email_mode;
    state.email_bootstrap_note = email_note;
    state.installer_free_text = installer_free_text;
    state.updated_at = now_iso();
    save_installation_bootstrap_state(paths, &state)?;

    apply_installation_bootstrap_homepage_defaults(paths, &state)?;
    apply_installation_bootstrap_owner_defaults(paths, &state)?;

    let detail = installation_bootstrap_detail(&state);
    append_boot_entry(
        paths,
        "installer",
        &format!(
            "Communication bootstrap captured. Terminal command: {}. Email mode: {}. Further details are stored in the installation bootstrap contract.",
            state.terminal_command,
            installation_email_mode_label(&state.email_assignment_mode),
        ),
    )?;
    record_memory(
        paths,
        "installation_bootstrap",
        &format!(
            "Installation communication briefing captured ({})",
            installation_email_mode_label(&state.email_assignment_mode)
        ),
        &detail,
        "install_bootstrap_tui",
    )?;
    let _ = enqueue_internal_task(
        paths,
        None,
        "installation_bootstrap",
        "Evaluate installation communication briefing",
        &detail,
        installation_bootstrap_priority(&state.email_assignment_mode),
    );

    println!();
    println!("Communication bootstrap saved.");
    if !state.owner_name.trim().is_empty() || !state.owner_contact_email.trim().is_empty() {
        println!(
            "Owner: {} {}",
            if state.owner_name.trim().is_empty() {
                "(without name)".to_string()
            } else {
                state.owner_name.trim().to_string()
            },
            if state.owner_contact_email.trim().is_empty() {
                String::new()
            } else {
                format!("<{}>", state.owner_contact_email.trim())
            }
        );
    }
    println!(
        "Terminal low-level: {} -> starts direct CTO communication.",
        state.terminal_command
    );
    println!("Comfort path: local dashboard or intranet page under the CTO control plane.");
    println!(
        "Email mode: {}",
        installation_email_mode_label(&state.email_assignment_mode)
    );
    if !state.email_bootstrap_note.trim().is_empty() {
        println!("Email notes were saved.");
    }
    if !state.installer_free_text.trim().is_empty() {
        println!("Additional startup notes were saved.");
    }
    println!("Later additions remain possible through the terminal and dashboard.");
    Ok(())
}

fn print_install_bootstrap_intro(state: &InstallationBootstrapState) {
    println!();
    println!("CTO-Agent Installation Bootstrap");
    println!("--------------------------------");
    println!(
        "Low-level communication always remains possible through the terminal command `{}`.",
        state.terminal_command
    );
    println!(
        "In parallel, the CTO-Agent will usually provide a local intranet or dashboard page for more comfortable communication."
    );
    println!(
        "If email becomes useful later, the owner can already drop free-form notes now or add them later in the terminal or dashboard."
    );
    println!();
}

fn prompt_email_assignment_mode(current: &str) -> anyhow::Result<String> {
    loop {
        println!("How should the CTO-Agent start with email?");
        println!("  1) A mailbox is assigned now");
        println!("  2) It should procure a mailbox later by itself");
        println!("  3) Decide later / start without email for now");
        let default_choice = match current {
            "assigned_now" => "1",
            "self_procure" => "2",
            _ => "3",
        };
        let raw = prompt_line(
            &format!("Choice [default {}]", default_choice),
            default_choice,
        )?;
        let normalized = raw.trim().to_lowercase();
        let mode = match normalized.as_str() {
            "1" | "assigned" | "assigned_now" | "zuweisen" => Some("assigned_now"),
            "2" | "self" | "self_procure" | "self-procure" | "selbst" => Some("self_procure"),
            "3" | "later" | "defer" | "decide_later" | "spaeter" => Some("decide_later"),
            _ => None,
        };
        if let Some(mode) = mode {
            return Ok(mode.to_string());
        }
        println!("Please enter 1, 2, or 3.");
    }
}

fn prompt_line(prompt: &str, default: &str) -> anyhow::Result<String> {
    print!("{prompt}: ");
    io::stdout().flush().context("failed to flush stdout")?;
    let mut buffer = String::new();
    io::stdin()
        .read_line(&mut buffer)
        .context("failed to read stdin line")?;
    let trimmed = buffer.trim_end_matches(['\r', '\n']);
    if trimmed.trim().is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.trim().to_string())
    }
}

fn prompt_multiline(prompt: &str, existing: &str) -> anyhow::Result<String> {
    println!();
    println!("{prompt}");
    if !existing.trim().is_empty() {
        println!("Current value:");
        println!("{}", existing.trim());
        println!("An empty first line keeps the current value.");
    }
    let mut lines = Vec::new();
    loop {
        let mut buffer = String::new();
        io::stdin()
            .read_line(&mut buffer)
            .context("failed to read stdin line")?;
        let trimmed = buffer.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            if lines.is_empty() && !existing.trim().is_empty() {
                return Ok(existing.trim().to_string());
            }
            break;
        }
        lines.push(trimmed.to_string());
    }
    Ok(lines.join("\n").trim().to_string())
}

fn installation_email_mode_label(mode: &str) -> &'static str {
    match mode {
        "assigned_now" => "Mailbox will be assigned",
        "self_procure" => "Agent should procure a mailbox itself",
        _ => "Decide later / without email for now",
    }
}

fn installation_bootstrap_priority(mode: &str) -> i64 {
    match mode {
        "assigned_now" => 335,
        "self_procure" => 320,
        _ => 250,
    }
}

fn installation_bootstrap_detail(state: &InstallationBootstrapState) -> String {
    format!(
        "Installation communication briefing. Interpret this as an early startup directive for communication and contact bootstrapping. Owner: {} <{}>. Additional owner contact details: {}. Terminal low-level command: `{}`. Dashboard/Intranet: {}. Email mode: {}. Email notes: {}. Additional installer notes: {}. If a mailbox was assigned directly, you should check whether you can build a suitable communication client with a skill and template, establish mail access, and send a test email after bounded verification. The owner may provide additional details later through the terminal or dashboard.",
        if state.owner_name.trim().is_empty() {
            "unknown".to_string()
        } else {
            state.owner_name.trim().to_string()
        },
        if state.owner_contact_email.trim().is_empty() {
            "none".to_string()
        } else {
            state.owner_contact_email.trim().to_string()
        },
        if state.owner_contact_info.trim().is_empty() {
            "none".to_string()
        } else {
            state.owner_contact_info.trim().to_string()
        },
        state.terminal_command,
        state.dashboard_note,
        installation_email_mode_label(&state.email_assignment_mode),
        if state.email_bootstrap_note.trim().is_empty() {
            "none".to_string()
        } else {
            state.email_bootstrap_note.trim().to_string()
        },
        if state.installer_free_text.trim().is_empty() {
            "none".to_string()
        } else {
            state.installer_free_text.trim().to_string()
        },
    )
}

fn apply_installation_bootstrap_owner_defaults(
    paths: &Paths,
    state: &InstallationBootstrapState,
) -> anyhow::Result<()> {
    if state.owner_name.trim().is_empty() && state.owner_contact_email.trim().is_empty() {
        return Ok(());
    }
    let mut organigram = load_organigram(paths);
    if !state.owner_name.trim().is_empty() {
        organigram.owner.name = state.owner_name.trim().to_string();
    }
    if !state.owner_contact_email.trim().is_empty() {
        organigram.owner.email = state.owner_contact_email.trim().to_string();
    }
    if organigram.owner.role.trim().is_empty() {
        organigram.owner.role = "owner".to_string();
    }
    organigram.updated_at = now_iso();
    save_organigram(paths, &organigram)
}

fn apply_installation_bootstrap_homepage_defaults(
    paths: &Paths,
    state: &InstallationBootstrapState,
) -> anyhow::Result<()> {
    let mut policy = load_homepage_policy(paths);
    let default_policy = default_homepage_policy();

    if policy.current_intro.trim().is_empty()
        || policy.current_intro == default_policy.current_intro
    {
        policy.current_intro = "Initial communication can always start through `cto` in the terminal. In parallel, this local control plane should grow into the more comfortable intranet surface.".to_string();
    }
    if policy.communication_note.trim().is_empty()
        || policy.communication_note == default_policy.communication_note
    {
        policy.communication_note = format!(
            "Low-level communication always runs through `{}`. The local dashboard or intranet page should become the more comfortable channel. Email startup mode: {}. Additional details can be added now or later through the terminal and dashboard.",
            state.terminal_command,
            installation_email_mode_label(&state.email_assignment_mode)
        );
    }
    if policy.terminal_fallback_note.trim().is_empty()
        || policy.terminal_fallback_note == default_policy.terminal_fallback_note
    {
        policy.terminal_fallback_note = format!(
            "If the homepage or email path is not ready yet, `{}` in the terminal remains the primary full-access layer.",
            state.terminal_command
        );
    }
    policy.updated_at = now_iso();
    save_homepage_policy(paths, &policy)
}

fn handle_input_line(
    paths: &Paths,
    line: &str,
    source_channel: &str,
) -> anyhow::Result<AttachLineOutcome> {
    let trimmed = line.trim();
    match trimmed {
        "/help" => return Ok(AttachLineOutcome::text(help_text())),
        "/status" => return terminal_status(paths).map(AttachLineOutcome::text),
        "/events" => return terminal_events(paths).map(AttachLineOutcome::text),
        "/turns" => return terminal_turns(paths).map(AttachLineOutcome::text),
        "/thread" => return terminal_thread(paths).map(AttachLineOutcome::text),
        "/signals" => return terminal_signals(paths).map(AttachLineOutcome::text),
        "/incidents" => return terminal_incidents(paths).map(AttachLineOutcome::text),
        _ => {}
    }
    if trimmed == "/exec-sessions" {
        return list_sessions(paths).map(AttachLineOutcome::text);
    }
    if let Some(rest) = trimmed.strip_prefix("/exec-start ") {
        return terminal_exec_start(paths, rest).map(AttachLineOutcome::text);
    }
    if let Some(rest) = trimmed.strip_prefix("/exec-write ") {
        return terminal_exec_write(paths, rest).map(AttachLineOutcome::text);
    }
    if let Some(rest) = trimmed.strip_prefix("/exec-resize ") {
        return terminal_exec_resize(paths, rest).map(AttachLineOutcome::text);
    }
    if let Some(rest) = trimmed.strip_prefix("/exec-read ") {
        return read_session(paths, rest.trim()).map(AttachLineOutcome::text);
    }
    if let Some(rest) = trimmed.strip_prefix("/exec-terminate ") {
        return terminate_session(
            paths,
            CommandExecTerminateParams {
                process_id: rest.trim().to_string(),
            },
        )
        .map(AttachLineOutcome::text);
    }
    if trimmed == "/browser-status" {
        return browser_status_text(paths).map(AttachLineOutcome::text);
    }
    if trimmed == "/browser-workers" {
        return terminal_browser_workers(paths).map(AttachLineOutcome::text);
    }
    if trimmed == "/browser-bridge" {
        return terminal_browser_bridge(paths).map(AttachLineOutcome::text);
    }
    if trimmed == "/browser-extension-path" {
        return terminal_browser_extension_path(paths).map(AttachLineOutcome::text);
    }
    if trimmed == "/browser-install" {
        return start_browser_install_session(paths).map(AttachLineOutcome::text);
    }
    if trimmed == "/browser-launch" {
        return start_browser_launch_session(paths).map(AttachLineOutcome::text);
    }
    if let Some(rest) = trimmed.strip_prefix("/browser-dom ") {
        return terminal_browser_dom(paths, rest).map(AttachLineOutcome::text);
    }
    if let Some(rest) = trimmed.strip_prefix("/browser-shot ") {
        return terminal_browser_screenshot(paths, rest).map(AttachLineOutcome::text);
    }
    if let Some(rest) = trimmed.strip_prefix("/browser-open ") {
        return terminal_browser_open(paths, rest).map(AttachLineOutcome::text);
    }

    let (speaker, raw_message) = parse_terminal_message(trimmed);
    let inline_mail_capture =
        maybe_capture_inline_mail_credentials(paths, source_channel, &speaker, &raw_message)?;
    let message = inline_mail_capture
        .as_ref()
        .map(|capture| capture.sanitized_interrupt_message.as_str())
        .unwrap_or(raw_message.as_str());
    let interrupt_id = enqueue_loop_interrupt(paths, source_channel, &speaker, message)?;
    let mut outcome = queue_interrupt(paths, interrupt_id, source_channel, &speaker, message)?;
    if let Some(capture) = inline_mail_capture {
        outcome.output = format!("{} {}", capture.runtime_note, outcome.output);
    }
    Ok(outcome)
}

pub fn maybe_apply_homepage_feedback(
    paths: &Paths,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> anyhow::Result<Option<String>> {
    if !looks_like_homepage_feedback(message) {
        return Ok(None);
    }

    let trust = load_owner_trust(paths).unwrap_or_default();
    let mut policy = load_homepage_policy(paths);
    let source_label = if source_channel == "bios" {
        "BIOS-Chat"
    } else {
        "Terminal"
    };

    policy.homepage_ready = true;
    policy.bios_visible = true;
    policy.terminal_primary = true;
    policy.terminal_fallback_enabled = true;
    policy.redesign_allowed_via_terminal = true;
    policy.redesign_allowed_via_bios_chat = true;
    policy.template_name = "homepage-bios-bridge-template".to_string();
    policy.stage = if trust.bios_primary_channel_confirmed {
        "bios_primary_revision".to_string()
    } else {
        "homepage_building".to_string()
    };

    if policy.current_title.trim().is_empty()
        || policy.current_title == "CTO-Agent Terminal Bridge"
        || policy.current_title == "CTO-Agent BIOS Bridge"
    {
        policy.current_title = if source_channel == "bios" {
            "CTO-Agent BIOS Bridge".to_string()
        } else {
            "CTO-Agent Terminal Bridge".to_string()
        };
    }

    policy.current_headline = build_homepage_headline(source_label, message);
    policy.current_intro = format!(
        "Latest design signal from {}: {}",
        speaker.trim(),
        summarize_feedback(message, 220)
    );
    policy.communication_note = format!(
        "This surface is currently being shaped through {} feedback. If it does not fit yet, the CTO-Agent may keep reshaping it through skill, template, and chat.",
        source_label
    );
    policy.terminal_fallback_note =
        "If the homepage is not carrying the interaction yet, the live terminal remains the primary full-access layer."
            .to_string();

    if !trust.bios_primary_channel_confirmed {
        policy.owner_branding_applied = false;
        policy.owner_branding_locked = false;
    }

    policy.updated_at = now_iso();
    save_homepage_policy(paths, &policy)?;
    record_homepage_revision(
        paths,
        source_channel,
        &policy,
        &format!(
            "homepage-bootstrap skill applied from {} feedback using homepage-bios-bridge-template.",
            source_label
        ),
    )?;

    Ok(Some(format!(
        "Homepage refreshed through `homepage-bootstrap` from {} feedback with `homepage-bios-bridge-template`.",
        source_label
    )))
}

fn parse_terminal_message(line: &str) -> (String, String) {
    if let Some((speaker, message)) = line.split_once(':') {
        let speaker = speaker.trim();
        let message = message.trim();
        if !speaker.is_empty()
            && !message.is_empty()
            && !looks_like_inline_config_prefix(speaker)
        {
            return (speaker.to_string(), message.to_string());
        }
    }

    ("Michael Welsch".to_string(), line.trim().to_string())
}

fn looks_like_inline_config_prefix(prefix: &str) -> bool {
    let lowered = prefix.trim().to_ascii_lowercase();
    matches!(
        lowered.as_str(),
        "email"
            | "email address"
            | "mail"
            | "mailbox"
            | "mail address"
            | "address"
            | "password"
            | "passwort"
            | "pw"
            | "imap"
            | "imap host"
            | "imap port"
            | "smtp"
            | "smtp host"
            | "smtp port"
            | "cto_email_address"
            | "cto_email_password"
            | "cto_email_imap_host"
            | "cto_email_imap_port"
            | "cto_email_smtp_host"
            | "cto_email_smtp_port"
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InlineMailCapture {
    sanitized_interrupt_message: String,
    runtime_note: String,
}

fn maybe_capture_inline_mail_credentials(
    paths: &Paths,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> anyhow::Result<Option<InlineMailCapture>> {
    if source_channel != "terminal" {
        return Ok(None);
    }
    if message.trim().is_empty() {
        return Ok(None);
    }

    let mut env_map = load_runtime_env_map(paths).unwrap_or_default();
    let Some(parsed) = parse_inline_mail_runtime_update(message, &env_map) else {
        return Ok(None);
    };

    upsert_runtime_env_value(&mut env_map, "CTO_EMAIL_ADDRESS", Some(&parsed.address));
    upsert_runtime_env_value(&mut env_map, "CTO_EMAIL_PASSWORD", Some(&parsed.password));
    upsert_runtime_env_value(
        &mut env_map,
        "CTO_EMAIL_IMAP_HOST",
        parsed.imap_host.as_deref(),
    );
    upsert_runtime_env_value(
        &mut env_map,
        "CTO_EMAIL_IMAP_PORT",
        parsed.imap_port.as_deref(),
    );
    upsert_runtime_env_value(
        &mut env_map,
        "CTO_EMAIL_SMTP_HOST",
        parsed.smtp_host.as_deref(),
    );
    upsert_runtime_env_value(
        &mut env_map,
        "CTO_EMAIL_SMTP_PORT",
        parsed.smtp_port.as_deref(),
    );
    save_runtime_env_map(paths, &env_map)?;

    let mut installation = load_installation_bootstrap_state(paths);
    installation.email_assignment_mode = "assigned_now".to_string();
    if installation.status.trim().is_empty() || installation.status == "unconfigured" {
        installation.status = "captured".to_string();
    }
    installation.updated_at = now_iso();
    save_installation_bootstrap_state(paths, &installation)?;

    let owner_label = if speaker.trim().is_empty() {
        "Owner".to_string()
    } else {
        speaker.trim().to_string()
    };
    Ok(Some(InlineMailCapture {
        sanitized_interrupt_message: format!(
            "{} supplied mail runtime credentials for {} via chat. Password was redacted and the runtime env was updated. Verify the IMAP/SMTP path now and confirm bidirectional mail readiness.",
            owner_label, parsed.address
        ),
        runtime_note: format!(
            "Mail runtime configuration updated from chat input for {}.",
            parsed.address
        ),
    }))
}

fn default_attach_chat_speaker(paths: &Paths) -> String {
    let bios = load_bios(paths);
    if !bios.owner.name.trim().is_empty() {
        return bios.owner.name.trim().to_string();
    }

    let organigram = load_organigram(paths);
    if !organigram.owner.name.trim().is_empty() {
        return organigram.owner.name.trim().to_string();
    }

    let installation = load_installation_bootstrap_state(paths);
    if !installation.owner_name.trim().is_empty() {
        return installation.owner_name.trim().to_string();
    }

    "Michael Welsch".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedInlineMailRuntimeUpdate {
    address: String,
    password: String,
    imap_host: Option<String>,
    imap_port: Option<String>,
    smtp_host: Option<String>,
    smtp_port: Option<String>,
}

fn parse_inline_mail_runtime_update(
    message: &str,
    existing_env: &BTreeMap<String, String>,
) -> Option<ParsedInlineMailRuntimeUpdate> {
    let address = extract_inline_mail_value(
        message,
        &[
            "CTO_EMAIL_ADDRESS",
            "mail_address",
            "email address",
            "mailbox",
            "email",
            "mail",
            "address",
        ],
    )
    .or_else(|| extract_first_email_address(message))
    .or_else(|| non_empty_env_value(existing_env, "CTO_EMAIL_ADDRESS"))?;
    let password = extract_inline_mail_value(
        message,
        &["CTO_EMAIL_PASSWORD", "mail_password", "password", "passwort", "pw"],
    )?;
    let imap_host = extract_inline_mail_value(
        message,
        &["CTO_EMAIL_IMAP_HOST", "imap host", "imap_host", "imap-host", "imap"],
    )
    .or_else(|| non_empty_env_value(existing_env, "CTO_EMAIL_IMAP_HOST"));
    let imap_port = extract_inline_mail_value(
        message,
        &["CTO_EMAIL_IMAP_PORT", "imap port", "imap_port", "imap-port"],
    )
    .or_else(|| non_empty_env_value(existing_env, "CTO_EMAIL_IMAP_PORT"));
    let smtp_host = extract_inline_mail_value(
        message,
        &["CTO_EMAIL_SMTP_HOST", "smtp host", "smtp_host", "smtp-host", "smtp"],
    )
    .or_else(|| non_empty_env_value(existing_env, "CTO_EMAIL_SMTP_HOST"));
    let smtp_port = extract_inline_mail_value(
        message,
        &["CTO_EMAIL_SMTP_PORT", "smtp port", "smtp_port", "smtp-port"],
    )
    .or_else(|| non_empty_env_value(existing_env, "CTO_EMAIL_SMTP_PORT"));

    Some(ParsedInlineMailRuntimeUpdate {
        address,
        password,
        imap_host,
        imap_port,
        smtp_host,
        smtp_port,
    })
}

fn extract_inline_mail_value(message: &str, labels: &[&str]) -> Option<String> {
    let normalized = message.replace("\r\n", "\n").replace('\r', "\n");
    let lowered = normalized.to_ascii_lowercase();
    for label in labels {
        let label_lower = label.to_ascii_lowercase();
        for separator in [":", "="] {
            let needle = format!("{label_lower}{separator}");
            let Some(start) = lowered.find(&needle) else {
                continue;
            };
            let raw_value = &normalized[start + needle.len()..];
            let value = truncate_inline_mail_value(raw_value);
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn truncate_inline_mail_value(raw_value: &str) -> String {
    let mut value = raw_value
        .trim_start_matches(|ch: char| ch.is_whitespace())
        .trim_start_matches(':')
        .trim_start_matches('=')
        .trim_start_matches(|ch: char| ch.is_whitespace())
        .to_string();

    for boundary in [
        "\n",
        "\r",
        ",",
        ";",
        " cto_email_",
        " password:",
        " password=",
        " passwort:",
        " passwort=",
        " pw:",
        " pw=",
        " imap host:",
        " imap host=",
        " imap_port=",
        " imap port:",
        " imap port=",
        " imap:",
        " imap=",
        " smtp host:",
        " smtp host=",
        " smtp_port=",
        " smtp port:",
        " smtp port=",
        " smtp:",
        " smtp=",
        " email address:",
        " email address=",
        " email:",
        " email=",
        " mailbox:",
        " mailbox=",
        " mail:",
        " mail=",
        " address:",
        " address=",
    ] {
        let lower_value = value.to_ascii_lowercase();
        if let Some(index) = lower_value.find(boundary) {
            value.truncate(index);
        }
    }

    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string()
}

fn extract_first_email_address(message: &str) -> Option<String> {
    for token in message.split(|ch: char| {
        ch.is_whitespace()
            || matches!(ch, ',' | ';' | '<' | '>' | '(' | ')' | '[' | ']' | '"' | '\'')
    }) {
        let candidate = token
            .trim()
            .trim_matches('.')
            .trim_matches(':')
            .trim_matches('=')
            .trim_matches('/')
            .trim_matches('\\');
        if candidate.contains('@') && candidate.split('@').count() == 2 {
            let mut parts = candidate.split('@');
            let local = parts.next().unwrap_or_default();
            let domain = parts.next().unwrap_or_default();
            if !local.is_empty() && domain.contains('.') {
                return Some(candidate.to_ascii_lowercase());
            }
        }
    }
    None
}

fn non_empty_env_value(env_map: &BTreeMap<String, String>, key: &str) -> Option<String> {
    env_map
        .get(key)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn upsert_runtime_env_value(
    env_map: &mut BTreeMap<String, String>,
    key: &str,
    value: Option<&str>,
) {
    let value = value.unwrap_or("").trim();
    if value.is_empty() {
        env_map.remove(key);
    } else {
        env_map.insert(key.to_string(), value.to_string());
    }
}

fn looks_like_homepage_feedback(message: &str) -> bool {
    let lowered = normalize_text(message);
    let keywords = [
        "homepage",
        "startseite",
        "webseite",
        "website",
        "bios",
        "kommunikationsweg",
        "kommunikationspfad",
        "kommunikation",
        "oberflaeche",
        "oberfläche",
        "branding",
        "sichtbar",
        "sichtbarer",
        "sichtbarkeit",
        "terminal",
        "chat",
        "bridge",
        "umbauen",
        "umgestalten",
        "aendern",
        "ändern",
        "klarer",
        "einfacher",
        "staerker",
        "stärker",
        "vertrauen",
        "trust",
    ];
    keywords.iter().any(|keyword| lowered.contains(keyword))
}

fn build_homepage_headline(source_label: &str, message: &str) -> String {
    let lowered = normalize_text(message);
    if lowered.contains("sichtbar") || lowered.contains("bios") {
        return format!(
            "This homepage will be rebuilt after {} so that BIOS stays visible and the communication direction becomes clear.",
            source_label
        );
    }
    if lowered.contains("vertrauen") || lowered.contains("trust") {
        return format!(
            "This homepage will be recalibrated after {} for trust, clarity, and root of trust.",
            source_label
        );
    }
    if lowered.contains("nicht")
        || lowered.contains("falsch")
        || lowered.contains("umbauen")
        || lowered.contains("umgestalten")
    {
        return format!(
            "This homepage will be rebuilt after {} and remain intentionally changeable.",
            source_label
        );
    }

    format!(
        "This homepage reacts to {} and remains a changeable first communication path.",
        source_label
    )
}

fn summarize_feedback(message: &str, limit: usize) -> String {
    let trimmed = message.trim();
    let mut out = String::new();
    for ch in trimmed.chars().take(limit) {
        out.push(ch);
    }
    if trimmed.chars().count() > limit {
        out.push_str("...");
    }
    out
}

fn normalize_text(value: &str) -> String {
    value.to_lowercase()
}

fn help_text() -> String {
    [
        "CTO-Agent infinity loop terminal",
        "  /status               shows bootstrap, BIOS, and homepage status",
        "  /thread               shows the persisted main-thread state",
        "  /signals              shows the latest turn steer/interrupt signals",
        "  /incidents            shows the latest loop incidents and recovery cases",
        "  /events               shows the latest agent events",
        "  /turns                shows the latest bounded agent turns",
        "  /exec-sessions        shows codex-backed exec sessions",
        "  /exec-start ...       starts a codex-backed exec session",
        "  /exec-write ...       writes to an exec session's stdin",
        "  /exec-resize ...      resizes a PTY exec session",
        "  /exec-read ...        shows buffered stdout/stderr from an exec session",
        "  /exec-terminate ...   terminates an exec session",
        "  /browser-status       shows status of the explicit browser engine",
        "  /browser-workers      shows browser/repair/specialist worker jobs",
        "  /browser-bridge       shows status of the Chrome extension bridge",
        "  /browser-extension-path  shows the decoupled extension workspace",
        "  /browser-install      starts the Chrome/browser installer through the CLI engine",
        "  /browser-launch       starts Chrome with the browser-agent extension on the desktop",
        "  /browser-dom URL      reads a page through headless Chrome",
        "  /browser-shot URL [FILE]  creates a screenshot through headless Chrome",
        "  /browser-open URL     opens a URL interactively in Chrome",
        "  /help                 shows this help",
        "  Michael Welsch: ...   is injected as an interrupt into the running always-on loop",
        "                        the current bounded step is not hard-aborted,",
        "                        but is finished cleanly and then reprioritized",
        "Example:",
        "  Michael Welsch: I do not like this communication path. Make the BIOS more visible and keep the terminal as fallback.",
        "  /exec-start --id shell --tty /bin/bash",
    ]
    .join("\n")
}

fn terminal_exec_start(paths: &Paths, rest: &str) -> anyhow::Result<String> {
    let tokens = shlex::split(rest).context("failed to parse /exec-start arguments")?;
    if tokens.is_empty() {
        anyhow::bail!(
            "usage: /exec-start [--id ID] [--cwd PATH] [--tty] [--stdin] [--timeout-ms N] [--rows N] [--cols N] command..."
        );
    }

    let mut session_id = None;
    let mut cwd: Option<PathBuf> = None;
    let mut tty = false;
    let mut stream_stdin = false;
    let mut timeout_ms = None;
    let mut rows = None;
    let mut cols = None;
    let mut index = 0_usize;
    let mut command = Vec::new();

    while index < tokens.len() {
        match tokens[index].as_str() {
            "--id" => {
                index += 1;
                session_id = tokens.get(index).cloned();
            }
            "--cwd" => {
                index += 1;
                cwd = tokens.get(index).map(PathBuf::from);
            }
            "--tty" => tty = true,
            "--stdin" => stream_stdin = true,
            "--timeout-ms" => {
                index += 1;
                timeout_ms = tokens
                    .get(index)
                    .and_then(|value| value.parse::<u64>().ok());
            }
            "--rows" => {
                index += 1;
                rows = tokens
                    .get(index)
                    .and_then(|value| value.parse::<u16>().ok());
            }
            "--cols" => {
                index += 1;
                cols = tokens
                    .get(index)
                    .and_then(|value| value.parse::<u16>().ok());
            }
            _ => {
                command.extend_from_slice(&tokens[index..]);
                break;
            }
        }
        index += 1;
    }

    if command.is_empty() {
        anyhow::bail!(
            "usage: /exec-start [--id ID] [--cwd PATH] [--tty] [--stdin] [--timeout-ms N] [--rows N] [--cols N] command..."
        );
    }

    let cwd = cwd
        .map(|value| {
            if value.is_absolute() {
                value
            } else {
                paths.root.join(value)
            }
        })
        .unwrap_or_else(|| paths.root.clone());

    let session_id =
        session_id.or_else(|| Some(format!("exec-{}", chrono::Utc::now().timestamp_millis())));

    start_session(
        paths,
        CommandExecParams {
            command,
            process_id: session_id,
            tty,
            stream_stdin,
            stream_stdout_stderr: true,
            output_bytes_cap: None,
            disable_output_cap: false,
            disable_timeout: false,
            timeout_ms: timeout_ms.map(|value| value as i64),
            cwd: Some(cwd),
            env: detect_desktop_session_env(),
            size: if tty {
                Some(CommandExecTerminalSize {
                    rows: rows.unwrap_or(24),
                    cols: cols.unwrap_or(80),
                })
            } else {
                None
            },
            sandbox_policy: None,
        },
    )
}

fn terminal_exec_write(paths: &Paths, rest: &str) -> anyhow::Result<String> {
    let mut parts = rest.splitn(2, ' ');
    let session_id = parts.next().unwrap_or_default().trim();
    let text = parts.next().unwrap_or_default();
    if session_id.is_empty() {
        anyhow::bail!("usage: /exec-write SESSION_ID text...");
    }
    write_session(
        paths,
        CommandExecWriteParams {
            process_id: session_id.to_string(),
            delta_base64: Some(STANDARD.encode(text.as_bytes())),
            close_stdin: false,
        },
    )
}

fn terminal_exec_resize(paths: &Paths, rest: &str) -> anyhow::Result<String> {
    let tokens = shlex::split(rest).context("failed to parse /exec-resize arguments")?;
    if tokens.len() != 3 {
        anyhow::bail!("usage: /exec-resize SESSION_ID ROWS COLS");
    }
    let rows = tokens[1]
        .parse::<u16>()
        .context("rows must be an integer")?;
    let cols = tokens[2]
        .parse::<u16>()
        .context("cols must be an integer")?;
    resize_session(
        paths,
        CommandExecResizeParams {
            process_id: tokens[0].clone(),
            size: CommandExecTerminalSize { rows, cols },
        },
    )
}

fn terminal_thread(paths: &Paths) -> anyhow::Result<String> {
    let thread = load_agent_thread(paths)?;
    Ok(format!(
        "thread={}; lifecycle={}; mode={}; active_turn={}; active_task={}; queue_depth={}; note={}",
        thread.thread_key,
        thread.lifecycle_status,
        thread.current_mode,
        thread
            .active_turn_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        thread
            .active_task_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        thread.queue_depth,
        thread.note
    ))
}

fn terminal_signals(paths: &Paths) -> anyhow::Result<String> {
    let signals = list_recent_turn_signals(paths, 12)?;
    if signals.is_empty() {
        return Ok("No turn signals available.".to_string());
    }
    Ok(signals
        .into_iter()
        .rev()
        .map(|signal| {
            format!(
                "[{}] {} :: turn={} task={} :: {} via {} :: {}",
                signal.created_at,
                signal.signal_kind,
                signal
                    .turn_id
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                signal
                    .task_id
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                signal.speaker,
                signal.source_channel,
                signal.message
            )
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn terminal_events(paths: &Paths) -> anyhow::Result<String> {
    let events = list_recent_agent_events(paths, 12)?;
    if events.is_empty() {
        return Ok("No agent events available.".to_string());
    }
    Ok(events
        .into_iter()
        .rev()
        .map(|event| {
            if let Some(task_id) = event.active_task_id {
                format!(
                    "[{}] {} :: #{} {} :: {}",
                    event.created_at, event.method, task_id, event.active_task_title, event.body
                )
            } else {
                format!("[{}] {} :: {}", event.created_at, event.method, event.body)
            }
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn terminal_incidents(paths: &Paths) -> anyhow::Result<String> {
    let incidents = list_recent_loop_incidents(paths, 12)?;
    if incidents.is_empty() {
        return Ok("No loop incidents available.".to_string());
    }
    Ok(incidents
        .into_iter()
        .rev()
        .map(|incident| {
            format!(
                "[{}] {} :: {} :: status={} :: hard_reset={} :: task={} :: {}",
                incident.created_at,
                incident.incident_key,
                incident.severity,
                incident.status,
                incident.hard_reset_required,
                incident
                    .related_task_id
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                incident.summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn terminal_turns(paths: &Paths) -> anyhow::Result<String> {
    let turns = list_recent_agent_turns(paths, 10)?;
    if turns.is_empty() {
        return Ok("No agent turns available.".to_string());
    }
    Ok(turns
        .into_iter()
        .rev()
        .map(|turn| {
            format!(
                "[{}] turn#{} :: task#{} {} :: {} -> {} :: status={} :: {}",
                turn.created_at,
                turn.id,
                turn.task_id,
                turn.task_title,
                turn.mode_at_start,
                turn.mode_at_end.unwrap_or_else(|| "open".to_string()),
                turn.status,
                turn.summary.unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn terminal_status(paths: &Paths) -> anyhow::Result<String> {
    let bios = load_bios(paths);
    let organigram = load_organigram(paths);
    let homepage = load_homepage_policy(paths);
    let trust = load_owner_trust(paths)?;
    let policy = load_model_policy(paths);
    let census = inspect_local_resources(paths).context("failed to inspect local resources")?;
    let browser = crate::browser_engine::inspect_browser_engine(paths);
    let kleinhirn = load_runtime_kleinhirn_display(paths).unwrap_or_else(|| {
        let recommended = recommended_kleinhirn(&policy, &census);
        (
            recommended.official_label.clone(),
            recommended.model_id.clone(),
        )
    });
    let focus = load_focus_state(paths)?;
    let next_task = list_open_tasks(paths, 1)?
        .into_iter()
        .next()
        .map(|task| format!("#{} {}", task.id, task.title))
        .unwrap_or_else(|| "none".to_string());

    Ok(format!(
        "status={}; homepage_stage={}; homepage_ready={}; bios_primary={}; branding_locked={}; owner={}; committed_owner={}; kleinhirn={} ({}); browser_status={}; chrome_binary={}; browser_desktop={}; browser_headless={}; agent_mode={}; active_task={}; queue_depth={}; next_task={}",
        if bios.frozen {
            "bios_frozen"
        } else {
            "bootstrap"
        },
        homepage.stage,
        homepage.homepage_ready,
        trust.bios_primary_channel_confirmed,
        homepage.owner_branding_locked,
        if organigram.owner.name.is_empty() {
            "unknown"
        } else {
            organigram.owner.name.as_str()
        },
        if trust.committed_owner_name.is_empty() {
            "uncommitted"
        } else {
            trust.committed_owner_name.as_str()
        },
        kleinhirn.0,
        kleinhirn.1,
        browser.status,
        browser.chrome_binary.as_deref().unwrap_or("none"),
        browser.desktop_available,
        browser.headless_ready,
        focus.mode,
        focus
            .active_task_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        focus.queue_depth,
        next_task,
    ))
}

fn terminal_browser_dom(paths: &Paths, rest: &str) -> anyhow::Result<String> {
    let url = rest.trim();
    if url.is_empty() {
        anyhow::bail!("usage: /browser-dom URL");
    }
    let result = run_browser_action(
        paths,
        &manual_browser_task("browser_dom"),
        &BrowserActionDirective {
            action: "dump_dom".to_string(),
            url: Some(url.to_string()),
            output_path: None,
            wait_ms: Some(5000),
            width: None,
            height: None,
            justification: Some("manual terminal browser dom inspection".to_string()),
            question: None,
        },
    )?;
    Ok(format!(
        "status={}; browser_status={}\n{}",
        result.status, result.browser_status, result.stdout
    ))
}

fn terminal_browser_screenshot(paths: &Paths, rest: &str) -> anyhow::Result<String> {
    let tokens = shlex::split(rest).context("failed to parse /browser-shot arguments")?;
    if tokens.is_empty() {
        anyhow::bail!("usage: /browser-shot URL [DATEI]");
    }
    let output_path = tokens.get(1).cloned();
    let result = run_browser_action(
        paths,
        &manual_browser_task("browser_screenshot"),
        &BrowserActionDirective {
            action: "screenshot".to_string(),
            url: Some(tokens[0].clone()),
            output_path,
            wait_ms: Some(5000),
            width: Some(1440),
            height: Some(2000),
            justification: Some("manual terminal browser screenshot".to_string()),
            question: None,
        },
    )?;
    Ok(format!(
        "status={}; browser_status={}; artifact={}",
        result.status,
        result.browser_status,
        result.artifact_path.as_deref().unwrap_or("none")
    ))
}

fn terminal_browser_open(paths: &Paths, rest: &str) -> anyhow::Result<String> {
    let url = rest.trim();
    if url.is_empty() {
        anyhow::bail!("usage: /browser-open URL");
    }
    let result = run_browser_action(
        paths,
        &manual_browser_task("browser_open"),
        &BrowserActionDirective {
            action: "open_url".to_string(),
            url: Some(url.to_string()),
            output_path: None,
            wait_ms: None,
            width: None,
            height: None,
            justification: Some("manual terminal browser open".to_string()),
            question: None,
        },
    )?;
    Ok(format!(
        "status={}; browser_status={}",
        result.status, result.browser_status
    ))
}

fn terminal_browser_workers(paths: &Paths) -> anyhow::Result<String> {
    let jobs = list_worker_jobs(paths, 10)?;
    if jobs.is_empty() {
        return Ok("No browser/repair worker jobs are available.".to_string());
    }
    Ok(jobs
        .iter()
        .map(|job| {
            format!(
                "#{} kind={} status={} title={} completed={} summary={}",
                job.id,
                job.worker_kind,
                job.status,
                job.contract_title,
                job.completed_at.as_deref().unwrap_or("-"),
                job.result_summary
                    .as_deref()
                    .unwrap_or(job.request_note.as_str()),
            )
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn terminal_browser_bridge(paths: &Paths) -> anyhow::Result<String> {
    let state = load_browser_agent_bridge_state(paths, 8)?;
    Ok(format!(
        "base_url={}; bridge_port={}; queued_jobs={}; leased_jobs={}; terminal_jobs={}; active_workers={}; extension_workspace={}; manifest={}",
        state.base_url,
        state.bridge_port,
        state.queued_jobs,
        state.leased_jobs,
        state.terminal_jobs,
        state.active_workers.len(),
        state.extension_workspace,
        state.manifest_path,
    ))
}

fn terminal_browser_extension_path(paths: &Paths) -> anyhow::Result<String> {
    Ok(format!(
        "workspace={}\nmanifest={}",
        browser_agent_extension_workspace(paths).display(),
        browser_agent_extension_manifest_path(paths).display(),
    ))
}

fn manual_browser_task(kind: &str) -> TaskRecord {
    TaskRecord {
        id: 0,
        created_at: String::new(),
        updated_at: String::new(),
        parent_task_id: None,
        worker_job_id: None,
        source_interrupt_id: None,
        source_channel: "terminal".to_string(),
        speaker: "terminal".to_string(),
        task_kind: kind.to_string(),
        title: kind.to_string(),
        detail: kind.to_string(),
        trust_level: "system".to_string(),
        priority_score: 0,
        status: "manual".to_string(),
        run_count: 0,
        last_checkpoint_summary: None,
        last_checkpoint_at: None,
        last_output: None,
    }
}

fn load_runtime_kleinhirn_display(paths: &Paths) -> Option<(String, String)> {
    let content = fs::read_to_string(paths.root.join("runtime/kleinhirn.env")).ok()?;
    let mut label = None;
    let mut model = None;
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some((key, value)) = trimmed.split_once('=') {
            let unquoted = value
                .trim()
                .trim_matches('\'')
                .trim_matches('"')
                .to_string();
            match key {
                "CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL" => label = Some(unquoted),
                "CTO_AGENT_KLEINHIRN_RUNTIME_MODEL" => model = Some(unquoted),
                _ => {}
            }
        }
    }
    match (label, model) {
        (Some(label), Some(model)) if !label.is_empty() && !model.is_empty() => {
            Some((label, model))
        }
        _ => None,
    }
}

pub fn drain_pending_loop_interrupts(paths: &Paths, max_items: usize) -> anyhow::Result<usize> {
    Ok(ingest_pending_loop_interrupts(paths, max_items)?.len())
}

pub fn queue_interrupt(
    paths: &Paths,
    interrupt_id: i64,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> anyhow::Result<AttachLineOutcome> {
    append_boot_entry(paths, speaker, message)?;
    let trust_note = if source_channel == "attach_terminal" {
        record_owner_chat_contact(paths, source_channel, speaker)?
    } else {
        record_terminal_feedback(paths, speaker, message)?
    };
    let signal_note = record_turn_signal_for_active_turn(paths, source_channel, speaker, message)?
        .map(|signal| {
            format!(
                "Signal {} was recorded on the running thread as {}.",
                signal.id, signal.signal_kind
            )
        });
    let focus = load_focus_state(paths)?;
    let open_tasks = list_open_tasks(paths, 3)?;
    let next_titles = open_tasks
        .iter()
        .map(|task| format!("#{} {}", task.id, task.title))
        .collect::<Vec<_>>()
        .join(" | ");
    let queued_task = if source_channel == "attach_terminal" {
        queue_loop_interrupt_as_task(paths, interrupt_id)?
    } else {
        None
    };
    let mut compaction_notes = Vec::new();
    if let Some(active_task_id) = focus.active_task_id {
        if let Ok(Some(active_task)) = load_task_by_id(paths, active_task_id) {
            if prepare_context_package_with_trigger(
                paths,
                &active_task,
                ContextCompactionTrigger::Interrupt,
            )
            .is_ok()
            {
                compaction_notes.push(format!(
                    "Interrupt compaction refreshed the running task context for #{} {}.",
                    active_task.id, active_task.title
                ));
            }
        }
    }
    if let Some(task) = queued_task.as_ref() {
        if prepare_context_package_with_trigger(paths, task, ContextCompactionTrigger::Interrupt)
            .is_ok()
        {
            compaction_notes.push(format!(
                "Interrupt compaction prepared the new task packet for #{} {}.",
                task.id, task.title
            ));
        }
    }

    let mut parts = vec![if let Some(task) = queued_task.as_ref() {
        format!(
            "Interrupt #{} from {} was recorded and immediately materialized as task #{} ({}).",
            interrupt_id, speaker, task.id, task.title
        )
    } else {
        format!(
            "Interrupt #{} from {} was recorded. The supervisor will materialize it into the task queue on the next intake tick.",
            interrupt_id, speaker
        )
    }];
    if let Some(note) = trust_note {
        parts.push(note);
    }
    if let Some(note) = signal_note {
        parts.push(note);
    }
    parts.push(format!(
        "The agent does not hard-abort the running bounded step, but finishes it cleanly up to the next safe turn boundary."
    ));
    parts.push(format!(
        "After that, the unified mode system prioritizes this input in the next reprioritization cycle."
    ));
    parts.extend(compaction_notes);
    parts.push(format!(
        "Current agent mode: mode={}; active_task={}; queue_depth={}.",
        focus.mode,
        focus
            .active_task_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        focus.queue_depth
    ));
    if !next_titles.is_empty() {
        parts.push(format!("Next open tasks: {}.", next_titles));
    }
    Ok(AttachLineOutcome {
        output: parts.join(" "),
        queued_task_id: queued_task.as_ref().map(|task| task.id),
        queued_task_title: queued_task.as_ref().map(|task| task.title.clone()),
    })
}

pub fn queue_channel_interrupt(
    paths: &Paths,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> anyhow::Result<AttachLineOutcome> {
    let signal_note = record_turn_signal_for_active_turn(paths, source_channel, speaker, message)?
        .map(|signal| {
            format!(
                "Signal {} was recorded on the running thread as {}.",
                signal.id, signal.signal_kind
            )
        });
    let interrupt_id = enqueue_loop_interrupt(paths, source_channel, speaker, message)?;
    let focus = load_focus_state(paths)?;
    let open_tasks = list_open_tasks(paths, 3)?;
    let next_titles = open_tasks
        .iter()
        .map(|task| format!("#{} {}", task.id, task.title))
        .collect::<Vec<_>>()
        .join(" | ");

    let mut parts = vec![format!(
        "Interrupt #{} from {} via {} was recorded. The supervisor will materialize it into the task queue on the next intake tick.",
        interrupt_id, speaker, source_channel
    )];
    if let Some(note) = signal_note {
        parts.push(note);
    }
    parts.push(
        "The agent does not hard-abort the running bounded step, but finishes it cleanly up to the next safe turn boundary."
            .to_string(),
    );
    parts.push(
        "After that, the unified mode system prioritizes this input in the next reprioritization cycle."
            .to_string(),
    );
    parts.push(format!(
        "Current agent mode: mode={}; active_task={}; queue_depth={}.",
        focus.mode,
        focus
            .active_task_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        focus.queue_depth
    ));
    if !next_titles.is_empty() {
        parts.push(format!("Next open tasks: {}.", next_titles));
    }
    Ok(AttachLineOutcome {
        output: parts.join(" "),
        queued_task_id: None,
        queued_task_title: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_inline_mail_runtime_update_extracts_structured_chat_credentials() {
        let existing = BTreeMap::new();
        let parsed = parse_inline_mail_runtime_update(
            "email: cto1@metric-space.ai, password: supersecret, imap host: imap.one.com, smtp host: send.one.com",
            &existing,
        )
        .expect("structured mail credentials should parse");

        assert_eq!(parsed.address, "cto1@metric-space.ai");
        assert_eq!(parsed.password, "supersecret");
        assert_eq!(parsed.imap_host.as_deref(), Some("imap.one.com"));
        assert_eq!(parsed.smtp_host.as_deref(), Some("send.one.com"));
    }

    #[test]
    fn parse_inline_mail_runtime_update_supports_env_style_fields() {
        let existing = BTreeMap::new();
        let parsed = parse_inline_mail_runtime_update(
            "CTO_EMAIL_ADDRESS=cto1@metric-space.ai; CTO_EMAIL_PASSWORD=supersecret; CTO_EMAIL_IMAP_PORT=993; CTO_EMAIL_SMTP_PORT=465",
            &existing,
        )
        .expect("env-style mail credentials should parse");

        assert_eq!(parsed.address, "cto1@metric-space.ai");
        assert_eq!(parsed.password, "supersecret");
        assert_eq!(parsed.imap_port.as_deref(), Some("993"));
        assert_eq!(parsed.smtp_port.as_deref(), Some("465"));
    }

    #[test]
    fn parse_inline_mail_runtime_update_reuses_existing_address_for_password_only_update() {
        let mut existing = BTreeMap::new();
        existing.insert(
            "CTO_EMAIL_ADDRESS".to_string(),
            "cto1@metric-space.ai".to_string(),
        );

        let parsed = parse_inline_mail_runtime_update("password: rotated-secret", &existing)
            .expect("password-only update should reuse existing mail address");

        assert_eq!(parsed.address, "cto1@metric-space.ai");
        assert_eq!(parsed.password, "rotated-secret");
    }

    #[test]
    fn parse_terminal_message_does_not_treat_mail_fields_as_speaker_prefix() {
        let (speaker, message) =
            parse_terminal_message("email: cto1@metric-space.ai, password: supersecret");

        assert_eq!(speaker, "Michael Welsch");
        assert_eq!(message, "email: cto1@metric-space.ai, password: supersecret");
    }
}

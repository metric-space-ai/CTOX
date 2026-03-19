use crate::browser_agent_bridge::browser_agent_extension_manifest_path;
use crate::browser_agent_bridge::browser_agent_extension_workspace;
use crate::browser_agent_bridge::load_browser_agent_bridge_state;
use crate::browser_engine::BrowserActionDirective;
use crate::browser_engine::browser_status_text;
use crate::browser_engine::run_browser_action;
use crate::browser_engine::start_browser_install_session;
use crate::browser_engine::start_browser_launch_session;
use crate::contracts::Paths;
use crate::contracts::append_boot_entry;
use crate::contracts::default_homepage_policy;
use crate::contracts::load_installation_bootstrap_state;
use crate::contracts::load_bios;
use crate::contracts::load_homepage_policy;
use crate::contracts::load_model_policy;
use crate::contracts::load_organigram;
use crate::contracts::now_iso;
use crate::contracts::recommended_kleinhirn;
use crate::contracts::save_homepage_policy;
use crate::contracts::save_installation_bootstrap_state;
use crate::contracts::save_organigram;
use crate::contracts::InstallationBootstrapState;
use crate::command_exec::list_sessions;
use crate::command_exec::read_session;
use crate::command_exec::resize_session;
use crate::command_exec::start_session;
use crate::command_exec::terminate_session;
use crate::command_exec::write_session;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::enqueue_loop_interrupt;
use crate::runtime_db::load_agent_thread;
use crate::runtime_db::ingest_pending_loop_interrupts;
use crate::runtime_db::list_worker_jobs;
use crate::runtime_db::list_open_tasks;
use crate::runtime_db::list_recent_loop_incidents;
use crate::runtime_db::list_recent_turn_signals;
use crate::runtime_db::list_recent_agent_turns;
use crate::runtime_db::record_memory;
use crate::runtime_db::record_homepage_revision;
use crate::runtime_db::record_terminal_feedback;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::enqueue_internal_task;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::queue_loop_interrupt_as_task;
use crate::runtime_db::record_turn_signal_for_active_turn;
use crate::runtime_db::list_recent_agent_events;
use crate::supervisor::inspect_local_resources;
use anyhow::Context;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use codex_app_server_protocol::CommandExecParams;
use codex_app_server_protocol::CommandExecResizeParams;
use codex_app_server_protocol::CommandExecTerminalSize;
use codex_app_server_protocol::CommandExecTerminateParams;
use codex_app_server_protocol::CommandExecWriteParams;
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
    handle_input_line(paths, line, "attach_terminal")
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

    let owner_name = prompt_line("Owner-Name", state.owner_name.trim())?;
    let owner_contact_email = prompt_line("Owner-Kontakt-E-Mail", state.owner_contact_email.trim())?;
    let owner_contact_info = prompt_multiline(
        "Optional: Weitere Owner-Kontaktinfos oder Hinweise.\nMehrere Zeilen sind moeglich. Leere Zeile beendet die Eingabe.",
        &state.owner_contact_info,
    )?;
    let email_mode = prompt_email_assignment_mode(&state.email_assignment_mode)?;
    let email_prompt = match email_mode.as_str() {
        "assigned_now" => "Gib jetzt alle Freitext-Infos zum zugewiesenen E-Mail-Postfach ein.\nBeispiele: Adresse, Anbieter, IMAP/SMTP-Hosts, Zugang, Regeln, Ansprechpartner.\nMehrere Zeilen sind moeglich. Leere Zeile beendet die Eingabe.",
        "self_procure" => "Beschreibe, wie der CTO-Agent sich selbst ein E-Mail-Postfach besorgen soll.\nBeispiele: bevorzugter Anbieter, Domain, Constraints, Budget, Freigaben, Ansprechpartner.\nMehrere Zeilen sind moeglich. Leere Zeile beendet die Eingabe.",
        _ => "Optional: Gib Freitext-Hinweise zur spaeteren E-Mail-Einrichtung oder zu Kommunikationswuenschen ein.\nMehrere Zeilen sind moeglich. Leere Zeile beendet die Eingabe.",
    };
    let email_note = prompt_multiline(email_prompt, &state.email_bootstrap_note)?;
    let installer_free_text = prompt_multiline(
        "Optional: Weitere Startup-Infos fuer die fruehe Kommunikation, das Dashboard oder die ersten Kontakte.\nMehrere Zeilen sind moeglich. Leere Zeile beendet die Eingabe.",
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
            "Kommunikations-Bootstrap erfasst. Terminalbefehl: {}. E-Mail-Modus: {}. Weitere Details liegen im Installation-Bootstrap-Contract.",
            state.terminal_command,
            installation_email_mode_label(&state.email_assignment_mode),
        ),
    )?;
    record_memory(
        paths,
        "installation_bootstrap",
        &format!(
            "Installations-Kommunikationsbriefing erfasst ({})",
            installation_email_mode_label(&state.email_assignment_mode)
        ),
        &detail,
        "install_bootstrap_tui",
    )?;
    let _ = enqueue_internal_task(
        paths,
        None,
        "installation_bootstrap",
        "Installations-Kommunikationsbriefing auswerten",
        &detail,
        installation_bootstrap_priority(&state.email_assignment_mode),
    );

    println!();
    println!("Kommunikations-Bootstrap gespeichert.");
    if !state.owner_name.trim().is_empty() || !state.owner_contact_email.trim().is_empty() {
        println!(
            "Owner: {} {}",
            if state.owner_name.trim().is_empty() {
                "(ohne Namen)".to_string()
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
    println!("Terminal low-level: {} -> startet die direkte CTO-Kommunikation.", state.terminal_command);
    println!("Komfortpfad: lokale Dashboard-/Intranet-Seite unter der CTO-Control-Plane.");
    println!("E-Mail-Modus: {}", installation_email_mode_label(&state.email_assignment_mode));
    if !state.email_bootstrap_note.trim().is_empty() {
        println!("E-Mail-Hinweise wurden gespeichert.");
    }
    if !state.installer_free_text.trim().is_empty() {
        println!("Weitere Startup-Hinweise wurden gespeichert.");
    }
    println!("Spaetere Ergaenzungen bleiben ueber Terminal und Dashboard moeglich.");
    Ok(())
}

fn print_install_bootstrap_intro(state: &InstallationBootstrapState) {
    println!();
    println!("CTO-Agent Installations-Bootstrap");
    println!("--------------------------------");
    println!(
        "Low-level-Kommunikation bleibt immer ueber den Terminalbefehl `{}` moeglich.",
        state.terminal_command
    );
    println!(
        "Parallel wird der CTO-Agent normalerweise eine lokale Intranet-/Dashboard-Seite fuer komfortablere Kommunikation bereitstellen."
    );
    println!(
        "Wenn E-Mail spaeter sinnvoll ist, kann der Besitzer jetzt schon Freitext-Infos droppen oder sie spaeter im Terminal bzw. Dashboard nachreichen."
    );
    println!();
}

fn prompt_email_assignment_mode(current: &str) -> anyhow::Result<String> {
    loop {
        println!("Wie soll der CTO-Agent mit E-Mail starten?");
        println!("  1) Ihm wird jetzt ein Postfach zugewiesen");
        println!("  2) Er soll sich spaeter selbst ein Postfach besorgen");
        println!("  3) Spaeter entscheiden / vorerst ohne E-Mail starten");
        let default_choice = match current {
            "assigned_now" => "1",
            "self_procure" => "2",
            _ => "3",
        };
        let raw = prompt_line(
            &format!("Auswahl [default {}]", default_choice),
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
        println!("Bitte 1, 2 oder 3 eingeben.");
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
        println!("Vorhandener Wert:");
        println!("{}", existing.trim());
        println!("Leere erste Zeile behaelt den vorhandenen Wert.");
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
        "assigned_now" => "Postfach wird zugewiesen",
        "self_procure" => "Agent soll Postfach selbst besorgen",
        _ => "Spaeter entscheiden / vorerst ohne E-Mail",
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
        "Installations-Kommunikationsbriefing. Interpretiere dies als fruehe Startup-Direktive fuer Kommunikation und Kontaktaufbau. Owner: {} <{}>. Weitere Owner-Kontaktinfos: {}. Terminal low-level command: `{}`. Dashboard/Intranet: {}. E-Mail-Modus: {}. E-Mail-Hinweise: {}. Weitere Installer-Hinweise: {}. Wenn ein Postfach direkt zugewiesen wurde, sollst du pruefen, ob du dir mit Skill und Template einen passenden Kommunikationsclient bauen kannst, den Mailzugang herstellen und nach bounded Verifikation eine Testmail schreiben. Der Owner darf spaeter weitere Angaben ueber Terminal oder Dashboard nachreichen.",
        if state.owner_name.trim().is_empty() {
            "unbekannt".to_string()
        } else {
            state.owner_name.trim().to_string()
        },
        if state.owner_contact_email.trim().is_empty() {
            "keine".to_string()
        } else {
            state.owner_contact_email.trim().to_string()
        },
        if state.owner_contact_info.trim().is_empty() {
            "keine".to_string()
        } else {
            state.owner_contact_info.trim().to_string()
        },
        state.terminal_command,
        state.dashboard_note,
        installation_email_mode_label(&state.email_assignment_mode),
        if state.email_bootstrap_note.trim().is_empty() {
            "keine".to_string()
        } else {
            state.email_bootstrap_note.trim().to_string()
        },
        if state.installer_free_text.trim().is_empty() {
            "keine".to_string()
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

    if policy.current_intro.trim().is_empty() || policy.current_intro == default_policy.current_intro {
        policy.current_intro = "Die erste Kommunikation kann immer ueber `cto` im Terminal beginnen. Parallel soll diese lokale Control-Plane zur komfortableren Intranet-Oberflaeche ausgebaut werden.".to_string();
    }
    if policy.communication_note.trim().is_empty()
        || policy.communication_note == default_policy.communication_note
    {
        policy.communication_note = format!(
            "Low-level-Kommunikation geht immer ueber `{}`. Die lokale Dashboard-/Intranet-Seite soll der komfortablere Kanal werden. E-Mail-Startmodus: {}. Weitere Angaben koennen jetzt oder spaeter im Terminal und Dashboard nachgereicht werden.",
            state.terminal_command,
            installation_email_mode_label(&state.email_assignment_mode)
        );
    }
    if policy.terminal_fallback_note.trim().is_empty()
        || policy.terminal_fallback_note == default_policy.terminal_fallback_note
    {
        policy.terminal_fallback_note = format!(
            "Wenn Homepage oder E-Mail noch nicht traegt, bleibt `{}` im Terminal die primaere Vollzugriffsebene.",
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

    let (speaker, message) = parse_terminal_message(trimmed);
    let interrupt_id = enqueue_loop_interrupt(paths, source_channel, &speaker, &message)?;
    queue_interrupt(paths, interrupt_id, source_channel, &speaker, &message)
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
    policy.template_name = if source_channel == "bios" {
        "bios-shaped-bridge".to_string()
    } else {
        "terminal-shaped-bridge".to_string()
    };
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
        "Letzter Gestaltungsimpuls von {}: {}",
        speaker.trim(),
        summarize_feedback(message, 220)
    );
    policy.communication_note = format!(
        "Diese Oberflaeche wird gerade ueber {}-Feedback aufgebaut. Wenn sie noch nicht passt, darf der CTO-Agent sie weiter ueber Skill, Template und Chat umformen.",
        source_label
    );
    policy.terminal_fallback_note =
        "Wenn die Homepage nicht traegt, bleibt das laufende Terminal die primaere Vollzugriffsebene."
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
            "homepage-bootstrap skill applied from {} feedback.",
            source_label
        ),
    )?;

    Ok(Some(format!(
        "Homepage ueber homepage-bootstrap aus {}-Feedback nachgezogen.",
        source_label
    )))
}

fn parse_terminal_message(line: &str) -> (String, String) {
    if let Some((speaker, message)) = line.split_once(':') {
        let speaker = speaker.trim();
        let message = message.trim();
        if !speaker.is_empty() && !message.is_empty() {
            return (speaker.to_string(), message.to_string());
        }
    }

    ("Michael Welsch".to_string(), line.trim().to_string())
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
            "Diese Homepage wird nach {} so umgebaut, dass das BIOS sichtbar und die Kommunikationsrichtung klar wird.",
            source_label
        );
    }
    if lowered.contains("vertrauen") || lowered.contains("trust") {
        return format!(
            "Diese Homepage wird nach {} neu auf Vertrauen, Klarheit und Root-of-Trust kalibriert.",
            source_label
        );
    }
    if lowered.contains("nicht")
        || lowered.contains("falsch")
        || lowered.contains("umbauen")
        || lowered.contains("umgestalten")
    {
        return format!(
            "Diese Homepage wird nach {} neu aufgebaut und bleibt absichtlich veraenderbar.",
            source_label
        );
    }

    format!(
        "Diese Homepage reagiert auf {} und bleibt ein veraenderbarer erster Kommunikationspfad.",
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
        "  /status               zeigt Bootstrap-, BIOS- und Homepage-Status",
        "  /thread               zeigt den persistierten Main-Thread-Zustand",
        "  /signals              zeigt die letzten Turn-Steer-/Interruptsignale",
        "  /incidents            zeigt die letzten Loop-Incidents und Recovery-Faelle",
        "  /events               zeigt die letzten Agentenereignisse",
        "  /turns                zeigt die letzten bounded Agent-Turns",
        "  /exec-sessions        zeigt codex-backed Exec-Sessions",
        "  /exec-start ...       startet eine codex-backed Exec-Session",
        "  /exec-write ...       schreibt in stdin einer Exec-Session",
        "  /exec-resize ...      resized eine PTY-Exec-Session",
        "  /exec-read ...        zeigt gepuffertes stdout/stderr einer Exec-Session",
        "  /exec-terminate ...   beendet eine Exec-Session",
        "  /browser-status       zeigt Status der expliziten Browser-Engine",
        "  /browser-workers      zeigt Browser-/Repair-/Specialist-Worker-Jobs",
        "  /browser-bridge       zeigt Status der Chrome-Extension-Bridge",
        "  /browser-extension-path  zeigt den entkoppelten Extension-Workspace",
        "  /browser-install      startet den Chrome-/Browser-Installer ueber die CLI-Engine",
        "  /browser-launch       startet Chrome mit der Browser-Agent-Extension im Desktop",
        "  /browser-dom URL      liest eine Seite ueber Chrome headless aus",
        "  /browser-shot URL [DATEI]  erstellt einen Screenshot ueber Chrome headless",
        "  /browser-open URL     oeffnet eine URL interaktiv in Chrome",
        "  /help                 zeigt diese Hilfe",
        "  Michael Welsch: ...   wird als Interrupt in den laufenden Always-on-Loop eingespeist",
        "                        der aktuelle bounded Schritt wird nicht hart abgebrochen,",
        "                        sondern erst sauber beendet und dann neu priorisiert",
        "Beispiel:",
        "  Michael Welsch: Dieser Kommunikationsweg gefaellt mir nicht. Mach das BIOS sichtbarer und halte das Terminal als Fallback.",
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
                timeout_ms = tokens.get(index).and_then(|value| value.parse::<u64>().ok());
            }
            "--rows" => {
                index += 1;
                rows = tokens.get(index).and_then(|value| value.parse::<u16>().ok());
            }
            "--cols" => {
                index += 1;
                cols = tokens.get(index).and_then(|value| value.parse::<u16>().ok());
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
            env: None,
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
        return Ok("Keine Turn-Signale vorhanden.".to_string());
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
        return Ok("Keine Agentenereignisse vorhanden.".to_string());
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
        return Ok("Keine Loop-Incidents vorhanden.".to_string());
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
        return Ok("Keine Agent-Turns vorhanden.".to_string());
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
        if bios.frozen { "bios_frozen" } else { "bootstrap" },
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
        return Ok("Keine Browser-/Repair-Worker-Jobs vorhanden.".to_string());
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
                job.result_summary.as_deref().unwrap_or(job.request_note.as_str()),
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
    let trust_note = record_terminal_feedback(paths, speaker, message)?;
    let signal_note = record_turn_signal_for_active_turn(paths, source_channel, speaker, message)?
        .map(|signal| {
            format!(
                "Signal {} wurde am laufenden Thread als {} vermerkt.",
                signal.id, signal.signal_kind
            )
        });
    let task = queue_loop_interrupt_as_task(paths, interrupt_id)?
        .ok_or_else(|| anyhow::anyhow!("interrupt {interrupt_id} could not be queued"))?;
    let focus = load_focus_state(paths)?;
    let open_tasks = list_open_tasks(paths, 3)?;
    let next_titles = open_tasks
        .iter()
        .map(|task| format!("#{} {}", task.id, task.title))
        .collect::<Vec<_>>()
        .join(" | ");

    let mut parts = vec![format!(
        "Interrupt von {} als Aufgabe #{} aufgenommen: {}.",
        speaker, task.id, task.title
    )];
    if let Some(note) = trust_note {
        parts.push(note);
    }
    if let Some(note) = signal_note {
        parts.push(note);
    }
    parts.push(format!(
        "Der Agent bricht den laufenden bounded Schritt nicht hart ab, sondern bringt ihn erst sauber bis zur naechsten sicheren Turn-Grenze zu Ende."
    ));
    parts.push(format!(
        "Danach zieht das einheitliche Modussystem diese Eingabe im naechsten Repriorisierungszyklus vor."
    ));
    parts.push(format!(
        "Aktueller Agentenmodus: mode={}; active_task={}; queue_depth={}.",
        focus.mode,
        focus
            .active_task_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        focus.queue_depth
    ));
    if !next_titles.is_empty() {
        parts.push(format!("Naechste offene Aufgaben: {}.", next_titles));
    }
    Ok(AttachLineOutcome {
        output: parts.join(" "),
        queued_task_id: Some(task.id),
        queued_task_title: Some(task.title.clone()),
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
                "Signal {} wurde am laufenden Thread als {} vermerkt.",
                signal.id, signal.signal_kind
            )
        });
    let interrupt_id = enqueue_loop_interrupt(paths, source_channel, speaker, message)?;
    let task = queue_loop_interrupt_as_task(paths, interrupt_id)?
        .ok_or_else(|| anyhow::anyhow!("interrupt {interrupt_id} could not be queued"))?;
    let focus = load_focus_state(paths)?;
    let open_tasks = list_open_tasks(paths, 3)?;
    let next_titles = open_tasks
        .iter()
        .map(|task| format!("#{} {}", task.id, task.title))
        .collect::<Vec<_>>()
        .join(" | ");

    let mut parts = vec![format!(
        "Interrupt von {} ueber {} als Aufgabe #{} aufgenommen: {}.",
        speaker, source_channel, task.id, task.title
    )];
    if let Some(note) = signal_note {
        parts.push(note);
    }
    parts.push(
        "Der Agent bricht den laufenden bounded Schritt nicht hart ab, sondern bringt ihn erst sauber bis zur naechsten sicheren Turn-Grenze zu Ende."
            .to_string(),
    );
    parts.push(
        "Danach zieht das einheitliche Modussystem diese Eingabe im naechsten Repriorisierungszyklus vor."
            .to_string(),
    );
    parts.push(format!(
        "Aktueller Agentenmodus: mode={}; active_task={}; queue_depth={}.",
        focus.mode,
        focus
            .active_task_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        focus.queue_depth
    ));
    if !next_titles.is_empty() {
        parts.push(format!("Naechste offene Aufgaben: {}.", next_titles));
    }
    Ok(AttachLineOutcome {
        output: parts.join(" "),
        queued_task_id: Some(task.id),
        queued_task_title: Some(task.title.clone()),
    })
}

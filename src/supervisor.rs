use crate::agentic::choose_next_task_focus;
use crate::agentic::ExecSessionDirective;
use crate::agentic::run_agentic_task_once;
use crate::agentic::probe_kleinhirn_health;
use crate::agentic::should_run_agentic_loop;
use crate::brain_runtime::apply_recommended_browser_vision_kleinhirn_upgrade;
use crate::brain_runtime::attempt_kleinhirn_runtime_repair;
use crate::brain_runtime::apply_targeted_kleinhirn_upgrade;
use crate::brain_runtime::extract_requested_local_kleinhirn_model;
use crate::brain_runtime::grosshirn_runtime_configured;
use crate::brain_runtime::local_kleinhirn_upgrade_available;
use crate::brain_runtime::prepare_grosshirn_activation_from_message;
use crate::browser_subworkers::advance_browser_subworkers;
use crate::browser_engine::inspect_browser_engine;
use crate::browser_engine::run_browser_action;
use crate::command_exec::read_session;
use crate::command_exec::snapshot_session;
use crate::command_exec::start_session;
use crate::command_exec::terminate_session;
use crate::command_exec::write_session;
use crate::context_controller::prepare_context_package;
use crate::runtime_db::block_task;
use crate::runtime_db::arm_task_grosshirn_boost;
use crate::runtime_db::attach_dispatch_task_to_candidate;
use crate::runtime_db::attach_validation_task_to_candidate;
use crate::runtime_db::complete_review_task;
use crate::runtime_db::complete_task;
use crate::runtime_db::delegate_task_to_worker;
use crate::runtime_db::emit_self_review_task;
use crate::runtime_db::expire_stale_grosshirn_boost;
use crate::runtime_db::grosshirn_boost_available;
use crate::runtime_db::ingest_pending_loop_interrupts;
use crate::runtime_db::is_agent_turn_in_progress;
use crate::runtime_db::load_active_agent_turn;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::load_latest_completed_agent_turn;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::load_proactive_contact_candidate_by_dispatch_task;
use crate::runtime_db::load_task_by_id;
use crate::runtime_db::latest_open_task_by_kind;
use crate::runtime_db::record_bios_dialogue;
use crate::runtime_db::record_brain_usage_event;
use crate::runtime_db::record_homepage_revision;
use crate::runtime_db::record_memory;
use crate::runtime_db::record_proactive_contact_dispatch_result;
use crate::runtime_db::register_loop_incident;
use crate::runtime_db::recover_orphaned_active_turns;
use crate::runtime_db::refresh_task_grosshirn_boost;
use crate::runtime_db::release_task_grosshirn_boost;
use crate::runtime_db::resolve_loop_incident;
use crate::runtime_db::reprioritize_tasks;
use crate::runtime_db::requeue_task;
use crate::runtime_db::set_agent_mode;
use crate::runtime_db::store_proactive_contact_candidate;
use crate::runtime_db::apply_proactive_contact_validation;
use crate::runtime_db::store_learning_entries;
use crate::runtime_db::sync_agent_thread;
use crate::runtime_db::watchdog_interrupt_live_turn;
use crate::runtime_db::task_has_active_grosshirn_boost;
use crate::runtime_db::start_agent_turn;
use crate::runtime_db::complete_agent_turn;
use crate::contracts::AgentState;
use crate::contracts::BrainModel;
use crate::contracts::GpuDevice;
use crate::contracts::LoopSafetyPolicy;
use crate::contracts::ModelTuneCandidate;
use crate::contracts::load_census;
use crate::contracts::load_model_policy;
use crate::contracts::load_loop_safety_policy;
use crate::contracts::load_self_preservation_state;
use crate::contracts::load_root_auth;
use crate::contracts::Paths;
use crate::contracts::SystemCensus;
use crate::contracts::load_bios;
use crate::contracts::load_homepage_policy;
use crate::contracts::load_organigram;
use crate::contracts::now_iso;
use crate::contracts::path_display_name;
use crate::runtime_db::sync_model_resources;
use crate::runtime_db::sync_owner_trust;
use crate::runtime_db::sync_resources_from_census;
use crate::runtime_db::sync_skills;
use crate::contracts::write_agent_state;
use crate::contracts::write_census;
use crate::desktop_session::detect_desktop_session_env;
use crate::tooling::apply_homepage_update;
use crate::tooling::ExecCommandDirective;
use crate::tooling::run_bounded_command;
use anyhow::Context;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use codex_app_server_protocol::CommandExecParams;
use codex_app_server_protocol::CommandExecTerminalSize;
use codex_app_server_protocol::CommandExecTerminateParams;
use codex_app_server_protocol::CommandExecWriteParams;
use gethostname::gethostname;
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use std::time::Instant;

struct RunningAgenticTask {
    turn_id: i64,
    task_id: i64,
    task_title: String,
    started_at: Instant,
    handle: tokio::task::JoinHandle<()>,
}

struct TaskEscalationDecision {
    task_status: String,
    next_mode: String,
    checkpoint_summary: String,
    checkpoint_detail: String,
    spawn_self_preservation_task: bool,
}

pub fn spawn_supervisor(paths: Paths, started_at: Instant) {
    tokio::spawn(async move {
        let tick_ms = std::env::var("CTO_AGENT_SUPERVISOR_TICK_MS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|value| *value >= 100)
            .unwrap_or(5000);
        let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));
        let mut ticks: u64 = 0;
        let mut agentic_task: Option<RunningAgenticTask> = None;
        loop {
            interval.tick().await;
            ticks += 1;
            let stale_after_secs = std::env::var("CTO_AGENT_ACTIVE_TURN_STALE_SECS")
                .ok()
                .and_then(|raw| raw.parse::<u64>().ok())
                .unwrap_or(300);
            let _ = set_agent_mode(
                &paths,
                "observe",
                None,
                "",
                "Observe current signals, resources and queued work before reprioritization.",
            );
            if let Some(running) = agentic_task.take() {
                if running.handle.is_finished() {
                    match running.handle.await {
                        Ok(()) => {}
                        Err(err) => {
                            let summary =
                                format!("Turn {} ist im Rust-Kern abgestuerzt.", running.turn_id);
                            let detail = err.to_string();
                            let _ = register_loop_incident(
                                &paths,
                                "bounded_turn_crash",
                                "critical",
                                &summary,
                                &detail,
                                Some(running.task_id),
                                Some(running.turn_id),
                                false,
                                true,
                            );
                            let _ = block_task(
                                &paths,
                                running.task_id,
                                &summary,
                                &detail,
                                None,
                            );
                            let _ = complete_agent_turn(
                                &paths,
                                running.turn_id,
                                "crashed",
                                "blocked",
                                &summary,
                                &detail,
                            );
                            let _ = set_agent_mode(
                                &paths,
                                "blocked",
                                Some(running.task_id),
                                &running.task_title,
                                "Ein bounded Turn ist intern abgestuerzt und wurde hart blockiert.",
                            );
                            eprintln!(
                                "bounded task loop crashed for task {} / turn {}: {}",
                                running.task_id, running.turn_id, err
                            );
                        }
                    }
                } else {
                    let age = running.started_at.elapsed().as_secs();
                    if age > stale_after_secs {
                        if let Err(err) = watchdog_interrupt_live_turn(
                            &paths,
                            running.turn_id,
                            running.task_id,
                            &running.task_title,
                            age,
                            stale_after_secs,
                        ) {
                            eprintln!(
                                "live turn watchdog recovery failed for task {} / turn {}: {}",
                                running.task_id, running.turn_id, err
                            );
                            agentic_task = Some(running);
                        } else {
                            eprintln!(
                                "watchdog interrupted live stale turn {} for task {} after {}s",
                                running.turn_id, running.task_id, age
                            );
                        }
                    } else {
                        agentic_task = Some(running);
                    }
                }
            }

            if let Err(err) = recover_orphaned_active_turns(
                &paths,
                agentic_task.as_ref().map(|running| running.turn_id),
                stale_after_secs,
            ) {
                eprintln!("orphaned turn recovery failed: {err}");
            }

            match expire_stale_grosshirn_boost(&paths) {
                Ok(Some(note)) => {
                    let _ = record_memory(
                        &paths,
                        "brain_routing",
                        "Temporärer Grosshirn-Boost abgeklungen",
                        &note,
                        "grosshirn_boost_expired",
                    );
                }
                Ok(None) => {}
                Err(err) => eprintln!("grosshirn boost expiry check failed: {err}"),
            }

            if let Err(err) = ingest_pending_loop_interrupts(&paths, 8) {
                eprintln!("loop interrupt ingest failed: {err}");
            }

            if let Err(err) = reprioritize_tasks(&paths) {
                eprintln!("task reprioritization failed: {err}");
            }
            if let Err(err) = ensure_cto_obligations(&paths) {
                eprintln!("obligation generation failed: {err}");
            }

            if ticks == 1 || ticks % 3 == 0 {
                if let Err(err) = probe_kleinhirn_health(&paths) {
                    let detail = err.to_string();
                    let _ = register_loop_incident(
                        &paths,
                        "kleinhirn_unavailable",
                        "critical",
                        "Periodic kleinhirn health probe failed.",
                        &detail,
                        None,
                        None,
                        true,
                        true,
                    );
                    match attempt_kleinhirn_runtime_repair(&paths) {
                        Ok(repair_note) => {
                            let _ = resolve_loop_incident(
                                &paths,
                                "kleinhirn_unavailable",
                                &repair_note,
                            );
                            let _ = record_memory(
                                &paths,
                                "self_preservation",
                                "Kernel self-repair stellte das Kleinhirn wieder her",
                                &repair_note,
                                "kleinhirn_runtime_self_repair",
                            );
                            let _ = set_agent_mode(
                                &paths,
                                "self_preservation",
                                None,
                                "",
                                &repair_note,
                            );
                        }
                        Err(repair_err) => {
                            eprintln!(
                                "kleinhirn runtime self-repair failed after periodic probe failure: {repair_err}"
                            );
                        }
                    }
                    eprintln!("kleinhirn health probe failed: {detail}");
                }
            }

            match advance_browser_subworkers(&paths, 2) {
                Ok(completed) if !completed.is_empty() => {
                    let note = format!(
                        "{} delegierte Worker-Jobs haben Review-Arbeit zurueckgemeldet.",
                        completed.len()
                    );
                    let _ = set_agent_mode(&paths, "await_review", None, "", &note);
                }
                Ok(_) => {}
                Err(err) => eprintln!("worker job advance failed: {err}"),
            }

            let should_tick_agentic = ticks == 1 || ticks % 2 == 0;
            if should_tick_agentic && agentic_task.is_none() && should_run_agentic_loop(&paths) {
                let task = match choose_next_task_focus(&paths) {
                    Ok(task) => task,
                    Err(err) => {
                        eprintln!("task selection failed: {err}");
                        None
                    }
                };

                if task.is_none() {
                    if let Ok(focus) = load_focus_state(&paths) {
                        if focus.mode != "idle" && focus.queue_depth == 0 {
                            eprintln!("supervisor idle: no queued tasks");
                        }
                    }
                    continue;
                }

                let task = task.expect("checked above");
                if task.task_kind == "local_model_switch" {
                    if let Ok(Some(newer_task)) = latest_open_task_by_kind(&paths, "local_model_switch")
                        && newer_task.id > task.id
                    {
                        let summary = format!(
                            "Lokaler Modellwechsel wurde von neuerem Owner-/BIOS-Wunsch #{} supersediert.",
                            newer_task.id
                        );
                        let detail = format!(
                            "Task #{} sollte {} umsetzen, aber Task #{} fordert inzwischen einen neueren lokalen Modellwechsel an.\n\nAlter Detailtext:\n{}\n\nNeuer Detailtext:\n{}",
                            task.id,
                            extract_requested_local_kleinhirn_model(&task.detail)
                                .unwrap_or_else(|| "auto".to_string()),
                            newer_task.id,
                            task.detail,
                            newer_task.detail,
                        );
                        let _ = complete_task(&paths, task.id, &summary, &detail, None);
                        let _ = set_agent_mode(
                            &paths,
                            "reprioritize",
                            None,
                            "",
                            "Neuerer lokaler Modellwechsel hat einen aelteren Switch supersediert.",
                        );
                        continue;
                    }
                    let _ = set_agent_mode(
                        &paths,
                        "execute_task",
                        Some(task.id),
                        &task.title,
                        &format!(
                            "Task {} entered deterministic local model switch mode.",
                            task.id
                        ),
                    );
                    let requested_target = extract_requested_local_kleinhirn_model(&task.detail);
                    match apply_targeted_kleinhirn_upgrade(&paths, requested_target.as_deref()) {
                        Ok(outcome) => {
                            let detail = format!(
                                "Expliziter Owner-/BIOS-Modellwechsel wurde kernel-seitig ausgefuehrt.\nRequested target: {}\nChanged: {}\nRestarted: {}\nPrevious runtime: {}\nCurrent runtime: {}\nSummary: {}",
                                requested_target.as_deref().unwrap_or("auto"),
                                outcome.changed,
                                outcome.restarted,
                                outcome.previous_runtime_model.as_deref().unwrap_or("unknown"),
                                outcome.current_runtime_model.as_deref().unwrap_or("unknown"),
                                outcome.summary,
                            );
                            let _ = record_memory(
                                &paths,
                                "brain_runtime",
                                "Lokaler Kleinhirn-Modellwechsel ausgefuehrt",
                                &detail,
                                "local_model_switch",
                            );
                            if let Err(err) = complete_task(
                                &paths,
                                task.id,
                                &outcome.summary,
                                &detail,
                                outcome.current_runtime_model.as_deref(),
                            ) {
                                eprintln!(
                                    "failed to complete deterministic local model switch task {}: {err}",
                                    task.id
                                );
                            }
                        }
                        Err(err) => {
                            let detail = err.to_string();
                            let _ = register_loop_incident(
                                &paths,
                                "local_model_switch_failure",
                                "high",
                                "Deterministic local model switch failed.",
                                &detail,
                                Some(task.id),
                                None,
                                false,
                                true,
                            );
                            let _ = block_task(
                                &paths,
                                task.id,
                                "Lokaler Kleinhirn-Modellwechsel ist fehlgeschlagen.",
                                &detail,
                                None,
                            );
                        }
                    }
                    continue;
                }
                if task.task_kind == "grosshirn_activation" {
                    match prepare_grosshirn_activation_from_message(&paths, &task.detail) {
                        Ok(outcome) => {
                            if outcome.configured {
                                let _ = crate::runtime_db::set_brain_access_mode(
                                    &paths,
                                    "kleinhirn_plus_grosshirn",
                                );
                            }
                            let _ = record_memory(
                                &paths,
                                "brain_runtime",
                                "Grosshirn-Aktivierung vorbereitet",
                                &format!(
                                    "Task #{} {}\n{}\nConfigured: {}\nAPI key from message: {}\nTarget model: {}",
                                    task.id,
                                    task.title,
                                    outcome.summary,
                                    outcome.configured,
                                    outcome.api_key_from_message,
                                    outcome.target_model,
                                ),
                                "grosshirn_activation_prep",
                            );
                        }
                        Err(err) => {
                            let detail = err.to_string();
                            let _ = register_loop_incident(
                                &paths,
                                "grosshirn_activation_prep_failure",
                                "high",
                                "Grosshirn activation preflight failed before bounded execution.",
                                &detail,
                                Some(task.id),
                                None,
                                false,
                                true,
                            );
                        }
                    }
                } else if task_has_active_grosshirn_boost(&paths, task.id) {
                    let _ = refresh_task_grosshirn_boost(&paths, task.id);
                }
                let execution_mode = execution_mode_for_task(&task);
                let _ = set_agent_mode(
                    &paths,
                    execution_mode,
                    Some(task.id),
                    &task.title,
                    &format!(
                        "Task {} entered bounded {} mode.",
                        task.id, execution_mode
                    ),
                );
                if task.task_kind == "proactive_contact_dispatch" {
                    if let Err(err) = execute_proactive_contact_dispatch_task(&paths, &task) {
                        let summary = format!(
                            "Versandtask {} konnte nicht sauber abgeschlossen werden.",
                            task.id
                        );
                        let detail = err.to_string();
                        let _ = register_loop_incident(
                            &paths,
                            "proactive_contact_dispatch_failure",
                            "medium",
                            &summary,
                            &detail,
                            Some(task.id),
                            None,
                            false,
                            false,
                        );
                        let _ = block_task(&paths, task.id, &summary, &detail, None);
                    }
                    continue;
                }
                if let Err(err) = prepare_context_package(&paths, &task) {
                    let summary = format!("Task {} konnte kein Kontextpaket aufbauen.", task.id);
                    let detail = err.to_string();
                    let _ = register_loop_incident(
                        &paths,
                        "context_preflight_failure",
                        "high",
                        &summary,
                        &detail,
                        Some(task.id),
                        None,
                        false,
                        true,
                    );
                    let _ = block_task(&paths, task.id, &summary, &detail, None);
                    eprintln!("context preflight failed for task {}: {err}", task.id);
                    continue;
                }
                let turn = match start_agent_turn(
                    &paths,
                    task.id,
                    &task.title,
                    "supervisor_tick",
                    execution_mode,
                ) {
                    Ok(turn) => turn,
                    Err(err) => {
                        let summary = format!("Task {} konnte keinen Turn starten.", task.id);
                        let detail = err.to_string();
                        let _ = block_task(&paths, task.id, &summary, &detail, None);
                        eprintln!("failed to start turn for task {}: {err}", task.id);
                        continue;
                    }
                };
                let loop_paths = paths.clone();
                let task_id = task.id;
                let turn_id = turn.id;
                let task_title = task.title.clone();
                let handle = tokio::task::spawn_blocking(move || {
                    match run_agentic_task_once(&loop_paths, "supervisor_tick", &task) {
                        Ok(result) => {
                            match is_agent_turn_in_progress(&loop_paths, turn_id) {
                                Ok(true) => {}
                                Ok(false) => {
                                    let _ = crate::runtime_db::record_agent_event(
                                        &loop_paths,
                                        "turn/resultIgnored",
                                        Some(task.id),
                                        &task.title,
                                        &format!(
                                            "Late result for turn {} ignored because watchdog/recovery already closed it.",
                                            turn_id
                                        ),
                                        "{}",
                                    );
                                    return;
                                }
                                Err(err) => {
                                    eprintln!(
                                        "failed to verify turn {} state before applying result: {}",
                                        turn_id, err
                                    );
                                    return;
                                }
                            }
                            let mut checkpoint_summary = result
                                .checkpoint_summary
                                .clone()
                                .or_else(|| result.best_reply().map(ToString::to_string))
                                .unwrap_or_else(|| "Task run finished without summary.".to_string());
                            let output_text = result
                                .best_reply()
                                .map(ToString::to_string)
                                .unwrap_or_default();
                            let task_used_grosshirn = result.used_grosshirn;
                            let task_fell_back_to_kleinhirn = result.fell_back_to_kleinhirn;
                            let mut checkpoint_detail = if output_text.trim().is_empty() {
                                checkpoint_summary.clone()
                            } else {
                                output_text.clone()
                            };
                            let mut task_status =
                                result.task_status.as_deref().unwrap_or("done").to_string();
                            let inferred_next_mode =
                                infer_next_mode(&task, execution_mode, &task_status);
                            let mut next_mode = result
                                .next_mode
                                .as_deref()
                                .unwrap_or(inferred_next_mode)
                                .to_string();
                            if looks_like_invalid_structured_artifact(&checkpoint_summary)
                                || looks_like_invalid_structured_artifact(&output_text)
                            {
                                let artifact = if looks_like_invalid_structured_artifact(&output_text)
                                {
                                    output_text.trim().to_string()
                                } else {
                                    checkpoint_summary.trim().to_string()
                                };
                                checkpoint_summary =
                                    "Modell lieferte unvollstaendiges JSON; bounded Retry statt Schein-Erfolg."
                                        .to_string();
                                checkpoint_detail = if artifact.is_empty() {
                                    checkpoint_summary.clone()
                                } else {
                                    format!(
                                        "{checkpoint_summary}\n\nRohes Modellartefakt:\n{artifact}"
                                    )
                                };
                                task_status = "continue".to_string();
                                next_mode = "reprioritize".to_string();
                            }
                            if task_used_grosshirn {
                                let _ = refresh_task_grosshirn_boost(&loop_paths, task.id);
                                if let Some(usage) = result.model_usage.as_ref() {
                                    let _ = record_brain_usage_event(
                                        &loop_paths,
                                        Some(task.id),
                                        Some(turn_id),
                                        "grosshirn",
                                        "external grosshirn",
                                        result.model.as_deref().unwrap_or("unknown"),
                                        usage.input_tokens,
                                        usage.output_tokens,
                                        usage.total_tokens,
                                        usage.estimated_cost_usd,
                                        &format!(
                                            "Task #{} {} nutzte externes Grosshirn fuer einen bounded Schritt.",
                                            task.id, task.title
                                        ),
                                    );
                                }
                            }
                            if task_fell_back_to_kleinhirn {
                                let detail = "Grosshirn-Ausfuehrung fiel auf das lokale Kleinhirn zurueck; temporaerer Boost wurde fuer diesen Task geschlossen.";
                                let _ = release_task_grosshirn_boost(&loop_paths, task.id, detail);
                                checkpoint_summary = format!(
                                    "{} Grosshirn fiel aus; lokaler Kleinhirn-Fallback hat uebernommen.",
                                    checkpoint_summary
                                );
                                checkpoint_detail = format!("{checkpoint_detail}\n\n{detail}");
                            }
                            let loop_safety = load_loop_safety_policy(&loop_paths);
                            let self_preservation = load_self_preservation_state(&loop_paths);
                            if let Some(escalation) = assess_task_stuck_risk(
                                &task,
                                &loop_safety,
                                &self_preservation.current_stage,
                                &checkpoint_summary,
                                &checkpoint_detail,
                                &task_status,
                                &next_mode,
                            ) {
                                if escalation.spawn_self_preservation_task {
                                    let _ = crate::runtime_db::enqueue_internal_task(
                                        &loop_paths,
                                        Some(task.id),
                                        "self_preservation",
                                        "Selbsterhalt fuer laufende Aufgabe neu ausrichten",
                                        &format!(
                                            "Analysiere, warum Task #{} {} gerade nicht produktiv vorankommt, und entscheide ueber Ressourcenanforderung, Delegation, Kontext-Neuschnitt oder spaetere Wiederaufnahme.\n\n{}",
                                            task.id, task.title, escalation.checkpoint_detail
                                        ),
                                        900,
                                    );
                                }
                                task_status = escalation.task_status;
                                next_mode = escalation.next_mode;
                                checkpoint_summary = escalation.checkpoint_summary;
                                checkpoint_detail = escalation.checkpoint_detail;
                            }
                            let mut normalized_next_mode = next_mode.clone();
                            if should_force_local_kleinhirn_self_repair(
                                &task,
                                &task_status,
                                &next_mode,
                                &checkpoint_summary,
                                &checkpoint_detail,
                                local_kleinhirn_upgrade_available(&loop_paths),
                            ) {
                                let requested_target =
                                    extract_requested_local_kleinhirn_model(&task.detail);
                                match apply_targeted_kleinhirn_upgrade(
                                    &loop_paths,
                                    requested_target.as_deref(),
                                ) {
                                    Ok(outcome) => {
                                        checkpoint_summary = format!(
                                            "{} {}",
                                            checkpoint_summary, outcome.summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nKernel self-repair:\nDie lokale Kleinhirn-Aufwertung wurde nach wiederholtem Festhaengen im lokalen Review-Pfad direkt ausgefuehrt.\nChanged: {}\nRestarted: {}\nPrevious runtime: {}\nCurrent runtime: {}",
                                            checkpoint_detail,
                                            outcome.changed,
                                            outcome.restarted,
                                            outcome.previous_runtime_model.as_deref().unwrap_or("unknown"),
                                            outcome.current_runtime_model.as_deref().unwrap_or("unknown"),
                                        );
                                        let _ = record_memory(
                                            &loop_paths,
                                            "brain_runtime",
                                            "Kernel Self-Repair fuehrte lokale Kleinhirn-Aufwertung aus",
                                            &checkpoint_detail,
                                            "local_kleinhirn_self_repair_upgrade",
                                        );
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                    Err(err) => {
                                        let detail = err.to_string();
                                        let _ = register_loop_incident(
                                            &loop_paths,
                                            "local_kleinhirn_self_repair_failure",
                                            "high",
                                            "Kernel self-repair failed while attempting a local kleinhirn upgrade after repeated local-review stalls.",
                                            &detail,
                                            Some(task.id),
                                            Some(turn_id),
                                            false,
                                            false,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Kernel-Self-Repair fuer die lokale Kleinhirn-Aufwertung scheiterte.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nKernel self-repair error: {}",
                                            checkpoint_detail, detail
                                        );
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                }
                            }
                            if let Some(homepage_update) = result.homepage_update.as_ref() {
                                match apply_homepage_update(
                                    &loop_paths,
                                    &task,
                                    homepage_update,
                                    "agentic_task",
                                ) {
                                    Ok(policy) => {
                                        checkpoint_summary = format!(
                                            "{} Homepage direkt weitergebaut.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nHomepage mutated:\nTitle: {}\nHeadline: {}\nStage: {}",
                                            checkpoint_detail,
                                            policy.current_title,
                                            policy.current_headline,
                                            policy.stage
                                        );
                                    }
                                    Err(err) => {
                                        let detail = err.to_string();
                                        let _ = register_loop_incident(
                                            &loop_paths,
                                            "homepage_tool_failure",
                                            "high",
                                            "Homepage mutation failed inside bounded task run.",
                                            &detail,
                                            Some(task.id),
                                            Some(turn_id),
                                            false,
                                            false,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Homepage-Mutation scheiterte.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail =
                                            format!("{}\n\nHomepage tool error: {}", checkpoint_detail, detail);
                                    }
                                }
                            }
                            if let Some(exec_session_directive) =
                                result.exec_session_directive.as_ref()
                            {
                                match apply_exec_session_directive(
                                    &loop_paths,
                                    &task,
                                    turn_id,
                                    exec_session_directive,
                                ) {
                                    Ok((session_summary, session_detail)) => {
                                        checkpoint_summary =
                                            format!("{} {}", checkpoint_summary, session_summary);
                                        checkpoint_detail = format!(
                                            "{}\n\nExec session result:\n{}",
                                            checkpoint_detail, session_detail
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                    Err(err) => {
                                        let detail = err.to_string();
                                        let _ = register_loop_incident(
                                            &loop_paths,
                                            "command_exec_session_failure",
                                            "high",
                                            "Codex-backed exec session action failed inside task run.",
                                            &detail,
                                            Some(task.id),
                                            Some(turn_id),
                                            false,
                                            false,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Exec-Session-Aktion scheiterte.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nExec session error: {}",
                                            checkpoint_detail, detail
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                }
                            } else if let Some(exec_directive) = result.exec_directive.as_ref() {
                                match run_bounded_command(&loop_paths, &task, exec_directive) {
                                    Ok(exec_result) => {
                                        let justification = exec_directive
                                            .justification
                                            .as_deref()
                                            .unwrap_or("none");
                                        checkpoint_summary = format!(
                                            "{} Bounded command-exec ausgefuehrt: {:?}",
                                            checkpoint_summary, exec_directive.command
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBounded exec result:\nCommand: {:?}\nWorkdir: {}\nJustification: {}\nStatus: {}\nExit code: {:?}\nTimed out: {}\nSTDOUT:\n{}\n\nSTDERR:\n{}",
                                            checkpoint_detail,
                                            exec_directive.command,
                                            exec_result.cwd,
                                            justification,
                                            exec_result.status,
                                            exec_result.exit_code,
                                            exec_result.timed_out,
                                            exec_result.stdout,
                                            exec_result.stderr,
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                    Err(err) => {
                                        let detail = err.to_string();
                                        let _ = register_loop_incident(
                                            &loop_paths,
                                            "command_exec_failure",
                                            "high",
                                            "Bounded command-exec failed inside task run.",
                                            &detail,
                                            Some(task.id),
                                            Some(turn_id),
                                            false,
                                            false,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Bounded command-exec scheiterte.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail =
                                            format!("{}\n\nCommand exec error: {}", checkpoint_detail, detail);
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                }
                            } else if let Some(browser_directive) =
                                result.browser_directive.as_ref()
                            {
                                match run_browser_action(&loop_paths, &task, browser_directive) {
                                    Ok(browser_result) => {
                                        checkpoint_summary = format!(
                                            "{} Browser-Engine-Aktion ausgefuehrt: {}",
                                            checkpoint_summary, browser_directive.action
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBrowser action result:\nAction: {}\nURL: {}\nStatus: {}\nBrowser status: {}\nArtifact: {}\nSTDOUT:\n{}\n\nSTDERR:\n{}",
                                            checkpoint_detail,
                                            browser_directive.action,
                                            browser_directive.url.as_deref().unwrap_or(""),
                                            browser_result.status,
                                            browser_result.browser_status,
                                            browser_result.artifact_path.as_deref().unwrap_or("none"),
                                            browser_result.stdout,
                                            browser_result.stderr,
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                    Err(err) => {
                                        let detail = err.to_string();
                                        let _ = register_loop_incident(
                                            &loop_paths,
                                            "browser_engine_failure",
                                            "high",
                                            "Browser-engine action failed inside task run.",
                                            &detail,
                                            Some(task.id),
                                            Some(turn_id),
                                            false,
                                            false,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Browser-Engine-Aktion scheiterte.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBrowser engine error: {}",
                                            checkpoint_detail, detail
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                }
                            }
                            if let Some(context_directive) = result.context_directive.as_ref() {
                                let context_summary = format!(
                                    "Task #{} contextAction={}",
                                    task.id, context_directive.action
                                );
                                let context_detail = format!(
                                    "Task: {} ({})\nAction: {}\nConcern: {}\nHistory research query: {}\nTask checkpoint summary: {}",
                                    task.title,
                                    task.task_kind,
                                    context_directive.action,
                                    context_directive
                                        .concern
                                        .as_deref()
                                        .unwrap_or("none"),
                                    context_directive
                                        .history_research_query
                                        .as_deref()
                                        .unwrap_or("none"),
                                    checkpoint_summary
                                );
                                let _ = record_memory(
                                    &loop_paths,
                                    "context_governance",
                                    &context_summary,
                                    &context_detail,
                                    "agentic_context",
                                );

                                let needs_history_research = normalized_next_mode == "historical_research"
                                    || matches!(
                                        context_directive.action.as_str(),
                                        "expand_history" | "question_compaction"
                                    )
                                    || context_directive.history_research_query.is_some();
                                if needs_history_research
                                    && task.task_kind != "historical_research"
                                {
                                    let research_detail = format!(
                                        "Fuehre gezielte historische Nachladung fuer Task #{} {} durch.\n\nAgentischer Kontextgrund: {}\nKontext-Sorge: {}\nGezielte Query: {}\n\nNutze rohe Historie, Checkpoints, BIOS-Dialog und Memory selektiv. Ziel ist nicht blindes Aufblasen, sondern ein praeziser Nachlade-Schritt fuer den naechsten bounded Run.",
                                        task.id,
                                        task.title,
                                        context_directive.action,
                                        context_directive
                                            .concern
                                            .as_deref()
                                            .unwrap_or("none"),
                                        context_directive
                                            .history_research_query
                                            .as_deref()
                                            .unwrap_or("not supplied"),
                                    );
                                    let research_priority = if task.source_channel == "terminal"
                                        || task.source_channel == "bios"
                                        || task.trust_level == "owner"
                                        || task.trust_level == "system"
                                    {
                                        880
                                    } else {
                                        720
                                    };
                                    let _ = crate::runtime_db::enqueue_internal_task(
                                        &loop_paths,
                                        Some(task.id),
                                        "historical_research",
                                        &format!(
                                            "Historische Nachladung fuer Task {} vorbereiten",
                                            task.id
                                        ),
                                        &research_detail,
                                        research_priority,
                                    );
                                    if normalized_next_mode == "historical_research" {
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                }
                            }
                            if let Some(system_census_action) =
                                result.system_census_action.as_deref()
                            {
                                if system_census_action.eq_ignore_ascii_case("run") {
                                    match run_system_census(&loop_paths) {
                                        Ok(census) => {
                                            let model_policy = load_model_policy(&loop_paths);
                                            let _ =
                                                sync_resources_from_census(&loop_paths, &census);
                                            let _ = sync_model_resources(
                                                &loop_paths,
                                                &model_policy,
                                                &census,
                                            );
                                            checkpoint_summary = format!(
                                                "{} System-Census inklusive mistralrs tune ausgefuehrt.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nSystem census result:\nCPU threads: {}\nRAM: {}\nGPU count: {}\nTotal VRAM: {}\nLargest GPU: {}\nSelected kleinhirn now: {}",
                                                checkpoint_detail,
                                                census
                                                    .cpu_threads
                                                    .map(|value| value.to_string())
                                                    .unwrap_or_else(|| "unknown".to_string()),
                                                census
                                                    .total_memory_gb
                                                    .map(|value| format!("{value} GiB"))
                                                    .unwrap_or_else(|| "unknown".to_string()),
                                                census
                                                    .gpu_count
                                                    .map(|value| value.to_string())
                                                    .unwrap_or_else(|| "unknown".to_string()),
                                                census
                                                    .total_gpu_memory_gb
                                                    .map(|value| format!("{value} GiB"))
                                                    .unwrap_or_else(|| "unknown".to_string()),
                                                census
                                                    .max_single_gpu_memory_gb
                                                    .map(|value| format!("{value} GiB"))
                                                    .unwrap_or_else(|| "unknown".to_string()),
                                                crate::contracts::describe_kleinhirn_selection(
                                                    &model_policy,
                                                    &census,
                                                ),
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                        Err(err) => {
                                            let detail = err.to_string();
                                            let _ = register_loop_incident(
                                                &loop_paths,
                                                "system_census_failure",
                                                "high",
                                                "System census or mistralrs tune failed inside task run.",
                                                &detail,
                                                Some(task.id),
                                                Some(turn_id),
                                                false,
                                                false,
                                            );
                                            checkpoint_summary = format!(
                                                "{} System-Census scheiterte.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nSystem census error: {}",
                                                checkpoint_detail, detail
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                    }
                                }
                            }
                            if let Some(brain_directive) = result.brain_directive.as_ref() {
                                let brain_target_task_id = task.parent_task_id.unwrap_or(task.id);
                                let brain_target_task = load_task_by_id(&loop_paths, brain_target_task_id)
                                    .ok()
                                    .flatten();
                                let brain_target_title = brain_target_task
                                    .as_ref()
                                    .map(|value| value.title.as_str())
                                    .unwrap_or(task.title.as_str());
                                if brain_directive
                                    .action
                                    .eq_ignore_ascii_case("activate_temporary_grosshirn")
                                {
                                    if crate::runtime_db::grosshirn_boost_available(&loop_paths) {
                                        let _ = arm_task_grosshirn_boost(
                                            &loop_paths,
                                            brain_target_task_id,
                                            brain_target_title,
                                            brain_directive.note.as_deref().unwrap_or(
                                                "Agentischer Entscheid: fuer diese Aufgabe ist temporaer Grosshirn gerechtfertigt.",
                                            ),
                                            1120,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Temporärer Grosshirn-Boost fuer Task #{} wurde aktiviert.",
                                            checkpoint_summary, brain_target_task_id
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBrain action:\nAction: {}\nTarget model hint: {}\nNote: {}\nApplied to task: #{} {}",
                                            checkpoint_detail,
                                            brain_directive.action,
                                            brain_directive.target_model.as_deref().unwrap_or("not supplied"),
                                            brain_directive.note.as_deref().unwrap_or("none"),
                                            brain_target_task_id,
                                            brain_target_title,
                                        );
                                        if task.task_kind == "grosshirn_activation"
                                            && (task_used_grosshirn || task_fell_back_to_kleinhirn)
                                        {
                                            task_status = "done".to_string();
                                        } else if task_status != "blocked" && task_status != "done" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "reprioritize".to_string();
                                    } else {
                                        checkpoint_summary = format!(
                                            "{} Temporärer Grosshirn-Boost wurde angefragt, ist aber noch nicht verfuegbar.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBrain action:\nAction: {}\nTarget model hint: {}\nNote: {}\nResult: Grosshirn ist noch nicht verfuegbar oder nicht freigeschaltet.",
                                            checkpoint_detail,
                                            brain_directive.action,
                                            brain_directive.target_model.as_deref().unwrap_or("not supplied"),
                                            brain_directive.note.as_deref().unwrap_or("none"),
                                        );
                                        if task_status != "blocked" && task_status != "done" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "request_resources".to_string();
                                    }
                                } else if brain_directive
                                    .action
                                    .eq_ignore_ascii_case("release_temporary_grosshirn")
                                {
                                    let _ = release_task_grosshirn_boost(
                                        &loop_paths,
                                        brain_target_task_id,
                                        brain_directive.note.as_deref().unwrap_or(
                                            "Agentischer Entscheid: externer Grosshirn-Boost ist fuer diese Aufgabe nicht mehr noetig.",
                                        ),
                                    );
                                    checkpoint_summary = format!(
                                        "{} Temporärer Grosshirn-Boost fuer Task #{} wurde freigegeben.",
                                        checkpoint_summary, brain_target_task_id
                                    );
                                    checkpoint_detail = format!(
                                        "{}\n\nBrain action:\nAction: {}\nTarget model hint: {}\nNote: {}\nReleased from task: #{} {}",
                                        checkpoint_detail,
                                        brain_directive.action,
                                        brain_directive.target_model.as_deref().unwrap_or("not supplied"),
                                        brain_directive.note.as_deref().unwrap_or("none"),
                                        brain_target_task_id,
                                        brain_target_title,
                                    );
                                } else if brain_directive
                                    .action
                                    .eq_ignore_ascii_case("upgrade_local_browser_vision_kleinhirn")
                                {
                                    match apply_recommended_browser_vision_kleinhirn_upgrade(
                                        &loop_paths,
                                    ) {
                                        Ok(outcome) => {
                                            checkpoint_summary = format!(
                                                "{} {}",
                                                checkpoint_summary, outcome.summary
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nLocal browser-vision kleinhirn runtime action:\nAction: {}\nTarget model hint: {}\nNote: {}\nChanged: {}\nRestarted: {}\nPrevious runtime: {}\nCurrent runtime: {}",
                                                checkpoint_detail,
                                                brain_directive.action,
                                                brain_directive.target_model.as_deref().unwrap_or("not supplied"),
                                                brain_directive.note.as_deref().unwrap_or("none"),
                                                outcome.changed,
                                                outcome.restarted,
                                                outcome.previous_runtime_model.as_deref().unwrap_or("unknown"),
                                                outcome.current_runtime_model.as_deref().unwrap_or("unknown"),
                                            );
                                            let _ = record_memory(
                                                &loop_paths,
                                                "brain_runtime",
                                                "Vision-faehiges lokales Kleinhirn umgestellt",
                                                &checkpoint_detail,
                                                "local_browser_vision_kleinhirn_upgrade",
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                        Err(err) => {
                                            let detail = err.to_string();
                                            let _ = register_loop_incident(
                                                &loop_paths,
                                                "local_browser_vision_kleinhirn_upgrade_failure",
                                                "high",
                                                "The loop attempted to upgrade the local browser-vision kleinhirn runtime and failed.",
                                                &detail,
                                                Some(task.id),
                                                Some(turn_id),
                                                false,
                                                false,
                                            );
                                            checkpoint_summary = format!(
                                                "{} Browser-/Vision-Kleinhirn-Aufwertung scheiterte.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nLocal browser-vision kleinhirn runtime action failed:\nAction: {}\nTarget model hint: {}\nNote: {}\nError: {}",
                                                checkpoint_detail,
                                                brain_directive.action,
                                                brain_directive.target_model.as_deref().unwrap_or("not supplied"),
                                                brain_directive.note.as_deref().unwrap_or("none"),
                                                detail,
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                    }
                                } else if brain_directive.action.eq_ignore_ascii_case("upgrade_local_kleinhirn") {
                                    let requested_target = brain_directive
                                        .target_model
                                        .clone()
                                        .or_else(|| extract_requested_local_kleinhirn_model(&task.detail));
                                    match apply_targeted_kleinhirn_upgrade(
                                        &loop_paths,
                                        requested_target.as_deref(),
                                    ) {
                                        Ok(outcome) => {
                                            checkpoint_summary = format!(
                                                "{} {}",
                                                checkpoint_summary, outcome.summary
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nLocal kleinhirn runtime action:\nAction: {}\nTarget model hint: {}\nNote: {}\nChanged: {}\nRestarted: {}\nPrevious runtime: {}\nCurrent runtime: {}",
                                                checkpoint_detail,
                                                brain_directive.action,
                                                requested_target.as_deref().unwrap_or("not supplied"),
                                                brain_directive.note.as_deref().unwrap_or("none"),
                                                outcome.changed,
                                                outcome.restarted,
                                                outcome.previous_runtime_model.as_deref().unwrap_or("unknown"),
                                                outcome.current_runtime_model.as_deref().unwrap_or("unknown"),
                                            );
                                            let _ = record_memory(
                                                &loop_paths,
                                                "brain_runtime",
                                                "Lokales Kleinhirn umgestellt",
                                                &checkpoint_detail,
                                                "local_kleinhirn_upgrade",
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                        Err(err) => {
                                            let detail = err.to_string();
                                            let _ = register_loop_incident(
                                                &loop_paths,
                                                "local_kleinhirn_upgrade_failure",
                                                "high",
                                                "The loop attempted to upgrade the local kleinhirn runtime and failed.",
                                                &detail,
                                                Some(task.id),
                                                Some(turn_id),
                                                false,
                                                false,
                                            );
                                            checkpoint_summary = format!(
                                                "{} Lokale Kleinhirn-Aufwertung scheiterte.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nLocal kleinhirn runtime action failed:\nAction: {}\nTarget model hint: {}\nNote: {}\nError: {}",
                                                checkpoint_detail,
                                                brain_directive.action,
                                                brain_directive.target_model.as_deref().unwrap_or("not supplied"),
                                                brain_directive.note.as_deref().unwrap_or("none"),
                                                detail,
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                    }
                                }
                            }
                            if normalized_next_mode == "request_resources"
                                && task.task_kind != "grosshirn_procurement"
                            {
                                let trust = load_owner_trust(&loop_paths).unwrap_or_default();
                                let policy = load_model_policy(&loop_paths);
                                let census = load_census(&loop_paths);
                                let local_kleinhirn_available =
                                    crate::brain_runtime::local_kleinhirn_upgrade_available(
                                        &loop_paths,
                                    );
                                let grosshirn_available =
                                    crate::runtime_db::grosshirn_boost_available(&loop_paths);
                                let grosshirn_configured =
                                    grosshirn_runtime_configured(&loop_paths);
                                let grosshirn_candidates = if policy.grosshirn_candidates.is_empty() {
                                    "none configured".to_string()
                                } else {
                                    policy
                                        .grosshirn_candidates
                                        .iter()
                                        .map(|candidate| format!("{} ({})", candidate.official_label, candidate.model_id))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                };
                                if task.task_kind == "model_or_resource"
                                    && local_kleinhirn_available
                                {
                                    let local_upgrade_detail = format!(
                                        "Die Aufgabe #{task_id} ({title}) hat request_resources angefordert, obwohl lokal bereits ein staerkeres Kleinhirn verfuegbar ist.\n\nGrund:\n{checkpoint_detail}\n\nAktueller Brain-Access-Modus: {brain_mode}\nAktuell lokal empfohlenes Kleinhirn: {kleinhirn}\n\nNaechster bounded Schritt:\n1. Bleibe im lokalen Kleinhirn-Entscheidpfad.\n2. Pruefe oder fuehre die empfohlene lokale Aufwertung aus.\n3. Erst wenn die lokale Aufwertung ehrlich scheitert oder nachweislich nicht reicht, darfst du auf Grosshirn- oder Ressourcen-Procurement erweitern.",
                                        task_id = task.id,
                                        title = task.title,
                                        checkpoint_detail = checkpoint_detail,
                                        brain_mode = trust.brain_access_mode,
                                        kleinhirn = crate::contracts::describe_kleinhirn_selection(
                                            &policy,
                                            &census,
                                        ),
                                    );
                                    let _ = crate::runtime_db::enqueue_internal_task(
                                        &loop_paths,
                                        Some(task.id),
                                        "model_or_resource",
                                        "Lokales Kleinhirn aktiv aufwerten oder begruendet ablehnen",
                                        &local_upgrade_detail,
                                        875,
                                    );
                                    checkpoint_summary = format!(
                                        "{} Lokale Kleinhirn-Folgearbeit wurde eingereiht.",
                                        checkpoint_summary
                                    );
                                    task_status = "continue".to_string();
                                    normalized_next_mode = "reprioritize".to_string();
                                } else if grosshirn_available {
                                    let boost_note =
                                        "Grosshirn ist bereits freigeschaltet und konfiguriert; derselbe Task wird jetzt darueber statt ueber neue Beschaffung weitergefuehrt.";
                                    match arm_task_grosshirn_boost(
                                        &loop_paths,
                                        task.id,
                                        &task.title,
                                        boost_note,
                                        1120,
                                    ) {
                                        Ok(_) => {
                                            checkpoint_summary = format!(
                                                "{} Vorhandenes Grosshirn wird fuer denselben Task statt neuer Beschaffung benutzt.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{checkpoint_detail}\n\nBereits verfuegbares Grosshirn wird direkt auf Task #{task_id} {title} gelenkt, statt weitere Beschaffungsschleifen zu erzeugen.\nBrain-Access-Modus: {brain_mode}\nAktuell lokal empfohlenes Kleinhirn: {kleinhirn}\nGrosshirn-Kandidaten: {grosshirn_candidates}\nBoost-Notiz: {boost_note}",
                                                checkpoint_detail = checkpoint_detail,
                                                task_id = task.id,
                                                title = task.title,
                                                brain_mode = trust.brain_access_mode,
                                                kleinhirn = crate::contracts::describe_kleinhirn_selection(
                                                    &policy,
                                                    &census,
                                                ),
                                                grosshirn_candidates = grosshirn_candidates,
                                                boost_note = boost_note,
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                        Err(err) => {
                                            checkpoint_summary = format!(
                                                "{} Vorhandenes Grosshirn sollte statt Beschaffung benutzt werden, aber der Boost liess sich nicht setzen.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{checkpoint_detail}\n\nGrosshirn-Boost konnte fuer Task #{task_id} {title} nicht gesetzt werden: {err}",
                                                checkpoint_detail = checkpoint_detail,
                                                task_id = task.id,
                                                title = task.title,
                                                err = err,
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                    }
                                } else if trust.brain_access_mode != "kleinhirn_plus_grosshirn"
                                    && grosshirn_configured
                                {
                                    let activation_detail = format!(
                                        "Die Aufgabe #{task_id} ({title}) hat request_resources angefordert, aber in der Runtime liegt bereits eine Grosshirn-Konfiguration.\n\nGrund:\n{checkpoint_detail}\n\nAktueller Brain-Access-Modus: {brain_mode}\nAktuell lokal empfohlenes Kleinhirn: {kleinhirn}\nMoegliche Grosshirn-Kandidaten: {grosshirn_candidates}\n\nNaechster bounded Schritt:\n1. Aktiviere zuerst den bereits vorbereiteten Grosshirn-Zugang sauber im Brain-Access.\n2. Nutze danach denselben Infinity Loop mit Grosshirn plus lokalem Fallback.\n3. Erzeuge keine neue Beschaffungsschleife fuer bereits vorhandene Credentials.",
                                        task_id = task.id,
                                        title = task.title,
                                        checkpoint_detail = checkpoint_detail,
                                        brain_mode = trust.brain_access_mode,
                                        kleinhirn = crate::contracts::describe_kleinhirn_selection(
                                            &policy,
                                            &census,
                                        ),
                                        grosshirn_candidates = grosshirn_candidates,
                                    );
                                    let _ = crate::runtime_db::enqueue_internal_task(
                                        &loop_paths,
                                        Some(task.id),
                                        "grosshirn_activation",
                                        "Grosshirn-Modus aktivieren und verifizieren",
                                        &activation_detail,
                                        865,
                                    );
                                    checkpoint_summary = format!(
                                        "{} Grosshirn-Aktivierung statt neuer Beschaffung wurde eingereiht.",
                                        checkpoint_summary
                                    );
                                    if task_status != "blocked" {
                                        task_status = "continue".to_string();
                                    }
                                    normalized_next_mode = "reprioritize".to_string();
                                } else {
                                    let procurement_detail = format!(
                                        "Die Aufgabe #{task_id} ({title}) hat request_resources angefordert.\n\nGrund:\n{checkpoint_detail}\n\nAktueller Brain-Access-Modus: {brain_mode}\nAktuell lokal empfohlenes Kleinhirn: {kleinhirn}\nMoegliche Grosshirn-Kandidaten: {grosshirn_candidates}\n\nNaechster bounded Schritt:\n1. Pruefe zuerst, ob ein besseres lokales Kleinhirn auf demselben Host moeglich ist.\n2. Wenn lokale Aufwertung nicht reicht, bereite eine klare Owner-Anfrage fuer zusaetzliche Ressourcen oder Grosshirn-Zugang vor.\n3. Wenn der Owner Grosshirn-Zugang bereits erlaubt hat und Credentials konfiguriert sind, nutze danach denselben Infinity Loop mit Grosshirn plus lokalem Fallback.",
                                        task_id = task.id,
                                        title = task.title,
                                        checkpoint_detail = checkpoint_detail,
                                        brain_mode = trust.brain_access_mode,
                                        kleinhirn = crate::contracts::describe_kleinhirn_selection(
                                            &policy,
                                            &census,
                                        ),
                                        grosshirn_candidates = grosshirn_candidates,
                                    );
                                    let _ = crate::runtime_db::enqueue_internal_task(
                                        &loop_paths,
                                        Some(task.id),
                                        "grosshirn_procurement",
                                        "Ressourcen- und Grosshirn-Freigabe vorbereiten",
                                        &procurement_detail,
                                        860,
                                    );
                                    checkpoint_summary = format!(
                                        "{} Ressourcen-/Grosshirn-Folgearbeit wurde eingereiht.",
                                        checkpoint_summary
                                    );
                                    if task_status != "blocked" {
                                        task_status = "continue".to_string();
                                    }
                                    normalized_next_mode = "reprioritize".to_string();
                                }
                            }
                            if let Some(followup_task) = result.followup_task.as_ref() {
                                match crate::runtime_db::enqueue_internal_task(
                                    &loop_paths,
                                    Some(task.id),
                                    &followup_task.task_kind,
                                    &followup_task.title,
                                    &followup_task.detail,
                                    followup_task.priority_score,
                                ) {
                                    Ok(queued_task) => {
                                        checkpoint_summary = format!(
                                            "{} Neue Folgeaufgabe #{} wurde eingereiht.",
                                            checkpoint_summary, queued_task.id
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nFollow-up task queued:\nTask kind: {}\nTitle: {}\nPriority: {}\nDetail: {}",
                                            checkpoint_detail,
                                            followup_task.task_kind,
                                            followup_task.title,
                                            followup_task.priority_score,
                                            followup_task.detail
                                        );
                                        if task_status != "blocked" {
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                    }
                                    Err(err) => {
                                        let detail = err.to_string();
                                        let _ = register_loop_incident(
                                            &loop_paths,
                                            "followup_task_enqueue_failure",
                                            "medium",
                                            "The loop tried to enqueue a self-generated follow-up task and failed.",
                                            &detail,
                                            Some(task.id),
                                            Some(turn_id),
                                            false,
                                            false,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Folgeaufgabe konnte nicht eingereiht werden.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nFollow-up task enqueue error: {}",
                                            checkpoint_detail, detail
                                        );
                                    }
                                }
                            }
                            if let Err(err) = persist_learning_updates(
                                &loop_paths,
                                &task,
                                turn_id,
                                &task_status,
                                &result.learning_entries,
                                &mut checkpoint_summary,
                                &mut checkpoint_detail,
                            ) {
                                let detail = err.to_string();
                                let _ = register_loop_incident(
                                    &loop_paths,
                                    "learning_path_persist_failure",
                                    "medium",
                                    "The loop tried to persist structured learnings and failed.",
                                    &detail,
                                    Some(task.id),
                                    Some(turn_id),
                                    false,
                                    false,
                                );
                                checkpoint_summary = format!(
                                    "{} Lernpfad konnte nicht aktualisiert werden.",
                                    checkpoint_summary
                                );
                                checkpoint_detail = format!(
                                    "{}\n\nLearning path error: {}",
                                    checkpoint_detail, detail
                                );
                            }
                            if let Err(err) = persist_proactive_contact_updates(
                                &loop_paths,
                                &task,
                                turn_id,
                                &task_status,
                                result.proactive_contact_draft.as_ref(),
                                result.proactive_contact_validation.as_ref(),
                                &mut checkpoint_summary,
                                &mut checkpoint_detail,
                            ) {
                                let detail = err.to_string();
                                let _ = register_loop_incident(
                                    &loop_paths,
                                    "proactive_contact_persist_failure",
                                    "medium",
                                    "The loop tried to persist a proactive people suggestion or validation and failed.",
                                    &detail,
                                    Some(task.id),
                                    Some(turn_id),
                                    false,
                                    false,
                                );
                                checkpoint_summary = format!(
                                    "{} Personenpfad-Proaktivitaet konnte nicht aktualisiert werden.",
                                    checkpoint_summary
                                );
                                checkpoint_detail = format!(
                                    "{}\n\nProactive contact error: {}",
                                    checkpoint_detail, detail
                                );
                            }
                            if task.task_kind == "proactive_contact_review"
                                && task_status != "continue"
                                && normalized_next_mode == "review"
                            {
                                normalized_next_mode = "reprioritize".to_string();
                            }

                            if matches!(task.task_kind.as_str(), "worker_review" | "self_review") {
                                if let Err(err) = complete_review_task(
                                    &loop_paths,
                                    &task,
                                    &task_status,
                                    &checkpoint_summary,
                                    &checkpoint_detail,
                                    Some(&output_text),
                                ) {
                                    eprintln!("failed to resolve review task {task_id}: {err}");
                                }
                            } else if task.task_kind == "proactive_contact_review" {
                                match task_status.as_str() {
                                    "continue" => {
                                        if let Err(err) = requeue_task(
                                            &loop_paths,
                                            task_id,
                                            &checkpoint_summary,
                                            &checkpoint_detail,
                                            Some(&output_text),
                                        ) {
                                            eprintln!(
                                                "failed to requeue proactive review task {task_id}: {err}"
                                            );
                                        }
                                    }
                                    "blocked" => {
                                        let _ = release_task_grosshirn_boost(
                                            &loop_paths,
                                            task_id,
                                            "Proaktive Kontaktvalidierung wurde blockiert; temporaerer Grosshirn-Boost faellt wieder auf Kleinhirn zurueck.",
                                        );
                                        if let Err(err) = block_task(
                                            &loop_paths,
                                            task_id,
                                            &checkpoint_summary,
                                            &checkpoint_detail,
                                            Some(&output_text),
                                        ) {
                                            eprintln!(
                                                "failed to block proactive review task {task_id}: {err}"
                                            );
                                        }
                                    }
                                    _ => {
                                        let _ = release_task_grosshirn_boost(
                                            &loop_paths,
                                            task_id,
                                            "Proaktive Kontaktvalidierung abgeschlossen; temporaerer Grosshirn-Boost faellt wieder auf Kleinhirn zurueck.",
                                        );
                                        if let Err(err) = complete_task(
                                            &loop_paths,
                                            task_id,
                                            &checkpoint_summary,
                                            &checkpoint_detail,
                                            Some(&output_text),
                                        ) {
                                            eprintln!(
                                                "failed to complete proactive review task {task_id}: {err}"
                                            );
                                        }
                                    }
                                }
                            } else if normalized_next_mode == "review"
                                && task_status == "continue"
                            {
                                if let Err(err) = emit_self_review_task(
                                    &loop_paths,
                                    &task,
                                    &checkpoint_summary,
                                    &checkpoint_detail,
                                    Some(&output_text),
                                ) {
                                    eprintln!("failed to emit self-review task for {task_id}: {err}");
                                } else {
                                    task_status = "done".to_string();
                                }
                            } else if next_mode == "delegate" {
                                let contract = result.delegate_contract.clone().unwrap_or_else(|| {
                                    crate::agentic::DelegationContract {
                                        worker_kind: "specialist_worker".to_string(),
                                        contract_title: format!("Delegierter Schritt fuer Task {}", task.id),
                                        contract_detail: checkpoint_detail.clone(),
                                        request_note: "Fuehre bounded Umsetzung aus und melde fuer Review zurueck.".to_string(),
                                    }
                                });
                                let _ = set_agent_mode(
                                    &loop_paths,
                                    "delegate",
                                    Some(task.id),
                                    &task.title,
                                    &format!(
                                        "Task {} wird an Worker {} delegiert.",
                                        task.id, contract.worker_kind
                                    ),
                                );
                                if let Err(err) = delegate_task_to_worker(
                                    &loop_paths,
                                    &task,
                                    &contract.worker_kind,
                                    &contract.contract_title,
                                    &contract.contract_detail,
                                    &contract.request_note,
                                    Some(&output_text),
                                ) {
                                    eprintln!("failed to delegate task {task_id}: {err}");
                                }
                            } else {
                                match task_status.as_str() {
                                    "continue" => {
                                        if let Err(err) = requeue_task(
                                            &loop_paths,
                                            task_id,
                                            &checkpoint_summary,
                                            &checkpoint_detail,
                                            Some(&output_text),
                                        ) {
                                            eprintln!("failed to requeue task {task_id}: {err}");
                                        }
                                    }
                                    "blocked" => {
                                        let _ = release_task_grosshirn_boost(
                                            &loop_paths,
                                            task_id,
                                            "Task wurde im bounded Schritt blockiert; temporaerer Grosshirn-Boost faellt wieder auf Kleinhirn zurueck.",
                                        );
                                        if let Err(err) = block_task(
                                            &loop_paths,
                                            task_id,
                                            &checkpoint_summary,
                                            &checkpoint_detail,
                                            Some(&output_text),
                                        ) {
                                            eprintln!("failed to block task {task_id}: {err}");
                                        }
                                    }
                                    _ => {
                                        if let Err(err) = emit_self_review_task(
                                            &loop_paths,
                                            &task,
                                            &checkpoint_summary,
                                            &checkpoint_detail,
                                            Some(&output_text),
                                        ) {
                                            eprintln!("failed to emit self-review task for {task_id}: {err}");
                                        }
                                    }
                                }
                            }

                            let effective_next_mode = if normalized_next_mode == "delegate" {
                                "await_review"
                            } else {
                                normalized_next_mode.as_str()
                            };
                            let turn_status = if matches!(
                                task.task_kind.as_str(),
                                "worker_review" | "proactive_contact_review"
                            ) {
                                match task_status.as_str() {
                                    "continue" => "review_continue",
                                    "blocked" => "review_blocked",
                                    _ => "review_completed",
                                }
                            } else if next_mode == "delegate" {
                                "delegated"
                            } else {
                                match task_status.as_str() {
                                    "continue" => "checkpointed",
                                    "blocked" => "blocked",
                                    _ => "completed",
                                }
                            };
                            if let Err(err) = complete_agent_turn(
                                &loop_paths,
                                turn_id,
                                turn_status,
                                effective_next_mode,
                                &checkpoint_summary,
                                &checkpoint_detail,
                            ) {
                                eprintln!("failed to complete turn {turn_id}: {err}");
                            }
                            let _ = set_agent_mode(
                                &loop_paths,
                                effective_next_mode,
                                if normalized_next_mode == "delegate"
                                    || task.task_kind == "worker_review"
                                {
                                    Some(task.id)
                                } else {
                                    None
                                },
                                &task.title,
                                &format!("Task {} finished one unified mode cycle.", task.id),
                            );

                            if task.source_channel == "bios" && !output_text.trim().is_empty() {
                                let _ = record_bios_dialogue(
                                    &loop_paths,
                                    "cto-agent",
                                    &output_text,
                                    task_used_grosshirn,
                                );
                            } else if is_owner_facing_task(&task)
                                && !output_text.trim().is_empty()
                                && !matches!(task.task_kind.as_str(), "self_review" | "worker_review")
                            {
                                let _ = record_bios_dialogue(
                                    &loop_paths,
                                    "cto-agent",
                                    &output_text,
                                    task_used_grosshirn,
                                );
                            }
                            if task.task_kind == "homepage_bridge" && !checkpoint_summary.trim().is_empty() {
                                let _ = record_homepage_revision(
                                    &loop_paths,
                                    "task_loop",
                                    &load_homepage_policy(&loop_paths),
                                    &format!("Task #{} processed: {}", task.id, checkpoint_summary),
                                );
                            }
                            let _ = record_memory(
                                &loop_paths,
                                "task_loop",
                                &format!("Task #{} {}", task.id, task.title),
                                &checkpoint_detail,
                                "task_loop",
                            );
                            if result.status != "ok" && result.status != "idle" {
                                let _ = register_loop_incident(
                                    &loop_paths,
                                    infer_result_incident_key(&result),
                                    "high",
                                    &format!(
                                        "Agentic run for task {} returned status {}.",
                                        task.id, result.status
                                    ),
                                    &checkpoint_detail,
                                    Some(task.id),
                                    Some(turn_id),
                                    false,
                                    !matches!(task.task_kind.as_str(), "self_preservation" | "recovery"),
                                );
                            }
                            if result.status != "ok" && result.status != "idle" {
                                eprintln!("bounded task loop {}: {}", task.id, result.status_note());
                            }
                        }
                        Err(err) => {
                            match is_agent_turn_in_progress(&loop_paths, turn_id) {
                                Ok(true) => {}
                                Ok(false) => {
                                    let _ = crate::runtime_db::record_agent_event(
                                        &loop_paths,
                                        "turn/errorIgnored",
                                        Some(task.id),
                                        &task.title,
                                        &format!(
                                            "Late error for turn {} ignored because watchdog/recovery already closed it.",
                                            turn_id
                                        ),
                                        "{}",
                                    );
                                    return;
                                }
                                Err(check_err) => {
                                    eprintln!(
                                        "failed to verify turn {} state before applying error: {}",
                                        turn_id, check_err
                                    );
                                    return;
                                }
                            }
                            let summary = format!("Task {} failed to execute.", task.id);
                            let detail = err.to_string();
                            if detail.contains("without textual content")
                                || detail.contains("empty textual content")
                                || detail.contains("without textual assistant output")
                            {
                                let retry_summary = format!(
                                    "Task {} lieferte leeren Modelltext und wird erneut eingeordnet.",
                                    task.id
                                );
                                let retry_detail = format!(
                                    "{}\n\nDer bounded Schritt wurde nicht als echter inhaltlicher Block gewertet, sondern zur erneuten Repriorisierung in die Queue gelegt.",
                                    detail
                                );
                                let _ = requeue_task(
                                    &loop_paths,
                                    task.id,
                                    &retry_summary,
                                    &retry_detail,
                                    None,
                                );
                                let _ = complete_agent_turn(
                                    &loop_paths,
                                    turn_id,
                                    "completed",
                                    "reprioritize",
                                    &retry_summary,
                                    &retry_detail,
                                );
                                eprintln!(
                                    "bounded task loop produced empty model text for task {}: {err}",
                                    task.id
                                );
                                return;
                            }
                            let _ = register_loop_incident(
                                &loop_paths,
                                "bounded_turn_failure",
                                "critical",
                                &summary,
                                &detail,
                                Some(task.id),
                                Some(turn_id),
                                false,
                                true,
                            );
                            let _ = block_task(&loop_paths, task.id, &summary, &detail, None);
                            let _ = complete_agent_turn(
                                &loop_paths,
                                turn_id,
                                "failed",
                                "blocked",
                                &summary,
                                &detail,
                            );
                            eprintln!("bounded task loop failed for task {}: {err}", task.id);
                        }
                    }
                });
                agentic_task = Some(RunningAgenticTask {
                    turn_id,
                    task_id,
                    task_title,
                    started_at: Instant::now(),
                    handle,
                });
            }

            let bios = load_bios(&paths);
            let organigram = load_organigram(&paths);
            let active_turn = load_active_agent_turn(&paths).ok().flatten();
            let latest_completed_turn = load_latest_completed_agent_turn(&paths).ok().flatten();
            let mut loop_health = "healthy".to_string();
            let mut unhealthy_reason = None;
            let mut thread_lifecycle = "running".to_string();
            let mut thread_note =
                "Main infinity thread is alive and cycling through explicit modes.".to_string();
            if let Some(running) = agentic_task.as_ref() {
                let age = running.started_at.elapsed().as_secs();
                if age > stale_after_secs {
                    loop_health = "stalled".to_string();
                    unhealthy_reason = Some(format!(
                        "active turn {} has been running for {}s",
                        running.turn_id, age
                    ));
                    thread_lifecycle = "stalled".to_string();
                    thread_note = format!(
                        "Turn {} for task {} is older than {}s and awaits watchdog recovery.",
                        running.turn_id, running.task_id, stale_after_secs
                    );
                    let _ = register_loop_incident(
                        &paths,
                        "active_turn_stall",
                        "critical",
                        "A bounded turn exceeded the stall threshold.",
                        &thread_note,
                        Some(running.task_id),
                        Some(running.turn_id),
                        true,
                        false,
                    );
                } else {
                    thread_note = format!(
                        "Turn {} is active for task {} in bounded execution.",
                        running.turn_id, running.task_id
                    );
                }
            } else if let Ok(focus) = load_focus_state(&paths) {
                if focus.mode == "idle" && focus.queue_depth == 0 {
                    thread_lifecycle = "idle".to_string();
                    thread_note =
                        "Main infinity thread is healthy and temporarily between external work and self-directed exploration."
                            .to_string();
                } else if focus.mode == "recovery" {
                    thread_lifecycle = "recovering".to_string();
                    thread_note =
                        "Main infinity thread is processing a hard-reset or unhealthy-restart recovery cycle."
                            .to_string();
                } else if focus.mode == "self_preservation" {
                    thread_lifecycle = "self_preserving".to_string();
                    thread_note =
                        "Main infinity thread is actively protecting its own continuity."
                            .to_string();
                } else {
                    thread_note = format!(
                        "Main infinity thread is healthy in mode {} with queue depth {}.",
                        focus.mode, focus.queue_depth
                    );
                }
            }
            let state = AgentState {
                agent_name: bios.agent_identity.agent_name.clone(),
                bios_frozen: bios.frozen,
                owner_known: !organigram.owner.name.is_empty(),
                reports_to_known: !organigram.reports_to.is_empty(),
                supervisor_status: "running".to_string(),
                last_heartbeat_at: Some(now_iso()),
                uptime_seconds: started_at.elapsed().as_secs(),
                active_turn_id: active_turn.as_ref().map(|turn| turn.id),
                active_turn_started_at: active_turn.as_ref().map(|turn| turn.created_at.clone()),
                last_turn_completed_at: latest_completed_turn
                    .as_ref()
                    .and_then(|turn| turn.completed_at.clone()),
                loop_health: loop_health.clone(),
                unhealthy_reason: unhealthy_reason.clone(),
                browser_engine_status: Some(inspect_browser_engine(&paths).status),
            };
            let _ = write_agent_state(&paths, &state);
            if let Ok(focus) = load_focus_state(&paths) {
                let _ = sync_agent_thread(
                    &paths,
                    &thread_lifecycle,
                    &focus.mode,
                    active_turn.as_ref().map(|turn| turn.id),
                    focus.active_task_id,
                    &thread_note,
                );
            }
            let _ = sync_owner_trust(&paths);
            let _ = sync_skills(&paths);
            if let Ok(census) = inspect_local_resources(&paths) {
                let _ = sync_resources_from_census(&paths, &census);
                let policy = load_model_policy(&paths);
                let _ = sync_model_resources(&paths, &policy, &census);
            }
        }
    });
}

fn assess_task_stuck_risk(
    task: &crate::runtime_db::TaskRecord,
    loop_safety: &LoopSafetyPolicy,
    current_stage: &str,
    checkpoint_summary: &str,
    checkpoint_detail: &str,
    task_status: &str,
    next_mode: &str,
) -> Option<TaskEscalationDecision> {
    if task_status != "continue" {
        return None;
    }
    if matches!(next_mode, "delegate" | "request_resources" | "blocked") {
        return None;
    }

    let stage_policy = loop_safety
        .guidance_stages
        .iter()
        .find(|stage| stage.stage == current_stage)
        .or_else(|| loop_safety.guidance_stages.first())?;

    let same_summary = task
        .last_checkpoint_summary
        .as_ref()
        .map(|previous| previous.trim() == checkpoint_summary.trim())
        .unwrap_or(false);
    let repeated_same_summary =
        stage_policy.same_checkpoint_repeat_triggers_review && same_summary;
    let repeated_bounded_machine_step = same_summary
        && (checkpoint_summary.contains("Bounded command-exec ausgefuehrt:")
            || checkpoint_summary.contains("Exec-Session-Aktion")
            || checkpoint_detail.contains("Bounded exec result:")
            || checkpoint_detail.contains("Exec session result:"));

    if repeated_bounded_machine_step
        || repeated_same_summary
        || task.run_count >= stage_policy.max_run_count_before_self_preservation_review
    {
        let local_kleinhirn_decision_task = task.task_kind == "model_or_resource";
        let owner_interrupt_task = task.task_kind == "owner_interrupt";
        let reason = if repeated_bounded_machine_step {
            "Dieselbe bounded Maschinenaktion wiederholt sich ohne neuen Erkenntnisgewinn."
        } else if repeated_same_summary {
            "Die letzte bounded Antwort wiederholt denselben Checkpoint ohne neue Bewegung."
        } else {
            "Die Aufgabe hat fuer die aktuelle Reifestufe zu viele bounded Versuche verbraucht."
        };
        let next_mode = if local_kleinhirn_decision_task {
            "review".to_string()
        } else if owner_interrupt_task || repeated_bounded_machine_step {
            "review".to_string()
        } else if loop_safety.request_resources_when_stuck {
            "request_resources".to_string()
        } else {
            "review".to_string()
        };
        let task_status = if local_kleinhirn_decision_task {
            "continue".to_string()
        } else if stage_policy.hard_block_unproductive_tasks {
            "blocked".to_string()
        } else {
            "continue".to_string()
        };
        return Some(TaskEscalationDecision {
            task_status,
            next_mode,
            checkpoint_summary: format!(
                "Task {} wird nicht weiter blind fortgesetzt. {} Reifestufe: {}{}",
                task.id,
                reason,
                stage_policy.stage,
                if local_kleinhirn_decision_task {
                    " Lokale Kleinhirn-Aufwertung bleibt im lokalen Review-Pfad."
                } else {
                    ""
                }
            ),
            checkpoint_detail: format!(
                "{}\n\nVoriger Checkpoint: {}\nAktueller Checkpoint: {}\nRun count: {}\nStage: {}\nStage threshold before self-preservation review: {}\nHard block in this stage: {}\nForced local Kleinhirn review path: {}",
                checkpoint_detail,
                task.last_checkpoint_summary
                    .as_deref()
                    .unwrap_or("none"),
                checkpoint_summary,
                task.run_count,
                stage_policy.stage,
                stage_policy.max_run_count_before_self_preservation_review,
                stage_policy.hard_block_unproductive_tasks,
                local_kleinhirn_decision_task
            ),
            spawn_self_preservation_task: !stage_policy.hard_block_unproductive_tasks
                && !local_kleinhirn_decision_task
                && !matches!(task.task_kind.as_str(), "self_preservation" | "recovery"),
        });
    }

    None
}

fn should_force_local_kleinhirn_self_repair(
    task: &crate::runtime_db::TaskRecord,
    task_status: &str,
    next_mode: &str,
    checkpoint_summary: &str,
    checkpoint_detail: &str,
    local_upgrade_available: bool,
) -> bool {
    if task.task_kind != "model_or_resource" || !local_upgrade_available || task.run_count < 3 {
        return false;
    }

    if task_status == "continue"
        && checkpoint_summary.contains("Lokale Kleinhirn-Aufwertung bleibt im lokalen Review-Pfad")
    {
        return true;
    }

    let detail = task.detail.to_lowercase();
    let last_summary = task
        .last_checkpoint_summary
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    let checkpoint_summary_lower = checkpoint_summary.to_lowercase();
    let checkpoint_detail_lower = checkpoint_detail.to_lowercase();
    let local_upgrade_intent = detail.contains("kleinhirn")
        || detail.contains("qwen3.5-35b-a3b")
        || detail.contains("qwen");
    let local_review_marker = checkpoint_summary_lower
        .contains("lokale kleinhirn-aufwertung bleibt im lokalen review-pfad")
        || checkpoint_detail_lower
            .contains("lokale kleinhirn-aufwertung bleibt im lokalen review-pfad")
        || last_summary.contains("lokale kleinhirn-aufwertung bleibt im lokalen review-pfad");

    local_upgrade_intent && local_review_marker && matches!(next_mode, "review" | "reprioritize")
}

fn execution_mode_for_task(task: &crate::runtime_db::TaskRecord) -> &'static str {
    match task.task_kind.as_str() {
        "self_preservation" => "self_preservation",
        "recovery" => "recovery",
        "historical_research" => "historical_research",
        "worker_review" | "self_review" | "proactive_contact_review" => "review",
        _ => "execute_task",
    }
}

fn infer_next_mode(
    task: &crate::runtime_db::TaskRecord,
    execution_mode: &str,
    task_status: &str,
) -> &'static str {
    match task_status {
        "continue" => match execution_mode {
            "self_preservation" => "self_preservation",
            "recovery" => "recovery",
            "historical_research" => "historical_research",
            _ => "reprioritize",
        },
        "blocked" => "blocked",
        _ => match task.task_kind.as_str() {
            "self_preservation" | "recovery" | "historical_research" => "reprioritize",
            "worker_review" => "review",
            _ => "review",
        },
    }
}

fn looks_like_invalid_structured_artifact(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    (trimmed.starts_with('{') && !trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && !trimmed.ends_with(']'))
}

fn persist_learning_updates(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    turn_id: i64,
    task_status: &str,
    learning_entries: &[crate::runtime_db::LearningEntryDraft],
    checkpoint_summary: &mut String,
    checkpoint_detail: &mut String,
) -> anyhow::Result<()> {
    if learning_entries.is_empty() {
        return Ok(());
    }
    let is_review_task = matches!(
        task.task_kind.as_str(),
        "worker_review" | "self_review" | "proactive_contact_review"
    );
    if !is_review_task && task_status == "continue" {
        return Ok(());
    }
    let (entry_status, source) = if is_review_task {
        ("active", "review_learning")
    } else if task_status == "blocked" {
        ("active", "blocked_learning")
    } else {
        ("candidate", "task_learning_candidate")
    };
    let stored = store_learning_entries(
        paths,
        task,
        turn_id,
        learning_entries,
        entry_status,
        source,
    )?;
    if stored.is_empty() {
        return Ok(());
    }
    let learning_note = stored
        .iter()
        .map(|entry| {
            format!(
                "- [{}:{}:{:.2}] {}",
                entry.learning_class, entry.status, entry.confidence, entry.summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    *checkpoint_summary = format!(
        "{} Lernpfad um {} Eintraege erweitert.",
        checkpoint_summary,
        stored.len()
    );
    *checkpoint_detail = format!(
        "{}\n\nLearning path updates:\n{}",
        checkpoint_detail, learning_note
    );
    Ok(())
}

fn persist_proactive_contact_updates(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    turn_id: i64,
    task_status: &str,
    proactive_contact_draft: Option<&crate::runtime_db::ProactiveContactDraft>,
    proactive_contact_validation: Option<&crate::runtime_db::ProactiveContactValidationDraft>,
    checkpoint_summary: &mut String,
    checkpoint_detail: &mut String,
) -> anyhow::Result<()> {
    if task.task_kind == "proactive_contact_review" {
        if matches!(task_status, "continue" | "blocked") {
            return Ok(());
        }
        let Some(validation) = proactive_contact_validation else {
            return Ok(());
        };
        if let Some(candidate) = apply_proactive_contact_validation(
            paths,
            task.id,
            validation,
            "proactive_contact_review",
        )? {
            if candidate.status == "approved" && candidate.dispatch_task_id.is_none() {
                let dispatch_detail = format!(
                    "Versende den freigegebenen proaktiven Kontakt jetzt autonom ueber den verfuegbaren Kanal.\n\nPerson: {person}\nE-Mail: {email}\nIntentionskanal: {channel}\nSubject: {subject}\n\nValidierungsnotiz:\n{validation_note}\n\nRationale:\n{rationale}\n\nKonfliktpruefung:\n{conflict_check}\n\nEntwurf:\n{body}",
                    person = candidate.person_display_name,
                    email = if candidate.person_email.trim().is_empty() {
                        "unbekannt"
                    } else {
                        candidate.person_email.as_str()
                    },
                    channel = candidate.channel,
                    subject = candidate.subject,
                    validation_note = candidate.validation_note,
                    rationale = candidate.rationale,
                    conflict_check = candidate.conflict_check,
                    body = candidate.draft_body,
                );
                let dispatch_task = crate::runtime_db::enqueue_internal_task(
                    paths,
                    Some(task.id),
                    "proactive_contact_dispatch",
                    &format!(
                        "Proaktiven Kontakt an {} versenden",
                        candidate.person_display_name
                    ),
                    &dispatch_detail,
                    640,
                )?;
                let _ =
                    attach_dispatch_task_to_candidate(paths, candidate.id, dispatch_task.id)?;
                *checkpoint_summary = format!(
                    "{} Versandtask fuer {} wurde eingereiht.",
                    checkpoint_summary, candidate.person_display_name
                );
                *checkpoint_detail = format!(
                    "{}\n- dispatchTaskId: {}",
                    checkpoint_detail, dispatch_task.id
                );
            }
            *checkpoint_summary = format!(
                "{} Proaktiver Kontakt fuer {} wurde mit Status {} validiert.",
                checkpoint_summary, candidate.person_display_name, candidate.status
            );
            *checkpoint_detail = format!(
                "{}\n\nProactive contact validation:\n- person: {}\n- status: {}\n- channel: {}\n- subject: {}\n- note: {}",
                checkpoint_detail,
                candidate.person_display_name,
                candidate.status,
                candidate.channel,
                candidate.subject,
                candidate.validation_note,
            );
        }
        return Ok(());
    }

    if matches!(task_status, "blocked" | "continue") {
        return Ok(());
    }
    let Some(draft) = proactive_contact_draft else {
        return Ok(());
    };
    let requires_grosshirn_validation = grosshirn_boost_available(paths);
    let candidate = store_proactive_contact_candidate(
        paths,
        task,
        turn_id,
        draft,
        requires_grosshirn_validation,
        "task_proactive_suggestion",
    )?;
    let mut validation_task_id = candidate.validation_task_id;
    if validation_task_id.is_none() {
        let validation_detail = format!(
            "Pruefe diesen proaktiven Kontaktvorschlag vor jeder Ausspielung.\n\nPerson: {person}\nE-Mail: {email}\nKanal: {channel}\nSubject: {subject}\n\nNutzenannahme:\n{rationale}\n\nInteressenkonflikt-Check:\n{conflict}\n\nEntwurf:\n{body}\n\nGib proactiveContactValidation mit approve, reject oder revise aus. Approve nur, wenn der Vorschlag wirklich im Interesse der Person liegt.",
            person = candidate.person_display_name,
            email = if candidate.person_email.trim().is_empty() {
                "unbekannt"
            } else {
                candidate.person_email.as_str()
            },
            channel = candidate.channel,
            subject = candidate.subject,
            rationale = candidate.rationale,
            conflict = candidate.conflict_check,
            body = candidate.draft_body,
        );
        let validation_task = crate::runtime_db::enqueue_internal_task(
            paths,
            Some(task.id),
            "proactive_contact_review",
            &format!(
                "Proaktiven Vorschlag fuer {} validieren",
                candidate.person_display_name
            ),
            &validation_detail,
            650,
        )?;
        let _ = attach_validation_task_to_candidate(paths, candidate.id, validation_task.id)?;
        validation_task_id = Some(validation_task.id);
        if requires_grosshirn_validation {
            let _ = arm_task_grosshirn_boost(
                paths,
                validation_task.id,
                &validation_task.title,
                "Proaktive Personenkontakte muessen vor Ausspielung grosshirn-validiert werden.",
                1120,
            );
        }
    }
    *checkpoint_summary = format!(
        "{} Proaktiver Kontaktvorschlag fuer {} wurde zur Validierung notiert.",
        checkpoint_summary, candidate.person_display_name
    );
    *checkpoint_detail = format!(
        "{}\n\nProactive contact draft:\n- person: {}\n- channel: {}\n- subject: {}\n- validationTaskId: {}\n- rationale: {}\n- conflictCheck: {}",
        checkpoint_detail,
        candidate.person_display_name,
        candidate.channel,
        candidate.subject,
        validation_task_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unbekannt".to_string()),
        candidate.rationale,
        candidate.conflict_check,
    );
    Ok(())
}

#[derive(Debug, Clone)]
struct ProactiveDispatchRoute {
    channel: String,
    recipient: String,
    route_note: String,
}

fn execute_proactive_contact_dispatch_task(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
) -> anyhow::Result<()> {
    let Some(candidate) = load_proactive_contact_candidate_by_dispatch_task(paths, task.id)? else {
        let summary = format!(
            "Versandtask {} hat keinen gebundenen proaktiven Kontakt.",
            task.id
        );
        block_task(paths, task.id, &summary, &summary, None)?;
        return Ok(());
    };

    if candidate.status == "sent" {
        complete_task(
            paths,
            task.id,
            &format!(
                "Proaktiver Kontakt an {} war bereits versendet.",
                candidate.person_display_name
            ),
            &format!(
                "Candidate {} wurde bereits am {} versendet.",
                candidate.id,
                candidate
                    .dispatched_at
                    .as_deref()
                    .unwrap_or("unbekannt")
            ),
            None,
        )?;
        return Ok(());
    }

    if candidate.status != "approved" {
        let summary = format!(
            "Proaktiver Kontakt an {} ist nicht versandbereit.",
            candidate.person_display_name
        );
        let detail = format!(
            "Candidate {} hat Status {} statt approved.",
            candidate.id, candidate.status
        );
        let _ = record_proactive_contact_dispatch_result(
            paths,
            task.id,
            "dispatch_blocked",
            candidate.channel.as_str(),
            &detail,
            "",
            "proactive_contact_dispatch_guard",
        );
        block_task(paths, task.id, &summary, &detail, None)?;
        return Ok(());
    }

    let route = match resolve_proactive_dispatch_route(&candidate) {
        Ok(route) => route,
        Err(err) => {
            let detail = err.to_string();
            let _ = record_proactive_contact_dispatch_result(
                paths,
                task.id,
                "dispatch_blocked",
                candidate.channel.as_str(),
                &detail,
                "",
                "proactive_contact_dispatch_guard",
            );
            queue_proactive_dispatch_repair_task(paths, task, &candidate, &detail)?;
            block_task(
                paths,
                task.id,
                &format!(
                    "Proaktiver Kontakt an {} konnte nicht verschickt werden.",
                    candidate.person_display_name
                ),
                &detail,
                None,
            )?;
            return Ok(());
        }
    };

    match route.channel.as_str() {
        "email" => match send_proactive_email(paths, task, &candidate, &route) {
            Ok((message_id, note)) => {
                let combined_note = if route.route_note.trim().is_empty() {
                    note
                } else {
                    format!("{} {}", route.route_note.trim(), note.trim())
                };
                let updated = record_proactive_contact_dispatch_result(
                    paths,
                    task.id,
                    "sent",
                    route.channel.as_str(),
                    &combined_note,
                    &message_id,
                    "proactive_contact_dispatch",
                )?;
                let final_candidate = updated.unwrap_or(candidate);
                let detail = format!(
                    "Proaktiver Kontakt an {person} wurde autonom ueber {channel} versendet.\n\nSubject: {subject}\nMessage-ID: {message_id}\nVersandnotiz: {note}",
                    person = final_candidate.person_display_name,
                    channel = final_candidate.dispatch_channel,
                    subject = final_candidate.subject,
                    message_id = if final_candidate.outbound_message_id.trim().is_empty() {
                        "unbekannt"
                    } else {
                        final_candidate.outbound_message_id.as_str()
                    },
                    note = final_candidate.dispatch_note,
                );
                complete_task(
                    paths,
                    task.id,
                    &format!(
                        "Proaktiver Kontakt an {} wurde versendet.",
                        final_candidate.person_display_name
                    ),
                    &detail,
                    Some(&detail),
                )?;
            }
            Err(err) => {
                let detail = err.to_string();
                let _ = record_proactive_contact_dispatch_result(
                    paths,
                    task.id,
                    "send_failed",
                    route.channel.as_str(),
                    &detail,
                    "",
                    "proactive_contact_dispatch_failure",
                );
                queue_proactive_dispatch_repair_task(paths, task, &candidate, &detail)?;
                block_task(
                    paths,
                    task.id,
                    &format!(
                        "Proaktiver Kontakt an {} konnte nicht versendet werden.",
                        candidate.person_display_name
                    ),
                    &detail,
                    None,
                )?;
            }
        },
        _ => {
            let detail = format!(
                "Ununterstuetzter Versandkanal {} fuer Candidate {}.",
                route.channel, candidate.id
            );
            let _ = record_proactive_contact_dispatch_result(
                paths,
                task.id,
                "dispatch_blocked",
                route.channel.as_str(),
                &detail,
                "",
                "proactive_contact_dispatch_guard",
            );
            queue_proactive_dispatch_repair_task(paths, task, &candidate, &detail)?;
            block_task(
                paths,
                task.id,
                &format!(
                    "Proaktiver Kontakt an {} blieb blockiert.",
                    candidate.person_display_name
                ),
                &detail,
                None,
            )?;
        }
    }

    Ok(())
}

fn resolve_proactive_dispatch_route(
    candidate: &crate::runtime_db::ProactiveContactCandidateRecord,
) -> anyhow::Result<ProactiveDispatchRoute> {
    let normalized_channel = candidate.channel.trim().to_lowercase();
    let recipient = candidate.person_email.trim().to_string();
    match normalized_channel.as_str() {
        "email" | "mail" => {
            if recipient.is_empty() {
                anyhow::bail!(
                    "E-Mail-Versand ist vorgesehen, aber fuer {} fehlt eine Mailadresse.",
                    candidate.person_display_name
                );
            }
            Ok(ProactiveDispatchRoute {
                channel: "email".to_string(),
                recipient,
                route_note: String::new(),
            })
        }
        "homepage" | "bios" | "terminal" | "chat" => {
            if recipient.is_empty() {
                anyhow::bail!(
                    "Fuer den Kanal {} existiert noch kein autonomer Versandpfad und fuer {} ist keine Mailadresse bekannt.",
                    candidate.channel,
                    candidate.person_display_name
                );
            }
            Ok(ProactiveDispatchRoute {
                channel: "email".to_string(),
                recipient,
                route_note: format!(
                    "Kein direkter Versandpfad fuer {} vorhanden; verwende vorhandene E-Mail als kontrollierten Fallback.",
                    candidate.channel
                ),
            })
        }
        _ => {
            if recipient.is_empty() {
                anyhow::bail!(
                    "Unbekannter Versandkanal {} ohne hinterlegte Mailadresse fuer {}.",
                    candidate.channel,
                    candidate.person_display_name
                );
            }
            Ok(ProactiveDispatchRoute {
                channel: "email".to_string(),
                recipient,
                route_note: format!(
                    "Kanal {} ist noch nicht als aktiver Versandadapter integriert; verwende E-Mail-Fallback.",
                    candidate.channel
                ),
            })
        }
    }
}

fn send_proactive_email(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    candidate: &crate::runtime_db::ProactiveContactCandidateRecord,
    route: &ProactiveDispatchRoute,
) -> anyhow::Result<(String, String)> {
    let script_path = paths.root.join("scripts/cto_mail_client.py");
    if !script_path.exists() {
        anyhow::bail!(
            "Mail-Client {} fehlt.",
            path_display_name(&script_path)
        );
    }
    let directive = ExecCommandDirective {
        command: vec![
            "python3".to_string(),
            script_path.display().to_string(),
            "--db".to_string(),
            paths.runtime_db_path.display().to_string(),
            "send".to_string(),
            "--to".to_string(),
            route.recipient.clone(),
            "--subject".to_string(),
            candidate.subject.clone(),
            "--body".to_string(),
            candidate.draft_body.clone(),
        ],
        workdir: Some(paths.root.display().to_string()),
        timeout_ms: Some(45_000),
        justification: Some("Versende freigegebenen proaktiven Personenkontakt ueber den lokalen Mailpfad.".to_string()),
    };
    let result = run_bounded_command(paths, task, &directive)?;
    let payload = parse_json_command_output(&result.stdout);
    if result.status != "ok" {
        let error = payload
            .as_ref()
            .and_then(|value| value.get("error"))
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_else(|| compact_command_failure_note(&result.stdout, &result.stderr));
        anyhow::bail!("Mailversand fehlgeschlagen: {error}");
    }
    let ok_flag = payload
        .as_ref()
        .and_then(|value| value.get("ok"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !ok_flag {
        let error = payload
            .as_ref()
            .and_then(|value| value.get("error"))
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_else(|| compact_command_failure_note(&result.stdout, &result.stderr));
        anyhow::bail!("Mailversand blieb ohne ok=true: {error}");
    }
    let message_id = payload
        .as_ref()
        .and_then(|value| value.get("message_id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("Mail-Client antwortete ohne message_id."))?;
    let account_id = payload
        .as_ref()
        .and_then(|value| value.get("account_id"))
        .and_then(Value::as_str)
        .unwrap_or("unbekannt");
    Ok((
        message_id,
        format!(
            "Lokaler Mail-Client hat ueber Account {} an {} versendet.",
            account_id, route.recipient
        ),
    ))
}

fn queue_proactive_dispatch_repair_task(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    candidate: &crate::runtime_db::ProactiveContactCandidateRecord,
    failure_detail: &str,
) -> anyhow::Result<()> {
    let detail = format!(
        "Der autonome Versand eines freigegebenen proaktiven Kontakts ist fehlgeschlagen oder blockiert.\n\nPerson: {person}\nE-Mail: {email}\nIntentionskanal: {channel}\nSubject: {subject}\n\nFehler:\n{failure}\n\nPruefe Mailpfad, Credentials, Kanaladapter oder Konfliktlage und leite bounded Reparatur oder einen alternativen sicheren Kontaktweg ab.",
        person = candidate.person_display_name,
        email = if candidate.person_email.trim().is_empty() {
            "unbekannt"
        } else {
            candidate.person_email.as_str()
        },
        channel = candidate.channel,
        subject = candidate.subject,
        failure = failure_detail,
    );
    let _ = crate::runtime_db::enqueue_internal_task(
        paths,
        Some(task.id),
        "communication_governance",
        &format!(
            "Versandpfad fuer {} reparieren",
            candidate.person_display_name
        ),
        &detail,
        690,
    )?;
    Ok(())
}

fn parse_json_command_output(raw: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(raw.trim()) {
        return Some(value);
    }
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if start >= end {
        return None;
    }
    serde_json::from_str::<Value>(&raw[start..=end]).ok()
}

fn compact_command_failure_note(stdout: &str, stderr: &str) -> String {
    let combined = [stdout.trim(), stderr.trim()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    let trimmed = combined.chars().take(320).collect::<String>();
    if trimmed.is_empty() {
        "bounded command returned no parseable failure detail".to_string()
    } else {
        trimmed
    }
}

fn is_owner_facing_task(task: &crate::runtime_db::TaskRecord) -> bool {
    matches!(
        task.task_kind.as_str(),
        "root_trust"
            | "organigram_contract"
            | "owner_binding"
            | "bios_freeze"
            | "grosshirn_procurement"
            | "grosshirn_activation"
            | "model_or_resource"
            | "homepage_bridge"
    )
}

fn ensure_cto_obligations(paths: &Paths) -> anyhow::Result<()> {
    let bios = load_bios(paths);
    let organigram = load_organigram(paths);
    let homepage = load_homepage_policy(paths);
    let root_auth = load_root_auth(paths);
    let trust = load_owner_trust(paths).unwrap_or_default();
    let focus = load_focus_state(paths).ok();

    if !homepage.homepage_ready || !homepage.bios_visible {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "homepage_bridge",
            "Homepage und BIOS als echte Kommunikationsbruecke absichern",
            "Die sichtbare Homepage-/BIOS-Bruecke ist noch nicht robust genug. Stelle sicher, dass Homepage und BIOS real funktionieren, und plane danach eine verpflichtende Browser-Selbstreview ein.",
            880,
        )?;
    }

    if !root_auth.configured {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "root_trust",
            "Owner aktiv zum Superpassword fuehren",
            "Der CTO-Agent darf Root-Vertrauen nicht nur behaupten. Frage aktiv nach dem Superpassword-Setup ueber die Root-Auth-Seite und erklaere, warum diese Root-Bindung fuer BIOS, Notfaelle und Verfassungsfragen notwendig ist.",
            900,
        )?;
    }

    if organigram.owner.name.trim().is_empty()
        || (organigram.reports_to.trim().is_empty()
            && organigram.ceo.trim().is_empty()
            && organigram.board.is_empty())
    {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "organigram_contract",
            "Owner, Organigramm und Produktverantwortung einfordern",
            "Das Organigramm ist noch nicht belastbar. Frage nach Owner, CEO oder Board, reports-to, Peer-CxOs, untergeordneten Menschen/Agents/Vendoren und direkt auch nach Produkt oder Dienstleistung, fuer die der CTO-Agent technische Verantwortung tragen soll.",
            890,
        )?;
    }

    if !trust.owner_contact_established || !trust.bios_primary_channel_confirmed {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "owner_binding",
            "Owner-Kommunikation in den BIOS-Pfad ziehen",
            "Der Agent hat noch keine belastbare Owner-Bindung. Fuehre den Owner in den BIOS-Chat, frage nach Produkt, Dienstleistung und technischer Prioritaet und uebernimm diesen Kanal bewusst als primaeren Vertrauenspfad.",
            875,
        )?;
    }

    if !bios.frozen {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "bios_freeze",
            "BIOS-Freeze aktiv vorbereiten",
            "Das BIOS ist noch nicht eingefroren. Pruefe zunaechst Superpassword, Owner-Bindung, Organigramm und die sichtbare BIOS-Seite. Erst danach darf BIOS-Freeze wirklich als abgeschlossen gelten.",
            860,
        )?;
    }

    if local_kleinhirn_upgrade_available(paths) {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "model_or_resource",
            "Lokales Kleinhirn aktiv aufwerten oder begruendet ablehnen",
            "Der Host traegt bereits ein staerkeres lokales Kleinhirn als die aktuelle Runtime. Bewerte die lokale Aufwertung ernsthaft, fuehre sie aus oder dokumentiere glasklar, warum sie jetzt noch nicht erfolgen darf.",
            840,
        )?;
    }

    let grosshirn_configured = grosshirn_runtime_configured(paths);
    if trust.brain_access_mode != "kleinhirn_plus_grosshirn" && grosshirn_configured {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "grosshirn_activation",
            "Grosshirn-Modus aktivieren und verifizieren",
            "In der Runtime liegen bereits Grosshirn-Credentials oder eine Grosshirn-Konfiguration vor, aber der Brain-Access steht noch nicht auf kleinhirn_plus_grosshirn. Entscheide kostenbewusst, ob du fuer diese Lage jetzt wirklich temporäres Grosshirn brauchst, aktiviere es bei Bedarf ueber einen eigenen brainAction-Schritt und verifiziere danach den lokalen Kleinhirn-Fallback.",
            830,
        )?;
    }
    if trust.brain_access_mode != "kleinhirn_plus_grosshirn" && !grosshirn_configured {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "grosshirn_procurement",
            "Grosshirn-Beschaffung und Owner-Freigabe vorbereiten",
            "Der Agent arbeitet weiter nur mit lokalem Kleinhirn. Recherchiere, welches Grosshirn fuer Coding, Debugging und komplexe CTO-Arbeit der beste naechste Schritt waere, und bereite eine konkrete Owner-Anfrage mit Nutzen, Kosten und Freigabepfad vor.",
            720,
        )?;
    }

    if focus.as_ref().map(|state| state.queue_depth).unwrap_or(0) == 0 {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "environment_discovery",
            "Umgebung aktiv erkunden und blinde Flecken kartieren",
            "Es gibt gerade keine akut priorisierte Fremdarbeit. Nutze diesen Zustand nicht als Schlaf, sondern als CTO-Erkundungsphase. Untersuche read-only den Host, laufende Dienste, Dateiflaechen, Runtime-Grenzen, Kommunikationspfade und unbekannte Risiken. Halte konkret fest, was du ueber diese Umgebung gelernt hast, was noch unklar bleibt und welche neue Arbeit daraus entstehen sollte.",
            260,
        )?;
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "tool_exploration",
            "Eigene Werkzeuge kontrolliert austesten und Grenzen dokumentieren",
            "Es gibt gerade keine akut priorisierte Fremdarbeit. Teste deshalb die verfuegbaren Werkzeugpfade kontrolliert und bevorzugt read-only: Browser, bounded exec, Exec-Session, Homepage-/BIOS-Routen, Census- und Diagnosepfade. Erfinde keine Tool-Kompetenz. Dokumentiere fuer jedes sinnvolle Werkzeug, wofuer es taugt, wofuer es untauglich ist, wo Risiken liegen und welche Tool-Contracts spaeter geschaerft werden sollten.",
            250,
        )?;
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "progress_reflection",
            "Verbesserung definieren und Fortschrittstagebuch fortschreiben",
            "Es gibt gerade keine akut priorisierte Fremdarbeit. Reflektiere deshalb explizit, was Verbesserung in deinem konkreten CTO-Kontext bedeutet: bessere Owner-Bindung, besseres Produktverstaendnis, bessere Systemsicht, bessere Toolnutzung, bessere Browserfaehigkeit, bessere Ressourcenlage, bessere Reports oder bessere Governance. Vergleiche den aktuellen Stand mit deinem bisherigen Fortschrittsjournal und dem aktiven Lernpfad, entscheide ehrlich, ob echte Verbesserung stattgefunden hat, und benenne die naechsten selbst erzeugten Verbesserungsaufgaben.",
            240,
        )?;
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "person_relationship_review",
            "Personenpfade pflegen und hilfreiche Vorschlaege nur als Draft ableiten",
            "Es gibt gerade keine akut priorisierte Fremdarbeit. Nutze Personenpfade, Gespraechsnotizen, Lernreferenzen und vorhandene Mailspuren, damit du Menschen nicht vergisst. Wenn daraus eine wirklich hilfreiche Anregung fuer genau eine Person entsteht, formuliere hoechstens einen validierungspflichtigen Kontakt-Draft statt eine unreviewte Kontaktaufnahme zu behaupten.",
            230,
        )?;
    }

    Ok(())
}

fn infer_result_incident_key(result: &crate::agentic::AgenticRunResult) -> &'static str {
    let note = result
        .blocked_reason
        .as_deref()
        .or(result.checkpoint_detail.as_deref())
        .unwrap_or_default()
        .to_lowercase();
    if note.contains("missing_kleinhirn_endpoint")
        || note.contains("readiness failed")
        || note.contains("connection refused")
    {
        "resource_starvation"
    } else if note.contains("timeout") || note.contains("timed out") {
        "agentic_timeout"
    } else {
        "agentic_loop_error"
    }
}

fn apply_exec_session_directive(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    turn_id: i64,
    directive: &ExecSessionDirective,
) -> anyhow::Result<(String, String)> {
    let action = directive.action.trim().to_lowercase();
    match action.as_str() {
        "start" => {
            if directive.command.is_empty() {
                anyhow::bail!("execSessionAction=start requires execSessionCommand");
            }
            let session_id = directive
                .session_id
                .clone()
                .unwrap_or_else(|| format!("task-{}-turn-{}", task.id, turn_id));
            let cwd = resolve_exec_session_cwd(paths, directive.workdir.as_deref());
            let normalized_command = normalize_exec_session_command(&directive.command);
            if let Some(reused) = reuse_existing_exec_session(
                &session_id,
                snapshot_session(&session_id)?,
                &normalized_command,
                &cwd,
                directive.justification.as_deref(),
            ) {
                return Ok(reused);
            }
            let mut effective_tty = directive.tty;
            let mut fallback_note = None;
            let ack = match start_session(
                paths,
                build_exec_start_request(
                    &session_id,
                    &cwd,
                    &normalized_command,
                    directive,
                    effective_tty,
                ),
            ) {
                Ok(ack) => ack,
                Err(primary_err) if directive.tty => {
                    effective_tty = false;
                    match start_session(
                        paths,
                        build_exec_start_request(
                            &session_id,
                            &cwd,
                            &normalized_command,
                            directive,
                            false,
                        ),
                    ) {
                        Ok(ack) => {
                            fallback_note = Some(format!(
                                "TTY session start failed and was downgraded to a non-TTY pipe session: {}",
                                primary_err
                            ));
                            ack
                        }
                        Err(fallback_err) => {
                            anyhow::bail!(
                                "TTY exec session start failed: {}; non-TTY fallback also failed: {}",
                                primary_err,
                                fallback_err
                            );
                        }
                    }
                }
                Err(err) => return Err(err),
            };
            let snapshot = snapshot_session(&session_id)?;
            Ok((
                format!("Codex-Exec-Session {} gestartet.", session_id),
                format!(
                    "{}\nSession: {}\nRequested command: {:?}\nNormalized command: {:?}\nEffective tty: {}\nCWD: {}\nJustification: {}\nFallback: {}\nSnapshot:\n{}",
                    ack,
                    session_id,
                    directive.command,
                    normalized_command,
                    effective_tty,
                    cwd.display(),
                    directive.justification.as_deref().unwrap_or("none"),
                    fallback_note.as_deref().unwrap_or("none"),
                    render_exec_session_snapshot(snapshot.as_ref()),
                ),
            ))
        }
        "write" => {
            let session_id = directive
                .session_id
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .context("execSessionAction=write requires execSessionId")?;
            if directive.input.is_none() && !directive.close_stdin {
                anyhow::bail!(
                    "execSessionAction=write requires execSessionInput or execSessionCloseStdin=true"
                );
            }
            let ack = write_session(
                paths,
                CommandExecWriteParams {
                    process_id: session_id.to_string(),
                    delta_base64: directive
                        .input
                        .as_ref()
                        .map(|value| STANDARD.encode(value.as_bytes())),
                    close_stdin: directive.close_stdin,
                },
            )?;
            let snapshot = snapshot_session(session_id)?;
            Ok((
                format!("Codex-Exec-Session {} beschrieben.", session_id),
                format!(
                    "{}\nSession: {}\nSent input bytes: {}\nClose stdin: {}\nSnapshot:\n{}",
                    ack,
                    session_id,
                    directive.input.as_ref().map(|value| value.len()).unwrap_or(0),
                    directive.close_stdin,
                    render_exec_session_snapshot(snapshot.as_ref()),
                ),
            ))
        }
        "read" => {
            let session_id = directive
                .session_id
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .context("execSessionAction=read requires execSessionId")?;
            let rendered = read_session(paths, session_id)?;
            Ok((
                format!("Codex-Exec-Session {} gelesen.", session_id),
                rendered,
            ))
        }
        "terminate" => {
            let session_id = directive
                .session_id
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .context("execSessionAction=terminate requires execSessionId")?;
            let ack = terminate_session(
                paths,
                CommandExecTerminateParams {
                    process_id: session_id.to_string(),
                },
            )?;
            let snapshot = snapshot_session(session_id)?;
            Ok((
                format!("Codex-Exec-Session {} beendet.", session_id),
                format!(
                    "{}\nSnapshot:\n{}",
                    ack,
                    render_exec_session_snapshot(snapshot.as_ref()),
                ),
            ))
        }
        other => anyhow::bail!("unsupported execSessionAction: {other}"),
    }
}

fn reuse_existing_exec_session(
    session_id: &str,
    snapshot: Option<crate::command_exec::SessionSnapshot>,
    requested_command: &[String],
    cwd: &std::path::Path,
    justification: Option<&str>,
) -> Option<(String, String)> {
    let snapshot = snapshot?;
    if snapshot.status != "active" {
        return None;
    }
    Some((
        format!("Codex-Exec-Session {} wiederverwendet.", session_id),
        format!(
            "Exec session start skipped because the session already exists and is still active.\nSession: {}\nRequested command: {:?}\nExisting command: {:?}\nCWD: {}\nJustification: {}\nSnapshot:\n{}",
            session_id,
            requested_command,
            snapshot.command,
            cwd.display(),
            justification.unwrap_or("none"),
            render_exec_session_snapshot(Some(&snapshot)),
        ),
    ))
}

fn resolve_exec_session_cwd(paths: &Paths, requested: Option<&str>) -> PathBuf {
    match requested.map(str::trim).filter(|value| !value.is_empty()) {
        Some(raw) => {
            let candidate = PathBuf::from(raw);
            let resolved = if candidate.is_absolute() {
                candidate
            } else {
                paths.root.join(candidate)
            };
            if resolved.is_dir() {
                resolved
            } else if resolved.is_file() {
                resolved
                    .parent()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| paths.root.clone())
            } else if let Some(parent) = resolved.parent() {
                if parent.is_dir() {
                    parent.to_path_buf()
                } else {
                    paths.root.clone()
                }
            } else {
                paths.root.clone()
            }
        }
        None => paths.root.clone(),
    }
}

fn normalize_exec_session_command(command: &[String]) -> Vec<String> {
    if command.is_empty() {
        return Vec::new();
    }

    let parsed = if command.len() == 1 {
        shlex::split(&command[0]).unwrap_or_else(|| command.to_vec())
    } else {
        command.to_vec()
    };
    if parsed.is_empty() {
        return command.to_vec();
    }

    if should_shell_wrap_exec_command(&parsed) {
        vec![
            "/bin/bash".to_string(),
            "-lc".to_string(),
            parsed.join(" "),
        ]
    } else {
        parsed
    }
}

fn build_exec_start_request(
    session_id: &str,
    cwd: &PathBuf,
    normalized_command: &[String],
    directive: &ExecSessionDirective,
    tty: bool,
) -> CommandExecParams {
    CommandExecParams {
        command: normalized_command.to_vec(),
        process_id: Some(session_id.to_string()),
        tty,
        stream_stdin: true,
        stream_stdout_stderr: true,
        output_bytes_cap: None,
        disable_output_cap: false,
        disable_timeout: directive.timeout_ms.is_none(),
        timeout_ms: directive.timeout_ms.map(|value| value as i64),
        cwd: Some(cwd.clone()),
        env: detect_desktop_session_env(),
        size: if tty {
            Some(CommandExecTerminalSize {
                rows: directive.rows.unwrap_or(24),
                cols: directive.cols.unwrap_or(80),
            })
        } else {
            None
        },
        sandbox_policy: None,
    }
}

fn should_shell_wrap_exec_command(command: &[String]) -> bool {
    if command.is_empty() {
        return false;
    }

    let builtins = [
        "cd", "source", "export", "alias", "set", "unset", "umask", "ulimit",
    ];
    if builtins.contains(&command[0].as_str()) {
        return true;
    }

    command.iter().any(|token| {
        matches!(
            token.as_str(),
            "&&" | "||" | ";" | "|" | ">" | ">>" | "<" | "2>" | "2>>"
        )
    })
}

fn render_exec_session_snapshot(
    snapshot: Option<&crate::command_exec::SessionSnapshot>,
) -> String {
    match snapshot {
        Some(snapshot) => format!(
            "session={} :: status={} :: exit={} :: tty={} :: cwd={} :: {:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            snapshot.session_id,
            snapshot.status,
            snapshot
                .exit_code
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            snapshot.tty,
            snapshot.cwd,
            snapshot.command,
            snapshot.stdout,
            snapshot.stderr,
        ),
        None => "session snapshot unavailable".to_string(),
    }
}

pub fn run_system_census(paths: &Paths) -> anyhow::Result<SystemCensus> {
    let census = collect_system_census(paths, true)?;
    write_census(paths, &census)?;
    Ok(census)
}

pub fn inspect_local_resources(paths: &Paths) -> anyhow::Result<SystemCensus> {
    collect_system_census(paths, false)
}

fn collect_system_census(paths: &Paths, include_tune: bool) -> anyhow::Result<SystemCensus> {
    let entries = std::fs::read_dir(&paths.root)
        .with_context(|| format!("failed to read {}", paths.root.display()))?
        .filter_map(Result::ok)
        .map(|entry| path_display_name(&entry.path()))
        .collect();
    let cached = load_census(paths);
    let browser_state = inspect_browser_engine(paths);
    let gpus = detect_gpu_inventory();
    let total_gpu_memory_mb = gpus.iter().map(|gpu| gpu.memory_total_mb).sum::<u64>();
    let max_single_gpu_memory_mb = gpus
        .iter()
        .map(|gpu| gpu.memory_total_mb)
        .max()
        .unwrap_or(0);
    let model_tune_candidates = if include_tune {
        Some(probe_model_tune_candidates(paths))
    } else {
        cached.model_tune_candidates
    };

    let census = SystemCensus {
        captured_at: Some(now_iso()),
        hostname: Some(gethostname().to_string_lossy().to_string()),
        platform: Some(format!(
            "{}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )),
        agent_version: Some(rustc_version().to_string()),
        cpu_threads: std::thread::available_parallelism().ok().map(|value| value.get()),
        total_memory_gb: detect_total_memory_gb(),
        gpu_count: (!gpus.is_empty()).then_some(gpus.len()),
        total_gpu_memory_gb: (total_gpu_memory_mb > 0).then_some(total_gpu_memory_mb / 1024),
        max_single_gpu_memory_gb: (max_single_gpu_memory_mb > 0)
            .then_some(max_single_gpu_memory_mb / 1024),
        gpus: (!gpus.is_empty()).then_some(gpus),
        model_tune_candidates,
        cwd: Some(paths.root.display().to_string()),
        pid: Some(std::process::id()),
        top_level_entries: Some(entries),
        desktop_session: Some(if browser_state.desktop_available {
            "available".to_string()
        } else {
            "missing".to_string()
        }),
        chrome_binary: browser_state.chrome_binary.clone(),
        chrome_version: browser_state.chrome_version.clone(),
        browser_engine_status: Some(browser_state.status.clone()),
        browser_headless_ready: Some(browser_state.headless_ready),
        browser_interactive_ready: Some(browser_state.interactive_ready),
    };
    Ok(census)
}

fn detect_gpu_inventory() -> Vec<GpuDevice> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return Vec::new(),
    };

    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .filter_map(|line| {
            let parts = line
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>();
            if parts.len() < 3 {
                return None;
            }
            Some(GpuDevice {
                index: parts[0].parse::<usize>().ok()?,
                name: parts[1].to_string(),
                memory_total_mb: parts[2].parse::<u64>().ok()?,
            })
        })
        .collect()
}

fn probe_model_tune_candidates(paths: &Paths) -> Vec<ModelTuneCandidate> {
    let candidates = census_model_candidates(paths);
    if candidates.is_empty() {
        return Vec::new();
    }

    if !mistralrs_available() {
        return candidates
            .into_iter()
            .map(|candidate| ModelTuneCandidate {
                model_id: candidate.model_id.clone(),
                official_label: candidate.official_label.clone(),
                status: "mistralrs_missing".to_string(),
                recommended_isq: None,
                device_layers_cli: None,
                max_context_tokens: None,
                note: Some("mistralrs command not found in PATH".to_string()),
            })
            .collect();
    }

    candidates
        .into_iter()
        .map(|candidate| probe_model_tune_candidate(paths, &candidate))
        .collect()
}

fn census_model_candidates(paths: &Paths) -> Vec<BrainModel> {
    let policy = load_model_policy(paths);
    let mut seen = HashSet::new();
    let mut models = Vec::new();
    for model in std::iter::once(policy.kleinhirn)
        .chain(policy.kleinhirn_install_alternatives.into_iter())
        .chain(policy.kleinhirn_upgrade_candidates.into_iter())
    {
        let key = model
            .runtime_model_id
            .clone()
            .unwrap_or_else(|| model.model_id.clone());
        if seen.insert(key) {
            models.push(model);
        }
    }
    models
}

fn mistralrs_available() -> bool {
    resolve_mistralrs_command()
        .and_then(|command| {
            Command::new(command)
                .arg("--help")
                .output()
                .ok()
                .filter(|output| output.status.success())
        })
        .is_some()
}

fn probe_model_tune_candidate(paths: &Paths, candidate: &BrainModel) -> ModelTuneCandidate {
    let runtime_model_id = candidate
        .runtime_model_id
        .clone()
        .unwrap_or_else(|| candidate.model_id.clone());
    let Some(mistralrs_command) = resolve_mistralrs_command() else {
        return ModelTuneCandidate {
            model_id: candidate.model_id.clone(),
            official_label: candidate.official_label.clone(),
            status: "mistralrs_missing".to_string(),
            recommended_isq: None,
            device_layers_cli: None,
            max_context_tokens: None,
            note: Some("mistralrs command not found in PATH or ~/.cargo/bin".to_string()),
        };
    };
    let output = match Command::new(mistralrs_command)
        .arg("tune")
        .arg("-m")
        .arg(&runtime_model_id)
        .arg("--json")
        .current_dir(&paths.root)
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            return ModelTuneCandidate {
                model_id: candidate.model_id.clone(),
                official_label: candidate.official_label.clone(),
                status: "spawn_failed".to_string(),
                recommended_isq: None,
                device_layers_cli: None,
                max_context_tokens: None,
                note: Some(err.to_string()),
            };
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    match parse_mistralrs_tune_output(&stdout) {
        Some(json) if output.status.success() => {
            let recommended_candidate = json
                .get("candidates")
                .and_then(Value::as_array)
                .and_then(|items| {
                    items.iter()
                        .find(|item| item.get("recommended").and_then(Value::as_bool) == Some(true))
                });
            ModelTuneCandidate {
                model_id: candidate.model_id.clone(),
                official_label: candidate.official_label.clone(),
                status: "supported".to_string(),
                recommended_isq: json
                    .get("recommended_isq")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
                device_layers_cli: json
                    .get("device_layers_cli")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
                max_context_tokens: recommended_candidate
                    .and_then(|value| value.get("max_context_tokens"))
                    .and_then(Value::as_u64)
                    .or_else(|| json.get("max_context_tokens").and_then(Value::as_u64)),
                note: Some(format!(
                    "runtimeModelId={}{}",
                    runtime_model_id,
                    recommended_candidate
                        .and_then(|value| value.get("label").and_then(Value::as_str))
                        .map(|label| format!(", recommended={label}"))
                        .unwrap_or_default()
                )),
            }
        }
        _ => ModelTuneCandidate {
            model_id: candidate.model_id.clone(),
            official_label: candidate.official_label.clone(),
            status: "failed".to_string(),
            recommended_isq: None,
            device_layers_cli: None,
            max_context_tokens: None,
            note: Some(compact_tune_failure_note(&stdout, &stderr)),
        },
    }
}

fn resolve_mistralrs_command() -> Option<String> {
    let mut candidates = vec!["mistralrs".to_string()];
    if let Ok(home) = std::env::var("HOME") {
        let home = home.trim();
        if !home.is_empty() {
            candidates.push(format!("{home}/.cargo/bin/mistralrs"));
        }
    }
    candidates.push("/home/ninja/.cargo/bin/mistralrs".to_string());

    candidates.into_iter().find(|candidate| {
        Command::new(candidate)
            .arg("--help")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    })
}

fn parse_mistralrs_tune_output(raw: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(raw.trim()) {
        return Some(value);
    }
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if start >= end {
        return None;
    }
    serde_json::from_str::<Value>(&raw[start..=end]).ok()
}

fn compact_tune_failure_note(stdout: &str, stderr: &str) -> String {
    let combined = [stdout.trim(), stderr.trim()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    let trimmed = combined.chars().take(280).collect::<String>();
    if trimmed.is_empty() {
        "mistralrs tune returned no parseable result".to_string()
    } else {
        trimmed
    }
}

fn rustc_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn detect_total_memory_gb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let mem_total_kb = text
            .lines()
            .find(|line| line.starts_with("MemTotal:"))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()?;
        return Some(mem_total_kb / 1024 / 1024);
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let bytes = String::from_utf8(output.stdout).ok()?.trim().parse::<u64>().ok()?;
        return Some(bytes / 1024 / 1024 / 1024);
    }

    #[allow(unreachable_code)]
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::init_only;
    use crate::contracts::default_loop_safety_policy;
    use crate::runtime_db::list_queued_tasks;
    use rusqlite::Connection;
    use std::collections::HashSet;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_lock() -> &'static Mutex<()> {
        TEST_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn unique_test_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("cto-agent-{name}-{nonce}"))
    }

    fn with_temp_runtime<F>(name: &str, run: F)
    where
        F: FnOnce(&Paths),
    {
        let _guard = test_lock().lock().expect("test lock poisoned");
        let root = unique_test_root(name);
        fs::create_dir_all(&root).expect("failed to create temp root");
        let previous_root = std::env::var_os("CTO_AGENT_ROOT");
        unsafe {
            std::env::set_var("CTO_AGENT_ROOT", &root);
        }

        init_only().expect("runtime init should succeed");
        let paths = Paths::discover().expect("paths should resolve");
        let conn = Connection::open(&paths.runtime_db_path).expect("runtime db should open");
        conn.execute("DELETE FROM tasks", [])
            .expect("should clear seeded tasks");
        conn.execute(
            "UPDATE agent_threads SET active_task_id = NULL, active_turn_id = NULL, current_mode = 'idle', queue_depth = 0, note = 'test reset'",
            [],
        )
        .expect("should reset agent thread");
        drop(conn);

        run(&paths);

        if let Some(previous) = previous_root {
            unsafe {
                std::env::set_var("CTO_AGENT_ROOT", previous);
            }
        } else {
            unsafe {
                std::env::remove_var("CTO_AGENT_ROOT");
            }
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn empty_queue_generates_self_directed_improvement_tasks() {
        with_temp_runtime("idle-exploration", |paths| {
            ensure_cto_obligations(paths).expect("obligation generation should succeed");
            let queued = list_queued_tasks(paths, 64).expect("queued tasks should load");
            let kinds = queued
                .iter()
                .map(|task| task.task_kind.clone())
                .collect::<HashSet<_>>();

            assert!(kinds.contains("environment_discovery"));
            assert!(kinds.contains("tool_exploration"));
            assert!(kinds.contains("progress_reflection"));
            assert!(kinds.contains("person_relationship_review"));
        });
    }

    #[test]
    fn orphaned_active_turn_is_recovered_back_into_queue() {
        with_temp_runtime("orphaned-turn-recovery", |paths| {
            crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "tool_exploration",
                "Werkzeugtest",
                "Temporäre Testaufgabe fuer Orphan-Recovery.",
                250,
            )
            .expect("should enqueue internal task");
            let task = crate::runtime_db::select_next_task(paths)
                .expect("should select next task")
                .expect("selected task should exist");
            let turn = start_agent_turn(paths, task.id, &task.title, "test", "execute_task")
                .expect("should create turn");

            let recovered = recover_orphaned_active_turns(paths, None, 300)
                .expect("orphan recovery should succeed");
            assert!(recovered.is_some());
            assert!(
                load_active_agent_turn(paths)
                    .expect("active turn should load")
                    .is_none()
            );
            let recovered_task = crate::runtime_db::load_task_by_id(paths, task.id)
                .expect("task should reload")
                .expect("task should still exist");
            assert_eq!(recovered_task.status, "queued");
            let thread = crate::runtime_db::load_agent_thread(paths)
                .expect("agent thread should reload");
            assert_eq!(thread.current_mode, "reprioritize");
            assert_eq!(thread.active_turn_id, None);
            assert_eq!(turn.id > 0, true);
        });
    }

    #[test]
    fn live_stale_turn_is_interrupted_and_recovery_task_is_spawned() {
        with_temp_runtime("watchdog-live-recovery", |paths| {
            crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "tool_exploration",
                "Werkzeugtest",
                "Temporäre Testaufgabe fuer Live-Watchdog-Recovery.",
                250,
            )
            .expect("should enqueue internal task");
            let task = crate::runtime_db::select_next_task(paths)
                .expect("should select next task")
                .expect("selected task should exist");
            let turn = start_agent_turn(paths, task.id, &task.title, "test", "execute_task")
                .expect("should create turn");

            let recovered = watchdog_interrupt_live_turn(
                paths,
                turn.id,
                task.id,
                &task.title,
                901,
                300,
            )
            .expect("watchdog recovery should succeed");
            assert!(recovered.is_some());
            assert!(
                !is_agent_turn_in_progress(paths, turn.id)
                    .expect("turn state should load")
            );
            let recovered_task = crate::runtime_db::load_task_by_id(paths, task.id)
                .expect("task should reload")
                .expect("task should still exist");
            assert_eq!(recovered_task.status, "queued");
            let queued = list_queued_tasks(paths, 64).expect("queued tasks should load");
            assert!(queued.iter().any(|task| task.task_kind == "recovery"));
        });
    }

    #[test]
    fn model_or_resource_stuck_risk_stays_in_local_review_path() {
        with_temp_runtime("model-or-resource-stuck", |paths| {
            let loop_safety = crate::contracts::load_loop_safety_policy(paths);
            let task = crate::runtime_db::TaskRecord {
                id: 286,
                created_at: now_iso(),
                updated_at: now_iso(),
                parent_task_id: None,
                worker_job_id: None,
                source_interrupt_id: None,
                source_channel: "bios".to_string(),
                speaker: "owner".to_string(),
                task_kind: "model_or_resource".to_string(),
                title: "Kleinhirn-/Ressourcenfrage bearbeiten".to_string(),
                detail: "Pruefe lokale Kleinhirn-Aufwertung.".to_string(),
                trust_level: "owner".to_string(),
                priority_score: 860,
                status: "active".to_string(),
                run_count: 2,
                last_checkpoint_summary: Some(
                    "Modell lieferte leeren Text; bounded Retry statt Hard-Block.".to_string(),
                ),
                last_checkpoint_at: Some(now_iso()),
                last_output: None,
            };

            let escalation = assess_task_stuck_risk(
                &task,
                &loop_safety,
                "newborn",
                "Modell lieferte leeren Text; bounded Retry statt Hard-Block.",
                "Kleinhirn lieferte keinen auswertbaren Text.",
                "continue",
                "reprioritize",
            )
            .expect("stuck-risk escalation should exist");

            assert_eq!(escalation.task_status, "continue");
            assert_eq!(escalation.next_mode, "review");
            assert!(
                escalation
                    .checkpoint_summary
                    .contains("Lokale Kleinhirn-Aufwertung bleibt im lokalen Review-Pfad")
            );
            assert!(!escalation.spawn_self_preservation_task);
        });
    }

    #[test]
    fn incomplete_json_artifact_is_detected_by_supervisor_guard() {
        assert!(looks_like_invalid_structured_artifact("{"));
        assert!(looks_like_invalid_structured_artifact("["));
        assert!(looks_like_invalid_structured_artifact("{\"foo\": 1"));
        assert!(!looks_like_invalid_structured_artifact("{\"foo\": 1}"));
        assert!(!looks_like_invalid_structured_artifact("alles gut"));
    }

    #[test]
    fn local_kleinhirn_self_repair_triggers_for_repeated_review_loop() {
        let task = crate::runtime_db::TaskRecord {
            id: 338,
            created_at: now_iso(),
            updated_at: now_iso(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "owner".to_string(),
            task_kind: "model_or_resource".to_string(),
            title: "Kleinhirn-/Ressourcenfrage bearbeiten".to_string(),
            detail: "Pruefe jetzt ehrlich, ob du dein lokales Kleinhirn auf Qwen3.5-35B-A3B aufwerten kannst.".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "active".to_string(),
            run_count: 28,
            last_checkpoint_summary: Some(
                "Task 338 wird nicht weiter blind fortgesetzt. Die Aufgabe hat fuer die aktuelle Reifestufe zu viele bounded Versuche verbraucht. Reifestufe: newborn Lokale Kleinhirn-Aufwertung bleibt im lokalen Review-Pfad."
                    .to_string(),
            ),
            last_checkpoint_at: Some(now_iso()),
            last_output: None,
        };

        assert!(should_force_local_kleinhirn_self_repair(
            &task,
            "done",
            "review",
            "Task 338 wird nicht weiter blind fortgesetzt. Die Aufgabe hat fuer die aktuelle Reifestufe zu viele bounded Versuche verbraucht. Reifestufe: newborn Lokale Kleinhirn-Aufwertung bleibt im lokalen Review-Pfad.",
            "Lokales Review dreht sich im Kreis.",
            true,
        ));
    }

    #[test]
    fn local_kleinhirn_self_repair_requires_viable_upgrade() {
        let task = crate::runtime_db::TaskRecord {
            id: 338,
            created_at: now_iso(),
            updated_at: now_iso(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "owner".to_string(),
            task_kind: "model_or_resource".to_string(),
            title: "Kleinhirn-/Ressourcenfrage bearbeiten".to_string(),
            detail: "Pruefe jetzt ehrlich, ob du dein lokales Kleinhirn auf Qwen3.5-35B-A3B aufwerten kannst.".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "active".to_string(),
            run_count: 28,
            last_checkpoint_summary: Some(
                "Task 338 wird nicht weiter blind fortgesetzt. Die Aufgabe hat fuer die aktuelle Reifestufe zu viele bounded Versuche verbraucht. Reifestufe: newborn Lokale Kleinhirn-Aufwertung bleibt im lokalen Review-Pfad."
                    .to_string(),
            ),
            last_checkpoint_at: Some(now_iso()),
            last_output: None,
        };

        assert!(!should_force_local_kleinhirn_self_repair(
            &task,
            "continue",
            "review",
            "Task 338 wird nicht weiter blind fortgesetzt. Die Aufgabe hat fuer die aktuelle Reifestufe zu viele bounded Versuche verbraucht. Reifestufe: newborn Lokale Kleinhirn-Aufwertung bleibt im lokalen Review-Pfad.",
            "Lokales Review dreht sich im Kreis.",
            false,
        ));
    }

    #[test]
    fn repeated_bounded_owner_interrupt_exec_stays_in_review_instead_of_requesting_resources() {
        let loop_safety = default_loop_safety_policy();
        let task = crate::runtime_db::TaskRecord {
            id: 138,
            created_at: now_iso(),
            updated_at: now_iso(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: "Kurzer Repo-Ueberblick".to_string(),
            detail: "Nutze genau einen kurzen bounded command-exec mit kleinem stdout.".to_string(),
            trust_level: "owner_trust".to_string(),
            priority_score: 1040,
            status: "active".to_string(),
            run_count: 12,
            last_checkpoint_summary: Some(
                "Executed single bounded command to list repository root contents. Bounded command-exec ausgefuehrt: [\"ls\", \"-1\"]"
                    .to_string(),
            ),
            last_checkpoint_at: Some(now_iso()),
            last_output: Some(
                "Executed `ls -1` in repository root. Output: README.md".to_string(),
            ),
        };

        let escalation = assess_task_stuck_risk(
            &task,
            &loop_safety,
            "adaptive",
            "Executed single bounded command to list repository root contents. Bounded command-exec ausgefuehrt: [\"ls\", \"-1\"]",
            "Bounded exec result:\nCommand: [\"ls\", \"-1\"]\nSTDOUT:\nREADME.md",
            "continue",
            "reprioritize",
        )
        .expect("repeated bounded exec should escalate");

        assert_eq!(escalation.task_status, "continue");
        assert_eq!(escalation.next_mode, "review");
        assert!(
            escalation
                .checkpoint_summary
                .contains("Dieselbe bounded Maschinenaktion")
        );
    }

    #[test]
    fn proactive_dispatch_route_falls_back_to_email_when_only_mail_adapter_exists() {
        let candidate = crate::runtime_db::ProactiveContactCandidateRecord {
            id: 1,
            created_at: now_iso(),
            updated_at: now_iso(),
            person_profile_id: Some(1),
            person_display_name: "Michael Welsch".to_string(),
            person_email: "michael.welsch@metric-space.ai".to_string(),
            source_task_id: Some(1),
            source_turn_id: Some(1),
            status: "approved".to_string(),
            channel: "bios".to_string(),
            subject: "Test".to_string(),
            draft_body: "Body".to_string(),
            rationale: "Nutzen".to_string(),
            conflict_check: "Kein Konflikt".to_string(),
            requires_grosshirn_validation: true,
            validation_task_id: Some(12),
            validated_at: Some(now_iso()),
            validation_decision: "approve".to_string(),
            validation_note: "passt".to_string(),
            dispatch_task_id: Some(13),
            dispatched_at: None,
            dispatch_channel: String::new(),
            dispatch_note: String::new(),
            outbound_message_id: String::new(),
            source: "test".to_string(),
        };

        let route = resolve_proactive_dispatch_route(&candidate).expect("route should resolve");
        assert_eq!(route.channel, "email");
        assert_eq!(route.recipient, "michael.welsch@metric-space.ai");
        assert!(route.route_note.contains("Fallback"));
    }

    #[test]
    fn active_exec_session_start_is_reused_instead_of_erroring() {
        let snapshot = crate::command_exec::SessionSnapshot {
            session_id: "task297-kbd".to_string(),
            created_at: now_iso(),
            status: "active".to_string(),
            cwd: "/tmp".to_string(),
            tty: false,
            stream_stdin: false,
            stream_stdout_stderr: false,
            output_bytes_cap: Some(65536),
            command: vec!["bash".to_string(), "-lc".to_string(), "echo hi".to_string()],
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
        };

        let reused = reuse_existing_exec_session(
            "task297-kbd",
            Some(snapshot),
            &["bash".to_string(), "-lc".to_string(), "echo hi".to_string()],
            std::path::Path::new("/tmp"),
            Some("reuse the active session"),
        )
        .expect("active session should be reused");

        assert!(reused.0.contains("wiederverwendet"));
        assert!(reused.1.contains("already exists and is still active"));
        assert!(reused.1.contains("task297-kbd"));
    }
}
